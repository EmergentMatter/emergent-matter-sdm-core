"""``Param.expr`` evaluation — bindings, gradients, refresh, and GLSL expansion.

Wire/schema/cycle validation lives in ``test_param_expr_wire.py``. This file
pins the behaviour that makes geometry track definitions: traced resolution
through ``ParamBinding``, cache refresh on ``to_dict`` /
``update_from_vector``, and GLSL expansion of relations over base uniforms.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    sdf_primitive,
)
from software_defined_matter.dsl.expr import (
    eval_expr_pure,
    expr_binop,
    expr_num,
    expr_param,
    expr_param_names,
    expr_reduce,
    expr_unop,
)
from software_defined_matter.dsl.resolve import make_binding
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.glsl.emit import _GLSLEmitter
from software_defined_matter.sdf.param_refs import ParamRelationError


def _part(params, r=None, **metadata) -> Part:
    """A one-sphere part whose radius slot is ``r`` (default: a plain 1.0)."""
    tree = sdf_primitive("sphere", r=1.0 if r is None else r)
    return Part(
        name="p",
        params={p.name: p for p in params},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
        metadata=metadata or {"bbox": [[-50, -50, -50], [50, 50, 50]]},
    )


def _bare_emitter(part: Part) -> _GLSLEmitter:
    return _GLSLEmitter(part, smooth_csg=False, smooth_k=0.0)


def test_expr_param_names_sees_both_spellings():
    tree = expr_binop("+", expr_param("a"), {"$ref": "b"})
    assert expr_param_names(tree) == {"a", "b"}


def test_value_is_always_emitted_as_the_cached_evaluation():
    """A reader that drops the unknown ``expr`` key must still get the number."""
    part = _part(
        [
            Param("width", 10.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("half", unit="mm", expr=expr_binop("/", expr_param("width"), expr_num(2.0))),
        ]
    )
    assert part.to_dict()["params"]["half"]["value"] == pytest.approx(5.0)

    part.params["width"].value = 30.0
    doc = part.to_dict()
    assert doc["params"]["half"]["value"] == pytest.approx(15.0)

    stripped = {k: v for k, v in doc["params"]["half"].items() if k != "expr"}
    assert stripped["value"] == pytest.approx(15.0)


def test_the_optimiser_vector_never_contains_a_derived_param():
    part = _part(
        [
            Param("width", 10.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("depth", 4.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("half", unit="mm", expr=expr_binop("/", expr_param("width"), expr_num(2.0))),
        ]
    )
    assert part.free_param_names() == ["width", "depth"]
    assert part.param_vector() == [10.0, 4.0]

    part.update_from_vector([20.0, 4.0])
    assert part.params["half"].value == pytest.approx(10.0)


def test_binding_resolves_a_relation():
    part = _part(
        [
            Param("width", 10.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("half", unit="mm", expr=expr_binop("/", expr_param("width"), expr_num(2.0))),
        ]
    )
    binding = make_binding(part)
    assert "half" not in binding.fixed
    got = binding.get("half", jnp.asarray([10.0]))
    assert float(got) == pytest.approx(5.0)


def test_grad_flows_through_the_relation_to_the_base_param():
    """A relation resolved as a cached constant would give gradient zero here."""
    part = _part(
        [
            Param("width", 10.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("half", unit="mm", expr=expr_binop("/", expr_param("width"), expr_num(2.0))),
        ]
    )
    binding = make_binding(part)

    def f(free_vec):
        return binding.get("half", free_vec)

    g = jax.grad(lambda v: f(v).sum())(jnp.asarray([10.0]))
    assert float(g[0]) == pytest.approx(0.5, abs=1e-6)


def test_grad_flows_through_a_chain_of_relations():
    part = _part(
        [
            Param("width", 10.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("half", unit="mm", expr=expr_binop("/", expr_param("width"), expr_num(2.0))),
            Param("quarter", unit="mm", expr=expr_binop("/", expr_param("half"), expr_num(2.0))),
        ]
    )
    binding = make_binding(part)
    g = jax.grad(lambda v: binding.get("quarter", v).sum())(jnp.asarray([10.0]))
    assert float(g[0]) == pytest.approx(0.25, abs=1e-6)


def test_a_relation_in_an_sdf_slot_is_differentiable_end_to_end():
    """d(sdf)/d(width) at the origin for a sphere of radius width/2 is -0.5."""
    from software_defined_matter.sdf.compile import make_sdf_closure

    part = _part(
        [
            Param("width", 10.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("half", unit="mm", expr=expr_binop("/", expr_param("width"), expr_num(2.0))),
        ],
        r={"$ref": "half"},
    )
    closure = make_sdf_closure(part.computed_envelope(), part)
    p = jnp.zeros((1, 3))

    val = closure(p, jnp.asarray([10.0]))
    assert float(val[0]) == pytest.approx(-5.0, abs=1e-5)

    g = jax.grad(lambda v: closure(p, v).sum())(jnp.asarray([10.0]))
    assert float(g[0]) == pytest.approx(-0.5, abs=1e-5)


def test_a_diamond_dag_resolves_once_per_branch_and_agrees():
    #     width
    #     /   \
    #   a       b        c = a + b = 1.5 * width
    #     \   /
    #       c
    part = _part(
        [
            Param("width", 8.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("a", unit="mm", expr=expr_binop("*", expr_param("width"), expr_num(0.5))),
            Param("b", unit="mm", expr=expr_binop("*", expr_param("width"), expr_num(1.0))),
            Param("c", unit="mm", expr=expr_binop("+", expr_param("a"), expr_param("b"))),
        ]
    )
    binding = make_binding(part)
    assert float(binding.get("c", jnp.asarray([8.0]))) == pytest.approx(12.0)
    g = jax.grad(lambda v: binding.get("c", v).sum())(jnp.asarray([8.0]))
    assert float(g[0]) == pytest.approx(1.5, abs=1e-6)


def test_a_cycle_built_in_memory_is_caught_when_a_binding_is_made():
    """A Part assembled in Python never went through a loader."""
    part = _part(
        [
            Param("a", 1.0, unit="mm"),
            Param("b", 1.0, unit="mm"),
        ]
    )
    part.params["a"].expr = expr_param("b")
    part.params["b"].expr = expr_param("a")
    with pytest.raises(ParamRelationError, match="cycle"):
        make_binding(part)


def test_a_derived_param_expands_instead_of_binding_a_uniform():
    part = _part(
        [
            Param("outer", 20.0, free=True, bounds=(10.0, 60.0), unit="mm"),
            Param("wall", 2.0, free=True, bounds=(0.8, 6.0), unit="mm"),
            Param("bore", unit="mm", expr=expr_binop("-", expr_param("outer"), expr_param("wall"))),
        ],
        r={"$ref": "bore"},
    )
    em = emit_glsl(part)
    assert "(u_p_outer - u_p_wall)" in em.scene_source
    assert sorted(u.source_param for u in em.uniforms) == ["outer", "wall"]
    assert "u_p_bore" not in em.scene_source


def test_expansion_is_recursive_through_a_chain():
    part = _part(
        [
            Param("w", 8.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("half", unit="mm", expr=expr_binop("/", expr_param("w"), expr_num(2.0))),
            Param("quarter", unit="mm", expr=expr_binop("/", expr_param("half"), expr_num(2.0))),
        ],
        r={"$ref": "quarter"},
    )
    em = emit_glsl(part)
    assert "(u_p_w / 2.0)" in em.scene_source
    assert [u.source_param for u in em.uniforms] == ["w"]


def test_add_uniform_refuses_a_derived_name():
    part = _part(
        [
            Param("w", 8.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("half", unit="mm", expr=expr_binop("/", expr_param("w"), expr_num(2.0))),
        ]
    )
    e = _bare_emitter(part)
    with pytest.raises(ValueError, match="cannot bind a uniform"):
        e._add_uniform("half")


def test_controls_class_a_relation_as_derived_never_re_emit():
    part = _part(
        [
            Param("outer", 20.0, free=True, bounds=(10.0, 60.0), unit="mm"),
            Param("wall", 2.0, free=True, bounds=(0.8, 6.0), unit="mm"),
            Param("bore", unit="mm", expr=expr_binop("-", expr_param("outer"), expr_param("wall"))),
        ],
        r={"$ref": "bore"},
    )
    controls = {c["param"]: c for c in emit_glsl(part).controls}
    assert controls["outer"]["class"] == "live"
    bore = controls["bore"]
    assert bore["class"] == "derived"
    assert bore["sources"] == ["outer", "wall"]
    assert bore["expr"] == part.params["bore"].expr
    assert bore["value"] == pytest.approx(18.0)
    assert "uniform" not in bore


def test_a_derived_param_no_geometry_touches_is_still_derived_not_re_emit():
    part = _part(
        [
            Param("w", 8.0, free=True, bounds=(1.0, 20.0), unit="mm"),
            Param("readout", unit="mm", expr=expr_binop("*", expr_param("w"), expr_num(3.0))),
        ]
    )
    controls = {c["param"]: c for c in emit_glsl(part).controls}
    assert controls["readout"]["class"] == "derived"
    assert controls["readout"]["value"] == pytest.approx(24.0)


_GLSL_NAMESPACE = {
    "abs": abs,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pow": math.pow,
    "min": min,
    "max": max,
}


def _eval_glsl(source: str, uniforms: dict) -> float:
    """Evaluate emitted scalar GLSL expression text in Python."""
    return float(eval(source, {"__builtins__": {}}, {**_GLSL_NAMESPACE, **uniforms}))


_PARITY_PARAMS = [
    Param("a", 1.7, free=True, bounds=(0.5, 3.0), unit="mm"),
    Param("b", 0.6, free=True, bounds=(0.1, 2.0), unit="mm"),
]

_A, _B = expr_param("a"), expr_param("b")

_PARITY_CORPUS = {
    "sum": expr_binop("+", _A, _B),
    "difference": expr_binop("-", _A, _B),
    "product": expr_binop("*", _A, _B),
    "quotient": expr_binop("/", _A, _B),
    "nested": expr_binop("-", _A, expr_binop("*", _B, expr_num(2.5))),
    "deep": expr_binop(
        "/", expr_binop("+", _A, _B), expr_binop("-", expr_binop("*", _A, expr_num(3.0)), _B)
    ),
    "neg": expr_unop("neg", expr_binop("+", _A, _B)),
    "abs": expr_unop("abs", expr_binop("-", _B, _A)),
    "sqrt": expr_unop("sqrt", _A),
    "square": expr_unop("square", expr_binop("-", _A, _B)),
    "exp": expr_unop("exp", _B),
    "log": expr_unop("log", _A),
    "pow": expr_binop("pow", _A, expr_num(3.0)),
    "min": expr_binop("min", _A, _B),
    "max": expr_binop("max", expr_binop("*", _A, expr_num(0.1)), _B),
    "reduce_sum": expr_reduce("sum", [_A, _B, expr_num(0.5)]),
    "reduce_mean": expr_reduce("mean", [_A, _B, expr_num(0.5)]),
    "reduce_min": expr_reduce("min", [_A, _B, expr_num(0.5)]),
    "reduce_max": expr_reduce("max", [_A, _B, expr_num(0.5)]),
    "ref_leaf_inside_tree": expr_binop("+", {"$ref": "a"}, _B),
}


@pytest.mark.parametrize("name", sorted(_PARITY_CORPUS))
def test_glsl_matches_eval_expr_pure(name):
    tree = _PARITY_CORPUS[name]
    part = _part(
        [Param(p.name, p.value, free=p.free, bounds=p.bounds, unit=p.unit) for p in _PARITY_PARAMS]
    )

    binding = make_binding(part)
    free_vec = jnp.asarray([p.value for p in _PARITY_PARAMS])
    expected = float(eval_expr_pure(tree, binding, free_vec))

    e = _bare_emitter(part)
    source = e._emit_expr(tree)
    uniforms = {f"u_p_{p.name}": float(p.value) for p in _PARITY_PARAMS}
    assert _eval_glsl(source, uniforms) == pytest.approx(expected, abs=1e-6, rel=1e-6)


@pytest.mark.parametrize("name", ["direct", "chain", "diamond"])
def test_glsl_matches_eval_expr_pure_through_derived_params(name):
    relations = {
        "direct": [Param("d", unit="mm", expr=expr_binop("-", _A, _B))],
        "chain": [
            Param("m", unit="mm", expr=expr_binop("*", _A, expr_num(2.0))),
            Param("d", unit="mm", expr=expr_binop("-", expr_param("m"), _B)),
        ],
        "diamond": [
            Param("m", unit="mm", expr=expr_binop("*", _A, expr_num(2.0))),
            Param("n", unit="mm", expr=expr_binop("+", expr_param("m"), _B)),
            Param("d", unit="mm", expr=expr_binop("/", expr_param("n"), expr_param("m"))),
        ],
    }[name]

    part = _part(
        [Param(p.name, p.value, free=p.free, bounds=p.bounds, unit=p.unit) for p in _PARITY_PARAMS]
        + relations,
    )
    binding = make_binding(part)
    free_vec = jnp.asarray([p.value for p in _PARITY_PARAMS])
    expected = float(binding.get("d", free_vec))

    e = _bare_emitter(part)
    source = e._emit_param_ref("d")
    uniforms = {f"u_p_{p.name}": float(p.value) for p in _PARITY_PARAMS}
    assert _eval_glsl(source, uniforms) == pytest.approx(expected, abs=1e-6, rel=1e-6)
