"""Expression trees in SDF slots, compiled to GLSL.

``dsl.expr`` evaluates these trees in JAX for metrics, meshing and export.
``glsl/emit.py`` compiles the same trees for the shader. A part is optimised
against the first and looked at through the second, so the two have to agree,
and nothing structural forces that: they are separate operator tables in
separate files.

WHAT THESE TESTS DO ABOUT IT. The parity checks below evaluate THE EMITTED
STRING rather than a copy of it. The GLSL subset this compiler produces
(float literals, identifiers, the four infix operators, and calls to
abs/sqrt/log/exp/pow/min/max) is also valid Python for the same arithmetic, so
the emitted text can be run directly against a namespace of uniform values and
compared with ``eval_expr_pure``. No transcription means nothing to go stale.

What that does NOT check is that the text compiles as GLSL. There is no GL
context here, so a syntax error would pass. `test_lib_glsl_contains_every_
dispatch_entry` covers the library side, and a consumer's shader compiler is
the real gate on generated source.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from software_defined_matter.dsl import expr as dsl_expr
from software_defined_matter.dsl.resolve import make_binding
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.glsl.emit import (
    _EXPR_BINARY,
    _EXPR_REDUCE,
    _EXPR_UNARY,
    _GLSLEmitter,
)
from software_defined_matter.model import Param

# Awkward on purpose: negative, fractional, and far enough from 1.0 that a
# dropped operand or a swapped operator moves the answer by a lot.
PARAMS = {
    "a": 3.25,
    "b": -1.75,
    "c": 0.5,
    "d": 7.0,
}


def _part():
    return Part(
        name="expr-part",
        params={
            n: Param(name=n, value=v, free=True, bounds=(-20.0, 20.0)) for n, v in PARAMS.items()
        },
        materials=[
            MaterialRegion(material_id=1, name="m", sdf_tree=sdf_primitive("sphere", r=1.0))
        ],
        metadata={"bbox": [[-9.0, -9.0, -9.0], [9.0, 9.0, 9.0]]},
    )


def _glsl(node) -> str:
    """The emitted GLSL for one expression tree."""
    return _GLSLEmitter(_part(), smooth_csg=False, smooth_k=0.25)._emit_expr(node)


# The GLSL builtins this compiler emits, as Python callables. `pow` and the
# rest are named the same in both languages, which is what makes running the
# emitted text possible at all.
_NS = {
    "abs": abs,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "pow": math.pow,
    "min": min,
    "max": max,
}


def _run_glsl(src: str) -> float:
    """Evaluate emitted GLSL as the arithmetic it is."""
    return float(
        eval(src, {"__builtins__": {}}, {**_NS, **{f"u_p_{k}": v for k, v in PARAMS.items()}})
    )  # noqa: S307


def _run_jax(node) -> float:
    part = _part()
    free = jnp.asarray(part.param_vector(), dtype=jnp.float32)
    return float(dsl_expr.eval_expr_pure(node, make_binding(part), free))


def _check(node):
    """Both evaluators, same tree, same answer."""
    got, want = _run_glsl(_glsl(node)), _run_jax(node)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    return got


def _p(name):
    return {"type": "param", "name": name}


def _n(v):
    return {"type": "num", "value": v}


# ---------------------------------------------------------------------------
# The anti-drift property
# ---------------------------------------------------------------------------


def test_the_two_operator_tables_carry_exactly_the_same_names():
    """An op on one side only is a part optimised with one arithmetic and
    rendered with another. Nothing structural prevents that, so it is asserted
    here rather than left to whoever adds the next operator."""
    assert set(_EXPR_UNARY) == set(dsl_expr._UNARY)
    assert set(_EXPR_BINARY) == set(dsl_expr._BINARY)
    assert set(_EXPR_REDUCE) == set(dsl_expr._REDUCE)


# ---------------------------------------------------------------------------
# Every operator, against the JAX evaluator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", sorted(dsl_expr._UNARY))
def test_every_unary_op_matches_jax(op):
    # sqrt and log need a positive argument in both languages.
    child = _p("a") if op in ("sqrt", "log") else _p("b")
    _check({"type": "unop", "op": op, "child": child})


@pytest.mark.parametrize("op", sorted(dsl_expr._BINARY))
def test_every_binary_op_matches_jax(op):
    # `pow` is the one asymmetric case: GLSL leaves a negative base undefined
    # while jnp.power does not, so the two agree on a positive base only.
    lhs = _p("a") if op == "pow" else _p("b")
    _check({"type": "binop", "op": op, "lhs": lhs, "rhs": _p("c")})


@pytest.mark.parametrize("op", sorted(dsl_expr._REDUCE))
def test_every_reduce_op_matches_jax(op):
    node = {"type": "reduce", "op": op, "children": [_p("a"), _p("b"), _p("c"), _n(2.5)]}
    _check(node)
    # Four children, so `mean` divides by the right number and a fold over
    # min/max reaches past the first pair.
    assert len(node["children"]) == 4


def test_a_nested_tree_matches_jax():
    """Precedence and bracketing, which is where a hand-written emitter goes
    wrong silently: the result is a number, just not the right one."""
    node = {
        "type": "binop",
        "op": "*",
        "lhs": {"type": "binop", "op": "-", "lhs": _p("a"), "rhs": _p("b")},
        "rhs": {
            "type": "binop",
            "op": "+",
            "lhs": {"type": "unop", "op": "neg", "child": _p("c")},
            "rhs": {"type": "reduce", "op": "min", "children": [_p("d"), _n(3.0), _p("a")]},
        },
    }
    got = _check(node)
    # Pin the value too, so a change that moves BOTH evaluators the same way is
    # still caught. (3.25 - -1.75) * (-0.5 + min(7, 3, 3.25)) = 5 * 2.5
    np.testing.assert_allclose(got, 12.5, rtol=1e-9)


def test_a_ref_leaf_inside_a_tree_resolves_like_a_param():
    """`dsl.expr` tolerates a `$ref` mixed into a tree, so this must too, or a
    document that loads and evaluates fails only at emission."""
    node = {"type": "binop", "op": "+", "lhs": {"$ref": "a"}, "rhs": _n(1.0)}
    _check(node)


def test_square_is_a_multiply_not_a_pow():
    """`pow(x, 2.0)` is undefined for a negative base in GLSL and defined in
    jnp, so the two would disagree over exactly the inputs a signed offset
    produces. Asserted on a negative argument, which is where it matters."""
    node = {"type": "unop", "op": "square", "child": _p("b")}
    src = _glsl(node)
    assert "pow" not in src
    np.testing.assert_allclose(_check(node), PARAMS["b"] ** 2, rtol=1e-9)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_metric_in_a_slot_refuses_and_says_why():
    """Not an unimplemented case. A metric integrates a sampled grid of the
    whole part, and a shader is evaluating one point."""
    node = {"type": "metric", "name": "volume", "args": {}}
    with pytest.raises(NotImplementedError, match="integrates a sampled grid"):
        _glsl(node)


@pytest.mark.parametrize(
    "node,match",
    [
        ({"type": "dof", "name": "j0"}, "Unknown expression node type"),
        ({"type": "unop", "op": "tan", "child": _n(1.0)}, "Unknown unary op"),
        ({"type": "binop", "op": "%", "lhs": _n(1.0), "rhs": _n(2.0)}, "Unknown binary op"),
    ],
    ids=["dof", "unknown-unop", "unknown-binop"],
)
def test_unknown_nodes_and_operators_refuse_by_name(node, match):
    """`dof` is declared in the 0.2 and 0.3 schemas and evaluated by neither
    compiler, so a document can carry one and be accepted by validation. It
    refuses here with the same message shape `dsl.expr` uses, rather than
    emitting something."""
    with pytest.raises(ValueError, match=match):
        _glsl(node)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_an_expression_in_a_vector_slot_emits_and_registers_its_uniforms():
    """A derived dimension computed in a primitive's slot from two live params."""
    part = _part()
    part.materials[0].sdf_tree = sdf_primitive(
        "box",
        b=[
            {"type": "binop", "op": "-", "lhs": _p("a"), "rhs": _p("c")},
            _p("d"),
            {"type": "binop", "op": "/", "lhs": _p("d"), "rhs": _n(2.0)},
        ],
    )
    em = emit_glsl(part)
    assert "(u_p_a - u_p_c)" in em.scene_source
    assert "(u_p_d / 2.0)" in em.scene_source
    # The uniforms an expression reads have to be declared, or the shader does
    # not link. `b` is untouched by this tree and must NOT appear.
    names = {u.name for u in em.uniforms}
    assert {"u_p_a", "u_p_c", "u_p_d"} <= names
    assert "u_p_b" not in names
