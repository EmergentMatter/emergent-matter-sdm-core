"""Normal entry points diagnose unproven length consumers before numerical evaluation."""

from __future__ import annotations

import json
import warnings

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import MaterialRegion, Param, Part, sdf_modifier, sdf_op, sdf_primitive
from software_defined_matter.dsl.resolve import make_binding
from software_defined_matter.grid_sampling import bind_sdf
from software_defined_matter.io import load, load_part, validate
from software_defined_matter.sdf.compile import make_sdf_closure, make_sdf_closure_with_binding
from software_defined_matter.sdf.exactness import DistanceAccuracyWarning


def _part(tree, **kwargs):
    return Part(
        name="diagnostics",
        materials=[MaterialRegion(material_id=1, name="body", sdf_tree=tree)],
        **kwargs,
    )


def _approximate():
    return sdf_primitive("ellipsoid", r=[1, 2, 3])


def _onion():
    return sdf_modifier("onion", _approximate(), thickness=0.1)


@pytest.mark.parametrize(
    "entry", [load, load_part, Part.load], ids=["load", "load-part", "Part.load"]
)
def test_normal_load_warns_once_with_the_material_path(tmp_path, entry):
    path = tmp_path / "part.sdm"
    path.write_text(json.dumps(_part(_onion()).to_dict()))
    with pytest.warns(DistanceAccuracyWarning, match=r"materials\[0\].sdf_tree: onion") as caught:
        result = entry(path)
    assert isinstance(result, Part)
    assert len(caught) == 1


@pytest.mark.parametrize("entry", ["part", "dict", "path"])
def test_normal_validation_warns_for_length_consumers(tmp_path, entry):
    part = _part(_onion())
    path = tmp_path / "part.sdm"
    path.write_text(json.dumps(part.to_dict()))
    source = {"part": part, "dict": part.to_dict(), "path": path}[entry]
    with pytest.warns(DistanceAccuracyWarning, match="onion"):
        validate(source)


@pytest.mark.parametrize("entry", ["compile", "binding", "grid"])
def test_compilation_warns_once_and_jitted_evaluation_does_not_warn(entry):
    tree = _onion()
    part = _part(tree)
    with pytest.warns(DistanceAccuracyWarning, match="sdf_tree: onion") as caught:
        if entry == "compile":
            fn = make_sdf_closure(tree, part)
        elif entry == "binding":
            fn = make_sdf_closure_with_binding(tree, make_binding(part))
        else:
            fn = bind_sdf(tree, part)
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        point = jnp.array([3.0, 1, 1])
        assert bool(jnp.isfinite(fn(point)))
        assert bool(jnp.isfinite(jax.jit(fn)(point)))
        assert bool(jnp.all(jnp.isfinite(jax.jit(jax.grad(fn))(point))))


@pytest.mark.parametrize(
    "op", ["smooth_union", "smooth_intersect", "smooth_subtract", "softmin_many", "softmin_chunked"]
)
def test_explicit_blends_warn_on_nonexact_inputs(op):
    tree = sdf_op(op, [_approximate(), sdf_primitive("sphere", r=1)], k=0.2)
    with pytest.warns(DistanceAccuracyWarning, match=f"{op} blends non-exact fields"):
        make_sdf_closure(tree, _part(tree))


def test_metadata_smoothing_and_explicit_compiler_override_use_the_same_contract():
    tree = sdf_op("union", [_approximate(), sdf_primitive("sphere", r=1)])
    part = _part(tree, metadata={"smooth_csg": True})
    with pytest.warns(DistanceAccuracyWarning, match="union blends non-exact fields"):
        make_sdf_closure(tree, part)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        make_sdf_closure(tree, part, b_smooth_csg=False)
    part.metadata["smooth_csg"] = False
    with pytest.warns(DistanceAccuracyWarning, match="union blends non-exact fields"):
        make_sdf_closure(tree, part, b_smooth_csg=True)


@pytest.mark.parametrize("radius", [0.0, 0.1, -0.1], ids=["identity", "dilation", "erosion"])
def test_rounding_warns_only_for_the_side_it_reads(radius):
    union = sdf_op("union", [sdf_primitive("sphere", r=1), sdf_primitive("sphere", r=2)])
    tree = sdf_modifier("round", union, r=radius)
    if radius < 0:
        with pytest.warns(DistanceAccuracyWarning, match="round reads a side"):
            make_sdf_closure(tree, _part(tree))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DistanceAccuracyWarning)
            make_sdf_closure(tree, _part(tree))


def test_nested_warning_names_the_consumer_path():
    tree = sdf_op("union", [sdf_primitive("sphere", r=1), _onion()])
    with pytest.warns(DistanceAccuracyWarning, match=r"sdf_tree.children\[1\]: onion") as caught:
        make_sdf_closure(tree, _part(tree))
    assert len(caught) == 1


def test_load_and_compile_do_not_trust_bounds_over_invalid_current_values():
    child = sdf_primitive("round_box", b=[1, 1, 1], r={"$ref": "r"})
    tree = sdf_modifier("onion", child, thickness=0.1)
    part = _part(tree, params={"r": Param("r", 2, unit="mm", bounds=(0, 1))})
    with pytest.warns(DistanceAccuracyWarning, match="onion"):
        validate(part)
    with pytest.warns(DistanceAccuracyWarning, match="onion"):
        make_sdf_closure(tree, part)


def test_free_vector_compilation_warns_without_restricting_parameter_gradients():
    tree = sdf_modifier("onion", sdf_primitive("sphere", r={"$ref": "r"}), thickness=0.1)
    part = _part(tree, params={"r": Param("r", 1, free=True, bounds=(0.5, 2))})
    with pytest.warns(DistanceAccuracyWarning, match="onion"):
        fn = make_sdf_closure(tree, part)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        grad = jax.jit(jax.grad(lambda v: fn(jnp.array([3.0, 0, 0]), v)))
        assert float(grad(jnp.array([1.0]))[0]) == pytest.approx(-1, abs=1e-6)


def test_exact_blend_inputs_and_unvalidated_load_do_not_emit_diagnostics(tmp_path):
    tree = sdf_op(
        "smooth_union", [sdf_primitive("sphere", r=1), sdf_primitive("sphere", r=2)], k=0.2
    )
    path = tmp_path / "part.sdm"
    path.write_text(json.dumps(_part(_onion()).to_dict()))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        make_sdf_closure(tree, _part(tree))
        load(path, b_validate_schema=False)


def test_binding_adapter_warns_without_assuming_a_parambinding_layout():
    from types import SimpleNamespace

    # Assembly occurrence bindings expose get/initial_free_vector, not fixed.
    binding = SimpleNamespace(
        get=lambda name, free: free[0], initial_free_vector=lambda: jnp.array([1.0])
    )
    tree = sdf_modifier("onion", sdf_primitive("sphere", r={"$ref": "r"}), thickness=0.1)
    with pytest.warns(DistanceAccuracyWarning, match="onion"):
        fn = make_sdf_closure_with_binding(tree, binding)
    assert float(fn(jnp.array([2.0, 0, 0]))) == pytest.approx(0.9, abs=1e-6)


def test_load_checks_the_envelope_while_fixed_compilation_checks_captured_values():
    tree = sdf_modifier(
        "onion", sdf_primitive("round_box", b=[1, 1, 1], r={"$ref": "r"}), thickness=0.1
    )
    part = _part(tree, params={"r": Param("r", 0.5, unit="mm", bounds=(0, 2))})
    with pytest.warns(DistanceAccuracyWarning, match="onion"):
        validate(part)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        make_sdf_closure(tree, part)


def test_metric_envelope_diagnostics_happen_before_objective_evaluation():
    from software_defined_matter.dsl.expr import compile_expr

    part = _part(
        sdf_primitive("sphere", r=1),
        metadata={"grid_resolution": 12, "metric_skip_resolution_check": True},
    )
    metric = {"type": "metric", "name": "volume", "args": {"envelope": _onion()}}
    with pytest.warns(DistanceAccuracyWarning, match="onion") as caught:
        fn = compile_expr(metric, part)
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        free = jnp.array([])
        assert bool(jnp.isfinite(fn(free)))
        assert bool(jnp.isfinite(jax.jit(fn)(free)))
