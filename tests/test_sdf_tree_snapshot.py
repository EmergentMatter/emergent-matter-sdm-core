"""Compiled fields and their contracts must survive edits to the source tree."""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import Param, Part, sdf_modifier, sdf_primitive, sdf_transform
from software_defined_matter.dsl.resolve import make_binding
from software_defined_matter.sdf.compile import make_sdf_closure, make_sdf_closure_with_binding
from software_defined_matter.sdf.exactness import EXACT, UNKNOWN, DistanceAccuracyWarning


def _compile(tree, part, entry):
    if entry == "part":
        return make_sdf_closure(tree, part)
    return make_sdf_closure_with_binding(tree, make_binding(part))


@pytest.mark.parametrize("entry", ["part", "binding"])
def test_mutating_rounding_cannot_change_a_compiled_field_or_its_contract(entry):
    tree = sdf_primitive("round_box", b=[1, 1, 1], r=0.5)
    part = Part(name="snapshot")
    fn = _compile(tree, part, entry)
    contract = fn.distance_exactness
    point = jnp.array([3.0, 0, 0])
    warm = jax.jit(fn)
    assert float(warm(point)) == pytest.approx(2, abs=1e-6)
    tree["params"]["r"] = 2
    assert float(fn(point)) == pytest.approx(2, abs=1e-6)
    assert float(warm(point)) == pytest.approx(2, abs=1e-6)
    # Force a fresh trace after mutation instead of only testing a cached executable.
    assert float(jax.jit(lambda p: fn(p))(point)) == pytest.approx(2, abs=1e-6)
    assert fn.distance_exactness == contract
    changed = _compile(tree, part, entry)
    assert float(changed(point)) == pytest.approx(math.sqrt(18) - 2, abs=1e-6)
    assert changed.distance_exactness.outside == UNKNOWN
    if entry == "part":
        assert contract.is_exact


@pytest.mark.parametrize("entry", ["part", "binding"])
def test_nested_vector_edits_cannot_change_a_compiled_field(entry):
    extents = [1, 2, 3]
    translation = [0, 0, 0]
    tree = sdf_transform("translate", sdf_primitive("box", b=extents), t=translation)
    part = Part(name="nested")
    fn = _compile(tree, part, entry)
    point = jnp.array([4.0, 0, 0])
    extents[0] = 0.25
    translation[0] = 10
    assert float(fn(point)) == pytest.approx(3, abs=1e-6)
    assert float(_compile(tree, part, entry)(point)) == pytest.approx(5.75, abs=1e-6)


@pytest.mark.parametrize("entry", ["part", "binding"])
def test_inline_expression_edits_cannot_escape_the_compiled_domain(entry):
    radius = {
        "type": "binop",
        "op": "+",
        "lhs": {"type": "num", "value": 0.25},
        "rhs": {"type": "num", "value": 0.25},
    }
    tree = sdf_primitive("round_box", b=[1, 1, 1], r=radius)
    part = Part(name="expression")
    fn = _compile(tree, part, entry)
    radius["lhs"]["value"] = 1.75
    assert float(fn(jnp.array([3.0, 0, 0]))) == pytest.approx(2, abs=1e-6)
    if entry == "part":
        assert fn.distance_exactness.is_exact
    assert _compile(tree, part, entry).distance_exactness.outside == UNKNOWN


@pytest.mark.parametrize("entry", ["part", "binding"])
def test_snapshot_preserves_live_free_vectors_and_float32_jitted_gradients(entry):
    with jax.enable_x64(False):
        radius = {"$ref": "r"}
        translation = [0, 0, 0]
        tree = sdf_transform("translate", sdf_primitive("sphere", r=radius), t=translation)
        part = Part(
            name="free", params={"r": Param("r", 0.5, free=True), "other": Param("other", 2.0)}
        )
        fn = _compile(tree, part, entry)
        radius["$ref"] = "other"
        translation[0] = 10
        value_and_grad = jax.jit(jax.value_and_grad(lambda free: fn(jnp.array([3.0, 0, 0]), free)))
        for value in (0.5, 1.0):
            distance, grad = value_and_grad(jnp.array([value], dtype=jnp.float32))
            assert float(distance) == pytest.approx(3 - value, abs=1e-6)
            assert float(grad[0]) == pytest.approx(-1, abs=1e-6)
        assert fn.distance_exactness.outside == UNKNOWN


def test_recompilation_rechecks_shell_contracts_and_diagnostics_after_tree_edits():
    child = sdf_primitive("round_box", b=[1, 1, 1], r=0.5)
    tree = sdf_modifier("onion", child, thickness=0.1)
    part = Part(name="shell")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        original = make_sdf_closure(tree, part)
    child["params"]["r"] = 2
    with pytest.warns(DistanceAccuracyWarning, match="onion reads a side"):
        changed = make_sdf_closure(tree, part)
    point = jnp.array([3.0, 0, 0])
    assert float(original(point)) == pytest.approx(1.9, abs=1e-6)
    assert original.distance_exactness.outside == EXACT
    assert changed.distance_exactness.outside == UNKNOWN
    assert float(changed(point)) == pytest.approx(math.sqrt(18) - 2.1, abs=1e-6)
