"""Evaluator contracts describe captured values, never an unenforced design envelope."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.audit import GapAuditError, assert_gaps
from software_defined_matter.sdf.bake import bake_raster_field
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.exactness import (
    EXACT,
    LOWER_BOUND,
    UNKNOWN,
    DistanceAccuracyWarning,
    infer_sdf_exactness,
)


def _rounded_box():
    return sdf_primitive("round_box", b=[1, 1, 1], r={"$ref": "r"})


@pytest.mark.parametrize("pose", [False, True], ids=["design-bounds", "pose-envelope"])
def test_fixed_union_contract_uses_captured_translation_instead_of_disjoint_bounds(pose):
    param = Param("x", 1, bounds=(4, 5))
    if pose:
        param.bounds = None
        param.ui = {"role": "pose", "explore_bounds": [4, 5]}
    part = Part(name="overlapping", params={"x": param})
    sphere = sdf_primitive("sphere", r=1)
    tree = sdf_op("union", [sphere, sdf_transform("translate", sphere, t=[{"$ref": "x"}, 0, 0])])
    assert infer_sdf_exactness(tree, part).is_exact  # Conditional envelope proof.
    fn = make_sdf_closure(tree, part)
    assert fn.distance_exactness.outside == EXACT
    assert fn.distance_exactness.inside == LOWER_BOUND
    # The union boundary at this midpoint is sqrt(3)/2 away, not 1/2.
    assert float(fn(jnp.array([0.5, 0, 0]))) == pytest.approx(-0.5, abs=1e-6)


def test_fixed_rounding_outside_declared_domain_cannot_claim_exactness():
    part = Part(name="rounding", params={"r": Param("r", 2, bounds=(0, 1))})
    tree = _rounded_box()
    assert infer_sdf_exactness(tree, part).is_exact
    fn = make_sdf_closure(tree, part)
    assert fn.distance_exactness.outside == fn.distance_exactness.inside == UNKNOWN
    node = bake_raster_field(fn, ((-2, -2, -2), (2, 2, 2)), 1)
    assert node["params"]["provenance"]["source_distance_exactness"]["outside"] == "unknown"


@pytest.mark.parametrize("initial", [0.5, 2], ids=["initial-in-bounds", "initial-out-of-bounds"])
def test_unrestricted_free_vectors_cannot_inherit_a_bounds_proof_under_jit(initial):
    part = Part(name="free-rounding", params={"r": Param("r", initial, free=True, bounds=(0, 1))})
    tree = _rounded_box()
    assert infer_sdf_exactness(tree, part).is_exact
    fn = make_sdf_closure(tree, part)
    assert fn.distance_exactness.outside == fn.distance_exactness.inside == UNKNOWN
    point = jnp.array([3.0, 0, 0], dtype=jnp.float32)
    value_and_grad = jax.jit(jax.value_and_grad(lambda free: fn(point, free)))
    for radius in (0.5, 2.0):
        value, grad = value_and_grad(jnp.array([radius], dtype=jnp.float32))
        assert bool(jnp.isfinite(value))
        assert bool(jnp.all(jnp.isfinite(grad)))
        reference = make_sdf_closure(
            sdf_primitive("round_box", b=[1, 1, 1], r=radius), Part(name="reference")
        )
        assert float(value) == pytest.approx(float(reference(point)), abs=1e-6)
    assert float(fn(point)) == pytest.approx(float(fn(point, jnp.array([initial]))), abs=1e-6)
    assert fn.distance_exactness.outside == UNKNOWN


def test_contract_and_evaluation_keep_the_same_captured_fixed_value_after_part_mutation():
    part = Part(name="snapshot", params={"r": Param("r", 0.5, bounds=(0, 1))})
    tree = _rounded_box()
    fn = make_sdf_closure(tree, part)
    point = jnp.array([3.0, 0, 0])
    before = float(fn(point))
    part.params["r"].value = 2
    assert fn.distance_exactness.is_exact
    assert float(fn(point)) == pytest.approx(before, abs=1e-6)
    assert make_sdf_closure(tree, part).distance_exactness.outside == UNKNOWN


@pytest.mark.parametrize("free", [False, True], ids=["fixed", "free-default"])
def test_gap_assertions_check_actual_values_instead_of_declared_bounds(free):
    part = Part(
        name="gap-domain",
        params={"r": Param("r", 2, free=free, bounds=(0, 1))},
        materials=[
            MaterialRegion(material_id=1, name="a", sdf_tree=_rounded_box()),
            MaterialRegion(
                material_id=2,
                name="b",
                sdf_tree=sdf_transform("translate", sdf_primitive("sphere", r=1), t=[5, 0, 0]),
            ),
        ],
    )
    with (
        pytest.warns(DistanceAccuracyWarning, match="unknown outside distance"),
        pytest.raises(GapAuditError, match="UNVERIFIED"),
    ):
        assert_gaps(part, {("a", "b"): 3}, d_voxel=1, d_tol=100)


@pytest.mark.parametrize("free", [False, True], ids=["fixed", "free-default"])
def test_gap_sampling_domain_tracks_actual_values_outside_the_declared_envelope(free):
    part = Part(
        name="translated-gap",
        params={"y": Param("y", 100, free=free, bounds=(0, 1))},
        materials=[
            MaterialRegion(
                material_id=i + 1,
                name=name,
                sdf_tree=sdf_transform(
                    "translate",
                    sdf_primitive("sphere", r=1),
                    t=[i * 2.2, {"$ref": "y"}, 0],
                ),
            )
            for i, name in enumerate(("a", "b"))
        ],
    )
    (check,) = assert_gaps(part, {("a", "b"): 0.2}, d_voxel=0.5, d_tol=0.02)
    assert check.b_ok
    assert check.d_measured == pytest.approx(0.2, abs=0.02)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_captured_translation_cannot_inherit_finite_bounds(value):
    part = Part(name="nonfinite", params={"x": Param("x", value, bounds=(0, 1))})
    tree = sdf_transform("translate", sdf_primitive("sphere", r=1), t=[{"$ref": "x"}, 0, 0])
    assert infer_sdf_exactness(tree, part).is_exact
    contract = make_sdf_closure(tree, part).distance_exactness
    assert contract.outside == contract.inside == UNKNOWN
