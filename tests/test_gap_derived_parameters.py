"""Gap audits must use evaluated relations, never stale parameter caches."""

from __future__ import annotations

from copy import deepcopy

import jax
import pytest

from software_defined_matter import MaterialRegion, Param, Part, sdf_primitive, sdf_transform
from software_defined_matter.audit import GapAuditError, assert_gaps, gap_audit
from software_defined_matter.sdf.exactness import DistanceAccuracyWarning


def _pair(params, *, radius=1, spacing=2.2, y=0):
    return Part(
        name="derived-gap",
        params=params,
        materials=[
            MaterialRegion(
                material_id=i + 1,
                name=name,
                sdf_tree=sdf_transform(
                    "translate", sdf_primitive("sphere", r=radius), t=[i * spacing, y, 0]
                ),
            )
            for i, name in enumerate(("a", "b"))
        ],
    )


@pytest.mark.parametrize("b_x64", [False, True], ids=["float32", "float64"])
@pytest.mark.parametrize("cached", [0, None, -100], ids=["stale", "missing", "opposite-side"])
def test_derived_translation_cannot_produce_a_false_clearance_pass(cached, b_x64):
    with jax.enable_x64(b_x64):
        part = _pair(
            {"y": Param("y", cached, expr={"type": "num", "value": 100})},
            y={"$ref": "y"},
        )
        original = deepcopy(part.params)
        # The centers share y=100 and are 2.2 apart: the true gap is 0.2.
        (check,) = assert_gaps(part, {("a", "b"): 0.2}, d_voxel=0.5)
        assert check.b_ok
        assert check.d_measured == pytest.approx(0.2, abs=0.01)  # Grid-refinement error.
        # Sampling around cached y=0 formerly returned this matching false pass.
        with pytest.raises(GapAuditError, match=r"declared 192\.7624, measured 0\.2"):
            assert_gaps(part, {("a", "b"): 192.7624}, d_voxel=0.5)
        assert part.params == original


def test_nested_relations_use_current_free_values_without_refreshing_the_caller():
    # Deliberately reverse dependency order and put the free value outside its
    # declared bounds. Evaluation resolves relations, not dictionary order or bounds.
    part = _pair(
        {
            "y": Param(
                "y",
                None,
                expr={
                    "type": "binop",
                    "op": "*",
                    "lhs": {"type": "param", "name": "mid"},
                    "rhs": {"type": "num", "value": 2},
                },
            ),
            "mid": Param(
                "mid",
                -1,
                expr={
                    "type": "binop",
                    "op": "+",
                    "lhs": {"type": "param", "name": "base"},
                    "rhs": {"type": "num", "value": 10},
                },
            ),
            "base": Param("base", 40.0, free=True, bounds=(0, 1)),
        },
        y={"$ref": "y"},
    )
    original = deepcopy(part.params)
    (check,) = assert_gaps(part, {("a", "b"): 0.2}, d_voxel=0.5)
    assert check.d_measured == pytest.approx(0.2, abs=0.01)
    assert part.params == original


def test_each_audit_resolves_derived_dimensions_again_after_a_base_value_change():
    part = _pair(
        {
            "base": Param("base", 2.0, free=True, bounds=(1, 2)),
            "r": Param("r", 99, expr={"type": "param", "name": "base"}),
        },
        radius={"$ref": "r"},
        spacing=4.2,
    )
    (first,) = assert_gaps(part, {("a", "b"): 0.2}, d_voxel=0.5)
    assert first.d_measured == pytest.approx(0.2, abs=0.01)
    part.params["base"].value = 1.0
    (second,) = assert_gaps(part, {("a", "b"): 2.2}, d_voxel=0.5)
    assert second.d_measured == pytest.approx(2.2, abs=0.01)
    assert part.params["r"].value == 99
    assert part.params["r"].expr is not None
    assert part.params["base"].free


def test_resolved_derived_values_must_satisfy_primitive_preconditions():
    part = _pair({"r": Param("r", 0.5, expr={"type": "num", "value": 2})}, spacing=5)
    part.materials[0].sdf_tree = sdf_primitive("round_box", b=[1, 1, 1], r={"$ref": "r"})
    with (
        pytest.warns(DistanceAccuracyWarning, match="unknown outside distance"),
        pytest.raises(GapAuditError, match="UNVERIFIED"),
    ):
        assert_gaps(part, {("a", "b"): 3}, d_voxel=1, d_tol=100)
    assert part.params["r"].value == 0.5


def test_nonfinite_derived_values_fail_before_sampling():
    part = _pair({"y": Param("y", 0, expr={"type": "num", "value": float("nan")})}, y={"$ref": "y"})
    with pytest.raises(ValueError, match="gap_audit: parameter 'y'.*finite"):
        gap_audit(part, {("a", "b"): 0.2})
