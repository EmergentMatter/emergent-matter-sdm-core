"""Overlap comparisons require exact interiors, even when exterior distances are exact."""

from __future__ import annotations

import math
import warnings

import pytest

from software_defined_matter import MaterialRegion, Part, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.audit import GapAuditError, GapCheck, assert_gaps, gap_audit
from software_defined_matter.sdf.exactness import (
    EXACT,
    LOWER_BOUND,
    UNKNOWN,
    DistanceAccuracyWarning,
)


def _sphere(d_radius, d_x):
    return sdf_transform("translate", sdf_primitive("sphere", r=d_radius), t=[d_x, 0, 0])


def _overlap_part():
    union = sdf_op("union", [_sphere(1, -0.5), _sphere(1, 0.5)])
    return Part(
        name="overlap",
        materials=[MaterialRegion(1, "union", union), MaterialRegion(2, "ball", _sphere(0.1, 0))],
    )


@pytest.mark.parametrize("pair", [("union", "ball"), ("ball", "union")], ids=["first", "second"])
def test_inexact_interior_cannot_pass_a_matching_negative_gap(pair):
    part = _overlap_part()
    with pytest.warns(DistanceAccuracyWarning, match="exact interior distances required"):
        (check,) = gap_audit(part, {pair: -0.6}, d_voxel=0.25)
    assert check.d_measured == pytest.approx(-0.6, abs=1e-6)
    # At the origin the true union depth is sqrt(1 - 0.5**2). Thus the
    # true distance-sum minimum is already <= -0.966 there, not -0.6.
    assert -math.sqrt(0.75) - 0.1 < check.d_measured - check.d_tol
    assert check.b_exact_outside
    assert (check.inside_a, check.inside_b) == (
        (LOWER_BOUND, EXACT) if pair[0] == "union" else (EXACT, LOWER_BOUND)
    )
    assert not check.b_ok
    assert "UNVERIFIED" in str(check)
    assert "exact interior distances required" in str(check)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DistanceAccuracyWarning)
        with pytest.raises(GapAuditError, match="UNVERIFIED.*exact interior distances required"):
            assert_gaps(part, {pair: -0.6}, d_voxel=0.25)


def test_exact_sphere_interiors_can_pass_a_negative_gap_comparison():
    part = Part(
        name="exact-overlap",
        materials=[MaterialRegion(1, "a", _sphere(1, 0)), MaterialRegion(2, "b", _sphere(1, 1))],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        (check,) = assert_gaps(part, {("a", "b"): -1}, d_voxel=0.25)
    assert check.inside_a == check.inside_b == EXACT
    assert check.b_ok


@pytest.mark.parametrize(
    "declared, measured",
    [(-0.005, 0.005), (0.005, -0.005), (-0.005, -0.005)],
    ids=["declared-negative", "measured-negative", "both-negative"],
)
def test_tolerance_cannot_bypass_the_interior_guard(declared, measured):
    check = GapCheck("a", "b", declared, measured, 0.02, outside_a=EXACT, outside_b=EXACT)
    assert not check.b_ok
    assert "exact interior distances required" in str(check)


@pytest.mark.parametrize("side", [LOWER_BOUND, UNKNOWN], ids=["lower-bound", "unknown"])
def test_either_inexact_interior_prevents_overlap_agreement(side):
    for inside_a, inside_b in ((side, EXACT), (EXACT, side)):
        check = GapCheck(
            "a",
            "b",
            -1,
            -1,
            0.02,
            outside_a=EXACT,
            outside_b=EXACT,
            inside_a=inside_a,
            inside_b=inside_b,
        )
        assert not check.b_ok


def test_exact_interiors_do_not_replace_required_exterior_contracts():
    check = GapCheck("a", "b", -1, -1, 0.02, inside_a=EXACT, inside_b=EXACT)
    assert not check.b_ok
    assert "exact exterior distances required" in str(check)


def test_nonnegative_comparisons_still_need_only_exact_exteriors():
    for d_gap in (0, 1):
        check = GapCheck("a", "b", d_gap, d_gap, 0.02, outside_a=EXACT, outside_b=EXACT)
        assert check.b_ok
        assert "UNVERIFIED" not in str(check)
