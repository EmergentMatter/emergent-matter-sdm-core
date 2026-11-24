"""Gap audit: measured separations between material regions must match declared clearances.

The double-dip test reproduces (in miniature) the roller-bearing stacked-clearance bug that
motivated this module -- see the org's single-source-clearances pattern (module docstring
above for the full motivation).
"""

from __future__ import annotations

import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive, sdf_transform
from software_defined_matter.audit import (
    GapAuditError,
    assert_gaps,
    gap_audit,
    measure_gap,
)
from software_defined_matter.sdf.compile import make_sdf_closure


def _sphere_at(r: float, x: float) -> dict:
    return sdf_transform("translate", sdf_primitive("sphere", r=r), t=[x, 0.0, 0.0])


def _two_sphere_part(d_gap: float, r: float = 1.0) -> Part:
    return Part(
        name="pair",
        materials=[
            MaterialRegion(material_id=1, name="a", sdf_tree=_sphere_at(r, 0.0)),
            MaterialRegion(material_id=2, name="b", sdf_tree=_sphere_at(r, 2 * r + d_gap)),
        ],
    )


def test_measure_gap_two_spheres_exact():
    part = _two_sphere_part(d_gap=0.3)
    sdf_a = make_sdf_closure(part.materials[0].sdf_tree, part)
    sdf_b = make_sdf_closure(part.materials[1].sdf_tree, part)
    bbox = ((-1.5, -1.5, -1.5), (4.0, 1.5, 1.5))
    assert measure_gap(sdf_a, sdf_b, bbox, d_voxel=0.2) == pytest.approx(0.3, abs=0.01)


def test_measure_gap_overlap_is_negative():
    part = _two_sphere_part(d_gap=-0.5)
    sdf_a = make_sdf_closure(part.materials[0].sdf_tree, part)
    sdf_b = make_sdf_closure(part.materials[1].sdf_tree, part)
    bbox = ((-1.5, -1.5, -1.5), (3.0, 1.5, 1.5))
    assert measure_gap(sdf_a, sdf_b, bbox, d_voxel=0.2) < 0.0


def test_assert_gaps_passes_when_declared_matches():
    part = _two_sphere_part(d_gap=0.2)
    checks = assert_gaps(part, {("a", "b"): 0.2}, d_voxel=0.2, d_tol=0.02)
    assert all(c.b_ok for c in checks)


def test_gap_audit_unknown_region_is_actionable():
    part = _two_sphere_part(d_gap=0.2)
    with pytest.raises(KeyError, match="no material region"):
        gap_audit(part, {("a", "nope"): 0.2})


def test_assert_gaps_catches_double_dipped_clearance():
    """The bearing bug in miniature.

    Intent: a roller with clearance 0.1 per side. The neighbouring bodies are (correctly)
    placed clr away from the roller's NOMINAL surface, but the roller is ALSO undersized
    by clr -- the classic stacking mistake. Declared gap 0.1; real gap 0.2. Must fail.
    """
    d_clr = 0.1
    r_nom = 1.0
    part = Part(
        name="bug",
        materials=[
            MaterialRegion(
                material_id=1,
                name="race_lo",
                sdf_tree=_sphere_at(1.0, -(1.0 + r_nom + d_clr)),
            ),
            # the bug: roller undersized by clr AS WELL as the races being offset by clr
            MaterialRegion(
                material_id=2,
                name="roller",
                sdf_tree=sdf_primitive("sphere", r=r_nom - d_clr),
            ),
            MaterialRegion(
                material_id=3,
                name="race_hi",
                sdf_tree=_sphere_at(1.0, +(1.0 + r_nom + d_clr)),
            ),
        ],
    )
    declared = {("race_lo", "roller"): d_clr, ("roller", "race_hi"): d_clr}

    checks = gap_audit(part, declared, d_voxel=0.2, d_tol=0.02)
    assert not any(c.b_ok for c in checks)  # the stacking is caught
    assert np.allclose([c.d_measured for c in checks], 2 * d_clr, atol=0.02)

    with pytest.raises(GapAuditError, match="measured"):
        assert_gaps(part, declared, d_voxel=0.2, d_tol=0.02)


def test_chunk_size_does_not_change_the_measurement():
    """Chunking is a memory decision, so it must not be an accuracy one.

    ``measure_gap`` evaluates the grid in blocks because a single call allocates
    ``n_points x sum(vertices over every polygon) x 2 x 8`` bytes. Splitting the
    grid changes which points share a call and nothing else, so every chunk size
    must agree exactly, including sizes that divide the grid unevenly and one
    large enough not to chunk at all.
    """
    part = _two_sphere_part(d_gap=0.3)
    sdf_a = make_sdf_closure(part.materials[0].sdf_tree, part)
    sdf_b = make_sdf_closure(part.materials[1].sdf_tree, part)
    bbox = ((-1.5, -1.5, -1.5), (4.0, 1.5, 1.5))

    d_ref = measure_gap(sdf_a, sdf_b, bbox, d_voxel=0.2, chunk_size=10**9)
    for n_chunk in (1, 7, 64, 1000):
        d_chunked = measure_gap(sdf_a, sdf_b, bbox, d_voxel=0.2, chunk_size=n_chunk)
        assert d_chunked == pytest.approx(d_ref, abs=1e-9), (
            f"chunk_size={n_chunk} measured {d_chunked}, unchunked {d_ref}"
        )


def test_matching_ellipsoid_field_estimate_cannot_pass_as_physical_clearance():
    from software_defined_matter.sdf.exactness import DistanceAccuracyWarning

    # The tips are at y=5 and y=7: the true clearance is 2. Along that
    # axis each ellipsoid field scales distance by 1/5, yielding about 0.4.
    tree = sdf_primitive("ellipsoid", r=[1, 5, 1])
    part = Part(
        name="ellipsoid-gap",
        materials=[
            MaterialRegion(material_id=1, name="a", sdf_tree=tree),
            MaterialRegion(
                material_id=2,
                name="b",
                sdf_tree=sdf_transform("translate", tree, t=[0, 12, 0]),
            ),
        ],
    )
    with pytest.warns(DistanceAccuracyWarning, match="field-value estimate"):
        (check,) = gap_audit(part, {("a", "b"): 0.4}, d_voxel=0.5)
    assert check.d_measured == pytest.approx(0.4, abs=0.01)
    assert not check.b_ok
    assert "UNVERIFIED" in str(check)
    # Ignoring warnings must not turn the approximate estimate into a pass.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DistanceAccuracyWarning)
        with pytest.raises(GapAuditError, match="UNVERIFIED.*exact exterior distances required"):
            assert_gaps(part, {("a", "b"): 0.4}, d_voxel=0.5)


@pytest.mark.parametrize("name", ["a", "b"])
@pytest.mark.parametrize("kind", ["ellipsoid", "round_box"])
def test_either_nonexact_exterior_prevents_a_matching_gap_assertion(name, kind):
    from software_defined_matter.sdf.exactness import LOWER_BOUND, UNKNOWN, DistanceAccuracyWarning

    tree = (
        sdf_primitive("ellipsoid", r=[1, 2, 1])
        if kind == "ellipsoid"
        else sdf_primitive("round_box", b=[1, 1, 1], r=2)
    )
    part = Part(
        name="unverified-pair",
        materials=[
            MaterialRegion(
                material_id=i + 1,
                name=region,
                sdf_tree=sdf_transform(
                    "translate",
                    tree if region == name else sdf_primitive("sphere", r=1),
                    t=[i * 5, 0, 0],
                ),
            )
            for i, region in enumerate(("a", "b"))
        ],
    )
    with pytest.warns(DistanceAccuracyWarning, match="field-value estimate"):
        (check,) = gap_audit(part, {("a", "b"): 0}, d_voxel=1)
    expected = LOWER_BOUND if kind == "ellipsoid" else UNKNOWN
    assert (check.outside_a if name == "a" else check.outside_b) == expected
    with (
        pytest.warns(DistanceAccuracyWarning, match="field-value estimate"),
        pytest.raises(GapAuditError, match=f"{name}={expected}"),
    ):
        assert_gaps(part, {("a", "b"): check.d_measured}, d_voxel=1)


def test_gap_check_without_distance_provenance_cannot_pass():
    from software_defined_matter.audit import GapCheck

    check = GapCheck("a", "b", 1, 1, 0.02)
    assert not check.b_ok
    assert "UNVERIFIED" in str(check)


def test_exact_exteriors_can_pass_without_exact_interiors():
    from software_defined_matter import sdf_op
    from software_defined_matter.sdf.exactness import EXACT, LOWER_BOUND, infer_sdf_exactness

    # Overlapping spheres lose exactness inside their union, but retain true
    # exterior lengths. The rightmost tip is x=2; the other body's tip is x=3.
    tree = sdf_op("union", [_sphere_at(1, 0), _sphere_at(1, 1)])
    part = Part(
        name="exterior-only",
        materials=[
            MaterialRegion(material_id=1, name="a", sdf_tree=tree),
            MaterialRegion(material_id=2, name="b", sdf_tree=_sphere_at(1, 4)),
        ],
    )
    contract = infer_sdf_exactness(tree, part)
    assert contract.outside == EXACT
    assert contract.inside == LOWER_BOUND
    (check,) = assert_gaps(part, {("a", "b"): 1}, d_voxel=0.5, d_tol=0.02)
    assert check.b_exact_outside
    assert check.b_ok
