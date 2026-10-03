"""``canonical_sector_fold``: N copies around Z for the cost of three children.

The fold maps a query point's azimuth into a single wedge, so a shape with
N-fold symmetry costs three child evaluations rather than N. What these tests
pin:

* the fold agrees with the thing it replaces, an explicit union of N rotated
  copies. Getting that requires evaluating the child at the two neighbouring
  wedges too, and the mutation gate below shows what a single evaluation
  costs: it reports distances up to 0.7 too large, which is the direction
  that lets a sphere tracer step through a surface.
* ``centered`` puts a child authored at angle 0 in the middle of the wedge
  rather than at its edge, for any ``n_sectors``.
* ``phase_frac`` rotates the whole set by a fraction of one sector.
* the boundary of all that: three evaluations see the child's own wedge and
  its two neighbours, so a child authored a whole sector away from the wedge
  is not reproduced. Width is not the constraint, placement is.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Part,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.model import Param
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.transforms import tf_canonical_sector_fold

N_SECTORS = 6
SECTOR = 2.0 * np.pi / N_SECTORS


def _part(tree):
    return Part(
        name="fold-part",
        params={},
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=tree)],
        metadata={},
    )


def _eval(tree, pts):
    closure = make_sdf_closure(tree, _part(tree))
    return np.asarray(closure(jnp.asarray(pts, float), jnp.zeros((0,))))


def _blade(half_y=0.35):
    """A bar along +X, centred at radius 2, thin enough to fit its wedge."""
    return sdf_transform("translate", sdf_primitive("box", b=[0.9, half_y, 0.4]), t=[2.0, 0.0, 0.0])


def _fat_blade():
    """The same bar, wide enough in Y to cross both wedge boundaries.

    Half-width 1.6 at radius 2 spans about +/-55 degrees against the wedge's
    +/-30, so each copy overlaps its two immediate neighbours, which is what
    the extra two evaluations reach.
    """
    return _blade(half_y=1.6)


def _explicit_ring(child):
    """What the fold replaces: the child unioned with N rotated copies."""
    copies = [sdf_transform("rotate_z", child, angle=float(k * SECTOR)) for k in range(N_SECTORS)]
    return sdf_op("union", copies)


def _folded(child, **kw):
    return sdf_transform("canonical_sector_fold", child, n_sectors=N_SECTORS, **kw)


def _ring_points(n_theta=180, radii=(0.4, 1.2, 2.0, 2.6), z=0.0):
    """Points on rings through the geometry, dense in angle.

    Angle is where a fold can be wrong, so sampling rings beats a cube: an
    axis-aligned grid can miss every seam.
    """
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    pts = [(r * np.cos(t), r * np.sin(t), z) for r in radii for t in theta]
    return np.asarray(pts, dtype=float)


# ---------------------------------------------------------------------------
# The fold reproduces the ring it replaces
# ---------------------------------------------------------------------------


def test_a_child_inside_its_wedge_matches_the_explicit_ring():
    pts = _ring_points()
    got = _eval(_folded(_blade()), pts)
    want = _eval(_explicit_ring(_blade()), pts)
    assert np.max(np.abs(got - want)) < 1e-4


def test_a_child_wider_than_its_wedge_also_matches():
    """Copies that overlap their immediate neighbours, the harder case to get
    right."""
    pts = _ring_points()
    got = _eval(_folded(_fat_blade()), pts)
    want = _eval(_explicit_ring(_fat_blade()), pts)
    assert np.max(np.abs(got - want)) < 1e-4


def test_a_child_authored_a_whole_sector_away_is_not_reproduced():
    """Where the three evaluations stop, pinned so the docs stay honest.

    They cover the child's own wedge and its two neighbours. Rotating the child
    a whole sector puts the copies a query should be nearest to outside that
    set, and the fold then over-reports by about 2 units. The explicit ring is
    unchanged by that same rotation, since rotating by one sector only relabels
    which copy is which.

    `transforms.tf_canonical_sector_fold` says width is not the constraint and
    placement is. This is the placement half; `_fat_blade` above is the width
    half, at +/-55 degrees against a +/-30 wedge.
    """
    pts = _ring_points()
    child = sdf_transform("rotate_z", _blade(), angle=float(SECTOR))

    err = _eval(_folded(child), pts) - _eval(_explicit_ring(child), pts)
    assert np.max(err) > 1.0, "a child a whole sector out should over-report"
    assert np.min(err) > -1e-4, "and still never under-report"


@pytest.mark.parametrize("child", [_blade(), _fat_blade()], ids=["fits", "overlaps"])
def test_mutation_gate_one_evaluation_reports_distances_that_are_too_large(child):
    """Fold the point, evaluate the child once, and the field is wrong.

    This holds whether or not the child fits inside its wedge, because the
    error is about the query, not the geometry: a point near a seam is nearer
    to the copy next door, and one evaluation never looks there. The error is
    one-sided, so what a single evaluation produces is not a distance bound.
    """
    pts = _ring_points()
    want = _eval(_explicit_ring(child), pts)

    folded_pts = np.asarray(tf_canonical_sector_fold(jnp.asarray(pts), N_SECTORS))
    one_eval = _eval(child, folded_pts)

    err = one_eval - want
    assert np.max(err) > 0.1, "a single evaluation should over-report distance"
    assert np.min(err) > -1e-4, "and can never under-report it"


# ---------------------------------------------------------------------------
# centered and phase_frac
# ---------------------------------------------------------------------------


def test_centered_puts_angle_zero_in_the_middle_of_the_wedge():
    # Uncentered folds into [0, sector), so a point at angle 0 stays at 0, the
    # wedge's edge. Centered folds into [-sector/2, sector/2), where 0 is the
    # middle. The difference shows on a point just BELOW angle 0: uncentered
    # wraps it up to nearly a whole sector, centered leaves it where it is.
    p = np.asarray([[np.cos(-0.05), np.sin(-0.05), 0.0]])

    plain = np.asarray(tf_canonical_sector_fold(jnp.asarray(p), N_SECTORS))
    assert np.arctan2(plain[0, 1], plain[0, 0]) == pytest.approx(SECTOR - 0.05, abs=1e-6)

    centered = np.asarray(tf_canonical_sector_fold(jnp.asarray(p), N_SECTORS, centered=True))
    assert np.arctan2(centered[0, 1], centered[0, 0]) == pytest.approx(-0.05, abs=1e-6)


def test_centering_is_independent_of_the_sector_count():
    """Why `centered` exists rather than pre-rotating the child by sector/2.

    A pre-rotation is baked at whatever count it was authored for, so it is
    wrong the moment the count is scrubbed. Angle 0 must land at angle 0 for
    every n.
    """
    p = np.asarray([[1.0, 0.0, 0.0]])
    for n in (3, 5, 6, 12, 37):
        q = np.asarray(tf_canonical_sector_fold(jnp.asarray(p), n, centered=True))
        assert np.arctan2(q[0, 1], q[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_phase_frac_rotates_the_set_by_a_fraction_of_one_sector():
    # Half a sector of phase turns the copies into the gaps of the unphased
    # set, which is how a rotor and a stator are authored from one child.
    child = _blade()
    pts = _ring_points()

    phased = _eval(_folded(child, centered=True, phase_frac=0.5), pts)
    rotated_ring = _eval(
        sdf_transform("rotate_z", _explicit_ring(child), angle=float(-0.5 * SECTOR)), pts
    )
    assert np.max(np.abs(phased - rotated_ring)) < 1e-4


# ---------------------------------------------------------------------------
# The closure survives jax.jit
# ---------------------------------------------------------------------------
# Every test above evaluates eagerly, which is why this shipped broken for
# three weeks: `int()` on a traced value raises, so a fold could be sampled
# point by point and could not be jitted. `export_part` meshes through a
# jitted grid evaluation, so no part using the fold could be meshed, exported
# or previewed. Assert on the jitted call, not on the eager one.


def _jit_eval(tree, pts, part=None):
    part = part or _part(tree)
    closure = make_sdf_closure(tree, part)
    free = part.param_vector()
    return np.asarray(jax.jit(lambda p: closure(p, free))(jnp.asarray(pts, jnp.float32)))


@pytest.mark.parametrize(
    "n_sectors",
    [6, {"$ref": "n_fixed"}, {"$ref": "n_free"}],
    ids=["literal", "ref-to-fixed-param", "ref-to-free-param"],
)
def test_the_fold_closure_is_jittable_however_the_count_is_authored(n_sectors):
    """All three authoring forms, because the fix could have covered only one.

    Hoisting the count to a build-time `int` would fix the literal and break
    the other two. `radial_bearing` authors 40 folds as `$ref`s, so that
    version of the fix would have taken the acceptance part off entirely.
    """
    tree = sdf_transform("canonical_sector_fold", _blade(), n_sectors=n_sectors)
    part = Part(
        name="fold-part",
        params={
            "n_fixed": Param(name="n_fixed", value=6.0, free=False),
            "n_free": Param(name="n_free", value=6.0, free=True, bounds=(3.0, 12.0)),
        },
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=tree)],
        metadata={},
    )
    got = _jit_eval(tree, _ring_points(), part)
    assert np.isfinite(got).all()
    # Same answer as the eager path, which the tests above pin against the
    # explicit ring. A jitted closure that returns something else would pass a
    # "does it run" check.
    closure = make_sdf_closure(tree, part)
    eager = np.asarray(closure(jnp.asarray(_ring_points(), jnp.float32), part.param_vector()))
    np.testing.assert_allclose(got, eager, rtol=1e-6, atol=1e-6)


def test_a_live_count_actually_changes_the_field_under_jit():
    """The count stays traced, so scrubbing it re-folds without a rebuild.

    A fix that made `n_sectors` static would still pass the test above by
    baking the initial value in. This fails unless the free param is read.
    """
    tree = sdf_transform("canonical_sector_fold", _blade(), n_sectors={"$ref": "n"})
    part = Part(
        name="fold-part",
        params={"n": Param(name="n", value=6.0, free=True, bounds=(3.0, 12.0))},
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=tree)],
        metadata={},
    )
    closure = make_sdf_closure(tree, part)
    jitted = jax.jit(lambda p, fv: closure(p, fv))
    pts = jnp.asarray(_ring_points(), jnp.float32)
    six = np.asarray(jitted(pts, jnp.asarray([6.0], jnp.float32)))
    nine = np.asarray(jitted(pts, jnp.asarray([9.0], jnp.float32)))
    assert not np.allclose(six, nine), "the count was baked in at build time"


def test_the_count_floors_and_agrees_with_the_glsl_twin():
    """4.7 gives 4 wedges, on both compilers.

    `glsl/emit.py` emits `floor(n)`. This used to be `int()`, which truncates
    toward zero, so the two disagreed for a negative count. They agree now.
    """
    tree = sdf_transform("canonical_sector_fold", _blade(), n_sectors={"$ref": "n"})
    part = Part(
        name="fold-part",
        params={"n": Param(name="n", value=4.0, free=True, bounds=(1.0, 12.0))},
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=tree)],
        metadata={},
    )
    closure = make_sdf_closure(tree, part)
    pts = jnp.asarray(_ring_points(), jnp.float32)
    four = np.asarray(closure(pts, jnp.asarray([4.0], jnp.float32)))
    fourish = np.asarray(closure(pts, jnp.asarray([4.7], jnp.float32)))
    np.testing.assert_allclose(four, fourish, rtol=1e-6, atol=1e-6)


def test_softmin_chunked_is_jittable_and_refuses_a_free_chunk_size():
    """The same defect, one node type over.

    `chunk_size` sizes `range(0, n, cs)`, so unlike the fold's count it cannot
    stay traced at any price. It is resolved when the closure is built, and a
    free param is refused BY NAME rather than raising a tracer error later.
    """
    children = [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1.0, 1.0, 1.0])]
    tree = sdf_op("softmin_chunked", children, k=0.2, chunk_size=2)
    got = _jit_eval(tree, _ring_points())
    assert np.isfinite(got).all()

    ref_tree = sdf_op("softmin_chunked", children, k=0.2, chunk_size={"$ref": "cs"})
    part = Part(
        name="c",
        params={"cs": Param(name="cs", value=2.0, free=True, bounds=(1.0, 8.0))},
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=ref_tree)],
        metadata={},
    )
    with pytest.raises(ValueError, match="chunk_size references the free param"):
        make_sdf_closure(ref_tree, part)
