"""The occupancy kernel and the grid it is sampled on.

``fraction_filled(d, h) = clip(0.5 - d/h, 0, 1)`` is exact for axis-aligned
surfaces because the ramp is one cell wide and the per-cell values telescope.
These tests pin that exactness, the compact support that follows from it, the
cubic cells the single ``h`` depends on, and the resolution check that fires
when the grid is too coarse for the exactness to hold obliquely.
"""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    make_param_ref,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.dsl.expr import compile_expr, expr_metric
from software_defined_matter.objectives.metrics import (
    GRID_PHASE,
    cubic_grid,
    fraction_filled,
)


def _part(tree, res=64, **md):
    return Part(
        name="k",
        params={},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
        metadata={"grid_resolution": res, **md},
    )


def _volume(tree, res=64, **md):
    part = _part(tree, res=res, **md)
    fn = compile_expr(expr_metric("volume"), part)
    return float(fn(jnp.zeros((0,), dtype=jnp.float32)))


# ---------------------------------------------------------------------------
# Is the kernel itself shaped the way the exactness argument assumes?
# ---------------------------------------------------------------------------


def test_kernel_is_one_half_on_the_surface():
    assert float(fraction_filled(jnp.asarray(0.0), 1.0)) == pytest.approx(0.5)


def test_kernel_saturates_exactly_half_a_cell_out():
    # Compact support is what lets an empty cell contribute exactly zero, so
    # the sampling box needs no padding.
    assert float(fraction_filled(jnp.asarray(0.51), 1.0)) == 0.0
    assert float(fraction_filled(jnp.asarray(-0.51), 1.0)) == 1.0


def test_kernel_is_a_partition_of_unity_on_the_grid():
    # The property the whole estimator rests on: for samples spaced h, the sum
    # of the ramp grows exactly with where the surface falls between them, at
    # every offset. Summed over a stack of cells, that makes the estimated
    # material length exact.
    h = 1.0
    samples = jnp.arange(-6.0, 7.0) * h
    base = float(jnp.sum(fraction_filled(samples, h)))
    for shift in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
        total = float(jnp.sum(fraction_filled(samples - shift, h)))
        assert total == pytest.approx(base + shift, abs=1e-5)


# ---------------------------------------------------------------------------
# Is a planar axis-aligned interface integrated exactly?
# ---------------------------------------------------------------------------


def _slab(t, *, z=0.0, res=48):
    """A slab far wider than the sampling box, so only its two flat Z faces
    are ever sampled and its rim never is."""
    tree = sdf_primitive("box", b=[50.0, 50.0, t / 2.0])
    if z:
        tree = sdf_transform("translate", tree, t=[0.0, 0.0, z])
    return _volume(
        tree, res=res, bbox=[[-5.0, -5.0, -3.0], [5.0, 5.0, 3.0]], metric_skip_resolution_check=True
    )


@pytest.mark.parametrize("thickness", [2.0, 1.0, 0.5, 0.25])
def test_slab_thicker_than_a_cell_is_exactly_linear_in_thickness(thickness):
    # Each face sits in its own band, so each is integrated exactly and the
    # volume is exactly proportional to the thickness. Cell here is 10/48 =
    # 0.208 mm, so every thickness above is more than one cell.
    ref = _slab(2.0)
    assert _slab(thickness) == pytest.approx(ref * thickness / 2.0, rel=1e-5)


def _half_space(z_face, *, res=48):
    """A block so deep that only its top face, at ``z_face``, is in the box."""
    tree = sdf_transform(
        "translate", sdf_primitive("box", b=[50.0, 50.0, 50.0]), t=[0.0, 0.0, z_face - 50.0]
    )
    return _volume(
        tree, res=res, bbox=[[-5.0, -5.0, -3.0], [5.0, 5.0, 3.0]], metric_skip_resolution_check=True
    )


def test_a_single_face_is_exact_wherever_it_falls_between_samples():
    # Slide one face through a whole cell. Each step moves it by the same
    # distance, so each step must add the same volume: area times the step.
    h = 10.0 / 48
    vals = [_half_space(k / 8.0 * h) for k in range(9)]
    steps = [b - a for a, b in itertools.pairwise(vals)]
    assert min(steps) > 0.0
    assert max(steps) == pytest.approx(min(steps), rel=2e-3)


def test_a_slab_thinner_than_a_cell_depends_on_where_it_sits():
    """Both faces share one band, so the far one is invisible and the volume
    swings with the slab's position.

    This is the failure the resolution check refuses, and pinning it here stops
    anyone re-asserting that thin walls are exact. The old sigmoid kernel was
    far worse in the same regime (+1479% at half a cell), but "better" is not
    "right", which is why the check raises rather than returning either.
    """
    h = 10.0 / 48
    t = 0.5 * h
    vals = [_slab(t, z=k / 8.0 * h) for k in range(8)]
    spread = (max(vals) - min(vals)) / max(vals)
    assert spread > 0.2, "sub-cell slabs are expected to be position-dependent"


# ---------------------------------------------------------------------------
# How fast does a solid with edges and curvature converge?
# ---------------------------------------------------------------------------


def test_box_converges_at_second_order():
    # Edges and corners are not planar, so a box is not exact. The error
    # should fall by about 4x per doubling.
    exact = 8.0 * 6.0 * 4.0
    errs = [
        abs(_volume(sdf_primitive("box", b=[4.0, 3.0, 2.0]), res=r) - exact) / exact
        for r in (16, 32, 64, 128)
    ]
    for coarse, fine in itertools.pairwise(errs):
        assert fine < coarse / 3.0
    assert errs[-1] < 1e-3


def test_curved_solid_converges():
    exact = 4.0 / 3.0 * math.pi * 4.0**3
    coarse = abs(_volume(sdf_primitive("sphere", r=4.0), res=24) - exact) / exact
    fine = abs(_volume(sdf_primitive("sphere", r=4.0), res=96) - exact) / exact
    assert fine < coarse
    assert fine < 1e-3


def test_scale_invariance():
    # A ball and the same ball a thousand times bigger fill the same fraction.
    small = _volume(sdf_primitive("sphere", r=0.05), res=48)
    large = _volume(sdf_primitive("sphere", r=50.0), res=48)
    assert small / (0.05**3) == pytest.approx(large / (50.0**3), rel=1e-4)


# ---------------------------------------------------------------------------
# Does the grid have the cubic cells that a single `h` requires?
# ---------------------------------------------------------------------------


def test_cubic_grid_gives_equal_spacing_on_a_lopsided_box():
    bbox = ((-10.0, -10.0, -8.0), (10.0, 10.0, 8.0))
    grown, res, h = cubic_grid(bbox, count=64)
    (x0, y0, z0), (x1, y1, z1) = grown
    for extent, n in zip((x1 - x0, y1 - y0, z1 - z0), res, strict=False):
        assert extent / n == pytest.approx(h, rel=1e-9)


def test_cubic_grid_never_shrinks_the_box():
    bbox = ((-10.0, -10.0, -8.0), (10.0, 10.0, 8.0))
    grown, _res, _h = cubic_grid(bbox, count=64)
    assert grown[0][0] <= bbox[0][0] and grown[1][0] >= bbox[1][0]
    assert grown[0][2] <= bbox[0][2] and grown[1][2] >= bbox[1][2]


def test_cubic_grid_keeps_cells_cubic_when_the_cap_bites():
    # Clamping one axis would skew the cells; h is raised uniformly instead.
    bbox = ((0.0, 0.0, 0.0), (500.0, 500.0, 100.0))
    grown, res, h = cubic_grid(bbox, voxel_size=0.2, cap=64)
    assert max(res) <= 64
    (x0, y0, z0), (x1, y1, z1) = grown
    for extent, n in zip((x1 - x0, y1 - y0, z1 - z0), res, strict=False):
        assert extent / n == pytest.approx(h, rel=1e-9)


def test_grid_phase_is_irrational_enough_to_miss_round_coordinates():
    # A surface landing exactly on a sample point halves the gradient. The
    # offset moves the samples off the round coordinates CAD produces.
    assert 0.0 < GRID_PHASE < 0.5
    assert abs(GRID_PHASE - 0.25) > 0.05


def test_volume_gradient_is_exact_for_a_moving_axis_aligned_face():
    # V = 48 * bx for a box of half-extents [bx, 3, 2], so dV/dbx = 48. The
    # phase offset is what keeps this off the degenerate alignment where every
    # kernel reads half.
    part = Part(
        name="g",
        params={"bx": Param("bx", 4.0, free=True, bounds=(1.0, 5.0))},
        materials=[
            MaterialRegion(
                material_id=1,
                name="m",
                sdf_tree=sdf_primitive("box", b=[make_param_ref("bx"), 3.0, 2.0]),
            )
        ],
        metadata={"grid_resolution": 64},
    )
    fn = compile_expr(expr_metric("volume"), part)
    g = jax.grad(lambda z: fn(z).reshape(()))(jnp.asarray(part.param_vector(), dtype=jnp.float32))
    assert float(g[0]) == pytest.approx(48.0, rel=0.02)


# ---------------------------------------------------------------------------
# Does the resolution check fire when the grid cannot resolve the geometry?
# ---------------------------------------------------------------------------


def test_under_resolved_lattice_raises_with_a_usable_message():
    tree = sdf_primitive("gyroid", period=5.0, min_thickness=1.0, n_periods=[4, 4, 4])
    with pytest.raises(ValueError) as exc:
        _volume(tree, res=16)
    msg = str(exc.value)
    assert "voxels" in msg
    assert "metric_voxel_size" in msg


def test_a_resolved_lattice_does_not_raise():
    tree = sdf_primitive("gyroid", period=5.0, min_thickness=1.0, n_periods=[4, 4, 4])
    assert _volume(tree, res=128) > 0.0


def test_the_check_can_be_turned_off():
    tree = sdf_primitive("gyroid", period=5.0, min_thickness=1.0, n_periods=[4, 4, 4])
    assert _volume(tree, res=16, metric_skip_resolution_check=True) > 0.0


def test_loose_param_bounds_do_not_block_a_wall_that_is_thick_right_now():
    """The check measures the wall the part currently has, not the thinnest
    its bounds allow.

    Here ``t`` is 1.0 mm and 128 cells resolve that comfortably, but its lower
    bound is 0.0, so a worst-case reading would call the wall unresolvable and
    refuse. Since design parameters are routinely given generous bounds, that
    would block measuring most parts before an optimiser had moved anything.

    The cost of reading current values is that an optimiser driving ``t``
    downward starts raising partway through a run rather than up front. That
    is the intended signal: at that point the grid really has stopped
    resolving the geometry.
    """
    part = Part(
        name="v",
        params={"t": Param("t", 1.0, free=True, bounds=(0.0, 2.0))},
        materials=[
            MaterialRegion(
                material_id=1,
                name="m",
                sdf_tree=sdf_primitive(
                    "gyroid", period=5.0, min_thickness=make_param_ref("t"), n_periods=[4, 4, 4]
                ),
            )
        ],
        metadata={"grid_resolution": 128},
    )
    fn = compile_expr(expr_metric("volume"), part)
    assert float(fn(jnp.asarray(part.param_vector(), dtype=jnp.float32))) > 0.0


# ---------------------------------------------------------------------------
# Deprecated knobs
# ---------------------------------------------------------------------------


def test_metric_tau_warns_and_has_no_effect():
    tree = sdf_primitive("box", b=[4.0, 3.0, 2.0])
    baseline = _volume(tree, res=32)
    with pytest.warns(DeprecationWarning, match="metric_tau"):
        v = _volume(tree, res=32, metric_tau=0.5)
    assert v == pytest.approx(baseline, rel=1e-9)


def test_metric_bbox_pad_warns_and_has_no_effect():
    tree = sdf_primitive("box", b=[4.0, 3.0, 2.0])
    baseline = _volume(tree, res=32)
    with pytest.warns(DeprecationWarning, match="metric_bbox_pad"):
        v = _volume(tree, res=32, metric_bbox_pad=3.0)
    assert v == pytest.approx(baseline, rel=1e-9)


def test_band_can_still_be_widened_deliberately():
    tree = sdf_primitive("sphere", r=4.0)
    narrow = _volume(tree, res=48)
    wide = _volume(tree, res=48, metric_band_cells=2.0)
    exact = 4.0 / 3.0 * math.pi * 4.0**3
    assert abs(narrow - exact) < abs(wide - exact)
