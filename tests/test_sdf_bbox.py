from __future__ import annotations

import math

import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    field_op,
    field_primitive,
    make_param_ref,
    sdf_2d_to_3d,
    sdf_deform,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.sdf.bbox import (
    UnboundedParamError,
    UnsupportedSDFNodeError,
    infer_sdf_bbox,
    pad_bbox,
)


def _part_with_tree(tree, params=None):
    return Part(
        name="bbox-part",
        params=params or {},
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=tree)],
    )


def test_sphere_bbox_from_param_bounds():
    tree = sdf_primitive("sphere", r=make_param_ref("r"))
    part = _part_with_tree(
        tree, params={"r": Param("r", 10.0, free=True, bounds=(5.0, 20.0), unit="mm")}
    )
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-20.0, -20.0, -20.0), (20.0, 20.0, 20.0))


def test_csg_subtract_uses_minuend_bbox():
    outer = sdf_primitive("sphere", r=5.0)
    inner = sdf_transform("translate", sdf_primitive("sphere", r=4.0), t=[100.0, 0.0, 0.0])
    tree = sdf_op("subtract", [outer, inner])
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))


def test_translate_rotate_and_round_modifier():
    tree = sdf_modifier(
        "round",
        sdf_transform(
            "rotate_z",
            sdf_transform("translate", sdf_primitive("box", b=[1.0, 2.0, 3.0]), t=[3.0, 0.0, 0.0]),
            angle=math.pi / 2.0,
        ),
        r=0.5,
    )
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    lo, hi = bbox
    # tf_rotate_z rotates the QUERY by +angle, so the geometry itself moves
    # by the inverse: the box centred at x=+3 lands at y=-3 (verified against
    # the JAX closure). The original assertion encoded the mirrored box.
    assert lo[0] <= -2.5 and hi[0] >= 2.5
    assert lo[1] <= -4.5 and hi[1] >= -1.5
    assert lo[2] <= -3.5 and hi[2] >= 3.5


def test_rotate_bbox_matches_query_rotation_convention():
    """Regression: an asymmetric child under rotate_x must land on the side
    the JAX closure puts it (geometry moves by R^T of the query rotation).
    The mirrored box was latent until AABB-pruned union folds used
    per-component boxes: a rotate_x(-pi/2) plate got a z-negated box
    and was pruned out of the viewport."""
    tree = sdf_transform(
        "rotate_x",
        sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[0.0, 0.0, 10.0]),
        angle=-math.pi / 2.0,
    )
    lo, hi = infer_sdf_bbox(tree, _part_with_tree(tree))
    # Geometry is at y = -10 (query rotated by -90deg => geometry at +90deg
    # from +z, i.e. -y under the p @ R.T convention).
    assert lo[1] <= -11.0 + 1e-6 and hi[1] >= -9.0 - 1e-6
    assert hi[1] < 0.0

    Rz90 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    tree_m = sdf_transform(
        "rotate_matrix",
        sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[3.0, 0.0, 0.0]),
        R=Rz90,
    )
    lo, hi = infer_sdf_bbox(tree_m, _part_with_tree(tree_m))
    assert lo[1] <= -4.0 + 1e-6 and hi[1] >= -2.0 - 1e-6


def test_infer_bbox_from_sdf_writes_metadata_field():
    tree = sdf_primitive("capped_cylinder", h=2.0, r=1.0)
    part = _part_with_tree(tree)
    bbox = part.infer_bbox_from_sdf()
    assert bbox == ((-1.0, -1.0, -2.0), (1.0, 1.0, 2.0))
    assert part.metadata["bbox"] == [[-1.0, -1.0, -2.0], [1.0, 1.0, 2.0]]


def test_values_mode_is_tight_vs_bounds_mode():
    tree = sdf_primitive("sphere", r=make_param_ref("r"))
    part = _part_with_tree(
        tree, params={"r": Param("r", 10.0, free=True, bounds=(5.0, 20.0), unit="mm")}
    )
    assert infer_sdf_bbox(tree, part, mode="bounds") == ((-20.0,) * 3, (20.0,) * 3)
    assert infer_sdf_bbox(tree, part, mode="values") == ((-10.0,) * 3, (10.0,) * 3)


def test_invalid_mode_raises():
    tree = sdf_primitive("sphere", r=1.0)
    with pytest.raises(ValueError, match="mode"):
        infer_sdf_bbox(tree, _part_with_tree(tree), mode="nope")


def test_pad_bbox_inflates_every_face():
    assert pad_bbox(((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0)), 0.5) == (
        (-1.5, -2.5, -3.5),
        (1.5, 2.5, 3.5),
    )


def test_unbounded_param_raises():
    tree = sdf_primitive("sphere", r=make_param_ref("r"))
    part = _part_with_tree(tree, params={"r": Param("r", 10.0, free=True, bounds=None, unit="mm")})
    with pytest.raises(UnboundedParamError):
        infer_sdf_bbox(tree, part)


def test_unbounded_primitive_raises():
    tree = sdf_primitive("plane", n=[0.0, 0.0, 1.0], h=0.0)
    part = _part_with_tree(tree)
    with pytest.raises(UnsupportedSDFNodeError):
        infer_sdf_bbox(tree, part)


# ---------------------------------------------------------------------------
# TPMS lattices: bbox derived from n_periods * period.
# ---------------------------------------------------------------------------


def test_gyroid_bbox_from_n_periods():
    """Half-extent per axis = 0.5 * n_periods * period."""
    tree = sdf_primitive("gyroid", period=2.0, min_thickness=0.036755, n_periods=[3, 3, 3])
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0))


def test_tpms_anisotropic_n_periods():
    """Per-axis n_periods produce an anisotropic bbox."""
    tree = sdf_primitive("schwarz_p", period=4.0, min_thickness=0.147021, n_periods=[2, 1, 5])
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-4.0, -2.0, -10.0), (4.0, 2.0, 10.0))


def test_tpms_clipped_by_intersect_uses_aabb_overlap():
    """`intersect(small_shell, large_gyroid)` -> the smaller box wins."""
    shell = sdf_primitive("sphere", r=3.0)
    infill = sdf_primitive("gyroid", period=4.0, min_thickness=0.294042, n_periods=[10, 10, 10])
    tree = sdf_op("intersect", [shell, infill])
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0))


# ---------------------------------------------------------------------------
# CSG bbox semantics: subtract / union / intersect handle unbounded children
# differently. Subtract ignores the subtrahend; intersect is bounded as long
# as at least one child is bounded; union still requires every child.
# ---------------------------------------------------------------------------


def test_intersect_uses_aabb_intersection_when_both_bounded():
    """Two overlapping boxes intersect to their AABB overlap, not their union."""
    a = sdf_primitive("box", b=[3.0, 3.0, 3.0])
    b = sdf_transform("translate", sdf_primitive("box", b=[3.0, 3.0, 3.0]), t=[2.0, 0.0, 0.0])
    tree = sdf_op("intersect", [a, b])
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-1.0, -3.0, -3.0), (3.0, 3.0, 3.0))


def test_intersect_resolves_via_bounded_sibling():
    """`intersect(bounded, unbounded)` should return the bounded child's bbox.

    The classic example is `intersect(shell, plane)`: the half-space defined
    by the plane is analytically unbounded, but the shell clamps the result.
    """
    shell = sdf_primitive("sphere", r=6.0)
    plane = sdf_primitive("plane", n=[0.0, 0.0, 1.0], h=0.0)
    tree = sdf_op("intersect", [shell, plane])
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-6.0, -6.0, -6.0), (6.0, 6.0, 6.0))


def test_intersect_all_children_unbounded_raises():
    """If every child is unbounded, intersect cannot be bounded analytically."""
    tree = sdf_op(
        "intersect",
        [
            sdf_primitive("plane", n=[1.0, 0.0, 0.0], h=0.0),
            sdf_primitive("plane", n=[0.0, 0.0, 1.0], h=0.0),
        ],
    )
    part = _part_with_tree(tree)
    with pytest.raises(UnsupportedSDFNodeError, match="no analytically-bounded"):
        infer_sdf_bbox(tree, part)


def test_intersect_disjoint_aabbs_raises():
    """Children whose AABBs do not overlap → intersection SDF is empty."""
    a = sdf_primitive("sphere", r=1.0)
    b = sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[10.0, 0.0, 0.0])
    tree = sdf_op("intersect", [a, b])
    part = _part_with_tree(tree)
    with pytest.raises(UnsupportedSDFNodeError, match="empty AABB overlap"):
        infer_sdf_bbox(tree, part)


def test_smooth_intersect_uses_aabb_intersection():
    """`smooth_intersect` follows the same AABB rule as `intersect`."""
    a = sdf_primitive("box", b=[2.0, 2.0, 2.0])
    b = sdf_transform("translate", sdf_primitive("box", b=[2.0, 2.0, 2.0]), t=[1.0, 0.0, 0.0])
    tree = sdf_op("smooth_intersect", [a, b], k=0.1)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-1.0, -2.0, -2.0), (2.0, 2.0, 2.0))


def test_subtract_allows_unbounded_subtrahend():
    """`plane` is unbounded but `subtract(bounded, plane)` is bounded by the minuend.

    The bbox inferrer must not evaluate the subtrahend, so a `plane` used as a
    CSG cutting tool resolves to the minuend's box.
    """
    tree = sdf_op(
        "subtract",
        [
            sdf_primitive("sphere", r=5.0),
            sdf_primitive("plane", n=[0.0, 0.0, 1.0], h=0.0),
        ],
    )
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))


def test_smooth_subtract_allows_unbounded_subtrahend():
    tree = sdf_op(
        "smooth_subtract",
        [
            sdf_primitive("sphere", r=5.0),
            sdf_primitive("plane", n=[0.0, 0.0, 1.0], h=0.0),
        ],
        k=0.1,
    )
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))


def test_union_still_requires_all_children_bounded():
    """`union(bounded, unbounded)` is unbounded: both children must resolve."""
    tree = sdf_op(
        "union",
        [
            sdf_primitive("sphere", r=5.0),
            sdf_primitive("plane", n=[0.0, 0.0, 1.0], h=0.0),
        ],
    )
    part = _part_with_tree(tree)
    with pytest.raises(UnsupportedSDFNodeError):
        infer_sdf_bbox(tree, part)


# ---------------------------------------------------------------------------
# repeat_finite: finite tile of the child SDF.
# ---------------------------------------------------------------------------


def test_repeat_finite_bbox():
    """Child sphere of radius 1; period 2; l=[2,1,0] -> ±2*2, ±1*2, ±0 axis inflation."""
    tree = sdf_transform(
        "repeat_finite",
        sdf_primitive("sphere", r=1.0),
        c=2.0,
        l=[2, 1, 0],
    )
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-5.0, -3.0, -1.0), (5.0, 3.0, 1.0))


def test_repeat_finite_zero_repetitions_is_child_bbox():
    tree = sdf_transform(
        "repeat_finite",
        sdf_primitive("box", b=[1.0, 2.0, 3.0]),
        c=5.0,
        l=[0, 0, 0],
    )
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0))


# ---------------------------------------------------------------------------
# 2d_to_3d: extrusion and revolution lifts.
# ---------------------------------------------------------------------------


def test_extrusion_with_circle_2d():
    tree = sdf_2d_to_3d("extrusion", sdf_primitive("circle_2d", r=2.0), h=5.0)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-2.0, -2.0, -5.0), (2.0, 2.0, 5.0))


def test_extrusion_with_box_2d():
    tree = sdf_2d_to_3d("extrusion", sdf_primitive("box_2d", b=[3.0, 1.5]), h=4.0)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-3.0, -1.5, -4.0), (3.0, 1.5, 4.0))


def test_extrusion_with_rounded_box_2d():
    tree = sdf_2d_to_3d(
        "extrusion",
        sdf_primitive("rounded_box_2d", b=[2.0, 1.0], r=0.5),
        h=3.0,
    )
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-2.5, -1.5, -3.0), (2.5, 1.5, 3.0))


def test_revolution_no_offset_is_sphere_like():
    """Revolving a circle around Z with no offset gives a sphere bbox."""
    tree = sdf_2d_to_3d("revolution", sdf_primitive("circle_2d", r=2.0))
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))


def test_revolution_with_offset_is_torus():
    """Circle of radius 2 at offset 3 -> torus major 3 minor 2."""
    tree = sdf_2d_to_3d(
        "revolution",
        sdf_primitive("circle_2d", r=2.0),
        offset=3.0,
    )
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-5.0, -5.0, -2.0), (5.0, 5.0, 2.0))


def test_revolution_with_trapezoid_2d():
    """Trapezoid profile revolved -> conical frustum-shaped bbox."""
    tree = sdf_2d_to_3d(
        "revolution",
        sdf_primitive("trapezoid_2d", r1=3.0, r2=1.0, he=2.0),
    )
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    # max half-width is max(r1, r2) = 3 (sets XY radius after revolution);
    # he sets the Z half-extent.
    assert bbox == ((-3.0, -3.0, -2.0), (3.0, 3.0, 2.0))


def test_2d_to_3d_rejects_non_primitive_child():
    """2D-to-3D lifts only accept a 2D primitive as child."""
    tree = sdf_2d_to_3d(
        "extrusion",
        sdf_op("union", [sdf_primitive("circle_2d", r=1.0)]),
        h=2.0,
    )
    part = _part_with_tree(tree)
    with pytest.raises(UnsupportedSDFNodeError, match="2D primitive"):
        infer_sdf_bbox(tree, part)


# ---------------------------------------------------------------------------
# Deformations: twist / bend / displace. Conservative bbox: twist and bend
# return the inscribing square of the swept disk (rate-independent), and
# displace inflates by the field amplitude.
# ---------------------------------------------------------------------------


def test_twist_bbox_is_xz_disk():
    """Twist around Y: xz extent grows to sqrt(max_x² + max_z²); y unchanged."""
    # Child: box with half-extents (3, 5, 1) -> bbox ((-3,-5,-1),(3,5,1))
    child = sdf_primitive("box", b=[3.0, 5.0, 1.0])
    tree = sdf_deform("twist", child, k=0.5)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    r = math.sqrt(9.0 + 1.0)
    assert bbox == ((-r, -5.0, -r), (r, 5.0, r))


def test_bend_bbox_is_xy_disk():
    """Bend around Z: xy extent grows to sqrt(max_x² + max_y²); z unchanged."""
    # Box with half-extents (3, 4, 2) -> R = 5
    child = sdf_primitive("box", b=[3.0, 4.0, 2.0])
    tree = sdf_deform("bend", child, k=0.5)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-5.0, -5.0, -2.0), (5.0, 5.0, 2.0))


def test_twist_offcenter_uses_outer_radius():
    """A child translated away from Y picks up a larger swept radius."""
    # Box at x ∈ [7, 13] (half-extents 3 translated by 10), y free, z ∈ [-1, 1].
    child = sdf_transform(
        "translate",
        sdf_primitive("box", b=[3.0, 1.0, 1.0]),
        t=[10.0, 0.0, 0.0],
    )
    tree = sdf_deform("twist", child, k=0.3)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    # Child bbox is ((7,-1,-1),(13,1,1)); max_x = 13, max_z = 1.
    r = math.sqrt(13.0 * 13.0 + 1.0)
    assert bbox == ((-r, -1.0, -r), (r, 1.0, r))


def test_bend_rate_independent_bounds():
    """The bound is the same for k=0.001 and k=10: it's a worst-case envelope."""
    child = sdf_primitive("box", b=[2.0, 3.0, 1.0])
    tiny = sdf_deform("bend", child, k=0.001)
    huge = sdf_deform("bend", child, k=10.0)
    part = _part_with_tree(tiny)
    assert infer_sdf_bbox(tiny, part) == infer_sdf_bbox(huge, part)


def test_displace_inflates_by_field_amplitude():
    """sin_xyz amplitude clamps |field|, so the bbox inflates by exactly that much."""
    child = sdf_primitive("sphere", r=5.0)
    field = field_primitive("sin_xyz", freq=[1.0, 1.0, 1.0], amplitude=0.3)
    tree = sdf_deform("displace", child, field=field)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-5.3, -5.3, -5.3), (5.3, 5.3, 5.3))


def test_displace_radial_field():
    child = sdf_primitive("sphere", r=4.0)
    field = field_primitive("radial", freq=0.5, amplitude=0.2, phase=0.0)
    tree = sdf_deform("displace", child, field=field)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-4.2, -4.2, -4.2), (4.2, 4.2, 4.2))


def test_displace_field_add_uses_triangle_inequality():
    """|f1 + f2| ≤ |f1| + |f2|: bbox inflates by the sum."""
    child = sdf_primitive("sphere", r=5.0)
    field = field_op(
        "add",
        [
            field_primitive("sin_xyz", freq=[1.0, 1.0, 1.0], amplitude=0.3),
            field_primitive("radial", freq=2.0, amplitude=0.2),
        ],
    )
    tree = sdf_deform("displace", child, field=field)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-5.5, -5.5, -5.5), (5.5, 5.5, 5.5))


def test_displace_default_amplitude_is_one():
    """field primitives default `amplitude` to 1.0 when omitted."""
    child = sdf_primitive("sphere", r=5.0)
    field = field_primitive("sin_xyz", freq=[1.0, 1.0, 1.0])  # no amplitude
    tree = sdf_deform("displace", child, field=field)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-6.0, -6.0, -6.0), (6.0, 6.0, 6.0))


def test_displace_param_ref_amplitude_uses_worst_case():
    """If `amplitude` is a $ref, the inflation uses the param's bounds (worst case)."""
    child = sdf_primitive("sphere", r=5.0)
    field = field_primitive("sin_xyz", freq=[1.0, 1.0, 1.0], amplitude=make_param_ref("a"))
    tree = sdf_deform("displace", child, field=field)
    part = _part_with_tree(
        tree, params={"a": Param("a", 0.5, free=True, bounds=(-1.0, 2.0), unit="mm")}
    )
    bbox = infer_sdf_bbox(tree, part)  # bounds mode
    # max(|-1|, |2|) = 2 -> inflate by 2.
    assert bbox == ((-7.0, -7.0, -7.0), (7.0, 7.0, 7.0))


def test_nested_deforms_compose():
    """twist(displace(sphere, field)): both inferrers apply."""
    inner = sdf_deform(
        "displace",
        sdf_primitive("sphere", r=5.0),
        field=field_primitive("sin_xyz", freq=[1.0, 1.0, 1.0], amplitude=0.5),
    )
    tree = sdf_deform("twist", inner, k=0.5)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    # inner bbox: ((-5.5,-5.5,-5.5),(5.5,5.5,5.5))
    # twist: x_max = 5.5, z_max = 5.5 -> R = sqrt(5.5² + 5.5²).
    r = math.sqrt(5.5 * 5.5 + 5.5 * 5.5)
    assert bbox == ((-r, -5.5, -r), (r, 5.5, r))
