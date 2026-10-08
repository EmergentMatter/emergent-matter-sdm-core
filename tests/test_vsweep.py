"""``vsweep``: any 2-D profile along a polyline, parameters varying along the
length, twist, mitred corners. Checked against shapes whose distance is
known exactly."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import Part, sdf_op, sdf_primitive, sdf_vsweep
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.sdf_vsweep import vsweep_distance as vsweep
from software_defined_matter.sdf.validate import (
    SemanticValidationError,
    validate_document_semantics,
)

PART = Part(name="t", params={}, materials=[], metadata={})


def _box(p, c, b):
    q = np.abs(np.asarray(p) - c) - b
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)


def _eval(node, pts):
    return np.asarray(make_sdf_closure(node, PART)(jnp.asarray(pts, dtype=float), None))


# ── the rounded-rectangle conductor (the original use) ────────────────────


def test_straight_constant_section_is_a_box() -> None:
    """Along +x from 0 to 10, 2 wide (y), 1 tall (z), sharp corners: a box."""
    path = [[0, 0, 0], [10, 0, 0]]
    sec = [[1.0, 0.5, 0.0]] * 2
    rng = np.random.default_rng(0)
    pts = rng.uniform([-3, -3, -3], [13, 3, 3], size=(4000, 3))
    got = np.asarray(vsweep(pts, path, sec))
    want = _box(pts, np.array([5, 0, 0]), np.array([5, 1, 0.5]))
    assert np.allclose(got, want, atol=1e-6)


def test_the_section_tapers_linearly_along_the_path() -> None:
    """Half-width 1 at x = 0, 2 at x = 10: at x = 5 the side is at |y| = 1.5."""
    path = [[0, 0, 0], [10, 0, 0]]
    sec = [[1.0, 0.5, 0.0], [2.0, 0.5, 0.0]]
    d_in, d_on, d_out = np.asarray(vsweep([[5, 1.4, 0], [5, 1.5, 0], [5, 1.6, 0]], path, sec))
    assert d_in < 0 < d_out and abs(d_on) < 1e-3


def test_up_vector_sets_which_way_is_tall() -> None:
    """Up = +y: the 2-wide-by-1-tall bar has its 1 along y and its 2 along z."""
    path = [[0, 0, 0], [10, 0, 0]]
    sec = [[1.0, 0.5, 0.0]] * 2
    up = [[0, 1, 0]] * 2
    d = np.asarray(vsweep([[5, 0.9, 0], [5, 0, 0.9]], path, sec, up=up))
    assert d[0] > 0 and d[1] < 0


def test_corners_are_mitred() -> None:
    """An L in plan: the outer corner is a sharp point at (11, -1), not a ball."""
    path = [[0, 0, 0], [10, 0, 0], [10, 10, 0]]
    sec = [[1.0, 0.5, 0.0]] * 3
    inside, outside = np.asarray(vsweep([[10.95, -0.95, 0], [11.05, -1.05, 0]], path, sec))
    assert inside < 0 < outside


def test_a_tall_bar_does_not_throw_a_fin_past_a_corner() -> None:
    """An L, 1 wide and 5 tall. Just past the corner along the first leg's
    direction, outside the second leg's width, is OUTSIDE. A ball joint would
    report it inside (the section spun round the path reaches 2.5)."""
    path = [[0, 0, 0], [10, 0, 0], [10, 10, 0]]
    sec = [[0.5, 2.5, 0.0]] * 3
    d = float(vsweep([[10 + 1.5, 0.0, 1.0]], path, sec)[0])
    assert d > 0, "a fin sticks out past the corner"
    d2 = float(vsweep([[10 + 0.45, -0.45, 1.0]], path, sec)[0])
    assert d2 < 0, "the mitred corner itself is solid"


def test_nothing_spills_past_a_sharp_corner_inside_the_first_legs_width() -> None:
    """A 135 degree turn. Just past the first leg's end mitre, still within
    its width but outside the second leg, is OUTSIDE: the first slab must
    not count it as inside just because its section contains it."""
    path = [[0, 0, 0], [10, 0, 0], [5, 5, 0]]
    sec = [[1.0, 0.5, 0.0]] * 3
    assert float(vsweep([[11.5, 0.0, 0.0]], path, sec)[0]) > 0
    assert float(vsweep([[9.5, 0.0, 0.0]], path, sec)[0]) < 0


def test_a_segment_along_the_up_vector_keeps_its_section() -> None:
    """With the default up (+Z), a leg running straight up has no up left
    once the along-path part is removed; it must fall back to another axis,
    not vanish."""
    profile = sdf_primitive("box_2d", b=[1.0, 1.0])
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0], [10, 0, 10]])
    d = _eval(node, [[10.0, 0.0, 5.0], [10.0, 0.0, 9.5], [10.0, 0.0, 11.0], [10.0, 2.0, 5.0]])
    assert d[0] < 0 and d[1] < 0 and d[2] > 0 and d[3] > 0


def test_a_modifier_amount_can_vary_along_the_length() -> None:
    """``round`` on a square profile, its radius growing along the path:
    the corner region swells (round grows the shape outward by r)."""
    from software_defined_matter import sdf_modifier

    profile = sdf_modifier("round", sdf_primitive("box_2d", b=[1.0, 1.0]), r={"$along": [0.0, 0.5]})
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0]])
    d = _eval(node, [[0.5, 1.2, 0.0], [9.5, 1.2, 0.0]])
    assert d[0] > 0 > d[1]
    lo, hi = infer_sdf_bbox(node, PART, mode="values")
    assert hi[1] >= 1.5


def test_collinear_vertices_change_nothing() -> None:
    path = [[0, 0, 0], [3, 0, 0], [7, 0, 0], [10, 0, 0]]
    sec = [[1.0, 0.5, 0.0]] * 4
    rng = np.random.default_rng(1)
    pts = rng.uniform([-3, -3, -3], [13, 3, 3], size=(2000, 3))
    one = np.asarray(vsweep(pts, [path[0], path[-1]], [sec[0], sec[-1]]))
    many = np.asarray(vsweep(pts, path, sec))
    assert np.allclose(one, many, atol=1e-6)


def test_closed_loop_has_no_caps() -> None:
    """A square loop, 1 x 1 section: the midpoint of every side is inside at
    depth 0.5, and the point in the middle of the loop is outside."""
    path = [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]]
    sec = [[0.5, 0.5, 0.0]] * 4
    d = np.asarray(
        vsweep([[0, -5, 0], [5, 0, 0], [0, 5, 0], [-5, 0, 0], [0, 0, 0]], path, sec, closed=True)
    )
    assert np.allclose(d[:4], -0.5, atol=1e-6) and d[4] > 0


# ── the general node: any profile, any parameter along the length, twist ──


def test_a_circle_profile_swells_along_the_path() -> None:
    """A circle of radius 1 at x = 0 growing to 3 at x = 10: a cone frustum.
    At x = 5 the surface is at radius 2."""
    profile = sdf_primitive("circle_2d", r={"$along": [1.0, 3.0]})
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0]])
    d = _eval(
        node, [[5, 1.9, 0], [5, 0, 2.1], [5, 0, 0], [0.5, 0.99, 0], [9.5, 0, 2.85], [9.5, 3.01, 0]]
    )
    assert d[0] < 0 < d[1] and d[2] < 0 and d[3] < 0 and d[4] < 0 < d[5]
    assert abs(d[0] + 0.1) < 1e-3  # r = 2 exactly at mid-length


def test_corner_radius_grows_along_the_length() -> None:
    """A 2 x 2 rounded box whose corner radius goes from 0 to 1 (a circle)
    along x. At x = 0 the corner (1, 1) is on the surface; at x = 10 it is
    well outside (the section is a circle of radius 1); at x = 5, r = 0.5."""
    profile = sdf_primitive("rounded_box_2d", b=[1.0, 1.0], r={"$along": [0.0, 1.0]})
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0]])
    d = _eval(node, [[0.5, 0.97, 0.97], [9.5, 0.9, 0.9], [5.0, 0.95, 0.95], [5.0, 0.0, 0.99]])
    assert d[0] < 0  # sharp corner at the start
    assert d[1] > 0  # a circle has no corner there
    assert d[2] > 0 and d[3] < 0  # r = 0.5 midway: corner cut back, side intact
    corner = 1.0 - 0.5 * (1 - 1 / math.sqrt(2))  # where a 0.5 radius meets the diagonal
    assert abs(float(_eval(node, [[5.0, corner, corner]])[0])) < 2e-2


def test_the_section_twists_about_the_path() -> None:
    """A 4 wide x 1 tall bar turned through 90 degrees over its length: at
    the start it is wide in y, at the end wide in z."""
    profile = sdf_primitive("box_2d", b=[2.0, 0.5])
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0]], twist=[0.0, math.pi / 2])
    d = _eval(node, [[0.5, 1.8, 0.0], [0.5, 0.0, 1.8], [9.5, 0.0, 1.8], [9.5, 1.8, 0.0]])
    assert d[0] < 0 < d[1] and d[2] < 0 < d[3]
    # halfway it is at 45 degrees: the diagonal direction is inside, the axes are not
    s = 1.5 / math.sqrt(2)
    d = _eval(node, [[5.0, s, s], [5.0, 1.5, 0.0]])
    assert d[0] < 0 < d[1]


def test_any_two_d_tree_works_as_the_profile() -> None:
    """A square with a circular hole, the hole growing along the length."""
    hole = sdf_primitive("circle_2d", r={"$along": [0.2, 0.8]})
    profile = sdf_op("subtract", [sdf_primitive("box_2d", b=[1.0, 1.0]), hole])
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0]])
    d = _eval(node, [[1.0, 0.5, 0.0], [9.0, 0.5, 0.0], [9.0, 0.9, 0.0]])
    assert d[0] < 0  # at the start the hole is small: (0.5, 0) is solid
    assert d[1] > 0  # near the end the hole has grown past it
    assert d[2] < 0  # the wall is still there


def test_an_along_leaf_needs_one_value_per_vertex() -> None:
    profile = sdf_primitive("circle_2d", r={"$along": [1.0, 2.0, 3.0]})
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0]])
    with pytest.raises(ValueError, match="one value per vertex"):
        make_sdf_closure(node, PART)


def _doc(tree):
    return {
        "schema_version": "0.6",
        "name": "p",
        "params": {},
        "materials": [{"material_id": 1, "name": "copper", "sdf_tree": tree}],
    }


def test_validation_accepts_along_only_inside_a_vsweep() -> None:
    profile = sdf_primitive(
        "rounded_box_2d", b={"$along": [[1.0, 0.5], [2.0, 0.5]]}, r={"$along": [0.0, 0.3]}
    )
    validate_document_semantics(
        _doc(sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0]], twist=[0.0, 0.5]))
    )
    with pytest.raises(SemanticValidationError, match="only accepted inside a vsweep"):
        validate_document_semantics(_doc(sdf_primitive("sphere", r={"$along": [1.0, 2.0]})))
    with pytest.raises(SemanticValidationError, match="at least 2"):
        validate_document_semantics(
            _doc(
                sdf_vsweep(sdf_primitive("circle_2d", r={"$along": [1.0]}), [[0, 0, 0], [1, 0, 0]])
            )
        )
    with pytest.raises(SemanticValidationError):  # a vec2 slot wants vec2 values per vertex
        validate_document_semantics(
            _doc(
                sdf_vsweep(
                    sdf_primitive("box_2d", b={"$along": [1.0, 2.0]}), [[0, 0, 0], [1, 0, 0]]
                )
            )
        )


def test_bbox_covers_the_largest_section() -> None:
    profile = sdf_primitive("circle_2d", r={"$along": [1.0, 3.0]})
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0]])
    lo, hi = infer_sdf_bbox(node, PART, mode="values")
    assert lo[0] <= -3.0 and hi[0] >= 13.0 and lo[1] <= -3.0 and hi[2] >= 3.0


def test_bbox_handles_a_csg_profile() -> None:
    """A box with a hole cut out, offset sideways in the profile plane: the
    bound covers the offset box at its largest section. The profile's first
    axis is +y here (up = +z, path along +x), so the offset moves it to y = 3."""
    from software_defined_matter import sdf_transform

    hole = sdf_primitive("circle_2d", r={"$along": [0.2, 0.8]})
    box = sdf_primitive("box_2d", b=[1.0, 2.0])
    profile = sdf_transform("translate", sdf_op("subtract", [box, hole]), t=[3.0, 0.0])
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0]])
    d = _eval(node, [[5.0, 3.9, 0.0], [5.0, 3.0, 1.9], [5.0, 3.0, 0.0], [5.0, 0.0, 0.0]])
    assert d[0] < 0 and d[1] < 0 and d[2] > 0 and d[3] > 0  # wall, wall, hole, not on the axis
    lo, hi = infer_sdf_bbox(node, PART, mode="values")
    assert lo[0] <= 0.0 and hi[0] >= 10.0
    assert lo[1] <= 2.0 and hi[1] >= 4.0 and lo[2] <= -2.0 and hi[2] >= 2.0


def test_the_field_is_finite_with_a_finite_gradient() -> None:
    import jax

    profile = sdf_primitive(
        "rounded_box_2d",
        b={"$along": [[1.0, 0.5], [1.5, 0.7], [1.5, 0.7]]},
        r={"$along": [0.1, 0.4, 0.4]},
    )
    node = sdf_vsweep(profile, [[0, 0, 0], [10, 0, 0], [10, 10, 0]], twist=[0.0, 0.3, 0.3])
    fn = make_sdf_closure(node, PART)
    rng = np.random.default_rng(2)
    pts = jnp.asarray(rng.uniform([-3, -3, -3], [13, 13, 3], size=(500, 3)))
    pts = jnp.concatenate([pts, jnp.asarray([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])])
    d = fn(pts, None)
    g = jax.vmap(jax.grad(lambda q: fn(q[None], None)[0]))(pts)
    assert bool(jnp.all(jnp.isfinite(d))) and bool(jnp.all(jnp.isfinite(g)))
