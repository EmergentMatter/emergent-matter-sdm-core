"""Tests for the 2-D smooth-curve profiles ``bezier_2d`` and ``bspline_2d``.

Both are closed 2-D regions whose boundary curve is sampled to a polyline and
evaluated with the exact :func:`polygon_2d` SDF. The bar (mirroring
``test_sdf_polygon_2d``): direct-call correctness, the curve-specific identities
(Bézier interpolates its anchors; the B-spline approximates, not interpolates),
an independent reference cross-check of the sampling, the DSL consumer paths
(extrusion + revolution), self-bounding bbox inference, gradient flow through a
parametric control point, and schema round-trip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    io,
    make_param_ref,
    sdf_primitive,
)
from software_defined_matter.sdf import sdf_shapes as shapes
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure

S = shapes._CURVE_SAMPLES_PER_SEGMENT
_NF = jnp.zeros((0,))

# A square control polygon (M=4) for the periodic cubic B-spline.
SQUARE = jnp.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])

# A closed 2-segment cubic Bézier oval: anchors P0=(-6,0), P3=(6,0); the rest
# are off-curve handles. seg0 = top arch, seg1 = bottom arch.
OVAL = jnp.array([[-6.0, 0.0], [-6.0, 4.0], [6.0, 4.0], [6.0, 0.0], [6.0, -4.0], [-6.0, -4.0]])


# ---------------------------------------------------------------------------
# Independent references for the sampling (plain per-segment loops, written
# differently from the einsum implementation, so agreement is a real check).
# ---------------------------------------------------------------------------


def _bspline_samples_ref(cps):
    v = [tuple(map(float, pt)) for pt in cps]
    m = len(v)
    out = []
    for j in range(m):
        w = [v[(j + a) % m] for a in range(4)]
        for i in range(S):
            t = i / S
            b = [(1 - t) ** 3, 3 * t**3 - 6 * t**2 + 4, -3 * t**3 + 3 * t**2 + 3 * t + 1, t**3]
            x = sum(b[a] * w[a][0] for a in range(4)) / 6.0
            y = sum(b[a] * w[a][1] for a in range(4)) / 6.0
            out.append([x, y])
    return jnp.array(out)


def _bezier_samples_ref(cps):
    v = [tuple(map(float, pt)) for pt in cps]
    m = len(v)
    k = m // 3
    out = []
    for seg in range(k):
        w = [v[(3 * seg + a) % m] for a in range(4)]
        for i in range(S):
            t = i / S
            mt = 1 - t
            b = [mt**3, 3 * mt**2 * t, 3 * mt * t**2, t**3]
            x = sum(b[a] * w[a][0] for a in range(4))
            y = sum(b[a] * w[a][1] for a in range(4))
            out.append([x, y])
    return jnp.array(out)


# ---------------------------------------------------------------------------
# bspline_2d: direct call
# ---------------------------------------------------------------------------


def test_bspline_inside_outside():
    assert float(shapes.bspline_2d(jnp.array([0.0, 0.0]), SQUARE)) < 0.0  # centroid inside
    assert float(shapes.bspline_2d(jnp.array([50.0, 50.0]), SQUARE)) > 0.0  # far outside


def test_bspline_approximates_not_interpolates():
    # A uniform B-spline does NOT pass through its control points; for a convex
    # control polygon the curve is pulled inward, so a corner control point sits
    # OUTSIDE the curve.
    corner = SQUARE[2]  # (5, 5)
    assert float(shapes.bspline_2d(corner, SQUARE)) > 0.0


def test_bspline_sample_vertex_is_on_boundary():
    # span-0 at t=0 is the curve point (P0 + 4 P1 + P2)/6, which is a polygon
    # vertex of the sampled outline -> distance 0. Also pins the basis math.
    p0, p1, p2 = SQUARE[0], SQUARE[1], SQUARE[2]
    on_curve = (p0 + 4.0 * p1 + p2) / 6.0
    assert abs(float(shapes.bspline_2d(on_curve, SQUARE))) < 1e-4


def test_bspline_matches_reference_polygon():
    ref = shapes.polygon_2d  # exact SDF of the independently-sampled outline
    pts = jnp.array([[0.0, 0.0], [3.0, -3.0], [7.0, 1.0], [-4.0, 2.0], [50.0, 0.0]])
    got = shapes.bspline_2d(pts, SQUARE)
    want = ref(pts, _bspline_samples_ref(SQUARE))
    assert jnp.allclose(got, want, atol=1e-5)


def test_bspline_batched_query():
    pts = jnp.array([[0.0, 0.0], [50.0, 50.0], [5.0, 5.0]])
    d = shapes.bspline_2d(pts, SQUARE)
    assert d.shape == (3,)
    assert d[0] < 0.0 and d[1] > 0.0 and d[2] > 0.0


def test_bspline_rejects_degenerate():
    with pytest.raises(ValueError, match="M >= 4"):
        shapes.bspline_2d(jnp.array([0.0, 0.0]), jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match=r"M >= 4|shape"):
        shapes.bspline_2d(jnp.array([0.0, 0.0]), jnp.zeros((4, 3)))


# ---------------------------------------------------------------------------
# bezier_2d: direct call
# ---------------------------------------------------------------------------


def test_bezier_inside_outside():
    assert float(shapes.bezier_2d(jnp.array([0.0, 0.0]), OVAL)) < 0.0
    assert float(shapes.bezier_2d(jnp.array([50.0, 0.0]), OVAL)) > 0.0


def test_bezier_interpolates_its_anchors():
    # Unlike the B-spline, a Bézier passes THROUGH its anchors (indices 0, 3, …).
    for anchor_idx in (0, 3):
        d = float(shapes.bezier_2d(OVAL[anchor_idx], OVAL))
        assert abs(d) < 1e-4, f"anchor {anchor_idx} should be on the curve, got {d}"


def test_bezier_matches_reference_polygon():
    pts = jnp.array([[0.0, 0.0], [-6.0, 0.0], [6.0, 0.0], [0.0, 5.0], [50.0, 0.0]])
    got = shapes.bezier_2d(pts, OVAL)
    want = shapes.polygon_2d(pts, _bezier_samples_ref(OVAL))
    assert jnp.allclose(got, want, atol=1e-5)


def test_bezier_rejects_degenerate():
    with pytest.raises(ValueError, match="divisible by 3"):
        shapes.bezier_2d(jnp.array([0.0, 0.0]), jnp.zeros((5, 2)))  # 5 not divisible by 3
    with pytest.raises(ValueError, match=">= 6"):
        shapes.bezier_2d(jnp.array([0.0, 0.0]), jnp.zeros((3, 2)))  # only 1 segment


# ---------------------------------------------------------------------------
# DSL consumer paths: extrusion (sweep) + revolution (lathe)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,cps",
    [
        ("bspline_2d", [[-5, -5], [5, -5], [5, 5], [-5, 5]]),
        ("bezier_2d", [[-6, 0], [-6, 4], [6, 4], [6, 0], [6, -4], [-6, -4]]),
    ],
)
def test_profile_compiles_in_extrusion(kind, cps):
    node = {
        "type": "2d_to_3d",
        "method": "extrusion",
        "child": sdf_primitive(kind, control_points=cps),
        "params": {"h": 3.0},
    }
    part = Part(name="x", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=node)])
    f = make_sdf_closure(node, part)
    assert float(f(jnp.array([[0.0, 0.0, 0.0]]), _NF)[0]) < 0.0  # inside the prism
    assert float(f(jnp.array([[0.0, 0.0, 9.0]]), _NF)[0]) > 0.0  # above the cap


def test_profile_compiles_in_revolution():
    # Profile offset from the Z axis (x in [3, 9]) -> a revolved solid of
    # revolution (a smooth torus-like ring).
    prof = sdf_primitive(
        "bspline_2d", control_points=[[6, -3], [9, -1], [9, 1], [6, 3], [3, 1], [3, -1]]
    )
    node = {"type": "2d_to_3d", "method": "revolution", "child": prof, "params": {}}
    part = Part(name="x", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=node)])
    f = make_sdf_closure(node, part)
    assert float(f(jnp.array([[6.0, 0.0, 0.0]]), _NF)[0]) < 0.0  # inside the ring wall
    assert float(f(jnp.array([[0.0, 0.0, 0.0]]), _NF)[0]) > 0.0  # on the axis -> hole


def test_bezier_profile_compiles_in_revolution():
    # Bezier oval offset from the Z axis (radial x in [3, 9]) -> a revolved ring,
    # mirroring the bspline revolution test for the other primitive.
    prof = sdf_primitive(
        "bezier_2d", control_points=[[3, 0], [3, 4], [9, 4], [9, 0], [9, -4], [3, -4]]
    )
    node = {"type": "2d_to_3d", "method": "revolution", "child": prof, "params": {}}
    part = Part(name="x", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=node)])
    f = make_sdf_closure(node, part)
    assert float(f(jnp.array([[6.0, 0.0, 0.0]]), _NF)[0]) < 0.0  # inside the ring wall
    assert float(f(jnp.array([[0.0, 0.0, 0.0]]), _NF)[0]) > 0.0  # on the axis -> hole


@pytest.mark.parametrize("deform", ["twist", "bend"])
def test_profile_extrusion_with_deform(deform):
    # The PR advertises a twisted column (extrude + twist) and an arched beam
    # (extrude + bend); lock in that both deform paths compile, sign correctly,
    # and actually warp the field away from the straight sweep.
    prof = sdf_primitive("bspline_2d", control_points=[[-5, -5], [5, -5], [5, 5], [-5, 5]])
    lift = {"type": "2d_to_3d", "method": "extrusion", "child": prof, "params": {"h": 3.0}}
    node = {"type": "deform", "deform": deform, "child": lift, "params": {"k": 0.08}}

    part_d = Part(name="d", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=node)])
    part_s = Part(name="s", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=lift)])
    fd = make_sdf_closure(node, part_d)
    fs = make_sdf_closure(lift, part_s)

    assert float(fd(jnp.array([[0.0, 0.0, 0.0]]), _NF)[0]) < 0.0  # inside the deformed prism
    assert float(fd(jnp.array([[0.0, 0.0, 9.0]]), _NF)[0]) > 0.0  # above the cap
    # the deform genuinely warps the field: an off-axis point differs from the
    # undeformed extrusion (the rotation angle there is non-zero).
    q = jnp.array([[4.0, 4.0, 2.0]])
    assert abs(float(fd(q, _NF)[0]) - float(fs(q, _NF)[0])) > 1e-3


def test_profile_bbox_inferred_from_control_points():
    # Extrusion bbox = control-point hull in XY, +/- h in Z. Self-bounding,
    # no explicit metadata bbox needed.
    prof = sdf_primitive("bspline_2d", control_points=[[-5, -5], [5, -5], [5, 5], [-5, 5]])
    node = {"type": "2d_to_3d", "method": "extrusion", "child": prof, "params": {"h": 3.0}}
    part = Part(name="x", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=node)])
    lo, hi = infer_sdf_bbox(node, part)
    assert lo == (-5.0, -5.0, -3.0)
    assert hi == (5.0, 5.0, 3.0)


def test_control_point_param_reacts_and_grad_flows():
    # A free Param driving one control-point x must flow through the sampling
    # and pass gradients (the profile stays optimisable).
    R = make_param_ref("r")
    prof = sdf_primitive("bspline_2d", control_points=[[R, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]])
    part = Part(
        name="g",
        params={"r": Param("r", 6.0, free=True, bounds=(1.0, 12.0), unit="mm")},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=prof)],
    )
    f = make_sdf_closure(prof, part)
    centre = jnp.array([[0.0, 0.0]])
    d6 = float(f(centre, jnp.array([6.0]))[0])
    d11 = float(f(centre, jnp.array([11.0]))[0])
    assert d6 < 0.0 and d11 < 0.0
    assert abs(d11 - d6) > 0.05  # the SDF reacts to the control-point param
    g = jax.grad(lambda fv: f(centre, fv)[0] ** 2)(jnp.array([6.0]))
    assert jnp.isfinite(g[0]) and abs(float(g[0])) > 0.0  # and passes a usable gradient


def test_profile_validates_and_roundtrips():
    for kind, cps in (
        ("bspline_2d", [[-5, -5], [5, -5], [5, 5], [-5, 5]]),
        ("bezier_2d", [[-6, 0], [-6, 4], [6, 4], [6, 0], [6, -4], [-6, -4]]),
    ):
        node = {
            "type": "2d_to_3d",
            "method": "extrusion",
            "child": sdf_primitive(kind, control_points=cps),
            "params": {"h": 2.0},
        }
        part = Part(name="x", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=node)])
        io.validate(part)  # schema + boundedness
        again = Part.from_dict(part.to_dict())
        assert again.materials[0].sdf_tree["child"]["kind"] == kind
