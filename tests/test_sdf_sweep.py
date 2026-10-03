"""Tests for the 3-D ``sweep`` node: a 2-D profile swept along a 3-D path.

The bar: an analytic ground-truth check (a circle swept along a circular path is
a torus), open-path flat caps, general (polygon) profiles, both path kinds
(polyline / bezier / bspline), self-bounding bbox inference, gradient flow
through both path and profile params, rotation-minimising-frame stability on an
inflecting path, and schema validation + round-trip.

Frame control (``frame`` / ``normal0``) has its own section at the end: the
``"cylindrical"`` frame that locks a coil winding's section radial/axial (and the
contrast proving an RMF drifts where it stays locked), and the RMF ``normal0``
seed that starts a swept ribbon in a chosen orientation.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    io,
    make_param_ref,
    sdf_primitive,
    sdf_sweep,
)
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure

_NF = jnp.zeros((0,))


def _part(tree, **params):
    return Part(
        name="x", params=params, materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)]
    )


def _closure(tree, **params):
    p = _part(tree, **params)
    return make_sdf_closure(tree, p), p


def _jclosure(tree, **params):
    """JIT-compiled closure for tests that issue many queries.

    A sweep evaluator dispatches hundreds of jnp ops per call; eager mode pays
    that dispatch cost on every query and materialises every intermediate, so
    multi-query tests spend their time on overhead rather than arithmetic. One
    XLA compile replaces it. The compiled path is the production path
    (grid_sampling evaluates jitted), and the lighter tests in this file keep
    the eager path covered.
    """
    f, p = _closure(tree, **params)
    return jax.jit(f), p


# ---------------------------------------------------------------------------
# Ground truth: circle swept along a circular path == torus
# ---------------------------------------------------------------------------


def test_sweep_circle_along_circle_is_torus():
    Rc, r, K = 8.0, 2.0, 200
    th = np.linspace(0.0, 2 * np.pi, K, endpoint=False)
    path = [[float(Rc * np.cos(a)), float(Rc * np.sin(a)), 0.0] for a in th]
    tree = sdf_sweep(sdf_primitive("circle_2d", r=r), path, path_kind="polyline", closed=True)
    f, _ = _jclosure(tree)

    def torus(P):
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        return jnp.sqrt((jnp.sqrt(x * x + y * y) - Rc) ** 2 + z * z) - r

    pts = jnp.array(
        [
            [Rc, 0, 0.0],
            [Rc + r, 0, 0],
            [Rc + 2 * r, 0, 0],
            [0, 0, 0.0],
            [Rc, 0, r],
            [Rc, 0, r + 1.0],
            [Rc - r, 0, 0],
        ]
    )
    got, ref = f(pts, _NF), torus(pts)
    assert float(jnp.max(jnp.abs(got - ref))) < 0.05  # faceting only


def test_sweep_open_path_has_flat_caps():
    # circle r=1 swept along x in [0, 10] -> a capped cylinder with FLAT ends.
    path = [[float(x), 0.0, 0.0] for x in np.linspace(0, 10, 41)]
    tree = sdf_sweep(sdf_primitive("circle_2d", r=1.0), path, path_kind="polyline", closed=False)
    f, _ = _closure(tree)
    d = f(
        jnp.array([[5, 0, 0.0], [5, 1, 0], [5, 2, 0], [-1, 0, 0.0], [11, 0, 0], [5, 0, 0.5]]), _NF
    )
    assert float(d[0]) < 0.0  # axis centre inside
    assert abs(float(d[1])) < 0.05  # on the wall
    assert abs(float(d[2]) - 1.0) < 0.05  # 1 outside the wall
    assert abs(float(d[3]) - 1.0) < 0.05  # 1 before the start cap (flat)
    assert abs(float(d[4]) - 1.0) < 0.05  # 1 past the end cap (flat)
    assert float(d[5]) < 0.0  # inside, off-axis


# ---------------------------------------------------------------------------
# General profiles + path kinds
# ---------------------------------------------------------------------------


def test_sweep_polygon_profile_inside_outside():
    bar = sdf_sweep(
        sdf_primitive("box_2d", b=[1.0, 1.0]),
        path=[[0, 0, 0], [10, 0, 0], [10, 10, 0]],
        path_kind="polyline",
        closed=False,
    )
    f, _ = _closure(bar)
    assert float(f(jnp.array([[5.0, 0, 0]]), _NF)[0]) < 0.0  # inside the first leg
    assert float(f(jnp.array([[10.0, 5, 0]]), _NF)[0]) < 0.0  # inside the second leg
    assert float(f(jnp.array([[5.0, 5, 0]]), _NF)[0]) > 0.0  # inside the elbow's notch -> outside


def test_sweep_bspline_path_loops_and_validates():
    ring = sdf_sweep(
        sdf_primitive("circle_2d", r=1.5),
        path=[[8, 0, 0], [0, 8, 2], [-8, 0, 0], [0, -8, -2]],
        path_kind="bspline",
    )
    f, p = _jclosure(ring)
    # the periodic loop encloses a hole around the axis: its centre is outside.
    assert float(f(jnp.array([[0.0, 0, 0]]), _NF)[0]) > 0.0
    assert jnp.all(jnp.isfinite(f(jnp.array([[8.0, 0, 0], [3.0, 3, 1]]), _NF)))
    io.validate(p)


def test_sweep_open_bezier_path_compiles():
    # open composite cubic Bezier: 3K+1 control points (K=1 -> 4)
    helix = sdf_sweep(
        sdf_primitive("circle_2d", r=0.8),
        path=[[0, 0, 0], [3, 0, 2], [6, 0, 2], [9, 0, 4]],
        path_kind="bezier",
        closed=False,
    )
    f, _ = _jclosure(helix)
    assert float(f(jnp.array([[0.0, 0, 0]]), _NF)[0]) < 0.0  # at the first anchor, on axis
    assert jnp.all(jnp.isfinite(f(jnp.array([[4.5, 0, 2.0], [9.0, 0, 4]]), _NF)))


# ---------------------------------------------------------------------------
# Bbox inference (self-bounding)
# ---------------------------------------------------------------------------


def test_sweep_bbox_contains_solid():
    tree = sdf_sweep(
        sdf_primitive("circle_2d", r=1.0),
        path=[[0, 0, 0], [10, 0, 0]],
        path_kind="polyline",
        closed=False,
    )
    (lo, hi) = infer_sdf_bbox(tree, _part(tree))
    assert all(np.isfinite(lo)) and all(np.isfinite(hi))
    # must contain the true solid (x in [0,10], y,z in [-1,1]).
    assert lo[0] <= 0.0 and lo[1] <= -1.0 and lo[2] <= -1.0
    assert hi[0] >= 10.0 and hi[1] >= 1.0 and hi[2] >= 1.0


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_sweep_grad_flows_through_path_param():
    # a free Param shifts a path control point; the SDF reacts and passes grads.
    Z = make_param_ref("z1")
    tree = sdf_sweep(
        sdf_primitive("circle_2d", r=1.0),
        path=[[0, 0, 0], [5, 0, Z], [10, 0, 0]],
        path_kind="polyline",
        closed=False,
    )
    f, _p = _jclosure(tree, z1=Param("z1", 0.0, free=True, bounds=(-5.0, 5.0), unit="mm"))
    q = jnp.array([[5.0, 0.0, 2.0]])  # above the mid control point
    d0 = float(f(q, jnp.array([0.0]))[0])
    d3 = float(f(q, jnp.array([3.0]))[0])
    assert abs(d3 - d0) > 0.1  # reacts to the path param
    g = jax.grad(lambda fv: f(q, fv)[0] ** 2)(jnp.array([0.0]))
    assert jnp.isfinite(g[0]) and abs(float(g[0])) > 0.0


def test_sweep_grad_flows_through_profile_param():
    R = make_param_ref("r")
    path = [[float(x), 0.0, 0.0] for x in np.linspace(0, 10, 21)]
    tree = sdf_sweep(sdf_primitive("circle_2d", r=R), path, path_kind="polyline", closed=False)
    f, _p = _jclosure(tree, r=Param("r", 1.0, free=True, bounds=(0.3, 3.0), unit="mm"))
    q = jnp.array([[5.0, 1.5, 0.0]])  # outside r=1, inside r=2
    assert float(f(q, jnp.array([1.0]))[0]) > 0.0
    assert float(f(q, jnp.array([2.0]))[0]) < 0.0
    g = jax.grad(lambda fv: f(q, fv)[0])(jnp.array([1.0]))
    assert jnp.isfinite(g[0]) and float(g[0]) < 0.0  # growing r moves the wall outward


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_sweep_rmf_stable_on_inflecting_path():
    # an S-curve (inflection) is where a Frenet frame flips; the rotation-
    # minimising frame must stay finite everywhere.
    ts = np.linspace(0, 4 * np.pi, 60)
    path = [[float(t), float(2 * np.sin(t)), float(np.cos(t))] for t in ts]
    tree = sdf_sweep(
        sdf_primitive("box_2d", b=[0.6, 0.3]), path, path_kind="polyline", closed=False
    )
    f, _ = _jclosure(tree)
    xs, ys, zs = np.meshgrid(
        np.linspace(0, 12, 12), np.linspace(-3, 3, 8), np.linspace(-3, 3, 8), indexing="ij"
    )
    grid = jnp.asarray(np.stack([xs, ys, zs], axis=-1).reshape(-1, 3))
    assert jnp.all(jnp.isfinite(f(grid, _NF)))


def test_sweep_rejects_bad_path():
    # open bezier needs 3K+1 control points; 5 is invalid.
    bad = sdf_sweep(
        sdf_primitive("circle_2d", r=1.0), path=[[0, 0, 0]] * 5, path_kind="bezier", closed=False
    )
    with pytest.raises(ValueError, match="bezier"):
        make_sdf_closure(bad, _part(bad))(jnp.array([[0.0, 0, 0]]), _NF)


def test_sweep_validates_and_roundtrips():
    tree = sdf_sweep(
        sdf_primitive("bspline_2d" if False else "circle_2d", r=1.0),
        path=[[0, 0, 0], [5, 5, 0], [10, 0, 0]],
        path_kind="polyline",
        closed=False,
    )
    p = _part(tree)
    io.validate(p)
    again = Part.from_dict(p.to_dict())
    assert again.materials[0].sdf_tree["type"] == "sweep"
    assert again.materials[0].sdf_tree["params"]["path_kind"] == "polyline"


def test_sweep_polyline_bend_no_phantom_geometry():
    # Regression: a swept polyline must not report "inside" far from the path.
    # Before the per-segment axial cap, a query beyond an INTERIOR vertex along
    # that segment's tangent kept the cross-section value with no axial penalty,
    # so the tube reported "inside" far off every bend (a phantom spear reaching
    # ~120 mm for a path bounded within ~22 mm). Caps were only at the two global
    # ends, so straight 2-pt paths were fine but every bend leaked.
    prof = sdf_primitive("circle_2d", r=1.0)
    cases = {
        "L-bend": ([[20.0, 0, 0], [30.0, 0, 0], [30.0, 10.0, 0]], False),
        "U-turn": ([[20.0, 0, 0], [30.0, 0, 0], [30.0, 2.0, 0], [20.0, 2.0, 0]], False),
        "closed-square": ([[-15.0, -15, 0], [15.0, -15, 0], [15.0, 15, 0], [-15.0, 15, 0]], True),
    }
    lim, vox = 48.0, 1.0
    xs = np.arange(-lim, lim + vox, vox)
    gx, gy, gz = np.meshgrid(xs, xs, np.arange(-3, 3 + vox, vox), indexing="ij")
    grid = jnp.asarray(np.stack([gx, gy, gz], axis=-1).reshape(-1, 3))
    for label, (path, closed) in cases.items():
        f, _ = _jclosure(sdf_sweep(prof, path, path_kind="polyline", closed=closed))
        d = np.asarray(f(grid, _NF))
        inside = np.asarray(grid)[d < 0.0]
        assert inside.shape[0] > 0, f"{label}: tube vanished"
        # All cases are bounded within radius ~22 mm; any inside point past
        # 40 mm is phantom far-field geometry.
        r = np.sqrt(inside[:, 0] ** 2 + inside[:, 1] ** 2)
        assert float(r.max()) < 40.0, (
            f"{label}: phantom inside geometry at r={float(r.max()):.1f} mm"
        )


# ===========================================================================
# Frame control: cylindrical (coil windings) + RMF normal0 seed
# ===========================================================================
# The default frame is the rotation-minimising frame (RMF). ``frame="cylindrical"``
# instead locks the profile's u-axis to the radial direction (outward from Z) and
# v to ~axial, which is what a coil winding needs: an RMF precesses relative to the
# cylindrical frame over many turns, shearing a rectangular section off its
# turn-to-turn spacing. ``normal0`` seeds the RMF's initial normal so an open sweep
# can start in a chosen orientation.


def _helix_path(Rc, z_lo, z_hi, turns, per):
    """Dense polyline helix: ``turns*per`` points on radius ``Rc`` climbing z."""
    th = np.linspace(0.0, 2 * np.pi * turns, turns * per)
    zc = np.linspace(z_lo, z_hi, th.size)
    path = [
        [float(Rc * np.cos(a)), float(Rc * np.sin(a)), float(z)]
        for a, z in zip(th, zc, strict=False)
    ]
    return path, th, zc


def _cardinal_wall_maxabs(f, Rc, hw, hh, am, zm):
    """Max |SDF| over the 4 points that sit exactly on a radial/axial-locked box
    section (r=Rc±hw at height zm; z=zm±hh at radius Rc). ~0 iff the section is
    still radial/axial-aligned there; large once the frame has rotated off it."""
    cr, sr = math.cos(am), math.sin(am)
    pts = jnp.array(
        [
            [(Rc + hw) * cr, (Rc + hw) * sr, zm],
            [(Rc - hw) * cr, (Rc - hw) * sr, zm],
            [Rc * cr, Rc * sr, zm + hh],
            [Rc * cr, Rc * sr, zm - hh],
        ]
    )
    return float(jnp.max(jnp.abs(f(pts, _NF))))


# ---------------------------------------------------------------------------
# Cylindrical frame
# ---------------------------------------------------------------------------


def test_sweep_cylindrical_circle_along_circle_is_torus():
    # A rotationally-symmetric profile can't tell the frames apart: the
    # cylindrical frame must reproduce the exact torus, same as the RMF default.
    Rc, r, K = 8.0, 2.0, 200
    th = np.linspace(0.0, 2 * np.pi, K, endpoint=False)
    path = [[float(Rc * np.cos(a)), float(Rc * np.sin(a)), 0.0] for a in th]
    tree = sdf_sweep(
        sdf_primitive("circle_2d", r=r),
        path,
        path_kind="polyline",
        closed=True,
        frame="cylindrical",
    )
    f, _ = _jclosure(tree)

    def torus(P):
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        return jnp.sqrt((jnp.sqrt(x * x + y * y) - Rc) ** 2 + z * z) - r

    pts = jnp.array(
        [
            [Rc, 0, 0.0],
            [Rc + r, 0, 0],
            [Rc + 2 * r, 0, 0],
            [Rc, 0, r],
            [Rc, 0, r + 1.0],
            [Rc - r, 0, 0],
        ]
    )
    assert float(jnp.max(jnp.abs(f(pts, _NF) - torus(pts)))) < 0.05


def test_sweep_cylindrical_flat_ring_locks_radial_axial():
    # A rectangular section on a flat circular ring: WIDTH (hw, the first box
    # half-extent) sits along the radius, HEIGHT (hh) along Z, at every azimuth.
    # hw != hh so a rotated frame -- or a swapped axis assignment -- would move the
    # walls to the wrong radii/heights. Narrow radial (hw < hh) keeps the swept
    # interior surface-accurate near the walls (a wide section on a tight ring
    # erodes in the deep interior -- the documented approximate-field regime).
    Rc, hw, hh = 10.0, 0.5, 1.2
    # 120 segments: polyline faceting on Rc=10 is ~0.003 mm there, far under the
    # 0.3 mm wall offsets these assertions use, and the segment count is what
    # the compile time scales with.
    th = np.linspace(0.0, 2 * np.pi, 120, endpoint=False)
    path = [[float(Rc * np.cos(a)), float(Rc * np.sin(a)), 0.0] for a in th]
    tree = sdf_sweep(
        sdf_primitive("box_2d", b=[hw, hh]),
        path,
        path_kind="polyline",
        closed=True,
        frame="cylindrical",
    )
    f, _ = _jclosure(tree)
    dth = 2 * math.pi / len(path)
    # mid-SEGMENT azimuths (vertices seam the radial interior)
    for i in (int(fr * (len(path) - 1)) for fr in (0.125, 0.375, 0.625, 0.87)):
        a = (i + 0.5) * dth
        cr, sr = math.cos(a), math.sin(a)
        assert float(f(jnp.array([[Rc * cr, Rc * sr, 0.0]]), _NF)[0]) < 0.0  # section centre
        assert (
            float(f(jnp.array([[(Rc + hw + 0.3) * cr, (Rc + hw + 0.3) * sr, 0.0]]), _NF)[0]) > 0.0
        )  # past radial wall
        assert (
            float(f(jnp.array([[(Rc + hw - 0.3) * cr, (Rc + hw - 0.3) * sr, 0.0]]), _NF)[0]) < 0.0
        )  # inside radial wall
        assert float(f(jnp.array([[Rc * cr, Rc * sr, hh + 0.3]]), _NF)[0]) > 0.0  # past axial wall
        assert (
            float(f(jnp.array([[Rc * cr, Rc * sr, hh - 0.3]]), _NF)[0]) < 0.0
        )  # inside axial wall
        # axes are not interchangeable: a purely-radial offset of hh is OUTSIDE
        # (hh > hw), while the same offset taken axially (hw < hh) is INSIDE.
        assert float(f(jnp.array([[(Rc + hh) * cr, (Rc + hh) * sr, 0.0]]), _NF)[0]) > 0.0
        assert float(f(jnp.array([[Rc * cr, Rc * sr, float(hw)]]), _NF)[0]) < 0.0


def test_sweep_cylindrical_locks_over_many_turns():
    # A TALL section (hw << hh) on a 6-turn helix: with frame='cylindrical' the
    # width stays radial and the height axial at EVERY turn -- early, middle, and
    # LATE (where an RMF has precessed tens of degrees). hw=1 << hh=2.5 so a
    # drifted frame would put the radial offset well inside the tall section and
    # flip these inside/outside checks.
    Rc, hw, hh = 12.0, 1.0, 2.5
    # per=30: faceting on Rc=12 is ~0.07 mm, far under the 0.5 mm clearances
    # below, while compile time scales with the segment count.
    turns, per = 6, 30
    path, th, zc = _helix_path(Rc, -24.0, 24.0, turns, per)  # pitch 8 > 2*hh -> turns clear
    tree = sdf_sweep(
        sdf_primitive("box_2d", b=[hw, hh]),
        path,
        path_kind="polyline",
        closed=False,
        frame="cylindrical",
    )
    f, _ = _jclosure(tree)
    for frac in (0.1, 0.5, 0.9):
        i = int(frac * (th.size - 2))
        am = 0.5 * (th[i] + th[i + 1])  # mid-segment (off the vertex corner)
        zm = 0.5 * (zc[i] + zc[i + 1])
        cr, sr = math.cos(am), math.sin(am)
        assert float(f(jnp.array([[(Rc + hw - 0.5) * cr, (Rc + hw - 0.5) * sr, zm]]), _NF)[0]) < 0.0
        assert float(f(jnp.array([[(Rc + hw + 0.5) * cr, (Rc + hw + 0.5) * sr, zm]]), _NF)[0]) > 0.0
        assert float(f(jnp.array([[Rc * cr, Rc * sr, zm + hh - 0.5]]), _NF)[0]) < 0.0
        assert float(f(jnp.array([[Rc * cr, Rc * sr, zm + hh + 0.5]]), _NF)[0]) > 0.0


def test_sweep_rmf_drifts_where_cylindrical_locks():
    # The whole reason 'cylindrical' exists. On one 6-turn helix, compare the
    # cylindrical sweep against the RMF default via the on-wall residual: the
    # cylindrical section stays radial/axial everywhere (residual ~0), while the
    # RMF agrees only near the seeded start and then precesses far off the walls.
    Rc, hw, hh = 12.0, 1.0, 2.5
    # RMF precession accumulates per unit of arc, not per segment, so the
    # coarser polyline drifts identically (faceting ~0.07 mm, margins are 1.0).
    turns, per = 6, 30
    path, th, zc = _helix_path(Rc, -24.0, 24.0, turns, per)
    prof = sdf_primitive("box_2d", b=[hw, hh])
    fc, _ = _jclosure(
        sdf_sweep(prof, path, path_kind="polyline", closed=False, frame="cylindrical")
    )
    fr, _ = _jclosure(sdf_sweep(prof, path, path_kind="polyline", closed=False, frame="rmf"))

    def maxabs(f, frac):
        i = int(frac * (th.size - 2))
        return _cardinal_wall_maxabs(
            f, Rc, hw, hh, 0.5 * (th[i] + th[i + 1]), 0.5 * (zc[i] + zc[i + 1])
        )

    for frac in (0.0, 0.3, 0.5, 0.7, 0.98):
        assert maxabs(fc, frac) < 0.1, "cylindrical must stay locked at every turn"
    assert maxabs(fr, 0.0) < 0.1  # RMF's world-axis seed happens to start ~radial here
    assert maxabs(fr, 0.3) > 1.0  # ...and has precessed right off the walls a third of the way in


def test_sweep_cylindrical_grad_flows_through_path_param():
    # frame='cylindrical' derives the radial direction from the path points, so a
    # free path param feeds the frame (normalise + cross) as well as the position:
    # grad must stay finite and non-zero -- the new frame code introduces no
    # NaN-producing singularity in the differentiable path.
    X = make_param_ref("cx")
    tree = sdf_sweep(
        sdf_primitive("box_2d", b=[1.0, 1.0]),
        path=[[X, 0, 0], [0, 10, 0], [-10, 0, 0], [0, -10, 0]],
        path_kind="polyline",
        closed=True,
        frame="cylindrical",
    )
    f, _ = _jclosure(tree, cx=Param("cx", 10.0, free=True, bounds=(8.0, 14.0), unit="mm"))
    # Query OUTSIDE the section (box interior grads are NaN at the corner-norm, a
    # box_2d property unrelated to the frame); the outer wall follows the vertex.
    q = jnp.array([[11.5, 0.0, 0.0]])
    d0 = float(f(q, jnp.array([10.0]))[0])
    d3 = float(f(q, jnp.array([13.0]))[0])
    assert abs(d3 - d0) > 0.05  # SDF reacts to the path param
    g = jax.grad(lambda fv: f(q, fv)[0])(jnp.array([10.0]))
    assert jnp.isfinite(g[0]) and abs(float(g[0])) > 0.0


# ---------------------------------------------------------------------------
# RMF normal0 seed
# ---------------------------------------------------------------------------


def test_sweep_normal0_seeds_initial_frame():
    # Box swept along +x. normal0=[0,0,1] forces the profile's u-axis (width hw)
    # into Z and v-axis (height hh) into Y, so the walls sit at z=+-hw, y=+-hh.
    hw, hh = 1.5, 0.5
    path = [[float(x), 0.0, 0.0] for x in np.linspace(0, 10, 21)]
    tree = sdf_sweep(
        sdf_primitive("box_2d", b=[hw, hh]),
        path,
        path_kind="polyline",
        closed=False,
        normal0=[0.0, 0.0, 1.0],
    )
    f, p = _jclosure(tree)
    assert float(f(jnp.array([[5.0, 0.0, hw - 0.2]]), _NF)[0]) < 0.0  # within hw in z -> inside
    assert float(f(jnp.array([[5.0, 0.0, hw + 0.2]]), _NF)[0]) > 0.0  # beyond hw in z -> outside
    assert float(f(jnp.array([[5.0, hh + 0.2, 0.0]]), _NF)[0]) > 0.0  # beyond hh in y -> outside
    assert float(f(jnp.array([[5.0, hh - 0.2, 0.0]]), _NF)[0]) < 0.0  # within hh in y -> inside
    # and the seed round-trips through the wire format.
    again = Part.from_dict(p.to_dict())
    assert again.materials[0].sdf_tree["params"]["normal0"] == [0.0, 0.0, 1.0]


def test_sweep_normal0_rotates_frame_vs_default():
    # The default world-axis seed puts the profile's width along Y; normal0=[0,0,1]
    # rotates it to Z. A point that is inside under one seed is outside under the
    # other -- proof the seed actually re-orients the section.
    hw, hh = 1.5, 0.5
    path = [[float(x), 0.0, 0.0] for x in np.linspace(0, 10, 21)]
    prof = sdf_primitive("box_2d", b=[hw, hh])
    f_def, _ = _jclosure(sdf_sweep(prof, path, path_kind="polyline", closed=False))
    f_seed, _ = _jclosure(
        sdf_sweep(prof, path, path_kind="polyline", closed=False, normal0=[0.0, 0.0, 1.0])
    )
    a = jnp.array(
        [[5.0, 0.0, 1.0]]
    )  # z=1: within hw along Z, beyond hh along... default width is Y
    b = jnp.array([[5.0, 1.0, 0.0]])  # y=1: within hw along Y (default), beyond hh under the seed
    assert float(f_def(a, _NF)[0]) > 0.0  # default: z=1 beyond hh -> outside
    assert float(f_seed(a, _NF)[0]) < 0.0  # seed:    z=1 within hw -> inside
    assert float(f_def(b, _NF)[0]) < 0.0  # default: y=1 within hw -> inside
    assert float(f_seed(b, _NF)[0]) > 0.0  # seed:    y=1 beyond hh -> outside


def test_sweep_normal0_parallel_to_tangent_falls_back():
    # A normal0 parallel to the start tangent projects to ~0 and must fall back to
    # the stable world-axis seed -- identical field to giving no normal0 at all.
    hw, hh = 1.5, 0.5
    path = [[float(x), 0.0, 0.0] for x in np.linspace(0, 10, 21)]
    prof = sdf_primitive("box_2d", b=[hw, hh])
    f_def, _ = _jclosure(sdf_sweep(prof, path, path_kind="polyline", closed=False))
    f_par, _ = _jclosure(
        sdf_sweep(prof, path, path_kind="polyline", closed=False, normal0=[1.0, 0.0, 0.0])
    )
    pts = jnp.array([[5.0, 0.0, 1.0], [5.0, 1.0, 0.0], [5.0, 0.0, 0.0], [3.0, 0.7, 0.0]])
    assert jnp.allclose(f_def(pts, _NF), f_par(pts, _NF), atol=1e-6)


def test_sweep_normal0_holds_along_straight_path():
    # On a straight path the seeded frame never rotates: the same walls hold at the
    # first station and the last.
    hw, hh = 1.5, 0.5
    path = [[float(x), 0.0, 0.0] for x in np.linspace(0, 10, 21)]
    tree = sdf_sweep(
        sdf_primitive("box_2d", b=[hw, hh]),
        path,
        path_kind="polyline",
        closed=False,
        normal0=[0.0, 0.0, 1.0],
    )
    f, _ = _jclosure(tree)
    for x in (1.0, 9.0):  # near start and near end
        assert float(f(jnp.array([[x, 0.0, hw - 0.2]]), _NF)[0]) < 0.0
        assert float(f(jnp.array([[x, 0.0, hw + 0.2]]), _NF)[0]) > 0.0
        assert float(f(jnp.array([[x, hh + 0.2, 0.0]]), _NF)[0]) > 0.0


def test_sweep_normal0_ignored_for_cylindrical():
    # normal0 is an RMF-only knob; with frame='cylindrical' it must have no effect.
    Rc = 10.0
    th = np.linspace(0.0, 2 * np.pi, 120, endpoint=False)
    path = [[float(Rc * np.cos(a)), float(Rc * np.sin(a)), 0.0] for a in th]
    prof = sdf_primitive("box_2d", b=[1.5, 0.8])
    f0, _ = _jclosure(sdf_sweep(prof, path, path_kind="polyline", closed=True, frame="cylindrical"))
    f1, _ = _jclosure(
        sdf_sweep(
            prof,
            path,
            path_kind="polyline",
            closed=True,
            frame="cylindrical",
            normal0=[0.0, 0.0, 1.0],
        )
    )
    pts = jnp.array([[Rc, 0.0, 0.0], [Rc + 2.0, 0.0, 0.0], [Rc, 0.0, 1.2], [0.0, Rc, 0.3]])
    assert jnp.allclose(f0(pts, _NF), f1(pts, _NF), atol=1e-6)


# ---------------------------------------------------------------------------
# Defaults, errors, serialisation
# ---------------------------------------------------------------------------


def test_sweep_default_frame_is_rmf():
    # Omitting frame must record "rmf" and evaluate identically to asking for it.
    ts = np.linspace(0, 4 * np.pi, 60)
    path = [[float(t), float(2 * np.sin(t)), float(np.cos(t))] for t in ts]
    prof = sdf_primitive("box_2d", b=[0.6, 0.3])
    t_default = sdf_sweep(prof, path, path_kind="polyline", closed=False)
    assert t_default["params"]["frame"] == "rmf"
    assert "normal0" not in t_default["params"]  # only emitted when given
    f_default, _ = _jclosure(t_default)
    f_rmf, _ = _jclosure(sdf_sweep(prof, path, path_kind="polyline", closed=False, frame="rmf"))
    xs, ys, zs = np.meshgrid(
        np.linspace(0, 12, 10), np.linspace(-3, 3, 6), np.linspace(-3, 3, 6), indexing="ij"
    )
    grid = jnp.asarray(np.stack([xs, ys, zs], axis=-1).reshape(-1, 3))
    assert jnp.allclose(f_default(grid, _NF), f_rmf(grid, _NF), atol=1e-6)


def test_sweep_rejects_unknown_frame():
    prof = sdf_primitive("circle_2d", r=1.0)
    path = [[0, 0, 0], [10, 0, 0]]
    bad = sdf_sweep(prof, path, path_kind="polyline", closed=False, frame="bogus")
    with pytest.raises(ValueError, match="frame"):
        make_sdf_closure(bad, _part(bad))(jnp.array([[5.0, 0, 0]]), _NF)
    # ...and at the ops layer directly.
    from software_defined_matter.sdf import sdf_ops as ops

    with pytest.raises(ValueError, match="frame"):
        ops.sweep(
            lambda q2d: jnp.zeros(q2d.shape[:-1]),
            jnp.array([[0.0, 0, 0], [1.0, 0, 0]]),
            jnp.zeros((1, 3)),
            frame="bogus",
        )


def test_sweep_cylindrical_roundtrips_and_grad():
    # frame survives serialisation and the cylindrical sweep stays differentiable.
    R = make_param_ref("r")
    th = np.linspace(0.0, 2 * np.pi, 40, endpoint=False)
    path = [[float(10 * np.cos(a)), float(10 * np.sin(a)), 0.0] for a in th]
    tree = sdf_sweep(
        sdf_primitive("circle_2d", r=R),
        path,
        path_kind="polyline",
        closed=True,
        frame="cylindrical",
    )
    p = _part(tree, r=Param("r", 1.0, free=True, bounds=(0.3, 3.0), unit="mm"))
    io.validate(p)
    again = Part.from_dict(p.to_dict())
    assert again.materials[0].sdf_tree["params"]["frame"] == "cylindrical"
    f = jax.jit(make_sdf_closure(tree, p))
    g = jax.grad(lambda fv: f(jnp.array([[11.5, 0.0, 0.0]]), fv)[0])(jnp.array([1.0]))
    assert jnp.isfinite(g[0]) and float(g[0]) < 0.0  # growing r pushes the wall out
