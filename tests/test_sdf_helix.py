"""Tests for the ``helix`` primitive: a circular tube wound about +Z.

The bar, in order of how much each would hurt to get wrong:

1. **Surface exactness**: displacing a centreline point perpendicular to the
   local tangent by ``t`` must read exactly ``(t - r) / k_wall``. The binormal
   direction is a direct test of the ``cos(lead angle)`` rescaling; ``k_wall``
   is the constant divisor that keeps the value a distance inside the winding
   radius, and :func:`_k_wall` recomputes it here from the closed form rather
   than importing it, so both halves of the math are pinned independently.
   Backed by a brute-force check that the field never OVERSTATES clearance
   (which would let a marcher step through the wall).
2. **Branch-cut continuity**: the construction picks the nearest turn off
   ``atan2``, which jumps by 2*pi at the -X half-plane. If the pitch bookkeeping
   is wrong there is a visible seam along -X. This is the single most likely
   way to break the primitive.
3. **Clocking**: ``phase`` must be exactly a rotation about Z. Threads mate by
   registering two helices that share a pitch and differ in phase, so this
   property is load-bearing for the whole superprim story.
4. Lipschitz bound |grad| <= 1, finite gradients on the axis, flat end caps,
   handedness, analytic bbox tightness, and .sdm round-trip.
"""

from __future__ import annotations

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
    sdf_helix,
)
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.sdf_shapes import helix

_NF = jnp.zeros((0,))


def _part(tree, **params):
    return Part(
        name="x", params=params, materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)]
    )


def _closure(tree, **params):
    p = _part(tree, **params)
    return make_sdf_closure(tree, p), p


def _k_wall(major_r, pitch, r):
    """The constant divisor ``helix`` applies to its tube term.

    Bounds ``|grad|`` over the wall shell ``rho in [major_r - r, major_r + r]``.
    Restated from the closed form so a regression in ``sdf_shapes`` cannot hide
    behind a shared expression. 1.0013 at thread proportions, 1.126 at 44 deg
    of lead.
    """
    h = pitch / (2.0 * np.pi)
    lead = 2.0 * np.pi * major_r
    cos_lambda = lead / np.sqrt(lead**2 + pitch**2)
    return max(cos_lambda * np.sqrt(1.0 + (h / (major_r - r)) ** 2), 1.0)


def _centreline(major_r, pitch, n_turns, phase=0.0, handedness=1.0, k=20000):
    """Densely sampled centreline points, for brute-force distance."""
    half_h = 0.5 * n_turns * abs(pitch)
    z = np.linspace(-half_h, half_h, k)
    a = phase + handedness * 2.0 * np.pi * z / pitch
    return np.stack([major_r * np.cos(a), major_r * np.sin(a), z], axis=-1)


def _brute_distance(pts, curve, r, chunk=64):
    """True distance to the tube = distance to the sampled centreline - r.

    Sampling the curve as points rather than segments over-estimates very
    slightly; at these densities the sampling error is far below the tolerances
    used here. Chunked because the full pairwise array is otherwise ~GB.
    """
    pts = np.asarray(pts)
    mins = [
        np.linalg.norm(pts[i : i + chunk, None, :] - curve[None, :, :], axis=-1).min(axis=1)
        for i in range(0, len(pts), chunk)
    ]
    return np.concatenate(mins) - r


# ---------------------------------------------------------------------------
# 1. Ground truth against a brute-forced centreline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "R,pitch,r,n",
    [
        (10.0, 6.0, 1.2, 3.0),  # thread-like: lead 5.5 deg, r/R = 0.12
        (28.0, 3.0, 1.0, 4.0),  # the spherical-bearing cap thread: lead 1.0 deg
    ],
)
def test_helix_is_exact_near_the_surface(R, pitch, r, n):
    """The field must be right where the geometry is: at the tube wall.

    Displacing a centreline point PERPENDICULAR to the local tangent by ``t``
    must read exactly ``(t - r) / k_wall``. Two independent directions are
    checked: radial (which the cos(lambda) rescaling leaves alone) and binormal
    (which is entirely a test OF that rescaling: get cos(lambda) wrong and
    this is the assertion that fails).
    """
    f, _ = _closure(sdf_helix(R, pitch, r, n))
    lam = np.arctan(pitch / (2.0 * np.pi * R))
    kw = _k_wall(R, pitch, r)
    half_h = 0.5 * n * pitch

    # How far we may step and still be nearest to THIS turn. Two ceilings:
    #   - past pitch/2 the neighbouring turn is closer, so `t - r` stops being
    #     the answer (at R=28/pitch=3/r=1 the turns are 3 apart, so a 2.0 step
    #     lands exactly on the next turn's wall: the field says 0, correctly).
    #   - past the band edge the end-cap slab takes over, also correctly.
    t_max = min(2.0 * r, 0.35 * pitch)
    margin = t_max + 0.5
    zc = np.linspace(-half_h + margin, half_h - margin, 23)
    az = 2.0 * np.pi * zc / pitch
    q = np.stack([R * np.cos(az), R * np.sin(az), zc], axis=-1)

    zero = np.zeros_like(az)
    e_r = np.stack([np.cos(az), np.sin(az), zero], axis=-1)
    e_th = np.stack([-np.sin(az), np.cos(az), zero], axis=-1)
    e_z = np.stack([zero, zero, np.ones_like(az)], axis=-1)
    binormal = -np.sin(lam) * e_th + np.cos(lam) * e_z  # perpendicular to tangent

    # atol is 1e-4 mm, i.e. 400x finer than the Fuse can hold: this is a check
    # on the math, and the residual is float32 (JAX's default dtype), not error.
    for t in (0.0, 0.25 * t_max, 0.5 * t_max, 0.75 * t_max, t_max):
        for name, direction in (("radial", e_r), ("binormal", binormal)):
            want = (t - r) / kw
            d = np.asarray(f(jnp.asarray(q + t * direction), _NF))
            assert np.allclose(d, want, atol=1e-4), (
                f"{name} t={t}: max err {float(np.max(np.abs(d - want)))}"
            )


def test_helix_never_overstates_clearance_for_a_thin_tube():
    """Against brute force over the whole neighbourhood: the field must not
    claim MORE room than there is, or a ray-marcher would step through the wall.

    Only a near-zero overshoot budget is allowed, and it shrinks as the tube
    thins relative to the winding radius. (Under-estimates are expected and
    unbounded near the flat end cuts; see the note in ``helix``'s docstring.)
    """
    R, pitch, r, n = 10.0, 6.0, 1.2, 3.0
    f, _ = _closure(sdf_helix(R, pitch, r, n))
    curve = _centreline(R, pitch, n, k=60000)

    rng = np.random.default_rng(0)
    ang = rng.uniform(-np.pi, np.pi, 400)
    rad = rng.uniform(R - 3.0, R + 3.0, 400)
    zs = rng.uniform(-0.5 * n * pitch + 1.0, 0.5 * n * pitch - 1.0, 400)
    pts = np.stack([rad * np.cos(ang), rad * np.sin(ang), zs], axis=-1)

    got = np.asarray(f(jnp.asarray(pts), _NF))
    ref = _brute_distance(pts, curve, r)
    assert np.all(got <= ref + 0.01), float(np.max(got - ref))


def test_helix_sign_and_centreline():
    """Zero-set sits on the tube wall; the centreline is exactly -r/k_wall deep.

    The zero set is what k_wall must NOT move, so the wall is still at r; the
    interior value is scaled with everything else.
    """
    R, pitch, r, n = 8.0, 5.0, 1.0, 2.0
    f, _ = _closure(sdf_helix(R, pitch, r, n))
    on_curve = jnp.asarray(_centreline(R, pitch, n, k=9)[1:-1])  # skip the caps
    d = f(on_curve, _NF)
    assert np.allclose(np.asarray(d), -r / _k_wall(R, pitch, r), atol=2e-3), np.asarray(d)


# ---------------------------------------------------------------------------
# 2. Continuity across the atan2 branch cut (the -X half-plane)
# ---------------------------------------------------------------------------


def test_helix_continuous_across_branch_cut():
    R, pitch, r, n = 10.0, 6.0, 1.2, 4.0
    f, _ = _closure(sdf_helix(R, pitch, r, n))

    eps = 1e-5
    zs = np.linspace(-0.4 * n * pitch, 0.4 * n * pitch, 41)
    # Straddle azimuth = +-pi: y = +eps is a = pi-, y = -eps is a = -pi+.
    above = np.stack([np.full_like(zs, -R), np.full_like(zs, +eps), zs], axis=-1)
    below = np.stack([np.full_like(zs, -R), np.full_like(zs, -eps), zs], axis=-1)
    d_above = np.asarray(f(jnp.asarray(above), _NF))
    d_below = np.asarray(f(jnp.asarray(below), _NF))
    assert np.max(np.abs(d_above - d_below)) < 1e-3, float(np.max(np.abs(d_above - d_below)))


def test_helix_no_seam_in_a_full_azimuth_sweep():
    """Walk a full circle at the winding radius: distance must stay periodic
    and smooth, with no step at -X."""
    R, pitch, r, n = 10.0, 6.0, 1.2, 4.0
    f, _ = _closure(sdf_helix(R, pitch, r, n))
    a = np.linspace(-np.pi, np.pi, 2001)
    pts = np.stack([R * np.cos(a), R * np.sin(a), np.zeros_like(a)], axis=-1)
    d = np.asarray(f(jnp.asarray(pts), _NF))
    # Neighbour-to-neighbour jumps scale with the step size; a seam would show
    # up as an O(pitch) discontinuity.
    assert np.max(np.abs(np.diff(d))) < 0.05, float(np.max(np.abs(np.diff(d))))


# ---------------------------------------------------------------------------
# 3. Clocking: phase is exactly a rotation about Z
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [0.3, 1.0, -2.2])
def test_helix_phase_is_a_z_rotation(delta):
    R, pitch, r, n = 9.0, 5.0, 1.1, 3.0
    f0, _ = _closure(sdf_helix(R, pitch, r, n, phase=0.0))
    fd, _ = _closure(sdf_helix(R, pitch, r, n, phase=delta))

    rng = np.random.default_rng(1)
    pts = rng.uniform(-12.0, 12.0, (300, 3))
    # Rotating the query point by -delta must undo a +delta phase shift.
    c, s = np.cos(-delta), np.sin(-delta)
    rot = np.stack(
        [c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1], pts[:, 2]], axis=-1
    )
    a = np.asarray(fd(jnp.asarray(pts), _NF))
    b = np.asarray(f0(jnp.asarray(rot), _NF))
    assert np.max(np.abs(a - b)) < 1e-4, float(np.max(np.abs(a - b)))


def test_helix_handedness_mirrors_in_z():
    """A left-handed helix is a right-handed one reflected through z=0."""
    R, pitch, r, n = 9.0, 5.0, 1.1, 3.0
    fr, _ = _closure(sdf_helix(R, pitch, r, n, handedness=+1.0))
    fl, _ = _closure(sdf_helix(R, pitch, r, n, handedness=-1.0))
    rng = np.random.default_rng(2)
    pts = rng.uniform(-12.0, 12.0, (300, 3))
    flipped = pts * np.array([1.0, 1.0, -1.0])
    a = np.asarray(fr(jnp.asarray(pts), _NF))
    b = np.asarray(fl(jnp.asarray(flipped), _NF))
    assert np.max(np.abs(a - b)) < 1e-4, float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------------
# 4. Marcher safety, caps, gradients, bbox, round-trip
# ---------------------------------------------------------------------------


def test_helix_is_lipschitz_bounded():
    """|grad| <= 1 is what makes the field safe to ray-march."""
    R, pitch, r, n = 10.0, 6.0, 1.2, 3.0
    f, _ = _closure(sdf_helix(R, pitch, r, n))
    rng = np.random.default_rng(3)
    pts = rng.uniform(-14.0, 14.0, (500, 3))
    h = 1e-3
    g = []
    for axis in range(3):
        off = np.zeros(3)
        off[axis] = h
        d_plus = np.asarray(f(jnp.asarray(pts + off), _NF))
        d_minus = np.asarray(f(jnp.asarray(pts - off), _NF))
        g.append((d_plus - d_minus) / (2 * h))
    mag = np.linalg.norm(np.stack(g, axis=-1), axis=-1)
    # Slack for the C0 crease midway between turns, where a central difference
    # straddles two branches of the nearest-turn min.
    assert np.percentile(mag, 99.0) <= 1.02, float(np.percentile(mag, 99.0))


def test_helix_has_flat_end_caps():
    R, pitch, r, n = 8.0, 5.0, 1.0, 2.0
    half_h = 0.5 * n * pitch
    f, _ = _closure(sdf_helix(R, pitch, r, n))
    # Just outside the band, directly above the winding radius: the slab term
    # dominates, so distance is the axial overshoot regardless of azimuth.
    for a in (0.0, 1.0, np.pi, -2.0):
        p = jnp.asarray([[R * np.cos(a), R * np.sin(a), half_h + 2.0]])
        assert float(f(p, _NF)[0]) == pytest.approx(2.0, abs=0.05)


def test_helix_gradient_finite_on_axis():
    """The azimuth JVP guard must keep grads finite on the Z axis."""
    g = jax.grad(lambda p: helix(p, 10.0, 6.0, 1.2, 3.0))(jnp.zeros(3))
    assert bool(jnp.all(jnp.isfinite(g))), g


def test_helix_gradient_flows_to_params():
    """Autodiff reaches pitch / major_r / phase: the point of a primitive
    rather than a baked control-point path."""

    def vol_proxy(major_r, pitch, phase):
        pts = jnp.asarray([[9.0, 0.5, 1.0], [10.0, -1.0, -2.0], [-9.0, 0.2, 3.0]])
        return jnp.sum(helix(pts, major_r, pitch, 1.2, 3.0, phase))

    g = jax.grad(vol_proxy, argnums=(0, 1, 2))(10.0, 6.0, 0.0)
    assert all(bool(jnp.isfinite(x)) for x in g), g
    assert any(abs(float(x)) > 1e-6 for x in g), g


def test_helix_bbox_is_tight_and_contains_the_surface():
    R, pitch, r, n = 10.0, 6.0, 1.2, 3.0
    tree = sdf_helix(R, pitch, r, n)
    lo, hi = infer_sdf_bbox(tree, _part(tree))
    assert lo == pytest.approx((-(R + r), -(R + r), -(0.5 * n * pitch + r)))
    assert hi == pytest.approx((R + r, R + r, 0.5 * n * pitch + r))

    # Nothing solid outside it: the field is positive on the bbox faces.
    f, _ = _closure(tree)
    rng = np.random.default_rng(4)
    u = rng.uniform(-1.0, 1.0, (200, 2))
    faces = np.concatenate(
        [
            np.stack([np.full(200, hi[0]), u[:, 0] * hi[1], u[:, 1] * hi[2]], axis=-1),
            np.stack([u[:, 0] * hi[0], u[:, 1] * hi[1], np.full(200, hi[2])], axis=-1),
        ]
    )
    assert np.all(np.asarray(f(jnp.asarray(faces), _NF)) > -1e-6)


def test_helix_param_refs_and_roundtrip(tmp_path):
    tree = sdf_helix(
        make_param_ref("major_r"), make_param_ref("pitch"), 1.2, 3.0, phase=make_param_ref("clock")
    )
    part = _part(
        tree,
        major_r=Param("major_r", 10.0, free=True, bounds=(6.0, 14.0), unit="mm"),
        pitch=Param("pitch", 6.0, free=True, bounds=(3.0, 9.0), unit="mm"),
        clock=Param("clock", 0.0, free=False, unit="rad"),
    )
    out = tmp_path / "helix.sdm"
    io.save(part, out)
    back = io.load(out)
    f = make_sdf_closure(back.materials[0].sdf_tree, back)
    d = f(jnp.asarray([[10.0, 0.0, 0.0]]), jnp.asarray(back.param_vector()))
    assert float(d[0]) == pytest.approx(-1.2 / _k_wall(10.0, 6.0, 1.2), abs=2e-3)


def test_helix_emits_glsl():
    """The viewer path must know the primitive too, and lib.glsl must declare
    the function the emitter calls."""
    from software_defined_matter.glsl import emit_glsl, load_lib_glsl

    em = emit_glsl(_part(sdf_helix(10.0, 6.0, 1.2, 3.0)))
    assert "sdf_helix(" in em.scene_source
    assert "float sdf_helix(" in load_lib_glsl()
