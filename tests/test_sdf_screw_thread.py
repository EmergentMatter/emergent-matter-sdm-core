"""Tests for the ``screw_thread`` primitive — a regular truncated-V thread.

The headline test is :func:`test_axial_tooth_thickness_is_the_declared_width`.
The construction this primitive replaces (a box twisted into a helix, as used by
``em_cad.fasteners`` and by an early ``prim-threads`` profile) puts the tooth's
thickness in the TANGENTIAL direction, so its true axial thickness degrades to
``width / (k * r)`` — 0.02 mm on a 58 mm thread. It looks like a thread in a
thumbnail and is a razor-thin spiral ramp in fact. That test is what catches it.
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
    sdf_screw_thread,
)
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.sdf_shapes import screw_thread

_NF = jnp.zeros((0,))


def _part(tree, **params):
    return Part(
        name="x", params=params, materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)]
    )


def _f(tree, **params):
    part = _part(tree, **params)
    closure = make_sdf_closure(tree, part)
    return lambda pts: np.asarray(closure(jnp.asarray(np.atleast_2d(pts)), _NF))


# The spherical bearing's cap thread, and a small one for contrast.
CAP = {"r_root": 27.0, "depth": 1.84, "pitch": 3.0, "width": 2.6, "n_turns": 4.0}
M6 = {"r_root": 2.4, "depth": 0.61, "pitch": 1.0, "width": 0.87, "n_turns": 6.0}


def _axial_extent(f, r, az_deg=0.0, span=None, n=4001):
    """Total axial extent of material at a given radius/azimuth, within one
    pitch either side of z=0."""
    a = math.radians(az_deg)
    zs = np.linspace(-span, span, n)
    pts = np.stack(
        [np.full_like(zs, r * math.cos(a)), np.full_like(zs, r * math.sin(a)), zs], axis=-1
    )
    inside = f(pts) < 0.0
    if not inside.any():
        return 0.0, None
    # Width of the interval containing the sample nearest z=0.
    idx = np.flatnonzero(inside)
    mid = idx[np.argmin(np.abs(zs[idx]))]
    lo = hi = mid
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    while hi < len(zs) - 1 and inside[hi + 1]:
        hi += 1
    return float(zs[hi] - zs[lo]), float(0.5 * (zs[hi] + zs[lo]))


# ---------------------------------------------------------------------------
# THE test: the tooth has real axial thickness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw,name", [(CAP, "cap58"), (M6, "m6")])
def test_axial_tooth_thickness_is_the_declared_width(kw, name):
    """At the root the tooth must be `width` thick MEASURED ALONG THE AXIS.

    This is the assertion the twisted-box construction fails by two orders of
    magnitude, and it is the whole reason this primitive exists.
    """
    f = _f(sdf_screw_thread(**kw))
    # Just above the root, where the profile is at full width.
    w, _ = _axial_extent(f, kw["r_root"] + 0.02, span=kw["pitch"])
    assert w == pytest.approx(kw["width"], rel=0.06), (
        f"{name}: axial tooth thickness {w:.4f} mm, declared {kw['width']} mm"
    )


@pytest.mark.parametrize("az", [0.0, 37.0, 123.0, -95.0])
def test_thickness_holds_at_every_azimuth(az):
    """A spiral ramp would thin out away from its start azimuth."""
    f = _f(sdf_screw_thread(**CAP))
    w, _ = _axial_extent(f, CAP["r_root"] + 0.02, az_deg=az, span=CAP["pitch"])
    assert w == pytest.approx(CAP["width"], rel=0.06), f"az={az}: {w:.4f} mm"


# ---------------------------------------------------------------------------
# The V profile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flank_deg", [60.0, 55.0, 29.0])
def test_flank_angle_matches_the_spec(flank_deg):
    """Measure the included angle from how fast the tooth narrows with radius.

    half_width(s) = width/2 - s*tan(flank/2), so the slope gives the angle back.
    60 = ISO metric, 55 = Whitworth, 29 = ACME.
    """
    kw = dict(CAP, flank_deg=flank_deg)
    f = _f(sdf_screw_thread(**kw))
    ss = np.linspace(0.05, kw["depth"] * 0.8, 8)
    widths = [_axial_extent(f, kw["r_root"] + s, span=kw["pitch"])[0] for s in ss]
    slope = np.polyfit(ss, [w / 2.0 for w in widths], 1)[0]
    got = 2.0 * math.degrees(math.atan(abs(slope)))
    assert got == pytest.approx(flank_deg, abs=2.5), (
        f"included angle {got:.1f} deg, want {flank_deg}"
    )


def test_tooth_spans_root_to_crest_and_no_further():
    f = _f(sdf_screw_thread(**CAP))
    r_crest = CAP["r_root"] + CAP["depth"]
    assert _axial_extent(f, CAP["r_root"] + 0.05, span=CAP["pitch"])[0] > 0.0
    assert _axial_extent(f, r_crest - 0.05, span=CAP["pitch"])[0] > 0.0
    assert _axial_extent(f, r_crest + 0.1, span=CAP["pitch"])[0] == 0.0
    # Below the root there is no ridge (the core cylinder is the caller's job).
    assert _axial_extent(f, CAP["r_root"] - 0.1, span=CAP["pitch"])[0] == 0.0


def test_crest_is_truncated_not_a_feather_edge():
    """Real threads truncate the sharp V; a printed feather edge crumbles."""
    f = _f(sdf_screw_thread(**CAP))
    w, _ = _axial_extent(f, CAP["r_root"] + CAP["depth"] - 0.03, span=CAP["pitch"])
    assert w > 0.2, f"crest width {w:.3f} mm — effectively a feather edge"


def test_teeth_repeat_at_the_pitch():
    """Consecutive teeth at one azimuth sit exactly `pitch` apart."""
    f = _f(sdf_screw_thread(**CAP))
    r = CAP["r_root"] + 0.02
    # Stay clear of the band ends: the teeth there are cut flat and half-width,
    # so their midpoints sit inboard and would read as a short pitch.
    half_h = 0.5 * CAP["n_turns"] * CAP["pitch"]
    lim = half_h - 0.55 * CAP["pitch"]
    zs = np.linspace(-lim, lim, 8801)
    pts = np.stack([np.full_like(zs, r), np.zeros_like(zs), zs], axis=-1)
    inside = f(pts) < 0.0
    # Interval midpoints.
    edges = np.flatnonzero(np.diff(inside.astype(int)) != 0)
    mids = [(zs[edges[i]] + zs[edges[i + 1]]) / 2.0 for i in range(0, len(edges) - 1, 2)]
    assert len(mids) >= 3, f"only {len(mids)} teeth found"
    spacing = np.diff(mids)
    assert np.allclose(spacing, CAP["pitch"], atol=0.02), spacing


# ---------------------------------------------------------------------------
# Helical behaviour, clocking, handedness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_hand", [1.0, -1.0])
def test_tooth_advances_one_pitch_per_turn(n_hand):
    """Follow one tooth around: its z must climb by `pitch` per revolution, in
    the direction the handedness says."""
    f = _f(sdf_screw_thread(**CAP, handedness=n_hand))
    r = CAP["r_root"] + 0.02
    zs_at = []
    for az in (0.0, 90.0, 180.0, 270.0):
        # Track the tooth near the expected height so we follow ONE tooth.
        expect = n_hand * CAP["pitch"] * az / 360.0
        a = math.radians(az)
        zs = np.linspace(expect - 1.2, expect + 1.2, 2401)
        pts = np.stack(
            [np.full_like(zs, r * math.cos(a)), np.full_like(zs, r * math.sin(a)), zs], axis=-1
        )
        inside = f(pts) < 0.0
        assert inside.any(), f"no tooth at az={az}"
        zs_at.append(float(zs[inside].mean()))
    climb = np.diff(zs_at)
    assert np.allclose(climb, n_hand * CAP["pitch"] / 4.0, atol=0.08), climb


@pytest.mark.parametrize("delta", [0.3, -1.1])
def test_phase_is_a_pure_z_rotation(delta):
    """Clocking: what a threaded cap registers its ports against."""
    f0 = _f(sdf_screw_thread(**CAP))
    fd = _f(sdf_screw_thread(**CAP, phase=delta))
    rng = np.random.default_rng(0)
    pts = np.stack(
        [rng.uniform(-31, 31, 400), rng.uniform(-31, 31, 400), rng.uniform(-5, 5, 400)], axis=-1
    )
    c, s = math.cos(-delta), math.sin(-delta)
    rot = np.stack(
        [c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1], pts[:, 2]], axis=-1
    )
    assert np.max(np.abs(fd(pts) - f0(rot))) < 1e-4


def test_handedness_mirrors_in_z():
    fr = _f(sdf_screw_thread(**CAP, handedness=1.0))
    fl = _f(sdf_screw_thread(**CAP, handedness=-1.0))
    rng = np.random.default_rng(1)
    pts = np.stack(
        [rng.uniform(-31, 31, 400), rng.uniform(-31, 31, 400), rng.uniform(-5, 5, 400)], axis=-1
    )
    assert np.max(np.abs(fr(pts) - fl(pts * np.array([1.0, 1.0, -1.0])))) < 1e-4
    assert np.max(np.abs(fr(pts) - fl(pts))) > 0.05


def test_continuous_across_the_branch_cut():
    f = _f(sdf_screw_thread(**CAP))
    eps = 1e-5
    zs = np.linspace(-4.0, 4.0, 81)
    R = CAP["r_root"] + 0.5
    above = np.stack([np.full_like(zs, -R), np.full_like(zs, +eps), zs], axis=-1)
    below = np.stack([np.full_like(zs, -R), np.full_like(zs, -eps), zs], axis=-1)
    assert np.max(np.abs(f(above) - f(below))) < 1e-3


# ---------------------------------------------------------------------------
# Marcher safety, bbox, plumbing
# ---------------------------------------------------------------------------


def test_is_lipschitz_bounded():
    f = _f(sdf_screw_thread(**CAP))
    rng = np.random.default_rng(2)
    pts = np.stack(
        [rng.uniform(-33, 33, 500), rng.uniform(-33, 33, 500), rng.uniform(-8, 8, 500)], axis=-1
    )
    h = 1e-3
    g = []
    for axis in range(3):
        off = np.zeros(3)
        off[axis] = h
        g.append((f(pts + off) - f(pts - off)) / (2 * h))
    mag = np.linalg.norm(np.stack(g, axis=-1), axis=-1)
    assert np.percentile(mag, 99.0) <= 1.05, float(np.percentile(mag, 99.0))


def test_gradient_finite_on_axis():
    g = jax.grad(lambda p: screw_thread(p, 27.0, 1.84, 3.0, 2.6, 4.0))(jnp.zeros(3))
    assert bool(jnp.all(jnp.isfinite(g))), g


def test_bbox_is_tight():
    tree = sdf_screw_thread(**CAP)
    lo, hi = infer_sdf_bbox(tree, _part(tree))
    radial = CAP["r_root"] + CAP["depth"]
    half_h = 0.5 * CAP["n_turns"] * CAP["pitch"]
    assert lo == pytest.approx((-radial, -radial, -half_h))
    assert hi == pytest.approx((radial, radial, half_h))


def test_param_refs_and_roundtrip(tmp_path):
    tree = sdf_screw_thread(
        r_root=make_param_ref("r_root"),
        depth=1.84,
        pitch=3.0,
        width=2.6,
        n_turns=4.0,
        phase=make_param_ref("clock"),
    )
    part = _part(
        tree,
        r_root=Param("r_root", 27.0, free=True, bounds=(20.0, 32.0), unit="mm"),
        clock=Param("clock", 0.0, free=False, unit="rad"),
    )
    out = tmp_path / "t.sdm"
    io.save(part, out)
    back = io.load(out)
    f = make_sdf_closure(back.materials[0].sdf_tree, back)
    d = f(jnp.asarray([[27.02, 0.0, 0.0]]), jnp.asarray(back.param_vector()))
    assert float(d[0]) < 0.0  # inside the tooth at its root, azimuth 0


def test_emits_glsl():
    from software_defined_matter.glsl import emit_glsl, load_lib_glsl

    em = emit_glsl(_part(sdf_screw_thread(**CAP)))
    assert "sdf_screw_thread(" in em.scene_source
    assert "float sdf_screw_thread(" in load_lib_glsl()
