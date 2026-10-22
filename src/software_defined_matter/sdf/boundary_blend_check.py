"""Checks on a boundary blend, in numbers a person can read.

boundary_error  distance (mm) between the patch's edges and the INPUT curves
side_report     along each boundary with a condition: how far the surface's
                cross-direction is from the one asked for (deg); for "tangent",
                also how far the surface normal is from the plane's (deg)
corner_report   wherever a curve lands on a boundary with a condition (corners
                and guide ends): how far that curve's direction is from what
                the condition asks for (deg). Big numbers = data disagree.
guide_report    every curve of the network vs the surface (mm), and the
                crossing / landing gaps found when building
quality_report  whether the surface folds anywhere (on the patch and on its
                extensions), the tightest curvature radius, and the above
                rolled up; `ok` is False if it folds

Angles involving a zero vector are reported as NaN (undefined), never 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from software_defined_matter.sdf.boundary_blend_patch import Patch, _bez_np, _closest_on


def _ang(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


_SIDES = {  # side -> (which partial, fixed param name, fixed value)
    "u0": (0, "u", 0.0),
    "u1": (0, "u", 1.0),
    "v0": (1, "v", 0.0),
    "v1": (1, "v", 1.0),
}


def _mode(patch, side):
    k = patch.side_of[side]
    return None if k is None else patch.modes[k]


def boundary_error(patch: Patch, curves, n=41):
    """Largest distance (mm) between the patch's edges and the input curves,
    both ways (edge samples to curve, curve samples to edge)."""
    curves = [np.asarray(c, float) for c in curves]
    edges = [
        lambda t: patch.S(t, 0.0),
        lambda t: patch.S(t, 1.0),
        lambda t: patch.S(0.0, t),
        lambda t: patch.S(1.0, t),
    ]
    ts = np.linspace(0, 1, n)
    edge_pts = [np.array([np.asarray(e(float(t))) for t in ts]) for e in edges]
    worst = 0.0
    for c in curves:
        if c.shape[0] == 1:
            continue
        cp = _bez_np(c, ts)
        # the edge this curve became: the one closest to it overall
        best = min(
            edge_pts,
            key=lambda E: np.mean(np.min(np.linalg.norm(E[:, None] - cp[None], axis=-1), 0)),
        )
        a = max(_closest_on(c, q)[1] for q in best)
        b = np.max(np.min(np.linalg.norm(best[:, None] - cp[None], axis=-1), 0))
        worst = max(worst, a, float(b))
    return worst


def side_report(patch: Patch, n=41):
    """Worst angle (deg) between the surface's cross-direction on each
    conditioned side and the direction asked for; for "tangent" sides also
    the worst angle between the surface normal and the plane normal. Free
    sides are not reported (there is nothing asked of them)."""
    J = jax.jit(jax.vmap(jax.jacfwd(patch.S, argnums=(0, 1))))
    ts = np.linspace(0.02, 0.98, n)
    out = {}
    for side, (k, which, val) in _SIDES.items():
        idx = patch.side_of[side]
        mode = _mode(patch, side)
        if idx is None or mode == "free":
            continue
        fixed = np.full_like(ts, val)
        u, v = (fixed, ts) if which == "u" else (ts, fixed)
        ju, jv = (np.asarray(x) for x in J(jnp.asarray(u), jnp.asarray(v)))
        got = ju if k == 0 else jv
        want = np.asarray(jax.vmap(patch.fields[side])(jnp.asarray(ts)))
        worst = float(np.nanmax([_ang(g, w) for g, w in zip(got, want, strict=True)]))
        worst_n = 0.0
        if mode == "tangent":
            angs = [_ang(c, patch.vecs[idx]) for c in np.cross(ju, jv)]
            worst_n = float(np.nanmax([min(a, 180.0 - a) for a in angs]))
        out[f"curve {idx + 1}"] = float(worst)
        if mode == "tangent":
            out[f"curve {idx + 1} normal vs plane"] = float(worst_n)
    return out


def corner_report(patch: Patch):
    """At every point where a curve lands on a conditioned boundary (corners
    and guide ends), the angle between that curve's direction there and the
    direction the condition asks for."""
    out = {}
    for side, (kn, A, sgn) in patch.targets.items():
        idx = patch.side_of[side]
        if idx is None or _mode(patch, side) == "free":
            continue
        for t, a in zip(kn, A, strict=True):
            have = sgn * np.asarray(a)
            if np.linalg.norm(have) < 1e-9:
                continue  # meets the pole: nothing to agree with
            want = np.asarray(patch.inward[side](float(t)))
            out[f"curve {idx + 1} at t={float(t):.3f}"] = _ang(have, want)
    return out


def guide_report(patch: Patch, n=41):
    """For each curve of the network (boundaries and guides), the largest
    distance (mm) between it and the surface's matching line; plus how
    closely the guides crossed and landed (from building the patch)."""
    out = {}
    ts = jnp.linspace(0.0, 1.0, n)
    S = jax.jit(jax.vmap(patch.S))
    for j, (f, vj) in enumerate(zip(patch.Uf, patch.vk, strict=True)):
        d = S(ts, jnp.full_like(ts, vj)) - jax.vmap(f)(ts)
        out[f"u-direction curve {j} (v={float(vj):.3f})"] = float(jnp.linalg.norm(d, axis=1).max())
    for i, (f, ui) in enumerate(zip(patch.Vf, patch.uk, strict=True)):
        d = S(jnp.full_like(ts, ui), ts) - jax.vmap(f)(ts)
        out[f"v-direction curve {i} (u={float(ui):.3f})"] = float(jnp.linalg.norm(d, axis=1).max())
    gaps = dict(patch.crossings)
    return out, gaps


def _normals(patch, reach, U, V):
    def nrm(u, v):
        vv = v
        if patch.pole:
            vv = jnp.where(jnp.abs(v) < 1e-3, jnp.where(v < 0, -1e-3, 1e-3), v)
        su, sv = jax.jacfwd(lambda a, b: patch.S_ext(a, b, reach), argnums=(0, 1))(u, vv)
        c = patch.orient * jnp.cross(su, sv)
        if patch.pole:
            c = jnp.where(vv < 0, -c, c)
        return c

    return np.asarray(
        jax.jit(jax.vmap(nrm))(jnp.asarray(U, dtype=float), jnp.asarray(V, dtype=float))
    )


def _curvature(patch, u, v):
    """Principal curvatures (1/mm) at (u, v), from the two fundamental forms."""
    f = lambda a: patch.S(a[0], a[1])  # noqa: E731
    x = jnp.array([u, v])
    J = jax.jacfwd(f)(x)  # (3, 2)
    H = jax.hessian(f)(x)  # (3, 2, 2)
    su, sv = J[:, 0], J[:, 1]
    n = jnp.cross(su, sv)
    n = n / jnp.linalg.norm(n)
    E, F, G = su @ su, su @ sv, sv @ sv
    L, M, N = n @ H[:, 0, 0], n @ H[:, 0, 1], n @ H[:, 1, 1]
    den = E * G - F * F
    K = (L * N - M * M) / den
    Hm = (E * N - 2 * F * M + G * L) / (2 * den)
    r = jnp.sqrt(jnp.maximum(Hm * Hm - K, 0.0))
    return Hm + r, Hm - r


def quality_report(patch: Patch, reach, n=33):
    """Does the surface fold anywhere, on the patch or on its extensions?

    A fold shows up as the normal turning over between neighbouring samples,
    or the surface pinching to nothing (|S_u x S_v| -> 0) away from a pole.
    Both make the inside/outside test meaningless there. Also reports the
    tightest curvature radius on the patch itself (G1 only is promised: no
    curvature continuity, fairness or draft checks are made)."""
    problems = []
    # Extended domain, matching what the distance solver searches.
    us = np.linspace(-1.0, 2.0, 3 * n)
    vs = np.linspace(-1.0, 2.0, 3 * n)
    UU, VV = np.meshgrid(us, vs, indexing="ij")
    keep = np.ones_like(UU, bool)
    if patch.pole:
        keep &= np.abs(VV) > 0.03  # the pole itself
        keep &= ~((VV < 0) & ((UU < 0) | (UU > 1)))  # degenerate past-pole corners
    Nv = _normals(patch, float(reach), UU.ravel(), VV.ravel()).reshape(UU.shape + (3,))
    mag = np.linalg.norm(Nv, axis=-1)
    inside = keep & (UU >= 0) & (UU <= 1) & (VV >= 0) & (VV <= 1)
    if patch.pole:
        inside &= VV > 0.03
    scale = np.median(mag[inside])
    pinch = inside & (mag < 1e-3 * scale)
    if pinch.any():
        i, j = np.argwhere(pinch)[0]
        problems.append(f"surface pinches to nothing at (u, v) = ({UU[i, j]:.2f}, {VV[i, j]:.2f})")
    Nn = Nv / np.maximum(mag[..., None], 1e-20)
    flips = 0
    where = (float("nan"), float("nan"))
    for axis in (0, 1):
        a = np.take(Nn, range(Nn.shape[axis] - 1), axis=axis)
        b = np.take(Nn, range(1, Nn.shape[axis]), axis=axis)
        ka = np.take(keep, range(keep.shape[axis] - 1), axis=axis)
        kb = np.take(keep, range(1, keep.shape[axis]), axis=axis)
        bad = (np.sum(a * b, axis=-1) < 0) & ka & kb
        flips += int(bad.sum())
        if bad.any() and flips == int(bad.sum()):
            i, j = np.argwhere(bad)[0]
            where = (UU[i, j], VV[i, j])
    if flips:
        problems.append(
            f"surface turns over (folds) in {flips} places, first near "
            f"(u, v) = ({where[0]:.2f}, {where[1]:.2f})"
            + ("" if 0 <= where[0] <= 1 and 0 <= where[1] <= 1 else " on the extension")
        )
    # Curvature on the patch.
    cu, cv = np.meshgrid(
        np.linspace(0.02, 0.98, 15), np.linspace(0.05 if patch.pole else 0.02, 0.98, 15)
    )
    curv = jax.jit(jax.vmap(lambda a, b: jnp.stack(_curvature(patch, a, b))))
    k = np.asarray(curv(jnp.asarray(cu.ravel()), jnp.asarray(cv.ravel()))).ravel()
    kmax = np.nanmax(np.abs(k))
    sides = side_report(patch)
    corners = corner_report(patch)
    return {
        "ok": not problems,
        "problems": problems,
        "tightest_radius_mm": float(1.0 / kmax) if kmax > 0 else float("inf"),
        "worst_condition_deg": float(np.nanmax(list(sides.values()) or [0.0])),
        "worst_landing_deg": float(np.nanmax(list(corners.values()) or [0.0])),
        "worst_crossing_gap_mm": float(max([g for _, g in patch.crossings] or [0.0])),
    }
