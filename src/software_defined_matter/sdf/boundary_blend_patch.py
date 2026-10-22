"""The surface: a boundary blend through a loop of boundary curves plus any
number of GUIDE curves across it in both directions, with a condition
(free / normal / tangent) on each boundary, exact at the corners.

This is Creo's boundary blend. With no guides it is a Coons patch; with
guides it is a Gordon surface (the curve-network generalisation of Coons).

CURVES. Each curve is a Bezier curve given by its control points. Point 0
is where it starts, the last point where it ends, and the first and last legs
of the control polygon set the end tangents (as in any CAD spline).

THE LOOP. The boundary curves run head to tail round the patch: curve 1 ends
where curve 2 starts, and so on, and the last ends where curve 1 starts.

  - 4 curves: an ordinary four-sided patch.
  - 3 curves: a four-sided patch with one side shrunk to a point, the POLE,
    which is the start of curve 1. Put the pole where the surface is
    smoothest (a soap bar's crown), never at a sharp corner.

INSIDE THE PATCH the surface S(u, v) is laid out on a unit square. With the
sides named by which parameter is fixed:

    v = 0 : c0(u)      v = 1 : c1(u)      u = 0 : d0(v)      u = 1 : d1(v)

    4 curves:  c0 = curve 1,  d1 = curve 2,  c1 = curve 3 reversed,
               d0 = curve 4 reversed
    3 curves:  d0 = curve 1,  c1 = curve 2,  d1 = curve 3 reversed,
               c0 = the pole (a point)

GUIDES. A guide runs across the patch from one boundary to the one facing it:
either d0 -> d1 (a "u guide", alongside c0 and c1) or c0 -> c1 (a "v guide",
alongside d0 and d1; in a 3-curve loop these start at the pole). Give them in
any order and either way round; each is sorted out from where its ends land.
u guides and v guides must cross each other (within `GAP_TOL`, reported by
`guide_report`). A guide's ends set the cross-direction of the boundaries it
lands on, so a guide must agree with those boundaries' conditions, the same
way the neighbouring boundary curves must at a corner.

HOW IT IS BUILT (Gordon). Where the curves cross, each curve is re-timed
(a smooth monotone change of its parameter, not of its shape) so every u
curve passes the k-th v curve at the same u, and vice versa. Then

    S = (spline in v through all u-direction curves)
      + (spline in u through all v-direction curves)
      - (spline in u and v through the crossing points)

The splines are cubic, clamped at the outer curves to the cross-directions
the conditions ask for; Gregory's twist keeps the corners exact.

CONDITIONS, one per boundary curve, say which way the surface leaves it:

  - "free":    no condition; the cross-direction is blended from the curves
               that land on it (neighbouring boundaries and guides).
  - "normal":  the surface leaves the curve along a fixed direction `vec`.
               For "normal to a plane": a symmetry plane (vec = its normal) or
               a vertical wall below the curve (vec = Z).
  - "tangent": the surface is tangent to a plane with normal `vec`, leaving
               the curve square to it within that plane.

The sense of `vec` does not matter: it is turned to point into the patch.
`weight` scales how far a condition reaches into the patch between the
points where other curves land on it (Creo's tangent stretch). 1 is neutral.

CONSISTENCY. A condition and every curve landing on its boundary must agree
there: if the patch must leave curve 2 straight down, curve 1 (and any guide
ending on curve 2) must arrive going straight down. `check.corner_report`
measures any mismatch. Where they disagree the curves are still matched
exactly; only the tangency near that point is off.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

MODES = ("free", "normal", "tangent")
GAP_TOL = 1e-3  # mm: guides must cross / land this close (the disagreement is averaged)
_EPS = 1e-9


# --------------------------------------------------------------------- curves
def bezier(ctrl, t):
    """Point on the Bezier curve with control points `ctrl` (K, 3) at t.
    De Casteljau, so any degree; K = 1 is a fixed point."""
    pts = ctrl
    for _ in range(ctrl.shape[0] - 1):
        pts = (1.0 - t) * pts[:-1] + t * pts[1:]
    return pts[0]


def _bez_np(ctrl, t):
    """numpy Bezier at many t: (T,) -> (T, 3)."""
    t = np.asarray(t, float)[:, None, None]
    pts = np.broadcast_to(ctrl, (t.shape[0],) + ctrl.shape).copy()
    for _ in range(ctrl.shape[0] - 1):
        pts = (1.0 - t) * pts[:, :-1] + t * pts[:, 1:]
    return pts[:, 0]


def _closest_on(ctrl, q, n=2001):
    """Parameter on a Bezier nearest point q, and the distance."""
    lo, hi = 0.0, 1.0
    for _ in range(4):
        ts = np.linspace(lo, hi, n)
        d = np.linalg.norm(_bez_np(ctrl, ts) - q, axis=1)
        k = int(np.argmin(d))
        w = (hi - lo) / (n - 1) * 2
        lo, hi = max(0.0, ts[k] - w), min(1.0, ts[k] + w)
    return float(ts[k]), float(d[k])


def _crossing(ca, cb, n=801):
    """Where two Bezier curves come closest: (t on a, t on b, gap)."""
    ta = np.linspace(0, 1, n)
    tb = np.linspace(0, 1, n)
    A, B = _bez_np(ca, ta), _bez_np(cb, tb)
    d = np.linalg.norm(A[:, None] - B[None], axis=-1)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    a, b = ta[i], tb[j]
    for _ in range(30):  # alternate projections to refine
        b, _ = _closest_on(cb, _bez_np(ca, [a])[0], 201)
        a, _ = _closest_on(ca, _bez_np(cb, [b])[0], 201)
    gap = float(np.linalg.norm(_bez_np(ca, [a])[0] - _bez_np(cb, [b])[0]))
    return float(a), float(b), gap


def _d(f):
    return jax.jacfwd(f)


def _unit(x):
    # smooth length: a zero vector (S_u at a pole) must not give NaN slopes
    return x / jnp.sqrt(jnp.sum(x * x) + _EPS * _EPS)


# -------------------------------------------------------- 1-D interpolation
def _herm(t):
    """Cubic Hermite blending: h0, h1 carry position, g0, g1 carry slope."""
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1, -2 * t3 + 3 * t2, t3 - 2 * t2 + t, t3 - t2)


def _bump(t, w):
    """1 at both ends, `w` in the middle: weight acts away from the knots."""
    return 1.0 + (w - 1.0) * 4.0 * t * (1.0 - t)


def _segment(knots, t):
    """Index of the knot span holding t, and t's place (0..1) in it."""
    kn = jnp.asarray(knots, dtype=float)
    i = jnp.clip(jnp.searchsorted(kn, t, side="right") - 1, 0, len(knots) - 2)
    h = kn[i + 1] - kn[i]
    return i, (t - kn[i]) / h, h


def _hermite_through(knots, Y, S, t):
    """Piecewise cubic through values Y (K, ...) with slopes S at `knots`."""
    i, s, h = _segment(knots, t)
    h0, h1, g0, g1 = _herm(s)
    return h0 * Y[i] + h1 * Y[i + 1] + h * (g0 * S[i] + g1 * S[i + 1])


def _clamped_matrix(knots):
    """M with slopes = M @ [y_0..y_K-1, s_0, s_K-1]: the clamped cubic spline
    through K values with the two end slopes given (C2 at interior knots).
    JAX, so it follows the knots when they move (K is static)."""
    x = jnp.asarray(knots, dtype=float)
    K = x.shape[0]
    top = jnp.zeros((1, K + 2)).at[0, K].set(1.0)
    bot = jnp.zeros((1, K + 2)).at[0, K + 1].set(1.0)
    if K == 2:
        return jnp.concatenate([top, bot], axis=0)
    h = jnp.diff(x)
    A = jnp.zeros((K - 2, K - 2))
    R = jnp.zeros((K - 2, K + 2))
    for r, k in enumerate(range(1, K - 1)):
        hl, hr = h[k - 1], h[k]
        A = A.at[r, r].set(2 * (hl + hr))
        if r > 0:
            A = A.at[r, r - 1].set(hr)
        if r < K - 3:
            A = A.at[r, r + 1].set(hl)
        # 3*[(y_k+1 - y_k) hl/hr + (y_k - y_k-1) hr/hl]
        R = R.at[r, k + 1].add(3 * hl / hr)
        R = R.at[r, k].add(-3 * hl / hr + 3 * hr / hl)
        R = R.at[r, k - 1].add(-3 * hr / hl)
        if k == 1:
            R = R.at[r, K].add(-hr)  # known slope s_0 moves across
        if k == K - 2:
            R = R.at[r, K + 1].add(-hl)  # known slope s_K-1
    return jnp.concatenate([top, jnp.linalg.solve(A, R), bot], axis=0)


def _spline(knots, M, Y, s0, s1, t):
    """Clamped cubic spline through Y at knots, end slopes s0, s1, at t."""
    data = jnp.concatenate([jnp.stack(Y), jnp.stack([s0, s1])], axis=0)
    S = jnp.tensordot(M, data, axes=1)
    return _hermite_through(knots, jnp.stack(Y), S, t)


def _pchip_slopes(x, y):
    """Monotone (Fritsch-Carlson) slopes for an increasing y: a re-timing that
    never runs back. JAX; y is increasing by construction (checked in the
    plan), so every interior slope takes the harmonic-mean branch."""
    h = jnp.diff(x)
    dl = jnp.maximum(jnp.diff(y) / h, 1e-9)
    m = [dl[0]]
    for k in range(1, x.shape[0] - 1):
        w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
        m.append((w1 + w2) / (w1 / dl[k - 1] + w2 / dl[k]))
    m.append(dl[-1])
    return jnp.stack(m)


class _Retime:
    """Smooth monotone change of a curve's parameter so it passes given
    points (curve params `t_at`) at the shared params `s_at`. Identity when
    there is nothing between the ends."""

    def __init__(self, s_at, t_at):
        z, o = jnp.zeros((1,)), jnp.ones((1,))
        self.n = 2 + len(s_at)
        if self.n == 2:
            return
        self.s = jnp.concatenate([z, jnp.stack(list(s_at)), o])
        self.t = jnp.concatenate([z, jnp.stack(list(t_at)), o])
        self.m = _pchip_slopes(self.s, self.t)

    def __call__(self, s):
        if self.n == 2:
            return s
        return _hermite_through(self.s, self.t, self.m, s)


def _refine_on(ctrl, q, s0, iters=8):
    """Parameter on a Bezier nearest point q, refined from s0 (Gauss-Newton,
    JAX: it moves smoothly with the curve and q)."""
    s = jnp.asarray(s0, dtype=float)
    f = lambda t: bezier(ctrl, t)  # noqa: E731
    df = jax.jacfwd(f)
    for _ in range(iters):
        r, d = f(s) - q, df(s)
        s = jnp.clip(s - jnp.dot(r, d) / (jnp.dot(d, d) + 1e-12), 0.0, 1.0)
    return s


def _refine_cross(ca, cb, a0, b0, iters=8):
    """Where two Bezier curves cross, refined from (a0, b0) (Gauss-Newton, JAX)."""
    fa = lambda t: bezier(ca, t)  # noqa: E731
    fb = lambda t: bezier(cb, t)  # noqa: E731
    da, db = jax.jacfwd(fa), jax.jacfwd(fb)
    a, b = jnp.asarray(a0, dtype=float), jnp.asarray(b0, dtype=float)
    for _ in range(iters):
        r = fa(a) - fb(b)
        J = jnp.stack([da(a), -db(b)], axis=-1)
        step = jnp.linalg.solve(J.T @ J + 1e-12 * jnp.eye(2), J.T @ r)
        a, b = jnp.clip(a - step[0], 0.0, 1.0), jnp.clip(b - step[1], 0.0, 1.0)
    return a, b


def _bez_tangent_np(ctrl, t):
    """numpy Bezier derivative at many t (the hodograph)."""
    ctrl = np.asarray(ctrl, float)
    if ctrl.shape[0] < 2:
        return np.zeros((len(t), 3))
    return (ctrl.shape[0] - 1) * _bez_np(np.diff(ctrl, axis=0), t)


def _slerp_vec(a, b, t):
    """Blend two vectors: direction swung on the arc, length blended straight.
    Falls back to a straight blend when either is near zero."""
    la, lb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    ok = (la > 1e-6) & (lb > 1e-6)
    ah, bh = a / jnp.maximum(la, _EPS), b / jnp.maximum(lb, _EPS)
    cos = jnp.clip(jnp.dot(ah, bh), -0.9999, 0.9999)
    th = jnp.arccos(cos)
    s = jnp.maximum(jnp.sin(th), 1e-6)
    dirv = (jnp.sin((1 - t) * th) * ah + jnp.sin(t * th) * bh) / s
    arc = dirv * ((1 - t) * la + t * lb)
    lin = (1 - t) * a + t * b
    return jnp.where(ok & (th > 1e-3), arc, lin)


def _piecewise_slerp(knots, A, t, w):
    """Through vectors A[k] at knots, swung on the arc between each pair."""
    i, s, _ = _segment(knots, t)
    Ast = jnp.stack(A)
    return _slerp_vec(Ast[i], Ast[i + 1], s) * _bump(s, w)


# ------------------------------------------------------------------- patch
class Patch:
    """S(u, v) for one boundary blend.

    Built in two parts, the way a CAD sketch separates its topology from its
    dimensions:

    PLAN (decided once, in numpy, from the nominal curves; `plan=` reuses
    one): which curve is which side, each guide's direction and order, which
    curves cross and roughly where, and every yes/no validity check.

    GEOMETRY (JAX, rebuilt from the curves every time): the crossing
    parameters refined from the plan's guesses, the re-timing, the splines,
    the edge conditions and the surface. So the surface is differentiable
    with respect to every control point while the plan holds.

    curves:  list of 3 or 4 (K_i, 3) control-point arrays, head to tail
    modes:   list of the same length, each in MODES
    vecs:    list of (3,) direction vectors (ignored for "free")
    weights: list of scalars
    guides:  list of (K, 3) control-point arrays, each running across the patch
    plan:    a `Patch.plan` from the nominal design (default: make one here,
             which needs concrete numbers)
    """

    def __init__(self, curves, modes, vecs, weights, guides=(), plan=None):
        n = len(curves)
        if n not in (3, 4):
            raise ValueError(f"boundary blend needs 3 or 4 curves, got {n}")
        if plan is None:
            plan = self.make_plan(curves, modes, vecs, guides)
        self.plan = plan
        self.n = n
        self.pole = n == 3
        self.side_of = plan["side_of"]
        self.crossings = plan["crossings"]
        self.n_guides = plan["n_guides"]
        # S_u x S_v points away from the side the loop runs counter-clockwise
        # when seen from, for a 3-curve loop, and toward it for a 4-curve
        # loop (the two lay out u and v differently). `orient` evens that out:
        # orient * S_u x S_v always points AWAY from the counter-clockwise side.
        self.orient = 1.0 if n == 3 else -1.0
        C = [jnp.asarray(c, dtype=float) for c in curves]
        G = [jnp.asarray(g, dtype=float) for g in guides]
        self._geometry(self._sides(C), G)
        self._build(modes, vecs, weights)

    @staticmethod
    def _sides(C):
        """The four sides from the loop (works on numpy or JAX arrays)."""
        if len(C) == 4:
            return {"c0": C[0], "d1": C[1], "c1": C[2][::-1], "d0": C[3][::-1]}
        return {"c0": C[0][:1], "d0": C[0], "c1": C[1], "d1": C[2][::-1]}

    # -------------------------------------------------------------- plan
    @classmethod
    def make_plan(cls, curves, modes, vecs, guides=()):
        """The topology, from nominal (concrete) curves; raises on any network
        or condition that cannot work."""
        C = [np.asarray(c, dtype=float) for c in curves]
        n = len(C)
        for k in range(n):
            gap = float(np.linalg.norm(C[k][-1] - C[(k + 1) % n][0]))
            if gap > 1e-6:
                raise ValueError(
                    f"curve {k + 1} ends {gap:.3g} mm from where curve {(k + 1) % n + 1} "
                    "starts: the loop must run head to tail"
                )
        side_of = (
            {"v0": 0, "u1": 1, "v1": 2, "u0": 3}
            if n == 4
            else {"u0": 0, "v1": 1, "u1": 2, "v0": None}
        )
        for k, (mode, c) in enumerate(zip(modes, C, strict=True)):
            if mode not in MODES:
                raise ValueError(f"curve {k + 1}: condition {mode!r} not in {MODES}")
            if mode == "tangent":
                nrm = np.asarray(vecs[k], float)
                nrm = nrm / np.linalg.norm(nrm)
                d = _bez_tangent_np(c, np.linspace(0.0, 1.0, 21))
                d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                worst = float(np.degrees(np.arcsin(np.clip(np.abs(d @ nrm), 0.0, 1.0))).max())
                if worst > 0.5:
                    raise ValueError(
                        f"'tangent' condition impossible: the boundary curve leaves the "
                        f"requested plane by up to {worst:.1f} deg"
                    )
        raw = cls._sides(C)
        plan = {"side_of": side_of, "crossings": []}
        crossings = plan["crossings"]
        ug, vg = [], []
        for k, g in enumerate(np.asarray(x, float) for x in guides):
            ends = {
                s: (_closest_on(raw[s], g[0])[1], _closest_on(raw[s], g[-1])[1])
                for s in ("c0", "c1", "d0", "d1")
            }
            fits = {
                "u": min(ends["d0"][0] + ends["d1"][1], ends["d0"][1] + ends["d1"][0]),
                "v": min(ends["c0"][0] + ends["c1"][1], ends["c0"][1] + ends["c1"][0]),
            }
            fam = min(fits, key=fits.get)
            a, b = ("d0", "d1") if fam == "u" else ("c0", "c1")
            rev = bool(ends[a][1] + ends[b][0] < ends[a][0] + ends[b][1])
            if rev:
                g = g[::-1]  # start it on d0 / c0
            for s_, end in ((a, g[0]), (b, g[-1])):
                gap = _closest_on(raw[s_], end)[1]
                crossings.append((f"guide {k + 1} end on boundary {s_}", gap))
                if gap > GAP_TOL:
                    raise ValueError(
                        f"guide {k + 1} does not land on boundary {s_} ({gap:.3g} mm off)"
                    )
            (ug if fam == "u" else vg).append((k, rev, g))

        def place(g, sides):
            ts = [
                _closest_on(raw[s_], g[0 if i == 0 else -1])[0]
                for i, s_ in enumerate(sides)
                if raw[s_].shape[0] > 1
            ]
            return float(np.mean(ts))

        ug.sort(key=lambda e: place(e[2], ("d0", "d1")))
        vg.sort(key=lambda e: place(e[2], ("c0", "c1")))
        tmp = cls.__new__(cls)
        tmp.pole = n == 3
        tmp._no_touching(raw, [(k, g) for k, _, g in ug], [(k, g) for k, _, g in vg])
        U = [raw["c0"]] + [g for _, _, g in ug] + [raw["c1"]]  # u-direction, by v
        V = [raw["d0"]] + [g for _, _, g in vg] + [raw["d1"]]  # v-direction, by u
        nu, nv = len(V), len(U)

        # What meets what, and where (nominal): T[j][i] on U_j, Sg[i][j] on V_i.
        kind = {}
        T = np.zeros((nv, nu))
        Sg = np.zeros((nu, nv))
        for j, cu in enumerate(U):
            for i, cv in enumerate(V):
                if i in (0, nu - 1) and j in (0, nv - 1):
                    kind[(j, i)] = "corner"
                    T[j, i], Sg[i, j] = (0.0 if i == 0 else 1.0), (0.0 if j == 0 else 1.0)
                elif cu.shape[0] == 1:
                    kind[(j, i)] = "pole"
                    T[j, i], Sg[i, j] = np.nan, 0.0
                elif i in (0, nu - 1):
                    kind[(j, i)] = "u_end"
                    T[j, i] = 0.0 if i == 0 else 1.0
                    Sg[i, j], _ = _closest_on(cv, _bez_np(cu, [T[j, i]])[0])
                elif j in (0, nv - 1):
                    kind[(j, i)] = "v_end"
                    Sg[i, j] = 0.0 if j == 0 else 1.0
                    T[j, i], _ = _closest_on(cu, _bez_np(cv, [Sg[i, j]])[0])
                else:
                    kind[(j, i)] = "cross"
                    T[j, i], Sg[i, j], gap = _crossing(cu, cv)
                    crossings.append((f"u guide {j} x v guide {i}", gap))
                    if gap > GAP_TOL:
                        raise ValueError(
                            f"guides do not cross: u guide {j} and v guide {i} miss by {gap:.3g} mm"
                        )
        uk = np.nanmean(T, axis=0)
        vk = np.nanmean(Sg, axis=0)
        uk[0], uk[-1], vk[0], vk[-1] = 0.0, 1.0, 0.0, 1.0
        if np.any(np.diff(uk) <= 0) or np.any(np.diff(vk) <= 0):
            raise ValueError("guides out of order across the patch")
        for j in range(nv):
            row = T[j]
            if not np.isnan(row).any() and np.any(np.diff(row) <= 0):
                raise ValueError(
                    "guides cross a curve out of order: each family of "
                    "guides must be in the same order along every curve it crosses"
                )
        for i in range(nu):
            if np.any(np.diff(Sg[i]) <= 0) and not (
                n == 3 and np.allclose(Sg[i][:1], 0.0) and np.all(np.diff(Sg[i][1:]) > 0)
            ):
                raise ValueError(
                    "guides cross a curve out of order: each family of "
                    "guides must be in the same order along every curve it crosses"
                )
        plan.update(
            u_guides=[(k, rev) for k, rev, _ in ug],
            v_guides=[(k, rev) for k, rev, _ in vg],
            kind=kind,
            T=T,
            Sg=Sg,
            n_guides=(len(ug), len(vg)),
        )
        return plan

    # ---------------------------------------------------------- geometry
    def _geometry(self, raw, G):
        """Crossing parameters, shared knots and re-timings, in JAX, from the
        plan's topology and guesses."""
        plan = self.plan

        def oriented(k, rev):
            return G[k][::-1] if rev else G[k]

        U = [raw["c0"]] + [oriented(k, r) for k, r in plan["u_guides"]] + [raw["c1"]]
        V = [raw["d0"]] + [oriented(k, r) for k, r in plan["v_guides"]] + [raw["d1"]]
        nu, nv = len(V), len(U)
        T0, S0 = plan["T"], plan["Sg"]
        T = [[None] * nu for _ in range(nv)]
        Sg = [[None] * nv for _ in range(nu)]
        for (j, i), kd in plan["kind"].items():
            cu, cv = U[j], V[i]
            if kd == "corner":
                T[j][i], Sg[i][j] = jnp.asarray(T0[j, i]), jnp.asarray(S0[i, j])
            elif kd == "pole":
                T[j][i], Sg[i][j] = None, jnp.asarray(0.0)
            elif kd == "u_end":
                T[j][i] = jnp.asarray(T0[j, i])
                Sg[i][j] = _refine_on(cv, bezier(cu, T[j][i]), S0[i, j])
            elif kd == "v_end":
                Sg[i][j] = jnp.asarray(S0[i, j])
                T[j][i] = _refine_on(cu, bezier(cv, Sg[i][j]), T0[j, i])
            else:
                T[j][i], Sg[i][j] = _refine_cross(cu, cv, T0[j, i], S0[i, j])

        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs)

        uk = [jnp.asarray(0.0)] + [mean([T[j][i] for j in range(nv)]) for i in range(1, nu - 1)]
        uk = jnp.stack(uk + [jnp.asarray(1.0)])
        vk = [jnp.asarray(0.0)] + [mean([Sg[i][j] for i in range(nu)]) for j in range(1, nv - 1)]
        vk = jnp.stack(vk + [jnp.asarray(1.0)])
        self.uk, self.vk = uk, vk
        self._U = [
            (
                cu,
                None
                if cu.shape[0] == 1
                else _Retime(uk[1:-1], [T[j][i] for i in range(1, nu - 1)]),
            )
            for j, cu in enumerate(U)
        ]
        self._V = [
            (cv, _Retime(vk[1:-1], [Sg[i][j] for j in range(1, nv - 1)])) for i, cv in enumerate(V)
        ]

    def _no_touching(self, raw, ug, vg):
        """A curve network folds the surface if two curves of the SAME family
        touch or cross, or if a u and a v curve cross more than once. Refuse
        both. (Curves of one family may share the pole: that is where every
        v curve of a 3-curve loop starts.)"""

        def pts(c, skip_pole):
            t = np.linspace(0.03 if skip_pole else 0.0, 1.0, 400)
            return _bez_np(c, t)

        size = np.ptp(np.concatenate([raw[k] for k in raw]), axis=0).max()
        near = 1e-3 * size
        fams = {
            "u": [raw["c0"], raw["c1"]] + [g for _, g in ug],
            "v": [raw["d0"], raw["d1"]] + [g for _, g in vg],
        }
        for fam, cs in fams.items():
            for a in range(len(cs)):
                for b in range(a + 1, len(cs)):
                    if cs[a].shape[0] == 1 or cs[b].shape[0] == 1:
                        continue
                    if a < 2 and b < 2:
                        continue  # the two opposite boundaries
                    pa = pts(cs[a], self.pole and fam == "v")[5:-5]
                    pb = pts(cs[b], self.pole and fam == "v")[5:-5]
                    gap = np.min(np.linalg.norm(pa[:, None] - pb[None], axis=-1))
                    if gap < near:
                        raise ValueError(
                            f"two {fam}-direction curves touch or cross "
                            f"({gap:.3g} mm apart): the surface would fold"
                        )
        for gu in fams["u"][2:]:
            for gv in fams["v"][2:]:
                d = np.linalg.norm(pts(gu, False)[:, None] - pts(gv, self.pole)[None], axis=-1).min(
                    1
                )
                close = d < max(near, 0.02 * size)
                runs = np.count_nonzero(np.diff(close.astype(int)) == 1) + int(close[0])
                if runs > 1:
                    raise ValueError(
                        "a u guide and a v guide cross more than once: the surface would fold"
                    )

    def normal(self, u, v):
        """Unit normal, pointing away from the side the loop runs
        counter-clockwise when seen from (the negative side)."""
        su, sv = jax.jacfwd(self.S, argnums=(0, 1))(u, v)
        return _unit(self.orient * jnp.cross(su, sv))

    # ------------------------------------------------------------- build
    def _build(self, modes, vecs, weights):
        self.modes = list(modes)
        self.vecs = [np.asarray(v, float) for v in vecs]

        def curve(ctrl, rt):
            cj = jnp.asarray(ctrl, dtype=float)
            if rt is None:
                return lambda t: cj[0] + 0.0 * t
            return lambda t: bezier(cj, rt(t))

        self.Uf = [curve(c, rt) for c, rt in self._U]  # u-direction curves (by v)
        self.Vf = [curve(c, rt) for c, rt in self._V]  # v-direction curves (by u)
        self.c0, self.c1 = self.Uf[0], self.Uf[-1]
        self.d0, self.d1 = self.Vf[0], self.Vf[-1]
        self.Mu, self.Mv = _clamped_matrix(self.uk), _clamped_matrix(self.vk)
        uk = [self.uk[i] for i in range(self.uk.shape[0])]
        vk = [self.vk[j] for j in range(self.vk.shape[0])]

        corners = jnp.stack([self.c0(0.0), self.c0(1.0), self.c1(0.0), self.c1(1.0)])
        self.centroid = jnp.mean(corners, axis=0)

        dU = [_d(f) for f in self.Uf]
        dV = [_d(f) for f in self.Vf]
        # Each cross field: (side curve, its knots, the vectors wanted there —
        # the derivatives of the curves landing on it — and the sign:
        # + where the parameter enters the patch, - where it leaves).
        spec = {
            "u0": (self.d0, vk, [g(0.0) for g in dU], +1.0),  # S_u(0, v)
            "u1": (self.d1, vk, [g(1.0) for g in dU], -1.0),  # S_u(1, v)
            "v0": (self.c0, uk, [g(0.0) for g in dV], +1.0),  # S_v(u, 0)
            "v1": (self.c1, uk, [g(1.0) for g in dV], -1.0),  # S_v(u, 1)
        }
        self.fields, self.inward, self.targets = {}, {}, {}
        for side, (crv, kn, A, sgn) in spec.items():
            k = self.side_of[side]
            mode = "free" if k is None else modes[k]
            if mode not in MODES:
                raise ValueError(f"curve {k + 1}: condition {mode!r} not in {MODES}")
            vec = None if k is None else jnp.asarray(vecs[k], dtype=float)
            w = 1.0 if k is None else weights[k]
            kn = jnp.stack(kn)
            self.fields[side], self.inward[side] = self._field(crv, kn, A, sgn, mode, vec, w)
            self.targets[side] = (kn, A, sgn)

        self.f0, self.f1 = self.fields["u0"], self.fields["u1"]
        self.e0, self.e1 = self.fields["v0"], self.fields["v1"]
        # Crossing points, and the boundary cross-directions at the knots.
        self.X = [
            [0.5 * (self.Uf[j](uk[i]) + self.Vf[i](vk[j])) for j in range(len(vk))]
            for i in range(len(uk))
        ]  # X[i][j]
        self.e0k = [self.e0(x) for x in uk]
        self.e1k = [self.e1(x) for x in uk]
        self.f0k = [self.f0(x) for x in vk]
        self.f1k = [self.f1(x) for x in vk]
        # Twist candidates at each corner, one from each side meeting there.
        df = {0: _d(self.f0), 1: _d(self.f1)}
        de = {0: _d(self.e0), 1: _d(self.e1)}
        self.tw_f = {(i, j): df[i](float(j)) for i in (0, 1) for j in (0, 1)}
        self.tw_e = {(i, j): de[j](float(i)) for i in (0, 1) for j in (0, 1)}

    def _field(self, crv, knots, A, sgn, mode, vec, w):
        """Cross-boundary tangent along one side, as a function of t, plus the
        unit inward direction (for reporting). Passes through the vectors A
        at the knots (where the other curves land), blends between."""
        if mode == "free":

            def field(t):
                return _piecewise_slerp(knots, A, t, w)

            def inward(t):
                return sgn * _unit(field(t))

            return field, inward

        mid = crv(0.5)
        if mode == "normal":
            dhat = _unit(vec)

            def raw(t):
                return dhat
        else:  # tangent: in the plane with normal vec, square to the curve
            nrm = _unit(vec)
            dcrv = _d(crv)
            # (that this is possible -- the curve lies along the plane -- is
            # checked once, in the plan)

            def raw(t):
                return _unit(jnp.cross(nrm, dcrv(t)))

        sigma = jnp.where(jnp.dot(raw(0.5), self.centroid - mid) >= 0.0, 1.0, -1.0)

        def inward(t):
            return sigma * raw(t)

        mk = jnp.stack(
            [jnp.maximum(sgn * jnp.dot(a, inward(knots[i])), 0.0) for i, a in enumerate(A)]
        )

        def field(t):
            i, s, _ = _segment(knots, t)
            m = (1 - s) * mk[i] + s * mk[i + 1]
            return sgn * inward(t) * m * _bump(s, w)

        return field, inward

    # ------------------------------------------------------------- surface
    def _gregory(self, u, v):
        W = {}
        for i in (0, 1):
            for j in (0, 1):
                s, t = jnp.abs(u - i), jnp.abs(v - j)
                # on side u=i take the twist the u-side implies, on side v=j
                # the v-side's, blended between: both sides exact
                W[(i, j)] = (s * self.tw_f[(i, j)] + t * self.tw_e[(i, j)]) / (s + t + 1e-12)
        return W

    def S(self, u, v):
        """The surface point at (u, v) in [0, 1]^2."""
        uk, vk = self.uk, self.vk
        Pv = _spline(vk, self.Mv, [f(u) for f in self.Uf], self.e0(u), self.e1(u), v)
        Pu = _spline(uk, self.Mu, [f(v) for f in self.Vf], self.f0(v), self.f1(v), u)
        W = self._gregory(u, v)
        cols = [
            _spline(vk, self.Mv, self.X[i], self.e0k[i], self.e1k[i], v) for i in range(len(uk))
        ]
        T0 = _spline(vk, self.Mv, self.f0k, W[(0, 0)], W[(0, 1)], v)
        T1 = _spline(vk, self.Mv, self.f1k, W[(1, 0)], W[(1, 1)], v)
        Puv = _spline(uk, self.Mu, cols, T0, T1, u)
        return Pu + Pv - Puv

    def S_ext(self, u, v, reach):
        """The patch carried on past its edges, straight along its own
        tangent, `reach` mm per unit of parameter. A trimming skin has to
        cover the whole solid it cuts; this is how it does. Past a pole
        (v < 0) it fans out flat on the pole's tangent plane."""
        cu = jnp.clip(u, 0.0, 1.0)
        cv = jnp.clip(v, 0.0, 1.0)
        Su, Sv = jax.jacfwd(self.S, argnums=(0, 1))(cu, cv)
        return self.S(cu, cv) + (u - cu) * reach * _unit(Su) + (v - cv) * reach * _unit(Sv)

    def sample(self, n_u=40, n_v=40):
        """(n_u, n_v, 3) grid of points on the patch, for plots and checks."""
        us = jnp.linspace(0.0, 1.0, n_u)
        vs = jnp.linspace(0.0, 1.0, n_v)
        f = jax.vmap(jax.vmap(self.S, in_axes=(None, 0)), in_axes=(0, None))
        return f(us, vs)
