"""`boundary_blend` as an SDF primitive: signed distance to the (extended)
skin, negative on one side, positive on the other. It is a CUTTING tool,
like a `plane` but curved: `subtract(block, skin)` keeps the side the skin's
positive side does not cover.

WHICH SIDE IS NEGATIVE. The side from which the curve loop runs
counter-clockwise. (The soap bar's loop runs counter-clockwise seen from
above, so the skin is negative above: it is the space to cut away, as is.)
`flip=True` swaps sides.

HOW THE DISTANCE IS FOUND. A curved patch has no closed-form distance. For
each query point: the nearest of a fixed grid of points on the patch and its
extension gives a starting (u, v); a few damped Gauss-Newton steps then slide
it to the foot of the perpendicular. The sign is which side of the surface's
normal the point is on.

DIFFERENTIABLE IN ITS SHAPE. Curve and guide control points may be named
parameters ($ref) or expressions. The topology (which curve is which side,
guide order, which curves cross) is fixed once from the nominal design; the
geometry is rebuilt in JAX from the live parameter values every evaluation
(see boundary_blend_patch.Patch). The foot of the perpendicular is held fixed
for the derivative (at the true foot, sliding along the skin does not change
the distance to first order), so d(distance)/d(parameter) is -n . dS/dparam
there: exact, and cheap. If the parameters move far enough to change the
topology (guides crossing twice, the skin folding), the derivative stops
meaning anything: re-run `boundary_blend_check.quality_report` at the new
design, as an optimiser's checkpoint would.

ACCURACY: AN APPROXIMATE SIGNED FIELD, NOT A GUARANTEED DISTANCE. The value
is the distance to a point that IS on the skin, so it is never shorter than
the true distance: it can only be too long. Near the skin (where the solver
starts next to the foot) it is exact to well under a micron. Far from it
(tens of mm, or across a deep fold of the part) the solver can settle on a
point that is not the nearest one, and the value can read long by a few
tenths of a mm, or jump where two candidate feet swap. So it does NOT carry
the primitive promise of a rate <= 1 everywhere: `sdf.lipschitz` reports it
as UNKNOWN (like `loft` and `vsweep`), and everything that culls space on a
rate bound stops trusting it. The sign and the zero set are right everywhere,
which is what meshing and volume integration use. A sphere-tracing renderer
must not trust it as a safe step far from the skin.

The GLSL viewer cannot draw it (the foot-point solver has no shader routine):
bake it to a raster_field for the web viewer. Python consumers (meshing,
export, the Fabrication Navigator) evaluate it directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from software_defined_matter.sdf.boundary_blend_patch import MODES, Patch

KIND = "boundary_blend"
N_GUIDES = 12  # guide_1 .. guide_12, either direction
_N_NEWTON = 20
_BATCH = 4096


def _grid(n_in=17, n_out=9, dense_at_0=False):
    """Parameter samples for the starting guess. On the patch [0, 1]: even,
    or bunched toward 0 next to a pole (where a step in v is a short step on
    the surface). On the extensions either side: doubling distances, from
    1/256 of the reach out to the whole reach, so a point just past an edge
    starts right next to its foot and a point far away still has one."""
    t = np.linspace(0.0, 1.0, n_in)
    inner = t * t if dense_at_0 else t
    out = 2.0 ** np.arange(-(n_out - 1), 1)  # 1/256 .. 1
    return np.concatenate([-out[::-1], inner, 1.0 + out])


def make_patch(params, plan=None) -> Patch:
    """Patch from node params (curves and guides as arrays: numbers, or
    traced values when `plan` is given). Curve i's condition is given by
    which key is present: `normal_i` (a direction), `tangent_i` (a plane
    normal), or neither (free)."""
    n = 4 if params.get("curve_4") is not None else 3
    curves = [params[f"curve_{i}"] for i in range(1, n + 1)]
    modes, vecs = [], []
    for i in range(1, n + 1):
        if params.get(f"normal_{i}") is not None:
            modes.append("normal")
            vecs.append(params[f"normal_{i}"])
        elif params.get(f"tangent_{i}") is not None:
            modes.append("tangent")
            vecs.append(params[f"tangent_{i}"])
        else:
            modes.append("free")
            vecs.append([0.0, 0.0, 1.0])
    weights = [params.get(f"weight_{i}", 1.0) for i in range(1, n + 1)]
    guides = [
        params[f"guide_{k}"] for k in range(1, N_GUIDES + 1) if params.get(f"guide_{k}") is not None
    ]
    return Patch(curves, modes, vecs, weights, guides, plan=plan)


def _patch_size(params) -> float:
    """Diagonal of the control points' box. A Bezier patch stays inside its
    control points' hull, so this bounds the patch (plain numpy: works
    while being traced)."""
    pts = np.concatenate(
        [
            np.asarray(params[k], dtype=float).reshape(-1, 3)
            for k in params
            if k.startswith(("curve_", "guide_"))
        ]
    )
    return float(np.linalg.norm(pts.max(0) - pts.min(0)))


def signed_distance(p, patch: Patch, reach, flip=False):
    """Signed distance from points `p` (..., 3) to the extended skin."""
    p = jnp.asarray(p, dtype=float)
    reach = jnp.asarray(reach, dtype=float)
    u_lo, u_hi = -1.0, 2.0
    v_lo, v_hi = -1.0, 2.0  # a pole is carried past too: flat, on its tangent plane
    gu = jnp.asarray(_grid(), dtype=float)
    gv = jnp.asarray(_grid(dense_at_0=patch.pole), dtype=float)
    UU, VV = jnp.meshgrid(gu, gv, indexing="ij")
    UU, VV = UU.ravel(), VV.ravel()
    S = lambda u, v: patch.S_ext(u, v, reach)  # noqa: E731
    G = jax.vmap(S)(UU, VV)  # (Ng, 3)
    J = jax.jacfwd(S, argnums=(0, 1))

    def one(q):
        d2 = jnp.sum((G - q) ** 2, axis=-1)
        k = jnp.argmin(d2)
        uv0 = jnp.stack([UU[k], VV[k]])

        def body(_, st):
            uv, lam = st
            r = S(uv[0], uv[1]) - q
            ju, jv = J(uv[0], uv[1])
            Jm = jnp.stack([ju, jv], axis=-1)  # (3, 2)
            A = Jm.T @ Jm
            g = Jm.T @ r
            A = A + lam * jnp.diag(jnp.diag(A)) + 1e-12 * jnp.eye(2)
            step = jnp.linalg.solve(A, g)
            cand = uv - step
            cand = jnp.stack([jnp.clip(cand[0], u_lo, u_hi), jnp.clip(cand[1], v_lo, v_hi)])
            rc = S(cand[0], cand[1]) - q
            better = jnp.sum(rc * rc) < jnp.sum(r * r)
            uv = jnp.where(better, cand, uv)
            lam = jnp.where(better, lam * 0.3, lam * 10.0)
            return uv, lam

        uv, _ = jax.lax.fori_loop(0, _N_NEWTON, body, (uv0, jnp.asarray(1e-3, dtype=float)))
        # Hold the foot's (u, v) fixed for the derivative: at the true foot
        # the distance does not change to first order as (u, v) slides, so
        # only how the skin itself moves (and the query point) counts.
        uv = jax.lax.stop_gradient(uv)
        foot = S(uv[0], uv[1])
        # Normal. At a pole S_u vanishes, so read it a hair off; past a pole
        # (v < 0) the parametrisation runs back through the point, which turns
        # S_u x S_v over: turn it back.
        un, vn = uv[0], uv[1]
        if patch.pole:
            vn = jnp.where(jnp.abs(vn) < 1e-3, jnp.where(vn < 0.0, -1e-3, 1e-3), vn)
        ju, jv = J(un, vn)
        nrm = patch.orient * jnp.cross(ju, jv)
        if patch.pole:
            nrm = jnp.where(vn < 0.0, -nrm, nrm)
        r = q - foot
        dist = jnp.sqrt(jnp.sum(r * r) + 1e-20)
        side = jnp.where(jnp.dot(r, nrm) >= 0.0, 1.0, -1.0)
        return dist * side

    flat = p.reshape(-1, 3)
    out = jax.lax.map(one, flat, batch_size=_BATCH)
    out = -out if flip else out
    return out.reshape(p.shape[:-1])


# ------------------------------------------------------------ sdm-core hook
_KEYS = [f"{k}_{i}" for i in range(1, 5) for k in ("curve", "normal", "tangent", "weight")] + [
    f"guide_{k}" for k in range(1, N_GUIDES + 1)
]


def _prim(p, reach=None, flip=False, fold=None, **kw):
    params = {k: v for k, v in kw.items() if v is not None}
    unknown = set(params) - set(_KEYS)
    if unknown:
        raise ValueError(f"boundary_blend: unknown parameters {sorted(unknown)}")
    patch = make_patch(params)
    if reach is None:
        reach = 2.0 * _patch_size(params)
    if fold is not None:
        # Symmetric skin: reflect every point onto the positive side of each
        # flagged plane (x=0, y=0, z=0) before measuring. Exact when the
        # skin meets those planes square, as a symmetry plane requires.
        f = jnp.asarray(fold) > 0.5
        p = jnp.where(f, jnp.abs(jnp.asarray(p)), p)
    return signed_distance(p, patch, reach, flip=float(np.asarray(flip)) > 0.5)


_GEOMETRY = tuple(f"curve_{i}" for i in range(1, 5)) + tuple(
    f"guide_{k}" for k in range(1, N_GUIDES + 1)
)


def compile_boundary_blend(node, binding=None):
    """Closure ``fn(p, free_vec)`` for a boundary-blend node.

    The plan (topology and validity checks) is made ONCE here, from the
    nominal parameter values; every call then rebuilds the geometry from the
    live values in JAX, so curve and guide control points written as
    ``$ref`` / expressions are differentiable. The other params (conditions,
    weights, reach, flip, fold) are structural and must be literal numbers.
    """
    from software_defined_matter.dsl.resolve import resolve_param_value

    raw = dict(node.get("params") or {})
    reach, flip, fold = raw.pop("reach", None), raw.pop("flip", False), raw.pop("fold", None)
    unknown = set(raw) - set(_KEYS)
    if unknown:
        raise ValueError(f"boundary_blend: unknown parameters {sorted(unknown)}")
    struct = {k: v for k, v in raw.items() if k not in _GEOMETRY}
    if any(isinstance(v, dict) for v in struct.values()) or any(
        isinstance(v, dict) for v in (reach, fold) if v is not None
    ):
        raise ValueError(
            "boundary_blend: only curve and guide control points may be named parameters"
        )
    geo = {k: v for k, v in raw.items() if k in _GEOMETRY}
    if binding is None:
        nominal = {k: np.asarray(v, float) for k, v in geo.items()}
        free0 = None
    else:
        free0 = binding.initial_free_vector()
        nominal = {k: np.asarray(resolve_param_value(v, binding, free0)) for k, v in geo.items()}
    patch0 = make_patch(nominal | struct)
    plan = patch0.plan
    if reach is None:
        reach = 2.0 * _patch_size(nominal)
    b_flip = bool(flip) if not isinstance(flip, (int, float)) else float(flip) > 0.5

    def fn(p, free_vec):
        if binding is None:
            patch = patch0
        else:
            live = {k: resolve_param_value(v, binding, free_vec) for k, v in geo.items()}
            patch = make_patch(live | struct, plan=plan)
        q = p
        if fold is not None:
            q = jnp.where(jnp.asarray(fold) > 0.5, jnp.abs(p), p)
        return signed_distance(q, patch, reach, flip=b_flip)

    return fn


def boundary_blend_params(
    curves, conditions=None, reach=None, flip=False, fold=None, *, guides=None, check=True
):
    """The params of a boundary blend node (see ``model.sdf_boundary_blend``).

    curves:     3 or 4 lists of Bezier control points, head to tail.
    conditions: one per curve (same count): "free", ("normal", vec),
                ("tangent", vec), optionally with a weight: ("normal", vec, 1.3).
                Stored in the .sdm as `normal_i` / `tangent_i` / `weight_i`.
    reach:      how far (mm) the skin carries on past its edges; default
                twice the patch's size. Make it cover the solid being cut.
    flip:       swap which side is negative.
    fold:       symmetry, e.g. "xy" or "xyz": the skin is mirrored about
                those planes through the origin (build one quarter / eighth
                in the positive corner, square to the planes). Use this, not
                sdm-core's `mirror`, which unions copies and lets each copy's
                far-off (inaccurate) region compete with the real one.
    guides:     (keyword) curves across the patch in either direction, each
                from one boundary to the one facing it; the two families must
                cross each other, once each. Up to 12.
    check:      (keyword) build the surface now and refuse it if it folds
                (see `check.quality_report`).
    """
    curves = [_points(c) for c in curves]
    if len(curves) not in (3, 4):
        raise ValueError("boundary blend needs 3 or 4 curves")
    literal = not any(_has_ref(c) for c in curves) and not any(_has_ref(g) for g in guides or [])
    if literal:
        _check_chain([np.asarray(c, float) for c in curves])
    conditions = list(conditions) if conditions is not None else ["free"] * len(curves)
    if len(conditions) != len(curves):
        raise ValueError(
            f"{len(curves)} curves but {len(conditions)} conditions: give one per curve"
        )
    params = {}
    for i, c in enumerate(curves, start=1):
        params[f"curve_{i}"] = c
    for i, cond in enumerate(conditions, start=1):
        if isinstance(cond, str):
            cond = (cond,)
        if cond[0] not in MODES:
            raise ValueError(f"curve {i}: condition {cond[0]!r} not in {MODES}")
        if cond[0] != "free":
            params[f"{cond[0]}_{i}"] = [float(x) for x in cond[1]]
        if len(cond) > 2:
            params[f"weight_{i}"] = float(cond[2])
    guides = list(guides or [])
    if len(guides) > N_GUIDES:
        raise ValueError(f"at most {N_GUIDES} guides")
    for k, g in enumerate(guides, start=1):
        params[f"guide_{k}"] = _points(g)
    if reach is not None:
        params["reach"] = float(reach)
    if flip:
        params["flip"] = True
    if fold:
        params["fold"] = [1.0 if a in fold else 0.0 for a in "xyz"]
    # With named parameters the network is checked when the part compiles
    # (on the nominal values); with literal numbers it is checked now.
    if check and literal:
        from software_defined_matter.sdf.boundary_blend_check import quality_report

        q = quality_report(
            make_patch(params), reach if reach is not None else 2.0 * _patch_size(params)
        )
        if not q["ok"]:
            raise ValueError(f"boundary blend folds over: {q['problems']}")
    return params


def _has_ref(c):
    if isinstance(c, dict):
        return True
    if isinstance(c, (list, tuple)):
        return any(_has_ref(x) for x in c)
    return False


def _points(c):
    """Control points for the .sdm: plain floats, or kept as given (lists
    holding $ref / expression leaves)."""
    if _has_ref(c):
        return [list(p) for p in c]
    return np.asarray(c, dtype=float).tolist()


def _check_chain(curves, tol=1e-6):
    n = len(curves)
    for i in range(n):
        a, b = curves[i][-1], curves[(i + 1) % n][0]
        gap = float(np.linalg.norm(a - b))
        if gap > tol:
            raise ValueError(
                f"curve {i + 1} ends at {a.tolist()} but curve {(i + 1) % n + 1} "
                f"starts at {b.tolist()} ({gap:.3g} mm apart): the loop must run head to tail"
            )
