"""`boundary_blend` as an SDF primitive: signed distance to the (extended)
skin, negative on one side, positive on the other. It is a CUTTING tool,
like a `plane` but curved: `subtract(block, skin)` keeps the side the skin's
positive side does not cover.

WHAT IT IS FOR, AND WHEN A LOFT IS BETTER. `bezier_2d` / `bspline_2d` describe
a flat region in a plane, and `loft` / `sweep` / extrusion carry such regions
through 3-D. A boundary blend is a different object: a surface stretched
across 3 or 4 edge curves that are themselves 3-D and need not lie in any
plane, plus guide curves across it, with the angle at which it leaves each
edge set by a condition. Use it when the shape is defined by its edges: a
saddle seat whose top edges rise and fall (examples/boundary_blend_
differentiable.py), a fairing or flare tangent to the faces it meets, a skin
through a curve network. If the shape is a stack of flat sections along one
axis (a soap bar, a bottle, a straight airfoil span), build it with
`loft` over `bspline_2d` sections instead: same surface, far cheaper.

COST. About 18-36 us per query point (jit, float32, CPU; measured 18 on an
Apple M-series laptop and 36 in review), against 0.2-2.4 us for a
3-section `bspline_2d` loft on the same machines: 15-90x more. One 128^3
grid is 40-75 s. Named-parameter control points cost the same per point
(the geometry is rebuilt once per call, not per point); a gradient with
respect to them about 1.7x that. Fine for a cutter evaluated a few times or
an optimisation over a coarse metric grid; budget for it in a loop.

PARAMETERS (one entry per curve in each per-curve list): `curves` (3 or 4
Bezier control-point lists, head to tail), `conditions` ("free" / "normal" /
"tangent"), `vectors` (the direction or plane normal each condition uses),
`weights`, `guides` (any number), `reach`, `flip`, `fold`.

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
    """Patch from node params: ``curves`` (3 or 4 control-point arrays) and
    optional ``guides``, as numbers or as traced values when ``plan`` is
    given; ``conditions`` (one of MODES per curve, default free), ``vectors``
    (one direction / plane normal per curve, ignored for free) and
    ``weights`` (one per curve, default 1)."""
    curves = list(params["curves"])
    n = len(curves)
    if n not in (3, 4):
        raise ValueError(f"boundary_blend needs 3 or 4 curves, got {n}")
    modes = list(params.get("conditions") or ["free"] * n)
    vecs = list(params.get("vectors") or [[0.0, 0.0, 1.0]] * n)
    weights = list(params.get("weights") or [1.0] * n)
    for name, vals in (("conditions", modes), ("vectors", vecs), ("weights", weights)):
        if len(vals) != n:
            raise ValueError(
                f"boundary_blend: {n} curves but {len(vals)} {name}: give one per curve"
            )
    for k, (m, v) in enumerate(zip(modes, vecs, strict=True)):
        if m == "free" and not np.any(np.asarray(v, float)):
            vecs[k] = [0.0, 0.0, 1.0]  # unused, but must be a direction
    return Patch(curves, modes, vecs, weights, list(params.get("guides") or []), plan=plan)


def _patch_size(params) -> float:
    """Diagonal of the control points' box. A Bezier patch stays inside its
    control points' hull, so this bounds the patch."""
    arrays = list(params["curves"]) + list(params.get("guides") or [])
    pts = np.concatenate([np.asarray(c, dtype=float).reshape(-1, 3) for c in arrays])
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
        n_hat = nrm / jnp.sqrt(jnp.sum(nrm * nrm) + 1e-30)
        along = jnp.dot(r, n_hat)  # the gap measured along the normal
        gap = jnp.sqrt(jnp.sum(r * r) + 1e-20)  # its length
        side = jnp.where(along >= 0.0, 1.0, -1.0)
        # At a true foot the gap lies along the normal, and the two agree; use
        # `along` there, so the spatial gradient is the unit normal even ON the
        # skin (the length's gradient vanishes where the gap is zero). Where
        # the search stopped at the end of the extension the gap is not along
        # the normal, and `along` could fall to zero beside the skin: keep the
        # length there (with its sign). Gaps under 1e-4 mm count as at the
        # foot whatever their direction: there the foot itself is only found
        # to ~1e-6 mm, sideways as often as not, and the two measures differ
        # by less than the allowance.
        at_foot = jnp.abs(along) >= 0.98 * gap - 1e-4
        return jnp.where(at_foot, along, side * gap)

    flat = p.reshape(-1, 3)
    out = jax.lax.map(one, flat, batch_size=_BATCH)
    out = -out if flip else out
    return out.reshape(p.shape[:-1])


# ------------------------------------------------------------ sdm-core hook
_STRUCTURAL = ("conditions", "vectors", "weights", "reach", "flip", "fold")


def _prim(
    p,
    curves,
    conditions=None,
    vectors=None,
    weights=None,
    guides=None,
    reach=None,
    flip=False,
    fold=None,
):
    """Signed distance to a boundary-blend skin, from literal params."""
    params = {
        "curves": curves,
        "conditions": conditions,
        "vectors": vectors,
        "weights": weights,
        "guides": guides,
    }
    patch = make_patch(params)
    if reach is None:
        reach = 2.0 * _patch_size(params)
    if fold is not None:
        # Symmetric skin: reflect every point onto the positive side of each
        # flagged plane (x=0, y=0, z=0) before measuring. Exact when the
        # skin meets those planes square, as a symmetry plane requires.
        f = jnp.asarray(fold) > 0.5
        p = jnp.where(f, jnp.abs(jnp.asarray(p)), p)
    return signed_distance(p, patch, reach, flip=bool(flip))


def compile_boundary_blend(node, binding=None):
    """Closure ``fn(p, free_vec)`` for a boundary-blend node.

    The plan (topology and validity checks) is made ONCE here, from the
    nominal parameter values; every call then rebuilds the geometry from the
    live values in JAX, so curve and guide control points written as
    ``$ref`` / expressions are differentiable. Everything else (conditions,
    vectors, weights, reach, flip, fold) is structural and must be literal.
    """
    from software_defined_matter.dsl.resolve import resolve_param_value

    raw = dict(node.get("params") or {})
    if any(_has_ref(raw.get(k)) for k in _STRUCTURAL):
        raise ValueError(
            "boundary_blend: only curve and guide control points may be named parameters"
        )
    struct = {k: raw.get(k) for k in ("conditions", "vectors", "weights")}
    reach, flip, fold = raw.get("reach"), bool(raw.get("flip", False)), raw.get("fold")
    geo = {"curves": list(raw["curves"]), "guides": list(raw.get("guides") or [])}

    def resolved(free_vec):
        return {k: [resolve_param_value(c, binding, free_vec) for c in v] for k, v in geo.items()}

    if binding is None:
        nominal = {k: [np.asarray(c, float) for c in v] for k, v in geo.items()}
    else:
        nominal = {
            k: [np.asarray(c) for c in v]
            for k, v in resolved(binding.initial_free_vector()).items()
        }
    patch0 = make_patch(nominal | struct)
    plan = patch0.plan
    if reach is None:
        reach = 2.0 * _patch_size(nominal)

    def fn(p, free_vec):
        patch = patch0 if binding is None else make_patch(resolved(free_vec) | struct, plan=plan)
        q = p
        if fold is not None:
            q = jnp.where(jnp.asarray(fold) > 0.5, jnp.abs(p), p)
        return signed_distance(q, patch, reach, flip=flip)

    return fn


def boundary_blend_params(
    curves, conditions=None, reach=None, flip=False, fold=None, *, guides=None, check=True
):
    """The params of a boundary blend node (see ``model.sdf_boundary_blend``).

    curves:     3 or 4 lists of Bezier control points, head to tail.
    conditions: one per curve (same count): "free", ("normal", vec),
                ("tangent", vec), optionally with a weight: ("normal", vec, 1.3).
                Stored in the .sdm as three lists, one entry per curve:
                ``conditions`` (names), ``vectors`` and ``weights``.
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
                cross each other, once each. Any number.
    check:      (keyword) build the surface now and refuse it if it folds
                (see `check.quality_report`).
    """
    curves = [_points(c) for c in curves]
    if len(curves) not in (3, 4):
        raise ValueError("boundary blend needs 3 or 4 curves")
    guides = [_points(g) for g in guides or []]
    literal = not any(_has_ref(c) for c in curves) and not any(_has_ref(g) for g in guides)
    if literal:
        _check_chain([np.asarray(c, float) for c in curves])
    conditions = list(conditions) if conditions is not None else ["free"] * len(curves)
    if len(conditions) != len(curves):
        raise ValueError(
            f"{len(curves)} curves but {len(conditions)} conditions: give one per curve"
        )
    names, vectors, weights = [], [], []
    for i, cond in enumerate(conditions, start=1):
        if isinstance(cond, str):
            cond = (cond,)
        if cond[0] not in MODES:
            raise ValueError(f"curve {i}: condition {cond[0]!r} not in {MODES}")
        names.append(cond[0])
        vectors.append([float(x) for x in cond[1]] if cond[0] != "free" else [0.0, 0.0, 0.0])
        weights.append(float(cond[2]) if len(cond) > 2 else 1.0)
    params: dict = {"curves": curves}
    if any(m != "free" for m in names):
        params["conditions"] = names
        params["vectors"] = vectors
    if any(w != 1.0 for w in weights):
        params["weights"] = weights
    if guides:
        params["guides"] = guides
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
