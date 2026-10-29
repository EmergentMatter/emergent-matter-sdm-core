"""Edge treatments for booleans: how faces meet at the edges they make.

A boolean of solids is decided by the distances to their faces; HOW those
faces meet at an edge is a separate choice, given as ``blend`` (with size
``blend_r``) on ``union`` / ``intersect`` / ``subtract`` and on ``box`` /
``capped_cylinder``:

    sharp            a sharp edge (plain max / min; the default)
    round, r         the round a ball of radius r rolls: a fillet where solids
                     meet (union), a round on an outside edge (intersect,
                     subtract)
    chamfer, r       the flat between the two lines where that same ball
                     touches the faces (legs of r where faces meet square)

THE ROLLING BALL. A union's round is the region a ball of radius r cannot
reach while staying outside every child (a morphological closing); an
intersection's is the same on the complements (an opening). Ball centres live
in E = {every child >= r}. For each query point the nearest feasible centre is
found: straight out from one face (that face's offset surface), or on the
ball's path along the edge where two offsets meet (the spine, found by
projected Newton). The surface sits at r from those centres. So:

  - the round is a true arc of radius r tangent to both faces at ANY angle,
    and the field is a true distance near it (rate 1);
  - where two children share a face, or their faces are parallel, there is no
    spine and nothing is added (no bulge on flush or coincident faces);
  - the result never leaves the children's hull, so their bounding box holds;
  - a gap narrower than 2r is bridged, because the ball cannot enter it (CAD
    would fail to fillet there).

Only the two children nearest each point form its edge, so the cost grows
linearly with the number of children (about a dozen child evaluations with
gradients per point; ~1 us per point for two boxes on a laptop CPU, against
~0.01 for a sharp union). Where three faces meet (a vertex) the round is the
two-face round of the nearest pair.

A single ``box`` / ``capped_cylinder`` rounds or chamfers its own edges with
``op_blend_faces``: its faces never coincide, so the closed form is already
the rolling-ball result there.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp

from software_defined_matter import wire

BLENDS = wire.BLENDS
BLEND_OPS = ("union", "intersect", "subtract")
_R2 = 0.7071067811865476
_ITERS = 8  # spine Newton steps
_TOL = 1e-4  # mm: how close a candidate centre must sit to its constraints
_BIG = 1e30


def is_blended(node):
    """True for a union / intersect / subtract that carries a non-sharp
    ``blend``: it is not a plain min / max of its children, so code that
    splits, flattens or re-orders such booleans must leave it whole."""
    if not isinstance(node, dict) or node.get("type") != "op":
        return False
    if node.get("op") not in BLEND_OPS:
        return False
    return (node.get("params") or {}).get("blend", "sharp") != "sharp"


# ----------------------------------------------------------- single primitive
def op_blend_faces(D, blend="sharp", r=0.0):
    """Combine ONE primitive's own face distances ``D`` (..., N) as an
    intersection (a box's three slabs, a cylinder's radial and axial pair).
    Exact there: those faces meet square and never coincide."""
    if blend == "sharp":
        return jnp.max(D, axis=-1)
    if blend == "round":
        out = jnp.sqrt(jnp.sum(jnp.maximum(D + r, 0.0) ** 2, axis=-1) + 1e-30)
        return jnp.minimum(-r, jnp.max(D, axis=-1)) + out
    if blend == "chamfer":
        pairs = [
            (D[..., i] + D[..., j] + r) * _R2
            for i, j in itertools.combinations(range(D.shape[-1]), 2)
        ]
        if not pairs:
            return jnp.max(D, axis=-1)
        return jnp.maximum(jnp.max(D, axis=-1), jnp.max(jnp.stack(pairs, -1), axis=-1))
    raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}")


# ----------------------------------------------------------- rolling ball
def _values_and_grads(fields, X):
    """Every child's value and gradient at points X (M, d): (K, M), (K, M, d).
    Each child maps points independently, so one pullback of ones gives the
    per-point gradients."""
    vs, gs = [], []
    for f in fields:
        v, pull = jax.vjp(f, X)
        vs.append(v)
        gs.append(pull(jnp.ones_like(v))[0])
    return jnp.stack(vs), jnp.stack(gs)


def _pick(A, idx):
    """A[idx[m], m, ...] for each point m."""
    if A.ndim == 2:
        return jnp.take_along_axis(A, idx[None], axis=0)[0]
    return jnp.take_along_axis(A, idx[None, :, None], axis=0)[0]


def _unit(v):
    return v / jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True) + 1e-30)


def _closing(fields, r, P, chamfer):
    """Rolling-ball union (round or chamfer) of ``fields`` at points P (M, d)."""
    dim = P.shape[-1]
    V, G = _values_and_grads(fields, P)
    order = jnp.argsort(V, axis=0)
    i0, i1 = order[0], order[1]
    a, b = _pick(V, i0), _pick(V, i1)
    ga, gb = _pick(G, i0), _pick(G, i1)
    ua, ub = a - r, b - r
    in_E = a >= r

    def lowest(X):
        return jnp.min(jnp.stack([f(X) for f in fields]), axis=0)

    # face candidates: straight out along one child's gradient to its offset
    qa = P + jnp.maximum(-ua, 0.0)[:, None] * ga
    qb = P + jnp.maximum(-ub, 0.0)[:, None] * gb
    ok_a = (ua < 0) & (lowest(qa) >= r - _TOL)
    ok_b = (ub < 0) & (lowest(qb) >= r - _TOL)

    def crossing(na, nb):
        if dim == 3:
            return jnp.sum(jnp.cross(na, nb) ** 2, axis=-1)
        return (na[:, 0] * nb[:, 1] - na[:, 1] * nb[:, 0]) ** 2

    # spine candidate: projected Newton onto {f_i0 = r, f_i1 = r}; in 3-D then
    # slide along the spine to the point nearest P. No crossing (parallel or
    # coincident faces): no spine, and the step is held at zero.
    C = P
    for _ in range(_ITERS):
        Vc, Gc = _values_and_grads(fields, C)
        na, nb = _pick(Gc, i0), _pick(Gc, i1)
        ok = crossing(na, nb) > 1e-6
        g = jnp.stack([_pick(Vc, i0) - r, _pick(Vc, i1) - r], axis=-1)  # (M, 2)
        Jm = jnp.stack([na, nb], axis=1)  # (M, 2, d)
        Mm = jnp.where(ok[:, None, None], Jm @ jnp.swapaxes(Jm, 1, 2), jnp.eye(2))
        step = jnp.einsum("mkd,mk->md", Jm, jnp.linalg.solve(Mm, g[..., None])[..., 0])
        C1 = C - jnp.where(ok[:, None], step, 0.0)
        if dim == 3:
            t = _unit(jnp.cross(na, nb))
            C1 = C1 + t * jnp.sum(t * (P - C1), axis=-1, keepdims=True)
        C = jnp.where(ok[:, None], C1, C)
    Vc, Gc = _values_and_grads(fields, C)
    na_c, nb_c = _unit(_pick(Gc, i0)), _unit(_pick(Gc, i1))
    on_spine = (
        (crossing(na_c, nb_c) > 1e-6)
        & (jnp.abs(_pick(Vc, i0) - r) < _TOL)
        & (jnp.abs(_pick(Vc, i1) - r) < _TOL)
        & (jnp.min(Vc, axis=0) >= r - _TOL)
    )
    d_spine = jnp.sqrt(jnp.sum((P - C) ** 2, axis=-1) + 1e-30)
    d_face = jnp.minimum(jnp.where(ok_a, -ua, _BIG), jnp.where(ok_b, -ub, _BIG))
    d_sp = jnp.where(on_spine, d_spine, _BIG)
    d_E = jnp.minimum(d_face, d_sp)
    # no candidate: an outside corner, where moving straight away from both
    # children is the way out
    d_E = jnp.where(d_E >= _BIG, jnp.maximum(-ua, -ub), d_E)
    rolled = jnp.where(in_E, a, r - d_E)
    if not chamfer:
        return rolled

    # chamfer: where the nearest centre is on the spine, the flat between the
    # two contact lines (the ball touches child i0 at C - r n0, i1 at C - r n1)
    spine_case = (~in_E) & on_spine & (d_sp <= d_face)
    ta, tb = C - r * na_c, C - r * nb_c
    e = tb - ta
    L = jnp.sqrt(jnp.sum(e * e, axis=-1) + 1e-30)
    e_hat = e / L[:, None]
    if dim == 3:
        T = _unit(jnp.cross(na_c, nb_c))
        m = _unit(jnp.cross(e_hat, T))
    else:
        T = None
        m = jnp.stack([-e_hat[:, 1], e_hat[:, 0]], axis=-1)
    m = m * jnp.where(jnp.sum(m * (C - ta), axis=-1) >= 0, 1.0, -1.0)[:, None]  # toward the ball
    s = jnp.sum((P - ta) * m, axis=-1)
    lam = jnp.sum((P - ta) * e_hat, axis=-1) / L

    def off_line(t0):
        w = P - t0
        if T is not None:
            w = w - T * jnp.sum(w * T, axis=-1, keepdims=True)
        return jnp.sqrt(jnp.sum(w * w, axis=-1) + 1e-30)

    outside = jnp.where((lam >= 0) & (lam <= 1), s, jnp.minimum(off_line(ta), off_line(tb)))
    chamfered = jnp.where(s < 0, jnp.minimum(a, s), outside)
    return jnp.where(spine_case, chamfered, a)


def _neg(f):
    return lambda X: -f(X)


def op_blend_union(fields, blend, r, P):
    """Union of the children ``fields`` (callables on points (M, d)) at P,
    with the edges it makes treated by ``blend`` of size ``r``."""
    if blend not in BLENDS:
        raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}")
    if blend == "sharp" or len(fields) == 1:
        return jnp.min(jnp.stack([f(P) for f in fields]), axis=0)
    return _closing(list(fields), r, P, chamfer=blend == "chamfer")


def op_blend_intersect(fields, blend, r, P):
    """Intersection, rounded or chamfered on the outside edges it makes (the
    rolling ball runs inside every child: the union of the complements)."""
    return -op_blend_union([_neg(f) for f in fields], blend, r, P)


def op_blend_subtract(fields, blend, r, P):
    """``fields[0]`` with every later child cut out, the edges of the cut
    treated by ``blend``."""
    return op_blend_intersect([fields[0]] + [_neg(f) for f in fields[1:]], blend, r, P)


__all__ = [
    "BLENDS",
    "BLEND_OPS",
    "is_blended",
    "op_blend_faces",
    "op_blend_intersect",
    "op_blend_subtract",
    "op_blend_union",
]
