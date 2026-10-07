"""``vsweep``: ANY 2-D profile swept along a 3-D polyline, with any of the
profile's parameters free to change along the length, the profile free to
twist about the path, and mitred corners.

``sweep`` carries one fixed profile along a smooth path with ball-jointed
corners. A printed conductor, a tapering duct, a blade root that morphs into
an airfoil, all want more: the section must grow, shrink, round its corners
or turn as it goes, and where the path bends the two legs must meet on a
clean mitre, not a sphere. That is this node.

DEFINITION. ``child`` is a 2-D SDF tree, any shape the 2-D vocabulary can
make. ``path`` is a polyline of K points. Inside the child, any parameter
value may be written as ``{"$along": [v_0, ..., v_(K-1)]}``: one value per
path vertex, the same shape the parameter normally has (a scalar, a vec2, a
point list ...). Between vertices the value is linear in distance along the
segment, so a rounded box's half-sizes can taper while its corner radius
grows, a circle can swell, a polygon's vertices can slide. ``twist`` is an
optional angle per vertex (radians) turning the section about the path, also
linear along each segment. ``up`` (per vertex, default +Z) fixes which way
the profile's second axis points, with its along-path part removed (a
segment running along ``up`` falls back to the world X or Y axis); ``mitre``
gives a vertex an explicit cut-plane normal (zero = bisector) for corners
whose edges meet off the bisector. ``closed`` joins the last vertex back to
the first.

THE SOLID is the union of one slab per segment: the profile carried along
the segment in its frame (u across, v up), its parameters interpolated at
the query's own position along the segment, cut at both ends by the mitre
planes (an open path's ends are cut square). Two consecutive coincident
vertices make a zero-length segment that contributes nothing: that is how a
section changes abruptly at a corner. Per slab the in-plane distance and the
distance past the end planes combine as for a box; the slabs are minimised
over, so the SIGN is exact everywhere and the surface sits where it should.
The value is exact where the section is constant along a segment and the end
planes are square; a taper or a leaning mitre makes it over-report slightly
away from the surface (a few percent in the corner wedges of a square loop),
the same documented-bound status as ``helix``.

Evaluated in chunks of segments with a running minimum, so memory does not
grow with path length. Everything is JAX: the field differentiates through
the path, the parameters and the twist.

Originated in the cooled axial stator CEM as a rounded-rectangle-only
primitive (``cooled_stator_sdf.sdm.vsweep``); generalised here.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from software_defined_matter.dsl.resolve import ParamBinding, resolve_param_value

__all__ = ["ALONG_KEY", "split_along", "compile_vsweep", "vsweep_distance"]

_CHUNK = 32
ALONG_KEY = "$along"
_REF_PREFIX = "__vsweep_along_"


# ── the `$along` leaves ───────────────────────────────────────────────────


def split_along(tree: Any) -> tuple[Any, dict[str, np.ndarray]]:
    """A copy of the profile tree with every ``{"$along": [...]}`` leaf replaced
    by a ``$ref`` to a synthetic name, plus ``{name: (K, ...) array}``."""
    found: dict[str, np.ndarray] = {}

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            if ALONG_KEY in x:
                name = f"{_REF_PREFIX}{len(found)}"
                found[name] = np.asarray(x[ALONG_KEY], dtype=float)
                return {"$ref": name}
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(copy.deepcopy(tree)), found


class _AlongBinding(ParamBinding):
    """The part's binding, plus the synthetic along names served from
    ``current`` (set per evaluation to the interpolated per-point arrays)."""

    def __init__(self, inner: ParamBinding, names: set[str]) -> None:
        self.__dict__.update(inner.__dict__)
        self._along_names = set(names)
        self.current: dict[str, jnp.ndarray] = {}

    def get(self, name: str, free_vec: jnp.ndarray) -> jnp.ndarray:
        if name in self._along_names:
            return self.current[name]
        return super().get(name, free_vec)


# ── geometry ──────────────────────────────────────────────────────────────


def _safe_hypot(x, y):
    """sqrt(x^2 + y^2) with a finite gradient at the origin."""
    s2 = x * x + y * y
    ok = s2 > 0.0
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, s2, 1.0)), 0.0)


def _unit(x):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _frames(path, up):
    a, b = path[:-1], path[1:]
    seg = b - a
    # (zero-length segments are masked out of the minimum later)
    ln = jnp.sqrt(jnp.maximum(jnp.sum(seg * seg, axis=-1), 1e-18))
    t = seg / ln[:, None]
    upm = 0.5 * (up[:-1] + up[1:])
    v = upm - jnp.sum(upm * t, axis=-1, keepdims=True) * t
    # A segment running along `up` leaves nothing of it: fall back to the
    # world axis least aligned with the segment, so its frame never collapses.
    fallback = jnp.where(
        (jnp.abs(t[:, 0]) < 0.9)[:, None],
        jnp.array([1.0, 0.0, 0.0], dtype=t.dtype),
        jnp.array([0.0, 1.0, 0.0], dtype=t.dtype),
    )
    fallback = fallback - jnp.sum(fallback * t, axis=-1, keepdims=True) * t
    small = jnp.linalg.norm(v, axis=-1, keepdims=True) < 1e-6
    v = _unit(jnp.where(small, fallback, v))
    u = jnp.cross(v, t)
    return a, t, ln, u, v


def _close(x):
    return jnp.concatenate([x, x[:1]], axis=0)


def _evaluate(p, path, profile: Callable, along: dict[str, jnp.ndarray], up, twist, mitre, closed):
    """The distance. ``profile(q2d, values)`` evaluates the 2-D child at
    ``q2d`` (M, 2) with the along names bound to ``values`` (name -> (M, ...))."""
    p = jnp.asarray(p)
    if not jnp.issubdtype(p.dtype, jnp.floating):
        p = p.astype(jnp.result_type(float))
    path = jnp.asarray(path, dtype=p.dtype)
    k = path.shape[0]
    upv = (
        jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0], dtype=p.dtype), path.shape)
        if up is None
        else jnp.asarray(up, dtype=p.dtype)
    )
    mit = jnp.zeros_like(path) if mitre is None else jnp.asarray(mitre, dtype=p.dtype)
    tw = jnp.zeros((k,), dtype=p.dtype) if twist is None else jnp.asarray(twist, dtype=p.dtype)
    along = {n: jnp.asarray(v, dtype=p.dtype) for n, v in along.items()}
    if closed:
        path, upv, mit, tw = _close(path), _close(upv), _close(mit), _close(tw)
        along = {n: _close(v) for n, v in along.items()}
    a, t, ln, u_ax, v_ax = _frames(path, upv)
    live = ln > 1e-6
    n_seg = a.shape[0]

    if closed:
        t_next, t_prev = jnp.roll(t, -1, axis=0), jnp.roll(t, 1, axis=0)
    else:
        t_next = jnp.concatenate([t[1:], t[-1:]], axis=0)
        t_prev = jnp.concatenate([t[:1], t[:-1]], axis=0)
    m_end = _unit(t + t_next)
    m_sta = _unit(t_prev + t)

    def explicit(m, tt, auto):
        n = jnp.linalg.norm(m, axis=-1, keepdims=True)
        m = m / jnp.maximum(n, 1e-12)
        m = jnp.where(jnp.sum(m * tt, -1, keepdims=True) < 0.0, -m, m)
        return jnp.where(n > 1e-9, m, auto)

    m_end = explicit(mit[1:], t, m_end)
    m_sta = explicit(mit[:-1], t, m_sta)
    b = a + ln[:, None] * t

    pad = (-n_seg) % _CHUNK

    def padded(x, fill=None):
        if fill is None:
            tail = jnp.broadcast_to(x[:1], (pad,) + x.shape[1:])
        else:
            tail = jnp.full((pad,) + x.shape[1:], fill, dtype=x.dtype)
        return jnp.concatenate([x, tail], axis=0)

    # Which end planes are real boundaries: an open path's first start plane
    # and last end plane. The planes between slabs are internal: they clip
    # each slab on the outside but must not show through on the inside, or a
    # point deep in the bar would report the distance to an invisible seam.
    idx = jnp.arange(n_seg)
    first = (idx == 0) & (not closed)
    last = (idx == n_seg - 1) & (not closed)

    names = sorted(along)
    arrs = [
        padded(a, 1e9),
        padded(b, 1e9),
        padded(t),
        padded(ln),
        padded(u_ax),
        padded(v_ax),
        padded(m_sta),
        padded(m_end),
        padded(live, False),
        padded(tw[:-1]),
        padded(tw[1:]),
        padded(first, False),
        padded(last, False),
    ]
    arrs += [padded(along[n][:-1]) for n in names] + [padded(along[n][1:]) for n in names]
    arrs = tuple(x.reshape((-1, _CHUNK) + x.shape[1:]) for x in arrs)

    flat_p = p.reshape(-1, 3)
    n_pts = flat_p.shape[0]
    n_al = len(names)

    def step(best, blk):
        ab, bb, tb, lb, ub, vb, msb, meb, lvb, twa, twb, fst, lst = blk[:13]
        al_a, al_b = blk[13 : 13 + n_al], blk[13 + n_al :]
        w = flat_p[:, None, :] - ab[None]  # (P, C, 3)
        s = jnp.sum(w * tb[None], axis=-1)
        perp = w - s[..., None] * tb[None]
        uu = jnp.sum(perp * ub[None], axis=-1)
        vv = jnp.sum(perp * vb[None], axis=-1)
        f = jnp.clip(s / lb[None], 0.0, 1.0)  # (P, C)
        # twist: turning the section by +theta is sampling the profile at the point turned by -theta
        th = twa[None] + f * (twb[None] - twa[None])
        c, sn = jnp.cos(th), jnp.sin(th)
        q_u, q_v = uu * c + vv * sn, -uu * sn + vv * c
        q2d = jnp.stack([q_u, q_v], axis=-1).reshape(-1, 2)  # (P*C, 2)
        vals = {}
        for name, va, vb_ in zip(names, al_a, al_b, strict=True):
            ff = f.reshape(f.shape + (1,) * (va.ndim - 1))  # (P, C, 1, ...)
            val = va[None] + ff * (vb_[None] - va[None])  # (P, C, ...)
            vals[name] = val.reshape((-1,) + val.shape[2:])
        d_sec = profile(q2d, vals).reshape(f.shape)
        d_sta = -jnp.sum(w * msb[None], axis=-1)
        d_end = jnp.sum((flat_p[:, None, :] - bb[None]) * meb[None], axis=-1)
        d_ax = jnp.maximum(d_sta, d_end)  # every plane clips outside
        d_ax_in = jnp.maximum(
            jnp.where(fst[None], d_sta, -jnp.inf),  # only real caps count inside
            jnp.where(lst[None], d_end, -jnp.inf),
        )
        # Past ANY of the slab's planes the point is outside this slab and the
        # inside term is zero; within them, only the real caps bound it.
        inside = jnp.where(d_ax > 0.0, 0.0, jnp.minimum(jnp.maximum(d_sec, d_ax_in), 0.0))
        d = _safe_hypot(jnp.maximum(d_sec, 0.0), jnp.maximum(d_ax, 0.0)) + inside
        d = jnp.where(lvb[None], d, jnp.inf)
        return jnp.minimum(best, jnp.min(d, axis=-1)), None

    best, _ = jax.lax.scan(step, jnp.full(n_pts, jnp.inf, dtype=p.dtype), arrs)
    return best.reshape(p.shape[:-1])


# ── compile hook ──────────────────────────────────────────────────────────


def compile_vsweep(
    node: dict[str, Any], binding: ParamBinding, compile_child: Callable
) -> Callable:
    """The ``(points, free_vec) -> distance`` closure for a ``vsweep`` node.
    ``compile_child(tree, binding)`` compiles the 2-D profile."""
    kw = node.get("params") or {}
    if "path" not in kw:
        raise ValueError("vsweep requires a 'path' param: a list of [x, y, z] vertices")
    child_tree, along_np = split_along(node["child"])
    n_vertices = len(kw["path"]) if isinstance(kw["path"], (list, tuple)) else None
    if n_vertices is not None:
        for arr in along_np.values():
            if arr.shape[0] != n_vertices:
                raise ValueError(
                    f"vsweep: an '$along' list has {arr.shape[0]} values but the path has "
                    f"{n_vertices} vertices; give one value per vertex"
                )
    along_binding = _AlongBinding(binding, set(along_np))
    child = compile_child(child_tree, along_binding)
    closed = bool(kw.get("closed", False))  # structural, not param-resolved

    def profile(q2d, vals):
        # One point at a time under vmap: inside the child every along value
        # is then a plain scalar / vector of its declared shape, so the 2-D
        # primitives see exactly the parameter shapes they were written for.
        def one(q, vals_one):
            along_binding.current = vals_one
            return child(q[None, :], free_vec_holder[0])[0]

        return jax.vmap(one)(q2d, vals)

    free_vec_holder: list = [None]

    def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
        free_vec_holder[0] = free_vec
        path = resolve_param_value(kw["path"], binding, free_vec)
        up = resolve_param_value(kw["up"], binding, free_vec) if "up" in kw else None
        twist = resolve_param_value(kw["twist"], binding, free_vec) if "twist" in kw else None
        mitre = resolve_param_value(kw["mitre"], binding, free_vec) if "mitre" in kw else None
        return _evaluate(p, path, profile, along_np, up, twist, mitre, closed)

    return fn


# ── convenience: the rounded-rectangle conductor ──────────────────────────


def vsweep_distance(p, path, sections, up=None, closed=False, mitre=None, twist=None):
    """Distance to a rounded-rectangle section ``(half_w, half_h, corner_r)``
    per vertex swept along ``path``: the printed-conductor case, as a plain
    function of arrays. Builds the node and evaluates it."""
    from software_defined_matter.model import Part, sdf_primitive, sdf_vsweep
    from software_defined_matter.sdf.compile import make_sdf_closure

    sec = np.asarray(sections, dtype=float)
    profile = sdf_primitive(
        "rounded_box_2d", b={ALONG_KEY: sec[:, :2].tolist()}, r={ALONG_KEY: sec[:, 2].tolist()}
    )
    node = sdf_vsweep(
        profile,
        np.asarray(path, dtype=float).tolist(),
        up=None if up is None else np.asarray(up, dtype=float).tolist(),
        twist=None if twist is None else list(map(float, twist)),
        mitre=None if mitre is None else np.asarray(mitre, dtype=float).tolist(),
        closed=closed,
    )
    fn = make_sdf_closure(node, Part(name="vsweep", params={}, materials=[], metadata={}))
    return fn(jnp.asarray(p), None)
