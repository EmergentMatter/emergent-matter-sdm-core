"""How thin is the thinnest wall in this JSON tree?

The volume metrics sample on a regular grid and give each cell a filled
fraction of ``clip(0.5 - d/h, 0, 1)``, with ``h`` the cell edge length. That
estimator is exact for a single flat face pointing along a grid axis, because
the ramp is one cell wide and the per-cell values telescope. A wall has two
faces, and once they are closer than one cell they share a band: the distance
at a sample reports only the nearer one, so the telescoping breaks and the
wall is over-counted. Measured against an exact reference, a flat wall half a
cell thick reads up to +49%, and a wall at 45/45 degrees reads +12.9% at one
cell and +78.9% at a quarter.

So the grid has to resolve the geometry, and
:func:`infer_min_feature_size` is how a caller finds out whether it does. It
walks the tree and returns the smallest wall thickness it can *name*, or
``None`` when the tree contains nothing it knows how to measure.

Why "can name"
--------------
Finding the thinnest part of an arbitrary solid means sampling it, which is the
cost we are trying to decide whether to pay. This module reads thicknesses off
the nodes that carry one as a parameter instead: a TPMS sheet's
``min_thickness``, a shell's wall, a box's shortest side. Those cover the thin
features somebody put there on purpose, which are the ones a coarse grid
usually gets wrong.

``None`` therefore means "nothing here was measurable", which is weaker than
"nothing here is thin". Treat it as an absence of evidence.

Conservative in the safe direction
----------------------------------
Where a primitive's thinnest dimension depends on a parameter interval, the
smallest value the interval allows is used, so refining the grid enough for the
answer here is enough for every point a bounded optimiser may visit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from software_defined_matter.sdf.bbox import (
    Interval,
    build_param_intervals,
)

if TYPE_CHECKING:
    from software_defined_matter.model import Part, SDFTree


def infer_min_feature_size(tree: SDFTree, part: Part, *, mode: str = "values") -> float | None:
    """Smallest nameable wall thickness in ``tree``, in mm, or ``None``.

    ``mode`` follows :func:`software_defined_matter.sdf.bbox.infer_sdf_bbox`:
    ``'values'`` reads each param's current value, ``'bounds'`` takes the
    thinnest the param's bounds allow.

    Never raises on an unrecognised node. An unknown node contributes nothing
    and the walk continues, because the point of this is to find evidence that
    the grid is too coarse, and a node nobody has analysed is not evidence.
    """
    intervals = build_param_intervals(part, mode=mode)
    return _smallest(_node_features(tree, intervals))


# ---------------------------------------------------------------------------
# Internal: recursive tree walk
# ---------------------------------------------------------------------------


def _smallest(values: Iterable[float | None]) -> float | None:
    """Smallest of the known values, or ``None`` if there are none.

    Zero counts as a value: a feature that can shrink to 0 mm makes
    ``check_grid_resolution`` raise an error.
    """
    found = [v for v in values if v is not None]
    return min(found) if found else None


def _min_scalar(value: Any, intervals: dict[str, Interval]) -> float | None:
    """Smallest magnitude the scalar ``value`` can take, or ``None``.

    Mirrors ``bbox._scalar_abs_max`` but takes the *low* end: a wall is at its
    thinnest, and hardest to resolve, at the small end of its range.
    """
    from software_defined_matter.sdf.bbox import _scalar_interval

    if value is None:
        return None
    try:
        lo, hi = _scalar_interval(value, intervals)
    except Exception:  # noqa: BLE001
        # Deliberately broad. This is best-effort feature-size inference: any
        # value it cannot reduce to an interval means "unknown", and returning
        # None is the documented answer. Narrowing this would turn a graceful
        # unknown into a crash the first time an unanticipated node shape
        # reaches it, taking down bbox inference with it.
        return None
    return _min_scalar_from_interval(lo, hi)


def _min_vector(value: Any, n: int, intervals: dict[str, Interval]) -> float | None:
    from software_defined_matter.sdf.bbox import _vec_intervals

    if value is None:
        return None
    try:
        comps = _vec_intervals(value, n, intervals)
    except Exception:  # noqa: BLE001 -- see _min_scalar above; same contract
        return None
    return _smallest(_min_scalar_from_interval(lo, hi) for (lo, hi) in comps)


def _min_scalar_from_interval(lo: float, hi: float) -> float | None:
    if lo <= 0.0 <= hi:
        # The interval straddles zero, so the feature can vanish entirely.
        return 0.0
    return min(abs(lo), abs(hi))


def _thinning_factor(node: dict[str, Any], intervals: dict[str, Interval]) -> float | None:
    """How much a non-rigid wrapper can THIN a wall, as a factor <= 1, or None.

    A per-axis scale shrinks walls by at most its smallest factor;
    a taper also shears off-axis walls, requiring a bound on its inverse map;
    a shear by at most the inverse of its largest singular value,
    (|k| + sqrt(k^2 + 4)) / 2. Rigid motions and the twists preserve thickness.
    """
    from software_defined_matter.sdf.bbox import (
        _infer_node_bbox,
        _scalar_interval,
        _vec_intervals,
    )

    kw = node.get("params") or {}
    name = node.get("transform") if node.get("type") == "transform" else node.get("deform")
    try:
        if name == "scale_axis":
            axes = _vec_intervals(kw["s"], len(kw["s"]), intervals)
            return min(1.0, min(float(lo) for lo, _hi in axes))
        if name == "taper_linear":
            a = _scalar_interval(kw["s_0"], intervals)
            b = _scalar_interval(kw["s_1"], intervals)
            z0 = _scalar_interval(kw["z0"], intervals)
            z1 = _scalar_interval(kw["z1"], intervals)
            width = z1[0] - z0[1]
            s_min, s_max = min(a[0], b[0]), max(a[1], b[1])
            if not all(math.isfinite(v) for v in (*a, *b, *z0, *z1)):
                return 0.0
            if width <= 0.0 or s_min <= 0.0:
                return 0.0
            slope = max(abs(b[1] - a[0]), abs(a[1] - b[0])) / width
            if slope == 0.0:
                return min(1.0, s_min)
            lo, hi = _infer_node_bbox(node["child"], intervals)
            rho = s_max * math.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[1]), abs(hi[1])))
            # On the enclosing world box, D(x/s(z), y/s(z), z) has a
            # diagonal term plus a shear of norm rho * |s'| / s_min**2.
            # Its reciprocal bounds contraction even between opposite walls.
            rate = max(1.0, 1.0 / s_min) + rho * slope / (s_min * s_min)
            return 1.0 / rate if math.isfinite(rate) else 0.0
        if name == "shear_linear":
            u0, u1 = _scalar_interval(kw["u0"], intervals), _scalar_interval(kw["u1"], intervals)
            a, b = _scalar_interval(kw["dz_0"], intervals), _scalar_interval(kw["dz_1"], intervals)
            width = u1[0] - u0[1]
            if width <= 0:
                return 0.0
            k = max(abs(b[1] - a[0]), abs(a[1] - b[0])) / width
            return 2.0 / (k + math.sqrt(k * k + 4.0))
    except Exception:  # noqa: BLE001 - an unresolvable param bounds nothing; say so
        return 0.0
    return None


def _node_features(node: Any, intervals: dict[str, Interval]) -> list[float | None]:
    if not isinstance(node, dict) or "type" not in node:
        return []

    node_type = node["type"]

    if node_type == "primitive":
        return [_primitive_feature(node, intervals)]

    if node_type == "modifier":
        out = _node_features(node.get("child"), intervals)
        kw = node.get("params") or {}
        if node.get("modifier") == "onion":
            # `onion` turns a solid into a shell of this wall thickness. The
            # shell is the thinnest thing in the subtree by construction.
            out.append(_min_scalar(kw.get("thickness", kw.get("t")), intervals))
        return out

    if node_type in ("op", "loft"):
        out = []
        for child in node.get("children") or []:
            out.extend(_node_features(child, intervals))
        return out

    if node_type in ("transform", "deform", "2d_to_3d", "sweep", "vsweep"):
        # A rigid motion or a lift does not change how thick a wall is. `scale`
        # does, and is handled by scaling the child's features.
        child = _node_features(node.get("child"), intervals)
        if node_type == "transform" and node.get("transform") == "scale":
            s = _min_scalar((node.get("params") or {}).get("s"), intervals)
            if s is not None:
                return [None if c is None else c * s for c in child]
        factor = _thinning_factor(node, intervals)
        if factor is not None:
            return [None if c is None else c * factor for c in child]
        return child

    return []


def _primitive_feature(node: dict[str, Any], intervals: dict[str, Interval]) -> float | None:
    """Thinnest dimension of a single primitive, or ``None`` if not nameable.

    Only primitives whose thinness is stated directly as a parameter appear
    here. A ``sphere`` is listed because a small enough one is a sub-voxel
    feature in its own right; a ``cone`` is not, because its thin end is a
    point and no grid resolves that.
    """
    kind = node.get("kind")
    kw = node.get("params") or {}

    # Sheets: the thickness is the parameter.
    if kind in ("gyroid", "schwarz_p", "schwarz_d", "neovius", "lidinoid"):
        return _min_scalar(kw.get("min_thickness"), intervals)

    # Solids whose thinnest dimension is a named half-extent or radius. These
    # are doubled where the parameter is a half-extent, so the number returned
    # is always a full wall thickness.
    if kind in ("box", "round_box"):
        m = _min_vector(kw.get("b"), 3, intervals)
        return None if m is None else 2.0 * m
    if kind == "box_frame":
        return _min_scalar(kw.get("e"), intervals)
    if kind == "sphere":
        m = _min_scalar(kw.get("r"), intervals)
        return None if m is None else 2.0 * m
    if kind == "ellipsoid":
        m = _min_vector(kw.get("r"), 3, intervals)
        return None if m is None else 2.0 * m
    if kind == "capped_cylinder":
        r = _min_scalar(kw.get("r"), intervals)
        h = _min_scalar(kw.get("h"), intervals)
        return _smallest([None if r is None else 2.0 * r, None if h is None else 2.0 * h])
    if kind == "capsule":
        m = _min_scalar(kw.get("r"), intervals)
        return None if m is None else 2.0 * m
    if kind == "torus":
        m = _min_vector(kw.get("t"), 2, intervals)
        return None if m is None else 2.0 * m
    if kind == "leaf_spring":
        return _min_scalar(kw.get("thickness"), intervals)
    if kind == "serpentine":
        return _smallest(
            [
                _min_scalar(kw.get("beam_width"), intervals),
                _min_scalar(kw.get("beam_height"), intervals),
            ]
        )

    return None


__all__ = ["infer_min_feature_size"]
