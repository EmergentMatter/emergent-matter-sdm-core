"""Conservative axis-aligned bounding-box inference for SDF trees."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from software_defined_matter.model import MaterialRegion, Part, SDFTree


BBox = tuple[tuple[float, float, float], tuple[float, float, float]]
BBox2 = tuple[tuple[float, float], tuple[float, float]]
Interval = tuple[float, float]
IntervalVec = tuple[Interval, ...]


def _vec3(values: Iterable[float]) -> tuple[float, float, float]:
    """Narrow a length-3 float iterable to a fixed-size tuple.

    ``tuple(x for x in ...)`` only ever type-checks as ``tuple[float, ...]``,
    even when the generator is known (by construction) to yield exactly
    three items. Every BBox corner built from a comprehension routes through
    this helper so its result satisfies the ``BBox`` alias above.
    """
    a, b, c = values
    return (a, b, c)


class BBoxInferenceError(ValueError):
    """Base class for bbox inference failures."""


class UnboundedParamError(BBoxInferenceError):
    """Raised when a referenced parameter has no finite bounds."""

    def __init__(self, param_name: str):
        super().__init__(
            f"Cannot infer bbox: parameter {param_name!r} has bounds=None. Add finite Param.bounds."
        )


class UnsupportedExprError(BBoxInferenceError):
    """Raised when interval evaluation sees an unsupported expression node."""


class UnsupportedSDFNodeError(BBoxInferenceError):
    """Raised when a node kind/op/primitive is unsupported for bbox inference."""


def infer_material_bbox(region: MaterialRegion, part: Part) -> BBox:
    return infer_sdf_bbox(region.sdf_tree, part)


def infer_sdf_bbox(tree: SDFTree, part: Part, *, mode: str = "bounds") -> BBox:
    """Conservative AABB for ``tree``.

    ``mode='bounds'`` (default): worst-case box over ``Param.bounds``, the
    smallest box that contains the part for *every* point a bounded optimiser
    may visit. Trajectory-stable; loose when bounds are wide.

    ``mode='values'``: tight box at each param's *current* ``value``. Use for
    per-iteration metric sampling (must be padded by the caller and recomputed
    each outer optimiser step; see ``software_defined_matter.dsl.expr``).
    """
    return _infer_node_bbox(tree, build_param_intervals(part, mode=mode))


def build_param_intervals(part: Part, *, mode: str = "bounds") -> dict[str, Interval]:
    """Per-param intervals for interval arithmetic over an SDF tree.

    ``mode='bounds'`` gives each free param its full ``Param.bounds``;
    ``mode='values'`` pins every param to its current value. Shared with
    :mod:`software_defined_matter.sdf.lipschitz`, which walks the same trees
    with the same interval semantics.
    """
    if mode not in ("bounds", "values"):
        raise ValueError(f"mode must be 'bounds' or 'values', got {mode!r}")
    intervals: dict[str, Interval] = {}
    for name, param in part.params.items():
        if mode == "values":
            v = float(param.numeric_value())
            intervals[name] = (v, v)
            continue
        # POSE params (kinematic DOFs) are typically fixed with bounds=None,
        # but a viewer scrubs them across ui.explore_bounds: a bounds-mode
        # box must cover the whole pose envelope or AABB pruning amputates
        # deflected geometry (livewing panels swinging out of their rest
        # box).
        ui = getattr(param, "ui", None) or {}
        if ui.get("role") == "pose" and ui.get("explore_bounds"):
            lo, hi = (float(x) for x in ui["explore_bounds"])
            intervals[name] = (min(lo, hi), max(lo, hi))
            continue
        if param.bounds is not None:
            lo, hi = float(param.bounds[0]), float(param.bounds[1])
            intervals[name] = (min(lo, hi), max(lo, hi))
            continue
        # Fixed params without optimisation bounds still have a finite value.
        if not param.free:
            v = float(param.numeric_value())
            intervals[name] = (v, v)
    return intervals


def pad_bbox(bbox: BBox, margin: float) -> BBox:
    """Inflate ``bbox`` outward by an isotropic ``margin`` on every face.

    Used by metric sampling to keep the soft-Heaviside transition band and any
    CSG smoothing radius from being clipped at the box face (which would bias
    ``volume`` / ``surface_area`` / ``mass`` and add a discontinuity to the
    gradient at the boundary).
    """
    m = abs(float(margin))
    return _inflate_bbox(bbox, (m, m, m))


def _infer_node_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox:
    if not isinstance(node, dict) or "type" not in node:
        raise UnsupportedSDFNodeError(f"Not an SDF node: {node!r}")

    node_type = node["type"]
    if node_type == "primitive":
        return _infer_primitive_bbox(node, intervals)
    if node_type == "op":
        return _infer_op_bbox(node, intervals)
    if node_type == "transform":
        return _infer_transform_bbox(node, intervals)
    if node_type == "modifier":
        return _infer_modifier_bbox(node, intervals)
    if node_type == "2d_to_3d":
        return _infer_2d_to_3d_bbox(node, intervals)
    if node_type == "loft":
        return _infer_loft_bbox(node, intervals)
    if node_type == "deform":
        return _infer_deform_bbox(node, intervals)
    if node_type == "sweep":
        return _infer_sweep_bbox(node, intervals)
    raise UnsupportedSDFNodeError(f"Unknown node type {node_type!r}")


def _primitive_kwargs(node: dict[str, Any]) -> dict[str, Any]:
    return node.get("params") or {}


def _infer_primitive_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox:
    kind = node["kind"]
    kw = _primitive_kwargs(node)

    if kind == "raster_field":
        # Negative boundary samples continue beyond the sample domain.
        from software_defined_matter.sdf.raster import (
            decode_raster_values,
            raster_bbox,
            require_literal_raster_params,
        )

        require_literal_raster_params(kw)
        lo, hi = raster_bbox(kw)
        values = decode_raster_values(kw)
        boundary_min = min(
            float(face.min())
            for face in (
                values[0],
                values[-1],
                values[:, 0],
                values[:, -1],
                values[:, :, 0],
                values[:, :, -1],
            )
        )
        margin = max(0.0, -boundary_min)
        return pad_bbox((lo, hi), margin)

    if kind == "sphere":
        r = _scalar_abs_max(kw["r"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (r, r, r))
    if kind == "box":
        hx, hy, hz = _vec_abs_max(kw["b"], 3, intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (hx, hy, hz))
    if kind == "round_box":
        hx, hy, hz = _vec_abs_max(kw["b"], 3, intervals)
        r = _scalar_abs_max(kw["r"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (hx + r, hy + r, hz + r))
    if kind == "box_frame":
        hx, hy, hz = _vec_abs_max(kw["b"], 3, intervals)
        e = _scalar_abs_max(kw["e"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (hx + e, hy + e, hz + e))
    if kind == "torus":
        major, minor = _vec_abs_max(kw["t"], 2, intervals)
        radial = major + minor
        return _bbox_center_half((0.0, 0.0, 0.0), (radial, radial, minor))
    if kind == "capped_torus":
        ra = _scalar_abs_max(kw["ra"], intervals)
        rb = _scalar_abs_max(kw["rb"], intervals)
        radial = ra + rb
        return _bbox_center_half((0.0, 0.0, 0.0), (radial, radial, rb))
    if kind == "helix":
        major = _scalar_abs_max(kw["major_r"], intervals)
        pitch = _scalar_abs_max(kw["pitch"], intervals)
        r = _scalar_abs_max(kw["r"], intervals)
        n_turns = _scalar_abs_max(kw["n_turns"], intervals)
        radial = major + r
        # Tight and analytic: the flat-cut band is |z| <= n_turns*|pitch|/2 and
        # the tube adds r on top. This tightness is the whole point of a helix
        # primitive: the same shape as a `sweep` only gets the (much looser)
        # control-point hull.
        return _bbox_center_half((0.0, 0.0, 0.0), (radial, radial, 0.5 * n_turns * pitch + r))
    if kind == "screw_thread":
        r_root = _scalar_abs_max(kw["r_root"], intervals)
        depth = _scalar_abs_max(kw["depth"], intervals)
        pitch = _scalar_abs_max(kw["pitch"], intervals)
        n_turns = _scalar_abs_max(kw["n_turns"], intervals)
        radial = r_root + depth
        return _bbox_center_half((0.0, 0.0, 0.0), (radial, radial, 0.5 * n_turns * pitch))
    if kind == "link":
        le = _scalar_abs_max(kw["le"], intervals)
        r1 = _scalar_abs_max(kw["r1"], intervals)
        r2 = _scalar_abs_max(kw["r2"], intervals)
        ext = le + r1 + r2
        return _bbox_center_half((0.0, 0.0, 0.0), (r1 + r2, ext, r2))
    if kind == "cone":
        h = _scalar_abs_max(kw["h"], intervals)
        c = _vec_abs_max(kw["c"], 2, intervals)
        tan_term = c[0] / max(c[1], 1e-8)
        r = abs(h * tan_term)
        return _bbox_center_half((0.0, 0.0, 0.0), (r, r, h))
    if kind == "hex_prism":
        radius, half_h = _vec_abs_max(kw["h"], 2, intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (radius, radius, half_h))
    if kind == "tri_prism":
        radius, half_h = _vec_abs_max(kw["h"], 2, intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (radius, radius, half_h))
    if kind == "capsule":
        a = _vec_intervals(kw["a"], 3, intervals)
        b = _vec_intervals(kw["b"], 3, intervals)
        r = _scalar_abs_max(kw["r"], intervals)
        lo = _vec3(min(a[i][0], b[i][0]) - r for i in range(3))
        hi = _vec3(max(a[i][1], b[i][1]) + r for i in range(3))
        return lo, hi
    if kind == "capped_cylinder":
        h = _scalar_abs_max(kw["h"], intervals)
        r = _scalar_abs_max(kw["r"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (r, r, h))
    if kind == "rounded_cylinder":
        ra = _scalar_abs_max(kw["ra"], intervals)
        rb = _scalar_abs_max(kw["rb"], intervals)
        h = _scalar_abs_max(kw["h"], intervals)
        radial = max(2.0 * ra, rb)
        return _bbox_center_half((0.0, 0.0, 0.0), (radial, radial, h + rb))
    if kind == "capped_cone":
        h = _scalar_abs_max(kw["h"], intervals)
        r1 = _scalar_abs_max(kw["r1"], intervals)
        r2 = _scalar_abs_max(kw["r2"], intervals)
        r = max(r1, r2)
        return _bbox_center_half((0.0, 0.0, 0.0), (r, r, h))
    if kind == "solid_angle":
        ra = _scalar_abs_max(kw["ra"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (ra, ra, ra))
    if kind == "cut_sphere":
        r = _scalar_abs_max(kw["r"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (r, r, r))
    if kind == "ellipsoid":
        rx, ry, rz = _vec_abs_max(kw["r"], 3, intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (rx, ry, rz))
    if kind == "octahedron":
        s = _scalar_abs_max(kw["s"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (s, s, s))
    if kind == "pyramid":
        h = _scalar_abs_max(kw["h"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (0.5, h, 0.5))
    if kind == "notch_hinge":
        width = _scalar_abs_max(kw["width"], intervals)
        depth = _scalar_abs_max(kw["depth"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (width / 2.0, depth / 2.0, width / 2.0))
    if kind == "leaf_spring":
        length = _scalar_abs_max(kw["length"], intervals)
        width = _scalar_abs_max(kw["width"], intervals)
        thickness = _scalar_abs_max(kw["thickness"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (length / 2.0, thickness / 2.0, width / 2.0))
    if kind == "bellows":
        outer_r = _scalar_abs_max(kw["outer_r"], intervals)
        period = _scalar_abs_max(kw["period"], intervals)
        n_periods = _scalar_abs_max(kw["n_periods"], intervals)
        half_h = 0.5 * period * n_periods
        return _bbox_center_half((0.0, 0.0, 0.0), (outer_r, outer_r, half_h))
    if kind == "serpentine":
        amplitude = _scalar_abs_max(kw["amplitude"], intervals)
        wave = _scalar_abs_max(kw["wavelength"], intervals)
        beam_w = _scalar_abs_max(kw["beam_width"], intervals)
        beam_h = _scalar_abs_max(kw["beam_height"], intervals)
        n_periods = _scalar_abs_max(kw["n_periods"], intervals)
        half_x = 0.5 * n_periods * wave
        half_y = amplitude + 0.5 * beam_w
        return _bbox_center_half((0.0, 0.0, 0.0), (half_x, half_y, beam_h / 2.0))
    if kind == "annular_sector":
        outer_r = _scalar_abs_max(kw["outer_r"], intervals)
        height = _scalar_abs_max(kw["height"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (outer_r, outer_r, height / 2.0))
    if kind in ("gyroid", "schwarz_p", "schwarz_d", "neovius", "lidinoid"):
        # TPMS shapes are clipped to a box of half-extents
        # `0.5 * n_periods * period` (see sdf_shapes._tpms_clip).
        nx, ny, nz = _vec_abs_max(kw["n_periods"], 3, intervals)
        s = _scalar_abs_max(kw["period"], intervals)
        return _bbox_center_half((0.0, 0.0, 0.0), (0.5 * nx * s, 0.5 * ny * s, 0.5 * nz * s))

    # Explicitly unbounded (`plane`) or only valid wrapped in a 2d_to_3d
    # node (2D primitives). Neither has a meaningful 3D bbox on its own.
    if kind in (
        "plane",
        "circle_2d",
        "box_2d",
        "rounded_box_2d",
        "segment_2d",
        "trapezoid_2d",
        "uneven_capsule_2d",
        "bezier_2d",
        "bspline_2d",
    ):
        raise UnsupportedSDFNodeError(
            f"Primitive {kind!r} is unsupported/unbounded for bbox inference"
        )
    raise UnsupportedSDFNodeError(f"Unknown primitive {kind!r}")


def _infer_op_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox:
    op = node["op"]
    children = node["children"]
    if not children:
        raise UnsupportedSDFNodeError("CSG op has no children")

    if op in ("subtract", "smooth_subtract"):
        # Subtract is bounded by the minuend; the subtrahend may be
        # unbounded (e.g. a half-space `plane` used as a cutting tool), so
        # only the first child is inferred.
        return _infer_node_bbox(children[0], intervals)

    if op in ("union", "smooth_union"):
        return _bbox_enclose(_infer_node_bbox(c, intervals) for c in children)

    if op in ("intersect", "smooth_intersect"):
        # Intersect is bounded iff at least one child is bounded: that
        # child's box already contains the result. Collect successes, take
        # their AABB intersection; raise only if every child is unbounded.
        bounded: list[BBox] = []
        last_err: BBoxInferenceError | None = None
        for c in children:
            try:
                bounded.append(_infer_node_bbox(c, intervals))
            except BBoxInferenceError as exc:
                last_err = exc
        if not bounded:
            raise UnsupportedSDFNodeError(
                "CSG intersect has no analytically-bounded children"
            ) from last_err
        return _bbox_intersect(bounded)

    raise UnsupportedSDFNodeError(f"Unsupported op {op!r} for bbox inference")


def _infer_transform_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox:
    tf = node["transform"]
    child_bbox = _infer_node_bbox(node["child"], intervals)
    params = node.get("params", {}) or {}

    if tf == "translate":
        t = _vec_intervals(params["t"], 3, intervals)
        return (
            _vec3(child_bbox[0][i] + t[i][0] for i in range(3)),
            _vec3(child_bbox[1][i] + t[i][1] for i in range(3)),
        )
    if tf == "scale":
        s_lo, s_hi = _scalar_interval(params["s"], intervals)
        if s_lo <= 0.0 <= s_hi:
            raise UnsupportedSDFNodeError(
                "Scale interval spans zero; unsupported for MVP bbox inference"
            )
        s_abs = max(abs(s_lo), abs(s_hi))
        corners = _bbox_corners(child_bbox)
        scaled = [_vec3(s_abs * c for c in corner) for corner in corners]
        return _points_to_bbox(scaled)
    if tf == "scale_axis":
        # Each axis scales independently, across the whole bounds of its
        # factor, so a live factor cannot let the box clip the part.
        s = _vec_intervals(params["s"], 3, intervals)
        if any(lo <= 0.0 for lo, _hi in s):
            raise UnsupportedSDFNodeError("scale_axis factors must stay positive")
        lo_out, hi_out = [], []
        for i in range(3):
            ends = [child_bbox[j][i] * f for j in (0, 1) for f in s[i]]
            lo_out.append(min(ends))
            hi_out.append(max(ends))
        return (_vec3(lo_out), _vec3(hi_out))
    if tf in ("rotate_x", "rotate_y", "rotate_z"):
        if isinstance(params["angle"], (int, float)):
            return _rotate_bbox(child_bbox, tf, float(params["angle"]))
        # Param-dependent angle (e.g. a live wing-panel twist DOF): sweep
        # the corners over the angle's interval: endpoint positions plus
        # any interior cos/sin extremum, so a small twist range stays a
        # TIGHT box instead of a full disc. Falls back to the angle-free
        # swept-disc envelope when the param has no usable bounds.
        try:
            a_lo, a_hi = _scalar_interval(params["angle"], intervals)
        except BBoxInferenceError:
            return _swept_rotation_bbox(child_bbox, tf)
        return _interval_rotation_bbox(child_bbox, tf, a_lo, a_hi)
    if tf == "rotate_matrix":
        matrix = _require_constant(params["R"])
        return _rotate_matrix_bbox(child_bbox, matrix)
    if tf == "canonical_sector_fold":
        # The fold makes the resulting SDF N-fold rotationally symmetric
        # about the Z axis, so the solid it represents occupies the
        # polar-array of the child geometry. The conservative axis-aligned
        # envelope is a square in XY of half-extent R = max distance from
        # the Z axis to any XY corner of the child's bbox. Z extent is
        # unchanged (the fold doesn't touch Z). This bound is valid for any
        # ``n_sectors`` because the swept envelope grows monotonically with
        # the number of sectors and saturates at the full disc as
        # n_sectors → ∞; the disc bound therefore covers every finite N.
        (cx0, cy0, cz0), (cx1, cy1, cz1) = child_bbox
        R = max(
            math.hypot(cx0, cy0),
            math.hypot(cx0, cy1),
            math.hypot(cx1, cy0),
            math.hypot(cx1, cy1),
        )
        return ((-R, -R, cz0), (R, R, cz1))
    if tf == "mirror":
        n = _require_constant(params["n"])
        o = _require_constant(params["o"])
        return _mirror_bbox(child_bbox, n, o)
    if tf == "repeat_inf":
        raise UnsupportedSDFNodeError(f"Transform {tf!r} is unsupported for bbox inference")
    if tf == "repeat_finite":
        # op_repeat_finite tiles the child SDF on integer-index cells
        # `clamp(round(p/c), -l, l)` along each axis. The solid region thus
        # spans `child_bbox` inflated by `c * l[i]` in each axis direction.
        c = _scalar_abs_max(params["c"], intervals)
        limits = _vec_abs_max(params["l"], 3, intervals)
        return _inflate_bbox(child_bbox, tuple(c * li for li in limits))
    raise UnsupportedSDFNodeError(f"Unknown transform {tf!r}")


def _infer_modifier_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox:
    mod = node["modifier"]
    child_bbox = _infer_node_bbox(node["child"], intervals)
    params = node.get("params", {}) or {}
    if mod == "round":
        r = _scalar_abs_max(params["r"], intervals)
        return _inflate_bbox(child_bbox, (r, r, r))
    if mod == "onion":
        return child_bbox
    if mod == "elongate":
        h = _vec_abs_max(params["h"], 3, intervals)
        return _inflate_bbox(child_bbox, h)
    raise UnsupportedSDFNodeError(f"Unknown modifier {mod!r}")


def _infer_2d_to_3d_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox:
    """3D bbox for an ``extrusion`` or ``revolution`` node.

    The child must be a 2D primitive (the 2D-to-3D lift is the only place a
    2D SDF makes sense). 2D primitives are resolved via
    :func:`_infer_2d_primitive_bbox`.
    """
    method = node["method"]
    params = node.get("params", {}) or {}
    (xlo, ylo), (xhi, yhi) = _infer_2d_primitive_bbox(node["child"], intervals)

    if method == "extrusion":
        h = _scalar_abs_max(params["h"], intervals)
        return ((xlo, ylo, -h), (xhi, yhi, h))

    if method == "revolution":
        offset = _scalar_abs_max(params.get("offset", 0.0), intervals)
        # revolution evaluates the 2D SDF at q = (|p_xy| - offset, p_z), so
        # the 2D x-coord is the radial distance shifted by offset. The 3D
        # XY bbox is a square centred on the Z axis with half-side equal to
        # the largest |R| in the profile: max(|x_lo|, |x_hi|) + offset
        # (conservative for profiles spanning the axis).
        r_max = max(abs(xlo), abs(xhi)) + offset
        return ((-r_max, -r_max, ylo), (r_max, r_max, yhi))

    raise UnsupportedSDFNodeError(f"Unknown 2d_to_3d method {method!r}")


def _infer_loft_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox:
    """3D bbox for a ``loft``: union of the children's 2D AABBs (XY) crossed with
    the Z span of the section stations in ``params['z']``."""
    params = node.get("params", {}) or {}
    boxes = [_infer_2d_primitive_bbox(c, intervals) for c in node["children"]]
    xlo = min(b[0][0] for b in boxes)
    ylo = min(b[0][1] for b in boxes)
    xhi = max(b[1][0] for b in boxes)
    yhi = max(b[1][1] for b in boxes)
    zis = [_scalar_interval(z, intervals) for z in params["z"]]
    zlo = min(a for a, _ in zis)
    zhi = max(b for _, b in zis)
    return ((xlo, ylo, zlo), (xhi, yhi, zhi))


def _infer_2d_primitive_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox2:
    """2D AABB for a 2D-primitive node, returned as ``((xlo, ylo), (xhi, yhi))``."""
    if not isinstance(node, dict) or node.get("type") != "primitive":
        raise UnsupportedSDFNodeError(f"2d_to_3d child must be a 2D primitive node, got {node!r}")
    kind = node["kind"]
    kw = node.get("params") or {}

    if kind == "polygon_2d":
        verts = kw["vertices"]
        xi = [_scalar_interval(v[0], intervals) for v in verts]
        yi = [_scalar_interval(v[1], intervals) for v in verts]
        return (
            (min(a for a, _ in xi), min(a for a, _ in yi)),
            (max(b for _, b in xi), max(b for _, b in yi)),
        )
    if kind == "circle_2d":
        r = _scalar_abs_max(kw["r"], intervals)
        return ((-r, -r), (r, r))
    if kind == "box_2d":
        bx, by = _vec_abs_max(kw["b"], 2, intervals)
        return ((-bx, -by), (bx, by))
    if kind == "rounded_box_2d":
        bx, by = _vec_abs_max(kw["b"], 2, intervals)
        r = _scalar_abs_max(kw["r"], intervals)
        return ((-bx - r, -by - r), (bx + r, by + r))
    if kind == "segment_2d":
        a_iv = _vec_intervals(kw["a"], 2, intervals)
        b_iv = _vec_intervals(kw["b"], 2, intervals)
        lo = (min(a_iv[0][0], b_iv[0][0]), min(a_iv[1][0], b_iv[1][0]))
        hi = (max(a_iv[0][1], b_iv[0][1]), max(a_iv[1][1], b_iv[1][1]))
        return (lo, hi)
    if kind == "trapezoid_2d":
        r1 = _scalar_abs_max(kw["r1"], intervals)
        r2 = _scalar_abs_max(kw["r2"], intervals)
        he = _scalar_abs_max(kw["he"], intervals)
        r = max(r1, r2)
        return ((-r, -he), (r, he))
    if kind == "uneven_capsule_2d":
        # SDF uses q = [|p_x|, p_y]; caps at (0, 0) radius r1 and (0, h) radius r2.
        r1 = _scalar_abs_max(kw["r1"], intervals)
        r2 = _scalar_abs_max(kw["r2"], intervals)
        h = _scalar_abs_max(kw["h"], intervals)
        x_max = max(r1, r2)
        return ((-x_max, -r1), (x_max, h + r2))
    if kind == "polygon_2d":
        # AABB is just the min/max over the vertices. Each coord may be a
        # scalar/$ref/expr, so take its interval and enclose both endpoints.
        verts = kw["vertices"]
        if not isinstance(verts, (list, tuple)) or len(verts) < 3:
            raise UnsupportedSDFNodeError(f"polygon_2d needs >=3 [x, y] vertices, got {verts!r}")
        xs = [_scalar_interval(v[0], intervals) for v in verts]
        ys = [_scalar_interval(v[1], intervals) for v in verts]
        return (
            (min(lo for lo, _ in xs), min(lo for lo, _ in ys)),
            (max(hi for _, hi in xs), max(hi for _, hi in ys)),
        )
    if kind in ("bezier_2d", "bspline_2d"):
        # Both curves lie inside the convex hull of their control points, so the
        # control-point AABB is a valid (slightly loose) 2D bound.
        cps = kw.get("control_points")
        if not isinstance(cps, (list, tuple)) or len(cps) == 0:
            raise UnsupportedSDFNodeError(
                f"{kind} needs a non-empty 'control_points' list for bbox inference"
            )
        ivs = [_vec_intervals(pt, 2, intervals) for pt in cps]
        xlo = min(iv[0][0] for iv in ivs)
        xhi = max(iv[0][1] for iv in ivs)
        ylo = min(iv[1][0] for iv in ivs)
        yhi = max(iv[1][1] for iv in ivs)
        return ((xlo, ylo), (xhi, yhi))
    raise UnsupportedSDFNodeError(f"Unsupported 2D primitive {kind!r}")


def _infer_sweep_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox:
    """3D bbox for a ``sweep`` node: the path's control-point hull inflated by
    the profile's in-plane extent. Conservative: the moving frame can orient
    the cross-section any way, so the farthest profile corner is added on every
    axis (the swept curve also lies inside its control-point hull)."""
    params = node.get("params") or {}
    cps = params.get("path")
    if not isinstance(cps, (list, tuple)) or not cps:
        raise UnsupportedSDFNodeError(
            "sweep needs a non-empty 'path' control-point list for bbox inference"
        )
    ivs = [_vec_intervals(pt, 3, intervals) for pt in cps]
    lo = [min(iv[a][0] for iv in ivs) for a in range(3)]
    hi = [max(iv[a][1] for iv in ivs) for a in range(3)]
    (pxlo, pylo), (pxhi, pyhi) = _infer_2d_primitive_bbox(node["child"], intervals)
    rad = math.sqrt(max(abs(pxlo), abs(pxhi)) ** 2 + max(abs(pylo), abs(pyhi)) ** 2)
    return (_vec3(lo[a] - rad for a in range(3)), _vec3(hi[a] + rad for a in range(3)))


def _infer_deform_bbox(node: dict[str, Any], intervals: dict[str, Interval]) -> BBox:
    """Conservative bbox for ``twist`` / ``bend`` / ``displace`` deforms.

    - ``twist`` rotates the xz plane around the Y axis; rotation preserves
      ``|p_xz|``, so the bent solid lies inside the disk of radius
      ``sqrt(max_x² + max_z²)``. Y is unchanged.
    - ``bend`` rotates the xy plane around the Z axis; symmetric. Z is
      unchanged.
    - ``twist_radial`` / ``twist_linear`` rotate about Z by an interpolated
      angle. Whatever the profile does, every point stays on its own circle
      about Z, so the child's max XY radius is invariant and Z is untouched.
      The bound therefore does not depend on the end angles at all -- which is
      what makes it safe under live pose params, whose values are not known
      when the box is inferred.
    - ``displace`` adds a scalar field; the solid is contained in the
      ``sup|field|``-shell around the original, so the bbox inflates by
      that amplitude on every axis.

    The twist/bend bounds are independent of the rate ``k``: they assume
    the worst case where the bend sweeps a full revolution. Use the
    numeric shrinkwrap (``software_defined_matter.bbox``) for a tighter
    box when ``k`` is small.
    """
    deform = node["deform"]
    child_bbox = _infer_node_bbox(node["child"], intervals)
    (xlo, ylo, zlo), (xhi, yhi, zhi) = child_bbox

    if deform == "twist":
        x_max = max(abs(xlo), abs(xhi))
        z_max = max(abs(zlo), abs(zhi))
        r = math.sqrt(x_max * x_max + z_max * z_max)
        return ((-r, ylo, -r), (r, yhi, r))

    if deform == "bend":
        x_max = max(abs(xlo), abs(xhi))
        y_max = max(abs(ylo), abs(yhi))
        r = math.sqrt(x_max * x_max + y_max * y_max)
        return ((-r, -r, zlo), (r, r, zhi))

    if deform in ("twist_radial", "twist_linear"):
        # Rotation about Z: radial extent invariant, z untouched. Conservative
        # square over the child's max XY radius -- the same argument `bend`
        # makes, and equally independent of the angles.
        x_max = max(abs(xlo), abs(xhi))
        y_max = max(abs(ylo), abs(yhi))
        r = math.sqrt(x_max * x_max + y_max * y_max)
        return ((-r, -r, zlo), (r, r, zhi))

    if deform == "taper_linear":
        # XY grows by at most the largest factor anywhere on the ramp; Z is
        # untouched. Bounds, not values, for the same live-drag reason.
        kw = node.get("params") or {}
        factors = [v for key in ("s_0", "s_1") for v in _scalar_interval(kw[key], intervals)]
        if min(factors) <= 0.0:
            raise UnsupportedSDFNodeError("taper_linear factors must stay positive")
        f = max(factors)
        x_max = max(abs(xlo), abs(xhi)) * f
        y_max = max(abs(ylo), abs(yhi)) * f
        return ((-x_max, -y_max, zlo), (x_max, y_max, zhi))

    if deform == "shear_linear":
        # Material only rises or falls along Z, by an amount between the two
        # end rises; XY is untouched. Bounds, not values: a live dz param
        # must not let the box clip the part mid-drag.
        kw = node.get("params") or {}
        rises = [v for key in ("dz_0", "dz_1") for v in _scalar_interval(kw[key], intervals)]
        return ((xlo, ylo, zlo + min(rises)), (xhi, yhi, zhi + max(rises)))

    if deform == "displace":
        field = node.get("field")
        if field is None:
            raise UnsupportedSDFNodeError(
                "deform 'displace' requires a 'field' subtree on the node"
            )
        amplitude = _infer_field_amplitude(field, intervals)
        return _inflate_bbox(child_bbox, (amplitude, amplitude, amplitude))

    raise UnsupportedSDFNodeError(f"Unknown deform {deform!r}")


def _infer_field_amplitude(node: Any, intervals: dict[str, Interval]) -> float:
    """Upper bound on ``|field(p)|`` for the displacement-field DSL.

    Field primitives are sinusoidal and clamped by ``amplitude``; combinators
    are bounded by the triangle inequality.
    """
    if not isinstance(node, dict) or "type" not in node:
        raise UnsupportedSDFNodeError(f"Not a field node: {node!r}")

    node_type = node["type"]
    if node_type == "field":
        kind = node["kind"]
        kw = node.get("params") or {}
        if kind in ("sin_xyz", "radial", "angular"):
            return _scalar_abs_max(kw.get("amplitude", 1.0), intervals)
        raise UnsupportedSDFNodeError(f"Unknown field primitive {kind!r}")

    if node_type == "field_op":
        op = node["op"]
        children = node.get("children", []) or []
        if op == "add":
            return sum(_infer_field_amplitude(c, intervals) for c in children)
        raise UnsupportedSDFNodeError(f"Unknown field_op {op!r}")

    raise UnsupportedSDFNodeError(f"Unknown field node type {node_type!r}")


def _scalar_interval(value: Any, intervals: dict[str, Interval]) -> Interval:
    if isinstance(value, bool):
        return (float(value), float(value))
    if isinstance(value, (int, float)):
        f = float(value)
        return (f, f)
    if isinstance(value, dict):
        if "$ref" in value:
            name = value["$ref"]
            if name not in intervals:
                raise UnboundedParamError(name)
            return intervals[name]
        node_type = value.get("type")
        if node_type == "num":
            f = float(value["value"])
            return (f, f)
        if node_type == "param":
            name = value["name"]
            if name not in intervals:
                raise UnboundedParamError(name)
            return intervals[name]
        if node_type == "unop":
            child = _scalar_interval(value["child"], intervals)
            return _interval_unop(value["op"], child)
        if node_type == "binop":
            lhs = _scalar_interval(value["lhs"], intervals)
            rhs = _scalar_interval(value["rhs"], intervals)
            return _interval_binop(value["op"], lhs, rhs)
        if node_type == "reduce":
            children = [_scalar_interval(c, intervals) for c in value["children"]]
            op = value["op"]
            if op == "sum":
                return (sum(c[0] for c in children), sum(c[1] for c in children))
            if op == "mean":
                n = max(len(children), 1)
                return (sum(c[0] for c in children) / n, sum(c[1] for c in children) / n)
            if op == "min":
                return (min(c[0] for c in children), min(c[1] for c in children))
            if op == "max":
                return (max(c[0] for c in children), max(c[1] for c in children))
            raise UnsupportedExprError(f"Unsupported reduce op {op!r} in interval inference")
    raise UnsupportedExprError(f"Unsupported scalar expression in interval inference: {value!r}")


def _vec_intervals(value: Any, n: int, intervals: dict[str, Interval]) -> IntervalVec:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise UnsupportedExprError(f"Expected vector length {n}, got {len(value)}")
        return tuple(_scalar_interval(v, intervals) for v in value)
    scalar = _scalar_interval(value, intervals)
    return tuple(scalar for _ in range(n))


def _scalar_abs_max(value: Any, intervals: dict[str, Interval]) -> float:
    lo, hi = _scalar_interval(value, intervals)
    return max(abs(lo), abs(hi))


def _vec_abs_max(value: Any, n: int, intervals: dict[str, Interval]) -> tuple[float, ...]:
    vec = _vec_intervals(value, n, intervals)
    return tuple(max(abs(lo), abs(hi)) for lo, hi in vec)


def _interval_unop(op: str, child: Interval) -> Interval:
    lo, hi = child
    if op == "neg":
        return (-hi, -lo)
    if op == "abs":
        if lo <= 0.0 <= hi:
            return (0.0, max(abs(lo), abs(hi)))
        return (min(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
    if op == "square":
        if lo <= 0.0 <= hi:
            return (0.0, max(lo * lo, hi * hi))
        return (min(lo * lo, hi * hi), max(lo * lo, hi * hi))
    if op == "sqrt":
        if lo < 0.0:
            raise UnsupportedExprError("sqrt over interval crossing negatives is unsupported")
        return (math.sqrt(lo), math.sqrt(hi))
    raise UnsupportedExprError(f"Unsupported unary op {op!r} in interval inference")


def _interval_binop(op: str, lhs: Interval, rhs: Interval) -> Interval:
    a0, a1 = lhs
    b0, b1 = rhs
    if op == "+":
        return (a0 + b0, a1 + b1)
    if op == "-":
        return (a0 - b1, a1 - b0)
    if op == "*":
        products = (a0 * b0, a0 * b1, a1 * b0, a1 * b1)
        return (min(products), max(products))
    if op == "/":
        if b0 <= 0.0 <= b1:
            raise UnsupportedExprError("Division by interval spanning zero is unsupported")
        quotients = (a0 / b0, a0 / b1, a1 / b0, a1 / b1)
        return (min(quotients), max(quotients))
    if op == "min":
        return (min(a0, b0), min(a1, b1))
    if op == "max":
        return (max(a0, b0), max(a1, b1))
    if op == "pow":
        if b0 != b1:
            raise UnsupportedExprError("pow with non-constant exponent is unsupported")
        p = b0
        if float(p).is_integer():
            vals = (a0**p, a1**p)
            if int(p) % 2 == 0 and a0 <= 0.0 <= a1:
                return (0.0, max(vals))
            return (min(vals), max(vals))
        raise UnsupportedExprError("pow with non-integer exponent is unsupported")
    raise UnsupportedExprError(f"Unsupported binary op {op!r} in interval inference")


def _require_constant(value: Any) -> Any:
    if isinstance(value, (int, float, list, tuple)):
        return value
    raise UnsupportedExprError("Param-dependent rotation is unsupported in MVP bbox inference")


def _bbox_center_half(center: Sequence[float], half: Sequence[float]) -> BBox:
    lo = _vec3(float(center[i] - abs(half[i])) for i in range(3))
    hi = _vec3(float(center[i] + abs(half[i])) for i in range(3))
    return lo, hi


def _inflate_bbox(bbox: BBox, delta: Sequence[float]) -> BBox:
    lo, hi = bbox
    return (
        _vec3(float(lo[i] - abs(delta[i])) for i in range(3)),
        _vec3(float(hi[i] + abs(delta[i])) for i in range(3)),
    )


def _bbox_enclose(boxes: Iterable[BBox]) -> BBox:
    boxes_list = list(boxes)
    lo = _vec3(min(b[0][i] for b in boxes_list) for i in range(3))
    hi = _vec3(max(b[1][i] for b in boxes_list) for i in range(3))
    return (lo, hi)


def _bbox_intersect(boxes: Iterable[BBox]) -> BBox:
    """Tightest AABB containing the geometric intersection of the inputs.

    Per-axis max-of-mins and min-of-maxes. Raises if any axis has min > max
    (the inputs' AABBs do not overlap, so the intersection SDF is empty).
    """
    boxes_list = list(boxes)
    lo = _vec3(max(b[0][i] for b in boxes_list) for i in range(3))
    hi = _vec3(min(b[1][i] for b in boxes_list) for i in range(3))
    if any(lo[i] > hi[i] for i in range(3)):
        raise UnsupportedSDFNodeError(
            "CSG intersect has empty AABB overlap: children do not intersect"
        )
    return (lo, hi)


def _bbox_corners(bbox: BBox) -> list[tuple[float, float, float]]:
    (x0, y0, z0), (x1, y1, z1) = bbox
    return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


def _points_to_bbox(points: Iterable[tuple[float, float, float]]) -> BBox:
    pts = list(points)
    lo = _vec3(min(p[i] for p in pts) for i in range(3))
    hi = _vec3(max(p[i] for p in pts) for i in range(3))
    return (lo, hi)


def _phase_in_interval(t: float, lo: float, hi: float) -> bool:
    """True if t + 2πk lies in [lo, hi] for some integer k."""
    two_pi = 2.0 * math.pi
    k = math.ceil((lo - t) / two_pi)
    return t + k * two_pi <= hi


def _cos_range(r: float, alpha: float, lo: float, hi: float) -> Interval:
    """Range of r*cos(φ + alpha) over φ ∈ [lo, hi]."""
    vals = [r * math.cos(lo + alpha), r * math.cos(hi + alpha)]
    if _phase_in_interval(-alpha, lo, hi):
        vals.append(r)
    if _phase_in_interval(math.pi - alpha, lo, hi):
        vals.append(-r)
    return (min(vals), max(vals))


def _interval_rotation_bbox(bbox: BBox, kind: str, a_lo: float, a_hi: float) -> BBox:
    """Envelope of ``bbox`` under a rotation whose QUERY angle spans
    [a_lo, a_hi] (geometry rotates by the negated interval).

    Per corner, the in-plane coordinates trace an arc r*cos(φ+α) /
    r*sin(φ+α); the range over the interval is the endpoint values plus any
    interior extremum. Degenerates to _rotate_bbox for a point interval and
    to the swept disc for a full turn.
    """
    g_lo, g_hi = -float(a_hi), -float(a_lo)
    axis = {"rotate_x": 0, "rotate_y": 1, "rotate_z": 2}[kind]
    ui, vi = {"rotate_x": (1, 2), "rotate_y": (2, 0), "rotate_z": (0, 1)}[kind]
    lo_out = [math.inf] * 3
    hi_out = [-math.inf] * 3
    for corner in _bbox_corners(bbox):
        u0, v0 = corner[ui], corner[vi]
        r = math.hypot(u0, v0)
        alpha = math.atan2(v0, u0)
        u_min, u_max = _cos_range(r, alpha, g_lo, g_hi)
        # sin(φ + α) = cos(φ + α − π/2)
        v_min, v_max = _cos_range(r, alpha - 0.5 * math.pi, g_lo, g_hi)
        lo_out[ui] = min(lo_out[ui], u_min)
        hi_out[ui] = max(hi_out[ui], u_max)
        lo_out[vi] = min(lo_out[vi], v_min)
        hi_out[vi] = max(hi_out[vi], v_max)
        lo_out[axis] = min(lo_out[axis], corner[axis])
        hi_out[axis] = max(hi_out[axis], corner[axis])
    return (_vec3(lo_out), _vec3(hi_out))


def _swept_rotation_bbox(bbox: BBox, kind: str) -> BBox:
    """Envelope of ``bbox`` under an UNKNOWN rotation about a principal axis.

    The rotated solid stays inside the cylinder of radius R about the axis
    (R = farthest corner distance from the axis), with the axis extent
    unchanged. Exact in the limit of a full revolution; conservative for
    any actual angle interval.
    """
    axis = {"rotate_x": 0, "rotate_y": 1, "rotate_z": 2}[kind]
    lo, hi = bbox
    plane = [i for i in range(3) if i != axis]
    radius = 0.0
    for corner in _bbox_corners(bbox):
        radius = max(radius, math.hypot(corner[plane[0]], corner[plane[1]]))
    out_lo = [0.0, 0.0, 0.0]
    out_hi = [0.0, 0.0, 0.0]
    out_lo[axis], out_hi[axis] = lo[axis], hi[axis]
    for i in plane:
        out_lo[i], out_hi[i] = -radius, radius
    return (_vec3(out_lo), _vec3(out_hi))


def _rotate_bbox(bbox: BBox, kind: str, angle: float) -> BBox:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    if kind == "rotate_x":
        mat = ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    elif kind == "rotate_y":
        mat = ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    else:
        mat = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    return _rotate_matrix_bbox(bbox, mat)


def _rotate_matrix_bbox(bbox: BBox, matrix: Sequence[Sequence[float]]) -> BBox:
    """Envelope of the child bbox under the DSL's rotation convention.

    ``matrix`` is the QUERY rotation (``tf_rotate*`` evaluates the child at
    ``R @ p``), so the geometry itself moves by the inverse ``R^T``: the
    corners must be transformed by the TRANSPOSE. Applying ``R`` directly
    mirrors the box for asymmetric children (caught when AABB-pruned union
    folds used per-component boxes: a ``rotate_x(-π/2)``-wrapped plate
    got a z-negated box and was pruned out of the viewport).
    """
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise UnsupportedExprError("rotate_matrix expects a 3x3 constant matrix")
    corners = _bbox_corners(bbox)
    rotated = []
    for x, y, z in corners:
        rx = matrix[0][0] * x + matrix[1][0] * y + matrix[2][0] * z
        ry = matrix[0][1] * x + matrix[1][1] * y + matrix[2][1] * z
        rz = matrix[0][2] * x + matrix[1][2] * y + matrix[2][2] * z
        rotated.append((float(rx), float(ry), float(rz)))
    return _points_to_bbox(rotated)


def _mirror_bbox(bbox: BBox, n: Sequence[float], o: Sequence[float]) -> BBox:
    """AABB of a child unioned with its reflection across the plane ``(o, n)``.

    ``mirror`` returns ``min(child(p), child(reflect(p)))``, so the solid is the
    child together with its mirror image. Reflect the child AABB's eight corners
    across the plane and hull them with the original eight (16 points):
    conservative for any child, since the solid is contained in
    ``child_bbox ∪ reflect(child_bbox)``. ``n`` is normalised here to match the
    runtime (:func:`transforms.reflect_plane`), so a non-unit normal is handled.
    """
    if len(n) != 3 or len(o) != 3:
        raise UnsupportedExprError("mirror expects constant 3-vectors n and o")
    try:
        # A per-element $ref (e.g. o=[ref("ox"), 0, 0]) survives _require_constant
        # as a list, so re-check here: non-numeric elements fall back to the
        # numeric bbox tightener rather than crashing.
        nx, ny, nz = (float(c) for c in n)
        ox, oy, oz = (float(c) for c in o)
    except (TypeError, ValueError):
        raise UnsupportedExprError(
            "mirror bbox inference requires numeric-constant n and o"
        ) from None
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag == 0.0:
        raise UnsupportedExprError("mirror normal must be non-zero")
    nx, ny, nz = nx / mag, ny / mag, nz / mag  # unit normal (match runtime)
    pts = list(_bbox_corners(bbox))
    for x, y, z in list(pts):
        d = (x - ox) * nx + (y - oy) * ny + (z - oz) * nz
        pts.append((x - 2.0 * d * nx, y - 2.0 * d * ny, z - 2.0 * d * nz))
    return _points_to_bbox(pts)


__all__ = [
    "BBox",
    "BBoxInferenceError",
    "UnboundedParamError",
    "UnsupportedExprError",
    "UnsupportedSDFNodeError",
    "infer_material_bbox",
    "infer_sdf_bbox",
    "pad_bbox",
]
