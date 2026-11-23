"""Sufficient parameter-domain proofs for primitive distance declarations.

Intervals deliberately lose correlations: failure means unproven, not invalid.
No numerical field sampling or JAX evaluation participates in these proofs.
"""

from __future__ import annotations

import math
from typing import Any

from software_defined_matter.sdf.bbox import Interval, _scalar_interval, _vec_intervals


def primitive_domain_proven(name: str, kw: dict[str, Any], intervals: dict[str, Interval]) -> bool:
    """Whether the primitive's analytic contract holds throughout the intervals.

    Missing bounds, unsupported expressions and malformed parameters are handled
    by the caller as UNKNOWN. Rounded boxes require a nonnegative core in every
    dimension; rounded cylinders require a nonnegative core radius. Angular
    vectors must describe a nondegenerate aperture and survive the evaluator's
    direct squared-norm normalization. Polygon simplicity is checked only for
    fixed vertices; curve tessellations and moving polygons remain unproven.
    """

    def scalar(key: str, default: float | None = None) -> Interval:
        value = kw[key] if default is None else kw.get(key, default)
        lo, hi = _scalar_interval(value, intervals)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError("non-finite or unordered primitive parameter interval")
        return lo, hi

    def vector(key: str, dim: int) -> tuple[Interval, ...]:
        result = _vec_intervals(kw[key], dim, intervals)
        if not all(math.isfinite(lo) and math.isfinite(hi) and lo <= hi for lo, hi in result):
            raise ValueError("non-finite or unordered primitive vector interval")
        return result

    def positive(*keys: str) -> bool:
        return all(scalar(key)[0] > 0 for key in keys)

    if name in ("sphere", "circle_2d"):
        return positive("r")
    if name in ("box", "box_2d", "round_box", "rounded_box_2d", "box_frame"):
        sizes = vector("b", 2 if name.endswith("_2d") else 3)
        smallest = min(lo for lo, _ in sizes)
        if smallest <= 0:
            return False
        if name in ("round_box", "rounded_box_2d"):
            r = scalar("r")
            return 0 <= r[0] <= r[1] <= smallest
        if name == "box_frame":
            e = scalar("e")
            return 0 < e[0] <= e[1] < smallest
        return True
    if name in ("torus", "link"):
        if name == "torus":
            radii = vector("t", 2)
        else:
            radii = (scalar("r1"), scalar("r2"))
            if scalar("le")[0] < 0:
                return False
        return all(lo > 0 for lo, _ in radii)
    if name == "solid_angle":
        if "c" in kw:
            return False
        lo, hi = scalar("theta")
        # Keep float32 angles away from a flushed sine or pi rounded onto
        # the opposite side of the aperture domain.
        return positive("ra") and 2.0**-126 <= lo <= hi <= math.pi - 2.0**-22
    if name == "capped_torus":
        direction = vector("sc", 2)
        # Positive sine restricts the aperture to (0, pi), without assuming a
        # unit input vector. At least one squared component must survive
        # float32 flushing; a normal sum of subnormal squares is not enough.
        min_component = max(max(lo, -hi, 0.0) for lo, hi in direction)
        max_norm = math.hypot(*(max(abs(lo), abs(hi)) for lo, hi in direction))
        return (
            positive("ra")
            and positive("rb")
            and direction[0][0] > 0
            and min_component >= 2.0**-63
            and max_norm < 2.0**63
        )
    if name == "plane":
        vector("n", 3)
        scalar("h")
        return True  # The stable-normalization range is checked by exactness.
    if name == "cone":
        direction = vector("c", 2)
        return positive("h") and all(lo > 0 for lo, _ in direction)
    if name in ("hex_prism", "tri_prism"):
        return all(lo > 0 for lo, _ in vector("h", 2))
    if name in ("capsule", "segment_2d"):
        dim = 3 if name == "capsule" else 2
        vector("a", dim)
        vector("b", dim)
        return name == "segment_2d" or positive("r")
    if name == "capped_cylinder":
        return positive("r", "h")
    if name == "rounded_cylinder":
        return (
            positive("ra", "h") and 0 <= scalar("rb")[0] <= scalar("rb")[1] <= 2 * scalar("ra")[0]
        )
    if name in ("capped_cone", "trapezoid_2d", "uneven_capsule_2d"):
        r1, r2 = scalar("r1"), scalar("r2")
        height = scalar("he" if name == "trapezoid_2d" else "h")
        if min(r1[0], r2[0], height[0]) <= 0:
            return False
        return name != "uneven_capsule_2d" or max(r1[1] - r2[0], r2[1] - r1[0]) < height[0]
    if name == "cut_sphere":
        radius, height = scalar("r"), scalar("h")
        return radius[0] > max(abs(height[0]), abs(height[1]))
    if name == "ellipsoid":
        return all(lo > 0 for lo, _ in vector("r", 3))
    if name == "octahedron":
        return positive("s")
    if name == "pyramid":
        return positive("h")
    if name == "leaf_spring":
        return positive("length", "width", "thickness")
    if name == "notch_hinge":
        return positive("width", "depth", "notch_radius")
    if name == "annular_sector":
        inner, outer, angle = scalar("inner_r"), scalar("outer_r"), scalar("half_angle")
        return (
            0 < inner[0] <= inner[1] < outer[0]
            and 0 < angle[0] <= angle[1] < math.pi
            and positive("height")
        )
    if name == "bellows":
        return (
            positive("inner_r", "outer_r", "period", "n_periods")
            and scalar("inner_r")[1] <= scalar("outer_r")[0]
        )
    if name == "serpentine":
        scalar("amplitude")
        return positive("wavelength", "beam_width", "beam_height", "n_periods")
    if name in ("gyroid", "schwarz_p", "schwarz_d", "neovius", "lidinoid"):
        return positive("period", "min_thickness") and all(
            lo > 0 for lo, _ in vector("n_periods", 3)
        )
    if name == "screw_thread":
        pitch, flank = scalar("pitch"), scalar("flank_deg", 60)
        handedness = scalar("handedness", 1)
        scalar("phase", 0)
        return (
            positive("r_root", "depth", "width", "n_turns")
            and pitch[0] * pitch[1] > 0
            and 0 <= flank[0] <= flank[1] <= 180
            and handedness in ((-1, -1), (1, 1))
        )
    if name == "polygon_2d":
        vertices = [_vec_intervals(v, 2, intervals) for v in kw["vertices"]]
        if not all(math.isfinite(lo) and lo == hi for v in vertices for lo, hi in v):
            return False
        return _simple_polygon([(v[0][0], v[1][0]) for v in vertices])
    # Curve control points do not establish simplicity of the sampled outline.
    # Helices and rasters have no distance contract even on ordinary domains.
    return False


def _simple_polygon(points: list[tuple[float, float]]) -> bool:
    """Prove a fixed polygon has nonzero area and no crossing or retraced edges."""
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3 or len(set(points)) != len(points):
        return False
    edges = list(zip(points, points[1:] + points[:1], strict=True))
    area = sum(a[0] * b[1] - a[1] * b[0] for a, b in edges)
    if not math.isfinite(area) or area == 0:
        return False

    def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for i, (a, b) in enumerate(edges):
        # polygon_2d substitutes a unit denominator at this squared-edge
        # threshold. A short nonzero edge then measures toward a vertex,
        # rather than its true perpendicular foot.
        if math.hypot(b[0] - a[0], b[1] - a[1]) <= 1e-6:
            return False
        # Reject collinear adjacent edges too: this excludes harmless redundant
        # vertices but also prevents an overlapping backtrack being certified.
        turn = orient(a, b, edges[(i + 1) % len(edges)][1])
        if not math.isfinite(turn) or turn == 0:
            return False
        for j in range(i + 2, len(edges)):
            if i == 0 and j == len(edges) - 1:
                continue
            c, d = edges[j]
            if any(
                max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k])
                for k in (0, 1)
            ):
                continue
            cross = (orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b))
            if not all(math.isfinite(v) for v in cross):
                return False
            if cross[0] * cross[1] <= 0 and cross[2] * cross[3] <= 0:
                return False
    return True
