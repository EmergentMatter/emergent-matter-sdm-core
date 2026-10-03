"""How fast can an SDF tree's value change per unit of travel through space?

An SDF is supposed to return the distance to the nearest surface, which means
moving one millimetre can change the value by at most one millimetre. Every
primitive satisfies that (``tests/test_sdf_is_distance.py`` verifies it), but a
whole *tree* need not: some operations stretch space, and the value stretches
with it.

    union(sphere, box)               1.0    both children are distances
    translate(sphere, t=[5,0,0])     1.0    moving a shape changes nothing
    twist(box, k=0.2)                >1     rotation angle varies with position,
                                            so the stretch grows with the domain
    displace(sphere, field)          4.09   an arbitrary field is added on top

:func:`infer_sdf_max_rate` walks the tree and returns the worst case, mirroring
the recursive structure of :mod:`software_defined_matter.sdf.bbox` (which walks
the same trees to return a box, using the same interval arithmetic).

Why a caller wants it
---------------------
Anything that reasons from the returned distance (e.g. for calculating volumes
with voxel skipping) needs to know how much to trust it. a cell whose centre
reports a distance greater than the cell's half-diagonal cannot contain any surface,
so it can be counted as wholly solid or wholly empty without further sampling.
That test is only valid at rate 1; in general it becomes::

    skip this cell if   abs(d) > half_diagonal * infer_sdf_max_rate(tree, part)

A tree with rate 4 must be four times more confident before it skips anything.
This is less efficient but correct.

Unknown nodes return :data:`UNKNOWN` (infinity) rather than raising, so the
formula above degrades to "never skip": correct, just slowest. Returning a
finite guess for a node nobody has analysed would silently discard geometry,
which is the failure this module exists to prevent.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from software_defined_matter.sdf.bbox import (
    BBox,
    Interval,
    UnsupportedSDFNodeError,
    _scalar_abs_max,
    _scalar_interval,
    _vec3,
    _vec_intervals,
    build_param_intervals,
)

if TYPE_CHECKING:
    from software_defined_matter.model import Part, SDFTree


#: No useful bound. Used for nodes whose rate has not been analysed, so callers
#: fall back to trusting nothing.
UNKNOWN = math.inf

#: Every registered primitive returns a true distance (rate <= 1). This is not
#: an assumption: ``tests/test_sdf_is_distance.py`` measures it for every
#: primitive over several parameter sets, and fails if one regresses.
_PRIMITIVE_RATE = 1.0


def infer_sdf_max_rate(
    tree: SDFTree,
    part: Part,
    *,
    mode: str = "bounds",
    domain: BBox | None = None,
) -> float:
    """Upper bound on ``|grad tree|``, or :data:`UNKNOWN` if none is available.

    ``mode`` follows :func:`software_defined_matter.sdf.bbox.infer_sdf_bbox`:
    ``'bounds'`` takes the worst case over each ``Param.bounds``, ``'values'``
    evaluates at the params' current values.

    ``domain`` is the region the bound has to hold over: normally the caller's
    sampling box. It matters because ``twist`` and ``bend`` rotate space by an
    angle proportional to *the query point's* position, so how much they
    stretch depends on how far out you evaluate, not on how big the shape is.
    Hermite ``twist_radial`` and ``twist_linear`` use the same query-domain
    contract. They require ramp endpoints with a provably positive separation.
    Without a domain, nonconstant twists return :data:`UNKNOWN`. Translation
    and nested query rotations propagate conservative child domains; other
    transform wrappers disable domain-sensitive bounds until their query-domain
    propagation is supported. Domain-independent children keep their bounds.
    A finite result bounds field variation, not deformation invertibility.
    """
    if domain is not None and (
        len(domain) != 2
        or any(len(corner) != 3 for corner in domain)
        or any(not math.isfinite(v) for corner in domain for v in corner)
        or any(lo > hi for lo, hi in zip(*domain, strict=True))
    ):
        raise ValueError("Sampling domain must be a finite ordered 3-D box")
    intervals = build_param_intervals(part, mode=mode)
    return _node_rate(tree, intervals, domain)


# ---------------------------------------------------------------------------
# Internal: recursive tree walk
# ---------------------------------------------------------------------------


def _node_rate(node: Any, intervals: dict[str, Interval], domain: BBox | None) -> float:
    if not isinstance(node, dict) or "type" not in node:
        raise UnsupportedSDFNodeError(f"Not an SDF node: {node!r}")

    node_type = node["type"]

    if node_type == "primitive":
        return _PRIMITIVE_RATE

    if node_type == "op":
        # min/max of the children (and their smooth blends, which only round
        # the seam) can be no steeper than the steepest child.
        children = node.get("children", []) or []
        if not children:
            return UNKNOWN
        return max(_node_rate(c, intervals, domain) for c in children)

    if node_type == "transform":
        return _transform_rate(node, intervals, domain)

    if node_type == "modifier":
        # `round` subtracts a constant, `onion` takes abs() and subtracts one.
        # Neither changes how fast the value moves.
        return _node_rate(node["child"], intervals, domain)

    if node_type == "2d_to_3d":
        # revolution maps p -> (|p.xy|, p.z) and extrusion combines the profile
        # with an axial term; both are 1-Lipschitz in p.
        return _node_rate(node["child"], intervals, domain)

    if node_type == "deform":
        return _deform_rate(node, intervals, domain)

    if node_type == "sweep":
        # The profile is evaluated in a per-segment frame built from unit
        # vectors, so the frame itself contributes no stretch.
        return _node_rate(node["child"], intervals, domain)

    # `loft` interpolates two 2-D fields along Z; the interpolation introduces
    # a Z-derivative that depends on how different the sections are, and no
    # bound has been worked out.
    return UNKNOWN


def _transform_rate(
    node: dict[str, Any], intervals: dict[str, Interval], domain: BBox | None
) -> float:
    tf = node["transform"]
    # A domain-sensitive child needs its query domain, not the parent's box.
    # Unhandled maps retain domain-independent rates but refuse local guesses.
    child_domain: BBox | None = None
    if domain is not None and tf == "translate":
        offsets = _vec_intervals(node["params"]["t"], 3, intervals)
        child_domain = (
            _vec3(domain[0][i] - offsets[i][1] for i in range(3)),
            _vec3(domain[1][i] - offsets[i][0] for i in range(3)),
        )
    child = _node_rate(node["child"], intervals, child_domain)

    # Rigid motions, and the folds, are isometries: they move points around
    # without stretching, so the child's rate carries through untouched.
    if tf in (
        "translate",
        "rotate_x",
        "rotate_y",
        "rotate_z",
        "rotate_matrix",
        "repeat_inf",
        "repeat_finite",
        "canonical_sector_fold",
    ):
        return child

    if tf == "scale_axis":
        # Compiled as `min(s) * child(p / s)`: each gradient component is
        # scaled by min(s) / s_i <= 1, so the child's rate is still a bound.
        # Only for factors that stay positive over their whole range.
        s = _vec_intervals(node["params"]["s"], len(node["params"]["s"]), intervals)
        return child if all(lo > 0 for lo, _hi in s) else UNKNOWN

    if tf == "scale":
        # Compiled as `child(p / s) * s`: the division and the multiplication
        # cancel in the derivative, so a uniform scale preserves the rate.
        # (Only uniform scale exists here; a per-axis one would divide by the
        # smallest factor.)
        return child

    return UNKNOWN


def _deform_rate(
    node: dict[str, Any], intervals: dict[str, Interval], domain: BBox | None
) -> float:
    name = node["deform"]
    kw = node.get("params") or {}

    if name in ("twist", "bend", "twist_radial", "twist_linear"):
        if domain is None:
            # A constant rotation still needs a rotated domain for a local
            # child bound. Domain-independent children remain usable.
            if name in ("twist", "bend") and _scalar_abs_max(kw["k"], intervals) == 0:
                return _node_rate(node["child"], intervals, None)
            return UNKNOWN
        maxima = [max(abs(lo), abs(hi)) for lo, hi in zip(*domain, strict=True)]
        plane = (0, 2) if name == "twist" else (0, 1)
        radius = math.hypot(*(maxima[i] for i in plane))
        lo, hi = list(domain[0]), list(domain[1])
        for i in plane:
            lo[i], hi[i] = -radius, radius
        # Every query rotation preserves the plane radius, including nested
        # deformations. The child must be bounded over this enlarged domain.
        child_domain: BBox = ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))
        child = _node_rate(node["child"], intervals, child_domain)
        if name in ("twist", "bend"):
            slope = _scalar_abs_max(kw["k"], intervals)
        else:
            radial = name == "twist_radial"
            low_key, high_key = ("r0", "r1") if radial else ("u0", "u1")
            a_key, b_key = ("angle_inner", "angle_outer") if radial else ("angle_0", "angle_1")
            low, high = (
                _scalar_interval(kw[low_key], intervals),
                _scalar_interval(kw[high_key], intervals),
            )
            width = high[0] - low[1]
            if width <= 0 or not math.isfinite(width):
                return UNKNOWN
            a, b = _scalar_interval(kw[a_key], intervals), _scalar_interval(kw[b_key], intervals)
            if not radial:
                axis = _vec_intervals(kw.get("axis", [1, 0]), 2, intervals)
                # Keep potentially vanishing directions unsupported in this bound.
                if all(lo <= 0 <= hi for lo, hi in axis):
                    return UNKNOWN
            # The derivative of cubic Hermite smoothstep peaks at 3/2.
            slope = 1.5 * max(abs(b[1] - a[0]), abs(a[1] - b[0])) / width
        if not math.isfinite(slope) or not math.isfinite(child):
            return UNKNOWN
        # Dq is a rotation plus an outer product of norm radius*|grad angle|.
        # sqrt(1+s^2) is NOT an upper bound: even a simple shear exceeds it.
        bound = child * (1.0 + radius * slope)
        return bound if math.isfinite(bound) else UNKNOWN

    if name == "taper_linear":
        # q = (x/s, y/s, z) with s = s(z) linear in z: Dq is diag(1/s, 1/s, 1)
        # plus an outer product of norm rho * |s'| / s^2, so the bound grows
        # with distance from the axis (like the twists) and needs a domain.
        if domain is None:
            return UNKNOWN
        z0, z1 = _scalar_interval(kw["z0"], intervals), _scalar_interval(kw["z1"], intervals)
        width = z1[0] - z0[1]
        s0, s1 = _scalar_interval(kw["s_0"], intervals), _scalar_interval(kw["s_1"], intervals)
        s_min = min(s0[0], s1[0])
        if width <= 0 or not math.isfinite(width) or s_min <= 0:
            return UNKNOWN
        slope = max(abs(s1[1] - s0[0]), abs(s0[1] - s1[0])) / width
        maxima = [max(abs(lo), abs(hi)) for lo, hi in zip(*domain, strict=True)]
        rho = math.hypot(maxima[0], maxima[1])
        r_child = rho / s_min
        (_, _, dz0), (_, _, dz1) = domain
        tapered_domain: BBox = ((-r_child, -r_child, dz0), (r_child, r_child, dz1))
        child = _node_rate(node["child"], intervals, tapered_domain)
        if not math.isfinite(child):
            return UNKNOWN
        bound = child * (max(1.0, 1.0 / s_min) + rho * slope / (s_min * s_min))
        return bound if math.isfinite(bound) else UNKNOWN

    if name == "shear_linear":
        # q = p - rise(u) e_z, so Dq = I - e_z (k n)^T with |k| the ramp slope:
        # a pure shear. Its largest singular value is (|k| + sqrt(k^2 + 4)) / 2,
        # NOT sqrt(1 + k^2) (see the twist note above). Unlike a twist the
        # stretch does not grow with distance from an axis, so the bound needs
        # no domain; only the child's domain shifts, by the rise.
        low, high = _scalar_interval(kw["u0"], intervals), _scalar_interval(kw["u1"], intervals)
        width = high[0] - low[1]
        if width <= 0 or not math.isfinite(width):
            return UNKNOWN
        axis = _vec_intervals(kw.get("axis", [1, 0]), 2, intervals)
        if all(lo <= 0 <= hi for lo, hi in axis):
            return UNKNOWN
        a, b = _scalar_interval(kw["dz_0"], intervals), _scalar_interval(kw["dz_1"], intervals)
        slope = max(abs(b[1] - a[0]), abs(a[1] - b[0])) / width
        sheared_domain: BBox | None = None
        if domain is not None:
            rise_lo, rise_hi = min(a[0], b[0]), max(a[1], b[1])
            (bx0, by0, bz0), (bx1, by1, bz1) = domain
            sheared_domain = ((bx0, by0, bz0 - rise_hi), (bx1, by1, bz1 - rise_lo))
        child = _node_rate(node["child"], intervals, sheared_domain)
        if not math.isfinite(slope) or not math.isfinite(child):
            return UNKNOWN
        bound = child * (slope + math.sqrt(slope * slope + 4.0)) / 2.0
        return bound if math.isfinite(bound) else UNKNOWN

    child = _node_rate(node["child"], intervals, domain)
    if name == "displace":
        # `sdf(p) + field(p)`: the two gradients add, so their bounds add.
        field = node.get("field")
        if field is None:
            return UNKNOWN
        field_rate = _field_rate(field, intervals)
        if field_rate == UNKNOWN:
            return UNKNOWN
        return child + field_rate

    return UNKNOWN


def _field_rate(node: Any, intervals: dict[str, Interval]) -> float:
    """Upper bound on ``|grad field|`` for the displacement-field DSL.

    Note this bounds the field's *slope*, where
    :func:`software_defined_matter.sdf.bbox._infer_field_amplitude` bounds its
    *value*: a bbox cares how far the surface moves, this cares how fast.
    """
    if not isinstance(node, dict) or "type" not in node:
        return UNKNOWN

    if node["type"] == "field":
        kind = node["kind"]
        kw = node.get("params") or {}
        amplitude = _scalar_abs_max(kw.get("amplitude", 1.0), intervals)

        if kind == "sin_xyz":
            # amplitude * prod_i sin(2 pi f_i x_i + phase_i). Differentiating
            # one factor leaves the others bounded by 1, so each partial is at
            # most amplitude * 2 pi f_i.
            freq = kw.get("freq", 0.0)
            if isinstance(freq, (list, tuple)):
                fs = [_scalar_abs_max(f, intervals) for f in freq]
            else:
                fs = [_scalar_abs_max(freq, intervals)] * 3
            return amplitude * 2.0 * math.pi * math.sqrt(sum(f * f for f in fs))

        if kind == "radial":
            # amplitude * sin(2 pi f |p.xy| + phase); |grad |p.xy|| == 1.
            f = _scalar_abs_max(kw.get("freq", 0.0), intervals)
            return amplitude * 2.0 * math.pi * f

        if kind == "angular":
            # Depends on the azimuth, whose gradient is 1/r, unbounded as the
            # Z axis is approached, so there is no finite bound to give.
            return UNKNOWN

        return UNKNOWN

    if node["type"] == "field_op" and node["op"] == "add":
        children = node.get("children", []) or []
        rates = [_field_rate(c, intervals) for c in children]
        if not rates or UNKNOWN in rates:
            return UNKNOWN
        return sum(rates)

    return UNKNOWN


__all__ = ["UNKNOWN", "infer_sdf_max_rate"]
