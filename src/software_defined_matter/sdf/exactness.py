"""Static distance contracts, separate from field variation and differentiation.

An EXACT side measures Euclidean distance to the represented boundary. A
LOWER_BOUND side has the correct sign and underestimates distance magnitude:
``abs(field) <= distance``. UNKNOWN makes neither promise. These are analytic
contracts for valid geometric inputs, not floating-point error certificates.
No inference evaluates a JAX field or inserts work into its traced graph.
"""

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from software_defined_matter.sdf._primitive_domains import primitive_domain_proven
from software_defined_matter.sdf.bbox import (
    BBoxInferenceError,
    Interval,
    _infer_node_bbox,
    _scalar_interval,
    _vec_intervals,
    build_param_intervals,
)

if TYPE_CHECKING:
    from software_defined_matter.dsl.resolve import ParamBinding
    from software_defined_matter.model import Part, SDFTree


class DistanceClass(StrEnum):
    """Strength of the distance-magnitude guarantee on one side."""

    EXACT = "exact"
    LOWER_BOUND = "lower_bound"
    UNKNOWN = "unknown"


EXACT = DistanceClass.EXACT
LOWER_BOUND = DistanceClass.LOWER_BOUND
UNKNOWN = DistanceClass.UNKNOWN


@dataclass(frozen=True)
class DistanceExactness:
    """Outside and inside guarantees, plus the reason for the classification.

    A label applies throughout the parameter range selected by inference,
    after proving the primitive's sufficient geometric preconditions. It does not
    certify nonemptiness, regularity, or the absence of internal zero sheets
    in degenerate booleans. UNKNOWN is deliberately not a lower bound.
    """

    outside: DistanceClass
    inside: DistanceClass
    reason: str

    @property
    def is_exact(self) -> bool:
        """Whether both sides have an exact-distance contract."""
        return self.outside == self.inside == EXACT

    def as_dict(self) -> dict[str, str]:
        """JSON-compatible provenance; this is not a claim about baked samples."""
        return {"outside": str(self.outside), "inside": str(self.inside), "reason": self.reason}


class DistanceAccuracyWarning(UserWarning):
    """A consumer interprets field values as lengths without an exact contract."""


# Conditional analytic declarations: _infer applies primitive_domain_proven
# before returning these. This table alone is not a parameter-specific certificate.
_EXACT_PRIMITIVES = frozenset(
    [
        "sphere",
        "box",
        "round_box",
        "torus",
        "capped_torus",
        "link",
        "cone",
        "plane",
        "hex_prism",
        "capsule",
        "capped_cylinder",
        "rounded_cylinder",
        "capped_cone",
        "solid_angle",
        "cut_sphere",
        "octahedron",
        "pyramid",
        "leaf_spring",
        "circle_2d",
        "box_2d",
        "rounded_box_2d",
        "segment_2d",
        "trapezoid_2d",
        "uneven_capsule_2d",
        "polygon_2d",
        "bezier_2d",
        "bspline_2d",
    ]
)
_BOUND_PRIMITIVES = frozenset(
    [
        "ellipsoid",
        "bellows",
        "serpentine",
        "screw_thread",
        "gyroid",
        "schwarz_p",
        "schwarz_d",
        "neovius",
        "lidinoid",
    ]
)
PRIMITIVE_EXACTNESS = {
    **{name: DistanceExactness(EXACT, EXACT, "analytic distance") for name in _EXACT_PRIMITIVES},
    **{
        name: DistanceExactness(LOWER_BOUND, LOWER_BOUND, "scaled implicit distance bound")
        for name in _BOUND_PRIMITIVES
    },
    **{
        name: DistanceExactness(LOWER_BOUND, EXACT, "intersection of exact fields")
        for name in ("annular_sector", "tri_prism", "notch_hinge")
    },
    "box_frame": DistanceExactness(EXACT, LOWER_BOUND, "union of edge boxes"),
    "capped_torus": DistanceExactness(
        EXACT, LOWER_BOUND, "arc tube may overlap across its end caps"
    ),
    "helix": DistanceExactness(UNKNOWN, UNKNOWN, "discontinuous on the axis"),
    "raster_field": DistanceExactness(
        UNKNOWN, UNKNOWN, "interpolated samples are not exact distances"
    ),
}


def _unknown(reason: str) -> DistanceExactness:
    return DistanceExactness(UNKNOWN, UNKNOWN, reason)


def _bound(*classes: DistanceClass) -> DistanceClass:
    return UNKNOWN if UNKNOWN in classes else LOWER_BOUND


def _common(classes: list[DistanceClass]) -> DistanceClass:
    return EXACT if all(c == EXACT for c in classes) else _bound(*classes)


def infer_sdf_exactness(
    tree: SDFTree,
    part: Part,
    *,
    mode: str = "bounds",
    b_smooth_csg: bool | None = None,
) -> DistanceExactness:
    """Infer side-specific contracts over parameter bounds or current values.

    ``mode`` has the same meaning as bbox inference. Missing bounds and
    unsupported expressions prevent geometric proofs; they never grant an
    exactness upgrade. Primitive declarations are conditional on their domain
    proofs: oversized rounding, invalid cuts, and unproven outline simplicity
    produce UNKNOWN, not an assumed lower bound. Derived parameters are left unresolved rather than
    trusting their cached values. ``b_smooth_csg`` follows the compiler's
    metadata override; smoothing hard booleans removes their exactness proof.

    Strictly disjoint inferred boxes can establish exact union interiors and
    disjoint subtraction. Simple sphere and box hosts can also prove interior
    containment of cutters. Box containment alone never proves solid containment.
    Folds, repeats, arbitrary matrix maps, sweeps and unanalysed deformations
    remain UNKNOWN. In particular a unit gradient is not an exactness proof.
    """
    intervals = _parameter_intervals(part, mode)
    smooth = bool(part.metadata.get("smooth_csg", False)) if b_smooth_csg is None else b_smooth_csg
    return _infer(tree, intervals, smooth)


def _infer_binding_exactness(
    tree: SDFTree, binding: ParamBinding, *, b_smooth_csg: bool
) -> DistanceExactness:
    """Infer a callable-wide contract from the constants its compiler captured.

    Free-vector arguments are unrestricted, even when the Part declares bounds.
    Advertising a range proof on that callable would let later eager or JIT
    evaluations silently escape it. Derived values remain unresolved.
    """
    from software_defined_matter.dsl.resolve import ParamBinding

    if not isinstance(binding, ParamBinding):
        return _unknown("binding adapter has no captured parameter-domain proof")
    if binding.n_free():
        return _unknown("free-vector evaluation does not enforce parameter bounds")
    if not all(math.isfinite(value) for value in binding.fixed.values()):
        return _unknown("captured fixed parameters must be finite")
    intervals = {name: (value, value) for name, value in binding.fixed.items()}
    return _infer(tree, intervals, b_smooth_csg)


def _parameter_intervals(part: Part, mode: str) -> dict[str, Interval]:
    # A relation can have no cached value at all. Remove it before asking
    # bbox's interval builder for numeric values, without mutating the Part.
    independent = copy(part)
    independent.params = {name: p for name, p in part.params.items() if p.expr is None}
    return build_param_intervals(independent, mode=mode)


def _disjoint(nodes: list[dict[str, Any]], intervals: dict[str, Interval]) -> bool:
    try:
        boxes = [_infer_node_bbox(node, intervals) for node in nodes]
    except (BBoxInferenceError, KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for box in boxes for corner in box for v in corner):
        return False
    return all(
        any(a[1][k] < b[0][k] or b[1][k] < a[0][k] for k in range(3))
        for i, a in enumerate(boxes)
        for b in boxes[i + 1 :]
    )


def _strictly_contains_box(node: dict[str, Any], box: Any, intervals: dict[str, Interval]) -> bool:
    """Prove solid containment for simple hosts, never from their outer boxes."""
    kw = node.get("params") or {}
    if node.get("type") == "transform" and node.get("transform") == "translate":
        offsets = _vec_intervals(kw["t"], 3, intervals)
        local = (
            tuple(box[0][i] - offsets[i][1] for i in range(3)),
            tuple(box[1][i] - offsets[i][0] for i in range(3)),
        )
        return _strictly_contains_box(node["child"], local, intervals)
    if node.get("type") != "primitive":
        return False
    extents = [max(abs(lo), abs(hi)) for lo, hi in zip(*box, strict=True)]
    if node.get("kind") == "sphere":
        radius, _ = _scalar_interval(kw["r"], intervals)
        return math.hypot(*extents) < radius
    if node.get("kind") == "box":
        half_sizes = _vec_intervals(kw["b"], 3, intervals)
        return all(extent < size[0] for extent, size in zip(extents, half_sizes, strict=True))
    return False


def _contained_cutters(nodes: list[dict[str, Any]], intervals: dict[str, Interval]) -> bool:
    try:
        # Cutters must not swallow each other's boundary either.
        if len(nodes) > 2 and not _disjoint(nodes[1:], intervals):
            return False
        return all(
            _strictly_contains_box(nodes[0], _infer_node_bbox(c, intervals), intervals)
            for c in nodes[1:]
        )
    except (BBoxInferenceError, KeyError, TypeError, ValueError):
        return False


def _finite_rigid_parameter(value: Any, intervals: dict[str, Interval]) -> bool:
    """Require finite intermediates as well as finite final map parameters.

    Python min/max can hide a NaN operand during interval evaluation, whereas
    the JAX expression propagates it. Such an expression is not a rigid map.
    """
    if isinstance(value, dict):
        operands = [value[key] for key in ("child", "lhs", "rhs") if key in value]
        operands.extend(value.get("children", []))
        if not all(_finite_rigid_parameter(operand, intervals) for operand in operands):
            return False
    lo, hi = _scalar_interval(value, intervals)
    return math.isfinite(lo) and math.isfinite(hi) and lo <= hi


def _infer(node: Any, intervals: dict[str, Interval], smooth: bool) -> DistanceExactness:
    if not isinstance(node, dict):
        return _unknown("not an SDF node")
    kind = node.get("type")
    kw = node.get("params") or {}
    try:
        if kind == "primitive":
            name = node.get("kind", "")
            if name not in PRIMITIVE_EXACTNESS:
                return _unknown("undeclared primitive")
            if not primitive_domain_proven(name, kw, intervals):
                return _unknown(f"{name}: primitive preconditions not proven over parameter range")
            if (
                name == "annular_sector"
                and _scalar_interval(kw["half_angle"], intervals)[1] > math.pi / 2
            ):
                return DistanceExactness(
                    LOWER_BOUND, LOWER_BOUND, "reflex sector unions angular half-spaces"
                )
            if name in ("torus", "link"):
                # The unsigned centreline distance minus a tube radius is
                # exact outside. Inside, a tube exceeding the centreline's
                # reach can cover the closest candidate boundary.
                try:
                    if name == "torus":
                        major, minor = _vec_intervals(kw["t"], 2, intervals)
                    else:
                        major = _scalar_interval(kw["r1"], intervals)
                        minor = _scalar_interval(kw["r2"], intervals)
                    if 0 <= minor[0] <= minor[1] < major[0]:
                        return PRIMITIVE_EXACTNESS[name]
                except (BBoxInferenceError, KeyError, TypeError, ValueError):
                    return _unknown("tube radii could not be bounded")
                return DistanceExactness(EXACT, LOWER_BOUND, "tube reach is not established")
            if name == "plane":
                n = _vec_intervals(kw["n"], 3, intervals)
                min_component = max(max(lo, -hi, 0.0) for lo, hi in n)
                # The shared JAX/GLSL contract cannot rely on subnormals:
                # float32 shader arithmetic may flush them to zero.
                if min_component < 2.0**-126:
                    return _unknown("plane normal may vanish or flush to zero in float32")
            return PRIMITIVE_EXACTNESS.get(name, _unknown("undeclared primitive"))
        if kind == "op":
            children = node.get("children") or []
            if not children:
                return _unknown("empty operation")
            op = node.get("op")
            if op not in ("union", "intersect", "subtract") or smooth:
                return _unknown("smooth or unanalysed operation")
            cs = [_infer(c, intervals, smooth) for c in children]
            if len(cs) == 1:
                return cs[0]
            if op == "union":
                outside = _common([c.outside for c in cs])
                inside = _bound(*(c.inside for c in cs))
                if all(c.is_exact for c in cs) and _disjoint(children, intervals):
                    inside = EXACT
                return DistanceExactness(
                    outside, inside, "hard union; interior needs disjoint solids"
                )
            if op == "subtract":
                if all(c.is_exact for c in cs) and (
                    _disjoint(children, intervals) or _contained_cutters(children, intervals)
                ):
                    return DistanceExactness(
                        EXACT, EXACT, "cutters strictly outside or inside host"
                    )
                cs = [cs[0]] + [DistanceExactness(c.inside, c.outside, c.reason) for c in cs[1:]]
            return DistanceExactness(
                _bound(*(c.outside for c in cs)),
                _common([c.inside for c in cs]),
                "intersection; subtraction swaps cutter sides",
            )
        child = _infer(node.get("child"), intervals, smooth)
        if kind == "transform":
            tf = node.get("transform")
            if tf in ("translate", "rotate_x", "rotate_y", "rotate_z"):
                # Distance preservation assumes a defined rigid map at every
                # parameter value. Singular expressions or nonfinite inputs
                # must not inherit an otherwise exact child's contract.
                if tf == "translate":
                    offset = kw["t"]
                    if not isinstance(offset, (list, tuple)) or len(offset) not in (2, 3):
                        return _unknown("translation requires an axis vector")
                    parameters = offset
                else:
                    parameters = [kw["angle"]]
                if not all(_finite_rigid_parameter(value, intervals) for value in parameters):
                    return _unknown("rigid-transform parameters must stay finite")
                return child
            if tf == "scale":
                lo, hi = _scalar_interval(kw["s"], intervals)
                if 0 < lo <= hi < math.inf:
                    return child
            if tf == "scale_axis":
                factors = _vec_intervals(kw["s"], len(kw["s"]), intervals)
                if all(0 < lo <= hi < math.inf for lo, hi in factors):
                    if all(f == factors[0] and f[0] == f[1] for f in factors):
                        return child
                    return DistanceExactness(
                        _bound(child.outside), _bound(child.inside), "minimum-axis scale bound"
                    )
            return _unknown("query map has no established distance-preservation proof")
        if kind == "modifier":
            modifier = node.get("modifier")
            if modifier == "round":
                lo, hi = _scalar_interval(kw["r"], intervals)
                if lo == hi == 0:
                    return child
                bound = _bound(child.outside, child.inside)
                if lo >= 0:
                    return DistanceExactness(
                        EXACT if child.outside == EXACT else bound,
                        bound,
                        "outward offset; medial axes remain",
                    )
                if hi <= 0:
                    return DistanceExactness(
                        bound,
                        EXACT if child.inside == EXACT else bound,
                        "inward offset; medial axes remain",
                    )
                return DistanceExactness(bound, bound, "offset may change sign")
            if modifier == "onion":
                lo, _ = _scalar_interval(kw["thickness"], intervals)
                if lo >= 0:
                    bound = _bound(child.outside, child.inside)
                    return DistanceExactness(
                        EXACT if child.is_exact else bound, bound, "shell may exceed local reach"
                    )
            return _unknown("modifier lacks an exact-distance proof")
        if kind == "2d_to_3d" and node.get("method") == "extrusion":
            lo, hi = _scalar_interval(kw["h"], intervals)
            if not (0 < lo <= hi < math.inf):
                return _unknown("extrusion half-height must stay positive and finite")
            if UNKNOWN in (child.outside, child.inside):
                return _unknown(
                    "extrusion requires sign-correct distance bounds on both child sides"
                )
            # Outside the Cartesian product, positive side/cap distances
            # combine by hypot; inside, depth is the smaller boundary
            # distance. Both preserve each side's exactness or lower bound.
            return DistanceExactness(child.outside, child.inside, "extrusion of a bounded profile")
        return _unknown("node has no established distance contract")
    except (BBoxInferenceError, KeyError, TypeError, ValueError, OverflowError):
        return _unknown("parameter interval could not establish the contract")


__all__ = [
    "EXACT",
    "LOWER_BOUND",
    "UNKNOWN",
    "DistanceAccuracyWarning",
    "DistanceClass",
    "DistanceExactness",
    "PRIMITIVE_EXACTNESS",
    "infer_sdf_exactness",
]
