"""Conservative body and flexure envelopes for shader clipping and component pruning.

Propagate a rest box through each body's ordered operations. Expressions use
intervals over authored motion ranges, converted to radians before evaluation.
Every corner's rotation arc includes its interior extrema. Repeated references
may make a box loose, but do not require a Cartesian product of motion inputs.

Design parameters use exploration bounds, then parameter bounds, then fixed
values. Derived parameters follow their expressions. Regenerate bounds after
changing geometry, ranges, or a fixed value without bounds. Values outside the
advertised ranges are not covered. Small outward margins accommodate float32
shader arithmetic; this is not a formally rounded interval arithmetic package.
"""

from __future__ import annotations

import copy
import math
import struct
from dataclasses import dataclass
from typing import Any

from software_defined_matter.model import Part
from software_defined_matter.sdf.bbox import (
    BBox,
    BBoxInferenceError,
    Interval,
    UnsupportedExprError,
    _bbox_corners,
    _bbox_enclose,
    _infer_node_bbox,
    _interval_binop,
    _interval_unop,
    _points_to_bbox,
    _vec3,
)

__all__ = ["BodyMotionBounds", "FlexureMotionBounds", "MotionBounds", "infer_motion_bounds"]


@dataclass(frozen=True)
class BodyMotionBounds:
    """A body's rest and swept boxes, with its motion-input dependencies.

    ``bbox=None`` disables component pruning when only an authored scene box
    is available. ``dof_names`` follows declaration order, not lexical order.
    """

    name: str
    dof_names: tuple[str, ...]
    rest_bbox: BBox | None
    bbox: BBox | None


@dataclass(frozen=True)
class FlexureMotionBounds(BodyMotionBounds):
    """A flexure's material-support and swept boxes, in the body bounds format.

    The rest box encloses all material because nearest-region ownership may
    extend beyond a classifier's negative set. None disables component pruning.
    """


@dataclass(frozen=True)
class MotionBounds:
    """Body and flexure bounds in document order and their enclosing scene box.

    An authored scene box can enlarge an inferred box but cannot shrink it.
    An empty body list has no inferred scene box.
    """

    bodies: tuple[BodyMotionBounds, ...]
    bbox: BBox | None
    flexures: tuple[FlexureMotionBounds, ...] = ()


def _finite(interval: Interval) -> Interval:
    if not all(math.isfinite(v) for v in interval) or interval[0] > interval[1]:
        raise BBoxInferenceError(f"Motion bounds require a finite ordered interval, got {interval}")
    return interval


def _outward(interval: Interval) -> Interval:
    """Allow float32 error at each arithmetic step, before later cancellation."""
    lo, hi = _finite(interval)
    if max(abs(lo), abs(hi)) > (2 - 2**-23) * 2**127:
        raise BBoxInferenceError("Motion interval exceeds finite float32 shader arithmetic")
    lower = lo - abs(lo) * 4 * 2**-23 - 2**-149
    upper = hi + abs(hi) * 4 * 2**-23 + 2**-149
    # Finite floating-point rounding cannot reverse an exact result's sign.
    return (max(0.0, lower) if lo >= 0 else lower, min(0.0, upper) if hi <= 0 else upper)


class _Intervals:
    def __init__(self, part: Part, dofs: dict[str, Interval]) -> None:
        self.part, self.dofs = part, dofs
        self.params: dict[str, Interval] = {}
        self.active: set[str] = set()

    def param(self, name: str) -> Interval:
        if name in self.params:
            return self.params[name]
        if name in self.active:
            raise BBoxInferenceError(f"Cyclic design parameter {name!r} in motion bounds")
        self.active.add(name)
        try:
            param = self.part.params[name]
            if param.expr is not None:
                result = self.expression(param.expr, None)
            else:
                bounds = (param.ui or {}).get("explore_bounds") or param.bounds
                if bounds is not None:
                    result = _finite((float(bounds[0]), float(bounds[1])))
                elif not param.free:
                    result = _finite((float(param.numeric_value()),) * 2)
                else:
                    raise BBoxInferenceError(f"Design parameter {name!r} needs finite bounds")
        finally:
            self.active.remove(name)
        self.params[name] = _outward(result)
        return self.params[name]

    def expression(self, node: Any, dependencies: set[str] | None) -> Interval:
        if isinstance(node, (float, int)):
            value = float(node)
            try:
                shader_value = struct.unpack("f", struct.pack("f", value))[0]
            except OverflowError as exc:
                raise BBoxInferenceError(
                    "Motion literal exceeds finite float32 arithmetic"
                ) from exc
            return _finite((min(value, shader_value), max(value, shader_value)))
        if "$ref" in node:
            return self.param(node["$ref"])
        kind = node["type"]
        if kind == "num":
            return self.expression(node["value"], dependencies)
        if kind == "param":
            return self.param(node["name"])
        if kind == "dof" and dependencies is not None:
            dependencies.add(node["name"])
            return _outward(self.dofs[node["name"]])
        try:
            if kind == "unop":
                lo, hi = self.expression(node["child"], dependencies)
                op = node["op"]
                if op == "exp":
                    result = (math.exp(lo), math.exp(hi))
                elif op == "log":
                    result = (math.log(lo), math.log(hi))
                else:
                    result = _interval_unop(op, (lo, hi))
            elif kind == "binop":
                lhs = self.expression(node["lhs"], dependencies)
                rhs = self.expression(node["rhs"], dependencies)
                result = (
                    _power(lhs, rhs)
                    if node["op"] == "pow"
                    else _interval_binop(node["op"], lhs, rhs)
                )
            elif kind == "reduce":
                children = [self.expression(c, dependencies) for c in node["children"]]
                op = node["op"]
                if op in {"sum", "mean"}:
                    divisor = len(children) if op == "mean" else 1
                    result = (
                        sum(c[0] for c in children) / divisor,
                        sum(c[1] for c in children) / divisor,
                    )
                elif op in {"min", "max"}:
                    reduce = min if op == "min" else max
                    result = (reduce(c[0] for c in children), reduce(c[1] for c in children))
                else:
                    raise UnsupportedExprError(f"Unsupported reduction {op!r}")
            else:
                raise UnsupportedExprError(f"Unsupported motion bounds expression {kind!r}")
        except (OverflowError, ZeroDivisionError, ValueError) as exc:
            raise BBoxInferenceError(f"Cannot bound motion expression: {exc}") from exc
        return _outward(result)

    def rest_bbox(self, tree: dict[str, Any], *, smooth_csg: bool) -> BBox:
        # Resolve scalar expressions through the same parameter intervals as
        # motion. In particular, a cached derived value is not a geometry bound.
        values: dict[str, Interval] = {}
        if smooth_csg:
            raise BBoxInferenceError("Smoothed rest fields require an authored motion box")

        def resolve(node: Any) -> Any:
            if isinstance(node, list):
                return [resolve(v) for v in node]
            if not isinstance(node, dict):
                return node
            if node.get("op") == "smooth_union" or node.get("modifier") == "onion":
                raise BBoxInferenceError("Expanded rest fields require an authored motion box")
            if node.get("transform") == "scale":
                lo, hi = self.expression(node["params"]["s"], None)
                if lo <= 0:
                    raise BBoxInferenceError("Nonpositive rest scale cannot be bounded")
                child_box = _infer_node_bbox(resolve(node["child"]), values)
                scaled = _points_to_bbox(
                    _vec3(s * v for v in corner)
                    for s in (lo, hi)
                    for corner in _bbox_corners(child_box)
                )
                # Collapse to a box instead of duplicating the subtree per
                # endpoint, which would double work at every nested scale.
                return {
                    "type": "transform",
                    "transform": "translate",
                    "params": {"t": [(a + b) / 2 for a, b in zip(*scaled, strict=True)]},
                    "child": {
                        "type": "primitive",
                        "kind": "box",
                        "params": {"b": [(b - a) / 2 for a, b in zip(*scaled, strict=True)]},
                    },
                }
            if "$ref" in node or node.get("type") in {
                "num",
                "param",
                "unop",
                "binop",
                "reduce",
                "dof",
            }:
                name = f"interval_{len(values)}"
                values[name] = self.expression(node, None)
                return {"$ref": name}
            return {k: resolve(v) for k, v in node.items()}

        resolved = resolve(tree)
        return _infer_node_bbox(resolved, values)


def _power(base: Interval, exponent: Interval) -> Interval:
    lo, hi = base
    a, b = exponent
    if a == b and float(a).is_integer():
        if a < 0 and lo <= 0 <= hi:
            raise UnsupportedExprError("Negative power over an interval spanning zero")
        if a == 0:
            return (1.0, 1.0)
        values = [lo**a, hi**a]
        if a > 0 and int(a) % 2 == 0 and lo <= 0 <= hi:
            values.append(0.0)
    elif lo < 0 or (lo == 0 and a <= 0):
        raise UnsupportedExprError("Power requires a nonnegative base and a defined exponent range")
    else:
        values = [x**y for x in base for y in exponent]
    return (min(values), max(values))


def _padded(box: BBox, scale: float = 0.0) -> BBox:
    # Account for float32 shader arithmetic, including cancellation at a pivot.
    margin = 32 * 2**-23 * max(1.0, scale, *(abs(v) for corner in box for v in corner))
    result = (_vec3(v - margin for v in box[0]), _vec3(v + margin for v in box[1]))
    for axis in zip(*result, strict=True):
        _finite(axis)
    return result


def _sweep(box: BBox, op: dict[str, Any], interval: Interval) -> BBox:
    norm = math.hypot(*op["axis"])
    axis = [float(v) / norm for v in op["axis"]]
    if op["kind"] == "translate":
        delta = [sorted(v * t for t in interval) for v in axis]
        return _padded(
            (
                _vec3(box[0][i] + delta[i][0] for i in range(3)),
                _vec3(box[1][i] + delta[i][1] for i in range(3)),
            ),
            max(abs(t) for t in interval),
        )
    origin = op.get("origin", (0, 0, 0))
    if not all(math.isfinite(v) for v in origin):
        raise BBoxInferenceError("Rotation origins must be finite")
    lo, hi = interval
    # Large authored angles can round by radians when uploaded as float32.
    angle_margin = 4 * 2**-23 * max(abs(lo), abs(hi))
    lo, hi = lo - angle_margin, hi + angle_margin
    mins, maxs = [math.inf] * 3, [-math.inf] * 3
    for corner in _bbox_corners(box):
        v = [corner[i] - origin[i] for i in range(3)]
        dot = sum(a * p for a, p in zip(axis, v, strict=True))
        cross = [
            axis[1] * v[2] - axis[2] * v[1],
            axis[2] * v[0] - axis[0] * v[2],
            axis[0] * v[1] - axis[1] * v[0],
        ]
        for i in range(3):
            c, a, b = origin[i] + axis[i] * dot, v[i] - axis[i] * dot, cross[i]
            radius = math.hypot(a, b)
            if hi - lo >= 2 * math.pi:
                lower, upper = c - radius, c + radius
            else:
                values = [c + a * math.cos(t) + b * math.sin(t) for t in (lo, hi)]
                phase = math.atan2(b, a)
                for t, value in ((phase, c + radius), (phase + math.pi, c - radius)):
                    if t + math.ceil((lo - t) / (2 * math.pi)) * (2 * math.pi) <= hi:
                        values.append(value)
                lower, upper = min(values), max(values)
            mins[i], maxs[i] = min(mins[i], lower), max(maxs[i], upper)
    return _padded((_vec3(mins), _vec3(maxs)), max(abs(v) for v in origin))


def _flexure_sweep(
    rest: BBox,
    first: list[tuple[dict[str, Any], Interval]],
    second: list[tuple[dict[str, Any], Interval]],
) -> BBox:
    """Bound all blend weights; share joint matching with the point evaluator."""
    from software_defined_matter._flexure_motion import compatible_joints
    from software_defined_matter.kinematics import _MotionOp

    def zero(_: Any) -> float:
        return 0.0

    def specs(chain: list[tuple[dict[str, Any], Interval]]) -> tuple[_MotionOp, ...]:
        return tuple(
            _MotionOp(
                op["kind"],
                tuple(v / math.hypot(*op["axis"]) for v in op["axis"]),
                tuple(op.get("origin", (0, 0, 0))),
                zero,
            )
            for op, _ in chain
        )

    a, b = specs(first), specs(second)
    if compatible_joints(a, b) is not None:
        if not first:
            first = [(op, (0.0, 0.0)) for op, _ in second]
            a = b
        if not second:
            second = [(op, (0.0, 0.0)) for op, _ in first]
            b = a
        swept = _padded(rest)
        for i, ((op, start), (_, end)) in enumerate(zip(first, second, strict=True)):
            sign = sum(x * y for x, y in zip(a[i].axis, b[i].axis, strict=True))
            lo, hi = sorted(sign * v for v in end)
            end = _finite((lo, hi))
            swept = _sweep(swept, op, (min(start[0], end[0]), max(start[1], end[1])))
        return swept

    # Principal screw: ||J(w omega)|| <= 1 and ||J(omega)^-1|| <= pi/2.
    # Thus ||posed(p)-t_from|| <= ||p|| + pi/2 ||t_to-t_from||.
    translations = []
    for chain in (first, second):
        box: BBox = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for op, interval in chain:
            box = _sweep(box, op, interval)
        translations.append(box)
    origin, target = translations
    difference = (
        _vec3(target[0][i] - origin[1][i] for i in range(3)),
        _vec3(target[1][i] - origin[0][i] for i in range(3)),
    )
    radius = max(math.hypot(*p) for p in _bbox_corners(rest))
    radius += math.pi / 2 * max(math.hypot(*p) for p in _bbox_corners(difference))
    return _padded(
        (_vec3(v - radius for v in origin[0]), _vec3(v + radius for v in origin[1])), radius
    )


def infer_motion_bounds(
    part: Part,
    *,
    smooth_csg: bool | None = None,
    include_flexures: bool = False,
    material_support: bool = False,
) -> MotionBounds | None:
    """Infer body, flexure, and scene boxes over their declared input ranges.

    Returns None when no motion block exists. Uses only referenced
    inputs; no poses are sampled or combined. Flexures sweep compatible joint
    intervals or use a conservative principal-screw sphere. With flexures or
    ``material_support=True``, every owner uses the complete material rest box,
    since classifier boxes need not
    enclose their owned material. Blend weights cover all of [0, 1].
    Set include_flexures=True only when consuming the returned flexure list;
    the default rejects flexures to protect existing body-only consumers. An
    authored ``metadata.bbox`` is included in the scene envelope. If rest-field
    inference fails, that authored box must cover the body's entire motion, and
    the body's box remains None to disable unsafe pruning. Motion expressions
    that cannot be bounded fail even with an authored box. Smoothed unions and
    onion shells require an authored box because static inference does not yet
    enclose their expanded fields. ``smooth_csg`` defaults to the metadata flag.
    """
    from software_defined_matter.io import validate
    from software_defined_matter.kinematics import resolve_region_tree

    if part.kinematics is None:
        return None
    part = copy.deepcopy(part)
    validate(part)
    if smooth_csg is None:
        smooth_csg = bool(part.metadata.get("smooth_csg", False))
    block = part.kinematics
    assert block is not None
    if block.get("flexures") and not include_flexures:
        raise BBoxInferenceError("Flexure motion bounds require include_flexures=True")
    authored = part.metadata.get("bbox")
    if authored is not None:
        if len(authored) != 2 or any(len(c) != 3 for c in authored):
            raise BBoxInferenceError("metadata.bbox must contain two finite 3-D corners")
        authored = (_vec3(authored[0]), _vec3(authored[1]))
        for interval in zip(*authored, strict=True):
            _finite(interval)
            if interval[0] == interval[1]:
                raise BBoxInferenceError("metadata.bbox needs finite positive extents")
    dofs = {}
    for dof in block.get("dofs", []):
        if dof["unit"] not in ({"rad", "deg"} if dof["kind"] == "angle" else {"mm"}):
            raise BBoxInferenceError("Motion input kind and unit are incompatible")
        factor = math.pi / 180 if dof["unit"] == "deg" else 1.0
        dofs[dof["name"]] = _finite((dof["range"][0] * factor, dof["range"][1] * factor))
    intervals = _Intervals(part, dofs)
    bodies = []
    operation_chains = []
    material_rest = None
    use_materials = bool(block.get("flexures")) or material_support
    if use_materials:
        try:
            if not part.materials:
                raise BBoxInferenceError("Material bounds require material geometry")
            material_rest = _bbox_enclose(
                [intervals.rest_bbox(m.sdf_tree, smooth_csg=smooth_csg) for m in part.materials]
            )
        except BBoxInferenceError as exc:
            if authored is None:
                label = "Flexure material support" if block.get("flexures") else "Material support"
                raise BBoxInferenceError(
                    f"{label}: {exc}. Supply metadata.bbox covering full motion"
                ) from exc
    boxes = [authored] if authored is not None else []
    for body in block.get("bodies", []):
        dependencies: set[str] = set()
        operations = [
            (
                op,
                intervals.expression(
                    op["angle"] if op["kind"] == "rotate" else op["distance"], dependencies
                ),
            )
            for op in body["motion"]["ops"]
        ]
        operation_chains.append(operations)
        region = resolve_region_tree(part, body["region"])
        try:
            rest = (
                material_rest
                if use_materials
                else intervals.rest_bbox(region, smooth_csg=smooth_csg)
            )
        except BBoxInferenceError as exc:
            if authored is None:
                raise BBoxInferenceError(
                    f"Body {body['name']!r}: {exc}. Supply metadata.bbox covering its full motion"
                ) from exc
            rest = None
        swept = _padded(rest) if rest is not None else None
        if swept is not None:
            for op, interval in operations:
                swept = _sweep(swept, op, interval)
            boxes.append(swept)
        bodies.append(
            BodyMotionBounds(body["name"], tuple(n for n in dofs if n in dependencies), rest, swept)
        )
    flexures = []
    index = {body.name: i for i, body in enumerate(bodies)}
    for flexure in block.get("flexures", []):
        first, second = index[flexure["from_body"]], index[flexure["to_body"]]
        reached = set(bodies[first].dof_names) | set(bodies[second].dof_names)
        swept = (
            _flexure_sweep(material_rest, operation_chains[first], operation_chains[second])
            if material_rest is not None
            else None
        )
        if swept is not None:
            boxes.append(swept)
        flexures.append(
            FlexureMotionBounds(
                flexure["name"], tuple(n for n in dofs if n in reached), material_rest, swept
            )
        )
    return MotionBounds(tuple(bodies), _bbox_enclose(boxes) if boxes else None, tuple(flexures))
