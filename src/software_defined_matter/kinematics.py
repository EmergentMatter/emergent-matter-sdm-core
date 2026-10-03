"""Differentiable body and flexure motion from a Part's authored kinematics block.

Points move from rest space to posed space. Operations apply in listed order,
using fixed rest-frame axes and origins. Positive rotation follows the right-hand
rule; querying a moved SDF requires the inverse transform instead.

Evaluator DOFs use radians for angles and millimetres for lengths. Authored degree
ranges and defaults are converted at compilation; ``to_evaluator_units`` converts
UI values before expression evaluation, including nonlinear expressions. Motion
``dof`` nodes read this vector, while ``param`` nodes read the supplied design vector
(including derived parameters), defaulting to the compilation snapshot.
Geometry expressions remain unchanged.

Ownership is the smallest rest-region distance. Bodies precede flexures; ties
follow document order.
Named regions retain ancestor transforms, deformations and modifiers; CSG siblings
are not part of the view. Flexure blends are evaluated in rest space. Compatible
joint chains interpolate authored coordinates; other endpoint frames use a
principal relative screw. Recompile after structural changes; numerical design
inputs may vary without recompilation.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from software_defined_matter.model import Part, SDFTree

__all__ = ["KinematicsEval", "compile_kinematics", "resolve_region_tree"]

_FRAME_TYPES = {"transform", "deform", "modifier", "2d_to_3d"}


def resolve_region_tree(part: Part, region: dict[str, Any]) -> SDFTree:
    """Copy an inline region or resolve a unique named geometry view in place.

    A named section nested inside a loft cannot independently define its 3-D
    region. Such references fail explicitly; name the complete loft instead.
    """
    if "$node" not in region:
        return copy.deepcopy(region)
    target = region["$node"]
    matches: list[SDFTree] = []

    def visit(node: Any, ancestors: list[SDFTree]) -> None:
        if not isinstance(node, dict):
            return
        if node.get("name") == target:
            tree = copy.deepcopy(node)
            for ancestor in reversed(ancestors):
                kind = ancestor.get("type")
                if kind == "op":
                    continue
                if kind not in _FRAME_TYPES:
                    raise ValueError(f"Region {target!r} is nested inside unsupported {kind!r}")
                shell = copy.deepcopy(ancestor)
                shell.pop("name", None)
                shell["child"] = tree
                tree = shell
            matches.append(tree)
        for child in ([node["child"]] if "child" in node else []) + node.get("children", []):
            visit(child, [*ancestors, node])

    for material in part.materials:
        visit(material.sdf_tree, [])
    if len(matches) != 1:
        raise ValueError(
            f"Region {target!r} must name exactly one geometry node; found {len(matches)}"
        )
    return matches[0]


def _compile_blend(field: dict[str, Any], binding: Any) -> Callable:
    import jax.numpy as jnp

    from software_defined_matter.sdf.compile import _compile_field_node

    if field.get("type") == "field" and field.get("kind") == "axis_ramp":
        params = field["params"]
        axis = jnp.asarray(params["axis"])
        lo, hi = params["lo"], params["hi"]

        def ramp(points: Any, free_vec: Any) -> Any:
            return (points @ axis - lo) / (hi - lo)

        return ramp
    if field.get("type") == "field" and field.get("kind") == "radial_hermite":
        params = field["params"]
        axis = jnp.asarray([v / math.hypot(*params["axis"]) for v in params["axis"]])
        origin = jnp.asarray(params["origin"])
        r0, r1 = params["r0"], params["r1"]

        def radial(points: Any, free_vec: Any) -> Any:
            delta = points - origin
            perpendicular = delta - (delta @ axis)[..., None] * axis
            radius2 = jnp.sum(perpendicular * perpendicular, axis=-1)
            # The clipped Hermite profile has zero derivative at the axis.
            # Keep the inactive sqrt branch finite under JAX reverse mode.
            radius = jnp.where(radius2 > 0, jnp.sqrt(jnp.where(radius2 > 0, radius2, 1)), 0)
            t = jnp.clip((radius - r0) / (r1 - r0), 0, 1)
            return t * t * (3 - 2 * t)

        return radial
    field_fn = _compile_field_node(field, binding)

    def evaluate(points: Any, free_vec: Any) -> Any:
        return field_fn(points, free_vec)

    return evaluate


def _compile_expression(tree: dict[str, Any], binding: Any, dof_index: dict[str, int]) -> Callable:
    # Share arithmetic with design expressions without adding DOF lookup to
    # their evaluator. The namespaces remain separate even when names match.
    import jax.numpy as jnp

    from software_defined_matter.dsl.expr import _BINARY, _REDUCE, _UNARY

    kind = tree["type"]
    if kind == "dof":
        name = tree["name"]
        if name not in dof_index:
            raise ValueError(f"Motion expression references unknown DOF {name!r}")
        index = dof_index[name]
        return lambda dofs, free_vec: dofs[index]
    if kind == "param":
        return lambda dofs, free_vec: binding.get(tree["name"], free_vec)
    if kind == "num":
        value = float(tree["value"])
        if not math.isfinite(value):
            raise ValueError("Motion expression constants must be finite")
        return lambda dofs, free_vec: jnp.asarray(value, dtype=dofs.dtype)
    if kind == "unop":
        op = _UNARY[tree["op"]]
        child = _compile_expression(tree["child"], binding, dof_index)
        return lambda dofs, free_vec: op(child(dofs, free_vec))
    if kind == "binop":
        binary = _BINARY[tree["op"]]
        lhs = _compile_expression(tree["lhs"], binding, dof_index)
        rhs = _compile_expression(tree["rhs"], binding, dof_index)
        return lambda dofs, free_vec: binary(lhs(dofs, free_vec), rhs(dofs, free_vec))
    if kind == "reduce":
        reduce = _REDUCE[tree["op"]]
        children = [_compile_expression(child, binding, dof_index) for child in tree["children"]]
        return lambda dofs, free_vec: reduce(
            jnp.stack([child(dofs, free_vec) for child in children])
        )
    raise ValueError(
        f"Unsupported motion expression {kind!r}; geometry metrics are not motion inputs"
    )


@dataclass(frozen=True)
class _MotionOp:
    kind: str
    axis: tuple[float, ...]
    origin: tuple[float, ...]
    value: Callable

    def matrix(self, dofs: Any, free_vec: Any) -> Any:
        import jax.numpy as jnp

        axis = jnp.asarray(self.axis, dtype=dofs.dtype)
        value = self.value(dofs, free_vec)
        result = jnp.eye(4, dtype=dofs.dtype)
        if self.kind == "translate":
            return result.at[:3, 3].set(axis * value)
        x, y, z = self.axis
        skew = jnp.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=dofs.dtype)
        cosine, sine = jnp.cos(value), jnp.sin(value)
        rotation = (
            cosine * jnp.eye(3, dtype=dofs.dtype)
            + (1 - cosine) * jnp.outer(axis, axis)
            + sine * skew
        )
        origin = jnp.asarray(self.origin, dtype=dofs.dtype)
        return result.at[:3, :3].set(rotation).at[:3, 3].set(origin - rotation @ origin)


def _float_array(values: Any) -> Any:
    import jax.numpy as jnp

    array = jnp.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(
        array.dtype, jnp.complexfloating
    ):
        raise ValueError("Motion inputs must be real numbers")
    return array.astype(jnp.result_type(array, 0.0))


@dataclass(frozen=True)
class KinematicsEval:
    """Compiled body and flexure motion, independent of later Part mutations.

    Every numerical method accepts keyword-only ``free_vec`` in the compilation
    binding's order. Omission uses ``design_defaults``. The design vector stays
    separate from runtime DOFs; differentiation flows through both. Fixed
    parameters remain fixed unless a supplied binding maps them to external inputs.
    Axes, origins, and specialized blend profiles retain their authored literals.

    Names follow document order. Ranges and defaults use radians/millimetres;
    ``dof_units`` records the original units for input conversion. Ranges describe
    UI limits, not clipping: evaluation also accepts values outside those limits.
    Empty body lists leave points unchanged and report ownership -1.
    ``inverse_flexure_names`` lists flexures with a supported per-region inverse;
    it does not certify global ownership inversion or shader rendering.
    """

    dof_names: tuple[str, ...]
    dof_units: tuple[str, ...]
    dof_ranges: tuple[tuple[float, float], ...]
    dof_defaults: tuple[float, ...]
    region_names: tuple[str, ...]
    _regions: tuple[Callable, ...]
    _motions: tuple[tuple[_MotionOp, ...], ...]
    _flexures: tuple[Any, ...] = ()
    inverse_flexure_names: tuple[str, ...] = ()
    design_defaults: tuple[float, ...] = ()
    port_names: tuple[str, ...] = ()
    _port_frames: tuple[Callable, ...] = ()
    _port_bodies: tuple[int | None, ...] = ()

    def port_transforms(self, dofs: Any, *, free_vec: Any = None) -> Any:
        """Return (P, 4, 4) port frames in posed part coordinates, in port_names order.

        A body-attached frame is authored in the part's rest coordinates and follows
        that body's rest-to-posed map. Reference-frame ports stay independent of
        body motion. A rigid flexure-end attachment uses its endpoint body's name.
        """
        import jax.numpy as jnp

        design = self._design(free_vec)
        bodies = self.body_transforms(dofs, free_vec=design)
        frames = []
        for frame, body in zip(self._port_frames, self._port_bodies, strict=True):
            matrix = frame(design)
            frames.append(matrix if body is None else bodies[body] @ matrix)
        return jnp.stack(frames) if frames else jnp.empty((0, 4, 4), dtype=bodies.dtype)

    def _design(self, free_vec: Any) -> Any:
        array = _float_array(self.design_defaults if free_vec is None else free_vec)
        if array.shape != (len(self.design_defaults),):
            raise ValueError(
                f"Expected design shape {(len(self.design_defaults),)}, got {array.shape}"
            )
        return array

    def n_dofs(self) -> int:
        """Return the required length of the motion input vector."""
        return len(self.dof_names)

    def _dofs(self, values: Any) -> Any:
        array = _float_array(values)
        if array.shape != (self.n_dofs(),):
            raise ValueError(f"Expected DOF shape {(self.n_dofs(),)}, got {array.shape}")
        return array

    def to_evaluator_units(self, values: Any) -> Any:
        """Convert an authored-unit DOF vector to radians and millimetres."""
        import jax.numpy as jnp

        array = self._dofs(values)
        return array * jnp.asarray(
            [math.pi / 180 if unit == "deg" else 1 for unit in self.dof_units], dtype=array.dtype
        )

    def body_transforms(self, dofs: Any, *, free_vec: Any = None) -> Any:
        """Return body-ordered (N, 4, 4) rest-to-posed matrices for column vectors.

        Each listed operation acts on the result of the previous operation.
        The inverse rigid transform maps a posed query point back to rest space.
        """
        import jax.numpy as jnp

        values = self._dofs(dofs)
        design = self._design(free_vec)
        matrices = []
        for ops in self._motions:
            matrix = jnp.eye(4, dtype=values.dtype)
            for op in ops:
                matrix = op.matrix(values, design) @ matrix
            matrices.append(matrix)
        return jnp.stack(matrices) if matrices else jnp.empty((0, 4, 4), dtype=values.dtype)

    def flexure_transforms(self, points: Any, dofs: Any, *, free_vec: Any = None) -> Any:
        """Return (F, ..., 4, 4) rest-to-posed transforms at rest points (..., 3).

        Flexures follow authored order. The blend is clamped to [0, 1]. These
        point-dependent matrices are forward maps, not an inverse deformation
        evaluator: recomputing the blend at a posed point generally changes it.
        """
        import jax.numpy as jnp

        points = _points(points)
        values = self._dofs(dofs)
        design = self._design(free_vec)
        bodies = self.body_transforms(values, free_vec=design)
        return (
            jnp.stack([f.matrices(points, values, bodies, design) for f in self._flexures])
            if self._flexures
            else jnp.empty((0, *points.shape[:-1], 4, 4), dtype=values.dtype)
        )

    def inverse_flexure_points(
        self, points: Any, dofs: Any, *, flexure: str, free_vec: Any = None
    ) -> Any:
        """Map posed queries (..., 3) to rest points for one supported flexure.

        ``inverse_flexure_names`` lists the supported names in document order.
        DOFs use evaluator radians/mm. Unknown names or unsupported maps raise
        ValueError. This is a per-flexure map, not an inverse of global ownership
        or a signed-distance evaluator. Different regions may overlap when posed.
        JAX differentiation uses the full point-dependent inverse Jacobian.
        """
        import jax.numpy as jnp

        names = self.region_names[len(self._motions) :]
        if flexure not in names:
            raise ValueError(f"Unknown flexure {flexure!r}")
        if flexure not in self.inverse_flexure_names:
            raise ValueError(f"Flexure {flexure!r} has no supported invariant-blend inverse")
        points, values = _points(points), self._dofs(dofs)
        design = self._design(free_vec)
        matrix = self._flexures[names.index(flexure)].matrices(
            points, values, self.body_transforms(values, free_vec=design), design
        )
        return jnp.einsum("...ji,...j->...i", matrix[..., :3, :3], points - matrix[..., :3, 3])

    def ownership(self, points: Any, *, free_vec: Any = None) -> Any:
        """Return nearest rest-region indices for points shaped (..., 3).

        Bodies precede flexures, each in authored order. Ties select the first.
        """
        import jax.numpy as jnp

        points = _points(points)
        design = self._design(free_vec)
        if not self._regions:
            return jnp.full(points.shape[:-1], -1, dtype=jnp.int32)
        distances = jnp.stack([region(points, design) for region in self._regions], axis=-1)
        return jnp.argmin(distances, axis=-1)

    def pose_points(
        self, points: Any, dofs: Any, owner: Any = None, *, free_vec: Any = None
    ) -> Any:
        """Move rest points, optionally reusing ownership computed in rest space.

        Differentiable in DOFs and points within each ownership region. Ownership
        boundaries themselves are discrete. Precomputed owners must have the point
        batch shape and integer dtype; out-of-range indices produce NaNs, including
        under JIT, rather than silently selecting a different body.
        """
        import jax.numpy as jnp

        points = _points(points)
        design = self._design(free_vec)
        matrices = self.body_transforms(dofs, free_vec=design)
        if not self._regions:
            return points
        owner = self.ownership(points, free_vec=design) if owner is None else jnp.asarray(owner)
        if owner.shape != points.shape[:-1] or not jnp.issubdtype(owner.dtype, jnp.integer):
            raise ValueError("Ownership must be an integer array matching the point batch shape")
        matrix = matrices[jnp.clip(owner, 0, len(self._motions) - 1)]
        posed = jnp.einsum("...ij,...j->...i", matrix[..., :3, :3], points) + matrix[..., :3, 3]
        for i, flexure in enumerate(self._flexures, start=len(self._motions)):
            local = flexure.matrices(points, self._dofs(dofs), matrices, design)
            candidate = (
                jnp.einsum("...ij,...j->...i", local[..., :3, :3], points) + local[..., :3, 3]
            )
            posed = jnp.where((owner == i)[..., None], candidate, posed)
        valid = (owner >= 0) & (owner < len(self._regions))
        return jnp.where(valid[..., None], posed, jnp.nan)


def _points(values: Any) -> Any:
    points = _float_array(values)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError(f"Expected point shape (..., 3), got {points.shape}")
    return points


def compile_kinematics(part: Part, *, binding: Any = None) -> KinematicsEval | None:
    """Compile a snapshot of ``part.kinematics``; return None if absent.

    ``binding`` optionally supplies an occurrence-scoped parameter binding; its
    free vector then follows the root assembly layout. Otherwise use the part's
    free-parameter order. A copy of the binding is retained to isolate this snapshot.
    Zero-motion checks run at the default design; callers must retain the authored
    zero-motion identity invariant over their design domain.

    Validate before constructing JAX functions. Motion expressions support the
    arithmetic DSL plus motion-only ``dof`` references, but not metric queries.
    Every body's all-zero-DOF transform must be identity. Flexure regions follow
    bodies in ownership order. Blend fields use the existing scalar-field DSL
    and the rest-space axis_ramp field: clip((dot(p, axis)-lo)/(hi-lo), 0, 1).
    ``radial_hermite`` uses a normalized axis and an origin in global rest space.
    With radius r from that axis line, t=clip((r-r0)/(r1-r0),0,1) and blend is
    t*t*(3-2*t). Axis and origin are finite numeric vec3 values, with a nonzero
    axis and 0 <= r0 < r1. The profile matches the existing radial twist operator.
    Compatible joint chains retain authored turns; other frames use the shortest
    relative screw, whose rotation branch is discontinuous at a half turn.
    """
    import jax.numpy as jnp
    import numpy as np

    from software_defined_matter.dsl.resolve import make_binding
    from software_defined_matter.io import validate
    from software_defined_matter.sdf.compile import make_sdf_closure_with_binding

    if part.kinematics is None:
        return None
    snapshot = copy.deepcopy(part)
    validate(snapshot)
    kin = snapshot.kinematics
    assert kin is not None
    dofs, bodies = kin.get("dofs", []), kin.get("bodies", [])
    index = {dof["name"]: i for i, dof in enumerate(dofs)}
    factors = []
    for dof in dofs:
        allowed = {"rad", "deg"} if dof["kind"] == "angle" else {"mm"}
        if dof["unit"] not in allowed:
            raise ValueError(f"DOF {dof['name']!r}: incompatible kind and unit")
        factors.append(math.pi / 180 if dof["unit"] == "deg" else 1.0)
    binding = make_binding(snapshot) if binding is None else copy.deepcopy(binding)
    design_defaults = tuple(float(v) for v in binding.initial_free_vector())
    motions = []
    for body in bodies:
        ops = []
        for op in body["motion"]["ops"]:
            norm = math.hypot(*op["axis"])
            axis = tuple(float(x) / norm for x in op["axis"])
            origin = tuple(float(x) for x in op.get("origin", (0, 0, 0)))
            if not all(math.isfinite(x) for x in origin):
                raise ValueError(f"Body {body['name']!r}: origin must be finite")
            expr = op["angle"] if op["kind"] == "rotate" else op["distance"]
            ops.append(
                _MotionOp(op["kind"], axis, origin, _compile_expression(expr, binding, index))
            )
        motions.append(tuple(ops))
    from software_defined_matter._flexure_motion import (
        FlexureMotion,
        compatible_joints,
        invariant_blend,
    )

    body_index = {body["name"]: i for i, body in enumerate(bodies)}
    flexures = kin.get("flexures", [])
    compiled_flexures = []
    inverse_names = []
    for flexure in flexures:
        blend = _compile_blend(flexure["blend"], binding)
        first, second = body_index[flexure["from_body"]], body_index[flexure["to_body"]]
        joints = compatible_joints(motions[first], motions[second])
        if joints is not None:
            zero = jnp.zeros(len(dofs))
            if any(
                not np.isclose(
                    start(zero, jnp.asarray(design_defaults)),
                    end(zero, jnp.asarray(design_defaults)),
                    atol=1e-7,
                    rtol=0,
                )
                for _, start, end in joints
            ):
                raise ValueError(
                    f"Flexure {flexure['name']!r}: zero DOFs must have "
                    "matching endpoint joint coordinates"
                )
        compiled_flexures.append(FlexureMotion(first, second, blend, joints))
        if invariant_blend(flexure["blend"], joints):
            inverse_names.append(flexure["name"])
    result = KinematicsEval(
        dof_names=tuple(d["name"] for d in dofs),
        dof_units=tuple(d["unit"] for d in dofs),
        dof_ranges=tuple(
            (d["range"][0] * f, d["range"][1] * f) for d, f in zip(dofs, factors, strict=True)
        ),
        dof_defaults=tuple(d.get("default", 0) * f for d, f in zip(dofs, factors, strict=True)),
        region_names=tuple(b["name"] for b in [*bodies, *flexures]),
        _regions=tuple(
            make_sdf_closure_with_binding(
                resolve_region_tree(snapshot, b["region"]),
                binding,
                b_smooth_csg=bool(snapshot.metadata.get("smooth_csg", False)),
            )
            for b in [*bodies, *flexures]
        ),
        _motions=tuple(motions),
        _flexures=tuple(compiled_flexures),
        inverse_flexure_names=tuple(inverse_names),
        design_defaults=design_defaults,
        port_names=tuple(p.name for p in snapshot.ports),
        _port_frames=tuple(
            lambda design, frame=p.frame: frame.evaluate(binding, design) for p in snapshot.ports
        ),
        _port_bodies=tuple(
            body_index[p.body] if p.body is not None else None for p in snapshot.ports
        ),
    )
    for body, matrix in zip(
        bodies, np.asarray(result.body_transforms(jnp.zeros(len(dofs)))), strict=True
    ):
        if not np.allclose(matrix, np.eye(4), atol=1e-6, rtol=1e-6):
            raise ValueError(
                f"Body {body['name']!r}: zero DOFs must produce the identity transform"
            )
    return result
