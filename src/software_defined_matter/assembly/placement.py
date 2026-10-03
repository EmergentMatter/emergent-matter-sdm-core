"""Compile a grounded mate graph, then evaluate poses without solving mechanisms.

Topology and traversal are fixed at compilation. Design values and independent
coordinates remain numerical inputs. Nested assemblies place their own children
before their exposed ports participate in the parent's mate graph.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

from software_defined_matter.assembly.bundle import AssemblyBundle, ResolvedPort
from software_defined_matter.assembly.model import Assembly, Mate
from software_defined_matter.model import Part

__all__ = ["PlacementEval", "PlacementState", "PlacementError", "compile_placement"]


class PlacementState(NamedTuple):
    """JAX-compatible result with world poses and mate residuals.

    ``instances`` includes assemblies and leaf parts, keyed by full occurrence
    address; the empty address is the identity root frame. ``ports`` includes
    terminal and promoted addresses. Residual rows follow ``mate_names`` on the
    evaluator: translation in mm then a principal rotation vector in radians,
    both in the expected child-port frame. ``valid`` checks finiteness and fixed
    closure tolerances. A finite pose alone does not establish a valid state.
    ``coordinates`` contains all canonical independent and driven DOFs in rad/mm.
    ``bodies`` maps occurrence-qualified body names to rest-part-to-world matrices.
    Flexure validity at individual points is checked by ``pose_points``; the
    state does not certify a deformation over an entire continuous region.
    """

    instances: dict[str, Any]
    ports: dict[str, Any]
    residuals: Any
    valid: Any
    coordinates: dict[str, Any]
    bodies: dict[str, Any]


class PlacementError(ValueError):
    """A supplied configuration violates mates or contains non-finite motion/geometry."""


@dataclass(frozen=True)
class _Edge:
    mate: Mate
    parent_owner: str
    child_owner: str
    parent_port: ResolvedPort
    child_port: ResolvedPort
    coordinate: int | None


@dataclass(frozen=True)
class _Step:
    source: str
    target: str
    edge: _Edge
    reverse: bool


@dataclass(frozen=True)
class _Plan:
    grounds: tuple[str, ...]
    steps: tuple[_Step, ...]
    edges: tuple[_Edge, ...]


def _address(scope: str, name: str) -> str:
    return f"{scope}.{name}" if scope else name


def _inverse(matrix: Any) -> Any:
    import jax.numpy as jnp

    rotation = matrix[:3, :3].T
    result = jnp.eye(4, dtype=matrix.dtype)
    return result.at[:3, :3].set(rotation).at[:3, 3].set(-rotation @ matrix[:3, 3])


def _joint(kind: str, value: Any, dtype: Any) -> Any:
    import jax.numpy as jnp

    result = jnp.eye(4, dtype=dtype)
    if kind == "revolute":
        c, s = jnp.cos(value), jnp.sin(value)
        return result.at[:3, :3].set(jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    if kind == "prismatic":
        return result.at[2, 3].set(value)
    return result


class PlacementEval:
    """Compiled occurrence placement with fixed graph structure and input layouts.

    Construct with ``compile_placement``. ``design_defaults`` follows the root
    free-parameter vector; ``dof_names`` lists canonical independent coordinates,
    sorted by occurrence and then coordinate name. Values use radians/mm even
    when ``dof_units`` contains authored degrees. Promotions are aliases, so they
    never add another input. Expression-driven coordinates are excluded from inputs.
    Binding expressions consume and produce radians/mm, including for degree
    declarations. Parameter values keep their design units. ``dof_ranges`` uses
    evaluator units and describes independent input bounds for consumers.

    Use ``evaluate_checked`` for host-side inspection/export. ``evaluate`` is the
    pure JAX path; callers must inspect its ``valid`` flag or residuals. No API here
    searches for a closed-mechanism solution. Runtime ranges remain UI bounds,
    consistent with the part kinematics evaluator; they do not clip coordinates.
    """

    def __init__(
        self, bundle: AssemblyBundle, position_tolerance: float, angle_tolerance: float
    ) -> None:
        from software_defined_matter.kinematics import compile_kinematics

        self._bundle = copy.deepcopy(bundle)
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        definitions = self._bundle.definitions
        self.design_defaults = tuple(float(v) for v in self._bundle.binding().initial_free_vector())
        self._bindings = {scope: self._bundle.binding(scope) for scope in definitions}
        self._motion = {
            scope: compile_kinematics(doc, binding=self._bindings[scope])
            for scope, doc in definitions.items()
            if isinstance(doc, Part)
        }
        from software_defined_matter.assembly._motion import _MotionGraph

        self._coordinates = _MotionGraph(self._bundle)
        self._indices = self._coordinates.indices
        self.dof_names = self._coordinates.input_names
        self.dof_units = self._coordinates.units
        self.dof_defaults = self._coordinates.defaults
        self.dof_ranges = self._coordinates.ranges
        self._plans = {
            scope: self._plan(scope, doc)
            for scope, doc in definitions.items()
            if isinstance(doc, Assembly)
        }
        self.mate_names = tuple(
            _address(scope, edge.mate.id)
            for scope in sorted(self._plans)
            for edge in self._plans[scope].edges
        )

    def _plan(self, scope: str, doc: Assembly) -> _Plan:
        children = {_address(scope, i.id): i for i in doc.instances}
        grounds = tuple(sorted(name for name, i in children.items() if i.transform is not None))

        def owner(port: ResolvedPort) -> str:
            relative = port.instance[len(scope) + 1 :] if scope else port.instance
            return _address(scope, relative.split(".")[0])

        edges = []
        for mate in sorted(doc.mates, key=lambda m: m.id):
            parent = self._bundle.resolve_port(mate.parent, scope=scope)
            child = self._bundle.resolve_port(mate.child, scope=scope)
            coordinate = (
                self._indices[_address(*self._bundle.resolve_dof(mate.dof, scope=scope))]
                if mate.dof is not None
                else None
            )
            edges.append(_Edge(mate, owner(parent), owner(child), parent, child, coordinate))
        placed = set(grounds)
        steps = []
        while True:
            progress = False
            for edge in edges:
                a, b = edge.parent_owner, edge.child_owner
                if (a in placed) == (b in placed):
                    continue
                reverse = b in placed
                source, target = (b, a) if reverse else (a, b)
                steps.append(_Step(source, target, edge, reverse))
                placed.add(target)
                progress = True
            if not progress:
                break
        missing = sorted(set(children) - placed)
        if missing:
            raise ValueError(f"Assembly {scope or '<root>'!r}: ungrounded instances {missing}")
        return _Plan(grounds, tuple(steps), tuple(edges))

    def _vector(self, values: Any, defaults: tuple[float, ...], name: str) -> Any:
        from software_defined_matter.kinematics import _float_array

        array = _float_array(defaults if values is None else values)
        if array.shape != (len(defaults),):
            raise ValueError(f"Expected {name} shape {(len(defaults),)}, got {array.shape}")
        return array

    def to_evaluator_units(self, dofs: Any = None) -> Any:
        """Convert explicit authored coordinates in dof_names order to radians/mm.

        None (or omission) returns the already-converted defaults, matching
        ``evaluate(dofs=None)``. Explicit vectors always use authored units.
        """
        import jax.numpy as jnp

        values = self._vector(dofs, self.dof_defaults, "DOF")
        if dofs is None:
            return values
        return values * jnp.asarray([math.pi / 180 if u == "deg" else 1 for u in self.dof_units])

    def evaluate(self, *, free_vec: Any = None, dofs: Any = None) -> PlacementState:
        """Evaluate poses and closure residuals without host-side branching on values.

        Inspect ``valid`` before using poses as an accepted configuration. This
        function is JIT/grad compatible; discrete topology was compiled beforehand.
        Rotation residuals use the principal branch, discontinuous at a half turn.
        """
        import jax.numpy as jnp

        from software_defined_matter._flexure_motion import _rotation_vector

        design = self._vector(free_vec, self.design_defaults, "design")
        inputs = self._vector(dofs, self.dof_defaults, "DOF")
        coordinates = self._coordinates.evaluate(inputs, design)
        dtype = jnp.result_type(design, coordinates)
        identity = jnp.eye(4, dtype=dtype)
        subtree: dict[str, dict[str, Any]] = {}
        local_ports: dict[str, dict[str, Any]] = {}
        residuals: dict[str, Any] = {}
        local_bodies: dict[str, dict[str, Any]] = {}
        definitions = self._bundle.definitions
        # Depth ordering makes this independent of the loader's insertion order.
        scopes = sorted(definitions, key=lambda s: (len(s.split(".")) if s else 0, s), reverse=True)
        for scope in scopes:
            doc = definitions[scope]
            binding = self._bindings[scope]
            if isinstance(doc, Part):
                motion = self._motion[scope]
                if motion is None:
                    frames = {p.name: p.frame.evaluate(binding, design) for p in doc.ports}
                else:
                    q = jnp.asarray(
                        [coordinates[self._indices[_address(scope, n)]] for n in motion.dof_names],
                        dtype=dtype,
                    )
                    body_matrices = motion.body_transforms(q, free_vec=design)
                    local_bodies[scope] = {
                        name: body_matrices[i]
                        for i, name in enumerate(motion.region_names[: len(body_matrices)])
                    }
                    matrices = motion.port_transforms(q, free_vec=design)
                    frames = {name: matrices[i] for i, name in enumerate(motion.port_names)}
                local_ports[scope] = frames
                subtree[scope] = {scope: identity}
                continue
            plan = self._plans[scope]
            by_name = {_address(scope, i.id): i for i in doc.instances}
            poses = {}
            for name in plan.grounds:
                frame = by_name[name].transform
                assert frame is not None
                poses[name] = frame.evaluate(binding, design)

            def endpoint(owner: str, port: ResolvedPort) -> Any:
                return subtree[owner][port.instance] @ local_ports[port.instance][port.port.name]

            def relation(edge: _Edge, binding: Any = binding) -> Any:
                value = coordinates[edge.coordinate] if edge.coordinate is not None else 0.0
                return edge.mate.offset.evaluate(binding, design) @ _joint(
                    edge.mate.kind, value, dtype
                )

            for step in plan.steps:
                edge = step.edge
                parent_frame = endpoint(edge.parent_owner, edge.parent_port)
                child_frame = endpoint(edge.child_owner, edge.child_port)
                relative = parent_frame @ relation(edge) @ _inverse(child_frame)
                poses[step.target] = poses[step.source] @ (
                    _inverse(relative) if step.reverse else relative
                )
            for edge in plan.edges:
                expected = (
                    poses[edge.parent_owner]
                    @ endpoint(edge.parent_owner, edge.parent_port)
                    @ relation(edge)
                )
                actual = poses[edge.child_owner] @ endpoint(edge.child_owner, edge.child_port)
                delta = _inverse(expected) @ actual
                residuals[_address(scope, edge.mate.id)] = jnp.concatenate(
                    (delta[:3, 3], _rotation_vector(delta[:3, :3]))
                )
            descendants = {scope: identity}
            for child in sorted(by_name):
                descendants.update(
                    {name: poses[child] @ pose for name, pose in subtree[child].items()}
                )
            subtree[scope] = descendants
        world = subtree[""]
        bodies = {
            _address(scope, name): world[scope] @ matrix
            for scope, frames in local_bodies.items()
            for name, matrix in frames.items()
        }
        ports = {}
        for scope, doc in definitions.items():
            if isinstance(doc, Part):
                for name, frame in local_ports[scope].items():
                    ports[_address(scope, name)] = world[scope] @ frame
        for scope, doc in definitions.items():
            if isinstance(doc, Assembly):
                for name in doc.port:
                    target = self._bundle.resolve_port(name, scope=scope)
                    ports[_address(scope, name)] = ports[
                        _address(target.instance, target.port.name)
                    ]
        errors = (
            jnp.stack([residuals[name] for name in self.mate_names])
            if self.mate_names
            else jnp.empty((0, 6), dtype=dtype)
        )
        finite = jnp.all(jnp.isfinite(design)) & jnp.all(jnp.isfinite(coordinates))
        for matrix in (*world.values(), *ports.values(), *bodies.values()):
            finite = finite & jnp.all(jnp.isfinite(matrix))
        valid = (
            finite
            & jnp.all(jnp.isfinite(errors))
            & jnp.all(jnp.linalg.norm(errors[:, :3], axis=1) <= self.position_tolerance)
            & jnp.all(jnp.linalg.norm(errors[:, 3:], axis=1) <= self.angle_tolerance)
        )
        return PlacementState(
            world,
            ports,
            errors,
            valid,
            dict(zip(self._coordinates.names, coordinates, strict=True)),
            bodies,
        )

    def pose_points(
        self,
        instance: str,
        points: Any,
        *,
        free_vec: Any = None,
        dofs: Any = None,
        owner: Any = None,
    ) -> Any:
        """Map a leaf occurrence's rest points (..., 3) to assembly world space.

        Internal body/flexure motion acts before occurrence placement. ``owner``
        follows the part kinematics region order, or is inferred in rest space.
        Invalid placement, ownership or deformation produces NaNs under JIT as
        well as eager evaluation. Use ``pose_points_checked`` at host boundaries.
        This forward map does not claim a deformed signed-distance function.
        """
        design = self._vector(free_vec, self.design_defaults, "design")
        state = self.evaluate(free_vec=design, dofs=dofs)
        return self._pose_points(instance, points, state, design, owner)

    def _pose_points(
        self, instance: str, points: Any, state: PlacementState, design: Any, owner: Any
    ) -> Any:
        """Pose points from an already evaluated ``state`` so callers evaluate once."""
        import jax.numpy as jnp

        from software_defined_matter.kinematics import _points

        if instance not in self._motion:
            raise ValueError(f"Expected a leaf part occurrence, got {instance!r}")
        points = _points(points)
        motion = self._motion[instance]
        if owner is not None and (motion is None or not motion.region_names):
            raise ValueError(f"Part occurrence {instance!r} has no kinematic ownership regions")
        if motion is not None:
            q = jnp.asarray([state.coordinates[_address(instance, n)] for n in motion.dof_names])
            points = motion.pose_points(points, q, owner=owner, free_vec=design)
        matrix = state.instances[instance]
        world = jnp.einsum("ij,...j->...i", matrix[:3, :3], points) + matrix[:3, 3]
        return jnp.where(
            state.valid & jnp.all(jnp.isfinite(world), axis=-1, keepdims=True), world, jnp.nan
        )

    def pose_points_checked(
        self,
        instance: str,
        points: Any,
        *,
        free_vec: Any = None,
        dofs: Any = None,
        owner: Any = None,
    ) -> Any:
        """Reject invalid placement or sampled point motion at a host-side boundary."""
        import numpy as np

        design = self._vector(free_vec, self.design_defaults, "design")
        state = self.evaluate_checked(free_vec=design, dofs=dofs)
        result = self._pose_points(instance, points, state, design, owner)
        if not np.isfinite(result).all():
            raise PlacementError(
                f"Invalid posed points for {instance!r}: non-finite motion or ownership"
            )
        return result

    def evaluate_checked(self, *, free_vec: Any = None, dofs: Any = None) -> PlacementState:
        """Reject invalid configurations with offending mate names and errors in mm/rad.

        Host-side boundary only; use ``evaluate`` inside JAX transformations.
        """
        import numpy as np

        state = self.evaluate(free_vec=free_vec, dofs=dofs)
        if not bool(state.valid):
            details: list[str] = []
            for label, values in (
                ("DOF", state.coordinates),
                ("body", state.bodies),
                ("port", state.ports),
                ("instance", state.instances),
            ):
                details.extend(
                    f"{label} {name!r}: non-finite value"
                    for name, value in values.items()
                    if not np.isfinite(value).all()
                )
            for name, row in zip(self.mate_names, np.asarray(state.residuals), strict=True):
                position, angle = float(np.linalg.norm(row[:3])), float(np.linalg.norm(row[3:]))
                if (
                    not np.isfinite(row).all()
                    or position > self.position_tolerance
                    or angle > self.angle_tolerance
                ):
                    details.append(f"{name}: position={position:.6g} mm, angle={angle:.6g} rad")
            raise PlacementError(
                "Invalid placement: " + ("; ".join(details) or "non-finite inputs or frames")
            )
        return state


def compile_placement(
    bundle: AssemblyBundle, *, position_tolerance: float = 1e-5, angle_tolerance: float = 1e-5
) -> PlacementEval:
    """Compile deterministic placement from explicit grounds and port mates.

    Every direct-child connected component needs an explicit ``Instance.transform``.
    Multiple grounds are retained; all mate edges, including non-tree edges, are
    checked against the resulting poses. Mates within a placed subassembly can be
    inspected from the parent but cannot solve that subassembly's internal graph.
    Tolerances are fixed acceptance inputs (mm/rad), separate from design variables.
    A bundle without mates retains the explicit placement behavior of schema 0.5.

    Raises:
        ValueError: Invalid references, tolerances or ungrounded components.
    """
    for name, value in (
        ("position_tolerance", position_tolerance),
        ("angle_tolerance", angle_tolerance),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name}: expected a finite positive tolerance, got {value!r}")
    return PlacementEval(bundle, position_tolerance, angle_tolerance)
