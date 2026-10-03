"""Rigid attachment frames evaluated in an instance's design-parameter scope.

Frames describe rest placement in part coordinates. A named body attachment
identifies the subsequent kinematic transform; it never changes the frame's
parameter scope. Quaternions use scalar-first (w, x, y, z) ordering.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Frame", "Port", "validate_identifier", "validate_scalar", "scalar_refs"]


def validate_identifier(value: str, where: str) -> None:
    """Reject names that make dotted instance addressing ambiguous."""
    if not isinstance(value, str) or re.fullmatch(r"[a-z_][a-z0-9_]*", value) is None:
        raise ValueError(f"{where}: expected an identifier [a-z_][a-z0-9_]*, got {value!r}")


def validate_scalar(value: Any, where: str) -> None:
    """Validate a finite scalar or a pure parameter expression without importing JAX."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return
    elif isinstance(value, dict):
        if set(value) == {"$ref"} and isinstance(value["$ref"], str):
            return
        kind = value.get("type")
        if kind == "num" and set(value) == {"type", "value"}:
            validate_scalar(value["value"], where)
            if isinstance(value["value"], (int, float)):
                return
        if kind == "param" and set(value) == {"type", "name"} and isinstance(value["name"], str):
            return
        keys: set[str] = set()
        children: list[Any] = []
        if kind == "unop" and value.get("op") in {
            "neg",
            "abs",
            "sqrt",
            "sin",
            "cos",
            "exp",
            "log",
            "square",
        }:
            keys = {"type", "op", "child"}
            children = [value.get("child")]
        elif kind == "binop" and value.get("op") in {"+", "-", "*", "/", "pow", "min", "max"}:
            keys = {"type", "op", "lhs", "rhs"}
            children = [value.get("lhs"), value.get("rhs")]
        elif kind == "reduce" and value.get("op") in {"sum", "mean", "min", "max"}:
            keys = {"type", "op", "children"}
            children = value.get("children", [])
            if not isinstance(children, list) or not children:
                raise ValueError(
                    f"{where}: expected nonempty expression children, got {children!r}"
                )
        if keys and set(value) == keys:
            for child in children:
                validate_scalar(child, where)
            return
    raise ValueError(
        f"{where}: expected a finite number or pure parameter expression, got {value!r}"
    )


def scalar_refs(value: Any) -> set[str]:
    """Collect parameter names without evaluating a frame or importing JAX."""
    if isinstance(value, dict):
        own = {value["$ref"]} if "$ref" in value else set()
        if value.get("type") == "param":
            own.add(value["name"])
        return own.union(*(scalar_refs(v) for v in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(scalar_refs(v) for v in value))
    return set()


@dataclass(frozen=True)
class Frame:
    """Rest position in mm and unit quaternion, optionally parameter-dependent.

    Expression quaternions are normalized on evaluation. Their norm must be
    nonzero throughout the design domain; a zero norm produces an invalid frame,
    never an identity fallback. Authoring helpers preserve expression dependence.
    """

    position: tuple[Any, Any, Any] = (0.0, 0.0, 0.0)
    orientation: tuple[Any, ...] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name, size in (("position", 3), ("orientation", 4)):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or len(values) != size:
                raise ValueError(f"Frame.{name}: expected {size} components, got {values!r}")
            for value in values:
                validate_scalar(value, f"Frame.{name}")
            object.__setattr__(self, name, tuple(copy.deepcopy(values)))
        if all(isinstance(v, (int, float)) for v in self.orientation):
            norm = math.hypot(*self.orientation)
            if not math.isclose(norm, 1.0, abs_tol=1e-8):
                raise ValueError(f"Frame.orientation: expected a unit quaternion, got norm {norm}")

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON frame data."""
        return copy.deepcopy(
            {"position": list(self.position), "orientation": list(self.orientation)}
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Frame:
        """Read a complete frame, rejecting misspelled fields."""
        if set(data) != {"position", "orientation"}:
            raise ValueError(f"Frame: expected position and orientation, got {sorted(data)}")
        return cls(data["position"], data["orientation"])

    @classmethod
    def from_axis_angle(
        cls,
        axis: tuple[float, float, float],
        angle: Any,
        *,
        position: tuple[Any, Any, Any] = (0.0, 0.0, 0.0),
    ) -> Frame:
        """Build a quaternion from a finite literal axis and an angle in radians."""
        if len(axis) != 3 or not all(math.isfinite(v) for v in axis) or math.hypot(*axis) == 0:
            raise ValueError(f"Frame.axis: expected a finite nonzero 3-vector, got {axis!r}")
        validate_scalar(angle, "Frame.angle")
        axis = (
            axis[0] / math.hypot(*axis),
            axis[1] / math.hypot(*axis),
            axis[2] / math.hypot(*axis),
        )
        if isinstance(angle, (int, float)):
            return cls(position, (math.cos(angle / 2), *(v * math.sin(angle / 2) for v in axis)))
        half = {"type": "binop", "op": "*", "lhs": angle, "rhs": {"type": "num", "value": 0.5}}
        sin = {"type": "unop", "op": "sin", "child": half}
        return cls(
            position,
            (
                {"type": "unop", "op": "cos", "child": half},
                *(
                    {"type": "binop", "op": "*", "lhs": {"type": "num", "value": v}, "rhs": sin}
                    for v in axis
                ),
            ),
        )

    @classmethod
    def from_normal_tangent(
        cls,
        normal: tuple[float, float, float],
        tangent: tuple[float, float, float],
        *,
        position: tuple[Any, Any, Any] = (0.0, 0.0, 0.0),
    ) -> Frame:
        """Build a literal orientation with Z along normal and X along projected tangent."""
        import numpy as np

        z, x = np.asarray(normal, dtype=float), np.asarray(tangent, dtype=float)
        if (
            z.shape != (3,)
            or x.shape != (3,)
            or not np.all(np.isfinite([z, x]))
            or np.linalg.norm(z) == 0
        ):
            raise ValueError("Frame.normal/tangent: expected finite nondegenerate 3-vectors")
        z = z / np.linalg.norm(z)
        x = x - np.dot(x, z) * z
        if np.linalg.norm(x) < 1e-12:
            raise ValueError("Frame.tangent: expected a direction not parallel to normal")
        x /= np.linalg.norm(x)
        matrix = np.column_stack((x, np.cross(z, x), z))
        # Largest-component conversion remains stable at rotations of pi.
        candidates = np.array(
            [
                1 + np.trace(matrix),
                1 + 2 * matrix[0, 0] - np.trace(matrix),
                1 + 2 * matrix[1, 1] - np.trace(matrix),
                1 + 2 * matrix[2, 2] - np.trace(matrix),
            ]
        )
        i = int(np.argmax(candidates))
        q = np.zeros(4)
        q[i] = math.sqrt(max(0.0, candidates[i])) / 2
        if i == 0:
            q[1:] = [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
            q[1:] /= 4 * q[0]
        else:
            a, b, c = i - 1, i % 3, (i + 1) % 3
            q[0] = (matrix[c, b] - matrix[b, c]) / (4 * q[i])
            q[b + 1] = (matrix[b, a] + matrix[a, b]) / (4 * q[i])
            q[c + 1] = (matrix[c, a] + matrix[a, c]) / (4 * q[i])
        return cls(position, tuple(float(v) for v in q / np.linalg.norm(q)))

    def evaluate(self, binding: Any, free_vec: Any) -> Any:
        """Return a JAX homogeneous rest transform using the supplied parameter scope."""
        import jax.numpy as jnp

        from software_defined_matter.dsl.resolve import resolve_param_value

        p = resolve_param_value(self.position, binding, free_vec)
        q = resolve_param_value(self.orientation, binding, free_vec)
        w, x, y, z = q / jnp.linalg.norm(q)
        rotation = jnp.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return jnp.eye(4, dtype=rotation.dtype).at[:3, :3].set(rotation).at[:3, 3].set(p)


@dataclass(frozen=True)
class Port:
    """Named rigid attachment, with optional body ownership and local interface geometry."""

    name: str
    frame: Frame = field(default_factory=Frame)
    body: str | None = None
    domains: dict[str, Any] = field(default_factory=dict)
    sdf_tree: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.name, "Port.name")
        if not isinstance(self.frame, Frame):
            raise ValueError(f"Port {self.name!r}.frame: expected Frame, got {self.frame!r}")
        if self.body is not None and (not isinstance(self.body, str) or not self.body):
            raise ValueError(f"Port {self.name!r}.body: expected a body name, got {self.body!r}")
        for name in ("domains", "metadata"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"Port {self.name!r}.{name}: expected a mapping")
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        object.__setattr__(self, "sdf_tree", copy.deepcopy(self.sdf_tree))

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON port declaration."""
        return {
            "name": self.name,
            "frame": self.frame.to_dict(),
            "body": self.body,
            "domains": copy.deepcopy(self.domains),
            "sdf_tree": copy.deepcopy(self.sdf_tree),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Port:
        """Read a port, rejecting unknown fields even outside schema validation."""
        extra = set(data) - {"name", "frame", "body", "domains", "sdf_tree", "metadata"}
        if extra:
            raise ValueError(f"Port: unknown fields {sorted(extra)}")
        return cls(
            name=data["name"],
            frame=Frame.from_dict(data["frame"]),
            body=data.get("body"),
            domains=data.get("domains", {}),
            sdf_tree=data.get("sdf_tree"),
            metadata=data.get("metadata", {}),
        )
