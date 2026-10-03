"""Interpolate endpoint motions without treating rotation matrices as linear data.

Compatible authored operation chains retain their joint coordinates and winding.
Other endpoint frames follow a constant relative screw using the principal
rotation. Blend coordinates are evaluated in rest space by the caller.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any


def _skew(vector: Any) -> Any:
    import jax.numpy as jnp

    x, y, z = vector
    return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=vector.dtype)


def _rotation_vector(rotation: Any) -> Any:
    """Principal logarithm, stable at identity and at a half turn.

    Select the largest quaternion component before division. All candidates have
    finite derivatives so batching the selection cannot leak inactive NaNs.
    """
    import jax.numpy as jnp

    r = rotation
    squares = jnp.array(
        [
            1 + r[0, 0] + r[1, 1] + r[2, 2],
            1 + r[0, 0] - r[1, 1] - r[2, 2],
            1 - r[0, 0] + r[1, 1] - r[2, 2],
            1 - r[0, 0] - r[1, 1] + r[2, 2],
        ]
    )
    roots = jnp.sqrt(jnp.maximum(squares, 1e-12))
    candidates = jnp.array(
        [
            [squares[0], r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
            [r[2, 1] - r[1, 2], squares[1], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0]],
            [r[0, 2] - r[2, 0], r[0, 1] + r[1, 0], squares[2], r[1, 2] + r[2, 1]],
            [r[1, 0] - r[0, 1], r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], squares[3]],
        ]
    ) / (2 * roots[:, None])
    q = candidates[jnp.argmax(squares)]
    q = q / jnp.linalg.norm(q)
    q = jnp.where(q[0] < 0, -q, q)
    norm2 = jnp.sum(q[1:] ** 2)
    norm = jnp.sqrt(jnp.maximum(norm2, 1e-12))
    scale = jnp.where(norm2 < 1e-6, 2 + norm2 / 3, 2 * jnp.arctan2(norm, q[0]) / norm)
    return scale * q[1:]


def _screw_matrix(first: Any, second: Any, weight: Any) -> Any:
    """T_from exp(weight log(inv(T_from) T_to)), using small-angle series."""
    import jax.numpy as jnp

    rotation = first[:3, :3].T @ second[:3, :3]
    translation = first[:3, :3].T @ (second[:3, 3] - first[:3, 3])
    omega = _rotation_vector(rotation)
    theta2 = jnp.sum(omega * omega)
    theta = jnp.sqrt(jnp.maximum(theta2, 1e-12))
    half = theta / 2
    coefficient = jnp.where(
        theta2 < 1e-4,
        1 / 12 + theta2 / 720 + theta2**2 / 30240,
        (1 - half * jnp.cos(half) / jnp.sin(half)) / theta**2,
    )
    skew = _skew(omega)
    velocity = (jnp.eye(3) - skew / 2 + coefficient * (skew @ skew)) @ translation
    step = weight * omega
    step2 = jnp.sum(step * step)
    angle = jnp.sqrt(jnp.maximum(step2, 1e-12))
    a = jnp.where(step2 < 1e-4, 1 - step2 / 6 + step2**2 / 120, jnp.sin(angle) / angle)
    b = jnp.where(step2 < 1e-4, 0.5 - step2 / 24 + step2**2 / 720, (1 - jnp.cos(angle)) / angle**2)
    c = jnp.where(
        step2 < 1e-4, 1 / 6 - step2 / 120 + step2**2 / 5040, (angle - jnp.sin(angle)) / angle**3
    )
    k = _skew(step)
    k2 = k @ k
    relative = jnp.eye(4, dtype=first.dtype)
    relative = relative.at[:3, :3].set(jnp.eye(3) + a * k + b * k2)
    relative = relative.at[:3, 3].set((jnp.eye(3) + b * k + c * k2) @ (weight * velocity))
    return first @ relative


@dataclass(frozen=True)
class FlexureMotion:
    from_index: int
    to_index: int
    blend: Callable
    joints: tuple[Any, ...] | None

    def matrices(self, points: Any, dofs: Any, bodies: Any, free_vec: Any) -> Any:
        import jax
        import jax.numpy as jnp

        raw = self.blend(points, free_vec)
        weights = jnp.where(jnp.isfinite(raw), jnp.clip(raw, 0, 1), jnp.nan)
        first, second = bodies[self.from_index], bodies[self.to_index]

        def at_weight(weight: Any) -> Any:
            if self.joints is None:
                result = _screw_matrix(first, second, weight)
            else:
                result = jnp.eye(4, dtype=dofs.dtype)
                for template, start, end in self.joints:
                    value = (1 - weight) * start(dofs, free_vec) + weight * end(dofs, free_vec)
                    result = (
                        replace(template, value=lambda _, __, value=value: value).matrix(
                            dofs, free_vec
                        )
                        @ result
                    )
            # Exact endpoints avoid roundoff in the logarithm/exponential pair.
            result = jnp.where(weight == 0, first, result)
            return jnp.where(weight == 1, second, result)

        return jax.vmap(at_weight)(weights.reshape(-1)).reshape((*points.shape[:-1], 4, 4))


def compatible_joints(first: tuple[Any, ...], second: tuple[Any, ...]) -> tuple | None:
    """Match fixed axes and origins, filling a ground body's chain with zeros."""
    import numpy as np

    def zero(_: Any, free_vec: Any) -> float:
        return 0.0

    if not first:
        return tuple((op, zero, op.value) for op in second)
    if not second:
        return tuple((op, op.value, zero) for op in first)
    if len(first) != len(second):
        return None
    result = []
    for a, b in zip(first, second, strict=True):
        sign = float(np.dot(a.axis, b.axis))
        if a.kind != b.kind or not math.isclose(abs(sign), 1, abs_tol=1e-12):
            return None
        if a.kind == "rotate" and not np.allclose(
            np.cross(np.asarray(b.origin) - a.origin, a.axis), 0, atol=1e-12, rtol=0
        ):
            return None
        result.append((a, a.value, lambda q, p, b=b, sign=sign: sign * b.value(q, p)))
    return tuple(result)


def invariant_blend(field: dict[str, Any], joints: tuple | None) -> bool:
    """Recognize a single coaxial rotation whose blend survives the motion."""
    import numpy as np

    if field.get("type") != "field" or field.get("kind") not in {"axis_ramp", "radial_hermite"}:
        return False
    if joints is None:
        return False
    if not joints:
        return True  # Both endpoints are ground.
    if len(joints) != 1 or joints[0][0].kind != "rotate":
        return False
    op = joints[0][0]
    params = field["params"]
    axis = np.asarray(params["axis"], dtype=float)
    axis = axis / math.hypot(*axis)
    if not np.allclose(np.cross(axis, op.axis), 0, atol=1e-12, rtol=0):
        return False
    return field["kind"] != "radial_hermite" or bool(
        np.allclose(
            np.cross(np.asarray(params["origin"]) - op.origin, op.axis), 0, atol=1e-12, rtol=0
        )
    )
