"""Internal helpers shared by ``sdf_shapes``, ``sdf_ops``, and ``transforms``.

``_length`` and ``_azimuth`` each carry a hand-written Jacobian Vector Product (JVP)
 definition via ``custom_jvp`` so that gradients through, respectively,
 ``sqrt(sum(x**2))`` and ``atan2(y, x)`` are well-defined on the Z axis.
 This module is the single home for those definitions: duplicating them elsewhere
 would create distinct JVP rules and silently break gradients.
"""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def _length(v):
    return jnp.sqrt(jnp.sum(v**2, axis=-1))


@_length.defjvp
def _length_jvp(primals, tangents):
    (v,), (v_dot,) = primals, tangents
    primal_out = _length(v)
    sq_norm = jnp.sum(v**2, axis=-1)
    inv_norm = jnp.where(sq_norm > 0.0, jax.lax.rsqrt(sq_norm), 0.0)
    tangent_out = jnp.sum(v * v_dot, axis=-1) * inv_norm
    return primal_out, tangent_out


# Floor (an absolute length, in part units) on the radius used when
# differentiating the azimuth ``atan2(y, x)``. The raw derivative
# ``d(theta) = (x*dy - y*dx) / (x**2 + y**2)`` is ``0/0`` (NaN) on the Z axis
# and grows like ``1/r`` approaching it; the ``+ _AXIS_EPSILON**2`` floor keeps
# it finite everywhere and caps its magnitude at ``~1/(2*_AXIS_EPSILON)``.
# Points past ~this radius are essentially unaffected; tune if a design lives at
# sub-``_AXIS_EPSILON`` radii from the axis.
_AXIS_EPSILON = 1e-4


@jax.custom_jvp
def _azimuth(p):
    """Azimuth angle ``atan2(p_y, p_x)`` with an axis-safe gradient.

    The value is exactly ``jnp.arctan2`` (``0`` on the Z axis by convention);
    only the JVP is overridden, floored by ``_AXIS_EPSILON`` so the gradient is
    finite at ``p_xy = 0`` instead of ``NaN``. See ``_length`` for the sibling
    guard on ``|p|``.
    """
    return jnp.arctan2(p[..., 1], p[..., 0])


@_azimuth.defjvp
def _azimuth_jvp(primals, tangents):
    (p,), (p_dot,) = primals, tangents
    primal_out = _azimuth(p)
    x, y = p[..., 0], p[..., 1]
    x_dot, y_dot = p_dot[..., 0], p_dot[..., 1]
    denom = x**2 + y**2 + _AXIS_EPSILON**2
    tangent_out = (x * y_dot - y * x_dot) / denom
    return primal_out, tangent_out


def _dot2(v):
    return jnp.sum(v * v, axis=-1)


def _clamp(x, lo, hi):
    return jnp.clip(x, lo, hi)


def _mix(a, b, t):
    return a + (b - a) * t


def _sign(x):
    return jnp.sign(x)


def bspline_outline(v, samples):
    """Sample a closed periodic uniform cubic B-spline to its outline points.

    ``v`` is an ``(M, 2)`` control polygon (``M >= 4``); returns ``(M*samples, 2)``
    points around the closed curve. Pure linear map of the control points (fixed
    basis), so it is differentiable and commutes with linear/PCHIP interpolation
    of the control points, which is why a loft can interpolate control points and
    sample once, identically to sampling each section then lofting the outlines.
    Shared by :func:`sdf_shapes.bspline_2d` and :func:`sdf_ops.loft_shape_curve`.
    """
    m = v.shape[0]
    t = jnp.linspace(0.0, 1.0, samples, endpoint=False)
    basis = (
        jnp.stack(
            [
                (1.0 - t) ** 3,
                3.0 * t**3 - 6.0 * t**2 + 4.0,
                -3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0,
                t**3,
            ],
            axis=-1,
        )
        / 6.0
    )  # (samples, 4)
    idx = (jnp.arange(m)[:, None] + jnp.arange(4)[None, :]) % m  # (m, 4) cyclic
    pts = jnp.einsum("sj,mjc->msc", basis, v[idx])  # (m, samples, 2)
    return pts.reshape(m * samples, 2)


def bezier_outline(v, samples):
    """Sample a closed composite cubic Bézier to its outline points.

    ``v`` is a ``(3K, 2)`` control array (SVG convention: indices ``0,3,6,…`` are
    on-curve anchors); returns ``(K*samples, 2)`` points. Pure linear map (fixed
    Bernstein basis); see :func:`bspline_outline` for why this is loft-safe.
    """
    m = v.shape[0]
    k = m // 3
    t = jnp.linspace(0.0, 1.0, samples, endpoint=False)
    mt = 1.0 - t
    basis = jnp.stack(
        [mt**3, 3.0 * mt**2 * t, 3.0 * mt * t**2, t**3], axis=-1
    )  # (samples, 4) Bernstein
    seg = jnp.arange(k) * 3
    idx = (seg[:, None] + jnp.arange(4)[None, :]) % m  # (k, 4) cyclic
    pts = jnp.einsum("sj,kjc->ksc", basis, v[idx])  # (k, samples, 2)
    return pts.reshape(k * samples, 2)


def _check_axis_vector(name, v, p):
    """Validate that ``v`` is a 1-D per-axis vector matching ``p``'s spatial dim.

    Used by leaf SDF ops/shapes whose parameter has one value per spatial
    axis (e.g. translation, half-extents, periods). Runs at JAX trace time
    only: all inputs are inspected via ``.shape``/``.ndim``, which are
    static under ``jax.jit``, so this check has zero runtime cost in the
    compiled program.

    Args:
        name: Identifier used in the error message (e.g. ``"tf_translate.t"``).
        v: The per-axis vector to validate.
        p: The query point (or batch of points). The check requires
            ``v.shape == (p.shape[-1],)``.

    Raises:
        ValueError: If ``v`` is not 1-D or its length differs from
            ``p.shape[-1]``.
    """
    if v.ndim != 1 or v.shape != p.shape[-1:]:
        raise ValueError(
            f"{name}: expected 1-D vector of length {p.shape[-1]}, got shape {tuple(v.shape)}"
        )
