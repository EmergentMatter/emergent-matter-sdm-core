"""Coordinate transforms: ``p -> p``.

These functions operate purely on the input space and have no knowledge of
SDFs. The compiler applies them to a query point before calling a child
shape's evaluator.

Includes:

  * Spatial transforms: ``tf_translate``, ``tf_scale``, ``tf_rotate_x|y|z``,
    ``tf_rotate``.
  * Symmetry: ``tf_canonical_sector_fold`` (N-fold rotational fold),
    ``reflect_plane`` (planar reflection; the ``mirror`` transform unions a
    child with its reflection).
  * Domain repetition: ``op_repeat_finite``.

Shape conventions
-----------------
All "per-axis" parameter vectors (``t``, ``c``, ``l``, and analogously
``h`` in :func:`sdf_ops.op_elongate`, ``b`` in :func:`sdf_shapes.box`,
``n`` in :func:`sdf_shapes.plane`, etc.) must be flat ``(D,)`` arrays
where ``D = p.shape[-1]``. To batch over parameters, wrap the *whole*
SDF / objective function in :func:`jax.vmap`; do not pass batched arrays
into individual leaves. Leaf functions enforce this with
:func:`_helpers._check_axis_vector`, which runs at trace time and has
zero runtime cost under ``jit``.
"""

import jax.numpy as jnp

from software_defined_matter.sdf._helpers import _check_axis_vector, _clamp, _length

# ===========================================================================
# Spatial Transforms  (apply before calling an SDF)
# ===========================================================================


def tf_translate(p, t):
    """Translate by t = [tx, ty, tz] (3-D) or t = [tx, ty] (2-D)."""
    p = jnp.asarray(p)
    t = jnp.asarray(t)
    _check_axis_vector("tf_translate.t", t, p)
    return p - t


def tf_scale(p, s):
    """Uniform scale. Remember to divide the result by s."""
    return p / s


def tf_rotate_x(p, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    R = jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return p @ R.T


def tf_rotate_y(p, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    R = jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return p @ R.T


def tf_rotate_z(p, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    R = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return p @ R.T


def tf_rotate(p, R):
    """Arbitrary rotation via a 3x3 matrix R."""
    return p @ jnp.asarray(R).T


def tf_canonical_sector_fold(p, n_sectors, centered=False, phase_frac=0.0):
    """Fold the query azimuth into one wedge of ``2π/n_sectors``, for N-fold symmetry.

    This function only moves points. ``sdf/compile.py`` builds the N-fold solid
    on top of it, by evaluating the child at three points and keeping the
    smallest distance:

    1. ``q``, the folded point: distance to the copy the query sits in.
    2. ``q`` turned one sector forward: distance to the next copy round.
    3. ``q`` turned one sector back: distance to the previous copy.

    The last two are what make the answer safe. Step 1 on its own gives the
    distance to the query's own copy, but a query near a wedge boundary is
    closer to the copy next door, so that distance comes out too large. Too
    large is the dangerous direction, because a sphere tracer steps by the
    distance it is given and lands past the surface.

    That is three child evaluations for any ``n_sectors``, not one per sector.

    Used by:

    * Any rotationally-symmetric SDF with N copies around the Z axis

    The three evaluations see the child's own wedge and its two neighbours, so
    a child wider than its wedge is still reproduced exactly. What is not is a
    child *placed* a whole sector or more away from the canonical wedge: the
    copies a query should be nearest to then fall outside those three and the
    fold over-reports.

    Args:
        p: Query points, ``(..., 3)``.
        n_sectors: Number of copies around the Z axis.
        centered: Fold into ``[-sector/2, sector/2)`` instead, so a child
            authored at angle 0 sits mid-wedge for any ``n_sectors``. Without
            it the child has to be pre-rotated by half a sector, and that
            rotation is wrong the moment the count changes.
        phase_frac: Offset the copies by this fraction of one sector. Also
            independent of ``n_sectors``, so a live count stays placed. Only
            read when ``centered`` is set.

    Returns:
        The folded query points, same shape as ``p``.
    """
    sector_angle = 2.0 * jnp.pi / n_sectors
    theta = jnp.arctan2(p[..., 1], p[..., 0])
    if centered:
        theta_local = (
            jnp.mod(theta - phase_frac * sector_angle + 0.5 * sector_angle, sector_angle)
            - 0.5 * sector_angle
        )
    else:
        theta_local = jnp.mod(theta, sector_angle)
    r = jnp.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
    px = r * jnp.cos(theta_local)
    py = r * jnp.sin(theta_local)
    return jnp.stack([px, py, p[..., 2]], axis=-1)


def reflect_plane(p, n, o):
    """Reflect points across the plane through ``o`` with normal ``n``.

    A distance-preserving reflection (isometry): **every** point is mirrored to
    the opposite side of the plane. ``n`` need not be unit: it is normalised
    here.

    Normalisation uses _length` (exact value, 0-safe JVP) divided by
    a ``where``-guarded denominator.

    A zero ``n`` has no plane to reflect about, so it degenerates to the
    identity (``mirror`` then reduces to the child unioned with itself). Note
    that :func:`sdf.bbox._mirror_bbox` *rejects* a constant zero normal, so
    such a tree fails at bbox inference rather than meshing as an un-mirrored
    child.

    This is the primitive behind the ``mirror`` transform, which the compiler
    turns into ``min(child(p), child(reflect(p)))``, the child unioned with its
    reflection. Because the child is evaluated at both ``p`` and its full
    reflection, ``mirror`` is robust to where the child sits (either side of the
    plane, or crossing it) and is an exact union. That union uses ``smooth_csg``.

    Dimension-agnostic **at evaluation time**: ``n`` and ``o`` are ``(D,)`` with
    ``D = p.shape[-1]``, so it mirrors 2-D profiles as well as 3-D solids, and
    batching over leading point dims follows for free, as with
    :func:`tf_translate`. Analytic bbox inference is 3-D only, however: a 2-D
    mirror cannot currently reach 3-D, because ``2d_to_3d`` requires a bare 2-D
    primitive as its child.
    """
    p = jnp.asarray(p)
    n = jnp.asarray(n)
    o = jnp.asarray(o)
    _check_axis_vector("reflect_plane.n", n, p)
    _check_axis_vector("reflect_plane.o", o, p)
    n_norm = _length(n)
    n = n / jnp.where(n_norm > 0.0, n_norm, 1.0)  # unit normal
    d = jnp.sum((p - o) * n, axis=-1, keepdims=True)
    return p - 2.0 * d * n


# ===========================================================================
# Domain Repetition
# ===========================================================================


def op_repeat_finite(p, c, limits):
    """Finite repetition: period c (scalar), limits = [lx, ly, lz] repetition count."""
    p = jnp.asarray(p)
    limits = jnp.asarray(limits)
    _check_axis_vector("op_repeat_finite.l", limits, p)
    return p - c * _clamp(jnp.round(p / c), -limits, limits)
