"""Operations on SDFs: combinators, modifiers, deformations, and 2-D-to-3-D
lifts.

Two flavours of "op" live here:

  * **Output combinators**: take SDF *values* and return an SDF value
    (CSG booleans, smooth booleans, ``op_round``, ``op_onion``).
  * **Higher-order ops**: wrap an SDF *function* and a query point, returning
    a value (``op_elongate``, ``op_twist``, ``op_bend``, ``op_displace``,
    ``revolution``, ``extrusion``).

Pure shape functions live in :mod:`software_defined_matter.sdf.sdf_shapes`;
coordinate transforms (``p -> p``) live in
:mod:`software_defined_matter.sdf.transforms`.

Per-axis parameter vectors (e.g. ``h`` in :func:`op_elongate`) follow the
shape convention documented in
:mod:`software_defined_matter.sdf.transforms`: flat ``(D,)`` arrays where
``D = p.shape[-1]``. Batch via :func:`jax.vmap` at the call site.
"""

import jax
import jax.numpy as jnp

from software_defined_matter.sdf._helpers import (
    _check_axis_vector,
    _clamp,
    _length,
    _mix,
    bezier_outline,
    bspline_outline,
)

# ===========================================================================
# CSG Boolean Operations
# ===========================================================================


def op_union(d1, d2):
    return jnp.minimum(d1, d2)


def op_subtract(d1, d2):
    """Subtract d2 from d1."""
    return jnp.maximum(d1, -d2)


def op_intersect(d1, d2):
    return jnp.maximum(d1, d2)


def op_smooth_union(d1, d2, k):
    h = _clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return _mix(d2, d1, h) - k * h * (1.0 - h)


def op_smooth_subtract(d1, d2, k):
    h = _clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
    return _mix(d1, -d2, h) + k * h * (1.0 - h)


def op_smooth_intersect(d1, d2, k):
    h = _clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return _mix(d2, d1, h) + k * h * (1.0 - h)


# ===========================================================================
# N-fold reductions (added 2026-05)
# ===========================================================================
#
# When constructing certain bearings with sun / ring conjugate cavities,
# we softmin-union hundreds of sphere SDFs sampled along closed planet
# trajectories. The pairwise smooth-union ops above don't compose
# scalably for that: `op_smooth_union` reduced over N children with
# matched ``k`` produces a deep tree of pairwise blends and JIT compile
# time grows poorly. The three ops below give the compiler a native
# N-fold reduce.
#
# ``op_softmin_many`` is the math primitive (input shape ``(N, ...)``,
# axis 0 = the stacked SDF values to reduce).
#
# ``reduce_softmin_chunked`` is the chunking policy: a single shared
# loop shared by both the DSL compiler (compile.py softmin_chunked
# branch) and the public ``op_softmin_chunked`` closure factory. It
# caps peak memory at ``chunk_size × point_batch`` instead of
# ``N × point_batch`` (e.g. a 576-sample envelope at voxel 0.25mm
# without chunking peaks at ~175 GB intermediate memory).
#
# ``op_softmin_chunked`` is a thin convenience wrapper around the
# chunking policy for non-DSL / test use (lists of callables).
#
# Mathematically equivalent to repeated pairwise softmin with the same
# ``k`` because softmin-of-softmins is associative when ``k`` matches:
# ``softmin([softmin(A;k), softmin(B;k)]; k) = softmin(A∪B; k)``.


def op_softmin_many(distances, k):
    """log-sum-exp softmin along axis 0; numerically stabilised.

    Args:
        distances: Shape ``(N, ...)``. The first axis enumerates the N
            stacked SDF values to softmin across; the result has the
            trailing shape with axis 0 reduced out.
        k: Smoothing temperature (float or ``jnp.ndarray``). As ``k → 0``
            this approaches ``jnp.min(distances, axis=0)``.

    Returns:
        jnp.ndarray of shape ``distances.shape[1:]``.

    Notes:
        The max-subtract trick keeps the ``jnp.exp`` arguments
        well-conditioned. Branch on ``k <= 0`` is JAX-traceable
        (uses ``jnp.where``, not Python ``if``).
    """
    safe_k = jnp.where(k <= 0.0, 1.0, k)
    m = jnp.min(distances, axis=0)
    soft = m - safe_k * jnp.log(jnp.sum(jnp.exp(-(distances - m) / safe_k), axis=0))
    return jnp.where(k <= 0.0, m, soft)


def reduce_softmin_chunked(values_fn, n, k, chunk_size):
    """Single chunking policy shared by the DSL compiler and op_softmin_chunked.

    Args:
        values_fn: ``values_fn(start, end)`` must return a ``jnp.ndarray``
            of shape ``(end - start, ...)`` -- the stacked SDF values for
            the slice ``[start, end)``. The indirection lets callers with
            different child-evaluation signatures (e.g. compile.py's
            ``(p, free_vec) -> d`` closures vs. ``op_softmin_chunked``'s
            ``(p) -> d`` callables) share the same chunking loop.
        n: Total number of input SDF values (length of the conceptual
            ``per_sample_sdfs`` list).
        k: Smoothing temperature (float or ``jnp.ndarray``; same value
            applied across all chunks and the final softmin-of-softmins --
            associativity requires this to be identical at both levels).
        chunk_size: Maximum chunk size; clamped to ``>= 1``.

    Returns:
        jnp.ndarray: Softmin distance over all N inputs. Trailing shape
        matches what ``values_fn`` returns sans its leading axis.

    Raises:
        ValueError: If ``n <= 0``.

    Notes:
        Mathematically equivalent to a single ``op_softmin_many`` over all
        N values stacked together -- softmin is associative under matched
        ``k``: ``softmin([softmin(A;k), softmin(B;k)]; k) = softmin(A∪B; k)``.
    """
    if n <= 0:
        raise ValueError(f"reduce_softmin_chunked: n must be >= 1, got {n}")
    cs = max(1, int(chunk_size))
    per_chunk = []
    for i in range(0, n, cs):
        stacked = values_fn(i, min(i + cs, n))
        per_chunk.append(op_softmin_many(stacked, k))
    return op_softmin_many(jnp.stack(per_chunk, axis=0), k)


def op_softmin_chunked(per_sample_sdfs, k, chunk_size):
    """Build an SDF closure that softmin-unions ``per_sample_sdfs`` in chunks.

    Thin wrapper around :func:`reduce_softmin_chunked` for callers that
    have a flat list of ``(p) -> d`` callables (e.g. tests, non-DSL
    construction). Memory peaks at ``chunk_size × point_batch`` instead
    of ``len(per_sample_sdfs) × point_batch``. Mathematically equivalent
    to ``op_softmin_many`` over the union of all chunks (softmin is
    associative under matched ``k``).

    Args:
        per_sample_sdfs: List of callables, each taking ``p : (..., 3)``
            and returning ``(...,)``.
        k: Smoothing temperature (float).
        chunk_size: Number of input SDFs to evaluate per chunk. A
            reference workload uses 48 for a 576-sample envelope
            construction; smaller values trade more chunked sequential
            work for lower peak memory.

    Returns:
        callable: ``sdf(p)`` returning the softmin distance.
    """
    n = len(per_sample_sdfs)
    if n == 0:
        raise ValueError("op_softmin_chunked: per_sample_sdfs is empty")

    def sdf(p):
        def values_fn(start: int, end: int) -> jnp.ndarray:
            return jnp.stack([s(p) for s in per_sample_sdfs[start:end]], axis=0)

        return reduce_softmin_chunked(values_fn, n, k, chunk_size)

    return sdf


# ===========================================================================
# Modifiers
# ===========================================================================


def op_round(d, r):
    """Round / offset a surface outward by r."""
    return d - r


def op_onion(d, thickness):
    """Shell / onion skin."""
    return jnp.abs(d) - thickness


def op_elongate(sdf_fn, p, h):
    """Elongate a primitive along each axis by h = [hx, hy, hz]."""
    p = jnp.asarray(p)
    h = jnp.asarray(h)
    _check_axis_vector("op_elongate.h", h, p)
    q = p - _clamp(p, -h, h)
    return sdf_fn(q)


# ===========================================================================
# Deformations
# ===========================================================================


def op_twist(sdf_fn, p, k):
    """Twist around Y axis with rate k."""
    c, s = jnp.cos(k * p[..., 1]), jnp.sin(k * p[..., 1])
    q = jnp.stack(
        [c * p[..., 0] - s * p[..., 2], p[..., 1], s * p[..., 0] + c * p[..., 2]], axis=-1
    )
    return sdf_fn(q)


def op_bend(sdf_fn, p, k):
    """Cheap bend in the XY plane (rotation about Z) with rate k."""
    c, s = jnp.cos(k * p[..., 0]), jnp.sin(k * p[..., 0])
    q = jnp.stack(
        [c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]], axis=-1
    )
    return sdf_fn(q)


def _hermite_ramp(t):
    """Smoothstep weight on an already-clamped ``t``.

    Shared by both interpolated twists so the profile is spelled once: the two
    deforms differ in the COORDINATE they ramp over and in their sign
    convention, never in the profile itself.
    """
    return t * t * (3.0 - 2.0 * t)


def _rotate_z(p, a):
    """``R_z(a) . p`` on a batched point array."""
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.stack(
        [c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]], axis=-1
    )


def op_twist_radial(sdf_fn, p, r0, r1, a0, a1):
    """Rotate about Z by an angle interpolated over cylindrical radius.

    ``angle(r) = a0 + (a1 - a0) * smoothstep(r0, r1, r)``: ``a0`` inside ``r0``
    (e.g. a flexure's central pivot), ``a1`` outside ``r1`` (the welded ring).
    The Hermite smoothstep stands in for a Beam Constraint Model deflection
    profile of a clamped-clamped blade, so the shape BENDS between the two end
    rotations instead of rotating rigidly.

    SIGN: the angle is NEGATED before it is applied to the query, so a positive
    ``a0``/``a1`` reads COUNTER-CLOCKWISE viewed from +Z. ``op_twist_linear``
    does not negate; see there for why the two disagree deliberately.

    Like ``op_twist`` / ``op_bend`` this inverse-maps the query point, so the
    result is an approximate (non-Euclidean) distance -- fine for visualisation
    and for optimization metrics over a padded integration box.

    Args:
        sdf_fn: The child field, evaluated at the mapped query point.
        p: Query points, ``(..., 3)``.
        r0: Radius at which the angle is ``a0``.
        r1: Radius at which the angle is ``a1``.
        a0: Angle inside ``r0``, radians.
        a1: Angle outside ``r1``, radians.

    Returns:
        jnp.ndarray: The child's distance at the mapped point.
    """
    r = jnp.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
    w = _hermite_ramp(jnp.clip((r - r0) / (r1 - r0), 0.0, 1.0))
    return sdf_fn(_rotate_z(p, -(a0 + (a1 - a0) * w)))


def op_twist_linear(sdf_fn, p, u_axis, u0, u1, a0, a1):
    """Rotate about Z by an angle interpolated over a signed LINEAR coordinate.

    The sibling of :func:`op_twist_radial` for shapes whose two boundaries sit
    at the same RADIUS on opposite sides. A full-diameter crossing plate is the
    motivating case: its rim welds to ring *s* at one end of a diameter and to
    ring *s+1* at the other, so the deflection ramps along that diameter, not
    along rho. ``twist_radial`` is symmetric in rho and bends both ends the same
    way -- correct for a radial half-blade (hub at 0, ring at r), wrong for a
    plate, and the reason a plate architecture can only swing its blades
    RIGIDLY without this::

        u(p)     = dot(p.xy, normalize(u_axis))
        angle(u) = a0 + (a1 - a0) * smoothstep(u0, u1, u)

    Same Hermite as :func:`op_twist_radial`, and for the same reason: it stands
    in for a Beam Constraint Model deflection profile of a clamped-clamped
    blade (zero slope at both welded ends), so the shape BENDS between the two
    end rotations. Feed the ends from a BCM solve and the profile is the
    model's, not a guess.

    SIGN -- QUERY SPACE, MATCHING ``rotate_z``, AND DELIBERATELY UNLIKE
    :func:`op_twist_radial`. The angle is applied to the query WITHOUT being
    negated, so an end angle ``A`` turns that end of the shape by ``-A`` about
    +Z, exactly as ``transform rotate_z`` does. A twist_linear's two ends are
    almost always driven by the SAME pose params that ``rotate_z`` the bodies
    it is welded between, so matching ``rotate_z`` lets one param feed both.
    Matching ``op_twist_radial`` instead makes the blade counter-rotate against
    its own ring at DOUBLE the intended angle -- observed live on a plate
    welded between two rotating rings, 2026-08-15, and still looking
    plausibly animated, which is what makes it worth a paragraph. A consumer
    cannot paper over the mismatch from outside:
    an angle slot is a plain ``$ref`` with nowhere to hang a negation.

    Args:
        sdf_fn: The child field, evaluated at the mapped query point.
        p: Query points, ``(..., 3)``.
        u_axis: Ramp direction in XY; normalised here, so its scale is free.
        u0: Coordinate at which the angle is ``a0``.
        u1: Coordinate at which the angle is ``a1``.
        a0: Angle at ``u0``, radians.
        a1: Angle at ``u1``, radians.

    Returns:
        jnp.ndarray: The child's distance at the mapped point.
    """
    ax = jnp.asarray(u_axis, dtype=p.dtype)
    n = jnp.sqrt(ax[0] ** 2 + ax[1] ** 2)
    ax = ax / jnp.where(n > 0, n, 1.0)
    u = p[..., 0] * ax[0] + p[..., 1] * ax[1]
    w = _hermite_ramp(jnp.clip((u - u0) / (u1 - u0), 0.0, 1.0))
    return sdf_fn(_rotate_z(p, a0 + (a1 - a0) * w))


def shear_linear_offset(p, u_axis, u0, u1, dz0, dz1):
    """How far the material has risen at each query point under ``shear_linear``.

    ``u(p) = dot(p.xy, normalize(u_axis))``; the rise is ``dz0`` up to ``u0``,
    ``dz1`` from ``u1`` on, and LINEAR between. Linear rather than the twists'
    Hermite on purpose: this is geometry, not a deflection profile. A plate
    welded between two rings climbs from one to the other along a straight
    ramp, and reproducing that ramp exactly is what lets its height ride a live
    param instead of being baked into polygon vertices.
    """
    ax = jnp.asarray(u_axis, dtype=p.dtype)
    n = jnp.sqrt(ax[0] ** 2 + ax[1] ** 2)
    ax = ax / jnp.maximum(n, jnp.asarray(1e-30, dtype=p.dtype))
    u = p[..., 0] * ax[0] + p[..., 1] * ax[1]
    t = jnp.clip((u - u0) / (u1 - u0), 0.0, 1.0)
    return dz0 + (dz1 - dz0) * t


def op_shear_linear(sdf_fn, p, u_axis, u0, u1, dz0, dz1):
    """Shear the child along +Z by a clamped linear ramp over a signed coordinate.

    The child's material at height ``z`` appears at ``z + rise(u)`` (see
    :func:`shear_linear_offset`), so the query is pulled DOWN by the rise
    before the child sees it. With ``dz0 = 0`` and ``dz1 = h`` a flat band
    becomes a clamped ramp whose far end sits ``h`` higher: exactly the
    octagon profile of a plate spanning one stage.

    Inverse-mapping the query makes the result a distance BOUND, not an exact
    distance, inside the ramp; see ``sdf/lipschitz.py`` for the factor.

    Args:
        sdf_fn: The child field, evaluated at the mapped query point.
        p: Query points, ``(..., 3)``.
        u_axis: Ramp direction in XY; normalised here, so its scale is free.
        u0: Coordinate at which the rise is ``dz0``.
        u1: Coordinate at which the rise is ``dz1``.
        dz0: Rise of the material at and before ``u0``.
        dz1: Rise of the material at and after ``u1``.

    Returns:
        jnp.ndarray: The child's distance at the mapped point.
    """
    rise = shear_linear_offset(p, u_axis, u0, u1, dz0, dz1)
    return sdf_fn(p - jnp.stack([jnp.zeros_like(rise), jnp.zeros_like(rise), rise], axis=-1))


def taper_linear_factor(p, z0, z1, s0, s1):
    """The XY scale ``taper_linear`` applies at each query point's height."""
    t = jnp.clip((p[..., 2] - z0) / (z1 - z0), 0.0, 1.0)
    return s0 + (s1 - s0) * t


def op_taper_linear(sdf_fn, p, z0, z1, s0, s1):
    """Scale the child in XY by a factor ramped linearly along Z.

    The factor is ``s0`` up to ``z0``, ``s1`` from ``z1`` on, linear between;
    the query's XY is divided by it, so the MATERIAL grows by it. A cylinder
    tapered from 1 to 0.5 over its length is a cone frustum. The result is a
    distance BOUND (see ``sdf/lipschitz.py``); factors must stay positive.

    Args:
        sdf_fn: The child field, evaluated at the mapped query point.
        p: Query points, ``(..., 3)``.
        z0: Height at which the factor is ``s0``.
        z1: Height at which the factor is ``s1``.
        s0: XY scale at and below ``z0``.
        s1: XY scale at and above ``z1``.

    Returns:
        jnp.ndarray: The child's distance at the mapped point.
    """
    s = taper_linear_factor(p, z0, z1, s0, s1)
    return sdf_fn(jnp.stack([p[..., 0] / s, p[..., 1] / s, p[..., 2]], axis=-1))


def op_displace(sdf_fn, displacement_fn, p):
    """Displace a surface by an arbitrary scalar field."""
    return sdf_fn(p) + displacement_fn(p)


# ===========================================================================
# 2-D -> 3-D Operations
# ===========================================================================


def revolution(sdf2d_fn, p, offset=0.0):
    """
    Revolve a 2-D SDF around the Z axis.
    The 2-D cross-section is placed in the (R_xy, Z) half-plane, where
    R_xy = distance from the Z axis.
    sdf2d_fn : callable(p2d) where p2d has shape (..., 2)
    offset   : radial offset of the profile from the Z axis (creates a torus-like shape)
    """
    q = jnp.stack([_length(p[..., :2]) - offset, p[..., 2]], axis=-1)
    return sdf2d_fn(q)


def extrusion(sdf2d_fn, p, h):
    """
    Extrude a 2-D SDF (in XY) along Z by half-height h.
    """
    d = sdf2d_fn(p[..., :2])
    w = jnp.stack([d, jnp.abs(p[..., 2]) - h], axis=-1)
    return jnp.minimum(jnp.max(w, axis=-1), 0.0) + _length(jnp.maximum(w, 0.0))


# ===========================================================================
# 3-D Sweep: a 2-D profile swept along a 3-D path
# ===========================================================================
# Sample the path into a polyline, build a rotation-minimising frame on each
# segment, and (per query point) project onto the NEAREST segment, evaluate the
# profile SDF in that segment's cross-section plane, and add flat caps at the
# two open ends. Exact for a circular profile (reduces to the path-offset tube,
# matching the analytic torus to faceting tolerance); for a general profile the
# surface is accurate but the per-segment frame gives mild faceting between
# samples (shrinks with sample density) and the field away from the surface is
# approximate: fine for marching-cubes meshing; do not smooth_union it.

_PATH_SAMPLES_PER_SEG = 12


def _sample_path_3d(control_points, kind, closed, samples=_PATH_SAMPLES_PER_SEG):
    """Sample a 3-D path's control points into an ordered polyline ``(K, 3)``.

    kind: ``"polyline"`` (control points used verbatim), ``"bezier"`` (composite
    cubic; open needs ``len = 3K+1``, closed needs ``len = 3K``), or ``"bspline"``
    (periodic uniform cubic, ``len >= 4``; inherently a closed loop).
    """
    v = jnp.asarray(control_points)
    if v.ndim != 2 or v.shape[-1] != 3:
        raise ValueError(f"path control_points must be (M, 3), got shape {v.shape}")
    m = v.shape[0]
    if kind == "polyline":
        if m < 2:
            raise ValueError("polyline path needs >= 2 control points")
        return v
    t = jnp.linspace(0.0, 1.0, samples, endpoint=False)
    if kind == "bezier":
        mt = 1.0 - t
        basis = jnp.stack([mt**3, 3.0 * mt**2 * t, 3.0 * mt * t**2, t**3], axis=-1)
        if closed:
            if m < 6 or m % 3 != 0:
                raise ValueError(f"closed bezier path needs len 3K >= 6, got {m}")
            k = m // 3
            idx = (jnp.arange(k)[:, None] * 3 + jnp.arange(4)[None, :]) % m
        else:
            if m < 4 or m % 3 != 1:
                raise ValueError(f"open bezier path needs len 3K+1 (>= 4), got {m}")
            k = (m - 1) // 3
            idx = jnp.arange(k)[:, None] * 3 + jnp.arange(4)[None, :]
        pts = jnp.einsum("sj,kjc->ksc", basis, v[idx]).reshape(k * samples, 3)
        if not closed:
            pts = jnp.concatenate([pts, v[-1:]], axis=0)  # reach the final anchor
        return pts
    if kind == "bspline":
        if m < 4:
            raise ValueError(f"bspline path needs >= 4 control points, got {m}")
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
        )
        idx = (jnp.arange(m)[:, None] + jnp.arange(4)[None, :]) % m
        return jnp.einsum("sj,mjc->msc", basis, v[idx]).reshape(m * samples, 3)
    raise ValueError(f"unknown path kind {kind!r} (use polyline|bezier|bspline)")


def _rmf_normals(A, that, n0_ref=None):
    """Rotation-minimising unit normals (one per segment) via Wang's discrete
    double-reflection. ``A`` are segment-start points ``(Sg, 3)``; ``that`` are
    unit tangents ``(Sg, 3)``. Avoids the Frenet frame's flips at inflections.

    ``n0_ref`` optionally seeds the initial normal: the reference direction is
    projected perpendicular to the first tangent so the swept profile can START
    in a chosen orientation (e.g. a ribbon leaving a coil radially). It falls
    back to the stable world-axis pick if the reference is ~parallel to ``t0``.
    Default (``None``) reproduces the original world-axis seed exactly."""
    t0 = that[0]
    world_ax = jnp.where(
        jnp.abs(t0[0]) < 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])
    )
    if n0_ref is None:
        ax = world_ax
    else:
        ref = jnp.asarray(n0_ref, dtype=that.dtype)
        proj = ref - jnp.sum(ref * t0) * t0
        ax = jnp.where(jnp.sum(proj * proj) > 1e-12, ref, world_ax)
    n0 = ax - jnp.sum(ax * t0) * t0
    n0 = n0 / jnp.maximum(jnp.sqrt(jnp.sum(n0 * n0)), 1e-9)

    def body(carry, inp):
        x_prev, t_prev, n_prev = carry
        x_i, t_i = inp
        v1 = x_i - x_prev
        c1 = jnp.sum(v1 * v1) + 1e-12
        n_l = n_prev - (2.0 / c1) * jnp.sum(v1 * n_prev) * v1
        t_l = t_prev - (2.0 / c1) * jnp.sum(v1 * t_prev) * v1
        v2 = t_i - t_l
        c2 = jnp.sum(v2 * v2) + 1e-12
        n_i = n_l - (2.0 / c2) * jnp.sum(v2 * n_l) * v2
        n_i = n_i / jnp.maximum(jnp.sqrt(jnp.sum(n_i * n_i)), 1e-9)
        return (x_i, t_i, n_i), n_i

    _, n_rest = jax.lax.scan(body, (A[0], that[0], n0), (A[1:], that[1:]))
    return jnp.concatenate([n0[None, :], n_rest], axis=0)


def _sweep_frames(path_pts, closed=False, frame="rmf", normal0=None):
    """Per-segment start points, unit tangents, lengths, and frame axes of a
    sampled sweep path: ``(A, that, L, N, Bf)``, each ``(Sg, ...)``.

    Split out of :func:`sweep` because the frame depends only on the path,
    never on the query point. The GLSL emitter bakes these arrays into the
    shader by calling THIS function, so the two compilers orient a profile
    identically by construction rather than by transcription.
    """
    C = jnp.asarray(path_pts)
    A = C if closed else C[:-1]
    B = jnp.roll(C, -1, axis=0) if closed else C[1:]
    seg = B - A  # (Sg, 3)
    L = jnp.sqrt(jnp.maximum(jnp.sum(seg * seg, axis=-1), 1e-12))  # (Sg,)
    that = seg / L[:, None]
    if frame == "cylindrical":
        # Lock u to the radial direction (per segment start), projected
        # perpendicular to the tangent; v = that x u follows as ~axial.
        rad = A * jnp.array([1.0, 1.0, 0.0], dtype=A.dtype)  # drop Z -> radial
        rad = rad / jnp.maximum(jnp.linalg.norm(rad, axis=-1, keepdims=True), 1e-9)
        N = rad - jnp.sum(rad * that, axis=-1, keepdims=True) * that
        N = N / jnp.maximum(jnp.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    elif frame == "rmf":
        N = _rmf_normals(A, that, normal0)
    else:
        raise ValueError(f"sweep frame must be 'rmf' or 'cylindrical', got {frame!r}")
    Bf = jnp.cross(that, N)
    return A, that, L, N, Bf


def sweep(profile2d_fn, path_pts, p, closed=False, frame="rmf", normal0=None):
    """Sweep a 2-D profile SDF along a 3-D polyline path.

    profile2d_fn : callable(q2d) -> distance, ``q2d`` shape ``(..., 2)``.
    path_pts     : ``(K, 3)`` ordered path points (see :func:`_sample_path_3d`).
    p            : ``(..., 3)`` query points.
    closed       : if True the path is a loop (wrap last->first, no end caps);
                   if False the two ends get flat caps. Interior vertices are
                   ball joints either way (see the body).
    frame        : how the cross-section is oriented along the path.
                   ``"rmf"`` (default): rotation-minimising frame; the profile
                   twists as little as the path allows. ``"cylindrical"``: the
                   profile's first axis (u) is LOCKED to the radial direction
                   (outward from the Z axis) and the second (v) to ~axial. An RMF
                   accumulates torsion twist over a coiled path (tens of degrees
                   over many turns), which shears a rectangular winding off its
                   turn-to-turn spacing; the cylindrical frame stays radial/axial
                   everywhere a coil needs it (the ribbon tilts only by the local
                   helix pitch). Well-defined wherever the path stays off the axis.
    normal0      : optional reference for the RMF initial normal (``"rmf"`` only;
                   ignored for ``"cylindrical"``). Projected perpendicular to the
                   first tangent so a swept ribbon can start in a chosen
                   orientation (e.g. radial, to continue a winding's cross-section).
    """
    A, that, L, N, Bf = _sweep_frames(path_pts, closed, frame, normal0)

    pe = p[..., None, :]  # (..., 1, 3)
    w = pe - A  # (..., Sg, 3)
    s = jnp.sum(w * that, axis=-1)  # (..., Sg) axial position
    foot = A + jnp.clip(s, 0.0, L)[..., None] * that
    perp = pe - foot  # (..., Sg, 3)

    # Pick the single genuinely-nearest segment (by true 3-D distance). NOT a
    # min over per-segment cross-section values: that would let a far segment
    # whose tangent points at the query report "inside" spuriously.
    istar = jnp.argmin(jnp.sum(perp * perp, axis=-1), axis=-1)  # (...)

    def take(arr):
        return jnp.take_along_axis(arr, istar[..., None], axis=-1)[..., 0]

    u = take(jnp.sum(perp * N, axis=-1))
    v = take(jnp.sum(perp * Bf, axis=-1))

    # Where the query overshoots the chosen segment (foot clamped to one of its
    # ends), `over` is the axial overshoot. Every vertex needs SOME treatment
    # there: with none, a query beyond an interior vertex along that segment's
    # tangent keeps the cross-section value with no axial penalty and reports a
    # spurious "inside" (a phantom tube shooting off every bend).
    #
    # Two treatments, chosen per query by `flat`:
    #   * The two true ends of an OPEN path get a flat cap: `d_ax = over`
    #     combines with the profile like a rounded box, so a straight sweep is
    #     a flat-ended cylinder.
    #   * Every other vertex is a ball joint: the overshoot folds into the
    #     profile-plane radius (`_sweep_joint`) so the segment ends in a
    #     spherical sweep of its own profile. Exact for a circular profile (each
    #     segment is a capsule and their union is seamless), and a sound
    #     approximation for any other. Flat caps there would leave a wedge void
    #     on the outside of every bend, thinner than a voxel but nonzero, which
    #     meshes as a notch at every vertex of a curved path.
    s_star = take(s)  # axial pos on chosen seg
    over_lo = -s_star
    over_hi = s_star - L[istar]
    over = jnp.maximum(over_lo, over_hi)  # >0 iff foot clamped
    if closed:
        flat = jnp.zeros(over.shape, dtype=bool)
    else:
        n_seg = A.shape[0]
        flat = ((istar == 0) & (over_lo > 0.0)) | ((istar == n_seg - 1) & (over_hi > 0.0))
    d2d = profile2d_fn(_sweep_joint(jnp.stack([u, v], axis=-1), over, flat))
    d_ax = jnp.where(flat & (over > 0.0), over, jnp.full_like(over, -1.0e9))
    return jnp.sqrt(jnp.maximum(d2d, 0.0) ** 2 + jnp.maximum(d_ax, 0.0) ** 2) + jnp.minimum(
        jnp.maximum(d2d, d_ax), 0.0
    )


def _sweep_joint(uv, over, flat):
    """Profile-plane query for a ball joint at a segment end.

    Past an interior vertex the true distance to the segment is the hypotenuse
    of the in-plane radius and the axial overshoot. Scaling ``uv`` out to that
    length, along its own direction, hands the profile a point whose distance
    from the axis is that hypotenuse, so the segment ends in a sphere-swept
    copy of its profile rather than a flat cap. ``flat`` (a true open end) and
    ``over <= 0`` (foot interior) leave ``uv`` untouched. A query exactly on the
    tangent line (``uv == 0``) has no direction; it goes out along +u, which is
    exact for a circle and the only defined choice otherwise. Mirrored by
    ``sdm_sweep_joint`` in ``glsl/lib.glsl``.
    """
    n = jnp.sqrt(jnp.maximum(jnp.sum(uv * uv, axis=-1), 1e-18))
    m = jnp.sqrt(n * n + jnp.maximum(over, 0.0) ** 2)
    along = uv * (m / n)[..., None]
    on_axis = jnp.stack([m, jnp.zeros_like(m)], axis=-1)
    rounded = jnp.where((n > 1e-9)[..., None], along, on_axis)
    return jnp.where(((over > 0.0) & ~flat)[..., None], rounded, uv)


def loft(sdf2d_fns, zs, p, smooth=False):
    """
    Loft N 2-D cross-section SDFs along Z. ``zs`` is the ascending axial
    station of each section (``len(zs) == len(sdf2d_fns) >= 2``). The 2-D
    distance field is interpolated between sections at each query point's Z, then
    capped to the span ``[zs[0], zs[-1]]``, a generalisation of :func:`extrusion`
    to a Z-varying cross-section.

    This is the natural representation for a swept/tapered/twisted profile (e.g.
    an airfoil whose chord and position vary along the span, or a transition that
    morphs one outline into another) without stacking discrete prism slabs. It
    interpolates the *2-D fields*, so it works for any 2-D children and needs no
    vertex correspondence; exact where the sections coincide.

    smooth : if False (default), piecewise-LINEAR interpolation (C0): fast, but
        each section span is a flat ruled panel with a crease at every section
        joint. If True, C1 cubic-Hermite interpolation with MONOTONE (PCHIP /
        Fritsch-Carlson) tangents: smooth through the sections, no joint creases,
        and crucially NO overshoot (a Catmull-Rom loft rings past the section
        values and the bulge self-intersects into a non-manifold mesh; PCHIP
        cannot overshoot, so the lofted solid stays watertight). Handles
        non-uniform stations; one-sided tangents at the ends.

    Not a metrically-exact loft of the in-between surface: fine for
    marching-cubes meshing; do not ``smooth_union`` the output with other
    non-metric fields.

    sdf2d_fns : list of callable(p2d) -> distance, ``p2d`` shape ``(..., 2)``
    zs        : 1-D array of section Z stations, STRICTLY ASCENDING. Order is
        assumed, not enforced here (searchsorted relies on it); a mis-ordered
        span yields empty/degenerate geometry. The compiler validates literal
        stations; param-driven ($ref) stations are the caller's responsibility.
    """
    z = p[..., 2]
    zs = jnp.asarray(zs)
    n = zs.shape[0]
    dvals = jnp.stack([f(p[..., :2]) for f in sdf2d_fns], axis=-1)  # (..., n)
    idx = jnp.clip(jnp.searchsorted(zs, z, side="right") - 1, 0, n - 2)  # segment

    def gather(ii):
        return jnp.take_along_axis(dvals, ii[..., None], axis=-1)[..., 0]

    z0, z1 = zs[idx], zs[idx + 1]
    t = jnp.clip((z - z0) / (z1 - z0 + 1e-12), 0.0, 1.0)
    d0, d1 = gather(idx), gather(idx + 1)
    if not smooth:
        d2d = d0 * (1.0 - t) + d1 * t
    else:
        # C1 cubic Hermite with MONOTONE (PCHIP / Fritsch-Carlson) tangents: no
        # overshoot, so the lofted solid stays watertight.
        im1 = jnp.clip(idx - 1, 0, n - 1)
        i2 = jnp.clip(idx + 2, 0, n - 1)
        dm1, d2 = gather(im1), gather(i2)
        hseg = z1 - z0
        m0 = _pchip_tan_arr(dm1, d0, d1, zs[idx] - zs[im1], hseg)  # tangent at left node
        m1 = _pchip_tan_arr(d0, d1, d2, hseg, zs[i2] - zs[idx + 1])  # tangent at right node
        t2, t3 = t * t, t * t * t
        d2d = (
            (2 * t3 - 3 * t2 + 1) * d0
            + (t3 - 2 * t2 + t) * hseg * m0
            + (-2 * t3 + 3 * t2) * d1
            + (t3 - t2) * hseg * m1
        )
    zc = (zs[0] + zs[-1]) * 0.5
    # abs() guards a mis-ordered (e.g. param-driven) span from going negative,
    # which would silently make the cap empty everywhere; see loft() docstring.
    h = jnp.abs(zs[-1] - zs[0]) * 0.5
    w = jnp.stack([d2d, jnp.abs(z - zc) - h], axis=-1)
    return jnp.minimum(jnp.max(w, axis=-1), 0.0) + _length(jnp.maximum(w, 0.0))


def _pchip_tan_arr(dl, dm, dr, hl, hr):
    """Monotone (PCHIP / Fritsch-Carlson) Hermite tangent at the middle node,
    vectorised: ``dl/dm/dr`` are arbitrary-shaped value arrays at three stations,
    ``hl/hr`` the (broadcastable) left/right station spacings. No overshoot
    (extremum -> flat); one-sided at the ends (hl or hr == 0)."""
    sl = (dm - dl) / jnp.where(hl != 0.0, hl, 1.0)
    sr = (dr - dm) / jnp.where(hr != 0.0, hr, 1.0)
    w1, w2 = 2.0 * hr + hl, hr + 2.0 * hl
    denom = w1 / jnp.where(sl != 0.0, sl, 1.0) + w2 / jnp.where(sr != 0.0, sr, 1.0)
    m_int = (w1 + w2) / jnp.where(denom != 0.0, denom, 1.0)
    m = jnp.where(sl * sr > 0.0, m_int, 0.0)
    m = jnp.where(hl == 0.0, sr, m)
    return jnp.where(hr == 0.0, sl, m)


def _poly_sdf_batched(p2d, v):
    """Exact 2-D signed distance to a polygon whose vertices VARY per query
    point. ``p2d`` is ``(..., 2)``; ``v`` is ``(..., N, 2)``: one polygon per
    query point. Same Inigo-Quilez edge-distance + winding-parity sign as
    :func:`sdf_shapes.polygon_2d`, but with the vertices carrying the leading
    batch dims (so a lofted, z-interpolated outline can be evaluated in one shot)."""
    v_prev = jnp.roll(v, 1, axis=-2)  # (..., N, 2)
    e = v_prev - v  # edge vectors v_prev - v
    ee = jnp.sum(e * e, axis=-1)  # (..., N)
    ee = jnp.where(ee > 1e-12, ee, 1.0)  # guard zero-length edges
    p_exp = p2d[..., jnp.newaxis, :]  # (..., 1, 2)
    w = p_exp - v  # (..., N, 2)
    tt = _clamp(jnp.sum(w * e, axis=-1) / ee, 0.0, 1.0)
    b = w - tt[..., jnp.newaxis] * e
    d = jnp.min(jnp.sum(b * b, axis=-1), axis=-1)  # (...)
    v_y, v_prev_y = v[..., 1], v_prev[..., 1]
    p_y = p2d[..., 1:2]  # (..., 1)
    c1 = p_y >= v_y
    c2 = p_y < v_prev_y
    c3 = (e[..., 0] * w[..., 1]) > (e[..., 1] * w[..., 0])
    flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
    inside = (jnp.sum(flip.astype(jnp.int32), axis=-1) % 2) == 1
    return jnp.where(inside, -1.0, 1.0) * jnp.sqrt(d)


def loft_shape(verts_sections, zs, p, smooth=False):
    """Loft N polygon cross-sections along Z by interpolating the VERTICES
    (shape interpolation), not the distance fields.

    Unlike :func:`loft` (which blends the per-section 2-D *fields*), this
    interpolates the polygon outline itself between sections and evaluates the
    exact 2-D polygon SDF of the interpolated outline. Blending fields makes the
    zero-set BULGE outward at sharp convex features (a swept leading edge
    scallops between stations); interpolating the shape places the edge exactly
    on the interpolated outline, so there is no bulge.

    Requires VERTEX CORRESPONDENCE: every section must have the SAME vertex count
    ``N`` and vertex ``k`` must denote the same feature across sections (e.g. all
    sections resampled to ``N`` points from the same start). Mismatched
    correspondence shears the interpolated outline. For arbitrary 2-D children
    with no correspondence, use :func:`loft` (field interpolation) instead.

    smooth : False -> piecewise-LINEAR vertex interp (C0, straight edges between
        stations); True -> C1 monotone-PCHIP vertex interp (smooth, no overshoot).

    verts_sections : list of ``(N, 2)`` arrays (same ``N``), one per section.
    zs             : 1-D array of section Z stations, STRICTLY ASCENDING (order
        assumed, not enforced; a mis-ordered span yields empty/degenerate
        geometry; see :func:`loft`).
    """
    z = p[..., 2]
    zs = jnp.asarray(zs)
    n = zs.shape[0]
    V = jnp.stack([jnp.asarray(v) for v in verts_sections], axis=0)  # (n, N, 2)
    idx = jnp.clip(jnp.searchsorted(zs, z, side="right") - 1, 0, n - 2)
    z0, z1 = zs[idx], zs[idx + 1]
    t = jnp.clip((z - z0) / (z1 - z0 + 1e-12), 0.0, 1.0)
    V0, V1 = V[idx], V[idx + 1]  # (..., N, 2)
    if not smooth:
        tt = t[..., jnp.newaxis, jnp.newaxis]
        Vq = V0 * (1.0 - tt) + V1 * tt
    else:
        im1 = jnp.clip(idx - 1, 0, n - 1)
        i2 = jnp.clip(idx + 2, 0, n - 1)
        Vm1, V2 = V[im1], V[i2]
        hseg = (z1 - z0)[..., jnp.newaxis, jnp.newaxis]
        hl0 = (zs[idx] - zs[im1])[..., jnp.newaxis, jnp.newaxis]
        hr1 = (zs[i2] - zs[idx + 1])[..., jnp.newaxis, jnp.newaxis]
        m0 = _pchip_tan_arr(Vm1, V0, V1, hl0, hseg)  # tangent at left end
        m1 = _pchip_tan_arr(V0, V1, V2, hseg, hr1)  # tangent at right end
        tt = t[..., jnp.newaxis, jnp.newaxis]
        t2, t3 = tt * tt, tt * tt * tt
        Vq = (
            (2 * t3 - 3 * t2 + 1) * V0
            + (t3 - 2 * t2 + tt) * hseg * m0
            + (-2 * t3 + 3 * t2) * V1
            + (t3 - t2) * hseg * m1
        )
    d2d = _poly_sdf_batched(p[..., :2], Vq)
    zc = (zs[0] + zs[-1]) * 0.5
    # abs() guards a mis-ordered (e.g. param-driven) span from going negative,
    # which would silently make the cap empty everywhere; see loft() docstring.
    h = jnp.abs(zs[-1] - zs[0]) * 0.5
    w = jnp.stack([d2d, jnp.abs(z - zc) - h], axis=-1)
    return jnp.minimum(jnp.max(w, axis=-1), 0.0) + _length(jnp.maximum(w, 0.0))


def loft_shape_curve(ctrl_sections, zs, p, kind, samples, smooth=False):
    """Loft N smooth-curve cross-sections (bspline_2d / bezier_2d) along Z by
    SHAPE interpolation of their CONTROL POINTS: the compact, scallop-free way
    to loft curve profiles.

    Each section is given by its control points (same count + correspondence
    across sections). This samples each section's control points to outline points
    once (via :func:`_helpers.bspline_outline` / :func:`bezier_outline`) and hands
    them to :func:`loft_shape`, so it inherits loft_shape's no-bulge, watertight,
    monotone-PCHIP guarantees on the outline, while the DSL stores only the handful
    of control points.

    Note the order: this SAMPLES then INTERPOLATES the outline vertices. Because
    sampling a B-spline / Bézier is a FIXED LINEAR map of the control points, for
    ``smooth=False`` (linear interp) this is exactly equivalent to interpolating the
    control points then sampling. For ``smooth=True`` the PCHIP interpolation is
    nonlinear, so the two orders differ: this deliberately interpolates the sampled
    outlines (which is what makes loft_shape's watertight/no-overshoot guarantee
    apply to the actual outline vertices).

    ctrl_sections : list of control-point arrays (same shape), one per section.
    kind          : "bspline_2d" or "bezier_2d".
    samples       : outline samples per curve span/segment (e.g. 16).
    """
    if kind == "bspline_2d":
        outline = bspline_outline
    elif kind == "bezier_2d":
        outline = bezier_outline
    else:
        raise ValueError(f"loft_shape_curve: kind must be bspline_2d/bezier_2d, got {kind!r}")
    verts_sections = [outline(jnp.asarray(c), samples) for c in ctrl_sections]
    return loft_shape(verts_sections, zs, p, smooth=smooth)
