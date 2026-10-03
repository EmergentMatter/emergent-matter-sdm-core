"""Pure shape and field functions: ``(p, *params) -> scalar``.

Every function here is a leaf evaluator: it consumes a query point ``p`` and
returns a value (a signed distance for SDFs, a displacement amplitude for
fields). It does NOT compose other SDFs, transform input space, or combine
multiple SDF outputs: those live in :mod:`software_defined_matter.sdf.sdf_ops`
and :mod:`software_defined_matter.sdf.transforms`.

Sources:
  - Inigo Quilez: https://iquilezles.org/articles/distfunctions/
  - Custom AM / compliant-mechanism primitives

Per-axis parameter vectors (``b`` in :func:`box`, ``r`` in :func:`ellipsoid`,
``n`` in :func:`plane`, endpoints ``a``/``b`` in :func:`capsule`, …) follow
the shape convention documented in
:mod:`software_defined_matter.sdf.transforms`: flat ``(D,)`` arrays where
``D = p.shape[-1]``. Batch via :func:`jax.vmap` at the call site.
"""

import math

import jax.numpy as jnp

from software_defined_matter.sdf._helpers import (
    _azimuth,
    _check_axis_vector,
    _clamp,
    _dot2,
    _length,
    _sign,
)
from software_defined_matter.sdf.sdf_ops import op_subtract

# ===========================================================================
# 3-D Exact Primitives  (Inigo Quilez)
# ===========================================================================


def sphere(p, r):
    """Exact SDF for a sphere of radius r centred at origin."""
    return _length(p) - r


def box(p, b):
    """Exact SDF for an axis-aligned box with half-extents b (array-like [bx,by,bz])."""
    p = jnp.asarray(p)
    b = jnp.asarray(b)
    _check_axis_vector("box.b", b, p)
    q = jnp.abs(p) - b
    return _length(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q, axis=-1), 0.0)


def round_box(p, b, r):
    """Exact SDF for a rounded box."""
    p = jnp.asarray(p)
    b = jnp.asarray(b)
    _check_axis_vector("round_box.b", b, p)
    q = jnp.abs(p) - b + r
    return _length(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q, axis=-1), 0.0) - r


def box_frame(p, b, e):
    """Exact SDF for a box frame (wireframe box) with edge thickness e."""
    p = jnp.asarray(p)
    b = jnp.asarray(b)
    _check_axis_vector("box_frame.b", b, p)
    p2 = jnp.abs(p) - b
    q = jnp.abs(p2 + e) - e
    px, py, pz = p2[..., 0], p2[..., 1], p2[..., 2]
    qx, qy, qz = q[..., 0], q[..., 1], q[..., 2]
    d1 = _length(jnp.maximum(jnp.stack([px, qy, qz], axis=-1), 0.0)) + jnp.minimum(
        jnp.maximum(px, jnp.maximum(qy, qz)), 0.0
    )
    d2 = _length(jnp.maximum(jnp.stack([qx, py, qz], axis=-1), 0.0)) + jnp.minimum(
        jnp.maximum(qx, jnp.maximum(py, qz)), 0.0
    )
    d3 = _length(jnp.maximum(jnp.stack([qx, qy, pz], axis=-1), 0.0)) + jnp.minimum(
        jnp.maximum(qx, jnp.maximum(qy, pz)), 0.0
    )
    return jnp.minimum(jnp.minimum(d1, d2), d3)


def torus(p, t):
    """Exact SDF for a torus. t = [major_radius, minor_radius]."""
    t = jnp.asarray(t)
    q = jnp.stack([_length(p[..., :2]) - t[0], p[..., 2]], axis=-1)
    return _length(q) - t[1]


def capped_torus(p, sc, ra, rb):
    """Exact SDF for a capped torus. sc = [sin(angle), cos(angle)]."""
    sc = jnp.asarray(sc)
    px = jnp.abs(p[..., 0])
    py = p[..., 1]
    pz = p[..., 2]
    pxy = jnp.stack([px, py], axis=-1)
    k = jnp.where(sc[1] * px > sc[0] * py, jnp.sum(pxy * sc, axis=-1), _length(pxy))
    p3 = jnp.stack([px, py, pz], axis=-1)
    return jnp.sqrt(_dot2(p3) + ra**2 - 2.0 * ra * k) - rb


def helix(p, major_r, pitch, r, n_turns, phase=0.0, handedness=1.0):
    """Circular tube of radius ``r`` wound helically about +Z, centred on ``z=0``.

    major_r    : radius of the helical centreline from the Z axis
    pitch      : axial rise per full turn (must be non-zero; a zero-pitch
                 helix is a :func:`torus`)
    r          : tube cross-section radius
    n_turns    : number of turns. The band spans ``|z| <= n_turns*|pitch|/2``
                 and is cut FLAT at both ends (a z-slab intersection), which is
                 what a thread flank wants.
    phase      : azimuth of the centreline at ``z = 0``, radians. This is the
                 clocking knob a threaded mate registers against.
    handedness : ``+1`` right-handed (azimuth advances with +z), ``-1`` left.

    Bound, not exact. The tube term measures to the *tangent line* of the
    nearest turn, rescaling the axial offset by ``cos(lead angle)`` so the
    constant-azimuth section is the correct ellipse. That construction is
    written for a query sitting AT the winding radius. Inside it the azimuthal
    lever arm is longer than assumed and the raw value climbs faster than 1 per
    mm travelled, so the tube term is divided by a closed-form constant that
    bounds the climb over the whole wall shell. See ``k_wall`` below.

    To first order the raw over-report grows as ``(r/major_r) * sin(lambda)**2``.

    Three things to know about the far field:

    - Distance is OVER-reported near the Z axis, by up to 2.7x at 44 deg of
      lead. The construction only ever considers centreline points at the
      query's own azimuth; near the axis the nearest one is at a different
      azimuth entirely, and the tangent line runs away from the coil. On the
      axis the value even depends on the approach azimuth, so the field is
      genuinely discontinuous there and no finite Lipschitz constant exists on
      a domain containing it. Recorded in ``KNOWN_BOUNDS`` in
      ``tests/test_sdf_is_distance.py``; harmless for meshing and metrics
      (those cells are wholly empty either way) but it means a sphere-tracer
      can overstep inside the bore.

    - Distance is UNDER-reported near the flat end cuts. The band is a ``max``
      (CSG intersection) against a z-slab, and ``max`` is the usual conservative
      intersection bound: a point can sit near a turn that the cut removed. The
      The actual geometry (distance=0) is unaffected.
    - There is a C0 crease midway between turns, where the nearest-turn choice
      switches. Ordinary for a repeated domain; gradient magnitude stays <= 1
      either side, so ray-marching is safe.

    Reach for :func:`software_defined_matter.sdf.sdf_ops.sweep` instead when the
    cross-section is not circular.
    """
    pz = p[..., 2]
    # Height at which the centreline crosses this query point's azimuth, then
    # the nearest such turn. Shifting z0 by a whole pitch leaves the candidate
    # set {z0 + k*pitch} unchanged, so z_h is continuous
    # across the atan2 branch cut at azimuth = +-pi.
    z0 = handedness * pitch * (_azimuth(p) - phase) / (2.0 * jnp.pi)
    z_h = z0 + pitch * jnp.round((pz - z0) / pitch)

    d_radial = _length(p[..., :2]) - major_r
    # cos(lead angle). The tube's constant-azimuth section is an ellipse
    # stretched along z by 1/cos(lambda); scaling the axial offset back by
    # cos(lambda) recovers the exact perpendicular offset from the tangent line.
    lead = 2.0 * jnp.pi * major_r
    cos_lambda = lead / jnp.sqrt(lead**2 + pitch**2)
    d_tube = _length(jnp.stack([d_radial, (pz - z_h) * cos_lambda], axis=-1)) - r

    # Worst rate of climb over the wall shell. Differentiating the tube term in
    # cylindrical coordinates (z_h depends on the azimuth, so the theta
    # derivative carries a 1/rho) gives
    #
    #     |grad|^2 = [u^2 + cos^4(lambda) * w^2 * (1 + h^2/rho^2)]
    #                / [u^2 + cos^2(lambda) * w^2]
    #
    # with u = rho - major_r, w = pz - z_h and h = pitch/(2*pi). Substituting
    # cos^2(lambda) = major_r^2/(major_r^2 + h^2) makes that exactly 1 on
    # rho = major_r, below 1 outside and above 1 inside. It is a weighted mean
    # of 1 and g(rho) = cos^2(lambda) * (1 + h^2/rho^2), which decreases in
    # rho, so over the wall shell rho in [major_r - r, major_r + r] it is
    # bounded by g(major_r - r).
    h_axial = pitch / (2.0 * jnp.pi)
    # Floors a degenerate r >= major_r (a tube fatter than its own winding
    # radius passes through the axis and self-intersects) rather than dividing
    # by zero. k_wall >= 1 for every well-formed helix; the max is defensive.
    inner_r = jnp.maximum(major_r - r, 1e-6)
    k_wall = jnp.maximum(cos_lambda * jnp.sqrt(1.0 + (h_axial / inner_r) ** 2), 1.0)

    # Only the tube term is scaled. The z-slab is an exact plane distance
    half_h = 0.5 * n_turns * jnp.abs(pitch)
    return jnp.maximum(d_tube / k_wall, jnp.abs(pz) - half_h)


def screw_thread(
    p, r_root, depth, pitch, width, n_turns, phase=0.0, handedness=1.0, flank_deg=60.0
):
    """Truncated-V helical thread ridge about +Z — a REGULAR screw thread.

    r_root     : root (core) radius the tooth stands on
    depth      : radial height of the tooth, root to (truncated) crest
    pitch      : axial advance per turn
    width      : tooth base width, axially, at the root. Keep it below ``pitch``
                 or adjacent teeth merge into a plain cylinder.
    n_turns    : band spans ``|z| <= n_turns*|pitch|/2``, cut flat
    phase      : azimuth of the tooth at z=0 — the clocking knob
    handedness : ``+1`` right-handed, ``-1`` left
    flank_deg  : INCLUDED flank angle. 60 is the ISO metric form, 55 Whitworth,
                 29 ACME.

    Union it onto a ``capped_cylinder`` of ``r_root`` for an external thread;
    subtract a dilated copy for the mating internal one.

    Why this exists rather than twisting a box into a helix: that trick leaves the
    tooth's thickness in the TANGENTIAL direction, so its real AXIAL thickness
    comes out as ``width / (k * r)`` — 0.02 mm on a 58 mm thread. The result is a
    razor-thin spiral ramp, not a thread, and no amount of reshaping the profile
    fixes it. Here the profile lives in the (radial, axial) plane, which is where
    a thread's profile actually lives.

    Construction: fold the query point into the tooth's own frame — ``s`` measured
    radially from the root, ``u`` axially from the nearest tooth at this azimuth —
    then take the truncated triangle as a max of half-planes (exact inside the
    profile, an under-estimate outside its corners, which is the safe direction
    for a sphere tracer). ``u`` is rescaled by cos(lead angle) as in :func:`helix`. The
    tooth-to-tooth fold leaves a C0 crease midway between teeth, ordinary for a
    repeated domain; ``|grad|`` stays <= 1, so ray-marching is safe.
    """
    pz = p[..., 2]
    r = _length(p[..., :2])

    # Axial offset from the nearest tooth crossing this azimuth. Shifting by whole
    # pitches leaves the candidate set unchanged, so this stays continuous across
    # the atan2 branch cut (same argument as `helix`).
    u = pz - handedness * pitch * (_azimuth(p) - phase) / (2.0 * jnp.pi)
    u = u - pitch * jnp.round(u / pitch)

    # cos(lead angle) at this radius: turns the axial offset into a true
    # perpendicular offset from the tooth's helical run.
    lead = 2.0 * jnp.pi * jnp.maximum(r, 1e-6)
    u = u * lead / jnp.sqrt(lead**2 + pitch**2)

    s = r - r_root
    half = 0.5 * width
    a = jnp.radians(0.5 * flank_deg)  # flank angle from the radial direction

    # Truncated triangle: root plane, the two symmetric flanks, crest cut.
    d = jnp.maximum(-s, jnp.sin(a) * s + jnp.cos(a) * (jnp.abs(u) - half))
    d = jnp.maximum(d, s - depth)

    half_h = 0.5 * n_turns * jnp.abs(pitch)
    return jnp.maximum(d, jnp.abs(pz) - half_h)


def link(p, le, r1, r2):
    """Exact SDF for a chain link."""
    q = jnp.stack([p[..., 0], jnp.maximum(jnp.abs(p[..., 1]) - le, 0.0), p[..., 2]], axis=-1)
    return _length(jnp.stack([_length(q[..., :2]) - r1, q[..., 2]], axis=-1)) - r2


def cone(p, c, h):
    """Exact SDF for a finite cone. c = [sin(angle), cos(angle)], h = height."""
    c = jnp.asarray(c)
    q_vec = h * jnp.array([c[0] / c[1], -1.0])
    w = jnp.stack([_length(p[..., :2]), p[..., 2]], axis=-1)
    a = w - q_vec * _clamp(jnp.sum(w * q_vec, axis=-1) / _dot2(q_vec), 0.0, 1.0)[..., None]
    b = w - q_vec * jnp.stack(
        [_clamp(w[..., 0] / q_vec[0], 0.0, 1.0), jnp.ones_like(w[..., 1])], axis=-1
    )
    k = _sign(q_vec[1])
    d = jnp.minimum(_dot2(a), _dot2(b))
    s = jnp.maximum(k * (w[..., 0] * q_vec[1] - w[..., 1] * q_vec[0]), k * (w[..., 1] - q_vec[1]))
    return jnp.sqrt(d) * _sign(s)


def plane(p, n, h):
    """Exact SDF for an infinite plane. n must be normalised."""
    p = jnp.asarray(p)
    n = jnp.asarray(n)
    _check_axis_vector("plane.n", n, p)
    return jnp.sum(p * n, axis=-1) + h


def hex_prism(p, h):
    """Bound SDF for a hexagonal prism. h = [hex_radius, half_height]."""
    k = jnp.array([-0.8660254, 0.5, 0.57735])
    p2 = jnp.abs(p)
    pxy = p2[..., :2]
    dot_kxy = jnp.sum(k[:2] * pxy, axis=-1)
    pxy = pxy - 2.0 * jnp.minimum(dot_kxy, 0.0)[..., None] * k[:2]
    d = jnp.stack(
        [
            _length(
                pxy
                - jnp.stack(
                    [
                        _clamp(pxy[..., 0], -k[2] * h[0], k[2] * h[0]),
                        jnp.full_like(pxy[..., 1], h[0]),
                    ],
                    axis=-1,
                )
            )
            * _sign(pxy[..., 1] - h[0]),
            p2[..., 2] - h[1],
        ],
        axis=-1,
    )
    return jnp.minimum(jnp.max(d, axis=-1), 0.0) + _length(jnp.maximum(d, 0.0))


def tri_prism(p, h):
    """Bound SDF for a triangular prism. h = [triangle_radius, half_height]."""
    q = jnp.abs(p)
    return jnp.maximum(
        q[..., 2] - h[1],
        jnp.maximum(q[..., 0] * 0.866025 + p[..., 1] * 0.5, -p[..., 1]) - h[0] * 0.5,
    )


def capsule(p, a, b, r):
    """Exact SDF for a capsule between points a and b with radius r."""
    p = jnp.asarray(p)
    a, b = jnp.asarray(a), jnp.asarray(b)
    _check_axis_vector("capsule.a", a, p)
    _check_axis_vector("capsule.b", b, p)
    pa = p - a
    ba = b - a
    h = _clamp(jnp.sum(pa * ba, axis=-1) / _dot2(ba), 0.0, 1.0)
    return _length(pa - ba * h[..., None]) - r


def capped_cylinder(p, h, r):
    """Exact SDF for a capped cylinder of half-height h and radius r."""
    d = jnp.stack([jnp.abs(_length(p[..., :2])) - r, jnp.abs(p[..., 2]) - h], axis=-1)
    return jnp.minimum(jnp.max(d, axis=-1), 0.0) + _length(jnp.maximum(d, 0.0))


def rounded_cylinder(p, ra, rb, h):
    """Exact SDF for a rounded cylinder."""
    d = jnp.stack([_length(p[..., :2]) - 2.0 * ra + rb, jnp.abs(p[..., 2]) - h], axis=-1)
    return jnp.minimum(jnp.max(d, axis=-1), 0.0) + _length(jnp.maximum(d, 0.0)) - rb


def capped_cone(p, h, r1, r2):
    """Exact SDF for a capped cone."""
    q = jnp.stack([_length(p[..., :2]), p[..., 2]], axis=-1)
    k1 = jnp.array([r2, h])
    k2 = jnp.array([r2 - r1, 2.0 * h])
    ca = jnp.stack(
        [
            q[..., 0] - jnp.minimum(q[..., 0], jnp.where(q[..., 1] < 0.0, r1, r2)),
            jnp.abs(q[..., 1]) - h,
        ],
        axis=-1,
    )
    cb = q - k1 + k2 * _clamp(jnp.sum((k1 - q) * k2, axis=-1) / _dot2(k2), 0.0, 1.0)[..., None]
    s = jnp.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)
    return s * jnp.sqrt(jnp.minimum(_dot2(ca), _dot2(cb)))


def solid_angle(p, c, ra):
    """Exact SDF for a solid angle. c = [sin(angle), cos(angle)]."""
    c = jnp.asarray(c)
    q = jnp.stack([_length(p[..., :2]), p[..., 2]], axis=-1)
    l_dist = _length(q) - ra
    m = _length(q - c * _clamp(jnp.sum(q * c, axis=-1), 0.0, ra)[..., None])
    return jnp.maximum(l_dist, m * _sign(c[1] * q[..., 0] - c[0] * q[..., 1]))


def cut_sphere(p, r, h):
    """Exact SDF for a cut sphere."""
    w = jnp.sqrt(r * r - h * h)
    q = jnp.stack([_length(p[..., :2]), p[..., 2]], axis=-1)
    s = jnp.maximum(
        (h - r) * q[..., 0] ** 2 + w**2 * (h + r - 2.0 * q[..., 1]), h * q[..., 0] - w * q[..., 1]
    )
    return jnp.where(
        s < 0.0,
        _length(q) - r,
        jnp.where(q[..., 0] < w, h - q[..., 1], _length(q - jnp.array([w, h]))),
    )


def ellipsoid(p, r):
    """Conservative SDF for an ellipsoid with semi-axes r = [rx, ry, rz].

    The function implements a *bound*, not an exact distance.
    It does under-report along the longer axes, by up to the aspect ratio. Exact
    for a sphere, and exact along the shortest semi-axis of any ellipsoid.

    Not IQ's ``k0*(k0-1)/k1``, which is tighter for near-spherical shapes but
    over-reports near the surface as soon as the ellipsoid is eccentric.
    It also evaluated 0/0 at the centre and returned NaN, which propagated
    into meshing, metrics and gradients. Here we return the exact signed distance
    at the centre. The zero level set is identical to IQ's, so the geometry is
    unchanged.
    """
    p = jnp.asarray(p)
    r = jnp.asarray(r)
    _check_axis_vector("ellipsoid.r", r, p)
    return (_length(p / r) - 1.0) * jnp.min(r)


def octahedron(p, s):
    """Exact SDF for an octahedron (port of IQ's ``sdOctahedron``).

    The field is the regular octahedron ``|x| + |y| + |z| = s``: vertices sit at
    distance ``s`` along each axis.
    """
    p2 = jnp.abs(p)
    px, py, pz = p2[..., 0], p2[..., 1], p2[..., 2]
    m = px + py + pz - s

    cond_a = 3.0 * px < m
    cond_b = 3.0 * py < m
    cond_c = 3.0 * pz < m

    # Fold into a canonical face frame: q = p.xyz / p.yzx / p.zxy depending on
    # which of the three octant faces the query point is nearest.
    qx = jnp.where(cond_a, px, jnp.where(cond_b, py, pz))
    qy = jnp.where(cond_a, py, jnp.where(cond_b, pz, px))
    qz = jnp.where(cond_a, pz, jnp.where(cond_b, px, py))

    k = _clamp(0.5 * (qz - qy + s), 0.0, s)
    face_dist = _length(jnp.stack([qx, qy - s + k, qz - k], axis=-1))

    # IQ's signed-core early-out: when none of the three faces is selected the
    # point lies in the central column and the distance is analytic.
    # WARNING:Omitting this branch leaves the interior positive, so the solid never meshes.
    core = m * 0.57735027
    return jnp.where(cond_a | cond_b | cond_c, face_dist, core)


def pyramid(p, h):
    """Exact SDF for a square-base pyramid of height ``h`` (port of IQ's
    ``sdPyramid``).

    Unit square base on the ``y = 0`` plane spanning ``[-0.5, 0.5]`` in x and z,
    apex at ``(0, h, 0)``.
    """
    m2 = h * h + 0.25
    px, py, pz = jnp.abs(p[..., 0]), p[..., 1], jnp.abs(p[..., 2])
    # order so the larger of |x|,|z| is treated as x (IQ's p.xz = p.zx swap),
    # then shift the base corner to the origin.
    swap = pz > px
    sx = jnp.where(swap, pz, px) - 0.5
    sz = jnp.where(swap, px, pz) - 0.5

    qx = sz
    qy = h * py - 0.5 * sx
    qz = h * sx + 0.5 * py

    s = jnp.maximum(-qx, 0.0)
    t = _clamp((qy - 0.5 * sz) / (m2 + 0.25), 0.0, 1.0)
    a = m2 * (qx + s) ** 2 + qy**2
    b = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2
    d2 = jnp.where(jnp.minimum(qy, -qx * m2 - qy * 0.5) > 0.0, 0.0, jnp.minimum(a, b))

    # The trailing sign is what gives the solid a negative interior; the previous
    # port dropped it (and left the result clamped to >= 0), so the field was
    # single-signed and not a valid SDF.
    return jnp.sqrt(jnp.maximum((d2 + qz**2) / m2, 0.0)) * _sign(jnp.maximum(qz, -py))


# ===========================================================================
# TPMS Lattice Primitives  (implicit / approximate SDFs)
# ===========================================================================
# Each TPMS pattern is mathematically periodic over all of R^3. For part
# design the pattern must occupy a finite extent, so every TPMS primitive
# takes a required ``n_periods=[nx, ny, nz]`` and is clipped to a box of
# half-extents ``0.5 * n_periods * period`` via ``jnp.maximum(pattern, box)``,
# the same "intersect-with-axial-bounds" pattern that ``bellows`` and
# ``serpentine`` already use. The bbox inferrer reads ``n_periods`` and
# ``period`` to compute the AABB exactly.

# Maximum of |grad f| over one period, per family, where ``f`` is the
# dimensionless pattern below evaluated in q-space (q = p * 2*pi/period).
#
# These convert the pattern from a bare number into a length. ``f`` is a sum of
# sines and cosines, so it carries no unit: it climbs at
# ``(2*pi/period) * |grad_q f|`` per millimetre, which is why
# ``abs(f) - thickness`` was not a distance and why its error scaled as
# 1/period. Multiplying by ``period / ((2*pi) * C)`` fixes both.
#
# Each value is exact and each is confirmed to 1e-16 by grid
# search plus gradient ascent in docs/sdf_distances/tpms_constants.py:
#   gyroid / schwarz_d  sqrt(3), attained at the origin
#   schwarz_p           sqrt(3), since |grad f|^2 = sum sin^2 <= 3
#   neovius             7, since df/dx = -sin(x) * (3 + 4 cos(y) cos(z))
#   lidinoid            3*sqrt(3)/2

# fmt: off
_TPMS_GRAD_MAX = {
    "gyroid":    math.sqrt(3.0),
    "schwarz_p": math.sqrt(3.0),
    "schwarz_d": math.sqrt(3.0),
    "neovius":   7.0,
    "lidinoid":  1.5 * math.sqrt(3.0),
}
# fmt: on


def _tpms_sheet(p, f, c_grad_max, period, min_thickness, n_periods):
    """Turn a dimensionless TPMS pattern value into a wall of a given thickness.

    ``abs(f) * inv_rate`` is the distance from the pattern's base surface, using
    the steepest climb the pattern can manage anywhere. Because that is the
    *maximum* rate, the product never over-states the distance: it
    under-states it wherever the pattern is locally flatter, which is the safe
    direction for every caller that reasons from the value.

    ``inv_rate`` depends only on ``period`` and the family constant, not on
    the query point, so it is a single scalar multiply per sample.

    The same reasoning is what makes ``min_thickness`` a floor rather than an
    exact figure: the sheet is thinnest where the pattern is steepest and
    thicker elsewhere. Measured spread across one sheet is 1.22x for gyroid,
    1.73x for schwarz_p and 5.19x for neovius.
    """
    inv_rate = period / ((2.0 * jnp.pi) * c_grad_max)
    return _tpms_clip(p, jnp.abs(f) * inv_rate - 0.5 * min_thickness, period, n_periods)


def _tpms_clip(p, pattern, period, n_periods):
    """Intersect a periodic TPMS ``pattern`` SDF with an axial bounding box.

    Half-extents are ``0.5 * n_periods * period`` (one period per axis is
    ``period`` units long in p-space).
    """
    p = jnp.asarray(p)
    n_periods = jnp.asarray(n_periods)
    _check_axis_vector("tpms.n_periods", n_periods, p)
    half_extents = 0.5 * n_periods * period
    return jnp.maximum(pattern, box(p, half_extents))


def gyroid(p, period, min_thickness, n_periods):
    """Gyroid TPMS lattice, clipped to a finite box.

    Args:
        p: Query point(s) of shape ``(..., 3)`` (array-like).
        period: Spatial period of the pattern, in millimetres: the field
            repeats every ``period`` units along each axis. Named to match
            ``bellows(period=)`` and ``serpentine(wavelength=)``.

            Note the *visible* cell is smaller than this. A gyroid is
            body-centred: it also repeats after ``period/2`` along the
            (1,1,1) diagonal.
        min_thickness: Minimum wall thickness of the sheet, in millimetres.
            TPMS walls are not uniform: measured spread across
            one sheet is 1.22x for gyroid, 1.73x for schwarz_p and 5.19x for
            neovius, so setting 1.0 on a gyroid yields walls between 1.00 and
            1.22 mm.
        n_periods: ``[nx, ny, nz]`` (array-like) -- number of full periods
            along each axis. The lattice is clipped to a box of
            half-extents ``0.5 * n_periods * period``.
    """
    q = p * (2.0 * jnp.pi / period)
    f = (
        jnp.sin(q[..., 0]) * jnp.cos(q[..., 1])
        + jnp.sin(q[..., 1]) * jnp.cos(q[..., 2])
        + jnp.sin(q[..., 2]) * jnp.cos(q[..., 0])
    )
    return _tpms_sheet(p, f, _TPMS_GRAD_MAX["gyroid"], period, min_thickness, n_periods)


def schwarz_p(p, period, min_thickness, n_periods):
    """Schwarz-P TPMS lattice, clipped to a finite box. See :func:`gyroid`."""
    q = p * (2.0 * jnp.pi / period)
    f = jnp.cos(q[..., 0]) + jnp.cos(q[..., 1]) + jnp.cos(q[..., 2])
    return _tpms_sheet(p, f, _TPMS_GRAD_MAX["schwarz_p"], period, min_thickness, n_periods)


def schwarz_d(p, period, min_thickness, n_periods):
    """Schwarz-D (Diamond) TPMS lattice, clipped to a finite box. See :func:`gyroid`."""
    q = p * (2.0 * jnp.pi / period)
    f = (
        jnp.sin(q[..., 0]) * jnp.sin(q[..., 1]) * jnp.sin(q[..., 2])
        + jnp.sin(q[..., 0]) * jnp.cos(q[..., 1]) * jnp.cos(q[..., 2])
        + jnp.cos(q[..., 0]) * jnp.sin(q[..., 1]) * jnp.cos(q[..., 2])
        + jnp.cos(q[..., 0]) * jnp.cos(q[..., 1]) * jnp.sin(q[..., 2])
    )
    return _tpms_sheet(p, f, _TPMS_GRAD_MAX["schwarz_d"], period, min_thickness, n_periods)


def neovius(p, period, min_thickness, n_periods):
    """Neovius TPMS lattice, clipped to a finite box. See :func:`gyroid`."""
    q = p * (2.0 * jnp.pi / period)
    f = 3.0 * (jnp.cos(q[..., 0]) + jnp.cos(q[..., 1]) + jnp.cos(q[..., 2])) + 4.0 * jnp.cos(
        q[..., 0]
    ) * jnp.cos(q[..., 1]) * jnp.cos(q[..., 2])
    return _tpms_sheet(p, f, _TPMS_GRAD_MAX["neovius"], period, min_thickness, n_periods)


def lidinoid(p, period, min_thickness, n_periods):
    """Lidinoid TPMS lattice, clipped to a finite box. See :func:`gyroid`."""
    q = p * (2.0 * jnp.pi / period)
    f = (
        0.5
        * (
            jnp.sin(2.0 * q[..., 0]) * jnp.cos(q[..., 1]) * jnp.sin(q[..., 2])
            + jnp.sin(2.0 * q[..., 1]) * jnp.cos(q[..., 2]) * jnp.sin(q[..., 0])
            + jnp.sin(2.0 * q[..., 2]) * jnp.cos(q[..., 0]) * jnp.sin(q[..., 1])
        )
        - 0.5
        * (
            jnp.cos(2.0 * q[..., 0]) * jnp.cos(2.0 * q[..., 1])
            + jnp.cos(2.0 * q[..., 1]) * jnp.cos(2.0 * q[..., 2])
            + jnp.cos(2.0 * q[..., 2]) * jnp.cos(2.0 * q[..., 0])
        )
        - 0.15
    )
    return _tpms_sheet(p, f, _TPMS_GRAD_MAX["lidinoid"], period, min_thickness, n_periods)


# ===========================================================================
# Compliant Mechanism Primitives
# ===========================================================================


def notch_hinge(p, width, depth, notch_radius):
    """
    Approximate SDF for a notch (leaf) flexure hinge.
    Centred at origin, hinge axis along Z.
    width  : full width of the beam
    depth  : full depth (thickness) of the beam
    notch_radius : radius of the circular notch cut
    """
    beam = box(p, jnp.array([width / 2.0, depth / 2.0, width / 2.0]))
    cy = depth / 2.0 - notch_radius
    notch_top = _length(p[..., :2] - jnp.array([0.0, cy])) - notch_radius
    notch_bot = _length(p[..., :2] - jnp.array([0.0, -cy])) - notch_radius
    return op_subtract(op_subtract(beam, notch_top), notch_bot)


def leaf_spring(p, length, width, thickness):
    """Approximate SDF for a straight leaf spring (thin rectangular beam)."""
    return box(p, jnp.array([length / 2.0, thickness / 2.0, width / 2.0]))


def bellows(p, outer_r, inner_r, period, n_periods):
    """Corrugated rod, axis along Z. Solid to the axis: there is no bore.

    The radius sweeps sinusoidally between ``inner_r`` and ``outer_r`` every
    ``period``, over a total length of ``n_periods * period``.

    ``length(p.xy) - r_mod(z)`` measures the offset *radially*, but the surface
    is tilted wherever ``r_mod`` is changing, so that offset is larger than the
    perpendicular distance by ``sqrt(1 + r_mod'(z)^2)``. Dividing by the
    steepest tilt the profile can reach turns it back into a distance:
    ``r_mod'`` is a sine of known amplitude, so the bound is exact and costs
    nothing to evaluate.

    The quotient under-reports wherever the profile is locally flatter than
    its steepest point, by up to that same factor (2.3x at the default proportions,
    more for deep or fine corrugations).
    """
    total_length = n_periods * period
    d_radius = outer_r - inner_r
    max_slope = d_radius * jnp.pi / period  # max |d(r_mod)/dz|
    pz_clamped = _clamp(p[..., 2], -total_length / 2.0, total_length / 2.0)
    r_mod = inner_r + d_radius * 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * pz_clamped / period))
    # hypot(1, s) is the hypotenuse of a right triangle with legs 1 and s
    # (i.e. sqrt(1 + s^2)), and is more numerically stable than jnp.sqrt.
    radial = (_length(p[..., :2]) - r_mod) / jnp.hypot(1.0, max_slope)
    axial = jnp.abs(p[..., 2]) - total_length / 2.0
    return jnp.maximum(radial, axial)


def serpentine(p, amplitude, wavelength, beam_width, beam_height, n_periods):
    """Serpentine (meander) spring in the XY plane, running along X.

    A beam of rectangular cross-section (``beam_width`` in Y, ``beam_height``
    in Z) following a sinusoidal centreline of the given ``amplitude`` and
    ``wavelength``, for ``n_periods * wavelength`` of travel.

    ``abs(p.y - y_centre(x))`` measures the offset *vertically*, which exceeds
    the perpendicular distance by ``sqrt(1 + y_centre'(x)^2)`` wherever the
    centreline is tilted. Dividing by the steepest tilt (known exactly, since
    the centreline is a sine of given amplitude and wavelength) turns it back
    into a distance.

    The division is applied after subtracting the half-width so the beam keeps
    its original Y extent; normalising the offset first would widen it by the
    same factor. the distance can be 4.1x smaller than the exact distance at
    the default proportions, more for tight, high-amplitude meanders.
    """
    total_length = n_periods * wavelength
    max_slope = amplitude * 2.0 * jnp.pi / wavelength  # max |d(y_centre)/dx|
    y_centre = amplitude * jnp.sin(2.0 * jnp.pi * p[..., 0] / wavelength)
    dist_to_centreline = jnp.abs(p[..., 1] - y_centre)
    beam = jnp.maximum(
        (dist_to_centreline - beam_width / 2.0) / jnp.hypot(1.0, max_slope),
        jnp.abs(p[..., 2]) - beam_height / 2.0,
    )
    axial = jnp.abs(p[..., 0]) - total_length / 2.0
    return jnp.maximum(beam, axial)


def annular_sector(p, inner_r, outer_r, half_angle, height):
    """SDF for an annular sector (arc-shaped beam).

    Centred at origin, arc in the XY plane, extruded along Z. ``half_angle`` is
    in radians and may exceed pi/2 (a reflex sector).

    The angular bound is the distance to the two bounding half-planes, not
    ``abs(theta) - half_angle``. An angle is not a length: ``theta`` changes at
    rate ``1/r``, so the Inigo Quilez form over-reported distance by exactly ``1/inner_r``
    near the bore (measured 5.06x at ``inner_r=0.2``). Both bounding planes pass
    through the Z axis, so their unit-normal dot products are exact distances and
    the rate is 1.000 everywhere.

    A sector up to pi/2 is the *intersection* of the two half-spaces; beyond
    pi/2 it wraps around and becomes their *union*.
    """
    s_a, c_a = jnp.sin(half_angle), jnp.cos(half_angle)
    d_plane_pos = -s_a * p[..., 0] + c_a * p[..., 1]  # +half_angle boundary
    d_plane_neg = -s_a * p[..., 0] - c_a * p[..., 1]  # -half_angle boundary
    d_angular = jnp.where(
        half_angle <= 0.5 * jnp.pi,
        jnp.maximum(d_plane_pos, d_plane_neg),
        jnp.minimum(d_plane_pos, d_plane_neg),
    )
    r = _length(p[..., :2])
    d_radial = jnp.maximum(inner_r - r, r - outer_r)
    d_axial = jnp.abs(p[..., 2]) - height / 2.0
    return jnp.maximum(jnp.maximum(d_radial, d_angular), d_axial)


# ===========================================================================
# 2-D Primitives  (for Revolution / Extrusion)
# ===========================================================================


def circle_2d(p, r):
    return _length(p) - r


def box_2d(p, b):
    p = jnp.asarray(p)
    b = jnp.asarray(b)
    _check_axis_vector("box_2d.b", b, p)
    q = jnp.abs(p) - b
    return _length(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q, axis=-1), 0.0)


def rounded_box_2d(p, b, r):
    p = jnp.asarray(p)
    b = jnp.asarray(b)
    _check_axis_vector("rounded_box_2d.b", b, p)
    q = jnp.abs(p) - b + r
    return _length(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q, axis=-1), 0.0) - r


def segment_2d(p, a, b):
    a, b = jnp.asarray(a), jnp.asarray(b)
    pa = p - a
    ba = b - a
    h = _clamp(jnp.sum(pa * ba, axis=-1) / _dot2(ba), 0.0, 1.0)
    return _length(pa - ba * h[..., None])


def trapezoid_2d(p, r1, r2, he):
    """2-D trapezoid: r1 = bottom half-width, r2 = top half-width, he = half-height."""
    k1 = jnp.array([r2, he])
    k2 = jnp.array([r2 - r1, 2.0 * he])
    q = jnp.stack([jnp.abs(p[..., 0]), p[..., 1]], axis=-1)
    ca = jnp.stack(
        [
            q[..., 0] - jnp.minimum(q[..., 0], jnp.where(q[..., 1] < 0.0, r1, r2)),
            jnp.abs(q[..., 1]) - he,
        ],
        axis=-1,
    )
    cb = q - k1 + k2 * _clamp(jnp.sum((k1 - q) * k2, axis=-1) / _dot2(k2), 0.0, 1.0)[..., None]
    s = jnp.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)
    return s * jnp.sqrt(jnp.minimum(_dot2(ca), _dot2(cb)))


def uneven_capsule_2d(p, r1, r2, h):
    """2-D uneven capsule (two different end radii)."""
    q = jnp.stack([jnp.abs(p[..., 0]), p[..., 1]], axis=-1)
    b = (r1 - r2) / h
    a = jnp.sqrt(1.0 - b * b)
    k = jnp.sum(q * jnp.array([-b, a]), axis=-1)
    return jnp.where(
        k < 0.0,
        _length(q) - r1,
        jnp.where(
            k > a * h,
            _length(q - jnp.array([0.0, h])) - r2,
            jnp.sum(q * jnp.array([a, b]), axis=-1) - r1,
        ),
    )


def polygon_2d(p, vertices):
    """2-D exact signed distance to a simple closed polygon (Inigo Quilez).

    The polygon may be convex or non-convex, but must be simple (non
    self-intersecting). Winding is arbitrary: the winding-parity sign test
    produces the correct inside/outside regardless of CW/CCW.

    Args:
        p: ``(..., 2)`` query points.
        vertices: ``(N, 2)`` ordered polygon vertices, ``N >= 3``. The
            ``vertices`` array is baked into the JIT trace via its shape;
            polygons with different vertex counts compile to different
            traces. For parametric vertices, drive via free-param values
            in the DSL (each vertex is two leaves).

            Duplicate closing vertex (last == first) is handled silently:
            it produces one zero-length edge that the guard below pins to
            length 1.0; that edge's foot-of-perpendicular distance is then
            equal to the distance to the duplicated vertex (also reported
            by an adjacent real edge), so it doesn't change ``min(d_i)``.
            Its ray-crossing test also contributes zero flips. Net effect:
            including the closing vertex is a no-op.

    Notes:
        Distance: ``min_i  d(p, edge_i)``, minimum perpendicular distance
        across all edges, where each edge distance is computed via the
        classic foot-of-perpendicular formula with a clamp to ``[0, 1]``.

        Sign: ray-cast parity. For each edge, flip the sign iff a +x ray
        from ``p`` crosses it. Odd crossings → inside.
    """
    v = jnp.asarray(vertices)  # (N, 2)
    if v.ndim != 2 or v.shape[-1] != 2 or v.shape[0] < 3:
        raise ValueError(f"polygon needs an (N, 2) array with N >= 3, got shape {v.shape}")
    v_prev = jnp.roll(v, 1, axis=0)  # (N, 2)
    e = v_prev - v  # (N, 2) edge vectors v_prev - v
    ee = jnp.sum(e * e, axis=-1)  # (N,)
    ee = jnp.where(ee > 1e-12, ee, 1.0)  # guard zero-length edges

    p_exp = p[..., jnp.newaxis, :]  # (..., 1, 2)
    w = p_exp - v  # (..., N, 2)
    t = _clamp(jnp.sum(w * e, axis=-1) / ee, 0.0, 1.0)
    b = w - t[..., jnp.newaxis] * e  # foot-of-perpendicular offset
    dist_sq = jnp.sum(b * b, axis=-1)  # (..., N)
    d = jnp.min(dist_sq, axis=-1)  # (...)

    # Winding-parity sign (Quilez): one bit-flip per edge crossed by the +x ray.
    v_y = v[..., 1]
    v_prev_y = v_prev[..., 1]
    p_y = p[..., 1:2]  # (..., 1)
    c1 = p_y >= v_y
    c2 = p_y < v_prev_y
    c3 = (e[..., 0] * w[..., 1]) > (e[..., 1] * w[..., 0])
    flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
    inside = (jnp.sum(flip.astype(jnp.int32), axis=-1) % 2) == 1
    sign = jnp.where(inside, -1.0, 1.0)

    return sign * jnp.sqrt(d)


# ---------------------------------------------------------------------------
# 2-D smooth-curve profiles (sampled to a polygon, then exact polygon SDF)
# ---------------------------------------------------------------------------
# ``bezier_2d`` / ``bspline_2d`` are closed 2-D regions whose boundary is a
# smooth curve. Both evaluate the curve at a fixed number of samples per
# segment, forming a closed polyline, and return its EXACT polygon SDF
# (:func:`polygon_2d`). The samples are a linear function of the control
# points, so the SDF stays differentiable w.r.t. them; the curve lies inside
# the convex hull of the control points, so the primitive is self-bounding
# (see :mod:`software_defined_matter.sdf.bbox`). Raise the sample count if a
# facet is visible at your export voxel size.
_CURVE_SAMPLES_PER_SEGMENT = 16


def bezier_2d(p, control_points):
    """Closed composite **cubic Bézier** region (exact polygon SDF of the
    sampled outline).

    ``control_points`` is a ``(3K, 2)`` array describing ``K`` cubic segments
    laid end-to-end and closed back to the start: segment ``i`` uses control
    points ``(P[3i], P[3i+1], P[3i+2], P[3i+3 mod 3K])``. Indices ``0, 3, 6, …``
    are on-curve anchors; the two between each pair are off-curve handles
    (SVG cubic-path convention). ``K >= 2`` (so ``len >= 6`` and divisible by 3).

    Sampled at :data:`_CURVE_SAMPLES_PER_SEGMENT` points per segment and handed
    to :func:`polygon_2d`. Differentiable w.r.t. the control points; drive any
    coordinate via a free-param leaf for a parametric profile.

    .. note::
        The control cage must produce a **simple** (non-self-intersecting)
        outline. The sign comes from :func:`polygon_2d`'s even-odd (ray-parity)
        rule, so if the sampled curve crosses itself (easy to do with the
        off-curve handles), a region covered an even number of times reads as
        *outside* (positive), giving a wrong-signed SDF and a disconnected solid.
        No check is performed here.

    Args:
        p: ``(..., 2)`` query points.
        control_points: ``(3K, 2)`` control points, ``K >= 2``.
    """
    v = jnp.asarray(control_points)
    if v.ndim != 2 or v.shape[-1] != 2 or v.shape[0] < 6 or v.shape[0] % 3 != 0:
        raise ValueError(
            "bezier_2d needs a (3K, 2) control-point array with K >= 2 "
            f"(length divisible by 3 and >= 6), got shape {v.shape}"
        )
    m = v.shape[0]
    k = m // 3
    s = _CURVE_SAMPLES_PER_SEGMENT
    t = jnp.linspace(0.0, 1.0, s, endpoint=False)  # (s,)
    mt = 1.0 - t
    basis = jnp.stack([mt**3, 3.0 * mt**2 * t, 3.0 * mt * t**2, t**3], axis=-1)  # (s, 4) Bernstein
    seg = jnp.arange(k) * 3  # (k,)
    idx = (seg[:, None] + jnp.arange(4)[None, :]) % m  # (k, 4) cyclic
    windows = v[idx]  # (k, 4, 2)
    pts = jnp.einsum("sj,kjc->ksc", basis, windows)  # (k, s, 2)
    return polygon_2d(p, pts.reshape(k * s, 2))


def bspline_2d(p, control_points):
    """Closed **periodic uniform cubic B-spline** region (exact polygon SDF of
    the sampled outline).

    ``control_points`` is an ``(M, 2)`` array, ``M >= 4``, treated as a closed
    control polygon. The curve *approximates* (does not interpolate) the control
    points and is automatically C²: the ergonomic way to get a smooth closed
    profile from a handful of points. Span ``j`` uses control points
    ``(P[j], P[j+1], P[j+2], P[j+3])`` (indices mod ``M``) with the uniform cubic
    basis; the ``M`` spans wrap around to close the loop.

    Sampled at :data:`_CURVE_SAMPLES_PER_SEGMENT` points per span and handed to
    :func:`polygon_2d`. Differentiable w.r.t. the control points.
    .. note::
        The control cage must produce a **simple** (non-self-intersecting)
        outline. The sign comes from :func:`polygon_2d`'s even-odd (ray-parity)
        rule, so if the sampled curve crosses itself (e.g. a re-ordered control
        polygon), a region covered an even number of times reads as *outside*
        (positive), giving a wrong-signed SDF and a disconnected solid. No check
        is performed here.

    Args:
        p: ``(..., 2)`` query points.
        control_points: ``(M, 2)`` control points, ``M >= 4``.
    """
    v = jnp.asarray(control_points)
    if v.ndim != 2 or v.shape[-1] != 2 or v.shape[0] < 4:
        raise ValueError(
            f"bspline_2d needs an (M, 2) control-point array with M >= 4, got shape {v.shape}"
        )
    m = v.shape[0]
    s = _CURVE_SAMPLES_PER_SEGMENT
    t = jnp.linspace(0.0, 1.0, s, endpoint=False)  # (s,)
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
    )  # (s, 4) uniform cubic
    idx = (jnp.arange(m)[:, None] + jnp.arange(4)[None, :]) % m  # (m, 4) cyclic
    windows = v[idx]  # (m, 4, 2)
    pts = jnp.einsum("sj,mjc->msc", basis, windows)  # (m, s, 2)
    return polygon_2d(p, pts.reshape(m * s, 2))


# ===========================================================================
# Scalar Field Primitives  (for use as displacement fields in op_displace)
# ===========================================================================
# A field is a function (p, free_vec) -> scalar that does NOT represent a
# distance. It is consumed by op_displace to perturb a true SDF's output.


def field_sin_xyz(p, freq, amplitude=1.0, phase=(0.0, 0.0, 0.0)):
    """Separable 3-D sinusoidal corrugation (product form).

    ``field(p) = amplitude * prod_i sin(2 pi * freq[i] * p[i] + phase[i])``

    Useful for lattice-like surface textures. Pass per-axis ``freq=0`` to
    drop a direction (the corresponding ``sin(phase)`` becomes a constant
    factor).
    """
    freq = jnp.asarray(freq)
    phase = jnp.asarray(phase)
    arg = 2.0 * jnp.pi * freq * p + phase
    s = jnp.sin(arg)
    return amplitude * s[..., 0] * s[..., 1] * s[..., 2]


def field_radial(p, freq, amplitude=1.0, phase=0.0):
    """Axially symmetric ripple in the XY plane.

    ``field(p) = amplitude * sin(2 pi * freq * |p_xy| + phase)``

    Produces concentric ripples around the Z axis; useful for fluted /
    grooved surfaces of revolution.
    """
    r = _length(p[..., :2])
    return amplitude * jnp.sin(2.0 * jnp.pi * freq * r + phase)


def field_angular(p, freq, amplitude=1.0, phase=0.0):
    """Azimuthal ripple about the Z axis (angular companion to ``field_radial``).

    ``field(p) = amplitude * sin(freq * atan2(p_y, p_x) + phase)``

    Produces ``freq`` evenly spaced lobes around the Z axis (a fluted /
    scalloped / petalled pattern), constant along any radius. Useful for
    angularly graded surfaces of revolution, e.g. grading a disc's
    thickness to follow an azimuthally varying load. Use an INTEGER
    ``freq`` for a seam-free closed pattern: the value is periodic in
    theta with period ``2 pi / freq``, and only an integer lobe count is
    continuous across the ``atan2`` branch cut at theta = +/- pi (a
    non-integer ``freq`` leaves a step there). On the Z axis itself
    (``p_xy = 0``) the angle is 0 by convention, so the field is
    ``amplitude * sin(phase)``, a removable seam that does not reach the
    surface of a bored annulus.

    The angle comes from :func:`_azimuth`, whose gradient is floored near the
    axis (see ``_AXIS_EPSILON``): the raw ``d(atan2)/dp`` is ``NaN`` on the Z
    axis and ``~1/r`` next to it, which would poison autodiff over any grid
    that samples the axis; the floored JVP is finite and bounded there. The
    field *value* is unchanged.
    """
    theta = _azimuth(p)
    return amplitude * jnp.sin(freq * theta + phase)


def raster_field(p, origin, spacing, values):
    """Bound SDF from a box-aligned grid of distance samples (trilinear).

    ``values`` is the decoded sample block, shape ``(nz, ny, nx)`` —
    ``values[k, j, i]`` sits at ``origin + spacing * (i, j, k)`` (see
    ``sdf/raster.py`` for the wire codec). The analytic tree stays the
    source of truth: a grid is a derived artifact baked from a reference
    field at authoring time (``sdf/bake.py``), carried inline so heavy
    constructions (OW-II's 242-pose ROM carves) evaluate as one fetch
    instead of thousands of node calls. See DR-0003.

    Semantics, kept EXACTLY in lockstep with GLSL ``sdf_raster_field``:

    - inside ``origin .. origin + spacing*(dims-1)``: trilinear
      interpolation of the 8 surrounding samples;
    - outside: the value at the clamped coordinates plus the euclidean
      distance to the domain box. Continuous across the boundary, and
      safely positive for a cutter whose zero-set keeps the authored
      >= 2-voxel margin (``bake_raster_field`` warns when it doesn't).

    BOUND, not exact — same discipline as :func:`quadric_halfspace`.
    Trilinear interpolation of exact-SDF samples has per-axis slope <= 1
    but gradient magnitude up to sqrt(3) across cell diagonals, plus
    O(h^2 * curvature) sampling error near curved surfaces. A sphere
    tracer should scale steps by the node's ``step_scale`` param
    (default 1/sqrt(3) ~ 0.577; bakes may store a measured, larger value).
    Degenerate axes (``n == 1``, the 2-D mask case) interpolate as
    constant along that axis.
    """
    p = jnp.asarray(p)
    values = jnp.asarray(values)
    nz, ny, nx = values.shape
    origin = jnp.asarray(origin, dtype=p.dtype)
    spacing = jnp.asarray(spacing, dtype=p.dtype)
    hi = origin + spacing * jnp.array([nx - 1, ny - 1, nz - 1], dtype=p.dtype)
    q = jnp.clip(p, origin, hi)
    # Safe norm: interior points have p == q and sqrt'(0) is NaN under
    # autodiff — the double-where keeps d_out (and its gradient) exactly 0
    # inside the domain instead of poisoning jax.grad.
    sq = jnp.sum((p - q) ** 2, axis=-1)
    d_out = jnp.where(sq > 0.0, jnp.sqrt(jnp.where(sq > 0.0, sq, 1.0)), 0.0)
    u = (q - origin) / spacing
    # Cell index, clamped so i0 + 1 stays in range; a degenerate axis
    # (n == 1) clamps to 0 and its stride below is 0, so the +1 corner
    # re-reads the same sample and the axis interpolates as constant.
    i_max = jnp.array([max(nx - 2, 0), max(ny - 2, 0), max(nz - 2, 0)])
    i0 = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, i_max)
    f = jnp.clip(u - i0, 0.0, 1.0)
    ix, iy, iz = i0[..., 0], i0[..., 1], i0[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    sx = 1 if nx > 1 else 0
    sy = 1 if ny > 1 else 0
    sz = 1 if nz > 1 else 0
    c000 = values[iz, iy, ix]
    c100 = values[iz, iy, ix + sx]
    c010 = values[iz, iy + sy, ix]
    c110 = values[iz, iy + sy, ix + sx]
    c001 = values[iz + sz, iy, ix]
    c101 = values[iz + sz, iy, ix + sx]
    c011 = values[iz + sz, iy + sy, ix]
    c111 = values[iz + sz, iy + sy, ix + sx]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz + d_out


def field_add(*field_values):
    """Sum a sequence of pre-evaluated field values."""
    out = field_values[0]
    for v in field_values[1:]:
        out = out + v
    return out
