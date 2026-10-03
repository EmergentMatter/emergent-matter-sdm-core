"""The two interpolated twists, and the sign convention that separates them.

``twist_radial`` ramps a Z-rotation over cylindrical radius; ``twist_linear``
ramps one over a signed linear coordinate. Same Hermite profile, same
Beam-Constraint-Model stand-in argument, and DELIBERATELY OPPOSITE SIGNS.

That disagreement is the reason this file exists. ``twist_linear``'s two end
angles are, in practice, the same pose params that ``rotate_z`` the bodies its
shape is welded between, so it has to match ``rotate_z``. Feeding one param to a
``twist_radial``-signed operator instead drove the two apart: every blade
counter-rotated against its own ring at DOUBLE the intended relative angle,
while still looking plausibly animated.

The test that failed to catch it hand-rolled ``R(-a)`` inline, restating the
implementation rather than pinning it to anything, so code and test were free to
be wrong together. Every sign assertion here is therefore made against
``transforms.tf_rotate_z`` -- an INDEPENDENT operator with its own tests -- or
against geometry, never against a rotation matrix spelled out locally.

A consumer cannot paper over the mismatch from outside, which is what makes the
convention load-bearing rather than a preference: an angle slot in a ``.sdm`` is
a plain ``$ref`` with nowhere to hang a negation.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import Part, sdf_deform, sdf_primitive
from software_defined_matter.glsl.emit import load_lib_glsl
from software_defined_matter.sdf import sdf_ops as ops
from software_defined_matter.sdf import transforms
from software_defined_matter.sdf.bbox import infer_sdf_bbox

# Well clear of float32 noise and inside every shipped deflection range, so a
# negated, halved or degree/radian-confused angle misses by orders of magnitude.
ANGLE = 0.35


def _probe_points():
    """Points spanning the ramp, off-axis in every octant.

    Deliberately NOT symmetric about the origin and never AT it: a symmetric box
    returns the same distance for a point and its mirror image, and a probe at
    the origin has no measurable rotation. Two earlier versions of these tests
    were blind rather than failing for exactly those two reasons.
    """
    xs = np.linspace(-2.0, 2.0, 9)
    return np.asarray(
        [(x, y, z) for x in xs for y in (0.3, 1.1) for z in (-0.4, 0.6)],
        dtype=np.float64,
    )


def _mapped(op, p, **kw):
    """The query point an op hands its child -- the map itself, not a distance.

    Asserting on the distance of a symmetric child is how the first version of
    these tests passed while the sign was wrong.
    """
    seen = {}

    def spy(q):
        seen["q"] = q
        return jnp.zeros(q.shape[:-1])

    op(spy, jnp.asarray(p), **kw)
    return np.asarray(seen["q"], dtype=np.float64)


def _turn(src, dst):
    """The signed rotation about Z carrying ``src`` to ``dst``, per point.

    The query map is a per-point Z-rotation, so its angle is recoverable and is
    the quantity every claim below is really about. Comparing raw coordinates
    instead is what made the first draft of two of these tests assert the
    opposite of the truth: the query turns by ``+a`` while the MATERIAL turns by
    ``-a``, and for a point at ``-X`` with angle ``-a`` those cancel into a
    coordinate that looks identical to the ``+X``/``+a`` case.
    """
    return np.arctan2(dst[:, 1], dst[:, 0]) - np.arctan2(src[:, 1], src[:, 0])


# ---------------------------------------------------------------------------
# The sign contract
# ---------------------------------------------------------------------------


def test_twist_linear_with_equal_ends_is_exactly_rotate_z():
    """The degenerate case IS the sign contract.

    Equal end angles make the ramp constant, so the whole map collapses to one
    Z-rotation -- and it must be the SAME rotation ``tf_rotate_z`` applies for
    that angle, because in practice one pose param drives both.
    """
    p = _probe_points()
    got = _mapped(ops.op_twist_linear, p, u_axis=[1.0, 0.0], u0=-2.0, u1=2.0, a0=ANGLE, a1=ANGLE)
    want = np.asarray(transforms.tf_rotate_z(jnp.asarray(p), ANGLE), dtype=np.float64)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_twist_radial_negates_where_twist_linear_does_not():
    """The disagreement, pinned so it cannot be 'fixed' into agreement.

    Both collapse to a rigid Z-rotation at constant angle; ``twist_radial``
    applies the opposite one. If a future edit makes these agree, one of the two
    conventions has silently moved and a part will bend the wrong way.
    """
    p = _probe_points()
    linear = _mapped(ops.op_twist_linear, p, u_axis=[1.0, 0.0], u0=-2.0, u1=2.0, a0=ANGLE, a1=ANGLE)
    radial = _mapped(ops.op_twist_radial, p, r0=0.0, r1=3.0, a0=ANGLE, a1=ANGLE)
    np.testing.assert_allclose(
        radial, np.asarray(transforms.tf_rotate_z(jnp.asarray(p), -ANGLE)), rtol=1e-6, atol=1e-6
    )
    assert not np.allclose(linear, radial), (
        "the two twists agree on sign: one convention has moved; twist_linear "
        "must match rotate_z and twist_radial must negate"
    )


def test_a_positive_end_angle_swings_the_MATERIAL_toward_minus_y():
    """The convention stated as geometry, checked on where the solid ends up.

    Query space and material space turn OPPOSITE ways, so an assertion about
    the mapped point proves the convention only to someone who already knows
    which one they are looking at. This asks the field instead: put the child's
    material at ``+X`` and a positive angle must move it to ``-y``, which is
    what ``rotate_z`` does and what a reader can check against a picture.
    """

    def child(q):
        return jnp.linalg.norm(q - jnp.asarray([1.0, 0.0, 0.0]), axis=-1) - 0.15

    kw = {"u_axis": [1.0, 0.0], "u0": -1.0, "u1": 1.0, "a0": ANGLE, "a1": ANGLE}
    c, s_ = float(np.cos(ANGLE)), float(np.sin(ANGLE))
    minus_y = jnp.asarray([[c, -s_, 0.0]])
    plus_y = jnp.asarray([[c, +s_, 0.0]])
    assert float(ops.op_twist_linear(child, minus_y, **kw)[0]) < 0.0, (
        "the material did not swing toward -y for a positive angle"
    )
    assert float(ops.op_twist_linear(child, plus_y, **kw)[0]) > 0.0


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


def test_the_two_ends_of_a_diameter_reach_their_own_authored_angles():
    """The whole reason ``twist_linear`` exists, plus the ramp's endpoints.

    A full-diameter plate welds to ring *s* at one end and ring *s+1* at the
    other, and those ends sit at the SAME radius. ``twist_radial`` is symmetric
    in rho, so it turns both ends identically and the plate can only swing
    rigidly. A linear ramp gives each end its OWN angle -- and here they are
    opposite, which is the bend.
    """
    ends = np.asarray([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    linear = _turn(
        ends,
        _mapped(ops.op_twist_linear, ends, u_axis=[1.0, 0.0], u0=-2.0, u1=2.0, a0=-ANGLE, a1=ANGLE),
    )
    np.testing.assert_allclose(linear, [-ANGLE, ANGLE], rtol=1e-6, atol=1e-6)

    radial = _turn(ends, _mapped(ops.op_twist_radial, ends, r0=0.0, r1=2.0, a0=-ANGLE, a1=ANGLE))
    (
        np.testing.assert_allclose(radial, [-ANGLE, -ANGLE], rtol=1e-6, atol=1e-6),
        (
            "twist_radial is meant to be symmetric in rho: if it is not, the "
            "argument for twist_linear existing has changed"
        ),
    )


def test_the_ramp_clamps_past_each_welded_end():
    """Outside ``[u0, u1]`` the angle holds at its end value.

    A shape that kept ramping past its weld would shear material that is bolted
    to something rigid.
    """
    far = np.asarray([[-50.0, 0.7, 0.0], [50.0, 0.7, 0.0]])
    at = np.asarray([[-2.0, 0.7, 0.0], [2.0, 0.7, 0.0]])
    kw = {"u_axis": [1.0, 0.0], "u0": -2.0, "u1": 2.0, "a0": -ANGLE, "a1": ANGLE}

    np.testing.assert_allclose(
        _turn(far, _mapped(ops.op_twist_linear, far, **kw)),
        _turn(at, _mapped(ops.op_twist_linear, at, **kw)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_the_axis_selects_the_ramp_direction_and_its_scale_is_free():
    """``axis`` is normalised, so only its direction matters.

    A part authoring ``[0, 3]`` rather than ``[0, 1]`` must not get three times
    the ramp rate.
    """
    p = _probe_points()
    kw = {"u0": -2.0, "u1": 2.0, "a0": -ANGLE, "a1": ANGLE}
    unit = _mapped(ops.op_twist_linear, p, u_axis=[0.0, 1.0], **kw)
    scaled = _mapped(ops.op_twist_linear, p, u_axis=[0.0, 3.0], **kw)
    np.testing.assert_allclose(unit, scaled, rtol=1e-6, atol=1e-6)

    along_x = _mapped(ops.op_twist_linear, p, u_axis=[1.0, 0.0], **kw)
    assert not np.allclose(unit, along_x), "the axis did not select the ramp direction"


# ---------------------------------------------------------------------------
# The bbox rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node",
    [
        sdf_deform(
            "twist_linear",
            sdf_primitive("box", b=[2.0, 0.5, 1.0]),
            axis=[1.0, 0.0],
            u0=-2.0,
            u1=2.0,
            angle_0=-ANGLE,
            angle_1=ANGLE,
        ),
        sdf_deform(
            "twist_radial",
            sdf_primitive("box", b=[2.0, 0.5, 1.0]),
            r0=0.0,
            r1=2.0,
            angle_inner=-ANGLE,
            angle_outer=ANGLE,
        ),
    ],
    ids=["twist_linear", "twist_radial"],
)
def test_the_bbox_is_the_swept_disc_and_leaves_z_alone(node):
    """Every point stays on its own circle about Z, so the bound is the child's
    max XY radius and Z is untouched -- and it does NOT depend on the angles,
    which is what makes it safe under live pose params whose values are not
    known when the box is inferred."""
    (xlo, ylo, zlo), (xhi, yhi, zhi) = infer_sdf_bbox(node, Part(name="t"))
    r = float(np.hypot(2.0, 0.5))
    np.testing.assert_allclose([xlo, ylo, xhi, yhi], [-r, -r, r, r], rtol=1e-6)
    assert (zlo, zhi) == (-1.0, 1.0)


# ---------------------------------------------------------------------------
# GLSL <-> JAX
# ---------------------------------------------------------------------------
# No GL context in CI, so the bodies are transcribed into numpy and compared
# against the JAX ops, with source needles pinning the transcription to
# lib.glsl. Same pattern, and same reasoning, as test_glsl_python_parity.


def _glsl_body(fn_name: str) -> str:
    lib = re.sub(r"//[^\n]*", "", load_lib_glsl())
    m = re.search(rf"^\w+\s+{re.escape(fn_name)}\s*\([^)]*\)\s*\{{(.*?)\n\}}", lib, re.S | re.M)
    assert m, f"{fn_name} not found in lib.glsl"
    return m.group(1)


def test_the_transcriptions_below_are_still_what_lib_glsl_says():
    """Pin the numpy copies to their source.

    Without this an edit to lib.glsl leaves the transcription behind and the
    comparisons go on passing against a body nobody ships. The SIGN needles are
    the point: they are the one character that distinguishes the two operators.
    """
    rotation = "vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z)"
    ramp = "float w = t * t * (3.0 - 2.0 * t);"

    radial = _glsl_body("op_twist_radial")
    assert "clamp((length(p.xy) - r0) / (r1 - r0), 0.0, 1.0)" in radial
    assert ramp in radial
    assert "float a = -(a0 + (a1 - a0) * w);" in radial  # NEGATED
    assert rotation in radial

    linear = _glsl_body("op_twist_linear")
    assert "normalize(uAxis)" in linear
    assert "clamp((dot(p.xy, ax) - u0) / (u1 - u0), 0.0, 1.0)" in linear
    assert ramp in linear
    # NOT negated -- the whole convention, in one line.
    assert "float a = a0 + (a1 - a0) * w;" in linear
    assert rotation in linear


def _glsl_twist_linear(p, ax, u0, u1, a0, a1):
    ax = np.asarray(ax, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    t = np.clip((p[:, :2] @ ax - u0) / (u1 - u0), 0.0, 1.0)
    a = a0 + (a1 - a0) * (t * t * (3.0 - 2.0 * t))
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], axis=-1)


def _glsl_twist_radial(p, r0, r1, a0, a1):
    t = np.clip((np.hypot(p[:, 0], p[:, 1]) - r0) / (r1 - r0), 0.0, 1.0)
    a = -(a0 + (a1 - a0) * (t * t * (3.0 - 2.0 * t)))
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], axis=-1)


def test_twist_linear_glsl_transcription_matches_jax():
    p = _probe_points()
    got = _mapped(ops.op_twist_linear, p, u_axis=[0.6, 0.8], u0=-1.5, u1=1.5, a0=-ANGLE, a1=ANGLE)
    np.testing.assert_allclose(
        _glsl_twist_linear(p, [0.6, 0.8], -1.5, 1.5, -ANGLE, ANGLE), got, rtol=1e-6, atol=1e-6
    )


def test_twist_radial_glsl_transcription_matches_jax():
    p = _probe_points()
    got = _mapped(ops.op_twist_radial, p, r0=0.2, r1=2.5, a0=-ANGLE, a1=ANGLE)
    np.testing.assert_allclose(
        _glsl_twist_radial(p, 0.2, 2.5, -ANGLE, ANGLE), got, rtol=1e-6, atol=1e-6
    )
