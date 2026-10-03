"""Unit tests for the ``field_angular`` displacement-field primitive.

``field_angular`` is the azimuthal companion to ``field_radial``: it varies
with the angle ``atan2(y, x)`` instead of the radius ``|xy|``, producing
``freq`` evenly spaced lobes around the Z axis. These are deliberately basic
examples (the math, the compile/eval path, and the bbox path), not a
downstream integration.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    field_primitive,
    make_param_ref,
    sdf_deform,
    sdf_primitive,
)
from software_defined_matter.sdf import sdf_shapes as shapes
from software_defined_matter.sdf._helpers import _AXIS_EPSILON
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure


def _part_with_tree(tree, params=None):
    return Part(
        name="angular-part",
        params=params or {},
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=tree)],
    )


# --------------------------------------------------------------------------
# Math of the raw field
# --------------------------------------------------------------------------


def test_field_angular_known_values():
    """sin(freq*theta + phase): zero on +x (theta=0, phase=0), amplitude on +y (freq=1)."""
    on_x = jnp.array([[1.0, 0.0, 0.0]])  # theta = 0
    on_y = jnp.array([[0.0, 1.0, 0.0]])  # theta = pi/2
    assert abs(float(shapes.field_angular(on_x, freq=1, amplitude=1.0)[0])) < 1e-6
    assert abs(float(shapes.field_angular(on_y, freq=1, amplitude=0.7)[0]) - 0.7) < 1e-6


def test_field_angular_default_amplitude_is_one():
    on_y = jnp.array([[0.0, 1.0, 0.0]])  # theta = pi/2 -> sin(pi/2) = 1
    assert abs(float(shapes.field_angular(on_y, freq=1)[0]) - 1.0) < 1e-6


def test_field_angular_bounded_by_amplitude():
    thetas = jnp.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
    p = jnp.stack([jnp.cos(thetas), jnp.sin(thetas), jnp.zeros_like(thetas)], axis=-1)
    vals = shapes.field_angular(p, freq=6, amplitude=0.3)
    assert float(jnp.max(jnp.abs(vals))) <= 0.3 + 1e-6


def test_field_angular_constant_along_radius():
    """Depends only on angle (and not on z): fixed theta -> same value at any radius."""
    theta = 0.9
    pts = jnp.array(
        [
            [r * math.cos(theta), r * math.sin(theta), z]
            for r, z in ((0.5, 0.0), (2.0, 3.0), (10.0, -1.0))
        ]
    )
    vals = shapes.field_angular(pts, freq=6, amplitude=1.0)
    assert jnp.allclose(vals, vals[0], atol=1e-6)


def test_field_angular_has_freq_lobes():
    """An integer freq is periodic in theta with period 2*pi/freq (freq identical lobes)."""
    freq, per_lobe = 6, 24
    thetas = jnp.linspace(0.0, 2.0 * math.pi, freq * per_lobe, endpoint=False)
    p = jnp.stack([jnp.cos(thetas), jnp.sin(thetas), jnp.zeros_like(thetas)], axis=-1)
    vals = shapes.field_angular(p, freq=freq, amplitude=1.0).reshape(freq, per_lobe)
    assert jnp.allclose(vals, vals[0][None, :], atol=1e-5)


def test_field_angular_integer_freq_is_seamless():
    """Integer freq is continuous across the atan2 branch cut at theta = +/- pi;
    a non-integer freq leaves a step there (documented constraint)."""
    eps = 1e-4
    below = jnp.array([[math.cos(math.pi - eps), math.sin(math.pi - eps), 0.0]])
    above = jnp.array([[math.cos(-math.pi + eps), math.sin(-math.pi + eps), 0.0]])
    seam_int = abs(
        float(shapes.field_angular(below, freq=6)[0])
        - float(shapes.field_angular(above, freq=6)[0])
    )
    seam_frac = abs(
        float(shapes.field_angular(below, freq=2.5)[0])
        - float(shapes.field_angular(above, freq=2.5)[0])
    )
    assert seam_int < 1e-2  # continuous
    assert seam_frac > 0.5  # discontinuous


# --------------------------------------------------------------------------
# Compile / eval path (as a displacement field)
# --------------------------------------------------------------------------


def test_field_angular_compiles_and_matches_direct():
    """displace(sphere, angular) compiles to sphere_sdf + field_angular."""
    child = sdf_primitive("sphere", r=4.0)
    field = field_primitive("angular", freq=6, amplitude=0.3, phase=0.2)
    tree = sdf_deform("displace", child, field=field)
    part = _part_with_tree(tree)
    sdf = make_sdf_closure(tree, part)

    p = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [-1.0, -1.0, 0.5],
            [3.0, 3.0, 3.0],
        ]
    )
    compiled = sdf(p, jnp.zeros((0,)))
    direct = shapes.sphere(p, 4.0) + shapes.field_angular(p, freq=6, amplitude=0.3, phase=0.2)
    assert jnp.allclose(compiled, direct, atol=1e-6)


def test_field_angular_scallops_a_cylinder_surface():
    """Displacing a cylinder by angular(freq=6) gives a 6-fold-symmetric surface:
    the zero-set radius repeats every 60 deg."""
    child = sdf_primitive("capped_cylinder", h=5.0, r=10.0)
    field = field_primitive("angular", freq=6, amplitude=1.5)
    tree = sdf_deform("displace", child, field=field)
    part = _part_with_tree(tree)
    sdf = make_sdf_closure(tree, part)

    freq, per_lobe = 6, 16
    thetas = jnp.linspace(0.0, 2.0 * math.pi, freq * per_lobe, endpoint=False)
    r_probe = 10.0
    p = jnp.stack(
        [r_probe * jnp.cos(thetas), r_probe * jnp.sin(thetas), jnp.zeros_like(thetas)], axis=-1
    )
    vals = sdf(p, jnp.zeros((0,))).reshape(freq, per_lobe)
    assert jnp.allclose(vals, vals[0][None, :], atol=1e-5)


# --------------------------------------------------------------------------
# Bounding-box inference
# --------------------------------------------------------------------------


def test_displace_angular_field_inflates_by_amplitude():
    child = sdf_primitive("sphere", r=4.0)
    field = field_primitive("angular", freq=6, amplitude=0.2)
    tree = sdf_deform("displace", child, field=field)
    part = _part_with_tree(tree)
    bbox = infer_sdf_bbox(tree, part)
    assert bbox == ((-4.2, -4.2, -4.2), (4.2, 4.2, 4.2))


def test_displace_angular_param_ref_amplitude_uses_worst_case():
    child = sdf_primitive("sphere", r=5.0)
    field = field_primitive("angular", freq=6, amplitude=make_param_ref("a"))
    tree = sdf_deform("displace", child, field=field)
    part = _part_with_tree(
        tree, params={"a": Param("a", 0.5, free=True, bounds=(-1.0, 2.0), unit="mm")}
    )
    bbox = infer_sdf_bbox(tree, part)  # bounds mode: max(|-1|, |2|) = 2
    assert bbox == ((-7.0, -7.0, -7.0), (7.0, 7.0, 7.0))


# --------------------------------------------------------------------------
# Axis contract: value convention + gradient guard (see _azimuth / _AXIS_EPSILON)
# --------------------------------------------------------------------------


def _angular_scalar(freq=6.0, amplitude=1.0, phase=0.0):
    """A point -> scalar field wrapper suitable for ``grad`` (takes p of shape (3,))."""

    def scalar(p):
        return shapes.field_angular(p[None, :], freq=freq, amplitude=amplitude, phase=phase)[0]

    return scalar


def test_field_angular_on_axis_value():
    """On the Z axis (p_xy = 0) the angle is 0 by convention, so the field is
    amplitude*sin(phase). Locks the atan2(0,0)=0 seam contract against refactors."""
    on_axis = jnp.array([[0.0, 0.0, 0.5], [0.0, 0.0, -2.0]])
    for phase in (0.0, 0.5, 1.3):
        vals = shapes.field_angular(on_axis, freq=6, amplitude=0.7, phase=phase)
        assert jnp.allclose(vals, 0.7 * math.sin(phase), atol=1e-6)


def test_field_angular_axis_gradient_is_finite():
    """The spatial gradient d(field)/dp is finite ON and NEAR the Z axis. The raw
    atan2 derivative is NaN at r=0 and ~1/r next to it (which would poison autodiff
    over any grid that samples the axis); the _azimuth epsilon-floor removes that."""
    g = jax.grad(_angular_scalar(freq=6.0, amplitude=1.0))
    for r in (0.0, 1e-4, 1e-2, 1.0):
        grad_val = g(jnp.array([r, 0.0, 0.0]))
        assert bool(jnp.all(jnp.isfinite(grad_val))), f"non-finite gradient at r={r}"


def test_field_angular_gradient_bounded_near_axis():
    """The floored gradient magnitude stays under the analytic cap
    amplitude*freq/(2*_AXIS_EPSILON) across a radial sweep through the axis."""
    freq, amplitude = 6.0, 1.0
    g = jax.grad(_angular_scalar(freq=freq, amplitude=amplitude))
    cap = amplitude * freq / (2.0 * _AXIS_EPSILON)
    ux, uy = math.cos(0.7), math.sin(0.7)  # a generic ray, so theta (hence cos) varies
    for r in jnp.linspace(0.0, 0.5, 200):
        rf = float(r)
        norm = float(jnp.linalg.norm(g(jnp.array([rf * ux, rf * uy, 0.0]))))
        assert norm <= cap + 1e-3, f"gradient norm {norm} exceeds cap {cap} at r={rf}"


def test_field_angular_gradient_matches_exact_away_from_axis():
    """For r >> _AXIS_EPSILON the floored gradient matches the exact analytic
    d(field)/dp = amplitude*freq*cos(freq*theta+phase) * [-y, x, 0]/r^2: the floor
    does not distort gradients where a real surface actually lives."""
    freq, amplitude, phase = 6.0, 1.0, 0.2
    x, y = 0.7, 0.4  # r ~ 0.8 >> _AXIS_EPSILON
    g = jax.grad(_angular_scalar(freq=freq, amplitude=amplitude, phase=phase))
    r2 = x * x + y * y
    coef = amplitude * freq * math.cos(freq * math.atan2(y, x) + phase)
    exact = jnp.array([coef * (-y) / r2, coef * x / r2, 0.0])
    assert jnp.allclose(g(jnp.array([x, y, 0.0])), exact, atol=1e-4)
