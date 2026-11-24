"""Plane distances and their bounds survive extreme float32 equation scales."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import Part, sdf_primitive, sdf_transform
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.lipschitz import infer_sdf_max_rate
from software_defined_matter.sdf.sdf_shapes import plane

SCALES = [1e-35, 1e-31, 1e-22, 1e-20, 1e-18, 1.0, 1e20, 1e35]
# Dot products of order-one coordinates differ by a few float32 ulps.
ATOL = 2e-6


@pytest.mark.parametrize("scale", SCALES, ids=[str(s) for s in SCALES])
def test_plane_equation_scale_preserves_float32_values_and_spatial_gradients(scale):
    with jax.enable_x64(False):
        normal = jnp.array([1, -2, 2], dtype=jnp.float32) * scale
        h = jnp.float32(-0.75 * scale)
        points = jnp.array([[0, 0, 0], [1, 2, 3], [-2, 1, -1]], dtype=jnp.float32)
        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(plane), in_axes=(0, None, None)))(
            points, normal, h
        )
        expected = (np.asarray(points) @ np.array([1, -2, 2]) - 0.75) / 3
        np.testing.assert_allclose(values, expected, atol=ATOL, rtol=0)
        np.testing.assert_allclose(
            gradients, np.tile([1 / 3, -2 / 3, 2 / 3], (3, 1)), atol=ATOL, rtol=0
        )
        assert values.dtype == gradients.dtype == jnp.float32


@pytest.mark.parametrize("scale", SCALES, ids=[str(s) for s in SCALES])
def test_plane_parameter_gradients_match_the_normalized_equation_under_jit(scale):
    with jax.enable_x64(False):
        p = jnp.array([1, 2, 3], dtype=jnp.float32)
        n = jnp.array([1, -2, 2], dtype=jnp.float32) * scale
        h = jnp.float32(-0.75 * scale)
        dn, dh = jax.jit(jax.grad(plane, argnums=(1, 2)))(p, n, h)
        # Compare dimensionless gradients to avoid loose absolute tolerances
        # on derivatives proportional to 1/scale.
        direction = np.array([1, -2, 2]) / 3
        distance = (np.dot([1, 2, 3], [1, -2, 2]) - 0.75) / 3
        np.testing.assert_allclose(
            np.asarray(dn) * scale, (np.asarray(p) - distance * direction) / 3, atol=ATOL, rtol=0
        )
        assert float(dh * scale) == pytest.approx(1 / 3, abs=ATOL)


def test_small_plane_normal_cannot_turn_a_unit_rate_bound_into_an_overshoot():
    with jax.enable_x64(False):
        tree = sdf_transform(
            "scale_axis", sdf_primitive("plane", n=[1e-20, 0, 0], h=0), s=[1, 2, 3]
        )
        part = Part(name="plane-bound")
        assert infer_sdf_max_rate(tree, part) == pytest.approx(1, abs=1e-12)
        points = jnp.array([[1, 0, 0], [-1, 3, 2], [0, 0, 0], [2, -1, 4]], dtype=jnp.float32)
        values = np.asarray(jax.jit(make_sdf_closure(tree, part))(points))
        np.testing.assert_allclose(values, np.asarray(points[:, 0]), atol=ATOL, rtol=0)
        assert np.all(np.abs(values) <= np.abs(np.asarray(points[:, 0])) + ATOL)


def test_zero_plane_normal_retains_its_fallback_without_nan_gradients():
    with jax.enable_x64(False):
        p, n = jnp.ones(3, dtype=jnp.float32), jnp.zeros(3, dtype=jnp.float32)
        value, gradients = jax.jit(jax.value_and_grad(plane, argnums=(0, 1, 2)))(
            p, n, jnp.float32(1e-30)
        )
        assert float(value) == pytest.approx(1, abs=ATOL)
        assert all(np.isfinite(np.asarray(g)).all() for g in gradients)
