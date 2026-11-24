"""Collapsed segments retain finite float32 design gradients under JAX JIT."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter.sdf import sdf_shapes
from software_defined_matter.sdf._helpers import _length


@pytest.mark.parametrize("dim", [2, 3], ids=["segment_2d", "capsule"])
def test_coincident_endpoints_have_finite_jitted_float32_parameter_gradients(dim):
    endpoint = jnp.full(dim, 0.5, dtype=jnp.float32)
    points = jnp.stack([endpoint, endpoint + 1.0, endpoint - 0.25])
    radius = jnp.float32(0.2)

    def value(p, a, b):
        if dim == 3:
            return sdf_shapes.capsule(p, a, b, radius)
        return sdf_shapes.segment_2d(p, a, b)

    evaluate = jax.jit(jax.vmap(jax.value_and_grad(value, argnums=(1, 2)), in_axes=(0, None, None)))
    values, (grad_a, grad_b) = evaluate(points, endpoint, endpoint)
    assert values.dtype == grad_a.dtype == grad_b.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(grad_a)))
    assert bool(jnp.all(jnp.isfinite(grad_b)))
    expected = _length(points - endpoint) - (radius if dim == 3 else 0.0)
    np.testing.assert_allclose(values, expected, atol=2e-7, rtol=0.0)

    # Moving both endpoints together is smooth away from the query. Their
    # individual derivatives at coincidence need not be unique; the sum must
    # agree with translating the collapsed point (or sphere).
    reference = jax.vmap(jax.grad(lambda a, p: _length(p - a)), in_axes=(None, 0))(endpoint, points)
    np.testing.assert_allclose(grad_a + grad_b, reference, atol=2e-7, rtol=0.0)


@pytest.mark.parametrize("dim", [2, 3], ids=["segment_2d", "capsule"])
def test_separated_endpoint_gradients_match_finite_differences(dim):
    p = jnp.array([0.25, 1.0] + ([0.3] if dim == 3 else []), dtype=jnp.float32)
    endpoints = jnp.stack([jnp.zeros(dim), jnp.eye(dim)[0]])

    def value(ab):
        if dim == 3:
            return sdf_shapes.capsule(p, ab[0], ab[1], 0.2)
        return sdf_shapes.segment_2d(p, ab[0], ab[1])

    gradient = jax.jit(jax.grad(value))(endpoints)
    step = 1e-3
    offsets = step * jnp.eye(2 * dim).reshape(2 * dim, 2, dim)
    differences = (jax.vmap(value)(endpoints + offsets) - jax.vmap(value)(endpoints - offsets)) / (
        2 * step
    )
    # Float32 subtraction error is amplified by the finite-difference step.
    np.testing.assert_allclose(gradient.reshape(-1), differences, atol=1e-4, rtol=1e-3)
