"""Free parameters flow into the compiled SDF via ``$ref`` leaves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    make_param_ref,
    sdf_primitive,
)
from software_defined_matter.sdf.compile import make_sdf_closure


def _build_sphere_part(radius=3.0, free=True):
    return Part(
        name="sphere",
        params={"r": Param("r", value=radius, free=free, bounds=(0.1, 10.0), unit="mm")},
        materials=[
            MaterialRegion(
                material_id=1, name="mat", sdf_tree=sdf_primitive("sphere", r=make_param_ref("r"))
            )
        ],
        metadata={},
    )


def test_ref_leaf_resolves_to_free_vec():
    part = _build_sphere_part(radius=3.0, free=True)
    sdf = make_sdf_closure(part.computed_envelope(), part)

    points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    # Evaluate with the built-in radius.
    d = sdf(points, jnp.array([3.0]))
    assert jnp.allclose(d, jnp.array([-3.0, 2.0, 0.0]), atol=1e-5)

    # Change the free vec; the SDF should react accordingly.
    d2 = sdf(points, jnp.array([2.0]))
    assert jnp.allclose(d2, jnp.array([-2.0, 3.0, 1.0]), atol=1e-5)


def test_fixed_params_do_not_appear_in_free_vec():
    part = _build_sphere_part(radius=4.0, free=False)
    sdf = make_sdf_closure(part.computed_envelope(), part)

    # Free-vector length should be 0.
    assert sdf.binding.n_free() == 0
    d = sdf(jnp.array([[0.0, 0.0, 0.0]]), jnp.zeros((0,)))
    assert jnp.allclose(d, jnp.array([-4.0]), atol=1e-5)


def test_grad_flows_through_ref():
    part = _build_sphere_part(radius=3.0, free=True)
    sdf = make_sdf_closure(part.computed_envelope(), part)
    point = jnp.array([[0.0, 0.0, 0.0]])

    def loss(free_vec):
        return sdf(point, free_vec)[0] ** 2

    g = jax.grad(loss)(jnp.array([3.0]))
    # d(loss)/dr at r=3 with d = -r -> loss = r^2 -> dloss/dr = 2r = 6
    assert jnp.allclose(g, jnp.array([6.0]), atol=1e-5)


def test_unknown_ref_raises():
    part = Part(
        name="x",
        params={"r": Param("r", 1.0, unit="mm")},
        materials=[
            MaterialRegion(
                material_id=1,
                name="mat",
                sdf_tree=sdf_primitive("sphere", r=make_param_ref("nonexistent")),
            )
        ],
    )
    sdf = make_sdf_closure(part.computed_envelope(), part)
    with pytest.raises(KeyError):
        sdf(jnp.zeros((1, 3)), jnp.zeros((0,)))
