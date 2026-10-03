"""Twist sampling rates bound actual gradients, including near ramp boundaries."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    make_param_ref,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.lipschitz import UNKNOWN, infer_sdf_max_rate
from software_defined_matter.sdf.sdf_ops import op_twist, op_twist_linear, op_twist_radial


def _tree(name, params, child=None):
    return {
        "type": "deform",
        "deform": name,
        "params": params,
        "child": sdf_primitive("box", b=[0.7, 1.2, 0.4]) if child is None else child,
    }


def _part(tree, params=()):
    return Part(
        name="sampling",
        materials=[MaterialRegion(1, "solid", tree)],
        params={p.name: p for p in params},
    )


def _measured(tree, part, points):
    closure = make_sdf_closure(tree, part)
    grad = jax.vmap(jax.grad(lambda p: jnp.reshape(closure(p[None, :]), ())))
    return np.linalg.norm(np.asarray(grad(jnp.asarray(points))), axis=1)


def test_axial_shear_exceeds_the_old_square_root_bound():
    point = jnp.array([1.0, 0, 0])
    jac = np.asarray(jax.jacfwd(lambda p: op_twist(lambda q: q, p, 1.0))(point))
    normal = np.linalg.svd(jac)[0][:, 0]
    tree = _tree("twist", {"k": 1.0}, sdf_primitive("plane", n=normal.tolist(), h=0.0))
    part = _part(tree)
    measured = _measured(tree, part, point[None, :])[0]
    assert measured > math.sqrt(2) + 0.1
    bound = infer_sdf_max_rate(tree, part, domain=((1.0, 0, 0), (1.0, 0, 0)))
    assert measured <= bound


@pytest.mark.parametrize("name", ["twist_radial", "twist_linear"])
@pytest.mark.parametrize("angle", [-2 * math.pi, 0.0, 2 * math.pi])
def test_hermite_twists_bound_gradients_inside_and_outside_the_ramp(name, angle):
    radial = name == "twist_radial"
    params = (
        {"r0": 0.5, "r1": 2.0, "angle_inner": -0.3, "angle_outer": angle}
        if radial
        else {"u0": -0.5, "u1": 2.0, "angle_0": -0.3, "angle_1": angle, "axis": [2.0, 1.0]}
    )
    tree = _tree(name, params)
    part = _part(tree)
    points = np.random.default_rng(208).uniform(-3, 3, (500, 3))
    points = np.concatenate([points, [[r, 0, 0.2] for r in [0.4999, 0.5001, 1.25, 1.9999, 2.0001]]])
    bound = infer_sdf_max_rate(tree, part, domain=((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0)))
    assert math.isfinite(bound)
    measured = _measured(tree, part, points)
    assert np.isfinite(measured).all()
    assert np.max(measured) <= bound + 2e-5  # Float32 gradient roundoff.


def test_bounds_cover_live_endpoint_angles_and_ramp_widths():
    tree = _tree(
        "twist_radial",
        {
            "r0": 0.0,
            "r1": make_param_ref("width"),
            "angle_inner": 0.0,
            "angle_outer": make_param_ref("angle"),
        },
    )
    part = _part(
        tree,
        [
            Param("width", 2.0, bounds=(1.0, 3.0), free=True),
            Param("angle", 0.1, bounds=(-6.0, 6.0), free=True),
        ],
    )
    domain = ((-3.0, -3.0, -1.0), (3.0, 3.0, 1.0))
    bounds = infer_sdf_max_rate(tree, part, domain=domain)
    assert bounds > infer_sdf_max_rate(tree, part, domain=domain, mode="values")
    for width in [1.0, 2.0, 3.0]:
        for angle in [-6.0, 0.0, 6.0]:
            concrete = _tree(
                "twist_radial", {"r0": 0.0, "r1": width, "angle_inner": 0.0, "angle_outer": angle}
            )
            assert infer_sdf_max_rate(concrete, _part(concrete), domain=domain) <= bounds


@pytest.mark.parametrize("width", [0.0, -1.0])
def test_unproved_ramp_width_disables_skipping(width):
    tree = _tree("twist_radial", {"r0": 0.0, "r1": width, "angle_inner": 0.0, "angle_outer": 1.0})
    assert infer_sdf_max_rate(tree, _part(tree), domain=((-2.0,) * 3, (2.0,) * 3)) == UNKNOWN


def test_missing_domain_and_zero_linear_axis_disable_skipping():
    tree = _tree(
        "twist_linear", {"u0": 0.0, "u1": 1.0, "angle_0": 0.0, "angle_1": 1.0, "axis": [0.0, 0.0]}
    )
    assert infer_sdf_max_rate(tree, _part(tree)) == UNKNOWN
    assert infer_sdf_max_rate(tree, _part(tree), domain=((-2.0,) * 3, (2.0,) * 3)) == UNKNOWN


def test_translated_and_nested_twists_use_child_query_domains():
    inner = _tree("twist", {"k": 2.0})
    nested = _tree(
        "twist_radial", {"r0": 0.5, "r1": 2.0, "angle_inner": 0.0, "angle_outer": 2.0}, inner
    )
    tree = sdf_transform("translate", nested, t=[-10.0, 0, 0])
    part = _part(tree)
    points = np.random.default_rng(208).uniform(-1, 1, (500, 3))
    bound = infer_sdf_max_rate(tree, part, domain=((-1.0,) * 3, (1.0,) * 3))
    assert math.isfinite(bound)
    assert max(_measured(tree, part, points)) <= bound


def test_query_signs_remain_distinct_for_existing_radial_and_linear_operators():
    p = jnp.array([3.0, 0, 0])
    radial = op_twist_radial(lambda q: q, p, 0.0, 1.0, 0.0, math.pi / 2)
    linear = op_twist_linear(lambda q: q, p, [1.0, 0], 0.0, 1.0, 0.0, math.pi / 2)
    np.testing.assert_allclose(radial, [0, -3, 0], atol=2e-6)
    np.testing.assert_allclose(linear, [0, 3, 0], atol=2e-6)


def test_translation_does_not_hide_large_local_shear():
    local = jnp.array([10.0, 0, 0])
    jac = np.asarray(jax.jacfwd(lambda p: op_twist(lambda q: q, p, 1.0))(local))
    normal = np.linalg.svd(jac)[0][:, 0]
    tree = sdf_transform(
        "translate",
        _tree("twist", {"k": 1.0}, sdf_primitive("plane", n=normal.tolist(), h=0.0)),
        t=[-10.0, 0, 0],
    )
    part = _part(tree)
    measured = _measured(tree, part, [[0.0, 0, 0]])[0]
    assert measured > 10
    assert infer_sdf_max_rate(tree, part, domain=((0.0,) * 3, (0.0,) * 3)) >= measured


def test_unhandled_transform_cannot_reuse_the_parent_sampling_box():
    tree = sdf_transform("rotate_z", _tree("twist", {"k": 1.0}), angle=1.0)
    assert infer_sdf_max_rate(tree, _part(tree), domain=((-1.0,) * 3, (1.0,) * 3)) == UNKNOWN


@pytest.mark.parametrize("domain", [((1.0,) * 3, (-1.0,) * 3), ((0.0,) * 3, (math.inf, 1.0, 1.0))])
def test_invalid_sampling_domains_are_rejected(domain):
    tree = _tree("twist", {"k": 1.0})
    with pytest.raises(ValueError, match="finite ordered 3-D box"):
        infer_sdf_max_rate(tree, _part(tree), domain=domain)
