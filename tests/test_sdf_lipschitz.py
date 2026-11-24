"""``infer_sdf_max_rate`` must return a genuine UPPER bound.

The point of the number is that a caller may trust it. Voxel skipping in the
metrics path discards a cell when ``abs(d) > half_diagonal * rate``; if the
inferred rate is too small, cells that do contain surface get skipped and the
material silently disappears. So every test here measures the real gradient by
autodiff and asserts the inferred bound is at least that large.

Over-estimating is fine: it only costs efficiency. Under-estimating is the
bug this file exists to catch.
"""

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
    sdf_deform,
    sdf_op,
    sdf_primitive,
    sdf_raster_field,
    sdf_transform,
)
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.lipschitz import UNKNOWN, infer_sdf_max_rate
from software_defined_matter.sdf.sdf_ops import op_twist

SEED = 0
N_SAMPLES = 30_000


def _part(tree, params=None):
    return Part(
        name="t",
        params={p.name: p for p in (params or [])},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
    )


def _measured_rate(tree, half, params=None):
    """Largest |grad sdf| actually attained on a random cloud."""
    part = _part(tree, params)
    closure = make_sdf_closure(tree, part)
    free = jnp.asarray(part.param_vector())
    rng = np.random.default_rng(SEED)
    p = jnp.asarray(rng.uniform(-half, half, size=(N_SAMPLES, 3)))

    def scalar(x):
        return jnp.reshape(closure(x[None, :], free), ())

    g = np.asarray(jnp.linalg.norm(jax.vmap(jax.grad(scalar))(p), axis=-1))
    return float(np.nanmax(g[np.isfinite(g)]))


def _assert_bounds(tree, half, *, params=None, expect=None):
    """The domain handed to the inferrer is the same cube the rate is measured
    over: twist/bend stretch more the further out you evaluate, so a bound
    quoted for a smaller region would not cover these samples."""
    part = _part(tree, params)
    domain = ((-half, -half, -half), (half, half, half))
    inferred = infer_sdf_max_rate(tree, part, mode="values", domain=domain)
    measured = _measured_rate(tree, half, params)
    assert inferred >= measured - 1e-6, (
        f"inferred rate {inferred:.4f} is BELOW the measured {measured:.4f}: "
        f"a caller trusting it would skip cells that contain surface"
    )
    if expect is not None:
        assert inferred == pytest.approx(expect, rel=1e-9)
    return inferred, measured


# ---------------------------------------------------------------------------
# Primitives and rigid combinations: rate 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tree,half",
    [
        (sdf_primitive("sphere", r=3.0), 8.0),
        (sdf_primitive("box", b=[3.0, 2.0, 1.5]), 8.0),
        (sdf_primitive("ellipsoid", r=[8.0, 1.0, 1.0]), 10.0),
        (
            sdf_primitive("annular_sector", inner_r=0.2, outer_r=3.0, half_angle=0.5, height=1.0),
            5.0,
        ),
        (sdf_primitive("gyroid", period=4.0, min_thickness=0.5, n_periods=[3, 3, 3]), 5.0),
        (sdf_primitive("bellows", outer_r=5.0, inner_r=1.0, period=2.0, n_periods=3), 8.0),
        (
            sdf_primitive(
                "serpentine",
                amplitude=2.0,
                wavelength=2.0,
                beam_width=0.4,
                beam_height=1.0,
                n_periods=3,
            ),
            6.0,
        ),
    ],
)
def test_primitives_are_rate_one(tree, half):
    _assert_bounds(tree, half, expect=1.0)


def test_union_takes_the_worst_child():
    tree = sdf_op(
        "union", [sdf_primitive("sphere", r=3.0), sdf_primitive("box", b=[2.0, 2.0, 2.0])]
    )
    _assert_bounds(tree, 8.0, expect=1.0)


@pytest.mark.parametrize(
    "tf,kwargs",
    [
        ("translate", {"t": [2.0, -1.0, 0.5]}),
        ("rotate_z", {"angle": 0.7}),
        ("scale", {"s": 2.5}),
    ],
)
def test_rigid_and_uniform_scale_preserve_the_rate(tf, kwargs):
    """A uniform scale compiles to ``child(p/s) * s``; the two factors cancel
    in the derivative, so the rate is unchanged."""
    tree = sdf_transform(tf, sdf_primitive("sphere", r=3.0), **kwargs)
    _assert_bounds(tree, 12.0, expect=1.0)


# ---------------------------------------------------------------------------
# Deforms: rate above 1, and the bound must cover it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0.0, 0.05, 0.1, 0.2])
def test_twist_bound_covers_the_measured_rate(k):
    tree = {
        "type": "deform",
        "deform": "twist",
        "child": sdf_primitive("box", b=[3.0, 8.0, 3.0]),
        "params": {"k": k},
    }
    _inferred, measured = _assert_bounds(tree, 10.0)
    assert measured > 1.02 or k == 0.0, "twist should stretch space for k > 0"


@pytest.mark.parametrize("k", [0.05, 0.2])
def test_bend_bound_covers_the_measured_rate(k):
    tree = {
        "type": "deform",
        "deform": "bend",
        "child": sdf_primitive("box", b=[6.0, 3.0, 3.0]),
        "params": {"k": k},
    }
    _assert_bounds(tree, 10.0)


@pytest.mark.parametrize("amplitude,freq", [(0.2, 0.1), (0.5, 0.3), (1.0, 0.5)])
def test_displace_bound_covers_the_measured_rate(amplitude, freq):
    """``sdf(p) + field(p)``: the gradients add, so the bounds add."""
    tree = {
        "type": "deform",
        "deform": "displace",
        "child": sdf_primitive("sphere", r=6.0),
        "field": {
            "type": "field",
            "kind": "sin_xyz",
            "params": {"freq": [freq, freq, freq], "amplitude": amplitude},
        },
    }
    inferred, _measured = _assert_bounds(tree, 9.0)
    expected = 1.0 + amplitude * 2.0 * math.pi * math.sqrt(3.0) * freq
    assert inferred == pytest.approx(expected, rel=1e-9)


def test_twist_bound_grows_with_the_domain():
    """The shear is proportional to the query's distance from the axis, so a
    wider evaluation domain must produce a larger bound."""
    tree = {
        "type": "deform",
        "deform": "twist",
        "child": sdf_primitive("box", b=[2.0, 4.0, 2.0]),
        "params": {"k": 0.15},
    }

    def rate(half):
        return infer_sdf_max_rate(
            tree, _part(tree), mode="values", domain=((-half, -half, -half), (half, half, half))
        )

    assert rate(2.0) < rate(6.0) < rate(12.0)


# ---------------------------------------------------------------------------
# Unknown must be infinite, never an optimistic guess
# ---------------------------------------------------------------------------


def test_angular_displacement_field_is_unknown():
    """The azimuth's gradient is 1/r, unbounded at the Z axis."""
    tree = {
        "type": "deform",
        "deform": "displace",
        "child": sdf_primitive("sphere", r=6.0),
        "field": {"type": "field", "kind": "angular", "params": {"freq": 2.0, "amplitude": 1.0}},
    }
    assert infer_sdf_max_rate(tree, _part(tree)) == UNKNOWN


def test_unrecognised_node_is_unknown():
    tree = {"type": "loft", "children": [], "params": {"z": [0.0, 1.0]}}
    assert infer_sdf_max_rate(tree, _part(tree)) == UNKNOWN


def test_unknown_propagates_through_a_union():
    """One un-analysable branch must poison the whole tree: taking the max
    with infinity does that automatically."""
    tree = sdf_op(
        "union",
        [
            sdf_primitive("sphere", r=3.0),
            {"type": "loft", "children": [], "params": {"z": [0.0, 1.0]}},
        ],
    )
    assert infer_sdf_max_rate(tree, _part(tree)) == UNKNOWN


def test_bounds_mode_is_at_least_as_loose_as_values_mode():
    """``mode='bounds'`` covers every point a bounded optimiser may visit, so
    it can never report a smaller rate than the current values do."""
    tree = {
        "type": "deform",
        "deform": "twist",
        "child": sdf_primitive("box", b=[make_param_ref("w"), 4.0, 2.0]),
        "params": {"k": 0.2},
    }
    part = _part(tree, [Param("w", 2.0, free=True, bounds=(1.0, 9.0), unit="mm")])
    assert infer_sdf_max_rate(tree, part, mode="bounds") >= infer_sdf_max_rate(
        tree, part, mode="values"
    )


@pytest.mark.parametrize("deform", ["twist", "bend"])
def test_twist_and_bend_are_unknown_without_a_domain(deform):
    """Their stretch grows without limit away from the axis, so quoting a
    number that only holds near the origin would be worse than admitting
    ignorance."""
    tree = {
        "type": "deform",
        "deform": deform,
        "child": sdf_primitive("box", b=[3.0, 3.0, 3.0]),
        "params": {"k": 0.2},
    }
    assert infer_sdf_max_rate(tree, _part(tree)) == UNKNOWN


HELIX = {"major_r": 7.0, "pitch": 4.0, "r": 1.2, "n_turns": 3.0, "phase": 0.0, "handedness": 1.0}


def test_a_helix_reports_unknown():
    """The helix field jumps across its own axis, so no finite rate holds on a
    domain that contains the axis, and a unit rate would be a guess."""
    helix = sdf_primitive("helix", **HELIX)
    closure = make_sdf_closure(helix, _part(helix))
    free = jnp.asarray(_part(helix).param_vector())
    pts = jnp.asarray([[-5e-4, 0.0, 0.0], [5e-4, 0.0, 0.0]], dtype=jnp.float32)
    values = np.asarray(closure(pts, free))
    assert abs(values[1] - values[0]) / 1e-3 > 10.0, "the axis jump this test documents is gone"
    assert infer_sdf_max_rate(helix, _part(helix)) == UNKNOWN


def test_a_pyramid_reports_a_unit_rate():
    """The pyramid measures to its base square as well as its slanted faces, so
    its value no longer jumps across the base plane (#30) and a unit rate holds."""
    pyramid = sdf_primitive("pyramid", h=1.5)
    closure = make_sdf_closure(pyramid, _part(pyramid))
    free = jnp.asarray(_part(pyramid).param_vector())
    pts = jnp.asarray([[0.0, -1e-4, 0.0], [0.0, 1e-4, 0.0]], dtype=jnp.float32)
    values = np.asarray(closure(pts, free))
    assert abs(values[1] - values[0]) / 2e-4 <= 1.02
    _assert_bounds(pyramid, 2.5, expect=1.0)


def test_a_raster_reports_unknown_whatever_its_samples():
    """A sampled field's rate follows its samples. These are three times a
    distance, so the field changes more than twice as fast as a distance,
    which the probe below checks."""
    z, y, x = np.mgrid[-2:3, -2:3, -2:3]
    raster = sdf_raster_field([-2.0] * 3, 1.0, 3.0 * (np.sqrt(x * x + y * y + z * z) - 1.2))
    assert _measured_rate(raster, 1.5) > 2.0
    assert infer_sdf_max_rate(raster, _part(raster)) == UNKNOWN


def test_a_tree_holding_a_primitive_without_a_unit_rate_reports_unknown():
    tree = sdf_op("union", [sdf_primitive("sphere", r=2.0), sdf_primitive("helix", **HELIX)])
    assert infer_sdf_max_rate(tree, _part(tree)) == UNKNOWN


# ---------------------------------------------------------------------------
# Hermite twists: bounds hold inside and outside the ramp, over live params,
# and through the query domains that translation and nesting hand down
# ---------------------------------------------------------------------------


def _deform_tree(name, params, child=None):
    return {
        "type": "deform",
        "deform": name,
        "params": params,
        "child": sdf_primitive("box", b=[0.7, 1.2, 0.4]) if child is None else child,
    }


def _gradient_norms(tree, part, points):
    """|grad sdf| at each of ``points``, by autodiff."""
    closure = make_sdf_closure(tree, part)
    grad = jax.vmap(jax.grad(lambda p: jnp.reshape(closure(p[None, :]), ())))
    return np.linalg.norm(np.asarray(grad(jnp.asarray(points))), axis=1)


def test_axial_shear_exceeds_the_old_square_root_bound():
    point = jnp.array([1.0, 0, 0])
    jac = np.asarray(jax.jacfwd(lambda p: op_twist(lambda q: q, p, 1.0))(point))
    normal = np.linalg.svd(jac)[0][:, 0]
    tree = _deform_tree("twist", {"k": 1.0}, sdf_primitive("plane", n=normal.tolist(), h=0.0))
    part = _part(tree)
    measured = _gradient_norms(tree, part, point[None, :])[0]
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
    tree = _deform_tree(name, params)
    part = _part(tree)
    points = np.random.default_rng(208).uniform(-3, 3, (500, 3))
    points = np.concatenate([points, [[r, 0, 0.2] for r in [0.4999, 0.5001, 1.25, 1.9999, 2.0001]]])
    bound = infer_sdf_max_rate(tree, part, domain=((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0)))
    assert math.isfinite(bound)
    measured = _gradient_norms(tree, part, points)
    assert np.isfinite(measured).all()
    assert np.max(measured) <= bound + 2e-5  # Float32 gradient roundoff.


def test_bounds_cover_live_endpoint_angles_and_ramp_widths():
    tree = _deform_tree(
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
            concrete = _deform_tree(
                "twist_radial", {"r0": 0.0, "r1": width, "angle_inner": 0.0, "angle_outer": angle}
            )
            assert infer_sdf_max_rate(concrete, _part(concrete), domain=domain) <= bounds


@pytest.mark.parametrize("width", [0.0, -1.0])
def test_unproved_ramp_width_disables_skipping(width):
    tree = _deform_tree(
        "twist_radial", {"r0": 0.0, "r1": width, "angle_inner": 0.0, "angle_outer": 1.0}
    )
    assert infer_sdf_max_rate(tree, _part(tree), domain=((-2.0,) * 3, (2.0,) * 3)) == UNKNOWN


def test_missing_domain_and_zero_linear_axis_disable_skipping():
    tree = _deform_tree(
        "twist_linear", {"u0": 0.0, "u1": 1.0, "angle_0": 0.0, "angle_1": 1.0, "axis": [0.0, 0.0]}
    )
    assert infer_sdf_max_rate(tree, _part(tree)) == UNKNOWN
    assert infer_sdf_max_rate(tree, _part(tree), domain=((-2.0,) * 3, (2.0,) * 3)) == UNKNOWN


def test_translated_and_nested_twists_use_child_query_domains():
    inner = _deform_tree("twist", {"k": 2.0})
    nested = _deform_tree(
        "twist_radial", {"r0": 0.5, "r1": 2.0, "angle_inner": 0.0, "angle_outer": 2.0}, inner
    )
    tree = sdf_transform("translate", nested, t=[-10.0, 0, 0])
    part = _part(tree)
    points = np.random.default_rng(208).uniform(-1, 1, (500, 3))
    bound = infer_sdf_max_rate(tree, part, domain=((-1.0,) * 3, (1.0,) * 3))
    assert math.isfinite(bound)
    assert max(_gradient_norms(tree, part, points)) <= bound


def test_translation_does_not_hide_large_local_shear():
    local = jnp.array([10.0, 0, 0])
    jac = np.asarray(jax.jacfwd(lambda p: op_twist(lambda q: q, p, 1.0))(local))
    normal = np.linalg.svd(jac)[0][:, 0]
    tree = sdf_transform(
        "translate",
        _deform_tree("twist", {"k": 1.0}, sdf_primitive("plane", n=normal.tolist(), h=0.0)),
        t=[-10.0, 0, 0],
    )
    part = _part(tree)
    measured = _gradient_norms(tree, part, [[0.0, 0, 0]])[0]
    assert measured > 10
    assert infer_sdf_max_rate(tree, part, domain=((0.0,) * 3, (0.0,) * 3)) >= measured


def test_unhandled_transform_cannot_reuse_the_parent_sampling_box():
    tree = sdf_transform("rotate_z", _deform_tree("twist", {"k": 1.0}), angle=1.0)
    assert infer_sdf_max_rate(tree, _part(tree), domain=((-1.0,) * 3, (1.0,) * 3)) == UNKNOWN


@pytest.mark.parametrize("domain", [((1.0,) * 3, (-1.0,) * 3), ((0.0,) * 3, (math.inf, 1.0, 1.0))])
def test_invalid_sampling_domains_are_rejected(domain):
    tree = _deform_tree("twist", {"k": 1.0})
    with pytest.raises(ValueError, match="finite ordered 3-D box"):
        infer_sdf_max_rate(tree, _part(tree), domain=domain)


# ---------------------------------------------------------------------------
# Axis warps: shear_linear, scale_axis, taper_linear
# ---------------------------------------------------------------------------


def test_shear_linear_bound_covers_the_measured_gradient():
    part = Part(name="live", params={"h": Param("h", 3.0, free=True, bounds=(1.0, 6.0))})
    node = sdf_deform(
        "shear_linear", sdf_primitive("sphere", r=1.0),
        u0=-1.0, u1=1.0, dz_0=0.0, dz_1={"$ref": "h"},
    )  # fmt: skip
    bound = infer_sdf_max_rate(node, part)
    k = 6.0 / 2.0  # steepest slope: the top of h over the ramp width
    assert bound == pytest.approx((k + math.sqrt(k * k + 4.0)) / 2.0)

    steep = Part(name="steep", params={"h": Param("h", 6.0, free=True, bounds=(1.0, 6.0))})
    f = make_sdf_closure(node, steep)
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.uniform([-1.0, -1.5, -2.0], [1.0, 1.5, 8.0], size=(4000, 3)))
    step = 1e-3
    grads = [
        (np.asarray(f(p + step * e)) - np.asarray(f(p - step * e))) / (2 * step) for e in jnp.eye(3)
    ]
    measured = float(np.max(np.linalg.norm(np.stack(grads, -1), axis=-1)))
    assert measured <= bound * 1.01
    assert measured > 1.5  # the bound is not vacuous: a shear really stretches


def test_shear_linear_ramp_with_no_width_has_no_bound():
    node = sdf_deform(
        "shear_linear", sdf_primitive("sphere", r=1.0), u0=1.0, u1=1.0, dz_0=0, dz_1=1
    )
    assert infer_sdf_max_rate(node, Part(name="t")) == UNKNOWN


def test_scale_axis_keeps_the_child_rate_across_a_live_factor():
    part = Part(name="live", params={"sx": Param("sx", 2.0, free=True, bounds=(1.0, 3.0))})
    node = sdf_transform("scale_axis", sdf_primitive("sphere", r=1.0), s=[{"$ref": "sx"}, 1.0, 1.0])
    assert infer_sdf_max_rate(node, part) == pytest.approx(1.0)  # min(s)/s_i <= 1


def test_scale_axis_factor_that_can_reach_zero_has_no_bound():
    part = Part(name="t", params={"sx": Param("sx", 1.0, free=True, bounds=(0.0, 2.0))})
    node = sdf_transform("scale_axis", sdf_primitive("sphere", r=1.0), s=[{"$ref": "sx"}, 1.0, 1.0])
    assert infer_sdf_max_rate(node, part) == UNKNOWN


def test_taper_rate_needs_a_domain_and_then_covers_the_measured_gradient():
    part = Part(name="t")
    node = sdf_deform(
        "taper_linear", sdf_primitive("capped_cylinder", h=1.0, r=1.0),
        z0=-1.0, z1=1.0, s_0=1.0, s_1=0.5,
    )  # fmt: skip
    assert infer_sdf_max_rate(node, part) == UNKNOWN
    domain = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
    bound = infer_sdf_max_rate(node, part, domain=domain)
    assert math.isfinite(bound)
    f = make_sdf_closure(node, part)
    rng = np.random.default_rng(2)
    p = jnp.asarray(rng.uniform(-1.4, 1.4, size=(4000, 3)))
    step = 1e-3
    grads = [
        (np.asarray(f(p + step * e)) - np.asarray(f(p - step * e))) / (2 * step) for e in jnp.eye(3)
    ]
    assert float(np.max(np.linalg.norm(np.stack(grads, -1), axis=-1))) <= bound * 1.01
