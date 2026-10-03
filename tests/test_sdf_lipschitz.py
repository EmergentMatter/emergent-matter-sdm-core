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
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.lipschitz import UNKNOWN, infer_sdf_max_rate

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
