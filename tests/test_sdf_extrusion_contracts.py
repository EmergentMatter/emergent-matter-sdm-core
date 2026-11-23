"""Extrusion distance claims require a justified profile and positive finite height bounds."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    sdf_2d_to_3d,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.exactness import (
    EXACT,
    LOWER_BOUND,
    UNKNOWN,
    infer_sdf_exactness,
)


def _profile(kind="circle"):
    circle = sdf_primitive("circle_2d", r=1)
    if kind in ("union", "intersect"):
        return sdf_op(kind, [sdf_transform("translate", circle, t=[x, 0]) for x in (-0.5, 0.5)])
    if kind == "scaled":
        return sdf_transform("scale_axis", circle, s=[2, 1])
    if kind == "unknown":
        return sdf_primitive("rounded_box_2d", b=[1, 1], r=2)
    return circle


def _extrusion(profile="circle", bounds=(1, 3)):
    tree = sdf_2d_to_3d("extrusion", _profile(profile), h={"$ref": "h"})
    part = Part(
        name="extrusion",
        params={"h": Param("h", 2, free=True, bounds=bounds, unit="mm")},
        materials=[MaterialRegion(1, "body", tree)],
    )
    return tree, part


@pytest.mark.parametrize(
    "profile, outside, inside",
    [
        ("circle", EXACT, EXACT),
        ("union", EXACT, LOWER_BOUND),
        ("intersect", LOWER_BOUND, EXACT),
        ("scaled", LOWER_BOUND, LOWER_BOUND),
        ("unknown", UNKNOWN, UNKNOWN),
    ],
    ids=["exact", "exact-exterior", "exact-interior", "bound", "unknown"],
)
def test_extrusion_preserves_only_justified_child_sides(profile, outside, inside):
    tree, part = _extrusion(profile)
    contract = infer_sdf_exactness(tree, part)
    assert (contract.outside, contract.inside) == (outside, inside)
    compiled = make_sdf_closure(tree, part)
    # Free vectors are unrestricted at runtime, so their compiled contract
    # must not reuse a conditional proof over declared parameter bounds.
    assert compiled.distance_exactness.outside == compiled.distance_exactness.inside == UNKNOWN
    tree["params"]["h"] = 2
    fixed = Part(name="fixed", materials=[MaterialRegion(1, "body", tree)])
    fixed_contract = make_sdf_closure(tree, fixed).distance_exactness
    assert (fixed_contract.outside, fixed_contract.inside) == (outside, inside)


@pytest.mark.parametrize(
    "bounds",
    [(-1, 3), (0, 3), (-3, -1), (0, 0), (1, math.inf)],
    ids=["crossing-zero", "touching-zero", "negative", "zero", "unbounded"],
)
def test_invalid_height_ranges_do_not_inherit_exactness(bounds):
    tree, part = _extrusion(bounds=bounds)
    contract = infer_sdf_exactness(tree, part)
    assert contract.outside == contract.inside == UNKNOWN
    assert infer_sdf_exactness(tree, part, mode="values").is_exact


@pytest.mark.parametrize(
    "height", [0, -1, math.inf, math.nan], ids=["zero", "negative", "inf", "nan"]
)
def test_invalid_literal_heights_are_unknown(height):
    tree = sdf_2d_to_3d("extrusion", _profile(), h=height)
    contract = infer_sdf_exactness(tree, Part(name="invalid-height"))
    assert contract.outside == contract.inside == UNKNOWN


def test_missing_height_or_bounds_cannot_establish_an_extrusion_contract():
    tree, part = _extrusion(bounds=None)
    assert not infer_sdf_exactness(tree, part).is_exact
    del tree["params"]["h"]
    assert not infer_sdf_exactness(tree, part).is_exact
    tree["method"] = "extrude"
    tree["params"]["h"] = 2
    assert not infer_sdf_exactness(tree, part).is_exact


def test_exact_extrusion_matches_cylinder_distances_at_height_endpoints_under_jit():
    tree, part = _extrusion()
    fn = jax.jit(make_sdf_closure(tree, part))
    for d_h in (1, 2, 3):
        points = jnp.array([[0.5, 0, 0], [0, 0, d_h + 0.5], [1.3, 0, d_h + 0.4]], dtype=jnp.float32)
        # Side, cap, and corner: the corner is a 3-4-5 distance triangle.
        assert fn(points, jnp.array([d_h], dtype=jnp.float32)).tolist() == pytest.approx(
            [-0.5, 0.5, 0.5], abs=1e-6
        )
