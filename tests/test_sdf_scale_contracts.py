"""Uniform-scale bounds must enclose offset children over the full parameter range."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from software_defined_matter import Param, Part, sdf_modifier, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.exactness import EXACT, LOWER_BOUND, infer_sdf_exactness


def _scaled_sphere(center):
    child = sdf_transform("translate", sdf_primitive("sphere", r=1), t=center)
    tree = sdf_transform("scale", child, s={"$ref": "scale"})
    part = Part(name="scale-range", params={"scale": Param("scale", 2, free=True, bounds=(1, 2))})
    return tree, part


def _sphere_union(d_other_x):
    scaled, part = _scaled_sphere((4, 0, 0))
    other = sdf_transform("translate", sdf_primitive("sphere", r=1), t=[d_other_x, 0, 0])
    return sdf_op("union", [scaled, other]), part


def test_overlap_at_small_scale_prevents_exact_union_and_shell_claims():
    tree, part = _sphere_union(4.5)
    assert infer_sdf_exactness(tree, part, mode="values").is_exact
    contract = infer_sdf_exactness(tree, part)
    assert contract.outside == EXACT
    assert contract.inside == LOWER_BOUND
    shell = sdf_modifier("onion", tree, thickness=0.1)
    assert infer_sdf_exactness(shell, part).outside == LOWER_BOUND

    fn = make_sdf_closure(tree, part)
    free = jnp.array([1.0])
    # Equal unit spheres meet in a circle of radius sqrt(1 - 0.25**2).
    # At the midpoint the min field underestimates depth to that exposed circle.
    d_true_depth = math.sqrt(1 - 0.25**2)
    d_field = float(fn(jnp.array([4.25, 0, 0]), free))
    assert d_field == pytest.approx(-0.75, abs=1e-6)
    assert abs(d_field) < d_true_depth
    assert float(fn(jnp.array([4.25, d_true_depth, 0]), free)) == pytest.approx(0, abs=1e-6)


def test_union_disjoint_across_the_full_scale_range_keeps_exact_contracts():
    tree, part = _sphere_union(0)
    assert infer_sdf_exactness(tree, part).is_exact
