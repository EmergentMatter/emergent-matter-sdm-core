"""Cone bounds must cover the angle ratio before they can certify CSG distance claims."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from software_defined_matter import Param, Part, sdf_modifier, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.exactness import (
    EXACT,
    LOWER_BOUND,
    DistanceAccuracyWarning,
    infer_sdf_exactness,
)
from software_defined_matter.sdf.validate import warn_distance_consumers


def _cone_pair(separation=3):
    cone = sdf_primitive("cone", c=[1, {"$ref": "cos"}], h=1)
    tree = sdf_op("union", [cone, sdf_transform("translate", cone, t=[separation, 0, 0])])
    part = Part(name="cone-range", params={"cos": Param("cos", 1.0, free=True, bounds=(0.5, 1))})
    return tree, part


def test_cones_that_overlap_within_the_angle_range_cannot_get_exact_union_interiors():
    tree, part = _cone_pair()
    assert infer_sdf_exactness(tree, part, mode="values").is_exact
    contract = infer_sdf_exactness(tree, part)
    assert contract.outside == EXACT
    assert contract.inside == LOWER_BOUND
    fn = make_sdf_closure(tree, part)
    free = jnp.array([0.5])
    # At this midpoint, the nearest exposed boundary is the cones' shared
    # intersection at z=-0.75, 0.1 away. The base caps are farther away.
    assert float(fn(jnp.array([1.5, 0, -0.85]), free)) == pytest.approx(
        -0.1 / math.sqrt(1.25), abs=2e-6
    )
    assert float(fn(jnp.array([1.5, 0, -0.75]), free)) == pytest.approx(0, abs=2e-6)


def test_shell_contracts_and_warnings_account_for_overlap_anywhere_in_the_angle_range():
    union, part = _cone_pair()
    shell = sdf_modifier("onion", union, thickness=0.02)
    assert infer_sdf_exactness(shell, part).outside == LOWER_BOUND
    with pytest.warns(DistanceAccuracyWarning, match="onion reads a side"):
        warn_distance_consumers(shell, part)


def test_cones_disjoint_over_the_full_angle_range_keep_exact_union_contracts():
    tree, part = _cone_pair(separation=5)
    assert infer_sdf_exactness(tree, part).is_exact
