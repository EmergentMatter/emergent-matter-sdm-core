"""Plane certificates reject normals that can flush to zero in float32."""

from __future__ import annotations

import pytest

from software_defined_matter import Param, Part, sdf_primitive, sdf_transform
from software_defined_matter.sdf.exactness import EXACT, LOWER_BOUND, UNKNOWN, infer_sdf_exactness


@pytest.mark.parametrize("normal", [[0, 0, 0], [1e-40, 0, 0]], ids=["zero", "subnormal"])
def test_normals_that_can_flush_to_zero_do_not_receive_a_distance_certificate(normal):
    tree = sdf_primitive("plane", n=normal, h=0)
    assert infer_sdf_exactness(tree, Part(name="uncertified")).outside == UNKNOWN
    supported = sdf_primitive("plane", n=[1e-35, 0, 0], h=0)
    assert infer_sdf_exactness(supported, Part(name="small-normal")).outside == EXACT


def test_plane_normal_bounds_must_stay_above_the_float32_flush_range():
    tree = sdf_primitive("plane", n=[{"$ref": "nx"}, 0, 0], h=0)
    part = Part(name="normal-range", params={"nx": Param("nx", 1, free=True, bounds=(1e-40, 1))})
    assert infer_sdf_exactness(tree, part).outside == UNKNOWN
    assert infer_sdf_exactness(tree, part, mode="values").outside == EXACT


def test_small_plane_normal_preserves_the_axis_scale_lower_bound_contract():
    tree = sdf_transform("scale_axis", sdf_primitive("plane", n=[1e-20, 0, 0], h=0), s=[1, 2, 3])
    contract = infer_sdf_exactness(tree, Part(name="plane-bound"))
    assert contract.outside == contract.inside == LOWER_BOUND
