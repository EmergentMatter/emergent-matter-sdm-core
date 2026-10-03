"""A fast rigid field must represent material, including away from classifiers."""

from __future__ import annotations

import copy
import json
from importlib.resources import files

import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.material_motion import compile_material_motion
from software_defined_matter.motion_bounds import infer_motion_bounds
from software_defined_matter.rigid_materials import RigidMaterialRefusalError, rigid_material_part
from tests.test_shader_body_motion import shader_runtime as _shader_runtime

shader_runtime = _shader_runtime


def sphere(radius=1, x=0):
    return sdf_transform("translate", sdf_primitive("sphere", r=radius), t=[x, 0, 0])


def part_with(regions, material):
    return Part(
        name="owned material",
        params={},
        materials=[MaterialRegion(1, "solid", material)],
        kinematics={
            "dofs": [
                {"name": "shift", "kind": "length", "unit": "mm", "range": [-10, 10], "default": 0}
            ],
            "bodies": [
                {
                    "name": f"body_{i}",
                    "region": r,
                    "motion": {
                        "ops": [
                            {
                                "kind": "translate",
                                "axis": [1, 0, 0],
                                "distance": {"type": "dof", "name": "shift"},
                            }
                        ]
                        if i == 0
                        else []
                    },
                }
                for i, r in enumerate(regions)
            ],
        },
    )


@pytest.mark.parametrize("shift", [0, 3, -4])
def test_single_owner_uses_material_not_classifier(shader_runtime, shift):
    fixture = files("software_defined_matter.schema.conformance.consumers") / "ownership"
    part = Part.from_dict(json.loads((fixture / "material_outside_classifier.sdm").read_text()))
    expected = json.loads((fixture / "expected.json").read_text())
    original = copy.deepcopy(part.to_dict())
    adapted = rigid_material_part(part)
    points = np.asarray(expected["rest_points"]) + [shift, 0, 0]
    emitted = shader_runtime.evaluate(emit_glsl(adapted), points, {"u_dof_0": shift})[:, 3]
    oracle = compile_material_motion(part)
    np.testing.assert_array_equal(emitted <= 0, oracle.contains(points, [shift]))
    np.testing.assert_allclose(emitted, expected["distances"], atol=1e-5)
    assert part.to_dict() == original


@pytest.mark.parametrize("shift", [0, 5, 10])
def test_separated_rest_owners_can_overlap_when_posed(shader_runtime, shift):
    regions = [sphere(2, -5), sphere(2, 5)]
    part = part_with(regions, sdf_op("union", regions))
    points = np.random.default_rng(22).uniform([-9, -2, -2], [9, 2, 2], (64, 3))
    adapted = rigid_material_part(part)
    emitted = shader_runtime.evaluate(emit_glsl(adapted), points, {"u_dof_0": shift})[:, 3]
    np.testing.assert_array_equal(
        emitted <= 0, compile_material_motion(part).contains(points, [shift])
    )


@pytest.mark.parametrize(
    "regions, material",
    [
        ([sphere(1, -1), sphere(1, 1)], sphere(5)),
        ([sphere(2), sphere(2)], sphere(2)),
        ([sphere(2, -2), sphere(2, 2)], sdf_op("union", [sphere(2, -2), sphere(2, 2)])),
    ],
)
def test_classification_disagreement_and_ties_require_queries(regions, material):
    with pytest.raises(RigidMaterialRefusalError):
        rigid_material_part(part_with(regions, material))


def test_separation_covers_design_ranges_not_current_values():
    regions = [sphere({"$ref": "radius"}, -5), sphere(2, 5)]
    part = part_with(regions, sdf_op("union", regions))
    part.params["radius"] = Param("radius", 1, bounds=[1, 10], unit="mm")
    with pytest.raises(RigidMaterialRefusalError, match="overlap or touch"):
        rigid_material_part(part)


def test_distinct_parameter_names_cannot_certify_matching_geometry():
    regions = [sphere({"type": "param", "name": "first"}, -5), sphere(2, 5)]
    material = copy.deepcopy(sdf_op("union", regions))
    material["children"][0]["child"]["params"]["r"]["name"] = "second"
    part = part_with(regions, material)
    part.params = {n: Param(n, 1, bounds=[1, 2], unit="mm") for n in ["first", "second"]}
    with pytest.raises(RigidMaterialRefusalError, match="differ from material"):
        rigid_material_part(part)


def test_float32_near_touching_boxes_are_not_certified():
    regions = [sphere(1, -1), sphere(1, 1 + 1e-8)]
    with pytest.raises(RigidMaterialRefusalError, match="overlap or touch"):
        rigid_material_part(part_with(regions, sdf_op("union", regions)))


def test_multiple_materials_are_a_hard_union_not_an_envelope(shader_runtime):
    part = part_with([sphere(1)], sphere(2, -3))
    part.materials.append(MaterialRegion(1, "other record", sphere(2, 3)))
    points = np.array([[-3, 0, 0], [0, 0, 0], [3, 0, 0]])
    emitted = shader_runtime.evaluate(emit_glsl(rigid_material_part(part)), points)[:, 3]
    np.testing.assert_array_equal(emitted <= 0, compile_material_motion(part).contains(points, [0]))
    np.testing.assert_allclose(emitted, [-2, 1, -2], atol=1e-5)


def test_query_bounds_include_material_outside_body_classifiers():
    part = part_with([sphere(1, -1), sphere(1, 1)], sphere(5))
    bounds = infer_motion_bounds(part, material_support=True)
    assert all(b.rest_bbox[0][0] <= -5 and b.rest_bbox[1][0] >= 5 for b in bounds.bodies)
    oracle = compile_material_motion(part)
    for shift in [-10, 0, 10]:
        points = np.random.default_rng(19).uniform(-16, 16, (1000, 3))
        occupied = points[np.asarray(oracle.contains(points, [shift]))]
        assert np.all(occupied >= bounds.bbox[0]) and np.all(occupied <= bounds.bbox[1])
