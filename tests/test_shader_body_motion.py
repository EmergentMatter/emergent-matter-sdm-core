"""Shader poses and live controls agree with the rigid-body reference evaluator."""

from __future__ import annotations

import copy
import json
import math
import os

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
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.glsl.emit import _GLSLEmitter
from software_defined_matter.kinematics import compile_kinematics
from tests._glsl_runtime import ShaderRuntime


@pytest.fixture(scope="module")
def shader_runtime():
    try:
        runtime = ShaderRuntime()
    except (OSError, AttributeError, RuntimeError) as exc:
        if os.environ.get("SDM_REQUIRE_GLSL") == "1":
            pytest.fail(f"Required software shader runtime unavailable: {exc}")
        pytest.skip(
            f"Install system EGL/GLES and Mesa software drivers to run shader parity: {exc}"
        )
    yield runtime
    runtime.close()


def _read(name):
    return {"type": "dof", "name": name}


def _part(unit="rad"):
    ball = sdf_primitive("sphere", r=make_param_ref("q"))
    ball["name"] = "ball"
    rest = sdf_transform("translate", ball, t=[2, 3, 0])
    return Part(
        name="moving ball",
        params={"q": Param("q", 1, unit="mm")},
        materials=[MaterialRegion(material_id=1, name="solid", sdf_tree=rest)],
        metadata={"bbox": [[-30, -30, -30], [30, 30, 30]]},
        kinematics={
            "dofs": [
                {"name": "q", "kind": "angle", "unit": unit, "range": [-180, 180], "default": 0},
                {"name": "slide", "kind": "length", "unit": "mm", "range": [-5, 5], "default": 0},
            ],
            "bodies": [
                {
                    "name": "moving",
                    "region": {"$node": "ball"},
                    "motion": {
                        "ops": [
                            {
                                "kind": "rotate",
                                "axis": [1, 2, -3],
                                "origin": [1, -2, 3],
                                "angle": _read("q"),
                            },
                            {"kind": "translate", "axis": [2, -1, 0], "distance": _read("slide")},
                            {
                                "kind": "rotate",
                                "axis": [0, 0, 2],
                                "origin": [-2, 1, 0],
                                "angle": _read("q"),
                            },
                        ]
                    },
                }
            ],
        },
    )


def _oracle(part, points, authored):
    evaluator = compile_kinematics(part)
    dofs = evaluator.to_evaluator_units(authored)
    matrices = np.asarray(evaluator.body_transforms(dofs))
    rest = [(points - m[:3, 3]) @ m[:3, :3] for m in matrices]
    distances = np.stack([fn(p) for fn, p in zip(evaluator._regions, rest, strict=True)])
    return rest, distances.min(axis=0)


def test_live_pose_controls_have_distinct_names_and_authored_units():
    part = _part("deg")
    before = copy.deepcopy(part.to_dict())
    emission = emit_glsl(part)
    controls = {c["param"]: c for c in emission.controls}
    assert controls["q"]["uniform"] == "u_p_q"
    pose = controls["kinematics.q"]
    assert pose["class"] == "live"
    assert pose["uniform"] == "u_dof_0"
    assert pose["dof"] == "q"
    assert pose["ui"]["role"] == "pose"
    assert pose["unit"] == "deg"
    assert pose["ui"]["explore_bounds"] == [-180, 180]
    assert all(c["bbox"] is not None for c in emission.components)
    assert part.to_dict() == before


@pytest.mark.parametrize("angle", np.linspace(-1.2, 1.2, 13))
def test_executed_shader_matches_nonzero_body_poses(shader_runtime, angle):
    part = _part()
    emission = emit_glsl(part)
    points = np.random.default_rng(42).uniform(-8, 8, (8, 3))
    authored = [angle, 2.5 * math.sin(angle)]
    rest, distance = _oracle(part, points, authored)
    got = shader_runtime.evaluate(
        emission, points, {"u_dof_0": authored[0], "u_dof_1": authored[1]}
    )
    # Float32 rotations and field evaluation across the CPU and software shader.
    np.testing.assert_allclose(got[:, :3], rest[0], atol=2e-5, rtol=2e-6)
    np.testing.assert_allclose(got[:, 3], distance, atol=2e-5, rtol=2e-6)


def test_degree_conversion_precedes_nonlinear_motion_and_default_upload(shader_runtime):
    part = _part("deg")
    part.kinematics["dofs"][0]["default"] = 60
    part.kinematics["bodies"][0]["motion"]["ops"][0]["angle"] = {
        "type": "unop",
        "op": "square",
        "child": _read("q"),
    }
    points = np.array([[2, 3, 0], [5, -1, 4]], dtype=float)
    rest, distance = _oracle(part, points, [60, 0])
    got = shader_runtime.evaluate(emit_glsl(part), points)
    np.testing.assert_allclose(got, np.column_stack([rest[0], distance]), atol=2e-5, rtol=2e-6)


def test_design_parameters_drive_geometry_and_motion_together(shader_runtime):
    part = _part()
    part.params["gain"] = Param(
        "gain",
        2,
        unit="ratio",
        expr={
            "type": "binop",
            "op": "*",
            "lhs": {"type": "param", "name": "q"},
            "rhs": {"type": "num", "value": 2},
        },
    )
    part.kinematics["bodies"][0]["motion"]["ops"][0]["angle"] = {
        "type": "binop",
        "op": "*",
        "lhs": _read("q"),
        "rhs": {"type": "param", "name": "gain"},
    }
    emission = emit_glsl(part)
    part.params["q"].value = 1.5
    points = np.array([[2, 3, 0], [-1, 2, 4]], dtype=float)
    rest, distance = _oracle(part, points, [0.3, 0])
    got = shader_runtime.evaluate(emission, points, {"u_p_q": 1.5, "u_dof_0": 0.3})
    np.testing.assert_allclose(got, np.column_stack([rest[0], distance]), atol=2e-5, rtol=2e-6)


def test_shared_regions_keep_distinct_body_fields_and_rest_cut_maps(shader_runtime):
    part = _part()
    fixed = copy.deepcopy(part.kinematics["bodies"][0])
    fixed["name"], fixed["motion"]["ops"] = "fixed", []
    part.kinematics["bodies"].append(fixed)
    emission = emit_glsl(part)
    assert len({c["machine"] for c in emission.components}) == 2
    points = np.array([[2, 3, 0], [3, 3, 0], [-1, 3, 0]], dtype=float)
    rest, distance = _oracle(part, points, [0.6, 2])
    uniforms = {"u_dof_0": 0.6, "u_dof_1": 2}
    got = shader_runtime.evaluate(emission, points, uniforms)
    np.testing.assert_allclose(got[:, 3], distance, atol=2e-5, rtol=2e-6)
    cut = shader_runtime.evaluate(
        emission,
        points,
        uniforms,
        "vec4(sdm_rest_point(p, 1), sdf_scene_rcut(p, vec3(1.0, 0.0, 0.0), 2.0))",
    )
    evaluator = compile_kinematics(part)
    expected = np.stack(
        [np.maximum(fn(p), p[:, 0] - 2) for fn, p in zip(evaluator._regions, rest, strict=True)]
    ).min(axis=0)
    np.testing.assert_allclose(cut[:, :3], points, atol=1e-6)
    np.testing.assert_allclose(cut[:, 3], expected, atol=2e-5, rtol=2e-6)


@pytest.mark.parametrize("box", [[[0, 0, 0], [0, 1, 1]], [[0, 0, 0], [1, 1, float("inf")]]])
def test_motion_rejects_invalid_authored_bounds(box):
    part = _part()
    part.metadata["bbox"] = box
    with pytest.raises(ValueError, match="finite"):
        emit_glsl(part)


def test_explicit_subtrees_remain_static_and_do_not_publish_pose_controls():
    part = _part()
    tree = part.materials[0].sdf_tree
    actual = emit_glsl(part, tree)
    part.kinematics = None
    expected = emit_glsl(part, tree)
    assert actual.scene_source == expected.scene_source
    assert actual.controls == expected.controls


def test_geometry_expression_namespace_still_rejects_motion_dofs():
    with pytest.raises(ValueError, match="Unknown expression node type 'dof'"):
        _GLSLEmitter(_part(), smooth_csg=False, smooth_k=0.25)._emit_expr(_read("q"))


def test_flexure_blocks_fail_instead_of_rendering_only_the_rigid_subset():
    part = _part()
    part.kinematics["flexures"] = [
        {
            "name": "flex",
            "region": {"$node": "ball"},
            "from_body": "moving",
            "to_body": "moving",
            "blend": {
                "type": "field",
                "kind": "axis_ramp",
                "params": {
                    "axis": [0, 0, 1],
                    "lo": -1,
                    "hi": 1,
                },
            },
        }
    ]
    with pytest.raises(ValueError, match="Flexure interpolation"):
        emit_glsl(part)


def test_saved_shader_artifacts_preserve_live_motion_controls(tmp_path):
    from software_defined_matter.glsl.__main__ import _write_artifacts

    emission = emit_glsl(_part("deg"))
    _write_artifacts(emission, tmp_path)
    metadata = json.loads((tmp_path / "meta.json").read_text())
    assert metadata["controls"] == emission.controls
    motion_uniform = next(u for u in metadata["uniforms"] if u["name"] == "u_dof_0")
    assert motion_uniform["source_param"] == "kinematics.q"
    assert motion_uniform["unit"] == "deg"
    assert metadata["components"][0]["bbox"] is not None


def test_automatic_bounds_contain_executed_shader_surfaces(shader_runtime):
    part = _part()
    part.metadata.pop("bbox")
    for dof in part.kinematics["dofs"]:
        dof["range"] = [-1, 1]
    emission = emit_glsl(part)
    evaluator = compile_kinematics(part)
    rest_surface = np.array([[3, 3, 0], [2, 4, 0], [1, 3, 0]], dtype=float)
    for angle in np.linspace(-1, 1, 7):
        posed = np.asarray(evaluator.pose_points(rest_surface, [angle, -angle]))
        result = shader_runtime.evaluate(emission, posed, {"u_dof_0": angle, "u_dof_1": -angle})
        np.testing.assert_allclose(result[:, 3], 0, atol=2e-5)
        for box in (emission.bbox, emission.components[0]["bbox"]):
            assert np.all(posed >= np.asarray(box[0]))
            assert np.all(posed <= np.asarray(box[1]))
