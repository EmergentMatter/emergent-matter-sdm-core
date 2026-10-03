"""Actual material is partitioned in rest space, then moved by its owner."""

from __future__ import annotations

import copy
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive, sdf_transform
from software_defined_matter.glsl import emit_glsl, emit_material_membership
from software_defined_matter.kinematics import compile_kinematics
from software_defined_matter.material_motion import compile_material_motion
from tests._glsl_runtime import ShaderRuntime


def _ball(center, radius):
    return sdf_transform("translate", sdf_primitive("sphere", r=radius), t=list(center))


def _bodies():
    return Part(
        name="partitioned material",
        materials=[MaterialRegion(7, "solid", sdf_primitive("sphere", r=2))],
        kinematics={
            "dofs": [{"name": "shift", "kind": "length", "unit": "mm", "range": [-3, 3]}],
            "bodies": [
                {
                    "name": name,
                    "region": _ball((side, 0, 0), 0.25),
                    "motion": {
                        "ops": [
                            {
                                "kind": "translate",
                                "axis": [side, 0, 0],
                                "distance": {"type": "dof", "name": "shift"},
                            }
                        ]
                    },
                }
                for name, side in [("left", -1), ("right", 1)]
            ],
        },
    )


def _flexure(radial=True, unit="rad"):
    part = _bodies()
    block = part.kinematics
    block["dofs"] = [
        {
            "name": "angle",
            "kind": "angle",
            "unit": unit,
            "range": [-720, 720] if unit == "deg" else [-15, 15],
        }
    ]
    block["bodies"][0]["motion"]["ops"] = []
    block["bodies"][1]["motion"]["ops"] = [
        {
            "kind": "rotate",
            "axis": [0, 0, 1],
            "origin": [0, 0, 0],
            "angle": {"type": "dof", "name": "angle"},
        }
    ]
    block["bodies"][0]["region"] = _ball((0, 0, -1), 0.1)
    block["bodies"][1]["region"] = _ball((0, 0, 2), 0.1)
    block["flexures"] = [
        {
            "name": "blade",
            "from_body": "left",
            "to_body": "right",
            "region": _ball((1, 0, 1), 0.2),
            "blend": {
                "type": "field",
                "kind": "radial_hermite" if radial else "axis_ramp",
                "params": {"axis": [0, 0, 1], "origin": [0, 0, 0], "r0": 0, "r1": 2}
                if radial
                else {"axis": [0, 0, 1], "lo": 0, "hi": 2},
            },
        }
    ]
    return part


@pytest.fixture(scope="module")
def shader_runtime():
    try:
        runtime = ShaderRuntime()
    except (OSError, AttributeError, RuntimeError) as exc:
        if os.environ.get("SDM_REQUIRE_GLSL") == "1":
            pytest.fail(f"Required software shader runtime unavailable: {exc}")
        pytest.skip(f"Software shader runtime unavailable: {exc}")
    yield runtime
    runtime.close()


def test_zero_pose_reproduces_material_including_internal_ownership_tie():
    part = _bodies()
    ev = compile_material_motion(part)
    points = np.array([[0, 0, 0], [-1.5, 0, 0], [1.5, 0, 0], [2, 0, 0], [2.1, 0, 0]])
    np.testing.assert_array_equal(
        ev.membership(points, [0])[:, 0],
        [
            [True, True, False, False, False],
            [False, False, True, True, False],
        ],
    )
    # At the origin both classifiers are positive, while real material is solid.
    # The proposed max(material, d_i-d_j) clipping incorrectly gives zero there.
    assert max(-2.0, 0.75 - 0.75) == 0
    assert ev.contains([0, 0, 0], [0])
    cloud = np.random.default_rng(6).uniform(-3, 3, (250, 3))
    np.testing.assert_array_equal(ev.contains(cloud, [0]), np.linalg.norm(cloud, axis=-1) <= 2)


def test_separation_gap_overlap_and_material_record_identity():
    part = _bodies()
    # Same external material ID, distinct geometry and document records.
    part.materials.append(MaterialRegion(7, "core", sdf_primitive("sphere", r=0.5)))
    ev = compile_material_motion(part)
    assert ev.material_ids == (7, 7)
    assert ev.region_names == ("left", "right")
    np.testing.assert_array_equal(ev.membership([0, 0, 0], [-0.25]), [[True, True], [True, True]])
    assert not ev.contains([0, 0, 0], [1])
    np.testing.assert_array_equal(ev.membership([-2, 0, 0], [1]), [[True, False], [False, False]])


@pytest.mark.parametrize("radial", [False, True])
def test_flexure_uses_rest_ownership_after_inverse_not_posed_classifiers(radial):
    part = _flexure(radial)
    ev = compile_material_motion(part)
    motion = compile_kinematics(part)
    rest = jnp.array([1.0, 0, 1])
    posed = motion.pose_points(rest, jnp.array([2 * math.pi]), jnp.array(2))
    np.testing.assert_allclose(posed, [-1, 0, 1], atol=1e-6)
    assert motion.ownership(posed) != 2
    np.testing.assert_allclose(ev.rest_points(posed, [2 * math.pi])[2], rest, atol=2e-6)
    assert ev.membership(posed, [2 * math.pi])[2, 0]


def test_snapshot_batch_jit_empty_and_invalid_points():
    part = _bodies()
    ev = compile_material_motion(part)
    part.materials[0].sdf_tree["params"]["r"] = 0.01
    part.kinematics["bodies"][0]["region"] = _ball((100, 0, 0), 1)
    points = jnp.array([[[0.2, 0, 0], [3, 0, 0]], [[float("nan"), 0, 0], [float("inf"), 0, 0]]])
    np.testing.assert_array_equal(
        jax.jit(ev.contains)(points, jnp.array([0.0])), [[True, False], [False, False]]
    )
    assert ev.membership(jnp.empty((0, 3)), [0]).shape == (2, 1, 0)
    with pytest.raises(ValueError, match="shape"):
        ev.contains([1, 2], [0])


@pytest.mark.parametrize("factory", [compile_material_motion, emit_material_membership])
def test_missing_motion_materials_and_unsupported_inverse(factory):
    assert factory(Part(name="empty")) is None
    part = _bodies()
    part.materials.clear()
    with pytest.raises(ValueError, match="material"):
        factory(part)
    part = _flexure()
    part.kinematics["flexures"][0]["blend"] = {
        "type": "field",
        "kind": "axis_ramp",
        "params": {"axis": [1, 0, 0], "lo": 0, "hi": 2},
    }
    with pytest.raises(ValueError, match="supported flexure inverses.*blade"):
        factory(part)


@pytest.mark.parametrize("radial", [False, True])
@pytest.mark.parametrize("unit", ["rad", "deg"])
def test_executed_shader_membership_and_inverse_parity(shader_runtime, radial, unit):
    part = _flexure(radial, unit)
    part.materials.append(MaterialRegion(8, "insert", _ball((1, 0, 1), 0.4)))
    ev = compile_material_motion(part)
    emission = emit_material_membership(part)
    assert emission.region_names == ev.region_names
    assert emission.material_ids == (7, 8)
    assert emission.entry_point == "sdm_material_contains"
    assert "sdf_scene" not in emission.scene_source
    points = np.concatenate(
        [np.random.default_rng(11).uniform(-2, 2, (60, 3)), [[-1, 0, 1], [0, 0, 0]]]
    )
    for angle in [0, 1.3, 2 * math.pi, -4 * math.pi]:
        uniform = math.degrees(angle) if unit == "deg" else angle
        expected = np.asarray(ev.membership(points, [angle]))
        for region in range(3):
            expression = (
                f"vec4(sdm_material_rest(p, {region}), float(sdm_material_member(p, {region}, 1)))"
            )
            got = shader_runtime.evaluate(emission, points, {"u_dof_0": uniform}, expression)
            np.testing.assert_allclose(
                got[:, :3], ev.rest_points(points, [angle])[region], atol=1e-5
            )
            np.testing.assert_array_equal(got[:, 3].astype(bool), expected[region, 1])
            got = shader_runtime.evaluate(
                emission,
                points,
                {"u_dof_0": uniform},
                f"vec4(float(sdm_material_member(p, {region}, 0)), "
                "float(sdm_material_contains(p)), "
                "float(sdm_material_member(p, -1, 0)), float(sdm_material_member(p, 0, 2)))",
            )
            np.testing.assert_array_equal(got[:, 0].astype(bool), expected[region, 0])
            np.testing.assert_array_equal(got[:, 1].astype(bool), expected.any(axis=(0, 1)))
            assert not got[:, 2:].any()
    with pytest.raises(ValueError, match="Flexure"):
        emit_glsl(part)


def test_executed_shader_overlap_and_ties(shader_runtime):
    part = _bodies()
    emission = emit_material_membership(part)
    ev = compile_material_motion(part)
    points = [[0, 0, 0], [-1.5, 0, 0], [1.5, 0, 0], [3, 0, 0]]
    for shift in [-0.25, 0, 1]:
        got = shader_runtime.evaluate(
            emission,
            points,
            {"u_dof_0": shift},
            "vec4(float(sdm_material_member(p, 0, 0)), float(sdm_material_member(p, 1, 0)), 0, 0)",
        )
        np.testing.assert_array_equal(
            got[:, :2].astype(bool), np.asarray(ev.membership(points, [shift]))[:, 0].T
        )


@pytest.mark.parametrize("radial", [False, True])
@pytest.mark.parametrize("ground", ["neither", "first", "second", "both"])
def test_executed_shader_offset_axes_ground_and_expression_inputs(shader_runtime, radial, ground):
    from software_defined_matter import Param

    part = _flexure(radial, "deg")
    part.params["gain"] = Param("gain", 0.7, unit="ratio")
    block = part.kinematics
    axis = np.array([1.0, 2, 3])
    origin = np.array([2.0, -1, 3])
    block["bodies"][0]["motion"]["ops"] = [
        {
            "kind": "rotate",
            "axis": axis.tolist(),
            "origin": origin.tolist(),
            "angle": {
                "type": "binop",
                "op": "*",
                "lhs": {"type": "num", "value": 0.4},
                "rhs": {"type": "dof", "name": "angle"},
            },
        }
    ]
    block["bodies"][1]["motion"]["ops"] = [
        {
            "kind": "rotate",
            "axis": (-axis).tolist(),
            "origin": (origin + axis).tolist(),
            "angle": {
                "type": "binop",
                "op": "*",
                "lhs": {"type": "param", "name": "gain"},
                "rhs": {
                    "type": "binop",
                    "op": "*",
                    "lhs": {"type": "dof", "name": "angle"},
                    "rhs": {"type": "dof", "name": "angle"},
                },
            },
        }
    ]
    for i, label in enumerate(["first", "second"]):
        if ground in {label, "both"}:
            block["bodies"][i]["motion"]["ops"] = []
    params = block["flexures"][0]["blend"]["params"]
    params["axis"] = axis.tolist()  # Axis ramp deliberately keeps this scale.
    if radial:
        params["origin"] = (origin - axis).tolist()
    else:
        params.update(lo=-4, hi=8)
    emission = emit_material_membership(part)
    # Edit a live design input after emission. The shader must match a fresh CPU snapshot.
    changed = copy.deepcopy(part)
    changed.params["gain"].value = 1.1
    reference = compile_material_motion(changed)
    points = np.random.default_rng(13).uniform(-3, 4, (50, 3))
    uniforms = {"u_dof_0": 250}
    uniforms.update({u.name: 1.1 for u in emission.uniforms if u.source_param == "gain"})
    got = shader_runtime.evaluate(
        emission, points, uniforms, "vec4(sdm_material_rest(p, 2), float(sdm_material_contains(p)))"
    )
    radians = [math.radians(250)]
    np.testing.assert_allclose(got[:, :3], reference.rest_points(points, radians)[2], atol=2e-5)
    np.testing.assert_array_equal(got[:, 3].astype(bool), reference.contains(points, radians))


def test_document_smoothing_matches_cpu_material_and_classifier_queries(shader_runtime):
    from software_defined_matter import sdf_op

    part = _bodies()
    part.materials[0].sdf_tree = sdf_op(
        "union", [_ball((-0.95, 0, 0), 0.9), _ball((0.95, 0, 0), 0.9)]
    )
    part.metadata["smooth_csg"] = True
    points = [[0, 0, 0], [0, 0.3, 0], [0.05, 0, 0]]
    ev = compile_material_motion(part)
    assert ev.contains(points, [0])[0]  # Smooth blend fills the hard union's gap.
    got = shader_runtime.evaluate(
        emit_material_membership(part),
        points,
        expression="vec4(float(sdm_material_contains(p)), 0, 0, 0)",
    )
    np.testing.assert_array_equal(got[:, 0].astype(bool), ev.contains(points, [0]))


def test_polygon_and_raster_resource_payloads_include_classifier_geometry():
    from software_defined_matter import sdf_2d_to_3d, sdf_raster_field

    part = _bodies()
    angles = np.linspace(0, 2 * math.pi, 600, endpoint=False)
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=-1).tolist()
    polygon = sdf_2d_to_3d("extrusion", sdf_primitive("polygon_2d", vertices=vertices), h=1)
    values = np.arange(27, dtype=np.float32).reshape(3, 3, 3) - 10
    raster = sdf_raster_field([0, 0, 0], 1, values)
    part.kinematics["bodies"][0]["region"] = polygon
    part.materials[0].sdf_tree = raster
    emission = emit_material_membership(part)
    np.testing.assert_array_equal(emission.grid_table, values.ravel())
    assert emission.grid_tex_width == 4096
    assert emission.poly_max_n == 600
    assert emission.poly_tex_width == 1024
    assert len(emission.poly_table) == 1202  # Repeated classifier emission is memoized.
    np.testing.assert_allclose(np.asarray(emission.poly_table[2:]).reshape(-1, 2), vertices)
    assert "#define SDM_GRID_LEN 27" in emission.lib_source
    assert "#define SDM_POLY_LEN 601" in emission.lib_source
