"""Design inputs drive motion, flexure blends, material queries and ports without recompiling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    Assembly,
    Frame,
    Instance,
    Param,
    PartRef,
    Port,
    load_bundle,
    save,
)
from software_defined_matter.kinematics import compile_kinematics
from software_defined_matter.material_motion import compile_material_motion
from tests.test_flexure_motion import _part
from tests.test_material_motion import _bodies

ATOL = 2e-5  # Float32 trigonometry and homogeneous transform products.


def _design_part():
    part = _part()
    part.params = {
        "gain": Param("gain", 1.0, free=True, unit="count"),
        "double_gain": Param(
            "double_gain",
            2.0,
            unit="count",
            expr={
                "type": "binop",
                "op": "*",
                "lhs": {"type": "param", "name": "gain"},
                "rhs": {"type": "num", "value": 2},
            },
        ),
    }
    part.kinematics["bodies"][1]["motion"]["ops"][0]["angle"] = {
        "type": "binop",
        "op": "*",
        "lhs": {"type": "param", "name": "double_gain"},
        "rhs": {"type": "dof", "name": "q"},
    }
    part.ports = [
        Port("tip", Frame(position=({"$ref": "gain"}, 0, 1)), body="to"),
        Port("reference", Frame(position=({"$ref": "gain"}, 0, 0))),
    ]
    return part


def test_compiled_motion_uses_live_design_relations_and_has_design_gradients():
    part = _design_part()
    motion = compile_kinematics(part)
    assert motion is not None

    def evaluate(design):
        return motion.port_transforms(jnp.array([0.3, 0]), free_vec=design)[0, :3, 3]

    compiled = jax.jit(evaluate)
    for gain in (1.0, 2.0):
        expected = [gain * np.cos(0.6 * gain), gain * np.sin(0.6 * gain), 1]
        np.testing.assert_allclose(compiled(jnp.array([gain])), expected, atol=ATOL)
    design = jnp.array([1.3])
    step = 1e-3
    finite_difference = (compiled(design + step) - compiled(design - step)) / (2 * step)
    np.testing.assert_allclose(jax.jacfwd(evaluate)(design)[:, 0], finite_difference, atol=2e-4)
    part.params["gain"].value = 99
    np.testing.assert_allclose(motion.port_transforms([0, 0])[0, :3, 3], [1, 0, 1], atol=ATOL)
    np.testing.assert_allclose(
        motion.port_transforms([1, 0], free_vec=[2])[1, :3, 3], [2, 0, 0], atol=ATOL
    )


def test_flexure_points_share_dynamic_endpoint_design_inputs():
    motion = compile_kinematics(_design_part())
    point = jnp.array([1.0, 0, 0.5])

    def evaluate(gain):
        return motion.pose_points(point, jnp.array([0.4, 0]), owner=jnp.array(2), free_vec=gain)

    np.testing.assert_allclose(
        jax.jit(evaluate)(jnp.array([2.0])), [np.cos(0.8), np.sin(0.8), 0.5], atol=ATOL
    )
    assert np.isfinite(jax.jacfwd(evaluate)(jnp.array([2.0]))).all()
    posed = evaluate(jnp.array([2.0]))
    np.testing.assert_allclose(
        motion.inverse_flexure_points(posed, [0.4, 0], flexure="bridge", free_vec=[2]),
        point,
        atol=ATOL,
    )


def test_assembly_binding_drives_contained_fixed_parameters(tmp_path):
    part = _design_part()
    part.params["gain"].free = False
    save(part, tmp_path / "part.sdm")
    save(
        Assembly(
            "root",
            params={"scale": Param("scale", 3, free=True, unit="count")},
            instances=(
                Instance("child", PartRef("part.sdm"), param_overrides={"gain": {"$ref": "scale"}}),
            ),
        ),
        tmp_path / "assembly.sdm",
    )
    bundle = load_bundle(tmp_path / "assembly.sdm")
    motion = compile_kinematics(part, binding=bundle.binding("child"))
    np.testing.assert_allclose(motion.port_transforms([0, 0])[0, :3, 3], [3, 0, 1], atol=ATOL)
    np.testing.assert_allclose(
        motion.port_transforms([0, 0], free_vec=[4])[0, :3, 3], [4, 0, 1], atol=ATOL
    )


def test_material_membership_and_ownership_use_supplied_design():
    part = _bodies()
    part.params = {"radius": Param("radius", 2, free=True, unit="mm")}
    part.materials[0].sdf_tree["params"]["r"] = {"$ref": "radius"}
    part.kinematics["bodies"][0]["region"]["child"]["params"]["r"] = {"$ref": "radius"}
    motion = compile_material_motion(part)
    assert not bool(motion.contains([3, 0, 0], [0], free_vec=[2]))
    assert bool(
        jax.jit(lambda p: motion.contains(jnp.array([3.0, 0, 0]), jnp.array([0.0]), free_vec=p))(
            jnp.array([4.0])
        )
    )
    assert int(motion.kinematics.ownership([0.5, 0, 0], free_vec=[0.1])) == 1
    assert int(motion.kinematics.ownership([0.5, 0, 0], free_vec=[4])) == 0


def test_motion_rejects_design_vector_with_wrong_shape():
    motion = compile_kinematics(_design_part())
    with pytest.raises(ValueError, match="Expected design shape"):
        motion.body_transforms([0, 0], free_vec=[1, 2])


def test_general_blend_field_reads_live_design_vector():
    part = _part()
    part.params = {"weight": Param("weight", 0.25, free=True, unit="count")}
    part.kinematics["flexures"][0]["blend"] = {
        "type": "field",
        "kind": "sin_xyz",
        "params": {"freq": [0, 0, 0], "phase": [np.pi / 2] * 3, "amplitude": {"$ref": "weight"}},
    }
    motion = compile_kinematics(part)

    def evaluate(design):
        return motion.pose_points(
            jnp.array([1.0, 0, 0]), jnp.array([1.0, 0]), owner=jnp.array(2), free_vec=design
        )

    np.testing.assert_allclose(
        jax.jit(evaluate)(jnp.array([0.5])), [np.cos(0.5), np.sin(0.5), 0], atol=ATOL
    )
    np.testing.assert_allclose(
        jax.jacfwd(evaluate)(jnp.array([0.5]))[:, 0], [-np.sin(0.5), np.cos(0.5), 0], atol=ATOL
    )


def test_empty_motion_retains_empty_port_array_and_default_behavior():
    part = _part()
    part.kinematics = {"dofs": [], "bodies": []}
    motion = compile_kinematics(part)
    assert motion.port_transforms([]).shape == (0, 4, 4)
    np.testing.assert_allclose(motion.pose_points([1, 2, 3], []), [1, 2, 3], atol=ATOL)
