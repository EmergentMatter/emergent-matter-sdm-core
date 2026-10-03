"""Swept flexure boxes enclose owned material throughout the advertised motion."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.kinematics import compile_kinematics
from software_defined_matter.motion_bounds import infer_motion_bounds
from software_defined_matter.sdf.bbox import BBoxInferenceError


def _rotate(name="q", axis=(0, 0, 1), origin=(0, 0, 0)):
    return {
        "kind": "rotate",
        "axis": list(axis),
        "origin": list(origin),
        "angle": {"type": "dof", "name": name},
    }


def _part(first=None, second=None):
    solid = sdf_primitive("box", b=[2, 2, 2])
    return Part(
        name="flexure bounds",
        materials=[MaterialRegion(1, "solid", solid)],
        kinematics={
            "dofs": [
                {"name": name, "kind": "angle", "unit": "rad", "range": [-7, 7]}
                for name in ("q", "r", "unused")
            ],
            "bodies": [
                {
                    "name": "a",
                    "region": sdf_primitive("sphere", r=0.1),
                    "motion": {"ops": [] if first is None else first},
                },
                {
                    "name": "b",
                    "region": sdf_primitive("sphere", r=0.2),
                    "motion": {"ops": [_rotate()] if second is None else second},
                },
            ],
            "flexures": [
                {
                    "name": "bridge",
                    "from_body": "a",
                    "to_body": "b",
                    "region": sdf_primitive("sphere", r=0.3),
                    "blend": {
                        "type": "field",
                        "kind": "axis_ramp",
                        "params": {"axis": [0, 0, 1], "lo": -2, "hi": 2},
                    },
                }
            ],
        },
    )


def _bounds(part):
    result = infer_motion_bounds(part, include_flexures=True)
    assert result is not None
    return result


def _contains(box, points):
    assert box is not None
    points = np.asarray(points)
    assert np.all(points >= np.asarray(box[0]))
    assert np.all(points <= np.asarray(box[1]))


@pytest.mark.parametrize("general", [False, True], ids=["joint", "screw"])
def test_material_outside_classifier_boxes_is_contained(general):
    part = (
        _part([_rotate(axis=(1, 0, 0), origin=(0, 3, 0))], [_rotate("r", (0, 1, 0), (4, 0, 0))])
        if general
        else _part()
    )
    bounds = _bounds(part)
    assert bounds.flexures[0].name == "bridge"
    assert bounds.flexures[0].dof_names == (("q", "r") if general else ("q",))
    ev = compile_kinematics(part)
    points = jnp.asarray(np.random.default_rng(205).uniform(-2, 2, (80, 3)))
    # Every point is outside or inside arbitrary classifier solids; nearest
    # ownership covers all material, not just the negative region sets.
    assert bounds.flexures[0].rest_bbox[1][0] >= 2
    for q in [0.0, 0.7, math.pi - 0.001, 2 * math.pi, -6.8]:
        dofs = jnp.array([q, -0.6 * q, 0.0])
        _contains(bounds.bbox, ev.pose_points(points, dofs))
        for owner, bound in enumerate((*bounds.bodies, *bounds.flexures)):
            _contains(bound.bbox, ev.pose_points(points, dofs, jnp.full(80, owner)))


def test_full_turn_midpoint_and_reversed_axes():
    part = _part([_rotate()], [_rotate("r", (0, 0, -1))])
    bounds = _bounds(part)
    ev = compile_kinematics(part)
    p = jnp.array([1.0, 0, 0])
    posed = ev.pose_points(p, jnp.array([2 * math.pi, 0.0, 0.0]), jnp.array(2))
    np.testing.assert_allclose(posed, [-1, 0, 0], atol=2e-5)
    _contains(bounds.flexures[0].bbox, posed)


def test_ground_and_ordered_rotation_translation_chain():
    move = {"kind": "translate", "axis": [1, 0, 0], "distance": {"type": "dof", "name": "r"}}
    part = _part(second=[_rotate(origin=(3, 0, 0)), move])
    part.kinematics["dofs"][0].update(unit="deg", range=[-720, 720])
    part.kinematics["dofs"][1].update(kind="length", unit="mm", range=[-4, 4])
    bounds = _bounds(part)
    ev = compile_kinematics(part)
    points = jnp.array([[2.0, 2, -2], [-2, -2, 0], [2, -2, 2]])
    for angle in [-720.0, 0.0, 270.0, 720.0]:
        posed = ev.pose_points(
            points, ev.to_evaluator_units(jnp.array([angle, 4.0, 0.0])), jnp.full(3, 2)
        )
        _contains(bounds.flexures[0].bbox, posed)


def test_unbounded_material_disables_all_owner_pruning_with_authored_box():
    part = _part()
    with pytest.raises(BBoxInferenceError, match="Flexure material support"):
        infer_motion_bounds(part, include_flexures=True, smooth_csg=True)
    part.metadata["bbox"] = [[-20, -20, -20], [20, 20, 20]]
    bounds = infer_motion_bounds(part, include_flexures=True, smooth_csg=True)
    assert all(b.bbox is None and b.rest_bbox is None for b in (*bounds.bodies, *bounds.flexures))
    assert bounds.bbox == ((-20.0, -20.0, -20.0), (20.0, 20.0, 20.0))


def test_legacy_consumers_and_shaders_still_refuse_flexures():
    part = _part()
    with pytest.raises(BBoxInferenceError, match="include_flexures=True"):
        infer_motion_bounds(part)
    with pytest.raises(ValueError, match="Flexure interpolation"):
        emit_glsl(part)


def test_rigid_only_api_has_an_empty_flexure_list():
    part = _part()
    part.kinematics["flexures"] = []
    assert infer_motion_bounds(part).flexures == ()
