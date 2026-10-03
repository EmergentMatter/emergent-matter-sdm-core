"""Flexures agree with their welds, preserve turns, and vanish at zero motion."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jsonschema import ValidationError

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from software_defined_matter.kinematics import compile_kinematics
from software_defined_matter.sdf.sdf_ops import op_twist

ATOL = 2e-5  # Float32 matrix composition and shader-compatible trigonometry.


def _read(name="q"):
    return {"type": "dof", "name": name}


def _rotate(axis=(0, 0, 1), origin=(0, 0, 0), name="q"):
    return {"kind": "rotate", "axis": list(axis), "origin": list(origin), "angle": _read(name)}


def _part(first=None, second=None):
    region = sdf_primitive("box", b=[2, 2, 2])
    return Part(
        name="flexure",
        materials=[MaterialRegion(1, "solid", region)],
        kinematics={
            "dofs": [
                {"name": "q", "kind": "angle", "unit": "rad", "range": [-20, 20], "default": 0},
                {"name": "r", "kind": "angle", "unit": "rad", "range": [-3, 3], "default": 0},
            ],
            "bodies": [
                {
                    "name": "from",
                    "region": sdf_primitive("sphere", r=0.1),
                    "motion": {"ops": [] if first is None else first},
                },
                {
                    "name": "to",
                    "region": sdf_primitive("sphere", r=0.2),
                    "motion": {"ops": [_rotate()] if second is None else second},
                },
            ],
            "flexures": [
                {
                    "name": "bridge",
                    "region": region,
                    "from_body": "from",
                    "to_body": "to",
                    "blend": {
                        "type": "field",
                        "kind": "axis_ramp",
                        "params": {"axis": [0, 0, 1], "lo": 0, "hi": 1},
                    },
                }
            ],
        },
    )


def _compile(part=None):
    ev = compile_kinematics(_part() if part is None else part)
    assert ev is not None
    return ev


def test_flexure_ownership_and_clamped_welds():
    ev = _compile()
    assert ev.region_names == ("from", "to", "bridge")
    points = jnp.array([[1.0, 0, -1], [1, 0, 0.5], [1, 0, 2]])
    assert np.all(ev.ownership(points) == 2)
    got = ev.pose_points(points, jnp.array([math.pi, 0]))
    np.testing.assert_allclose(got, [[1, 0, -1], [0, 1, 0.5], [-1, 0, 2]], atol=ATOL)
    matrices = ev.flexure_transforms(points, jnp.array([math.pi, 0]))
    assert matrices.shape == (1, 3, 4, 4)
    np.testing.assert_allclose(
        matrices[0, 0], ev.body_transforms(jnp.array([math.pi, 0]))[0], atol=ATOL
    )
    np.testing.assert_allclose(
        matrices[0, 2], ev.body_transforms(jnp.array([math.pi, 0]))[1], atol=ATOL
    )


def test_joint_interpolation_preserves_full_turns_and_off_axis_origins():
    ev = _compile(_part(second=[_rotate(origin=(2, 0, 0))]))
    got = ev.pose_points(jnp.array([3.0, 0, 0.5]), jnp.array([2 * math.pi, 0]), owner=jnp.array(2))
    np.testing.assert_allclose(got, [1, 0, 0.5], atol=ATOL)


def test_general_screw_has_exact_welds_and_rigid_interior_frames():
    ev = _compile(_part([_rotate((1, 0, 0), (0, 1, 0))], [_rotate((0, 1, 0), (1, 0, 0), "r")]))
    points = jnp.array([[1.0, 2, 0], [1, 2, 0.25], [1, 2, 0.5], [1, 2, 1]])
    q = jnp.array([0.8, -1.2])
    matrices = np.asarray(ev.flexure_transforms(points, q)[0])
    np.testing.assert_allclose(matrices[[0, 3]], ev.body_transforms(q), atol=ATOL)
    for m in matrices:
        np.testing.assert_allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=ATOL)
        assert np.linalg.det(m[:3, :3]) == pytest.approx(1, abs=ATOL)
    # Equal increments of the same relative screw compose to the half step.
    rel_quarter = np.linalg.inv(matrices[0]) @ matrices[1]
    rel_half = np.linalg.inv(matrices[0]) @ matrices[2]
    np.testing.assert_allclose(rel_quarter @ rel_quarter, rel_half, atol=ATOL)


@pytest.mark.parametrize("general", [False, True], ids=["joint", "screw"])
def test_zero_pose_jit_vmap_and_derivatives_are_finite(general):
    part = _part([_rotate((1, 0, 0))], [_rotate((0, 1, 0), name="r")]) if general else _part()
    ev = _compile(part)
    points = jnp.array([[1.0, 2, 0.25], [2, -1, 0.75]])
    owner = jnp.array([2, 2])
    fn = jax.jit(lambda p, q: ev.pose_points(p, q, owner))
    np.testing.assert_allclose(fn(points, jnp.zeros(2)), points, atol=ATOL)
    for q in [jnp.zeros(2), jnp.array([0.5, -0.4])]:
        deriv = np.asarray(jax.jacfwd(fn, argnums=1)(points, q))
        assert np.isfinite(deriv).all()
        assert np.isfinite(jax.jacrev(fn, argnums=1)(points, q)).all()
        h = 0.001
        for i in range(2):
            step = jnp.eye(2)[i] * h
            finite = (fn(points, q + step) - fn(points, q - step)) / (2 * h)
            np.testing.assert_allclose(deriv[..., i], finite, atol=5e-4)
    poses = jax.jit(jax.vmap(lambda q: ev.pose_points(points, q, owner)))(
        jnp.array([[0.0, 0], [0.3, 0.2]])
    )
    assert poses.shape == (2, 2, 3)
    assert np.isfinite(jax.jacfwd(fn, argnums=0)(points, jnp.zeros(2))).all()


def test_linear_axial_twist_agrees_with_query_space_sign():
    part = _part(second=[_rotate((0, 1, 0))])
    part.kinematics["flexures"][0]["blend"]["params"]["axis"] = [0, 1, 0]
    ev = _compile(part)
    rest = jnp.array([[1.0, 0.2, 2], [-2, 0.6, 1], [1, 0.9, -1]])
    angle = 0.7
    posed = ev.pose_points(rest, jnp.array([angle, 0]), jnp.full(3, 2))
    # The y coordinate is invariant. op_twist rotates query points opposite
    # to the authored right-hand body rotation.
    recovered = op_twist(lambda p: p, posed, angle)
    np.testing.assert_allclose(recovered, rest, atol=ATOL)


def test_snapshot_and_degree_conversion_precede_interpolation():
    part = _part()
    part.kinematics["dofs"][0].update(unit="deg", range=[-720, 720])
    ev = _compile(part)
    part.kinematics["flexures"][0]["blend"]["params"]["hi"] = 100
    got = ev.pose_points(
        jnp.array([1.0, 0, 0.5]), ev.to_evaluator_units(jnp.array([180.0, 0.0])), jnp.array(2)
    )
    np.testing.assert_allclose(got, [0, 1, 0.5], atol=ATOL)


def test_invalid_owner_and_empty_batches_keep_the_contract():
    ev = _compile()
    assert np.isnan(ev.pose_points(jnp.ones(3), jnp.zeros(2), jnp.array(3))).all()
    assert ev.pose_points(jnp.empty((0, 3)), jnp.zeros(2)).shape == (0, 3)
    assert ev.flexure_transforms(jnp.empty((0, 3)), jnp.zeros(2)).shape == (1, 0, 4, 4)
    part = _part()
    part.kinematics["flexures"] = []
    assert _compile(part).flexure_transforms(jnp.ones((2, 3)), jnp.zeros(2)).shape == (0, 2, 4, 4)


def test_unknown_blend_is_refused_before_animation():
    part = _part()
    part.kinematics["flexures"][0]["blend"]["kind"] = "unknown"
    with pytest.raises(ValueError, match="Unknown field primitive"):
        _compile(part)


def test_hidden_rest_turn_cannot_distort_a_flexure():
    part = _part()
    part.kinematics["bodies"][1]["motion"]["ops"][0]["angle"] = {
        "type": "num",
        "value": 2 * math.pi,
    }
    with pytest.raises(ValueError, match="matching endpoint joint coordinates"):
        _compile(part)


@pytest.mark.parametrize(
    "axis,lo,hi", [([0, 0, 0], 0, 1), ([0, 1], 0, 1), ([0, 0, 1], 1, 1), ([0, 0, 1], 2, 1)]
)
def test_invalid_axis_ramp_is_rejected(axis, lo, hi):
    part = _part()
    part.kinematics["flexures"][0]["blend"]["params"] = {"axis": axis, "lo": lo, "hi": hi}
    with pytest.raises((ValueError, ValidationError)):
        _compile(part)


@pytest.mark.parametrize("angle", [0.0, 0.7, math.pi - 0.001])
def test_general_screw_matches_known_offset_axis_with_axial_translation(angle):
    # Different chain lengths force the general path. The first body's zero
    # rotation is identity; the other describes a screw around z through x=2.
    move = {"kind": "translate", "axis": [0, 0, 1], "distance": _read("r")}
    part = _part([_rotate((1, 0, 0), name="r")], [_rotate(origin=(2, 0, 0)), move])
    part.kinematics["bodies"][0]["motion"]["ops"][0]["angle"] = {"type": "num", "value": 0}
    ev = _compile(part)
    got = ev.pose_points(jnp.array([3.0, 0, 0.5]), jnp.array([angle, 2.0]), jnp.array(2))
    expected = [2 + math.cos(angle / 2), math.sin(angle / 2), 1.5]
    np.testing.assert_allclose(got, expected, atol=ATOL)


def test_reversed_joint_axes_preserve_authored_rotation():
    ev = _compile(_part([_rotate()], [_rotate((0, 0, -1), name="r")]))
    got = ev.pose_points(jnp.array([1.0, 0, 0.5]), jnp.array([2 * math.pi, 0.0]), jnp.array(2))
    np.testing.assert_allclose(got, [-1, 0, 0.5], atol=ATOL)


def test_scalar_field_blend_uses_the_same_clamping_and_batch_contract():
    part = _part()
    part.kinematics["flexures"][0]["blend"] = {
        "type": "field",
        "kind": "sin_xyz",
        "params": {"freq": [0, 0, 0], "phase": [math.pi / 2] * 3, "amplitude": 0.5},
    }
    ev = _compile(part)
    points = jnp.array([[1.0, 0, 0], [2.0, 0, 1]])
    got = ev.pose_points(points, jnp.array([math.pi, 0]), jnp.array([2, 2]))
    np.testing.assert_allclose(got, [[0, 1, 0], [0, 2, 1]], atol=ATOL)
