"""Radial blends round-trip on the wire and supported inverses undo forward motion."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jsonschema
import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from software_defined_matter.io import load_schema, validate
from software_defined_matter.kinematics import compile_kinematics
from software_defined_matter.motion_bounds import infer_motion_bounds
from software_defined_matter.sdf.sdf_ops import op_twist_radial

ATOL = 3e-5  # Float32 matrix composition, including multiple full turns.


def _rotate(name="q", axis=(0, 0, 1), origin=(0, 0, 0)):
    return {
        "kind": "rotate",
        "axis": list(axis),
        "origin": list(origin),
        "angle": {"type": "dof", "name": name},
    }


def _part():
    solid = sdf_primitive("box", b=[3, 3, 3])
    return Part(
        name="radial flexure",
        materials=[MaterialRegion(1, "solid", solid)],
        kinematics={
            "dofs": [
                {"name": name, "kind": "angle", "unit": "rad", "range": [-15, 15]}
                for name in ["q", "r"]
            ],
            "bodies": [
                {
                    "name": "inner",
                    "region": sdf_primitive("sphere", r=0.2),
                    "motion": {"ops": [_rotate()]},
                },
                {"name": "outer", "region": solid, "motion": {"ops": [_rotate("r")]}},
            ],
            "flexures": [
                {
                    "name": "blade",
                    "from_body": "inner",
                    "to_body": "outer",
                    "region": solid,
                    "blend": {
                        "type": "field",
                        "kind": "radial_hermite",
                        "params": {"axis": [0, 0, 1], "origin": [0, 0, 0], "r0": 0.0, "r1": 2.0},
                    },
                }
            ],
        },
    )


def _compile(part):
    ev = compile_kinematics(part)
    assert ev is not None
    return ev


def test_radial_document_requires_0_4_without_churning_older_documents():
    part = _part()
    doc = part.to_dict()
    assert doc["schema_version"] == "0.4"
    validate(part)
    assert Part.from_dict(json.loads(json.dumps(doc))).to_dict() == doc
    old = copy.deepcopy(doc)
    old["schema_version"] = "0.3"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(old, load_schema("0.3"))
    part.kinematics["flexures"][0]["blend"] = {
        "type": "field",
        "kind": "axis_ramp",
        "params": {"axis": [0, 0, 1], "lo": 0, "hi": 2},
    }
    assert part.to_dict()["schema_version"] == "0.3"
    part.kinematics = None
    assert part.to_dict()["schema_version"] == "0.2"


@pytest.mark.parametrize(
    "angles", [(0.0, 0.0), (0.4, -0.9), (0.0, 2 * math.pi), (-4 * math.pi, 2 * math.pi)]
)
def test_radial_forward_inverse_and_existing_operator_parity(angles):
    ev = _compile(_part())
    assert ev.inverse_flexure_names == ("blade",)
    rest = jnp.array([[0.0, 0, 0.5], [0.5, 0, 0], [1.0, 0.3, 0], [2.0, 0, 0], [3.0, 0, 0]])
    dofs = jnp.array(angles)
    posed = ev.pose_points(rest, dofs, jnp.full(5, 2))
    inverse = ev.inverse_flexure_points(posed, dofs, flexure="blade")
    np.testing.assert_allclose(inverse, rest, atol=ATOL)
    np.testing.assert_allclose(
        inverse, op_twist_radial(lambda p: p, posed, 0.0, 2.0, *angles), atol=ATOL
    )
    np.testing.assert_allclose(posed[-1], ev.pose_points(rest[-1], dofs, jnp.array(1)), atol=ATOL)
    np.testing.assert_allclose(posed[0], ev.pose_points(rest[0], dofs, jnp.array(0)), atol=ATOL)


def test_full_turn_has_a_half_turn_at_the_radial_midpoint():
    ev = _compile(_part())
    got = ev.pose_points(jnp.array([1.0, 0, 0]), jnp.array([0.0, 2 * math.pi]), jnp.array(2))
    np.testing.assert_allclose(got, [-1, 0, 0], atol=ATOL)


@pytest.mark.parametrize("radial", [False, True], ids=["axial", "radial"])
def test_arbitrary_axis_offset_pivot_reversed_axis_and_degree_inputs(radial):
    part = _part()
    part.kinematics["dofs"][0].update(unit="deg", range=[-720, 720])
    part.kinematics["bodies"][0]["motion"]["ops"] = [_rotate(axis=(1, 2, 3), origin=(3, 1, 2))]
    part.kinematics["bodies"][1]["motion"]["ops"] = [_rotate("r", (-1, -2, -3), (4, 3, 5))]
    blend = (
        {
            "kind": "radial_hermite",
            "params": {"axis": [2, 4, 6], "origin": [3, 1, 2], "r0": 0, "r1": 3},
        }
        if radial
        else {"kind": "axis_ramp", "params": {"axis": [-1, -2, -3], "lo": -5, "hi": 4}}
    )
    part.kinematics["flexures"][0]["blend"] = {"type": "field", **blend}
    ev = _compile(part)
    dofs = ev.to_evaluator_units(jnp.array([360.0, 0.7]))
    rest = jnp.array([[3.0, 1, 2], [1.0, -2, 1], [4.0, 3, 1]])
    posed = ev.pose_points(rest, dofs, jnp.full(3, 2))
    np.testing.assert_allclose(
        ev.inverse_flexure_points(posed, dofs, flexure="blade"), rest, atol=ATOL
    )


def test_inverse_jit_batching_and_derivatives_at_the_axis_and_zero_motion():
    ev = _compile(_part())
    inverse = jax.jit(lambda p, q: ev.inverse_flexure_points(p, q, flexure="blade"))
    for point in [jnp.zeros(3), jnp.array([1.0, 0.2, 0.3]), jnp.array([2.0, 0, 0])]:
        for dofs in [jnp.zeros(2), jnp.array([0.2, 0.9])]:
            jf = jax.jacfwd(inverse, argnums=0)(point, dofs)
            jr = jax.jacrev(inverse, argnums=0)(point, dofs)
            assert np.isfinite(jf).all() and np.isfinite(jr).all()
            np.testing.assert_allclose(jf, jr, atol=ATOL)
            assert np.isfinite(jax.jacrev(inverse, argnums=1)(point, dofs)).all()
    p = jnp.array([1.0, 0.2, 0.3])
    q = jnp.array([0.2, 0.9])

    def forward(p):
        return ev.pose_points(p, q, jnp.array(2))

    np.testing.assert_allclose(
        jax.jacfwd(inverse, argnums=0)(forward(p), q) @ jax.jacfwd(forward)(p), np.eye(3), atol=ATOL
    )
    assert jax.vmap(inverse, in_axes=(None, 0))(p, jnp.array([[0.0, 0], [0.3, 0.5]])).shape == (
        2,
        3,
    )
    assert inverse(jnp.empty((0, 3)), q).shape == (0, 3)
    assert inverse(jnp.ones((2, 4, 3)), q).shape == (2, 4, 3)


@pytest.mark.parametrize(
    "case", ["mixed_axes", "offset_blend", "transverse_blend", "translation", "multiple_ops"]
)
def test_unsupported_inverse_is_explicit_but_forward_motion_remains_available(case):
    part = _part()
    kin = part.kinematics
    if case == "mixed_axes":
        kin["bodies"][1]["motion"]["ops"][0]["axis"] = [1, 0, 0]
    elif case == "offset_blend":
        kin["flexures"][0]["blend"]["params"]["origin"] = [1, 0, 0]
    elif case == "transverse_blend":
        kin["flexures"][0]["blend"] = {
            "type": "field",
            "kind": "axis_ramp",
            "params": {"axis": [1, 0, 0], "lo": 0, "hi": 2},
        }
    elif case == "translation":
        kin["bodies"][0]["motion"]["ops"] = []
        kin["bodies"][1]["motion"]["ops"] = [
            {"kind": "translate", "axis": [0, 0, 1], "distance": {"type": "dof", "name": "r"}}
        ]
    else:
        kin["bodies"][0]["motion"]["ops"].append(_rotate("r"))
    ev = _compile(part)
    assert ev.inverse_flexure_names == ()
    assert np.isfinite(ev.pose_points(jnp.ones(3), jnp.array([0.3, 0.4]), jnp.array(2))).all()
    with pytest.raises(ValueError, match="no supported invariant-blend inverse"):
        ev.inverse_flexure_points(jnp.ones(3), jnp.zeros(2), flexure="blade")


@pytest.mark.parametrize("params", [{"axis": [0, 0, 0]}, {"r0": -1}, {"r0": 2, "r1": 1}, {"r1": 0}])
def test_invalid_radial_parameters_are_rejected(params):
    part = _part()
    part.kinematics["flexures"][0]["blend"]["params"].update(params)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        _compile(part)


def test_snapshot_unknown_names_and_conservative_bounds():
    part = _part()
    ev = _compile(part)
    part.kinematics["flexures"][0]["blend"]["params"]["r1"] = 100
    with pytest.raises(ValueError, match="Unknown flexure"):
        ev.inverse_flexure_points(jnp.ones(3), jnp.zeros(2), flexure="missing")
    bounds = infer_motion_bounds(_part(), include_flexures=True).flexures[0].bbox
    p = ev.pose_points(jnp.array([1.0, 0, 0]), jnp.array([0.0, math.pi]), jnp.array(2))
    np.testing.assert_allclose(p, [0, 1, 0], atol=ATOL)
    assert np.all(p >= np.array(bounds[0])) and np.all(p <= np.array(bounds[1]))


def test_minimal_0_4_fixture_exercises_radial_blend():
    path = (
        Path(__file__).parents[1]
        / "src/software_defined_matter/schema/conformance/valid/minimal_0.4.sdm"
    )
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == "0.4"
    assert _compile(Part.from_dict(doc)).inverse_flexure_names == ("blade",)


@pytest.mark.parametrize("ground", ["inner", "outer", "both"])
def test_ground_endpoints_keep_the_inverse_contract(ground):
    part = _part()
    for body in part.kinematics["bodies"]:
        if ground == "both" or body["name"] == ground:
            body["motion"]["ops"] = []
    ev = _compile(part)
    assert ev.inverse_flexure_names == ("blade",)
    q = jnp.array([0.7, -0.2])
    p = jnp.array([1.0, 0, 0.3])
    posed = ev.pose_points(p, q, jnp.array(2))
    np.testing.assert_allclose(ev.inverse_flexure_points(posed, q, flexure="blade"), p, atol=ATOL)


def test_radial_inner_clamp_matches_its_attachment():
    part = _part()
    part.kinematics["flexures"][0]["blend"]["params"]["r0"] = 0.5
    ev = _compile(part)
    p = jnp.array([0.25, 0, 0.3])
    q = jnp.array([0.7, -0.2])
    np.testing.assert_allclose(
        ev.pose_points(p, q, jnp.array(2)), ev.pose_points(p, q, jnp.array(0)), atol=ATOL
    )


def test_schema_index_identifies_the_new_blend_capability():
    from software_defined_matter.schema._generate import generate_index

    assert generate_index()["versions"]["0.4"]["added"] == [
        "kinematics.flexures.blend.radial_hermite"
    ]


def test_inverse_capability_is_per_flexure_and_does_not_enable_shader_emission():
    from software_defined_matter.glsl import emit_glsl

    part = _part()
    unsupported = copy.deepcopy(part.kinematics["flexures"][0])
    unsupported["name"] = "offset"
    unsupported["blend"]["params"]["origin"] = [2, 0, 0]
    part.kinematics["flexures"].insert(0, unsupported)
    ev = _compile(part)
    assert ev.inverse_flexure_names == ("blade",)
    p = jnp.array([1.0, 0, 0])
    q = jnp.array([0.2, 0.9])
    posed = ev.pose_points(p, q, jnp.array(3))
    np.testing.assert_allclose(ev.inverse_flexure_points(posed, q, flexure="blade"), p, atol=ATOL)
    with pytest.raises(ValueError, match="Flexure interpolation"):
        emit_glsl(part)
