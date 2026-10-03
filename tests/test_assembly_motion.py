"""Bound coordinates drive occurrence geometry and diagnose invalid motion without solving it."""

from __future__ import annotations

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    Assembly,
    Dof,
    Frame,
    Instance,
    Param,
    PartRef,
    PlacementError,
    compile_placement,
)
from tests.test_assembly_placement import _link, _load, _pair
from tests.test_dynamic_kinematics import _design_part

ATOL = 2e-5  # Float32 motion composition over nested occurrences.


def _read(name):
    return {"type": "dof", "name": name}


def _product(lhs, rhs):
    return {"type": "binop", "op": "*", "lhs": lhs, "rhs": rhs}


def _nested(tmp_path):
    inner = replace(_pair("revolute"), dofs={"q": Dof("angle", (-180, 180), "deg", 90)})
    root = Assembly(
        "root",
        params={"ratio": Param("ratio", 2, free=True, unit="count")},
        dofs={"drive": Dof("angle", (-180, 180), "deg", 45)},
        instances=(
            Instance(
                "a",
                PartRef("inner.sdm"),
                transform=Frame(),
                dof_bindings={"q": _product(_read("b.q"), {"type": "param", "name": "ratio"})},
            ),
            Instance(
                "b",
                PartRef("inner.sdm"),
                transform=Frame(position=(10, 0, 0)),
                dof_bindings={"q": _read("drive")},
            ),
        ),
        motion_inputs={"alias": "a.q"},
    )
    return _load(tmp_path, root, inner=inner, link=_link())


def test_dependency_order_and_degree_targets_use_evaluator_units(tmp_path):
    evaluator = compile_placement(_nested(tmp_path))
    assert evaluator.dof_names == ("drive",)
    np.testing.assert_allclose(evaluator.dof_ranges, [[-math.pi, math.pi]], atol=ATOL)
    state = evaluator.evaluate_checked()
    assert float(state.coordinates["a.q"]) == pytest.approx(math.pi / 2, abs=ATOL)
    assert float(state.coordinates["b.q"]) == pytest.approx(math.pi / 4, abs=ATOL)
    np.testing.assert_allclose(state.ports["a.output"][:3, 3], [4, 2, 0], atol=ATOL)
    np.testing.assert_allclose(
        state.ports["b.output"][:3, 3], [14 + math.sqrt(2), math.sqrt(2), 0], atol=ATOL
    )
    with pytest.raises(ValueError, match="Expected DOF shape"):
        evaluator.evaluate(dofs=[0, 0, 0])


def test_motion_and_design_gradients_flow_through_bindings_and_mates(tmp_path):
    evaluator = compile_placement(_nested(tmp_path))

    def position(design, q):
        return evaluator.evaluate(free_vec=design, dofs=q).ports["a.output"][:3, 3]

    design, q = jnp.array([1.3]), jnp.array([0.4])
    compiled = jax.jit(position)
    step = 1e-3
    for arg in (0, 1):
        inputs = [design, q]
        plus, minus = inputs.copy(), inputs.copy()
        plus[arg], minus[arg] = inputs[arg] + step, inputs[arg] - step
        finite = (compiled(*plus) - compiled(*minus)) / (2 * step)
        np.testing.assert_allclose(
            jax.jacfwd(position, argnums=arg)(design, q)[:, 0], finite, atol=3e-4
        )
    states = jax.jit(jax.vmap(lambda values: evaluator.evaluate(dofs=values)))(
        jnp.array([[0.0], [math.pi / 4], [math.pi / 2]])
    )
    assert np.asarray(states.valid).all()
    np.testing.assert_allclose(
        states.ports["a.output"][:, :3, 3], [[6, 0, 0], [4, 2, 0], [2, 0, 0]], atol=ATOL
    )


def test_promoted_bindings_use_parent_parameter_scope_and_snapshot(tmp_path):
    child = Assembly(
        "child",
        params={"gain": Param("gain", 3, unit="count")},
        dofs={"drive": Dof("angle", (-4, 4), "rad")},
        instances=(
            Instance(
                "pair",
                PartRef("pair.sdm"),
                transform=Frame(),
                dof_bindings={"q": _product(_read("drive"), {"type": "param", "name": "gain"})},
            ),
        ),
        motion_inputs={"input": "pair.q"},
    )
    root = Assembly(
        "root",
        params={"gain": Param("gain", 2, free=True, unit="count")},
        dofs={"q": Dof("angle", (-4, 4), "rad", 0.2)},
        instances=(
            Instance(
                "child",
                PartRef("child.sdm"),
                {"gain": {"$ref": "gain"}},
                {"drive": _read("q")},
                Frame(),
            ),
        ),
    )
    bundle = _load(tmp_path, root, child=child, pair=_pair("revolute"), link=_link())
    evaluator = compile_placement(bundle)
    bundle.root.params["gain"].value = 99
    state = evaluator.evaluate_checked()
    assert float(state.coordinates["child.pair.q"]) == pytest.approx(0.4, abs=ATOL)
    assert float(evaluator.evaluate_checked(free_vec=[3]).coordinates["child.pair.q"]) == (
        pytest.approx(0.6, abs=ATOL)
    )
    conflict = replace(root.instances[0], dof_bindings={"input": _read("q")})
    with pytest.raises(ValueError, match="conflicting drivers"):
        _load(
            tmp_path,
            replace(root, instances=(conflict,)),
            child=child,
            pair=_pair("revolute"),
            link=_link(),
        )


def _flexure_assembly(tmp_path):
    root = Assembly(
        "root",
        params={"gain": Param("gain", 2, free=True, unit="count")},
        dofs={"turn": Dof("angle", (-4, 4), "rad", 0.2)},
        instances=(
            Instance(
                "flexure",
                PartRef("flexure.sdm"),
                {"gain": {"$ref": "gain"}},
                {"q": _read("turn"), "r": {"type": "num", "value": 0}},
                Frame.from_axis_angle((0, 1, 0), math.pi / 2, position=(5, 3, 0)),
            ),
        ),
    )
    return compile_placement(_load(tmp_path, root, flexure=_design_part()))


def test_world_bodies_ports_and_flexure_points_share_bound_motion(tmp_path):
    evaluator = _flexure_assembly(tmp_path)
    state = evaluator.evaluate_checked()
    rest = jnp.array([2.0, 0, 1])
    tip = evaluator.pose_points_checked("flexure", rest, owner=jnp.array(1))
    np.testing.assert_allclose(tip, state.ports["flexure.tip"][:3, 3], atol=ATOL)
    np.testing.assert_allclose(
        tip, (state.bodies["flexure.to"] @ jnp.append(rest, 1))[:3], atol=ATOL
    )
    midpoint = jnp.array([1.0, 0, 0.5])

    def pose(q):
        return evaluator.pose_points("flexure", midpoint, dofs=q, owner=jnp.array(2))

    np.testing.assert_allclose(
        jax.jit(pose)(jnp.array([0.2])), [5.5, 3 + math.sin(0.4), -math.cos(0.4)], atol=ATOL
    )
    derivative = jax.jacfwd(pose)(jnp.array([0.2]))[:, 0]
    np.testing.assert_allclose(derivative, [0, 2 * math.cos(0.4), 2 * math.sin(0.4)], atol=ATOL)
    assert np.isfinite(evaluator.pose_points_checked("flexure", rest)).all()


def test_point_validation_rejects_bad_ownership_shapes_and_occurrences(tmp_path):
    evaluator = _flexure_assembly(tmp_path)
    for owner in (-1, 99):
        with pytest.raises(PlacementError, match="flexure.*non-finite motion or ownership"):
            evaluator.pose_points_checked("flexure", [1, 0, 0.5], owner=jnp.array(owner))
    with pytest.raises(ValueError, match="Expected a leaf part occurrence"):
        evaluator.pose_points("", [0, 0, 0])
    with pytest.raises(ValueError, match="point batch shape"):
        evaluator.pose_points("flexure", [0, 0, 0], owner=jnp.array([0]))
    with pytest.raises(PlacementError, match="non-finite"):
        evaluator.pose_points_checked("flexure", [float("nan"), 0, 0])


def test_nonfinite_bound_coordinates_fail_even_without_ports_or_mates(tmp_path):
    child = Assembly("child", dofs={"q": Dof("angle", (-4, 4), "rad")})
    root = Assembly(
        "root",
        dofs={"q": Dof("angle", (-4, 4), "rad")},
        instances=(
            Instance(
                "child",
                PartRef("child.sdm"),
                transform=Frame(),
                dof_bindings={
                    "q": {"type": "binop", "op": "/", "lhs": _read("q"), "rhs": _read("q")}
                },
            ),
        ),
    )
    evaluator = compile_placement(_load(tmp_path, root, child=child))
    assert not bool(jax.jit(lambda q: evaluator.evaluate(dofs=q))(jnp.array([0.0])).valid)
    with pytest.raises(PlacementError, match="DOF 'child.q': non-finite"):
        evaluator.evaluate_checked()
    assert bool(evaluator.evaluate_checked(dofs=[1]).valid)


def test_internal_body_with_no_port_is_still_validated(tmp_path):
    part = _design_part()
    part.ports = []
    part.kinematics["bodies"][1]["motion"]["ops"][0]["angle"] = {
        "type": "unop",
        "op": "sqrt",
        "child": _read("q"),
    }
    root = Assembly("root", instances=(Instance("part", PartRef("part.sdm"), transform=Frame()),))
    evaluator = compile_placement(_load(tmp_path, root, part=part))
    with pytest.raises(PlacementError, match="body 'part.to': non-finite"):
        evaluator.evaluate_checked(dofs=[-1, 0])


def test_static_points_follow_placement_and_invalid_closure_is_not_rendered(tmp_path):
    evaluator = compile_placement(_load(tmp_path, _pair(), link=_link()))
    np.testing.assert_allclose(
        evaluator.pose_points_checked("arm", [[0, 0, 0], [1, 0, 0]]),
        [[4, 0, 0], [5, 0, 0]],
        atol=ATOL,
    )
    with pytest.raises(ValueError, match="no kinematic ownership regions"):
        evaluator.pose_points("arm", [0, 0, 0], owner=0)
    root = _pair()
    root = replace(
        root, instances=(root.instances[0], replace(root.instances[1], transform=Frame()))
    )
    invalid = compile_placement(_load(tmp_path, root, link=_link()))
    assert np.isnan(invalid.pose_points("arm", [0, 0, 0])).all()
    with pytest.raises(PlacementError, match="joint: position"):
        invalid.pose_points_checked("arm", [0, 0, 0])


def test_motion_conformance_gate_uses_resolved_coordinates_and_world_landmarks():
    import json
    from pathlib import Path

    from software_defined_matter import load_bundle

    corpus = Path(__file__).parents[1] / "src/software_defined_matter/schema/conformance"
    evaluator = compile_placement(load_bundle(corpus / "valid/assembly_motion_0.5.sdm"))
    expected = json.loads((corpus / "placement/assembly_motion_0.5.expected.json").read_text())
    assert list(evaluator.dof_names) == expected["dof_names"]
    assert list(evaluator.dof_units) == expected["dof_units"]
    for case in expected["cases"]:
        state = evaluator.evaluate_checked(
            free_vec=case["design"], dofs=evaluator.to_evaluator_units(case["authored_dofs"])
        )
        for name, value in case["coordinates"].items():
            assert float(state.coordinates[name]) == pytest.approx(value, abs=ATOL)
        np.testing.assert_allclose(
            state.instances["mechanism.moving"], case["moving_world"], atol=ATOL
        )
        landmark = evaluator.pose_points_checked(
            "mechanism.moving",
            expected["landmark_local"],
            free_vec=case["design"],
            dofs=evaluator.to_evaluator_units(case["authored_dofs"]),
        )
        np.testing.assert_allclose(landmark, case["landmark_world"], atol=ATOL)


@pytest.mark.parametrize(
    "expression, message",
    [
        ({"type": "dof", "name": "missing"}, "Unknown DOF"),
        ({"type": "param", "name": "missing"}, "unknown parameters"),
        ({"type": "metric", "name": "volume"}, "pure parameter expression"),
        ({"type": "dof", "name": "child.q"}, "cycle"),
    ],
    ids=["unknown_motion", "unknown_design", "metric", "self_cycle"],
)
def test_motion_binding_validation_rejects_unresolvable_or_impure_drivers(
    tmp_path, expression, message
):
    child = Assembly("child", dofs={"q": Dof("angle", (-4, 4), "rad")})
    root = Assembly(
        "root",
        instances=(
            Instance(
                "child", PartRef("child.sdm"), dof_bindings={"q": expression}, transform=Frame()
            ),
        ),
    )
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, root, child=child)


def test_empty_kinematics_has_no_ownership_regions(tmp_path):
    from software_defined_matter import Part

    part = Part("empty", kinematics={"dofs": [], "bodies": [], "flexures": []})
    root = Assembly("root", instances=(Instance("part", PartRef("part.sdm"), transform=Frame()),))
    evaluator = compile_placement(_load(tmp_path, root, part=part))
    np.testing.assert_allclose(
        evaluator.pose_points_checked("part", [1, 2, 3]), [1, 2, 3], atol=ATOL
    )
    with pytest.raises(ValueError, match="no kinematic ownership regions"):
        evaluator.pose_points("part", [1, 2, 3], owner=0)
