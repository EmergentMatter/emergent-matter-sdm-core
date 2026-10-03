"""Rigid motion follows authored order, units and rest-region ownership."""

from __future__ import annotations

import copy
import math

import jax
import jax.numpy as jnp
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
from software_defined_matter.dsl.expr import (
    expr_binop,
    expr_num,
    expr_param,
    expr_reduce,
    expr_unop,
)
from software_defined_matter.dsl.resolve import make_binding
from software_defined_matter.kinematics import compile_kinematics, resolve_region_tree


def _dof(name="q", unit="rad", default=0):
    return {
        "name": name,
        "kind": "length" if unit == "mm" else "angle",
        "range": [-180, 180],
        "unit": unit,
        "default": default,
    }


def _read(name="q"):
    return {"type": "dof", "name": name}


def _rotate(angle=None, axis=(0, 0, 1), origin=(0, 0, 0)):
    return {
        "kind": "rotate",
        "axis": list(axis),
        "origin": list(origin),
        "angle": _read() if angle is None else angle,
    }


def _translate(distance=None, axis=(1, 0, 0)):
    return {
        "kind": "translate",
        "axis": list(axis),
        "distance": _read() if distance is None else distance,
    }


def _part(ops=None, dofs=None):
    sphere = sdf_primitive("sphere", r=1.0)
    sphere["name"] = "ball"
    return Part(
        name="rigid",
        materials=[MaterialRegion(material_id=1, name="solid", sdf_tree=sphere)],
        kinematics={
            "dofs": [_dof()] if dofs is None else dofs,
            "bodies": [
                {
                    "name": "body",
                    "region": {"$node": "ball"},
                    "motion": {"ops": [_rotate()] if ops is None else ops},
                }
            ],
        },
    )


def _compile(part):
    result = compile_kinematics(part)
    assert result is not None
    return result


def test_absent_empty_and_fixed_motion():
    assert compile_kinematics(Part(name="static")) is None
    empty = _compile(Part(name="empty", kinematics={}))
    np.testing.assert_array_equal(empty.ownership([[1, 2, 3]]), [-1])
    np.testing.assert_array_equal(empty.pose_points([[1, 2, 3]], []), [[1, 2, 3]])
    fixed = _compile(_part(ops=[], dofs=[]))
    np.testing.assert_array_equal(fixed.pose_points([[1, 2, 3]], []), [[1, 2, 3]])


def test_rotation_sign_origin_normalization_and_inverse():
    evaluator = _compile(_part(ops=[_rotate(axis=(0, 0, 5), origin=(2, 3, 0))]))
    points = np.array([[3, 3, 0], [2, 4, 0]], dtype=float)
    posed = evaluator.pose_points(points, [math.pi / 2])
    np.testing.assert_allclose(posed, [[2, 4, 0], [1, 3, 0]], atol=1e-6)
    matrix = np.asarray(evaluator.body_transforms([math.pi / 2]))[0]
    recovered = (np.asarray(posed) - matrix[:3, 3]) @ matrix[:3, :3]
    np.testing.assert_allclose(recovered, points, atol=1e-6)
    np.testing.assert_array_equal(evaluator.pose_points(points, [0]), points)


@pytest.mark.parametrize("reverse,expected", [(False, [0, 2, 0]), (True, [1, 1, 0])])
def test_operation_order_is_noncommutative(reverse, expected):
    ops = [_translate(_read("slide"), axis=(4, 0, 0)), _rotate()]
    if reverse:
        ops.reverse()
    evaluator = _compile(_part(ops=ops, dofs=[_dof(), _dof("slide", "mm")]))
    np.testing.assert_allclose(
        evaluator.pose_points([1, 0, 0], [math.pi / 2, 1]), expected, atol=1e-6
    )


def test_chained_rotations_use_fixed_rest_frame_origins():
    evaluator = _compile(
        _part(
            ops=[_rotate(_read("a")), _rotate(_read("b"), origin=(1, 0, 0))],
            dofs=[_dof("a"), _dof("b")],
        )
    )
    # (2,0) -> (0,2) -> (-1,-1), rotating about the listed world origins.
    np.testing.assert_allclose(
        evaluator.pose_points([2, 0, 0], [math.pi / 2, math.pi / 2]), [-1, -1, 0], atol=1e-6
    )


def test_oblique_rotation_matches_independent_scipy_oracle_over_multiple_poses():
    from scipy.spatial.transform import Rotation

    axis = np.array([1.0, 2.0, -3.0])
    origin = np.array([10.0, -20.0, 30.0])
    points = np.random.default_rng(42).uniform(-100, 100, (8, 3))
    evaluator = _compile(_part(ops=[_rotate(axis=axis, origin=origin)]))
    angles = np.linspace(-math.pi, math.pi, 13)
    expected = np.stack(
        [
            Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).apply(points - origin)
            + origin
            for angle in angles
        ]
    )
    actual = jax.jit(jax.vmap(lambda q: evaluator.pose_points(points, q)))(angles[:, None])
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=1e-6)


def test_degrees_convert_before_nonlinear_expression_evaluation():
    evaluator = _compile(
        _part(ops=[_rotate(expr_unop("square", _read()))], dofs=[_dof(unit="deg", default=90)])
    )
    dofs = evaluator.to_evaluator_units([90])
    np.testing.assert_allclose(dofs, [math.pi / 2], atol=1e-6)
    assert evaluator.dof_defaults == pytest.approx((math.pi / 2,))
    assert evaluator.dof_ranges[0] == pytest.approx((-math.pi, math.pi))
    theta = (math.pi / 2) ** 2
    np.testing.assert_allclose(
        evaluator.pose_points([1, 0, 0], dofs), [math.cos(theta), math.sin(theta), 0], atol=1e-6
    )


def test_design_parameters_are_distinct_from_dofs_and_snapshot_derived_values():
    angle = expr_reduce("sum", [expr_binop("*", _read(), expr_param("gain")), expr_num(0)])
    part = _part(ops=[_rotate(angle)])
    part.params = {
        "q": Param("q", 2, unit="ratio", free=False),
        "gain": Param(
            "gain",
            999,
            unit="ratio",
            free=False,
            expr=expr_binop("*", expr_param("q"), expr_num(2)),
        ),
    }
    original = copy.deepcopy(part.to_dict())
    evaluator = _compile(part)
    np.testing.assert_allclose(
        evaluator.pose_points([1, 0, 0], [math.pi / 8]), [0, 1, 0], atol=1e-6
    )
    assert part.to_dict() == original
    part.params["q"].value = 4
    part.kinematics["bodies"][0]["motion"]["ops"] = []
    np.testing.assert_allclose(
        evaluator.pose_points([1, 0, 0], [math.pi / 8]), [0, 1, 0], atol=1e-6
    )


def test_jit_vmap_and_grad_match_analytic_nonzero_pose():
    evaluator = _compile(_part())
    points = jnp.array([[2.0, 0.0, 0.0]])
    owner = evaluator.ownership(points)
    pose = jax.jit(lambda q: evaluator.pose_points(points, q, owner))
    np.testing.assert_allclose(
        pose(jnp.array([0.3])), [[2 * math.cos(0.3), 2 * math.sin(0.3), 0]], atol=1e-6
    )
    derivative = jax.grad(lambda q: pose(q)[0, 1])(jnp.array([0.3]))
    np.testing.assert_allclose(derivative, [2 * math.cos(0.3)], atol=1e-6)
    at_zero = jax.grad(lambda q: pose(q)[0, 1])(jnp.array([0.0]))
    np.testing.assert_allclose(at_zero, [2], atol=1e-6)
    batched = jax.vmap(pose)(jnp.array([[0.0], [0.3]]))
    assert batched.shape == (2, 1, 3)
    assert jnp.isfinite(batched).all()
    jacobian = jax.jacfwd(lambda p: evaluator.pose_points(p, [0.3], jnp.array(0)))(
        jnp.array([2.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(jacobian.T @ jacobian, np.eye(3), atol=1e-6)


def test_nearest_region_ownership_is_cached_in_rest_space_and_ties_use_order():
    part = _part(ops=[_translate()], dofs=[_dof(unit="mm")])
    second = sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[4, 0, 0])
    part.kinematics["bodies"].append({"name": "fixed", "region": second, "motion": {"ops": []}})
    evaluator = _compile(part)
    points = np.array([[0.0, 0, 0], [4, 0, 0], [2, 0, 0]])
    np.testing.assert_array_equal(evaluator.ownership(points), [0, 1, 0])
    np.testing.assert_allclose(
        evaluator.pose_points(points, [10]), [[10, 0, 0], [4, 0, 0], [12, 0, 0]]
    )
    np.testing.assert_allclose(
        evaluator.pose_points(
            points.reshape(1, 3, 3), [10], evaluator.ownership(points).reshape(1, 3)
        ),
        np.array([[[10, 0, 0], [4, 0, 0], [12, 0, 0]]]),
    )
    assert evaluator.pose_points(np.empty((0, 3)), [0]).shape == (0, 3)


def test_named_region_keeps_ancestors_but_excludes_csg_siblings():
    part = _part(ops=[])
    ball = part.materials[0].sdf_tree
    sibling = sdf_primitive("sphere", r=3.0)
    part.materials[0].sdf_tree = sdf_transform(
        "translate", sdf_op("union", [ball, sibling]), t=[4, 0, 0]
    )
    resolved = resolve_region_tree(part, {"$node": "ball"})
    assert resolved["child"] == ball
    assert resolved["params"]["t"] == [4, 0, 0]
    evaluator = _compile(part)
    np.testing.assert_allclose(evaluator._regions[0](np.array([[4.0, 0, 0], [0, 0, 0]])), [-1, 3])
    resolved["child"]["params"]["r"] = 99
    assert ball["params"]["r"] == 1


@pytest.mark.parametrize("invalid", ["duplicate", "missing"])
def test_named_regions_must_resolve_uniquely(invalid):
    part = _part()
    if invalid == "duplicate":
        ball = part.materials[0].sdf_tree
        part.materials[0].sdf_tree = sdf_op("union", [ball, copy.deepcopy(ball)])
    else:
        part.kinematics["bodies"][0]["region"] = {"$node": "absent"}
    with pytest.raises(ValueError, match="exactly one|unknown node"):
        _compile(part)


def test_flexures_are_not_silently_ignored():
    part = _part()
    part.kinematics["flexures"] = [
        {
            "name": "flex",
            "region": {"$node": "ball"},
            "from_body": "body",
            "to_body": "body",
            "blend": {
                "type": "field",
                "kind": "axis_ramp",
                "params": {"axis": [0, 0, 1], "lo": -1, "hi": 1},
            },
        }
    ]
    evaluator = _compile(part)
    assert evaluator.region_names == ("body", "flex")
    point = jnp.array([[1.0, 2.0, 0.5]])
    np.testing.assert_allclose(
        evaluator.pose_points(point, jnp.array([0.4]), owner=jnp.array([1])),
        evaluator.pose_points(point, jnp.array([0.4]), owner=jnp.array([0])),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "bad", ["zero_axis", "nan_axis", "nan_origin", "unit", "missing_dof", "metric", "rest_offset"]
)
def test_invalid_motion_fails_during_compilation(bad):
    part = _part()
    op = part.kinematics["bodies"][0]["motion"]["ops"][0]
    if bad == "zero_axis":
        op["axis"] = [0, 0, 0]
    elif bad == "nan_axis":
        op["axis"] = [0, float("nan"), 1]
    elif bad == "nan_origin":
        op["origin"] = [0, float("nan"), 0]
    elif bad == "unit":
        part.kinematics["dofs"][0]["unit"] = "mm"
    elif bad == "missing_dof":
        op["angle"] = _read("missing")
    elif bad == "metric":
        op["angle"] = {"type": "metric", "name": "volume"}
    elif bad == "rest_offset":
        op["angle"] = expr_num(1)
    with pytest.raises(ValueError):
        _compile(part)


def test_motion_dof_nodes_do_not_change_geometry_expression_semantics():
    from software_defined_matter.dsl.expr import eval_expr_pure

    binding = make_binding(_part())
    with pytest.raises(ValueError, match="Unknown expression node type"):
        eval_expr_pure(_read(), binding, binding.initial_free_vector())


@pytest.mark.parametrize(
    "points,dofs,owner",
    [([1, 2], [0], None), ([1, 2, 3], [], None), ([[1, 2, 3]], [0], [0.5]), ([[1, 2, 3]], [0], 0)],
)
def test_bad_input_shapes_or_ownership_types_fail(points, dofs, owner):
    with pytest.raises(ValueError):
        _compile(_part()).pose_points(points, dofs, owner)


def test_invalid_cached_owner_produces_nan_even_under_jit():
    evaluator = _compile(_part())
    result = jax.jit(evaluator.pose_points)(
        jnp.array([[1.0, 0.0, 0.0]]), jnp.array([0.0]), jnp.array([99])
    )
    assert jnp.isnan(result).all()
