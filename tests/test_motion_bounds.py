"""Motion envelopes contain intermediate poses without enumerating DOF combinations."""

from __future__ import annotations

import copy
import math

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
from software_defined_matter.kinematics import KinematicsEval, compile_kinematics
from software_defined_matter.motion_bounds import infer_motion_bounds
from software_defined_matter.sdf.bbox import BBoxInferenceError, _bbox_corners


def _read(name="q"):
    return {"type": "dof", "name": name}


def _dof(name="q", unit="rad", bounds=(-1, 1)):
    return {
        "name": name,
        "kind": "length" if unit == "mm" else "angle",
        "unit": unit,
        "range": list(bounds),
        "default": 0,
    }


def _rotate(expr=None, axis=(0, 0, 1), origin=(0, 0, 0)):
    return {
        "kind": "rotate",
        "axis": list(axis),
        "origin": list(origin),
        "angle": _read() if expr is None else expr,
    }


def _translate(expr=None, axis=(1, 0, 0)):
    return {"kind": "translate", "axis": list(axis), "distance": _read() if expr is None else expr}


def _part(ops=None, dofs=None, region=None):
    tree = region or sdf_transform("translate", sdf_primitive("sphere", r=0.1), t=[4, 0, 0])
    tree["name"] = "region"
    return Part(
        name="motion bounds",
        materials=[MaterialRegion(material_id=1, name="solid", sdf_tree=tree)],
        kinematics={
            "dofs": [_dof()] if dofs is None else dofs,
            "bodies": [
                {
                    "name": "body",
                    "region": {"$node": "region"},
                    "motion": {"ops": [_rotate()] if ops is None else ops},
                }
            ],
        },
    )


def _contains(box, points):
    assert np.all(points >= np.asarray(box[0]))
    assert np.all(points <= np.asarray(box[1]))


def test_absent_empty_and_fixed_motion_bounds():
    assert infer_motion_bounds(Part(name="static")) is None
    empty = infer_motion_bounds(Part(name="empty", kinematics={}))
    assert empty.bbox is None and empty.bodies == ()
    result = infer_motion_bounds(_part(ops=[], dofs=[]))
    _contains(result.bbox, _bbox_corners(result.bodies[0].rest_bbox))
    assert result.bodies[0].dof_names == ()


def test_rotation_includes_extrema_between_range_endpoints():
    part = _part(dofs=[_dof(unit="deg", bounds=(-60, 60))])
    result = infer_motion_bounds(part)
    expected = math.hypot(4.1, 0.1)
    assert result.bbox[1][0] == pytest.approx(expected, abs=1e-4)
    _contains(result.bbox, np.array([[4.1, 0, 0]]))


def test_large_angles_cover_float32_input_rounding():
    dof = _dof(bounds=(100000001, 100000002))
    dof["default"] = 100000001
    part = _part(dofs=[dof])
    bounds = infer_motion_bounds(part)
    evaluator = compile_kinematics(part)
    points = np.asarray(_bbox_corners(bounds.bodies[0].rest_bbox))
    posed = np.asarray(evaluator.pose_points(points, [100000001]))
    _contains(bounds.bbox, posed)


@pytest.mark.parametrize("literal_only", [False, True], ids=["input", "literals"])
def test_expression_bounds_include_float32_cancellation(literal_only):
    constant = {"type": "num", "value": 1e8}
    expr = {
        "type": "binop",
        "op": "-",
        "lhs": {
            "type": "binop",
            "op": "+",
            "lhs": constant,
            "rhs": _read(),
        },
        "rhs": constant,
    }
    if literal_only:
        expr = {
            "type": "binop",
            "op": "*",
            "lhs": _read(),
            "rhs": {
                "type": "binop",
                "op": "-",
                "lhs": {"type": "num", "value": 100000004},
                "rhs": constant,
            },
        }
    dof = _dof(unit="mm", bounds=(1, 2))
    dof["default"] = 1
    part = _part(ops=[_translate(expr)], dofs=[dof])
    bounds = infer_motion_bounds(part)
    evaluator = compile_kinematics(part)
    points = np.asarray(_bbox_corners(bounds.bodies[0].rest_bbox))
    _contains(bounds.bbox, np.asarray(evaluator.pose_points(points, [1])))


@pytest.mark.parametrize("turns", [1, 3, 100], ids=str)
def test_full_turns_cover_the_same_rotation_cylinder(turns):
    part = _part(dofs=[_dof(bounds=(-turns * math.pi, turns * math.pi))])
    box = infer_motion_bounds(part).bbox
    radius = math.hypot(4.1, 0.1)
    np.testing.assert_allclose(
        box, [[-radius, -radius, -0.1], [radius, radius, 0.1]], atol=1e-4, rtol=0
    )


def test_oblique_chains_contain_sampled_corner_poses_and_do_not_mutate_input():
    part = _part(
        ops=[
            _rotate(axis=(1, 2, -3), origin=(1, -2, 3)),
            _translate(_read("slide"), axis=(2, -1, 1)),
            _rotate(_read("other"), axis=(-1, 3, 2), origin=(-2, 1, 0)),
        ],
        dofs=[_dof(), _dof("slide", "mm", (-2, 3)), _dof("other", "deg", (-30, 80))],
    )
    before = copy.deepcopy(part.to_dict())
    bounds = infer_motion_bounds(part)
    evaluator = compile_kinematics(part)
    points = np.asarray(_bbox_corners(bounds.bodies[0].rest_bbox))
    rng = np.random.default_rng(42)
    for values in rng.uniform([-1, -2, -30], [1, 3, 80], (31, 3)):
        posed = np.asarray(evaluator.pose_points(points, evaluator.to_evaluator_units(values)))
        _contains(bounds.bodies[0].bbox, posed)
    assert part.to_dict() == before


def test_nonlinear_degree_expression_uses_radians_before_squaring():
    expr = {"type": "unop", "op": "square", "child": _read()}
    part = _part(ops=[_translate(expr)], dofs=[_dof(unit="deg", bounds=(-90, 90))])
    bounds = infer_motion_bounds(part)
    assert bounds.bbox[1][0] == pytest.approx(4.1 + (math.pi / 2) ** 2, abs=1e-4)
    assert bounds.bbox[0][0] == pytest.approx(3.9, abs=1e-4)


@pytest.mark.parametrize("kind", ["exp", "log", "sqrt", "power", "variable-power", "reduce"])
def test_nonlinear_expression_ranges_contain_evaluator_poses(kind):
    two = {"type": "num", "value": 2}
    base = {"type": "binop", "op": "+", "lhs": _read(), "rhs": two}

    def expression(child):
        if kind in {"exp", "log", "sqrt"}:
            return {"type": "unop", "op": kind, "child": child}
        if kind in {"power", "variable-power"}:
            return {
                "type": "binop",
                "op": "pow",
                "lhs": child,
                "rhs": two if kind == "power" else child,
            }
        return {"type": "reduce", "op": "mean", "children": [child, two]}

    expr = {"type": "binop", "op": "-", "lhs": expression(base), "rhs": expression(two)}
    part = _part(ops=[_translate(expr)])
    bounds = infer_motion_bounds(part)
    evaluator = compile_kinematics(part)
    points = np.asarray(_bbox_corners(bounds.bodies[0].rest_bbox))
    for value in np.linspace(-1, 1, 5):
        _contains(bounds.bbox, np.asarray(evaluator.pose_points(points, [value])))


def test_derived_design_ranges_enclose_geometry_and_motion_independently_of_cached_values():
    part = _part(
        ops=[
            _translate(
                {
                    "type": "binop",
                    "op": "*",
                    "lhs": _read(),
                    "rhs": {"type": "param", "name": "gain"},
                }
            )
        ],
        dofs=[_dof(unit="mm")],
        region=sdf_primitive("sphere", r={"$ref": "gain"}),
    )
    part.params = {
        "q": Param("q", 1, unit="mm", bounds=(1, 2)),
        "gain": Param(
            "gain",
            999,
            unit="mm",
            expr={
                "type": "binop",
                "op": "*",
                "lhs": {"type": "param", "name": "q"},
                "rhs": {"type": "num", "value": 2},
            },
        ),
        "unused": Param("unused", 0, unit="ratio", free=True),
    }
    result = infer_motion_bounds(part)
    assert result.bodies[0].dof_names == ("q",)
    _contains(result.bbox, np.array([[-8, 0, 0], [8, 0, 0], [0, 4, 0]]))
    assert result.bbox[1][0] < 8.001


def test_positive_scale_range_covers_near_and_far_rest_positions():
    part = _part(
        ops=[],
        dofs=[],
        region=sdf_transform(
            "scale",
            sdf_transform("translate", sdf_primitive("sphere", r=1), t=[10, 0, 0]),
            s={"$ref": "size"},
        ),
    )
    part.params = {"size": Param("size", 1, unit="ratio", bounds=(1, 2))}
    _contains(infer_motion_bounds(part).bbox, np.array([[9, 0, 0], [22, 0, 0]]))


def test_body_dependencies_do_not_form_a_global_product(monkeypatch):
    import software_defined_matter.motion_bounds as module

    part = _part(dofs=[_dof(f"q{i}") for i in range(97)])
    prototype = part.kinematics["bodies"][0]
    part.kinematics["bodies"] = [
        dict(
            copy.deepcopy(prototype),
            name=f"body{i}",
            motion={"ops": [_rotate(_read(f"q{i}")), _rotate(_read("q96"))]},
        )
        for i in range(64)
    ]

    def no_sampling(*args):
        pytest.fail("Motion bounds must not enumerate evaluator poses")

    monkeypatch.setattr(KinematicsEval, "body_transforms", no_sampling)
    calls = 0
    sweep = module._sweep

    def counted(*args):
        nonlocal calls
        calls += 1
        return sweep(*args)

    monkeypatch.setattr(module, "_sweep", counted)
    result = infer_motion_bounds(part)
    assert calls == 128
    assert result.bodies[3].dof_names == ("q3", "q96")
    assert len(result.bodies) == 64


def test_shader_automatically_publishes_body_and_scene_envelopes():
    part = _part(ops=[_translate()], dofs=[_dof(unit="mm", bounds=(-10, 10))])
    emission = emit_glsl(part)
    _contains(emission.bbox, np.array([[-6.1, 0, 0], [14.1, 0, 0]]))
    _contains(emission.components[0]["bbox"], np.array([[14.1, 0, 0]]))
    part.metadata["bbox"] = [[-1, -1, -1], [1, 1, 1]]
    enlarged = emit_glsl(part)
    _contains(enlarged.bbox, _bbox_corners(emission.bbox))
    _contains(enlarged.bbox, np.array([[-1, -1, -1], [1, 1, 1]]))


def test_uninferable_rest_field_requires_an_authored_full_motion_box():
    region = sdf_op(
        "smooth_union", [sdf_primitive("sphere", r=1), sdf_primitive("sphere", r=1)], k=0.5
    )
    part = _part(region=region)
    with pytest.raises(BBoxInferenceError, match="Supply metadata.bbox"):
        infer_motion_bounds(part)
    part.metadata["bbox"] = [[-10, -10, -10], [10, 10, 10]]
    result = infer_motion_bounds(part)
    assert result.bodies[0].bbox is None
    assert result.bbox == ((-10, -10, -10), (10, 10, 10))
    assert emit_glsl(part).components[0]["bbox"] is None


@pytest.mark.parametrize(
    "expr",
    [
        {"type": "binop", "op": "pow", "lhs": _read(), "rhs": {"type": "num", "value": -2}},
        {"type": "binop", "op": "/", "lhs": {"type": "num", "value": 1}, "rhs": _read()},
        {"type": "unop", "op": "log", "child": _read()},
    ],
    ids=["negative-power", "division", "log"],
)
def test_singular_motion_ranges_fail_even_with_an_authored_box(expr):
    part = _part(ops=[_translate(expr)])
    part.metadata["bbox"] = [[-10, -10, -10], [10, 10, 10]]
    with pytest.raises(BBoxInferenceError, match="Cannot bound motion expression"):
        infer_motion_bounds(part)
