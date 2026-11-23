"""Shared parameter relations stay bounded in shader source and live under edits."""

from __future__ import annotations

import math
import os
import re

import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    sdf_primitive,
)
from software_defined_matter.dsl.expr import expr_binop, expr_num, expr_param
from software_defined_matter.glsl import emit_glsl
from tests._glsl_runtime import ShaderRuntime


def _part(n_depth: int = 60) -> Part:
    params = {"width": Param("width", 3.0, free=True, bounds=(1, 10), unit="mm")}
    previous = "width"
    for i in range(n_depth):
        name = f"level_{i}"
        params[name] = Param(
            name,
            unit="mm",
            expr=expr_binop(
                "/", expr_binop("+", expr_param(previous), expr_param(previous)), expr_num(2)
            ),
        )
        previous = name
    # An unreachable relation must not add a live input or local assignment.
    params["unused"] = Param("unused", 9.0, free=True, bounds=(1, 20), unit="mm")
    params["unused_relation"] = Param("unused_relation", unit="mm", expr=expr_param("unused"))
    return Part(
        name="shared_relations",
        params=params,
        materials=[
            MaterialRegion(
                material_id=1, name="ball", sdf_tree=sdf_primitive("sphere", r={"$ref": previous})
            )
        ],
        metadata={"bbox": [[-12, -12, -12], [12, 12, 12]]},
    )


def _function_body(source: str, needle: str) -> str:
    return next(
        body
        for body in re.findall(r"(?:float|vec3) \w+\([^{}]*\) \{([^{}]*)\}", source)
        if needle in body
    )


def _run_assignments(body: str, uniforms: dict[str, float]) -> dict[str, float]:
    namespace = {
        **uniforms,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "min": min,
        "max": max,
        "pow": pow,
        "abs": abs,
    }
    for name, expression in re.findall(r"float (sdm_rel_\d+) = ([^;]+);", body):
        # Only arithmetic emitted by the library is evaluated, with no builtins.
        namespace[name] = float(eval(expression, {"__builtins__": {}}, namespace))  # noqa: S307 - generated scalar arithmetic
    return namespace


def test_repeated_relation_diamond_emits_linear_source_and_only_reachable_uniforms():
    part = _part()
    emission = emit_glsl(part)
    assert len(emission.scene_source) < 8000
    assert len(re.findall(r"float sdm_rel_\d+ =", emission.scene_source)) == 60
    assert [uniform.source_param for uniform in emission.uniforms] == ["width"]
    body = _function_body(emission.scene_source, "return sdf_sphere")
    radius = re.search(r"sdf_sphere\(p, (sdm_rel_\d+)\)", body).group(1)
    for width in (1.25, 3.0, 8.5):
        values = _run_assignments(body, {"u_p_width": width})
        assert values[radius] == pytest.approx(width, abs=1e-12)


def test_each_component_function_declares_its_own_relation_closure():
    part = _part(8)
    part.materials.append(
        MaterialRegion(
            material_id=2,
            name="box",
            sdf_tree=sdf_primitive("box", b=[{"$ref": "level_7"}, {"$ref": "level_3"}, 1.0]),
        )
    )
    emission = emit_glsl(part)
    for primitive in ("sdf_sphere", "sdf_box"):
        body = _function_body(emission.scene_source, f"return {primitive}")
        assert len(re.findall(r"float sdm_rel_\d+ =", body)) == 8
        declared = set(re.findall(r"float (sdm_rel_\d+) =", body))
        assert set(re.findall(r"\bsdm_rel_\d+\b", body)) == declared
    assert [uniform.source_param for uniform in emission.uniforms] == ["width"]


def test_motion_rest_function_resolves_derived_design_inputs_with_live_dofs():
    part = _part(3)
    part.materials[0].sdf_tree["name"] = "ball"
    part.kinematics = {
        "dofs": [
            {"name": "travel", "kind": "length", "unit": "mm", "range": [-1, 1], "default": 0}
        ],
        "bodies": [
            {
                "name": "ball",
                "region": {"$node": "ball"},
                "motion": {
                    "ops": [
                        {
                            "kind": "translate",
                            "axis": [1, 0, 0],
                            "distance": expr_binop(
                                "*", {"type": "dof", "name": "travel"}, expr_param("level_2")
                            ),
                        }
                    ]
                },
            }
        ],
    }
    emission = emit_glsl(part)
    body = _function_body(emission.scene_source, "p -=")
    expression = re.search(r"p -= vec3\([^;]+?\) \* \(([^;]+)\);", body).group(1)
    values = _run_assignments(body, {"u_p_width": 7.0, "u_dof_0": 0.5})
    actual = eval(expression, {"__builtins__": {}}, values)  # noqa: S307 - generated scalar arithmetic
    assert actual == pytest.approx(3.5, abs=1e-12)
    assert {u.source_param for u in emission.uniforms} == {"width", "kinematics.travel"}


def test_compiled_shader_relations_follow_uniform_changes():
    try:
        runtime = ShaderRuntime()
    except (OSError, AttributeError, RuntimeError) as exc:
        if os.environ.get("SDM_REQUIRE_GLSL") == "1":
            pytest.fail(f"Required software shader runtime unavailable: {exc}")
        pytest.skip(
            f"Install system EGL/GLES and Mesa software drivers to run shader parity: {exc}"
        )
    try:
        emission = emit_glsl(_part(30))
        points = np.asarray([[0, 0, 0], [4, 0, 0], [8, 0, 0]], dtype=np.float32)
        for width in (2.0, 6.0):
            result = runtime.evaluate(emission, points, uniforms={"u_p_width": width})[:, 3]
            np.testing.assert_allclose(
                result, np.linalg.norm(points, axis=1) - width, atol=1e-5, rtol=1e-5
            )
    finally:
        runtime.close()


def test_scoped_parameter_uniforms_are_legal_and_do_not_collide_after_sanitization():
    names = ["pose__hinge", "pose_hinge", "pose_hinge_2", "_leading", "trailing_", "__"]
    part = Part(
        name="scoped_parameters",
        params={name: Param(name, i + 1.0, free=True, unit="mm") for i, name in enumerate(names)},
        materials=[
            MaterialRegion(
                1,
                "ball",
                sdf_primitive(
                    "sphere",
                    r={
                        "type": "reduce",
                        "op": "sum",
                        "children": [expr_param(name) for name in names],
                    },
                ),
            )
        ],
        metadata={"bbox": [[-30, -30, -30], [30, 30, 30]]},
    )
    emission = emit_glsl(part)
    uniforms = {uniform.source_param: uniform.name for uniform in emission.uniforms}
    assert set(uniforms) == set(names)
    assert len(set(uniforms.values())) == len(names)
    assert all(
        re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", name) and "__" not in name
        for name in uniforms.values()
    )
    assert uniforms["pose__hinge"] != uniforms["pose_hinge"]
    controls = {control["param"]: control["uniform"] for control in emission.controls}
    assert controls == uniforms
