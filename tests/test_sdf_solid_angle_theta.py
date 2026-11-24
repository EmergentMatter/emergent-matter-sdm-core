"""A solid-angle aperture is one live scalar with a unit sine/cosine direction."""

from __future__ import annotations

import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    load_part,
    load_schema,
    make_param_ref,
    save,
    sdf_primitive,
    sdf_solid_angle,
)
from software_defined_matter.dsl.expr import expr_binop, expr_num, expr_param
from software_defined_matter.glsl.emit import emit_glsl
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.sdf_shapes import solid_angle
from software_defined_matter.sdf.validate import (
    SemanticValidationError,
    validate_document_semantics,
)
from tests._glsl_runtime import ShaderRuntime

# Float32 trig and GPU arithmetic differ by a few ulps at unit-scale distances.
FLOAT32_ATOL = 2e-6


def _angle_part(bounds=(0.1, 3.0)):
    tree = sdf_solid_angle(theta=make_param_ref("theta"), ra=2)
    part = Part(
        name="angular-sector",
        params={"theta": Param("theta", math.pi / 4, free=True, bounds=bounds, unit="rad")},
        materials=[MaterialRegion(material_id=1, name="solid", sdf_tree=tree)],
    )
    return tree, part


def test_angle_builder_round_trips_with_theta_schema_and_one_free_parameter(tmp_path):
    tree, part = _angle_part()
    path = tmp_path / "sector.sdm"
    save(part, path)
    loaded = load_part(path)
    assert loaded.to_dict()["schema_version"] == "0.8"
    assert loaded.materials[0].sdf_tree == tree
    assert set(tree["params"]) == {"theta", "ra"}
    assert [name for name, param in loaded.params.items() if param.free] == ["theta"]
    lo, hi = infer_sdf_bbox(tree, loaded)
    assert lo == pytest.approx((-2, -2, -2), abs=1e-12)
    assert hi == pytest.approx((2, 2, 2), abs=1e-12)


def test_float32_jitted_angle_gradient_matches_the_conical_side_reference():
    tree, part = _angle_part()
    fn = make_sdf_closure(tree, part)
    point = jnp.array([0.5, 0, 1], dtype=jnp.float32)
    value_and_grad = jax.jit(jax.value_and_grad(lambda free: fn(point, free)))
    for d_theta in (math.pi / 6, math.pi / 4, math.pi / 3):
        value, gradient = value_and_grad(jnp.array([d_theta], dtype=jnp.float32))
        assert float(value) == pytest.approx(
            0.5 * math.cos(d_theta) - math.sin(d_theta), abs=FLOAT32_ATOL
        )
        assert float(gradient[0]) == pytest.approx(
            -0.5 * math.sin(d_theta) - math.cos(d_theta), abs=FLOAT32_ATOL
        )


def test_literal_angle_and_angle_expression_describe_the_same_sector():
    tree = sdf_solid_angle(math.pi / 4, 2)
    expression = expr_binop("*", expr_param("theta"), expr_num(0.5))
    expr_tree = sdf_solid_angle(expression, 2)
    part = Part(name="expression", params={"theta": Param("theta", math.pi / 2, bounds=(1, 2))})
    point = jnp.array([0.5, 0, 1], dtype=jnp.float32)
    literal_value = make_sdf_closure(tree, Part(name="literal"))(point, jnp.zeros(0))
    expr_value = make_sdf_closure(expr_tree, part)(point, jnp.zeros(0))
    assert float(expr_value) == pytest.approx(float(literal_value), abs=FLOAT32_ATOL)


@pytest.mark.parametrize(
    "theta",
    [0, -0.1, math.pi, math.inf, math.nan],
    ids=["zero", "negative", "pi", "infinite", "nan"],
)
def test_literal_degenerate_or_nonfinite_angles_are_rejected(theta):
    with pytest.raises(ValueError, match=r"theta must be in \(0, pi\) radians"):
        sdf_solid_angle(theta, 2)


@pytest.mark.parametrize("with_theta", [False, True], ids=["c-only", "both"])
def test_direction_vector_inputs_are_rejected_by_compile_shader_and_schema(with_theta):
    kw = {"c": [8e-20, 8e-20], "ra": 2}
    if with_theta:
        kw["theta"] = math.pi / 4
    tree = sdf_primitive("solid_angle", **kw)
    part = Part(name="removed-vector", materials=[MaterialRegion(1, "solid", tree)])
    with pytest.raises(ValueError, match="c is not supported"):
        make_sdf_closure(tree, part)
    with pytest.raises(ValueError, match="c is not supported"):
        emit_glsl(part)
    with pytest.raises(SemanticValidationError, match="solid_angle"):
        validate_document_semantics(part.to_dict())
    assert not Draft202012Validator(load_schema("0.8")).is_valid(part.to_dict())


@pytest.mark.parametrize("version", ["0.6", "0.7"])
def test_frozen_schema_can_describe_old_c_but_loading_it_is_rejected(tmp_path, version):
    tree = sdf_primitive("solid_angle", c=[0.70710678, 0.70710678], ra=2)
    part = Part(name="old-vector", materials=[MaterialRegion(1, "solid", tree)])
    doc = part.to_dict()
    doc["schema_version"] = version
    assert Draft202012Validator(load_schema(version)).is_valid(doc)
    path = tmp_path / "old.sdm"
    path.write_text(json.dumps(doc))
    with pytest.raises(SemanticValidationError, match="solid_angle"):
        load_part(path)


@pytest.mark.parametrize("version", ["0.6", "0.7"])
def test_theta_document_requires_the_new_schema(version):
    _, part = _angle_part()
    doc = part.to_dict()
    assert Draft202012Validator(load_schema("0.8")).is_valid(doc)
    doc["schema_version"] = version
    assert not Draft202012Validator(load_schema(version)).is_valid(doc)


def test_direct_evaluators_and_builder_refuse_c_and_vector_theta():
    point = jnp.array([0, 0, 1], dtype=jnp.float32)
    with pytest.raises(TypeError, match="unexpected keyword argument 'c'"):
        solid_angle(point, c=[1, 1], ra=2)
    with pytest.raises(TypeError, match="unexpected keyword argument 'c'"):
        sdf_solid_angle(c=[1, 1], ra=2)
    with pytest.raises(ValueError, match="theta must be a scalar"):
        jax.jit(solid_angle)(point, jnp.array([8e-20, 8e-20]), 2)
    tree = sdf_primitive("solid_angle", theta=[8e-20, 8e-20], ra=2)
    part = Part(name="vector-theta", materials=[MaterialRegion(1, "solid", tree)])
    with pytest.raises(SemanticValidationError, match="theta: expected a number"):
        validate_document_semantics(part.to_dict())
    assert not Draft202012Validator(load_schema("0.8")).is_valid(part.to_dict())


def test_angle_builder_avoids_the_tiny_vector_normalization_failure():
    tree = sdf_solid_angle(math.pi / 4, 2)
    fn = jax.jit(make_sdf_closure(tree, Part(name="unit-direction")))
    point = jnp.array([0, 0, 1], dtype=jnp.float32)
    assert float(fn(point, jnp.zeros(0))) == pytest.approx(-math.sqrt(0.5), abs=FLOAT32_ATOL)


def test_live_angle_shader_matches_jitted_jax_across_the_aperture_range():
    tree, part = _angle_part()
    try:
        runtime = ShaderRuntime()
    except (OSError, AttributeError, RuntimeError) as exc:
        if os.environ.get("SDM_REQUIRE_GLSL") == "1":
            pytest.fail(f"Required software shader runtime unavailable: {exc}")
        pytest.skip(f"Install EGL/GLES and Mesa software drivers for shader parity: {exc}")
    try:
        emission = emit_glsl(part)
        fn = jax.jit(make_sdf_closure(tree, part))
        points = np.random.default_rng(42).uniform(-2.5, 2.5, size=(1000, 3)).astype(np.float32)
        for d_theta in (0.1, math.pi / 4, math.pi / 2, 2.5, 3.0):
            gpu = runtime.evaluate(emission, points, uniforms={"u_p_theta": d_theta})[:, 3]
            cpu = np.asarray(fn(jnp.asarray(points), jnp.array([d_theta], dtype=jnp.float32)))
            reference = solid_angle(jnp.asarray(points), d_theta, 2)
            np.testing.assert_allclose(cpu, reference, atol=FLOAT32_ATOL, rtol=1e-6)
            np.testing.assert_allclose(gpu, cpu, atol=FLOAT32_ATOL, rtol=1e-6)
    finally:
        runtime.close()
