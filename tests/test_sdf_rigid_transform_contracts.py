"""Rigid maps preserve distance contracts only on proven finite parameter domains."""

from __future__ import annotations

import json
import math
import warnings

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    load_part,
    sdf_modifier,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.dsl.expr import compile_expr, expr_binop, expr_num, expr_param
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.exactness import (
    UNKNOWN,
    DistanceAccuracyWarning,
    infer_sdf_exactness,
)

TRANSFORMS = ["translate", "rotate_x", "rotate_y", "rotate_z"]


def _tree(transform, value, child=None):
    return sdf_transform(
        transform,
        child or sdf_primitive("sphere", r=1),
        **({"t": [value, 0, 0]} if transform == "translate" else {"angle": value}),
    )


def _part(tree, **kwargs):
    return Part(name="rigid", materials=[MaterialRegion(1, "body", tree)], **kwargs)


def _inverse_square():
    return expr_binop("pow", expr_param("x"), expr_num(-2))


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize(
    "value", [math.inf, -math.inf, math.nan], ids=["inf", "negative-inf", "nan"]
)
def test_nonfinite_rigid_literals_cannot_inherit_exactness(transform, value):
    tree = _tree(transform, value)
    part = _part(tree)
    contract = infer_sdf_exactness(tree, part)
    assert contract.outside == contract.inside == UNKNOWN
    assert not make_sdf_closure(tree, part).distance_exactness.is_exact


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_singular_transform_ranges_are_unknown_even_when_current_values_are_valid(transform):
    tree = _tree(transform, _inverse_square())
    part = _part(tree, params={"x": Param("x", 1, free=True, bounds=(-1, 1), unit="ratio")})
    assert not infer_sdf_exactness(tree, part).is_exact
    assert infer_sdf_exactness(tree, part, mode="values").is_exact


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_fixed_singular_transforms_warn_before_jit_and_do_not_claim_exactness(transform):
    tree = _tree(transform, _inverse_square())
    part = _part(tree, params={"x": Param("x", 0, free=False, unit="ratio")})
    fn = make_sdf_closure(tree, part)
    assert fn.distance_exactness.outside == fn.distance_exactness.inside == UNKNOWN
    with pytest.warns(DistanceAccuracyWarning, match="onion reads a side"):
        shell = make_sdf_closure(sdf_modifier("onion", tree, thickness=0.1), part)
    # Diagnostics occur at compilation, not repeatedly inside a JAX trace.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        value = jax.jit(shell)(jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32))
    assert not bool(jnp.isfinite(value))
    with pytest.warns(DistanceAccuracyWarning, match="not geometric lengths"):
        compile_expr({"type": "metric", "name": "min_value"}, part, sdf_closure=fn)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_loading_warns_when_a_rigid_parameter_range_crosses_a_singularity(tmp_path, transform):
    tree = sdf_modifier("onion", _tree(transform, _inverse_square()), thickness=0.1)
    part = _part(tree, params={"x": Param("x", 1, free=True, bounds=(-1, 1), unit="ratio")})
    path = tmp_path / "singular.sdm"
    path.write_text(json.dumps(part.to_dict()))
    with pytest.warns(DistanceAccuracyWarning, match="onion reads a side"):
        load_part(path)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_finite_rigid_ranges_and_fixed_values_preserve_exactness(transform):
    tree = _tree(transform, _inverse_square())
    part = _part(tree, params={"x": Param("x", 1, free=True, bounds=(0.5, 2), unit="ratio")})
    assert infer_sdf_exactness(tree, part).is_exact
    fixed = _part(tree, params={"x": Param("x", 1, free=False, unit="ratio")})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        fn = make_sdf_closure(tree, fixed)
        make_sdf_closure(sdf_modifier("onion", tree, thickness=0.1), fixed)
    assert fn.distance_exactness.is_exact
    point = jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32)
    expected = 0 if transform == "translate" else 1
    assert float(jax.jit(fn)(point)) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_missing_or_nonfinite_parameter_bounds_cannot_prove_rigid_contracts(transform):
    tree = _tree(transform, {"$ref": "x"})
    for bounds in (None, (-math.inf, math.inf)):
        part = _part(tree, params={"x": Param("x", 1, free=True, bounds=bounds)})
        assert not infer_sdf_exactness(tree, part).is_exact


def test_finite_2d_translation_keeps_its_distance_contract():
    tree = sdf_transform("translate", sdf_primitive("circle_2d", r=1), t=[2, -3])
    assert infer_sdf_exactness(tree, _part(tree)).is_exact


def test_finite_rigid_maps_do_not_upgrade_an_approximate_child():
    child = sdf_primitive("ellipsoid", r=[1, 2, 3])
    part = _part(child)
    before = infer_sdf_exactness(child, part)
    for transform in TRANSFORMS:
        assert infer_sdf_exactness(_tree(transform, 0.5, child), part) == before


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_composed_rigid_expressions_cannot_hide_a_nonfinite_operand(transform):
    angle = expr_binop("min", expr_num(1), expr_num(math.nan))
    tree = _tree(transform, angle)
    part = _part(tree)
    assert not infer_sdf_exactness(tree, part).is_exact
    with pytest.warns(DistanceAccuracyWarning, match="onion reads a side"):
        make_sdf_closure(sdf_modifier("onion", tree, thickness=0.1), part)
