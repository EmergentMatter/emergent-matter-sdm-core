"""Negative powers cannot certify finite bounds across their singularity at zero."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from software_defined_matter import (
    Param,
    Part,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.dsl.expr import expr_binop, expr_num, expr_param
from software_defined_matter.sdf.bbox import UnsupportedExprError, infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.exactness import UNKNOWN, infer_sdf_exactness


def _inverse_radius_sphere(bounds):
    radius = expr_binop("+", expr_num(1), expr_binop("pow", expr_param("t"), expr_num(-2)))
    tree = sdf_primitive("sphere", r=radius)
    part = Part(name="inverse-radius", params={"t": Param("t", 1, free=True, bounds=bounds)})
    return tree, part


def test_singular_radius_cannot_certify_primitive_union_or_shell_distances():
    sphere, part = _inverse_radius_sphere((-1, 1))
    union = sdf_op("union", [sphere, sdf_transform("translate", sphere, t=[5, 0, 0])])
    shell = sdf_modifier("onion", union, thickness=0.1)
    for tree in (sphere, union, shell):
        with pytest.raises(
            UnsupportedExprError, match="Negative power over interval spanning zero"
        ):
            infer_sdf_bbox(tree, part)
        contract = infer_sdf_exactness(tree, part)
        assert contract.outside == UNKNOWN
        assert contract.inside == UNKNOWN
    assert infer_sdf_exactness(union, part, mode="values").is_exact
    # At t=0.5 each radius is 5, exceeding the old endpoint-only bound of 2.
    fn = make_sdf_closure(sphere, part)
    assert float(fn(jnp.array([5.0, 0, 0]), jnp.array([0.5]))) == pytest.approx(0, abs=1e-6)


def test_inverse_radius_away_from_zero_retains_finite_bounds_and_exactness():
    tree, part = _inverse_radius_sphere((0.5, 1))
    lo, hi = infer_sdf_bbox(tree, part)
    assert lo == pytest.approx((-5, -5, -5), abs=1e-12)
    assert hi == pytest.approx((5, 5, 5), abs=1e-12)
    assert infer_sdf_exactness(tree, part).is_exact
