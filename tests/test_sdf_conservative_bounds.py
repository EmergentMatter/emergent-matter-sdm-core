"""Geometry bounds enclose parameter envelopes and reject singular intervals."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from software_defined_matter import Param, Part, sdf_modifier, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.dsl.expr import expr_binop, expr_num, expr_param
from software_defined_matter.sdf.bbox import (
    UnsupportedExprError,
    UnsupportedSDFNodeError,
    _interval_binop,
    infer_sdf_bbox,
)


@pytest.mark.parametrize("apothem", [0.25, 1, 3], ids=["small", "unit", "large"])
def test_hex_prism_bbox_encloses_all_vertices_over_parameter_bounds(apothem):
    part = Part(
        name="hex-range",
        params={"a": Param("a", apothem / 2, free=True, bounds=(apothem / 2, apothem))},
    )
    tree = sdf_primitive("hex_prism", h=[{"$ref": "a"}, 2])
    lo, hi = map(np.asarray, infer_sdf_bbox(tree, part))
    # The horizontal faces are y=+/-a. Their intersections with faces at
    # +/-60 degrees give x=+/-a/sqrt(3) and the axial tips x=+/-2a/sqrt(3).
    x = apothem / math.sqrt(3)
    outline = [(2 * x, 0), (x, apothem), (-x, apothem), (-2 * x, 0), (-x, -apothem), (x, -apothem)]
    vertices = np.array([(px, py, z) for px, py in outline for z in (-2, 2)])
    assert np.all(vertices >= lo)
    assert np.all(vertices <= hi)
    values_lo, values_hi = map(np.asarray, infer_sdf_bbox(tree, part, mode="values"))
    np.testing.assert_allclose(values_hi, [x, apothem / 2, 2], rtol=1e-14, atol=0)
    np.testing.assert_allclose(values_lo, -values_hi, rtol=1e-14, atol=0)


def test_cone_bbox_encloses_base_circles_across_all_parameter_ranges():
    tree = sdf_primitive("cone", c=[{"$ref": "sin"}, {"$ref": "cos"}], h={"$ref": "height"})
    part = Part(
        name="cone-range",
        params={
            "sin": Param("sin", 0.5, free=True, bounds=(0.5, 1)),
            "cos": Param("cos", 1.0, free=True, bounds=(0.25, 1)),
            "height": Param("height", 1.0, free=True, bounds=(0.5, 2)),
        },
    )
    lo, hi = map(np.asarray, infer_sdf_bbox(tree, part))
    # A cone's base is z=-h and its radius is h*sin/cos. Sample endpoints
    # and interior parameter values, including the widest reachable cone.
    for sine, cosine, height in itertools.product((0.5, 0.75, 1), (0.25, 0.5, 1), (0.5, 1, 2)):
        radius = height * sine / cosine
        vertices = np.array(
            [
                (radius, 0, -height),
                (-radius, 0, -height),
                (0, radius, -height),
                (0, -radius, -height),
                (0, 0, 0),
            ]
        )
        assert np.all(vertices >= lo)
        assert np.all(vertices <= hi)
    assert hi[0] == pytest.approx(8, abs=1e-12)
    _, values_hi = infer_sdf_bbox(tree, part, mode="values")
    assert values_hi[0] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    "bounds",
    [(-1, 1), (0, 1), (-1, 0), (0, 0)],
    ids=["crossing", "lower-zero", "upper-zero", "zero"],
)
def test_cone_bbox_rejects_angle_ranges_with_a_zero_denominator(bounds):
    tree = sdf_primitive("cone", c=[1, {"$ref": "cos"}], h=1)
    part = Part(name="singular", params={"cos": Param("cos", 0.5, free=True, bounds=bounds)})
    with pytest.raises(UnsupportedExprError, match="interval spanning zero"):
        infer_sdf_bbox(tree, part)


def test_small_positive_angle_denominator_is_not_replaced_by_a_floor():
    tree = sdf_primitive("cone", c=[1, 1e-10], h=1)
    lo, hi = infer_sdf_bbox(tree, Part(name="wide-cone"))
    assert hi[0] == pytest.approx(1e10, rel=1e-12)
    assert lo[0] == pytest.approx(-1e10, rel=1e-12)


def test_cone_bbox_rejects_an_unrepresentable_radius_instead_of_returning_infinity():
    tree = sdf_primitive("cone", c=[1, 1e-320], h=1)
    with pytest.raises(UnsupportedSDFNodeError, match="finite bounding box"):
        infer_sdf_bbox(tree, Part(name="unrepresentable-cone"))


def _scaled_sphere(center):
    child = sdf_transform("translate", sdf_primitive("sphere", r=1), t=center)
    tree = sdf_transform("scale", child, s={"$ref": "scale"})
    part = Part(name="scale-range", params={"scale": Param("scale", 2, free=True, bounds=(1, 2))})
    return tree, part


@pytest.mark.parametrize(
    "center",
    [(4, 3, 2), (-4, -3, -2), (4, -3, 0), (0, 0, 0)],
    ids=["positive", "negative", "mixed", "centered"],
)
def test_scale_bounds_enclose_sphere_extrema_at_endpoints_and_interior_scales(center):
    tree, part = _scaled_sphere(center)
    lo, hi = map(np.asarray, infer_sdf_bbox(tree, part))
    for d_scale in (1, 1.25, 1.5, 2):
        extrema = d_scale * (np.asarray(center) + np.concatenate((np.eye(3), -np.eye(3))))
        assert np.all(extrema >= lo)
        assert np.all(extrema <= hi)
    expected_lo = np.minimum(np.asarray(center) - 1, 2 * (np.asarray(center) - 1))
    expected_hi = np.maximum(np.asarray(center) + 1, 2 * (np.asarray(center) + 1))
    assert lo == pytest.approx(expected_lo, abs=1e-12)
    assert hi == pytest.approx(expected_hi, abs=1e-12)
    values_lo, values_hi = infer_sdf_bbox(tree, part, mode="values")
    assert values_lo == pytest.approx(2 * (np.asarray(center) - 1), abs=1e-12)
    assert values_hi == pytest.approx(2 * (np.asarray(center) + 1), abs=1e-12)


@pytest.mark.parametrize(
    "bounds",
    [(-2, -1), (-1, 1), (0, 1), (0, 0)],
    ids=["negative", "crossing-zero", "lower-zero", "zero"],
)
def test_nonpositive_scales_cannot_establish_a_finite_solid_bbox(bounds):
    tree, _ = _scaled_sphere((4, 0, 0))
    part = Part(name="invalid-scale", params={"scale": Param("scale", bounds[0], bounds=bounds)})
    with pytest.raises(
        UnsupportedSDFNodeError, match="Scale factors must stay positive and finite"
    ):
        infer_sdf_bbox(tree, part)


@pytest.mark.parametrize("exponent", [-1, -2, -3, -4], ids=["inverse", "even", "odd", "fourth"])
@pytest.mark.parametrize(
    "bounds",
    [(-1, 1), (0, 1), (-1, 0), (0, 0)],
    ids=["crossing", "lower-zero", "upper-zero", "zero"],
)
def test_negative_powers_reject_intervals_containing_zero(bounds, exponent):
    with pytest.raises(UnsupportedExprError, match="Negative power over interval spanning zero"):
        _interval_binop("pow", bounds, (exponent, exponent))


@pytest.mark.parametrize(
    "bounds, exponent, expected",
    [
        ((1, 2), -1, (0.5, 1)),
        ((-2, -1), -1, (-1, -0.5)),
        ((1, 2), -2, (0.25, 1)),
        ((-2, -1), -2, (0.25, 1)),
        ((-2, 3), 2, (0, 9)),
        ((-2, 3), 3, (-8, 27)),
        ((-2, 3), 0, (1, 1)),
    ],
    ids=[
        "positive-odd",
        "negative-odd",
        "positive-even",
        "negative-even",
        "square",
        "cube",
        "zero-power",
    ],
)
def test_integer_power_intervals_enclose_regular_domains(bounds, exponent, expected):
    assert _interval_binop("pow", bounds, (exponent, exponent)) == pytest.approx(
        expected, abs=1e-12
    )


def _inverse_radius_sphere(bounds):
    radius = expr_binop("+", expr_num(1), expr_binop("pow", expr_param("t"), expr_num(-2)))
    tree = sdf_primitive("sphere", r=radius)
    part = Part(name="inverse-radius", params={"t": Param("t", 1, free=True, bounds=bounds)})
    return tree, part


def test_singular_radius_cannot_produce_finite_primitive_union_or_shell_bounds():
    sphere, part = _inverse_radius_sphere((-1, 1))
    union = sdf_op("union", [sphere, sdf_transform("translate", sphere, t=[5, 0, 0])])
    shell = sdf_modifier("onion", union, thickness=0.1)
    for tree in (sphere, union, shell):
        with pytest.raises(
            UnsupportedExprError, match="Negative power over interval spanning zero"
        ):
            infer_sdf_bbox(tree, part)


def test_inverse_radius_away_from_zero_retains_finite_bounds():
    tree, part = _inverse_radius_sphere((0.5, 1))
    lo, hi = infer_sdf_bbox(tree, part)
    assert lo == pytest.approx((-5, -5, -5), abs=1e-12)
    assert hi == pytest.approx((5, 5, 5), abs=1e-12)
