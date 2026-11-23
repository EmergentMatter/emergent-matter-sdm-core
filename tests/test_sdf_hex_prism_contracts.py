"""Hexagonal-prism boxes must enclose the vertices used by CSG distance proofs."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from software_defined_matter import Param, Part, sdf_modifier, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.exactness import (
    EXACT,
    LOWER_BOUND,
    infer_sdf_exactness,
)


def _hex_union(separation=2.1, apothem=1):
    prism = sdf_primitive("hex_prism", h=[apothem, 1])
    return sdf_op("union", [prism, sdf_transform("translate", prism, t=[separation, 0, 0])])


def test_overlapping_hex_prisms_do_not_receive_an_exact_interior_contract():
    tree = _hex_union()
    part = Part(name="hex-union")
    fn = make_sdf_closure(tree, part)
    # At x=1.05, both nearest individual faces lie inside the other prism.
    # The nearest exposed boundary is their intersection on the Y axis
    # through this midpoint: y=2-sqrt(3)*1.05. The caps are farther away.
    depth = 2 - math.sqrt(3) * 1.05
    assert float(fn(jnp.array([1.05, 0, 0]))) == pytest.approx(-depth / 2, abs=2e-6)
    assert float(fn(jnp.array([1.05, depth, 0]))) == pytest.approx(0, abs=2e-6)
    assert infer_sdf_exactness(tree, part).outside == EXACT
    assert infer_sdf_exactness(tree, part).inside == LOWER_BOUND
    assert infer_sdf_exactness(tree, part).inside == LOWER_BOUND


def test_hex_union_shell_does_not_claim_exact_exterior_distance():
    tree = sdf_modifier("onion", _hex_union(), thickness=0.05)
    part = Part(name="hex-shell")
    fn = make_sdf_closure(tree, part)
    # The shell's inner edge is reached when the union field is -0.05.
    # Its slope along this seam is 1/2, so a 0.05 field offset moves it 0.1.
    distance = 2 - math.sqrt(3) * 1.05 - 0.1
    assert float(fn(jnp.array([1.05, 0, 0]))) == pytest.approx(distance / 2, abs=2e-6)
    assert float(fn(jnp.array([1.05, distance, 0]))) == pytest.approx(0, abs=2e-6)
    assert (
        infer_sdf_exactness(tree, part).outside
        == infer_sdf_exactness(tree, part).inside
        == LOWER_BOUND
    )


def test_hex_union_disjointness_must_hold_over_the_whole_parameter_range():
    tree = _hex_union(apothem={"$ref": "a"})
    part = Part(name="hex-range", params={"a": Param("a", 0.8, free=True, bounds=(0.8, 1))})
    assert infer_sdf_exactness(tree, part, mode="values").is_exact
    assert infer_sdf_exactness(tree, part).inside == LOWER_BOUND


def test_genuinely_disjoint_hex_prisms_keep_the_exact_union_contract():
    tree = _hex_union(separation=3)
    assert infer_sdf_exactness(tree, Part(name="disjoint-hexes")).is_exact


def test_hex_prism_tips_prevent_a_false_cutter_containment_proof():
    cutter = sdf_primitive("hex_prism", h=[1, 0.5])
    host = sdf_primitive("box", b=[1.1, 1.1, 1.1])
    part = Part(name="protruding-cutter")
    point = jnp.array([1.12, 0, 0])
    assert float(make_sdf_closure(cutter, part)(point)) < 0
    assert float(make_sdf_closure(host, part)(point)) > 0
    contract = infer_sdf_exactness(sdf_op("subtract", [host, cutter]), part)
    assert contract.outside == LOWER_BOUND
    assert contract.inside == EXACT
