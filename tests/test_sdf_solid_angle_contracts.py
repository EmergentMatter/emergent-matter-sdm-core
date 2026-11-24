"""Angular distance contracts require safe domains and scalar solid-angle inputs."""

from __future__ import annotations

import math

import pytest

from software_defined_matter import Param, Part, make_param_ref, sdf_primitive, sdf_solid_angle
from software_defined_matter.dsl.expr import expr_binop, expr_num, expr_param
from software_defined_matter.sdf.exactness import UNKNOWN, infer_sdf_exactness


def _angle_part(bounds):
    tree = sdf_solid_angle(theta=make_param_ref("theta"), ra=2)
    return tree, Part(
        name="angle", params={"theta": Param("theta", math.pi / 4, free=True, bounds=bounds)}
    )


@pytest.mark.parametrize(
    "bounds",
    [(0, 1), (-1, 1), (1, math.pi), (1, math.pi - 1e-9)],
    ids=["zero", "negative", "pi", "rounds-to-pi"],
)
def test_live_angle_contracts_reject_degenerate_or_float32_unsafe_ranges(bounds):
    tree, part = _angle_part(bounds)
    contract = infer_sdf_exactness(tree, part)
    assert contract.outside == UNKNOWN
    assert contract.inside == UNKNOWN


def test_capped_torus_subnormal_component_squares_are_not_certified_exact():
    tree = sdf_primitive("capped_torus", sc=[8e-20, 8e-20], ra=2, rb=0.5)
    assert not infer_sdf_exactness(tree, Part(name="tiny-vector")).is_exact


def test_scalar_angle_literals_references_and_expressions_receive_exact_contracts():
    literal = sdf_solid_angle(math.pi / 4, 2)
    expression = sdf_solid_angle(expr_binop("*", expr_param("theta"), expr_num(0.5)), 2)
    part = Part(name="expression", params={"theta": Param("theta", math.pi / 2, bounds=(1, 2))})
    tree, live_part = _angle_part((0.1, 3.0))
    for candidate, source in ((literal, part), (expression, part), (tree, live_part)):
        assert infer_sdf_exactness(candidate, source).is_exact


@pytest.mark.parametrize("with_theta", [False, True], ids=["c-only", "both"])
def test_direction_vector_inputs_cannot_receive_exact_contracts(with_theta):
    kw = {"c": [8e-20, 8e-20], "ra": 2}
    if with_theta:
        kw["theta"] = math.pi / 4
    tree = sdf_primitive("solid_angle", **kw)
    assert not infer_sdf_exactness(tree, Part(name="removed-vector")).is_exact
