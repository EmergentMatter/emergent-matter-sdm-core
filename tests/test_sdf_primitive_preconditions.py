"""Primitive distance claims require a domain proof, not just a registered name."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import Param, Part, sdf_primitive
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.exactness import EXACT, LOWER_BOUND, UNKNOWN, infer_sdf_exactness


def _contract(kind, params, part=None, **options):
    return infer_sdf_exactness(
        sdf_primitive(kind, **params), part or Part(name="domains"), **options
    )


@pytest.mark.parametrize("dim", [2, 3], ids=["rounded-box-2d", "rounded-box-3d"])
def test_oversized_rounding_is_not_certified_as_an_exact_distance(dim):
    kind = "rounded_box_2d" if dim == 2 else "round_box"
    tree = sdf_primitive(kind, b=[1] * dim, r=2)
    part = Part(name="oversized")
    fn = make_sdf_closure(tree, part)
    # In the positive orthant the boundary is |p + 1|=2. Its largest
    # first coordinate lies on the first axis, so this is the closest
    # boundary point to an exterior query farther along that axis.
    boundary = np.zeros(dim)
    boundary[0] = math.sqrt(5 - dim) - 1
    query = np.zeros(dim)
    query[0] = 3
    assert float(fn(jnp.asarray(boundary))) == pytest.approx(0, abs=2e-6)
    assert float(fn(jnp.asarray(query))) == pytest.approx(math.sqrt(15 + dim) - 2, abs=2e-6)
    assert float(fn(jnp.asarray(query))) < np.linalg.norm(query - boundary) - 0.01
    assert infer_sdf_exactness(tree, part).outside == UNKNOWN


@pytest.mark.parametrize("kind,dim", [("round_box", 3), ("rounded_box_2d", 2)])
def test_rounding_must_fit_the_smallest_extent_over_the_whole_range(kind, dim):
    params = {"b": [1] * dim, "r": {"$ref": "r"}}
    part = Part(name="rounded-range", params={"r": Param("r", 0.5, free=True, bounds=(0, 2))})
    assert _contract(kind, params, part).outside == UNKNOWN
    assert _contract(kind, params, part, mode="values").is_exact
    part.params["r"].bounds = (0, 1)
    assert _contract(kind, params, part).is_exact
    params["b"] = [{"$ref": "b"}] + [1] * (dim - 1)
    part.params["b"] = Param("b", 1, free=True, bounds=(0.25, 1))
    assert _contract(kind, params, part).inside == UNKNOWN


@pytest.mark.parametrize(
    "kind,valid,invalid",
    [
        ("sphere", {"r": 1}, {"r": -1}),
        ("circle_2d", {"r": 1}, {"r": 0}),
        ("box", {"b": [1, 2, 3]}, {"b": [1, -2, 3]}),
        ("capped_cylinder", {"r": 1, "h": 2}, {"r": 1, "h": 0}),
        ("rounded_cylinder", {"ra": 1, "rb": 0.5, "h": 1}, {"ra": 1, "rb": 3, "h": 1}),
        ("cut_sphere", {"r": 2, "h": 1}, {"r": 1, "h": 2}),
        ("uneven_capsule_2d", {"r1": 2, "r2": 1, "h": 3}, {"r1": 2, "r2": 1, "h": 0.5}),
        ("cone", {"c": [1, 1], "h": 1}, {"c": [1, 0], "h": 1}),
        ("capped_cone", {"h": 1, "r1": 2, "r2": 1}, {"h": 0, "r1": 2, "r2": 1}),
        ("trapezoid_2d", {"he": 1, "r1": 2, "r2": 1}, {"he": 0, "r1": 2, "r2": 1}),
        ("torus", {"t": [3, 1]}, {"t": [3, -1]}),
        ("link", {"le": 1, "r1": 3, "r2": 1}, {"le": -1, "r1": 3, "r2": 1}),
        ("pyramid", {"h": 1}, {"h": -1}),
        ("octahedron", {"s": 1}, {"s": -1}),
        ("solid_angle", {"theta": 0.7, "ra": 2}, {"theta": 0, "ra": 2}),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_analytic_primitives_require_their_geometric_preconditions(kind, valid, invalid):
    assert _contract(kind, valid).is_exact
    result = _contract(kind, invalid)
    assert result.outside == result.inside == UNKNOWN


def test_rounded_cylinder_with_negative_core_radius_is_not_exact_outside():
    tree = sdf_primitive("rounded_cylinder", ra=1, rb=3, h=1)
    part = Part(name="negative-core")
    fn = make_sdf_closure(tree, part)
    p = jnp.array([0.0, 0.0, 5.0])
    boundary_z = 1 + math.sqrt(8)
    assert float(fn(jnp.array([0.0, 0.0, boundary_z]))) == pytest.approx(0, abs=2e-6)
    assert float(fn(p)) == pytest.approx(math.sqrt(17) - 3, abs=2e-6)
    assert float(fn(p)) < 5 - boundary_z - 0.01
    assert infer_sdf_exactness(tree, part).outside == UNKNOWN


@pytest.mark.parametrize(
    "kind,params,name,value,bounds",
    [
        ("cut_sphere", {"r": 2, "h": {"$ref": "h"}}, "h", 0, (-3, 3)),
        ("uneven_capsule_2d", {"r1": 2, "r2": 1, "h": {"$ref": "h"}}, "h", 3, (0.5, 4)),
        ("rounded_cylinder", {"ra": 1, "rb": {"$ref": "rb"}, "h": 1}, "rb", 0.5, (0, 3)),
        ("cone", {"c": [1, {"$ref": "cos"}], "h": 1}, "cos", 1, (-1, 1)),
        ("sphere", {"r": {"$ref": "r"}}, "r", 1, (-1, 2)),
    ],
    ids=["cut-height", "uneven-separation", "cylinder-rounding", "cone-angle", "sphere-radius"],
)
def test_valid_current_parameters_do_not_certify_an_invalid_range(
    kind, params, name, value, bounds
):
    part = Part(name="range", params={name: Param(name, value, free=True, bounds=bounds)})
    assert _contract(kind, params, part).outside == UNKNOWN
    assert _contract(kind, params, part, mode="values").is_exact


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_parameters_never_get_a_distance_contract(radius):
    result = _contract("sphere", {"r": radius})
    assert result.outside == result.inside == UNKNOWN


@pytest.mark.parametrize(
    "kind,params",
    [
        ("sphere", {"r": {"$ref": "r"}}),
        ("torus", {"t": [3, {"$ref": "r"}]}),
        ("ellipsoid", {"r": [1, 2, {"$ref": "r"}]}),
    ],
)
def test_unbounded_radius_cannot_get_an_exact_or_lower_bound_fallback(kind, params):
    part = Part(name="unbounded", params={"r": Param("r", 1, free=True)})
    result = _contract(kind, params, part)
    assert result.outside == result.inside == UNKNOWN


def test_derived_radius_does_not_use_its_cache_as_a_domain_proof():
    part = Part(name="derived", params={"r": Param("r", 1, expr={"type": "num", "value": -1})})
    for mode in ("bounds", "values"):
        assert _contract("sphere", {"r": {"$ref": "r"}}, part, mode=mode).outside == UNKNOWN


def test_reflex_sector_retains_only_a_bound_inside():
    params = {"inner_r": 0.1, "outer_r": 10, "half_angle": 2.5, "height": 20}
    tree = sdf_primitive("annular_sector", **params)
    part = Part(name="reflex")
    fn = make_sdf_closure(tree, part)
    # The nearest boundary at (1,0,0) is the inner circle, 0.9 away.
    assert float(fn(jnp.array([1.0, 0.0, 0.0]))) == pytest.approx(-math.sin(2.5), abs=2e-6)
    assert float(fn(jnp.array([0.1, 0.0, 0.0]))) == pytest.approx(0, abs=2e-6)
    assert infer_sdf_exactness(tree, part).inside == LOWER_BOUND
    params["half_angle"] = 1
    assert _contract("annular_sector", params).inside == EXACT


@pytest.mark.parametrize(
    "vertices,exact",
    [
        ([[0, 0], [2, 0], [2, 2], [0, 2]], True),
        ([[0, 0], [2, 0], [1, 1], [2, 2], [0, 2]], True),
        ([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]], True),
        ([[0, 0], [2, 2], [0, 2], [2, 0]], False),
        ([[0, 0], [1, 0], [2, 0]], False),
        ([[0, 0], [2, 0], [1, 0], [1, 2], [0, 2]], False),
    ],
    ids=["convex", "concave", "closing-vertex", "crossed", "zero-area", "retraced"],
)
def test_polygon_contract_requires_a_proven_simple_outline(vertices, exact):
    assert _contract("polygon_2d", {"vertices": vertices}).is_exact is exact


def test_moving_polygon_cannot_use_simplicity_of_just_the_current_shape():
    part = Part(name="polygon", params={"x": Param("x", 2, free=True, bounds=(-2, 2))})
    params = {"vertices": [[0, 0], [{"$ref": "x"}, 0], [2, 2], [0, 2]]}
    assert _contract("polygon_2d", params, part).outside == UNKNOWN
    assert _contract("polygon_2d", params, part, mode="values").is_exact


def test_polygon_edges_affected_by_the_denominator_floor_are_not_exact():
    params = {"vertices": [[0, 0], [1e-7, 0], [1e-7, 1e-7], [0, 1e-7]]}
    assert _contract("polygon_2d", params).outside == UNKNOWN


@pytest.mark.parametrize("kind", ["bezier_2d", "bspline_2d"])
def test_curve_control_points_do_not_certify_a_simple_tessellated_outline(kind):
    points = [[0, 0], [1, 1], [1, -1], [2, 0], [1, 1], [1, -1]]
    assert _contract(kind, {"control_points": points}).outside == UNKNOWN


@pytest.mark.parametrize(
    "kind,params",
    [
        ("ellipsoid", {"r": [1, 0, 2]}),
        ("bellows", {"outer_r": 2, "inner_r": 1, "period": 0, "n_periods": 2}),
        (
            "serpentine",
            {"amplitude": 1, "wavelength": 0, "beam_width": 1, "beam_height": 1, "n_periods": 2},
        ),
        ("gyroid", {"period": 0, "min_thickness": 1, "n_periods": [2, 2, 2]}),
        (
            "screw_thread",
            {"r_root": 2, "depth": 1, "pitch": 3, "width": 1, "n_turns": 2, "handedness": 2},
        ),
    ],
)
def test_invalid_bound_primitives_are_unknown_rather_than_certified_safe(kind, params):
    result = _contract(kind, params)
    assert result.outside == result.inside == UNKNOWN


def test_sector_interior_contract_accounts_for_reflex_angles_anywhere_in_bounds():
    part = Part(
        name="sector-range", params={"angle": Param("angle", 1, free=True, bounds=(0.5, 2.5))}
    )
    params = {"inner_r": 1, "outer_r": 3, "half_angle": {"$ref": "angle"}, "height": 2}
    assert _contract("annular_sector", params, part).inside == LOWER_BOUND
    assert _contract("annular_sector", params, part, mode="values").inside == EXACT
    part.params["angle"].bounds = (0.5, 1.5)
    assert _contract("annular_sector", params, part).inside == EXACT
