"""Distance contracts must never upgrade a geometric counterexample to exact."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_transform,
    wire,
)
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.exactness import (
    EXACT,
    LOWER_BOUND,
    PRIMITIVE_EXACTNESS,
    UNKNOWN,
    infer_sdf_exactness,
)


def _sphere(x=0.0, r=1.0):
    return sdf_transform("translate", sdf_primitive("sphere", r=r), t=[x, 0, 0])


def _part(tree=None, **kwargs):
    return Part(
        name="distance-contract",
        materials=[MaterialRegion(name="body", material_id=1, sdf_tree=tree or _sphere())],
        **kwargs,
    )


def _sides(tree, part=None, **kwargs):
    result = infer_sdf_exactness(tree, part or _part(tree), **kwargs)
    return result.outside, result.inside


def test_every_registered_primitive_has_an_explicit_declaration():
    assert set(PRIMITIVE_EXACTNESS) == set(wire.PRIMITIVES)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("union", (EXACT, LOWER_BOUND)),
        ("intersect", (LOWER_BOUND, EXACT)),
        ("subtract", (LOWER_BOUND, EXACT)),
    ],
    ids=["union", "intersection", "difference"],
)
def test_overlapping_booleans_only_preserve_the_proven_side(op, expected):
    assert _sides(sdf_op(op, [_sphere(-0.5), _sphere(0.5)])) == expected


def test_union_midpoint_is_shorter_than_the_true_boundary_distance():
    tree = sdf_op("union", [_sphere(-0.5), _sphere(0.5)])
    value = float(make_sdf_closure(tree, _part(tree))(jnp.zeros(3)))
    # The closest exposed boundary is the circle where the spheres meet.
    true_depth = np.sqrt(1.0 - 0.5**2)
    assert value == pytest.approx(-0.5, abs=1e-6)
    assert abs(value) < true_depth
    assert _sides(tree) == (EXACT, LOWER_BOUND)


def test_subtraction_reads_the_cutters_opposite_side():
    cutter = sdf_primitive("box_frame", b=[1, 1, 1], e=0.1)
    assert _sides(sdf_op("subtract", [_sphere(r=3), cutter])) == (LOWER_BOUND, EXACT)
    cutter = sdf_primitive("tri_prism", h=[1, 1])
    assert _sides(sdf_op("subtract", [_sphere(r=3), cutter])) == (LOWER_BOUND, LOWER_BOUND)


@pytest.mark.parametrize("op", ["union", "subtract"], ids=["union", "disjoint-cutter"])
def test_strictly_disjoint_boxes_prove_exact_booleans(op):
    assert _sides(sdf_op(op, [_sphere(-3), _sphere(3)])) == (EXACT, EXACT)


def test_disjointness_must_hold_over_the_whole_parameter_range():
    tree = sdf_op("union", [_sphere(), _sphere({"$ref": "x"})])
    part = _part(tree, params={"x": Param(name="x", value=4, bounds=(1, 5), free=True)})
    assert _sides(tree, part) == (EXACT, LOWER_BOUND)
    assert _sides(tree, part, mode="values") == (EXACT, EXACT)


def test_missing_parameter_bounds_do_not_prove_disjointness():
    tree = sdf_op("union", [_sphere(), _sphere({"$ref": "x"})])
    part = _part(tree, params={"x": Param(name="x", value=4, free=True)})
    assert _sides(tree, part) == (UNKNOWN, UNKNOWN)


def test_containment_of_boxes_does_not_prove_containment_of_solids():
    # A cutter in a torus's hole is inside its box but outside the solid.
    tree = sdf_op("subtract", [sdf_primitive("torus", t=[3, 1]), _sphere(r=0.5)])
    assert _sides(tree) == (LOWER_BOUND, EXACT)


def test_dilation_distributes_over_union_without_fixing_its_interior_distance():
    tree = sdf_modifier("round", sdf_op("union", [_sphere(-2), _sphere(2)]), r=2)
    assert _sides(tree) == (EXACT, LOWER_BOUND)
    value = float(make_sdf_closure(tree, _part(tree))(jnp.zeros(3)))
    assert value == pytest.approx(-1, abs=1e-6)
    assert abs(value) < np.sqrt(3**2 - 2**2)


def test_a_shell_thicker_than_a_sphere_is_not_exact_inside():
    tree = sdf_modifier("onion", _sphere(), thickness=2)
    assert _sides(tree) == (EXACT, LOWER_BOUND)
    value = float(make_sdf_closure(tree, _part(tree))(jnp.zeros(3)))
    # The inner shell has vanished, leaving a radius-three ball.
    assert value == pytest.approx(-1, abs=1e-6)


@pytest.mark.parametrize("scale", [0, -1, {"$ref": "s"}], ids=["zero", "negative", "unbounded"])
def test_unproven_positive_scale_is_unknown(scale):
    tree = sdf_transform("scale", _sphere(), s=scale)
    part = _part(tree, params={"s": Param(name="s", value=1, free=True)})
    assert _sides(tree, part) == (UNKNOWN, UNKNOWN)


def test_anisotropic_scale_is_a_bound_even_when_the_child_is_exact():
    assert _sides(sdf_transform("scale_axis", _sphere(), s=[1, 2, 3])) == (LOWER_BOUND, LOWER_BOUND)
    assert _sides(sdf_transform("scale", _sphere(), s=2)) == (EXACT, EXACT)


def test_smoothing_metadata_and_unknown_nodes_cannot_claim_exactness():
    tree = sdf_op("union", [_sphere(-3), _sphere(3)])
    part = _part(tree, metadata={"smooth_csg": True})
    assert _sides(tree, part) == (UNKNOWN, UNKNOWN)
    assert _sides(tree, part, b_smooth_csg=False) == (EXACT, EXACT)
    assert _sides({"type": "primitive", "kind": "future_shape"}) == (UNKNOWN, UNKNOWN)


def test_plane_normal_that_can_vanish_is_unknown():
    assert _sides(sdf_primitive("plane", n=[0, 0, 0], h=1)) == (UNKNOWN, UNKNOWN)
    assert _sides(sdf_primitive("plane", n=[0, 0, 3], h=1)) == (EXACT, EXACT)


def test_a_cutter_proven_inside_a_solid_preserves_both_sides():
    tree = sdf_op("subtract", [_sphere(r=4), _sphere(r=1)])
    assert _sides(tree) == (EXACT, EXACT)


def test_overlapping_internal_cutters_do_not_preserve_the_outside_distance():
    tree = sdf_op("subtract", [_sphere(r=4), _sphere(-0.5), _sphere(0.5)])
    assert _sides(tree) == (LOWER_BOUND, EXACT)


def test_erosion_preserves_only_the_inside_contract():
    tree = sdf_modifier("round", sdf_primitive("box", b=[2, 2, 2]), r=-1)
    assert _sides(tree) == (LOWER_BOUND, EXACT)


def test_unverified_query_maps_do_not_inherit_exactness():
    tree = sdf_transform("rotate_matrix", _sphere(), R=[[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert _sides(tree) == (UNKNOWN, UNKNOWN)


@pytest.mark.parametrize("cached", [None, 100], ids=["uncached", "stale-cache"])
def test_derived_parameters_do_not_supply_a_false_disjointness_proof(cached):
    tree = sdf_op("union", [_sphere(), _sphere({"$ref": "offset"})])
    part = _part(
        tree,
        params={"offset": Param("offset", cached, expr={"type": "num", "value": 1})},
    )
    assert _sides(tree, part) == (UNKNOWN, UNKNOWN)
    assert part.params["offset"].value is cached


def test_spindle_torus_is_not_exact_inside_even_though_its_boundary_is_correct():
    tree = sdf_primitive("torus", t=[1, 2])
    assert _sides(tree) == (EXACT, LOWER_BOUND)
    fn = make_sdf_closure(tree, _part(tree))
    assert float(fn(jnp.zeros(3))) == pytest.approx(-1, abs=1e-6)
    # The nearest exposed boundary is on the axis at z = sqrt(minor^2-major^2).
    assert float(fn(jnp.array([0, 0, np.sqrt(3)]))) == pytest.approx(0, abs=1e-6)
    assert _sides(sdf_primitive("torus", t=[3, 1])) == (EXACT, EXACT)


def test_capped_arc_tubes_do_not_claim_unproven_interior_reach():
    tree = sdf_primitive("capped_torus", sc=[0.01, -1], ra=1, rb=0.2)
    assert _sides(tree) == (EXACT, LOWER_BOUND)


def test_tube_radius_parameter_bounds_must_preserve_the_hole():
    tree = sdf_primitive("torus", t=[1, {"$ref": "r"}])
    part = _part(tree, params={"r": Param("r", 0.2, free=True, bounds=(0.1, 2))})
    assert _sides(tree, part) == (EXACT, LOWER_BOUND)
    assert _sides(tree, part, mode="values") == (EXACT, EXACT)
