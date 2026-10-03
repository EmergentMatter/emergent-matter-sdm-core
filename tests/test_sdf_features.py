"""The thinnest-wall walk that the metric resolution check runs on.

``infer_min_feature_size`` reports the smallest wall thickness it can name, so
:func:`~software_defined_matter.objectives.metrics.check_grid_resolution` can
refuse to integrate geometry the grid cannot resolve. These tests pin what it
names, what it deliberately does not, and that ``None`` means "no evidence"
rather than "nothing thin here".
"""

from __future__ import annotations

import math

import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    make_param_ref,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.sdf.features import infer_min_feature_size


def _size(tree, mode="values", **params):
    part = Part(
        name="t", params=params, materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)]
    )
    return infer_min_feature_size(tree, part, mode=mode)


# ---------------------------------------------------------------------------
# Does infer_min_feature_size return the primitive's actual smallest dimension?
# ---------------------------------------------------------------------------


def test_tpms_reports_its_wall_thickness():
    assert _size(
        sdf_primitive("gyroid", period=5.0, min_thickness=0.8, n_periods=[4, 4, 4])
    ) == pytest.approx(0.8)


def test_onion_reports_its_wall():
    tree = sdf_modifier("onion", sdf_primitive("sphere", r=5.0), thickness=0.4)
    assert _size(tree) == pytest.approx(0.4)


def test_box_reports_its_shortest_full_side():
    # `b` is a half-extent, so a full side is twice it.
    assert _size(sdf_primitive("box", b=[10.0, 3.0, 0.5])) == pytest.approx(1.0)


def test_sphere_reports_its_diameter():
    assert _size(sdf_primitive("sphere", r=0.25)) == pytest.approx(0.5)


def test_cylinder_reports_the_smaller_of_diameter_and_height():
    assert _size(sdf_primitive("capped_cylinder", r=4.0, h=0.3)) == pytest.approx(0.6)
    assert _size(sdf_primitive("capped_cylinder", r=0.2, h=9.0)) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Does a rotation hide a thin wall? (It must not: an oblique thin wall is
# exactly the case the clip kernel gets wrong, so it is the case the check
# most needs to catch.)
# ---------------------------------------------------------------------------


def test_a_thin_disc_still_reports_its_thickness_when_rotated():
    disc = sdf_primitive("capped_cylinder", r=5.0, h=0.1)  # 0.2 mm thick
    assert _size(disc) == pytest.approx(0.2)
    tilted = sdf_transform("rotate_y", disc, angle=math.radians(45.0))
    assert _size(tilted) == pytest.approx(0.2)


def test_a_thin_disc_reports_through_a_stack_of_transforms():
    disc = sdf_primitive("capped_cylinder", r=5.0, h=0.1)
    buried = sdf_transform(
        "rotate_z",
        sdf_transform(
            "translate",
            sdf_transform("rotate_y", disc, angle=math.radians(37.0)),
            t=[4.0, -2.0, 1.0],
        ),
        angle=math.radians(63.0),
    )
    assert _size(buried) == pytest.approx(0.2)


def test_rigid_transforms_do_not_change_a_thickness():
    thin = sdf_primitive("box", b=[5.0, 5.0, 0.25])
    moved = sdf_transform(
        "rotate_z", sdf_transform("translate", thin, t=[3.0, 0.0, 0.0]), angle=0.7
    )
    assert _size(moved) == pytest.approx(0.5)


def test_scale_does_change_a_thickness():
    thin = sdf_primitive("box", b=[5.0, 5.0, 0.5])
    assert _size(sdf_transform("scale", thin, s=0.1)) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Does the walk find the smallest feature anywhere in the tree, cut or kept?
# ---------------------------------------------------------------------------


def test_takes_the_minimum_across_a_union():
    tree = sdf_op("union", [sdf_primitive("sphere", r=5.0), sdf_primitive("sphere", r=0.3)])
    assert _size(tree) == pytest.approx(0.6)


def test_a_thin_cutter_counts_too():
    # A narrow slot is as hard to resolve as a narrow rib, and getting it wrong
    # removes the wrong amount of material.
    tree = sdf_op(
        "subtract",
        [
            sdf_primitive("box", b=[10.0, 10.0, 10.0]),
            sdf_primitive("capped_cylinder", r=0.1, h=11.0),
        ],
    )
    assert _size(tree) == pytest.approx(0.2)


def test_a_tilted_thin_cutter_inside_a_solid_is_found():
    slot = sdf_transform(
        "rotate_x", sdf_primitive("box", b=[9.0, 9.0, 0.15]), angle=math.radians(45.0)
    )
    tree = sdf_op("subtract", [sdf_primitive("sphere", r=8.0), slot])
    assert _size(tree) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# What does it do when nothing in the tree is nameable?
# ---------------------------------------------------------------------------


def test_unnameable_geometry_returns_none():
    # A cone tapers to a point; no grid resolves that, and reporting the point
    # would make every cone unmeasurable. It contributes nothing instead.
    assert _size(sdf_primitive("cone", h=5.0, c=[0.6, 0.8])) is None


def test_unknown_node_does_not_raise():
    assert _size({"type": "not_a_real_node"}) is None


def test_known_sibling_still_reports_past_an_unknown_one():
    tree = sdf_op(
        "union",
        [sdf_primitive("cone", h=5.0, c=[0.6, 0.8]), sdf_primitive("box", b=[9.0, 9.0, 0.35])],
    )
    assert _size(tree) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Do param bounds change which thickness is reported?
# ---------------------------------------------------------------------------


def test_bounds_mode_takes_the_thinnest_the_optimiser_may_reach():
    tree = sdf_primitive(
        "gyroid", period=5.0, min_thickness=make_param_ref("t"), n_periods=[4, 4, 4]
    )
    p = {"t": Param("t", 1.0, free=True, bounds=(0.2, 2.0))}
    assert _size(tree, mode="values", **p) == pytest.approx(1.0)
    assert _size(tree, mode="bounds", **p) == pytest.approx(0.2)


def test_a_thickness_that_can_reach_zero_reports_zero():
    tree = sdf_primitive(
        "gyroid", period=5.0, min_thickness=make_param_ref("t"), n_periods=[4, 4, 4]
    )
    p = {"t": Param("t", 1.0, free=True, bounds=(0.0, 2.0))}
    assert _size(tree, mode="bounds", **p) == 0.0
