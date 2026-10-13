"""`count_part_bodies` / `check_part_is_one_body` on parts whose piece count is known."""

from __future__ import annotations

import warnings

import pytest

from software_defined_matter import MaterialRegion, Part, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.sdf.bodies import (
    PartBodiesWarning,
    check_part_is_one_body,
    count_part_bodies,
)


def _part(*trees, bbox=None):
    mats = [
        MaterialRegion(material_id=i + 1, name=f"m{i}", sdf_tree=t) for i, t in enumerate(trees)
    ]
    meta = {"bbox": bbox} if bbox is not None else {}
    return Part(name="p", params={}, materials=mats, metadata=meta)


def _sphere(r, x):
    return sdf_transform("translate", sdf_primitive("sphere", r=r), t=[x, 0.0, 0.0])


def test_one_solid_is_one_body() -> None:
    assert count_part_bodies(_part(sdf_primitive("box", b=[5.0, 3.0, 2.0]))).n_bodies == 1


def test_two_separate_spheres_in_one_region_are_two_bodies_and_warn() -> None:
    part = _part(
        sdf_op("union", [_sphere(2.0, -6.0), _sphere(3.0, 6.0)]),
        bbox=[[-10, -4, -4], [10, 4, 4]],
    )
    with pytest.warns(PartBodiesWarning, match="2 disconnected pieces"):
        result = check_part_is_one_body(part)
    assert result.n_bodies == 2
    assert result.voxels[0] > result.voxels[1]  # the bigger sphere first


def test_a_co_made_multi_material_part_is_one_body() -> None:
    """Two copper rods inside a steel block: the copper alone is two pieces,
    the part as a whole is one connected object, so no warning."""
    steel = sdf_op(
        "subtract",
        [
            sdf_primitive("box", b=[8.0, 4.0, 4.0]),
            sdf_op("union", [_sphere(1.5, -4.0), _sphere(1.5, 4.0)]),
        ],
    )
    copper = sdf_op("union", [_sphere(1.5, -4.0), _sphere(1.5, 4.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", PartBodiesWarning)
        result = check_part_is_one_body(_part(steel, copper, bbox=[[-9, -5, -5], [9, 5, 5]]))
    assert result.n_bodies == 1


def test_separate_bodies_in_separate_regions_are_still_separate() -> None:
    part = _part(_sphere(2.0, -6.0), _sphere(2.0, 6.0), bbox=[[-9, -3, -3], [9, 3, 3]])
    assert count_part_bodies(part).n_bodies == 2


def test_a_ring_is_one_body() -> None:
    """A torus: connected all the way round, the labelling must not split it."""
    part = _part(sdf_primitive("torus", t=[6.0, 1.5]), bbox=[[-8, -8, -2], [8, 8, 2]])
    assert count_part_bodies(part, n_cells=48).n_bodies == 1


def test_an_empty_part_has_no_bodies() -> None:
    assert count_part_bodies(Part(name="e", params={}, materials=[], metadata={})).n_bodies == 0
