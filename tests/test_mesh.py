"""Marching cubes: the degenerate-crossing case, and what cleanup leaves.

A sample that lands exactly on the isosurface (``sdf == 0``) makes marching
cubes put a crossing at a grid corner. The triangles around it come out with
two coincident vertices, cleanup drops them as degenerate, and the mesh is left
with holes and the wrong genus. The blob fixture below hits this at voxel 0.5,
where a planar CSG face lands on the grid and 201 corners evaluate to exactly
zero.
"""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("skimage")

from software_defined_matter import (  # noqa: E402
    MaterialRegion,
    Part,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter._meshing import MarchingCubesConfig, MeshCleanupConfig  # noqa: E402
from software_defined_matter._meshing.mesh import (  # noqa: E402
    cleanup_mesh,
    extract_mesh,
    snap_iso_degeneracies,
)
from software_defined_matter._meshing.topology import (  # noqa: E402
    edge_census,
    euler_characteristic,
    n_pinched_vertices,
)
from software_defined_matter.grid_sampling import (  # noqa: E402
    bind_sdf,
    eval_chunked,
    make_grid,
    material_bbox,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _blob_part():
    """Spheres unioned, then a box subtracted.

    The box's planar faces are axis-aligned at integer coordinates, so on a
    0.5 mm grid they land exactly on grid corners. The subtraction also leaves
    a through-slot, so the solid is genus 1 and chi = 0.
    """
    tree = sdf_primitive("sphere", r=3.0)
    for t in ([2, 0, 0], [-2, 1, 0], [0, 2, 1], [1, -2, 0]):
        tree = sdf_op(
            "union",
            [tree, sdf_transform("translate", sdf_primitive("sphere", r=2.0), t=t)],
        )
    tree = sdf_op(
        "subtract",
        [tree, sdf_transform("translate", sdf_primitive("box", b=[1, 1, 5]), t=[0, 0, 0])],
    )
    return Part(
        name="blob",
        materials=[MaterialRegion(material_id=1, name="M", sdf_tree=tree)],
        metadata={"bbox": [[-6, -6, -6], [6, 6, 6]]},
    )


def _sample(part, voxel):
    """Grid values, origin, and the bound SDF, as ``export_part`` builds them."""
    region = part.materials[0]
    bbox = material_bbox(region, part).padded(voxel * 2)
    points, shape = make_grid(bbox, voxel)
    sdf = bind_sdf(region.sdf_tree, part)
    return eval_chunked(sdf, points, 1_000_000).reshape(shape), bbox.min_pt, sdf


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


def test_blob_grid_really_has_samples_on_the_isosurface():
    """Without this the rest of the file would pass for the wrong reason."""
    grid, _origin, _sdf = _sample(_blob_part(), 0.5)
    assert int((grid.astype(np.float32) == 0.0).sum()) == 201


def test_snap_moves_only_the_exact_samples_and_only_inward():
    grid = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)
    before = grid.copy()
    assert snap_iso_degeneracies(grid, 0.0) == 1
    assert grid[0, 0, 0] == before[0, 0, 0]
    assert grid[0, 0, 2] == before[0, 0, 2]
    assert grid[0, 0, 1] < 0.0  # interior side: the solid is {f <= iso}
    assert grid[0, 0, 1] > -1e-5  # and by a hair


def test_snap_survives_the_float32_cast_at_a_nonzero_iso_level():
    """8 * spacing() is what keeps the nudge from rounding back to the level."""
    grid = np.full((1, 1, 1), 100.0, dtype=np.float32)
    assert snap_iso_degeneracies(grid, 100.0) == 1
    assert grid[0, 0, 0] < np.float32(100.0)


def test_snap_is_a_noop_when_nothing_sits_on_the_isosurface():
    grid = np.array([[[-1.0, 1.0]]], dtype=np.float32)
    before = grid.copy()
    assert snap_iso_degeneracies(grid, 0.0) == 0
    assert np.array_equal(grid, before)


def test_snap_catches_near_zero_samples_not_only_exact_ones():
    """A sample 1e-17 off the isosurface is just as degenerate as one on it.

    Whether a field lands on exact zero depends on float32 against float64
    evaluation, so an equality test fixes one part and leaves the identical
    geometry broken in the other precision.
    """
    grid = np.array([[[-1.0, 1e-17, 1e-9, 0.5]]], dtype=np.float32)
    assert snap_iso_degeneracies(grid, 0.0) == 2
    assert grid[0, 0, 1] < 0.0
    assert grid[0, 0, 2] < 0.0
    assert grid[0, 0, 3] == np.float32(0.5)


# ---------------------------------------------------------------------------
# End to end: the mesh the guard is there to produce
# ---------------------------------------------------------------------------


def test_blob_meshes_closed_and_manifold_with_the_guard_on():
    grid, origin, _sdf = _sample(_blob_part(), 0.5)
    mesh = cleanup_mesh(extract_mesh(grid, origin, 0.5), MeshCleanupConfig())

    census = edge_census(mesh.faces)
    assert census.boundary == 0
    assert census.non_manifold == 0
    assert n_pinched_vertices(mesh.faces) == 0
    assert euler_characteristic(mesh.faces) == 0  # genus 1: the box left a slot
    assert trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False).is_watertight


def test_blob_leaks_with_the_guard_off():
    """The failure this commit fixes, pinned so it cannot come back unnoticed."""
    grid, origin, _sdf = _sample(_blob_part(), 0.5)
    unguarded = extract_mesh(
        grid, origin, 0.5, config=MarchingCubesConfig(snap_iso_degeneracies=False)
    )
    raw = edge_census(unguarded.faces)
    assert raw.boundary > 0

    cleaned = cleanup_mesh(unguarded, MeshCleanupConfig())
    census = edge_census(cleaned.faces)
    assert census.boundary > 0
    assert euler_characteristic(cleaned.faces) != 0  # wrong genus


def _volume(part, voxel, *, guard=True):
    grid, origin, _sdf = _sample(part, voxel)
    mesh = cleanup_mesh(
        extract_mesh(grid, origin, voxel, config=MarchingCubesConfig(snap_iso_degeneracies=guard)),
        MeshCleanupConfig(),
    )
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False).volume


def test_guard_snaps_inward_so_the_part_is_not_eroded():
    """The direction of the nudge is worth a test of its own.

    Snapping a boundary sample outward looks equally arbitrary and is not: it
    erodes every planar face that lands on the grid. On this fixture at voxel
    0.5 that cost 2.6% of the volume (129.38 mm^3 against a 4M-sample Monte
    Carlo value of 132.84 +/- 0.46), which is far more than the topology fix
    is worth. Snapping inward lands on 132.50 and agrees with the same mesh
    at half the voxel size.
    """
    part = _blob_part()
    coarse = _volume(part, 0.5)
    fine = _volume(part, 0.25)
    assert abs(coarse - fine) / fine < 0.01


def test_mesh_without_iso_degeneracies_is_untouched():
    """r = 4.77 puts no grid corner on the surface, so the guard is a no-op."""
    part = Part(
        name="s",
        materials=[
            MaterialRegion(material_id=1, name="M", sdf_tree=sdf_primitive("sphere", r=4.77))
        ],
        metadata={"bbox": [[-6, -6, -6], [6, 6, 6]]},
    )
    grid, origin, _sdf = _sample(part, 0.5)
    assert int((grid.astype(np.float32) == 0.0).sum()) == 0

    guarded = extract_mesh(grid, origin, 0.5)
    unguarded = extract_mesh(
        grid, origin, 0.5, config=MarchingCubesConfig(snap_iso_degeneracies=False)
    )
    assert np.array_equal(guarded.vertices, unguarded.vertices)
    assert np.array_equal(guarded.faces, unguarded.faces)
