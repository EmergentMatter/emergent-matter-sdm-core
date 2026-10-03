"""The topology oracle every mesher and every mesh test is judged by.

"Watertight and single-bodied" passes for a mesher that sealed a torus shut, so
:mod:`software_defined_matter._meshing.topology` counts edges by how many faces
use them, computes ``chi = V - E + F``, and checks that no vertex has two
surface sheets meeting at a point.

These tests use hand-built meshes whose answers are known without running a
mesher, so a failure elsewhere is never blamed on the yardstick. The
adversarial cases here are connectivity defects, not complicated shapes: a
bowtie vertex, an edge used by four faces, a duplicated face, a flipped
winding. Whether a *mesher* gets the genus of a pierced-box solid right is
tested against the mesher, in ``test_export.py`` and the per-mesher files.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from software_defined_matter._meshing.topology import (  # noqa: E402
    NonManifoldError,
    assert_closed_manifold,
    edge_census,
    euler_characteristic,
    n_bodies,
    n_pinched_vertices,
)

#: Regular tetrahedron: 4 vertices, 6 edges, 4 faces, chi = 2, wound outward.
TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


def _shifted_tet(offset: int, shared: dict[int, int] | None = None) -> np.ndarray:
    """A second tetrahedron, optionally re-using some of the first's vertices."""
    faces = TET_FACES + offset
    for old, new in (shared or {}).items():
        faces[faces == old + offset] = new
    return faces


def _torus_grid(n_major: int = 8, n_minor: int = 5) -> np.ndarray:
    """Triangulated quad grid on a torus. Genus 1, so chi = 0."""
    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = i * n_minor + (j + 1) % n_minor
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = ((i + 1) % n_major) * n_minor + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(faces)


# ---------------------------------------------------------------------------
# Closed, well-formed surfaces
# ---------------------------------------------------------------------------


def test_closed_tetrahedron_is_manifold_with_chi_2():
    census = edge_census(TET_FACES)
    assert census == (0, 0, 6)
    assert census.is_closed_manifold
    assert euler_characteristic(TET_FACES) == 2
    assert n_bodies(TET_FACES) == 1
    assert n_pinched_vertices(TET_FACES) == 0


def test_torus_has_chi_zero_and_stays_manifold():
    faces = _torus_grid()
    assert assert_closed_manifold(faces).is_closed_manifold
    assert euler_characteristic(faces) == 0
    assert n_pinched_vertices(faces) == 0
    assert n_bodies(faces) == 1


def test_chi_and_census_ignore_winding():
    """Topology is about which vertices meet, not which way a face is wound."""
    flipped = TET_FACES[:, [0, 2, 1]]
    assert edge_census(flipped) == edge_census(TET_FACES)
    assert euler_characteristic(flipped) == euler_characteristic(TET_FACES)
    assert n_pinched_vertices(flipped) == 0


def test_unreferenced_vertices_do_not_change_chi():
    """Tetra and DC both carry points no triangle ends up using."""
    with_orphans = np.concatenate([TET_FACES, TET_FACES])[:4]  # same 4 vertices
    assert euler_characteristic(with_orphans) == 2


# ---------------------------------------------------------------------------
# The defects the oracle exists to catch
# ---------------------------------------------------------------------------


def test_open_surface_reports_boundary_edges():
    """Drop one face: three edges are left with a single user."""
    open_faces = TET_FACES[:3]
    census = edge_census(open_faces)
    assert census.boundary == 3
    assert census.non_manifold == 0
    assert not census.is_closed_manifold
    assert euler_characteristic(open_faces) == 1  # a disc, not a sphere


def test_sheets_welded_along_an_edge_are_non_manifold():
    """Two tetrahedra sharing an edge: that edge is used by four faces."""
    faces = np.concatenate([TET_FACES, _shifted_tet(4, {0: 0, 1: 1})])
    census = edge_census(faces)
    assert census.non_manifold == 1
    assert census.boundary == 0
    assert not census.is_closed_manifold


def test_bowtie_vertex_passes_the_edge_census_and_is_still_caught():
    """Two tetrahedra glued at one vertex only.

    Every edge still has exactly two faces, so an edge census calls this
    closed and manifold. It is not: a neighbourhood of the shared vertex is
    two discs. Quadric collapse across a thin handle produces this shape, so
    the check is what the decimation guard leans on.
    """
    faces = np.concatenate([TET_FACES, _shifted_tet(4, {0: 0})])
    census = edge_census(faces)
    assert census.boundary == 0
    assert census.non_manifold == 0
    assert census.is_closed_manifold  # the edge census is fooled

    assert n_pinched_vertices(faces) == 1
    with pytest.raises(NonManifoldError, match="1 pinched vertex"):
        assert_closed_manifold(faces)


def test_duplicated_face_shows_up_as_non_manifold_edges():
    faces = np.concatenate([TET_FACES, TET_FACES[:1]])
    census = edge_census(faces)
    assert census.non_manifold == 3  # the duplicate's three edges now have 3 users


def test_assert_closed_manifold_names_the_counts():
    with pytest.raises(NonManifoldError, match="3 of 6 edges are used by one face"):
        assert_closed_manifold(TET_FACES[:3], context="open tetra")


def test_assert_closed_manifold_rejects_empty():
    with pytest.raises(NonManifoldError, match="no faces"):
        assert_closed_manifold(np.zeros((0, 3), dtype=int))


# ---------------------------------------------------------------------------
# Body counting
# ---------------------------------------------------------------------------


def test_disjoint_bodies_are_counted_separately():
    two = np.concatenate([TET_FACES, _shifted_tet(4)])
    assert n_bodies(two) == 2
    assert euler_characteristic(two) == 4  # 2 per component


def test_bodies_are_counted_through_shared_edges_not_shared_points():
    """Two solids touching at a single vertex count as two.

    This is the same convention trimesh's ``split`` uses (face adjacency runs
    through edges), and it is the one the decimation body-count guard compares
    against, so it is pinned here rather than left to a library default.
    """
    touching = np.concatenate([TET_FACES, _shifted_tet(4, {0: 0})])
    assert n_bodies(touching) == 2


def test_empty_mesh_censuses_to_zero():
    empty = np.zeros((0, 3), dtype=int)
    assert edge_census(empty) == (0, 0, 0)
    assert euler_characteristic(empty) == 0
    assert n_bodies(empty) == 0
    assert n_pinched_vertices(empty) == 0
