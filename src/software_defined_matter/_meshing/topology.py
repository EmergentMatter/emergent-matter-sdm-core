"""Topology predicates shared by every polygoniser and every mesh test.

"Watertight and single-bodied" is not a topology check: it still passes for a
mesher that sealed a torus' tunnel shut, or for a decimation that filled a hole
it should have kept. What separates those cases is the edge census (how many
faces use each undirected edge) and the Euler characteristic
``chi = V - E + F``, which for a closed orientable surface is ``2 - 2 * genus``.
A sphere is 2, a torus 0, a gyroid strongly negative, and an *odd* value is not
a closed orientable surface at all.

Every mesher in :mod:`software_defined_matter._meshing` and every test that
asserts on mesh topology uses the functions here, so the oracle is written once.

Pure NumPy plus ``scipy.sparse.csgraph`` (already an install dependency via
scikit-image), so importing this module never pulls in ``trimesh``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class EdgeCensus(NamedTuple):
    """Counts of undirected edges by how many faces use them.

    ``boundary`` is edges used by exactly one face (a hole), ``non_manifold``
    edges used by three or more (two surface sheets welded along an edge), and
    ``total`` every distinct undirected edge. A closed manifold surface has
    ``boundary == 0`` and ``non_manifold == 0``.
    """

    boundary: int
    non_manifold: int
    total: int

    @property
    def is_closed_manifold(self) -> bool:
        return self.boundary == 0 and self.non_manifold == 0


def edge_census(faces: np.ndarray) -> EdgeCensus:
    """Count boundary and non-manifold edges of a triangle soup.

    Faces are ``(F, 3)`` vertex indices. Edges are compared as unordered pairs,
    so winding does not affect the counts.
    """
    faces = np.asarray(faces)
    if len(faces) == 0:
        return EdgeCensus(0, 0, 0)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _uniq, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return EdgeCensus(
        boundary=int((counts == 1).sum()),
        non_manifold=int((counts > 2).sum()),
        total=int(len(counts)),
    )


def euler_characteristic(faces: np.ndarray) -> int:
    """``chi = V - E + F`` counting only vertices the faces actually reference.

    Unreferenced vertices are ignored so a mesh carrying leftover points (dual
    contouring cells that ended up unused, tetrahedralization corner points)
    reports the characteristic of the surface rather than of the vertex array.
    """
    faces = np.asarray(faces)
    if len(faces) == 0:
        return 0
    n_verts = len(np.unique(faces))
    census = edge_census(faces)
    return int(n_verts - census.total + len(faces))


def n_bodies(faces: np.ndarray) -> int:
    """Number of connected components, counting faces joined by a shared edge.

    Uses ``scipy.sparse.csgraph.connected_components`` over the edge-sharing
    graph rather than a Python union-find: on a production mesh (millions of
    faces, ~1.5 adjacency pairs per face) the pure-Python loop dominates the
    runtime of everything it is called from.
    """
    faces = np.asarray(faces)
    if len(faces) == 0:
        return 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_faces = len(faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    keyed = np.sort(edges, axis=1)
    order = np.lexsort((keyed[:, 1], keyed[:, 0]))
    keyed, owner = keyed[order], owner[order]

    # Consecutive rows sharing an edge key are faces that touch; linking each
    # to the next one in the run connects the whole run.
    same = np.all(keyed[1:] == keyed[:-1], axis=1)
    rows, cols = owner[:-1][same], owner[1:][same]
    if len(rows) == 0:
        return n_faces
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_faces, n_faces),
    )
    return int(connected_components(graph, directed=False, return_labels=False))


def n_pinched_vertices(faces: np.ndarray) -> int:
    """Vertices where two surface sheets meet at a point (a "bowtie").

    An edge census cannot see this one. Take two cones and glue their tips: no
    edge is used by more than two faces, yet the surface is not a manifold,
    because a neighbourhood of the tip is two discs rather than one. Quadric
    edge collapse across a thin handle produces exactly this, which is why the
    decimation guard needs it.

    The test is on each vertex's fan: the faces around a manifold vertex form
    one cycle, reachable from one another through edges that contain the
    vertex. Working on face corners (one node per ``(face, vertex)`` pair)
    turns "every fan is connected" into a single sparse connected-components
    call: a closed mesh has exactly one component per referenced vertex, and
    each extra component is one extra sheet pinched onto a vertex.
    """
    faces = np.asarray(faces)
    if len(faces) == 0:
        return 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_faces = len(faces)
    # Node 3*f + k is corner k of face f, sitting at vertex faces[f, k].
    n_corners = 3 * n_faces

    # One row per (face, edge): the edge key, the owning face, and the corner
    # slots holding the edge's low and high vertex.
    slot_a = np.repeat(np.arange(3, dtype=np.int64), n_faces)
    slot_b = (slot_a + 1) % 3
    owner = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    va, vb = faces[owner, slot_a], faces[owner, slot_b]
    keyed = np.stack([np.minimum(va, vb), np.maximum(va, vb)], axis=1)
    a_is_low = va == keyed[:, 0]
    slot_low = np.where(a_is_low, slot_a, slot_b)
    slot_high = np.where(a_is_low, slot_b, slot_a)

    order = np.lexsort((keyed[:, 1], keyed[:, 0]))
    keyed, owner = keyed[order], owner[order]
    slot_low, slot_high = slot_low[order], slot_high[order]
    same = np.all(keyed[1:] == keyed[:-1], axis=1)

    if not same.any():
        n_components = n_corners
    else:
        left = np.flatnonzero(same)
        right = left + 1
        # Faces sharing an edge join their corners at each shared endpoint, so
        # a fan stays one component only while its faces chain around edges
        # that touch the vertex. Two cones glued at a tip never chain.
        rows = np.concatenate([3 * owner[left] + slot_low[left], 3 * owner[left] + slot_high[left]])
        cols = np.concatenate(
            [3 * owner[right] + slot_low[right], 3 * owner[right] + slot_high[right]]
        )
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n_corners, n_corners),
        )
        n_components = int(connected_components(graph, directed=False, return_labels=False))
    return int(n_components - len(np.unique(faces)))


class NonManifoldError(Exception):
    """Raised when a surface that must be a closed 2-manifold is not one."""


def assert_closed_manifold(faces: np.ndarray, *, context: str = "") -> EdgeCensus:
    """Fail loud unless the surface is closed, edge-manifold, and unpinched.

    Returns the census so a caller that wants to log it does not recount.

    Raises:
        NonManifoldError: If the mesh is empty, has boundary edges (a hole),
            has edges used by three or more faces (two sheets welded along an
            edge), or has a vertex where two sheets meet at a point.
    """
    prefix = f"[{context}] " if context else ""
    census = edge_census(faces)
    if census.total == 0:
        raise NonManifoldError(f"{prefix}Mesh has no faces")
    if not census.is_closed_manifold:
        raise NonManifoldError(
            f"{prefix}Surface is not a closed manifold: {census.boundary} of "
            f"{census.total} edges are used by one face (holes) and "
            f"{census.non_manifold} by three or more (welded sheets)."
        )
    n_pinched = n_pinched_vertices(faces)
    if n_pinched:
        raise NonManifoldError(
            f"{prefix}Surface has {n_pinched} pinched vertex/vertices: every "
            f"edge is shared by two faces, but two sheets meet at a point."
        )
    return census


__all__ = [
    "EdgeCensus",
    "NonManifoldError",
    "assert_closed_manifold",
    "edge_census",
    "euler_characteristic",
    "n_bodies",
    "n_pinched_vertices",
]
