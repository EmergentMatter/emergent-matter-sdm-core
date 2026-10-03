"""Mesh types for isosurface extraction, cleanup, and decimation.

These are the contracts between extraction, validation, and cleanup in
:mod:`software_defined_matter._meshing`. The marching-cubes types are ported
**field-for-field** from an upstream mesh pipeline
(same field names, same defaults; not part of this repo). See
``docs/adr/0002-preview-and-export-modules.md``.

:class:`DualContourConfig` and :class:`TetraConfig` tune the two optional
polygonisers reached through ``export_part(method=...)``. Marching cubes stays
the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class MeshData(NamedTuple):
    """Triangle mesh produced by marching cubes.

    ``vertices`` is ``(V, 3)`` float in world coordinates.
    ``faces`` is ``(F, 3)`` int (vertex indices).
    """

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3)


@dataclass(frozen=True)
class MeshCleanupConfig:
    """Controls post-marching-cubes mesh cleanup.

    Conservative defaults: fix normals / winding / duplicates ON,
    ``fill_holes`` and ``keep_only_largest_component`` OFF (both
    topology-changing: opt-in only).
    """

    fix_normals: bool = True
    fix_winding: bool = True
    remove_duplicates: bool = True
    fill_holes: bool = False  # opt-in: topology-changing
    # When True, split into connected components and keep only the one with
    # the most faces. Use when the caller KNOWS the part is a single solid
    # and any dangling islands are marching-cubes artifacts to discard.
    keep_only_largest_component: bool = False


@dataclass(frozen=True)
class MarchingCubesConfig:
    """Controls the default polygoniser (``method="marching_cubes"``)."""

    #: Nudge grid samples that sit exactly on the isosurface (``sdf = 0``, or
    #: whatever ``iso_level`` the caller passed) to the exterior side before
    #: contouring. A sample of exactly the iso-level makes marching cubes place
    #: a crossing at a grid corner, which emits zero-area triangles and leaves
    #: the welded mesh non-manifold; a planar CSG face landing on the grid does
    #: this on hundreds of corners at once. The nudge is about 1e-6 mm, far
    #: less than the voxel already moves the surface.
    snap_iso_degeneracies: bool = True


@dataclass(frozen=True)
class DualContourConfig:
    """Controls the Dual Contouring mesher (``method="dual_contour"``).

    DC consults the SDF gradient to place a QEF vertex inside each surface
    cell, so it keeps sharp features (chamfers, slot mouths) that marching
    cubes bevels at voxel scale. These knobs are "how to mesh"; the ones in
    :class:`MeshCleanupConfig` are "how to clean the mesh afterwards".

    Defaults reproduce DC at roughly marching-cubes density: manifold vertex
    splitting on, no flat-region simplification.
    """

    #: Flat-region triangle reduction strength in ``[0, 1]``, mapped to a
    #: fraction of a voxel. ``0.0`` disables the coplanar merge entirely.
    #: Ignored when ``simplify_error_mm`` is given.
    adaptivity: float = 0.0
    #: Explicit surface-deviation limit (mm) for the coplanar merge, bounding
    #: how far the merged surface may sit from the unmerged one. ``None``
    #: derives a limit from ``adaptivity``.
    simplify_error_mm: float | None = None
    #: QEF singular-value truncation ratio, relative to the largest singular
    #: value. Larger truncates more aggressively and rounds corners off; 0.1
    #: keeps sharp features crisp while staying numerically stable.
    svd_truncation: float = 0.1
    #: Emit one vertex per connected surface sheet in a cell rather than one
    #: vertex per cell. A cell whose surface has two disjoint sheets otherwise
    #: welds them at a single vertex, which leaves the surrounding edges used
    #: by four triangles. Turn off only to reproduce classic DC.
    manifold_vertices: bool = True


@dataclass(frozen=True)
class TetraConfig:
    """Controls the tetrahedral mesher (``method="tetra"``).

    Every triangle is a face of a Delaunay tetrahedralization, so the output
    has no self-intersections and every interior face is shared by exactly two
    tetrahedra. See :mod:`software_defined_matter._meshing.tetra` for the
    algorithm and its provenance.

    Defaults favour correctness over speed: sharp features on, and enough
    refinement rounds that thin walls close without the caller tuning anything.
    """

    #: Bisection steps per surface point. Each halves the bracket, so ``n``
    #: steps land the point within ``2**-n`` of the edge length; 20 is about
    #: 1e-6 of a voxel. The loop is one JIT-compiled ``fori_loop``, so raising
    #: this is nearly free.
    bisection_iters: int = 20
    #: Solve a QEF per active cell and keep the cells holding a real feature
    #: (rank >= 2), so chamfers and slot mouths stay crisp.
    sharp_features: bool = True
    #: QEF singular-value truncation ratio, as in :class:`DualContourConfig`.
    #: Also sets the rank test deciding whether a cell holds a feature.
    svd_truncation: float = 0.1
    #: Reject a QEF point whose ``|f|`` exceeds this many voxels. A minimiser
    #: is not guaranteed to land on the surface and is about to be tagged as
    #: though it did.
    feature_tol_voxels: float = 0.75
    #: Fractional margin keeping every inserted surface point strictly interior
    #: to its segment. This is what makes the sheet-closing loop terminate: a
    #: point coincident with an endpoint does not break the Delaunay edge, so
    #: the same gap is rediscovered every round. Costs at most
    #: ``segment_eps * edge_length`` of surface accuracy. Also sets the
    #: near-duplicate merge radius for newly inserted points.
    segment_eps: float = 1e-3
    #: Maximum outer topology rounds. Each closes the sheet, classifies, and
    #: repairs pinches; a pinch that resists reclassification triggers point
    #: injection and another round. Typical geometry needs one.
    max_topology_rounds: int = 6
    #: Maximum pinch-repair rounds. Exceeding this raises rather than returning
    #: a non-manifold surface.
    max_pinch_rounds: int = 12
    #: Maximum sheet-closing rounds. Each bisects every inside-to-outside
    #: tetrahedron edge and re-tetrahedralizes. Typical geometry converges in
    #: 0-2 rounds; thin unaligned walls need more.
    max_refine_iters: int = 8
    #: Refuse a job whose estimated face count exceeds this. Measured cost is
    #: roughly 6,400 faces/s and 8 kB of resident memory per face, so a part
    #: far above this cap will exhaust memory rather than finish. Marching
    #: cubes plus decimation is the path for production-size parts.
    max_faces: int = 300_000
    #: Reuse one incremental Qhull triangulation across refinement rounds
    #: (``Delaunay(..., incremental=True)`` plus ``add_points``) instead of
    #: rebuilding from scratch each round. Falls back to a rebuild if Qhull
    #: refuses the incremental insert.
    incremental_delaunay: bool = True


@dataclass(frozen=True)
class DecimateConfig:
    """Controls optional quadric decimation after marching cubes + cleanup.

    Collapses low-curvature regions (including curved-path flats such as a
    helical wire ribbon) via quadric edge collapse, bounded by a surface
    deviation limit. The result is guaranteed watertight with the same
    body count as the input: a reduction that leaks, shatters, or drops a
    component is rejected (back off / fall back to the undecimated mesh).
    Operates on the mesh, not the SDF gradient, so it works on non-exact /
    swept / embossed SDFs.

    Give **either** an explicit ``simplify_error_mm`` (preferred physical
    knob; the reduction is searched to stay within it) **or** a direct
    ``target_reduction``.
    """

    #: Surface-deviation limit (mm). The decimated surface stays within this
    #: of the undecimated one; ``target_reduction`` is searched to meet it.
    simplify_error_mm: float | None = None
    #: Direct fraction of triangles to remove in ``[0, 1)`` (e.g. ``0.8`` →
    #: keep ~20%). Used when ``simplify_error_mm`` is ``None``.
    target_reduction: float | None = None
    #: Quadric-collapse aggressiveness (``fast_simplification`` ``agg``):
    #: lower preserves geometry better (slower), higher is faster and
    #: rougher. ``5.0`` is a feature-friendly default below the library's 7.0.
    aggressiveness: float = 5.0
    #: Max binary-search passes when honouring ``simplify_error_mm`` (caps
    #: cost on large meshes; the best acceptable result inside it is kept).
    max_passes: int = 6
    #: Stop the search once the bracket on ``target_reduction`` is narrower
    #: than this, before ``max_passes`` is spent. Further passes past this
    #: point cost a full decimation each and change the face count by under a
    #: percent.
    reduction_tol: float = 0.01
    #: Cap on the deterministic sample set used to measure surface deviation.
    #: Points are taken as mesh vertices, face centroids, and edge midpoints,
    #: strided when the mesh is larger than this, so the measurement is
    #: reproducible run to run rather than drawn at random. Lowering it makes
    #: the bound soft: on the cylinder-and-slab fixture 5,000 points reported
    #: 0.086 mm where 20,000 found 0.141 mm, so the search would have accepted
    #: a mesh outside the limit it was given.
    deviation_samples: int = 20_000
    #: Also measure how far the *input* surface sits from the decimated one
    #: (the old-to-new direction). Without it, decimation that erases a small
    #: boss is invisible: the replacement surface lies flat across the base
    #: where ``|sdf|`` is already near zero.
    two_sided_deviation: bool = True
    #: Require the decimated surface to keep the input's Euler characteristic.
    #: Body count alone does not catch a sealed torus tunnel (one watertight
    #: body before and after, genus 1 to genus 0).
    preserve_euler: bool = True


__all__ = [
    "DecimateConfig",
    "DualContourConfig",
    "MarchingCubesConfig",
    "MeshCleanupConfig",
    "MeshData",
    "TetraConfig",
]
