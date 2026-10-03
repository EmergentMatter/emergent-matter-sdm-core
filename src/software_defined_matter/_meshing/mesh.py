"""Marching-cubes mesh extraction + validation + cleanup.

This is the default polygoniser and the one production parts use: scikit-image
marching cubes, driven by field *values* only, so nothing here depends on the
SDF gradient. Blender's VolumeToMesh / OpenVDB pipeline is not an option, it
introduced visible moiré streaks on a JAX SDF. For adaptive triangle
reduction, pair this with quadric decimation
(:mod:`software_defined_matter._meshing.decimate`, via
``export_part(..., decimate=DecimateConfig(...))``). The two alternative
polygonisers live in :mod:`software_defined_matter._meshing.dual_contour`
(sharp features) and :mod:`software_defined_matter._meshing.tetra` (manifold
by construction).

Marching cubes is close to watertight but not unconditionally so. A sample
that lands exactly on the isosurface makes it place a crossing at a grid
corner, which produces zero-area triangles and, once those are dropped, holes.
:func:`snap_iso_degeneracies` removes that case and runs by default.

``scikit-image`` and ``trimesh`` are optional dependencies (the ``[export]``
extra). They are imported lazily so importing ``_meshing`` without ``.mesh``
never pulls them in; calling any function here without them raises a clear
install hint.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from software_defined_matter._meshing.topology import edge_census
from software_defined_matter._meshing.types import (
    MarchingCubesConfig,
    MeshCleanupConfig,
    MeshData,
)

logger = logging.getLogger(__name__)

try:  # optional [export] extra
    import trimesh
    from skimage.measure import marching_cubes

    _HAVE_EXPORT_DEPS = True
    _IMPORT_ERR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only without extra
    trimesh = None  # type: ignore[assignment]
    marching_cubes = None  # type: ignore[assignment]
    _HAVE_EXPORT_DEPS = False
    _IMPORT_ERR = exc


def _require_export_deps() -> None:
    """Raise a friendly error if the optional ``[export]`` extra is missing."""
    if not _HAVE_EXPORT_DEPS:
        raise ImportError(
            "The mesh export path requires the optional 'export' extra "
            "(scikit-image + trimesh). Install with:\n"
            "    pip install 'emergent-matter-sdm-core[export]'\n"
            "or:\n"
            "    uv pip install 'scikit-image>=0.21' 'trimesh>=4.0'"
        ) from _IMPORT_ERR


class MeshExtractionError(Exception):
    """Raised when mesh extraction produces unusable output.

    Either marching cubes failed outright (no zero-crossing in the grid,
    invalid input, etc.) or the resulting mesh is degenerate (empty,
    non-finite vertices, etc.).
    """


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def snap_iso_degeneracies(grid: np.ndarray, iso_level: float) -> int:
    """Move samples sitting on the isosurface to the interior side.

    A sample of exactly ``iso_level`` is neither inside nor outside, and
    marching cubes resolves it by putting the crossing at the grid corner
    itself. Every cell around that corner then emits a triangle with two
    coincident vertices. Dropping those zero-area triangles (which cleanup
    does) leaves a hole, so the mesh comes out non-watertight with the wrong
    genus. It is not a rare case: a planar CSG face that lands on the grid
    does it to hundreds of corners at once. The blob fixture in
    ``tests/test_mesh.py`` has 201 such corners at voxel 0.5, which produced
    203 zero-area triangles and 160 non-manifold edges.

    The test is ``|f - iso| <= epsilon``, not equality, because a sample that
    misses zero by 1e-17 is just as degenerate: the crossing still lands on
    the corner once float32 rounds it. Which one you get depends on whether
    the field was evaluated in float32 or float64, so equality alone fixes the
    blob and leaves the same chamfer broken under ``jax_enable_x64``. On the
    chamfered box at voxel 0.2 in float64, 8,777 samples sit within 1e-12 of
    the isosurface and snapping them gives 476.0003 mm^3 against an analytic
    476.0, where leaving them gives 474.96.

    Interior is the direction, because the solid is the closed set
    ``{f <= iso}`` and a sample on its boundary belongs to it. The other
    direction is not merely a convention: pushing those samples outward erodes
    every face that lands on the grid, and on the blob at voxel 0.5 it cost
    2.6% of the volume (129.38 mm^3 against a Monte Carlo value of
    132.84 +/- 0.46), while snapping inward gives 132.50.

    ``epsilon`` is ``max(1e-6, 8 * spacing(iso))`` in float32: large enough to
    survive the cast and to keep the crossing clear of trimesh's vertex merge
    tolerance, and far below the accuracy the voxel size already sets.

    Mutates ``grid`` in place and returns how many samples were moved.
    """
    iso32 = np.float32(iso_level)
    epsilon = np.float32(max(1e-6, 8.0 * float(np.spacing(np.float32(abs(iso_level))))))
    degenerate = np.abs(grid - iso32) <= epsilon
    n_degenerate = int(degenerate.sum())
    if n_degenerate:
        grid[degenerate] = iso32 - epsilon
    return n_degenerate


def extract_mesh(
    grid: np.ndarray,
    origin: tuple[float, float, float] | np.ndarray,
    voxel_size: float = 0.4,
    *,
    iso_level: float = 0.0,
    config: MarchingCubesConfig | None = None,
) -> MeshData:
    """Run marching cubes on a 3D SDF grid and return a triangle mesh.

    Vertices are offset to world coordinates using ``origin``.

    Args:
        grid: 3D array of signed distances, any float dtype. Internally cast
            to ``np.float32`` (skimage's preferred input) and to a writable
            copy (JAX arrays are read-only).
        origin: World-space ``(x, y, z)`` of grid index ``(0, 0, 0)``.
        voxel_size: Grid spacing in mm. Used as the marching-cubes
            ``spacing`` so vertex coordinates land in correct world units.
        iso_level: Isosurface level (``sdf = 0`` by default).
        config: :class:`MarchingCubesConfig`. Defaults to snapping samples
            that sit exactly on the isosurface, see
            :func:`snap_iso_degeneracies`.

    Raises:
        MeshExtractionError: On marching-cubes failure or degenerate output.
    """
    _require_export_deps()
    if config is None:
        config = MarchingCubesConfig()

    # JAX arrays are read-only; force a writable numpy copy.
    grid_np = np.array(grid, dtype=np.float32, copy=True)

    if config.snap_iso_degeneracies:
        n_snapped = snap_iso_degeneracies(grid_np, iso_level)
        if n_snapped:
            logger.info(
                "_meshing.extract_mesh: %d sample(s) sat exactly on the "
                "isosurface and were snapped to the interior side (|f - iso| <= epsilon)",
                n_snapped,
            )

    try:
        verts, faces, _normals, _values = marching_cubes(
            grid_np,
            level=iso_level,
            spacing=(voxel_size, voxel_size, voxel_size),
        )
    except (ValueError, RuntimeError) as e:
        raise MeshExtractionError(f"Marching cubes failed: {e}") from e

    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    verts[:, 0] += ox
    verts[:, 1] += oy
    verts[:, 2] += oz

    mesh = MeshData(vertices=verts, faces=faces)
    validate_mesh(mesh, context="after marching cubes")
    return mesh


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_mesh(
    mesh: MeshData,
    *,
    watertight_required: bool = False,
    context: str = "",
) -> None:
    """Fail-loud mesh validation.

    Raises :class:`MeshExtractionError` on real problems; logs warnings on
    suggestive-but-not-fatal conditions.
    """
    _require_export_deps()
    prefix = f"[{context}] " if context else ""

    if len(mesh.vertices) == 0:
        raise MeshExtractionError(f"{prefix}Mesh has zero vertices")
    if len(mesh.faces) == 0:
        raise MeshExtractionError(f"{prefix}Mesh has zero faces")
    if not np.all(np.isfinite(mesh.vertices)):
        n_bad = int(np.sum(~np.isfinite(mesh.vertices)))
        raise MeshExtractionError(f"{prefix}Mesh has {n_bad} non-finite vertex coordinates")

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    n_components = len(tm.split(only_watertight=False))
    if n_components > 100:
        logger.warning(
            "%sMesh has %d connected components (may indicate noise)",
            prefix,
            n_components,
        )

    if not tm.is_watertight:
        # Say *how* it is broken. A hole and two sheets welded along an edge
        # both read as "not watertight" and have different causes: the first
        # is usually a degenerate crossing (see snap_iso_degeneracies), the
        # second a cell carrying two surface sheets.
        census = edge_census(np.asarray(mesh.faces))
        detail = (
            f"{census.boundary} boundary edge(s), "
            f"{census.non_manifold} edge(s) used by 3+ faces, "
            f"of {census.total}"
        )
        if watertight_required:
            raise MeshExtractionError(
                f"{prefix}Mesh is not watertight (required by caller): {detail}"
            )
        logger.warning(
            "%sMesh is not watertight (%s): consider enabling fill_holes in cleanup",
            prefix,
            detail,
        )
    else:
        logger.info(
            "%sMesh is watertight (%d verts, %d faces)",
            prefix,
            len(mesh.vertices),
            len(mesh.faces),
        )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_mesh(mesh: MeshData, config: MeshCleanupConfig) -> MeshData:
    """Apply conservative mesh cleanup according to ``config``.

    Default operations (fix winding, fix normals, remove duplicates) are
    safe: they don't change topology. ``fill_holes`` and
    ``keep_only_largest_component`` ARE topology-changing and are opt-in.
    """
    _require_export_deps()

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    v_before, f_before = len(tm.vertices), len(tm.faces)

    if config.fix_winding:
        trimesh.repair.fix_winding(tm)

    if config.fix_normals:
        trimesh.repair.fix_normals(tm)

    if config.remove_duplicates:
        tm.merge_vertices()
        mask = tm.nondegenerate_faces()
        if not mask.all():
            tm.update_faces(mask)
        tm.remove_unreferenced_vertices()

    if config.fill_holes:
        trimesh.repair.fill_holes(tm)

    if config.keep_only_largest_component:
        components = tm.split(only_watertight=False)
        if len(components) > 1:
            components_sorted = sorted(
                components,
                key=lambda c: len(c.faces),
                reverse=True,
            )
            kept = components_sorted[0]
            dropped_faces = sum(len(c.faces) for c in components_sorted[1:])
            logger.info(
                "_meshing.cleanup_mesh: keep_only_largest_component dropped "
                "%d islands (%d faces total): kept main body with %d faces",
                len(components_sorted) - 1,
                dropped_faces,
                len(kept.faces),
            )
            tm = kept

    v_after, f_after = len(tm.vertices), len(tm.faces)
    logger.info(
        "_meshing.cleanup_mesh: %d->%d verts, %d->%d faces "
        "(fix_normals=%s, fix_winding=%s, remove_duplicates=%s, "
        "fill_holes=%s, keep_only_largest=%s)",
        v_before,
        v_after,
        f_before,
        f_after,
        config.fix_normals,
        config.fix_winding,
        config.remove_duplicates,
        config.fill_holes,
        config.keep_only_largest_component,
    )

    return MeshData(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces))


def write_mesh(mesh: MeshData, path: str | os.PathLike[str], fmt: str) -> None:
    """Write ``mesh`` to ``path`` via trimesh. ``fmt`` in {stl, obj, ply}.

    STL is the canonical manufacturing export format. Creates parent dirs.
    """
    _require_export_deps()
    from pathlib import Path

    fmt = fmt.lower()
    if fmt not in ("stl", "obj", "ply"):
        raise ValueError(f"Unsupported mesh format {fmt!r}. Use one of ['obj', 'ply', 'stl']")
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    tm.export(str(path), file_type=fmt)
    logger.info(
        "_meshing.write_mesh: %s (%d verts, %d faces)",
        path,
        len(mesh.vertices),
        len(mesh.faces),
    )


__all__ = [
    "MeshExtractionError",
    "cleanup_mesh",
    "extract_mesh",
    "snap_iso_degeneracies",
    "validate_mesh",
    "write_mesh",
]
