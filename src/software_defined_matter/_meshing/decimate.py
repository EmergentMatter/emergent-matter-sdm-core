"""Quadric decimation after a polygoniser, with the topology kept intact.

Marching cubes tessellates uniformly: a flat slab, or a *curved-path* flat
such as a helical wire ribbon (flat in section, curved along its path), gets
far more triangles than its curvature needs. Quadric edge collapse removes
them. It works on non-exact, swept, and embossed fields where a
gradient-based mesher cannot.

:func:`decimate_mesh` searches for the largest reduction that keeps
every one of these invariants:

* watertight, with the same number of bodies as the input;
* the same Euler characteristic, so a torus tunnel cannot be sealed shut and
  a through-hole cannot be closed (body count alone does not see either);
* no pinched vertices, the shape quadric collapse across a thin handle
  produces, where every edge still has two faces but two sheets meet at a
  point;
* a surface-deviation limit in mm, measured in both directions.

The two directions catch different failures. Distance from the decimated
surface to the original catches a surface that moved. Distance from the
original to the decimated one catches a feature that is gone.

``trimesh``, ``fast_simplification`` and ``scipy`` are optional ``[export]``
deps, imported lazily (same pattern as
:mod:`software_defined_matter._meshing.mesh`).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, NamedTuple

import numpy as np

from software_defined_matter._meshing.topology import (
    edge_census,
    euler_characteristic,
    n_bodies,
    n_pinched_vertices,
)
from software_defined_matter._meshing.types import DecimateConfig, MeshData
from software_defined_matter.grid_sampling import DEFAULT_CHUNK_SIZE, eval_chunked
from software_defined_matter.grid_sampling.types import SDFFunc

logger = logging.getLogger(__name__)

try:  # optional [export] extra
    import fast_simplification as _fs
    import trimesh as _trimesh

    _HAVE_DECIMATE_DEPS = True
    _IMPORT_ERR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only without extra
    _fs = None
    _trimesh = None  # type: ignore[assignment]
    _HAVE_DECIMATE_DEPS = False
    _IMPORT_ERR = exc

try:  # rtree backs trimesh's triangle index; without it proximity queries fail
    import rtree as _rtree  # noqa: F401  (imported for availability only)

    _HAVE_RTREE = True
    _RTREE_ERR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only without extra
    _HAVE_RTREE = False
    _RTREE_ERR = exc

#: Keep at least 3% of the triangles. Past this a mesh has too few degrees of
#: freedom left to hold its topology, and every candidate gets rejected anyway.
_MAX_REDUCTION = 0.97


def _require_proximity() -> None:
    """The deviation-limit path needs an indexed triangle query;
    ``target_reduction`` does not, so the check sits here rather than at
    import."""
    if not _HAVE_RTREE:
        raise ImportError(
            "A surface-deviation limit (simplify_error_mm) needs exact "
            "point-to-mesh distances, which trimesh computes with rtree. "
            "Install with:\n"
            "    uv sync --extra export\n"
            "or drop the limit and pass DecimateConfig(target_reduction=...)."
        ) from _RTREE_ERR


def _require_deps() -> None:
    if not _HAVE_DECIMATE_DEPS:
        raise ImportError(
            "Mesh decimation requires the optional 'export' extra "
            "(trimesh + fast-simplification). Install with:\n"
            "    uv sync --extra export\n"
            "or:\n"
            "    uv pip install 'trimesh>=4.0' 'fast-simplification>=0.1'"
        ) from _IMPORT_ERR


# ---------------------------------------------------------------------------
# Topology of a candidate
# ---------------------------------------------------------------------------


class _Topology(NamedTuple):
    """What a decimated mesh has to keep from its input."""

    n_bodies: int
    euler: int
    watertight: bool
    pinched: int


def _topology_of(tm: Any) -> _Topology:
    faces = np.asarray(tm.faces)
    return _Topology(
        n_bodies=n_bodies(faces),
        euler=euler_characteristic(faces),
        watertight=bool(tm.is_watertight),
        pinched=n_pinched_vertices(faces),
    )


def _topology_verdict(candidate: _Topology, reference: _Topology, config: DecimateConfig) -> str:
    """``""`` when the candidate is acceptable, else why it is not."""
    if not candidate.watertight:
        return "not watertight"
    if candidate.n_bodies != reference.n_bodies:
        return f"body count {reference.n_bodies} -> {candidate.n_bodies}"
    if candidate.pinched:
        return f"{candidate.pinched} pinched vertex/vertices"
    if config.preserve_euler and candidate.euler != reference.euler:
        return f"Euler characteristic {reference.euler} -> {candidate.euler}"
    return ""


# ---------------------------------------------------------------------------
# Deviation, measured the same way every run
# ---------------------------------------------------------------------------


def _surface_points(tm: Any, max_points: int) -> np.ndarray:
    """A fixed set of points covering a mesh's surface.

    Vertices, face centroids, and edge midpoints: no randomness, so two runs
    on the same mesh compare the same points and report the same deviation.
    Strided down to ``max_points`` when the mesh is large, which keeps the
    coverage uniform because the three groups are interleaved by the stride.
    """
    vertices = np.asarray(tm.vertices, dtype=np.float64)
    faces = np.asarray(tm.faces)
    centroids = vertices[faces].mean(axis=1)
    edges = np.unique(
        np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            axis=1,
        ),
        axis=0,
    )
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])

    points = np.concatenate([vertices, centroids, midpoints])
    if len(points) > max_points:
        stride = int(np.ceil(len(points) / max_points))
        points = points[::stride]
    return points


def _distance_to_mesh(points: np.ndarray, tm: Any) -> np.ndarray:
    """Distance from each point to the nearest triangle of ``tm``.

    ``trimesh.proximity`` indexes the triangles with ``rtree`` and returns the
    exact point-to-triangle distance. The obvious dependency-free substitute,
    a KD-tree over triangle centroids, is not sound here: decimation is what
    produces one huge triangle across a flat face, and a query point near that
    triangle's edge has hundreds of small triangles whose centroids are nearer
    than its own. Measured on the cylinder-and-slab fixture, taking the 16
    nearest centroids over-reported a 0.0003 mm deviation as 0.2 mm.
    """
    _require_proximity()
    return _trimesh.proximity.closest_point(tm, points)[1]


def _hausdorff_mm(a_tm: Any, b_tm: Any, max_points: int) -> float:
    """Symmetric surface deviation between two meshes, in mm.

    Both directions, because they catch different failures: a moved surface
    shows up going from the decimated mesh to the original, and a feature that
    was deleted outright only shows up going the other way.
    """
    a_pts = _surface_points(a_tm, max_points)
    b_pts = _surface_points(b_tm, max_points)
    forward = float(_distance_to_mesh(a_pts, b_tm).max()) if len(a_pts) else 0.0
    backward = float(_distance_to_mesh(b_pts, a_tm).max()) if len(b_pts) else 0.0
    return max(forward, backward)


def _field_deviation_mm(tm: Any, sdf: SDFFunc, chunk_size: int, max_points: int) -> float:
    """Max ``|sdf|`` over the mesh's surface points: distance to the true part.

    Reported, not enforced. The limit applies to the distance from the
    undecimated mesh, and
    this says how far that mesh itself sits from the field it came from, which
    is the voxel size's error, not decimation's.
    """
    points = _surface_points(tm, max_points)
    if not len(points):
        return 0.0
    d = np.abs(np.asarray(eval_chunked(sdf, points, chunk_size)))
    return float(d.max())


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------


def _repair_mesh(vertices: np.ndarray, faces: np.ndarray) -> Any | None:
    """Weld, drop degenerate/duplicate faces, fill small holes, fix normals.

    Does **not** keep-largest-component: that would destroy a legitimately
    multi-body material such as a gyroid infill. Body count is enforced by the
    caller. Returns a ``trimesh.Trimesh`` or ``None`` if nothing survives.
    """
    tm = _trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tm.merge_vertices()
    mask = tm.nondegenerate_faces()
    if not mask.all():
        tm.update_faces(mask)
    tm.update_faces(tm.unique_faces())
    tm.remove_unreferenced_vertices()
    with contextlib.suppress(Exception):  # pragma: no cover - repair is best-effort
        _trimesh.repair.fill_holes(tm)
    if len(tm.faces) == 0:
        return None
    _trimesh.repair.fix_normals(tm)
    return tm


def _simplify(vertices: np.ndarray, faces: np.ndarray, reduction: float, agg: float) -> Any | None:
    """One quadric-decimation pass; returns a repaired Trimesh or ``None``."""
    try:
        v2, f2 = _fs.simplify(
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int32),
            target_reduction=float(reduction),
            agg=float(agg),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("decimate: fast_simplification failed at r=%.3f (%s)", reduction, exc)
        return None
    return _repair_mesh(v2, f2)


def _as_mesh_data(tm: Any) -> MeshData:
    return MeshData(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces))


def _warn_if_input_is_broken(input_tm: Any, topology: _Topology) -> None:
    """Say so up front when the mesh handed to us is already not closed.

    Decimation cannot repair this, and the repair inside ``_repair_mesh`` will
    quietly fill the holes in the *output*, so the result would look better
    than the input while describing different geometry.
    """
    if topology.watertight and not topology.pinched:
        return
    census = edge_census(np.asarray(input_tm.faces))
    logger.warning(
        "decimate: input mesh is already not a closed manifold "
        "(%d boundary edge(s), %d edge(s) used by 3+ faces, %d pinched vertex/vertices "
        "of %d edges). Decimation preserves topology, it does not repair it.",
        census.boundary,
        census.non_manifold,
        topology.pinched,
        census.total,
    )


def decimate_mesh(
    mesh: MeshData,
    sdf: SDFFunc,
    config: DecimateConfig,
    *,
    chunk_size: int | None = None,
) -> MeshData:
    """Quadric-decimate ``mesh``, keeping its topology and a deviation limit.

    With neither ``simplify_error_mm`` nor ``target_reduction`` set, returns
    ``mesh`` unchanged. If no acceptable reduction is found, the undecimated
    ``mesh`` comes back with a warning naming what blocked it.

    Args:
        mesh: The polygonised, cleaned mesh.
        sdf: The (overlap-resolved) field the mesh was extracted from. Used
            only to log how far the surface sits from the field; the limit is
            measured against ``mesh`` itself.
        config: :class:`DecimateConfig`.
        chunk_size: Maximum rows per chunk in the SDF evaluation. ``None``
            (default) falls back to ``DEFAULT_CHUNK_SIZE``: ``sdf`` is an
            opaque callable here, so tree-aware sizing is the caller's job
            (``export_part`` passes a tree-sized value down).
    """
    _require_deps()
    chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return mesh
    if config.simplify_error_mm is None and config.target_reduction is None:
        return mesh

    input_tm = _trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    reference = _topology_of(input_tm)
    _warn_if_input_is_broken(input_tm, reference)
    max_points = max(int(config.deviation_samples), 100)

    if config.simplify_error_mm is None:
        return _decimate_to_reduction(mesh, vertices, faces, reference, config)
    return _decimate_to_limit(
        mesh, input_tm, vertices, faces, reference, config, sdf, chunk_size, max_points
    )


def _decimate_to_reduction(
    mesh: MeshData,
    vertices: np.ndarray,
    faces: np.ndarray,
    reference: _Topology,
    config: DecimateConfig,
) -> MeshData:
    """Honour ``target_reduction``, halving it until the topology survives."""
    assert config.target_reduction is not None
    reduction = float(np.clip(config.target_reduction, 0.0, _MAX_REDUCTION))
    blocked = "no candidate survived simplification"
    for _ in range(max(int(config.max_passes), 1)):
        tm = _simplify(vertices, faces, reduction, config.aggressiveness)
        if tm is not None:
            verdict = _topology_verdict(_topology_of(tm), reference, config)
            if not verdict:
                logger.info(
                    "decimate: %d -> %d faces (reduction=%.2f, topology preserved)",
                    len(faces),
                    len(tm.faces),
                    reduction,
                )
                return _as_mesh_data(tm)
            blocked = verdict
        reduction *= 0.5  # too aggressive to keep the topology; ease off
    logger.warning(
        "decimate: no reduction preserved the topology (%s); keeping the undecimated mesh",
        blocked,
    )
    return mesh


def _decimate_to_limit(
    mesh: MeshData,
    input_tm: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    reference: _Topology,
    config: DecimateConfig,
    sdf: SDFFunc,
    chunk_size: int,
    max_points: int,
) -> MeshData:
    """Bisect for the largest reduction inside ``simplify_error_mm``.

    Every pass measures on the full ``deviation_samples`` point set, rather
    than searching cheaply and checking the winner at the end. Two attempts at
    the cheaper arrangement both misbehaved: verifying each accepted candidate
    cost more than measuring densely throughout, and easing off after a failed
    final check made the result non-monotone, a 0.1 mm limit reducing less
    than a 0.05 mm one, which is the complaint this rewrite exists to fix.

    The search reports which constraint stopped it. That matters in practice:
    on parts where topology binds first, every limit from 0.02 mm to 0.2 mm
    returns the identical mesh, and without the reason that reads as the knob
    being ignored.
    """
    assert config.simplify_error_mm is not None  # the caller dispatched on this
    limit = float(config.simplify_error_mm)
    lo, hi = 0.0, _MAX_REDUCTION
    best: Any | None = None
    best_deviation = 0.0
    blocked = "nothing was attempted"

    for _ in range(max(int(config.max_passes), 1)):
        if hi - lo < float(config.reduction_tol):
            break
        mid = 0.5 * (lo + hi)
        tm = _simplify(vertices, faces, mid, config.aggressiveness)
        if tm is None:
            blocked = "simplification produced nothing"
            hi = mid
            continue
        verdict = _topology_verdict(_topology_of(tm), reference, config)
        if verdict:
            blocked = verdict
            hi = mid
            continue
        deviation = _hausdorff_mm(tm, input_tm, max_points)
        if deviation > limit:
            blocked = f"deviation {deviation:.4g} mm over the {limit:.4g} mm limit"
            hi = mid
            continue
        best, best_deviation = tm, deviation
        lo = mid

    if best is None:
        logger.warning(
            "decimate: no reduction fits %.4g mm (%s); keeping the undecimated mesh",
            limit,
            blocked,
        )
        return mesh

    logger.info(
        "decimate: %d -> %d faces, deviation %.4g mm of the %.4g mm limit "
        "(search stopped on: %s; surface sits %.4g mm from the field)",
        len(faces),
        len(best.faces),
        best_deviation,
        limit,
        blocked,
        _field_deviation_mm(best, sdf, chunk_size, max_points),
    )
    return _as_mesh_data(best)


__all__ = ["decimate_mesh"]
