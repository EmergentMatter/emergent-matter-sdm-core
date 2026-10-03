"""Grid generation + chunked SDF evaluation.

#: Grids can be huge (millions of points), so we want to evaluate the SDF
#: in chunks to avoid memory issues. Instead, JAX calls an
#: evaluation slice i.e. evaluates blocks of `chunk_size` points.

**grid samples are at exact ``voxel_size`` spacing
via ``np.arange``, not ``np.linspace``.** Why?
``np.linspace`` silently produces samples at ``(max - min) / (N - 1)``
spacing, slightly off from ``voxel_size``. Marching cubes is then told a
``spacing`` that  causes systematic radial drift on canonical primitives
(e.g. a radius-5 sphere at voxel 0.5 produced vertices as far as 0.6 mm
off the true surface). ``np.arange`` fixes this by construction.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from software_defined_matter.grid_sampling.types import BBox3, SDFFunc

logger = logging.getLogger(__name__)

#: When the SDF is a compiled callable, we can't call tree_point_width
#: to estimate and allocate memory to the chunks based on nodes and params.
#: (For non-compiled SDFs, we use tree_point_width to count rows of
#: array-valued params like polygon vertices, sweep paths, etc.)
#:
#: A row count prices memory as if a tree cost O(1) per point. That holds for
#: analytic primitives and fails for polygon profiles, where ``polygon_2d``
#: materialises an ``(n_points x n_vertices)`` intermediate, so peak memory is
#: ``chunk x total_polygon_vertices``.
#:
#: A caller holding the TREE should ask :func:`chunk_for_tree` instead.
#: For compiled SDFs, we use a default chunk size of 65,536 points.
DEFAULT_CHUNK_SIZE = 65_536

#: Intermediate-memory budget per slice. :func:`chunk_for_tree` sizes
#: the slice so ``chunk x tree_point_width x 4`` lands near this, which makes
#: peak memory roughly constant across parts rather than proportional to their
#: polygon content.
CHUNK_BUDGET_BYTES = 1_000_000_000


#: tree_point_width sums array-parameter rows across the tree. That is
#: accurate for loft-style nodes that materialise all sections together, but
#: conservative for sequential CSG folds (e.g. a union of 1000 boxes), where
#: real peak memory is O(1) per point.

#: The floor: Lower bound on rows per slice. Very wide trees may still exceed
#: CHUNK_BUDGET_BYTES; the floor only prevents an explosion of tiny slices.
MIN_CHUNK_SIZE = 1_024
#: The ceiling: Upper bound on rows per slice. 1M was the default before
#: chunk sizing became tree-aware; narrow trees still get it.
MAX_CHUNK_SIZE = 1_000_000


def tree_point_width(tree: Any) -> int:
    """Estimate of temporary floats allocated per evaluated point.
    (e.g. sphere: 1 float/point, 50-vertex polygon: 50 floats/point)

    Sums, over all nodes, the length of each node's longest array-valued
    parameter (e.g. polygon vertices). Analytic nodes count as 1. Used by
    :func:`chunk_for_tree` to size evaluation slices.

    Args:
        tree: An SDF tree in wire form (nested dicts and lists).

    Returns:
        Floats of intermediate per point, at least 1.
    """
    total = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            # Only SDF nodes carry a per-point cost. A nested `params` dict
            # belongs to its node rather than being another one;
            if "type" in node:
                total += _node_width(node)
            for val in node.values():
                if isinstance(val, dict | list):
                    stack.append(val)
        elif isinstance(node, list):
            stack.extend(node)
    return max(total, 1)


def _node_width(node: dict) -> int:
    """Per-point width of one node: its widest array-valued parameter, min 1.
    Never descends into ``child`` / ``children``, which are separate nodes and
    are charged in their own right.

    Args:
        node: One wire-form SDF node.

    Returns:
        The node's own per-point width, at least 1.
    """
    widest = 1
    pending = [v for k, v in node.items() if k not in ("child", "children")]
    while pending:
        val = pending.pop()
        widest = max(widest, _array_len(val))
        if isinstance(val, dict):
            pending.extend(v for k, v in val.items() if k not in ("child", "children"))
    return widest


def _array_len(val: Any) -> int:
    """Rows of an array-valued parameter, else 0.

    A polygon's ``vertices`` is ``(N, 2)``, and N is the multiplier. A bare
    ``[x, y, z]`` translate is 3 scalars and must not read as width 3, so only
    sequences whose elements are themselves short sequences count.

    Args:
        val: Any parameter value off a node.

    Returns:
        The number of rows, or 0 when the value is not an array of points.
    """
    if isinstance(val, list | tuple):
        if val and all(isinstance(e, list | tuple) and len(e) <= 4 for e in val):
            return len(val)
        return 0
    shape = getattr(val, "shape", None)
    if shape is not None and len(shape) == 2 and shape[1] <= 4:
        return int(shape[0])
    return 0


def chunk_for_tree(tree: Any) -> int:
    """Slice size that keeps ``tree``'s intermediates near CHUNK_BUDGET_BYTES.

    A fixed chunk size is safe for one part but can exhaust memory on a
    part with more polygon content. Sizing from the tree keeps peak memory
    roughly constant across different parts.

    Args:
        tree: An SDF tree in wire form.

    Returns:
        Rows per slice, clamped to [MIN_CHUNK_SIZE, MAX_CHUNK_SIZE].
    """
    chunk = int(CHUNK_BUDGET_BYTES // (4 * max(tree_point_width(tree), 1)))
    return max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, chunk))


def make_grid(
    bbox: BBox3,
    voxel_size: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Build a flat ``(N, 3)`` array of grid points and the grid shape.

    Returns a ``np.ndarray`` (not JAX) so the caller can chunk the points
    and feed each chunk to a JIT-compiled SDF without triggering
    recompilation per chunk.

    Samples are at **exact** ``voxel_size`` spacing (``np.arange``, not
    ``np.linspace``) so the spacing later handed to marching cubes is the
    spacing the grid actually has.
    """
    half_step = voxel_size * 0.5
    x = np.arange(float(bbox.min_pt[0]), float(bbox.max_pt[0]) + half_step, voxel_size)
    y = np.arange(float(bbox.min_pt[1]), float(bbox.max_pt[1]) + half_step, voxel_size)
    z = np.arange(float(bbox.min_pt[2]), float(bbox.max_pt[2]) + half_step, voxel_size)
    nx, ny, nz = len(x), len(y), len(z)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    return points, (nx, ny, nz)


def eval_chunked(
    sdf_fn: SDFFunc,
    points: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Evaluate ``sdf_fn`` on ``points`` in slices of ``chunk_size`` rows.

    JIT-compiles ``sdf_fn`` once on the first chunk and reuses the compiled
    artefact for the rest (JAX caches by abstract shape; all chunks except
    possibly the last share a shape). Returns a flat numpy array of length
    ``len(points)``.

    ``chunk_size`` trades peak memory against slice count. See
    :data:`DEFAULT_CHUNK_SIZE` for why the default is not larger, and
    :func:`chunk_for_tree` for the tree-aware size.
    """
    n = len(points)
    if n == 0:
        # Empty input: probe the SDF's output dtype on a single dummy point
        # so the empty result has a sensible dtype.
        probe = np.asarray(sdf_fn(jnp.zeros((1, 3))))
        return np.empty((0,), dtype=probe.dtype)

    jit_fn = jax.jit(sdf_fn)
    out_chunks: list[np.ndarray] = []
    for i in range(0, n, chunk_size):
        chunk = jnp.asarray(points[i : i + chunk_size])
        out_chunks.append(np.asarray(jit_fn(chunk)))
    return np.concatenate(out_chunks)


def eval_sdf_grid(
    sdf_fn: SDFFunc,
    bbox: BBox3,
    voxel_size: float,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a canonical ``(p: (..., 3)) -> (...,)`` SDF on a 3D grid.

    Pads the bbox by 2 voxels (so marching cubes has a clean clipping
    boundary) before generating the grid. Returns the SDF values reshaped
    into a 3D array of shape ``grid_shape``, plus the world-space origin of
    grid index ``(0, 0, 0)``.
    """
    padded_bbox = bbox.padded(voxel_size * 2)
    nx, ny, nz = padded_bbox.grid_dims(voxel_size)
    total = nx * ny * nz
    logger.info(
        "grid_sampling.eval_sdf_grid: %dx%dx%d = %s pts, voxel=%.6f mm",
        nx,
        ny,
        nz,
        f"{total:,}",
        voxel_size,
    )

    points, grid_shape = make_grid(padded_bbox, voxel_size)
    t0 = time.perf_counter()
    flat = eval_chunked(sdf_fn, points, chunk_size)
    elapsed = time.perf_counter() - t0
    logger.info(
        "grid_sampling.eval_sdf_grid: %.2fs (%.0f pts/s)",
        elapsed,
        total / max(elapsed, 1e-9),
    )

    if not np.all(np.isfinite(flat)):
        n_bad = int(np.sum(~np.isfinite(flat)))
        logger.warning("grid_sampling.eval_sdf_grid: %d non-finite values", n_bad)

    grid = flat.reshape(grid_shape)
    origin = np.array([float(padded_bbox.min_pt[i]) for i in range(3)])
    return grid, origin


__all__ = [
    "CHUNK_BUDGET_BYTES",
    "DEFAULT_CHUNK_SIZE",
    "MAX_CHUNK_SIZE",
    "MIN_CHUNK_SIZE",
    "chunk_for_tree",
    "eval_chunked",
    "eval_sdf_grid",
    "make_grid",
    "tree_point_width",
]
