"""Bake a reference JAX field into a ``raster_field`` node (DR-0003).

The analytic construction stays the source of truth; a bake is a derived
artifact stamped with provenance (content hash, generator note) so it can
be regenerated from the reference field.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from software_defined_matter.sdf.raster import (
    decode_raster_values,
    encode_raster_data,
    raster_content_hash,
)

__all__ = ["bake_raster_field"]

Vec3 = tuple[float, float, float]


def bake_raster_field(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    bbox: tuple[Sequence[float], Sequence[float]],
    voxel: float,
    *,
    pad_voxels: int = 2,
    chunk: int = 262144,
    encoding: str = "f32le",
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample ``fn`` on a grid over ``bbox`` and return a raster_field node.

    Parameters
    ----------
    fn : callable
        Reference field, ``(N, 3) -> (N,)``. Will be chunked; jit it for
        speed if it isn't already.
    bbox : (lo, hi)
        Region the field must be valid in. The grid domain extends past it
        by ``pad_voxels`` on every side so the out-of-domain fallback
        (clamped value + distance to box) engages only in the margin.
    voxel : float
        Sample pitch, all axes. Trilinear error is O(voxel^2 * curvature);
        pick it from the consumer's tolerance budget and verify with the
        consumer's own parity check (OW-II: the two-sided golden check).
    pad_voxels : int
        Margin samples added on every side of ``bbox``.
    chunk : int
        Points per evaluation batch (memory bound, not accuracy).
    encoding : str
        Wire dtype (``"f32le"`` default; ``"f16le"`` halves the payload).
    provenance : dict | None
        Free-form generator notes merged into ``params["provenance"]``
        (the content hash and bake settings are always stamped).

    Returns the complete node dict::

        {"type": "primitive", "kind": "raster_field", "params": {...}}
    """
    lo_in = np.asarray(bbox[0], dtype=np.float64)
    hi_in = np.asarray(bbox[1], dtype=np.float64)
    if lo_in.shape != (3,) or hi_in.shape != (3,):
        raise ValueError(f"bbox must be ((x,y,z), (x,y,z)), got {bbox!r}")
    if not np.all(hi_in > lo_in):
        raise ValueError(f"bbox must have positive extent, got {bbox!r}")
    if not np.isfinite(voxel) or voxel <= 0.0:
        raise ValueError(f"voxel must be positive, got {voxel!r}")

    if not isinstance(chunk, int) or chunk <= 0:
        raise ValueError("chunk must be a positive integer")
    if not isinstance(pad_voxels, int) or pad_voxels < 0:
        raise ValueError("pad_voxels must be a nonnegative integer")
    if not np.all(np.isfinite(lo_in)) or not np.all(np.isfinite(hi_in)):
        raise ValueError("bbox must be finite")
    lo = lo_in - pad_voxels * voxel
    span = (hi_in + pad_voxels * voxel) - lo
    dims = tuple(int(np.ceil(span[a] / voxel)) + 1 for a in range(3))
    nx, ny, nz = dims

    xs = lo[0] + voxel * np.arange(nx)
    ys = lo[1] + voxel * np.arange(ny)
    zs = lo[2] + voxel * np.arange(nz)
    # x-fastest wire order == C-order (nz, ny, nx) block, see sdf/raster.py.
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    out = np.empty(points.shape[0], dtype=np.float32)
    for start in range(0, points.shape[0], chunk):
        stop = min(start + chunk, points.shape[0])
        out[start:stop] = np.asarray(fn(jnp.asarray(points[start:stop])), dtype=np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("reference field returned non-finite samples — bake aborted")
    values = out.reshape(nz, ny, nx)
    data = encode_raster_data(values, encoding)
    # Measure the samples that consumers decode, including f16 quantization.
    values = decode_raster_values({"dims": list(dims), "encoding": encoding, "data": data})

    # Zero-set containment: the out-of-domain fallback (clamped value +
    # distance to box) is only safe when the surface stays inside the
    # margin. A touching zero-set is an authoring error worth hearing about.
    shell = np.concatenate(
        [
            values[0].ravel(),
            values[-1].ravel(),
            values[:, 0].ravel(),
            values[:, -1].ravel(),
            values[:, :, 0].ravel(),
            values[:, :, -1].ravel(),
        ]
    )
    shell_min = float(shell.min())
    if shell_min <= 0.0:
        warnings.warn(
            f"bake_raster_field: zero-set touches the domain boundary "
            f"(boundary min {shell_min:.4g} <= 0). Outside-domain queries "
            "will see a clipped surface — grow bbox or pad_voxels.",
            stacklevel=2,
        )

    # Realized Lipschitz bound of the trilinear interpolant: within a cell
    # the gradient components are bounded by the per-axis finite-difference
    # slopes of the corner samples, so the interpolant's constant is at most
    # max over cells of |(sx, sy, sz)|. A sphere tracer multiplies steps by
    # 1/L. Exact-SDF samples give sx,sy,sz <= 1 (L <= sqrt(3)); measured L
    # is usually far smaller, so storing it recovers march speed.
    slopes_sq = np.zeros_like(values)
    for axis, n in ((2, nx), (1, ny), (0, nz)):
        if n < 2:
            continue
        d = np.abs(np.diff(values, axis=axis)) / voxel
        # max of the slopes of the up-to-two cells adjacent to each sample
        pad_width = [(0, 0)] * 3
        pad_width[axis] = (1, 1)
        d_pad = np.pad(d, pad_width, constant_values=0.0)
        take_lo = [slice(None)] * 3
        take_hi = [slice(None)] * 3
        take_lo[axis] = slice(0, values.shape[axis])
        take_hi[axis] = slice(1, values.shape[axis] + 1)
        slopes_sq += np.maximum(d_pad[tuple(take_lo)], d_pad[tuple(take_hi)]) ** 2
    # Outside the box, the clamped field has tangential gradient and the
    # distance continuation adds an orthogonal unit gradient.
    lipschitz = float(np.sqrt(slopes_sq.max() + 1.0))
    step_scale = float(min(1.0, 1.0 / lipschitz))

    origin = tuple(float(v) for v in lo)
    prov: dict[str, Any] = dict(provenance or {})
    prov["content_hash"] = raster_content_hash(values, origin, (voxel,) * 3, dims)
    prov["bake"] = {
        "voxel": float(voxel),
        "pad_voxels": int(pad_voxels),
        "bbox": [list(map(float, lo_in)), list(map(float, hi_in))],
        "boundary_min": shell_min,
        "lipschitz": lipschitz,
    }

    return {
        "type": "primitive",
        "kind": "raster_field",
        "params": {
            "origin": list(origin),
            "spacing": [float(voxel)] * 3,
            "dims": [nx, ny, nz],
            "encoding": encoding,
            "data": data,
            "step_scale": step_scale,
            "provenance": prov,
        },
    }


__all__ = ["bake_raster_field"]
