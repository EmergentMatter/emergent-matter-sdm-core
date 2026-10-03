"""Codec + validation for ``raster_field`` params (DR-0003).

A ``raster_field`` primitive carries a box-aligned grid of signed distance
samples inline in the ``.sdm`` as base64-encoded raw binary — the format's
first binary payload. Inline JSON floats were ruled out at grid scale
(~2.2 M samples for the OW-II ROM carve is 40+ MB of JSON text, and
jsonschema's recursive ``oneOf`` is superlinear); sidecar files were ruled
out because ``.sdm`` stays a single portable file.

Wire shape (all TOPOLOGY-class — literals only, never ``$ref``):

    { "type": "primitive", "kind": "raster_field",
      "params": {
        "origin":  [x, y, z],          # min corner, sample [0,0,0] sits here
        "spacing": h | [hx, hy, hz],   # voxel pitch (scalar broadcasts)
        "dims":    [nx, ny, nz],       # samples per axis (>= 1 each)
        "encoding": "f32le",           # sample dtype, little-endian
        "data":    "<base64>",         # nx*ny*nz samples, x-fastest:
                                       #   flat[i + nx*(j + ny*k)]
        "step_scale": 0.577,           # optional marcher hint (see shapes)
        "provenance": {...}            # optional; see sdf/bake.py
      } }

This module is deliberately numpy-only (no JAX) so the emitter and io layers
can use it without touching the JAX runtime.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
from typing import Any

import numpy as np

__all__ = [
    "require_literal_raster_params",
    "normalize_spacing",
    "raster_dims",
    "raster_origin",
    "raster_bbox",
    "decode_raster_values",
    "encode_raster_data",
    "raster_content_hash",
]

# Sample dtypes by wire encoding. ``f16le`` is the reserved size upgrade —
# decoding widens to float32 either way (GLSL tables and JAX default match).
_ENCODINGS: dict[str, np.dtype] = {
    "f32le": np.dtype("<f4"),
    "f16le": np.dtype("<f2"),
}

_TOPOLOGY_KEYS = ("origin", "spacing", "dims", "data")


def require_literal_raster_params(params: dict[str, Any]) -> None:
    """Reject ``$ref`` / expression leaves in topology-class slots.

    Grid geometry and data select how many samples exist and where — the
    same ruling as swept-op pose counts: topology, not value. A live
    parameter can never drive them; changing one means re-emitting the tree.
    """
    for key in _TOPOLOGY_KEYS:
        raw = params.get(key)
        if isinstance(raw, dict):
            raise ValueError(
                f"raster_field's {key!r} must be a literal, not a $ref or "
                f"expression (got a dict). Grid geometry and data are "
                "topology: they select how many samples the node holds, so "
                "they cannot be driven by a live parameter — re-emit the tree."
            )


def normalize_spacing(params: dict[str, Any]) -> tuple[float, float, float]:
    """Voxel pitch as a per-axis 3-tuple (scalar spacing broadcasts)."""
    raw = params.get("spacing")
    if raw is None:
        raise ValueError("raster_field requires a 'spacing' param")
    if isinstance(raw, (int, float)):
        h = float(raw)
        spacing = (h, h, h)
    else:
        values = tuple(float(v) for v in raw)
        if len(values) != 3:
            raise ValueError(f"raster_field 'spacing' must be a scalar or a 3-list, got {raw!r}")
        spacing = (values[0], values[1], values[2])
    if any(not math.isfinite(h) or h <= 0.0 for h in spacing):
        raise ValueError(f"raster_field 'spacing' must be positive, got {raw!r}")
    return spacing


def raster_dims(params: dict[str, Any]) -> tuple[int, int, int]:
    raw = params.get("dims")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"raster_field 'dims' must be [nx, ny, nz], got {raw!r}")
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in raw):
        raise ValueError(f"raster_field dims must contain integers, got {raw!r}")
    dims = tuple(int(v) for v in raw)
    if any(n < 1 for n in dims):
        raise ValueError(f"raster_field 'dims' must all be >= 1, got {raw!r}")
    return dims  # type: ignore[return-value]


def raster_origin(params: dict[str, Any]) -> tuple[float, float, float]:
    raw = params.get("origin")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"raster_field 'origin' must be [x, y, z], got {raw!r}")
    if not all(math.isfinite(float(v)) for v in raw):
        raise ValueError(f"raster_field origin must be finite, got {raw!r}")
    return tuple(float(v) for v in raw)  # type: ignore[return-value]


def raster_bbox(
    params: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Exact domain box ``origin .. origin + spacing*(dims-1)``.

    This is the sample domain. Geometry bounds may extend beyond it when
    boundary samples are negative; sdf.bbox accounts for that continuation.
    """
    lo = raster_origin(params)
    spacing = normalize_spacing(params)
    dims = raster_dims(params)
    hi = tuple(lo[a] + spacing[a] * (dims[a] - 1) for a in range(3))
    return lo, (hi[0], hi[1], hi[2])


def decode_raster_values(params: dict[str, Any]) -> np.ndarray:
    """Decode ``data`` to a float32 array of shape ``(nz, ny, nx)``.

    The wire order is x-fastest (``flat[i + nx*(j + ny*k)]``), so a C-order
    reshape lands sample ``(i, j, k)`` at ``values[k, j, i]``.
    """
    dims = raster_dims(params)
    encoding = params.get("encoding", "f32le")
    dtype = _ENCODINGS.get(encoding)
    if dtype is None:
        raise ValueError(
            f"raster_field 'encoding' must be one of {sorted(_ENCODINGS)}, got {encoding!r}"
        )
    raw = params.get("data")
    if not isinstance(raw, str):
        raise ValueError("raster_field requires base64 string 'data'")
    try:
        buf = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"raster_field 'data' is not valid base64: {exc}") from exc
    n_expected = dims[0] * dims[1] * dims[2]
    n_got = len(buf) // dtype.itemsize
    if n_got != n_expected or len(buf) % dtype.itemsize:
        raise ValueError(
            f"raster_field 'data' holds {n_got} {encoding} samples "
            f"({len(buf)} bytes) but dims {list(dims)} needs {n_expected}"
        )
    flat = np.frombuffer(buf, dtype=dtype).astype(np.float32)
    if not np.all(np.isfinite(flat)):
        raise ValueError("raster_field 'data' contains non-finite samples")
    nx, ny, nz = dims
    return flat.reshape(nz, ny, nx)


def encode_raster_data(values: np.ndarray, encoding: str = "f32le") -> str:
    """Encode a ``(nz, ny, nx)`` array to the wire base64 (x-fastest)."""
    dtype = _ENCODINGS.get(encoding)
    if dtype is None:
        raise ValueError(
            f"raster_field 'encoding' must be one of {sorted(_ENCODINGS)}, got {encoding!r}"
        )
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if arr.ndim != 3:
        raise ValueError(f"expected (nz, ny, nx) samples, got shape {arr.shape}")
    return base64.b64encode(arr.astype(dtype).tobytes()).decode("ascii")


def raster_content_hash(values: np.ndarray, origin: Any, spacing: Any, dims: Any) -> str:
    """Stable identity for a bake: geometry header + raw f32 samples."""
    h = hashlib.sha256()
    header = (
        f"origin={[float(v) for v in origin]};"
        f"spacing={[float(v) for v in spacing]};"
        f"dims={[int(v) for v in dims]}"
    )
    h.update(header.encode("ascii"))
    h.update(np.ascontiguousarray(np.asarray(values, dtype=np.float32)).tobytes())
    return "sha256:" + h.hexdigest()
