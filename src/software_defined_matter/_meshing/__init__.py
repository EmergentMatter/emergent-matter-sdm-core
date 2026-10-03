"""Private mesh extraction used by ``export``.

Owns mesh types and, in modules not re-exported here, the three polygonisers
(``.mesh`` for marching cubes, ``.dual_contour``, ``.tetra``), validation,
cleanup, and optional quadric decimation. ``.topology`` holds the edge-census
and Euler-characteristic predicates all three share.

The polygoniser modules and ``.decimate`` are intentionally **not**
re-exported: they lazily depend on the optional ``[export]`` extra
(scikit-image, trimesh, fast-simplification, scipy). Import them from their
own modules on the export path so callers that only need the config
dataclasses or :class:`MeshData` never touch those deps.

Grid evaluation of compiled SDFs lives in
:mod:`software_defined_matter.grid_sampling`.
"""

from __future__ import annotations

from software_defined_matter._meshing.types import (
    DecimateConfig,
    DualContourConfig,
    MarchingCubesConfig,
    MeshCleanupConfig,
    MeshData,
    TetraConfig,
)

__all__ = [
    "DecimateConfig",
    "DualContourConfig",
    "MarchingCubesConfig",
    "MeshCleanupConfig",
    "MeshData",
    "TetraConfig",
]
