"""Public grid sampler for compiled SDF trees.

Evaluate a part on a voxel grid without pulling in mesh-export dependencies
(scikit-image, trimesh). Preview, numeric bbox tightening, and downstream
volume writers all need this; marching cubes does not belong here.

The grid uses exact ``voxel_size`` spacing via ``np.arange`` (not
``np.linspace``) so consumers that later hand the spacing to marching cubes
get the spacing the samples actually have.

This is a different module from :mod:`software_defined_matter.sample`, which
draws Monte Carlo vectors from ``Param`` priors.
"""

from __future__ import annotations

from software_defined_matter.grid_sampling.bind import (
    BBoxResolutionError,
    bind_sdf,
    material_bbox,
    resolve_bbox,
)
from software_defined_matter.grid_sampling.grid import (
    CHUNK_BUDGET_BYTES,
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    chunk_for_tree,
    eval_chunked,
    eval_sdf_grid,
    make_grid,
    tree_point_width,
)
from software_defined_matter.grid_sampling.types import (
    BBox3,
    SDFFunc,
)

__all__ = [
    "CHUNK_BUDGET_BYTES",
    "DEFAULT_CHUNK_SIZE",
    "MAX_CHUNK_SIZE",
    "MIN_CHUNK_SIZE",
    "BBox3",
    "BBoxResolutionError",
    "SDFFunc",
    "bind_sdf",
    "chunk_for_tree",
    "eval_chunked",
    "eval_sdf_grid",
    "make_grid",
    "material_bbox",
    "resolve_bbox",
    "tree_point_width",
]
