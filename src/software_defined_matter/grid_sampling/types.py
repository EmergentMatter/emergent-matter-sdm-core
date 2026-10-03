"""Grid-domain types: bounding boxes and bound SDF callables.

These are the contracts between :func:`~software_defined_matter.grid_sampling.bind.bind_sdf`
/ :func:`~software_defined_matter.grid_sampling.bind.resolve_bbox` and the grid
evaluator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

#: Canonical SDF closure: ``f(p: (..., 3)) -> (...,)``. Any free parameters
#: were bound at construction (see :mod:`software_defined_matter.grid_sampling.bind`).
SDFFunc = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class BBox3:
    """Axis-aligned bounding box in 3D.

    Frozen dataclass; ``min_pt`` / ``max_pt`` are read-only ``np.ndarray``
    of shape ``(3,)``. Methods return new instances; no in-place mutation.
    """

    min_pt: np.ndarray  # shape (3,), float
    max_pt: np.ndarray  # shape (3,), float

    @property
    def size(self) -> np.ndarray:
        """Edge lengths along each axis: ``(dx, dy, dz)``."""
        return self.max_pt - self.min_pt

    @property
    def center(self) -> np.ndarray:
        """Centre point ``((min + max) / 2)``."""
        return (self.min_pt + self.max_pt) * 0.5

    def padded(self, margin: float) -> BBox3:
        """Return a new bbox expanded by ``margin`` in every direction."""
        return BBox3(
            min_pt=self.min_pt - margin,
            max_pt=self.max_pt + margin,
        )

    @staticmethod
    def enclosing(*boxes: BBox3) -> BBox3:
        """Return the smallest bbox enclosing all given boxes.

        Raises ``ValueError`` if called with no boxes.
        """
        if not boxes:
            raise ValueError("BBox3.enclosing requires at least one box")
        mins = np.stack([b.min_pt for b in boxes])
        maxs = np.stack([b.max_pt for b in boxes])
        return BBox3(min_pt=np.min(mins, axis=0), max_pt=np.max(maxs, axis=0))

    def grid_dims(self, voxel_size: float) -> tuple[int, int, int]:
        """Grid dimensions ``(nx, ny, nz)`` for a given voxel size.

        Each dimension is at least 2 (a single voxel can't be
        marching-cubes'd).
        """
        s = self.size
        nx = max(int(np.ceil(s[0] / voxel_size)), 2)
        ny = max(int(np.ceil(s[1] / voxel_size)), 2)
        nz = max(int(np.ceil(s[2] / voxel_size)), 2)
        return (nx, ny, nz)


__all__ = [
    "BBox3",
    "SDFFunc",
]
