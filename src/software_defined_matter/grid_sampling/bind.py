"""Bridge sdm-core's :class:`Part` model to the grid sampler.

Two concerns:

1. **Closure shape.** :func:`software_defined_matter.sdf.compile.make_sdf_closure`
   returns a two-arg ``(points, free_vec=None) -> distances`` closure that
   late-binds the free-parameter vector (so the same compiled tree can drive
   an optimiser). The grid evaluator wants a one-arg
   :data:`~software_defined_matter.grid_sampling.types.SDFFunc`. :func:`bind_sdf`
   captures the free vector once so ``jax.jit`` caches by a stable closure
   identity.

2. **Sampling domain.** :func:`resolve_bbox` produces a finite
   :class:`BBox3` for a material via the fail-loud fallback chain documented
   in ``docs/adr/0002-preview-and-export-modules.md``:

       metadata ``bbox``  →  per-material ``infer_sdf_bbox(mode="values")``
       →  metadata ``bbox_half_size``  →  raise :class:`BBoxResolutionError`

   Explicit metadata wins first because authors set it deliberately (the
   inferred box is conservative and may want overriding). The remaining
   case where inference cannot succeed is param-dependent rotation
   (``rotate_x|y|z|matrix`` with an ``angle`` / ``R`` that is a ``$ref``),
   which the inferrer refuses rather than guessing. The export path lets
   :class:`BBoxResolutionError` propagate (fail loud); the preview path
   catches it per-material and skips with a warning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from software_defined_matter.grid_sampling.types import BBox3, SDFFunc
from software_defined_matter.sdf.bbox import BBoxInferenceError, infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure

if TYPE_CHECKING:
    from software_defined_matter.model import MaterialRegion, Part, SDFTree


class BBoxResolutionError(ValueError):
    """Raised when no finite sampling domain can be resolved for a tree."""


def bind_sdf(tree: SDFTree, part: Part) -> SDFFunc:
    """Compile ``tree`` and bind ``part``'s free-parameter vector.

    Returns a one-arg ``f(p: (..., 3)) -> (...,)`` closure. The free vector
    is captured once at bind time so the closure is JIT-stable.
    """
    closure = make_sdf_closure(tree, part)
    free_vec = jnp.asarray(closure.binding.initial_free_vector())

    def sdf(p: jnp.ndarray) -> jnp.ndarray:
        return closure(p, free_vec)

    return sdf


def resolve_bbox(tree: SDFTree, part: Part) -> BBox3:
    """Resolve a finite :class:`BBox3` for ``tree`` (fail-loud fallback chain).

    Order: explicit metadata ``bbox`` → ``infer_sdf_bbox(mode="values")``
    → metadata ``bbox_half_size`` → raise :class:`BBoxResolutionError`.
    """
    md_bbox = part.metadata.get("bbox")
    if md_bbox is not None:
        lo, hi = md_bbox
        return BBox3(
            min_pt=np.asarray(lo, dtype=float),
            max_pt=np.asarray(hi, dtype=float),
        )

    infer_err: Exception | None = None
    try:
        lo, hi = infer_sdf_bbox(tree, part, mode="values")
        return BBox3(
            min_pt=np.asarray(lo, dtype=float),
            max_pt=np.asarray(hi, dtype=float),
        )
    except BBoxInferenceError as exc:
        infer_err = exc

    half = part.metadata.get("bbox_half_size")
    if half is not None:
        h = abs(float(half))
        return BBox3(
            min_pt=np.array([-h, -h, -h], dtype=float),
            max_pt=np.array([h, h, h], dtype=float),
        )

    raise BBoxResolutionError(
        "Cannot resolve a finite sampling domain: no metadata 'bbox', "
        f"per-material inference failed ({infer_err}), and no metadata "
        "'bbox_half_size'. Add an explicit metadata['bbox'] = [[xlo,ylo,zlo],"
        "[xhi,yhi,zhi]] (typically needed for param-dependent rotation, "
        "which the analytic inferrer refuses) or set finite Param.bounds."
    ) from infer_err


def material_bbox(region: MaterialRegion, part: Part) -> BBox3:
    """:func:`resolve_bbox` for a :class:`MaterialRegion`'s SDF tree."""
    return resolve_bbox(region.sdf_tree, part)


__all__ = [
    "BBoxResolutionError",
    "bind_sdf",
    "material_bbox",
    "resolve_bbox",
]
