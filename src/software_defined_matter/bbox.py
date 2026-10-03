"""Numeric ("shrink-wrap") bounding-box tightening.

Counterpart to :mod:`software_defined_matter.sdf.bbox` (analytic interval
inference). The analytic inferrer is exact and free but the boxes it
returns are conservative (e.g. a twisted shape returns the inscribing
square of its swept disk regardless of the bend rate). This module
tightens the box by actually evaluating the SDF.

The analytic inferrer also **fails loud** on the one case it refuses to
guess: param-dependent rotation (a ``rotate_x|y|z|matrix`` whose ``angle``
or ``R`` is a ``$ref``). For those parts, an explicit ``metadata['bbox']``
is required.

This module tightens the box the other way: sample the compiled SDF on a
loose seed box and take the axis-aligned bounding box of the solid region
(``sdf <= iso``), padded. It works on **any** SDF because it only evaluates
it: no analytic support needed. Cost is one coarse probe pass, trivially
amortized against the much finer export grid it shrinks (on the bundled
example: ~7x fewer points at 0.125 mm).

Seeding
-------
The loose box must actually contain the part. By default it is resolved
from the part exactly the way the export path resolves its sampling domain
(:func:`software_defined_matter.grid_sampling.resolve_bbox`): explicit
``metadata['bbox']`` → analytic ``infer_sdf_bbox(mode='values')`` →
``metadata['bbox_half_size']``. Pass ``loose=`` to override.

This is the *numeric* sibling of the analytic
:func:`software_defined_matter.sdf.bbox.infer_sdf_bbox`; see
``docs/adr/0002-preview-and-export-modules.md``
for where each fits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from software_defined_matter.grid_sampling import (
    BBox3,
    bind_sdf,
    chunk_for_tree,
    eval_chunked,
    make_grid,
    resolve_bbox,
)

if TYPE_CHECKING:
    from software_defined_matter.model import Part, SDFTree


class EmptyShrinkwrapError(ValueError):
    """Raised when no solid region (``sdf <= iso``) lies in the loose box.

    Almost always means the seed box does not contain the part, or
    ``probe_voxel`` is too coarse to land a sample inside a thin feature.
    """


def shrinkwrap_bbox(
    tree: SDFTree,
    part: Part,
    *,
    loose: BBox3 | tuple | None = None,
    probe_voxel: float = 0.5,
    pad: float = 1.0,
    iso: float = 0.0,
) -> BBox3:
    """Numeric tight AABB of ``tree`` (sampled on a loose seed box).

    Args:
        tree: SDF expression tree (e.g. ``region.sdf_tree`` or
            ``part.computed_envelope()``).
        part: The part the tree belongs to (supplies params + metadata).
        loose: Seed box that must contain the part. ``BBox3`` or a
            ``((xlo,ylo,zlo),(xhi,yhi,zhi))`` pair. ``None`` (default)
            resolves it via :func:`software_defined_matter.grid_sampling.resolve_bbox`.
        probe_voxel: Spacing of the coarse probe grid, mm. Cheap; the result
            is only as tight as this is fine, so the surface-band ``pad``
            covers the slack.
        pad: Isotropic margin added to the solid region's extents, mm. Must
            exceed any ``displace`` amplitude and any CSG smoothing radius
            so the soft surface is not clipped.
        iso: Solid threshold. ``0.0`` is the SDF surface.

    Raises:
        EmptyShrinkwrapError: If no sample satisfies ``sdf <= iso`` in the
            loose box.
    """
    if probe_voxel <= 0:
        raise ValueError(f"probe_voxel must be positive, got {probe_voxel}")
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")

    if loose is None:
        loose = resolve_bbox(tree, part)
    elif not isinstance(loose, BBox3):
        lo, hi = loose
        loose = BBox3(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))

    points, _shape = make_grid(loose, probe_voxel)
    d = np.asarray(eval_chunked(bind_sdf(tree, part), points, chunk_for_tree(tree)))

    mask = d <= iso
    if not mask.any():
        raise EmptyShrinkwrapError(
            f"No solid region (sdf <= {iso}) found in the loose box "
            f"{loose.min_pt.tolist()}..{loose.max_pt.tolist()} at "
            f"probe_voxel={probe_voxel}. The seed box may not contain the "
            f"part, or probe_voxel is too coarse for a thin feature."
        )

    inside = points[mask]
    return BBox3(
        min_pt=inside.min(axis=0) - pad,
        max_pt=inside.max(axis=0) + pad,
    )


def tighten_part_bbox(
    part: Part,
    *,
    persist: bool = True,
    probe_voxel: float = 0.5,
    pad: float = 1.0,
    iso: float = 0.0,
) -> BBox3:
    """Shrink-wrap the whole-part envelope and (by default) write it back.

    Convenience over :func:`shrinkwrap_bbox` for the common "tighten this
    part before exporting" flow. The tightened box is written to
    ``part.metadata['bbox']`` when ``persist=True`` so the export / preview
    sampler picks it up (``resolve_bbox`` prefers explicit metadata).

    Note: with ``persist=True`` the new (tight) box becomes the seed for any
    later tighten call on the same part: re-running is near-idempotent but
    keep ``pad`` sufficient.

    Raises:
        ValueError: If the part has no materials (no envelope to wrap).
    """
    envelope = part.computed_envelope()
    if envelope is None:
        raise ValueError(f"Part {part.name!r} has no materials; nothing to tighten.")
    box = shrinkwrap_bbox(envelope, part, probe_voxel=probe_voxel, pad=pad, iso=iso)
    if persist:
        part.metadata["bbox"] = [box.min_pt.tolist(), box.max_pt.tolist()]
    return box


__all__ = [
    "BBox3",
    "EmptyShrinkwrapError",
    "shrinkwrap_bbox",
    "tighten_part_bbox",
]
