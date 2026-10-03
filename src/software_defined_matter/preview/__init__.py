"""Fast, display-only PyVista preview of a ``.sdm`` part.

Per-material zero-isosurfaces, contoured directly on a PyVista
``ImageData`` (vtk flying-edges): no scikit-image, no trimesh, no mesh
validation. The preview mesh is **not** manufacturable; for that use
:func:`software_defined_matter.export.export_part`.

The grid is the *same* ``np.arange`` exact-voxel-spacing sampler the export
path uses (:mod:`software_defined_matter.grid_sampling.grid`), so the preview is
honest about geometry: it differs only in resolution, polygoniser, and the
fact that it degrades gracefully instead of failing loud. Design rationale:
``docs/adr/0002-preview-and-export-modules.md`` and
``docs/adr/0008-pyvista-preview-extra.md``.

``pyvista`` is an optional dependency (the ``[preview]`` extra). It is
imported lazily so importing this module never pulls in VTK; calling
:func:`preview_part` without it raises a clear install hint.

Example
-------
::

    from software_defined_matter.preview import preview_part
    preview_part("part.sdm").show()
"""

from __future__ import annotations

import logging
import os

# Stop JAX pre-allocating ~90% of GPU VRAM on first use (best effort: only
# effective if JAX has not been imported yet in this process).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np

from software_defined_matter.grid_sampling import (
    BBox3,
    BBoxResolutionError,
    bind_sdf,
    chunk_for_tree,
    eval_chunked,
    make_grid,
    resolve_bbox,
)

if TYPE_CHECKING:
    import os

    import pyvista as pv

    from software_defined_matter.model import Part

    PartOrPath = Part | str | os.PathLike[str]

logger = logging.getLogger(__name__)

try:  # optional [preview] extra
    import pyvista as _pv

    _HAVE_PREVIEW_DEPS = True
    _IMPORT_ERR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only without extra
    _pv = None  # type: ignore[assignment]
    _HAVE_PREVIEW_DEPS = False
    _IMPORT_ERR = exc


def _require_preview_deps() -> None:
    """Raise a friendly error if the optional ``[preview]`` extra is missing."""
    if not _HAVE_PREVIEW_DEPS:
        raise ImportError(
            "The preview path requires the optional 'preview' extra "
            "(pyvista). Install with:\n"
            "    pip install 'emergent-matter-sdm-core[preview]'\n"
            "or:\n"
            "    uv pip install 'pyvista>=0.48.0'"
        ) from _IMPORT_ERR


# Distinct, colour-blind-friendly-ish palette for material bands.
_PALETTE = (
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
)


def _load_part(part_or_path: PartOrPath) -> Part:
    from software_defined_matter.model import Part

    if isinstance(part_or_path, Part):
        return part_or_path
    from software_defined_matter.io import load_part

    return load_part(part_or_path)


def _voxel_for(
    bbox: BBox3,
    voxel_size: float | None,
    resolution: int | None,
    target_points: int,
) -> float:
    """Pick the grid spacing for one material's bbox.

    Priority: explicit ``voxel_size`` → ``resolution`` samples along the
    largest axis → point budget (≈ ``target_points`` along the largest
    axis). The budget keeps open time predictable regardless of part scale.
    """
    if voxel_size is not None:
        return float(voxel_size)
    max_extent = float(np.max(bbox.size))
    n = resolution if resolution is not None else target_points
    return max_extent / max(int(n), 4)


def preview_part(
    part_or_path: PartOrPath,
    *,
    voxel_size: float | None = None,
    resolution: int | None = None,
    smooth: bool = True,
    target_points: int = 80,
    cpu: bool = False,
) -> pv.Plotter:
    """Build a PyVista plotter showing each material's zero-isosurface.

    Returns the (un-shown) ``pv.Plotter`` so callers can tweak it; call
    ``.show()`` to display. Materials whose sampling domain can't be
    resolved, or whose isosurface is empty, are skipped with a warning
    (graceful degradation, a partial preview beats no preview).

    Args:
        part_or_path: A :class:`~software_defined_matter.model.Part` or a
            ``.sdm`` path.
        voxel_size: Grid spacing in mm. Overrides ``resolution`` / the point
            budget.
        resolution: Samples along the largest axis (per material) if
            ``voxel_size`` is not given.
        smooth: Apply a light Laplacian smooth to each isosurface
            (cosmetic).
        target_points: Point budget along the largest axis when neither
            ``voxel_size`` nor ``resolution`` is given. Default 80.
        cpu: Force JAX onto CPU (avoids GPU VRAM; slower).

    Raises:
        ValueError: If the part has no materials, or every material was
            skipped.
    """
    _require_preview_deps()
    import jax

    pv = _pv
    part = _load_part(part_or_path)
    if not part.materials:
        raise ValueError(f"Part {part.name!r} has no materials; nothing to preview.")

    ctx = jax.default_device(jax.devices("cpu")[0]) if cpu else nullcontext()

    plotter = pv.Plotter()
    legend: list[list[str]] = []
    n_shown = 0

    with ctx:
        for mi, region in enumerate(part.materials):
            try:
                bbox = resolve_bbox(region.sdf_tree, part)
            except BBoxResolutionError as exc:
                logger.warning(
                    "preview: skipping material %r (id=%s): %s",
                    region.name,
                    region.material_id,
                    exc,
                )
                continue

            v = _voxel_for(bbox, voxel_size, resolution, target_points)
            padded = bbox.padded(v)
            points, shape = make_grid(padded, v)
            sdf_fn = bind_sdf(region.sdf_tree, part)
            grid = eval_chunked(sdf_fn, points, chunk_for_tree(region.sdf_tree)).reshape(shape)

            image = pv.ImageData(dimensions=shape)
            image.origin = tuple(float(x) for x in padded.min_pt)
            image.spacing = (v, v, v)
            image.point_data["sdf"] = grid.ravel(order="F")
            surf = image.contour(isosurfaces=[0.0], scalars="sdf")

            if surf.n_points == 0:
                logger.warning(
                    "preview: empty isosurface for material %r (id=%s); "
                    "skipping. Raise resolution or widen the bbox.",
                    region.name,
                    region.material_id,
                )
                continue

            if smooth:
                surf = surf.smooth(n_iter=25, relaxation_factor=0.1)

            color = _PALETTE[mi % len(_PALETTE)]
            plotter.add_mesh(surf, color=color, smooth_shading=True)
            legend.append([region.name, color])
            n_shown += 1

    if n_shown == 0:
        raise ValueError(
            "Every material was skipped (unresolved bbox or empty "
            "isosurface). Raise resolution, widen metadata['bbox'], or "
            "check the SDF trees."
        )

    # pyvista's @_deprecate_positional_args wrapper confuses mypy into
    # treating these bound methods as unbound (it asks for an explicit
    # `self` / a BasePlotter first argument) -- an upstream stub quirk, not
    # a real call-signature mismatch. Narrowly ignored rather than pulling
    # pyvista out of the blanket ignore_missing_imports override.
    plotter.add_legend(legend, bcolor="white", loc="upper right")  # type: ignore[arg-type]
    plotter.add_axes()  # type: ignore[call-arg]
    plotter.show_grid()  # type: ignore[call-arg]
    plotter.camera_position = "iso"
    return plotter


__all__ = ["preview_part"]
