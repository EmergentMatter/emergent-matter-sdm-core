"""Manufacturing-fidelity mesh export for ``.sdm`` parts.

One mesh file per :class:`~software_defined_matter.model.MaterialRegion`,
extracted with scikit-image marching cubes on the shared ``np.arange``
voxel grid and cleaned/written via trimesh. ``voxel_size`` (mm) is the
primary fidelity knob: it ties the mesh to the physical AM process, not an
abstract sample count. Pass ``decimate=DecimateConfig(...)`` for optional
manifold-safe quadric reduction after cleanup.

Requires the optional ``[export]`` extra (scikit-image + trimesh +
fast-simplification); a clear install hint is raised if it is missing.

Design rationale and tradeoffs: ``docs/adr/0002-preview-and-export-modules.md``.

Example
-------
::

    from software_defined_matter.export import DecimateConfig, export_part
    paths = export_part(
        "part.sdm", "out/", voxel_size=0.2, fmt="stl",
        decimate=DecimateConfig(simplify_error_mm=0.05),
    )
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from software_defined_matter._meshing import DecimateConfig, MeshCleanupConfig
from software_defined_matter._meshing.decimate import decimate_mesh
from software_defined_matter._meshing.mesh import (
    MeshExtractionError,
    cleanup_mesh,
    extract_mesh,
    validate_mesh,
    write_mesh,
)
from software_defined_matter.grid_sampling import (
    bind_sdf,
    chunk_for_tree,
    eval_chunked,
    make_grid,
    material_bbox,
)

if TYPE_CHECKING:
    import os

    from software_defined_matter.grid_sampling.types import SDFFunc
    from software_defined_matter.model import Part

    PartOrPath = Part | str | os.PathLike[str]

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _slug(name: str) -> str:
    """Filesystem-safe slug for a part / material name."""
    s = _SLUG_RE.sub("_", str(name).strip())
    return s.strip("_") or "unnamed"


def _stamp_suffix(stamp: bool | str) -> str:
    """ISO 8601 filename suffix, per the org convention of always appending
    an ISO 8601 date to a generated artifact's filename.

    - ``False`` / ``None`` / ``""`` → ``""`` (no stamp; deterministic).
    - ``True`` / ``"date"`` → ``_YYYY-MM-DD``.
    - ``"datetime"`` → ``_YYYY-MM-DDTHHMM`` (``T`` separator, no colons).
    - any other str → ``_<that string>`` verbatim (caller-formatted stamp).

    Computed once per :func:`export_part` call so every material mesh in a
    run shares the same timestamp.
    """
    if not stamp:
        return ""
    # The DTZ005 suppressions below are deliberate: these stamps are LOCAL
    # wall-clock on purpose. The playbook's date-stamp convention exists so a human reading a
    # directory listing can tell a fresh export from a stale one; a UTC stamp
    # would read hours off local time and cross the date boundary in the
    # evening, which is exactly the stale-file confusion it is meant to prevent.
    # These are filename fragments, never parsed back or compared across zones.
    if stamp is True or stamp == "date":
        return "_" + datetime.now().strftime("%Y-%m-%d")  # noqa: DTZ005
    if stamp == "datetime":
        return "_" + datetime.now().strftime("%Y-%m-%dT%H%M")  # noqa: DTZ005
    return "_" + str(stamp)


def _load_part(part_or_path: PartOrPath) -> Part:
    from software_defined_matter.model import Part

    if isinstance(part_or_path, Part):
        return part_or_path
    from software_defined_matter.io import load_part

    return load_part(part_or_path)


def _effective_closure(bound: list, i: int, resolve_overlaps: bool) -> SDFFunc:
    """Closure for the overlap-resolved field material ``i`` is meshed from.

    Mirrors the export loop's ``d = np.maximum(d, -d_later)``: the effective
    field is ``max(sdf_i, -sdf_{i+1}, -sdf_{i+2}, ...)`` when overlaps are
    resolved, else just ``sdf_i``. Decimation measures surface deviation
    against this same field. Built from the bound closures (so
    ``smooth_csg`` inside each tree is honoured); the inter-material
    reduction is always hard ``max``.
    """
    base = bound[i]
    if not resolve_overlaps or i >= len(bound) - 1:
        return base
    laters = bound[i + 1 :]

    def eff(p: jnp.ndarray) -> jnp.ndarray:
        return reduce(lambda d, fj: jnp.maximum(d, -fj(p)), laters, base(p))

    return eff


def export_part(
    part_or_path: PartOrPath,
    out_dir: str | os.PathLike[str],
    *,
    voxel_size: float,
    fmt: str = "stl",
    decimate: DecimateConfig | None = None,
    resolve_overlaps: bool = True,
    cleanup: MeshCleanupConfig | None = None,
    chunk_size: int | None = None,
    stamp: bool | str = "datetime",
) -> list[Path]:
    """Export each material of a part to its own mesh file.

    Args:
        part_or_path: A :class:`~software_defined_matter.model.Part` or a
            path to a ``.sdm`` file.
        out_dir: Directory to write meshes into (created if needed).
        voxel_size: Grid spacing in mm -- the primary fidelity knob. Choose
            it to match the target AM process resolution.
        fmt: ``"stl"`` (default, canonical manufacturing format), ``"obj"``
            or ``"ply"``.
        decimate: Optional :class:`~software_defined_matter._meshing.DecimateConfig`.
            When set, runs manifold-safe quadric decimation after cleanup:
            collapsing low-curvature regions (including curved-path flats)
            within a bounded mm surface deviation while keeping the input's
            body count and watertightness. ``None`` (default) → no
            decimation. Default export stays bit-identical to a call without
            this argument.
        resolve_overlaps: If ``True`` (default), material *i*'s mesh is its
            region minus the union of every later-listed material's region,
            so no voxel belongs to two materials -- honouring
            ``Part.materials``' "later overrides earlier" semantics.
            Subtraction is applied to the sampled grids before marching
            cubes. If ``False``, each material is meshed raw (faster;
            correct only when the ``.sdm`` authors disjoint regions).
        cleanup: Mesh cleanup configuration. Defaults to the conservative
            topology-preserving :class:`MeshCleanupConfig`.
        chunk_size: Rows per grid-evaluation slice. ``None`` (default) sizes
            each material's slice from its own SDF tree via
            :func:`~software_defined_matter.grid_sampling.chunk_for_tree`, so
            peak memory stays near ``CHUNK_BUDGET_BYTES`` regardless of
            polygon content. Pass an explicit value to override.
        stamp: ISO 8601 suffix appended before the extension (prevents
            stale-file confusion when overwriting). Default ``"datetime"``
            → ``..._YYYY-MM-DDTHHMM.stl`` (observes the convention out of
            the box). ``True`` / ``"date"`` → ``..._YYYY-MM-DD.stl``.
            ``False`` → no suffix. Any other string is used verbatim -- pass
            a fixed value (e.g. from a test fixture) when you need
            deterministic filenames. One timestamp is shared by all
            material meshes in a single call.

    Returns:
        One written file path per material, in ``Part.materials`` order.

    Raises:
        software_defined_matter.grid_sampling.BBoxResolutionError: If no
            finite sampling domain can be resolved for a material.
        software_defined_matter._meshing.mesh.MeshExtractionError: If a
            material produces no zero-crossing / degenerate mesh.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if cleanup is None:
        cleanup = MeshCleanupConfig()

    part = _load_part(part_or_path)
    if not part.materials:
        raise ValueError(f"Part {part.name!r} has no materials; nothing to export.")

    out_dir = Path(out_dir)
    part_slug = _slug(part.name)
    stamp_suffix = _stamp_suffix(stamp)  # one timestamp for the whole run
    written: list[Path] = []

    # Bind every material's SDF once; reused across the overlap subtraction.
    bound = [bind_sdf(m.sdf_tree, part) for m in part.materials]

    # Keep the caller's override separate: the loop rebinds `chunk_size`
    # per material.
    chunk_size_override = chunk_size

    for i, region in enumerate(part.materials):
        bbox = material_bbox(region, part)
        padded = bbox.padded(voxel_size * 2)
        points, shape = make_grid(padded, voxel_size)

        chunk_size = (
            chunk_size_override
            if chunk_size_override is not None
            else chunk_for_tree(region.sdf_tree)
        )
        d = eval_chunked(bound[i], points, chunk_size)

        if resolve_overlaps:
            # Iterated op_subtract(d, d_j) == subtract the union of later
            # regions: max(d, -d_a, -d_b, ...) = max(d, -min(d_a, d_b, ...)).
            for j in range(i + 1, len(part.materials)):
                chunk_size_j = (
                    chunk_size_override
                    if chunk_size_override is not None
                    else chunk_for_tree(part.materials[j].sdf_tree)
                )
                d_later = eval_chunked(bound[j], points, chunk_size_j)
                d = np.maximum(d, -d_later)

        grid = d.reshape(shape)
        origin = padded.min_pt

        ctx = f"material {region.name!r} (id={region.material_id})"
        try:
            mesh = extract_mesh(grid, origin, voxel_size)
        except MeshExtractionError as exc:
            raise MeshExtractionError(
                f"{ctx}: {exc}. The material may be empty within its bbox or "
                "fully occluded by a later material (resolve_overlaps=True)."
            ) from exc

        mesh = cleanup_mesh(mesh, cleanup)
        validate_mesh(mesh, context=f"post-cleanup {ctx}")

        if decimate is not None:
            eff = _effective_closure(bound, i, resolve_overlaps)
            # eff is the overlap-resolved field, so its true width is the max
            # over the involved trees; material i's slice approximates it.
            mesh = decimate_mesh(mesh, eff, decimate, chunk_size=chunk_size)
            validate_mesh(mesh, context=f"post-decimate {ctx}")

        path = out_dir / (f"{part_slug}_{_slug(region.name)}{stamp_suffix}.{fmt.lower()}")
        write_mesh(mesh, path, fmt)
        written.append(path)
        logger.info("export_part: wrote %s", path)

    return written


__all__ = [
    "DecimateConfig",
    "MeshCleanupConfig",
    "MeshExtractionError",
    "export_part",
]
