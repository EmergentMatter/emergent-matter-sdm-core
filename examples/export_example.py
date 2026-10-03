"""Export the bundled ``.sdm`` example to per-material STL files.

Thin shim over :func:`software_defined_matter.export.export_part`: each
:class:`MaterialRegion` is sampled on the shared ``np.arange`` voxel grid,
polygonised with scikit-image marching cubes, cleaned with trimesh, and
written as one ``.stl`` per material. ``--voxel-size`` (mm) is the fidelity
knob: smaller = closer to the SDF, larger grids, slower. ``--auto-bbox``
numerically shrink-wraps the (deliberately conservative) ``metadata['bbox']``
to the actual part first, strongly recommended at the fine default voxel,
where the bundled part is otherwise ~66M voxels.

Requires the optional ``[export]`` extra (scikit-image + trimesh)::

    uv pip install -e '.[export]'

Output filenames carry an ISO 8601 stamp by default
(``hollow_cylinder_with_hinge_PLA_2026-05-19T1430.stl``) per the org's
filename-stamp convention -- this prevents stale-file confusion when
overwriting. Use ``--date-only`` for a date-only ``...2026-05-19`` stamp,
or ``--no-stamp`` to opt out (not recommended for deliverable artifacts).

``--decimate-error-mm X`` runs manifold-safe quadric decimation after
marching cubes, within an ``X`` mm surface-deviation budget (watertight,
same body count).

Run with::

    uv run python examples/export_example.py --auto-bbox
    uv run python examples/export_example.py --voxel-size 0.25 --fmt stl
    uv run python examples/export_example.py --auto-bbox --decimate-error-mm 0.05
    uv run python examples/export_example.py --auto-bbox --date-only
    uv run python examples/export_example.py --no-resolve-overlaps --no-stamp
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before anything imports JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> None:
    here = Path(__file__).parent
    default_sdm = here / "hollow_cylinder_with_hinge.sdm"
    default_out = here / "exports"

    ap = argparse.ArgumentParser(
        description="Export each material of a .sdm part to its own mesh file."
    )
    ap.add_argument(
        "--sdm",
        type=Path,
        default=default_sdm,
        help=f"path to .sdm file (default: {default_sdm.name})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"output directory (default: {default_out.relative_to(here.parent)})",
    )
    ap.add_argument(
        "--voxel-size",
        type=float,
        default=0.125,
        help="grid spacing in mm: primary fidelity knob (default: 0.125; "
        "4x finer than the original 0.5, ~64x more voxels and slower)",
    )
    ap.add_argument(
        "--fmt",
        choices=("stl", "obj", "ply"),
        default="stl",
        help="output mesh format (default: stl)",
    )
    ap.add_argument(
        "--decimate-error-mm",
        type=float,
        default=None,
        help="after marching cubes, quadric-decimate within this mm "
        "surface-deviation budget (watertight, same body count)",
    )
    ap.add_argument(
        "--no-resolve-overlaps",
        action="store_true",
        help="mesh each material raw instead of carving later materials out "
        "(faster; correct only if the .sdm authors disjoint regions)",
    )
    ap.add_argument(
        "--auto-bbox",
        action="store_true",
        help="numerically shrink-wrap metadata['bbox'] to the part before "
        "exporting (recommended for parts with a loose/conservative bbox, "
        "the bundled example drops from ~66M to ~9M voxels at 0.125 mm)",
    )
    ap.add_argument(
        "--bbox-pad",
        type=float,
        default=1.0,
        help="mm margin kept around the shrink-wrapped box; must exceed any "
        "displace amplitude + CSG smoothing radius (default: 1.0)",
    )
    ap.add_argument(
        "--date-only",
        action="store_true",
        help="use a date-only YYYY-MM-DD stamp instead of the default "
        "YYYY-MM-DDTHHMM datetime stamp",
    )
    ap.add_argument(
        "--no-stamp",
        action="store_true",
        help="disable the ISO 8601 filename stamp (not recommended for deliverable artifacts)",
    )
    args = ap.parse_args()

    # Org convention: always append an ISO 8601 stamp to
    # generated artifacts. Datetime by default; --date-only drops the time.
    if args.no_stamp:
        stamp: bool | str = False
    elif args.date_only:
        stamp = "date"
    else:
        stamp = "datetime"

    from software_defined_matter.export import DecimateConfig, export_part

    # Pass a path by default (export_part loads it). For --auto-bbox we need
    # the Part in hand to tighten its metadata['bbox'] first.
    part_or_path = args.sdm
    if args.auto_bbox:
        from software_defined_matter import load
        from software_defined_matter.bbox import tighten_part_bbox

        part = load(args.sdm)
        before = part.metadata.get("bbox")
        box = tighten_part_bbox(part, pad=args.bbox_pad)  # persists into metadata
        print(
            f"auto-bbox: {before} -> "
            f"[{[round(v, 2) for v in box.min_pt.tolist()]}, "
            f"{[round(v, 2) for v in box.max_pt.tolist()]}]"
        )
        part_or_path = part

    decimate = None
    if args.decimate_error_mm is not None:
        decimate = DecimateConfig(simplify_error_mm=args.decimate_error_mm)

    paths = export_part(
        part_or_path,
        args.out,
        voxel_size=args.voxel_size,
        fmt=args.fmt,
        decimate=decimate,
        resolve_overlaps=not args.no_resolve_overlaps,
        stamp=stamp,
    )

    print(
        f"Exported {len(paths)} material mesh(es) from {args.sdm.name} "
        f"(voxel={args.voxel_size} mm, auto_bbox={args.auto_bbox}, "
        f"decimate_error_mm={args.decimate_error_mm}, "
        f"resolve_overlaps={not args.no_resolve_overlaps}):"
    )
    for p in paths:
        size_kb = p.stat().st_size / 1024.0
        print(f"  {p}  ({size_kb:.1f} KiB)")


if __name__ == "__main__":
    main()
