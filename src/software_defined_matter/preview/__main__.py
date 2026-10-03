"""CLI: ``python -m software_defined_matter.preview file.sdm [opts]``.

Sets the JAX VRAM env var *before* anything imports JAX, then delegates to
:func:`software_defined_matter.preview.preview_part`.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m software_defined_matter.preview",
        description="View each material SDF as its own isosurface in PyVista.",
    )
    ap.add_argument("sdm", help="path to a .sdm file")
    ap.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="grid spacing in mm (overrides --resolution and the point budget)",
    )
    ap.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="samples along the largest axis per material (if --voxel-size unset)",
    )
    ap.add_argument(
        "--target-points",
        type=int,
        default=80,
        help="point budget along the largest axis (default: 80)",
    )
    ap.add_argument(
        "--no-smooth",
        action="store_true",
        help="disable the cosmetic Laplacian smoothing pass",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="run JAX on CPU only (avoids GPU VRAM; slower)",
    )
    args = ap.parse_args()

    from software_defined_matter.preview import preview_part

    plotter = preview_part(
        args.sdm,
        voxel_size=args.voxel_size,
        resolution=args.resolution,
        smooth=not args.no_smooth,
        target_points=args.target_points,
        cpu=args.cpu,
    )
    plotter.show(title=f"{args.sdm} · per-material SDF")


if __name__ == "__main__":
    main()
