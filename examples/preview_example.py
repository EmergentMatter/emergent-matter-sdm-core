"""Interactive PyVista viewer for the bundled ``.sdm`` example.

Thin shim over :func:`software_defined_matter.preview.preview_part`: each
material's SDF zero-isosurface is shown as a separate mesh. The real logic
(and the shared ``np.arange`` exact-voxel sampler) lives in the
``software_defined_matter.preview`` module; this file just wires the
bundled example and a couple of CLI flags.

Run with::

    uv run python examples/visualize_example.py
    uv run python examples/visualize_example.py --resolution 96 --cpu

Equivalent module entry point::

    uv run python -m software_defined_matter.preview examples/hollow_cylinder_with_hinge.sdm
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before anything imports JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> None:
    default_sdm = Path(__file__).parent / "hollow_cylinder_with_hinge.sdm"
    ap = argparse.ArgumentParser(
        description="View each material SDF as its own isosurface in PyVista."
    )
    ap.add_argument(
        "--sdm",
        type=Path,
        default=default_sdm,
        help=f"path to .sdm file (default: {default_sdm.name})",
    )
    ap.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="samples along the largest axis per material (default: point budget)",
    )
    ap.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="grid spacing in mm (overrides --resolution)",
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
        cpu=args.cpu,
    )
    plotter.show(title=f"{args.sdm.name} · per-material SDF")


if __name__ == "__main__":
    main()
