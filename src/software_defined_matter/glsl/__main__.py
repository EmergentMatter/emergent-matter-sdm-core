"""CLI: ``python -m software_defined_matter.glsl file.sdm --out dir/``.

Loads an ``.sdm`` file, emits GLSL artifacts to ``--out`` (default: alongside
the .sdm), and exits non-zero with a clear message if the geometry is
unbounded and no ``metadata['bbox']`` override is set.

Files written
-------------
* ``<dir>/sdf_lib.glsl``: static helper library (verbatim copy of
  software_defined_matter/glsl/lib.glsl).
* ``<dir>/sdf_scene.glsl``: generated per-tree functions plus
  ``float sdf_scene(vec3 p)``.
* ``<dir>/meta.json``: uniform schema, control and component manifests, bbox,
  smooth-CSG flag, and raw polygon and sweep payload metadata.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from software_defined_matter.glsl.emit import GLSLEmission


def _write_artifacts(emission: GLSLEmission, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sdf_lib.glsl").write_text(emission.lib_source)
    (out_dir / "sdf_scene.glsl").write_text(emission.scene_source)
    meta = {
        "entry_point": emission.entry_point,
        "smooth_csg": emission.smooth_csg,
        "smooth_k": emission.smooth_k,
        "bbox": [list(emission.bbox[0]), list(emission.bbox[1])],
        "uniforms": [
            {
                "name": u.name,
                "glsl_type": u.glsl_type,
                "initial": u.initial,
                "bounds": list(u.bounds) if u.bounds else None,
                "unit": u.unit,
                "source_param": u.source_param,
            }
            for u in emission.uniforms
        ],
        # The control manifest, so a host builds its parameter panel from the
        # emission rather than re-deriving which params are scrubbable. It is a
        # superset of `uniforms`: every live control names its uniform, and the
        # re-emit and topology params have no uniform to appear under at all.
        "controls": emission.controls,
        # Component ids share an ordering contract with sdf_scene_comp. Keep
        # the manifest beside the shader artifact that computes those ids.
        "components": emission.components,
        # Sizes sdf_polygon_2d's array parameter. Also emitted as
        # `#define SDM_POLY_MAX_N` at the head of sdf_lib.glsl; carried here so
        # a host that reassembles the library does not parse it back out.
        "poly_max_n": emission.poly_max_n,
        # Outline LOD this emission was resampled to; null = full resolution.
        # A host reading unexpectedly light polygons needs to know whether the
        # author decimated them or this CLI did.
        "poly_lod": emission.poly_lod,
        # Raw table-mode polygon storage in fetch order. The writer does not
        # pack GPU rows or choose a texture format: a host combines these
        # floats with the reported row width. Empty values explicitly mean
        # inline mode, and give future raw table payloads the same JSON shape.
        "poly_table": emission.poly_table,
        "poly_tex_width": emission.poly_tex_width,
        # Raw table-mode sweep frames, RGBA texels flat in fetch order, on the
        # same terms as `poly_table`. `sweep_max_s` is the largest segment
        # count in the scene, inline or tabled, for a host sizing its budget.
        "sweep_table": emission.sweep_table,
        "sweep_tex_width": emission.sweep_tex_width,
        "sweep_max_s": emission.sweep_max_s,
        # Raw raster samples (x-fastest f32). Host packs u_sdm_grid.
        "grid_table": emission.grid_table,
        "grid_table_b64": base64.b64encode(
            struct.pack(f"<{len(emission.grid_table)}f", *emission.grid_table)
        ).decode("ascii"),
        "grid_tex_width": emission.grid_tex_width,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m software_defined_matter.glsl",
        description="Emit GLSL ray-march artifacts for an .sdm file.",
    )
    ap.add_argument("sdm", help="path to a .sdm file")
    ap.add_argument(
        "--out",
        default=None,
        help="output directory (default: <sdm_dir>/<sdm_stem>_glsl/)",
    )
    ap.add_argument(
        "--smooth-csg",
        action="store_true",
        help="emit smooth CSG variants regardless of part metadata",
    )
    ap.add_argument(
        "--smooth-k",
        type=float,
        default=0.25,
        help="smoothing radius for smooth CSG (default: 0.25)",
    )
    ap.add_argument(
        "--stdout",
        action="store_true",
        help="print the scene source to stdout instead of writing files",
    )
    ap.add_argument(
        "--poly-lod",
        type=int,
        default=None,
        help=(
            "resample literal outlines above N vertices down to N for the "
            "viewer (march cost is linear in vertex count per step; "
            "curvature-weighted so sharp features keep their density). "
            "Default: no resampling — opt-in, it changes emitted geometry"
        ),
    )
    args = ap.parse_args()

    # JAX is imported transitively through Part.computed_envelope; avoid
    # preallocating the whole GPU just for emission.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from software_defined_matter.glsl import emit_glsl
    from software_defined_matter.io import load_part
    from software_defined_matter.sdf.bbox import BBoxInferenceError

    sdm_path = Path(args.sdm)
    if not sdm_path.exists():
        print(f"error: {sdm_path} not found", file=sys.stderr)
        return 2

    part = load_part(sdm_path)

    try:
        emission = emit_glsl(
            part,
            smooth_csg=args.smooth_csg or None,
            smooth_k=args.smooth_k,
            poly_lod=args.poly_lod,
        )
    except BBoxInferenceError as exc:
        print(f"error: cannot derive a bounding box for {sdm_path.name}:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print(
            "  Add an explicit bbox via part.metadata['bbox'] = "
            "[[xlo, ylo, zlo], [xhi, yhi, zhi]] before emitting.",
            file=sys.stderr,
        )
        return 3

    if args.stdout:
        sys.stdout.write(emission.scene_source)
        return 0

    out_dir = Path(args.out) if args.out else sdm_path.parent / f"{sdm_path.stem}_glsl"
    _write_artifacts(emission, out_dir)
    print(f"wrote {out_dir}/{{sdf_lib.glsl, sdf_scene.glsl, meta.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
