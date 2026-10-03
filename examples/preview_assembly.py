"""Inspect grounded and mated assembly instances and their port frames with PyVista.

Run ``uv run python examples/preview_assembly.py examples/assembly_example/assembly.sdm``.
This static viewer uses core placement at default design and joint inputs.
It does not solve mechanisms or render internal body deformation.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["preview_assembly", "main"]


def preview_assembly(
    path: str | Path, *, resolution: int = 64, off_screen: bool = False
) -> pv.Plotter:
    """Build a static assembly plotter at the root design's current values.

    Geometry uses the existing preview grid sampler. Colours identify occurrences;
    red, green and blue arrows show port x, y and z axes. Promoted ports retain
    their assembly addresses. The caller must close the returned plotter.

    Raises:
        ValueError: Ungrounded components, failed mates, empty geometry or invalid frames.
        NotImplementedError: A part requires body motion evaluation.
        ImportError: Optional preview dependencies are unavailable.
    """
    if resolution < 4:
        raise ValueError("resolution must be at least 4")
    try:
        import jax.numpy as jnp
        import numpy as np
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("Install preview dependencies with: uv sync") from exc

    from software_defined_matter import Assembly, compile_placement, load_bundle
    from software_defined_matter.grid_sampling import (
        bind_sdf,
        chunk_for_tree,
        eval_chunked,
        make_grid,
        resolve_bbox,
    )

    bundle = load_bundle(path)
    free_vec = jnp.asarray(bundle.binding().initial_free_vector())
    state = compile_placement(bundle).evaluate_checked()
    poses = {name: np.asarray(matrix) for name, matrix in state.instances.items()}
    frames = {name: np.asarray(matrix) for name, matrix in state.ports.items()}
    snapshots = {}
    for scope, document in bundle.definitions.items():
        binding = bundle.binding(scope)
        if isinstance(document, Assembly):
            continue
        if document.kinematics or any(port.body is not None for port in document.ports):
            raise NotImplementedError(f"Instance {scope!r} requires body motion evaluation")
        snapshot = copy.deepcopy(document)
        for name, param in snapshot.params.items():
            param.value = float(binding.get(name, free_vec))
            param.free = False
            param.expr = None
        snapshots[scope] = snapshot

    plotter = pv.Plotter(off_screen=off_screen)
    palette = ("#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#e41a1c", "#a65628")
    try:
        for index, (scope, part) in enumerate(snapshots.items()):
            if not part.materials:
                raise ValueError(f"Instance {scope!r} has no materials")
            for region in part.materials:
                bbox = resolve_bbox(region.sdf_tree, part)
                spacing = float(np.max(bbox.size)) / resolution
                padded = bbox.padded(spacing)
                points, shape = make_grid(padded, spacing)
                values = eval_chunked(
                    bind_sdf(region.sdf_tree, part), points, chunk_for_tree(region.sdf_tree)
                ).reshape(shape)
                image = pv.ImageData(dimensions=shape, origin=padded.min_pt, spacing=(spacing,) * 3)
                image.point_data["sdf"] = values.ravel(order="F")
                surface = image.contour([0.0], scalars="sdf")
                if not surface.n_points:
                    raise ValueError(f"Instance {scope!r}, material {region.name!r}: empty surface")
                surface.transform(poses[scope], inplace=True)
                plotter.add_mesh(
                    surface,
                    name=f"{scope}/{region.name}",
                    color=palette[index % len(palette)],
                    label=f"{scope}/{region.name}",
                    smooth_shading=True,
                )
        if not snapshots:
            raise ValueError("Assembly has no part instances")
        axis_length = max(float(plotter.length) * 0.06, 0.1)
        # Aliases share a frame, so group their labels instead of drawing overlapping text.
        labels: dict[tuple[float, ...], list[str]] = {}
        for address, frame in frames.items():
            origin = frame[:3, 3]
            labels.setdefault(tuple(origin), []).append(address)
            for axis, color in enumerate(("red", "green", "blue")):
                plotter.add_mesh(
                    pv.Arrow(start=origin, direction=frame[:3, axis], scale=axis_length),
                    color=color,
                    name=f"port/{address}/{axis}",
                )
        if labels:
            plotter.add_point_labels(
                np.asarray(list(labels)),
                ["\n".join(names) for names in labels.values()],
                always_visible=True,
                font_size=12,
            )
        plotter.add_legend()
        plotter.add_axes()
        plotter.show_grid()
        plotter.camera_position = "iso"
        return plotter
    except Exception:
        plotter.close()
        raise


def main(argv: list[str] | None = None) -> int:
    """Open an assembly or save an off-screen screenshot, reporting failures to stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sdm",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "assembly_example" / "assembly.sdm",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="evaluate geometry on CPU")
    parser.add_argument("--screenshot", type=Path, help="save a PNG instead of opening a window")
    args = parser.parse_args(argv)
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        plotter = preview_assembly(
            args.sdm, resolution=args.resolution, off_screen=args.screenshot is not None
        )
        try:
            plotter.show(title=f"{args.sdm.name} · static assembly", screenshot=args.screenshot)
        finally:
            plotter.close()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
