"""Preview path: smoke + graceful degradation.

Preview is a display artifact, so the bar is "non-empty surfaces, right
material count, no exception", not geometric regression (that's the export
path's job, see test_export.py).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from software_defined_matter.dsl.expr import expr_param, expr_unop

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

# E402: these import the preview path, which pulls in pyvista -- they must
# come after the importorskip above so a missing optional dependency skips
# the module instead of failing the import outright.
from software_defined_matter import (  # noqa: E402
    MaterialRegion,
    Param,
    Part,
    sdf_2d_to_3d,
    sdf_primitive,
)
from software_defined_matter.grid_sampling import chunk_for_tree  # noqa: E402
from software_defined_matter.grid_sampling.grid import DEFAULT_CHUNK_SIZE  # noqa: E402
from software_defined_matter.preview import preview_part  # noqa: E402


def test_preview_smoke_single_material():
    part = Part(
        name="s",
        materials=[
            MaterialRegion(material_id=1, name="Steel", sdf_tree=sdf_primitive("sphere", r=4.0))
        ],
    )
    plotter = preview_part(part, resolution=48)
    assert len(plotter.meshes) == 1
    assert plotter.meshes[0].n_points > 0
    plotter.close()


def test_preview_skips_unresolvable_material_but_shows_rest():
    """Sphere resolves via values-mode inference; param-dependent rotation
    is analytically unsupported -> skipped with a warning, sphere still shown.
    """
    part = Part(
        name="mix",
        params={"theta": Param("theta", 0.5, free=False, unit="rad")},
        materials=[
            MaterialRegion(material_id=1, name="Steel", sdf_tree=sdf_primitive("sphere", r=3.0)),
            MaterialRegion(
                material_id=2,
                name="Lattice",
                sdf_tree=sdf_primitive("sphere", r=expr_unop("exp", expr_param("theta"))),
            ),
        ],
    )
    plotter = preview_part(part, resolution=40)
    assert len(plotter.meshes) == 1  # rotated sphere skipped, sphere kept
    plotter.close()


def test_preview_no_materials_raises():
    with pytest.raises(ValueError, match="no materials"):
        preview_part(Part(name="empty"))


def test_preview_sizes_the_slice_from_the_tree(monkeypatch):
    """Preview resolves each material's tree, so the slice must come from
    chunk_for_tree, not the opaque-callable default.

    The budget is patched down so a 32-gon lands on a chunk size that is
    neither DEFAULT_CHUNK_SIZE nor MAX_CHUNK_SIZE with trivial memory use.
    """
    import software_defined_matter.preview as preview_mod
    from software_defined_matter.grid_sampling import grid as grid_mod

    monkeypatch.setattr(grid_mod, "CHUNK_BUDGET_BYTES", 4_000_000)

    verts = [
        [5.0 * np.cos(t), 5.0 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ]
    tree = sdf_2d_to_3d("extrusion", sdf_primitive("polygon_2d", vertices=verts), h=2.0)
    part = Part(
        name="poly",
        materials=[MaterialRegion(material_id=1, name="M", sdf_tree=tree)],
        metadata={"bbox": [[-8, -8, -4], [8, 8, 4]]},
    )

    seen: list[int] = []
    real_eval = preview_mod.eval_chunked

    def spy(sdf_fn, points, chunk_size=DEFAULT_CHUNK_SIZE):
        seen.append(chunk_size)
        return real_eval(sdf_fn, points, chunk_size)

    monkeypatch.setattr(preview_mod, "eval_chunked", spy)

    plotter = preview_part(part, resolution=24)
    try:
        assert seen == [chunk_for_tree(tree)]
        assert seen[0] != DEFAULT_CHUNK_SIZE
    finally:
        plotter.close()


def test_preview_all_skipped_raises():
    part = Part(
        name="r",
        params={"theta": Param("theta", 0.5, free=False, unit="rad")},
        materials=[
            MaterialRegion(
                material_id=1,
                name="M",
                sdf_tree=sdf_primitive("sphere", r=expr_unop("exp", expr_param("theta"))),
            )
        ],
    )
    with pytest.raises(ValueError, match="[Ee]very material"):
        preview_part(part)
