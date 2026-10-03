"""Public grid sampler: usable without the ``[export]`` extra.

The invariant: ``software_defined_matter.grid_sampling`` evaluates a compiled
SDF on a voxel grid, and importing it never touches scikit-image, trimesh, or
``_meshing.mesh``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from software_defined_matter.grid_sampling import (
    BBox3,
    bind_sdf,
    eval_chunked,
    make_grid,
)


def _imported_modules(package_dir: Path) -> set[str]:
    mods: set[str] = set()
    for py in package_dir.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
    return mods


def test_grid_sampling_package_does_not_import_export_deps():
    """The public sampler's source must not mention mesh-export libraries."""
    import software_defined_matter.grid_sampling as grid_sampling

    imported = _imported_modules(Path(grid_sampling.__file__).parent)
    assert "trimesh" not in imported
    assert "skimage" not in imported
    assert "skimage.measure" not in imported
    assert not any(
        mod == "software_defined_matter._meshing"
        or mod.startswith("software_defined_matter._meshing.")
        for mod in imported
    )


def test_grid_sampling_import_does_not_load_export_deps():
    """A fresh interpreter can import grid_sampling without trimesh or skimage."""
    script = (
        "import sys\n"
        "import software_defined_matter.grid_sampling as grid_sampling\n"
        "from software_defined_matter.grid_sampling import bind_sdf, eval_chunked, make_grid\n"
        "leaked = [\n"
        "    m for m in (\n"
        "        'trimesh', 'skimage', 'skimage.measure',\n"
        "        'software_defined_matter._meshing.mesh',\n"
        "    ) if m in sys.modules\n"
        "]\n"
        "assert not leaked, leaked\n"
        "assert callable(make_grid) and callable(eval_chunked) and callable(bind_sdf)\n"
        "assert grid_sampling.__all__\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_meshing_exports_only_mesh_types():
    """``_meshing.__all__`` is the mesh types and the per-mesher configs.

    Everything that needs the optional ``[export]`` extra (the polygonisers,
    cleanup, decimation) stays behind its own module import.
    """
    import software_defined_matter._meshing as meshing

    assert set(meshing.__all__) == {
        "DecimateConfig",
        "DualContourConfig",
        "MarchingCubesConfig",
        "MeshCleanupConfig",
        "MeshData",
        "TetraConfig",
    }
    for name in meshing.__all__:
        assert hasattr(meshing, name)


def test_make_grid_uses_exact_voxel_spacing():
    """Samples sit on ``voxel_size`` centres, not ``linspace`` leftovers."""
    voxel = 0.5
    bbox = BBox3(min_pt=np.array([-2.0, -2.0, -2.0]), max_pt=np.array([2.0, 2.0, 2.0]))
    points, shape = make_grid(bbox, voxel)
    assert points.shape[1] == 3
    assert shape[0] * shape[1] * shape[2] == len(points)
    xs = np.unique(np.round(points[:, 0], decimals=12))
    gaps = np.diff(xs)
    assert np.allclose(gaps, voxel), gaps


def test_bind_sdf_and_eval_chunked_on_a_sphere():
    """A radius-5 sphere is negative at the origin and positive outside."""
    part = Part(
        name="s",
        materials=[
            MaterialRegion(material_id=1, name="M", sdf_tree=sdf_primitive("sphere", r=5.0))
        ],
    )
    sdf_fn = bind_sdf(part.materials[0].sdf_tree, part)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    d = eval_chunked(sdf_fn, points)
    assert d[0] == pytest.approx(-5.0, abs=1e-5)
    assert d[1] == pytest.approx(0.0, abs=1e-5)
    assert d[2] == pytest.approx(5.0, abs=1e-5)


def test_eval_chunked_empty_points_returns_empty():
    part = Part(
        name="s",
        materials=[
            MaterialRegion(material_id=1, name="M", sdf_tree=sdf_primitive("sphere", r=1.0))
        ],
    )
    sdf_fn = bind_sdf(part.materials[0].sdf_tree, part)
    out = eval_chunked(sdf_fn, np.empty((0, 3)))
    assert out.shape == (0,)
