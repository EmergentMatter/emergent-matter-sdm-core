"""Smoke test: every 3-D solid / lattice primitive meshes to a real surface.

Marching cubes only produces geometry where the field crosses zero. A primitive
whose field is single-signed (all positive or all negative) yields *no* surface:
``extract_mesh`` then raises, or returns an empty mesh. That is exactly how the
``octahedron`` bug shipped silently: its interior never went negative, so the
exported STL was an empty 84-byte file and nothing rendered.

This test binds each primitive with representative parameters, meshes it on a
grid over a generous bounding cube, and asserts a non-empty, finite mesh: a
cheap, Blender-free guard against the next single-signed regression.

Deliberately excluded: ``plane`` (infinite / open, no closed interior) and the
2-D profile primitives (``circle_2d`` … ``bspline_2d``), which are 2-D fields
meant for ``extrusion`` / ``revolution`` / ``sweep``, not direct 3-D meshing.
"""

from __future__ import annotations

import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from tests.conftest import PRIMITIVES

pytest.importorskip("skimage")  # marching cubes lives in the export/dev extra

from software_defined_matter._meshing.mesh import extract_mesh
from software_defined_matter.grid_sampling import BBox3, bind_sdf, eval_chunked, make_grid


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_primitive_meshes_to_nonempty_surface(name):
    kwargs, half = PRIMITIVES[name]
    part = Part(
        name=name,
        params={},
        materials=[
            MaterialRegion(material_id=1, name="mat", sdf_tree=sdf_primitive(name, **kwargs))
        ],
    )

    bbox = BBox3(min_pt=np.full(3, -half), max_pt=np.full(3, half))
    voxel = (2.0 * half) / 50.0
    points, shape = make_grid(bbox, voxel)
    grid = np.asarray(eval_chunked(bind_sdf(part.materials[0].sdf_tree, part), points))

    # The field must straddle zero, or there is nothing to mesh (the octahedron
    # bug: all-positive field). Asserted explicitly for a clear failure message.
    assert grid.min() < 0.0 < grid.max(), (
        f"{name}: field is single-signed (min={grid.min():.3f}, max={grid.max():.3f}): "
        f"no iso-surface, primitive would not mesh"
    )

    mesh = extract_mesh(grid.reshape(shape), bbox.min_pt, voxel)
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0, f"{name}: empty mesh"
    assert np.all(np.isfinite(mesh.vertices)), f"{name}: non-finite vertices"
