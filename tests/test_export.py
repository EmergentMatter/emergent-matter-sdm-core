"""Export path: sampler fidelity + per-material files + overlap resolution.

The radius regression directly guards the ``np.linspace`` -> ``np.arange``
grid-drift class of bug (see docs/adr/0002-preview-and-export-modules.md): if the
grid is sampled at a spacing different from the one handed to marching
cubes, a canonical sphere's vertices drift off its true radius.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from software_defined_matter.dsl.expr import expr_param, expr_unop

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("skimage")

# E402: these import the export path, which pulls in trimesh/skimage -- they
# must come after the importorskip calls above so a missing optional
# dependency skips the module instead of failing the import outright.
from software_defined_matter import (  # noqa: E402
    MaterialRegion,
    Part,
    sdf_2d_to_3d,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.export import export_part  # noqa: E402
from software_defined_matter.grid_sampling import chunk_for_tree  # noqa: E402
from software_defined_matter.grid_sampling.grid import DEFAULT_CHUNK_SIZE  # noqa: E402


def _sphere_part(name="s", r=5.0, mat="Steel"):
    return Part(
        name=name,
        materials=[MaterialRegion(material_id=1, name=mat, sdf_tree=sdf_primitive("sphere", r=r))],
    )


def _radii(path):
    m = trimesh.load(str(path), process=False)
    return np.linalg.norm(np.asarray(m.vertices), axis=1)


def test_sphere_radius_regression(tmp_path):
    """All MC vertices of a radius-5 sphere land within ~1 voxel of r=5."""
    voxel = 0.25
    part = _sphere_part(r=5.0)
    paths = export_part(part, tmp_path, voxel_size=voxel, fmt="stl")

    assert len(paths) == 1
    radii = _radii(paths[0])
    err = np.abs(radii - 5.0)
    # arange sampler keeps this well under one voxel; the linspace bug
    # produced errors several times this on comparable grids.
    assert err.max() < voxel, f"max radial error {err.max():.4f} >= voxel {voxel}"
    assert abs(radii.mean() - 5.0) < voxel * 0.5


def test_one_file_per_material_disjoint(tmp_path, fixed_stamp):
    """Two well-separated materials -> two named files (fixed-stamp fixture
    keeps the assertion deterministic instead of racing the wall clock)."""
    a = MaterialRegion(
        material_id=1,
        name="Cu",
        sdf_tree=sdf_transform("translate", sdf_primitive("sphere", r=2.0), t=[-6.0, 0.0, 0.0]),
    )
    b = MaterialRegion(
        material_id=2,
        name="PLA",
        sdf_tree=sdf_transform("translate", sdf_primitive("sphere", r=2.0), t=[6.0, 0.0, 0.0]),
    )
    part = Part(name="duo", materials=[a, b], metadata={"bbox": [[-9, -3, -3], [9, 3, 3]]})

    paths = export_part(part, tmp_path, voxel_size=0.3, fmt="stl", stamp=fixed_stamp)
    names = sorted(p.name for p in paths)
    assert names == [f"duo_Cu_{fixed_stamp}.stl", f"duo_PLA_{fixed_stamp}.stl"]
    assert all(p.exists() for p in paths)


def test_resolve_overlaps_carves_later_material(tmp_path, fixed_stamp):
    """Material A (big sphere) listed before B (small concentric sphere):
    B overrides A where they overlap, so A's exported mesh is a shell with
    an inner cavity at B's radius. With resolve_overlaps=False A is solid.
    """
    big = MaterialRegion(material_id=1, name="A", sdf_tree=sdf_primitive("sphere", r=5.0))
    small = MaterialRegion(material_id=2, name="B", sdf_tree=sdf_primitive("sphere", r=2.5))
    part = Part(name="nest", materials=[big, small], metadata={"bbox": [[-6, -6, -6], [6, 6, 6]]})

    resolved = export_part(
        part, tmp_path / "on", voxel_size=0.2, resolve_overlaps=True, stamp=fixed_stamp
    )
    raw = export_part(
        part, tmp_path / "off", voxel_size=0.2, resolve_overlaps=False, stamp=fixed_stamp
    )

    a_name = f"nest_A_{fixed_stamp}.stl"
    a_resolved = _radii(next(p for p in resolved if p.name == a_name))
    a_raw = _radii(next(p for p in raw if p.name == a_name))

    # Carved: an inner cavity surface appears near r=2.5.
    assert a_resolved.min() < 3.0
    assert a_resolved.max() > 4.5  # outer surface still ~5
    # Solid: only the outer surface, nothing near the centre.
    assert a_raw.min() > 4.0


def test_default_stamp_observes_convention(tmp_path):
    """The DEFAULT export observes the org's ISO 8601 filename-stamp convention:
    a live YYYY-MM-DDTHHMM datetime stamp, T separator, no colons/spaces.
    """
    paths = export_part(_sphere_part(mat="Steel"), tmp_path, voxel_size=0.4)
    name = paths[0].name
    assert re.fullmatch(r"s_Steel_\d{4}-\d{2}-\d{2}T\d{4}\.stl", name), name
    assert ":" not in name and " " not in name


def test_fixed_stamp_is_verbatim_and_deterministic(tmp_path, fixed_stamp):
    """An explicit stamp string is used verbatim -> exact, race-free name."""
    paths = export_part(_sphere_part(mat="Steel"), tmp_path, voxel_size=0.4, stamp=fixed_stamp)
    assert paths[0].name == f"s_Steel_{fixed_stamp}.stl"


def test_date_stamp_suffix(tmp_path):
    """stamp='date' -> date-only YYYY-MM-DD variant of the convention."""
    paths = export_part(_sphere_part(mat="Steel"), tmp_path, voxel_size=0.4, stamp="date")
    assert re.fullmatch(r"s_Steel_\d{4}-\d{2}-\d{2}\.stl", paths[0].name), paths[0].name


def test_stamp_false_is_unstamped(tmp_path):
    """Opt-out: stamp=False keeps a bare deterministic filename."""
    paths = export_part(_sphere_part(mat="Steel"), tmp_path, voxel_size=0.4, stamp=False)
    assert paths[0].name == "s_Steel.stl"


def test_stamp_shared_across_materials(tmp_path):
    """All material meshes in one call share a single timestamp."""
    a = MaterialRegion(
        material_id=1,
        name="Cu",
        sdf_tree=sdf_transform("translate", sdf_primitive("sphere", r=2.0), t=[-6.0, 0.0, 0.0]),
    )
    b = MaterialRegion(
        material_id=2,
        name="PLA",
        sdf_tree=sdf_transform("translate", sdf_primitive("sphere", r=2.0), t=[6.0, 0.0, 0.0]),
    )
    part = Part(name="duo", materials=[a, b], metadata={"bbox": [[-9, -3, -3], [9, 3, 3]]})
    paths = export_part(part, tmp_path, voxel_size=0.3, stamp="datetime")
    stamps = {re.search(r"_(\d{4}-\d{2}-\d{2}T\d{4})\.stl$", p.name).group(1) for p in paths}
    assert len(stamps) == 1


def test_no_materials_raises(tmp_path):
    with pytest.raises(ValueError, match="no materials"):
        export_part(Part(name="empty"), tmp_path, voxel_size=0.5)


def _polygon_part_pair():
    """A polygon-heavy material (wide tree) and an analytic one (narrow)."""
    verts = [
        [5.0 * np.cos(t), 5.0 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ]
    poly = sdf_transform(
        "translate",
        sdf_2d_to_3d("extrusion", sdf_primitive("polygon_2d", vertices=verts), h=2.0),
        t=[-6.0, 0.0, 0.0],
    )
    a = MaterialRegion(material_id=1, name="Poly", sdf_tree=poly)
    b = MaterialRegion(
        material_id=2,
        name="Steel",
        sdf_tree=sdf_transform("translate", sdf_primitive("sphere", r=2.0), t=[6.0, 0.0, 0.0]),
    )
    part = Part(name="duo", materials=[a, b], metadata={"bbox": [[-12, -6, -3], [9, 6, 3]]})
    return part, a, b


def _spy_on_export_eval(monkeypatch):
    import software_defined_matter.export as export_mod

    seen: list[int] = []
    real_eval = export_mod.eval_chunked

    def spy(sdf_fn, points, chunk_size=DEFAULT_CHUNK_SIZE):
        seen.append(chunk_size)
        return real_eval(sdf_fn, points, chunk_size)

    monkeypatch.setattr(export_mod, "eval_chunked", spy)
    return seen


def test_export_sizes_each_materials_slice_from_its_tree(tmp_path, monkeypatch):
    """Each material's slice comes from its own tree: the polygon material
    gets a smaller slice than the analytic one, not a shared constant.

    The budget is patched down so a 32-gon lands on a chunk size that is
    neither DEFAULT_CHUNK_SIZE nor MAX_CHUNK_SIZE with trivial memory use.
    """
    from software_defined_matter.grid_sampling import grid as grid_mod

    monkeypatch.setattr(grid_mod, "CHUNK_BUDGET_BYTES", 4_000_000)
    part, a, b = _polygon_part_pair()
    seen = _spy_on_export_eval(monkeypatch)

    export_part(part, tmp_path, voxel_size=0.5, fmt="stl", stamp=False)

    # resolve_overlaps=True: material A evaluates A then B (subtraction),
    # material B evaluates B again.
    assert seen == [
        chunk_for_tree(a.sdf_tree),
        chunk_for_tree(b.sdf_tree),
        chunk_for_tree(b.sdf_tree),
    ]
    assert seen[0] != DEFAULT_CHUNK_SIZE


def test_export_explicit_chunk_size_overrides_tree_sizing(tmp_path, monkeypatch):
    part, _a, _b = _polygon_part_pair()
    seen = _spy_on_export_eval(monkeypatch)

    export_part(part, tmp_path, voxel_size=0.5, fmt="stl", stamp=False, chunk_size=777)

    assert seen == [777, 777, 777]


def test_unresolvable_bbox_fails_loud(tmp_path):
    """Param-dependent rotation is analytically unsupported; without metadata
    bbox the export path must fail loud rather than silently fabricate a
    domain.
    """
    from software_defined_matter import Param
    from software_defined_matter.grid_sampling import BBoxResolutionError

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
    with pytest.raises(BBoxResolutionError):
        export_part(part, tmp_path, voxel_size=0.5)
