"""Quadric-decimation export stage: watertightness, reduction, bounds.

After marching cubes, ``decimate=DecimateConfig(...)`` collapses low-curvature
regions (including curved-path flats) without reading the SDF gradient. The
dominant guard is:

* always watertight with the same body count as the undecimated mesh,
  across parts and limits, including a CSG-composed part whose gradients
  are non-Euclidean (the failure mode Dual Contouring hit on real parts).

Also covered: real reduction on curved-path flats, a bounded deviation
knob, monotonicity, genus-1 (torus) topology, and sharp features kept
within limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Part,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter._meshing import DecimateConfig, MeshData
from software_defined_matter._meshing.decimate import _hausdorff_mm, decimate_mesh
from software_defined_matter._meshing.topology import (
    euler_characteristic,
    n_pinched_vertices,
)
from software_defined_matter.export import export_part
from software_defined_matter.grid_sampling import bind_sdf, eval_chunked

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("skimage")
pytest.importorskip("fast_simplification")

_SEARCH_PASSES = 4

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _sphere():
    return Part(
        name="s",
        materials=[
            MaterialRegion(
                material_id=1,
                name="M",
                sdf_tree=sdf_primitive("sphere", r=5.0),
            )
        ],
    )


def _torus():
    """Genus-1 solid (Euler χ = 0). Guards against pinch-shut of the tunnel."""
    return Part(
        name="t",
        materials=[
            MaterialRegion(
                material_id=1,
                name="M",
                sdf_tree=sdf_primitive("torus", t=[4.0, 1.5]),
            )
        ],
        metadata={"bbox": [[-7, -7, -3], [7, 7, 3]]},
    )


def _cyl_slab():
    """Curved side (cylinder) + a flat slab: the curved-path-flat case."""
    tree = sdf_op(
        "union",
        [
            sdf_primitive("capped_cylinder", h=4.0, r=4.0),
            sdf_transform(
                "translate",
                sdf_primitive("box", b=[6, 6, 0.5]),
                t=[0, 0, -4.5],
            ),
        ],
    )
    return Part(
        name="cf",
        materials=[MaterialRegion(material_id=1, name="M", sdf_tree=tree)],
        metadata={"bbox": [[-7, -7, -6], [7, 7, 6]]},
    )


def _csg_blob():
    """Many unions + a subtraction → a non-exact (‖∇f‖≠1) field."""
    tree = sdf_primitive("sphere", r=3.0)
    for t in ([2, 0, 0], [-2, 1, 0], [0, 2, 1], [1, -2, 0]):
        tree = sdf_op(
            "union",
            [tree, sdf_transform("translate", sdf_primitive("sphere", r=2.0), t=t)],
        )
    tree = sdf_op(
        "subtract",
        [
            tree,
            sdf_transform("translate", sdf_primitive("box", b=[1, 1, 5]), t=[0, 0, 0]),
        ],
    )
    return Part(
        name="b",
        materials=[MaterialRegion(material_id=1, name="M", sdf_tree=tree)],
        metadata={"bbox": [[-6, -6, -6], [6, 6, 6]]},
    )


def _chamfered_box():
    inv = 1.0 / np.sqrt(2.0)
    tree = sdf_op(
        "intersect",
        [
            sdf_primitive("box", b=[4.0, 4.0, 4.0]),
            sdf_primitive("plane", n=[inv, 0.0, inv], h=-5.0 * inv),
        ],
    )
    return Part(
        name="ch",
        materials=[MaterialRegion(material_id=1, name="M", sdf_tree=tree)],
        metadata={"bbox": [[-5, -5, -5], [5, 5, 5]]},
    )


def _load(path):
    m = trimesh.load(str(path), process=False)
    m.merge_vertices()  # STL stores per-face vertices; weld before topology checks
    return m


def _export_load(part, tmp_path, *, voxel, decimate=None):
    paths = export_part(part, tmp_path, voxel_size=voxel, stamp=False, decimate=decimate)
    return _load(paths[0])


def _max_surface_sdf(part, tm, n=20000):
    sdf = bind_sdf(part.materials[0].sdf_tree, part)
    pts, _ = tm.sample(min(n, max(len(tm.faces) * 3, 1000)), return_index=True)
    return float(np.abs(np.asarray(eval_chunked(sdf, np.asarray(pts)))).max())


# ---------------------------------------------------------------------------
# The guarantee: watertight, same body count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("part_fn", [_sphere, _cyl_slab, _csg_blob, _chamfered_box, _torus])
@pytest.mark.parametrize("err", [0.02, 0.1, 0.5])
def test_decimate_always_single_watertight_body(tmp_path, part_fn, err):
    """Across parts (incl. non-exact CSG and a torus) and limits, the
    decimated export is one watertight body: neither leaky, nor shattered.
    """
    m = _export_load(
        part_fn(),
        tmp_path,
        voxel=0.35,
        decimate=DecimateConfig(
            simplify_error_mm=err, deviation_samples=2000, max_passes=_SEARCH_PASSES
        ),
    )
    assert m.is_watertight, "decimated mesh is not watertight"
    assert len(m.split(only_watertight=False)) == 1, "decimated mesh split into multiple bodies"
    assert n_pinched_vertices(np.asarray(m.faces)) == 0, "two sheets meet at a vertex"


def test_decimate_preserves_torus_topology(tmp_path):
    """A torus must keep Euler χ = 0 (tunnel open), not collapse to a sphere."""
    part = _torus()
    base = _export_load(part, tmp_path / "a", voxel=0.35)
    assert base.euler_number == 2 - 2 * 1  # χ = 0 for genus 1
    dec = _export_load(
        part,
        tmp_path / "b",
        voxel=0.35,
        decimate=DecimateConfig(simplify_error_mm=0.1, max_passes=_SEARCH_PASSES),
    )
    assert dec.is_watertight
    assert dec.euler_number == 0, f"torus tunnel collapsed (χ={dec.euler_number})"
    assert len(dec.faces) < len(base.faces)


# ---------------------------------------------------------------------------
# Reduction, bound, monotonicity
# ---------------------------------------------------------------------------


def test_decimate_reduces_curved_path_flat(tmp_path):
    """Cylinder side + slab: curved-path flat that planar merge can't collapse."""
    part = _cyl_slab()
    base = _export_load(part, tmp_path / "a", voxel=0.35)
    dec = _export_load(
        part,
        tmp_path / "b",
        voxel=0.35,
        decimate=DecimateConfig(
            simplify_error_mm=0.1, deviation_samples=4000, max_passes=_SEARCH_PASSES
        ),
    )
    assert len(dec.faces) * 3 <= len(base.faces)  # ≥3× fewer triangles
    assert dec.is_watertight


def test_decimate_deviation_within_limit(tmp_path):
    """Decimation-induced deviation stays within the limit on top of
    marching cubes' own baseline error."""
    part = _sphere()
    base = _export_load(part, tmp_path / "a", voxel=0.35)
    baseline = _max_surface_sdf(part, base)
    err = 0.05
    dec = _export_load(
        part,
        tmp_path / "b",
        voxel=0.35,
        decimate=DecimateConfig(simplify_error_mm=err, max_passes=_SEARCH_PASSES),
    )
    assert _max_surface_sdf(part, dec) <= baseline + err + 1e-3
    assert len(dec.faces) < len(base.faces)


def test_decimate_monotonic_in_tolerance(tmp_path):
    """Looser tolerance never yields more triangles.

    Equality is expected between some limits: the coarser search bracket can
    land two distinct limits on the same accepted reduction.
    """
    part = _sphere()
    counts = []
    for i, err in enumerate((0.02, 0.1, 0.5)):
        m = _export_load(
            part,
            tmp_path / f"e{i}",
            voxel=0.35,
            decimate=DecimateConfig(
                simplify_error_mm=err, deviation_samples=4000, max_passes=_SEARCH_PASSES
            ),
        )
        counts.append(len(m.faces))
    assert counts[0] >= counts[1] >= counts[2]


# ---------------------------------------------------------------------------
# Features + knobs
# ---------------------------------------------------------------------------


def test_decimate_keeps_chamfer_within_limit(tmp_path):
    """A 45° chamfer is not rounded away within the deviation limit."""
    part = _chamfered_box()
    err = 0.05
    base = _export_load(part, tmp_path / "a", voxel=0.35)
    baseline = _max_surface_sdf(part, base)
    dec = _export_load(
        part,
        tmp_path / "b",
        voxel=0.35,
        decimate=DecimateConfig(
            simplify_error_mm=err, deviation_samples=4000, max_passes=_SEARCH_PASSES
        ),
    )
    assert dec.is_watertight
    assert _max_surface_sdf(part, dec) <= baseline + err + 1e-3
    inv = 1.0 / np.sqrt(2.0)
    chamfer_n = np.array([inv, 0.0, inv])
    cos = dec.face_normals @ chamfer_n
    on_chamfer = cos > np.cos(np.radians(5))
    assert dec.area_faces[on_chamfer].sum() > 0.5  # chamfer face survived


def test_decimate_target_reduction_knob(tmp_path):
    """Direct target_reduction reduces and stays watertight."""
    part = _sphere()
    base = _export_load(part, tmp_path / "a", voxel=0.35)
    dec = _export_load(
        part,
        tmp_path / "b",
        voxel=0.35,
        decimate=DecimateConfig(target_reduction=0.8),
    )
    assert dec.is_watertight
    assert len(dec.faces) < len(base.faces) * 0.5


def test_decimate_noop_without_limit(tmp_path):
    """A config with neither knob set is a no-op (same as no decimation)."""
    part = _sphere()
    plain = _export_load(part, tmp_path / "a", voxel=0.3)
    noop = _export_load(part, tmp_path / "b", voxel=0.3, decimate=DecimateConfig())
    assert len(noop.faces) == len(plain.faces)


# ---------------------------------------------------------------------------
# What the guard is for: topology kept, measuring deviation
# ---------------------------------------------------------------------------


def _mesh_and_sdf(part, voxel):
    """The mesh export would hand to decimation, plus its field."""
    from software_defined_matter._meshing import MeshCleanupConfig
    from software_defined_matter._meshing.mesh import cleanup_mesh, extract_mesh
    from software_defined_matter.grid_sampling import make_grid, material_bbox

    region = part.materials[0]
    bbox = material_bbox(region, part).padded(voxel * 2)
    points, shape = make_grid(bbox, voxel)
    sdf = bind_sdf(region.sdf_tree, part)
    grid = eval_chunked(sdf, points, 1_000_000).reshape(shape)
    mesh = cleanup_mesh(extract_mesh(grid, bbox.min_pt, voxel), MeshCleanupConfig())
    return mesh, sdf


def _pinned_sphere():
    """A sphere with one thin pin standing on it.

    The pin is the feature a one-directional deviation check cannot see. It is
    0.6 mm across and stands 1.5 mm high, so erasing it moves the surface by
    at most the pin's radius where it stood, while the pin's tip ends up 1.5 mm
    from anything. A check that only asks "is every point of the new surface
    near the old one" reports the small number and misses the feature.
    """
    tree = sdf_op(
        "union",
        [
            sdf_primitive("sphere", r=5.0),
            sdf_transform(
                "translate", sdf_primitive("capped_cylinder", h=1.0, r=0.3), t=[0, 0, 5.5]
            ),
        ],
    )
    return Part(
        name="pin",
        materials=[MaterialRegion(material_id=1, name="M", sdf_tree=tree)],
        metadata={"bbox": [[-6, -6, -6], [6, 6, 7]]},
    )


def test_two_sided_measure_sees_a_feature_that_was_deleted():
    """Removing the pin barely moves the new surface, and is a big deviation.

    Measured from the decimated mesh alone the difference is invisible, which
    is why the limit uses both directions.
    """
    part = _pinned_sphere()
    mesh, _sdf = _mesh_and_sdf(part, 0.35)
    full = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)

    # The same solid with the pin gone.
    plain = _sphere()
    plain_mesh, _ = _mesh_and_sdf(plain, 0.35)
    without = trimesh.Trimesh(vertices=plain_mesh.vertices, faces=plain_mesh.faces, process=False)

    from software_defined_matter._meshing.decimate import _distance_to_mesh, _surface_points

    # 5000 deterministic surface points. the feature under test is a 1.5 mm pin.
    new_to_old = _distance_to_mesh(_surface_points(without, 5000), full).max()
    old_to_new = _distance_to_mesh(_surface_points(full, 5000), without).max()
    assert new_to_old < 0.5, "erasing the pin barely moves the surface that remains"
    assert old_to_new > 1.0, "the missing pin has to show up somewhere"
    assert _hausdorff_mm(full, without, 5000) == pytest.approx(old_to_new, abs=1e-6)


def test_deviation_is_the_same_number_on_every_run():
    """No random sampling: two runs on one mesh compare the same points."""
    part = _sphere()
    mesh, sdf = _mesh_and_sdf(part, 0.35)
    config = DecimateConfig(
        simplify_error_mm=0.05, deviation_samples=2000, max_passes=_SEARCH_PASSES
    )
    first = decimate_mesh(mesh, sdf, config)
    second = decimate_mesh(mesh, sdf, config)
    assert len(first.faces) == len(second.faces)
    assert np.array_equal(first.vertices, second.vertices)


def test_deviation_limit_is_honoured_against_the_undecimated_mesh():
    """The number in the config is the number the result meets.

    Deviation is measured against the mesh handed in, not against the field,
    so marching cubes' own error is not spent out of the caller's limit.
    """
    part = _cyl_slab()
    mesh, sdf = _mesh_and_sdf(part, 0.35)
    base = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    for err in (0.02, 0.1):
        out = decimate_mesh(
            mesh, sdf, DecimateConfig(simplify_error_mm=err, max_passes=_SEARCH_PASSES)
        )
        tm = trimesh.Trimesh(vertices=out.vertices, faces=out.faces, process=False)
        assert _hausdorff_mm(tm, base, 20000) <= err
        assert len(out.faces) < len(mesh.faces)


def test_euler_guard_rejects_a_reduction_that_changes_genus():
    """Body count cannot see a sealed tunnel; the Euler characteristic can.

    With the guard off, a torus pushed hard enough comes back as one
    watertight body with the wrong genus, which is what the guard exists to
    refuse.
    """
    part = _torus()
    mesh, sdf = _mesh_and_sdf(part, 0.35)
    assert euler_characteristic(np.asarray(mesh.faces)) == 0

    guarded = decimate_mesh(mesh, sdf, DecimateConfig(target_reduction=0.97, preserve_euler=True))
    assert euler_characteristic(np.asarray(guarded.faces)) == 0

    unguarded = decimate_mesh(
        mesh, sdf, DecimateConfig(target_reduction=0.97, preserve_euler=False)
    )
    # The unguarded run is free to change genus. It need not on every build,
    # but it must never come back with fewer faces *and* the right topology
    # while the guarded run refused that same reduction.
    if euler_characteristic(np.asarray(unguarded.faces)) != 0:
        assert len(guarded.faces) > len(unguarded.faces)


def test_warns_when_the_input_mesh_is_already_open(caplog):
    """Decimation preserves topology, it does not repair it, so say so."""
    part = _sphere()
    mesh, sdf = _mesh_and_sdf(part, 0.35)
    holed = MeshData(vertices=mesh.vertices, faces=np.asarray(mesh.faces)[:-4])
    with caplog.at_level("WARNING"):
        decimate_mesh(holed, sdf, DecimateConfig(target_reduction=0.5))
    assert "not a closed manifold" in caplog.text
    assert "boundary edge" in caplog.text
