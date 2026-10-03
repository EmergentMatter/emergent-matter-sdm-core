"""Numeric shrink-wrap bbox util (software_defined_matter.bbox).

Counterpart to test_sdf_bbox.py (the analytic inferrer). The defining
property here is "works where analytic inference fails loud": today that
means param-dependent rotation (the only remaining case the analytic
inferrer refuses), so we use one to exercise the numeric path.
"""

from __future__ import annotations

import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    sdf_2d_to_3d,
    sdf_primitive,
)
from software_defined_matter.bbox import (
    BBox3,
    EmptyShrinkwrapError,
    shrinkwrap_bbox,
    tighten_part_bbox,
)
from software_defined_matter.grid_sampling import chunk_for_tree
from software_defined_matter.grid_sampling.grid import DEFAULT_CHUNK_SIZE
from software_defined_matter.sdf.bbox import BBoxInferenceError, infer_sdf_bbox


def _sphere_part(r=5.0):
    return Part(
        name="s",
        materials=[MaterialRegion(material_id=1, name="M", sdf_tree=sdf_primitive("sphere", r=r))],
    )


def test_shrinkwrap_sphere_is_tight():
    part = _sphere_part(r=5.0)
    loose = BBox3(np.array([-20.0, -20.0, -20.0]), np.array([20.0, 20.0, 20.0]))
    box = shrinkwrap_bbox(part.computed_envelope(), part, loose=loose, probe_voxel=0.5, pad=1.0)
    # Sphere surface at +-5; +1 pad; probe lands on +-5.0 exactly.
    assert np.all(box.min_pt >= -6.6) and np.all(box.min_pt <= -5.0)
    assert np.all(box.max_pt <= 6.6) and np.all(box.max_pt >= 5.0)
    # Much smaller than the seed.
    assert np.prod(box.size) < 0.1 * np.prod(loose.size)


def test_shrinkwrap_accepts_tuple_seed():
    part = _sphere_part(r=3.0)
    box = shrinkwrap_bbox(
        part.computed_envelope(),
        part,
        loose=((-15, -15, -15), (15, 15, 15)),
        probe_voxel=0.5,
        pad=0.5,
    )
    assert isinstance(box, BBox3)
    assert np.all(box.max_pt <= 4.1) and np.all(box.min_pt >= -4.1)


def test_shrinkwrap_default_seed_from_metadata():
    """loose=None resolves the seed from metadata['bbox'] (resolve_bbox)."""
    part = _sphere_part(r=4.0)
    part.metadata["bbox"] = [[-25, -25, -25], [25, 25, 25]]
    box = shrinkwrap_bbox(part.computed_envelope(), part, probe_voxel=0.5)
    assert np.all(box.max_pt <= 6.0) and np.all(box.min_pt >= -6.0)


def test_shrinkwrap_empty_raises():
    part = _sphere_part(r=5.0)
    far = BBox3(np.array([50.0, 50.0, 50.0]), np.array([60.0, 60.0, 60.0]))
    with pytest.raises(EmptyShrinkwrapError, match="No solid region"):
        shrinkwrap_bbox(part.computed_envelope(), part, loose=far)


def test_shrinkwrap_works_where_analytic_fails():
    """An exp() expression slot is analytically unsupported (interval
    inference has no exp rule; $ref rotations now resolve via the swept
    envelope), but the numeric shrinkwrap evaluates the SDF and finds the
    finite solid region.
    """
    import math

    from software_defined_matter.dsl.expr import expr_param, expr_unop

    # exp(ln 5) = 5: same sphere the old fixture rotated.
    tree = sdf_primitive("sphere", r=expr_unop("exp", expr_param("theta")))
    part = Part(
        name="r",
        params={"theta": Param("theta", math.log(5.0), free=False, unit="ratio")},
        materials=[MaterialRegion(material_id=1, name="M", sdf_tree=tree)],
        metadata={"bbox": [[-10, -10, -10], [10, 10, 10]]},
    )

    with pytest.raises(BBoxInferenceError):
        infer_sdf_bbox(tree, part, mode="values")

    box = shrinkwrap_bbox(tree, part, probe_voxel=0.4, pad=1.0)
    assert np.all(np.isfinite(box.min_pt)) and np.all(np.isfinite(box.max_pt))
    # Rotating a sphere of radius 5 doesn't change its bbox. With pad=1.0
    # the shrinkwrap fits inside ±6.5.
    assert np.all(box.max_pt <= 6.5) and np.all(box.min_pt >= -6.5)


def test_tighten_part_bbox_persists_into_metadata():
    part = _sphere_part(r=5.0)
    part.metadata["bbox"] = [[-30, -30, -30], [30, 30, 30]]
    box = tighten_part_bbox(part, probe_voxel=0.5, pad=1.0)
    assert part.metadata["bbox"] == [box.min_pt.tolist(), box.max_pt.tolist()]
    assert np.all(box.max_pt <= 6.6)


def test_tighten_part_bbox_persist_false_leaves_metadata():
    part = _sphere_part(r=5.0)
    part.metadata["bbox"] = [[-30, -30, -30], [30, 30, 30]]
    tighten_part_bbox(part, persist=False, probe_voxel=0.5)
    assert part.metadata["bbox"] == [[-30, -30, -30], [30, 30, 30]]


def test_tighten_part_bbox_no_materials_raises():
    with pytest.raises(ValueError, match="no materials"):
        tighten_part_bbox(Part(name="empty"))


def test_shrinkwrap_sizes_the_slice_from_the_tree(monkeypatch):
    """shrinkwrap holds the tree, so the slice must come from chunk_for_tree,
    not the opaque-callable default.

    The budget is patched down so a 32-gon lands on a chunk size that is
    neither DEFAULT_CHUNK_SIZE nor MAX_CHUNK_SIZE with trivial memory use.
    """
    import software_defined_matter.bbox as bbox_mod
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
    real_eval = bbox_mod.eval_chunked

    def spy(sdf_fn, points, chunk_size=DEFAULT_CHUNK_SIZE):
        seen.append(chunk_size)
        return real_eval(sdf_fn, points, chunk_size)

    monkeypatch.setattr(bbox_mod, "eval_chunked", spy)

    shrinkwrap_bbox(tree, part, probe_voxel=1.0, pad=1.0)

    assert seen == [chunk_for_tree(tree)]
    assert seen[0] != DEFAULT_CHUNK_SIZE


def test_invalid_probe_voxel_raises():
    part = _sphere_part()
    with pytest.raises(ValueError, match="probe_voxel must be positive"):
        shrinkwrap_bbox(
            part.computed_envelope(),
            part,
            loose=BBox3(np.array([-9.0] * 3), np.array([9.0] * 3)),
            probe_voxel=0.0,
        )
