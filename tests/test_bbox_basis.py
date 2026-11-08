"""A stored bounding box is used only while what it was inferred over still holds.

``Part.infer_bbox_from_sdf`` and ``tighten_part_bbox`` persist a box with a
``metadata["bbox_basis"]`` record. Every reader of the stored box (sampling,
export, metric compilation, shader emission, motion bounds) sets such a box
aside once the part has changed under it, and keeps an authored or unlabelled
box exactly as stored.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    load_part,
    make_param_ref,
    save,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.bbox import tighten_part_bbox
from software_defined_matter.dsl.expr import _resolve_bbox as metric_domain
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.grid_sampling import material_bbox, resolve_bbox
from software_defined_matter.motion_bounds import infer_motion_bounds
from software_defined_matter.sdf.bbox import (
    BBOX_BASIS_KEY,
    bbox_basis,
    infer_sdf_bbox,
    stored_bbox,
    stored_bbox_status,
)

#: A point of the radius-2 sphere that lies outside every box made for radius 1.
GROWN_POINT = np.array([1.9, 0.0, 0.0])

#: Analytic inference of a sphere is exact up to float rounding.
EXACT = 1e-9

CROP = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def _ball(r: float = 1.0, *, free: bool = False, bounds: tuple[float, float] | None = None) -> Part:
    tree = sdf_primitive("sphere", r=make_param_ref("r"))
    return Part(
        name="ball",
        params={"r": Param("r", value=r, free=free, bounds=bounds, unit="mm")},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
    )


def _grow(part: Part, r: float = 2.0) -> None:
    part.params["r"].value = r
    part.refresh_derived()


def _domain(part: Part) -> tuple[np.ndarray, np.ndarray]:
    box = resolve_bbox(part.computed_envelope(), part)
    return box.min_pt, box.max_pt


def _contains(lo: object, hi: object, point: np.ndarray) -> bool:
    return bool(np.all(np.asarray(lo) <= point) and np.all(point <= np.asarray(hi)))


def test_a_part_with_no_stored_box_reports_it_absent():
    part = _ball()
    assert stored_bbox_status(part) == "absent"
    assert stored_bbox(part) is None


def test_a_box_inferred_for_a_fixed_radius_is_set_aside_once_the_radius_grows():
    part = _ball()
    part.infer_bbox_from_sdf()
    assert stored_bbox_status(part) == "valid"
    _grow(part)
    assert stored_bbox_status(part) == "stale"
    assert stored_bbox(part) is None
    lo, hi = _domain(part)
    assert lo == pytest.approx([-2.0, -2.0, -2.0], abs=EXACT)
    assert hi == pytest.approx([2.0, 2.0, 2.0], abs=EXACT)


@pytest.mark.parametrize(
    "edit_before_save", [True, False], ids=["edit then save", "save then edit"]
)
def test_the_set_aside_holds_across_a_save_and_a_load(tmp_path, edit_before_save):
    part = _ball()
    part.infer_bbox_from_sdf()
    if edit_before_save:
        _grow(part)
    save(part, tmp_path / "ball.sdm")
    loaded = load_part(tmp_path / "ball.sdm")
    if not edit_before_save:
        assert stored_bbox_status(loaded) == "valid"
        _grow(loaded)
    assert stored_bbox_status(loaded) == "stale"
    assert _contains(*_domain(loaded), GROWN_POINT)


def test_the_export_domain_and_mesh_reach_past_the_old_box_and_its_padding(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("skimage")
    from software_defined_matter.export import export_part

    part = _ball()
    part.infer_bbox_from_sdf()
    _grow(part)
    domain = material_bbox(part.materials[0], part)
    assert _contains(domain.min_pt, domain.max_pt, GROWN_POINT)
    # Export pads each region's box by two voxels, so the radius-1 box would
    # reach 1.2 at most: a mesh past 1.9 comes only from the fresh domain.
    paths = export_part(part, tmp_path, voxel_size=0.1, stamp=False)
    mesh = trimesh.load(paths[0])
    assert mesh.bounds[1][0] > GROWN_POINT[0]


@pytest.mark.parametrize(
    "basis, status",
    [({"source": "authored"}, "authored"), (None, "unlabelled")],
    ids=["labelled authored", "unlabelled"],
)
def test_an_authored_crop_is_kept_as_stored_after_the_radius_grows(basis, status):
    part = _ball()
    part.metadata["bbox"] = [list(corner) for corner in CROP]
    if basis is not None:
        part.metadata[BBOX_BASIS_KEY] = basis
    _grow(part)
    # Never "valid": a crop says nothing about whether the part fits inside it.
    assert stored_bbox_status(part) == status
    assert stored_bbox(part) == CROP
    lo, hi = _domain(part)
    assert hi == pytest.approx(CROP[1], abs=EXACT)
    assert part.metadata["bbox"] == CROP


def test_an_unlabelled_box_is_used_as_stored_until_inference_labels_it():
    part = _ball()
    # The form an earlier version persisted, with no basis to judge it by.
    part.metadata["bbox"] = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    _grow(part)
    assert stored_bbox_status(part) == "unlabelled"
    assert _domain(part)[1] == pytest.approx([1.0, 1.0, 1.0], abs=EXACT)
    part.infer_bbox_from_sdf()
    assert stored_bbox_status(part) == "valid"
    assert _domain(part)[1] == pytest.approx([2.0, 2.0, 2.0], abs=EXACT)


def test_a_box_replaced_by_hand_is_not_judged_by_the_record_of_the_box_it_replaced():
    part = _ball()
    part.infer_bbox_from_sdf()
    part.metadata["bbox"] = [list(corner) for corner in CROP]
    _grow(part)
    assert stored_bbox_status(part) == "unlabelled"
    assert stored_bbox(part) == CROP


def test_a_bounds_mode_box_stays_valid_while_the_radius_moves_inside_its_bounds():
    part = _ball(free=True, bounds=(0.5, 3.0))
    part.infer_bbox_from_sdf()
    _grow(part)
    assert stored_bbox_status(part) == "valid"
    assert _domain(part)[1] == pytest.approx([3.0, 3.0, 3.0], abs=EXACT)


def _value_past_bounds(part: Part) -> None:
    part.params["r"].value = 4.0


def _bounds_widened(part: Part) -> None:
    part.params["r"].bounds = (0.5, 4.0)


def _geometry_moved(part: Part) -> None:
    moved = sdf_transform("translate", part.materials[0].sdf_tree, t=[1.0, 0.0, 0.0])
    part.materials[0] = MaterialRegion(material_id=1, name="m", sdf_tree=moved)


@pytest.mark.parametrize(
    "change",
    [_value_past_bounds, _bounds_widened, _geometry_moved],
    ids=["value past its bounds", "bounds widened", "geometry moved"],
)
def test_a_bounds_mode_box_is_set_aside_when_its_basis_changes(change: Callable[[Part], None]):
    part = _ball(free=True, bounds=(0.5, 3.0))
    part.infer_bbox_from_sdf()
    change(part)
    assert stored_bbox_status(part) == "stale"
    lo, hi = _domain(part)
    fresh_lo, fresh_hi = infer_sdf_bbox(part.computed_envelope(), part, mode="values")
    assert lo == pytest.approx(fresh_lo, abs=EXACT)
    assert hi == pytest.approx(fresh_hi, abs=EXACT)


def test_retightening_after_growth_seeds_from_a_fresh_box_not_the_old_tight_one():
    part = _ball(free=True, bounds=(0.5, 3.0))
    tighten_part_bbox(part, probe_voxel=0.1, pad=0.1)
    assert not _contains(*part.metadata["bbox"], GROWN_POINT)
    part.params["r"].value = 2.0
    assert stored_bbox_status(part) == "stale"
    box = tighten_part_bbox(part, probe_voxel=0.1, pad=0.1)
    assert _contains(box.min_pt, box.max_pt, GROWN_POINT)
    assert stored_bbox_status(part) == "valid"


def _relation_ball() -> Part:
    double = {
        "type": "binop",
        "op": "*",
        "lhs": {"type": "num", "value": 2.0},
        "rhs": {"type": "param", "name": "a"},
    }
    part = Part(
        name="ball",
        params={
            "a": Param("a", value=0.5, free=True, bounds=(0.25, 1.5), unit="mm"),
            "r": Param("r", value=1.0, unit="mm", expr=double),
        },
        materials=[
            MaterialRegion(
                material_id=1, name="m", sdf_tree=sdf_primitive("sphere", r=make_param_ref("r"))
            )
        ],
    )
    part.refresh_derived()
    return part


def test_a_relation_driven_radius_sets_the_box_aside_when_its_input_moves_inside_its_bounds():
    part = _relation_ball()
    part.infer_bbox_from_sdf()
    part.params["a"].value = 1.0
    # Stale even before the relation's cached value is refreshed.
    assert stored_bbox_status(part) == "stale"
    part.refresh_derived()
    assert stored_bbox_status(part) == "stale"
    assert _contains(*_domain(part), GROWN_POINT)


def test_metric_compilation_sets_a_stale_box_aside():
    part = _ball()
    part.infer_bbox_from_sdf()
    _grow(part)
    lo, hi = metric_domain(part, None)
    assert _contains(lo, hi, GROWN_POINT)


def test_shader_emission_sets_a_stale_box_aside():
    part = _ball()
    part.infer_bbox_from_sdf()
    _grow(part)
    assert _contains(*emit_glsl(part).bbox, GROWN_POINT)


def test_motion_bounds_leave_a_stale_inferred_box_out_of_the_scene_box():
    region = sdf_transform("translate", sdf_primitive("sphere", r=make_param_ref("r")), t=[4, 0, 0])
    region["name"] = "region"
    part = Part(
        name="arm",
        params={"r": Param("r", value=2.0, unit="mm")},
        materials=[MaterialRegion(material_id=1, name="solid", sdf_tree=region)],
        kinematics={
            "dofs": [{"name": "q", "kind": "angle", "unit": "rad", "range": [-1, 1], "default": 0}],
            "bodies": [
                {
                    "name": "body",
                    "region": {"$node": "region"},
                    "motion": {
                        "ops": [
                            {
                                "kind": "rotate",
                                "axis": [0, 0, 1],
                                "origin": [0, 0, 0],
                                "angle": {"type": "dof", "name": "q"},
                            }
                        ]
                    },
                }
            ],
        },
    )
    part.infer_bbox_from_sdf()
    _grow(part, r=0.1)
    bounds = infer_motion_bounds(part)
    assert bounds is not None and bounds.bbox is not None
    # The box inferred at radius 2 reaches z = 2; the moving radius-0.1 body does not.
    assert bounds.bbox[1][2] < 1.0


def test_a_malformed_basis_record_fails_loud():
    part = _ball()
    part.metadata["bbox"] = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    part.metadata[BBOX_BASIS_KEY] = {"source": "guessed"}
    with pytest.raises(ValueError, match="must be a mapping whose 'source'"):
        stored_bbox_status(part)


def test_bbox_basis_refuses_to_fingerprint_an_authored_box():
    with pytest.raises(ValueError, match="source must be 'inferred' or 'tightened'"):
        bbox_basis(_ball(), [[-1, -1, -1], [1, 1, 1]], source="authored", mode="bounds")
