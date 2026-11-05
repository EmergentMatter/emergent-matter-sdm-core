"""Dimension audit: free, input and driven dimensions pass; a dangling number, an unused
dimension, a typed-in offset and a bad declaration are each caught."""

from __future__ import annotations

import dataclasses

import pytest

from software_defined_matter import MaterialRegion, Part, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.dimension_audit import (
    DimensionAuditError,
    assert_dimensions,
    dimension_audit,
    find_declaration_problems,
    optimizable,
)


@dataclasses.dataclass(frozen=True)
class Block:
    d_half_width: float = 10.0  # input: a requirement
    d_hole_radius: float = 3.0  # input: a standard clearance
    d_hole_inset: float = 0.4  # free: where the hole sits, as a fraction of the half width
    d_min_wall: float = 1.2  # input: a process floor feeding validate(), not geometry
    d_spare: float = 5.0  # declared and used by nothing

    @property
    def d_hole_x(self) -> float:
        """Driven: computed from the half width and the inset."""
        return self.d_half_width * self.d_hole_inset


def _block(p: Block, hole_x: float, hole_r: float) -> Part:
    body = sdf_primitive("box", b=[p.d_half_width, p.d_half_width, p.d_half_width / 2])
    hole = sdf_transform("translate", sdf_primitive("sphere", r=hole_r), t=[hole_x, 0.0, 0.0])
    tree = sdf_op("subtract", [body, hole])
    return Part(name="block", materials=[MaterialRegion(material_id=1, name="body", sdf_tree=tree)])


def build_clean(p: Block) -> Part:
    return _block(p, p.d_hole_x, p.d_hole_radius)


def build_dangling(p: Block) -> Part:
    return _block(p, 4.0, p.d_hole_radius)  # the hole's place is typed in


def build_offset(p: Block) -> Part:
    return _block(p, p.d_hole_x, p.d_hole_radius + 0.3)  # a clearance typed onto a dimension


DECLARED: dict[str, dict] = {
    "d_half_width": {"origin": "input", "unit": "mm", "source": "requirement:F1.1"},
    "d_hole_radius": {
        "origin": "input",
        "unit": "mm",
        "source": "standard:ISO 273 medium clearance for M6",
        "tradeable": True,
    },
    "d_hole_inset": {
        "origin": "free",
        "unit": "ratio",
        "bounds": [0.1, 0.7],
        "source": "choice: anywhere the ligaments stay above the minimum wall works",
    },
    "d_min_wall": {
        "origin": "input",
        "unit": "mm",
        "role": "derived",
        "source": "process:fdm_pla:min_wall",
    },
}


def test_free_input_and_driven_pass():
    report = assert_dimensions(build_clean, Block(), DECLARED)
    # the inset drives only the hole's place; the half width drives the box and, through the
    # driven hole position, the hole too
    assert set(report.driving) == {"d_half_width", "d_hole_radius", "d_hole_inset"}
    assert len(report.drives["d_hole_inset"]) == 1
    assert len(report.drives["d_half_width"]) > len(report.drives["d_hole_inset"])
    assert report.drives["d_min_wall"] == []  # a limit, not geometry: allowed


def test_optimizer_moves_free_and_tradeable_inputs_only():
    assert optimizable(DECLARED) == ["d_hole_radius", "d_hole_inset"]


def test_a_dangling_number_is_found():
    kinds = [f.kind for f in dimension_audit(build_dangling, Block(), DECLARED).findings]
    assert kinds.count("dangling") == 1


def test_an_unused_dimension_is_found():
    declared = {**DECLARED, "d_spare": {"origin": "free", "bounds": [1, 9], "source": "tool: x"}}
    found = dimension_audit(build_clean, Block(), declared).findings
    assert [(f.kind, f.where) for f in found] == [("unused", "d_spare")]


def test_a_typed_in_offset_is_found():
    kinds = [f.kind for f in dimension_audit(build_offset, Block(), DECLARED).findings]
    assert kinds == ["offset"]


def test_declarations_are_checked():
    found = find_declaration_problems(
        {
            "no_origin": {"source": "requirement:F1"},
            "no_source": {"origin": "input"},
            "bad_source": {"origin": "input", "source": "because"},
            "thin_choice": {"origin": "input", "source": "choice: looks fine"},
            "free_unbounded": {"origin": "free", "source": "choice: anything in reason works here"},
            "free_tradeable": {
                "origin": "free",
                "bounds": [0, 1],
                "tradeable": True,
                "source": "choice: anything in reason works here",
            },
            "tool": {"role": "tool", "source": "tool: meshing resolution"},
            "parent": {"origin": "input", "source": "parent:motor:case_radius"},
        }
    )
    assert [f.where for f in found] == [
        "no_origin",
        "no_source",
        "bad_source",
        "thin_choice",
        "free_unbounded",
        "free_tradeable",
    ]


def test_assert_lists_every_finding():
    with pytest.raises(DimensionAuditError) as err:
        assert_dimensions(build_dangling, Block(), {**DECLARED, "x": {}})
    assert "[declaration] x" in str(err.value) and "[dangling]" in str(err.value)


@dataclasses.dataclass(frozen=True)
class Plate:
    d_thickness: float = 3.0  # input
    d_voxel_size_mm: float = 0.25  # tool setting

    @property
    def d_overshoot(self) -> float:
        """Driven by a tool setting: a cut passes the faces by a few voxels."""
        return 4.0 * self.d_voxel_size_mm


def build_plate(p: Plate) -> Part:
    body = sdf_primitive("box", b=[20.0 * p.d_thickness, 20.0 * p.d_thickness, p.d_thickness])
    cut = sdf_primitive("capped_cylinder", r=p.d_thickness, h=p.d_thickness + p.d_overshoot)
    tree = sdf_op("subtract", [body, cut])
    return Part(name="plate", materials=[MaterialRegion(material_id=1, name="body", sdf_tree=tree)])


def test_a_cut_overshoot_from_a_tool_setting_is_not_an_offset():
    declared = {
        "d_thickness": {"origin": "input", "unit": "mm", "source": "requirement:F2"},
        "d_voxel_size_mm": {"role": "tool", "unit": "mm", "source": "tool: mesh resolution"},
    }
    assert_dimensions(build_plate, Plate(), declared)
