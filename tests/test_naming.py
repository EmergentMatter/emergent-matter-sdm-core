"""Occurrence names read from the top assembly down to the detail (ADR 0009)."""

from __future__ import annotations

from software_defined_matter import Assembly, Instance, PartRef
from software_defined_matter.assembly import full_name, naming_problems


def test_full_name_reads_top_down():
    assert full_name(("hand", "mech", "crank_driven_thumb")) == "HAND-MECH-CRANK-DRIVEN-THUMB"


def test_instances_start_with_their_parts_kind():
    asm = Assembly(
        "hand",
        instances=(
            Instance("screw_m2x6_clampring_1", PartRef("parts/screw_m2x6.sdm")),
            Instance("clampring_1_screw", PartRef("parts/screw_m2x6.sdm")),
            Instance("mech", PartRef("mech.sdm")),
        ),
    )
    stems = {"screw_m2x6_clampring_1": "screw_m2x6", "clampring_1_screw": "screw_m2x6"}
    problems = naming_problems(asm, stems)
    assert len(problems) == 1 and "clampring_1_screw" in problems[0]
