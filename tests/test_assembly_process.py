"""An assembly's build sequence round-trips through a file and is checked against the assembly."""

from __future__ import annotations

import pytest

from software_defined_matter import Assembly, Instance, PartRef, load, save
from software_defined_matter.assembly import (
    AssemblyProcess,
    AssemblyStep,
    Twist,
    process_of,
    with_process,
)


def _assembly():
    return Assembly(
        "box",
        instances=(
            Instance("frame", PartRef("frame.sdm")),
            Instance("lid", PartRef("lid.sdm")),
            Instance("screw", PartRef("screw.sdm")),
        ),
    )


PROCESS = AssemblyProcess(
    base=("frame",),
    steps=(
        AssemblyStep("lid", "Fit the lid", ("lid",), direction=(0, 0, -2), travel=30),
        AssemblyStep(
            "screw",
            "Screw the lid down",
            ("screw",),
            direction=(0, 0, -1),
            rotate=Twist((0, 0, 1), 720),
            tool="2.5 mm hex key, 1.2 N m",
            fasteners=("screw",),
        ),
    ),
)


def test_round_trips_through_a_saved_file(tmp_path):
    path = tmp_path / "box.sdm"
    save(with_process(_assembly(), PROCESS), path)
    back = process_of(load(path, b_validate_schema=False))
    assert back == PROCESS
    assert back.steps[0].direction == (0.0, 0.0, -1.0)  # normalised


def test_a_complete_sequence_has_no_problems():
    assert PROCESS.problems(_assembly()) == []


def test_problems_name_what_is_wrong():
    bad = AssemblyProcess(
        steps=(
            AssemblyStep("a", "A", ("lid", "ghost"), direction=(1, 0, 0)),
            AssemblyStep("b", "B", ("lid",)),
        )
    )
    found = " ".join(bad.problems(_assembly()))
    assert "'ghost'" in found and "already installed" in found and "no direction" in found
    assert "'frame' is never installed" in found and "'screw' is never installed" in found


def test_with_process_refuses_an_incomplete_sequence():
    incomplete = AssemblyProcess(
        steps=(AssemblyStep("lid", "Fit the lid", ("lid",), direction=(0, 0, -1)),)
    )
    with pytest.raises(ValueError, match="never installed"):
        with_process(_assembly(), incomplete)


def test_a_step_must_install_something_and_point_somewhere():
    with pytest.raises(ValueError):
        AssemblyStep("x", "Nothing", ())
    with pytest.raises(ValueError):
        AssemblyStep("x", "Zero", ("lid",), direction=(0, 0, 0))


def test_with_process_lists_instances_in_build_order():
    shuffled = Assembly(
        "box",
        instances=(
            Instance("screw", PartRef("screw.sdm")),
            Instance("frame", PartRef("frame.sdm")),
            Instance("lid", PartRef("lid.sdm")),
        ),
    )
    built = with_process(shuffled, PROCESS)
    assert [i.id for i in built.instances] == PROCESS.build_order() == ["frame", "lid", "screw"]
