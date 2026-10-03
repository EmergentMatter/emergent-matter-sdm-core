"""Build a grounded actuator with a sliding carriage and a rotating arm.

Run ``uv run python examples/build_placement_example.py OUTPUT_DIRECTORY``.
The root bundle retains separate part files and a nested arm assembly. Placement
is evaluated by core; no trajectory generator or closed-mechanism solver runs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from software_defined_matter import (
    Assembly,
    Dof,
    Frame,
    Instance,
    Mate,
    MaterialRegion,
    Param,
    Part,
    PartRef,
    Port,
    compile_placement,
    load_bundle,
    save,
    sdf_primitive,
    sdf_transform,
)

__all__ = ["build_placement_bundle", "main"]


def build_placement_bundle(directory: Path) -> Path:
    """Write, pin and validate an actuator whose arm length changes its output pose."""
    directory.mkdir(parents=True, exist_ok=True)
    parts = directory / "parts"
    parts.mkdir(exist_ok=True)
    base = Part(
        "base",
        materials=[MaterialRegion(1, "base", sdf_primitive("box", b=[5, 5, 1]))],
        ports=[Port("mount", Frame(position=(0, 0, 1)))],
    )
    carriage = Part(
        "carriage",
        materials=[MaterialRegion(1, "carriage", sdf_primitive("box", b=[2, 2, 1]))],
        ports=[
            Port("bottom", Frame(position=(0, 0, -1))),
            Port("pivot", Frame(position=(0, 0, 1))),
        ],
    )
    arm = Part(
        "arm",
        params={"length": Param("length", 8, unit="mm")},
        materials=[
            MaterialRegion(
                1,
                "arm",
                sdf_transform(
                    "translate",
                    sdf_primitive(
                        "box",
                        b=[
                            {
                                "type": "binop",
                                "op": "/",
                                "lhs": {"type": "param", "name": "length"},
                                "rhs": {"type": "num", "value": 2},
                            },
                            0.6,
                            0.6,
                        ],
                    ),
                    t=[
                        {
                            "type": "binop",
                            "op": "/",
                            "lhs": {"type": "param", "name": "length"},
                            "rhs": {"type": "num", "value": 2},
                        },
                        0,
                        0,
                    ],
                ),
            )
        ],
        ports=[Port("pivot"), Port("tip", Frame(position=({"$ref": "length"}, 0, 0)))],
    )
    for part in (base, carriage, arm):
        save(part, parts / f"{part.name}.sdm")

    def reference(path: str) -> PartRef:
        return PartRef(
            path, "sha256:" + hashlib.sha256((directory / path).read_bytes()).hexdigest()
        )

    nested = Assembly(
        "arm_module",
        params={"length": Param("length", 8, unit="mm")},
        instances=(
            Instance(
                "arm", reference("parts/arm.sdm"), {"length": {"$ref": "length"}}, transform=Frame()
            ),
        ),
        port={"pivot": "arm.pivot", "tip": "arm.tip"},
    )
    save(nested, directory / "arm_module.sdm")
    root = Assembly(
        "actuator",
        params={"length": Param("length", 8, free=True, bounds=(4, 15), unit="mm")},
        instances=(
            Instance("base", reference("parts/base.sdm"), transform=Frame()),
            Instance("carriage", reference("parts/carriage.sdm")),
            Instance("arm_module", reference("arm_module.sdm"), {"length": {"$ref": "length"}}),
        ),
        dofs={
            "stroke": Dof("length", (0, 10), "mm", 3),
            "turn": Dof("angle", (-180, 180), "deg", 30),
        },
        mates=(
            Mate("slide", "prismatic", "base.mount", "carriage.bottom", "stroke"),
            Mate("hinge", "revolute", "carriage.pivot", "arm_module.pivot", "turn"),
        ),
        port={"output": "arm_module.tip"},
    )
    path = directory / "assembly.sdm"
    save(root, path)
    compile_placement(load_bundle(path)).evaluate_checked()
    return path


def main(argv: list[str] | None = None) -> int:
    """Build the example and print its root path, reporting failures to stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        print(build_placement_bundle(args.directory))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
