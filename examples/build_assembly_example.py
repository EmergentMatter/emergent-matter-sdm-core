"""Build an actuator bundle to exercise nested definitions and independent occurrences.

Placement is explicitly grounded in this foundation example. Mates and trajectory
playback belong to the subsequent motion layer.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from software_defined_matter import (
    Assembly,
    Constraint,
    Frame,
    Instance,
    MaterialRegion,
    Param,
    Part,
    PartRef,
    Port,
    load_bundle,
    save,
    sdf_primitive,
)

__all__ = ["build_assembly_bundle", "main"]


def build_assembly_bundle(directory: Path) -> Path:
    """Write reusable parts and a nested assembly, then validate the complete bundle."""
    directory.mkdir(parents=True, exist_ok=True)
    definitions = directory / "parts"
    definitions.mkdir(exist_ok=True)
    shaft = Part(
        "shaft",
        params={
            "radius": Param("radius", 2.0, unit="mm"),
            "half_length": Param("half_length", 10.0, free=True, unit="mm"),
        },
        materials=[
            MaterialRegion(
                1,
                "shaft",
                sdf_primitive("capped_cylinder", r={"$ref": "radius"}, h={"$ref": "half_length"}),
            )
        ],
        ports=[Port("output", Frame(position=(0, 0, {"$ref": "half_length"})))],
        constraints=[
            Constraint("positive_length", {"type": "param", "name": "half_length"}, ">=", 1.0)
        ],
    )
    bolt = Part(
        "bolt",
        params={"radius": Param("radius", 1.0, free=True, unit="mm")},
        materials=[MaterialRegion(1, "bolt", sdf_primitive("sphere", r={"$ref": "radius"}))],
        ports=[Port("seat")],
    )
    housing = Part(
        "housing",
        materials=[MaterialRegion(1, "housing", sdf_primitive("box", b=[8, 8, 2]))],
        ports=[Port("mount", Frame(position=(0, 0, 2)))],
    )
    for document in (shaft, bolt, housing):
        save(document, definitions / f"{document.name}.sdm")

    def reference(path: str) -> PartRef:
        return PartRef(
            path, "sha256:" + hashlib.sha256((directory / path).read_bytes()).hexdigest()
        )

    rotor = Assembly(
        "rotor",
        params={"length": Param("length", 10.0, free=True, unit="mm")},
        instances=(
            Instance(
                "shaft",
                reference("parts/shaft.sdm"),
                {"half_length": {"$ref": "length"}},
                transform=Frame(),
            ),
        ),
        port={"output": "shaft.output"},
    )
    save(rotor, directory / "rotor.sdm")
    actuator = Assembly(
        "actuator",
        params={
            "length": Param("length", 12.0, free=True, bounds=(5, 20), unit="mm"),
            "bolt_radius": Param("bolt_radius", 1.0, free=True, bounds=(0.5, 2), unit="mm"),
        },
        instances=(
            Instance("housing", reference("parts/housing.sdm"), transform=Frame()),
            Instance(
                "rotor", reference("rotor.sdm"), {"length": {"$ref": "length"}}, transform=Frame()
            ),
            Instance(
                "bolt_left",
                reference("parts/bolt.sdm"),
                {"radius": {"$ref": "bolt_radius"}},
                transform=Frame(position=(-6, 0, 2)),
            ),
            Instance(
                "bolt_right",
                reference("parts/bolt.sdm"),
                {"radius": {"$ref": "bolt_radius"}},
                transform=Frame(position=(6, 0, 2)),
            ),
        ),
        port={"output": "rotor.output"},
    )
    output = directory / "assembly.sdm"
    save(actuator, output)
    load_bundle(output)
    return output


def main(argv: list[str] | None = None) -> int:
    """Write the example to the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    print(build_assembly_bundle(args.directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
