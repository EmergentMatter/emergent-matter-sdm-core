"""How an assembly goes together: its build sequence.

An assembly's placement says where each instance ends up; nothing says how it
gets there. The build sequence does: steps in order, each installing some of
the assembly's own instances along a direction, with the tool and notes a
person building it needs. A sub-assembly carries its own sequence and is
installed into its parent as one unit, in a step of the parent.

It is stored in the assembly's free-form ``metadata`` under
``"assembly_process"``, so it needs no schema change and tools that do not
know it ignore it. :func:`with_process` writes it; :func:`process_of` reads
it back; :meth:`AssemblyProcess.problems` checks it against the assembly.

A part can be installed many ways in many contexts. Only the assembly that
places it knows which way applies, which is why the sequence belongs to the
assembly and not to the part.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

from software_defined_matter.assembly.model import Assembly
from software_defined_matter.ports import validate_identifier

__all__ = ["METADATA_KEY", "AssemblyProcess", "AssemblyStep", "Twist", "process_of", "with_process"]

#: Where the sequence lives in ``Assembly.metadata``.
METADATA_KEY = "assembly_process"
_VERSION = 1


def _unit3(v: Any, where: str) -> tuple[float, float, float]:
    if (
        not isinstance(v, (list, tuple))
        or len(v) != 3
        or not all(isinstance(x, (int, float)) for x in v)
    ):
        raise ValueError(f"{where}: expected three numbers, got {v!r}")
    n = math.sqrt(sum(float(x) ** 2 for x in v))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError(f"{where}: expected a nonzero finite vector, got {v!r}")
    return (float(v[0]) / n, float(v[1]) / n, float(v[2]) / n)


@dataclass(frozen=True)
class Twist:
    """A turn made while going in (screwing in, a bayonet, a twist-lock).

    ``axis`` is in the assembly's frame, through the installed parts' own
    origin; ``angle_deg`` is right-handed about it.
    """

    axis: tuple[float, float, float]
    angle_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", _unit3(self.axis, "Twist.axis"))
        if not isinstance(self.angle_deg, (int, float)) or not math.isfinite(self.angle_deg):
            raise ValueError(f"Twist.angle_deg: expected a finite number, got {self.angle_deg!r}")
        object.__setattr__(self, "angle_deg", float(self.angle_deg))

    def to_dict(self) -> dict[str, Any]:
        return {"axis": list(self.axis), "angle_deg": self.angle_deg}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Twist:
        return cls(tuple(data["axis"]), data["angle_deg"])


@dataclass(frozen=True)
class AssemblyStep:
    """One step: install ``instances`` by moving them along ``direction``.

    ``direction`` is the way the parts MOVE to reach their place, in the
    assembly's frame; they start ``travel`` mm back along it. It may be left
    out only for instances placed by a mate, whose child port's +Z (the axis a
    prismatic mate already slides on) is then used. ``travel`` left out means
    "about the parts' own size along the direction".
    """

    id: str
    name: str
    instances: tuple[str, ...]
    direction: tuple[float, float, float] | None = None
    travel: float | None = None
    rotate: Twist | None = None
    tool: str | None = None
    fasteners: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.id, "AssemblyStep.id")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"AssemblyStep {self.id!r}.name: expected a nonempty description")
        object.__setattr__(self, "instances", tuple(self.instances))
        object.__setattr__(self, "fasteners", tuple(self.fasteners))
        if not self.instances:
            raise ValueError(f"AssemblyStep {self.id!r}: installs nothing")
        for i in self.instances + self.fasteners:
            validate_identifier(i, f"AssemblyStep {self.id!r} instance")
        if set(self.fasteners) - set(self.instances):
            raise ValueError(f"AssemblyStep {self.id!r}.fasteners: must be among its instances")
        if self.direction is not None:
            object.__setattr__(
                self, "direction", _unit3(self.direction, f"AssemblyStep {self.id!r}.direction")
            )
        if self.travel is not None and not (
            isinstance(self.travel, (int, float)) and self.travel > 0
        ):
            raise ValueError(f"AssemblyStep {self.id!r}.travel: expected a positive length in mm")
        if self.rotate is not None and not isinstance(self.rotate, Twist):
            raise ValueError(f"AssemblyStep {self.id!r}.rotate: expected Twist")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "name": self.name, "instances": list(self.instances)}
        if self.direction is not None:
            out["direction"] = list(self.direction)
        if self.travel is not None:
            out["travel"] = float(self.travel)
        if self.rotate is not None:
            out["rotate"] = self.rotate.to_dict()
        for key in ("tool", "notes"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.fasteners:
            out["fasteners"] = list(self.fasteners)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssemblyStep:
        unknown = set(data) - {
            "id",
            "name",
            "instances",
            "direction",
            "travel",
            "rotate",
            "tool",
            "fasteners",
            "notes",
        }
        if unknown:
            raise ValueError(f"AssemblyStep: unknown keys {sorted(unknown)}")
        return cls(
            data["id"],
            data["name"],
            tuple(data["instances"]),
            tuple(data["direction"]) if data.get("direction") is not None else None,
            data.get("travel"),
            Twist.from_dict(data["rotate"]) if data.get("rotate") else None,
            data.get("tool"),
            tuple(data.get("fasteners") or ()),
            data.get("notes"),
        )


@dataclass(frozen=True)
class AssemblyProcess:
    """The build sequence of one assembly: ``base`` is in place, then ``steps`` in order."""

    steps: tuple[AssemblyStep, ...]
    base: tuple[str, ...] = ()
    version: int = _VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "base", tuple(self.base))
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("AssemblyProcess.steps: step ids must be unique")
        for b in self.base:
            validate_identifier(b, "AssemblyProcess.base")

    def problems(self, assembly: Assembly) -> list[str]:
        """What does not add up against ``assembly``; empty when it is complete.

        Every instance installed exactly once (or in ``base``), only instances
        that exist, and a direction for every step (stated, or taken from a
        mate).
        """
        out: list[str] = []
        known = [i.id for i in assembly.instances]
        mated = {m.child.split(".")[0] for m in assembly.mates}
        placed = dict.fromkeys(self.base, "base")
        out += [f"base names {b!r}, which is not an instance" for b in self.base if b not in known]
        for step in self.steps:
            for i in step.instances:
                if i not in known:
                    out.append(f"step {step.id!r} installs {i!r}, which is not an instance")
                elif i in placed:
                    out.append(
                        f"step {step.id!r} installs {i!r}, already installed by {placed[i]!r}"
                    )
                else:
                    placed[i] = step.id
            if step.direction is None and not set(step.instances) <= mated:
                out.append(f"step {step.id!r} has no direction and is not all mated")
        out += [f"{i!r} is never installed and is not in base" for i in known if i not in placed]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "base": list(self.base),
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssemblyProcess:
        if data.get("version", _VERSION) != _VERSION:
            raise ValueError(
                f"AssemblyProcess.version: this reader understands {_VERSION}, "
                f"got {data.get('version')!r}"
            )
        return cls(
            tuple(AssemblyStep.from_dict(s) for s in data.get("steps") or ()),
            tuple(data.get("base") or ()),
        )


def with_process(assembly: Assembly, process: AssemblyProcess, *, b_check: bool = True) -> Assembly:
    """A copy of ``assembly`` carrying ``process`` in its metadata.

    Refuses a sequence with problems unless ``b_check`` is False: a wrong
    build sequence does not crash anything, it misleads the person building.
    """
    if b_check:
        found = process.problems(assembly)
        if found:
            raise ValueError(f"Assembly {assembly.name!r} build sequence: " + "; ".join(found))
    return dataclasses.replace(
        assembly, metadata={**assembly.metadata, METADATA_KEY: process.to_dict()}
    )


def process_of(assembly: Assembly) -> AssemblyProcess | None:
    """The assembly's build sequence, or None if it has none."""
    data = assembly.metadata.get(METADATA_KEY)
    return AssemblyProcess.from_dict(data) if data is not None else None
