"""Human-readable names for assembly occurrences (docs/adr/0009).

The full name is the instance ids from the top assembly down, upper case with
hyphens: ``HAND-MECH-CRANK-DRIVEN-THUMB``. Each instance id names what it is,
most general word first, so the name narrows from the assembly to the detail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from software_defined_matter.assembly.model import Assembly


def full_name(path: Sequence[str]) -> str:
    """``("hand", "mech", "crank_driven_thumb")`` -> ``"HAND-MECH-CRANK-DRIVEN-THUMB"``."""
    return "-".join(part.upper().replace("_", "-") for part in path if part)


def naming_problems(assembly: Assembly, part_stems: Mapping[str, str]) -> list[str]:
    """Instances that do not start with their part's kind.

    ``part_stems`` maps an instance id to the stem of the part file it places
    (``screw_m2x6``); instances of sub-assemblies are left out of it, since
    their id is the assembly's short name. The kind is the stem's first word.
    """
    out: list[str] = []
    for inst in assembly.instances:
        stem = part_stems.get(inst.id)
        if not stem:
            continue
        kind = stem.split("_")[0]
        if inst.id != kind and not inst.id.startswith(kind + "_"):
            out.append(
                f"{inst.id!r} places {stem!r}: name it {kind}_... (what it is first, then the detail)"
            )
    return out
