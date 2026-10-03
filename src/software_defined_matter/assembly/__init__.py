"""Assembly declarations, reference resolution, and occurrence-scoped bindings."""

from __future__ import annotations

from software_defined_matter.assembly.bundle import (
    AssemblyBundle,
    ResolvedPort,
    ScopedConstraint,
    load_bundle,
)
from software_defined_matter.assembly.model import Assembly, Dof, Instance, Mate, PartRef
from software_defined_matter.assembly.placement import (
    PlacementError,
    PlacementEval,
    PlacementState,
    compile_placement,
)

__all__ = [
    "PlacementEval",
    "PlacementState",
    "PlacementError",
    "compile_placement",
    "Assembly",
    "AssemblyBundle",
    "Dof",
    "Instance",
    "Mate",
    "PartRef",
    "ResolvedPort",
    "ScopedConstraint",
    "load_bundle",
]
