"""emergent-matter: ``.sdm`` format core.

Materials data lives in the separate ``emergent_matter_materials`` package
(see https://github.com/EmergentMatter/emergent-matter-sdm-materials). Import
it directly::

    from emergent_matter_materials import get, get_material, MATERIALS
"""

from __future__ import annotations

from typing import Any

from software_defined_matter.assembly import (
    Assembly,
    AssemblyBundle,
    Dof,
    Instance,
    Mate,
    PartRef,
    PlacementError,
    PlacementEval,
    PlacementState,
    compile_placement,
    load_bundle,
)
from software_defined_matter.io import load, load_part, load_schema, save, validate
from software_defined_matter.model import (
    KNOWN_SCHEMA_VERSIONS,
    LATEST_SCHEMA_VERSION,
    PRIOR_DISTS,
    Constraint,
    MaterialRegion,
    Objective,
    Param,
    Part,
    Port,
    assert_supported,
    field_op,
    field_primitive,
    make_param_ref,
    min_schema_version_for,
    sdf_2d_to_3d,
    sdf_deform,
    sdf_helix,
    sdf_loft,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_raster_field,
    sdf_screw_thread,
    sdf_sweep,
    sdf_transform,
    sdf_vsweep,
    supports_version,
)
from software_defined_matter.ports import Frame
from software_defined_matter.process import effective_prior, resolve_process_profile

# JAX-importing helpers, not re-exported here (the package root stays light):
#   from software_defined_matter.grid_sampling import bind_sdf, make_grid, eval_chunked
#   from software_defined_matter.sample import sample_free_params, propagate
#
# audit.py is JAX-importing (gap measurement runs through JAX closures), but
# its functions are part of the public root API. Resolving them lazily here
# instead of at import time keeps `import software_defined_matter` -- and
# therefore its `schema` subpackage -- jax-free.
_LAZY_AUDIT_EXPORTS = frozenset(
    {"GapAuditError", "GapCheck", "assert_gaps", "gap_audit", "measure_gap"}
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_AUDIT_EXPORTS:
        from software_defined_matter import audit

        value = getattr(audit, name)
        globals()[name] = value  # cache: __getattr__ runs once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep the lazily-resolved audit names visible in ``dir()``, matching ``__all__``."""
    return sorted(set(globals()) | _LAZY_AUDIT_EXPORTS)


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
    "load_bundle",
    "KNOWN_SCHEMA_VERSIONS",
    "LATEST_SCHEMA_VERSION",
    "PRIOR_DISTS",
    "Constraint",
    "Port",
    "Frame",
    "GapAuditError",
    "GapCheck",
    "MaterialRegion",
    "Objective",
    "Param",
    "Part",
    "assert_gaps",
    "assert_supported",
    "effective_prior",
    "field_op",
    "field_primitive",
    "gap_audit",
    "load",
    "load_part",
    "load_schema",
    "make_param_ref",
    "measure_gap",
    "min_schema_version_for",
    "resolve_process_profile",
    "save",
    "sdf_2d_to_3d",
    "sdf_deform",
    "sdf_helix",
    "sdf_screw_thread",
    "sdf_loft",
    "sdf_modifier",
    "sdf_op",
    "sdf_primitive",
    "sdf_raster_field",
    "sdf_sweep",
    "sdf_vsweep",
    "sdf_transform",
    "supports_version",
    "validate",
]
