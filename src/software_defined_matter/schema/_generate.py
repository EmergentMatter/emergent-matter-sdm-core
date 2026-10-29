"""Generate the current ``.sdm`` JSON Schema and version index from ``wire.py``.

Only the newest, unreleased schema version is a valid generation target
(``sdm-0.1.schema.json`` / ``sdm-0.2.schema.json`` are frozen; see
``docs/adr/0001-sdm-wire-contract.md``). This module never touches them.

Most of the document contract (``Part`` / ``Param`` / ``MaterialRegion`` /
``CouplingNode`` / ``Objective`` / ``Constraint``, and the SDF/field node
vocabulary) is derived from :mod:`software_defined_matter.wire`. So is
``expr``: the contract declares which expression node types exist
(``wire.EXPR_NODE_TYPES``, the same set ``sdf/validate.py`` gates an
expression nested in an SDF kwarg against), and only each branch's interior
is written here, because operator vocabularies belong to
:mod:`software_defined_matter.dsl.expr`.

A few ``$defs`` have no counterpart in the contract at all, because they
describe things outside its stated scope -- ``kinematics`` and its nested
defs (no ``Part`` field exists for it, see ``wire.PART_FIELDS``), and
``Param``'s ``ui`` block (free-form, consumed by viewers, never validated
beyond "object"). Those stay as static literals below.

Run directly (``python -m software_defined_matter.schema._generate``) to
regenerate the committed files in place; CI fails if that changes anything
(see ``.github/workflows/ci.yml``, job ``schema``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from software_defined_matter import wire
from software_defined_matter.model import PRIOR_DISTS

__all__ = ["generate_index", "generate_schema", "write_generated_files"]

_SCHEMA_DIR = Path(__file__).parent

#: The eventual open-source $id host, sourced from one place so the fix
#: deferred to the release checklist (see the ADR's Known Limitations) is a
#: one-line change here instead of an edit per generated file.
_ID_HOST = "https://emergent-matter.example"

#: The sole generation target: the newest version in wire.py's declared
#: ordering. Every version before it is a frozen, hand-authored artifact.
_GENERATED_VERSION = wire.SCHEMA_VERSIONS[-1]


# ===========================================================================
# SDF/field kwarg shapes, derived from wire.ParamSpec
# ===========================================================================


def _value_schema(pspec: wire.ParamSpec) -> dict[str, Any]:
    """The JSON Schema for one SDF/field node kwarg's value. Number-shaped
    values may also be an ``{"$along": [...]}`` leaf, one value per path
    vertex (meaningful only inside a ``vsweep`` profile; the semantic
    validator enforces where)."""
    base = _plain_value_schema(pspec)
    if pspec.wire_shape in ("string", "bool", "object"):
        return base
    along = {
        "type": "object",
        "additionalProperties": False,
        "required": ["$along"],
        "properties": {"$along": {"type": "array", "minItems": 2, "items": base}},
    }
    return {"oneOf": [base, along]}


def _plain_value_schema(pspec: wire.ParamSpec) -> dict[str, Any]:
    shape = pspec.wire_shape
    if shape == "scalar":
        return {"$ref": "#/$defs/scalarValue"}
    if shape == "axis_vector":
        # translate.t is 2-D in a 2-D subtree, 3-D in 3-D -- tf_translate
        # takes either, so the wire contract must too
        return {
            "type": "array",
            "items": {"$ref": "#/$defs/scalarValue"},
            "minItems": 2,
            "maxItems": 3,
        }
    if shape in ("vec2", "vec3"):
        n = 2 if shape == "vec2" else 3
        return {
            "type": "array",
            "items": {"$ref": "#/$defs/scalarValue"},
            "minItems": n,
            "maxItems": n,
        }
    if shape == "mat3":
        row = {
            "type": "array",
            "items": {"$ref": "#/$defs/scalarValue"},
            "minItems": 3,
            "maxItems": 3,
        }
        return {"type": "array", "items": row, "minItems": 3, "maxItems": 3}
    if shape == "scalar_list":
        return {"type": "array", "items": {"$ref": "#/$defs/scalarValue"}}
    if shape == "point_list":
        point: dict[str, Any] = {"type": "array", "items": {"$ref": "#/$defs/scalarValue"}}
        if pspec.point_dim is not None:
            point["minItems"] = pspec.point_dim
            point["maxItems"] = pspec.point_dim
        out: dict[str, Any] = {"type": "array", "items": point}
        if pspec.min_points is not None:
            out["minItems"] = pspec.min_points
        # multiple_of (bezier_2d's 3K control points) has no native JSON
        # Schema array-length keyword; sdf/validate.py enforces it.
        return out
    if shape == "string":
        return (
            {"enum": list(pspec.choices)} if pspec.choices else {"type": "string", "minLength": 1}
        )
    if shape == "bool":
        return {"type": "boolean"}
    if shape == "object":
        return {"type": "object"}
    raise AssertionError(f"ParamSpec {pspec.name!r}: wire_shape {shape!r} has no SDF kwarg schema")


def _params_schema(spec: wire.NodeSpec) -> dict[str, Any]:
    """The JSON Schema for one node kind's whole ``params`` object."""
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {p.name: _value_schema(p) for p in spec.params},
    }
    required = spec.required_param_names()
    if required:
        schema["required"] = list(required)
    return schema


def _name_constraints(
    node_type_const: str, key: str, registry: dict[str, wire.NodeSpec]
) -> list[dict[str, Any]]:
    """One ``if``/``then`` block per registry entry, keyed on ``key``'s literal
    value. ``if``/``then`` rather than a nested ``oneOf``: nested inside the
    existing ``$defs/sdf`` ``oneOf``, a second ``oneOf`` would only ever
    report "matches 0 of N subschemas", while ``if``/``then`` names the
    offending field directly.
    """
    blocks = []
    for name, spec in sorted(registry.items()):
        then: dict[str, Any] = {"properties": {"params": _params_schema(spec)}}
        if spec.requires_field:
            then["required"] = ["field"]
        blocks.append(
            {
                "if": {
                    "properties": {"type": {"const": node_type_const}, key: {"const": name}},
                    "required": ["type", key],
                },
                "then": then,
            }
        )
    return blocks


def _discriminator_enum_constraint(
    node_type_const: str, key: str, registry: dict[str, wire.NodeSpec]
) -> dict[str, Any]:
    """One ``if``/``then`` block rejecting an unknown ``key`` value (a typo'd
    ``kind`` / ``op`` / ... name) for the whole node type, independent of
    which specific name it should have been.

    Kept out of the ``oneOf`` branch's own ``properties`` deliberately: an
    ``enum`` there would make a typo fail that branch's structural match
    entirely, so ``oneOf`` would report "matches 0 of N" and, combined with
    the sibling ``unevaluatedProperties: false``, an unhelpful "unevaluated
    properties" error instead of naming ``key``. Checking the enum here, on
    an ``if`` keyed only on ``type`` (matched once the branch already won),
    reports the actual offending value against ``key`` directly.
    """
    return {
        "if": {"properties": {"type": {"const": node_type_const}}, "required": ["type"]},
        "then": {"properties": {key: {"enum": sorted(registry)}}},
    }


# ===========================================================================
# $defs.sdf / $defs.field
# ===========================================================================


def _sdf_def() -> dict[str, Any]:
    branches = [
        {
            "properties": {
                "type": {"const": "primitive"},
                "kind": {"type": "string", "minLength": 1},
                "params": {"type": "object"},
            },
            "required": ["kind"],
        },
        {
            "properties": {
                "type": {"const": "op"},
                "op": {"type": "string", "minLength": 1},
                "children": {"type": "array", "items": {"$ref": "#/$defs/sdf"}, "minItems": 1},
                "params": {"type": "object"},
            },
            "required": ["op", "children"],
        },
        {
            "properties": {
                "type": {"const": "transform"},
                "transform": {"type": "string", "minLength": 1},
                "child": {"$ref": "#/$defs/sdf"},
                "params": {"type": "object"},
            },
            "required": ["transform", "child"],
        },
        {
            "properties": {
                "type": {"const": "modifier"},
                "modifier": {"type": "string", "minLength": 1},
                "child": {"$ref": "#/$defs/sdf"},
                "params": {"type": "object"},
            },
            "required": ["modifier", "child"],
        },
        {
            "properties": {
                "type": {"const": "deform"},
                "deform": {"type": "string", "minLength": 1},
                "child": {"$ref": "#/$defs/sdf"},
                "field": {"$ref": "#/$defs/field"},
                "params": {"type": "object"},
            },
            "required": ["deform", "child"],
        },
        {
            "properties": {
                "type": {"const": "2d_to_3d"},
                "method": {"type": "string", "minLength": 1},
                "child": {"$ref": "#/$defs/sdf"},
                "params": {"type": "object"},
            },
            "required": ["method", "child"],
        },
        {
            "properties": {
                "type": {"const": "sweep"},
                "child": {"$ref": "#/$defs/sdf"},
                "params": _params_schema(wire.SWEEP),
            },
            "required": ["child", "params"],
        },
        {
            "properties": {
                "type": {"const": "loft"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/sdf"}, "minItems": 2},
                "params": _params_schema(wire.LOFT),
            },
            "required": ["children", "params"],
        },
        {
            "properties": {
                "type": {"const": "vsweep"},
                "child": {"$ref": "#/$defs/sdf"},
                "params": _params_schema(wire.VSWEEP),
            },
            "required": ["child", "params"],
        },
    ]
    all_of = (
        [
            _discriminator_enum_constraint("primitive", "kind", wire.PRIMITIVES),
            _discriminator_enum_constraint("op", "op", wire.OPS),
            _discriminator_enum_constraint("transform", "transform", wire.TRANSFORMS),
            _discriminator_enum_constraint("modifier", "modifier", wire.MODIFIERS),
            _discriminator_enum_constraint("deform", "deform", wire.DEFORMS),
            _discriminator_enum_constraint("2d_to_3d", "method", wire.TWO_D_TO_3D),
        ]
        + _name_constraints("primitive", "kind", wire.PRIMITIVES)
        + _name_constraints("op", "op", wire.OPS)
        + _name_constraints("transform", "transform", wire.TRANSFORMS)
        + _name_constraints("modifier", "modifier", wire.MODIFIERS)
        + _name_constraints("deform", "deform", wire.DEFORMS)
        + _name_constraints("2d_to_3d", "method", wire.TWO_D_TO_3D)
    )
    return {
        "type": "object",
        "required": ["type"],
        "unevaluatedProperties": False,
        "oneOf": branches,
        "allOf": all_of,
        "properties": {"name": {"type": "string", "minLength": 1}},
    }


def _field_def() -> dict[str, Any]:
    branches = [
        {
            "properties": {
                "type": {"const": "field"},
                "kind": {"type": "string", "minLength": 1},
                "params": {"type": "object"},
            },
            "required": ["kind"],
        },
        {
            "properties": {
                "type": {"const": "field_op"},
                "op": {"type": "string", "minLength": 1},
                "children": {"type": "array", "items": {"$ref": "#/$defs/field"}, "minItems": 1},
                "params": {"type": "object"},
            },
            "required": ["op", "children"],
        },
    ]
    all_of = (
        [
            _discriminator_enum_constraint("field", "kind", wire.FIELD_PRIMITIVES),
            _discriminator_enum_constraint("field_op", "op", wire.FIELD_OPS),
        ]
        + _name_constraints("field", "kind", wire.FIELD_PRIMITIVES)
        + _name_constraints("field_op", "op", wire.FIELD_OPS)
    )
    return {
        "type": "object",
        "required": ["type"],
        "unevaluatedProperties": False,
        "oneOf": branches,
        "allOf": all_of,
    }


# ===========================================================================
# Document structure ($defs.param / .material / .coupling / .objective /
# .constraint, and Part's own top-level properties), derived from
# wire.DOCUMENT_FIELDS
# ===========================================================================


def _document_value_schema(dataclass_name: str, fspec: wire.FieldSpec) -> dict[str, Any]:
    """The JSON Schema for a document field with no entry in
    _DOCUMENT_FIELD_OVERRIDES below: shapes plain enough to need no further
    refinement. A ``$ref`` target, an enum, or a numeric bound always needs
    an explicit override instead, so an unhandled shape raises here rather
    than emitting a silently under-constrained schema.
    """
    shape = fspec.wire_shape
    if shape == "string":
        return {"type": "string", "minLength": 1}
    if shape == "scalar":
        return {"type": "number"}
    if shape == "bool":
        return {"type": "boolean"}
    if shape == "object":
        return {"type": "object"}
    raise AssertionError(
        f"{dataclass_name}.{fspec.name}: wire_shape {shape!r} has no generic document mapping; "
        "add an entry to _DOCUMENT_FIELD_OVERRIDES."
    )


#: Document fields whose schema is more than the generic wire_shape mapping
#: can express -- a $ref to another $defs entry, an enum sourced from a wire
#: vocabulary, or a numeric refinement (exclusiveMinimum) that wire.py's
#: FieldSpec has no column for.
_DOCUMENT_FIELD_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("Part", "params"): {"type": "object", "additionalProperties": {"$ref": "#/$defs/param"}},
    ("Part", "materials"): {"type": "array", "items": {"$ref": "#/$defs/material"}},
    ("Part", "ports"): {"type": "array", "items": {"$ref": "#/$defs/port"}},
    ("Part", "objectives"): {"type": "array", "items": {"$ref": "#/$defs/objective"}},
    ("Part", "constraints"): {"type": "array", "items": {"$ref": "#/$defs/constraint"}},
    ("Part", "history"): {"type": "array", "items": {"type": "object"}},
    ("Param", "bounds"): {
        "oneOf": [
            {"type": "null"},
            {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        ]
    },
    ("Param", "unit"): {"enum": sorted(wire.UNIT_NAMES)},
    ("Param", "ui"): {"$ref": "#/$defs/paramUi"},
    ("Param", "prior"): {"$ref": "#/$defs/paramPrior"},
    ("Param", "tolerance"): {"type": "number", "exclusiveMinimum": 0},
    ("Param", "expr"): {"$ref": "#/$defs/expr"},
    ("MaterialRegion", "material_id"): {"type": "integer"},
    ("MaterialRegion", "sdf_tree"): {"$ref": "#/$defs/sdf"},
    ("Port", "name"): {"type": "string", "pattern": "^[a-z_][a-z0-9_]*$"},
    ("Port", "frame"): {"$ref": "#/$defs/frame"},
    ("Port", "body"): {"type": ["string", "null"]},
    ("Port", "sdf_tree"): {"$ref": "#/$defs/sdfOrNull"},
    ("Objective", "sense"): {"enum": list(wire.OBJECTIVE_SENSES)},
    ("Objective", "expr"): {"$ref": "#/$defs/expr"},
    ("Constraint", "expr"): {"$ref": "#/$defs/expr"},
    ("Constraint", "op"): {"enum": list(wire.CONSTRAINT_OPS)},
}


def _properties_for(dataclass_name: str, field_specs: tuple[wire.FieldSpec, ...]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for fspec in field_specs:
        override = _DOCUMENT_FIELD_OVERRIDES.get((dataclass_name, fspec.name))
        properties[fspec.name] = (
            override if override is not None else _document_value_schema(dataclass_name, fspec)
        )
    return properties


def _document_def(dataclass_name: str, field_specs: tuple[wire.FieldSpec, ...]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": _properties_for(dataclass_name, field_specs),
    }
    required = [f.name for f in field_specs if f.required]
    if required:
        schema["required"] = required
    return schema


def _param_def() -> dict[str, Any]:
    schema = _document_def("Param", wire.PARAM_FIELDS)
    schema["not"] = {"required": ["prior", "tolerance"]}
    return schema


def _prior_def() -> dict[str, Any]:
    """Built from ``model.PRIOR_DISTS`` (dist name -> its extra required
    keys). The numeric refinement on each extra key (sigma / half_width must
    be positive) has no column in ``PRIOR_DISTS`` and is authored here.
    """
    numeric_constraint = {
        "lo": {"type": "number"},
        "hi": {"type": "number"},
        "sigma": {"type": "number", "exclusiveMinimum": 0},
        "half_width": {"type": "number", "exclusiveMinimum": 0},
    }
    branches = []
    for dist, extra_keys in PRIOR_DISTS.items():
        properties: dict[str, Any] = {"dist": {"const": dist}}
        for key in extra_keys:
            properties[key] = numeric_constraint[key]
        branches.append(
            {
                "type": "object",
                "required": ["dist", *extra_keys],
                "additionalProperties": False,
                "properties": properties,
            }
        )
    return {"oneOf": branches}


# ===========================================================================
# Static $defs: outside wire.py's stated scope (see the module docstring).
# ===========================================================================

_PARAM_UI_DEF: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "step": {"type": "number", "exclusiveMinimum": 0},
        "explore_bounds": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "group": {"type": "string"},
        "order": {"type": "number"},
        "role": {"enum": ["topology", "pose"]},
        "rebuild": {"type": "boolean"},
        "axis": {"enum": ["radial", "axial", "tangential"]},
        "collapsed": {"type": "boolean"},
        "driven": {"type": "boolean"},
        "choices": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 2},
    },
}

_VEC3_DEF: dict[str, Any] = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
}

_NODE_REF_DEF: dict[str, Any] = {
    "type": "object",
    "required": ["$node"],
    "additionalProperties": False,
    "properties": {"$node": {"type": "string", "minLength": 1}},
}

_PARAM_REF_DEF: dict[str, Any] = {
    "type": "object",
    "required": ["$ref"],
    "additionalProperties": False,
    "properties": {"$ref": {"type": "string", "minLength": 1}},
}

#: An SDF/field kwarg's scalar leaf: a literal number, a {"$ref": name}
#: pointing at a document param, or an expression subtree -- the three
#: shapes dsl.resolve.resolve_param_value accepts (see wire.py's module
#: docstring). Every "scalar" wire_shape, and every element of a vec2/vec3/
#: mat3/point_list/scalar_list, resolves through this.
_SCALAR_VALUE_DEF: dict[str, Any] = {
    "oneOf": [
        {"type": "number"},
        {"$ref": "#/$defs/paramRef"},
        {"$ref": "#/$defs/expr"},
    ]
}

#: The body of one expression branch, keyed by its ``type`` const. The key
#: set is checked against ``wire.EXPR_NODE_TYPES`` when the def is built, so
#: this table cannot gain or lose a node type without the contract saying so.
#:
#: The bodies stay here rather than in ``wire.py``. Which node types exist is
#: the wire contract's business, and ``sdf/validate.py`` gates kwargs against
#: it. What an ``unop``'s ``op`` may be, and which operands a node carries,
#: is ``dsl/expr.py``'s vocabulary, which the contract deliberately does not
#: describe.
_EXPR_BRANCH_BODIES: dict[str, dict[str, Any]] = {
    "num": {
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
    },
    "param": {
        "properties": {"name": {"type": "string", "minLength": 1}},
        "required": ["name"],
    },
    "metric": {
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "args": {"type": "object"},
        },
        "required": ["name"],
    },
    "unop": {
        "properties": {
            "op": {"enum": ["neg", "abs", "sqrt", "log", "exp", "square", "sin", "cos"]},
            "child": {"$ref": "#/$defs/expr"},
        },
        "required": ["op", "child"],
    },
    "binop": {
        "properties": {
            "op": {"enum": ["+", "-", "*", "/", "pow", "min", "max"]},
            "lhs": {"$ref": "#/$defs/expr"},
            "rhs": {"$ref": "#/$defs/expr"},
        },
        "required": ["op", "lhs", "rhs"],
    },
    "reduce": {
        "properties": {
            "op": {"enum": ["sum", "mean", "min", "max"]},
            "children": {"type": "array", "items": {"$ref": "#/$defs/expr"}, "minItems": 1},
        },
        "required": ["op", "children"],
    },
    "dof": {
        "properties": {"name": {"type": "string", "minLength": 1}},
        "required": ["name"],
    },
}


def _expr_def() -> dict[str, Any]:
    """Build ``$defs/expr`` with one branch per ``wire.EXPR_NODE_TYPES`` entry.

    Sourced from the contract rather than hand-listed, so the generated
    schema and the gate ``sdf/validate.py`` applies to an expression nested
    in an SDF kwarg cannot disagree about which node types exist. Branch
    order follows the contract's own order.
    """
    declared = set(wire.EXPR_NODE_TYPES)
    described = set(_EXPR_BRANCH_BODIES)
    if declared != described:
        raise ValueError(
            "expression node types disagree: wire.EXPR_NODE_TYPES has "
            f"{sorted(declared - described)} with no branch body here, and this "
            f"table has {sorted(described - declared)} the contract does not declare."
        )
    return {
        "type": "object",
        "required": ["type"],
        "unevaluatedProperties": False,
        "oneOf": [
            {
                "properties": {
                    "type": {"const": node_type},
                    **_EXPR_BRANCH_BODIES[node_type]["properties"],
                },
                "required": _EXPR_BRANCH_BODIES[node_type]["required"],
            }
            for node_type in wire.EXPR_NODE_TYPES
        ],
    }


_DOF_DEF: dict[str, Any] = {
    "description": (
        "An animation degree of freedom. Distinct from optimization `params`; "
        'referenced in motion exprs via {"type":"dof"}. `range` is required so '
        "viewers can auto-build a per-DOF scrub animation with zero authoring; "
        "`rate` is the nominal sweep rate in unit/s (viewers default to range/4s "
        "when absent)."
    ),
    "type": "object",
    "required": ["name", "kind", "range", "unit"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "kind": {"enum": ["angle", "length"]},
        "range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "default": {"type": "number"},
        "rate": {"type": "number", "exclusiveMinimum": 0},
        "unit": {"enum": ["rad", "deg", "mm"]},
    },
}

_BODY_DEF: dict[str, Any] = {
    "description": (
        "A rigid region. `region` SDF defines membership; `motion` is a "
        "DOF-parameterized rigid transform (empty ops => fixed/ground)."
    ),
    "type": "object",
    "required": ["name", "region", "motion"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "region": {"$ref": "#/$defs/region"},
        "motion": {"$ref": "#/$defs/motion"},
    },
}

_FLEXURE_DEF: dict[str, Any] = {
    "description": (
        "A compliant region whose transform interpolates the motions of two "
        "bodies. `blend` is a scalar field in [0,1]: 0 -> from_body, 1 -> "
        "to_body. Interpolation is in joint-coordinate space (e.g. angle lerp "
        "for revolute bodies on a shared axis; screw interp in general)."
    ),
    "type": "object",
    "required": ["name", "region", "from_body", "to_body", "blend"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "region": {"$ref": "#/$defs/region"},
        "from_body": {"type": "string", "minLength": 1},
        "to_body": {"type": "string", "minLength": 1},
        "blend": {
            "oneOf": [
                {"$ref": "#/$defs/field"},
                {
                    "type": "object",
                    "required": ["type", "kind", "params"],
                    "additionalProperties": False,
                    "properties": {
                        "type": {"const": "field"},
                        "kind": {"const": "axis_ramp"},
                        "params": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["axis", "lo", "hi"],
                            "properties": {
                                "axis": {"$ref": "#/$defs/vec3"},
                                "lo": {"type": "number"},
                                "hi": {"type": "number"},
                            },
                        },
                    },
                },
            ]
        },
    },
}

_FLEXURE_DEF["properties"]["blend"]["oneOf"].append(
    {
        "type": "object",
        "required": ["type", "kind", "params"],
        "additionalProperties": False,
        "properties": {
            "type": {"const": "field"},
            "kind": {"const": "radial_hermite"},
            "params": {
                "type": "object",
                "additionalProperties": False,
                "required": ["axis", "origin", "r0", "r1"],
                "properties": {
                    "axis": {"$ref": "#/$defs/vec3"},
                    "origin": {"$ref": "#/$defs/vec3"},
                    "r0": {"type": "number", "minimum": 0},
                    "r1": {"type": "number", "exclusiveMinimum": 0},
                },
            },
        },
    }
)


_MOTION_DEF: dict[str, Any] = {
    "description": (
        "An ordered list of DOF-parameterized rigid ops, composed left-to-right. Empty => identity."
    ),
    "type": "object",
    "required": ["ops"],
    "additionalProperties": False,
    "properties": {"ops": {"type": "array", "items": {"$ref": "#/$defs/motionOp"}}},
}

_MOTION_OP_DEF: dict[str, Any] = {
    "type": "object",
    "required": ["kind", "axis"],
    "oneOf": [
        {
            "properties": {
                "kind": {"const": "rotate"},
                "axis": {"$ref": "#/$defs/vec3"},
                "origin": {"$ref": "#/$defs/vec3"},
                "angle": {"$ref": "#/$defs/expr"},
            },
            "required": ["angle"],
        },
        {
            "properties": {
                "kind": {"const": "translate"},
                "axis": {"$ref": "#/$defs/vec3"},
                "distance": {"$ref": "#/$defs/expr"},
            },
            "required": ["distance"],
        },
    ],
}

_KINEMATICS_DEF: dict[str, Any] = {
    "description": (
        "Declarative motion model. A point is assigned to the body or flexure "
        "whose `region` SDF is smallest (nearest-region ownership). A body "
        "moves rigidly by its DOF-parameterized `motion`; a flexure "
        "interpolates the motions of `from_body` and `to_body` by a spatial "
        "`blend` field in [0,1]. Vanishes to identity when all DOFs are 0."
    ),
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dofs": {"type": "array", "items": {"$ref": "#/$defs/dof"}},
        "bodies": {"type": "array", "items": {"$ref": "#/$defs/body"}},
        "flexures": {"type": "array", "items": {"$ref": "#/$defs/flexure"}},
    },
}


# ===========================================================================
# Top-level assembly
# ===========================================================================


def _assembly_defs() -> dict[str, Any]:
    """Structural assembly vocabulary; cross-file checks live in the bundle loader."""

    def record(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def mapping(value: dict[str, Any]) -> dict[str, Any]:
        return {"type": "object", "additionalProperties": value}

    def array(value: dict[str, Any]) -> dict[str, Any]:
        return {"type": "array", "items": value}

    name = {"type": "string", "pattern": "^[a-z_][a-z0-9_]*$"}
    scalar: dict[str, Any] = {"$ref": "#/$defs/frameScalar"}

    def vector(size: int) -> dict[str, Any]:
        return {"type": "array", "minItems": size, "maxItems": size, "items": scalar}

    import copy

    frame_expr = copy.deepcopy(_expr_def())
    frame_expr["oneOf"] = [
        branch
        for branch in frame_expr["oneOf"]
        if branch["properties"]["type"]["const"] not in {"metric", "dof"}
    ]

    def recursive(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "#/$defs/frameScalar"
                if key == "$ref" and item == "#/$defs/expr"
                else recursive(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [recursive(item) for item in value]
        return value

    return {
        "frameScalar": {
            "oneOf": [
                {"type": "number"},
                {"$ref": "#/$defs/paramRef"},
                {"$ref": "#/$defs/frameExpr"},
            ]
        },
        "frameExpr": recursive(frame_expr),
        "frame": record(
            {"position": vector(3), "orientation": vector(4)}, ["position", "orientation"]
        ),
        "partRef": record(
            {
                "path": {"type": "string", "minLength": 1},
                "content_hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            },
            ["path"],
        ),
        "instance": record(
            {
                "id": name,
                "part_ref": {"$ref": "#/$defs/partRef"},
                "param_overrides": mapping(
                    {"oneOf": [{"type": "number"}, {"$ref": "#/$defs/paramRef"}]}
                ),
                "dof_bindings": mapping({"oneOf": [{"type": "number"}, {"$ref": "#/$defs/expr"}]}),
                "transform": {"$ref": "#/$defs/frame"},
            },
            ["id", "part_ref"],
        ),
        "assemblyDof": record(
            {
                "kind": {"enum": ["angle", "length"]},
                "range": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
                "unit": {"enum": ["rad", "deg", "mm"]},
                "default": {"type": "number"},
            },
            ["kind", "range", "unit"],
        ),
        "mate": record(
            {
                "id": name,
                "kind": {"enum": ["fixed", "revolute", "prismatic"]},
                "parent": {"type": "string", "minLength": 1},
                "child": {"type": "string", "minLength": 1},
                "dof": {"type": "string", "minLength": 1},
                "offset": {"$ref": "#/$defs/frame"},
            },
            ["id", "kind", "parent", "child"],
        ),
        "assembly": record(
            {
                "schema_version": {"const": _GENERATED_VERSION},
                "kind": {"const": "assembly"},
                "name": {"type": "string", "minLength": 1},
                "params": mapping({"$ref": "#/$defs/param"}),
                "instances": array({"$ref": "#/$defs/instance"}),
                "mates": array({"$ref": "#/$defs/mate"}),
                "port": mapping({"type": "string"}),
                "dofs": mapping({"$ref": "#/$defs/assemblyDof"}),
                "motion_inputs": mapping({"type": "string"}),
                "objectives": array({"$ref": "#/$defs/objective"}),
                "constraints": array({"$ref": "#/$defs/constraint"}),
                "metadata": {"type": "object"},
            },
            ["schema_version", "kind", "name"],
        ),
    }


def generate_schema() -> dict[str, Any]:
    """Build the current ``.sdm`` JSON Schema from ``wire.py``."""
    version = _GENERATED_VERSION
    added = _added_since(version)
    title = f"SDM Part ({version})" if not added else f"SDM Part ({version}: + {', '.join(added)})"

    part_properties: dict[str, Any] = {"schema_version": {"const": version}}
    part_properties.update(_properties_for("Part", wire.PART_FIELDS))
    # Kinematics has a nested vocabulary beyond the generic object field.
    part_properties["kinematics"] = {"$ref": "#/$defs/kinematics"}
    part_required = ["schema_version", *(f.name for f in wire.PART_FIELDS if f.required)]
    part_properties["kind"] = {"const": "part"}

    result: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{_ID_HOST}/schemas/sdm-{version}.schema.json",
        "title": title,
        "description": "Source-of-truth file format for a part or an assembly.",
        "type": "object",
        "required": part_required,
        "additionalProperties": False,
        "properties": part_properties,
        "$defs": {
            "param": _param_def(),
            "paramUi": _PARAM_UI_DEF,
            "paramPrior": _prior_def(),
            "material": _document_def("MaterialRegion", wire.MATERIAL_REGION_FIELDS),
            "port": _document_def("Port", wire.PORT_FIELDS),
            **_assembly_defs(),
            "objective": _document_def("Objective", wire.OBJECTIVE_FIELDS),
            "constraint": _document_def("Constraint", wire.CONSTRAINT_FIELDS),
            "vec3": _VEC3_DEF,
            "sdfOrNull": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/sdf"}]},
            "sdf": _sdf_def(),
            "field": _field_def(),
            "expr": _expr_def(),
            "nodeRef": _NODE_REF_DEF,
            "paramRef": _PARAM_REF_DEF,
            "scalarValue": _SCALAR_VALUE_DEF,
            "region": {"oneOf": [{"$ref": "#/$defs/nodeRef"}, {"$ref": "#/$defs/sdf"}]},
            "kinematics": _KINEMATICS_DEF,
            "dof": _DOF_DEF,
            "body": _BODY_DEF,
            "flexure": _FLEXURE_DEF,
            "motion": _MOTION_DEF,
            "motionOp": _MOTION_OP_DEF,
        },
    }
    part_def = {
        key: result.pop(key) for key in ("type", "required", "additionalProperties", "properties")
    }
    result["$defs"]["part"] = part_def
    result["oneOf"] = [{"$ref": "#/$defs/part"}, {"$ref": "#/$defs/assembly"}]
    result["title"] = f"SDM Part or Assembly ({version})"
    for branch in result["$defs"]["expr"]["oneOf"]:
        if branch["properties"]["type"]["const"] in {"param", "metric"}:
            branch["properties"]["instance"] = {"type": "string", "minLength": 1}
    return result


# ===========================================================================
# index.json: what each version added, derived from since=
# ===========================================================================


def _node_additions(name: str, spec: wire.NodeSpec, version: wire.SchemaVersion) -> list[str]:
    if spec.since == version:
        return [name]
    return [f"{name}.{p.name}" for p in spec.params if p.since == version]


def _added_since(version: wire.SchemaVersion) -> list[str]:
    """Every dotted name (a node kind, or ``Dataclass.field``) whose
    ``since=`` is exactly ``version``, sourced from wire.py's registries.

    The blind spot to know about: an addition living entirely inside one of
    the generator's static literal blocks has no ``since=`` to derive from,
    so nothing here can see it. Declaring a field in the contract is what
    makes it visible. ``Part.kinematics`` is declared in the field contract
    and the index reports it at 0.2. Static-literal additions require an
    explicit version entry to be discoverable here.
    """
    added: list[str] = []
    registries: tuple[dict[str, wire.NodeSpec], ...] = (
        wire.PRIMITIVES,
        wire.OPS,
        wire.TRANSFORMS,
        wire.MODIFIERS,
        wire.DEFORMS,
        wire.TWO_D_TO_3D,
        wire.FIELD_PRIMITIVES,
        wire.FIELD_OPS,
    )
    for registry in registries:
        for name, spec in registry.items():
            added.extend(_node_additions(name, spec, version))
    for spec in (wire.SWEEP, wire.LOFT, wire.VSWEEP):
        added.extend(_node_additions(spec.name, spec, version))
    for dataclass_name, field_specs in wire.DOCUMENT_FIELDS.items():
        for fspec in field_specs:
            if fspec.since == version:
                added.append(f"{dataclass_name}.{fspec.name}")
    added.extend(name for name, since in wire.KINEMATICS_CAPABILITIES.items() if since == version)
    added.extend(name for name, since in wire.EXPR_CAPABILITIES.items() if since == version)
    if version == "0.1":
        # Removed runtime APIs remain part of the immutable historical wire contract.
        added.extend(
            [
                "Part.couplings",
                *(
                    f"CouplingNode.{name}"
                    for name in ("name", "position", "normal", "sdf_tree", "metadata")
                ),
            ]
        )
    return sorted(added)


def generate_index() -> dict[str, Any]:
    """Build the machine-readable version index: which versions exist, which
    is newest, each version's file, and what each added.
    """
    return {
        "latest": wire.SCHEMA_VERSIONS[-1],
        "versions": {
            v: {"file": f"sdm-{v}.schema.json", "added": _added_since(v)}
            for v in wire.SCHEMA_VERSIONS
        },
    }


def write_generated_files(schema_dir: Path = _SCHEMA_DIR) -> None:
    """Write the generated schema and index to ``schema_dir``.

    Touches only the current version's schema file and the index -- never
    a frozen version's file.
    """
    schema_path = schema_dir / f"sdm-{_GENERATED_VERSION}.schema.json"
    schema_path.write_text(json.dumps(generate_schema(), indent=2) + "\n")
    (schema_dir / "index.json").write_text(json.dumps(generate_index(), indent=2) + "\n")


if __name__ == "__main__":
    write_generated_files()
