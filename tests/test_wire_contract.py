"""Pins that software_defined_matter.wire cannot drift from what it describes:
the JAX dispatch tables in sdf/compile.py, the dataclasses in model.py, and
the frozen JSON schemas under software_defined_matter/schema/.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import software_defined_matter as public
from software_defined_matter import model, wire
from software_defined_matter.sdf import compile as sdf_compile
from software_defined_matter.sdf import sdf_shapes

_SDF_DIR = Path(__file__).parent.parent / "src" / "software_defined_matter" / "sdf"
_SCHEMA_DIR = Path(__file__).parent.parent / "src" / "software_defined_matter" / "schema"
_VERSION_ORDER = {v: i for i, v in enumerate(wire.SCHEMA_VERSIONS)}


def _load_schema(version: str) -> dict[str, Any]:
    return json.loads((_SCHEMA_DIR / f"sdm-{version}.schema.json").read_text())


def _sdf_oneof_enum(schema: dict[str, Any], node_type_const: str, key: str) -> list[str]:
    """The literal ``enum`` for ``key`` on the ``$defs.sdf.oneOf`` branch whose
    ``type`` is ``node_type_const`` (e.g. the ``op`` enum on the ``op`` branch).
    """
    definition = schema["$defs"]["sdf"]
    for branch in definition["oneOf"]:
        if branch.get("properties", {}).get("type", {}).get("const") == node_type_const:
            enum = branch["properties"][key].get("enum")
            if enum is not None:
                assert isinstance(enum, list)
                return enum
    # Generated 0.3 is now frozen too. Its enum lives in a conditional
    # sibling so schema errors point to the misspelled discriminator.
    for branch in definition.get("allOf", []):
        condition = branch.get("if", {}).get("properties", {})
        if condition == {"type": {"const": node_type_const}}:
            enum = branch.get("then", {}).get("properties", {}).get(key, {}).get("enum")
            if enum is not None:
                assert isinstance(enum, list)
                return enum
    raise AssertionError(f"no {node_type_const!r} enum in this schema")


def _runtime_param_names(fn: Callable[..., Any]) -> tuple[str, ...]:
    """Every parameter of a leaf evaluator except the leading query point."""
    return tuple(list(inspect.signature(fn).parameters)[1:])


# ---------------------------------------------------------------------------
# wire.py stays importable without JAX.
# ---------------------------------------------------------------------------


def test_wire_module_imports_without_pulling_in_jax():
    """A subprocess import: pytest may already have jax in sys.modules from
    collecting another test file, which would hide a real dependency if
    checked in-process."""
    code = "import sys; import software_defined_matter.wire; assert 'jax' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Bijection: contract registries <-> an INDEPENDENT source, not compile.py's
# dispatch tables.
#
# compile.py's _PRIMITIVES / _FIELD_PRIMITIVES are now BUILT from
# wire.PRIMITIVES / wire.FIELD_PRIMITIVES (``{name: getattr(shapes, name)
# for name in wire.PRIMITIVES}``), so `set(wire.PRIMITIVES) ==
# set(sdf_compile._PRIMITIVES)` would be true by construction and could
# never fail -- a primitive dropped from the contract vanishes from both
# sides of that comparison at once. sdf_shapes's own public function names
# are genuinely independent of wire.py, so the membership check below is
# against those instead. (Per-entry param-name agreement, below, already
# reads sdf_shapes directly via getattr and was never affected by this.)
# ---------------------------------------------------------------------------


def _sdf_shapes_public_functions() -> dict[str, Callable[..., Any]]:
    """Every function DEFINED IN (not merely imported into) sdf_shapes.py."""
    return {
        name: obj
        for name, obj in vars(sdf_shapes).items()
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and obj.__module__ == sdf_shapes.__name__
    }


# sdf_shapes exports one public helper that is neither a primitive nor a
# field primitive: field_add(*field_values) sums pre-evaluated field
# values, takes no query point, and corresponds to wire.FIELD_OPS["add"]
# instead. Named explicitly rather than matched by a "field_" pattern, so
# adding a real field primitive can never be silently swallowed by it.
_FIELD_OP_FUNCTIONS = {"field_add": "add"}


def test_primitive_contract_is_an_exact_bijection_with_sdf_shapes_public_api():
    functions = _sdf_shapes_public_functions()
    primitive_functions = {name for name in functions if not name.startswith("field_")}
    assert primitive_functions == set(wire.PRIMITIVES)


def test_field_primitive_contract_is_an_exact_bijection_with_sdf_shapes_public_api():
    functions = _sdf_shapes_public_functions()
    field_primitive_functions = {
        name.removeprefix("field_")
        for name in functions
        if name.startswith("field_") and name not in _FIELD_OP_FUNCTIONS
    }
    assert field_primitive_functions == set(wire.FIELD_PRIMITIVES)


def test_field_op_contract_matches_the_field_op_function_in_sdf_shapes():
    functions = _sdf_shapes_public_functions()
    field_op_functions = {name for name in _FIELD_OP_FUNCTIONS if name in functions}
    assert field_op_functions == set(_FIELD_OP_FUNCTIONS)  # the allowlist itself isn't stale
    assert set(_FIELD_OP_FUNCTIONS.values()) == set(wire.FIELD_OPS)


def test_op_contract_is_an_exact_bijection_with_compile_dispatch():
    compiled = (
        set(sdf_compile._BINARY_OPS)
        | set(sdf_compile._SMOOTH_BINARY_OPS)
        | {"softmin_many", "softmin_chunked"}
    )
    assert set(wire.OPS) == compiled


# Wire carries a codec of the JAX sample block (dims/encoding/data), not the
# decoded ``values`` array the evaluator takes. Same special-case class as the
# GLSL call shape (base/n/lo/inv_h) — see DR-0003.
_WIRE_RUNTIME_PARAM_EXCEPTIONS = frozenset({"raster_field"})


@pytest.mark.parametrize(
    "name",
    sorted(set(wire.PRIMITIVES) - _WIRE_RUNTIME_PARAM_EXCEPTIONS),
    ids=sorted(set(wire.PRIMITIVES) - _WIRE_RUNTIME_PARAM_EXCEPTIONS),
)
def test_primitive_param_names_match_runtime_signature(name):
    spec = wire.PRIMITIVES[name]
    runtime_fn = getattr(sdf_shapes, name)
    assert spec.param_names() == _runtime_param_names(runtime_fn)


def test_raster_field_wire_is_the_codec_of_the_runtime_args():
    """Pin the intentional split so it cannot drift into a silent mismatch."""
    wire_names = wire.PRIMITIVES["raster_field"].param_names()
    runtime_names = _runtime_param_names(sdf_shapes.raster_field)
    assert runtime_names == ("origin", "spacing", "values")
    assert wire_names[:2] == ("origin", "spacing")
    assert "values" not in wire_names
    for required in ("dims", "encoding", "data"):
        assert required in wire_names
    assert "raster_field" in wire.PRIMITIVES


@pytest.mark.parametrize("name", sorted(wire.FIELD_PRIMITIVES), ids=sorted(wire.FIELD_PRIMITIVES))
def test_field_primitive_param_names_match_runtime_signature(name):
    spec = wire.FIELD_PRIMITIVES[name]
    runtime_fn = getattr(sdf_shapes, f"field_{name}")
    assert spec.param_names() == _runtime_param_names(runtime_fn)


@pytest.mark.parametrize(
    "name", sorted(set(sdf_compile._BINARY_OPS) | set(sdf_compile._SMOOTH_BINARY_OPS))
)
def test_op_k_param_matches_smooth_dispatch(name):
    """Every op compile.py routes through the smoothing table takes exactly
    ``k``; the three hard ops take none."""
    spec = wire.OPS[name]
    if name in sdf_compile._SMOOTH_BINARY_OPS:
        assert spec.param_names() == ("k",)
    else:
        assert spec.param_names() == ()


# ---------------------------------------------------------------------------
# Every contract-declared transform / modifier / deform / 2d_to_3d method
# actually compiles (or, for repeat_inf, fails for the documented reason).
# compile.py dispatches these as if/elif chains rather than dicts, so there
# is no registry to diff against; round-tripping through make_sdf_closure is
# the bijection check available for this category.
# ---------------------------------------------------------------------------


def _leaf() -> dict[str, Any]:
    return model.sdf_primitive("sphere", r=1.0)


def _leaf_2d() -> dict[str, Any]:
    return model.sdf_primitive("circle_2d", r=1.0)


def _sample_kwargs(spec: wire.NodeSpec) -> dict[str, Any]:
    """One well-shaped value per param, enough to compile a representative
    tree -- not a general-purpose fixture."""
    out: dict[str, Any] = {}
    for p in spec.params:
        if p.wire_shape == "scalar":
            out[p.name] = 1.0
        elif p.wire_shape == "vec2":
            out[p.name] = [1.0, 1.0]
        elif p.wire_shape == "vec3":
            out[p.name] = [1.0, 1.0, 1.0]
        elif p.wire_shape == "mat3":
            out[p.name] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        elif p.wire_shape == "point_list":
            n = p.min_points or 2
            out[p.name] = [[float(i), 0.0, 0.0][: p.point_dim] for i in range(n)]
        elif p.wire_shape == "scalar_list":
            out[p.name] = [0.0, 1.0]
        elif p.wire_shape == "string":
            out[p.name] = p.choices[0] if p.choices else "x"
        elif p.wire_shape == "bool":
            out[p.name] = False
    return out


_COMPILED_TRANSFORMS = sorted(n for n in wire.TRANSFORMS if n != "repeat_inf")


@pytest.mark.parametrize("name", _COMPILED_TRANSFORMS, ids=_COMPILED_TRANSFORMS)
def test_every_compiled_transform_is_recognised_by_compile(name):
    from software_defined_matter.sdf.compile import make_sdf_closure

    tree = model.sdf_transform(name, _leaf(), **_sample_kwargs(wire.TRANSFORMS[name]))
    part = model.Part(name="t")
    make_sdf_closure(tree, part)  # must not raise "Unknown transform"


def test_repeat_inf_is_contract_legal_but_not_compiled():
    """Documented exception: repeat_inf is a real wire-legal transform (used
    deliberately by sdf/envelope.py and sdf/bbox.py for an actionable "no
    finite envelope" error) but has no compiled evaluator."""
    from software_defined_matter.sdf.compile import make_sdf_closure

    assert "repeat_inf" in wire.TRANSFORMS
    tree = model.sdf_transform("repeat_inf", _leaf(), c=[1.0, 1.0, 1.0])
    part = model.Part(name="t")
    with pytest.raises(ValueError, match="repeat_inf"):
        make_sdf_closure(tree, part)


@pytest.mark.parametrize("name", sorted(wire.MODIFIERS), ids=sorted(wire.MODIFIERS))
def test_every_modifier_is_recognised_by_compile(name):
    from software_defined_matter.sdf.compile import make_sdf_closure

    tree = model.sdf_modifier(name, _leaf(), **_sample_kwargs(wire.MODIFIERS[name]))
    part = model.Part(name="t")
    make_sdf_closure(tree, part)


@pytest.mark.parametrize("name", sorted(wire.DEFORMS), ids=sorted(wire.DEFORMS))
def test_every_deform_is_recognised_by_compile(name):
    from software_defined_matter.sdf.compile import make_sdf_closure

    kwargs = _sample_kwargs(wire.DEFORMS[name])
    if wire.DEFORMS[name].requires_field:
        kwargs["field"] = model.field_primitive("radial", freq=1.0)
    tree = model.sdf_deform(name, _leaf(), **kwargs)
    part = model.Part(name="t")
    make_sdf_closure(tree, part)


@pytest.mark.parametrize("name", sorted(wire.TWO_D_TO_3D), ids=sorted(wire.TWO_D_TO_3D))
def test_every_2d_to_3d_method_is_recognised_by_compile(name):
    from software_defined_matter.sdf.compile import make_sdf_closure

    tree = model.sdf_2d_to_3d(name, _leaf_2d(), **_sample_kwargs(wire.TWO_D_TO_3D[name]))
    part = model.Part(name="t")
    make_sdf_closure(tree, part)


def test_sweep_and_loft_are_single_node_kinds_not_keyed_registries():
    assert wire.SWEEP.name == "sweep"
    assert wire.LOFT.name == "loft"


def test_sweep_is_recognised_by_compile():
    from software_defined_matter.sdf.compile import make_sdf_closure

    path = [[0.0, 0.0, float(i)] for i in range(4)]
    tree = model.sdf_sweep(_leaf_2d(), path=path)
    make_sdf_closure(tree, model.Part(name="t"))


def test_loft_is_recognised_by_compile():
    from software_defined_matter.sdf.compile import make_sdf_closure

    tree = model.sdf_loft([_leaf_2d(), _leaf_2d()], z=[0.0, 1.0])
    make_sdf_closure(tree, model.Part(name="t"))


# ---------------------------------------------------------------------------
# Structure: document field contracts match dataclasses.fields() on model.py.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataclass_name", "field_specs"),
    list(wire.DOCUMENT_FIELDS.items()),
    ids=list(wire.DOCUMENT_FIELDS),
)
def test_document_field_contract_matches_model_dataclass_fields(dataclass_name, field_specs):
    dc = getattr(public, dataclass_name)
    runtime_fields = {f.name for f in dataclasses.fields(dc)}
    contract_fields = {f.name for f in field_specs if not f.wire_only}
    assert contract_fields == runtime_fields


def test_wire_and_model_agree_on_the_known_schema_versions():
    """``model.KNOWN_SCHEMA_VERSIONS`` and ``wire.SchemaVersion`` are separate
    declarations, because ``model`` cannot import ``wire`` without a cycle.
    Nothing but this gate stops a new version being added to one and not the
    other, which would let a document declare a version the contract cannot
    describe.
    """
    assert tuple(model.KNOWN_SCHEMA_VERSIONS) == tuple(wire.SCHEMA_VERSIONS)
    assert wire.SCHEMA_VERSIONS[-1] == model.LATEST_SCHEMA_VERSION
    assert wire.SCHEMA_VERSIONS[0] == model.FALLBACK_SCHEMA_VERSION


def test_kinematics_is_a_roundtrippable_model_field():
    spec = next(f for f in wire.PART_FIELDS if f.name == "kinematics")
    assert not spec.wire_only
    assert spec.since == "0.2"
    assert "kinematics" in {f.name for f in dataclasses.fields(model.Part)}


def test_every_wire_only_field_is_absent_from_its_dataclass():
    """A ``wire_only`` field that DOES have a dataclass counterpart is a
    mislabel: it would silently skip the structure gate for a field the gate
    should be checking.
    """
    for dataclass_name, field_specs in wire.DOCUMENT_FIELDS.items():
        runtime_fields = {f.name for f in dataclasses.fields(getattr(public, dataclass_name))}
        for spec in field_specs:
            if spec.wire_only:
                assert spec.name not in runtime_fields, (
                    f"{dataclass_name}.{spec.name} is marked wire_only but the dataclass "
                    f"does have that field; drop the flag so the structure gate checks it."
                )


# ---------------------------------------------------------------------------
# since= consistency: gated against the frozen sdm-0.2.schema.json.
# ---------------------------------------------------------------------------

_SCHEMA_0_2_LOCATION: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "Part": lambda schema: schema["properties"],
    "Param": lambda schema: schema["$defs"]["param"]["properties"],
    "MaterialRegion": lambda schema: schema["$defs"]["material"]["properties"],
    "Objective": lambda schema: schema["$defs"]["objective"]["properties"],
    "Constraint": lambda schema: schema["$defs"]["constraint"]["properties"],
}


@pytest.mark.parametrize(
    ("dataclass_name", "field_specs"),
    list(wire.DOCUMENT_FIELDS.items()),
    ids=list(wire.DOCUMENT_FIELDS),
)
def test_document_field_since_is_consistent_with_frozen_0_2_schema(dataclass_name, field_specs):
    schema_0_2 = _load_schema("0.2")
    properties = (
        _SCHEMA_0_2_LOCATION[dataclass_name](schema_0_2)
        if dataclass_name in _SCHEMA_0_2_LOCATION
        else {}
    )
    for spec in field_specs:
        appears = spec.name in properties
        if _VERSION_ORDER[spec.since] <= _VERSION_ORDER["0.2"]:
            assert appears, (
                f"{dataclass_name}.{spec.name} is since={spec.since!r} but is "
                "missing from the frozen sdm-0.2.schema.json"
            )
        else:
            assert not appears, (
                f"{dataclass_name}.{spec.name} is since={spec.since!r} but "
                "already appears in the frozen sdm-0.2.schema.json"
            )


# ---------------------------------------------------------------------------
# Enum drift: the schema's own literal enums are a subset of the contract.
# The frozen schemas never gate primitive.kind / modifier / deform with a
# literal, so this check applies only to the vocabularies the schema DOES
# enumerate (op, transform, unit): see wire.py's module docstring.
#
# op/transform are checked only on the FROZEN versions. The current version
# is generated straight from wire.OPS / wire.TRANSFORMS (see
# schema/_generate.py), so its enum agrees with the contract by construction
# -- a subset check there could never fail. tests/test_schema_generate.py's
# fixed-point test (regenerating produces no diff) is the strictly stronger
# guarantee that replaces it for that version.
# ---------------------------------------------------------------------------

_FROZEN_SCHEMA_VERSIONS = wire.SCHEMA_VERSIONS[:-1]


@pytest.mark.parametrize("version", _FROZEN_SCHEMA_VERSIONS)
def test_schema_op_enum_is_a_subset_of_the_contract(version):
    schema = _load_schema(version)
    assert set(_sdf_oneof_enum(schema, "op", "op")) <= set(wire.OPS)


@pytest.mark.parametrize("version", _FROZEN_SCHEMA_VERSIONS)
def test_schema_transform_enum_is_a_subset_of_the_contract(version):
    schema = _load_schema(version)
    assert set(_sdf_oneof_enum(schema, "transform", "transform")) <= set(wire.TRANSFORMS)


@pytest.mark.parametrize("version", wire.SCHEMA_VERSIONS)
def test_schema_unit_enum_is_a_subset_of_the_contract(version):
    schema = _load_schema(version)
    enum = schema["$defs"]["param"]["properties"]["unit"]["enum"]
    assert set(enum) <= set(wire.UNIT_NAMES)


@pytest.mark.parametrize("version", wire.SCHEMA_VERSIONS)
def test_schema_objective_sense_enum_is_a_subset_of_the_contract(version):
    schema = _load_schema(version)
    enum = schema["$defs"]["objective"]["properties"]["sense"]["enum"]
    assert set(enum) <= set(wire.OBJECTIVE_SENSES)


@pytest.mark.parametrize("version", wire.SCHEMA_VERSIONS)
def test_schema_constraint_op_enum_is_a_subset_of_the_contract(version):
    schema = _load_schema(version)
    enum = schema["$defs"]["constraint"]["properties"]["op"]["enum"]
    assert set(enum) <= set(wire.CONSTRAINT_OPS)


@pytest.mark.parametrize("version", _FROZEN_SCHEMA_VERSIONS)
def test_schema_expr_node_types_are_a_subset_of_the_contract(version):
    """0.1 predates the `dof` branch, so the frozen versions are a subset of
    the contract rather than equal to it. Only the current version, generated
    from wire.EXPR_NODE_TYPES, matches exactly, and the fixed-point test in
    test_schema_generate.py covers that one."""
    schema = _load_schema(version)
    types = {branch["properties"]["type"]["const"] for branch in schema["$defs"]["expr"]["oneOf"]}
    assert types <= set(wire.EXPR_NODE_TYPES)


def test_generator_describes_exactly_the_declared_expression_node_types():
    """The generator writes each expr branch's interior (operand names,
    operator enums) but takes the set of node types from the contract. If the
    two disagree, generation raises rather than emitting a schema that is
    missing a declared type, or that pins one the contract never declared."""
    from software_defined_matter.schema._generate import _EXPR_BRANCH_BODIES

    assert set(_EXPR_BRANCH_BODIES) == set(wire.EXPR_NODE_TYPES)


def test_expr_def_branch_order_follows_the_contract():
    """Branch order is contract order, not table order. The committed schema
    is regenerated byte-for-byte in CI, so a reordering here would show up as
    a spurious diff on an already-released-shaped file."""
    from software_defined_matter.schema._generate import _expr_def

    order = [branch["properties"]["type"]["const"] for branch in _expr_def()["oneOf"]]
    assert order == list(wire.EXPR_NODE_TYPES)


# ---------------------------------------------------------------------------
# Hand-authored entries the seed tables in conftest.py don't cover.
# ---------------------------------------------------------------------------


def test_plane_is_hand_authored_as_the_csg_cutter_primitive():
    """plane is absent from tests/conftest.py's PRIMITIVES table (it cannot
    mesh on its own -- see sdf/validate.py's UnboundedRootError) and had to
    be authored by hand from sdf_shapes.plane's signature."""
    spec = wire.PRIMITIVES["plane"]
    assert spec.param_names() == ("n", "h")
    assert spec.required_param_names() == ("n", "h")


# ---------------------------------------------------------------------------
# ParamSpec / NodeSpec / FieldSpec validate themselves at construction.
# ---------------------------------------------------------------------------


def test_paramspec_rejects_unknown_wire_shape():
    with pytest.raises(ValueError, match="wire_shape"):
        wire.ParamSpec("x", True, "vec4")  # type: ignore[arg-type]


def test_paramspec_rejects_choices_on_a_non_string_shape():
    with pytest.raises(ValueError, match="choices"):
        wire.ParamSpec("x", True, "scalar", choices=("a", "b"))


def test_paramspec_rejects_point_list_refinements_on_a_non_point_list_shape():
    with pytest.raises(ValueError, match="point_dim"):
        wire.ParamSpec("x", True, "scalar", point_dim=2)


def test_nodespec_rejects_duplicate_param_names():
    with pytest.raises(ValueError, match="duplicate"):
        wire.NodeSpec(
            "x",
            "primitive",
            params=(wire.ParamSpec("r", True, "scalar"), wire.ParamSpec("r", True, "scalar")),
        )


def test_nodespec_rejects_a_column_set_on_the_wrong_category():
    with pytest.raises(ValueError, match="is_lattice"):
        wire.NodeSpec("x", "op", is_lattice=True)


def test_fieldspec_rejects_unknown_schema_version():
    with pytest.raises(ValueError, match="since"):
        wire.FieldSpec("x", True, "scalar", since="0.9")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# since= on the SDF/field vocabulary records the first schema that accepts a
# node name. The original vocabulary predates enumerated discriminators and
# therefore uses the 0.1 baseline; later additions name their schema cut.
# ---------------------------------------------------------------------------


def test_sdf_node_versions_match_their_schema_introduction():
    registries: list[dict[str, wire.NodeSpec]] = [
        wire.PRIMITIVES,
        wire.OPS,
        wire.TRANSFORMS,
        wire.MODIFIERS,
        wire.DEFORMS,
        wire.TWO_D_TO_3D,
        wire.FIELD_PRIMITIVES,
        wire.FIELD_OPS,
    ]
    for registry in registries:
        for spec in registry.values():
            expected = (
                "0.6" if spec.name in {"shear_linear", "taper_linear", "scale_axis"} else "0.1"
            )
            assert spec.since == expected, (
                f"{spec.name}: expected since={expected!r}, got {spec.since!r}"
            )
            for p in spec.params:
                assert p.since == expected, (
                    f"{spec.name}.{p.name}: expected since={expected!r}, got {p.since!r}"
                )
    for singleton in (wire.SWEEP, wire.LOFT):
        assert singleton.since == "0.1"
        for p in singleton.params:
            assert p.since == "0.1"


# ---------------------------------------------------------------------------
# Gate: every column the contract declares has a real reader. A declared
# column nothing reads is exactly how the dim defect happened -- wire.py
# populated dim on every primitive, but sdf/validate.py never looked at it,
# and nothing caught that until code review.
#
# Two different reading patterns both count as "consumed":
#
#   direct  -- sdf/validate.py dereferences the attribute itself (dim gates
#              the 2-D/3-D check this test exists to prevent from
#              recurring; requires_field gates the deform("displace") field
#              check; name/category/params drive its error messages and
#              per-kwarg validation).
#   derived -- wire.py dereferences the attribute ONCE, to build a public
#              value every caller uses instead (is_lattice/removes_material/
#              hollows/tiles feed the LATTICE_PRIMITIVES/SUBTRACT_OPS/
#              HOLLOWING_MODIFIERS/TILING_TRANSFORMS frozensets
#              sdf/envelope.py reads; required feeds
#              NodeSpec.required_param_names(), which sdf/validate.py
#              calls). The attribute access lives in wire.py either way;
#              this is still a real reader, not a dead column.
#
# since is neither: it is uniform across the whole SDF vocabulary by
# construction (pinned above), not something any reader branches on.
# ---------------------------------------------------------------------------

_WIRE_SOURCE = (
    Path(__file__).parent.parent / "src" / "software_defined_matter" / "wire.py"
).read_text()
_VALIDATOR_SOURCE = (_SDF_DIR / "validate.py").read_text()

_DIRECT_NODESPEC_FIELDS = ("name", "category", "params", "dim", "requires_field")
_DERIVED_NODESPEC_FIELDS = ("is_lattice", "removes_material", "hollows", "tiles")
_DOCUMENTATION_ONLY_NODESPEC_FIELDS = ("since",)

_DIRECT_PARAMSPEC_FIELDS = (
    "name",
    "wire_shape",
    "ref_ok",
    "choices",
    "point_dim",
    "min_points",
    "multiple_of",
)
_DERIVED_PARAMSPEC_FIELDS = ("required",)
_DOCUMENTATION_ONLY_PARAMSPEC_FIELDS = ("since",)


def _is_referenced(attribute: str, source: str) -> bool:
    return re.search(rf"\.{re.escape(attribute)}\b", source) is not None


def test_every_nodespec_field_is_assigned_to_a_real_reader():
    declared = {f.name for f in dataclasses.fields(wire.NodeSpec)}
    accounted_for = (
        set(_DIRECT_NODESPEC_FIELDS)
        | set(_DERIVED_NODESPEC_FIELDS)
        | set(_DOCUMENTATION_ONLY_NODESPEC_FIELDS)
    )
    assert declared == accounted_for, (
        f"NodeSpec field(s) {declared ^ accounted_for} are declared but not assigned "
        "to a reader list above (or a reader list names a field NodeSpec no longer has)."
    )


@pytest.mark.parametrize("field", _DIRECT_NODESPEC_FIELDS)
def test_validator_directly_reads_the_nodespec_field_it_claims_to(field):
    assert _is_referenced(field, _VALIDATOR_SOURCE), (
        f"NodeSpec.{field} is claimed to be read directly by sdf/validate.py but no "
        f"'.{field}' attribute access was found there."
    )


@pytest.mark.parametrize("field", _DERIVED_NODESPEC_FIELDS)
def test_wire_derives_a_public_value_from_the_nodespec_field_it_claims_to(field):
    assert _is_referenced(field, _WIRE_SOURCE), (
        f"NodeSpec.{field} is claimed to feed a derived value in wire.py but no "
        f"'.{field}' attribute access was found there."
    )


def test_every_paramspec_field_is_assigned_to_a_real_reader():
    declared = {f.name for f in dataclasses.fields(wire.ParamSpec)}
    accounted_for = (
        set(_DIRECT_PARAMSPEC_FIELDS)
        | set(_DERIVED_PARAMSPEC_FIELDS)
        | set(_DOCUMENTATION_ONLY_PARAMSPEC_FIELDS)
    )
    assert declared == accounted_for, (
        f"ParamSpec field(s) {declared ^ accounted_for} are declared but not assigned "
        "to a reader list above (or a reader list names a field ParamSpec no longer has)."
    )


@pytest.mark.parametrize("field", _DIRECT_PARAMSPEC_FIELDS)
def test_validator_directly_reads_the_paramspec_field_it_claims_to(field):
    assert _is_referenced(field, _VALIDATOR_SOURCE), (
        f"ParamSpec.{field} is claimed to be read directly by sdf/validate.py but no "
        f"'.{field}' attribute access was found there."
    )


@pytest.mark.parametrize("field", _DERIVED_PARAMSPEC_FIELDS)
def test_wire_derives_a_public_value_from_the_paramspec_field_it_claims_to(field):
    assert _is_referenced(field, _WIRE_SOURCE), (
        f"ParamSpec.{field} is claimed to feed a derived value in wire.py but no "
        f"'.{field}' attribute access was found there."
    )
