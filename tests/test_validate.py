"""Semantic validation of SDF trees (software_defined_matter.sdf.validate).

The JSON Schema validates structure; this module pins two independent
layers of contextual checks it cannot express:

  - Position: a `plane` used as a material root (or in any position where
    the bounded-sibling argument doesn't apply) is unbounded and rejected.
  - Vocabulary and shape: `validate_document_semantics` walks every SDF
    tree in a document (both `materials` and `couplings`) against the
    node-kind contract in `software_defined_matter.wire`, independent of
    the document's declared `schema_version`. This covers unknown kind /
    modifier / deform / op names, missing or unrecognised kwargs, a kwarg
    with the wrong wire shape, a primitive whose dimension doesn't match
    the position it sits in, and a dangling `$ref`.
"""

from __future__ import annotations

import pytest

from software_defined_matter import (
    LATEST_SCHEMA_VERSION,
    MaterialRegion,
    Part,
    field_op,
    field_primitive,
    sdf_2d_to_3d,
    sdf_deform,
    sdf_loft,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_sweep,
    sdf_transform,
    wire,
)
from software_defined_matter.io import load_schema, validate
from software_defined_matter.sdf.validate import (
    SemanticValidationError,
    UnboundedRootError,
    validate_document_semantics,
    validate_material_tree,
)

_PLANE = sdf_primitive("plane", n=[0.0, 0.0, 1.0], h=0.0)
_SPHERE = sdf_primitive("sphere", r=2.0)
_CIRCLE_2D = sdf_primitive("circle_2d", r=1.0)


def _wrap(sdf_tree):
    return Part(
        name="p",
        materials=[MaterialRegion(material_id=1, name="Steel", sdf_tree=sdf_tree)],
    )


# ---------------------------------------------------------------------------
# Plane at the root (and in clearly-unbounded positions) is rejected.
# ---------------------------------------------------------------------------


def test_plane_as_root_is_rejected():
    with pytest.raises(UnboundedRootError, match="plane"):
        validate_material_tree(_PLANE, region_name="Steel")


def test_plane_as_root_rejected_via_part_validate():
    """End-to-end through io.validate(Part(...)): the message names the region."""
    with pytest.raises(UnboundedRootError, match="Steel"):
        validate(_wrap(_PLANE))


def test_plane_inside_union_is_rejected():
    tree = sdf_op("union", [_SPHERE, _PLANE])
    with pytest.raises(UnboundedRootError):
        validate_material_tree(tree)


def test_plane_inside_smooth_union_is_rejected():
    tree = sdf_op("smooth_union", [_SPHERE, _PLANE], k=0.1)
    with pytest.raises(UnboundedRootError):
        validate_material_tree(tree)


def test_plane_as_first_child_of_subtract_is_rejected():
    """The minuend must be bounded; an unbounded minuend gives an unbounded result."""
    tree = sdf_op("subtract", [_PLANE, _SPHERE])
    with pytest.raises(UnboundedRootError):
        validate_material_tree(tree)


def test_plane_wrapped_in_transform_is_rejected():
    """Transforms preserve unboundedness."""
    tree = sdf_transform("translate", _PLANE, t=[1.0, 0.0, 0.0])
    with pytest.raises(UnboundedRootError):
        validate_material_tree(tree)


def test_plane_wrapped_in_modifier_is_rejected():
    tree = sdf_modifier("round", _PLANE, r=0.5)
    with pytest.raises(UnboundedRootError):
        validate_material_tree(tree)


def test_plane_wrapped_in_deform_is_rejected():
    tree = sdf_deform("twist", _PLANE, k=0.3)
    with pytest.raises(UnboundedRootError):
        validate_material_tree(tree)


# ---------------------------------------------------------------------------
# Plane is allowed where a bounded sibling clamps the result.
# ---------------------------------------------------------------------------


def test_plane_as_subtrahend_of_subtract_is_ok():
    tree = sdf_op("subtract", [_SPHERE, _PLANE])
    validate_material_tree(tree)  # must not raise


def test_plane_in_intersect_is_ok():
    tree = sdf_op("intersect", [_SPHERE, _PLANE])
    validate_material_tree(tree)


def test_plane_in_smooth_intersect_is_ok():
    tree = sdf_op("smooth_intersect", [_SPHERE, _PLANE], k=0.1)
    validate_material_tree(tree)


def test_plane_in_nested_subtract_inside_union_is_ok():
    """`union(sphere, subtract(box, plane))`: the inner subtract clamps the plane."""
    inner = sdf_op("subtract", [sdf_primitive("box", b=[1.0, 1.0, 1.0]), _PLANE])
    tree = sdf_op("union", [_SPHERE, inner])
    validate_material_tree(tree)


# ---------------------------------------------------------------------------
# Other trees pass: bounded primitives, 2d_to_3d lifts, TPMS with n_periods.
# ---------------------------------------------------------------------------


def test_bounded_primitive_passes():
    validate_material_tree(_SPHERE)


def test_2d_to_3d_lift_passes():
    tree = sdf_2d_to_3d("extrusion", sdf_primitive("circle_2d", r=2.0), h=3.0)
    validate_material_tree(tree)


def test_tpms_with_n_periods_passes():
    tree = sdf_primitive("gyroid", period=2.0, min_thickness=0.036755, n_periods=[3, 3, 3])
    validate_material_tree(tree)


# ---------------------------------------------------------------------------
# Error message quality.
# ---------------------------------------------------------------------------


def test_plane_error_message_is_actionable():
    """The error must tell the user where plane *is* valid."""
    with pytest.raises(UnboundedRootError) as exc_info:
        validate_material_tree(_PLANE, region_name="Cu")
    msg = str(exc_info.value)
    assert "Cu" in msg
    assert "subtract" in msg
    assert "intersect" in msg


# ---------------------------------------------------------------------------
# validate_document_semantics: the registry-driven layer.
#
# The probes below all PASSED validate() before this layer existed -- a
# typo'd kind, a wrong kwarg name, a missing required kwarg, an unknown
# modifier / deform, and a dangling param $ref. Each is checked twice: once
# on its own, and once inside a document declaring schema_version="0.1" --
# the whole point of a version-independent layer is that the older, more
# permissive schema versions catch these exactly as well as the current one.
# ---------------------------------------------------------------------------


def _doc(tree, params=None, schema_version="0.2"):
    return {
        "schema_version": schema_version,
        "name": "p",
        "params": params or {},
        "materials": [{"material_id": 1, "name": "Steel", "sdf_tree": tree}],
    }


def _coupling_doc(tree, params=None, schema_version="0.2"):
    return {
        "schema_version": schema_version,
        "name": "p",
        "params": params or {},
        "couplings": [{"name": "c", "position": [0.0, 0.0, 0.0], "sdf_tree": tree}],
    }


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_unknown_primitive_kind_is_rejected(schema_version):
    tree = sdf_primitive("spere", r=1.0)
    with pytest.raises(SemanticValidationError, match="kind"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_made_up_primitive_kind_is_rejected(schema_version):
    tree = sdf_primitive("made_up_primitive")
    with pytest.raises(SemanticValidationError, match="kind"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_wrong_keyword_argument_name_is_rejected(schema_version):
    tree = sdf_primitive("sphere", radius=5.0)
    with pytest.raises(SemanticValidationError, match="radius"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_missing_required_argument_is_rejected(schema_version):
    tree = sdf_primitive("sphere")
    with pytest.raises(SemanticValidationError, match="r"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_unknown_modifier_name_is_rejected(schema_version):
    tree = sdf_modifier("shel", _SPHERE, r=0.1)
    with pytest.raises(SemanticValidationError, match="modifier"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_unknown_deform_name_is_rejected(schema_version):
    tree = sdf_deform("twst", _SPHERE, k=0.1)
    with pytest.raises(SemanticValidationError, match="deform"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_dangling_param_ref_is_rejected(schema_version):
    tree = sdf_primitive("sphere", r={"$ref": "no_such_param"})
    with pytest.raises(SemanticValidationError, match="no_such_param"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


def test_unknown_primitive_kind_is_rejected_end_to_end_via_io_validate():
    """Same probe, through the public entry point WS2 wires this into."""
    tree = sdf_primitive("spere", r=1.0)
    with pytest.raises(SemanticValidationError, match="kind"):
        validate(_doc(tree))


def test_unknown_primitive_kind_is_rejected_end_to_end_on_schema_0_1():
    tree = sdf_primitive("spere", r=1.0)
    with pytest.raises(SemanticValidationError, match="kind"):
        validate(_doc(tree, schema_version="0.1"))


# ---------------------------------------------------------------------------
# Structural rules beyond the probe table: an op needs children, and
# deform("displace") needs its structural 'field' subtree.
# ---------------------------------------------------------------------------


def test_op_with_no_children_is_rejected():
    tree = {"type": "op", "op": "union", "children": []}
    with pytest.raises(SemanticValidationError, match="children"):
        validate_document_semantics(_doc(tree))


def test_displace_without_a_field_subtree_is_rejected():
    tree = {"type": "deform", "deform": "displace", "child": _SPHERE}
    with pytest.raises(SemanticValidationError, match="field"):
        validate_document_semantics(_doc(tree))


def test_sweep_rejects_a_path_kind_outside_its_choices():
    tree = sdf_sweep(
        sdf_primitive("circle_2d", r=1.0),
        path=[[0.0, 0.0, float(i)] for i in range(4)],
        path_kind="loop",
    )
    with pytest.raises(SemanticValidationError, match="path_kind"):
        validate_document_semantics(_doc(tree))


# ---------------------------------------------------------------------------
# Expression subtrees nested in a kwarg: the node type is checked, one level
# deep, and nothing below it.
#
# Everywhere a number is expected, three things are accepted: a literal, a
# {"$ref": name} leaf, and an expression subtree. Only the third had no name
# check. The JSON Schema pins expression node types under $defs/expr, but
# only reaches a field *typed* as an expr -- an objective's or constraint's
# `expr`. In the frozen schemas an sdf.params object is a bare
# {"type": "object"}, so a subtree nested in a kwarg was checked by nothing
# and surfaced at first evaluation, which is lazy, as a raw ValueError naming
# no field. The generated current version constrains params per kind, but
# these checks run on a document of any declared version.
# ---------------------------------------------------------------------------

_PARAM_RADIUS = {"radius": {"name": "radius", "value": 2.0, "unit": "mm"}}


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_unknown_expression_node_type_in_a_kwarg_is_rejected(schema_version):
    tree = sdf_primitive("sphere", r={"type": "binp", "op": "*", "lhs": 2.0, "rhs": 3.0})
    with pytest.raises(SemanticValidationError, match="binp"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


def test_unknown_expression_node_type_error_names_the_path_and_the_allowed_set():
    """The error has to say which kwarg and what was allowed there; the
    failure this replaces was a ValueError from inside a JAX trace."""
    tree = sdf_primitive("sphere", r={"type": "binp", "op": "*", "lhs": 2.0, "rhs": 3.0})
    with pytest.raises(SemanticValidationError) as excinfo:
        validate_document_semantics(_doc(tree))
    message = str(excinfo.value)
    assert "materials[0].sdf_tree.params.r.type" in message
    assert "binop" in message


def test_unknown_expression_node_type_nested_in_a_vector_element_is_rejected():
    """A vector slot validates per element, so the check has to reach an
    expression sitting in one entry of a vec3 as well as a bare scalar slot."""
    tree = sdf_primitive("box", b=[1.0, {"type": "nmu", "value": 2.0}, 3.0])
    with pytest.raises(SemanticValidationError, match="nmu"):
        validate_document_semantics(_doc(tree))


@pytest.mark.parametrize("node_type", wire.EXPR_NODE_TYPES)
def test_every_declared_expression_node_type_is_accepted_in_a_kwarg(node_type):
    """The gate has to be a real bijection with the contract, not a list that
    happens to contain whatever the other tests probe: every name wire
    declares must pass here, or the check rejects valid documents."""
    tree = sdf_primitive("sphere", r={"type": node_type})
    validate_document_semantics(_doc(tree, params=_PARAM_RADIUS))  # must not raise


def test_expression_node_contents_are_not_this_layers_contract():
    """Deliberately shallow: a declared node type with a nonsense operator
    and no operands passes here. Validating that is dsl/expr.py's job, and
    duplicating it would put the expression grammar in two places that could
    disagree."""
    tree = sdf_primitive("sphere", r={"type": "binop", "op": "no_such_op"})
    validate_document_semantics(_doc(tree, params=_PARAM_RADIUS))  # must not raise


def _expr_types_in_schema(schema_version):
    oneof = load_schema(schema_version)["$defs"]["expr"]["oneOf"]
    return {branch["properties"]["type"]["const"] for branch in oneof}


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_no_schema_version_allows_an_expression_type_the_gate_rejects(schema_version):
    """wire.EXPR_NODE_TYPES is a hand-written mirror of what the JSON Schema
    already pins under $defs/expr, and two hand-written copies of one
    vocabulary drift. This direction is the one that would break documents:
    a name some schema version accepts but the gate does not is a valid
    document falsely rejected."""
    assert _expr_types_in_schema(schema_version) <= set(wire.EXPR_NODE_TYPES), (
        f"sdm-{schema_version}.schema.json $defs/expr allows a node type "
        f"wire.EXPR_NODE_TYPES would reject"
    )


def test_the_gate_allows_nothing_the_newest_schema_does_not():
    """The other direction, against the newest version only: 0.1 predates the
    `dof` branch, so the gate is deliberately wider than 0.1's expr
    definition (see wire.EXPR_NODE_TYPES). It must not be wider than the
    newest one -- that would mean a name nothing in the format defines."""
    assert set(wire.EXPR_NODE_TYPES) == _expr_types_in_schema(LATEST_SCHEMA_VERSION)


def test_the_gate_is_wider_than_0_1_only_by_dof():
    """Pins the single deliberate exception, so a second one cannot be added
    silently by widening the Literal and re-running the suite."""
    assert set(wire.EXPR_NODE_TYPES) - _expr_types_in_schema("0.1") == {"dof"}


# ---------------------------------------------------------------------------
# A resolvable $ref, and trees exercising every node kind, both pass.
# ---------------------------------------------------------------------------


def test_resolvable_param_ref_passes():
    tree = sdf_primitive("sphere", r={"$ref": "radius"})
    doc = _doc(tree, params={"radius": {"name": "radius", "value": 2.0, "unit": "mm"}})
    validate_document_semantics(doc)  # must not raise


def test_tree_combining_op_transform_and_modifier_passes():
    tree = sdf_op(
        "smooth_union",
        [
            sdf_modifier("round", _SPHERE, r=0.2),
            sdf_transform("translate", _SPHERE, t=[1.0, 0.0, 0.0]),
        ],
        k=0.1,
    )
    validate_document_semantics(_doc(tree))  # must not raise


def test_2d_to_3d_and_deform_displace_pass():
    tree = sdf_deform(
        "displace",
        sdf_2d_to_3d("extrusion", sdf_primitive("circle_2d", r=1.0), h=2.0),
        field=field_primitive("radial", freq=1.0),
    )
    validate_document_semantics(_doc(tree))  # must not raise


# ---------------------------------------------------------------------------
# Dimension mismatch. A primitive's declared dimension (wire.NodeSpec.dim)
# must match the position it sits in. Before this check existed both
# directions were reachable and silent or ugly:
#
#   sdf_2d_to_3d("extrusion", sdf_primitive("sphere", r=1.0), h=2.0)
#     -> compiled and evaluated fine: d(origin) = -1.0, no error anywhere.
#
#   sdf_primitive("circle_2d", r=1.0) as a MaterialRegion root
#     -> circle_2d(p, r) = |p| - r is dimension-agnostic, so on a 3-D query
#        point it silently becomes a SPHERE (d(origin) = -r): bounded and
#        entirely plausible-looking, not an obvious break. trapezoid_2d and
#        uneven_capsule_2d truly ignore z and are unbounded along it; other
#        2-D primitives (box_2d, polygon_2d, ...) at least raise -- late,
#        from deep inside a JAX trace, but not silently.
#
# Both directions are checked at every position the review named, and the
# silent 2-D-at-a-3-D-root cases (circle_2d, trapezoid_2d, uneven_capsule_2d)
# are pinned by name since those have no other backstop at all.
# ---------------------------------------------------------------------------

_3D_LEAF = sdf_primitive("sphere", r=1.0)


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_3d_primitive_as_a_2d_to_3d_child_is_rejected(schema_version):
    tree = sdf_2d_to_3d("extrusion", _3D_LEAF, h=2.0)
    with pytest.raises(SemanticValidationError, match="2-D"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("circle_2d", {"r": 1.0}),
        ("trapezoid_2d", {"r1": 1.0, "r2": 0.5, "he": 1.0}),
        ("uneven_capsule_2d", {"r1": 1.0, "r2": 0.5, "h": 2.0}),
    ],
    ids=["circle_2d", "trapezoid_2d", "uneven_capsule_2d"],
)
def test_silently_dimension_agnostic_2d_primitive_as_material_root_is_rejected(kind, params):
    tree = sdf_primitive(kind, **params)
    with pytest.raises(SemanticValidationError, match="3-D"):
        validate_document_semantics(_doc(tree))


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_2d_primitive_as_material_root_is_rejected_on_every_schema_version(schema_version):
    tree = sdf_primitive("circle_2d", r=1.0)
    with pytest.raises(SemanticValidationError, match="3-D"):
        validate_document_semantics(_doc(tree, schema_version=schema_version))


def test_2d_primitive_inside_a_3d_op_is_rejected():
    tree = sdf_op("union", [_3D_LEAF, sdf_primitive("circle_2d", r=1.0)])
    with pytest.raises(SemanticValidationError, match="3-D"):
        validate_document_semantics(_doc(tree))


def test_2d_primitive_inside_a_3d_transform_is_rejected():
    tree = sdf_transform("translate", sdf_primitive("circle_2d", r=1.0), t=[1.0, 0.0, 0.0])
    with pytest.raises(SemanticValidationError, match="3-D"):
        validate_document_semantics(_doc(tree))


def test_3d_primitive_as_a_sweep_profile_is_rejected():
    tree = sdf_sweep(
        sdf_primitive("box", b=[1.0, 1.0, 1.0]),
        path=[[0.0, 0.0, float(i)] for i in range(4)],
    )
    with pytest.raises(SemanticValidationError, match="2-D"):
        validate_document_semantics(_doc(tree))


def test_3d_primitive_as_a_loft_child_is_rejected():
    tree = sdf_loft(
        [sdf_primitive("sphere", r=1.0), sdf_primitive("circle_2d", r=1.0)],
        z=[0.0, 1.0],
    )
    with pytest.raises(SemanticValidationError, match="2-D"):
        validate_document_semantics(_doc(tree))


def test_extrusion_of_a_2d_primitive_passes():
    tree = sdf_2d_to_3d("extrusion", sdf_primitive("circle_2d", r=1.0), h=2.0)
    validate_document_semantics(_doc(tree))  # must not raise


def test_sweep_of_a_2d_profile_passes():
    tree = sdf_sweep(
        sdf_primitive("circle_2d", r=1.0),
        path=[[0.0, 0.0, float(i)] for i in range(4)],
    )
    validate_document_semantics(_doc(tree))  # must not raise


def test_loft_of_2d_children_passes():
    tree = sdf_loft(
        [sdf_primitive("circle_2d", r=1.0), sdf_primitive("circle_2d", r=1.5)],
        z=[0.0, 1.0],
    )
    validate_document_semantics(_doc(tree))  # must not raise


# ---------------------------------------------------------------------------
# CouplingNode.sdf_tree gets the same checks as a MaterialRegion's. The walk
# in validate_document_semantics has always covered both branches, but
# nothing here exercised the couplings one -- a coupling's local geometry
# (a bolt-hole pattern, a mating face) could carry a typo'd kind, a bad
# kwarg, or a dangling $ref with nothing to catch it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_unknown_primitive_kind_in_a_coupling_tree_is_rejected(schema_version):
    tree = sdf_primitive("spere", r=1.0)
    with pytest.raises(SemanticValidationError, match="kind"):
        validate_document_semantics(_coupling_doc(tree, schema_version=schema_version))


def test_wrong_keyword_argument_name_in_a_coupling_tree_is_rejected():
    tree = sdf_primitive("sphere", radius=5.0)
    with pytest.raises(SemanticValidationError, match="radius"):
        validate_document_semantics(_coupling_doc(tree))


def test_missing_required_argument_in_a_coupling_tree_is_rejected():
    tree = sdf_primitive("sphere")
    with pytest.raises(SemanticValidationError, match="r"):
        validate_document_semantics(_coupling_doc(tree))


def test_unknown_modifier_in_a_coupling_tree_is_rejected():
    tree = sdf_modifier("shel", _SPHERE, r=0.1)
    with pytest.raises(SemanticValidationError, match="modifier"):
        validate_document_semantics(_coupling_doc(tree))


def test_unknown_deform_in_a_coupling_tree_is_rejected():
    tree = sdf_deform("twst", _SPHERE, k=0.1)
    with pytest.raises(SemanticValidationError, match="deform"):
        validate_document_semantics(_coupling_doc(tree))


def test_dangling_param_ref_in_a_coupling_tree_is_rejected():
    tree = sdf_primitive("sphere", r={"$ref": "no_such_param"})
    with pytest.raises(SemanticValidationError, match="no_such_param"):
        validate_document_semantics(_coupling_doc(tree))


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_2d_primitive_as_coupling_root_is_rejected(schema_version):
    """A coupling root is a 3-D context exactly like a material root."""
    tree = sdf_primitive("circle_2d", r=1.0)
    with pytest.raises(SemanticValidationError, match="3-D"):
        validate_document_semantics(_coupling_doc(tree, schema_version=schema_version))


def test_3d_primitive_as_a_2d_to_3d_child_in_a_coupling_tree_is_rejected():
    tree = sdf_2d_to_3d("extrusion", _3D_LEAF, h=2.0)
    with pytest.raises(SemanticValidationError, match="2-D"):
        validate_document_semantics(_coupling_doc(tree))


def test_well_formed_coupling_tree_passes():
    tree = sdf_op(
        "smooth_union",
        [
            sdf_modifier("round", _SPHERE, r=0.2),
            sdf_2d_to_3d("extrusion", sdf_primitive("circle_2d", r=1.0), h=2.0),
        ],
        k=0.1,
    )
    validate_document_semantics(_coupling_doc(tree))  # must not raise


def test_coupling_with_no_sdf_tree_is_not_walked():
    """sdf_tree is optional on a CouplingNode; absent means no local geometry."""
    doc = {
        "schema_version": "0.2",
        "name": "p",
        "params": {},
        "couplings": [{"name": "c", "position": [0.0, 0.0, 0.0]}],
    }
    validate_document_semantics(doc)  # must not raise


# ---------------------------------------------------------------------------
# Wire-shape enforcement in _validate_value. The probe table above only ever
# exercises the "scalar" branch (sphere's "r"); every other branch --
# vec2/vec3/mat3/point_list/scalar_list, and the point_dim/min_points/
# multiple_of point-list refinements -- had no test of its own and nothing
# would have caught a future refactor silently gutting one of them.
# ---------------------------------------------------------------------------


def test_vec2_slot_given_the_wrong_number_of_elements_is_rejected():
    tree = sdf_primitive("torus", t=[1.0, 1.0, 1.0])  # t is vec2
    with pytest.raises(SemanticValidationError, match="length-2"):
        validate_document_semantics(_doc(tree))


def test_vec3_slot_given_the_wrong_number_of_elements_is_rejected():
    tree = sdf_primitive("box", b=[1.0, 1.0])  # b is vec3
    with pytest.raises(SemanticValidationError, match="length-3"):
        validate_document_semantics(_doc(tree))


# ---------------------------------------------------------------------------
# "axis_vector": a per-axis vector is as long as the subtree it sits in has
# axes, on a transform (translate, mirror, repeat_finite) or a modifier
# (elongate) alike. The contract cannot name a single length for these, so
# validation reads it from expected_dim. Accepting either length instead would
# let a 2-D vector reach the GLSL emitter inside a 3-D subtree, where it
# surfaces as a bare ValueError rather than a validation error.
# ---------------------------------------------------------------------------


def test_two_d_translate_inside_a_two_d_subtree_passes():
    inner = sdf_transform("translate", _CIRCLE_2D, t=[1.0, 2.0])
    tree = sdf_2d_to_3d("extrusion", inner, h=1.0)
    validate_document_semantics(_doc(tree))  # must not raise


def test_two_d_translate_in_a_three_d_subtree_is_rejected():
    tree = sdf_transform("translate", _SPHERE, t=[1.0, 2.0])
    with pytest.raises(SemanticValidationError, match="length-3 array in a 3-D subtree"):
        validate_document_semantics(_doc(tree))


def test_three_d_translate_in_a_two_d_subtree_is_rejected():
    inner = sdf_transform("translate", _CIRCLE_2D, t=[1.0, 2.0, 3.0])
    tree = sdf_2d_to_3d("extrusion", inner, h=1.0)
    with pytest.raises(SemanticValidationError, match="length-2 array in a 2-D subtree"):
        validate_document_semantics(_doc(tree))


def test_two_d_mirror_inside_a_two_d_subtree_passes():
    inner = sdf_transform("mirror", _CIRCLE_2D, n=[1.0, 0.0], o=[0.0, 0.0])
    tree = sdf_2d_to_3d("extrusion", inner, h=1.0)
    validate_document_semantics(_doc(tree))  # must not raise


def test_two_d_mirror_in_a_three_d_subtree_is_rejected():
    tree = sdf_transform("mirror", _SPHERE, n=[1.0, 0.0], o=[0.0, 0.0])
    with pytest.raises(SemanticValidationError, match="length-3 array in a 3-D subtree"):
        validate_document_semantics(_doc(tree))


def test_two_d_repeat_finite_inside_a_two_d_subtree_passes():
    inner = sdf_transform("repeat_finite", _CIRCLE_2D, c=4.0, l=[2.0, 1.0])
    tree = sdf_2d_to_3d("extrusion", inner, h=1.0)
    validate_document_semantics(_doc(tree))  # must not raise


def test_two_d_repeat_finite_in_a_three_d_subtree_is_rejected():
    tree = sdf_transform("repeat_finite", _SPHERE, c=4.0, l=[2.0, 1.0])
    with pytest.raises(SemanticValidationError, match="length-3 array in a 3-D subtree"):
        validate_document_semantics(_doc(tree))


def test_two_d_elongate_inside_a_two_d_subtree_passes():
    inner = sdf_modifier("elongate", _CIRCLE_2D, h=[2.0, 0.5])
    tree = sdf_2d_to_3d("extrusion", inner, h=1.0)
    validate_document_semantics(_doc(tree))  # must not raise


def test_two_d_elongate_in_a_three_d_subtree_is_rejected():
    tree = sdf_modifier("elongate", _SPHERE, h=[2.0, 0.5])
    with pytest.raises(SemanticValidationError, match="length-3 array in a 3-D subtree"):
        validate_document_semantics(_doc(tree))


def test_mat3_slot_given_a_2x2_matrix_is_rejected():
    tree = sdf_transform("rotate_matrix", _SPHERE, R=[[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(SemanticValidationError, match="3x3"):
        validate_document_semantics(_doc(tree))


def test_mat3_slot_correctly_shaped_passes():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    tree = sdf_transform("rotate_matrix", _SPHERE, R=identity)
    validate_document_semantics(_doc(tree))  # must not raise


def test_scalar_list_slot_given_a_bare_number_is_rejected():
    tree = {
        "type": "loft",
        "children": [sdf_primitive("circle_2d", r=1.0), sdf_primitive("circle_2d", r=1.5)],
        "params": {"z": 1.0},
    }
    with pytest.raises(SemanticValidationError, match="list of numbers"):
        validate_document_semantics(_doc(tree))


@pytest.mark.parametrize(
    ("kind", "key", "points"),
    [
        ("polygon_2d", "vertices", [[0.0, 0.0], [1.0, 0.0]]),
        ("bspline_2d", "control_points", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        ("bezier_2d", "control_points", [[0.0, 0.0]] * 5),
    ],
    ids=["polygon_2d_below_min", "bspline_2d_below_min", "bezier_2d_below_min"],
)
def test_point_list_below_its_min_points_is_rejected(kind, key, points):
    tree = sdf_primitive(kind, **{key: points})
    with pytest.raises(SemanticValidationError, match="at least"):
        validate_document_semantics(_doc(tree))


def test_bezier_2d_control_point_count_not_a_multiple_of_3_is_rejected():
    tree = sdf_primitive("bezier_2d", control_points=[[0.0, 0.0]] * 7)  # >= 6, not a multiple of 3
    with pytest.raises(SemanticValidationError, match="multiple of"):
        validate_document_semantics(_doc(tree))


def test_point_list_entry_with_the_wrong_point_dimension_is_rejected():
    tree = sdf_primitive("polygon_2d", vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    with pytest.raises(SemanticValidationError, match="length-2 point"):
        validate_document_semantics(_doc(tree))


@pytest.mark.parametrize(
    ("kind", "key", "points"),
    [
        ("polygon_2d", "vertices", [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        ("bspline_2d", "control_points", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        ("bezier_2d", "control_points", [[0.0, 0.0]] * 6),
    ],
    ids=["polygon_2d_ok", "bspline_2d_ok", "bezier_2d_ok"],
)
def test_point_list_correctly_shaped_passes(kind, key, points):
    # 2-D primitives need a 2-D position; extrusion gives them one.
    tree = sdf_2d_to_3d("extrusion", sdf_primitive(kind, **{key: points}), h=1.0)
    validate_document_semantics(_doc(tree))  # must not raise


def test_nested_field_op_add_passes():
    tree = sdf_deform(
        "displace",
        _SPHERE,
        field=field_op(
            "add", [field_primitive("radial", freq=1.0), field_primitive("angular", freq=2.0)]
        ),
    )
    validate_document_semantics(_doc(tree))  # must not raise


def test_field_op_with_no_children_is_rejected():
    tree = sdf_deform("displace", _SPHERE, field={"type": "field_op", "op": "add", "children": []})
    with pytest.raises(SemanticValidationError, match="children"):
        validate_document_semantics(_doc(tree))


def test_unknown_field_op_name_is_rejected():
    tree = sdf_deform(
        "displace", _SPHERE, field=field_op("subtract", [field_primitive("radial", freq=1.0)])
    )
    with pytest.raises(SemanticValidationError, match="subtract"):
        validate_document_semantics(_doc(tree))


def test_unknown_field_primitive_kind_nested_inside_field_op_is_rejected():
    tree = sdf_deform(
        "displace",
        _SPHERE,
        field=field_op("add", [{"type": "field", "kind": "not_a_field", "params": {}}]),
    )
    with pytest.raises(SemanticValidationError, match="not_a_field"):
        validate_document_semantics(_doc(tree))


# canonical_sector_fold: phase_frac requires centered
# ---------------------------------------------------------------------------


def test_canonical_sector_fold_phase_frac_without_centered_is_rejected():
    tree = sdf_transform("canonical_sector_fold", _SPHERE, n_sectors=6.0, phase_frac=0.25)
    with pytest.raises(SemanticValidationError, match="phase_frac"):
        validate_document_semantics(_doc(tree))


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_canonical_sector_fold_phase_frac_ref_without_centered_is_rejected(schema_version):
    tree = sdf_transform(
        "canonical_sector_fold",
        _SPHERE,
        n_sectors=6.0,
        phase_frac={"$ref": "fold_phase"},
    )
    with pytest.raises(SemanticValidationError, match="phase_frac"):
        validate_document_semantics(
            _doc(tree, params={"fold_phase": {"value": 0.25}}, schema_version=schema_version)
        )


def test_canonical_sector_fold_zero_phase_frac_without_centered_passes():
    tree = sdf_transform("canonical_sector_fold", _SPHERE, n_sectors=6.0, phase_frac=0.0)
    validate_document_semantics(_doc(tree))  # must not raise


def test_canonical_sector_fold_phase_frac_with_centered_passes():
    tree = sdf_transform(
        "canonical_sector_fold",
        _SPHERE,
        n_sectors=6.0,
        centered=True,
        phase_frac=0.25,
    )
    validate_document_semantics(_doc(tree))  # must not raise


def test_softmin_many_without_k_is_rejected():
    tree = sdf_op("softmin_many", [_SPHERE, sdf_primitive("box", b=[1.0, 1.0, 1.0])])
    with pytest.raises(SemanticValidationError, match="missing required parameter"):
        validate_document_semantics(_doc(tree))


def test_softmin_many_with_k_passes():
    tree = sdf_op("softmin_many", [_SPHERE, sdf_primitive("box", b=[1.0, 1.0, 1.0])], k=0.2)
    validate_document_semantics(_doc(tree))  # must not raise


def test_softmin_chunked_with_a_non_numeric_chunk_size_is_rejected():
    tree = sdf_op(
        "softmin_chunked",
        [_SPHERE, sdf_primitive("box", b=[1.0, 1.0, 1.0])],
        k=0.2,
        chunk_size=[1, 2],
    )
    with pytest.raises(SemanticValidationError, match="expected a number"):
        validate_document_semantics(_doc(tree))


def test_softmin_chunked_with_a_valid_chunk_size_passes():
    tree = sdf_op(
        "softmin_chunked",
        [_SPHERE, sdf_primitive("box", b=[1.0, 1.0, 1.0])],
        k=0.2,
        chunk_size=32,
    )
    validate_document_semantics(_doc(tree))  # must not raise


# ---------------------------------------------------------------------------
# extrusion / revolution, sweep, and loft always produce 3-D output
# regardless of their own 2-D children's shape, so nesting one inside
# another's 2-D child slot is exactly as wrong as a bare 3-D primitive
# there -- and was just as undetected before wire.TWO_D_TO_3D / SWEEP /
# LOFT got a fixed dim=3. An op or transform of 2-D children, by contrast,
# is still 2-D: that pass-through is correct and is pinned below too, so
# it can't be "fixed" into a false rejection later.
# ---------------------------------------------------------------------------

_CIRCLE = sdf_primitive("circle_2d", r=1.0)
_STRAIGHT_PATH = [[0.0, 0.0, float(i)] for i in range(4)]


def test_extrusion_nested_inside_an_extrusion_child_is_rejected():
    inner = sdf_2d_to_3d("extrusion", _CIRCLE, h=1.0)
    outer = sdf_2d_to_3d("extrusion", inner, h=2.0)
    with pytest.raises(SemanticValidationError, match="2-D"):
        validate_document_semantics(_doc(outer))


def test_sweep_nested_inside_a_sweep_profile_is_rejected():
    inner = sdf_sweep(_CIRCLE, path=_STRAIGHT_PATH)
    outer = sdf_sweep(inner, path=_STRAIGHT_PATH)
    with pytest.raises(SemanticValidationError, match="2-D"):
        validate_document_semantics(_doc(outer))


def test_loft_nested_inside_a_loft_child_is_rejected():
    inner = sdf_loft([_CIRCLE, _CIRCLE], z=[0.0, 1.0])
    outer = sdf_loft([inner, _CIRCLE], z=[0.0, 1.0])
    with pytest.raises(SemanticValidationError, match="2-D"):
        validate_document_semantics(_doc(outer))


def test_sweep_nested_inside_an_extrusion_child_is_rejected():
    """Cross-nesting between the 2-D-to-3-D lifting kinds is caught the same way."""
    tree = sdf_2d_to_3d("extrusion", sdf_sweep(_CIRCLE, path=_STRAIGHT_PATH), h=2.0)
    with pytest.raises(SemanticValidationError, match="2-D"):
        validate_document_semantics(_doc(tree))


def test_op_of_2d_profiles_inside_extrusion_still_passes():
    """Ops (and transforms) pass expected_dim through unchanged: a union of
    2-D profiles is still 2-D. Correct pass-through behavior, not a hole.
    """
    tree = sdf_2d_to_3d("extrusion", sdf_op("union", [_CIRCLE, _CIRCLE]), h=2.0)
    validate_document_semantics(_doc(tree))  # must not raise
