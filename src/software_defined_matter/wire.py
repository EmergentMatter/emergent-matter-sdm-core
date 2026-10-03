"""The single declaration of the ``.sdm`` wire format.

Every other layer derives from this module or is checked against it:
``sdf/compile.py`` binds its evaluators to the node names declared here,
``sdf/envelope.py`` reads its property sets as columns on the records
below, and ``sdf/validate.py`` walks a document's SDF trees against this
contract before either ever reaches a JAX trace.

Hand-authored, not derived: a leaf evaluator's Python signature cannot
tell a scalar from a per-axis vector from a list of points -- all three
arrive as plain JSON. Only this module says which is which.

``since=`` records the schema_version that introduced a wire field or node
kind. Schemas through 0.2 left SDF discriminator names open, so the original
vocabulary uses the 0.1 baseline. Generated schemas enumerate those names;
new vocabulary added after that boundary carries the version whose schema
first accepts it. Document fields follow the same rule, checked against the
frozen schemas by ``tests/test_wire_contract.py``.

This module must stay importable without JAX: it is read by both the JAX
SDF compiler and the plain-Python semantic validator, and the validator
runs on documents nobody has decided to trace yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, get_args

from software_defined_matter.model import ALLOWED_UNITS, PRIOR_DISTS, ConstraintOp, ObjectiveSense

# ===========================================================================
# Shared vocabulary
# ===========================================================================

#: The shape a wire value takes in JSON, independent of what the runtime
#: evaluator turns it into.
#:
#:   scalar      - a number, or a leaf that resolves to one (``$ref`` /
#:                 an expression subtree)
#:   vec2/vec3   - a fixed-length JSON array of ``scalar`` entries
#:   mat3        - a 3x3 array of arrays of ``scalar`` entries (only
#:                 ``rotate_matrix``'s ``R``)
#:   point_list  - a variable-length array of ``vec2`` or ``vec3`` points
#:   scalar_list - a variable-length array of ``scalar`` entries with no
#:                 fixed point-dimension (only ``loft``'s ``z``)
#:   string      - a literal string, never a ``$ref`` leaf
#:   bool        - a literal boolean, never a ``$ref`` leaf
#:   object      - a free-form JSON object, validated elsewhere (or not
#:                 at all): ``metadata``, ``ui``, ``prior``, an ``expr``
#:                 or ``sdf_tree`` subtree stored under a document field
#:   object_map  - a JSON object whose values are records of a fixed shape,
#:                 keyed by name (``Part.params``)
#:   object_list - a JSON array of records of a fixed shape
#:                 (``Part.materials``, ``Part.history``, ...)
WireShape = Literal[
    "scalar",
    "vec2",
    "vec3",
    # One component per spatial axis of the subtree the node sits in, so its
    # length is 2 or 3 depending on where it appears rather than fixed by the
    # contract. `vec2`/`vec3` are fixed lengths and say nothing about axes:
    # `torus.t` is a vec2 on a 3-D primitive. See TRANSFORMS["translate"].
    "axis_vector",
    "mat3",
    "point_list",
    "scalar_list",
    "string",
    "bool",
    "object",
    "object_map",
    "object_list",
]
WIRE_SHAPES: tuple[WireShape, ...] = get_args(WireShape)

#: Schema versions this contract's ``since=`` fields may name. Kept in sync
#: with ``model.KNOWN_SCHEMA_VERSIONS`` by ``test_wire_contract.py``, not by
#: import, so this module never needs the version-ordering machinery in
#: ``model.py`` to describe a static fact about when a field was added.
SchemaVersion = Literal["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
SCHEMA_VERSIONS: tuple[SchemaVersion, ...] = get_args(SchemaVersion)

#: Versioned additions inside static kinematics schema definitions.
EXPR_CAPABILITIES: dict[str, SchemaVersion] = {
    "expr.unop.sin": "0.5",
    "expr.unop.cos": "0.5",
    "expr.param.instance": "0.5",
    "expr.metric.instance": "0.5",
}

KINEMATICS_CAPABILITIES: dict[str, SchemaVersion] = {
    "kinematics.flexures.blend.radial_hermite": "0.4",
}


#: The ``type`` discriminator values a top-level SDF tree node may carry.
SDFNodeType = Literal[
    "primitive",
    "op",
    "transform",
    "modifier",
    "deform",
    "2d_to_3d",
    "sweep",
    "loft",
]
SDF_NODE_TYPES: tuple[SDFNodeType, ...] = get_args(SDFNodeType)

#: The ``type`` discriminator values a displacement-field tree node may
#: carry. Field trees are structurally parallel to SDF trees but are not
#: SDFs -- their leaf output is a displacement amplitude, not a distance --
#: and only ever appear nested under a ``deform("displace", ...)`` node's
#: ``field`` slot.
FieldNodeType = Literal["field", "field_op"]
FIELD_NODE_TYPES: tuple[FieldNodeType, ...] = get_args(FieldNodeType)

#: The ``type`` discriminator values an expression-tree node may carry.
#:
#: An expression subtree is accepted anywhere a number is expected inside
#: an SDF node's ``params``, alongside a literal number and a ``$ref``
#: leaf. The JSON Schema pins these names with a ``oneOf`` of ``const``\ s
#: under ``$defs/expr``, but only where a document field is *typed* as an
#: expr (an objective's or constraint's ``expr``). In the frozen schemas an
#: ``sdf.params`` object is a bare ``{"type": "object"}``, so a misspelled
#: node type nested in a kwarg reaches no schema check at all there. The
#: current version constrains ``params`` per kind and so does catch it, but
#: most documents in the wild declare ``0.2``.
#:
#: ``sdf/validate.py`` checks membership here and stops. What an ``unop``'s
#: ``op`` may be, or whether a ``binop`` carries both operands, is
#: ``dsl/expr.py``'s contract; this is the shallow name check that catches
#: the typo. Declared here rather than imported from ``dsl/expr.py``
#: because that module imports JAX and this one must not.
#:
#: ``dof`` is the one entry the schema did not always carry -- it arrived
#: at 0.2 with the kinematics block, and ``dsl/expr.py`` does not evaluate
#: it (a kinematics consumer does). It is listed unconditionally anyway,
#: because this gate is version-independent like the rest of the semantic
#: layer, and a gate that rejects on the way in is the one that can produce
#: a false rejection of a valid document. Where the version distinction is
#: enforceable it already is: a ``dof`` node in an objective's ``expr``
#: fails 0.1's JSON Schema, because that field is typed as an expr.
ExprNodeType = Literal["num", "param", "metric", "unop", "binop", "reduce", "dof"]
EXPR_NODE_TYPES: tuple[ExprNodeType, ...] = get_args(ExprNodeType)

#: Objective/Constraint vocabularies are already canonical on the ``Part``
#: dataclasses in ``model.py``; re-exported here rather than re-declared so
#: the enum-drift check has one source for them instead of two that could
#: disagree.
OBJECTIVE_SENSES: tuple[str, ...] = get_args(ObjectiveSense)
CONSTRAINT_OPS: tuple[str, ...] = get_args(ConstraintOp)
UNIT_NAMES = ALLOWED_UNITS
PRIOR_DIST_NAMES = frozenset(PRIOR_DISTS)


def _check_literal(
    value: str, allowed: tuple[str, ...] | frozenset[str], owner: str, field_name: str
) -> None:
    if value not in allowed:
        raise ValueError(f"{owner}: {field_name} must be one of {sorted(allowed)}, got {value!r}.")


# ===========================================================================
# ParamSpec -- one keyword argument of one SDF/field node kind
# ===========================================================================


@dataclass(frozen=True)
class ParamSpec:
    """One entry in an SDF/field node's ``params`` object.

    ``ref_ok`` is ``False`` only for the handful of structural flags that
    the compiler reads with a bare ``dict.get`` instead of routing through
    ``dsl.resolve.resolve_param_value`` -- ``sweep``'s ``path_kind`` /
    ``closed`` / ``frame`` and ``loft``'s ``smooth`` / ``interp``. Every
    other slot accepts a ``{"$ref": "param_name"}`` leaf (or an expression
    subtree) anywhere a number is expected, including per-element inside a
    vector or point list.

    ``choices`` constrains a ``string`` slot to a closed set (e.g.
    ``sweep``'s ``path_kind``). ``point_dim`` / ``min_points`` /
    ``multiple_of`` refine a ``point_list`` slot: each point's dimension,
    the minimum count, and a required multiple.
    """

    name: str
    required: bool
    wire_shape: WireShape
    ref_ok: bool = True
    since: SchemaVersion = "0.1"
    choices: tuple[str, ...] | None = None
    point_dim: Literal[2, 3] | None = None
    min_points: int | None = None
    multiple_of: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParamSpec: name must be non-empty.")
        _check_literal(self.wire_shape, WIRE_SHAPES, f"ParamSpec {self.name!r}", "wire_shape")
        _check_literal(self.since, SCHEMA_VERSIONS, f"ParamSpec {self.name!r}", "since")
        if self.choices is not None and self.wire_shape != "string":
            raise ValueError(
                f"ParamSpec {self.name!r}: choices is only meaningful for wire_shape='string', "
                f"got wire_shape={self.wire_shape!r}."
            )
        point_only = (self.point_dim, self.min_points, self.multiple_of)
        if self.wire_shape != "point_list" and any(v is not None for v in point_only):
            raise ValueError(
                f"ParamSpec {self.name!r}: point_dim/min_points/multiple_of are only "
                f"meaningful for wire_shape='point_list', got wire_shape={self.wire_shape!r}."
            )


# ===========================================================================
# NodeSpec -- one named SDF/field node kind (a primitive, an op, ...)
# ===========================================================================

NodeCategory = Literal[
    "primitive",
    "op",
    "transform",
    "modifier",
    "deform",
    "2d_to_3d",
    "sweep",
    "loft",
    "field_primitive",
    "field_op",
]
NODE_CATEGORIES: tuple[NodeCategory, ...] = get_args(NodeCategory)

#: Category -> the property whose semantic meaning is column-only for that
#: category. Used solely by ``NodeSpec.__post_init__`` to reject a flag set
#: on the wrong kind of node; not consulted by callers.
_COLUMN_OWNER: dict[str, NodeCategory] = {
    "is_lattice": "primitive",
    "removes_material": "op",
    "hollows": "modifier",
    "tiles": "transform",
    "requires_field": "deform",
}


@dataclass(frozen=True)
class NodeSpec:
    """One named SDF or field node kind, e.g. ``sphere``, ``smooth_union``,
    ``onion``.

    The boolean columns fold in the property sets that used to live as
    separate frozensets in ``sdf/envelope.py``:

      - ``is_lattice``: a triply-periodic primitive with no outer surface
        of its own; its envelope is the box it was designed to fill.
      - ``removes_material``: a CSG op whose non-first children are
        porosity, not solid (``subtract`` / ``smooth_subtract``).
      - ``hollows``: a modifier that turns a solid into a shell
        (``onion``); its envelope is the solid it was hollowed from.
      - ``tiles``: a transform that repeats a child through a finite
        domain (``repeat_finite``); its envelope is that domain.

    Each column is meaningful only for the category it names in
    ``_COLUMN_OWNER``; set on any other category it raises.
    """

    name: str
    category: NodeCategory
    params: tuple[ParamSpec, ...] = ()
    since: SchemaVersion = "0.1"
    dim: Literal[2, 3] | None = None
    is_lattice: bool = False
    removes_material: bool = False
    hollows: bool = False
    tiles: bool = False
    requires_field: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("NodeSpec: name must be non-empty.")
        _check_literal(self.category, NODE_CATEGORIES, f"NodeSpec {self.name!r}", "category")
        _check_literal(self.since, SCHEMA_VERSIONS, f"NodeSpec {self.name!r}", "since")
        seen: set[str] = set()
        for p in self.params:
            if p.name in seen:
                raise ValueError(f"NodeSpec {self.name!r}: duplicate param {p.name!r}.")
            seen.add(p.name)
        for column, owner in _COLUMN_OWNER.items():
            if getattr(self, column) and self.category != owner:
                raise ValueError(
                    f"NodeSpec {self.name!r}: {column} is only meaningful for "
                    f"category={owner!r}, got category={self.category!r}."
                )

    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def required_param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params if p.required)

    def get_param(self, name: str) -> ParamSpec | None:
        for p in self.params:
            if p.name == name:
                return p
        return None


# ===========================================================================
# 3-D primitives
# ===========================================================================
# fmt: off
PRIMITIVES: dict[str, NodeSpec] = {
    "sphere": NodeSpec("sphere", "primitive", dim=3, params=(
        ParamSpec("r", True, "scalar"),
    )),
    "box": NodeSpec("box", "primitive", dim=3, params=(
        ParamSpec("b", True, "vec3"),
    )),
    "round_box": NodeSpec("round_box", "primitive", dim=3, params=(
        ParamSpec("b", True, "vec3"),
        ParamSpec("r", True, "scalar"),
    )),
    "box_frame": NodeSpec("box_frame", "primitive", dim=3, params=(
        ParamSpec("b", True, "vec3"),
        ParamSpec("e", True, "scalar"),
    )),
    "torus": NodeSpec("torus", "primitive", dim=3, params=(
        ParamSpec("t", True, "vec2"),
    )),
    "capped_torus": NodeSpec("capped_torus", "primitive", dim=3, params=(
        ParamSpec("sc", True, "vec2"),
        ParamSpec("ra", True, "scalar"),
        ParamSpec("rb", True, "scalar"),
    )),
    "helix": NodeSpec("helix", "primitive", dim=3, params=(
        ParamSpec("major_r", True, "scalar"),
        ParamSpec("pitch", True, "scalar"),
        ParamSpec("r", True, "scalar"),
        ParamSpec("n_turns", True, "scalar"),
        ParamSpec("phase", False, "scalar"),
        ParamSpec("handedness", False, "scalar"),
    )),
    "screw_thread": NodeSpec("screw_thread", "primitive", dim=3, params=(
        ParamSpec("r_root", True, "scalar"),
        ParamSpec("depth", True, "scalar"),
        ParamSpec("pitch", True, "scalar"),
        ParamSpec("width", True, "scalar"),
        ParamSpec("n_turns", True, "scalar"),
        ParamSpec("phase", False, "scalar"),
        ParamSpec("handedness", False, "scalar"),
        ParamSpec("flank_deg", False, "scalar"),
    )),
    "link": NodeSpec("link", "primitive", dim=3, params=(
        ParamSpec("le", True, "scalar"),
        ParamSpec("r1", True, "scalar"),
        ParamSpec("r2", True, "scalar"),
    )),
    "cone": NodeSpec("cone", "primitive", dim=3, params=(
        ParamSpec("c", True, "vec2"),
        ParamSpec("h", True, "scalar"),
    )),
    "plane": NodeSpec("plane", "primitive", dim=3, params=(
        ParamSpec("n", True, "vec3"),
        ParamSpec("h", True, "scalar"),
    )),
    # Sampled distance grid (DR-0003). Topology-class slots are ref_ok=False:
    # changing them selects how many samples exist, so they require re-emit.
    # ``spacing`` is authored as vec3 (scalar broadcasts happen in the builder).
    "raster_field": NodeSpec("raster_field", "primitive", dim=3, params=(
        ParamSpec("origin", True, "vec3", ref_ok=False),
        ParamSpec("spacing", True, "vec3", ref_ok=False),
        ParamSpec("dims", True, "vec3", ref_ok=False),
        ParamSpec("encoding", True, "string", ref_ok=False, choices=("f32le", "f16le")),
        ParamSpec("data", True, "string", ref_ok=False),
        ParamSpec("step_scale", False, "scalar", ref_ok=False),
        ParamSpec("provenance", False, "object", ref_ok=False),
    )),
    "hex_prism": NodeSpec("hex_prism", "primitive", dim=3, params=(
        ParamSpec("h", True, "vec2"),
    )),
    "tri_prism": NodeSpec("tri_prism", "primitive", dim=3, params=(
        ParamSpec("h", True, "vec2"),
    )),
    "capsule": NodeSpec("capsule", "primitive", dim=3, params=(
        ParamSpec("a", True, "vec3"),
        ParamSpec("b", True, "vec3"),
        ParamSpec("r", True, "scalar"),
    )),
    "capped_cylinder": NodeSpec("capped_cylinder", "primitive", dim=3, params=(
        ParamSpec("h", True, "scalar"),
        ParamSpec("r", True, "scalar"),
    )),
    "rounded_cylinder": NodeSpec("rounded_cylinder", "primitive", dim=3, params=(
        ParamSpec("ra", True, "scalar"),
        ParamSpec("rb", True, "scalar"),
        ParamSpec("h", True, "scalar"),
    )),
    "capped_cone": NodeSpec("capped_cone", "primitive", dim=3, params=(
        ParamSpec("h", True, "scalar"),
        ParamSpec("r1", True, "scalar"),
        ParamSpec("r2", True, "scalar"),
    )),
    "solid_angle": NodeSpec("solid_angle", "primitive", dim=3, params=(
        ParamSpec("c", True, "vec2"),
        ParamSpec("ra", True, "scalar"),
    )),
    "cut_sphere": NodeSpec("cut_sphere", "primitive", dim=3, params=(
        ParamSpec("r", True, "scalar"),
        ParamSpec("h", True, "scalar"),
    )),
    "ellipsoid": NodeSpec("ellipsoid", "primitive", dim=3, params=(
        ParamSpec("r", True, "vec3"),
    )),
    "octahedron": NodeSpec("octahedron", "primitive", dim=3, params=(
        ParamSpec("s", True, "scalar"),
    )),
    "pyramid": NodeSpec("pyramid", "primitive", dim=3, params=(
        ParamSpec("h", True, "scalar"),
    )),
    # -- triply-periodic minimal-surface lattices ---------------------------
    "gyroid": NodeSpec("gyroid", "primitive", dim=3, is_lattice=True, params=(
        ParamSpec("period", True, "scalar"),
        ParamSpec("min_thickness", True, "scalar"),
        ParamSpec("n_periods", True, "vec3"),
    )),
    "schwarz_p": NodeSpec("schwarz_p", "primitive", dim=3, is_lattice=True, params=(
        ParamSpec("period", True, "scalar"),
        ParamSpec("min_thickness", True, "scalar"),
        ParamSpec("n_periods", True, "vec3"),
    )),
    "schwarz_d": NodeSpec("schwarz_d", "primitive", dim=3, is_lattice=True, params=(
        ParamSpec("period", True, "scalar"),
        ParamSpec("min_thickness", True, "scalar"),
        ParamSpec("n_periods", True, "vec3"),
    )),
    "neovius": NodeSpec("neovius", "primitive", dim=3, is_lattice=True, params=(
        ParamSpec("period", True, "scalar"),
        ParamSpec("min_thickness", True, "scalar"),
        ParamSpec("n_periods", True, "vec3"),
    )),
    "lidinoid": NodeSpec("lidinoid", "primitive", dim=3, is_lattice=True, params=(
        ParamSpec("period", True, "scalar"),
        ParamSpec("min_thickness", True, "scalar"),
        ParamSpec("n_periods", True, "vec3"),
    )),
    # -- compliant mechanisms ------------------------------------------------
    "notch_hinge": NodeSpec("notch_hinge", "primitive", dim=3, params=(
        ParamSpec("width", True, "scalar"),
        ParamSpec("depth", True, "scalar"),
        ParamSpec("notch_radius", True, "scalar"),
    )),
    "leaf_spring": NodeSpec("leaf_spring", "primitive", dim=3, params=(
        ParamSpec("length", True, "scalar"),
        ParamSpec("width", True, "scalar"),
        ParamSpec("thickness", True, "scalar"),
    )),
    "bellows": NodeSpec("bellows", "primitive", dim=3, params=(
        ParamSpec("outer_r", True, "scalar"),
        ParamSpec("inner_r", True, "scalar"),
        ParamSpec("period", True, "scalar"),
        ParamSpec("n_periods", True, "scalar"),
    )),
    "serpentine": NodeSpec("serpentine", "primitive", dim=3, params=(
        ParamSpec("amplitude", True, "scalar"),
        ParamSpec("wavelength", True, "scalar"),
        ParamSpec("beam_width", True, "scalar"),
        ParamSpec("beam_height", True, "scalar"),
        ParamSpec("n_periods", True, "scalar"),
    )),
    "annular_sector": NodeSpec("annular_sector", "primitive", dim=3, params=(
        ParamSpec("inner_r", True, "scalar"),
        ParamSpec("outer_r", True, "scalar"),
        ParamSpec("half_angle", True, "scalar"),
        ParamSpec("height", True, "scalar"),
    )),
    # -- 2-D (used inside extrusion / revolution / sweep / loft) -------------
    "circle_2d": NodeSpec("circle_2d", "primitive", dim=2, params=(
        ParamSpec("r", True, "scalar"),
    )),
    "box_2d": NodeSpec("box_2d", "primitive", dim=2, params=(
        ParamSpec("b", True, "vec2"),
    )),
    "rounded_box_2d": NodeSpec("rounded_box_2d", "primitive", dim=2, params=(
        ParamSpec("b", True, "vec2"),
        ParamSpec("r", True, "scalar"),
    )),
    "segment_2d": NodeSpec("segment_2d", "primitive", dim=2, params=(
        ParamSpec("a", True, "vec2"),
        ParamSpec("b", True, "vec2"),
    )),
    "trapezoid_2d": NodeSpec("trapezoid_2d", "primitive", dim=2, params=(
        ParamSpec("r1", True, "scalar"),
        ParamSpec("r2", True, "scalar"),
        ParamSpec("he", True, "scalar"),
    )),
    "uneven_capsule_2d": NodeSpec("uneven_capsule_2d", "primitive", dim=2, params=(
        ParamSpec("r1", True, "scalar"),
        ParamSpec("r2", True, "scalar"),
        ParamSpec("h", True, "scalar"),
    )),
    "polygon_2d": NodeSpec("polygon_2d", "primitive", dim=2, params=(
        ParamSpec("vertices", True, "point_list", point_dim=2, min_points=3),
    )),
    "bezier_2d": NodeSpec("bezier_2d", "primitive", dim=2, params=(
        ParamSpec("control_points", True, "point_list", point_dim=2, min_points=6, multiple_of=3),
    )),
    "bspline_2d": NodeSpec("bspline_2d", "primitive", dim=2, params=(
        ParamSpec("control_points", True, "point_list", point_dim=2, min_points=4),
    )),
}
# fmt: on

#: Primitives with no outer surface of their own -- ``sdf/envelope.py``
#: replaces each with the box it was designed to fill.
LATTICE_PRIMITIVES = frozenset(name for name, spec in PRIMITIVES.items() if spec.is_lattice)

# ===========================================================================
# CSG ops
# ===========================================================================
# fmt: off
OPS: dict[str, NodeSpec] = {
    "union": NodeSpec("union", "op"),
    "subtract": NodeSpec("subtract", "op", removes_material=True),
    "intersect": NodeSpec("intersect", "op"),
    "smooth_union": NodeSpec("smooth_union", "op", params=(
        ParamSpec("k", True, "scalar"),
    )),
    "smooth_subtract": NodeSpec("smooth_subtract", "op", removes_material=True, params=(
        ParamSpec("k", True, "scalar"),
    )),
    "smooth_intersect": NodeSpec("smooth_intersect", "op", params=(
        ParamSpec("k", True, "scalar"),
    )),
    "softmin_many": NodeSpec("softmin_many", "op", params=(
        ParamSpec("k", True, "scalar"),
    )),
    "softmin_chunked": NodeSpec("softmin_chunked", "op", params=(
        ParamSpec("k", True, "scalar"),
        ParamSpec("chunk_size", False, "scalar"),
    )),
}
# fmt: on

#: Ops whose non-first children are removed material, not solid --
#: ``sdf/envelope.py`` measures only the first child for these.
SUBTRACT_OPS = frozenset(name for name, spec in OPS.items() if spec.removes_material)

# ===========================================================================
# Transforms
# ===========================================================================
# fmt: off
TRANSFORMS: dict[str, NodeSpec] = {
    # A per-axis vector has one component per spatial axis of the subtree it
    # sits in, not three. `_check_axis_vector` enforces exactly that on the JAX
    # side and the emitter picks the vec2 helper at dim=2, but the contract
    # declared vec3 unconditionally and so refused every 2-D translate, mirror,
    # repeat_finite and elongate in the wild. "axis_vector" defers the length to the
    # subtree, where `validate.py` pins it from `expected_dim`. It is not "either
    # length is fine": within one subtree exactly one length is correct.
    "translate": NodeSpec("translate", "transform", params=(
        ParamSpec("t", True, "axis_vector"),
    )),
    "scale": NodeSpec("scale", "transform", params=(
        ParamSpec("s", True, "scalar"),
    )),
    # Per-axis scale. Compiled as `child(p / s) * min(s)`: dividing by the
    # smallest factor keeps the result a distance BOUND (exact only when all
    # factors are equal). `s` is per-axis like translate.t, so it works in 2-D.
    "scale_axis": NodeSpec("scale_axis", "transform", since="0.6", params=(
        ParamSpec("s", True, "axis_vector", since="0.6"),
    )),
    "rotate_x": NodeSpec("rotate_x", "transform", params=(
        ParamSpec("angle", True, "scalar"),
    )),
    "rotate_y": NodeSpec("rotate_y", "transform", params=(
        ParamSpec("angle", True, "scalar"),
    )),
    "rotate_z": NodeSpec("rotate_z", "transform", params=(
        ParamSpec("angle", True, "scalar"),
    )),
    "rotate_matrix": NodeSpec("rotate_matrix", "transform", params=(
        ParamSpec("R", True, "mat3"),
    )),
    # Schema-legal, never compiled (a shape that tiles forever has no
    # finite closure to trace); param shape inferred from its one usage in
    # tests/test_sdf_envelope.py, with no implementation to check it against.
    "repeat_inf": NodeSpec("repeat_inf", "transform", params=(
        ParamSpec("c", True, "vec3"),
    )),
    "repeat_finite": NodeSpec("repeat_finite", "transform", tiles=True, params=(
        ParamSpec("c", True, "scalar"),
        ParamSpec("l", True, "axis_vector"),
    )),
    # `centered` folds into [-sector/2, sector/2) instead of [0, sector), so a
    # child authored at angle 0 sits mid-wedge whatever n_sectors is.
    # `phase_frac` then offsets the copies by a fraction of one sector (a gear
    # pair uses 0 and 0.5).
    "canonical_sector_fold": NodeSpec("canonical_sector_fold", "transform", params=(
        ParamSpec("n_sectors", True, "scalar"),
        ParamSpec("centered", False, "bool", ref_ok=False),
        ParamSpec("phase_frac", False, "scalar"),
    )),
    # Bilateral symmetry: the child unioned with its reflection across
    # the plane (n, o).
    "mirror": NodeSpec("mirror", "transform", params=(
        ParamSpec("n", True, "axis_vector"),
        ParamSpec("o", True, "axis_vector"),
    )),
}
# fmt: on

#: Transforms that tile a child through a finite domain -- ``sdf/envelope.py``
#: replaces each with that domain, for the same reason as the lattice
#: primitives above.
TILING_TRANSFORMS = frozenset(name for name, spec in TRANSFORMS.items() if spec.tiles)

# ===========================================================================
# Modifiers
# ===========================================================================
MODIFIERS: dict[str, NodeSpec] = {
    "round": NodeSpec(
        "round",
        "modifier",
        params=(ParamSpec("r", True, "scalar"),),
    ),
    "onion": NodeSpec(
        "onion",
        "modifier",
        hollows=True,
        params=(ParamSpec("thickness", True, "scalar"),),
    ),
    # `h` is a per-axis vector, same as translate.t -- see TRANSFORMS.
    "elongate": NodeSpec(
        "elongate",
        "modifier",
        params=(ParamSpec("h", True, "axis_vector"),),
    ),
}

#: Modifiers that hollow the child out -- ``sdf/envelope.py`` drops the
#: modifier and measures the solid it was hollowed from.
HOLLOWING_MODIFIERS = frozenset(name for name, spec in MODIFIERS.items() if spec.hollows)

# ===========================================================================
# Deforms
# ===========================================================================
DEFORMS: dict[str, NodeSpec] = {
    "twist": NodeSpec(
        "twist",
        "deform",
        params=(ParamSpec("k", True, "scalar"),),
    ),
    "bend": NodeSpec(
        "bend",
        "deform",
        params=(ParamSpec("k", True, "scalar"),),
    ),
    "twist_radial": NodeSpec(
        "twist_radial",
        "deform",
        params=(
            ParamSpec("r0", True, "scalar"),
            ParamSpec("r1", True, "scalar"),
            ParamSpec("angle_inner", True, "scalar"),
            ParamSpec("angle_outer", True, "scalar"),
        ),
    ),
    "twist_linear": NodeSpec(
        "twist_linear",
        "deform",
        params=(
            # Optional: the ramp direction in XY, defaulting to +X. Every other
            # slot is required -- a ramp with no ends is not a ramp.
            ParamSpec("axis", False, "vec2"),
            ParamSpec("u0", True, "scalar"),
            ParamSpec("u1", True, "scalar"),
            ParamSpec("angle_0", True, "scalar"),
            ParamSpec("angle_1", True, "scalar"),
        ),
    ),
    # A clamped-ramp SHEAR along Z: the translation sibling of twist_linear.
    # `dz_0` / `dz_1` are how far the MATERIAL rises at each end of the ramp;
    # between them it rises linearly, beyond them it holds. Params are read
    # per frame by consumers that lift the warp above a baked grid, so they are
    # literals or bare $refs, the same rule as twist_linear's.
    "shear_linear": NodeSpec(
        "shear_linear",
        "deform",
        since="0.6",
        params=(
            ParamSpec("axis", False, "vec2", since="0.6"),
            ParamSpec("u0", True, "scalar", since="0.6"),
            ParamSpec("u1", True, "scalar", since="0.6"),
            ParamSpec("dz_0", True, "scalar", since="0.6"),
            ParamSpec("dz_1", True, "scalar", since="0.6"),
        ),
    ),
    # XY scale ramped linearly along Z: s_0 up to z0, s_1 from z1 on. A cone
    # from a cylinder, a draft angle from a prism. Factors must stay positive.
    "taper_linear": NodeSpec(
        "taper_linear",
        "deform",
        since="0.6",
        params=(
            ParamSpec("z0", True, "scalar", since="0.6"),
            ParamSpec("z1", True, "scalar", since="0.6"),
            ParamSpec("s_0", True, "scalar", since="0.6"),
            ParamSpec("s_1", True, "scalar", since="0.6"),
        ),
    ),
    "displace": NodeSpec(
        "displace",
        "deform",
        requires_field=True,
    ),
}

# ===========================================================================
# 2-D -> 3-D lifts
# ===========================================================================
# dim=3 on both: a 2d_to_3d node always produces 3-D output regardless of
# its own child's dimension, so nesting one in another's 2-D slot is wrong
# the same way a bare 3-D primitive there is.
TWO_D_TO_3D: dict[str, NodeSpec] = {
    "revolution": NodeSpec(
        "revolution",
        "2d_to_3d",
        dim=3,
        params=(ParamSpec("offset", False, "scalar"),),
    ),
    "extrusion": NodeSpec(
        "extrusion",
        "2d_to_3d",
        dim=3,
        params=(ParamSpec("h", True, "scalar"),),
    ),
}

# ===========================================================================
# Sweep and loft -- single node kinds, not keyed by a sub-name
# ===========================================================================

#: A 2-D profile swept along a 3-D path. ``path_kind`` / ``closed`` /
#: ``frame`` are read as plain Python strings/bools by the compiler
#: (``dict.get`` on the raw node, never through the ``$ref`` resolver), so
#: they cannot carry a param reference.
SWEEP = NodeSpec(
    "sweep",
    "sweep",
    dim=3,
    params=(
        ParamSpec("path", True, "point_list", point_dim=3, min_points=2),
        ParamSpec(
            "path_kind", False, "string", ref_ok=False, choices=("bspline", "bezier", "polyline")
        ),
        ParamSpec("closed", False, "bool", ref_ok=False),
        ParamSpec("frame", False, "string", ref_ok=False, choices=("rmf", "cylindrical")),
        ParamSpec("normal0", False, "vec3"),
    ),
)

#: N 2-D cross-section profiles lofted along Z. ``smooth`` / ``interp`` are
#: structural flags for the same reason as ``sweep``'s ``path_kind`` above.
LOFT = NodeSpec(
    "loft",
    "loft",
    dim=3,
    params=(
        ParamSpec("z", True, "scalar_list"),
        ParamSpec("smooth", False, "bool", ref_ok=False),
        ParamSpec("interp", False, "string", ref_ok=False, choices=("field", "shape")),
    ),
)

# ===========================================================================
# Displacement-field primitives and ops (sdf_deform("displace", ...))
# ===========================================================================
FIELD_PRIMITIVES: dict[str, NodeSpec] = {
    "sin_xyz": NodeSpec(
        "sin_xyz",
        "field_primitive",
        params=(
            ParamSpec("freq", True, "vec3"),
            ParamSpec("amplitude", False, "scalar"),
            ParamSpec("phase", False, "vec3"),
        ),
    ),
    "radial": NodeSpec(
        "radial",
        "field_primitive",
        params=(
            ParamSpec("freq", True, "scalar"),
            ParamSpec("amplitude", False, "scalar"),
            ParamSpec("phase", False, "scalar"),
        ),
    ),
    "angular": NodeSpec(
        "angular",
        "field_primitive",
        params=(
            ParamSpec("freq", True, "scalar"),
            ParamSpec("amplitude", False, "scalar"),
            ParamSpec("phase", False, "scalar"),
        ),
    ),
}

FIELD_OPS: dict[str, NodeSpec] = {
    "add": NodeSpec("add", "field_op"),
}

#: ``type`` -> registry, for the node kinds that dispatch on a sub-name.
#: ``sweep`` and ``loft`` are single kinds (dispatched purely on ``type``)
#: and are looked up directly via :data:`SWEEP` / :data:`LOFT` instead.
NODE_REGISTRIES: dict[str, dict[str, NodeSpec]] = {
    "primitive": PRIMITIVES,
    "op": OPS,
    "transform": TRANSFORMS,
    "modifier": MODIFIERS,
    "deform": DEFORMS,
    "2d_to_3d": TWO_D_TO_3D,
}

# ===========================================================================
# Document structure -- Part and its nested dataclasses in model.py
# ===========================================================================


@dataclass(frozen=True)
class FieldSpec:
    """One attribute of a document-structure dataclass (``Part``, ``Param``,
    ...). Mirrors :class:`ParamSpec` without the SDF-specific columns that
    don't apply to a document field (``ref_ok``, ``choices``, the point-list
    refinements): a document field is never itself a ``$ref`` target.
    """

    name: str
    required: bool
    wire_shape: WireShape
    since: SchemaVersion = "0.1"
    #: True for a wire field with no dataclass counterpart, written and read
    #: by tooling outside this package. It belongs to the wire format, so it
    #: is declared here, but it never round-trips through the model.
    wire_only: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FieldSpec: name must be non-empty.")
        _check_literal(self.wire_shape, WIRE_SHAPES, f"FieldSpec {self.name!r}", "wire_shape")
        _check_literal(self.since, SCHEMA_VERSIONS, f"FieldSpec {self.name!r}", "since")


PART_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("kind", False, "string", since="0.5", wire_only=True),
    FieldSpec("name", True, "string"),
    FieldSpec("params", False, "object_map"),
    FieldSpec("materials", False, "object_list"),
    FieldSpec("ports", False, "object_list", since="0.5"),
    FieldSpec("objectives", False, "object_list"),
    FieldSpec("constraints", False, "object_list"),
    FieldSpec("metadata", False, "object"),
    FieldSpec("history", False, "object_list"),
    FieldSpec("kinematics", False, "object", since="0.2"),
)

PARAM_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("name", True, "string"),
    FieldSpec("value", True, "scalar"),
    FieldSpec("free", False, "bool"),
    FieldSpec("bounds", False, "vec2"),
    FieldSpec("unit", True, "string"),
    FieldSpec("ui", False, "object"),
    # Probabilistic param content: since=0.3, the one place in the document
    # contract where a field genuinely postdates the wire baseline.
    FieldSpec("prior", False, "object", since="0.3"),
    FieldSpec("tolerance", False, "scalar", since="0.3"),
    FieldSpec("expr", False, "object", since="0.3"),
)

MATERIAL_REGION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("material_id", True, "scalar"),
    FieldSpec("name", True, "string"),
    FieldSpec("sdf_tree", True, "object"),
)

PORT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("name", True, "string", since="0.5"),
    FieldSpec("frame", True, "object", since="0.5"),
    FieldSpec("body", False, "string", since="0.5"),
    FieldSpec("domains", False, "object", since="0.5"),
    FieldSpec("sdf_tree", False, "object", since="0.5"),
    FieldSpec("metadata", False, "object", since="0.5"),
)

ASSEMBLY_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("kind", True, "string", since="0.5", wire_only=True),
) + tuple(
    FieldSpec(name, name == "name", cast(WireShape, shape), since="0.5")
    for name, shape in (
        ("name", "string"),
        ("params", "object_map"),
        ("instances", "object_list"),
        ("port", "object_map"),
        ("dofs", "object_map"),
        ("motion_inputs", "object_map"),
        ("objectives", "object_list"),
        ("constraints", "object_list"),
        ("metadata", "object"),
    )
)

ASSEMBLY_FIELDS += (FieldSpec("mates", False, "object_list", since="0.5"),)
MATE_FIELDS = tuple(
    FieldSpec(name, name in {"id", "kind", "parent", "child"}, cast(WireShape, shape), since="0.5")
    for name, shape in (
        ("id", "string"),
        ("kind", "string"),
        ("parent", "string"),
        ("child", "string"),
        ("dof", "string"),
        ("offset", "object"),
    )
)

FRAME_FIELDS = (
    FieldSpec("position", True, "vec3", since="0.5"),
    FieldSpec("orientation", True, "object_list", since="0.5"),
)
PART_REF_FIELDS = (
    FieldSpec("path", True, "string", since="0.5"),
    FieldSpec("content_hash", False, "string", since="0.5"),
)
INSTANCE_FIELDS = tuple(
    FieldSpec(name, name in {"id", "part_ref"}, cast(WireShape, shape), since="0.5")
    for name, shape in (
        ("id", "string"),
        ("part_ref", "object"),
        ("param_overrides", "object_map"),
        ("dof_bindings", "object_map"),
        ("transform", "object"),
    )
)
DOF_FIELDS = tuple(
    FieldSpec(name, name != "default", cast(WireShape, shape), since="0.5")
    for name, shape in (
        ("kind", "string"),
        ("range", "vec2"),
        ("unit", "string"),
        ("default", "scalar"),
    )
)


OBJECTIVE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("name", True, "string"),
    FieldSpec("sense", True, "string"),
    FieldSpec("expr", True, "object"),
    FieldSpec("weight", False, "scalar"),
)

CONSTRAINT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("name", True, "string"),
    FieldSpec("expr", True, "object"),
    FieldSpec("op", True, "string"),
    FieldSpec("rhs", True, "scalar"),
    FieldSpec("tolerance", False, "scalar"),
)

#: Dataclass name (as it appears on ``model.py``) -> its field contract.
#: Entries marked ``wire_only`` have no dataclass counterpart and are skipped
#: when the contract is checked against ``dataclasses.fields()``.
DOCUMENT_FIELDS: dict[str, tuple[FieldSpec, ...]] = {
    "Part": PART_FIELDS,
    "Param": PARAM_FIELDS,
    "MaterialRegion": MATERIAL_REGION_FIELDS,
    "Port": PORT_FIELDS,
    "Frame": FRAME_FIELDS,
    "Assembly": ASSEMBLY_FIELDS,
    "Mate": MATE_FIELDS,
    "PartRef": PART_REF_FIELDS,
    "Instance": INSTANCE_FIELDS,
    "Dof": DOF_FIELDS,
    "Objective": OBJECTIVE_FIELDS,
    "Constraint": CONSTRAINT_FIELDS,
}


__all__ = [
    "CONSTRAINT_FIELDS",
    "CONSTRAINT_OPS",
    "PORT_FIELDS",
    "FRAME_FIELDS",
    "ASSEMBLY_FIELDS",
    "PART_REF_FIELDS",
    "INSTANCE_FIELDS",
    "DOF_FIELDS",
    "DEFORMS",
    "DOCUMENT_FIELDS",
    "EXPR_NODE_TYPES",
    "EXPR_CAPABILITIES",
    "FIELD_NODE_TYPES",
    "FIELD_OPS",
    "FIELD_PRIMITIVES",
    "HOLLOWING_MODIFIERS",
    "LATTICE_PRIMITIVES",
    "LOFT",
    "MATERIAL_REGION_FIELDS",
    "MODIFIERS",
    "NODE_CATEGORIES",
    "NODE_REGISTRIES",
    "OBJECTIVE_FIELDS",
    "OBJECTIVE_SENSES",
    "OPS",
    "PART_FIELDS",
    "PARAM_FIELDS",
    "PRIMITIVES",
    "PRIOR_DIST_NAMES",
    "SCHEMA_VERSIONS",
    "SDF_NODE_TYPES",
    "SUBTRACT_OPS",
    "SWEEP",
    "TILING_TRANSFORMS",
    "TRANSFORMS",
    "TWO_D_TO_3D",
    "UNIT_NAMES",
    "WIRE_SHAPES",
    "ExprNodeType",
    "FieldNodeType",
    "FieldSpec",
    "NodeCategory",
    "NodeSpec",
    "ParamSpec",
    "SDFNodeType",
    "SchemaVersion",
    "KINEMATICS_CAPABILITIES",
    "WireShape",
]
