"""Semantic checks on SDF trees that the JSON Schema cannot express.

The JSON Schema validates structure; it cannot enforce context. Two kinds
of context live here:

  - **Position.** A primitive can be well-formed alone and still make the
    tree unbounded where it sits -- today only
    :func:`software_defined_matter.sdf.sdf_shapes.plane`, a CSG cutting
    tool that is otherwise infinite. :func:`validate_material_tree` flags
    a raw plane at a material region's root.

  - **Vocabulary.** ``primitive.kind`` / ``op.op`` / ``modifier.modifier``
    / ``deform.deform`` and their ``params`` are almost unconstrained by
    the frozen schemas. :func:`validate_document_semantics` checks every
    SDF node against the contract in
    :mod:`software_defined_matter.wire`, independent of the document's
    declared schema_version -- an unknown kind was never valid on any
    version; the older ones just had no way to say so. The current
    version, generated from that same contract, does gate these names, but
    a check that only ran there would miss the documents that actually
    exist: most in the wild declare ``0.2``.

  - **Contextual kwargs.** Some params only apply when another is set --
    ``canonical_sector_fold``'s ``phase_frac`` (when not the default zero)
    requires ``centered: true``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from software_defined_matter import wire

if TYPE_CHECKING:
    from software_defined_matter.model import Part


class UnboundedRootError(ValueError):
    """A material region's SDF tree resolves to an unbounded shape."""


class SemanticValidationError(ValueError):
    """A document's SDF tree uses a kind / modifier / deform name, a kwarg,
    or a ``$ref`` that :mod:`software_defined_matter.wire` does not allow.

    Raised by :func:`validate_document_semantics`, independent of the
    document's declared ``schema_version``.
    """


def validate_part_material_trees(part: Part) -> None:
    """Validate every :class:`MaterialRegion` on ``part`` (no-op if none)."""
    for region in part.materials:
        validate_material_tree(region.sdf_tree, region_name=region.name)


def validate_material_tree(tree: dict[str, Any], *, region_name: str = "") -> None:
    """Raise :class:`UnboundedRootError` if ``tree`` cannot be a finite solid.

    Walks the tree and flags ``plane`` primitives that appear in positions
    where they would leave the result unbounded:

      - As the root of the material region.
      - As any child of ``union`` / ``smooth_union``.
      - As the first child (minuend) of ``subtract`` / ``smooth_subtract``.
      - Inside ``modifier``, ``transform``, ``deform``, or ``2d_to_3d``
        nodes (those preserve unboundedness of the wrapped child).

    Allowed positions (where a bounded sibling clamps the result):
      - As a non-first child of ``subtract`` / ``smooth_subtract``.
      - As any child of ``intersect`` / ``smooth_intersect``.
    """
    _walk(tree, can_be_unbounded=False, region_name=region_name)


def _walk(node: Any, *, can_be_unbounded: bool, region_name: str) -> None:
    if not isinstance(node, dict) or "type" not in node:
        return  # Malformed; downstream compile/schema will catch it.

    node_type = node["type"]

    if node_type == "primitive":
        if node.get("kind") == "plane" and not can_be_unbounded:
            raise UnboundedRootError(_plane_message(region_name))
        return

    if node_type == "op":
        op = node.get("op", "")
        children = node.get("children", []) or []
        if op in ("union", "smooth_union"):
            for c in children:
                _walk(c, can_be_unbounded=False, region_name=region_name)
        elif op in ("subtract", "smooth_subtract"):
            if children:
                _walk(children[0], can_be_unbounded=False, region_name=region_name)
                for c in children[1:]:
                    _walk(c, can_be_unbounded=True, region_name=region_name)
        elif op in ("intersect", "smooth_intersect"):
            # A single bounded child clamps the result; we permit any
            # individual child to be unbounded. The bbox inferrer raises
            # later if *every* child turns out to be unbounded.
            for c in children:
                _walk(c, can_be_unbounded=True, region_name=region_name)
        else:
            # Unknown op: leave to the compiler/schema validator.
            for c in children:
                _walk(c, can_be_unbounded=can_be_unbounded, region_name=region_name)
        return

    if node_type in ("transform", "modifier", "deform"):
        # These wrap a single child and preserve unboundedness.
        child = node.get("child")
        if child is not None:
            _walk(child, can_be_unbounded=can_be_unbounded, region_name=region_name)
        return

    if node_type == "2d_to_3d":
        # 2D primitives are the only thing a 2d_to_3d node can wrap, and
        # they are inherently bounded: nothing further to check.
        return

    if node_type == "vsweep":
        # A finite path carrying a 2-D profile: bounded by construction.
        return


def _check_phase_frac_requires_centered(node: dict[str, Any], path: str) -> None:
    """``phase_frac`` offsets the centered wedge only; fail loud otherwise."""
    raw = node.get("params") or {}
    if not isinstance(raw, dict) or "phase_frac" not in raw:
        return
    if raw.get("centered") is True:
        return
    if isinstance(raw["phase_frac"], (int, float)) and raw["phase_frac"] == 0:
        return
    raise SemanticValidationError(
        f"{path}.params.phase_frac: requires centered: true on "
        "canonical_sector_fold; without it, rotate the child with rotate_z instead"
    )


def _plane_message(region_name: str) -> str:
    prefix = f"MaterialRegion {region_name!r}: " if region_name else ""
    return (
        f"{prefix}`plane` is mathematically infinite and cannot appear here. "
        "A plane is only valid as a CSG cutting tool: wrap it as the second "
        "operand of `subtract(<bounded shape>, plane)` to cut, or as an "
        "operand of `intersect(<bounded shape>, plane)` to keep a half-space. "
        "If you want a finite slab, use `box` with a thin axis instead."
    )


# ---------------------------------------------------------------------------
# Registry-driven semantic validation
# ---------------------------------------------------------------------------
# This is the cross-workstream interface: software_defined_matter.io.validate
# calls validate_document_semantics(doc) once per document, on the raw wire
# dict, regardless of its declared schema_version.


def validate_document_semantics(doc: dict[str, Any]) -> None:
    """Validate every SDF node in ``doc`` against the node-kind contract in
    :mod:`software_defined_matter.wire`.

    Checks, at any depth, in every material region's and coupling node's
    ``sdf_tree``:

      - ``kind`` / ``op`` / ``transform`` / ``modifier`` / ``deform`` /
        ``method`` names exist in the contract.
      - A primitive's dimension matches the position it sits in (see
        :func:`_check_dim`).
      - The node's ``params`` object carries every required kwarg, no
        unrecognised ones, and each value has the shape the contract
        declares (a number where a scalar is expected, a fixed-length
        array for a ``vec2`` / ``vec3`` / ``mat3`` slot, and so on).
      - Every ``{"$ref": name}`` leaf, at any nesting depth inside a kwarg,
        names a param that is actually present in ``doc["params"]``.

    Version-independent: the wire vocabulary it enforces has never varied
    by version (see the module docstring). Does not repeat the JSON
    Schema's own checks, and does not evaluate ``Param.prior``. An ``expr``
    subtree nested in a kwarg is checked one level deep, for a ``type`` in
    :data:`software_defined_matter.wire.EXPR_NODE_TYPES`; below that it is
    :mod:`software_defined_matter.dsl.expr`'s contract, not this one's.

    Raises:
        SemanticValidationError: Naming the offending field and the node
            path (e.g. ``materials[0].sdf_tree.params.r``) it was found at.
    """
    if not isinstance(doc, dict):
        raise SemanticValidationError(f"document: expected an object, got {type(doc).__name__}")

    raw_params = doc.get("params") or {}
    if not isinstance(raw_params, dict):
        raise SemanticValidationError(
            f"params: expected an object, got {type(raw_params).__name__}"
        )
    known_params = set(raw_params)

    for i, material in enumerate(doc.get("materials") or []):
        if not isinstance(material, dict):
            continue  # Malformed; the JSON Schema already rejects this shape.
        tree = material.get("sdf_tree")
        if tree is not None:
            _validate_node(tree, known_params, f"materials[{i}].sdf_tree", 3)

    for key in ("ports", "couplings"):
        for i, port in enumerate(doc.get(key) or []):
            if not isinstance(port, dict):
                continue
            tree = port.get("sdf_tree")
            if tree is not None:
                _validate_node(tree, known_params, f"{key}[{i}].sdf_tree", 3)


def _validate_node(
    node: Any, known_params: set[str], path: str, expected_dim: Literal[2, 3]
) -> None:
    """Validate one SDF tree node and recurse into its children, if any.

    ``expected_dim`` is the dimension a primitive is required to have at
    this position: 3 at a material/coupling root and inside a 3-D op /
    transform / modifier / deform, 2 inside a ``2d_to_3d`` child, a
    ``sweep`` profile, or a ``loft`` child. CSG ops, transforms, modifiers,
    and deforms are dimension-agnostic themselves (``jnp.minimum`` doesn't
    care how many axes a point has) and simply pass ``expected_dim``
    through unchanged to whatever they wrap.
    """
    if not isinstance(node, dict):
        raise SemanticValidationError(f"{path}: expected an SDF node (object), got {node!r}")

    node_type = node.get("type")
    if node_type not in wire.SDF_NODE_TYPES:
        raise SemanticValidationError(
            f"{path}.type: expected one of {list(wire.SDF_NODE_TYPES)}, got {node_type!r}"
        )

    if node_type == "primitive":
        spec = _validate_named(node, "kind", wire.PRIMITIVES, known_params, path)
        _check_dim(spec, expected_dim, path, "kind")
        if spec.name == "raster_field":
            from software_defined_matter.sdf.raster import (
                decode_raster_values,
                normalize_spacing,
                raster_origin,
            )

            try:
                decode_raster_values(node["params"])
                normalize_spacing(node["params"])
                raster_origin(node["params"])
            except ValueError as exc:
                raise SemanticValidationError(f"{path}: {exc}") from exc
        return

    if node_type == "op":
        spec = _validate_named(node, "op", wire.OPS, known_params, path)
        children = node.get("children")
        if not isinstance(children, list) or not children:
            raise SemanticValidationError(
                f"{path}.children: op {spec.name!r} needs a non-empty list of children"
            )
        for i, child in enumerate(children):
            _validate_node(child, known_params, f"{path}.children[{i}]", expected_dim)
        return

    if node_type == "transform":
        spec = _validate_named(node, "transform", wire.TRANSFORMS, known_params, path, expected_dim)
        if spec.name == "canonical_sector_fold":
            _check_phase_frac_requires_centered(node, path)
        _validate_child(node, known_params, path, expected_dim)
        return

    if node_type == "modifier":
        _validate_named(node, "modifier", wire.MODIFIERS, known_params, path, expected_dim)
        _validate_child(node, known_params, path, expected_dim)
        return

    if node_type == "deform":
        spec = _validate_named(node, "deform", wire.DEFORMS, known_params, path)
        _validate_child(node, known_params, path, expected_dim)
        if spec.requires_field:
            field_tree = node.get("field")
            if field_tree is None:
                raise SemanticValidationError(
                    f"{path}.field: deform {spec.name!r} requires a 'field' subtree"
                )
            _validate_field_node(field_tree, known_params, f"{path}.field")
        return

    if node_type == "2d_to_3d":
        spec = _validate_named(node, "method", wire.TWO_D_TO_3D, known_params, path)
        _check_dim(spec, expected_dim, path, "method")
        _validate_child(node, known_params, path, 2)
        return

    if node_type == "sweep":
        _check_dim(wire.SWEEP, expected_dim, path, "type")
        _validate_params(node, wire.SWEEP, known_params, path)
        _validate_child(node, known_params, path, 2)
        return

    if node_type == "vsweep":
        _check_dim(wire.VSWEEP, expected_dim, path, "type")
        _validate_params(node, wire.VSWEEP, known_params, path)
        global _ALONG_DEPTH
        _ALONG_DEPTH += 1
        try:
            _validate_child(node, known_params, path, 2)
        finally:
            _ALONG_DEPTH -= 1
        return

    if node_type == "loft":
        _check_dim(wire.LOFT, expected_dim, path, "type")
        _validate_params(node, wire.LOFT, known_params, path)
        children = node.get("children")
        if not isinstance(children, list) or len(children) < 2:
            raise SemanticValidationError(
                f"{path}.children: loft needs at least 2 cross-section children"
            )
        for i, child in enumerate(children):
            _validate_node(child, known_params, f"{path}.children[{i}]", 2)
        return


def _validate_field_node(node: Any, known_params: set[str], path: str) -> None:
    """Validate one displacement-field tree node (a ``deform("displace")``
    child), recursing into ``field_op`` children."""
    if not isinstance(node, dict):
        raise SemanticValidationError(f"{path}: expected a field node (object), got {node!r}")

    node_type = node.get("type")
    if node_type not in wire.FIELD_NODE_TYPES:
        raise SemanticValidationError(
            f"{path}.type: expected one of {list(wire.FIELD_NODE_TYPES)}, got {node_type!r}"
        )

    if node_type == "field":
        _validate_named(node, "kind", wire.FIELD_PRIMITIVES, known_params, path)
        return

    spec = _validate_named(node, "op", wire.FIELD_OPS, known_params, path)
    children = node.get("children")
    if not isinstance(children, list) or not children:
        raise SemanticValidationError(
            f"{path}.children: field op {spec.name!r} needs a non-empty list of children"
        )
    for i, child in enumerate(children):
        _validate_field_node(child, known_params, f"{path}.children[{i}]")


def _validate_named(
    node: dict[str, Any],
    key: str,
    registry: dict[str, wire.NodeSpec],
    known_params: set[str],
    path: str,
    expected_dim: Literal[2, 3] = 3,
) -> wire.NodeSpec:
    """Look ``node[key]`` up in ``registry``, then validate its ``params``.

    Shared by every node kind that dispatches on a sub-name (``kind`` /
    ``op`` / ``transform`` / ``modifier`` / ``deform`` / ``method``): the
    lookup-and-validate sequence is identical, only the field name and the
    registry differ.
    """
    name = node.get(key)
    if not isinstance(name, str) or name not in registry:
        raise SemanticValidationError(
            f"{path}.{key}: unknown {key} {name!r}. Allowed: {sorted(registry)}."
        )
    spec = registry[name]
    _validate_params(node, spec, known_params, path, expected_dim)
    return spec


def _check_dim(spec: wire.NodeSpec, expected_dim: Literal[2, 3], path: str, key: str) -> None:
    """Reject a node whose declared dimension doesn't match the position it
    sits in.

    ``spec.dim`` is set on every primitive (its actual dimension) and,
    fixed at 3, on every ``2d_to_3d`` / ``sweep`` / ``loft`` entry -- those
    always lift to 3-D, so ``extrusion(extrusion(circle_2d))`` is wrong the
    same way a bare 3-D primitive there would be. ``None`` everywhere else,
    so this is a no-op for ops/transforms/modifiers/deforms.
    """
    if spec.dim is not None and spec.dim != expected_dim:
        raise SemanticValidationError(
            f"{path}.{key}: expected a {expected_dim}-D {spec.category} here, "
            f"but {spec.name!r} is {spec.dim}-D."
        )


def _validate_params(
    node: dict[str, Any],
    spec: wire.NodeSpec,
    known_params: set[str],
    path: str,
    expected_dim: Literal[2, 3] = 3,
) -> None:
    """Validate ``node["params"]`` against ``spec``: no unknown keys, every
    required key present, and every value shaped as the contract declares.
    """
    raw = node.get("params") or {}
    if not isinstance(raw, dict):
        raise SemanticValidationError(f"{path}.params: expected an object, got {raw!r}")

    allowed = spec.param_names()
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise SemanticValidationError(
            f"{path}.params: unknown parameter(s) {unknown} for {spec.category} "
            f"{spec.name!r}. Allowed: {list(allowed)}."
        )
    missing = [n for n in spec.required_param_names() if n not in raw]
    if missing:
        raise SemanticValidationError(
            f"{path}.params: missing required parameter(s) {missing} for "
            f"{spec.category} {spec.name!r}."
        )
    for pspec in spec.params:
        if pspec.name in raw:
            _validate_value(
                raw[pspec.name],
                pspec,
                known_params,
                f"{path}.params.{pspec.name}",
                expected_dim,
            )


def _validate_child(
    node: dict[str, Any], known_params: set[str], path: str, expected_dim: Literal[2, 3]
) -> None:
    child = node.get("child")
    if child is None:
        raise SemanticValidationError(f"{path}.child: missing required 'child' subtree")
    _validate_node(child, known_params, f"{path}.child", expected_dim)


def _validate_scalar_leaf(value: Any, *, ref_ok: bool, known_params: set[str], path: str) -> None:
    """Validate one number-shaped leaf: a literal number, a ``$ref``, or an
    expression subtree.

    The expression subtree is checked one level deep: its ``type`` must name
    a node kind in :data:`software_defined_matter.wire.EXPR_NODE_TYPES`.
    Nothing below that is this layer's business -- an operator name, an
    operand's presence, and the tree's shape are
    :mod:`software_defined_matter.dsl.expr`'s contract. The shallow check
    exists because ``sdf.params`` is an unconstrained object in the frozen
    schemas, so on a document declaring one of those a misspelled
    expression node nested in a kwarg is otherwise caught by nothing until
    evaluation.
    """
    if isinstance(value, dict):
        if "$ref" in value:
            if not ref_ok:
                raise SemanticValidationError(f"{path}: a '$ref' leaf is not accepted here")
            ref_name = value["$ref"]
            if not isinstance(ref_name, str) or ref_name not in known_params:
                raise SemanticValidationError(
                    f"{path}.$ref: no such param {ref_name!r} in this document's params"
                )
            return
        if "type" in value:
            node_type = value["type"]
            if node_type not in wire.EXPR_NODE_TYPES:
                raise SemanticValidationError(
                    f"{path}.type: unknown expression node type {node_type!r}. "
                    f"Allowed: {list(wire.EXPR_NODE_TYPES)}."
                )
            return
        raise SemanticValidationError(f"{path}: unrecognised leaf {value!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticValidationError(f"{path}: expected a number, got {value!r}")


#: How many ``vsweep`` profiles the validator is currently inside. A
#: ``{"$along": [...]}`` leaf (one value per path vertex) is only meaningful
#: there; anywhere else it is an unrecognised leaf.
_ALONG_DEPTH = 0


def _validate_value(
    value: Any,
    pspec: wire.ParamSpec,
    known_params: set[str],
    path: str,
    expected_dim: Literal[2, 3] = 3,
) -> None:
    """Validate one kwarg value against the shape ``pspec`` declares.

    ``expected_dim`` only matters for ``axis_vector``, whose length is fixed by
    the subtree rather than by the contract. Every other shape is a fixed
    length and ignores it.

    Inside a ``vsweep`` profile a value may be ``{"$along": [v_0, ...]}``:
    one value per path vertex, each shaped as the contract declares.
    """
    shape = pspec.wire_shape

    if isinstance(value, dict) and "$along" in value:
        if _ALONG_DEPTH == 0:
            raise SemanticValidationError(
                f"{path}: an '$along' leaf is only accepted inside a vsweep profile"
            )
        if set(value) != {"$along"} or shape in ("string", "bool", "object"):
            raise SemanticValidationError(f"{path}: malformed '$along' leaf {value!r}")
        values = value["$along"]
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise SemanticValidationError(
                f"{path}.$along: expected a list of at least 2 per-vertex values"
            )
        for i, item in enumerate(values):
            _validate_value(item, pspec, known_params, f"{path}.$along[{i}]", expected_dim)
        return

    if shape == "scalar":
        _validate_scalar_leaf(value, ref_ok=pspec.ref_ok, known_params=known_params, path=path)
        return

    if shape == "axis_vector":
        # "2-D or 3-D" is a property of the contract, not of any one document:
        # the subtree this node sits in fixes which. Accepting either length
        # here would let a 2-D vector through into a 3-D subtree and surface as
        # an unhandled ValueError in the GLSL emitter instead of a validation
        # error, which is the failure this shape exists to prevent.
        if not isinstance(value, (list, tuple)) or len(value) != expected_dim:
            raise SemanticValidationError(
                f"{path}: expected a length-{expected_dim} array in a "
                f"{expected_dim}-D subtree, got {value!r}"
            )
        for i, v in enumerate(value):
            _validate_scalar_leaf(
                v, ref_ok=pspec.ref_ok, known_params=known_params, path=f"{path}[{i}]"
            )
        return

    if shape in ("vec2", "vec3"):
        n = 2 if shape == "vec2" else 3
        if not isinstance(value, (list, tuple)) or len(value) != n:
            raise SemanticValidationError(f"{path}: expected a length-{n} array, got {value!r}")
        for i, v in enumerate(value):
            _validate_scalar_leaf(
                v, ref_ok=pspec.ref_ok, known_params=known_params, path=f"{path}[{i}]"
            )
        return

    if shape == "mat3":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise SemanticValidationError(f"{path}: expected a 3x3 array, got {value!r}")
        for i, row in enumerate(value):
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise SemanticValidationError(f"{path}[{i}]: expected a length-3 row, got {row!r}")
            for j, v in enumerate(row):
                _validate_scalar_leaf(
                    v, ref_ok=pspec.ref_ok, known_params=known_params, path=f"{path}[{i}][{j}]"
                )
        return

    if shape == "scalar_list":
        if not isinstance(value, (list, tuple)):
            raise SemanticValidationError(f"{path}: expected a list of numbers, got {value!r}")
        for i, v in enumerate(value):
            _validate_scalar_leaf(
                v, ref_ok=pspec.ref_ok, known_params=known_params, path=f"{path}[{i}]"
            )
        return

    if shape == "point_list":
        if not isinstance(value, (list, tuple)):
            raise SemanticValidationError(f"{path}: expected a list of points, got {value!r}")
        if pspec.min_points is not None and len(value) < pspec.min_points:
            raise SemanticValidationError(
                f"{path}: needs at least {pspec.min_points} points, got {len(value)}"
            )
        if pspec.multiple_of is not None and len(value) % pspec.multiple_of != 0:
            raise SemanticValidationError(
                f"{path}: needs a point count that is a multiple of "
                f"{pspec.multiple_of}, got {len(value)}"
            )
        dim = pspec.point_dim
        for i, pt in enumerate(value):
            if not isinstance(pt, (list, tuple)) or (dim is not None and len(pt) != dim):
                raise SemanticValidationError(
                    f"{path}[{i}]: expected a length-{dim} point, got {pt!r}"
                )
            for j, v in enumerate(pt):
                _validate_scalar_leaf(
                    v, ref_ok=pspec.ref_ok, known_params=known_params, path=f"{path}[{i}][{j}]"
                )
        return

    if shape == "string":
        if not isinstance(value, str):
            raise SemanticValidationError(f"{path}: expected a string, got {value!r}")
        if pspec.choices is not None and value not in pspec.choices:
            raise SemanticValidationError(f"{path}: expected one of {pspec.choices}, got {value!r}")
        return

    if shape == "object":
        if not isinstance(value, dict):
            raise SemanticValidationError(f"{path}: expected an object, got {value!r}")
        return

    if shape == "bool":
        if not isinstance(value, bool):
            raise SemanticValidationError(f"{path}: expected a boolean, got {value!r}")
        return

    # "object_map" / "object_list" describe document-structure
    # fields (Part.metadata, Part.materials, ...), never an SDF kwarg; no
    # ParamSpec in software_defined_matter.wire declares one of these, so
    # reaching here would be a contract bug, not a document one.
    raise AssertionError(f"{path}: unsupported wire_shape {shape!r} for an SDF kwarg")


__all__ = [
    "SemanticValidationError",
    "UnboundedRootError",
    "validate_document_semantics",
    "validate_material_tree",
    "validate_part_material_trees",
]
