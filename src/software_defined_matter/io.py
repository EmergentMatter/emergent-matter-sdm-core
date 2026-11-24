"""Load / save / validate ``.sdm`` files.

A ``.sdm`` file is a JSON document with the top-level shape described by the
versioned schemas in ``src/software_defined_matter/schema/`` (one JSON Schema
per ``schema_version``; see ``KNOWN_SCHEMA_VERSIONS``). The file extension is
a convention; the wire format is UTF-8 JSON.
"""

from __future__ import annotations

import functools
import json
import math
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from software_defined_matter.assembly.model import Assembly
from software_defined_matter.model import (
    FALLBACK_SCHEMA_VERSION,
    KNOWN_SCHEMA_VERSIONS,
    LATEST_SCHEMA_VERSION,
    Part,
)


def _warn_if_version_mismatch(doc: dict[str, Any]) -> None:
    """Emit a UserWarning if ``doc["schema_version"]`` is not a version this
    package knows. Lenient policy: load anyway, downstream consumers handle
    drift on a best-effort basis.
    """
    sv = doc.get("schema_version")
    if sv is not None and sv not in KNOWN_SCHEMA_VERSIONS:
        warnings.warn(
            f".sdm schema_version={sv!r} is not one this package knows "
            f"({KNOWN_SCHEMA_VERSIONS}); loading on a best-effort basis.",
            stacklevel=3,
        )


_SCHEMA_DIR = Path(__file__).parent / "schema"

PathLike = str | os.PathLike


def _validate_json_schema(doc: dict[str, Any], schema: dict[str, Any]) -> None:
    """Reuse successful local reference checks within one immutable document walk.

    Draft 2020-12's unevaluatedProperties checks can revisit the same subtree
    through several composition branches. A cache scoped to this call avoids
    exponential work on nested expressions without caching failed validations
    or reusing results after the caller changes its document.
    """
    import jsonschema

    base = jsonschema.validators.validator_for(schema)
    reference = base.VALIDATORS["$ref"]
    one_of = base.VALIDATORS["oneOf"]
    successful: dict[tuple[int, str, str], Any] = {}

    def cached_reference(validator: Any, ref: str, instance: Any, subschema: Any) -> Iterator[Any]:
        resolver = getattr(validator, "_resolver", None)
        base_uri = getattr(resolver, "_base_uri", None)
        if base_uri is None:
            yield from reference(validator, ref, instance, subschema)
            return
        key = (id(instance), ref, base_uri)
        if key in successful:
            return
        b_failed = False
        for error in reference(validator, ref, instance, subschema):
            b_failed = True
            yield error
        if not b_failed:
            # Retain the object as well as its identity so a temporary value's
            # id cannot be recycled during validation of the same document.
            successful[key] = instance

    def definitely_invalid(validator: Any, instance: Any, alternative: Any) -> bool:
        # This is only a rejection filter. Compatible branches still undergo
        # full validation, including constraints not inspected here.
        if not isinstance(alternative, dict):
            return alternative is False
        if "$ref" in alternative:
            resolver = getattr(validator, "_resolver", None)
            if resolver is None or not callable(getattr(resolver, "lookup", None)):
                return False
            resolved = resolver.lookup(alternative["$ref"]).contents
            if resolved is not alternative:
                alternative = resolved
        if not isinstance(alternative, dict):
            return alternative is False
        expected_type = alternative.get("type")
        if isinstance(expected_type, str) and not validator.TYPE_CHECKER.is_type(
            instance, expected_type
        ):
            return True
        if isinstance(instance, dict):
            if any(key not in instance for key in alternative.get("required", [])):
                return True
            for key, constraints in alternative.get("properties", {}).items():
                # String tags discriminate expression and geometry variants.
                # Restricting this shortcut to strings avoids JSON/Python
                # differences such as True == 1 in arbitrary const values.
                if not isinstance(constraints, dict) or key not in instance:
                    continue
                tag = constraints.get("const")
                if isinstance(tag, str) and instance[key] != tag:
                    return True
        return False

    def discriminated_one_of(
        validator: Any, alternatives: Any, instance: Any, subschema: Any
    ) -> Iterator[Any]:
        candidates = [
            (index, alternative)
            for index, alternative in enumerate(alternatives)
            if not definitely_invalid(validator, instance, alternative)
        ]
        if len(candidates) == 1:
            index, alternative = candidates[0]
            if not any(validator.descend(instance, alternative, schema_path=index)):
                return
        # The ordinary implementation supplies the complete standard error
        # context for invalid or ambiguous unions, preserving diagnostics.
        yield from one_of(validator, alternatives, instance, subschema)

    cached = jsonschema.validators.extend(
        base, {"$ref": cached_reference, "oneOf": discriminated_one_of}
    )
    jsonschema.validate(doc, schema, cls=cached)


def _schema_path(version: str) -> Path:
    return _SCHEMA_DIR / f"sdm-{version}.schema.json"


def load_schema(version: str | None = None) -> dict[str, Any]:
    """Return the parsed JSON Schema for ``version``.

    ``None`` (default) means ``LATEST_SCHEMA_VERSION``, the newest schema
    this package ships. Any member of ``KNOWN_SCHEMA_VERSIONS`` is accepted;
    anything else raises ``ValueError``.
    """
    if version is None:
        version = LATEST_SCHEMA_VERSION
    if version not in KNOWN_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unknown .sdm schema version {version!r}; known: {KNOWN_SCHEMA_VERSIONS}."
        )
    with _schema_path(version).open() as fh:
        return json.load(fh)


@functools.cache
def _compiled_schema(version: str) -> Any:
    """``jsonschema-rs``'s compiled validator for ``version``, or ``None`` without the package.

    It implements draft 2020-12 in full, ``unevaluatedProperties`` included, and checks a CEM
    document of about 24,000 numbers in about a millisecond where ``jsonschema`` takes seconds
    (every number may also be an expression, so each is tried against both).
    """
    try:
        import jsonschema_rs
    except ImportError:
        return None
    return jsonschema_rs.validator_for(load_schema(version))


def _is_plain_json(node: Any) -> bool:
    """Whether ``node`` holds only what JSON can carry, by exact type: dicts with string keys,
    lists, strings, booleans, None, ints and finite floats. A numpy scalar or array, a tuple, or
    a NaN or infinity makes it False."""
    t = type(node)
    if t is dict:
        return all(type(k) is str and _is_plain_json(v) for k, v in node.items())
    if t is list:
        return all(_is_plain_json(v) for v in node)
    if t is float:
        return math.isfinite(node)
    return t in (str, int, bool) or node is None


def validate_schema(doc: dict[str, Any]) -> None:
    """Check ``doc`` against the JSON Schema of its declared version.

    The compiled validator decides for a plain-JSON document; when it rejects, ``jsonschema``
    runs on the same document so the error raised is the reference implementation's
    ``ValidationError``, with its path and message. A document holding anything JSON cannot
    carry (numpy scalars from a Python build, NaN, infinity, which Python's jsonschema takes as
    numbers and sdm-core's semantic checks reject later) goes to ``jsonschema`` alone, as does
    every document when ``jsonschema-rs`` is not installed. That reference path runs through
    ``_validate_json_schema``, which bounds the repeated reference checks of nested expressions.
    ``tests/test_schema_validators_agree.py`` holds the two to the same verdict.
    """
    version = _schema_version_for(doc)
    compiled = _compiled_schema(version) if _is_plain_json(doc) else None
    if compiled is not None and compiled.is_valid(doc):
        return
    _validate_json_schema(doc, load_schema(version))
    if compiled is not None:  # the reference accepts a plain-JSON document jsonschema-rs rejected
        warnings.warn(
            "jsonschema-rs rejected a document that jsonschema accepts; the document is accepted. "
            "Please report it with the document.",
            stacklevel=2,
        )


def _schema_version_for(doc: dict[str, Any]) -> str:
    """The schema version to validate ``doc`` against.

    Dispatches on the document's own declared ``schema_version``: each
    version pins itself with a ``const``, so a 0.2 file can never validate
    against the 0.3 schema by accident. Unknown or absent declarations fall
    back to ``FALLBACK_SCHEMA_VERSION``, whose permissive pattern lets a
    well-formed future version validate structurally and warn instead of
    fail (see :func:`_warn_if_version_mismatch`).
    """
    sv = doc.get("schema_version")
    if sv in KNOWN_SCHEMA_VERSIONS:
        return sv
    return FALLBACK_SCHEMA_VERSION


def validate(source: PathLike | dict[str, Any] | Part | Assembly) -> None:
    """Validate a ``.sdm`` file, dict, or :class:`Part`.

    Runs the JSON Schema check and the semantic checks in
    :mod:`software_defined_matter.sdf.validate` (which catch e.g. a ``plane``
    used as a material root, or a primitive kind/kwarg that no JSON Schema
    version constrains).

    Raises:
        jsonschema.ValidationError: If the document does not conform to the
            JSON Schema for its declared version.
        software_defined_matter.sdf.validate.UnboundedRootError: If any
            material region's SDF tree would be unbounded.
        software_defined_matter.sdf.validate.SemanticValidationError: If any
            SDF node names a kind / op / modifier / deform / method the wire
            contract does not declare, carries a wrong, missing or
            wrong-shaped kwarg, sits at a dimension it cannot occupy, or
            holds a ``$ref`` naming a param the document does not define.
            Raised on a document of any declared version.
        software_defined_matter.sdf.param_refs.UndeclaredParamRefError: If any
            expression or animation track names an undeclared parameter.
        software_defined_matter.sdf.param_refs.ParamRelationError: If a
            parameter relation contains a metric or participates in a cycle.
    """
    from software_defined_matter.assembly.model import Assembly
    from software_defined_matter.sdf.param_refs import (
        validate_param_refs,
        validate_param_relations,
    )
    from software_defined_matter.sdf.validate import (
        validate_document_semantics,
        validate_material_tree,
    )

    if isinstance(source, (Part, Assembly)):
        doc = source.to_dict()
    elif isinstance(source, dict):
        doc = source
    else:
        with Path(source).open() as fh:
            doc = json.load(fh)

    validate_schema(doc)
    if doc.get("kind") == "assembly":
        Assembly.from_dict(doc)
        return
    if "ports" in doc:
        Part.from_dict(doc).validate_ports()
    for region in doc.get("materials", []) or []:
        if region.get("sdf_tree") is not None:
            validate_material_tree(region["sdf_tree"], region_name=region.get("name", ""))
    validate_document_semantics(doc)
    from software_defined_matter.kinematics_validation import validate_kinematics

    validate_kinematics(doc)
    validate_param_refs(doc)
    validate_param_relations(doc)
    _warn_if_version_mismatch(doc)


def load(path: PathLike, *, b_validate_schema: bool = True) -> Part | Assembly:
    """Load a Part or Assembly declaration from the shared document format.

    Structural and local semantic validation run by default;
    ``b_validate_schema=False`` skips both, not only the JSON Schema. Use ``load_bundle``
    to resolve and validate external references, promotions, and bindings.
    """
    with Path(path).open() as fh:
        doc = json.load(fh)
    if b_validate_schema:
        validate(doc)
    return Assembly.from_dict(doc) if doc.get("kind") == "assembly" else Part.from_dict(doc)


def load_part(path: PathLike, *, b_validate_schema: bool = True) -> Part:
    """Load a part for consumers that cannot yet evaluate assemblies."""
    document = load(path, b_validate_schema=b_validate_schema)
    if not isinstance(document, Part):
        raise TypeError("load_part expected a Part; use load or load_bundle for an Assembly")
    return document


def save(
    part: Part | Assembly,
    path: PathLike,
    *,
    b_validate_schema: bool = True,
    n_indent: int | None = 2,
) -> None:
    """Save a :class:`Part` to an ``.sdm`` file.

    Args:
        part: The part to serialise.
        path: Path to write the ``.sdm`` file to.
        b_validate_schema: If True (default), the serialised dict is
            validated against the schema before it is written. False
            skips the semantic checks as well as the JSON Schema.
        n_indent: JSON indent; set to ``None`` for a compact single-line file.
    """
    doc = part.to_dict()
    if b_validate_schema:
        validate(doc)
    with Path(path).open("w") as fh:
        json.dump(doc, fh, indent=n_indent)


__all__ = ["load_part", "load", "load_schema", "save", "validate"]
