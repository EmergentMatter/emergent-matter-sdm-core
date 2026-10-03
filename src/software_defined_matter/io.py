"""Load / save / validate ``.sdm`` files.

A ``.sdm`` file is a JSON document with the top-level shape described by the
versioned schemas in ``src/software_defined_matter/schema/`` (one JSON Schema
per ``schema_version``; see ``KNOWN_SCHEMA_VERSIONS``). The file extension is
a convention; the wire format is UTF-8 JSON.
"""

from __future__ import annotations

import json
import os
import warnings
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
    import jsonschema

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

    jsonschema.validate(doc, load_schema(_schema_version_for(doc)))
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

    Structural and local semantic validation run by default. Use ``load_bundle``
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
            validated against the schema before it is written.
        n_indent: JSON indent; set to ``None`` for a compact single-line file.
    """
    doc = part.to_dict()
    if b_validate_schema:
        validate(doc)
    with Path(path).open("w") as fh:
        json.dump(doc, fh, indent=n_indent)


__all__ = ["load_part", "load", "load_schema", "save", "validate"]
