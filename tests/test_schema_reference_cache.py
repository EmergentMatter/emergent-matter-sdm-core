"""Structural validation keeps standard errors and never reuses stale successes.

These tests drive the reference path of `io.validate_schema`, the one rejected documents,
documents holding numpy values or non-finite floats, and installs without jsonschema-rs take.
`_reference_path` turns the compiled validator off so each test reaches it.
"""

from __future__ import annotations

import copy

import jsonschema
import pytest

from software_defined_matter import MaterialRegion, Param, Part, sdf_primitive
from software_defined_matter import io as sdm_io
from software_defined_matter.dsl.expr import expr_binop, expr_num, expr_param
from software_defined_matter.io import load_schema, validate
from software_defined_matter.model import LATEST_SCHEMA_VERSION


def _document(depth: int = 8) -> dict:
    expression = expr_param("width")
    for _ in range(depth):
        expression = expr_binop("+", expression, expr_num(0.25))
    part = Part(
        name="nested_expression",
        params={"width": Param("width", 2.0, free=True, unit="mm")},
        materials=[MaterialRegion(1, "solid", sdf_primitive("sphere", r=expression))],
    )
    doc = part.to_dict()
    doc["schema_version"] = LATEST_SCHEMA_VERSION
    return doc


def _reference_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send every document down the reference path, as an install without jsonschema-rs would."""
    monkeypatch.setattr(sdm_io, "_compiled_schema", lambda version: None)


def test_nested_expression_validation_matches_standard_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reference_path(monkeypatch)
    doc = _document()
    validate(doc)
    jsonschema.validate(doc, load_schema(doc["schema_version"]))


def test_deep_expression_validation_does_not_repeat_reference_checks_exponentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reference_path(monkeypatch)
    reference = jsonschema.Draft202012Validator.VALIDATORS["$ref"]
    calls = 0

    def bounded_reference(validator, ref, instance, subschema):
        nonlocal calls
        # Count the document's local schema definitions, not jsonschema's
        # separate validation of the schema against its own metaschema.
        if ref.startswith("#/$defs/"):
            calls += 1
        # A traversal budget is stable across machines. The ordinary validator
        # exhausts it through repeated composition checks on this small tree.
        if calls > 5000:
            pytest.fail("schema validation exceeded 5000 reference checks for one expression")
        yield from reference(validator, ref, instance, subschema)

    monkeypatch.setitem(jsonschema.Draft202012Validator.VALIDATORS, "$ref", bounded_reference)
    validate(_document(depth=20))


def test_validation_rechecks_a_mutated_document_and_preserves_error_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reference_path(monkeypatch)
    doc = _document()
    validate(doc)
    doc["materials"][0]["sdf_tree"]["params"]["r"] = {"$ref": 17}
    with pytest.raises(jsonschema.ValidationError) as actual:
        validate(doc)
    with pytest.raises(jsonschema.ValidationError) as expected:
        jsonschema.validate(doc, load_schema(doc["schema_version"]))
    assert list(actual.value.absolute_path) == list(expected.value.absolute_path)
    assert actual.value.message == expected.value.message


def test_one_valid_region_does_not_mask_an_invalid_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reference_path(monkeypatch)
    doc = _document()
    second = copy.deepcopy(doc["materials"][0])
    second["material_id"] = 2
    second["sdf_tree"]["unexpected"] = True
    doc["materials"].append(second)
    with pytest.raises(jsonschema.ValidationError, match="not valid|Unevaluated"):
        validate(doc)
