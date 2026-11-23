"""The compiled schema validator (jsonschema-rs) and the reference (jsonschema) agree.

`io.validate_schema` lets jsonschema-rs decide and uses jsonschema only for the error message,
so a document the compiled validator wrongly accepted would pass unchecked. These tests hold the
two to the same verdict over the conformance suite, the examples, and documents broken on
purpose, including the 2020-12 keyword (`unevaluatedProperties`) a draft-7 validator ignores.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from software_defined_matter import io as sdm_io

ROOT = Path(__file__).resolve().parent.parent
CONFORMANCE = ROOT / "src" / "software_defined_matter" / "schema" / "conformance"
DOCS = (
    sorted(CONFORMANCE.glob("valid/*.sdm"))
    + sorted(CONFORMANCE.glob("invalid/*.sdm"))
    + sorted((ROOT / "examples").glob("*.sdm"))
)


def _verdicts(doc: dict) -> tuple[bool, bool]:
    version = sdm_io._schema_version_for(doc)
    compiled = sdm_io._compiled_schema(version)
    assert compiled is not None, "jsonschema-rs is a dependency"
    try:
        jsonschema.validate(doc, sdm_io.load_schema(version))
        reference = True
    except jsonschema.ValidationError:
        reference = False
    return compiled.is_valid(doc), reference


@pytest.mark.parametrize("path", DOCS, ids=[p.name for p in DOCS])
def test_the_validators_agree_on_every_document(path: Path) -> None:
    fast, reference = _verdicts(json.loads(path.read_text()))
    assert fast == reference


def _first_sdf_node(doc: dict) -> dict:
    for region in doc.get("materials", []):
        if isinstance(region.get("sdf_tree"), dict):
            return region["sdf_tree"]
    raise LookupError("no SDF tree")


BREAKS = {
    "unknown key on an SDF node": lambda d: _first_sdf_node(d).__setitem__("surprise", 1),
    "unknown node type": lambda d: _first_sdf_node(d).__setitem__("type", "not_a_type"),
    "unknown top-level key": lambda d: d.__setitem__("surprise", 1),
    "missing name": lambda d: d.pop("name"),
    "number as a string": lambda d: d.__setitem__("schema_version", 7),
}


@pytest.mark.parametrize("s_break", list(BREAKS))
def test_the_validators_agree_on_broken_documents(s_break: str) -> None:
    doc = json.loads((CONFORMANCE / "valid" / "minimal_0.7.sdm").read_text())
    broken = copy.deepcopy(doc)
    BREAKS[s_break](broken)
    fast, reference = _verdicts(broken)
    assert reference is False, "the break should make the document invalid"
    assert fast == reference


def test_a_rejected_document_raises_the_reference_error() -> None:
    doc = json.loads((CONFORMANCE / "valid" / "minimal_0.7.sdm").read_text())
    _first_sdf_node(doc)["surprise"] = 1
    with pytest.raises(jsonschema.ValidationError):
        sdm_io.validate_schema(doc)


def test_non_finite_numbers_are_left_to_the_semantic_checks() -> None:
    """jsonschema takes NaN for a number and jsonschema-rs does not; the reference decides,
    silently, as it did before, and the semantic checks downstream reject the value."""
    import warnings

    doc = json.loads((CONFORMANCE / "valid" / "minimal_0.7.sdm").read_text())
    params = doc.get("params") or {}
    if not params:
        pytest.skip("the minimal document has no param to break")
    name = next(iter(params))
    doc["params"][name]["value"] = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sdm_io.validate_schema(doc)


def test_numpy_values_go_to_the_reference() -> None:
    """A document built in Python may hold numpy scalars until it is written; jsonschema takes
    them as numbers, jsonschema-rs cannot read them, so the reference decides."""
    import warnings

    import numpy as np

    doc = json.loads((CONFORMANCE / "valid" / "minimal_0.7.sdm").read_text())
    assert sdm_io._is_plain_json(doc)
    doc["numpy_free_check"] = np.float64(1.0)
    assert not sdm_io._is_plain_json(doc)
    del doc["numpy_free_check"]
    node = _first_sdf_node(doc)
    for k, v in list(node.get("params", {}).items()):
        if isinstance(v, float):
            node["params"][k] = np.float64(v)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sdm_io.validate_schema(doc)
