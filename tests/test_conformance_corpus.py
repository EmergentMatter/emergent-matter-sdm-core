"""The shipped `.sdm` conformance corpus, checked against its own contract.

`schema/conformance/` exists so a downstream reader (`sdm-view`,
a web viewer, a reimplementation in another language) can assert "I accept
all of valid/, I reject all of invalid/" against this package's fixtures
instead of guessing at the wire format. That promise is only worth anything
if the fixtures still hold here, so this module runs the exact contract the
corpus README hands consumers:

  - every `valid/` document passes `io.validate()`;
  - every `invalid/` document passes the JSON Schema for its own declared
    version and *then* fails `io.validate()` with `SemanticValidationError`.

The second half is the load-bearing one. A fixture that fails the JSON
Schema instead would still "fail validation" and look fine from the outside,
while no longer demonstrating the semantic layer this corpus exists to pin --
a consumer running only a JSON Schema check would then pass a test suite it
should have failed.

Fixtures are discovered through `importlib.resources`, not a relative path,
so this is also the check that the corpus is reachable the way an installed
consumer reaches it rather than only in a source checkout.
"""

from __future__ import annotations

import json
from importlib.resources import files
from importlib.resources.abc import Traversable

import jsonschema
import pytest

from software_defined_matter.io import load_schema, validate
from software_defined_matter.sdf.validate import SemanticValidationError

CORPUS = files("software_defined_matter.schema.conformance")


def _fixtures(bucket: str) -> list[Traversable]:
    return sorted(
        (p for p in (CORPUS / bucket).iterdir() if p.name.endswith(".sdm")), key=lambda p: p.name
    )


VALID = _fixtures("valid")
INVALID = _fixtures("invalid")


def _ids(paths: list[Traversable]) -> list[str]:
    return [p.name for p in paths]


def test_corpus_is_not_empty():
    """Guard on the parametrization itself: `iterdir` over a corpus that
    failed to ship yields nothing, and every parametrized test below would
    then pass by collecting zero cases. This is the one assertion that
    cannot be silently vacuous."""
    assert VALID, "no valid/ fixtures discovered -- did the corpus ship?"
    assert INVALID, "no invalid/ fixtures discovered -- did the corpus ship?"


@pytest.mark.parametrize("path", VALID, ids=_ids(VALID))
def test_valid_fixture_passes_validate(path: Traversable):
    validate(json.loads(path.read_text()))


@pytest.mark.parametrize("path", INVALID, ids=_ids(INVALID))
def test_invalid_fixture_passes_json_schema(path: Traversable):
    """Half the corpus's point: these documents are structurally fine. If
    one starts failing the JSON Schema, it has stopped being a fixture for
    the semantic layer, whatever else it still fails."""
    doc = json.loads(path.read_text())
    jsonschema.validate(doc, load_schema(doc["schema_version"]))


@pytest.mark.parametrize("path", INVALID, ids=_ids(INVALID))
def test_invalid_fixture_is_rejected_semantically(path: Traversable):
    with pytest.raises(SemanticValidationError):
        validate(json.loads(path.read_text()))


@pytest.mark.parametrize("path", INVALID, ids=_ids(INVALID))
def test_invalid_fixture_name_matches_its_filename(path: Traversable):
    """A consumer's failure report names the document, not the file it came
    from. When every fixture shared one `name`, "conformance_invalid_example
    was accepted" identified nothing."""
    stem = path.name[: -len(".sdm")]
    assert json.loads(path.read_text())["name"] == f"conformance_invalid_{stem}"


def test_every_fixture_name_is_unique():
    names = [json.loads(p.read_text())["name"] for p in (*VALID, *INVALID)]
    assert len(names) == len(set(names)), f"duplicate fixture name(s) in {sorted(names)}"
