"""Generated schema and index are fixed points; released schemas remain byte-identical."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from software_defined_matter import wire
from software_defined_matter.schema import _generate

_SCHEMA_DIR = Path(__file__).parent.parent / "src" / "software_defined_matter" / "schema"
_CONFORMANCE_DIR = _SCHEMA_DIR / "conformance"

# Frozen released artifacts, pinned by hash rather than re-derived: a test
# that regenerated these to compare would defeat the point of freezing them.
_FROZEN_SHA256 = {
    "sdm-0.5.schema.json": "541de9dc2bd356468b6c772cfdd6891bdc65154b68a004ad68de7c724738f806",
    "sdm-0.4.schema.json": "0ebcc66b2bd1781b8a779fe61f9cf9aaadf3811bfde497952d185aa30ce9e636",
    "sdm-0.3.schema.json": "a6e73d81f437df0ccf83635198a1cdd99d57536afa4781e5e866bd9d90db344f",
    "sdm-0.1.schema.json": "1ea4819fa2bef6847d5d26fc61142970f4eab70ad4a4c0725e92ac0fd08b34a1",
    "sdm-0.2.schema.json": "4fd7233c09000f37e320e39391757261d3e9edae38422b1c6232ff0d4c99f67e",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The generator is a fixed point.
# ---------------------------------------------------------------------------


def test_generated_schema_matches_the_committed_file():
    generated = json.dumps(_generate.generate_schema(), indent=2) + "\n"
    committed = (_SCHEMA_DIR / "sdm-0.6.schema.json").read_text()
    assert generated == committed


def test_generated_index_matches_the_committed_file():
    generated = json.dumps(_generate.generate_index(), indent=2) + "\n"
    committed = (_SCHEMA_DIR / "index.json").read_text()
    assert generated == committed


def test_write_generated_files_touches_only_the_current_version(tmp_path):
    """The frozen schemas have no counterpart in tmp_path at all --
    write_generated_files never has a path to write them through."""
    _generate.write_generated_files(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"sdm-0.6.schema.json", "index.json"}


# ---------------------------------------------------------------------------
# Frozen versions are byte-identical to their released form.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", sorted(_FROZEN_SHA256))
def test_frozen_schema_is_untouched(filename):
    assert _sha256(_SCHEMA_DIR / filename) == _FROZEN_SHA256[filename], (
        f"{filename} changed -- frozen released schemas must never be regenerated"
    )


# ---------------------------------------------------------------------------
# index.json content.
# ---------------------------------------------------------------------------


def test_index_lists_every_known_version_with_latest_pinned():
    index = _generate.generate_index()
    assert index["latest"] == "0.6"
    assert set(index["versions"]) == set(wire.SCHEMA_VERSIONS)
    for version in wire.SCHEMA_VERSIONS:
        assert index["versions"][version]["file"] == f"sdm-{version}.schema.json"


def test_index_0_3_additions_are_expr_and_probabilistic_param_fields():
    index = _generate.generate_index()
    assert index["versions"]["0.3"]["added"] == [
        "Param.expr",
        "Param.prior",
        "Param.tolerance",
    ]


# ---------------------------------------------------------------------------
# The generated schema itself, not only the Python semantic layer, rejects
# what schema/conformance/invalid/ pins -- the new capability this PR
# delivers. Each fixture declares an older schema_version (they predate this
# PR); overriding it to "0.3" tests the generated schema's own content
# constraints, independent of the top-level version `const` check.
# ---------------------------------------------------------------------------


def _fixture(name: str) -> dict:
    doc = json.loads((_CONFORMANCE_DIR / name).read_text())
    doc["schema_version"] = "0.6"
    if doc.get("couplings") == []:
        del doc["couplings"]
    return doc


@pytest.mark.parametrize(
    "name",
    [
        "invalid/misspelled_primitive_kind.sdm",
        "invalid/wrong_kwarg_name.sdm",
        "invalid/missing_required_kwarg.sdm",
    ],
)
def test_generated_schema_rejects_invalid_conformance_fixture(name):
    schema = _generate.generate_schema()
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_fixture(name), schema)


def test_generated_schema_cannot_catch_an_unresolvable_param_ref():
    """The one invalid/ fixture the generated schema can never catch on its
    own: a $ref target's existence depends on the document's own `params`,
    a cross-field check JSON Schema has no way to express here. This stays
    the semantic layer's job (sdf/validate.py); pinned so a future change
    that makes this pass isn't mistaken for a regression."""
    schema = _generate.generate_schema()
    jsonschema.validate(_fixture("invalid/unresolvable_param_ref.sdm"), schema)  # does not raise


@pytest.mark.parametrize("name", sorted(p.name for p in (_CONFORMANCE_DIR / "valid").glob("*.sdm")))
def test_generated_schema_accepts_every_valid_conformance_fixture(name):
    """Confirms the new constraints are a pure tightening: every fixture that
    validated before still validates, regardless of which version it
    originally declared."""
    schema = _generate.generate_schema()
    jsonschema.validate(_fixture(f"valid/{name}"), schema)


# ---------------------------------------------------------------------------
# A typo'd discriminator (kind/op/...) reports directly against that field,
# not as an oneOf "matches 0 of N subschemas" / unevaluatedProperties error.
# ---------------------------------------------------------------------------


def test_unknown_kind_error_names_the_kind_field_directly():
    schema = _generate.generate_schema()
    doc = _fixture("invalid/misspelled_primitive_kind.sdm")
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(doc, schema)
    error = exc_info.value
    assert list(error.absolute_path)[-1] == "kind"
    assert "spere" in error.message


# ---------------------------------------------------------------------------
# Gates that must be provable-false: an override missing from
# _DOCUMENT_FIELD_OVERRIDES, or a wire_shape the SDF kwarg mapping doesn't
# know, both fail generation loudly instead of emitting an under-constrained
# schema silently.
# ---------------------------------------------------------------------------


def test_document_field_without_an_override_fails_generation(monkeypatch):
    monkeypatch.setattr(_generate, "_DOCUMENT_FIELD_OVERRIDES", {})
    with pytest.raises(AssertionError, match="_DOCUMENT_FIELD_OVERRIDES"):
        _generate.generate_schema()


def test_sdf_kwarg_with_an_unknown_wire_shape_fails_generation():
    bogus = wire.ParamSpec.__new__(wire.ParamSpec)
    object.__setattr__(bogus, "name", "x")
    object.__setattr__(bogus, "wire_shape", "not-a-real-shape")
    with pytest.raises(AssertionError, match="wire_shape"):
        _generate._value_schema(bogus)


# ---------------------------------------------------------------------------
# Channel checks.
# ---------------------------------------------------------------------------


def test_generate_module_imports_without_jax_or_jsonschema():
    code = (
        "import sys; import software_defined_matter.schema._generate; "
        "assert 'jax' not in sys.modules and 'jsonschema' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_schema_cli_index_flag_prints_the_version_index():
    result = subprocess.run(
        [sys.executable, "-m", "software_defined_matter.schema", "--index"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["latest"] == "0.6"


def test_schema_cli_index_missing_file_fails_loud(monkeypatch, tmp_path, capsys):
    from software_defined_matter.schema import __main__ as schema_main

    monkeypatch.setattr(schema_main, "_INDEX_PATH", tmp_path / "missing.json")
    rc = schema_main.main(["--index"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err.startswith("error:")
