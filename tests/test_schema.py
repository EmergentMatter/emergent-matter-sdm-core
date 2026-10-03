"""Schema validation of the canonical example and a few negative cases."""

from __future__ import annotations

import subprocess
import sys

import jsonschema
import pytest

from software_defined_matter import (
    LATEST_SCHEMA_VERSION,
    assert_supported,
    supports_version,
    validate,
)
from software_defined_matter.io import _schema_version_for, load_schema
from software_defined_matter.model import (
    KNOWN_SCHEMA_VERSIONS,
    _parse_version,
    min_schema_version_for,
)


def test_schema_loads():
    schema = load_schema()
    assert schema["$id"].endswith(f"sdm-{LATEST_SCHEMA_VERSION}.schema.json")


def test_schema_subpackage_imports_without_jax():
    """The schema subpackage is the subprocess boundary a host without a JAX
    install (Blender's bundled interpreter, for example) reaches this
    package through. A subprocess isolates the check from whatever the rest
    of this test session has already imported."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import software_defined_matter.schema; "
            "sys.exit(1 if 'jax' in sys.modules else 0)",
        ],
        check=False,
    )
    assert result.returncode == 0


@pytest.mark.parametrize(
    "argv, expect_returncode",
    [(["--version", "0.3"], 0), (["--version", "9.9"], 1)],
    ids=["known_version_prints_schema", "unknown_version_fails_loud"],
)
def test_schema_cli_returns_expected_exit_code(argv, expect_returncode):
    result = subprocess.run(
        [sys.executable, "-m", "software_defined_matter.schema", *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expect_returncode
    if expect_returncode == 0:
        assert '"$id"' in result.stdout
    else:
        assert result.stderr.startswith("error:")


def test_example_part_validates(example_part):
    validate(example_part)  # should not raise


def test_missing_name_rejected():
    bad = {"schema_version": "0.1"}
    with pytest.raises(jsonschema.ValidationError):
        validate(bad)


def test_future_schema_version_warns_but_validates():
    """Lenient policy: an UNKNOWN-but-well-formed schema_version validates
    (against the permissive 0.1 shape) and emits a UserWarning. Migration is
    best-effort."""
    doc = {"schema_version": "0.9", "name": "x"}
    with pytest.warns(UserWarning, match="schema_version"):
        validate(doc)


def test_known_schema_versions_load_without_warning(recwarn):
    """0.1 / 0.2 / 0.3 are all KNOWN versions now: none of them is 'drift',
    so none may warn. Each validates against its own version's schema."""
    for sv in KNOWN_SCHEMA_VERSIONS:
        validate({"schema_version": sv, "name": "x"})
    assert not [w for w in recwarn.list if "schema_version" in str(w.message)]


def test_malformed_schema_version_rejected():
    bad = {"schema_version": "not-a-version", "name": "x"}
    with pytest.raises(jsonschema.ValidationError):
        validate(bad)


def test_bad_objective_sense_rejected(example_part):
    doc = example_part.to_dict()
    doc["objectives"][0]["sense"] = "wiggle"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


# ---------------------------------------------------------------------------
# supports_version / assert_supported: core owns version ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sv", KNOWN_SCHEMA_VERSIONS)
def test_supports_version_accepts_every_known_version(sv):
    assert supports_version(sv) is True


@pytest.mark.parametrize(
    "sv",
    ["0.0", "0.9", "0.10", "0.15"],
    ids=["below_floor", "above_ceiling", "above_ceiling_two_digit", "between_known_and_ceiling"],
)
def test_supports_version_rejects_every_version_without_a_schema_file(sv):
    """'Supported' means a schema file exists to validate against, so the
    answer is membership in KNOWN_SCHEMA_VERSIONS, not a ceiling compare.

    Each id is a distinct way a ceiling compare gets it wrong. '0.0' is
    below the floor: no schema file has ever covered it, but it is
    numerically <= '0.3'. '0.15' sits inside the range and was never
    released. Both would read as supported under a compare, while
    io._schema_version_for -- which dispatches on this same membership --
    would quietly validate such a document against the 0.1 fallback schema.
    That disagreement is the version confusion this contract exists to
    close. '0.9' and '0.10' are the ordering trap from the other side, and
    stay pinned here because a future range check would have to get both
    right."""
    assert supports_version(sv) is False


@pytest.mark.parametrize("sv", ["not-a-version", "1", "1.2.3", ""])
def test_supports_version_rejects_malformed_strings(sv):
    assert supports_version(sv) is False


def test_supports_version_agrees_with_io_schema_dispatch():
    """The invariant behind the membership rule: a version supports_version
    accepts is exactly one io validates against on its own terms, and one it
    rejects is exactly one io falls back for. If these two ever diverge, a
    consumer is told it is on a version it is not being checked against."""
    for sv in [*KNOWN_SCHEMA_VERSIONS, "0.0", "0.15", "0.9", "not-a-version"]:
        dispatched = _schema_version_for({"schema_version": sv})
        assert (dispatched == sv) is supports_version(sv)


def test_assert_supported_passes_for_known_version():
    assert_supported({"schema_version": "0.2", "name": "x"})  # should not raise


@pytest.mark.parametrize(
    "doc",
    [
        {"schema_version": "9.9", "name": "x"},
        {"schema_version": "0.0", "name": "x"},
        {"name": "x"},
    ],
    ids=["version_beyond_ceiling", "version_below_floor", "missing_version"],
)
def test_assert_supported_names_version_and_known_set(doc):
    """The error has to name what this package does accept, not only that it
    refused: the caller's next move is to pick a version off that list."""
    with pytest.raises(
        ValueError,
        match=r"Known versions: \['0\.1', '0\.2', '0\.3', '0\.4', '0\.5', '0\.6'\]",
    ):
        assert_supported(doc)
    with pytest.raises(ValueError, match=repr(doc.get("schema_version"))):
        assert_supported(doc)


# ---------------------------------------------------------------------------
# Version ordering: still owned here, now exercised where it actually matters
# ---------------------------------------------------------------------------


def test_version_ordering_is_numeric_not_lexicographic():
    """The ordering trap: a bare string compare puts '0.10' before '0.2', so
    min_schema_version_for would pick the wrong maximum the day a two-digit
    minor exists. supports_version no longer exercises this (it is a set
    membership now), so the trap is pinned directly against the comparison
    key that min_schema_version_for still sorts by."""
    assert max(["0.2", "0.10"], key=_parse_version) == "0.10"
    assert max(["0.9", "0.10"], key=_parse_version) == "0.10"
    assert sorted(KNOWN_SCHEMA_VERSIONS, key=_parse_version) == list(KNOWN_SCHEMA_VERSIONS)
    assert max(KNOWN_SCHEMA_VERSIONS, key=_parse_version) == LATEST_SCHEMA_VERSION


@pytest.mark.parametrize("sv", ["not-a-version", "1", "1.2.3", "", "0.x"])
def test_parse_version_rejects_malformed_strings(sv):
    with pytest.raises(ValueError, match="Malformed schema version"):
        _parse_version(sv)


# ---------------------------------------------------------------------------
# min_schema_version_for: the oldest schema file that can validate the part
# ---------------------------------------------------------------------------


def test_port_content_requires_schema_0_5_even_without_priors(example_part):
    """Explicit port frames require the new reader regardless of parameter priors."""
    assert min_schema_version_for(example_part) == "0.5"


# ---------------------------------------------------------------------------
# Package root public surface: __all__ and dir() must agree
# ---------------------------------------------------------------------------


def test_every_root_export_is_reachable_and_listed_in_dir():
    """A lazily-resolved export (the JAX-importing audit helpers) that only
    satisfies attribute access and not dir() breaks REPL tab-completion and
    dir()-based introspection tools -- a module __getattr__ needs a paired
    __dir__ per PEP 562, and this pins that pairing."""
    import software_defined_matter as sdm

    visible = dir(sdm)
    for name in sdm.__all__:
        assert name in visible, f"{name!r} missing from dir(software_defined_matter)"
        assert hasattr(sdm, name), f"{name!r} not reachable as an attribute"
