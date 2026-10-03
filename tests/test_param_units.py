"""Strict Param units: controlled vocabulary, required on the wire.

In-memory a ``Param`` may carry ``unit=""`` while under construction, but a
non-empty unit must come from ``ALLOWED_UNITS`` (typos raise at construction,
not as a downstream mis-scale), and the ``.sdm`` schema requires a real unit
on every param.
"""

from __future__ import annotations

import jsonschema
import pytest

from software_defined_matter import Param, validate
from software_defined_matter.model import ALLOWED_UNITS


def test_known_units_accepted():
    for unit in sorted(ALLOWED_UNITS):
        Param("x", 1.0, unit=unit)  # should not raise


def test_empty_unit_tolerated_in_memory():
    assert Param("x", 1.0).unit == ""


@pytest.mark.parametrize("bad", ["milimeter", "MM", "inches", "Nmm", "mm3"])
def test_unknown_unit_raises_at_construction(bad):
    with pytest.raises(ValueError, match="unknown unit"):
        Param("x", 1.0, unit=bad)


def test_schema_requires_unit(example_part):
    doc = example_part.to_dict()
    name = next(iter(doc["params"]))
    del doc["params"][name]["unit"]
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_schema_rejects_empty_unit(example_part):
    doc = example_part.to_dict()
    name = next(iter(doc["params"]))
    doc["params"][name]["unit"] = ""
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_schema_enum_matches_model_vocabulary():
    """The schema enum and ALLOWED_UNITS must stay in lockstep: extending
    one without the other silently splits what's valid in memory vs on disk."""
    from software_defined_matter.io import load_schema

    enum = set(load_schema()["$defs"]["param"]["properties"]["unit"]["enum"])
    assert enum == set(ALLOWED_UNITS)
