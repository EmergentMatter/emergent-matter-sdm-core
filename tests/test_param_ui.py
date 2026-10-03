"""Param.ui exploration/presentation spec: round-trip + schema validation.

The ``ui`` block carries how a param should be *explored* (slider step,
scrub range, panel grouping, topology role). It is presentation metadata:
optimisers ignore it, and a Part without any ``ui`` serialises exactly as
before the field existed (no ``"ui"`` key in the JSON).
"""

from __future__ import annotations

import jsonschema
import pytest

from software_defined_matter import Param, load, save, validate

UI = {
    "step": 0.1,
    "explore_bounds": [5.0, 80.0],
    "group": "rings",
    "order": 1,
}


def test_param_ui_roundtrips():
    p = Param("outer_radius", 25.0, free=True, bounds=(10.0, 60.0), unit="mm", ui=dict(UI))
    d = p.to_dict()
    assert d["ui"] == UI
    assert Param.from_dict(d).ui == UI


def test_param_without_ui_serialises_as_before():
    """No 'ui' key at all when unset: existing .sdm files stay byte-stable."""
    d = Param("outer_radius", 25.0).to_dict()
    assert "ui" not in d
    assert Param.from_dict(d).ui is None


def test_part_file_roundtrip_preserves_ui(example_part, tmp_path):
    example_part.add_param(Param("stages", 1.0, unit="count", ui={"role": "topology", "step": 1}))
    path = tmp_path / "part.sdm"
    save(example_part, path)
    reloaded = load(path)
    assert reloaded.params["stages"].ui == {"role": "topology", "step": 1}
    assert reloaded.to_dict() == example_part.to_dict()


def test_schema_accepts_ui(example_part):
    doc = example_part.to_dict()
    name = next(iter(doc["params"]))
    doc["params"][name]["ui"] = dict(UI)
    validate(doc)  # should not raise


@pytest.mark.parametrize(
    "bad_ui",
    [
        {"step": 0},  # step must be > 0
        {"step": -0.1},
        {"explore_bounds": [1.0]},  # needs [lo, hi]
        {"role": "livewire"},  # only "topology" is defined
        {"stepp": 0.1},  # unknown keys rejected
    ],
)
def test_schema_rejects_bad_ui(example_part, bad_ui):
    doc = example_part.to_dict()
    name = next(iter(doc["params"]))
    doc["params"][name]["ui"] = bad_ui
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)
