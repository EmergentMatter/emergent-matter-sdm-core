"""Round-trip: build -> to_dict -> from_dict -> to_dict must be equal,
and saving then loading a ``.sdm`` file must reproduce the same object."""

from __future__ import annotations

import pytest

from software_defined_matter import Part, load, save


def test_to_dict_roundtrip(example_part):
    d1 = example_part.to_dict()
    part2 = Part.from_dict(d1)
    assert part2.to_dict() == d1


def test_file_roundtrip(example_part, tmp_path):
    path = tmp_path / "part.sdm"
    save(example_part, path)
    reloaded = load(path)
    assert reloaded.to_dict() == example_part.to_dict()


def test_history_snapshot_persists(example_part, tmp_path):
    example_part.snapshot({"objective_value": 1.23})
    example_part.snapshot({"objective_value": 1.10})
    path = tmp_path / "part.sdm"
    save(example_part, path)
    reloaded = load(path)
    assert len(reloaded.history) == 2
    assert reloaded.history[0]["objective_value"] == 1.23
    assert set(reloaded.history[0]["params"]) == set(example_part.free_param_names())


@pytest.mark.parametrize("validate_schema", [True, False])
def test_part_save_forwards_validation_option(example_part, tmp_path, monkeypatch, validate_schema):
    from software_defined_matter import io

    seen = []
    real_validate = io.validate

    def record_validate(part):
        seen.append(part)
        return real_validate(part)

    monkeypatch.setattr(io, "validate", record_validate)
    path = tmp_path / "wrapper.sdm"
    example_part.save(str(path), b_validate_schema=validate_schema)
    assert len(seen) == int(validate_schema)
    assert load(path).to_dict() == example_part.to_dict()


def test_part_save_validates_by_default(example_part, tmp_path, monkeypatch):
    from software_defined_matter import io

    def reject(part):
        raise ValueError("validation invoked")

    monkeypatch.setattr(io, "validate", reject)
    with pytest.raises(ValueError, match="validation invoked"):
        example_part.save(str(tmp_path / "default.sdm"))
