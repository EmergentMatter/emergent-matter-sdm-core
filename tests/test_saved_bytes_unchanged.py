"""Files with no extension node save to exactly the bytes v1.0.1 wrote."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import software_defined_matter as sdm

_ROOT = Path(__file__).parent.parent
_GOLDEN = json.loads((Path(__file__).parent / "data" / "saved_bytes_v1_0_1.json").read_text())
_SOURCES = {
    "conformance/": _ROOT / "src" / "software_defined_matter" / "schema" / "conformance",
    "examples/": _ROOT / "examples",
}


def _source(key: str) -> Path:
    for prefix, root in _SOURCES.items():
        if key.startswith(prefix):
            return root / key[len(prefix) :]
    raise AssertionError(key)


def test_golden_list_is_not_empty():
    assert len(_GOLDEN) >= 20


@pytest.mark.parametrize("key", sorted(_GOLDEN))
def test_load_then_save_is_byte_identical_to_v1_0_1(key, tmp_path):
    out = tmp_path / "out.sdm"
    sdm.save(sdm.load(str(_source(key))), str(out))
    assert hashlib.sha256(out.read_bytes()).hexdigest() == _GOLDEN[key]
