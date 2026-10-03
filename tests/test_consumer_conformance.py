"""Consumer conformance corpus — structural and numerical fingerprints.

``schema/conformance/consumers/`` pins the emission *shape* a downstream
host depends on (lib helpers, macros, component order, polygon payload,
controls) plus a few JAX sample values. It is the anti-drift gate that
``valid/`` / ``invalid/`` cannot be: those only check schema acceptance.

Fingerprints live next to the fixtures under ``consumers/expected/``. They
are deliberately not byte-exact golden shaders — only portable structure
and sample numbers. Load/save and CLI round-trips are asserted per fixture.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import jax.numpy as jnp
import pytest

from software_defined_matter import Part
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.glsl.__main__ import _write_artifacts
from software_defined_matter.io import load, save, validate
from software_defined_matter.sdf.compile import make_sdf_closure

CONSUMERS = files("software_defined_matter.schema.conformance.consumers")


def _sdm_fixtures() -> list[Traversable]:
    return sorted(
        (p for p in CONSUMERS.iterdir() if p.name.endswith(".sdm")),
        key=lambda p: p.name,
    )


def _expected(stem: str) -> dict:
    text = (CONSUMERS / "expected" / f"{stem}.json").read_text()
    return json.loads(text)


def _part_from(path: Traversable) -> Part:
    """Materialise a corpus fixture without requiring a real filesystem path.

    ``importlib.resources`` may hand back a zip entry; ``io.load`` needs an
    openable ``Path``. The wire JSON is the source of truth either way.
    """
    return Part.from_dict(json.loads(path.read_text()))


SDMS = _sdm_fixtures()
IDS = [p.name for p in SDMS]


def _lib_fns(scene: str) -> list[str]:
    refs = set(re.findall(r"\b((?:sdf_|op_|field_|sdm_)[A-Za-z0-9_]+)\s*\(", scene))
    return sorted(r for r in refs if not re.match(r"^(sdf_n\d|field_n\d|sdf_scene)", r))


def _control_classes(controls: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in controls:
        out.setdefault(c["class"], []).append(c["param"])
    return {k: sorted(v) for k, v in out.items()}


def test_consumer_corpus_is_not_empty() -> None:
    assert SDMS, "no consumers/ fixtures discovered — did the corpus ship?"
    for path in SDMS:
        stem = path.name[: -len(".sdm")]
        assert (CONSUMERS / "expected" / f"{stem}.json").is_file(), (
            f"missing expected fingerprint for {path.name}"
        )


@pytest.mark.parametrize("path", SDMS, ids=IDS)
def test_consumer_fixture_validates_and_names_itself(path: Traversable) -> None:
    doc = json.loads(path.read_text())
    validate(doc)
    stem = path.name[: -len(".sdm")]
    assert doc["name"] == f"conformance_consumer_{stem}"


@pytest.mark.parametrize("path", SDMS, ids=IDS)
def test_consumer_load_save_roundtrip(path: Traversable, tmp_path: Path) -> None:
    fp = _expected(path.name[: -len(".sdm")])
    if not fp["roundtrip"]["load_save"]:
        pytest.skip("fingerprint disables load/save")
    part = _part_from(path)
    out = tmp_path / path.name
    save(part, out)
    assert load(out).to_dict() == part.to_dict()
    authored = json.loads(path.read_text())
    if "kinematics" in authored:
        assert load(out).kinematics == authored["kinematics"]


@pytest.mark.parametrize("path", SDMS, ids=IDS)
def test_consumer_emission_fingerprint(path: Traversable) -> None:
    stem = path.name[: -len(".sdm")]
    fp = _expected(stem)
    part = _part_from(path)
    em = fp["emission"]

    if em["status"] == "refused":
        with pytest.raises(ValueError, match=re.escape(em["refuse_substr"])):
            emit_glsl(part)
        return

    assert em["status"] == "ok"
    emission = emit_glsl(part)
    assert emission.entry_point == em["entry_point"]
    assert _lib_fns(emission.scene_source) == em["lib_functions"]
    assert [c["label"] for c in emission.components] == em["component_labels"]
    assert [c["machine"] for c in emission.components] == em["component_machines"]
    assert emission.poly_max_n == em["poly_max_n"]
    assert len(emission.poly_table) == em["poly_table_len"]
    assert emission.poly_tex_width == em["poly_tex_width"]
    assert len(emission.grid_table) == em.get("grid_table_len", 0)
    assert _control_classes(emission.controls) == em["control_classes"]
    assert [u.source_param for u in emission.uniforms] == em["uniform_sources"]

    for needle in em.get("scene_needles", []):
        assert needle in emission.scene_source, f"missing scene needle {needle!r}"
    for macro in em.get("macros_required", []):
        assert f"#define {macro}" in emission.lib_source, f"missing macro {macro}"
    for macro in em.get("macros_forbidden", []):
        # lib.glsl may mention the name inside #ifdef; the emitter must not
        # *enable* the table path by defining SDM_POLY_TABLE.
        assert f"#define {macro}\n" not in emission.lib_source, (
            f"forbidden macro {macro} is defined"
        )


@pytest.mark.parametrize("path", SDMS, ids=IDS)
def test_consumer_jax_numerical_fingerprint(path: Traversable) -> None:
    stem = path.name[: -len(".sdm")]
    fp = _expected(stem)
    numerical = fp.get("numerical")
    if not numerical or "jax_values" not in numerical:
        pytest.skip("no jax numerical fingerprint")
    part = _part_from(path)
    points = jnp.asarray(numerical["points"], dtype=jnp.float32)
    free = jnp.asarray(numerical.get("free_vec", []), dtype=jnp.float32)
    fn = make_sdf_closure(part.materials[0].sdf_tree, part)
    got = [float(v) for v in fn(points, free)]
    assert got == pytest.approx(numerical["jax_values"], abs=numerical.get("jax_atol", 1e-5))


@pytest.mark.parametrize(
    "stem",
    ["field_radial_amplitude", "field_sin_xyz_amplitude"],
)
def test_consumer_field_amplitude_call_site_is_correct(stem: str) -> None:
    """Amplitude on the wire must appear at the emitted field call site."""
    fp = _expected(stem)
    site = fp["numerical"]["emitted_call_site"]
    assert site["assert"] == "correct"
    part = _part_from(CONSUMERS / f"{stem}.sdm")
    scene = emit_glsl(part).scene_source
    m = re.search(site["pattern"], scene)
    assert m, f"call site not found for {stem}"
    got = m.group(1).strip()
    assert got == site["correct"]


@pytest.mark.parametrize("path", SDMS, ids=IDS)
def test_consumer_cli_roundtrip(path: Traversable, tmp_path: Path) -> None:
    stem = path.name[: -len(".sdm")]
    fp = _expected(stem)
    if not fp["roundtrip"].get("cli", False):
        pytest.skip("fingerprint disables CLI round-trip")

    part = _part_from(path)
    sdm = tmp_path / path.name
    out = tmp_path / "glsl"
    save(part, sdm)

    # In-process write must match emit_glsl (same path the CLI uses).
    emission = emit_glsl(part)
    _write_artifacts(emission, out)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["entry_point"] == emission.entry_point
    assert meta["grid_table"] == emission.grid_table
    assert meta["poly_max_n"] == emission.poly_max_n
    assert meta.get("poly_table", []) == pytest.approx(emission.poly_table)
    assert meta.get("poly_tex_width", 0) == emission.poly_tex_width
    assert meta.get("components", []) == emission.components
    assert (out / "sdf_scene.glsl").read_text() == emission.scene_source

    # Module entry point must produce the same meta payload.
    cli_out = tmp_path / "cli"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "software_defined_matter.glsl",
            str(sdm),
            "--out",
            str(cli_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    cli_meta = json.loads((cli_out / "meta.json").read_text())
    assert cli_meta["poly_max_n"] == emission.poly_max_n
    assert cli_meta.get("poly_table", []) == pytest.approx(emission.poly_table)
    assert cli_meta.get("poly_tex_width", 0) == emission.poly_tex_width
    assert cli_meta.get("components", []) == emission.components
