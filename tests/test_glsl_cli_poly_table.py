"""CLI meta.json must keep the polygon and sweep tables that emit_glsl returns.

Without this, shaders that call ``sdm_polygon_2d_tab`` or ``sdm_sweep_tab``
ship with an empty host payload: the program compiles, the table is gone, and
the field is wrong.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from software_defined_matter import (
    MaterialRegion,
    Part,
    sdf_2d_to_3d,
    sdf_primitive,
    sdf_sweep,
)
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.glsl.__main__ import _write_artifacts
from software_defined_matter.io import save


def _big_poly(n: int, r: float = 5.0) -> dict:
    return sdf_primitive(
        "polygon_2d",
        vertices=[
            [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)] for i in range(n)
        ],
    )


def _part(tree) -> Part:
    return Part(
        name="cli_poly",
        params={},
        materials=[MaterialRegion(material_id=1, name="body", sdf_tree=tree)],
        metadata={"bbox": [[-10.0, -10.0, -2.0], [10.0, 10.0, 2.0]]},
    )


def test_write_artifacts_keeps_in_memory_poly_table(tmp_path: Path) -> None:
    em = emit_glsl(_part(sdf_2d_to_3d("extrusion", _big_poly(600), h=1.0)))
    assert em.poly_table, "fixture must exercise table mode"

    out = tmp_path / "out"
    _write_artifacts(em, out)
    meta = json.loads((out / "meta.json").read_text())

    assert meta["poly_table"] == pytest.approx(em.poly_table)
    assert meta["poly_tex_width"] == em.poly_tex_width
    assert meta["poly_tex_width"] > 0
    assert meta["poly_max_n"] == em.poly_max_n
    assert meta["components"] == em.components


def test_write_artifacts_inline_mode_still_reports_empty_table(tmp_path: Path) -> None:
    em = emit_glsl(_part(sdf_2d_to_3d("extrusion", _big_poly(8), h=1.0)))
    assert em.poly_table == []

    out = tmp_path / "out"
    _write_artifacts(em, out)
    meta = json.loads((out / "meta.json").read_text())

    assert meta["poly_table"] == []
    assert meta["poly_tex_width"] == 0


def test_cli_meta_json_matches_emit_glsl(tmp_path: Path) -> None:
    """End-to-end: the module entry point must not drop what emit returns."""
    part = _part(sdf_2d_to_3d("extrusion", _big_poly(600), h=1.0))
    sdm = tmp_path / "poly.sdm"
    out = tmp_path / "glsl"
    save(part, sdm)

    em = emit_glsl(part)
    proc = subprocess.run(
        [sys.executable, "-m", "software_defined_matter.glsl", str(sdm), "--out", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    meta = json.loads((out / "meta.json").read_text())
    assert meta["poly_table"] == pytest.approx(em.poly_table)
    assert meta["poly_tex_width"] == em.poly_tex_width


def _long_sweep(n: int, r: float = 5.0) -> dict:
    path = [
        [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0]
        for i in range(n)
    ]
    return sdf_sweep(sdf_primitive("circle_2d", r=0.3), path, path_kind="polyline")


def test_write_artifacts_keeps_in_memory_sweep_table(tmp_path: Path) -> None:
    em = emit_glsl(_part(_long_sweep(600)))
    assert em.sweep_table, "fixture must exercise sweep table mode"

    out = tmp_path / "out"
    _write_artifacts(em, out)
    meta = json.loads((out / "meta.json").read_text())

    assert meta["sweep_table"] == pytest.approx(em.sweep_table)
    assert meta["sweep_tex_width"] == em.sweep_tex_width > 0
    assert meta["sweep_max_s"] == em.sweep_max_s == 599


def test_write_artifacts_inline_sweep_still_reports_empty_table(tmp_path: Path) -> None:
    em = emit_glsl(_part(_long_sweep(20)))
    assert em.sweep_table == []

    out = tmp_path / "out"
    _write_artifacts(em, out)
    meta = json.loads((out / "meta.json").read_text())

    assert meta["sweep_table"] == []
    assert meta["sweep_tex_width"] == 0
    assert meta["sweep_max_s"] == 19
