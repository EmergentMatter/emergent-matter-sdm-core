"""Shared sweep fixtures execute inline and tabled fields with live profiles.

Strict samples pin distances and occupancy. Near-tie samples instead pin the
finite set of segment candidates, because float32 rounding can change which
almost-equidistant segment wins without either backend changing its algorithm.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import Part
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.glsl.emit import _sweep_frame_arrays
from software_defined_matter.io import load, save, validate
from software_defined_matter.sdf.compile import make_sdf_closure
from tests.test_glsl_python_parity import shader_runtime as _shader_runtime

shader_runtime = _shader_runtime
CORPUS = files("software_defined_matter.schema.conformance.consumers") / "sweep"
EXPECTED = json.loads((CORPUS / "expected.json").read_text())
# Float32 arithmetic across JAX and GLSL; not a geometric approximation budget.
DISTANCE_TOL = 1e-5


def _part(name):
    return Part.from_dict(json.loads((CORPUS / EXPECTED[name]["fixture"]).read_text()))


@pytest.mark.parametrize("name", ["inline", "tabled"])
def test_shared_sweep_documents_validate_and_roundtrip(name, tmp_path):
    document = json.loads((CORPUS / EXPECTED[name]["fixture"]).read_text())
    validate(document)
    part = Part.from_dict(document)
    destination = tmp_path / EXPECTED[name]["fixture"]
    save(part, destination)
    assert load(destination).to_dict() == part.to_dict()


@pytest.mark.parametrize("name", ["inline", "tabled"])
def test_shared_sweep_payload_and_live_profile_match_executed_glsl(shader_runtime, name):
    expected = EXPECTED[name]
    part = _part(name)
    emission = emit_glsl(part)
    assert emission.sweep_max_s == expected["sweep_max_s"]
    assert emission.sweep_tex_width == expected["sweep_tex_width"]
    assert len(emission.sweep_table) // 4 == expected["sweep_table_texels"]
    control = next(c for c in emission.controls if c["param"] == expected["param"])
    assert control["class"] == "live"
    assert control["uniform"] == expected["uniform"]
    if name == "tabled":
        headers = [int(h) for h in re.findall(r"sdm_sweep_tab\(p, (\d+)\)", emission.scene_source)]
        assert headers == expected["headers"]
        # Both a frame and the later header must be fetched after a row boundary.
        assert len(emission.sweep_table) // 4 > 2 * emission.sweep_tex_width
        assert headers[1] > emission.sweep_tex_width
    points = np.asarray(expected["points"], dtype=np.float32)
    for case in expected["live_cases"]:
        document = part.to_dict()
        document["params"][expected["param"]]["value"] = case["value"]
        updated = Part.from_dict(document)
        tree = updated.materials[0].sdf_tree
        closure = make_sdf_closure(tree, updated)
        actual = np.asarray(closure(jnp.asarray(points), jnp.zeros((0,))))
        glsl = shader_runtime.evaluate(
            emission, points, uniforms={expected["uniform"]: case["value"]}
        )[:, 3]
        np.testing.assert_allclose(actual, case["distances"], rtol=0, atol=DISTANCE_TOL)
        np.testing.assert_allclose(glsl, actual, rtol=0, atol=DISTANCE_TOL)
        np.testing.assert_array_equal(glsl < 0, actual < 0)


def test_shared_near_tie_outputs_belong_to_declared_segment_candidates(shader_runtime):
    part = _part("inline")
    tree = part.materials[0].sdf_tree
    policy = EXPECTED["inline"]["near_ties"]
    emission = emit_glsl(part)
    params = tree["params"]
    A, T, L, N, B = _sweep_frame_arrays(
        params["path"], params["path_kind"], False, params["frame"], None
    )
    closure = make_sdf_closure(tree, part)
    for case in policy["cases"]:
        point = np.asarray(case["point"], dtype=np.float32)
        s = np.sum((point - A) * T, axis=-1)
        perp = point - (A + np.clip(s, 0, L)[:, None] * T)
        squared = np.sum(perp * perp, axis=-1)
        margin = policy["relative_squared_distance_margin"] * (1 + squared.min())
        candidates = np.flatnonzero(squared - squared.min() <= margin)
        assert candidates.tolist() == case["segments"]
        assert len(candidates) > 1
        uv = np.stack([np.sum(perp * N, axis=-1), np.sum(perp * B, axis=-1)], axis=-1)[candidates]
        over_lo = -s[candidates]
        over_hi = s[candidates] - L[candidates]
        over = np.maximum(over_lo, over_hi)
        # The fixture is an open path: its two true ends keep a flat cap and
        # every other vertex is a ball joint, so the profile is evaluated on
        # the query pushed out to the hypotenuse of radius and overshoot.
        n_seg = len(L)
        flat = ((candidates == 0) & (over_lo > 0)) | ((candidates == n_seg - 1) & (over_hi > 0))
        radius = np.linalg.norm(uv, axis=-1)
        hyp = np.sqrt(radius**2 + np.maximum(over, 0) ** 2)
        joint = (over > 0) & ~flat
        uv = np.where(joint[:, None], uv * (hyp / np.maximum(radius, 1e-9))[:, None], uv)
        q = np.abs(uv) - [policy["param_value"], 0.15]
        profile = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)
        axial = np.where(flat & (over > 0), over, -1e9)
        w = np.stack([profile, axial], axis=-1)
        distances = np.linalg.norm(np.maximum(w, 0), axis=-1) + np.minimum(w.max(axis=-1), 0)
        np.testing.assert_allclose(distances, case["distances"], rtol=0, atol=DISTANCE_TOL)
        # The fixture must distinguish candidate selection from loose allclose.
        assert np.ptp(distances) > 100 * policy["distance_tolerance"]
        glsl = shader_runtime.evaluate(emission, point[None, :])[0, 3]
        jax = float(closure(jnp.asarray(point[None, :]), jnp.zeros((0,)))[0])
        for actual in [glsl, jax]:
            assert np.min(np.abs(distances - actual)) <= policy["distance_tolerance"]
            assert bool(actual < 0) == case["inside"]
        assert np.all((distances < 0) == case["inside"])
