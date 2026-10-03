"""raster_field: JAX eval, emission raw payload, and mutation gates (DR-0003)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_raster_field
from software_defined_matter.glsl.emit import emit_glsl
from software_defined_matter.sdf.bake import bake_raster_field
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.raster import (
    decode_raster_values,
    encode_raster_data,
    require_literal_raster_params,
)


def _linear_ramp_values(nx=5, ny=5, nz=5, spacing=1.0):
    """Axis-aligned ramp: value at sample (i,j,k) = i*spacing (exact on lattice)."""
    vals = np.zeros((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                vals[k, j, i] = i * spacing
    return vals


def test_codec_round_trip_and_rejects_ref():
    vals = _linear_ramp_values()
    data = encode_raster_data(vals)
    params = {
        "origin": [0.0, 0.0, 0.0],
        "spacing": [1.0, 1.0, 1.0],
        "dims": [5, 5, 5],
        "encoding": "f32le",
        "data": data,
    }
    got = decode_raster_values(params)
    assert np.allclose(got, vals)
    with pytest.raises(ValueError, match="literal"):
        require_literal_raster_params({**params, "dims": {"$ref": "n"}})


def test_jax_matches_lattice_samples():
    spacing = 1.0
    vals = _linear_ramp_values(spacing=spacing)
    node = sdf_raster_field([0, 0, 0], spacing, vals)
    part = Part(
        name="ramp",
        materials=[MaterialRegion(material_id=1, name="a", sdf_tree=node)],
    )
    sdf = make_sdf_closure(node, part)
    # Lattice points must be exact
    pts = np.array([[i, 2.0, 2.0] for i in range(5)], dtype=np.float32)
    out = np.asarray(sdf(pts))
    assert np.allclose(out, pts[:, 0], atol=1e-5)


def test_out_of_box_is_clamped_plus_distance():
    vals = np.zeros((3, 3, 3), dtype=np.float32)
    node = sdf_raster_field([0, 0, 0], 1.0, vals)
    part = Part(
        name="box",
        materials=[MaterialRegion(material_id=1, name="a", sdf_tree=node)],
    )
    sdf = make_sdf_closure(node, part)
    # Outside on +x: clamp to face value 0 + distance 3
    p = np.array([[5.0, 1.0, 1.0]], dtype=np.float32)
    assert float(np.asarray(sdf(p))[0]) == pytest.approx(3.0, abs=1e-5)


def test_bake_sphere_and_emit_raw_payload():
    def sphere(pts):
        return jnp.linalg.norm(pts, axis=-1) - 1.0

    node = bake_raster_field(
        sphere,
        bbox=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        voxel=0.5,
        pad_voxels=1,
    )
    part = Part(
        name="sph",
        materials=[MaterialRegion(material_id=1, name="a", sdf_tree=node)],
        metadata={"bbox": [[-2, -2, -2], [2, 2, 2]]},
    )
    sdf = make_sdf_closure(node, part)
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    out = np.asarray(sdf(pts))
    assert out[0] < 0.0
    assert out[1] > 0.0

    emission = emit_glsl(part)
    assert "sdf_raster_field" in emission.scene_source
    assert "SDM_GRID_TABLE" in emission.lib_source
    nx, ny, nz = node["params"]["dims"]
    assert len(emission.grid_table) == nx * ny * nz
    assert emission.grid_tex_width == 4096

    # Mutation gate: corrupt one sample → field must move
    flat = np.array(emission.grid_table, dtype=np.float32).copy()
    mid = flat.size // 2
    flat[mid] += 0.25
    mutated = sdf_raster_field(
        node["params"]["origin"],
        node["params"]["spacing"],
        flat.reshape(nz, ny, nx),
    )
    sdf_m = make_sdf_closure(mutated, part)
    assert not np.allclose(np.asarray(sdf(pts)), np.asarray(sdf_m(pts)))


def test_lipschitz_bound_sqrt3_on_kink():
    """Trilinear of a |x|-like field can reach gradient magnitude sqrt(3)."""
    # Build a field whose samples are distance to origin on a coarse grid
    origin = (-1.0, -1.0, -1.0)
    spacing = 1.0
    dims = (3, 3, 3)
    vals = np.zeros(dims[::-1], dtype=np.float32)
    for k in range(3):
        for j in range(3):
            for i in range(3):
                p = np.array(origin) + spacing * np.array([i, j, k])
                vals[k, j, i] = np.linalg.norm(p)
    node = sdf_raster_field(origin, spacing, vals, step_scale=1.0 / math.sqrt(3.0))
    assert node["params"]["step_scale"] == pytest.approx(1.0 / math.sqrt(3.0))


def test_distinct_named_nodes_share_samples_and_cli_preserves_fetch_order(tmp_path):
    import base64
    import copy
    import json

    from software_defined_matter.glsl.__main__ import _write_artifacts

    values = _linear_ramp_values(3, 3, 3)
    first = sdf_raster_field([0, 0, 0], 1.0, values)
    second = copy.deepcopy(first)
    first["name"], second["name"] = "first", "second"
    tree = {"type": "op", "op": "union", "children": [first, second]}
    part = Part(name="shared", materials=[MaterialRegion(material_id=1, name="r", sdf_tree=tree)])
    emission = emit_glsl(part)
    assert emission.grid_table == values.reshape(-1).tolist()
    _write_artifacts(emission, tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    decoded = np.frombuffer(base64.b64decode(meta["grid_table_b64"]), dtype="<f4")
    np.testing.assert_array_equal(decoded, values.reshape(-1))
    assert meta["grid_table"] == emission.grid_table


def test_bake_bound_covers_tangential_gradient_outside_the_box():
    node = bake_raster_field(
        lambda points: points[:, 1] + 10.0, ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)), 1.0, pad_voxels=0
    )
    # At x > 2 the continuation is y + 10 + (x - 2), with gradient (1, 1, 0).
    bound = node["params"]["provenance"]["bake"]["lipschitz"]
    assert bound >= math.sqrt(2.0) - 1e-7
    assert node["params"]["step_scale"] <= 1.0 / math.sqrt(2.0) + 1e-7


def test_negative_boundary_samples_expand_the_geometry_bbox():
    from software_defined_matter.sdf.bbox import infer_sdf_bbox

    node = sdf_raster_field([0, 0, 0], 1.0, -np.ones((3, 3, 3), dtype=np.float32))
    part = Part(
        name="extended", materials=[MaterialRegion(material_id=1, name="body", sdf_tree=node)]
    )
    lo, hi = infer_sdf_bbox(node, part)
    assert lo == pytest.approx((-1, -1, -1))
    assert hi == pytest.approx((3, 3, 3))


def test_baked_node_saves_under_its_required_schema(tmp_path):
    from software_defined_matter.io import load, save

    node = bake_raster_field(
        lambda points: jnp.linalg.norm(points, axis=-1) - 1, ((-2, -2, -2), (2, 2, 2)), 1.0
    )
    part = Part(name="baked", materials=[MaterialRegion(material_id=1, name="body", sdf_tree=node)])
    path = tmp_path / "baked.sdm"
    save(part, path)
    assert load(path).to_dict() == part.to_dict()
    assert part.to_dict()["schema_version"] == "0.3"
