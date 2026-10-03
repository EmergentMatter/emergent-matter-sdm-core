"""A shear_linear of a flat band IS the clamped-ramp plate, and its bounds hold under live params.

``shear_linear`` lifts the child's material along +Z by a clamped linear ramp. It exists so a
plate welded between two rings can be authored as a flat band sheared by a live ``$ref`` (the
stage height) instead of an 8-vertex polygon whose vertices must be literals, which is what
forced a staged flexure's ``stage_height`` to rebuild on every drag. The golden test below pins
that claim against the octagon plate those dimensions emit.

Sign and ramp assertions are made on the query map itself (via a spy child) or on geometry,
never on a formula restated locally, per the lesson recorded in test_twist_deforms.py.
"""

from __future__ import annotations

import math
import re

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import Param, Part, sdf_deform, sdf_primitive, sdf_transform
from software_defined_matter.glsl.emit import emit_glsl, load_lib_glsl
from software_defined_matter.model import MaterialRegion, sdf_2d_to_3d
from software_defined_matter.sdf import sdf_ops as ops
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.lipschitz import UNKNOWN, infer_sdf_max_rate


def _mapped(p: np.ndarray, **kw) -> np.ndarray:
    """The query point op_shear_linear hands its child."""
    seen = {}

    def spy(q):
        seen["q"] = q
        return jnp.zeros(q.shape[:-1])

    ops.op_shear_linear(spy, jnp.asarray(p), **kw)
    return np.asarray(seen["q"], dtype=np.float64)


RAMP = {"u_axis": [1.0, 0.0], "u0": -2.0, "u1": 2.0, "dz0": 0.5, "dz1": 3.5}


def test_the_material_rises_by_each_end_and_linearly_between():
    p = np.array([[-2.0, 0.3, 1.0], [0.0, 0.3, 1.0], [2.0, 0.3, 1.0]])
    q = _mapped(p, **RAMP)
    # The query is pulled DOWN by the rise, so the material appears that much higher.
    np.testing.assert_allclose(p[:, 2] - q[:, 2], [0.5, 2.0, 3.5], atol=1e-9)
    np.testing.assert_allclose(q[:, :2], p[:, :2], atol=0)  # XY untouched


def test_the_ramp_clamps_past_each_end():
    p = np.array([[-9.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    np.testing.assert_allclose(p[:, 2] - _mapped(p, **RAMP)[:, 2], [0.5, 3.5], atol=1e-9)


def test_the_axis_selects_the_ramp_direction_and_its_scale_is_free():
    p = np.array([[0.0, 2.0, 0.0]])
    rise_y = p[:, 2] - _mapped(p, **{**RAMP, "u_axis": [0.0, 5.0]})[:, 2]
    rise_x = p[:, 2] - _mapped(p, **RAMP)[:, 2]
    np.testing.assert_allclose(rise_y, [3.5], atol=1e-9)  # y = 2 is the far end along +Y
    np.testing.assert_allclose(rise_x, [2.0], atol=1e-9)  # x = 0 is mid-ramp along +X


def test_a_zero_axis_has_the_same_finite_fallback_as_the_gpu_helper():
    p = np.array([[-3.0, 7.0, 2.0]])
    q = _mapped(p, **{**RAMP, "u_axis": [0.0, 0.0]})
    # A zero direction gives u=0, so the clamped ramp evaluates at its midpoint.
    np.testing.assert_allclose(p[:, 2] - q[:, 2], [2.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Golden: a sheared octagon plate
# ---------------------------------------------------------------------------

# emit_sdm.py `_plate_octagon_verts` at the shipped values: pitch = stage_height 30.3,
# ring_len = cylinder_length 30, r_in = cylinder_inner_radius 22, big_u = reach + 10 = 33.5.
PITCH, RING_LEN, R_IN, BIG_U, HALF_T = 30.3, 30.0, 22.0, 33.5, 0.4


def _octagon_plate():
    verts = [
        [-BIG_U, 0.0], [-R_IN, 0.0], [R_IN, PITCH], [BIG_U, PITCH],
        [BIG_U, PITCH + RING_LEN], [R_IN, PITCH + RING_LEN], [-R_IN, RING_LEN], [-BIG_U, RING_LEN],
    ]  # fmt: skip
    poly = sdf_primitive("polygon_2d", vertices=verts)
    return sdf_transform("rotate_x", sdf_2d_to_3d("extrusion", poly, h=HALF_T), angle=-math.pi / 2)


def _sheared_band(dz1):
    band = [[-BIG_U, 0.0], [BIG_U, 0.0], [BIG_U, RING_LEN], [-BIG_U, RING_LEN]]
    poly = sdf_primitive("polygon_2d", vertices=band)
    flat = sdf_transform("rotate_x", sdf_2d_to_3d("extrusion", poly, h=HALF_T), angle=-math.pi / 2)
    return sdf_deform("shear_linear", flat, axis=[1.0, 0.0], u0=-R_IN, u1=R_IN, dz_0=0.0, dz_1=dz1)


def _grid() -> jnp.ndarray:
    xs = np.linspace(-36.0, 36.0, 73)
    ys = np.linspace(-1.2, 1.2, 7)
    zs = np.linspace(-3.0, 64.0, 68)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)


def test_a_sheared_flat_band_is_the_octagon_plate():
    part = Part(name="plate")
    p = _grid()
    ref = np.asarray(make_sdf_closure(_octagon_plate(), part)(p))
    got = np.asarray(make_sdf_closure(_sheared_band(PITCH), part)(p))
    # Same solid: identical inside/outside away from the surface. Distances may differ
    # inside the ramp (a shear is a bound, not an isometry), so only the sign is pinned.
    clear = np.abs(ref) > 0.05
    assert np.array_equal(np.sign(ref[clear]), np.sign(got[clear]))
    assert (ref < 0).sum() > 100  # the grid actually cuts the plate


def test_the_stage_height_can_ride_a_live_ref():
    part = Part(name="plate", params={"stage_height": Param("stage_height", 40.0, free=True)})
    tree = _sheared_band({"$ref": "stage_height"})
    far_end_mid_band = jnp.asarray([[30.0, 0.0, 40.0 + RING_LEN / 2]])
    assert float(make_sdf_closure(tree, part)(far_end_mid_band)[0]) < 0


# ---------------------------------------------------------------------------
# Bounds: bbox and Lipschitz, under a live (interval) rise
# ---------------------------------------------------------------------------


def _live_part() -> Part:
    return Part(
        name="live",
        params={"h": Param("h", 3.0, free=True, bounds=(1.0, 6.0))},
    )


def test_the_bbox_moves_only_z_and_covers_the_whole_live_range():
    node = sdf_deform(
        "shear_linear", sdf_primitive("box", b=[2.0, 0.5, 1.0]),
        u0=-1.0, u1=1.0, dz_0=0.0, dz_1={"$ref": "h"},
    )  # fmt: skip
    (xlo, ylo, zlo), (xhi, yhi, zhi) = infer_sdf_bbox(node, _live_part())
    assert (xlo, ylo, xhi, yhi) == (-2.0, -0.5, 2.0, 0.5)
    assert zlo == pytest.approx(-1.0)  # the low end never rises (dz_0 = 0)
    assert zhi == pytest.approx(1.0 + 6.0)  # the top of the h bounds, not its value


def test_the_lipschitz_bound_covers_the_measured_gradient():
    part = _live_part()
    node = sdf_deform(
        "shear_linear", sdf_primitive("sphere", r=1.0),
        u0=-1.0, u1=1.0, dz_0=0.0, dz_1={"$ref": "h"},
    )  # fmt: skip
    bound = infer_sdf_max_rate(node, part)
    k = 6.0 / 2.0  # steepest slope: the top of h over the ramp width
    assert bound == pytest.approx((k + math.sqrt(k * k + 4.0)) / 2.0)

    steep = Part(name="steep", params={"h": Param("h", 6.0, free=True, bounds=(1.0, 6.0))})
    f = make_sdf_closure(node, steep)
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.uniform([-1.0, -1.5, -2.0], [1.0, 1.5, 8.0], size=(4000, 3)))
    step = 1e-3
    grads = [
        (np.asarray(f(p + step * e)) - np.asarray(f(p - step * e))) / (2 * step) for e in jnp.eye(3)
    ]
    measured = float(np.max(np.linalg.norm(np.stack(grads, -1), axis=-1)))
    assert measured <= bound * 1.01
    assert measured > 1.5  # the bound is not vacuous: a shear really stretches


def test_a_ramp_with_no_width_has_no_bound():
    node = sdf_deform(
        "shear_linear", sdf_primitive("sphere", r=1.0), u0=1.0, u1=1.0, dz_0=0, dz_1=1
    )
    assert infer_sdf_max_rate(node, Part(name="t")) == UNKNOWN


# ---------------------------------------------------------------------------
# GLSL
# ---------------------------------------------------------------------------


def _glsl_body(fn_name: str) -> str:
    lib = re.sub(r"//[^\n]*", "", load_lib_glsl())
    m = re.search(rf"^\w+\s+{re.escape(fn_name)}\s*\([^)]*\)\s*\{{(.*?)\n\}}", lib, re.S | re.M)
    assert m, f"{fn_name} not found in lib.glsl"
    return m.group(1)


def _glsl_shear_linear(p, ax, u0, u1, dz0, dz1):
    ax = np.asarray(ax, dtype=np.float64)
    ax = ax / max(np.linalg.norm(ax), 1e-30)
    t = np.clip((p[:, :2] @ ax - u0) / (u1 - u0), 0.0, 1.0)
    return np.stack([p[:, 0], p[:, 1], p[:, 2] - (dz0 + (dz1 - dz0) * t)], axis=-1)


def test_the_transcription_below_is_still_what_lib_glsl_says():
    body = _glsl_body("op_shear_linear")
    assert "uAxis / max(length(uAxis), 1e-30)" in body
    assert "clamp((dot(p.xy, ax) - u0) / (u1 - u0), 0.0, 1.0)" in body
    assert "float rise = dz0 + (dz1 - dz0) * t;" in body
    assert "vec3(p.x, p.y, p.z - rise)" in body


def test_the_glsl_transcription_matches_jax():
    rng = np.random.default_rng(1)
    p = rng.uniform(-3.0, 3.0, size=(200, 3))
    got = _mapped(p, u_axis=[0.6, 0.8], u0=-1.5, u1=1.5, dz0=-0.4, dz1=2.2)
    np.testing.assert_allclose(
        _glsl_shear_linear(p, [0.6, 0.8], -1.5, 1.5, -0.4, 2.2), got, rtol=1e-6, atol=1e-6
    )
    zero_got = _mapped(p, u_axis=[0.0, 0.0], u0=-1.5, u1=1.5, dz0=-0.4, dz1=2.2)
    np.testing.assert_allclose(
        _glsl_shear_linear(p, [0.0, 0.0], -1.5, 1.5, -0.4, 2.2),
        zero_got,
        rtol=1e-6,
        atol=1e-6,
    )


def test_the_emitter_writes_the_warp_with_a_live_ref():
    part = _live_part()
    tree = sdf_deform(
        "shear_linear", sdf_primitive("box", b=[2.0, 0.5, 1.0]),
        u0=-1.0, u1=1.0, dz_0=0.0, dz_1={"$ref": "h"},
    )  # fmt: skip
    part.add_material(MaterialRegion(name="m", material_id="PA12", sdf_tree=tree))
    emission = emit_glsl(part)
    assert "op_shear_linear(" in emission.scene_source
    assert "vec3 op_shear_linear(" in emission.lib_source
    assert any(u.name.endswith("h") for u in emission.uniforms)  # the rise is a live uniform


def test_a_part_using_it_declares_the_schema_that_knows_it():
    """The frozen 0.3 schema enumerates deform names, so a part that needs 0.3 for its
    priors must still declare 0.6 once it uses shear_linear, and then validate."""
    from software_defined_matter.io import validate
    from software_defined_matter.model import min_schema_version_for

    part = Part(
        name="p",
        params={"h": Param("h", 3.0, free=True, bounds=(1.0, 6.0), unit="mm",
                           prior={"dist": "normal", "sigma": 0.1})},
    )  # fmt: skip
    part.add_material(
        MaterialRegion(name="m", material_id=1, sdf_tree=_sheared_band({"$ref": "h"}))
    )
    assert min_schema_version_for(part) == "0.6"
    assert part.to_dict()["kind"] == "part"
    validate(part)  # against the declared (computed) version
