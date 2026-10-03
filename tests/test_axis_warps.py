"""scale_axis and taper_linear reproduce independent primitives, and their bounds hold live.

Each is pinned against a primitive that exists on its own: a sphere scaled per-axis IS the
``ellipsoid`` primitive, equal factors ARE the uniform ``scale`` transform, and a cylinder
tapered from 1 to 0.5 IS a ``capped_cone``. Also: a non-rigid wrapper that can thin a wall
(including ``shear_linear``) now says so to ``sdf/features.py``, which used to assume every
deform preserves thickness.
"""

from __future__ import annotations

import math
import re

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import Param, Part, sdf_deform, sdf_primitive, sdf_transform
from software_defined_matter.glsl.emit import emit_glsl, load_lib_glsl
from software_defined_matter.model import MaterialRegion
from software_defined_matter.sdf import sdf_ops as ops
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.features import infer_min_feature_size
from software_defined_matter.sdf.lipschitz import UNKNOWN, infer_sdf_max_rate


def _grid(lo: float, hi: float, n: int = 25) -> jnp.ndarray:
    xs = np.linspace(lo, hi, n)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    return jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)


def _same_solid(a, b, part: Part, p: jnp.ndarray, clear: float = 0.05) -> None:
    da = np.asarray(make_sdf_closure(a, part)(p))
    db = np.asarray(make_sdf_closure(b, part)(p))
    keep = np.abs(da) > clear
    assert np.array_equal(np.sign(da[keep]), np.sign(db[keep]))
    assert (da < 0).sum() > 50


# ---------------------------------------------------------------------------
# scale_axis
# ---------------------------------------------------------------------------


def test_a_sphere_scaled_per_axis_is_the_ellipsoid_primitive():
    scaled = sdf_transform("scale_axis", sdf_primitive("sphere", r=1.0), s=[2.0, 1.0, 0.5])
    _same_solid(
        scaled, sdf_primitive("ellipsoid", r=[2.0, 1.0, 0.5]), Part(name="t"), _grid(-2.5, 2.5)
    )


def test_equal_factors_are_exactly_the_uniform_scale():
    child = sdf_primitive("box", b=[1.0, 0.4, 0.7])
    p = _grid(-3.0, 3.0, 13)
    part = Part(name="t")
    a = make_sdf_closure(sdf_transform("scale_axis", child, s=[1.7, 1.7, 1.7]), part)(p)
    b = make_sdf_closure(sdf_transform("scale", child, s=1.7), part)(p)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def test_scale_axis_bbox_and_rate_hold_across_a_live_factor():
    part = Part(name="live", params={"sx": Param("sx", 2.0, free=True, bounds=(1.0, 3.0))})
    node = sdf_transform("scale_axis", sdf_primitive("sphere", r=1.0), s=[{"$ref": "sx"}, 1.0, 1.0])
    (xlo, ylo, zlo), (xhi, yhi, zhi) = infer_sdf_bbox(node, part)
    assert (xlo, xhi) == (-3.0, 3.0)  # the top of the sx bounds
    assert (ylo, yhi, zlo, zhi) == (-1.0, 1.0, -1.0, 1.0)
    assert infer_sdf_max_rate(node, part) == pytest.approx(1.0)  # min(s)/s_i <= 1


def test_a_factor_that_can_reach_zero_has_no_bound():
    part = Part(name="t", params={"sx": Param("sx", 1.0, free=True, bounds=(0.0, 2.0))})
    node = sdf_transform("scale_axis", sdf_primitive("sphere", r=1.0), s=[{"$ref": "sx"}, 1.0, 1.0])
    assert infer_sdf_max_rate(node, part) == UNKNOWN


def test_scale_axis_works_in_2d():
    from software_defined_matter.model import sdf_2d_to_3d

    circle = sdf_primitive("circle_2d", r=1.0)

    tree = sdf_2d_to_3d("extrusion", sdf_transform("scale_axis", circle, s=[2.0, 1.0]), h=0.5)
    f = make_sdf_closure(tree, Part(name="t"))
    inside = jnp.asarray([[1.8, 0.0, 0.0]])  # within x-radius 2, outside the unscaled circle
    outside = jnp.asarray([[0.0, 1.2, 0.0]])
    assert float(f(inside)[0]) < 0 < float(f(outside)[0])


# ---------------------------------------------------------------------------
# taper_linear
# ---------------------------------------------------------------------------


def _taper(s1=0.5, **kw):
    cyl = sdf_primitive("capped_cylinder", h=1.0, r=1.0)
    return sdf_deform("taper_linear", cyl, z0=-1.0, z1=1.0, s_0=1.0, s_1=s1, **kw)


def test_a_tapered_cylinder_is_the_capped_cone_primitive():
    cone = sdf_primitive("capped_cone", h=1.0, r1=1.0, r2=0.5)
    _same_solid(_taper(), cone, Part(name="t"), _grid(-1.5, 1.5, 31))


def _mapped_taper(p: np.ndarray, **kw) -> np.ndarray:
    seen = {}

    def spy(q):
        seen["q"] = q
        return jnp.zeros(q.shape[:-1])

    ops.op_taper_linear(spy, jnp.asarray(p), **kw)
    return np.asarray(seen["q"], dtype=np.float64)


def test_the_factor_holds_past_each_end_and_leaves_z_alone():
    p = np.array([[1.0, 1.0, -5.0], [1.0, 1.0, 0.0], [1.0, 1.0, 5.0]])
    q = _mapped_taper(p, z0=-1.0, z1=1.0, s0=2.0, s1=4.0)
    np.testing.assert_allclose(q[:, 0], [0.5, 1.0 / 3.0, 0.25], atol=1e-9)  # x / s(z)
    np.testing.assert_allclose(q[:, 2], p[:, 2], atol=0)


def test_taper_bbox_grows_xy_by_the_largest_factor_only():
    part = Part(name="live", params={"s": Param("s", 1.5, free=True, bounds=(0.5, 2.5))})
    (xlo, ylo, zlo), (xhi, yhi, zhi) = infer_sdf_bbox(_taper(s1={"$ref": "s"}), part)
    assert (xlo, xhi, ylo, yhi) == (-2.5, 2.5, -2.5, 2.5)
    assert (zlo, zhi) == (-1.0, 1.0)


def test_taper_rate_needs_a_domain_and_then_covers_the_measured_gradient():
    part = Part(name="t")
    node = _taper(s1=0.5)
    assert infer_sdf_max_rate(node, part) == UNKNOWN
    domain = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
    bound = infer_sdf_max_rate(node, part, domain=domain)
    assert math.isfinite(bound)
    f = make_sdf_closure(node, part)
    rng = np.random.default_rng(2)
    p = jnp.asarray(rng.uniform(-1.4, 1.4, size=(4000, 3)))
    step = 1e-3
    grads = [
        (np.asarray(f(p + step * e)) - np.asarray(f(p - step * e))) / (2 * step) for e in jnp.eye(3)
    ]
    assert float(np.max(np.linalg.norm(np.stack(grads, -1), axis=-1))) <= bound * 1.01


# ---------------------------------------------------------------------------
# Wall thinning (features.py)
# ---------------------------------------------------------------------------


def test_non_rigid_wrappers_report_the_walls_they_can_thin():
    part = Part(name="t")
    slab = sdf_primitive("box", b=[2.0, 2.0, 0.5])  # thinnest feature: 1.0 (full z)
    base = infer_min_feature_size(slab, part)
    assert infer_min_feature_size(
        sdf_transform("scale_axis", slab, s=[1.0, 1.0, 0.5]), part
    ) == pytest.approx(base * 0.5)
    tapered = sdf_deform("taper_linear", slab, z0=-1.0, z1=1.0, s_0=1.0, s_1=0.25)
    assert 0 < infer_min_feature_size(tapered, part) < base * 0.25
    k = 2.0  # rise 4 over width 2
    sheared = sdf_deform("shear_linear", slab, u0=-1.0, u1=1.0, dz_0=0.0, dz_1=4.0)
    assert infer_min_feature_size(sheared, part) == pytest.approx(
        base * 2.0 / (k + math.sqrt(k * k + 4))
    )
    rigid = sdf_transform("rotate_z", slab, angle=0.3)
    assert infer_min_feature_size(rigid, part) == pytest.approx(base)


def test_taper_feature_bound_covers_off_axis_wall_distance_across_live_scales():
    part = Part("wall", params={"end": Param("end", 2.0, bounds=(1.5, 3.0))})
    slab = sdf_transform("translate", sdf_primitive("box", b=[0.5, 2.0, 2.0]), t=[10, 0, 0])
    tree = sdf_deform("taper_linear", slab, z0=0.0, z1=1.0, s_0=1.0, s_1={"$ref": "end"})
    bound = infer_min_feature_size(tree, part, mode="bounds")
    assert bound is not None and bound > 0.0
    for end in (1.5, 2.0, 3.0):
        # The outer face is the line x=10.5*(1+(end-1)*z).
        # Measure from the inner face at z=0.5 along the outer face's normal.
        slope = end - 1.0
        x_inner = 9.5 * (1.0 + slope * 0.5)
        distance = (10.5 * (1.0 + slope * 0.5) - x_inner) / math.hypot(1, 10.5 * slope)
        assert bound <= distance
        assert distance < 0.4  # The old bound of 1 mm misses this thinning.


def test_constant_taper_keeps_the_diagonal_scale_bound():
    tree = sdf_deform(
        "taper_linear", sdf_primitive("box", b=[1, 1, 0.5]), z0=0.0, z1=1.0, s_0=0.5, s_1=0.5
    )
    assert infer_min_feature_size(tree, Part("constant")) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    "params",
    [
        {"z0": 1.0, "z1": 1.0, "s_0": 1.0, "s_1": 2.0},
        {"z0": 0.0, "z1": 1.0, "s_0": 0.0, "s_1": 2.0},
    ],
    ids=["zero-width", "zero-scale"],
)
def test_degenerate_taper_does_not_claim_a_positive_feature_size(params):
    tree = sdf_deform("taper_linear", sdf_primitive("box", b=[1, 1, 0.5]), **params)
    assert infer_min_feature_size(tree, Part("degenerate")) == 0.0


def test_taper_without_a_bounded_child_does_not_claim_a_positive_feature_size():
    child = sdf_primitive("gyroid", scale=1.0, min_thickness=0.5)
    tree = sdf_deform("taper_linear", child, z0=0.0, z1=1.0, s_0=1.0, s_1=2.0)
    assert infer_min_feature_size(tree, Part("unbounded")) == 0.0


# ---------------------------------------------------------------------------
# GLSL
# ---------------------------------------------------------------------------


def _glsl_body(fn_name: str) -> str:
    lib = re.sub(r"//[^\n]*", "", load_lib_glsl())
    m = re.search(rf"^\w+\s+{re.escape(fn_name)}\s*\([^)]*\)\s*\{{(.*?)\n\}}", lib, re.S | re.M)
    assert m, f"{fn_name} not found in lib.glsl"
    return m.group(1)


def test_the_taper_transcription_below_is_still_what_lib_glsl_says():
    body = _glsl_body("op_taper_linear")
    assert "clamp((p.z - z0) / (z1 - z0), 0.0, 1.0)" in body
    assert "float s = s0 + (s1 - s0) * t;" in body
    assert "vec3(p.x / s, p.y / s, p.z)" in body


def test_the_taper_glsl_transcription_matches_jax():
    rng = np.random.default_rng(3)
    p = rng.uniform(-2.0, 2.0, size=(200, 3))
    t = np.clip((p[:, 2] + 1.0) / 2.0, 0.0, 1.0)
    s = 0.8 + (1.6 - 0.8) * t
    expect = np.stack([p[:, 0] / s, p[:, 1] / s, p[:, 2]], -1)
    np.testing.assert_allclose(
        _mapped_taper(p, z0=-1.0, z1=1.0, s0=0.8, s1=1.6), expect, rtol=1e-6, atol=1e-6
    )


def test_the_emitter_writes_both_with_live_refs():
    part = Part(
        name="live",
        params={
            "sx": Param("sx", 2.0, free=True, bounds=(1.0, 3.0)),
            "s": Param("s", 0.5, free=True, bounds=(0.25, 1.0)),
        },
    )
    tree = sdf_transform("scale_axis", _taper(s1={"$ref": "s"}), s=[{"$ref": "sx"}, 1.0, 1.0])
    part.add_material(MaterialRegion(name="m", material_id="PA12", sdf_tree=tree))
    emission = emit_glsl(part)
    assert "op_taper_linear(" in emission.scene_source
    assert "q = p / sa;" in emission.scene_source
    assert "vec3 op_taper_linear(" in emission.lib_source


@pytest.mark.parametrize(
    "tree",
    [
        sdf_transform("scale_axis", sdf_primitive("sphere", r=1.0), s=[2.0, 1.0, 1.0]),
        sdf_deform(
            "taper_linear", sdf_primitive("sphere", r=1.0), z0=-1.0, z1=1.0, s_0=1.0, s_1=0.5
        ),
    ],
    ids=["scale_axis", "taper_linear"],
)
def test_a_part_using_one_declares_the_schema_that_knows_it(tree):
    from software_defined_matter.io import validate
    from software_defined_matter.model import min_schema_version_for

    part = Part(name="p", params={"k": Param("k", 1.0, free=True, unit="mm")})
    part.add_material(MaterialRegion(name="m", material_id=1, sdf_tree=tree))
    assert min_schema_version_for(part) == "0.6"
    assert part.to_dict()["kind"] == "part"
    validate(part)
