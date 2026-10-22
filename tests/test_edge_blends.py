"""Edge treatments (``blend`` / ``blend_r``) on booleans, ``box`` and
``capped_cylinder``: real radii and chamfer sizes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.sdf import sdf_shapes as sh
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.sdf_edge_blends import blend_intersect, blend_union


def _closure(tree):
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    return jax.jit(make_sdf_closure(tree, part))


def _square_corner(fn, blend, r):
    def f(p):
        return fn(jnp.stack([p[..., 0], p[..., 2]], axis=-1), blend, r)

    return f


def test_round_intersect_is_a_true_arc():
    f = _square_corner(blend_intersect, "round", 3.0)
    th = np.linspace(0.0, np.pi / 2, 50)
    pts = np.c_[-3 + 3 * np.cos(th), np.zeros(50), -3 + 3 * np.sin(th)]
    assert np.abs(np.asarray(f(jnp.asarray(pts)))).max() < 1e-5


def test_round_intersect_is_a_true_distance_outside():
    f = _square_corner(blend_intersect, "round", 3.0)
    q = jnp.array([[2.0, 0.0, 2.0], [5.0, 0.0, -10.0], [-10.0, 0.0, 4.0]])
    assert np.allclose(f(q), [np.hypot(5, 5) - 3, 5.0, 4.0], atol=1e-5)


def test_round_union_is_a_true_fillet():
    f = _square_corner(blend_union, "round", 3.0)
    th = np.linspace(np.pi, 1.5 * np.pi, 50)
    pts = np.c_[3 + 3 * np.cos(th), np.zeros(50), 3 + 3 * np.sin(th)]
    assert np.abs(np.asarray(f(jnp.asarray(pts)))).max() < 1e-5


def test_chamfer_legs_and_angle():
    f = _square_corner(blend_intersect, "chamfer", 1.0)
    t = np.linspace(0, 1, 11)
    pts = np.c_[-(1 - t), np.zeros(11), -t]
    assert np.abs(np.asarray(f(jnp.asarray(pts)))).max() < 1e-6


def test_rounded_box_is_round_box():
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.uniform(-8, 8, (500, 3)), jnp.float32)
    ours = sh.box(q, jnp.array([5.0, 3.0, 2.0]), blend="round", blend_r=1.0)
    assert np.allclose(ours, sh.round_box(q, jnp.array([5.0, 3.0, 2.0]), 1.0), atol=1e-5)


def test_sharp_and_absent_blend_are_unchanged():
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.uniform(-8, 8, (300, 3)), jnp.float32)
    a = sdf_primitive("box", b=[5.0, 3.0, 2.0])
    b = sdf_transform("translate", sdf_primitive("sphere", r=3.0), t=[4.0, 0.0, 0.0])
    plain = _closure(sdf_op("union", [a, b]))(q)
    sharp = _closure(sdf_op("union", [a, b], blend="sharp", blend_r=2.0))(q)
    assert np.allclose(plain, sharp)


@pytest.mark.parametrize("op", ["union", "intersect", "subtract"])
@pytest.mark.parametrize("blend", ["round", "chamfer"])
def test_compiled_ops_match_the_kernel(op, blend):
    a = sdf_primitive("box", b=[5.0, 5.0, 1.0])
    b = sdf_transform("translate", sdf_primitive("box", b=[1.0, 5.0, 5.0]), t=[-4.0, 0.0, 4.0])
    f = _closure(sdf_op(op, [a, b], blend=blend, blend_r=1.5))
    q = jnp.asarray(np.random.default_rng(2).uniform(-7, 7, (400, 3)), jnp.float32)
    da = sh.box(q, jnp.array([5.0, 5.0, 1.0]))
    db = sh.box(q - jnp.array([-4.0, 0.0, 4.0]), jnp.array([1.0, 5.0, 5.0]))
    from software_defined_matter.sdf import sdf_edge_blends as eb

    fn = {"union": eb.blend_union, "intersect": eb.blend_intersect, "subtract": eb.blend_subtract}
    assert np.allclose(f(q), fn[op](jnp.stack([da, db], -1), blend, 1.5), atol=1e-5)


def test_blend_radius_can_be_a_param():
    from software_defined_matter import Param, make_param_ref

    a = sdf_primitive("box", b=[5.0, 5.0, 1.0])
    b = sdf_transform("translate", sdf_primitive("box", b=[1.0, 5.0, 5.0]), t=[-4.0, 0.0, 4.0])
    tree = sdf_op("union", [a, b], blend="round", blend_r=make_param_ref("fillet"))
    part = Part(
        name="t",
        params={"fillet": Param(name="fillet", value=2.0, unit="mm")},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
    )
    f = jax.jit(make_sdf_closure(tree, part))
    g = _closure(sdf_op("union", [a, b], blend="round", blend_r=2.0))
    q = jnp.array([[-2.0, 0.0, 2.0]])
    assert np.allclose(f(q), g(q), atol=1e-6)


def test_cylinder_chamfer():
    q = jnp.array([[3.0 - 0.5, 0.0, 5.0 - 0.5]])  # the middle of a 1 mm chamfer
    d = sh.capped_cylinder(q, 5.0, 3.0, blend="chamfer", blend_r=1.0)
    assert abs(float(d[0])) < 1e-6


def test_part_declares_schema_0_6():
    tree = sdf_primitive("box", b=[1.0, 1.0, 1.0], blend="round", blend_r=0.2)
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    assert part.to_dict()["schema_version"] == "0.6"


def test_glsl_emits_blends():
    a = sdf_primitive("box", b=[5.0, 5.0, 1.0], blend="chamfer", blend_r=0.5)
    b = sdf_primitive("capped_cylinder", h=4.0, r=1.0, blend="round", blend_r=0.3)
    tree = sdf_op("union", [a, b], blend="round", blend_r=1.0)
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    src = emit_glsl(part, tree).scene_source
    assert "sqrt(" in src and "0.7071067811865476" in src


def test_glsl_refuses_boundary_blend():
    from software_defined_matter import sdf_boundary_blend

    sq = [
        [[0, 0, 0], [10, 0, 0]],
        [[10, 0, 0], [10, 10, 0]],
        [[10, 10, 0], [0, 10, 0]],
        [[0, 10, 0], [0, 0, 0]],
    ]
    skin = sdf_boundary_blend(sq, reach=10.0)
    tree = sdf_op("subtract", [sdf_primitive("box", b=[5.0, 5.0, 5.0]), skin])
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    with pytest.raises(ValueError, match="raster_field"):
        emit_glsl(part, tree)
