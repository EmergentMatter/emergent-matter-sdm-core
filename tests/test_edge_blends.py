"""Edge treatments (``blend`` / ``blend_r``) on booleans, ``box`` and
``capped_cylinder``: real radii and chamfer sizes, and nothing added where
children share a face (the rolling ball of sdf_edge_blends)."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.sdf import sdf_edge_blends as eb
from software_defined_matter.sdf import sdf_shapes as sh
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.validate import (
    SemanticValidationError,
    validate_document_semantics,
)
from tests._glsl_runtime import ShaderRuntime


def _part(tree, **metadata):
    return Part(
        name="t",
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
        metadata=metadata,
    )


def _closure(tree, **metadata):
    return jax.jit(make_sdf_closure(tree, _part(tree, **metadata)))


def _box(center, half):
    return sdf_transform("translate", sdf_primitive("box", b=list(half)), t=list(center))


def _top(f, x, y=0.0, z0=-1.0, z1=20.0):
    """Height of the top surface above (x, y), by bisection on the field."""
    lo, hi = z0, z1
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if float(f(jnp.array([[x, y, m]]))[0]) < 0:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


# L-bracket: plate x 0..40, z 0..6; wall x 0..6, z 0..40 (Alvaro's finding 4)
_PLATE = _box([20.0, 0.0, 3.0], [20.0, 10.0, 3.0])
_WALL = _box([3.0, 0.0, 20.0], [3.0, 10.0, 20.0])


def test_round_union_is_a_true_r3_fillet():
    f = _closure(sdf_op("union", [_PLATE, _WALL], blend="round", blend_r=3.0))
    th = np.linspace(np.pi, 1.5 * np.pi, 40)
    arc = np.c_[9 + 3 * np.cos(th), np.zeros(40), 9 + 3 * np.sin(th)]  # centre (9, 9)
    assert np.abs(np.asarray(f(jnp.asarray(arc)))).max() < 1e-4


def test_chamfer_union_legs_are_r():
    f = _closure(sdf_op("union", [_PLATE, _WALL], blend="chamfer", blend_r=3.0))
    t = np.linspace(0.0, 1.0, 11)
    flat = np.c_[6 + 3 * t, np.zeros(11), 9 - 3 * t]  # (6, 9) to (9, 6)
    assert np.abs(np.asarray(f(jnp.asarray(flat)))).max() < 1e-4


@pytest.mark.parametrize("blend", ["round", "chamfer"])
def test_flush_blocks_get_no_bulge(blend):
    """Two blocks sharing their top and bottom faces (finding 1): the union
    is the plain union, with no material added across the shared faces."""
    a = _box([-12.5, 0.0, -5.0], [17.5, 10.0, 5.0])  # x -30..5
    b = _box([12.5, 0.0, -5.0], [17.5, 10.0, 5.0])  # x -5..30
    f = _closure(sdf_op("union", [a, b], blend=blend, blend_r=3.0))
    for x in (-8.0, -3.0, 0.0, 3.0, 8.0):
        assert abs(_top(f, x, z0=-5.0, z1=8.0)) < 1e-4


def test_bracket_with_flush_tab_keeps_its_faces():
    """Alvaro's bracket: plate, end wall and a tab overlapping the plate's
    other end flush top and bottom. The fillet is at the wall, and the
    overlap stays 6 mm thick."""
    plate = _box([30.0, 0.0, 3.0], [30.0, 20.0, 3.0])
    wall = _box([3.0, 0.0, 23.0], [3.0, 20.0, 23.0])
    tab = _box([60.0, 0.0, 3.0], [20.0, 10.0, 3.0])
    f = _closure(sdf_op("union", [plate, wall, tab], blend="round", blend_r=3.0))
    assert abs(_top(f, 50.0, z0=0.0, z1=10.0) - 6.0) < 1e-4
    assert _top(f, 7.0, z0=0.0, z1=12.0) > 6.5  # on the R3 fillet: 9 - sqrt(5)


def test_round_union_is_a_distance_near_the_surface():
    """Rate 1 (finding 2): the field grows as fast as the distance, no faster."""
    tree = sdf_op("union", [_PLATE, _WALL], blend="round", blend_r=3.0)
    f = make_sdf_closure(tree, _part(tree))
    pts = jnp.array([[20.0, 0.0, 7.0], [20.0, 0.0, 8.5], [8.0, 0.0, 8.0], [7.0, 0.0, 10.0]])
    g = jax.vmap(jax.grad(lambda p: f(p[None])[0]))(pts)
    assert np.allclose(np.linalg.norm(np.asarray(g), axis=1), 1.0, atol=1e-3)


def test_round_union_stays_inside_the_children_bbox():
    """Finding 3: nothing outside the children's hull, so the inferred box
    holds the whole part."""
    f = _closure(sdf_op("union", [_PLATE, _WALL], blend="round", blend_r=3.0))
    rng = np.random.default_rng(0)
    outside = rng.uniform([-10, -15, -10], [50, 15, 50], (4000, 3))
    keep = (
        (outside[:, 0] < 0)
        | (outside[:, 0] > 40)
        | (np.abs(outside[:, 1]) > 10)
        | (outside[:, 2] < 0)
        | (outside[:, 2] > 40)
    )
    assert np.all(np.asarray(f(jnp.asarray(outside[keep]))) > 0)


def test_intersect_rounds_the_edge_it_makes():
    """An R2 round where a box is cut by a slab; the box's own edges elsewhere
    stay sharp."""
    a = sdf_primitive("box", b=[10.0, 10.0, 10.0])
    slab = _box([0.0, 0.0, -5.0], [20.0, 20.0, 10.0])  # top at z = 5
    f = _closure(sdf_op("intersect", [a, slab], blend="round", blend_r=2.0))
    th = np.linspace(0.0, np.pi / 2, 20)
    arc = np.c_[8 + 2 * np.cos(th), np.zeros(20), 3 + 2 * np.sin(th)]
    assert np.abs(np.asarray(f(jnp.asarray(arc)))).max() < 1e-4
    assert abs(float(f(jnp.array([[10.0, 10.0, -10.0]]))[0])) < 1e-5


def test_subtract_chamfers_the_bore_rim():
    plate = sdf_primitive("box", b=[10.0, 10.0, 2.0])
    bore = sdf_primitive("capped_cylinder", h=5.0, r=3.0)
    f = _closure(sdf_op("subtract", [plate, bore], blend="chamfer", blend_r=1.0))
    t = np.linspace(0.0, 1.0, 11)
    flat = np.c_[3 + t, np.zeros(11), 1 + t]  # (3, 1) to (4, 2)
    assert np.abs(np.asarray(f(jnp.asarray(flat)))).max() < 1e-4


@pytest.mark.parametrize("op", ["union", "intersect", "subtract"])
@pytest.mark.parametrize("blend", ["round", "chamfer"])
def test_one_child_is_the_child(op, blend):
    """Finding 7: one child has no edge to treat, and does not crash."""
    f = _closure(sdf_op(op, [_PLATE], blend=blend, blend_r=1.0))
    g = _closure(_PLATE)
    q = jnp.asarray(np.random.default_rng(1).uniform(-5, 45, (200, 3)), jnp.float32)
    assert np.allclose(f(q), g(q), atol=1e-6)


def test_many_children_cost_grows_linearly():
    """Finding 8: only the two nearest children form a point's edge, so a
    strut lattice compiles and the shader has one term per child."""
    struts = [_box([4.0 * i, 0.0, 0.0], [0.5, 10.0, 0.5]) for i in range(40)]
    tree = sdf_op("union", struts, blend="round", blend_r=0.3)
    f = _closure(tree)
    assert np.isfinite(np.asarray(f(jnp.zeros((8, 3))))).all()
    assert emit_glsl(_part(tree), tree).scene_source.count("if (k == ") == 39


def test_rounded_box_is_round_box():
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.uniform(-8, 8, (500, 3)), jnp.float32)
    ours = sh.box(q, jnp.array([5.0, 3.0, 2.0]), blend="round", blend_r=1.0)
    assert np.allclose(ours, sh.round_box(q, jnp.array([5.0, 3.0, 2.0]), 1.0), atol=1e-5)


def test_cylinder_chamfer():
    q = jnp.array([[3.0 - 0.5, 0.0, 5.0 - 0.5]])  # the middle of a 1 mm chamfer
    d = sh.capped_cylinder(q, 5.0, 3.0, blend="chamfer", blend_r=1.0)
    assert abs(float(d[0])) < 1e-6


def test_single_face_chamfer_does_not_crash():
    assert np.allclose(eb.op_blend_faces(jnp.array([[1.0], [-2.0]]), "chamfer", 1.0), [1.0, -2.0])


def test_sharp_and_absent_blend_are_unchanged():
    q = jnp.asarray(np.random.default_rng(1).uniform(-8, 40, (300, 3)), jnp.float32)
    plain = _closure(sdf_op("union", [_PLATE, _WALL]))(q)
    sharp = _closure(sdf_op("union", [_PLATE, _WALL], blend="sharp", blend_r=2.0))(q)
    assert np.allclose(plain, sharp)


def test_explicit_sharp_wins_over_smooth_csg():
    """Finding 6: one boolean can stay sharp in a smooth part."""
    sharp = sdf_op("union", [_PLATE, _WALL], blend="sharp")
    f = _closure(sharp, smooth_csg=True)
    assert abs(_top(f, 6.2, z0=0.0, z1=12.0) - 6.0) < 1e-4
    promoted = _closure(sdf_op("union", [_PLATE, _WALL]), smooth_csg=True)
    assert _top(promoted, 6.2, z0=0.0, z1=12.0) > 6.0 + 1e-3


def test_blend_radius_can_be_a_param():
    from software_defined_matter import Param, make_param_ref

    tree = sdf_op("union", [_PLATE, _WALL], blend="round", blend_r=make_param_ref("fillet"))
    part = Part(
        name="t",
        params={"fillet": Param(name="fillet", value=2.0, unit="mm")},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
    )
    f = jax.jit(make_sdf_closure(tree, part))
    g = _closure(sdf_op("union", [_PLATE, _WALL], blend="round", blend_r=2.0))
    q = jnp.array([[7.0, 0.0, 7.0]])
    assert np.allclose(f(q), g(q), atol=1e-6)


@pytest.mark.parametrize(
    "params",
    [{"blend": "round"}, {"blend": "chamfer", "blend_r": 0.0}, {"blend_r": -1.0}],
)
def test_blend_r_is_validated(params):
    with pytest.raises(SemanticValidationError, match="blend_r"):
        validate_document_semantics(_part(sdf_op("union", [_PLATE, _WALL], **params)).to_dict())


def test_part_declares_schema_0_7():
    tree = sdf_primitive("box", b=[1.0, 1.0, 1.0], blend="round", blend_r=0.2)
    assert _part(tree).to_dict()["schema_version"] == "0.7"


def test_blended_root_union_is_not_split_into_components():
    """Finding 4: the viewer must draw the fillet, so a blended root union
    (or a blended union one level down) stays one component."""
    tree = sdf_op("union", [_PLATE, _WALL], blend="round", blend_r=3.0)
    em = emit_glsl(_part(tree), tree)
    assert len(em.components) == 0
    assert "float R = " in em.scene_source
    nested = sdf_op("union", [tree, _box([30.0, 0.0, 20.0], [1.0, 1.0, 1.0])])
    assert len(emit_glsl(_part(nested), nested).components) == 2


def test_rigid_materials_do_not_split_a_blended_union():
    """Finding 5: the fillet belongs to neither body."""
    from software_defined_matter.rigid_materials import _union_members

    tree = sdf_op("union", [_PLATE, _WALL], blend="round", blend_r=3.0)
    assert _union_members(tree) == [tree]
    assert len(_union_members(sdf_op("union", [_PLATE, _WALL]))) == 2


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
    with pytest.raises(ValueError, match="raster_field"):
        emit_glsl(_part(tree), tree)


# ------------------------------------------------- executed shader parity
@pytest.fixture(scope="module")
def shader_runtime():
    try:
        runtime = ShaderRuntime()
    except (OSError, AttributeError, RuntimeError) as exc:
        if os.environ.get("SDM_REQUIRE_GLSL") == "1":
            pytest.fail(f"Required software shader runtime unavailable: {exc}")
        pytest.skip(f"Install system EGL/GLES and Mesa software drivers: {exc}")
    yield runtime
    runtime.close()


@pytest.mark.parametrize("op", ["union", "intersect", "subtract"])
@pytest.mark.parametrize("blend", ["round", "chamfer"])
def test_emitted_blend_matches_the_compiled_blend(shader_runtime, op, blend):
    """The viewer draws what export meshes: the emitted rolling ball, run,
    against the compiled one near the treated edges."""
    a = sdf_primitive("box", b=[10.0, 10.0, 3.0], blend="chamfer", blend_r=0.5)
    b = _box([-6.0, 0.0, 6.0], [3.0, 10.0, 6.0])
    tree = sdf_op(op, [a, b], blend=blend, blend_r=1.5)
    rng = np.random.default_rng(3)
    pts = rng.uniform([-12, -12, -6], [12, 12, 14], (2000, 3)).astype(np.float32)
    want = np.asarray(_closure(tree)(jnp.asarray(pts)))
    near = np.abs(want) < 3.0
    got = shader_runtime.evaluate(emit_glsl(_part(tree), tree), pts[near])[:, 3]
    np.testing.assert_allclose(got, want[near], atol=2e-3)
