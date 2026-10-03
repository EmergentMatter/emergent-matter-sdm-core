"""Compiled SDF must match the hand-written composition of primitives."""

from __future__ import annotations

import jax.numpy as jnp

from software_defined_matter import (
    MaterialRegion,
    Part,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.sdf import sdf_ops as ops
from software_defined_matter.sdf import sdf_shapes as shapes
from software_defined_matter.sdf import transforms
from software_defined_matter.sdf.compile import make_sdf_closure


def _points():
    # A handful of points spread around the unit volume.
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [-1.0, -1.0, 0.5],
            [3.0, 3.0, 3.0],
        ]
    )


def test_sphere_matches_direct_call():
    part = Part(
        name="s",
        params={},
        materials=[
            MaterialRegion(material_id=1, name="mat", sdf_tree=sdf_primitive("sphere", r=2.0))
        ],
    )
    sdf = make_sdf_closure(part.computed_envelope(), part)

    p = _points()
    direct = shapes.sphere(p, 2.0)
    compiled = sdf(p, jnp.zeros((0,)))
    assert jnp.allclose(compiled, direct, atol=1e-6)


def test_hollow_cylinder_matches_direct_call():
    tree = sdf_op(
        "subtract",
        [
            sdf_primitive("capped_cylinder", h=15.0, r=10.0),
            sdf_primitive("capped_cylinder", h=15.0, r=7.0),
        ],
    )
    part = Part(name="x")
    sdf = make_sdf_closure(tree, part)
    p = _points()
    direct = ops.op_subtract(
        shapes.capped_cylinder(p, 15.0, 10.0),
        shapes.capped_cylinder(p, 15.0, 7.0),
    )
    assert jnp.allclose(sdf(p, jnp.zeros((0,))), direct, atol=1e-5)


def test_translate_and_union_match_direct_call():
    hinge = sdf_primitive("notch_hinge", width=4.0, depth=4.0, notch_radius=1.5)
    hinge_placed = sdf_transform("translate", hinge, t=[0.0, 0.0, 5.0])
    tree = sdf_op(
        "union",
        [
            sdf_primitive("sphere", r=1.0),
            hinge_placed,
        ],
    )
    part = Part(name="x")
    sdf = make_sdf_closure(tree, part)

    p = _points()
    direct = jnp.minimum(
        shapes.sphere(p, 1.0),
        shapes.notch_hinge(transforms.tf_translate(p, jnp.asarray([0.0, 0.0, 5.0])), 4.0, 4.0, 1.5),
    )
    assert jnp.allclose(sdf(p, jnp.zeros((0,))), direct, atol=1e-5)


def test_make_sdf_closure_smooth_csg_toggle_changes_hard_boolean():
    tree = sdf_op(
        "union",
        [
            sdf_primitive("sphere", r=1.0),
            sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[1.0, 0.0, 0.0]),
        ],
    )
    part = Part(name="smooth-toggle")
    p = jnp.array([[0.5, 0.0, 0.0]], dtype=jnp.float32)

    hard_sdf = make_sdf_closure(tree, part, b_smooth_csg=False)
    smooth_sdf = make_sdf_closure(tree, part, b_smooth_csg=True, d_smooth_k=0.2)

    hard = hard_sdf(p, jnp.zeros((0,), dtype=jnp.float32))
    smooth = smooth_sdf(p, jnp.zeros((0,), dtype=jnp.float32))
    expected_hard = jnp.minimum(
        shapes.sphere(p, 1.0),
        shapes.sphere(transforms.tf_translate(p, jnp.array([1.0, 0.0, 0.0])), 1.0),
    )
    assert jnp.allclose(hard, expected_hard, atol=1e-6)
    assert not jnp.allclose(smooth, hard, atol=1e-6)


def test_example_part_compiles(example_part):
    import jax

    sdf = make_sdf_closure(example_part.computed_envelope(), example_part)
    free_vec = jnp.asarray(example_part.param_vector(), dtype=jnp.float32)
    p = _points()
    d = sdf(p, free_vec)
    assert d.shape == (p.shape[0],)
    assert jnp.all(jnp.isfinite(d))

    # jit compile on the free-vec path
    jit_sdf = jax.jit(sdf)
    d2 = jit_sdf(p, free_vec)
    assert jnp.allclose(d, d2, atol=1e-5)
