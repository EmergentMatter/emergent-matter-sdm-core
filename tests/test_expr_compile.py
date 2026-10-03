"""Expression DSL: evaluation, metric resolution, and ``jax.grad``."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    make_param_ref,
    sdf_primitive,
)
from software_defined_matter.dsl.expr import (
    compile_expr,
    eval_expr_pure,
    expr_binop,
    expr_metric,
    expr_num,
    expr_param,
    expr_unop,
)
from software_defined_matter.dsl.resolve import make_binding


def _simple_part(radius=3.0, free=True):
    envelope = sdf_primitive("sphere", r=make_param_ref("r"))
    return Part(
        name="sphere",
        params={"r": Param("r", radius, free=free, bounds=(0.1, 20.0), unit="mm")},
        materials=[
            MaterialRegion(material_id=1, name="X", sdf_tree=envelope),
        ],
        metadata={
            "bbox_half_size": 6.0,
            "grid_resolution": 32,
        },
    )


def test_eval_expr_pure_arithmetic():
    part = _simple_part()
    binding = make_binding(part)
    expr = expr_binop("+", expr_param("r"), expr_num(5.0))
    val = eval_expr_pure(expr, binding, jnp.array([3.0]))
    assert jnp.allclose(val, jnp.asarray(8.0), atol=1e-6)


def test_eval_expr_pure_rejects_metric_leaf():
    part = _simple_part()
    binding = make_binding(part)
    expr = expr_metric("volume")
    with pytest.raises(ValueError):
        eval_expr_pure(expr, binding, jnp.array([3.0]))


def test_compile_expr_volume_is_finite_and_differentiable():
    part = _simple_part()
    expr = expr_metric("volume")
    fn = compile_expr(expr, part)
    free_vec = jnp.asarray(part.param_vector(), dtype=jnp.float32)
    val = fn(free_vec)
    assert jnp.isfinite(val)
    assert val > 0.0

    grad = jax.grad(fn)(free_vec)
    assert jnp.isfinite(grad).all()
    # Volume should increase with radius, so dV/dr > 0.
    assert grad[0] > 0.0


def test_compile_expr_mass_uses_density_arg():
    part = _simple_part()
    fn_vol = compile_expr(expr_metric("volume"), part)
    fn_mass = compile_expr(expr_metric("mass", density=1000.0), part)
    free_vec = jnp.asarray(part.param_vector(), dtype=jnp.float32)
    assert jnp.allclose(fn_mass(free_vec), fn_vol(free_vec) * 1000.0, rtol=1e-5)


def test_compile_expr_mass_without_density_raises():
    part = _simple_part()
    with pytest.raises(ValueError, match="density"):
        compile_expr(expr_metric("mass"), part)(jnp.asarray(part.param_vector(), dtype=jnp.float32))


def test_compile_expr_symbolic_constraint_wall_thickness(example_part):
    """For the canonical example, ``outer_radius - inner_radius`` is evaluated
    against the free-param vector without ever touching the SDF."""
    wall = next(c for c in example_part.constraints if c.name == "wall_thickness")
    fn = compile_expr(wall.expr, example_part)
    free_vec = jnp.asarray(example_part.param_vector(), dtype=jnp.float32)
    # outer_radius (10) - inner_radius (7) == 3.0
    # Note order depends on dict insertion; use names to compute ground truth:
    names = example_part.free_param_names()
    outer = free_vec[names.index("outer_radius")]
    inner = free_vec[names.index("inner_radius")]
    assert jnp.allclose(fn(free_vec), outer - inner, atol=1e-6)

    grad = jax.grad(fn)(free_vec)
    # Gradient should be +1 on outer_radius, -1 on inner_radius.
    assert jnp.allclose(grad[names.index("outer_radius")], 1.0)
    assert jnp.allclose(grad[names.index("inner_radius")], -1.0)


def _gyroid_clipped_part(with_bbox=False):
    """Sphere clipped by a gyroid infill. The intersect makes the result
    analytically bounded (sphere clamps the gyroid via the AABB-intersection
    rule).

    These tests are about which sampling box gets inferred, not about how
    accurate the integral is, so the resolution check is turned off: resolving
    a 0.294 mm wall across a 24 mm box needs 326 cells per axis, past the
    default cap, and the volume itself is only asserted to be finite.
    """
    from software_defined_matter import sdf_op

    shell = sdf_primitive("sphere", r=make_param_ref("r"))
    infill = sdf_primitive("gyroid", period=4.0, min_thickness=0.294042, n_periods=[6, 6, 6])
    md = {"grid_resolution": 16, "metric_skip_resolution_check": True}
    if with_bbox:
        md["bbox"] = [[-12.0, -12.0, -12.0], [12.0, 12.0, 12.0]]
    return Part(
        name="gyroid",
        params={"r": Param("r", 10.0, free=True, bounds=(1.0, 11.0), unit="mm")},
        materials=[
            MaterialRegion(material_id=1, name="X", sdf_tree=sdf_op("intersect", [shell, infill]))
        ],
        metadata=md,
    )


def _raw_unbounded_part():
    """Analytically-unbounded geometry: a `rotate_z` whose angle is a
    parameter reference. The bbox inferrer's ``_require_constant`` rejects
    ``$ref`` angles, so this exercises the fail-loud path for metric
    sampling on a part whose sampling domain cannot be inferred.
    """
    return Part(
        name="param_rotated",
        params={
            "r": Param("r", 10.0, free=True, bounds=(1.0, 11.0), unit="mm"),
            "theta": Param("theta", 0.5, free=False, unit="rad"),
        },
        materials=[
            MaterialRegion(
                material_id=1,
                name="X",
                sdf_tree=sdf_primitive("sphere", r=expr_unop("exp", expr_param("theta"))),
            )
        ],
        metadata={"grid_resolution": 16},
    )


def test_unbounded_geometry_metric_fails_loud():
    """A metric on truly unbounded geometry without an explicit bbox must
    raise a clear error, not silently fabricate a domain."""
    fn = compile_expr(expr_metric("volume"), _raw_unbounded_part())
    with pytest.raises(ValueError, match=r"metadata\['bbox'\]"):
        fn(jnp.asarray([10.0], dtype=jnp.float32))


def test_pure_expr_on_unbounded_geometry_still_compiles():
    """No metric node => no sampling box needed => must not raise even when
    the geometry is unbounded (bbox resolution is lazy)."""
    expr = expr_binop("+", expr_param("r"), expr_num(1.0))
    fn = compile_expr(expr, _raw_unbounded_part())
    assert jnp.allclose(fn(jnp.asarray([10.0], dtype=jnp.float32)), 11.0, atol=1e-6)


def test_explicit_bbox_unblocks_unbounded_metric():
    fn = compile_expr(expr_metric("volume"), _gyroid_clipped_part(with_bbox=True))
    val = fn(jnp.asarray([10.0], dtype=jnp.float32))
    assert jnp.isfinite(val) and val > 0.0


def test_intersect_clipped_metric_resolves_via_bounded_sibling():
    """`intersect(sphere, gyroid)` is now analytically bounded by the sphere,
    so a metric compiles and evaluates without explicit `bbox` metadata."""
    fn = compile_expr(expr_metric("volume"), _gyroid_clipped_part(with_bbox=False))
    val = fn(jnp.asarray([10.0], dtype=jnp.float32))
    assert jnp.isfinite(val) and val > 0.0


def test_voxel_size_gives_per_axis_resolution():
    """Cells come out cubic on a 4 x 4 x 20 box, whichever way the grid is asked for.

    ``count`` now sizes the *longest* axis and the others follow at the same
    cell size, so a 4 x 4 x 20 box asked for 32 gets (8, 8, 33) rather than
    (32, 32, 32). Cubic cells are what let the occupancy ramp have a single
    width ``h``; a per-axis count would make cells 5x longer in Z here.
    """
    from software_defined_matter.objectives.metrics import (
        cubic_grid,
        grid_resolution_for,
    )

    bbox = ((-2.0, -2.0, -10.0), (2.0, 2.0, 10.0))
    assert grid_resolution_for(bbox, voxel_size=1.0) == (5, 5, 21)
    assert grid_resolution_for(bbox, count=32) == (8, 8, 33)

    for kwargs in ({"voxel_size": 1.0}, {"count": 32}, {"voxel_size": 0.001, "cap": 64}):
        grown, res, h = cubic_grid(bbox, **kwargs)
        (x0, y0, z0), (x1, y1, z1) = grown
        for extent, n in zip((x1 - x0, y1 - y0, z1 - z0), res, strict=False):
            assert extent / n == pytest.approx(h, rel=1e-9)

    # The cap raises h uniformly rather than clamping one axis and skewing cells.
    assert max(grid_resolution_for(bbox, voxel_size=0.001, cap=64)) == 64


def test_compile_expr_example_volume_grad_is_finite(example_part):
    part = dataclasses.replace(
        example_part,
        metadata={
            k: v
            for k, v in example_part.metadata.items()
            if k not in ("metric_voxel_size", "metric_grid_cap")
        }
        | {"grid_resolution": 48, "metric_skip_resolution_check": True},
    )
    max_vol = next(c for c in part.constraints if c.name == "max_volume")
    fn = compile_expr(max_vol.expr, part)
    free_vec = jnp.asarray(part.param_vector(), dtype=jnp.float32)
    val = fn(free_vec)
    assert jnp.isfinite(val)
    grad = jax.grad(fn)(free_vec)
    assert jnp.isfinite(grad).all()
