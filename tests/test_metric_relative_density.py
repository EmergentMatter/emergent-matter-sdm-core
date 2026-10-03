"""``relative_density``: material over the part's own design volume.

The denominator is the part's *envelope*: the same solid with its holes filled
back in, derived by walking the SDF tree (see
:mod:`software_defined_matter.sdf.envelope`). A solid part therefore reads 1.0,
and a porous one reads the fraction of its outer shape that is material.

These tests pin the analytic cases, the gradient through both numerator and
denominator, the sampling box being sized to the envelope rather than the part,
and the serialization contract.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    Constraint,
    MaterialRegion,
    Param,
    Part,
    io,
    make_param_ref,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.dsl.expr import compile_expr, expr_metric
from software_defined_matter.objectives.metrics import list_metrics


def _part(tree, *, res=96, params=None, **md):
    return Part(
        name="rd",
        params=params or {},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
        metadata={"grid_resolution": res, **md},
    )


def _x(part):
    return jnp.asarray(part.param_vector(), dtype=jnp.float32)


def _density(part, **args):
    return float(compile_expr(expr_metric("relative_density", **args), part)(_x(part)))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_relative_density_is_registered():
    assert "relative_density" in list_metrics()


# ---------------------------------------------------------------------------
# Is a part with no holes reported as fully dense?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tree",
    [
        sdf_primitive("sphere", r=5.0),
        sdf_primitive("box", b=[4.0, 3.0, 2.0]),
        sdf_primitive("capped_cylinder", r=4.0, h=3.0),
    ],
)
def test_a_solid_part_is_one(tree):
    # The envelope of a solid is the solid, so numerator and denominator are
    # the same integral of the same field and cancel exactly.
    assert _density(_part(tree)) == pytest.approx(1.0, abs=1e-6)


def test_a_solid_part_has_zero_gradient():
    # Follows from reading exactly 1.0 everywhere, and makes the metric useless
    # as an objective for non-porous geometry. Intended.
    part = _part(
        sdf_primitive("sphere", r=make_param_ref("r")),
        params={"r": Param("r", 5.0, free=True, bounds=(1.0, 9.0))},
    )
    fn = compile_expr(expr_metric("relative_density"), part)
    g = jax.grad(lambda z: fn(z).reshape(()))(_x(part))
    assert float(g[0]) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Do porous parts match their closed-form densities?
# ---------------------------------------------------------------------------


def test_perforated_plate_matches_analytic():
    half, period, n_reps, r = 10.0, 4.0, 2, 1.0
    box = sdf_primitive("box", b=[half, half, half])
    cyl = sdf_primitive("capped_cylinder", r=r, h=half + 1.0)
    holes = sdf_transform("repeat_finite", cyl, c=period, l=[n_reps, n_reps, 0])
    n = (2 * n_reps + 1) ** 2
    v_box = (2 * half) ** 3
    expected = (v_box - n * math.pi * r**2 * (2 * half)) / v_box
    assert _density(_part(sdf_op("subtract", [box, holes]))) == pytest.approx(expected, rel=2e-3)


def test_hollow_shell_measures_its_wall_against_the_solid_ball():
    # `onion` is a porosity operator, so the walk drops it and the denominator
    # is the ball the shell was hollowed out of.
    r_out, wall = 5.0, 0.5
    tree = sdf_modifier("onion", sdf_primitive("sphere", r=r_out), thickness=wall)
    expected = (r_out**3 - (r_out - wall) ** 3) / r_out**3
    assert _density(_part(tree, res=128)) == pytest.approx(expected, rel=5e-3)


def test_lattice_clipped_to_a_sphere_reports_the_lattices_own_fill_fraction():
    # The case that motivates filling lattice leaves. A bbox denominator gives
    # 0.377 here, which is a property of the box rather than of the geometry.
    clipped = _density(
        _part(
            sdf_op(
                "intersect",
                [
                    sdf_primitive("gyroid", period=5.0, min_thickness=1.0, n_periods=[5, 5, 5]),
                    sdf_primitive("sphere", r=9.0),
                ],
            ),
            res=160,
        )
    )
    bare = _density(
        _part(sdf_primitive("gyroid", period=5.0, min_thickness=1.0, n_periods=[5, 5, 5]), res=160)
    )
    assert clipped == pytest.approx(bare, rel=0.02)
    assert clipped == pytest.approx(0.72, rel=0.05)


# ---------------------------------------------------------------------------
# Is the sampling box sized to the envelope rather than to the part?
# ---------------------------------------------------------------------------


def test_a_shaping_cut_does_not_truncate_the_denominator():
    """A sphere with its cap cut flat reaches z=4, but its envelope reaches z=5.

    The walk treats every subtraction as porosity, so the denominator is the
    whole ball. A sampling box sized to the part would stop at z=4 and lose the
    top of that ball, inflating the density. The expected value is the ball
    minus one spherical cap, over the ball.
    """
    r, z_cut = 5.0, 4.0
    tree = sdf_op(
        "subtract",
        [
            sdf_primitive("sphere", r=r),
            sdf_transform(
                "translate",
                sdf_primitive("capped_cylinder", r=2 * r, h=2.0),
                t=[0.0, 0.0, z_cut + 2.0],
            ),
        ],
    )
    h_cap = r - z_cut
    v_cap = math.pi * h_cap**2 * (3 * r - h_cap) / 3.0
    v_ball = 4.0 / 3.0 * math.pi * r**3
    assert _density(_part(tree, res=128)) == pytest.approx((v_ball - v_cap) / v_ball, rel=5e-3)


# ---------------------------------------------------------------------------
# Does an explicit envelope still override the walk?
# ---------------------------------------------------------------------------


def test_explicit_envelope_wins_over_the_walked_one():
    # Sphere r=5 measured against a box of half-extent 6 that the caller names.
    part = _part(sdf_primitive("sphere", r=5.0), bbox=[[-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]])
    env = sdf_primitive("box", b=[6.0, 6.0, 6.0])
    expected = (4.0 / 3.0 * math.pi * 5.0**3) / (12.0**3)
    assert _density(part, envelope=env) == pytest.approx(expected, rel=0.02)


def test_envelope_disjoint_from_the_part_is_zero_and_finite():
    part = _part(sdf_primitive("sphere", r=5.0), bbox=[[-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]])
    far = sdf_transform("translate", sdf_primitive("box", b=[1.0, 1.0, 1.0]), t=[100.0, 0.0, 0.0])
    v = _density(part, envelope=far)
    assert math.isfinite(v)
    assert v < 0.02


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_grad_matches_the_closed_form_through_a_shrinking_cavity():
    # density = 1 - (r_in/r_out)^3, so d(density)/d(r_in) = -3 r_in^2 / r_out^3.
    r_out, r_in = 5.0, 4.0
    part = _part(
        sdf_op(
            "subtract",
            [sdf_primitive("sphere", r=r_out), sdf_primitive("sphere", r=make_param_ref("r_in"))],
        ),
        res=128,
        params={"r_in": Param("r_in", r_in, free=True, bounds=(1.0, 4.8))},
    )
    fn = compile_expr(expr_metric("relative_density"), part)
    g = float(jax.grad(lambda z: fn(z).reshape(()))(_x(part))[0])
    assert g == pytest.approx(-3.0 * r_in**2 / r_out**3, rel=0.05)


def test_grad_matches_finite_difference():
    part = _part(
        sdf_op(
            "subtract",
            [sdf_primitive("sphere", r=5.0), sdf_primitive("sphere", r=make_param_ref("r_in"))],
        ),
        res=96,
        params={"r_in": Param("r_in", 4.0, free=True, bounds=(1.0, 4.8))},
    )
    fn = compile_expr(expr_metric("relative_density"), part)
    x = _x(part)
    g = float(jax.grad(lambda z: fn(z).reshape(()))(x)[0])
    step = 0.05
    fd = float((fn(x + step) - fn(x - step)) / (2.0 * step))
    assert g == pytest.approx(fd, rel=0.05, abs=1e-3)


# ---------------------------------------------------------------------------
# Edge cases and scale
# ---------------------------------------------------------------------------


def test_density_is_scale_invariant():
    # The same shell a hundred times bigger has the same relative density,
    # because the band is tied to the cell and the cell to the box.
    def shell(scale):
        return _density(
            _part(
                sdf_op(
                    "subtract",
                    [
                        sdf_primitive("sphere", r=5.0 * scale),
                        sdf_primitive("sphere", r=4.0 * scale),
                    ],
                ),
                res=96,
            )
        )

    assert shell(0.02) == pytest.approx(shell(100.0), rel=1e-3)


def test_denominator_is_floored_at_one_cell():
    # A part far smaller than a cell cannot be resolved; the floor keeps the
    # value finite instead of dividing by a denominator that rounds to zero.
    part = _part(
        sdf_primitive("sphere", r=1e-9),
        bbox=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        res=16,
        metric_skip_resolution_check=True,
    )
    v = _density(part)
    assert math.isfinite(v)


# ---------------------------------------------------------------------------
# jit and serialization
# ---------------------------------------------------------------------------


def test_jit_matches_eager():
    part = _part(
        sdf_op(
            "subtract",
            [sdf_primitive("sphere", r=5.0), sdf_primitive("sphere", r=make_param_ref("r_in"))],
        ),
        res=64,
        params={"r_in": Param("r_in", 4.0, free=True, bounds=(1.0, 4.8))},
    )
    fn = compile_expr(expr_metric("relative_density"), part)
    jfn = jax.jit(fn)
    x = _x(part)
    assert jnp.allclose(jfn(x), fn(x), atol=1e-5)
    # A second call with a different free_vec must not error or drift.
    assert jnp.allclose(jfn(x - 1.0), fn(x - 1.0), atol=1e-5)


def test_envelope_constraint_roundtrips_through_sdm(tmp_path):
    env = sdf_primitive("box", b=[6.0, 6.0, 6.0])
    part = _part(sdf_primitive("sphere", r=5.0), res=40, bbox=[[-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]])
    part.constraints.append(
        Constraint(
            name="max_infill",
            expr=expr_metric("relative_density", envelope=env),
            op="<=",
            rhs=0.30,
        )
    )

    path = tmp_path / "infill.sdm"
    io.save(part, path)
    io.validate(path)  # metric args are schema-free; the envelope is opaque JSON
    reloaded = io.load(path)

    x = _x(part)
    fn0 = compile_expr(part.constraints[0].expr, part)
    fn1 = compile_expr(reloaded.constraints[0].expr, reloaded)
    assert jnp.allclose(fn0(x), fn1(x), atol=1e-5)
