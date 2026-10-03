"""Derive the per-family TPMS normalisation constant C = max |grad_q f|.

Each TPMS primitive returns ``abs(f) - thickness`` where ``f`` is a sum of
sines and cosines of ``q = p * 2*pi/period``. Nothing in that has units of
length, so the result is not a distance: it changes at rate
``(2*pi/period) * |grad_q f|`` per millimetre. Dividing by
``L = (2*pi/period) * C`` with ``C = max |grad_q f|`` makes the value a
conservative distance (rate <= 1 everywhere).

C must be an UPPER bound. A sampled maximum is a *lower* bound, so a constant
taken straight from sampling would leave a residual over-report and the audit
would fail intermittently. This script therefore does a dense grid search
followed by gradient ascent from the best candidates, and prints the result
alongside the closed form where one is known.

Run::

    uv run python docs/sdf_distance_audit/tpms_constants.py
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

TWO_PI = 2.0 * math.pi


# The pattern functions, in q-space (one period is 2*pi per axis). Kept
# separate from sdf_shapes so this derivation does not depend on the very code
# it is used to fix.
def f_gyroid(q):
    return (
        jnp.sin(q[..., 0]) * jnp.cos(q[..., 1])
        + jnp.sin(q[..., 1]) * jnp.cos(q[..., 2])
        + jnp.sin(q[..., 2]) * jnp.cos(q[..., 0])
    )


def f_schwarz_p(q):
    return jnp.cos(q[..., 0]) + jnp.cos(q[..., 1]) + jnp.cos(q[..., 2])


def f_schwarz_d(q):
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    return (
        jnp.sin(x) * jnp.sin(y) * jnp.sin(z)
        + jnp.sin(x) * jnp.cos(y) * jnp.cos(z)
        + jnp.cos(x) * jnp.sin(y) * jnp.cos(z)
        + jnp.cos(x) * jnp.cos(y) * jnp.sin(z)
    )


def f_neovius(q):
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    return 3.0 * (jnp.cos(x) + jnp.cos(y) + jnp.cos(z)) + 4.0 * jnp.cos(x) * jnp.cos(y) * jnp.cos(z)


def f_lidinoid(q):
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    return (
        0.5
        * (
            jnp.sin(2 * x) * jnp.cos(y) * jnp.sin(z)
            + jnp.sin(2 * y) * jnp.cos(z) * jnp.sin(x)
            + jnp.sin(2 * z) * jnp.cos(x) * jnp.sin(y)
        )
        - 0.5
        * (
            jnp.cos(2 * x) * jnp.cos(2 * y)
            + jnp.cos(2 * y) * jnp.cos(2 * z)
            + jnp.cos(2 * z) * jnp.cos(2 * x)
        )
        - 0.15
    )


FAMILIES = {
    "gyroid": (f_gyroid, math.sqrt(3.0), "sqrt(3), attained at the origin"),
    "schwarz_p": (f_schwarz_p, math.sqrt(3.0), "sqrt(3): |grad|^2 = sum sin^2 <= 3"),
    "schwarz_d": (f_schwarz_d, math.sqrt(3.0), "sqrt(3), attained at the origin"),
    "neovius": (f_neovius, 7.0, "7: df/dx = -sin(x)(3 + 4cos(y)cos(z))"),
    "lidinoid": (f_lidinoid, None, "no simple closed form"),
}

GRID = 48  # coarse search resolution per axis
N_SEEDS = 256  # best candidates to refine
N_STEPS = 600
STEP = 3e-3


def grad_norm_fn(f):
    g = jax.grad(lambda x: jnp.reshape(f(x[None, :]), ()))
    return jax.jit(jax.vmap(lambda x: jnp.linalg.norm(g(x))))


def max_grad_norm(f):
    """Dense grid search, then gradient ascent on |grad f| from the best seeds."""
    gn = grad_norm_fn(f)

    axis = jnp.linspace(0.0, TWO_PI, GRID, endpoint=False)
    q = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    vals = gn(q)
    coarse = float(vals.max())

    seeds = q[jnp.argsort(vals)[-N_SEEDS:]]
    ascend = jax.jit(
        jax.grad(lambda x: jnp.linalg.norm(jax.grad(lambda y: jnp.reshape(f(y[None, :]), ()))(x)))
    )

    pts = seeds
    for _ in range(N_STEPS):
        pts = pts + STEP * jax.vmap(ascend)(pts)
    refined = float(gn(pts).max())

    return coarse, max(coarse, refined)


def main() -> None:
    print(f"grid {GRID}^3, {N_SEEDS} seeds refined for {N_STEPS} ascent steps\n")
    print(f"{'family':<12}{'coarse':>10}{'refined':>10}{'closed form':>13}{'err':>10}   note")
    results = {}
    for name, (f, closed, note) in FAMILIES.items():
        coarse, refined = max_grad_norm(f)
        results[name] = refined
        cf = f"{closed:.6f}" if closed is not None else "-"
        err = f"{abs(refined - closed):.2e}" if closed is not None else "-"
        print(f"{name:<12}{coarse:>10.6f}{refined:>10.6f}{cf:>13}{err:>10}   {note}")

    print()
    print("Constants to use (closed form where known; otherwise the refined")
    print("maximum rounded UP, so the normalised field never over-reports):")
    print()
    for name, (_f, closed, _) in FAMILIES.items():
        if closed is not None:
            expr = "math.sqrt(3.0)" if abs(closed - math.sqrt(3.0)) < 1e-12 else f"{closed}"
            print(f'    "{name}": {expr},   # exact')
        else:
            safe = math.ceil(results[name] * 1e6) / 1e6
            print(f'    "{name}": {safe},   # numerical max {results[name]:.6f}, rounded up')


if __name__ == "__main__":
    main()
