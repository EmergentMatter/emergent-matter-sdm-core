"""Sampling / forward-propagation primitives over ``Param`` priors.

This is the evaluation half of the schema-0.3 probabilistic layer: with
priors on params, a ``.sdm`` is a *generative model* over geometry, and the
pipeline being JAX end-to-end makes forward uncertainty propagation nearly
free: draw a Monte Carlo cloud of free-parameter vectors and ``vmap`` the
already-compiled closure over it.

Deliberately dumb and composable, per the org's core-describes /
physics-judges cut: this module SAMPLES priors and PROPAGATES them through a
closure, full stop. No inference, no metrics, no pass/fail judgement -- this
is an integration point for a separate physics/verification layer, not part
of this repo, whose chance-constrained referees (P(interference) < eps) would
consume these primitives; MCMC/SVI calibration is likewise out of scope here.

Column order in every sampled matrix is ``part.free_param_names()``: the
same order :class:`software_defined_matter.dsl.resolve.ParamBinding` gives
the free vector, so sampled rows feed compiled closures directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from software_defined_matter.process import effective_prior

if TYPE_CHECKING:  # pragma: no cover
    from software_defined_matter.model import Part


def _sample_dist(
    prior: dict[str, Any],
    value: float,
    key: jax.Array,
    n: int,
) -> jnp.ndarray:
    """Draw ``n`` samples from one normalised prior spec, centred on ``value``."""
    dist = prior["dist"]
    if dist == "delta":
        return jnp.full((n,), value)
    if dist == "normal":
        return value + float(prior["sigma"]) * jax.random.normal(key, (n,))
    if dist == "uniform":
        return jax.random.uniform(key, (n,), minval=float(prior["lo"]), maxval=float(prior["hi"]))
    if dist == "uniform_pm":
        hw = float(prior["half_width"])
        return jax.random.uniform(key, (n,), minval=value - hw, maxval=value + hw)
    raise ValueError(
        f"Unknown prior dist {dist!r}: Param validation should have rejected this spec."
    )


def sample_free_params(
    part: Part,
    n: int,
    key: jax.Array,
    profile: dict[str, Any] | None = None,
) -> jnp.ndarray:
    """Draw ``n`` free-parameter vectors from the part's effective priors.

    Args:
        part: The part supplying free params and their priors.
        n: Number of Monte Carlo draws.
        key: A ``jax.random`` PRNG key. Split internally, one child per free
            param, so per-param draws are independent and the whole matrix
            is reproducible from one key.
        profile: A resolved process profile
            (:func:`software_defined_matter.process.resolve_process_profile`).
            Feeds :func:`software_defined_matter.process.effective_prior` --
            authored priors always win; the profile only fills in free
            ``"mm"`` params the author left delta.

    Returns:
        ``jnp.ndarray``, shape ``(n, n_free)``: one row per draw, columns in
        ``part.free_param_names()`` order -- the exact free-vector layout
        compiled closures expect. A delta prior yields a constant column at
        the authored value, so a part with no priors degenerates to ``n``
        copies of ``part.param_vector()``.
    """
    if n < 1:
        raise ValueError(f"sample_free_params requires n >= 1, got {n}.")
    free = list(part.free_params().values())
    if not free:
        return jnp.zeros((n, 0))
    keys = jax.random.split(key, len(free))
    cols = [
        _sample_dist(effective_prior(p, profile), float(p.numeric_value()), k, n)
        for p, k in zip(free, keys, strict=False)
    ]
    return jnp.stack(cols, axis=1)


def propagate(
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    part: Part,
    points: jnp.ndarray,
    *,
    n: int = 256,
    key: jax.Array | None = None,
    profile: dict[str, Any] | None = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict[str, Any]:
    """Push param priors through a compiled closure at fixed query points.

    ``vmap``-Monte-Carlo: draws ``n`` free vectors with
    :func:`sample_free_params` and evaluates
    ``fn(points, free_vec)``, the ``(points, free_vec) -> values`` interface
    of :func:`software_defined_matter.sdf.compile.make_sdf_closure` (any
    callable with that signature works, e.g. a metric closure), once per
    draw via ``jax.vmap``.

    Args:
        fn: ``(points, free_vec) -> (m,) array``.
        part: Supplies the free params and their priors.
        points: Fixed query points, shape ``(m, 3)`` (a single ``(3,)`` point
            is promoted).
        n: Monte Carlo draws (default 256 -- interactive; raise it for
            referee numbers, the cost is one ``vmap`` axis).
        key: PRNG key; ``None`` means ``jax.random.PRNGKey(0)`` --
            deterministic by default, pass your own key for fresh draws.
        profile: Resolved process profile, forwarded to
            :func:`sample_free_params`.
        quantiles: Which per-point quantiles to report (each in ``[0, 1]``).

    Returns:
        ``{"mean": (m,), "std": (m,),
        "quantiles": {q: (m,) for q in quantiles}}``: per-point statistics
        over the ``n`` draws. ``std`` is the population standard deviation;
        for a single param with a ``normal(sigma)`` prior entering the field
        linearly (e.g. a sphere radius), ``std`` at the surface approaches
        ``sigma`` as ``n`` grows.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    points = jnp.atleast_2d(jnp.asarray(points))
    samples = sample_free_params(part, n, key, profile)
    values = jax.vmap(lambda fv: fn(points, fv))(samples)  # (n, m)
    return {
        "mean": jnp.mean(values, axis=0),
        "std": jnp.std(values, axis=0),
        "quantiles": {float(q): jnp.quantile(values, q, axis=0) for q in quantiles},
    }


__all__ = ["propagate", "sample_free_params"]
