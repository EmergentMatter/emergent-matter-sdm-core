"""Edge treatments for booleans: how faces meet at the edges they make.

A boolean of solids is decided by the distances to their faces; HOW those
faces meet at an edge is a separate choice, given as ``blend`` (with size
``blend_r``) on ``union`` / ``intersect`` / ``subtract`` and on ``box`` /
``capped_cylinder``:

    sharp            a sharp edge (plain max / min; the default)
    round, r         a circular round of radius r, tangent to both faces
    chamfer, s       a 45-degree chamfer with legs of s on each face

``intersect`` treats convex edges (outside corners), ``union`` concave ones
(inside corners: a fillet), ``subtract`` the edges of the cut.

Exact where faces meet square: for two faces at 90 degrees the round is a
true circular arc of radius r, the chamfer a true 45-degree flat with legs s,
and the result stays a true distance near the edge (rate <= 1). At other
angles the round is still tangent to both faces but is not an arc of radius
r, and the chamfer legs are not s. Unlike ``smooth_union``'s ``k`` (a blend
width), ``blend_r`` is a dimension.

Every pair of inputs is treated: a box rounds all twelve edges and its
corners become spherical; a chamfered box gets a chamfer on every edge.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp

BLENDS = ("sharp", "round", "chamfer")
_R2 = 0.7071067811865476


def blend_intersect(D, blend="sharp", r=0.0):
    """Combine face distances ``D`` (..., N) as an intersection."""
    if blend == "sharp":
        return jnp.max(D, axis=-1)
    if blend == "round":
        out = jnp.sqrt(jnp.sum(jnp.maximum(D + r, 0.0) ** 2, axis=-1) + 1e-30)
        return jnp.minimum(-r, jnp.max(D, axis=-1)) + out
    if blend == "chamfer":
        pairs = [
            (D[..., i] + D[..., j] + r) * _R2
            for i, j in itertools.combinations(range(D.shape[-1]), 2)
        ]
        return jnp.maximum(jnp.max(D, axis=-1), jnp.max(jnp.stack(pairs, -1), axis=-1))
    raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}")


def blend_union(D, blend="sharp", r=0.0):
    """Combine solid distances ``D`` (..., N) as a union."""
    if blend == "sharp":
        return jnp.min(D, axis=-1)
    if blend == "round":
        inside = jnp.sqrt(jnp.sum(jnp.maximum(r - D, 0.0) ** 2, axis=-1) + 1e-30)
        return jnp.maximum(r, jnp.min(D, axis=-1)) - inside
    if blend == "chamfer":
        pairs = [
            (D[..., i] + D[..., j] - r) * _R2
            for i, j in itertools.combinations(range(D.shape[-1]), 2)
        ]
        return jnp.minimum(jnp.min(D, axis=-1), jnp.min(jnp.stack(pairs, -1), axis=-1))
    raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}")


def blend_subtract(D, blend="sharp", r=0.0):
    """``D[..., 0]`` with every later input cut out of it."""
    cut = jnp.concatenate([D[..., :1], -D[..., 1:]], axis=-1)
    return blend_intersect(cut, blend, r)
