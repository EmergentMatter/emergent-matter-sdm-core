"""An angle given as a ``[sin, cos]`` vector means its direction, at any length.

``capped_torus(sc)`` and ``solid_angle(c)`` take such a vector as a free
``vec2`` param, so an optimiser moves the two components independently and the
vector leaves the unit circle on its first step. The field must not care:
scaling the vector changes neither the surface nor the distance, in JAX or in
the GLSL preview. ``capped_torus`` must also stay finite on its own centreline,
where IQ's expanded ``sqrt`` cancels to a negative argument in float32.
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from software_defined_matter.glsl.emit import emit_glsl
from software_defined_matter.sdf import sdf_shapes
from software_defined_matter.sdf.compile import make_sdf_closure
from tests._glsl_runtime import ShaderRuntime
from tests.conftest import _sc

# Values are O(10) mm in float32 (JAX's default), whose spacing there is ~1e-6.
# Normalising first and normalising a scaled copy round differently, by a few
# ulps of the result.
FLOAT32_ATOL = 2e-5

# (kind, the direction param's name, the other params). Same angles as the
# conftest table, so the shapes are the ones the mesh and distance audits see.
CASES = [
    ("capped_torus", "sc", {"ra": 7.0, "rb": 2.2}, 120.0),
    ("solid_angle", "c", {"ra": 9.0}, 50.0),
]
IDS = [c[0] for c in CASES]

# Off the unit circle in both directions, plus a hand-typed near-miss.
SCALES = [0.5, 0.97, 2.0, 7.3]


def _points(n=20_000, half=12.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-half, half, size=(n, 3)).astype(np.float32)


@pytest.mark.parametrize("scale", SCALES, ids=[f"x{s}" for s in SCALES])
@pytest.mark.parametrize("kind,vec_name,kw,deg", CASES, ids=IDS)
def test_direction_param_length_does_not_change_the_field(kind, vec_name, kw, deg, scale):
    fn = getattr(sdf_shapes, kind)
    p = jnp.asarray(_points())
    unit = jnp.asarray(_sc(deg))
    expected = fn(p, **{vec_name: unit}, **kw)
    got = fn(p, **{vec_name: scale * unit}, **kw)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=FLOAT32_ATOL)


def test_capped_torus_is_finite_on_its_centreline():
    """The circle the arc lies on gives finite values and gradients everywhere,
    and depth exactly ``rb`` where it is the arc.

    The angle sweep crosses the cap at 120 deg, so both branches (the arc's end
    and the foot on the circle) are hit. Before the rewrite, hundreds of these
    values and every one of these gradients were NaN in float32.
    """
    ra, rb = 7.3, 2.2
    sc = jnp.asarray(_sc(120.0), dtype=jnp.float32)
    th = np.linspace(-3.0, 3.0, 20_001)  # +-172 deg, past the cap on both sides
    p = jnp.asarray(
        np.stack([ra * np.sin(th), ra * np.cos(th), np.zeros_like(th)], axis=-1),
        dtype=jnp.float32,
    )

    def scalar(x):
        return sdf_shapes.capped_torus(x[None], sc, ra, rb)[0]

    values = jax.vmap(scalar)(p)
    grads = jax.vmap(jax.grad(scalar))(p)
    assert bool(jnp.all(jnp.isfinite(grads)))
    # Inside the cap the centreline is the arc, at depth exactly rb; beyond it
    # the nearest centreline point is the arc's end, so the value only rises.
    inside_cap = np.abs(th) <= math.radians(120.0)
    np.testing.assert_allclose(np.asarray(values)[inside_cap], -rb, atol=FLOAT32_ATOL)
    assert bool(jnp.all(values >= -rb - FLOAT32_ATOL))


@pytest.mark.parametrize("kind,vec_name,kw,deg", CASES, ids=IDS)
def test_zero_direction_vector_gives_finite_values_and_gradients(kind, vec_name, kw, deg):
    """A zero vector has no direction, so the shape is degenerate, but it must
    not put NaN into metrics or gradients."""
    fn = getattr(sdf_shapes, kind)
    p = jnp.asarray(_points(n=2_000))

    def total(v):
        return jnp.sum(fn(p, **{vec_name: v}, **kw))

    zero = jnp.zeros(2)
    assert bool(jnp.all(jnp.isfinite(fn(p, **{vec_name: zero}, **kw))))
    assert bool(jnp.all(jnp.isfinite(jax.grad(total)(zero))))


@pytest.fixture(scope="module")
def shader_runtime():
    try:
        runtime = ShaderRuntime()
    except (OSError, AttributeError, RuntimeError) as exc:
        if os.environ.get("SDM_REQUIRE_GLSL") == "1":
            pytest.fail(f"Required software shader runtime unavailable: {exc}")
        pytest.skip(
            f"Install system EGL/GLES and Mesa software drivers to run shader parity: {exc}"
        )
    yield runtime
    runtime.close()


@pytest.mark.parametrize("kind,vec_name,kw,deg", CASES, ids=IDS)
def test_emitted_shader_matches_jax_for_a_non_unit_direction(
    shader_runtime, kind, vec_name, kw, deg
):
    """The preview must draw what gets manufactured, at any vector length."""
    vec = [2.0 * x for x in _sc(deg)]
    tree = sdf_primitive(kind, **{vec_name: vec}, **kw)
    part = Part(
        name="direction parity",
        params={},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
        metadata={"bbox": [[-12, -12, -12], [12, 12, 12]]},
    )
    pts = _points(n=4_000)
    glsl = shader_runtime.evaluate(emit_glsl(part), pts)[:, 3]
    jax_values = make_sdf_closure(tree, part)(jnp.asarray(pts), jnp.zeros((0,)))
    np.testing.assert_allclose(glsl, np.asarray(jax_values), atol=FLOAT32_ATOL)
