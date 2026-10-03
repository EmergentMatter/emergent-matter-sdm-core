"""Regression tests for the 3-D ``octahedron`` primitive (IQ exact formula).

The original port dropped IQ's signed-core early-out (``else m*0.57735027``),
which left the *entire* field non-negative: the interior never went below zero,
so the solid never meshed (marching cubes found no iso-crossing). These tests
pin the field's sign so that regression cannot return silently.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from software_defined_matter.sdf import sdf_shapes as shapes
from software_defined_matter.sdf.compile import make_sdf_closure

# Field is the regular octahedron |x| + |y| + |z| = s.
S = 6.0


def test_interior_is_negative():
    """The core regression: the centre is the deepest interior point.

    For ``|x|+|y|+|z| = s`` the origin is at distance ``-s/sqrt(3)``. The bug
    returned ``+s/sqrt(3)`` here, so nothing inside the hull ever meshed.
    """
    d0 = shapes.octahedron(jnp.zeros((1, 3)), S)[0]
    assert d0 < 0.0
    assert jnp.isclose(d0, -S / jnp.sqrt(3.0), atol=1e-5)


def test_axis_vertices_lie_on_surface():
    """All six ``±s`` axis vertices are on the surface and mutually identical."""
    verts = jnp.array(
        [
            [S, 0.0, 0.0],
            [-S, 0.0, 0.0],
            [0.0, S, 0.0],
            [0.0, -S, 0.0],
            [0.0, 0.0, S],
            [0.0, 0.0, -S],
        ]
    )
    d = shapes.octahedron(verts, S)
    assert jnp.allclose(d, 0.0, atol=1e-5)
    # Symmetry guards the per-branch coordinate swizzles, not just the early-out.
    assert float(d.max() - d.min()) < 1e-5


def test_exterior_is_positive():
    far = jnp.array([[S, S, S], [2 * S, 0.0, 0.0]])
    assert jnp.all(shapes.octahedron(far, S) > 0.0)


def test_is_unit_gradient_field():
    """A true SDF satisfies the eikonal equation away from creases."""
    g = jax.grad(lambda q: shapes.octahedron(q[None], S)[0])(jnp.array([2.0, 0.5, 0.3]))
    assert jnp.isclose(jnp.linalg.norm(g), 1.0, atol=1e-4)


def test_compiled_primitive_matches_direct_call():
    part = Part(
        name="octa",
        params={},
        materials=[
            MaterialRegion(
                material_id=1,
                name="mat",
                sdf_tree=sdf_primitive("octahedron", s=S),
            )
        ],
    )
    sdf = make_sdf_closure(part.computed_envelope(), part)

    p = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],
            [S, 0.0, 0.0],
            [S, S, S],
        ]
    )
    compiled = sdf(p, jnp.zeros((0,)))
    direct = shapes.octahedron(p, S)
    assert jnp.allclose(compiled, direct, atol=1e-6)
