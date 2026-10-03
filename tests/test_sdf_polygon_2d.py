"""Tests for the 2-D ``polygon_2d`` primitive (Inigo Quilez exact formula)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from software_defined_matter.sdf import sdf_shapes as shapes
from software_defined_matter.sdf.compile import make_sdf_closure

# ---------------------------------------------------------------------------
# Canonical test polygons
# ---------------------------------------------------------------------------

# Unit equilateral triangle, centred near (0, 0).
TRIANGLE_VERTICES = jnp.array(
    [
        [-1.0, -0.5773502691896257],  # bottom-left  (apex angle 60°)
        [1.0, -0.5773502691896257],  # bottom-right
        [0.0, 1.1547005383792515],  # top
    ]
)

# Non-convex L-shape (Quilez sign test must handle concave regions).
# A 2x2 square with the top-right 1x1 corner cut out.
L_SHAPE_VERTICES = jnp.array(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [-1.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Direct-call correctness
# ---------------------------------------------------------------------------


def test_polygon_2d_distance_at_vertex_is_zero():
    for v_idx in range(TRIANGLE_VERTICES.shape[0]):
        p = TRIANGLE_VERTICES[v_idx]
        d = shapes.polygon_2d(p, TRIANGLE_VERTICES)
        assert jnp.isclose(d, 0.0, atol=1e-5), (
            f"Distance at vertex {v_idx} {p} should be 0, got {float(d)}"
        )


def test_polygon_2d_centroid_is_inside_negative():
    centroid = jnp.mean(TRIANGLE_VERTICES, axis=0)
    d = shapes.polygon_2d(centroid, TRIANGLE_VERTICES)
    assert d < 0.0, f"Centroid {centroid} should be inside (d<0), got {float(d)}"


def test_polygon_2d_far_point_is_outside_positive():
    p = jnp.array([5.0, 5.0])
    d = shapes.polygon_2d(p, TRIANGLE_VERTICES)
    assert d > 0.0, f"Far point {p} should be outside (d>0), got {float(d)}"


def test_polygon_2d_edge_midpoint_is_zero():
    # Midpoint of the bottom edge of the triangle.
    p = (TRIANGLE_VERTICES[0] + TRIANGLE_VERTICES[1]) / 2.0
    d = shapes.polygon_2d(p, TRIANGLE_VERTICES)
    assert jnp.isclose(d, 0.0, atol=1e-5), (
        f"Edge midpoint {p} should be at distance 0, got {float(d)}"
    )


def test_polygon_2d_nonconvex_lshape_inside_outside():
    # Inside the L's main body: well inside the bottom-left arm.
    p_inside = jnp.array([-0.5, -0.5])
    d_inside = shapes.polygon_2d(p_inside, L_SHAPE_VERTICES)
    assert d_inside < 0.0, f"L-shape inside point should be d<0, got {float(d_inside)}"

    # In the cut-out region (top-right quadrant): should be OUTSIDE.
    p_in_cut = jnp.array([0.5, 0.5])
    d_in_cut = shapes.polygon_2d(p_in_cut, L_SHAPE_VERTICES)
    assert d_in_cut > 0.0, f"L-shape cut-out point should be d>0 (outside), got {float(d_in_cut)}"

    # Distance from the cut-out point to the nearest edge is 0.5 (vertical
    # edge at x=0 from y=0 to y=1 is the nearest edge to (0.5, 0.5)).
    assert jnp.isclose(d_in_cut, 0.5, atol=1e-5), (
        f"L-shape cut-out (0.5, 0.5) → 0.5 distance to nearest edge, got {float(d_in_cut)}"
    )


def test_polygon_2d_batched_query():
    # The function vectorises over the leading axes of p.
    pts = jnp.array(
        [
            [0.0, 0.0],  # inside the triangle
            [5.0, 5.0],  # outside, far
            [1.0, -0.5773502691896257],  # at a vertex (distance 0)
        ]
    )
    d = shapes.polygon_2d(pts, TRIANGLE_VERTICES)
    assert d.shape == (3,)
    assert d[0] < 0.0
    assert d[1] > 0.0
    assert jnp.isclose(d[2], 0.0, atol=1e-5)


def test_polygon_2d_winding_invariant():
    # Reversed winding should give the same SDF (sign test handles both).
    forward = TRIANGLE_VERTICES
    reverse = TRIANGLE_VERTICES[::-1]
    p = jnp.array([0.0, 0.0])  # inside
    d_fwd = shapes.polygon_2d(p, forward)
    d_rev = shapes.polygon_2d(p, reverse)
    assert jnp.isclose(d_fwd, d_rev, atol=1e-6), (
        f"Distance must be winding-invariant: forward={float(d_fwd)}, reverse={float(d_rev)}"
    )


def test_polygon_2d_rejects_degenerate_inputs():
    # Fewer than 3 vertices.
    with pytest.raises(ValueError, match="N >= 3"):
        shapes.polygon_2d(jnp.array([0.0, 0.0]), jnp.array([[0.0, 0.0], [1.0, 0.0]]))

    # Wrong inner dimension (3D points by mistake).
    with pytest.raises(ValueError, match=r"N >= 3|got shape"):
        shapes.polygon_2d(
            jnp.array([0.0, 0.0]),
            jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )


# ---------------------------------------------------------------------------
# DSL round-trip: JSON-style spec → compiled JAX closure
# ---------------------------------------------------------------------------


def test_polygon_2d_dsl_compiles_inside_extrusion():
    # Build a Part with a 3D extrusion of a 2D polygon, the canonical
    # consumer path for 2D primitives in sdm-core.
    triangle_spec = sdf_primitive(
        "polygon_2d",
        vertices=[
            [-1.0, -0.5773502691896257],
            [1.0, -0.5773502691896257],
            [0.0, 1.1547005383792515],
        ],
    )
    extruded = {
        "type": "2d_to_3d",
        "method": "extrusion",
        "child": triangle_spec,
        "params": {"h": 2.0},
    }
    part = Part(
        name="prism",
        params={},
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=extruded)],
    )
    sdf = make_sdf_closure(part.computed_envelope(), part)

    # Inside the prism: at the triangle centroid, mid-height.
    p_in = jnp.array([[0.0, 0.0, 0.0]])
    d_in = sdf(p_in, jnp.zeros((0,)))
    assert d_in[0] < 0.0, f"Inside prism: expected d<0, got {float(d_in[0])}"

    # Outside the prism: above the top cap.
    p_out = jnp.array([[0.0, 0.0, 5.0]])
    d_out = sdf(p_out, jnp.zeros((0,)))
    assert d_out[0] > 0.0, f"Above prism cap: expected d>0, got {float(d_out[0])}"


def test_polygon_2d_compiled_matches_direct():
    triangle_spec = sdf_primitive(
        "polygon_2d",
        vertices=[
            [-1.0, -0.5773502691896257],
            [1.0, -0.5773502691896257],
            [0.0, 1.1547005383792515],
        ],
    )
    extruded = {
        "type": "2d_to_3d",
        "method": "extrusion",
        "child": triangle_spec,
        "params": {"h": 2.0},
    }
    part = Part(
        name="prism",
        params={},
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=extruded)],
    )
    sdf = make_sdf_closure(part.computed_envelope(), part)

    pts_3d = jnp.array(
        [
            [0.0, 0.0, 0.0],  # inside prism
            [0.0, 0.0, 5.0],  # above cap
            [3.0, 3.0, 0.0],  # outside laterally
        ]
    )
    compiled = sdf(pts_3d, jnp.zeros((0,)))

    # Hand composition: extrude(polygon_2d, h=2.0) at p_3d → exact cylinder
    # extrusion of the 2D distance with the axial clamp at z = +/- 2.
    pts_2d = pts_3d[..., :2]
    d_2d = shapes.polygon_2d(pts_2d, TRIANGLE_VERTICES)
    d_axial = jnp.abs(pts_3d[..., 2]) - 2.0
    direct = jnp.where(
        (d_2d <= 0.0) & (d_axial <= 0.0),
        jnp.maximum(d_2d, d_axial),
        jnp.sqrt(jnp.maximum(d_2d, 0.0) ** 2 + jnp.maximum(d_axial, 0.0) ** 2)
        + jnp.minimum(jnp.maximum(d_2d, d_axial), 0.0),
    )

    assert jnp.allclose(compiled, direct, atol=1e-5), (
        f"Compiled DSL result {compiled} should match direct construction {direct}"
    )
