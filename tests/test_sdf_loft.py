"""The ``loft`` node: Z-interpolated 2-D cross-sections, capped in Z.

A loft of two identical sections must reduce to an extrusion; a loft of two
different sections must interpolate between them; lofts of polygons must
validate and infer a finite bbox (which also exercises the new ``polygon_2d``
2-D bbox support)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    io,
    make_param_ref,
    sdf_2d_to_3d,
    sdf_loft,
    sdf_primitive,
)
from software_defined_matter.sdf import sdf_ops as ops
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure

_NF = jnp.zeros((0,))


def _jit_closure(tree, part=None):
    """JIT-compiled closure for tests that issue many queries.

    Eager mode dispatches every jnp op per query and materialises each
    intermediate, so multi-query tests spend their time on dispatch overhead
    rather than arithmetic. One XLA compile replaces it. The compiled path is
    the production path (grid_sampling evaluates jitted), and the lighter
    tests in this file keep the eager path covered.
    """
    return jax.jit(make_sdf_closure(tree, part or Part(name="x")))


def _square(n):
    """A closed square polygon of half-size n (numeric verts)."""
    return sdf_primitive("polygon_2d", vertices=[[-n, -n], [n, -n], [n, n], [-n, n]])


def test_loft_two_equal_sections_equals_extrusion():
    # Equal circles at z=±5 -> identical to a circle extruded by half-height 5.
    loft = sdf_loft(
        [sdf_primitive("circle_2d", r=2.0), sdf_primitive("circle_2d", r=2.0)], z=[-5.0, 5.0]
    )
    extr = sdf_2d_to_3d("extrusion", sdf_primitive("circle_2d", r=2.0), h=5.0)
    part = Part(name="x")
    f_loft = _jit_closure(loft, part)
    f_extr = _jit_closure(extr, part)
    p = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 0.0, 4.9],
            [0.0, 0.0, 5.5],
            [3.0, 0.0, 0.0],
        ]
    )
    assert jnp.allclose(f_loft(p, _NF), f_extr(p, _NF), atol=1e-5)


def test_loft_interpolates_between_sections():
    # Box half-size 4 at z=-5 tapering to half-size 2 at z=+5. At the midplane
    # (z=0) the interpolated half-width is ~3, so (±3, 0, 0) sits on the surface.
    loft = sdf_loft([_square_box(4.0), _square_box(2.0)], z=[-5.0, 5.0])
    f = _jit_closure(loft)
    mid_on = f(jnp.array([[3.0, 0.0, 0.0]]), _NF)[0]
    mid_in = f(jnp.array([[2.5, 0.0, 0.0]]), _NF)[0]
    mid_out = f(jnp.array([[3.5, 0.0, 0.0]]), _NF)[0]
    assert abs(float(mid_on)) < 1e-4
    assert float(mid_in) < 0.0 < float(mid_out)
    # the wide end (z~-5) is half-width 4, the narrow end (z~+5) half-width 2
    assert float(f(jnp.array([[3.5, 0.0, -4.9]]), _NF)[0]) < 0.0  # inside wide end
    assert float(f(jnp.array([[2.5, 0.0, 4.9]]), _NF)[0]) > 0.0  # outside narrow end


def _square_box(n):
    return sdf_primitive("box_2d", b=[n, n])


def test_loft_z_cap():
    loft = sdf_loft([sdf_primitive("circle_2d", r=2.0)] * 2, z=[0.0, 10.0])
    f = make_sdf_closure(loft, Part(name="x"))
    assert float(f(jnp.array([[0.0, 0.0, 5.0]]), _NF)[0]) < 0.0  # inside the span
    assert float(f(jnp.array([[0.0, 0.0, -1.0]]), _NF)[0]) > 0.0  # below the span
    assert float(f(jnp.array([[0.0, 0.0, 11.0]]), _NF)[0]) > 0.0  # above the span


def test_loft_three_sections_piecewise():
    # waist: r=3 at the ends, r=1 in the middle.
    loft = sdf_loft(
        [
            sdf_primitive("circle_2d", r=3.0),
            sdf_primitive("circle_2d", r=1.0),
            sdf_primitive("circle_2d", r=3.0),
        ],
        z=[-10.0, 0.0, 10.0],
    )
    f = make_sdf_closure(loft, Part(name="x"))
    assert float(f(jnp.array([[2.0, 0.0, -9.9]]), _NF)[0]) < 0.0  # inside fat end
    assert float(f(jnp.array([[2.0, 0.0, 0.0]]), _NF)[0]) > 0.0  # outside thin waist


def test_loft_polygon_bbox_inferred():
    # lofted polygons must yield a finite bbox WITHOUT an explicit metadata bbox
    # (exercises the new polygon_2d 2-D bbox + loft bbox).
    loft = sdf_loft([_square(5.0), _square(2.0)], z=[0.0, 20.0])
    part = Part(name="x", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=loft)])
    (lo, hi) = infer_sdf_bbox(loft, part)
    assert lo == (-5.0, -5.0, 0.0)
    assert hi == (5.0, 5.0, 20.0)


def test_polygon_2d_extrusion_bbox_inferred():
    # bonus: a plain polygon extrusion is now self-bounding (no metadata bbox).
    extr = sdf_2d_to_3d("extrusion", _square(3.0), h=4.0)
    part = Part(name="x", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=extr)])
    (lo, hi) = infer_sdf_bbox(extr, part)
    assert lo == (-3.0, -3.0, -4.0)
    assert hi == (3.0, 3.0, 4.0)


def test_loft_validates_against_schema_and_roundtrips():
    loft = sdf_loft([_square(5.0), _square(3.0)], z=[0.0, 15.0])
    part = Part(
        name="loft_part", materials=[MaterialRegion(material_id=1, name="PA12", sdf_tree=loft)]
    )
    io.validate(part)  # schema + semantic (boundedness)
    again = Part.from_dict(part.to_dict())
    assert again.materials[0].sdf_tree["type"] == "loft"
    assert again.materials[0].sdf_tree["params"]["z"] == [0.0, 15.0]


def test_loft_two_circles_is_a_frustum():
    # Independent analytic check: a loft of two circles (r0,r1) is a truncated
    # cone: at station z the cross-section is a circle of radius r(z) =
    # lerp(r0, r1). Verify the zero-isosurface lands at exactly r(z), inside is
    # negative, outside positive, without referring to ops.loft at all.
    r0, r1, z0, z1 = 2.0, 5.0, -4.0, 6.0
    loft = sdf_loft(
        [sdf_primitive("circle_2d", r=r0), sdf_primitive("circle_2d", r=r1)], z=[z0, z1]
    )
    f = _jit_closure(loft)
    for z in (-3.0, 0.0, 3.0, 5.0):
        frac = (z - z0) / (z1 - z0)
        r = r0 * (1 - frac) + r1 * frac
        assert abs(float(f(jnp.array([[r, 0.0, z]]), _NF)[0])) < 1e-3  # on surface
        assert float(f(jnp.array([[r - 0.5, 0.0, z]]), _NF)[0]) < 0.0  # inside
        assert float(f(jnp.array([[r + 0.5, 0.0, z]]), _NF)[0]) > 0.0  # outside


def test_loft_section_param_reacts_and_grad_flows():
    # A free Param on the section profiles must flow into the loft SDF, react to
    # the free-vec, and pass gradients (the loft must stay optimisable).
    part = Part(
        name="x",
        params={"r": Param("r", value=2.0, free=True, bounds=(0.5, 8.0), unit="mm")},
        materials=[
            MaterialRegion(
                material_id=1,
                name="m",
                sdf_tree=sdf_loft(
                    [
                        sdf_primitive("circle_2d", r=make_param_ref("r")),
                        sdf_primitive("circle_2d", r=make_param_ref("r")),
                    ],
                    z=[-5.0, 5.0],
                ),
            )
        ],
    )
    f = _jit_closure(part.materials[0].sdf_tree, part)
    centre = jnp.array([[0.0, 0.0, 0.0]])
    assert jnp.allclose(f(centre, jnp.array([2.0]))[0], -2.0, atol=1e-4)  # r=2 -> d=-2
    assert jnp.allclose(f(centre, jnp.array([4.0]))[0], -4.0, atol=1e-4)  # reacts to free-vec
    assert abs(float(f(jnp.array([[3.0, 0.0, 0.0]]), jnp.array([3.0]))[0])) < 1e-4  # surface
    # loss = d(centre)^2 = r^2 -> dloss/dr = 2r = 6 at r=3
    g = jax.grad(lambda fv: f(centre, fv)[0] ** 2)(jnp.array([3.0]))
    assert jnp.allclose(g, 6.0, atol=1e-4)


def test_loft_z_stations_can_be_param_driven():
    # The axial stations themselves can be free params (a span knob).
    part = Part(
        name="x",
        params={"span": Param("span", value=10.0, free=True, bounds=(2.0, 40.0), unit="mm")},
        materials=[
            MaterialRegion(
                material_id=1,
                name="m",
                sdf_tree=sdf_loft(
                    [sdf_primitive("circle_2d", r=2.0), sdf_primitive("circle_2d", r=2.0)],
                    z=[0.0, make_param_ref("span")],
                ),
            )
        ],
    )
    f = make_sdf_closure(part.materials[0].sdf_tree, part)
    p = jnp.array([[0.0, 0.0, 8.0]])
    assert float(f(p, jnp.array([10.0]))[0]) < 0.0  # z=8 inside a span of 10
    assert float(f(p, jnp.array([6.0]))[0]) > 0.0  # z=8 outside a span of 6


def test_loft_rejects_degenerate_inputs():
    # Fewer than 2 sections.
    with pytest.raises(ValueError, match="at least 2"):
        make_sdf_closure(sdf_loft([sdf_primitive("circle_2d", r=1.0)], z=[0.0]), Part(name="x"))
    # Missing the required 'z' stations.
    bad = {
        "type": "loft",
        "children": [sdf_primitive("circle_2d", r=1.0), sdf_primitive("circle_2d", r=1.0)],
        "params": {},
    }
    with pytest.raises(ValueError, match="'z'"):
        make_sdf_closure(bad, Part(name="x"))
    # len(z) != len(children).
    with pytest.raises(ValueError, match="len\\(z\\)"):
        make_sdf_closure(
            sdf_loft(
                [sdf_primitive("circle_2d", r=2.0), sdf_primitive("circle_2d", r=4.0)],
                z=[0.0, 5.0, 10.0],
            ),
            Part(name="x"),
        )
    # Non-ascending / duplicate z stations.
    with pytest.raises(ValueError, match="ascending"):
        make_sdf_closure(
            sdf_loft(
                [sdf_primitive("circle_2d", r=2.0), sdf_primitive("circle_2d", r=4.0)],
                z=[10.0, 0.0],
            ),
            Part(name="x"),
        )
    with pytest.raises(ValueError, match="ascending"):
        make_sdf_closure(
            sdf_loft(
                [sdf_primitive("circle_2d", r=2.0), sdf_primitive("circle_2d", r=4.0)], z=[0.0, 0.0]
            ),
            Part(name="x"),
        )


def test_loft_smooth_passes_through_sections_and_curves():
    # Sections r=4,2,4 at z=-10,0,10. Smooth (C1 monotone PCHIP) must still pass
    # THROUGH the interior section (z=0 -> r=2), but interpolate as a curve, not
    # the straight line the linear loft gives (mid-segment radius != lerp).
    secs = [
        sdf_primitive("circle_2d", r=4.0),
        sdf_primitive("circle_2d", r=2.0),
        sdf_primitive("circle_2d", r=4.0),
    ]
    z = [-10.0, 0.0, 10.0]
    lin = _jit_closure(sdf_loft(secs, z=z))
    sm = _jit_closure(sdf_loft(secs, z=z, smooth=True))

    def radius(f, zq):
        rs = jnp.linspace(0.0, 6.0, 2401)
        pts = jnp.stack([rs, jnp.zeros_like(rs), jnp.full_like(rs, zq)], axis=-1)
        return float(rs[int(jnp.argmin(jnp.abs(f(pts, _NF))))])

    assert abs(radius(sm, 0.0) - 2.0) < 0.05  # passes through interior section
    assert abs(radius(lin, -5.0) - 3.0) < 0.05  # linear: midpoint lerp(4,2)=3
    assert abs(radius(sm, -5.0) - 3.0) > 0.05  # smooth: curved, not the lerp


def test_loft_smooth_flag_roundtrips():
    loft = sdf_loft([_square(5.0), _square(3.0)], z=[0.0, 15.0], smooth=True)
    assert loft["params"]["smooth"] is True
    part = Part(name="x", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=loft)])
    io.validate(part)
    assert Part.from_dict(part.to_dict()).materials[0].sdf_tree["params"]["smooth"] is True


def test_loft_op_matches_direct():
    # the compiled loft equals a direct ops.loft call.
    loft = sdf_loft(
        [sdf_primitive("circle_2d", r=2.0), sdf_primitive("circle_2d", r=4.0)], z=[-3.0, 3.0]
    )
    f = _jit_closure(loft)
    from software_defined_matter.sdf import sdf_shapes as shapes

    p = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, -3.0], [3.0, 0.0, 3.0], [0.0, 0.0, 4.0]])
    direct = ops.loft(
        [lambda q: shapes.circle_2d(q, 2.0), lambda q: shapes.circle_2d(q, 4.0)],
        jnp.array([-3.0, 3.0]),
        p,
    )
    assert jnp.allclose(f(p, _NF), direct, atol=1e-6)


# ---------------------------------------------------------------------------
# interp="shape": interpolate the polygon VERTICES (not the field). Eliminates
# the convex-feature bulge that field interpolation produces at a swept edge.
# ---------------------------------------------------------------------------


def _sq(cx, n):
    """A 4-vertex square of half-size n centred at (cx, 0), consistent winding."""
    return sdf_primitive(
        "polygon_2d", vertices=[[cx - n, -n], [cx + n, -n], [cx + n, n], [cx - n, n]]
    )


def test_loft_shape_equal_sections_equals_polygon_extrusion():
    # Shape-interp loft of two equal squares == that square extruded. Also checks
    # the batched polygon SDF agrees with the canonical fixed-vertex polygon_2d.
    loft = sdf_loft([_sq(0.0, 3.0), _sq(0.0, 3.0)], z=[-5.0, 5.0], interp="shape")
    extr = sdf_2d_to_3d("extrusion", _sq(0.0, 3.0), h=5.0)
    f_loft = _jit_closure(loft)
    f_extr = _jit_closure(extr)
    p = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 0.0, 4.9],
            [0.0, 0.0, 5.5],
        ]
    )
    assert jnp.allclose(f_loft(p, _NF), f_extr(p, _NF), atol=1e-5)


def test_loft_shape_no_convex_bulge_at_swept_corner():
    # THE point of shape interp. Two equal squares SWEPT in x (centre 0 -> 6) at
    # z=±5. The midplane outline is the square centred at x=3 (vertex-interpolated),
    # so its corner sits EXACTLY at (6, 3). Shape interp places it there; field
    # interp blends two corner-distance cones and the zero-set misses the corner.
    secs = [_sq(0.0, 3.0), _sq(6.0, 3.0)]
    z = [-5.0, 5.0]
    f_shape = _jit_closure(sdf_loft(secs, z=z, interp="shape"))
    f_field = _jit_closure(sdf_loft(secs, z=z))  # default field
    corner = jnp.array([[6.0, 3.0, 0.0]])
    # shape interp: the swept corner is exactly on the surface.
    assert abs(float(f_shape(corner, _NF)[0])) < 1e-3
    # field interp: the corner is NOT on the surface (it deviates by ~1.5 mm).
    assert abs(float(f_field(corner, _NF)[0])) > 0.5
    # both agree on a FLAT face midpoint (field interp is exact for flat faces):
    face = jnp.array([[6.0, 0.0, 0.0]])  # +x face of the midplane square (centre 3, half 3)
    assert abs(float(f_shape(face, _NF)[0])) < 1e-3
    assert abs(float(f_field(face, _NF)[0])) < 1e-3
    # interior / exterior sanity for shape interp at the midplane.
    assert float(f_shape(jnp.array([[3.0, 0.0, 0.0]]), _NF)[0]) < 0.0  # centre, inside
    assert float(f_shape(jnp.array([[7.0, 0.0, 0.0]]), _NF)[0]) > 0.0  # outside


def test_loft_shape_interpolates_vertices_linearly():
    # Square half 4 -> half 2, shape interp. Midplane half-width is the vertex
    # lerp = 3 (same as field interp for flat faces), confirming the linear path.
    f = make_sdf_closure(
        sdf_loft([_sq(0.0, 4.0), _sq(0.0, 2.0)], z=[-5.0, 5.0], interp="shape"), Part(name="x")
    )
    assert abs(float(f(jnp.array([[3.0, 0.0, 0.0]]), _NF)[0])) < 1e-3  # on surface
    assert float(f(jnp.array([[2.5, 0.0, 0.0]]), _NF)[0]) < 0.0  # inside
    assert float(f(jnp.array([[3.5, 0.0, 0.0]]), _NF)[0]) > 0.0  # outside


def test_loft_shape_smooth_curves_through_interior_section():
    # Swept square: centre 0 -> 8 -> 0 at z=-10,0,10. Smooth (PCHIP) shape interp
    # passes THROUGH the interior section (midplane centre at x=8), but at z=-5
    # curves (centre != linear lerp of 0 and 8 = 4).
    secs = [_sq(0.0, 2.0), _sq(8.0, 2.0), _sq(0.0, 2.0)]
    z = [-10.0, 0.0, 10.0]
    lin = _jit_closure(sdf_loft(secs, z=z, interp="shape"))
    sm = _jit_closure(sdf_loft(secs, z=z, interp="shape", smooth=True))

    def centre_x(f, zq):
        xs = jnp.linspace(-4.0, 14.0, 3601)
        pts = jnp.stack([xs, jnp.zeros_like(xs), jnp.full_like(xs, zq)], axis=-1)
        d = f(pts, _NF)
        inside = xs[d < 0.0]
        return float(0.5 * (inside.min() + inside.max()))

    assert abs(centre_x(sm, 0.0) - 8.0) < 0.05  # passes through interior section
    assert abs(centre_x(lin, -5.0) - 4.0) < 0.05  # linear: lerp(0,8)=4
    assert abs(centre_x(sm, -5.0) - 4.0) > 0.05  # smooth: curved, not the lerp


def test_loft_shape_requires_polygon_children():
    # circle_2d has no 'vertices' -> shape interp is rejected at compile.
    bad = sdf_loft(
        [sdf_primitive("circle_2d", r=2.0), sdf_primitive("circle_2d", r=2.0)],
        z=[0.0, 5.0],
        interp="shape",
    )
    with pytest.raises(ValueError, match="polygon_2d"):
        make_sdf_closure(bad, Part(name="x"))


def test_loft_shape_requires_equal_vertex_counts():
    # a square (4 verts) and a pentagon (5 verts) have no vertex correspondence.
    penta = sdf_primitive("polygon_2d", vertices=[[3, 0], [1, 3], [-3, 2], [-3, -2], [1, -3]])
    f = make_sdf_closure(
        sdf_loft([_sq(0.0, 3.0), penta], z=[0.0, 5.0], interp="shape"), Part(name="x")
    )
    with pytest.raises(ValueError, match="equal vertex counts"):
        f(jnp.array([[0.0, 0.0, 2.5]]), _NF)


def test_loft_shape_param_vertices_react_and_grad_flows():
    # A free Param driving a vertex must flow through shape interp and pass grads.
    # 'h' drives the +x face of a box x in [-3, h], y in [-3, 3].
    H = make_param_ref("h")

    def sq():
        return sdf_primitive("polygon_2d", vertices=[[-3, -3], [H, -3], [H, 3], [-3, 3]])

    part = Part(
        name="x",
        params={"h": Param("h", value=2.0, free=True, bounds=(1.0, 8.0), unit="mm")},
        materials=[
            MaterialRegion(
                material_id=1,
                name="m",
                sdf_tree=sdf_loft([sq(), sq()], z=[-5.0, 5.0], interp="shape"),
            )
        ],
    )
    f = _jit_closure(part.materials[0].sdf_tree, part)
    on = jnp.array([[2.0, 0.0, 0.0]])
    assert abs(float(f(on, jnp.array([2.0]))[0])) < 1e-3  # +x face at h=2
    assert float(f(on, jnp.array([4.0]))[0]) < 0.0  # reacts: h=4 -> (2,0) inside
    # at (0,0) the nearest face is +x at distance h (for h<3): f=-h, loss=h^2 ->
    # dloss/dh = 2h = 4 at h=2.
    g = jax.grad(lambda fv: f(jnp.array([[0.0, 0.0, 0.0]]), fv)[0] ** 2)(jnp.array([2.0]))
    assert jnp.allclose(g, 4.0, atol=1e-3)


def test_loft_shape_interp_flag_roundtrips_and_validates():
    loft = sdf_loft([_sq(0.0, 5.0), _sq(0.0, 3.0)], z=[0.0, 15.0], interp="shape")
    assert loft["params"]["interp"] == "shape"
    part = Part(name="x", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=loft)])
    io.validate(part)  # schema + bbox (polygon inference)
    again = Part.from_dict(part.to_dict())
    assert again.materials[0].sdf_tree["params"]["interp"] == "shape"


# ---------------------------------------------------------------------------
# interp="shape" over CURVE children: loft bspline_2d / bezier_2d by
# interpolating their control points (compact + smooth + no scallop).
# ---------------------------------------------------------------------------


def _rounded(cx=0.0):
    """8-point closed control polygon (a rounded blob) centred at (cx, 0)."""
    base = [(4, 0), (3, 3), (0, 4), (-3, 3), (-4, 0), (-3, -3), (0, -4), (3, -3)]
    return [[float(x + cx), float(y)] for x, y in base]


def _bspl(cx=0.0):
    return sdf_primitive("bspline_2d", control_points=_rounded(cx))


def test_loft_shape_bspline_equal_sections_equals_extrusion():
    # Two identical bspline sections shape-lofted == that bspline extruded.
    loft = sdf_loft([_bspl(0.0), _bspl(0.0)], z=[-5.0, 5.0], interp="shape")
    extr = sdf_2d_to_3d("extrusion", _bspl(0.0), h=5.0)
    fl = _jit_closure(loft)
    fe = _jit_closure(extr)
    p = jnp.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 4.9], [0.0, 0.0, 5.5]]
    )
    assert jnp.allclose(fl(p, _NF), fe(p, _NF), atol=1e-4)


def test_loft_shape_curve_equals_sampled_polygon_shape_loft():
    # THE correctness identity: shape-lofting bspline CONTROL POINTS == shape-
    # lofting the SAMPLED outline polygons (sampling is a fixed linear map, so it
    # commutes with the vertex interpolation). Same no-bulge/watertight guarantee.
    import numpy as np

    from software_defined_matter.sdf._helpers import bspline_outline

    c0, c6 = _rounded(0.0), _rounded(6.0)
    f_curve = _jit_closure(
        sdf_loft(
            [
                sdf_primitive("bspline_2d", control_points=c0),
                sdf_primitive("bspline_2d", control_points=c6),
            ],
            z=[-5.0, 5.0],
            smooth=True,
            interp="shape",
        )
    )
    o0 = np.asarray(bspline_outline(jnp.asarray(c0), 16)).tolist()
    o6 = np.asarray(bspline_outline(jnp.asarray(c6), 16)).tolist()
    f_poly = _jit_closure(
        sdf_loft(
            [sdf_primitive("polygon_2d", vertices=o0), sdf_primitive("polygon_2d", vertices=o6)],
            z=[-5.0, 5.0],
            smooth=True,
            interp="shape",
        )
    )
    pts = jnp.array(
        [
            [x, y, z]
            for x in (-2.0, 0.0, 3.0, 8.0)
            for y in (-3.0, 0.0, 3.0)
            for z in (-3.0, 0.0, 3.0)
        ]
    )
    assert jnp.allclose(f_curve(pts, _NF), f_poly(pts, _NF), atol=1e-5)


def test_loft_shape_bspline_tracks_interpolated_curve_not_field():
    # At the midplane, shape interp == the bspline of the midpoint control polygon
    # (no convex bulge); field interp deviates from it.
    c0, c6 = _rounded(0.0), _rounded(6.0)
    secs = [
        sdf_primitive("bspline_2d", control_points=c0),
        sdf_primitive("bspline_2d", control_points=c6),
    ]
    f_shape = _jit_closure(sdf_loft(secs, z=[-5.0, 5.0], interp="shape"))
    f_field = _jit_closure(sdf_loft(secs, z=[-5.0, 5.0]))  # field
    cmid = [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in zip(c0, c6, strict=False)]
    f_mid = _jit_closure(sdf_primitive("bspline_2d", control_points=cmid))
    xs = jnp.linspace(-6.0, 12.0, 40)
    ys = jnp.linspace(-7.0, 7.0, 30)
    gx, gy = jnp.meshgrid(xs, ys)
    P3 = jnp.stack([gx.ravel(), gy.ravel(), jnp.zeros(gx.size)], axis=-1)
    d_shape, d_field = f_shape(P3, _NF), f_field(P3, _NF)
    d_mid = f_mid(P3[:, :2], _NF)
    assert float(jnp.max(jnp.abs(d_shape - d_mid))) < 0.05  # shape == interpolated curve
    assert float(jnp.max(jnp.abs(d_shape - d_field))) > 0.1  # and is NOT the field interp


def test_loft_shape_bezier_children_compile_and_cap():
    def cp(cx):
        return [
            [-4 + cx, 0],
            [-4 + cx, 3],
            [4 + cx, 3],
            [4 + cx, 0],
            [4 + cx, -3],
            [-4 + cx, -3],
        ]  # 3K=6, K=2

    secs = [
        sdf_primitive("bezier_2d", control_points=cp(0.0)),
        sdf_primitive("bezier_2d", control_points=cp(6.0)),
    ]
    f = make_sdf_closure(sdf_loft(secs, z=[-5.0, 5.0], interp="shape"), Part(name="x"))
    assert float(f(jnp.array([[0.0, 0.0, 0.0]]), _NF)[0]) < 0.0  # inside the span
    assert float(f(jnp.array([[0.0, 0.0, -6.0]]), _NF)[0]) > 0.0  # below the z-cap


def test_loft_shape_rejects_mixed_child_kinds():
    bad = sdf_loft([_sq(0.0, 3.0), _bspl(0.0)], z=[0.0, 5.0], interp="shape")
    with pytest.raises(ValueError, match="SAME kind"):
        make_sdf_closure(bad, Part(name="x"))


def test_loft_shape_curve_rejects_unequal_control_point_counts():
    a = sdf_primitive("bspline_2d", control_points=_rounded(0.0))  # 8
    b = sdf_primitive("bspline_2d", control_points=[[4, 0], [0, 4], [-4, 0], [0, -4], [2, 2]])  # 5
    f = make_sdf_closure(sdf_loft([a, b], z=[0.0, 5.0], interp="shape"), Part(name="x"))
    with pytest.raises(ValueError, match="equal control-point counts"):
        f(jnp.array([[0.0, 0.0, 2.5]]), _NF)


def test_loft_shape_curve_param_control_point_grad_flows():
    # A free param driving a bspline control point must flow + pass gradients.
    R = make_param_ref("r")

    def prof():
        return sdf_primitive("bspline_2d", control_points=[[R, -3], [-3, -3], [-3, 3], [R, 3]])

    part = Part(
        name="x",
        params={"r": Param("r", value=4.0, free=True, bounds=(1.0, 8.0), unit="mm")},
        materials=[
            MaterialRegion(
                material_id=1,
                name="m",
                sdf_tree=sdf_loft([prof(), prof()], z=[-5.0, 5.0], interp="shape"),
            )
        ],
    )
    f = _jit_closure(part.materials[0].sdf_tree, part)
    c = jnp.array([[0.0, 0.0, 0.0]])
    assert (
        abs(float(f(c, jnp.array([6.0]))[0]) - float(f(c, jnp.array([4.0]))[0])) > 1e-3
    )  # reacts to r
    g = jax.grad(lambda fv: f(c, fv)[0])(jnp.array([4.0]))
    assert bool(jnp.all(jnp.isfinite(g))) and abs(float(g[0])) > 1e-6  # grad flows, nonzero
