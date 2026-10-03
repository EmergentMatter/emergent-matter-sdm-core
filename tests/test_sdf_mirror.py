"""The ``mirror`` transform: reflective (bilateral) symmetry about a plane.

``sdf_transform("mirror", child, n=..., o=...)`` unions the child with its
reflection across the plane ``(o, n)`` (``min(child(p), child(reflect(p)))``),
so it is robust to where the child sits: the positive half, the negative half,
or crossing the plane (unlike a one-sided fold). These tests pin the reflection
map, the exactness (vs. an explicitly modelled symmetric part), the reviewer's
negative-side and crossing-plane cases, non-unit normals, ``$ref``-able planes,
the analytic bbox, the differentiable volume, and a watertight mesh.

They also pin the numerical extrema of the normalisation (short and zero
normals, and gradients w.r.t. ``n``), plus the oblique/offset plane that
axis-aligned cases cannot distinguish, the ``smooth_csg`` seam, and nesting.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    make_param_ref,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.sdf.bbox import (
    UnsupportedExprError,
    infer_sdf_bbox,
)
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.transforms import reflect_plane


def _part(tree, params=None, metadata=None):
    return Part(
        name="mirror-part",
        params=params or {},
        materials=[MaterialRegion(material_id=1, name="mat", sdf_tree=tree)],
        metadata=metadata or {},
    )


def _sphere_at(cx, r=1.0):
    """Unit-ish sphere translated to ``(cx, 0, 0)``, a half authored off-plane."""
    return sdf_transform("translate", sdf_primitive("sphere", r=r), t=[cx, 0.0, 0.0])


def _grid(half=4.0, n=13):
    axis = np.linspace(-half, half, n)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    return jnp.asarray(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1))


def _eval(tree, pts):
    return np.asarray(make_sdf_closure(tree, _part(tree))(jnp.asarray(pts, float), jnp.zeros((0,))))


# ---------------------------------------------------------------------------
# The reflection map itself
# ---------------------------------------------------------------------------


def test_reflect_plane_reflects_all_points():
    # A full reflection (isometry): BOTH sides mirror across the plane; a point
    # on the plane is unchanged.
    n = jnp.array([1.0, 0.0, 0.0])
    o = jnp.array([0.0, 0.0, 0.0])
    out = reflect_plane(
        jnp.array(
            [
                [2.0, 1.0, 0.0],  # +x -> -x
                [-2.0, 1.0, 0.0],  # -x -> +x
                [0.0, 5.0, 3.0],  # on the plane -> unchanged
            ]
        ),
        n,
        o,
    )
    assert jnp.allclose(out[0], jnp.array([-2.0, 1.0, 0.0]))
    assert jnp.allclose(out[1], jnp.array([2.0, 1.0, 0.0]))
    assert jnp.allclose(out[2], jnp.array([0.0, 5.0, 3.0]))


def test_reflect_plane_offset_plane():
    # Plane x = 3: x=1 reflects to x=5 and x=5 reflects to x=1.
    n = jnp.array([1.0, 0.0, 0.0])
    o = jnp.array([3.0, 0.0, 0.0])
    out = reflect_plane(jnp.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), n, o)
    assert jnp.allclose(out[0], jnp.array([5.0, 0.0, 0.0]))
    assert jnp.allclose(out[1], jnp.array([1.0, 0.0, 0.0]))


def test_reflect_plane_normalizes_normal():
    # A non-unit normal gives the same reflection as its normalized version.
    o = jnp.array([0.0, 0.0, 0.0])
    p = jnp.array([[3.0, 1.0, 0.0]])
    a = reflect_plane(p, jnp.array([2.0, 0.0, 0.0]), o)  # |n| = 2
    b = reflect_plane(p, jnp.array([1.0, 0.0, 0.0]), o)
    assert jnp.allclose(a, b, atol=1e-5)
    assert jnp.allclose(a[0], jnp.array([-3.0, 1.0, 0.0]))


# ---------------------------------------------------------------------------
# Normalisation extrema: short / zero normals and gradients w.r.t. n
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mag", [1e2, 1.0, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
def test_reflect_plane_exact_for_any_normal_magnitude(mag):
    # Regression: normalising as `n / sqrt(n·n + 1e-12)` floors |n| at ~1e-6 and
    # silently shrinks the reflection below that: at |n| = 1e-6 the point barely
    # moved (2.4e-7 instead of -3.0). |n| carries no geometric meaning, so every
    # magnitude must give the same reflection as the unit normal.
    o = jnp.array([0.0, 0.0, 0.0])
    p = jnp.array([[3.0, 1.0, -2.0]])
    out = reflect_plane(p, jnp.array([mag, 0.0, 0.0]), o)
    assert jnp.allclose(out[0], jnp.array([-3.0, 1.0, -2.0]), atol=1e-5)


def test_reflect_plane_zero_normal_is_identity_and_finite():
    # A zero normal has no plane to reflect about: degenerate to the identity,
    # with no NaN in the value. (`bbox._mirror_bbox` rejects a constant zero
    # normal, so such a tree fails at inference rather than meshing silently.)
    p = jnp.array([[3.0, 1.0, -2.0]])
    out = reflect_plane(p, jnp.zeros(3), jnp.zeros(3))
    assert jnp.all(jnp.isfinite(out))
    assert jnp.allclose(out, p)


@pytest.mark.parametrize("mag", [1.0, 1e-3, 1e-6, 0.0])
def test_reflect_plane_grad_wrt_normal_is_finite(mag):
    # The normal is $ref-able, so gradients flow into `n`. `sqrt(n·n)` is the
    # classic NaN source at n = 0; `_helpers._length`'s custom JVP plus the
    # where-guarded denominator keep every entry finite, including there.
    p = jnp.array([3.0, 1.0, -2.0])
    o = jnp.zeros(3)
    jac = jax.jacobian(lambda nn: reflect_plane(p, nn, o))(jnp.array([mag, 0.0, 0.0]))
    assert jnp.all(jnp.isfinite(jac))
    if mag == 0.0:
        # n=0 is a stationary point of the reflection (quadratic in n there),
        # so the gradient is exactly zero rather than a ~1/eps spike.
        assert jnp.allclose(jac, 0.0)


def test_reflect_plane_grad_wrt_origin_is_finite():
    p = jnp.array([3.0, 1.0, -2.0])
    n = jnp.array([1.0, 1.0, 0.0])
    jac = jax.jacobian(lambda oo: reflect_plane(p, n, oo))(jnp.array([0.5, 0.0, 0.0]))
    assert jnp.all(jnp.isfinite(jac))


# ---------------------------------------------------------------------------
# Oblique + offset plane: the case axis-aligned tests cannot distinguish
# ---------------------------------------------------------------------------


def test_reflect_plane_oblique_offset_matches_hand_computation():
    # Plane through o=(1,0,0) with n=(1,1,0)/sqrt(2). For p=(3,1,-2):
    #   d = (p-o)·n̂ = (2 + 1)/sqrt(2) = 3/sqrt(2)
    #   p' = p - 2d n̂ = (3,1,-2) - 3*(1,1,0) = (0,-2,-2)
    p = jnp.array([[3.0, 1.0, -2.0]])
    out = reflect_plane(p, jnp.array([1.0, 1.0, 0.0]), jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(out[0], jnp.array([0.0, -2.0, -2.0]), atol=1e-5)


def test_reflect_plane_is_an_involution_and_isometry():
    # Reflecting twice is the identity, and pairwise distances are preserved:
    # the two properties that make `min(child(p), child(reflect(p)))` a true
    # union of congruent halves rather than a distorted copy.
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.normal(size=(16, 3)))
    n = jnp.array([0.3, -1.7, 0.9])  # oblique, non-unit
    o = jnp.array([0.4, 1.1, -0.2])  # off the origin

    q = reflect_plane(p, n, o)
    assert jnp.allclose(reflect_plane(q, n, o), p, atol=1e-5)

    d_p = jnp.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    d_q = jnp.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
    assert jnp.allclose(d_p, d_q, atol=1e-5)


def test_mirror_oblique_plane_matches_explicit_union():
    # End-to-end oblique case: a sphere at (2,0,0) mirrored about the plane
    # x+y=0 lands its image at (0,-2,0).
    mirrored = sdf_transform("mirror", _sphere_at(2.0), n=[1.0, 1.0, 0.0], o=[0.0, 0.0, 0.0])
    image = sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[0.0, -2.0, 0.0])
    explicit = sdf_op("union", [_sphere_at(2.0), image])

    pts = _grid()
    empty = jnp.zeros((0,))
    d_m = make_sdf_closure(mirrored, _part(mirrored))(pts, empty)
    d_e = make_sdf_closure(explicit, _part(explicit))(pts, empty)
    assert jnp.allclose(d_m, d_e, atol=1e-5)


# ---------------------------------------------------------------------------
# Exactness vs. an explicitly modelled symmetric part
# ---------------------------------------------------------------------------


def test_mirror_matches_explicit_union():
    # Author the right sphere only; mirror about the YZ plane. Reflect+union
    # equals an explicit union of the sphere and its mirror image.
    mirrored = sdf_transform("mirror", _sphere_at(2.0), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    explicit = sdf_op("union", [_sphere_at(2.0), _sphere_at(-2.0)])

    f_m = make_sdf_closure(mirrored, _part(mirrored))
    f_e = make_sdf_closure(explicit, _part(explicit))

    pts = _grid()
    empty = jnp.zeros((0,))
    assert jnp.allclose(f_m(pts, empty), f_e(pts, empty), atol=1e-5)


def test_mirror_is_2d_agnostic():
    # A 2-D disc off the y-axis mirrored about the x=0 line matches two discs.
    disc = sdf_transform("translate", sdf_primitive("circle_2d", r=1.0), t=[2.0, 0.0])
    mirrored = sdf_transform("mirror", disc, n=[1.0, 0.0], o=[0.0, 0.0])
    p = jnp.array([[-2.0, 0.0], [2.0, 0.0], [-3.0, 0.0]])
    d = make_sdf_closure(mirrored, _part(mirrored))(p, jnp.zeros((0,)))
    # Centres of both discs are interior (-1.0); a point 1 unit outside is +1.0.
    assert jnp.allclose(d, jnp.array([-1.0, -1.0, 0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Reviewer cases: robust to child placement (regressions the fold failed)
# ---------------------------------------------------------------------------


def test_mirror_child_entirely_on_negative_side():
    # Reviewer case: a child fully on the NEGATIVE half-space. A one-sided fold
    # produced *nothing*; reflect+union gives the child AND its mirror image.
    mirrored = sdf_transform(
        "mirror", _sphere_at(-2.0, r=1.0), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0]
    )
    d = _eval(mirrored, [(-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    assert d[0] == pytest.approx(-1.0, abs=1e-5)  # original, on the negative side
    assert d[1] == pytest.approx(-1.0, abs=1e-5)  # reflected, on the positive side


def test_mirror_child_crossing_plane_has_no_gap():
    # Reviewer case: a non-convex child crossing the plane (a body clear of the
    # plane + a nub straddling it, with a gap between them). The fold dropped the
    # nub's negative-side slice; reflect+union keeps it and stays symmetric.
    body = sdf_transform(
        "translate", sdf_primitive("sphere", r=0.8), t=[1.2, 0.0, 0.0]
    )  # x in [0.4, 2.0]
    nub = sdf_transform(
        "translate", sdf_primitive("sphere", r=0.5), t=[-0.35, 0.0, 0.0]
    )  # x in [-0.85, 0.15]
    mirrored = sdf_transform(
        "mirror", sdf_op("union", [body, nub]), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0]
    )
    sdf = make_sdf_closure(mirrored, _part(mirrored))
    # The exact point the one-sided fold dropped (inside the nub, negative side):
    assert float(sdf(jnp.array([[-0.25, 0.0, 0.0]]), jnp.zeros((0,)))[0]) < 0.0
    # Whole result is symmetric about the plane: f(x) == f(-x).
    xs = np.linspace(-2.4, 2.4, 25)
    line = jnp.asarray(np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=-1))
    d = np.asarray(sdf(line, jnp.zeros((0,))))
    assert np.allclose(d, d[::-1], atol=1e-5)


def test_mirror_non_unit_normal_matches_unit():
    # Reviewer issue 2: a non-unit normal must give the same field (and bbox) as
    # its normalized version: the runtime and bbox both normalize n.
    unit = sdf_transform("mirror", _sphere_at(2.0), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    scaled = sdf_transform("mirror", _sphere_at(2.0), n=[5.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    pts = _grid()
    d_u = make_sdf_closure(unit, _part(unit))(pts, jnp.zeros((0,)))
    d_s = make_sdf_closure(scaled, _part(scaled))(pts, jnp.zeros((0,)))
    assert jnp.allclose(d_u, d_s, atol=1e-5)
    assert infer_sdf_bbox(unit, _part(unit)) == infer_sdf_bbox(scaled, _part(scaled))


# ---------------------------------------------------------------------------
# The seam honours smooth CSG
# ---------------------------------------------------------------------------


def _crossing_child():
    """A sphere straddling x=0, so its mirror image overlaps it at the seam."""
    return sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[0.6, 0.0, 0.0])


def test_mirror_seam_honours_smooth_csg():
    # The mirror's union is a CSG join like any other: with smooth CSG on it must
    # blend, not crease. Previously it hardcoded `jnp.minimum`, so the one seam
    # the transform introduces was the one place the flag did nothing.
    child = _crossing_child()
    mirrored = sdf_transform("mirror", child, n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    explicit = sdf_op(
        "union",
        [
            child,
            sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[-0.6, 0.0, 0.0]),
        ],
    )

    k = 0.3
    pts = _grid(half=2.0, n=15)
    empty = jnp.zeros((0,))
    d_m = make_sdf_closure(mirrored, _part(mirrored), b_smooth_csg=True, d_smooth_k=k)(pts, empty)
    d_e = make_sdf_closure(explicit, _part(explicit), b_smooth_csg=True, d_smooth_k=k)(pts, empty)
    d_hard = make_sdf_closure(mirrored, _part(mirrored))(pts, empty)

    # Smoothed mirror == smoothed explicit union of the same two lobes.
    assert jnp.allclose(d_m, d_e, atol=1e-5)
    # And it actually differs from the hard min near the seam (the blend pulls
    # the field down by up to ~k/4 there).
    assert float(jnp.max(jnp.abs(d_m - d_hard))) > 0.01


def test_mirror_smooth_csg_reads_part_metadata():
    # The metadata route (not just the explicit kwarg) reaches the mirror seam.
    mirrored = sdf_transform("mirror", _crossing_child(), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    part_smooth = _part(mirrored, metadata={"smooth_csg": True})
    seam = jnp.array([[0.0, 0.9, 0.0]])
    empty = jnp.zeros((0,))
    d_smooth = make_sdf_closure(mirrored, part_smooth)(seam, empty)
    d_hard = make_sdf_closure(mirrored, _part(mirrored))(seam, empty)
    assert not jnp.allclose(d_smooth, d_hard, atol=1e-4)


def test_mirror_stays_symmetric_under_smooth_csg():
    # Blending must not break the bilateral symmetry the transform exists for.
    mirrored = sdf_transform("mirror", _crossing_child(), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    sdf = make_sdf_closure(mirrored, _part(mirrored), b_smooth_csg=True, d_smooth_k=0.3)
    xs = np.linspace(-2.0, 2.0, 21)
    line = jnp.asarray(np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=-1))
    d = np.asarray(sdf(line, jnp.zeros((0,))))
    assert np.allclose(d, d[::-1], atol=1e-5)


# ---------------------------------------------------------------------------
# Nesting (the x/y/z octant idiom)
# ---------------------------------------------------------------------------


def test_nested_mirrors_give_octant_symmetry():
    # Three nested mirrors turn one octant into all eight. Note the cost: each
    # mirror evaluates its child twice, so this traces the child 8 times.
    import itertools

    def sphere_at(t):
        return sdf_transform("translate", sdf_primitive("sphere", r=0.5), t=list(t))

    tree = sphere_at((0.9, 0.9, 0.9))
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        tree = sdf_transform("mirror", tree, n=axis, o=[0.0, 0.0, 0.0])
    explicit = sdf_op(
        "union", [sphere_at(signs) for signs in itertools.product((0.9, -0.9), repeat=3)]
    )

    pts = _grid(half=2.0, n=15)
    empty = jnp.zeros((0,))
    d_n = make_sdf_closure(tree, _part(tree))(pts, empty)
    d_e = make_sdf_closure(explicit, _part(explicit))(pts, empty)
    assert jnp.allclose(d_n, d_e, atol=1e-5)


def test_nested_mirror_bbox_covers_all_octants():
    tree = sdf_transform("translate", sdf_primitive("sphere", r=0.5), t=[0.9, 0.9, 0.9])
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        tree = sdf_transform("mirror", tree, n=axis, o=[0.0, 0.0, 0.0])
    (lo, hi) = infer_sdf_bbox(tree, _part(tree))
    assert lo == pytest.approx((-1.4, -1.4, -1.4))
    assert hi == pytest.approx((1.4, 1.4, 1.4))


# ---------------------------------------------------------------------------
# $ref-able plane
# ---------------------------------------------------------------------------


def test_mirror_plane_offset_is_ref_able():
    # The plane's x-offset is a free param; the mirror image tracks it.
    part_params = {"ox": Param("ox", 0.0, free=True, bounds=(-1.0, 1.0), unit="mm")}
    mirrored = sdf_transform(
        "mirror",
        _sphere_at(2.0),
        n=[1.0, 0.0, 0.0],
        o=[make_param_ref("ox"), 0.0, 0.0],
    )
    part = _part(mirrored, part_params)
    sdf = make_sdf_closure(mirrored, part)

    # With ox=0 the mirror image is centred at x=-2; probe its centre.
    probe = jnp.array([[-2.0, 0.0, 0.0]])
    assert jnp.allclose(sdf(probe, jnp.array([0.0])), jnp.array([-1.0]), atol=1e-5)
    # Shift the plane to x=1: negative side reflects about x=1, image centre -> 0.
    assert jnp.allclose(
        sdf(jnp.array([[0.0, 0.0, 0.0]]), jnp.array([1.0])), jnp.array([-1.0]), atol=1e-5
    )


# ---------------------------------------------------------------------------
# Analytic bounding box
# ---------------------------------------------------------------------------


def test_mirror_bbox_reflects_and_hulls():
    mirrored = sdf_transform("mirror", _sphere_at(2.0, r=1.0), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    bbox = infer_sdf_bbox(mirrored, _part(mirrored))
    # Right sphere spans x in [1, 3]; its reflection spans [-3, -1]; hull -> [-3, 3].
    (x0, y0, z0), (x1, y1, z1) = bbox
    assert (x0, x1) == pytest.approx((-3.0, 3.0))
    assert (y0, y1) == pytest.approx((-1.0, 1.0))
    assert (z0, z1) == pytest.approx((-1.0, 1.0))


def test_mirror_bbox_contains_meshed_geometry():
    mirrored = sdf_transform("mirror", _sphere_at(2.0, r=1.0), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    (lo, hi) = infer_sdf_bbox(mirrored, _part(mirrored))
    sdf = make_sdf_closure(mirrored, _part(mirrored))
    # Sample densely; every negative (interior) sample must lie inside the bbox.
    pts = _grid(half=4.0, n=25)
    d = np.asarray(sdf(pts, jnp.zeros((0,))))
    interior = np.asarray(pts)[d < 0.0]
    assert interior.size > 0
    assert np.all(interior >= np.array(lo) - 1e-6)
    assert np.all(interior <= np.array(hi) + 1e-6)


def test_mirror_bbox_refed_plane_raises():
    # A $ref-ed plane cannot be inferred analytically; the caller falls back to
    # the numeric bbox tightener. Same contract as a $ref-ed rotate_matrix.
    part_params = {"ox": Param("ox", 0.0, free=True, bounds=(-1.0, 1.0), unit="mm")}
    mirrored = sdf_transform(
        "mirror",
        _sphere_at(2.0),
        n=[1.0, 0.0, 0.0],
        o=[make_param_ref("ox"), 0.0, 0.0],
    )
    with pytest.raises(UnsupportedExprError):
        infer_sdf_bbox(mirrored, _part(mirrored, part_params))


# ---------------------------------------------------------------------------
# Differentiable metric parity
# ---------------------------------------------------------------------------


def test_mirror_volume_and_grad_match_explicit():
    from software_defined_matter.dsl.expr import compile_expr, expr_metric

    # Radius is a shared free param; volume and its gradient must match between
    # the mirrored half and the explicitly-unioned symmetric model.
    def build(tree):
        return _part(tree, {"r": Param("r", 1.0, free=True, bounds=(0.5, 2.0), unit="mm")})

    r_half = sdf_transform(
        "translate", sdf_primitive("sphere", r=make_param_ref("r")), t=[2.0, 0.0, 0.0]
    )
    l_half = sdf_transform(
        "translate", sdf_primitive("sphere", r=make_param_ref("r")), t=[-2.0, 0.0, 0.0]
    )
    mirrored = sdf_transform("mirror", r_half, n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    explicit = sdf_op("union", [r_half, l_half])

    part_m, part_e = build(mirrored), build(explicit)
    fn_m = compile_expr(expr_metric("volume"), part_m)
    fn_e = compile_expr(expr_metric("volume"), part_e)

    x = jnp.array([1.0])
    v_m, v_e = fn_m(x), fn_e(x)
    assert jnp.isfinite(v_m)
    assert jnp.allclose(v_m, v_e, rtol=0.02)

    g_m = jax.grad(lambda z: fn_m(z).sum())(x)
    g_e = jax.grad(lambda z: fn_e(z).sum())(x)
    assert jnp.all(jnp.isfinite(g_m))
    assert g_m[0] > 0.0  # volume grows with radius
    assert jnp.allclose(g_m, g_e, rtol=0.05, atol=1e-3)


# ---------------------------------------------------------------------------
# Mesh + round-trip
# ---------------------------------------------------------------------------


def test_mirror_meshes_to_symmetric_surface():
    pytest.importorskip("skimage")
    from software_defined_matter._meshing.mesh import extract_mesh
    from software_defined_matter.grid_sampling import BBox3, bind_sdf, eval_chunked, make_grid

    mirrored = sdf_transform("mirror", _sphere_at(2.0, r=1.0), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    part = _part(mirrored)

    half = 4.0
    bbox = BBox3(min_pt=np.full(3, -half), max_pt=np.full(3, half))
    voxel = (2.0 * half) / 60.0
    points, shape = make_grid(bbox, voxel)
    grid = np.asarray(eval_chunked(bind_sdf(part.materials[0].sdf_tree, part), points))

    assert grid.min() < 0.0 < grid.max()
    mesh = extract_mesh(grid.reshape(shape), bbox.min_pt, voxel)
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
    assert np.all(np.isfinite(mesh.vertices))
    # Two mirror-image lobes -> vertex centroid sits on the mirror plane (x≈0).
    assert abs(float(mesh.vertices[:, 0].mean())) < 0.2


def test_mirror_roundtrips_through_sdm(tmp_path):
    from software_defined_matter import io

    mirrored = sdf_transform("mirror", _sphere_at(2.0), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0])
    part = _part(mirrored)
    path = tmp_path / "mirror.sdm"
    io.save(part, path)
    io.validate(path)  # JSON-schema enum must accept "mirror"
    reloaded = io.load(path)

    pts = _grid()
    empty = jnp.zeros((0,))
    d0 = make_sdf_closure(mirrored, part)(pts, empty)
    d1 = make_sdf_closure(reloaded.materials[0].sdf_tree, reloaded)(pts, empty)
    assert jnp.allclose(d0, d1, atol=1e-6)
