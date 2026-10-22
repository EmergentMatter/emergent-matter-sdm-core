"""``boundary_blend``: a cutting skin through 3-4 boundary curves and guides."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Part,
    sdf_boundary_blend,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.sdf.bake import bake_raster_field
from software_defined_matter.sdf.boundary_blend_check import (
    boundary_error,
    guide_report,
    quality_report,
    side_report,
)
from software_defined_matter.sdf.boundary_blend_patch import Patch
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.sdf_boundary_blend import make_patch, signed_distance
from software_defined_matter.sdf.validate import UnboundedRootError, validate_material_tree


def _line(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.stack([a, (2 * a + b) / 3, (a + 2 * b) / 3, b])


_SQ = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]  # counter-clockwise from above


def _square():
    a, b, c, d = _SQ
    return [_line(a, b), _line(b, c), _line(c, d), _line(d, a)]


def _triangle():
    a, b, c = _SQ[0], _SQ[1], _SQ[3]
    return [_line(a, b), _line(b, c), _line(c, a)]


def _closure(tree):
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    return jax.jit(make_sdf_closure(tree, part))


def _soap_quarter():
    """A soap bar's top, one quarter: 3 curves, conditions on every side."""
    a, b, h, z = 45.0, 30.0, 14.0, 7.0
    zs = z + 0.6 * (h - z)
    c1 = [[0, 0, h], [0.55 * a, 0, h], [a, 0, zs], [a, 0, z]]
    c2 = [[a, 0, z], [a, 0.72 * b, z], [0.72 * a, b, z], [0, b, z]]
    c3 = [[0, b, z], [0, b, zs], [0, 0.55 * b, h], [0, 0, h]]
    conds = [("normal", (0, 1, 0)), ("normal", (0, 0, 1)), ("normal", (1, 0, 0))]
    return [c1, c2, c3], conds


def _seat_network():
    """Four edges and a 4 x 4 network of guides over a saddle with a pocket."""
    A, B = 55.0, 35.0

    def z(x, y):
        saddle = 18.0 + 10.0 * (x / A) ** 2 - 9.0 * (y / B) ** 2 - 3.0 * (x / A)
        return saddle - 8.0 * (1 - (x / A) ** 2) * (1 - (y / B) ** 2) * (1 - 0.6 * (x / A))

    xs, ys = np.array([-A, -A / 3, A / 3, A]), np.array([-B, -B / 3, B / 3, B])
    t = np.array([0, 1 / 3, 2 / 3, 1.0])
    M = np.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3], axis=1)

    def along(y):
        return np.linalg.solve(M, [(x, y, z(x, y)) for x in xs])

    def across(x):
        return np.linalg.solve(M, [(x, y, z(x, y)) for y in ys])

    edges = [along(-B), across(A), along(B)[::-1], across(-A)[::-1]]
    guides = [along(ys[1]), along(ys[2]), across(xs[1]), across(xs[2])]
    return edges, guides


# ---------------------------------------------------------------- geometry
def test_flat_square_is_a_plane():
    p = Patch(_square(), ["free"] * 4, [[0, 0, 1]] * 4, [1] * 4)
    q = jnp.array([[3, 4, 2.0], [5, 5, -1.5], [15, 5, 3.0]])  # the last is on the extension
    assert np.allclose(signed_distance(q, p, 40.0), [-2.0, 1.5, -3.0], atol=1e-3)


def test_sign_parity_three_and_four_curves():
    """Both loops run counter-clockwise seen from above: above is negative."""
    q = jnp.array([[3, 3, 2.0], [3, 3, -2.0]])
    for curves in (_square(), _triangle()):
        n = len(curves)
        p = Patch(curves, ["free"] * n, [[0, 0, 1]] * n, [1] * n)
        d = np.asarray(signed_distance(q, p, 40.0))
        assert d[0] < 0 < d[1], (n, d)


def test_conditions_are_met():
    curves, conds = _soap_quarter()
    p = Patch(curves, [c[0] for c in conds], [c[1] for c in conds], [1, 1, 1])
    assert boundary_error(p, curves) < 1e-4
    assert max(side_report(p).values()) < 0.1


def test_flip_swaps_sides():
    q = jnp.array([[3, 4, 2.0], [3, 4, -1.0]])
    a = _closure(sdf_boundary_blend(_square(), reach=40.0, check=False))(q)
    b = _closure(sdf_boundary_blend(_square(), reach=40.0, flip=True, check=False))(q)
    assert np.allclose(a, -b, atol=1e-4)


def test_fold_mirrors_a_quarter():
    curves, conds = _soap_quarter()
    f = _closure(sdf_boundary_blend(curves, conds, reach=56.0, fold="xyz", check=False))
    q = jnp.array([[10.0, 8.0, 12.0], [-10.0, 8.0, 12.0], [10.0, -8.0, -12.0]])
    d = np.asarray(f(q))
    assert np.allclose(d, d[0], atol=1e-4)


# ---------------------------------------------------------------- guides
def test_guides_pass_through_every_curve():
    edges, guides = _seat_network()
    p = make_patch({"curves": edges, "guides": guides})
    curves, gaps = guide_report(p)
    assert max(curves.values()) < 1e-4
    assert max(gaps.values()) < 1e-4
    assert quality_report(p, 80.0, n=11)["ok"]


def test_guide_that_misses_is_refused():
    edges, guides = _seat_network()
    guides[0] = guides[0] + np.array([0.0, 0.0, 0.04])
    with pytest.raises(ValueError):
        sdf_boundary_blend(edges, guides=guides)


def test_same_family_guides_crossing_is_refused():
    g1 = np.array([[0, 2, 0], [4, 2, 0], [6, 8, 0], [10, 8, 0.0]])
    g2 = np.array([[0, 8, 0], [4, 8, 0], [6, 2, 0], [10, 2, 0.0]])
    with pytest.raises(ValueError, match="cross|touch"):
        sdf_boundary_blend(_square(), guides=[g1, g2])


def test_compiled_with_guides_keeps_material_below():
    edges, guides = _seat_network()
    # one guide each way keeps the compile quick; the full 4 x 4 network is
    # checked above
    skin = sdf_boundary_blend(edges, guides=[guides[0], guides[2]], reach=80.0, check=False)
    block = sdf_transform("translate", sdf_primitive("box", b=[55.0, 35.0, 20.0]), t=[0, 0, 20])
    f = _closure(sdf_op("subtract", [block, skin]))
    d = np.asarray(f(jnp.array([[0.0, 1.0, 2.0], [0.0, 1.0, 39.0]])))
    assert d[0] < 0 < d[1]


# ---------------------------------------------------------------- API and rules
def test_conditions_must_match_curves():
    with pytest.raises(ValueError, match="conditions"):
        sdf_boundary_blend(_square(), ["free"])


def test_impossible_tangent_is_refused():
    params = {
        "curves": _square(),
        "conditions": ["tangent", "free", "free", "free"],
        "vectors": [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    }
    with pytest.raises(ValueError, match="tangent"):
        make_patch(params)


def test_loop_must_close():
    with pytest.raises(ValueError, match="head to tail"):
        sdf_boundary_blend([[[0, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]])


def test_only_valid_as_a_cutter():
    skin = sdf_boundary_blend(_square(), reach=40.0, check=False)
    with pytest.raises(UnboundedRootError, match="boundary_blend"):
        validate_material_tree(skin)
    validate_material_tree(sdf_op("subtract", [sdf_primitive("box", b=[5.0, 5.0, 5.0]), skin]))


def test_part_declares_schema_0_6():
    skin = sdf_boundary_blend(_square(), reach=40.0, check=False)
    tree = sdf_op("subtract", [sdf_primitive("box", b=[5.0, 5.0, 5.0]), skin])
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    assert part.to_dict()["schema_version"] == "0.6"


def test_bake_matches_analytic():
    tree = sdf_transform(
        "translate", sdf_boundary_blend(_square(), reach=40.0, check=False), t=[0, 0, 5.0]
    )
    f = _closure(tree)
    # an unbounded skin always meets the edge of the box it is baked over
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*domain boundary.*")
        baked = bake_raster_field(f, ([-2, -2, 2], [12, 12, 9]), 0.25, encoding="f32le")
    q = jnp.array([[5.0, 5.0, 5.0], [5.0, 5.0, 7.0], [5.0, 5.0, 3.0], [2.0, 8.0, 8.0]])
    assert np.allclose(f(q), _closure(baked)(q), atol=0.02)


def test_wire_format_is_lists():
    """One list per thing (curves, conditions, vectors, weights, guides),
    one entry per curve, and as many guides as wanted."""
    from software_defined_matter.sdf.validate import validate_document_semantics

    curves, conds = _soap_quarter()
    node = sdf_boundary_blend(curves, conds, reach=56.0, check=False)
    prm = node["params"]
    assert set(prm) == {"curves", "conditions", "vectors", "reach"}
    assert prm["conditions"] == ["normal", "normal", "normal"]
    assert len(prm["vectors"]) == 3
    tree = sdf_op("subtract", [sdf_primitive("box", b=[50.0, 35.0, 17.0]), node])
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    validate_document_semantics(part.to_dict())
    bad = dict(node, params=dict(prm, conditions=["normal", "sideways", "normal"]))
    tree = sdf_op("subtract", [sdf_primitive("box", b=[50.0, 35.0, 17.0]), bad])
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    with pytest.raises(Exception, match="sideways"):
        validate_document_semantics(part.to_dict())


# ---------------------------------------------------------------- differentiable
def _crowned_part():
    """The soap quarter with its crown height a free parameter, folded to a
    whole bar and cut from a block."""
    from software_defined_matter import Param, make_param_ref

    curves, conds = _soap_quarter()
    h = make_param_ref("crown")
    c1 = [[0, 0, h], [0.55 * 45.0, 0, h], curves[0][2], curves[0][3]]
    c3 = [curves[2][0], curves[2][1], [0, 0.55 * 30.0, h], [0, 0, h]]
    skin = sdf_boundary_blend([c1, curves[1], c3], conds, reach=56.0, fold="xyz")
    block = sdf_primitive("box", b=[50.0, 35.0, 17.0])
    tree = sdf_op("subtract", [block, skin])
    part = Part(
        name="t",
        params={"crown": Param(name="crown", value=14.0, free=True, unit="mm")},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
    )
    return part, tree


@pytest.fixture(scope="module")
def crowned():
    part, tree = _crowned_part()
    return make_sdf_closure(tree, part), part


def test_distance_is_differentiable_in_a_control_point(crowned):
    f, part = crowned
    part_free = jnp.asarray([14.0])
    fv = part_free
    q = jnp.array([[5.0, 4.0, 13.5], [10.0, 8.0, 12.6], [30.0, 20.0, 9.0]])
    grad = jax.jacfwd(lambda v: f(q, v))(fv)[:, 0]
    e = 0.01
    fd = (f(q, part_free + e) - f(q, part_free - e)) / (2 * e)
    assert np.allclose(grad, fd, atol=2e-3), (grad, fd)
    # the first point is inside the bar just under the crown: raising the
    # crown buries it deeper, near one for one (the field there is minus the
    # depth below the skin)
    assert -1.05 < float(grad[0]) < -0.8


def test_volume_is_differentiable_in_a_control_point(crowned):
    from software_defined_matter.objectives.metrics import fraction_filled

    f, _ = crowned
    h = 1.0
    xs = jnp.arange(-46.0, 46.01, h)
    ys = jnp.arange(-31.0, 31.01, h)
    zs = jnp.arange(-15.0, 15.01, h)
    g = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)

    def volume(v):
        return jnp.sum(fraction_filled(f(g, v), h)) * h**3

    fv = jnp.asarray([14.0])
    dv = float(jax.grad(volume)(fv)[0])
    e = 0.05
    fd = float((volume(fv + e) - volume(fv - e)) / (2 * e))
    assert abs(dv - fd) < 0.02 * abs(fd), (dv, fd)
    assert dv > 0  # a higher crown is a bigger bar


def test_guides_are_differentiable_too():
    """A dome over a square set by two crossing guides whose height is a
    free parameter: the crossing is re-found as the guides move."""
    from software_defined_matter import Param, make_param_ref

    a = make_param_ref("a")
    u_guide = [[0, 5, 0], [10 / 3, 5, a], [20 / 3, 5, a], [10, 5, 0]]
    v_guide = [[5, 0, 0], [5, 10 / 3, a], [5, 20 / 3, a], [5, 10, 0]]
    skin = sdf_boundary_blend(_square(), guides=[u_guide, v_guide], reach=40.0)
    tree = sdf_op("subtract", [sdf_primitive("box", b=[5.0, 5.0, 5.0]), skin])
    part = Part(
        name="t",
        params={"a": Param(name="a", value=4.0, free=True, unit="mm")},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
    )
    f = make_sdf_closure(skin, part)
    q = jnp.array([[5.0, 5.0, 4.0], [3.0, 6.0, 2.0]])
    fv = jnp.asarray([4.0])
    grad = jax.jacfwd(lambda v: f(q, v))(fv)[:, 0]
    e = 0.01
    fd = (f(q, fv + e) - f(q, fv - e)) / (2 * e)
    assert np.allclose(grad, fd, atol=2e-3), (grad, fd)
    # the guides cross at the middle at 0.75 a: the skin there rises 0.75 per mm of a
    assert abs(float(grad[0]) - 0.75) < 0.02


def test_rate_is_unknown():
    from software_defined_matter.sdf.lipschitz import UNKNOWN, infer_sdf_max_rate

    skin = sdf_boundary_blend(_square(), reach=40.0, check=False)
    tree = sdf_op("subtract", [sdf_primitive("box", b=[5.0, 5.0, 5.0]), skin])
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    assert infer_sdf_max_rate(tree, part) == UNKNOWN


def test_spatial_gradient_is_the_normal_on_and_near_the_skin():
    """Exactly on the skin, and a hair off it, the field's slope is the unit
    normal (not zero, not tangential)."""
    p = Patch(_square(), ["free"] * 4, [[0, 0, 1]] * 4, [1] * 4)
    grad = jax.vmap(jax.grad(lambda x: signed_distance(x[None], p, 40.0)[0]))
    q = jnp.array([[3.0, 4.0, 0.0], [3.0, 4.0, 1e-4], [7.0, 2.0, -1e-4]])
    assert np.allclose(grad(q), [[0, 0, -1]] * 3, atol=1e-5)

    curves, conds = _soap_quarter()
    dome = Patch(curves, [c[0] for c in conds], [c[1] for c in conds], [1, 1, 1])
    on = jnp.stack([dome.S(0.4, 0.5), dome.S(0.7, 0.2)])  # points ON the skin
    g = jax.vmap(jax.grad(lambda x: signed_distance(x[None], dome, 56.0)[0]))(on)
    n = jnp.stack([dome.normal(0.4, 0.5), dome.normal(0.7, 0.2)])
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0, atol=1e-4)
    assert np.allclose(g, n, atol=1e-3)
