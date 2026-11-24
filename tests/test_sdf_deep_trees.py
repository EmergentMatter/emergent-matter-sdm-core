"""Deep built-in trees preserve snapshots, numerical order, and differentiable inputs."""

from __future__ import annotations

import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    Param,
    Part,
    sdf_deform,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.dsl.resolve import make_binding
from software_defined_matter.sdf._tree import snapshot_tree
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure, make_sdf_closure_with_binding
from software_defined_matter.sdf.exactness import (
    EXACT,
    UNKNOWN,
    DistanceAccuracyWarning,
    infer_sdf_exactness,
)


def _mixed_tree(depth):
    leaf = sdf_primitive("sphere", r={"$ref": "r"})
    tree = leaf
    for i in range(depth):
        mode = i % 100
        if mode == 0:
            tree = sdf_op("union", [tree, sdf_primitive("box", b=[1.4, 0.65, 0.7])])
        elif mode == 4:
            tree = sdf_op("subtract", [tree, sdf_primitive("sphere", r=0.25)])
        elif mode == 6:
            tree = sdf_op("intersect", [tree, sdf_primitive("box", b=[2.5, 2, 2])])
        elif mode == 3:
            tree = sdf_transform("scale", tree, s=1.0001)
        elif mode in (2, 8):
            tree = sdf_transform("rotate_z", tree, angle=0.001)
        elif mode == 9:
            tree = sdf_deform("twist", tree, k=0.0002)
        else:
            tree = sdf_transform("translate", tree, t=[0.0004, -0.0002, 0.0001])
    return tree, leaf


def _reference(points, radius, depth):
    """Evaluate the construction schedule with NumPy, independently of the DSL walkers."""
    local = np.asarray(points, dtype=np.float64)
    queries = []
    for i in reversed(range(depth)):
        queries.append(local)
        mode = i % 100
        if mode == 3:
            local = local / 1.0001
        elif mode in (2, 8):
            c, s = np.cos(0.001), np.sin(0.001)
            local = local @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        elif mode == 9:
            angle = 0.0002 * local[..., 1]
            c, s = np.cos(angle), np.sin(angle)
            local = np.stack(
                [
                    c * local[..., 0] - s * local[..., 2],
                    local[..., 1],
                    s * local[..., 0] + c * local[..., 2],
                ],
                axis=-1,
            )
        elif mode not in (0, 4, 6):
            local = local - [0.0004, -0.0002, 0.0001]
    distance = np.linalg.norm(local, axis=-1) - radius
    for i, query in enumerate(reversed(queries)):
        mode = i % 100
        if mode in (0, 6):
            q = np.abs(query) - ([1.4, 0.65, 0.7] if mode == 0 else [2.5, 2, 2])
            box = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)
            distance = (np.minimum if mode == 0 else np.maximum)(distance, box)
        elif mode == 4:
            distance = np.maximum(distance, 0.25 - np.linalg.norm(query, axis=-1))
        elif mode == 3:
            distance = distance * 1.0001
    return distance


@pytest.mark.parametrize("depth", [300, 400, 600])
@pytest.mark.parametrize("entry", ["part", "binding"])
def test_mixed_depth_keeps_float32_values_gradients_and_source_isolation(depth, entry):
    with jax.enable_x64(False):
        limit = sys.getrecursionlimit()
        tree, leaf = _mixed_tree(depth)
        part = Part(name="deep", params={"r": Param("r", 1.6, free=True, bounds=(1.5, 1.8))})
        fn = (
            make_sdf_closure(tree, part)
            if entry == "part"
            else make_sdf_closure_with_binding(tree, make_binding(part))
        )
        points = jnp.array(
            [[3, 0.2, 0.1], [1.6, 0.85, 0.85], [0.1, 0.05, 0.07], [-0.9, 0.2, 0.2]],
            dtype=jnp.float32,
        )
        leaf["params"]["r"] = 20
        compiled = jax.jit(fn)
        for radius in (1.55, 1.7):
            free = jnp.array([radius], dtype=jnp.float32)
            expected = _reference(np.asarray(points), float(free[0]), depth)
            # Repeated float32 coordinate transforms accumulate rounding error.
            assert np.asarray(fn(points, free)) == pytest.approx(expected, abs=3e-4)
            assert np.asarray(compiled(points, free)) == pytest.approx(expected, abs=3e-4)
        assert fn.distance_exactness.outside == UNKNOWN
        if depth == 600:
            value_and_grad = jax.jit(
                jax.value_and_grad(lambda p, v: jnp.sum(fn(p, v)), argnums=(0, 1))
            )
            free = jnp.array([1.6], dtype=jnp.float32)
            value, (point_grad, radius_grad) = value_and_grad(points, free)
            host = np.asarray(points, dtype=np.float64)
            radius = float(free[0])
            step = 1e-4
            expected_radius = (
                _reference(host, radius + step, depth).sum()
                - _reference(host, radius - step, depth).sum()
            ) / (2 * step)
            expected_points = np.empty_like(host)
            for i in range(len(host)):
                for axis in range(3):
                    offset = np.zeros_like(host)
                    offset[i, axis] = step
                    expected_points[i, axis] = (
                        _reference(host + offset, radius, depth).sum()
                        - _reference(host - offset, radius, depth).sum()
                    ) / (2 * step)
            assert np.isfinite(float(value))
            assert point_grad.dtype == radius_grad.dtype == jnp.float32
            assert abs(expected_radius) > 0.5
            # Finite differences avoid branch ties; allow accumulated float32 AD error.
            assert np.asarray(radius_grad) == pytest.approx([expected_radius], abs=2e-3)
            assert np.asarray(point_grad) == pytest.approx(expected_points, abs=2e-3)
        assert sys.getrecursionlimit() == limit


def test_deep_fixed_union_retains_its_exact_disjointness_proof():
    tree = sdf_primitive("sphere", r=1)
    for _ in range(599):
        tree = sdf_transform("translate", tree, t=[0.001, 0, 0])
    tree = sdf_op(
        "union", [tree, sdf_transform("translate", sdf_primitive("box", b=[1, 1, 1]), t=[10, 0, 0])]
    )
    part = Part(name="disjoint")
    fn = make_sdf_closure(tree, part)
    assert fn.distance_exactness.is_exact
    assert np.asarray(infer_sdf_bbox(tree, part)) == pytest.approx(
        np.array([[-0.401, -1, -1], [11, 1, 1]]), abs=1e-9
    )
    assert float(jax.jit(fn)(jnp.array([3, 0, 0], dtype=jnp.float32))) == pytest.approx(
        1.401, abs=1e-4
    )


def test_six_hundred_fold_union_compiles_without_reassociating():
    tree = sdf_primitive("sphere", r=1.2)
    for i in range(600):
        # Keep 600 distinct joins and leaves for preparation and tracing, but let
        # XLA reuse box calculations instead of optimizing 600 different kernels.
        tree = sdf_op(
            "union", [tree, sdf_primitive("box", b=[0.5 + 0.599 * (i % 6) / 5, 0.6, 0.7])]
        )
    fn = make_sdf_closure(tree, Part(name="fold"))
    points = jnp.array([[3, 0, 0], [0, 3, 0], [1.3, 0.6, 0.7], [0, 0, 0]], dtype=jnp.float32)
    assert np.asarray(jax.jit(fn)(points)) == pytest.approx([1.8, 1.8, 0.201, -1.2], abs=1e-6)
    assert fn.distance_exactness.outside == EXACT


def test_deep_consumer_warning_survives_iterative_analysis():
    tree = sdf_primitive("sphere", r=1)
    for _ in range(600):
        tree = sdf_transform("translate", tree, t=[0.001, 0, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        fn = make_sdf_closure(sdf_modifier("round", tree, r=0.1), Part(name="offset"))
    assert fn.distance_exactness.outside == EXACT
    tree["child"] = sdf_primitive("round_box", b=[1, 1, 1], r=2)
    with pytest.warns(DistanceAccuracyWarning, match="round reads a side"):
        make_sdf_closure(sdf_modifier("round", tree, r=0.1), Part(name="invalid"))


def test_snapshot_copies_arrays_tuples_and_shared_subtrees():
    extents = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    child = sdf_transform("translate", sdf_primitive("box", b=extents), t=(0.0, 0.0, 0.0))
    tree = sdf_op("union", [child, child])
    snapshot = snapshot_tree(tree)
    assert snapshot["children"][0] is snapshot["children"][1]
    assert snapshot["children"][0] is not child
    assert isinstance(snapshot["children"][0]["params"]["t"], tuple)
    extents[0] = 10
    assert snapshot["children"][0]["child"]["params"]["b"] == pytest.approx([1, 2, 3], abs=0)


def test_cyclic_geometry_is_rejected_without_recursing():
    tree = sdf_op("union", [])
    tree["children"].append(tree)
    with pytest.raises(ValueError, match="cyclic container"):
        make_sdf_closure(tree, Part(name="cycle"))


@pytest.mark.parametrize("kind", ["loft", "smooth_union", "smoothed_union"])
def test_unknown_multi_child_contracts_do_not_analyse_unused_profiles(monkeypatch, kind):
    from software_defined_matter.sdf import exactness

    def unexpected_proof(*args):
        pytest.fail("An unanalysed operation must not prove its unused child domains")

    monkeypatch.setattr(exactness, "primitive_domain_proven", unexpected_proof)
    polygon = sdf_primitive("polygon_2d", vertices=[[0, 0], [1, 0], [0, 1]])
    if kind == "loft":
        tree = {"type": "loft", "children": [polygon, polygon], "params": {"z": [0, 1]}}
    else:
        tree = sdf_op("union" if kind == "smoothed_union" else kind, [polygon, polygon], k=0.1)
    contract = infer_sdf_exactness(
        tree, Part(name="early-exit"), b_smooth_csg=kind == "smoothed_union"
    )
    assert contract.outside == contract.inside == UNKNOWN
