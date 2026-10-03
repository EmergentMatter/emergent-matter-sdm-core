"""Chunk sizing stays near CHUNK_BUDGET_BYTES as polygon load grows.
Another way to express it s that the slice shrinks as the tree widens.

``eval_chunked`` used to price memory in points, which assumes an SDF tree
costs O(1) per evaluated point. That holds for analytic primitives and fails
for polygon profiles.
"""

from __future__ import annotations

import pytest

from software_defined_matter.grid_sampling.grid import (
    CHUNK_BUDGET_BYTES,
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    chunk_for_tree,
    tree_point_width,
)


def _polygon(n_verts: int) -> dict:
    return {
        "type": "primitive",
        "kind": "polygon_2d",
        "params": {"vertices": [[float(i), 0.0] for i in range(n_verts)]},
    }


def _union(children: list) -> dict:
    return {"type": "op", "op": "union", "children": children}


def _analytic(n: int) -> dict:
    return _union(
        [{"type": "primitive", "kind": "sphere", "params": {"radius": 1.0 + i}} for i in range(n)]
    )


def test_polygon_vertices_count_toward_the_width() -> None:
    """A polygon is priced by its vertex count, not as one node."""
    assert tree_point_width(_polygon(50)) >= 50


def test_an_analytic_tree_is_narrow() -> None:
    """Spheres are O(1) per point, so 40 of them must not read as wide."""
    assert tree_point_width(_analytic(40)) < 100


def test_a_translate_offset_is_not_mistaken_for_width() -> None:
    """``[x, y, z]`` is three scalars, not a 3-row array.

    Counting it would price every transform in the tree as a profile.
    """
    plain = {"type": "primitive", "kind": "sphere", "params": {"radius": 1.0}}
    moved = {
        "type": "transform",
        "transform": "translate",
        "params": {"offset": [10.0, 20.0, 30.0]},
        "child": plain,
    }
    assert tree_point_width(moved) - tree_point_width(plain) <= 1


def test_the_slice_shrinks_as_the_tree_widens() -> None:
    """Ten times the polygon content, roughly a tenth of the slice."""
    one = chunk_for_tree(_union([_polygon(50) for _ in range(20)]))
    ten = chunk_for_tree(_union([_polygon(50) for _ in range(200)]))
    assert ten < one, "a wider tree must get a smaller slice"
    assert ten <= one / 5, f"slice barely moved: {one} -> {ten}"


def test_peak_is_flat_in_polygon_content() -> None:
    """The property a constant cannot give.

    ``chunk x width``, the intermediate footprint, must stay near the budget
    across two decades of polygon load instead of growing with it.
    """
    for n_polys in (10, 50, 200, 1000):
        tree = _union([_polygon(50) for _ in range(n_polys)])
        width = tree_point_width(tree)
        footprint = chunk_for_tree(tree) * width * 4
        assert footprint <= CHUNK_BUDGET_BYTES * 1.05, (
            f"{n_polys} polygons: {footprint / 1e9:.1f} GB exceeds the budget"
        )


def test_analytic_trees_keep_the_largest_slice() -> None:
    """The fix must not tax the parts that were never the problem."""
    assert chunk_for_tree(_analytic(30)) == MAX_CHUNK_SIZE


def test_the_slice_stays_within_its_bounds() -> None:
    """A very wide tree is still evaluated, on the floor."""
    huge = _union([_polygon(500) for _ in range(5000)])
    assert MIN_CHUNK_SIZE <= chunk_for_tree(huge) <= MAX_CHUNK_SIZE


def test_width_survives_nesting_under_transforms_and_ops() -> None:
    """Real trees bury profiles under transforms.

    A walk that stopped at the first ``child`` would price a whole loft stack
    as one node.
    """
    buried = _polygon(40)
    for _ in range(6):
        buried = {
            "type": "transform",
            "transform": "rotate_z",
            "params": {"angle": 0.3},
            "child": buried,
        }
    assert tree_point_width(_union([buried, _polygon(40)])) >= 80


def test_the_shipped_default_is_the_cut_one() -> None:
    """The constant callers get when they hold only an opaque ``sdf_fn``."""
    import inspect

    from software_defined_matter.grid_sampling.grid import eval_chunked, eval_sdf_grid

    assert DEFAULT_CHUNK_SIZE == 65_536
    for fn in (eval_chunked, eval_sdf_grid):
        got = inspect.signature(fn).parameters["chunk_size"].default
        assert got == DEFAULT_CHUNK_SIZE, f"{fn.__name__} defaults to {got}"


def test_tree_aware_entry_points_default_to_none() -> None:
    """Callers that hold the tree must not ship a point-priced constant:
    ``None`` means "size from the tree via :func:`chunk_for_tree`".
    """
    import inspect

    pytest.importorskip("trimesh")
    pytest.importorskip("skimage")

    from software_defined_matter._meshing.decimate import decimate_mesh
    from software_defined_matter.export import export_part

    for fn in (export_part, decimate_mesh):
        got = inspect.signature(fn).parameters["chunk_size"].default
        assert got is None, f"{fn.__name__} defaults to {got}"
