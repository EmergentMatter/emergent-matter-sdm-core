"""Part.computed_envelope() builds a smooth-union SDF from material regions."""

from __future__ import annotations

from software_defined_matter import MaterialRegion, Part, sdf_primitive


def test_no_materials_returns_none():
    part = Part(name="empty")
    assert part.computed_envelope() is None


def test_single_material_returns_its_sdf_tree():
    tree = sdf_primitive("sphere", r=1.0)
    part = Part(
        name="one",
        materials=[MaterialRegion(material_id=1, name="A", sdf_tree=tree)],
    )
    assert part.computed_envelope() is tree


def test_two_materials_produces_smooth_union_with_default_k():
    tree_a = sdf_primitive("sphere", r=1.0)
    tree_b = sdf_primitive("sphere", r=2.0)
    part = Part(
        name="two",
        materials=[
            MaterialRegion(material_id=1, name="A", sdf_tree=tree_a),
            MaterialRegion(material_id=2, name="B", sdf_tree=tree_b),
        ],
    )
    result = part.computed_envelope()
    assert result["type"] == "op"
    assert result["op"] == "smooth_union"
    assert result["params"]["k"] == 0.05


def test_multi_material_uses_k_from_metadata():
    trees = [sdf_primitive("sphere", r=float(i)) for i in range(1, 4)]
    part = Part(
        name="three",
        materials=[
            MaterialRegion(material_id=i + 1, name=f"M{i}", sdf_tree=t) for i, t in enumerate(trees)
        ],
        metadata={"envelope_smooth_k": 0.1},
    )
    result = part.computed_envelope()
    # Top node: smooth_union(smooth_union(M0, M1), M2)
    assert result["params"]["k"] == 0.1
    assert result["children"][0]["params"]["k"] == 0.1
