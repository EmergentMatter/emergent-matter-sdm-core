"""Subtree sharing preserves distinct bodies and component entry points."""

from __future__ import annotations

import re

import pytest

from software_defined_matter import MaterialRegion, Part, sdf_op, sdf_primitive, sdf_transform
from software_defined_matter.glsl.emit import _GLSLEmitter, emit_glsl


def _part(children):
    return Part(
        name="memo",
        materials=[MaterialRegion(material_id=1, name="body", sdf_tree=sdf_op("union", children))],
    )


def _definitions(source):
    return re.findall(r"float (sdf_n\d+_d3)\(vec3 p\)", source)


@pytest.mark.parametrize("copies", [2, 3])
def test_identical_subtrees_share_bodies_but_keep_component_entry_points(copies):
    children = [
        sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[1.0, 0.0, 0.0])
        for _ in range(copies)
    ]
    emission = emit_glsl(_part(children))
    definitions = _definitions(emission.scene_source)
    # One sphere and one translate body, with aliases for subsequent components.
    assert len(definitions) == copies + 1
    assert emission.scene_source.count("return sdf_sphere(") == 1
    roots = [component["machine"] for component in emission.components]
    assert len(roots) == len(set(roots)) == copies
    assert set(roots) <= set(definitions)


def test_distinct_subtrees_do_not_share():
    children = [
        sdf_transform("translate", sdf_primitive("sphere", r=radius), t=[1.0, 0.0, 0.0])
        for radius in (1.0, 2.0)
    ]
    emission = emit_glsl(_part(children))
    definitions = _definitions(emission.scene_source)
    assert len(definitions) == len(set(definitions)) == 4
    assert emission.scene_source.count("return sdf_sphere(") == 2


def test_cache_uses_structure_not_dictionary_order():
    emitter = _GLSLEmitter(Part(name="memo"), smooth_csg=False, smooth_k=0.25)
    first = sdf_primitive("sphere", r=1.0)
    reordered = dict(reversed(list(first.items())))
    assert emitter.emit_node(first) == emitter.emit_node(reordered)
    assert len(emitter.functions) == 1


def test_unserializable_annotation_emits_without_caching():
    emitter = _GLSLEmitter(Part(name="memo"), smooth_csg=False, smooth_k=0.25)
    node = sdf_primitive("sphere", r=1.0)
    # Direct Python callers may attach annotations ignored by the emitter.
    node["annotation"] = object()
    first, second = emitter.emit_node(node), emitter.emit_node(node)
    assert first != second
    assert len(emitter.functions) == 2
    assert all("return sdf_sphere(" in body for body in emitter.functions)


def test_uncached_fallback_does_not_hide_invalid_geometry():
    emitter = _GLSLEmitter(Part(name="memo"), smooth_csg=False, smooth_k=0.25)
    with pytest.raises(TypeError, match="Cannot emit GLSL"):
        emitter.emit_node(sdf_primitive("sphere", r=object()))
