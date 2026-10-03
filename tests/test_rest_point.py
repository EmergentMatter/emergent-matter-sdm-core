"""``sdm_rest_point`` and the rest-space cutaway: material coordinates.

A component's animated deform prefix -- a run of point-warp deforms at the top
of its tree -- IS its motion; everything below is static placement. Material
space is the query pulled through exactly that prefix, and two things read it:

* ``sdf_scene_rcut`` cuts each component at its REST point, so a section is
  taken once in material coordinates and then travels with the material instead
  of re-slicing deformed geometry against a stationary world plane every frame;
* ``sdm_rest_point(p, cid)`` hands the same coordinates to the host, which is
  what makes motion visible at all on a monochrome surface -- a pattern painted
  in world space stands still while the body turns through it.

THE FAILURE THIS FILE EXISTS FOR IS SILENCE. Every way of getting it wrong emits
a scene that compiles and renders: a missing prefix emits ``return p`` and the
pattern simply never moves, which was reported from a live viewer as "the
pattern on the surface does not change, and that's the whole reason for the
pattern". So the assertions are that the map is NOT the identity where a warp
exists, and that it is EXACTLY the identity where none does.
"""

from __future__ import annotations

import re

import pytest

from software_defined_matter import (
    MaterialRegion,
    Part,
    sdf_deform,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.glsl import emit_glsl
from software_defined_matter.glsl.emit import _deform_prefix

ANGLE = 0.35


def _emit(tree):
    part = Part(name="t", materials=[MaterialRegion(material_id="m", name="m", sdf_tree=tree)])
    part.metadata["bbox"] = [[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]]
    return emit_glsl(part).scene_source


def _fn(src, name):
    m = re.search(rf"^\w+\s+{re.escape(name)}\s*\([^)]*\)\s*\{{(.*?)\n\}}", src, re.S | re.M)
    assert m, f"{name} not emitted"
    return m.group(1)


def _plate(**kw):
    """A crossing plate under a twist_linear -- the motivating shape."""
    return sdf_deform(
        "twist_linear",
        sdf_primitive("box", b=[2.0, 0.3, 1.0]),
        axis=[1.0, 0.0],
        u0=-2.0,
        u1=2.0,
        angle_0=-ANGLE,
        angle_1=ANGLE,
        **kw,
    )


# ---------------------------------------------------------------------------
# The prefix
# ---------------------------------------------------------------------------


def test_the_prefix_stops_at_the_first_non_warp():
    """The chain is the component's MOTION, so it ends where motion ends.

    Descending past a transform would fold static placement into the material
    map and put the pattern in the wrong frame.
    """
    inner = sdf_transform("translate", sdf_primitive("box", b=[1.0, 1.0, 1.0]), t=[1.0, 0.0, 0.0])
    tree = sdf_deform("twist", sdf_deform("bend", inner, k=0.1), k=0.2)
    assert [n["deform"] for n in _deform_prefix(tree)] == ["twist", "bend"]

    # A transform on top ends the prefix before it starts.
    assert _deform_prefix(sdf_transform("translate", tree, t=[0.0, 0.0, 1.0])) == []


def test_displace_is_not_a_point_warp():
    """``displace`` perturbs the DISTANCE, not the point, so it moves no
    material and must not enter a rest chain -- material coordinates through it
    would be a coordinate the geometry never occupied."""
    tree = sdf_deform(
        "displace",
        sdf_primitive("sphere", r=1.0),
        field={"type": "field", "kind": "radial", "params": {"freq": 1.0}},
    )
    assert _deform_prefix(tree) == []


# ---------------------------------------------------------------------------
# The emitted map
# ---------------------------------------------------------------------------


def test_a_static_part_gets_an_identity_rest_point_and_cuts_at_p():
    """No warp, no map. The static case must be byte-identical to a build with
    no rest maps at all, or every part in the tree pays for a feature only
    deforming parts use."""
    src = _emit(sdf_primitive("box", b=[1.0, 1.0, 1.0]))
    assert _fn(src, "sdm_rest_point").strip() == "return p;"
    assert "sdm_cut_plane(p, cn, co)" in _fn(src, "sdf_scene_rcut")
    assert not re.search(r"sdm_rest_\d", src), "a static part emitted a rest chain it cannot use"


@pytest.mark.parametrize(
    "warp",
    [
        _plate(),
        sdf_deform("twist", sdf_primitive("box", b=[1.0, 1.0, 2.0]), k=0.3),
        sdf_deform(
            "twist_radial",
            sdf_primitive("box", b=[2.0, 0.3, 1.0]),
            r0=0.0,
            r1=2.0,
            angle_inner=-ANGLE,
            angle_outer=ANGLE,
        ),
    ],
    ids=["twist_linear", "twist", "twist_radial"],
)
def test_a_warped_part_gets_a_real_map_and_cuts_in_it(warp):
    """The identity here is the whole reported bug, so it is what is asserted
    against -- for EVERY point-warp deform, because the last time one was added
    to the distance emitter and forgotten here, the map silently stayed ``p``.
    """
    src = _emit(warp)
    body = _fn(src, "sdm_rest_point").strip()
    assert body != "return p;", "the rest map is the identity on a warped part"
    m = re.search(r"return (sdm_rest_\d+)\(p\);", body)
    assert m, f"expected a rest chain call, got: {body}"
    assert f"sdm_cut_plane({m.group(1)}(p), cn, co)" in _fn(src, "sdf_scene_rcut"), (
        "the cutaway is not taken at the rest point"
    )


def test_the_rest_chain_applies_the_same_call_the_distance_emission_does():
    """ONE SPELLING, TWO READERS.

    The rest map and the child's query warp are the same map, so they must be
    the same text. Written out separately they drift, and the drift is silent:
    a deform present in one and missing from the other renders correctly and
    patterns wrongly.
    """
    src = _emit(_plate())
    call = "op_twist_linear(q, vec2(1.0, 0.0), -2.0, 2.0, -0.35, 0.35)"
    rest = _fn(src, re.search(r"vec3 (sdm_rest_\d+)\(", src).group(1))
    assert call in rest
    # The distance side spells the identical call on `p`.
    assert call.replace("(q,", "(p,") in src


def test_every_component_gets_its_own_map_and_static_ones_fall_through():
    """Per-component, because components deform independently.

    A union of a warped plate and a static post must give the plate a map and
    the post none -- one shared map would texture the post in the plate's
    frame, and one shared identity would lose the plate's motion.
    """
    tree = sdf_op(
        "union",
        [
            _plate(),
            sdf_transform("translate", sdf_primitive("box", b=[0.4, 0.4, 2.0]), t=[3.0, 0.0, 0.0]),
        ],
    )
    src = _emit(tree)
    body = _fn(src, "sdm_rest_point")
    guards = re.findall(r"if \(cid == (\d+)\) return (sdm_rest_\d+)\(p\);", body)
    assert len(guards) == 1, f"expected exactly one component to carry a map, got {guards}"
    assert body.rstrip().endswith("return p;"), "the static component must fall through to p"

    cut = _fn(src, "sdf_scene_rcut")
    assert f"sdm_cut_plane({guards[0][1]}(p), cn, co)" in cut
    assert "sdm_cut_plane(p, cn, co)" in cut, "the static component must cut at p"
