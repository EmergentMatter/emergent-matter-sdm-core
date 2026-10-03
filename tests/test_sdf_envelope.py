"""The envelope walk: the same solid with its holes filled in.

Pins the rewrite rules in :mod:`software_defined_matter.sdf.envelope`, which
supply ``relative_density``'s denominator. Each test states the rule it holds
and the shape it holds it on.
"""

from __future__ import annotations

import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    make_param_ref,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.sdf.envelope import (
    EnvelopeInferenceError,
    infer_sdf_envelope,
)


def _part(tree, **params):
    return Part(
        name="t",
        params=params,
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
    )


def _env(tree, **params):
    return infer_sdf_envelope(tree, _part(tree, **params))


# ---------------------------------------------------------------------------
# Leaves pass through
# ---------------------------------------------------------------------------


def test_solid_primitive_is_its_own_envelope():
    tree = sdf_primitive("sphere", r=5.0)
    assert _env(tree) == tree


def test_envelope_does_not_mutate_the_input():
    tree = sdf_op(
        "subtract", [sdf_primitive("box", b=[5.0, 5.0, 5.0]), sdf_primitive("sphere", r=2.0)]
    )
    before = repr(tree)
    env = infer_sdf_envelope(tree, _part(tree))
    assert repr(tree) == before
    # and the result is a distinct object, safe to edit
    env["params"]["b"] = [1.0, 1.0, 1.0]
    assert tree["children"][0]["params"]["b"] == [5.0, 5.0, 5.0]


# ---------------------------------------------------------------------------
# Material-removing nodes are dropped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["subtract", "smooth_subtract"])
def test_subtract_keeps_only_the_minuend(op):
    box = sdf_primitive("box", b=[10.0, 10.0, 10.0])
    hole = sdf_primitive("capped_cylinder", r=1.0, h=11.0)
    kwargs = {"k": 0.2} if op == "smooth_subtract" else {}
    assert _env(sdf_op(op, [box, hole], **kwargs)) == box


def test_onion_is_dropped_so_a_shell_measures_against_its_solid():
    ball = sdf_primitive("sphere", r=5.0)
    assert _env(sdf_modifier("onion", ball, thickness=0.5)) == ball


def test_nested_subtractions_all_count_as_porosity():
    # subtract(subtract(box, holes_a), holes_b) -> box
    box = sdf_primitive("box", b=[4.0, 4.0, 4.0])
    inner = sdf_op("subtract", [box, sdf_primitive("sphere", r=1.0)])
    outer = sdf_op("subtract", [inner, sdf_primitive("sphere", r=0.5)])
    assert _env(outer) == box


# ---------------------------------------------------------------------------
# Combining nodes recurse
# ---------------------------------------------------------------------------


def test_union_recurses_into_every_child():
    a = sdf_op("subtract", [sdf_primitive("sphere", r=5.0), sdf_primitive("sphere", r=4.0)])
    b = sdf_op(
        "subtract", [sdf_primitive("box", b=[2.0, 2.0, 2.0]), sdf_primitive("sphere", r=1.0)]
    )
    env = _env(sdf_op("union", [a, b]))
    assert env["op"] == "union"
    assert env["children"] == [
        sdf_primitive("sphere", r=5.0),
        sdf_primitive("box", b=[2.0, 2.0, 2.0]),
    ]


def test_intersect_recurses_and_keeps_the_op():
    env = _env(
        sdf_op(
            "intersect", [sdf_primitive("sphere", r=9.0), sdf_primitive("box", b=[5.0, 5.0, 5.0])]
        )
    )
    assert env["op"] == "intersect"
    assert len(env["children"]) == 2


def test_transform_wraps_the_child_envelope():
    tree = sdf_transform(
        "translate",
        sdf_op("subtract", [sdf_primitive("sphere", r=3.0), sdf_primitive("sphere", r=2.0)]),
        t=[1.0, 2.0, 3.0],
    )
    env = _env(tree)
    assert env["transform"] == "translate"
    assert env["params"]["t"] == [1.0, 2.0, 3.0]
    assert env["child"] == sdf_primitive("sphere", r=3.0)


# ---------------------------------------------------------------------------
# Lattices become their design domain
# ---------------------------------------------------------------------------


def test_tpms_becomes_its_clip_box():
    # A gyroid's pores run straight through, so it has no outer surface of its
    # own. Its envelope is the box it was clipped to: 0.5 * n_periods * period.
    env = _env(sdf_primitive("gyroid", period=5.0, min_thickness=1.0, n_periods=[4, 4, 4]))
    assert env["kind"] == "box"
    assert env["params"]["b"] == pytest.approx([10.0, 10.0, 10.0])


@pytest.mark.parametrize("kind", ["gyroid", "schwarz_p", "schwarz_d", "neovius", "lidinoid"])
def test_every_tpms_family_is_filled(kind):
    env = _env(sdf_primitive(kind, period=4.0, min_thickness=0.8, n_periods=[2, 3, 4]))
    assert env["kind"] == "box"
    assert env["params"]["b"] == pytest.approx([4.0, 6.0, 8.0])


def test_repeat_finite_becomes_its_tiling_domain():
    strut = sdf_primitive("box", b=[0.4, 0.4, 12.0])
    env = _env(sdf_transform("repeat_finite", strut, c=3.0, l=[3, 3, 0]))
    assert env["kind"] == "box"
    # child half-extent inflated by c * l on each axis
    assert env["params"]["b"] == pytest.approx([9.4, 9.4, 12.0])


def test_lattice_clipped_to_a_shape_yields_that_shape():
    # The point of filling lattice leaves: an intersect then clips the design
    # box back down to the real bound, so the density is the lattice's own
    # fill fraction rather than an artefact of the sampling box.
    tree = sdf_op(
        "intersect",
        [
            sdf_primitive("gyroid", period=5.0, min_thickness=1.0, n_periods=[5, 5, 5]),
            sdf_primitive("sphere", r=9.0),
        ],
    )
    env = _env(tree)
    assert env["op"] == "intersect"
    assert env["children"][0]["kind"] == "box"
    assert env["children"][1] == sdf_primitive("sphere", r=9.0)


def test_offset_lattice_domain_is_translated_back():
    strut = sdf_primitive("box", b=[0.5, 0.5, 1.0])
    tree = sdf_transform(
        "repeat_finite",
        sdf_transform("translate", strut, t=[10.0, 0.0, 0.0]),
        c=2.0,
        l=[1, 0, 0],
    )
    env = _env(tree)
    assert env["type"] == "transform"
    assert env["transform"] == "translate"
    assert env["params"]["t"] == pytest.approx([10.0, 0.0, 0.0])
    assert env["child"]["kind"] == "box"


# ---------------------------------------------------------------------------
# No finite envelope
# ---------------------------------------------------------------------------


def test_repeat_inf_raises_and_says_what_to_do():
    tree = sdf_transform("repeat_inf", sdf_primitive("sphere", r=1.0), c=[3.0, 3.0, 3.0])
    with pytest.raises(EnvelopeInferenceError) as exc:
        _env(tree)
    assert "repeat_inf" in str(exc.value)
    assert "intersect" in str(exc.value)


def test_unknown_node_type_raises():
    with pytest.raises(EnvelopeInferenceError):
        _env({"type": "no_such_node_kind", "child": sdf_primitive("sphere", r=1.0)})


# ---------------------------------------------------------------------------
# Params stay symbolic where they can
# ---------------------------------------------------------------------------


def test_param_refs_survive_the_walk():
    # Only lattice domains are frozen to numbers; ordinary geometry keeps its
    # $refs so the denominator stays differentiable.
    tree = sdf_op(
        "subtract",
        [
            sdf_primitive("sphere", r=make_param_ref("r_out")),
            sdf_primitive("sphere", r=make_param_ref("r_in")),
        ],
    )
    env = _env(
        tree,
        r_out=Param("r_out", 5.0, free=True, bounds=(1.0, 9.0)),
        r_in=Param("r_in", 4.0, free=True, bounds=(0.5, 8.0)),
    )
    assert env["params"]["r"] == make_param_ref("r_out")
