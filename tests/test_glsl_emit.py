"""Phase-1 string-assertion tests for the GLSL emitter.

These tests check that ``emit_glsl`` produces well-formed source for each
node kind: primitives, hard / smooth CSG ops, transforms (incl. 2-D /
3-D variants), modifiers, deforms (incl. ``displace`` with a field tree),
2-D to 3-D lifts, and ``$ref`` -> uniform plumbing.

Phase-2 (a separate PR) adds a headless-GL parity test against JAX. These
tests are deliberately lighter: they confirm the emitter writes valid-
looking GLSL strings without requiring a GPU in CI.
"""

from __future__ import annotations

import re

import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    field_primitive,
    make_param_ref,
    sdf_2d_to_3d,
    sdf_deform,
    sdf_loft,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_sweep,
    sdf_transform,
)
from software_defined_matter.glsl import GLSLEmission, emit_glsl, load_lib_glsl
from software_defined_matter.model import SDFTree
from software_defined_matter.sdf.validate import validate_document_semantics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap(tree, *, params=None, metadata=None) -> Part:
    """Wrap a raw SDF tree as a single-material Part with the given Params."""
    return Part(
        name="t",
        params={p.name: p for p in (params or [])},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
        metadata=metadata or {},
    )


def _emit(tree, *, params=None, metadata=None, **kw) -> GLSLEmission:
    return emit_glsl(_wrap(tree, params=params, metadata=metadata), **kw)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


def test_lib_glsl_contains_every_dispatch_entry():
    """Every primitive / op / transform / modifier / deform / field referenced
    by the emitter must have a matching GLSL function in lib.glsl. This is the
    contract that keeps the two compilers from drifting silently.
    """
    lib = load_lib_glsl()
    # Primitive functions
    from software_defined_matter.glsl.emit import _FIELD_SPECS, _PRIM_SPECS

    missing = []
    for kind in _PRIM_SPECS:
        if f"sdf_{kind}" not in lib:
            missing.append(f"sdf_{kind}")
    for kind in _FIELD_SPECS:
        if f"field_{kind}" not in lib:
            missing.append(f"field_{kind}")
    # Combinators / transforms / modifiers / deforms referenced by emit.py
    for fn in [
        "op_union",
        "op_subtract",
        "op_intersect",
        "op_smooth_union",
        "op_smooth_subtract",
        "op_smooth_intersect",
        "op_round",
        "op_onion",
        "tf_translate2",
        "tf_translate3",
        "tf_scale2",
        "tf_scale3",
        "tf_rotate_x",
        "tf_rotate_y",
        "tf_rotate_z",
        "tf_rotate_matrix",
        "op_repeat_finite2",
        "op_repeat_finite3",
        "op_elongate2",
        "op_elongate3",
        "op_twist",
        "op_bend",
        "lift_revolution_q",
        "lift_extrusion_finish",
    ]:
        if fn not in lib:
            missing.append(fn)
    assert not missing, f"lib.glsl is missing: {missing}"


# ---------------------------------------------------------------------------
# Single primitives
# ---------------------------------------------------------------------------


def test_sphere_emits_scalar_literal():
    em = _emit(
        sdf_primitive("sphere", r=2.0),
        metadata={"bbox": [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]},
    )
    assert "float sdf_scene(vec3 p)" in em.scene_source
    assert "sdf_sphere(p, 2.0)" in em.scene_source
    assert em.uniforms == []


def test_box_emits_vec3_literal():
    em = _emit(
        sdf_primitive("box", b=[1.0, 2.0, 3.0]),
        metadata={"bbox": [[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]},
    )
    assert "sdf_box(p, vec3(1.0, 2.0, 3.0))" in em.scene_source


def test_param_ref_becomes_uniform():
    em = _emit(
        sdf_primitive("sphere", r=make_param_ref("radius")),
        params=[Param(name="radius", value=1.0, free=True, bounds=(0.5, 3.0), unit="mm")],
        metadata={"bbox": [[-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]]},
    )
    assert len(em.uniforms) == 1
    u = em.uniforms[0]
    assert u.name == "u_p_radius"
    assert u.source_param == "radius"
    assert u.initial == 1.0
    assert u.bounds == (0.5, 3.0)
    assert u.unit == "mm"
    assert "uniform float u_p_radius;" in em.scene_source
    assert "sdf_sphere(p, u_p_radius)" in em.scene_source


def test_param_ref_dedup_when_used_twice():
    tree = sdf_op(
        "union",
        [
            sdf_primitive("sphere", r=make_param_ref("r")),
            sdf_primitive("sphere", r=make_param_ref("r")),
        ],
    )
    em = _emit(
        tree,
        params=[Param(name="r", value=1.0, free=True, bounds=(0.1, 2.0), unit="mm")],
        metadata={"bbox": [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]},
    )
    assert len(em.uniforms) == 1
    # One declaration only.
    assert em.scene_source.count("uniform float u_p_r;") == 1


# ---------------------------------------------------------------------------
# Hard / smooth CSG
# ---------------------------------------------------------------------------


def test_union_emits_op_union_fold():
    tree = sdf_op(
        "union",
        [
            sdf_primitive("sphere", r=1.0),
            sdf_primitive("sphere", r=2.0),
        ],
    )
    em = _emit(tree, metadata={"bbox": [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]})
    assert "op_union(" in em.scene_source
    assert "op_smooth_union(" not in em.scene_source


def test_smooth_csg_flag_promotes_hard_to_smooth():
    tree = sdf_op(
        "union",
        [sdf_primitive("sphere", r=1.0), sdf_primitive("sphere", r=1.0)],
    )
    em = _emit(
        tree,
        smooth_csg=True,
        smooth_k=0.5,
        metadata={"bbox": [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]},
    )
    assert "op_smooth_union(" in em.scene_source
    assert "0.5" in em.scene_source


def test_explicit_smooth_union_uses_k_kwarg():
    tree = sdf_op(
        "smooth_union",
        [sdf_primitive("sphere", r=1.0), sdf_primitive("sphere", r=1.0)],
        k=0.3,
    )
    em = _emit(tree, metadata={"bbox": [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]})
    assert "op_smooth_union(d, " in em.scene_source
    assert "0.3" in em.scene_source


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def test_translate_emits_tf_translate3_call():
    tree = sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[0.0, 0.0, 5.0])
    em = _emit(tree, metadata={"bbox": [[-1.0, -1.0, 4.0], [1.0, 1.0, 6.0]]})
    assert "tf_translate3(p, vec3(0.0, 0.0, 5.0))" in em.scene_source


def test_rotate_matrix_transposes_for_glsl_column_major():
    R = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    tree = sdf_transform("rotate_matrix", sdf_primitive("sphere", r=1.0), R=R)
    em = _emit(tree, metadata={"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]})
    # GLSL column-major: column0 = (R[0][0], R[1][0], R[2][0]) = (1, 4, 7).
    assert "mat3(vec3(1.0, 4.0, 7.0), vec3(2.0, 5.0, 8.0), vec3(3.0, 6.0, 9.0))" in em.scene_source


def test_scale_emits_distance_correction():
    tree = sdf_transform("scale", sdf_primitive("sphere", r=1.0), s=2.0)
    em = _emit(tree, metadata={"bbox": [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]})
    # child(p/s, ...) * s
    assert "tf_scale3(p, 2.0)" in em.scene_source
    assert ") * 2.0;" in em.scene_source


def test_canonical_sector_fold_evaluates_the_child_at_both_neighbours():
    tree = sdf_transform("canonical_sector_fold", sdf_primitive("sphere", r=1.0), n_sectors=6.0)
    em = _emit(tree, metadata={"bbox": [[-3, -3, -3], [3, 3, 3]]})
    src = em.scene_source
    assert "float n_sec = floor(6.0);" in src
    assert "tf_canonical_sector_fold(p, n_sec)" in src
    # The three calls compile.py makes. A query near a wedge seam is closer to
    # the copy next door, so folding alone reports too large a distance.
    assert "tf_rotate_z(q, sec)" in src
    assert "tf_rotate_z(q, -sec)" in src


def test_centered_fold_emits_the_centered_helper_and_its_phase():
    tree = sdf_transform(
        "canonical_sector_fold",
        sdf_primitive("sphere", r=1.0),
        n_sectors=6.0,
        centered=True,
        phase_frac=0.5,
    )
    em = _emit(tree, metadata={"bbox": [[-3, -3, -3], [3, 3, 3]]})
    assert "float n_sec = floor(6.0);" in em.scene_source
    assert "tf_canonical_sector_fold_c(p, n_sec, 0.5)" in em.scene_source


def test_a_fractional_sector_count_is_truncated_like_compile_py():
    """`n_sectors` is a count, and compile.py takes `int()` of it.

    A `$ref` count is scrubbable, so a slider between two integers reaches the
    shader as 4.7 while the mesh pipeline is already folding into 4 wedges. Both
    the fold and the sector width read the floored value, so a half-scrubbed
    count cannot show a preview that no mesh corresponds to.
    """
    tree = sdf_transform(
        "canonical_sector_fold",
        sdf_primitive("sphere", r=1.0),
        n_sectors=make_param_ref("foil_count"),
    )
    em = _emit(
        tree,
        params=[Param(name="foil_count", value=6.0, free=True, bounds=(3.0, 15.0), unit="count")],
        metadata={"bbox": [[-9, -9, -9], [9, 9, 9]]},
    )
    src = em.scene_source
    assert "float n_sec = floor(u_p_foil_count);" in src
    # The raw uniform must not reach either place the count is used, or the fold
    # and the sector width disagree by the fractional part.
    assert "tf_canonical_sector_fold(p, n_sec)" in src
    assert "6.28318530717958647692 / n_sec;" in src
    assert "tf_canonical_sector_fold(p, u_p_foil_count)" not in src
    assert "/ u_p_foil_count" not in src


def test_mirror_unions_the_child_with_its_reflection():
    tree = sdf_transform(
        "mirror", sdf_primitive("sphere", r=1.0), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0]
    )
    em = _emit(tree, metadata={"bbox": [[-2, -2, -2], [2, 2, 2]]})
    src = em.scene_source
    assert "tf_reflect_plane(p, vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0))" in src
    assert "op_union(" in src


def test_mirror_follows_the_smooth_csg_flag():
    """compile.py promotes this union under smooth_csg, so the emitter must."""
    tree = sdf_transform(
        "mirror", sdf_primitive("sphere", r=1.0), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0]
    )
    em = _emit(
        tree,
        metadata={"bbox": [[-2, -2, -2], [2, 2, 2]], "smooth_csg": True},
    )
    assert "op_smooth_union(" in em.scene_source


# One tree per transform, so the coverage test below can actually emit each.
# A transform added to wire.py without an entry here fails that test, which is
# the point: `canonical_sector_fold` and `mirror` were both compilable and
# un-emittable for months because nothing compared the two lists.
_TRANSFORM_TREES = {
    "translate": lambda c: sdf_transform("translate", c, t=[0.0, 0.0, 1.0]),
    "scale": lambda c: sdf_transform("scale", c, s=2.0),
    "scale_axis": lambda c: sdf_transform("scale_axis", c, s=[2.0, 1.0, 0.5]),
    "rotate_x": lambda c: sdf_transform("rotate_x", c, angle=0.5),
    "rotate_y": lambda c: sdf_transform("rotate_y", c, angle=0.5),
    "rotate_z": lambda c: sdf_transform("rotate_z", c, angle=0.5),
    "rotate_matrix": lambda c: sdf_transform(
        "rotate_matrix", c, R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ),
    # `c` collides with the child argument's name, hence the dict.
    "repeat_finite": lambda c: sdf_transform(
        "repeat_finite", c, **{"c": 4.0, "l": [1.0, 1.0, 1.0]}
    ),
    "canonical_sector_fold": lambda c: sdf_transform("canonical_sector_fold", c, n_sectors=6.0),
    "mirror": lambda c: sdf_transform("mirror", c, n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0]),
}

# `repeat_inf` is schema-legal and deliberately never compiled: a shape that
# tiles forever has no finite closure to trace. See wire.py.
_NOT_EMITTED = {"repeat_inf"}


def test_every_declared_transform_is_emittable():
    """The emitter must keep up with the wire contract.

    `glsl/emit.py`'s own docstring says every kind the JAX compiler handles
    needs an entry here plus a function in lib.glsl. Nothing checked it, and
    two transforms drifted out of the emitter as a result.
    """
    from software_defined_matter.wire import TRANSFORMS

    declared = set(TRANSFORMS) - _NOT_EMITTED
    assert set(_TRANSFORM_TREES) == declared, (
        "add the new transform to _TRANSFORM_TREES (or to _NOT_EMITTED with a reason)"
    )

    for name, build in sorted(_TRANSFORM_TREES.items()):
        tree = build(sdf_primitive("sphere", r=1.0))
        em = _emit(tree, metadata={"bbox": [[-9, -9, -9], [9, 9, 9]]})
        assert "float sdf_scene(vec3 p)" in em.scene_source, name


# GLSL builtins and type constructors, which need no definition anywhere, plus
# the keywords a parenthesised clause follows (a sweep's segment loop). Being
# generous here costs nothing: every helper in lib.glsl is named `sdf_`, `op_`,
# `tf_`, `field_`, `lift_` or `sdm_`, so no builtin name can mask a missing one.
# fmt: off
_GLSL_BUILTINS = frozenset({
    "abs", "acos", "asin", "atan", "bool", "ceil", "clamp", "cos", "cross",
    "distance", "dot", "exp", "float", "floor", "fract", "int", "inversesqrt",
    "length", "log", "mat2", "mat3", "mat4", "max", "min", "mix", "mod",
    "normalize", "pow", "reflect", "round", "sign", "sin", "smoothstep",
    "sqrt", "step", "tan", "vec2", "vec3", "vec4",
    "for", "if", "while",
})
# fmt: on


def _strip_glsl_comments(src: str) -> str:
    """Remove GLSL comments so regex scans do not match inside them."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", src)


def _glsl_calls(src: str) -> set[str]:
    """Every name the source calls, i.e. an identifier followed by ``(``."""
    return set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", _strip_glsl_comments(src)))


def _glsl_definitions(src: str) -> set[str]:
    """Every function the source defines, by its return type and name."""
    return set(
        re.findall(
            r"^\s*(?:float|vec2|vec3|vec4|mat3|bool|int|void)\s+(\w+)\s*\(",
            _strip_glsl_comments(src),
            re.MULTILINE,
        )
    )


# An open polyline with a 3-D bend; the sweep fixture the tests below share.
def _bent_path() -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 2.0]]


def _bent_sweep(**kwargs) -> SDFTree:
    return sdf_sweep(
        sdf_primitive("circle_2d", r=0.4), _bent_path(), path_kind="polyline", **kwargs
    )


def _emit_transform_corpus():
    """One emitted scene per transform, plus the branches a bare tree misses.

    `_TRANSFORM_TREES` holds exactly one tree per transform and is 3-D
    throughout, so two GLSL helpers are unreachable from it: `tf_reflect_plane2`
    needs a `mirror` in a 2-D subtree, which only a `2d_to_3d` lift produces,
    and `tf_canonical_sector_fold_c` needs `centered`.
    """
    cases = [
        pytest.param(build(sdf_primitive("sphere", r=1.0)), id=name)
        for name, build in sorted(_TRANSFORM_TREES.items())
    ]
    cases.append(
        pytest.param(
            sdf_2d_to_3d(
                "extrusion",
                sdf_transform(
                    "mirror", sdf_primitive("circle_2d", r=0.5), n=[1.0, 0.0], o=[0.0, 0.0]
                ),
                h=1.0,
            ),
            id="mirror-2d",
        )
    )
    cases.append(
        pytest.param(
            sdf_transform(
                "canonical_sector_fold",
                sdf_primitive("sphere", r=1.0),
                n_sectors=6.0,
                centered=True,
                phase_frac=0.25,
            ),
            id="canonical_sector_fold-centered",
        )
    )
    # Not a transform, but the same two checks apply: a sweep has its own
    # emitter branch and its own lib.glsl helper, and a tree the emitter
    # compiles has to be a tree the wire contract admits.
    cases.append(pytest.param(_bent_sweep(), id="sweep"))
    return cases


@pytest.mark.parametrize("tree", _emit_transform_corpus())
def test_every_function_the_emitter_calls_is_defined_in_lib_glsl(tree):
    """A transform needs an emitter branch AND a GLSL function, and only the
    branch is checked above.

    `test_lib_glsl_contains_every_dispatch_entry` names its helpers by hand, and
    the four this PR adds were not on that list, so deleting them from lib.glsl
    would have left CI green and the shader failing to compile. Reading the call
    names off the emitted source instead means a new transform is covered the
    moment it enters `_TRANSFORM_TREES`, with nothing to remember to update.
    """
    src = _emit(tree, metadata={"bbox": [[-9, -9, -9], [9, 9, 9]]}).scene_source
    undefined = _glsl_calls(src) - _glsl_definitions(src) - _glsl_definitions(load_lib_glsl())
    assert not undefined - _GLSL_BUILTINS, (
        f"emitted source calls {sorted(undefined - _GLSL_BUILTINS)}, which lib.glsl does not define"
    )


@pytest.mark.parametrize("tree", _emit_transform_corpus())
def test_every_emitter_corpus_tree_is_semantically_valid(tree):
    """The emitter corpus and the wire contract must describe the same language.

    They were allowed to drift: the `mirror-2d` case above passes `n=[1.0, 0.0]`
    because the emitter (and `_check_axis_vector` on the JAX side) sizes a
    per-axis vector by the subtree's dimension, while `wire.TRANSFORMS` declared
    it `vec3` -- so this corpus emitted a tree `validate_document_semantics`
    rejected outright, and nothing compared the two. A tree the emitter is
    expected to compile must be a tree the contract admits.
    """
    validate_document_semantics(_wrap(tree).to_dict())  # must not raise


# ---------------------------------------------------------------------------
# Modifiers / deforms
# ---------------------------------------------------------------------------


def test_round_modifier():
    tree = sdf_modifier("round", sdf_primitive("box", b=[1.0, 1.0, 1.0]), r=0.1)
    em = _emit(tree, metadata={"bbox": [[-1.1, -1.1, -1.1], [1.1, 1.1, 1.1]]})
    assert "op_round(" in em.scene_source
    assert "0.1" in em.scene_source


def test_twist_deform():
    tree = sdf_deform("twist", sdf_primitive("box", b=[1.0, 1.0, 1.0]), k=0.5)
    em = _emit(tree, metadata={"bbox": [[-2.0, -2.0, -1.0], [2.0, 2.0, 1.0]]})
    assert "op_twist(p, 0.5)" in em.scene_source


def test_displace_with_field_tree():
    tree = sdf_deform(
        "displace",
        sdf_primitive("sphere", r=1.0),
        field=field_primitive("radial", freq=0.5, amplitude=0.1, phase=0.0),
    )
    em = _emit(tree, metadata={"bbox": [[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]})
    assert "field_radial(p, 0.5, 0.1, 0.0)" in em.scene_source
    # The displace adds child + field, as in op_displace.
    assert "+ field_n" in em.scene_source


def test_field_defaults_do_not_mask_authored_arguments():
    tree = sdf_deform(
        "displace",
        sdf_primitive("sphere", r=1.0),
        field=field_primitive("sin_xyz", freq=[0.2, 0.3, 0.4], amplitude=0.175),
    )
    em = _emit(tree, metadata={"bbox": [[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]})
    assert "field_sin_xyz(p, vec3(0.2, 0.3, 0.4), 0.175, vec3(0.0, 0.0, 0.0))" in em.scene_source


def test_unconsumed_field_argument_is_rejected():
    tree = sdf_deform(
        "displace",
        sdf_primitive("sphere", r=1.0),
        field=field_primitive("radial", freq=0.5, amp=0.1),
    )
    with pytest.raises(KeyError, match=r"does not accept param\(s\) \['amp'\]"):
        _emit(tree, metadata={"bbox": [[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]})


def test_angular_field_emits_with_defaults():
    tree = sdf_deform(
        "displace",
        sdf_primitive("sphere", r=1.0),
        field=field_primitive("angular", freq=6.0),
    )
    em = _emit(tree, metadata={"bbox": [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]})
    assert "field_angular(p, 6.0, 1.0, 0.0)" in em.scene_source


# ---------------------------------------------------------------------------
# 2D -> 3D lifts
# ---------------------------------------------------------------------------


def test_revolution_lifts_2d_subtree():
    tree = sdf_2d_to_3d(
        "revolution",
        sdf_primitive("circle_2d", r=0.5),
        offset=2.0,
    )
    em = _emit(tree, metadata={"bbox": [[-3.0, -3.0, -1.0], [3.0, 3.0, 1.0]]})
    # The lift uses lift_revolution_q to reshuffle p into vec2 q.
    assert "lift_revolution_q(p, 2.0)" in em.scene_source
    # The 2D child function is declared with vec2 input.
    assert "sdf_circle_2d(p, 0.5)" in em.scene_source


def test_extrusion_lifts_2d_subtree():
    tree = sdf_2d_to_3d(
        "extrusion",
        sdf_primitive("box_2d", b=[1.0, 2.0]),
        h=3.0,
    )
    em = _emit(tree, metadata={"bbox": [[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]})
    assert "sdf_box_2d(p, vec2(1.0, 2.0))" in em.scene_source
    assert "lift_extrusion_finish(d, p, 3.0)" in em.scene_source


# ---------------------------------------------------------------------------
# polygon_2d
# ---------------------------------------------------------------------------

_TRIANGLE = [[-1.0, -0.5], [1.0, -0.5], [0.0, 1.0]]


def test_polygon_2d_extrusion_emits_helper_and_define():
    tree = sdf_2d_to_3d(
        "extrusion",
        sdf_primitive("polygon_2d", vertices=_TRIANGLE),
        h=1.0,
    )
    em = _emit(tree, metadata={"bbox": [[-1.0, -0.5, -1.0], [1.0, 1.0, 1.0]]})
    assert "#define SDM_POLY_MAX_N 3" in em.lib_source
    assert "sdf_polygon_2d" in em.lib_source
    assert "vec2 V[SDM_POLY_MAX_N] = vec2[SDM_POLY_MAX_N](" in em.scene_source
    assert "return sdf_polygon_2d(p, V, 3);" in em.scene_source
    assert "vec2(-1.0, -0.5)" in em.scene_source
    assert "vec2(1.0, -0.5)" in em.scene_source
    assert "vec2(0.0, 1.0)" in em.scene_source


def test_polygon_2d_duplicate_closing_vertex_emits():
    """Duplicate first==last is accepted by Python as a no-op; the emitter must
    pass the closing vertex through and size N accordingly. The GLSL helper's
    zero-length edge guard (ee > 1e-12) keeps raymarching free of NaNs.
    """
    closed = _TRIANGLE + [_TRIANGLE[0]]  # 4 verts, last == first
    tree = sdf_2d_to_3d(
        "extrusion",
        sdf_primitive("polygon_2d", vertices=closed),
        h=1.0,
    )
    em = _emit(tree, metadata={"bbox": [[-1.0, -0.5, -1.0], [1.0, 1.0, 1.0]]})
    assert "#define SDM_POLY_MAX_N 4" in em.lib_source
    assert "return sdf_polygon_2d(p, V, 4);" in em.scene_source
    # Closing vertex appears twice in the initializer (first and last).
    assert em.scene_source.count("vec2(-1.0, -0.5)") == 2
    # lib.glsl must keep the zero-length edge guard that makes this safe.
    assert "ee > 1e-12" in em.lib_source


def test_polygon_2d_param_ref_coords_become_uniforms():
    """Numeric vertex coords bake to literals; $ref coords stay live uniforms
    so a host can reshape the polygon without re-emitting."""
    verts = [
        [-1.0, -0.5],
        [1.0, -0.5],
        [make_param_ref("tip_x"), make_param_ref("tip_y")],
    ]
    tree = sdf_2d_to_3d(
        "extrusion",
        sdf_primitive("polygon_2d", vertices=verts),
        h=1.0,
    )
    em = _emit(
        tree,
        params=[
            Param(name="tip_x", value=0.0, free=True, bounds=(-2.0, 2.0), unit="mm"),
            Param(name="tip_y", value=1.0, free=True, bounds=(0.0, 3.0), unit="mm"),
        ],
        metadata={"bbox": [[-1.0, -0.5, -1.0], [1.0, 1.0, 1.0]]},
    )
    assert {u.name for u in em.uniforms} == {"u_p_tip_x", "u_p_tip_y"}
    assert "uniform float u_p_tip_x;" in em.scene_source
    assert "uniform float u_p_tip_y;" in em.scene_source
    assert "vec2(u_p_tip_x, u_p_tip_y)" in em.scene_source
    # Fixed corners remain literals (not frozen param values).
    assert "vec2(-1.0, -0.5)" in em.scene_source
    assert "vec2(1.0, -0.5)" in em.scene_source


def test_polygon_2d_pads_shorter_polygon_to_scene_max():
    """When two polygons share a scene, every V[] is sized to the largest N
    and shorter ones pad with vec2(0.0)."""
    tri = sdf_2d_to_3d(
        "extrusion",
        sdf_primitive("polygon_2d", vertices=_TRIANGLE),
        h=1.0,
    )
    quad = sdf_2d_to_3d(
        "extrusion",
        sdf_primitive(
            "polygon_2d",
            vertices=[[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]],
        ),
        h=1.0,
    )
    tree = sdf_op("union", [tri, quad])
    em = _emit(tree, metadata={"bbox": [[-2.0, -2.0, -1.0], [2.0, 2.0, 1.0]]})
    assert "#define SDM_POLY_MAX_N 4" in em.lib_source
    assert "return sdf_polygon_2d(p, V, 3);" in em.scene_source
    assert "return sdf_polygon_2d(p, V, 4);" in em.scene_source
    # Triangle initializer pads one slot to reach SDM_POLY_MAX_N == 4.
    assert "vec2(0.0, 1.0), vec2(0.0)" in em.scene_source


def test_polygon_2d_rejects_too_few_vertices():
    tree = sdf_2d_to_3d(
        "extrusion",
        sdf_primitive("polygon_2d", vertices=[[0.0, 0.0], [1.0, 0.0]]),
        h=1.0,
    )
    with pytest.raises(ValueError, match=">=3"):
        _emit(tree, metadata={"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]})


# ---------------------------------------------------------------------------
# End-to-end: the example part
# ---------------------------------------------------------------------------


def test_example_part_emits(example_part):
    """The headline hollow_cylinder_with_hinge part must emit without errors
    and produce a sdf_scene + uniforms for the three free Params."""
    em = emit_glsl(example_part)
    assert "float sdf_scene(vec3 p)" in em.scene_source
    uniform_names = {u.name for u in em.uniforms}
    assert "u_p_outer_radius" in uniform_names
    assert "u_p_inner_radius" in uniform_names
    assert "u_p_notch_radius" in uniform_names
    # bbox is finite (gyroid is bounded by n_periods).
    (xlo, ylo, zlo), (xhi, yhi, zhi) = em.bbox
    assert xlo < xhi and ylo < yhi and zlo < zhi


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_part_without_materials_errors():
    part = Part(name="empty")
    with pytest.raises(ValueError, match="no materials"):
        emit_glsl(part)


def test_expression_tree_leaf_compiles_to_arithmetic():
    """This used to raise, and the refusal was pinned here as deliberate scope.

    An expression leaf now compiles (see `test_glsl_expr.py` for the operator
    table and its parity with `dsl.expr`). Asserting on the emitted arithmetic
    rather than on "it did not raise": binding a uniform to the param's cached
    value would also not raise, and would render a stale number.
    """
    tree = sdf_primitive(
        "sphere",
        r={"type": "binop", "op": "+", "lhs": {"type": "param", "name": "x"}, "rhs": 1.0},
    )
    part = _wrap(tree, params=[Param(name="x", value=1.0, free=True, bounds=(0.0, 2.0), unit="mm")])
    em = emit_glsl(part)
    assert "(u_p_x + 1.0)" in em.scene_source
    assert "u_p_x" in {u.name for u in em.uniforms}


def test_bad_primitive_kind_raises():
    tree = sdf_primitive("not_a_real_kind", r=1.0)
    with pytest.raises(ValueError, match="Unknown primitive"):
        emit_glsl(_wrap(tree, metadata={"bbox": [[-1, -1, -1], [1, 1, 1]]}))


def test_unknown_param_ref_raises():
    tree = sdf_primitive("sphere", r=make_param_ref("not_a_real_param"))
    with pytest.raises(KeyError, match="does not match"):
        emit_glsl(_wrap(tree, metadata={"bbox": [[-1, -1, -1], [1, 1, 1]]}))


# ---------------------------------------------------------------------------
# The emission contract: what a host gets besides the source
# ---------------------------------------------------------------------------
#
# A host cannot build a parameter panel from `uniforms` alone. That list holds
# only the params the walk happened to bind, so every param a viewer must NOT
# offer as a live slider is simply absent from it, indistinguishable from a
# param that does not exist. `controls` is the manifest that says what each one
# costs, and the classification is computed here rather than authored because
# only the emitter knows what it bound.


def test_controls_cover_every_param_exactly_once():
    """A missing entry is a param with no widget; a duplicate is two widgets
    over one value, which is the graph-pane rule's other failure direction."""
    params = [
        Param(name="r", value=1.0, unit="mm"),
        Param(name="h", value=2.0, unit="mm"),
        Param(name="unused", value=3.0, unit="mm"),
    ]
    em = _emit(
        sdf_primitive("capped_cylinder", h=make_param_ref("h"), r=make_param_ref("r")),
        params=params,
    )
    names = [c["param"] for c in em.controls]
    assert names == ["r", "h", "unused"]  # declaration order, not uniform order


def test_a_bound_param_is_live_and_names_its_own_uniform():
    """`live` is a fact about THIS emission, and the name is the emitter's to
    spell; a viewer that re-derived the mangling would drift from it."""
    em = _emit(
        sdf_primitive("sphere", r=make_param_ref("radius")),
        params=[Param(name="radius", value=1.0, unit="mm", bounds=(0.5, 2.0))],
    )
    (c,) = em.controls
    assert c["class"] == "live"
    assert c["uniform"] == "u_p_radius"
    assert c["uniform"] in {u.name for u in em.uniforms}
    assert c["value"] == 1.0
    assert c["bounds"] == [0.5, 2.0]
    assert c["unit"] == "mm"


def test_an_unbound_param_is_re_emit_and_has_no_uniform_key():
    """Absent, not null: `uniform` present-but-None reads as a uniform whose
    name failed to resolve, which is a different and much worse claim."""
    em = _emit(
        sdf_primitive("sphere", r=1.0),
        params=[Param(name="spare", value=7.0, unit="mm")],
    )
    (c,) = em.controls
    assert c["class"] == "re-emit"
    assert "uniform" not in c


def test_topology_is_the_one_class_that_is_authored():
    """`ui.role == "topology"` says the authoring script emits a different
    NUMBER of nodes, which the stamped-out tree cannot show. Everything else is
    inferred, so this is the only place authoring can override the emitter."""
    em = _emit(
        sdf_primitive("sphere", r=1.0),
        params=[Param(name="n_teeth", value=22.0, unit="count", ui={"role": "topology"})],
    )
    (c,) = em.controls
    assert c["class"] == "topology"


def test_binding_beats_the_topology_role():
    """A param the walk bound IS scrubbable, whatever the author labelled it.
    Reporting it as topology would send every scrub through a re-emit for a
    value the shader already reads from a uniform."""
    em = _emit(
        sdf_primitive("sphere", r=make_param_ref("r")),
        params=[Param(name="r", value=1.0, ui={"role": "topology"})],
    )
    (c,) = em.controls
    assert c["class"] == "live"


def test_ui_hints_are_copied_not_shared():
    """A caller mutating a control must not reach back into the Part."""
    prm = Param(name="r", value=1.0, ui={"step": 0.25, "group": "sun"})
    em = _emit(sdf_primitive("sphere", r=make_param_ref("r")), params=[prm])
    (c,) = em.controls
    assert c["ui"] == {"step": 0.25, "group": "sun"}
    c["ui"]["step"] = 999
    assert prm.ui["step"] == 0.25


def test_poly_max_n_is_reported_and_agrees_with_the_define():
    """The host needs this number to size `sdf_polygon_2d`'s array parameter
    when it reassembles the library out of per-leaf emissions. It is in the
    source as a #define; handing it over as a field is what stops the host
    parsing it back out of source it was just given."""
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-0.5, 0.5]]
    em = _emit(sdf_2d_to_3d("extrusion", sdf_primitive("polygon_2d", vertices=verts), h=1.0))
    assert em.poly_max_n == 5
    assert re.search(r"^#define SDM_POLY_MAX_N 5$", em.lib_source, re.M)


def test_a_scene_with_no_polygons_reports_zero_and_prepends_no_override():
    """lib.glsl carries its own `#ifndef SDM_POLY_MAX_N / #define 3` fallback,
    so the TOKEN is always present and its presence proves nothing. What the
    emitter controls is the PREPENDED override, and with no polygons in the
    scene there must not be one: the library is handed over unmodified."""
    em = _emit(sdf_primitive("sphere", r=1.0))
    assert em.poly_max_n == 0
    assert not em.lib_source.startswith("#define SDM_POLY_MAX_N")
    assert em.lib_source == load_lib_glsl()


def test_poly_table_is_empty_because_outlines_are_baked_into_the_source():
    """Empty means "const mode", NOT "no polygons": the distinction the
    `poly_max_n` field above exists to carry. A host that read emptiness as
    "this scene has no outlines" would size its array parameter to zero."""
    verts = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    em = _emit(sdf_2d_to_3d("extrusion", sdf_primitive("polygon_2d", vertices=verts), h=1.0))
    assert em.poly_table == []
    assert em.poly_max_n == 3
    assert "vec2[" in em.scene_source  # the outline really is inline


def test_on_stage_names_each_stage_in_order():
    """A whole-mechanism part spends minutes in `bbox` alone. Without this a
    host can only show a spinner, and a spinner cannot say which phase stalled."""
    seen: list[str] = []
    part = _wrap(sdf_primitive("sphere", r=1.0))
    emit_glsl(part, on_stage=seen.append)
    assert seen == ["envelope", "glsl-walk", "bbox"]


def test_on_stage_skips_envelope_when_the_tree_is_given():
    """The stage list describes work actually done, so a caller passing its own
    tree must not be told an envelope was computed."""
    seen: list[str] = []
    tree = sdf_primitive("sphere", r=1.0)
    emit_glsl(_wrap(tree), tree, on_stage=seen.append)
    assert seen == ["glsl-walk", "bbox"]


def test_a_failing_on_stage_cannot_fail_the_emit():
    """A status display is not allowed to take the geometry down with it."""

    def boom(_stage: str) -> None:
        raise RuntimeError("the status strip is broken")

    em = emit_glsl(_wrap(sdf_primitive("sphere", r=1.0)), on_stage=boom)
    assert em.scene_source  # the emit completed anyway


def test_omitting_on_stage_is_the_default_and_emits_the_same_source():
    """The callback is observation only; it must not change the artifact."""
    tree = sdf_primitive("sphere", r=make_param_ref("r"))
    params = [Param(name="r", value=1.0)]
    quiet = _emit(tree, params=params)
    watched = _emit(tree, params=params, on_stage=lambda _s: None)
    assert quiet.scene_source == watched.scene_source
    assert quiet.controls == watched.controls


# ---------------------------------------------------------------------------
# Component segmentation
# ---------------------------------------------------------------------------
#
# A component id indexes a host's colour table, selection mask and visibility
# mask, and is what a pick resolves to. So the id the SHADER computes must be
# the id the `components` list names, and both are produced by walking the same
# tree in the same order. Every test below is really about that agreement.


def _comp_labels(em: GLSLEmission) -> list[str]:
    return [c["label"] for c in em.components]


def test_a_union_root_becomes_one_component_per_child():
    em = _emit(
        sdf_op("union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])]),
        metadata={"components": ["ball", "brick"]},
    )
    assert _comp_labels(em) == ["ball", "brick"]
    assert [c["id"] for c in em.components] == [0, 1]
    # `machine` must be a function the emission actually defines: it is the
    # anchor a host rewrites a call site on, and a name that is not there
    # cannot be found, or worse, matches something else.
    for c in em.components:
        assert f"float {c['machine']}(" in em.scene_source


def test_a_child_union_expands_one_level_and_extends_the_label():
    em = _emit(
        sdf_op(
            "union",
            [
                sdf_primitive("sphere", r=1.0),
                sdf_op(
                    "union", [sdf_primitive("box", b=[1, 1, 1]), sdf_primitive("torus", t=[2, 0.5])]
                ),
            ],
        ),
        metadata={"components": ["ball", "pair"]},
    )
    assert _comp_labels(em) == ["ball", "pair/0", "pair/1"]


def test_expansion_stops_after_one_level():
    """Two levels would keep splitting until every leaf is its own component,
    which is not a part's structure; it is its parse tree."""
    inner = sdf_op("union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])])
    em = _emit(
        sdf_op(
            "union",
            [
                sdf_op("union", [inner, sdf_primitive("torus", t=[2, 0.5])]),
                sdf_primitive("torus", t=[3, 0.5]),
            ],
        ),
        metadata={"components": ["deep", "floor"]},
    )
    assert _comp_labels(em) == ["deep/0", "deep/1", "floor"]


def test_a_transform_wrapper_distributes_over_the_union_it_wraps():
    """`T(union(a, b)) == union(T(a), T(b))` for a pointwise query reshuffle,
    so a posed limb's parts get their own ids instead of sharing one."""
    em = _emit(
        sdf_op(
            "union",
            [
                sdf_transform(
                    "translate",
                    sdf_op(
                        "union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])]
                    ),
                    t=[5.0, 0.0, 0.0],
                ),
                sdf_primitive("torus", t=[2, 0.5]),
            ],
        ),
        metadata={"components": ["limb", "hub"]},
    )
    assert _comp_labels(em) == ["limb/0", "limb/1", "hub"]
    # Each expanded component must still carry the translate, or it is the
    # geometry from before the pose.
    assert em.scene_source.count("tf_translate3") >= 2


def test_a_modifier_does_not_distribute_and_stops_the_descent():
    """`onion(union(a, b))` is not `union(onion(a), onion(b))`: the shell of a
    union is not the union of the shells: the seam between a and b is interior
    to the union and has no shell there."""
    em = _emit(
        sdf_op(
            "union",
            [
                sdf_modifier(
                    "onion",
                    sdf_op(
                        "union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])]
                    ),
                    thickness=0.1,
                ),
                sdf_primitive("torus", t=[2, 0.5]),
            ],
        ),
        metadata={"components": ["shell", "hub"]},
    )
    assert _comp_labels(em) == ["shell", "hub"]


def test_smooth_csg_is_not_segmentable():
    """A smooth union BLENDS its children, so no point belongs to exactly one
    of them and "which component owns p" has no answer to give."""
    em = _emit(
        sdf_op("union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])]),
        metadata={"smooth_csg": True, "components": ["a", "b"]},
    )
    assert em.components == []
    assert "int sdf_scene_comp(vec3 p) { return 0; }" in em.scene_source


def test_a_non_union_root_is_one_component_and_says_so_as_zero():
    """Empty `components` means ONE component, not "unknown", so the shader
    answers 0, which is the id of the only thing there."""
    em = _emit(sdf_primitive("sphere", r=1.0))
    assert em.components == []
    assert "int sdf_scene_comp(vec3 p) { return 0; }" in em.scene_source
    assert "float sdf_scene_rcut(vec3 p, vec3 cn, float co)" in em.scene_source


def test_missing_labels_fall_back_to_a_positional_name():
    em = _emit(sdf_op("union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])]))
    assert _comp_labels(em) == ["component_0", "component_1"]


def test_expansion_is_skipped_not_truncated_past_the_cap():
    """Truncating would number the components differently from a host walking
    the same tree, and an id that means two things is worse than a coarse id."""
    from software_defined_matter.glsl.emit import _MAX_COMPONENTS

    big = sdf_op(
        "union", [sdf_primitive("sphere", r=float(i + 1)) for i in range(_MAX_COMPONENTS + 1)]
    )
    em = _emit(
        sdf_op("union", [big, sdf_primitive("box", b=[1, 1, 1])]),
        metadata={"components": ["many", "one"]},
    )
    # The over-budget child stays whole rather than contributing 65 of 66.
    assert _comp_labels(em) == ["many", "one"]


def test_sdf_scene_reaches_the_components_through_sdf_scene_rcut():
    """Not a stylistic choice: it is what makes hiding a component possible.

    A host masks a hidden component by rewriting its call site INSIDE the
    emitted field, anchored on the per-component call in `sdf_scene_rcut`,
    because nothing painted over a finished frame can make a pick through the
    hidden space answer with what is actually behind it. If `sdf_scene` reached
    the components by its own separate path, the march would read the unmasked
    fold and hiding would change nothing on screen, measured exactly that way:
    `test_part_visibility` reported "hiding no0 changed no texel".
    """
    tree = sdf_op("union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])])
    em = _emit(tree)
    assert "sdf_scene_rcut(p, vec3(0.0, 0.0, 1.0), 3.0e38)" in em.scene_source
    # and the root fold must not ALSO be reachable from sdf_scene directly
    body = em.scene_source.split("float sdf_scene(vec3 p)")[1]
    assert "sdf_n" not in body.split("}")[0]


def test_the_delegation_does_not_change_the_field_value():
    """The call graph moved; the numbers must not. `max(child, p.z - 3e38)` is
    `child` for any point a part occupies, so pushing the plane away is exact
    rather than merely close: asserted at zero tolerance, not a tolerance."""
    import jax.numpy as jnp
    import numpy as np

    from software_defined_matter.sdf.compile import make_sdf_closure

    tree = sdf_op(
        "union",
        [
            sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[3.0, 0.0, 0.0]),
            sdf_primitive("box", b=[1.0, 2.0, 0.5]),
            sdf_primitive("torus", t=[2.0, 0.4]),
        ],
    )
    part = _wrap(tree)
    pts = jnp.asarray(np.random.default_rng(3).uniform(-8.0, 8.0, size=(4000, 3)))
    root = np.asarray(make_sdf_closure(tree, part)(pts))

    # What sdf_scene now computes: union over components, each cut at +3e38.
    from software_defined_matter.glsl.emit import _expand_components

    per = np.stack(
        [np.asarray(make_sdf_closure(n, part)(pts)) for n, _ in _expand_components(tree, [])]
    )
    cut = np.asarray(pts)[:, 2] - 3.0e38
    delegated = np.min(np.maximum(per, cut[None, :]), axis=0)
    assert np.array_equal(delegated, root)


def test_no_aabb_early_out_is_emitted():
    """The prune is a large win and is NOT unconditionally sound: a max()-built
    field can report a value below its own box distance, so the skip drops the
    candidate owning the true minimum (OW-II: wrong by up to 7.08 mm). It
    belongs in its own change with the per-candidate safety analysis. This
    asserts we did not quietly take the win."""
    em = _emit(
        sdf_op("union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])]),
        metadata={"components": ["a", "b"]},
    )
    assert "sdm_aabb_dist" not in em.scene_source


def test_rcut_with_the_plane_pushed_away_is_the_scene_field():
    """`sdf_scene_rcut` is the cut field; with no cut it must be the same
    geometry, or the cutaway view and the march disagree about the part."""
    em = _emit(
        sdf_op("union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])]),
        metadata={"components": ["a", "b"]},
    )
    body = em.scene_source.split("float sdf_scene_rcut")[1]
    # One union term per component, each cut at p.
    assert body.count("sdm_cut_plane(p, cn, co)") == 2
    assert body.count("op_union") == 2


def test_the_components_partition_the_root_field_exactly():
    """THE INVARIANT SEGMENTATION RESTS ON: `min` over the component fields is
    the root field, pointwise.

    Everything else here is a string assertion about names and labels. This is
    the one that says the split did not change the geometry, and it is not
    approximate: wrapper redistribution is an identity, not a numerical
    approximation, so the tolerance is exact zero.

    Checked against the JAX compiler rather than the emitted GLSL, because a
    GPU is not available in CI and the two pipelines are already held together
    by the parity suite.
    """
    import jax.numpy as jnp
    import numpy as np

    from software_defined_matter.glsl.emit import _expand_components, _segmentable
    from software_defined_matter.sdf.compile import make_sdf_closure

    tree = sdf_op(
        "union",
        [
            sdf_transform(
                "translate",
                sdf_op(
                    "union",
                    [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])],
                ),
                t=[5.0, 0.0, 0.0],
            ),
            sdf_primitive("torus", t=[2.0, 0.5]),
            sdf_primitive("capped_cylinder", h=1.0, r=0.4),
        ],
    )
    part = _wrap(tree)
    assert _segmentable(tree, smooth_csg=False)

    pts = jnp.asarray(np.random.default_rng(0).uniform(-8.0, 8.0, size=(4000, 3)))
    root = make_sdf_closure(tree, part)(pts)
    per_component = jnp.stack(
        [make_sdf_closure(node, part)(pts) for node, _ in _expand_components(tree, [])]
    )
    assert float(jnp.max(jnp.abs(jnp.min(per_component, axis=0) - root))) == 0.0


def test_a_modifier_child_still_partitions_the_root():
    """The descent stopping is only correct if the un-expanded child is still a
    faithful component: the identity has to hold for the coarse split too."""
    import jax.numpy as jnp
    import numpy as np

    from software_defined_matter.glsl.emit import _expand_components
    from software_defined_matter.sdf.compile import make_sdf_closure

    tree = sdf_op(
        "union",
        [
            sdf_modifier(
                "onion",
                sdf_op(
                    "union",
                    [sdf_primitive("sphere", r=2.0), sdf_primitive("box", b=[1, 1, 1])],
                ),
                thickness=0.1,
            ),
            sdf_primitive("torus", t=[3.0, 0.5]),
        ],
    )
    part = _wrap(tree)
    pts = jnp.asarray(np.random.default_rng(1).uniform(-6.0, 6.0, size=(3000, 3)))
    root = make_sdf_closure(tree, part)(pts)
    per_component = jnp.stack(
        [make_sdf_closure(node, part)(pts) for node, _ in _expand_components(tree, [])]
    )
    assert float(jnp.max(jnp.abs(jnp.min(per_component, axis=0) - root))) == 0.0


def test_a_tie_is_won_by_the_LOWER_component_id():
    """Two components touching at a seam are exactly equidistant along it, and
    the fold has to break that tie the same way every time or the surface is
    painted in two colours that swap under any sub-voxel perturbation. `<`
    keeps the first; `<=` would keep the last.

    This is not hypothetical: a shaded pixel moving 23/255 across a re-bake
    was traced to a hit point crossing a seam between two touching components
    and the ownership answer flipping.
    """
    em = _emit(
        sdf_op("union", [sdf_primitive("sphere", r=1.0), sdf_primitive("box", b=[1, 1, 1])]),
        metadata={"components": ["a", "b"]},
    )
    body = em.scene_source.split("int sdf_scene_comp")[1]
    assert "if (dk < d)" in body
    assert "<=" not in body, "a tie must go to the lower id, not the later one"


def test_each_component_carries_a_box_that_contains_it():
    """A host culls and packs with these, and `webgl.js` gates whole features on
    every component having one. A box that does not CONTAIN its component makes
    the cull drop visible geometry, so containment is the property to assert,
    not merely that a box is present."""
    import jax.numpy as jnp
    import numpy as np

    from software_defined_matter.glsl.emit import _expand_components

    tree = sdf_op(
        "union",
        [
            sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[5.0, 0.0, 0.0]),
            sdf_primitive("box", b=[1.0, 2.0, 0.5]),
        ],
    )
    em = _emit(tree, metadata={"components": ["ball", "brick"]})
    assert all(c["bbox"] is not None for c in em.components)

    from software_defined_matter.sdf.compile import make_sdf_closure

    part = _wrap(tree)
    for c, (node, _) in zip(em.components, _expand_components(tree, []), strict=True):
        (lo, hi) = c["bbox"]
        # Sample the box's own volume: every point the component calls solid
        # (d <= 0) must lie inside the reported box, with a float32 margin.
        pts = jnp.asarray(
            np.random.default_rng(2).uniform(np.array(lo) - 1.0, np.array(hi) + 1.0, size=(4000, 3))
        )
        d = make_sdf_closure(node, part)(pts)
        solid = np.asarray(pts)[np.asarray(d) <= 0.0]
        assert len(solid) > 0, "vacuous: no solid points sampled"
        assert (solid >= np.array(lo) - 1e-4).all()
        assert (solid <= np.array(hi) + 1e-4).all()


def test_a_component_whose_box_cannot_be_inferred_reports_None():
    """Unbounded geometry has no finite box, and that is an answer. Faking one
    from the part box would claim the component occupies space it does not, and
    a cull is only sound while the box CONTAINS the component."""
    em = _emit(
        sdf_op(
            "union", [sdf_primitive("sphere", r=1.0), sdf_primitive("plane", n=[0, 0, 1], h=0.0)]
        ),
        metadata={"bbox": [[-9.0, -9.0, -9.0], [9.0, 9.0, 9.0]], "components": ["ball", "ground"]},
    )
    boxes = [c["bbox"] for c in em.components]
    assert boxes[0] is not None
    assert boxes[1] is None


# ---------------------------------------------------------------------------
# Loft, and the curve sections that lower into it
# ---------------------------------------------------------------------------


def _square(n: float) -> SDFTree:
    """A closed square outline of half-size ``n``, four numeric vertices."""
    return sdf_primitive("polygon_2d", vertices=[[-n, -n], [n, -n], [n, n], [-n, n]])


def _ring(n: int, r: float = 1.0) -> list[list[float]]:
    """``n`` control points evenly around a circle of radius ``r``."""
    import math

    return [
        [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)] for i in range(n)
    ]


def _loft_fn_body(em: GLSLEmission) -> str:
    """The body of the last emitted `float sdf_nN_d3(vec3 p)`, comments gone."""
    src = _strip_glsl_comments(em.scene_source)
    spans = list(re.finditer(r"^float (sdf_n\d+_d3)\(vec3 p\) \{", src, flags=re.M))
    assert spans, "no 3-D node function was emitted"
    start = spans[-1].end()
    return src[start : src.index("\n}", start)]


def test_a_field_loft_calls_every_section_and_caps_the_span():
    """`interp='field'` interpolates DISTANCES, so each section's own 2-D
    function has to be evaluated and the result handed to the span cap."""
    em = _emit(sdf_loft([_square(2.0), _square(1.0)], z=[-1.0, 3.0]))
    body = _loft_fn_body(em)
    assert body.count("(p.xy)") == 2, "one call per section, at the query's XY"
    assert "float ZS[2] = float[2](-1.0, 3.0)" in body
    assert "sdm_loft_cap(d2d, p.z, zc, hh)" in body
    assert "mix(dv[idx], dv[idx + 1], t)" in body


def test_a_smooth_loft_uses_the_monotone_tangent_and_a_linear_one_does_not():
    """PCHIP is what keeps a smooth loft watertight, so its presence is not a
    detail of the emitted text — a smooth loft that silently emitted the linear
    branch would render a creased solid that still looks like a loft."""
    smooth = _loft_fn_body(_emit(sdf_loft([_square(2.0), _square(1.0)], z=[0.0, 1.0], smooth=True)))
    linear = _loft_fn_body(_emit(sdf_loft([_square(2.0), _square(1.0)], z=[0.0, 1.0])))
    assert "sdm_pchip_tan(" in smooth
    assert "sdm_pchip_tan" not in linear
    assert "mix(" in linear


def test_a_shape_loft_bakes_its_outlines_in_section_major_order():
    """The flat SEC array is indexed `idx * N + j`, so section-major is not a
    convention here, it is the arithmetic. Transposing it would interpolate
    vertex 0 of one section against vertex 1 of the same section and shear
    every outline, which reads as a twisted loft rather than as an error."""
    a, b = _square(2.0), _square(1.0)
    em = _emit(sdf_loft([a, b], z=[0.0, 4.0], interp="shape"))
    body = _loft_fn_body(em)
    baked = [
        [float(x), float(y)] for x, y in re.findall(r"vec2\((-?[\d.e+-]+), (-?[\d.e+-]+)\)", body)
    ]
    expected = a["params"]["vertices"] + b["params"]["vertices"]
    assert baked == expected
    assert "vec2 SEC[8] = vec2[8](" in body
    assert "sdf_polygon_2d(p.xy, V, 4)" in body


def test_a_shape_loft_over_curve_sections_is_refused_by_name():
    """The JAX field interpolates a curve loft's CONTROL POINTS
    (ops.loft_shape_curve) and samples afterwards. Sampling first and
    interpolating the outlines agrees only while the interpolation is linear,
    and PCHIP is not, so emitting it would put a different solid on screen from
    the one that gets meshed. Refused, and the message says what to use."""
    curve = sdf_primitive("bspline_2d", control_points=_ring(6))
    with pytest.raises(NotImplementedError, match="loft_shape_curve"):
        _emit(sdf_loft([curve, curve], z=[0.0, 1.0], interp="shape", smooth=True))


def test_a_curve_section_is_fine_in_a_field_loft():
    """Nothing is wrong with a curve per se: `interp='field'` evaluates each
    section's own SDF, and a curve's own SDF IS its sampled outline's."""
    curve = sdf_primitive("bspline_2d", control_points=_ring(6))
    em = _emit(sdf_loft([curve, curve], z=[0.0, 1.0], smooth=True))
    assert "sdm_loft_cap" in em.scene_source


def test_a_curve_primitive_emits_the_outline_its_jax_shape_evaluates():
    """Field parity for the lowering, without a GPU.

    The outline is baked as literals, so it can be read back out of the emitted
    source and pushed through the JAX polygon SDF. If the emitter sampled a
    different basis, a different point count, or a different start vertex from
    `sdf_shapes.bspline_2d`, the two fields separate here."""
    import jax.numpy as jnp
    import numpy as np

    from software_defined_matter.sdf import sdf_shapes

    cps = _ring(7, r=3.0)
    em = _emit(sdf_2d_to_3d("extrusion", sdf_primitive("bspline_2d", control_points=cps), h=1.0))
    baked = np.array(
        [
            [float(x), float(y)]
            for x, y in re.findall(
                r"vec2\((-?[\d.e+-]+), (-?[\d.e+-]+)\)", _strip_glsl_comments(em.scene_source)
            )
        ]
    )
    pts = jnp.asarray(np.random.default_rng(11).uniform(-5.0, 5.0, size=(2000, 2)))
    assert np.allclose(
        np.asarray(sdf_shapes.polygon_2d(pts, jnp.asarray(baked))),
        np.asarray(sdf_shapes.bspline_2d(pts, jnp.asarray(cps))),
        atol=1e-6,
    )


def test_the_polygon_prescan_counts_a_curves_sampled_outline():
    """SDM_POLY_MAX_N is fixed before anything is emitted, and a curve's node
    does not state its vertex count. Undercounting it declares V[] shorter than
    the `n` sdf_polygon_2d loops to, which reads past the end of the array."""
    from software_defined_matter.glsl.emit import _curve_samples

    cps = _ring(7, r=3.0)
    em = _emit(sdf_2d_to_3d("extrusion", sdf_primitive("bspline_2d", control_points=cps), h=1.0))
    assert f"#define SDM_POLY_MAX_N {7 * _curve_samples()}" in em.lib_source


def test_a_loft_station_ref_becomes_a_uniform():
    """Stations are the one part of a loft a host can scrub, so a $ref there
    has to reach the shader as a uniform and not as its resolved value."""
    em = _emit(
        sdf_loft([_square(2.0), _square(1.0)], z=[0.0, make_param_ref("span")]),
        params=[Param(name="span", value=4.0, unit="mm")],
    )
    assert "u_p_span" in _loft_fn_body(em)
    assert "span" in {u.name.removeprefix("u_p_") for u in em.uniforms}


def test_a_curve_control_point_ref_is_refused_rather_than_frozen():
    """The outline is baked into the shader text, so a $ref control point would
    keep its emit-time shape while the param moved — geometry that silently
    stops tracking its own parameter."""
    curve = sdf_primitive(
        "bspline_2d", control_points=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], make_param_ref("k")]
    )
    with pytest.raises(NotImplementedError, match="control_points"):
        _emit(
            sdf_2d_to_3d("extrusion", curve, h=1.0),
            params=[Param(name="k", value=0.5, unit="mm")],
        )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"z": [0.0]}, "at least 2"),
        ({"z": [0.0, 1.0, 2.0]}, "one axial station per section"),
        ({"z": [0.0, 1.0], "interp": "spline"}, "interp must be"),
    ],
)
def test_a_malformed_loft_is_rejected_with_a_reason(kwargs, message):
    children = [_square(2.0)] if kwargs["z"] == [0.0] else [_square(2.0), _square(1.0)]
    node = {"type": "loft", "children": children, "params": dict(kwargs)}
    with pytest.raises(ValueError, match=message):
        _emit(node)


def test_a_shape_loft_needs_the_same_vertex_count_in_every_section():
    """Shape interpolation is per-vertex, so unequal counts have no
    correspondence to interpolate along."""
    tri = sdf_primitive("polygon_2d", vertices=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="vertex correspondence"):
        _emit(sdf_loft([_square(2.0), tri], z=[0.0, 1.0], interp="shape"))


# ---------------------------------------------------------------------------
# Nothing unreachable
# ---------------------------------------------------------------------------


def _reachable_from_entries(src: str) -> set[str]:
    """Every function name reachable from the scene's entry points."""
    src = _strip_glsl_comments(src)
    bodies: dict[str, str] = {}
    for m in re.finditer(r"(?m)^(?:float|vec2|vec3|int)\s+(\w+)\s*\([^)]*\)\s*\{", src):
        end = src.index("\n}", m.end())
        bodies[m.group(1)] = src[m.end() : end]
    seen: set[str] = set()
    stack = [
        n
        for n in ("sdf_scene", "sdf_scene_rcut", "sdf_scene_comp", "sdm_rest_point")
        if n in bodies
    ]
    while stack:
        fn = stack.pop()
        if fn in seen:
            continue
        seen.add(fn)
        stack += [c for c in re.findall(r"\b(\w+)\s*\(", bodies.get(fn, "")) if c in bodies]
    return seen


def test_a_segmented_scene_emits_nothing_it_cannot_reach():
    """The root fold is not emitted when the components replace it.

    Every entry point of a segmented scene is built out of the component
    functions, and `sdf_scene` delegates to `sdf_scene_rcut`, so the root fold
    is unreachable the moment segmentation fires. Walking it anyway emitted its
    whole subtree dead: 48.6% of the emitted GLSL over a downstream consumer's
    13-part corpus, and 643,718 chars down to 330,397 on the largest.

    It cost nothing to correctness, which is why it survived. It is not free to
    a host: WebGL2 gates its grid bake on `scene_glsl.length > 150000`, and a
    whole-scene program compiles once per rebuild in seconds.
    """
    tree = sdf_op(
        "union",
        [
            sdf_transform("translate", sdf_primitive("sphere", r=1.0), t=[3.0, 0.0, 0.0]),
            sdf_primitive("box", b=[1.0, 2.0, 0.5]),
            sdf_primitive("torus", t=[2.0, 0.4]),
        ],
    )
    em = _emit(tree)
    assert len(em.components) == 3, "fixture must segment, or this proves nothing"
    defined = set(_glsl_definitions(em.scene_source))
    dead = {d for d in defined if d.startswith("sdf_n")} - _reachable_from_entries(em.scene_source)
    assert not dead, f"emitted but unreachable: {sorted(dead)}"


def test_an_unsegmented_scene_still_emits_and_calls_its_root_fold():
    """The other half. A scene with nothing to segment has no component
    functions to build the entry points from, so the root fold is what they
    call and it must still be there."""
    em = _emit(
        sdf_op("subtract", [sdf_primitive("sphere", r=2.0), sdf_primitive("box", b=[1.0] * 3)])
    )
    assert em.components == []
    defined = set(_glsl_definitions(em.scene_source))
    reach = _reachable_from_entries(em.scene_source)
    roots = {d for d in defined if d.startswith("sdf_n")}
    assert roots and roots <= reach, f"unreachable in an unsegmented scene: {sorted(roots - reach)}"


# ---------------------------------------------------------------------------
# Table-stored polygon vertices
# ---------------------------------------------------------------------------


def _big_poly(n: int, r: float = 5.0) -> SDFTree:
    """A polygon with ``n`` numeric vertices, evenly around a circle."""
    import math

    return sdf_primitive(
        "polygon_2d",
        vertices=[
            [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)] for i in range(n)
        ],
    )


def test_a_small_part_keeps_its_outlines_in_the_source():
    """Below the threshold nothing changes shape, which is what keeps every
    part that renders today rendering the same way."""
    em = _emit(sdf_2d_to_3d("extrusion", _big_poly(8), h=1.0))
    assert em.poly_table == []
    # The library always CARRIES the table half behind an #ifdef; what says
    # this emission does not use it is the absence of the define.
    assert "#define SDM_POLY_TABLE" not in em.lib_source
    assert "vec2[SDM_POLY_MAX_N]" in em.scene_source


def test_a_large_part_moves_its_outlines_into_the_table():
    """Above it the coordinates leave the program text entirely.

    That is the whole point: a host that gates its fast path on shader BYTES
    was refusing banana over 7,800 `vec2` literals, 89% of a 207 KB shader.
    """
    em = _emit(sdf_2d_to_3d("extrusion", _big_poly(600), h=1.0))
    assert len(em.poly_table) == 2 * (600 + 1), "600 vertices plus one header texel"
    assert not re.search(r"vec2\(-?[\d.]+, -?[\d.]+\)", em.scene_source)
    assert "sdm_polygon_2d_tab(p, 0)" in em.scene_source
    for d in ("#define SDM_POLY_TABLE", "#define SDM_POLY_TEX_W 1024", "#define SDM_POLY_LEN 601"):
        assert d in em.lib_source


def test_the_header_texel_carries_the_count_and_the_vertices_follow():
    """`sdm_polygon_2d_tab` reads the count out of the table at run time, so a
    header that disagreed with what follows would size the edge loop wrongly
    and read a neighbouring outline's vertices into this one."""
    poly = _big_poly(600)
    em = _emit(sdf_2d_to_3d("extrusion", poly, h=1.0))
    assert em.poly_table[0] == 600.0
    assert em.poly_table[1] == 0.0
    flat = [c for v in poly["params"]["vertices"] for c in v]
    assert em.poly_table[2:] == pytest.approx(flat)


def test_a_ref_vertex_keeps_its_outline_in_the_source():
    """A table texel is uploaded once, so a coordinate the host can still
    scrub has to stay a uniform in the program text. Mixing the two would
    freeze the param at its emit-time value and look like geometry that
    stopped responding."""
    verts = _big_poly(600)["params"]["vertices"]
    verts[0] = [make_param_ref("nose_x"), 0.0]
    tree = sdf_2d_to_3d("extrusion", sdf_primitive("polygon_2d", vertices=verts), h=1.0)
    em = _emit(tree, params=[Param(name="nose_x", value=5.0, unit="mm")])
    assert em.poly_table == []
    assert "u_p_nose_x" in em.scene_source
    assert "vec2[SDM_POLY_MAX_N]" in em.scene_source


def test_a_shape_loft_fetches_its_sections_in_section_major_order():
    """The loft is where banana's vertices actually are, and none of them are
    at a polygon call site. Its sections go in the table headerless, because
    the count is a compile-time constant here and the loop indexes
    `base + section * N + j` itself. Transposing that shears every outline.
    """
    secs = [_big_poly(300, r=r) for r in (5.0, 4.0)]
    em = _emit(sdf_loft(secs, z=[0.0, 10.0], interp="shape", smooth=True))
    assert len(em.poly_table) == 2 * 2 * 300, "two sections of 300, no header"
    assert not re.search(r"vec2 SEC\[", em.scene_source)
    assert "sdm_poly_fetch(0 + idx * 300 + j)" in _strip_glsl_comments(em.scene_source)
    expected = [c for s in secs for v in s["params"]["vertices"] for c in v]
    assert em.poly_table == pytest.approx(expected)


def test_the_threshold_sums_vertices_rather_than_taking_the_largest():
    """One 600-vertex outline and six hundred 1-vertex ones weigh the same in
    the emitted file, and `_max_polygon_verts` cannot tell them apart."""
    from software_defined_matter.glsl.emit import (
        _POLY_TABLE_THRESHOLD,
        _max_polygon_verts,
        _total_polygon_verts,
    )

    many = sdf_op(
        "union", [sdf_2d_to_3d("extrusion", _big_poly(60, r=1.0 + i), h=1.0) for i in range(10)]
    )
    assert _total_polygon_verts(many) == 600
    assert _max_polygon_verts(many) == 60
    assert _total_polygon_verts(many) > _POLY_TABLE_THRESHOLD
    assert _emit(many).poly_table, "ten medium outlines cross it together"


def test_the_tabled_outline_is_the_field_the_jax_shape_evaluates():
    """Field parity for the storage change, without a GPU.

    The table is returned as data, so it can be pushed straight through the
    JAX polygon SDF and compared with the same polygon evaluated normally. A
    reordering, an off-by-one in the header, or a dropped vertex separates
    them here.
    """
    import jax.numpy as jnp
    import numpy as np

    from software_defined_matter.sdf import sdf_shapes

    poly = _big_poly(600, r=5.0)
    em = _emit(sdf_2d_to_3d("extrusion", poly, h=1.0))
    tabled = np.asarray(em.poly_table[2:], dtype=np.float64).reshape(-1, 2)
    pts = jnp.asarray(np.random.default_rng(5).uniform(-8.0, 8.0, size=(2000, 2)))
    assert np.allclose(
        np.asarray(sdf_shapes.polygon_2d(pts, jnp.asarray(tabled))),
        np.asarray(sdf_shapes.polygon_2d(pts, jnp.asarray(poly["params"]["vertices"]))),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# poly_lod outline resampling
# ---------------------------------------------------------------------------


def _ngon(n: int, r: float = 10.0) -> list[list[float]]:
    import math

    return [
        [r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)] for k in range(n)
    ]


def _poly3d(verts) -> SDFTree:
    return sdf_2d_to_3d("extrusion", sdf_primitive("polygon_2d", vertices=verts), h=1.0)


def test_poly_lod_default_is_full_resolution():
    em = _emit(_poly3d(_ngon(64)))
    assert em.poly_lod is None
    assert em.poly_max_n == 64
    assert "sdf_polygon_2d(p, V, 64)" in em.scene_source


def test_poly_lod_resamples_a_literal_outline_and_sizes_the_array_to_match():
    em = _emit(_poly3d(_ngon(64)), poly_lod=16)
    assert em.poly_lod == 16
    # SDM_POLY_MAX_N is sized to the POST-lod count, not the authored one.
    assert em.poly_max_n == 16
    assert "sdf_polygon_2d(p, V, 16)" in em.scene_source
    assert "#define SDM_POLY_MAX_N 16" in em.lib_source


def test_poly_lod_keeps_sharp_features():
    """A mostly-flat outline with one spike keeps the spike's vertices."""
    verts = _ngon(64, r=10.0)
    verts[7] = [30.0, 0.5]  # a spike far off the circle
    em = _emit(_poly3d(verts), poly_lod=16)
    assert "30.0" in em.scene_source  # the spike survived resampling


def test_poly_lod_never_touches_a_ref_outline():
    verts = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [5.0, 12.0], [0.0, 10.0]]
    verts_with_ref = [[make_param_ref("px"), 0.0]] + verts[1:]
    em = _emit(
        _poly3d(verts_with_ref),
        params=[Param(name="px", value=0.0, free=False)],
        poly_lod=4,
    )
    # 5 vertices, lod 4 — but a $ref vertex means hands off, full count kept.
    assert "sdf_polygon_2d(p, V, 5)" in em.scene_source


def test_poly_lod_below_a_polygon_is_rejected():
    with pytest.raises(ValueError, match="poly_lod"):
        _emit(_poly3d(_ngon(8)), poly_lod=2)


def test_poly_lod_loft_sections_share_one_index_set():
    """Vertex correspondence survives: every section is cut at the SAME
    indices, so section k's vertex j still interpolates against section
    k+1's vertex j."""

    a = _ngon(32, r=2.0)
    b = _ngon(32, r=1.0)
    em = _emit(
        sdf_loft(
            [sdf_primitive("polygon_2d", vertices=a), sdf_primitive("polygon_2d", vertices=b)],
            z=[0.0, 1.0],
            interp="shape",
        ),
        poly_lod=8,
    )
    m = re.search(r"vec2 SEC\[(\d+)\] = vec2\[\d+\]\((.*?)\);", em.scene_source, re.S)
    assert m, "loft SEC array not found"
    total = int(m.group(1))
    assert total == 16  # 2 sections x 8 kept vertices
    pairs = re.findall(r"vec2\(([-\d.e]+), ([-\d.e]+)\)", m.group(2))
    sec = [(float(x), float(y)) for x, y in pairs]
    a_kept, b_kept = sec[:8], sec[8:]
    # Same indices kept in both sections: b is a scaled by exactly 0.5,
    # pairwise in order — the correspondence a shape-loft interpolates.
    for (ax, ay), (bx, by) in zip(a_kept, b_kept, strict=True):
        assert abs(ax * 0.5 - bx) < 1e-9 and abs(ay * 0.5 - by) < 1e-9


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _sweep_bbox() -> dict:
    return {"bbox": [[-9, -9, -9], [9, 9, 9]]}


def test_a_sweep_bakes_its_frames_and_calls_the_child_in_the_section_plane():
    """The whole branch: frames as arrays, the nearest-segment loop, the child
    as a 2-D function of (u, v), and the cap as a library call."""
    em = _emit(_bent_sweep(), metadata=_sweep_bbox())
    src = em.scene_source
    assert (
        "vec3 A[3] = vec3[3](vec3(0.0, 0.0, 0.0), vec3(2.0, 0.0, 0.0), vec3(2.0, 2.0, 0.0));" in src
    )
    assert "float L[3] = float[3](2.0, 2.0, 2.0);" in src
    assert "for (int i = 0; i < 3; ++i) {" in src
    assert "vec2(dot(perp_star, NN[istar]), dot(perp_star, BB[istar])), over, end_cap);" in src
    assert re.search(r"float d2d = sdf_n\d+_d2\(uv\);", src)
    assert "return sdm_sweep_finish(d2d, end_cap ? over : 0.0);" in src
    assert "sdf_circle_2d(p, 0.4)" in src
    assert em.sweep_max_s == 3
    assert em.sweep_table == []
    assert em.sweep_tex_width == 0


def test_a_sweep_path_param_reports_re_emit():
    """The path is baked, so a param the authoring script used to compute it
    never binds a uniform. Re-emitting is what an edit to it costs, and that
    is what the manifest has to say."""
    em = _emit(
        _bent_sweep(),
        params=[Param(name="coil_radius", value=2.0, unit="mm")],
        metadata=_sweep_bbox(),
    )
    (c,) = em.controls
    assert c["param"] == "coil_radius"
    assert c["class"] == "re-emit"
    assert "uniform" not in c


def test_a_ref_sweep_path_point_is_refused_rather_than_frozen():
    """A `$ref` point would be baked at its emit-time value while the param
    moved. The frame depends on the whole path, so no per-point uniform could
    carry it either; refusing by name is the honest answer."""
    path = _bent_path()
    path[1] = [make_param_ref("k"), 0.0, 0.0]
    tree = sdf_sweep(sdf_primitive("circle_2d", r=0.4), path, path_kind="polyline")
    with pytest.raises(NotImplementedError, match="path control points must be literal"):
        _emit(tree, params=[Param(name="k", value=2.0, unit="mm")], metadata=_sweep_bbox())


def test_an_expression_sweep_path_point_is_refused_too():
    path = _bent_path()
    path[2] = [2.0, {"type": "binop", "op": "mul", "lhs": {"$ref": "k"}, "rhs": 2.0}, 0.0]
    tree = sdf_sweep(sdf_primitive("circle_2d", r=0.4), path, path_kind="polyline")
    with pytest.raises(NotImplementedError, match="path control points must be literal"):
        _emit(tree, params=[Param(name="k", value=1.0, unit="mm")], metadata=_sweep_bbox())


def test_a_ref_sweep_normal0_is_refused_rather_than_frozen():
    """normal0 seeds the rotation-minimising frame, which is baked with it."""
    tree = _bent_sweep(normal0=[0.0, 0.0, make_param_ref("nz")])
    with pytest.raises(NotImplementedError, match="normal0 must be a literal"):
        _emit(tree, params=[Param(name="nz", value=1.0, unit="mm")], metadata=_sweep_bbox())


def test_a_literal_normal0_reseeds_the_frame():
    """The default seed for a +x first tangent is world y, so seeding with z
    turns the first normal onto z. Visible in the baked NN array."""
    em = _emit(_bent_sweep(normal0=[0.0, 0.0, 1.0]), metadata=_sweep_bbox())
    m = re.search(r"vec3 NN\[3\] = vec3\[3\]\(vec3\(([^)]*)\)", em.scene_source)
    assert m, "NN array not found"
    first = [float(c) for c in m.group(1).split(",")]
    assert first == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_a_sweep_without_a_path_is_rejected():
    node = {"type": "sweep", "child": sdf_primitive("circle_2d", r=0.4), "params": {}}
    with pytest.raises(ValueError, match="requires a 'path'"):
        _emit(node, metadata=_sweep_bbox())


def test_a_sweep_with_an_unknown_frame_is_rejected_by_the_kernels_own_check():
    """The frame builder is the kernel's, so its error is the kernel's."""
    node = _bent_sweep()
    node["params"]["frame"] = "frenet"
    with pytest.raises(ValueError, match="frame must be 'rmf' or 'cylindrical'"):
        _emit(node, metadata=_sweep_bbox())


def test_a_sweep_in_a_2d_context_is_rejected():
    """A sweep produces a 3-D function; lifting one as if it were a profile is
    a tree the compiler would refuse too."""
    tree = sdf_2d_to_3d("extrusion", _bent_sweep(), h=1.0)
    with pytest.raises(ValueError, match="sweep produces a 3-D function"):
        _emit(tree, metadata=_sweep_bbox())


def test_a_bspline_sweep_path_is_always_a_loop():
    """`_compile_sweep` treats a bspline path as periodic whatever `closed`
    says, so the emitter has to bake as many segments as points."""
    ctrl = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.5], [-2.0, 0.0, 0.0], [0.0, -2.0, -0.5]]
    em = _emit(
        sdf_sweep(sdf_primitive("circle_2d", r=0.3), ctrl, path_kind="bspline", closed=False),
        metadata=_sweep_bbox(),
    )
    # 4 spans x 12 samples per span, wrapped: as many segments as points.
    assert em.sweep_max_s == 48
    assert "vec3 A[48] = vec3[48](" in em.scene_source


# ---------------------------------------------------------------------------
# Table-stored sweep frames
# ---------------------------------------------------------------------------


def _long_sweep(n: int, r: float = 5.0, z_amp: float = 0.5) -> SDFTree:
    """A polyline of ``n`` points around a wavy circle, ``n - 1`` segments."""
    import math

    path = [
        [
            r * math.cos(2 * math.pi * i / n),
            r * math.sin(2 * math.pi * i / n),
            z_amp * math.sin(6 * math.pi * i / n),
        ]
        for i in range(n)
    ]
    return sdf_sweep(sdf_primitive("circle_2d", r=0.3), path, path_kind="polyline")


def test_a_small_sweep_keeps_its_frames_in_the_source():
    em = _emit(_long_sweep(50), metadata=_sweep_bbox())
    assert em.sweep_table == []
    assert "#define SDM_SWEEP_TABLE" not in em.lib_source
    assert "vec3 A[49] = vec3[49](" in em.scene_source


def test_a_large_sweep_moves_its_frames_into_the_table():
    """Above the budget the frames leave the program text entirely. A coil is
    hundreds of segments times dozens of conductors, and as literals that is
    more shader than every polygon in banana."""
    em = _emit(_long_sweep(600), metadata=_sweep_bbox())
    assert len(em.sweep_table) == 4 * (1 + 4 * 599), "header plus four texels per segment"
    assert "vec3 A[" not in em.scene_source
    assert "vec3 uvo = sdm_sweep_tab(p, 0);" in em.scene_source
    assert "return sdm_sweep_finish(d2d, uvo.z);" in em.scene_source
    assert em.sweep_tex_width == 1024
    assert em.sweep_max_s == 599
    for d in (
        "#define SDM_SWEEP_TABLE",
        "#define SDM_SWEEP_TEX_W 1024",
        f"#define SDM_SWEEP_LEN {1 + 4 * 599}",
    ):
        assert d in em.lib_source


def test_the_sweep_header_texel_carries_the_count_and_the_frames_follow():
    """`sdm_sweep_tab` strides four texels per segment from the header, in the
    order (A, L), (T, 0), (N, 0), (B, 0). A slot out of order swaps a frame
    axis for a tangent and the profile lies down along the path."""
    import numpy as np

    from software_defined_matter.glsl.emit import _sweep_frame_arrays

    tree = _long_sweep(600)
    em = _emit(tree, metadata=_sweep_bbox())
    tex = np.asarray(em.sweep_table).reshape(-1, 4)
    assert tex[0].tolist() == [599.0, 0.0, 0.0, 0.0]
    A, T, L, N, B = _sweep_frame_arrays(tree["params"]["path"], "polyline", False, "rmf", None)
    assert tex[1::4, :3] == pytest.approx(A)
    assert tex[1::4, 3] == pytest.approx(L)
    assert tex[2::4, :3] == pytest.approx(T)
    assert tex[3::4, :3] == pytest.approx(N)
    assert tex[4::4, :3] == pytest.approx(B)
    assert np.all(tex[2::4, 3] == 0.0)


def test_the_sweep_budget_sums_points_across_conductors():
    """The motor case: ninety-six conductors of two hundred points, no single
    one of which crosses the budget. Together they are the whole shader."""
    from software_defined_matter.glsl.emit import _POLY_TABLE_THRESHOLD, _total_sweep_points

    many = sdf_op(
        "union",
        [
            sdf_transform("translate", _long_sweep(60, r=1.0 + 0.1 * i), t=[0.0, 0.0, float(i)])
            for i in range(10)
        ],
    )
    assert _total_sweep_points(many) == 600 > _POLY_TABLE_THRESHOLD
    em = _emit(many, metadata=_sweep_bbox())
    headers = re.findall(r"sdm_sweep_tab\(p, (\d+)\);", em.scene_source)
    assert len(headers) == 10
    # Each header sits right after the previous conductor's last texel.
    assert [int(h) for h in headers] == [i * (1 + 4 * 59) for i in range(10)]


def test_the_sweep_budget_is_counted_after_sampling():
    """A bspline of a few control points samples to many segments, and the
    budget has to see what will be emitted, not what was authored."""
    import math

    from software_defined_matter.glsl.emit import _total_sweep_points

    ctrl = [
        [3.0 * math.cos(a), 3.0 * math.sin(a), 0.0] for a in [i * 2 * math.pi / 5 for i in range(5)]
    ]
    tree = sdf_sweep(sdf_primitive("circle_2d", r=0.3), ctrl, path_kind="bspline")
    assert _total_sweep_points(tree) == 5 * 12


def test_a_ref_path_counts_nothing_toward_the_sweep_budget():
    """Emission refuses it before it could be stored either way."""
    from software_defined_matter.glsl.emit import _total_sweep_points

    path = _bent_path()
    path[0] = [make_param_ref("k"), 0.0, 0.0]
    tree = sdf_sweep(sdf_primitive("circle_2d", r=0.4), path, path_kind="polyline")
    assert _total_sweep_points(tree) == 0


def test_sweep_and_polygon_tables_are_decided_independently():
    """A big sweep does not push polygons into their table, or the reverse:
    the two are different textures a host binds separately."""
    tree = sdf_op(
        "union",
        [_long_sweep(600), sdf_2d_to_3d("extrusion", _big_poly(8), h=1.0)],
    )
    em = _emit(tree, metadata=_sweep_bbox())
    assert em.sweep_table and not em.poly_table
    assert "vec2[SDM_POLY_MAX_N]" in em.scene_source
