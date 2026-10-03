"""Grounded mates compose nested poses, retain live inputs and diagnose inconsistent loops."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jsonschema import ValidationError

from software_defined_matter import (
    Assembly,
    Dof,
    Frame,
    Instance,
    Mate,
    Param,
    Part,
    PartRef,
    PlacementError,
    Port,
    compile_placement,
    load_bundle,
    save,
    validate,
)

ATOL = 2e-5  # Float32 rigid-transform composition through a small mate graph.


def _load(tmp_path, root, **parts):
    for name, document in parts.items():
        save(document, tmp_path / f"{name}.sdm")
    path = tmp_path / "assembly.sdm"
    save(root, path)
    return load_bundle(path)


def _link(name="link", length=2):
    return Part(
        name,
        params={"length": Param("length", length, unit="mm")},
        ports=[Port("start"), Port("end", Frame(position=({"$ref": "length"}, 0, 0)))],
    )


def _pair(kind="fixed", ground_child=False):
    moving = kind != "fixed"
    return Assembly(
        "pair",
        params={"length": Param("length", 4, free=True, unit="mm")},
        instances=(
            Instance(
                "base",
                PartRef("link.sdm"),
                {"length": {"$ref": "length"}},
                transform=None if ground_child else Frame(),
            ),
            Instance("arm", PartRef("link.sdm"), transform=Frame() if ground_child else None),
        ),
        dofs={
            "q": Dof(
                "angle" if kind == "revolute" else "length",
                (-10, 10),
                "rad" if kind == "revolute" else "mm",
            )
        }
        if moving
        else {},
        mates=(Mate("joint", kind, "base.end", "arm.start", "q" if moving else None),),
        port={"output": "arm.end"},
    )


def test_fixed_placement_follows_design_length_without_recompilation(tmp_path):
    evaluator = compile_placement(_load(tmp_path, _pair(), link=_link()))
    run = jax.jit(lambda p: evaluator.evaluate(free_vec=p))
    for length in (4.0, 7.0):
        state = run(jnp.array([length]))
        assert bool(state.valid)
        np.testing.assert_allclose(state.instances["arm"][:3, 3], [length, 0, 0], atol=ATOL)
        np.testing.assert_allclose(state.ports["output"][:3, 3], [length + 2, 0, 0], atol=ATOL)
    derivative = jax.jacfwd(lambda p: evaluator.evaluate(free_vec=p).ports["output"][:3, 3])(
        jnp.array([4.0])
    )
    np.testing.assert_allclose(derivative[:, 0], [1, 0, 0], atol=ATOL)
    assert evaluator.mate_names == ("joint",)


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_joint_coordinates_use_right_handed_z_axis_after_offset(tmp_path, kind):
    root = _pair(kind)
    offset = Frame.from_axis_angle((0, 1, 0), math.pi / 2, position=(0, 0, 1))
    root = replace(root, mates=(replace(root.mates[0], offset=offset),))
    evaluator = compile_placement(_load(tmp_path, root, link=_link()))
    q = math.pi / 2 if kind == "revolute" else 3
    state = evaluator.evaluate_checked(dofs=[q])
    if kind == "revolute":
        np.testing.assert_allclose(state.ports["output"][:3, 3], [4, 2, 1], atol=ATOL)
    else:
        np.testing.assert_allclose(state.instances["arm"][:3, 3], [7, 0, 1], atol=ATOL)
        np.testing.assert_allclose(state.ports["output"][:3, 3], [7, 0, -1], atol=ATOL)


def test_reverse_traversal_preserves_parent_child_convention(tmp_path):
    root = _pair("revolute", ground_child=True)
    evaluator = compile_placement(_load(tmp_path, root, link=_link()))
    state = evaluator.evaluate_checked(dofs=[math.pi / 2])
    np.testing.assert_allclose(state.instances["base"][:3, 3], [0, 4, 0], atol=ATOL)
    np.testing.assert_allclose(state.ports["base.end"][:3, 3], [0, 0, 0], atol=ATOL)
    np.testing.assert_allclose(state.instances["arm"], np.eye(4), atol=ATOL)


def test_nested_promoted_ports_preserve_child_reference_frames(tmp_path):
    inner = Assembly(
        "inner",
        instances=(
            Instance(
                "leaf", PartRef("link.sdm"), {"length": 4}, transform=Frame(position=(3, 0, 0))
            ),
        ),
        port={"tip": "leaf.end"},
    )
    root = Assembly(
        "root",
        instances=(
            Instance(
                "anchor",
                PartRef("link.sdm"),
                transform=Frame.from_axis_angle((0, 0, 1), math.pi / 2, position=(10, 0, 0)),
            ),
            Instance("nested", PartRef("inner.sdm")),
        ),
        mates=(Mate("mount", "fixed", "anchor.start", "nested.tip"),),
        port={"output": "nested.tip"},
    )
    evaluator = compile_placement(_load(tmp_path, root, link=_link(), inner=inner))
    state = evaluator.evaluate_checked()
    np.testing.assert_allclose(state.instances["nested"][:3, 3], [10, -7, 0], atol=ATOL)
    np.testing.assert_allclose(state.instances["nested.leaf"][:3, 3], [10, -4, 0], atol=ATOL)
    np.testing.assert_allclose(state.ports["output"][:3, 3], [10, 0, 0], atol=ATOL)
    np.testing.assert_allclose(state.ports["output"], state.ports["nested.tip"], atol=ATOL)


def test_four_bar_reports_inconsistent_supplied_state_without_solving(tmp_path):
    instances = tuple(
        Instance(
            name,
            PartRef("link.sdm"),
            {"length": length},
            transform=Frame() if name == "base" else None,
        )
        for name, length in (("base", 2), ("right", 1), ("top", 2), ("left", 1))
    )
    mates = tuple(
        Mate(name, "revolute", parent + ".end", child + ".start", name)
        for name, parent, child in (
            ("a", "base", "right"),
            ("b", "right", "top"),
            ("c", "top", "left"),
            ("d", "left", "base"),
        )
    )
    root = Assembly(
        "four_bar",
        instances=instances,
        mates=mates,
        dofs={name: Dof("angle", (-4, 4), "rad", math.pi / 2) for name in "abcd"},
    )
    evaluator = compile_placement(_load(tmp_path, root, link=_link()))
    state = evaluator.evaluate_checked()
    np.testing.assert_allclose(state.instances["left"][:3, 3], [0, 1, 0], atol=ATOL)
    bad = np.array(evaluator.dof_defaults)
    bad[1] += 0.1
    assert not bool(jax.jit(lambda q: evaluator.evaluate(dofs=q))(bad).valid)
    with pytest.raises(PlacementError, match=r"d: position=.*mm, angle=.*rad"):
        evaluator.evaluate_checked(dofs=bad)
    # Residual differentiation is finite at regular configurations, including closure.
    jacobian = jax.jacfwd(lambda q: evaluator.evaluate(dofs=q).residuals)(
        jnp.asarray(evaluator.dof_defaults)
    )
    assert np.isfinite(jacobian).all()


def test_multiple_grounds_are_checked_and_ungrounded_components_are_rejected(tmp_path):
    root = _pair()
    extra_ground = replace(root.instances[1], transform=Frame())
    evaluator = compile_placement(
        _load(tmp_path, replace(root, instances=(root.instances[0], extra_ground)), link=_link())
    )
    with pytest.raises(PlacementError, match="joint: position=4"):
        evaluator.evaluate_checked()
    with pytest.raises(ValueError, match="ungrounded instances"):
        compile_placement(
            _load(
                tmp_path,
                replace(root, instances=tuple(replace(i, transform=None) for i in root.instances)),
                link=_link(),
            )
        )


def test_mate_roundtrip_and_schema_gate(tmp_path):
    root = _pair("revolute")
    bundle = _load(tmp_path, root, link=_link())
    assert bundle.root.to_dict() == root.to_dict()
    assert root.to_dict()["schema_version"] == "0.5"
    assert replace(root, mates=()).to_dict()["schema_version"] == "0.5"
    document = root.to_dict()
    document["schema_version"] = "0.4"
    with pytest.raises(
        ValueError, match="expected kind=assembly and a known schema_version >= 0.5"
    ):
        Assembly.from_dict(document)
    with pytest.raises(ValidationError, match="Additional properties"):
        validate(document)
    assert json.loads((tmp_path / "assembly.sdm").read_text())["mates"][0]["kind"] == "revolute"


def test_assembly_reader_accepts_the_newer_compatible_schema():
    document = _pair("revolute").to_dict()
    document["schema_version"] = "0.6"
    assert Assembly.from_dict(document).to_dict()["schema_version"] == "0.5"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"dof": None}, "requires a coordinate"),
        ({"kind": "unknown"}, "expected fixed/revolute/prismatic"),
        ({"child": "base.end"}, "distinct ports"),
    ],
)
def test_invalid_mate_declarations_fail_at_authoring(change, message):
    with pytest.raises(ValueError, match=message):
        replace(_pair("revolute").mates[0], **change)


def test_mate_references_and_coordinate_kinds_are_validated_on_load(tmp_path):
    root = _pair("revolute")
    with pytest.raises(ValueError, match="expected angle DOF"):
        _load(tmp_path, replace(root, dofs={"q": Dof("length", (-1, 1), "mm")}), link=_link())
    with pytest.raises(ValueError, match="Unknown port"):
        _load(
            tmp_path,
            replace(root, mates=(replace(root.mates[0], child="arm.absent"),)),
            link=_link(),
        )


def test_invalid_numerical_states_and_shapes_are_not_accepted(tmp_path):
    evaluator = compile_placement(_load(tmp_path, _pair("revolute"), link=_link()))
    with pytest.raises(ValueError, match="Expected DOF shape"):
        evaluator.evaluate(dofs=[0, 1])
    with pytest.raises(ValueError, match="Expected design shape"):
        evaluator.evaluate(free_vec=[])
    with pytest.raises(PlacementError, match="Invalid placement"):
        evaluator.evaluate_checked(dofs=[float("nan")])
    with pytest.raises(ValueError, match="finite positive tolerance"):
        compile_placement(_load(tmp_path, _pair(), link=_link()), position_tolerance=0)


@pytest.mark.parametrize("degrees", [0, 45, 90], ids=["zero", "quarter_right_angle", "right_angle"])
def test_degrees_are_converted_once_and_promoted_inputs_are_aliases(tmp_path, degrees):
    inner = replace(_pair("revolute"), dofs={"q": Dof("angle", (-180, 180), "deg", degrees)})
    root = Assembly(
        "root",
        instances=(Instance("nested", PartRef("inner.sdm"), transform=Frame()),),
        motion_inputs={"turn": "nested.q"},
    )
    evaluator = compile_placement(_load(tmp_path, root, inner=inner, link=_link()))
    radians = math.radians(degrees)
    assert evaluator.dof_names == ("nested.q",)
    np.testing.assert_allclose(evaluator.dof_defaults, [radians], atol=ATOL)
    for converted in (
        evaluator.to_evaluator_units([degrees]),
        evaluator.to_evaluator_units(None),
        evaluator.to_evaluator_units(),
    ):
        np.testing.assert_allclose(converted, [radians], atol=ATOL)
        state = evaluator.evaluate_checked(dofs=converted)
        np.testing.assert_allclose(
            state.instances["nested.arm"],
            evaluator.evaluate_checked().instances["nested.arm"],
            atol=ATOL,
        )
        np.testing.assert_allclose(
            state.ports["nested.output"][:3, 3],
            [4 + 2 * math.cos(radians), 2 * math.sin(radians), 0],
            atol=ATOL,
        )


def test_mixed_coordinate_units_preserve_defaults_and_convert_only_explicit_degrees(tmp_path):
    root = Assembly(
        "root",
        dofs={
            "angle_degrees": Dof("angle", (-180, 180), "deg", 45),
            "angle_radians": Dof("angle", (-3, 3), "rad", 0.7),
            "travel": Dof("length", (-10, 10), "mm", 2),
        },
    )
    evaluator = compile_placement(_load(tmp_path, root))
    expected = [math.pi / 4, 0.7, 2]
    np.testing.assert_allclose(evaluator.to_evaluator_units(None), expected, atol=ATOL)
    np.testing.assert_allclose(
        jax.jit(evaluator.to_evaluator_units)(jnp.array([45.0, 0.7, 2.0])), expected, atol=ATOL
    )
    with pytest.raises(ValueError, match="Expected DOF shape"):
        evaluator.to_evaluator_units([45])


def test_constant_bound_motion_requires_no_independent_input(tmp_path):
    inner = _pair("revolute")
    root = Assembly(
        "root",
        instances=(
            Instance(
                "nested",
                PartRef("inner.sdm"),
                transform=Frame(),
                dof_bindings={"q": {"type": "num", "value": 1}},
            ),
        ),
    )
    evaluator = compile_placement(_load(tmp_path, root, inner=inner, link=_link()))
    assert evaluator.dof_names == ()
    state = evaluator.evaluate_checked()
    np.testing.assert_allclose(
        state.ports["nested.output"][:3, 3], [4 + 2 * math.cos(1), 2 * math.sin(1), 0], atol=ATOL
    )
    assert float(state.coordinates["nested.q"]) == pytest.approx(1, abs=ATOL)


def test_design_angle_offset_and_end_frame_both_have_live_gradients(tmp_path):
    root = _pair()
    root = replace(
        root,
        params={**root.params, "angle": Param("angle", 0, free=True, unit="rad")},
        mates=(
            replace(
                root.mates[0],
                offset=Frame.from_axis_angle((0, 0, 1), {"type": "param", "name": "angle"}),
            ),
        ),
    )
    evaluator = compile_placement(_load(tmp_path, root, link=_link()))

    def evaluate(design):
        return evaluator.evaluate(free_vec=design).ports["output"][:3, 3]

    # Root binding preserves authored free-parameter order: length then angle.
    design = jnp.array([4.0, 0.4])
    step = 1e-3
    jacobian = jax.jacfwd(evaluate)(design)
    for axis in range(2):
        delta = jnp.eye(2)[axis] * step
        finite = (evaluate(design + delta) - evaluate(design - delta)) / (2 * step)
        np.testing.assert_allclose(jacobian[:, axis], finite, atol=3e-4)
    state = evaluator.evaluate_checked(free_vec=[4, math.pi / 2])
    np.testing.assert_allclose(state.ports["output"][:3, 3], [4, 2, 0], atol=ATOL)


def test_flexure_endpoint_port_places_attached_part_at_supplied_state(tmp_path):
    from tests.test_dynamic_kinematics import _design_part

    flexure = _design_part()
    root = Assembly(
        "root",
        params={"gain": Param("gain", 1, free=True, unit="count")},
        instances=(
            Instance(
                "flexure", PartRef("flexure.sdm"), {"gain": {"$ref": "gain"}}, transform=Frame()
            ),
            Instance("tool", PartRef("link.sdm")),
        ),
        mates=(Mate("attach", "fixed", "flexure.tip", "tool.start"),),
    )
    evaluator = compile_placement(_load(tmp_path, root, flexure=flexure, link=_link()))
    assert evaluator.dof_names == ("flexure.q", "flexure.r")
    prescribed = evaluator.evaluate_checked(free_vec=[2], dofs=[0.3, 0])
    engine_supplied = evaluator.evaluate_checked(free_vec=[2], dofs=np.array([0.3, 0]))
    np.testing.assert_allclose(
        prescribed.instances["tool"], engine_supplied.instances["tool"], atol=ATOL
    )
    np.testing.assert_allclose(
        prescribed.instances["tool"][:3, 3], [2 * np.cos(1.2), 2 * np.sin(1.2), 1], atol=ATOL
    )
    np.testing.assert_allclose(
        prescribed.ports["tool.start"], prescribed.ports["flexure.tip"], atol=ATOL
    )


def test_declaration_order_does_not_change_placement_or_residual_order(tmp_path):
    root = _pair("revolute")
    original = compile_placement(_load(tmp_path, root, link=_link())).evaluate_checked(dofs=[0.8])
    reordered = compile_placement(
        _load(tmp_path, replace(root, instances=tuple(reversed(root.instances))), link=_link())
    ).evaluate_checked(dofs=[0.8])
    for name in original.instances:
        np.testing.assert_allclose(reordered.instances[name], original.instances[name], atol=ATOL)
    np.testing.assert_allclose(reordered.residuals, original.residuals, atol=ATOL)


def test_placement_retains_compiled_snapshot_and_supports_empty_assemblies(tmp_path):
    bundle = _load(tmp_path, _pair(), link=_link())
    evaluator = compile_placement(bundle)
    bundle.root.params["length"].value = 99
    np.testing.assert_allclose(
        evaluator.evaluate_checked().ports["output"][:3, 3], [6, 0, 0], atol=ATOL
    )
    empty = compile_placement(_load(tmp_path, Assembly("empty"))).evaluate_checked()
    assert empty.residuals.shape == (0, 6)
    assert set(empty.instances) == {""}


def test_nonzero_rotated_child_port_is_inverted_without_an_implicit_flip(tmp_path):
    root = _pair()
    child = Part(
        "child",
        ports=[
            Port("start", Frame.from_axis_angle((0, 0, 1), math.pi / 2, position=(1, 0, 0))),
            Port("end"),
        ],
    )
    root = replace(
        root,
        instances=(root.instances[0], replace(root.instances[1], part_ref=PartRef("child.sdm"))),
    )
    state = compile_placement(_load(tmp_path, root, link=_link(), child=child)).evaluate_checked()
    np.testing.assert_allclose(state.instances["arm"][:3, 3], [4, 1, 0], atol=ATOL)
    np.testing.assert_allclose(state.ports["arm.start"], state.ports["base.end"], atol=ATOL)


def test_mate_conformance_bundle_resolves_and_places():
    from pathlib import Path

    corpus = Path(__file__).parents[1] / "src/software_defined_matter/schema/conformance"
    evaluator = compile_placement(load_bundle(corpus / "valid/assembly_mates_0.5.sdm"))
    expected = json.loads((corpus / "placement/assembly_mates_0.5.expected.json").read_text())
    assert list(evaluator.dof_names) == expected["dof_names"]
    assert list(evaluator.dof_units) == expected["authored_units"]
    for case in expected["cases"]:
        coordinates = evaluator.to_evaluator_units(case["authored_dofs"])
        np.testing.assert_allclose(coordinates, case["evaluator_dofs"], atol=ATOL)
        state = evaluator.evaluate_checked(dofs=coordinates)
        for name, matrix in case["instances"].items():
            np.testing.assert_allclose(state.instances[name], matrix, atol=ATOL)
        for name, matrix in case["ports"].items():
            np.testing.assert_allclose(state.ports[name], matrix, atol=ATOL)
        point = expected["landmark_local"]
        world = state.instances["moving"] @ np.array([*point, 1.0])
        np.testing.assert_allclose(world[:3], case["landmark_world"], atol=ATOL)
    default_case = expected["cases"][expected["default_case"]]
    for coordinates in (None, evaluator.to_evaluator_units(None)):
        state = evaluator.evaluate_checked(dofs=coordinates)
        np.testing.assert_allclose(
            state.instances["moving"], default_case["instances"]["moving"], atol=ATOL
        )
    zero = evaluator.evaluate_checked(dofs=[0]).instances["moving"]
    assert not np.allclose(zero, default_case["instances"]["moving"], atol=ATOL)
