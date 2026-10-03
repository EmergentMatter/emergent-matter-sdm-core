"""Assembly instances share definitions but never parameter scopes or inherited constraints."""

from __future__ import annotations

import hashlib
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    Assembly,
    Dof,
    Frame,
    Instance,
    Param,
    Part,
    PartRef,
    Port,
    load_bundle,
    save,
    validate,
)
from software_defined_matter.dsl.resolve import make_binding
from software_defined_matter.io import load, load_part
from software_defined_matter.model import Constraint


def _write(path, document):
    save(document, path)
    return path


def _part():
    return Part(
        "rod",
        params={
            "length": Param("length", 4.0, free=True, unit="mm"),
            "twice": Param(
                "twice",
                8.0,
                unit="mm",
                expr={
                    "type": "binop",
                    "op": "*",
                    "lhs": {"type": "param", "name": "length"},
                    "rhs": {"type": "num", "value": 2.0},
                },
            ),
        },
        ports=[Port("end", Frame(position=(0, 0, {"$ref": "length"})))],
        constraints=[Constraint("minimum", {"type": "param", "name": "length"}, ">=", 1.0)],
    )


def _bundle(tmp_path):
    _write(tmp_path / "rod.sdm", _part())
    wrist = Assembly(
        "wrist",
        params={"size": Param("size", 7.0, free=True, unit="mm")},
        instances=(Instance("rod", PartRef("rod.sdm"), {"length": {"$ref": "size"}}),),
        port={"output": "rod.end"},
        constraints=(Constraint("nested_minimum", {"type": "param", "name": "size"}, ">=", 2.0),),
    )
    _write(tmp_path / "wrist.sdm", wrist)
    assembly = Assembly(
        "actuator",
        params={"length": Param("length", 10.0, free=True, unit="mm")},
        instances=(
            Instance("left", PartRef("wrist.sdm"), {"size": {"$ref": "length"}}),
            Instance("right", PartRef("wrist.sdm"), {"size": 3.0}),
            Instance("default", PartRef("rod.sdm")),
        ),
        port={"tip": "left.output"},
    )
    return load_bundle(_write(tmp_path / "assembly.sdm", assembly))


def test_nested_instances_share_definitions_but_resolve_distinct_values(tmp_path):
    bundle = _bundle(tmp_path)
    assert bundle.definitions["left.rod"] is bundle.definitions["right.rod"]
    vector = bundle.binding().initial_free_vector()
    assert bundle.binding().free_names == ["length"]
    assert float(bundle.binding("left.rod").get("length", vector)) == pytest.approx(10, abs=1e-6)
    assert float(bundle.binding("right.rod").get("length", vector)) == pytest.approx(3, abs=1e-6)
    assert float(bundle.binding("default").get("length", vector)) == pytest.approx(4, abs=1e-6)
    assert float(bundle.binding("left.rod").get("twice", vector)) == pytest.approx(20, abs=1e-6)
    assert bundle.definitions["left.rod"].params["length"].value == 4


def test_promoted_port_frame_tracks_the_root_parameter_under_jit_and_grad(tmp_path):
    bundle = _bundle(tmp_path)
    port = bundle.resolve_port("tip")
    assert port.instance == "left.rod"
    assert bundle.resolve_port("left.rod.end") == port
    fn = jax.jit(lambda v: port.port.frame.evaluate(bundle.binding(port.instance), v)[2, 3])
    assert float(fn(jnp.array([12.0]))) == pytest.approx(12, abs=1e-6)
    assert float(jax.grad(fn)(jnp.array([12.0]))[0]) == pytest.approx(1, abs=1e-6)


def test_constraints_keep_their_local_scope_and_root_expressions_can_cross_instances(tmp_path):
    bundle = _bundle(tmp_path)
    vector = bundle.binding().initial_free_vector()
    values = {
        item.instance: float(
            bundle.evaluate_expression(item.constraint.expr, vector, scope=item.instance)
        )
        for item in bundle.constraints()
    }
    assert values == {"left": 10, "left.rod": 10, "right": 3, "right.rod": 3, "default": 4}
    expression = {"type": "param", "name": "length", "instance": "right.rod"}
    assert float(bundle.evaluate_expression(expression, vector)) == pytest.approx(3, abs=1e-6)
    with pytest.raises(ValueError, match="requires an assembly metric evaluator"):
        bundle.evaluate_expression(
            {"type": "metric", "name": "volume", "instance": "left.rod"}, vector
        )


def test_instance_metric_dispatch_receives_the_originating_scope(tmp_path):
    bundle = _bundle(tmp_path)
    vector = bundle.binding().initial_free_vector()
    calls = []

    def metric(scope, name, args, values):
        calls.append((scope, name, args))
        return bundle.binding(scope).get("length", values)

    result = bundle.evaluate_expression(
        {"type": "metric", "name": "length", "instance": "left.rod"}, vector, metric=metric
    )
    assert float(result) == pytest.approx(10, abs=1e-6)
    assert calls == [("left.rod", "length", {})]


def test_content_hash_is_checked_on_every_reference_even_when_cached(tmp_path):
    path = _write(tmp_path / "rod.sdm", _part())
    pin = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assembly = Assembly(
        "pins",
        instances=(
            Instance("first", PartRef("rod.sdm", pin)),
            Instance("second", PartRef("rod.sdm", "sha256:" + "0" * 64)),
        ),
    )
    with pytest.raises(ValueError, match="second.*content hash mismatch"):
        load_bundle(_write(tmp_path / "assembly.sdm", assembly))


def test_relative_references_resolve_from_the_declaring_document(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    _write(nested / "rod.sdm", _part())
    _write(
        nested / "wrist.sdm", Assembly("wrist", instances=(Instance("rod", PartRef("rod.sdm")),))
    )
    bundle = load_bundle(
        _write(
            tmp_path / "assembly.sdm",
            Assembly("root", instances=(Instance("wrist", PartRef("nested/wrist.sdm")),)),
        )
    )
    assert bundle.sources["wrist.rod"] == nested / "rod.sdm"


def test_recursive_references_report_the_cycle(tmp_path):
    root = Assembly("root", instances=(Instance("self", PartRef("assembly.sdm")),))
    with pytest.raises(ValueError, match="reference cycle.*assembly.sdm"):
        load_bundle(_write(tmp_path / "assembly.sdm", root))


def test_unknown_promotions_and_overrides_fail_during_loading(tmp_path):
    _write(tmp_path / "rod.sdm", _part())
    for assembly, message in [
        (
            Assembly(
                "bad", instances=(Instance("rod", PartRef("rod.sdm")),), port={"end": "rod.missing"}
            ),
            "Unknown port",
        ),
        (
            Assembly("bad", instances=(Instance("rod", PartRef("rod.sdm"), {"missing": 1.0}),)),
            "unknown overridden parameters",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            load_bundle(_write(tmp_path / "assembly.sdm", assembly))


def test_duplicate_instance_ids_and_ambiguous_names_are_rejected():
    occurrence = Instance("rod", PartRef("rod.sdm"))
    with pytest.raises(ValueError, match="unique Instance IDs"):
        Assembly("bad", instances=(occurrence, occurrence))
    with pytest.raises(ValueError, match="ambiguous address"):
        Assembly("bad", instances=(occurrence,), port={"rod": "rod.end"})


def test_missing_parent_parameters_and_relation_cycles_are_rejected():
    with pytest.raises(ValueError, match="undeclared parent parameters"):
        Assembly(
            "bad", instances=(Instance("rod", PartRef("rod.sdm"), {"length": {"$ref": "missing"}}),)
        )
    with pytest.raises(ValueError, match="cycle"):
        Assembly(
            "bad",
            params={"a": Param("a", 1, expr={"$ref": "b"}), "b": Param("b", 1, expr={"$ref": "a"})},
        )


def test_document_loading_dispatches_without_weakening_part_load(tmp_path):
    document = Assembly("empty")
    path = _write(tmp_path / "assembly.sdm", document)
    assert load(path).to_dict() == document.to_dict()
    with pytest.raises(TypeError, match="expected a Part"):
        load_part(path)


def test_legacy_coupling_data_requires_explicit_migration():
    with pytest.raises(ValueError, match="migrate explicitly"):
        Part.from_dict({"name": "old", "couplings": []})


def test_part_port_roundtrip_uses_new_schema_without_changing_simple_parts():
    part = _part()
    doc = part.to_dict()
    assert doc["schema_version"] == "0.5"
    assert "couplings" not in doc
    validate(doc)
    assert Part.from_dict(doc).to_dict() == doc
    assert Part("simple").to_dict()["schema_version"] == "0.2"


def test_parameter_dependent_orientation_preserves_gradients():
    part = Part("rotating", params={"angle": Param("angle", 0.0, free=True, unit="rad")})
    frame = Frame.from_axis_angle((0, 0, 1), {"$ref": "angle"})
    binding = make_binding(part)
    fn = jax.jit(lambda x: frame.evaluate(binding, x)[1, 0])
    assert float(fn(jnp.array([math.pi / 2]))) == pytest.approx(1, abs=2e-6)
    assert float(jax.grad(fn)(jnp.array([0.0]))[0]) == pytest.approx(1, abs=2e-6)


@pytest.mark.parametrize(
    "normal,tangent",
    [
        ((0, 0, 1), (1, 0, 0)),
        ((0, 0, -1), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((0, -1, 0), (-1, 0, 0)),
    ],
    ids=["identity", "half_turn", "x_normal", "negative_axes"],
)
def test_normal_tangent_helper_preserves_the_authored_axes(normal, tangent):
    frame = Frame.from_normal_tangent(normal, tangent)
    matrix = np.asarray(frame.evaluate(make_binding(Part("empty")), jnp.zeros(0)))
    np.testing.assert_allclose(matrix[:3, 2], normal, atol=2e-6)
    np.testing.assert_allclose(matrix[:3, 0], tangent, atol=2e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"orientation": (0, 0, 0, 0)},
        {"position": (0, 0, float("nan"))},
        {"position": (0, 0, True)},
        {"position": (0, 0, {"type": "metric", "name": "volume"})},
    ],
    ids=["zero_quaternion", "nan", "bool", "metric"],
)
def test_invalid_frames_are_rejected(kwargs):
    with pytest.raises(ValueError, match="Frame"):
        Frame(**kwargs)


def test_ports_reject_unknown_body_and_frame_parameters():
    with pytest.raises(ValueError, match="unknown body"):
        Part("bad", ports=[Port("end", body="missing")])
    with pytest.raises(ValueError, match="undeclared parameters"):
        Part("bad", ports=[Port("end", Frame(position=(0, 0, {"$ref": "missing"})))])
    with pytest.raises(ValueError, match="not parallel"):
        Frame.from_normal_tangent((0, 0, 1), (0, 0, 2))


def test_nested_motion_promotions_resolve_and_conflicting_drivers_fail(tmp_path):
    leaf = Assembly("leaf", dofs={"angle": Dof("angle", (-1, 1), "rad")})
    _write(tmp_path / "leaf.sdm", leaf)
    child = Assembly(
        "child",
        instances=(Instance("leaf", PartRef("leaf.sdm")),),
        motion_inputs={"rotation": "leaf.angle"},
    )
    _write(tmp_path / "child.sdm", child)
    root = Assembly(
        "root",
        dofs={"drive": Dof("angle", (-1, 1), "rad")},
        instances=(
            Instance(
                "child",
                PartRef("child.sdm"),
                dof_bindings={"rotation": {"type": "dof", "name": "drive"}},
            ),
        ),
    )
    path = _write(tmp_path / "root.sdm", root)
    assert load_bundle(path).resolve_dof("child.rotation") == ("child.leaf", "angle")
    data = json.loads(path.read_text())
    data["instances"][0]["dof_bindings"]["leaf.angle"] = {"type": "dof", "name": "drive"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="conflicting drivers"):
        load_bundle(path)


def test_motion_binding_cycles_are_rejected(tmp_path):
    _write(tmp_path / "leaf.sdm", Assembly("leaf", dofs={"angle": Dof("angle", (-1, 1), "rad")}))
    root = Assembly(
        "root",
        instances=(
            Instance(
                "leaf",
                PartRef("leaf.sdm"),
                dof_bindings={"angle": {"type": "dof", "name": "rotation"}},
            ),
        ),
        motion_inputs={"rotation": "leaf.angle"},
    )
    with pytest.raises(ValueError, match="cycle"):
        load_bundle(_write(tmp_path / "root.sdm", root))


def test_instance_binding_compiles_sdf_without_freezing_root_parameters(tmp_path):
    from software_defined_matter.model import sdf_primitive
    from software_defined_matter.sdf.compile import make_sdf_closure_with_binding

    bundle = _bundle(tmp_path)
    sdf = make_sdf_closure_with_binding(
        sdf_primitive("sphere", r={"$ref": "length"}), bundle.binding("left.rod")
    )
    points = jnp.zeros((1, 3))
    assert float(sdf(points)[0]) == pytest.approx(-10, abs=1e-6)
    assert float(jax.grad(lambda v: sdf(points, v)[0])(jnp.array([12.0]))[0]) == pytest.approx(
        -1, abs=1e-6
    )


def test_expression_orientation_roundtrips_through_schema():
    part = Part(
        "tilt",
        params={"angle": Param("angle", 0.0, free=True, unit="rad")},
        ports=[Port("mount", Frame.from_axis_angle((1, 0, 0), {"$ref": "angle"}))],
    )
    validate(part)
    assert Part.from_dict(part.to_dict()).to_dict() == part.to_dict()


def test_binding_rejects_incompatible_parameter_units(tmp_path):
    _write(tmp_path / "rod.sdm", _part())
    root = Assembly(
        "bad",
        params={"angle": Param("angle", 1.0, unit="rad")},
        instances=(Instance("rod", PartRef("rod.sdm"), {"length": {"$ref": "angle"}}),),
    )
    with pytest.raises(ValueError, match="binding unit.*does not match"):
        load_bundle(_write(tmp_path / "root.sdm", root))


def test_actuator_example_is_a_valid_bundle_with_shared_fasteners(tmp_path):
    from examples.build_assembly_example import build_assembly_bundle

    bundle = load_bundle(build_assembly_bundle(tmp_path))
    assert bundle.resolve_port("output").instance == "rotor.shaft"
    assert bundle.definitions["bolt_left"] is bundle.definitions["bolt_right"]
    assert bundle.binding().free_names == ["length", "bolt_radius"]
