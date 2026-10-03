"""The bundled notch-hinge kinematics example validates and evaluates."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from build_notch_hinge_kinematics import build_notch_hinge  # type: ignore[import-not-found]

from software_defined_matter import min_schema_version_for, validate
from software_defined_matter.io import save
from software_defined_matter.kinematics import compile_kinematics
from software_defined_matter.sdf.compile import make_sdf_closure

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "notch_hinge_kinematics.sdm"


def _named_nodes(tree: object) -> set[str]:
    names: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("name"), str) and "type" in node:
                names.add(node["name"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(tree)
    return names


def test_builder_declares_schema_0_6():
    part = build_notch_hinge()
    validate(part)
    assert min_schema_version_for(part) == "0.6"


def test_committed_example_matches_the_builder(tmp_path):
    path = tmp_path / "notch_hinge_kinematics.sdm"
    save(build_notch_hinge(), path)
    assert json.loads(path.read_text()) == json.loads(EXAMPLE.read_text())


def test_example_regions_resolve_and_form_a_serial_chain():
    doc = json.loads(EXAMPLE.read_text())
    validate(doc)

    names = set()
    for material in doc["materials"]:
        names |= _named_nodes(material["sdf_tree"])

    kin = doc["kinematics"]
    regions = [body["region"] for body in kin["bodies"]] + [
        flexure["region"] for flexure in kin["flexures"]
    ]
    for region in regions:
        if "$node" in region:
            assert region["$node"] in names, f"unresolved $node {region['$node']!r}"

    assert kin["bodies"][0]["motion"]["ops"] == []
    assert len(kin["flexures"]) == len(kin["bodies"]) - 1


def test_neck_and_blocks_evaluate_and_grads_are_finite():
    part = build_notch_hinge()
    fn = make_sdf_closure(part.materials[0].sdf_tree, part)
    points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [-12.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
        ]
    )
    distances = np.asarray(fn(points))
    np.testing.assert_allclose(distances, [-1.0, -4.0, -4.0, 0.5], atol=1e-5)

    def d_neck(free_vec):
        return fn(jnp.array([[0.0, 1.5, 0.0]]), free_vec)[0]

    grad = np.asarray(jax.grad(d_neck)(jnp.asarray(part.param_vector(), dtype=jnp.float32)))
    assert np.isfinite(grad).all()
    assert np.any(np.abs(grad) > 1e-6)


def test_compile_kinematics_vanishes_at_rest():
    part = build_notch_hinge()
    ev = compile_kinematics(part)
    assert ev is not None
    rest = jnp.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
    posed = ev.pose_points(rest, jnp.zeros(len(ev.dof_names)))
    np.testing.assert_allclose(np.asarray(posed), np.asarray(rest), atol=1e-5)
