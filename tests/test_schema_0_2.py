"""Schema 0.2 (kinematics block): positive + negative validation.

Focuses on the DOF contract that viewers rely on for auto-built animations:
every ``dof`` must carry ``range`` and a strict ``unit``: a DOF without a
range gives an animation auto-builder nothing to sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "software_defined_matter"
    / "schema"
    / "sdm-0.2.schema.json"
)


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text())


def _doc(dof):
    return {
        "schema_version": "0.2",
        "name": "t",
        "kinematics": {
            "dofs": [dof],
            "bodies": [],
            "flexures": [],
        },
    }


GOOD_DOF = {
    "name": "twist",
    "kind": "angle",
    "range": [-20.0, 20.0],
    "default": 0.0,
    "rate": 10.0,
    "unit": "deg",
}


def test_dof_with_range_rate_unit_validates(schema):
    jsonschema.validate(_doc(GOOD_DOF), schema)


def test_dof_requires_range(schema):
    dof = {k: v for k, v in GOOD_DOF.items() if k != "range"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_doc(dof), schema)


def test_dof_requires_unit(schema):
    dof = {k: v for k, v in GOOD_DOF.items() if k != "unit"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_doc(dof), schema)


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD_DOF, "unit": "degrees"},  # strict unit vocabulary
        {**GOOD_DOF, "rate": 0},  # rate must be > 0
        {**GOOD_DOF, "range": [1.0]},  # range needs [lo, hi]
    ],
)
def test_dof_negative_cases(schema, bad):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_doc(bad), schema)


def test_param_def_mirrors_0_1_strictness(schema):
    """0.2 must stay a superset of 0.1: unit required + enum, ui block present."""
    param = schema["$defs"]["param"]
    assert "unit" in param["required"]
    assert "enum" in param["properties"]["unit"]
    assert "ui" in param["properties"]


# ---------------------------------------------------------------------------
# Vocabulary parity with 0.1 (caught during schema review as a drift risk)
# ---------------------------------------------------------------------------


def _schema_01():
    p = SCHEMA_PATH.parent / "sdm-0.1.schema.json"
    return json.loads(p.read_text())


def test_vocabulary_tracks_0_1(schema):
    """0.2's shared sdf/field/expr vocabulary must equal current 0.1's:
    0.2 extends the format, it must never silently re-open what 0.1 closed.
    Allowed additive diffs only: `name` on sdf nodes, the `dof` expr branch."""
    s01 = _schema_01()
    assert schema["$defs"]["field"] == s01["$defs"]["field"], "field vocabulary drifted"

    expr02 = json.loads(json.dumps(schema["$defs"]["expr"]))
    expr02["oneOf"] = [
        b for b in expr02["oneOf"] if b.get("properties", {}).get("type", {}).get("const") != "dof"
    ]
    assert expr02 == s01["$defs"]["expr"], "expr vocabulary drifted beyond +dof"
    sdf02 = json.loads(json.dumps(schema["$defs"]["sdf"]))
    assert sdf02.get("properties", {}).pop("name", None) is not None
    if not sdf02.get("properties"):
        sdf02.pop("properties", None)
    assert sdf02 == s01["$defs"]["sdf"], "sdf vocabulary drifted beyond +name"


# ---------------------------------------------------------------------------
# Named nodes + $node regions (views, not instances)
# ---------------------------------------------------------------------------


def test_named_sdf_node_validates(schema):
    doc = {
        "schema_version": "0.2",
        "name": "t",
        "materials": [
            {
                "material_id": 1,
                "name": "PA12",
                "sdf_tree": {
                    "type": "primitive",
                    "kind": "sphere",
                    "name": "ball",
                    "params": {"r": 1.0},
                },
            }
        ],
    }
    jsonschema.validate(doc, schema)


def test_region_by_node_ref_validates(schema):
    doc = _doc(GOOD_DOF)
    doc["kinematics"]["bodies"] = [
        {
            "name": "ring_0",
            "region": {"$node": "ring_0_geo"},
            "motion": {"ops": []},
        }
    ]
    jsonschema.validate(doc, schema)


def test_region_inline_sdf_still_validates(schema):
    doc = _doc(GOOD_DOF)
    doc["kinematics"]["bodies"] = [
        {
            "name": "ring_0",
            "region": {"type": "primitive", "kind": "sphere", "params": {"r": 1.0}},
            "motion": {"ops": []},
        }
    ]
    jsonschema.validate(doc, schema)


@pytest.mark.parametrize(
    "bad_region",
    [
        {"$node": ""},  # empty target
        {"$node": "x", "extra": 1},  # stray keys rejected
        {"node": "x"},  # wrong key
    ],
)
def test_region_negative_cases(schema, bad_region):
    doc = _doc(GOOD_DOF)
    doc["kinematics"]["bodies"] = [
        {
            "name": "b",
            "region": bad_region,
            "motion": {"ops": []},
        }
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)
