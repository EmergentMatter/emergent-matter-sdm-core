"""Motion blocks survive model and file round trips without aliasing input data."""

from __future__ import annotations

import copy

import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive
from software_defined_matter.io import load, save, validate


def _part() -> Part:
    tree = sdf_primitive("sphere", r=1.0)
    tree["name"] = "ball"
    return Part(
        name="motion",
        materials=[MaterialRegion(material_id=1, name="body", sdf_tree=tree)],
        kinematics={
            "dofs": [
                {"name": "angle", "kind": "angle", "range": [-1, 1], "default": 0, "unit": "rad"}
            ],
            "bodies": [
                {
                    "name": "ball",
                    "region": {"$node": "ball"},
                    "motion": {
                        "ops": [
                            {
                                "kind": "rotate",
                                "axis": [0, 0, 1],
                                "angle": {"type": "dof", "name": "angle"},
                            }
                        ]
                    },
                }
            ],
            "flexures": [
                {
                    "name": "blend",
                    "region": {"$node": "ball"},
                    "from_body": "ball",
                    "to_body": "ball",
                    "blend": {
                        "type": "field",
                        "kind": "axis_ramp",
                        "params": {"axis": [0, 0, 1], "lo": -1, "hi": 1},
                    },
                }
            ],
        },
    )


def test_motion_block_survives_validated_file_roundtrip(tmp_path):
    part = _part()
    target = tmp_path / "motion.sdm"
    save(part, target)
    assert load(target).kinematics == part.kinematics
    assert part.to_dict()["schema_version"] == "0.3"


def test_motion_input_and_serialized_output_are_independent():
    original = _part().to_dict()
    part = Part.from_dict(original)
    original["kinematics"]["dofs"][0]["default"] = 0.5
    result = part.to_dict()
    result["kinematics"]["dofs"][0]["default"] = -0.5
    assert part.kinematics["dofs"][0]["default"] == 0
    assert "kinematics" not in Part(name="empty").to_dict()


@pytest.mark.parametrize("defect", ["dof", "body", "node", "range", "axis", "blend"], ids=str)
def test_invalid_motion_contract_is_rejected(defect):
    doc = copy.deepcopy(_part().to_dict())
    block = doc["kinematics"]
    if defect == "dof":
        block["bodies"][0]["motion"]["ops"][0]["angle"]["name"] = "missing"
    elif defect == "body":
        block["flexures"][0]["to_body"] = "missing"
    elif defect == "node":
        block["bodies"][0]["region"]["$node"] = "missing"
    elif defect == "range":
        block["dofs"][0]["range"] = [1, -1]
    elif defect == "axis":
        block["bodies"][0]["motion"]["ops"][0]["axis"] = [0, 0, 0]
    else:
        block["flexures"][0]["blend"]["params"]["hi"] = -1
    with pytest.raises(ValueError, match="kinematics:"):
        validate(doc)
