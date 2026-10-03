"""The `Param.ui` contract must agree across schema versions and with its docs.

Three separate things describe `ui`, and they had all drifted apart:

* `Param.ui`'s docstring: the contract authors read
* `sdm-0.1.schema.json`: what old parts validate against
* `sdm-0.2.schema.json`: what new parts validate against

`0.2` had silently DROPPED `collapsed`, `driven` and `choices`, which `0.1` has
and the docstring documents. Because `ui` is `additionalProperties: false`, any
part that used them would have started failing the moment it moved to 0.2. It had
not bitten yet only because parts still declare 0.1.

And neither version accepted `role: "pose"`, which a downstream viewer project
depends on in two places (preserving pose controls across a rebuild; widening a
bounds-mode bbox to the pose envelope). A part declaring it correctly failed
validation; a part that validated lost its pose on every rebuild.

These tests pin all three descriptions to each other so the next divergence is a
test failure rather than a bug report from a CEM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    io,
    sdf_primitive,
)
from software_defined_matter.model import Param as ParamClass

SCHEMA_DIR = Path(io.__file__).resolve().parent / "schema"
VERSIONS = ("sdm-0.1.schema.json", "sdm-0.2.schema.json")

# Every key the docstring promises. `bounds` appears in the prose describing how
# `explore_bounds` differs from the optimiser box, so it is not a ui key.
DOCUMENTED = {
    "step",
    "explore_bounds",
    "group",
    "order",
    "role",
    "collapsed",
    "driven",
    "choices",
}
ROLES = {"topology", "pose"}


def _find_ui(node):
    """The `ui` sub-schema, wherever it sits in the document."""
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict) and {"role", "group"} <= set(props):
            return node
        for value in node.values():
            found = _find_ui(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_ui(value)
            if found:
                return found
    return None


@pytest.fixture(params=VERSIONS)
def ui_schema(request):
    schema = json.loads((SCHEMA_DIR / request.param).read_text())
    ui = _find_ui(schema)
    assert ui is not None, f"no ui block in {request.param}"
    return request.param, ui


def test_every_schema_version_allows_the_documented_keys(ui_schema):
    name, ui = ui_schema
    allowed = set(ui["properties"])
    missing = DOCUMENTED - allowed
    assert not missing, (
        f"{name} rejects documented ui keys {sorted(missing)}, and because ui is "
        f"additionalProperties:false, a part using them fails validation"
    )


def test_schema_versions_agree_on_ui(ui_schema):
    """0.2 must not be NARROWER than 0.1. Moving a part forward should never
    invalidate a key that already worked."""
    name, ui = ui_schema
    reference = _find_ui(json.loads((SCHEMA_DIR / VERSIONS[0]).read_text()))
    assert set(reference["properties"]) <= set(ui["properties"]), (
        f"{name} dropped ui keys present in {VERSIONS[0]}: "
        f"{sorted(set(reference['properties']) - set(ui['properties']))}"
    )


def test_role_enum_covers_every_role_consumers_use(ui_schema):
    name, ui = ui_schema
    assert set(ui["properties"]["role"]["enum"]) >= ROLES, (
        f"{name} role enum {ui['properties']['role']['enum']} is missing "
        f"{sorted(ROLES - set(ui['properties']['role']['enum']))}"
    )


def test_docstring_documents_every_schema_key(ui_schema):
    """The reverse direction: a key the schema accepts but nobody documented is
    a key no author will ever use."""
    name, ui = ui_schema
    undocumented = set(ui["properties"]) - DOCUMENTED
    assert not undocumented, (
        f"{name} allows undocumented ui keys {sorted(undocumented)}; add them to "
        f"Param.ui's docstring (and to DOCUMENTED here)"
    )


def test_param_ui_docstring_mentions_both_roles():
    doc = ParamClass.__doc__ or ""
    body = doc[doc.find("ui (dict") :]
    for role in ROLES:
        assert f'"{role}"' in body or f"``{role}``" in body, (
            f"Param.ui's docstring does not document role {role!r}"
        )


@pytest.mark.parametrize("role", sorted(ROLES))
def test_a_part_using_each_role_round_trips(tmp_path, role):
    """End to end: the thing that was actually broken. A pose param must save,
    validate and reload."""
    part = Part(
        name="ui_roles",
        params={
            "tilt": Param(
                "tilt",
                0.0,
                free=True,
                bounds=(-1.0, 1.0),
                unit="rad",
                ui={"role": role, "explore_bounds": [-1.0, 1.0], "group": "Pose", "order": 1},
            ),
        },
        materials=[
            MaterialRegion(material_id=1, name="m", sdf_tree=sdf_primitive("sphere", r=1.0))
        ],
    )
    path = tmp_path / "p.sdm"
    io.save(part, path)  # validates on the way out
    back = io.load(path)
    assert (back.params["tilt"].ui or {})["role"] == role


def test_a_part_using_driven_and_collapsed_round_trips(tmp_path):
    """0.2 dropped both. A CEM that marks derived read-outs read-only and folds
    them (which is the whole point of a large driven group) must survive."""
    part = Part(
        name="ui_driven",
        params={
            "socket_r": Param(
                "socket_r",
                25.0,
                free=False,
                unit="mm",
                ui={"driven": True, "collapsed": True, "group": "Driven"},
            ),
        },
        materials=[
            MaterialRegion(material_id=1, name="m", sdf_tree=sdf_primitive("sphere", r=1.0))
        ],
    )
    path = tmp_path / "p.sdm"
    io.save(part, path)
    back = io.load(path)
    ui = back.params["socket_r"].ui or {}
    assert ui["driven"] is True and ui["collapsed"] is True


def test_an_unknown_role_is_still_rejected(tmp_path):
    """Widening the enum must not turn it into a free-for-all: a typo should
    still fail loudly."""
    part = Part(
        name="bad_role",
        params={"x": Param("x", 1.0, free=False, unit="mm", ui={"role": "poze"})},
        materials=[
            MaterialRegion(material_id=1, name="m", sdf_tree=sdf_primitive("sphere", r=1.0))
        ],
    )
    with pytest.raises(Exception, match="poze|enum|not one of"):
        io.save(part, tmp_path / "p.sdm")
