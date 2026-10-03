"""Wire, schema, and integrity contract for ``Param.expr``."""

from __future__ import annotations

import jsonschema
import pytest

from software_defined_matter import Param, Part, validate
from software_defined_matter.dsl.expr import expr_binop, expr_num, expr_param
from software_defined_matter.sdf.param_refs import (
    ParamRelationError,
    UndeclaredParamRefError,
)


def _doc(params: dict) -> dict:
    return {
        "schema_version": "0.3",
        "name": "relations",
        "params": params,
        "materials": [],
    }


def _param(name: str, value: float, *, expr: dict | None = None) -> dict:
    out = {"name": name, "value": value, "free": False, "bounds": None, "unit": "mm"}
    if expr is not None:
        out["expr"] = expr
    return out


def test_param_expr_round_trips_with_cached_value():
    expr = expr_binop("/", expr_param("width"), expr_num(2))
    part = Part(
        "p",
        params={
            "width": Param("width", 10, unit="mm"),
            "half": Param("half", 5, unit="mm", expr=expr),
        },
    )
    doc = part.to_dict()
    assert doc["schema_version"] == "0.3"
    assert doc["params"]["half"]["expr"] == expr
    assert Part.from_dict(doc).params["half"].expr == expr


def test_expr_and_free_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Param("half", 5, free=True, unit="mm", expr=expr_num(5))


def test_relation_without_cached_value_cannot_serialize_before_evaluation():
    with pytest.raises(ValueError, match="no cached value"):
        Param("half", unit="mm", expr=expr_num(5)).to_dict()


def test_schema_accepts_relation_and_ui_rebuild_axis():
    doc = _doc(
        {
            "width": {
                **_param("width", 10),
                "ui": {"rebuild": True, "axis": "radial"},
            },
            "half": _param("half", 5, expr=expr_param("width")),
        }
    )
    validate(doc)


@pytest.mark.parametrize("axis", ["diagonal", "", 7])
def test_schema_rejects_unknown_ui_axis(axis):
    doc = _doc({"width": {**_param("width", 10), "ui": {"axis": axis}}})
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_relation_unknown_reference_is_rejected_with_site():
    doc = _doc({"half": _param("half", 5, expr=expr_param("missing"))})
    with pytest.raises(UndeclaredParamRefError, match=r"missing.*relation of param 'half'"):
        validate(doc)


def test_relation_cycle_is_rejected_with_cycle_path():
    doc = _doc(
        {
            "a": _param("a", 1, expr=expr_param("b")),
            "b": _param("b", 1, expr=expr_param("a")),
        }
    )
    with pytest.raises(ParamRelationError, match=r"cycle: (a -> b -> a|b -> a -> b)"):
        validate(doc)


def test_relation_metric_is_rejected():
    doc = _doc({"a": _param("a", 1, expr={"type": "metric", "name": "volume", "args": {}})})
    with pytest.raises(ParamRelationError, match="metric"):
        validate(doc)


def test_constraint_and_animation_unknown_refs_are_rejected():
    doc = _doc({"x": _param("x", 1)})
    doc["constraints"] = [
        {
            "name": "bad",
            "expr": expr_param("constraint_missing"),
            "op": ">=",
            "rhs": 0,
        }
    ]
    doc["metadata"] = {"animations": [{"name": "bad_anim", "tracks": [{"param": "track_missing"}]}]}
    with pytest.raises(UndeclaredParamRefError) as exc:
        validate(doc)
    assert "constraint_missing" in str(exc.value)
    assert "track_missing" in str(exc.value)
