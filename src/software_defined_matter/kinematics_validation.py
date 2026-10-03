"""Validate motion references before a preserved kinematics block is saved.

JSON Schema checks shape; these checks bind motion names to the document and
reject degenerate motion axes and interpolation intervals.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["validate_kinematics"]


def validate_kinematics(doc: dict[str, Any]) -> None:
    """Reject ambiguous names, dangling references, and invalid motion ranges."""
    block = doc.get("kinematics")
    if block is None:
        return

    def names(records: list[dict[str, Any]], kind: str) -> set[str]:
        result: set[str] = set()
        for record in records:
            name = record["name"]
            if name in result:
                raise ValueError(f"kinematics: duplicate {kind} name {name!r}")
            result.add(name)
        return result

    dofs = names(block.get("dofs", []), "DOF")
    bodies = names(block.get("bodies", []), "body")
    flexures = names(block.get("flexures", []), "flexure")
    if bodies & flexures:
        raise ValueError("kinematics: body and flexure names must be distinct")
    node_names: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "type" in value and "name" in value:
                node_names.add(value["name"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(doc.get("materials", []))

    def references(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "dof" and value.get("name") not in dofs:
                raise ValueError(f"kinematics: unknown DOF {value.get('name')!r}")
            if "$node" in value and value["$node"] not in node_names:
                raise ValueError(f"kinematics: unknown node {value['$node']!r}")
            for child in value.values():
                references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)

    references(block)
    for dof in block.get("dofs", []):
        lo, hi = dof["range"]
        default = dof.get("default", 0)
        if not all(math.isfinite(v) for v in (lo, hi, default)) or not lo <= default <= hi:
            raise ValueError(
                f"kinematics: DOF {dof['name']!r} requires a finite ordered range "
                "containing its default"
            )
    for body in block.get("bodies", []):
        for op in body["motion"]["ops"]:
            axis = op["axis"]
            if not all(math.isfinite(v) for v in axis) or sum(v * v for v in axis) == 0:
                raise ValueError(
                    f"kinematics: body {body['name']!r} requires a finite nonzero axis"
                )
    for flexure in block.get("flexures", []):
        for key in ("from_body", "to_body"):
            if flexure[key] not in bodies:
                raise ValueError(
                    f"kinematics: unknown body {flexure[key]!r} in {flexure['name']!r}"
                )
        blend = flexure["blend"]
        if blend.get("kind") == "axis_ramp":
            params = blend.get("params", {})
            axis = params.get("axis")
            values = [*(axis if isinstance(axis, list) else []), params.get("lo"), params.get("hi")]
            if (
                not isinstance(axis, list)
                or len(axis) != 3
                or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in values
                )
                or not params["lo"] < params["hi"]
                or sum(v * v for v in axis) == 0
            ):
                raise ValueError("kinematics: axis_ramp requires a finite nonzero axis and lo < hi")

        if blend.get("kind") == "radial_hermite":
            params = blend.get("params", {})
            axis, origin = params.get("axis"), params.get("origin")
            if not (
                isinstance(axis, list)
                and len(axis) == 3
                and isinstance(origin, list)
                and len(origin) == 3
                and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in [*axis, *origin, params.get("r0"), params.get("r1")]
                )
                and 0 < math.hypot(*axis) < math.inf
                and 0 <= params["r0"] < params["r1"]
            ):
                raise ValueError(
                    "kinematics: radial_hermite requires a finite axis and origin, "
                    "a nonzero axis, and 0 <= r0 < r1"
                )
