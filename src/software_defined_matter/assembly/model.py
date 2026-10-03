"""Assembly declarations independent of file resolution and numerical placement.

Definitions retain local scopes. Instance overrides read their parent's scope;
port and motion promotions expose child addresses without copying definitions.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from software_defined_matter.model import (
    KNOWN_SCHEMA_VERSIONS,
    Constraint,
    Objective,
    Param,
    Part,
    _parse_version,
)
from software_defined_matter.ports import Frame, scalar_refs, validate_identifier, validate_scalar

__all__ = ["Assembly", "Dof", "Instance", "Mate", "PartRef"]


def _keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"{where}: unknown fields {sorted(extra)}")


def _address(value: str, where: str) -> None:
    if not isinstance(value, str) or len(value.split(".")) < 2:
        raise ValueError(f"{where}: expected a dotted child address, got {value!r}")
    for segment in value.split("."):
        validate_identifier(segment, where)


@dataclass(frozen=True)
class PartRef:
    """Relative document reference, optionally pinned to sha256 of its exact bytes."""

    path: str
    content_hash: str | None = None

    def __post_init__(self) -> None:
        import re

        if (
            not isinstance(self.path, str)
            or not self.path
            or PurePosixPath(self.path).is_absolute()
            or "\\" in self.path
            or ":" in self.path
        ):
            raise ValueError(f"PartRef.path: expected a relative POSIX path, got {self.path!r}")
        if self.content_hash is not None and (
            not isinstance(self.content_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.content_hash) is None
        ):
            raise ValueError(
                "PartRef.content_hash: expected sha256:<64 lowercase hex digits>, "
                f"got {self.content_hash!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the reference without materializing its target."""
        return {
            "path": self.path,
            **({"content_hash": self.content_hash} if self.content_hash else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PartRef:
        """Read a strict relative reference."""
        _keys(data, {"path", "content_hash"}, "PartRef")
        return cls(**data)


@dataclass(frozen=True)
class Dof:
    """Runtime coordinate, distinct from a design parameter and bounded in authored units."""

    kind: str
    range: tuple[float, float]
    unit: str
    default: float = 0.0

    def __post_init__(self) -> None:
        expected = {"angle": {"rad", "deg"}, "length": {"mm"}}
        if self.kind not in expected or self.unit not in expected[self.kind]:
            raise ValueError(
                "Dof.kind/unit: expected angle with rad/deg or length with mm, "
                f"got {self.kind!r}/{self.unit!r}"
            )
        if not isinstance(self.range, (tuple, list)) or len(self.range) != 2:
            raise ValueError(f"Dof.range: expected two finite ordered numbers, got {self.range!r}")
        values = [*self.range, self.default]
        if (
            any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in values
            )
            or not self.range[0] <= self.default <= self.range[1]
            or self.range[0] >= self.range[1]
        ):
            raise ValueError(
                "Dof.range/default: expected finite lo <= default <= hi with lo < hi, "
                f"got {values!r}"
            )
        object.__setattr__(self, "range", tuple(self.range))

    def to_dict(self) -> dict[str, Any]:
        """Return an unnamed DOF record for a name-keyed assembly map."""
        return {
            "kind": self.kind,
            "range": list(self.range),
            "unit": self.unit,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dof:
        """Read a bounded coordinate declaration."""
        _keys(data, {"kind", "range", "unit", "default"}, "Dof")
        return cls(**data)


@dataclass(frozen=True)
class Instance:
    """Occurrence of a definition, with parent-scoped parameter and motion bindings."""

    id: str
    part_ref: PartRef
    param_overrides: dict[str, Any] = field(default_factory=dict)
    dof_bindings: dict[str, Any] = field(default_factory=dict)
    transform: Frame | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.id, "Instance.id")
        if not isinstance(self.part_ref, PartRef):
            raise ValueError(f"Instance {self.id!r}.part_ref: expected PartRef")
        for name in ("param_overrides", "dof_bindings"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"Instance {self.id!r}.{name}: expected a mapping")
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        for key, value in self.param_overrides.items():
            if not isinstance(key, str):
                raise ValueError(f"Instance {self.id!r}.param_overrides: expected parameter names")
            if isinstance(value, dict) and set(value) != {"$ref"}:
                raise ValueError(
                    f"Instance {self.id!r}.param_overrides[{key!r}]: "
                    "expected literal or parent $ref"
                )
            validate_scalar(value, f"Instance {self.id!r}.param_overrides[{key!r}]")
        if self.transform is not None and (
            not isinstance(self.transform, Frame) or scalar_refs(self.transform.to_dict())
        ):
            raise ValueError(f"Instance {self.id!r}.transform: expected a literal grounding Frame")

    def to_dict(self) -> dict[str, Any]:
        """Return bindings and references without copying target documents."""
        return {
            "id": self.id,
            "part_ref": self.part_ref.to_dict(),
            "param_overrides": copy.deepcopy(self.param_overrides),
            "dof_bindings": copy.deepcopy(self.dof_bindings),
            **({"transform": self.transform.to_dict()} if self.transform is not None else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Instance:
        """Read an instance, rejecting unknown placement or binding keys."""
        _keys(data, {"id", "part_ref", "param_overrides", "dof_bindings", "transform"}, "Instance")
        return cls(
            data["id"],
            PartRef.from_dict(data["part_ref"]),
            data.get("param_overrides", {}),
            data.get("dof_bindings", {}),
            Frame.from_dict(data["transform"]) if "transform" in data else None,
        )


@dataclass(frozen=True)
class Mate:
    """Port-to-port placement: child = parent @ offset @ joint(q).

    Frames map column vectors from port coordinates into their owner's coordinates.
    ``offset`` is expressed in the parent port frame and may use assembly parameters.
    Fixed mates have no coordinate. Revolute and prismatic mates require ``dof``
    to name an angle or length coordinate, respectively. The joint acts along local
    +Z after the offset: right-handed radians for rotation, millimetres for travel.
    There is no implicit normal reversal; author a half-turn offset when needed.
    Parent and child are port addresses relative to the containing assembly.
    """

    id: str
    kind: str
    parent: str
    child: str
    dof: str | None = None
    offset: Frame = field(default_factory=Frame)

    def __post_init__(self) -> None:
        validate_identifier(self.id, "Mate.id")
        if self.kind not in {"fixed", "revolute", "prismatic"}:
            raise ValueError(
                f"Mate {self.id!r}.kind: expected fixed/revolute/prismatic, got {self.kind!r}"
            )
        for name in ("parent", "child"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Mate {self.id!r}.{name}: expected a port address, got {value!r}")
            for segment in value.split("."):
                validate_identifier(segment, f"Mate {self.id!r}.{name}")
        if self.parent == self.child:
            raise ValueError(f"Mate {self.id!r}: parent and child must name distinct ports")
        if self.kind == "fixed":
            if self.dof is not None:
                raise ValueError(f"Mate {self.id!r}.dof: a fixed mate cannot consume a DOF")
        elif not isinstance(self.dof, str) or not self.dof:
            raise ValueError(f"Mate {self.id!r}.dof: {self.kind} requires a coordinate")
        if self.dof is not None:
            for segment in self.dof.split("."):
                validate_identifier(segment, f"Mate {self.id!r}.dof")
        if not isinstance(self.offset, Frame):
            raise ValueError(f"Mate {self.id!r}.offset: expected Frame, got {self.offset!r}")
        object.__setattr__(self, "offset", copy.deepcopy(self.offset))

    def to_dict(self) -> dict[str, Any]:
        """Return the placement declaration without evaluating its frames."""
        return {
            "id": self.id,
            "kind": self.kind,
            "parent": self.parent,
            "child": self.child,
            "offset": self.offset.to_dict(),
            **({"dof": self.dof} if self.dof is not None else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mate:
        """Read a strict placement declaration."""
        _keys(data, {"id", "kind", "parent", "child", "dof", "offset"}, "Mate")
        return cls(
            data["id"],
            data["kind"],
            data["parent"],
            data["child"],
            data.get("dof"),
            Frame.from_dict(data["offset"]) if "offset" in data else Frame(),
        )


@dataclass(frozen=True)
class Assembly:
    """Scoped assembly declaration; reference-dependent checks run when its bundle loads."""

    name: str
    params: dict[str, Param] = field(default_factory=dict)
    instances: tuple[Instance, ...] = ()
    port: dict[str, str] = field(default_factory=dict)
    dofs: dict[str, Dof] = field(default_factory=dict)
    motion_inputs: dict[str, str] = field(default_factory=dict)
    objectives: tuple[Objective, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    mates: tuple[Mate, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Assembly.name: expected a nonempty name, got {self.name!r}")
        for name in ("params", "port", "dofs", "motion_inputs", "metadata"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"Assembly {self.name!r}.{name}: expected a mapping")
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        for name in ("instances", "objectives", "constraints", "mates"):
            object.__setattr__(self, name, tuple(copy.deepcopy(getattr(self, name))))
        for key, param in self.params.items():
            if not isinstance(param, Param) or key != param.name:
                raise ValueError(f"Assembly.params[{key!r}]: expected matching Param.name")
        names: set[str] = set()
        for instance in self.instances:
            if not isinstance(instance, Instance) or instance.id in names:
                raise ValueError(
                    f"Assembly.instances: expected unique Instance IDs, got {instance!r}"
                )
            names.add(instance.id)
            missing = scalar_refs(instance.param_overrides) - set(self.params)
            if missing:
                raise ValueError(
                    f"Instance {instance.id!r}: undeclared parent parameters {sorted(missing)}"
                )
        for collection in (self.port, self.dofs, self.motion_inputs):
            for key in collection:
                validate_identifier(key, "Assembly address")
                if key in names:
                    raise ValueError(f"Assembly: ambiguous address name {key!r}")
                names.add(key)
        mate_names: set[str] = set()
        for mate in self.mates:
            if not isinstance(mate, Mate) or mate.id in mate_names:
                raise ValueError(f"Assembly.mates: expected unique Mate IDs, got {mate!r}")
            mate_names.add(mate.id)
            missing = scalar_refs(mate.offset.to_dict()) - set(self.params)
            if missing:
                raise ValueError(f"Mate {mate.id!r}: unknown offset parameters {sorted(missing)}")
        for key, dof in self.dofs.items():
            if not isinstance(dof, Dof):
                raise ValueError(f"Assembly.dofs[{key!r}]: expected Dof")
        for mapping in (self.port, self.motion_inputs):
            for key, address in mapping.items():
                _address(address, f"Assembly promotion {key!r}")
        from software_defined_matter.sdf.param_refs import (
            validate_param_refs,
            validate_param_relations,
        )

        param_doc = {"params": {key: param.to_dict() for key, param in self.params.items()}}
        validate_param_refs(param_doc)
        validate_param_relations(param_doc)
        self.parameter_part().free_params()

    def free_params(self) -> dict[str, Param]:
        """Return root-owned optimizable parameters in their vector order."""
        return self.parameter_part().free_params()

    def derived_order(self) -> list[str]:
        """Return local parameter relations in dependency order."""
        return self.parameter_part().derived_order()

    def parameter_part(self) -> Part:
        """Return the parameter scope for the existing relation and binding machinery."""
        return Part(self.name, params=copy.deepcopy(self.params))

    def to_dict(self) -> dict[str, Any]:
        """Serialize only this definition; referenced documents remain external."""
        params = self.parameter_part()
        params.refresh_derived()
        return {
            "schema_version": "0.5",
            "kind": "assembly",
            "name": self.name,
            "params": {k: p.to_dict() for k, p in params.params.items()},
            "instances": [i.to_dict() for i in self.instances],
            "port": dict(self.port),
            "dofs": {k: d.to_dict() for k, d in self.dofs.items()},
            "motion_inputs": dict(self.motion_inputs),
            "objectives": [o.to_dict() for o in self.objectives],
            "constraints": [c.to_dict() for c in self.constraints],
            "metadata": copy.deepcopy(self.metadata),
            **({"mates": [m.to_dict() for m in self.mates]} if self.mates else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assembly:
        """Read an assembly declaration without accessing its referenced files."""
        _keys(
            data,
            {
                "schema_version",
                "kind",
                "name",
                "params",
                "instances",
                "port",
                "dofs",
                "motion_inputs",
                "objectives",
                "constraints",
                "metadata",
                "mates",
            },
            "Assembly",
        )
        schema_version = data.get("schema_version")
        if (
            data.get("kind") != "assembly"
            or schema_version not in KNOWN_SCHEMA_VERSIONS
            or _parse_version(schema_version) < (0, 5)
        ):
            raise ValueError("Assembly: expected kind=assembly and a known schema_version >= 0.5")
        return cls(
            name=data["name"],
            params={k: Param.from_dict(p) for k, p in data.get("params", {}).items()},
            instances=tuple(Instance.from_dict(i) for i in data.get("instances", [])),
            port=data.get("port", {}),
            dofs={k: Dof.from_dict(d) for k, d in data.get("dofs", {}).items()},
            motion_inputs=data.get("motion_inputs", {}),
            objectives=tuple(Objective.from_dict(o) for o in data.get("objectives", [])),
            constraints=tuple(Constraint.from_dict(c) for c in data.get("constraints", [])),
            metadata=data.get("metadata", {}),
            mates=tuple(Mate.from_dict(m) for m in data.get("mates", [])),
        )
