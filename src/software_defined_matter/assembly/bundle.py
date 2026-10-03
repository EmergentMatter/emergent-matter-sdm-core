"""Resolve external definitions once while retaining distinct occurrence scopes.

Canonical filesystem paths detect recursive inclusion, including symlink aliases.
Content pins hash the bytes actually parsed. A bundle owns a snapshot of its
loaded definitions; numerical bindings never mutate shared part parameters.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from software_defined_matter.assembly.model import Assembly
from software_defined_matter.model import Constraint, Part
from software_defined_matter.ports import Port, scalar_refs, validate_scalar

__all__ = ["AssemblyBundle", "ResolvedPort", "ScopedConstraint", "load_bundle"]


@dataclass(frozen=True)
class ResolvedPort:
    """Terminal port and the physical occurrence it belongs to."""

    instance: str
    port: Port


@dataclass(frozen=True)
class ScopedConstraint:
    """Constraint paired with its originating occurrence, including the empty root scope."""

    instance: str
    constraint: Constraint


class _InstanceBinding:
    def __init__(self, document: Part | Assembly, parent: Any, overrides: dict[str, Any]) -> None:
        self.params = copy.deepcopy(document.params)
        self.parent = parent
        self.overrides = copy.deepcopy(overrides)

    def initial_free_vector(self) -> Any:
        """Use the root vector, including when an SDF closure supplies its default."""
        return self.parent.initial_free_vector()

    def get(self, name: str, free_vec: Any) -> Any:
        import jax.numpy as jnp

        from software_defined_matter.dsl.expr import eval_expr_pure

        if name not in self.params:
            raise ValueError(f"Unknown instance parameter {name!r}")
        if name in self.overrides:
            value = self.overrides[name]
            return (
                self.parent.get(value["$ref"], free_vec)
                if isinstance(value, dict)
                else jnp.asarray(value, dtype=free_vec.dtype)
            )
        param = self.params[name]
        if param.expr is not None:
            return eval_expr_pure(param.expr, self, free_vec)
        return jnp.asarray(param.numeric_value(), dtype=free_vec.dtype)


class AssemblyBundle:
    """Resolved assembly snapshot with address resolution and root-owned design bindings.

    Construction is through :func:`load_bundle`. Placement and metric evaluation
    are deliberately separate: address resolution never guesses a world pose.
    """

    def __init__(
        self, root: Assembly, definitions: dict[str, Part | Assembly], sources: dict[str, Path]
    ) -> None:
        self.root = root
        self.definitions = definitions
        self.sources = sources
        self._bindings: dict[str, Any] | None = None
        self._validate()

    def _child(self, scope: str, name: str) -> str:
        address = f"{scope}.{name}" if scope else name
        if address not in self.definitions:
            raise ValueError(f"Unknown instance segment {name!r} in scope {scope or '<root>'!r}")
        return address

    def resolve_port(self, address: str, *, scope: str = "") -> ResolvedPort:
        """Resolve deep addresses and recursively promoted ports to a terminal occurrence."""
        document = self.definitions[scope]
        first, dot, rest = address.partition(".")
        if dot:
            return self.resolve_port(rest, scope=self._child(scope, first))
        if isinstance(document, Assembly):
            if first not in document.port:
                raise ValueError(f"Unknown promoted port {address!r} in {scope or '<root>'!r}")
            return self.resolve_port(document.port[first], scope=scope)
        for port in document.ports:
            if port.name == first:
                return ResolvedPort(scope, port)
        raise ValueError(f"Unknown port {address!r} in instance {scope!r}")

    def resolve_dof(self, address: str, *, scope: str = "") -> tuple[str, str]:
        """Resolve promoted motion inputs to an occurrence and local DOF name."""
        document = self.definitions[scope]
        first, dot, rest = address.partition(".")
        if dot:
            return self.resolve_dof(rest, scope=self._child(scope, first))
        if isinstance(document, Assembly):
            if first in document.motion_inputs:
                return self.resolve_dof(document.motion_inputs[first], scope=scope)
            names = set(document.dofs)
        else:
            names = {d["name"] for d in (document.kinematics or {}).get("dofs", [])}
        if first not in names:
            raise ValueError(f"Unknown DOF {address!r} in {scope or '<root>'!r}")
        return scope, first

    def _validate(self) -> None:
        drivers: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for scope, document in self.definitions.items():
            if isinstance(document, Assembly):
                for address in document.port.values():
                    self.resolve_port(address, scope=scope)
                for address in document.motion_inputs.values():
                    self.resolve_dof(address, scope=scope)
                for mate in document.mates:
                    self.resolve_port(mate.parent, scope=scope)
                    self.resolve_port(mate.child, scope=scope)
                    if mate.dof is not None:
                        dof_scope, name = self.resolve_dof(mate.dof, scope=scope)
                        owner = self.definitions[dof_scope]
                        kind = (
                            owner.dofs[name].kind
                            if isinstance(owner, Assembly)
                            else next(
                                d["kind"]
                                for d in (owner.kinematics or {}).get("dofs", [])
                                if d["name"] == name
                            )
                        )
                        expected = "angle" if mate.kind == "revolute" else "length"
                        if kind != expected:
                            raise ValueError(
                                f"Mate {mate.id!r}: expected {expected} DOF, got {kind}"
                            )
                for instance in document.instances:
                    child_scope = self._child(scope, instance.id)
                    child = self.definitions[child_scope]
                    unknown = set(instance.param_overrides) - set(child.params)
                    if unknown:
                        raise ValueError(
                            f"Instance {child_scope!r}: unknown overridden parameters "
                            f"{sorted(unknown)}"
                        )
                    for name, override in instance.param_overrides.items():
                        if isinstance(override, dict):
                            parent_unit = document.params[override["$ref"]].unit
                            child_unit = child.params[name].unit
                            if parent_unit != child_unit:
                                raise ValueError(
                                    f"Instance {child_scope!r} parameter {name!r}: "
                                    f"binding unit {parent_unit!r} does not match {child_unit!r}"
                                )
                    for target, expr in instance.dof_bindings.items():
                        target_id = self.resolve_dof(target, scope=child_scope)
                        if target_id in drivers:
                            raise ValueError(f"DOF {target_id!r}: conflicting drivers")
                        refs: set[tuple[str, str]] = set()
                        pure = self._motion_expression(expr, scope, refs)
                        validate_scalar(pure, f"DOF binding {child_scope}.{target}")
                        missing = scalar_refs(pure) - set(document.params)
                        if missing:
                            raise ValueError(
                                f"DOF binding {child_scope}.{target}: "
                                f"unknown parameters {sorted(missing)}"
                            )
                        drivers[target_id] = refs
            for item in (*document.objectives, *document.constraints):
                self._validate_expression(item.expr, scope)
        from software_defined_matter.sdf.param_refs import relation_order

        relation_order({repr(key): {repr(ref) for ref in refs} for key, refs in drivers.items()})

    def _motion_expression(self, node: Any, scope: str, refs: set[tuple[str, str]]) -> Any:
        """Collect DOF dependencies while validating the remaining pure expression."""
        if isinstance(node, dict):
            if node.get("type") == "dof":
                if set(node) != {"type", "name"}:
                    raise ValueError("DOF binding: expected type and name only")
                refs.add(self.resolve_dof(node["name"], scope=scope))
                return {"type": "num", "value": 0.0}
            return {key: self._motion_expression(value, scope, refs) for key, value in node.items()}
        if isinstance(node, list):
            return [self._motion_expression(value, scope, refs) for value in node]
        return node

    def _validate_expression(self, node: Any, scope: str) -> None:
        if isinstance(node, dict):
            kind = node.get("type")
            target = scope
            if "instance" in node:
                if kind not in {"param", "metric"} or not isinstance(node["instance"], str):
                    raise ValueError("instance scope is only valid on param and metric nodes")
                for segment in node["instance"].split("."):
                    target = self._child(target, segment)
            name = (
                node.get("$ref")
                if "$ref" in node
                else node.get("name")
                if kind == "param"
                else None
            )
            if name is not None and name not in self.definitions[target].params:
                raise ValueError(
                    f"Expression in {scope or '<root>'!r}: unknown parameter {name!r} "
                    f"in {target or '<root>'!r}"
                )
            for value in node.values():
                self._validate_expression(value, scope)
        elif isinstance(node, list):
            for value in node:
                self._validate_expression(value, scope)

    def binding(self, instance: str = "") -> Any:
        """Return a lookup that consumes only the root assembly's free vector."""
        from software_defined_matter.dsl.resolve import make_binding

        if self._bindings is None:
            bindings: dict[str, Any] = {"": make_binding(self.root.parameter_part())}
            for scope, document in self.definitions.items():
                if not isinstance(document, Assembly):
                    continue
                for occurrence in document.instances:
                    child = self._child(scope, occurrence.id)
                    bindings[child] = _InstanceBinding(
                        self.definitions[child], bindings[scope], occurrence.param_overrides
                    )
            self._bindings = bindings
        if instance not in self._bindings:
            raise ValueError(f"Unknown instance scope {instance!r}")
        return self._bindings[instance]

    def constraints(self) -> tuple[ScopedConstraint, ...]:
        """Collect all constraints in occurrence scope; child objectives are not inherited."""
        return tuple(
            ScopedConstraint(scope, copy.deepcopy(item))
            for scope, doc in self.definitions.items()
            for item in doc.constraints
        )

    def evaluate_expression(
        self, expr: dict[str, Any], free_vec: Any, *, scope: str = "", metric: Any = None
    ) -> Any:
        """Evaluate scoped parameter expressions, delegating metrics to an explicit callback.

        ``metric(instance, name, args, free_vec)`` must supply placement-aware
        metric semantics. Without it metric expressions fail explicitly.
        """
        from software_defined_matter.dsl.expr import eval_expr_pure

        self._validate_expression(expr, scope)
        values: dict[str, Any] = {}

        def rewrite(node: Any) -> Any:
            if isinstance(node, dict):
                kind = node.get("type")
                if kind in {"param", "metric"} or "$ref" in node:
                    target = scope
                    if "instance" in node:
                        for segment in node["instance"].split("."):
                            target = self._child(target, segment)
                    if kind == "metric":
                        if metric is None:
                            raise ValueError(
                                f"Metric {node['name']!r} requires an assembly metric evaluator"
                            )
                        value = metric(target, node["name"], node.get("args", {}), free_vec)
                    else:
                        value = self.binding(target).get(
                            node.get("$ref", node.get("name")), free_vec
                        )
                    key = str(len(values))
                    values[key] = value
                    return {"type": "param", "name": key}
                return {key: rewrite(value) for key, value in node.items()}
            if isinstance(node, list):
                return [rewrite(value) for value in node]
            return node

        class Values:
            def get(self, name: str, free_vec: Any) -> Any:
                return values[name]

        return eval_expr_pure(rewrite(expr), Values(), free_vec)


def load_bundle(path: str | Path) -> AssemblyBundle:
    """Load and validate a reference graph, pinning exact bytes and detecting cycles.

    Relative references resolve beside the declaring file, not the root bundle.
    Repeated references share a parsed definition but retain distinct bindings.
    """
    from software_defined_matter.io import validate

    cache: dict[Path, tuple[bytes, Part | Assembly]] = {}
    active: list[Path] = []
    definitions: dict[str, Part | Assembly] = {}
    sources: dict[str, Path] = {}

    def visit(source: Path, scope: str, pin: str | None = None) -> Part | Assembly:
        source = source.resolve()
        if source in active:
            raise ValueError(
                "Assembly reference cycle: " + " -> ".join(str(p) for p in [*active, source])
            )
        if source not in cache:
            raw = source.read_bytes()
            doc = json.loads(raw)
            validate(doc)
            definition = (
                Assembly.from_dict(doc) if doc.get("kind") == "assembly" else Part.from_dict(doc)
            )
            cache[source] = raw, definition
        raw, definition = cache[source]
        if pin is not None and pin != "sha256:" + hashlib.sha256(raw).hexdigest():
            raise ValueError(f"Instance {scope!r}: content hash mismatch for {source}")
        definitions[scope], sources[scope] = definition, source
        active.append(source)
        if isinstance(definition, Assembly):
            for instance in definition.instances:
                child = f"{scope}.{instance.id}" if scope else instance.id
                visit(source.parent / instance.part_ref.path, child, instance.part_ref.content_hash)
        active.pop()
        return definition

    root = visit(Path(path), "")
    if not isinstance(root, Assembly):
        raise ValueError("load_bundle expected an Assembly document")
    return AssemblyBundle(root, definitions, sources)
