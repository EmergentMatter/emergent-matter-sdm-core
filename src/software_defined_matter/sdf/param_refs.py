"""Cross-field referential integrity for parameter names and relations."""

from __future__ import annotations

from typing import Any

from software_defined_matter.sdf.validate import SemanticValidationError


class UndeclaredParamRefError(SemanticValidationError):
    """A document references a parameter name it does not declare."""


class ParamRelationError(SemanticValidationError):
    """A parameter relation is cyclic or uses a non-pure expression."""


def _walk_refs(node: Any, where: str, out: list[tuple[str, str]]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            out.append((ref, where))
        if node.get("type") == "param" and isinstance(node.get("name"), str):
            out.append((node["name"], where))
        for value in node.values():
            _walk_refs(value, where, out)
    elif isinstance(node, list):
        for value in node:
            _walk_refs(value, where, out)


def collect_param_refs(doc: dict[str, Any]) -> list[tuple[str, str]]:
    """Return every parameter reference as ``(name, human-readable site)``."""
    refs: list[tuple[str, str]] = []
    for key in ("materials", "couplings", "ports"):
        for i, item in enumerate(doc.get(key) or []):
            if not isinstance(item, dict):
                continue
            tree = item.get("sdf_tree")
            if tree is not None:
                label = item.get("name") or f"{key}[{i}]"
                _walk_refs(tree, f"sdf_tree of {key[:-1]} {label!r}", refs)
    for port in doc.get("ports", []):
        _walk_refs(port.get("frame"), f"port {port.get('name')!r} frame", refs)
    for key in ("objectives", "constraints"):
        for i, item in enumerate(doc.get(key) or []):
            if not isinstance(item, dict):
                continue
            label = item.get("name") or f"{key}[{i}]"
            _walk_refs(item.get("expr"), f"{key[:-1]} {label!r}", refs)
    for name, param in (doc.get("params") or {}).items():
        if isinstance(param, dict) and param.get("expr") is not None:
            _walk_refs(param["expr"], f"relation of param {name!r}", refs)
    for i, animation in enumerate(((doc.get("metadata") or {}).get("animations")) or []):
        if not isinstance(animation, dict):
            continue
        label = animation.get("name") or f"animations[{i}]"
        for track in animation.get("tracks") or []:
            if isinstance(track, dict) and isinstance(track.get("param"), str):
                refs.append((track["param"], f"animation {label!r} track"))
    return refs


def validate_param_refs(doc: dict[str, Any]) -> None:
    """Raise with every parameter reference that has no declaration."""
    declared = set(doc.get("params") or {})
    missing: dict[str, set[str]] = {}
    for name, where in collect_param_refs(doc):
        if name not in declared:
            missing.setdefault(name, set()).add(where)
    if missing:
        lines = [
            f"  {name!r} referenced by " + ", ".join(sorted(sites))
            for name, sites in sorted(missing.items())
        ]
        raise UndeclaredParamRefError(
            f"{len(missing)} parameter name(s) referenced but not declared in 'params':\n"
            + "\n".join(lines)
        )


def relation_order(deps: dict[str, set[str]]) -> list[str]:
    """Topologically sort a relation dependency graph.

    ``deps`` maps each derived param name to the set of *other derived* names
    it reads. Raises :class:`ParamRelationError` naming the cycle, so load-time
    validation and :meth:`Part.derived_order` cannot disagree about what a
    cycle is.
    """
    state: dict[str, int] = {}
    path: list[str] = []
    order: list[str] = []

    def visit(name: str) -> None:
        mark = state.get(name, 0)
        if mark == 2:
            return
        if mark == 1:
            start = path.index(name)
            cycle = path[start:] + [name]
            raise ParamRelationError("Parameter relation cycle: " + " -> ".join(cycle))
        state[name] = 1
        path.append(name)
        for dep in sorted(deps.get(name, ())):
            if dep in deps:
                visit(dep)
        path.pop()
        state[name] = 2
        order.append(name)

    for name in deps:
        visit(name)
    return order


def validate_param_relations(doc: dict[str, Any]) -> None:
    """Reject metric-bearing or cyclic ``Param.expr`` dependency graphs."""
    params = doc.get("params") or {}
    graph: dict[str, set[str]] = {}
    for name, param in params.items():
        expr = param.get("expr") if isinstance(param, dict) else None
        if expr is None:
            continue
        stack = [expr]
        deps: set[str] = set()
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if node.get("type") == "metric":
                    raise ParamRelationError(
                        f"Relation of param {name!r} contains a 'metric' node; "
                        "parameter relations must be pure numeric expressions."
                    )
                ref = node.get("$ref")
                if isinstance(ref, str):
                    deps.add(ref)
                if node.get("type") == "param" and isinstance(node.get("name"), str):
                    deps.add(node["name"])
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        valid_deps = set()
        for dep in deps:
            if dep in graph or dep in params and params[dep].get("expr"):
                valid_deps.add(dep)
        graph[name] = valid_deps

    relation_order(graph)


__all__ = [
    "UndeclaredParamRefError",
    "ParamRelationError",
    "collect_param_refs",
    "relation_order",
    "validate_param_refs",
    "validate_param_relations",
]
