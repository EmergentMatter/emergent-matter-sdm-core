"""Copy and traverse geometry trees without recursive preparation walks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from typing import Any


def child_nodes(node: Any) -> list[Any]:
    if not isinstance(node, dict) or node.get("type") in ("primitive", "extension"):
        return []
    children = list(node.get("children") or [])
    if "child" in node:
        children.insert(0, node["child"])
    return children


def iter_postorder(
    tree: Any, children: Callable[[Any], Iterable[Any]] = child_nodes
) -> Iterator[Any]:
    """Visit shared nodes once, leaving extension traversal to its registered handler."""
    done: set[int] = set()
    active: set[int] = set()
    pending = [(tree, False)]
    while pending:
        node, ready = pending.pop()
        key = id(node)
        if key in done:
            continue
        if ready:
            active.remove(key)
            done.add(key)
            yield node
            continue
        if key in active:
            raise ValueError("SDF tree contains a cycle")
        active.add(key)
        pending.append((node, True))
        pending.extend((child, False) for child in reversed(list(children(node))))


def snapshot_tree(tree: dict[str, Any]) -> dict[str, Any]:
    """Isolate dictionaries, lists, tuples and leaves, preserving shared subtrees.

    Numeric arrays and other leaves retain ``deepcopy`` semantics. Cyclic
    containers are not geometry and are rejected before compilation.
    """
    memo: dict[int, Any] = {}
    active: set[int] = set()
    pending = [(tree, False)]
    while pending:
        value, ready = pending.pop()
        key = id(value)
        if key in memo:
            continue
        if not isinstance(value, (dict, list, tuple)):
            memo[key] = deepcopy(value, memo)
        elif ready:
            if isinstance(value, dict):
                memo[key] = {deepcopy(k, memo): memo[id(v)] for k, v in value.items()}
            else:
                items = [memo[id(v)] for v in value]
                memo[key] = tuple(items) if isinstance(value, tuple) else items
            active.remove(key)
        else:
            if key in active:
                raise ValueError("SDF tree contains a cyclic container")
            active.add(key)
            pending.append((value, True))
            children = list(value.values()) if isinstance(value, dict) else value
            pending.extend((child, False) for child in reversed(children))
    return memo[id(tree)]
