"""Compile occurrence motion bindings before numerical placement.

Independent inputs and derived coordinates have separate layouts. Expressions
read radians/mm, matching part kinematics, regardless of authored DOF units.
Design parameters retain their declared numerical units; constants and scale
factors in expressions must produce the target's evaluator units.
"""

from __future__ import annotations

import math
from typing import Any

from software_defined_matter.assembly.bundle import AssemblyBundle
from software_defined_matter.assembly.model import Assembly


def _name(key: tuple[str, str]) -> str:
    return ".".join(value for value in key if value)


class _MotionGraph:
    def __init__(self, bundle: AssemblyBundle) -> None:
        from software_defined_matter.kinematics import _compile_expression
        from software_defined_matter.sdf.param_refs import relation_order

        declarations = {}
        for scope, doc in sorted(bundle.definitions.items()):
            coords = (
                {name: d.to_dict() for name, d in doc.dofs.items()}
                if isinstance(doc, Assembly)
                else {d["name"]: d for d in (doc.kinematics or {}).get("dofs", [])}
            )
            for name, declaration in sorted(coords.items()):
                declarations[_name((scope, name))] = declaration
        self.names = tuple(declarations)
        self.indices = {name: i for i, name in enumerate(self.names)}
        drivers = {}
        dependencies = {}
        for scope, doc in sorted(bundle.definitions.items()):
            if not isinstance(doc, Assembly):
                continue
            for instance in doc.instances:
                child = _name((scope, instance.id))
                for target, expr in instance.dof_bindings.items():
                    name = _name(bundle.resolve_dof(target, scope=child))
                    refs: set[str] = set()

                    def rewrite(node: Any, scope: str = scope, refs: set[str] = refs) -> Any:
                        if isinstance(node, (int, float)):
                            return {"type": "num", "value": node}
                        if "$ref" in node:
                            return {"type": "param", "name": node["$ref"]}
                        if node["type"] == "dof":
                            source = _name(bundle.resolve_dof(node["name"], scope=scope))
                            refs.add(source)
                            return {"type": "dof", "name": source}
                        result = dict(node)
                        for key in ("child", "lhs", "rhs"):
                            if key in result:
                                result[key] = rewrite(result[key])
                        if "children" in result:
                            result["children"] = [rewrite(c) for c in result["children"]]
                        return result

                    tree = rewrite(expr)
                    drivers[name] = _compile_expression(tree, bundle.binding(scope), self.indices)
                    dependencies[name] = refs
        self._drivers = tuple(
            (self.indices[name], drivers[name]) for name in relation_order(dependencies)
        )
        self.input_names = tuple(name for name in self.names if name not in drivers)
        self._input_indices = tuple(self.indices[name] for name in self.input_names)
        self.units = tuple(declarations[name]["unit"] for name in self.input_names)
        factors = tuple(math.pi / 180 if u == "deg" else 1.0 for u in self.units)
        self.defaults = tuple(
            declarations[name].get("default", 0.0) * factor
            for name, factor in zip(self.input_names, factors, strict=True)
        )
        self.ranges = tuple(
            tuple(value * factor for value in declarations[name]["range"])
            for name, factor in zip(self.input_names, factors, strict=True)
        )

    def evaluate(self, inputs: Any, design: Any) -> Any:
        import jax.numpy as jnp

        values = jnp.zeros(len(self.names), dtype=jnp.result_type(inputs, design))
        values = values.at[jnp.asarray(self._input_indices, dtype=jnp.int32)].set(inputs)
        for index, expression in self._drivers:
            values = values.at[index].set(expression(values, design))
        return values
