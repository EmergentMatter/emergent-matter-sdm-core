"""Symbolic expression DSL for objectives and constraints.

Trees built with the ``expr_*`` helpers below are plain JSON-serialisable
Python dicts that live inside :class:`~software_defined_matter.model.Objective` and
:class:`~software_defined_matter.model.Constraint`. They are evaluated in one of
two places:

* :func:`eval_expr_pure` - for expressions that only reference params and
  numeric operations. Used by :mod:`software_defined_matter.dsl.resolve` to allow
  expressions inside SDF primitive kwargs without creating a circular
  dependency with the SDF compiler.
* :func:`compile_expr` - the full evaluator, which additionally resolves
  ``metric`` nodes by calling the registry in
  :mod:`software_defined_matter.objectives.metrics` against a compiled SDF closure.

Both paths return JAX scalars so ``jax.grad`` works end-to-end.

Node shapes
-----------

- ``{"type": "num",    "value": 3.14}``
- ``{"type": "param",  "name":  "outer_radius"}``
- ``{"type": "metric", "name":  "volume", "args": {...}}``
- ``{"type": "unop",   "op":    "neg|abs|sqrt|log|exp|square", "child": <expr>}``
- ``{"type": "binop",  "op":    "+|-|*|/|pow|min|max", "lhs": <expr>, "rhs": <expr>}``
- ``{"type": "reduce", "op":    "sum|mean|min|max", "children": [<expr>, ...]}``

Metric sampling domain (caveat)
-------------------------------
:func:`compile_expr` resolves the integration box via :func:`_resolve_bbox`
(explicit ``metadata["bbox"]`` > inference at current param values, taken over
the part's *envelope* rather than the part), then hands it to
:func:`~software_defined_matter.objectives.metrics.cubic_grid`, which grows it
to a whole number of cubic cells. The box is **never traced**: it is a
constant w.r.t. ``free_vec``, so gradients are clean. But its *tightness is
only per outer iteration*: the
optimiser must re-run :func:`compile_expr` after
:meth:`Part.update_from_vector` each step. A single compiled fn reused across
many optimiser steps keeps a stale (but still-enclosing, given the pad) box;
if you cannot re-derive per step, sample on the worst-case box instead
(``infer_sdf_bbox(..., mode="bounds")``). Geometry the inferrer cannot
reason about, today param-dependent rotation (a ``rotate_x|y|z|matrix``
with a ``$ref`` angle), has no inferable box and **must** supply
``metadata["bbox"]`` explicitly; compilation fails loud otherwise rather
than fabricating a domain that would silently corrupt every integral
metric.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp

from software_defined_matter.dsl.resolve import ParamLookup, make_binding

if TYPE_CHECKING:
    from software_defined_matter.model import ExprTree, Part, SDFTree
    from software_defined_matter.sdf.compile import SDFClosure


# ===========================================================================
# Builders
# ===========================================================================


def expr_num(d_value: float) -> dict[str, Any]:
    return {"type": "num", "value": float(d_value)}


def expr_param(name: str) -> dict[str, Any]:
    return {"type": "param", "name": name}


def expr_metric(name: str, **args: Any) -> dict[str, Any]:
    """Reference to a named metric (see
    :mod:`software_defined_matter.objectives.metrics`)."""
    return {"type": "metric", "name": name, "args": args}


def expr_unop(op: str, child: dict[str, Any]) -> dict[str, Any]:
    return {"type": "unop", "op": op, "child": child}


def expr_binop(op: str, lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    return {"type": "binop", "op": op, "lhs": lhs, "rhs": rhs}


def expr_reduce(op: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "reduce", "op": op, "children": children}


# Convenience shortcuts used heavily by tests & example scripts.
def neg(e: dict[str, Any]) -> dict[str, Any]:
    return expr_unop("neg", e)


def abs_(e: dict[str, Any]) -> dict[str, Any]:
    return expr_unop("abs", e)


def sqrt(e: dict[str, Any]) -> dict[str, Any]:
    return expr_unop("sqrt", e)


def log(e: dict[str, Any]) -> dict[str, Any]:
    return expr_unop("log", e)


def exp(e: dict[str, Any]) -> dict[str, Any]:
    return expr_unop("exp", e)


def square(e: dict[str, Any]) -> dict[str, Any]:
    return expr_unop("square", e)


def add(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return expr_binop("+", a, b)


def sub(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return expr_binop("-", a, b)


def mul(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return expr_binop("*", a, b)


def div(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return expr_binop("/", a, b)


def power(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return expr_binop("pow", a, b)


def min_(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return expr_binop("min", a, b)


def max_(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return expr_binop("max", a, b)


# ===========================================================================
# Operator table
# ===========================================================================

# fmt: off
_UNARY: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "neg":    lambda x: -x,
    "abs":    jnp.abs,
    "sqrt":   jnp.sqrt,
    "log":    jnp.log,
    "exp":    jnp.exp,
    "square": jnp.square,
    "sin": jnp.sin,
    "cos": jnp.cos,
}

_BINARY: dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    "+":   jnp.add,
    "-":   jnp.subtract,
    "*":   jnp.multiply,
    "/":   jnp.divide,
    "pow": jnp.power,
    "min": jnp.minimum,
    "max": jnp.maximum,
}

_REDUCE: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "sum":  jnp.sum,
    "mean": jnp.mean,
    "min":  jnp.min,
    "max":  jnp.max,
}
# fmt: on


# ===========================================================================
# Evaluation
# ===========================================================================


def _eval_node(
    node: Any,
    binding: ParamLookup,
    free_vec: jnp.ndarray,
    metric_fn: Callable[[str, dict[str, Any]], jnp.ndarray] | None,
) -> jnp.ndarray:
    # A bare number is accepted (useful for the $ref resolver case).
    if isinstance(node, (int, float)):
        return jnp.asarray(float(node), dtype=free_vec.dtype)
    if not isinstance(node, dict):
        raise ValueError(f"Not an expression node: {node!r}")

    # A bare `$ref` leaf mixed into a tree. Checked HERE, above the `type`
    # guard, because a `$ref` leaf carries no `type`: the branch at the foot of
    # this function was unreachable for the case its comment describes, and the
    # tolerance it claimed was never real. `glsl/emit.py::_emit_expr` accepts
    # the same leaf, and the two evaluators have to take the same documents.
    if "$ref" in node:
        return binding.get(node["$ref"], free_vec)

    if "type" not in node:
        raise ValueError(f"Not an expression node: {node!r}")

    t = node["type"]

    if t == "num":
        return jnp.asarray(float(node["value"]), dtype=free_vec.dtype)

    if t == "param":
        return binding.get(node["name"], free_vec)

    if t == "metric":
        if metric_fn is None:
            raise ValueError(
                "Metric nodes are not allowed inside SDF leaf expressions. "
                "Use expr_num / expr_param / expr_unop / expr_binop / expr_reduce instead, "
                "or compile via software_defined_matter.dsl.expr.compile_expr with a compiled SDF."
            )
        return metric_fn(node["name"], node.get("args", {}))

    if t == "unop":
        op = node["op"]
        if op not in _UNARY:
            raise ValueError(f"Unknown unary op {op!r}")
        return _UNARY[op](_eval_node(node["child"], binding, free_vec, metric_fn))

    if t == "binop":
        op = node["op"]
        if op not in _BINARY:
            raise ValueError(f"Unknown binary op {op!r}")
        lhs = _eval_node(node["lhs"], binding, free_vec, metric_fn)
        rhs = _eval_node(node["rhs"], binding, free_vec, metric_fn)
        return _BINARY[op](lhs, rhs)

    if t == "reduce":
        op = node["op"]
        if op not in _REDUCE:
            raise ValueError(f"Unknown reduce op {op!r}")
        stacked = jnp.stack([_eval_node(c, binding, free_vec, metric_fn) for c in node["children"]])
        return _REDUCE[op](stacked)

    raise ValueError(f"Unknown expression node type {t!r}")


def eval_expr_pure(
    tree: ExprTree,
    binding: ParamLookup,
    free_vec: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate an expression tree that does not contain ``metric`` nodes."""
    return _eval_node(tree, binding, free_vec, metric_fn=None)


def expr_param_names(tree: Any) -> set[str]:
    """Every parameter name an expression tree reads.

    Both spellings count: a ``{"type": "param"}`` node and a ``{"$ref": ...}``
    leaf, because :func:`_eval_node` resolves both. Used by the relation-DAG
    sort and by the GLSL emitter's dependency reporting — anything that has to
    know what a relation depends on must agree with what the evaluator reads.
    """
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("name"), str) and node.get("type") == "param":
                out.add(node["name"])
            if isinstance(node.get("$ref"), str):
                out.add(node["$ref"])
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(tree)
    return out


# ---------------------------------------------------------------------------
# Full compiler (with metric resolution)
# ---------------------------------------------------------------------------


def compile_expr(
    tree: ExprTree,
    part: Part,
    sdf_closure: SDFClosure | None = None,
    *,
    b_smooth_csg: bool | None = None,
    d_smooth_k: float | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return ``f(free_vec) -> scalar`` for an objective / constraint tree.

    If ``sdf_closure`` is ``None`` a fresh one is compiled from
    ``part.computed_envelope()``. Metric nodes call into
    :mod:`software_defined_matter.objectives.metrics` with the compiled SDF and the
    current free-param vector.
    """
    from software_defined_matter.objectives.metrics import (
        MIN_FEATURE_VOXELS,
        check_grid_resolution,
        cubic_grid,
        get_metric,
    )
    from software_defined_matter.sdf.compile import make_sdf_closure

    if b_smooth_csg is None:
        b_smooth_csg = bool(part.metadata.get("smooth_csg", False))
    if d_smooth_k is None:
        d_smooth_k = float(part.metadata.get("metric_smooth_k", 0.25))

    if sdf_closure is None:
        envelope = part.computed_envelope()
        if envelope is None:
            raise ValueError(
                "compile_expr needs either sdf_closure or a Part with at least one material"
            )
        sdf_closure = make_sdf_closure(
            envelope,
            part,
            b_smooth_csg=b_smooth_csg,
            d_smooth_k=d_smooth_k,
        )

    binding = make_binding(part)

    from software_defined_matter.sdf.compile import make_sdf_closure_with_binding

    # Envelope subtrees (relative_density's `envelope` arg) compile once over the
    # shared binding and are cached, so a metric reused across many free_vec
    # calls under jax.grad / jit never rebuilds the closure. Keyed on the tree
    # object's id: the compiled expr reuses the same dict, so the id is stable.
    _env_cache: dict[int, SDFClosure] = {}

    def _envelope_closure(env_tree: Any) -> SDFClosure:
        key = id(env_tree)
        if key not in _env_cache:
            _env_cache[key] = make_sdf_closure_with_binding(
                env_tree, binding, b_smooth_csg=b_smooth_csg, d_smooth_k=d_smooth_k
            )
        return _env_cache[key]

    if "metric_bbox_pad" in part.metadata:
        warnings.warn(
            "metric_bbox_pad padded the sampling box so the old soft-Heaviside "
            "tails were not clipped at a face. The occupancy ramp now has "
            "compact support, so empty cells contribute exactly zero and no pad "
            "is needed; the key is ignored. Delete it.",
            DeprecationWarning,
            stacklevel=2,
        )

    # The sampling box is resolved lazily and memoised: it is computed only if
    # a metric node is actually evaluated (a pure expr/constraint on unbounded
    # geometry must still compile), once per compiled fn (reused across the
    # many free_vec calls of one jax.grad), and never traced. Per-iteration
    # tightness comes from the optimiser re-deriving via compile_expr each
    # outer step (see _resolve_bbox / metrics.py sampling-domain contract).
    _cache: dict[str, Any] = {}

    def _sampling_domain() -> tuple:
        if "bbox" not in _cache:
            env_tree = _default_envelope_tree(part)
            raw = _resolve_bbox(part, env_tree)
            bbox, res, d_h = cubic_grid(
                raw,
                count=int(part.metadata.get("grid_resolution", 64)),
                voxel_size=part.metadata.get("metric_voxel_size"),
                cap=int(part.metadata.get("metric_grid_cap", 192)),
            )
            check_grid_resolution(
                # computed_envelope() is None only when the part has no
                # materials, in which case no metric closure exists to call
                # _sampling_domain() in the first place -- cast documents that
                # invariant to mypy without changing the (unreachable-in-
                # practice) None-input behaviour below.
                cast("SDFTree", part.computed_envelope()),
                part,
                d_h,
                min_voxels=float(
                    part.metadata.get("metric_min_feature_voxels", MIN_FEATURE_VOXELS)
                ),
            )
            _cache["bbox"] = bbox
            _cache["res"] = res
            _cache["cell"] = d_h
            _cache["env_tree"] = env_tree
        return (_cache["bbox"], _cache["res"], _cache["cell"], _cache["env_tree"])

    def _fn(free_vec: jnp.ndarray) -> jnp.ndarray:
        free_vec = jnp.asarray(free_vec)

        def metric_fn(name: str, args: dict[str, Any]) -> jnp.ndarray:
            registered = get_metric(name)
            bbox, res, d_cell, default_env = _sampling_domain()
            kwargs: dict[str, Any] = {
                "sdf": lambda p: sdf_closure(p, free_vec),
                "part": part,
                "binding": binding,
                "free_vec": free_vec,
                "bbox": bbox,
                "grid_resolution": args.get("grid_resolution", res),
                "cell_size": d_cell,
                "args": args,
            }
            # An explicit `envelope=` wins; otherwise relative_density divides
            # by the part's own envelope, derived by walking the SDF tree.
            env_tree = args.get("envelope", default_env)
            if env_tree is not None:
                env_closure = _envelope_closure(env_tree)
                kwargs["env_sdf"] = lambda p: env_closure(p, free_vec)
            return registered(**kwargs)

        return _eval_node(tree, binding, free_vec, metric_fn=metric_fn)

    return _fn


def _default_envelope_tree(part: Part) -> SDFTree | None:
    """The part's envelope tree: the same solid with its holes filled in.

    ``relative_density`` divides by this. Returns ``None`` when the geometry
    has no finite envelope, in which case the metric falls back to the sampling
    box and every other metric is unaffected.
    """
    from software_defined_matter.sdf.envelope import (
        EnvelopeInferenceError,
        infer_sdf_envelope,
    )

    materials = part.computed_envelope()
    if materials is None:
        return None
    try:
        return infer_sdf_envelope(materials, part)
    except EnvelopeInferenceError:
        return None


def _resolve_bbox(part: Part, envelope_tree: SDFTree | None) -> tuple:
    """Return the sampling box: a domain that encloses the part's envelope.
    The box is inferred from the **envelope**, not from the part. The envelope
    always contains the part and is sometimes larger, and a box sized to the
    part alone would truncate the denominator. E.g. for ``subtract(sphere, cap_cylinder)``,
    the part stops at z = 4 where its envelope reaches z = 5.

    The box is a constant w.r.t. ``free_vec`` (stop-gradient by construction);
    its tightness is per outer iteration, so the optimiser must re-derive
    (re-run ``compile_expr`` after :meth:`Part.update_from_vector`) each step.

    Fails loud on geometry the inferrer cannot reason about (today,
    param-dependent rotation): a silently fabricated domain in an
    optimisation loop corrupts every integral metric with no signal. The
    fix is to state the domain explicitly via ``metadata["bbox"]``.
    """
    override = part.metadata.get("bbox")
    if override is not None:
        lo, hi = override
        return (tuple(lo), tuple(hi))

    from software_defined_matter.sdf.bbox import BBoxInferenceError, infer_sdf_bbox

    target = envelope_tree if envelope_tree is not None else part.computed_envelope()
    if target is None:
        raise ValueError("compile_expr needs a Part with >=1 material to infer a sampling bbox.")
    try:
        return infer_sdf_bbox(target, part, mode="values")
    except BBoxInferenceError as exc:
        raise ValueError(
            f"Cannot infer a metric-sampling bbox for part {part.name!r}: {exc}\n"
            "The geometry has nodes the inferrer cannot reason about "
            "(today, this is param-dependent rotation, a `rotate_x|y|z|"
            "matrix` with a `$ref` angle). Set an explicit domain:\n"
            "    part.metadata['bbox'] = [[xlo, ylo, zlo], [xhi, yhi, zhi]]"
        ) from exc


__all__ = [
    "abs_",
    "add",
    "compile_expr",
    "div",
    # Evaluators
    "eval_expr_pure",
    "exp",
    "expr_binop",
    "expr_metric",
    "expr_param_names",
    # Builders
    "expr_num",
    "expr_param",
    "expr_reduce",
    "expr_unop",
    "log",
    "max_",
    "min_",
    "mul",
    "neg",
    "power",
    "sqrt",
    "square",
    "sub",
]
