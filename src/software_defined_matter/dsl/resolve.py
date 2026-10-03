"""Resolve leaf values inside an SDF or expression tree against a Part's params.

A leaf value produced by a DSL builder may be any of:

* a plain Python number (``float``, ``int``)
* a list / tuple of numbers (for vector-valued primitive args, e.g. box extents)
* a ``$ref`` node: ``{"$ref": "param_name"}`` - resolved against the Part's
  ``params`` dict, honouring the split between fixed, free (optimisable) and
  DERIVED (defined by a ``Param.expr`` relation) params
* an expression tree (see :mod:`software_defined_matter.dsl.expr`) that does **not**
  contain ``metric`` nodes - these require a compiled SDF closure and must
  be resolved via the expression compiler itself.

The resolver returns JAX values so the whole SDF / objective pipeline stays
``jit`` / ``grad`` compatible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import jax.numpy as jnp

if TYPE_CHECKING:
    from software_defined_matter.model import Part


class ParamLookup(Protocol):
    """Anything :func:`~software_defined_matter.dsl.expr.eval_expr_pure` can resolve against.

    :class:`ParamBinding` and the private ``_DerivedView`` façade both satisfy
    this; the evaluator only calls ``get``.
    """

    def get(self, name: str, free_vec: jnp.ndarray) -> jnp.ndarray: ...


class ParamBinding:
    """Maps param names to positions in the free-parameter vector or fixed values.

    Construct once per ``Part``, then pass to the SDF/expr evaluators along
    with the current free-parameter vector (a ``jnp.ndarray`` of shape
    ``(n_free,)``).
    """

    def __init__(self, part: Part) -> None:
        free = part.free_params()
        self.free_names: list[str] = list(free.keys())
        self.free_idx = {n: i for i, n in enumerate(self.free_names)}
        # A derived param is NOT fixed: its cached ``value`` is a snapshot, and
        # binding it as a constant would freeze the relation at whatever the
        # base params happened to be when the binding was built — the geometry
        # would then stop tracking its own definition under both optimisation
        # and a live scrub. It is resolved from its expr instead, below.
        self.derived = {name: p.expr for name, p in part.params.items() if p.expr is not None}
        self.fixed = {
            name: float(p.numeric_value())
            for name, p in part.params.items()
            if not p.free and p.expr is None
        }
        self._initial_values = {n: float(part.params[n].numeric_value()) for n in self.free_names}
        # Acyclicity is checked at load (io.validate); re-check here because a
        # Part built in memory never went through a loader, and a cycle reached
        # from inside a jit trace surfaces as a RecursionError with a stack of
        # tracer frames instead of a sentence naming two param names.
        if self.derived:
            part.derived_order()

    def n_free(self) -> int:
        return len(self.free_names)

    def get(self, name: str, free_vec: jnp.ndarray) -> jnp.ndarray:
        """Return the JAX scalar for the parameter named ``name``.

        A derived param is evaluated from its relation through THIS binding, so
        the result is a traced function of ``free_vec`` and ``jax.grad`` flows
        through the relation to the base params.
        """
        if name in self.free_idx:
            return free_vec[self.free_idx[name]]
        if name in self.fixed:
            return jnp.asarray(self.fixed[name], dtype=free_vec.dtype)
        if name in self.derived:
            return self._eval_derived(name, free_vec, {})
        raise KeyError(
            f"Unknown param {name!r} (free={self.free_names}, "
            f"fixed={list(self.fixed)}, derived={list(self.derived)})"
        )

    def _eval_derived(
        self,
        name: str,
        free_vec: jnp.ndarray,
        memo: dict,
    ) -> jnp.ndarray:
        """Evaluate one relation, memoised for the duration of THIS resolution.

        The memo is per-call, not per-binding, on purpose: a cached value is a
        JAX tracer belonging to one trace, and stashing it on the binding
        (which outlives the trace, and is shared across ``jax.grad`` /
        ``jit`` calls) leaks a tracer between traces. Within one resolution the
        memo is what stops a diamond-shaped relation DAG from being
        re-evaluated exponentially; across resolutions XLA's CSE does the same
        job for free.
        """
        if name in memo:
            return memo[name]
        from software_defined_matter.dsl.expr import eval_expr_pure

        value = eval_expr_pure(self.derived[name], _DerivedView(self, memo), free_vec)
        memo[name] = value
        return value

    def initial_free_vector(self) -> jnp.ndarray:
        """Return the initial values of the free params as a ``jnp.ndarray``.

        The dtype follows the caller's ``jax.config`` (float32 by default,
        float64 when ``jax_enable_x64`` is set).  No explicit dtype is forced
        here so that downstream gradient loops that the caller enabled work
        correctly.
        """
        values = [self._initial_value(n) for n in self.free_names]
        return jnp.asarray(values)

    def _initial_value(self, name: str) -> float:
        return self._initial_values[name]


class _DerivedView:
    """A :class:`ParamBinding` façade that threads one resolution's memo.

    Handed to ``eval_expr_pure`` while evaluating a relation so that a nested
    ``param`` leaf naming another derived param shares the same memo. Exposes
    only ``get`` — that is the whole of what the expression evaluator uses.
    """

    __slots__ = ("_binding", "_memo")

    def __init__(self, binding: ParamBinding, memo: dict) -> None:
        self._binding = binding
        self._memo = memo

    def get(self, name: str, free_vec: jnp.ndarray) -> jnp.ndarray:
        if name in self._binding.derived:
            return self._binding._eval_derived(name, free_vec, self._memo)
        return self._binding.get(name, free_vec)


def make_binding(part: Part) -> ParamBinding:
    """Construct a :class:`ParamBinding` for ``part``."""
    return ParamBinding(part)


# ---------------------------------------------------------------------------
# Value resolution
# ---------------------------------------------------------------------------


def resolve_param_value(value: Any, binding: ParamBinding, free_vec: jnp.ndarray) -> jnp.ndarray:
    """Resolve a DSL leaf into a JAX scalar or 1-D array.

    - ``int | float``                    -> scalar ``jnp.ndarray``
    - ``list | tuple``                   -> 1-D array (elements resolved recursively)
    - ``{"$ref": "name"}``               -> scalar from the binding
    - expression tree (non-``metric``)   -> scalar from ``eval_expr_pure``
    """
    if isinstance(value, dict):
        if "$ref" in value:
            return binding.get(value["$ref"], free_vec)
        if "type" in value:
            from software_defined_matter.dsl.expr import eval_expr_pure

            return eval_expr_pure(value, binding, free_vec)
        raise ValueError(f"Unrecognised DSL leaf dict: {value!r}")

    if isinstance(value, (list, tuple)):
        return jnp.stack([resolve_param_value(v, binding, free_vec) for v in value])

    if isinstance(value, bool):
        # Guard: jnp treats bool as 0/1 which is almost never what the DSL
        # wants; if it ever is, the user can pass 0.0 / 1.0 explicitly.
        raise TypeError("Boolean DSL leaf is not supported; use 0.0 or 1.0 explicitly.")

    if isinstance(value, (int, float)):
        return jnp.asarray(float(value), dtype=free_vec.dtype)

    # Fall-through: a JAX array or numpy array passed through directly.
    try:
        return jnp.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"Cannot resolve DSL leaf of type {type(value).__name__}: {value!r}"
        ) from exc


def resolve_param_kwargs(
    params: dict | None,
    binding: ParamBinding,
    free_vec: jnp.ndarray,
) -> dict:
    """Resolve every value in a primitive/op/transform kwargs dict."""
    if not params:
        return {}
    return {k: resolve_param_value(v, binding, free_vec) for k, v in params.items()}


__all__ = [
    "ParamBinding",
    "ParamLookup",
    "make_binding",
    "resolve_param_kwargs",
    "resolve_param_value",
]
