"""Walk an SDF expression tree and produce a JAX-traceable closure.

The returned callable has signature::

    sdf(points: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray

where ``points`` has shape ``(..., 3)`` (or ``(..., 2)`` for a 2-D subtree)
and ``free_vec`` is the vector of free-parameter values in the order given
by ``Part.free_params().keys()``.

The closure is safe to ``jax.jit`` and ``jax.grad`` with respect to
``free_vec``. All DSL leaf values are resolved via
:mod:`software_defined_matter.dsl.resolve` so that both numeric literals and
``$ref`` / expression leaves flow through.

Dispatch tables at the top of the file map DSL string names to concrete
functions in :mod:`software_defined_matter.sdf.sdf_shapes`,
:mod:`software_defined_matter.sdf.sdf_ops`, and
:mod:`software_defined_matter.sdf.transforms`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import jax.numpy as jnp

from software_defined_matter import wire
from software_defined_matter.dsl.resolve import (
    ParamBinding,
    make_binding,
    resolve_param_kwargs,
    resolve_param_value,
)
from software_defined_matter.sdf import sdf_ops as ops
from software_defined_matter.sdf import sdf_shapes as shapes
from software_defined_matter.sdf import transforms

if TYPE_CHECKING:
    from software_defined_matter.model import Part, SDFTree


SDFClosure = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


class SDFEvaluator(Protocol):
    """The public closure returned by :func:`make_sdf_closure` /
    :func:`make_sdf_closure_with_binding`.

    Distinct from :data:`SDFClosure`, which the internal per-node compilers
    use (always both args, no attributes): this reflects what those two
    factories actually hand back -- ``free_vec`` is optional (defaults to
    ``binding.initial_free_vector()``), and ``.binding`` exposes the
    :class:`ParamBinding` it was compiled against, e.g. so a caller can pull
    the param order or build a correctly-shaped ``free_vec``.
    """

    binding: ParamBinding

    def __call__(self, points: jnp.ndarray, free_vec: jnp.ndarray | None = None) -> jnp.ndarray: ...


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

# Binds each contract name in software_defined_matter.wire to the
# same-named evaluator in sdf_shapes; a name with no matching function
# fails here with an AttributeError at import time. Built FROM the
# contract, so it agrees with wire.PRIMITIVES by construction --
# test_wire_contract.py checks the reverse direction (a stray sdf_shapes
# function) independently of this dict.
_PRIMITIVES: dict[str, Callable[..., jnp.ndarray]] = {
    name: getattr(shapes, name) for name in wire.PRIMITIVES
}

# The hard/smooth split is a compile-time evaluation strategy, not wire
# vocabulary, so it stays hand-listed here rather than read off
# software_defined_matter.wire.OPS.
_BINARY_OPS: dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    "union": ops.op_union,
    "subtract": ops.op_subtract,
    "intersect": ops.op_intersect,
}

_SMOOTH_BINARY_OPS: dict[str, Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    "smooth_union": ops.op_smooth_union,
    "smooth_subtract": ops.op_smooth_subtract,
    "smooth_intersect": ops.op_smooth_intersect,
}

_HARD_TO_SMOOTH: dict[str, str] = {
    "union": "smooth_union",
    "subtract": "smooth_subtract",
    "intersect": "smooth_intersect",
}

# Scalar field primitives used by the displacement DSL. These are NOT SDFs
# (their output is not a distance) and live in their own dispatch table.
# Bound the same way as _PRIMITIVES above, against the "field_" prefixed
# name sdf_shapes actually uses.
_FIELD_PRIMITIVES: dict[str, Callable[..., jnp.ndarray]] = {
    name: getattr(shapes, f"field_{name}") for name in wire.FIELD_PRIMITIVES
}

_FIELD_OPS: dict[str, Callable[..., jnp.ndarray]] = {
    "add": shapes.field_add,
}

# 2-D children a loft can SHAPE-interpolate: polygon vertices, or curve control
# points (sampled to an outline). All children of one loft must share a kind.
_SHAPE_LOFT_KINDS = {"polygon_2d", "bspline_2d", "bezier_2d"}


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


def make_sdf_closure(
    tree: SDFTree,
    part: Part,
    *,
    b_smooth_csg: bool | None = None,
    d_smooth_k: float = 0.25,
) -> SDFEvaluator:
    """Return a ``(points, free_vec) -> distance`` closure for ``tree``.

    Args:
        tree: SDF expression tree (as produced by the builders in
            :mod:`software_defined_matter.model` or loaded from a ``.sdm`` file).
        part: The Part the tree belongs to. Used to build the :class:`ParamBinding`
            and to resolve the ``smooth_csg`` metadata flag.
        b_smooth_csg: If ``None`` (default), reads
            ``part.metadata.get("smooth_csg", False)``. Pass ``True`` or
            ``False`` to override the metadata value.
    """
    if b_smooth_csg is None:
        b_smooth_csg = bool(part.metadata.get("smooth_csg", False))
    binding = make_binding(part)
    node_fn = _compile_node(
        tree,
        binding,
        b_smooth_csg=b_smooth_csg,
        d_smooth_k=d_smooth_k,
    )

    def evaluate_sdf(points: jnp.ndarray, free_vec: jnp.ndarray | None = None) -> jnp.ndarray:
        if free_vec is None:
            free_vec = binding.initial_free_vector()
        return node_fn(jnp.asarray(points), jnp.asarray(free_vec))

    evaluate_sdf.binding = binding  # type: ignore[attr-defined]
    # The dynamic attribute above is what makes evaluate_sdf satisfy
    # SDFEvaluator at runtime; mypy cannot see an attribute attached after
    # def, so the structural check on the function object itself is
    # narrowly silenced here rather than widening SDFEvaluator or SDFClosure.
    return evaluate_sdf  # type: ignore[return-value]


def make_sdf_closure_with_binding(
    tree: SDFTree,
    binding: ParamBinding,
    *,
    b_smooth_csg: bool | None = None,
    d_smooth_k: float = 0.25,
) -> SDFEvaluator:
    """Like :func:`make_sdf_closure` but reuses an existing :class:`ParamBinding`.

    Returns a closure with the same interface as :func:`make_sdf_closure`:
    ``evaluate_sdf(points, free_vec=None) -> jnp.ndarray``. When ``free_vec``
    is ``None``, ``binding.initial_free_vector()`` is used. Both inputs are
    coerced via ``jnp.asarray``. The closure also carries a ``.binding``
    attribute.

    ``b_smooth_csg=None`` resolves to ``False`` here because no ``Part`` is
    available to consult metadata. Pass ``True`` explicitly or prefer
    :func:`make_sdf_closure` (which reads ``part.metadata["smooth_csg"]``)
    when metadata-driven behaviour is desired.
    """
    if b_smooth_csg is None:
        b_smooth_csg = False
    node_fn = _compile_node(
        tree,
        binding,
        b_smooth_csg=b_smooth_csg,
        d_smooth_k=d_smooth_k,
    )

    def evaluate_sdf(points: jnp.ndarray, free_vec: jnp.ndarray | None = None) -> jnp.ndarray:
        if free_vec is None:
            free_vec = binding.initial_free_vector()
        return node_fn(jnp.asarray(points), jnp.asarray(free_vec))

    evaluate_sdf.binding = binding  # type: ignore[attr-defined]
    # The dynamic attribute above is what makes evaluate_sdf satisfy
    # SDFEvaluator at runtime; mypy cannot see an attribute attached after
    # def, so the structural check on the function object itself is
    # narrowly silenced here rather than widening SDFEvaluator or SDFClosure.
    return evaluate_sdf  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal: recursive tree walk
# ---------------------------------------------------------------------------


def _compile_node(
    node: SDFTree,
    binding: ParamBinding,
    *,
    b_smooth_csg: bool,
    d_smooth_k: float,
) -> SDFClosure:
    if not isinstance(node, dict) or "type" not in node:
        raise ValueError(f"Not an SDF node: {node!r}")

    kind = node["type"]
    if kind == "primitive":
        return _compile_primitive(node, binding)
    if kind == "op":
        return _compile_op(
            node,
            binding,
            b_smooth_csg=b_smooth_csg,
            d_smooth_k=d_smooth_k,
        )
    if kind == "transform":
        return _compile_transform(
            node,
            binding,
            b_smooth_csg=b_smooth_csg,
            d_smooth_k=d_smooth_k,
        )
    if kind == "modifier":
        return _compile_modifier(
            node,
            binding,
            b_smooth_csg=b_smooth_csg,
            d_smooth_k=d_smooth_k,
        )
    if kind == "deform":
        return _compile_deform(
            node,
            binding,
            b_smooth_csg=b_smooth_csg,
            d_smooth_k=d_smooth_k,
        )
    if kind == "2d_to_3d":
        return _compile_2d_to_3d(
            node,
            binding,
            b_smooth_csg=b_smooth_csg,
            d_smooth_k=d_smooth_k,
        )
    if kind == "loft":
        return _compile_loft(
            node,
            binding,
            b_smooth_csg=b_smooth_csg,
            d_smooth_k=d_smooth_k,
        )
    if kind == "sweep":
        return _compile_sweep(
            node,
            binding,
            b_smooth_csg=b_smooth_csg,
            d_smooth_k=d_smooth_k,
        )
    raise ValueError(f"Unknown SDF node type {kind!r}")


def _primitive_kwargs(node: dict[str, Any]) -> dict[str, Any]:
    return node.get("params") or {}


def _compile_primitive(node: dict[str, Any], binding: ParamBinding) -> SDFClosure:
    kind = node["kind"]
    if kind not in _PRIMITIVES:
        raise ValueError(f"Unknown primitive {kind!r}")
    if kind == "raster_field":
        return _compile_raster_field(node)
    prim_fn = _PRIMITIVES[kind]
    raw_kwargs = _primitive_kwargs(node)

    def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
        resolved = resolve_param_kwargs(raw_kwargs, binding, free_vec)
        return prim_fn(p, **resolved)

    return fn


def _compile_raster_field(node: dict[str, Any]) -> SDFClosure:
    """Decode the base64 sample block ONCE at compile time (DR-0003).

    Grid params are topology-class — a ``$ref`` here is rejected the same way
    swept-op pose counts are, not treated as a missing live-uniform feature.
    """
    from software_defined_matter.sdf.raster import (
        decode_raster_values,
        normalize_spacing,
        raster_origin,
        require_literal_raster_params,
    )

    params = _primitive_kwargs(node)
    require_literal_raster_params(params)
    values = jnp.asarray(decode_raster_values(params))
    origin = jnp.asarray(raster_origin(params))
    spacing = jnp.asarray(normalize_spacing(params))

    def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
        return shapes.raster_field(p, origin, spacing, values)

    return fn


def _op_kwargs(node: dict[str, Any]) -> dict[str, Any]:
    return node.get("params") or {}


def _static_count(raw: Any, binding: ParamBinding, *, what: str) -> int:
    """A count that has to be a Python ``int`` when the closure is BUILT.

    For the handful of slots that size a Python-level loop or slice rather than
    entering the arithmetic. ``softmin_chunked``'s chunk size drives
    ``range(0, n, cs)``, so no traced value can work there at any price.

    Contrast ``canonical_sector_fold``'s ``n_sectors``, which only ever divides
    (``2*pi/n``) and therefore stays traced. Reaching for this helper where a
    value could stay traced is a real cost: it forbids a ``$ref`` to a free
    param, and the caller loses the ability to scrub that value live.

    Resolved from ``binding`` rather than from ``free_vec``, so a ``$ref`` to a
    NON-free param still works -- its value is a constant of the ``Part``, and
    refusing it would rule out the common authoring pattern where a count is a
    named param rather than a number spelled at the call site.

    Args:
        raw: The slot's authored value: a number, or a ``{"$ref": name}`` leaf.
        binding: The Part's param binding.
        what: Names the slot in any error, e.g. ``"softmin_chunked chunk_size"``.

    Returns:
        int: The resolved count.

    Raises:
        ValueError: If the slot references a free param, or carries an
            expression tree, neither of which has a value at build time.
    """
    if isinstance(raw, bool):
        raise ValueError(f"{what} must be a number, got a bool")
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, dict) and "$ref" in raw:
        name = raw["$ref"]
        if name in binding.fixed:
            return int(binding.fixed[name])
        if name in binding.free_idx:
            raise ValueError(
                f"{what} references the free param {name!r}. It sizes a Python "
                "loop, so it must have a value when the closure is built, and a "
                "free param only has one per evaluation. Mark it free=False."
            )
        raise ValueError(f"{what} references unknown param {name!r}")
    raise ValueError(f"{what} must be a number or a $ref to a non-free param, got {raw!r}")


def _compile_op(
    node: dict[str, Any],
    binding: ParamBinding,
    *,
    b_smooth_csg: bool,
    d_smooth_k: float,
) -> SDFClosure:
    op_name = node["op"]
    children: list[SDFClosure] = [
        _compile_node(c, binding, b_smooth_csg=b_smooth_csg, d_smooth_k=d_smooth_k)
        for c in node["children"]
    ]
    if not children:
        raise ValueError("CSG op has no children")

    kwargs = _op_kwargs(node)

    if op_name in _BINARY_OPS:
        if b_smooth_csg:
            smooth_name = _HARD_TO_SMOOTH[op_name]
            combine_smooth = _SMOOTH_BINARY_OPS[smooth_name]
            d_smooth_k_arr = jnp.asarray(d_smooth_k)

            def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
                d = children[0](p, free_vec)
                for c in children[1:]:
                    d = combine_smooth(d, c(p, free_vec), d_smooth_k_arr)
                return d

            return fn

        combine = _BINARY_OPS[op_name]

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            d = children[0](p, free_vec)
            for c in children[1:]:
                d = combine(d, c(p, free_vec))
            return d

        return fn

    if op_name in _SMOOTH_BINARY_OPS:
        combine_k = _SMOOTH_BINARY_OPS[op_name]

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            k = resolve_param_value(kwargs["k"], binding, free_vec)
            d = children[0](p, free_vec)
            for c in children[1:]:
                d = combine_k(d, c(p, free_vec), k)
            return d

        return fn

    # Both produce a single SDF value from N children softmin-unioned with
    # matched k; softmin_chunked caps peak memory via ops.reduce_softmin_chunked
    # instead of stacking all N children at once.
    if op_name == "softmin_many":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            k = resolve_param_value(kwargs["k"], binding, free_vec)
            stacked = jnp.stack([c(p, free_vec) for c in children], axis=0)
            return ops.op_softmin_many(stacked, k)

        return fn

    if op_name == "softmin_chunked":
        # Resolved at BUILD time: it sizes `range(0, n, cs)` inside
        # `reduce_softmin_chunked`, and `int()` on a tracer raises
        # ConcretizationTypeError, which took the whole closure off under
        # `jax.jit` -- so meshing and export too (same defect as the
        # fold below).
        cs = max(
            1,
            _static_count(kwargs.get("chunk_size", 64), binding, what="softmin_chunked chunk_size"),
        )

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            k = resolve_param_value(kwargs["k"], binding, free_vec)
            n = len(children)

            def values_fn(start: int, end: int) -> jnp.ndarray:
                return jnp.stack([c(p, free_vec) for c in children[start:end]], axis=0)

            return ops.reduce_softmin_chunked(values_fn, n, k, cs)

        return fn

    raise ValueError(f"Unknown CSG op {op_name!r}")


def _compile_transform(
    node: dict[str, Any],
    binding: ParamBinding,
    *,
    b_smooth_csg: bool,
    d_smooth_k: float,
) -> SDFClosure:
    tf_name = node["transform"]
    child = _compile_node(node["child"], binding, b_smooth_csg=b_smooth_csg, d_smooth_k=d_smooth_k)
    raw_kwargs = _op_kwargs(node)

    if tf_name == "translate":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            t = resolve_param_value(raw_kwargs["t"], binding, free_vec)
            return child(transforms.tf_translate(p, t), free_vec)

        return fn

    if tf_name == "scale":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            s = resolve_param_value(raw_kwargs["s"], binding, free_vec)
            # Remember to divide the result by the scale factor.
            return child(transforms.tf_scale(p, s), free_vec) * s

        return fn

    if tf_name == "scale_axis":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            s = jnp.asarray(
                [resolve_param_value(v, binding, free_vec) for v in raw_kwargs["s"]], dtype=p.dtype
            )
            # Divide by the SMALLEST factor: the only one that keeps a bound.
            return child(p / s, free_vec) * jnp.min(s)

        return fn

    if tf_name == "rotate_x":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            angle = resolve_param_value(raw_kwargs["angle"], binding, free_vec)
            return child(transforms.tf_rotate_x(p, angle), free_vec)

        return fn

    if tf_name == "rotate_y":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            angle = resolve_param_value(raw_kwargs["angle"], binding, free_vec)
            return child(transforms.tf_rotate_y(p, angle), free_vec)

        return fn

    if tf_name == "rotate_z":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            angle = resolve_param_value(raw_kwargs["angle"], binding, free_vec)
            return child(transforms.tf_rotate_z(p, angle), free_vec)

        return fn

    if tf_name == "rotate_matrix":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            R = resolve_param_value(raw_kwargs["R"], binding, free_vec)
            return child(transforms.tf_rotate(p, R), free_vec)

        return fn

    if tf_name == "repeat_finite":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            c = resolve_param_value(raw_kwargs["c"], binding, free_vec)
            lim = resolve_param_value(raw_kwargs["l"], binding, free_vec)
            return child(transforms.op_repeat_finite(p, c, lim), free_vec)

        return fn

    if tf_name == "canonical_sector_fold":
        b_centered = bool(raw_kwargs.get("centered", False))

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            # A count, so a `$ref` scrubbed to 4.7 gives 4 wedges. glsl/emit.py
            # emits `floor(n)` on a float for the same reason, so preview and
            # mesh agree.
            #
            # FLOORED, NOT `int()`-ED, and that is the whole fix.
            # `int()` on a traced value raises ConcretizationTypeError, so every
            # part using this fold could be sampled point-by-point in eager mode
            # and could not be jitted -- which meant it could not be meshed,
            # exported or previewed, because `export_part` meshes through a
            # jitted grid evaluation. Reported 2026-08-09 against an 8-legged
            # radially symmetric body, exactly what the transform is for.
            #
            # It can stay traced because `n_sectors` never leaves the
            # arithmetic: it sets `2*pi/n` here and in `tf_canonical_sector_fold`
            # and is never an index, a shape or a branch. So a `$ref` to a FREE
            # param keeps working and the count can be scrubbed live, which
            # hoisting it to a build-time `int` would have forbidden --
            # `radial_bearing` alone authors 40 of these as `$ref`s.
            n_sectors = jnp.floor(resolve_param_value(raw_kwargs["n_sectors"], binding, free_vec))
            phase_frac = resolve_param_value(raw_kwargs.get("phase_frac", 0.0), binding, free_vec)
            q = transforms.tf_canonical_sector_fold(
                p, n_sectors, centered=b_centered, phase_frac=phase_frac
            )
            # Evaluate the child at the folded point and at that point rotated
            # one sector each way, then take the nearest. So the child may be
            # wider than its wedge, but must be authored within a sector of it.
            #
            # Folding only asks how far away your own copy is. Near a wedge
            # boundary the closest surface is the copy next door, so a single
            # evaluation reports too large a distance. That is the unsafe
            # direction: a sphere tracer steps by the distance it is given and
            # will cross a surface it was told was further off.
            #
            # glsl/emit.py emits the same three calls; keep them in lockstep.
            sector = 2.0 * jnp.pi / n_sectors
            d = child(q, free_vec)
            d = jnp.minimum(d, child(transforms.tf_rotate_z(q, sector), free_vec))
            return jnp.minimum(d, child(transforms.tf_rotate_z(q, -sector), free_vec))

        return fn

    # Union of the child with its reflection across the plane: robust to
    # where the child sits, since both p and its full reflection are
    # evaluated. Obeys `b_smooth_csg`.
    if tf_name == "mirror":
        if b_smooth_csg:
            combine_smooth = _SMOOTH_BINARY_OPS[_HARD_TO_SMOOTH["union"]]
            d_smooth_k_arr = jnp.asarray(d_smooth_k)

            def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
                n = resolve_param_value(raw_kwargs["n"], binding, free_vec)
                o = resolve_param_value(raw_kwargs["o"], binding, free_vec)
                p_reflected = transforms.reflect_plane(p, n, o)
                return combine_smooth(
                    child(p, free_vec), child(p_reflected, free_vec), d_smooth_k_arr
                )

            return fn

        combine = _BINARY_OPS["union"]

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            n = resolve_param_value(raw_kwargs["n"], binding, free_vec)
            o = resolve_param_value(raw_kwargs["o"], binding, free_vec)
            p_reflected = transforms.reflect_plane(p, n, o)
            return combine(child(p, free_vec), child(p_reflected, free_vec))

        return fn

    raise ValueError(f"Unknown transform {tf_name!r}")


def _compile_modifier(
    node: dict[str, Any],
    binding: ParamBinding,
    *,
    b_smooth_csg: bool,
    d_smooth_k: float,
) -> SDFClosure:
    name = node["modifier"]
    child = _compile_node(node["child"], binding, b_smooth_csg=b_smooth_csg, d_smooth_k=d_smooth_k)
    raw_kwargs = _op_kwargs(node)

    if name == "round":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            r = resolve_param_value(raw_kwargs["r"], binding, free_vec)
            return ops.op_round(child(p, free_vec), r)

        return fn

    if name == "onion":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            thickness = resolve_param_value(raw_kwargs["thickness"], binding, free_vec)
            return ops.op_onion(child(p, free_vec), thickness)

        return fn

    if name == "elongate":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            h = resolve_param_value(raw_kwargs["h"], binding, free_vec)
            return ops.op_elongate(lambda q: child(q, free_vec), p, h)

        return fn

    raise ValueError(f"Unknown modifier {name!r}")


def _compile_deform(
    node: dict[str, Any],
    binding: ParamBinding,
    *,
    b_smooth_csg: bool,
    d_smooth_k: float,
) -> SDFClosure:
    name = node["deform"]
    child = _compile_node(node["child"], binding, b_smooth_csg=b_smooth_csg, d_smooth_k=d_smooth_k)
    raw_kwargs = _op_kwargs(node)

    if name == "twist":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            k = resolve_param_value(raw_kwargs["k"], binding, free_vec)
            return ops.op_twist(lambda q: child(q, free_vec), p, k)

        return fn

    if name == "bend":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            k = resolve_param_value(raw_kwargs["k"], binding, free_vec)
            return ops.op_bend(lambda q: child(q, free_vec), p, k)

        return fn

    if name == "twist_radial":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            r0 = resolve_param_value(raw_kwargs["r0"], binding, free_vec)
            r1 = resolve_param_value(raw_kwargs["r1"], binding, free_vec)
            a0 = resolve_param_value(raw_kwargs["angle_inner"], binding, free_vec)
            a1 = resolve_param_value(raw_kwargs["angle_outer"], binding, free_vec)
            return ops.op_twist_radial(lambda q: child(q, free_vec), p, r0, r1, a0, a1)

        return fn

    if name == "twist_linear":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            # `axis` is optional and defaults to +X. Each component resolves
            # separately: a vec2 slot takes a $ref per element, like every
            # other vector slot in the wire format.
            axis = raw_kwargs.get("axis", [1.0, 0.0])
            u_axis = [
                resolve_param_value(axis[0], binding, free_vec),
                resolve_param_value(axis[1], binding, free_vec),
            ]
            u0 = resolve_param_value(raw_kwargs["u0"], binding, free_vec)
            u1 = resolve_param_value(raw_kwargs["u1"], binding, free_vec)
            a0 = resolve_param_value(raw_kwargs["angle_0"], binding, free_vec)
            a1 = resolve_param_value(raw_kwargs["angle_1"], binding, free_vec)
            return ops.op_twist_linear(lambda q: child(q, free_vec), p, u_axis, u0, u1, a0, a1)

        return fn

    if name == "shear_linear":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            axis = raw_kwargs.get("axis", [1.0, 0.0])
            u_axis = [
                resolve_param_value(axis[0], binding, free_vec),
                resolve_param_value(axis[1], binding, free_vec),
            ]
            u0 = resolve_param_value(raw_kwargs["u0"], binding, free_vec)
            u1 = resolve_param_value(raw_kwargs["u1"], binding, free_vec)
            dz0 = resolve_param_value(raw_kwargs["dz_0"], binding, free_vec)
            dz1 = resolve_param_value(raw_kwargs["dz_1"], binding, free_vec)
            return ops.op_shear_linear(lambda q: child(q, free_vec), p, u_axis, u0, u1, dz0, dz1)

        return fn

    if name == "taper_linear":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            z0 = resolve_param_value(raw_kwargs["z0"], binding, free_vec)
            z1 = resolve_param_value(raw_kwargs["z1"], binding, free_vec)
            s0 = resolve_param_value(raw_kwargs["s_0"], binding, free_vec)
            s1 = resolve_param_value(raw_kwargs["s_1"], binding, free_vec)
            return ops.op_taper_linear(lambda q: child(q, free_vec), p, z0, z1, s0, s1)

        return fn

    if name == "displace":
        if "field" not in node:
            raise ValueError("deform 'displace' requires a 'field' subtree on the node")
        field_fn = _compile_field_node(node["field"], binding)

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            return ops.op_displace(
                lambda q: child(q, free_vec),
                lambda q: field_fn(q, free_vec),
                p,
            )

        return fn

    raise ValueError(f"Unknown / unsupported deform {name!r}")


def _compile_2d_to_3d(
    node: dict[str, Any],
    binding: ParamBinding,
    *,
    b_smooth_csg: bool,
    d_smooth_k: float,
) -> SDFClosure:
    method = node["method"]
    child = _compile_node(node["child"], binding, b_smooth_csg=b_smooth_csg, d_smooth_k=d_smooth_k)
    raw_kwargs = _op_kwargs(node)

    if method == "revolution":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            offset_raw = raw_kwargs.get("offset", 0.0)
            offset = resolve_param_value(offset_raw, binding, free_vec)
            return ops.revolution(lambda q2d: child(q2d, free_vec), p, offset)

        return fn

    if method == "extrusion":

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            h = resolve_param_value(raw_kwargs["h"], binding, free_vec)
            return ops.extrusion(lambda q2d: child(q2d, free_vec), p, h)

        return fn

    raise ValueError(f"Unknown 2d_to_3d method {method!r}")


def _is_static_z_list(z_raw: Any) -> bool:
    """True when every loft station is a numeric literal (not a ``$ref``)."""
    if not isinstance(z_raw, list):
        return False
    return all(isinstance(z, (int, float)) and not isinstance(z, bool) for z in z_raw)


def _validate_loft_stations(z_stations: Any, n_sections: int) -> None:
    """Raise if loft Z stations cannot define a valid piecewise span."""
    import numpy as np

    z = np.asarray(z_stations, dtype=float).reshape(-1)
    if z.size != n_sections:
        raise ValueError(
            f"loft requires len(z) == len(children): got {z.size} z stations "
            f"for {n_sections} cross-sections"
        )
    if z.size >= 2 and np.any(np.diff(z) <= 0.0):
        raise ValueError(
            "loft requires strictly ascending z stations (no duplicates or "
            f"decreasing values); got z={z.tolist()}"
        )


def _compile_loft(
    node: dict[str, Any],
    binding: ParamBinding,
    *,
    b_smooth_csg: bool,
    d_smooth_k: float,
) -> SDFClosure:
    """Loft N 2D cross-section children along Z at the stations in ``params['z']``
    (see :func:`software_defined_matter.sdf.sdf_ops.loft`)."""
    children = [
        _compile_node(c, binding, b_smooth_csg=b_smooth_csg, d_smooth_k=d_smooth_k)
        for c in node["children"]
    ]
    if len(children) < 2:
        raise ValueError("loft needs at least 2 cross-section children")
    raw_kwargs = _op_kwargs(node)
    if "z" not in raw_kwargs:
        raise ValueError("loft requires a 'z' param: one axial station per section")
    smooth = bool(raw_kwargs.get("smooth", False))  # structural flag, not param-resolved
    interp = str(raw_kwargs.get("interp", "field"))  # 'field' (default) | 'shape'
    if interp not in ("field", "shape"):
        raise ValueError(f"loft interp must be 'field' or 'shape', got {interp!r}")

    raw_z = raw_kwargs["z"]
    if _is_static_z_list(raw_z):
        _validate_loft_stations(raw_z, len(children))
    # NOTE: param-driven ($ref) stations can't be checked here: their values
    # aren't known until eval. The caller must keep them strictly ascending; a
    # non-ascending resolved z breaks searchsorted and yields empty geometry.

    if interp == "shape":
        # SHAPE interpolation: interpolate the section OUTLINE (not the distance
        # field) between stations, then take the exact polygon SDF of the
        # interpolated outline: eliminates the convex-feature bulge that field
        # interpolation produces at a swept edge. Children must all be the SAME
        # kind (vertex/control-point correspondence): polygon_2d (interpolate
        # 'vertices') OR bspline_2d / bezier_2d (sample each section's
        # 'control_points' to its outline, then interpolate those outline vertices
        # via loft_shape, inheriting the same no-bulge/watertight guarantees while
        # the DSL stays compact). Sampling is a linear map, so for smooth=False this
        # equals interpolating the control points then sampling; for smooth=True the
        # PCHIP interp is nonlinear, so the outlines are what get interpolated.
        nodes = node["children"]
        kinds = {c.get("kind") for c in nodes}
        if any(c.get("type") != "primitive" for c in nodes) or not kinds <= _SHAPE_LOFT_KINDS:
            raise ValueError(
                "loft interp='shape' requires polygon_2d / bspline_2d / bezier_2d "
                f"children, got {sorted(k or '?' for k in kinds)}"
            )
        if len(kinds) != 1:
            raise ValueError(
                "loft interp='shape' requires all children to be the SAME kind "
                f"(outline correspondence); got {sorted(kinds)}"
            )
        kind = kinds.pop()
        key = "vertices" if kind == "polygon_2d" else "control_points"
        noun = "vertex" if kind == "polygon_2d" else "control-point"
        prim_kwargs = [_primitive_kwargs(c) for c in nodes]
        if any(key not in pk for pk in prim_kwargs):
            raise ValueError(f"loft interp='shape': each {kind} child needs '{key}'")
        samples = shapes._CURVE_SAMPLES_PER_SEGMENT

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            zs = resolve_param_value(raw_kwargs["z"], binding, free_vec)
            arrs = [
                jnp.asarray(resolve_param_kwargs(pk, binding, free_vec)[key]) for pk in prim_kwargs
            ]
            counts = {int(a.shape[0]) for a in arrs}
            if len(counts) != 1:
                raise ValueError(
                    f"loft interp='shape' needs equal {noun} counts across sections "
                    f"(correspondence); got {sorted(counts)}"
                )
            if kind == "polygon_2d":
                return ops.loft_shape(arrs, zs, p, smooth=smooth)
            return ops.loft_shape_curve(arrs, zs, p, kind, samples, smooth=smooth)

        return fn

    def fn_field(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
        zs = resolve_param_value(raw_kwargs["z"], binding, free_vec)
        sdf2d_fns = [(lambda q2d, c=c: c(q2d, free_vec)) for c in children]
        return ops.loft(sdf2d_fns, zs, p, smooth=smooth)

    return fn_field


def _compile_sweep(
    node: dict[str, Any],
    binding: ParamBinding,
    *,
    b_smooth_csg: bool,
    d_smooth_k: float,
) -> SDFClosure:
    child = _compile_node(node["child"], binding, b_smooth_csg=b_smooth_csg, d_smooth_k=d_smooth_k)
    raw_kwargs = _op_kwargs(node)
    if "path" not in raw_kwargs:
        raise ValueError("sweep requires a 'path' param: a list of [x, y, z] control points")
    kind = str(raw_kwargs.get("path_kind", "bspline"))  # structural, not param-resolved
    frame = str(raw_kwargs.get("frame", "rmf"))  # structural, not param-resolved
    # bspline paths are periodic loops; honour an explicit closed flag otherwise.
    closed = bool(raw_kwargs.get("closed", False)) or kind == "bspline"
    has_normal0 = "normal0" in raw_kwargs

    def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
        ctrl = resolve_param_value(raw_kwargs["path"], binding, free_vec)  # (M, 3)
        n0 = resolve_param_value(raw_kwargs["normal0"], binding, free_vec) if has_normal0 else None
        path_pts = ops._sample_path_3d(ctrl, kind, closed)
        return ops.sweep(
            lambda q2d: child(q2d, free_vec), path_pts, p, closed=closed, frame=frame, normal0=n0
        )

    return fn


# ---------------------------------------------------------------------------
# Field-tree compiler (displacement-field DSL)
# ---------------------------------------------------------------------------
# Field trees are parallel to SDF trees: each compiled node returns a
# ``(points, free_vec) -> scalar`` closure. They are NOT SDFs (the scalar
# value is a displacement amplitude, not a distance) and live in their own
# dispatch tables.


def _compile_field_node(node: Any, binding: ParamBinding) -> SDFClosure:
    if not isinstance(node, dict) or "type" not in node:
        raise ValueError(f"Not a field node: {node!r}")

    kind = node["type"]
    if kind == "field":
        prim_name = node["kind"]
        if prim_name not in _FIELD_PRIMITIVES:
            raise ValueError(f"Unknown field primitive {prim_name!r}")
        prim_fn = _FIELD_PRIMITIVES[prim_name]
        raw_kwargs = _primitive_kwargs(node)

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            resolved = resolve_param_kwargs(raw_kwargs, binding, free_vec)
            return prim_fn(p, **resolved)

        return fn

    if kind == "field_op":
        op_name = node["op"]
        if op_name not in _FIELD_OPS:
            raise ValueError(f"Unknown field op {op_name!r}")
        combine = _FIELD_OPS[op_name]
        children: list[SDFClosure] = [_compile_field_node(c, binding) for c in node["children"]]
        if not children:
            raise ValueError("field_op has no children")

        def fn(p: jnp.ndarray, free_vec: jnp.ndarray) -> jnp.ndarray:
            return combine(*[c(p, free_vec) for c in children])

        return fn

    raise ValueError(f"Unknown field node type {kind!r}")


__all__ = [
    "SDFClosure",
    "SDFEvaluator",
    "make_sdf_closure",
    "make_sdf_closure_with_binding",
]
