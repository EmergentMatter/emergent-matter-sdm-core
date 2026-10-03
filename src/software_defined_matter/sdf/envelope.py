"""The solid a part would be if you filled in its holes.

``relative_density`` needs a denominator. Using a bounding box makes a solid
ball report 52% dense, because a sphere fills only ``pi/6`` of its own tight
cube. The number a designer wants is material over *design volume*: the outer
shape the part occupies, with its porosity filled back in.

:func:`infer_sdf_envelope` walks an SDF tree and rewrites it into that outer
shape, mirroring the recursive structure of
:mod:`software_defined_matter.sdf.bbox` (which walks the same trees to return a
box) and :mod:`software_defined_matter.sdf.lipschitz` (which returns a rate).

The rules
---------
Two node kinds *remove* material, and the walk drops them::

    subtract(A, B)   ->  envelope(A)      the removed branch is porosity
    onion(A, t)      ->  envelope(A)      a shell is measured against its solid

Two node kinds have no outer surface at all, because their pores run straight
through, and the walk replaces them with the box they were designed to fill::

    gyroid / schwarz_p / schwarz_d / neovius / lidinoid
    repeat_finite(...)

Everything else keeps its own node and recurses into its children, so a
``union`` of two lattices is the union of their two design boxes, and an
``intersect`` clips a lattice down to whatever bounds it.

Worked example, ``gyroid`` clipped to a sphere::

    intersect(gyroid(period=5, n_periods=[5,5,5]), sphere(r=9))
      ->  intersect(box(b=[12.5, 12.5, 12.5]), sphere(r=9))
      ->  a sphere of radius 9

and the density comes out 0.720, the gyroid's own fill fraction, instead of
0.377, which is an artefact of whatever box you happened to sample in.

What this deliberately does not do
----------------------------------
It cannot tell a hole from a shaping cut. ``subtract(sphere, cylinder)`` used
to flatten a sphere's cap produces a *solid*, which should read 1.0, but the
walk treats the cut as porosity and reports 0.972. Distinguishing the two is
not possible from the tree: a plate's through-holes break the surface exactly
as the flattening cut does, and those must count as porosity. Every subtraction
is therefore porosity, by decision.

Lattice domains are frozen
--------------------------
A lattice's design box is emitted as a numeric ``box`` primitive, evaluated at
the params' *current* values via :func:`~software_defined_matter.sdf.bbox
.infer_sdf_bbox`. That subtree therefore contributes nothing to
``d(density)/d(param)``. Everything else in the envelope stays symbolic and
differentiable, so a lattice clipped to a param-driven sphere still moves with
the sphere. Re-derive the envelope each outer optimiser iteration, exactly as
the sampling box is re-derived.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from software_defined_matter import wire
from software_defined_matter.sdf.bbox import BBoxInferenceError, infer_sdf_bbox

if TYPE_CHECKING:
    from software_defined_matter.model import Part, SDFTree


class EnvelopeInferenceError(ValueError):
    """Raised when a tree has no finite envelope."""


#: Primitives that are lattices by definition: a sheet repeating through a
#: clip box, with no outer surface of their own. Replaced by that clip box.
_LATTICE_PRIMITIVES = wire.LATTICE_PRIMITIVES

#: Ops whose second and later children remove material.
_SUBTRACT_OPS = wire.SUBTRACT_OPS

#: Ops that combine children without removing anything, so the envelope is the
#: same op over the children's envelopes: every registered op that isn't one
#: of the subtract-like ones above. Read from the contract rather than
#: hand-listed, so a new op in ``wire.OPS`` is combining by default and only
#: needs a mention here if it turns out to remove material instead.
_COMBINING_OPS = frozenset(wire.OPS) - _SUBTRACT_OPS

#: Modifiers that hollow the child out. Dropped, like ``subtract``.
_HOLLOWING_MODIFIERS = wire.HOLLOWING_MODIFIERS

#: Transforms that tile a child through a finite domain. Replaced by that
#: domain, for the same reason as ``_LATTICE_PRIMITIVES``.
_TILING_TRANSFORMS = wire.TILING_TRANSFORMS

#: Node kinds that wrap exactly one child under the ``child`` key and pass the
#: envelope through unchanged. ``deform``'s ``field`` slot is a displacement
#: field, not an SDF subtree, so it is carried over untouched.
_PASSTHROUGH_TYPES = frozenset({"deform", "2d_to_3d", "sweep"})

#: Node kinds that hold several children under ``children`` and are not CSG
#: ops. ``loft`` interpolates N 2-D profiles along Z.
_PASSTHROUGH_MULTI_TYPES = frozenset({"loft"})


def infer_sdf_envelope(tree: SDFTree, part: Part) -> SDFTree:
    """Return the SDF tree of ``tree``'s envelope: the same solid, holes filled.

    The result is a fresh tree. ``tree`` is never mutated, and subtrees that
    survive unchanged are copied rather than shared, so a caller may edit
    either independently.

    Raises :class:`EnvelopeInferenceError` for geometry with no finite
    envelope, today meaning ``repeat_inf``.
    """
    return _node_envelope(tree, part)


# ---------------------------------------------------------------------------
# Internal: recursive tree walk
# ---------------------------------------------------------------------------


def _node_envelope(node: Any, part: Part) -> SDFTree:
    if not isinstance(node, dict) or "type" not in node:
        raise EnvelopeInferenceError(f"Not an SDF node: {node!r}")

    node_type = node["type"]

    if node_type == "primitive":
        if node.get("kind") in _LATTICE_PRIMITIVES:
            return _design_box(node, part)
        return _copy(node)

    if node_type == "op":
        return _op_envelope(node, part)

    if node_type == "transform":
        return _transform_envelope(node, part)

    if node_type == "modifier":
        if node.get("modifier") in _HOLLOWING_MODIFIERS:
            # A shell's envelope is the solid it was hollowed out of, so the
            # modifier is dropped and only its child survives.
            return _node_envelope(node["child"], part)
        return _wrap_child(node, part)

    if node_type in _PASSTHROUGH_TYPES:
        return _wrap_child(node, part)

    if node_type in _PASSTHROUGH_MULTI_TYPES:
        out = _copy(node)
        out["children"] = [_node_envelope(c, part) for c in node.get("children") or []]
        return out

    raise EnvelopeInferenceError(
        f"Unknown node type {node_type!r}; no envelope rule for it. "
        "Add one in software_defined_matter.sdf.envelope."
    )


def _op_envelope(node: dict[str, Any], part: Part) -> SDFTree:
    op = node["op"]
    children: list[Any] = node.get("children") or []
    if not children:
        raise EnvelopeInferenceError(f"CSG op {op!r} has no children")

    if op in _SUBTRACT_OPS:
        # Everything after the first child is removed material, which is what
        # the density is measuring. Only the minuend survives.
        return _node_envelope(children[0], part)

    if op in _COMBINING_OPS:
        out = _copy(node)
        out["children"] = [_node_envelope(c, part) for c in children]
        return out

    raise EnvelopeInferenceError(
        f"Unsupported op {op!r} for envelope inference. Add it to _SUBTRACT_OPS or _COMBINING_OPS."
    )


def _transform_envelope(node: dict[str, Any], part: Part) -> SDFTree:
    tf = node["transform"]

    if tf == "repeat_inf":
        raise EnvelopeInferenceError(
            "repeat_inf tiles forever, so it has no finite envelope and no "
            "finite denominator for relative_density. Wrap it in an "
            "`intersect` with a bounded shape, or pass an explicit "
            "`envelope=` to the metric."
        )

    if tf in _TILING_TRANSFORMS:
        return _design_box(node, part)

    return _wrap_child(node, part)


def _wrap_child(node: dict[str, Any], part: Part) -> SDFTree:
    """Keep ``node`` as-is, replacing its child with the child's envelope."""
    out = _copy(node)
    out["child"] = _node_envelope(node["child"], part)
    return out


def _design_box(node: dict[str, Any], part: Part) -> SDFTree:
    """The solid box a lattice node was designed to fill.

    Read off :func:`~software_defined_matter.sdf.bbox.infer_sdf_bbox` at the
    params' current values, which already knows both the TPMS clip box
    (``0.5 * n_periods * period``) and the ``repeat_finite`` tiling domain
    (the child inflated by ``c * l``). Numeric, hence frozen; see the module
    docstring.
    """
    from software_defined_matter.model import sdf_primitive, sdf_transform

    try:
        (x0, y0, z0), (x1, y1, z1) = infer_sdf_bbox(node, part, mode="values")
    except BBoxInferenceError as exc:
        raise EnvelopeInferenceError(
            f"Cannot size the design domain of this lattice node: {exc}"
        ) from exc

    half = [0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * (z1 - z0)]
    centre = [0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)]
    box = sdf_primitive("box", b=half)
    if any(abs(c) > 1e-12 for c in centre):
        return sdf_transform("translate", box, t=centre)
    return box


def _copy(node: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy that also copies the ``params`` dict.

    Structural slots (``child`` / ``children``) are overwritten by the caller,
    and leaf ``params`` are copied so an edit to the envelope cannot reach
    back into the original tree.
    """
    out = dict(node)
    if isinstance(out.get("params"), dict):
        out["params"] = dict(out["params"])
    return out


__all__ = ["EnvelopeInferenceError", "infer_sdf_envelope"]
