"""Walk an SDF expression tree and emit GLSL source for ray-march visualisation.

The emitter mirrors the structure of
:mod:`software_defined_matter.sdf.compile` so the two stay in lockstep when
the DSL is extended: every primitive / op / transform / modifier / deform /
2d_to_3d / field kind that the JAX compiler handles must also have an entry
here, plus a matching GLSL function in ``lib.glsl``. The phase-2 parity test
walks the same fixture trees through both pipelines and asserts identical
distance fields at a sampled grid (float32-tolerance).

What the emitter does NOT do
----------------------------
* No ray-march loop, no normal estimation, no shading. The output is a
  ``float sdf_scene(vec3 p)`` function plus a static helper library; the
  Blender addon (or any host) supplies the rendering wrapper.
* No per-pixel metric queries. Arithmetic expressions compile to shader code;
  design references and authored motion inputs become distinct uniforms.
* No bbox handling for unbounded geometry. Where ``infer_sdf_bbox`` raises,
  the CLI surfaces the same message so the user adds ``metadata["bbox"]``
  explicitly.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from software_defined_matter.model import Part, SDFTree
from software_defined_matter.sdf.bbox import BBox, infer_sdf_bbox

# The 2-D curve primitives, which have no PrimSpec because they are not lib
# functions: each evaluates as the exact polygon SDF of its own sampled
# outline, so the emitter samples once and reuses the polygon path.
_CURVE_KINDS = ("bezier_2d", "bspline_2d")

# What `interp='shape'` will accept as a section. Curve sections are refused,
# and `_emit_loft` says why.
_SHAPE_LOFT_KINDS = ("polygon_2d",)

# Above _POLY_TABLE_THRESHOLD number of polygon vertices in one tree,
# outlines move OUT of the program text and into a flat table
# that the viewer will binds (see lib.glsl's SDM_POLY_TABLE block).
_POLY_TABLE_THRESHOLD = 512

# Default row width for the table, and only a DEFAULT: lib.glsl guards it with
# `#ifndef` and a host that packs differently defines its own. Per the emitter
# remit, sdm-core says what is fetched and the host decides how it is packed.
_POLY_TEX_WIDTH = 1024

# Same role for the sweep frame table (lib.glsl's SDM_SWEEP_TABLE block), and
# the same whole-tree budget decides it: `_POLY_TABLE_THRESHOLD` sampled path
# points across every sweep in one emission.
_SWEEP_TEX_WIDTH = 1024


def _is_literal_point(v: Any, n: int) -> bool:
    """An ``n``-vector whose coordinates are numbers, not `$ref`s or expressions.

    Only these can be baked: a table texel is uploaded once and an inline
    frame array is emitted once, so a coordinate the host can still scrub has
    to stay a uniform in the program text.
    """
    return (
        isinstance(v, (list, tuple, np.ndarray))
        and len(v) == n
        and all(
            isinstance(c, (int, float, np.integer, np.floating)) and not isinstance(c, bool)
            for c in v
        )
    )


def _is_literal_vertex(v: Any) -> bool:
    """A 2-D outline vertex whose coordinates are numbers; see `_is_literal_point`."""
    return _is_literal_point(v, 2)


def _curve_samples() -> int:
    """Samples per curve span, read from the shapes module rather than pinned.

    A copy of this number here would put the emitter's outline at a different
    resolution from the JAX field's the first time someone tuned it, and the
    two would disagree by a chord error that looks like a tolerance problem.
    """
    from software_defined_matter.sdf import sdf_shapes

    return int(sdf_shapes._CURVE_SAMPLES_PER_SEGMENT)


def _curve_outline(kind: str, control_points: Any, samples: int) -> list[list[float]]:
    """Sample a curve primitive's control points to its outline polygon.

    Calls the SAME helpers `sdf_ops` calls, so the emitted outline is the one
    the JAX field evaluates by construction. Transcribing the two bases into
    this file instead would be a second copy of a fixed basis, which is the
    drift this emitter's parity tests exist to catch.

    Args:
        kind: ``"bezier_2d"`` or ``"bspline_2d"``.
        control_points: The node's ``control_points``, numeric literals only.
        samples: Samples per span, from :func:`_curve_samples`.

    Returns:
        list[list[float]]: ``[[x, y], ...]`` around the closed outline.
    """
    from software_defined_matter.sdf._helpers import bezier_outline, bspline_outline

    outline = bspline_outline if kind == "bspline_2d" else bezier_outline
    pts = np.asarray(outline(np.asarray(control_points, dtype=float), samples))
    return [[float(x), float(y)] for x, y in pts]


# ---------------------------------------------------------------------------
# Primitive spec tables
# ---------------------------------------------------------------------------
# For each DSL primitive kind, list (arg_name, glsl_type) in the order the
# matching GLSL function declares them. ``dim`` is 2 or 3: the spatial
# dimension of the input point. These mirror sdf_shapes.py one-for-one.


@dataclass(frozen=True)
class PrimSpec:
    dim: int
    args: tuple[tuple[str, str], ...]  # ((arg_name, glsl_type), ...)


# fmt: off
_GRID_TEX_WIDTH = 4096

_PRIM_SPECS: dict[str, PrimSpec] = {
    # 3-D exact / bound primitives
    "sphere":           PrimSpec(3, (("r", "float"),)),
    "box":              PrimSpec(3, (("b", "vec3"),)),
    "round_box":        PrimSpec(3, (("b", "vec3"), ("r", "float"))),
    "box_frame":        PrimSpec(3, (("b", "vec3"), ("e", "float"))),
    "torus":            PrimSpec(3, (("t", "vec2"),)),
    "capped_torus":     PrimSpec(3, (("sc", "vec2"), ("ra", "float"), ("rb", "float"))),
    "helix":            PrimSpec(3, (("major_r", "float"), ("pitch", "float"),
                                     ("r", "float"), ("n_turns", "float"),
                                     ("phase", "float"), ("handedness", "float"))),
    "screw_thread":     PrimSpec(3, (("r_root", "float"), ("depth", "float"),
                                     ("pitch", "float"), ("width", "float"),
                                     ("n_turns", "float"), ("phase", "float"),
                                     ("handedness", "float"), ("flank_deg", "float"))),
    "link":             PrimSpec(3, (("le", "float"), ("r1", "float"), ("r2", "float"))),
    "cone":             PrimSpec(3, (("c", "vec2"), ("h", "float"))),
    "plane":            PrimSpec(3, (("n", "vec3"), ("h", "float"))),
    # Variable-length sample block — dedicated emit path (polygon_2d model).
    "raster_field":     PrimSpec(3, ()),
    "hex_prism":        PrimSpec(3, (("h", "vec2"),)),
    "tri_prism":        PrimSpec(3, (("h", "vec2"),)),
    "capsule":          PrimSpec(3, (("a", "vec3"), ("b", "vec3"), ("r", "float"))),
    "capped_cylinder":  PrimSpec(3, (("h", "float"), ("r", "float"))),
    "rounded_cylinder": PrimSpec(3, (("ra", "float"), ("rb", "float"), ("h", "float"))),
    "capped_cone":      PrimSpec(3, (("h", "float"), ("r1", "float"), ("r2", "float"))),
    "solid_angle":      PrimSpec(3, (("c", "vec2"), ("ra", "float"))),
    "cut_sphere":       PrimSpec(3, (("r", "float"), ("h", "float"))),
    "ellipsoid":        PrimSpec(3, (("r", "vec3"),)),
    "octahedron":       PrimSpec(3, (("s", "float"),)),
    "pyramid":          PrimSpec(3, (("h", "float"),)),
    # TPMS lattices (n_periods is vec3)
    "gyroid":           PrimSpec(3, (("period", "float"), ("min_thickness", "float"), ("n_periods", "vec3"))),  # noqa: E501
    "schwarz_p":        PrimSpec(3, (("period", "float"), ("min_thickness", "float"), ("n_periods", "vec3"))),  # noqa: E501
    "schwarz_d":        PrimSpec(3, (("period", "float"), ("min_thickness", "float"), ("n_periods", "vec3"))),  # noqa: E501
    "neovius":          PrimSpec(3, (("period", "float"), ("min_thickness", "float"), ("n_periods", "vec3"))),  # noqa: E501
    "lidinoid":         PrimSpec(3, (("period", "float"), ("min_thickness", "float"), ("n_periods", "vec3"))),  # noqa: E501
    # Compliant mechanisms (NB: bellows/serpentine n_periods is scalar here)
    "notch_hinge":      PrimSpec(3, (("width", "float"), ("depth", "float"), ("notch_radius", "float"))),  # noqa: E501
    "leaf_spring":      PrimSpec(3, (("length", "float"), ("width", "float"), ("thickness", "float"))),  # noqa: E501
    "bellows":          PrimSpec(3, (("outer_r", "float"), ("inner_r", "float"), ("period", "float"), ("n_periods", "float"))),  # noqa: E501
    "serpentine":       PrimSpec(3, (
        ("amplitude", "float"), ("wavelength", "float"),
        ("beam_width", "float"), ("beam_height", "float"),
        ("n_periods", "float"),
    )),
    "annular_sector":   PrimSpec(3, (("inner_r", "float"), ("outer_r", "float"),
                                     ("half_angle", "float"), ("height", "float"))),
    # 2-D primitives
    "circle_2d":         PrimSpec(2, (("r", "float"),)),
    "box_2d":            PrimSpec(2, (("b", "vec2"),)),
    "rounded_box_2d":    PrimSpec(2, (("b", "vec2"), ("r", "float"))),
    "segment_2d":        PrimSpec(2, (("a", "vec2"), ("b", "vec2"))),
    "trapezoid_2d":      PrimSpec(2, (("r1", "float"), ("r2", "float"), ("he", "float"))),
    "uneven_capsule_2d": PrimSpec(2, (("r1", "float"), ("r2", "float"), ("h", "float"))),
    # polygon_2d takes a variable-length `vertices` list rather than fixed
    # scalar/vector args, so it is emitted by a dedicated code path in
    # _emit_primitive (see _emit_polygon_2d) rather than the generic arg loop.
    "polygon_2d":        PrimSpec(2, ()),
}
# fmt: on


# Maps DSL primitive kind to its lib.glsl function name.
def _lib_fn(kind: str) -> str:
    # Conveniently every primitive in lib.glsl is exposed as `sdf_<kind>`,
    # but `leaf_spring` collides with GLSL's `length` keyword via its `length`
    # arg name, handled in the emitter, not here.
    return f"sdf_{kind}"


# CSG ops
_HARD_BINARY_OPS = {"union", "subtract", "intersect"}
_SMOOTH_BINARY_OPS = {"smooth_union", "smooth_subtract", "smooth_intersect"}
_HARD_TO_SMOOTH = {
    "union": "smooth_union",
    "subtract": "smooth_subtract",
    "intersect": "smooth_intersect",
}

# Transforms: keyed by name, value is (kwarg_name, glsl_type, glsl_fn_template).
# 2-D variants for nodes whose subtree is 2-D.

# Field primitives: same arg-spec idea as 3-D primitives.
# fmt: off
_FIELD_SPECS: dict[str, tuple[tuple[str, str], ...]] = {
    "sin_xyz": (("freq", "vec3"), ("amplitude", "float"), ("phase", "vec3")),
    "radial":  (("freq", "float"), ("amplitude", "float"), ("phase", "float")),
    "angular": (("freq", "float"), ("amplitude", "float"), ("phase", "float")),
}

_FIELD_DEFAULTS: dict[str, dict[str, Any]] = {
    # Match the Python signatures' defaults in sdf_shapes.field_*.
    "sin_xyz": {"amplitude": 1.0, "phase": (0.0, 0.0, 0.0)},
    "radial":  {"amplitude": 1.0, "phase": 0.0},
    "angular": {"amplitude": 1.0, "phase": 0.0},
}
# fmt: on


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniformDecl:
    """A shader input backed by a design parameter or authored motion DOF.

    Motion inputs use ``source_param='kinematics.<name>'`` and authored units.
    Their indexed ``u_dof_*`` identifiers cannot collide with design uniforms.
    """

    name: str  # GLSL identifier, e.g. "u_p_outer_radius"
    glsl_type: str  # always "float" since each Param is scalar
    initial: float
    bounds: tuple[float, float] | None
    unit: str
    source_param: str  # design name or namespaced motion-control key


@dataclass
class GLSLEmission:
    """All artifacts a host (Blender, web viewer, parity test, ...) needs.

    Full-part motion emissions append live controls with ``ui.role='pose'``.
    Their ``param`` keys are ``kinematics.<name>``; ``dof`` preserves the authored
    name. Values, units and exploration bounds remain in authored units. Body
    components carry conservative boxes over their declared input ranges. A
    component box is None when only an authored scene envelope is available.

    ``controls`` is the per-param control manifest: every ``Part.params`` entry
    classified by what an edit to it costs, which is what decides the widget a
    viewer builds and the path the edit takes:

    - ``"live"``: the param bound a ``u_p_*`` uniform (a ``$ref`` in a
                  supported slot); scrub it with zero recompilation.
    - ``"derived"``: the param carries a relation (``Param.expr``). It has no
                  uniform of its own: the emitter expanded the relation
                  inline over its base params' uniforms, so the value
                  follows them live and cannot go stale. The entry carries
                  ``expr`` and ``sources`` (immediate dependencies) so a
                  viewer can show the number read-only and recompute it
                  without re-emitting anything.
    - ``"topology"``: authored ``ui.role == "topology"``. The param changes
                  how many nodes the authoring script emits, which no emitter
                  can infer from the stamped-out tree, so the viewer must
                  re-run that script and re-emit.
    - ``"re-emit"``: everything else. The param exists but never bound a
                  uniform (a baked slot such as ``polygon_2d`` vertices, or
                  per-site variation); re-emitting this part is enough.

    The class is COMPUTED here, not authored: only the ``topology`` refinement
    comes from ``Param.ui``, because it is genuinely uninferrable. A viewer that
    guessed instead would either recompile on every scrub or scrub a value the
    shader cannot see.

    ``poly_table`` and ``poly_max_n`` are the polygon PAYLOAD, reported rather
    than packed: this emitter bakes every outline into the source as a ``vec2``
    constant, so ``poly_table`` is empty and ``poly_max_n`` is the largest
    vertex count it sized those constants to. A host needs that number because
    it also sizes ``sdf_polygon_2d``'s array parameter. It is emitted as
    ``#define SDM_POLY_MAX_N`` at the head of ``lib_source``, and a host
    assembling its own library out of per-leaf emissions has to restore the
    define per leaf. Handing it over as a field is what stops the host parsing
    it back out of the source it was just given.
    """

    scene_source: str
    lib_source: str
    uniforms: list[UniformDecl]
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]]
    smooth_csg: bool
    smooth_k: float
    entry_point: str = "sdf_scene"
    controls: list[dict[str, Any]] = field(default_factory=list)
    # Component table for segmentation: [{"id", "machine", "label", "bbox"}, ...].
    # `id` is what `sdf_scene_comp` returns; `machine` is the emitted GLSL
    # function that IS that component's field, which is the anchor a host needs
    # to rewrite a call site (masking one component out, say). Populated for
    # authored rigid bodies or a segmentable hard union; empty means the
    # scene is ONE component, and a host must not read that as "unknown".
    components: list[dict[str, Any]] = field(default_factory=list)
    # Flat [x0, y0, x1, y1, ...] of every outline vertex the SCENE FETCHES, in
    # fetch order. Empty means every outline is baked into the source instead,
    # which is this emitter's only mode today; NOT that the scene has no
    # polygons. Read `poly_max_n` for that.
    poly_table: list[float] = field(default_factory=list)
    # Row width the emitted index arithmetic assumes, 0 when there is no
    # table. REPORTED, not dictated: `sdm_poly_fetch` needs a width to turn an
    # index into a texel, so the emitted source has to name one, and a host
    # cannot pack the texture without knowing which. lib.glsl guards the macro
    # with `#ifndef`, so a host that wants a different width defines its own
    # and this value stops applying to it.
    poly_tex_width: int = 0
    # Largest polygon_2d vertex count in this emission; 0 = no polygons.
    poly_max_n: int = 0
    # Flat RGBA texels of every sweep frame the SCENE FETCHES, in fetch order
    # (lib.glsl's SDM_SWEEP_TABLE block gives the layout). Empty means every
    # sweep's frames are baked into the source as vec3 arrays instead, NOT
    # that the scene has no sweeps. Read `sweep_max_s` for that.
    sweep_table: list[float] = field(default_factory=list)
    # Row width the emitted `sdm_sweep_fetch` index arithmetic assumes, 0
    # when there is no table. Reported, not dictated, like `poly_tex_width`.
    sweep_tex_width: int = 0
    # Largest sampled segment count of any sweep in this emission, inline or
    # tabled; 0 = no sweeps. It is what sizes the biggest per-node frame
    # arrays in the source, so a host budgeting shader size wants it.
    sweep_max_s: int = 0
    # raster_field raw sample payload (x-fastest f32). Empty = no grids.
    # Host packs into an R32F texture ``u_sdm_grid`` at ``grid_tex_width``.
    grid_table: list[float] = field(default_factory=list)
    grid_tex_width: int = 0
    # Outline LOD this emission was resampled to; None = full resolution.
    # REPORTED so a host can tell a decimated emission from an authored one —
    # the march cost of sdf_polygon_2d is linear in vertex count per step, so
    # this is the number a viewer perf report needs next to frame time.
    poly_lod: int | None = None


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


class _GLSLEmitter:
    """Recursive walker; one instance per emit_glsl call."""

    def __init__(
        self,
        part: Part,
        *,
        smooth_csg: bool,
        smooth_k: float,
        poly_lod: int | None = None,
    ) -> None:
        self.part = part
        self.smooth_csg = smooth_csg
        self.smooth_k = smooth_k
        # Viewer-fidelity lever: literal outlines above this vertex count are
        # curvature-resampled down to it before emission. None = never touch.
        self.poly_lod = poly_lod
        self.uniforms: dict[str, UniformDecl] = {}  # keyed by source_param
        self.functions: list[str] = []
        self._next_id = 0
        # Largest polygon_2d vertex count seen; sizes the shared sdf_polygon_2d
        # array parameter (SDM_POLY_MAX_N). 0 means no polygons in the scene.
        self.poly_max_n = 0
        # Flat (x, y) pairs, one pair per texel. Empty unless this tree
        # crossed `_POLY_TABLE_THRESHOLD`, which `emit_glsl` decides and
        # sets `poly_table_on` for.
        self.poly_table: list[float] = []
        self.poly_table_on = False
        # Sweep frames, RGBA texels flat. Same threshold and the same
        # decision point as `poly_table_on`: `emit_glsl` sets it from the
        # whole tree before the walk.
        self.sweep_table: list[float] = []
        self.sweep_table_on = False
        self.sweep_max_s = 0
        self._node_memo: dict[tuple[int, bool, float, str], str] = {}

        self.grid_arrays: list = []
        self.grid_len = 0
        self.grid_offsets: dict[bytes, int] = {}

    # -- ID + helpers ----------------------------------------------------

    def _fn_name(self, dim: int) -> str:
        n = self._next_id
        self._next_id += 1
        return f"sdf_n{n}_d{dim}"

    def _field_name(self) -> str:
        n = self._next_id
        self._next_id += 1
        return f"field_n{n}"

    def _emit_function(self, signature: str, body_lines: list[str]) -> None:
        body = "\n".join(f"    {ln}" for ln in body_lines)
        self.functions.append(f"{signature} {{\n{body}\n}}")

    # -- Uniform / literal resolution -----------------------------------

    def _add_uniform(self, param_name: str) -> str:
        if param_name in self.uniforms:
            return self.uniforms[param_name].name
        if param_name not in self.part.params:
            raise KeyError(
                f"$ref {param_name!r} does not match any Param in part "
                f"{self.part.name!r} (params={list(self.part.params)})"
            )
        p = self.part.params[param_name]
        if p.expr is not None:
            # A derived param must never get a uniform of its own: the host has
            # no way to keep it in step with its base params, so the first
            # scrub of a base param would leave the geometry inconsistent with
            # the relation that defines it. Callers route through
            # _emit_param_ref, which expands the relation instead.
            raise ValueError(
                f"Param {param_name!r} is defined by a relation (expr) and "
                f"cannot bind a uniform. Emit it via _emit_param_ref, which "
                f"expands it to arithmetic over its base params' uniforms."
            )
        uname = f"u_p_{param_name}"
        self.uniforms[param_name] = UniformDecl(
            name=uname,
            glsl_type="float",
            initial=float(p.numeric_value()),
            bounds=(float(p.bounds[0]), float(p.bounds[1])) if p.bounds is not None else None,
            unit=p.unit,
            source_param=param_name,
        )
        return uname

    def _emit_param_ref(self, param_name: str, _stack: tuple[str, ...] = ()) -> str:
        """A parameter reference as a float-typed GLSL expression.

        A plain param becomes its uniform. A param carrying a relation becomes
        that relation, EXPANDED — recursively, so a chain
        ``a -> b -> free`` lands as arithmetic over ``free``'s uniform.
        Expansion rather than a per-relation uniform is what makes a derived
        dimension follow its base params live: there is no second value that
        could be stale, because there is no second value.
        """
        if param_name not in self.part.params:
            raise KeyError(
                f"$ref {param_name!r} does not match any Param in part "
                f"{self.part.name!r} (params={list(self.part.params)})"
            )
        p = self.part.params[param_name]
        if p.expr is None:
            return self._add_uniform(param_name)
        if param_name in _stack:
            # Unreachable on any loaded part — validate refuses a cyclic
            # relation DAG, and ParamBinding re-checks for in-memory parts.
            raise ValueError(
                "Cyclic parameter relation reached the GLSL emitter: "
                + " -> ".join(_stack + (param_name,))
            )
        return "(" + self._emit_expr(p.expr, _stack + (param_name,)) + ")"

    def _emit_expr(self, node: Any, _stack: tuple[str, ...] = ()) -> str:
        """Compile one expression-tree node to a float-valued GLSL expression.

        Mirrors ``dsl.expr._eval_node``, node type for node type and operator
        for operator, so the shader computes what the JAX evaluator computes.
        The two operator tables are checked against each other by a test rather
        than by inspection: an op added to one side and not the other is the
        drift that shows up as a part rendering differently from the thing it
        was optimised as.

        Every result is a ``float``. The DSL has no vector arithmetic, and a
        slot wanting a vector builds one from per-component expressions.

        Args:
            node: An expression tree, a bare number, or a ``{"$ref": name}``
                leaf (which :mod:`dsl.expr` also tolerates inside a tree).
            _stack: Derived params already being expanded; cycle guard for
                relation expansion via :meth:`_emit_param_ref`.

        Returns:
            str: A parenthesised GLSL expression.

        Raises:
            NotImplementedError: For ``metric``, which has no meaning per pixel.
            ValueError: For an unknown node type or operator.
        """
        if isinstance(node, bool):
            raise TypeError("Boolean expression leaf unsupported; use 0.0 or 1.0.")
        if isinstance(node, (int, float)):
            return _float_literal(node)
        if not isinstance(node, dict):
            raise ValueError(f"Not an expression node: {node!r}")

        # A `$ref` leaf mixed into a tree, which dsl.expr accepts too.
        if "$ref" in node and "type" not in node:
            return self._emit_param_ref(node["$ref"], _stack)

        t = node.get("type")

        if t == "num":
            return _float_literal(node["value"])

        if t == "param":
            return self._emit_param_ref(node["name"], _stack)

        if t == "metric":
            # Not a shortcoming to be filled in later. A metric integrates a
            # sampled grid of the whole compiled SDF -- volume, mass, relative
            # density -- and a fragment shader is evaluating one point with no
            # access to any of that. `dsl.expr.eval_expr_pure` refuses it in an
            # SDF slot for the same reason.
            raise NotImplementedError(
                f"metric {node.get('name')!r} cannot be compiled into a shader: "
                "it integrates a sampled grid of the whole part, which a "
                "per-point evaluation does not have. Metrics belong in "
                "objectives and constraints, not in an SDF slot."
            )

        if t == "unop":
            op = node["op"]
            if op not in _EXPR_UNARY:
                raise ValueError(f"Unknown unary op {op!r}")
            return _EXPR_UNARY[op](self._emit_expr(node["child"], _stack))

        if t == "binop":
            op = node["op"]
            if op not in _EXPR_BINARY:
                raise ValueError(f"Unknown binary op {op!r}")
            return _EXPR_BINARY[op](
                self._emit_expr(node["lhs"], _stack),
                self._emit_expr(node["rhs"], _stack),
            )

        if t == "reduce":
            op = node["op"]
            if op not in _EXPR_REDUCE:
                raise ValueError(f"Unknown reduce op {op!r}")
            parts = [self._emit_expr(c, _stack) for c in node["children"]]
            if not parts:
                raise ValueError(f"reduce {op!r} has no children")
            return _EXPR_REDUCE[op](parts)

        raise ValueError(f"Unknown expression node type {t!r}")

    def _emit_value(self, value: Any, glsl_type: str) -> str:
        """Render a DSL leaf as a GLSL expression of the given type.

        Rules
        -----
        * Scalars (int/float) → GLSL float literal (e.g. ``1.5``). When
          ``glsl_type`` is a vector, the scalar broadcasts via
          ``vecN(scalar)``.
        * ``$ref`` → uniform name (registered on first use), or, when the named
          param carries a relation, the expanded arithmetic of that relation.
        * Lists/tuples → ``vecN(...)`` constructor over elements (each may
          itself be scalar or ``$ref``).
        * 3x3 nested list → ``mat3(...)``. GLSL mat3 is column-major, so we
          transpose to keep ``R * p`` (GLSL) equal to ``p @ R.T`` (numpy).
        * Expression-tree leaves (``{"type": ..., ...}`` other than
          ``$ref``) → compiled by :meth:`_emit_expr`, which mirrors
          ``dsl.expr``. A scalar expression broadcasts into a vector slot the
          same way a scalar literal does.
        """
        if isinstance(value, dict):
            if "$ref" in value:
                expr = self._emit_param_ref(value["$ref"])
                if glsl_type == "float":
                    return expr
                # A scalar uniform being passed where a vec is expected is
                # almost certainly a DSL bug, but we don't have signal here.
                # Broadcast and surface it loudly via the type string.
                return f"{glsl_type}({expr})"
            if "type" in value:
                expr = self._emit_expr(value)
                return expr if glsl_type == "float" else f"{glsl_type}({expr})"
            raise ValueError(f"Unrecognised DSL leaf dict: {value!r}")

        if isinstance(value, bool):
            raise TypeError("Boolean DSL leaf unsupported; use 0.0 or 1.0 explicitly.")

        if isinstance(value, (int, float)):
            return (
                _float_literal(value)
                if glsl_type == "float"
                else f"{glsl_type}({_float_literal(value)})"
            )

        if isinstance(value, (list, tuple)):
            return self._emit_vector_or_matrix(value, glsl_type)

        raise TypeError(f"Cannot emit GLSL for DSL leaf of type {type(value).__name__}: {value!r}")

    def _emit_vector_or_matrix(self, value: Any, glsl_type: str) -> str:
        if glsl_type == "mat3":
            # Expect a 3-row row-major list of lists. Transpose for GLSL.
            rows = list(value)
            if len(rows) != 3 or any(len(r) != 3 for r in rows):
                raise ValueError(f"Expected a 3x3 nested list for mat3, got {value!r}")
            # GLSL mat3 columns:
            cols = [[rows[0][j], rows[1][j], rows[2][j]] for j in range(3)]
            col_exprs = [
                f"vec3({self._emit_value(c[0], 'float')}, "
                f"{self._emit_value(c[1], 'float')}, "
                f"{self._emit_value(c[2], 'float')})"
                for c in cols
            ]
            return f"mat3({', '.join(col_exprs)})"

        # vec2 / vec3 / vec4 / ...
        if not glsl_type.startswith("vec"):
            raise ValueError(f"Cannot emit list literal as type {glsl_type!r}")
        expected = int(glsl_type[3:])
        items = list(value)
        if len(items) != expected:
            raise ValueError(
                f"Expected {expected} components for {glsl_type}, got {len(items)}: {value!r}"
            )
        parts = [self._emit_value(v, "float") for v in items]
        return f"{glsl_type}({', '.join(parts)})"

    # -- Node dispatch ---------------------------------------------------

    def emit_node(self, node: SDFTree, *, dim: int = 3) -> str:
        """Share functions only within this emission and dimensional context."""
        try:
            structure = json.dumps(node, sort_keys=True)
        except TypeError:
            # Memoization must not reject nodes the uncached emitter accepts.
            return self._emit_node_uncached(node, dim=dim)
        key = (dim, self.smooth_csg, self.smooth_k, structure)
        if key not in self._node_memo:
            self._node_memo[key] = self._emit_node_uncached(node, dim=dim)
        return self._node_memo[key]

    def _emit_node_uncached(self, node: SDFTree, *, dim: int = 3) -> str:
        """Emit ``node`` as a function returning ``float`` taking a ``vec{dim}``
        argument. Returns the name of the emitted function.
        """
        if not isinstance(node, dict) or "type" not in node:
            raise ValueError(f"Not an SDF node: {node!r}")

        kind = node["type"]
        if kind == "primitive":
            return self._emit_primitive(node, dim)
        if kind == "op":
            return self._emit_op(node, dim)
        if kind == "transform":
            return self._emit_transform(node, dim)
        if kind == "modifier":
            return self._emit_modifier(node, dim)
        if kind == "deform":
            return self._emit_deform(node, dim)
        if kind == "loft":
            if dim != 3:
                raise ValueError(f"loft produces a 3-D function (got dim={dim})")
            return self._emit_loft(node)
        if kind == "sweep":
            if dim != 3:
                raise ValueError(f"sweep produces a 3-D function (got dim={dim})")
            return self._emit_sweep(node)
        if kind == "2d_to_3d":
            return self._emit_2d_to_3d(node, dim)
        raise ValueError(f"Unknown SDF node type {kind!r}")

    # -- Primitive -------------------------------------------------------

    def _emit_primitive(self, node: dict[str, Any], dim: int) -> str:
        kind = node["kind"]
        # Curves have no PrimSpec and no lib function: they ARE the polygon SDF
        # of their sampled outline, so they lower before the table is consulted.
        if kind in _CURVE_KINDS:
            if dim != 2:
                raise ValueError(f"Primitive {kind!r} is 2-D but used in a {dim}-D context.")
            return self._emit_polygon_2d(self._curve_to_polygon(node))
        if kind not in _PRIM_SPECS:
            raise ValueError(f"Unknown primitive {kind!r}")
        spec = _PRIM_SPECS[kind]
        if spec.dim != dim:
            raise ValueError(f"Primitive {kind!r} is {spec.dim}-D but used in a {dim}-D context.")

        if kind == "raster_field":
            return self._emit_raster_field(node)
        if kind == "polygon_2d":
            return self._emit_polygon_2d(node)

        raw_kwargs = node.get("params") or {}
        arg_exprs: list[str] = []
        for arg_name, arg_type in spec.args:
            if arg_name not in raw_kwargs:
                raise KeyError(
                    f"Primitive {kind!r} missing required arg {arg_name!r} "
                    f"(provided: {sorted(raw_kwargs)})"
                )
            arg_exprs.append(self._emit_value(raw_kwargs[arg_name], arg_type))

        fn = self._fn_name(dim)
        call = f"{_lib_fn(kind)}(p, {', '.join(arg_exprs)})"
        self._emit_function(f"float {fn}(vec{dim} p)", [f"return {call};"])
        return fn

    def _emit_raster_field(self, node: dict[str, Any]) -> str:
        """Emit a 3-D ``raster_field`` (DR-0003): samples leave the source.

        Append x-fastest samples to the shared grid table; the wrapper carries
        only the base offset plus geometry. Host packs ``grid_table`` into
        ``u_sdm_grid`` — core does not own texture layout beyond row width.
        """
        from software_defined_matter.sdf.raster import (
            decode_raster_values,
            normalize_spacing,
            raster_dims,
            raster_origin,
            require_literal_raster_params,
        )

        params = node.get("params") or {}
        try:
            require_literal_raster_params(params)
        except ValueError as exc:
            raise NotImplementedError(str(exc)) from exc
        values = decode_raster_values(params)  # (nz, ny, nx) float32
        nx, ny, nz = raster_dims(params)
        origin = raster_origin(params)
        spacing = normalize_spacing(params)

        flat = values.reshape(-1)
        key = flat.astype("<f4").tobytes()
        base = self.grid_offsets.get(key)
        if base is None:
            base = self.grid_len
            self.grid_offsets[key] = base
            self.grid_arrays.append(flat)
            self.grid_len += int(flat.size)

        inv_h = tuple(1.0 / h for h in spacing)

        def _v3(v: tuple[float, ...]) -> str:
            return f"vec3({float(v[0])}, {float(v[1])}, {float(v[2])})"

        call = f"sdf_raster_field(p, {base}, ivec3({nx}, {ny}, {nz}), {_v3(origin)}, {_v3(inv_h)})"
        fn = self._fn_name(3)
        self._emit_function(f"float {fn}(vec3 p)", [f"return {call};"])
        return fn

    def _emit_polygon_2d(self, node: dict[str, Any]) -> str:
        """Emit a 2-D ``polygon_2d`` node.

        The ``vertices`` param is a fixed-length ``[[x, y], ...]`` list. Each
        coordinate is resolved via :meth:`_emit_value`: numeric coords become
        GLSL float literals; ``$ref`` coords become live ``u_p_<name>``
        uniforms (so a host can reshape the polygon at draw time). The
        resolved expressions initialize a per-node ``vec2 V[SDM_POLY_MAX_N]``
        and we return ``sdf_polygon_2d(p, V, N)`` (see lib.glsl).

        Expression-tree leaves (arithmetic over params, etc.) are still out of
        scope and raise ``NotImplementedError``; see #54.
        """
        raw_kwargs = node.get("params") or {}
        if "vertices" not in raw_kwargs:
            raise KeyError("Primitive 'polygon_2d' missing required arg 'vertices'")
        verts = raw_kwargs["vertices"]
        if not isinstance(verts, (list, tuple)) or len(verts) < 3:
            raise ValueError(
                f"polygon_2d 'vertices' must be a list of >=3 [x, y] pairs, got {verts!r}"
            )
        verts = self._resample_outline(verts)

        vert_exprs = [self._emit_value(list(v), "vec2") for v in verts]
        n = len(vert_exprs)

        # GLSL array declarations are `vec2 V[N] = vec2[N](e0, ...)`, and the
        # initializer count must match the declared size exactly. The array is
        # sized to SDM_POLY_MAX_N (largest polygon in the scene, pre-scanned in
        # emit_glsl) so its type matches sdf_polygon_2d's parameter; shorter
        # polygons pad the tail with vec2(0.0). sdf_polygon_2d loops only over
        # the real count `n`, so padding slots are never read.
        # Above the threshold the vertices are DATA, not source. A `$ref`
        # vertex keeps the inline form: it resolves to a uniform the host can
        # scrub, and a table texel is uploaded once.
        if self.poly_table_on and all(_is_literal_vertex(v) for v in verts):
            hdr = self._table_polygon(list(verts))
            fn = self._fn_name(2)
            self._emit_function(f"float {fn}(vec2 p)", [f"return sdm_polygon_2d_tab(p, {hdr});"])
            return fn

        pad = self.poly_max_n - n
        if pad < 0:  # pragma: no cover - guaranteed non-negative by pre-scan
            raise AssertionError("poly_max_n underestimated the polygon vertex count")
        init = ", ".join(vert_exprs + ["vec2(0.0)"] * pad)
        fn = self._fn_name(2)
        body = [
            f"vec2 V[SDM_POLY_MAX_N] = vec2[SDM_POLY_MAX_N]({init});",
            f"return sdf_polygon_2d(p, V, {n});",
        ]
        self._emit_function(f"float {fn}(vec2 p)", body)
        return fn

    def _outline_weights(self, verts: list[Any]) -> np.ndarray:
        """Per-vertex importance for outline resampling.

        Local turning angle plus a uniform floor: half the importance budget is
        spread evenly so long flat runs still get sampled and the selection
        cannot collapse onto a single sharp corner.
        """
        v = np.asarray([[float(x), float(y)] for x, y in verts])
        n = len(v)
        e_in = v - np.roll(v, 1, axis=0)  # edge arriving at vertex i
        e_out = np.roll(v, -1, axis=0) - v  # edge leaving vertex i
        li = np.linalg.norm(e_in, axis=1)
        lo = np.linalg.norm(e_out, axis=1)
        cross = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
        dot = (e_in * e_out).sum(axis=1)
        turn = np.abs(np.arctan2(cross, dot))  # exterior angle at i
        # Degenerate (repeated) vertices contribute no angle.
        turn = np.where((li > 0.0) & (lo > 0.0), turn, 0.0)
        total = turn.sum()
        return (turn / total if total > 0.0 else np.zeros(n)) + 1.0 / n

    @staticmethod
    def _keep_indices(w: np.ndarray, lod: int) -> list[int]:
        """Exactly ``lod`` vertex indices, the quantiles of cumulative ``w``.

        Winding order is preserved (indices come back sorted). Quantile
        collisions can select the same vertex twice; top up with the
        highest-importance unselected vertices so the count is exact.
        """
        n = len(w)
        cum = np.concatenate([[0.0], np.cumsum(w)])
        targets = np.arange(lod) * (cum[-1] / lod)
        idx = np.searchsorted(cum, targets, side="right") - 1
        keep = set(np.clip(idx, 0, n - 1).tolist())
        if len(keep) < lod:
            for i in np.argsort(-w):
                keep.add(int(i))
                if len(keep) == lod:
                    break
        return sorted(keep)

    def _resample_outline(self, verts: Iterable[Any]) -> list[Any]:
        """Downsample a literal outline to ``poly_lod`` vertices when above it.

        Curvature-weighted: high-curvature features (an airfoil's leading
        edge) keep their density; flats give up points first. Uniform stride
        was tried first and visibly faceted the leading edge at lod 32. This
        is a VIEWER-fidelity lever: the march cost of ``sdf_polygon_2d`` is
        linear in n per evaluation per step (measured 18x frame time on a
        126-vertex airfoil at lod 32); meshing/JAX paths never see it. An
        outline with any ``$ref`` vertex is left alone — those resolve to
        live uniforms a host scrubs, and dropping one would break its panel.
        """
        vlist = list(verts)
        lod = self.poly_lod
        if not lod or len(vlist) <= lod or not all(_is_literal_vertex(v) for v in vlist):
            return vlist
        w = self._outline_weights(vlist)
        return [vlist[i] for i in self._keep_indices(w, lod)]

    def _resample_sections(self, outlines: list[list[Any]]) -> list[list[Any]]:
        """Downsample loft sections with ONE shared index set.

        A shape-loft interpolates vertex j of section k against vertex j of
        section k+1, so per-section curvature selection would silently break
        that correspondence (each section keeping different indices). The
        weights are summed across sections and one index set is applied to
        all, so a feature sharp in ANY section keeps its vertex in EVERY
        section. Only applies when every section is literal and all counts
        already match (the caller validates counts).
        """
        lod = self.poly_lod
        if not lod or not outlines:
            return outlines
        n = len(outlines[0])
        if n <= lod or any(len(o) != n for o in outlines):
            return outlines
        if not all(_is_literal_vertex(v) for o in outlines for v in o):
            return outlines
        w = np.sum([self._outline_weights(o) for o in outlines], axis=0)
        keep = self._keep_indices(w, lod)
        return [[o[i] for i in keep] for o in outlines]

    def _table_outline(self, verts: list[Any]) -> int:
        """Append an outline's vertices to the table, headerless. Returns its
        first texel index.

        For a caller that already knows the count as a compile-time constant
        and indexes the vertices itself, which is what a shape-loft does: it
        needs section `k`'s vertex `j`, not a self-describing polygon.

        Literals only, because a table texel is data the host uploads once. A
        `$ref` vertex is refused earlier, by whoever built this list.
        """
        base = len(self.poly_table) // 2
        for v in verts:
            self.poly_table.extend((float(v[0]), float(v[1])))
        return base

    def _table_polygon(self, verts: list[Any]) -> int:
        """Append an outline WITH its header texel. Returns the header index.

        `sdm_polygon_2d_tab` reads the count out of the header at run time
        rather than taking it as an argument; lib.glsl says why.
        """
        hdr = len(self.poly_table) // 2
        self.poly_table.extend((float(len(verts)), 0.0))
        for v in verts:
            self.poly_table.extend((float(v[0]), float(v[1])))
        return hdr

    def _curve_to_polygon(self, node: dict[str, Any]) -> dict[str, Any]:
        """Lower a ``bezier_2d`` / ``bspline_2d`` node to a sampled ``polygon_2d``.

        This is EXACT, not an approximation of the curve: `sdf_shapes` defines
        both primitives as the polygon SDF of their own sampled outline, so
        sampling here reproduces the field rather than standing in for it. It
        happens once at emission because the basis is fixed and the control
        points are literals.

        Args:
            node: The curve primitive node.

        Returns:
            dict: An equivalent ``polygon_2d`` node.

        Raises:
            NotImplementedError: A ``$ref`` or expression control point. Those
                would have to become uniforms, and the outline is baked into
                the shader text, so the curve would silently keep its emit-time
                shape while the param moved.
        """
        kind = node["kind"]
        raw = (node.get("params") or {}).get("control_points")
        numeric = isinstance(raw, (list, tuple)) and all(
            isinstance(pt, (list, tuple))
            and len(pt) == 2
            and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in pt)
            for pt in raw
        )
        if not numeric:
            raise NotImplementedError(
                f"{kind}: GLSL emission needs numeric literal control_points; a $ref "
                "or expression control point would be baked at its emit-time value "
                "and stop tracking the param"
            )
        verts = _curve_outline(kind, raw, _curve_samples())
        return {"type": "primitive", "kind": "polygon_2d", "params": {"vertices": verts}}

    # -- Loft ------------------------------------------------------------

    def _emit_loft(self, node: dict[str, Any]) -> str:
        """Emit a ``loft`` node, mirroring ``sdf_ops.loft`` / ``loft_shape``.

        Two modes, and they are different fields rather than two spellings of
        one. ``interp='field'`` evaluates every section's own 2-D function at
        ``p.xy`` and interpolates the DISTANCES. ``interp='shape'`` interpolates
        the section OUTLINES and takes the exact polygon SDF of the result,
        which is what removes the bulge field interpolation leaves at a swept
        convex edge.

        Station z's may be ``$ref``s or expressions and become uniforms, so a
        span is scrubbable. Outlines are baked literals, like every other
        polygon in this emitter.

        Args:
            node: The ``loft`` node.

        Returns:
            str: The emitted function's name.

        Raises:
            ValueError: Fewer than two sections, a ``z`` list of the wrong
                length, an unknown ``interp``, or shape sections that do not
                correspond.
            NotImplementedError: Curve sections under ``interp='shape'``. See
                the comment at that branch.
        """
        children = node.get("children") or []
        params = node.get("params") or {}
        zs = params.get("z")
        if len(children) < 2:
            raise ValueError("loft needs at least 2 cross-section children")
        if not isinstance(zs, (list, tuple)) or len(zs) != len(children):
            raise ValueError("loft requires params['z']: one axial station per section")
        interp = str(params.get("interp", "field"))
        if interp not in ("field", "shape"):
            raise ValueError(f"loft interp must be 'field' or 'shape', got {interp!r}")
        # Structural, not param-resolved, exactly as _compile_loft reads it.
        smooth = bool(params.get("smooth", False))
        n_sec = len(children)

        # `idx` is the segment containing p.z. ops.loft spells it
        # `clip(searchsorted(zs, z, 'right') - 1, 0, n - 2)`; the loop below is
        # the same index, written for a GLSL that has no searchsorted.
        zs_init = ", ".join(self._emit_value(z, "float") for z in zs)
        span = [
            f"float ZS[{n_sec}] = float[{n_sec}]({zs_init});",
            "int idx = 0;",
            f"for (int i = 1; i < {n_sec - 1}; ++i) {{ if (p.z >= ZS[i]) idx = i; }}",
            "float za = ZS[idx];",
            "float zb = ZS[idx + 1];",
            "float t = clamp((p.z - za) / (zb - za + 1e-12), 0.0, 1.0);",
        ]
        if smooth:
            span += [
                "int im1 = max(idx - 1, 0);",
                f"int i2 = min(idx + 2, {n_sec - 1});",
                "float hseg = zb - za;",
            ]
        # abs() on the half-span guards a mis-ordered ($ref-driven) span from
        # going negative and emptying the cap everywhere, as ops.loft does.
        cap = [
            f"float zc = (ZS[0] + ZS[{n_sec - 1}]) * 0.5;",
            f"float hh = abs(ZS[{n_sec - 1}] - ZS[0]) * 0.5;",
            "return sdm_loft_cap(d2d, p.z, zc, hh);",
        ]

        if interp == "field":
            body = self._loft_field_body(children, span, smooth, n_sec)
        else:
            body = self._loft_shape_body(children, span, smooth, n_sec)
        fn = self._fn_name(3)
        self._emit_function(f"float {fn}(vec3 p)", body + cap)
        return fn

    def _loft_field_body(
        self, children: list[SDFTree], span: list[str], smooth: bool, n_sec: int
    ) -> list[str]:
        """Interpolate the section DISTANCES. Mirrors ``sdf_ops.loft``."""
        child_fns = [self.emit_node(c, dim=2) for c in children]
        gather = " ".join(f"dv[{i}] = {fn}(p.xy);" for i, fn in enumerate(child_fns))
        body = [f"float dv[{n_sec}];", gather] + span
        if not smooth:
            return body + ["float d2d = mix(dv[idx], dv[idx + 1], t);"]
        return body + [
            "float m0 = sdm_pchip_tan(dv[im1], dv[idx], dv[idx + 1], ZS[idx] - ZS[im1], hseg);",
            "float m1 = sdm_pchip_tan(dv[idx], dv[idx + 1], dv[i2], hseg, ZS[i2] - ZS[idx + 1]);",
            "float t2 = t * t;",
            "float t3 = t2 * t;",
            "float d2d = (2.0 * t3 - 3.0 * t2 + 1.0) * dv[idx]",
            "          + (t3 - 2.0 * t2 + t) * hseg * m0",
            "          + (-2.0 * t3 + 3.0 * t2) * dv[idx + 1]",
            "          + (t3 - t2) * hseg * m1;",
        ]

    def _loft_shape_body(
        self, children: list[SDFTree], span: list[str], smooth: bool, n_sec: int
    ) -> list[str]:
        """Interpolate the section OUTLINES. Mirrors ``sdf_ops.loft_shape``."""
        kinds = {c.get("kind") for c in children}
        if any(c.get("type") != "primitive" for c in children) or not kinds <= set(
            _SHAPE_LOFT_KINDS
        ) | set(_CURVE_KINDS):
            raise ValueError(
                "loft interp='shape' requires polygon_2d / bspline_2d / bezier_2d "
                f"children, got {sorted(str(k) for k in kinds)}"
            )
        if len(kinds) != 1:
            raise ValueError(
                "loft interp='shape' requires all children to be the SAME kind "
                f"(outline correspondence); got {sorted(str(k) for k in kinds)}"
            )
        kind = kinds.pop()
        if kind in _CURVE_KINDS:
            # REFUSED RATHER THAN SAMPLED, and this is where the fork's version
            # would have shipped a silent divergence. `_compile_loft` sends
            # curve sections to `ops.loft_shape_curve`, which interpolates the
            # CONTROL POINTS and samples afterwards. Sampling to outlines here
            # and interpolating those agrees only while the interpolation is
            # linear; PCHIP is not, so a smooth curve-sectioned loft would
            # render a different solid from the one that gets meshed, and it
            # would look plausible.
            raise NotImplementedError(
                f"loft interp='shape' over {kind} sections: the JAX field interpolates "
                "CONTROL POINTS (ops.loft_shape_curve) and this emitter would have to "
                "interpolate them per fragment to agree. Use interp='field', or "
                "polygon_2d sections"
            )

        outlines: list[list[Any]] = []
        for c in children:
            verts = (c.get("params") or {}).get("vertices")
            if not isinstance(verts, (list, tuple)) or len(verts) < 3:
                raise ValueError("loft interp='shape' section needs >= 3 outline vertices")
            outlines.append(list(verts))
        outlines = self._resample_sections(outlines)
        counts = {len(o) for o in outlines}
        if len(counts) != 1:
            raise ValueError(
                "loft interp='shape' needs the same vertex count in every section "
                f"(vertex correspondence); got {sorted(counts)}"
            )
        n_vert = counts.pop()
        if n_vert > self.poly_max_n:  # pragma: no cover - pre-scan covers lofts
            raise AssertionError("poly_max_n underestimated a loft section outline")

        # WHERE THE SECTIONS LIVE, and on banana this is the whole file.
        # Thirteen shape-lofts of twelve 50-vertex sections is 7,800 vertices,
        # and as `vec2` literals that was 184,398 of a 207,159-char shader,
        # 89%. Above the threshold they go in the table and the loop fetches
        # them; `_table_outline` is headerless because the count is a
        # compile-time constant here and the loop indexes section-major.
        literal = all(_is_literal_vertex(v) for o in outlines for v in o)
        tabled = self.poly_table_on and literal
        if tabled:
            base = self._table_outline([v for o in outlines for v in o])
            body: list[str] = []
        else:
            flat = ", ".join(self._emit_value(list(v), "vec2") for o in outlines for v in o)
            body = [f"vec2 SEC[{n_sec * n_vert}] = vec2[{n_sec * n_vert}]({flat});"]
        body += span
        fetch = (lambda k: f"sdm_poly_fetch({base} + {k})") if tabled else (lambda k: f"SEC[{k}]")
        body += [
            "vec2 V[SDM_POLY_MAX_N];",
            f"for (int j = 0; j < {n_vert}; ++j) {{",
            f"    vec2 a = {fetch(f'idx * {n_vert} + j')};",
            f"    vec2 b = {fetch(f'(idx + 1) * {n_vert} + j')};",
        ]
        if not smooth:
            body += ["    V[j] = mix(a, b, t);"]
        else:
            body += [
                f"    vec2 am = {fetch(f'im1 * {n_vert} + j')};",
                f"    vec2 b2 = {fetch(f'i2 * {n_vert} + j')};",
                "    vec2 m0 = sdm_pchip_tan2(am, a, b, ZS[idx] - ZS[im1], hseg);",
                "    vec2 m1 = sdm_pchip_tan2(a, b, b2, hseg, ZS[i2] - ZS[idx + 1]);",
                "    float t2 = t * t;",
                "    float t3 = t2 * t;",
                "    V[j] = (2.0 * t3 - 3.0 * t2 + 1.0) * a",
                "         + (t3 - 2.0 * t2 + t) * hseg * m0",
                "         + (-2.0 * t3 + 3.0 * t2) * b",
                "         + (t3 - t2) * hseg * m1;",
            ]
        return body + ["}", f"float d2d = sdf_polygon_2d(p.xy, V, {n_vert});"]

    # -- Sweep ------------------------------------------------------------

    def _emit_sweep(self, node: dict[str, Any]) -> str:
        """Emit a ``sweep`` node using ``sdf_ops.sweep`` segment-search semantics.

        The path is sampled and its per-segment frames computed at emit time
        by the SAME functions the JAX compiler calls (``_sample_path_3d`` and
        ``_sweep_frames``), then baked: the frame is a function of the whole
        path and never of the query point, so it is data, not shader
        arithmetic. The emitted body does what the kernel does per query:
        find the single nearest segment by true 3-D distance, evaluate the
        child's 2-D function in that segment's cross-section plane (through
        ``sdm_sweep_joint``, which rounds an interior vertex), and cap a true
        open end's axial overshoot with ``sdm_sweep_finish``.

        Path control points and ``normal0`` must be literal. Both are baked,
        and a ``$ref`` in either would freeze the geometry at its emit-time
        value while the param moved; a per-point uniform cannot help because
        the rotation-minimising frame depends nonlinearly on every point.
        ``loft`` bakes its outlines for the same reason. The param reports
        as ``"re-emit"`` in ``controls``, which is what an edit to it costs.

        Above the table threshold the frames go into ``sweep_table`` and the
        body calls ``sdm_sweep_tab`` with a header index, the polygon
        precedent. Paths are never resampled: unlike a polygon outline,
        decimating a path moves the frame and the profile with it.

        Args:
            node: The ``sweep`` node.

        Returns:
            str: The emitted function's name.

        Raises:
            ValueError: Missing ``path``, or a path/frame the sampler or the
                frame builder refuses (wrong count for the kind, unknown
                ``frame``).
            NotImplementedError: A ``$ref`` or expression in a path point or
                in ``normal0``.
        """
        params = node.get("params") or {}
        if "path" not in params:
            raise ValueError("sweep requires a 'path' param: a list of [x, y, z] control points")
        kind, closed, frame = _sweep_structure(params)
        ctrl = params["path"]
        if not (isinstance(ctrl, (list, tuple, np.ndarray)) and len(ctrl) > 0) or not all(
            _is_literal_point(v, 3) for v in ctrl
        ):
            raise NotImplementedError(
                "sweep path control points must be literal [x, y, z] numbers for GLSL "
                "emission; a `$ref` or expression point would be baked at its emit-time "
                "value and stop tracking its param. The frame depends on the whole path, "
                "so no per-point uniform can carry it; re-emit the part instead."
            )
        normal0 = params.get("normal0")
        if normal0 is not None and not _is_literal_point(normal0, 3):
            raise NotImplementedError(
                "sweep normal0 must be a literal [x, y, z] for GLSL emission; it seeds the "
                "rotation-minimising frame, which is baked, so a `$ref` there would be frozen "
                "at its emit-time value. Re-emit the part instead."
            )
        child_fn = self.emit_node(node["child"], dim=2)

        A, T, L, N, B = _sweep_frame_arrays(ctrl, kind, closed, frame, normal0)
        n_seg = int(A.shape[0])
        self.sweep_max_s = max(self.sweep_max_s, n_seg)

        fn = self._fn_name(3)
        if self.sweep_table_on:
            hdr = self._table_sweep(A, T, L, N, B, closed)
            body = [
                f"vec3 uvo = sdm_sweep_tab(p, {hdr});",
                f"float d2d = {child_fn}(uvo.xy);",
                "return sdm_sweep_finish(d2d, uvo.z);",
            ]
            self._emit_function(f"float {fn}(vec3 p)", body)
            return fn

        def vec3s(arr: np.ndarray) -> str:
            return ", ".join(
                f"vec3({_float_literal(x)}, {_float_literal(y)}, {_float_literal(z)})"
                for x, y, z in arr
            )

        body = [
            f"vec3 A[{n_seg}] = vec3[{n_seg}]({vec3s(A)});",
            f"vec3 T[{n_seg}] = vec3[{n_seg}]({vec3s(T)});",
            f"vec3 NN[{n_seg}] = vec3[{n_seg}]({vec3s(N)});",
            f"vec3 BB[{n_seg}] = vec3[{n_seg}]({vec3s(B)});",
            f"float L[{n_seg}] = float[{n_seg}]({', '.join(_float_literal(x) for x in L)});",
            # `foot` then `p - foot`, associated as ops.sweep writes it, and a
            # strict `<` that keeps the FIRST minimum as jnp.argmin does. Both
            # matter where two segments tie exactly (a query beyond a vertex,
            # both feet clamped to it): the kernel's answer there is whichever
            # segment rounding puts first, and the shader has to round the
            # same way to give the same answer.
            "int istar = 0;",
            "float best = 3.0e38;",
            "float s_star = 0.0;",
            "vec3 perp_star = vec3(0.0);",
            f"for (int i = 0; i < {n_seg}; ++i) {{",
            "    float s = dot(p - A[i], T[i]);",
            "    vec3 perp = p - (A[i] + clamp(s, 0.0, L[i]) * T[i]);",
            "    float d2 = dot(perp, perp);",
            "    if (d2 < best) { best = d2; istar = i; s_star = s; perp_star = perp; }",
            "}",
            # Interior vertices are ball joints, the two true ends of an open
            # path flat caps: the same `flat` split ops.sweep makes (named
            # `end_cap` in GLSL, where `flat` is a reserved qualifier), with the
            # loop's closedness baked as a literal.
            "float over_lo = -s_star;",
            "float over_hi = s_star - L[istar];",
            "float over = max(over_lo, over_hi);",
            (
                "bool end_cap = false;"
                if closed
                else "bool end_cap = (istar == 0 && over_lo > 0.0)"
                f" || (istar == {n_seg - 1} && over_hi > 0.0);"
            ),
            "vec2 uv = sdm_sweep_joint(",
            "    vec2(dot(perp_star, NN[istar]), dot(perp_star, BB[istar])), over, end_cap);",
            f"float d2d = {child_fn}(uv);",
            "return sdm_sweep_finish(d2d, end_cap ? over : 0.0);",
        ]
        self._emit_function(f"float {fn}(vec3 p)", body)
        return fn

    def _table_sweep(
        self,
        A: np.ndarray,
        T: np.ndarray,
        L: np.ndarray,
        N: np.ndarray,
        B: np.ndarray,
        closed: bool,
    ) -> int:
        """Append one sweep's frames to the table WITH a header texel. Returns
        the header index.

        The header is ``(segment count, closed, 0, 0)``; ``sdm_sweep_tab``
        needs ``closed`` to tell a true open end (flat cap) from an interior
        vertex (ball joint). Four RGBA texels per segment follow, in the order
        it reads them: ``(A, L)``, ``(T, 0)``, ``(N, 0)``, ``(B, 0)``. The
        count is fetched at run time for the same reason the polygon header
        is: a literal would let the driver unroll the loop per call site.
        """
        hdr = len(self.sweep_table) // 4
        self.sweep_table.extend((float(A.shape[0]), 1.0 if closed else 0.0, 0.0, 0.0))
        for a, t, ln, n, b in zip(A, T, L, N, B, strict=True):
            self.sweep_table.extend((float(a[0]), float(a[1]), float(a[2]), float(ln)))
            self.sweep_table.extend((float(t[0]), float(t[1]), float(t[2]), 0.0))
            self.sweep_table.extend((float(n[0]), float(n[1]), float(n[2]), 0.0))
            self.sweep_table.extend((float(b[0]), float(b[1]), float(b[2]), 0.0))
        return hdr

    # -- CSG op ----------------------------------------------------------

    def _emit_op(self, node: dict[str, Any], dim: int) -> str:
        op_name = node["op"]
        children = node.get("children", [])
        if not children:
            raise ValueError("CSG op has no children")
        child_fns = [self.emit_node(c, dim=dim) for c in children]
        kwargs = node.get("params") or {}

        # Hard CSG: may be promoted to smooth via metadata flag.
        if op_name in _HARD_BINARY_OPS:
            if self.smooth_csg:
                smooth_name = _HARD_TO_SMOOTH[op_name]
                k_expr = _float_literal(self.smooth_k)
                body = _csg_fold(child_fns, f"op_{smooth_name}", extra_arg=k_expr)
            else:
                body = _csg_fold(child_fns, f"op_{op_name}")
        elif op_name in _SMOOTH_BINARY_OPS:
            if "k" not in kwargs:
                raise KeyError(f"Smooth op {op_name!r} requires a 'k' kwarg")
            k_expr = self._emit_value(kwargs["k"], "float")
            body = _csg_fold(child_fns, f"op_{op_name}", extra_arg=k_expr)
        else:
            raise ValueError(f"Unknown CSG op {op_name!r}")

        fn = self._fn_name(dim)
        self._emit_function(f"float {fn}(vec{dim} p)", body)
        return fn

    # -- Transform -------------------------------------------------------

    def _emit_transform(self, node: dict[str, Any], dim: int) -> str:
        tf_name = node["transform"]
        child_fn = self.emit_node(node["child"], dim=dim)
        kwargs = node.get("params") or {}

        if tf_name == "translate":
            t_expr = self._emit_value(kwargs["t"], f"vec{dim}")
            body = [f"vec{dim} q = tf_translate{dim}(p, {t_expr});", f"return {child_fn}(q);"]
        elif tf_name == "scale":
            s_expr = self._emit_value(kwargs["s"], "float")
            # Mirror compile.py: child(p/s, free) * s
            body = [
                f"vec{dim} q = tf_scale{dim}(p, {s_expr});",
                f"return {child_fn}(q) * {s_expr};",
            ]
        elif tf_name == "scale_axis":
            s_expr = self._emit_value(kwargs["s"], f"vec{dim}")
            # Mirror compile.py: min(s) * child(p / s). A named local so the
            # vector is spelled once and every min() reads the same value.
            smin = "min(sa.x, sa.y)" if dim == 2 else "min(min(sa.x, sa.y), sa.z)"
            body = [
                f"vec{dim} sa = {s_expr};",
                f"vec{dim} q = p / sa;",
                f"return {child_fn}(q) * {smin};",
            ]
        elif tf_name == "rotate_x":
            if dim != 3:
                raise ValueError("rotate_x is 3-D only")
            angle_expr = self._emit_value(kwargs["angle"], "float")
            body = [f"vec3 q = tf_rotate_x(p, {angle_expr});", f"return {child_fn}(q);"]
        elif tf_name == "rotate_y":
            if dim != 3:
                raise ValueError("rotate_y is 3-D only")
            angle_expr = self._emit_value(kwargs["angle"], "float")
            body = [f"vec3 q = tf_rotate_y(p, {angle_expr});", f"return {child_fn}(q);"]
        elif tf_name == "rotate_z":
            if dim != 3:
                raise ValueError("rotate_z is 3-D only")
            angle_expr = self._emit_value(kwargs["angle"], "float")
            body = [f"vec3 q = tf_rotate_z(p, {angle_expr});", f"return {child_fn}(q);"]
        elif tf_name == "rotate_matrix":
            if dim != 3:
                raise ValueError("rotate_matrix is 3-D only")
            R_expr = self._emit_value(kwargs["R"], "mat3")
            body = [f"vec3 q = tf_rotate_matrix(p, {R_expr});", f"return {child_fn}(q);"]
        elif tf_name == "repeat_finite":
            c_expr = self._emit_value(kwargs["c"], "float")
            l_expr = self._emit_value(kwargs["l"], f"vec{dim}")
            body = [
                f"vec{dim} q = op_repeat_finite{dim}(p, {c_expr}, {l_expr});",
                f"return {child_fn}(q);",
            ]
        elif tf_name == "canonical_sector_fold":
            if dim != 3:
                raise ValueError("canonical_sector_fold is 3-D only")
            n_expr = self._emit_value(kwargs["n_sectors"], "float")
            if kwargs.get("centered"):
                ph_expr = self._emit_value(kwargs.get("phase_frac", 0.0), "float")
                fold = f"vec3 q = tf_canonical_sector_fold_c(p, n_sec, {ph_expr});"
            else:
                fold = "vec3 q = tf_canonical_sector_fold(p, n_sec);"
            # compile.py wraps n_sectors in int(). floor() truncates the count.
            # thus preview and mesh agree about the count.
            #
            # The child is evaluated at the folded point and at that point
            # rotated one sector each way, and the nearest wins. See the same
            # three calls in compile.py for why; they must stay in lockstep.
            body = [
                f"float n_sec = floor({n_expr});",
                fold,
                "float sec = 6.28318530717958647692 / n_sec;",
                f"float d = {child_fn}(q);",
                f"d = min(d, {child_fn}(tf_rotate_z(q, sec)));",
                f"d = min(d, {child_fn}(tf_rotate_z(q, -sec)));",
                "return d;",
            ]
        elif tf_name == "mirror":
            n_expr = self._emit_value(kwargs["n"], f"vec{dim}")
            o_expr = self._emit_value(kwargs["o"], f"vec{dim}")
            suffix = "" if dim == 3 else str(dim)
            reflect = f"tf_reflect_plane{suffix}(p, {n_expr}, {o_expr})"
            # The child unioned with its own reflection, so it does not matter
            # which side of the plane the child sits on, or whether it crosses.
            # compile.py promotes this union under smooth_csg, so this does too.
            if self.smooth_csg:
                k_expr = _float_literal(self.smooth_k)
                combine = f"op_smooth_union({child_fn}(p), {child_fn}(q), {k_expr})"
            else:
                combine = f"op_union({child_fn}(p), {child_fn}(q))"
            body = [f"vec{dim} q = {reflect};", f"return {combine};"]
        else:
            raise ValueError(f"Unknown transform {tf_name!r}")

        fn = self._fn_name(dim)
        self._emit_function(f"float {fn}(vec{dim} p)", body)
        return fn

    # -- Modifier --------------------------------------------------------

    def _emit_modifier(self, node: dict[str, Any], dim: int) -> str:
        name = node["modifier"]
        child_fn = self.emit_node(node["child"], dim=dim)
        kwargs = node.get("params") or {}

        if name == "round":
            r_expr = self._emit_value(kwargs["r"], "float")
            body = [f"return op_round({child_fn}(p), {r_expr});"]
        elif name == "onion":
            t_expr = self._emit_value(kwargs["thickness"], "float")
            body = [f"return op_onion({child_fn}(p), {t_expr});"]
        elif name == "elongate":
            h_expr = self._emit_value(kwargs["h"], f"vec{dim}")
            body = [f"vec{dim} q = op_elongate{dim}(p, {h_expr});", f"return {child_fn}(q);"]
        else:
            raise ValueError(f"Unknown modifier {name!r}")

        fn = self._fn_name(dim)
        self._emit_function(f"float {fn}(vec{dim} p)", body)
        return fn

    # -- Deform ----------------------------------------------------------

    def _warp_call(self, node: dict[str, Any], src: str) -> str:
        """The ``vec3`` query-warp expression one point-warp deform applies.

        The distance emission (:meth:`_emit_deform`) warps the query before
        handing it to the child;
        ``emit_rest_fn`` applies the identical warp to recover material
        coordinates.

        Args:
            node: A ``deform`` node whose name is in ``_POINT_WARP_DEFORMS``.
            src: The GLSL expression naming the point to warp.

        Returns:
            str: A ``vec3``-valued GLSL expression.

        Raises:
            ValueError: If ``node`` is not a point-warp deform.
        """
        name = node["deform"]
        kwargs = node.get("params") or {}
        if name == "twist":
            return f"op_twist({src}, {self._emit_value(kwargs['k'], 'float')})"
        if name == "bend":
            return f"op_bend({src}, {self._emit_value(kwargs['k'], 'float')})"
        if name == "twist_radial":
            # angle_inner/angle_outer may be $refs -> live uniforms: DOF end
            # rotations that BEND the shape between them (a BCM stand-in).
            return (
                f"op_twist_radial({src}, "
                f"{self._emit_value(kwargs['r0'], 'float')}, "
                f"{self._emit_value(kwargs['r1'], 'float')}, "
                f"{self._emit_value(kwargs['angle_inner'], 'float')}, "
                f"{self._emit_value(kwargs['angle_outer'], 'float')})"
            )
        if name == "twist_linear":
            # The plate sibling of twist_radial: angle_0/angle_1 are the two
            # WELDED-EDGE rotations, ramped along a diameter rather than rho.
            axis = kwargs.get("axis", [1.0, 0.0])
            return (
                f"op_twist_linear({src}, "
                f"vec2({self._emit_value(axis[0], 'float')}, "
                f"{self._emit_value(axis[1], 'float')}), "
                f"{self._emit_value(kwargs['u0'], 'float')}, "
                f"{self._emit_value(kwargs['u1'], 'float')}, "
                f"{self._emit_value(kwargs['angle_0'], 'float')}, "
                f"{self._emit_value(kwargs['angle_1'], 'float')})"
            )
        if name == "taper_linear":
            return (
                f"op_taper_linear({src}, "
                f"{self._emit_value(kwargs['z0'], 'float')}, "
                f"{self._emit_value(kwargs['z1'], 'float')}, "
                f"{self._emit_value(kwargs['s_0'], 'float')}, "
                f"{self._emit_value(kwargs['s_1'], 'float')})"
            )
        if name == "shear_linear":
            axis = kwargs.get("axis", [1.0, 0.0])
            return (
                f"op_shear_linear({src}, "
                f"vec2({self._emit_value(axis[0], 'float')}, "
                f"{self._emit_value(axis[1], 'float')}), "
                f"{self._emit_value(kwargs['u0'], 'float')}, "
                f"{self._emit_value(kwargs['u1'], 'float')}, "
                f"{self._emit_value(kwargs['dz_0'], 'float')}, "
                f"{self._emit_value(kwargs['dz_1'], 'float')})"
            )
        raise ValueError(f"Not a point-warp deform: {name!r}")

    def emit_rest_fn(self, chain: list[dict[str, Any]]) -> str:
        """Emit ``vec3 fn(vec3 p)`` returning a component's MATERIAL-space point.

        Applies the same query warps the component's distance function applies,
        in the same order, via :meth:`_warp_call`. A host cuts and textures in
        the result, so both travel with the material as the deform params
        animate -- which is the whole reason the function exists: a monochrome
        surface that twists has no feature to track, and a pattern painted in
        world space stands still while the body moves through it.

        Args:
            chain: The component's point-warp deform prefix, outermost first.

        Returns:
            str: The emitted function's name.
        """
        fn = f"sdm_rest_{self._next_id}"
        self._next_id += 1
        lines = ["vec3 q = p;"]
        lines += [f"q = {self._warp_call(node, 'q')};" for node in chain]
        lines.append("return q;")
        self._emit_function(f"vec3 {fn}(vec3 p)", lines)
        return fn

    def _emit_deform(self, node: dict[str, Any], dim: int) -> str:
        if dim != 3:
            raise ValueError(f"Deforms are 3-D only (got dim={dim})")
        name = node["deform"]
        child_fn = self.emit_node(node["child"], dim=3)

        if name in _POINT_WARP_DEFORMS:
            body = [
                f"vec3 q = {self._warp_call(node, 'p')};",
                f"return {child_fn}(q);",
            ]
        elif name == "displace":
            if "field" not in node:
                raise ValueError("deform 'displace' requires a 'field' subtree")
            field_fn = self._emit_field_node(node["field"])
            body = [f"return {child_fn}(p) + {field_fn}(p);"]
        else:
            raise ValueError(f"Unknown / unsupported deform {name!r}")

        fn = self._fn_name(3)
        self._emit_function(f"float {fn}(vec3 p)", body)
        return fn

    # -- 2D -> 3D lift ---------------------------------------------------

    def _emit_2d_to_3d(self, node: dict[str, Any], dim: int) -> str:
        if dim != 3:
            raise ValueError(f"2d_to_3d lift produces a 3-D function (got dim={dim})")
        method = node["method"]
        child_fn = self.emit_node(node["child"], dim=2)
        kwargs = node.get("params") or {}

        if method == "revolution":
            offset_raw = kwargs.get("offset", 0.0)
            offset_expr = self._emit_value(offset_raw, "float")
            body = [f"vec2 q = lift_revolution_q(p, {offset_expr});", f"return {child_fn}(q);"]
        elif method == "extrusion":
            h_expr = self._emit_value(kwargs["h"], "float")
            body = [
                f"float d = {child_fn}(p.xy);",
                f"return lift_extrusion_finish(d, p, {h_expr});",
            ]
        else:
            raise ValueError(f"Unknown 2d_to_3d method {method!r}")

        fn = self._fn_name(3)
        self._emit_function(f"float {fn}(vec3 p)", body)
        return fn

    # -- Field tree (displacement) --------------------------------------

    def _emit_field_node(self, node: Any) -> str:
        if not isinstance(node, dict) or "type" not in node:
            raise ValueError(f"Not a field node: {node!r}")
        kind = node["type"]

        if kind == "field":
            prim = node["kind"]
            if prim not in _FIELD_SPECS:
                raise ValueError(f"Unknown field primitive {prim!r}")
            spec = _FIELD_SPECS[prim]
            authored = dict(node.get("params") or {})
            defaults = _FIELD_DEFAULTS.get(prim, {})
            arg_exprs: list[str] = []
            for arg_name, arg_type in spec:
                if arg_name in authored:
                    value = authored.pop(arg_name)
                elif arg_name in defaults:
                    value = defaults[arg_name]
                else:
                    raise KeyError(f"Field primitive {prim!r} missing required arg {arg_name!r}")
                arg_exprs.append(self._emit_value(value, arg_type))
            if authored:
                raise KeyError(
                    f"Field primitive {prim!r} does not accept param(s) {sorted(authored)} "
                    f"(expected {[name for name, _ in spec]})"
                )
            fn = self._field_name()
            call = f"field_{prim}(p, {', '.join(arg_exprs)})"
            self._emit_function(f"float {fn}(vec3 p)", [f"return {call};"])
            return fn

        if kind == "field_op":
            op_name = node["op"]
            children = node.get("children", [])
            if not children:
                raise ValueError("field_op has no children")
            child_fns = [self._emit_field_node(c) for c in children]
            if op_name == "add":
                terms = " + ".join(f"{cf}(p)" for cf in child_fns)
                body = [f"return {terms};"]
            else:
                raise ValueError(f"Unknown field op {op_name!r}")
            fn = self._field_name()
            self._emit_function(f"float {fn}(vec3 p)", body)
            return fn

        raise ValueError(f"Unknown field node type {kind!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ===========================================================================
# Expression-tree operators
# ===========================================================================
# One entry per operator in `dsl.expr`, keyed by the SAME name, and
# `test_glsl_expr` asserts the two key sets are equal. An operator that exists
# on one side only is a part that optimises against one arithmetic and renders
# with another.
#
# Everything is parenthesised rather than tracked by precedence. GLSL and the
# DSL agree on precedence for what is here, but the trees nest arbitrarily and
# a missing bracket is a wrong number, not a compile error.

# fmt: off
_EXPR_UNARY: dict[str, Callable[[str], str]] = {
    "neg":    lambda x: f"(-{x})",
    "abs":    lambda x: f"abs({x})",
    "sqrt":   lambda x: f"sqrt({x})",
    "log":    lambda x: f"log({x})",
    "exp":    lambda x: f"exp({x})",
    # DUPLICATES ITS CHILD, deliberately. GLSL has no `sqr`, and `pow(x, 2.0)`
    # is undefined for a negative base in the GLSL spec while `jnp.square` is
    # not, so `pow` here would disagree with the JAX side over exactly the
    # inputs a signed offset produces. Multiplying is exact everywhere. The
    # cost is that nested squares double the emitted text each time, which is
    # bounded in practice: a slot expression is a handful of nodes.
    "square": lambda x: f"({x} * {x})",
    "sin": lambda x: f"sin({x})",
    "cos": lambda x: f"cos({x})",
}

_EXPR_BINARY: dict[str, Callable[[str, str], str]] = {
    "+":   lambda a, b: f"({a} + {b})",
    "-":   lambda a, b: f"({a} - {b})",
    "*":   lambda a, b: f"({a} * {b})",
    "/":   lambda a, b: f"({a} / {b})",
    # GLSL leaves `pow` undefined for a negative base, and for a zero base with
    # a non-positive exponent; `jnp.power` defines both. The two therefore
    # agree only for a positive base, which is where a dimension lives. Nothing
    # here can check that statically, so it is stated rather than guarded.
    "pow": lambda a, b: f"pow({a}, {b})",
    "min": lambda a, b: f"min({a}, {b})",
    "max": lambda a, b: f"max({a}, {b})",
}

_EXPR_REDUCE: dict[str, Callable[[list[str]], str]] = {
    "sum":  lambda xs: "(" + " + ".join(xs) + ")",
    "mean": lambda xs: "((" + " + ".join(xs) + f") / {_float_literal(len(xs))})",
    "min":  lambda xs: _fold_call("min", xs),
    "max":  lambda xs: _fold_call("max", xs),
}
# fmt: on


class _MotionExpressionEmitter(_GLSLEmitter):
    """Use motion-only DOF inputs while sharing geometry's design uniforms."""

    def _emit_expr(self, node: Any, _stack: tuple[str, ...] = ()) -> str:
        if isinstance(node, dict) and node.get("type") == "dof":
            return self.dof_expressions[node["name"]]
        return super()._emit_expr(node, _stack)

    dof_expressions: dict[str, str]


def _motion_expressions(emitter: _GLSLEmitter) -> _MotionExpressionEmitter:
    """Share design uniforms and canonical DOF expressions across motion emitters."""
    block = emitter.part.kinematics
    assert block is not None
    expressions = _MotionExpressionEmitter(
        emitter.part, smooth_csg=emitter.smooth_csg, smooth_k=emitter.smooth_k
    )
    expressions.uniforms = emitter.uniforms
    expressions.dof_expressions = {
        dof["name"]: f"(u_dof_{i} * {_float_literal(math.pi / 180)})"
        if dof["unit"] == "deg"
        else f"u_dof_{i}"
        for i, dof in enumerate(block.get("dofs", []))
    }
    return expressions


class _BodyMotionEmission(NamedTuple):
    functions: list[str]
    rest_functions: list[str]
    uniforms: list[UniformDecl]
    controls: list[dict[str, Any]]


def _emit_body_motion(emitter: _GLSLEmitter, regions: list[SDFTree]) -> _BodyMotionEmission:
    """Wrap rest-region fields in inverse operations, in reverse authored order."""
    block = emitter.part.kinematics
    assert block is not None
    expressions = _motion_expressions(emitter)
    uniforms: list[UniformDecl] = []
    controls: list[dict[str, Any]] = []
    for i, dof in enumerate(block.get("dofs", [])):
        key, uniform = f"kinematics.{dof['name']}", f"u_dof_{i}"
        if key in emitter.part.params:
            raise ValueError(f"Design parameter {key!r} conflicts with a motion control key")
        bounds = (float(dof["range"][0]), float(dof["range"][1]))
        value = float(dof.get("default", 0))
        uniforms.append(UniformDecl(uniform, "float", value, bounds, dof["unit"], key))
        controls.append(
            {
                "param": key,
                "dof": dof["name"],
                "class": "live",
                "uniform": uniform,
                "value": value,
                "free": False,
                "unit": dof["unit"],
                "bounds": list(bounds),
                "ui": {
                    "role": "pose",
                    "label": dof["name"],
                    "group": "Motion",
                    "explore_bounds": list(bounds),
                },
            }
        )
    functions, rest_functions = [], []
    for i, (body, region) in enumerate(zip(block["bodies"], regions, strict=True)):
        fn = emitter.emit_node(region, dim=3)
        rest_fn, body_fn = f"sdm_body_rest_{i}", f"sdf_body_{i}"
        lines = []
        for j, op in enumerate(reversed(body["motion"]["ops"])):
            norm = math.hypot(*op["axis"])
            axis = "vec3(" + ", ".join(_float_literal(x / norm) for x in op["axis"]) + ")"
            value_expr = expressions._emit_expr(
                op["angle"] if op["kind"] == "rotate" else op["distance"]
            )
            if op["kind"] == "translate":
                lines.append(f"p -= {axis} * ({value_expr});")
            else:
                origin = (
                    "vec3("
                    + ", ".join(_float_literal(x) for x in op.get("origin", [0, 0, 0]))
                    + ")"
                )
                lines.extend(
                    [
                        f"vec3 q{j} = p - {origin};",
                        f"float a{j} = {value_expr};",
                        f"p = {origin} + cos(a{j}) * q{j} - sin(a{j}) * cross({axis}, q{j})"
                        f" + (1.0 - cos(a{j})) * {axis} * dot({axis}, q{j});",
                    ]
                )
        emitter._emit_function(f"vec3 {rest_fn}(vec3 p)", [*lines, "return p;"])
        emitter._emit_function(f"float {body_fn}(vec3 p)", [f"return {fn}({rest_fn}(p));"])
        functions.append(body_fn)
        rest_functions.append(rest_fn)
    return _BodyMotionEmission(functions, rest_functions, uniforms, controls)


def _fold_call(fn: str, xs: list[str]) -> str:
    """``min(min(a, b), c)``. GLSL's min/max are strictly binary."""
    out = xs[0]
    for x in xs[1:]:
        out = f"{fn}({out}, {x})"
    return out


def _float_literal(x: Any) -> str:
    """Format a Python number as an unambiguous GLSL float literal.

    GLSL forbids implicit int->float conversion in some expressions, so we
    always emit a decimal point.
    """
    f = float(x)
    if f != f:  # noqa: PLR0124 -- self-comparison IS the NaN test (NaN != NaN)
        raise ValueError("NaN is not a valid GLSL literal")
    if f == float("inf"):
        return "1.0/0.0"
    if f == float("-inf"):
        return "-1.0/0.0"
    s = repr(f)
    return s if ("." in s or "e" in s or "E" in s) else f"{s}.0"


def _scan_polygon_verts(node: Any, poly_lod: int | None) -> int:
    """Vertex count one node will emit, LOD-aware; 0 for non-outline nodes.

    Must agree with what emission actually does: a literal outline above
    ``poly_lod`` is resampled down to it, a ``$ref``-bearing outline is left
    at full resolution (its vertices are live uniforms), and a curve's sampled
    outline is always literal.
    """
    if not (isinstance(node, dict) and node.get("type") == "primitive"):
        return 0
    kind = node.get("kind")
    if kind == "polygon_2d":
        verts = (node.get("params") or {}).get("vertices") or []
        n = len(verts)
        if poly_lod and n > poly_lod and all(_is_literal_vertex(v) for v in verts):
            return poly_lod
        return n
    if kind in _CURVE_KINDS:
        cps = (node.get("params") or {}).get("control_points") or []
        spans = len(cps) if kind == "bspline_2d" else len(cps) // 3
        n = spans * _curve_samples()
        return min(n, poly_lod) if poly_lod else n
    return 0


def _total_polygon_verts(node: Any, poly_lod: int | None = None) -> int:
    """Every polygon vertex in ``node``, summed, curve outlines included.

    The SUM rather than the maximum, because the cost this sizes is the length
    of the emitted source: one 600-vertex outline and six hundred 1-vertex ones
    weigh the same in the file and `_max_polygon_verts` cannot tell them apart.
    Loft sections are counted through `children`, which is where they sit.
    LOD-aware so table-mode and array-size decisions match what gets emitted.
    """
    total = _scan_polygon_verts(node, poly_lod)
    if isinstance(node, dict):
        for key in ("child", "children", "field"):
            if key in node:
                total += _total_polygon_verts(node[key], poly_lod)
    elif isinstance(node, (list, tuple)):
        for item in node:
            total += _total_polygon_verts(item, poly_lod)
    return total


def _max_polygon_verts(node: Any, poly_lod: int | None = None) -> int:
    """Largest ``polygon_2d`` vertex count anywhere in ``node`` (0 if none).

    A cheap recursive walk over the raw tree dict, independent of the emitter,
    so ``SDM_POLY_MAX_N`` is known before any function is emitted. Walks all
    child-bearing keys used by the DSL (``child``, ``children``, ``field``).
    LOD-aware: a resampled outline is emitted at ``poly_lod`` vertices, and
    sizing the shared array to the pre-LOD count would waste registers on
    every polygon call in the scene. Curve outlines are counted here rather
    than after lowering, because SDM_POLY_MAX_N has to be fixed before
    anything is emitted, and an undercount emits V[] shorter than the `n`
    the polygon helper loops to.
    """
    best = _scan_polygon_verts(node, poly_lod)
    if isinstance(node, dict):
        for key in ("child", "children", "field"):
            if key in node:
                best = max(best, _max_polygon_verts(node[key], poly_lod))
    elif isinstance(node, (list, tuple)):
        for item in node:
            best = max(best, _max_polygon_verts(item, poly_lod))
    return best


def _sweep_structure(params: dict[str, Any]) -> tuple[str, bool, str]:
    """``(path_kind, closed, frame)`` read structurally, as ``_compile_sweep``
    reads them: plain ``dict.get``, never through the ``$ref`` resolver, and a
    ``bspline`` path is a loop whatever ``closed`` says."""
    kind = str(params.get("path_kind", "bspline"))
    frame = str(params.get("frame", "rmf"))
    closed = bool(params.get("closed", False)) or kind == "bspline"
    return kind, closed, frame


def _sweep_frame_arrays(
    ctrl: Any, kind: str, closed: bool, frame: str, normal0: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a literal path and build its per-segment frames, ``(A, T, L, N, B)``.

    Calls the sampler and frame builder the JAX kernel calls, on the same
    dtype the compiler resolves the path to, and converts the results to
    numpy. Transcribing either into this file would be a second copy of the
    double-reflection recurrence, and a frame that drifted by a degree over a
    coil would still render a plausible winding.
    """
    import jax.numpy as jnp

    from software_defined_matter.sdf import sdf_ops

    pts = sdf_ops._sample_path_3d(jnp.asarray(np.asarray(ctrl, dtype=float)), kind, closed)
    n0 = jnp.asarray(np.asarray(normal0, dtype=float)) if normal0 is not None else None
    A, T, L, N, B = sdf_ops._sweep_frames(pts, closed, frame, n0)
    return np.asarray(A), np.asarray(T), np.asarray(L), np.asarray(N), np.asarray(B)


def _total_sweep_points(node: Any) -> int:
    """Every sampled path point of every ``sweep`` in ``node``, summed.

    The sum, as `_total_polygon_verts` sums, because the cost this sizes is
    the length of the emitted source. A non-literal path counts as nothing:
    emission refuses it before it could be stored either way.
    """
    total = 0
    if isinstance(node, dict):
        if node.get("type") == "sweep":
            params = node.get("params") or {}
            ctrl = params.get("path")
            if isinstance(ctrl, (list, tuple, np.ndarray)) and all(
                _is_literal_point(v, 3) for v in ctrl
            ):
                kind, closed, _frame = _sweep_structure(params)
                total += _sampled_path_count(len(ctrl), kind, closed)
        for key in ("child", "children", "field"):
            if key in node:
                total += _total_sweep_points(node[key])
    elif isinstance(node, (list, tuple)):
        for item in node:
            total += _total_sweep_points(item)
    return total


def _sampled_path_count(n_ctrl: int, kind: str, closed: bool) -> int:
    """Points ``_sample_path_3d`` returns for ``n_ctrl`` control points.

    Asks the sampler rather than repeating its arithmetic: a path whose
    count is wrong for its kind raises there, and the pre-scan must not
    decide table mode from a path emission will then refuse.
    """
    import jax.numpy as jnp

    from software_defined_matter.sdf import sdf_ops

    try:
        pts = sdf_ops._sample_path_3d(jnp.zeros((n_ctrl, 3)), kind, closed)
    except ValueError:
        return 0
    return int(pts.shape[0])


def _csg_fold(child_fns: list[str], op_name: str, *, extra_arg: str | None = None) -> list[str]:
    """Emit a left-fold over child functions, calling ``op_name`` at each step."""
    extra = f", {extra_arg}" if extra_arg is not None else ""
    if len(child_fns) == 1:
        # Mirror the JAX compiler: a single-child op is just the child.
        return [f"return {child_fns[0]}(p);"]
    lines: list[str] = [f"float d = {child_fns[0]}(p);"]
    for cf in child_fns[1:]:
        lines.append(f"d = {op_name}(d, {cf}(p){extra});")
    lines.append("return d;")
    return lines


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

_LIB_PATH = Path(__file__).parent / "lib.glsl"


def load_lib_glsl() -> str:
    """Return the static GLSL helper library as a string."""
    return _LIB_PATH.read_text()


def _emitter_resources(emitter: _GLSLEmitter) -> tuple[str, list[float], int]:
    """Build the shared helper library and raw grid payload for emitted fields."""
    # sdf_polygon_2d's array parameter is sized by SDM_POLY_MAX_N; define it
    # ahead of the (static) lib source so the helper compiles at the right size.
    lib_source = load_lib_glsl()
    if emitter.poly_max_n > 0:
        lib_source = f"#define SDM_POLY_MAX_N {emitter.poly_max_n}\n" + lib_source
    # This emission's real values, ahead of the library's #ifndef defaults. A
    # consumer slices this prefix off by identity to learn what THIS emission
    # needs, so it has to be the prepend and not an edit inside the library.
    if emitter.poly_table:
        lib_source = (
            "#define SDM_POLY_TABLE\n"
            f"#define SDM_POLY_TEX_W {_POLY_TEX_WIDTH}\n"
            f"#define SDM_POLY_LEN {len(emitter.poly_table) // 2}\n"
        ) + lib_source

    if emitter.sweep_table:
        lib_source = (
            "#define SDM_SWEEP_TABLE\n"
            f"#define SDM_SWEEP_TEX_W {_SWEEP_TEX_WIDTH}\n"
            f"#define SDM_SWEEP_LEN {len(emitter.sweep_table) // 4}\n"
        ) + lib_source

    grid_table: list[float] = []
    grid_tex_width = 0
    if emitter.grid_arrays:
        import numpy as _np

        flat = _np.concatenate(emitter.grid_arrays).astype(_np.float32)
        grid_table = flat.tolist()
        grid_tex_width = _GRID_TEX_WIDTH
        lib_source = (
            f"#define SDM_GRID_TABLE\n"
            f"#define SDM_GRID_TEX_W {_GRID_TEX_WIDTH}\n"
            f"#define SDM_GRID_LEN {emitter.grid_len}\n" + lib_source
        )

    return lib_source, grid_table, grid_tex_width


def emit_glsl(
    part: Part,
    tree: SDFTree | None = None,
    *,
    smooth_csg: bool | None = None,
    smooth_k: float = 0.25,
    poly_lod: int | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> GLSLEmission:
    """Emit GLSL for an SDF tree belonging to ``part``.

    Args:
        part: The :class:`~software_defined_matter.model.Part`. Used for
            parameter resolution and bbox inference (and to read
            ``metadata["smooth_csg"]`` when ``smooth_csg`` is ``None``).
        tree: SDF tree to emit in rest space. When omitted, a part with rigid
            bodies emits the union of its posed body regions; include stationary
            geometry as bodies with empty motion operations. Otherwise defaults
            to ``part.computed_envelope()``. Explicit subtree emissions do not
            apply the part's motion block.
        smooth_csg: Promote hard CSG to smooth CSG with radius ``smooth_k``.
            Mirrors the same flag in
            :func:`software_defined_matter.sdf.compile.make_sdf_closure`.
        smooth_k: Smoothing radius applied when ``smooth_csg`` is true.
        poly_lod: Resample literal outlines above this vertex count down to it
            (curvature-weighted, so an airfoil's leading edge keeps its
            density). ``None`` (the default) never touches a vertex — opt-in
            only, because it changes emitted geometry. A VIEWER-fidelity
            lever: ``sdf_polygon_2d``'s march cost is linear in vertex count
            per step; meshing/JAX paths never see this. Outlines with ``$ref``
            vertices are never resampled, and loft sections are resampled
            with one shared index set so vertex correspondence survives.
        on_stage: Called with the name of each stage as it is entered:
            ``"envelope"``, ``"glsl-walk"``, ``"bbox"``. A whole-mechanism part
            spends minutes in ``bbox`` alone, so a host with no progress signal
            can only show a spinner and hope. Exceptions raised by the callback
            are suppressed via ``contextlib.suppress(Exception)`` and do not
            abort emission.

    Returns:
        GLSLEmission: ``scene_source`` is the generated per-node functions
        plus a ``float sdf_scene(vec3 p)`` wrapper. ``lib_source`` is the
        static helper library. ``uniforms`` is the list of GLSL uniforms
        backed by design parameters and, for full-part motion, authored DOFs.
        ``bbox`` uses an explicit metadata override or static field inference.
        Rigid motion uses conservative component and scene envelopes over the
        declared motion and design ranges. An explicit scene box can enlarge
        this envelope or cover rest geometry that cannot be bounded; it never
        shrinks an inferred envelope. Regenerate after changing these ranges.
        ``controls`` classifies every param by what editing it costs,
        ``poly_max_n`` / ``poly_table`` are the polygon payload, and
        ``sweep_table`` / ``sweep_max_s`` the sweep frame payload; see the
        class docstring for each.

    Raises:
        ValueError: No geometry, invalid or unsupported motion, or motion without
            explicit bounds. Flexures cannot be rendered by this rigid-body path.
        software_defined_matter.sdf.bbox.BBoxInferenceError: Geometry is
            unbounded and no ``metadata["bbox"]`` override is set.
    """

    def _stage(name: str) -> None:
        # on_stage is optional progress reporting. Suppress callback exceptions
        # so a host bug cannot propagate and abort emit_part_to_glsl.
        if on_stage is not None:
            with contextlib.suppress(Exception):
                on_stage(name)

    motion = None
    motion_bounds = None
    motion_regions: list[SDFTree] = []
    if tree is None and part.kinematics:
        from software_defined_matter.kinematics import compile_kinematics, resolve_region_tree

        try:
            if part.kinematics.get("flexures"):
                raise NotImplementedError(
                    "Flexure interpolation is not supported by the rigid-body evaluator "
                    "used for shader emission"
                )
            motion = compile_kinematics(part)
        except NotImplementedError as exc:
            # Consumer conformance uses ValueError for refused emissions.
            raise ValueError(f"Cannot emit authored body motion: {exc}") from exc
        assert motion is not None
        if motion.region_names:
            from software_defined_matter.motion_bounds import infer_motion_bounds

            motion_bounds = infer_motion_bounds(part, smooth_csg=smooth_csg)
            motion_regions = [
                resolve_region_tree(part, body["region"]) for body in part.kinematics["bodies"]
            ]
            tree = {"type": "op", "op": "union", "children": motion_regions}
    if tree is None:
        _stage("envelope")
        tree = part.computed_envelope()
        if tree is None:
            raise ValueError(
                f"Part {part.name!r} has no materials; cannot derive an SDF tree to emit."
            )
    if smooth_csg is None:
        smooth_csg = bool(part.metadata.get("smooth_csg", False))

    # Refresh relation caches before the walk so uniform initials and the
    # controls manifest carry evaluated numbers, not stale authoring snapshots.
    part.refresh_derived()

    if poly_lod is not None and poly_lod < 3:
        raise ValueError(f"poly_lod must be >= 3 (a polygon), got {poly_lod}")

    _stage("glsl-walk")
    emitter = _GLSLEmitter(part, smooth_csg=smooth_csg, smooth_k=smooth_k, poly_lod=poly_lod)
    # Pre-scan for the largest polygon_2d so every emitted V[] can be sized to
    # a single SDM_POLY_MAX_N that also types sdf_polygon_2d's array parameter.
    # LOD-aware, so the shared array is sized to what actually gets emitted.
    emitter.poly_max_n = _max_polygon_verts(tree, poly_lod)
    # Decided from the WHOLE tree before anything is walked, so every
    # outline in one emission makes the same choice and the emitted
    # source has one storage form rather than two.
    emitter.poly_table_on = _total_polygon_verts(tree, poly_lod) > _POLY_TABLE_THRESHOLD
    # Sweep frames make the same choice on the same budget, from their own
    # count: a coil winding is thousands of vec3 literals with not one
    # polygon in sight.
    emitter.sweep_table_on = _total_sweep_points(tree) > _POLY_TABLE_THRESHOLD

    # Decide whether to split the tree before emitting any nodes.
    #
    # For a split root, the entry points below use the component functions.
    # `sdf_scene` calls `sdf_scene_rcut`, so no entry point needs a function
    # for the complete root tree. Emitting the root first would add unused
    # GLSL for the entire tree. That unused code accounted for 46% of the GLSL
    # in a downstream consumer's test fixtures, 48.9% of radial_bearing's
    # 133 KB, and 49.8% of banana's 412 KB.
    #
    # Removing it does not change the result. It keeps a WebGL2 consumer's
    # grid bake below its `scene_glsl.length > 150000` limit and reduces a
    # whole-scene program that takes seconds to compile on each rebuild. The
    # WebGPU transpiler already removes unused functions, so this only reduces
    # GLSL generation time, payload size, and cache use for that backend.
    #
    comp_fns: list[str] = []
    comp_labels: list[str] = []
    comp_nodes: list[SDFTree] = []
    motion_uniforms: list[UniformDecl] = []
    motion_controls: list[dict[str, Any]] = []
    motion_rest: list[str] = []
    if motion_regions:
        comp_fns, motion_rest, motion_uniforms, motion_controls = _emit_body_motion(
            emitter, motion_regions
        )
        assert motion is not None
        comp_labels = list(motion.region_names)
        comp_nodes = motion_regions
    elif _segmentable(tree, smooth_csg=smooth_csg):
        labels = list(part.metadata.get("components") or [])
        for node, label in _expand_components(tree, labels):
            component_fn = emitter.emit_node(node, dim=3)
            if component_fn in comp_fns:
                # Visibility masks address call sites by component function.
                # Shared bodies still need a distinct root for each component.
                alias = emitter._fn_name(3)
                emitter._emit_function(f"float {alias}(vec3 p)", [f"return {component_fn}(p);"])
                component_fn = alias
            comp_fns.append(component_fn)
            comp_labels.append(label)
            comp_nodes.append(node)

    # The root fold is reachable only in the UNSEGMENTED scene, where
    # `_scene_component_fns` calls it directly. Emitted there and nowhere else.
    # `comp_fns` rather than `_segmentable` is the condition, because an
    # expansion that yields nothing has to fall back to the root.
    root_fn = "" if comp_fns else emitter.emit_node(tree, dim=3)

    # Material-space rest maps, one per component that carries an animated
    # point-warp prefix. Emitted HERE, before `emitter.functions` is spliced
    # into the scene below -- `emit_rest_fn` appends to that list.
    rest_fns: list[str | None]
    if motion_regions:
        rest_fns = list(motion_rest)
    elif comp_fns:
        rest_fns = [_emit_rest_prefix(emitter, node) for node in comp_nodes]
    else:
        rest_fns = [_emit_rest_prefix(emitter, tree)]

    _stage("bbox")
    if motion_bounds is not None:
        assert motion_bounds.bbox is not None
        bbox = motion_bounds.bbox
        comp_boxes = [body.bbox for body in motion_bounds.bodies]
    else:
        bbox = _resolve_bbox(part, tree)
        comp_boxes = _component_bboxes(comp_nodes, part)

    lib_source, grid_table, grid_tex_width = _emitter_resources(emitter)

    scene_lines = [
        "// Generated by software_defined_matter.glsl.emit",
        f"// part: {part.name}",
        f"// uniforms: {[u.name for u in emitter.uniforms.values()]}",
        "",
    ]
    all_uniforms = [*emitter.uniforms.values(), *motion_uniforms]
    scene_lines.extend(_uniform_declarations(all_uniforms))
    scene_lines.append("")
    scene_lines.extend(emitter.functions)
    scene_lines.append("")
    scene_lines.extend(_scene_component_fns(root_fn, comp_fns, rest_fns))
    # sdf_scene routes through sdf_scene_rcut with the cut plane at infinity,
    # not the root fold directly. The distance is the same on occupied points,
    # but the call graph is not. Hosts hide components by rewriting
    # per-component call sites inside sdf_scene_rcut; sdf_scene must share that
    # path so ray marching and picking both see the masked field.
    scene_lines.append(
        "float sdf_scene(vec3 p) {\n    return sdf_scene_rcut(p, vec3(0.0, 0.0, 1.0), 3.0e38);\n}"
    )

    components: list[dict[str, Any]] = []
    for i, comp_fn in enumerate(comp_fns):
        box = comp_boxes[i]
        components.append(
            {
                "id": i,
                "machine": comp_fn,
                "label": comp_labels[i],
                "bbox": [list(box[0]), list(box[1])] if box is not None else None,
            }
        )
    scene_source = "\n\n".join(scene_lines).rstrip() + "\n"

    return GLSLEmission(
        scene_source=scene_source,
        lib_source=lib_source,
        uniforms=all_uniforms,
        bbox=bbox,
        smooth_csg=smooth_csg,
        smooth_k=smooth_k,
        controls=_build_controls(part, emitter.uniforms) + motion_controls,
        poly_max_n=emitter.poly_max_n,
        poly_lod=poly_lod,
        poly_table=emitter.poly_table,
        poly_tex_width=_POLY_TEX_WIDTH if emitter.poly_table else 0,
        sweep_table=emitter.sweep_table,
        sweep_tex_width=_SWEEP_TEX_WIDTH if emitter.sweep_table else 0,
        sweep_max_s=emitter.sweep_max_s,
        grid_table=grid_table,
        grid_tex_width=grid_tex_width,
        components=components,
    )


# The width of the selection mask, the colour table and the visibility mask a
# per component. Expansion is SKIPPED, not truncated, past this: a
# partially expanded tree would number its components differently from the one
# the host walks.
_MAX_COMPONENTS = 64


def _expand_components(tree: SDFTree, labels: list[str]) -> list[tuple[SDFTree, str]]:
    """(node, label) pairs for segmentation: the root union's children, expanded
    ONE level into any child that is itself a union.

    Transform and deform wrappers are pointwise reshuffles of the query point,
    so they distribute over ``min``: ``T(union(a, b)) == union(T(a), T(b))``.
    Each grandchild is therefore re-wrapped in a copy of the same wrapper chain
    and becomes its own component.

    Modifiers do NOT distribute (``onion(union(a, b))`` is not
    ``union(onion(a), onion(b))``), so they stop the descent, and the wrapped
    subtree stays a single component.

    Args:
        tree: The root node. Only a hard ``union`` is expanded; the caller
            checks that.
        labels: Author labels for the root's children, positionally. Missing
            entries fall back to ``component_<i>``.

    Returns:
        list[tuple[SDFTree, str]]: One (node, label) pair per component, in the
        order a host must number them. Sub-labels extend the parent's, e.g.
        ``pivot_0/3``.
    """
    out: list[tuple[SDFTree, str]] = []
    children = tree.get("children", [])
    for i, child in enumerate(children):
        label = labels[i] if i < len(labels) else f"component_{i}"
        wrappers: list[dict[str, Any]] = []
        node = child
        while (
            isinstance(node, dict)
            and node.get("type") in ("transform", "deform")
            and node.get("child") is not None
        ):
            wrappers.append(node)
            node = node["child"]
        # `len(children) - i - 1` is the children not yet visited: the budget
        # has to hold for the WHOLE expansion, so a union late in the list
        # cannot spend slots the remaining siblings still need.
        if (
            isinstance(node, dict)
            and node.get("type") == "op"
            and node.get("op") == "union"
            and node.get("children")
            and len(out) + len(node["children"]) + (len(children) - i - 1) <= _MAX_COMPONENTS
        ):
            for j, grandchild in enumerate(node["children"]):
                rewrapped = grandchild
                for w in reversed(wrappers):
                    rewrapped = dict(w, child=rewrapped)
                out.append((rewrapped, f"{label}/{j}"))
        else:
            out.append((child, label))
    return out


def _component_bboxes(nodes: list[SDFTree], part: Part) -> list[BBox | None]:
    """One inferred AABB per component, or None where inference cannot answer.

    A host culls and packs with these, and gates whole features on every
    component having one, so `None` has to stay expressible rather than be
    faked from the part box. An inflated box is not a conservative fallback
    here: a cull is only sound while the box CONTAINS the component, and
    substituting the part's box would claim geometry occupies space it does not.

    Inference is per node and raises on unbounded geometry (a bare half-space
    has no finite box). That is a normal answer, not an error.

    Args:
        nodes: The component subtrees, in id order.
        part: The part, for param resolution.

    Returns:
        list[BBox | None]: Positional, one entry per node.
    """
    out: list[BBox | None] = []
    for node in nodes:
        try:
            out.append(infer_sdf_bbox(node, part))
        except Exception:  # noqa: BLE001 -- unbounded is an answer, see docstring
            out.append(None)
    return out


def _segmentable(tree: SDFTree, *, smooth_csg: bool) -> bool:
    """Whether ``tree``'s root is a union this emitter will segment.

    `smooth_csg` disqualifies it: a smooth union BLENDS its children, so no
    point belongs to exactly one of them and "which component owns p" has no
    answer to give.

    Args:
        tree: The root node.
        smooth_csg: Whether hard CSG was promoted to smooth.

    Returns:
        bool: True when the root is a hard union of more than one child.
    """
    return (
        isinstance(tree, dict)
        and tree.get("type") == "op"
        and tree.get("op") == "union"
        and not smooth_csg
        and len(tree.get("children", [])) > 1
    )


def _build_controls(part: Part, uniforms: dict[str, UniformDecl]) -> list[dict[str, Any]]:
    """Classify every ``Part.params`` entry by what editing it costs.

    See :class:`GLSLEmission` for what each class means. Order follows
    ``part.params`` insertion order; viewers re-sort by ``ui.group`` /
    ``ui.order`` when those are present.

    Args:
        part: The part whose params are being classified.
        uniforms: The emitter's uniform table, keyed by source param name.
            Membership is the whole ``live`` test: a param is live exactly
            when the walk bound a uniform for it, which is a fact about THIS
            emission and not about how the param was authored.

    Returns:
        list[dict[str, Any]]: One entry per param, in declaration order.
    """
    from software_defined_matter.dsl.expr import expr_param_names

    controls: list[dict[str, Any]] = []
    for name, prm in part.params.items():
        ui = prm.ui or {}
        if prm.expr is not None:
            # BEFORE the uniform test, and never 're-emit'. A derived param is
            # not a uniform (the emitter expands its relation inline instead of
            # binding one), and classifying it 're-emit' would re-run the
            # authoring script on every scrub of a base param — for a value the
            # GPU is already computing per pixel.
            cls = "derived"
        elif name in uniforms:
            cls = "live"
        elif ui.get("role") == "topology":
            cls = "topology"
        else:
            cls = "re-emit"
        entry: dict[str, Any] = {
            "param": name,
            "class": cls,
            "value": float(prm.numeric_value()),
            "free": prm.free,
            "unit": prm.unit,
            "bounds": list(prm.bounds) if prm.bounds is not None else None,
        }
        if cls == "live":
            # The uniform's GLSL name, so a viewer can write the value without
            # re-deriving the mangling rule. It is the emitter's to spell.
            entry["uniform"] = uniforms[name].name
        if cls == "derived":
            entry["expr"] = copy.deepcopy(prm.expr)
            # IMMEDIATE dependencies, not the transitive base set: a source that
            # is itself derived carries its own relation in its own control
            # entry, so a viewer resolves the chain compositionally.
            entry["sources"] = sorted(expr_param_names(prm.expr))
        if ui:
            # Copied, not referenced: a caller that mutates the control must not
            # reach back into the Part it was built from.
            entry["ui"] = dict(ui)
        controls.append(entry)
    return controls


#: Deforms that are pure query-point WARPS, i.e. an animated material motion.
#: ``displace`` is deliberately absent: it perturbs the DISTANCE, not the point,
#: so it moves no material and has no place in a rest-point chain.
_POINT_WARP_DEFORMS = (
    "twist",
    "bend",
    "twist_radial",
    "twist_linear",
    "shear_linear",
    "taper_linear",
)


def _deform_prefix(node: Any) -> list[dict[str, Any]]:
    """The consecutive point-warp deform wrappers at the top of a component.

    That prefix IS the component's animated motion; everything below it
    (transforms, folds, primitives) is static placement. Material space is the
    query after pulling through exactly this prefix, and no further.

    Args:
        node: A component's root node.

    Returns:
        list[dict]: The wrappers, outermost first. Empty for static geometry.
    """
    chain: list[dict[str, Any]] = []
    n = node
    while (
        isinstance(n, dict)
        and n.get("type") == "deform"
        and n.get("deform") in _POINT_WARP_DEFORMS
        and n.get("child") is not None
    ):
        chain.append(n)
        n = n["child"]
    return chain


def _emit_rest_prefix(emitter: _GLSLEmitter, node: Any) -> str | None:
    """The rest-map function for ``node``, or None when it needs none.

    None rather than an emitted identity, so a static component costs no
    function and its call sites stay literally ``p`` -- which is what makes the
    static case byte-identical to a build with no rest maps at all.
    """
    chain = _deform_prefix(node)
    return emitter.emit_rest_fn(chain) if chain else None


def _scene_component_fns(
    root_fn: str, comp_fns: list[str], rest_fns: list[str | None]
) -> list[str]:
    """`sdf_scene_comp` and `sdf_scene_rcut`, the two per-component entry points.

    ``sdf_scene_comp(p)`` answers WHICH component owns ``p`` (the argmin over
    the component fields) and is what a host indexes its colour table,
    selection mask and visibility mask with. It returns 0 for an unsegmented
    scene, which is correct there: the whole part IS component 0.

    ``sdf_scene_rcut(p, cn, co)`` is the field intersected against a cut plane,
    per component. It is a separate entry point rather than a flag on
    ``sdf_scene`` because a host rewrites call sites in it: masking a hidden
    component out has to happen inside the emitted field, or a pick through the
    hidden space answers with geometry that is no longer drawn.

    THE ``r`` IS "REST", AND IT IS LOAD-BEARING. Each component meets the plane
    at its REST point (the query pulled back through its animated deform prefix),
    so the section is taken once in material coordinates and then travels with
    the material, instead of re-slicing deformed geometry against a stationary
    world plane every frame. A component with no such prefix cuts at ``p``
    directly, which is the same thing for static geometry, so a part with no
    deforms emits exactly what it did before rest maps existed.

    ``sdm_rest_point(p, cid)`` exposes those same coordinates to the host. Its
    reason to exist is that a monochrome surface which twists has no feature to
    track: a pattern painted in world space stands still while the body moves
    through it, and one painted in rest space rides along and makes the motion
    visible. It answers ``p`` for any component with no prefix, and for any
    ``cid`` it does not recognise.

    Args:
        root_fn: The emitted root field function.
        comp_fns: One emitted function per component, in id order. Empty for an
            unsegmented scene.
        rest_fns: One rest map per entry in ``comp_fns``, or a single-element
            list for the unsegmented case. ``None`` where the component is
            static and cuts at ``p``.

    Returns:
        list[str]: GLSL function definitions, in emission order.
    """
    out: list[str] = []
    if comp_fns:
        body = ["    float d = 1e30;", "    int id = 0;", "    float dk;"]
        for i, fn in enumerate(comp_fns):
            body.append(f"    dk = {fn}(p);")
            body.append(f"    if (dk < d) {{ d = dk; id = {i}; }}")
        body.append("    return id;")
        out.append("int sdf_scene_comp(vec3 p) {\n" + "\n".join(body) + "\n}")

        cut = ["    float d = 1e30;"]
        for fn, rest in zip(comp_fns, rest_fns, strict=True):
            q = f"{rest}(p)" if rest else "p"
            cut.append(f"    d = op_union(d, max({fn}(p), sdm_cut_plane({q}, cn, co)));")
        cut.append("    return d;")
        out.append("float sdf_scene_rcut(vec3 p, vec3 cn, float co) {\n" + "\n".join(cut) + "\n}")

        # One guard per component that HAS a map. A component without one falls
        # through to `return p`, which is its material space.
        guards = [
            f"    if (cid == {i}) return {fn}(p);"
            for i, fn in enumerate(rest_fns)
            if fn is not None
        ]
        out.append(
            "vec3 sdm_rest_point(vec3 p, int cid) {\n"
            + "".join(f"{g}\n" for g in guards)
            + "    return p;\n}"
        )
    else:
        rest = rest_fns[0] if rest_fns else None
        q = f"{rest}(p)" if rest else "p"
        out.append("int sdf_scene_comp(vec3 p) { return 0; }")
        out.append(
            "float sdf_scene_rcut(vec3 p, vec3 cn, float co) {\n"
            f"    return max({root_fn}(p), sdm_cut_plane({q}, cn, co));\n}}"
        )
        # `cid` is unused here by construction: an unsegmented scene is one
        # component, so every id maps through the same chain.
        out.append(f"vec3 sdm_rest_point(vec3 p, int cid) {{\n    return {q};\n}}")
    return out


def _uniform_declarations(uniforms: Iterable[UniformDecl]) -> list[str]:
    lines: list[str] = []
    for u in uniforms:
        lines.append(
            f"uniform {u.glsl_type} {u.name};  // {u.source_param}"
            + (f" [{u.bounds[0]}, {u.bounds[1]}]" if u.bounds else "")
            + (f" {u.unit}" if u.unit else "")
        )
    return lines


def _resolve_bbox(part: Part, tree: SDFTree) -> BBox:
    """Honour ``part.metadata['bbox']`` if present, else infer."""
    meta_bbox = part.metadata.get("bbox")
    if meta_bbox is not None:
        lo, hi = meta_bbox
        lx, ly, lz = (float(x) for x in lo)
        hx, hy, hz = (float(x) for x in hi)
        return ((lx, ly, lz), (hx, hy, hz))
    return infer_sdf_bbox(tree, part)


__all__ = [
    "GLSLEmission",
    "UniformDecl",
    "emit_glsl",
    "load_lib_glsl",
]
