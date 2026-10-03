"""Boolean posed-material queries for shaders; these are not marching fields."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from software_defined_matter.glsl.emit import (
    _POLY_TABLE_THRESHOLD,
    _POLY_TEX_WIDTH,
    UniformDecl,
    _build_controls,
    _emit_body_motion,
    _emitter_resources,
    _float_literal,
    _GLSLEmitter,
    _max_polygon_verts,
    _motion_expressions,
    _MotionExpressionEmitter,
    _total_polygon_verts,
    _uniform_declarations,
)
from software_defined_matter.model import Part

__all__ = ["GLSLMaterialMembership", "emit_material_membership"]


@dataclass(frozen=True)
class GLSLMaterialMembership:
    """Shader point-membership contract, deliberately separate from GLSLEmission.

    ``sdm_material_member(p, region, material)`` returns a bool for zero-based
    region and material-record indices; invalid indices return false.
    ``sdm_material_contains(p)`` returns their union. ``sdm_material_rest`` gives
    a candidate rest point for a valid region index, not an ownership claim.
    Region order is bodies then flexures; repeated material IDs remain separate
    records. DOF uniforms use authored units, including degrees. Design edits
    may change geometry and classifier ownership through the listed uniforms.
    ``material_functions`` and ``region_functions`` name scalar rest-space
    field functions in document order, for consumers building bounded queries.
    Polygon and raster textures use the ordinary emitter's resource contract.

    No function supplies a signed distance, safe ray step, normal or surface.
    These queries do not enable flexures in the normal SDF renderer.
    """

    scene_source: str
    lib_source: str
    uniforms: list[UniformDecl]
    controls: list[dict[str, Any]]
    region_names: tuple[str, ...]
    material_ids: tuple[int, ...]
    poly_max_n: int
    poly_table: list[float]
    poly_tex_width: int
    grid_table: list[float]
    grid_tex_width: int
    entry_point: str = "sdm_material_contains"
    material_functions: tuple[str, ...] = ()
    region_functions: tuple[str, ...] = ()


def _vec(values: Any) -> str:
    return "vec3(" + ", ".join(_float_literal(v) for v in values) + ")"


def _flexure_inverse(
    emitter: _GLSLEmitter,
    expressions: _MotionExpressionEmitter,
    flexure: dict[str, Any],
    index: int,
) -> str:
    """Emit the invariant single-axis subset already checked by the CPU compiler."""
    block = emitter.part.kinematics
    assert block is not None
    bodies = {b["name"]: b["motion"]["ops"] for b in block["bodies"]}
    first, second = bodies[flexure["from_body"]], bodies[flexure["to_body"]]
    name = f"sdm_flexure_rest_{index}"
    if not first and not second:
        emitter._emit_function(f"vec3 {name}(vec3 p)", ["return p;"])
        return name
    template = (first or second)[0]
    axis = [x / math.hypot(*template["axis"]) for x in template["axis"]]
    angles = []
    for ops in (first, second):
        if not ops:
            angles.append("0.0")
        else:
            op = ops[0]
            sign = 1 if sum(a * b for a, b in zip(axis, op["axis"], strict=True)) > 0 else -1
            angles.append(f"({_float_literal(sign)} * ({expressions._emit_expr(op['angle'])}))")
    blend = flexure["blend"]
    params = blend["params"]
    if blend["kind"] == "axis_ramp":
        lines = [
            f"float w = clamp((dot(p, {_vec(params['axis'])}) - {_float_literal(params['lo'])})"
            f" / {_float_literal(params['hi'] - params['lo'])}, 0.0, 1.0);"
        ]
    else:
        blend_axis = [x / math.hypot(*params["axis"]) for x in params["axis"]]
        lines = [
            f"vec3 delta = p - {_vec(params['origin'])};",
            f"float rho = length(delta - {_vec(blend_axis)} * dot({_vec(blend_axis)}, delta));",
            f"float t = clamp((rho - {_float_literal(params['r0'])})"
            f" / {_float_literal(params['r1'] - params['r0'])}, 0.0, 1.0);",
            "float w = t * t * (3.0 - 2.0 * t);",
        ]
    origin, axis_source = _vec(template.get("origin", [0, 0, 0])), _vec(axis)
    lines += [
        f"float a = mix({angles[0]}, {angles[1]}, w);",
        f"vec3 q = p - {origin};",
        f"return {origin} + cos(a) * q - sin(a) * cross({axis_source}, q)"
        f" + (1.0 - cos(a)) * {axis_source} * dot({axis_source}, q);",
    ]
    emitter._emit_function(f"vec3 {name}(vec3 p)", lines)
    return name


def emit_material_membership(part: Part) -> GLSLMaterialMembership | None:
    """Emit exact Boolean membership semantics for bodies and supported flexures.

    Return None without motion regions. Unsupported inverses fail before emission.
    Material trees define solids; region trees only classify rest points by
    minimum distance, with first-declared ties. As in the CPU compiler, the
    document's smooth_csg metadata applies to materials and classifiers.
    CPU and GLSL arithmetic can disagree at floating-point boundary tolerances.
    """
    from software_defined_matter.kinematics import resolve_region_tree
    from software_defined_matter.material_motion import compile_material_motion

    snapshot = copy.deepcopy(part)
    reference = compile_material_motion(snapshot)
    if reference is None:
        return None
    snapshot.refresh_derived()
    block = snapshot.kinematics
    assert block is not None
    bodies, flexures = block["bodies"], block.get("flexures", [])
    regions = [resolve_region_tree(snapshot, r["region"]) for r in [*bodies, *flexures]]
    trees = [m.sdf_tree for m in snapshot.materials]
    smooth_csg = bool(snapshot.metadata.get("smooth_csg", False))
    emitter = _GLSLEmitter(snapshot, smooth_csg=smooth_csg, smooth_k=0.25)
    emitter.poly_max_n = _max_polygon_verts([*trees, *regions], None)
    emitter.poly_table_on = _total_polygon_verts([*trees, *regions]) > _POLY_TABLE_THRESHOLD
    materials = [emitter.emit_node(t, dim=3) for t in trees]
    classifiers = [emitter.emit_node(t, dim=3) for t in regions]
    motion = _emit_body_motion(emitter, regions[: len(bodies)])
    expressions = _motion_expressions(emitter)
    rest = motion.rest_functions + [
        _flexure_inverse(emitter, expressions, f, i) for i, f in enumerate(flexures)
    ]
    emitter._emit_function(
        "vec3 sdm_material_rest(vec3 p, int region)",
        [f"if (region == {i}) return {fn}(p);" for i, fn in enumerate(rest)] + ["return p;"],
    )
    lines = ["int owner = 0;", f"float best = {classifiers[0]}(q);", "if (isnan(best)) return -1;"]
    for i, fn in enumerate(classifiers[1:], 1):
        lines += [
            f"float d{i} = {fn}(q);",
            f"if (isnan(d{i})) return -1;",
            f"if (d{i} < best) {{ best = d{i}; owner = {i}; }}",
        ]
    emitter._emit_function("int sdm_material_rest_owner(vec3 q)", [*lines, "return owner;"])
    emitter._emit_function(
        "bool sdm_material_member(vec3 p, int region, int material)",
        [
            f"if (region < 0 || region >= {len(rest)} || material < 0"
            f" || material >= {len(materials)}) return false;",
            "vec3 q = sdm_material_rest(p, region);",
            "if (any(isnan(q)) || any(isinf(q))) return false;",
            "if (sdm_material_rest_owner(q) != region) return false;",
            *[f"if (material == {i}) return {fn}(q) <= 0.0;" for i, fn in enumerate(materials)],
            "return false;",
        ],
    )
    emitter._emit_function(
        "bool sdm_material_contains(vec3 p)",
        [
            f"if (sdm_material_member(p, {i}, {j})) return true;"
            for i in range(len(rest))
            for j in range(len(materials))
        ]
        + ["return false;"],
    )
    lib, grid, grid_width = _emitter_resources(emitter)
    uniforms = list(emitter.uniforms.values()) + motion.uniforms
    return GLSLMaterialMembership(
        "\n".join([*_uniform_declarations(uniforms), *emitter.functions]),
        lib,
        uniforms,
        _build_controls(snapshot, emitter.uniforms) + motion.controls,
        reference.region_names,
        reference.material_ids,
        emitter.poly_max_n,
        emitter.poly_table,
        _POLY_TEX_WIDTH if emitter.poly_table else 0,
        grid,
        grid_width,
        material_functions=tuple(materials),
        region_functions=tuple(classifiers),
    )
