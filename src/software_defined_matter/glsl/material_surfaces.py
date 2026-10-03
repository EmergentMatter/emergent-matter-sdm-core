"""Shader surface queries with host-prepared, atomic pose and rate updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from software_defined_matter.glsl.emit import UniformDecl, _float_literal
from software_defined_matter.glsl.material_motion import emit_material_membership
from software_defined_matter.material_surfaces import MaterialSurfaces, compile_material_surfaces
from software_defined_matter.model import Part
from software_defined_matter.sdf.bbox import BBox

__all__ = ["GLSLMaterialSurfaces", "emit_material_surfaces"]


@dataclass(frozen=True)
class GLSLMaterialSurfaces:
    """Separate surface-query shader contract; this is not a GLSLEmission.

    sdm_surface_trace(ro, rd, near, far, tolerance, max_steps) returns
    vec4(bracket_lo, bracket_hi, status, work_count). Status is 1 for a proved
    occupancy change, 0 for a proved miss, -1 unresolved, -2 invalid input.
    Distances use normalized rd and world units. Only status 1 is a hit; -1
    must remain distinguishable from empty space. sdm_surface_normal(p) gives
    an estimated world normal, or zero when the gradient is unavailable.

    sdm_surface_trace_contacts takes the same arguments plus contact_radius.
    Zero preserves the crossing-only API. Positive radii enable status 2: a
    witnessed boundary within that world-space distance of a ray point. For
    status 2 only, the fourth component is contact_t rather than a work count;
    the first two components retain the earliest uncertain ray window. This
    is not proof of an exact tangency or an exact ray-intersection bracket.
    sdm_surface_contact_witness(ro + contact_t*normalize(rd), contact_radius)
    returns vec4(outward_probe_direction, 1), or zero without a witness.
    Subtracting/adding radius times that direction reconstructs the inside and
    outside endpoints. Both lie in the query domain and clear the field-error
    allowance. The normal estimator remains separate from this probe direction.

    Bind all values from pose_uniforms together after every DOF edit. Updating
    DOF uniforms alone invalidates the rate and grouping contract. Geometry,
    design inputs, domain and field_error are frozen: re-emit after edits.
    The packet includes rest-field rates and ramp slopes for ray-specific
    bounds. No shader rebuild or static-geometry rebake is needed for a pose update.
    Normal estimates and field-error limitations match the CPU API.
    """

    scene_source: str
    lib_source: str
    uniforms: list[UniformDecl]
    poly_table: list[float]
    poly_tex_width: int
    grid_table: list[float]
    grid_tex_width: int
    surface: MaterialSurfaces

    def pose_uniforms(self, dofs: Any) -> dict[str, float]:
        """Prepare a complete canonical radians/mm pose update for this shader."""
        pose = self.surface.prepare(dofs)
        result = {
            f"u_dof_{i}": math.degrees(value) if unit == "deg" else value
            for i, (value, unit) in enumerate(
                zip(pose.dofs, self.surface.membership.kinematics.dof_units, strict=True)
            )
        }
        result["u_surface_rate"] = pose.max_rate
        result["u_surface_rest_rate"] = pose.rest_rate
        result.update(
            {f"u_surface_slope_{i}": term.slope for i, term in enumerate(pose._ray_terms)}
        )
        result.update({f"u_surface_group_{i}": float(g) for i, g in enumerate(pose.groups)})
        return result


def emit_material_surfaces(
    part: Part, *, domain: BBox, field_error: float = 1e-5
) -> GLSLMaterialSurfaces:
    """Emit bounded CPU-parity ray queries; normal whole-part emission is unchanged."""
    surface = compile_material_surfaces(part, domain=domain, field_error=field_error)
    membership = emit_material_membership(part)
    assert membership is not None
    count = len(membership.region_names)
    lines = ["uniform float u_surface_rate;"]
    lines.extend(f"uniform float u_surface_group_{i};" for i in range(count))
    lines += ["float sdm_surface_field(vec3 p) {", "float result = 1.0 / 0.0;"]
    if surface.ownership_invariant:
        lines.append("float coverage = -1.0 / 0.0;")
    for i in range(count):
        lines += [
            f"if (int(u_surface_group_{i}) == {i}) {{",
            f"vec3 q = sdm_material_rest(p, {i});",
            "float material = 1.0 / 0.0;",
            "float inside = 1.0 / 0.0;",
            "float outside = 1.0 / 0.0;",
        ]
        lines.extend(f"material = min(material, {fn}(q));" for fn in membership.material_functions)
        for j, fn in enumerate(membership.region_functions):
            lines += [
                f"if (int(u_surface_group_{j}) == {i}) inside = min(inside, {fn}(q));",
                f"else outside = min(outside, {fn}(q));",
            ]
        lines.append("result = min(result, max(material, inside - outside));")
        if surface.ownership_invariant:
            lines.append("coverage = max(coverage, min(material, outside - inside));")
        lines.append("}")
    if surface.ownership_invariant:
        lines.append(
            "result = result + coverage < 0.0 ? min(result, coverage) : max(result, coverage);"
        )
    lines += ["return result;", "}"]
    initial_pose = surface.prepare(surface.membership.kinematics.dof_defaults)
    lines.append("uniform float u_surface_rest_rate;")
    for i in range(len(initial_pose._ray_terms)):
        lines.append(f"uniform float u_surface_slope_{i};")
    lines += [
        "float sdm_surface_ray_rate(vec3 a, vec3 b, vec3 direction) {",
        "float stretch = 1.0;",
    ]
    for i, term in enumerate(initial_pose._ray_terms):
        axis = "vec3(" + ",".join(_float_literal(v) for v in term.axis) + ")"
        blend_axis = "vec3(" + ",".join(_float_literal(v) for v in term.blend_axis) + ")"
        pivot = "vec3(" + ",".join(_float_literal(v) for v in term.pivot) + ")"
        lines += [
            "{",
            f"float radius = max(length(cross(a-{pivot}, {axis})),"
            f" length(cross(b-{pivot}, {axis})));",
            f"float projection = dot(direction, {blend_axis});",
        ]
        variation = (
            f"length(direction - projection*{blend_axis})" if term.radial else "abs(projection)"
        )
        lines += [f"stretch = max(stretch, 1.0 + u_surface_slope_{i} * radius * {variation});", "}"]
    lines += ["return u_surface_rest_rate * stretch;", "}"]
    low = "vec3(" + ",".join(_float_literal(v) for v in surface.domain[0]) + ")"
    high = "vec3(" + ",".join(_float_literal(v) for v in surface.domain[1]) + ")"
    lines.append(
        _TRACE.replace("DOMAIN_LOW", low)
        .replace("DOMAIN_HIGH", high)
        .replace("FIELD_ERROR", _float_literal(field_error))
    )
    emission = GLSLMaterialSurfaces(
        membership.scene_source + "\n" + "\n".join(lines),
        membership.lib_source,
        list(membership.uniforms),
        membership.poly_table,
        membership.poly_tex_width,
        membership.grid_table,
        membership.grid_tex_width,
        surface,
    )
    packet = emission.pose_uniforms(surface.membership.kinematics.dof_defaults)
    for name, value in packet.items():
        if name.startswith("u_surface_"):
            emission.uniforms.append(UniformDecl(name, "float", value, None, "ratio", name))
    return emission


_TRACE = """
vec3 sdm_surface_normal(vec3 p) {
    float h = 0.0001;
    vec3 g = vec3(
        sdm_surface_field(p + vec3(h,0,0)) - sdm_surface_field(p - vec3(h,0,0)),
        sdm_surface_field(p + vec3(0,h,0)) - sdm_surface_field(p - vec3(0,h,0)),
        sdm_surface_field(p + vec3(0,0,h)) - sdm_surface_field(p - vec3(0,0,h)));
    if (any(isnan(g)) || any(isinf(g)) || length(g) == 0.0) return vec3(0);
    return normalize(g);
}
bool sdm_surface_in_domain(vec3 p) {
    return !any(isnan(p)) && !any(isinf(p)) &&
        all(greaterThanEqual(p, DOMAIN_LOW)) && all(lessThanEqual(p, DOMAIN_HIGH));
}
vec4 sdm_surface_contact_witness(vec3 p, float radius) {
    if (isnan(radius) || isinf(radius) || radius <= 0.0) return vec4(0);
    vec3 normal = sdm_surface_normal(p);
    for (int k = 0; k < 4; k++) {
        vec3 d = k == 0 ? normal : vec3(k == 1 ? 1.0 : 0.0,
                                      k == 2 ? 1.0 : 0.0, k == 3 ? 1.0 : 0.0);
        if (length(d) == 0.0) continue;
        vec3 a = p - radius*d; vec3 b = p + radius*d;
        if (!sdm_surface_in_domain(a) || !sdm_surface_in_domain(b)) continue;
        float fa = sdm_surface_field(a); float fb = sdm_surface_field(b);
        if (isnan(fa) || isinf(fa) || isnan(fb) || isinf(fb)) continue;
        bool ia = sdm_material_contains(a); bool ib = sdm_material_contains(b);
        if (ia && !ib && fa < -FIELD_ERROR && fb > FIELD_ERROR) return vec4(d, 1);
        if (ib && !ia && fb < -FIELD_ERROR && fa > FIELD_ERROR) return vec4(-d, 1);
    }
    return vec4(0);
}
vec4 sdm_surface_unresolved(float lo, float hi, float contact, int work) {
    return contact >= 0.0 ? vec4(lo, hi, 2, contact) : vec4(lo, hi, -1, work);
}
vec4 sdm_surface_trace_contacts(vec3 ro, vec3 rd, float near, float far,
                                float tol, int budget, float contact_radius) {
    float len = length(rd);
    if (any(isnan(ro)) || any(isinf(ro)) || isnan(len) || isinf(len) || len == 0.0 ||
        isnan(near) || isinf(near) || isnan(far) || isinf(far) ||
        isnan(tol) || isinf(tol) || near < 0.0 || far <= near || tol <= 0.0 ||
        budget < 1 || budget > 4096 || isnan(u_surface_rate) || isinf(u_surface_rate) ||
        u_surface_rate <= 0.0 || isnan(contact_radius) || isinf(contact_radius) ||
        contact_radius < 0.0) return vec4(near, far, -2, 0);
    rd /= len;
    if (!sdm_surface_in_domain(ro + near*rd) || !sdm_surface_in_domain(ro + far*rd))
        return vec4(near, far, -2, 0);
    float ray_rate = sdm_surface_ray_rate(ro + near*rd, ro + far*rd, rd);
    if (isnan(ray_rate) || isinf(ray_rate) || ray_rate <= 0.0) return vec4(near, far, -2, 0);
    bool initial = sdm_material_contains(ro + near*rd);
    float lows[64]; float highs[64];
    lows[0] = near; highs[0] = far; int size = 1;
    float pending = -1.0;
    float contact = -1.0;
    for (int work = 0; work < 4096; work++) {
        if (size == 0) return pending >= 0.0
            ? sdm_surface_unresolved(pending, min(far, pending+tol), contact, work)
            : vec4(near, far, 0, work);
        if (work >= budget) break;
        size--;
        float lo = lows[size]; float hi = highs[size]; float mid = (lo + hi)*0.5;
        if (pending >= 0.0 && lo - pending >= tol)
            return sdm_surface_unresolved(pending, min(hi, pending + tol), contact, work+1);
        if (pending >= 0.0) { hi = min(hi, pending + tol); mid = (lo+hi)*0.5; }
        float value = sdm_surface_field(ro + mid*rd);
        float allowance = ray_rate*(hi-lo)*0.5 + FIELD_ERROR;
        if (!isnan(value) && !isinf(value) &&
            (initial ? value < -allowance : value > allowance)) continue;
        if (hi-lo <= tol/(8.0*max(1.0, ray_rate))) {
            if (sdm_material_contains(ro + lo*rd) != initial ||
                sdm_material_contains(ro + mid*rd) != initial ||
                sdm_material_contains(ro + hi*rd) != initial)
                return vec4(pending >= 0.0 ? pending : lo, hi, 1, work+1);
            if (pending < 0.0) pending = lo;
            if (contact_radius > 0.0 && contact < 0.0 &&
                sdm_surface_contact_witness(ro + mid*rd, contact_radius).w > 0.0) contact = mid;
            continue;
        }
        if (size >= 63 || mid == lo || mid == hi)
            return sdm_surface_unresolved(pending >= 0.0 ? pending : lo, hi, contact, work+1);
        lows[size] = mid; highs[size] = hi; size++;
        lows[size] = lo; highs[size] = mid; size++;
    }
    if (pending >= 0.0)
        return sdm_surface_unresolved(pending, min(far, pending+tol), contact, budget);
    return size > 0 ? vec4(lows[size-1], highs[size-1], -1, budget) : vec4(near, far, 0, budget);
}
vec4 sdm_surface_trace(vec3 ro, vec3 rd, float near, float far, float tol, int budget) {
    return sdm_surface_trace_contacts(ro, rd, near, far, tol, budget, 0.0);
}
"""
