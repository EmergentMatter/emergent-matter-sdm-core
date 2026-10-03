// software_defined_matter/glsl/lib.glsl
//
// GLSL counterparts of the JAX SDF library. Each function is a direct
// translation of its Python sibling in
//   - software_defined_matter.sdf.sdf_shapes
//   - software_defined_matter.sdf.sdf_ops
//   - software_defined_matter.sdf.transforms
// Semantics must stay in lockstep with the Python so the parity test
// (JAX-on-CPU vs GLSL-on-headless-GL) passes within float32 tolerance.
//
// Conventions
// -----------
// * 3-D primitives operate on vec3, 2-D primitives on vec2.
// * Per-axis vectors that Python keeps as flat (D,) arrays (box half-extents,
//   capsule endpoints, TPMS n_periods, ...) become vec2 or vec3 here.
// * No precision qualifiers; the surrounding shader chooses (highp for
//   ray-march, mediump for normals).
// * Helpers _length / _dot2 / _clamp / _mix / _sign from the Python helpers
//   map to built-ins (length, dot, clamp, mix, sign).

#ifndef SDM_GLSL_LIB
#define SDM_GLSL_LIB

// Max polygon vertex count in the scene. GLSL array-typed function
// parameters need a compile-time size, so a single sdf_polygon_2d helper is
// sized to the largest polygon and each per-node V[] is padded up to it (the real
// vertex count is passed as `n`; padding entries are never read). The emitter
// overrides this by prepending a `#define SDM_POLY_MAX_N <max>` to lib_source;
// this fallback keeps the file self-compilable for tooling / editors.
#ifndef SDM_POLY_MAX_N
#define SDM_POLY_MAX_N 3
#endif


// ===========================================================================
// CSG boolean operations
// ===========================================================================

float op_union(float d1, float d2)     { return min(d1, d2); }
float op_subtract(float d1, float d2)  { return max(d1, -d2); }
float op_intersect(float d1, float d2) { return max(d1, d2); }

float op_smooth_union(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float op_smooth_subtract(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d1, -d2, h) + k * h * (1.0 - h);
}

float op_smooth_intersect(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}


// ===========================================================================
// Distance modifiers (operate on the scalar distance)
// ===========================================================================

float op_round(float d, float r)          { return d - r; }
float op_onion(float d, float thickness)  { return abs(d) - thickness; }


// ===========================================================================
// Coordinate transforms (operate on the input point)
// ===========================================================================

vec3 tf_translate3(vec3 p, vec3 t) { return p - t; }
vec2 tf_translate2(vec2 p, vec2 t) { return p - t; }

vec3 tf_scale3(vec3 p, float s) { return p / s; }
vec2 tf_scale2(vec2 p, float s) { return p / s; }

vec3 tf_rotate_x(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

vec3 tf_rotate_y(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

vec3 tf_rotate_z(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

vec3 tf_rotate_matrix(vec3 p, mat3 R) {
    // Python: p @ R.T  (row-vector p times R-transpose)
    //       == (R @ p^T)^T = R * p  when reading p as a column vector.
    return R * p;
}

vec3 op_repeat_finite3(vec3 p, float c, vec3 l) {
    return p - c * clamp(round(p / c), -l, l);
}

vec2 op_repeat_finite2(vec2 p, float c, vec2 l) {
    return p - c * clamp(round(p / c), -l, l);
}


// ===========================================================================
// op_elongate (modifier with point reshuffle)
// ===========================================================================
// Used as: child(op_elongate3(p, h)) in the generated scene.

vec3 op_elongate3(vec3 p, vec3 h) { return p - clamp(p, -h, h); }
vec2 op_elongate2(vec2 p, vec2 h) { return p - clamp(p, -h, h); }


// ===========================================================================
// Deformations (also point reshuffles; child(...) is called by the scene)
// ===========================================================================

vec3 op_twist(vec3 p, float k) {
    float c = cos(k * p.y);
    float s = sin(k * p.y);
    return vec3(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

vec3 op_bend(vec3 p, float k) {
    float c = cos(k * p.x);
    float s = sin(k * p.x);
    return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}


// ===========================================================================
// 3-D exact primitives  (Inigo Quilez)
// ===========================================================================

float sdf_sphere(vec3 p, float r) {
    return length(p) - r;
}

float sdf_box(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdf_round_box(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdf_box_frame(vec3 p, vec3 b, float e) {
    vec3 p2 = abs(p) - b;
    vec3 q = abs(p2 + e) - e;
    float d1 = length(max(vec3(p2.x, q.y, q.z), 0.0)) + min(max(p2.x, max(q.y, q.z)), 0.0);
    float d2 = length(max(vec3(q.x, p2.y, q.z), 0.0)) + min(max(q.x, max(p2.y, q.z)), 0.0);
    float d3 = length(max(vec3(q.x, q.y, p2.z), 0.0)) + min(max(q.x, max(q.y, p2.z)), 0.0);
    return min(d1, min(d2, d3));
}

float sdf_torus(vec3 p, vec2 t) {
    // t = [major_radius, minor_radius]
    vec2 q = vec2(length(p.xy) - t.x, p.z);
    return length(q) - t.y;
}

float sdf_capped_torus(vec3 p, vec2 sc, float ra, float rb) {
    // sc = [sin(angle), cos(angle)]
    float px = abs(p.x);
    float py = p.y;
    float pz = p.z;
    vec2 pxy = vec2(px, py);
    float k = (sc.y * px > sc.x * py) ? dot(pxy, sc) : length(pxy);
    vec3 p3 = vec3(px, py, pz);
    return sqrt(dot(p3, p3) + ra * ra - 2.0 * ra * k) - rb;
}

float sdf_link(vec3 p, float le, float r1, float r2) {
    vec3 q = vec3(p.x, max(abs(p.y) - le, 0.0), p.z);
    return length(vec2(length(q.xy) - r1, q.z)) - r2;
}

float sdf_cone(vec3 p, vec2 c, float h) {
    // c = [sin(angle), cos(angle)]
    vec2 q_vec = h * vec2(c.x / c.y, -1.0);
    vec2 w = vec2(length(p.xy), p.z);
    vec2 a = w - q_vec * clamp(dot(w, q_vec) / dot(q_vec, q_vec), 0.0, 1.0);
    vec2 b = w - q_vec * vec2(clamp(w.x / q_vec.x, 0.0, 1.0), 1.0);
    float k = sign(q_vec.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (w.x * q_vec.y - w.y * q_vec.x),
                  k * (w.y - q_vec.y));
    return sqrt(d) * sign(s);
}

float sdf_plane(vec3 p, vec3 n, float h) {
    // Python: sum(p * n, axis=-1) + h  ==  dot(p, n) + h
    return dot(p, n) + h;
}

float sdf_hex_prism(vec3 p, vec2 h) {
    // h = [hex_radius, half_height]
    vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    vec3 p2 = abs(p);
    vec2 pxy = p2.xy;
    pxy = pxy - 2.0 * min(dot(k.xy, pxy), 0.0) * k.xy;
    vec2 d = vec2(
        length(pxy - vec2(clamp(pxy.x, -k.z * h.x, k.z * h.x), h.x)) * sign(pxy.y - h.x),
        p2.z - h.y
    );
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdf_tri_prism(vec3 p, vec2 h) {
    // h = [triangle_radius, half_height]
    vec3 q = abs(p);
    return max(q.z - h.y,
               max(q.x * 0.866025 + p.y * 0.5, -p.y) - h.x * 0.5);
}

float sdf_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdf_capped_cylinder(vec3 p, float h, float r) {
    vec2 d = vec2(abs(length(p.xy)) - r, abs(p.z) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdf_rounded_cylinder(vec3 p, float ra, float rb, float h) {
    vec2 d = vec2(length(p.xy) - 2.0 * ra + rb, abs(p.z) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - rb;
}

float sdf_capped_cone(vec3 p, float h, float r1, float r2) {
    vec2 q = vec2(length(p.xy), p.z);
    vec2 k1 = vec2(r2, h);
    vec2 k2 = vec2(r2 - r1, 2.0 * h);
    vec2 ca = vec2(q.x - min(q.x, q.y < 0.0 ? r1 : r2),
                   abs(q.y) - h);
    vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    float s = ((cb.x < 0.0) && (ca.y < 0.0)) ? -1.0 : 1.0;
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

float sdf_solid_angle(vec3 p, vec2 c, float ra) {
    // c = [sin(angle), cos(angle)]
    vec2 q = vec2(length(p.xy), p.z);
    float l = length(q) - ra;
    float m = length(q - c * clamp(dot(q, c), 0.0, ra));
    return max(l, m * sign(c.y * q.x - c.x * q.y));
}

float sdf_cut_sphere(vec3 p, float r, float h) {
    float w = sqrt(r * r - h * h);
    vec2 q = vec2(length(p.xy), p.z);
    float s = max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y),
                  h * q.x - w * q.y);
    if (s < 0.0)    return length(q) - r;
    if (q.x < w)    return h - q.y;
    return length(q - vec2(w, h));
}

float sdf_ellipsoid(vec3 p, vec3 r) {
    // Conservative bound, not IQ's k0*(k0-1)/k1: that form over-reports near
    // the surface once the ellipsoid is eccentric, which makes sphere tracing
    // overshoot, and it evaluates 0/0 at the centre. Same zero level set.
    return (length(p / r) - 1.0) * min(r.x, min(r.y, r.z));
}

float sdf_octahedron(vec3 p, float s) {
    // Port of IQ's sdOctahedron; must stay in lockstep with sdf_shapes.octahedron.
    vec3 p2 = abs(p);
    float m = p2.x + p2.y + p2.z - s;
    bool cond_a = 3.0 * p2.x < m;
    bool cond_b = 3.0 * p2.y < m;
    bool cond_c = 3.0 * p2.z < m;
    // Fold into a canonical face frame: q = p.xyz / p.yzx / p.zxy.
    vec3 q;
    if (cond_a) {
        q = p2;
    } else if (cond_b) {
        q = vec3(p2.y, p2.z, p2.x);
    } else {
        q = vec3(p2.z, p2.x, p2.y);
    }
    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    float face_dist = length(vec3(q.x, q.y - s + k, q.z - k));
    // Signed-core early-out: without this the interior stays positive.
    float core = m * 0.57735027;
    return (cond_a || cond_b || cond_c) ? face_dist : core;
}

float sdf_pyramid(vec3 p, float h) {
    // NB: mirrors the existing Python in sdf_shapes.pyramid; the formula
    // does not consume `d2`, matching the Python source (potential bug there,
    // preserved here so the parity test stays exact).
    float m2 = h * h + 0.25;
    float px = abs(p.x);
    float py = p.y;
    float pz = abs(p.z);
    float px2 = (pz > px) ? pz : px;
    float pz2 = (pz > px) ? px : pz;
    float dx = px2 - 0.5;
    float dz = pz2 - 0.5;
    float ind = (max(dx, dz) < 0.0) ? -1.0 : 1.0;
    float dx2 = (dx > dz) ? dx : dz;
    float dz2 = (dx > dz) ? dz : dx;
    float px3 = px2 - clamp(px2, 0.0, 0.5);
    float pz3 = pz2 - clamp(pz2, 0.0, 0.5);
    float d1 = sqrt(px3 * px3 + pz3 * pz3) * ind;
    // d2 deliberately computed and discarded to match Python.
    float d2 = (h * py - 0.5 * dx2 + m2 * clamp((h * py + 0.5 * dx2) / m2, 0.0, 1.0)) / sqrt(m2);
    float py_min = min(py, 0.0);
    return sqrt(max(d1 * d1 + py_min * py_min, 0.0));
}


// ===========================================================================
// TPMS lattice primitives (implicit / approximate SDFs, clipped to a box)
// ===========================================================================
// n_periods is a vec3: number of full periods along each axis.
// Half-extents of the clipping box are 0.5 * n_periods * period.

float _tpms_clip(vec3 p, float pattern, float period, vec3 n_periods) {
    vec3 half_extents = 0.5 * n_periods * period;
    return max(pattern, sdf_box(p, half_extents));
}

// Turn a dimensionless TPMS pattern value into a wall of a given thickness.
// f is a sum of sines and cosines, so it carries no unit: it climbs at
// (2*pi/period) * |grad f| per millimetre. Multiplying by inv_rate = period /
// ((2*pi) * c_grad_max) (the steepest climb the pattern can reach anywhere,
// exact per family) makes the value a conservative distance, and makes
// min_thickness a length in mm.
//
// min_thickness is a floor, not an exact figure: the sheet is thinnest where
// the pattern is steepest and thicker elsewhere.
float _tpms_sheet(vec3 p, float f, float c_grad_max, float period,
                  float min_thickness, vec3 n_periods) {
    // inv_rate is independent of p: one scalar, then a multiply per query.
    float inv_rate = period / (6.283185307179586 * c_grad_max);
    return _tpms_clip(p, abs(f) * inv_rate - 0.5 * min_thickness, period, n_periods);
}


float sdf_gyroid(vec3 p, float period, float min_thickness, vec3 n_periods) {
    vec3 q = p * (6.283185307179586 / period);  // 2*pi/period
    float f = sin(q.x) * cos(q.y) + sin(q.y) * cos(q.z) + sin(q.z) * cos(q.x);
    return _tpms_sheet(p, f, 1.7320508075688772, period, min_thickness, n_periods);
}

float sdf_schwarz_p(vec3 p, float period, float min_thickness, vec3 n_periods) {
    vec3 q = p * (6.283185307179586 / period);
    float f = cos(q.x) + cos(q.y) + cos(q.z);
    return _tpms_sheet(p, f, 1.7320508075688772, period, min_thickness, n_periods);
}

float sdf_schwarz_d(vec3 p, float period, float min_thickness, vec3 n_periods) {
    vec3 q = p * (6.283185307179586 / period);
    float f = sin(q.x) * sin(q.y) * sin(q.z)
            + sin(q.x) * cos(q.y) * cos(q.z)
            + cos(q.x) * sin(q.y) * cos(q.z)
            + cos(q.x) * cos(q.y) * sin(q.z);
    return _tpms_sheet(p, f, 1.7320508075688772, period, min_thickness, n_periods);
}

float sdf_neovius(vec3 p, float period, float min_thickness, vec3 n_periods) {
    vec3 q = p * (6.283185307179586 / period);
    float f = 3.0 * (cos(q.x) + cos(q.y) + cos(q.z))
            + 4.0 * cos(q.x) * cos(q.y) * cos(q.z);
    return _tpms_sheet(p, f, 7.0, period, min_thickness, n_periods);
}

float sdf_lidinoid(vec3 p, float period, float min_thickness, vec3 n_periods) {
    vec3 q = p * (6.283185307179586 / period);
    float f =
        0.5 * (sin(2.0 * q.x) * cos(q.y) * sin(q.z)
             + sin(2.0 * q.y) * cos(q.z) * sin(q.x)
             + sin(2.0 * q.z) * cos(q.x) * sin(q.y))
      - 0.5 * (cos(2.0 * q.x) * cos(2.0 * q.y)
             + cos(2.0 * q.y) * cos(2.0 * q.z)
             + cos(2.0 * q.z) * cos(2.0 * q.x))
      - 0.15;
    return _tpms_sheet(p, f, 2.598076211353316, period, min_thickness, n_periods);
}


// ===========================================================================
// Compliant mechanism primitives
// ===========================================================================

float sdf_notch_hinge(vec3 p, float width, float depth, float notch_radius) {
    float beam = sdf_box(p, vec3(width * 0.5, depth * 0.5, width * 0.5));
    float cy = depth * 0.5 - notch_radius;
    float notch_top = length(p.xy - vec2(0.0, cy)) - notch_radius;
    float notch_bot = length(p.xy - vec2(0.0, -cy)) - notch_radius;
    return op_subtract(op_subtract(beam, notch_top), notch_bot);
}

float sdf_leaf_spring(vec3 p, float length_, float width, float thickness) {
    return sdf_box(p, vec3(length_ * 0.5, thickness * 0.5, width * 0.5));
}

float sdf_bellows(vec3 p, float outer_r, float inner_r, float period, float n_periods) {
    float total_length = n_periods * period;
    float pz_clamped = clamp(p.z, -total_length * 0.5, total_length * 0.5);
    float r_mod = inner_r + (outer_r - inner_r) * 0.5
                * (1.0 + cos(6.283185307179586 * pz_clamped / period));
    // length(p.xy) - r_mod measures radially, but the surface is tilted
    // wherever r_mod changes, so that offset exceeds the perpendicular
    // distance by sqrt(1 + r_mod'^2). r_mod' is a sine of known amplitude, so
    // dividing by its steepest value restores a (conservative) distance.
    float max_slope = (outer_r - inner_r) * 3.141592653589793 / period;
    // length(vec2(1, s)) is the hypotenuse of a 1-by-s right triangle,
    // more stable than sqrt(1 + s*s).
    float radial = (length(p.xy) - r_mod) / length(vec2(1.0, max_slope));
    float axial = abs(p.z) - total_length * 0.5;
    return max(radial, axial);
}

float sdf_serpentine(vec3 p, float amplitude, float wavelength,
                     float beam_width, float beam_height, float n_periods) {
    float total_length = n_periods * wavelength;
    float y_centre = amplitude * sin(6.283185307179586 * p.x / wavelength);
    float dist_to_centreline = abs(p.y - y_centre);
    // Vertical offset exceeds the perpendicular distance by
    // sqrt(1 + y_centre'^2); divide by the steepest tilt. Applied AFTER the
    // half-width subtraction so the beam keeps its Y extent.
    float max_slope = amplitude * 6.283185307179586 / wavelength;
    float beam = max((dist_to_centreline - beam_width * 0.5)
                     / length(vec2(1.0, max_slope)),
                     abs(p.z) - beam_height * 0.5);
    float axial = abs(p.x) - total_length * 0.5;
    return max(beam, axial);
}

float sdf_annular_sector(vec3 p, float inner_r, float outer_r,
                         float half_angle, float height) {
    // Distance to the two bounding half-planes, not abs(theta) - half_angle:
    // an angle is not a length (theta changes at rate 1/r, so the old form
    // over-reported by 1/inner_r near the bore). Both planes pass through the
    // Z axis, so the unit-normal dot products are exact distances.
    // A sector up to pi/2 is the intersection of the two half-spaces; beyond
    // pi/2 it wraps and becomes their union.
    float s_a = sin(half_angle);
    float c_a = cos(half_angle);
    float d_plane_pos = -s_a * p.x + c_a * p.y;
    float d_plane_neg = -s_a * p.x - c_a * p.y;
    float d_angular = (half_angle <= 1.5707963267948966)
                    ? max(d_plane_pos, d_plane_neg)
                    : min(d_plane_pos, d_plane_neg);
    float r = length(p.xy);
    float d_radial = max(inner_r - r, r - outer_r);
    float d_axial = abs(p.z) - height * 0.5;
    return max(max(d_radial, d_angular), d_axial);
}


// ===========================================================================
// 2-D primitives  (used inside extrusion / revolution)
// ===========================================================================

float sdf_circle_2d(vec2 p, float r) {
    return length(p) - r;
}

float sdf_box_2d(vec2 p, vec2 b) {
    vec2 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}

float sdf_rounded_box_2d(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float sdf_segment_2d(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdf_trapezoid_2d(vec2 p, float r1, float r2, float he) {
    vec2 k1 = vec2(r2, he);
    vec2 k2 = vec2(r2 - r1, 2.0 * he);
    vec2 q = vec2(abs(p.x), p.y);
    vec2 ca = vec2(q.x - min(q.x, q.y < 0.0 ? r1 : r2),
                   abs(q.y) - he);
    vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    float s = ((cb.x < 0.0) && (ca.y < 0.0)) ? -1.0 : 1.0;
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

float sdf_uneven_capsule_2d(vec2 p, float r1, float r2, float h) {
    vec2 q = vec2(abs(p.x), p.y);
    float b = (r1 - r2) / h;
    float a = sqrt(1.0 - b * b);
    float k = dot(q, vec2(-b, a));
    if (k < 0.0)     return length(q) - r1;
    if (k > a * h)   return length(q - vec2(0.0, h)) - r2;
    return dot(q, vec2(a, b)) - r1;
}

// Exact signed distance to a simple (non-self-intersecting) polygon.
// Direct translation of sdf_shapes.polygon_2d, which itself follows the
// Inigo Quilez sdPolygon: per-edge min foot-of-perpendicular distance for
// the magnitude, ray-crossing winding parity for the inside/outside sign.
//
// `v` is sized to SDM_POLY_MAX_N so a single helper serves polygons of any
// vertex count; the caller supplies `n = number of real vertices` and pads
// the tail. Each per-node function emits its own `V[SDM_POLY_MAX_N]` and
// calls this with its real N. Edge convention matches the Python: edge i runs
// from v[i] to v[i-1] (i.e. e = v_prev - v), so roll direction and sign line up.
// (Named sdf_polygon_2d to satisfy the emitter/lib `sdf_<kind>` lockstep test.)
float sdf_polygon_2d(vec2 p, vec2 v[SDM_POLY_MAX_N], int n) {
    float d = dot(p - v[0], p - v[0]);   // seed with squared dist to v[0]
    float s = 1.0;                        // sign accumulator (winding parity)
    for (int i = 0, j = n - 1; i < n; j = i, i++) {
        // Edge from v[i] to v[j] (j = i-1 with wrap), matching e = v_prev - v.
        vec2 e = v[j] - v[i];
        vec2 w = p - v[i];
        // Guard zero-length edges (duplicate closing vertex is a no-op in Python).
        float ee = dot(e, e);
        ee = (ee > 1e-12) ? ee : 1.0;
        vec2 b = w - e * clamp(dot(w, e) / ee, 0.0, 1.0);
        d = min(d, dot(b, b));
        // Winding number via three half-plane tests; flip sign on crossing.
        bvec3 c = bvec3(p.y >= v[i].y, p.y < v[j].y, e.x * w.y > e.y * w.x);
        if (all(c) || all(not(c))) s = -s;
    }
    return s * sqrt(d);
}


// ===========================================================================
// 2-D -> 3-D lifts (applied to a 2-D child SDF; the generated scene supplies
// the child via a function call from the emitted parent body)
// ===========================================================================
//
// Python:
//   revolution: q2d = [length(p.xy) - offset, p.z]; return sdf2d(q2d)
//   extrusion:  d = sdf2d(p.xy); w = [d, |p.z| - h]; return min(max(w),0) + length(max(w,0))
//
// We expose only the point-shaping / value-shaping helpers here; the
// emitter places the child SDF call.

vec2 lift_revolution_q(vec3 p, float offset) {
    return vec2(length(p.xy) - offset, p.z);
}

float lift_extrusion_finish(float d, vec3 p, float h) {
    vec2 w = vec2(d, abs(p.z) - h);
    return min(max(w.x, w.y), 0.0) + length(max(w, 0.0));
}


// ===========================================================================
// Scalar field primitives (used as displacement fields by op_displace).
// These return a displacement amplitude, NOT a distance.
// ===========================================================================

float field_sin_xyz(vec3 p, vec3 freq, float amp, vec3 phase) {
    vec3 arg = 6.283185307179586 * freq * p + phase;
    vec3 s = sin(arg);
    return amp * s.x * s.y * s.z;
}

float field_radial(vec3 p, float freq, float amp, float phase) {
    float r = length(p.xy);
    return amp * sin(6.283185307179586 * freq * r + phase);
}

#endif  // SDM_GLSL_LIB
