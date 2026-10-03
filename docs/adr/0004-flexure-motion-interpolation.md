# Flexure motion, rendering, and conservative bounds

## Status

Interpolation, supported per-flexure inverses, and conservative bounds are
accepted and implemented. CPU and GLSL point-membership queries now establish
which posed material each region owns. Bounded CPU and GLSL ray queries now
find occupancy changes for a restricted continuous-field subset, prove solid
coverage at supported welds, and optionally return witnessed grazing contacts. Viewer
integration remains proposed; the ordinary whole-part SDF emitter still refuses
flexures.

This record consolidates the interpolation decision from ADR 0004 and the
rendering and bounds proposal formerly recorded as ADR 0005.

## Interpolation

### Context

A flexible region needs a motion at every rest point between its two attached
bodies. Interpolating matrix entries loses rigidity. Extracting angles from
endpoint matrices loses full turns that were present in authored joint motion.
ADR 0003 left the general interpolation described as "slerp + lerp"; this
needs a precise definition before consumers can agree on intermediate poses.

### Decision

For matching operation chains, interpolate each authored angle or distance and
compose the operations in their original order. Matching rotations have the
same axis line; reversed axes are accounted for. Matching translations have
parallel axes. An empty ground chain supplies zero coordinates for the other
chain. This preserves authored winding, including complete turns. At zero
DOFs, matching endpoint joint coordinates must agree, as well as both body
transforms being identity, so hidden complete turns cannot deform the rest pose.

For other chains use `T_from exp(w log(inv(T_from) T_to))`, a constant relative
rigid screw. This couples translation and rotation; it is not independent
quaternion slerp and translation lerp. The relative rotation takes the principal
branch, with magnitude at most a half turn. It cannot preserve winding absent
from endpoint matrices, and it is not differentiable across the half-turn branch.
Small-angle series keep derivatives finite at identity.

Evaluate the blend in rest space, then clamp it to [0, 1]. The `axis_ramp` blend
is `(dot(point, axis) - lo) / (hi - lo)` before clamping. Its axis is not
normalized; authors control the coordinate scale. Other supported scalar fields
use the compiled snapshot of design parameters. At blend zero and one, return
the exact attached body matrices. Ownership compares all body regions followed
by all flexure regions; the first region wins a tie.

### Consequences

The evaluator can pose flexible vertices using JAX, including differentiation
away from ownership, clamping, and principal-rotation branch boundaries.
A linear axial blend agrees with the existing axial twist's inverse query-space
rotation because its axial coordinate is invariant under that rotation.
This does not establish parity with every nonlinear twist modifier.

These are forward transforms evaluated at rest points. Evaluating the same blend
at a posed query point does not generally invert the deformation. Conservative
motion bounds are available with an explicit consumer opt-in. Shader emission
still rejects flexures; the viewer cannot yet render them through its current
rigid-body shader path.

## Rendering and conservative bounds

### Context

Core main at 9a1c3d4 evaluates forward flexure motion. Web main at 0945b6f
uses one inverse rigid transform per cached geometry placement. A placement
slot, sometimes called a lane, groups the motion data needed to locate that
geometry. A flexure cannot use a single rigid matrix because its motion varies
with the rest point.

Relevant implementation seams are core `kinematics.py`, `_flexure_motion.py`,
`motion_bounds.py`, and `glsl/emit.py`; web `bodymotion.py`, `frontend/posexf.js`,
`frontend/webgl.js`, and `frontend/webgpu.js`. Web's `mesh.py` describes its mesh
as an interaction proxy, and the WebGPU renderer advertises
`meshFallbackOk: false`. Promoting a deformed mesh to the primary representation
would require a separate renderer decision, material and picking support, and
an explicit approximation tolerance.

A forward map need not have a unique inverse. Even when it does, applying the
inverse of a matrix whose blend was recomputed at the posed point is generally
wrong. For a z rotation by `(pi/2)*clip(x,0,1)`, the rest point `(1,0,0)` moves
to `(0,1,0)`. Re-reading x gives blend zero, leaving an inverse error of sqrt(2).
A numerical root finder alone cannot certify uniqueness, find every preimage,
or provide a safe ray step through folds.

### Rendering proposal and implemented bounds

#### Preserve field rendering and state the supported subset

Keep the field renderer. Introduce explicit flexure rendering capability checks
in core, shared by analytic emission and the web planner. Initially target a
single shared rotation axis line with either an axial blend or a radial blend,
including either attached body being ground. Both the axial coordinate and
radius are invariant under this rotation, so the posed point supplies the same
blend as the rest point. Reuse the existing `twist_radial` Python/GLSL operator
and its swept-radius bounds where the authored profile matches. The core library
already supplies the radial Hermite profile and live endpoint-angle emission;
those mechanisms are present in main. This is an extension of existing field
rendering, not a new renderer. Preserve authored angles, including full turns. Other joint chains and general screws retain
forward point evaluation but remain refused by this renderer until supported
with equivalent guarantees. This is partial viewer support, not completion of
the general flexure integration plan.

Profile matching is a capability requirement: a linear `axis_ramp` must not
silently become Hermite smoothstep. The `radial_hermite` blend explicitly
provides that profile, with parity tests against `twist_radial`. Radial positive endpoint
angles use the inverse query rotation. Existing `twist_linear` deliberately uses
the opposite query sign, and its transverse linear coordinate is not invariant
under z rotation. Reusing it for the forward evaluator requires a separate
mapping proof; it is not in the initial certified subset.

For this subset, with inverse `g`, test material membership and rest ownership
at `g(q)`. A distance-based renderer additionally needs a safe surface field.
A scalar field used for surface rendering is not automatically a distance
after deformation.
For axial angle slope k and maximum radius r about the rotation axis,
`||Dg|| <= 1 + |k| r` gives a conservative local-to-domain Lipschitz factor.
Combine it with a proved bound for the rest field and any interpolation error
before using values for ray steps, empty-cell skipping, or grid truncation.
The radius bound must cover the padded query domain, not only surface vertices.
For a radial Hermite ramp, a conservative slope bound is
`1.5*|a1-a0|/(r1-r0)` with `0 <= r0 < r1`; multiply it by the maximum radius in
the query domain in the same Jacobian estimate. Outside the ramp the slope is
zero. Field-rate inference now supports this bound for radial twist. Its
approximate-distance return still requires the sampling bound before it can
be used to justify ray steps.
Clamping makes the map piecewise differentiable but does not increase this bound.
Transform normals using the inverse-map Jacobian transpose, not just the local
rotation. GPU and CPU reference results must agree at welds and clamp boundaries.

A flexure placement therefore needs a deformation description and safe sampling
metadata, rather than another rigid matrix row. Motion updates change only
reached DOFs and affected bounds. Rest geometry and blend design parameters are
snapshot data; editing them invalidates the relevant cache. Range changes or
motion outside advertised ranges must invalidate the corresponding envelope.

#### Establish the material and ownership contract first

Ownership uses the smallest rest-region distance, with declaration-order ties.
These regions classify points; they are not necessarily the material surface.
The current body shader/planner renders region trees as solids. The new
point-membership queries instead use the actual material trees, with regressions
where material and classifier geometry differ. Surface rendering must adopt
this ownership contract before the viewer can render flexures correctly.

The proposed clipping field
`max(material_field, d_i - min(d_j for j != i))` is rejected as a complete
rendering solution. At a tie inside material, `material_field` is negative but
the ownership difference is zero. Every tied region therefore returns zero,
and taking their minimum leaves a false surface inside the solid even when
all poses agree. A Lipschitz bound cannot remove that false zero set.

The implemented reference instead answers a Boolean question for every
owner/material pair. Pull the posed query back with that owner's inverse;
keep it only if the rest classifier selects that owner and the actual material
field is nonpositive. First-declared regions win rest-distance ties. After
posing, different owners can overlap, so preserve all matches rather than
selecting one global inverse. Material records retain document order even when
they reference the same external material ID. Document smoothing settings apply
consistently to CPU and GLSL material and classifier fields.

This contract establishes occupancy for shaders, tests and future picking. It
is deliberately separate from the SDF emission contract: a Boolean query gives
no distance, safe ray step, surface normal or reliable way to discover a thin
surface along a ray. Unsupported per-region inverses fail before emission.
The bounded surface queries below consume this contract. Integrating their
explicit hit/miss/unresolved results remains a prerequisite for the viewer.

For initial conservative rest bounds, use the complete finite material envelope
for each owner. This may be loose but covers all assigned material. Do not use a
region's negative-set box as a bound on its nearest-region ownership cell: that
cell can extend far beyond the region. Tightening owned-material boxes is a
separate optimization. Unsupported material bounds must fail or disable pruning
under an explicit full-motion scene-box contract; never substitute a rest box
for a swept box.

#### Bound compatible joint interpolation without sampling

For each matched operation, enclose the two endpoint coordinate intervals after
normalizing reversed axes. Every convex blend lies in that interval hull for
all blend weights in [0,1]. Propagate the material rest box through these
interval operations in authored order using the existing rotation-arc sweep.
This includes intermediate extrema and complete turns. Do not wrap authored
angles to a principal branch. Independent interval hulls may lose correlations
but remain conservative without enumerating the combined DOF space.

#### Bound general screw interpolation independently of shader support

Write the endpoints as `(R_f,t_f)` and `(R_t,t_t)`. The relative principal
rotation vector omega has norm at most pi. The relative screw translation is
`J(w omega) w J(omega)^-1 R_f^T (t_t-t_f)`.

The SO(3) left Jacobian has singular values 1 along its axis and
`2*sin(theta/2)/theta` perpendicular to it. Therefore over the principal branch,
`||J(w omega)|| <= 1` and `||J(omega)^-1|| <= pi/2`, including their identity
limits. Rotations preserve norms. For a rest point p:

`||posed(p)-t_f|| <= ||p|| + (pi/2)*||t_t-t_f||`.

Infer endpoint translation boxes by sweeping the origin through each body's
operation intervals, rather than using the body's geometry box. Let rho be the
maximum corner norm of the material rest box, and delta the maximum corner norm
of the interval difference between endpoint translation boxes. Expand the
from-body translation box by `rho + (pi/2)*delta` on every axis. This bounds all
rest points, DOFs in range, and weights in [0,1]. It is deliberately loose and
works across the principal half-turn discontinuity. Outward arithmetic margins
and finite-value checks remain necessary, following the existing bounds module.

Use this bound only for the general screw branch. Compatible authored joint
chains preserve winding and must use their operation sweep instead. Endpoint
positions alone are insufficient: a full turn has identical endpoints while its
half-blend point lies on the opposite side of the axis.

The existing `twist_radial` full-radius envelope is also a valid tighter special
case for coaxial radial motion: radius and axial extent remain unchanged. Retain
the general screw envelope for incompatible endpoint chains instead of assuming
radial invariance there.

#### Core and viewer responsibilities

Core owns interpolation classification, owned-material semantics, inverse
capability checks, derivative/sampling bounds, and swept envelopes. The
bounds output includes flexures in authored order while preserving the existing
body list; the scene box encloses both. Each flexure depends on the union of its
two bodies' DOFs and relevant design inputs. Shader capability remains separate
from availability of a conservative box. During rollout, flexure bounds require
an explicit `include_flexures=True` opt-in. Old consumers that iterate only the
body list must keep failing rather than silently dropping flexible components.

Web consumes that contract for payloads, buffer updates, world clipping, dirty
regions, picking, and both renderer backends. Add deformation error allowance to
cached-grid sampling; padding a rest grid by a rigid-body halo is not a substitute
for bounding its deformed support. An unsupported flexure must produce a clear
capability error before a partial rigid-only scene can be displayed.

### Consequences and delivery order

1. Implemented core bounds prerequisite: establish material-support bounds, expose flexure
   envelopes and dependencies, implement matched-chain sweeps and the general
   screw bound. Retain shader refusal. Test actual material points outside their
   classifier boxes, mixed axes, translated origins, full turns, zero motion,
   derived parameter ranges, and unbounded inputs.
2. Core rendering prerequisite: CPU/GLSL point membership, the axial/radial
   inverse subsets, and bounded ray queries are implemented. Ray queries retain
   explicit unresolved results and an analysed geometry subset. They estimate
   normals in world space and test thin material, gaps, overlaps and seams.
   This is not a universal surface-distance field or complete flexure rendering.
3. Web integration: consume the contract in GLSL/WGSL and cached fields. Verify
   material identity, ownership seams, landmark parity, conservative containment,
   live motion without static rebakes, and range/cache invalidation in-browser.
4. General rendering remains a separate decision: a certified inverse/domain
   method or an explicitly approximate forward-surface renderer. Neither follows
   automatically from forward point interpolation or conservative envelopes.

### Investigation evidence

A deterministic exploratory check against the current core screw evaluator used
2,000 random endpoint frame pairs, rest points, and blend weights. All posed
points were inside the derived sphere; the largest measured radius ratio was
0.887500. The inverse counterexample above produced 1.414214 mm error. These
checks corroborate the derivation; they do not prove interval implementation or
GPU correctness. The accompanying bounds tests exercise material containment,
full turns, reversed axes, mixed-axis screws, unit conversion, and legacy refusal.

### Sampling prerequisite progress

Field-rate inference now uses the conservative query-Jacobian estimate above
for axial twists and bends, and the Hermite slope bound for radial and linear
twists. It propagates query domains through translations and nested twists.
Other transform wrappers return an unknown rate for domain-sensitive children;
this disables unsafe skipping and may increase sampling work. Existing radial
and linear query-sign conventions are unchanged. A rate bound for `twist_linear`
does not make it the inverse of the authored forward flexure motion.

The axial implementation previously used `sqrt(1+s*s)`, which underestimates
even a simple shear: at s=1 the actual largest singular value is about 1.618,
not 1.414. Gradient regressions now cover this case and a translated local
sampling domain that previously hid large shear.

## Radial blend and inverse contract

The `radial_hermite` blend requires schema 0.4. Its explicit axis, origin, and
inner/outer radii describe the same clamped cubic Hermite profile as the existing
radial twist. Older schema artifacts remain unchanged; documents without the
new blend retain their existing minimum schema version. The version index names
the new nested capability through `wire.KINEMATICS_CAPABILITIES`.

The evaluator advertises per-flexure inverse support only for ground endpoints
or a single matching rotation about a shared axis line. An axial blend must be
parallel to that axis. A radial blend must measure radius from the same axis
line. Both coordinates survive the motion unchanged, so the inverse can use
the posed query's blend value and undo its rigid matrix. Reversed axes and
origins displaced along the same line are compatible; full authored turns are
preserved. The initial capability check conservatively excludes longer chains,
translations, mixed axes, and non-invariant blend coordinates.

The inverse takes a flexure name explicitly. It does not recover global
ownership, resolve overlap between different posed regions, or provide a signed
distance. Tests cover forward/inverse round trips, welds, JAX derivatives, degree
conversion, full turns, and agreement with `twist_radial` query signs.

CPU and executed GLSL membership tests now cover actual material distinct from
classifiers, internal ties, gaps and overlapping posed owners. They also cover
axial/radial inverses, complete turns, degree inputs, reversed axes, displaced
origins, ground endpoints, nonlinear motion expressions and live design inputs.
Polygon and raw raster payloads use the existing shader resource contract.

The bounded ray contract below is the next implemented step. Whole-part
flexure SDF emission remains refused until the viewer consumes that contract.
Neither schema acceptance nor a Boolean shader query implies a rendered
flexure surface.


## Bounded surface queries

### Decision

Use ordered interval subdivision to find a demonstrated change in material
occupancy. The exclusion field guides interval skipping; its zeros alone never
count as hits. Crossing-only queries return a hit bracket, a proved miss within the requested
segment, or an unresolved interval. An optional contact tolerance adds a
distinct witnessed-contact result as described below. The host
must preserve unresolved results rather than displaying them as empty space.
Invalid shader inputs have a separate status; the CPU API raises an error.

At a fixed pose, combine owners whose inverse maps agree. Rigid matrices use
exact equality. A flexure joins a rigid group only when its compatible authored
endpoint joint coordinates agree, not merely when their matrices look equal.
This preserves full turns while removing internal classifier seams at zero or
shared rigid motion.

For each group G, evaluate all fields at its common inverse query. Let m be the
minimum material field, a the minimum classifier field within G, and b the
minimum classifier field outside G. The group's exclusion value is
`max(m, a-b)`; when G contains every owner, use m alone. Take the minimum across
groups. A strict negative value proves occupancy and a strict positive value
proves emptiness. Zero remains ambiguous. Different inverse groups may still
create internal zeros; these cause further search or an unresolved result,
never an automatic surface hit. The ownership-invariance proof below now resolves the rotationally invariant
weld regression, allowing the ray to continue through solid material to its
external exit. Other unproved classifier/motion combinations retain the
unresolved outcome; the solver does not fill potential gaps by assumption.

### Rate and search guarantees

The maximum analysed rest-field rate, multiplied by twice the maximum inverse
Jacobian bound, bounds this exclusion field. The factor two covers classifier
differences. Rigid inverse maps have unit rate; supported flexures use the
axial/radial estimates above. Rest-field bounds cover a sphere-enclosed box
containing every inverse query from the entire supplied world query box.
Plane fields explicitly account for the authored normal's length.

At an interval midpoint, a value whose magnitude exceeds the rate times the
interval half-length plus the numerical error allowance proves that the whole
interval has the same occupancy. Only intervals matching the initial occupancy
are skipped. Other intervals are subdivided in ray order. A small uncertain
interval may be combined with subsequent intervals within the requested spatial
tolerance to demonstrate a crossing. Earlier uncertainty is never skipped to
return a later hit outside that tolerance. Budget exhaustion, grazing contacts,
and features too small to establish a crossing remain unresolved.

These are ordinary floating-point bounds with outward margins and an explicit
field-evaluation error allowance, not formally rounded interval arithmetic.
The default allowance targets modest-scale float32 evaluation; consumers must
choose an allowance appropriate to their model scale. The supplied query box
bounds where the rate is valid; it is not a claim that all scene geometry lies
inside it. Unknown rates and unsupported geometry are rejected. Raster fields,
unanalysed primitives, coordinate repetition and general inverse maps are not
silently admitted through a unit-distance assumption.

### Normals and host updates

World-space finite differences of the exclusion field include the inverse map's
spatial variation. At a smooth exposed surface this approximates the inverse
Jacobian transpose applied to the rest normal. Corners have no unique normal;
a nonfinite or zero gradient yields an unavailable normal. These are shading
estimates, not certified differential geometry.

A design snapshot emits one shader. Every pose update supplies an atomic packet
containing DOF inputs, inverse-group indices, the whole-domain and rest-field
rate bounds, and each flexure's angular slope for ray-specific bounds. Pose
changes need neither shader regeneration nor static geometry rebaking. Design,
geometry or query-domain edits require a new snapshot. Existing membership
emission exposes rest-field function names so the surface implementation shares
its geometry and resource payloads instead of parsing shader text.

### Remaining integration

The viewer must clip rays to the query domain, bind complete pose packets,
consume bounded hit brackets and estimated normals, and expose unresolved
queries. It must preserve these semantics in WGSL and exercise actual browser
motion, seams, thin material and cache invalidation. The new query API is
separate from ordinary distance-field emission; existing flexure refusal stays
in place until the host can handle this contract. Performance at scene scale
and broader geometry support remain to be established.


## Weld coverage and grazing contacts

### Prove solid coverage across a weld

When every ownership classifier is invariant under the common rotation axis,
all inverse candidates agree on classifier values even when the material
fields differ. The compiler proves this structurally for centred spheres,
aligned planes and cylinders/tori, and supported CSG, modifier, translation and
uniform-scale wrappers. All authored motion operations must rotate around the
same axis line. Material geometry need not have this symmetry. Axis alignment
and centre tests use exact rational snapshot coefficients; almost-aligned
geometry does not activate the proof. Unrecognised symmetries remain outside
this sufficient test.

Retain the exclusion field `E = min_G max(m_G, a_G-b_G)` defined above. Add the
coverage field `C = max_G min(m_G, b_G-a_G)`. A strictly negative C means every
group either loses ownership or contains material. Since some group must win,
that proves occupancy even at an ownership tie. The invariance proof ensures
E and C cannot have opposite strict signs. Select the more negative margin
inside material and the more positive margin outside. Internal weld zeros now
disappear when every tied owner contains material, while true boundaries and
gaps remain. Declaration-order tie semantics still come from membership.

This selection preserves the original rate bound. Within each sign region it
is a min or max of two fields with that bound. A transition between signs must
pass through a common zero of both fields, so the selected field is continuous
there as well. No unproved smoothing or surface offset is introduced.

Regressions cross axial and radial welds, including concentric radial shells
and full turns, and find the external boundary. They also check material that
is not rotationally symmetric, tilted axes, displaced pivots, and rejection of
almost-centred classifiers and mixed rotation axes.

### Bound work along the actual ray

The rest-field rate still covers the complete inverse query domain. For each
flexure, tighten its inverse-Jacobian estimate along the ray: the maximum
radius about the rotation axis occurs at one of the segment endpoints. Axial
blend variation is proportional to the ray direction's axial projection;
radial blend variation is bounded by its perpendicular component. Multiply
these by the authored angular slope. This avoids charging a near-axis ray for
the radius of the whole query box, or charging an axial ray for radial blend
variation. Interval subdivision also accounts for the resulting rate before
opening a small uncertainty window. The finite work budget remains explicit.

### Witness a grazing contact within a stated tolerance

A true tangent need not change occupancy along the ray. The optional contact
mode therefore looks off the ray, along an estimated normal or coordinate axis,
for an inside/outside sample pair within a specified world-space radius of a
ray point. Both endpoints must lie in the query domain, have opposite actual
membership, and clear the field-error allowance. Their segment proves that a
material boundary lies within that radius. Merely having a small exclusion
value or an ownership zero is never sufficient.

A crossing found within the current uncertainty window takes priority. If the
search would otherwise return unresolved and has found a witness, it instead
returns contact, the earliest uncertain ray window, the witness ray parameter,
and the inside/outside pair. The GLSL API exposes the witness direction so the
host can reconstruct that pair. This is an explicit proximity result: it does
not assert an exact ray intersection, exact tangency, or a bracket containing
the mathematical tangent parameter. Near misses within the allowance may
produce contacts. The routine is not a complete search for every boundary
within the proximity radius. Tolerances smaller than numerical resolution,
unproved contacts, and exhausted searches without a witness stay unresolved.

Contact handling is opt-in, preserving the crossing-only API and its result
codes. CPU/GLSL regressions cover a non-dyadic tangent, the witness endpoints,
near misses outside the tolerance, ordinary crossings, budget exhaustion and
invalid tolerances. The viewer must keep contact distinct from an exact crossing
and bind the complete rate packet; no web-side geometry workaround is required.
