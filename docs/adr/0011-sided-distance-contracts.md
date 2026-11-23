# Side-specific distance contracts

## Status

Accepted.

## Context

A signed distance function (SDF) can preserve the intended boundary while
losing its distance interpretation. Hard constructive solid geometry (CSG)
operations provide familiar examples: union preserves exterior distances but
can underestimate interior depth. A rate bound alone does not distinguish
these fields from true distances. See the
[SDM glossary](https://github.com/EmergentMatter/emergent-matter-sdm/blob/main/docs/glossary.md)
for terminology.

The distinction matters when a consumer treats a field value as a length.
It must also survive compilation and baking without introducing numerical
work into a differentiable evaluation. The geometric limitation is described
in [Constructive Solid Geometry on Neural Signed Distance Fields](https://www.cs.columbia.edu/~silviasellan/pdf/papers/csg-sdf.pdf).

## Decision

Infer outside and inside contracts separately. Exact means distance to the
represented boundary. Lower bound means the magnitude does not exceed that
distance, including on the negative side. Unknown makes no distance promise.
These contracts assume valid geometric parameter domains and regular solids;
they do not certify floating-point accuracy or degenerate boolean topology.
Primitive declarations are conditional analytic statements. Before using one,
inference proves sufficient parameter conditions over the requested range.
Missing bounds, invalid dimensions, invalid rounding/cut/separation ranges,
and unproven outline simplicity yield unknown, not an assumed lower bound.
Fixed simple polygons can be checked directly; moving polygons and sampled
curve outlines remain unknown until their simplicity is established over the
whole range. Edges affected by the polygon evaluator's denominator floor
cannot receive an exact label. The declarations are also checked against the
distance audit's known failures. Unknown nodes never inherit an exact label.

Geometric refinements require a proof over the selected parameter range.
Strictly disjoint inferred boxes suffice for union. A cutter's box strictly
inside a host's box does not prove that the cutter is inside the solid: the
host might be a torus. Containment upgrades require an interior proof, with
simple sphere and box hosts supported initially. Derived parameters whose
ranges have not been established cannot support these upgrades.

Offset identities describe sets, not the resulting field's metric accuracy.
Dilating a union distributes over its operands, but formerly separated
components can overlap. Their minimum then underestimates interior depth.
Likewise, a shell thicker than a sphere's radius loses its inner cavity while
the absolute-value field retains a spurious interior ridge. Preserve only
the side whose distance follows from the offset construction. Do not upgrade
both sides merely because the input was exact.

Tube formulas need the same caution. A torus whose tube closes its hole
still has the intended boundary, but its field can underestimate interior
depth. Parameter ranges must establish a non-overlapping tube before its
interior is labelled exact. Capped arcs remain conservative inside until a
reach proof also rules out overlap across their endpoints.

Compilation, diagnostics and contract inference share a private deep snapshot
of the source SDF tree. Source dictionary, list or inline-expression edits
require recompilation; they cannot change a field behind its attached contract.
Explicit free vectors and supplied parameter bindings retain their existing
interfaces. Attach static contracts to compiled evaluators using their captured
fixed values, not declared bounds or pose envelopes. Evaluators accepting a free
vector remain unknown: their arguments are unrestricted, so a conditional
bounds proof is not a callable-wide guarantee. Bounds-based inference remains
available separately for callers that enforce its domain. Keep inference
outside JAX traces and leave numerical kernels, gradients, and shader semantics
unchanged. Binding-only and arbitrary callables carry no inferred guarantee.
A bake records the source contract separately from the interpolated field's
unknown distance accuracy. Source exactness does not survive interpolation
or quantization as an exactness certificate.

Length-valued objectives warn at expression compilation. Validated loading
and SDF compilation diagnose modifiers and blends whose inputs lack the needed
distance contract, including hard booleans and mirror transforms compiled
with smoothing enabled. A smooth mirror uses the child's contract for both
the original and reflected fields; hard mirrors do not consume a blend radius.
Loading checks current values as well as the declared envelope. Compilation
checks captured bindings and respects explicit smoothing overrides. Emit these
diagnostics once per consumer at the entry point, outside numerical evaluation
and JAX tracing. These diagnostics do not reject intentional field-value uses.
Gap audits reject a global rate bound above one, including an unknown rate, and warn about non-exact exterior fields. A sampled minimum is an
estimate: field underestimation and sampling overestimation can cancel, so
it must not be advertised as a certified lower bound. Raw callable gap
measurement remains a low-level operation whose caller supplies the field
contract. Gap audits resolve current values through the SDF parameter binding,
including derived relations whose caches may be stale or absent. Distance,
rate, sampling-domain checks and field evaluation all use that private fixed
snapshot. Auditing does not refresh the caller's parameter caches.
Gap results retain both exterior classifications. Non-exact results
are unverified and cannot pass a gap assertion even when their numbers match
the declaration or warnings are ignored. Exact exterior fields permit sampled
agreement checks, not certification of a minimum physical clearance.

## Consequences

Inference can be conservative without changing existing geometry. Matrix
maps, repeats, folds, sweeps and unanalysed deformations remain unknown until
they have a distance-preservation proof. This is deliberately independent of
their rate inference. Refining those classifications does not require a wire
schema change.

Warnings identify an unproven length interpretation, not invalid geometry.
They neither redistance the field nor establish a physical tolerance. Baking
provenance is descriptive and must not be used to upgrade the sampled field's
classification. True-distance queries and stronger gap certification remain
separate algorithms.
