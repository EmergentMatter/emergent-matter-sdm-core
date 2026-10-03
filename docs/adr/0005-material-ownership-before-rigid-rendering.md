# 0005. Establish material ownership before choosing rigid rendering

Status: Implemented for review, 2026-09-14.

## Context

The rigid GLSL emitter renders body-region solids. Material-motion queries use
those regions to classify ownership of the actual material. A radius-1 region
owning a radius-5 material sphere demonstrates the difference even at rest.
The existing rigid emitter remains useful as a lower-level geometry API, but
passing every authored kinematics document to it loses material meaning.

## Decision

Core decides when owned material can be represented by independent rigid solids.
The viewer uses the resulting render-only snapshot consistently for emission
and grid planning. The authored part remains unchanged and is what gets saved.

A sole owner receives all material. Multiple owners may use the fast path when
structural equality establishes complete material coverage and conservative,
strictly separated rest bounds establish exclusive ownership throughout the
declared design ranges. Bounds include the existing float32 arithmetic margin.
Pose transforms may subsequently overlap without invalidating rest ownership.
Authored boxes and sampled agreement cannot establish this admission.

Other cases use bounded material queries, retaining explicit unresolved results
and refusals. Their clipping bounds enclose material support for every owner,
including bodies-only documents. Classifier boxes cannot serve that purpose.

## Consequences

The admission is deliberately incomplete: touching or overlapping bounds can
reject geometry that a stronger proof could admit. Smooth multi-owner or
multi-material cases also require queries. Such cases inherit the consumer's
query-renderer capability and performance limits; no region-solid or unposed
mesh fallback may hide those limits.

The adapter preserves body order and body component identity. It does not turn
body component IDs into material IDs or implement material-specific picking.
Design and range edits require admission and bounds to be recomputed. Inputs
outside advertised ranges are not covered by the proof.

The shared ownership fixture compares core membership with both executed GLSL
and downstream WGSL. Ordinary region-emitter tests remain separate because
they intentionally exercise a different contract.
