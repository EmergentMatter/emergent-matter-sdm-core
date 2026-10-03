# 0005. Resolve assembly definitions separately from occurrence state

## Status

Accepted.

## Context

A reusable part definition can appear in multiple places with different parameter
bindings. Treating those occurrences as independent copies of a mutable `Part`
would duplicate definitions and confuse their design variables. Treating them as
one shared parameter scope would make changing one occurrence change the others.

Ports also need a complete orientation. A normal alone leaves their clocking
undefined, and a literal position becomes stale when design dimensions change.

## Decision

Use a document discriminator for parts and assemblies in schema 0.5. An absent
`kind` still denotes a part. Keep referenced documents external, with relative
paths and optional sha256 pins over exact file bytes. Resolve paths relative to
the declaring file and reject recursive inclusion using canonical paths.

Replace `CouplingNode` and `Part.couplings` with `Port` and `Part.ports`. This is a
breaking API change. There is no runtime alias or automatic conversion of old
coupling records. Historical schemas remain immutable for structural inspection.

Ports store a position and scalar-first unit quaternion. Frame components may
contain pure design-parameter expressions. A body attachment identifies which
rigid body subsequently moves the frame. A rigid attachment at a flexure endpoint
uses that endpoint body; it is not an attachment to an arbitrary deforming point.

An assembly declares its own design variables. Overrides read the parent scope;
non-overridden part parameters retain their local defaults and derived relations.
Contained `free` flags do not add optimization variables. Repeated references
share parsed definitions, while each occurrence gets its own numerical binding.

Promoted ports and motion inputs resolve to terminal occurrence addresses. Deep
addresses remain available. Multiple drivers and cyclic motion bindings are
invalid. Each inherited constraint retains its occurrence scope; child objectives
are not included in the parent objective.

Core evaluates configuration and geometry. Time-dependent prescribed or
physics-solved trajectories belong to the physics repository. This foundation
layer validates motion bindings but does not evaluate mate placement or run a
solver. Metric expressions require an explicit occurrence-aware evaluator.

## Consequences

Dependent callers and authored files must explicitly migrate old coupling data.
They must choose the missing clocking orientation rather than rely on a guessed
tangent. The package release is breaking even though historical structural schema
validation remains available.

Structural document validation does not establish cross-file correctness. Bundle
loading checks references, pins, promotions, overridden names, and driver graphs.
Only the root free vector drives numerical instance bindings, so JAX gradients
can propagate through nested overrides without mutating source definitions.

Compiled bindings describe a loaded snapshot. Reload and recompile after changing
its structure. Numerical inputs can change within that fixed structure. A
parameter-dependent quaternion must remain nonzero over its evaluation domain;
normalizing a zero quaternion is never interpreted as a valid identity frame.
