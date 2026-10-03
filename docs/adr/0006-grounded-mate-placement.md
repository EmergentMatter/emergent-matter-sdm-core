# 0006. Evaluate mate placement from grounded components

## Status

Accepted.

## Context

A reusable part's ports depend on design values and may follow rigid bodies.
Placement must consume those evaluated frames, retain occurrence identities, and
remain differentiable. Closed mechanisms also need inspection when an external
engine supplies their coordinates, without adding a mechanism solver to core.

## Decision

Include fixed, revolute and prismatic mates in the schema 0.5 assembly contract.
Assemblies with and without mates share that version. Schemas through 0.4 remain
unchanged.

A mate relates child and parent port frames through an explicit offset followed
by a joint transform. Use the offset frame's positive Z axis, right-handed
rotation, radians and millimetres. There is no implicit face-normal reversal.
The `Mate` docstring defines the matrix convention.

Compile each assembly's direct-child graph from explicit instance grounds.
Sort mate IDs to make traversal independent of authoring order. Ground every
connected component; never infer a world origin from whichever part appears
first. Retain additional grounds and evaluate every mate as a residual, including
edges that were not needed to place an instance.

Evaluate nested assemblies locally before using their deep or promoted ports in
parent placement. A parent positions the whole nested assembly; it cannot ground
or solve an ungrounded internal component. Rigid flexure-end ports follow their
endpoint body through the existing kinematics evaluator.

Separate structural compilation from numerical design and coordinate inputs.
Promoted coordinates are aliases of canonical occurrence coordinates. Expression
bindings between coordinates require the subsequent motion-composition layer;
the placement compiler rejects them until that evaluator exists.

Expose a pure numerical result with poses, mate residuals and a validity flag for
JAX callers. Provide a checked host boundary that rejects invalid configurations
and reports mate names, positional errors and angular errors. Closure tolerances
are fixed requirements, not optimization variables. Core does not repair or solve
an inconsistent supplied state.

## Consequences

The static preview can use the same core placement contract as other consumers.
It rejects unsupported body deformation instead of rendering rigid geometry with
misleading moving-port labels. The web viewer integration remains separate.

Changing graph structure requires recompilation. Changing numerical inputs does
not. Rotation residuals use the principal branch, with a discontinuity at a half
turn. Passing a closure check establishes consistency at the supplied state only;
it does not prove valid geometry, collision freedom or continuous-motion closure.
