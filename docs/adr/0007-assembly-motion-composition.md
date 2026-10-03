# 0007. Calculate bound motion before placing parts

## Status

Accepted.

## Context

An assembly can expose a child's motion coordinate under another name. This is
called a promoted motion input. Both names refer to the same coordinate. An
expression binding calculates a coordinate from other inputs. Allowing callers
to supply separate values for an alias or a calculated coordinate could give
that coordinate conflicting values.

Motion within a part can move its ports. Core must calculate the ports' positions
and orientations before using mates to place neighbouring parts. Otherwise, those
parts would attach to the ports' old locations and lose contact when the ports
move.

## Decision

Resolve each alias to the coordinate it names. Evaluate binding expressions in
an order that makes each result available before another expression uses it.
Callers supply values only for coordinates that have no expression binding.

When the same part is used more than once, each instance has its own motion
state. A binding expression uses parameter values from the assembly that declares
it, including values overridden by a containing assembly.

Use radians and millimetres in motion expressions and calculations, as in part
kinematics. Convert degree defaults and inputs supplied in authored units before
evaluation. Expression results are already in evaluator units, so a target
coordinate declared in degrees needs no further conversion. Design parameters
keep their declared units. Authors must include any necessary conversion when
using them in a motion expression. Core does not infer units from the arithmetic.

Calculate body motion within each part before using its ports for mate placement.
To move a point from its rest position into world coordinates, apply the internal
motion first, then the part instance's placement. A rigid attachment at a
flexure's end follows the body at that end. Moving sampled points this way does
not provide a way to recover their rest positions or calculate exact signed
distances to the deformed geometry.

Check the supplied configuration without solving for a different one. JAX calculations
return a validity flag and closure residuals, which measure how far the mates are
from satisfying their required relative positions and orientations. The checked evaluation
methods reject invalid states and identify the affected instances or coordinates.
Check calculated coordinates and every rigid body's transform for undefined or infinite
values, even when no port or mate uses them. For flexures, check motion at the
sampled points. Those checks do not prove that motion is valid everywhere in the
region. Declared coordinate ranges describe bounds, but evaluation does not clamp
values to them.

Trajectory generation, interpolation over time and physics enforcement remain
outside Core. Prescribed and physics-solved trajectories use the same independent
coordinate inputs. Core can evaluate a batch of supplied states and report
validity separately for each one.

## Consequences

Callers must get the input names and their order from the compiled evaluator.
Adding a value for an expression-bound coordinate gives the input vector the
wrong size and causes an error. Callers can inspect calculated coordinates and
world body transforms along with port frames and mate residuals. Checks on posed
points also reject invalid placement.

Viewers and mesh generators still need to support the deformation being used.
Core's ability to move points does not give a viewer that support automatically.
Finite results at sampled points do not prove that parts are free of collisions
or suitable for manufacturing.
