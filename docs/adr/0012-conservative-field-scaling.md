# Conservative field scaling and physical lengths

## Status

Accepted.

## Context

Signed distance function (SDF) evaluators can preserve a boundary while changing
faster than one unit per unit of travel. A consumer that assumes unit rate can
then skip a surface. Dividing by a proven rate bound prevents this overshoot,
but generally does not recover true distance. See the
[SDM glossary](https://github.com/EmergentMatter/emergent-matter-sdm/blob/main/docs/glossary.md).

## Decision

Use conservative scaling where an analytic bound establishes it, and retain
unknown status where no bound is established. A unit-rate claim is conditional
on the operation's input and parameter assumptions. It is not an exact-distance
claim. Distance audits continue to track under-reporting separately from
excessive rates and discontinuities; moving an operation between those lists
does not mean its distance error has been eliminated.

Consumers must choose guarantees for their use. Conservative stepping can use
a proven rate bound. Surface extraction can use the zero set. Physical offsets,
shell thicknesses, blend radii, and clearance assertions need the relevant
interior or exterior distance guarantee. A normalized approximate field alone
does not supply that guarantee. Until a consumer checks it, callers must not
interpret its length-valued inputs as certified physical dimensions.

Keep evaluator normalization, shader normalization, and inferred field rates
consistent. Tests must cover every registered displacement field, composed
fields, and parameter changes. A bound over a parameter interval must cover
individual evaluations; it need not equal their tighter bounds.

## Consequences

Positive scaling preserves an operation's own zero set, but can change a parent
operation's zero set. If a field changes from F to F/C, subtracting an offset r
selects F = C*r instead of F = r. Neither expression establishes a physical
offset unless the necessary distance contract holds. Smooth booleans and
integrals that depend on field magnitudes can change as well.

These changes require a minor release entry and updated consumer fingerprints.
Existing models that compose affected operations with length-valued consumers
need their resulting dimensions checked. Uniformly rescaling the requested
length is not a general migration strategy: the divisor can depend on position
or on optimized parameters.

Conservative scaling can be loose, especially for displacement and deformation.
Tighter analytic bounds remain useful, but must retain their proof over the
whole claimed domain. Redistancing is a separate design choice because it also
changes evaluation cost and differentiation behavior.
