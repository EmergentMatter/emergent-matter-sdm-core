# 0009. Name parts from the top of the assembly down to the detail

## Status

Accepted.

## Context

A part's name should tell a person where it lives and what it is without
opening the tree: high-level assembly, sub-assembly, deeper sub-assemblies,
the part, then the detail that tells it from its siblings. Part definitions
are reused (one screw file placed in many assemblies), so a part file cannot
carry one assembly path in its name. Instance ids must also stay identifiers
(`[a-z_][a-z0-9_]*`), because dotted addresses such as `crank.pin` depend on it.

## Decision

The name lives in the tree, not in any one file. Every instance id names what
it is inside its parent, most general word first: the kind of part, then the
detail (`screw_m2x6_clampring_1`, `crank_driven_thumb`, not
`clampring_1_screw`). A sub-assembly's instance id is that assembly's short
name (`mech`, `wrist`), so it reads as a level. An instance of a part file
starts with the first word of that file's name (`screw_m2x6.sdm` placed as
`screw_...`): the kind comes first, and the parts list groups by it.

The full name of an occurrence is its instance ids from the top assembly down,
written upper case with hyphens: `HAND-MECH-CRANK-DRIVEN-THUMB`.
`software_defined_matter.assembly.naming.full_name` builds it and
`naming_problems` lists instances that do not start with their part's kind.
Nothing in the file format changes: ids stay identifiers, and the full name is
derived, so it can never disagree with the tree.

## Consequences

Renaming an instance renames every full name under it; tools that keep
references by path (saved views, markups, motions, build steps) must be
rebuilt or carried over with it. A part placed directly in the top assembly
has a two-level name (`HAND-SCREW-M2X6-CLAMPRING-1`); that is the tree's
shape, not a fault of the name.
