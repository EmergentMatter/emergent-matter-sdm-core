# 0009. A Part is one physically inseparable body

## Status

Proposed (2026-10-08)

## Context

A `Part` can hold several `MaterialRegion`s, and nothing stops a CEM from
packing many separate bodies into one Part: nine coils in one file, or a
bolt and the plate it fastens. The format accepts it, but the file then
stops matching the thing being built. A viewer colours it as one object,
the file tree no longer reads as a bill of materials, and a part cannot be
inspected, toleranced or replaced on its own. Assemblies
([ADR 0005](0005-assembly-foundations.md)) already exist to place separate
Parts, so packing them into one Part is never necessary.

The modelling rule itself (a Part is one inseparable object, separable
things are their own Parts placed by an Assembly, and every dimension has
one owner) is documented once, in the `emergent-matter-sdm` concepts page.
This ADR records only what `sdm-core` does about it.

## Decision

**`sdm-core` offers a check, not a hard rule.** `count_part_bodies` samples
the union of a Part's material regions on a coarse grid and counts the
6-connected solid bodies; `check_part_is_one_body` emits a
`PartBodiesWarning` when there is more than one. A multi-material Part
made as one piece (a co-printed coil in its radiator) counts as one body,
because regions are unioned before counting.

It is a warning, not an error, and it is not called by `save` or `load`:
- a grid count can merge two bodies closer than one cell, or split a
  feature thinner than one cell, so it can be wrong both ways;
- existing files that break the rule must keep loading.

CEM test suites and the CEM template call it on every Part they write.

## Consequences

- Authors get a cheap, explicit signal when a Part holds separate bodies.
- The cell size is a choice per call (`n_cells`); thin-walled parts may
  need a finer grid to avoid false splits.
- Promoting the check to `save`, or to an error, is a later decision once
  its false-positive rate on real parts is known.
