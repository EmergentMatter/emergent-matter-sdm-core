# 0001. The `.sdm` wire contract

## Status

Accepted

## Context

Tools that read `.sdm` files, `sdm-view` and the web viewer, had no
supported way to find out which schema version `sdm-core` emits or which
versions it accepts. They hardcoded a version string instead, and got it
wrong. `sdm-view`'s loader pins `"0.1"` and rejects this repo's own
kinematics example.

The repo gave them nothing better to use:

- `SCHEMA_VERSION` was `"0.2"` while the package version was `0.3.0` and
  `sdm-0.3.schema.json` sat in the tree.
- `PRIOR_SCHEMA_VERSION` read as "the previous schema version" but meant
  "the version a document needs when a param carries a prior".
- Nothing named the current version, and `load_schema()` was not exported
  at the package root, so no consumer could ask in process.
- A document that declared an old version validated cleanly against that
  version's schema even when its author believed they were writing a newer
  one. `validate()` warned only on a version it did not recognize at all.

A second problem surfaced while investigating. Every schema file at the
time left an SDF primitive's `kind` a bare string and its `params` an
unconstrained object, so none of these were rejected by `validate()`:

- a misspelled or unknown `kind`, `modifier`, or `deform`
- a keyword argument the primitive does not accept
- a missing required keyword argument
- a `$ref` naming a param the document does not define

They failed later instead, at SDF compile or, because `make_sdf_closure` is
lazy, at first evaluation, as a Python exception naming no field.

## Decision

### What `schema_version` means

A document's `schema_version` is the minimum version a tool must support
to read it. It is not the version of `sdm-core` that wrote the file, and it
is not a label the author picks.

`min_schema_version_for(part)` computes it, as the newest version any
content the part uses requires. A part with a non-delta param `prior`,
which only `sdm-0.3.schema.json` accepts, requires `0.3`. Every other part
gets `0.2`, the floor: the wire format's top-level `kinematics` block
arrived at `0.2`, and `Part` has no field for it to test against, so `0.2`
is asserted for every part rather than checked per part.

The floor makes this a minimum requirement rather than "the oldest schema
file that would accept this document". Those differ: a part with no `0.2`
content validates cleanly against `sdm-0.1.schema.json` and still declares
`0.2`. `0.1` is a version this package reads and never writes.

Because the value comes from content, it moves when content moves, and it
can move down. Remove a param's prior and the part validates against an
older schema again. A consumer reading the field as "when was this written"
or "which sdm-core made this" will read that drop as a bug. It is not.

### Validation splits into two layers

The JSON Schema stays structural and versioned. It checks which keys exist
and the shape of leaf values, dispatched on the document's own declared
version, so a `0.1` document is checked against `0.1`'s schema.

Semantic validation lives in `sdf/validate.py`, checks a document against
`wire.py`, and ignores the declared version. It covers what the JSON Schema
cannot express: whether a `kind` / `modifier` / `deform` name exists,
whether a node's keyword arguments are the right ones with the right wire
shapes, whether a primitive's dimension fits the position it sits in, and
whether every `$ref` resolves. It raises `SemanticValidationError` naming
the offending field and node path.

The split exists because a JSON Schema is written per version, and most
documents in the wild declare `0.2`. A constraint added only to the newest
schema file would rarely fire. A misspelled primitive name was never valid
under any version, so the check belongs in a layer that runs on every
version rather than only the newest one.

### Released schema files are frozen

`sdm-0.1.schema.json` and `sdm-0.2.schema.json` are never edited or
regenerated once released. `since=` metadata in `wire.py` is checked
against their existing content, not used to reproduce it.

Regenerating a released schema file could change what an already shipped
document validates as, retroactively. Only the current unreleased version
is a valid generation target, so a schema bug is fixed in the next version
rather than in the one that shipped it.

### The contract publishes on two channels

The primary channel is the package root: `load_schema`,
`KNOWN_SCHEMA_VERSIONS`, `LATEST_SCHEMA_VERSION`, `supports_version`,
`assert_supported`, and `min_schema_version_for`. Version ordering is owned
here so no consumer reimplements it and breaks the day `0.10` exists.

The second channel needs no import at all. `schema/index.json` lists every
known version, what each one added, and which is current, and
`python -m software_defined_matter.schema` prints either a schema file
(`--version X`) or the index (`--index`) using only the standard library.
`sdm-view` runs inside Blender's bundled Python interpreter, which has
neither `jax` nor `jsonschema`, and reaches this package only across a
subprocess boundary. A Python API alone would not have served it.

The index is generated from the contract, so it cannot fall behind the
schema files it describes. A subprocess call, in any language, can read it.

`schema/conformance/` ships alongside both, inside the installed package.
It is a small set of `.sdm` files a consumer can point its own tests at, to
check that its reader agrees with this one about what is valid.

## Consequences

- A document's declared `schema_version` can go down across a re-save. That
  is correct behaviour, and anything treating the field as provenance will
  read it as a bug.
- Semantic validation rejects documents that previously loaded, on every
  declared version. An existing `.sdm` file with a misspelled kind, a wrong
  keyword argument, a missing required argument, or a dangling `$ref` now
  fails `validate()` where it used to fail later with a vaguer error.
- A schema bug can only be fixed going forward, in the next unreleased
  version. There is no path to correcting a version that already shipped.
- A consumer that cannot install `jax` and `jsonschema` has a supported
  path to the contract.

### Known limitations

- This decision covers `sdm-core` only. `sdm-view` and the web viewer keep
  their own version literals until they are updated separately.
- The wire format itself does not change here. No field changes meaning or
  structure. This tightens what counts as valid, and adds no syntax.
- Every schema file's `$id` is a placeholder
  (`https://emergent-matter.example/...`) and stays one until the
  open-source release checklist runs alongside LICENSE and IP clearance.
  This is decided, not overlooked: an `$id` is an identifier, not a URL
  that has to resolve, and nothing here fetches it. Do not replace it
  early. A URL chosen before the repo is public would not resolve either,
  and editing an `$id` in a released schema file is the retroactive edit
  the frozen-files decision rules out.
  `schema/_generate.py` reads the host from one named constant rather than
  a literal per `$id`, so the eventual fix is one line there instead of an
  edit per generated file.
- `schema/_generate.py` derives most of the current schema from `wire.py`,
  but not all of it. `kinematics` and its nested defs, and `Param.ui`'s
  internal shape, stay as static literals in the generator.

  That matters for the version index, whose "what did this version add" is
  derived from `since=`. An addition living entirely inside a static block
  is invisible to it unless the contract declares the field. Schema `0.2`'s
  `kinematics` block was that case: `Part` has no field for it, so nothing
  carried a `since=` and `index.json` reported nothing added at `0.2`.
  `FieldSpec` now carries a `wire_only` column for a wire field with no
  dataclass counterpart, and `kinematics` is declared that way. The
  structure gate skips such a field when comparing against
  `dataclasses.fields()`; the index counts it. A future static-literal
  addition needs the same treatment, or the index will understate its
  version.
