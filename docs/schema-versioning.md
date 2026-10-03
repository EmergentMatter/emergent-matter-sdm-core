# Cutting a new `.sdm` schema version

How to add a schema version, and how to decide whether you need one at all.
Background and rationale live in
[adr/0001-sdm-wire-contract.md](adr/0001-sdm-wire-contract.md).

## First: do you need a new version?

`schema_version` is the minimum version a tool must support to read a
document. It is not the version of the package that wrote the file.

You need a new version only when a document could carry something an older
reader cannot interpret correctly. That means adding a wire field, or widening
a vocabulary a reader dispatches on.

You do **not** need a new version for:

- tightening validation. Rejecting a document that was always malformed is a
  bug fix, not a format change.
- anything that does not appear in a `.sdm` file.

Generated schemas enumerate primitive, op, transform, modifier and deform
names. Adding one therefore widens a vocabulary that readers dispatch on and
requires a new version. Set the node's `since` value to that version.

A released schema file is immutable. Never edit one to fix a problem in a
version that already shipped. A new version is the only path to a change in
what a document validates as.

## Adding a version

Worked through as `0.4`. Do these in order.

1. **Extend the version vocabulary** in `src/software_defined_matter/wire.py`:

   ```python
   SchemaVersion = Literal["0.1", "0.2", "0.3", "0.4"]
   ```

   `SCHEMA_VERSIONS` derives from it through `get_args`, so there is no second
   tuple to update here.

2. **Extend `model.py`** to match:

   ```python
   LATEST_SCHEMA_VERSION = "0.4"
   KNOWN_SCHEMA_VERSIONS = ("0.1", "0.2", "0.3", "0.4")
   ```

   These are a separate declaration because `model` cannot import `wire`
   without a cycle. `test_wire_and_model_agree_on_the_known_schema_versions`
   fails if the two lists disagree.

3. **Declare the new fields** on the relevant record in `wire.py`, each with
   `since="0.4"`. A field with no dataclass counterpart, written by tooling
   outside this package, also needs `wire_only=True`; see `Part.kinematics`
   for the worked case. Without that flag the structure gate expects a model
   field that does not exist, and the version index understates the version.

4. **Gate emission, if the new content should not churn existing files.** Add
   an entry to `_CAPABILITY_MIN_VERSION` in `model.py` and teach
   `_capabilities_used` to detect it. A part that does not use the new content
   keeps emitting the older version, so existing files do not churn.

5. **Regenerate**:

   ```
   uv run python -m software_defined_matter.schema._generate
   ```

   This writes `sdm-0.4.schema.json` and refreshes `index.json`. The generator
   only ever emits the newest version, so `0.3` is frozen from the moment `0.4`
   exists. Never hand-edit either output. CI's `schema` job fails if the
   committed files differ from what the generator produces.

6. **Freeze the previous version.** Add `0.3` to the frozen-schema test in
   `tests/test_schema_generate.py` alongside the versions already pinned there,
   so a later change cannot quietly alter it.

7. **Add a conformance fixture.** `schema/conformance/valid/minimal_0.4.sdm`,
   carrying the content that only `0.4` accepts, so the corpus exercises what
   makes the version a superset rather than a copy of its predecessor. See that
   directory's README.

8. **Write a changeset**: `uv run scripts/changeset.py`, minor.

## What the generator does not derive

Some `$defs` are static literals in `_generate.py` because they describe things
outside the wire contract's scope: the `kinematics` block and its nested defs,
and `Param.ui`'s internal shape. If a new version adds content that lives
entirely inside one of those, the version index cannot see it unless something
in `wire.py` carries a matching `since=`. That is what `wire_only` is for.

`expr` is half and half. The contract declares which node types exist
(`wire.EXPR_NODE_TYPES`), and the generator builds one branch per entry, so
adding an expression node type means adding it there first. Each branch's
interior, meaning operand names and operator enums, is written in
`_generate.py`, because those vocabularies belong to `dsl/expr.py`. Generation
raises if the two ever disagree about the set of node types.

## Checking your work

- `uv run pytest` green, including the contract gates in
  `tests/test_wire_contract.py`.
- `python -m software_defined_matter.schema --index` reports your version, with
  the fields you added listed against it.
- Regenerating a second time produces no diff.
- The previous version's schema file is untouched in the diff.


For new vocabulary inside the static kinematics definitions, register the dotted
capability name in `wire.KINEMATICS_CAPABILITIES` with its introduction version.
The generator includes it in the version index without inventing a dataclass
field. Schema 0.4's `kinematics.flexures.blend.radial_hermite` is the worked
example; its content gate is in `model._CAPABILITY_MIN_VERSION`.
