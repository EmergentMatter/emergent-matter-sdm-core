# `.sdm` conformance corpus

Small, hand-authored `.sdm` documents that pin what this package accepts and
rejects. They exist so a downstream consumer (a viewer, an authoring tool, a
reimplementation in another language) can test against this package's own
fixtures instead of inventing its own guesses about the wire format. See
[docs/adr/0001-sdm-wire-contract.md](../../../../docs/adr/0001-sdm-wire-contract.md)
for the design this corpus checks against.

This directory ships inside the built wheel, alongside the schema files it
sits next to, so it is available to an installed package, not only a source
checkout.

## Layout

- `valid/` -- one minimal document per schema version this package knows
  (named `minimal_<version>.sdm`). The older examples are
  single-material parts: small enough to read in full, rich
  enough to exercise a param reference through an SDF tree rather than an
  empty shell. `minimal_0.3.sdm` additionally carries a param `prior` and a
  derived param `expr` relation, the pieces of content that only schema `0.3`
  accepts, so the fixture set actually exercises what makes `0.3` a superset
  instead of being identical to `minimal_0.2.sdm` in every field but its
  declared version. `minimal_0.4.sdm` adds two attached bodies and a radial
  Hermite flexure, exercising the new blend and its inverse capability.
  `minimal_0.6.sdm` adds the `shear_linear` deform introduced by that schema.
  `axis_warps_0.6.sdm` exercises `scale_axis` and `taper_linear` in the same version.
  `assembly_mates_0.5.sdm` adds a grounded revolute mate referencing the port-bearing
  `minimal_0.5.sdm` part. It can be resolved as a complete assembly bundle.
- `placement/` -- numerical expectations for assembly placement consumers.
  `assembly_mates_0.5.expected.json` accompanies `valid/assembly_mates_0.5.sdm`.
  It specifies authored degree inputs, evaluator radians, world matrices and an
  off-axis landmark at 0, 45 and 90 degrees. The default is 45 degrees. Its
  translated offset and 90-degree X rotation distinguish joint sign and offset
  order; checking only coincident port positions cannot establish rotation.
  `assembly_motion_0.5.expected.json` checks a nested degree-authored mechanism
  driven by a root coordinate and live design ratio. Its resolved coordinates and
  world landmarks distinguish independent inputs from driven aliases.
- `consumers/` -- emission-contract fixtures for downstream hosts. Schema
  acceptance alone does not catch consumer drift (wrong helper names, dropped
  `poly_table`, reordered components, ignored `amplitude`). Each `.sdm` has an
  `expected/<stem>.json` fingerprint covering emission structure, JAX sample
  values, and load/save + CLI round-trips. See `consumers/README.md`.
- `invalid/` -- documents that are syntactically valid JSON but describe
  something that cannot actually be built or evaluated. Every one is
  rejected by `software_defined_matter.io.validate()`, which raises
  `software_defined_matter.sdf.validate.SemanticValidationError` naming the
  offending field and node path. The check runs on a document of any
  declared version.

  These files declare `0.2`, whose schema is permissive enough to accept
  them structurally, which is what leaves the semantic layer something to
  catch. Checked against the newer `0.3` schema instead, most of them fail
  there too, so a consumer running any JSON Schema library rejects them
  without needing this package's Python. The ones that still get through
  are listed under Known limitations, with why.

  Each fixture's `name` field matches its filename
  (`conformance_invalid_<stem>`), so a consumer's "which fixture failed"
  report identifies the document without needing the path it was read from.

  Vocabulary -- a name the wire contract does not declare:


  - `misspelled_primitive_kind.sdm` -- an SDF primitive `kind` that is not
    one this package implements (`"spere"`).
  - `unknown_modifier.sdm` -- a `modifier` name outside the contract
    (`"shell"`; the allowed set is `elongate` / `onion` / `round`). The
    frozen schemas leave `modifier` a bare string, as they do `kind`.
  - `unknown_deform.sdm` -- a `deform` name outside the contract
    (`"twist_z"` for `twist`).
  - `expr_unknown_node_type.sdm` -- an expression subtree nested in a
    primitive's kwarg whose `type` names no expression node kind
    (`"binp"` for `binop`). A schema pins expression node types only where
    a field is typed as an `expr`, and in the frozen schemas an
    `sdf.params` object is unconstrained, so on a `0.2` document this one
    reaches no schema check at all.

  Keyword arguments -- the right node kind, the wrong arguments:

  - `wrong_kwarg_name.sdm` -- a primitive whose `params` object uses a
    keyword name that primitive does not accept.
  - `missing_required_kwarg.sdm` -- a primitive whose `params` object omits
    a keyword argument that primitive requires.
  - `wrong_wire_shape_vec.sdm` -- a kwarg carrying the wrong wire shape: a
    3-D `box`'s `b` given as a length-2 array. Param *values* are untyped in
    the JSON Schema, so only the wire contract can catch this.
  - `displace_without_field.sdm` -- `deform: "displace"` with no `field`
    subtree, which that deform requires.
  - `unresolvable_param_ref.sdm` -- a `{"$ref": "..."}` node naming a param
    that does not exist in the document's own `params`.

  Dimension -- a well-formed node in a position it cannot occupy:

  - `2d_primitive_as_material_root.sdm` -- a 2-D primitive (`circle_2d`)
    at a material region's root, where a 3-D tree is required.
  - `3d_inside_extrusion_child.sdm` -- the mismatch the other way: a 3-D
    primitive in the 2-D profile slot of a `2d_to_3d` extrusion.

  Position in the document -- the same class of mistake outside `materials`:

  - `coupling_misspelled_kind.sdm` -- an unknown primitive `kind` under
    `couplings[].sdf_tree`. Every check above applies to a coupling node's
    tree exactly as it does a material region's.

  Deliberately **not** in `invalid/`, because they fail at a different layer
  and would break the "passes JSON Schema, then fails `validate()` with
  `SemanticValidationError`" contract every file here holds to:

  - a `plane` at a material root -- rejected, but as `UnboundedRootError`,
    from the position layer rather than the vocabulary one.
  - a param `prior` on a `0.2` document -- rejected by the JSON Schema
    itself, so it never reaches semantic validation.

  Both are real failures worth fixtures; they need their own bucket with its
  own stated contract, not a slot in this one.

## How to use this corpus

From an installed copy of this package:

```python
import importlib.resources
import json

import jsonschema

from software_defined_matter.io import load_schema, validate
from software_defined_matter.sdf.validate import SemanticValidationError

conformance = importlib.resources.files("software_defined_matter.schema.conformance")

for path in (conformance / "valid").iterdir():
    if path.name.endswith(".sdm"):  # not path.suffix: a zip entry is not a Path
        validate(json.loads(path.read_text()))  # must not raise

for path in (conformance / "invalid").iterdir():
    if path.name.endswith(".sdm"):
        doc = json.loads(path.read_text())
        # Passes the JSON Schema on its own -- that is the point of this
        # fixture set -- but validate() must still reject it.
        jsonschema.validate(doc, load_schema(doc["schema_version"]))
        try:
            validate(doc)
        except SemanticValidationError:
            pass
        else:
            raise AssertionError(f"{path.name} should have failed validation")
```

Catch `SemanticValidationError` rather than `Exception`. A bare `except`
also swallows a fixture that failed the JSON Schema, or one rejected by some
other layer, so the suite keeps passing after the corpus has stopped testing
what it says it tests.

`tests/test_conformance_corpus.py` runs this same loop over every fixture, so
a hand-edited fixture that no longer matches the description above fails the
build.

A consumer that cannot import this package at all (for example, a tool
running inside a host application's bundled interpreter with no `jax` or
`jsonschema` available) can still read these files directly as plain JSON;
the wire format itself is defined by the schema files this corpus sits next
to, not by anything this package's Python code does at runtime.

## Known limitations

Some fixtures the JSON Schema does not reject. A consumer validating with a
JSON Schema library alone will accept them, so treat schema validation as
necessary rather than sufficient. The reasons are not the same, and only
one of them is permanent.

`unresolvable_param_ref.sdm` cannot be expressed. Checking that a
`{"$ref": name}` leaf resolves against the document's own `params` means
reading a value in one part of the document and looking it up among the
keys of an object in another. Standard JSON Schema has no way to reach
across an instance like that; `$ref` addresses schemas, not data. Ajv's
`$data` extension can, but depending on it would break the promise that any
JSON Schema library will do.

`2d_primitive_as_material_root.sdm` and `3d_inside_extrusion_child.sdm`
could be expressed, and currently are not. Both use a primitive that is
valid on its own, in a slot that cannot hold it: a 2-D profile where a 3-D
solid is required, and the reverse. Nothing about JSON Schema prevents
catching that. It would take splitting `$defs/sdf` into a 3-D and a 2-D
definition, each recursing into the right one, with a `2d_to_3d` node's
child pointing at the 2-D def. `wire.py` already carries `NodeSpec.dim`, so
the generator has what it needs.

That split is not free: it duplicates the op, transform, modifier and
deform branches across both definitions, and it can only appear in a new
schema version, since released files are frozen. It would also not remove
the semantic layer. `plane` as a material root is a third contextual rule
of a different shape, and the bbox rules are not structural at all.

This corpus is deliberately small and hand-authored. It pins named failure
modes and the currently known schema versions; it is not a fuzz corpus and
does not attempt to cover every invalid document shape. Add a fixture here
when a new class of mistake needs a name, not for every possible malformed
input.
