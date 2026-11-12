# 0010. A cached bounding box records what it was inferred over

## Status

Accepted.

## Context

`Part.metadata["bbox"]` holds two different kinds of box. An author may set it to choose a sampling domain, including one that deliberately crops the part or one the analytic inferrer refuses. `Part.infer_bbox_from_sdf` and `tighten_part_bbox` also persist the box they compute there, so it is a cache of a result derived from the part.

A cache goes stale when the part changes. A box inferred for a sphere of radius 1 does not contain the same sphere at radius 2. If every reader treats every stored box as an authored domain, a stale cache silently becomes the domain: sampling misses part of the solid, export writes a cut mesh, and a later tightening seeds from the undersized box.

The two kinds cannot be told apart from the box alone, and many existing documents and tests set a box without saying which kind it is.

## Decision

### Record the basis of a cached box

A writer that persists a cached box also writes `metadata["bbox_basis"]` beside it. The record holds:

- the source: `"inferred"` from `Part.infer_bbox_from_sdf`, or `"tightened"` from `tighten_part_bbox`;
- the inference mode, `"bounds"` or `"values"`;
- the box it describes;
- a fingerprint of the part's envelope together with the parameter intervals that mode takes from the part.

An author can mark a chosen box explicitly with `{"source": "authored"}`. An authored box needs no fingerprint, because it is not derived from the part.

### Read a stored box only through its basis

Every reader of the stored box, including sampling, export, preview, the shrink-wrap seed, metric compilation, shader emission and motion bounds, calls `sdf.bbox.stored_bbox(part)` instead of reading `metadata["bbox"]` directly. `stored_bbox_status(part)` reports one of five states:

| Status | Meaning | The reader |
|---|---|---|
| `absent` | No box is stored. | infers a box, or refuses |
| `authored` | The record marks the box as the author's. | uses it as stored |
| `unlabelled` | No record describes this box. | uses it as stored |
| `valid` | A cache whose basis still holds. | uses it |
| `stale` | A cache whose basis no longer holds. | infers a box, or refuses, as if none were stored |

A cache's basis holds while the part has the same envelope, the same intervals for the record's mode, and every current value inside those intervals. Relation values are evaluated fresh for this check, not read from a cache.

Using an authored or unlabelled box is never evidence that the part fits inside it. Only `valid` says that.

### Two modes

A bounds-mode box, which `Part.infer_bbox_from_sdf` writes, covers each bounded parameter's whole range, so it stays valid while values move inside their bounds. It goes stale when a value leaves its bounds, when the bounds change, when the geometry changes or when a relation's value moves.

A values-mode box, which `tighten_part_bbox` writes, holds only at the values it was taken at. Any value change makes it stale.

### A tightened box inherits its seed's label

Tightening samples only inside its seed box, so the result is only as complete as the seed. `tighten_part_bbox` writes a `"tightened"` record only when its seed was derived from the part: a stored box whose record still holds, or a fresh analytic inference. A box tightened inside an authored box stays authored. A box tightened inside an unlabelled box, or inside `metadata["bbox_half_size"]`, is left unlabelled. Neither is ever reported `valid`.

### Unlabelled boxes keep their old meaning

A stored box with no record cannot be told apart from an authored one, so it is used as stored and reported `unlabelled`, never `valid`. This includes boxes written before this decision and boxes replaced by hand after an inference, which no longer match the record beside them. Re-running the inference replaces an unlabelled box with one inferred from the part as it is now, which is labelled.

An unlabelled box raises no warning, because a warning would fail every caller that turns warnings into errors.

### No wire change

`metadata` is a free object, so the new key needs no schema or version change. The metadata table in `docs/optimization.md` lists it.

## Consequences

A stale cache is no longer used as a domain. Where it used to crop the part, and export wrote a cut, non-watertight mesh with only a logged warning, the reader now infers the box again, or refuses where inference refuses, as it does when no box is stored.

Authored boxes, and boxes stored without a record, behave as before. An author who wants a box treated as a deliberate domain, rather than as unlabelled, records it as `{"source": "authored"}`.

The check runs inside this package. A consumer that reads `metadata["bbox"]` from the file itself, without `stored_bbox`, still trusts a stale box.

The fingerprint covers the part's envelope, so checking a basis costs more for a part that embeds large sampled data.
