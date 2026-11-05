# 0010. Extension nodes let another package add node kinds

Status: proposed for review.

## Context

Every node kind the format knows is listed in `wire.py` and dispatched by name in about a dozen places: compile, bounding box, slope bound, envelope, feature size, validation, shader emission and the generated schema. A package that wants a kind of its own has had to edit all of them in a copy of this one. Released schema files are immutable, so each new kind also costs a schema version.

## Decision

Schema 0.7 adds one node type, `extension`:

```json
{"type": "extension", "id": "vendor.kind", "version": 1,
 "payload": {}, "children": [], "children_dim": 3, "fallback": {}}
```

- `id` names the kind. The prefix `em.` is reserved for Emergent Matter packages.
- `version` is the version of that kind's own payload shape.
- `payload` is an object this package never interprets. It only finds the `$ref` leaves in it, so a payload may refer to parameters, and it keeps the object unchanged.
- `children` are ordinary SDF subtrees of dimension `children_dim`.
- `fallback` is required and is an ordinary SDF tree of this package's own kinds. It may not contain an extension.

A package registers an `ExtensionSpec` with `register_extension`, explicitly. Compile, bounding box and slope bound are required, so a kind cannot be registered without saying how to evaluate, bound and rate it; the slope bound may be infinity, but it has to say so.

Where no spec is registered for a node's `id` and `version`, every consumer uses the node's `fallback`, so a document always opens. Compiling such a node when its payload refers to a free or derived parameter raises, because the fallback is fixed and the parameter would do nothing; `extensions.policy(frozen_params="allow")` opts a viewer in.

Using any extension node raises a part's minimum schema version to 0.7. Nothing else about saving changes: parts without an extension write the same bytes as before, and saving never consults the registry.

A kind this package later makes native is recorded in `extensions.PROMOTIONS` with a function that converts its payload. Reading converts such nodes; a read never rewrites a file. Registering a promoted id is skipped without error.

## Consequences

- Another package can add kinds without a fork and without a schema version of its own.
- Contexts passed to the callbacks are the only surface a kind sees. They are API and stay stable.
- A reader older than the release that knows schema 0.7 cannot open a document that uses an extension.
- A fallback is a snapshot. Nothing detects that one has gone stale; `refresh_fallbacks` rebuilds fallbacks on request.
- A shader draws the fallback of every extension, and `GLSLEmission.fallbacks` names them so a viewer can say the picture is approximate. A shader routine for an extension is future work and needs no schema change.
