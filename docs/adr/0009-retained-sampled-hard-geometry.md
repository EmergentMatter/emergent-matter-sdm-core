# 0009. Retained sampled hard geometry

## Status

Proposed.

On adoption, this record qualifies the source-retention policy described in [the sampled-field record](0003-raster-field.md). It does not mark that earlier proposed record accepted as a whole. The earlier file remains unchanged. The [wire contract](0001-sdm-wire-contract.md) continues to govern document validity and reader capability.

## Context

A sampled field can arise from a numerical design process that has no equivalent analytic expression tree. Requiring such a tree as the permanent source would exclude a useful retained definition even when the saved geometry can be reconstructed and checked by its intended consumers.

The existing sampled-field route can also be a cache of an analytic construction. These roles must remain distinguishable. Allowing retained samples must not silently promote every bake, optimization field or mesh cache to the authoritative design.

A readable file does not by itself establish geometric fidelity, physical distance accuracy or safe use by every query algorithm. A density value, a distance sample and a bound on field variation are different objects. The format must preserve its existing interpretation, while an application owns the evidence for its engineering claims.

## Decision

### Permit a retained sampled hard definition

An authoring application may designate a supported sampled hard field as the retained geometry definition. It need not retain an analytic expression tree that did not produce the definition. When an analytic construction remains authoritative, a bake remains derived from that construction.

The designation must identify the exact retained artifact and distinguish it from numerical working state. A hard material region has the declared categorical membership and boundary semantics. This does not mean that its stored scalar samples must be binary. Relaxed material fractions or arbitrary optimization variables are not accepted hard geometry merely because they can be stored as numbers.

This decision does not define where a file records that role. A retained definition and a bake are both `raster_field` nodes and look the same to a reader of the file alone; the marker that tells them apart, for example a key in the node's `provenance` object, is left to a later decision. Until then, the application's acceptance profile records which artifact it retains.

Available generating inputs, methods and ancestry remain provenance. Reconstruction of the retained geometry must not depend on an unavailable generator or an external sample file. Reproducing the full design campaign is a separate evidence question; primary retention does not claim that the whole campaign is portable or reproducible from the geometry alone.

### Preserve representation ownership

Reuse the existing sampled-field vocabulary and its native numerical semantics. This decision changes the permitted source-retention role, not the codec, compiler, interpolation, exterior continuation, parameter mutability or supported operation set. Those contracts remain in their owning source and reference records.

The permitted retained format does not select an application's live optimization coordinate, every analysis mesh, fabrication resolution or sole future representation. Separate numerical representations may refer to the same physical design through an explicitly checked relationship.

Creating a new hard artifact from a working field is an explicit transformation. Saving, loading or accepting that artifact does not define how an optimizer changes its live state. That lifecycle remains with the application.

### Require an identified acceptance profile

An application claiming retained-primary use must identify its profile for artifact identity, geometric interpretation, provenance and consumer acceptance. The profile includes:

- The authority of the saved definition and the exact artifact/reader identity.
- Coordinate and value units, sign and frame, material association and precedence.
- The generating method and available ancestry, including declared hardening, ties, repairs and regeneration inputs where applicable.
- Encoding and precision, the relevant geometric and numerical error claims, and the operations for which the artifact is qualified.
- Identity-bound evidence or explicit unavailable/unsupported outcomes for those claims.

These are acceptance information requirements, not new mandatory wire fields. Existing metadata and provenance containers may carry declarations. The wire rejects unknown raster parameter keys, so declarations belong inside the node's `provenance` object or the part's `metadata`. Declarations in `metadata` must not reuse a key the package already reads; `computed_envelope()`, for example, takes its smoothing from `metadata["envelope_smooth_k"]`. Application records may hold supporting evidence without making the geometry depend on an external sample payload. Do not duplicate an entire campaign ledger in every file.

Native validation does not enforce this application profile by virtue of this decision. A document may remain valid under the wire contract while being unacceptable for an application's retained-primary or engineering claim. Any later proposal for universal validation of additional provenance must identify its compatibility and schema consequences separately.

### Qualify the actual consumer

Keep boundary fidelity, physical distance accuracy and conservative field-variation bounds separate. Evidence for one does not grant the others. The decoded representation and the actual consumer are the qualification subjects, including the exterior rule and any relevant transforms or material-resolution operations.

A positive boundary or a padding setting does not alone establish a required geometric margin. Finite gradient probes can expose a bad bound but cannot alone prove a global bound. A stored step hint supplies no protection to a consumer that does not use it. Existing unsupported surface or rate-dependent operations remain unsupported until their own contracts are qualified.

Changing encoding, resampling, repairing or reinitializing a field must preserve the intended geometry within the application's declared tolerance or create an explicit proposed design change. A sample-content hash is not a substitute for full artifact, material and evidence identity.

### Preserve schema and release discipline

No wire extension is introduced by this decision. Apply [schema-versioning.md](../schema-versioning.md) to any actual implementation change. New dispatch vocabulary and wire content require the corresponding version treatment. Rejecting always-malformed input is distinct from making previously valid documents invalid. Released schema files remain unchanged.

Follow the normal contribution and release process for documentation and any separately proposed code change.

## Consequences

A numerical author can retain a supported hard definition without inventing an analytic parent. Analytic-first authoring and derived bakes remain valid workflows. Reconstruction uses the saved definition; a generator remains provenance rather than an undeclared alternative authority.

The application bears responsibility for its accuracy, material and operation-specific acceptance. This decision provides no physical validation, optimizer, signed-distance repair, automatic level of detail, transient state model or new geometry operation. It does not grant a uniform grid or memory budget across the design and analysis stack.

Consumers must not infer support from successful deserialization. Unsupported requirements remain explicit until the corresponding representation and consumer have been qualified.
