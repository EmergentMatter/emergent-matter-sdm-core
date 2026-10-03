# DR-0003: Sampled distance fields

Status: proposed for review.

## Context

Some fields are expensive to represent as analytic expression trees. A sampled
field keeps the emitted shader small while storing the sampled values in a
separate table. The format must preserve portable documents and let consumers
choose their own GPU storage layout.

## Decision

Add the `raster_field` primitive to schema 0.3. Its parameters describe a regular
three-dimensional grid using `origin`, positive `spacing`, integer `dims`, and
base64 `data` encoded as little-endian `f32le` or `f16le`. Samples are ordered
with x varying fastest. Validation rejects malformed payloads and nonfinite
samples or coordinates. Optional provenance records how the grid was produced.

Evaluation uses trilinear interpolation inside the sample domain. Outside it,
the field is the value at clamped coordinates plus the distance to the domain
box. Bounds include the possible zero-set continuation when boundary samples
are negative; the sample box alone is insufficient in that case.

The bake helper samples a reference field in chunks and records its content
hash and sampling parameters. It computes a conservative step factor from the
decoded, quantized samples, including the exterior distance gradient. A baked
field is an approximation of its reference field; neither interpolation nor
this step factor guarantees exact signed distances or preservation of features
smaller than the sampling interval.

Core emission exposes decoded float samples as `grid_table`, deduplicating
identical sample buffers. Consumers own texture packing, row widths, and
uploads. The GLSL library performs manual trilinear sampling through table
fetches, so it does not require float texture filtering. The CLI exports the
raw array and a legacy `grid_table_b64` representation for compatibility.
Consumers must use the same row width in shader macros and uploaded payloads.

Grid layout and samples require re-emission when changed; they are not live
uniform parameters. Raster fields can coexist with polygon tables in the
same shader. Corpus fixtures, numerical evaluation tests, and downstream
shader execution cover this contract.

## Consequences

Shader size no longer grows with the analytic tree used to produce the bake,
but documents and GPU sample storage grow with grid resolution. Authoring
systems must choose resolution and padding for their application and retain
the source needed to regenerate a bake. This primitive does not provide a
Part-level artifact lifecycle, automatic LOD, or posed kinematics.
