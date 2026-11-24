# Changelog

<!-- towncrier release notes start -->

## 1.4.0 (2026-11-24)

### Minor

- Change distance-field magnitudes for `shear_linear`, `taper_linear`, `displace`, shape-interpolated `loft`, and screw-thread flanks to limit over-reporting under their documented assumptions. Their own zero sets are preserved, but enclosing `round`, `onion`, and smooth booleans can change geometry; field-dependent metrics and optimization objectives can also change. Recheck dimensions in composed models. These fields remain approximate, so rounding distances, shell thicknesses, and blend radii are not certified physical lengths. Displacement consumer fingerprints have changed. See `docs/adr/0012-conservative-field-scaling.md` for the policy and limitations.
- Normalize plane equations in JAX and GLSL, including tiny and large float32 normals, while preserving their zero sets. Correct pyramid interior distances near the base and elongation's interior core term. These value corrections can change geometry when composed with rounding, shells, or smooth booleans. Degenerate zero plane normals do not define a plane.
- Schema validation runs on jsonschema-rs (draft 2020-12, compiled), about a thousand times faster on large documents; jsonschema still writes the error message for a rejected document and decides alone for documents holding numpy values, NaN or infinity, or when jsonschema-rs is not installed.

### Patch

- A negated negative number (for example a rotation by minus a negative angle) now emits valid GLSL instead of a `--` decrement that failed to compile.
- Bound screw-thread flank distances throughout the bore, including shallow coarse-pitch teeth, with matching JAX and GLSL evaluation.
- Keep JAX gradients finite at zero-length norms, polygon and cone boundaries, coincident capsule and segment endpoints, sector folds, radial-twist axes, and zero-width smooth booleans. Preserve shader value parity for the corresponding branches.
- Keep capsule and segment_2d endpoint gradients finite at coincident endpoints under float32 JAX JIT, with matching GLSL point-distance evaluation.
- Keep open sweep cap depths continuous across segment switches in JAX and GLSL, including subdivided straight paths and cap faces partly covered by another leg.

  Open sweeps now evaluate the profile at both endpoint caps in addition to the selected segment, in JAX and emitted GLSL. Shader consumers such as sdm-view should account for these additional profile evaluations, particularly with complex profiles. This is not a measured whole-sweep slowdown. Cap correction does not make general sweeps exact distance fields.
- consolidate and expand tests for eikonal property (distance=1 or move of 1)

## 1.3.0 (2026-11-21)

### Minor

- Make compile_raster_field public in software_defined_matter.sdf.compile, so tools that walk an SDF tree reuse the raster decoder.

## 1.2.0 (2026-11-20)

### Minor

- Add extension nodes (schema 0.7): a registry lets another package add node kinds, each carrying a public fallback tree that stands in wherever the kind is not registered. Parts that do not use one save the same bytes as before.

## 1.1.0 (2026-11-16)

### Minor

- A box that `Part.infer_bbox_from_sdf` or `tighten_part_bbox` persists into `metadata["bbox"]` now carries a `metadata["bbox_basis"]` record of what it was inferred over, and sampling, export, metric compilation, shader emission and motion bounds set the box aside once the part has changed under it, so a stale box no longer cuts off a grown part. `tighten_part_bbox` writes the record only when it tightened inside a box derived from the part; tightened inside an authored or unlabelled box or `bbox_half_size`, the box keeps that label. An authored or unlabelled box is used as stored, as before. `sdf.bbox.stored_bbox` and `stored_bbox_status` expose the check. ADR 0010 records the convention.

### Patch

- Clarify when a sampled hard field may be retained as the geometry definition and which consumer checks remain application-owned.
- Raster spacing is checked where it is built and decoded: a string, a boolean, a nested or non-numeric entry, or a value that is not positive and finite now raises a ValueError naming spacing, instead of a bare TypeError or a silent conversion. A NumPy number or a 0-d numeric array is accepted as a scalar pitch.
- The consumer conformance README now lists the raster_field and kinematics_roundtrip fixtures, which were in the corpus but missing from its layout table.
- infer_sdf_max_rate now returns UNKNOWN for helix, pyramid and raster_field instead of a unit rate: a helix's value jumps across its axis, a pyramid's across its base plane, and a raster's rate follows its samples, so none of them has an established unit bound.

## 1.0.1 (2026-10-29)

### Patch

- capped_torus and solid_angle now use only the direction of their [sin, cos] angle vector, so a vector off the unit circle no longer moves the surface or returns NaN, and capped_torus stays finite on its own centreline.

## 1.0.0 (2026-10-03)

### Major

- Open Source Launch
