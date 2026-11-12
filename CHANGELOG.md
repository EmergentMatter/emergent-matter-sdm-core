# Changelog

<!-- towncrier release notes start -->

## 1.1.0 (2026-11-12)

### Minor

- A box that `Part.infer_bbox_from_sdf` or `tighten_part_bbox` persists into `metadata["bbox"]` now carries a `metadata["bbox_basis"]` record of what it was inferred over, and sampling, export, metric compilation, shader emission and motion bounds set the box aside once the part has changed under it, so a stale box no longer cuts off a grown part. `tighten_part_bbox` writes the record only when it tightened inside a box derived from the part; tightened inside an authored or unlabelled box or `bbox_half_size`, the box keeps that label. An authored or unlabelled box is used as stored, as before. `sdf.bbox.stored_bbox` and `stored_bbox_status` expose the check. ADR 0010 records the convention.

### Patch

- Clarify when a sampled hard field may be retained as the geometry definition and which consumer checks remain application-owned.
- Raster spacing is checked where it is built and decoded: a string, a boolean, a nested or non-numeric entry, or a value that is not positive and finite now raises a ValueError naming spacing, instead of a bare TypeError or a silent conversion. A NumPy number or a 0-d numeric array is accepted as a scalar pitch.
- The consumer conformance README now lists the raster_field and kinematics_roundtrip fixtures, which were in the corpus but missing from its layout table.


## 1.0.1 (2026-10-29)

### Patch

- capped_torus and solid_angle now use only the direction of their [sin, cos] angle vector, so a vector off the unit circle no longer moves the surface or returns NaN, and capped_torus stays finite on its own centreline.


## 1.0.0 (2026-10-03)

### Major

- Open Source Launch
