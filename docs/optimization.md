# Optimizing with `sdm-core`

How to run gradient-based design loops against a `.sdm` part: what is
differentiable, which optimizers suit which geometry, and how the metric
sampling domain (the bounding box) works.

Moved here from the root `README.md`, which now links to this page.

## Differentiability

`sdm-core` is designed to support gradient-based design loops with JAX (`jax.grad`, `jax.jit`) over free parameters. In practice, SDF programs are usually **piecewise differentiable**: gradients are well-defined almost everywhere, with non-smooth points at CSG seams, sharp edges, clamps, and absolute-value transitions.

### What is differentiable in this library?

- **JAX-traceable compiler path**: `software_defined_matter.sdf.compile.make_sdf_closure(...)` returns closures intended to be compatible with autodiff over parameter vectors.
- **Analytic primitives** (`sphere`, `box`, `capsule`, `capped_cylinder`, etc.): generally stable and useful for gradient optimization, except at expected geometric singularities (medial axis, corners, exact symmetry points).
- **Hard CSG ops** (`union`, `subtract`, `intersect`): implemented with `min`/`max`, so they are non-smooth at switching boundaries.
- **Smooth CSG ops** (`smooth_union`, `smooth_subtract`, `smooth_intersect`): provide softened transitions that are typically better behaved for first-order optimization.
- **TPMS / implicit lattices** (`gyroid`, `schwarz_`*, `neovius`, `lidinoid`): differentiable almost everywhere but include `abs(...)` level-set thickening, creating kinks on the zero level-set.

### Suitability by optimization strategy

- **Best for gradient-based methods (LBFGS, Adam, projected gradient):**
  - Primitive-heavy models with moderate transforms and smooth CSG.
  - Objectives evaluated from sampled fields/meshes that avoid binary thresholding in the inner loop.
- **Works, but may need care:**
  - Hard CSG-heavy trees with many contacts/seams.
  - Designs with repeating/clamping operators (`repeat_`*, explicit clamps), where gradient signals can be discontinuous across cell boundaries.
- **Often better with hybrid or derivative-free outer loops:**
  - Highly combinatorial topology changes, discontinuous penalties, or aggressive thresholding/post-processing in objectives.

### Practical recommendations

- Prefer `smooth_`* CSG during optimization; switch to hard CSG for final crisp geometry if needed.
- Start with a conservative smoothing radius (`k`) and reduce it progressively as optimization converges.
- Scale and bound parameters so gradients have similar magnitudes across dimensions.
- Use multi-start runs because non-convex SDF landscapes can trap local minima.
- Validate with finite-difference spot checks on critical parameters when introducing a new primitive/objective combination.

### Current limitations to be aware of

- Some primitives are explicitly documented as **bound** or **approximate** SDFs; they can still be optimized, but distance values are not exact everywhere.
- Non-smoothness at geometric events (feature creation/removal, seam crossing) is expected; optimizers may require smaller step sizes or line search.
- If your optimization depends on strict smoothness, prefer smooth CSG + regularized objectives and avoid hard threshold operations inside the loss.

### Bounding boxes (the metric sampling domain)

A bounding box in `sdm-core` is **not** what bounds the geometry. Your
`Param` values and bounds already do that. The bbox is the **numerical
integration domain**: every SDF-sampled metric (`volume`, `mass`,
`surface_area`, `centroid`, `bbox_extent`) estimates an integral on a voxel
grid laid down *inside* the box (`occ.mean() * |bbox|`). The box tells the
sampler *where* to look and supplies the normalisation. The headline example
on this page is a good illustration: it contains a `gyroid` infill, which is
unbounded, so its sampling domain cannot be derived from the geometry and
must be stated explicitly.

**When you need one:**

- SDF-sampled metrics in objectives/constraints (the list above).
- Mesh extraction / marching cubes and grid-based visualization.

**When you don't:**

- Pure-expression objectives/constraints (e.g. `outer_radius - inner_radius`).
  These never touch the SDF, so no box is resolved (resolution is lazy).
- Closed-form metrics, or external FEA/EM back-ends, which own their own
  (typically much larger, e.g. far-field air) simulation domain. Do not
  derive a physics domain from the SDF bbox.

**Two inference modes** (`software_defined_matter.sdf.bbox.infer_sdf_bbox`):

- `mode="bounds"` (default): the worst-case axis-aligned box over every
  `Param.bounds`, the smallest box guaranteed to contain the part for
  *every* point a bounded optimiser may visit. Trajectory-stable; loose when
  bounds are wide (a small instance then occupies a tiny fraction of the
  grid, starving gradient signal).
- `mode="values"`: a tight box at each param's *current* value. Used for
  per-iteration metric sampling by `compile_expr`, padded by
  `4·τ + k_smooth` so the soft-Heaviside band and CSG smoothing are not
  clipped.

**Key properties / tradeoffs:**

- *Stop-gradient by construction.* The box is a constant w.r.t. the
  free-parameter vector, never traced. This is required for correct shape
  derivatives: a parameter-dependent (traced) domain injects a spurious
  "domain is moving" term into the gradient.
- *Tightness is only per outer iteration.* Because the `values`-mode box
  reads current `Param` values while the SDF inside is traced, the optimiser
  must re-derive (re-run `compile_expr` after `Part.update_from_vector`) each
  step. A compiled fn reused across many steps keeps a stale (but, given the
  pad, still-enclosing) box; if you cannot re-derive per step, use
  `mode="bounds"` instead.
- *Tight + padded + stop-gradient beats a fixed worst-case box* for
  parametric optimization: same correct gradient, far better sample density
  on the actual part. The pad must exceed one optimiser step's boundary
  motion so the part stays fully enclosed and the volume estimate stays
  unbiased.
- *Sampling density is decoupled from box size.* Set `metric_voxel_size` so
  the *part* is sampled at constant resolution regardless of how loose the
  box is; otherwise a fixed `grid_resolution` is used. `metric_grid_cap`
  bounds peak memory.
- *Fail-loud on unbounded geometry.* `gyroid`/TPMS, `plane`,
  param-dependent rotation, etc. have no inferable box. Compilation raises a
  clear error rather than silently fabricating a domain (which would corrupt
  every integral metric with no signal). Supply one explicitly via
  `part.metadata["bbox"]`. An explicit override is used exactly (unpadded)
  and takes precedence over inference.

**Relevant `metadata` knobs.** These are `Part.metadata` keys rather than
function arguments, so they have no docstring of their own; this table is
a discovery aid. For the current default of any of them, read
`dsl/expr.py`, which is where they are consumed.

| key | effect |
|---|---|
| `bbox` | explicit `[[xlo,ylo,zlo],[xhi,yhi,zhi]]`; exact, unpadded, and takes precedence over inference |
| `metric_voxel_size` | target voxel size in mm, giving constant part sampling density |
| `grid_resolution` | per-axis sample count, used when `metric_voxel_size` is unset |
| `metric_grid_cap` | per-axis cap on sample count, as a memory guard |
| `metric_bbox_pad` | override the automatic `4·τ + k_smooth` pad |
| `metric_tau` | soft-Heaviside temperature, which also drives the automatic pad |
