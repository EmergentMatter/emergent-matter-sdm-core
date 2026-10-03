# Sweep consumer contract

These fixtures ship with core so hosts can test the same authored geometry and
expected outputs. `expected.json` records payload sizes, live profile values,
strict sample distances, and explicit near-tie candidates.

`inline.sdm` is a cylindrical helix with an asymmetric box profile. Changing
`half_width` updates its uniform without rebuilding its baked path frames.
`tabled.sdm` contains separated straight sweeps sharing a live radius. Their
combined point count crosses the table threshold; the later header and frame
records cross texture row boundaries. Hosts must upload every RGBA texel and
bind the sweep sampler in every program that evaluates the scene.

## Numerical acceptance

At the strict points, compare both distances (absolute tolerance `1e-5` mm) and
inside/outside signs for every `live_cases` value using the same emission.
The near-tie cases are a separate contract, not discarded samples or a relaxed
distance tolerance. Compute squared distances to all sampled path segments and
retain candidates within `relative_squared_distance_margin * (1 + minimum)`.
The reported distance must match one of those candidates within
`distance_tolerance`; the fixtures also require all candidates and both backends
to agree with the declared `inside` value.

JAX and GLSL both select the first strictly minimal segment using their own
floating-point arithmetic. Near a shared endpoint their rounding can select
different segments, whose profile frames can produce different distances.
This contract acknowledges that discontinuity; it does not promise bitwise
agreement, globally identical distance fields, or occupancy agreement for every
possible path and query. No tie-breaking rule in the production kernels changes.

Core executes both storage forms through `tests/_glsl_runtime.py`, with a
nonzero sampler unit. Consumer implementations should run these fixtures through
their own texture upload, translation, and scene evaluation paths as well.
