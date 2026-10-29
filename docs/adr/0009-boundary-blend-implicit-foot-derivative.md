# 0009. `boundary_blend` differentiates its foot point implicitly only on request

## Status

Accepted (2026-10-29)

## Context

`boundary_blend` has no closed-form distance. For each query point `q`,
`sdf_boundary_blend.signed_distance` searches the skin `S(u, v)` for the
foot of the perpendicular (the point whose connecting line meets the skin at
a right angle). It starts from the nearest of a fixed grid of samples and
finishes with damped Gauss-Newton steps. The value is the gap from that foot
to `q`, measured along the skin's normal there.

The search is a loop, so differentiating through it is neither exact nor
cheap. There are two ways to give the foot a derivative.

- Hold the foot's `(u, v)` fixed (`jax.lax.stop_gradient`). Sliding the foot
  along the skin does not change the distance to first order, so the value's
  derivative with respect to a design parameter (`-n . dS/dparam`) and the
  spatial gradient (the unit normal) are exact. What depends on where the
  foot is comes out wrong: the derivative of the normal with respect to a
  parameter, and the spatial Hessian, which is exactly zero instead of the
  skin's curvature.
- Differentiate the foot with the implicit function theorem. The foot is a
  zero of `F(u, v) = J^T (S(u, v) - q)`, the gradient of `|S - q|^2 / 2`, so
  under differentiation it moves by `-H^-1 dF` with `H = dF/d(u, v)`. That
  makes the normal's derivative and the Hessian exact too, at the price of a
  2 x 2 solve per point and the skin's second derivatives inside every
  gradient.

The metrics in sdm-core (`volume`, `surface_area`) and meshing read only the
value and the gradient's length. An objective based on the normal's
direction (an overhang angle, a draft check) needs the second kind.

Measured on the saddle seat in `examples/boundary_blend_differentiable.py`
(6 parameters, 4 mm metric voxels, CPU limited to 4 cores), implicit against
held:

| | compile | run |
|---|---|---|
| `volume` | 1.0x (44 s) | 1.0x |
| `surface_area` | 1.5x (141 s) | 1.7x |
| `jax.grad(volume)` | 2.4x (747 s) | 1.9x |
| `jax.grad(surface_area)` | more than 1.7x (over 860 s) | |

The two give the same first derivative of the value and the same gradient
length, so the metrics' gradients are the same in either mode.

## Decision

The foot is held fixed by default. `make_sdf_closure(...,
b_implicit_foot=True)` (and `make_sdf_closure_with_binding`) differentiates
it implicitly for every `boundary_blend` in the tree. The option is passed
down the compile walk to `compile_boundary_blend(..., implicit_foot=True)`
and `signed_distance(..., implicit_foot=True)`. It is an evaluation choice,
not part of the geometry, so it is not on the wire.

With the option on, the foot goes through `jax.lax.custom_root`
(`sdf_boundary_blend._implicit_foot`):

- Plain evaluation is unchanged. `custom_root` returns the search's result,
  and the solve runs only when a derivative is taken.
- `H` enters as a value only (`stop_gradient`). Every term that would need
  its derivative is multiplied by `J^T n` or `(q - S) . dn`, both zero at a
  foot, so first and second derivatives stay exact. Differentiating `H` as
  well would make XLA compile fifth derivatives of the Gordon surface for a
  second-order quantity.
- A coordinate the search stopped at a bound of its `(u, v)` box is held
  there. A small ridge on `H` keeps the solve finite at a 3-curve pole,
  where `S_u` vanishes. Where `H` is not finite the foot is held fixed.
- The foot's tangent is a real array, often zero, not a symbolic zero, so
  JAX evaluates the skin's higher derivatives at the foot even where they
  end up multiplied by zero. The Gregory corner-twist blend in
  `boundary_blend_patch` therefore takes the mean of the two twists at a
  patch corner through a safe denominator, so that its derivatives stay
  finite there and inf times zero cannot turn into NaN. The surface and its
  first derivatives at the corner do not depend on the twist, so the skin
  does not move.

`test_second_derivatives_follow_the_sliding_foot` checks the normal's
derivative and the spatial Hessian against finite differences on a Coons
patch with the option on.

## Consequences

- Every existing caller keeps the cheaper gradients, and they are exact for
  everything the metrics and meshing read.
- A caller that needs the normal's derivative or the Hessian has to ask for
  it. Without the option those quantities are wrong away from the skin and
  nothing reports it.
- With the option on, the normal's derivative and the spatial Hessian are
  exact wherever the foot is a strict local minimum of the distance. On the
  medial axis (a point equidistant from two feet) `H` is singular and the
  derivative is large, as it is for any distance field there.
- In both modes, a design that moves far enough to change the skin's
  topology still needs re-checking (`check_boundary_blends`). The
  derivative describes the plan the part was compiled with.
