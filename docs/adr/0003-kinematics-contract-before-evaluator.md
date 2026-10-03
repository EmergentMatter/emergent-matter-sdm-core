# 0003. Pin the kinematics contract before building the evaluator

## Status

Accepted

The `kinematics` block ships in `sdm-0.2.schema.json`. The bundled example
is [`examples/notch_hinge_kinematics.sdm`](../../examples/notch_hinge_kinematics.sdm).
No evaluator ships with it. That gap is deliberate and is the subject of
this record.

## Context

An `.sdm` document describes geometry, params, materials, couplings,
objectives, and constraints: everything about a part *at rest*. It said
nothing about how a part **moves**.

So every consumer that wanted to animate a flexure or a joint
re-implemented the deflection in its own code, and those implementations
drifted. Two viewer scripts for the same downstream CEM already disagreed
about which modules they load.

This is the missing complement to `CouplingNode`. Couplings say *where*
parts connect (interface ports). Nothing said *how* degrees of freedom
move the geometry.

The ordering question is the real decision here. An evaluator, a model
binding, and a host-app animation panel are all larger and slower to build
than the data format they would share, and each consumer was already
writing its own. Shipping the evaluator first would have meant every
consumer waiting on `sdm-core`, then migrating twice.

## Decision

**Pin the data contract first, in schema 0.2, and build the evaluator
against it afterwards.** Animation becomes data rather than code:
versioned, diffable, and language-agnostic, so a Blender viewer, a
simulation, and a notebook can all ingest one graph at one version with no
per-CEM animation code.

A part's `kinematics` is a small graph:

- **dofs**: named degrees of freedom (`angle` or `length`) with a required
  `range`, a strict `unit` (`rad`/`deg`/`mm`), a `default`, and an
  optional nominal `rate`. `range` is required so a viewer can auto-build
  a per-DOF scrub animation with no authoring; a DOF without one gives the
  auto-builder nothing to sweep. These are distinct from optimization
  `params`, and are referenced in motion expressions through the expression
  node `{"type": "dof", "name": ...}`.
- **bodies** (nodes): rigid regions, each with a `region` defining
  membership and a `motion`, an ordered list of DOF-parameterized rigid
  ops (`rotate` about an axis by an `expr`, `translate` along an axis by
  an `expr`). No ops means fixed, that is, ground.
- **flexures** (edges): compliant regions between two bodies, each with a
  `region`, a `from_body`, a `to_body`, and a `blend`, a scalar field in
  `[0, 1]` reusing the existing `field` vocabulary. A flexure's transform
  interpolates the two bodies' motions by `blend(p)`.

**Regions are views, never instances.** Any `sdf` node may carry an
optional `name`, and a `region` is either `{"$node": "<name>"}`, a view
onto that named subtree in the material trees evaluated in situ with its
ancestor frame applied, or an inline SDF tree for regions that are not
part geometry (a clearance zone, say).

A view duplicates no geometry, so editing a node propagates to every
region that views it, with no intra-file drift. `$node` refs are legal
only where a region is expected, never as geometry meaning "paste subtree
X here". Instancing is where named references stop being tractable: a
nested node's meaning depends on its ancestor transform chain, shared
subtrees turn the tree into a graph, and node edits acquire an
edit-the-source-or-the-copy ambiguity. Membership views have none of those
problems, because ownership resolves during traversal by construction.

**Ownership is nearest-region.** A point belongs to the body or flexure
whose `region` SDF is smallest. This matches how analytic flexure code
already classifies vertices (argmin over region SDFs) and is robust near
an axis.

## Consequences

- Consumers can author and validate `kinematics` blocks today, and the
  bundled kinematics example exercises the contract. What they
  cannot do is ask `sdm-core` to evaluate one.
- `Part` has no `kinematics` field, because the block is a wire-format
  concept produced by external generators rather than by this package's
  domain model. A reader of the domain model alone will not find it,
  which is a real discoverability cost of deciding the contract first.
  `wire.py` now declares it explicitly as a `wire_only` field, which
  narrows that gap and is what lets the generated version index report
  what 0.2 added; see
  [ADR 0001](0001-sdm-wire-contract.md) for why that flag exists.
- Because the schema version floor for `kinematics` is 0.2, any document
  carrying one declares at least 0.2, and a 0.1-only reader correctly
  rejects it rather than silently ignoring the motion.
- The `blend` field wants a primitive `field` kind, an `axis_ramp`
  (`value = clip((p·axis - lo) / (hi - lo), 0, 1)`). It is not in the
  schema's enumerated kinds. Field kinds are open strings today, so such a
  document validates, but it should be added to the field-kind registry
  when that lands. This is the one place the contract is under-specified.
- Cross-part kinematics, driving DOFs through `CouplingNode`s to assemble
  machines, is out of scope. The contract is single-part, and extending it
  later is an additive schema change.
- The longer arc this enables is node-level editability: stable names let
  any node be addressed, modified, and everything downstream re-derived
  (compiled closures, emitted GLSL, regions, animations) from one
  authoritative tree, where structural child-index paths would not survive
  regeneration.

## Remaining work

Recorded here because the contract shipped without it: the `model.py`
binding (`Dof` / `Body` / `Flexure` dataclasses with `to_dict` /
`from_dict`), the JAX evaluator, and the `axis_ramp` field kind.
