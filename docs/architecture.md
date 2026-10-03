# Architecture

Where `sdm-core` sits, and how its directories divide the work. For the
concepts behind the pipeline (CEM, `.sdm`, the Define / Standardize /
Evolve loop) see
[Concepts](https://github.com/EmergentMatter/emergent-matter-sdm/blob/main/docs/concepts.md)
in the `emergent-matter-sdm` documentation repo.

This page is deliberately at directory granularity. What each module
contains, and what each function takes, lives in the docstrings next to
the code, which are reviewed alongside it.

## Package layout

```text
src/software_defined_matter/
  model.py         # the domain model: Part and its nested types
  wire.py          # the declarative wire-format contract, JAX-free
  io.py            # load / save / validate
  sample.py        # Monte Carlo sampling and propagation over param priors
  process.py       # manufacturing process-profile prior resolution
  audit.py         # gap audits: declared clearance vs measured SDF gap
  bbox.py          # numeric bbox tightening
  sdf/             # the SDF runtime: primitives, CSG ops, transforms,
                   #   the tree-to-JAX-closure compiler, analytic bbox
                   #   inference, and semantic validation against wire.py
  dsl/             # $ref resolution and the objective/constraint
                   #   expression compiler
  objectives/      # the differentiable metric registry
  grid_sampling/   # public grid evaluation of a compiled SDF
  _meshing/        # private mesh extraction shared by preview and export
  preview/         # display-only PyVista isosurface
  export/          # manufacturing-fidelity mesh export, [export] extra
  glsl/            # GLSL ray-march emitter for interactive viewers
  schema/          # versioned JSON schemas generated from wire.py, a
                   #   no-import channel to read them and the version
                   #   index, and a conformance fixture set
```

Data flows `.sdm` to `io.load` to `Part` to SDF compile, then out to
either metrics (the optimization path) or `_meshing` (the mesh path).
Those two paths are deliberately separate; see
[ADR 0002](adr/0002-preview-and-export-modules.md).

Material properties live in the separate `emergent_matter_materials`
package, not here. See [Install](../README.md#install) for how this
repository depends on it.

## What this repository is for

`sdm-core` is the source-of-truth format and the differentiable
evaluation layer that every downstream optimizer, exporter, and
simulation tool builds on. That shapes two design commitments:

- **Geometry stays executable, not merely descriptive.** SDF trees in an
  `.sdm` compile to JAX-callable closures, so geometry can be sampled,
  differentiated, and optimized directly rather than being re-derived
  from a mesh.
- **Optimization intent is data, not code.** Objectives and constraints
  are expression trees in the file, so a consumer reads what a part is
  trying to achieve instead of inferring it.

Together those are what let an external tool consume the same artifact
consistently: `.sdm` is schema-validated JSON, and evaluation is pure
Python and JAX.

## Current boundaries: what is intentionally outside core

These are left to downstream repositories and services, so that core can
stay stable while the layers above it move:

- Optimizer orchestration: algorithm selection, multi-run scheduling,
  convergence management
- High-fidelity multiphysics solving, with FEA/CFD/EM as external
  back-ends
- Manufacturing toolpath generation and machine-specific export
- CAD and scene authoring UX. Host tools consume the `.sdm` source of
  truth; they do not replace it

Planned downstream modules, outside this repository: `.sdm` authoring
helpers from natural language or existing CAD, optimizer orchestration,
and report generation.
