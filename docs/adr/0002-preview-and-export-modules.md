# 0002. `preview` and `export` modules, over one shared sampler

## Status

Accepted (implemented 2026-05-19)

## Context

`sdm-core` could describe a part (`.sdm` to `Part` to per-material SDF
trees) and compile each SDF to a JAX-traceable closure, but it had no
first-class way to do either of the two things an author needs constantly:

1. look at a part quickly while authoring or debugging SDF trees, and
2. get a manufacturable mesh out of a part, at a fidelity tied to the
   additive process.

The only viewer was the example script
[`examples/preview_example.py`](../../examples/preview_example.py), and
there was no export path at all.

That example script also carried a live correctness bug, and it is the
reason this decision is shaped the way it is. Sampling an SDF at
`(max - min) / (N - 1)` spacing (`np.linspace`) while telling marching
cubes the spacing is `voxel_size` places vertices off the true surface: on
a radius-5 sphere at voxel 0.5, up to 0.6 mm off. The fix is `np.arange`
with exact spacing.

The general form of that bug is the real risk. If `preview` and `export`
sample the SDF differently, the preview is lying about what will be
manufactured, and nothing in the codebase would catch it.

A consolidated grid-eval, marching-cubes, cleanup, export pipeline already
existed in a downstream bearing-mechanism project (not part of this repo),
whose own module docstring anticipated extraction "once a second part
family adopts sdm-core."

## Decision

**One sampler, shared by every consumer.** `preview` and `export` share a single
private meshing foundation and diverge deliberately only in their
defaults, their polygoniser, and the guarantees they make, never in how
grid points are placed in world space.

```
software_defined_matter/
  _meshing/            # private foundation, NOT public API
    types.py           #   BBox3, MeshData, MeshCleanupConfig
    grid.py            #   np.arange grid + chunked JIT eval
    mesh.py            #   skimage MC + fail-loud validate + cleanup
    bind.py            #   Part/MaterialRegion -> 1-arg SDFFunc + bbox
  preview/             # fast, in-memory, display-only (PyVista)
  export/              # manufacturing fidelity (skimage MC + trimesh)
```

`_meshing` is private: it is an implementation detail shared by the two
public modules, not a supported surface.

**Port the downstream pipeline rather than depend on it.** The source
lives in a consumer of this repo, so depending on it would point the
dependency arrow backwards (core to consumer). `grid`, `mesh`, and `types`
were ported field-for-field, preserving the exact-spacing sampler and the
fail-loud behaviour, with one new file (`bind.py`) adapting `sdm-core`'s
closure shape. Extracting a shared `emergent-matter-sdf-mesh` library
remains a separate org decision, out of scope here.

**Export dependencies are optional.** `scikit-image` and `trimesh` sit in
an optional `[export]` extra, imported lazily with a clear install hint on
`ImportError`. This keeps the core lean for optimizer and CI use, and
preview-only users never pull `trimesh`. `pyvista` was a hard dependency at
the time this decision was made; see
[ADR 0008](0008-pyvista-preview-extra.md) for why that changed.

**`export` resolves material overlaps by default.**
`resolve_overlaps=True` produces priority-resolved exclusive shells,
honouring the domain model's "later entries override earlier ones"
semantics, with `resolve_overlaps=False` as the opt-out.

**The two modules are tested at different levels,** because they make
different promises. `export` is tested geometrically: a sphere of radius
*R* at voxel *v* must have every vertex within tolerance of *R* (this is
the direct guard against the spacing class of bug), meshes must come out
watertight with the right component count, and two overlapping materials
must yield exclusive shells sharing no voxels. `preview` gets a smoke test
only: non-empty surfaces, no exception, correct material count. It is a
display artifact, so a full geometric regression there would be wasted
cost.

## Consequences

- The preview and the exported mesh cannot silently disagree about where
  the surface is. That property depends on both modules continuing to go
  through `_meshing/grid.py`; a future module that samples the SDF its own
  way reintroduces the risk this decision exists to remove.
- `examples/preview_example.py` becomes a thin shim over `preview`, which
  removes its live spacing bug rather than leaving a second sampler in the
  tree to drift.
- `sdm-core` carries a ported copy of code that also lives in a downstream
  project. The two can diverge, and there is no mechanism here that
  detects it. That cost was accepted to keep the dependency direction
  correct.
- Installing `sdm-core` without the `[export]` extra gives a working
  library and a working `preview`, and `export_part` fails with an install
  hint rather than an import traceback. (Superseded for `preview` by
  [ADR 0008](0008-pyvista-preview-extra.md): `preview` now needs its own
  `[preview]` extra too.)
- `_meshing` being private means its API can change without a version
  bump. Anything that needs to reach into it is a signal that something
  belongs in the public surface instead.
