# CLAUDE.md

Instructions for Claude Code working in `emergent-matter-sdm-core`.

## Code standard

Follow [STYLE.md](STYLE.md) at the repo root for all code, comments, tests,
and docs. It is the org standard and wins over habit. Pull request
descriptions follow
[.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).

## What this repo is

`sdm-core` is the Software Defined Matter core Python library: the `.sdm`
part format, a JSON-serializable SDF geometry DSL, and a JAX-traceable
evaluation core.

The domain model, in one block:

```
  Part             - top-level object consisting of:
      Param            - a scalar design variable (free or fixed)
      MaterialRegion   - material identity + SDF sub-tree
      CouplingNode     - interface port in local part coordinates
      Objective        - symbolic expression to minimise/maximise
      Constraint       - symbolic expression compared to an RHS
      metadata         - schema free list of additional characteristics
```

Do not re-explain SDM concepts in this repo's docs. The CEM, the `.sdm`
file, and the Define / Standardize / Evolve pipeline are documented once,
in the
[`emergent-matter-sdm`](https://github.com/EmergentMatter/emergent-matter-sdm)
documentation repo. Link there instead of restating it.

For what any given module or function does, read its docstring. That is
the reference, and prose docs must not restate it.

## Build and run

Run tests:

```
uv sync # installs pytest/ruff/mypy from the dev dependency group
uv run pytest
uv run pytest -n auto  # same suite across all cores. CI uses this.
```

Run examples (generate `.sdm`, preview, export):

```
uv run examples/build_example.py     # generate .sdm file
uv run examples/preview_example.py   # preview .sdm file
uv run examples/export_example.py    # export .sdm file to .stl
```

## Architecture

Package layout and data flow: [`docs/architecture.md`](docs/architecture.md).

## Technical context

The things that are easy to get wrong when changing this code.

- **Export** (skimage MC + trimesh, fail-loud) is manufacturing-fidelity and needs `uv sync --extra export`.
- **Optimization path ≠ mesh path.** JAX SDF compile + metric integrals for objectives; separate MC/cleanup pipeline for `.stl`/`.obj`/`.ply`. Do not merge.
- **BBox is the integration domain, not geometry bounds.** Stop-gradient; `mode="bounds"` vs `"values"` trade trajectory stability vs tightness. Unbounded/prickly nodes (`gyroid`, param-dependent rotation) need explicit `metadata["bbox"]`. `plane` only valid as a CSG cutter, see `sdf/validate.py`.
- **Materials are external** (`emergent_matter_materials`); this repo references the material IDs and name only.
- **`schema_version` is the minimum version a tool must support to read the document**, not the version of `sdm-core` that wrote it. `min_schema_version_for` computes it from content: a part with a non-delta param `prior` requires `0.3`, every other part gets the `0.2` floor (`kinematics` arrived at `0.2` and is asserted for every part, not checked per part). It is a minimum requirement, not "the oldest schema that would accept this file" -- a simple part validates against `sdm-0.1.schema.json` and still declares `0.2`, so `0.1` is read but never written. The value therefore changes when content changes, including downward. Reading the field as provenance is the confusion `docs/adr/0001-sdm-wire-contract.md` exists to close.
- **Cutting a new schema version** follows [docs/schema-versioning.md](docs/schema-versioning.md): what warrants a version, the order to change things in, and why a released schema file is never edited.
- **JSON Schema ≠ semantic validation.** The JSON Schema is structural and versioned, dispatched on a document's declared `schema_version`. Semantic validation (does a primitive/modifier/deform kind exist, are its keyword arguments correct, does every `$ref` resolve) is version-independent, registry-driven against `wire.py`, and lives in `sdf/validate.py` (`SemanticValidationError`) alongside the `plane`-as-cutter rule; it runs on a document of any declared version. Bbox inference is a separate layer again.
- **JAX DSL:** `bind_sdf` captures `free_vec` once for JIT stability.

## Naming conventions

Shared baseline: see [STYLE.md](STYLE.md) -- Python/JAX `d_` / `n_` / `b_` prefixes on typed scalars where helpful, and functions named by return type (`extract_mesh`, `eval_grid`). Output filenames follow the org's ISO 8601 stamp convention (see `export_part`).

SDM-specific additions (not in STYLE.md):

| Pattern | Meaning | Examples |
| -------- | ------- | -------- |
| `sdf_*` builders | JSON-serialisable SDF DSL nodes | `sdf_primitive`, `sdf_op`, `sdf_transform`, `sdf_deform` |
| `expr_*` builders | Objective/constraint expression nodes | `expr_metric`, `expr_param`, `expr_binop` |
| `op_*` | CSG combinators (Inigo Quilez style) | `op_union`, `op_subtract`, `op_smooth_union` |
| `make_*` / `bind_*` / `infer_*` | Factory / compile / inference by return type | `make_sdf_closure`, `bind_sdf`, `infer_sdf_bbox` |
| `export_part` / `preview_part` | Top-level entry points named by what they produce | `export_part`, `preview_part` |
| PascalCase classes | Domain model | `Part`, `Param`, `MaterialRegion`, `CouplingNode` |
| snake_case JSON keys | Wire format (`.sdm`) | `sdf_tree`, `schema_version`, `material_id` |
| `_meshing/` | Private mesh extraction (leading underscore) | Not part of public API |
| `grid_sampling/` | Public grid evaluation of compiled SDFs | `bind_sdf`, `make_grid`, `eval_chunked`: no `[export]` extra |
| `p_*` in examples | Local `Param` objects (informal) | `p_outer_r`, `p_height` in `examples/build_example.py` |

SDF leaf evaluators follow the Quilez/community convention: `p` = query point, `r` = radius, `b` = box half-extents, `d1`/`d2` = signed distances in CSG ops.

## Documentation layout

`docs/` is grouped by how a file ages, and
[`docs/README.md`](docs/README.md) is the index. In short:

- **Recording a decision** and its reasoning: write an ADR at
  `docs/adr/NNNN-kebab-title.md` with `Status` / `Context` / `Decision` /
  `Consequences`.
- **Explaining why** the code is shaped as it is, where that reasoning is
  too broad for one docstring: extend a reference page under `docs/`.
  What a function does belongs in its docstring, not in prose.
- Neither of those: it belongs in a GitHub issue, not in `docs/`.

A document that reads as a proposal means a decision has not been recorded
yet. Convert it once the decision is made.

## Org playbook

This project follows the EmergentMatter engineering playbook for org-wide
patterns, conventions, and the repo catalog. When in doubt, check it before
inventing a new pattern; where its content overlaps this file,
[STYLE.md](STYLE.md) is the authoritative copy for this repo.
