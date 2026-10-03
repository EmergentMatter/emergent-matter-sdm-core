# 0008. `pyvista` moves into an optional `preview` extra

## Status

Accepted (2026-09-24)

## Context

`sdm-core` is about to be published as a public wheel (open tier, on the
package index). `pyvista` has been a hard dependency since before
[ADR 0002](0002-preview-and-export-modules.md), which explicitly kept it
that way while making `scikit-image` and `trimesh` optional.

That made sense while every consumer of `sdm-core` was an internal repo
that wanted the preview path anyway. It does not make sense for a public
package: `pyvista` pulls in VTK, matplotlib and cyclopts, none of which an
optimizer, an API service, or CI needs. `software_defined_matter.preview`
is a display-only convenience (see ADR 0002), the exact category ADR 0002
already put `export`'s dependencies in, for the same reason.

## Decision

**`pyvista` moves to an optional `preview` extra**, following the same
shape `export` already uses: a module-level `try`/`except ImportError`
import guard in `software_defined_matter/preview/__init__.py`, a
`_require_preview_deps()` helper that raises a clear install hint (`pip
install 'emergent-matter-sdm-core[preview]'`), and `preview_part` calling
it before touching `pyvista`.

`pillow`'s CVE floor (pinned since GHSA advisories fixed in 12.3.0) moves
with it: it was only ever a transitive pin for `pyvista -> matplotlib` and
`scikit-image -> imageio`, never imported directly, so it is duplicated
into both the `export` and `preview` extras rather than kept as a
top-level dependency nothing in a bare install actually needs.

The `dev` dependency group gains `pyvista` alongside the `export` extras
it already duplicates, for the same reason stated in `pyproject.toml`:
`uv sync --locked` in CI installs `[dependency-groups]`, not
`[project.optional-dependencies]`, so the preview test suite
(`tests/test_preview.py`, `tests/test_preview_assembly.py`) would
otherwise silently skip via `pytest.importorskip("pyvista")` on every CI
run instead of actually exercising the preview path.

## Consequences

- A bare `pip install emergent-matter-sdm-core` no longer pulls VTK,
  matplotlib, or cyclopts. `import software_defined_matter` and the
  optimization/export paths are unaffected.
- `preview_part()` and `python -m software_defined_matter.preview` raise
  a clear `ImportError` with an install hint, not a bare "no module named
  pyvista", when the extra is missing.
- This is a breaking change for anyone who was relying on
  `pip install emergent-matter-sdm-core` alone to make `pyvista` available
  transitively, without declaring it themselves or asking for `[preview]`.
  That is why this ships as a major-version bump rather than a minor one:
  `sdm-core`'s own Python API is unchanged, but the install contract for a
  fresh, unextra'd install is not.
- `docs/architecture.md` and any other reference to "`preview` needs no
  extra" should be read against this ADR, not ADR 0002, from this point
  on.
