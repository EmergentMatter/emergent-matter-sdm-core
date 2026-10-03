# Consumer conformance corpus

The `ownership/` subdirectory covers the material-aware rigid adapter rather
than the ordinary region-solid emitter. Its authored fixture and expected
distances are consumed by core's `test_rigid_materials.py` and the web viewer's
`test_material_ownership_rendering.py`. It intentionally places material beyond
the classifier and tests nonzero motion; a region-only render cannot pass.

Portable `.sdm` fixtures plus expected **structural / numerical fingerprints**
for hosts that consume this package's emitter (web viewers, CEMs, other
language reimplementations). Schema `valid/` / `invalid/` only prove a
document is accepted or rejected; this tier proves the *emission contract*
did not drift.

Ships inside the wheel next to `valid/` and `invalid/`. Discover it the same
way:

```python
from importlib.resources import files

consumers = files("software_defined_matter.schema.conformance.consumers")
```

## Layout

| fixture | what it pins |
|---|---|
| `field_radial_amplitude.sdm` | radial displace field; non-default `amplitude` |
| `field_sin_xyz_amplitude.sdm` | `sin_xyz` displace field; non-default `amplitude` |
| `field_angular.sdm` | angular field (JAX); GLSL emission refused until `feat/angular-field-emission` |
| `param_expr.sdm` | `Param.expr` derived diameter expands over a live uniform |
| `polygon_inline.sdm` | polygon below the table threshold (inline `vec2 V[...]`) |
| `polygon_table.sdm` | polygon above the table threshold (`poly_table` + `sdm_polygon_2d_tab`) |
| `components_union.sdm` | labelled root-union components |

Each fixture has a matching `expected/<stem>.json` fingerprint:

- `emission` — entry point, lib helpers referenced by the scene, component
  labels/machines, polygon payload sizes, control classes, scene needles,
  required/forbidden macros. `status: "refused"` fixtures assert the
  refusal substring instead of an emission.
- `numerical` — sample points and JAX field values (where applicable).
- `roundtrip` — whether load/save and the GLSL CLI must round-trip.
- `pending` — named upstream gaps this fingerprint documents.

## Amplitude pending

`field_*_amplitude` fixtures author `amplitude` on the wire. On this tip the
emitter still looks up `amp` and falls back to `1.0`, so the call site is
wrong. The fingerprint records:

- `observed_on_this_tip` — what this tip emits today
- `correct` — the contract (`amplitude` must reach the call site)
- `assert: "correct"` with `xfail_until: fix/field-amplitude-emission`

`tests/test_consumer_conformance.py` xfails that call-site check until the
fix lands; structural and JAX fingerprints still run green.

## How the gate runs

`tests/test_consumer_conformance.py` loads every fixture through
`importlib.resources`, checks load/save equality, matches the emission
fingerprint, checks JAX samples, and for CLI-enabled fixtures asserts
`python -m software_defined_matter.glsl` writes the same `meta.json`
polygon/component payload as `emit_glsl`.

Downstream hosts should run the same fixtures through their own pipeline
and fail by fixture name when a fingerprint drifts.

The `sweep/` subdirectory supplies inline and tabled sweeps, live profile
fingerprints, and explicit near-tie candidate acceptance. See its README for
numerical tolerances and the distinction between strict samples and near ties.
Core executes these fixtures in `tests/test_sweep_consumer_conformance.py`.
