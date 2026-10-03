"""Every primitive must return a signed **distance**, not just a correctly
signed number.

An SDF promises that its value at a point is how far that point is from the
surface. That has a testable consequence: move one millimetre through space and
the value can change by at most one millimetre. Measuring the exact gradient
magnitude therefore gives 1.0 for a correct primitive.

    rate ~1.00   exact
    rate <1.00   under-reports the distance -- safe, merely conservative
    rate >1.00   OVER-reports -- claims the surface is further away than it is

Only the rate **near the surface** matters for correctness. A high rate far
outside the solid, or deep inside it, is harmless: those cells are wholly empty
or wholly full either way. ``ellipsoid`` and ``annular_sector`` look alarming on
a whole-volume maximum (8.7 and 22.1) yet measure 1.000 at the surface, and are
fine.

Why it matters: callers are entitled to reason from the distance. Two do so
today. ``objectives.metrics`` will skip any voxel whose centre reports a
distance greater than the cell's half-diagonal, because no surface can be
inside it -- on an over-reporting primitive that silently discards material.
And ``audit.py`` computes clearances as ``min(d_A + d_B)``, which only holds for
exact SDFs.

Sampled fields are audited separately: their raw interpolants need not be
exact distances. Their authored step factors must bound the measured gradient
both near the surface and across the sampling domain.

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter.sdf.compile import _PRIMITIVES
from software_defined_matter.sdf.raster import decode_raster_values
from tests.conftest import PRIMITIVES, PRIMITIVES_2D

# A primitive is allowed to exceed 1 by this much before it counts as
# over-reporting. The exact primitives all measure 1.000; the margin absorbs
# sampling luck, not real error.
RATE_TOL = 1.02

# Fraction of samples nearest the zero level set that count as "at the surface".
NEAR_SURFACE_FRACTION = 0.02

# Rank-by-|d| is meaningless if those ranks never reach a zero crossing (a
# fully solid TPMS sheet is the usual cause). The median |d| of the near set
# must be well inside the sampling box.
NEAR_MEDIAN_FRAC = 0.05

N_SAMPLES = 40_000
SEED = 0

# ``plane`` is deliberately absent from the shared PRIMITIVES table (it is
# infinite, so it cannot mesh and has no bbox), but it is an exact distance
# function and worth pinning down here.
EXTRA_3D = {
    "plane": ({"n": [0.0, 0.0, 1.0], "h": 0.0}, 11.0),
}

ALL_3D = {**PRIMITIVES, **EXTRA_3D}

# Sampling half-extent overrides.
#
# A TPMS is clipped to a box of half-extent ``0.5 * n_periods * period``, so the
# solid has TWO surfaces: the lattice sheet, and the six flat faces of the clip
# box. The box faces come from ``sdf_shapes.box`` and are exact. Sampling beyond
# the clip box therefore fills the "nearest the surface" set with box-face
# points measuring 1.000 and hides whatever the lattice itself is doing: the
# defect under test. Stay inside the clip box so the sheet is what gets
# measured.
SAMPLING_HALF = {
    "gyroid": 9.0,
    "schwarz_p": 9.0,
    "schwarz_d": 9.0,
    "neovius": 9.0,
    "lidinoid": 9.0,
}

# The shared mesh table keeps thick TPMS sheets so the meshed solid stays
# frozen against the pre-#108 geometry. At those thicknesses gyroid,
# schwarz_d and lidinoid fill the clip box, and rank-by-|d| never sees a
# lattice surface. The distance audit uses a thinner wall on those defaults
# (old thickness=0.4 migrated at period=10 for C=sqrt(3) is 0.7352).
# fmt: off
DISTANCE_KWARGS = {
    "gyroid":    {"period": 10.0, "min_thickness": 0.7352, "n_periods": [2, 2, 2]},
    "schwarz_d": {"period": 10.0, "min_thickness": 0.7352, "n_periods": [2, 2, 2]},
    "lidinoid":  {"period": 10.0, "min_thickness": 0.5, "n_periods": [2, 2, 2]},
}
# fmt: on

# Extra parameter sets for primitives whose distance behaviour depends on their
# own parameters. The shared conftest table carries one representative set,
# chosen for meshing; these push into the regimes where a defect actually
# shows. The TPMS error scales as 1/period, so a single long-period sample can
# sit near 1.0 by coincidence and prove nothing.
STRESS_VARIANTS = {
    name: [
        ({"period": 4.0, "min_thickness": 0.5, "n_periods": [5, 5, 5]}, 9.0),
        ({"period": 2.0, "min_thickness": 0.25, "n_periods": [9, 9, 9]}, 8.0),
    ]
    for name in ("gyroid", "schwarz_p", "schwarz_d", "neovius", "lidinoid")
}
STRESS_VARIANTS.update(
    {
        "bellows": [
            ({"outer_r": 5.0, "inner_r": 1.0, "period": 2.0, "n_periods": 3}, 8.0),
        ],
        "serpentine": [
            (
                {
                    "amplitude": 2.0,
                    "wavelength": 2.0,
                    "beam_width": 0.4,
                    "beam_height": 1.0,
                    "n_periods": 3,
                },
                6.0,
            ),
        ],
        "ellipsoid": [
            ({"r": [10.0, 1.0, 1.0]}, 12.0),  # extreme aspect ratio
            ({"r": [10.0, 10.0, 1.0]}, 12.0),  # flat disc
        ],
        "capped_cone": [
            ({"h": 8.0, "r1": 8.0, "r2": 0.0}, 12.0),  # degenerate tip
        ],
        "helix": [
            # The wall-rate defect scales as (r/major_r) * sin(lead)^2, so the 5.2
            # deg thread in the conftest table cannot see it: it measures 1.001
            # there and 1.06 / 1.17 here. Both of these sample INSIDE the band
            # (half_h = 45) and close to the winding radius on purpose: rank-by-|d|
            # on a thin wire in a wide box fills the "nearest 2%" with far-field
            # points (median |d| = 1.3 against r = 1.0 at half=24) and measures
            # nothing at all.
            (
                {
                    "major_r": 5.0,
                    "pitch": 30.0,
                    "r": 1.0,
                    "n_turns": 3.0,
                    "phase": 0.0,
                    "handedness": 1.0,
                },
                7.0,
            ),  # lead 43.7 deg
            (
                {
                    "major_r": 5.0,
                    "pitch": 30.0,
                    "r": 2.0,
                    "n_turns": 3.0,
                    "phase": 0.0,
                    "handedness": 1.0,
                },
                7.5,
            ),  # same lead, fat tube
            # The opposite extreme: 0.3 deg of lead, with pitch < 2r so successive
            # turns overlap and the nearest-turn round() has to compose them as a
            # union. Exact to 1e-4 mm; here to keep it that way.
            (
                {
                    "major_r": 30.0,
                    "pitch": 1.0,
                    "r": 6.0,
                    "n_turns": 10.0,
                    "phase": 0.0,
                    "handedness": 1.0,
                },
                38.0,
            ),
        ],
        "annular_sector": [
            # Small bore: with the old angle-based bound this over-reported by
            # exactly 1/inner_r near the surface.
            ({"inner_r": 0.2, "outer_r": 3.0, "half_angle": 0.5, "height": 1.0}, 3.5),
            # Reflex sector: the wedge wraps past pi/2 and becomes a union of the
            # two bounding half-spaces rather than an intersection.
            ({"inner_r": 1.0, "outer_r": 3.0, "half_angle": 2.5, "height": 2.0}, 5.0),
        ],
    }
)

# Primitives known to over-report near the surface, with the cause. Empty since
# #108 closed. ``test_known_over_reporting_list_is_current`` fails if an entry
# here has actually been fixed, so the list cannot rot; add to it only to record
# a defect that is genuinely still open.
KNOWN_OVER_REPORTING: dict[str, str] = {}


# Primitives that are correct at the surface but knowingly over-report
# elsewhere, with the reason. Exempt from the whole-domain check only.
# ``ellipsoid`` and ``annular_sector`` both used to belong here and were
# fixed instead (#108), since in both cases an exact or conservative form
# existed at no cost to the geometry.
KNOWN_BOUNDS: dict[str, str] = {
    # The tube term measures to the tangent line of the centreline point at the
    # query's OWN azimuth. Near the Z axis the nearest centreline point is at a
    # different azimuth entirely and the tangent line runs away from the coil,
    # so the value over-reports by up to 2.7x in the bore. On the axis itself it
    # depends on the approach azimuth (4.00 to 10.95 mm at major_r=5/pitch=30),
    # i.e. the field is genuinely discontinuous there and NO finite constant can
    # fix it, unlike bellows and serpentine, which is why this is a documented
    # bound rather than a #108-style division. The wall rate IS fixed, by the
    # k_wall divisor in sdf_shapes.helix, and the stress variants above hold it.
    "helix": "tangent-line tube; over-reports up to 2.7x near the Z axis",
}


def _rates(name, kwargs, half, dim):
    """Return (near-surface max rate, whole-domain max rate).

    Gradients are exact (reverse-mode autodiff), so a measured maximum is a
    lower bound on the true one -- good enough to catch a broken primitive,
    and the reason RATE_TOL is tight rather than generous.
    """
    rng = np.random.default_rng(SEED)
    p = jnp.asarray(rng.uniform(-half, half, size=(N_SAMPLES, dim)))

    if name == "raster_field":
        # The shared table stores wire parameters for the mesh audit; the
        # primitive evaluator receives decoded samples, not the wire codec.
        kwargs = {
            "origin": kwargs["origin"],
            "spacing": kwargs["spacing"],
            "values": decode_raster_values(kwargs),
        }

    def scalar(x):
        return jnp.reshape(_PRIMITIVES[name](x[None, :], **kwargs), ())

    d = np.asarray(jax.vmap(scalar)(p))
    g = np.asarray(jnp.linalg.norm(jax.vmap(jax.grad(scalar))(p), axis=-1))

    keep = np.isfinite(d) & np.isfinite(g)
    assert keep.all(), (
        f"{name}: {int((~keep).sum())} of {keep.size} samples produced a "
        f"non-finite value or gradient: an SDF must be finite everywhere"
    )

    # "Near the surface" is defined by rank rather than an absolute band, so it
    # stays meaningful for a field whose units are not millimetres in the first
    # place (which is exactly the defect under test).
    n_near = max(int(NEAR_SURFACE_FRACTION * d.size), 100)
    near = np.argsort(np.abs(d))[:n_near]
    near_abs = np.abs(d)[near]
    near_min = float(near_abs.min())
    near_median = float(np.median(near_abs))
    frac_inside = float((d <= 0).mean())
    budget = NEAR_MEDIAN_FRAC * half
    # Rank-by-|d| only measures a surface if a zero crossing is in the cloud.
    # A filled clip box (thick TPMS at the mesh-table parameters) has every
    # sample inside and |d| bounded away from zero. Median-vs-half alone
    # cannot tell that apart from a small solid in a padded box (pyramid):
    # both have median |d| ≈ 0.05–0.07 of the half-extent. Occupancy does.
    close_enough = near_min < budget if 0.0 < frac_inside < 1.0 else near_median < budget
    assert close_enough, (
        f"{name} {kwargs}: occupied fraction {frac_inside:.3f}, nearest "
        f"|d|={near_min:.3f}, median |d| of the nearest 2%={near_median:.3f}, "
        f"sampling half-extent {half}. The near-surface rate is being measured "
        f"away from any surface. Thin the walls, or shrink the sampling box."
    )
    return float(g[near].max()), float(g.max())


def _cases(table, dim):
    """(test id, name, kwargs, half, dim) for every parameter set of every
    primitive in ``table``."""
    out = []
    for name, (kwargs, half) in sorted(table.items()):
        kw = DISTANCE_KWARGS.get(name, kwargs)
        out.append((name, name, kw, SAMPLING_HALF.get(name, half), dim))
        for i, (skw, h) in enumerate(STRESS_VARIANTS.get(name, [])):
            out.append((f"{name}-stress{i}", name, skw, h, dim))
    return out


CASES_3D = _cases(ALL_3D, 3)
CASES_2D = _cases(PRIMITIVES_2D, 2)
ALL_CASES = CASES_3D + CASES_2D

# Sampled fields promise a conservative step factor, not exact raw distances.
# Audit that factor explicitly below rather than skipping their gradients.
_SAMPLED_CASES = [c for c in ALL_CASES if c[1] == "raster_field"]
_OK_CASES = [c for c in ALL_CASES if c[1] not in KNOWN_OVER_REPORTING and c[1] != "raster_field"]
_BROKEN_CASES = [c for c in ALL_CASES if c[1] in KNOWN_OVER_REPORTING]


@pytest.mark.parametrize("case", _OK_CASES, ids=[c[0] for c in _OK_CASES])
def test_primitive_returns_a_true_distance(case):
    _, name, kwargs, half, dim = case
    near, _ = _rates(name, kwargs, half, dim)
    assert near <= RATE_TOL, (
        f"{name} {kwargs}: the value changes by up to {near:.3f} per unit of "
        f"travel near the surface, so it over-reports distance by ~{near:.1f}x. "
        f"Callers that trust the distance (voxel skipping, the gap audit, GLSL "
        f"sphere tracing) will be silently wrong."
    )


@pytest.mark.parametrize("case", _SAMPLED_CASES, ids=[c[0] for c in _SAMPLED_CASES])
def test_sampled_field_step_scale_bounds_gradient(case):
    _, name, kwargs, half, dim = case
    near, overall = _rates(name, kwargs, half, dim)
    # The coarse sphere must expose interpolation's excess gradient, so an
    # unsafe unit step factor would fail this audit rather than pass by luck.
    assert near > RATE_TOL
    scale = kwargs["step_scale"]
    assert 0.0 < scale <= 1.0
    assert overall * scale <= RATE_TOL


def test_known_over_reporting_list_is_current():
    """The KNOWN_OVER_REPORTING list must name exactly the primitives that are
    still broken.

    A plain xfail is not enough here: at some parameter values a broken TPMS
    measures ~1.0 by coincidence (the error scales as 1/scale), so a per-case
    xfail would spuriously "pass". Judging a primitive by its WORST parameter
    set is the honest test, and it makes the list self-cleaning: fix a
    primitive and this fails until its entry is deleted.
    """
    still_broken = set()
    for _, name, kwargs, half, dim in _BROKEN_CASES:
        near, _ = _rates(name, kwargs, half, dim)
        if near > RATE_TOL:
            still_broken.add(name)

    fixed = sorted(set(KNOWN_OVER_REPORTING) - still_broken)
    assert not fixed, (
        f"These primitives now return true distances and must be removed from "
        f"KNOWN_OVER_REPORTING: {fixed}"
    )


@pytest.mark.parametrize("case", _OK_CASES, ids=[c[0] for c in _OK_CASES])
def test_primitive_never_over_reports_anywhere(case):
    """Stricter companion: the rate must also hold away from the surface.

    Not required for correctness (a wrong value where there is no surface
    changes no decision), but every primitive that is not a documented bound
    happens to satisfy it, so asserting it catches a class of regression the
    near-surface check would miss.
    """
    _, name, kwargs, half, dim = case
    if name in KNOWN_BOUNDS:
        pytest.skip(f"{name} is a documented bound: {KNOWN_BOUNDS[name]}")
    _, overall = _rates(name, kwargs, half, dim)
    assert overall <= RATE_TOL, (
        f"{name} {kwargs}: whole-domain rate {overall:.3f}. It is correct at "
        f"the surface but over-reports elsewhere: either fix it, or record it "
        f"in KNOWN_BOUNDS with the reason."
    )


def test_every_registered_primitive_is_audited():
    """A primitive added without a parameter-table entry is not covered by this
    audit at all. Fail loudly rather than let it slip through unmeasured."""
    covered = set(ALL_3D) | set(PRIMITIVES_2D)
    missing = sorted(set(_PRIMITIVES) - covered)
    assert not missing, (
        f"Primitives registered in sdf/compile.py but absent from the test "
        f"tables in tests/conftest.py: {missing}. Add representative "
        f"parameters so they are audited."
    )
