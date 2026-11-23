"""Every primitive and every operation returns a signed **distance**, or is
listed here with the reason it does not.

An SDF promises that its value at a point is how far that point is from the
surface. A field can break that promise in three ways, and each needs its own
measurement:

    over-reports        the value climbs faster than 1 per unit of travel near
                        the surface. Unsafe: voxel skipping in
                        ``objectives.metrics``, the gap audit's
                        ``min(d_A + d_B)`` and the shader's sphere tracer all
                        trust the value as an upper bound on clearance.
    under-reports       it climbs slower. Safe for those, but wrong for anything
                        that turns the value into a length: ``round``,
                        ``onion``, blend radii, reported gaps.
    is not a distance   the gradient has unit length but the value is still
                        wrong. Inside a union, ``min`` measures to surface the
                        other child has swallowed. No gradient check sees this.

The first two are read off the exact (autodiff) gradient magnitude, the
**rate**, over the samples nearest the zero level set. Only the rate near the
surface decides correctness: far outside or deep inside, a cell is wholly empty
or wholly full whatever the value says. ``ellipsoid`` and ``annular_sector``
look alarming on a whole-volume maximum and are fine.

The third is read off the **projection residual**: step from ``p`` along the
gradient by the reported value, to ``q = p - f(p) * grad f(p)``. An exact
distance lands on the surface, so ``|f(q)| / |f(p)|`` is zero up to rounding.

Neither check sees a field that measures to the wrong part of its own surface.
Below ``pyramid``'s base the value is the distance to a slanted face: its
gradient has unit length and the projection lands on that face, a real surface
point. What gives it away is a **jump in value** across the face it never
measures to (#32). So values are also compared at pairs of nearby points, and a
pair whose value changes far faster than any rate could explain is a jump.

Gradients are also checked on a grid of round-number coordinates. Random
samples never land exactly on a face, an edge or the Z axis, which is where a
bare ``sqrt`` of a squared distance has an infinite derivative; part
dimensions are round numbers, so evaluation grids do land there.

Primitives are judged by name, at their worst parameter set: a defect that only
shows at some parameters is still a defect. Operations are judged per case,
because ``union`` of disjoint children is an exact distance and ``union`` of
overlapping ones is not, and both facts are worth pinning. Every known-failure
table is compared as a set, so fixing something fails the suite until its entry
is deleted.

Sampled fields (``raster_field``) promise a step factor, not exact raw values;
their factor is audited separately below. Whether ``infer_sdf_max_rate``'s
bound covers these rates is a separate invariant, pinned in
``tests/test_sdf_lipschitz.py``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Part,
    field_primitive,
    sdf_2d_to_3d,
    sdf_deform,
    sdf_loft,
    sdf_modifier,
    sdf_op,
    sdf_primitive,
    sdf_sweep,
    sdf_transform,
    wire,
)
from software_defined_matter.sdf import sdf_shapes
from software_defined_matter.sdf.compile import _PRIMITIVES, make_sdf_closure
from software_defined_matter.sdf.raster import decode_raster_values
from tests.conftest import PRIMITIVES, PRIMITIVES_2D, _sc

# A field may exceed rate 1 by this much, or fall short of it by the same
# factor, before it counts. Exact fields measure 1.000; the margin absorbs
# sampling luck and float32 rounding, not real error.
RATE_TOL = 1.02

# Fraction of samples nearest the zero level set that count as "at the surface".
NEAR_SURFACE_FRACTION = 0.02

# Rank-by-|d| is meaningless if those ranks never reach a zero crossing (a
# fully solid TPMS sheet is the usual cause). The median |d| of the near set
# must be well inside the sampling box.
NEAR_MEDIAN_FRAC = 0.05

# A projection residual above this means the step along the gradient missed the
# surface. Exact fields measure below 1e-4 in float32.
PROJECTION_TOL = 0.02

# Fraction of samples on one side of the surface whose projection may miss
# before that side counts as not a distance. Leaves room for samples sitting on
# a kink, where the one-sided gradient is legitimately ambiguous.
PROJECTION_BUDGET = 0.005

# A side needs this many samples before its projection result means anything
# (``segment_2d`` has no inside at all).
MIN_SIDE_SAMPLES = 50

N_SAMPLES = 40_000
SEED = 0

SIDES = ("outside", "inside")

# Nearby pairs for the value-jump check sit this fraction of the sampling
# half-extent apart. Every sample anchors one pair in a random direction, and
# every near-surface sample anchors JUMP_DIRECTIONS_NEAR more, since a face the
# field never measures to is still part of the surface.
JUMP_STEP_FRAC = 0.002
JUMP_DIRECTIONS_NEAR = 16

# A pair whose value changes by more than this many times its separation is a
# jump. Far above the largest genuine rate any field here reaches (about 3.4,
# a taper over-reporting), far below a real jump (pyramid's base: ~38).
JUMP_RATIO = 10.0

# Values are O(10) mm in float32 (JAX's default), whose spacing there is ~1e-6.
FLOAT32_ATOL = 2e-5


# ===========================================================================
# Measurement
# ===========================================================================


@dataclass(frozen=True)
class Audit:
    """What one field measured on one random cloud.

    ``miss_outside`` / ``miss_inside`` are the fractions of samples on that side
    whose projection missed the surface, or ``None`` when the side had too few
    samples to judge. ``jump_ratio`` is the largest change in value across a
    nearby pair, divided by the pair's separation.
    """

    near_max: float
    near_min: float
    overall_max: float
    miss_outside: float | None
    miss_inside: float | None
    n_nonfinite: int
    jump_ratio: float

    @property
    def jumps(self) -> bool:
        return self.jump_ratio > JUMP_RATIO

    def misses(self, side: str) -> bool:
        miss = self.miss_outside if side == "outside" else self.miss_inside
        return miss is not None and miss > PROJECTION_BUDGET


def _measure(label, scalar, half, dim) -> Audit:
    """Audit ``scalar`` (one point in, one value out) over a cube of half-extent ``half``.

    Gradients are exact (reverse-mode autodiff), so a measured maximum is a lower
    bound on the true one, which is why RATE_TOL is tight rather than generous.
    """
    rng = np.random.default_rng(SEED)
    p = jnp.asarray(rng.uniform(-half, half, size=(N_SAMPLES, dim)))
    d, g = jax.vmap(jax.value_and_grad(scalar))(p)
    dq = np.asarray(jax.vmap(scalar)(p - d[:, None] * g))
    d = np.asarray(d)
    rate = np.asarray(jnp.linalg.norm(g, axis=-1))

    finite = np.isfinite(d) & np.isfinite(rate)
    d_f, rate_f = d[finite], rate[finite]

    # "Near the surface" is defined by rank rather than an absolute band, so it
    # stays meaningful for a field whose units are not millimetres in the first
    # place (which is exactly one of the defects under test).
    n_near = max(int(NEAR_SURFACE_FRACTION * d_f.size), 100)
    near = np.argsort(np.abs(d_f))[:n_near]
    near_abs = np.abs(d_f)[near]
    frac_inside = float((d_f <= 0).mean())
    budget = NEAR_MEDIAN_FRAC * half
    # Rank-by-|d| only measures a surface if a zero crossing is in the cloud.
    # A filled clip box (thick TPMS at the mesh-table parameters) has every
    # sample inside and |d| bounded away from zero. Median-vs-half alone
    # cannot tell that apart from a small solid in a padded box (pyramid):
    # both have median |d| ≈ 0.05–0.07 of the half-extent. Occupancy does.
    if 0.0 < frac_inside < 1.0:
        close_enough = float(near_abs.min()) < budget
    else:
        close_enough = float(np.median(near_abs)) < budget
    assert close_enough, (
        f"{label}: occupied fraction {frac_inside:.3f}, nearest |d|="
        f"{float(near_abs.min()):.3f}, median |d| of the nearest 2%="
        f"{float(np.median(near_abs)):.3f}, sampling half-extent {half}. The "
        f"near-surface rate is being measured away from any surface. Thin the "
        f"walls, or shrink the sampling box."
    )

    floor = 1e-3 * half
    residual = np.abs(dq) / np.maximum(np.abs(d), floor)
    judged = finite & np.isfinite(dq) & (np.abs(d) > floor)

    def miss(side_mask):
        mask = judged & side_mask
        if mask.sum() < MIN_SIDE_SAMPLES:
            return None
        return float(np.mean(residual[mask] > PROJECTION_TOL))

    return Audit(
        near_max=float(rate_f[near].max()),
        near_min=float(rate_f[near].min()),
        overall_max=float(rate_f.max()),
        miss_outside=miss(d > 0),
        miss_inside=miss(d < 0),
        n_nonfinite=int((~finite).sum()),
        jump_ratio=_jump_ratio(scalar, np.asarray(p), d, half, rng),
    )


def _jump_ratio(scalar, p, d, half, rng) -> float:
    """Largest ``|f(b) - f(a)| / |b - a|`` over pairs of points ``|b - a|`` apart.

    Anchored on every sample whose VALUE is finite, whether or not its gradient
    is: a sweep's interior has NaN gradients (#36) and a finite, wrong value,
    and that interior is exactly where its end caps jump. Seeded, so the same
    pairs every run.
    """
    keep = np.isfinite(d)
    p, d = p[keep], d[keep]
    near = np.argsort(np.abs(d))[: max(int(NEAR_SURFACE_FRACTION * d.size), 100)]
    step = JUMP_STEP_FRAC * half
    dim = p.shape[1]
    anchors = np.concatenate([p, np.repeat(p[near], JUMP_DIRECTIONS_NEAR, axis=0)])
    base = np.concatenate([d, np.repeat(d[near], JUMP_DIRECTIONS_NEAR)])
    u = rng.normal(size=(len(anchors), dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    moved = np.asarray(jax.vmap(scalar)(jnp.asarray(anchors + step * u)))
    change = np.abs(moved - base)
    return float(np.nanmax(change[np.isfinite(change)]) / step)


def _round_grid(half, dim):
    """Round-number points spanning ``[-half, half]``: step 1 in 3-D, 0.25 in
    2-D, always including the origin and the axes."""
    step = 1.0 if dim == 3 else 0.25
    n = int(np.floor(half / step))
    axis = np.arange(-n, n + 1) * step
    mesh = np.meshgrid(*([axis] * dim), indexing="ij")
    return jnp.asarray(np.stack([m.ravel() for m in mesh], axis=-1))


def _count_nonfinite_on_grid(scalar, half, dim) -> int:
    d, g = jax.vmap(jax.value_and_grad(scalar))(_round_grid(half, dim))
    bad = ~np.isfinite(np.asarray(d)) | ~np.all(np.isfinite(np.asarray(g)), axis=-1)
    return int(bad.sum())


# ===========================================================================
# Primitives
# ===========================================================================

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
        # A normal that is not unit length, as a $ref or an optimiser can leave
        # it. The docstring asks for a unit normal; nothing enforces it (#31).
        "plane": [
            ({"n": [0.0, 0.0, 3.0], "h": 0.0}, 11.0),
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
            # The thread-like case test_sdf_helix.py pins against brute force.
            (
                {
                    "major_r": 10.0,
                    "pitch": 6.0,
                    "r": 1.2,
                    "n_turns": 3.0,
                    "phase": 0.0,
                    "handedness": 1.0,
                },
                12.0,
            ),
        ],
        "screw_thread": [
            # The spherical-bearing cap thread from test_sdf_screw_thread.py.
            (
                {
                    "r_root": 27.0,
                    "depth": 1.84,
                    "pitch": 3.0,
                    "width": 2.6,
                    "n_turns": 4.0,
                    "phase": 0.0,
                    "handedness": 1.0,
                    "flank_deg": 60.0,
                },
                31.0,
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
        # Angle vectors twice unit length, as an optimiser leaves them: the
        # conftest angles (120 and 50 deg) scaled by 2. Before normalisation,
        # capped_torus gave NaN here and solid_angle measured rate 3.0.
        "capped_torus": [
            ({"sc": [1.7320508, -1.0], "ra": 7.0, "rb": 2.2}, 11.0),
        ],
        "solid_angle": [
            ({"c": [1.5320889, 1.2855752], "ra": 9.0}, 12.0),
        ],
    }
)

# --- Known primitive failures, by name, each with its cause -----------------

# Primitives known to over-report near the surface, with the cause. Empty since
# #108 closed. ``test_known_over_reporting_list_is_current`` fails if an entry
# here has actually been fixed, so the list cannot rot; add to it only to record
# a defect that is genuinely still open.
KNOWN_OVER_REPORTING: dict[str, str] = {
    "plane": "dot(p, n) + h with n as authored: changes by |n| per unit (#31)",
}

# Of the above, those whose rate depends on a parameter rather than on the kind.
# ``infer_sdf_max_rate`` cannot list them in its per-kind table, so they are
# excused from matching it; its module docstring states each one instead.
PARAMETER_DEPENDENT_RATE: dict[str, str] = {
    "plane": "exact for a unit normal; the rate is |n| (#31)",
}

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

# Primitives whose VALUE jumps somewhere. Neither the rate nor the projection
# sees that (see the module docstring); the nearby-pair check does (#32). Fix
# the primitive and ``test_known_value_jumps_list_is_current`` fails until the
# entry here, and its rate declaration in ``sdf/lipschitz.py``, are deleted.
KNOWN_VALUE_JUMPS: dict[str, str] = {
    "pyramid": "measures to the slanted faces only, never to the base square (#30)",
    "helix": "on its own axis the value depends on the approach azimuth; see KNOWN_BOUNDS",
}

_TPMS = ("gyroid", "schwarz_p", "schwarz_d", "neovius", "lidinoid")
_TPMS_SLOW = "divided by the pattern's steepest gradient, so slower wherever it is flatter (#40)"
_BOTH = frozenset(SIDES)
_OUTSIDE = frozenset({"outside"})
_INSIDE = frozenset({"inside"})

# Primitives whose rate falls below 1 near the surface: safe for skipping and
# tracing, wrong as a length.
# fmt: off
KNOWN_UNDER_REPORTING: dict[str, str] = {
    **dict.fromkeys(_TPMS, _TPMS_SLOW),
    "ellipsoid":  "(length(p/r) - 1) * min(r) is slow along the longer axes (#40)",
    "bellows":    "divided by the steepest wall slope, so slow where it is flatter (#40)",
    "serpentine": "divided by the steepest centreline slope, so slow where it is flatter (#40)",
    "helix":      "the k_wall divisor that holds the wall rate to 1 at steep lead; see "
                  "KNOWN_BOUNDS",
}

# Primitives whose value is not a distance on the named side(s) of the surface:
# the projection along the gradient misses.
KNOWN_NOT_DISTANCES: dict[str, tuple[frozenset[str], str]] = {
    **dict.fromkeys(_TPMS, (_BOTH, "a scaled level set, not a closest-point distance (#40)")),
    "ellipsoid":      (_BOTH, "a scaled level set, not a closest-point distance (#40)"),
    "bellows":        (_BOTH, "a scaled radial offset, not a closest-point distance (#40)"),
    "serpentine":     (_BOTH, "a scaled vertical offset, not a closest-point distance (#40)"),
    "helix":          (_BOTH, "tangent-line tube; see KNOWN_BOUNDS"),
    "annular_sector": (_OUTSIDE, "max of half-spaces: short near its edges outside (#38)"),
    "tri_prism":      (_OUTSIDE, "max of half-spaces: short near its edges outside (#38)"),
    "notch_hinge":    (_OUTSIDE, "box minus two cylinders: short near the notch rims (#38)"),
    "screw_thread":   (_OUTSIDE, "max of half-planes in the tooth frame: short near the "
                                 "tooth corners (#38)"),
    "box_frame":      (_INSIDE, "union of three edge boxes: min is short inside (#37)"),
    "plane":          (_BOTH, "a non-unit normal scales the value by |n| (#31)"),
}

# Primitives with a non-finite value or gradient, on random samples or on the
# round-number grid. A bare sqrt of a squared distance that is exactly 0 on an
# edge, face or cap (#36).
KNOWN_NONFINITE: dict[str, str] = {
    "polygon_2d":   "sqrt of the squared edge distance; NaN on an edge (#36)",
    "bezier_2d":    "evaluated through polygon_2d (#36)",
    "bspline_2d":   "evaluated through polygon_2d (#36)",
    "trapezoid_2d": "sqrt of the squared distance; NaN on its boundary (#36)",
    "cone":         "sqrt of the squared distance; NaN on its surface (#36)",
    "capped_cone":  "sqrt of the squared distance; NaN on its caps (#36)",
}
# fmt: on


def _primitive_cases(table, dim):
    """(test id, name, kwargs, half, dim) for every parameter set of every
    primitive in ``table``."""
    out = []
    for name, (kwargs, half) in sorted(table.items()):
        kw = DISTANCE_KWARGS.get(name, kwargs)
        out.append((name, name, kw, SAMPLING_HALF.get(name, half), dim))
        for i, (skw, h) in enumerate(STRESS_VARIANTS.get(name, [])):
            out.append((f"{name}-stress{i}", name, skw, h, dim))
    return out


CASES_3D = _primitive_cases(ALL_3D, 3)
CASES_2D = _primitive_cases(PRIMITIVES_2D, 2)
ALL_CASES = CASES_3D + CASES_2D
_CASE_BY_ID = {c[0]: c for c in ALL_CASES}

# Sampled fields promise a conservative step factor, not exact raw distances.
# Audit that factor explicitly below rather than skipping their gradients.
_SAMPLED_CASES = [c for c in ALL_CASES if c[1] == "raster_field"]
_DISTANCE_CASES = [c for c in ALL_CASES if c[1] != "raster_field"]
_OK_CASES = [c for c in _DISTANCE_CASES if c[1] not in KNOWN_OVER_REPORTING]
_BROKEN_CASES = [c for c in ALL_CASES if c[1] in KNOWN_OVER_REPORTING]


def _primitive_scalar(name, kwargs):
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

    return scalar


@functools.cache
def _primitive_audit(case_id: str) -> Audit:
    _, name, kwargs, half, dim = _CASE_BY_ID[case_id]
    return _measure(f"{name} {kwargs}", _primitive_scalar(name, kwargs), half, dim)


@functools.cache
def _primitive_nonfinite_on_grid(case_id: str) -> int:
    _, name, kwargs, half, dim = _CASE_BY_ID[case_id]
    return _count_nonfinite_on_grid(_primitive_scalar(name, kwargs), half, dim)


def _ids(cases):
    return [c[0] for c in cases]


# --- Per-case checks ---------------------------------------------------------


_FINITE_CASES = [c for c in ALL_CASES if c[1] not in KNOWN_NONFINITE]


@pytest.mark.parametrize("case", _FINITE_CASES, ids=_ids(_FINITE_CASES))
def test_primitive_is_finite_on_random_samples(case):
    n = _primitive_audit(case[0]).n_nonfinite
    assert n == 0, (
        f"{case[1]}: {n} of {N_SAMPLES} samples produced a non-finite value or "
        f"gradient: an SDF must be finite everywhere"
    )


@pytest.mark.parametrize("case", _OK_CASES, ids=_ids(_OK_CASES))
def test_primitive_returns_a_true_distance(case):
    _, name, kwargs, _, _ = case
    near = _primitive_audit(case[0]).near_max
    assert near <= RATE_TOL, (
        f"{name} {kwargs}: the value changes by up to {near:.3f} per unit of "
        f"travel near the surface, so it over-reports distance by ~{near:.1f}x. "
        f"Callers that trust the distance (voxel skipping, the gap audit, GLSL "
        f"sphere tracing) will be silently wrong."
    )


@pytest.mark.parametrize("case", _OK_CASES, ids=_ids(_OK_CASES))
def test_primitive_never_over_reports_anywhere(case):
    """Stricter companion: the rate must also hold away from the surface.

    Not required for correctness (a wrong value where there is no surface
    changes no decision), but every primitive that is not a documented bound
    happens to satisfy it, so asserting it catches a class of regression the
    near-surface check would miss.
    """
    _, name, kwargs, _, _ = case
    if name in KNOWN_BOUNDS:
        pytest.skip(f"{name} is a documented bound: {KNOWN_BOUNDS[name]}")
    overall = _primitive_audit(case[0]).overall_max
    assert overall <= RATE_TOL, (
        f"{name} {kwargs}: whole-domain rate {overall:.3f}. It is correct at "
        f"the surface but over-reports elsewhere: either fix it, or record it "
        f"in KNOWN_BOUNDS with the reason."
    )


_JUMP_CASES = [c for c in _DISTANCE_CASES if c[1] not in KNOWN_VALUE_JUMPS]


@pytest.mark.parametrize("case", _JUMP_CASES, ids=_ids(_JUMP_CASES))
def test_primitive_value_does_not_jump(case):
    _, name, kwargs, half, _ = case
    ratio = _primitive_audit(case[0]).jump_ratio
    assert ratio <= JUMP_RATIO, (
        f"{name} {kwargs}: two points {JUMP_STEP_FRAC * half:.3g} apart differ in "
        f"value by {ratio:.1f}x their separation. The value jumps, usually because "
        f"some face of the solid is never measured to. Fix it, or list it in "
        f"KNOWN_VALUE_JUMPS with the cause."
    )


_UNDER_CASES = [c for c in _DISTANCE_CASES if c[1] not in KNOWN_UNDER_REPORTING]


@pytest.mark.parametrize("case", _UNDER_CASES, ids=_ids(_UNDER_CASES))
def test_primitive_does_not_under_report(case):
    _, name, kwargs, _, _ = case
    near = _primitive_audit(case[0]).near_min
    assert near >= 1.0 / RATE_TOL, (
        f"{name} {kwargs}: the value changes by as little as {near:.3f} per unit "
        f"of travel near the surface, so lengths read from it (round, onion, "
        f"blend radii, gaps) come out wrong. Fix it, or list it in "
        f"KNOWN_UNDER_REPORTING with the cause."
    )


_PROJECTION_CASES = [
    (c, side)
    for c in _DISTANCE_CASES
    for side in SIDES
    if side not in KNOWN_NOT_DISTANCES.get(c[1], (frozenset(),))[0]
]


@pytest.mark.parametrize(
    ("case", "side"), _PROJECTION_CASES, ids=[f"{c[0]}-{s}" for c, s in _PROJECTION_CASES]
)
def test_primitive_projection_lands_on_surface(case, side):
    _, name, kwargs, _, _ = case
    audit = _primitive_audit(case[0])
    assert not audit.misses(side), (
        f"{name} {kwargs}: stepping along the gradient by the reported value "
        f"misses the surface for {getattr(audit, 'miss_' + side):.1%} of samples "
        f"{side} it, so the value is not a distance there. Fix it, or list the "
        f"side in KNOWN_NOT_DISTANCES with the cause."
    )


@pytest.mark.parametrize("case", _FINITE_CASES, ids=_ids(_FINITE_CASES))
def test_primitive_is_finite_on_round_grid(case):
    n = _primitive_nonfinite_on_grid(case[0])
    assert n == 0, (
        f"{case[1]}: {n} round-number grid points give a non-finite value or "
        f"gradient. A bare sqrt of a squared distance is the usual cause; use "
        f"_length or a safe square root."
    )


@pytest.mark.parametrize("case", _SAMPLED_CASES, ids=_ids(_SAMPLED_CASES))
def test_sampled_field_step_scale_bounds_gradient(case):
    audit = _primitive_audit(case[0])
    # The coarse sphere must expose interpolation's excess gradient, so an
    # unsafe unit step factor would fail this audit rather than pass by luck.
    assert audit.near_max > RATE_TOL
    scale = case[2]["step_scale"]
    assert 0.0 < scale <= 1.0
    assert audit.overall_max * scale <= RATE_TOL


def test_capped_torus_is_finite_on_its_centreline():
    """The circle the arc lies on gives finite values and gradients everywhere,
    and depth exactly ``rb`` where it is the arc.

    The round grid only crosses this circle at a few points. The angle sweep
    here crosses the cap at 120 deg, so both branches (the arc's end and the
    foot on the circle) are hit. Before the rewrite, hundreds of these values
    and every one of these gradients were NaN in float32: IQ's expanded
    ``sqrt`` cancels to a negative argument on the centreline.
    """
    ra, rb = 7.3, 2.2
    sc = jnp.asarray(_sc(120.0), dtype=jnp.float32)
    th = np.linspace(-3.0, 3.0, 20_001)  # +-172 deg, past the cap on both sides
    p = jnp.asarray(
        np.stack([ra * np.sin(th), ra * np.cos(th), np.zeros_like(th)], axis=-1),
        dtype=jnp.float32,
    )

    def scalar(x):
        return sdf_shapes.capped_torus(x[None], sc, ra, rb)[0]

    values = jax.vmap(scalar)(p)
    grads = jax.vmap(jax.grad(scalar))(p)
    assert bool(jnp.all(jnp.isfinite(grads)))
    # Inside the cap the centreline is the arc, at depth exactly rb; beyond it
    # the nearest centreline point is the arc's end, so the value only rises.
    inside_cap = np.abs(th) <= math.radians(120.0)
    np.testing.assert_allclose(np.asarray(values)[inside_cap], -rb, atol=FLOAT32_ATOL)
    assert bool(jnp.all(values >= -rb - FLOAT32_ATOL))


# --- The primitive tables stay current ---------------------------------------


def test_known_over_reporting_list_is_current():
    """The KNOWN_OVER_REPORTING list must name exactly the primitives that are
    still broken.

    A plain xfail is not enough here: at some parameter values a broken TPMS
    measures ~1.0 by coincidence (the error scales as 1/period), so a per-case
    xfail would spuriously "pass". Judging a primitive by its WORST parameter
    set is the honest test, and it makes the list self-cleaning: fix a
    primitive and this fails until its entry is deleted.
    """
    still_broken = {c[1] for c in _BROKEN_CASES if _primitive_audit(c[0]).near_max > RATE_TOL}
    fixed = sorted(set(KNOWN_OVER_REPORTING) - still_broken)
    assert not fixed, (
        f"These primitives now return true distances and must be removed from "
        f"KNOWN_OVER_REPORTING: {fixed}"
    )


def test_known_under_reporting_list_is_current():
    still = {
        c[1]
        for c in _DISTANCE_CASES
        if c[1] in KNOWN_UNDER_REPORTING and _primitive_audit(c[0]).near_min < 1.0 / RATE_TOL
    }
    fixed = sorted(set(KNOWN_UNDER_REPORTING) - still)
    assert not fixed, f"No longer under-report; delete from KNOWN_UNDER_REPORTING: {fixed}"


def test_known_not_distances_list_is_current():
    fixed = []
    for name, (sides, _) in sorted(KNOWN_NOT_DISTANCES.items()):
        cases = [c for c in _DISTANCE_CASES if c[1] == name]
        for side in sorted(sides):
            if not any(_primitive_audit(c[0]).misses(side) for c in cases):
                fixed.append(f"{name}-{side}")
    assert not fixed, f"Now distances on these sides; delete from KNOWN_NOT_DISTANCES: {fixed}"


def test_known_nonfinite_list_is_current():
    still = {
        c[1]
        for c in ALL_CASES
        if c[1] in KNOWN_NONFINITE
        and (_primitive_audit(c[0]).n_nonfinite > 0 or _primitive_nonfinite_on_grid(c[0]) > 0)
    }
    fixed = sorted(set(KNOWN_NONFINITE) - still)
    assert not fixed, f"Now finite everywhere; delete from KNOWN_NONFINITE: {fixed}"


def test_known_value_jumps_list_is_current():
    """Every listed primitive still jumps at one of its parameter sets. Fix one
    and this fails until its entry here and its rate declaration are deleted,
    the same self-cleaning rule as ``test_known_over_reporting_list_is_current``."""
    still = {c[1] for c in _DISTANCE_CASES if _primitive_audit(c[0]).jumps}
    fixed = sorted(set(KNOWN_VALUE_JUMPS) - still)
    assert not fixed, (
        f"No longer jump in value; delete from KNOWN_VALUE_JUMPS and from "
        f"_PRIMITIVES_WITHOUT_UNIT_RATE: {fixed}"
    )


def test_primitive_tables_name_real_primitives():
    tables = (
        KNOWN_OVER_REPORTING,
        KNOWN_BOUNDS,
        KNOWN_VALUE_JUMPS,
        PARAMETER_DEPENDENT_RATE,
        KNOWN_UNDER_REPORTING,
        KNOWN_NOT_DISTANCES,
        KNOWN_NONFINITE,
    )
    unknown = sorted({n for t in tables for n in t} - set(_PRIMITIVES))
    assert not unknown, f"Known-failure tables name primitives that do not exist: {unknown}"


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


def test_rate_declarations_match_the_audit_exceptions():
    """``infer_sdf_max_rate`` declares exactly the primitives this audit
    exempts or knows to jump in value, so none of them can report a unit rate
    and an entry no longer needed fails here; it names only real primitives."""
    from software_defined_matter.sdf.lipschitz import _PRIMITIVES_WITHOUT_UNIT_RATE

    sampled = {c[1] for c in _SAMPLED_CASES}
    over = set(KNOWN_OVER_REPORTING) - set(PARAMETER_DEPENDENT_RATE)
    exempt = set(KNOWN_BOUNDS) | over | sampled
    assert set(_PRIMITIVES_WITHOUT_UNIT_RATE) == exempt | set(KNOWN_VALUE_JUMPS)
    assert set(_PRIMITIVES_WITHOUT_UNIT_RATE) <= set(_PRIMITIVES)


# ===========================================================================
# Operations
# ===========================================================================
#
# Each case is built with the public DSL builders and compiled with
# ``make_sdf_closure``, so it measures the field a part actually gets,
# compile-time factors included.


def _sphere(r, at=(0.0, 0.0, 0.0)):
    return sdf_transform("translate", sdf_primitive("sphere", r=r), t=list(at))


def _box(b, at=(0.0, 0.0, 0.0)):
    return sdf_transform("translate", sdf_primitive("box", b=list(b)), t=list(at))


def _square(half_side):
    s = half_side
    return sdf_primitive("polygon_2d", vertices=[[-s, -s], [s, -s], [s, s], [-s, s]])


def _chain(n=64, r=1.0, length=4.0):
    """``n`` spheres along X: a densely sampled swept sphere, the softmin use case."""
    xs = np.linspace(-length / 2, length / 2, n)
    return [_sphere(r, (float(x), 0.0, 0.0)) for x in xs]


_PROFILE_BOX = sdf_primitive("box_2d", b=[0.6, 0.2])
_PROFILE_CIRCLE = sdf_primitive("circle_2d", r=0.5)
_PATH_STRAIGHT = [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
_PATH_CORNER_90 = [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0]]
_PATH_CORNER_60 = [[-3.0, -1.0, 0.0], [1.0, -1.0, 0.0], [3.0, 2.4641, 0.0]]
_PATH_LOOP = [
    [2.5 * float(np.cos(a)), 2.5 * float(np.sin(a)), 0.0]
    for a in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
]

# case id -> (tree, half-extent of the centred sampling cube, node kinds covered).
# fmt: off
OP_CASES: dict[str, tuple[dict, float, tuple[str, ...]]] = {
    # -- booleans ---------------------------------------------------------------
    "union-overlapping-spheres":
        (sdf_op("union", [_sphere(2.0, (-1, 0, 0)), _sphere(2.0, (1, 0, 0))]), 4.0, ("union",)),
    "union-disjoint-spheres":
        (sdf_op("union", [_sphere(1.0, (-3, 0, 0)), _sphere(1.0, (3, 0, 0))]), 5.0, ("union",)),
    "intersect-overlapping-spheres":
        (sdf_op("intersect", [_sphere(2.0, (-1, 0, 0)), _sphere(2.0, (1, 0, 0))]), 4.0,
         ("intersect",)),
    "subtract-corner-sphere-from-box":
        (sdf_op("subtract", [_box([2.0, 1.5, 1.0]), _sphere(1.2, (2.0, 1.5, 1.0))]), 4.0,
         ("subtract",)),
    "subtract-interior-cavity":
        (sdf_op("subtract", [_box([3.0, 3.0, 3.0]), _sphere(1.5)]), 4.5, ("subtract",)),
    "smooth_union":
        (sdf_op("smooth_union", [_sphere(2.0, (-1.5, 0, 0)), _sphere(2.0, (1.5, 0, 0))], k=1.0),
         4.5, ("smooth_union",)),
    "smooth_subtract":
        (sdf_op("smooth_subtract", [_box([2.0, 1.5, 1.0]), _sphere(1.2, (2.0, 1.5, 1.0))], k=0.5),
         4.0, ("smooth_subtract",)),
    "smooth_intersect":
        (sdf_op("smooth_intersect", [_sphere(2.0, (-1, 0, 0)), _sphere(2.0, (1, 0, 0))], k=0.5),
         4.0, ("smooth_intersect",)),
    "softmin_many-chain":
        (sdf_op("softmin_many", _chain(), k=0.25), 4.0, ("softmin_many",)),
    "softmin_chunked-chain":
        (sdf_op("softmin_chunked", _chain(), k=0.25, chunk_size=16), 4.0, ("softmin_chunked",)),
    # -- transforms -------------------------------------------------------------
    "translate": (_box([2.0, 1.0, 0.5], at=(0.5, -0.3, 0.2)), 3.5, ("translate",)),
    "rotate_x": (sdf_transform("rotate_x", _box([2.0, 1.0, 0.5]), angle=0.4), 3.5, ("rotate_x",)),
    "rotate_y": (sdf_transform("rotate_y", _box([2.0, 1.0, 0.5]), angle=0.4), 3.5, ("rotate_y",)),
    "rotate_z": (sdf_transform("rotate_z", _box([2.0, 1.0, 0.5]), angle=0.4), 3.5, ("rotate_z",)),
    "rotate_matrix":
        (sdf_transform("rotate_matrix", _box([2.0, 1.0, 0.5]),
                       R=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 3.5,
         ("rotate_matrix",)),
    "scale": (sdf_transform("scale", _box([1.0, 0.6, 0.4]), s=1.5), 3.0, ("scale",)),
    "scale_axis":
        (sdf_transform("scale_axis", sdf_primitive("sphere", r=1.0), s=[2.0, 1.0, 1.0]), 3.0,
         ("scale_axis",)),
    "repeat_finite":
        (sdf_transform("repeat_finite", sdf_primitive("sphere", r=0.5), c=2.0, l=[1.0, 1.0, 1.0]),
         4.0, ("repeat_finite",)),
    "canonical_sector_fold":
        (sdf_transform("canonical_sector_fold", _box([0.5, 0.5, 0.5], at=(3.0, 0.0, 0.0)),
                       n_sectors=6), 4.5, ("canonical_sector_fold",)),
    "mirror-disjoint":
        (sdf_transform("mirror", _sphere(1.0, (2.0, 0, 0)), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0]),
         4.0, ("mirror",)),
    "mirror-overlapping":
        (sdf_transform("mirror", _sphere(1.0, (0.5, 0, 0)), n=[1.0, 0.0, 0.0], o=[0.0, 0.0, 0.0]),
         2.5, ("mirror",)),
    # -- modifiers --------------------------------------------------------------
    "round-box": (sdf_modifier("round", _box([1.0, 1.0, 1.0]), r=0.5), 3.0, ("round",)),
    "round-intersect":
        (sdf_modifier("round", sdf_op("intersect", [_sphere(2.0, (-1, 0, 0)),
                                                    _sphere(2.0, (1, 0, 0))]), r=0.3), 4.0,
         ("round",)),
    "onion-sphere":
        (sdf_modifier("onion", sdf_primitive("sphere", r=2.0), thickness=0.3), 3.0, ("onion",)),
    "onion-ellipsoid":
        (sdf_modifier("onion", sdf_primitive("ellipsoid", r=[3.0, 1.0, 1.0]), thickness=0.1),
         3.5, ("onion",)),
    "elongate-one-axis":
        (sdf_modifier("elongate", sdf_primitive("sphere", r=0.5), h=[2.0, 0.0, 0.0]), 3.0,
         ("elongate",)),
    "elongate-three-axes":
        (sdf_modifier("elongate", sdf_primitive("sphere", r=0.5), h=[2.0, 2.0, 2.0]), 3.0,
         ("elongate",)),
    # -- deforms ----------------------------------------------------------------
    "twist": (sdf_deform("twist", _box([1.0, 2.0, 1.0]), k=0.5), 3.0, ("twist",)),
    "bend": (sdf_deform("bend", _box([2.0, 0.5, 0.5]), k=0.3), 3.0, ("bend",)),
    "twist_radial":
        (sdf_deform("twist_radial", _box([2.5, 0.3, 0.5]), r0=0.5, r1=2.5,
                    angle_inner=0.0, angle_outer=0.8), 3.0, ("twist_radial",)),
    "twist_linear":
        (sdf_deform("twist_linear", _box([2.5, 0.3, 0.5]), axis=[1.0, 0.0], u0=-2.0, u1=2.0,
                    angle_0=0.0, angle_1=0.6), 3.0, ("twist_linear",)),
    "shear_linear":
        (sdf_deform("shear_linear", _box([3.0, 1.0, 0.3]), axis=[1.0, 0.0], u0=-2.0, u1=2.0,
                    dz_0=-2.0, dz_1=2.0), 4.0, ("shear_linear",)),
    "taper_linear-constant":
        (sdf_deform("taper_linear", _box([2.0, 2.0, 2.0]), z0=-2.0, z1=2.0, s_0=0.5, s_1=0.5),
         3.0, ("taper_linear",)),
    "taper_linear-ramp":
        (sdf_deform("taper_linear", _box([2.0, 2.0, 2.0]), z0=-2.0, z1=2.0, s_0=1.0, s_1=0.3),
         3.0, ("taper_linear",)),
    "displace":
        (sdf_deform("displace", sdf_primitive("sphere", r=3.0),
                    field=field_primitive("sin_xyz", freq=[0.5, 0.5, 0.5], amplitude=0.3)),
         4.0, ("displace",)),
    # -- 2-D to 3-D -------------------------------------------------------------
    "revolution":
        (sdf_2d_to_3d("revolution", sdf_primitive("box_2d", b=[1.0, 2.0]), offset=3.0), 6.0,
         ("revolution",)),
    "extrusion":
        (sdf_2d_to_3d("extrusion", sdf_primitive("polygon_2d", vertices=[
            [-2.0, -1.0], [2.0, -1.0], [1.5, 1.0], [-1.0, 1.5]]), h=1.5), 4.0, ("extrusion",)),
    # -- sweep ------------------------------------------------------------------
    "sweep-circle-straight":
        (sdf_sweep(_PROFILE_CIRCLE, _PATH_STRAIGHT, path_kind="polyline"), 3.0, ("sweep",)),
    "sweep-box-straight":
        (sdf_sweep(_PROFILE_BOX, _PATH_STRAIGHT, path_kind="polyline"), 3.0, ("sweep",)),
    "sweep-circle-corner-90":
        (sdf_sweep(_PROFILE_CIRCLE, _PATH_CORNER_90, path_kind="polyline"), 3.5, ("sweep",)),
    "sweep-box-corner-90":
        (sdf_sweep(_PROFILE_BOX, _PATH_CORNER_90, path_kind="polyline"), 3.5, ("sweep",)),
    "sweep-box-corner-60":
        (sdf_sweep(_PROFILE_BOX, _PATH_CORNER_60, path_kind="polyline"), 4.0, ("sweep",)),
    "sweep-circle-bspline-loop":
        (sdf_sweep(sdf_primitive("circle_2d", r=0.4), _PATH_LOOP, path_kind="bspline"), 4.0,
         ("sweep",)),
    # -- loft -------------------------------------------------------------------
    "loft-identical-sections":
        (sdf_loft([sdf_primitive("box_2d", b=[1.0, 1.0]), sdf_primitive("box_2d", b=[1.0, 1.0])],
                  z=[-1.0, 1.0]), 3.0, ("loft",)),
    "loft-field-taper":
        (sdf_loft([sdf_primitive("box_2d", b=[5.0, 5.0]), sdf_primitive("box_2d", b=[1.0, 1.0])],
                  z=[-1.0, 1.0]), 6.0, ("loft",)),
    "loft_shape-taper":
        (sdf_loft([_square(5.0), _square(1.0)], z=[-1.0, 1.0], interp="shape"), 6.0, ("loft",)),
}
# fmt: on

# --- Known operation failures, by case id, each with its cause ---------------

# Kinds the wire schema accepts but the compiler never compiles, so there is no
# field to audit.
UNCOMPILED_KINDS: dict[str, str] = {
    "repeat_inf": "schema-legal, never compiled: a shape that tiles forever has no finite "
    "closure (see wire.TRANSFORMS)",
}

_WARPS = ("twist", "bend", "twist_radial", "twist_linear")
_SWEEPS = tuple(c for c in OP_CASES if c.startswith("sweep-"))
_WARP_STRETCH = "rotation angle varies with position, so space stretches more the further out"
_WARP_COMPRESS = "a varying rotation compresses space as well as stretching it"
_SWEEP_NAN = "sqrt of max(d2d, 0)**2 + max(d_ax, 0)**2, which is 0 at every interior point (#36)"

# fmt: off
# Over 1 near the surface, but bounded by ``infer_sdf_max_rate`` over an
# evaluation domain: the stretch grows with distance from an axis, so no
# constant inside the op can remove it. Exempt from the over-reporting check.
OP_RATE_BOUNDED_BY_DOMAIN: dict[str, str] = dict.fromkeys(_WARPS, _WARP_STRETCH)

# Over 1 near the surface with no domain-dependent excuse.
OP_KNOWN_OVER_REPORTING: dict[str, str] = {
    "shear_linear":          "not divided by the shear's largest singular value (#36)",
    "taper_linear-constant": "child(p / s) with no rescale: over by 1/s (#36)",
    "taper_linear-ramp":     "child(p / s) with no rescale: over by 1/s (#36)",
    "displace":              "not divided by 1 + the field's rate (#36)",
    "loft-field-taper":      "radial offset across a tilted wall (#36)",
    "loft_shape-taper":      "radial offset across a tilted wall (#36)",
    "sweep-box-corner-90":   "ball joint stretches a non-circular profile (#42)",
    "sweep-box-corner-60":   "ball joint stretches a non-circular profile (#42)",
}

# Under 1 near the surface.
OP_KNOWN_UNDER_REPORTING: dict[str, str] = {
    **dict.fromkeys(("twist", "bend", "twist_radial"), _WARP_COMPRESS),
    "displace":              "an added field slows the value as well as speeding it",
    "scale_axis":            "min(s) * child(p / s) is slow along the stretched axes",
    "onion-ellipsoid":       "inherits the ellipsoid's scaled field (#40)",
    "smooth_union":          "the blend region is not a distance by design (#37)",
    "smooth_subtract":       "the blend region is not a distance by design (#37)",
    "smooth_intersect":      "the blend region is not a distance by design (#37)",
    "softmin_many-chain":    "log-sum-exp over many samples (#43)",
    "softmin_chunked-chain": "log-sum-exp over many samples (#43)",
}

# Not a distance on the named side(s): the projection misses.
OP_KNOWN_NOT_DISTANCES: dict[str, tuple[frozenset[str], str]] = {
    "union-overlapping-spheres":       (_INSIDE, "min measures to swallowed surface (#39)"),
    "mirror-overlapping":              (_INSIDE, "union with its reflection (#39)"),
    "intersect-overlapping-spheres":   (_OUTSIDE, "max measures to removed surface (#39)"),
    "subtract-corner-sphere-from-box": (_OUTSIDE, "box-sphere seam has no closed form (#37)"),
    "round-intersect":                 (_OUTSIDE, "rounding an intersection grows its edges "
                                                  "(#37)"),
    "elongate-three-axes":             (_INSIDE, "flat interior plateau (#36)"),
    "onion-ellipsoid":                 (_BOTH, "inherits the ellipsoid's scaled field (#40)"),
    "scale_axis":                      (_BOTH, "per-axis scale has no exact form (#37)"),
    "smooth_union":                    (_BOTH, "blend region (#37)"),
    "smooth_subtract":                 (_BOTH, "blend region (#37)"),
    "smooth_intersect":                (_BOTH, "blend region (#37)"),
    "softmin_many-chain":              (_BOTH, "log-sum-exp offset (#43)"),
    "softmin_chunked-chain":           (_BOTH, "log-sum-exp offset (#43)"),
    **dict.fromkeys(_WARPS, (_BOTH, "deformed space (#37)")),
    "shear_linear":                    (_BOTH, "deformed space (#37)"),
    "taper_linear-constant":           (_BOTH, "deformed space (#37)"),
    "taper_linear-ramp":               (_BOTH, "deformed space (#37)"),
    "displace":                        (_BOTH, "displaced surface (#37)"),
    "loft-field-taper":                (_BOTH, "interpolated sections (#36, #37)"),
    "loft_shape-taper":                (_BOTH, "interpolated sections (#36, #37)"),
    "sweep-circle-straight":           (_OUTSIDE, "inside the tube the value ignores the end "
                                                  "caps, where outside projections land (#36)"),
    "sweep-circle-corner-90":          (_OUTSIDE, "inside the tube the value ignores the end "
                                                  "caps, where outside projections land (#36)"),
    "sweep-box-corner-90":             (_OUTSIDE, "ball joint (#42), and the end caps (#36)"),
    "sweep-box-corner-60":             (_OUTSIDE, "ball joint (#42), and the end caps (#36)"),
}

# Value jumps somewhere: the nearby-pair check (#32).
_CAP_JUMP = "inside the tube the value ignores the end caps, so it jumps across each cap (#36)"
OP_KNOWN_VALUE_JUMPS: dict[str, str] = dict.fromkeys(
    (c for c in _SWEEPS if c != "sweep-circle-bspline-loop"), _CAP_JUMP
)

# Non-finite value or gradient, on random samples or on the round-number grid.
OP_KNOWN_NONFINITE: dict[str, str] = {
    **dict.fromkeys(_SWEEPS, _SWEEP_NAN),
    "twist_radial":          "its radius bypasses _length; NaN on the Z axis (#36)",
    "canonical_sector_fold": "atan2 without the _azimuth guard; NaN on the Z axis (#36)",
    "extrusion":             "inherits polygon_2d's NaN on an edge (#36)",
    "loft_shape-taper":      "_poly_sdf_batched's NaN on an edge (#36)",
}
# fmt: on


def _op_scalar(case_id):
    tree = OP_CASES[case_id][0]
    part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
    closure = make_sdf_closure(tree, part)
    free = jnp.asarray(part.param_vector())

    def scalar(x):
        return jnp.reshape(closure(x[None, :], free), ())

    return scalar


@functools.cache
def _op_audit(case_id: str) -> Audit:
    return _measure(case_id, _op_scalar(case_id), OP_CASES[case_id][1], 3)


@functools.cache
def _op_nonfinite_on_grid(case_id: str) -> int:
    return _count_nonfinite_on_grid(_op_scalar(case_id), OP_CASES[case_id][1], 3)


OP_IDS = sorted(OP_CASES)

# --- Per-case checks ---------------------------------------------------------


_OP_FINITE_IDS = [c for c in OP_IDS if c not in OP_KNOWN_NONFINITE]


@pytest.mark.parametrize("case_id", _OP_FINITE_IDS)
def test_op_is_finite_on_random_samples(case_id):
    n = _op_audit(case_id).n_nonfinite
    assert n == 0, f"{case_id}: {n} samples produced a non-finite value or gradient"


_OP_OVER_IDS = [
    c for c in OP_IDS if c not in OP_KNOWN_OVER_REPORTING and c not in OP_RATE_BOUNDED_BY_DOMAIN
]


@pytest.mark.parametrize("case_id", _OP_OVER_IDS)
def test_op_does_not_over_report(case_id):
    near = _op_audit(case_id).near_max
    assert near <= RATE_TOL, (
        f"{case_id}: the value changes by up to {near:.3f} per unit of travel near "
        f"the surface, so it over-reports distance by ~{near:.2f}x. Fix the op, or "
        f"list the case in OP_KNOWN_OVER_REPORTING with the cause."
    )


_OP_JUMP_IDS = [c for c in OP_IDS if c not in OP_KNOWN_VALUE_JUMPS]


@pytest.mark.parametrize("case_id", _OP_JUMP_IDS)
def test_op_value_does_not_jump(case_id):
    ratio = _op_audit(case_id).jump_ratio
    assert ratio <= JUMP_RATIO, (
        f"{case_id}: nearby points differ in value by {ratio:.1f}x their "
        f"separation, so the value jumps. Fix it, or list the case in "
        f"OP_KNOWN_VALUE_JUMPS with the cause."
    )


_OP_UNDER_IDS = [c for c in OP_IDS if c not in OP_KNOWN_UNDER_REPORTING]


@pytest.mark.parametrize("case_id", _OP_UNDER_IDS)
def test_op_does_not_under_report(case_id):
    near = _op_audit(case_id).near_min
    assert near >= 1.0 / RATE_TOL, (
        f"{case_id}: the value changes by as little as {near:.3f} per unit of "
        f"travel near the surface, so lengths read from it come out wrong. Fix "
        f"the op, or list the case in OP_KNOWN_UNDER_REPORTING with the cause."
    )


_OP_PROJECTION_CASES = [
    (c, side)
    for c in OP_IDS
    for side in SIDES
    if side not in OP_KNOWN_NOT_DISTANCES.get(c, (frozenset(),))[0]
]


@pytest.mark.parametrize(
    ("case_id", "side"),
    _OP_PROJECTION_CASES,
    ids=[f"{c}-{s}" for c, s in _OP_PROJECTION_CASES],
)
def test_op_projection_lands_on_surface(case_id, side):
    audit = _op_audit(case_id)
    assert not audit.misses(side), (
        f"{case_id}: stepping along the gradient by the reported value misses the "
        f"surface for {getattr(audit, 'miss_' + side):.1%} of samples {side} it, so "
        f"the value is not a distance there. Fix it, or list the side in "
        f"OP_KNOWN_NOT_DISTANCES with the cause."
    )


@pytest.mark.parametrize("case_id", _OP_FINITE_IDS)
def test_op_is_finite_on_round_grid(case_id):
    n = _op_nonfinite_on_grid(case_id)
    assert n == 0, (
        f"{case_id}: {n} round-number grid points give a non-finite value or "
        f"gradient. A bare sqrt of a squared distance is the usual cause; use "
        f"_length or a safe square root."
    )


# --- The operation tables stay current ---------------------------------------


def test_op_known_over_reporting_is_current():
    fixed = sorted(c for c in OP_KNOWN_OVER_REPORTING if _op_audit(c).near_max <= RATE_TOL)
    assert not fixed, f"No longer over-report; delete from OP_KNOWN_OVER_REPORTING: {fixed}"


def test_op_rate_bounded_by_domain_is_current():
    fixed = sorted(c for c in OP_RATE_BOUNDED_BY_DOMAIN if _op_audit(c).near_max <= RATE_TOL)
    assert not fixed, f"No longer over-report; delete from OP_RATE_BOUNDED_BY_DOMAIN: {fixed}"


def test_op_known_under_reporting_is_current():
    fixed = sorted(c for c in OP_KNOWN_UNDER_REPORTING if _op_audit(c).near_min >= 1.0 / RATE_TOL)
    assert not fixed, f"No longer under-report; delete from OP_KNOWN_UNDER_REPORTING: {fixed}"


def test_op_known_not_distances_is_current():
    fixed = sorted(
        f"{c}-{side}"
        for c, (sides, _) in OP_KNOWN_NOT_DISTANCES.items()
        for side in sides
        if not _op_audit(c).misses(side)
    )
    assert not fixed, f"Now distances on these sides; delete from OP_KNOWN_NOT_DISTANCES: {fixed}"


def test_op_known_value_jumps_is_current():
    fixed = sorted(c for c in OP_KNOWN_VALUE_JUMPS if not _op_audit(c).jumps)
    assert not fixed, f"No longer jump in value; delete from OP_KNOWN_VALUE_JUMPS: {fixed}"


def test_op_known_nonfinite_is_current():
    fixed = sorted(
        c
        for c in OP_KNOWN_NONFINITE
        if _op_audit(c).n_nonfinite == 0 and _op_nonfinite_on_grid(c) == 0
    )
    assert not fixed, f"Now finite everywhere; delete from OP_KNOWN_NONFINITE: {fixed}"


def test_op_tables_name_real_cases():
    tables = (
        OP_RATE_BOUNDED_BY_DOMAIN,
        OP_KNOWN_OVER_REPORTING,
        OP_KNOWN_UNDER_REPORTING,
        OP_KNOWN_NOT_DISTANCES,
        OP_KNOWN_VALUE_JUMPS,
        OP_KNOWN_NONFINITE,
    )
    unknown = sorted({c for t in tables for c in t} - set(OP_CASES))
    assert not unknown, f"Known-failure tables name cases that do not exist: {unknown}"


def test_uncompiled_kinds_really_do_not_compile():
    """An exemption from the coverage check below holds only while the kind
    still has no compiler; once it compiles, it needs a case instead."""
    for kind in UNCOMPILED_KINDS:
        tree = sdf_transform(kind, sdf_primitive("sphere", r=0.5), c=[2.0, 2.0, 2.0])
        part = Part(name="t", materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)])
        with pytest.raises(ValueError, match="Unknown transform"):
            make_sdf_closure(tree, part)


def test_every_node_kind_is_audited():
    """A kind added to the wire registry without a case here is not audited at all."""
    kinds = (
        set(wire.OPS)
        | set(wire.TRANSFORMS)
        | set(wire.MODIFIERS)
        | set(wire.DEFORMS)
        | set(wire.TWO_D_TO_3D)
        | {"sweep", "loft"}
    )
    covered = {k for _, _, cover in OP_CASES.values() for k in cover}
    missing = sorted(kinds - covered - set(UNCOMPILED_KINDS))
    assert not missing, f"Node kinds with no case in OP_CASES: {missing}"
