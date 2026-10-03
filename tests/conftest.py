"""Shared pytest fixtures and the canonical primitive parameter tables."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from software_defined_matter import sdf_raster_field

# Make ``examples`` importable as a regular package so tests can reuse the
# canonical ``build_example`` helper.
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


# ===========================================================================
# Primitive parameter tables
# ===========================================================================
# These live here, rather than in the test module that first needed them, so
# that every suite exercising "one entry per primitive" shares one list and a
# newly added primitive cannot be covered by one test but silently missed by
# another.
#
# The original home was ``test_all_primitives_mesh.py``, which opens with
# ``pytest.importorskip("skimage")``. That skips the whole module (and anything
# importing from it) when the export extra is absent. Meshing needs skimage;
# checking that a field is a true distance does not, and must not inherit a skip
# that would make it quietly stop running.


def _sc(deg):
    """IQ angle parameter ``[sin, cos]``."""
    a = math.radians(deg)
    return [math.sin(a), math.cos(a)]


def _sampled_sphere_params():
    """Coarse sphere samples with a conservative interpolation step factor."""
    z, y, x = np.mgrid[-2:3, -2:3, -2:3]
    values = np.sqrt(x * x + y * y + z * z) - 1.2
    # Unit-spaced exact-distance samples have per-axis slopes at most one.
    # sqrt(3 + 1) also bounds the exterior distance continuation.
    return sdf_raster_field([-2.0] * 3, 1.0, values, step_scale=0.5)["params"]


# name -> (primitive kwargs, half-extent of a centred sampling cube). The cube
# is generous enough to contain the solid plus exterior margin on all sides; the
# voxel size adapts to it so every primitive is sampled at ~50 voxels across.
# fmt: off
PRIMITIVES = {
    # -- sampled distance bounds ---------------------------------------------
    "raster_field":     (_sampled_sphere_params(), 3.0),
    # -- exact 3-D solids ----------------------------------------------------
    "sphere":           ({"r": 8.0}, 11.0),
    "box":              ({"b": [6.0, 6.0, 6.0]}, 11.0),
    "round_box":        ({"b": [6.0, 6.0, 6.0], "r": 2.0}, 11.0),
    "box_frame":        ({"b": [7.0, 7.0, 7.0], "e": 0.9}, 11.0),
    "torus":            ({"t": [7.0, 2.0]}, 11.0),
    "capped_torus":     ({"sc": _sc(120), "ra": 7.0, "rb": 2.2}, 11.0),
    "helix":            ({"major_r": 7.0, "pitch": 4.0, "r": 1.2, "n_turns": 3.0,
                              "phase": 0.0, "handedness": 1.0}, 11.0),
    "screw_thread":     ({"r_root": 5.0, "depth": 2.0, "pitch": 4.0, "width": 3.0,
                              "n_turns": 3.0, "phase": 0.0, "handedness": 1.0,
                              "flank_deg": 60.0}, 11.0),
    "link":             ({"le": 4.0, "r1": 4.0, "r2": 1.6}, 13.0),
    "cone":             ({"c": _sc(28), "h": 13.0}, 16.0),
    "hex_prism":        ({"h": [7.0, 7.0]}, 11.0),
    "tri_prism":        ({"h": [7.0, 7.0]}, 11.0),
    "capsule":          ({"a": [0.0, 0.0, -6.0], "b": [0.0, 0.0, 6.0], "r": 3.0}, 11.0),
    "capped_cylinder":  ({"h": 8.0, "r": 5.0}, 11.0),
    "rounded_cylinder": ({"ra": 5.0, "rb": 2.0, "h": 7.0}, 11.0),
    "capped_cone":      ({"h": 8.0, "r1": 8.0, "r2": 2.0}, 12.0),
    "solid_angle":      ({"c": _sc(50), "ra": 9.0}, 12.0),
    "cut_sphere":       ({"r": 8.0, "h": 0.0}, 11.0),
    "ellipsoid":        ({"r": [8.0, 5.0, 5.0]}, 11.0),
    "octahedron":       ({"s": 8.0}, 11.0),
    "pyramid":          ({"h": 1.5}, 2.5),
    # -- triply-periodic minimal-surface lattices ----------------------------
    "gyroid":           ({"period": 10.0, "min_thickness": 3.675526, "n_periods": [2, 2, 2]}, 11.0),
    "schwarz_p":        ({"period": 10.0, "min_thickness": 3.675526, "n_periods": [2, 2, 2]}, 11.0),
    "schwarz_d":        ({"period": 10.0, "min_thickness": 3.675526, "n_periods": [2, 2, 2]}, 11.0),
    "neovius":          ({"period": 10.0, "min_thickness": 1.364185, "n_periods": [2, 2, 2]}, 11.0),
    "lidinoid":         ({"period": 10.0, "min_thickness": 2.450351, "n_periods": [2, 2, 2]}, 11.0),
    # -- compliant mechanisms ------------------------------------------------
    "notch_hinge":      ({"width": 6.0, "depth": 5.0, "notch_radius": 1.6}, 9.0),
    "leaf_spring":      ({"length": 16.0, "width": 6.0, "thickness": 1.6}, 12.0),
    "bellows":          ({"outer_r": 6.0, "inner_r": 4.0, "period": 3.0, "n_periods": 3}, 12.0),
    "serpentine":       ({"amplitude": 5.0, "wavelength": 8.0, "beam_width": 1.5,
                              "beam_height": 3.0, "n_periods": 3}, 16.0),
    "annular_sector":   ({"inner_r": 4.0, "outer_r": 7.0, "half_angle": 1.2, "height": 4.0}, 10.0),
}

# 2-D profile primitives. Excluded from PRIMITIVES because they are 2-D fields
# meant for ``extrusion`` / ``revolution`` / ``sweep`` and do not mesh directly,
# but they are still SDFs and must still report true distances.
#
# ``bezier_2d`` takes a ``(3K, 2)`` array of K cubic segments laid end to end
# and closed back to the start; the entry below is the standard 4-segment
# circle approximation (handle length 0.5523 r).
_BEZ_K = 0.5523 * 2.0
PRIMITIVES_2D = {
    "circle_2d":         ({"r": 2.0}, 6.0),
    "box_2d":            ({"b": [2.0, 1.0]}, 6.0),
    "rounded_box_2d":    ({"b": [2.0, 1.0], "r": 0.3}, 6.0),
    "segment_2d":        ({"a": [-2.0, 0.0], "b": [2.0, 0.0]}, 6.0),
    "trapezoid_2d":      ({"r1": 2.0, "r2": 1.0, "he": 1.5}, 6.0),
    "uneven_capsule_2d": ({"r1": 1.0, "r2": 0.5, "h": 2.0}, 6.0),
    "polygon_2d":        ({"vertices": [[-2.0, -1.0], [2.0, -1.0],
                                         [1.5, 1.0], [-1.0, 1.5]]}, 6.0),
    "bezier_2d":         ({"control_points": [
                             [2.0, 0.0], [2.0, _BEZ_K], [_BEZ_K, 2.0],
                             [0.0, 2.0], [-_BEZ_K, 2.0], [-2.0, _BEZ_K],
                             [-2.0, 0.0], [-2.0, -_BEZ_K], [-_BEZ_K, -2.0],
                             [0.0, -2.0], [_BEZ_K, -2.0], [2.0, -_BEZ_K]]}, 6.0),
    "bspline_2d":        ({"control_points": [[-2.0, 0.0], [-1.0, 2.0],
                                               [1.0, -2.0], [2.0, 0.0]]}, 6.0),
}
# fmt: on


@pytest.fixture
def example_part():
    from build_example import build_example  # type: ignore[import-not-found]

    return build_example()


@pytest.fixture
def fixed_stamp() -> str:
    """A frozen ISO 8601 datetime stamp for deterministic export filenames.

    ``export_part(..., stamp=<this>)`` uses the string verbatim, so tests
    assert exact names instead of racing the wall clock (the default
    ``stamp="datetime"`` produces a live timestamp by design).
    """
    return "2026-05-19T1430"
