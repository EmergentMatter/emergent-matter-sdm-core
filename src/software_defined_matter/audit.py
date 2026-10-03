"""Gap audit: assert that the gaps a Part actually has match the clearances it declares.

Motivation (the org's single-source-clearances pattern): in print-in-place
mechanisms the classic silent failure is clearance *stacking* -- two code paths each apply
"a little safety" to the same interface (e.g. raceways offset by ``clr`` AND the roller also
undersized by ``clr``). The model builds, the parts verify as free bodies, and the mechanism
runs 2x as loose as its params promise. Free-body checks answer "do they touch?"; this module
answers "is the gap the number you promised?"

For exact SDFs the separation between two disjoint bodies A and B is::

    gap(A, B) = min over p of ( d_A(p) + d_B(p) )

attained on the segment realising the closest approach. On a sample grid the estimate
converges from above with error O(spacing); :func:`measure_gap` refines the coarse minimum
locally, so the returned value is accurate to a small fraction of ``d_voxel``.

Typical use in a CEM test suite, with one MaterialRegion per moving body::

    declared = {("inner_ring", "roller"): 0.20, ("roller", "outer_ring"): 0.20}
    assert_gaps(part, declared, d_tol=0.02)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from software_defined_matter.grid_sampling import eval_chunked
from software_defined_matter.sdf.bbox import BBox, infer_material_bbox, pad_bbox
from software_defined_matter.sdf.compile import make_sdf_closure

if TYPE_CHECKING:
    from software_defined_matter.model import Part

SDFCallable = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class GapCheck:
    """Result of auditing one interface (pair of material regions)."""

    name_a: str
    name_b: str
    d_declared: float
    d_measured: float
    d_tol: float

    @property
    def b_ok(self) -> bool:
        return abs(self.d_measured - self.d_declared) <= self.d_tol

    def __str__(self) -> str:
        verdict = "ok" if self.b_ok else "FAIL"
        return (
            f"[{verdict}] {self.name_a} <-> {self.name_b}: "
            f"declared {self.d_declared:.4f}, measured {self.d_measured:.4f} "
            f"(tol {self.d_tol:.4f})"
        )


class GapAuditError(AssertionError):
    """Raised by :func:`assert_gaps` when any measured gap disagrees with its declaration."""


def _grid_points(bbox: BBox, d_spacing: float) -> np.ndarray:
    (x0, y0, z0), (x1, y1, z1) = bbox
    ax = [
        np.linspace(lo, hi, max(2, round((hi - lo) / d_spacing) + 1))
        for lo, hi in ((x0, x1), (y0, y1), (z0, z1))
    ]
    gx, gy, gz = np.meshgrid(*ax, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)


def measure_gap(
    sdf_a: SDFCallable,
    sdf_b: SDFCallable,
    bbox: BBox,
    d_voxel: float = 0.25,
    n_refine: int = 2,
    chunk_size: int = 32_768,
) -> float:
    """Minimum separation between two SDF bodies (negative if they overlap).

    ``sdf_a`` / ``sdf_b`` are ``points -> distance`` callables (e.g. from
    :func:`~software_defined_matter.sdf.compile.make_sdf_closure`). The coarse
    grid minimum of ``d_a + d_b`` is refined ``n_refine`` times on a local grid
    around the argmin, each pass 8x finer, so accuracy ~ ``d_voxel / 8**n_refine``.

    ``chunk_size`` caps how many grid points are evaluated per call. Peak memory
    here is ``n_points x sum(vertices over every polygon) x 2 x 8`` bytes because
    ``polygon_2d`` allocates an ``(n_points, n_vertices, 2)`` intermediate. Measured
    on a print-in-place differential (two gear bodies, ~2,380 polygon verts over
    472,566 points) a single call peaked at **18.0 GB**; chunked at 32,768 it
    peaks at ~2.2 GB with an identical minimum. Raising this is a memory
    decision, not an accuracy one.
    """

    def total(points: jnp.ndarray) -> jnp.ndarray:
        return sdf_a(points) + sdf_b(points)

    d_spacing = d_voxel
    vals: np.ndarray | None = None
    for _ in range(n_refine + 1):
        points = _grid_points(bbox, d_spacing)
        vals = eval_chunked(total, points, chunk_size=chunk_size)
        p_best = points[int(np.argmin(vals))]
        d_half = 2.0 * d_spacing
        lo, hi = p_best - d_half, p_best + d_half
        bbox = (
            (float(lo[0]), float(lo[1]), float(lo[2])),
            (float(hi[0]), float(hi[1]), float(hi[2])),
        )
        d_spacing /= 8.0
    # n_refine + 1 >= 1 for the documented (non-negative) n_refine, so the
    # loop above always runs at least once and assigns vals.
    assert vals is not None, "measure_gap requires n_refine >= 0"
    return float(vals.min())


def gap_audit(
    part: Part,
    declared: dict[tuple[str, str], float],
    d_voxel: float = 0.25,
    d_tol: float = 0.02,
    chunk_size: int = 32_768,
) -> list[GapCheck]:
    """Measure every declared interface of ``part`` against its declared clearance.

    ``declared`` maps ``(region_name_a, region_name_b)`` to the design clearance for
    that interface -- the same numbers the Part's params promise. Region names are
    ``MaterialRegion.name``. Returns one :class:`GapCheck` per pair without raising;
    use :func:`assert_gaps` in tests.
    """
    regions = {r.name: r for r in part.materials}
    closures: dict[str, SDFCallable] = {}
    boxes: dict[str, BBox] = {}
    checks: list[GapCheck] = []
    for (name_a, name_b), d_declared in declared.items():
        for name in (name_a, name_b):
            if name not in regions:
                raise KeyError(
                    f"gap_audit: part has no material region {name!r} (has: {sorted(regions)})"
                )
            if name not in closures:
                closures[name] = make_sdf_closure(regions[name].sdf_tree, part)
                boxes[name] = infer_material_bbox(regions[name], part)
        (a0, a1), (b0, b1) = boxes[name_a], boxes[name_b]
        lo = tuple(min(a, b) for a, b in zip(a0, b0, strict=False))
        hi = tuple(max(a, b) for a, b in zip(a1, b1, strict=False))
        enclosing: BBox = (
            (lo[0], lo[1], lo[2]),
            (hi[0], hi[1], hi[2]),
        )
        d_measured = measure_gap(
            closures[name_a],
            closures[name_b],
            pad_bbox(enclosing, d_voxel),
            d_voxel=d_voxel,
            chunk_size=chunk_size,
        )
        checks.append(GapCheck(name_a, name_b, float(d_declared), d_measured, d_tol))
    return checks


def assert_gaps(
    part: Part,
    declared: dict[tuple[str, str], float],
    d_voxel: float = 0.25,
    d_tol: float = 0.02,
    chunk_size: int = 32_768,
) -> list[GapCheck]:
    """:func:`gap_audit` that raises :class:`GapAuditError` listing every failed interface."""
    checks = gap_audit(part, declared, d_voxel=d_voxel, d_tol=d_tol, chunk_size=chunk_size)
    failures = [c for c in checks if not c.b_ok]
    if failures:
        raise GapAuditError("gap audit failed:\n" + "\n".join(f"  {c}" for c in failures))
    return checks
