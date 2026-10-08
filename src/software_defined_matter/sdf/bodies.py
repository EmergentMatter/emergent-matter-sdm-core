"""How many separate solid pieces a Part is made of.

The modelling rule (see the SDM docs hub, concepts: Parts and Assemblies) is
that a Part is ONE physically inseparable object: anything that can be
separated is its own Part, placed by an Assembly. A Part whose solid falls
apart into several disconnected pieces cannot be one physical object, which
is the usual sign of separate parts packed into one file's material regions.

The check samples the WHOLE Part, every material region together, on a
coarse grid and counts the connected pieces of the inside cells. Taking the
regions together keeps a co-made multi-material part legitimate: a copper
coil printed inside a steel body is several pieces of copper but one
connected object.

It is a heuristic, so it WARNS rather than refuses: a gap thinner than a
grid cell reads as connected (two parts with a small clearance look like
one), and a neck thinner than a cell can read as broken. It cannot tell
whether touching pieces are bonded or merely in contact; that is the
designer's call.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from software_defined_matter.model import Part
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure

__all__ = ["PartBodies", "PartBodiesWarning", "count_part_bodies", "check_part_is_one_body"]


class PartBodiesWarning(UserWarning):
    """A Part's solid is several disconnected pieces: likely separate parts in one file."""


@dataclass(frozen=True)
class PartBodies:
    """The result of `count_part_bodies`.

    ``n_bodies`` is the number of connected solid pieces found; ``voxels``
    the cell count of each, largest first; ``d_cell`` the grid spacing used
    (part units).
    """

    n_bodies: int
    voxels: tuple[int, ...]
    d_cell: float


def _label(inside: np.ndarray) -> np.ndarray:
    """Connected-component labels of a 3-D boolean grid (6-connected), -1 outside.

    Label propagation: every inside cell starts with its own index and
    repeatedly takes the smallest label among itself and its inside
    neighbours until nothing changes. Vectorised; the number of sweeps is
    bounded by the longest path through a piece.
    """
    big = np.iinfo(np.int64).max
    lab = np.where(inside, np.arange(inside.size, dtype=np.int64).reshape(inside.shape), big)
    while True:
        new = lab.copy()
        for ax in range(3):
            for s in (1, -1):
                shifted = np.roll(lab, s, axis=ax)
                edge = [slice(None)] * 3
                edge[ax] = slice(0, 1) if s == 1 else slice(-1, None)
                shifted[tuple(edge)] = big  # no wrap-around
                new = np.minimum(new, shifted)
        new = np.where(inside, new, big)
        if np.array_equal(new, lab):
            break
        lab = new
    return np.where(inside, lab, -1)


def count_part_bodies(part: Part, *, n_cells: int = 64) -> PartBodies:
    """Count the connected solid pieces of ``part`` (all material regions together).

    The grid spans the union of the regions' bounding boxes (``metadata['bbox']``
    when the part carries one, else inferred) with about ``n_cells`` cells
    along its longest side.

    Args:
        part: The Part to check.
        n_cells: Cells along the longest side of the box. Finer finds thinner
            gaps between pieces and costs n_cells**3 evaluations.

    Returns:
        PartBodies: the count, each piece's size in cells, the spacing used.
    """
    if not part.materials:
        return PartBodies(0, (), 0.0)
    box = (part.metadata or {}).get("bbox")
    if box is not None:
        lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    else:
        los, his = [], []
        for region in part.materials:
            a, b = infer_sdf_bbox(region.sdf_tree, part, mode="values")
            los.append(a)
            his.append(b)
        lo, hi = np.min(np.asarray(los, float), axis=0), np.max(np.asarray(his, float), axis=0)
    d = float(np.max(hi - lo)) / float(n_cells)
    axes = [np.arange(lo[k] + 0.5 * d, hi[k], d) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = jnp.asarray(np.stack([gx, gy, gz], axis=-1).reshape(-1, 3))
    inside = np.zeros(pts.shape[0], dtype=bool)
    for region in part.materials:
        fn = jax.jit(make_sdf_closure(region.sdf_tree, part))
        inside |= np.asarray(fn(pts, None)) < 0.0
    lab = _label(inside.reshape(gx.shape))
    _, counts = np.unique(lab[lab >= 0], return_counts=True)
    sizes = tuple(sorted((int(c) for c in counts), reverse=True))
    return PartBodies(len(sizes), sizes, d)


def check_part_is_one_body(part: Part, *, n_cells: int = 64) -> PartBodies:
    """Count ``part``'s pieces and warn (`PartBodiesWarning`) when there is more than one.

    Returns the `PartBodies` either way so a caller can report it.
    """
    result = count_part_bodies(part, n_cells=n_cells)
    if result.n_bodies > 1:
        warnings.warn(
            f"Part {part.name!r} is {result.n_bodies} disconnected pieces (cell sizes "
            f"{list(result.voxels)}, grid {result.d_cell:.3g}). A Part should be one physical, "
            "inseparable object; make each separable piece its own Part and place them with "
            "an Assembly.",
            PartBodiesWarning,
            stacklevel=2,
        )
    return result
