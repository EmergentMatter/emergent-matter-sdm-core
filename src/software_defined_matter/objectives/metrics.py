"""Differentiable metric registry used by the objective / constraint expression DSL.

A metric is any callable with the signature::

    metric(*, sdf, part, binding, free_vec, bbox, grid_resolution, args) -> jnp.ndarray

It samples the compiled SDF on a regular voxel grid inside the bounding box
and returns a JAX scalar (or vector) that is differentiable with respect to
``free_vec`` via the SDF closure.

Register a new metric with :func:`register_metric` or the
``@register_metric("name")`` decorator. Built-in metrics:

- ``volume``          - ``int Vol K(-sdf) dV`` (K: metric kernel)
- ``surface_area``    - ``int Vol |grad sdf| delta(sdf) dV``
- ``mass``            - ``volume * density``; the density is passed in via
                        ``args["density"]`` (looked up from the Matter
                        Library at use-site; not stored on the Part)
- ``relative_density``- material volume over *envelope* volume: the fraction
                        of the part's own design volume that is solid
- ``bbox_extent``     - axis-aligned size of the occupied region along one axis
- ``centroid``        - occupied-region centroid along one axis
- ``max_value``       - soft maximum of the SDF inside the bbox
- ``min_value``       - soft minimum of the SDF inside the bbox

How a cell's filled fraction is computed
----------------------------------------
Every volume-like metric turns a sampled distance into an occupancy with
:func:`fraction_filled`::

    fraction_filled(d, h) = clip(0.5 - d/h, 0, 1)

``d`` is the SDF at the cell centre and ``h`` is the cell edge length, so the
ramp is exactly one cell wide.

A slab half a cell thick reads anywhere from +0% to +49% depending on where
it sits between samples. The :func:`check_grid_resolution` exists to catch that.

Edges and corners are where a real solid stops being flat, and they carry an
``O(h^2)`` error. A box of 8x6x4 mm reads +1.67% at 16 cells per axis and
+0.026% at 128, halving twice per doubling. Curved surfaces behave the same
way: a sphere reads +0.21% at 24 and +0.013% at 96.

Two consequences follow from the ramp having compact support. Cells more than
``h/2`` outside the surface contribute exactly zero, so the sampling box needs
no padding and extra empty cells cannot bias the result. And the derivative is
``-1/h`` on the band and zero elsewhere, which is a surface integral over
exactly the cells that can respond to a parameter change.

For an *oblique* surface the telescoping does not apply and a wall thinner than
about four voxels is over-counted (+12.9% at one voxel, +78.9% at a quarter).
:func:`check_grid_resolution` refuses to return a number in that regime rather
than returning a quietly wrong one.

``surface_area`` is the exception: it needs ``-d(fraction_filled)/dd``, which is
a rectangle exactly one cell wide, and the count of samples falling inside such
a band is an integer that jumps. On a cube whose faces land midway between
sample points it counts nothing at all. It therefore keeps a smooth delta of
width ``0.7 h`` (see :data:`SURFACE_DELTA_CELLS`), at the cost of no longer
being exactly ``jax.grad`` of ``volume``.

Sampling-domain contract
------------------------
``bbox`` is the numerical integration domain, **not** the geometry. Build it
with :func:`cubic_grid`, which picks a cell size, rounds each axis up to a whole
number of cells plus one of margin, and returns cells that are exactly cubic.
Cubic cells matter because ``h`` is a single number: on a 1.25-aspect cell any
choice of ``h`` is wrong for at least one axis.

Sample points sit at cell centres offset by :data:`GRID_PHASE`, an irrational
fraction of a cell. Values are unaffected by the offset (the telescoping holds
at *any* offset), but a surface landing exactly on a sample point makes every
kernel's gradient read half its true value, and round-number geometry on a
round-number box lands there often.

Caveat (optimisation): the box is a constant w.r.t. ``free_vec``
(stop-gradient by construction). Its *tightness* is only per outer iteration:
callers (see :func:`software_defined_matter.dsl.expr.compile_expr`) must
re-derive it after :meth:`Part.update_from_vector` each optimiser step. A
compiled metric fn reused across many steps without re-deriving keeps a stale
box; use ``mode='bounds'`` (worst-case over ``Param.bounds``) in that case.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from software_defined_matter.model import Part, SDFTree


MetricFn = Callable[..., jnp.ndarray]

#: A compiled SDF closure: an ``(N, 3)`` points array in, an ``(N,)`` distance
#: array out. Matches the closures returned by
#: :func:`software_defined_matter.sdf.compile.make_sdf_closure`.
SDFFn = Callable[[jnp.ndarray], jnp.ndarray]

#: Axis-aligned bbox as ``((xlo, ylo, zlo), (xhi, yhi, zhi))``. Kept as a
#: local alias (rather than importing ``software_defined_matter.sdf.bbox``)
#: since only the shape, not the inference logic, matters here.
BBoxLike = tuple[tuple[float, float, float], tuple[float, float, float]]

#: Per-axis sample counts, or a single int for a cubic grid on every axis.
GridResolution = int | tuple[int, int, int] | list[int]

_REGISTRY: dict[str, MetricFn] = {}


#: Offset of the sample points from the cell centres, as a fraction of a cell.
#: ``(sqrt(5)-1)/2 - 1/2``, so a sample sits at an irrational fraction of its
#: cell. A surface at a round coordinate never lands exactly on a sample.
GRID_PHASE = (math.sqrt(5.0) - 1.0) / 2.0 - 0.5

#: Width of ``surface_area``'s smooth delta, in cells. Narrower bands make the
#: sample count inside the band jump; wider ones smooth away real curvature.
SURFACE_DELTA_CELLS = 0.7

#: Voxels required across the thinnest nameable feature. Below this an oblique
#: wall is over-counted by more than a percent or so, and by tens of percent
#: below one voxel.
MIN_FEATURE_VOXELS = 4.0

#: Default cells per axis when no ``voxel_size`` is given, applied to the
#: longest axis of the sampling box.
DEFAULT_GRID_COUNT = 64


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator. Register ``fn`` in the global metric registry under ``name``."""

    def _wrap(fn: MetricFn) -> MetricFn:
        if name in _REGISTRY:
            raise ValueError(f"Metric {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return _wrap


def get_metric(name: str) -> MetricFn:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_metrics() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------


def fraction_filled(d: jnp.ndarray, d_band: float) -> jnp.ndarray:
    """Fraction of a cell that is inside the solid.

    ``d`` is the signed distance at the sample point and ``d_band`` is the ramp
    width, normally one cell edge.
    """
    return jnp.clip(0.5 - d / d_band, 0.0, 1.0)


def _soft_delta(x: jnp.ndarray, d_tau: float) -> jnp.ndarray:
    """Smooth approximation of the surface delta. Integral over x is 1.

    Used only by ``surface_area``. This is ``-d/dx`` of a *sigmoid* occupancy,
    not of :func:`fraction_filled`.
    """
    s = jax.nn.sigmoid(x / d_tau)
    return s * (1.0 - s) / d_tau


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def cubic_grid(
    bbox: BBoxLike,
    *,
    voxel_size: float | None = None,
    count: int | None = None,
    cap: int = 192,
) -> tuple[BBoxLike, tuple[int, int, int], float]:
    """Return ``(bbox, (Rx, Ry, Rz), h)`` with exactly cubic cells.

    Picks a cell size, rounds each axis up to a whole number of cells, adds one
    cell of margin per axis so the phase offset cannot uncover geometry near a
    face, and re-centres the box on its original centre. The returned box is
    therefore never smaller than the one passed in.

    ``voxel_size`` (mm) sets ``h`` directly. Otherwise ``count`` cells span the
    *longest* axis. ``cap`` bounds the per-axis count; when the request exceeds
    it, ``h`` is raised uniformly so the cells stay cubic rather than clamping
    one axis and skewing them.
    """
    (x0, y0, z0), (x1, y1, z1) = bbox
    ext = [abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)]
    longest = max(ext)

    if voxel_size is not None:
        h = float(voxel_size)
    else:
        n = int(count) if count is not None else DEFAULT_GRID_COUNT
        h = longest / max(n, 1)
    if not h > 0.0:
        raise ValueError(f"Cell size must be positive, got {h!r}")

    # +1 cell of margin per axis, so a sample offset by GRID_PHASE still has
    # at least half a cell of clearance beyond the geometry on every face.
    res = [max(math.ceil(e / h) + 1, 2) for e in ext]
    if max(res) > cap:
        h *= max(res) / float(cap)
        res = [max(math.ceil(e / h) + 1, 2) for e in ext]
        # Rounding up can still leave one axis a cell over the cap.
        res = [min(r, cap) for r in res]

    centre = [0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)]
    half = [0.5 * r * h for r in res]
    cx, cy, cz = centre
    hx, hy, hz = half
    grown: BBoxLike = ((cx - hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz + hz))
    return grown, (res[0], res[1], res[2]), h


def grid_resolution_for(
    bbox: BBoxLike,
    *,
    count: int | None = None,
    voxel_size: float | None = None,
    cap: int = 192,
) -> tuple[int, int, int]:
    """Per-axis sample counts ``(Rx, Ry, Rz)`` for a cubic-celled grid.

    Thin wrapper over :func:`cubic_grid` for callers that only want the counts.
    Prefer :func:`cubic_grid`, whose returned box matches those counts: using
    these counts against the *original* box reintroduces non-cubic cells.
    """
    return cubic_grid(bbox, voxel_size=voxel_size, count=count, cap=cap)[1]


def _res3(res: GridResolution) -> tuple[int, int, int]:
    """Normalise an int (cubic, legacy) or ``(Rx, Ry, Rz)`` to a 3-tuple."""
    if isinstance(res, (tuple, list)):
        return (int(res[0]), int(res[1]), int(res[2]))
    return (int(res), int(res), int(res))


def _voxel_grid(bbox: BBoxLike, n_resolution: GridResolution) -> jnp.ndarray:
    """Return an ``(Rx,Ry,Rz,3)`` grid of sample points.

    Points sit at cell centres shifted by :data:`GRID_PHASE`. Note this is a
    *cell-centre* grid at spacing ``L/R``, not ``linspace``'s ``L/(R-1)`` with
    both endpoints included: the spacing has to match the ``L^3/R^3`` weight
    each sample carries, or the quadrature carries a ``-3/R`` bias.
    """
    (x0, y0, z0), (x1, y1, z1) = bbox
    Rx, Ry, Rz = _res3(n_resolution)
    off = 0.5 + GRID_PHASE
    xs = x0 + (jnp.arange(Rx) + off) * (x1 - x0) / Rx
    ys = y0 + (jnp.arange(Ry) + off) * (y1 - y0) / Ry
    zs = z0 + (jnp.arange(Rz) + off) * (z1 - z0) / Rz
    gx, gy, gz = jnp.meshgrid(xs, ys, zs, indexing="ij")
    return jnp.stack([gx, gy, gz], axis=-1)


def _cell_size(bbox: BBoxLike, n_resolution: GridResolution) -> float:
    """Cell edge length. Equal on all axes for a :func:`cubic_grid` box."""
    (x0, y0, z0), (x1, y1, z1) = bbox
    Rx, Ry, Rz = _res3(n_resolution)
    return max(abs(x1 - x0) / Rx, abs(y1 - y0) / Ry, abs(z1 - z0) / Rz)


def _voxel_volume(bbox: BBoxLike, n_resolution: GridResolution) -> jnp.ndarray:
    (x0, y0, z0), (x1, y1, z1) = bbox
    Rx, Ry, Rz = _res3(n_resolution)
    return jnp.asarray(((x1 - x0) * (y1 - y0) * (z1 - z0)) / (Rx * Ry * Rz))


def _bbox_volume(bbox: BBoxLike) -> jnp.ndarray:
    (x0, y0, z0), (x1, y1, z1) = bbox
    return jnp.asarray((x1 - x0) * (y1 - y0) * (z1 - z0))


def _band_cells(part: Part, args: dict[str, Any]) -> float:
    """Ramp width in *cells*. One cell unless deliberately widened.

    ``metric_tau`` used to set an absolute width in mm; it is read here only
    to warn. Reinterpreting 0.25 mm as a quarter of a cell would
    break the exactness the one-cell ramp provides.
    """
    if "band_cells" in args:
        return float(args["band_cells"])
    if "metric_tau" in part.metadata or "tau" in args:
        warnings.warn(
            "metric_tau / args['tau'] set an absolute smoothing width in mm and "
            "no longer have any effect: the occupancy ramp is now exactly one "
            "voxel wide, which is what makes it exact for axis-aligned "
            "surfaces. Use metadata['metric_band_cells'] (in cells) to widen "
            "it, or delete the key.",
            DeprecationWarning,
            stacklevel=3,
        )
    return float(part.metadata.get("metric_band_cells", 1.0))


def _band(part: Part, args: dict[str, Any], d_h: float) -> float:
    return _band_cells(part, args) * d_h


def _grid_and_band(
    part: Part,
    bbox: BBoxLike,
    grid_resolution: GridResolution,
    args: dict[str, Any],
    cell_size: float | None = None,
) -> tuple[jnp.ndarray, float, float]:
    """The three things every volume-like metric needs:
    - voxel grid
    - cell size
    - band
    """
    d_h = float(cell_size) if cell_size is not None else _cell_size(bbox, grid_resolution)
    return _voxel_grid(bbox, grid_resolution), d_h, _band(part, args, d_h)


def check_grid_resolution(
    tree: SDFTree,
    part: Part,
    d_h: float,
    *,
    min_voxels: float = MIN_FEATURE_VOXELS,
) -> None:
    """Raise if a cell of size ``d_h`` cannot resolve ``tree``'s thinnest wall.

    Silent when the tree carries no nameable feature size, which is an absence
    of evidence rather than a clean bill of health; see
    :mod:`software_defined_matter.sdf.features`.
    """
    from software_defined_matter.sdf.features import infer_min_feature_size

    if part.metadata.get("metric_skip_resolution_check"):
        return
    d_feature = infer_min_feature_size(tree, part)
    if d_feature is None:
        return
    if d_feature <= 0.0:
        raise ValueError(
            f"Part {part.name!r} has a feature whose size can reach 0 mm, which "
            "no grid resolves. Give the parameter a positive lower bound."
        )
    spans = d_feature / d_h
    if spans >= min_voxels:
        return
    needed = d_feature / min_voxels
    raise ValueError(
        f"Grid too coarse for part {part.name!r}: cells are {d_h:.4g} mm and the "
        f"thinnest feature is {d_feature:.4g} mm, so it spans {spans:.2f} voxels "
        f"where {min_voxels:g} are needed.\n"
        f"An oblique wall this thin is over-counted by roughly "
        f"{'80%' if spans < 0.5 else '13%' if spans < 1.5 else '2%'}, so the "
        f"metric refuses rather than returning it.\n"
        f"Fix: set metadata['metric_voxel_size'] = {needed:.4g} (or smaller), or "
        f"raise metadata['grid_resolution']. Note the 192-per-axis cap in "
        f"metadata['metric_grid_cap'] may make that unreachable for a large "
        f"part."
    )


# ===========================================================================
# Built-in metrics
# ===========================================================================


@register_metric("volume")
def _metric_volume(
    *,
    sdf: SDFFn,
    part: Part,
    bbox: BBoxLike,
    grid_resolution: GridResolution,
    args: dict[str, Any],
    cell_size: float | None = None,
    **_: Any,
) -> jnp.ndarray:
    grid, d_h, band = _grid_and_band(part, bbox, grid_resolution, args, cell_size)
    return jnp.sum(fraction_filled(sdf(grid), band)) * d_h**3


@register_metric("surface_area")
def _metric_surface_area(
    *,
    sdf: SDFFn,
    part: Part,
    bbox: BBoxLike,
    grid_resolution: GridResolution,
    args: dict[str, Any],
    cell_size: float | None = None,
    **_: Any,
) -> jnp.ndarray:
    grid, d_h, _band_unused = _grid_and_band(part, bbox, grid_resolution, args, cell_size)

    def sdf_scalar(p: jnp.ndarray) -> jnp.ndarray:
        return sdf(p[None, :])[0]

    grad_fn = jax.vmap(jax.grad(sdf_scalar))
    flat = grid.reshape(-1, 3)
    grads = grad_fn(flat)
    grad_norm = jnp.linalg.norm(grads, axis=-1).reshape(grid.shape[:-1])

    # A smooth delta, NOT the derivative of fraction_filled. The one-cell
    # rectangle that derivative gives holds an integer number of samples, and
    # on a cube whose faces land midway between samples that integer is zero.
    tau = float(args.get("delta_cells", SURFACE_DELTA_CELLS)) * d_h
    delta = _soft_delta(sdf(grid), tau)
    return jnp.sum(grad_norm * delta) * d_h**3


@register_metric("mass")
def _metric_mass(
    *,
    sdf: SDFFn,
    part: Part,
    bbox: BBoxLike,
    grid_resolution: GridResolution,
    args: dict[str, Any],
    cell_size: float | None = None,
    **_: Any,
) -> jnp.ndarray:
    """``mass = volume * density``.

    Physical properties are not stored on the ``Part``; the ``density``
    value must be supplied via ``args`` (the caller typically looks it up
    in the Matter Library and injects it, e.g.
    ``expr_metric("mass", density=8960.0)``).
    """
    if "density" not in args:
        raise ValueError(
            "metric 'mass' requires a 'density' arg. "
            "Look it up via emergent_matter_materials.get(name, 'rho') "
            "and pass it in, e.g. expr_metric('mass', density=8940.0)."
        )
    rho = float(args["density"])
    grid, d_h, band = _grid_and_band(part, bbox, grid_resolution, args, cell_size)
    return jnp.sum(fraction_filled(sdf(grid), band)) * d_h**3 * rho


@register_metric("bbox_extent")
def _metric_bbox_extent(
    *,
    sdf: SDFFn,
    part: Part,
    bbox: BBoxLike,
    grid_resolution: GridResolution,
    args: dict[str, Any],
    cell_size: float | None = None,
    **_: Any,
) -> jnp.ndarray:
    axis = {"x": 0, "y": 1, "z": 2}.get(str(args.get("axis", "x")).lower(), 0)
    grid, _d_h, band = _grid_and_band(part, bbox, grid_resolution, args, cell_size)
    occ = fraction_filled(sdf(grid), band)

    coords = grid[..., axis]
    # Weighted soft min / max along the occupied region.
    w = occ + 1e-12
    # soft-max via weighted log-sum-exp
    beta = float(args.get("beta", 5.0))
    smax = jnp.log(jnp.sum(w * jnp.exp(beta * (coords - jnp.max(coords))))) / beta + jnp.max(coords)
    smin = -jnp.log(jnp.sum(w * jnp.exp(-beta * (coords - jnp.min(coords))))) / beta + jnp.min(
        coords
    )
    return smax - smin


@register_metric("centroid")
def _metric_centroid(
    *,
    sdf: SDFFn,
    part: Part,
    bbox: BBoxLike,
    grid_resolution: GridResolution,
    args: dict[str, Any],
    cell_size: float | None = None,
    **_: Any,
) -> jnp.ndarray:
    axis = {"x": 0, "y": 1, "z": 2}.get(str(args.get("axis", "x")).lower(), 0)
    grid, _d_h, band = _grid_and_band(part, bbox, grid_resolution, args, cell_size)
    occ = fraction_filled(sdf(grid), band)
    coords = grid[..., axis]
    total = jnp.sum(occ) + 1e-12
    return jnp.sum(coords * occ) / total


@register_metric("relative_density")
def _metric_relative_density(
    *,
    sdf: SDFFn,
    part: Part,
    bbox: BBoxLike,
    grid_resolution: GridResolution,
    args: dict[str, Any],
    cell_size: float | None = None,
    env_sdf: SDFFn | None = None,
    envelope_volume: jnp.ndarray | float | None = None,
    **_: Any,
) -> jnp.ndarray:
    """Fraction of the part's own design volume that is solid.

    The denominator is the part's **envelope**: the same solid with its holes
    filled back in, obtained by walking the SDF tree (see
    :mod:`software_defined_matter.sdf.envelope`). A solid ball reads 1.0, a
    gyroid clipped to a sphere reads the gyroid's fill fraction, and a lattice
    inside a skin reads material over outer-sphere volume.

    ``compile_expr`` derives that envelope, compiles it over the same
    ``ParamBinding``, and injects it as ``env_sdf`` (a ``p -> distance``
    closure with ``free_vec`` already bound, like ``sdf``). Pass
    ``args["envelope"]`` to override it with a region of your own, in which
    case keeping that region inside the sampling box is your responsibility:
    the metric integrates what it can see and does not check.

    Both volumes are quadratures of the same compiled fields on the same grid,
    so ``jax.grad`` flows through numerator and denominator alike. A solid part
    therefore reads exactly 1.0 with a gradient of zero, which is correct and
    makes it useless as an objective for non-porous geometry.

    Falls back to the sampling-box volume when called directly with no
    envelope, which over-counts by exactly the box's empty space. Prefer the
    compiled path.
    """
    grid, d_h, band = _grid_and_band(part, bbox, grid_resolution, args, cell_size)
    d_cell = d_h**3
    d_part = sdf(grid)

    if env_sdf is None:
        v_part = jnp.sum(fraction_filled(d_part, band)) * d_cell
        d_env_vol = envelope_volume if envelope_volume is not None else _bbox_volume(bbox)
        return v_part / jnp.maximum(d_env_vol, d_cell)

    d_env = env_sdf(grid)
    # The envelope contains the part, so max() is the part's own field for the
    # walked envelope and a genuine intersection for a caller-supplied one.
    v_num = jnp.sum(fraction_filled(jnp.maximum(d_part, d_env), band)) * d_cell
    v_den = jnp.sum(fraction_filled(d_env, band)) * d_cell
    # Floor at one cell: below that the denominator is not resolved at all, and
    # 1e-12 mm^3 would be meaningless at either end of the scale range.
    return v_num / jnp.maximum(v_den, d_cell)


@register_metric("max_value")
def _metric_max_value(
    *,
    sdf: SDFFn,
    bbox: BBoxLike,
    grid_resolution: GridResolution,
    args: dict[str, Any],
    part: Part,
    **_: Any,
) -> jnp.ndarray:
    grid = _voxel_grid(bbox, grid_resolution)
    d = sdf(grid).ravel()
    beta = float(args.get("beta", 5.0))
    return jax.scipy.special.logsumexp(beta * d) / beta - jnp.log(d.size) / beta


@register_metric("min_value")
def _metric_min_value(
    *,
    sdf: SDFFn,
    bbox: BBoxLike,
    grid_resolution: GridResolution,
    args: dict[str, Any],
    part: Part,
    **_: Any,
) -> jnp.ndarray:
    grid = _voxel_grid(bbox, grid_resolution)
    d = sdf(grid).ravel()
    beta = float(args.get("beta", 5.0))
    return -(jax.scipy.special.logsumexp(-beta * d) / beta - jnp.log(d.size) / beta)


# ---------------------------------------------------------------------------
# Not-yet-implemented (belong to optimiser back-ends)
# ---------------------------------------------------------------------------


def _unimplemented(name: str) -> MetricFn:
    def _fn(**_: Any) -> jnp.ndarray:
        raise NotImplementedError(
            f"Metric {name!r} is reserved for an optimiser back-end "
            f"and is not computable from the SDF alone."
        )

    return _fn


for _name in (
    "max_von_mises",
    "compliance",
    "thermal_resistance",
    "eigenfrequency",
    "em_absorption",
):
    _REGISTRY[_name] = _unimplemented(_name)


__all__ = [
    "GRID_PHASE",
    "MIN_FEATURE_VOXELS",
    "SURFACE_DELTA_CELLS",
    "BBoxLike",
    "GridResolution",
    "MetricFn",
    "SDFFn",
    "check_grid_resolution",
    "cubic_grid",
    "fraction_filled",
    "get_metric",
    "grid_resolution_for",
    "list_metrics",
    "register_metric",
]
