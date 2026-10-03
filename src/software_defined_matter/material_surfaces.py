"""Bounded ray queries for posed material; unresolved intervals are never misses.

The search uses a continuous exclusion field only to certify uniform intervals.
A zero of that field is not a surface: a hit requires a demonstrated change in
actual material occupancy. Owners with the same inverse are combined before
building the field, removing partition seams when their poses agree.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from software_defined_matter.material_motion import MaterialMotionEval, compile_material_motion
from software_defined_matter.model import Part
from software_defined_matter.sdf.bbox import BBox

__all__ = ["MaterialSurfaces", "SurfacePose", "SurfaceHit", "compile_material_surfaces"]


@dataclass(frozen=True)
class SurfaceHit:
    """Bounded ray result, including an optional witnessed tolerance contact.

    ``status`` is hit, miss, unresolved, or contact. A hit proves an occupancy change
    within ``interval`` after a certified uniform prefix; it is not an exact
    root. Features smaller than the bracket need not be individually resolved.
    An unresolved interval must not be rendered as empty. ``normal`` is a
    finite-difference estimate at the bracket midpoint; corners have no unique
    normal and a zero/invalid gradient yields None.

    A contact retains the earliest uncertain ray window in interval, and names
    a ray point with contact_t. contact_segment contains inside/outside witness
    endpoints within contact_tolerance of that point. This proves a nearby
    boundary, not an exact ray intersection or a bracket of the exact tangent
    parameter. Its normal is estimated at contact_t. Near misses within the
    allowance may produce contacts; this is not a complete proximity search.
    """

    status: str
    interval: tuple[float, float] | None
    normal: tuple[float, float, float] | None = None
    contact_t: float | None = None
    contact_segment: tuple[tuple[float, ...], tuple[float, ...]] | None = None


def _supported(tree: dict[str, Any]) -> None:
    """Restrict rate inference to continuous, analysed geometry, not all primitives.

    In particular raster sample amplitudes have no universal unit rate, and
    repeated/folded coordinates can introduce discontinuities in arbitrary trees.
    """
    kind = tree.get("type")
    allowed = (
        kind == "primitive"
        and tree.get("kind")
        in {
            "sphere",
            "box",
            "capped_cylinder",
            "capsule",
            "torus",
            "plane",
            "circle_2d",
            "box_2d",
            "polygon_2d",
        }
        or kind == "op"
        and tree.get("op")
        in {
            "union",
            "intersect",
            "subtract",
            "smooth_union",
            "smooth_intersect",
            "smooth_subtract",
        }
        or kind == "transform"
        and tree.get("transform") in {"translate", "rotate_x", "rotate_y", "rotate_z", "scale"}
        or kind == "modifier"
        and tree.get("modifier") in {"round", "onion"}
        or kind == "2d_to_3d"
        and tree.get("method") in {"extrusion", "revolution"}
        or kind == "deform"
        and tree.get("deform")
        in {"twist", "bend", "twist_radial", "twist_linear", "shear_linear", "taper_linear"}
    )
    if not allowed:
        name = tree.get("kind", tree.get("op", tree.get(str(kind), kind)))
        raise ValueError(f"Surface finding has no supported continuous rate contract for {name!r}")
    for child in tree.get("children", []):
        _supported(child)
    if "child" in tree:
        _supported(tree["child"])


def _primitive_scale(tree: dict[str, Any], intervals: Any) -> float:
    """Plane fields use the authored normal verbatim, which need not be unit length."""
    from software_defined_matter.sdf.bbox import _vec_intervals

    rate = 1.0
    if tree.get("type") == "primitive" and tree.get("kind") == "plane":
        normal = _vec_intervals(tree["params"]["n"], 3, intervals)
        rate = max(rate, math.hypot(*(max(abs(a), abs(b)) for a, b in normal)))
    for child in tree.get("children", []):
        rate = max(rate, _primitive_scale(child, intervals))
    if "child" in tree:
        rate = max(rate, _primitive_scale(tree["child"], intervals))
    return rate


def _invariant_classifiers(part: Part, trees: tuple[dict[str, Any], ...]) -> bool:
    """Prove classifier invariance under a common authored rotation axis.

    Compare exact rational representations of snapshot coefficients. Almost
    coaxial axes or almost centred classifiers must not enable a coverage proof.
    This is a sufficient structural test, not a numerical sampling heuristic.
    """
    from fractions import Fraction

    from software_defined_matter.sdf.bbox import (
        _scalar_interval,
        _vec_intervals,
        build_param_intervals,
    )

    def vector(xs: Any) -> tuple[Fraction, ...]:
        return tuple(Fraction(float(x)) for x in xs)

    def parallel(a: Any, b: Any) -> bool:
        return all(a[i] * b[j] == a[j] * b[i] for i in range(3) for j in range(3))

    block = part.kinematics
    assert block is not None
    ops = [op for body in block["bodies"] for op in body["motion"]["ops"]]
    if not ops:
        return True
    if any(op["kind"] != "rotate" for op in ops):
        return False
    axis = vector(ops[0]["axis"])
    pivot = vector(ops[0].get("origin", [0, 0, 0]))
    for op in ops:
        offset = tuple(
            a - b for a, b in zip(vector(op.get("origin", [0, 0, 0])), pivot, strict=True)
        )
        if not parallel(vector(op["axis"]), axis) or not parallel(offset, axis):
            return False
    intervals = build_param_intervals(part, mode="values")

    def invariant(node: dict[str, Any], origin: Any) -> bool:
        kind = node["type"]
        params = node.get("params", {})
        if kind == "primitive":
            primitive = node["kind"]
            if primitive == "sphere":
                return parallel(origin, axis)
            if primitive == "plane":
                normal = _vec_intervals(params["n"], 3, intervals)
                return all(a == b for a, b in normal) and parallel(
                    vector([a for a, _ in normal]), axis
                )
            if primitive in {"capped_cylinder", "torus"}:
                return parallel(axis, (0, 0, 1)) and parallel(origin, axis)
            return False
        if kind == "op":
            return all(invariant(child, origin) for child in node["children"])
        if kind == "modifier":
            return invariant(node["child"], origin)
        if kind == "transform":
            if node["transform"] == "translate":
                offsets = _vec_intervals(params["t"], 3, intervals)
                if not all(a == b for a, b in offsets):
                    return False
                shifted = tuple(
                    a - b for a, b in zip(origin, vector([a for a, _ in offsets]), strict=True)
                )
                return invariant(node["child"], shifted)
            if node["transform"] == "scale":
                lo, hi = _scalar_interval(params["s"], intervals)
                if lo == hi and lo != 0:
                    return invariant(node["child"], tuple(x / Fraction(float(lo)) for x in origin))
        return False

    return all(invariant(tree, pivot) for tree in trees)


@dataclass(frozen=True)
class _FlexureRayRate:
    axis: tuple[float, ...]
    blend_axis: tuple[float, ...]
    pivot: tuple[float, ...]
    slope: float
    radial: bool


@dataclass(frozen=True)
class MaterialSurfaces:
    """Design snapshot for a finite query domain and supported material geometry.

    Recompile after any design or geometry edit. Prepare a new pose after every
    DOF edit. Rate bounds apply only to the supplied query domain. Bounds use
    ordinary floating-point arithmetic with a field-error allowance, not formal
    interval rounding. Unknown geometry/rates fail explicitly at preparation.
    ownership_invariant advertises a structural proof that all classifier
    fields are unchanged by the common authored rotation axis. It permits a
    solid-coverage proof across internal welds; no symmetry of material trees
    is assumed. Other classifier/motion combinations retain conservative search.
    This API does not enable whole-part emit_glsl or change viewer rendering.
    """

    membership: MaterialMotionEval
    domain: BBox
    field_error: float
    ownership_invariant: bool
    _part: Part
    _trees: tuple[dict[str, Any], ...]
    _field: Any
    _contains: Any

    def prepare(self, dofs: Any) -> SurfacePose:
        """Freeze canonical radians/mm inputs and compute inverse groups and rates.

        No advertised DOF-range assumption is needed: bounds are recomputed for
        this pose. Equality grouping is exact, never based on a pose tolerance.
        Equal endpoint matrices alone cannot collapse a flexure with full turns.
        """
        import jax.numpy as jnp
        import numpy as np

        from software_defined_matter.sdf.bbox import build_param_intervals
        from software_defined_matter.sdf.lipschitz import infer_sdf_max_rate

        kin = self.membership.kinematics
        values = np.asarray(kin._dofs(dofs), dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Surface pose DOFs must be finite")
        matrices = list(np.asarray(kin.body_transforms(values)))
        count = len(matrices)
        corners = np.array(
            [
                [x, y, z]
                for x in (self.domain[0][0], self.domain[1][0])
                for y in (self.domain[0][1], self.domain[1][1])
                for z in (self.domain[0][2], self.domain[1][2])
            ]
        )
        rates = [1.0] * count
        ray_terms = []
        radii = [
            float(np.linalg.norm((corners - m[:3, 3]) @ m[:3, :3], axis=1).max()) for m in matrices
        ]
        block = self._part.kinematics
        assert block is not None
        for f, spec in zip(kin._flexures, block.get("flexures", []), strict=True):
            joints = f.joints
            assert joints is not None
            if not joints:
                matrices.append(np.eye(4, dtype=np.float32))
                rates.append(1.0)
                radii.append(float(np.linalg.norm(corners, axis=1).max()))
                continue
            op, first, second = joints[0]
            # Surface bounds and shader geometry belong to the compiled design snapshot.
            design = jnp.asarray(kin.design_defaults)
            a = float(first(jnp.asarray(values), design))
            b = float(second(jnp.asarray(values), design))
            pivot = np.asarray(op.origin)
            radius = float(np.linalg.norm(corners - pivot, axis=1).max())
            params = spec["blend"]["params"]
            slope = (
                math.hypot(*params["axis"]) / (params["hi"] - params["lo"])
                if spec["blend"]["kind"] == "axis_ramp"
                else 1.5 / (params["r1"] - params["r0"])
            )
            angular_slope = abs(b - a) * float(slope)
            rates.append(1 + angular_slope * radius)
            blend_axis = np.asarray(params["axis"], dtype=float)
            blend_axis /= math.hypot(*blend_axis)
            ray_terms.append(
                _FlexureRayRate(
                    tuple(op.axis),
                    tuple(blend_axis),
                    tuple(pivot),
                    angular_slope,
                    spec["blend"]["kind"] == "radial_hermite",
                )
            )
            radii.append(float(np.linalg.norm(pivot)) + radius)
            # Match the actual inverse implementation only for a constant joint coordinate.
            matrices.append(
                np.asarray(kin.body_transforms(values))[f.from_index] if a == b else None
            )
        radius = max(radii) * (1 + 1e-5) + self.field_error
        rest_domain = ((-radius, -radius, -radius), (radius, radius, radius))
        tree_rates = [
            infer_sdf_max_rate(t, self._part, mode="values", domain=rest_domain)
            for t in self._trees
        ]
        intervals = build_param_intervals(self._part, mode="values")
        primitive_scale = max(_primitive_scale(t, intervals) for t in self._trees)
        rest_rate = max(tree_rates) * primitive_scale * 2 * (1 + 1e-5)
        rate = rest_rate * max(rates)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Surface finding requires finite positive field-rate bounds")
        groups: list[int] = []
        for i, matrix in enumerate(matrices):
            group = next(
                (
                    groups[j]
                    for j in range(i)
                    if matrix is not None
                    and matrices[j] is not None
                    and np.array_equal(matrix, matrices[j])
                ),
                i,
            )
            groups.append(group)
        return SurfacePose(
            self,
            tuple(float(x) for x in values),
            tuple(groups),
            float(rate),
            float(rest_rate),
            tuple(ray_terms),
        )


@dataclass(frozen=True)
class SurfacePose:
    """Pose-specific exclusion field, membership oracle and bounded ray search."""

    surface: MaterialSurfaces
    dofs: tuple[float, ...]
    groups: tuple[int, ...]
    max_rate: float
    rest_rate: float
    _ray_terms: tuple[_FlexureRayRate, ...]

    def field(self, points: Any) -> Any:
        """Continuous exclusion value; zero is only a candidate, never a hit test."""
        import jax.numpy as jnp

        return self.surface._field(
            jnp.asarray(points), jnp.asarray(self.dofs), jnp.asarray(self.groups)
        )

    def contains(self, points: Any) -> Any:
        import jax.numpy as jnp

        return self.surface._contains(jnp.asarray(points), jnp.asarray(self.dofs))

    def _ray_rate(self, ends: Any, direction: Any) -> float:
        """Bound inverse variation along this segment, not across the whole box."""
        import numpy as np

        stretch = 1.0
        for term in self._ray_terms:
            radius = float(np.linalg.norm(np.cross(ends - term.pivot, term.axis), axis=1).max())
            projection = float(np.dot(direction, term.blend_axis))
            variation = (
                float(np.linalg.norm(direction - projection * np.asarray(term.blend_axis)))
                if term.radial
                else abs(projection)
            )
            stretch = max(stretch, 1 + term.slope * radius * variation)
        return self.rest_rate * stretch

    def normal(self, point: Any, *, step: float = 1e-4) -> tuple[float, float, float] | None:
        """Estimate a world-space normal, including inverse-map spatial variation.

        At smooth exposed surfaces this approximates the inverse-Jacobian
        transpose action. At corners/seams it is only a shading estimate.
        """
        import numpy as np

        if not math.isfinite(step) or step <= 0:
            raise ValueError("Normal step must be finite and positive")
        point = np.asarray(point, dtype=float)
        offsets = np.eye(3) * step
        gradient = np.asarray(self.field(point + offsets) - self.field(point - offsets))
        norm = float(np.linalg.norm(gradient))
        if not math.isfinite(norm) or norm == 0:
            return None
        return tuple(float(x) for x in gradient / norm)  # type: ignore[return-value]

    def contact_witness(
        self, point: Any, radius: float
    ) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """Return inside/outside endpoints proving a boundary within radius of point.

        Probe along the estimated normal, then coordinate axes. Both endpoints
        must lie in the query domain and have field values beyond the numerical
        error allowance as well as opposite actual membership. Failure to find
        a witness does not prove absence of contact. The segment is a geometric
        proximity certificate, not proof that the ray itself intersects.
        """
        import numpy as np

        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Contact point must be a finite 3-vector")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Contact radius must be finite and positive")
        normal = self.normal(point)
        directions = list(np.eye(3))
        if normal is not None:
            directions.insert(0, np.asarray(normal))
        for direction in directions:
            ends = point + np.array([-radius, radius])[:, None] * direction
            if np.any(ends < self.surface.domain[0]) or np.any(ends > self.surface.domain[1]):
                continue
            values = np.asarray(self.field(ends))
            if not np.isfinite(values).all():
                continue
            inside = np.asarray(self.contains(ends))
            error = self.surface.field_error
            if inside[0] and not inside[1] and values[0] < -error and values[1] > error:
                return tuple(ends[0]), tuple(ends[1])
            if inside[1] and not inside[0] and values[1] < -error and values[0] > error:
                return tuple(ends[1]), tuple(ends[0])
        return None

    def trace(
        self,
        origin: Any,
        direction: Any,
        near: float,
        far: float,
        *,
        tolerance: float = 1e-3,
        max_steps: int = 1024,
        contact_tolerance: float = 0.0,
    ) -> SurfaceHit:
        """Find the first demonstrated occupancy change along a bounded ray segment.

        Direction is normalized; near/far and tolerance are world distances.
        Both segment endpoints must lie in the query domain. Search visits
        intervals in ray order. Thin features cannot become misses merely
        because endpoint samples agree: an unproved interval is subdivided or
        returned as unresolved. max_steps bounds work, not geometric accuracy.

        contact_tolerance=0 preserves crossing-only behavior. With a positive
        world-space contact tolerance, a pending unresolved window can return
        contact after an off-ray inside/outside witness is found. Crossings
        found in that window take priority. The witness does not prove an exact
        tangency; tolerances below numerical resolution can remain unresolved.
        """
        import numpy as np

        origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        if (
            origin.shape != (3,)
            or direction.shape != (3,)
            or not np.isfinite([origin, direction]).all()
        ):
            raise ValueError("Ray origin and direction must be finite 3-vectors")
        length = float(np.linalg.norm(direction))
        if length == 0 or not math.isfinite(length):
            raise ValueError("Ray direction must have finite nonzero length")
        if (
            not all(math.isfinite(x) for x in (near, far, tolerance))
            or near < 0
            or far <= near
            or tolerance <= 0
        ):
            raise ValueError("Ray requires 0 <= near < far and positive finite tolerance")
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 4096:
            raise ValueError("Ray max_steps must be an integer in [1, 4096]")
        direction = direction / length
        ends = origin + np.array([near, far])[:, None] * direction
        if np.any(ends < self.surface.domain[0]) or np.any(ends > self.surface.domain[1]):
            raise ValueError("Ray segment must lie inside the surface query domain")
        if not math.isfinite(contact_tolerance) or contact_tolerance < 0:
            raise ValueError("Contact tolerance must be finite and nonnegative")
        ray_rate = self._ray_rate(ends, direction)
        initial = bool(self.contains(ends[0]))
        stack = [(near, far)]
        pending = None
        candidate = None

        def unresolved(lo: float, hi: float) -> SurfaceHit:
            if candidate is None:
                return SurfaceHit("unresolved", (lo, hi))
            t, witness = candidate
            return SurfaceHit("contact", (lo, hi), self.normal(origin + t * direction), t, witness)

        for _ in range(max_steps):
            if not stack:
                return (
                    unresolved(pending, min(far, pending + tolerance))
                    if pending is not None
                    else SurfaceHit("miss", None)
                )
            lo, hi = stack.pop()
            mid = (lo + hi) / 2
            value = float(self.field(origin + mid * direction))
            if pending is not None and lo - pending >= tolerance:
                return unresolved(pending, min(hi, pending + tolerance))
            if pending is not None:
                hi = min(hi, pending + tolerance)
                mid = (lo + hi) / 2
                value = float(self.field(origin + mid * direction))
            allowance = ray_rate * ((hi - lo) / 2) + self.surface.field_error
            if math.isfinite(value) and (value < -allowance if initial else value > allowance):
                continue
            if hi - lo <= tolerance / (8 * max(1.0, ray_rate)):
                samples = origin + np.array([lo, mid, hi])[:, None] * direction
                if np.any(np.asarray(self.contains(samples)) != initial):
                    start = lo if pending is None else pending
                    point = origin + ((start + hi) / 2) * direction
                    return SurfaceHit("hit", (start, hi), self.normal(point))
                if pending is None:
                    pending = lo
                if contact_tolerance > 0 and candidate is None:
                    witness = self.contact_witness(samples[1], contact_tolerance)
                    if witness is not None:
                        candidate = (mid, witness)
                continue
            if len(stack) >= 63 or mid in (lo, hi):
                return unresolved(lo if pending is None else pending, hi)
            stack.extend([(mid, hi), (lo, mid)])
        if pending is not None:
            return unresolved(pending, min(far, pending + tolerance))
        return SurfaceHit("unresolved", stack[-1]) if stack else SurfaceHit("miss", None)


def compile_material_surfaces(
    part: Part, *, domain: BBox, field_error: float = 1e-5
) -> MaterialSurfaces:
    """Compile surface queries for a design snapshot and a finite ordered domain.

    Explicitly reject geometry without the analysed continuous rate contract.
    The caller chooses field_error to cover numerical evaluation error at its
    model scale. The default is a practical float32 allowance, not a universal
    floating-point error proof. Returned hit brackets retain the requested
    spatial tolerance; an unresolved result must remain visible to the host.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from software_defined_matter.kinematics import resolve_region_tree

    box = np.asarray(domain, dtype=float)
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
        raise ValueError("Surface domain must be a finite nonempty ordered 3-D box")
    if not math.isfinite(field_error) or field_error <= 0:
        raise ValueError("Surface field_error must be finite and positive")
    snapshot = copy.deepcopy(part)
    membership = compile_material_motion(snapshot)
    if membership is None:
        raise ValueError("Surface finding requires authored motion regions")
    block = snapshot.kinematics
    assert block is not None
    trees = tuple(
        [m.sdf_tree for m in snapshot.materials]
        + [
            resolve_region_tree(snapshot, r["region"])
            for r in [*block["bodies"], *block.get("flexures", [])]
        ]
    )
    for tree in trees:
        _supported(tree)

    ownership_invariant = _invariant_classifiers(snapshot, trees[len(snapshot.materials) :])

    def field(points: Any, dofs: Any, groups: Any) -> Any:
        rest = membership.rest_points(points, dofs)
        results = []
        coverage = []
        count = len(membership.region_names)
        for i in range(count):
            q = rest[i]
            material = jnp.min(jnp.stack([fn(q) for fn in membership._materials]), axis=0)
            distances = jnp.stack([fn(q) for fn in membership.kinematics._regions])
            mask = (groups == i).reshape((count,) + (1,) * (distances.ndim - 1))
            inside = jnp.min(jnp.where(mask, distances, jnp.inf), axis=0)
            outside = jnp.min(jnp.where(mask, jnp.inf, distances), axis=0)
            value = jnp.maximum(material, inside - outside)
            results.append(jnp.where(groups[i] == i, value, jnp.inf))
            if ownership_invariant:
                cover = jnp.minimum(material, outside - inside)
                coverage.append(jnp.where(groups[i] == i, cover, -jnp.inf))
        result = jnp.min(jnp.stack(results), axis=0)
        if ownership_invariant:
            cover = jnp.max(jnp.stack(coverage), axis=0)
            # Invariance proves these fields cannot have opposite strict signs.
            # Select the stronger margin without adding their rate bounds.
            result = jnp.where(
                result + cover < 0, jnp.minimum(result, cover), jnp.maximum(result, cover)
            )
        return result

    return MaterialSurfaces(
        membership,
        (tuple(box[0]), tuple(box[1])),
        field_error,
        ownership_invariant,
        snapshot,
        trees,
        jax.jit(field),
        jax.jit(membership.contains),
    )
