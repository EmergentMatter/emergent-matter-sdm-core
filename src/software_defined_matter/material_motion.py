"""Exact point membership in posed material, independent of rendering distance.

Each owner pulls the query back through its own inverse, then checks ownership
and material membership in rest space. An ownership region is a classifier,
not a solid. Overlapping posed owners may both contain the same query.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from software_defined_matter.kinematics import KinematicsEval, _points, compile_kinematics
from software_defined_matter.model import Part

__all__ = ["MaterialMotionEval", "compile_material_motion"]


@dataclass(frozen=True)
class MaterialMotionEval:
    """Snapshot point queries for materials under body and supported flexure motion.

    Region order is bodies then flexures. Materials follow document order and
    retain their IDs, including repeated IDs on distinct material records.
    Numerical methods accept ``free_vec`` in the kinematics binding order.
    Recompile after structural edits. DOFs use radians/mm; use
    ``kinematics.to_evaluator_units`` to convert authored-unit inputs.
    The document's smooth_csg metadata applies to materials and classifiers.
    """

    region_names: tuple[str, ...]
    material_ids: tuple[int, ...]
    kinematics: KinematicsEval
    _materials: tuple[Callable, ...]

    def rest_points(self, points: Any, dofs: Any, *, free_vec: Any = None) -> Any:
        """Return one candidate rest point per owner: (N, ..., 3).

        These are inverse candidates, not claims that the query belongs to
        those owners. Use membership to check the corresponding rest ownership.
        """
        import jax.numpy as jnp

        points = _points(points)
        values = self.kinematics._dofs(dofs)
        bodies = self.kinematics.body_transforms(values, free_vec=free_vec)
        rest = [
            jnp.einsum("ij,...j->...i", matrix[:3, :3].T, points - matrix[:3, 3])
            for matrix in bodies
        ]
        rest.extend(
            self.kinematics.inverse_flexure_points(points, values, flexure=name, free_vec=free_vec)
            for name in self.region_names[len(bodies) :]
        )
        return jnp.stack(rest)

    def membership(self, points: Any, dofs: Any, *, free_vec: Any = None) -> Any:
        """Return Boolean (N, M, ...) membership for points shaped (..., 3).

        A material includes its zero surface. Equal classifier distances choose
        the first owner. Different posed owners and different material records
        may overlap; all memberships are retained. Nonfinite query candidates,
        NaN classifier values, and NaN material values never count as occupied.
        This discontinuous Boolean query is not an SDF or a ray-step bound.
        """
        import jax.numpy as jnp

        design = self.kinematics._design(free_vec)
        rest = self.rest_points(points, dofs, free_vec=design)
        distances = jnp.stack([fn(rest, design) for fn in self.kinematics._regions], axis=-1)
        owners = jnp.argmin(distances, axis=-1)
        shape = (len(self.region_names),) + (1,) * (rest.ndim - 2)
        own = owners == jnp.arange(len(self.region_names)).reshape(shape)
        own &= jnp.all(jnp.isfinite(rest), axis=-1) & ~jnp.any(jnp.isnan(distances), axis=-1)
        return jnp.stack([own & (fn(rest, design) <= 0) for fn in self._materials], axis=1)

    def contains(self, points: Any, dofs: Any, *, free_vec: Any = None) -> Any:
        """Return whether each posed query belongs to any owner/material pair."""
        import jax.numpy as jnp

        return jnp.any(self.membership(points, dofs, free_vec=free_vec), axis=(0, 1))


def compile_material_motion(part: Part, *, binding: Any = None) -> MaterialMotionEval | None:
    """Compile posed-material membership, or return None without motion regions.

    An optional occurrence-scoped ``binding`` shares the root design-vector layout
    with geometry and kinematics. Omitted vectors use its initial values.

    Reject documents with unsupported flexure inverses or no material records.
    This provides occupancy only. It neither emits a rendered surface nor
    establishes a global distance field, continuity, or invertibility.
    """
    from software_defined_matter.dsl.resolve import make_binding
    from software_defined_matter.sdf.compile import make_sdf_closure_with_binding

    snapshot = copy.deepcopy(part)
    binding = make_binding(snapshot) if binding is None else copy.deepcopy(binding)
    motion = compile_kinematics(snapshot, binding=binding)
    if motion is None or not motion.region_names:
        return None
    assert snapshot.kinematics is not None
    flexures = snapshot.kinematics.get("flexures", [])
    missing = [f["name"] for f in flexures if f["name"] not in motion.inverse_flexure_names]
    if missing:
        raise ValueError(f"Material membership requires supported flexure inverses: {missing}")
    if not snapshot.materials:
        raise ValueError("Material membership requires at least one material record")
    return MaterialMotionEval(
        motion.region_names,
        tuple(m.material_id for m in snapshot.materials),
        motion,
        tuple(
            make_sdf_closure_with_binding(
                m.sdf_tree, binding, b_smooth_csg=bool(snapshot.metadata.get("smooth_csg", False))
            )
            for m in snapshot.materials
        ),
    )
