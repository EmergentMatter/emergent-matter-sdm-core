"""Admit material ownership to the region-solid renderer only when equivalent.

The ordinary rigid emitter and grid consumers render one solid per body.
Material-motion regions instead classify ownership. This adapter constructs a
render-only snapshot for the subset whose owned material is a whole rigid SDF;
other documents require bounded material queries. Never save the adapted part
in place of the authored document.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from software_defined_matter.kinematics import compile_kinematics, resolve_region_tree
from software_defined_matter.model import Part
from software_defined_matter.motion_bounds import _Intervals, _padded
from software_defined_matter.sdf.bbox import BBoxInferenceError

__all__ = ["RigidMaterialRefusalError", "rigid_material_part"]


class RigidMaterialRefusalError(ValueError):
    """Owned material has no certified representation as independent body solids."""


def _key(tree: Any) -> str:
    return json.dumps(tree, sort_keys=True)


def _union_members(tree: dict[str, Any]) -> list[dict[str, Any]]:
    if tree.get("type") == "op" and tree.get("op") == "union":
        return [member for child in tree["children"] for member in _union_members(child)]
    return [tree]


def rigid_material_part(part: Part) -> Part:
    """Return an independent render snapshot whose body solids equal owned material.

    One owner owns every material point, so its region is replaced with the
    hard union of material trees. Multiple owners are admitted only when those
    trees are exactly the union of the classifiers and their conservative rest
    boxes are strictly disjoint over declared design ranges. Every material
    point then has exactly one nonpositive classifier. Arbitrary subsequent
    rigid transforms, including posed overlaps, preserve this equivalence.

    Bounds are inferred, never taken from an authored box for this proof.
    Failure to prove separation is a refusal, not evidence of non-equivalence.
    Smooth multi-material/owner unions and flexures use bounded queries.
    Recompute after design or range edits; live inputs must stay in their bounds.
    Material identities remain on the snapshot; emitted component IDs are body
    IDs, not material IDs. The input part and its ownership rules are untouched.

    Args:
        part: Authored part, validated by the motion compiler.

    Raises:
        RigidMaterialRefusalError: A bounded material-query renderer is required.
        ValueError: The authored motion is invalid.
    """
    snapshot = copy.deepcopy(part)
    motion = compile_kinematics(snapshot)
    block = snapshot.kinematics
    if motion is None or not block or not block.get("bodies"):
        raise RigidMaterialRefusalError("Rigid material rendering requires body owners")
    if block.get("flexures"):
        raise RigidMaterialRefusalError("Flexures require bounded material queries")
    if not snapshot.materials:
        raise RigidMaterialRefusalError("Rigid material rendering requires material geometry")
    bodies = block["bodies"]
    materials = [m.sdf_tree for m in snapshot.materials]
    smooth = bool(snapshot.metadata.get("smooth_csg", False))
    if smooth and (len(bodies) > 1 or len(materials) > 1):
        raise RigidMaterialRefusalError(
            "Smoothed ownership unions require bounded material queries"
        )
    if len(bodies) == 1:
        bodies[0]["region"] = (
            materials[0]
            if len(materials) == 1
            else {"type": "op", "op": "union", "children": materials}
        )
        return snapshot

    regions = [resolve_region_tree(snapshot, b["region"]) for b in bodies]
    material_keys = {_key(t) for m in materials for t in _union_members(m)}
    region_keys = {_key(t) for r in regions for t in _union_members(r)}
    if material_keys != region_keys:
        raise RigidMaterialRefusalError("Body classifiers differ from material geometry")
    try:
        intervals = _Intervals(snapshot, {})
        boxes = [_padded(intervals.rest_bbox(r, smooth_csg=False)) for r in regions]
    except BBoxInferenceError as exc:
        raise RigidMaterialRefusalError(f"Cannot certify owned-material separation: {exc}") from exc
    for i, box in enumerate(boxes):
        for other in boxes[:i]:
            if not any(box[1][a] < other[0][a] or other[1][a] < box[0][a] for a in range(3)):
                raise RigidMaterialRefusalError(
                    "Body classifiers overlap or touch within their design bounds"
                )
    for body, region in zip(bodies, regions, strict=True):
        body["region"] = region
    return snapshot
