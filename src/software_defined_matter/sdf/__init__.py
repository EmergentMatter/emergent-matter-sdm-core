"""JAX SDF primitives, ops, transforms, and the DSL compiler."""

from software_defined_matter.sdf import sdf_ops, sdf_shapes, transforms
from software_defined_matter.sdf.bbox import (
    BBoxInferenceError,
    UnboundedParamError,
    UnsupportedExprError,
    UnsupportedSDFNodeError,
    infer_material_bbox,
    infer_sdf_bbox,
)
from software_defined_matter.sdf.compile import (
    make_sdf_closure,
    make_sdf_closure_with_binding,
)

__all__ = [
    "BBoxInferenceError",
    "UnboundedParamError",
    "UnsupportedExprError",
    "UnsupportedSDFNodeError",
    "infer_material_bbox",
    "infer_sdf_bbox",
    "make_sdf_closure",
    "make_sdf_closure_with_binding",
    "sdf_ops",
    "sdf_shapes",
    "transforms",
]
