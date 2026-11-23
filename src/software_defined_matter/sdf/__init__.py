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
from software_defined_matter.sdf.exactness import infer_sdf_exactness

__all__ = [
    "BBoxInferenceError",
    "UnboundedParamError",
    "UnsupportedExprError",
    "UnsupportedSDFNodeError",
    "infer_material_bbox",
    "infer_sdf_bbox",
    "infer_sdf_exactness",
    "make_sdf_closure",
    "make_sdf_closure_with_binding",
    "sdf_ops",
    "sdf_shapes",
    "transforms",
]
