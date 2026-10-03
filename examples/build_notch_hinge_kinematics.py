"""Build the kinematics-reference ``.sdm``: two blocks joined by a notch hinge.

Run with::

    uv run python examples/build_notch_hinge_kinematics.py

The part is a notch-hinge flexure block: two rigid blocks joined by a
compliant neck, with one revolute DOF. It is the smallest geometry that
still exercises every feature of the ``kinematics`` block:

* a ground body (empty ``motion.ops``)
* a moving body whose rigid rotation is parameterised by a DOF
* a flexure between them whose transform blends the two bodies' motions
  across a spatial field
* ``$node`` regions that reference named nodes in the SDF tree rather than
  duplicating geometry

The hinge primitive is square in XZ. A ``scale_axis`` stretch along Z turns
it into a plate, which is also what puts this document on schema 0.6.

The two notch circles are centred at ``y = ±(depth/2 - r)``, so each eats
``2r`` inward from its face and the surviving neck is ``depth - 4r`` thick,
not ``depth - 2r``. ``r >= depth/4`` severs the beam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emergent_matter_materials import get as _get_material_property

from software_defined_matter import (
    Constraint,
    Frame,
    MaterialRegion,
    Objective,
    Param,
    Part,
    Port,
    make_param_ref,
    min_schema_version_for,
    save,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.dsl.expr import expr_binop, expr_metric, expr_num, expr_param

OUT = Path(__file__).parent / "notch_hinge_kinematics.sdm"

# The hinge beam is centred at the origin with its axis along Z. It spans
# ±beam_width/2 in X (and in Z, before scale_axis), and ±beam_depth/2 in Y.
# The two blocks butt against its X faces.
BEAM_WIDTH = 8.0
BEAM_DEPTH = 8.0
NOTCH_RADIUS = 1.5  # neck = BEAM_DEPTH - 4 * NOTCH_RADIUS = 2.0 mm
HINGE_AXIS_SCALE = 2.0  # scale_axis s.z: plate half-height = 8 mm
BLOCK_HALF_LENGTH = 8.0
BLOCK_HALF_DEPTH = BEAM_DEPTH / 2.0
BLOCK_HALF_HEIGHT = BEAM_WIDTH / 2.0 * HINGE_AXIS_SCALE
BLOCK_OFFSET = BEAM_WIDTH / 2.0 + BLOCK_HALF_LENGTH

FLEX_RANGE_DEG = 10.0
MIN_NECK_MM = 0.5  # SLS printability guardrail

# Catalog rho is kg/m^3. The mass metric multiplies SDF volume in mm^3, so
# convert to g/mm^3 (1 kg/m^3 = 1e-6 g/mm^3) to keep the units consistent.
PA12_DENSITY_G_PER_MM3 = _get_material_property("nylon12_sls", "rho") / 1e6


def _named(name: str, node: dict[str, Any]) -> dict[str, Any]:
    """Stamp a ``$node`` target onto an SDF tree without copying it."""
    node["name"] = name
    return node


def _block(name: str, x_offset: float) -> dict[str, Any]:
    """A rigid block, translated along X, carrying an addressable name."""
    return _named(
        name,
        sdf_transform(
            "translate",
            sdf_primitive(
                "box",
                b=[BLOCK_HALF_LENGTH, BLOCK_HALF_DEPTH, BLOCK_HALF_HEIGHT],
            ),
            t=[x_offset, 0.0, 0.0],
        ),
    )


def build_notch_hinge() -> Part:
    """Return the notch-hinge kinematics example as a ``Part``."""
    p_beam_width = Param(
        "beam_width",
        BEAM_WIDTH,
        free=False,
        bounds=(4.0, 20.0),
        unit="mm",
        ui={"step": 0.5, "group": "hinge", "order": 1},
    )
    p_beam_depth = Param(
        "beam_depth",
        BEAM_DEPTH,
        free=False,
        bounds=(2.0, 20.0),
        unit="mm",
        ui={"step": 0.5, "group": "hinge", "order": 2},
    )
    p_notch_r = Param(
        "notch_radius",
        NOTCH_RADIUS,
        free=True,
        bounds=(0.2, 1.9),
        unit="mm",
        ui={"step": 0.05, "group": "hinge", "order": 3},
    )

    hinge = _named(
        "hinge_neck",
        sdf_transform(
            "scale_axis",
            sdf_primitive(
                "notch_hinge",
                width=make_param_ref("beam_width"),
                depth=make_param_ref("beam_depth"),
                notch_radius=make_param_ref("notch_radius"),
            ),
            s=[1.0, 1.0, HINGE_AXIS_SCALE],
        ),
    )
    sdf_tree = sdf_op(
        "union",
        [_block("block_fixed", -BLOCK_OFFSET), hinge, _block("block_moving", BLOCK_OFFSET)],
    )

    # block_fixed is ground. block_moving rotates about the hinge axis (Z,
    # through the origin) by the flex DOF. The neck blends the two across X:
    # at x = -beam_width/2 it matches block_fixed, at +beam_width/2 it
    # matches block_moving. At flex = 0 the whole thing is identity.
    kinematics = {
        "dofs": [
            {
                "name": "flex",
                "kind": "angle",
                "range": [-FLEX_RANGE_DEG, FLEX_RANGE_DEG],
                "default": 0.0,
                "rate": FLEX_RANGE_DEG / 2.0,
                "unit": "deg",
            }
        ],
        "bodies": [
            {
                "name": "block_fixed",
                "region": {"$node": "block_fixed"},
                "motion": {"ops": []},
            },
            {
                "name": "block_moving",
                "region": {"$node": "block_moving"},
                "motion": {
                    "ops": [
                        {
                            "kind": "rotate",
                            "axis": [0, 0, 1],
                            "origin": [0.0, 0.0, 0.0],
                            "angle": {"type": "dof", "name": "flex"},
                        }
                    ]
                },
            },
        ],
        "flexures": [
            {
                "name": "hinge_neck",
                "region": {"$node": "hinge_neck"},
                "from_body": "block_fixed",
                "to_body": "block_moving",
                "blend": {
                    "type": "field",
                    "kind": "axis_ramp",
                    "params": {
                        "axis": [1.0, 0.0, 0.0],
                        "lo": -BEAM_WIDTH / 2.0,
                        "hi": BEAM_WIDTH / 2.0,
                    },
                },
            }
        ],
    }

    span_x = BLOCK_OFFSET + BLOCK_HALF_LENGTH
    # Y is widened past the rest shape so the swept pose stays inside the
    # march bounds: the block tip reaches span_x, so ±FLEX_RANGE_DEG sweeps
    # a corner by span_x * sin(flex) past the rest half-depth.
    span_y = 9.0
    span_z = BLOCK_HALF_HEIGHT + 1.0

    return Part(
        name="notch_hinge_kinematics",
        params={p.name: p for p in [p_beam_width, p_beam_depth, p_notch_r]},
        materials=[MaterialRegion(1, "nylon12_sls", sdf_tree)],
        ports=[
            Port(
                "ground",
                Frame(position=(-span_x, 0.0, 0.0)),
                body="block_fixed",
            ),
            Port(
                "tip",
                Frame(position=(span_x, 0.0, 0.0)),
                body="block_moving",
            ),
        ],
        objectives=[
            Objective(
                name="mass",
                sense="minimize",
                expr=expr_metric("mass", density=PA12_DENSITY_G_PER_MM3),
                weight=1.0,
            )
        ],
        constraints=[
            Constraint(
                name="neck_thickness_printable",
                expr=expr_binop(
                    "-",
                    expr_param("beam_depth"),
                    expr_binop("*", expr_num(4.0), expr_param("notch_radius")),
                ),
                op=">=",
                rhs=MIN_NECK_MM,
            )
        ],
        metadata={
            "bbox": [
                [-span_x - 1.0, -span_y, -span_z],
                [span_x + 1.0, span_y, span_z],
            ],
            "units": "mm",
            "process": "SLS",
            "author": "Emergent Matter",
            "note": (
                "Kinematics reference: two rigid blocks joined by a notch-hinge "
                "neck, one revolute DOF (flex, deg). block_fixed is ground; "
                "block_moving rotates about Z through the origin; the neck is a "
                "flexure blending the two across X. Identity at flex=0. Y bbox "
                "is widened past the rest shape to contain the swept pose. "
                "Regenerate with examples/build_notch_hinge_kinematics.py."
            ),
        },
        kinematics=kinematics,
    )


def main() -> None:
    part = build_notch_hinge()
    save(part, OUT)
    print(f"wrote {OUT} (schema_version={min_schema_version_for(part)})")


if __name__ == "__main__":
    main()
