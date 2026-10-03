"""Build and save the canonical example ``.sdm`` file.

Run with::

    uv run python examples/build_example.py
"""

from __future__ import annotations

from pathlib import Path

from emergent_matter_materials import get as _get_material_property

from software_defined_matter import (
    Constraint,
    Frame,
    MaterialRegion,
    Objective,
    Param,
    Part,
    Port,
    field_primitive,
    make_param_ref,
    save,
    sdf_deform,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.dsl.expr import (
    expr_binop,
    expr_metric,
    expr_param,
)


def build_example() -> Part:
    # ------------------------------------------------------------------
    # 1. Parameters
    # ------------------------------------------------------------------
    # fmt: off
    p_outer_r = Param("outer_radius", 20.0, free=True, bounds=(10.0, 40.0), unit="mm")
    p_inner_r = Param("inner_radius", 14.0, free=True, bounds=(6.0, 30.0), unit="mm")
    p_height  = Param("height",       30.0, free=False, unit="mm")
    p_notch_r = Param("notch_radius",  3.0, free=True, bounds=(1.0, 6.0), unit="mm")
    # fmt: on

    # ------------------------------------------------------------------
    # 2. Envelope SDF tree (uses $ref so free params flow into the SDF)
    # ------------------------------------------------------------------
    outer_cyl = sdf_primitive(
        "capped_cylinder", h=make_param_ref("height"), r=make_param_ref("outer_radius")
    )
    inner_cyl = sdf_primitive(
        "capped_cylinder", h=make_param_ref("height"), r=make_param_ref("inner_radius")
    )
    shell = sdf_op("subtract", [outer_cyl, inner_cyl])

    # n_periods sized to cover the cylinder envelope (outer_radius up to 40
    # in bounds mode, half-height 30). 0.5 * n_periods * period is the
    # gyroid's half-extent per axis: here ±40 in xy and ±32 in z.
    gyroid_infill = sdf_primitive("gyroid", period=8.0, min_thickness=1.0, n_periods=[10, 10, 8])
    infilled = sdf_op("intersect", [shell, gyroid_infill])

    hinge = sdf_primitive(
        "notch_hinge", width=8.0, depth=8.0, notch_radius=make_param_ref("notch_radius")
    )
    hinge_placed = sdf_transform("translate", hinge, t=[0.0, 0.0, 30.0])

    # Flange disk connecting the on-axis hinge to the outer shell wall.
    top_disk = sdf_primitive("capped_cylinder", h=1.0, r=make_param_ref("outer_radius"))
    top_disk_placed = sdf_transform("translate", top_disk, t=[0.0, 0.0, 29.8])
    radial_wave = field_primitive("radial", freq=0.25, amplitude=0.35, phase=0.0)
    top_disk_textured = sdf_deform("displace", top_disk_placed, field=radial_wave)
    hinge_assembly = sdf_op("union", [hinge_placed, top_disk_textured])

    # ------------------------------------------------------------------
    # 3. Materials - each one has its own SDF describing *where* it lives.
    #    Physical properties live in the (future) Matter Library, not here.
    # ------------------------------------------------------------------
    copper = MaterialRegion(
        material_id=1,
        name="Cu",
        # The metal hinge at the top of the part.
        sdf_tree=hinge_assembly,
    )
    pla = MaterialRegion(
        material_id=2,
        name="PLA",
        # Gyroid infill confined to the shell walls.
        sdf_tree=infilled,
    )

    # ------------------------------------------------------------------
    # 4. Coupling nodes (local coordinates)
    # ------------------------------------------------------------------
    bottom_port = Port(
        name="bottom_flange",
        frame=Frame(position=(0.0, 0.0, -30.0), orientation=(0.0, 1.0, 0.0, 0.0)),
        metadata={"joint_type": "fixed"},
    )
    top_hinge = Port(
        name="hinge_axis",
        frame=Frame(position=(0.0, 0.0, 30.0)),
        metadata={"joint_type": "revolute", "axis": [0, 0, 1]},
    )

    # ------------------------------------------------------------------
    # 5. Objectives and constraints (symbolic expression trees).
    #    Densities etc. are supplied at use-site from the Matter Library
    #    (stubbed for now) so that the .sdm file stays pure geometry.
    # ------------------------------------------------------------------
    CU_DENSITY_KG_M3 = _get_material_property("Cu", "rho")

    minimise_mass = Objective(
        name="min_mass",
        sense="minimize",
        expr=expr_metric("mass", density=CU_DENSITY_KG_M3),
        weight=1.0,
    )

    wall_thickness = Constraint(
        name="wall_thickness",
        expr=expr_binop("-", expr_param("outer_radius"), expr_param("inner_radius")),
        op=">=",
        rhs=3.0,
    )

    max_volume = Constraint(
        name="max_volume",
        expr=expr_metric("volume"),
        op="<=",
        rhs=40000.0,
    )

    # ------------------------------------------------------------------
    # 6. Assemble Part
    # ------------------------------------------------------------------
    return Part(
        name="hollow_cylinder_with_hinge",
        params={p.name: p for p in [p_outer_r, p_inner_r, p_height, p_notch_r]},
        materials=[copper, pla],
        ports=[bottom_port, top_hinge],
        objectives=[minimise_mass],
        constraints=[wall_thickness, max_volume],
        metadata={
            "version": "0.1",
            "process": "FDM",
            "author": "Emergent Matter",
            "units": "mm",
            # Sampling domain is inferred from the SDF tree alone: the
            # gyroid contributes its `n_periods` clip, the `displace`
            # deform inflates by the ripple amplitude, and the CSG ops
            # propagate. No explicit `bbox` metadata required.
            # The 1 mm gyroid wall needs cells of 0.25 mm or finer before the
            # volume metrics will integrate it.
            "metric_voxel_size": 0.24,
            "metric_grid_cap": 288,
        },
    )


if __name__ == "__main__":
    part = build_example()
    out = Path(__file__).parent / "hollow_cylinder_with_hinge.sdm"
    save(part, out)
    print(part)
    print("\nFree params:", list(part.free_params()))
    print("Param vector:", part.param_vector())
    print(f"\nSaved to {out}")
