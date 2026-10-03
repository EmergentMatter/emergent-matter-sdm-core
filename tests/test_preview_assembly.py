"""Static previews preserve occurrence overrides and compose nested rigid transforms."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
pytest.importorskip("pyvista")

from examples.build_assembly_example import build_assembly_bundle
from examples.preview_assembly import preview_assembly
from software_defined_matter import (
    Assembly,
    Frame,
    Instance,
    MaterialRegion,
    Part,
    PartRef,
    Port,
    save,
    sdf_primitive,
)


def test_preview_uses_assembly_overrides_and_distinct_occurrence_positions(tmp_path):
    path = build_assembly_bundle(tmp_path)
    plotter = preview_assembly(path, resolution=32, off_screen=True)
    try:
        shaft = plotter.actors["rotor.shaft/shaft"].mapper.dataset
        # One grid cell absorbs contour discretisation at the cylinder's end caps.
        assert shaft.bounds[4:6] == pytest.approx((-12, 12), abs=0.8)
        for name, x in (("bolt_left", -6), ("bolt_right", 6)):
            mesh = plotter.actors[f"{name}/bolt"].mapper.dataset
            assert mesh.center == pytest.approx((x, 0, 2), abs=1e-5)
        assert "port/output/2" in plotter.actors
    finally:
        plotter.close()


def test_preview_composes_parent_rotation_with_child_translation(tmp_path):
    save(
        Part(
            "ball",
            materials=[MaterialRegion(1, "ball", sdf_primitive("sphere", r=1))],
            ports=[Port("tip", Frame(position=(0, 0, 2)))],
        ),
        tmp_path / "ball.sdm",
    )
    save(
        Assembly(
            "inner",
            instances=(Instance("ball", PartRef("ball.sdm"), transform=Frame(position=(3, 0, 0))),),
        ),
        tmp_path / "inner.sdm",
    )
    save(
        Assembly(
            "outer",
            instances=(
                Instance(
                    "inner",
                    PartRef("inner.sdm"),
                    transform=Frame.from_axis_angle((0, 0, 1), math.pi / 2, position=(10, 0, 0)),
                ),
            ),
        ),
        tmp_path / "assembly.sdm",
    )
    plotter = preview_assembly(tmp_path / "assembly.sdm", resolution=16, off_screen=True)
    try:
        mesh = plotter.actors["inner.ball/ball"].mapper.dataset
        np.testing.assert_allclose(mesh.center, (10, 3, 0), atol=1e-5)
        arrow = plotter.actors["port/inner.ball.tip/2"].mapper.dataset
        assert arrow.bounds[4] == pytest.approx(2, abs=1e-5)
    finally:
        plotter.close()


def test_preview_rejects_unspecified_placement(tmp_path):
    save(Part("empty"), tmp_path / "empty.sdm")
    save(
        Assembly("root", instances=(Instance("empty", PartRef("empty.sdm")),)),
        tmp_path / "assembly.sdm",
    )
    with pytest.raises(ValueError, match="ungrounded instances"):
        preview_assembly(tmp_path / "assembly.sdm", off_screen=True)


def test_preview_rejects_invalid_resolution():
    with pytest.raises(ValueError, match="resolution must be at least 4"):
        preview_assembly("unused.sdm", resolution=0)


def test_preview_mated_instances_use_core_world_transforms(tmp_path):
    from examples.build_placement_example import build_placement_bundle
    from software_defined_matter import compile_placement, load_bundle

    path = build_placement_bundle(tmp_path)
    state = compile_placement(load_bundle(path)).evaluate_checked()
    plotter = preview_assembly(path, resolution=20, off_screen=True)
    try:
        mesh = plotter.actors["carriage/carriage"].mapper.dataset
        np.testing.assert_allclose(mesh.center, state.instances["carriage"][:3, 3], atol=1e-5)
        assert "port/output/2" in plotter.actors
        # Local arm centre is halfway between its pivot and tip.
        arm = plotter.actors["arm_module.arm/arm"].mapper.dataset
        expected = (state.ports["arm_module.pivot"][:3, 3] + state.ports["output"][:3, 3]) / 2
        np.testing.assert_allclose(arm.center, expected, atol=1e-4)
    finally:
        plotter.close()
