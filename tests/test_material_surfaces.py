"""Surface queries prove occupancy changes rather than drawing classifier zeros."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import sdf_primitive, sdf_raster_field, sdf_transform
from software_defined_matter.glsl.material_surfaces import emit_material_surfaces
from software_defined_matter.material_surfaces import compile_material_surfaces
from tests.test_material_motion import _bodies, _flexure
from tests.test_material_motion import shader_runtime as _runtime_fixture


@pytest.fixture(scope="module", name="surface_runtime")
def surface_runtime_fixture():
    yield from _runtime_fixture.__wrapped__()


DOMAIN = ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))


def _hit(result, root):
    assert result.status == "hit", result
    assert result.interval[0] - 1e-5 <= root <= result.interval[1] + 1e-5
    assert result.interval[1] - result.interval[0] <= 1e-3 + 1e-8


def test_zero_pose_removes_internal_ownership_surface_and_finds_exit():
    pose = compile_material_surfaces(_bodies(), domain=DOMAIN).prepare([0])
    assert pose.groups == (0, 0)
    assert float(pose.field([0.0, 0, 0])) == -2
    entry = pose.trace([-3, 0, 0], [1, 0, 0], 0, 6)
    _hit(entry, 1)
    np.testing.assert_allclose(entry.normal, [-1, 0, 0], atol=1e-4)
    exit = pose.trace([-1, 0, 0], [1, 0, 0], 0, 4)
    _hit(exit, 3)  # Crosses x=0 inside material without inventing a hit.
    np.testing.assert_allclose(exit.normal, [1, 0, 0], atol=1e-4)
    assert pose.trace([-3, 3, 0], [1, 0, 0], 0, 6).status == "miss"


def test_gap_and_overlap_are_actual_surface_boundaries():
    surface = compile_material_surfaces(_bodies(), domain=DOMAIN)
    _hit(surface.prepare([1]).trace([0, 0, 0], [1, 0, 0], 0, 4), 1)
    # Overlap has no internal surface at x=0; the outer boundary is x=1.
    _hit(surface.prepare([-1]).trace([0, 0, 0], [1, 0, 0], 0, 4), 1)


@pytest.mark.parametrize("width", [2e-4, 1e-7])
def test_thin_material_is_never_a_false_miss(width):
    part = _bodies()
    part.materials[0].sdf_tree = sdf_transform(
        "translate", sdf_primitive("box", b=[width, 1, 1]), t=[0.123456, 0, 0]
    )
    pose = compile_material_surfaces(part, domain=DOMAIN).prepare([0])
    result = pose.trace([-1, 0, 0], [1, 0, 0], 0, 2)
    assert result.status in {"hit", "unresolved"}
    if width > 1e-6:
        _hit(result, 1.123456 - width)


def test_budget_and_tangent_uncertainty_are_not_hits_or_misses():
    pose = compile_material_surfaces(_bodies(), domain=DOMAIN).prepare([0])
    assert pose.trace([-3, 0, 0], [1, 0, 0], 0, 6, max_steps=1).status == "unresolved"
    # A non-dyadic tangent may never be sampled exactly. It cannot be declared empty.
    result = pose.trace([-3.123, 2, 0], [1, 0, 0], 0, 6)
    assert result.status in {"hit", "unresolved"}


def _blade(radial):
    part = _flexure(radial)
    for body in part.kinematics["bodies"]:
        body["region"] = sdf_primitive("sphere", r=0.1)
    part.kinematics["flexures"][0]["region"] = sdf_primitive("sphere", r=100)
    part.materials[0].sdf_tree = sdf_transform(
        "translate", sdf_primitive("box", b=[0.6, 0.2, 1]), t=[1, 0, 0]
    )
    return part


@pytest.mark.parametrize("radial", [False, True])
def test_flexure_surface_and_inverse_jacobian_normal(radial):
    surface = compile_material_surfaces(_blade(radial), domain=DOMAIN)
    assert surface.prepare([0]).groups == (0, 0, 0)
    pose = surface.prepare([1.3])
    kin = surface.membership.kinematics
    rest = jnp.array([1.6, 0.0, 0.7])
    point = np.asarray(kin.pose_points(rest, jnp.array([1.3]), jnp.array(2)))

    def field(q):
        return surface.membership._materials[0](
            kin.inverse_flexure_points(q, jnp.array([1.3]), flexure="blade")
        )

    gradient = np.asarray(jax.grad(field)(jnp.asarray(point)))
    expected = gradient / np.linalg.norm(gradient)
    np.testing.assert_allclose(pose.normal(point), expected, atol=2e-3)
    result = pose.trace(point + 0.15 * expected, -expected, 0, 0.3)
    _hit(result, 0.15)
    np.testing.assert_allclose(result.normal, expected, atol=3e-3)
    assert surface.prepare([2 * math.pi]).groups[2] == 2  # Full turn is not a rigid zero pose.
    assert surface.prepare([3]).max_rate > pose.max_rate


def test_invalid_domain_ray_and_unproved_rate_fail_explicitly():
    with pytest.raises(ValueError, match="domain"):
        compile_material_surfaces(_bodies(), domain=((0, 0, 0), (0, 1, 1)))
    pose = compile_material_surfaces(_bodies(), domain=DOMAIN).prepare([0])
    for direction, far in [([0, 0, 0], 1), ([1, 0, 0], 100)]:
        with pytest.raises(ValueError):
            pose.trace([0, 0, 0], direction, 0, far)
    with pytest.raises(ValueError, match="finite"):
        pose.surface.prepare([float("nan")])
    part = _bodies()
    part.materials[0].sdf_tree = sdf_raster_field([0, 0, 0], 1, np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="rate contract"):
        compile_material_surfaces(part, domain=DOMAIN)


def test_executed_shader_ray_status_brackets_and_pose_updates(surface_runtime):
    emission = emit_material_surfaces(_bodies(), domain=DOMAIN)
    source = emission.scene_source
    points = np.array([[-3.0, 0, 0], [0.0, 0, 0], [-3.0, 3, 0]])
    for q in [0, 1, -1]:
        packet = emission.pose_uniforms([q])
        pose = emission.surface.prepare([q])
        gpu = surface_runtime.evaluate(
            emission, points, packet, "sdm_surface_trace(p, vec3(1,0,0), 0.0, 4.0, 0.001, 1024)"
        )
        for point, got in zip(points, gpu, strict=True):
            cpu = pose.trace(point, [1, 0, 0], 0, 4)
            assert got[2] == {"hit": 1, "miss": 0, "unresolved": -1}[cpu.status]
            if cpu.interval is not None:
                np.testing.assert_allclose(got[:2], cpu.interval, atol=1e-5)
    assert source == emission.scene_source  # Only the atomic uniform packet changes.
    got = surface_runtime.evaluate(
        emission, [[0, 0, 0]], expression="sdm_surface_trace(p, vec3(0), 0.0, 1.0, 0.001, 1024)"
    )
    assert got[0, 2] == -2


@pytest.mark.parametrize("radial", [False, True])
def test_executed_shader_flexure_field_normal_and_hit(surface_runtime, radial):
    emission = emit_material_surfaces(_blade(radial), domain=DOMAIN)
    pose = emission.surface.prepare([1.3])
    kin = emission.surface.membership.kinematics
    point = np.asarray(kin.pose_points(jnp.array([1.6, 0.0, 0.7]), jnp.array([1.3]), jnp.array(2)))
    normal = np.array(pose.normal(point))
    direction = "vec3(" + ",".join(str(float(x)) for x in -normal) + ")"
    packet = emission.pose_uniforms([1.3])
    got = surface_runtime.evaluate(
        emission,
        [point + 0.15 * normal],
        packet,
        f"sdm_surface_trace(p, {direction}, 0.0, 0.3, 0.001, 1024)",
    )
    assert got[0, 2] == 1
    assert got[0, 0] - 1e-5 <= 0.15 <= got[0, 1] + 1e-5
    got = surface_runtime.evaluate(
        emission, [point], packet, "vec4(sdm_surface_normal(p), sdm_surface_field(p))"
    )
    np.testing.assert_allclose(got[0, :3], normal, atol=3e-3)
    assert abs(got[0, 3]) < 1e-5


def test_equal_nonzero_endpoint_motion_removes_partition_seams():
    import copy

    part = _flexure()
    part.kinematics["bodies"][0]["motion"] = copy.deepcopy(part.kinematics["bodies"][1]["motion"])
    pose = compile_material_surfaces(part, domain=DOMAIN).prepare([1.3])
    assert pose.groups == (0, 0, 0)
    _hit(pose.trace([0, 0, 0], [1, 0, 0], 0, 4), 2)


def test_shader_thin_material_budget_and_degree_packet(surface_runtime):
    part = _blade(True)
    part.kinematics["dofs"][0].update(unit="deg", range=[-720, 720], default=75)
    emission = emit_material_surfaces(part, domain=DOMAIN)
    packet = emission.pose_uniforms([1.3])
    assert packet["u_dof_0"] == pytest.approx(math.degrees(1.3))
    points = [[1.5, 0.1, 0.5], [0, 0, 0]]
    got = surface_runtime.evaluate(emission, points, packet, "vec4(sdm_surface_field(p), 0, 0, 0)")
    np.testing.assert_allclose(got[:, 0], emission.surface.prepare([1.3]).field(points), atol=2e-5)
    got = surface_runtime.evaluate(
        emission, points, expression="vec4(sdm_surface_field(p), 0, 0, 0)"
    )
    np.testing.assert_allclose(
        got[:, 0], emission.surface.prepare([math.radians(75)]).field(points), atol=2e-5
    )
    part = _bodies()
    part.materials[0].sdf_tree = sdf_transform(
        "translate", sdf_primitive("box", b=[1e-7, 1, 1]), t=[0.123456, 0, 0]
    )
    emission = emit_material_surfaces(part, domain=DOMAIN)
    for budget in [1, 1024]:
        got = surface_runtime.evaluate(
            emission,
            [[-1, 0, 0]],
            expression=f"sdm_surface_trace(p, vec3(1,0,0), 0.0, 2.0, 0.001, {budget})",
        )
        assert got[0, 2] in {1, -1}


def test_snapshot_and_polygon_extrusion_rate_contract():
    from software_defined_matter import sdf_2d_to_3d

    part = _bodies()
    part.materials[0].sdf_tree = sdf_2d_to_3d(
        "extrusion", sdf_primitive("polygon_2d", vertices=[[-1, -1], [1, -1], [1, 1], [-1, 1]]), h=1
    )
    surface = compile_material_surfaces(part, domain=DOMAIN)
    part.materials[0].sdf_tree = sdf_primitive("sphere", r=4)
    _hit(surface.prepare([0]).trace([-3, 0, 0], [1, 0, 0], 0, 6), 2)


def test_scaled_plane_normals_are_not_assumed_to_have_unit_rate():
    from software_defined_matter import sdf_op

    part = _bodies()
    part.materials[0].sdf_tree = sdf_op(
        "intersect",
        [
            sdf_primitive("box", b=[2, 2, 2]),
            sdf_primitive("plane", n=[100, 0, 0], h=-12.345),
        ],
    )
    pose = compile_material_surfaces(part, domain=DOMAIN).prepare([0])
    assert pose.max_rate >= 200
    gradient = float((pose.field([0.1236, 0, 0]) - pose.field([0.1235, 0, 0])) / 0.0001)
    assert gradient == pytest.approx(100, rel=1e-3)


def test_invariant_flexure_weld_is_proved_solid(surface_runtime):
    part = _flexure(False)
    part.kinematics["bodies"][0]["region"] = sdf_transform(
        "translate", sdf_primitive("sphere", r=1), t=[0, 0, -1]
    )
    part.kinematics["bodies"][1]["region"] = sdf_transform(
        "translate", sdf_primitive("sphere", r=0.1), t=[0, 0, 100]
    )
    part.kinematics["flexures"][0]["region"] = sdf_transform(
        "translate", sdf_primitive("sphere", r=1), t=[0, 0, 1]
    )
    emission = emit_material_surfaces(part, domain=DOMAIN)
    pose = emission.surface.prepare([1.3])
    # Classifiers and material are rotationally invariant: this entire segment
    # stays solid, although its inverse maps differ on either side of the weld.
    points = np.array([[0.3, 0, z] for z in np.linspace(-0.5, 0.5, 11)])
    assert np.asarray(pose.contains(points)).all()
    result = pose.trace([0.3, 0, -0.5], [0, 0, 1], 0, 1)
    assert emission.surface.ownership_invariant
    assert result.status == "miss"
    assert result.interval is None
    got = surface_runtime.evaluate(
        emission,
        [[0.3, 0, -0.5]],
        emission.pose_uniforms([1.3]),
        "sdm_surface_trace(p, vec3(0,0,1), 0.0, 1.0, 0.001, 1024)",
    )
    assert got[0, 2] == 0
