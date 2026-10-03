"""Weld coverage and witnessed grazing contacts retain a shared CPU/GLSL contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from software_defined_matter import sdf_primitive, sdf_transform
from software_defined_matter.glsl import emit_material_surfaces
from software_defined_matter.material_surfaces import compile_material_surfaces
from tests.test_material_motion import _bodies, _flexure
from tests.test_material_motion import shader_runtime as _runtime_fixture

DOMAIN = ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))


@pytest.fixture(scope="module", name="runtime")
def runtime_fixture():
    yield from _runtime_fixture.__wrapped__()


def _weld(radial=False):
    part = _flexure(radial)
    for body, z, radius in zip(part.kinematics["bodies"], [-1, 100], [1, 0.1], strict=True):
        body["region"] = sdf_transform("translate", sdf_primitive("sphere", r=radius), t=[0, 0, z])
    part.kinematics["flexures"][0]["region"] = sdf_transform(
        "translate", sdf_primitive("sphere", r=1), t=[0, 0, 1]
    )
    # Material need not share the classifiers' rotational symmetry.
    part.materials[0].sdf_tree = sdf_primitive("box", b=[1, 0.7, 2])
    return part


@pytest.mark.parametrize("radial", [False, True])
@pytest.mark.parametrize("angle", [1.3, 2 * math.pi])
def test_ray_crosses_weld_and_finds_external_exit(runtime, radial, angle):
    emission = emit_material_surfaces(_weld(radial), domain=DOMAIN)
    assert emission.surface.ownership_invariant
    pose = emission.surface.prepare([angle])
    assert pose.groups[2] == 2  # Actual varying inverse, even with a complete turn.
    assert float(pose.field([0.3, 0, 0])) < -0.1
    cpu = pose.trace([0.3, 0, -0.5], [0, 0, 1], 0, 3)
    assert cpu.status == "hit", cpu
    assert cpu.interval[0] <= 2.5 <= cpu.interval[1]
    gpu = runtime.evaluate(
        emission,
        [[0.3, 0, -0.5]],
        emission.pose_uniforms([angle]),
        "sdm_surface_trace(p, vec3(0,0,1), 0.0, 3.0, 0.001, 1024)",
    )[0]
    assert gpu[2] == 1
    np.testing.assert_allclose(gpu[:2], cpu.interval, atol=2e-5)
    # A shorter segment entirely inside material has no crossing or contact.
    short = pose.trace([0.3, 0, -0.5], [0, 0, 1], 0, 1, contact_tolerance=0.001)
    assert short.status == "miss"
    assert pose.contact_witness([0.3, 0, 0], 0.001) is None


@pytest.mark.parametrize("radial", [False, True])
def test_coverage_field_signs_match_membership_for_anisotropic_material(runtime, radial):
    emission = emit_material_surfaces(_weld(radial), domain=DOMAIN)
    points = np.random.default_rng(44).uniform([-1.5, -1.5, -1], [1.5, 1.5, 1], (200, 3))
    for angle in [0.2, 1.3, 6.2]:
        pose = emission.surface.prepare([angle])
        field = np.asarray(pose.field(points))
        mask = np.abs(field) > 1e-4
        np.testing.assert_array_equal(field[mask] < 0, np.asarray(pose.contains(points))[mask])
        gpu = runtime.evaluate(
            emission, points, emission.pose_uniforms([angle]), "vec4(sdm_surface_field(p), 0, 0, 0)"
        )[:, 0]
        np.testing.assert_allclose(gpu, field, atol=5e-5)


def test_symmetry_proof_rejects_almost_centred_and_noncoaxial_classifiers():
    part = _weld()
    assert compile_material_surfaces(part, domain=DOMAIN).ownership_invariant
    part.kinematics["flexures"][0]["region"]["params"]["t"][0] = 1e-13
    assert not compile_material_surfaces(part, domain=DOMAIN).ownership_invariant
    part = _weld()
    part.kinematics["bodies"].append(
        {
            "name": "other-axis",
            "region": sdf_primitive("sphere", r=0.1),
            "motion": {
                "ops": [
                    {"kind": "rotate", "axis": [1, 0, 0], "angle": {"type": "dof", "name": "angle"}}
                ]
            },
        }
    )
    assert not compile_material_surfaces(part, domain=DOMAIN).ownership_invariant


def test_shared_tilted_axis_and_offset_pivot_prove_coverage():
    part = _weld()
    axis, pivot = np.array([1, 2, 3]), np.array([2, -1, 3])
    part.kinematics["bodies"][1]["motion"]["ops"][0].update(
        axis=axis.tolist(), origin=pivot.tolist()
    )
    for body, z in zip(part.kinematics["bodies"], [-1, 100], strict=True):
        body["region"]["params"]["t"] = (pivot + z * axis).tolist()
    flex = part.kinematics["flexures"][0]
    flex["region"]["params"]["t"] = (pivot + axis).tolist()
    flex["blend"]["params"].update(axis=axis.tolist(), lo=-10, hi=20)
    part.materials[0].sdf_tree = sdf_transform(
        "translate", sdf_primitive("sphere", r=2), t=pivot.tolist()
    )
    surface = compile_material_surfaces(part, domain=((-20.0,) * 3, (20.0,) * 3))
    assert surface.ownership_invariant
    pose = surface.prepare([1.3])
    normal = axis / np.linalg.norm(axis)
    start = pivot - 0.2 * normal
    assert pose.trace(start, normal, 0, 0.4, max_steps=4096).status == "miss"


def _verify_witness(pose, result, origin, direction, radius):
    assert result.status == "contact", result
    assert result.interval[0] <= result.contact_t <= result.interval[1]
    point = np.asarray(origin) + result.contact_t * np.asarray(direction)
    ends = np.asarray(result.contact_segment)
    assert np.max(np.linalg.norm(ends - point, axis=1)) <= radius * (1 + 1e-6)
    np.testing.assert_array_equal(pose.contains(ends), [True, False])
    values = np.asarray(pose.field(ends))
    assert values[0] < -pose.surface.field_error
    assert values[1] > pose.surface.field_error


def test_non_dyadic_tangent_has_a_witness_without_claiming_a_crossing(runtime):
    emission = emit_material_surfaces(_bodies(), domain=DOMAIN)
    pose = emission.surface.prepare([0])
    origin, direction, radius = [-3.123, 2, 0], [1, 0, 0], 0.001
    strict = pose.trace(origin, direction, 0, 6)
    assert strict.status == "unresolved"
    contact = pose.trace(origin, direction, 0, 6, contact_tolerance=radius)
    _verify_witness(pose, contact, origin, direction, radius)
    gpu = runtime.evaluate(
        emission,
        [origin],
        expression="sdm_surface_trace_contacts(p, vec3(1,0,0), 0.0, 6.0, 0.001, 1024, 0.001)",
    )[0]
    assert gpu[2] == 2
    np.testing.assert_allclose(gpu[:2], contact.interval, atol=2e-5)
    assert gpu[3] == pytest.approx(contact.contact_t, abs=2e-5)
    point = np.asarray(origin) + gpu[3] * np.asarray(direction)
    witness = runtime.evaluate(
        emission, [point], expression="sdm_surface_contact_witness(p, 0.001)"
    )[0]
    assert witness[3] == 1
    endpoints = point + np.array([-0.001, 0.001])[:, None] * witness[:3]
    np.testing.assert_array_equal(pose.contains(endpoints), [True, False])


def test_contacts_preserve_crossings_misses_and_budget_failure(runtime):
    emission = emit_material_surfaces(_bodies(), domain=DOMAIN)
    pose = emission.surface.prepare([0])
    crossing = pose.trace([-3.123, 0, 0], [1, 0, 0], 0, 6, contact_tolerance=0.001)
    assert crossing.status == "hit"
    assert crossing.contact_segment is None
    miss = pose.trace([-3.123, 2.002, 0], [1, 0, 0], 0, 6, contact_tolerance=0.001)
    assert miss.status == "miss"
    tiny = pose.trace([-3.123, 2, 0], [1, 0, 0], 0, 6, contact_tolerance=1e-8)
    assert tiny.status == "unresolved"
    exhausted = pose.trace([-3.123, 2, 0], [1, 0, 0], 0, 6, max_steps=1, contact_tolerance=0.001)
    assert exhausted.status == "unresolved"
    got = runtime.evaluate(
        emission,
        [[-3.123, 2.002, 0]],
        expression="sdm_surface_trace_contacts(p, vec3(1,0,0), 0.0, 6.0, 0.001, 1024, 0.001)",
    )
    assert got[0, 2] == 0
    with pytest.raises(ValueError, match="Contact tolerance"):
        pose.trace([0, 0, 0], [1, 0, 0], 0, 1, contact_tolerance=-1)
    got = runtime.evaluate(
        emission,
        [[0, 0, 0]],
        expression="sdm_surface_trace_contacts(p, vec3(1,0,0), 0.0, 1.0, 0.001, 1024, -1.0)",
    )
    assert got[0, 2] == -2


def test_radial_shell_welds_remain_solid_through_a_complete_turn(runtime):
    from software_defined_matter import sdf_modifier

    part = _flexure(True)
    part.materials[0].sdf_tree = sdf_primitive("sphere", r=3)
    part.kinematics["bodies"][0]["region"] = sdf_primitive("sphere", r=0.1)
    part.kinematics["bodies"][1]["region"] = sdf_modifier(
        "onion", sdf_primitive("sphere", r=2), thickness=0.1
    )
    part.kinematics["flexures"][0]["region"] = sdf_modifier(
        "onion", sdf_primitive("sphere", r=1), thickness=0.1
    )
    part.kinematics["flexures"][0]["blend"]["params"].update(r0=0.5, r1=1.5)
    emission = emit_material_surfaces(part, domain=DOMAIN)
    pose = emission.surface.prepare([2 * math.pi])
    assert emission.surface.ownership_invariant
    result = pose.trace([0, 0, 0], [1, 0, 0], 0, 4, max_steps=4096)
    assert result.status == "hit"
    assert result.interval[0] <= 3 <= result.interval[1]
    got = runtime.evaluate(
        emission,
        [[0, 0, 0]],
        emission.pose_uniforms([2 * math.pi]),
        "sdm_surface_trace(p, vec3(1,0,0), 0.0, 4.0, 0.001, 4096)",
    )
    assert got[0, 2] == 1
    np.testing.assert_allclose(got[0, :2], result.interval, atol=2e-5)


def test_ray_bound_covers_inverse_field_variation_and_uses_segment_radius():
    emission = emit_material_surfaces(_weld(False), domain=DOMAIN)
    pose = emission.surface.prepare([2 * math.pi])
    ends = np.array([[0.3, 0, -0.5], [0.3, 0, 2.5]])
    direction = np.array([0.0, 0, 1])
    rate = pose._ray_rate(ends, direction)
    assert rate < pose.max_rate / 5
    t = np.linspace(0, 3, 1001)
    points = ends[0] + t[:, None] * direction
    observed = np.max(np.abs(np.diff(np.asarray(pose.field(points)))) / np.diff(t))
    assert observed <= rate


@pytest.mark.parametrize(
    "primitive,params", [("capped_cylinder", {"r": 0.2, "h": 1}), ("torus", {"t": [1, 0.1]})]
)
def test_round_classifiers_use_the_core_z_axis_convention(primitive, params):
    for axis, expected in [([0, 0, 1], True), ([0, 1, 0], False)]:
        part = _flexure(False)
        part.kinematics["bodies"][1]["motion"]["ops"][0]["axis"] = axis
        for body in part.kinematics["bodies"]:
            body["region"] = sdf_primitive("sphere", r=0.1)
        flex = part.kinematics["flexures"][0]
        flex["blend"]["params"]["axis"] = axis
        flex["region"] = sdf_primitive(primitive, **params)
        assert compile_material_surfaces(part, domain=DOMAIN).ownership_invariant is expected


@pytest.mark.parametrize("radial", [False, True])
def test_shader_ray_rates_match_degree_pose_packets(runtime, radial):
    part = _weld(radial)
    part.kinematics["dofs"][0].update(unit="deg", range=[-720, 720])
    part.kinematics["flexures"][0]["blend"]["params"]["axis"] = [0, 0, 4]
    emission = emit_material_surfaces(part, domain=DOMAIN)
    starts = np.random.default_rng(7).uniform(-2, 2, (12, 3))
    offset = np.array([1.0, 2, 0.5])
    direction = offset / np.linalg.norm(offset)
    for angle in [0.2, 2 * math.pi]:
        pose = emission.surface.prepare([angle])
        expected = [pose._ray_rate(np.array([p, p + offset]), direction) for p in starts]
        got = runtime.evaluate(
            emission,
            starts,
            emission.pose_uniforms([angle]),
            "vec4(sdm_surface_ray_rate(p, p+vec3(1,2,0.5), normalize(vec3(1,2,0.5))), 0, 0, 0)",
        )[:, 0]
        np.testing.assert_allclose(got, expected, rtol=3e-6)
        assert np.max(got) <= pose.max_rate
