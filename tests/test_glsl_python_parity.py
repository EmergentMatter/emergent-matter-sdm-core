"""Keep the GLSL library and the JAX primitives from drifting apart.

There are two independent implementations of every primitive: the JAX one in
``sdf/sdf_shapes.py`` used for metrics, meshing and export, and the GLSL one in
``glsl/lib.glsl`` used for shader preview. Nothing forces an edit to one to
reach the other, and a divergence is invisible: the preview simply shows a
different shape from the thing that gets manufactured.

``test_lib_glsl_contains_every_dispatch_entry`` already checks that a GLSL
function *exists* per primitive. These tests check that it takes the same
parameters, in the same order, and carries the same magic numbers.

Most checks are structural or compare NumPy transcriptions. Sweep checks also
execute emitted GLSL through the headless EGL/GLES runtime required in CI.
The shared sweep consumer tests execute both inline and texture-backed frames,
including live profile parameters and explicitly bounded near-tie candidates.

Where a wrong body would be silent and expensive, the body is transcribed into
numpy here and compared against the JAX implementation, with source needles
pinning the transcription to ``lib.glsl`` so it cannot go stale unnoticed. That
covers ``sdf_annular_sector`` and the symmetry transforms at the end of the
file.
"""

from __future__ import annotations

import inspect
import os
import re

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    field_primitive,
    make_param_ref,
    sdf_deform,
    sdf_primitive,
    sdf_sweep,
    sdf_transform,
)
from software_defined_matter.glsl.emit import (
    _FIELD_DEFAULTS,
    _FIELD_SPECS,
    _PRIM_SPECS,
    _sweep_frame_arrays,
    _sweep_structure,
    emit_glsl,
    load_lib_glsl,
)
from software_defined_matter.sdf import sdf_shapes, transforms
from software_defined_matter.sdf.compile import _PRIMITIVES, make_sdf_closure
from tests._glsl_runtime import ShaderRuntime

# Primitives whose GLSL parameter names legitimately differ from the Python
# ones, with the reason. Anything not listed here must match exactly.
NAME_ALIASES = {
    # `length` is a GLSL builtin, so it cannot be a parameter name.
    ("leaf_spring", "length"): "length_",
}

# Primitives the emitter handles specially rather than through a plain
# positional call, so PrimSpec/signature comparison does not apply.
SPECIAL_CASED = {
    # GLSL has no variable-length arrays: vertices are padded to a
    # compile-time SDM_POLY_MAX_N and the real count passed alongside.
    "polygon_2d",
    # Samples leave the shader as a host-owned table. lib.glsl takes
    # (base, n, lo, inv_h); JAX takes (origin, spacing, values); the
    # emitter bridges them. PrimSpec stays empty for that reason.
    "raster_field",
}


def _strip_comments(src: str) -> str:
    return re.sub(r"//[^\n]*", "", src)


def _glsl_params(lib: str, fn_name: str) -> list[str]:
    """Parameter names of ``float <fn_name>(...)`` in declaration order,
    excluding the leading query point."""
    m = re.search(
        rf"^float\s+{re.escape(fn_name)}\s*\(([^)]*)\)", _strip_comments(lib), re.MULTILINE
    )
    assert m, f"{fn_name} not found in lib.glsl"
    args = [a.strip() for a in m.group(1).split(",") if a.strip()]
    # Last token is the name; drop any `[SIZE]` array suffix. Do NOT strip
    # digits: `r1` and `r2` are distinct parameters.
    names = [re.sub(r"\[.*\]$", "", a.split()[-1]) for a in args]
    return names[1:]  # drop `p`


def _expected(kind: str) -> list[str]:
    """Python parameter names, with documented GLSL aliases applied."""
    py = list(inspect.signature(_PRIMITIVES[kind]).parameters)[1:]
    return [NAME_ALIASES.get((kind, n), n) for n in py]


@pytest.mark.parametrize("kind", sorted(set(_PRIM_SPECS) - SPECIAL_CASED))
def test_glsl_signature_matches_python(kind):
    """lib.glsl's parameter names and order must match the JAX function's."""
    glsl = _glsl_params(load_lib_glsl(), f"sdf_{kind}")
    assert glsl == _expected(kind), (
        f"sdf_{kind} parameter mismatch\n  python: {_expected(kind)}\n  glsl  : {glsl}"
    )


@pytest.mark.parametrize("kind", sorted(set(_PRIM_SPECS) - SPECIAL_CASED))
def test_emitter_spec_matches_python(kind):
    """The emitter's PrimSpec drives argument ORDER at the call site, so a
    mismatch silently passes arguments to the wrong parameters.

    Compared against the Python names rather than the GLSL ones: PrimSpec keys
    are looked up in the ``.sdm`` params dict, which uses the Python spelling.
    """
    py = list(inspect.signature(_PRIMITIVES[kind]).parameters)[1:]
    spec = [name for name, _ in _PRIM_SPECS[kind].args]
    assert spec == py, f"_PRIM_SPECS[{kind!r}] mismatch\n  python: {py}\n  spec  : {spec}"


def test_special_cased_primitives_still_exist():
    """Guard the exemption list: if one of these gains a normal signature the
    exemption should be removed rather than silently skipping coverage."""
    lib = load_lib_glsl()
    for kind in SPECIAL_CASED:
        assert f"sdf_{kind}" in lib, f"sdf_{kind} vanished from lib.glsl"
        assert kind in _PRIMITIVES, f"{kind} vanished from the JAX registry"


@pytest.mark.parametrize("kind", sorted(_FIELD_SPECS))
def test_field_emitter_spec_and_defaults_match_python(kind):
    """Authored lookup names and fallback defaults must match the JAX callable."""
    signature = inspect.signature(getattr(sdf_shapes, f"field_{kind}"))
    py_params = list(signature.parameters.values())[1:]
    assert [name for name, _ in _FIELD_SPECS[kind]] == [param.name for param in py_params]
    assert _FIELD_DEFAULTS[kind] == {
        param.name: param.default
        for param in py_params
        if param.default is not inspect.Parameter.empty
    }


def test_tpms_normalisation_constants_match():
    """The per-family max |grad f| is a magic number duplicated in both
    implementations. Derived in docs/sdf_distance_audit/tpms_constants.py."""
    lib = load_lib_glsl()
    for family, expected in sdf_shapes._TPMS_GRAD_MAX.items():
        m = re.search(
            rf"float\s+sdf_{family}\s*\([^)]*\)\s*\{{.*?_tpms_sheet\("
            rf"\s*p\s*,\s*f\s*,\s*([0-9.eE+-]+)",
            lib,
            re.DOTALL,
        )
        assert m, f"sdf_{family} does not call _tpms_sheet with a literal constant"
        assert float(m.group(1)) == pytest.approx(expected, rel=1e-12), (
            f"{family}: lib.glsl uses {m.group(1)}, python uses {expected}"
        )


@pytest.mark.parametrize(
    "fn_name,needle",
    [
        # The slope bound that turns each offset back into a distance.
        ("sdf_bellows", "(outer_r - inner_r) * 3.141592653589793 / period"),
        ("sdf_serpentine", "amplitude * 6.283185307179586 / wavelength"),
    ],
)
def test_slope_normalisation_present_in_glsl(fn_name, needle):
    """bellows/serpentine measure an offset that is not perpendicular to the
    surface; without the divisor the GLSL value over-reports and sphere tracing
    steps straight through the geometry."""
    lib = _strip_comments(load_lib_glsl())
    body = re.search(rf"float\s+{fn_name}\s*\([^)]*\)\s*\{{(.*?)\n\}}", lib, re.DOTALL)
    assert body, f"{fn_name} not found"
    assert needle in body.group(1), f"{fn_name} is missing its slope bound: {needle}"
    assert "length(vec2(1.0, max_slope))" in body.group(1)


def test_ellipsoid_uses_the_conservative_bound():
    """IQ's k0*(k0-1)/k1 over-reports near the surface when eccentric, which
    makes sphere tracing overshoot, and is 0/0 at the centre."""
    lib = _strip_comments(load_lib_glsl())
    body = re.search(r"float\s+sdf_ellipsoid\s*\([^)]*\)\s*\{(.*?)\n\}", lib, re.DOTALL)
    assert body, "sdf_ellipsoid not found"
    assert "min(r.x, min(r.y, r.z))" in body.group(1)
    assert "k0 * (k0 - 1.0) / k1" not in body.group(1)


def test_annular_sector_uses_plane_distances():
    """abs(theta) - half_angle compares an angle against lengths; theta changes
    at rate 1/r, so it over-reports by 1/inner_r near the bore."""
    lib = _strip_comments(load_lib_glsl())
    body = re.search(r"float\s+sdf_annular_sector\s*\([^)]*\)\s*\{(.*?)\n\}", lib, re.DOTALL)
    assert body, "sdf_annular_sector not found"
    assert "abs(theta) - half_angle" not in body.group(1)
    assert "d_plane_pos" in body.group(1) and "d_plane_neg" in body.group(1)
    # The numeric test below transcribes this body. If a sign or the pi/2
    # branch threshold moves here, that transcription is stale.
    assert "-s_a * p.x + c_a * p.y" in body.group(1)
    assert "-s_a * p.x - c_a * p.y" in body.group(1)
    assert "half_angle <= 1.5707963267948966" in body.group(1)


def _glsl_annular_sector(p, inner_r, outer_r, half_angle, height):
    """Line-for-line transcription of ``sdf_annular_sector`` in lib.glsl.

    CI has no GL context, so this is how a sign error in the GLSL planes
    would still be caught: the source needles above pin the transcription
    to the file, and the values are compared to the JAX primitive.
    """
    s_a = np.sin(half_angle)
    c_a = np.cos(half_angle)
    d_plane_pos = -s_a * p[..., 0] + c_a * p[..., 1]
    d_plane_neg = -s_a * p[..., 0] - c_a * p[..., 1]
    d_angular = np.where(
        half_angle <= 1.5707963267948966,
        np.maximum(d_plane_pos, d_plane_neg),
        np.minimum(d_plane_pos, d_plane_neg),
    )
    r = np.linalg.norm(p[..., :2], axis=-1)
    d_radial = np.maximum(inner_r - r, r - outer_r)
    d_axial = np.abs(p[..., 2]) - height / 2.0
    return np.maximum(np.maximum(d_radial, d_angular), d_axial)


@pytest.mark.parametrize(
    "kwargs,points",
    [
        # Small bore: the old |theta|-half_angle form over-reported by 1/inner_r.
        (
            {"inner_r": 0.2, "outer_r": 3.0, "half_angle": 0.5, "height": 1.0},
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],  # near the axis, outside the bore
                [0.2, 0.0, 0.0],  # inner wall along +X
                [3.0, 0.0, 0.0],  # outer wall along +X
                [1.5, 0.0, 0.0],  # mid-annulus, inside the wedge
                [1.5 * np.cos(0.5), 1.5 * np.sin(0.5), 0.0],  # +half_angle plane
                [1.5 * np.cos(0.5), -1.5 * np.sin(0.5), 0.0],  # -half_angle plane
                [0.0, 1.5, 0.0],  # outside the wedge
                [1.5, 0.0, 0.5],  # top cap
            ],
        ),
        # Reflex sector: the wedge wraps past pi/2 (union of half-spaces).
        (
            {"inner_r": 1.0, "outer_r": 3.0, "half_angle": 2.5, "height": 2.0},
            [
                [2.0, 0.0, 0.0],  # +X, inside a reflex wedge
                [-2.0, 0.0, 0.0],  # -X, outside (half_angle=2.5 < pi)
                [0.0, 2.0, 0.0],  # +Y, inside
                [0.0, -2.0, 0.0],  # -Y, inside
                [2.0 * np.cos(2.5), 2.0 * np.sin(2.5), 0.0],  # +half_angle plane
                [1.0, 0.0, 0.0],  # inner wall
                [3.0, 0.0, 0.0],  # outer wall
                [2.0, 0.0, 1.0],  # top cap
            ],
        ),
    ],
)
def test_annular_sector_glsl_transcription_matches_python(kwargs, points):
    p = np.asarray(points, dtype=np.float64)
    glsl = _glsl_annular_sector(p, **kwargs)
    py = np.asarray(sdf_shapes.annular_sector(jnp.asarray(p), **kwargs), dtype=np.float64)
    # JAX is float32; the transcription is float64. GLSL would be float32 too.
    np.testing.assert_allclose(glsl, py, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Symmetry transforms: canonical_sector_fold and mirror
# ---------------------------------------------------------------------------
# A wrong fold or reflection is the same silent failure as a wrong primitive,
# and worse to spot: the preview still shows a plausible symmetric part, just
# not the one being manufactured. The bodies are four lines each, and every
# line is a place to get an argument order or a sign wrong (`atan(p.y, p.x)`
# takes y first, the centered wedge shifts by half a sector in both
# directions). So they are transcribed and compared, as annular_sector is.

# lib.glsl spells 2*pi out as a literal. It is the same double as 2*np.pi, and
# the assertion below holds it to that.
_GLSL_TAU = 6.28318530717958647692


def _glsl_body(fn_name: str) -> str:
    """The body of ``fn_name`` in lib.glsl, comments stripped."""
    lib = _strip_comments(load_lib_glsl())
    pattern = rf"^\w+\s+{re.escape(fn_name)}\s*\([^)]*\)\s*\{{(.*?)\n\}}"
    m = re.search(pattern, lib, re.DOTALL | re.MULTILINE)
    assert m, f"{fn_name} not found in lib.glsl"
    return m.group(1)


def _field_part(kind, field_params, params):
    tree = sdf_deform(
        "displace",
        sdf_primitive("sphere", r=2.0),
        field=field_primitive(kind, **field_params),
    )
    return Part(
        name=f"{kind}-parity",
        params=params,
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
        metadata={"bbox": [[-4, -4, -4], [4, 4, 4]]},
    ), tree


def test_emitted_radial_field_with_refs_matches_jax():
    """The emitted call order and radial GLSL body produce the JAX field."""
    params = {
        "frequency": Param("frequency", 0.35, free=True, bounds=(0.1, 1.0), unit="ratio"),
        "amplitude": Param("amplitude", 0.4, free=True, bounds=(0.0, 1.0), unit="mm"),
        "phase": Param("phase", 0.2, free=True, bounds=(-1.0, 1.0), unit="rad"),
    }
    part, tree = _field_part(
        "radial",
        {
            "freq": make_param_ref("frequency"),
            "amplitude": make_param_ref("amplitude"),
            "phase": make_param_ref("phase"),
        },
        params,
    )
    emitted = emit_glsl(part).scene_source
    assert "field_radial(p, u_p_frequency, u_p_amplitude, u_p_phase)" in emitted
    body = _glsl_body("field_radial")
    assert "length(p.xy)" in body
    assert "amp * sin(6.283185307179586 * freq * r + phase)" in body

    points = _ring_points(n_theta=73, radii=(0.0, 0.2, 1.1, 2.8), z=(-0.4, 0.7))
    radius = np.linalg.norm(points[..., :2], axis=-1)
    field = 0.4 * np.sin(2.0 * np.pi * 0.35 * radius + 0.2)
    glsl = np.linalg.norm(points, axis=-1) - 2.0 + field
    jax = make_sdf_closure(tree, part)(jnp.asarray(points))
    np.testing.assert_allclose(glsl, np.asarray(jax), rtol=1e-5, atol=1e-5)


def test_emitted_sin_xyz_field_with_refs_matches_jax():
    """Vector refs retain axis order and amplitude reaches the emitted field."""
    params = {
        "fx": Param("fx", 0.2, free=True, bounds=(0.1, 1.0), unit="ratio"),
        "fy": Param("fy", 0.3, free=True, bounds=(0.1, 1.0), unit="ratio"),
        "fz": Param("fz", 0.4, free=True, bounds=(0.1, 1.0), unit="ratio"),
        "amplitude": Param("amplitude", 0.25, free=True, bounds=(0.0, 1.0), unit="mm"),
        "phase": Param("phase", 0.15, free=True, bounds=(-1.0, 1.0), unit="rad"),
    }
    part, tree = _field_part(
        "sin_xyz",
        {
            "freq": [make_param_ref("fx"), make_param_ref("fy"), make_param_ref("fz")],
            "amplitude": make_param_ref("amplitude"),
            "phase": [make_param_ref("phase"), 0.4, -0.2],
        },
        params,
    )
    emitted = emit_glsl(part).scene_source
    assert (
        "field_sin_xyz(p, vec3(u_p_fx, u_p_fy, u_p_fz), u_p_amplitude, "
        "vec3(u_p_phase, 0.4, -0.2))" in emitted
    )
    body = _glsl_body("field_sin_xyz")
    assert "6.283185307179586 * freq * p + phase" in body
    assert "amp * s.x * s.y * s.z" in body

    points = np.asarray([[-1.2, 0.3, 0.7], [0.0, 0.0, 0.0], [0.8, -0.5, 1.4], [2.1, 1.3, -0.9]])
    arg = 2.0 * np.pi * np.asarray([0.2, 0.3, 0.4]) * points + [0.15, 0.4, -0.2]
    field = 0.25 * np.prod(np.sin(arg), axis=-1)
    glsl = np.linalg.norm(points, axis=-1) - 2.0 + field
    jax = make_sdf_closure(tree, part)(jnp.asarray(points))
    np.testing.assert_allclose(glsl, np.asarray(jax), rtol=1e-5, atol=1e-5)


def test_emitted_angular_field_with_refs_matches_jax():
    """Angular emission matches JAX around the seam and on the guarded axis."""
    params = {
        "lobes": Param("lobes", 6.0, free=True, bounds=(1.0, 12.0), unit="count"),
        "amplitude": Param("amplitude", 0.3, free=True, bounds=(0.0, 1.0), unit="mm"),
        "phase": Param("phase", 0.2, free=True, bounds=(-1.0, 1.0), unit="rad"),
    }
    part, tree = _field_part(
        "angular",
        {
            "freq": make_param_ref("lobes"),
            "amplitude": make_param_ref("amplitude"),
            "phase": make_param_ref("phase"),
        },
        params,
    )
    emitted = emit_glsl(part).scene_source
    assert "field_angular(p, u_p_lobes, u_p_amplitude, u_p_phase)" in emitted
    body = _glsl_body("field_angular")
    assert "dot(p.xy, p.xy) > 0.0 ? atan(p.y, p.x) : 0.0" in body
    assert "amp * sin(freq * theta + phase)" in body

    points = _ring_points(
        n_theta=181,
        radii=(0.0, 0.1, 1.0, 2.5),
        z=(-0.3, 0.8),
        skew=0.0,
    )
    radius2 = np.sum(points[..., :2] ** 2, axis=-1)
    theta = np.where(radius2 > 0.0, np.arctan2(points[..., 1], points[..., 0]), 0.0)
    field = 0.3 * np.sin(6.0 * theta + 0.2)
    glsl = np.linalg.norm(points, axis=-1) - 2.0 + field
    jax = make_sdf_closure(tree, part)(jnp.asarray(points))
    np.testing.assert_allclose(glsl, np.asarray(jax), rtol=1e-5, atol=1e-5)


def _glsl_canonical_sector_fold(p, n_sectors):
    """Transcription of ``tf_canonical_sector_fold``.

    GLSL ``mod(x, y)`` is ``x - y*floor(x/y)``, which is what ``np.mod`` and
    ``jnp.mod`` compute, so a negative azimuth folds the same way on all three.
    """
    sector = _GLSL_TAU / n_sectors
    theta = np.mod(np.arctan2(p[..., 1], p[..., 0]), sector)
    r = np.linalg.norm(p[..., :2], axis=-1)
    return np.stack([r * np.cos(theta), r * np.sin(theta), p[..., 2]], axis=-1)


def _glsl_canonical_sector_fold_c(p, n_sectors, phase_frac):
    """Transcription of ``tf_canonical_sector_fold_c``."""
    sector = _GLSL_TAU / n_sectors
    theta = (
        np.mod(np.arctan2(p[..., 1], p[..., 0]) - phase_frac * sector + 0.5 * sector, sector)
        - 0.5 * sector
    )
    r = np.linalg.norm(p[..., :2], axis=-1)
    return np.stack([r * np.cos(theta), r * np.sin(theta), p[..., 2]], axis=-1)


def _glsl_reflect_plane(p, n, o):
    """Transcription of ``tf_reflect_plane``."""
    len_n = np.linalg.norm(n)
    u = n / (len_n if len_n > 0.0 else 1.0)
    return p - 2.0 * np.sum((p - o) * u, axis=-1, keepdims=True) * u


def _glsl_rotate_z(p, angle):
    """Transcription of ``tf_rotate_z``, which the emitted fold body calls."""
    c, s = np.cos(angle), np.sin(angle)
    return np.stack(
        [c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]], axis=-1
    )


def test_the_transcribed_glsl_bodies_are_still_what_lib_glsl_says():
    """Pin each transcription above to the source it was copied from.

    Without this a later edit to lib.glsl leaves the numpy copy behind, and the
    comparisons below go on passing against a body nobody ships.
    """
    assert 2.0 * np.pi == _GLSL_TAU

    fold = _glsl_body("tf_canonical_sector_fold")
    assert "6.28318530717958647692 / n_sectors" in fold
    assert "mod(atan(p.y, p.x), sector)" in fold
    assert "length(p.xy)" in fold
    assert "vec3(r * cos(theta), r * sin(theta), p.z)" in fold

    centered = _glsl_body("tf_canonical_sector_fold_c")
    assert "atan(p.y, p.x) - phase_frac * sector + 0.5 * sector" in centered
    assert "sector) - 0.5 * sector" in centered

    for name in ("tf_reflect_plane", "tf_reflect_plane2"):
        reflect = _glsl_body(name)
        # The zero-normal guard, without which a zero n gives NaN rather than
        # the identity that transforms.reflect_plane degenerates to.
        assert "len > 0.0 ? len : 1.0" in reflect
        assert "p - 2.0 * dot(p - o, u) * u" in reflect

    rotate_z = _glsl_body("tf_rotate_z")
    assert "vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z)" in rotate_z

    finish = _glsl_body("sdm_sweep_finish")
    assert "float d_ax = (over > 0.0) ? over : -1.0e9;" in finish
    assert "vec2 w = vec2(d2d, d_ax);" in finish
    assert "length(max(w, vec2(0.0))) + min(max(w.x, w.y), 0.0)" in finish

    tab = _glsl_body("sdm_sweep_tab")
    assert "n = clamp(n, 0, (SDM_SWEEP_LEN - hdr - 1) / 4);" in tab
    assert "vec4 al = sdm_sweep_fetch(hdr + 1 + 4 * i);" in tab
    assert "vec3 t = sdm_sweep_fetch(hdr + 2 + 4 * i).xyz;" in tab
    assert "vec3 perp = p - (al.xyz + clamp(s, 0.0, al.w) * t);" in tab
    assert (
        "if (d2 < best) { best = d2; istar = i; s_star = s; l_star = al.w; perp_star = perp; }"
        in tab
    )
    assert "vec3 nn = sdm_sweep_fetch(hdr + 3 + 4 * istar).xyz;" in tab
    assert "vec3 bb = sdm_sweep_fetch(hdr + 4 + 4 * istar).xyz;" in tab
    assert "bool closed = head.y > 0.5;" in tab
    assert (
        "vec2 uv = sdm_sweep_joint(vec2(dot(perp_star, nn), dot(perp_star, bb)), over, end_cap);"
        in tab
    )
    assert "return vec3(uv, end_cap ? over : 0.0);" in tab

    joint = _glsl_body("sdm_sweep_joint")
    assert "if (over <= 0.0 || end_cap) return uv;" in joint
    assert "float m = sqrt(n * n + over * over);" in joint
    assert "return (n > 1e-9) ? uv * (m / n) : vec2(m, 0.0);" in joint


def _ring_points(n_theta=180, radii=(0.0, 0.4, 1.3, 2.0, 2.9), z=(-0.7, 0.0, 1.1), skew=0.013):
    """Points on rings, dense in angle, including the Z axis itself.

    A fold can only be wrong in angle, and an axis-aligned grid can miss every
    wedge boundary, so rings beat a cube here. r=0 is included because the
    azimuth is undefined there and atan2(0, 0) is a real disagreement risk.

    ``skew`` rotates the whole grid off the wedge boundaries. Landing exactly on
    one is ambiguous rather than wrong: the fold picks a representative, and
    float32 and float64 can pick neighbouring wedges, which differ by a whole
    sector. Callers that compare a folded point pass the default. Callers that
    compare a distance can pass ``skew=0`` -- see the node test below.
    """
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False) + skew
    pts = [(r * np.cos(t), r * np.sin(t), zz) for zz in z for r in radii for t in theta]
    return np.asarray(pts, dtype=np.float64)


@pytest.mark.parametrize("n_sectors", [3, 6, 20])
def test_canonical_sector_fold_glsl_transcription_matches_python(n_sectors):
    p = _ring_points()
    glsl = _glsl_canonical_sector_fold(p, float(n_sectors))
    py = np.asarray(
        transforms.tf_canonical_sector_fold(jnp.asarray(p), n_sectors), dtype=np.float64
    )
    np.testing.assert_allclose(glsl, py, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("phase_frac", [0.0, 0.25, 0.5, -0.25])
def test_centered_canonical_sector_fold_glsl_transcription_matches_python(phase_frac):
    p = _ring_points()
    glsl = _glsl_canonical_sector_fold_c(p, 6.0, phase_frac)
    py = np.asarray(
        transforms.tf_canonical_sector_fold(
            jnp.asarray(p), 6, centered=True, phase_frac=phase_frac
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(glsl, py, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "n,o",
    [
        ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 2.5, 0.0]),  # plane off the origin
        ([3.0, -4.0, 0.0], [0.5, 0.5, 0.5]),  # non-unit normal, normalised inside
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),  # zero normal degenerates to identity
    ],
    ids=["yz-plane", "offset-plane", "non-unit-normal", "zero-normal"],
)
def test_reflect_plane_glsl_transcription_matches_python(n, o):
    p = _ring_points()
    glsl = _glsl_reflect_plane(p, np.asarray(n), np.asarray(o))
    py = np.asarray(transforms.reflect_plane(jnp.asarray(p), n, o), dtype=np.float64)
    np.testing.assert_allclose(glsl, py, rtol=1e-5, atol=1e-5)


def _part(tree) -> Part:
    return Part(
        name="parity",
        params={},
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
        metadata={"bbox": [[-9, -9, -9], [9, 9, 9]]},
    )


def _compiled(tree, pts):
    closure = make_sdf_closure(tree, _part(tree))
    return np.asarray(closure(jnp.asarray(pts), jnp.zeros((0,))), dtype=np.float64)


# The child both node tests below use, and its transcription. Off-axis on
# purpose: a child centred on Z or on the mirror plane would hide an argument
# swapped between `p` and the transformed point.
_BLADE_T = np.array([2.0, 0.0, 0.0])
_BLADE_R = 0.4
_BLADE_TREE = sdf_transform("translate", sdf_primitive("sphere", r=_BLADE_R), t=list(_BLADE_T))


def _glsl_blade(p):
    """``sdf_sphere(tf_translate3(p, vec3(2.0, 0.0, 0.0)), 0.4)``."""
    return np.linalg.norm(p - _BLADE_T, axis=-1) - _BLADE_R


def test_emitted_fold_body_matches_the_compiled_fold():
    """The three evaluations live in emit.py, not lib.glsl, so the fold as a
    whole is only right if the emitted composition is right too.

    A flipped sign on one `sec` still produces a symmetric-looking preview, and
    still reports a distance that is too large near a seam, which is the error
    that lets a sphere tracer step through the surface.
    """
    n_sectors = 6.0
    tree = sdf_transform("canonical_sector_fold", _BLADE_TREE, n_sectors=n_sectors)

    src = emit_glsl(_part(tree)).scene_source
    assert "float n_sec = floor(6.0);" in src
    assert "vec3 q = tf_canonical_sector_fold(p, n_sec);" in src
    assert "float sec = 6.28318530717958647692 / n_sec;" in src
    assert "tf_rotate_z(q, sec)" in src
    assert "tf_rotate_z(q, -sec)" in src

    # skew=0 puts samples exactly on the wedge boundaries. The folded point
    # there is ambiguous, but the distance is not: taking the min over the
    # folded wedge and its two neighbours is continuous across a seam, which is
    # the property the three evaluations exist to provide.
    p = _ring_points(skew=0.0)
    n_sec = np.floor(n_sectors)
    q = _glsl_canonical_sector_fold(p, n_sec)
    sec = _GLSL_TAU / n_sec
    glsl = _glsl_blade(q)
    glsl = np.minimum(glsl, _glsl_blade(_glsl_rotate_z(q, sec)))
    glsl = np.minimum(glsl, _glsl_blade(_glsl_rotate_z(q, -sec)))

    np.testing.assert_allclose(glsl, _compiled(tree, p), rtol=1e-5, atol=1e-5)


def test_emitted_mirror_body_matches_the_compiled_mirror():
    """`op_union` is `min`, so the emitted mirror is the child at `p` unioned
    with the child at the reflected point, the same pair compile.py takes."""
    n, o = [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    tree = sdf_transform("mirror", _BLADE_TREE, n=n, o=o)

    src = emit_glsl(_part(tree)).scene_source
    assert "vec3 q = tf_reflect_plane(p, vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));" in src
    assert "op_union(" in src

    p = _ring_points()
    glsl = np.minimum(
        _glsl_blade(p), _glsl_blade(_glsl_reflect_plane(p, np.asarray(n), np.asarray(o)))
    )

    np.testing.assert_allclose(glsl, _compiled(tree, p), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Sweep: nearest-segment search, baked frames, flat caps
# ---------------------------------------------------------------------------
# The emitted body is where a sweep can go wrong, not lib.glsl: the frames are
# baked data, the segment pick is a loop the emitter writes, and the cap is a
# call. Each is transcribed and compared to the JAX kernel here, and the GPU
# runs the emitted text itself where a software GL context exists.

#: A query whose two nearest segments tie to within float32 arithmetic is
#: dropped from the cloud before comparing. Beyond a vertex, with both feet
#: clamped to it, the two segments are equidistant in exact arithmetic and
#: the kernel's own answer is whichever rounding puts first; the shader can
#: round the other way, and on a bend the two segments' frames differ by the
#: bend angle, so a box profile reads a different (u, v). That is an
#: distance discontinuity that prevents strict cross-backend parity there.
#: Shared consumer tests retain an explicit near-tie probe and require each
#: backend's result to match an admissible segment candidate. On a fine loop
#: those vertex wedges are most of the far field, so the filter
#: drops a large share of a uniform cloud there. An exact float64 tie is
#: dropped too: it is common on a sampled path (the float32 frame values
#: make ``A + L * T`` land on the next start point exactly), and there the
#: float32 sides still disagree by rounding. The phantom-tube probes are
#: appended AFTER the filter: on the axis-aligned bent path every operand is
#: exactly representable, both sides compute bit-identical squared distances
#: for the tied pair, and both take the first index.
_TIE_MARGIN = 1e-5


def _glsl_sweep_finish(d2d, over):
    """Transcription of ``sdm_sweep_finish``."""
    d_ax = np.where(over > 0.0, over, -1.0e9)
    w = np.stack([d2d, d_ax], axis=-1)
    return np.linalg.norm(np.maximum(w, 0.0), axis=-1) + np.minimum(
        np.maximum(w[..., 0], w[..., 1]), 0.0
    )


def _segment_search(p, A, T, L):
    """The loop the emitter writes: first-minimum segment by true 3-D distance.

    Returns ``(istar, s_star, perp_star, sorted_d2)``; the last is every
    segment's squared distance, ascending, for the tie filter.
    """
    s = np.sum((p[:, None, :] - A[None, :, :]) * T[None, :, :], axis=-1)
    foot = A[None, :, :] + np.clip(s, 0.0, L[None, :])[..., None] * T[None, :, :]
    perp = p[:, None, :] - foot
    d2 = np.sum(perp * perp, axis=-1)
    istar = np.argmin(d2, axis=-1)
    rows = np.arange(p.shape[0])
    return istar, s[rows, istar], perp[rows, istar], np.sort(d2, axis=-1)


def _glsl_sweep_joint(u, v, over, flat):
    """Transcription of ``sdm_sweep_joint``."""
    n = np.hypot(u, v)
    m = np.sqrt(n * n + over * over)
    scale = np.where(n > 1e-9, m / np.maximum(n, 1e-9), 1.0)
    ru = np.where(n > 1e-9, u * scale, m)
    rv = np.where(n > 1e-9, v * scale, 0.0)
    use = (over > 0.0) & ~flat
    return np.where(use, ru, u), np.where(use, rv, v)


def _glsl_sweep_inline(p, frames, profile, closed):
    """Transcription of the inline body ``_emit_sweep`` writes."""
    A, T, L, N, B = frames
    istar, s_star, perp_star, _ = _segment_search(p, A, T, L)
    u = np.sum(perp_star * N[istar], axis=-1)
    v = np.sum(perp_star * B[istar], axis=-1)
    over_lo = -s_star
    over_hi = s_star - L[istar]
    over = np.maximum(over_lo, over_hi)
    if closed:
        flat = np.zeros(over.shape, dtype=bool)
    else:
        flat = ((istar == 0) & (over_lo > 0.0)) | ((istar == len(L) - 1) & (over_hi > 0.0))
    u, v = _glsl_sweep_joint(u, v, over, flat)
    return _glsl_sweep_finish(profile(u, v), np.where(flat, over, 0.0))


def _glsl_sweep_tab(p, table, hdr, profile):
    """Transcription of ``sdm_sweep_tab`` reading ``table`` (RGBA texels flat),
    followed by the tabled body: profile at (u, v), then the cap."""
    tex = np.asarray(table, dtype=np.float64).reshape(-1, 4)
    n = int(tex[hdr, 0])
    closed = bool(tex[hdr, 1] > 0.5)
    al = tex[hdr + 1 : hdr + 1 + 4 * n : 4]
    A, L = al[:, :3], al[:, 3]
    T = tex[hdr + 2 : hdr + 2 + 4 * n : 4, :3]
    N = tex[hdr + 3 : hdr + 3 + 4 * n : 4, :3]
    B = tex[hdr + 4 : hdr + 4 + 4 * n : 4, :3]
    return _glsl_sweep_inline(p, (A, T, L, N, B), profile, closed)


def _drop_ties(p, frames):
    _, _, _, d2 = _segment_search(p, frames[0], frames[1], frames[2])
    if d2.shape[1] < 2:
        return p
    return p[(d2[:, 1] - d2[:, 0]) > _TIE_MARGIN * (1.0 + d2[:, 0])]


def _circle(r):
    return lambda u, v: np.hypot(u, v) - r


def _box(bx, by):
    def f(u, v):
        d = np.stack([np.abs(u) - bx, np.abs(v) - by], axis=-1)
        return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(
            np.maximum(d[..., 0], d[..., 1]), 0.0
        )

    return f


def _helix_polyline(r=2.0, pitch=0.6, turns=2.5, n=61):
    t = np.linspace(0.0, 2.0 * np.pi * turns, n)
    return [[float(r * np.cos(a)), float(r * np.sin(a)), float(pitch * a / (2 * np.pi))] for a in t]


# An open polyline with a genuine 3-D bend: +x, then +y, then +z, then -x.
_BENT_PATH = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 2.0], [0.0, 2.0, 2.0]]

# Queries the kernel's cap exists for. Past the vertex at (2, 0, 0) along the
# first segment's tangent, and past the one at (2, 2, 0) along the second's:
# a min over per-segment profile values reports "inside" there, the phantom
# tube. Then past each open end, where the cap has to be flat.
_BENT_PROBES = [
    [2.5, 0.0, 0.0],
    [3.0, 0.1, 0.0],
    [2.0, 2.6, 0.0],
    [2.0, 3.0, 0.1],
    [-0.5, 0.0, 0.0],
    [-1.0, 0.1, 0.1],
    [-0.5, 2.0, 2.0],
    [-1.0, 2.1, 2.0],
]

_SWEEP_CASES = [
    pytest.param(
        sdf_sweep(sdf_primitive("circle_2d", r=0.4), _BENT_PATH, path_kind="polyline"),
        _circle(0.4),
        (-1.0, 3.0),
        _BENT_PROBES,
        id="open-polyline-circle",
    ),
    pytest.param(
        # A box can see u/v orientation and the seed, which a circle cannot:
        # the default seed for a +x first tangent is the world y axis, so a z
        # seed rotates the whole profile a quarter turn.
        sdf_sweep(
            sdf_primitive("box_2d", b=[0.5, 0.2]),
            _BENT_PATH,
            path_kind="polyline",
            normal0=[0.0, 0.0, 1.0],
        ),
        _box(0.5, 0.2),
        (-1.0, 3.0),
        _BENT_PROBES,
        id="open-polyline-box-normal0",
    ),
    pytest.param(
        # bspline is a periodic loop: no open ends, and the emit-time sampler
        # has to be the compiler's or every sampled point disagrees.
        sdf_sweep(
            sdf_primitive("box_2d", b=[0.3, 0.15]),
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.5],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, -0.5],
                [1.5, -1.5, 0.3],
            ],
            path_kind="bspline",
        ),
        _box(0.3, 0.15),
        (-3.0, 3.0),
        [],
        id="closed-bspline-box",
    ),
    pytest.param(
        sdf_sweep(
            sdf_primitive("box_2d", b=[0.3, 0.15]),
            _helix_polyline(),
            path_kind="polyline",
            frame="cylindrical",
        ),
        _box(0.3, 0.15),
        (-3.0, 3.0),
        [],
        id="cylindrical-helix-box",
    ),
]


def _sweep_frames_of(tree):
    params = tree["params"]
    kind, closed, frame = _sweep_structure(params)
    return _sweep_frame_arrays(params["path"], kind, closed, frame, params.get("normal0"))


def _sweep_points(tree, lo_hi, probes, n=2500, seed=11):
    """A uniform cloud with its near-ties dropped, then the probes verbatim."""
    lo, hi = lo_hi
    cloud = np.random.default_rng(seed).uniform(lo, hi, size=(n, 3))
    cloud = _drop_ties(cloud, _sweep_frames_of(tree))
    return np.vstack([cloud, np.asarray(probes, dtype=np.float64).reshape(-1, 3)])


@pytest.mark.parametrize("tree,profile,lo_hi,probes", _SWEEP_CASES)
def test_the_transcribed_sweep_body_matches_the_compiled_sweep(tree, profile, lo_hi, probes):
    """The emitted composition, run in numpy against ``ops.sweep``.

    Pins the transcription to the emitted text first, so an edit to the loop
    the emitter writes cannot leave this copy passing against a body nobody
    ships. Then compares the field, including the phantom-tube probes.
    """
    src = emit_glsl(_part(tree)).scene_source
    assert "int istar = 0;" in src
    assert "float s = dot(p - A[i], T[i]);" in src
    assert "vec3 perp = p - (A[i] + clamp(s, 0.0, L[i]) * T[i]);" in src
    assert "if (d2 < best) { best = d2; istar = i; s_star = s; perp_star = perp; }" in src
    assert "vec2(dot(perp_star, NN[istar]), dot(perp_star, BB[istar])), over, end_cap);" in src
    assert "return sdm_sweep_finish(d2d, end_cap ? over : 0.0);" in src

    pts = _sweep_points(tree, lo_hi, probes)
    assert len(pts) >= 1000, "the tie filter must leave a cloud worth comparing"
    closed = _sweep_structure(tree["params"])[1]
    glsl = _glsl_sweep_inline(pts, _sweep_frames_of(tree), profile, closed)
    np.testing.assert_allclose(glsl, _compiled(tree, pts), rtol=1e-5, atol=1e-5)


def test_the_phantom_tube_probes_are_outside_and_exactly_tied():
    """The probes are only worth having if the kernel says 'outside' there,
    and only safe to compare if the tie they sit on is exact.

    A min over per-segment profile values reports the first two as inside the
    tube (their perpendicular distance to the first segment's LINE is under
    the radius). Each probe beyond a vertex is equidistant from the two
    segments meeting there, in exact arithmetic and in float32 alike, so
    both compilers take the first index rather than a rounding pick.
    """
    tree = _SWEEP_CASES[0].values[0]
    probes = np.asarray(_BENT_PROBES, dtype=np.float64)
    d = _compiled(tree, probes)
    assert np.all(d > 0.0), d
    A, T, L, _, _ = _sweep_frames_of(tree)
    _, _, _, d2 = _segment_search(probes, A, T, L)
    # The probes on an interior segment's own extension. Those past the open
    # ends have one nearest segment and nothing to tie with.
    beyond_a_vertex = [0, 2]
    assert np.all(d2[beyond_a_vertex, 1] == d2[beyond_a_vertex, 0])
    assert np.all(d2[[4, 6], 1] > d2[[4, 6], 0])


@pytest.fixture(scope="module")
def shader_runtime():
    try:
        runtime = ShaderRuntime()
    except (OSError, AttributeError, RuntimeError) as exc:
        if os.environ.get("SDM_REQUIRE_GLSL") == "1":
            pytest.fail(f"Required software shader runtime unavailable: {exc}")
        pytest.skip(
            f"Install system EGL/GLES and Mesa software drivers to run shader parity: {exc}"
        )
    yield runtime
    runtime.close()


@pytest.mark.parametrize("tree,profile,lo_hi,probes", _SWEEP_CASES)
def test_emitted_sweep_matches_the_compiled_sweep(shader_runtime, tree, profile, lo_hi, probes):
    """The emitted GLSL itself, compiled and run, against ``ops.sweep``.

    This is the test the transcription above stands in for where there is no
    GL context. CI has one and requires it (``SDM_REQUIRE_GLSL``).
    """
    emission = emit_glsl(_part(tree))
    assert emission.sweep_table == [], "the inline form is what this test drives"
    pts = _sweep_points(tree, lo_hi, probes)
    glsl = shader_runtime.evaluate(emission, pts.astype(np.float32))[:, 3]
    np.testing.assert_allclose(glsl, _compiled(tree, pts), rtol=1e-5, atol=1e-5)


def test_the_tabled_sweep_is_the_field_the_compiled_sweep_evaluates(shader_runtime):
    """Field parity for the storage change, including executed GLSL.

    A dense bspline crosses the table budget on its own. The table is pushed
    through the transcription of ``sdm_sweep_tab`` and compared with the
    kernel and the actual emitted shader; a transposed texel, an off-by-one
    in the header stride, or a frame axis in the wrong slot separates them here.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    ctrl = [
        [float(3.0 * np.cos(a)), float(3.0 * np.sin(a)), float(0.6 * np.sin(3.0 * a))]
        for a in angles
    ]
    tree = sdf_sweep(sdf_primitive("box_2d", b=[0.3, 0.15]), ctrl, path_kind="bspline")
    emission = emit_glsl(_part(tree))
    assert emission.sweep_table, "the fixture must cross the table budget"
    assert "vec3 uvo = sdm_sweep_tab(p, 0);" in emission.scene_source
    assert "return sdm_sweep_finish(d2d, uvo.z);" in emission.scene_source

    pts = _sweep_points(tree, (-4.0, 4.0), [])
    glsl = _glsl_sweep_tab(pts, emission.sweep_table, 0, _box(0.3, 0.15))
    np.testing.assert_allclose(glsl, _compiled(tree, pts), rtol=1e-5, atol=1e-5)
    executed = shader_runtime.evaluate(emission, pts.astype(np.float32))[:, 3]
    np.testing.assert_allclose(executed, _compiled(tree, pts), rtol=1e-5, atol=1e-5)
