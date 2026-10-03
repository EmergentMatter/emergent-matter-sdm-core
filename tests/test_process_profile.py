"""Tests for ``software_defined_matter.process``: profile resolution + prior precedence.

The manufacturing substrate is an OPTIONAL dependency by design (string-ID
handoff, the materials pattern): ``.sdm`` stores ``metadata["process_profile"]``,
``emergent_matter_processes`` owns the numbers, and sdm-core must behave
correctly in BOTH worlds. It is not installed in this repo's env (checked:
absent from pyproject on purpose), so the resolver is exercised against a
stub injected into ``sys.modules`` that mirrors the real v0.3.0 surface
(``get_preset(s_id).noise_model`` -> ``ProcessNoise`` with ``PropertyValue``
slots in SI metres); one test runs against the real package and skips when
it is not importable.

The bar:

1. **The precedence table** in :func:`effective_prior`'s contract (authored
   wins > profile fills free mm gaps > delta), including every "does not
   apply" branch (wrong unit, not free, no profile).
2. **m -> mm conversion**: the preset speaks SI metres, params speak mm; a
   silent factor-1000 error here poisons every derived prior.
3. **Absence handling**: no metadata key (silent None), package missing
   (warn + None), preset without a noise model (warn + None), unknown preset
   ID (the substrate's own KeyError, loud).
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from software_defined_matter import Param, Part
from software_defined_matter.process import effective_prior, resolve_process_profile

# ---------------------------------------------------------------------------
# Stub of the emergent_matter_processes v0.3.0 surface
# ---------------------------------------------------------------------------


class _PV:
    """PropertyValue look-alike: d_value + s_units is all the resolver reads."""

    def __init__(self, d_value, s_units="m"):
        self.d_value = d_value
        self.s_units = s_units


class _Noise:
    def __init__(self, sigma_abs, differential_sigma=None, s_calibration_status="uncalibrated"):
        self.sigma_abs = sigma_abs
        self.differential_sigma = differential_sigma
        self.s_calibration_status = s_calibration_status


class _Preset:
    def __init__(self, s_id, noise_model):
        self.s_id = s_id
        self.noise_model = noise_model


# The org's two real machines, values from manufacturing v0.3.0.
_PRESETS = {
    "formlabs_fuse1_pa12": _Preset(
        "formlabs_fuse1_pa12",
        _Noise(_PV(0.0001)),  # sigma_abs = 0.1 mm
    ),
    "prusa_core_one": _Preset(
        "prusa_core_one",
        _Noise(_PV(0.0002), differential_sigma=_PV(0.00005), s_calibration_status="witness"),
    ),
    "generic_uncharacterized": _Preset("generic_uncharacterized", None),
}


def _get_preset(s_id):
    try:
        return _PRESETS[s_id]
    except KeyError:
        raise KeyError(f"Unknown preset {s_id!r}") from None


@pytest.fixture
def stub_manufacturing(monkeypatch):
    mod = types.ModuleType("emergent_matter_processes")
    mod.get_preset = _get_preset
    monkeypatch.setitem(sys.modules, "emergent_matter_processes", mod)
    return mod


@pytest.fixture
def no_manufacturing(monkeypatch):
    # A None entry in sys.modules makes `import x` raise ImportError: the
    # documented way to force the not-installed path even if a real package
    # ever appears in this env.
    monkeypatch.setitem(sys.modules, "emergent_matter_processes", None)


def _part_with_profile(profile_id="formlabs_fuse1_pa12") -> Part:
    part = Part(name="profile_fixture")
    if profile_id is not None:
        part.metadata["process_profile"] = profile_id
    return part


# ---------------------------------------------------------------------------
# resolve_process_profile
# ---------------------------------------------------------------------------


def test_resolves_sigma_in_mm(stub_manufacturing):
    profile = resolve_process_profile(_part_with_profile())
    assert profile == {
        "profile_id": "formlabs_fuse1_pa12",
        "sigma_abs_mm": pytest.approx(0.1),
        "differential_sigma_mm": None,
        "calibration_status": "uncalibrated",
    }


def test_resolves_differential_sigma_when_present(stub_manufacturing):
    profile = resolve_process_profile(_part_with_profile("prusa_core_one"))
    assert profile["sigma_abs_mm"] == pytest.approx(0.2)
    assert profile["differential_sigma_mm"] == pytest.approx(0.05)
    assert profile["calibration_status"] == "witness"


def test_no_declared_profile_is_silent_none(stub_manufacturing, recwarn):
    assert resolve_process_profile(_part_with_profile(None)) is None
    assert not recwarn.list


def test_missing_package_warns_and_returns_none(no_manufacturing):
    with pytest.warns(UserWarning, match="emergent_matter_processes"):
        assert resolve_process_profile(_part_with_profile()) is None


def test_preset_without_noise_model_warns_and_returns_none(stub_manufacturing):
    part = _part_with_profile("generic_uncharacterized")
    with pytest.warns(UserWarning, match="no sigma_abs"):
        assert resolve_process_profile(part) is None


def test_unknown_preset_id_raises(stub_manufacturing):
    with pytest.raises(KeyError, match="no_such_machine"):
        resolve_process_profile(_part_with_profile("no_such_machine"))


def test_unit_drift_fails_loud(stub_manufacturing):
    _PRESETS["bad_units"] = _Preset("bad_units", _Noise(_PV(0.1, s_units="mm")))
    try:
        with pytest.raises(ValueError, match="metres"):
            resolve_process_profile(_part_with_profile("bad_units"))
    finally:
        del _PRESETS["bad_units"]


@pytest.mark.skipif(
    importlib.util.find_spec("emergent_matter_processes") is None,
    reason="emergent_matter_processes not installed in this env "
    "(optional by design; stub tests cover the contract)",
)
def test_real_package_fuse_profile():
    profile = resolve_process_profile(_part_with_profile("formlabs_fuse1_pa12"))
    assert profile["sigma_abs_mm"] == pytest.approx(0.1)
    assert profile["calibration_status"] in ("uncalibrated", "witness", "volumetric")


# ---------------------------------------------------------------------------
# effective_prior precedence: authored > profile (free, mm) > delta
# ---------------------------------------------------------------------------

_PROFILE = {
    "profile_id": "formlabs_fuse1_pa12",
    "sigma_abs_mm": 0.1,
    "differential_sigma_mm": None,
    "calibration_status": "uncalibrated",
}


def test_authored_prior_wins_over_profile():
    p = Param("r", 5.0, free=True, unit="mm", prior={"dist": "uniform_pm", "half_width": 0.02})
    assert effective_prior(p, _PROFILE) == {
        "dist": "uniform_pm",
        "half_width": 0.02,
    }


def test_authored_tolerance_wins_over_profile():
    p = Param("r", 5.0, free=True, unit="mm", tolerance=0.3)
    assert effective_prior(p, _PROFILE) == {
        "dist": "uniform_pm",
        "half_width": 0.3,
    }


def test_profile_fills_free_mm_param():
    p = Param("r", 5.0, free=True, unit="mm")
    assert effective_prior(p, _PROFILE) == {"dist": "normal", "sigma": 0.1}


def test_profile_skips_non_mm_units():
    p = Param("rom", 0.5, free=True, unit="rad")
    assert effective_prior(p, _PROFILE) == {"dist": "delta"}


def test_profile_skips_fixed_params():
    p = Param("r", 5.0, free=False, unit="mm")
    assert effective_prior(p, _PROFILE) == {"dist": "delta"}


def test_no_profile_no_prior_is_delta():
    p = Param("r", 5.0, free=True, unit="mm")
    assert effective_prior(p, None) == {"dist": "delta"}


def test_authored_prior_survives_without_profile():
    p = Param("r", 5.0, free=True, unit="mm", prior={"dist": "normal", "sigma": 0.05})
    assert effective_prior(p, None) == {"dist": "normal", "sigma": 0.05}
