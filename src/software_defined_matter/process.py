"""Resolve a part's manufacturing process profile into param-prior defaults.

This is the schema-0.3 profile linkage: a ``.sdm`` declares WHICH process
will print it as a string ID in ``metadata["process_profile"]``, e.g.
``"formlabs_fuse1_pa12"`` -- and the noise numbers themselves are meant to
live in the optional, separate ``emergent_matter_processes`` package
(``Preset.noise_model``, a ``ProcessNoise``), imported lazily below. Same
string-ID handoff as materials: sdm-core stores the ID, the data package
owns the data, and there is deliberately NO hard dependency in
``pyproject.toml`` -- a core install without the manufacturing package still
loads, validates, and samples every ``.sdm``; profile-derived priors just
resolve to nothing.

Division of labour (the org's core-describes / physics-judges cut): this
module only converts "which printer" into "what sigma"; chance-constrained
referees that *judge* a part under that sigma belong to a separate
physics/verification project, not part of this repo.

Prior precedence (implemented by :func:`effective_prior`)
---------------------------------------------------------
1. **Authored wins.** A ``Param.prior`` or ``Param.tolerance`` written by the
   author is used verbatim: the profile never overrides an explicit spec,
   even a tighter or looser one.
2. **Profile fills the mm-sized free gaps.** Otherwise, if the param is
   ``free``, its unit is ``"mm"``, and a profile resolved, the param gets
   ``{"dist": "normal", "sigma": sigma_abs_mm}``: the process's absolute,
   scale-invariant post-compensation residual. Only linear-mm params: angles,
   counts, ratios and derived quantities do not inherit a length noise.
   Only free params: fixed params are frame/topology givens, not printed
   dimensions being explored.
3. **Delta otherwise.** No authored prior, no applicable profile; the param
   is exact: ``{"dist": "delta"}``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from software_defined_matter.model import Param, Part


_M_TO_MM = 1000.0


def resolve_process_profile(part: Part) -> dict[str, Any] | None:
    """Resolve ``part.metadata["process_profile"]`` to noise figures in mm.

    Returns ``None`` when the part declares no profile, when the optional
    ``emergent_matter_processes`` package is not installed (with a clear
    warning: the declaration exists but cannot be honoured), or when the
    preset carries no usable noise model (also warned: a declared profile
    that silently contributes nothing would be worse than an error).

    An unknown profile ID raises the substrate's own ``KeyError``: that is
    an authoring bug, not a missing optional dependency.

    Returns:
        ``{"profile_id": str,
        "sigma_abs_mm": float,``: absolute (scale-invariant) 1-sigma
        dimensional residual, converted from the preset's SI metres;
        ``"differential_sigma_mm": float | None,``: nearby-feature relative
        sigma (the print-in-place clearance number), ``None`` until the
        gap-ladder artifact measures it;
        ``"calibration_status": str}``: the preset's calibration provenance
        (``uncalibrated`` / ``witness`` / ``volumetric``).
    """
    profile_id = (part.metadata or {}).get("process_profile")
    if not profile_id:
        return None

    try:
        import emergent_matter_processes as _emm
    except ImportError:
        warnings.warn(
            f"Part {part.name!r} declares process_profile={profile_id!r} but "
            f"the optional 'emergent_matter_processes' package is not "
            f"installed: profile-derived priors resolve to None (params "
            f"keep their authored priors / delta). Install the manufacturing "
            f"substrate to honour the profile.",
            stacklevel=2,
        )
        return None

    preset = _emm.get_preset(profile_id)  # unknown ID -> KeyError, fail loud
    noise = preset.noise_model
    if noise is None or noise.sigma_abs is None:
        warnings.warn(
            f"Process profile {preset.s_id!r} has no sigma_abs noise model: "
            f"it cannot supply default priors. Params keep their authored "
            f"priors / delta.",
            stacklevel=2,
        )
        return None

    def _mm(pv: Any) -> float:
        # ProcessNoise validates its own units, but a silent unit drift here
        # would mis-scale every derived prior by 1000x; check anyway.
        if pv.s_units != "m":
            raise ValueError(
                f"Process profile {preset.s_id!r}: expected SI metres for "
                f"noise figures, got units {pv.s_units!r}."
            )
        return float(pv.d_value) * _M_TO_MM

    return {
        "profile_id": preset.s_id,
        "sigma_abs_mm": _mm(noise.sigma_abs),
        "differential_sigma_mm": (
            _mm(noise.differential_sigma) if noise.differential_sigma is not None else None
        ),
        "calibration_status": noise.s_calibration_status,
    }


def effective_prior(
    param: Param,
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """The prior a sampler should actually draw from for ``param``.

    Precedence (see module docstring): authored ``prior``/``tolerance`` wins;
    else a free ``"mm"`` param inherits ``normal(sigma=sigma_abs_mm)`` from
    ``profile`` (as returned by :func:`resolve_process_profile`); else delta.

    Always returns a normalised distribution spec (``tolerance`` already
    rewritten to ``uniform_pm``), suitable for
    :func:`software_defined_matter.sample.sample_free_params`.
    """
    if param.has_authored_prior():
        return param.authored_prior()
    if profile is not None and param.free and param.unit == "mm":
        return {"dist": "normal", "sigma": float(profile["sigma_abs_mm"])}
    return {"dist": "delta"}


__all__ = ["effective_prior", "resolve_process_profile"]
