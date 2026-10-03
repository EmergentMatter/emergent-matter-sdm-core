"""Tests for ``Param.prior`` / ``Param.tolerance``: the schema-0.3 probabilistic layer.

The bar, in order of how much each would hurt to get wrong:

1. **Back-compat is absolute**: an absent prior IS a delta at the authored
   value. Every 0.1/0.2 file must load unchanged, warning-free, and a part
   with no priors must keep serialising as 0.2. The probabilistic layer may
   not churn a single existing file.
2. **The wire-version rule**: ``schema_version`` is emitted as 0.3 ONLY when
   some param carries a non-delta prior (or a tolerance). An explicit
   ``{"dist": "delta"}`` normalises away and does NOT bump the version.
3. **Validation fails loud at construction**: unknown dists, missing/extra
   keys, non-positive scales, a uniform that excludes its own nominal, and
   the prior+tolerance double-authoring all raise ``ValueError`` in
   ``Param.__post_init__``, not at sample time.
4. **Round-trip fidelity**: to_dict/from_dict and save/load preserve priors
   and the tolerance shorthand byte-for-byte (tolerance stays tolerance; it
   is NOT rewritten to its uniform_pm equivalent on the wire).
5. Schema strictness: the 0.3 JSON Schema accepts exactly the specs the
   model accepts, and the 0.2 schema still rejects a prior key outright
   (additionalProperties: false), so a mislabelled file cannot pass.
"""

from __future__ import annotations

import warnings

import jsonschema
import pytest

from software_defined_matter import (
    KNOWN_SCHEMA_VERSIONS,
    MaterialRegion,
    Param,
    Part,
    load,
    save,
    sdf_primitive,
    validate,
)
from software_defined_matter.io import load_schema
from software_defined_matter.model import min_schema_version_for


def _part(**param_kwargs) -> Part:
    """A minimal valid part with one free radius param."""
    part = Part(name="prior_fixture")
    part.add_param(Param("r", 5.0, free=True, unit="mm", **param_kwargs))
    part.add_material(MaterialRegion(1, "PA12", sdf_primitive("sphere", r={"$ref": "r"})))
    part.metadata["bbox"] = [[-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]]
    return part


# ---------------------------------------------------------------------------
# Defaults and normalisation
# ---------------------------------------------------------------------------


def test_default_is_delta():
    p = Param("r", 5.0, unit="mm")
    assert p.prior is None
    assert p.tolerance is None
    assert p.authored_prior() == {"dist": "delta"}
    assert not p.has_authored_prior()


def test_explicit_delta_normalises_to_absent():
    """Absent-means-delta is the canonical spelling: an explicit delta must
    round-trip identically to no prior at all (and not bump the version)."""
    p = Param("r", 5.0, unit="mm", prior={"dist": "delta"})
    assert p.prior is None
    assert p == Param("r", 5.0, unit="mm")
    assert "prior" not in p.to_dict()


def test_tolerance_is_uniform_pm_shorthand():
    p = Param("r", 5.0, unit="mm", tolerance=0.1)
    assert p.has_authored_prior()
    assert p.authored_prior() == {"dist": "uniform_pm", "half_width": 0.1}
    # ... but the wire keeps the author's spelling.
    assert p.to_dict()["tolerance"] == 0.1
    assert "prior" not in p.to_dict()


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_prior, match",
    [
        ({"dist": "lognormal", "sigma": 0.1}, "unknown prior dist"),
        ({"sigma": 0.1}, "unknown prior dist"),
        ({"dist": "normal"}, "missing"),
        ({"dist": "normal", "sigma": 0.0}, "sigma > 0"),
        ({"dist": "normal", "sigma": -0.1}, "sigma > 0"),
        ({"dist": "normal", "sigma": "0.1"}, "must be a number"),
        ({"dist": "normal", "sigma": 0.1, "mu": 5.0}, "unexpected"),
        ({"dist": "delta", "value": 5.0}, "unexpected"),
        ({"dist": "uniform", "lo": 4.0}, "missing"),
        ({"dist": "uniform", "lo": 6.0, "hi": 4.0}, "lo < hi"),
        ({"dist": "uniform", "lo": 6.0, "hi": 8.0}, "excludes"),
        ({"dist": "uniform_pm", "half_width": 0.0}, "half_width > 0"),
        ({"dist": "uniform_pm", "half_width": -1.0}, "half_width > 0"),
    ],
)
def test_bad_priors_raise(bad_prior, match):
    with pytest.raises(ValueError, match=match):
        Param("r", 5.0, unit="mm", prior=bad_prior)


def test_prior_must_be_a_dict():
    with pytest.raises(ValueError, match="must be a dict"):
        Param("r", 5.0, unit="mm", prior="normal")


@pytest.mark.parametrize("bad_tol", [0.0, -0.5, True, "0.1"])
def test_bad_tolerance_raises(bad_tol):
    with pytest.raises(ValueError, match="tolerance"):
        Param("r", 5.0, unit="mm", tolerance=bad_tol)


def test_prior_and_tolerance_together_raise():
    with pytest.raises(ValueError, match="ambiguous"):
        Param("r", 5.0, unit="mm", prior={"dist": "normal", "sigma": 0.1}, tolerance=0.1)


def test_uniform_containing_value_is_accepted():
    p = Param("r", 5.0, unit="mm", prior={"dist": "uniform", "lo": 4.5, "hi": 5.5})
    assert p.authored_prior() == {"dist": "uniform", "lo": 4.5, "hi": 5.5}


# ---------------------------------------------------------------------------
# Wire-version rule
# ---------------------------------------------------------------------------


def test_part_without_priors_stays_on_baseline_version():
    part = _part()
    assert part.to_dict()["schema_version"] == min_schema_version_for(part) == "0.2"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prior": {"dist": "normal", "sigma": 0.1}},
        {"prior": {"dist": "uniform", "lo": 4.0, "hi": 6.0}},
        {"prior": {"dist": "uniform_pm", "half_width": 0.05}},
        {"tolerance": 0.1},
    ],
)
def test_any_nondelta_prior_bumps_to_0_3(kwargs):
    part = _part(**kwargs)
    doc = part.to_dict()
    assert doc["schema_version"] == min_schema_version_for(part) == "0.3"
    validate(doc)  # and the emitted doc passes its own declared schema


def test_explicit_delta_does_not_bump():
    part = _part(prior={"dist": "delta"})
    doc = part.to_dict()
    assert doc["schema_version"] == min_schema_version_for(part) == "0.2"


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"prior": {"dist": "normal", "sigma": 0.1}},
        {"prior": {"dist": "uniform", "lo": 4.0, "hi": 6.0}},
        {"prior": {"dist": "uniform_pm", "half_width": 0.05}},
        {"tolerance": 0.1},
    ],
)
def test_to_dict_roundtrip(kwargs):
    part = _part(**kwargs)
    d1 = part.to_dict()
    part2 = Part.from_dict(d1)
    assert part2.to_dict() == d1
    assert part2.params["r"] == part.params["r"]


def test_file_roundtrip_with_priors(tmp_path):
    part = _part(prior={"dist": "normal", "sigma": 0.1})
    path = tmp_path / "prior_part.sdm"
    save(part, path)  # validates against the 0.3 schema on the way out
    reloaded = load(path)  # and on the way back in
    assert reloaded.to_dict() == part.to_dict()
    assert reloaded.params["r"].prior == {"dist": "normal", "sigma": 0.1}


def test_file_roundtrip_with_tolerance(tmp_path):
    part = _part(tolerance=0.05)
    path = tmp_path / "tol_part.sdm"
    save(part, path)
    reloaded = load(path)
    assert reloaded.params["r"].tolerance == 0.05
    assert reloaded.params["r"].prior is None


# ---------------------------------------------------------------------------
# Back-compat: 0.1 / 0.2 documents
# ---------------------------------------------------------------------------


def test_loading_a_0_2_file_is_warning_free(tmp_path):
    """A pre-prior 0.2 document loads with no warning and delta semantics."""
    doc = _part().to_dict()
    assert doc["schema_version"] == "0.2"
    path = tmp_path / "legacy_0_2.sdm"
    import json

    path.write_text(json.dumps(doc))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reloaded = load(path)
    assert reloaded.params["r"].authored_prior() == {"dist": "delta"}
    # Re-saving does not churn the version.
    assert reloaded.to_dict()["schema_version"] == "0.2"


def test_loading_a_0_1_file_is_warning_free(tmp_path):
    doc = _part().to_dict()
    doc["schema_version"] = "0.1"
    path = tmp_path / "legacy_0_1.sdm"
    import json

    path.write_text(json.dumps(doc))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reloaded = load(path)
    assert reloaded.params["r"].authored_prior() == {"dist": "delta"}


def test_all_wire_versions_are_known():
    assert {"0.1", "0.2", "0.3", "0.4", "0.5", "0.6"} == set(KNOWN_SCHEMA_VERSIONS)


# ---------------------------------------------------------------------------
# JSON Schema strictness
# ---------------------------------------------------------------------------


def test_0_2_schema_rejects_prior_key():
    """A prior in a document still claiming 0.2 must fail validation: the
    param def is additionalProperties:false, and version dispatch selects the
    0.2 schema for it."""
    doc = _part(prior={"dist": "normal", "sigma": 0.1}).to_dict()
    doc["schema_version"] = "0.2"  # mislabel it
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


@pytest.mark.parametrize(
    "bad_prior",
    [
        {"dist": "lognormal", "sigma": 0.1},
        {"dist": "normal", "sigma": -0.1},
        {"dist": "normal"},
        {"dist": "uniform", "lo": 4.0},
        {"dist": "uniform_pm", "half_width": 0},
        {"dist": "normal", "sigma": 0.1, "mu": 5.0},
    ],
)
def test_0_3_schema_rejects_malformed_priors(bad_prior):
    doc = _part().to_dict()
    doc["schema_version"] = "0.3"
    doc["params"]["r"]["prior"] = bad_prior
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, load_schema("0.3"))


def test_0_3_schema_rejects_prior_plus_tolerance():
    doc = _part().to_dict()
    doc["schema_version"] = "0.3"
    doc["params"]["r"]["prior"] = {"dist": "normal", "sigma": 0.1}
    doc["params"]["r"]["tolerance"] = 0.1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, load_schema("0.3"))


def test_0_3_schema_tracks_0_2_vocabulary():
    """0.3 is 0.2 plus ``expr``/``prior``/``tolerance`` param keys, and drops nothing.

    ``sdf``/``field`` are deliberately NOT compared byte-for-byte here any
    more: 0.3 is generated from ``wire.py`` (schema/_generate.py) and is now
    a strictly TIGHTER superset of 0.2's vocabulary (per-kind kwarg shapes
    0.2 never enforced), which is the point of this PR --
    tests/test_schema_generate.py proves that tightening rejects nothing
    that was previously valid. ``Param.expr`` reuses the shared ``$defs/expr``
    vocabulary (objectives/constraints already used it in 0.2); that def stays
    byte-identical.
    """
    s2 = load_schema("0.2")
    s3 = load_schema("0.3")
    assert s3["$defs"]["expr"] == s2["$defs"]["expr"], "expr vocabulary drifted"
    p2 = dict(s2["$defs"]["param"]["properties"])
    p3 = dict(s3["$defs"]["param"]["properties"])
    assert set(p3) - set(p2) == {"expr", "prior", "tolerance"}
    assert p3["expr"] == {"$ref": "#/$defs/expr"}
    assert set(p2) <= set(p3), "a 0.2 param key was dropped in 0.3"
