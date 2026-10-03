"""Tests for ``software_defined_matter.sample``: vmap-MC over Param priors.

The bar:

1. **Column contract**: ``(n, n_free)`` in ``free_param_names()`` order, the
   exact free-vector layout compiled closures consume. Get this wrong and
   every propagated draw silently evaluates the wrong geometry.
2. **Delta degeneracy**: no priors means every row IS ``param_vector()``:
   the probabilistic layer collapses exactly to today's deterministic
   behaviour, bit-for-bit constant columns.
3. **Distribution shape**: uniform / uniform_pm samples respect their
   bounds STRICTLY (every draw), normal matches its sigma statistically.
4. **Analytic propagation sanity**: for a sphere whose radius carries a
   ``normal(sigma)`` prior, the field at a fixed point is ``|p| - r``:
   exactly linear in the radius, so the per-point std equals sigma (up to MC
   error) EVERYWHERE, and the mean at the nominal surface is 0. This pins
   the whole propagate path (sampling -> vmap over the compiled closure ->
   statistics) to a hand-computable answer.
5. Plumbing: determinism under a fixed key, profile-driven sampling, empty
   free-param edge, quantile keys.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import MaterialRegion, Param, Part, sdf_primitive
from software_defined_matter.sample import propagate, sample_free_params
from software_defined_matter.sdf.compile import make_sdf_closure

KEY = jax.random.PRNGKey(42)

_PROFILE = {
    "profile_id": "formlabs_fuse1_pa12",
    "sigma_abs_mm": 0.1,
    "differential_sigma_mm": None,
    "calibration_status": "uncalibrated",
}


def _sphere_part(**r_kwargs) -> Part:
    part = Part(name="sample_fixture")
    part.add_param(Param("r", 5.0, free=True, unit="mm", **r_kwargs))
    part.add_material(MaterialRegion(1, "PA12", sdf_primitive("sphere", r={"$ref": "r"})))
    part.metadata["bbox"] = [[-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]]
    return part


# ---------------------------------------------------------------------------
# sample_free_params: shapes and column order
# ---------------------------------------------------------------------------


def test_shape_and_column_order():
    part = Part(name="order")
    part.add_param(Param("a", 1.0, free=True, unit="mm"))
    part.add_param(Param("fixed", 9.0, free=False, unit="mm"))
    part.add_param(Param("b", 2.0, free=True, unit="mm", prior={"dist": "normal", "sigma": 0.5}))
    samples = sample_free_params(part, 64, KEY)
    assert samples.shape == (64, 2)
    assert part.free_param_names() == ["a", "b"]
    # Column 0 is `a` (delta -> constant 1.0); column 1 is `b` (scattered).
    np.testing.assert_array_equal(np.asarray(samples[:, 0]), 1.0)
    assert np.std(np.asarray(samples[:, 1])) > 0.1


def test_delta_degeneracy_reproduces_param_vector():
    """A part with no priors samples to n identical copies of param_vector()."""
    part = Part(name="delta")
    part.add_param(Param("a", 1.5, free=True, unit="mm"))
    part.add_param(Param("b", -3.0, free=True, unit="rad"))
    samples = sample_free_params(part, 16, KEY)
    expected = np.tile(np.asarray(part.param_vector(), dtype=samples.dtype), (16, 1))
    np.testing.assert_array_equal(np.asarray(samples), expected)


def test_no_free_params_gives_zero_width():
    part = Part(name="empty")
    part.add_param(Param("fixed", 1.0, free=False, unit="mm"))
    assert sample_free_params(part, 8, KEY).shape == (8, 0)


def test_n_must_be_positive():
    with pytest.raises(ValueError, match="n >= 1"):
        sample_free_params(Part(name="bad"), 0, KEY)


# ---------------------------------------------------------------------------
# sample_free_params: distribution shapes
# ---------------------------------------------------------------------------


def test_uniform_bounds_respected_strictly():
    part = Part(name="uniform")
    part.add_param(
        Param("r", 5.0, free=True, unit="mm", prior={"dist": "uniform", "lo": 4.0, "hi": 6.0})
    )
    s = np.asarray(sample_free_params(part, 4096, KEY)[:, 0])
    assert s.min() >= 4.0 and s.max() < 6.0
    assert s.mean() == pytest.approx(5.0, abs=0.05)
    # It actually spreads over the box (std of U(4,6) is 2/sqrt(12) ~ 0.577).
    assert s.std() == pytest.approx(2.0 / np.sqrt(12.0), rel=0.05)


def test_uniform_pm_and_tolerance_agree_on_bounds():
    prior_part = Part(name="pm").add_param(
        Param("r", 5.0, free=True, unit="mm", prior={"dist": "uniform_pm", "half_width": 0.25})
    )
    tol_part = Part(name="tol").add_param(Param("r", 5.0, free=True, unit="mm", tolerance=0.25))
    for part in (prior_part, tol_part):
        s = np.asarray(sample_free_params(part, 2048, KEY)[:, 0])
        assert s.min() >= 4.75 and s.max() < 5.25
        assert s.std() == pytest.approx(0.5 / np.sqrt(12.0), rel=0.1)
    # Same key + same effective prior -> identical draws: the shorthand is
    # not merely similar, it IS uniform_pm.
    np.testing.assert_array_equal(
        np.asarray(sample_free_params(prior_part, 64, KEY)),
        np.asarray(sample_free_params(tol_part, 64, KEY)),
    )


def test_normal_matches_sigma():
    part = Part(name="normal").add_param(
        Param("r", 5.0, free=True, unit="mm", prior={"dist": "normal", "sigma": 0.2})
    )
    s = np.asarray(sample_free_params(part, 8192, KEY)[:, 0])
    assert s.mean() == pytest.approx(5.0, abs=0.02)
    assert s.std() == pytest.approx(0.2, rel=0.05)


def test_profile_supplies_sigma_for_unauthored_mm_param():
    part = Part(name="prof").add_param(Param("r", 5.0, free=True, unit="mm"))
    s = np.asarray(sample_free_params(part, 8192, KEY, profile=_PROFILE)[:, 0])
    assert s.std() == pytest.approx(0.1, rel=0.05)
    # And without the profile the same param is a delta.
    np.testing.assert_array_equal(np.asarray(sample_free_params(part, 8, KEY)), 5.0)


def test_same_key_same_samples():
    part = _sphere_part(prior={"dist": "normal", "sigma": 0.2})
    a = sample_free_params(part, 32, KEY)
    b = sample_free_params(part, 32, KEY)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    c = sample_free_params(part, 32, jax.random.PRNGKey(7))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


# ---------------------------------------------------------------------------
# propagate: analytic sphere sanity
# ---------------------------------------------------------------------------


def test_propagate_sphere_std_equals_sigma():
    """sphere(r): field at p is |p| - r, linear in r with coefficient -1, so
    std(field) == sigma at EVERY point and mean(field at surface) == 0."""
    sigma = 0.2
    part = _sphere_part(prior={"dist": "normal", "sigma": sigma})
    fn = make_sdf_closure(part.materials[0].sdf_tree, part)
    points = jnp.array(
        [
            [5.0, 0.0, 0.0],  # on the nominal surface
            [0.0, 0.0, 0.0],  # centre
            [0.0, 7.0, 0.0],  # outside
        ]
    )
    stats = propagate(fn, part, points, n=8192, key=KEY)
    np.testing.assert_allclose(np.asarray(stats["std"]), [sigma, sigma, sigma], rtol=0.05)
    assert float(stats["mean"][0]) == pytest.approx(0.0, abs=3 * sigma / np.sqrt(8192))
    assert float(stats["mean"][1]) == pytest.approx(-5.0, abs=0.01)
    assert float(stats["mean"][2]) == pytest.approx(2.0, abs=0.01)
    # Median tracks the mean for a symmetric prior through a linear map.
    assert float(stats["quantiles"][0.5][0]) == pytest.approx(0.0, abs=0.01)
    assert set(stats["quantiles"]) == {0.05, 0.5, 0.95}
    # ~2 sigma spread between the 5th and 95th percentile bands, per point.
    spread = np.asarray(stats["quantiles"][0.95]) - np.asarray(stats["quantiles"][0.05])
    np.testing.assert_allclose(spread, 2 * 1.6449 * sigma, rtol=0.05)


def test_propagate_all_delta_is_deterministic():
    part = _sphere_part()  # no prior anywhere
    fn = make_sdf_closure(part.materials[0].sdf_tree, part)
    stats = propagate(fn, part, jnp.array([[5.0, 0.0, 0.0]]), n=32, key=KEY)
    assert float(stats["std"][0]) == pytest.approx(0.0, abs=1e-7)
    assert float(stats["mean"][0]) == pytest.approx(0.0, abs=1e-6)


def test_propagate_accepts_single_point_and_default_key():
    part = _sphere_part(tolerance=0.1)
    fn = make_sdf_closure(part.materials[0].sdf_tree, part)
    stats = propagate(fn, part, jnp.array([5.0, 0.0, 0.0]), n=64)
    assert stats["mean"].shape == (1,)
    assert stats["std"].shape == (1,)
    # tolerance = +/-0.1 uniform: every draw within the band, so the field at
    # the nominal surface never leaves [-0.1, 0.1].
    assert float(stats["quantiles"][0.05][0]) >= -0.1
    assert float(stats["quantiles"][0.95][0]) <= 0.1


def test_propagate_with_profile():
    """End-to-end: an unauthored free mm radius + resolved profile -> the
    field carries the machine's sigma."""
    part = _sphere_part()
    fn = make_sdf_closure(part.materials[0].sdf_tree, part)
    stats = propagate(fn, part, jnp.array([[5.0, 0.0, 0.0]]), n=8192, key=KEY, profile=_PROFILE)
    assert float(stats["std"][0]) == pytest.approx(0.1, rel=0.05)
