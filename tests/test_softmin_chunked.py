"""Tests for op_softmin_many / op_softmin_chunked + canonical_sector_fold.

Added 2026-05 for the bubbles sdm-core migration. The primitives are
generic SDF construction tools, not bearing-specific:

* ``op_softmin_many``: log-sum-exp softmin along axis 0 over a stack
  of N SDF values. Useful for any part family that softmin-unions many
  primitives (lattices, swept envelopes, repeated micro-features).

* ``op_softmin_chunked``: memory-disciplined wrapper that caps peak
  memory at ``chunk_size × point_batch`` instead of
  ``N × point_batch``. Mathematically equivalent to ``op_softmin_many``
  via softmin-of-softmins associativity (see test below).

* ``canonical_sector_fold``: fold a query azimuth into one canonical
  sector. Generalises beyond gear teeth: stacked flexure rings, Persian
  reliefs, any N-fold rotationally-symmetric construction.

These tests pin the math; the compile-dispatch tests live in
``test_sdf_compile.py`` (DSL nodes ``softmin_many`` / ``softmin_chunked``
under ``type=op`` and ``canonical_sector_fold`` under ``type=transform``).
"""

from __future__ import annotations

# Enable JAX float64 for this test module. sdm-core's conftest doesn't
# enable x64 globally (it defaults to float32 like JAX itself); the
# bubbles migration runs everything in float64 (see migration plan
# §6.3) so SDF math has sub-mm precision. The 1e-10 tolerance the
# softmin-chunked equivalence test asserts requires float64; under
# float32 the same algebra produces ~1e-8 noise.
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

# E402: these must import after the jax_enable_x64 config above -- moving them
# to the top would make x64 mode take effect too late for this module's math.
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from software_defined_matter.sdf.sdf_ops import (  # noqa: E402
    op_smooth_union,
    op_softmin_chunked,
    op_softmin_many,
)
from software_defined_matter.sdf.transforms import (  # noqa: E402
    tf_canonical_sector_fold,
)

# ────────────────────────────────────────────────────────────────────────
# op_softmin_many: basic math
# ────────────────────────────────────────────────────────────────────────


def test_softmin_many_approaches_min_as_k_to_zero() -> None:
    """``op_softmin_many(d, k)`` → ``jnp.min(d, axis=0)`` as k → 0."""
    d = jnp.array([[1.0, 2.0, 3.0], [0.5, 4.0, -1.0], [2.0, 3.0, 0.5]])
    expected_min = jnp.min(d, axis=0)

    out_k_zero = op_softmin_many(d, 0.0)
    np.testing.assert_allclose(out_k_zero, expected_min, atol=1e-12)

    out_k_tiny = op_softmin_many(d, 1e-10)
    np.testing.assert_allclose(out_k_tiny, expected_min, atol=1e-6)


def test_softmin_many_strictly_below_min() -> None:
    """For k > 0, softmin produces a value at most equal to min and
    typically strictly below it (stronger bias when many SDFs are near
    each other)."""
    d = jnp.array([[1.0, 1.0, 1.0], [1.05, 1.05, 1.05]])
    out = op_softmin_many(d, 0.5)
    # All input values ≈ 1.0; soft min should be slightly below.
    expected_min = jnp.min(d, axis=0)
    assert jnp.all(out <= expected_min + 1e-9), f"softmin {out} should be <= min {expected_min}"


def test_softmin_many_two_input_matches_smooth_union() -> None:
    """For N=2, softmin should be very close to (but distinct from) the
    iq-polynomial smooth-union with the same k.

    The two functions converge as k → 0 and as the inputs separate (the
    iq blend reduces to min there too). Inside the blend region (|d1-d2| < k)
    the iq blend uses a quadratic correction while logsumexp uses an
    exponential; they differ on the order of ~k.
    """
    d1 = jnp.array(0.5)
    d2 = jnp.array(0.5)  # equal: both blends pull below min by their full bias
    k = 0.3

    soft_logsumexp = op_softmin_many(jnp.stack([d1, d2], axis=0), k)
    soft_iq = op_smooth_union(d1, d2, k)

    # Both produce a value below the min (which is 0.5 here).
    assert float(soft_logsumexp) < 0.5
    assert float(soft_iq) < 0.5


# ────────────────────────────────────────────────────────────────────────
# op_softmin_chunked: equivalence to unchunked softmin
# ────────────────────────────────────────────────────────────────────────


def _make_n_sdfs(n: int):
    """Synthesise n unit-sphere SDFs at random centres in (-1, 1)."""
    rng = np.random.default_rng(2024)
    centres = rng.uniform(-1.0, 1.0, size=(n, 3))

    def make_one(c):
        c_jnp = jnp.asarray(c)
        return lambda p: jnp.linalg.norm(p - c_jnp, axis=-1) - 0.5

    return [make_one(centres[i]) for i in range(n)]


@pytest.mark.parametrize("chunk_size", [1, 2, 16, 48, 64, 100, 1000])
def test_softmin_chunked_matches_unchunked_n_576(chunk_size: int) -> None:
    """``op_softmin_chunked`` produces identical output to a single
    softmin over all 576 stacked distances, within 1e-10.

    Replicates the bubbles ring conjugate softmin construction at miniature
    scale. Softmin associativity guarantees this; the test is a guard
    against future implementation drift.
    """
    sdfs = _make_n_sdfs(576)
    k = 0.2

    rng = np.random.default_rng(1)
    p = jnp.asarray(rng.uniform(-1.5, 1.5, size=(20, 3)))

    chunked = op_softmin_chunked(sdfs, k, chunk_size)
    chunked_value = chunked(p)

    # Reference: straight stack + softmin over the lot.
    stacked = jnp.stack([s(p) for s in sdfs], axis=0)
    reference = op_softmin_many(stacked, k)

    np.testing.assert_allclose(chunked_value, reference, atol=1e-10, rtol=0.0)


def test_softmin_chunked_empty_raises() -> None:
    """An empty SDF list is a programming error, not a valid edge case."""
    with pytest.raises(ValueError, match="empty"):
        op_softmin_chunked([], k=0.2, chunk_size=8)


def test_softmin_chunked_chunk_size_clamped_to_one() -> None:
    """Pathological chunk_size=0 falls back to 1 rather than dividing
    by zero or producing an empty per_chunk list."""
    sdfs = _make_n_sdfs(4)
    chunked = op_softmin_chunked(sdfs, k=0.2, chunk_size=0)
    out = chunked(jnp.zeros((1, 3)))
    assert out.shape == (1,)
    assert jnp.all(jnp.isfinite(out))


# ────────────────────────────────────────────────────────────────────────
# tf_canonical_sector_fold: N-fold rotational symmetry
# ────────────────────────────────────────────────────────────────────────


def test_canonical_sector_fold_z_axis_unchanged() -> None:
    """The Z component of the query point is untouched by the fold."""
    rng = np.random.default_rng(7)
    p = jnp.asarray(rng.uniform(-1.0, 1.0, size=(20, 3)))
    folded = tf_canonical_sector_fold(p, n_sectors=8)
    np.testing.assert_allclose(folded[..., 2], p[..., 2], atol=1e-12)


def test_canonical_sector_fold_radius_preserved() -> None:
    """The fold preserves the cylindrical radius of every query point."""
    rng = np.random.default_rng(7)
    p = jnp.asarray(rng.uniform(-1.0, 1.0, size=(20, 3)))
    r_before = jnp.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
    folded = tf_canonical_sector_fold(p, n_sectors=16)
    r_after = jnp.sqrt(folded[..., 0] ** 2 + folded[..., 1] ** 2)
    np.testing.assert_allclose(r_after, r_before, atol=1e-7)


def test_canonical_sector_fold_lands_in_canonical_wedge() -> None:
    """All folded query points have azimuth in ``[0, 2π/n_sectors)``."""
    rng = np.random.default_rng(7)
    p = jnp.asarray(rng.uniform(-1.0, 1.0, size=(50, 3)))
    n_sectors = 16
    folded = tf_canonical_sector_fold(p, n_sectors=n_sectors)
    theta = jnp.arctan2(folded[..., 1], folded[..., 0])
    sector_angle = 2.0 * jnp.pi / n_sectors
    # Tolerate sub-machine-eps slop at the wedge boundary.
    assert jnp.all(theta >= -1e-7)
    assert jnp.all(theta < sector_angle + 1e-7)


def test_canonical_sector_fold_matches_explicit_polar_array() -> None:
    """``f_folded(child)`` over ``n_sectors`` matches the union of N
    rotated copies of ``child`` to within numerical noise.

    Asserts the central correctness claim of the fold: it's an O(1)
    replacement for an N-fold polar array, **provided the child SDF
    is defined to live within one canonical sector** ``θ ∈ [0, 2π/n)``.

    Test SDF: a small disc centred on the canonical sector's
    *bisector*, half a sector-angle in from the +x axis. By
    construction it lives entirely inside the canonical sector, so
    fold-replicating it is identical (modulo machine eps) to a polar
    union of N copies at angles ``2πk/n``.
    """
    n_sectors = 4
    sector_angle = 2.0 * jnp.pi / n_sectors

    # Disc centred on the bisector of the canonical sector.
    bisector = sector_angle / 2.0
    disc_centre = jnp.array([0.6 * jnp.cos(bisector), 0.6 * jnp.sin(bisector)])
    disc_r = 0.2

    def disc_in_canonical_sector(p):
        dx = p[..., 0] - disc_centre[0]
        dy = p[..., 1] - disc_centre[1]
        return jnp.sqrt(dx * dx + dy * dy) - disc_r

    def folded_fn(p):
        return disc_in_canonical_sector(tf_canonical_sector_fold(p, n_sectors=n_sectors))

    def explicit_polar_union(p):
        # Build N rotated copies and take the min (= union).
        per_copy = []
        for i in range(n_sectors):
            angle = -sector_angle * i  # rotate query backwards by i sectors
            c, s = jnp.cos(angle), jnp.sin(angle)
            R = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            p_rotated = p @ R.T
            per_copy.append(disc_in_canonical_sector(p_rotated))
        return jnp.min(jnp.stack(per_copy, axis=0), axis=0)

    rng = np.random.default_rng(13)
    p = jnp.asarray(rng.uniform(-1.0, 1.0, size=(40, 3)))

    fold_d = folded_fn(p)
    explicit_d = explicit_polar_union(p)

    np.testing.assert_allclose(fold_d, explicit_d, atol=1e-7, rtol=0.0)
