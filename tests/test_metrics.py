"""Metrics against hand calculations, closed forms and known identities."""

import math

import numpy as np
import pytest

from sctwin import metrics as M

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _normal_ppf(p):
    # inverse CDF via bisection on erf (avoids a scipy dependency in core tests)
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_mae_wape():
    y, f = np.array([1.0, 2, 3]), np.array([2.0, 2, 1])
    assert M.mae(y, f) == pytest.approx(1.0)
    assert M.wape(y, f) == pytest.approx(3 / 6)
    assert M.wape(np.zeros(3), f) is None


def test_pinball_hand():
    out = M.pinball(np.array([1.0]), np.array([[0.0, 2.0]]), [0.1, 0.9])
    np.testing.assert_allclose(out, [[0.1, 0.1]])


def test_wis_hand():
    # y=0, quantiles (0.25, 0.5, 0.75) = (1, 2, 3): (0.5*2 + 0.25*(2 + 4*1)) / 1.5
    w = M.wis(np.array([0.0]), np.array([[1.0, 2.0, 3.0]]), [0.25, 0.5, 0.75])
    np.testing.assert_allclose(w, [2.5 / 1.5])


def test_wis_equals_scaled_pinball_sum():
    rng = np.random.default_rng(0)
    y = rng.gamma(2, 1, 500)
    q, _ = M.sort_quantiles(rng.gamma(2, 1, (500, len(TAUS))))
    k = (len(TAUS) - 1) / 2
    np.testing.assert_allclose(M.wis(y, q, TAUS), M.pinball(y, q, TAUS).sum(1) / (k + 0.5))


def test_wis_rejects_bad_taus():
    with pytest.raises(ValueError):
        M.wis(np.zeros(1), np.zeros((1, 2)), [0.1, 0.9])
    with pytest.raises(ValueError):
        M.wis(np.zeros(1), np.zeros((1, 3)), [0.1, 0.5, 0.8])


def test_crps_normal_closed_form_value():
    # y = mu, sigma = 1: 2 phi(0) - 1/sqrt(pi)
    np.testing.assert_allclose(M.crps_normal(np.array([0.0]), 0.0, 1.0), [0.2336949], atol=1e-6)


def test_crps_samples_matches_normal():
    rng = np.random.default_rng(1)
    y = np.array([-1.0, 0.0, 0.7, 2.5])
    samples = rng.normal(0.0, 1.0, (4, 200_000))
    np.testing.assert_allclose(M.crps_samples(y, samples), M.crps_normal(y, 0.0, 1.0), rtol=1e-2)


def test_crps_single_sample_is_absolute_error():
    np.testing.assert_allclose(M.crps_samples(np.array([1.0]), np.array([[3.5]])), [2.5])


def test_crps_quantiles_dense_grid_matches_normal():
    taus = np.linspace(0.005, 0.995, 199)
    q = np.array([[_normal_ppf(t) for t in taus]] * 3)
    y = np.array([-0.5, 0.0, 1.2])
    np.testing.assert_allclose(M.crps_quantiles(y, q, taus), M.crps_normal(y, 0.0, 1.0), rtol=2e-2)


def test_interval_coverage_calibrated():
    rng = np.random.default_rng(2)
    y = rng.normal(size=20_000)
    q = np.tile([_normal_ppf(t) for t in TAUS], (y.size, 1))
    cov = M.interval_coverage(y, q, TAUS)
    for level in ("0.9", "0.8", "0.5"):
        assert abs(cov[level]["coverage"] - cov[level]["nominal"]) < 0.015


def test_randomized_pit_uniform_with_point_mass():
    rng = np.random.default_rng(3)
    y = rng.poisson(0.7, 20_000).astype(float)  # ~50% zeros
    s = rng.poisson(0.7, (20_000, 400)).astype(float)
    hist, _ = np.histogram(M.randomized_pit(y, s, rng), bins=10, range=(0, 1))
    assert np.all(np.abs(hist / y.size - 0.1) < 0.015)


def test_sort_quantiles():
    q, frac = M.sort_quantiles(np.array([[1.0, 0.5, 2.0], [0.0, 1.0, 2.0]]))
    np.testing.assert_array_equal(q, [[0.5, 1, 2], [0, 1, 2]])
    assert frac == 0.5


def test_bootstrap_identical_losses():
    loss = np.random.default_rng(0).random(100)
    r = M.paired_bootstrap(loss, loss, np.arange(100) // 10)
    assert r.mean_diff == 0 and r.ci_low == 0 and r.ci_high == 0 and r.p_value == 1.0


def test_bootstrap_detects_shift():
    rng = np.random.default_rng(1)
    b = rng.random(2000)
    a = b - 0.1 + rng.normal(0, 0.05, 2000)
    r = M.paired_bootstrap(a, b, np.arange(2000) // 20)
    assert r.ci_high < 0 and r.p_value < 0.01


def test_cluster_bootstrap_wider_under_within_cluster_correlation():
    rng = np.random.default_rng(2)
    n_cl, per = 40, 50
    effect = np.repeat(rng.normal(0, 1, n_cl), per)  # shared per-cluster component
    d = effect + rng.normal(0, 0.1, n_cl * per)
    clustered = M.paired_bootstrap(d, np.zeros_like(d), np.repeat(np.arange(n_cl), per))
    iid = M.paired_bootstrap(d, np.zeros_like(d), np.arange(d.size))
    assert (clustered.ci_high - clustered.ci_low) > 3 * (iid.ci_high - iid.ci_low)


def test_holm_hand():
    np.testing.assert_allclose(M.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(M.holm([0.5, 0.6]), [1.0, 1.0])
