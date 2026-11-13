"""Step 5 demand model: likelihood, samplers, AR(1), data, reconstruction (point-in-time)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sctwin import demand_model as dm
from sctwin.panel import CLS_A, CLS_P_FIRST, CLS_P_PRE, CLS_U

PAR = {"alpha": -1.0, "sigma_y": 0.7, "k0": 0.3, "k1": 0.9}


def _p_pos(eta: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(PAR["k0"] + PAR["k1"] * (eta - PAR["alpha"]))))


def _ll(y, code, eta):
    return np.asarray(dm.loglik(np.asarray(y), np.asarray(code), np.asarray(eta), PAR["alpha"],
                                PAR["sigma_y"], PAR["k0"], PAR["k1"]))  # fmt: skip


def test_loglik_matches_scipy():
    eta = np.array([-2.0, -0.5, 0.3, 0.3, 1.0])
    y = np.array([0.0, 0.4, 1.7, 0.0, 3.0])
    p = _p_pos(eta)
    m = np.asarray(dm.hurdle_terms(eta, PAR["alpha"], PAR["sigma_y"], PAR["k0"], PAR["k1"],
                                   xp=np)[2])  # fmt: skip
    ln = stats.lognorm(s=PAR["sigma_y"], scale=np.exp(m))
    exact = np.where(y > 0, np.log(p) + ln.logpdf(np.where(y > 0, y, 1.0)), np.log1p(-p))
    np.testing.assert_allclose(_ll(y, np.zeros(5, int), eta), exact, rtol=1e-10)
    cens = np.where(y > 0, np.log(p) + ln.logsf(np.where(y > 0, y, 1.0)), 0.0)
    np.testing.assert_allclose(_ll(y, np.ones(5, int), eta), cens, rtol=1e-10)
    np.testing.assert_array_equal(_ll(y, np.full(5, 2), eta), 0.0)


def test_mean_is_exp_eta():
    rng = np.random.default_rng(0)
    eta = np.full(400_000, 0.2)
    d = dm.sample_hurdle(eta, PAR["alpha"], PAR["sigma_y"], PAR["k0"], PAR["k1"], rng)
    assert d.mean() == pytest.approx(np.exp(0.2), rel=0.01)
    assert (d == 0).mean() == pytest.approx(1 - _p_pos(np.array([0.2]))[0], abs=0.005)


def test_truncated_sampler_matches_conditional_distribution():
    rng = np.random.default_rng(1)
    eta, lower = np.full(50_000, -0.5), np.full(50_000, 1.5)  # far in the right tail
    x = dm.sample_truncated(eta, lower, PAR["alpha"], PAR["sigma_y"], PAR["k0"], PAR["k1"], rng)
    assert (x >= 1.5).all()
    m = float(dm.hurdle_terms(-0.5, PAR["alpha"], PAR["sigma_y"], PAR["k0"], PAR["k1"], xp=np)[2])
    ln = stats.lognorm(s=PAR["sigma_y"], scale=np.exp(m))
    cdf = lambda v: (ln.cdf(v) - ln.cdf(1.5)) / ln.sf(1.5)  # noqa: E731
    assert stats.kstest(x, cdf).pvalue > 0.01


def test_ladder_treatments():
    m0, m1, m2 = (dm.LADDER[k].codes() for k in ("M0", "M1", "M2"))
    assert m0[CLS_A] == m0[CLS_P_PRE] == dm.EXACT and m0[CLS_P_FIRST] == m0[CLS_U] == dm.DROP
    assert m1[CLS_P_PRE] == m1[CLS_P_FIRST] == dm.CENSORED and m1[CLS_U] == dm.DROP
    assert (m1 == m2).all() and dm.LADDER["M2"].dynamic and not dm.LADDER["M1"].dynamic


def test_ar1_matches_loop():
    rng = np.random.default_rng(2)
    raw = rng.standard_normal((3, 9))
    got = np.asarray(dm.ar1(raw, 0.6, 0.4))
    want = np.empty_like(raw)
    want[:, 0] = raw[:, 0] * 0.4 / np.sqrt(1 - 0.36)
    for t in range(1, 9):
        want[:, t] = 0.6 * want[:, t - 1] + 0.4 * raw[:, t]
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_make_model_data():
    rng = np.random.default_rng(3)
    sales = rng.gamma(1, 1, (4, 10, 24))
    stock = np.zeros((4, 10, 24), np.int8)
    stock[1, 2, 15:] = 1
    disc = np.full((4, 10), 0.9)
    disc[0, 0] = -1.0  # anomaly -> 0
    data = dm.make_model_data(sales, stock, disc, np.zeros((4, 10)), np.arange(10) % 7,
                              np.array([0, 1]), np.arange(0, 5))  # fmt: skip
    assert data.y.shape == (2, 5, dm.NH) and data.x_disc[0, 0] == 0.0
    assert data.x_disc[1, 1] == pytest.approx(0.1)
    assert data.cls[1, 2, 15 - dm.LO - 1] == CLS_P_PRE and data.cls[1, 2, 15 - dm.LO] == CLS_P_FIRST
    with pytest.raises(ValueError):
        dm.make_model_data(sales, stock, disc, disc, np.arange(10), np.array([0]), np.array([1, 3]))


# --------------------------------------------------------------------------- reconstruction


def fake_fit(spec: dm.ModelSpec, n_s: int, n_d: int, n_draws: int = 8) -> dm.FitResult:
    """A FitResult whose draws are all equal to known parameters (no MCMC needed)."""
    one = {
        "alpha": -1.0, "a": np.linspace(-0.3, 0.3, n_s), "gamma": np.zeros(dm.NH),
        "g": np.zeros((n_s, dm.NH)), "dow_eff": np.zeros(7), "b_disc": 0.0, "b_act": 0.0,
        "sigma_u": 0.3, "u": np.zeros((n_s, n_d)), "rho": 0.5, **PAR,
    }  # fmt: skip
    samples = {
        k: np.repeat(np.asarray(v, dtype=float)[None], n_draws, axis=0) for k, v in one.items()
    }
    return dm.FitResult(spec, samples, {})


def fitted_data(n_s: int, n_d: int) -> dm.ModelData:
    z = np.zeros((n_s, n_d))
    return dm.ModelData(np.zeros((n_s, n_d, dm.NH)), np.zeros((n_s, n_d, dm.NH), np.int8),
                        np.arange(n_d) % 7, z, z, np.arange(n_s), np.arange(n_d))  # fmt: skip


def rows_for(y: np.ndarray, cls: np.ndarray, day: int) -> dm.DayRows:
    r = y.shape[0]
    return dm.DayRows(np.arange(r) % 2, np.full(r, day), y, cls, np.zeros(r, int),
                      np.zeros(r), np.zeros(r))  # fmt: skip


@pytest.mark.parametrize("model", ["M0", "M1", "M2"])
def test_reconstruction_respects_observations(model):
    spec = dm.LADDER[model]
    res, data = fake_fit(spec, 2, 10), fitted_data(2, 10)
    rng = np.random.default_rng(4)
    y = np.round(rng.gamma(1, 0.5, (6, dm.NH)), 1)
    cls = np.full((6, dm.NH), CLS_A, np.int8)
    cls[:, 9] = CLS_P_PRE
    cls[:, 10] = CLS_P_FIRST
    cls[:, 11:] = CLS_U
    out = dm.reconstruct(res, data, rows_for(y, cls, 12), n_draws=8, seed=1)
    assert out.shape == (6, 8, dm.NH)
    np.testing.assert_array_equal(out[:, :, :9], np.repeat(y[:, None, :9], 8, axis=1))
    assert (out >= y[:, None, :] - 1e-12).all()  # never below what was sold
    if model == "M0":  # depletion hour taken at face value
        np.testing.assert_array_equal(out[:, :, 9], np.repeat(y[:, None, 9], 8, axis=1))


def test_reconstruction_is_point_in_time():
    res, data = fake_fit(dm.LADDER["M1"], 2, 10), fitted_data(2, 10)
    y = np.ones((2, dm.NH))
    cls = np.zeros((2, dm.NH), np.int8)
    with pytest.raises(ValueError, match="after every day"):
        dm.reconstruct(res, data, rows_for(y, cls, 9))
    # same observations -> same reconstruction, whatever the (unseen) demand was
    a = dm.reconstruct(res, data, rows_for(y, cls, 10), n_draws=4, seed=2)
    b = dm.reconstruct(res, data, rows_for(y.copy(), cls.copy(), 10), n_draws=4, seed=2)
    np.testing.assert_array_equal(a, b)


def test_reconstruction_without_information_follows_the_prior():
    """All hours dropped: the day effect is drawn from its prior, so the mean demand per hour
    is E[exp(eta + u)] = exp(eta + sigma_u^2 / 2)."""
    res, data = fake_fit(dm.LADDER["M1"], 2, 10, n_draws=100), fitted_data(2, 10)
    y = np.zeros((400, dm.NH))
    cls = np.full((400, dm.NH), CLS_U, np.int8)
    out = dm.reconstruct(res, data, rows_for(y, cls, 10), n_draws=100, seed=3, chunk=400)
    eta = -1.0 + np.where(np.arange(400) % 2 == 0, -0.3, 0.3)
    want = np.mean(np.exp(eta + 0.3**2 / 2))
    assert out.mean() == pytest.approx(want, rel=0.04)


def test_censoring_raises_the_reconstruction():
    """A high sale in the depletion hour signals a busy day: M1 (censored) reconstructs more
    demand than M0 (which takes that hour at face value and ignores the lower bound)."""
    y = np.full((200, dm.NH), 0.3)
    y[:, 9] = 1.2
    cls = np.full((200, dm.NH), CLS_A, np.int8)
    cls[:, 9], cls[:, 10], cls[:, 11:] = CLS_P_PRE, CLS_P_FIRST, CLS_U
    y[:, 10:] = 0.0
    data = fitted_data(2, 10)
    tot = {
        m: dm.reconstruct(fake_fit(dm.LADDER[m], 2, 10, 50), data, rows_for(y, cls, 10),
                          n_draws=50, seed=5).sum(-1).mean()
        for m in ("M0", "M1")
    }  # fmt: skip
    assert tot["M1"] > tot["M0"]


# --------------------------------------------------------------------------- tiny fit


def test_tiny_fit_runs_and_reports_diagnostics():
    rng = np.random.default_rng(6)
    n_s, n_d = 3, 10
    shp = (n_s, n_d, dm.NH)
    data = dm.ModelData(
        y=np.round(rng.gamma(1.0, 0.3, shp) * (rng.random(shp) < 0.6), 1),
        cls=np.zeros((n_s, n_d, dm.NH), np.int8), dow=np.arange(n_d) % 7,
        x_disc=np.zeros((n_s, n_d)), x_act=np.zeros((n_s, n_d)),
        series=np.arange(n_s), days=np.arange(n_d),
    )  # fmt: skip
    res = dm.fit(data, dm.LADDER["M2"], num_warmup=40, num_samples=20, num_chains=1, seed=0)
    d = res.diagnostics
    assert {"divergences", "max_rhat", "min_ess", "runtime_s", "scalars"} <= set(d)
    assert res.samples["u"].shape == (20, n_s, n_d) and np.isfinite(res.samples["alpha"]).all()
    ll = dm.pointwise_loglik(res, data, max_draws=5)
    assert ll.shape[0] == 5 and np.isfinite(ll).all()
    ppc = dm.predictive_checks(res, data, n_draws=5)
    assert 0 <= ppc["zero_frac"]["ppc_p"] <= 1
