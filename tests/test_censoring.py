"""Step 4 censoring benchmark: injection invariants, point-in-time history, reconstruction
baselines, calibration and the fully synthetic generator."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sctwin.censoring import (
    HI,
    IMPUTERS,
    LO,
    CensorConfig,
    SyntheticParams,
    build_benchmark,
    calibrate,
    candidate_days,
    censor_synthetic,
    expected_window_demand,
    hour_profile,
    inject_stockouts,
    position_ratios,
    recovery_frames,
    simulate_latent,
    stock_levels,
    stockout_stats,
    trailing_hour_means,
)
from sctwin.evaluation import evaluate

NO_NOISE = CensorConfig(leak_first=0.0, leak_later=0.0, p_restock=0.0)


def demand_rows(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.gamma(1.0, 1.0, (n, 24)) * (rng.random((n, 24)) < 0.7)
    d[:, :LO] *= 0.1
    return d


# --------------------------------------------------------------------------- injection


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    leak_first=st.floats(0, 1),
    leak_later=st.floats(0, 1),
    p_restock=st.floats(0, 1),
    scale=st.floats(0.05, 3.0),
)
def test_injection_keeps_truth_and_bounds(seed, leak_first, leak_later, p_restock, scale):
    d = demand_rows(50, seed)
    d0 = d.copy()
    cfg = CensorConfig(leak_first=leak_first, leak_later=leak_later, p_restock=p_restock)
    k = scale * d[:, LO:HI].sum(1) + 1e-3
    inj = inject_stockouts(d, k, np.random.default_rng(seed), cfg)
    np.testing.assert_array_equal(d, d0)  # the truth is never modified
    assert (inj.observed <= d + 1e-12).all() and (inj.observed >= 0).all()
    np.testing.assert_array_equal(inj.observed[~inj.censored], d[~inj.censored])
    assert not inj.flag[:, :LO].any() and not inj.flag[:, HI:].any()
    assert not inj.censored[:, :LO].any() and not inj.censored[:, HI:].any()
    assert ((inj.flag == 1) <= inj.censored).all()  # flagged hours are censored


def test_depletion_follows_cumulative_demand():
    d = demand_rows(500, 1)
    rng = np.random.default_rng(1)
    k = rng.uniform(0.1, 1.2, 500) * d[:, LO:HI].sum(1) + 1e-6
    inj = inject_stockouts(d, k, np.random.default_rng(2), NO_NOISE)
    cum = np.cumsum(d[:, LO:HI], 1)
    for i in range(500):
        h = inj.depletion[i]
        if h < 0:
            assert cum[i, -1] < k[i] and not inj.flag[i].any()
            continue
        j = h - LO
        assert cum[i, j] >= k[i] and (j == 0 or cum[i, j - 1] < k[i])
        # everything sold up to and including the depletion hour is exactly the stock
        assert inj.observed[i, LO : h + 1].sum() == pytest.approx(k[i])
        # no restock: flagged from h + 1 to the end of the window, nothing sold
        assert inj.flag[i, h + 1 : HI].all() and not inj.flag[i, LO : h + 1].any()
        assert (inj.observed[i, h + 1 : HI] == 0).all()


def test_leak_and_restock():
    d = np.zeros((1, 24))
    d[0, LO:HI] = 1.0
    cfg = CensorConfig(leak_first=0.5, leak_later=0.25, p_restock=1.0, restock_mean_hours=1.0)
    inj = inject_stockouts(d, np.array([3.5]), np.random.default_rng(0), cfg)
    # cumulative demand reaches 3.5 in the 4th window hour; restock after exactly 1 hour
    h = LO + 3
    assert inj.depletion[0] == h and inj.restock[0] == h + 2
    np.testing.assert_allclose(inj.observed[0, h], 0.5)  # 3.5 - 3 units sold before
    assert inj.flag[0, h + 1] == 1 and inj.observed[0, h + 1] == 0.5  # first flagged hour leaks
    assert inj.flag[0, h + 2 :].sum() == 0 and (inj.observed[0, h + 2 : HI] == 1.0).all()


def test_common_random_numbers_and_determinism():
    d = demand_rows(200, 3)
    e = d[:, LO:HI].sum(1)
    a, b = CensorConfig(coverage=0.8), CensorConfig(coverage=1.2)
    ka = stock_levels(e, np.random.default_rng(5), a)
    kb = stock_levels(e, np.random.default_rng(5), b)
    np.testing.assert_allclose(ka / kb, 0.8 / 1.2)  # same draws, different coverage
    i1 = inject_stockouts(d, ka, np.random.default_rng(5), a)
    i2 = inject_stockouts(d, ka, np.random.default_rng(5), a)
    np.testing.assert_array_equal(i1.observed, i2.observed)
    # more stock never creates more stockouts
    assert (
        inject_stockouts(d, kb, np.random.default_rng(5), b).stocked_out <= i1.stocked_out
    ).all()


def test_config_validation():
    with pytest.raises(ValueError):
        CensorConfig(leak_first=1.5)
    with pytest.raises(ValueError):
        inject_stockouts(np.ones((2, 24)), np.array([1.0, 0.0]), np.random.default_rng(0),
                         CensorConfig())  # fmt: skip


# --------------------------------------------------------------------------- point-in-time history


def panel_arrays(n_series: int, n_days: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sales = rng.gamma(1.0, 1.0, (n_series, n_days, 24))
    stock = (rng.random((n_series, n_days, 24)) < 0.2).astype(np.int8)
    return sales, stock


def test_trailing_hour_means_known_value():
    sales, stock = panel_arrays(2, 10, 0)
    s, d, w = np.array([1]), np.array([7]), 3
    got = trailing_hour_means(sales, stock, s, d, w)[0]
    x, ok = sales[1, 4:7], stock[1, 4:7] == 0
    with np.errstate(invalid="ignore"):
        want = np.where(ok.sum(0) > 0, (x * ok).sum(0) / ok.sum(0), np.nan)
    np.testing.assert_allclose(got, want, equal_nan=True)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), day=st.integers(1, 19), window=st.integers(1, 10))
def test_history_uses_only_days_before_the_censored_day(seed, day, window):
    """Origin T = end of the censored day d. The methods' history must not change when day d
    itself or any later day is replaced (day d enters only through its censored observation)."""
    sales, stock = panel_arrays(3, 20, seed)
    s, d = np.arange(3), np.full(3, day)
    a = trailing_hour_means(sales, stock, s, d, window)
    rng = np.random.default_rng(seed + 1)
    sales2, stock2 = sales.copy(), stock.copy()
    sales2[:, day:] = rng.gamma(3.0, 5.0, sales2[:, day:].shape)
    stock2[:, day:] = 1 - stock2[:, day:]
    np.testing.assert_array_equal(a, trailing_hour_means(sales2, stock2, s, d, window))


def test_reconstruction_ignores_data_after_the_day():
    """End-to-end: replacing all days after d (and the demand of censored hours beyond what is
    observed) leaves every method's reconstruction of day d unchanged."""
    sales, stock = panel_arrays(6, 30, 2)
    stock[:, 20:] = 0
    rows = candidate_days(sales, stock, np.arange(6), np.array([20]), window=14)
    bm = build_benchmark(sales, rows, CensorConfig(), seed=4)
    sales2 = sales.copy()
    sales2[:, 21:] *= 10.0
    stock2 = stock.copy()
    stock2[:, 21:] = 1
    bm2 = build_benchmark(sales2, candidate_days(sales2, stock2, np.arange(6), np.array([20]),
                                                 window=14), CensorConfig(), seed=4)  # fmt: skip
    for name, f in IMPUTERS.items():
        a = f(bm.injection.observed, bm.injection.flag, bm.hour_means)
        b = f(bm2.injection.observed, bm2.injection.flag, bm2.hour_means)
        np.testing.assert_array_equal(a, b, err_msg=name)


# --------------------------------------------------------------------------- reconstruction


def test_imputers_leave_uncensored_days_alone():
    d = demand_rows(20, 5)
    flag = np.zeros((20, 24), dtype=np.int8)
    hm = np.full((20, 24), 0.7)
    for name, f in IMPUTERS.items():
        np.testing.assert_array_equal(f(d, flag, hm), d, err_msg=name)


def test_profile_recovers_exact_profile_demand():
    """Demand that is exactly level x profile is recovered by profile-scaling; ignoring the
    censoring under-estimates it."""
    prof = hour_profile()
    n = 50
    level = np.random.default_rng(0).uniform(0.5, 3.0, n)
    demand = level[:, None] * prof[None, :]
    k = 0.6 * demand[:, LO:HI].sum(1)
    inj = inject_stockouts(demand, k, np.random.default_rng(1), NO_NOISE)
    hm = np.tile(prof, (n, 1))  # history has the same shape (any scale)
    est = IMPUTERS["profile_pre"](inj.observed, inj.flag, hm)
    np.testing.assert_allclose(est, demand, rtol=1e-12)
    obs = IMPUTERS["observed"](inj.observed, inj.flag, hm)
    assert obs[:, LO:HI].sum() < 0.7 * demand[:, LO:HI].sum()
    # without treating the depletion hour as censored, its lower bound biases the level down
    plain = IMPUTERS["profile"](inj.observed, inj.flag, hm)
    assert plain[:, LO:HI].sum() < est[:, LO:HI].sum()


def test_hour_mean_imputer_never_below_observed():
    d = demand_rows(100, 7)
    inj = inject_stockouts(d, 0.5 * d[:, LO:HI].sum(1) + 1e-6, np.random.default_rng(0),
                           CensorConfig())  # fmt: skip
    hm = np.full((100, 24), 0.3)
    for f in IMPUTERS.values():
        assert (f(inj.observed, inj.flag, hm) >= inj.observed).all()


# --------------------------------------------------------------------------- diagnostics


def test_stockout_stats_known_example():
    f = np.zeros((4, 24), dtype=np.int8)
    f[0, 10:HI] = 1  # first flag 10, no restock
    f[1, 14:16] = 1  # first flag 14, restock after 2 hours
    f[2, LO] = 1  # out at hour 6: excluded (not in stock at the start)
    st_ = stockout_stats(f)
    assert st_["n_days"] == 3 and st_["stockout_rate"] == pytest.approx(2 / 3)
    assert st_["first_hour_q50"] == 12 and st_["restock_rate"] == 0.5
    assert st_["restock_mean_hours"] == 2


def test_position_ratios_known_example():
    obs = np.ones((1, 24))
    obs[0, 11] = 3.0
    flag = np.zeros((1, 24), dtype=np.int8)
    flag[0, 12:HI] = 1
    obs[0, 12:HI] = 0.5
    r = position_ratios(obs, flag, np.ones((1, 24)), (-2, -1, 0, 1))
    assert r == {"-2": 1.0, "-1": 3.0, "+0": 0.5, "+1": 0.5}


def test_calibration_recovers_known_coverage():
    d = demand_rows(3000, 11)
    e = d[:, LO:HI].sum(1)
    true = CensorConfig(coverage=0.9, coverage_sd=0.3)
    rng = np.random.default_rng(9)
    target = stockout_stats(inject_stockouts(d, stock_levels(e, rng, true), rng, true).flag)
    cfg, grid = calibrate(d, e, target, CensorConfig(), np.round(np.arange(0.5, 1.31, 0.1), 2),
                          np.array([0.1, 0.3, 0.5]), seed=9)  # fmt: skip
    assert len(grid) == 27
    assert cfg.coverage == pytest.approx(0.9) and cfg.coverage_sd == pytest.approx(0.3)


def test_recovery_frames_and_evaluation():
    sales, stock = panel_arrays(8, 40, 3)
    stock[:, 30:] = 0
    rows = candidate_days(sales, stock, np.arange(8), np.arange(30, 40), window=14)
    bm = build_benchmark(sales, rows, CensorConfig(coverage=0.7), seed=0)
    est = {m: f(bm.injection.observed, bm.injection.flag, bm.hour_means)
           for m, f in IMPUTERS.items()}  # fmt: skip
    hourly, daily = recovery_frames(bm, est)
    assert hourly.height == int(bm.injection.censored.sum())
    assert daily.height == int(bm.injection.stocked_out.sum()) > 0
    np.testing.assert_allclose(hourly["observed__point"], hourly["observed"])
    res = evaluate(daily, (0.5,))
    assert res["overall"]["models"]["observed"]["bias"] < 0  # ignoring censoring under-states
    assert expected_window_demand(bm.hour_means).shape == (bm.series.size,)


# --------------------------------------------------------------------------- synthetic generator


def test_synthetic_generator_moments():
    p = SyntheticParams(level_mean=0.0, level_sd=0.3, rho=0.5, sigma_z=0.3)
    sp = simulate_latent(400, 120, p, seed=0)
    assert sp.demand.shape == sp.mu.shape == (400, 120, 24)
    assert sp.demand.mean() == pytest.approx(sp.mu.mean(), rel=0.03)
    assert sp.z.var() == pytest.approx(p.sigma_z**2 / (1 - p.rho**2), rel=0.1)
    lag1 = np.corrcoef(sp.z[:, 1:].ravel(), sp.z[:, :-1].ravel())[0, 1]
    assert lag1 == pytest.approx(p.rho, abs=0.05)
    assert hour_profile()[LO:HI].mean() == pytest.approx(1.0)
    np.testing.assert_array_equal(simulate_latent(5, 10, p, 3).demand,
                                  simulate_latent(5, 10, p, 3).demand)  # fmt: skip


def test_censor_synthetic_bounds():
    sp = simulate_latent(50, 30, SyntheticParams(), seed=1)
    inj = censor_synthetic(sp, CensorConfig(coverage=0.9), seed=2)
    assert inj.observed.shape == sp.demand.shape
    assert (inj.observed <= sp.demand + 1e-12).all()
    assert 0.1 < inj.stocked_out.mean() < 0.9


def test_leak_calibration_matches_target_ratios():
    from sctwin.censoring import calibrate_leak

    d = demand_rows(3000, 13)
    e = d[:, LO:HI].sum(1)
    base = np.tile(d.mean(0), (3000, 1))
    true = CensorConfig(coverage=0.8, leak_first=0.1, leak_later=0.03)
    rng = np.random.default_rng(4)
    inj = inject_stockouts(d, stock_levels(e, rng, true), rng, true)
    target = position_ratios(inj.observed, inj.flag, base, (0, 1))
    start = dataclasses.replace(true, leak_first=0.4, leak_later=0.2)
    got = calibrate_leak(d, e, base, target, start, seed=4)
    assert got.leak_first == pytest.approx(0.1) and got.leak_later == pytest.approx(0.03)


def test_restock_calibration_recovers_truncated_parameters():
    from sctwin.censoring import calibrate_restock

    d = demand_rows(4000, 17)
    e = d[:, LO:HI].sum(1)
    true = CensorConfig(coverage=0.6, p_restock=0.3, restock_mean_hours=4.0)
    rng = np.random.default_rng(8)
    target = stockout_stats(inject_stockouts(d, stock_levels(e, rng, true), rng, true).flag)
    assert target["restock_mean_hours"] < 4.0  # truncation by the end of the window
    got = calibrate_restock(d, e, target, dataclasses.replace(true, p_restock=0.1),
                            np.array([0.1, 0.2, 0.3, 0.4]), np.array([2.0, 3.0, 4.0, 5.0]),
                            seed=8)  # fmt: skip
    assert got.p_restock == pytest.approx(0.3) and got.restock_mean_hours == pytest.approx(4.0)
