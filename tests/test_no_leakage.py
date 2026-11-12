"""Leakage tests (CLAUDE.md Rules 1 and 2), run for both units "day" and "hour".

``perturb_after`` replaces everything not available at the end of unit T: hourly sales/flags
after T (including later hours of the same day) and the daily fields (weather, discount,
activity) of every day not complete at T. The calendar is untouched (known in advance).
Features at origins <= T, training sets of the fold at T, and (in test_baseline.py) model
predictions must be unchanged.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sctwin.backtest import BacktestSpec, check_split, make_origins
from sctwin.features import FeatureBuilder, FeatureConfig
from sctwin.panel import HIERARCHY, Panel
from sctwin.timeunit import resample

UNITS = ("day", "hour")
SMALL = {
    "day": FeatureConfig(lags=(0, 1, 6), windows=(3, 7), season=7, n_seasons=3,
                         extra_seasons=(), stockout_windows=(3,), cov_days=3),
    "hour": FeatureConfig(lags=(0, 1, 23), windows=(6, 24, 48), season=24, n_seasons=3,
                          extra_seasons=(168,), stockout_windows=(6, 24), cov_days=3),
}  # fmt: skip


def random_panel(n_series: int, n_days: int, seed: int) -> Panel:
    rng = np.random.default_rng(seed)
    keys = pl.DataFrame({c: np.arange(n_series) + 10 * j for j, c in enumerate(HIERARCHY)})
    shape = (n_series, n_days, 24)
    return Panel(
        keys=keys,
        dates=[date(2024, 1, 1) + timedelta(days=d) for d in range(n_days)],
        sales=np.round(rng.gamma(1.0, 0.3, shape) * (rng.random(shape) < 0.5), 1),
        stock=(rng.random(shape) < 0.3).astype(np.int8),
        discount=rng.choice([0.0, 0.8, 1.0, 1.05], (n_series, n_days)),
        activity=(rng.random((n_series, n_days)) < 0.3).astype(np.int8),
        holiday=(rng.random(n_days) < 0.3).astype(np.int8),
        weather=rng.normal(size=(n_series, n_days, 4)),
    )


def perturb_after(p: Panel, unit: str, origin: int, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed + 1)
    n_series, n_days = p.shape
    sales, stock = p.sales.copy(), p.stock.copy()
    first_hour = (origin + 1) * 24 if unit == "day" else origin + 1
    last_day = origin if unit == "day" else (origin + 1) // 24 - 1  # last complete day
    fs, fk = sales.reshape(n_series, -1), stock.reshape(n_series, -1)  # views
    fs[:, first_hour:] = rng.gamma(2.0, 5.0, fs[:, first_hour:].shape)
    fk[:, first_hour:] = 1 - fk[:, first_hour:]
    weather, discount, activity = p.weather.copy(), p.discount.copy(), p.activity.copy()
    weather[:, last_day + 1 :] += 100.0
    discount[:, last_day + 1 :] = 0.5
    activity[:, last_day + 1 :] = 1 - activity[:, last_day + 1 :]
    return Panel(p.keys, p.dates, sales, stock, discount, activity, p.holiday.copy(), weather)


def builder(p: Panel, unit: str, cfg: FeatureConfig | None = None) -> FeatureBuilder:
    return FeatureBuilder(resample(p, unit), p.keys, cfg or SMALL[unit])


def _features(df: pl.DataFrame, fb: FeatureBuilder) -> np.ndarray:
    return df.sort(["series_idx", "origin_idx", "step"]).select(fb.feature_names).to_numpy()


def _case(unit: str, n_days: int, frac: float, seed: int) -> tuple[Panel, int]:
    p = random_panel(2, n_days, seed)
    n_units = n_days * (24 if unit == "hour" else 1)
    return p, int(frac * (n_units - 2))


# --------------------------------------------------------------------------- features


@pytest.mark.parametrize("unit", UNITS)
@settings(max_examples=30, deadline=None)
@given(
    n_days=st.integers(2, 20),
    frac=st.floats(0, 1),
    lead=st.integers(0, 3),
    horizon=st.integers(1, 10),
    seed=st.integers(0, 10_000),
)
def test_features_ignore_data_after_origin(unit, n_days, frac, lead, horizon, seed):
    p, origin = _case(unit, n_days, frac, seed)
    q = perturb_after(p, unit, origin, seed)
    origins = np.arange(origin + 1)  # every origin <= T
    fa, fq = builder(p, unit), builder(q, unit)
    a = fa.rows(origins, lead, horizon)
    b = fq.rows(origins, lead, horizon)
    np.testing.assert_array_equal(_features(a, fa), _features(b, fq))


@pytest.mark.parametrize("unit", UNITS)
@settings(max_examples=30, deadline=None)
@given(
    n_days=st.integers(4, 20),
    frac=st.floats(0.3, 1),
    lead=st.integers(0, 3),
    horizon=st.integers(1, 10),
    window=st.sampled_from(["sliding", "expanding"]),
    seed=st.integers(0, 10_000),
)
def test_training_set_ignores_data_after_origin(unit, n_days, frac, lead, horizon, window, seed):
    """Every training feature AND training target of the fold at T is unchanged when all data
    after T is replaced, and every training target lies at or before T."""
    p, origin = _case(unit, n_days, frac, seed)
    spec = BacktestSpec(unit, lead, horizon, (origin,), window=window, window_length=30)
    q = perturb_after(p, unit, origin, seed)
    fa, fq = builder(p, unit), builder(q, unit)
    ta = fa.rows(spec.train_times(origin), lead, horizon)
    tb = fq.rows(spec.train_times(origin), lead, horizon)
    check_split(ta, fa.rows([origin], lead, horizon), origin, spec)
    if ta.height:
        assert int(ta["target_idx"].max()) <= origin
    cols = [*fa.feature_names, "y", "y_stockout"]
    key = ["series_idx", "origin_idx", "step"]
    np.testing.assert_array_equal(ta.sort(key).select(cols).to_numpy(),
                                  tb.sort(key).select(cols).to_numpy())  # fmt: skip


def test_hour_instock_indicator_does_not_peek_at_next_hour():
    """Regression: the hourly in-stock indicator of hour u uses flag u+1. Changing ONLY the
    flag of hour T+1 must not change any feature at origin T."""
    p = random_panel(2, 10, 5)
    for origin in (30, 100, 191):
        q = Panel(p.keys, p.dates, p.sales, p.stock.copy(), p.discount, p.activity, p.holiday,
                  p.weather)  # fmt: skip
        flat = q.stock.reshape(2, -1)
        flat[:, origin + 1] = 1 - flat[:, origin + 1]
        fa, fq = builder(p, "hour"), builder(q, "hour")
        np.testing.assert_array_equal(
            _features(fa.rows([origin], 1, 24), fa), _features(fq.rows([origin], 1, 24), fq)
        )


@pytest.mark.parametrize("unit", UNITS)
def test_perturbation_is_not_vacuous(unit):
    """Changing unit T itself (available at T) must change the features at origin T."""
    p = random_panel(2, 20, 0)
    origin = 15 if unit == "day" else 15 * 24 + 23  # end of day 15 in both cases
    q = perturb_after(p, unit, origin - 1)
    fa, fq = builder(p, unit), builder(q, unit)
    a = fa.rows([origin], 1, 3).select(fa.feature_names).to_numpy()
    b = fq.rows([origin], 1, 3).select(fq.feature_names).to_numpy()
    changed = {
        c for k, c in enumerate(fa.feature_names)
        if not np.array_equal(a[:, k], b[:, k], equal_nan=True)
    }  # fmt: skip
    w = SMALL[unit].stockout_windows[0]
    assert {"sale_lag0", f"stockout_mean{w}", "precpt_lastday"} <= changed


def test_hour_origin_mid_day_uses_previous_complete_day():
    p = random_panel(1, 5, 1)
    fb = builder(p, "hour")
    for origin, expect_day in ((2 * 24 + 10, 1), (2 * 24 + 23, 2), (5, -1)):
        got = fb.rows([origin], 0, 1)["precpt_lastday"][0]
        if expect_day < 0:
            assert np.isnan(got)
        else:
            assert got == p.weather[0, expect_day, 0]


# --------------------------------------------------------------------------- splits


def test_train_times_full_window_rule():
    spec = BacktestSpec("day", lead=1, horizon=7, origins=(74,), window="sliding",
                        window_length=42, min_history=28)  # fmt: skip
    t = spec.train_times(74)
    assert t.max() == 74 - 1 - 7 and t.min() == max(28, 74 - 8 - 42 + 1)
    assert BacktestSpec("day", 1, 7, (90,), window_length=10).train_times(90).tolist() == list(
        range(73, 83)
    )
    exp = BacktestSpec("hour", 1, 24, (500,), window="expanding", min_history=48,
                       train_stride=6)  # fmt: skip
    t = exp.train_times(500)
    assert t.max() == 475 and t.min() >= 48 and (np.diff(t) == 6).all()


def test_check_split_catches_violations():
    p = random_panel(1, 60, 0)
    fb = builder(p, "day")
    spec = BacktestSpec("day", 1, 7, (40,), window="sliding", window_length=20)
    test = fb.rows([40], 1, 7)
    with pytest.raises(AssertionError, match="t > T - L - H"):
        check_split(fb.rows(np.arange(20, 35), 1, 7), test, 40, spec)
    with pytest.raises(AssertionError, match="partial window"):
        check_split(fb.rows(np.arange(20, 33), 1, 7, max_target=38), test, 40, spec)
    with pytest.raises(AssertionError, match="another origin"):
        check_split(fb.rows(np.arange(20, 33), 1, 7), fb.rows([40, 41], 1, 7), 40, spec)
    check_split(fb.rows(spec.train_times(40), 1, 7), test, 40, spec)


def test_make_origins():
    assert make_origins(76, 89, lead=1, horizon=7, step=7) == (74, 81)
    hour = make_origins(76 * 24, 90 * 24 - 1, lead=1, horizon=24, step=24)
    assert hour[0] == 76 * 24 - 2 and len(hour) == 14
    assert make_origins(76 * 24, 90 * 24 - 1, 1, 24, 24, n_folds=3) == hour[:3]
