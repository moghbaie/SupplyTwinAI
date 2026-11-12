"""Leakage tests: features for day t must not change when anything not allowed at the forecast
origin (end of day t-1) is changed. Allowed for day t: calendar, and planned promotions when
planned_promo=True. Everything from day t onward (sales, stock flags, weather) is forbidden."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from sctwin.features import (
    daily_feature_names,
    daily_features,
    hourly_feature_names,
    hourly_features,
)
from sctwin.panel import HIERARCHY, Panel


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


def perturb_after(p: Panel, t0: int, seed: int, planned_promo: bool) -> Panel:
    """Change everything that must not be visible to rows t <= t0."""
    rng = np.random.default_rng(seed + 1)
    q = Panel(p.keys, p.dates, p.sales.copy(), p.stock.copy(), p.discount.copy(),
              p.activity.copy(), p.holiday.copy(), p.weather.copy())  # fmt: skip
    q.sales[:, t0:] = rng.gamma(2.0, 5.0, q.sales[:, t0:].shape)
    q.stock[:, t0:] = 1 - q.stock[:, t0:]
    q.weather[:, t0:] += 100.0
    known_from = t0 + 1 if planned_promo else t0  # day-t0 promos are known ahead only if planned
    q.discount[:, known_from:] = 0.5
    q.activity[:, known_from:] = 1 - q.activity[:, known_from:]
    q.holiday[t0 + 1 :] = 1 - q.holiday[t0 + 1 :]
    return q


def _rows(df: pl.DataFrame, t0: int, cols: tuple[str, ...]) -> np.ndarray:
    return df.filter(pl.col("t_idx") <= t0).sort(["series_idx", "t_idx"]).select(cols).to_numpy()


@settings(max_examples=40, deadline=None)
@given(
    n_series=st.integers(1, 4),
    n_days=st.integers(2, 45),
    seed=st.integers(0, 10_000),
    t0_frac=st.floats(0, 1),
    planned_promo=st.booleans(),
)
def test_daily_features_ignore_future(n_series, n_days, seed, t0_frac, planned_promo):
    p = random_panel(n_series, n_days, seed)
    t0 = 1 + int(t0_frac * (n_days - 2))
    q = perturb_after(p, t0, seed, planned_promo)
    cols = daily_feature_names(planned_promo)
    a = _rows(daily_features(p, planned_promo=planned_promo), t0, cols)
    b = _rows(daily_features(q, planned_promo=planned_promo), t0, cols)
    np.testing.assert_array_equal(a, b)


@settings(max_examples=25, deadline=None)
@given(
    n_days=st.integers(2, 40),
    seed=st.integers(0, 10_000),
    t0_frac=st.floats(0, 1),
    planned_promo=st.booleans(),
)
def test_hourly_features_ignore_future(n_days, seed, t0_frac, planned_promo):
    p = random_panel(2, n_days, seed)
    t0 = 1 + int(t0_frac * (n_days - 2))
    q = perturb_after(p, t0, seed, planned_promo)
    cols = hourly_feature_names(planned_promo)
    a = _rows(hourly_features(p, planned_promo=planned_promo), t0, cols)
    b = _rows(hourly_features(q, planned_promo=planned_promo), t0, cols)
    np.testing.assert_array_equal(a, b)


def test_perturbation_is_not_vacuous():
    """Changing day t0-1 (visible at origin t0) must change row t0, else the test above
    would pass trivially."""
    p = random_panel(2, 30, 0)
    t0 = 20
    q = Panel(p.keys, p.dates, p.sales.copy(), p.stock.copy(), p.discount, p.activity,
              p.holiday, p.weather.copy())  # fmt: skip
    q.sales[:, t0 - 1] += 1.0
    q.stock[:, t0 - 1] = 1 - q.stock[:, t0 - 1]
    q.weather[:, t0 - 1] += 1.0
    cols = daily_feature_names()
    a = daily_features(p).filter(pl.col("t_idx") == t0).select(cols).to_numpy()
    b = daily_features(q).filter(pl.col("t_idx") == t0).select(cols).to_numpy()
    changed = {
        c for j, c in enumerate(cols) if not np.array_equal(a[:, j], b[:, j], equal_nan=True)
    }
    assert {"sale_lag1", "sale_mean7", "stockout_hours_mean7", "precpt_lag1"} <= changed


def test_truncated_history_gives_same_features():
    """Features for t <= t0 computed on data truncated after day t0 equal those on full data."""
    p = random_panel(3, 40, 1)
    t0 = 25
    trunc = Panel(p.keys, p.dates[: t0 + 1], p.sales[:, : t0 + 1], p.stock[:, : t0 + 1],
                  p.discount[:, : t0 + 1], p.activity[:, : t0 + 1], p.holiday[: t0 + 1],
                  p.weather[:, : t0 + 1])  # fmt: skip
    cols = hourly_feature_names()
    np.testing.assert_array_equal(
        _rows(hourly_features(p), t0, cols), _rows(hourly_features(trunc), t0, cols)
    )
