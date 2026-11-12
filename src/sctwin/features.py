"""Point-in-time ("as-of") feature construction.

Forecast setup: the origin is the end of day t-1; targets are day t (daily total and each hour).
A feature row for day t may use:

* any observation from days < t (sales, stock flags, weather, discounts, ...)
* "known-ahead" fields of day t: the calendar (weekday, holiday_flag) and, only when
  ``planned_promo=True``, discount and activity_flag (assumed planned in advance; ablated).

It must never use day-t sales, stock flags or weather. ``tests/test_no_leakage.py`` enforces this
by perturbing everything not allowed and checking the features do not change.

Outcome columns are prefixed ``y_`` and are never in ``*_FEATURES``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from sctwin.panel import CLS_A, DAY_HOURS, HIERARCHY, WEATHER, Panel, availability_classes

ID_COLUMNS = ("series_idx", "t_idx", "dt", *HIERARCHY)
CALENDAR_FEATURES = ("dow", "holiday")
PROMO_FEATURES = ("discount", "discount_anomaly", "activity")
HISTORY_FEATURES = (
    "sale_lag1",
    "sale_lag7",
    "sale_lag14",
    "sale_mean7",
    "sale_mean14",
    "sale_mean28",
    "sale_instock_day_mean28",
    "frac_zero_days28",
    "stockout_hours_mean7",
    "stockout_day_frac7",
    "days_since_stockout",
    *(f"{w}_lag1" for w in WEATHER),
)
HOURLY_EXTRA_FEATURES = (
    "hour",
    "hsale_lag1",
    "hsale_lag7",
    "hstock_lag1",
    "hprofile_a28",
    "hprofile_share",
)


def daily_feature_names(planned_promo: bool = True) -> tuple[str, ...]:
    return (*CALENDAR_FEATURES, *(PROMO_FEATURES if planned_promo else ()), *HISTORY_FEATURES)


def hourly_feature_names(planned_promo: bool = True) -> tuple[str, ...]:
    return (*daily_feature_names(planned_promo), *HOURLY_EXTRA_FEATURES)


# --------------------------------------------------------------------------- primitives


def lag(x: np.ndarray, k: int) -> np.ndarray:
    """Value from k days earlier along axis 1 (NaN where unavailable)."""
    out = np.full(x.shape, np.nan)
    if k < x.shape[1]:
        out[:, k:] = x[:, :-k]
    return out


def trailing_mean(x: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean over days [t-k, t-1] along axis 1, optionally restricted to ``mask``.

    Uses strictly earlier days only. NaN where no qualifying day exists.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.ones(x.shape, dtype=np.float64) if mask is None else mask.astype(np.float64)
    zeros = np.zeros((x.shape[0], 1, *x.shape[2:]))
    cs = np.concatenate([zeros, np.cumsum(x * m, axis=1)], axis=1)  # cs[:, t] = sum days < t
    cn = np.concatenate([zeros, np.cumsum(m, axis=1)], axis=1)
    n_days = x.shape[1]
    t = np.arange(n_days)
    lo = np.maximum(t - k, 0)
    s = cs[:, t] - cs[:, lo]
    n = cn[:, t] - cn[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def days_since(event: np.ndarray) -> np.ndarray:
    """For each t: t - (last day < t with event). NaN if none."""
    n_series, n_days = event.shape
    out = np.full((n_series, n_days), np.nan)
    last = np.full(n_series, np.nan)
    for t in range(n_days):
        out[:, t] = t - last
        last = np.where(event[:, t], t, last)
    return out


# --------------------------------------------------------------------------- daily


def _daily_arrays(p: Panel, planned_promo: bool) -> dict[str, np.ndarray]:
    a, b = DAY_HOURS
    daily = p.sales.sum(axis=-1)
    so_hours = p.stock[:, :, a:b].sum(axis=-1).astype(np.float64)
    n_series, n_days = p.shape
    dow = np.array([d.weekday() for d in p.dates], dtype=np.float64)

    f: dict[str, np.ndarray] = {
        "dow": np.broadcast_to(dow, (n_series, n_days)),
        "holiday": np.broadcast_to(p.holiday.astype(np.float64), (n_series, n_days)),
    }
    if planned_promo:
        bad = (p.discount <= 0) | (p.discount > 1)
        f["discount"] = np.where(bad, np.nan, p.discount)
        f["discount_anomaly"] = bad.astype(np.float64)
        f["activity"] = p.activity.astype(np.float64)
    f |= {
        "sale_lag1": lag(daily, 1),
        "sale_lag7": lag(daily, 7),
        "sale_lag14": lag(daily, 14),
        "sale_mean7": trailing_mean(daily, 7),
        "sale_mean14": trailing_mean(daily, 14),
        "sale_mean28": trailing_mean(daily, 28),
        "sale_instock_day_mean28": trailing_mean(daily, 28, mask=so_hours == 0),
        "frac_zero_days28": trailing_mean(daily == 0, 28),
        "stockout_hours_mean7": trailing_mean(so_hours, 7),
        "stockout_day_frac7": trailing_mean(so_hours > 0, 7),
        "days_since_stockout": days_since(so_hours > 0),
    }
    for j, w in enumerate(WEATHER):
        f[f"{w}_lag1"] = lag(p.weather[:, :, j], 1)
    return f


def _outcomes_daily(p: Panel) -> dict[str, np.ndarray]:
    a, b = DAY_HOURS
    return {
        "y_daily": p.sales.sum(axis=-1),
        "y_stockout_hours": p.stock[:, :, a:b].sum(axis=-1).astype(np.float64),
    }


def _ids(p: Panel, s_idx: np.ndarray, t_idx: np.ndarray) -> dict[str, Any]:
    ids: dict[str, Any] = {
        "series_idx": s_idx.astype(np.int32),
        "t_idx": t_idx.astype(np.int16),
        "dt": np.asarray(p.dates)[t_idx],
    }
    for c in HIERARCHY:
        ids[c] = p.keys[c].to_numpy()[s_idx]
    return ids


def daily_features(p: Panel, t_min: int = 1, planned_promo: bool = True) -> pl.DataFrame:
    """One row per (series, day t >= t_min): ids, features, ``y_`` outcomes."""
    f = _daily_arrays(p, planned_promo)
    y = _outcomes_daily(p)
    n_series, n_days = p.shape
    sg, tg = np.meshgrid(np.arange(n_series), np.arange(t_min, n_days), indexing="ij")
    s_idx, t_idx = sg.ravel(), tg.ravel()
    cols = _ids(p, s_idx, t_idx)
    for name in daily_feature_names(planned_promo):
        cols[name] = f[name][s_idx, t_idx]
    for name, arr in y.items():
        cols[name] = arr[s_idx, t_idx]
    return pl.DataFrame(cols)


# --------------------------------------------------------------------------- hourly


def hourly_features(
    p: Panel,
    t_min: int = 1,
    hours: tuple[int, int] = DAY_HOURS,
    planned_promo: bool = True,
) -> pl.DataFrame:
    """One row per (series, day t >= t_min, hour in [hours)): daily features + hourly ones.

    Hourly history uses days < t only (no intra-day updating; that is the particle filter's job).
    """
    f = _daily_arrays(p, planned_promo)
    cls = availability_classes(p.stock)
    prof = trailing_mean(p.sales, 28, mask=cls == CLS_A)  # (S, T, 24)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = prof / np.nansum(prof, axis=-1, keepdims=True)
    hourly = {
        "hsale_lag1": lag(p.sales, 1),
        "hsale_lag7": lag(p.sales, 7),
        "hstock_lag1": lag(p.stock.astype(np.float64), 1),
        "hprofile_a28": prof,
        "hprofile_share": share,
    }
    n_series, n_days = p.shape
    h_lo, h_hi = hours
    sg, tg, hg = np.meshgrid(
        np.arange(n_series), np.arange(t_min, n_days), np.arange(h_lo, h_hi), indexing="ij"
    )
    s_idx, t_idx, h_idx = sg.ravel(), tg.ravel(), hg.ravel()
    cols = _ids(p, s_idx, t_idx)
    cols["hour"] = h_idx.astype(np.int8)
    for name in daily_feature_names(planned_promo):
        cols[name] = f[name][s_idx, t_idx]
    for name, arr in hourly.items():
        cols[name] = arr[s_idx, t_idx, h_idx]
    cols["y_sale"] = p.sales[s_idx, t_idx, h_idx]
    cols["y_stockout"] = p.stock[s_idx, t_idx, h_idx]
    cols["y_class"] = cls[s_idx, t_idx, h_idx]
    return pl.DataFrame(cols)
