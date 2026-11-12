"""Point-in-time features at a forecast origin T, for a configurable unit ("day" / "hour").

Follows CLAUDE.md Rule 2:

* every feature is computed as of the origin T (end of unit T), never as of the target time;
* "lag k" = value of unit T - k; rolling windows cover units [T - w + 1, T] (backward only);
* daily fields (weather, discount, activity) come from the last *complete* day at T;
* an in-stock indicator that needs unit u + 1 is only used when u + 1 <= T;
* target-time inputs are restricted to the calendar (steps ahead, weekday, hour, holiday).

Rows are (series, origin T, step h) with target unit u = T + L + h, h = 1..H.
Outcome columns start with ``y`` and are never features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from sctwin.panel import HIERARCHY
from sctwin.timeunit import UnitSeries, weather_names

# --------------------------------------------------------------------------- primitives


def trailing_mean(x: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean over units [j-k, j-1] along axis 1 for each index j (strictly earlier units),
    optionally restricted to ``mask``. NaN where no qualifying unit exists."""
    x = np.asarray(x, dtype=np.float64)
    m = np.ones(x.shape, dtype=np.float64) if mask is None else mask.astype(np.float64)
    zeros = np.zeros((x.shape[0], 1, *x.shape[2:]))
    cs = np.concatenate([zeros, np.cumsum(x * m, axis=1)], axis=1)
    cn = np.concatenate([zeros, np.cumsum(m, axis=1)], axis=1)
    j = np.arange(x.shape[1])
    lo = np.maximum(j - k, 0)
    s = cs[:, j] - cs[:, lo]
    n = cn[:, j] - cn[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def units_since(event: np.ndarray) -> np.ndarray:
    """For each j: j - (last unit < j with event). NaN if none."""
    n_series, n_units = event.shape
    out = np.full((n_series, n_units), np.nan)
    last = np.full(n_series, np.nan)
    for j in range(n_units):
        out[:, j] = j - last
        last = np.where(event[:, j], j, last)
    return out


def _pad(x: np.ndarray) -> np.ndarray:
    """Append one dummy unit so index N ("information up to the end of unit N-1") exists.
    Strictly-earlier statistics never read it."""
    return np.concatenate([x, np.zeros((x.shape[0], 1, *x.shape[2:]), dtype=x.dtype)], axis=1)


def nanmean_rows(v: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, axis=1)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple[int, ...]  # sale_lag{k} = sales of unit T - k
    windows: tuple[int, ...]  # rolling means over the last w units
    season: int  # main seasonal period in units
    n_seasons: int  # seasonal history length (in periods) for seasonal statistics
    extra_seasons: tuple[int, ...]  # additional seasonal "last value" periods
    stockout_windows: tuple[int, ...]
    cov_days: int = 7  # rolling window for daily covariates (in days)

    @staticmethod
    def default(unit: str) -> FeatureConfig:
        if unit == "day":
            return FeatureConfig(lags=(0, 1, 6, 13), windows=(7, 14, 28), season=7, n_seasons=8,
                                 extra_seasons=(), stockout_windows=(7, 28))  # fmt: skip
        return FeatureConfig(
            lags=(0, 1, 2, 23, 167),
            windows=(24, 168, 672),
            season=24,
            n_seasons=28,
            extra_seasons=(168,),
            stockout_windows=(24, 168),
        )


# --------------------------------------------------------------------------- builder


class FeatureBuilder:
    """Builds as-of-T feature rows from a :class:`UnitSeries`.

    Internal convention: ``info[name][:, j]`` uses only units < j, so at origin T the value is
    column ``j = T + 1``.
    """

    def __init__(self, us: UnitSeries, keys: pl.DataFrame, cfg: FeatureConfig | None = None):
        self.us = us
        self.keys = keys
        self.cfg = cfg or FeatureConfig.default(us.unit)
        c = self.cfg
        x, so = _pad(us.sales), _pad(us.stockout)
        info: dict[str, np.ndarray] = {}
        for w in c.windows:
            info[f"sale_mean{w}"] = trailing_mean(x, w)
            info[f"frac_zero{w}"] = trailing_mean(x == 0, w)
        for w in c.stockout_windows:
            info[f"stockout_frac{w}"] = trailing_mean(so > 0, w)
            info[f"stockout_mean{w}"] = trailing_mean(so, w)
        info["units_since_stockout"] = units_since(so > 0)
        # In-stock mean: the indicator of unit u needs unit u + lookahead, so at origin T only
        # units u <= T - lookahead qualify: shift the information index by the lookahead.
        la = us.instock_lookahead
        w = max(c.windows)
        im = trailing_mean(x, w, mask=_pad(us.instock_next_known))
        if la:
            im = np.concatenate([np.full((im.shape[0], la), np.nan), im[:, :-la]], axis=1)
        info[f"instock_mean{w}"] = im
        self.info = info
        # daily covariates, indexed by day; at origin T use the last complete day
        bad = (us.discount <= 0) | (us.discount > 1)
        self.discount = np.where(bad, np.nan, us.discount)
        self.discount_bad = bad.astype(np.float64)
        self.cov_mean = {
            "discount_mean7d": trailing_mean(
                _pad(np.nan_to_num(self.discount, nan=1.0)), c.cov_days
            ),
            "activity_mean7d": trailing_mean(_pad(us.activity), c.cov_days),
        }
        self.dow = np.array([d.weekday() for d in us.dates], dtype=np.float64)

    # ------------------------------------------------------------------ names

    @property
    def feature_names(self) -> tuple[str, ...]:
        c = self.cfg
        names = ["steps_ahead", "target_dow", "target_holiday"]
        if self.us.unit == "hour":
            names.append("target_hour")
        names += [f"sale_lag{k}" for k in c.lags]
        names += [f"sale_mean{w}" for w in c.windows] + [f"frac_zero{w}" for w in c.windows]
        names += [f"stockout_frac{w}" for w in c.stockout_windows]
        names += [f"stockout_mean{w}" for w in c.stockout_windows]
        names += ["units_since_stockout", f"instock_mean{max(c.windows)}"]
        names += ["sale_seasonal_last", "sale_seasonal_instock_mean"]
        names += [f"sale_seasonal{p}_last" for p in c.extra_seasons]
        names += [f"{w}_lastday" for w in weather_names()]
        names += ["discount_lastday", "discount_anomaly_lastday", "activity_lastday"]
        names += list(self.cov_mean)
        return tuple(names)

    # ------------------------------------------------------------------ seasonal gathers

    @staticmethod
    def seasonal_index(origin: np.ndarray, target: np.ndarray, period: int, n: int) -> np.ndarray:
        """(rows, n) unit indices of the n most recent units at the target's seasonal position
        that are observed at the origin (index <= origin); -1 where not available."""
        k0 = np.ceil((target - origin) / period).astype(np.int64)  # smallest k: u - kP <= T
        idx = (target - period * k0)[:, None] - period * np.arange(n)[None, :]
        return np.where(idx >= 0, idx, -1)

    def seasonal_values(
        self,
        s: np.ndarray,
        origin: np.ndarray,
        target: np.ndarray,
        period: int,
        n: int,
        instock_only: bool = False,
    ) -> np.ndarray:
        idx = self.seasonal_index(origin, target, period, n)
        ok = idx >= 0
        safe = np.maximum(idx, 0)
        vals = np.where(ok, self.us.sales[s[:, None], safe], np.nan)
        if instock_only:
            known = idx + self.us.instock_lookahead <= origin[:, None]
            good = self.us.instock_next_known[s[:, None], safe]
            vals = np.where(ok & known & good, vals, np.nan)
        return vals

    # ------------------------------------------------------------------ rows

    def rows(
        self,
        origins: np.ndarray | list[int],
        lead: int,
        horizon: int,
        series_idx: np.ndarray | None = None,
        max_target: int | None = None,
    ) -> pl.DataFrame:
        us, c = self.us, self.cfg
        n_series, n_units = us.sales.shape
        s_all = np.arange(n_series) if series_idx is None else np.asarray(series_idx)
        sg, og, hg = np.meshgrid(s_all, np.asarray(origins, dtype=np.int64),
                                 np.arange(1, horizon + 1), indexing="ij")  # fmt: skip
        s, o, h = sg.ravel(), og.ravel(), hg.ravel()
        u = o + lead + h
        limit = n_units - 1 if max_target is None else min(max_target, n_units - 1)
        keep = (o >= 0) & (u <= limit)
        s, o, h, u = s[keep], o[keep], h[keep], u[keep]
        j = o + 1
        tday = us.day_of_unit[u]
        cols: dict[str, Any] = {
            "series_idx": s.astype(np.int32),
            "origin_idx": o.astype(np.int32),
            "target_idx": u.astype(np.int32),
            "step": h.astype(np.int16),
            "target_dt": np.asarray(us.dates, dtype="datetime64[D]")[tday],
        }
        for col in HIERARCHY:
            cols[col] = self.keys[col].to_numpy()[s]
        # target-time calendar (known in advance)
        cols["steps_ahead"] = (lead + h).astype(np.float64)
        cols["target_dow"] = self.dow[tday]
        cols["target_holiday"] = us.holiday[tday]
        if us.unit == "hour":
            cols["target_hour"] = us.hour_of_unit[u].astype(np.float64)
        # history up to and including unit T
        for k in c.lags:
            src = o - k
            cols[f"sale_lag{k}"] = np.where(src >= 0, us.sales[s, np.maximum(src, 0)], np.nan)
        for name, arr in self.info.items():
            cols[name] = arr[s, j]
        cols["sale_seasonal_last"] = self.seasonal_values(s, o, u, c.season, 1)[:, 0]
        cols["sale_seasonal_instock_mean"] = nanmean_rows(
            self.seasonal_values(s, o, u, c.season, c.n_seasons, instock_only=True)
        )
        for p in c.extra_seasons:
            cols[f"sale_seasonal{p}_last"] = self.seasonal_values(s, o, u, p, 1)[:, 0]
        # daily covariates from the last complete day at T
        d = us.last_complete_day(o)
        dd = np.maximum(d, 0)
        has = d >= 0
        for k, w in enumerate(weather_names()):
            cols[f"{w}_lastday"] = np.where(has, us.weather[s, dd, k], np.nan)
        cols["discount_lastday"] = np.where(has, self.discount[s, dd], np.nan)
        cols["discount_anomaly_lastday"] = np.where(has, self.discount_bad[s, dd], np.nan)
        cols["activity_lastday"] = np.where(has, us.activity[s, dd], np.nan)
        for name, arr in self.cov_mean.items():
            cols[name] = arr[s, d + 1]  # days <= d (d = -1 -> index 0 -> NaN)
        # outcomes (never features)
        cols["y"] = us.sales[s, u]
        cols["y_stockout"] = us.stockout[s, u]
        cols["y_day"] = tday.astype(np.int32)
        return pl.DataFrame(cols)
