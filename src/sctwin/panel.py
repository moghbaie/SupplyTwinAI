"""Dense panel representation: series x day x hour.

Valid because every series shares one complete calendar (Step 1 audit). ``build_panel`` checks
that the input is a full rectangle and raises otherwise, so the reshape cannot silently mix
series or dates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from sctwin.data import KEY, N_HOURS, SERIES_KEY, hourly_matrix

HIERARCHY = (
    "store_id",
    "product_id",
    "city_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
)
WEATHER = ("precpt", "avg_temperature", "avg_humidity", "avg_wind_level")
DAY_HOURS = (6, 22)  # hours 6..21: the window covered by stock_hour6_22_cnt

# Hourly availability classes (docs/data_model.md section 4.3)
CLS_A, CLS_P_PRE, CLS_P_FIRST, CLS_U, CLS_NIGHT = 0, 1, 2, 3, 4


@dataclass
class Panel:
    keys: pl.DataFrame  # (S,) hierarchy columns, row i <-> series i
    dates: list[date]  # (T,)
    sales: np.ndarray  # (S, T, 24) float64, normalized units
    stock: np.ndarray  # (S, T, 24) int8, 1 = out of stock
    discount: np.ndarray  # (S, T) float64
    activity: np.ndarray  # (S, T) int8
    holiday: np.ndarray  # (T,) int8
    weather: np.ndarray  # (S, T, 4) float64, order = WEATHER
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.sales.shape[0], self.sales.shape[1]

    def subset(self, idx: np.ndarray | list[int]) -> Panel:
        i = np.asarray(idx)
        return Panel(
            keys=self.keys[i.tolist()],
            dates=list(self.dates),
            sales=self.sales[i],
            stock=self.stock[i],
            discount=self.discount[i],
            activity=self.activity[i],
            holiday=self.holiday.copy(),
            weather=self.weather[i],
            meta=dict(self.meta),
        )


def build_panel(df: pl.DataFrame) -> Panel:
    if df.select(pl.struct(*KEY).is_duplicated().any()).item():
        raise ValueError("duplicate (store_id, product_id, dt) keys")
    d = df.sort([*SERIES_KEY, "dt"])
    keys = d.select(HIERARCHY).unique(subset=list(SERIES_KEY), maintain_order=True)
    dates = sorted(d["dt"].str.to_date().unique().to_list())
    n_series, n_days = keys.height, len(dates)
    if d.height != n_series * n_days:
        raise ValueError(f"not a full rectangle: {d.height} rows != {n_series} x {n_days}")

    def grid(col: str, dtype: type) -> np.ndarray:
        return d[col].to_numpy().astype(dtype).reshape(n_series, n_days)

    hol = grid("holiday_flag", np.int8)
    if not (hol == hol[:1]).all():
        raise ValueError("holiday_flag differs across series on the same date")
    return Panel(
        keys=keys,
        dates=dates,
        sales=hourly_matrix(d, "hours_sale").reshape(n_series, n_days, N_HOURS),
        stock=hourly_matrix(d, "hours_stock_status", np.int8).reshape(n_series, n_days, N_HOURS),
        discount=grid("discount", np.float64),
        activity=grid("activity_flag", np.int8),
        holiday=hol[0].copy(),
        weather=np.stack([grid(w, np.float64) for w in WEATHER], axis=-1),
    )


def availability_classes(stock: np.ndarray) -> np.ndarray:
    """Per-hour class codes from a (..., 24) 0/1 out-of-stock array (within-day neighbours).

    A: in stock, next hour in stock          P_PRE: in stock, next hour out (depletion hour)
    P_FIRST: out, previous hour in stock     U: out, previous hour out
    NIGHT: hours outside 6..21 regardless of flags.
    """
    st = stock.astype(bool)
    nxt = np.empty_like(st)
    nxt[..., :-1] = st[..., 1:]
    nxt[..., -1] = st[..., -1]
    prv = np.zeros_like(st)
    prv[..., 1:] = st[..., :-1]
    cls = np.where(
        st,
        np.where(prv, CLS_U, CLS_P_FIRST),
        np.where(nxt, CLS_P_PRE, CLS_A),
    ).astype(np.int8)
    hours = np.arange(N_HOURS)
    night = (hours < DAY_HOURS[0]) | (hours >= DAY_HOURS[1])
    cls[..., night] = CLS_NIGHT
    return cls
