"""Resampling to a configurable time unit ("day" or "hour") with availability semantics.

Unit index ``u``: the day index for "day", ``day * 24 + hour`` for "hour". Origin ``T`` means
"end of unit T": every value whose availability time is <= end of unit T may be used.

Availability (see CLAUDE.md): hourly sales/flags are available at the end of their hour; every
daily field (weather, discount, activity, daily totals) at the end of its day; the calendar
(weekday, hour of day, holiday_flag) is treated as known in advance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from sctwin.panel import DAY_HOURS, WEATHER, Panel, availability_classes

UNITS = ("day", "hour")
TIMEZONE = "Asia/Shanghai"  # assumed store-local time (UTC+8, no DST); see CLAUDE.md


@dataclass
class UnitSeries:
    unit: str
    sales: np.ndarray  # (S, N) sales per unit, normalized
    stockout: np.ndarray  # (S, N) hour: 0/1 flag; day: stockout hours in 6..21
    instock_next_known: np.ndarray  # (S, N) bool: unit fully available (needs unit u+1 for hour)
    day_of_unit: np.ndarray  # (N,) day index of each unit
    hour_of_unit: np.ndarray  # (N,) hour of day (0 for "day")
    dates: list[date]  # (D,) calendar days
    holiday: np.ndarray  # (D,) calendar, known in advance
    weather: np.ndarray  # (S, D, 4) available at end of its day
    discount: np.ndarray  # (S, D) available at end of its day
    activity: np.ndarray  # (S, D) available at end of its day
    units_per_day: int
    # hour only: the in-stock indicator of unit u depends on unit u + 1 (lookahead of 1 unit)
    instock_lookahead: int
    # hour only, EVALUATION ONLY (uses neighbouring hours): availability class of each hour
    eval_class: np.ndarray | None = None

    @property
    def n_units(self) -> int:
        return self.sales.shape[1]

    def last_complete_day(self, origin: np.ndarray) -> np.ndarray:
        """Index of the last day fully observed at the end of unit ``origin`` (-1 if none)."""
        o = np.asarray(origin)
        return (o + 1) // self.units_per_day - 1


def resample(p: Panel, unit: str) -> UnitSeries:
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    n_series, n_days = p.shape
    a, b = DAY_HOURS
    common = dict(
        dates=list(p.dates),
        holiday=p.holiday.astype(np.float64),
        weather=p.weather,
        discount=p.discount,
        activity=p.activity.astype(np.float64),
    )
    if unit == "day":
        so = p.stock[:, :, a:b].sum(axis=-1).astype(np.float64)
        return UnitSeries(
            unit="day",
            sales=p.sales.sum(axis=-1),
            stockout=so,
            instock_next_known=so == 0,  # a complete day with no stockout hour in 6..21
            day_of_unit=np.arange(n_days),
            hour_of_unit=np.zeros(n_days, dtype=np.int64),
            units_per_day=1,
            instock_lookahead=0,
            **common,  # type: ignore[arg-type]
        )
    sales = p.sales.reshape(n_series, n_days * 24)
    flag = p.stock.reshape(n_series, n_days * 24).astype(np.float64)
    nxt = np.ones_like(flag)  # beyond the data: unknown -> treated as not in stock
    nxt[:, :-1] = flag[:, 1:]
    return UnitSeries(
        unit="hour",
        sales=sales,
        stockout=flag,
        instock_next_known=(flag == 0) & (nxt == 0),  # class A: in stock now and next hour
        day_of_unit=np.repeat(np.arange(n_days), 24),
        hour_of_unit=np.tile(np.arange(24), n_days),
        units_per_day=24,
        instock_lookahead=1,
        eval_class=availability_classes(p.stock).reshape(n_series, n_days * 24),
        **common,  # type: ignore[arg-type]
    )


def weather_names() -> tuple[str, ...]:
    return WEATHER
