"""Synthetic FreshRetailNet-shaped fixtures. No network or real data needed."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

QUANTUM = 0.0123  # synthetic "global normalization coefficient"


def make_frame(
    n_stores: int = 4,
    n_products: int = 5,
    n_days: int = 10,
    start: date = date(2024, 3, 1),
    seed: int = 0,
) -> pl.DataFrame:
    """Daily rows with 24-hour arrays. Stockouts start at a random hour and persist to 23,
    so every series has known, internally consistent stockout counts."""
    rng = np.random.default_rng(seed)
    rows = []
    profile = np.r_[np.zeros(6), np.ones(16), np.zeros(2)] + 0.05
    for s in range(n_stores):
        for p in range(n_products):
            for d in range(n_days):
                units = rng.poisson(3 * profile)
                status = np.zeros(24, dtype=np.int64)
                if rng.random() < 0.3:
                    status[rng.integers(8, 24) :] = 1
                    units = np.where(status == 1, 0, units)
                hs = units * QUANTUM
                rows.append(
                    {
                        "city_id": s % 2,
                        "store_id": 100 + s,
                        "management_group_id": p % 2,
                        "first_category_id": p,
                        "second_category_id": p,
                        "third_category_id": p,
                        "product_id": 500 + p,
                        "dt": (start + timedelta(days=d)).isoformat(),
                        "sale_amount": float(hs.sum()),
                        "hours_sale": hs.tolist(),
                        "stock_hour6_22_cnt": int(status[6:22].sum()),
                        "hours_stock_status": status.tolist(),
                        "discount": float(rng.choice([1.0, 0.9])),
                        "holiday_flag": int(d % 7 == 0),
                        "activity_flag": int(rng.random() < 0.2),
                        "precpt": float(d % 3),
                        "avg_temperature": 20.0 + d % 5,
                        "avg_humidity": 60.0,
                        "avg_wind_level": 2.0,
                    }
                )
    return pl.DataFrame(rows).with_columns(pl.col("stock_hour6_22_cnt").cast(pl.Int32))


@pytest.fixture
def frame() -> pl.DataFrame:
    return make_frame()


@pytest.fixture
def train_test() -> tuple[pl.DataFrame, pl.DataFrame]:
    tr = make_frame(n_days=20, seed=1)
    te = make_frame(n_days=7, start=date(2024, 3, 21), seed=2)
    return tr, te
