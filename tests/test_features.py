import numpy as np
import polars as pl
import pytest

from sctwin.features import (
    daily_feature_names,
    daily_features,
    days_since,
    hourly_feature_names,
    hourly_features,
    lag,
    trailing_mean,
)
from sctwin.panel import (
    CLS_A,
    CLS_NIGHT,
    CLS_P_FIRST,
    CLS_P_PRE,
    CLS_U,
    availability_classes,
    build_panel,
)

# --------------------------------------------------------------------------- primitives


def test_lag():
    x = np.array([[1.0, 2, 3, 4]])
    np.testing.assert_array_equal(lag(x, 1), [[np.nan, 1, 2, 3]])
    np.testing.assert_array_equal(lag(x, 4), [[np.nan] * 4])


def test_trailing_mean_hand():
    x = np.array([[1.0, 2, 3, 4]])
    np.testing.assert_array_equal(trailing_mean(x, 2), [[np.nan, 1, 1.5, 2.5]])
    m = np.array([[True, False, True, True]])
    # day 3 window = days 1,2 -> only day 2 qualifies; day 2 window = days 0,1 -> only day 0
    np.testing.assert_array_equal(trailing_mean(x, 2, mask=m), [[np.nan, 1, 1, 3]])


def test_trailing_mean_extra_axes():
    rng = np.random.default_rng(0)
    x = rng.random((2, 10, 3))
    out = trailing_mean(x, 4)
    np.testing.assert_allclose(out[1, 7, 2], x[1, 3:7, 2].mean())


def test_days_since():
    e = np.array([[0, 1, 0, 0, 1, 0]], dtype=bool)
    np.testing.assert_array_equal(days_since(e), [[np.nan, np.nan, 1, 2, 3, 1]])


def test_availability_classes_hand():
    st = np.zeros(24, dtype=np.int8)
    st[[10, 11, 12, 15]] = 1
    st[3] = 1
    c = availability_classes(st)
    expect = {9: CLS_P_PRE, 10: CLS_P_FIRST, 11: CLS_U, 12: CLS_U, 13: CLS_A, 14: CLS_P_PRE,
              15: CLS_P_FIRST, 16: CLS_A, 3: CLS_NIGHT, 22: CLS_NIGHT, 6: CLS_A}  # fmt: skip
    for h, v in expect.items():
        assert c[h] == v, h


# --------------------------------------------------------------------------- panel


def test_build_panel_roundtrip(frame):
    p = build_panel(frame.sample(fraction=1.0, shuffle=True, seed=1))
    n_series, n_days = p.shape
    assert (n_series, n_days) == (20, 10)
    row = frame.filter((pl.col("store_id") == 102) & (pl.col("product_id") == 503))
    i = p.keys.with_row_index().filter((pl.col("store_id") == 102) & (pl.col("product_id") == 503))[
        "index"
    ][0]
    row = row.sort("dt")
    np.testing.assert_allclose(p.sales[i], np.array(row["hours_sale"].to_list()))
    np.testing.assert_array_equal(p.discount[i], row["discount"].to_numpy())


def test_build_panel_rejects_holes_and_duplicates(frame):
    with pytest.raises(ValueError, match="rectangle"):
        build_panel(frame.slice(1))
    with pytest.raises(ValueError, match="duplicate"):
        build_panel(pl.concat([frame, frame.head(1)]))


# --------------------------------------------------------------------------- features


def test_daily_features_shape_and_lag(frame):
    p = build_panel(frame)
    df = daily_features(p)
    n_series, n_days = p.shape
    assert df.height == n_series * (n_days - 1)
    assert not any(c.startswith("y_") for c in daily_feature_names())
    s = df.filter(pl.col("series_idx") == 3).sort("t_idx")
    np.testing.assert_allclose(s["sale_lag1"].to_numpy()[1:], s["y_daily"].to_numpy()[:-1])


def test_planned_promo_toggle(frame):
    p = build_panel(frame)
    assert "discount" not in daily_features(p, planned_promo=False).columns
    assert "discount" in daily_features(p, planned_promo=True).columns


def test_discount_anomalies_flagged(frame):
    bad = frame.with_columns(
        discount=pl.when(pl.int_range(pl.len()) < 30).then(0.0).otherwise(pl.col("discount"))
    )
    df = daily_features(build_panel(bad), t_min=0)
    assert df["discount_anomaly"].sum() == 30
    assert df.filter(pl.col("discount_anomaly") == 1)["discount"].is_nan().all()


def test_hourly_profile_matches_bruteforce():
    from conftest import make_frame

    p = build_panel(make_frame(n_stores=2, n_products=2, n_days=40, seed=4))
    df = hourly_features(p, t_min=30)
    cls = availability_classes(p.stock)
    s, t, h = 1, 35, 12
    days = [d for d in range(t - 28, t) if cls[s, d, h] == CLS_A]
    expect = np.mean([p.sales[s, d, h] for d in days])
    got = df.filter((pl.col("series_idx") == s) & (pl.col("t_idx") == t) & (pl.col("hour") == h))
    np.testing.assert_allclose(got["hprofile_a28"][0], expect)
    assert set(hourly_feature_names()) <= set(df.columns)
    assert df["hour"].min() == 6 and df["hour"].max() == 21
