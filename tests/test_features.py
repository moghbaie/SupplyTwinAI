import numpy as np
import polars as pl
import pytest
from test_no_leakage import SMALL, builder, random_panel

from sctwin.features import FeatureBuilder, trailing_mean, units_since
from sctwin.panel import (
    CLS_A,
    CLS_NIGHT,
    CLS_P_FIRST,
    CLS_P_PRE,
    CLS_U,
    availability_classes,
    build_panel,
)
from sctwin.timeunit import resample

# --------------------------------------------------------------------------- primitives


def test_trailing_mean_hand():
    x = np.array([[1.0, 2, 3, 4]])
    np.testing.assert_array_equal(trailing_mean(x, 2), [[np.nan, 1, 1.5, 2.5]])
    m = np.array([[True, False, True, True]])
    np.testing.assert_array_equal(trailing_mean(x, 2, mask=m), [[np.nan, 1, 1, 3]])


def test_units_since():
    e = np.array([[0, 1, 0, 0, 1, 0]], dtype=bool)
    np.testing.assert_array_equal(units_since(e), [[np.nan, np.nan, 1, 2, 3, 1]])


def test_availability_classes_hand():
    st = np.zeros(24, dtype=np.int8)
    st[[10, 11, 12, 15, 3]] = 1
    c = availability_classes(st)
    expect = {9: CLS_P_PRE, 10: CLS_P_FIRST, 11: CLS_U, 12: CLS_U, 13: CLS_A, 14: CLS_P_PRE,
              15: CLS_P_FIRST, 16: CLS_A, 3: CLS_NIGHT, 22: CLS_NIGHT, 6: CLS_A}  # fmt: skip
    for h, v in expect.items():
        assert c[h] == v, h


# --------------------------------------------------------------------------- panel / resample


def test_build_panel_roundtrip(frame):
    p = build_panel(frame.sample(fraction=1.0, shuffle=True, seed=1))
    assert p.shape == (20, 10)
    i = p.keys.with_row_index().filter((pl.col("store_id") == 102) & (pl.col("product_id") == 503))[
        "index"
    ][0]
    row = frame.filter((pl.col("store_id") == 102) & (pl.col("product_id") == 503)).sort("dt")
    np.testing.assert_allclose(p.sales[i], np.array(row["hours_sale"].to_list()))


def test_build_panel_rejects_holes_and_duplicates(frame):
    with pytest.raises(ValueError, match="rectangle"):
        build_panel(frame.slice(1))
    with pytest.raises(ValueError, match="duplicate"):
        build_panel(pl.concat([frame, frame.head(1)]))


def test_resample_day_and_hour():
    p = random_panel(2, 5, 0)
    d, h = resample(p, "day"), resample(p, "hour")
    np.testing.assert_allclose(d.sales, p.sales.sum(-1))
    assert h.sales.shape == (2, 120)
    np.testing.assert_array_equal(h.sales[1, 2 * 24 + 7], p.sales[1, 2, 7])
    np.testing.assert_array_equal(d.stockout[0], p.stock[0, :, 6:22].sum(-1))
    assert h.last_complete_day(np.array([22, 23, 47, 48])).tolist() == [-1, 0, 1, 1]
    assert d.last_complete_day(np.array([0, 3])).tolist() == [0, 3]
    with pytest.raises(ValueError):
        resample(p, "week")


# --------------------------------------------------------------------------- features


@pytest.mark.parametrize("unit", ["day", "hour"])
def test_lags_are_relative_to_origin(unit):
    p = random_panel(1, 12, 2)
    fb = builder(p, unit)
    x = fb.us.sales[0]
    origin = 200 if unit == "hour" else 6
    r = fb.rows([origin], lead=2, horizon=3)
    for k in fb.cfg.lags:
        np.testing.assert_array_equal(r[f"sale_lag{k}"].to_numpy(), x[origin - k])
    w = fb.cfg.windows[0]
    np.testing.assert_allclose(r[f"sale_mean{w}"][0], x[origin - w + 1 : origin + 1].mean())
    assert (r["steps_ahead"].to_numpy() == [3, 4, 5]).all()
    assert (r["target_idx"].to_numpy() == origin + np.array([3, 4, 5])).all()


def test_seasonal_last_and_instock_mean_bruteforce():
    p = random_panel(1, 20, 3)
    fb = builder(p, "hour")
    origin, lead, horizon = 10 * 24 + 13, 1, 24
    r = fb.rows([origin], lead, horizon)
    x, ok = fb.us.sales[0], fb.us.instock_next_known[0]
    for row in r.iter_rows(named=True):
        u = row["target_idx"]
        src = [u - 24 * k for k in range(1, 40) if u - 24 * k <= origin][: fb.cfg.n_seasons]
        assert row["sale_seasonal_last"] == x[src[0]]
        good = [x[i] for i in src if ok[i] and i + 1 <= origin]
        exp = np.mean(good) if good else np.nan
        np.testing.assert_allclose(row["sale_seasonal_instock_mean"], exp)


def test_no_target_time_promotions_in_features():
    fb = builder(random_panel(1, 10, 0), "day")
    assert not any(n in fb.feature_names for n in ("discount", "activity", "target_discount"))
    assert all(not n.startswith("y") for n in fb.feature_names)


def test_discount_anomalies_flagged():
    p = random_panel(2, 10, 0)
    fb = builder(p, "day")
    bad = (p.discount <= 0) | (p.discount > 1)
    assert fb.discount_bad.sum() == bad.sum() and np.isnan(fb.discount[bad]).all()


def test_default_configs_build():
    p = random_panel(2, 40, 1)
    for unit in ("day", "hour"):
        fb = FeatureBuilder(resample(p, unit), p.keys)
        r = fb.rows([fb.us.n_units - 30], 1, 7)
        assert set(fb.feature_names) <= set(r.columns) and r.height == 14
    assert SMALL["day"].season == 7
