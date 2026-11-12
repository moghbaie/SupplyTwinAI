"""The audit must find defects we plant, and must not raise false alarms on clean data."""

import json

import numpy as np
import polars as pl
from conftest import QUANTUM

from sctwin import audit


def test_clean_fixture_passes_all_gates(train_test):
    rep = audit.run_audit(*train_test)
    assert rep["all_gates_pass"], {k: v for k, v in rep["gates"].items() if not v}
    json.dumps(rep)  # serializable


def test_detects_duplicate_keys(frame):
    dup = pl.concat([frame, frame.head(3)])
    assert audit.check_structure(dup)["duplicate_keys"] == 3


def test_detects_date_gap(frame):
    gapped = frame.filter(pl.col("dt") != "2024-03-05")
    s = audit.check_structure(gapped)
    assert s["series_with_date_gaps"] == s["n_series"]


def test_detects_daily_hourly_mismatch(frame):
    bad = frame.with_columns(
        pl.when(pl.int_range(pl.len()) < 4)
        .then(pl.col("sale_amount") + 1)
        .otherwise(pl.col("sale_amount"))
    )
    v = audit.check_values(bad)
    assert v["daily_vs_hourly_sum"]["frac_rows_abs_diff_gt_1e-6"] == round(4 / frame.height, 8)


def test_detects_negative_sales(frame):
    bad = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("hours_sale").list.eval(pl.element() - 1))
        .otherwise(pl.col("hours_sale"))
    )
    assert audit.check_values(bad)["hourly_negative"] > 0


def test_quantum_recovered():
    rng = np.random.default_rng(0)
    vals = rng.integers(1, 50, 10_000) * QUANTUM
    q = audit.detect_quantum(vals)["quantum"]
    assert q is not None and q["divisor"] == 1
    np.testing.assert_allclose(q["q"], QUANTUM)


def test_quantum_when_min_is_not_one_unit():
    rng = np.random.default_rng(0)
    vals = rng.integers(2, 50, 10_000) * 2 * QUANTUM + rng.integers(0, 2, 10_000) * QUANTUM
    vals = vals[vals > 0]
    q = audit.detect_quantum(vals)["quantum"]
    assert q is not None
    np.testing.assert_allclose(q["q"], QUANTUM, rtol=1e-9)


def test_no_quantum_for_continuous_values():
    vals = np.random.default_rng(0).gamma(2.0, 1.0, 10_000)
    assert audit.detect_quantum(vals)["quantum"] is None


def test_stockout_count_window_identified(frame):
    s = audit.check_stockouts(frame)
    assert s["cnt_field_match_by_window"]["h6_21"] == 1.0
    assert set(s["status_values"]) <= {"0", "1"}


def test_stockout_runs_persist_in_fixture(frame):
    r = audit.check_stockouts(frame)["runs"]
    assert r["frac_runs_ending_at_hour_23"] == 1.0
    assert r["frac_positive_later_flagged_hours"] == 0.0


def test_non_binary_status_fails_gate(train_test):
    tr, te = train_test
    bad = tr.with_columns(pl.col("hours_stock_status").list.eval(pl.element() * 2))
    rep = audit.run_audit(bad, te)
    assert not rep["gates"]["train.status_binary"]


def test_cross_split_detects_overlap(train_test):
    tr, te = train_test
    leaky = pl.concat([te, tr.tail(5)])
    c = audit.check_cross_split(tr, leaky)
    assert c["overlapping_keys"] == 5
    assert c["series_test_not_strictly_after_train"] > 0


def test_covariate_granularity(frame):
    c = audit.check_covariates(frame)
    assert c["max_distinct_holiday_per_date"] == 1
    assert c["max_distinct_city_per_store"] == 1


def test_lattice_detects_tenth_grid(frame):
    lat = audit.check_lattice(
        frame.with_columns(pl.col("hours_sale").list.eval(pl.element() * 0) + pl.lit(0.0)),
        steps=(0.1,),
    )
    assert lat["0.1"]["frac_series_fully_on_lattice"] == 1.0  # all zeros: trivially on lattice
    tenth = frame.with_columns(
        pl.col("hours_sale").list.eval((pl.element() / QUANTUM).round(0) * 0.1)
    )
    assert audit.check_lattice(tenth, steps=(0.1,))["0.1"]["frac_positive_on_lattice"] == 1.0
    assert audit.check_lattice(frame, steps=(0.1,))["0.1"]["frac_positive_on_lattice"] < 0.5


def _plant_surge(df: pl.DataFrame, factor: float) -> pl.DataFrame:
    hs = np.array(df["hours_sale"].to_list())
    st = np.array(df["hours_stock_status"].to_list())
    for i in range(len(hs)):
        on = np.flatnonzero(st[i])
        if on.size and on[0] >= 7:
            hs[i, on[0] - 1] *= factor
    return df.with_columns(hours_sale=pl.Series(hs.tolist()), sale_amount=pl.Series(hs.sum(axis=1)))


def test_transitions_exogenous_stockouts_ratio_near_one():
    from conftest import make_frame

    df = make_frame(n_stores=6, n_products=6, n_days=60, seed=3)
    t = audit.check_availability_transitions(df, min_full_days=5)
    assert 0.7 < t["before_run_start"]["-1h"]["ratio"] < 1.3
    assert t["first_flagged_hour"]["ratio"] == 0.0


def test_transitions_detect_demand_driven_depletion():
    from conftest import make_frame

    df = _plant_surge(make_frame(n_stores=6, n_products=6, n_days=60, seed=3), factor=3.0)
    t = audit.check_availability_transitions(df, min_full_days=5)
    assert t["before_run_start"]["-1h"]["ratio"] > 2.0
    assert 0.7 < t["before_run_start"]["-2h"]["ratio"] < 1.3
