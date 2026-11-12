"""Data audit for FreshRetailNet-style data.

Every check is a pure function of a DataFrame returning a JSON-serializable dict, so the same
code runs on the real data and on synthetic fixtures with planted defects. The audit's purpose
is to establish *what the data measures* (normalization, stockout semantics) before any
modelling choice is made; it reports evidence, it does not decide.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from sctwin.data import KEY, N_HOURS, SERIES_KEY, hourly_matrix, validate_schema

# Hours covered by stock_hour6_22_cnt are ambiguous in the card ("between 6:00 and 22:00").
CNT_WINDOWS = {"h6_21": (6, 22), "h6_22": (6, 23), "h7_22": (7, 23)}


def _f(x: Any) -> float:
    return float(np.round(float(x), 8))


def _mean(x: np.ndarray) -> float | None:
    """Mean that returns None (valid JSON) instead of NaN on an empty selection."""
    return _f(x.mean()) if x.size else None


def _with_dates(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("dt").str.to_date(strict=False).alias("_date"))


# --------------------------------------------------------------------------- structure


def check_structure(df: pl.DataFrame) -> dict[str, Any]:
    sch = validate_schema(df)
    d = _with_dates(df)
    n_dup = d.height - d.unique(subset=list(KEY)).height
    per_series = d.group_by(list(SERIES_KEY)).agg(
        n_days=pl.len(),
        first=pl.col("_date").min(),
        last=pl.col("_date").max(),
    )
    span = (per_series["last"] - per_series["first"]).dt.total_days() + 1
    gaps = int((span != per_series["n_days"]).sum())
    nulls = {c: int(v) for c, v in d.null_count().row(0, named=True).items() if v}
    return {
        "rows": d.height,
        "schema_ok": sch.ok,
        "schema_missing": sch.missing,
        "schema_unexpected": sch.unexpected,
        "schema_wrong_type": {k: list(v) for k, v in sch.wrong_type.items()},
        "unparseable_dates": int(d["_date"].null_count()),
        "duplicate_keys": n_dup,
        "n_series": per_series.height,
        "n_stores": d["store_id"].n_unique(),
        "n_products": d["product_id"].n_unique(),
        "n_cities": d["city_id"].n_unique(),
        "date_min": str(d["_date"].min()),
        "date_max": str(d["_date"].max()),
        "n_distinct_dates": d["_date"].n_unique(),
        "days_per_series": {
            "min": int(per_series["n_days"].min()),  # type: ignore[arg-type]
            "max": int(per_series["n_days"].max()),  # type: ignore[arg-type]
        },
        "series_with_date_gaps": gaps,
        "series_not_sharing_calendar": int(
            int((per_series["first"] != per_series["first"].min()).sum())
            + int((per_series["last"] != per_series["last"].max()).sum())
        ),
        "null_counts": nulls,
        "hours_sale_len": _len_range(df, "hours_sale"),
        "hours_stock_status_len": _len_range(df, "hours_stock_status"),
    }


def _len_range(df: pl.DataFrame, col: str) -> dict[str, int]:
    lens = df[col].list.len()
    return {
        "min": int(lens.min()),  # type: ignore[arg-type]
        "max": int(lens.max()),  # type: ignore[arg-type]
        "rows_not_24": int((lens != N_HOURS).sum()),
    }


# --------------------------------------------------------------------------- values


def check_values(df: pl.DataFrame) -> dict[str, Any]:
    hs = hourly_matrix(df, "hours_sale")
    daily = df["sale_amount"].to_numpy()
    diff = np.abs(hs.sum(axis=1) - daily)
    out: dict[str, Any] = {
        "hourly_nan": int(np.isnan(hs).sum()),
        "hourly_negative": int((hs < 0).sum()),
        "daily_negative": int((daily < 0).sum()),
        "daily_vs_hourly_sum": {
            "max_abs_diff": _f(np.nanmax(diff)),
            "frac_rows_abs_diff_gt_1e-6": _f((diff > 1e-6).mean()),
            "frac_rows_rel_diff_gt_1pct": _f(
                (diff > 0.01 * np.maximum(np.abs(daily), 1e-12)).mean()
            ),
        },
        "frac_zero_days": _f((daily == 0).mean()),
        "frac_zero_hours": _f((hs == 0).mean()),
        "daily_sale_quantiles": {
            str(q): _f(np.quantile(daily, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
    }
    return out


def detect_quantum(
    values: np.ndarray, max_divisor: int = 12, tol: float = 1e-6, target: float = 0.999
) -> dict[str, Any]:
    """Look for a global quantum q such that values are integer multiples of q.

    Evidence only: a quantum existing does NOT establish that v/q are transaction counts
    (weighed products, rounding by the provider, etc. could produce the same pattern).
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return {"n_positive": 0, "quantum": None}
    vmin = float(v.min())
    distinct = np.unique(v)
    candidates = []
    chosen = None
    for k in range(1, max_divisor + 1):
        q = vmin / k
        r = v / q
        frac = float((np.abs(r - np.round(r)) < tol * np.maximum(1.0, r)).mean())
        candidates.append({"divisor": k, "q": q, "frac_integer": _f(frac)})
        if chosen is None and frac >= target:
            chosen = {"divisor": k, "q": q, "frac_integer": _f(frac)}
    head = distinct[:10]
    return {
        "n_positive": int(v.size),
        "n_distinct_positive": int(distinct.size),
        "min_positive": vmin,
        "smallest_distinct": [float(x) for x in head],
        "smallest_distinct_over_min": [_f(x / vmin) for x in head],
        "min_gap_between_distinct": float(np.diff(distinct).min()) if distinct.size > 1 else None,
        "candidates": candidates,
        "quantum": chosen,
    }


def check_normalization(df: pl.DataFrame, sample: int = 2_000_000, seed: int = 0) -> dict[str, Any]:
    hs = hourly_matrix(df, "hours_sale").ravel()
    rng = np.random.default_rng(seed)
    pos = hs[hs > 0]
    if pos.size > sample:
        pos = rng.choice(pos, sample, replace=False)
    overall = detect_quantum(pos)
    # Per management group: weighed vs counted goods may differ.
    per_group: dict[str, Any] = {}
    if overall["quantum"] is not None:
        q = overall["quantum"]["q"]
        for (g,), sub in df.group_by(["management_group_id"]):
            v = hourly_matrix(sub, "hours_sale").ravel()
            v = v[v > 0]
            r = v / q
            per_group[str(g)] = {
                "n_positive": int(v.size),
                "frac_integer_multiple": _f(
                    (np.abs(r - np.round(r)) < 1e-6 * np.maximum(1, r)).mean()
                ),
            }
    daily = df["sale_amount"].to_numpy()
    return {"hourly": overall, "daily": detect_quantum(daily), "per_management_group": per_group}


# --------------------------------------------------------------------------- stockouts


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of runs of ones."""
    padded = np.concatenate(([0], row, [0]))
    d = np.diff(padded)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def check_stockouts(df: pl.DataFrame, run_sample: int = 200_000, seed: int = 0) -> dict[str, Any]:
    st = hourly_matrix(df, "hours_stock_status", np.int8)
    hs = hourly_matrix(df, "hours_sale")
    cnt = df["stock_hour6_22_cnt"].to_numpy()
    values, counts = np.unique(st, return_counts=True)
    so = st == 1
    ins = st == 0

    cnt_match = {
        name: _f((st[:, a:b].sum(axis=1) == cnt).mean()) for name, (a, b) in CNT_WINDOWS.items()
    }

    # Is the flag "out of stock for the whole hour" or "went out of stock during the hour"?
    # Compare sales in the first flagged hour of a run with later flagged hours.
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(so.any(axis=1))
    if idx.size > run_sample:
        idx = rng.choice(idx, run_sample, replace=False)
    first_sales, later_sales, recover, runs_per_day, end_at_23 = [], [], 0, [], 0
    n_runs = 0
    for i in idx:
        runs = _runs(st[i])
        runs_per_day.append(len(runs))
        for a, b in runs:
            n_runs += 1
            first_sales.append(hs[i, a])
            later_sales.extend(hs[i, a + 1 : b].tolist())
            if b < N_HOURS:
                recover += 1
            else:
                end_at_23 += 1
    first_arr = np.asarray(first_sales)
    later_arr = np.asarray(later_sales)

    by_hour = {
        "stockout_rate": [_f(x) for x in so.mean(axis=0)],
        "mean_sale_instock": [_f(x) for x in _masked_mean(hs, ins)],
        "mean_sale_stockout": [_f(x) for x in _masked_mean(hs, so)],
        "frac_positive_sale_stockout": [_f(x) for x in _masked_mean((hs > 0).astype(float), so)],
    }
    return {
        "status_values": {str(int(v)): int(c) for v, c in zip(values, counts, strict=True)},
        "frac_hours_stockout": _f(so.mean()),
        "frac_days_any_stockout": _f(so.any(axis=1).mean()),
        "frac_days_all_24_stockout": _f(so.all(axis=1).mean()),
        "frac_days_all_6_21_stockout": _f(so[:, 6:22].all(axis=1).mean()),
        "cnt_field_match_by_window": cnt_match,
        "frac_stockout_hours_with_positive_sale": _f((hs[so] > 0).mean()) if so.any() else None,
        "mean_sale_stockout_hours": _f(hs[so].mean()) if so.any() else None,
        "mean_sale_instock_hours": _f(hs[ins].mean()) if ins.any() else None,
        "runs": {
            "days_sampled": int(idx.size),
            "n_runs": n_runs,
            "mean_runs_per_stockout_day": _f(np.mean(runs_per_day)) if runs_per_day else None,
            "frac_runs_recovering_same_day": _f(recover / n_runs) if n_runs else None,
            "frac_runs_ending_at_hour_23": _f(end_at_23 / n_runs) if n_runs else None,
            "mean_sale_first_flagged_hour": _f(first_arr.mean()) if first_arr.size else None,
            "frac_positive_first_flagged_hour": _f((first_arr > 0).mean())
            if first_arr.size
            else None,
            "mean_sale_later_flagged_hours": _f(later_arr.mean()) if later_arr.size else None,
            "frac_positive_later_flagged_hours": _f((later_arr > 0).mean())
            if later_arr.size
            else None,
        },
        "by_hour": by_hour,
    }


def _status_values(df: pl.DataFrame) -> dict[str, int]:
    v, c = np.unique(hourly_matrix(df, "hours_stock_status", np.int8), return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(v, c, strict=True)}


def _masked_mean(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    n = m.sum(axis=0)
    s = np.where(m, x, 0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def check_lattice(df: pl.DataFrame, steps: tuple[float, ...] = (0.01, 0.05, 0.1)) -> dict[str, Any]:
    """Fraction of positive hourly values on each candidate lattice, overall / per series /
    per management group. A lattice is evidence of discreteness, NOT proof of unit counts."""
    hs = hourly_matrix(df, "hours_sale")
    pos = hs > 0
    sid = df.select(pl.struct(*SERIES_KEY).rank("dense")).to_series().to_numpy() - 1
    n_series = int(sid.max()) + 1
    mg = df["management_group_id"].to_numpy()
    out: dict[str, Any] = {}
    for step in steps:
        r = hs / step
        on = np.abs(r - np.round(r)) < 1e-6 * np.maximum(1.0, r)
        off_rows = (pos & ~on).sum(axis=1)
        ser_off = np.bincount(sid, weights=off_rows, minlength=n_series)
        ser_pos = np.bincount(sid, weights=pos.sum(axis=1), minlength=n_series)
        out[str(step)] = {
            "frac_positive_on_lattice": _mean(on[pos]),
            "frac_series_fully_on_lattice": _f((ser_off == 0).mean()),
            "frac_series_gt5pct_off_lattice": _f((ser_off / np.maximum(ser_pos, 1) > 0.05).mean()),
            "per_management_group": {
                str(g): _mean(on[mg == g][pos[mg == g]]) for g in np.unique(mg)
            },
        }
    return out


def check_availability_transitions(df: pl.DataFrame, min_full_days: int = 10) -> dict[str, Any]:
    """Within-series sales around stockout-flag transitions, relative to the same series'
    mean at the same hour on days fully in stock over hours 6-21.

    Ratios near 1 before a run would suggest exogenous stockouts; ratios >> 1 indicate
    demand-driven depletion (availability is endogenous). The first flagged hour's ratio
    measures residual sales (partial availability or flag noise). Hours 6-21 only.
    """
    d = df.sort([*SERIES_KEY, "dt"])
    hs = hourly_matrix(d, "hours_sale")
    st = hourly_matrix(d, "hours_stock_status", np.int8)
    sid = d.select(pl.struct(*SERIES_KEY).rank("dense")).to_series().to_numpy() - 1
    n_series = int(sid.max()) + 1
    full = st[:, 6:22].sum(axis=1) == 0
    base = np.zeros((n_series, N_HOURS))
    n_full = np.bincount(sid[full], minlength=n_series)
    np.add.at(base, sid[full], hs[full])
    base /= np.maximum(n_full, 1)[:, None]
    ok = n_full >= min_full_days

    prev = np.zeros_like(st)
    prev[:, 1:] = st[:, :-1]
    start = (st == 1) & (prev == 0)
    end = (st == 0) & (prev == 1)
    start[:, 0] = end[:, 0] = False

    def ratio(mask: np.ndarray, offset: int) -> dict[str, Any]:
        r, h = np.nonzero(mask)
        hh = h + offset
        keep = (hh >= 6) & (hh <= 21) & ok[sid[r]]
        r, hh = r[keep], hh[keep]
        den = base[sid[r], hh].sum()
        return {"n": int(r.size), "ratio": _f(hs[r, hh].sum() / den) if den > 0 else None}

    return {
        "series_with_baseline": int(ok.sum()),
        "min_full_days": min_full_days,
        "before_run_start": {f"-{k}h": ratio(start, -k) for k in (3, 2, 1)},
        "first_flagged_hour": ratio(start, 0),
        "second_flagged_hour": ratio(start & np.roll(st == 1, -1, axis=1), 1),
        "first_instock_after_restock": ratio(end, 0),
        "second_instock_after_restock": ratio(end, 1),
    }


HIERARCHY_EDGES = (
    ("store_id", "city_id"),
    ("first_category_id", "management_group_id"),
    ("second_category_id", "first_category_id"),
    ("third_category_id", "second_category_id"),
    ("product_id", "third_category_id"),
)


def check_hierarchy(df: pl.DataFrame) -> dict[str, Any]:
    """Each child id must map to exactly one parent id (strict trees)."""
    out: dict[str, Any] = {}
    for child, parent in HIERARCHY_EDGES:
        n_par = df.group_by(child).agg(pl.col(parent).n_unique())[parent]
        out[f"{child}->{parent}"] = {
            "n_children": int(n_par.len()),
            "children_with_multiple_parents": int((n_par > 1).sum()),
        }
    out["is_strict_tree"] = all(
        v["children_with_multiple_parents"] == 0 for k, v in out.items() if "->" in k
    )
    return out


# --------------------------------------------------------------------------- covariates


def check_covariates(df: pl.DataFrame) -> dict[str, Any]:
    disc = df["discount"].to_numpy()

    def nuniq_within(group: list[str], col: str) -> int:
        return int(df.group_by(group).agg(pl.col(col).n_unique())[col].max())  # type: ignore[arg-type]

    weather = ["precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]
    return {
        "discount": {
            "min": _f(disc.min()),
            "max": _f(disc.max()),
            "frac_discounted_lt_1": _f((disc < 1).mean()),
            "frac_gt_1": _f((disc > 1).mean()),
        },
        "activity_flag_rate": _f(df["activity_flag"].mean()),
        "holiday_flag_rate": _f(df["holiday_flag"].mean()),
        # Granularity evidence: 1 => covariate constant at that level.
        "max_distinct_holiday_per_date": nuniq_within(["dt"], "holiday_flag"),
        "max_distinct_weather_per_city_date": {
            w: nuniq_within(["city_id", "dt"], w) for w in weather
        },
        "max_distinct_weather_per_store_date": {
            w: nuniq_within(["store_id", "dt"], w) for w in weather
        },
        "max_distinct_activity_per_store_date": nuniq_within(["store_id", "dt"], "activity_flag"),
        "max_distinct_city_per_store": nuniq_within(["store_id"], "city_id"),
        "max_distinct_category_per_product": nuniq_within(["product_id"], "third_category_id"),
    }


# --------------------------------------------------------------------------- cross split


def check_cross_split(train: pl.DataFrame, test: pl.DataFrame) -> dict[str, Any]:
    tr, te = _with_dates(train), _with_dates(test)
    s_tr = tr.select(SERIES_KEY).unique()
    s_te = te.select(SERIES_KEY).unique()
    last_tr = tr.group_by(list(SERIES_KEY)).agg(last_train=pl.col("_date").max())
    first_te = te.group_by(list(SERIES_KEY)).agg(first_test=pl.col("_date").min())
    j = last_tr.join(first_te, on=list(SERIES_KEY), how="inner")
    gap = (j["first_test"] - j["last_train"]).dt.total_days()
    overlap = tr.select(KEY).join(te.select(KEY), on=list(KEY), how="inner").height
    return {
        "series_train": s_tr.height,
        "series_test": s_te.height,
        "series_in_both": s_tr.join(s_te, on=list(SERIES_KEY), how="inner").height,
        "series_test_only": s_te.join(s_tr, on=list(SERIES_KEY), how="anti").height,
        "overlapping_keys": overlap,
        "series_test_not_strictly_after_train": int((gap <= 0).sum()),
        "gap_days_min": int(gap.min()) if gap.len() else None,  # type: ignore[arg-type]
        "gap_days_max": int(gap.max()) if gap.len() else None,  # type: ignore[arg-type]
    }


# --------------------------------------------------------------------------- gates


def gate(report: dict[str, Any]) -> dict[str, bool]:
    """Pass/fail gates for Step 1. Evidence fields (normalization, stockouts) are not gated."""
    g: dict[str, bool] = {}
    for split in ("train", "eval"):
        s = report[split]["structure"]
        v = report[split]["values"]
        g[f"{split}.schema_ok"] = s["schema_ok"]
        g[f"{split}.no_duplicate_keys"] = s["duplicate_keys"] == 0
        g[f"{split}.dates_parse"] = s["unparseable_dates"] == 0
        g[f"{split}.arrays_len_24"] = (
            s["hours_sale_len"]["rows_not_24"] == 0
            and s["hours_stock_status_len"]["rows_not_24"] == 0
        )
        g[f"{split}.no_negative_or_nan_sales"] = (
            v["hourly_nan"] == 0 and v["hourly_negative"] == 0 and v["daily_negative"] == 0
        )
        g[f"{split}.status_binary"] = set(report[split]["stockouts"]["status_values"]) <= {"0", "1"}
    g["train.hierarchy_strict_tree"] = report["train"]["hierarchy"]["is_strict_tree"]
    c = report["cross_split"]
    g["cross.no_overlapping_keys"] = c["overlapping_keys"] == 0
    g["cross.test_strictly_after_train"] = c["series_test_not_strictly_after_train"] == 0
    return g


def run_audit(train: pl.DataFrame, test: pl.DataFrame) -> dict[str, Any]:
    """Full evidence audit on train; integrity-only audit on the locked test split, so that no
    distributional finding from the test set can inform modelling decisions."""
    report: dict[str, Any] = {
        "train": {
            "structure": check_structure(train),
            "values": check_values(train),
            "normalization": check_normalization(train),
            "stockouts": check_stockouts(train),
            "lattice": check_lattice(train),
            "availability_transitions": check_availability_transitions(train),
            "covariates": check_covariates(train),
            "hierarchy": check_hierarchy(train),
        },
        "eval": {
            "structure": check_structure(test),
            "values": check_values(test),
            "stockouts": {"status_values": _status_values(test)},
        },
    }
    report["cross_split"] = check_cross_split(train, test)
    report["gates"] = gate(report)
    report["all_gates_pass"] = all(report["gates"].values())
    return report
