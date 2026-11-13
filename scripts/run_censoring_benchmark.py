"""Step 4: semi-synthetic censoring benchmark on real fully in-stock days.

Usage: python scripts/run_censoring_benchmark.py [--config configs/censoring.yaml] [--smoke]

1. Measure real stockout statistics on the fit window (rate, timing, restocks, sales around
   the first flagged hour).
2. Calibrate the injection (stock coverage and spread) on fully in-stock fit-window days.
3. Build the benchmark on fully in-stock validation days, check its realism against step 1,
   and score the reconstruction baselines on the injected (known) demand proxy.
4. Repeat for the pre-declared sensitivity scenarios.

Writes runs/step4/<config_hash>/{metrics.json, run_manifest.json}. Only the official train
split is loaded; the locked test split is never read here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from sctwin.censoring import (
    HI,
    IMPUTERS,
    LO,
    Benchmark,
    CensorConfig,
    build_benchmark,
    calibrate,
    calibrate_leak,
    calibrate_restock,
    candidate_days,
    expected_window_demand,
    position_ratios,
    recovery_frames,
    stockout_stats,
)
from sctwin.data import load_split
from sctwin.evaluation import compare_to_reference, evaluate
from sctwin.panel import build_panel
from sctwin.repro import config_hash, load_config, seed_everything, write_run_manifest
from sctwin.splits import read_manifest

TAUS = (0.5,)  # point methods only in Step 4


def score(bm: Benchmark, n_boot: int, reference: str, seed: int) -> dict[str, Any]:
    est = {m: f(bm.injection.observed, bm.injection.flag, bm.hour_means)
           for m, f in IMPUTERS.items()}  # fmt: skip
    hourly, daily = recovery_frames(bm, est)
    out: dict[str, Any] = {
        "n_days": int(bm.series.size),
        "n_stockout_days": int(daily.height),
        "n_censored_hours": int(hourly.height),
        "hourly": evaluate(hourly, TAUS),
        "daily": evaluate(daily, TAUS),
    }
    if n_boot:
        out["compare_hourly_ae"] = compare_to_reference(hourly, reference, "ae", TAUS, n_boot, seed)
        out["compare_daily_ae"] = compare_to_reference(daily, reference, "ae", TAUS, n_boot, seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/censoring.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg.update(n_series=300, out_dir="runs/step4_smoke")
        cfg["bootstrap"]["n_boot"] = 200
    rng = seed_everything(cfg["seed"])
    out = Path(cfg["out_dir"]) / config_hash(cfg)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    manifest = read_manifest(cfg["manifest"])
    panel = build_panel(load_split(cfg["data_dir"], "train"))
    g = manifest["group_holdout"]
    hold = panel.keys["store_id"].is_in(g["stores"]) | panel.keys["product_id"].is_in(g["products"])
    in_sample = np.flatnonzero(~hold.to_numpy())
    series = np.sort(rng.choice(in_sample, min(cfg["n_series"], in_sample.size), replace=False))
    sub = panel.subset(series)
    sales, stock = sub.sales, sub.stock
    s_local = np.arange(series.size)
    n_fit, n_days, w = manifest["temporal"]["fit"]["n_days"], panel.shape[1], cfg["window"]
    fit_days, val_days = np.arange(w, n_fit), np.arange(n_fit, n_days)

    # 1. real stockout statistics, fit window only
    real_flags = stock[:, w:n_fit].reshape(-1, 24)
    real = stockout_stats(real_flags)
    full_fit = (stock[:, :n_fit, LO:HI] == 0).all(-1)  # (S, n_fit) fully in-stock days
    cnt = full_fit.sum(1)[:, None]
    in_stock_sum = (sales[:, :n_fit] * full_fit[..., None]).sum(1)
    baseline = np.where(cnt > 0, in_stock_sum / np.maximum(cnt, 1), np.nan)
    pos = tuple(cfg["positions"])
    real_pos = position_ratios(
        sales[:, w:n_fit].reshape(-1, 24), real_flags,
        np.repeat(baseline, n_fit - w, axis=0), pos,
    )  # fmt: skip

    # 2. calibration on fully in-stock fit-window days
    cal = cfg["calibration"]
    base = CensorConfig(leak_first=cal["leak_first"], leak_later=cal["leak_later"],
                        p_restock=real["restock_rate"],
                        restock_mean_hours=max(1.0, real["restock_mean_hours"]))  # fmt: skip
    fit_rows = candidate_days(sales, stock, s_local, fit_days, w)
    c = cal["coverages"]
    best, grid = calibrate(
        sales[fit_rows[0], fit_rows[1]].astype(np.float64),
        expected_window_demand(fit_rows[2]),
        real, base,
        np.round(np.arange(c["start"], c["stop"] + 1e-9, c["step"]), 4),
        np.asarray(cal["sds"], dtype=np.float64), seed=cfg["seed"],
    )  # fmt: skip
    best = calibrate_restock(
        sales[fit_rows[0], fit_rows[1]].astype(np.float64),
        expected_window_demand(fit_rows[2]),
        real, best,
        np.round(np.linspace(0.5, 2.0, 16) * real["restock_rate"], 4),
        np.round(np.arange(1.0, 8.01, 0.25), 2), seed=cfg["seed"],
    )  # fmt: skip
    best = calibrate_leak(
        sales[fit_rows[0], fit_rows[1]].astype(np.float64),
        expected_window_demand(fit_rows[2]),
        baseline[fit_rows[0]], real_pos, best, seed=cfg["seed"],
    )  # fmt: skip
    print(f"calibrated: {dataclasses.asdict(best)}")

    # 3. benchmark on fully in-stock validation days
    val_rows = candidate_days(sales, stock, s_local, val_days, w)
    bs = cfg["bootstrap"]
    bm = build_benchmark(sales, val_rows, best, seed=cfg["seed"] + 1)
    inj_pos = position_ratios(bm.injection.observed, bm.injection.flag,
                              baseline[bm.series], pos)  # fmt: skip
    main_res = score(bm, bs["n_boot"], bs["reference"], cfg["seed"])

    # 4. sensitivity scenarios (same days, same random numbers)
    sens: dict[str, Any] = {}
    for name, ov in cfg["sensitivity"].items():
        ov = dict(ov)
        mult = ov.pop("coverage_mult", 1.0)
        sc = dataclasses.replace(best, coverage=best.coverage * mult, **ov)
        b2 = build_benchmark(sales, val_rows, sc, seed=cfg["seed"] + 1)
        r = score(b2, 0, bs["reference"], cfg["seed"])
        sens[name] = {"config": dataclasses.asdict(sc),
                      "stats": stockout_stats(b2.injection.flag),
                      "hourly": r["hourly"]["overall"], "daily": r["daily"]["overall"]}  # fmt: skip

    report = {
        "truth": "demand proxy = real sales on fully in-stock days (hours 6-21); not demand",
        "series": int(series.size),
        "window_days": w,
        "calibration_days": [int(fit_days[0]), int(fit_days[-1])],
        "benchmark_days": [int(val_days[0]), int(val_days[-1])],
        "real_fit_window": {"stats": real, "position_ratios": real_pos},
        "injected": {"stats": stockout_stats(bm.injection.flag), "position_ratios": inj_pos},
        "calibrated_config": dataclasses.asdict(best),
        "calibration_grid_top": sorted(grid, key=lambda r: r["loss"])[:10],
        "benchmark": main_res,
        "sensitivity": sens,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_run_manifest(out, cfg, stage="04_censoring", smoke=args.smoke)

    print(f"\nreal (fit)  : {real}\ninjected    : {report['injected']['stats']}")
    print(f"positions real {real_pos}\n          inj  {inj_pos}")
    print(f"\n== {main_res['n_stockout_days']} stockout days / {main_res['n_days']} days, "
          f"{main_res['n_censored_hours']} censored hours")  # fmt: skip
    for level in ("hourly", "daily"):
        for m, r in main_res[level]["overall"]["models"].items():
            print(f"  {level:6s} {m:12s} mae={r['mae']:.4f} wape={r['wape']:.3f} "
                  f"bias={r['bias']:+.4f}")  # fmt: skip
    print(f"\nwritten to {out}  ({report['runtime_s']} s)")


if __name__ == "__main__":
    main()
