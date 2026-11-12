"""Step 3: rolling-origin backtest of baseline forecasters on the validation window.

Usage: python scripts/run_baselines.py --unit day|hour [--config ...] [--smoke]
Writes runs/step3/<unit>/<config_hash>/{predictions.parquet, metrics.json, run_manifest.json}.
Only the official train split is loaded; the locked test split is never read here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sctwin.backtest import BacktestSpec, make_origins
from sctwin.baseline_runner import add_groups, run_backtest
from sctwin.data import load_split
from sctwin.evaluation import compare_to_reference, evaluate
from sctwin.features import FeatureBuilder
from sctwin.panel import build_panel
from sctwin.repro import config_hash, load_config, seed_everything, write_run_manifest
from sctwin.splits import read_manifest
from sctwin.timeunit import TIMEZONE, resample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=["day", "hour"], required=True)
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    args = ap.parse_args()
    cfg = load_config(args.config)
    uc = cfg["units"][args.unit]
    if args.smoke:
        uc.update(train_series=200, eval_series=200, n_folds=2)
        cfg["lgbm"]["num_rounds"] = 20
        cfg["glm"]["n_samples"] = 100
        cfg["bootstrap"]["n_boot"] = 200
        cfg["out_dir"] = "runs/step3_smoke"
    run_cfg = {k: v for k, v in cfg.items() if k != "units"} | {"unit": args.unit, "unit_cfg": uc}
    rng = seed_everything(cfg["seed"])
    out = Path(cfg["out_dir"]) / args.unit / config_hash(run_cfg)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    manifest = read_manifest(cfg["manifest"])
    panel = build_panel(load_split(cfg["data_dir"], "train"))
    g = manifest["group_holdout"]
    holdout = panel.keys["store_id"].is_in(g["stores"]) | panel.keys["product_id"].is_in(
        g["products"]
    )
    in_sample = np.flatnonzero(~holdout.to_numpy())

    def sample(n: int | str) -> np.ndarray:
        if n == "all" or int(n) >= in_sample.size:
            return in_sample
        return np.sort(rng.choice(in_sample, int(n), replace=False))

    train_s = sample(uc["train_series"])
    eval_s = train_s if uc["eval_series"] == uc["train_series"] else sample(uc["eval_series"])
    used = np.union1d(train_s, eval_s)
    sub = panel.subset(used)  # resample only the series we need (hourly is large)
    pos = {int(s): i for i, s in enumerate(used)}
    tr_local = np.array([pos[int(s)] for s in train_s])
    ev_local = np.array([pos[int(s)] for s in eval_s])

    us = resample(sub, args.unit)
    upd = us.units_per_day
    n_fit_days = manifest["temporal"]["fit"]["n_days"]
    first_target, last_target = n_fit_days * upd, panel.shape[1] * upd - 1  # validation units
    origins = make_origins(first_target, last_target, uc["lead"], uc["horizon"], uc["step"],
                           uc["n_folds"])  # fmt: skip
    spec = BacktestSpec(args.unit, uc["lead"], uc["horizon"], origins, uc["window"],
                        uc["window_length"], uc["min_history"], uc["train_stride"])  # fmt: skip

    # low volume: bottom quartile of mean sales over units <= the first origin (no later data)
    hist = us.sales[:, : origins[0] + 1].mean(axis=1)
    low_vol = set(np.flatnonzero(hist <= np.quantile(hist[ev_local], 0.25)).tolist())

    fb = FeatureBuilder(us, sub.keys)
    pred, fitlog = run_backtest(fb, spec, tr_local, ev_local, cfg)
    pred = add_groups(pred, fb, low_vol)
    pred.write_parquet(out / "predictions.parquet")
    taus = tuple(cfg["taus"])
    bs = cfg["bootstrap"]
    report = {
        "protocol": {
            "unit": args.unit,
            "timezone_assumed": TIMEZONE,
            "lead": spec.lead,
            "horizon": spec.horizon,
            "step": uc["step"],
            "window": spec.window,
            "window_length": spec.window_length,
            "train_stride": spec.train_stride,
            "origins": list(origins),
            "n_folds": len(origins),
            "rule": "train on feature times t <= T - L - H; predict (T + L, T + L + H]",
            "targets": "validation window only; official test split not loaded",
            "features": list(fb.feature_names),
        },
        "n_train_series": int(train_s.size),
        "n_eval_series": int(eval_s.size),
        "fit_log": fitlog,
        "metrics": evaluate(pred, taus),
        "compare_wis": compare_to_reference(
            pred, bs["reference"], "wis", taus, n_boot=bs["n_boot"], seed=cfg["seed"]
        ),  # fmt: skip
        "compare_ae": compare_to_reference(
            pred, bs["point_reference"], "ae", taus, n_boot=bs["n_boot"], seed=cfg["seed"]
        ),  # fmt: skip
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_run_manifest(out, run_cfg, stage="03_baselines", smoke=args.smoke)

    ov = report["metrics"]["overall"]["models"]
    print(f"\n== {args.unit}: {len(origins)} folds, {eval_s.size} series, validation window")
    for m, r in ov.items():
        cov = r.get("intervals", {}).get("0.9", {}).get("coverage", float("nan"))
        mae = r.get("mae", r.get("mae_median", float("nan")))
        print(f"  {m:8s} mae={mae:.4f} wis={r.get('wis', float('nan')):.4f} cov90={cov:.3f}")
    print(f"\nwritten to {out}  ({report['runtime_s']} s)")


if __name__ == "__main__":
    main()
