"""Step 5 revision: replicated level recovery of the flagged-hour treatments (plan.md).

Usage: python scripts/run_level_recovery.py [--reps 8] [--chains 4] [--workers N]
                                            [--stock day_effect|no_day_effect] [--smoke]

Checks whether dropping flagged hours (M1d) recovers the demand level without bias, with the
acceptance rule declared in plan.md ("Step 5 revision") before this run. Static truth (rho = 0),
so the i.i.d. models are correctly specified. Each replication draws new series, hour and day
effects, simulates demand on the real covariates and calendar of the Step 5 series, injects
stockouts with the calibrated Step 4 mechanism (configs/bayes.yaml), and fits

  oracle     M0 with every hour exact on the *uncensored* demand
  M0, M1, M1d  the ladder on the censored data.

The level is the expected hours 6-21 demand per series-day, averaged. Comparing each model
with the oracle fit of the same replication removes the realisation noise of the simulated
demand (about 1.3% sd).

--stock: the opening stock is coverage x expected window demand x noise. With "day_effect"
(default, as in the Step 5 recovery) the expectation includes the true day effect u, so the
stock carries information about u that the likelihood ignores; "no_day_effect" removes u from
it. Comparing the two isolates that channel (the follow-up declared in plan.md).

Replications run in parallel worker processes, one fit at a time per worker with its chains on
separate cores. Only the train split's fit window is read.
Writes runs/level_recovery/<hash>/{results.json, run_manifest.json}; results.json is
rewritten after every replication, so partial results survive an interrupted run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/bayes.yaml")
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--models", default="oracle,M0,M1,M1d")
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--samples", type=int, default=500)
ap.add_argument("--workers", type=int, default=0, help="0 = cores // chains")
ap.add_argument(
    "--stock",
    choices=["day_effect", "no_day_effect"],
    default="day_effect",
    help="whether the opening stock scales with the true day effect u",
)
ap.add_argument("--smoke", action="store_true")
ARGS = ap.parse_args()
if ARGS.smoke:
    ARGS.reps, ARGS.warmup, ARGS.samples = 2, 100, 100

import numpyro  # noqa: E402

numpyro.set_host_device_count(ARGS.chains)  # before JAX starts; inherited by the workers

from sctwin import demand_model as dm  # noqa: E402
from sctwin.censoring import (  # noqa: E402
    HI,
    LO,
    CensorConfig,
    hour_profile,
    inject_stockouts,
    stock_levels,
)
from sctwin.panel import CLS_A, availability_classes  # noqa: E402
from sctwin.repro import config_hash, load_config, write_run_manifest  # noqa: E402

PK = ("alpha", "a", "gamma", "g", "dow_eff", "b_disc", "b_act")
DOW_TRUTH = np.array([0.0, -0.05, -0.05, 0.0, 0.1, 0.2, 0.15])


def level_draws(res: dm.FitResult, cov: dict[str, np.ndarray], n: int = 200) -> np.ndarray:
    s = res.samples
    out = []
    for i in np.linspace(0, len(s["alpha"]) - 1, n).astype(int):
        e = dm.linear_predictor({k: s[k][i] for k in PK}, cov["dow"], cov["x_disc"], cov["x_act"])
        out.append(np.exp(np.asarray(e) + s["u"][i][:, :, None]).sum(-1).mean())
    return np.asarray(out)


def one_rep(rep: int, cov: dict[str, np.ndarray], truth_cfg: dict[str, float],
            injection: dict[str, float], models: list[str], mcmc: dict[str, int],
            stock_with_u: bool) -> dict:  # fmt: skip
    """Simulate, censor and fit one replication (runs in a worker process)."""
    n_s, n_d = cov["x_disc"].shape
    t = truth_cfg
    r = np.random.default_rng(10_000 + rep)
    g_raw = r.standard_normal((n_s, dm.NH))
    truth = {**t, "a": t["sigma_a"] * r.standard_normal(n_s),
             "gamma": dm._centred(np.log(hour_profile()[LO:HI])),
             "g": t["tau_g"] * np.asarray(dm._centred(dm._centred(g_raw, -1), 0)),
             "dow_eff": dm._centred(DOW_TRUTH),
             "u": t["sigma_u"] * r.standard_normal((n_s, n_d))}  # fmt: skip
    dow, x_disc, x_act = cov["dow"], cov["x_disc"], cov["x_act"]
    demand = dm.simulate(truth, dow, x_disc, x_act, seed=20_000 + rep)
    eta = np.asarray(dm.linear_predictor(truth, dow, x_disc, x_act)) + truth["u"][:, :, None]
    full = np.zeros((n_s * n_d, 24))
    full[:, LO:HI] = demand.reshape(-1, dm.NH)
    r2 = np.random.default_rng(30_000 + rep)
    cfg = CensorConfig(**injection)
    expected = np.exp(eta if stock_with_u else eta - truth["u"][:, :, None]).sum(-1).ravel()
    inj = inject_stockouts(full, stock_levels(expected, r2, cfg), r2, cfg)
    common = {"dow": dow, "x_disc": x_disc, "x_act": x_act, "series": np.arange(n_s),
              "days": np.arange(n_d)}  # fmt: skip
    censored = dm.ModelData(
        y=inj.observed[:, LO:HI].reshape(n_s, n_d, dm.NH),
        cls=availability_classes(inj.flag)[:, LO:HI].reshape(n_s, n_d, dm.NH), **common
    )  # fmt: skip
    oracle = dm.ModelData(y=demand, cls=np.full(demand.shape, CLS_A, dtype=np.int8), **common)
    row: dict[str, Any] = {"rep": rep, "true_level": float(np.exp(eta).sum(-1).mean()),
                           "realised_level": float(demand.sum(-1).mean()),
                           "stockout_day_rate": float(inj.stocked_out.mean()),
                           "models": {}}  # fmt: skip
    for j, name in enumerate(models):
        data, spec = (oracle, dm.LADDER["M0"]) if name == "oracle" else (censored, dm.LADDER[name])
        t0 = time.perf_counter()
        res = dm.fit(data, spec, seed=100 * rep + j, **mcmc)
        lv = level_draws(res, cov)
        d = res.diagnostics
        row["models"][name] = {
            "mean": float(lv.mean()), "q05": float(np.quantile(lv, 0.05)),
            "q95": float(np.quantile(lv, 0.95)), "runtime_s": round(time.perf_counter() - t0, 1),
            "divergences": d["divergences"], "max_rhat": d["max_rhat"], "min_ess": d["min_ess"],
            "sigma_u": float(res.samples["sigma_u"].mean()),
        }  # fmt: skip
    return row


def summarise(rows: list[dict], models: list[str], declared: bool) -> dict[str, Any]:
    from scipy import stats

    out: dict[str, Any] = {"n_reps": len(rows)}
    for name in models:
        e = np.array([r["models"][name]["mean"] / r["true_level"] - 1 for r in rows])
        cov90 = np.mean([r["models"][name]["q05"] <= r["true_level"] <= r["models"][name]["q95"]
                         for r in rows])  # fmt: skip
        m: dict[str, Any] = {"rel_error_vs_truth": float(e.mean()), "coverage90": float(cov90)}
        if name != "oracle" and "oracle" in models:
            p = np.array([r["models"][name]["mean"] / r["models"]["oracle"]["mean"] - 1
                          for r in rows])  # fmt: skip
            if len(p) > 1:
                half = stats.t.ppf(0.975, len(p) - 1) * p.std(ddof=1) / np.sqrt(len(p))
            else:
                half = float("nan")
            ci = [float(p.mean() - half), float(p.mean() + half)]
            m["rel_diff_vs_oracle"] = {"mean": float(p.mean()), "ci95": ci}
        out[name] = m
    if declared and "M1d" in out and "rel_diff_vs_oracle" in out["M1d"]:
        d = out["M1d"]["rel_diff_vs_oracle"]
        out["acceptance_M1d"] = {
            "rule": "mean relative difference vs oracle within +-1% and its 95% CI includes 0 "
            "(plan.md, Step 5 revision, declared 2026-11-17)",
            "passed": bool(abs(d["mean"]) <= 0.01 and d["ci95"][0] <= 0 <= d["ci95"][1]),
        }
    return out


def main() -> None:
    from sctwin.data import load_split
    from sctwin.panel import build_panel
    from sctwin.splits import read_manifest

    cfg = load_config(ARGS.config)
    settings = {"reps": ARGS.reps, "models": ARGS.models, "warmup": ARGS.warmup,
                "samples": ARGS.samples, "chains": ARGS.chains, "smoke": ARGS.smoke,
                "stock": ARGS.stock, "n_series": cfg["n_series"], "seed": cfg["seed"],
                "truth": cfg["recovery"]["truth"],
                "injection": cfg["benchmark"]["injection"]}  # fmt: skip
    out = Path("runs/level_recovery") / config_hash(settings)
    out.mkdir(parents=True, exist_ok=True)
    models = ARGS.models.split(",")
    declared = ARGS.stock == "day_effect"  # the acceptance rule applies to this setting only

    manifest = read_manifest(cfg["manifest"])
    panel = build_panel(load_split(cfg["data_dir"], "train"))
    g = manifest["group_holdout"]
    hold = panel.keys["store_id"].is_in(g["stores"]) | panel.keys["product_id"].is_in(g["products"])
    rng = np.random.default_rng(cfg["seed"])  # same series selection as scripts/run_bayes.py
    series = np.sort(rng.permutation(np.flatnonzero(~hold.to_numpy()))[: cfg["n_series"]])
    if ARGS.smoke:
        series = series[:4]
    n_fit = manifest["temporal"]["fit"]["n_days"]
    weekday = np.array([d.weekday() for d in panel.dates])
    cov = {"dow": weekday[:n_fit].astype(np.int32),
           "x_disc": dm.discount_feature(panel.discount[series, :n_fit]),
           "x_act": panel.activity[series, :n_fit].astype(np.float64)}  # fmt: skip
    del panel

    truth_cfg = {**cfg["recovery"]["truth"], "rho": 0.0}
    mcmc = {"num_warmup": ARGS.warmup, "num_samples": ARGS.samples, "num_chains": ARGS.chains,
            "target_accept": cfg["mcmc"]["target_accept"],
            "max_tree_depth": cfg["mcmc"]["max_tree_depth"]}  # fmt: skip
    workers = ARGS.workers or max(1, (os.cpu_count() or 1) // ARGS.chains)
    workers = min(workers, ARGS.reps)
    print(f"{ARGS.reps} replications x {models} on {series.size} series x {n_fit} days, "
          f"stock {ARGS.stock}; "
          f"{workers} workers x {ARGS.chains} chains; out {out}", flush=True)  # fmt: skip

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one_rep, rep, cov, truth_cfg, cfg["benchmark"]["injection"], models,
                            mcmc, ARGS.stock == "day_effect")
                for rep in range(ARGS.reps)]  # fmt: skip
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            rows.sort(key=lambda r: r["rep"])
            msg = "  ".join(f"{k} {v['mean'] / row['true_level'] - 1:+.2%}"
                            for k, v in row["models"].items())  # fmt: skip
            print(f"rep {row['rep']}: {msg}  ({time.perf_counter() - t0:.0f} s)", flush=True)
            report = {"settings": settings, "series": series.tolist(),
                      "summary": summarise(rows, models, declared),
                      "replications": rows}  # fmt: skip
            (out / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_run_manifest(out, settings, stage="05_level_recovery", smoke=ARGS.smoke)
    print(json.dumps(summarise(rows, models, declared), indent=1))


if __name__ == "__main__":
    main()
