"""Step 5: fit the Bayesian demand-model ladder and evaluate it.

Usage:
  python scripts/run_bayes.py [--config configs/bayes.yaml] [--models M0,M1,M2]
                              [--n-series 20] [--part real|recovery|all] [--smoke]

real      fit each model on the fit window (days 0..75) of n seeded series; diagnostics
          (R-hat, ESS, divergences), posterior predictive checks, PSIS-LOO on in-stock hours;
          reconstruct the Step 4 benchmark days of those series (validation days, all after the
          fitted window) and score them against the baselines on the same rows.
recovery  simulate from the model with known parameters on the same series' covariates and
          calendar, inject stockouts with the Step 4 mechanism, fit the ladder, check that the
          90% credible intervals cover the true values.

Writes runs/step5/<config_hash>/{metrics.json, posterior_<model>.npz, run_manifest.json}.
Only the official train split is loaded; the locked test split is never read here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bayes.yaml")
    ap.add_argument("--models", default=None, help="comma-separated, e.g. M0,M1,M2")
    ap.add_argument("--n-series", type=int, default=None)
    ap.add_argument("--part", choices=["real", "recovery", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


ARGS = parse()

import numpyro  # noqa: E402  (host devices must be set before JAX initialises)

from sctwin.repro import config_hash, load_config, seed_everything, write_run_manifest  # noqa: E402

CFG = load_config(ARGS.config)
if ARGS.smoke:
    CFG.update(n_series=4, out_dir="runs/step5_smoke")
    CFG["mcmc"].update(num_warmup=150, num_samples=100, num_chains=2)
    CFG["reconstruction"]["n_draws"] = 50
    CFG["checks"].update(ppc_draws=20, loo_draws=100)
    CFG["bootstrap"]["n_boot"] = 200
if ARGS.models:
    CFG["models"] = ARGS.models.split(",")
if ARGS.n_series:
    CFG["n_series"] = ARGS.n_series
numpyro.set_host_device_count(CFG["mcmc"]["num_chains"])

import arviz as az  # noqa: E402

from sctwin import demand_model as dm  # noqa: E402
from sctwin.censoring import (  # noqa: E402
    HI,
    IMPUTERS,
    LO,
    CensorConfig,
    build_benchmark,
    candidate_days,
    hour_profile,
    inject_stockouts,
    recovery_frames,
    stock_levels,
)
from sctwin.data import load_split  # noqa: E402
from sctwin.evaluation import compare_to_reference, evaluate, qcol  # noqa: E402
from sctwin.panel import Panel, availability_classes, build_panel  # noqa: E402
from sctwin.splits import read_manifest  # noqa: E402


def mcmc_kwargs(seed: int) -> dict[str, Any]:
    m = CFG["mcmc"]
    return {"num_warmup": m["num_warmup"], "num_samples": m["num_samples"],
            "num_chains": m["num_chains"], "target_accept": m["target_accept"],
            "max_tree_depth": m["max_tree_depth"], "seed": seed}  # fmt: skip


def loo(ll: np.ndarray) -> Any:
    return az.loo(az.from_dict({"posterior": {"x": np.zeros((1, ll.shape[0]))},
                                "log_likelihood": {"y": ll[None]}}))  # fmt: skip


def save_posterior(path: Path, res: dm.FitResult) -> None:
    keep = {k: v.astype(np.float32) for k, v in res.samples.items()
            if not k.endswith("_raw") and k != "u"}  # fmt: skip
    keep["u_last"] = res.samples["u"][:, :, -1].astype(np.float32)  # state at the window end
    np.savez_compressed(path, **keep)


# --------------------------------------------------------------------------- real data


def run_real(panel: Panel, series: np.ndarray, weekday: np.ndarray, n_fit: int,
             out: Path) -> dict[str, Any]:  # fmt: skip
    sub = panel.subset(series)
    n_days = panel.shape[1]
    s_local = np.arange(series.size)
    fit_days = np.arange(n_fit)
    data = dm.make_model_data(sub.sales, sub.stock, sub.discount, sub.activity, weekday,
                              s_local, fit_days)  # fmt: skip

    b = CFG["benchmark"]
    rows = candidate_days(sub.sales, sub.stock, s_local, np.arange(n_fit, n_days), b["window"])
    bm = build_benchmark(sub.sales, rows, CensorConfig(**b["injection"]),
                         seed=CFG["seed"] + b["seed_offset"])  # fmt: skip
    if bm.day.min() <= fit_days.max():
        raise AssertionError("benchmark days must be after the fitted window (Rule 1)")
    inj = bm.injection
    so = np.flatnonzero(inj.stocked_out)
    day_rows = dm.DayRows(
        s_pos=bm.series[so], day=bm.day[so], y=inj.observed[so, LO:HI],
        cls=availability_classes(inj.flag[so])[:, LO:HI], dow=weekday[bm.day[so]],
        x_disc=dm.discount_feature(sub.discount[bm.series[so], bm.day[so]]),
        x_act=sub.activity[bm.series[so], bm.day[so]].astype(np.float64),
    )  # fmt: skip
    print(f"fit: {series.size} series x {n_fit} days; benchmark: {bm.series.size} days, "
          f"{so.size} with a stockout", flush=True)  # fmt: skip

    est = {m: f(inj.observed, inj.flag, bm.hour_means) for m, f in IMPUTERS.items()}
    totals: dict[str, np.ndarray] = {}
    fits: dict[str, Any] = {}
    lls: dict[str, np.ndarray] = {}
    for k, name in enumerate(CFG["models"]):
        spec = dm.LADDER[name]
        print(f"[{name}] fitting ...", flush=True)
        res = dm.fit(data, spec, **mcmc_kwargs(CFG["seed"] + k))
        d = res.diagnostics
        print(f"[{name}] {d['runtime_s']} s, divergences={d['divergences']}, "
              f"max R-hat={d['max_rhat']:.3f}, min ESS={d['min_ess']:.0f}", flush=True)  # fmt: skip
        save_posterior(out / f"posterior_{name}.npz", res)
        lls[name] = dm.pointwise_loglik(res, data, max_draws=CFG["checks"]["loo_draws"])
        lo = loo(lls[name])
        rc = CFG["reconstruction"]
        draws = dm.reconstruct(res, data, day_rows, n_draws=rc["n_draws"], grid=rc["grid"],
                               seed=CFG["seed"])  # fmt: skip
        hourly_est = inj.observed.copy()
        hourly_est[so, LO:HI] = np.median(draws, axis=1)
        est[name] = hourly_est
        totals[name] = draws.sum(axis=-1)  # (R, n_draws) hours 6-21
        fits[name] = {
            "diagnostics": d,
            "ppc_in_stock_hours": dm.predictive_checks(
                res, data, CFG["checks"]["ppc_draws"], seed=CFG["seed"]
            ),  # fmt: skip
            "loo_in_stock_hours": {
                "elpd": float(lo.elpd),
                "se": float(lo.se),
                "p_loo": float(lo.p),
                "pareto_k_gt_0.7": int((np.asarray(lo.pareto_k) > 0.7).sum()),
                "n_obs": int(lls[name].shape[1]),
            },  # fmt: skip
        }

    hourly, daily = recovery_frames(bm, est)
    taus = tuple(CFG["taus"])
    for name, t in totals.items():
        q = np.quantile(t, taus, axis=1).T
        daily = daily.with_columns(
            **{f"{name}__point": np.quantile(t, 0.5, axis=1)},
            **{qcol(name, tau): q[:, i] for i, tau in enumerate(taus)},
        )
    bs = CFG["bootstrap"]
    comp = {}
    for ref in bs["references"]:
        comp[f"daily_ae_vs_{ref}"] = compare_to_reference(daily, ref, "ae", taus, bs["n_boot"],
                                                          CFG["seed"])  # fmt: skip
        comp[f"hourly_ae_vs_{ref}"] = compare_to_reference(hourly, ref, "ae", (0.5,),
                                                           bs["n_boot"], CFG["seed"])  # fmt: skip
    if len(totals) > 1 and "M0" in totals:
        comp["daily_wis_vs_M0"] = compare_to_reference(daily, "M0", "wis", taus, bs["n_boot"],
                                                       CFG["seed"])  # fmt: skip
    names = list(lls)
    best = max(names, key=lambda n: fits[n]["loo_in_stock_hours"]["elpd"])
    loo_cmp = {}
    for n in names:  # paired pointwise differences (same in-stock observations)
        da = np.asarray(loo(lls[n]).elpd_i) - np.asarray(loo(lls[best]).elpd_i)
        loo_cmp[n] = {"elpd_diff": float(da.sum()), "se_diff": float(np.sqrt(da.size) * da.std())}
    return {
        "truth": "demand proxy = real sales on fully in-stock validation days (Step 4)",
        "n_series": int(series.size), "fit_days": [0, int(n_fit - 1)],
        "benchmark_days": [int(bm.day.min()), int(bm.day.max())],
        "n_benchmark_days": int(bm.series.size), "n_stockout_days": int(so.size),
        "models": fits, "loo_compare": {"best": best, "models": loo_cmp},
        "daily": evaluate(daily, taus), "hourly": evaluate(hourly, (0.5,)),
        "comparisons": comp,
    }  # fmt: skip


# --------------------------------------------------------------------------- recovery


def run_recovery(panel: Panel, series: np.ndarray, weekday: np.ndarray, n_fit: int) -> dict:
    sub = panel.subset(series)
    n_s, days = series.size, np.arange(n_fit)
    rng = np.random.default_rng(CFG["seed"] + 7)
    t = dict(CFG["recovery"]["truth"])
    x_disc = dm.discount_feature(sub.discount[:, :n_fit])
    x_act = sub.activity[:, :n_fit].astype(np.float64)
    raw_g = rng.standard_normal((n_s, dm.NH))
    raw_u = rng.standard_normal((n_s, n_fit))
    rho = t["rho"] if CFG["recovery"]["dynamic_truth"] else 0.0
    truth = {
        **t, "a": t["sigma_a"] * rng.standard_normal(n_s),
        "gamma": dm._centred(np.log(hour_profile()[LO:HI])),
        "g": t["tau_g"] * np.asarray(dm._centred(dm._centred(raw_g, -1), 0)),
        "dow_eff": dm._centred(np.array([0.0, -0.05, -0.05, 0.0, 0.1, 0.2, 0.15])),
        "u": np.asarray(dm.ar1(raw_u, rho, t["sigma_u"])),
    }  # fmt: skip
    dow = weekday[days]
    demand = dm.simulate(truth, dow, x_disc, x_act, seed=CFG["seed"] + 8)
    eta = np.asarray(dm.linear_predictor(truth, dow, x_disc, x_act)) + truth["u"][:, :, None]
    full = np.zeros((n_s * n_fit, 24))
    full[:, LO:HI] = demand.reshape(-1, dm.NH)
    cfg = CensorConfig(**CFG["benchmark"]["injection"])
    r2 = np.random.default_rng(CFG["seed"] + 9)
    k = stock_levels(np.exp(eta).sum(-1).ravel(), r2, cfg)
    inj = inject_stockouts(full, k, r2, cfg)
    data = dm.ModelData(
        y=inj.observed[:, LO:HI].reshape(n_s, n_fit, dm.NH),
        cls=availability_classes(inj.flag)[:, LO:HI].reshape(n_s, n_fit, dm.NH),
        dow=dow, x_disc=x_disc, x_act=x_act, series=series, days=days,
    )  # fmt: skip
    true_level = float(np.exp(eta).sum(-1).mean())
    out: dict[str, Any] = {"truth": t, "stockout_day_rate": float(inj.stocked_out.mean()),
                           "true_mean_window_demand": true_level, "models": {}}  # fmt: skip
    for j, name in enumerate(CFG["models"]):
        res = dm.fit(data, dm.LADDER[name], **mcmc_kwargs(CFG["seed"] + 100 + j))
        s = res.samples
        rows = {}
        for p in (*dm.SCALARS, "rho"):
            if p not in s or p not in t:
                continue
            lo, hi = np.quantile(s[p], [0.05, 0.95])
            rows[p] = {"true": t[p], "mean": float(s[p].mean()), "q05": float(lo),
                       "q95": float(hi), "covered": bool(lo <= t[p] <= hi)}  # fmt: skip
        # posterior mean of the expected hours 6-21 demand per day, averaged (level recovery)
        idx = np.linspace(0, len(s["alpha"]) - 1, 200).astype(int)
        lv = []
        for i in idx:
            p = {k: s[k][i] for k in ("alpha", "a", "gamma", "g", "dow_eff", "b_disc", "b_act")}
            e = np.asarray(dm.linear_predictor(p, dow, x_disc, x_act)) + s["u"][i][:, :, None]
            lv.append(np.exp(e).sum(-1).mean())
        lo, hi = np.quantile(lv, [0.05, 0.95])
        out["models"][name] = {
            "diagnostics": {k: res.diagnostics[k]
                            for k in ("runtime_s", "divergences", "max_rhat", "min_ess")},
            "scalars": rows,
            "mean_window_demand": {"mean": float(np.mean(lv)), "q05": float(lo), "q95": float(hi),
                                   "covered": bool(lo <= true_level <= hi)},
        }  # fmt: skip
        print(
            f"[recovery {name}] level {np.mean(lv):.3f} vs true {true_level:.3f}; covered "
            f"{sum(r['covered'] for r in rows.values())}/{len(rows)} scalars",
            flush=True,
        )
    return out


def main() -> None:
    rng = seed_everything(CFG["seed"])
    out = Path(CFG["out_dir"]) / config_hash(CFG)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    manifest = read_manifest(CFG["manifest"])
    panel = build_panel(load_split(CFG["data_dir"], "train"))
    g = manifest["group_holdout"]
    hold = panel.keys["store_id"].is_in(g["stores"]) | panel.keys["product_id"].is_in(g["products"])
    in_sample = np.flatnonzero(~hold.to_numpy())
    series = np.sort(rng.permutation(in_sample)[: CFG["n_series"]])  # prefix: 20 within 100
    weekday = np.array([d.weekday() for d in panel.dates])
    n_fit = manifest["temporal"]["fit"]["n_days"]

    report: dict[str, Any] = {"config": CFG, "series": series.tolist()}
    if ARGS.part in ("real", "all"):
        report["real"] = run_real(panel, series, weekday, n_fit, out)
    if ARGS.part in ("recovery", "all"):
        report["recovery"] = run_recovery(panel, series, weekday, n_fit)
    report["runtime_s"] = round(time.perf_counter() - t0, 1)
    (out / f"metrics_{ARGS.part}.json").write_text(json.dumps(report, indent=2, default=str),
                                                   encoding="utf-8")  # fmt: skip
    write_run_manifest(out, CFG, stage="05_bayes", part=ARGS.part, smoke=ARGS.smoke)

    if "real" in report:
        dly = report["real"]["daily"]["overall"]["models"]
        print("\n== daily totals on stockout days (proxy truth)")
        for m, r in dly.items():
            cov = r.get("intervals", {}).get("0.9", {}).get("coverage", float("nan"))
            print(f"  {m:12s} mae={r['mae']:.4f} bias={r['bias']:+.4f} cov90={cov:.3f}")
    print(f"\nwritten to {out}  ({report['runtime_s']} s)")


if __name__ == "__main__":
    main()
