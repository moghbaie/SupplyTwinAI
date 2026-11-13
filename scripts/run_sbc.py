"""Step 5: simulation-based calibration (SBC) of the censored model M1.

Usage: python scripts/run_sbc.py [--reps 200] [--n-series 4] [--n-days 14] [--seed 1]

Each replication draws all parameters from the prior, simulates hourly demand, censors it with
the Step 4 mechanism and fits M1. If inference is correct, the rank of every true scalar among
the (thinned) posterior draws is uniform. For the censored likelihood to be exact here, the
opening stock is drawn independently of the parameters and flagged hours sell nothing
(leak = 0). Writes runs/sbc/<hash>/{ranks.json, run_manifest.json}.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--n-series", type=int, default=4)
ap.add_argument("--n-days", type=int, default=14)
ap.add_argument("--warmup", type=int, default=400)
ap.add_argument("--samples", type=int, default=400)
ap.add_argument("--thin", type=int, default=4)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--model", default="M1")
args = ap.parse_args()

import jax  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from scipy import stats  # noqa: E402

from sctwin import demand_model as dm  # noqa: E402
from sctwin.censoring import HI, LO, CensorConfig, inject_stockouts  # noqa: E402
from sctwin.panel import availability_classes  # noqa: E402
from sctwin.repro import config_hash, write_run_manifest  # noqa: E402

SCAL = [s for s in dm.SCALARS] + (["rho"] if dm.LADDER[args.model].dynamic else [])


def main() -> None:
    cfg = vars(args)
    out = Path("runs/sbc") / config_hash(cfg)
    out.mkdir(parents=True, exist_ok=True)
    spec = dm.LADDER[args.model]
    rng = np.random.default_rng(args.seed)
    n_s, n_d = args.n_series, args.n_days
    inj_cfg = CensorConfig(leak_first=0.0, leak_later=0.0, p_restock=0.187, restock_mean_hours=5.75)
    model = functools.partial(dm.demand_model, dynamic=spec.dynamic)
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=args.warmup,
                num_samples=args.samples, num_chains=1, progress_bar=False,
                jit_model_args=True)  # fmt: skip
    ranks: dict[str, list[int]] = {k: [] for k in SCAL}
    log = []
    t0 = time.perf_counter()
    for r in range(args.reps):
        x_disc = np.where(rng.random((n_s, n_d)) < 0.3, rng.uniform(0, 0.4, (n_s, n_d)), 0.0)
        x_act = (rng.random((n_s, n_d)) < 0.2).astype(np.float64)
        dow = np.arange(n_d) % 7
        theta = dm.prior_draw(n_s, n_d, spec.dynamic, int(rng.integers(2**31)), x_disc, x_act, dow)
        demand = dm.simulate(theta, dow, x_disc, x_act, seed=int(rng.integers(2**31)))
        full = np.zeros((n_s * n_d, 24))
        full[:, LO:HI] = demand.reshape(-1, dm.NH)
        stock = np.exp(rng.normal(np.log(2.0), 1.0, n_s * n_d))  # independent of theta
        inj = inject_stockouts(
            full, stock, np.random.default_rng(int(rng.integers(2**31))), inj_cfg
        )
        data = dm.ModelData(
            y=inj.observed[:, LO:HI].reshape(n_s, n_d, dm.NH),
            cls=availability_classes(inj.flag)[:, LO:HI].reshape(n_s, n_d, dm.NH),
            dow=dow, x_disc=x_disc, x_act=x_act, series=np.arange(n_s), days=np.arange(n_d),
        )  # fmt: skip
        a = dm.model_args(data, spec)
        a.pop("dynamic")
        mcmc.run(jax.random.PRNGKey(int(rng.integers(2**31))), **a, extra_fields=("diverging",))
        s = {k: np.asarray(v)[:: args.thin] for k, v in mcmc.get_samples().items()}
        div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
        for k in SCAL:
            ranks[k].append(int((s[k] < theta[k]).sum()))
        log.append({"rep": r, "divergences": div, "stockout_rate": float(inj.stocked_out.mean())})
        if (r + 1) % 10 == 0:
            print(f"rep {r + 1}/{args.reps}  {time.perf_counter() - t0:.0f} s", flush=True)

    n_post = len(range(0, args.samples, args.thin))
    bins = 10
    res = {}
    for k, v in ranks.items():
        h = np.histogram(v, bins=bins, range=(0, n_post + 1))[0]
        res[k] = {"hist": h.tolist(), "chi2_p": float(stats.chisquare(h).pvalue),
                  "mean_rank_frac": float(np.mean(v) / n_post)}  # fmt: skip
    p_holm = np.minimum(
        1.0, np.sort([r["chi2_p"] for r in res.values()]) * np.arange(len(res), 0, -1)
    )
    report = {"config": cfg, "n_post_draws": n_post, "scalars": res, "ranks": ranks,
              "min_p_holm": float(np.min(np.maximum.accumulate(p_holm))),
              "divergent_reps": int(sum(x["divergences"] > 0 for x in log)),
              "mean_stockout_rate": float(np.mean([x["stockout_rate"] for x in log])),
              "runtime_s": round(time.perf_counter() - t0, 1)}  # fmt: skip
    (out / "ranks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_run_manifest(out, cfg, stage="05_sbc")
    for k, r in res.items():
        print(f"  {k:8s} chi2 p={r['chi2_p']:.3f} mean rank={r['mean_rank_frac']:.2f} {r['hist']}")
    print(f"min Holm-adjusted p = {report['min_p_holm']:.3f}; written to {out}")


if __name__ == "__main__":
    main()
