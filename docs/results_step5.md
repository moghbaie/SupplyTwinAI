# Step 5 results: Bayesian latent-demand model ladder

Run on Colab, 2026-11-14 (commit d7ec4b4, `notebooks/supply_twin_colab.ipynb`, full mode, 12 cores):
- the ladder: `runs/step5/c1b8de3a7091` (`scripts/run_bayes.py --part all`, `configs/bayes.yaml`
  with 4 chains);
- SBC: `runs/sbc/4dcdf65898da` (`scripts/run_sbc.py --reps 200`).

The paired bootstrap tables were not printed in that notebook run. They come from an earlier
local run with the same settings for M0–M2 only (`runs/step5/4a52d18919b5`). Its daily MAEs
match the Colab run to within 0.001.

**The "truth" on real data is the Step 4 proxy:** real sales on fully in-stock validation days,
with stockouts injected. It is not observed demand. True demand is known only in the synthetic
recovery and SBC experiments.

**Status: Step 5 is not accepted yet.** Sampling is clean and the censored dynamic model fits the
in-stock data best. However, it fails the targets declared in Step 4 (see
`docs/results_step4.md`). There are also three open problems: the latent level is biased, a
discount effect is not identified, and 17% of SBC replications diverge.

## Design

- **Model** (`sctwin.demand_model`): a hurdle-lognormal model of hourly demand in hours 6–21.
  - Effects: series, hour, series × hour, weekday, discount and activity, plus a day effect.
  - The likelihood is continuous because sales are normalized (`docs/data_card.md`).
- **Each hour's availability class decides how it enters the likelihood:**

| Model | In stock (A) | Depletion hour (P_pre) | First flagged hour (P_first) | Unavailable (U) | Day effect |
|---|---|---|---|---|---|
| M0 | exact | exact | dropped | dropped | i.i.d. |
| M1 | exact | censored (D ≥ y) | censored | dropped | i.i.d. |
| M2 | exact | censored | censored | dropped | AR(1) |
| M1d* | exact | censored | dropped | dropped | i.i.d. |
| M2d* | exact | censored | dropped | dropped | AR(1) |

  \* M1d and M2d were added after the first recovery run and were **not pre-declared**. They are
  exploratory.
- **Data.** 20 seeded in-sample series, fitted on days 0–75 (the fit window).
- **Benchmark.** Built with the Step 4 mechanism and calibrated injection on the validation days
  of the same 20 series: 174 days, 71 of them with a stockout. These are 20 of the 5,000 Step 4
  series, so the baselines' numbers here differ from Step 4.
- **No leakage.** Each censored day is reconstructed as of its end:
  - global parameters come from the posterior fitted on earlier days;
  - the day's own effect is inferred from that day's censored hours only.
- **Sampler.** NUTS with 4 chains × 1,000 warmup + 1,000 draws, target acceptance 0.9.

## Sampler diagnostics (real data)

| Model | Runtime (s) | Divergences | Max R-hat | Min ESS |
|---|---|---|---|---|
| M0 | 666 | 0 | 1.022 | 251 |
| M1 | 596 | 0 | 1.011 | 286 |
| M2 | 2,703 | 0 | 1.014 | 243 |
| M1d | 606 | 0 | 1.023 | 292 |
| M2d | 2,856 | 0 | 1.029 | 304 |

There are no divergences, and the ESS is adequate. Max R-hat is above the strict 1.01 threshold
for every model; it comes from the global level and hour-profile parameters. Longer chains are
needed before final runs.

## Fit to in-stock hours

**PSIS-LOO** is computed on the 18,618 in-stock hours. Differences are relative to the best model,
M2d:

| Model | elpd difference | SE |
|---|---|---|
| M2d | 0 | — |
| M2 | −8.3 | 1.7 |
| M0 | −47.9 | 16.3 |
| M1d | −74.6 | 14.4 |
| M1 | −75.3 | 14.6 |

The AR(1) day effect is the largest improvement. The posterior ρ on real data is about 0.86 (M2,
local run). Among the dynamic models, dropping P_first (M2d) beats censoring it (M2).

**Posterior predictive checks** on in-stock hours (local run, M0–M2) show two clear misfits:
- the share of zero-sales hours is under-predicted: 0.638 observed vs 0.62–0.63 replicated
  (PPC p ≈ 0);
- the 99th percentile is under-predicted: 0.5 observed vs 0.43–0.47 replicated.

The mean and the 90th percentile are reproduced (censored models). The hurdle and the lognormal
tail need revisiting.

## Reconstruction of stockout days (proxy truth)

**Daily totals, hours 6–21, 71 stockout days:**

| Method | MAE | Bias | WIS | Coverage 50% | Coverage 90% |
|---|---|---|---|---|---|
| hour_mean | **0.146** | −0.092 | — | — | — |
| M2d* | 0.153 | +0.052 | **0.095** | 0.521 | 0.831 |
| M0 | 0.160 | −0.063 | 0.097 | 0.507 | 0.859 |
| M1d* | 0.165 | +0.068 | 0.100 | 0.465 | 0.845 |
| M2 | 0.170 | +0.094 | 0.099 | 0.423 | 0.859 |
| M1 | 0.184 | +0.109 | 0.104 | 0.366 | 0.845 |
| profile_pre | 0.229 | −0.009 | — | — | — |
| observed | 0.346 | −0.346 | — | — | — |
| profile | 0.398 | +0.204 | — | — | — |

The 80% coverage of M0–M2 is 0.75–0.78 (local run).

**Paired cluster bootstrap** (20 series clusters, 2,000 resamples, Holm-corrected; local run,
M0–M2):
- **Daily absolute error vs `hour_mean`:**
  - M0 +9%, M1 +26%, M2 +16%; all p_holm = 0.69 (no detectable difference);
  - `profile_pre` is worse (+57%, p_holm = 0.02);
  - `observed` is worse (+137%, p_holm = 0.003).
- **Hourly absolute error vs `hour_mean`** on censored hours: M0 −14%, M1 −25%, M2 −27%; all
  p_holm = 0.003. The censored models are clearly better hour by hour.
- **The depletion hour (P_pre, mean true sales 0.196):**
  - censoring it cuts the bias from −0.095 (M0, `hour_mean`) to −0.018 (M1, M2);
  - MAE drops from 0.095 to 0.060.
- **Daily WIS of M1 and M2 vs M0:** not significant (p_holm = 1.0).

**Against the targets declared in Step 4:**

| Target | Result | Verdict |
|---|---|---|
| Daily MAE below `hour_mean` on the same rows (0.146 here) | best 0.153 (M2d*), 0.160 (M0) | **not met** |
| \|bias\| no worse than `profile` (0.037 in Step 4) | best 0.052 (M2d*) | **not met** |
| Nominal coverage of 50/80/90% intervals | 0.37–0.52 / 0.75–0.78 / 0.83–0.86 | **not met** (too narrow) |

These are **negative results** for this configuration and are reported as such. The test has
little power: with 20 clusters and 71 stockout days, the daily confidence intervals span about
±0.05. Even so, no pre-declared Bayesian model beats `hour_mean` on daily totals.

Censoring helps exactly where it should: the depletion hour, and hourly error overall. However,
M1 and M2 then over-shoot daily totals (+0.09 to +0.11). The likely cause is that the day effect
is inferred from a few censored hours and pushed upward by them; P_first adds to this. Dropping
P_first (M1d, M2d) halves the over-shoot.

## Parameter recovery (synthetic, known truth)

Setup:
- Demand is simulated from M2's structure (ρ = 0.5) on the real covariates and calendar of the
  same 20 series.
- Stockouts are injected with the Step 4 mechanism (46.6% of days have a stockout).
- The ladder is fitted on the result.
- The true mean demand over hours 6–21 is 3.913.

| Model | Scalars covered by the 90% CI | Mean window demand [90% CI] | Error |
|---|---|---|---|
| M0 | 6/9 | 3.782 [3.693, 3.865] | −3.3% |
| M1 | 6/9 | 4.124 [4.029, 4.219] | +5.4% |
| M2 | 9/10 | 4.085 [4.007, 4.178] | +4.4% |
| M1d* | 7/9 | 4.038 [3.955, 4.127] | +3.2% |
| M2d* | **10/10** | 4.011 [3.922, 4.100] | +2.5% |

- **The demand level is biased for every model,** and no interval covers the true value. This is
  the same pattern as on real data: M0 is too low, and the censored models are too high.
- **The censored models' level is too high even when the model is correctly specified.** This
  points at the censoring treatment (P_first, and the leak in flagged hours that the likelihood
  does not model), not at model misspecification.
- **σ_u is over-estimated** by the i.i.d. models: 0.42–0.44 vs a true 0.35. These models absorb
  the AR(1) persistence into day-to-day noise.
- **b_disc is not identified:** posterior means are 0.08–0.15 vs a true 0.8, and the intervals
  run from about −0.6 to 0.85. Discounts are rare in 20 series × 76 days. "Covered" here only
  reflects how wide the interval is.

## Simulation-based calibration (M1)

Setup:
- 200 replications of 4 series × 14 days.
- Parameters are drawn from the prior; stockouts are injected without leak and with the stock
  independent of the parameters, so the censored likelihood is exact.
- Each replication uses 1 chain of 400 + 400 draws, thinned to 100.

| Parameter | χ² p | Parameter | χ² p |
|---|---|---|---|
| alpha | 0.021 | b_act | 0.627 |
| sigma_a | 0.401 | sigma_y | 0.130 |
| tau_g | 0.276 | k0 | 0.475 |
| sigma_u | 0.679 | k1 | 0.437 |
| b_disc | 0.225 | | |

- **No calibration failure is detected:** the smallest Holm-adjusted p is 0.185. The censored
  likelihood itself is therefore not the source of the level bias above; the cause lies in what
  the likelihood leaves out, namely leak and P_first.
- **35 of 200 replications (17.5%) had divergent transitions.** SBC ranks from divergent fits are
  unreliable, so this result is conditional on the sampler. alpha is borderline before
  correction.
- **The run is not reproducible bit for bit across machines.** A local run with the same
  configuration hash gave 31 divergent replications and a smallest Holm p of 0.498 (JAX CPU
  results are not deterministic across platforms). Both runs reach the same conclusion.

## Findings

1. **Censoring works where the information is.** It removes most of the depletion-hour bias and
   cuts hourly error by about 25% vs `hour_mean` (p_holm = 0.003).
2. **It does not yet improve daily totals.** No pre-declared model beats `hour_mean` on daily MAE,
   and the censored models over-state stockout-day demand by 7–11%.
3. **Uncertainty is too small.** Daily intervals cover 83–86% at the nominal 90%, and about 40%
   at the nominal 50% for the censored models.
4. **Persistent day effects matter.** AR(1) is the clear LOO winner, with ρ ≈ 0.86 on real data.
5. **The level bias appears even when the model is correctly specified,** so it is a modelling
   problem, not just a data-mismatch problem.

## Remaining work before Step 5 is accepted

1. **Fix the level bias.**
   - Model the leak in flagged hours: sales seen after a flag currently enter as "dropped" or as a
     lower bound.
   - Treat P_first consistently. M2d's choice should be decided on held-out data, not on these
     results.
   - Re-check recovery after the change.
2. **Widen the intervals.** Propagate the day effect's uncertainty fully in reconstruction, and
   check whether σ_u is too tight.
3. **Fix the PPC misfits.** The share of zero hours and the 99th percentile are under-predicted;
   consider a hurdle that depends on the hour and series, or a heavier tail.
4. **Fix the SBC divergences.** Raise target acceptance or reparameterize the hurdle (k0, k1),
   then rerun.
5. **Make `b_disc` estimable.** Use more series (M3: 100 series) or a hierarchical prior; until
   then, report discount effects as unidentified.
6. **Rerun the final comparisons.** Use longer chains (R-hat ≤ 1.01) and the bootstrap tables now
   added to the notebook. Declare whether M2d replaces M2 before that run.
