# Step 3 results: baseline forecasters (validation backtest)

Run on 2026-11-12. Scores are on **observed sales** (normalized units) in the validation window
(days 76–89 of `train`). The official `eval` split was not loaded. Stockout-affected targets are
scored against censored sales, so their errors measure fit to what was sold, not to demand.

## Protocol (CLAUDE.md Rules 1 and 2)

| | day | hour |
|---|---|---|
| Lead L / horizon H / step S | 1 / 7 / 7 days | 1 / 24 / 24 hours |
| Training window | sliding, W = 42 days | sliding, W = 504 h, every 6th hour |
| Origins (folds) | 74, 81 (2 folds) | 1822 … 2134, 22:00 each day (14 folds) |
| Training feature times | t ≤ T − 8 | t ≤ T − 25 |
| Series fitted / scored | 12,000 / 43,883 in-sample | 500 / 500 (seeded sample) |
| Run folder | `runs/step3/day/1cdd2e3f9599` | `runs/step3/hour/4c76dcfb809e` |

The logged maximum training feature time equals T − L − H in every fold of both units, and
`check_split` passed at every fold.

**Models:**
- `snaive`: the last value at the same weekday or hour.
- `clim`: empirical quantiles of the last values at the same seasonal position, up to T.
- `glm`: a Tweedie GLM (p = 1.5) whose quantiles come from the fitted compound Poisson–Gamma distribution.
- `lgbm`: a LightGBM Tweedie model giving a point forecast (the mean).
- `lgbm_q`: LightGBM quantile models, one per τ.

## Day

| Model | MAE | WAPE | WIS | 90% cov. | 50% cov. |
|---|---|---|---|---|---|
| snaive | 0.484 | — | — | — | — |
| clim (median) | 0.478 | 0.419 | 0.306 | 0.670 | 0.394 |
| glm | 0.418 | 0.367 | 0.251 | **0.888** | 0.537 |
| lgbm | 0.392 | 0.344 | — | — | — |
| lgbm_q (median) | **0.390** | — | **0.230** | 0.856 | 0.451 |

Per fold (WIS): clim 0.278 / 0.334, glm 0.236 / 0.266, lgbm_q 0.208 / 0.253 (origin 74 / 81).
The model ranking is the same in both folds.

**Paired cluster bootstrap** (43,883 series clusters, 2,000 resamples, Holm-corrected):
- WIS against clim:
  - glm −18.0% [CI of the difference −0.057, −0.053];
  - lgbm_q −24.8% [−0.078, −0.074];
  - both p_holm = 0.001.
- Absolute error against snaive:
  - glm −13.6%;
  - lgbm −19.1%;
  - both p_holm = 0.001.

**Subgroups (WIS, lgbm_q / glm / clim):**

| Subgroup | n | lgbm_q | glm | clim | lgbm_q 90% cov. |
|---|---|---|---|---|---|
| fully available day | 347k | 0.215 | 0.232 | 0.290 | 0.881 |
| stockout-affected day | 267k | 0.249 | 0.275 | 0.327 | 0.824 |
| promo | 252k | 0.247 | 0.262 | 0.331 | 0.832 |
| no promo | 362k | 0.218 | 0.243 | 0.289 | 0.873 |
| low volume | 154k | 0.140 | 0.146 | 0.180 | 0.869 |

Error rises from 2 to 8 steps ahead, with a weekday peak at steps 5–6. lgbm_q has the lowest WIS at every
step.

## Hour

| Model | MAE | WIS | 90% cov. | 50% cov. |
|---|---|---|---|---|
| snaive | 0.0561 | — | — | — |
| clim (median) | 0.0404 | 0.0258 | 0.945 | 0.862 |
| glm | 0.0520 | 0.0278 | 0.968 | 0.799 |
| lgbm | 0.0477 | — | — | — |
| lgbm_q (median) | **0.0400** | **0.0249** | 0.948 | 0.805 |

Mean hourly sales are 0.046, and most hours are zero. Mean forecasts (glm, lgbm) therefore lose
on MAE to median forecasts, and WAPE exceeds 1 for them.

lgbm_q has the lowest WIS in all 14 folds, but its margin over clim is small (at most 0.002).

**Paired cluster bootstrap** (500 series clusters):
- WIS against clim:
  - lgbm_q −3.4% (p_holm = 0.001);
  - glm **+7.8%**, i.e. worse than climatology (p_holm = 0.002).
- Absolute error against snaive:
  - glm −7.4%;
  - lgbm −15.1%.

**Availability classes (WIS, lgbm_q / glm / clim):**

| Class | n | lgbm_q | glm | clim | lgbm_q 90% cov. |
|---|---|---|---|---|---|
| A (available, day hours) | 87.6k | 0.037 | 0.043 | 0.039 | 0.934 |
| P_pre (hour before a stockout starts) | 2.9k | 0.089 | 0.099 | 0.094 | **0.742** |
| P_first (first stockout hour) | 2.8k | 0.024 | 0.021 | 0.025 | 0.988 |
| U (unavailable) | 18.7k | 0.014 | 0.012 | 0.014 | 0.994 |
| night | 56k | 0.006 | 0.006 | 0.006 | 0.964 |

## Findings

1. **Learned models beat seasonal baselines in both units.** The gain is large for days
   (−25% WIS) and small but significant for hours (−3%).
2. **Climatology intervals are too narrow for days.** clim's 90% interval covers only 67% of
   outcomes, because it uses only 8 past values. glm is closest to nominal (89%).
3. **lgbm_q under-covers when stock runs out.** Its 90% coverage is 82% on stockout-affected
   days and 74% on the P_pre hours, where sales are high just before stock runs out. These are
   the observations the censoring models of Step 4 must explain.
4. **Point models under-forecast.** For days, lgbm's bias is −0.087 and glm's is −0.050. A
   censoring-aware model would predict higher latent demand, but that cannot be scored against
   observed sales; it is evaluated on synthetic censoring with known truth (plan, Scoring).
5. **The hourly GLM is mis-specified.** A single Tweedie dispersion parameter cannot fit both
   night and daytime hours, so the GLM is worse than climatology. A Tweedie model with dispersion
   that depends on the hour is a candidate for the Bayesian model ladder (Step 5).

## Caveats

- The hourly results use 500 sampled series and are a smoke-scale estimate.
- Fold 1966 of the hourly run logged 3,085 s against about 130 s for the others. That is
  wall-clock time (the machine was most likely suspended), not a difference in compute.
- No hyperparameters were tuned. All models use fixed defaults (`configs/baselines.yaml`).
