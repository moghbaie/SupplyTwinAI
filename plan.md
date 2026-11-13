# Supply Chain Twin — Revised Implementation Plan (v2, 2026-11-12)

Supersedes the schedule in `plan_Supply_Twin_AI.md` (kept as the original brief).
The scientific boundaries of the original brief still apply: reconstructed demand is never
called observed ground truth, replay is never called live telemetry, synthetic costs are never
called retailer savings.

## Guiding principle

The first irreversible decision is **what FreshRetailNet actually measures**: how sales are
normalized and what the hourly stockout indicators encode. Likelihood choice, inference engine
and policy architecture are downstream of that and are decided only after the Step 1 audit.

## Data facts (verified on the Hub, 2026-11-12)

- `train`: 4.5M daily rows = 50k store-product series x 90 days. `eval`: 350k rows = the next 7 days.
- Each row is a **daily record with 24-element hourly arrays** (`hours_sale`, `hours_stock_status`).
  Storage stays compact (daily rows + arrays); hourly long format is materialized only on demand.
- Weather fields are daily averages; `stock_hour6_22_cnt` covers hours 6-22 only.
- Parquet total ~115 MB: fits on a laptop. Colab is needed only for large MCMC / meta-learning.

## Splits

- **Locked final test:** official `eval` (7 days after day 90). Read only through the test ledger.
- **Validation:** last 14 days of `train`. **Fit:** first 76 days of `train`.
- **Group hold-out:** disjoint stores (~10%) and products (~5%) removed from all fitting.

## Backtesting protocol (CLAUDE.md Rules 1 and 2, adopted 2026-11-12)

All forecasting evaluation uses rolling-origin backtesting in a configurable unit ("day" or
"hour"). Lead L, horizon H, origin step S and window W are in that unit (`configs/baselines.yaml`).

- **Origin T** = end of unit T. Features are computed as of T only. "Lag k" = unit T − k.
  Daily fields (weather, discount, activity) are used only from the last complete day at T.
  Target-time inputs are restricted to the calendar (steps ahead, weekday, hour, holiday flag).
  Target-time promotions are not features (their availability timestamp is unknown).
- **Training at T:** examples with feature time t ≤ T − L − H, i.e. full target windows observed
  by T. Sliding (length W) or expanding window. Scaling, imputation and models are fitted per fold.
  No hyperparameter tuning yet; any tuning must use a nested time split inside the window.
- **Prediction at T:** targets (T + L, T + L + H]. Metrics are reported per fold and aggregated.
- **Chosen settings.** Day: L = 1, H = 7, S = 7, W = 42 (2 folds covering validation days 76–89).
  Hour: L = 1, H = 24, S = 24, W = 504 with stride 6 (14 folds, origins at 22:00).
- **Timezone.** Day boundaries are the dataset's `dt` in store-local time, assumed Asia/Shanghai.
- **Enforcement.** `backtest.check_split` asserts the rules on every fold at run time.
  `tests/test_no_leakage.py`, `tests/test_baseline.py` and `tests/test_real_data.py` replace all
  data after T, for both units, and require features, training sets (features and targets) and
  model predictions to be unchanged.

The audit against these rules found these gaps in the first Step 3 draft, all now fixed:

1. Only daily origins existed.
2. Training included partial target windows (t ≤ T − L − 1 with per-row label filtering).
3. There were no per-fold metrics.
4. Target-day promotions were an optional feature.
5. The hourly in-stock class would read hour T + 1 at an hourly origin.
6. A low-volume grouping used data after the first origin.

## Censoring benchmark protocol (Step 4)

- Truth is a **proxy**: real sales on fully in-stock days (hours 6-21). Stockouts are injected
  with a demand-driven mechanism (opening stock K; depletion when cumulative sales reach K;
  lower-bound sales in the depletion hour; calibrated residual sales in flagged hours; restocks).
- Injection parameters are calibrated on the fit window only; the benchmark uses validation days.
- Reconstruction is defined at the origin T = end of the censored day: methods see that day's
  censored sales and flags plus summaries of days before it, never the truth, K or later days
  (Rules 1-2; `tests/test_censoring.py`).
- Primary metrics: daily-total MAE and bias on stockout days; hourly errors by availability
  class; paired cluster bootstrap with Holm; pre-declared sensitivity scenarios. A fully
  synthetic generator with known parameters (`simulate_latent`) is used for parameter recovery
  in Step 5. Results: `docs/results_step4.md`.

## Milestones and steps

| Milestone | Steps | Acceptance |
|---|---|---|
| M1 Reproducible data + baselines | 0 Scaffold, 1 Ingest/audit, 2 As-of features + metrics, 3 Baselines | Audit + leakage tests pass |
| M2 Bayesian demand estimator | 4 Censoring benchmark, 5 Bayesian ladder, 6 Particle filter/replay | Diagnostics + calibration evaluated |
| M3 Stateful simulator | 7 Simulator + **TS vertical slice** (1 scenario, 1 policy, Python = TS bit-for-bit within tolerance) | Conservation + cross-language parity tests pass |
| M4 Classical decision engine | 8 Order-up-to, newsvendor, SAA-MPC, CVaR | Analytic + MPC benchmarks pass |
| M5 Adaptive / meta-learning | 9 Contextual selection, contextual optimization, Reptile (time-boxed) | Improvement with CIs, or documented negative result |
| M6 Public app | 10 Final eval/report, 11 Export + full app + deploy | Both pipelines interactive on Pages; reproducible artifacts |

## Corrections adopted (from review)

1. **Censoring.** Distinguish full availability / partial-hour availability / full unavailability
   where the data permit. A partially available hour gives a lower bound on demand; a fully
   unavailable hour with zero sales carries almost no direct demand information (likelihood
   contribution ~ uninformative, not P(D >= 0) misread as evidence). If only binary hourly flags
   exist, the censoring mechanism is treated as uncertain and reported via sensitivity analyses
   (e.g. availability fraction in a stockout hour in {0, 0.25, 0.5, 1}, flag-noise rates).
2. **Units.** An integer-scale test is necessary but not sufficient to call values counts.
   Without documentation of the normalization, use a continuous likelihood (hurdle-Gamma /
   Tweedie / hurdle-lognormal) or an explicitly qualified count approximation.
3. **Computation.** Start with daily aggregation and a small hourly subset; benchmark memory and
   runtime before scaling. Use marginalization (e.g. Kalman / forward algorithm where the model
   permits), structured VI, or particle methods when full NUTS on hourly latent states is impractical.
4. **Filter validation.** Particle filter is compared to a **batch filtering reference at the same
   information cutoff**, not to the smoothed batch posterior. Kalman equivalence test retained.
5. **Meta-learning.** Baselines: MPC, contextual selection, **contextual optimization**. Test
   adaptation to unseen lead-time distributions, demand shifts and cost structures, measuring
   decision quality as a function of the number of adaptation observations (k = 0, 1, 3, 10 ...).
   Guard against learning the simulator: evaluate on simulator variants not used in meta-training.
6. **Scoring.** WIS/CRPS on observed sales for the official test. Latent-demand forecasts are not
   scored against censored sales as truth: report observation likelihood / censoring-aware scores
   separately, and use synthetic ground truth for latent-demand recovery.
7. **Statistics.** Paired bootstrap resampling at product-store or store-cluster level (block
   bootstrap in time where needed) instead of naive Diebold-Mariano. Holm correction across
   pre-declared comparisons.

## Evaluation tracks

1. Observed-sales forecasting on the locked official test (WIS, CRPS, WAPE, coverage, sharpness,
   subgroups: low-volume, promotion, stockout periods).
2. Latent-demand recovery on controlled synthetic censoring with known truth (labelled proxy).
3. Decision quality on the synthetic supply-chain environment with common random numbers.

Pre-declared hypotheses are recorded in `configs/hypotheses.yaml` before any test-set run; each
final configuration is run against the test split once, logged by config hash.
