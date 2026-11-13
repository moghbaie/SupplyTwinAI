# Step 4 results: censoring benchmark (latent-demand recovery, proxy truth)

Run on 2026-11-12, `runs/step4/8ba303c79bd4` (`scripts/run_censoring_benchmark.py`,
`configs/censoring.yaml`).

**The "truth" here is a proxy:** real hourly sales on days that were fully in stock in hours 6–21.
In-stock sales are the closest observable stand-in for demand, not demand itself. Every number
below is about recovering that proxy after an artificial stockout.

## Design

- **Data.** 5,000 seeded in-sample series. The official test split is not loaded.
- **Real stockout statistics** come from the fit window (days 28–75).
- **Injection settings are calibrated on fit-window days only:**
  - opening stock, set as a multiple of expected demand and its spread;
  - sales still seen in flagged hours;
  - restock probability and duration.
- **The benchmark** uses the fully in-stock validation days (76–89): 39,409 days, of which 16,628
  received an injected stockout (119,351 censored hours).
- **Mechanism.** Each day gets an opening stock K, set from the series' in-stock sales over the
  previous 28 days. The stockout starts in the hour where cumulative sales reach K. That hour
  keeps its flag at 0 but sells only the remaining stock, so its sales are a lower bound. Later
  hours are flagged, keep a small calibrated share of sales, and may restock.
- **No leakage.** A reconstruction is made at the end of the censored day. It uses that day's
  censored sales and flags plus summaries of the previous 28 days only. It never sees the truth,
  K, the depletion hour or any later day. Tests replace the day itself and all later days and
  require identical inputs and outputs (`tests/test_censoring.py`).

Calibrated configuration: coverage 1.15, spread 0.6, leak 0.167 (first flagged hour) / 0.074
(later hours), restock probability 0.187, mean restock duration 5.75 h. The raw restock values are
smaller because a restock after hour 21 is never seen.

## Realism check (benchmark days vs real fit-window days)

| Statistic | Real | Injected |
|---|---|---|
| Stockout rate (days in stock at 06:00) | 0.381 | 0.409 |
| First stockout hour, quartiles | 12 / 16 / 18 | 12 / 16 / 18 |
| Same-day restock rate / mean hours | 0.110 / 3.1 | 0.113 / 3.3 |
| Sales vs usual, 3 h before the first flag | 1.56 | 1.37 |
| … 1 h before (depletion hour) | **2.39** | **1.78** |
| … first flagged hour | 0.20 | 0.23 |
| … second flagged hour | 0.09 | 0.10 |

Timing, prevalence, restocks and flagged-hour sales match. **The gap:** in the hours before a
stockout, real days sell more above their usual level than injected days. Only fully in-stock
days can serve as truth, and those are quieter days. Real stockout days are therefore more
demand-driven than the benchmark. The benchmark probably **understates** how much a method that
ignores this selection under-estimates demand on real stockout days.

## Reconstruction baselines

- `observed`: ignores censoring.
- `hour_mean`: fills flagged hours with the series' trailing in-stock mean for that hour.
- `profile`: scales the series' hourly profile to the day's in-stock hours and fills the flagged
  hours.
- `profile_pre`: as `profile`, but the hour before each stockout is also treated as censored.

**Daily totals, hours 6–21, stockout days (mean true total 1.356):**

| Method | MAE | Bias | Bias % |
|---|---|---|---|
| observed | 0.485 | −0.485 | −35.8% |
| hour_mean | **0.226** | −0.143 | −10.5% |
| profile | 0.330 | +0.037 | +2.8% |
| profile_pre | 0.345 | +0.039 | +2.9% |

Paired cluster bootstrap against `profile` (5,000 series clusters, Holm-corrected):
- `hour_mean` has 31.5% lower absolute error, p = 0.0015;
- `observed` has 46.9% higher error, p = 0.0015;
- `profile_pre` vs `profile` is not significant (p = 0.10).

**By when the stockout started (daily bias):**

| Stockout start | n | hour_mean | profile | profile_pre |
|---|---|---|---|---|
| early (before 12:00) | 4,588 | −0.205 | **+0.367** | +0.342 |
| mid (12–16) | 7,100 | −0.122 | −0.068 | −0.056 |
| late (17–21) | 4,940 | −0.116 | −0.118 | −0.105 |

**Censored hours (mean true sales 0.091):**
- The depletion hour (P_pre, n = 16,115, mean 0.242) is the largest single error.
  - Every method that trusts its flag of 0 under-states it by 0.114.
  - `profile_pre` reduces this to 0.075.
- Per-hour MAE favours `observed` (0.068). Hourly demand is sparse and MAE rewards forecasts
  close to zero, so daily totals are the primary metric.

**Sensitivity (pre-declared):**
- Scenarios: no leak, high leak, no restock, 0.8× and 1.25× stock. The stockout rate ranges from
  0.31 to 0.52 across them.
- The ranking is unchanged in every scenario:
  - `hour_mean` has the lowest daily MAE (0.211–0.234), but its bias stays near −0.14;
  - `profile` keeps the smallest bias (+0.02 to +0.06);
  - `observed` under-states demand by 0.40–0.52.

## Findings

1. **Ignoring stockouts under-states demand by about a third on stockout days.** That is the size
   of the problem the Bayesian model has to fix.
2. **No baseline is both accurate and unbiased:**
   - `hour_mean` is accurate but 10% low, because it ignores that stockouts happen on busy days.
   - `profile` is unbiased overall but unstable when only a few hours were in stock: +24% on
     early stockouts.
3. **The depletion hour matters.** Treating the last in-stock hour as censored improves that hour
   (P_pre bias −0.114 → −0.075). It does not significantly change daily totals.

## Targets for Step 5 (declared before fitting)

The censoring-aware Bayesian model (M1+) is evaluated on this benchmark with the same days and
the same seed:
- **Daily-total MAE** must beat `hour_mean` (0.226).
- **Absolute daily bias** must be no worse than `profile` (0.037).
- **Its 50/80/90% intervals** for daily totals must cover the proxy truth at their nominal rates.

Failing any of these is reported as a negative result.
