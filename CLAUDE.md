# CLAUDE.md — Supply Chain Twin

Research code: Bayesian latent-demand estimation on FreshRetailNet-50K and an adaptive
supply-chain decision twin. Plan: `plan.md`. Data facts: `docs/data_card.md`, `docs/data_model.md`.

## Non-negotiable rules

### Rule 1: Evaluate with rolling-origin backtesting, without leakage

- The time unit is configurable: "day" or "hour". Lead time L, horizon H, origin step S and window
  length W are all measured in that unit. Never hardcode a unit.
- Data must be resampled to the chosen unit before backtesting. For "day", define day boundaries in
  a fixed timezone (state which one). Treat a day as complete only once its last hour has passed.
- At each origin T:
  - Training: use only examples whose full target window ends at or before T. An example with
    feature time t has target window (t + L, t + L + H], so include it only if t ≤ T − L − H.
  - Prediction: compute features as of T and predict the window (T + L, T + L + H].
- Data after T must never be used in training, feature fitting, scaling, imputation, target
  encoding, or hyperparameter tuning for that fold. If tuning, use a nested time split inside the
  training window.
- Use an [expanding / sliding (length W)] training window. Advance the origin by S units for N
  folds. Report metrics per fold and aggregated across folds.

### Rule 2: Build features using only past data (point-in-time correctness)

- Every feature is computed as of the origin T, never as of the target time. "Lag k" means k units
  before T, in the configured unit.
- A feature may only use data available at or before T. Rolling and lag features must look backward
  only (shift before rolling; no centered windows, no backfill).
- When the unit is "day" but raw data is hourly, aggregate only complete days at or before T. Never
  include a partial day after T.
- Statistics fitted on data (means, scalers, encoders, normalizers) are fitted only on each fold's
  training window, then applied to its test window.
- Join external data on its availability timestamp, not its event timestamp.

Before changing the pipeline: check it against both rules. If it breaks them, explain where,
update the plan, rewrite only the affected code, and keep the leakage tests (run for both "day" and
"hour") passing.

## How the rules are applied in this project

- **Timezone / day boundaries.** A day is the dataset's `dt` with hour index 0–23 in store-local
  time. The provider does not state the timezone; all stores are Dingdong (China) stores, so we
  assume **Asia/Shanghai (UTC+8, no DST)**. A day is complete after hour 23.
- **Unit indexing.** Unit index `u` = day index (day) or `day*24 + hour` (hour). Origin `T` = the
  end of unit `T`; everything in units ≤ T is observed. Lag k = value of unit `T − k`.
- **Availability timestamps of fields** (when a value may first be used):

  | Field | Event time | Available at |
  |---|---|---|
  | `hours_sale[h]`, `hours_stock_status[h]` | hour h of day d | end of that hour |
  | `sale_amount`, `stock_hour6_22_cnt` | day d | end of day d (complete day) |
  | weather, `discount`, `activity_flag` | day d | **end of day d** (daily record; not known in advance) |
  | weekday, hour of day | target time | deterministic calendar: always available |
  | `holiday_flag` | day d | treated as a published calendar: available in advance (decided 2026-11-12) |

  Target-time promotions are NOT features: no availability timestamp proves they are known ahead.
- **Hourly availability class** of hour u needs the flag of hour u+1, so it is only usable when
  u + 1 ≤ T.
- **Code.** `sctwin.timeunit` resamples to the unit; `sctwin.features.FeatureBuilder` builds
  as-of-T features; `sctwin.backtest` defines folds and `check_split` asserts the rules on every
  fold at run time; `tests/test_no_leakage.py` perturbs all data after T (both units) and requires
  features, training sets and predictions to be unchanged.

## Other conventions

- The official `eval` split is the locked final test; development code never loads it.
- Raw data, runs and model outputs stay out of git (`data/`, `runs/`).
- Before finishing: `ruff format`, `ruff check`, `mypy src`, `pytest -W error` (plus `-m data`).
- Do not commit or push unless the user asks.
- Never describe reconstructed demand as observed, replay as live telemetry, or synthetic costs
  as retailer savings. Sales are normalized units.

## Commands

```bash
uv pip install -e ".[dev,models]"
.venv/Scripts/python -m pytest -W error -m "not data"   # synthetic tests
.venv/Scripts/python -m pytest -m data                  # real-data tests
.venv/Scripts/python scripts/audit_data.py              # Step 1 audit
.venv/Scripts/python scripts/run_baselines.py --unit day|hour [--smoke]
```
