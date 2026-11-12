# Supply Chain Twin

**Bayesian demand intelligence and an adaptive supply-chain decision twin, built on censored
fresh-retail sales.**

This is research code with two connected pipelines, each evaluated on its own:

- **A. Demand intelligence:** estimates latent demand from sales that stockouts have censored,
  produces calibrated probabilistic forecasts, and updates a latent demand state as historical
  data is replayed in order.
- **B. Decision twin:** feeds those predictive demand distributions into a simulated perishable
  inventory system. It then compares classical replenishment policies, stochastic MPC,
  contextual policy selection and meta-learning on held-out scenarios.

The public demo will be a static site at <https://moghbaie.github.io/supply-chain-twin/>. It
shows precomputed research artifacts. No live model training or inference runs in the browser.

> **What this project does not claim**
> - Reconstructed demand is an *estimate*, never observed ground truth.
> - Replay of historical data is not live retail telemetry.
> - Inventory costs and lead times are **synthetic research scenarios**, not retailer savings.
> - Sales are in the provider's **normalized units**, not physical unit counts.

## Status

| Milestone | Steps | Status |
|---|---|---|
| M1 Reproducible data and baselines | 0 Scaffold · 1 Ingest and audit · 2 Point-in-time features and metrics · 3 Baselines | Steps 0–1 done |
| M2 Bayesian demand estimator | 4 Censoring benchmark · 5 Bayesian model ladder · 6 Particle filter and replay | Not started |
| M3 Stateful simulator | 7 Simulator plus a Python ↔ TypeScript parity slice | Not started |
| M4 Classical decision engine | 8 Order-up-to, newsvendor, SAA-MPC, CVaR | Not started |
| M5 Adaptive and meta-learning | 9 Contextual selection and optimization, Reptile (time-boxed) | Not started |
| M6 Public app | 10 Final evaluation and report · 11 Export, web app, deploy | Not started |

The full plan, with its tests, validation gates and evaluation design, is in [`plan.md`](plan.md).
The original brief is [`plan_Supply_Twin_AI.md`](plan_Supply_Twin_AI.md).

## Key data findings so far

These come from the Step 1 audit. Details and numbers are in [`docs/data_card.md`](docs/data_card.md).

- **The data is clean.** It has 50,000 store-product series × 90 training days, plus the 7-day
  official eval week, which is kept as a locked test set. There are no gaps, duplicates or nulls,
  and daily sales equal the sum of hourly sales exactly.
- **Units:** 99.56% of hourly sales fall on a 0.1 grid, but the normalization is undocumented.
  Models therefore use a continuous likelihood. Treating sales as counts appears only as a
  labelled sensitivity analysis.
- **Stockouts are driven by demand.** Within each series, the in-stock hour just before a
  stockout flag sells **2.2×** that series' normal amount for the hour, while the first flagged
  hour sells **0.18×**. Availability is endogenous, and the pre-flag hour is treated as a
  censored depletion hour.
- **The stockout count field** `stock_hour6_22_cnt` covers hours 6–21. Weather is daily,
  per store, and observed after the fact.

## Repository layout

```
src/sctwin/          library code (the notebooks and scripts only call into it)
  data.py            pinned download, checksums, schema validation, hourly expansion
  audit.py           data audit: integrity gates plus normalization and stockout evidence
  splits.py          temporal and group hold-out splits, tamper-evident manifests
  repro.py           seeding, config hashing, run manifests (environment and package versions)
scripts/             command-line entry points (fetch_data.py, audit_data.py)
tests/               unit tests on synthetic fixtures; tests marked `data` use the real dataset
configs/             experiment configs; manifests/ holds the frozen split definitions
docs/data_card.md    dataset provenance, audit findings, modelling implications
notebooks/           Colab drivers for heavy runs (MCMC, meta-learning), from Step 5 on
artifacts/demo/      small, versioned JSON for the web demo (the only data in git)
web/                 Vite + React + TypeScript app (Step 11)
```

Raw data, posterior samples, model weights and large Parquet files never go in git. They live
in `data/` locally (git-ignored), or on Google Drive when running in Colab.

## Quickstart

Requirements: Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). Node 20+ is needed only
for the web app.

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"

# Unit tests run on synthetic data, with no download needed
.venv/Scripts/python -m pytest -m "not data"

# Download the dataset at the pinned revision (~115 MB) and record checksums
.venv/Scripts/python scripts/fetch_data.py

# Run the audit; writes data/audit/audit_report.json, the split manifest and a 20-series dev subset
.venv/Scripts/python scripts/audit_data.py

# Tests against the real data (checksums, gates, manifest provenance)
.venv/Scripts/python -m pytest -m data
```

On Linux, macOS or Colab, use `.venv/bin/python`. Optional extras: `.[models]` for LightGBM and
SciPy, `.[bayes]` for JAX, NumPyro and ArviZ.

## Reproducibility

- **Data:** pinned to Hub revision `c374e85975c82816c495acfa904cc55ab14a47d3`. SHA256 checksums
  are verified before every audit run.
- **Splits:** `configs/manifests/splits_v1.json` is written once and protected by a content
  digest. It refuses to be silently overwritten, and edits made after writing are detected.
- **Runs:** every stage writes a `run_manifest.json` recording the config hash, seed, Python and
  package versions, CPU and accelerator.
- **Locked test:** the official eval split is only integrity-checked during development. It will
  be read through a test ledger that allows one logged run per final configuration.
- **CI:** GitHub Actions runs ruff, mypy and the synthetic-data test suite on every push and
  pull request.

## Evaluation tracks

1. **Observed-sales forecasting** on the locked official test week: WIS/CRPS, WAPE, interval
   coverage and sharpness, plus subgroups (low volume, promotions, stockout periods).
2. **Latent-demand recovery** on controlled synthetic censoring, where the truth is known. This
   is reported as a proxy.
3. **Decision quality** in the synthetic inventory environment, using common random numbers and
   a cluster-level paired bootstrap with Holm correction.

## Data and license

- **Code:** MIT, see [`LICENSE`](LICENSE).
- **Data:** FreshRetailNet-50K by Dingdong-Inc, licensed CC BY 4.0. It is not redistributed
  here. Citation and provenance are in [`CITATIONS.md`](CITATIONS.md).

If you use this work, please cite the dataset paper:
Wang et al., *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand
Recovery and Forecasting in Fresh Retail*, arXiv:2505.16319 (2025).
