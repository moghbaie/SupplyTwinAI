# Bayesian Demand Intelligence and Adaptive Supply Chain Decision Twin

Research-grade implementation and deployment plan

Target: https://moghbaie.github.io/supply-chain-twin/
Training: Google Colab Pro. Frontend: React + TypeScript + Vite, hosted on GitHub Pages.

## 0. Goals and boundaries

Build two independently evaluated, connected pipelines:

A. Demand Intelligence: estimate latent demand from censored sales, produce calibrated probabilistic forecasts, and update a latent state sequentially as historical observations are replayed.
B. Adaptive Supply Chain Decision Twin: use predictive demand distributions, simulated inventory and lead times to select replenishment actions, compare conventional optimization with contextual policy selection and genuine meta-learning, and evaluate on held-out scenarios.

GitHub Pages hosts static HTML/CSS/JavaScript and precomputed artifacts, NOT Colab notebooks, Python inference services, live training, or a persistent database. Deliver a functioning interactive static demo first. A future live API must be deployed separately.

Do not describe reconstructed demand as observed ground truth. Do not describe replay as live retail telemetry. Do not present synthetic inventory costs as observed retailer savings.

## 1. Sources and licensing

Primary: FreshRetailNet-50K, https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K (CC BY 4.0; attribute the original authors). About 50,000 product-store 90-day series, 4.5M train rows, hourly sales arrays, hourly stockout flags, discount, holiday, activity, weather, product/store/category identifiers. Inspect the dataset card, original technical report, official repository and split definitions before using it. Sales are normalized, not literal physical unit counts.

External robustness source (optional): FreshRetailNet-LT, https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-LT ; verify schema, license and whether it overlaps primary data before treating it as independent external validation.

Supplementary benchmark (optional): M5 uncertainty competition, https://www.kaggle.com/c/m5-forecasting-uncertainty/data ; this is observed-sales forecasting, not latent-demand ground truth.

Supply-chain simulator: openly documented, synthetic supplier lead times, initial inventory, arrivals, purchasing/holding/shortage/spoilage costs, storage limits, and replenishment policies. Version and disclose every assumption.

## 2. Repository layout

```
supply-chain-twin/
  README.md
  plan.md
  LICENSE
  CITATIONS.md
  requirements-colab.txt
  pyproject.toml
  notebooks/
    01_data_audit.ipynb
    02_baselines.ipynb
    03_bayesian_demand.ipynb
    04_online_filter.ipynb
    05_inventory_simulator.ipynb
    06_decision_policies.ipynb
    07_meta_learning.ipynb
    08_evaluation_export.ipynb
  src/sctwin/
    data.py
    features.py
    baseline.py
    bayes.py
    filter.py
    simulation.py
    optimization.py
    contextual.py
    meta.py
    evaluation.py
    export.py
  tests/
    test_data.py
    test_temporal_splits.py
    test_censoring.py
    test_filter.py
    test_simulator.py
    test_optimization.py
    test_no_leakage.py
    test_export.py
  configs/
    small.yaml
    medium.yaml
    full.yaml
  artifacts/demo/              # small, non-sensitive, versioned JSON only
  web/
    package.json
    vite.config.ts
    src/
      App.tsx
      pages/Demand.tsx
      pages/Decision.tsx
      pages/Evaluation.tsx
      components/
      data/
  .github/workflows/deploy.yml
```

Store raw datasets, model weights, posterior samples, and large Parquet files in Google Drive, not the Git repository. Pin package versions and seeds; record source dataset revision hashes.

## 3. Stage 1: Data acquisition and audit (days 1-3)

In Colab Pro:

```python
!pip -q install datasets polars pyarrow pandas numpy pytest
from datasets import load_dataset
raw = load_dataset('Dingdong-Inc/FreshRetailNet-50K')
print(raw)
print(raw['train'].features)
print(raw['train'][0])
```

If the remote dataset loader or network fails, use the dataset's Files tab and download Parquet files explicitly. Mount Google Drive and save an immutable local copy plus checksum.

Audit: schema, unique keys (store_id, product_id, dt), date coverage, hourly array length, missing/negative values, stockout status coding, sales normalization, daily-vs-hourly consistency, duplicate dates, and covariate availability at prediction time. Do not assume hourly stockout flags are perfect; compare count summaries and flag patterns. Note that hourly availability may be correlated with demand and replenishment actions.

Split within each 90-day series, e.g. first 56 days train, next 14 validation, last 20 test, with sufficient history for lags. ALSO reserve disjoint product-store groups (and ideally stores) for generalization tests. The official train/test split may have different intended use; document any conflict. Keep all tuning off final test. Avoid future weather leakage: use observed weather only for retrospective analysis or provide a forecast/proxy available at decision time.

Deliverables: data dictionary, audit_report.json, immutable split manifests, 20-series Colab subset, data card. Gate: all required fields validated; no overlapping train/test timestamps for a series; deterministic rerun; clear normalized-units labeling.

## 4. Stage 2: Baseline demand forecasting (days 4-6)

Targets: next-hour observed sales and next-day observed sales, plus a full-availability proxy on reliably in-stock periods. Baselines: seasonal naive (hour-of-day/day-of-week), pooled negative-binomial GLM where applicable, LightGBM point forecast, LightGBM quantile forecast. Features: lagged sales, trailing means, historical availability, discount, activity/holiday, time-of-day, product/store/category, weather available at forecast time. No current-hour sales or stockout flags as predictive features before the hour happens.

Evaluate rolling-origin forecasts with MAE, WAPE (aggregated and with zero-denominator safeguards), WIS/CRPS or quantile loss, 50/80/90% predictive interval coverage, sharpness, and subgroup results for low-volume, promotion, and stockout periods. Report performance on available periods separately from censored periods.

Gate: all baselines train and evaluate without temporal leakage; quantiles ordered or corrected; reproducible saved predictions; simple seasonal naive is not silently omitted.

## 5. Stage 3: Bayesian latent-demand model (days 7-13)

Start with 20 product-store series, then 100 if diagnostics and runtime permit. Do not fit a 50,000-series NUTS model in Colab as the first experiment.

Let latent demand D_{i,s,t} have a Negative Binomial distribution:

D_{ist} ~ NB(mu_{ist}, phi_i)
log(mu_{ist}) = alpha_category(i) + u_i + v_s + z_{ist} + beta^T X_{ist} + seasonal(hour,weekday)
z_{ist} = rho_i z_{is,t-1} + epsilon_{ist}, epsilon ~ Normal(0, sigma_z)

Hierarchical priors: u_i ~ Normal(0,sigma_product), v_s ~ Normal(0,sigma_store), shrinkage on promotion/weather coefficients, regularizing priors on dispersion and state noise. Check identifiability of global intercept versus random effects.

Observation model: with full availability and no measurement noise, observed sales Y = D. When known inventory capacity K is exhausted, Y = min(D,K) and likelihood is P(D>=K) for a censored record. FreshRetailNet contains hourly stockout indicators but not exact inventory capacity K or true lost demand, so this exact censoring likelihood is NOT automatically justified. Design an availability-aware observation model conditioned on stock status and uncertain availability duration, with explicit assumptions; validate it using controlled synthetic censoring of known fully-available hours. Do not treat the stockout indicator as an exact measurement of unmet demand. Continuous normalized sales may require an appropriate continuous likelihood or carefully justified transformation; do not directly fit integer count likelihood to arbitrary scaled floats.

Inference: PyMC NUTS on a small model for a reference posterior; NumPyro/JAX or ADVI for scalability. Record 4 chains where feasible, warmup/draw counts, R-hat (<1.01 target), effective sample sizes, divergences (0 target), prior predictive checks, posterior predictive checks, and model runtime. Adjust model complexity if diagnostics fail rather than hiding failures.

Compare to uncensored Bayesian model, hierarchical versus non-hierarchical, state-space versus static model, and quantile LightGBM. Primary latent-demand recovery evaluation: synthetic stockout injection on fully available segments with known pre-censoring sales, with caveat that full-availability sales are only a proxy for realized demand. Use observed sales likelihood on held-out data as a secondary metric.

Gate: reproducible posterior diagnostics; calibrated forecasts; explicit model assumptions and identifiable limitations; saved posterior and compact forecast artifacts.

## 6. Stage 4: Sequential Bayesian updates and replay (days 14-16)

Offline fit global parameters. During replay, use a particle filter for latent demand state, with transition, observation likelihood, log-weight normalization, ESS threshold, systematic resampling and rejuvenation if needed. For the first version, freeze global parameters between periodic offline refits. Particle filtering does NOT automatically update every global model parameter.

Replay ordered historical observations with simulated clock. Never read future data into current prediction. On each event, update the state, predict next 1/6/24 hours, emit posterior quantiles and data-quality warnings. Persist checkpoint snapshots for deterministic restart.

Tests: synthetic linear-Gaussian case versus Kalman filter; particle weight sums to one; no NaNs; deterministic fixed seed; no lookahead; late/out-of-order events handled explicitly; drift alarms. Measure update p50/p95 latency and posterior coverage.

Gate: online replay produces valid predictions and meets a declared target (initial research target: p95 under 2 seconds per selected series on allocated Colab runtime; revise if infeasible).

## 7. Stage 5: Inventory digital twin (days 17-20)

Simulation state: inventory on hand, pipeline orders with due dates, backlog or lost sales, product age for perishables, storage capacity, forecast distribution, supplier reliability, and remaining budget. Each time step: receive deliveries, age/spoil stock, realize exogenous demand, fulfill demand, record shortage, place orders, accrue costs. Define event order clearly.

Synthetic supply-chain parameters: lead times (e.g. 1-7 days), lead-time variance, fixed/variable ordering cost, holding cost, shortage penalty, spoilage cost, storage constraints, initial stock, order minimums. These are research scenarios, not retailer measurements. Demand trajectories from held-out full-availability periods can be used as a proxy; generate additional latent-demand trajectories from an independent data-generating process for controlled experiments. Preserve normalized-unit semantics or explicitly rescale in a synthetic environment.

Tests: inventory conservation, nonnegative on-hand inventory, arrival accounting, no fulfillment beyond stock, correct expiration, no negative orders, capacity and budget constraints, fixed-seed reproducibility, and zero-demand/zero-lead-time edge cases.

Gate: simulation passes property-based tests and hand-calculated examples.

## 8. Stage 6: Decision policy baselines (days 21-24)

Policy A: reorder point/order-up-to with tuned safety stock. Policy B: single-period newsvendor under appropriate assumptions. Policy C: scenario-based stochastic optimization / rolling-horizon model predictive control (MPC), using forecast samples and simulated lead times. All policies receive identical information available at decision time.

Objective: minimize expected total ordering + holding + shortage + spoilage costs subject to service-level, capacity, budget and nonnegative inventory constraints. Risk-aware variant: CVaR penalty or explicit chance constraints. Do not optimize on test scenario realizations.

Gate: optimize simple analytical newsvendor cases correctly; compare every policy on paired common random-number simulation trajectories.

## 9. Stage 7: Contextual selection and genuine meta-learning (days 25-27)

Start with contextual bandit or supervised policy selector that chooses among baseline policies based on volatility, stockout rate, lead-time variability, product shelf life and costs. This is an adaptive baseline, not inherently meta-learning.

Then train a meta-learner across tasks T=(product group, store group, supplier regime, demand regime, cost structure). Options: Reptile/MAML for fast adaptation of a policy network, or a recurrent meta-RL policy for sequential decisions. First experiment: Reptile on a compact policy network with task-level support/query splits. Inner loop adapts on past task episodes; outer loop evaluates on distinct future episodes. Use constrained action projection or a fallback MPC policy for invalid/unsafe actions.

Avoid train/test task leakage: hold out product-store groups and unseen supplier regimes. Do not select the best model based on final test episodes. Baselines: fixed policy, stochastic MPC, contextual selector, meta-learned policy, and ablations without posterior uncertainty or task adaptation.

Metrics: total cost, fill rate, stockout frequency, spoilage, service-level violations, tail cost (CVaR), adaptation sample efficiency, compute time. Report paired bootstrap confidence intervals over independent test episodes. Gate: meta-learning is only called an improvement if it beats strong MPC and contextual baselines with uncertainty intervals and acceptable constraint violations; otherwise report a negative result.

## 10. Stage 8: Evaluation and scientific reporting (days 28-31)

Three independent evaluation tracks:
1. Observed sales forecasting on real held-out records.
2. Latent-demand reconstruction on controlled censoring experiments, with explicit synthetic ground truth and caveats.
3. Inventory decision performance on a synthetic supply-chain environment using common demand realizations.

Ablations: no hierarchy, no dynamic state, ignore stockouts, no promotions, no uncertainty, no contextual selection, no meta-adaptation. Robustness: new stores/products, rare promotions, regime shifts, supplier delays, higher stockout fractions, noisy stockout labels. Report seed sensitivity and compute budgets.

Produce: model cards, data provenance, split manifests, metric tables, failure cases, reproducibility command, research report and honest limitations. Do not claim causal effects from observational promotions without identification assumptions.

## 11. Stage 9: Export a browser-compatible research demo (days 32-33)

Export only compact artifacts from Colab:
- `products.json`: anonymized identifiers and categories.
- `forecasts.json`: timestamp, observed sales, estimated demand, q05/q10/q50/q90/q95, stockout status, model/version.
- `policies.json`: precomputed policy comparison trajectories, cost breakdown, constraints, seeds.
- `metrics.json`: all test metrics, sample sizes, uncertainty intervals.
- `manifest.json`: dataset attribution, experiment version, generation date, unit semantics, caveats.

Limit exported records (e.g. 10-30 curated product-store series). No full dataset, large model checkpoints, secret credentials, or personal data in public repo. Validate JSON schema and compressed asset sizes. The demo's slider-driven simple newsvendor/order-up-to calculations can run in TypeScript. Advanced meta-policy outcomes must be precomputed unless a separate API is deployed. Do not claim the UI retrains a model in browser.

## 12. Stage 10: GitHub Pages app and deployment (days 34-36)

Create a project repository named `supply-chain-twin` under GitHub account `moghbaie`. Expected URL: `https://moghbaie.github.io/supply-chain-twin/`. Do not modify the existing `moghbaie.github.io` repository until you have reviewed its structure and chosen how to add a portfolio link.

Frontend: Vite + React + TypeScript, Plotly or Recharts, accessible responsive layout. Pages:
- `/demand`: choose product/store, step historical replay, show observed sales versus inferred latent demand, posterior intervals, stockout overlays, and prediction horizon.
- `/decision`: choose scenario, edit holding/shortage costs, lead times and service targets, inspect optimized orders and policy comparisons; distinguish browser-computed simple policies from precomputed advanced policy outputs.
- `/evaluation`: metrics, calibration, model diagnostics, ablations, dataset/model citations, compute environment, limitations.

Configure `base: '/supply-chain-twin/'` in `web/vite.config.ts`. Prefer hash-based routing or configure SPA fallback for GitHub Pages. Build and test `npm run build` and `npm run preview` locally. GitHub Actions workflow checks out code, sets up Node, installs dependencies with `npm ci`, runs TypeScript/lint/unit tests, builds `web/dist`, uploads Pages artifact, deploys with `actions/deploy-pages`; grant `pages: write` and `id-token: write` and configure Pages source to GitHub Actions. Publish on pushes to main only after tests pass.

Deployment tests: homepage HTTP 200, asset paths correct under project base URL, no 404 on navigation/reload, JSON loads, charts render, keyboard accessible controls, mobile layout, clear replay/synthetic labels, attribution visible, no private artifacts or API tokens. Add link to existing portfolio work section after reviewing its source. GitHub Pages supports static content, not persistent Python inference.

## 13. Optional Phase 2: real backend

Deploy FastAPI on a separate supported Python hosting platform, store per-user inference state in a database, protect write endpoints, use HTTPS and CORS allowlist, rate-limit requests, schedule offline refits, and add monitoring. Keep GitHub Pages as frontend. Do not expose Colab notebooks as production API servers. The same frontend can switch from static JSON to API responses via configuration.

## 14. Colab Pro runtime policy

At notebook start log CPU count, RAM, accelerator type, package versions, disk, seed and runtime ID. Colab Pro does not guarantee a particular GPU, RAM or session length. Begin CPU-based NUTS on 20 series, checkpoint posterior to Drive, and benchmark JAX/NumPyro on available accelerator before scaling. Avoid full MCMC on 50k series. Use Polars/Parquet, mini-batches/VI where appropriate, and small deterministic CI tests. Save intermediate results to Drive after every validated stage.

## 15. Final acceptance checklist

- [ ] Dataset source, version, license and normalization documented
- [ ] Temporal and group-held-out splits verified leakage-free
- [ ] Classical, LightGBM and probabilistic baselines reproduced
- [ ] Bayesian posterior diagnostics and predictive calibration reported
- [ ] Stockout-aware model evaluated without claiming unobserved demand ground truth
- [ ] Sequential filter tested on controlled data and replay
- [ ] Inventory simulator conservation and constraints tested
- [ ] Stochastic optimizer validated against simple exact cases
- [ ] Contextual and meta-learned policies compared to strong baselines
- [ ] Independent final evaluation with confidence intervals and ablations
- [ ] Both dashboards functional as static GitHub Pages demo
- [ ] Existing portfolio links to project, with no destructive edits
- [ ] Public site labels replay, normalized units and synthetic costs accurately

## 16. Sources verified 2026-11-12

- Existing portfolio: https://moghbaie.github.io/
- FreshRetailNet-50K: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- GitHub Pages documentation: https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages
