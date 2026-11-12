# Data card and Step 1 audit findings — FreshRetailNet-50K

Source: `Dingdong-Inc/FreshRetailNet-50K`, Hub revision `c374e85975c82816c495acfa904cc55ab14a47d3`,
CC BY 4.0, card v1.0. Checksums in `data/raw/provenance.json`. Full numbers:
`data/audit/audit_report.json` (regenerate with `python scripts/audit_data.py`).
All findings below are computed on the official **train** split only; the locked test split
received integrity checks only.

## Structure (all gates pass)

| Item | Value |
|---|---|
| Train | 4,500,000 daily rows = 50,000 series x 90 days, 2024-03-28 .. 2024-06-25 |
| Test (official eval, locked) | 350,000 rows = same 50,000 series x 7 days, 2024-06-26 .. 2024-07-02 |
| Entities | 898 stores, 865 products, 18 cities |
| Calendar | every series shares the same calendar; no gaps, no duplicates, no nulls |
| Arrays | every `hours_sale` / `hours_stock_status` has length 24 |
| Consistency | `sale_amount` == sum(`hours_sale`) exactly on every row |
| Train/test | no overlapping keys; test starts the day after train ends for every series |

## Finding 1 — Units: a 0.1 lattice, not established counts

- 99.56% of positive hourly values lie on a 0.1 grid (0.1 alone: 14.4M occurrences, 0.2: 5.2M ...).
- The remaining 0.44% are values such as 0.06, 0.07, 0.08, 0.09, 0.119, concentrated in
  management groups 1 and 5 (on-lattice 98.5% / 98.9%) - plausibly weighed goods.
- Only 50.8% of series are entirely on the lattice; 0.8% have >5% off-lattice values.
- The card says values are multiplied by "a specific coefficient" but does not document it.

**Implication.** Consistent with "count x 0.1", but not proof (correction 2). Primary modelling
uses a **continuous / semi-continuous likelihood on normalized units** (hurdle with a continuous
positive part, or a likelihood discretized to the 0.1 grid). A count approximation `round(y/0.1)`
is allowed only as an explicitly labelled sensitivity analysis.

## Finding 2 — Stockout flag semantics

- `hours_stock_status`: 1 = out of stock, 0 = in stock (binary everywhere).
- `stock_hour6_22_cnt` == sum of flags over **hours 6-21 (16 hours)**, exactly, on 100% of rows.
- 24.9% of hours flagged; 60.0% of days have >=1 flagged hour; 3.8% of days flagged all 24 hours.
- Night hours (22-05) are flagged 33-42% of the time while sales are near zero; flags at night
  likely reflect operations (no replenishment) more than lost demand. Modelling uses hours 6-21
  for censoring inference; night hours are a separate low-rate regime.
- Runs: mean 1.41 runs per stockout day; 50.6% of runs recover the same day (intra-day restocks).
- 2.9% of flagged hours have positive sales - flags are not perfect, or availability is partial.

## Finding 3 — Stockouts are demand-driven (availability is endogenous)

Within-series sales relative to the same series' mean at the same hour on fully-in-stock days
(hours 6-21, 49,571 series with >= 10 baseline days):

| Position | Ratio |
|---|---|
| 3 h before first flagged hour | 1.42 |
| 2 h before | 1.49 |
| **1 h before (flag = 0)** | **2.21** |
| first flagged hour | 0.18 |
| second flagged hour | 0.08 |
| first in-stock hour after restock | 1.21 |
| second in-stock hour after restock | 1.19 |

**Implications for the observation model (correction 1).**

1. The hour *before* the first flag is the most likely **depletion hour**: it is flagged in stock
   but probably sold out during the hour, so its sales are a **lower bound** (right-censored), not
   an uncensored observation. A naive model that treats flag = 0 as fully observed is biased.
2. The first flagged hour retains ~18% of normal sales: partial availability at the hour's start
   and/or flag noise. Later flagged hours (~8%) are close to fully unavailable and carry little
   direct demand information.
3. Availability depends on demand, so stockout hours are **not missing at random**. Methods that
   impute flagged hours from unconditional in-stock averages will under-estimate latent demand.
4. Post-restock uplift (~1.2x) suggests pent-up or redirected demand; to be modelled or flagged.

**Planned availability classes** (hours 6-21), each with its likelihood treatment varied in a
pre-declared sensitivity analysis:

| Class | Definition | Default treatment | Sensitivity |
|---|---|---|---|
| A full | flag 0, next hour flag 0 | uncensored | - |
| P-pre partial (depletion) | flag 0, next hour flag 1 | right-censored at y | uncensored |
| P-first partial | first flagged hour of a run | right-censored at y, availability fraction a ~ Beta | a in {0, 0.25, 0.5} |
| U unavailable | later flagged hours | excluded from likelihood | right-censored at y; flag-noise rate in {0, 2, 5}% |

The synthetic censoring benchmark (Step 4) must reproduce this demand-dependent mechanism
(stockout hazard increasing with cumulative sales vs. stock) rather than random masking.

## Finding 4 — Covariates

- Weather (`precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level`) is constant per
  store-date but varies within city-date: it is **store-level daily** weather, observed after the
  fact. Use only lagged values or an explicitly declared forecast proxy for prediction.
- `holiday_flag` is constant per date (34.4% of train days flagged - likely includes weekends; check).
- `activity_flag` varies across products within a store-date (product-level promotion), rate 37.8%.
- `discount`: 51.5% of rows < 1.0; min 0.0 and a few values > 1.0 (max 1.088, 7.6e-6 of rows)
  - treat 0.0 and > 1.0 as anomalies to clip/flag in feature engineering.
- Each store belongs to one city; each product to one category path.

## Splits (`configs/manifests/splits_v1.json`, digest-protected)

| Role | Definition |
|---|---|
| fit | 2024-03-28 .. 2024-06-11 (76 days), in-sample series |
| val | 2024-06-12 .. 2024-06-25 (14 days), in-sample series |
| test (locked) | official eval, 2024-06-26 .. 2024-07-02 |
| group hold-out | 90 stores (10%) and 43 products (5%): 6,117 series never used for fitting/tuning |

In-sample series: 43,883. Development subset `data/subsets/dev20.parquet`: 20 in-sample series,
fit+val days only.

## Open questions

1. Can the provider confirm the normalization coefficient and whether weighed goods are included?
2. Is the hourly flag a snapshot (at hour start / end) or "out of stock for any part of the hour"?
   Finding 3 suggests a flag at time t marks an hour mostly unavailable, with depletion in t-1.
3. What does `holiday_flag` include (weekends, regional holidays)?
