"""Step 4: censoring benchmark (evaluation track 2, latent-demand recovery).

Real FreshRetailNet sales are censored by stockouts and the lost demand is never observed. To
test methods that reconstruct demand we need cases with a known answer:

* **Semi-synthetic injection.** Take real days that were fully in stock in hours 6-21, treat
  their sales as the demand (a *proxy*: in-stock sales are the closest observable stand-in for
  demand, not demand itself) and inject a stockout whose timing is driven by that demand.
  Each day gets an opening stock K; the stockout starts in the hour where cumulative sales
  reach K, as in the real data, where stockouts are demand-driven (docs/data_card.md,
  Finding 3).
* **Fully synthetic panel.** Latent demand from a known generative model, censored with the
  same mechanism. Used for parameter recovery in Step 5.

Injection mechanism, for one day, over hours 6-21 (the window of ``DAY_HOURS``):

1. Opening stock ``K = coverage * E * exp(sd * eps - sd^2 / 2)``, where ``E`` is the expected
   window demand (for real days: the trailing in-stock mean over the previous ``window`` days,
   i.e. data available before the day; the methods being evaluated never see K).
2. The depletion hour ``h*`` is the first hour where cumulative demand reaches ``K``. It stays
   flagged in stock and its observed sales are ``K - (demand before h*)``: a lower bound.
3. Hours after ``h*`` are flagged out of stock. A fraction ``leak_first`` (first flagged hour)
   and ``leak_later`` (later hours) of demand is still observed, matching the residual sales in
   flagged hours seen in the real data (partial availability or flag noise).
4. With probability ``p_restock`` stock returns after a geometric number of hours (mean
   ``restock_mean_hours``); from then on observed sales equal demand.

Hours outside 6-21 are never censored. Observed sales never exceed demand.

The reconstruction task is defined at the origin T = end of the censored day: methods see that
day's observed sales and flags plus summaries of days *before* it (``trailing_hour_means``),
never the demand, K, the depletion hour or any later day.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from sctwin.panel import CLS_A, CLS_P_FIRST, CLS_P_PRE, CLS_U, DAY_HOURS, availability_classes

LO, HI = DAY_HOURS
NW = HI - LO  # hours in the censoring window (16)


@dataclass(frozen=True)
class CensorConfig:
    coverage: float = 1.0  # opening stock as a multiple of expected window demand
    coverage_sd: float = 0.5  # log-normal spread of the stock level
    leak_first: float = 0.18  # share of demand still sold in the first flagged hour
    leak_later: float = 0.08  # ... and in later flagged hours
    p_restock: float = 0.10  # probability that stock returns within the window
    restock_mean_hours: float = 3.0  # mean flagged hours before a restock

    def __post_init__(self) -> None:
        if not (0 <= self.leak_first <= 1 and 0 <= self.leak_later <= 1):
            raise ValueError("leak fractions must lie in [0, 1]")
        if not 0 <= self.p_restock <= 1:
            raise ValueError("p_restock must lie in [0, 1]")
        if self.coverage <= 0 or self.coverage_sd < 0 or self.restock_mean_hours < 1:
            raise ValueError("coverage > 0, coverage_sd >= 0, restock_mean_hours >= 1")


@dataclass
class Injection:
    observed: np.ndarray  # (N, 24) observed sales after censoring
    flag: np.ndarray  # (N, 24) int8, 1 = flagged out of stock
    depletion: np.ndarray  # (N,) hour of day of the depletion hour, -1 = no stockout
    restock: np.ndarray  # (N,) first in-stock hour after the stockout, -1 = none in window
    censored: np.ndarray  # (N, 24) bool: depletion hour and flagged hours (observed <= demand)

    @property
    def stocked_out(self) -> np.ndarray:
        return self.depletion >= 0


def stock_levels(expected: np.ndarray, rng: np.random.Generator, cfg: CensorConfig) -> np.ndarray:
    eps = rng.standard_normal(np.shape(expected))
    s = cfg.coverage_sd
    return cfg.coverage * np.asarray(expected, dtype=np.float64) * np.exp(s * eps - s * s / 2)


def inject_stockouts(
    demand: np.ndarray, stock: np.ndarray, rng: np.random.Generator, cfg: CensorConfig
) -> Injection:
    """Censor (N, 24) demand with opening stocks (N,). Random draws are made for every row, so
    the same seed gives common random numbers across configurations."""
    demand = np.asarray(demand, dtype=np.float64)
    stock = np.asarray(stock, dtype=np.float64)
    if demand.ndim != 2 or demand.shape[1] != 24 or stock.shape != demand.shape[:1]:
        raise ValueError("demand must be (N, 24) and stock (N,)")
    if (demand < 0).any() or not (stock > 0).all():
        raise ValueError("demand must be >= 0 and stock > 0")
    n = demand.shape[0]
    restock_draw = rng.random(n) < cfg.p_restock
    duration = rng.geometric(1.0 / cfg.restock_mean_hours, n)  # flagged hours, >= 1

    w = demand[:, LO:HI]
    cum = np.cumsum(w, axis=1)
    hit = cum >= stock[:, None]
    has = hit.any(axis=1)
    j = np.where(has, hit.argmax(axis=1), -1)  # depletion index within the window
    r = np.where(has & restock_draw, j + 1 + duration, NW)  # window index where stock returns
    r_end = np.minimum(r, NW)
    k = np.arange(NW)[None, :]
    flag_w = has[:, None] & (k > j[:, None]) & (k < r_end[:, None])
    first = flag_w & (k == j[:, None] + 1)

    obs_w = w.copy()
    rows = np.flatnonzero(has)
    jr = j[rows]
    obs_w[rows, jr] = stock[rows] - (cum[rows, jr] - w[rows, jr])
    obs_w[first] *= cfg.leak_first
    obs_w[flag_w & ~first] *= cfg.leak_later
    depl = np.zeros_like(flag_w)
    depl[rows, jr] = True

    observed = demand.copy()
    observed[:, LO:HI] = obs_w
    flag = np.zeros(demand.shape, dtype=np.int8)
    flag[:, LO:HI] = flag_w
    censored = np.zeros(demand.shape, dtype=bool)
    censored[:, LO:HI] = depl | flag_w
    return Injection(
        observed=observed,
        flag=flag,
        depletion=np.where(has, j + LO, -1),
        restock=np.where(has & (r < NW), r + LO, -1),
        censored=censored,
    )


# --------------------------------------------------------------------------- point-in-time history


def _trailing(x: np.ndarray, days: np.ndarray, window: int) -> np.ndarray:
    """Sum of x[s, d - window : d] (days strictly before d) for rows (s, d); x is (S, D, ...)."""
    c = np.concatenate([np.zeros_like(x[:, :1]), np.cumsum(x, axis=1)], axis=1)
    return c[:, days] - c[:, np.maximum(days - window, 0)]


def trailing_hour_means(
    sales: np.ndarray, stock: np.ndarray, series: np.ndarray, days: np.ndarray, window: int
) -> np.ndarray:
    """(N, 24) mean in-stock sales per hour of day over days [d - window, d - 1] for each row
    (series[i], days[i]). NaN where the hour was never in stock. Uses no data from day d or later.
    """
    if (np.asarray(days) < 1).any():
        raise ValueError("every row needs at least one day of history")
    inst = (stock == 0).astype(np.float64)
    out = np.full((len(series), 24), np.nan)
    for s in np.unique(series):  # per-series cumulative sums keep memory O(D x 24)
        i = np.flatnonzero(series == s)
        num = _trailing((sales[s] * inst[s])[None], days[i], window)[0]
        den = _trailing(inst[s][None], days[i], window)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(den > 0, num / den, np.nan)
    return out


def expected_window_demand(hour_means: np.ndarray) -> np.ndarray:
    """Expected hours 6-21 demand from trailing in-stock hour means (NaN if any hour unknown)."""
    return hour_means[:, LO:HI].sum(axis=1)


# --------------------------------------------------------------------------- benchmark


@dataclass
class Benchmark:
    series: np.ndarray  # (N,) panel series index
    day: np.ndarray  # (N,) day index
    demand: np.ndarray  # (N, 24) real in-stock sales of that day (demand proxy)
    stock: np.ndarray  # (N,) opening stock K
    hour_means: np.ndarray  # (N, 24) trailing in-stock hour means (inputs for the methods)
    injection: Injection
    cfg: CensorConfig = field(default_factory=CensorConfig)


def candidate_days(
    sales: np.ndarray,
    stock: np.ndarray,
    series: np.ndarray,
    days: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows (series, day) whose hours 6-21 were fully in stock, with a finite positive expected
    window demand from the previous ``window`` days. Returns (series, day, hour_means)."""
    ss, dd = np.meshgrid(np.asarray(series), np.asarray(days), indexing="ij")
    ss, dd = ss.ravel(), dd.ravel()
    full = (stock[ss, dd, LO:HI] == 0).all(axis=1) & (dd >= window)
    ss, dd = ss[full], dd[full]
    hm = trailing_hour_means(sales, stock, ss, dd, window)
    e = expected_window_demand(hm)
    ok = np.isfinite(e) & (e > 0)
    return ss[ok], dd[ok], hm[ok]


def build_benchmark(
    sales: np.ndarray,
    rows: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: CensorConfig,
    seed: int,
) -> Benchmark:
    ss, dd, hm = rows
    rng = np.random.default_rng(seed)
    demand = sales[ss, dd].astype(np.float64)
    k = stock_levels(expected_window_demand(hm), rng, cfg)
    return Benchmark(ss, dd, demand, k, hm, inject_stockouts(demand, k, rng, cfg), cfg)


# --------------------------------------------------------------------------- realism diagnostics

QUANTILES = (25, 50, 75)


def stockout_stats(flag: np.ndarray) -> dict[str, float]:
    """Stockout prevalence and timing on (N, 24) flags, over days in stock at hour 6."""
    f = flag[:, LO:HI].astype(bool)
    f = f[~f[:, 0]]
    so = f.any(axis=1)
    first = f[so].argmax(axis=1)
    k = np.arange(NW)[None, :]
    after = k >= first[:, None]
    back = (after & ~f[so]).any(axis=1)
    dur = np.argmax(after & ~f[so], axis=1) - first
    out = {"n_days": float(f.shape[0]), "stockout_rate": float(so.mean()) if f.size else np.nan,
           "restock_rate": float(back.mean()) if so.any() else np.nan,
           "restock_mean_hours": float(dur[back].mean()) if back.any() else np.nan}  # fmt: skip
    for q in QUANTILES:
        out[f"first_hour_q{q}"] = float(np.percentile(first + LO, q)) if so.any() else np.nan
    return out


def position_ratios(
    observed: np.ndarray, flag: np.ndarray, baseline: np.ndarray, offsets: tuple[int, ...]
) -> dict[str, float]:
    """Sales at hours relative to the first flagged hour (offset 0), as a ratio of summed
    observed sales to summed baseline (the series' usual in-stock sales at that hour)."""
    f = flag[:, LO:HI].astype(bool)
    keep = f.any(axis=1) & ~f[:, 0]
    f = f[keep]
    j = f.argmax(axis=1)  # first flagged index in the window
    k = np.arange(NW)[None, :]
    back = (k >= j[:, None]) & ~f
    run = np.where(back.any(axis=1), back.argmax(axis=1), NW) - j  # length of the first run
    first = j + LO
    obs, base = observed[keep], baseline[keep]
    out = {}
    for o in offsets:
        h = first + o
        ok = (h >= LO) & (h < HI) & (o < run)  # offsets >= 0 must still be in the first run
        r = np.flatnonzero(ok)
        num, den = obs[r, h[r]].sum(), np.nansum(base[r, h[r]])
        out[f"{o:+d}"] = float(num / den) if den > 0 else np.nan
    return out


def calibrate(
    demand: np.ndarray,
    expected: np.ndarray,
    target: dict[str, float],
    base: CensorConfig,
    coverages: np.ndarray,
    sds: np.ndarray,
    seed: int,
) -> tuple[CensorConfig, list[dict[str, float]]]:
    """Grid search (coverage, coverage_sd) so that injected stockouts on fully in-stock days
    match the real stockout rate and first-stockout-hour quartiles. Common random numbers."""
    grid = []
    for c in coverages:
        for sd in sds:
            cfg = CensorConfig(**{**base.__dict__, "coverage": float(c), "coverage_sd": float(sd)})
            rng = np.random.default_rng(seed)
            inj = inject_stockouts(demand, stock_levels(expected, rng, cfg), rng, cfg)
            st = stockout_stats(inj.flag)
            loss = ((st["stockout_rate"] - target["stockout_rate"]) / 0.02) ** 2 + sum(
                ((st[f"first_hour_q{q}"] - target[f"first_hour_q{q}"]) / 1.0) ** 2
                for q in QUANTILES
            )
            grid.append({"coverage": float(c), "coverage_sd": float(sd), "loss": float(loss), **st})
    best = min(grid, key=lambda g: g["loss"] if np.isfinite(g["loss"]) else np.inf)
    cfg = CensorConfig(**{**base.__dict__, "coverage": best["coverage"],
                          "coverage_sd": best["coverage_sd"]})  # fmt: skip
    return cfg, grid


def calibrate_restock(
    demand: np.ndarray,
    expected: np.ndarray,
    target: dict[str, float],
    cfg: CensorConfig,
    p_grid: np.ndarray,
    mean_grid: np.ndarray,
    seed: int,
) -> CensorConfig:
    """Grid search (p_restock, restock_mean_hours) so that the *measured* restock rate and mean
    restock duration match the real data. Both are truncated by the end of the window in the
    same way, which is why the raw real values cannot be plugged in directly."""
    best, best_loss = cfg, np.inf
    for p in p_grid:
        for m in mean_grid:
            c = dataclasses.replace(cfg, p_restock=float(p), restock_mean_hours=float(m))
            rng = np.random.default_rng(seed)
            st = stockout_stats(
                inject_stockouts(demand, stock_levels(expected, rng, c), rng, c).flag
            )
            loss = ((st["restock_rate"] - target["restock_rate"]) / 0.01) ** 2 + (
                (st["restock_mean_hours"] - target["restock_mean_hours"]) / 0.25
            ) ** 2
            if np.isfinite(loss) and loss < best_loss:
                best, best_loss = c, loss
    return best


def calibrate_leak(
    demand: np.ndarray,
    expected: np.ndarray,
    baseline: np.ndarray,
    target: dict[str, float],
    cfg: CensorConfig,
    seed: int,
) -> CensorConfig:
    """Set leak_first / leak_later so that sales in the first and second flagged hour, relative
    to the usual in-stock sales at that hour (``position_ratios`` offsets 0 and +1), match the
    real data. Flags do not depend on the leak, so the ratio is linear in it: one exact step."""
    rng = np.random.default_rng(seed)
    inj = inject_stockouts(demand, stock_levels(expected, rng, cfg), rng, cfg)
    got = position_ratios(inj.observed, inj.flag, baseline, (0, 1))
    new = {}
    for key, name in (("+0", "leak_first"), ("+1", "leak_later")):
        cur = getattr(cfg, name)
        ok = cur > 0 and np.isfinite(got[key]) and got[key] > 0 and np.isfinite(target[key])
        new[name] = float(min(1.0, cur * target[key] / got[key])) if ok else cur
    return dataclasses.replace(cfg, **new)


# ------------------------------------------------------------------- reconstruction baselines

Imputer = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def impute_observed(observed: np.ndarray, flag: np.ndarray, hour_means: np.ndarray) -> np.ndarray:
    """Ignore censoring: demand = observed sales."""
    return observed.copy()


def impute_hour_mean(observed: np.ndarray, flag: np.ndarray, hour_means: np.ndarray) -> np.ndarray:
    """Flagged hours get the series' trailing in-stock mean for that hour (never below observed).
    Ignores that stockouts happen on high-demand days."""
    est = observed.copy()
    f = flag.astype(bool)
    f[:, :LO] = f[:, HI:] = False
    fill = np.maximum(observed, np.nan_to_num(hour_means))
    est[f] = fill[f]
    return est


def _impute_profile(
    observed: np.ndarray, flag: np.ndarray, hour_means: np.ndarray, pre: bool
) -> np.ndarray:
    m = np.nan_to_num(hour_means[:, LO:HI])
    tot = m.sum(axis=1, keepdims=True)
    share = np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)
    f = flag[:, LO:HI].astype(bool)
    cens = f.copy()
    if pre:  # the in-stock hour before a flagged hour is the likely depletion hour
        nxt = np.zeros_like(f)
        nxt[:, :-1] = f[:, 1:]
        cens |= ~f & nxt
    obs = observed[:, LO:HI]
    den = (share * ~cens).sum(axis=1)
    num = (obs * ~cens).sum(axis=1)
    level = np.where(den > 0, num / np.where(den > 0, den, 1.0), tot[:, 0])
    fill = np.maximum(obs, level[:, None] * share)
    est = observed.copy()
    w = est[:, LO:HI]
    w[cens] = fill[cens]
    return est


def impute_profile(observed: np.ndarray, flag: np.ndarray, hour_means: np.ndarray) -> np.ndarray:
    """Scale the series' hourly profile to the day's in-stock hours; fill flagged hours."""
    return _impute_profile(observed, flag, hour_means, pre=False)


def impute_profile_pre(
    observed: np.ndarray, flag: np.ndarray, hour_means: np.ndarray
) -> np.ndarray:
    """As ``profile``, but the hour before each stockout is also treated as censored."""
    return _impute_profile(observed, flag, hour_means, pre=True)


IMPUTERS: dict[str, Imputer] = {
    "observed": impute_observed,
    "hour_mean": impute_hour_mean,
    "profile": impute_profile,
    "profile_pre": impute_profile_pre,
}


# --------------------------------------------------------------------------- scoring frames


def recovery_frames(
    bm: Benchmark, estimates: dict[str, np.ndarray]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluation frames (sctwin.evaluation format) for reconstructed demand.

    hourly: one row per censored hour (depletion hour and flagged hours), y = demand proxy.
    daily: one row per stocked-out day, y = hours 6-21 demand total.
    Subgroups: availability class of the hour, depletion time (early < 12 <= mid < 17 <= late),
    restock within the day, and censoring a method cannot see (depletion in the last hour).
    """
    inj = bm.injection
    cls = availability_classes(inj.flag)
    r, h = np.nonzero(inj.censored)
    dep = inj.depletion
    groups = {
        "g_early": dep < 12,
        "g_mid": (dep >= 12) & (dep < 17),
        "g_late": dep >= 17,
        "g_restock": inj.restock >= 0,
    }
    hourly = pl.DataFrame(
        {
            "row": r,
            "series_idx": bm.series[r],
            "day_idx": bm.day[r],
            "hour": h,
            "cluster": bm.series[r],
            "y": bm.demand[r, h],
            "observed": inj.observed[r, h],
            "g_P_pre": cls[r, h] == CLS_P_PRE,
            "g_P_first": cls[r, h] == CLS_P_FIRST,
            "g_U": cls[r, h] == CLS_U,
            "g_undetected": cls[r, h] == CLS_A,
            **{g: v[r] for g, v in groups.items()},
            **{f"{m}__point": e[r, h] for m, e in estimates.items()},
        }
    )
    d = np.flatnonzero(inj.stocked_out)
    daily = pl.DataFrame(
        {
            "row": d,
            "series_idx": bm.series[d],
            "day_idx": bm.day[d],
            "cluster": bm.series[d],
            "y": bm.demand[d, LO:HI].sum(axis=1),
            "observed": inj.observed[d, LO:HI].sum(axis=1),
            **{g: v[d] for g, v in groups.items()},
            **{f"{m}__point": e[d, LO:HI].sum(axis=1) for m, e in estimates.items()},
        }
    )
    return hourly, daily


# --------------------------------------------------------------------------- fully synthetic panel


@dataclass(frozen=True)
class SyntheticParams:
    level_mean: float = -1.0  # mean log hourly demand at a profile value of 1
    level_sd: float = 0.7  # between-series spread of the log level
    dow: tuple[float, ...] = (0.0, -0.05, -0.05, 0.0, 0.1, 0.2, 0.15)  # log weekday effects
    rho: float = 0.6  # AR(1) coefficient of the daily log state
    sigma_z: float = 0.25  # innovation sd of the daily log state
    phi: float = 1.0  # Tweedie dispersion
    power: float = 1.5  # Tweedie power


def hour_profile() -> np.ndarray:
    """Default hourly demand profile: small at night, two daytime peaks. Mean 1 over 6-21."""
    h = np.arange(24)
    day = (
        0.6
        + 0.5 * np.exp(-0.5 * ((h - 11) / 2.0) ** 2)
        + 0.8 * np.exp(-0.5 * ((h - 18) / 2.0) ** 2)
    )
    p = np.where((h >= LO) & (h < HI), day, 0.05)
    return p / p[LO:HI].mean()


@dataclass
class SyntheticPanel:
    params: SyntheticParams
    profile: np.ndarray  # (24,)
    log_level: np.ndarray  # (S,)
    z: np.ndarray  # (S, D) daily latent state
    mu: np.ndarray  # (S, D, 24) mean demand
    demand: np.ndarray  # (S, D, 24) latent demand (known truth)


def simulate_latent(
    n_series: int, n_days: int, params: SyntheticParams, seed: int
) -> SyntheticPanel:
    """Latent demand D ~ Tweedie(mu, phi, power) with
    log mu[s, d, h] = log_level[s] + dow[d % 7] + z[s, d] + log profile[h],
    z[s, d] = rho z[s, d-1] + N(0, sigma_z^2), z stationary at d = 0."""
    from sctwin.baseline import tweedie_sample

    rng = np.random.default_rng(seed)
    a = rng.normal(params.level_mean, params.level_sd, n_series)
    z = np.empty((n_series, n_days))
    z[:, 0] = rng.normal(0, params.sigma_z / np.sqrt(1 - params.rho**2), n_series)
    for d in range(1, n_days):
        z[:, d] = params.rho * z[:, d - 1] + rng.normal(0, params.sigma_z, n_series)
    prof = hour_profile()
    dow = np.asarray(params.dow)[np.arange(n_days) % 7]
    mu = np.exp(a[:, None, None] + dow[None, :, None] + z[:, :, None]) * prof[None, None, :]
    dem = tweedie_sample(mu.ravel(), params.phi, params.power, 1, rng)[:, 0].reshape(mu.shape)
    return SyntheticPanel(params, prof, a, z, mu, dem)


def censor_synthetic(sp: SyntheticPanel, cfg: CensorConfig, seed: int) -> Injection:
    """Censor every synthetic day; the opening stock uses the true expected demand."""
    n_s, n_d, _ = sp.demand.shape
    rng = np.random.default_rng(seed)
    expected = sp.mu[:, :, LO:HI].sum(axis=-1).ravel()
    inj = inject_stockouts(sp.demand.reshape(-1, 24), stock_levels(expected, rng, cfg), rng, cfg)
    return Injection(
        observed=inj.observed.reshape(n_s, n_d, 24),
        flag=inj.flag.reshape(n_s, n_d, 24),
        depletion=inj.depletion.reshape(n_s, n_d),
        restock=inj.restock.reshape(n_s, n_d),
        censored=inj.censored.reshape(n_s, n_d, 24),
    )
