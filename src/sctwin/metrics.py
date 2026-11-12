"""Forecast evaluation metrics and paired comparison statistics.

Conventions: ``y`` has shape (n,). Quantile forecasts have shape (n, K) with matching ``taus``
(K,). Sample forecasts have shape (n, m). All scores are "lower is better" and returned as
per-observation arrays unless the name says otherwise, so they can be aggregated by subgroup
and passed to :func:`paired_bootstrap`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- point


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def wape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-12) -> float | None:
    """sum|y - yhat| / sum|y|. Returns None (not inf/NaN) when the denominator is ~0."""
    y = np.asarray(y, dtype=np.float64)
    den = float(np.abs(y).sum())
    if den <= eps:
        return None
    return float(np.abs(y - np.asarray(yhat)).sum() / den)


# --------------------------------------------------------------------------- quantiles


def sort_quantiles(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Fix quantile crossing by sorting along the last axis. Returns (sorted, frac_rows_fixed)."""
    q = np.asarray(q, dtype=np.float64)
    crossed = np.any(np.diff(q, axis=-1) < 0, axis=-1)
    return np.sort(q, axis=-1), float(crossed.mean()) if crossed.size else 0.0


def pinball(y: np.ndarray, q: np.ndarray, taus: Sequence[float]) -> np.ndarray:
    """Quantile (pinball) loss, shape (n, K)."""
    y = np.asarray(y, dtype=np.float64)[:, None]
    q = np.asarray(q, dtype=np.float64)
    t = np.asarray(taus, dtype=np.float64)[None, :]
    return np.where(y < q, (1 - t) * (q - y), t * (y - q))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    """Gneiting & Raftery (2007) interval score for a central (1 - alpha) interval."""
    y, lo, hi = (np.asarray(a, dtype=np.float64) for a in (y, lo, hi))
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def _central_pairs(taus: Sequence[float]) -> tuple[int, list[tuple[int, int, float]]]:
    t = np.round(np.asarray(taus, dtype=np.float64), 10)
    med = np.flatnonzero(np.isclose(t, 0.5))
    if med.size != 1:
        raise ValueError("WIS needs the median (tau = 0.5) among taus")
    pairs = []
    for i, ti in enumerate(t):
        if ti < 0.5:
            j = np.flatnonzero(np.isclose(t, 1 - ti))
            if j.size != 1:
                raise ValueError(f"tau {ti} has no symmetric partner {1 - ti}")
            pairs.append((i, int(j[0]), float(2 * ti)))
    if 2 * len(pairs) + 1 != t.size:
        raise ValueError("taus must be the median plus symmetric pairs")
    return int(med[0]), pairs


def wis(y: np.ndarray, q: np.ndarray, taus: Sequence[float]) -> np.ndarray:
    """Weighted interval score (Bracher et al. 2021), per observation.

    WIS = (0.5|y - m| + sum_k (alpha_k / 2) IS_alpha_k) / (K + 0.5).
    """
    q = np.asarray(q, dtype=np.float64)
    m, pairs = _central_pairs(taus)
    total = 0.5 * np.abs(np.asarray(y, dtype=np.float64) - q[:, m])
    for lo, hi, alpha in pairs:
        total = total + (alpha / 2) * interval_score(y, q[:, lo], q[:, hi], alpha)
    return total / (len(pairs) + 0.5)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_coverage(
    y: np.ndarray, q: np.ndarray, taus: Sequence[float]
) -> dict[str, dict[str, float]]:
    """Empirical coverage and mean width of every central interval in the quantile set."""
    _, pairs = _central_pairs(taus)
    out = {}
    for lo, hi, alpha in pairs:
        level = round(1 - alpha, 6)
        out[str(level)] = {
            "nominal": level,
            "coverage": coverage(y, q[:, lo], q[:, hi]),
            "mean_width": float(np.mean(q[:, hi] - q[:, lo])),
        }
    return out


# --------------------------------------------------------------------------- samples


def crps_samples(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """CRPS from an ensemble: E|X - y| - 0.5 E|X - X'| (empirical-distribution form)."""
    x = np.sort(np.asarray(samples, dtype=np.float64), axis=1)
    y = np.asarray(y, dtype=np.float64)
    m = x.shape[1]
    term1 = np.mean(np.abs(x - y[:, None]), axis=1)
    # E|X - X'| for the empirical distribution = (2 / m^2) * sum_i (2i - m - 1) x_(i)
    w = 2 * np.arange(1, m + 1) - m - 1
    term2 = (2.0 / m**2) * (x @ w)
    return term1 - 0.5 * term2


def crps_quantiles(y: np.ndarray, q: np.ndarray, taus: Sequence[float]) -> np.ndarray:
    """CRPS approximated as 2 x mean pinball loss over an (approximately uniform) tau grid.

    Exact in the limit of a dense uniform grid; a coarse grid gives a biased approximation,
    so compare models only on the same grid.
    """
    return 2.0 * pinball(y, q, taus).mean(axis=1)


def crps_normal(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma^2) (Gneiting et al. 2005). Used as a reference."""
    z = (np.asarray(y) - mu) / sigma
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def randomized_pit(
    y: np.ndarray, samples: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Randomized PIT from samples; handles point masses (e.g. zero sales) correctly."""
    rng = rng or np.random.default_rng(0)
    s = np.asarray(samples)
    y = np.asarray(y)[:, None]
    below = (s < y).mean(axis=1)
    equal = (s == y).mean(axis=1)
    return below + rng.uniform(size=below.shape) * equal


# --------------------------------------------------------------------------- comparison


@dataclass
class BootstrapResult:
    mean_diff: float  # mean(loss_a - loss_b); negative => A better
    ci_low: float
    ci_high: float
    p_value: float  # two-sided, from the bootstrap distribution
    n_clusters: int
    n_obs: int


def paired_bootstrap(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Cluster (block) bootstrap of the mean paired loss difference.

    Whole clusters (e.g. series or stores) are resampled with replacement, preserving temporal
    and within-cluster dependence. The statistic is the observation-weighted mean difference.
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    _, cid = np.unique(np.asarray(clusters), return_inverse=True)
    n_cl = int(np.max(cid)) + 1
    sums = np.bincount(cid, weights=d, minlength=n_cl)
    cnts = np.bincount(cid, minlength=n_cl).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_cl, size=(n_boot, n_cl))
    boot = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    est = float(sums.sum() / cnts.sum())
    a = (1 - level) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    # Two-sided p-value: how often the centred bootstrap distribution is as extreme as est.
    p = float(np.mean(np.abs(boot - est) >= abs(est)))
    return BootstrapResult(est, float(lo), float(hi), max(p, 1.0 / n_boot), n_cl, d.size)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    n = p.size
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()
