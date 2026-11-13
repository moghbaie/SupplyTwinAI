"""Step 3 baseline forecasters for observed sales.

All models predict day t from the as-of features of :mod:`sctwin.features` (origin = end of
day t-1). They forecast *observed* sales; nothing here estimates latent demand.

* ``snaive``   - seasonal naive point forecast (same weekday last week / same hour last week)
* ``clim``     - empirical quantiles of the last n seasonal-position values observed by T
* ``glm``      - pooled Tweedie GLM (log link); predictive quantiles by simulating the
                 compound Poisson-Gamma distribution with a Pearson dispersion estimate
* ``lgbm``     - LightGBM point forecast (Tweedie objective)
* ``lgbm_q``   - LightGBM quantile regression, one model per tau, crossings fixed by sorting
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


# --------------------------------------------------------------------------- GLM


@dataclass
class GLMDesign:
    """Feature transformation for the GLM, fitted on training rows only."""

    numeric: list[str]
    categorical: list[str]
    log_cols: list[str]
    no_missing_flag: list[str] = field(default_factory=list)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    levels: dict[str, list[Any]] = field(default_factory=dict)

    def fit(self, df: pl.DataFrame) -> GLMDesign:
        for c in self.numeric:
            v = self._num(df, c)
            v = v[~np.isnan(v)]
            # an all-missing column becomes a constant 0 plus its missing indicator
            self.means[c] = float(v.mean()) if v.size else 0.0
            self.stds[c] = (float(v.std()) if v.size else 0.0) or 1.0
        for c in self.categorical:
            self.levels[c] = sorted(df[c].unique().to_list())[1:]  # drop first level
        return self

    def _num(self, df: pl.DataFrame, c: str) -> np.ndarray:
        v = df[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(np.float64)
        return np.log1p(v) if c in self.log_cols else v

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        cols = []
        for c in self.numeric:
            v = self._num(df, c)
            miss = np.isnan(v)
            cols.append(np.where(miss, 0.0, (v - self.means[c]) / self.stds[c]))
            if c not in self.no_missing_flag:
                cols.append(miss.astype(np.float64))
        for c in self.categorical:
            v = df[c].to_numpy()
            cols.extend((v == lev).astype(np.float64) for lev in self.levels[c])
        return np.column_stack(cols)


def fit_tweedie_glm(
    x: np.ndarray, y: np.ndarray, power: float = 1.5, alpha: float = 1e-4, max_iter: int = 300
) -> tuple[Any, float]:
    """Fit a log-link Tweedie GLM; return (model, Pearson dispersion phi)."""
    from sklearn.linear_model import TweedieRegressor

    m = TweedieRegressor(power=power, link="log", alpha=alpha, max_iter=max_iter)
    with warnings.catch_warnings():
        # scikit-learn < 1.6 passes `iprint` to SciPy's L-BFGS-B, deprecated since SciPy 1.15
        # (seen on Colab's preinstalled versions); harmless, and not ours to fix.
        warnings.filterwarnings(
            "ignore", message=r".*`disp` and `iprint` options", category=DeprecationWarning
        )
        m.fit(x, y)
    mu = m.predict(x)
    dof = max(len(y) - x.shape[1] - 1, 1)
    phi = float(np.sum((y - mu) ** 2 / np.maximum(mu, 1e-9) ** power) / dof)
    return m, phi


def tweedie_sample(
    mu: np.ndarray, phi: float, power: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Samples (len(mu), n) from compound Poisson-Gamma Tweedie(mu, phi, 1 < power < 2)."""
    mu = np.asarray(mu, dtype=np.float64)[:, None]
    lam = mu ** (2 - power) / (phi * (2 - power))
    shape = (2 - power) / (power - 1)
    scale = phi * (power - 1) * mu ** (power - 1)
    counts = rng.poisson(np.broadcast_to(lam, (mu.shape[0], n)))
    out = np.zeros(counts.shape)
    pos = counts > 0
    out[pos] = rng.gamma(counts[pos] * shape, np.broadcast_to(scale, counts.shape)[pos])
    return out


def tweedie_quantiles(
    mu: np.ndarray,
    phi: float,
    power: float,
    taus: tuple[float, ...] = TAUS,
    n: int = 400,
    seed: int = 0,
    batch: int = 50_000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((len(mu), len(taus)))
    for i in range(0, len(mu), batch):
        s = tweedie_sample(mu[i : i + batch], phi, power, n, rng)
        out[i : i + batch] = np.quantile(s, taus, axis=1).T
    return out


# --------------------------------------------------------------------------- LightGBM

LGBM_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
}


def fit_lgbm(
    x: pl.DataFrame,
    y: np.ndarray,
    categorical: list[str],
    objective: str,
    num_rounds: int,
    seed: int,
    num_threads: int,
    alpha: float | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    import lightgbm as lgb

    prm = {**LGBM_DEFAULTS, **(params or {}), "objective": objective, "seed": seed,
           "num_threads": num_threads}  # fmt: skip
    if alpha is not None:
        prm["alpha"] = alpha
    ds = lgb.Dataset(x.to_pandas(), label=y, categorical_feature=categorical, free_raw_data=True)
    return lgb.train(prm, ds, num_boost_round=num_rounds)


def predict_lgbm(model: Any, x: pl.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(x.to_pandas(), num_threads=model.params.get("num_threads", 0)))
