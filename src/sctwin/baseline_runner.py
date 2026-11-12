"""Rolling-origin backtest of the Step 3 baselines, in a configurable unit.

For every fold origin T: training examples with t <= T - L - H (full target windows observed by
T), verified by ``check_split``; every fitted statistic (GLM scaling/imputation, model
parameters) is fitted on that fold's training rows only; predictions for (T + L, T + L + H].
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import polars as pl

from sctwin.backtest import BacktestSpec, check_split
from sctwin.baseline import GLMDesign, fit_lgbm, fit_tweedie_glm, predict_lgbm, tweedie_quantiles
from sctwin.evaluation import qcol
from sctwin.features import FeatureBuilder
from sctwin.metrics import sort_quantiles
from sctwin.panel import CLS_A, CLS_NIGHT, CLS_P_FIRST, CLS_P_PRE, CLS_U

LGBM_CATEGORICAL = ["management_group_id", "first_category_id", "city_id"]


def _glm_design(fb: FeatureBuilder) -> GLMDesign:
    cat = ["target_dow", "management_group_id"]
    if fb.us.unit == "hour":
        cat.append("target_hour")
    num = [f for f in fb.feature_names if f not in cat]
    log_cols = [c for c in num if c.startswith(("sale_", "instock_mean"))]
    return GLMDesign(numeric=num, categorical=cat, log_cols=log_cols,
                     no_missing_flag=["steps_ahead", "target_holiday"])  # fmt: skip


def clim_quantiles(fb: FeatureBuilder, test: pl.DataFrame, taus: tuple[float, ...]) -> np.ndarray:
    """Empirical quantiles of the last n_seasons values at the target's seasonal position that
    are observed at the origin."""
    vals = fb.seasonal_values(
        test["series_idx"].to_numpy(), test["origin_idx"].to_numpy().astype(np.int64),
        test["target_idx"].to_numpy().astype(np.int64), fb.cfg.season, fb.cfg.n_seasons,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(vals, taus, axis=1).T


def run_backtest(
    fb: FeatureBuilder,
    spec: BacktestSpec,
    train_series: np.ndarray,
    eval_series: np.ndarray,
    cfg: dict[str, Any],
    log: Any = print,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Returns (predictions, per-fold log)."""
    if spec.unit != fb.us.unit:
        raise ValueError(f"spec unit {spec.unit} != feature unit {fb.us.unit}")
    taus = tuple(cfg["taus"])
    features = list(fb.feature_names)
    lgbm_x = features + LGBM_CATEGORICAL
    seed = int(cfg["seed"])
    lg, gl = cfg["lgbm"], cfg["glm"]
    out, fitlog = [], []

    for o in spec.origins:
        t0 = time.perf_counter()
        train = fb.rows(spec.train_times(o), spec.lead, spec.horizon, series_idx=train_series)
        test = fb.rows([o], spec.lead, spec.horizon, series_idx=eval_series)
        check_split(train, test, o, spec)
        y_tr = train["y"].to_numpy().astype(np.float64)
        pred: dict[str, np.ndarray] = {}

        pred["snaive__point"] = test["sale_seasonal_last"].to_numpy()
        cq = clim_quantiles(fb, test, taus)
        for k, tau in enumerate(taus):
            pred[qcol("clim", tau)] = cq[:, k]

        design = _glm_design(fb).fit(train)  # scaling/imputation fitted on this fold's train
        glm, phi = fit_tweedie_glm(design.transform(train), y_tr, power=gl["power"],
                                   alpha=gl["alpha"])  # fmt: skip
        mu = glm.predict(design.transform(test))
        pred["glm__point"] = mu
        gq = tweedie_quantiles(mu, phi, gl["power"], taus, n=gl["n_samples"], seed=seed + o)
        for k, tau in enumerate(taus):
            pred[qcol("glm", tau)] = gq[:, k]

        xtr, xte = train.select(lgbm_x), test.select(lgbm_x)
        common = dict(categorical=LGBM_CATEGORICAL, num_rounds=lg["num_rounds"], seed=seed,
                      num_threads=lg["num_threads"], params=lg.get("params"))  # fmt: skip
        pred["lgbm__point"] = predict_lgbm(fit_lgbm(xtr, y_tr, objective="tweedie", **common), xte)
        qs = np.column_stack(
            [
                predict_lgbm(fit_lgbm(xtr, y_tr, objective="quantile", alpha=t, **common), xte)
                for t in taus
            ]  # fmt: skip
        )
        qs, crossed = sort_quantiles(np.maximum(qs, 0.0))
        for k, tau in enumerate(taus):
            pred[qcol("lgbm_q", tau)] = qs[:, k]

        keep = ["series_idx", "origin_idx", "target_idx", "step", "steps_ahead", "target_dt",
                "y_day", "y_stockout", "management_group_id", "store_id", "product_id"]  # fmt: skip
        block = test.select(keep).with_columns(
            y=pl.Series(test["y"].to_numpy().astype(np.float64)),
            cluster=pl.col("series_idx"),
            **{k: pl.Series(v.astype(np.float64)) for k, v in pred.items()},
        )
        out.append(block)
        t_range = (
            [int(train["origin_idx"].min()), int(train["origin_idx"].max())]  # type: ignore[arg-type]
            if train.height
            else None
        )
        rec = {"origin": int(o), "train_rows": train.height, "test_rows": test.height,
               "train_t_range": t_range, "glm_phi": phi, "lgbm_q_crossing_frac": crossed,
               "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
        fitlog.append(rec)
        log(f"[{spec.unit}] origin {o}: {rec}")
    return pl.concat(out), fitlog


def add_groups(pred: pl.DataFrame, fb: FeatureBuilder, low_volume_series: set[int]) -> pl.DataFrame:
    """Boolean subgroup columns (``g_*``) for evaluation. Uses target-time information, which
    is allowed for *grouping outcomes* only, never as features."""
    us = fb.us
    s = pred["series_idx"].to_numpy()
    act = us.activity[s, pred["y_day"].to_numpy()]
    g: dict[str, Any] = {
        "g_promo": pl.Series(act == 1),
        "g_no_promo": pl.Series(act == 0),
        "g_low_volume": pl.col("series_idx").is_in(sorted(low_volume_series)),
    }
    if us.unit == "day":
        g["g_available"] = pl.col("y_stockout") == 0
        g["g_censored"] = pl.col("y_stockout") > 0
    else:
        assert us.eval_class is not None
        cls = us.eval_class[s, pred["target_idx"].to_numpy()]
        for name, code in (("A", CLS_A), ("P_pre", CLS_P_PRE), ("P_first", CLS_P_FIRST),
                           ("U", CLS_U), ("night", CLS_NIGHT)):  # fmt: skip
            g[f"g_class_{name}"] = pl.Series(cls == code)
    for k in sorted(pred["steps_ahead"].unique().to_list()):
        g[f"g_steps_ahead_{int(k)}"] = pl.col("steps_ahead") == k
    return pred.with_columns(**g)
