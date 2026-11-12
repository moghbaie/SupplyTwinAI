"""Evaluation harness: metrics per model, per subgroup, and paired comparisons.

A predictions frame has one row per forecast target with:
  * ``y`` (observed outcome), ``cluster`` (bootstrap cluster id, e.g. series_idx)
  * per model ``m``: ``{m}__point`` and optionally ``{m}__q{tau}`` quantile columns
  * boolean subgroup columns named ``g_<name>``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from sctwin import metrics as M


def qcol(model: str, tau: float) -> str:
    return f"{model}__q{tau:g}"


def model_names(df: pl.DataFrame) -> list[str]:
    return sorted({c.split("__")[0] for c in df.columns if "__" in c})


def _quantiles(df: pl.DataFrame, model: str, taus: tuple[float, ...]) -> np.ndarray | None:
    cols = [qcol(model, t) for t in taus]
    if not all(c in df.columns for c in cols):
        return None
    return df.select(cols).to_numpy().astype(np.float64)


def score_frame(df: pl.DataFrame, taus: tuple[float, ...]) -> pl.DataFrame:
    """Add per-row loss columns ``{m}__ae`` and ``{m}__wis`` / ``{m}__crpsq``."""
    y = df["y"].to_numpy().astype(np.float64)
    new = {}
    for m in model_names(df):
        if f"{m}__point" in df.columns:
            new[f"{m}__ae"] = np.abs(y - df[f"{m}__point"].to_numpy())
        q = _quantiles(df, m, taus)
        if q is not None:
            new[f"{m}__wis"] = M.wis(y, q, taus)
            new[f"{m}__crpsq"] = M.crps_quantiles(y, q, taus)
    return df.with_columns(**{k: pl.Series(v) for k, v in new.items()})


def summarize(df: pl.DataFrame, taus: tuple[float, ...]) -> dict[str, Any]:
    """Metrics for every model on the rows of ``df`` (already scored)."""
    y = df["y"].to_numpy().astype(np.float64)
    out: dict[str, Any] = {"n": int(df.height), "n_clusters": int(df["cluster"].n_unique()),
                           "y_mean": float(y.mean()) if y.size else None, "models": {}}  # fmt: skip
    for m in model_names(df):
        r: dict[str, Any] = {}
        if f"{m}__point" in df.columns:
            f = df[f"{m}__point"].to_numpy()
            r["mae"] = M.mae(y, f)
            r["wape"] = M.wape(y, f)
            r["bias"] = float(np.mean(f - y))
        q = _quantiles(df, m, taus)
        if q is not None:
            r["wis"] = float(df[f"{m}__wis"].mean())  # type: ignore[arg-type]
            r["crps_q"] = float(df[f"{m}__crpsq"].mean())  # type: ignore[arg-type]
            r["intervals"] = M.interval_coverage(y, q, taus)
            if "point" not in r and 0.5 in taus:
                med = q[:, taus.index(0.5)]
                r["mae_median"] = M.mae(y, med)
                r["wape_median"] = M.wape(y, med)
        out["models"][m] = r
    return out


def evaluate(df: pl.DataFrame, taus: tuple[float, ...]) -> dict[str, Any]:
    scored = score_frame(df, taus)
    res: dict[str, Any] = {"overall": summarize(scored, taus), "folds": {}, "subgroups": {}}
    if "origin_idx" in scored.columns:  # Rule 1: per-fold and aggregated
        for (o,), sub in scored.group_by(["origin_idx"], maintain_order=True):
            res["folds"][str(o)] = summarize(sub, taus)
    for g in [c for c in scored.columns if c.startswith("g_")]:
        sub = scored.filter(pl.col(g))
        if sub.height:
            res["subgroups"][g[2:]] = summarize(sub, taus)
    return res


def compare_to_reference(
    df: pl.DataFrame,
    reference: str,
    loss: str,
    taus: tuple[float, ...],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired cluster bootstrap of ``loss`` for every model vs ``reference``, Holm-adjusted.

    Negative mean_diff means the model has lower loss than the reference.
    """
    scored = score_frame(df, taus)
    ref = f"{reference}__{loss}"
    others = [m for m in model_names(df) if m != reference and f"{m}__{loss}" in scored.columns]
    cl = scored["cluster"].to_numpy()
    res: dict[str, dict[str, Any]] = {}
    ref_mean = float(np.mean(scored[ref].to_numpy()))
    for m in others:
        r = M.paired_bootstrap(scored[f"{m}__{loss}"].to_numpy(), scored[ref].to_numpy(), cl,
                               n_boot=n_boot, seed=seed)  # fmt: skip
        res[m] = {
            "mean_diff": r.mean_diff,
            "ci": [r.ci_low, r.ci_high],
            "p": r.p_value,
            "relative": r.mean_diff / ref_mean,
        }
    adj = M.holm([float(v["p"]) for v in res.values()])
    for (_, v), a in zip(res.items(), adj, strict=True):
        v["p_holm"] = a
    return {"reference": reference, "loss": loss, "comparisons": res}
