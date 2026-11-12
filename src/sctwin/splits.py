"""Temporal and group hold-out splits with immutable manifests.

Roles:
  fit   - first ``fit_days`` dates of the official train split, non-held-out series
  val   - remaining dates of official train, non-held-out series
  test  - official eval split (locked; accessed only via the test ledger, Step 10)
  group hold-out - series whose store OR product is held out; never used for fitting or tuning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sctwin.data import SERIES_KEY

MANIFEST_VERSION = 1


def _pick(ids: np.ndarray, frac: float, seed: int, salt: str) -> list[int]:
    """Deterministic subset independent of input order: rank ids by a seeded hash."""
    ids = np.unique(ids)
    n = int(round(frac * ids.size))
    keyed = sorted(
        ids.tolist(),
        key=lambda i: hashlib.sha256(f"{seed}:{salt}:{i}".encode()).hexdigest(),
    )
    return sorted(int(i) for i in keyed[:n])


def make_manifest(
    train: pl.DataFrame,
    *,
    fit_days: int,
    val_days: int,
    holdout_store_frac: float,
    holdout_product_frac: float,
    seed: int,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dates = sorted(train["dt"].str.to_date().unique().to_list())
    if len(dates) != fit_days + val_days:
        raise ValueError(
            f"train has {len(dates)} distinct dates, config expects {fit_days}+{val_days}"
        )
    stores = _pick(train["store_id"].to_numpy(), holdout_store_frac, seed, "store")
    products = _pick(train["product_id"].to_numpy(), holdout_product_frac, seed, "product")
    series = train.select(SERIES_KEY).unique()
    hs, hp = pl.col("store_id").is_in(stores), pl.col("product_id").is_in(products)
    counts = series.select(
        total=pl.len(),
        in_sample=(~hs & ~hp).sum(),
        holdout_store_only=(hs & ~hp).sum(),
        holdout_product_only=(~hs & hp).sum(),
        holdout_both=(hs & hp).sum(),
    ).row(0, named=True)
    return {
        "manifest_version": MANIFEST_VERSION,
        "seed": seed,
        "source": source or {},
        "temporal": {
            "fit": {"start": str(dates[0]), "end": str(dates[fit_days - 1]), "n_days": fit_days},
            "val": {"start": str(dates[fit_days]), "end": str(dates[-1]), "n_days": val_days},
            "test": {"split": "official eval", "locked": True},
        },
        "group_holdout": {
            "rule": "series is held out if store_id in stores OR product_id in products",
            "store_frac": holdout_store_frac,
            "product_frac": holdout_product_frac,
            "stores": stores,
            "products": products,
        },
        "series_counts": {k: int(v) for k, v in counts.items()},
    }


def manifest_digest(m: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


def write_manifest(m: dict[str, Any], path: str | Path, force: bool = False) -> Path:
    """Write once. Re-writing identical content is a no-op; different content needs force."""
    p = Path(path)
    if p.exists() and not force:
        old = json.loads(p.read_text(encoding="utf-8"))
        old_digest = old.pop("digest", None)
        if old_digest != manifest_digest(m):
            raise FileExistsError(f"{p} exists with different content; pass force=True to replace")
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**m, "digest": manifest_digest(m)}, indent=2), encoding="utf-8")
    return p


def read_manifest(path: str | Path) -> dict[str, Any]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = m.pop("digest")
    if digest != manifest_digest(m):
        raise ValueError(f"manifest {path} was modified after writing (digest mismatch)")
    return m


def assign_roles(df: pl.DataFrame, m: dict[str, Any]) -> pl.DataFrame:
    """Add ``role`` (fit/val) and ``holdout`` (bool) columns to official-train rows."""
    fit_end = pl.lit(m["temporal"]["fit"]["end"]).str.to_date()
    g = m["group_holdout"]
    return df.with_columns(
        role=pl.when(pl.col("dt").str.to_date() <= fit_end)
        .then(pl.lit("fit"))
        .otherwise(pl.lit("val")),
        holdout=pl.col("store_id").is_in(g["stores"]) | pl.col("product_id").is_in(g["products"]),
    )
