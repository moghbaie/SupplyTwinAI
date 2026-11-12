"""Dataset acquisition, schema validation and representation conversion.

Storage representation is the compact one shipped by the provider: one row per
(store_id, product_id, dt) with 24-element hourly arrays. Hourly long format is
materialized only on demand via :func:`to_hourly_long`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sctwin.repro import sha256_file

REPO_ID = "Dingdong-Inc/FreshRetailNet-50K"
SPLIT_FILES = {"train": "data/train.parquet", "eval": "data/eval.parquet"}
KEY = ("store_id", "product_id", "dt")
SERIES_KEY = ("store_id", "product_id")
N_HOURS = 24

# Expected column -> polars dtype family. Integer widths are not enforced (provider may vary).
SCHEMA: dict[str, str] = {
    "city_id": "int",
    "store_id": "int",
    "management_group_id": "int",
    "first_category_id": "int",
    "second_category_id": "int",
    "third_category_id": "int",
    "product_id": "int",
    "dt": "str",
    "sale_amount": "float",
    "hours_sale": "list[float]",
    "stock_hour6_22_cnt": "int",
    "hours_stock_status": "list[int]",
    "discount": "float",
    "holiday_flag": "int",
    "activity_flag": "int",
    "precpt": "float",
    "avg_temperature": "float",
    "avg_humidity": "float",
    "avg_wind_level": "float",
}


# --------------------------------------------------------------------------- fetch


def fetch(
    out_dir: str | Path, revision: str | None = None, repo_id: str = REPO_ID
) -> dict[str, Any]:
    """Download both splits at a pinned Hub commit and write ``provenance.json``.

    If ``revision`` is None the current head commit SHA is resolved and pinned.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision)
    sha = info.sha
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for split, fname in SPLIT_FILES.items():
        local = hf_hub_download(
            repo_id, fname, repo_type="dataset", revision=sha, local_dir=str(out)
        )
        p = Path(local)
        files[split] = {
            "hub_path": fname,
            "local_path": p.relative_to(out).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
    prov = {
        "repo_id": repo_id,
        "revision": sha,
        "license": "CC BY 4.0",
        "card_version": "1.0 (05/08/2025)",
        "files": files,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return prov


def verify_checksums(data_dir: str | Path) -> dict[str, bool]:
    """Re-hash local files against provenance.json (immutability check)."""
    d = Path(data_dir)
    prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    return {
        split: sha256_file(d / meta["local_path"]) == meta["sha256"]
        for split, meta in prov["files"].items()
    }


def load_split(data_dir: str | Path, split: str) -> pl.DataFrame:
    d = Path(data_dir)
    prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    return pl.read_parquet(d / prov["files"][split]["local_path"])


# --------------------------------------------------------------------------- schema


def _family(dtype: pl.DataType) -> str:
    if dtype.is_integer():
        return "int"
    if dtype.is_float():
        return "float"
    if dtype == pl.String:
        return "str"
    if isinstance(dtype, pl.List | pl.Array):
        return f"list[{_family(dtype.inner)}]"  # type: ignore[arg-type]
    return str(dtype)


@dataclass
class SchemaReport:
    missing: list[str]
    unexpected: list[str]
    wrong_type: dict[str, tuple[str, str]]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.wrong_type)


def validate_schema(df: pl.DataFrame | pl.LazyFrame) -> SchemaReport:
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    missing = [c for c in SCHEMA if c not in schema]
    unexpected = [c for c in schema if c not in SCHEMA]
    wrong = {
        c: (SCHEMA[c], _family(schema[c]))
        for c in SCHEMA
        if c in schema and _family(schema[c]) != SCHEMA[c]
    }
    return SchemaReport(missing, unexpected, wrong)


# --------------------------------------------------------------------------- representations


def hourly_matrix(df: pl.DataFrame, col: str, dtype: Any = np.float64) -> np.ndarray:
    """(n_rows, 24) array from a list column. Raises if any row is not length 24."""
    lens = df[col].list.len()
    lo, hi = int(lens.min()), int(lens.max())  # type: ignore[arg-type]
    if lo != N_HOURS or hi != N_HOURS:
        raise ValueError(f"{col}: list lengths in [{lo}, {hi}], expected 24")
    return df[col].list.to_array(N_HOURS).to_numpy().astype(dtype, copy=False)


def to_hourly_long(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Expand daily rows with hourly arrays to one row per (series, dt, hour). Lazy."""
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    return (
        lf.with_columns(hour=pl.int_ranges(0, pl.col("hours_sale").list.len()))
        .explode(["hours_sale", "hours_stock_status", "hour"], empty_as_null=True)
        .rename({"hours_sale": "sale", "hours_stock_status": "stockout"})
        .with_columns(pl.col("hour").cast(pl.Int8))
    )
