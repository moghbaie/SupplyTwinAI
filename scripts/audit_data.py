"""Run the Step 1 audit and write split manifests + the 20-series development subset.

Usage: python scripts/audit_data.py [--config configs/small.yaml]
Outputs (all under data/, never committed except the manifests copied to configs/manifests/):
  data/audit/audit_report.json
  data/audit/run_manifest.json
  configs/manifests/splits_v1.json      (immutable)
  data/subsets/dev20.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from sctwin.audit import run_audit
from sctwin.data import SERIES_KEY, load_split, verify_checksums
from sctwin.repro import load_config, seed_everything, write_run_manifest
from sctwin.splits import assign_roles, make_manifest, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["seed"])
    data_dir = Path(cfg["dataset"]["data_dir"])

    checks = verify_checksums(data_dir)
    if not all(checks.values()):
        raise SystemExit(f"checksum mismatch: {checks}")
    prov = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    train, test = load_split(data_dir, "train"), load_split(data_dir, "eval")
    report = run_audit(train, test)
    report["provenance"] = prov
    report["runtime_s"] = round(time.perf_counter() - t0, 1)

    out = Path("data/audit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_run_manifest(out, cfg, stage="01_data_audit", all_gates_pass=report["all_gates_pass"])

    sp = cfg["splits"]
    m = make_manifest(
        train,
        fit_days=sp["fit_days"],
        val_days=sp["val_days"],
        holdout_store_frac=sp["holdout_store_frac"],
        holdout_product_frac=sp["holdout_product_frac"],
        seed=cfg["seed"],
        source={
            "repo_id": prov["repo_id"],
            "revision": prov["revision"],
            "train_sha256": prov["files"]["train"]["sha256"],
        },
    )
    write_manifest(m, "configs/manifests/splits_v1.json")

    # Development subset: in-sample series only, seeded sample (train + val days only).
    tr = assign_roles(train, m).filter(~pl.col("holdout"))
    keys = tr.select(SERIES_KEY).unique().sort(SERIES_KEY)
    keys = keys.sample(n=cfg["subset"]["n_series"], seed=cfg["seed"])
    sub = tr.join(keys, on=list(SERIES_KEY), how="semi")
    Path("data/subsets").mkdir(parents=True, exist_ok=True)
    sub.write_parquet("data/subsets/dev20.parquet")

    print(
        json.dumps(
            {
                "gates": report["gates"],
                "all_gates_pass": report["all_gates_pass"],
                "series_counts": m["series_counts"],
                "runtime_s": report["runtime_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
