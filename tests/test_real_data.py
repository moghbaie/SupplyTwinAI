"""Checks against the downloaded dataset. Run with `pytest -m data`; skipped when absent."""

import json
from pathlib import Path

import polars as pl
import pytest

from sctwin.data import verify_checksums
from sctwin.splits import assign_roles, read_manifest

RAW = Path("data/raw")
REPORT = Path("data/audit/audit_report.json")
MANIFEST = Path("configs/manifests/splits_v1.json")

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not (RAW / "provenance.json").exists(), reason="dataset not downloaded"),
]


def test_checksums_unchanged():
    assert all(verify_checksums(RAW).values())


def test_audit_gates_pass():
    rep = json.loads(REPORT.read_text())
    assert rep["all_gates_pass"], {k: v for k, v in rep["gates"].items() if not v}


def test_manifest_matches_pinned_data():
    m = read_manifest(MANIFEST)
    prov = json.loads((RAW / "provenance.json").read_text())
    assert m["source"]["revision"] == prov["revision"]
    assert m["source"]["train_sha256"] == prov["files"]["train"]["sha256"]


def test_dev_subset_is_in_sample_and_pre_test():
    m = read_manifest(MANIFEST)
    sub = pl.read_parquet("data/subsets/dev20.parquet")
    assert not assign_roles(sub, m)["holdout"].any()
    assert sub["dt"].max() <= m["temporal"]["val"]["end"]


def test_real_panel_and_no_leakage():
    """Leakage perturbation test on the real 20-series development subset."""
    from test_no_leakage import perturb_after

    from sctwin.features import (
        daily_feature_names,
        daily_features,
        hourly_feature_names,
        hourly_features,
    )
    from sctwin.panel import build_panel

    sub = pl.read_parquet("data/subsets/dev20.parquet").drop(["role", "holdout"])
    p = build_panel(sub)
    assert p.shape == (20, 90)
    for t0 in (10, 45, 88):
        for planned in (True, False):
            q = perturb_after(p, t0, seed=t0, planned_promo=planned)
            for fn, names in (
                (daily_features, daily_feature_names(planned)),
                (hourly_features, hourly_feature_names(planned)),
            ):
                a = fn(p, planned_promo=planned).filter(pl.col("t_idx") <= t0)
                b = fn(q, planned_promo=planned).filter(pl.col("t_idx") <= t0)
                cols = list(names)
                assert a.select(cols).to_numpy().tobytes() == b.select(cols).to_numpy().tobytes()
