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
