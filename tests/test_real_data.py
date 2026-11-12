"""Checks against the downloaded dataset. Run with `pytest -m data`; skipped when absent."""

import json
from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize("unit", ["day", "hour"])
def test_real_features_and_training_sets_ignore_data_after_origin(unit):
    """Rules 1 and 2 on the real 20-series development subset: features at every origin <= T
    and the training set (features and targets) of the fold at T are unchanged when all data
    after T is replaced."""
    from test_no_leakage import perturb_after

    from sctwin.backtest import BacktestSpec, check_split
    from sctwin.features import FeatureBuilder
    from sctwin.panel import build_panel
    from sctwin.timeunit import resample

    sub = pl.read_parquet("data/subsets/dev20.parquet").drop(["role", "holdout"])
    p = build_panel(sub)
    assert p.shape == (20, 90)
    upd = 24 if unit == "hour" else 1
    lead, horizon = (1, 7) if unit == "day" else (1, 24)
    for day in (30, 60, 80):
        origin = day * upd + (upd - 3 if unit == "hour" else 0)  # hour: 21:00, mid-evening
        q = perturb_after(p, unit, origin, seed=day)
        fa = FeatureBuilder(resample(p, unit), p.keys)
        fq = FeatureBuilder(resample(q, unit), q.keys)
        cols = list(fa.feature_names)
        origins = np.arange(max(0, origin - 3 * upd), origin + 1)
        a = fa.rows(origins, lead, horizon).select(cols).to_numpy()
        b = fq.rows(origins, lead, horizon).select(cols).to_numpy()
        np.testing.assert_array_equal(a, b)
        spec = BacktestSpec(unit, lead, horizon, (origin,), window="expanding")
        ta = fa.rows(spec.train_times(origin), lead, horizon)
        tb = fq.rows(spec.train_times(origin), lead, horizon)
        check_split(ta, fa.rows([origin], lead, horizon), origin, spec)
        np.testing.assert_array_equal(
            ta.select([*cols, "y"]).to_numpy(), tb.select([*cols, "y"]).to_numpy()
        )
