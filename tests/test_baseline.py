"""Baselines: known-answer checks, determinism, and end-to-end leakage safety (both units)."""

import numpy as np
import pytest
from test_no_leakage import builder, perturb_after, random_panel

from sctwin.backtest import BacktestSpec
from sctwin.baseline import TAUS, GLMDesign, tweedie_quantiles, tweedie_sample
from sctwin.baseline_runner import add_groups, run_backtest
from sctwin.evaluation import evaluate, qcol

CFG = {
    "taus": list(TAUS),
    "seed": 1,
    "glm": {"power": 1.5, "alpha": 1e-3, "n_samples": 100},
    "lgbm": {"num_rounds": 10, "num_threads": 2, "params": {"min_data_in_leaf": 5}},
}
SPECS = {
    "day": dict(lead=1, horizon=7, origins=(44, 51), window="sliding", window_length=20,
                min_history=7),
    "hour": dict(lead=1, horizon=24, origins=(20 * 24 + 22, 21 * 24 + 22), window="sliding",
                 window_length=72, min_history=48, train_stride=3),
}  # fmt: skip
QUIET = {"log": lambda *_: None}


def run(p, unit, origins=None):
    kw = dict(SPECS[unit])
    if origins is not None:
        kw["origins"] = origins
    spec = BacktestSpec(unit, **kw)
    fb = builder(p, unit)
    s = np.arange(p.shape[0])
    pred, log = run_backtest(fb, spec, s, s, CFG, **QUIET)
    return fb, spec, pred, log


def test_tweedie_sample_moments():
    rng = np.random.default_rng(0)
    mu, phi, power = np.array([0.3, 2.0]), 1.2, 1.5
    s = tweedie_sample(mu, phi, power, 400_000, rng)
    np.testing.assert_allclose(s.mean(1), mu, rtol=0.02)
    np.testing.assert_allclose(s.var(1), phi * mu**power, rtol=0.03)
    assert (s >= 0).all() and (s[0] == 0).mean() > 0.2  # point mass at zero


def test_tweedie_quantiles_ordered():
    q = tweedie_quantiles(np.array([0.1, 1.0, 5.0]), 1.0, 1.5, TAUS, n=500)
    assert (np.diff(q, axis=1) >= 0).all()


def test_glm_design_fitted_on_train_only():
    import polars as pl

    tr = pl.DataFrame({"a": [1.0, 2.0, 3.0], "c": [0, 1, 1]})
    te = pl.DataFrame({"a": [100.0, float("nan")], "c": [1, 0]})
    d = GLMDesign(numeric=["a"], categorical=["c"], log_cols=[]).fit(tr)
    x = d.transform(te)
    assert d.means["a"] == 2.0  # test values never enter the statistics
    np.testing.assert_allclose(x[1], [0.0, 1.0, 0.0])  # imputed with train mean -> 0, flag 1


@pytest.fixture(scope="module", params=["day", "hour"])
def result(request):
    unit = request.param
    n_days = 60 if unit == "day" else 25
    p = random_panel(4, n_days, 7)
    return (unit, p, *run(p, unit))


def test_runner_output_and_protocol(result):
    unit, p, fb, spec, pred, log = result
    assert pred.height == len(spec.origins) * 4 * spec.horizon
    assert set(pred["origin_idx"].unique().to_list()) == set(spec.origins)
    lag = pred["target_idx"] - pred["origin_idx"]
    assert lag.min() == spec.lead + 1 and lag.max() == spec.lead + spec.horizon
    for m in ("clim", "glm", "lgbm_q"):
        q = pred.select([qcol(m, t) for t in TAUS]).to_numpy()
        ok = ~np.isnan(q).any(axis=1)
        assert (np.diff(q[ok], axis=1) >= -1e-12).all(), m
    for r in log:
        assert r["train_rows"] > 0
        assert r["train_t_range"][1] <= r["origin"] - spec.lead - spec.horizon
    res = evaluate(add_groups(pred, fb, {0}), TAUS)
    assert {"snaive", "clim", "glm", "lgbm", "lgbm_q"} <= set(res["overall"]["models"])
    assert set(res["folds"]) == {str(o) for o in spec.origins}


def test_runner_is_deterministic(result):
    unit, p, _, _, pred, _ = result
    assert pred.equals(run(p, unit)[2])


def test_predictions_ignore_data_after_origin(result):
    """End-to-end: replacing all data after the fold origin T leaves every model's prediction
    unchanged (features, training set, fitted statistics and models all as of T)."""
    unit, p, _, spec, _, _ = result
    o = spec.origins[0]
    a = run(p, unit, origins=(o,))[2]
    b = run(perturb_after(p, unit, o, seed=3), unit, origins=(o,))[2]
    cols = [c for c in a.columns if "__" in c]
    np.testing.assert_array_equal(a.select(cols).to_numpy(), b.select(cols).to_numpy())
    assert not a["y"].equals(b["y"])  # the outcomes did change: the check is not vacuous
