"""Fast unit tests (SYNTHETIC arrays) for the helpers in experiments/backtest_extended.py:
the loss functions, the stationary bootstrap, Hansen's SPA test, the HAC regression and the
run-length helper. Written by an AI agent (Agent A of the backtest round)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "backtest_extended", Path(__file__).resolve().parents[1] / "experiments" / "backtest_extended.py")
bx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bx)


def test_losses_zero_at_perfect_forecast_and_positive_otherwise():
    y = np.array([1e-6, 2e-6, 5e-7, 3e-6])
    for name, fn in bx.LOSSES.items():
        assert np.allclose(fn(y, y), 0.0, atol=1e-30), name
        assert (fn(y, 1.7 * y) > 0).all(), name
        assert (fn(y, 0.4 * y) > 0).all(), name


def test_patton_family_special_cases():
    y = np.array([1e-6, 2e-6, 5e-7])
    f = np.array([1.5e-6, 1e-6, 6e-7])
    assert np.allclose(bx.patton(y, f, -2), bx.qlike(y, f))
    assert np.allclose(bx.patton(y, f, 0), bx.mse_var(y, f) / 2)
    # b = -1 by hand: f - y + y log(y / f)
    assert np.allclose(bx.patton(y, f, -1), f - y + y * np.log(y / f))
    # b = 1 by hand: (y^3 - f^3) / 6 - f^2 (y - f) / 2
    assert np.allclose(bx.patton(y, f, 1), (y ** 3 - f ** 3) / 6 - f ** 2 * (y - f) / 2)


def test_qlike_scale_invariance_and_vol_scale():
    y = np.array([1e-6, 2e-6])
    f = np.array([1.5e-6, 1e-6])
    assert np.allclose(bx.qlike(y, f), bx.qlike(1e6 * y, 1e6 * f))
    assert np.allclose(bx.qlike_vol(y, f), bx.qlike(np.sqrt(y), np.sqrt(f)))
    assert np.allclose(bx.mse_log(y, f), np.log(y / f) ** 2)


def test_gain_pct():
    assert bx.gain_pct(np.array([1.0, 1.0]), np.array([0.9, 0.9])) == pytest.approx(10.0)
    assert bx.gain_pct(np.array([1.0]), np.array([1.2])) == pytest.approx(-20.0)


def test_stationary_bootstrap_indices_shape_and_blocks():
    rng = np.random.default_rng(0)
    n, draws = 50, 20
    idx = bx.stationary_bootstrap_indices(n, draws, 5.0, rng)
    assert idx.shape == (draws, n)
    assert idx.min() >= 0 and idx.max() < n
    # a very long mean block gives one circular block per draw: consecutive indices mod n
    idx1 = bx.stationary_bootstrap_indices(n, draws, 1e12, np.random.default_rng(1))
    assert ((np.diff(idx1, axis=1) % n) == 1).all()
    # mean block 1 is an i.i.d. resample: not all steps are +1
    idx0 = bx.stationary_bootstrap_indices(n, draws, 1.0, np.random.default_rng(2))
    assert ((np.diff(idx0, axis=1) % n) == 1).mean() < 0.2


def test_sb_variance_matches_sample_variance_for_block_one():
    rng = np.random.default_rng(3)
    d = rng.standard_normal((300, 2))
    v = bx.sb_variance(d, 1.0)
    assert np.allclose(v, d.var(axis=0, ddof=0))


def test_spa_rejects_clear_improvement_and_accepts_noise():
    rng = np.random.default_rng(4)
    n = 200
    bench = 1.0 + 0.1 * rng.standard_normal(n)
    noise = bench[:, None] + 0.05 * rng.standard_normal((n, 3))      # same expected loss
    res0 = bx.spa_test(bench, noise, draws=300, mean_block=5.0, seed=11)
    assert res0["n_models"] == 3 and len(res0["tstat"]) == 3
    assert res0["p_consistent"] > 0.05
    better = noise.copy()
    better[:, 1] -= 0.2                                               # one model clearly better
    res1 = bx.spa_test(bench, better, draws=300, mean_block=5.0, seed=11)
    assert res1["best_model_index"] == 1
    assert res1["p_consistent"] < 0.02 and res1["p_lower"] <= res1["p_consistent"] <= res1["p_upper"] + 1e-12
    # seed makes it reproducible
    res2 = bx.spa_test(bench, better, draws=300, mean_block=5.0, seed=11)
    assert res2["p_consistent"] == res1["p_consistent"]


def test_hac_ols_recovers_exact_line_and_wald():
    x = np.linspace(0, 1, 60)
    y = 2.0 + 3.0 * x
    day = np.repeat(np.arange(12), 5)
    r = bx.hac_ols(np.column_stack([np.ones(60), x]), y, day)
    assert np.allclose(r["coef"], [2.0, 3.0])
    assert r["n_days"] == 12
    # with noise the Wald test of the true values should not reject strongly
    rng = np.random.default_rng(5)
    r2 = bx.hac_ols(np.column_stack([np.ones(60), x]), y + 0.1 * rng.standard_normal(60), day)
    stat, p = bx.wald(r2["coef"], r2["V"], np.eye(2), np.array([2.0, 3.0]))
    assert stat >= 0 and 0 <= p <= 1


def test_longest_run():
    assert bx.longest_run(np.array([False, True, True, False, True, True, True, False])) == (3, 4, 6)
    assert bx.longest_run(np.array([False, False])) == (0, -1, -1)
    assert bx.longest_run(np.array([True])) == (1, 0, 0)
