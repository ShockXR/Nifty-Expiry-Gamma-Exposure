"""Unit tests for experiments/referee_forecast.py. SYNTHETIC data only (no real files are read).

Covers the stationary bootstrap, Hansen's SPA test, White's Reality Check, the max-t
adjusted p-values, the DM test with extra estimator noise, the AR(1)-EGARCH(1,1)-X
used for mcsGARCH-X (against the arch package and on simulated data), and a small
end-to-end run of the SPA and hold-out sections on a synthetic forecast file.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2
from scipy.stats import t as student_t

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "referee_forecast.py"
_spec = importlib.util.spec_from_file_location("referee_forecast", _PATH)
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)


# ---------------------------------------------------------------- stationary bootstrap


def test_stationary_bootstrap_block_length_and_coverage():
    rng = np.random.default_rng(1)
    n = 500
    idx = rf.stationary_bootstrap_indices(n, 400, 10.0, rng)
    assert idx.shape == (400, n) and idx.min() >= 0 and idx.max() < n
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % n).mean()
    # a new block starts with probability 0.1 (and lands on the next index by chance with probability 1/n)
    assert abs((1 - cont) - 0.1 * (1 - 1 / n)) < 0.01
    freq = np.bincount(idx.ravel(), minlength=n) / idx.size
    assert freq.min() > 0.5 / n and freq.max() < 1.5 / n


def test_bootstrap_means_equal_direct_indexing():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 3))
    idx = rf.stationary_bootstrap_indices(50, 20, 4.0, rng)
    np.testing.assert_allclose(rf.bootstrap_means(x, idx), x[idx].mean(axis=1), rtol=1e-12)


def test_kernel_variance_iid_and_ar1():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    # with mean block 1 every lag weight is zero: the plain variance (divisor n)
    assert rf.sb_kernel_variance(x, 1.0)[0] == pytest.approx(np.var(x))
    # AR(1), phi 0.5, unit innovations: long-run variance 1 / (1 - 0.5)^2 = 4
    n = 20000
    e = rng.normal(size=n)
    y = np.empty(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + e[t]
    assert rf.sb_kernel_variance(y, 50.0)[0] == pytest.approx(4.0, rel=0.15)


def test_kernel_variance_matches_arch():
    from arch.bootstrap import SPA

    rng = np.random.default_rng(4)
    bench = rng.normal(size=300) ** 2
    models = bench[:, None] + rng.normal(0, 0.3, size=(300, 4))
    spa = SPA(bench, models, block_size=10, reps=10, seed=1)
    spa._compute_variance()
    np.testing.assert_allclose(rf.sb_kernel_variance(bench[:, None] - models, 10.0), spa._loss_diff_var, rtol=1e-10)


# ---------------------------------------------------------------- SPA and Reality Check


def test_spa_rejects_when_one_model_is_better():
    rng = np.random.default_rng(5)
    d = rng.normal(size=(500, 10))
    d[:, 3] += 0.25
    r = rf.spa_test(d, reps=2000, mean_block=5, seed=1)
    assert r["best"] == 3
    assert r["p_spa_c"] < 0.01 and r["p_rc"] < 0.01


def test_spa_size_under_null():
    rng = np.random.default_rng(6)
    rej_c, rej_rc = [], []
    for k in range(200):
        d = rng.normal(size=(200, 5))
        r = rf.spa_test(d, reps=299, mean_block=2, seed=k)
        rej_c.append(r["p_spa_c"] <= 0.10)
        rej_rc.append(r["p_rc"] <= 0.10)
    assert 0.02 <= np.mean(rej_c) <= 0.20
    assert 0.02 <= np.mean(rej_rc) <= 0.20


def test_spa_is_not_fooled_by_poor_noisy_models_but_rc_is():
    # Hansen's (2005) point: many poor, noisy models destroy the Reality Check's power, not the SPA test's.
    rng = np.random.default_rng(7)
    n = 500
    good = rng.normal(0.3, 1.0, size=(n, 1))
    poor = rng.normal(-1.0, 5.0, size=(n, 30))
    r = rf.spa_test(np.hstack([good, poor]), reps=3000, mean_block=2, seed=2)
    assert r["n_excluded_consistent"] == 30
    assert r["t"][0] > 4
    assert r["p_spa_c"] < 0.01
    assert r["p_rc"] > 0.5


def test_spa_pvalue_ordering():
    rng = np.random.default_rng(8)
    for k in range(5):
        d = rng.normal(size=(150, 6)) + rng.normal(0, 0.1, size=6)
        d[:, 0] -= 0.8
        r = rf.spa_test(d, reps=500, mean_block=3, seed=k)
        assert r["p_spa_l"] <= r["p_spa_c"] <= r["p_spa_u"]


def test_reality_check_matches_arch_upper_pvalue():
    # arch's SPA statistic is not studentized, and its "upper" p-value (every model recentred)
    # is White's Reality Check p-value. Different bootstrap draws, so compare within Monte Carlo error.
    from arch.bootstrap import SPA

    rng = np.random.default_rng(9)
    n = 300
    bench = rng.normal(1.0, 1.0, size=n)
    models = bench[:, None] + rng.normal(0, 1.0, size=(n, 5))
    models[:, 2] -= 0.08
    spa = SPA(bench, models, block_size=5, reps=6000, seed=11)
    spa.compute()
    r = rf.spa_test(bench[:, None] - models, reps=6000, mean_block=5, seed=12)
    assert 0.02 < r["p_rc"] < 0.6
    assert r["p_rc"] == pytest.approx(float(spa.pvalues["upper"]), abs=0.04)


def test_stepdown_adjusted_pvalues():
    rng = np.random.default_rng(10)
    d = rng.normal(size=(400, 8))
    d[:, 5] += 0.3
    r = rf.spa_test(d, reps=2000, mean_block=2, seed=3)
    assert r["p_stepdown"][5] < 0.01
    assert np.all(r["p_stepdown"] <= r["p_single"] + 1e-12)
    order = np.argsort(-r["t"])
    assert np.all(np.diff(r["p_stepdown"][order]) >= -1e-12)
    # adjusted p-values are never below the plain bootstrap p-value of the same model
    raw = np.array([(np.sqrt(400) * (rf.bootstrap_means(d, rf.stationary_bootstrap_indices(
        400, 2000, 2, np.random.default_rng(3)))[:, j] - d[:, j].mean()) /
        np.sqrt(rf.sb_kernel_variance(d[:, j], 2)[0]) >= r["t"][j]).mean() for j in range(8)])
    assert np.all(r["p_single"] >= raw - 1e-12)


# ---------------------------------------------------------------- DM with extra noise


def test_dm_with_extra_noise_and_breakeven():
    from niftygex.evaluation import diebold_mariano

    rng = np.random.default_rng(11)
    l1 = rng.gamma(2.0, 1.0, size=300)
    l2 = l1 - 0.1 + rng.normal(0, 0.5, size=300)
    dm = diebold_mariano(l1, l2, alternative="greater")
    se = dm["mean_diff"] / dm["stat"]
    stat, p = rf.dm_with_extra_noise(dm["mean_diff"], se, dm["n"], 0.0)
    assert stat == pytest.approx(dm["stat"]) and p == pytest.approx(dm["p"])
    sd = rf.breakeven_noise_sd(dm["mean_diff"], se, dm["n"], 0.05)
    assert sd > 0
    assert rf.dm_with_extra_noise(dm["mean_diff"], se, dm["n"], sd)[1] == pytest.approx(0.05, abs=1e-9)
    # more noise, larger p
    assert rf.dm_with_extra_noise(dm["mean_diff"], se, dm["n"], 2 * sd)[1] > 0.05
    # a margin that is already insignificant needs no extra noise
    assert rf.breakeven_noise_sd(0.01, 1.0, 300, 0.05) == 0.0


def test_sd_upper_bound_hand_worked():
    # chi2(2) 5% quantile 0.102587: sqrt(2 / 0.102587) = 4.4154
    assert rf.sd_upper_bound(1.0, 2) == pytest.approx(np.sqrt(2 / chi2.ppf(0.05, 2)))
    assert rf.sd_upper_bound(1.0, 2) == pytest.approx(4.4154, abs=1e-3)


# ---------------------------------------------------------------- EGARCH-X


def _simulate(n, mu, phi, omega, alpha, gamma, beta, nu, delta=0.0, x=None, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2) / nu)
    x = np.zeros(n) if x is None else x
    y = np.zeros(n)
    lns = np.log(1.0)
    zp = 0.0
    for t in range(1, n):
        lns = omega + alpha * (abs(zp) - np.sqrt(2 / np.pi)) + gamma * zp + beta * lns + delta * x[t]
        y[t] = mu + phi * y[t - 1] + np.exp(0.5 * lns) * z[t]
        zp = z[t]
    return y


def test_egarchx_filter_reproduces_arch_without_exogenous_terms():
    from arch import arch_model

    y = _simulate(800, 0.03, 0.05, 0.0, 0.15, -0.08, 0.9, 7.0, seed=1)
    am = arch_model(y, mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    ref = np.asarray(am.fix(res.params.values).conditional_volatility, float) ** 2
    mine, _, _ = rf.egarchx_filter(res.params.values, y, None)
    np.testing.assert_allclose(mine, ref[1:], rtol=1e-8)
    # and the likelihood matches arch's at the same parameters
    assert rf.egarchx_loglik(res.params.values, y, None) == pytest.approx(res.loglikelihood, abs=0.5)


def test_egarchx_fit_matches_arch_and_ignores_noise_regressor():
    from arch import arch_model

    y = _simulate(900, 0.02, 0.03, 0.0, 0.12, -0.06, 0.92, 8.0, seed=2)
    res = arch_model(y, mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False).fit(disp="off")
    p0 = res.params.values
    own = rf.fit_egarchx(y, None, [p0])
    assert own["loglik"] == pytest.approx(res.loglikelihood, abs=1.0)
    noise = np.random.default_rng(3).normal(size=(900, 1))
    h, info = rf.egarchx_daily_variance(y / 100.0, noise, 900, p0)
    assert info["ok"] and h is not None
    assert abs(info["delta_1"]) < 0.15
    assert info["loglik"] >= own["loglik"] - 1e-6


def test_egarchx_recovers_an_exogenous_effect():
    rng = np.random.default_rng(4)
    n = 1500
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.97 * x[t - 1] + 0.25 * rng.normal()
    y = _simulate(n, 0.0, 0.0, 0.0, 0.1, -0.05, 0.5, 8.0, delta=0.5, x=x, seed=5)
    p0 = np.array([0.0, 0.0, 0.0, 0.1, -0.05, 0.9, 8.0])
    base = rf.fit_egarchx(y, None, [p0])
    h, info = rf.egarchx_daily_variance(y / 100.0, x[:, None], n, p0)
    assert info["ok"]
    assert info["delta_1"] == pytest.approx(0.5, abs=0.15)
    assert 2 * (info["loglik"] - base["loglik"]) > 20


def test_egarchx_variance_has_no_lookahead():
    rng = np.random.default_rng(6)
    n = 400
    y = _simulate(n, 0.0, 0.05, 0.0, 0.1, -0.05, 0.9, 8.0, seed=7) / 100.0
    X = rng.normal(size=(n, 1))
    p0 = np.array([0.0, 0.05, 0.0, 0.1, -0.05, 0.9, 8.0])
    h1, i1 = rf.egarchx_daily_variance(y, X, 300, p0)
    y2, X2 = y.copy(), X.copy()
    y2[350:] += rng.normal(0, 0.005, size=n - 350)  # returns of day 350 onwards
    X2[351:] += 1.0                                 # regressor known before day 351 opens, onwards
    h2, i2 = rf.egarchx_daily_variance(y2, X2, 300, p0)
    assert i1["ok"] and i2["ok"]
    np.testing.assert_allclose(h1[:351], h2[:351], rtol=1e-12)
    assert not np.allclose(h1[351:], h2[351:])


def test_realtime_guard_switches_only_from_the_first_bad_test_day():
    y_train = np.full(100, 0.01) * np.where(np.arange(100) % 2, 1, -1)   # daily variance 1e-4
    h = np.full(200, 1e-4)
    fb = np.full(200, 2e-4)
    out, j = rf.realtime_guard(h, fb, y_train, 120, 141)
    assert j is None and out is h
    h2 = h.copy()
    h2[130] = 1e-4 * np.exp(6)          # 6 log points away from the training variance
    h2[180] = 1e-4 * np.exp(9)          # a later day outside the block must not matter
    out, j = rf.realtime_guard(h2, fb, y_train, 120, 141)
    assert j == 130
    np.testing.assert_allclose(out[:130], h2[:130])
    np.testing.assert_allclose(out[130:], fb[130:])
    out, j = rf.realtime_guard(h2, fb, y_train, 141, 162)
    assert j is None


def test_lagged_log_daily_variance_uses_the_previous_close():
    level = np.array([0.10, 0.20, 0.15, 0.30])
    x = rf.lagged_log_daily_variance(level)
    np.testing.assert_allclose(x[1:], np.log(level[:-1] ** 2 / 252 * 1e4))
    assert x[0] == x[1]


# ---------------------------------------------------------------- end to end on a synthetic forecast file


def _synthetic_run(tmp_path: Path) -> Path:
    rng = np.random.default_rng(12)
    days = pd.bdate_range("2024-06-03", "2026-10-30")
    n = len(days)
    vol = np.exp(np.cumsum(rng.normal(0, 0.05, size=n)) * 0.3) * 1e-5
    y = vol * rng.chisquare(4, size=n) / 4
    df = pd.DataFrame({"t": np.arange(n), "date": days, "minute": 315, "horizon": 60,
                       "is_expiry": (days.weekday == 1), "tdte": 0.0, "regime": 2, "y": y})
    df["mcsgarch"] = vol * np.exp(rng.normal(0, 0.45, size=n))
    df["hybrid_base"] = vol * np.exp(rng.normal(0, 0.30, size=n))
    df["lgbm_base"] = vol * np.exp(rng.normal(0, 0.35, size=n))
    df["lgbmt_base"] = vol * np.exp(rng.normal(0, 0.35, size=n))
    df["linear_base"] = vol * np.exp(rng.normal(0, 0.40, size=n))
    df["har"] = vol * np.exp(rng.normal(0.2, 0.6, size=n))
    run = tmp_path / "run"
    run.mkdir()
    for target in rf.TARGETS:
        df.to_csv(run / f"forecasts_{target}.csv", index=False)
    json.dump({"models": ["mcsgarch", "hybrid_base", "lgbm_base", "lgbmt_base", "linear_base", "har"]},
              open(run / "run_info.json", "w"))
    return run


def test_spa_and_holdout_sections_on_synthetic_forecasts(tmp_path):
    run = _synthetic_run(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    models = json.load(open(run / "run_info.json"))["models"]
    spa, pm = rf.section_spa(run, out, models, reps=300, blocks=[5.0], main_block=5.0)
    assert set(spa["subset"]) == set(rf.SUBSETS) and len(spa) == 9
    assert spa["p_spa_consistent"].between(0, 1).all()
    # the synthetic hybrid has the smallest forecast error, so it is the best model on all days
    assert (spa[spa.subset == "all"]["best_by_mean"] == "hybrid_base").all()
    assert (pm["p_maxt_stepdown"] <= pm["p_maxt_single_step"] + 1e-12).all()
    t1, t2 = rf.section_holdout(run, out)
    assert {"development", "holdout", "holdout_pre_auction"} <= set(t1["period"])
    assert t2["p_two_sided"].between(0, 1).all()
    assert (out / "holdout_table.csv").exists() and (out / "spa_reality_check.csv").exists()
