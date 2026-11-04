"""Unit tests for experiments/referee_forecast_gw.py (Giacomini and White 2006 tests).
SYNTHETIC data only (no real files are read)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.evaluation import diebold_mariano

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "referee_forecast_gw.py"
_spec = importlib.util.spec_from_file_location("referee_forecast_gw", _PATH)
gw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gw)


def _calendar(n, rng):
    """An expiry dummy (about one day in five) and three consecutive regimes, as in the real sample."""
    e = (rng.random(n) < 0.21).astype(float)
    reg = np.repeat([0, 1, 2], [n // 3, n // 4, n - n // 3 - n // 4])
    return e, reg


def _H(d, e, reg):
    day = pd.DataFrame({"is_expiry": e.astype(bool), "regime": reg})
    return gw.instruments(day, d)


def test_newey_west_lag0_is_the_sample_covariance_and_symmetric():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 3))
    S0 = gw.newey_west(Z, lag=0)
    Zc = Z - Z.mean(axis=0)
    assert np.allclose(S0, Zc.T @ Zc / 400)
    S = gw.newey_west(Z, lag=6)
    assert np.allclose(S, S.T) and np.all(np.linalg.eigvalsh(S) > 0)


def test_unconditional_test_is_the_squared_dm_statistic():
    rng = np.random.default_rng(1)
    l1, l2 = rng.gamma(2, 1, 500), rng.gamma(2, 1, 500) * 0.97
    d = l1 - l2
    res = gw.gw_test(d, np.ones(len(d)), "nw")
    dm = diebold_mariano(l1, l2)
    n = len(d)
    # the pipeline's DM statistic carries the Harvey-Leybourne-Newbold factor sqrt((n - 1) / n) for h = 1
    assert res["stat"] == np.float64(res["stat"])
    assert abs(res["stat"] - dm["stat"] ** 2 * n / (n - 1)) < 1e-8 * max(1.0, res["stat"])


def test_instruments_use_only_information_from_before_day_t():
    rng = np.random.default_rng(2)
    n = 60
    d = rng.normal(size=n)
    e, reg = _calendar(n, rng)
    H, names, dd = _H(d, e, reg)
    assert names == ["const", "d_lag", "expiry", "regime_single_weekly_thu", "regime_single_weekly_tue"]
    assert np.allclose(dd, d[1:]) and np.allclose(H[:, 1], d[:-1])       # row t holds d_{t-1}
    assert np.allclose(H[:, 2], e[1:]) and np.allclose(H[:, 0], 1.0)      # and day t's calendar


def test_size_under_equal_predictive_ability():
    # H0 of Giacomini and White: E[d_t | F_{t-1}] = 0. Heavy-tailed iid differentials,
    # the real-sample size and the real instruments; every test should reject about 5% of the time.
    rng = np.random.default_rng(3)
    n, reps = 790, 1000
    rej = {k: 0 for k in ("unc", "c1", "c2", "c2_nw", "slope_all", "slope_exp", "slope_reg")}
    for _ in range(reps):
        d = rng.standard_t(5, size=n)
        e, reg = _calendar(n, rng)
        H, names, dd = _H(d, e, reg)
        rej["unc"] += gw.gw_test(dd, H[:, :1], "nw")["p"] < 0.05
        rej["c1"] += gw.gw_test(dd, H[:, :2], "gw")["p"] < 0.05
        rej["c2"] += gw.gw_test(dd, H, "gw")["p"] < 0.05
        rej["c2_nw"] += gw.gw_test(dd, H, "nw")["p"] < 0.05
        rej["slope_all"] += gw.slope_wald(dd, H, [1, 2, 3, 4])["p"] < 0.05
        rej["slope_exp"] += gw.slope_wald(dd, H, [2])["p"] < 0.05
        rej["slope_reg"] += gw.slope_wald(dd, H, [3, 4])["p"] < 0.05
    for k, v in rej.items():
        assert 0.03 <= v / reps <= 0.08, (k, v / reps)


def test_slope_tests_keep_their_size_when_the_advantage_is_constant():
    # one model is better by a constant amount and the differential is autocorrelated:
    # the conditional tests reject (the mean is not zero) but the slope tests should not
    rng = np.random.default_rng(4)
    n, reps = 790, 600
    rej_slope, rej_c2 = 0, 0
    for _ in range(reps):
        u = np.empty(n)
        u[0] = rng.normal()
        eps = rng.normal(size=n)
        for t in range(1, n):
            u[t] = 0.3 * u[t - 1] + eps[t]
        d = 0.5 + u
        e, reg = _calendar(n, rng)
        H, names, dd = _H(d, e, reg)
        rej_slope += gw.slope_wald(dd, H, [2, 3, 4])["p"] < 0.05
        rej_c2 += gw.gw_test(dd, H, "gw")["p"] < 0.05
    assert 0.025 <= rej_slope / reps <= 0.09
    assert rej_c2 / reps > 0.99


def test_conditional_test_detects_an_advantage_that_only_exists_on_expiry_days():
    # zero average advantage, but the hybrid is better on expiry days and worse on the others:
    # the unconditional test cannot see it, the conditional test and the expiry slope can
    rng = np.random.default_rng(5)
    n = 790
    e, reg = _calendar(n, rng)
    noise = rng.normal(size=n)
    d = 1.5 * (e - e[1:].mean()) + noise - noise[1:].mean()
    H, names, dd = _H(d, e, reg)
    assert abs(dd.mean()) < 1e-12
    assert gw.gw_test(dd, H[:, :1], "nw")["p"] > 0.99
    assert gw.gw_test(dd, H, "gw")["p"] < 1e-6
    s = gw.slope_wald(dd, H, [2])
    assert s["p"] < 1e-6 and abs(s["coef"][2] - 1.5) < 0.4


def test_end_to_end_on_synthetic_files(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    days = pd.bdate_range("2023-08-10", periods=300)
    n = len(days)
    e, reg = _calendar(n, rng)
    vol = np.exp(np.cumsum(rng.normal(0, 0.05, size=n)) * 0.3) * 1e-5
    rows = []
    for k in range(2):     # two anchors per day
        y = vol * rng.chisquare(4, size=n) / 4
        rows.append(pd.DataFrame({"t": np.arange(n) + 150, "date": days, "minute": 15 * k, "horizon": 15,
                                  "is_expiry": e.astype(bool), "tdte": 0.0, "regime": reg, "y": y,
                                  "mcsgarch": vol * np.exp(rng.normal(0, 0.45, size=n)),
                                  "mcsgarch_iv": vol * np.exp(rng.normal(0, 0.40, size=n)),
                                  "hybrid_base": vol * np.exp(rng.normal(0, 0.30, size=n))}))
    f = pd.concat(rows).sort_values(["t", "minute"]).reset_index(drop=True)
    run, out = tmp_path / "run", tmp_path / "out"
    run.mkdir()
    out.mkdir()
    for target in gw.TARGETS:
        f.to_csv(run / f"forecasts_{target}.csv", index=False)
        x = f[["t", "date", "minute"]].copy()
        x["mcsgarch_vix"] = vol[f["t"].to_numpy() - 150] * np.exp(rng.normal(0, 0.40, size=len(f)))
        x.to_csv(out / f"mcsgarchx_forecasts_{target}.csv", index=False)
    pd.DataFrame({"target": gw.TARGETS, "subset": "all", "benchmark": "best_matched_mcsgarch",
                  "benchmark_model": ["mcsgarch_iv", "mcsgarch_iv", "mcsgarch_vix"]}).to_csv(
        out / "fair_benchmarks.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["x", "--run", str(run), "--out", str(out)])
    gw.main()
    res = pd.read_csv(out / "gw_tests.csv")
    assert len(res) == 54 and (res.family == "main").sum() == 42
    assert res["p"].between(0, 1).all()
    m = res[res.family == "main"]
    assert (m["p_holm_main_family"] >= m["p"] - 1e-12).all()
    assert (res["n"] == n - 1).all()
    gm = pd.read_csv(out / "gw_group_margins.csv")
    assert {"expiry", "non_expiry", "multi_weekly"} <= set(gm["group"])
    assert (out / "gw_regression_coefficients.csv").exists()
