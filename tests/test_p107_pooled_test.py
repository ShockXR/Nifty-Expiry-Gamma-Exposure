"""Unit tests for experiments/p107_pooled_test.py. SYNTHETIC data only (no real file is read).

Covers the Driscoll-Kraay covariance (a hand calculation, the identity with Newey-West on the cross-sectional mean,
the identity with the project's DM test for one index, agreement with statsmodels' ``hac-groupsum`` for balanced and
unbalanced panels, and its size under strong cross-sectional dependence), the Wald test of equal means, the effective
numbers of indices, the joint stationary bootstrap, and a small end-to-end run on synthetic forecast files.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from niftygex.evaluation import diebold_mariano

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p107_pooled_test.py"
_spec = importlib.util.spec_from_file_location("p107_pooled_test", _PATH)
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def nw_variance_of_mean(x, lag):
    """Explicit-loop Newey-West variance of the sample mean (hand calculation)."""
    x = np.asarray(x, float)
    n = x.size
    c = x - x.mean()
    s = sum(c[t] * c[t] for t in range(n))
    for j in range(1, lag + 1):
        w = 1 - j / (lag + 1)
        s += 2 * w * sum(c[t] * c[t - j] for t in range(j, n))
    return s / n / n


def factor_panel(T, N, rho, phi, rng, mu=0.0):
    """Equicorrelated (corr rho) AR(1) (coefficient phi) panel with mean mu: SYNTHETIC."""
    e = rng.standard_normal((T + 50, N))
    f = rng.standard_normal(T + 50)
    z = np.sqrt(rho) * f[:, None] + np.sqrt(1 - rho) * e
    x = np.zeros_like(z)
    for t in range(1, T + 50):
        x[t] = phi * x[t - 1] + z[t]
    return mu + x[50:]


# ---------------------------------------------------------------- bandwidth and kernel


def test_bandwidth_rule_matches_project_dm_rule():
    assert pt.dk_lag(604) == 5
    assert pt.dk_lag(252) == 4
    for n in (50, 82, 130, 300, 604, 792, 1000):
        assert pt.dk_lag(n) == int(np.floor(4 * (n / 100) ** (2 / 9)))
    assert np.allclose(pt.bartlett_weights(3), [1, 0.75, 0.5, 0.25])


def test_hand_calculation_tiny_panel():
    # D (3 dates x 2 indices); means (2, 2); residuals U = [[-1, 0], [1, -2], [0, 2]]; lag 1, w1 = 0.5.
    # U'U = [[2, -2], [-2, 8]]; G = sum_t u_t u_{t-1}' = [[-1, 0], [4, -4]]; S = U'U + 0.5 (G + G') = [[1, 0], [0, 4]].
    D = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 4.0]])
    r = pt.index_means_dk(D, lag=1)
    assert np.allclose(r["means"], [2, 2])
    assert np.allclose(r["S"], [[1, 0], [0, 4]])
    assert np.allclose(r["cov"], [[1 / 9, 0], [0, 4 / 9]])
    c = pt.combo_test(r["means"], r["cov"], [0.5, 0.5], r["T"], hln=False)
    assert np.isclose(c["se"] ** 2, 5 / 36)


def test_balanced_pooled_mean_equals_newey_west_of_cross_sectional_mean():
    rng = np.random.default_rng(3)
    D = factor_panel(120, 3, 0.6, 0.3, rng, mu=0.1)
    lag = pt.dk_lag(120)
    r = pt.index_means_dk(D, lag=lag)
    a = np.full(3, 1 / 3)
    v_dk = a @ r["cov"] @ a
    assert np.isclose(v_dk, nw_variance_of_mean(D.mean(axis=1), lag), rtol=1e-12)
    # the same as pooled OLS on a constant with the generic Driscoll-Kraay function
    T, N = D.shape
    tt = np.repeat(np.arange(T), N)
    o = pt.panel_ols_dk(D.ravel(), np.ones(T * N), tt, lag=lag)
    assert np.isclose(o["beta"][0], D.mean())
    assert np.isclose(o["cov"][0, 0], v_dk, rtol=1e-12)


def test_one_index_equals_project_dm_test():
    rng = np.random.default_rng(4)
    l1 = rng.gamma(2.0, 0.1, 300)
    l2 = l1 - 0.01 + 0.05 * rng.standard_normal(300)
    dm = diebold_mariano(l1, l2, alternative="greater")
    r = pt.index_means_dk((l1 - l2)[:, None])
    c = pt.combo_test(r["means"], r["cov"], [1.0], r["T"])
    assert np.isclose(c["stat"], dm["stat"], rtol=1e-12)
    assert np.isclose(c["p"], dm["p"], rtol=1e-10)
    two = pt.combo_test(r["means"], r["cov"], [1.0], r["T"], alternative="two-sided")
    assert np.isclose(two["p"], diebold_mariano(l1, l2)["p"], rtol=1e-10)


# ---------------------------------------------------------------- against statsmodels hac-groupsum


def test_driscoll_kraay_matches_statsmodels_regression():
    rng = np.random.default_rng(5)
    T, N = 150, 4
    x = factor_panel(T, N, 0.5, 0.4, rng)
    u = factor_panel(T, N, 0.7, 0.3, rng)
    y = 0.3 + 0.8 * x + u
    tt = np.repeat(np.arange(T), N)
    X = np.column_stack([np.ones(T * N), x.ravel()])
    lag = 4
    mine = pt.panel_ols_dk(y.ravel(), X, tt, lag=lag)
    sm_res = sm.OLS(y.ravel(), X).fit(cov_type="hac-groupsum",
                                      cov_kwds={"time": tt, "maxlags": lag, "use_correction": False})
    assert np.allclose(mine["beta"], sm_res.params, rtol=1e-12)
    assert np.allclose(mine["cov"], sm_res.cov_params(), rtol=1e-10)


def test_unbalanced_index_means_match_statsmodels_dummies():
    rng = np.random.default_rng(6)
    T, N = 90, 3
    D = factor_panel(T, N, 0.5, 0.2, rng, mu=0.2)
    obs = rng.random((T, N)) < 0.35          # sparse, like expiry days
    obs[5] = False                           # a date with no observation is dropped
    r = pt.index_means_dk(D, observed=obs, lag=3)
    keep = obs.any(axis=1)
    Dk, ok = D[keep], obs[keep]
    tt, ii = np.nonzero(ok)
    X = np.zeros((tt.size, N))
    X[np.arange(tt.size), ii] = 1.0
    sm_res = sm.OLS(Dk[tt, ii], X).fit(cov_type="hac-groupsum",
                                       cov_kwds={"time": tt, "maxlags": 3, "use_correction": False})
    assert r["T"] == keep.sum()
    assert np.array_equal(r["n"], ok.sum(axis=0))
    assert np.allclose(r["means"], [D[obs[:, i], i].mean() for i in range(N)])
    assert np.allclose(r["means"], sm_res.params)
    assert np.allclose(r["cov"], sm_res.cov_params(), rtol=1e-10)


def test_driscoll_kraay_size_under_strong_cross_sectional_dependence():
    """Under the null (mean 0), equicorrelated (0.8) and autocorrelated (0.3) indices: the DK test keeps its size,
    a naive test that treats every index-day as independent does not."""
    rng = np.random.default_rng(7)
    rej_dk = rej_naive = 0
    sims = 400
    for _ in range(sims):
        D = factor_panel(300, 3, 0.8, 0.3, rng)
        r = pt.index_means_dk(D)
        c = pt.combo_test(r["means"], r["cov"], np.full(3, 1 / 3), r["T"])
        rej_dk += c["p"] <= 0.05
        flat = D.ravel()
        t_naive = flat.mean() / (flat.std(ddof=1) / np.sqrt(flat.size))
        rej_naive += t_naive > 1.645
    assert 0.02 <= rej_dk / sims <= 0.10
    assert rej_naive / sims > 0.15


def test_driscoll_kraay_has_power():
    rng = np.random.default_rng(8)
    D = factor_panel(600, 3, 0.5, 0.2, rng, mu=0.25)
    r = pt.index_means_dk(D)
    c = pt.combo_test(r["means"], r["cov"], np.full(3, 1 / 3), r["T"])
    assert c["p"] < 0.01 and c["estimate"] > 0


# ---------------------------------------------------------------- Wald test of equal means


def test_wald_two_indices_equals_squared_t_of_difference():
    rng = np.random.default_rng(9)
    D = factor_panel(200, 2, 0.6, 0.2, rng) + np.array([0.0, 0.2])
    r = pt.index_means_dk(D)
    w = pt.wald_equal(r["means"], r["cov"], r["T"])
    c = pt.combo_test(r["means"], r["cov"], [1.0, -1.0], r["T"], alternative="two-sided")
    assert w["q"] == 1
    assert np.isclose(w["F"], c["stat"] ** 2, rtol=1e-10)
    assert np.isclose(w["p"], c["p"], rtol=1e-8)


def test_wald_size_and_power():
    rng = np.random.default_rng(10)
    rej = 0
    for _ in range(300):
        D = factor_panel(300, 3, 0.7, 0.3, rng, mu=0.1)     # equal means
        r = pt.index_means_dk(D)
        rej += pt.wald_equal(r["means"], r["cov"], r["T"])["p"] <= 0.05
    assert 0.01 <= rej / 300 <= 0.11
    D = factor_panel(300, 3, 0.7, 0.3, rng) + np.array([0.0, 0.0, 0.6])
    r = pt.index_means_dk(D)
    assert pt.wald_equal(r["means"], r["cov"], r["T"])["p"] < 0.01
    assert pt.contrast_matrix(3).tolist() == [[1, -1, 0], [0, 1, -1]]


# ---------------------------------------------------------------- effective number of indices


def test_effective_numbers_extremes_and_equicorrelation():
    e = pt.effective_numbers(np.eye(3), np.diag([1.0, 2.0, 3.0]))
    for k in ("kish", "participation", "nyholt", "li_ji", "design_lr", "kish_lr"):
        assert np.isclose(e[k], 3.0), k
    e1 = pt.effective_numbers(np.ones((3, 3)), np.ones((3, 3)) * 2.0)
    for k in ("kish", "participation", "nyholt", "li_ji", "design_lr", "kish_lr", "nifty_equiv_equal", "nifty_equiv_gls"):
        assert np.isclose(e1[k], 1.0), k
    rho = 0.5
    R = np.full((3, 3), rho) + (1 - rho) * np.eye(3)
    e2 = pt.effective_numbers(R, R)
    assert np.isclose(e2["kish"], 3 / (1 + 2 * rho))
    assert np.isclose(e2["design_lr"], 3 / (1 + 2 * rho))
    assert np.isclose(e2["participation"], 9 / (3 + 6 * rho ** 2))
    assert np.isclose(e2["eigenvalue_1"], 1 + 2 * rho) and np.isclose(e2["eigenvalue_3"], 1 - rho)
    # equal variances and independence: the equal-weight pooled mean is worth three single indices
    e3 = pt.effective_numbers(None, np.eye(3))
    assert np.isclose(e3["nifty_equiv_equal"], 3) and np.isclose(e3["nifty_equiv_gls"], 3)


def test_design_effect_equals_variance_ratio_of_pooled_test():
    rng = np.random.default_rng(11)
    D = factor_panel(400, 3, 0.4, 0.2, rng) * np.array([1.0, 1.5, 0.7])
    lag = pt.dk_lag(400)
    r = pt.index_means_dk(D, lag=lag)
    a = np.full(3, 1 / 3)
    v_dk = a @ r["cov"] @ a
    v_ind = np.sum(np.diag(r["cov"])) / 9
    e = pt.effective_numbers(None, pt.long_run_cov(D, lag))
    assert np.isclose(e["design_lr"], 3 * v_ind / v_dk, rtol=1e-12)
    assert np.allclose(pt.long_run_cov(D, lag) / 400, r["cov"], rtol=1e-12)


def test_effective_number_recovers_population_value():
    rng = np.random.default_rng(12)
    D = factor_panel(20000, 3, 0.5, 0.0, rng)
    e = pt.effective_numbers(np.corrcoef(D, rowvar=False), pt.long_run_cov(D, 5))
    assert abs(e["kish"] - 1.5) < 0.05 and abs(e["design_lr"] - 1.5) < 0.08


# ---------------------------------------------------------------- joint stationary bootstrap


def test_stationary_bootstrap_indices():
    n = 400
    idx = pt.stationary_bootstrap_indices(n, 300, 10, np.random.default_rng(1))
    assert idx.shape == (300, n) and idx.min() >= 0 and idx.max() < n
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % n).mean()
    assert abs((1 - cont) - 0.1 * (1 - 1 / n)) < 0.01
    again = pt.stationary_bootstrap_indices(n, 300, 10, np.random.default_rng(1))
    assert np.array_equal(idx, again)


def test_bootstrap_p_value_formula():
    assert np.isclose(pt.bootstrap_p_greater(np.array([0.5, 1.5, 2.5, 3.0]), 1.0), 3 / 5)
    assert np.isclose(pt.bootstrap_p_greater(np.zeros(99), 1.0), 1 / 100)


def test_joint_resampling_keeps_cross_index_structure():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(200)
    D = np.column_stack([x, x, 2 * x])
    idx = pt.stationary_bootstrap_indices(200, 50, 10, rng)
    m = pt.boot_index_means(D, idx, chunk=7)
    assert np.allclose(m[:, 0], m[:, 1]) and np.allclose(m[:, 2], 2 * m[:, 0])
    assert np.allclose(m, np.stack([D[i].mean(axis=0) for i in idx]))


def test_boot_dependence_matches_direct_computation():
    rng = np.random.default_rng(13)
    D = factor_panel(150, 3, 0.5, 0.3, rng)
    idx = pt.stationary_bootstrap_indices(150, 20, 10, rng)
    bd = pt.boot_dependence(D, idx, 3, chunk=6)
    for r in (0, 7, 19):
        assert np.allclose(bd["corr"][r], np.corrcoef(D[idx[r]], rowvar=False))
        assert np.allclose(bd["omega"][r], pt.long_run_cov(D[idx[r]], 3))


def test_bounds_from_committed_table(tmp_path):
    names = ["Nifty (v2 forecasts, same dates)", "Bank Nifty (P91 forecasts, same dates)", "Fin Nifty"]
    rows = []
    for target, mus, ts in (("next15", [0.02, 0.01, 0.015], [8.0, 6.0, 5.0]), ("last_hour", [0.01, 0.02, 0.002], [2.0, 4.0, 0.5])):
        for nm, mu, t in zip(names, mus, ts):
            rows.append({"index": nm, "target": target, "subset": "all", "qlike_mcsgarch": 0.3, "qlike_model": 0.3 - mu,
                         "dm_stat": t})
            rows.append({"index": nm, "target": target, "subset": f"expiry ({nm} own)", "qlike_mcsgarch": 0.3,
                         "qlike_model": 0.3 - mu / 2, "dm_stat": t / 2})
    f = tmp_path / "ref.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    b = pt.bounds_from_committed(f)
    assert len(b) == 4
    r = b[(b.target == "next15") & (b.subset == "all")].iloc[0]
    mu, t = np.array([0.02, 0.01, 0.015]), np.array([8.0, 6.0, 5.0])
    s = mu / t
    assert np.isclose(r.pooled_t_if_perfectly_correlated, mu.sum() / s.sum())
    assert np.isclose(r.pooled_t_if_independent, mu.sum() / np.sqrt((s ** 2).sum()))
    assert (b.pooled_t_if_perfectly_correlated >= b.min_t - 1e-12).all()
    # the floor holds for the DK test itself, whatever the correlation (synthetic panel, positive means)
    rng = np.random.default_rng(14)
    for rho in (0.0, 0.5, 0.95):
        D = factor_panel(500, 3, rho, 0.2, rng) * np.array([1.0, 2.0, 0.5]) + np.array([0.15, 0.2, 0.05])
        r_ = pt.index_means_dk(D)
        tp = pt.combo_test(r_["means"], r_["cov"], np.full(3, 1 / 3), r_["T"])["stat"]
        ti = [pt.combo_test(r_["means"], r_["cov"], np.eye(3)[i], r_["T"])["stat"] for i in range(3)]
        assert tp >= min(ti) - 1e-12


# ---------------------------------------------------------------- end to end on synthetic forecast files


def write_synthetic_index(folder: Path, dates, rng, gain, common_shock, expiry_weekday):
    """Synthetic forecast files (SYNTHETIC) with the columns the study reads. The hybrid's forecast error is smaller
    than mcsGARCH's by ``gain``; ``common_shock`` (one value per date) is shared by every index."""
    folder.mkdir(parents=True, exist_ok=True)
    for target, minutes in (("next15", [0, 15, 30]), ("next30", [0, 30]), ("last_hour", [315])):
        rows = []
        for j, d in enumerate(dates):
            is_exp = pd.Timestamp(d).weekday() == expiry_weekday
            for m in minutes:
                lv = np.exp(common_shock[d] + 0.3 * rng.standard_normal())
                y = lv * rng.chisquare(5) / 5
                e_m = np.exp(0.5 * rng.standard_normal() + 0.3 * common_shock[d])
                e_h = np.exp((0.5 - gain) * rng.standard_normal() + 0.3 * common_shock[d])
                rows.append({"date": d, "minute": m, "is_expiry": is_exp, "regime": 0, "y": y,
                             "mcsgarch": lv * e_m, "hybrid_base": lv * e_h, "mcsgarch_iv": lv * e_m * 1.02})
        pd.DataFrame(rows).to_csv(folder / f"forecasts_{target}.csv", index=False)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("p107_synthetic")
    rng = np.random.default_rng(21)
    all_dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=140)]
    shock = {d: 0.6 * rng.standard_normal() for d in all_dates}
    dirs = {"nifty": base / "a", "banknifty": base / "b", "finnifty": base / "c"}
    write_synthetic_index(dirs["nifty"], all_dates, rng, 0.30, shock, 3)
    write_synthetic_index(dirs["banknifty"], all_dates[:130], rng, 0.25, shock, 2)
    write_synthetic_index(dirs["finnifty"], all_dates[5:], rng, 0.25, shock, 1)
    out = base / "out"
    res = pt.run(dirs, out, label="SYNTHETIC", draws=199, verbose=False)
    return dirs, out, res, all_dates


def test_end_to_end_outputs(synthetic_run):
    dirs, out, res, all_dates = synthetic_run
    tests = pd.read_csv(out / "tests.csv")
    required = ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness",
                "p_method", "p_adjusted", "adjust_method"]
    assert list(tests.columns[:11]) == required
    assert (tests["data"] == "SYNTHETIC").all()
    assert tests["p_raw"].between(0, 1).all()
    assert set(tests["role"]) <= {"primary", "secondary", "robustness", "placebo", "exploratory", "post hoc"}
    prim = tests[tests.role == "primary"]
    assert len(prim) == 2 and (prim["p_raw"] < 0.01).all() and (prim["estimate"] > 0).all()
    # Holm over the two primary tests
    p = np.sort(prim["p_raw"].to_numpy())
    assert np.isclose(prim["p_adjusted"].min(), min(1, 2 * p[0]))
    for f in ("correlations.csv", "effective_n.csv", "pooled_estimates.csv", "se_comparison.csv", "bootstrap.csv",
              "per_index.csv", "daily_losses.csv", "verdicts.csv", "run_info.json", "stdout.txt"):
        assert (out / f).exists(), f
    info = json.loads((out / "run_info.json").read_text())
    assert info["label"] == "SYNTHETIC"
    # common dates = the intersection: dates 6 to 130
    assert info["common_dates"]["n"] == 125
    assert info["common_dates"]["first"] == all_dates[5] and info["common_dates"]["last"] == all_dates[129]
    assert len(info["inputs"]) == 9 and all(len(v["sha256"]) == 64 for v in info["inputs"].values())
    v = pd.read_csv(out / "verdicts.csv")
    assert v.loc[v.family == "P1 primary", "verdict"].iloc[0] == "pooled gain confirmed"


def test_end_to_end_numbers_are_consistent(synthetic_run):
    dirs, out, res, all_dates = synthetic_run
    daily = pd.read_csv(out / "daily_losses.csv")
    d15 = daily[daily.target == "next15"].pivot(index="date", columns="index",
                                                values="qlike_mcsgarch") - \
        daily[daily.target == "next15"].pivot(index="date", columns="index", values="qlike_hybrid_base")
    pooled = d15[["Nifty", "Bank Nifty", "Fin Nifty"]].mean(axis=1).mean()
    tests = pd.read_csv(out / "tests.csv").set_index("test_id")
    assert np.isclose(tests.loc["P1_next15", "estimate"], pooled, rtol=1e-8)
    # the DK statistic equals the project's DM test on the cross-sectional mean series
    cs = d15[["Nifty", "Bank Nifty", "Fin Nifty"]].mean(axis=1).to_numpy()
    dm = diebold_mariano(cs, np.zeros_like(cs), alternative="greater")
    assert np.isclose(tests.loc["P1_next15", "statistic"], dm["stat"], rtol=1e-8)
    assert np.isclose(tests.loc["P1_next15", "p_raw"], dm["p"], rtol=1e-8)
    # expiry rows use each index's own expiry days (weekday 3, 2 and 1 here: disjoint day sets)
    e = daily[(daily.target == "next15") & daily.is_expiry]
    n_by = e.groupby("index").size()
    assert f"Nifty {n_by['Nifty']}" in tests.loc["S1d_next15", "sample"]
    assert f"{int(n_by.sum())} index-days on {int(n_by.sum())} dates" in tests.loc["S1d_next15", "sample"]
    neff = pd.read_csv(out / "effective_n.csv")
    main = neff[(neff.target == "next15") & (neff.benchmark == "mcsgarch") & (neff.set == "all three")]
    v = main.set_index("measure")["value"]
    # the design effect is the variance ratio of the pooled test (it may exceed 3 if correlations are negative)
    se = pd.read_csv(out / "se_comparison.csv").set_index("target")
    assert np.isclose(v["design_lr"], 3 / se.loc["next15", "variance_ratio_dk_over_independent"], rtol=1e-10)
    assert v["design_lr"] > 0 and v["kish"] > 0
    assert main.loc[main.measure == "design_lr", "ci95_lo"].notna().all()


def test_count_only_and_misaligned_minutes(synthetic_run, tmp_path):
    dirs, out, res, all_dates = synthetic_run
    c = pt.count_only(dirs)
    assert c["common_dates"] == 125 and c["same_anchor_minutes_next15"]
    bad = tmp_path / "bad"
    bad.mkdir()
    for t in pt.TARGETS:
        f = pd.read_csv(Path(dirs["finnifty"]) / f"forecasts_{t}.csv")
        if t == "next15":
            f.loc[f.minute == 30, "minute"] = 45
        f.to_csv(bad / f"forecasts_{t}.csv", index=False)
    dates = pt.common_dates({**dirs, "finnifty": bad})
    with pytest.raises(ValueError):
        pt.load_daily({**dirs, "finnifty": bad}, "next15", dates)
