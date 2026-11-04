"""Fast unit tests for experiments/referee_f2.py (synthetic data only, no real files)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "referee_f2.py"
_spec = importlib.util.spec_from_file_location("referee_f2", _PATH)
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)


def _ar_errors(rng, n, k=4):
    return np.convolve(rng.normal(size=n + k - 1), np.ones(k) / 2, mode="valid")[:n]


# ---------------------------------------------------------------- standard errors


def test_ols_focus_matches_statsmodels_hac():
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = X @ np.array([0.5, 1.0, -2.0]) + _ar_errors(rng, n)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    for j in range(3):
        f = rf.ols_focus(y, X, j, lags=5)
        assert np.isclose(f["coef"], ref.params[j])
        assert np.isclose(f["se"], ref.bse[j], rtol=1e-10)
        assert np.isclose(f["p"], ref.pvalues[j], rtol=1e-8)


def test_cluster_se_matches_statsmodels_cr1():
    rng = np.random.default_rng(1)
    n, G = 400, 25
    g = np.repeat(np.arange(G), n // G)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = X @ np.array([0.1, 0.3, 0.0]) + rng.normal(size=G)[g] + rng.normal(size=n)
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    f = rf.ols_focus(y, X, 2)
    se, nG = rf.cluster_se(f["h"], f["u"], g, X.shape[1])
    assert nG == G
    assert np.isclose(se, ref.bse[2], rtol=1e-10)


def test_dk_lag_rule():
    assert rf.dk_lag(932) == 6
    assert rf.dk_lag(452) == 5
    assert rf.dk_lag(100) == 4


def test_wild_cluster_bootstrap_size_and_power():
    rng = np.random.default_rng(2)
    n, G = 240, 20
    g = np.repeat(np.arange(G), n // G)
    x = rng.normal(size=G)[g] + rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y_null = 0.2 + rng.normal(size=G)[g] + rng.normal(size=n)
    y_alt = 0.2 + 1.0 * x + rng.normal(size=G)[g] + rng.normal(size=n)
    p_null = rf.wild_cluster_p(y_null, X, 1, g, 399, np.random.default_rng(3))["p"]
    p_alt = rf.wild_cluster_p(y_alt, X, 1, g, 399, np.random.default_rng(3))["p"]
    assert p_alt < 0.01
    assert p_null > p_alt
    # rejection rate under the null over a few replications is not wildly above 5%
    rej = 0
    for r in range(40):
        yy = 0.2 + rng.normal(size=G)[g] + rng.normal(size=n)
        rej += rf.wild_cluster_p(yy, X, 1, g, 199, np.random.default_rng(100 + r))["p"] < 0.05
    assert rej <= 7


def test_fixed_b_critical_values_grow_with_bandwidth():
    rng = np.random.default_rng(4)
    small = rf.fixed_b_null(400, 2, 4000, rng)
    large = rf.fixed_b_null(400, 80, 4000, rng)
    q_small, q_large = np.quantile(small, 0.95), np.quantile(large, 0.95)
    assert 1.8 < q_small < 2.15  # close to 1.96 when b = M/T is tiny
    assert q_large > q_small + 0.2


# ---------------------------------------------------------------- resampling


def test_block_relocate_keeps_values_and_inner_order():
    rng = np.random.default_rng(5)
    x = np.arange(100)
    y = rf.block_relocate(x, 10, rng)
    assert sorted(y) == list(x)
    # apart from block joins and the one wrap point, consecutive values differ by +1
    steps = np.diff(y)
    assert np.sum(steps != 1) <= 11


def test_within_block_shuffle_keeps_block_contents():
    rng = np.random.default_rng(6)
    x = np.arange(95)
    y = rf.within_block_shuffle(x, 20, rng)
    for i in range(0, 95, 20):
        assert sorted(y[i:i + 20]) == list(x[i:i + 20])


def test_moving_block_indices_shape_and_contiguity():
    rng = np.random.default_rng(7)
    idx = rf.moving_block_indices(103, 10, rng)
    assert len(idx) == 103
    assert idx.min() >= 0 and idx.max() <= 102
    assert np.all(np.diff(idx[:10]) == 1)


# ---------------------------------------------------------------- persistence measures


def test_runs_and_episode_ids():
    ind = [1, 1, 0, 0, 0, 1]
    assert rf.runs_of(ind) == [(0, 1, 1.0), (2, 4, 0.0), (5, 5, 1.0)]
    assert list(rf.episode_ids(ind)) == [0, 0, 1, 1, 1, 2]


def test_effective_sample_size_of_a_markov_indicator():
    rng = np.random.default_rng(8)
    n, stay = 20000, 0.9
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] if rng.random() < stay else 1 - x[t - 1]
    rho = 2 * stay - 1  # lag-1 autocorrelation of a symmetric two-state chain
    assert abs(rf.acf(x, 1) - rho) < 0.02
    assert abs(rf.n_eff_ar1(x) - n * (1 - rho) / (1 + rho)) / (n * (1 - rho) / (1 + rho)) < 0.15
    tau, _ = rf.iat_sokal(x)
    assert abs(tau - (1 + rho) / (1 - rho)) < 2.5


# ---------------------------------------------------------------- angles


def test_closed_form_angle_matches_the_grid_estimator():
    df = rf.dsg.valid(rf.dsg.synthetic_frame(n=600, angle=120.0, snr_b=0.3, seed=1))
    y, Z, Bm = rf.dsg.design_arrays(df, "b", "participants")
    th = rf.theta_from_arrays(y, Z, Bm)
    grid = rf.dsg.estimate_direction([rf.dsg.gram_stats(y, Z, Bm)])["angle"]
    assert abs(rf.circ_diff(rf.angle_of(th), grid)) <= 1.0
    assert abs(rf.circ_diff(rf.angle_of(th), 120.0)) < 25.0


def test_same_direction_minimum_distance_test():
    V = np.eye(4) * 0.01
    same = rf.same_direction_md([-1.0, -0.2], [-2.0, -0.45], V)
    opposite = rf.same_direction_md([-1.0, -0.2], [2.0, 0.4], V)
    assert same["p"] > 0.2
    assert opposite["p"] < 1e-6
    assert abs(rf.circ_diff(same["phi_common"], np.degrees(np.arctan2(0.2, 1.0)))) < 5


def test_circular_helpers():
    assert np.isclose(rf.circ_diff(10, 350), 20)
    assert np.isclose(rf.circ_diff(21, 190), -169)
    assert rf.arc_contains_zero(-20, 30)
    assert not rf.arc_contains_zero(-254, -63)
    assert rf.arc_contains_zero(300, 370)


# ---------------------------------------------------------------- microstructure noise


def test_pooled_first_order_autocorrelation_of_noisy_returns():
    rng = np.random.default_rng(9)
    days, m = 400, 31
    sig, om = 1.0, 0.5
    p = np.cumsum(rng.normal(scale=sig, size=(days, m)), axis=1) + rng.normal(scale=om, size=(days, m))
    R = np.diff(p, axis=1)
    S = rf.pooled_pairs_stats(R, list(range(R.shape[1])))
    est = float(rf.pooled_corr(S[["n", "sx", "sy", "sxx", "syy", "sxy"]].to_numpy().sum(axis=0)))
    theory = -om ** 2 / (sig ** 2 + 2 * om ** 2)
    assert abs(est - theory) < 0.03
    bt = rf.boot_pooled_corr(S[["n", "sx", "sy", "sxx", "syy", "sxy"]].to_numpy(), 200, rng)
    assert np.percentile(bt, 2.5) < theory < np.percentile(bt, 97.5)


# ---------------------------------------------------------------- data builders


def test_rule_gex_on_a_synthetic_panel():
    idx = pd.bdate_range("2024-01-01", periods=5)
    C = np.array([10.0, 12.0, 8.0, 9.0, 11.0])
    P = np.array([6.0, 7.0, 9.0, 5.0, 4.0])
    panel = pd.DataFrame({"gross_gex": C + P, "net_gex_us_convention": C - P}, index=idx)
    nets = pd.DataFrame({f"{c}_{s}": np.linspace(-0.02, 0.02, 5) * (k + 1) * (1 if s == "call" else -1)
                         for k, c in enumerate(rf.CATEGORIES) for s in ("call", "put")}, index=idx)
    g = rf.rule_gex(panel, nets)
    assert np.allclose(g["pro"], nets["pro_call"] * C + nets["pro_put"] * P)
    assert np.allclose(g["client_mirror"], -g["client"])
    assert np.allclose(g["fii_plus_pro"], g["fii"] + g["pro"])
    assert np.allclose(g["us_convention"], C - P)
    assert np.allclose(g["short_all"], -(C + P))


def test_sensex_days_roll_back_over_a_holiday(tmp_path):
    cal = tmp_path / "cal.csv"
    pd.DataFrame([{"fact_type": "weekday_regime", "date": "2024-01-01", "date_end": "2024-01-31", "weekday": "Fri"}]).to_csv(cal, index=False)
    days = pd.bdate_range("2024-01-01", "2024-01-31").drop(pd.Timestamp("2024-01-26"))  # a Friday holiday
    out = rf.sensex_expiry_days(cal, days)
    assert list(out.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-25"]


def _synthetic_f2_frame(n=600, c=-0.07, seed=10):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    neg = np.zeros(n)
    for t in range(1, n):
        neg[t] = neg[t - 1] if rng.random() < 0.9 else 1 - neg[t - 1]
    r = rng.normal(scale=60, size=n)
    y = -0.01 * r + c * r * neg + rng.normal(scale=15, size=n)
    d = pd.DataFrame({"r_rod": r, "r_last": y, "neg_pro": neg, "neg_short_all": 1.0,
                      "lvix_prev": rng.normal(size=n), "is_expiry": (np.arange(n) % 5 == 3)}, index=idx)
    d["month"] = d.index.to_period("M").astype(str)
    d["yq"] = d.index.to_period("Q").astype(str)
    d["qoy"], d["moy"] = d.index.quarter, d.index.month
    d["janmar"] = (d.index.month <= 3).astype(float)
    d["regime4"] = "multi_weekly"
    d["sensex_expiry"] = (np.arange(n) % 5 == 4).astype(float)
    return d


def test_infer_f2_recovers_the_interaction_and_reports_every_method():
    d = _synthetic_f2_frame()
    cfg = {"n_perm": 99, "n_wild": 99, "n_fixedb": 200}
    methods = ("hac5", "dk", "nw60", "cluster_month", "wild_month", "cluster_episode", "wild_episode",
               "cluster_quarter", "wild_quarter", "perm_relocate_60", "perm_within_60")
    rows = pd.DataFrame(rf.infer_f2(d, "pro", "binary", np.ones(len(d), bool), "r_last", methods, cfg, [1], {}))
    assert abs(rows["coef"].iloc[0] - (-0.07)) < 0.03
    assert {"newey_west_5", "driscoll_kraay", "newey_west_60_fixed_b", "cluster_month", "wild_cluster_month",
            "cluster_episode", "wild_cluster_episode", "cluster_quarter", "wild_cluster_quarter",
            "block_permutation_60_t", "block_permutation_60_coef", "within_block_shuffle_60_t"} <= set(rows["method"])
    assert (rows.loc[rows["method"] == "newey_west_5", "p"] < 0.01).all()
    assert rows["n_episodes"].iloc[0] == len(rf.runs_of(d["neg_pro"].to_numpy()))
    # Jan to Mar split and the all-negative short_all rule
    jm = pd.DataFrame(rf.infer_f2(d, "pro", "janmar", np.ones(len(d), bool), "r_last", ("hac5",), cfg, [2], {}))
    assert set(jm["term"]) == {"N_x_r_rod_notJM", "N_x_r_rod_JM"}
    sa = rf.infer_f2(d, "short_all", "binary", np.ones(len(d), bool), "r_last", ("hac5",), cfg, [3], {})
    assert sa[0]["method"] == "not estimable"


def test_null_interaction_is_not_significant_under_block_permutation():
    d = _synthetic_f2_frame(c=0.0, seed=11)
    cfg = {"n_perm": 199, "n_wild": 99, "n_fixedb": 200}
    rows = pd.DataFrame(rf.infer_f2(d, "pro", "binary", np.ones(len(d), bool), "r_last", ("perm_relocate_60",), cfg, [4], {}))
    assert (rows["p"] > 0.01).all()


def test_build_frame_split_returns_add_up():
    idx = pd.bdate_range("2024-03-01", periods=4)
    rng = np.random.default_rng(12)
    closes = 20000 * np.exp(np.cumsum(rng.normal(scale=1e-4, size=(4, rf.N_SLOTS)), axis=1))
    dm = pd.DataFrame({"is_expiry": [False, True, False, False], "regime": "multi_weekly", "vix": 0.13}, index=idx)
    gex = pd.DataFrame({r: [1.0, -1.0, 1.0, -1.0] for r in rf.RULES}, index=idx)
    gex["gross"] = 5.0
    d = rf.build_frame(dm, gex, closes, pd.DatetimeIndex([idx[2]]), {"sebi_feb2025": pd.Timestamp("2024-03-05")})
    ok = d["r_rod"].notna()
    assert ok.sum() == 3
    assert np.allclose(d["r_1459_1515"] + d["r_1515_1529"], d["r_last"])
    assert np.allclose(d["r_1459_1500"] + d["r_1500_1529"], d["r_last"])
    assert list(d["neg_pro"].iloc[1:]) == [0.0, 1.0, 0.0]  # the sign of the evening before
    assert d["sensex_expiry"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert d["sebi_feb2025"].tolist() == [0.0, 0.0, 1.0, 1.0]
