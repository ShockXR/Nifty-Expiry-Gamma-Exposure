"""Tests for study P113 (experiments/p113_connectedness_settings.py and the new decompositions in
niftygex.connectedness). SYNTHETIC data only: no real data file is read."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex import connectedness as cn
from niftygex import replication as rp

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p113_connectedness_settings.py"
_spec = importlib.util.spec_from_file_location("p113_connectedness_settings", _PATH)
p113 = importlib.util.module_from_spec(_spec)
sys.modules["p113_connectedness_settings"] = p113
_spec.loader.exec_module(p113)

# A known 2-variable VAR(1) used for the hand calculations (SYNTHETIC).
A1 = np.array([[[0.5, 0.2], [0.3, 0.4]]])
S1 = np.array([[1.0, 0.6], [0.6, 2.0]])


def _simulate(T=4000, seed=0, A=A1[0], Sigma=S1):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.normal(size=2)
    return Y


def test_unnormalised_gfevd_matches_hand_calculation():
    # H = 2: Psi_0 = I, Psi_1 = A. A Sigma = [[0.62, 0.70], [0.54, 0.98]];
    # A Sigma A' = [[0.45, 0.466], [0.466, 0.554]].
    # theta_12 = (0.6^2 + 0.70^2) / 2.0 / (1.0 + 0.45) = 0.425 / 1.45
    # theta_11 = (1.0^2 + 0.62^2) / 1.0 / 1.45 = 1.3844 / 1.45
    # theta_21 = (0.6^2 + 0.54^2) / 1.0 / (2.0 + 0.554) = 0.6516 / 2.554
    # theta_22 = (2.0^2 + 0.98^2) / 2.0 / 2.554 = 2.4802 / 2.554
    hand = np.array([[1.3844 / 1.45, 0.425 / 1.45], [0.6516 / 2.554, 2.4802 / 2.554]])
    th = cn.fevd_table(A1, S1, 2, "generalized_raw")
    assert np.allclose(th, hand, atol=1e-12)
    assert th[0].sum() > 1 and th[1].sum() > 1  # unnormalised rows sum to more than 1
    assert np.allclose(cn.fevd_table(A1, S1, 2, "generalized_row"), hand / hand.sum(1, keepdims=True))
    assert np.allclose(cn.fevd_table(A1, S1, 2, "generalized_column"), hand / hand.sum(0, keepdims=True))
    assert np.allclose(cn.fevd_table(A1, S1, 2), cn.gfevd(A1, S1, 2))  # default unchanged


def test_unnormalised_cross_shares_can_both_exceed_half():
    # No dynamics, correlation 0.8: raw cross share = rho^2 = 0.64 in both rows, total 128 on the paper's scale;
    # the row-normalised share is rho^2 / (1 + rho^2) = 0.39.
    A = np.zeros((1, 2, 2))
    S = np.array([[1.0, 0.8], [0.8, 1.0]])
    raw = cn.fevd_table(A, S, 10, "generalized_raw")
    assert np.allclose([raw[0, 1], raw[1, 0]], 0.64)
    assert np.isclose(cn.fevd_table(A, S, 10)[0, 1], 0.64 / 1.64)


def test_cholesky_impact_and_order():
    rho = 0.6 / np.sqrt(2.0)
    A = np.zeros((1, 2, 2))
    th = cn.cholesky_fevd(A, S1, 1)
    assert np.allclose(th, [[1.0, 0.0], [rho**2, 1 - rho**2]])  # Nifty first: Bank Nifty gets rho^2 from Nifty
    thr = cn.fevd_table(A, S1, 1, "cholesky_reversed")
    assert np.allclose(thr, [[1 - rho**2, rho**2], [0.0, 1.0]])
    # with dynamics, rows still sum to 1 and the order matters
    a = cn.cholesky_fevd(A1, S1, 10)
    b = cn.cholesky_fevd(A1, S1, 10, order=[1, 0])
    assert np.allclose(a.sum(1), 1) and np.allclose(b.sum(1), 1) and not np.allclose(a, b)


def test_cholesky_matches_statsmodels_fevd():
    from statsmodels.tsa.api import VAR

    Y = _simulate(3000, seed=4)
    res = VAR(Y).fit(1)
    sm = res.fevd(10).decomp  # (N, H, N): variable, horizon, shock
    fit = cn.fit_var(Y, 1)
    # Cholesky shares do not change if Sigma is multiplied by a constant, so the degrees-of-freedom convention does not matter
    mine = cn.cholesky_fevd(res.coefs, res.sigma_u, 10)
    assert np.allclose(mine, sm[:, 9, :], atol=1e-10)
    assert np.allclose(fit.A, res.coefs, atol=1e-10)


def test_batched_decompositions_match_package_functions():
    rng = np.random.default_rng(1)
    A = rng.normal(scale=0.25, size=(6, 3, 2, 2))
    S = np.stack([np.array([[1.0, r], [r, s]]) for r, s in zip(rng.uniform(-0.5, 0.9, 6), rng.uniform(0.5, 3, 6))])
    out = p113.decomp_batch(A, S)
    meth = {"GR": "generalized_row", "GU": "generalized_raw", "GC": "generalized_column", "CN": "cholesky", "CB": "cholesky_reversed"}
    for b in range(6):
        for k, d in enumerate(p113.DECOMPS):
            for hi, H in enumerate(p113.HS):
                th = 100 * cn.fevd_table(A[b], S[b], H, meth[d])
                assert np.allclose(out[b, k, hi], [th[0, 1], th[1, 0]], atol=1e-9), (b, d, H)


def test_rolling_fits_equal_direct_fits():
    Y = _simulate(700, seed=2) * np.array([1e-4, 3e-4])  # small scale, as minute returns
    ends = np.array([10, 199, 250, 433, 699])
    for p in (1, 2, 5):
        rc = p113.rolling_cross(Y, ends, p, w=200)
        assert np.isnan(rc[0]).all()  # window not full
        for i, e in enumerate(ends[1:], start=1):
            fit = cn.fit_var(Y[e - 199 : e + 1], p)
            ref = p113.decomp_batch(fit.A[None], fit.Sigma[None])[0]
            assert np.allclose(rc[i], ref, atol=1e-7), (p, e)


def test_segmented_and_concatenated_designs():
    Y = _simulate(600, seed=3)
    seg = np.repeat([0, 1, 2], 200)
    sh_s, p = p113.fit_cross(Y, seg, "2")
    ref_s = rp.fit_var_segments(Y, seg, 2)
    assert p == 2 and ref_s.resid.shape[0] == 600 - 3 * 2  # lags never cross a block
    assert np.allclose(sh_s, p113.decomp_batch(ref_s.A[None], ref_s.Sigma[None])[0])
    sh_c, _ = p113.fit_cross(Y, np.zeros(600, int), "2")
    ref_c = cn.fit_var(Y, 2)  # concatenated = one plain VAR on the joined rows
    assert np.allclose(sh_c, p113.decomp_batch(ref_c.A[None], ref_c.Sigma[None])[0])
    assert not np.allclose(sh_s, sh_c)


def test_bic_lag_selection_finds_true_order():
    rng = np.random.default_rng(5)
    T = 6000
    Y = np.zeros((T, 2))
    for t in range(2, T):
        Y[t] = 0.4 * Y[t - 1] - 0.3 * Y[t - 2] + rng.normal(size=2)
    seg = np.zeros(T, int)
    assert p113.var_lag_ic(Y, seg, 10, "bic") == 2
    assert p113.var_lag_ic(Y, seg, 10, "aic") >= 2


def test_spec_grid_size_and_baseline_present():
    s = p113.spec_list()
    assert len(s) == 5 * 8 * 8 * 17 == 5440
    assert s["spec"].is_unique
    b = p113.BASELINE
    assert ((s.D == b["D"]) & (s.I == b["I"]) & (s.C == b["C"]) & (s.L == b["L"]) & (s.H == b["H"])).sum() == 1


def _fake_grid(rng, n=400):
    return rng.uniform(55, 90, size=(n, 15))


def test_shuffle_guard_detects_a_planted_fit_and_not_a_random_grid():
    rng = np.random.default_rng(7)
    tgt = np.array([84.0, 74.7, 99.42, 68.64, 105.22, 76.38, 74.58, 80.18, 68.81, 66.44, 73.02, 72.74, 70.97, 66.54, 74.43])
    T = _fake_grid(rng)
    planted = T.copy()
    planted[17] = tgt + rng.normal(scale=0.3, size=15)  # one setting reproduces the table
    g = p113.permutation_guard(planted, tgt, n_perm=200, seed=1)
    assert g["best_mad15"] < 0.5 and g["p_best_mad15"] < 0.01
    assert g["obs_b"] and g["p_b"] < 0.05
    assert g["obs_c"] and g["p_c"] < 0.05
    g0 = p113.permutation_guard(T, tgt, n_perm=200, seed=1)
    assert g0["p_best_mad15"] > 0.05 and not g0["obs_b"]


def test_verdict_rule_on_planted_and_random_grids():
    rng = np.random.default_rng(8)
    tgt = np.array([84.0, 74.7, 99.42, 68.64, 105.22, 76.38, 74.58, 80.18, 68.81, 66.44, 73.02, 72.74, 70.97, 66.54, 74.43])
    specs = pd.DataFrame({"spec": [f"s{i}" for i in range(400)], "order": np.arange(400)})
    T = _fake_grid(rng)
    planted = T.copy()
    planted[5] = tgt + 0.2
    v = p113.verdict(planted, tgt, specs, p113.permutation_guard(planted, tgt, 200, 1), p113.loto(planted, tgt, specs))
    assert v["verdict"] == "reproduced" and v["best_spec"] == "s5"
    v0 = p113.verdict(T, tgt, specs, p113.permutation_guard(T, tgt, 200, 1), p113.loto(T, tgt, specs))
    assert v0["verdict"] == "not reproduced"


def test_ranking_tie_breaks():
    tgt = np.full(15, 70.0)
    T = np.vstack([tgt + 1.0, tgt + np.r_[np.zeros(14), 15.0], tgt + 1.0, np.full(15, np.nan)])
    specs = pd.DataFrame({"spec": list("abcd"), "order": [2, 1, 0, 3]})
    m = p113.fit_metrics(T, tgt)
    assert m["MAD15"][3] == np.inf  # a missing cell ranks last
    order = p113.rank_specs(specs, m)
    # a and c tie on MAD15 = 1 and K2 = 15; b has MAD15 = 1 too but K2 = 14; c wins on the fixed order
    assert list(order) == [2, 0, 1, 3]


def test_holm_and_bootstrap_p():
    assert np.allclose(p113.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    d = np.r_[np.full(990, -1.0), np.full(9, 1.0)]
    assert np.isclose(p113.boot_p(d), 2 * 10 / 1000)
    assert p113.boot_p(np.full(999, -1.0)) == 2 / 1000
