"""Fast unit tests for experiments/connectedness_gamma.py (SYNTHETIC data only, no real files)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from niftygex import replication as rp
from niftygex.connectedness import VarFit, connectedness_from_fit

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "connectedness_gamma.py"
_spec = importlib.util.spec_from_file_location("connectedness_gamma", _PATH)
cg = importlib.util.module_from_spec(_spec)
sys.modules["connectedness_gamma"] = cg  # dataclasses need the module registered
_spec.loader.exec_module(cg)


# ---------------------------------------------------------------- simulation helpers


def markov_labels(n, stay, rng):
    """Persistent 0/1 labels: each day keeps yesterday's label with probability ``stay``."""
    x = np.empty(n, int)
    x[0] = rng.integers(0, 2)
    for t in range(1, n):
        x[t] = x[t - 1] if rng.random() < stay else 1 - x[t - 1]
    return x


def simulate_days(A_days, chol_days, M, rng, burn=30):
    """One bivariate VAR(1) path per day (days x M x 2); day d uses A_days[d] and shock factor chol_days[d]."""
    D = len(A_days)
    y = np.zeros((D, 2))
    out = np.empty((D, M, 2))
    for t in range(burn + M):
        e = np.einsum("dij,dj->di", chol_days, rng.standard_normal((D, 2)))
        y = np.einsum("dij,dj->di", A_days, y) + e
        if t >= burn:
            out[:, t - burn] = y
    return out + 3.0  # a non-zero level, as for volatility


def true_total(A1, Sigma, H=10, p=2):
    A = np.zeros((p, 2, 2))
    A[0] = A1
    return connectedness_from_fit(VarFit(A=A, c=np.zeros(2), Sigma=Sigma, resid=np.empty((0, 2)), p=p), H).paper_scale_total


# ---------------------------------------------------------------- fast pooled VAR


def test_fast_pooled_var_equals_replication_segments():
    rng = np.random.default_rng(1)
    Y = np.cumsum(rng.normal(size=(15, 90, 2)), axis=1) * 0.1 + 5
    Y[3, 10] = np.nan  # a missing minute in both series
    Y[6, 40, 1] = np.nan  # a missing minute in one series
    for p, H in ((2, 10), (4, 30)):
        ds = cg.day_stats(Y, p)
        pick = [0, 3, 3, 6, 9]  # day 3 twice, as in a bootstrap draw
        w = np.bincount(pick, minlength=15).astype(float)
        fast = cg.conn_values(ds, ds.sums(w)[0], H)
        Ys, segs = [], []
        for n, d in enumerate(pick):
            ok = np.all(np.isfinite(Y[d]), axis=1)
            Ys.append(Y[d][ok])
            segs.append(n * 1_000_000 + np.cumsum(~ok)[ok])
        ref = rp.connectedness_segments(np.vstack(Ys), np.concatenate(segs), p, H)
        assert np.allclose(fast, [ref.paper_scale_total, ref.table[0, 1], ref.table[1, 0]], atol=1e-8)


def test_connectedness_does_not_change_when_a_series_is_rescaled():
    rng = np.random.default_rng(2)
    Y = np.cumsum(rng.normal(size=(6, 80, 2)), axis=1) * 0.1 + 5
    a = cg.conn_values(cg.day_stats(Y, 2), cg.day_stats(Y, 2).sums(np.ones(6))[0], 10)
    Z = Y * np.array([1e-4, 3e-4])
    b = cg.conn_values(cg.day_stats(Z, 2), cg.day_stats(Z, 2).sums(np.ones(6))[0], 10)
    assert np.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------- labels, permutations, bootstrap


def test_gamma_state_comes_from_previous_trading_day_of_the_panel():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    panel = pd.DataFrame({"gross_gex": [10.0, 10, 10, 10, 10], "net_gex_participant": [1.0, -2, 3, -4, 5],
                          "net_gex_us_convention": [1.0, 1, 1, 1, 1], "net_gex_client_mirror": [0.0, 0, 0, 0, 0]}, index=idx)
    days = idx[[1, 3, 4]]  # 3 January missing (for example no Bank Nifty bars)
    st = cg.gamma_states(panel, days)
    assert np.allclose(st["share_pro"], [0.1, 0.3, -0.4])  # 4 Jan gets the 3 Jan close, not the 2 Jan close
    assert list(st["publish_date_used"]) == list(idx[[0, 2, 3]])


def test_block_permutation_keeps_groups_and_blocks():
    rng = np.random.default_rng(3)
    groups = np.r_[np.zeros(47, int), np.ones(23, int), np.full(30, 2)]
    for block in (1, 5, 20):
        perm = cg.block_permutation(groups, block, rng)
        assert sorted(perm) == list(range(len(groups)))
        assert np.array_equal(groups[perm], groups)  # labels never leave their regime
    perm = cg.block_permutation(groups, 10, rng)
    steps = np.diff(perm)
    # inside a block positions run consecutively; only block edges break the run
    assert np.mean(steps == 1) > 0.8


def test_block_bootstrap_weights_keep_group_sizes():
    rng = np.random.default_rng(4)
    groups = np.r_[np.zeros(47, int), np.ones(23, int)]
    w = cg.block_bootstrap_weights(groups, 20, rng)
    assert w[:47].sum() == 47 and w[47:].sum() == 23
    assert np.all(w >= 0) and np.allclose(w, np.round(w))


def test_holm_matches_statsmodels():
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.2, 0.012])
    assert np.allclose(cg.holm(p), multipletests(p, method="holm")[1])
    q = cg.holm([0.01, np.nan, 0.02])
    assert np.isnan(q[1]) and np.allclose(q[[0, 2]], [0.02, 0.02])


def test_centered_perm_p():
    draws = np.r_[np.linspace(-1, 1, 99), np.nan]
    p, m = cg.centered_perm_p(5.0, draws)
    assert abs(m) < 1e-12 and np.isclose(p, 1 / 100)
    p0, _ = cg.centered_perm_p(0.0, draws)
    assert p0 == 1.0


def test_contrast_drops_small_strata_and_reweights():
    strata = np.r_[np.zeros(20, int), np.ones(20, int)]
    lab = np.r_[np.tile([0, 1], 10), np.ones(18, int), np.zeros(2, int)]  # stratum 1 has only 2 long days
    c = cg.Contrast("x", "pro", np.ones(40, bool), strata, "main", 10).setup({"pro": lab}, min_days=5)
    assert c.included == [0] and np.allclose(c.weights, [1.0])
    assert c.n_a == {0: 10, 1: 18} and c.n_b == {0: 10, 1: 2}


# ---------------------------------------------------------------- two regimes with known spillovers


def test_two_var_regimes_with_different_spillovers_are_told_apart():
    rng = np.random.default_rng(5)
    D, M = 120, 150
    lab = markov_labels(D, 0.9, rng)
    A_hi, S_hi = np.array([[0.5, 0.3], [0.3, 0.5]]), np.array([[1.0, 0.6], [0.6, 1.0]])
    A_lo, S_lo = np.array([[0.7, 0.0], [0.0, 0.7]]), np.array([[1.0, 0.1], [0.1, 1.0]])
    A_days = np.where(lab[:, None, None] == 1, A_hi, A_lo)
    L_days = np.where(lab[:, None, None] == 1, np.linalg.cholesky(S_hi), np.linalg.cholesky(S_lo))
    Y = simulate_days(A_days, L_days, M, rng)
    stats_ = {"main": cg.day_stats(Y, 2)}
    groups = np.zeros(D, int)
    c = cg.Contrast("t", "pro", np.ones(D, bool), groups, "main", 10).setup({"pro": lab})
    diff, vals = c.evaluate(stats_, {"pro": lab}, np.ones(D))
    t_hi, t_lo = true_total(A_hi, S_hi), true_total(A_lo, S_lo)
    assert abs(vals[0, 0, 0] - t_hi) < 4 and abs(vals[0, 1, 0] - t_lo) < 4
    assert abs(diff[0] - (t_hi - t_lo)) < 5 and t_hi - t_lo > 50
    draws = np.array([c.evaluate(stats_, {"pro": lab[cg.block_permutation(groups, 10, np.random.default_rng([9, i]))]}, np.ones(D))[0][0]
                      for i in range(199)])
    p, _ = cg.centered_perm_p(diff[0], draws)
    assert p <= 0.01
    boots = np.array([c.evaluate(stats_, {"pro": lab}, cg.block_bootstrap_weights(groups, 10, np.random.default_rng([8, i])))[0][0]
                      for i in range(100)])
    assert np.percentile(boots, 2.5) > 0


def test_permutation_p_roughly_uniform_under_the_null():
    """Labels persistent and independent of the data; day-level spillover also persistent across days."""
    rng = np.random.default_rng(6)
    D, M, reps, n_perm = 80, 60, 150, 199
    groups = np.r_[np.zeros(40, int), np.ones(40, int)]
    pvals = []
    for r in range(reps):
        z = np.empty(D)
        z[0] = rng.standard_normal()
        for t in range(1, D):
            z[t] = 0.9 * z[t - 1] + np.sqrt(1 - 0.81) * rng.standard_normal()
        cross = np.clip(0.15 + 0.1 * z, 0.0, 0.3)
        rho = np.clip(0.4 + 0.25 * z, -0.2, 0.9)
        A_days = np.stack([np.array([[0.6, c], [c, 0.6]]) for c in cross])
        L_days = np.stack([np.linalg.cholesky(np.array([[1.0, q], [q, 1.0]])) for q in rho])
        Y = simulate_days(A_days, L_days, M, rng)
        lab = markov_labels(D, 0.9, rng)
        stats_ = {"main": cg.day_stats(Y, 2)}
        c = cg.Contrast("n", "pro", np.ones(D, bool), groups, "main", 10).setup({"pro": lab})
        if not c.included:
            continue
        real = c.evaluate(stats_, {"pro": lab}, np.ones(D))[0][0]
        prng = np.random.default_rng([7, r])
        draws = np.array([c.evaluate(stats_, {"pro": lab[cg.block_permutation(groups, 10, prng)]}, np.ones(D))[0][0]
                          for _ in range(n_perm)])
        pvals.append(cg.centered_perm_p(real, draws)[0])
    pvals = np.array(pvals)
    assert len(pvals) >= 120
    assert stats.kstest(pvals, "uniform").pvalue > 0.001
    assert 0.02 <= np.mean(pvals <= 0.10) <= 0.22
    assert 0.0 <= np.mean(pvals <= 0.05) <= 0.14
