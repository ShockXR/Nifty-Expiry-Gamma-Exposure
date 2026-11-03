"""Tests for P114 (experiments/p114_persistence_fixed_model.py and the persistence helpers at the end of
niftygex.volatility). SYNTHETIC data only (no raw data file is read).

Covers: the impulse response of GARCH(1,1) and its half-life ln(0.5) / ln(alpha + beta); the impulse response of
the extended recursion against a plain deterministic loop and against a Monte Carlo of the full model (the night
step and the daily restart of the running mean included); the half-life interpolation rules; the beta solver; the
profile likelihood (equal to the replication's when the extensions are off, equal to the unconstrained fit at the
fitted value); the bootstrap reproduces the replication's draws exactly; the daily realized variance leaves out the
overnight gap; the drift measure; the decision rule and the bootstrap p; and an end-to-end dry run of every stage.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex import replication as rp
from niftygex import volatility as vol

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p114_persistence_fixed_model.py"
_spec = importlib.util.spec_from_file_location("p114_persistence_fixed_model", _PATH)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def simulate_ext(T, M, om, al, be, gx, de, nu=7.0, seed=0, z=None):
    """Simulate normalised residuals x from the extended recursion (no leverage) with a plain loop."""
    rng = np.random.default_rng(seed)
    qbar = om / (1 - al - be - gx)
    if z is None:
        z = rng.standard_t(nu, size=(T, M)) / np.sqrt(nu / (nu - 2))
    x, q = np.zeros((T, M)), np.zeros((T, M))
    qq = qbar
    for t in range(T):
        run = 0.0
        for i in range(M):
            q[t, i] = qq
            x[t, i] = np.sqrt(qq) * z[t, i]
            run += x[t, i] ** 2
            qq = om + al * x[t, i] ** 2 + gx * run / (i + 1) + be * qq
        qq = qbar + de * (qq - qbar)
    return x, q


def expected_path(T, M, om, al, be, gx, de, shock=None):
    """The deterministic expected path of q (every x^2 replaced by q), optionally with one extra unit of x^2 at
    (day 0, minute shock)."""
    qbar = om / (1 - al - be - gx)
    q = np.zeros((T, M))
    qq = qbar
    for t in range(T):
        run = 0.0
        for i in range(M):
            q[t, i] = qq
            x2 = qq + (1.0 if (shock is not None and t == 0 and i == shock) else 0.0)
            run += x2
            qq = om + al * x2 + gx * run / (i + 1) + be * qq
        qq = qbar + de * (qq - qbar)
    return q


# ---------------------------------------------------------------- impulse response and half-life


@pytest.mark.parametrize("a,b", [(0.03, 0.96), (0.0246, 0.9735), (0.0498, 0.9407), (0.2, 0.5), (0.01, 0.98989)])
def test_garch11_impulse_response_and_half_life(a, b):
    r = vol.ext_impulse_response({"alpha": a, "beta": b}, 360, 180, n_steps=800)
    assert np.allclose(r, a * (a + b) ** np.arange(800), rtol=1e-12, atol=0)
    s = vol.ext_persistence_summary({"alpha": a, "beta": b}, 360, 180)
    assert s["hl_irf"] == pytest.approx(np.log(0.5) / np.log(a + b), rel=1e-12)
    assert s["hl_ab"] == pytest.approx(np.log(0.5) / np.log(a + b), rel=1e-12)
    assert s["rho_irf"] == pytest.approx(a + b, rel=1e-12)
    # the shock minute does not matter for GARCH(1,1), and neither does the replication's half_life formula
    for m in (0, 60, 359):
        assert vol.ext_persistence_summary({"alpha": a, "beta": b}, 360, m)["hl_irf"] == pytest.approx(
            float(rp.half_life(a + b)), rel=1e-12)


def test_extended_impulse_response_equals_deterministic_loop():
    M, T = 40, 5
    om, al, be, gx, de = 0.01, 0.05, 0.88, 0.04, 0.3
    for shock in (0, 17, M - 1):
        q0 = expected_path(T, M, om, al, be, gx, de)
        q1 = expected_path(T, M, om, al, be, gx, de, shock=shock)
        diff = (q1 - q0).ravel()[shock + 1:]
        r = vol.ext_impulse_response({"alpha": al, "beta": be, "gam_x": gx, "delta": de}, M, shock, n_steps=len(diff))
        assert np.allclose(r, diff, rtol=1e-9, atol=1e-15)


def test_extended_impulse_response_equals_monte_carlo():
    """The model is linear in (q, x^2), so the mean over paths of q(shocked) - q(baseline), with the same random
    draws, is the impulse response (SYNTHETIC)."""
    M, T, n = 30, 3, 4000
    om, al, be, gx, de, nu = 0.02, 0.06, 0.85, 0.05, 0.4, 8.0
    shock = 12
    rng = np.random.default_rng(7)
    acc = np.zeros(T * M)
    for _ in range(n):
        z = rng.standard_t(nu, size=(T, M)) / np.sqrt(nu / (nu - 2))
        _, qa = simulate_ext(T, M, om, al, be, gx, de, z=z)
        z2 = z.copy()
        # one extra unit of x^2 at the shock minute: x^2 = q z^2 + 1
        xa2 = qa[0, shock] * z[0, shock] ** 2
        z2[0, shock] = np.sqrt((xa2 + 1.0) / qa[0, shock]) * np.sign(z[0, shock])
        _, qb = simulate_ext(T, M, om, al, be, gx, de, z=z2)
        acc += (qb - qa).ravel()
    mc = (acc / n)[shock + 1:]
    r = vol.ext_impulse_response({"alpha": al, "beta": be, "gam_x": gx, "delta": de}, M, shock, n_steps=len(mc))
    assert np.max(np.abs(mc - r)) < 0.03 * r[0]


def test_impulse_response_night_and_running_mean():
    M = 360
    p = {"alpha": 0.035, "beta": 0.923, "gam_x": 0.018, "delta": 0.09}
    r = vol.ext_impulse_response(p, M, 180, n_steps=400)
    k_close, k_open = M - 1 - 180, M - 180            # steps to the last minute and to the next open
    assert r[k_open - 1] == pytest.approx(0.09 * (0.035 * r[k_close - 1] + 0.923 * r[k_close - 1]
                                                  + 0.018 * (1 + r[:k_close].sum()) / M), rel=1e-12)
    # delta = 0: nothing survives the night, and the half-life is at most the steps to the close
    r0 = vol.ext_impulse_response({**p, "delta": 0.0}, M, 180, n_steps=400)
    assert r0[k_open - 1] == 0.0
    s = vol.ext_persistence_summary({"alpha": 0.01, "beta": 0.98989, "delta": 0.0}, M, 180)
    assert s["hl_irf"] == pytest.approx(k_close - 1) and s["share_next_open"] == 0.0
    # with today's running mean the response decays more slowly than (alpha + beta) ** k
    s = vol.ext_persistence_summary(p, M, 180)
    assert s["hl_irf"] > s["hl_ab"]


def test_half_life_rules():
    assert vol.irf_half_life([1.0, 0.8, 0.4]) == pytest.approx(1 + np.log(0.8 / 0.5) / np.log(0.8 / 0.4))
    assert vol.irf_half_life([1.0, 0.9, 0.0]) == 1.0
    assert vol.irf_half_life([1.0, 0.9, 0.8]) == np.inf
    assert np.isnan(vol.irf_half_life([0.0, 0.0]))
    assert vol.equivalent_persistence(np.inf) == 1.0
    assert vol.equivalent_persistence(np.log(0.5) / np.log(0.97)) == pytest.approx(0.97)


def test_solve_beta_for_half_life():
    for hl in (8.0, 40.0, 150.0):
        b = vol.solve_beta_for_half_life(0.02, 0.005, 0.3, 360, 180, hl)
        s = vol.ext_persistence_summary({"alpha": 0.02, "beta": b, "gam_x": 0.005, "delta": 0.3}, 360, 180)
        assert s["hl_irf"] == pytest.approx(hl, rel=1e-8)
    # alpha + gam_x = 0.06 leaves beta at most 0.94, a half-life of about 51 minutes
    assert vol.solve_beta_for_half_life(0.04, 0.02, 0.3, 360, 180, 150.0) is None
    # with delta = 0 a shock at 12:15 cannot have a half-life beyond the close
    assert vol.solve_beta_for_half_life(0.04, 0.0, 0.0, 360, 180, 400.0) is None


# ---------------------------------------------------------------- profile likelihood


@pytest.fixture(scope="module")
def sim_grid():
    x, _ = simulate_ext(40, 120, 0.02, 0.04, 0.92, 0.02, 0.3, seed=3)
    return x


def test_profile_without_extensions_equals_replication_profile(sim_grid):
    x = sim_grid
    g = vol.fit_garch11(x.ravel())
    for hl in (15.0, 60.0):
        a = rp.profile_half_life(x.ravel(), g, [hl])["loglik"].iloc[0]
        b = vol.fit_garch_ext_fixed(x, None, False, False, "ab", 0.5 ** (1 / hl), start=g)[0]
        c = vol.fit_garch_ext_fixed(x, None, False, False, "irf", hl, M=x.shape[1], shock_minute=60, start=g)[0]
        assert b == pytest.approx(a, abs=1e-3)
        assert c == pytest.approx(a, abs=1e-3)


def test_profile_at_fitted_value_equals_unconstrained_fit(sim_grid):
    x = sim_grid
    M = x.shape[1]
    f = vol.fit_garch_ext(x, x < 0, None, overnight_decay=True, garchx_today=True)
    s = vol.ext_persistence_summary(f, M, 60)
    ll_irf, p, ok = vol.fit_garch_ext_fixed(x, None, True, True, "irf", s["hl_irf"], M=M, shock_minute=60, start=f)
    ll_ab, _, _ = vol.fit_garch_ext_fixed(x, None, True, True, "ab", f["alpha"] + f["beta"], start=f)
    assert ll_irf == pytest.approx(f["loglik"], abs=0.05) and ll_ab == pytest.approx(f["loglik"], abs=0.05)
    assert vol.ext_persistence_summary(p, M, 60)["hl_irf"] == pytest.approx(s["hl_irf"], rel=1e-6)
    far = vol.fit_garch_ext_fixed(x, None, True, True, "irf", 5 * s["hl_irf"], M=M, shock_minute=60, start=f)[0]
    assert far < f["loglik"] - 1


def test_joint_lr_without_extensions_matches_replication_worker(sim_grid):
    x = sim_grid
    xa, _ = simulate_ext(40, 120, 0.05, 0.08, 0.85, 0.0, 1.0, seed=4)
    fb, fa = vol.fit_garch11(x.ravel()), vol.fit_garch11(xa.ravel())
    w = np.ones(x.size)
    ref = ex.rep._joint_lr_worker((x.ravel(), w, fb, xa.ravel(), w, fa))
    for p in (fb, fa):
        p.update(gam_x=0.0, delta=1.0, gam_l=0.0)
    mine = ex.joint_lr(x, np.ones_like(x), fb, xa, np.ones_like(xa), fa, (False, False), "ab", x.shape[1], shock=60)
    assert mine["lr_stat"] == pytest.approx(ref["lr_stat"], abs=1e-3)


# ---------------------------------------------------------------- bootstrap, data, measures, rules


def test_bootstrap_reference_draws_equal_replication_worker(tmp_path):
    rng = np.random.default_rng(11)
    T, M = 30, 200
    x, _ = simulate_ext(T, M, 0.02, 0.05, 0.9, 0.0, 1.0, seed=5)
    h = np.exp(0.3 * rng.standard_normal(T)) * 1e-6
    s = 1 + np.linspace(1, 0, M)
    eps = x * np.sqrt(h[:, None] * s[None, :])
    valid = np.ones((T, M), bool)
    valid[0, 0] = False
    eps[0, 0] = 0.0
    key = ("before_inferred", "nifty")
    store = {key: {"valid": valid, "M": M, "ref": {"eps": eps, "h": h, "params": {}},
                   "F": {"eps": eps, "h": h * 1.1, "params": {}}}}
    import pickle
    pickle.dump(store, open(tmp_path / "fits.pkl", "wb"))
    ex.boot_key((key[0], key[1], str(tmp_path), 8, ["ref", "F"]))
    d = pd.read_csv(tmp_path / f"boot_{key[0]}_{key[1]}.csv", float_precision="round_trip")
    mine = d[d["model"] == "ref"].sort_values("draw")
    theirs = np.vstack([ex.rep._boot_persistence_worker((eps, h, valid, ex.BLOCK_DAYS, 2, 1000 * ex.SEED_INDEX[key] + c))
                        for c in range(ex.BOOT_CHUNKS)])
    assert np.array_equal(mine["alpha"].to_numpy(), theirs[:, 0])
    assert np.array_equal(mine["beta"].to_numpy(), theirs[:, 1])
    assert np.array_equal(mine["ab"].to_numpy(), theirs[:, 0] + theirs[:, 1])
    assert set(d["model"]) == {"ref", "F"} and len(d) == 16


def test_daily_rv_leaves_out_the_overnight_gap():
    C = np.array([[101.0, 102.0, 103.0], [110.0, 111.0, np.nan]])
    O = np.array([[100.0, 101.0, 102.0], [109.0, 110.0, np.nan]])
    G = {"x": {"close": C, "open": O, "r": rp.continuous_log_returns(C, overnight=True)}}
    rv = ex.daily_rv(G, "x")
    assert rv[0] == pytest.approx(np.log(1.01) ** 2 + np.log(102 / 101) ** 2 + np.log(103 / 102) ** 2)
    assert rv[1] == pytest.approx(np.log(110 / 109) ** 2 + np.log(111 / 110) ** 2)


def test_drift_measure():
    m = np.array([1.0, 2.0, 0.5, 1.5])
    x = np.sqrt(m)[:, None] * np.ones((4, 10))
    d = ex.drift_measures(x, np.ones((4, 10), bool))
    assert d["drift_sd_log_daily_mean_x2"] == pytest.approx(np.std(np.log(m), ddof=1))
    assert d["drift_ac1_log_daily_mean_x2"] == pytest.approx(np.corrcoef(np.log(m)[1:], np.log(m)[:-1])[0, 1])


def test_decision_rule_and_bootstrap_p():
    assert ex.verdict_rule(-0.01, -0.02, -0.001) == "survives"
    assert ex.verdict_rule(-0.01, -0.02, 0.001) == "weakened"
    assert ex.verdict_rule(0.001, -0.02, 0.01) == "gone"
    assert ex.verdict_rule(0.01, 0.001, 0.02) == "reversed"
    assert ex.size_label(0.7).startswith("keeps most") and ex.size_label(0.2).startswith("keeps less")
    assert ex.size_label(-0.1) == "keeps none of the fall"
    assert ex.boot_p_fell(np.array([-1.0] * 99 + [0.0])) == pytest.approx(2 / 101)


# ---------------------------------------------------------------- end to end


def test_end_to_end_synthetic(tmp_path, monkeypatch):
    monkeypatch.setitem(ex.SYNTH, "n_boot", 4)
    work, out = tmp_path / "work", tmp_path / "out"
    for stage in ("fit", "boot", "lr", "rolling"):
        ex.main([stage, "--synthetic", "--work", str(work)])
    ex.main(["analyze", "--synthetic", "--work", str(work), "--out", str(out)])
    t = pd.read_csv(out / "tests.csv")
    assert list(t.columns) == ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw",
                               "sidedness", "p_method", "p_adjusted", "adjust_method"]
    assert (t["role"] == "primary").sum() == 1
    assert {"secondary", "robustness", "exploratory"} <= set(t["role"])
    v = pd.read_csv(out / "verdict.csv")
    assert v.loc[v["role"] == "primary", "verdict"].iloc[0] in {"survives", "weakened", "gone", "reversed"}
    f = pd.read_csv(out / "fits.csv")
    assert f["data"].str.startswith("SYNTHETIC").all()
    # the reference's impulse-response half-life is its ln(0.5) / ln(alpha + beta)
    r = f[f["model"] == "ref"]
    assert np.allclose(r["hl_irf"], np.log(0.5) / np.log(r["alpha"] + r["beta"]), rtol=1e-10)
    for name in ("persistence.csv", "persistence_change.csv", "irf_curves.csv", "rolling.csv", "lr.csv",
                 "fig_irf.png", "run_info.json"):
        assert (out / name).exists()
