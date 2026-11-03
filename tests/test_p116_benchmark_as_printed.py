"""Tests for P116 (experiments/p116_benchmark_as_printed.py, the P116 switches of the P109 driver and the
ARMA(1,1) mean-square forecast in niftygex.volatility). SYNTHETIC data only (no raw data file is read).

Covers: the conditional mean of the ARMA(1,1) intraday mean against its definition; the expected squared
conditional mean over a window against closed forms; the full forecast (variance part plus mean part) against a
Monte Carlo of the model; P109's variants are unchanged by the new switches; the registered 2 x 2 x 2 factorial of
switches; the driver's mcs_ours and mcs_cpp equal direct fits of McsGarch; the Shapley split of the change; the
change in the hybrid's gain and its interval; the registered decision rule; and an end-to-end dry run of run,
check and analyze on a simulated market.
"""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex import volatility as vol
from niftygex.simulate import SimConfig, simulate_market

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p116_benchmark_as_printed", _ROOT / "experiments" / "p116_benchmark_as_printed.py")
p116 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p116)
ex = p116.ex


def _fresh_p109():
    spec = importlib.util.spec_from_file_location("p109_fresh_for_p116_test",
                                                  _ROOT / "experiments" / "p109_mcsgarch_extensions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def market():
    return simulate_market(SimConfig(n_days=100, kappa=0.0, seed=8, start="2024-08-01"))


def _toy_model(c=0.2, phi=0.5, theta=0.3, omega=0.05, alpha=0.1, beta=0.85, nu=7.0):
    m = vol.McsGarch(mean="arma11")
    m.mean_params = {"c": c, "phi": phi, "theta": theta}
    m.params = {"omega": omega, "alpha": alpha, "beta": beta, "nu": nu}
    return m


# ---------------------------------------------------------------- the ARMA(1,1) mean part of the forecast


def test_conditional_mean_matches_definition():
    rng = np.random.default_rng(1)
    m = _toy_model()
    R = rng.standard_normal((3, 7))
    mu = vol.arma11_conditional_mean(m, R)
    e = m._residuals(R).ravel()
    r = R.ravel()
    c, phi, th = 0.2, 0.5, 0.3
    ref = np.empty_like(r)
    ref[0] = c + phi * c / (1 - phi)
    for n in range(1, len(r)):
        ref[n] = c + phi * r[n - 1] + th * e[n - 1]
    assert np.allclose(mu.ravel(), ref, rtol=0, atol=1e-14)
    assert np.allclose(mu.ravel(), r - e, atol=1e-12)        # r = mu + e
    # the conditional mean of minute i uses nothing from minute i onwards
    R2 = R.copy()
    R2[1, 4:] += 5.0
    assert np.array_equal(vol.arma11_conditional_mean(m, R2)[1, :5], mu[1, :5])
    z = vol.McsGarch(mean="zero")
    assert np.array_equal(vol.arma11_conditional_mean(z, R), np.zeros_like(R))


def test_mean_square_term_closed_forms():
    rng = np.random.default_rng(2)
    T, M = 2, 12
    R = rng.standard_normal((T, M))
    S = 0.5 + rng.random((T, M))
    filt = {"S": S, "q2": 0.5 + rng.random((T, M))}
    h = np.array([1.0, 1.3])
    at, ai, hz = np.array([1, 1, 1]), np.array([0, 3, 9]), np.array([5, 6, 6])   # the last window is cut at the close
    n = np.minimum(ai + hz, M) - ai
    # a zero mean adds nothing, so mcs_ours's forecasts are untouched
    assert np.array_equal(vol.arma11_mean_square_forecast(vol.McsGarch(mean="zero"), R, filt, h, at, ai, hz), np.zeros(3))
    # phi = theta = 0: the mean is c at every minute
    m = _toy_model(c=0.3, phi=0.0, theta=0.0)
    assert np.allclose(vol.arma11_mean_square_forecast(m, R, filt, h, at, ai, hz), n * 0.09, rtol=1e-12)
    # theta = -phi: the shocks do not feed the mean (V = 0); the mean decays geometrically from mu_i toward c / (1 - phi)
    m = _toy_model(c=0.1, phi=0.6, theta=-0.6)
    mu = vol.arma11_conditional_mean(m, R)
    got = vol.arma11_mean_square_forecast(m, R, filt, h, at, ai, hz)
    for k in range(3):
        mm, tot = mu[at[k], ai[k]], 0.0
        for _ in range(n[k]):
            tot += mm * mm
            mm = 0.1 + 0.6 * mm
        assert got[k] == pytest.approx(tot, rel=1e-12)
    # the variance part of the recursion: with c = phi = 0 and theta != 0, only V_j = theta^2 sigma2_{j-1} remains
    # after the first minute, and sigma2 follows forecast_variance's own expected path
    m = _toy_model(c=0.0, phi=0.0, theta=0.4)
    mu = vol.arma11_conditional_mean(m, R)
    got = vol.arma11_mean_square_forecast(m, R, filt, h, at, ai, hz)
    pers = m.params["alpha"] + m.params["beta"]
    qbar = m.params["omega"] / (1 - pers)
    for k in range(3):
        t, i = at[k], ai[k]
        sig2 = [h[t] * S[t, i + j] * (qbar + pers ** j * (filt["q2"][t, i] - qbar)) for j in range(n[k])]
        assert got[k] == pytest.approx(mu[t, i] ** 2 + 0.16 * sum(sig2[:-1]), rel=1e-12)


def test_full_forecast_matches_monte_carlo():
    rng = np.random.default_rng(3)
    T, M = 2, 12
    m = _toy_model()
    c, phi, th = 0.2, 0.5, 0.3
    om, al, be, nu = 0.05, 0.1, 0.85, 7.0
    R = 0.5 * rng.standard_normal((T, M))
    S = 0.5 + rng.random((T, M))
    q2 = 0.5 + rng.random((T, M))
    filt = {"S": S, "q2": q2}
    h = np.array([1.0, 1.3])
    t, i, hn = 1, 3, 6
    f_var = m.forecast_variance(filt, h, [t], [i], hn)[0]
    f_mean = vol.arma11_mean_square_forecast(m, R, filt, h, [t], [i], hn)[0]
    e = m._residuals(R)
    n_paths = 400_000
    r_p = np.full(n_paths, R[t, i - 1])
    e_p = np.full(n_paths, e[t, i - 1])
    q = np.full(n_paths, q2[t, i])
    tot = np.zeros(n_paths)
    for j in range(i, i + hn):
        mu = c + phi * r_p + th * e_p
        z = rng.standard_t(nu, n_paths) / np.sqrt(nu / (nu - 2))
        eps = np.sqrt(h[t] * S[t, j] * q) * z
        r = mu + eps
        tot += r * r
        q = om + al * eps * eps / (h[t] * S[t, j]) + be * q
        r_p, e_p = r, eps
    se = tot.std() / np.sqrt(n_paths)
    assert abs(tot.mean() - (f_var + f_mean)) < 4 * se
    assert f_mean > 20 * se                       # the mean part is far from negligible here, so the test has power
    assert abs(tot.mean() - f_var) > 10 * se


# ---------------------------------------------------------------- the switches and the factorial


def test_p109_variants_unchanged_by_the_switches():
    fresh = _fresh_p109()
    assert set(fresh.SPECS) == set(fresh.NIFTY_VARIANTS)          # P116's variants are not in P109's own copy
    assert not any(k in fresh.SPECS[v] for v in fresh.SPECS for k in fresh.SPEC_DEFAULTS)
    assert fresh.SPEC_DEFAULTS == {"expiry_diurnal": True, "smooth": 5, "mean": "zero", "mean_sq": False}
    assert len({fresh.fit_key(v) for v in fresh.NIFTY_VARIANTS}) == 10
    assert len({fresh.fit_key(v) for v in fresh.OTHER_VARIANTS}) == 6
    for v in fresh.NIFTY_VARIANTS:
        assert fresh.fit_key(v)[4:] == (True, 5, "zero")
    assert ex.EXPIRY_DIURNAL is True and ex.DIURNAL_SMOOTH == 5


def _switches(v):
    return (ex.spec_value(v, "expiry_diurnal"), ex.spec_value(v, "smooth"), ex.spec_value(v, "mean"))


def test_factorial_specs():
    ours = {True: 1, False: 0}
    for v in p116.VARIANTS:
        name = p116.RENAME.get(v, v)
        if name == "mcs_cpp_varonly":
            continue
        e, s, mn = _switches(v)
        assert p116.FACTORIAL[name] == (ours[e], int(s == 5), int(mn == "zero"))
        assert ex.spec_value(v, "mean_sq") == (mn == "arma11")     # every ARMA model forecasts with its mean part
        assert {k: ex.SPECS[v][k] for k in ("daily", "decay", "lev", "todayrv", "cal")} == p116._BASE
    assert sorted(p116.FACTORIAL.values()) == sorted(itertools.product((0, 1), repeat=3))
    assert p116.FACTORIAL["mcs_ours"] == (1, 1, 1) and p116.FACTORIAL["mcs_cpp"] == (0, 0, 0)
    for v in p116.FROM_OURS:
        assert sum(a != b for a, b in zip(p116.FACTORIAL[v], (1, 1, 1))) == 1
    for v in p116.FROM_CPP:
        assert sum(a != b for a, b in zip(p116.FACTORIAL[v], (0, 0, 0))) == 1
    assert _switches("mcs_cpp") == (False, 1, "arma11")
    d = {k for k in ex.SPECS["mcs_cpp"] if ex.SPECS["mcs_cpp"][k] != ex.SPECS["mcs_cpp_varonly"][k]}
    assert d == {"mean_sq"}
    assert ex.fit_key("mcs_cpp") == ex.fit_key("mcs_cpp_varonly")
    assert len({ex.fit_key(v) for v in p116.VARIANTS}) == 8


def test_driver_equals_direct_fits(market, tmp_path):
    R, is_exp = market["R"], market["labels"]["is_expiry"].to_numpy()
    T, M = R.shape
    ex._STATE.update(R=R, is_exp=is_exp, days=market["days"], block=25, variants=p116.VARIANTS,
                     targets={"next30": (30, 45, None)})
    start = 60
    ex.run_block(("SYNTH", start, str(tmp_path)))
    got = ex.assemble(tmp_path, "SYNTH")
    h, _ = vol.daily_egarch_variance(R.sum(axis=1), train_size=start)
    # the same rows as the driver and run_walkforward (every day before the block's end), so that the vectorised
    # forecast sums run over identical arrays; then the test rows
    rows = ex.dataset_rows(T, M, start + 25, 30, 45)
    t, i, hz = rows["t"].to_numpy(), rows["minute"].to_numpy(), rows["horizon"].to_numpy()
    test = t >= start
    assert np.array_equal(t[test], got["t"].to_numpy()) and np.array_equal(i[test], got["minute"].to_numpy())
    for v, kw in (("mcsgarch", dict(expiry_diurnal=True, smooth=5)),
                  ("mcs_cpp", dict(mean="arma11", dist="t", expiry_diurnal=False, smooth=1))):
        mdl = vol.McsGarch(**kw).fit(R[:start], h[:start], is_exp[:start])
        filt = mdl.filter(R, h, is_exp)
        f = mdl.forecast_variance(filt, h, t, i, hz)[test]
        msq = vol.arma11_mean_square_forecast(mdl, R, filt, h, t, i, hz)[test]
        if v == "mcsgarch":
            assert np.array_equal(got[v].to_numpy(), f) and not msq.any()
        else:
            assert np.array_equal(got["mcs_cpp_varonly"].to_numpy(), f)
            assert np.array_equal(got[v].to_numpy(), f + msq)
            assert msq.min() > 0
            assert abs(mdl.mean_params["phi"]) < 1 and abs(mdl.mean_params["theta"]) < 1
    assert np.array_equal(got["y"].to_numpy(), ex.realized_target(R, got))
    P = pd.read_csv(tmp_path / f"SYNTH_block{start:04d}_params.csv")
    assert {"arma_c", "arma_phi", "arma_theta", "mean", "smooth", "expiry_diurnal"} <= set(P.columns)
    assert P.loc[P["variant"] == "mcsgarch", "mean"].isna().all()     # the committed benchmark's row is as in P109
    for v in p116.VARIANTS:                                            # every model differs from mcs_ours
        if v != "mcsgarch":
            assert not np.array_equal(got[v].to_numpy(), got["mcsgarch"].to_numpy()), v


# ---------------------------------------------------------------- statistics and the decision rule


def test_shapley_adds_up_and_matches_additive_case():
    rng = np.random.default_rng(4)
    for additive in (True, False):
        eff = rng.random(3)
        L = {}
        for name, lv in p116.FACTORIAL.items():
            base = 1.0 - eff @ np.array(lv)
            L[name] = base if additive else base - 0.05 * lv[0] * lv[1] + 0.03 * lv[1] * lv[2]
        sh = p116.shapley(L, L["mcs_cpp"])
        total = 100 * (L["mcs_cpp"] - L["mcs_ours"]) / L["mcs_cpp"]
        assert sum(sh.values()) == pytest.approx(total, abs=1e-12)
        if additive:
            for k, f in enumerate(p116.FACTORS):
                assert sh[f] == pytest.approx(100 * eff[k] / L["mcs_cpp"], abs=1e-12)


def test_delta_points_and_interval():
    rng = np.random.default_rng(5)
    n = 400
    cpp = 1 + rng.random(n)
    D = pd.DataFrame({"mcs_cpp": cpp, "mcs_ours": cpp - 0.03 + 0.01 * rng.standard_normal(n),
                      "hybrid": cpp - 0.10 + 0.02 * rng.standard_normal(n)})
    g_c = 100 * (D["mcs_cpp"].mean() - D["hybrid"].mean()) / D["mcs_cpp"].mean()
    g_o = 100 * (D["mcs_ours"].mean() - D["hybrid"].mean()) / D["mcs_ours"].mean()
    assert p116.delta_points(D) == pytest.approx(g_c - g_o)
    assert p116.gain_pct(D, "hybrid", "mcs_cpp") == pytest.approx(g_c)
    lo, hi = p116.delta_interval(D, reps=500)
    assert lo < p116.delta_points(D) < hi and lo > 0
    # same sign as L_cpp - L_ours (the H2 direction)
    assert np.sign(p116.delta_points(D)) == np.sign(D["mcs_cpp"].mean() - D["mcs_ours"].mean())


def test_decision_rule():
    ho = p116.horizon_outcome
    assert ho(0.01, 1.2, (0.6, 1.8)) == "headline understated"
    assert ho(0.01, -1.2, (-1.8, -0.6)) == "headline overstated"
    assert ho(0.01, 0.3, (0.1, 0.5)).startswith("headline understated (by less than the 0.5-point margin")
    assert ho(0.30, 0.3, (-0.2, 0.45)).startswith("no material difference (the 95% interval")
    assert ho(0.30, -0.3, (-0.9, 0.2)).startswith("no material difference (point estimate")
    assert ho(0.30, 0.8, (-0.1, 1.5)).startswith("inconclusive")
    assert ho(0.30, 0.5, (-0.1, 0.5)).startswith("no material difference")          # the margin is inclusive
    p_ok, p_no = {"next15": 0.001, "next30": 0.002}, {"next15": 0.4, "next30": 0.5}
    ci = {"next15": (-0.2, 0.3), "next30": (-0.3, 0.4)}
    v = p116.p116_verdict(p_ok, p_no, {"next15": 0.1, "next30": -0.1}, ci)
    assert v["verdict"] == "no material difference" and v["h1"].startswith("the hybrid beats mcs_cpp at every")
    v = p116.p116_verdict(p_ok, p_ok, {"next15": 1.0, "next30": 1.5}, ci)
    assert v["verdict"] == "headline understated"
    v = p116.p116_verdict({"next15": 0.001, "next30": 0.2}, {"next15": 0.01, "next30": 0.5},
                          {"next15": 1.0, "next30": 0.2}, ci)
    assert v["verdict"].startswith("mixed: headline understated at next 15 min; no material difference at next 30")
    assert v["h1"] == "the hybrid beats mcs_cpp at next 15 min only"
    v = p116.p116_verdict(p_no, p_no, {"next15": 0.1, "next30": 0.1}, ci)
    assert v["h1"] == "the hybrid does not beat mcs_cpp at any primary horizon"


# ---------------------------------------------------------------- end-to-end dry run


def test_end_to_end_dry_run(tmp_path, monkeypatch):
    monkeypatch.setitem(ex.SYNTH, "n_days", 110)
    monkeypatch.setitem(ex.SYNTH, "min_train", 60)
    monkeypatch.setitem(ex.SYNTH, "block", 25)
    work, out = tmp_path / "work", tmp_path / "out"
    p116.main(["run", "--synthetic", "--workers", "1", "--work", str(work)])
    rc = p116.main(["check", "--synthetic", "--work", str(work)])
    assert (rc["mcsgarch_n_identical"] == rc["rows_committed"]).all()
    assert (rc["y_n_identical"] == rc["rows_committed"]).all()
    p116.main(["analyze", "--synthetic", "--reps", "100", "--work", str(work), "--out", str(out)])
    for f in ("tests.csv", "tests_detail.csv", "verdict.csv", "losses.csv", "attribution.csv", "gain_by_anchor.csv",
              "reproduction_check.csv", "params_by_refit.csv", "params_summary.csv", "run_info.json",
              "daily_losses_synth_next15.csv", "daily_losses_synth_next30.csv"):
        assert (out / f).exists(), f
    tests = pd.read_csv(out / "tests.csv")
    assert list(tests.columns) == ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw",
                                   "sidedness", "p_method", "p_adjusted", "adjust_method"]
    prim = tests[tests["role"] == "primary"]
    assert set(prim["test_id"]) == {"P_next15_hybrid_vs_cpp", "P_next30_hybrid_vs_cpp", "P_next15_ours_vs_cpp",
                                    "P_next30_ours_vs_cpp"}
    assert prim["adjust_method"].str.startswith("Holm over 4").all()
    assert prim.loc[prim["test_id"].str.contains("ours_vs_cpp"), "sidedness"].str.startswith("two-sided").all()
    assert tests["p_adjusted"].between(0, 1).all()
    assert (tests["p_adjusted"] >= tests["p_raw"] - 1e-15).all()
    v = pd.read_csv(out / "verdict.csv")
    assert len(v) == 1 and v["role"].iloc[0] == "primary (decision rule)"
    L = pd.read_csv(out / "losses.csv")
    a = L[(L["subset"] == "all days") & (L["target"] == "next15")].set_index("model")
    assert a.loc["mcs_ours", "gain_pct_vs_mcs_ours"] == 0 and a.loc["mcs_cpp", "gain_pct_vs_mcs_cpp"] == 0
    att = pd.read_csv(out / "attribution.csv")
    s = att[[f"shapley_{f}_points" for f in p116.FACTORS]].sum(axis=1)
    assert np.allclose(s, att["total_points"], atol=1e-10)
