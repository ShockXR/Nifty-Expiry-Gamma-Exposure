"""Tests for P109 (experiments/p109_mcsgarch_extensions.py and the McsGarchExt code path in
niftygex.volatility). SYNTHETIC data only (no raw data file is read).

Covers: the extended intraday recursion against a plain loop and against mcsGARCH's own recursion; McsGarchExt
with every flag off is McsGarch bit for bit; the multi-step forecast against its closed form and against a Monte
Carlo of the model; parameter recovery by maximum likelihood; the HAR daily level (OLS, no look-ahead, scale only
in the constant); the time-of-day calibration (definition, training rows only, QLIKE-optimal); the rows and target
that the driver rebuilds equal features.build_dataset's; the driver reproduces run_walkforward's mcsGARCH exactly;
the variant specifications; the share of the gain and its interval; the registered decision rule; and an
end-to-end dry run of run, check and analyze on a simulated market.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex import volatility as vol
from niftygex.evaluation import qlike
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel
from niftygex.forecast import WalkForwardConfig, run_walkforward
from niftygex.simulate import SimConfig, simulate_market

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p109_mcsgarch_extensions.py"
_spec = importlib.util.spec_from_file_location("p109_mcsgarch_extensions", _PATH)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def simulate_ext(T, M, om, al, be, gl, gx, de, nu=7.0, seed=0):
    """Simulate normalised residuals x from the extended recursion with a plain loop."""
    rng = np.random.default_rng(seed)
    pers = al + be + vol.KAPPA_NEG * gl + gx
    qbar = om / (1 - pers)
    z = rng.standard_t(nu, size=(T, M)) / np.sqrt(nu / (nu - 2))
    x, q = np.zeros((T, M)), np.zeros((T, M))
    qq = qbar
    for t in range(T):
        run = 0.0
        for i in range(M):
            q[t, i] = qq
            x[t, i] = np.sqrt(qq) * z[t, i]
            run += x[t, i] ** 2
            qq = om + al * x[t, i] ** 2 + gl * x[t, i] ** 2 * (x[t, i] < 0) + gx * run / (i + 1) + be * qq
        qq = qbar + de * (qq - qbar)
    return x, q, qbar


@pytest.fixture(scope="module")
def market():
    return simulate_market(SimConfig(n_days=100, kappa=0.0, seed=5, start="2024-08-01"))


# ---------------------------------------------------------------- the recursion


def test_ext_recursion_equals_plain_loop():
    x, q, qbar = simulate_ext(40, 60, 0.004, 0.04, 0.93, 0.03, 0.01, 0.3)
    x2, negx2, z = vol._ext_inputs(x, x < 0)
    q2 = vol.ext_q2(x2, negx2, z, 0.004, 0.04, 0.93, 0.03, 0.01, 0.3, qbar, qbar)
    assert np.max(np.abs(q2 / q - 1)) < 1e-12


def test_ext_recursion_without_extensions_is_mcsgarch_recursion():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((30, 50))
    x2, negx2, z = vol._ext_inputs(x, x < 0)
    a = vol.ext_q2(x2, negx2, z, 0.01, 0.05, 0.94, 0.0, 0.0, 1.0, 1.3, 0.7)
    b = vol._garch_q2(x2.ravel(), 0.01, 0.05, 0.94, 1.3).reshape(30, 50)
    assert np.max(np.abs(a / b - 1)) < 1e-12


def test_flags_off_is_mcsgarch_bit_for_bit(market):
    R, is_exp = market["R"], market["labels"]["is_expiry"].to_numpy()
    h, _ = vol.daily_egarch_variance(R.sum(axis=1), train_size=60)
    a = vol.McsGarch(expiry_diurnal=True, smooth=5).fit(R[:60], h[:60], is_exp[:60])
    b = vol.McsGarchExt(expiry_diurnal=True, smooth=5).fit(R[:60], h[:60], is_exp[:60])
    assert not b.extended and a.params == b.params
    fa, fb = a.filter(R, h, is_exp), b.filter(R, h, is_exp)
    assert np.array_equal(fa["q2"], fb["q2"])
    tt = np.repeat(np.arange(60, 100), 3)
    ii = np.tile([0, 150, 300], 40)
    assert np.array_equal(a.forecast_variance(fa, h, tt, ii, 30), b.forecast_variance(fb, h, tt, ii, 30))


def test_ext_forecast_reduces_to_closed_form_and_matches_monte_carlo(market):
    R, is_exp = market["R"], market["labels"]["is_expiry"].to_numpy()
    h, _ = vol.daily_egarch_variance(R.sum(axis=1), train_size=60)
    base = vol.McsGarch(expiry_diurnal=True, smooth=5).fit(R[:60], h[:60], is_exp[:60])
    ext = vol.McsGarchExt(expiry_diurnal=True, smooth=5, overnight_decay=True)
    ext.s, ext.s_expiry = base.s, base.s_expiry
    p = base.params
    ext.params = {**p, "gam_l": 0.0, "gam_x": 0.0, "delta": 1.0,
                  "qbar": p["omega"] / (1 - p["alpha"] - p["beta"])}
    fb, fe = base.filter(R, h, is_exp), ext.filter(R, h, is_exp)
    assert np.max(np.abs(fe["q2"] / fb["q2"] - 1)) < 1e-10
    tt = np.repeat(np.arange(60, 100), 4)
    ii = np.tile([0, 15, 200, 345], 40)
    a = base.forecast_variance(fb, h, tt, ii, 30)
    b = ext.forecast_variance(fe, h, tt, ii, 30)
    assert np.max(np.abs(a / b - 1)) < 1e-10
    # Monte Carlo of the full extended model from one state: mean realized window variance = forecast
    ext2 = vol.McsGarchExt(expiry_diurnal=False, smooth=1, leverage=True, garchx_today=True)
    ext2.s = np.ones(20)
    om, al, be, gl, gx = 0.02, 0.06, 0.88, 0.04, 0.02
    pers = al + be + 0.5 * gl + gx
    ext2.params = {"omega": om, "alpha": al, "beta": be, "gam_l": gl, "gam_x": gx, "delta": 1.0, "nu": np.inf,
                   "qbar": om / (1 - pers)}
    rng = np.random.default_rng(3)
    x_hist = rng.standard_normal((1, 20)) * 1.5
    filt = ext2.filter(np.where(np.arange(20) < 8, x_hist, 0.0), np.ones(1), None)
    f = ext2.forecast_variance(filt, np.ones(1), [0], [8], 10)[0]
    n = 200000
    q = np.full(n, filt["q2"][0, 8])
    run = np.full(n, filt["cum_x2"][0, 8])
    tot = np.zeros(n)
    for j in range(8, 18):
        xx = np.sqrt(q) * rng.standard_normal(n)
        tot += xx**2
        run += xx**2
        q = om + al * xx**2 + gl * xx**2 * (xx < 0) + gx * run / (j + 1) + be * q
    assert abs(tot.mean() / f - 1) < 0.01


def test_fit_recovers_parameters():
    x, _, _ = simulate_ext(300, 375, 0.004, 0.04, 0.93, 0.03, 0.01, 0.3, seed=2)
    p = vol.fit_garch_ext(x, x < 0, None, overnight_decay=True, leverage=True, garchx_today=True)
    assert p["converged"]
    assert abs(p["alpha"] - 0.04) < 0.008 and abs(p["beta"] - 0.93) < 0.015
    assert abs(p["gam_l"] - 0.03) < 0.01 and abs(p["gam_x"] - 0.01) < 0.006
    assert abs(p["delta"] - 0.3) < 0.1 and abs(p["nu"] - 7.0) < 1.0
    # nested: the extended likelihood is at least the plain GARCH(1,1) one on the same data
    b = vol.fit_garch11(x.ravel())
    assert p["loglik"] >= b["loglik"] - 1e-6


# ---------------------------------------------------------------- HAR daily level and calibration


def test_har_daily_variance_ols_and_no_lookahead():
    rng = np.random.default_rng(4)
    rv = np.exp(rng.standard_normal(200) * 0.5 - 9)
    h, info = vol.har_daily_variance(rv, train_size=120)
    t = np.arange(1, 120)
    X = np.column_stack([np.ones(119), np.log(rv[t - 1]),
                         np.log([rv[max(0, s - 5):s].mean() for s in t]),
                         np.log([rv[max(0, s - 22):s].mean() for s in t])])
    b, *_ = np.linalg.lstsq(X, np.log(rv[1:120]), rcond=None)
    assert np.allclose(list(info["params"].values()), b, rtol=1e-10, atol=1e-12)
    rv2 = rv.copy()
    rv2[150:] *= 50                                   # later days change: nothing up to day 150 may change
    h2, _ = vol.har_daily_variance(rv2, train_size=120)
    assert np.array_equal(h[:151], h2[:151])
    h3, info3 = vol.har_daily_variance(rv * 7.0, train_size=120)
    assert np.allclose(h3 / h, 7.0)                   # a scale enters the constant only
    assert np.allclose(list(info3["params"].values())[1:], b[1:])


def test_calibration_factors_definition_and_optimality():
    rng = np.random.default_rng(5)
    minute = np.tile([0, 15, 30], 50)
    f = np.exp(rng.standard_normal(150))
    y = f * np.exp(rng.standard_normal(150)) * np.where(minute == 0, 2.0, 1.0)
    train = np.arange(150) < 120
    c = ex.calibration_factors(y, f, minute, train)
    for m in (0, 15, 30):
        sel = train & (minute == m)
        assert c[m] == pytest.approx(np.mean(y[sel] / f[sel]))
        base = qlike(y[sel], c[m] * f[sel]).mean()
        for k in (0.9, 1.1):
            assert qlike(y[sel], k * c[m] * f[sel]).mean() > base
    y2 = y.copy()
    y2[~train] *= 100                                 # test rows do not enter
    assert ex.calibration_factors(y2, f, minute, train) == c


# ---------------------------------------------------------------- the driver


def test_rows_and_target_equal_build_dataset(market):
    ds = build_dataset(market, build_gex_panel(market), horizon=30, step=15)
    T, M = market["R"].shape
    rows = ex.dataset_rows(T, M, T, 30, 15)
    assert np.array_equal(rows[["t", "minute", "horizon"]].to_numpy(), ds[["t", "minute", "horizon"]].to_numpy())
    assert np.array_equal(ex.realized_target(market["R"], rows), ds["y"].to_numpy())


def test_driver_reproduces_run_walkforward_mcsgarch(market, tmp_path):
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle"}
    ds = build_dataset(market, build_gex_panel(market), horizon=15, step=45)
    cfg = WalkForwardConfig(models=["mcsgarch"], gex_sets=sets, min_train=50, block=20, mcs_refit_every=20)
    pred, _ = run_walkforward(market, ds, cfg)
    ex._STATE.update(R=market["R"], is_exp=market["labels"]["is_expiry"].to_numpy(), days=market["days"], block=20,
                     variants=["mcsgarch", "x5_calibration"], targets={"next15": (15, 45, None)})
    for s in range(50, 100, 20):
        ex.run_block(("SYNTH", s, str(tmp_path)))
    got = ex.assemble(tmp_path, "SYNTH")
    m = pred.merge(got, on=["t", "minute"], suffixes=("", "_p109"))
    assert len(m) == len(pred) == len(got)
    assert np.array_equal(m["mcsgarch"].to_numpy(), m["mcsgarch_p109"].to_numpy())
    assert np.array_equal(m["y"].to_numpy(), m["y_p109"].to_numpy())
    assert not np.array_equal(m["mcsgarch"].to_numpy(), m["x5_calibration"].to_numpy())


def test_specs():
    assert ex.NIFTY_VARIANTS[0] == "mcsgarch" and len(ex.NIFTY_VARIANTS) == 12
    assert set(ex.OTHER_VARIANTS) <= set(ex.NIFTY_VARIANTS) and len(ex.OTHER_VARIANTS) == 7
    comb = ex.SPECS["combination"]
    assert comb == dict(daily="har", decay=True, lev=True, todayrv=True, cal=True)
    on = {"x1_har": ("daily", "har"), "x2_overnight": ("decay", True), "x3_todayrv": ("todayrv", True),
          "x4_leverage": ("lev", True), "x5_calibration": ("cal", True)}
    base = ex.SPECS["mcsgarch"]
    for k, x in enumerate(ex.EXTENSIONS, start=1):
        key, val = on[x]
        assert {a for a in base if ex.SPECS[x][a] != base[a]} == {key} and ex.SPECS[x][key] == val
        loo = ex.SPECS[f"comb_no_x{k}"]
        assert {a for a in comb if loo[a] != comb[a]} == {key} and loo[key] == base[key]
    assert len({ex.fit_key(v) for v in ex.NIFTY_VARIANTS}) == 10
    assert len({ex.fit_key(v) for v in ex.OTHER_VARIANTS}) == 6


# ---------------------------------------------------------------- statistics and the decision rule


def test_recovered_share():
    rng = np.random.default_rng(6)
    n = 300
    mcs = 1 + rng.random(n)
    D = pd.DataFrame({"mcsgarch": mcs, "hybrid": mcs - 0.1 + 0.02 * rng.standard_normal(n),
                      "m": mcs - 0.06 + 0.02 * rng.standard_normal(n)})
    s, lo, hi = ex.recovered_share(D, "m", reps=500)
    assert s == pytest.approx((D["mcsgarch"].mean() - D["m"].mean()) / (D["mcsgarch"].mean() - D["hybrid"].mean()))
    assert lo < s < hi and 0.4 < s < 0.8
    assert ex.recovered_share(D, "hybrid", reps=50)[0] == pytest.approx(1.0)
    assert ex.gain(D, "hybrid") == pytest.approx(100 * (D["mcsgarch"].mean() - D["hybrid"].mean()) / D["mcsgarch"].mean())


def test_primary_verdict_rules():
    p_ok = {"next15": 0.001, "next30": 0.002}
    p_no = {"next15": 0.2, "next30": 0.3}
    v = ex.primary_verdict(p_ok, {"next15": (0.8, 0.6, 1.0), "next30": (0.7, 0.55, 0.9)})
    assert v["verdict"].startswith("supported: recovers most")
    v = ex.primary_verdict(p_ok, {"next15": (0.6, 0.4, 0.8), "next30": (0.7, 0.55, 0.9)})
    assert v["verdict"].startswith("supported in the point estimate only")
    v = ex.primary_verdict(p_ok, {"next15": (0.3, 0.2, 0.45), "next30": (0.2, 0.1, 0.4)})
    assert v["verdict"].startswith("contradicted") and "still beats" in v["verdict"]
    v = ex.primary_verdict(p_no, {"next15": (0.1, -0.1, 0.3), "next30": (0.0, -0.2, 0.2)})
    assert v["verdict"].startswith("contradicted") and "does not beat" in v["verdict"]
    v = ex.primary_verdict(p_ok, {"next15": (0.45, 0.3, 0.6), "next30": (0.7, 0.55, 0.9)})
    assert v["verdict"].startswith("inconclusive: beats")
    v = ex.primary_verdict({"next15": 0.01, "next30": 0.2}, {"next15": (0.6, 0.4, 0.8), "next30": (0.6, 0.4, 0.8)})
    assert v["verdict"].startswith("inconclusive: beats")


# ---------------------------------------------------------------- end-to-end dry run


def test_end_to_end_dry_run(tmp_path, monkeypatch):
    monkeypatch.setitem(ex.SYNTH, "n_days", 110)
    monkeypatch.setitem(ex.SYNTH, "min_train", 60)
    monkeypatch.setitem(ex.SYNTH, "block", 25)
    work, out = tmp_path / "work", tmp_path / "out"
    ex.main(["run", "--synthetic", "--workers", "1", "--work", str(work)])
    chk = ex.main(["check", "--synthetic", "--work", str(work)])
    assert chk is None
    rc = ex.stage_check(type("A", (), {"work": str(work), "synthetic": True, "indexes": []})())
    assert (rc["mcsgarch_n_identical"] == rc["rows_committed"]).all()
    assert (rc["y_n_identical"] == rc["rows_committed"]).all()
    ex.main(["analyze", "--synthetic", "--reps", "100", "--work", str(work), "--out", str(out)])
    for f in ("tests.csv", "verdict.csv", "extensions.csv", "gains.csv", "shares.csv", "gain_by_anchor.csv",
              "reproduction_check.csv", "params_by_refit.csv", "params_summary.csv", "calibration_factors.csv",
              "run_info.json"):
        assert (out / f).exists(), f
    tests = pd.read_csv(out / "tests.csv")
    assert list(tests.columns) == ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw",
                                   "sidedness", "p_method", "p_adjusted", "adjust_method"]
    prim = tests[tests["role"] == "primary"]
    assert set(prim["test_id"]) == {"P1_next15_combination", "P1_next30_combination"}
    assert prim["adjust_method"].str.startswith("Holm over 2").all()
    assert (tests.loc[tests["test_id"].str.startswith("S1_"), "adjust_method"].str.startswith("Holm over 10")).all()
    assert tests["p_adjusted"].between(0, 1).all()
    assert len(pd.read_csv(out / "verdict.csv")) == 1
