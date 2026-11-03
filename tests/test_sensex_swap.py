"""Fast unit tests for experiments/sensex_swap.py (P92). SYNTHETIC data only: no real file is read."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from niftygex import replication as rp
from niftygex.connectedness import connectedness_from_fit
from niftygex.expiry_calendar import weekly_expiries

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "sensex_swap.py"
if "sensex_swap" in sys.modules:
    ss = sys.modules["sensex_swap"]
else:
    _spec = importlib.util.spec_from_file_location("sensex_swap", _PATH)
    ss = importlib.util.module_from_spec(_spec)
    sys.modules["sensex_swap"] = ss  # dataclasses need the module registered
    _spec.loader.exec_module(ss)
cg = ss.cg

# the reported Sensex weekday regimes (dates only), as load_regimes returns them
SX_REGIMES = [(pd.Timestamp("2023-05-15"), pd.Timestamp("2025-01-03"), 4),
              (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-08-31"), 1),
              (pd.Timestamp("2025-09-01"), pd.Timestamp("2100-01-01"), 3)]


# ---------------------------------------------------------------- synthetic helpers


def synth_calendar(start="2025-01-01", end="2026-06-30", holidays=()):
    """A calendar like the real one: Nifty Thursday then Tuesday, Sensex by the reported regimes, holiday roll-back."""
    days = pd.bdate_range(start, end).difference(pd.DatetimeIndex(holidays))
    nexp = weekly_expiries(days)
    nifty = pd.DataFrame({"is_expiry": days.isin(nexp), "is_monthly_expiry": days.isin(list(ss.last_per_month(nexp))),
                          "regime": np.where(days < ss.SWAP, "single_weekly_thu", "single_weekly_tue")}, index=days)
    sx = ss.sc.sensex_rule_expiries(days, SX_REGIMES)
    return ss.build_calendar(days, nifty, sx)


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


def day_net(inc, expiry_effect, rng, M=120):
    """Per-day NET from simulated minute series (order: Nifty, Sensex). The index that expires leads the other by
    ``expiry_effect``; Tuesdays add a Nifty lead and the after period a Sensex lead (weekday and period effects that
    the DiD must remove)."""
    n_exp = (inc["status"] == "N").to_numpy()
    s_exp = (inc["status"] == "S").to_numpy()
    tue = (inc["weekday"] == 1).to_numpy()
    after = inc["after"].to_numpy() == 1
    c10 = 0.10 + expiry_effect * n_exp + 0.10 * tue  # lagged Nifty in the Sensex equation: Nifty to Sensex
    c01 = 0.10 + expiry_effect * s_exp + 0.10 * after  # lagged Sensex in the Nifty equation: Sensex to Nifty
    A = np.stack([np.array([[0.5, b], [a, 0.5]]) for a, b in zip(c10, c01)])
    L = np.tile(np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]])), (len(inc), 1, 1))
    Y = simulate_days(A, L, M, rng)
    vals = ss.per_day_values(cg.day_stats(Y, ss.P_MAIN), ss.H_MAIN)
    return ss.net_from_values(vals)


# ---------------------------------------------------------------- reuse of the connectedness study's code


def test_build_pair_grids_matches_connectedness_study_grids():
    rng = np.random.default_rng(1)
    rows = []
    for k, lvl in (("nifty", 20000.0), ("banknifty", 45000.0)):
        for d in pd.bdate_range("2025-03-03", "2025-03-14"):
            ts = pd.date_range(d + pd.Timedelta("09:15:00"), d + pd.Timedelta("15:29:00"), freq="1min")
            c = lvl * np.exp(np.cumsum(rng.normal(0, 3e-4, len(ts))))
            df = pd.DataFrame({"timestamp": ts, "open": c, "high": c, "low": c, "close": c, "k": k})
            if k == "banknifty" and d == pd.Timestamp("2025-03-05"):
                df = df.iloc[::2]  # too few bars: the day must drop out for both
            if k == "nifty" and d == pd.Timestamp("2025-03-07"):
                df = df.drop(df.index[100:110])  # ten missing bars: the day stays
            rows.append(df)
    allb = pd.concat(rows)
    bars = {}
    for k in ("nifty", "banknifty"):
        b = allb[allb["k"] == k].drop(columns="k").reset_index(drop=True)
        b["date"] = b["timestamp"].dt.normalize()
        b["tod"] = b["timestamp"].dt.strftime("%H:%M")
        bars[k] = b
    ref = cg.build_grids(bars, cg.TRIM)
    new = ss.build_pair_grids(bars, cg.TRIM, keys=("nifty", "banknifty"))
    assert ref["days"].equals(new["days"]) and pd.Timestamp("2025-03-05") not in new["days"]
    for k in ("nifty", "banknifty"):
        assert np.allclose(ref[k]["r"], new[k]["r"], equal_nan=True)
        assert np.allclose(ref[k]["daily"], new[k]["daily"], equal_nan=True)


def test_net_is_nifty_net_directional_connectedness():
    rng = np.random.default_rng(2)
    A = np.array([[0.5, 0.0], [0.35, 0.5]])  # Nifty leads Sensex
    L = np.tile(np.linalg.cholesky(np.array([[1.0, 0.4], [0.4, 1.0]])), (4, 1, 1))
    Y = simulate_days(np.tile(A, (4, 1, 1)), L, 300, rng)
    ds = cg.day_stats(Y, 2)
    vals = ss.per_day_values(ds, 10)
    net = ss.net_from_values(vals)
    assert np.all(net > 5)
    fit = cg.fit_from_sums(ds, ds.F[0])
    assert np.isclose(net[0], connectedness_from_fit(fit, 10).net[0])
    Y2 = Y[:, :, ::-1]  # swap the two series: Sensex now leads
    assert np.allclose(ss.net_from_values(ss.per_day_values(cg.day_stats(Y2, 2), 10)), -net, atol=1e-8)


def test_pooled_cell_did_matches_replication_segments():
    rng = np.random.default_rng(3)
    Y = np.cumsum(rng.normal(size=(12, 80, 2)), axis=1) * 0.1 + 5
    Y[2, 7] = np.nan
    ds = cg.day_stats(Y, 2)
    pos = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    is_a = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    after = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def net_of(days_):
        Ys, segs = [], []
        for n, d in enumerate(days_):
            ok = np.all(np.isfinite(Y[d]), axis=1)
            Ys.append(Y[d][ok])
            segs.append(n * 1_000_000 + np.cumsum(~ok)[ok])
        c = rp.connectedness_segments(np.vstack(Ys), np.concatenate(segs), 2, 10)
        return c.table[1, 0] - c.table[0, 1]

    want = (net_of([4, 5]) - net_of([0, 1])) - (net_of([6, 7]) - net_of([2, 3]))
    assert np.isclose(ss.pooled_cell_did(ds, 10, pos, is_a, after), want, atol=1e-8)


def test_volatility_inputs_places_each_fit_group():
    days = pd.bdate_range("2025-01-01", periods=10)
    G = {"days": days, "minutes": ["09:15", "09:16", "09:17"]}

    def fake_fit(G_, k, rows):
        val = rows[:, None] * 1.0 + (0.5 if k == "sensex" else 0.0) + rows.min() * 100
        return {"cond_vol": np.repeat(val, 3, axis=1), "intraday_vol": np.ones((len(rows), 3)), "params": {"alpha": 0.1, "beta": 0.8}}

    sample = days[2:]
    out = ss.volatility_inputs(G, sample, {"first": sample[:4], "second": sample[4:]}, fit_fn=fake_fit)
    c = out["cond"]["nifty"][:, 0]
    assert np.allclose(c[:4], np.arange(2, 6) + 200) and np.allclose(c[4:], np.arange(6, 10) + 600)
    assert np.allclose(out["cond"]["sensex"][:, 0] - c, 0.5)
    assert set(out["fits"]["fit"]) == {"first", "second"} and len(out["fits"]) == 4


def test_window_only_keeps_the_last_hour():
    minutes = rp.session_minutes(ss.TRIM)
    Y = np.ones((2, len(minutes), 2))
    Z = ss.window_only(Y, minutes, *ss.LAST_HOUR)
    keep = np.isfinite(Z[0, :, 0])
    assert keep.sum() == 60 and minutes[np.argmax(keep)] == "14:15" and minutes[-1] == "15:14"
    rng = np.random.default_rng(4)
    Yr = np.cumsum(rng.normal(size=(2, len(minutes), 2)), axis=1) * 0.1 + 5
    Z = ss.window_only(Yr, minutes, *ss.LAST_HOUR)
    Z[1, -15:, 0] = np.nan  # only 45 valid minutes left on day 1
    vals = ss.per_day_values(cg.day_stats(Z, 2), 10, ss.MIN_LAST_HOUR, Z)
    assert np.all(np.isfinite(vals[0])) and np.all(np.isnan(vals[1]))


# ---------------------------------------------------------------- calendar and cells


def test_calendar_status_cells_and_holiday_shifts():
    cal = synth_calendar("2024-11-21", "2026-06-30", holidays=["2025-04-10", "2025-10-02"])
    assert cal.loc["2025-04-09", "status"] == "N" and cal.loc["2025-04-09", "nifty_shifted"]
    assert cal.loc["2025-10-01", "status"] == "S" and cal.loc["2025-10-01", "sensex_shifted"]
    assert cal.loc["2025-01-03", "status"] == "S" and cal.loc["2025-01-07", "status"] == "S"  # last Friday, first Tuesday
    assert cal.loc["2025-01-10", "status"] == "none" and cal.loc["2024-12-31", "status"] == "none"
    assert cal.loc["2025-09-02", "status"] == "N" and cal.loc["2025-09-04", "status"] == "S"
    assert list(pd.unique(cal["sensex_period"])) == ["sensex_fri", "sensex_tue", "sensex_thu"]
    inc, ex = ss.select_cells(cal, ss.P1)
    assert set(inc["cell"]) == {"Tue before", "Thu before", "Tue after", "Thu after"} and ex.empty
    assert (inc.groupby("cell")["status"].agg(lambda s: s.nunique()) == 1).all()
    assert inc.loc[inc["cell"] == "Tue before", "status"].iloc[0] == "S" and inc.loc[inc["cell"] == "Tue after", "status"].iloc[0] == "N"
    assert pd.Timestamp("2025-01-02") in inc.index and inc.index.min() >= pd.Timestamp("2025-01-01")
    inc, ex = ss.select_cells(cal, ss.PL1)
    assert {pd.Timestamp("2025-04-09"), pd.Timestamp("2025-10-01")} <= set(ex.index)
    assert (inc["status"] == "none").all()
    inc, _ = ss.select_cells(cal, ss.S3)
    assert inc.loc[inc["cell"] == "Fri before"].index.max() == pd.Timestamp("2025-01-03")
    assert inc.loc[inc["cell"] == "Tue before"].index.max() == pd.Timestamp("2024-12-31")
    drop = pd.Series(cal["nifty_monthly"] | cal["sensex_monthly_rule"], index=cal.index)
    inc2, ex2 = ss.select_cells(cal, ss.P1, drop)
    assert len(inc2) < len(ss.select_cells(cal, ss.P1)[0]) and (ex2["reason"] == "dropped by this check").all()
    bad = ss.DidSpec("x", "t", "x", 1, 3, "2025-01-01", "2025-09-02", "2025-12-31", ss.P1.status, "greater")
    with pytest.raises(ValueError):
        ss.select_cells(cal, bad)


def test_last_per_month():
    d = pd.to_datetime(["2025-01-07", "2025-01-28", "2025-02-04", "2025-02-25"])
    assert ss.last_per_month(d) == {pd.Timestamp("2025-01-28"), pd.Timestamp("2025-02-25")}


# ---------------------------------------------------------------- statistics


def test_did_equals_cell_means_and_ols_interaction():
    rng = np.random.default_rng(5)
    n = 200
    is_a = rng.integers(0, 2, n)
    after = rng.integers(0, 2, n)
    y = rng.normal(size=n) + 2 * is_a + 3 * after + 1.5 * is_a * after
    m = lambda a, f: y[(is_a == a) & (after == f)].mean()  # noqa: E731
    want = (m(1, 1) - m(1, 0)) - (m(0, 1) - m(0, 0))
    assert np.isclose(ss.did_many(y, is_a, after)[0], want)
    assert np.isclose(ss.did_hac(y, is_a, after, "greater")["hac_coef"], want)
    two = ss.did_many(y, is_a, np.vstack([after, 1 - after]))
    assert np.isclose(two[1], -want)


def test_paired_week_did_equals_did_when_every_week_is_complete():
    cal = synth_calendar("2025-01-01", "2025-12-31")
    inc, _ = ss.select_cells(cal, dataclasses.replace(ss.P1, lo="2025-01-06", hi="2025-12-28"))  # complete weeks only
    rng = np.random.default_rng(6)
    y = rng.normal(size=len(inc))
    est, wk = ss.paired_week_did(y, inc["is_a"], inc["after"], inc["week"])
    assert np.isclose(est, ss.did_many(y, inc["is_a"].to_numpy(), inc["after"].to_numpy())[0])
    assert len(wk) == inc["week"].nunique()


def test_perm_p_directions():
    d = np.linspace(-1, 1, 99)
    assert np.isclose(ss.perm_p(2.0, d, "greater"), 1 / 100) and np.isclose(ss.perm_p(2.0, d, "less"), 1.0)
    assert np.isclose(ss.perm_p(-2.0, d, "two-sided"), 1 / 100)


def test_planted_swap_effect_is_detected():
    """The expiring index leads the other on its own expiry day; weekday and period effects are added on top."""
    cal = synth_calendar("2025-01-01", "2026-06-30")
    inc, _ = ss.select_cells(cal, ss.P1)
    rng = np.random.default_rng(7)
    net = day_net(inc, 0.25, rng)
    r = ss.did_perm_test(net, inc["is_a"], inc["after"], inc["week"], "greater", 999, ss.BLOCK_WEEKS, np.random.default_rng(8))
    h = ss.did_hac(net, inc["is_a"].to_numpy(), inc["after"].to_numpy(), "greater")
    assert r["estimate"] > 20 and r["perm_p"] <= 0.002 and h["hac_p"] < 0.001
    assert abs(r["perm_null_mean"]) < 2
    w = ss.did_perm_test(net, inc["is_a"], inc["after"], inc["week"], "greater", 499, 4, np.random.default_rng(9), how="within_weekday")
    assert w["perm_p"] <= 0.01
    v = ss.primary_verdict(r["estimate"], r["perm_p_greater"], r["perm_p_less"], h["hac_upper95_one_sided"], 5.0, {})
    assert v["verdict"] == "SUPPORTED"


def test_no_swap_effect_is_not_detected_despite_weekday_and_period_effects():
    cal = synth_calendar("2025-01-01", "2026-06-30")
    inc, _ = ss.select_cells(cal, ss.P1)
    net = day_net(inc, 0.0, np.random.default_rng(10), M=240)
    m = pd.Series(net, index=inc.index).groupby(inc["cell"]).mean()
    assert (m["Tue before"] + m["Tue after"]) - (m["Thu before"] + m["Thu after"]) > 4  # the weekday effect is there
    assert (m["Tue after"] + m["Thu after"]) - (m["Tue before"] + m["Thu before"]) < -4  # and the period effect
    r = ss.did_perm_test(net, inc["is_a"], inc["after"], inc["week"], "greater", 999, ss.BLOCK_WEEKS, np.random.default_rng(11))
    assert r["perm_p"] > 0.05 and abs(r["estimate"]) < 4
    two = ss.did_perm_test(net, inc["is_a"], inc["after"], inc["week"], "two-sided", 999, ss.BLOCK_WEEKS, np.random.default_rng(12))
    assert two["perm_p"] > 0.05


def test_permutation_p_roughly_uniform_under_the_null():
    """No swap effect; each weekday has its own persistent (AR) component and weeks share a common shock."""
    cal = synth_calendar("2025-01-01", "2026-06-30")
    inc, _ = ss.select_cells(cal, ss.P1)
    code, uniq = ss.week_codes(inc["week"])
    W = len(uniq)
    is_a = inc["is_a"].to_numpy()
    rng = np.random.default_rng(13)
    pv = []
    for r in range(150):
        ar = np.zeros((2, W))
        e = rng.standard_normal((2, W))
        for t in range(1, W):
            ar[:, t] = 0.3 * ar[:, t - 1] + e[:, t]
        common = rng.standard_normal(W) * 2
        y = common[code] + ar[is_a, code] + rng.standard_normal(len(inc))
        res = ss.did_perm_test(y, is_a, inc["after"], inc["week"], "greater", 299, ss.BLOCK_WEEKS, np.random.default_rng([14, r]))
        pv.append(res["perm_p"])
    pv = np.array(pv)
    assert stats.kstest(pv, "uniform").pvalue > 0.001
    assert 0.0 <= np.mean(pv <= 0.05) <= 0.12


def test_h2_regression_recovers_an_expiry_effect():
    cal = synth_calendar("2024-11-21", "2026-06-30", holidays=["2025-04-10", "2025-10-02", "2026-03-03"])
    rng = np.random.default_rng(15)
    wd_eff = np.array([0.0, 1.0, -0.5, 0.5, 2.0])[cal["weekday"].to_numpy()]
    per_eff = pd.Series(cal["sensex_period"]).map({"sensex_fri": 0.0, "sensex_tue": 3.0, "sensex_thu": 6.0}).to_numpy()
    y = 80 + wd_eff + per_eff + 4.0 * cal["either_expiry"].to_numpy() + rng.normal(0, 1.0, len(cal))
    r = ss.expiry_regression(y, cal["either_expiry"], cal["weekday"], cal["sensex_period"])
    assert abs(r["estimate"] - 4.0) < 1.0 and r["hac_p"] < 0.001
    y0 = 80 + wd_eff + per_eff + rng.normal(0, 1.0, len(cal))
    r0 = ss.expiry_regression(y0, cal["either_expiry"], cal["weekday"], cal["sensex_period"])
    assert abs(r0["estimate"]) < 1.0
    rd = ss.expiry_regression(y0, cal["either_expiry"], cal["weekday"], cal["sensex_period"], weekday_effects=False)
    assert "wd_Tue" not in rd["params"]


def test_mde_formulas():
    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.8)
    assert np.isclose(ss.mde_did([34, 33, 56, 56], 1.0), z * np.sqrt(1 / 34 + 1 / 33 + 2 / 56))
    assert np.isclose(ss.mde_did([10, 10, 10, 10], 2.0, one_sided=False), (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * 2 * np.sqrt(0.4))
    X = np.column_stack([np.ones(40), np.r_[np.ones(20), np.zeros(20)]])
    assert np.isclose(ss.mde_regression(X, 1, 1.0), (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * np.sqrt(0.1))
    assert ss.design_effect(-0.2) == 1.0 and np.isclose(ss.design_effect(0.5), 3.0)


def test_proxy_variance_on_a_synthetic_file(tmp_path):
    rng = np.random.default_rng(16)
    days = pd.bdate_range("2024-11-21", "2026-04-22")
    df = pd.DataFrame({"date": days, "regime": np.where(days < ss.SWAP, "single_weekly_thu", "single_weekly_tue"),
                       "day_nifty_to_bn": 40 + rng.normal(0, 3, len(days)), "day_bn_to_nifty": 40 + rng.normal(0, 4, len(days))})
    df["day_total_conn"] = df["day_nifty_to_bn"] + df["day_bn_to_nifty"]
    df.to_csv(tmp_path / "sd.csv", index=False)
    p = ss.proxy_variance(tmp_path / "sd.csv")
    assert abs(p["sigma_net"] - 5.0) < 0.5 and abs(p["rho_week_net"]) < 0.15 and abs(p["corr_tue_thu_net"]) < 0.3


def test_primary_verdict_rules():
    v = ss.primary_verdict(-3.0, 0.99, 0.01, 1.0, 5.0, {})
    assert v["verdict"].startswith("EVIDENCE AGAINST")
    v = ss.primary_verdict(1.0, 0.3, 0.7, 4.0, 5.0, {})
    assert "ruled out" in v["verdict"] and "not ruled out" not in v["verdict"]
    v = ss.primary_verdict(1.0, 0.3, 0.7, 6.0, 5.0, {"PL2": (2.0, 0.01), "PL1": (3.0, 0.02)})
    assert "inconclusive" in v["verdict"]
    assert any("DESIGN CHECK FAILED: PL2" in f for f in v["design_flags"])
    assert any("same direction" in f for f in v["design_flags"])


# ---------------------------------------------------------------- end to end on synthetic files


def _write_synthetic_inputs(tmp: Path, rng) -> None:
    raw = tmp / "raw"
    raw.mkdir()
    days = pd.bdate_range("2024-11-18", ss.END)
    mins = pd.to_timedelta(np.arange(360), unit="min") + pd.Timedelta("09:15:00")
    ts = (days.values[:, None] + mins.values[None, :]).ravel()
    common = np.cumsum(rng.normal(0, 4e-4, len(ts)))
    for k, lvl in (("nifty", 23000.0), ("sensex", 76000.0)):
        c = lvl * np.exp(common + np.cumsum(rng.normal(0, 1e-4, len(ts))))
        pd.DataFrame({"timestamp": pd.DatetimeIndex(ts).strftime("%Y-%m-%d %H:%M:%S"), "open": c.round(2), "high": c.round(2),
                      "low": c.round(2), "close": c.round(2)}).to_csv(raw / f"{k}_1min.csv", index=False)
    nexp = weekly_expiries(days)
    nexp = nexp[nexp < ss.END]  # the end-of-data artifact of the weekday rule
    dm = pd.DataFrame({"is_expiry": days.isin(nexp), "is_monthly_expiry": days.isin(list(ss.last_per_month(nexp))),
                       "regime": np.where(days < ss.SAMPLE_START, "multi_weekly", np.where(days < ss.SWAP, "single_weekly_thu", "single_weekly_tue"))},
                      index=pd.Index(days, name="date"))
    (tmp / "run").mkdir()
    dm.to_csv(tmp / "run" / "daily_market.csv")
    pd.DataFrame({"fact_type": ["weekday_regime"] * 3, "date": ["2023-05-15", "2025-01-01", "2025-09-01"],
                  "date_end": ["2025-01-03", "2025-08-31", ""], "weekday": ["Fri", "Tue", "Thu"]}).to_csv(tmp / "cal.csv", index=False)
    pdays = pd.bdate_range("2024-11-21", "2026-04-22")
    sd = pd.DataFrame({"date": pdays, "regime": np.where(pdays < ss.SWAP, "single_weekly_thu", "single_weekly_tue"),
                       "day_nifty_to_bn": 40 + rng.normal(0, 3, len(pdays)), "day_bn_to_nifty": 40 + rng.normal(0, 4, len(pdays))})
    sd["day_total_conn"] = sd["day_nifty_to_bn"] + sd["day_bn_to_nifty"]
    sd.to_csv(tmp / "sample_days.csv", index=False)


def test_full_pipeline_runs_on_synthetic_files(tmp_path, monkeypatch):
    """Phase 1 and phase 2 end to end on synthetic bars and calendars, with a fast stand-in for the mcsGARCH fit."""
    rng = np.random.default_rng(17)
    _write_synthetic_inputs(tmp_path, rng)
    out = tmp_path / "out"
    for name, val in (("RAW", tmp_path / "raw"), ("RUN", tmp_path / "run"), ("CAL_PATH", tmp_path / "cal.csv"),
                      ("SX_USED", tmp_path / "missing.csv"), ("CG_DAYS", tmp_path / "sample_days.csv"), ("OUT", out),
                      ("LOG", tmp_path / "log.txt"), ("LOG_COVERAGE", tmp_path / "log_cov.txt"), ("PREREG", tmp_path / "prereg.md")):
        monkeypatch.setattr(ss, name, val)

    def fake_fit(G, k, rows):
        R = G[k]["r"][rows]
        v = np.sqrt(pd.DataFrame(R.T ** 2).rolling(5, min_periods=1).mean().to_numpy().T) + 1e-5
        v = np.where(np.isfinite(R), v, np.nan)
        return {"cond_vol": v, "intraday_vol": v * 2, "params": {"alpha": 0.05, "beta": 0.9, "nu": 5.0, "loglik": 0.0}, "daily_method": "fake"}

    monkeypatch.setattr(cg, "fit_regime", fake_fit)
    stdout = sys.stdout
    ss.run_coverage()
    assert sys.stdout is stdout
    cov = pd.read_csv(out / "coverage.csv")
    p1 = cov[cov["test"] == "P1"].set_index("cell")["n_days"]
    assert set(p1.index) == {"Tue before", "Thu before", "Tue after", "Thu after"} and p1.min() >= 30
    assert (out / "mde.csv").exists() and (out / "calendar_days.csv").exists()
    ss.run_full(n_perm=29)
    assert sys.stdout is stdout
    tests = pd.read_csv(out / "tests.csv").set_index("test")
    assert set(tests.index) == {"P1", "S1", "S2", "S3", "PL1", "PL2", "PL3"}
    assert np.isfinite(tests.loc[["P1", "S1", "S3", "PL1", "PL2", "PL3"], "perm_p"]).all()
    assert np.isfinite(tests.loc[["S1", "S2", "S3"], "holm_p_secondary"]).all() and np.isnan(tests.loc["P1", "holm_p_secondary"])
    dv = pd.read_csv(out / "day_values.csv")
    assert np.isfinite(dv["net_day"]).all() and np.isfinite(dv["net_lh"]).all()
    rob = pd.read_csv(out / "robustness.csv")
    assert len(rob) >= 15 and rob["estimate"].notna().all()
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["verdict"].split(" ")[0] in {"SUPPORTED", "EVIDENCE", "NOT"}
