"""Unit tests for experiments/referee_expiry.py on small SYNTHETIC data: the full-bhavcopy
reader, the all-index expiry calendar, the clean-baseline R2, the few-unit inference
(non-overlapping placebo windows, prediction t, wild cluster bootstrap), the trend-adjusted
change and the noise-robust estimators (checked on simulated noisy prices)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load("referee_expiry", _ROOT / "experiments" / "referee_expiry.py")
rdl = _load("real_day_level_t", _ROOT / "experiments" / "real_day_level.py")
es = rx.es


# ---------------------------------------------------------------- full bhavcopy reader


def test_reader_legacy_layout_aggregates_index_rows():
    csv = (
        "INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP\n"
        "OPTIDX,FINNIFTY,03-Jan-2023,18000,CE,1,1,1,1,1,10,2.5,100,0,03-JAN-2023\n"
        "OPTIDX,FINNIFTY,03-Jan-2023,18100,PE,1,1,1,1,1,5,1.5,50,0,03-JAN-2023\n"
        "FUTIDX,FINNIFTY,31-Jan-2023,0,XX,1,1,1,1,1,7,9,70,0,03-JAN-2023\n"
        "OPTSTK,RELIANCE,25-Jan-2023,2500,CE,1,1,1,1,1,3,1,10,0,03-JAN-2023\n"
    ).encode()
    r = rx.index_rows_from_full_bhavcopy(csv)
    assert set(r["symbol"]) == {"FINNIFTY"}
    o = r[r["inst"] == "opt"].iloc[0]
    assert o["n_series"] == 2 and o["contracts"] == 15
    assert o["turnover_rs"] == pytest.approx(4.0e5)
    assert o["expiry"] == pd.Timestamp("2023-01-03") and o["date"] == pd.Timestamp("2023-01-03")
    assert (r["inst"] == "fut").sum() == 1


def test_reader_udiff_layout_uses_actual_expiry():
    csv = (
        "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,"
        "FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,"
        "TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
        "2024-07-15,2024-07-15,FO,NSE,IDO,1,,MIDCPNIFTY,,2024-07-15,2024-07-15,12000,CE,X,1,1,1,1,1,1,1,1,40,0,8,1000,1,F1,75,,,,,\n"
        "2024-07-15,2024-07-15,FO,NSE,STO,2,,TCS,,2024-07-25,2024-07-25,4000,CE,Y,1,1,1,1,1,1,1,1,40,0,8,1000,1,F1,75,,,,,\n"
    ).encode()
    r = rx.index_rows_from_full_bhavcopy(csv)
    assert len(r) == 1
    assert r.iloc[0]["symbol"] == "MIDCPNIFTY" and r.iloc[0]["actual_expiry"] == pd.Timestamp("2024-07-15")


# ---------------------------------------------------------------- calendar


def _ext(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "inst", "expiry", "contracts"]).assign(
        date=lambda d: pd.to_datetime(d["date"]), expiry=lambda d: pd.to_datetime(d["expiry"]),
        turnover_rs=lambda d: d["contracts"] * 1e5)


def test_nse_expiries_weekly_monthly_and_traded_only():
    ext = _ext([
        ("2024-01-02", "FINNIFTY", "opt", "2024-01-02", 100),   # weekly expiry, traded
        ("2024-01-09", "FINNIFTY", "opt", "2024-01-09", 0),     # listed, not traded
        ("2024-01-30", "FINNIFTY", "opt", "2024-01-30", 50),    # monthly (future expires too)
        ("2024-01-02", "FINNIFTY", "fut", "2024-01-30", 10),
        ("2024-01-30", "FINNIFTY", "fut", "2024-01-30", 10),    # the monthly future runs to its expiry
        ("2024-01-02", "FINNIFTY", "opt", "2024-01-09", 70),    # not an expiry on 2 Jan
        ("2024-01-02", "FINNIFTY", "fut", "2024-01-02", 5),     # a weekly future: not the month's last
        ("2024-01-02", "FINNIFTY", "fut", "2024-01-31", 3),     # old-calendar future withdrawn early: ignored
    ])
    e = rx.nse_expiries_from_extract(ext, symbols=["FINNIFTY"])
    assert list(e["expiry"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-30"]
    assert list(e["kind"]) == ["weekly", "monthly"]
    assert e["expiring_option_turnover_rs_cr"].tolist() == pytest.approx([1.0, 0.5])
    e2 = rx.nse_expiries_from_extract(ext, symbols=["FINNIFTY"], traded_only=False)
    assert len(e2) == 3


def test_day_calendar_and_week_counts():
    days = pd.bdate_range("2024-01-01", "2024-01-12")
    exp = pd.DataFrame({"symbol": ["NIFTY", "FINNIFTY", "SENSEX", "NIFTY"],
                        "expiry": pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-05", "2024-01-11"])})
    cal = rx.day_calendar(days, exp, symbols=["NIFTY", "FINNIFTY", "SENSEX"])
    assert cal.loc["2024-01-04", "NIFTY"] == 1 and cal.loc["2024-01-04", "n_expiring"] == 1
    assert bool(cal.loc["2024-01-05", "any_bse"]) and not bool(cal.loc["2024-01-05", "any_nse"])
    wc = rx.week_counts(cal, days)
    assert wc.loc["2024-01-01", "days_no_expiry"] == 2          # Mon and Wed
    assert wc.loc["2024-01-01", "days_no_nse_expiry"] == 3      # the Sensex Friday counts as free of NSE expiries
    assert wc.loc["2024-01-01", "no_expiry_weekdays"] == "Mon;Wed"
    assert wc.loc["2024-01-08", "days_no_expiry"] == 4
    lab = rx.label_regimes(wc.index, [("x", "2024-01-08", "2024-02-01")])
    assert list(lab) == ["", "x"]
    s = rx.summarise_weeks(wc.assign(reg=lab), "reg")
    assert s.iloc[0]["weeks"] == 1 and s.iloc[0]["mean_no_expiry_days"] == 4


def test_bse_expiries_rule_and_holiday_rollback():
    days = pd.bdate_range("2023-05-15", "2023-06-02")
    days = days[days != pd.Timestamp("2023-05-26")]  # a Friday holiday
    sx_cal = pd.DataFrame({"fact_type": ["weekday_regime"], "date": ["2023-05-15"], "date_end": ["2025-01-03"],
                           "weekday": ["Fri"]})
    other = pd.DataFrame({"index": ["BANKEX"], "exchange": ["BSE"], "start": ["2023-05-15"], "end": ["2023-10-15"],
                          "weekday": ["Fri"]})
    b = rx.bse_expiries(days, sx_cal, other)
    sx = b[b["symbol"] == "SENSEX"]["expiry"].dt.strftime("%Y-%m-%d").tolist()
    assert sx == ["2023-05-19", "2023-05-25", "2023-06-02"]
    assert (b["symbol"] == "BANKEX").sum() == 3
    assert b["status"].str.startswith("reported").all()


# ---------------------------------------------------------------- R2 with a clean baseline


def _dm():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2025-01-06", "2025-03-28")
    dm = pd.DataFrame(index=days)
    dm["is_expiry"] = days.weekday == 3
    dm["regime"] = "single_weekly_thu"
    dm["weekday"] = days.weekday
    dm["rv_day"] = np.exp(rng.normal(0, 0.3, len(days)))
    dm.loc[days.weekday == 1, "rv_day"] *= 3.0  # another index's expiry day is more volatile
    return dm


def test_r2_week_relative_matches_published_function():
    dm = _dm()
    a = rx.r2_week_relative(dm, "rv_day", 2)
    b = rdl.week_relative(dm, "rv_day")
    assert np.allclose(a["rel"].to_numpy(), b["rel"].to_numpy())
    assert list(a["n_other"]) == list(b["n_other"])


def test_r2_table_clean_baseline_removes_other_expiry_days():
    dm = _dm()
    tues = set(dm.index[dm.index.weekday == 1])
    per = {"all": lambda w: w["regime"] == "single_weekly_thu"}
    t = rx.r2_table(dm, ["rv_day"], {"pub": (set(), 2), "clean": (tues, 2), "clean1": (tues | set(dm.index[dm.index.weekday == 0]), 1)}, per)
    pub, clean, c1 = (t[t["baseline"] == v].iloc[0] for v in ("pub", "clean", "clean1"))
    # removing the more volatile Tuesdays from the baseline raises Thursday's premium by about log(3) / 4
    assert clean["mean_rel"] - pub["mean_rel"] == pytest.approx(np.log(3) / 4, abs=0.12)
    assert clean["mean_baseline_days"] == pytest.approx(3.0)
    assert c1["mean_baseline_days"] == pytest.approx(2.0)


def test_two_day_difference():
    v = pd.Series([1.0, 2.0, 4.0, 8.0, 1.0], index=pd.bdate_range("2025-01-06", periods=5))
    d = rx.two_day_difference(v, 2, 3)
    assert d.iloc[0] == pytest.approx(np.log(4) - np.log(8))
    assert d.index[0] == pd.Timestamp("2025-01-06")


# ---------------------------------------------------------------- few-unit inference


def test_nonoverlap_sets_match_event_study_count_and_floor():
    ks = np.arange(10, 101)
    sets = rx.nonoverlap_sets(ks, 10)
    assert list(sets[0]) == [10, 30, 50, 70, 90]
    assert len(sets[0]) == es.n_independent(ks, 10)
    assert all(np.all(np.diff(s) >= 20) for s in sets)
    assert rx.randomization_p_nonoverlap(9.0, [0.1, -0.3, 0.5, 1.0]) == pytest.approx(1 / 5)
    assert rx.randomization_p_nonoverlap(0.2, [0.1, -0.3, 0.5, 1.0]) == pytest.approx(4 / 5)


def test_prediction_t_p():
    tp = [-1.0, 0.0, 1.0, 0.5, -0.5]
    assert np.isnan(rx.prediction_t_p(1.0, [0.1, 0.2]))
    assert rx.prediction_t_p(0.0, tp) == pytest.approx(1.0)
    p_big = rx.prediction_t_p(6.0, tp)
    sd = np.std(tp, ddof=1)
    from scipy import stats
    assert p_big == pytest.approx(2 * stats.t.sf(6.0 / (sd * np.sqrt(1 + 1 / 5)), 4))


def test_wild_cluster_bootstrap_size_and_power():
    rng = np.random.default_rng(3)
    ps = [rx.wild_cluster_p(rng.normal(size=20), rng.normal(size=20), n_boot=199, seed=i)[0] for i in range(150)]
    rej = np.mean(np.asarray(ps) < 0.05)
    assert rej < 0.12
    p, G = rx.wild_cluster_p(rng.normal(size=20), rng.normal(size=20) + 2.0, n_boot=199, seed=1)
    assert G == 10 and p < 0.05
    assert np.isnan(rx.wild_cluster_p([1.0, 2.0], [1.0, 2.0, 3.0])[0])


def test_trend_adjusted_change_separates_trend_and_jump():
    weeks = pd.date_range("2024-01-01", periods=60, freq="7D")
    t = np.arange(60) / 52.18
    y = pd.Series(0.5 * t, index=weeks)
    pre, post = y.iloc[:30], y.iloc[30:]
    c, se, p = rx.trend_adjusted_change(pre, post + 0.0)
    assert abs(c) < 1e-8
    c2, _, _ = rx.trend_adjusted_change(pre, post + 0.7)
    assert c2 == pytest.approx(0.7, abs=1e-8)


# ---------------------------------------------------------------- noise-robust estimators


def test_subsampled_kernel_and_parzen_basics():
    rng = np.random.default_rng(0)
    lp = np.cumsum(rng.normal(0, 0.001, (5, 31)), axis=1)
    r = np.diff(lp, axis=1)
    assert np.allclose(rx.rv_subsampled(lp, 1), rx.rv_from_logp(lp))
    assert np.allclose(rx.realized_kernel(r, 0), (r ** 2).sum(axis=1))
    assert rx.parzen(0.0) == pytest.approx(1.0)
    assert rx.parzen(0.25) == pytest.approx(0.71875)
    assert rx.parzen(0.75) == pytest.approx(2 * 0.25 ** 3)
    assert rx.parzen(1.2) == pytest.approx(0.0)
    rho, num, den = rx.autocorr1(np.array([[1.0, -1.0, 1.0, -1.0]]))
    assert rho[0] == pytest.approx(-3 / 4)


def test_estimators_on_simulated_noisy_prices():
    # SYNTHETIC: no noise -> all estimators close to the truth
    lp, iv = rx.simulate_noisy_windows(600, seed=11)
    r = np.diff(lp, axis=1)
    for v in (rx.rv_from_logp(lp), rx.rv_subsampled(lp, 5), rx.realized_kernel(r, 3), rx.pre_averaged(r, 6)):
        assert np.mean(v) / iv == pytest.approx(1.0, abs=0.08)
    # iid noise that inflates 1-minute RV by about 50%
    lp, iv = rx.simulate_noisy_windows(600, seed=12, noise_sd=2.58e-4)
    r = np.diff(lp, axis=1)
    b_rv1 = np.mean(rx.rv_from_logp(lp)) / iv - 1
    assert b_rv1 > 0.35
    for v in (rx.rv_subsampled(lp, 5), rx.realized_kernel(r, 3), rx.pre_averaged(r, 6)):
        assert abs(np.mean(v) / iv - 1) < 0.5 * b_rv1
    assert np.nanmean(rx.autocorr1(r)[0]) < -0.08
    # stale index: positive autocorrelation, 1-minute RV biased down, kernel and pre-averaging closer
    lp, iv = rx.simulate_noisy_windows(300, seed=13, n_stocks=10, update_prob=1 / 30)
    r = np.diff(lp, axis=1)
    b_rv1 = np.mean(rx.rv_from_logp(lp)) / iv - 1
    assert b_rv1 < -0.2
    assert np.nanmean(rx.autocorr1(r)[0]) > 0.1
    assert abs(np.mean(rx.realized_kernel(r, 3)) / iv - 1) < abs(b_rv1)
    assert abs(np.mean(rx.pre_averaged(r, 6)) / iv - 1) < abs(b_rv1)


def test_eod_measures_on_a_grid():
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 5e-4, (3, 375)), axis=1))
    close[1, 360] = np.nan  # a missing bar is forward-filled
    m = rx.eod_measures(close, pd.bdate_range("2025-01-06", periods=3))
    lp = np.log(pd.DataFrame(close).ffill(axis=1).to_numpy()[:, 344:375])
    assert np.allclose(m["last30_rv1"], (np.diff(lp, axis=1) ** 2).sum(axis=1))
    assert (m["last30_rk"] > 0).all()
    assert m.shape[0] == 3 and "midday30_rho1" in m


# ---------------------------------------------------------------- calendar change table


def test_calendar_change_table_marks_event_windows():
    ch = pd.DataFrame({"symbol": ["MIDCPNIFTY"], "kind": ["weekly"], "change": ["weekday"],
                       "last_old_expiry": [pd.Timestamp("2023-08-16")], "old_weekday": ["Wed"],
                       "first_new_expiry": [pd.Timestamp("2023-08-21")], "new_weekday": ["Mon"]})
    ev = pd.Series({"a": pd.Timestamp("2023-09-04"), "b": pd.Timestamp("2024-03-04"), "c": pd.Timestamp("2024-11-18"),
                    "d": pd.Timestamp("2025-09-01"), "e": pd.Timestamp("2026-08-03")})
    t = rx.calendar_change_table(ch, ev)
    row = t[(t["index"] == "Nifty Midcap Select") & t["change"].str.contains("to Mon")].iloc[0]
    assert str(row["date"]) == "2023-08-21"
    assert "a 6m" in row["inside_event_windows"] and "a 12m" in row["inside_event_windows"]
    bankex = t[(t["index"] == "Bankex") & t["change"].str.contains("Monday")].iloc[0]
    assert "a 6m" in bankex["inside_event_windows"] and bankex["status"] == "reported"
