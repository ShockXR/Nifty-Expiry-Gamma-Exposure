"""Unit tests for experiments/expiry_event_study.py (week-relative measure,
expiry detection from bhavcopy-like tables, placebo machinery)."""

import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "expiry_event_study.py"
_spec = importlib.util.spec_from_file_location("expiry_event_study", _PATH)
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)


# ---------------------------------------------------------------- week-relative


def _daily(values: dict) -> pd.Series:
    return pd.Series(values).rename(index=pd.Timestamp).sort_index().astype(float)


def test_week_relative_log_and_level():
    # Week of Mon 6 Jan 2025: Thursday is the target
    v = _daily({"2025-01-06": 1.0, "2025-01-07": 2.0, "2025-01-08": 4.0, "2025-01-09": 8.0, "2025-01-10": 1.0})
    tg = es.weekday_targets(v.index, 3)
    wr = es.week_relative(v, tg)
    expected = np.log(8.0) - np.mean(np.log([1.0, 2.0, 4.0, 1.0]))
    assert len(wr) == 1
    assert wr.index[0] == pd.Timestamp("2025-01-06")
    assert wr["rel"].iloc[0] == pytest.approx(expected)
    assert wr["n_other"].iloc[0] == 4
    lv = es.week_relative(v, tg, log=False)
    assert lv["rel"].iloc[0] == pytest.approx(8.0 - 2.0)


def test_week_relative_holiday_min_other_and_exclude():
    # Week 1: Thursday missing (holiday) -> no value. Week 2: only one other day -> dropped.
    # Week 3: excluding Wednesday from the baseline.
    v = _daily({
        "2025-01-06": 1.0, "2025-01-07": 1.0, "2025-01-08": 1.0, "2025-01-10": 1.0,
        "2025-01-15": 2.0, "2025-01-16": 3.0,
        "2025-01-20": 1.0, "2025-01-21": 1.0, "2025-01-22": 100.0, "2025-01-23": np.e, "2025-01-24": 1.0,
    })
    tg = es.weekday_targets(v.index, 3)
    wr = es.week_relative(v, tg)
    assert list(wr.index) == [pd.Timestamp("2025-01-20")]
    full = wr["rel"].iloc[0]
    assert full == pytest.approx(1.0 - np.log(100.0) / 4)
    ex = es.week_relative(v, tg, exclude={pd.Timestamp("2025-01-22")})
    assert ex["rel"].iloc[0] == pytest.approx(1.0)
    strict = es.week_relative(v, tg, min_other=5)  # week 3 has only 4 other days
    assert strict.empty


def test_week_relative_cancels_week_level_shifts():
    # Multiplying a whole week by a constant must not change the measure.
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    v = pd.Series(np.exp(rng.normal(size=len(days))), index=days)
    scale = pd.Series(np.exp(rng.normal(scale=2, size=len(days))), index=days)
    wk_scale = scale.groupby(es.week_start(days)).transform("first")
    tg = es.weekday_targets(days, 2)
    a = es.week_relative(v, tg)["rel"]
    b = es.week_relative(v * wk_scale, tg)["rel"]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_date_targets_drop_weeks_with_two_dates():
    days = pd.bdate_range("2025-04-21", "2025-05-09")
    t = es.date_targets(days, pd.to_datetime(["2025-04-24", "2025-04-30", "2025-05-02", "2025-05-08"]))
    # 30 Apr and 2 May fall in the same week -> that week is dropped
    assert list(t.index) == [pd.Timestamp("2025-04-21"), pd.Timestamp("2025-05-05")]


# ---------------------------------------------------------------- expiry detection


def _bhav_rows(symbol, trade_days, expiries, fut_expiries):
    """Bhavcopy-like rows: on each trade day, options for every expiry not yet passed."""
    rows = []
    for d in trade_days:
        for e in expiries:
            if e >= d:
                rows.append({"date": d, "symbol": symbol, "inst": "opt", "expiry": e})
        for e in fut_expiries:
            if e >= d:
                rows.append({"date": d, "symbol": symbol, "inst": "fut", "expiry": e})
    return rows


def test_expiry_detection_weekday_change_holiday_shift_and_stop():
    days = pd.bdate_range("2023-07-03", "2024-01-31")
    days = days.drop(pd.Timestamp("2023-08-17"))  # pretend Thursday 17 Aug is a holiday
    thursdays = [d for d in days if d.weekday() == 3 and d < pd.Timestamp("2023-09-04")]
    thursdays.append(pd.Timestamp("2023-08-16"))  # holiday shift to Wednesday
    wednesdays = [d for d in days if d.weekday() == 2 and pd.Timestamp("2023-09-04") <= d <= pd.Timestamp("2023-11-15")]
    bn_exp = sorted(thursdays + wednesdays)
    # monthly futures: last Thursday of Jul, Aug, Sep, Oct, Nov
    fut = pd.to_datetime(["2023-07-27", "2023-08-31", "2023-09-28", "2023-10-26", "2023-11-30"])
    n_exp = sorted(set([d for d in days if d.weekday() == 3]) | set(fut))
    bhav = pd.DataFrame(_bhav_rows("BANKNIFTY", days, bn_exp + list(fut), fut) + _bhav_rows("NIFTY", days, n_exp, fut))
    exp = es.expiry_table(bhav)
    bn = exp[exp["symbol"] == "BANKNIFTY"].set_index("expiry")
    assert bn.loc[pd.Timestamp("2023-08-31"), "kind"] == "monthly"
    assert bn.loc[pd.Timestamp("2023-08-24"), "kind"] == "weekly"
    assert pd.Timestamp("2023-08-17") not in bn.index  # holiday: no trading, no expiry
    changes, flagged = es.detect_schedule_changes(exp, data_end=days.max())
    wk = changes[(changes["symbol"] == "BANKNIFTY") & (changes["kind"] == "weekly")]
    ch = wk[wk["change"] == "weekday"]
    assert len(ch) == 1
    assert ch["first_new_expiry"].iloc[0] == pd.Timestamp("2023-09-06")
    assert ch["old_weekday"].iloc[0] == "Thu" and ch["new_weekday"].iloc[0] == "Wed"
    stop = wk[wk["change"] == "stopped"]
    assert len(stop) == 1 and stop["last_old_expiry"].iloc[0] == pd.Timestamp("2023-11-15")
    f = flagged.set_index(["symbol", "expiry"])
    assert bool(f.loc[("BANKNIFTY", pd.Timestamp("2023-08-16")), "shifted"])
    assert not bool(f.loc[("BANKNIFTY", pd.Timestamp("2023-08-24")), "shifted"])
    # Nifty: no change, and BN monthly stays on Thursday (no monthly change reported)
    assert changes[changes["symbol"] == "NIFTY"].empty
    assert changes[(changes["symbol"] == "BANKNIFTY") & (changes["kind"] == "monthly")].empty


def test_rule_expiries_roll_back_and_month_label():
    days = pd.bdate_range("2025-04-01", "2025-05-30").drop(pd.to_datetime(["2025-04-10", "2025-05-01"]))
    e = es.rule_expiries(days, 3, "2025-04-01", "2025-05-30", "NIFTY").set_index("expiry")
    assert pd.Timestamp("2025-04-09") in e.index and bool(e.loc["2025-04-09", "shifted"])
    # 1 May (holiday) rolls back to 30 Apr: a weekly of May's first week, not April's monthly
    assert e.loc["2025-04-30", "kind"] == "weekly"
    assert e.loc["2025-04-24", "kind"] == "monthly"
    assert e.loc["2025-05-29", "kind"] == "monthly"


def test_read_bhavcopy_layouts_and_turnover(tmp_path):
    legacy = ("INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,"
              "OPEN_INT,CHG_IN_OI,TIMESTAMP\n"
              "OPTIDX,NIFTY,04-Jan-2024,21500,CE,10,12,8,10,10,4,43.02,0,0,04-JAN-2024\n"
              "FUTIDX,NIFTY,25-Jan-2024,0,XX,21600,21650,21550,21600,21600,2,21.6,0,0,04-JAN-2024\n"
              "OPTSTK,RELIANCE,25-Jan-2024,2500,CE,1,1,1,1,1,1,1,0,0,04-JAN-2024\n")
    p1 = tmp_path / "fo_20240104.csv.gz"
    with gzip.open(p1, "wt") as fh:
        fh.write(legacy)
    b = es.fill_legacy_lots(es.read_bhavcopy_file(p1))
    assert set(b["symbol"]) == {"NIFTY"}
    opt = b[b["inst"] == "opt"].iloc[0]
    # 4 contracts x 50 x (21500 + 11.0 average premium) = 4,302,200 rupees = 43.022 lakh
    assert opt["lot"] == 50
    t = es.daily_turnover(b).iloc[0]
    assert t["opt_premium"] == pytest.approx(43.02e5 - 21500 * 200)
    assert t["fut_notional"] == pytest.approx(21.6e5)
    assert t["share_premium_0dte"] == pytest.approx(1.0)
    udiff = ("TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,"
             "OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,"
             "ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
             "2025-01-02,2025-01-02,FO,NSE,IDO,1,,BANKNIFTY,,2025-02-27,2025-02-27,55300.00,CE,X,170,170.35,170,170.35,"
             "170.35,170,51605.55,354,780,780,2,3328210.50,2,F1,30,,,,,\n")
    p2 = tmp_path / "fo_20250102.csv.gz"
    with gzip.open(p2, "wt") as fh:
        fh.write(udiff)
    u = es.read_bhavcopy_file(p2)
    assert u["lot"].iloc[0] == 30 and u["contracts"].iloc[0] == 2
    tu = es.daily_turnover(u).iloc[0]
    assert tu["opt_premium"] == pytest.approx(3328210.50 - 55300 * 60)

    # lot-size history: one run per (lot, source), legacy lots marked as inferred
    both = pd.concat([es.read_bhavcopy_file(p1), u], ignore_index=True)
    hist = es.lot_history(es.fill_legacy_lots(both), both["lot"].notna().to_numpy())
    nifty = hist[hist["symbol"] == "NIFTY"].iloc[0]
    bank = hist[hist["symbol"] == "BANKNIFTY"].iloc[0]
    assert (nifty["lot"], nifty["source"].startswith("inferred")) == (50, True)
    assert (bank["lot"], bank["source"].startswith("printed")) == (30, True)


# ---------------------------------------------------------------- inference and placebo machinery


def test_placebo_positions_avoid_real_events():
    ks = es.placebo_event_indices(200, None, 10, blocked_idx=[100])
    assert len(ks) > 0
    assert np.all((ks + 10 <= 100) | (ks - 10 >= 100))
    assert ks.min() >= 10 and ks.max() <= 190
    assert 100 not in ks
    assert es.n_independent(np.arange(10, 91), 10) == 5  # 10, 30, 50, 70, 90


def test_placebo_detects_planted_break_but_not_noise():
    rng = np.random.default_rng(7)
    y = rng.normal(size=400)
    y_break = y.copy()
    y_break[300:] += 1.5
    W = 26
    t_obs = es.welch_t(y_break[300 - W:300], y_break[300:300 + W])
    ks = es.placebo_event_indices(400, None, W, blocked_idx=[300])
    tp = es.placebo_t_stats(y_break, ks, W)
    p, n = es.randomization_p(t_obs, tp, es.n_independent(ks, W))
    assert n == len(ks) and p < 0.02
    # same machinery on pure noise at a "fake" event: not significant
    t0 = es.welch_t(y[300 - W:300], y[300:300 + W])
    p0, _ = es.randomization_p(t0, es.placebo_t_stats(y, ks, W), es.n_independent(ks, W))
    assert p0 > 0.05
    # too few independent placebo windows -> no p-value
    assert np.isnan(es.randomization_p(t_obs, tp, n_indep=2)[0])


def test_circular_bootstrap_centred_and_compare_detects_shift():
    rng = np.random.default_rng(3)
    x = np.r_[5.0, rng.normal(size=14), -5.0]  # extreme first and last weeks
    draws = es.mbb_means(x, 4, 20000, np.random.default_rng(0))
    assert abs(draws.mean() - x.mean()) < 0.02
    pre = rng.normal(size=40)
    post = rng.normal(size=40) + 1.0
    r = es.compare(pre, post, n_boot=999)
    assert r["boot_lo"] < r["diff"] < r["boot_hi"]
    assert r["boot_lo"] > 0 and r["hac_p"] < 0.01 and r["mwu_p"] < 0.01
    flat = es.compare(pre, rng.normal(size=40), n_boot=999)
    assert flat["boot_lo"] < 0 < flat["boot_hi"]


def test_bh_qvalues():
    q = es.bh_qvalues([0.01, 0.04, 0.03, np.nan, 0.5])
    assert np.isnan(q[3])
    # sorted p: 0.01, 0.03, 0.04, 0.5 -> raw 0.04, 0.06, 0.0533, 0.5 -> step-up minimum
    assert q[0] == pytest.approx(0.04)
    assert q[1] == pytest.approx(0.04 * 4 / 3)
    assert q[2] == pytest.approx(0.04 * 4 / 3)
    assert q[4] == pytest.approx(0.5)


def test_day_level_did_regression_recovers_planted_effect():
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2024-01-01", "2024-12-27")
    post = days >= pd.Timestamp("2024-07-01")
    wk_level = pd.Series(rng.normal(size=len(days)), index=days).groupby(es.week_start(days)).transform("first")
    a = wk_level + 0.4 * (days.weekday == 2) - 0.5 * ((days.weekday == 2) & post) + rng.normal(scale=0.3, size=len(days))
    b = wk_level + 0.3 * (days.weekday == 3) + rng.normal(scale=0.3, size=len(days))
    weeks = es.week_start(days).unique()
    r = es.did_day_regression({"A": a, "B": b}, {"A": days[days.weekday == 2], "B": days[days.weekday == 3]},
                              weeks[weeks < pd.Timestamp("2024-07-01")], weeks[weeks >= pd.Timestamp("2024-07-01")])
    assert r["did"] == pytest.approx(-0.5, abs=0.2)
    assert r["p"] < 0.01


def test_detect_closing_auction():
    days = pd.bdate_range("2026-07-01", "2026-08-31")
    flat = np.where(days >= pd.Timestamp("2026-08-03"), 14, 0)
    flat[3] = 9  # one isolated frozen day before the change must not trigger it
    df = pd.DataFrame({"flat_bars_last20": flat}, index=days)
    assert es.detect_closing_auction(df) == pd.Timestamp("2026-08-03")
    assert es.detect_closing_auction(df.iloc[:-15]) is None  # run too short
