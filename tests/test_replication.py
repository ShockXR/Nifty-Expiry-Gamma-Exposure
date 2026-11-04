"""Fast unit tests for the helpers used by experiments/replicate_paper.py."""

import numpy as np
import pandas as pd

from niftygex import replication as rp
from niftygex.connectedness import connectedness


def _bars(days, start="09:15", end="15:29"):
    rows = []
    for d in pd.DatetimeIndex(days):
        for t in pd.date_range(f"{d.date()} {start}", f"{d.date()} {end}", freq="1min"):
            rows.append({"timestamp": t, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    df = pd.DataFrame(rows)
    df["date"] = df["timestamp"].dt.normalize()
    df["tod"] = df["timestamp"].dt.strftime("%H:%M")
    return df


def test_session_rules_count_minutes():
    assert len(rp.session_minutes("375_full")) == 375
    assert len(rp.session_minutes("360_drop_last15")) == 360
    assert rp.session_minutes("360_drop_last15")[-1] == "15:14"
    assert rp.session_minutes("360_drop_first15")[0] == "09:30"


def test_trim_drops_weekend_sessions_and_out_of_session_bars():
    bars = pd.concat([_bars(["2024-03-01"]), _bars(["2024-03-02"], "09:15", "10:00"), _bars(["2023-11-13"], "18:15", "19:15")])
    t = rp.trim_session(bars, "360_drop_last15")
    assert set(t["date"].dt.date.astype(str)) == {"2024-03-01"}  # Saturday special session dropped
    assert len(t) == 360


def test_continuous_returns_put_overnight_gap_in_first_minute():
    close = np.array([[100.0, 101.0, 102.0], [105.0, 105.0, np.nan]])
    r = rp.continuous_log_returns(close)
    assert np.isnan(r[0, 0])
    assert np.isclose(r[1, 0], np.log(105 / 102))  # overnight return sits in day 2, minute 1
    assert np.isnan(r[1, 2])
    # 2 days x 3 minutes gives 6 - 1 returns, minus the missing bar
    assert np.isfinite(r).sum() == 4
    r2 = rp.continuous_log_returns(close, overnight=False)
    assert np.isnan(r2[:, 0]).all()
    # another reference price for the overnight gap (for example NSE's official close) changes only minute 1
    r3 = rp.continuous_log_returns(close, prev_ref=np.array([103.0, 999.0]))
    assert np.isclose(r3[1, 0], np.log(105 / 103))
    assert np.isnan(r3[0, 0])
    assert np.allclose(r3[:, 1:], r[:, 1:], equal_nan=True)


def test_minute_grid_marks_missing_bars():
    bars = _bars(["2024-03-01", "2024-03-04"])
    bars = bars[~((bars["date"] == "2024-03-04") & (bars["tod"] == "10:00"))]
    g = rp.minute_grid(bars, "360_drop_last15")
    assert g["close"].shape == (2, 360)
    assert np.isnan(g["close"][1, g["minutes"].index("10:00")])
    assert np.isfinite(g["close"]).sum() == 719


def test_expiry_calendar_uses_last_trade_and_futures_for_monthly():
    days = pd.bdate_range("2024-01-22", "2024-02-02")
    rows = []
    for d in days:
        # monthly futures and options expiring Thursday 25 Jan; a weekly option expiring Wednesday 31 Jan
        if d <= pd.Timestamp("2024-01-25"):
            rows += [("BANKNIFTY", "FUTIDX", "2024-01-25", d), ("BANKNIFTY", "OPTIDX", "2024-01-25", d)]
        if d <= pd.Timestamp("2024-01-31"):
            rows.append(("BANKNIFTY", "OPTIDX", "2024-01-31", d))
        # a contract first listed for 28 Feb then realigned: it stops trading long before 28 Feb
        if d <= pd.Timestamp("2024-01-24"):
            rows.append(("BANKNIFTY", "FUTIDX", "2024-02-28", d))
        rows.append(("BANKNIFTY", "FUTIDX", "2024-02-27", d))
    c = pd.DataFrame(rows, columns=["SYMBOL", "INSTRUMENT", "expiry", "date"])
    c["expiry"] = pd.to_datetime(c["expiry"])
    cal = rp.expiry_calendar(c).set_index("expiry")
    assert list(cal.index) == [pd.Timestamp("2024-01-25"), pd.Timestamp("2024-01-31")]
    assert cal.loc["2024-01-25", "is_monthly"]  # futures expiry, although not the month's last expiry
    assert not cal.loc["2024-01-31", "is_monthly"]  # a weekly, although it is the month's last expiry


def test_week_relative_compares_with_own_week():
    days = pd.bdate_range("2024-03-04", "2024-03-15")
    vals = pd.Series(10.0, index=days)
    vals[pd.Timestamp("2024-03-07")] = 20.0  # Thursday doubles
    vals[days >= "2024-03-11"] *= 5  # whole second week is louder; Thursday equal to its week
    target = pd.Series(days.weekday == 3, index=days)
    wr = rp.week_relative(vals, target)
    assert len(wr) == 2
    assert np.isclose(wr["rel_log"].iloc[0], np.log(2.0))
    assert np.isclose(wr["rel_log"].iloc[1], 0.0)


def test_segment_var_matches_plain_var_on_one_segment_and_ignores_cross_segment_lags():
    rng = np.random.default_rng(0)
    T = 3000
    y = np.zeros((T, 2))
    e = rng.standard_normal((T, 2))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 1][::-1] + e[t]
    one = rp.connectedness_segments(y, np.zeros(T), 2, 10)
    plain = connectedness(y, 2, 10)
    assert np.allclose(one.table, plain.table)
    # two segments: no row uses a lag from the other segment
    seg = np.r_[np.zeros(1500), np.ones(1500)]
    X, Z = rp.var_design(y, seg, 2)
    assert len(Z) == T - 4
    assert rp.var_lag_aic(y, np.zeros(T), 5) in (1, 2)


def test_segment_bootstrap_resamples_whole_segments():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((600, 2))
    y[:, 1] += 0.8 * y[:, 0]
    seg = np.repeat(np.arange(10), 60)
    draws = rp.segment_bootstrap_connectedness(y, seg, 1, 5, n_boot=30, seed=3)
    assert draws.shape == (30,) and np.all(np.isfinite(draws))
    point = rp.connectedness_segments(y, seg, 1, 5).paper_scale_total
    assert np.percentile(draws, 2.5) < point < np.percentile(draws, 97.5)


def test_half_life_and_profile_ci():
    assert np.isclose(rp.half_life(0.5 ** (1 / 100)), 100.0)
    prof = pd.DataFrame({"half_life": [10, 20, 40, 80, 160], "loglik": [-10.0, -2.0, 0.0, -1.0, -8.0]})
    ci = rp.profile_ci(prof, 0.0)
    # LR statistics are 20, 4, 0, 2, 16 against a 3.84 cut-off: the interval runs from just above 20 to just above 80
    assert 20 < ci["lo"] < 40 and 80 < ci["hi"] < 160
    assert not ci["lo_open"] and not ci["hi_open"]


def test_lot_size_and_premium_from_notional():
    day = pd.DataFrame({
        "SYMBOL": ["NIFTY"] * 3, "INSTRUMENT": ["FUTIDX", "OPTIDX", "OPTIDX"], "STRIKE_PR": [0, 22000, 22100],
        "OPEN": [22050, 100, 50], "HIGH": [22050, 100, 50], "LOW": [22050, 100, 50], "CLOSE": [22050, 100, 50],
        "CONTRACTS": [10, 4, 2], "date": pd.Timestamp("2024-05-15"),
    })
    lot = 25
    day["VAL_INLAKH"] = [10 * lot * 22050 / 1e5, 4 * lot * (22000 + 100) / 1e5, 2 * lot * (22100 + 50) / 1e5]
    out = rp.daily_fo_volume(day).iloc[0]
    assert out["lot_size"] == 25
    assert np.isclose(out["opt_premium_cr"], (4 * lot * 100 + 2 * lot * 50) / 1e7)
    assert out["opt_qty"] == 6 * lot
