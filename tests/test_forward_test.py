"""Fast unit tests for scripts/forward_test.py (synthetic data only, no real files).

Covers the pre-registered decision rules (Holm over H1 and H2, confirm / fail to confirm / contradict,
pending when the other test has not stopped), the stopping rules (count, date cap, expiry-weekday change,
dry run), and the window filter (nothing from outside --start/--end reaches a test).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "forward_test.py"
_spec = importlib.util.spec_from_file_location("forward_test", _PATH)
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)


# ---------------------------------------------------------------- synthetic market


def synth_market(start="2025-01-01", end="2025-12-31", seed=0, neg_share=0.35):
    """Weekday trading days, Tuesday expiries (Monday when Tuesday is a holiday), minute closes as a random
    walk, a GEX panel with random sign. Returns dm, panel, grid in the layouts the script reads."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    days = days[~days.isin(pd.DatetimeIndex(["2025-04-15", "2025-08-26"]))]  # two Tuesday holidays
    n = len(days)
    level = 20000 * np.exp(np.cumsum(rng.normal(0, 4e-3, n)))
    close = level[:, None] * np.exp(np.cumsum(rng.normal(0, 3e-4, (n, ft.N_SLOTS)), axis=1))
    c = pd.DataFrame(close, index=days, columns=range(ft.N_SLOTS))
    grid = {"close": c, "open": c.copy(), "high": c * (1 + 1e-6), "low": c * (1 - 1e-6)}
    trading = set(days)
    is_exp = np.array([ft.is_tuesday_rule_expiry(d, trading) and (d.weekday() == 1 or d.weekday() == 0) for d in days])
    dm = pd.DataFrame({"is_expiry": is_exp, "regime": ft.TUE_REGIME, "weekday": days.weekday}, index=days)
    dm["rv_last_hour"] = ft.rv_between(c, ft.SLOT_1429, ft.SLOT_1529)
    sign = np.where(rng.random(n) < neg_share, -1.0, 1.0)
    panel = pd.DataFrame({"gross_gex": rng.uniform(1e11, 1e12, n), "net_gex_participant": sign * rng.uniform(1e9, 1e10, n)},
                         index=days)
    return dm, panel, grid


def _numbers(res):
    out = {}
    for key in ("h1", "h1b", "h2", "h2b"):
        for k, v in res.get(key, {}).items():
            if isinstance(v, (int, float, np.floating)):
                out[f"{key}.{k}"] = float(v)
    return out


# ---------------------------------------------------------------- decision rules


def test_holm_two_hypotheses_hand_worked():
    assert ft.holm({"a": 0.01, "b": 0.04}) == pytest.approx({"a": 0.02, "b": 0.04})
    assert ft.holm({"a": 0.03, "b": 0.04}) == pytest.approx({"a": 0.06, "b": 0.06})
    assert ft.holm({"a": 0.30, "b": 0.001}) == pytest.approx({"a": 0.30, "b": 0.002})


def _ev(p):
    return {"status": "evaluated", "p_predicted": p, "p_opposite": 1 - p}


def test_classify_both_evaluated():
    d = ft.classify({"H1": _ev(0.01), "H2": _ev(0.04)})
    assert d["H1"]["verdict"] == "confirm" and d["H1"]["holm_p"] == pytest.approx(0.02)
    assert d["H2"]["verdict"] == "confirm" and d["H2"]["holm_p"] == pytest.approx(0.04)
    d = ft.classify({"H1": _ev(0.03), "H2": _ev(0.04)})  # both pass at 0.05 but neither at 0.025
    assert d["H1"]["verdict"] == d["H2"]["verdict"] == "fail to confirm"
    d = ft.classify({"H1": _ev(0.99), "H2": _ev(0.001)})  # H1 significant the wrong way
    assert d["H1"]["verdict"] == "contradict" and d["H2"]["verdict"] == "confirm"
    d = ft.classify({"H1": _ev(0.20), "H2": _ev(0.94)})  # neither direction significant
    assert d["H1"]["verdict"] == d["H2"]["verdict"] == "fail to confirm"


def test_classify_not_evaluable_counts_as_p_one():
    d = ft.classify({"H1": {"status": "not evaluable"}, "H2": _ev(0.03)})
    assert d["H1"]["verdict"] == "not evaluable"
    assert d["H2"]["holm_p"] == pytest.approx(0.06) and d["H2"]["verdict"] == "fail to confirm"
    assert ft.classify({"H1": {"status": "not evaluable"}, "H2": _ev(0.02)})["H2"]["verdict"] == "confirm"


def test_classify_when_the_other_test_has_not_stopped():
    nr = {"status": "not reached"}
    assert ft.classify({"H1": nr, "H2": _ev(0.02)})["H2"]["verdict"] == "confirm"
    assert ft.classify({"H1": nr, "H2": _ev(0.04)})["H2"]["verdict"].startswith("pending")
    assert ft.classify({"H1": nr, "H2": _ev(0.20)})["H2"]["verdict"] == "fail to confirm"
    assert ft.classify({"H1": nr, "H2": _ev(0.99)})["H2"]["verdict"] == "contradict"
    assert ft.classify({"H1": nr, "H2": _ev(0.02)})["H1"]["verdict"] == "stopping point not reached"


def test_classify_data_check_failed_is_rerun_not_final():
    d = ft.classify({"H1": {"status": "data check failed"}, "H2": _ev(0.04)})
    assert d["H1"]["verdict"].startswith("not run") and d["H2"]["verdict"].startswith("pending")


def test_classify_secondary():
    assert ft.classify_secondary(0.01, 0.99).startswith("supports")
    assert ft.classify_secondary(0.99, 0.01).startswith("opposite")
    assert ft.classify_secondary(0.30, 0.70).startswith("does not support")


def test_h1_test_direction_and_one_sided_p():
    rng = np.random.default_rng(1)
    n = 400
    s = pd.DataFrame({"r_rod_bp": rng.normal(0, 60, n), "neg": (rng.random(n) < 0.3).astype(float)},
                     index=pd.bdate_range("2027-01-01", periods=n))
    s["r_last_bp"] = -0.3 * s["r_rod_bp"] * s["neg"] + rng.normal(0, 5, n)
    t = ft.h1_test(s, "r_last_bp")
    assert t["coef_r_rod_x_neg"] < 0 and t["p_predicted"] < 1e-6
    assert t["p_predicted"] == pytest.approx(norm.cdf(t["z"]))
    assert t["p_predicted"] + t["p_opposite"] == pytest.approx(1.0)
    assert t["p_two_sided_statsmodels"] == pytest.approx(2 * min(t["p_predicted"], t["p_opposite"]))
    s["r_last_bp"] = 0.3 * s["r_rod_bp"] * s["neg"] + rng.normal(0, 5, n)
    t = ft.h1_test(s, "r_last_bp")
    assert t["p_opposite"] < 1e-6
    assert ft.classify({"H1": {"status": "evaluated", **t}, "H2": _ev(0.5)})["H1"]["verdict"] == "contradict"


# ---------------------------------------------------------------- stopping rules


def _h1_days(start, end, every):
    days = pd.bdate_range(start, end)
    return pd.DataFrame({"r_rod_bp": 1.0, "r_last_bp": 1.0, "neg": (np.arange(len(days)) % every == 0).astype(float)},
                        index=days)


def test_h1_stops_on_the_130th_negative_day():
    s = _h1_days(ft.FORWARD_START, "2029-06-30", every=3)
    st = ft.h1_stopping(s, pd.Timestamp("2029-06-30"), True, s.index.max())
    assert st["reached"] and st["stop_day"] == s.index[3 * 129]  # negatives at 0, 3, 6, ...
    assert int((s.loc[: st["stop_day"], "neg"] == 1).sum()) == ft.H1_STOP_NEG_DAYS


def test_h1_cap_and_not_reached():
    s = _h1_days(ft.FORWARD_START, "2030-01-15", every=20)  # far fewer than 130 negatives by the cap
    st = ft.h1_stopping(s, pd.Timestamp("2030-01-15"), True, s.index.max())
    assert st["reached"] and st["stop_day"] == ft.H1_CAP and "cap" in st["reason"]
    early = s[s.index <= "2028-06-30"]
    st = ft.h1_stopping(early, pd.Timestamp("2028-06-30"), True, early.index.max())
    assert not st["reached"] and st["stop_day"] is None
    # --end past the cap but the data stop months before it: not reached
    st = ft.h1_stopping(early, pd.Timestamp("2030-01-15"), True, early.index.max())
    assert not st["reached"] and "data end" in st["reason"]


def test_h1_dry_run_uses_window_end():
    s = _h1_days("2025-01-01", "2025-12-31", every=20)
    st = ft.h1_stopping(s, pd.Timestamp("2025-12-31"), False, s.index.max())
    assert st["reached"] and st["stop_day"] == pd.Timestamp("2025-12-31") and "dry run" in st["reason"]


def _weeks(first_tuesday, n):
    dates = pd.DatetimeIndex([pd.Timestamp(first_tuesday) + pd.Timedelta(weeks=i) for i in range(n)])
    return pd.DataFrame({"week": [str(d.to_period("W-SUN")) for d in dates], "date": dates, "regime": ft.TUE_REGIME,
                         "weekday": 1, "rel": 0.1, "n_other": 4, "week_friday": dates + pd.Timedelta(days=3),
                         "tuesday_rule": True})


def test_h2_stops_at_100_weeks():
    w = _weeks(ft.FORWARD_START, 120)
    st, sel = ft.h2_stopping(w, None, pd.Timestamp("2029-03-01"), True, pd.Timestamp("2029-03-01"))
    assert st["reached"] and len(sel) == ft.H2_STOP_WEEKS and sel["date"].iloc[-1] == w["date"].iloc[99]
    assert st["stop_day"] == w["week_friday"].iloc[99]


def test_h2_stops_at_an_expiry_weekday_change():
    w = _weeks(ft.FORWARD_START, 60)
    change = w["date"].iloc[40] + pd.Timedelta(days=2)  # a Thursday expiry in week 41
    st, sel = ft.h2_stopping(w, change, pd.Timestamp("2028-03-01"), True, pd.Timestamp("2028-03-01"))
    assert st["reached"] and len(sel) == 40 and sel["date"].max() < change and "Tuesday rule" in st["reason"]


def test_h2_cap_incomplete_weeks_and_not_reached():
    w = _weeks(ft.FORWARD_START, 60)
    st, sel = ft.h2_stopping(w, None, pd.Timestamp("2027-06-30"), True, pd.Timestamp("2027-06-30"))
    assert not st["reached"] and len(sel) == 0
    assert st["counting_weeks_so_far"] == int((w["week_friday"] <= "2027-06-30").sum())
    # a week whose Friday is after --end does not count yet
    end = w["date"].iloc[10] + pd.Timedelta(days=1)  # the Wednesday of week 11
    st, _ = ft.h2_stopping(w, None, end, True, end)
    assert st["counting_weeks_so_far"] == 10
    w2 = _weeks("2028-06-06", 40)  # runs past the cap of 31 Dec 2028
    st, sel = ft.h2_stopping(w2, None, pd.Timestamp("2029-03-01"), True, pd.Timestamp("2029-03-01"))
    assert st["reached"] and "cap" in st["reason"] and sel["week_friday"].max() <= ft.H2_CAP


def test_tuesday_rule_with_holiday_rollback():
    days = set(pd.bdate_range("2025-03-01", "2025-03-31")) - {pd.Timestamp("2025-03-11"), pd.Timestamp("2025-03-17"),
                                                              pd.Timestamp("2025-03-18")}
    assert ft.is_tuesday_rule_expiry(pd.Timestamp("2025-03-04"), days)       # a Tuesday
    assert ft.is_tuesday_rule_expiry(pd.Timestamp("2025-03-10"), days)       # Monday, Tuesday 11th a holiday
    assert not ft.is_tuesday_rule_expiry(pd.Timestamp("2025-03-24"), days)   # Monday, Tuesday 25th trades
    assert ft.is_tuesday_rule_expiry(pd.Timestamp("2025-03-14"), days)       # Friday, Mon 17 and Tue 18 holidays
    assert not ft.is_tuesday_rule_expiry(pd.Timestamp("2025-03-06"), days)   # a Thursday


# ---------------------------------------------------------------- measures


def test_rv_between_hand_worked_and_missing():
    c = pd.DataFrame([[100.0, 101.0, 100.0, 102.0], [100.0, np.nan, 100.0, 100.0]], columns=range(4))
    expected = np.log(1.01) ** 2 + np.log(100 / 101) ** 2 + np.log(1.02) ** 2
    rv = ft.rv_between(c, 0, 3)
    assert rv.iloc[0] == pytest.approx(expected) and np.isnan(rv.iloc[1])


def test_h2_weeks_equals_r2_week_relative_on_complete_data():
    dm, _, grid = synth_market()
    w = ft.h2_weeks(dm["rv_last_hour"], dm, dm.index.min(), dm.index.max())
    ref = ft.rdl.week_relative(dm, "rv_last_hour")
    m = w.merge(ref, on="date", suffixes=("", "_ref"))
    assert len(m) == len(ref) == len(w)
    assert np.allclose(m["rel"], m["rel_ref"])


# ---------------------------------------------------------------- the window filter does not leak


def test_window_filter_does_not_leak_days_from_outside():
    dm, panel, grid = synth_market()
    start, end = pd.Timestamp("2025-03-04"), pd.Timestamp("2025-10-29")  # a Tuesday and a Wednesday
    sx = set(d for d in dm.index if d.weekday() == 3)
    res = ft.analyse(dm, panel, grid, start, end, forward=False, sensex_days=sx)
    assert res["h1"]["status"] == res["h2"]["status"] == "evaluated"
    h1, h2 = res["_h1_days"], res["_h2_weeks"]
    window_days = dm.index[(dm.index >= start) & (dm.index <= end)]
    assert h1.index.min() == start and h1.index.max() == end and len(h1) == len(window_days)
    assert h2["date"].min() >= start and h2["date"].max() <= end
    first = h2.iloc[0]  # week of Mon 3 Mar: Monday is outside the window, so only Wed to Fri form the baseline
    assert first["date"] == start and first["n_other_1415_1515"] == 3
    assert pd.Timestamp("2025-10-28") not in set(h2["date"])  # its week ends after --end: incomplete, not counted

    # Scramble everything outside the window that is not a legitimate day t-1 input (the previous close at
    # 15:29 and the GEX of the evening before the first day). The results must not move.
    rng = np.random.default_rng(7)
    g2 = {k: v.copy() for k, v in grid.items()}
    dm2, panel2 = dm.copy(), panel.copy()
    prev = dm.index[dm.index < start].max()
    before, after = dm.index < start, dm.index > end
    keep = g2["close"].loc[prev, ft.SLOT_1529]
    for k in g2:
        g2[k].loc[before | after] = g2[k].loc[before | after] * np.exp(rng.normal(0, 0.01, g2[k].loc[before | after].shape))
    g2["close"].loc[prev, ft.SLOT_1529] = keep
    dm2.loc[before | after, "rv_last_hour"] *= rng.uniform(0.2, 5, int((before | after).sum()))
    flip = after | (dm.index < prev)
    panel2.loc[flip, "net_gex_participant"] *= -1
    res2 = ft.analyse(dm2, panel2, g2, start, end, forward=False, sensex_days=sx)
    a, b = _numbers(res), _numbers(res2)
    assert a.keys() == b.keys() and all(a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-12) for k in a)
    pd.testing.assert_frame_equal(res["_h1_days"], res2["_h1_days"])
    pd.testing.assert_frame_equal(res["_h2_weeks"], res2["_h2_weeks"])

    # Sanity check that the test can fail: changing the legitimate t-1 inputs does move H1.
    g3 = {k: v.copy() for k, v in grid.items()}
    g3["close"].loc[prev, ft.SLOT_1529] *= 1.01
    res3 = ft.analyse(dm, panel, g3, start, end, forward=False, sensex_days=sx)
    assert res3["_h1_days"]["r_rod_bp"].iloc[0] != pytest.approx(res["_h1_days"]["r_rod_bp"].iloc[0])


def test_forward_mode_reports_counts_only_before_the_stopping_point():
    dm, panel, grid = synth_market("2026-09-01", "2027-03-31")
    res = ft.analyse(dm, panel, grid, ft.FORWARD_START, pd.Timestamp("2027-03-31"), forward=True, sensex_days=set())
    assert res["h1"] == {"status": "not reached"} and res["h2"] == {"status": "not reached"}
    assert "_h1_days" not in res and "_h2_weeks" not in res and "h1b" not in res and "h2_descriptive" not in res
    assert res["h1_stopping"]["observations_so_far"] > 0 and res["h2_stopping"]["counting_weeks_so_far"] > 0
    assert res["decisions"]["H1"]["verdict"] == res["decisions"]["H2"]["verdict"] == "stopping point not reached"


def test_command_line_refuses_windows_that_break_the_preregistration():
    with pytest.raises(SystemExit):
        ft.main(["--start", "2026-11-10", "--end", "2027-12-31"])  # the forward test starts on 3 Nov 2026
    with pytest.raises(SystemExit):
        ft.main(["--dry-run", "--start", "2026-06-01", "--end", "2026-11-03"])  # a dry run may not touch new days


# ---------------------------------------------------------------- power arithmetic


def test_power_hand_worked():
    # z = 0.0682 / 0.02704 = 2.52219; power = Phi(2.52219 - 1.64485) = Phi(0.87734) = 0.80985
    assert ft.one_sided_power(-0.0682, 0.02704, 0.05) == pytest.approx(0.80985, abs=1e-4)
    assert ft.h1_se(463, 466, 0.018433, 463, 466) == pytest.approx(0.018433)
    # four times the days in each group halves the standard error
    assert ft.h1_se(4 * 463, 4 * 466, 0.018433, 463, 466) == pytest.approx(0.018433 / 2)
    assert ft.h2_se(61, 0.08238, 61) == pytest.approx(0.08238)
    assert ft.h2_se(244, 0.08238, 61) == pytest.approx(0.04119)
