"""Unit tests for experiments/sensex_confound.py on small SYNTHETIC data
(calendar rule, date comparison, week-relative table with exclusions, the
before/after test, the permutation test, the fixed-effects regression, the R2
rerun with excluded days and the Sensex bar download)."""

import importlib.util
import io
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


sc = _load("sensex_confound", _ROOT / "experiments" / "sensex_confound.py")
rdl = _load("real_day_level", _ROOT / "experiments" / "real_day_level.py")


def _bdays(start, end, holidays=()):
    d = pd.bdate_range(start, end)
    return d[~d.isin(pd.DatetimeIndex(list(holidays)))]


# ---------------------------------------------------------------- calendar


def test_load_regimes_reads_weekday_rows_and_open_end():
    cal = pd.DataFrame({
        "fact_type": ["weekday_regime", "expiry_date", "weekday_regime"],
        "date": ["2025-01-01", "2025-01-07", "2025-09-01"],
        "date_end": ["2025-08-31", "", ""],
        "weekday": ["Tue", "Tue", "Thu"],
    })
    reg = sc.load_regimes(cal)
    assert reg == [(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-08-31"), 1),
                   (pd.Timestamp("2025-09-01"), pd.Timestamp("2100-01-01"), 3)]


def test_rule_expiries_weekday_switch_and_holiday_rollback():
    # SYNTHETIC calendar: Friday regime to 3 Jan 2025, Tuesday regime from 1 Jan 2025.
    # Fri 27 Dec 2024 is made a holiday, so that expiry moves back to Thu 26 Dec.
    days = _bdays("2024-12-16", "2025-01-24", holidays=["2024-12-27"])
    regimes = [(pd.Timestamp("2024-12-16"), pd.Timestamp("2025-01-03"), 4),
               (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-08-31"), 1)]
    e = sc.sensex_rule_expiries(days, regimes)
    got = [d.date().isoformat() for d in e["expiry"]]
    assert got == ["2024-12-20", "2024-12-26", "2025-01-03", "2025-01-07", "2025-01-14", "2025-01-21"]
    row = e[e["expiry"] == pd.Timestamp("2024-12-26")].iloc[0]
    assert bool(row["shifted"]) and row["weekday"] == "Thu" and row["scheduled_weekday"] == "Fri"
    assert not e.loc[e["expiry"] != pd.Timestamp("2024-12-26"), "shifted"].any()


def test_rule_expiries_stops_at_last_trading_day():
    days = _bdays("2025-09-01", "2025-09-15")
    e = sc.sensex_rule_expiries(days, [(pd.Timestamp("2025-09-01"), pd.Timestamp("2100-01-01"), 3)])
    assert [d.date().isoformat() for d in e["expiry"]] == ["2025-09-04", "2025-09-11"]


def test_compare_dates_counts_inside_window_only():
    rule = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19", "2024-03-01"])
    other = pd.to_datetime(["2024-01-05", "2024-01-11", "2024-03-01"])
    r = sc.compare_dates(rule, other, "2024-01-01", "2024-01-31")
    assert (r["n_rule"], r["n_other"], r["n_both"]) == (3, 2, 1)
    assert r["rule_only"] == "2024-01-12;2024-01-19"
    assert r["other_only"] == "2024-01-11"


def test_sensex_paths_yearly_then_monthly():
    p = sc.sensex_paths(pd.Timestamp("2026-03-10"))
    assert p[:3] == ["1min/2023/SENSEX_1min_2023.csv", "1min/2024/SENSEX_1min_2024.csv",
                     "1min/2025/SENSEX_1min_2025.csv"]
    assert p[3:] == ["1min/2026/SENSEX_1min_2026-01.csv", "1min/2026/SENSEX_1min_2026-02.csv",
                     "1min/2026/SENSEX_1min_2026-03.csv"]


def test_fetch_sensex_bars_converts_layout(tmp_path, monkeypatch):
    # SYNTHETIC CSVs served by a fake requests.get; two files overlap by one minute.
    import requests

    body = {
        "1min/2023/SENSEX_1min_2023.csv": "Epoch,Timestamp,Open,High,Low,Close\n"
                                          "1,2023-01-02T09:15:00,10,11,9,10.5\n2,2023-01-02T09:16:00,10.5,12,10,11\n",
        "1min/2024/SENSEX_1min_2024.csv": "Epoch,Timestamp,Open,High,Low,Close\n"
                                          "2,2023-01-02T09:16:00,10.5,12,10,11\n3,2024-01-01T09:15:00,20,21,19,20\n",
    }

    class R:
        def __init__(self, text, code):
            self.content, self.status_code = text.encode(), code

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=0):
        path = url.split("/main/")[1]
        return R(body[path], 200) if path in body else R("", 404)

    monkeypatch.setattr(requests, "get", fake_get)
    out = tmp_path / "sensex_1min.csv"
    meta = sc.fetch_sensex_bars(out, pd.Timestamp("2026-01-15"))
    df = pd.read_csv(out)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close"]
    assert len(df) == 3 and meta["rows"] == 3 and len(meta["files"]) == 2
    assert (tmp_path / "sensex_1min_source.json").exists()


# ---------------------------------------------------------------- week-relative and tests


def test_relative_table_exclusion_changes_only_the_baseline():
    # SYNTHETIC week: Friday (a "Sensex day") is high; target Tuesday.
    days = _bdays("2025-01-06", "2025-01-10")
    v = pd.Series([1.0, 2.0, 1.0, 1.0, 8.0], index=days)
    daily = {"N": pd.DataFrame({"rv_last30": v})}
    tg = {"N": {"Tue": sc.es.weekday_targets(days, 1), "Fri": sc.es.weekday_targets(days, 4)}}
    plain = sc.relative_table(daily, tg, ["rv_last30"])
    excl = sc.relative_table(daily, tg, ["rv_last30"], exclude={"N": {pd.Timestamp("2025-01-10")}})
    tue_plain = plain[(plain["target"] == "Tue")]["rel"].iloc[0]
    tue_excl = excl[(excl["target"] == "Tue")]["rel"].iloc[0]
    assert tue_plain == pytest.approx(np.log(2.0) - np.mean(np.log([1.0, 1.0, 1.0, 8.0])))
    assert tue_excl == pytest.approx(np.log(2.0))
    # the excluded day is still a valid target itself
    fri_excl = excl[(excl["target"] == "Fri")]["rel"].iloc[0]
    assert fri_excl == pytest.approx(np.log(8.0) - np.mean(np.log([1.0, 2.0, 1.0, 1.0])))


def test_tue_minus_fri_is_1_25_times_raw_difference():
    days = _bdays("2025-01-06", "2025-01-10")
    rng = np.random.default_rng(1)
    v = pd.Series(np.exp(rng.normal(size=5)), index=days)
    daily = {"N": pd.DataFrame({"rv_last30": v})}
    tg = {"N": {"Tue": sc.es.weekday_targets(days, 1), "Fri": sc.es.weekday_targets(days, 4)}}
    rel = sc.relative_table(daily, tg, ["rv_last30"]).set_index("target")["rel"]
    assert rel["Tue"] - rel["Fri"] == pytest.approx(1.25 * (np.log(v.iloc[1]) - np.log(v.iloc[4])))


def test_event_test_detects_a_large_step_and_not_noise():
    # SYNTHETIC weekly series: 300 weeks of noise, a step of +2 at week 250 (placebo windows fit before it).
    weeks = pd.date_range("2019-01-07", periods=300, freq="7D")
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(0, 0.3, 300), index=weeks)
    y_step = y.copy()
    y_step.iloc[250:] += 2.0
    r = sc.event_test(y_step, weeks[250], 26, [weeks[250]], n_boot=499, seed=1)
    assert r["diff"] == pytest.approx(y_step.iloc[250:276].mean() - y_step.iloc[224:250].mean())
    assert r["hac_p"] < 1e-6 and r["ri_n_independent"] >= 3
    assert r["ri_p"] == pytest.approx(1 / (1 + r["ri_n_placebo"]))  # larger than every placebo t
    r0 = sc.event_test(y, weeks[250], 26, [weeks[250]], n_boot=499, seed=1)
    assert r0["hac_p"] > 0.01 and r0["ri_p"] > 0.05


def test_permutation_mean_diff():
    rng = np.random.default_rng(3)
    a = rng.normal(2.0, 0.5, 6)
    b = rng.normal(0.0, 0.5, 40)
    d, p = sc.permutation_mean_diff(a, b, n_perm=1999, seed=0)
    assert d == pytest.approx(a.mean() - b.mean())
    assert p < 0.01
    d0, p0 = sc.permutation_mean_diff(b[:6], b[6:], n_perm=1999, seed=0)
    assert p0 > 0.05
    assert np.isnan(sc.permutation_mean_diff([], b)[1])


def test_within_week_ols_recovers_dummy_with_week_effects():
    # SYNTHETIC panel: large week effects, a dummy worth +0.5 on one day a week.
    rng = np.random.default_rng(7)
    days = _bdays("2024-01-01", "2024-12-31")
    week = sc.es.week_start(days)
    w_eff = pd.Series(rng.normal(0, 3, len(week.unique())), index=week.unique())
    dummy = (days.weekday == 4).astype(float)
    y = w_eff.reindex(week).to_numpy() + 0.5 * dummy + rng.normal(0, 0.1, len(days))
    df = pd.DataFrame({"y": y, "week": week, "sx": dummy}, index=days)
    tab, n, nw = sc.within_week_ols(df, "y", ["sx"])
    assert tab.loc[tab["term"] == "sx", "coef"].iloc[0] == pytest.approx(0.5, abs=0.03)
    assert n == len(days) and nw == len(week.unique())


def test_r2_with_exclusion_drops_only_non_expiry_days():
    # SYNTHETIC daily_market-like frame: expiry Thursday, a high Friday excluded from the baseline.
    days = _bdays("2023-01-02", "2023-03-31")
    rv = pd.Series(1.0, index=days)
    rv[days.weekday == 4] = np.e  # Fridays: log value 1
    dm = pd.DataFrame({"rv_day": rv, "rv_first_hour": rv, "rv_last_hour": rv,
                       "is_expiry": days.weekday == 3, "weekday": days.weekday,
                       "regime": "multi_weekly"}, index=days)
    plain = rdl.r2_regimes(dm)
    ctrl = sc.r2_with_exclusion(rdl, dm, days[days.weekday == 4])
    m_plain = plain[(plain["measure"] == "rv_day") & (plain["regime"] == "multi_weekly")]["mean_rel"].iloc[0]
    m_ctrl = ctrl[(ctrl["measure"] == "rv_day") & (ctrl["regime"] == "multi_weekly")]["mean_rel"].iloc[0]
    assert m_plain == pytest.approx(-0.25)
    assert m_ctrl == pytest.approx(0.0)
