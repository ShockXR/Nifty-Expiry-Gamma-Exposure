"""Fast tests of experiments/synthetic_control.py (P96) on SYNTHETIC data only."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "synthetic_control.py"
_spec = importlib.util.spec_from_file_location("synthetic_control", _PATH)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


# ---------------------------------------------------------------- solver


def test_weights_nonnegative_and_sum_to_one():
    rng = np.random.default_rng(1)
    for J in (2, 5, 19, 40):
        X = rng.normal(size=(35, J))
        y = rng.normal(size=35) * 2.0
        w = sc.simplex_weights(X, y)
        assert w.shape == (J,)
        assert np.all(w >= 0)
        assert abs(w.sum() - 1.0) < 1e-9


def test_weights_recover_exact_convex_combination():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 8))
    w_true = np.array([0.5, 0.3, 0.2, 0, 0, 0, 0, 0])
    w = sc.simplex_weights(X, X @ w_true)
    assert np.allclose(w, w_true, atol=1e-5)


def test_weights_are_optimal_against_random_simplex_points():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20) + 3.0  # outside the convex hull: a corner or edge solution
    w = sc.simplex_weights(X, y)
    best = np.sum((y - X @ w) ** 2)
    cand = rng.dirichlet(np.ones(4), size=20000)
    vals = np.sum((y[None, :] - cand @ X.T) ** 2, axis=1)
    assert best <= vals.min() + 1e-9


def test_demeaned_fit_has_zero_mean_pre_gap_and_recovers_shift():
    rng = np.random.default_rng(4)
    T0, T1 = 30, 15
    X = rng.normal(size=(T0 + T1, 6))
    y = X @ np.array([0.6, 0.4, 0, 0, 0, 0]) + 0.8  # level difference: demeaning removes it
    post = np.arange(T0 + T1) >= T0
    y = y - 0.5 * post
    fit = sc.sc_fit(y, X, ~post, post, demean=True)
    assert abs(fit["pre_gap_mean"]) < 1e-10
    assert fit["effect"] == pytest.approx(-0.5, abs=1e-5)
    classic = sc.sc_fit(y, X, ~post, post, demean=False)
    assert classic["pre_gap_mean"] > 0.1  # classic SC cannot absorb the level difference


# ---------------------------------------------------------------- inference


def test_planted_effect_is_detected():
    rng = np.random.default_rng([sc.MASTER_SEED, 1])
    Y, pre, post = sc.simulate_panel(rng, J=19, T0=40, T1=20, effect=-3.0)
    donors = [c for c in Y.columns if c != "T"]
    tab, fit = sc.placebo_in_space(Y, "T", donors, pre, post)
    assert fit["effect"] < -2.0
    assert tab.attrs["p_ratio"] == pytest.approx(1 / 20)
    assert tab.attrs["p_neg"] == pytest.approx(1 / 20)
    inv = sc.test_inversion(Y, "T", donors, pre, post, tab, np.round(np.arange(-5, 1.01, 0.05), 2))
    lo, hi, _ = sc.confidence_set(inv)
    assert lo <= -3.0 <= hi and hi < 0
    assert sc.verdict(fit["effect"], tab.attrs["p_ratio"]) == "support"


def test_no_effect_gives_roughly_uniform_placebo_rank():
    rng = np.random.default_rng([sc.MASTER_SEED, 2])
    J, n = 9, 150
    ps = []
    for _ in range(n):
        Y, pre, post = sc.simulate_panel(rng, J=J, T0=25, T1=12, effect=0.0)
        donors = [c for c in Y.columns if c != "T"]
        tab, _ = sc.placebo_in_space(Y, "T", donors, pre, post)
        ps.append(tab.attrs["p_ratio"])
    ps = np.asarray(ps)
    ranks = np.round(ps * (J + 1)).astype(int)  # 1..10, uniform under no effect
    counts = np.bincount(ranks, minlength=J + 2)[1:J + 2]
    assert counts.sum() == n
    from scipy import stats
    assert stats.chisquare(counts).pvalue > 0.001
    assert 0.45 < ps.mean() < 0.65          # 0.55 for an exactly uniform rank
    assert np.mean(ps <= 0.1) < 0.2         # about 0.10 expected


def test_simple_did_and_hac_on_noiseless_data():
    T0, T1, J = 20, 10, 5
    post = np.arange(T0 + T1) >= T0
    base = np.sin(np.arange(T0 + T1))
    Y = pd.DataFrame({f"D{j}": base + j for j in range(J)})
    Y["T"] = base + 2.0 - 0.7 * post
    out = sc.simple_did(Y, "T", [f"D{j}" for j in range(J)], ~post, post)
    assert out["did"] == pytest.approx(-0.7)
    coef, se, p = sc.hac_mean_diff(np.r_[np.zeros(T0), -np.ones(T1)] + 0.01 * np.cos(np.arange(T0 + T1)),
                                   post)
    assert coef == pytest.approx(-1.0, abs=0.02)
    assert p < 1e-6


def test_holm_and_verdict():
    h = sc.holm({"a": 0.01, "b": 0.04})
    assert h["a"] == pytest.approx(0.02) and h["b"] == pytest.approx(0.04)
    h = sc.holm({"a": 0.05, "b": 1 / 24})
    assert h["b"] == pytest.approx(2 / 24) and h["a"] == pytest.approx(2 / 24)
    assert sc.verdict(-0.3, 0.05) == "support"
    assert sc.verdict(0.3, 0.05) == "evidence against"
    assert sc.verdict(-0.3, 0.2, level_pre=-0.1).startswith("no support (no pre-period premium")
    assert sc.verdict(-0.05, 0.4, level_pre=0.3, ci=(-0.2, 0.1)).startswith("no support (removal of the whole")
    assert sc.verdict(-0.05, 0.4, level_pre=0.3, ci=(-0.5, 0.1)).startswith("no support (too imprecise")


def test_power_curve_runs_on_simulated_data():
    pc = sc.power_curve(J=5, T0=10, T1=5, effects=[0.0, 2.0], n_sim=4)
    assert list(pc["effect_sd_units"]) == [0.0, -2.0]
    assert pc["power_ratio_test"].between(0, 1).all()
    assert (pc["median_rmspe_pre"] > 0).all()


def test_mde_interpolation():
    pc = pd.DataFrame({"effect_sd_units": [0.0, -1.0, -2.0], "p": [0.05, 0.6, 1.0]})
    assert sc.mde(pc, "p") == pytest.approx(1.5)
    assert np.isnan(sc.mde(pc.iloc[:2], "p"))


# ---------------------------------------------------------------- outcome


def test_log_parkinson_and_garman_klass():
    h, lo = np.array([110.0, 100.0]), np.array([100.0, 100.0])
    lp = sc.log_parkinson(h, lo)
    assert lp[0] == pytest.approx(np.log(np.log(1.1) ** 2 / (4 * np.log(2))))
    assert np.isnan(lp[1])  # High == Low: no range, no value
    gk = sc.log_garman_klass([105.0], [110.0], [100.0], [108.0])
    v = 0.5 * np.log(1.1) ** 2 - (2 * np.log(2) - 1) * np.log(108 / 105) ** 2
    assert gk[0] == pytest.approx(np.log(v))


def test_week_relative_target_minus_mean_of_other_days():
    days = pd.bdate_range("2024-01-08", "2024-01-19")  # two full weeks
    daily = pd.DataFrame({"a": np.arange(10.0), "b": np.ones(10)}, index=days)
    out = sc.week_relative(daily, 2, sc.week_start(days).unique())  # Wednesday
    assert out.loc[pd.Timestamp("2024-01-08"), "a"] == pytest.approx(2.0 - np.mean([0, 1, 3, 4]))
    assert out.loc[pd.Timestamp("2024-01-15"), "b"] == pytest.approx(0.0)
    # baseline exclusions and the minimum number of other days
    ex = sc.week_relative(daily, 2, [pd.Timestamp("2024-01-08")], exclude_days=[days[3]])
    assert ex.iloc[0]["a"] == pytest.approx(2.0 - np.mean([0, 1, 4]))
    short = daily.drop(index=days[[0, 1, 3]])
    assert np.isnan(sc.week_relative(short, 2, [pd.Timestamp("2024-01-08")]).iloc[0]["a"])


# ---------------------------------------------------------------- calendar and donors


def _fo_rows(date, symbol, expiries, fut=()):
    rows = [{"date": pd.Timestamp(date), "symbol": symbol, "inst": "opt", "expiry": pd.Timestamp(e), "contracts": 10}
            for e in expiries]
    rows += [{"date": pd.Timestamp(date), "symbol": symbol, "inst": "fut", "expiry": pd.Timestamp(e), "contracts": 5}
             for e in fut]
    return rows


def test_expiry_table_schedule_and_week_calendar():
    # Weekly Wednesday expiries; one moved to Tuesday by a holiday; the last one of the month is monthly.
    wed = ["2024-01-03", "2024-01-10", "2024-01-16", "2024-01-24", "2024-01-31", "2024-02-07"]
    rows = []
    for e in wed:
        rows += _fo_rows(e, "BANKNIFTY", [e], fut=["2024-01-31"] if e == "2024-01-31" else [])
    fo = pd.DataFrame(rows)
    exp = sc.expiry_table(fo)
    assert set(exp.loc[exp["kind"] == "monthly", "expiry"]) == {pd.Timestamp("2024-01-31")}
    flagged, changes = sc.flag_schedule(exp)
    assert changes.empty
    assert set(flagged.loc[flagged["shifted"], "expiry"]) == {pd.Timestamp("2024-01-16")}
    days = pd.bdate_range("2024-01-01", "2024-02-09").drop(pd.Timestamp("2024-01-17"))
    cal = sc.week_calendar(days, flagged)
    clean = sc.clean_mask(cal, ("NIFTY", "BANKNIFTY"))
    assert not clean[pd.Timestamp("2024-01-15")]   # shifted expiry
    assert not clean[pd.Timestamp("2024-01-29")]   # monthly expiry
    assert sc.clean_mask(cal, ("NIFTY",)).all()    # Nifty has no expiries in this toy file
    uw = sc.usable_weeks(cal, 2, "2024-01-01", "2024-02-05", symbols=sc.cal_symbols("BN"))
    assert list(uw) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-22"),
                        pd.Timestamp("2024-02-05")]
    # the week of 15 Jan has no Wednesday (holiday) and is shifted; without the shift rule it is
    # still dropped because the target day does not trade
    assert pd.Timestamp("2024-01-15") not in sc.usable_weeks(cal, 2, "2024-01-01", "2024-02-05", symbols=())


def test_schedule_change_detected_after_three_expiries():
    thu = ["2023-08-03", "2023-08-10", "2023-08-17", "2023-08-24"]
    wed = ["2023-09-06", "2023-09-13", "2023-09-20"]
    fo = pd.DataFrame(sum((_fo_rows(e, "BANKNIFTY", [e]) for e in thu + wed), []))
    flagged, changes = sc.flag_schedule(sc.expiry_table(fo))
    assert len(changes) == 1
    assert changes.iloc[0]["old_wd"] == "Thu" and changes.iloc[0]["new_wd"] == "Wed"
    assert not flagged["shifted"].any()
    assert sc.last_weekly(flagged, "BANKNIFTY") == pd.Timestamp("2023-09-20")


def test_name_classification_and_overlap_rule():
    cases = {"Nifty Auto": "sector/theme", "Nifty Private Bank": "sector/theme", "Nifty 50": "broad",
             "Nifty Midcap Select": "broad", "NIFTY Midcap 100": "broad", "India VIX": "non-equity",
             "Nifty50 TR 2x Leverage": "non-equity", "Nifty 8-13 yr G-Sec": "non-equity",
             "Nifty Alpha 50": "strategy", "Nifty50 Shariah": "strategy", "Nifty100 Liquid 15": "strategy",
             "Nifty India Corporate Group Index - Tata Group 25% Cap": "strategy", "Nifty CPSE": "sector/theme",
             "Nifty MidSmall Healthcare": "sector/theme"}
    for name, cat in cases.items():
        assert sc.classify_name(name) == cat, name
    days = pd.bdate_range("2024-01-01", periods=5)
    names = ["Nifty Auto", "Nifty Private Bank", "Nifty Services Sector", "Nifty MidSmall Healthcare",
             "Nifty Financial Services", "Nifty IT"]
    panel = pd.DataFrame([{"date": d, "key": sc.name_key(n), "name": n, "open": 100.0, "high": 101.0, "low": 99.0,
                           "close": 100.5} for d in days for n in names])
    panel.loc[(panel["name"] == "Nifty IT") & (panel["date"] == days[2]), "high"] = 99.0  # High == Low
    tab = sc.donor_table(sc.complete_and_valid(panel, days), {"Nifty Financial Services"})
    assert not tab.loc["nifty it", "complete"]
    assert sorted(tab.index[tab["donor"]]) == ["nifty auto"]
    assert tab.loc["nifty financial services", "has_derivatives"]
    assert tab.loc[["nifty private bank", "nifty services sector", "nifty midsmall healthcare"], "overlap"].all()


def test_readers_on_small_files(tmp_path):
    d = tmp_path / "idx"
    d.mkdir()
    pd.DataFrame({"Index Name": ["Nifty Auto ", "Nifty IT"], "Index Date": ["06-04-2023", "04-06-2023"],
                  "Open Index Value": ["100", "-"], "High Index Value": ["101", "-"],
                  "Low Index Value": ["99", "-"], "Closing Index Value": ["100.5", "50"]}
                 ).to_csv(d / "ind_close_all_06042023.csv", index=False)
    p = sc.load_index_panel(d)
    assert set(p["date"]) == {pd.Timestamp("2023-04-06")}  # date from the file name
    assert np.isnan(p.loc[p["key"] == "nifty it", "high"]).all()
    legacy = tmp_path / "fo_20230904.csv"
    pd.DataFrame({"INSTRUMENT": ["OPTIDX", "FUTIDX"], "SYMBOL": ["FINNIFTY", "FINNIFTY"],
                  "EXPIRY_DT": ["05-Sep-2023", "26-Sep-2023"], "CONTRACTS": ["7", "3"],
                  "TIMESTAMP": ["04-SEP-2023", "04-SEP-2023"]}).to_csv(legacy, index=False)
    udiff = tmp_path / "fo_20241104.csv"
    pd.DataFrame({"TradDt": ["2024-11-04"], "TckrSymb": ["MIDCPNIFTY"], "FinInstrmTp": ["IDO"],
                  "XpryDt": ["2024-11-04"], "TtlTradgVol": ["12"]}).to_csv(udiff, index=False)
    fo = sc.load_fo([tmp_path])
    assert set(fo["symbol"]) == {"FINNIFTY", "MIDCPNIFTY"}
    e = sc.expiry_table(fo)
    assert list(e["expiry"]) == [pd.Timestamp("2024-11-04")]


def _synthetic_study(effect_bn: float, seed: int):
    """Calendar shaped like the real one (weekly expiries end in November 2024, Nifty keeps
    Thursday, monthly on the last Thursday) and a SYNTHETIC daily log-variance panel."""
    days = pd.bdate_range(sc.STUDY_START, sc.STUDY_END)
    rows = []
    last = {"BANKNIFTY": ("2024-11-13", 2), "FINNIFTY": ("2024-11-19", 1), "MIDCPNIFTY": ("2024-11-18", 0),
            "NIFTY": (str(sc.STUDY_END.date()), 3)}
    month_end = pd.Series(days[days.weekday == 3]).groupby(pd.Series(days[days.weekday == 3]).dt.to_period("M")).max()
    for sym, (end, wd) in last.items():
        for d in days[(days.weekday == wd) & (days <= pd.Timestamp(end))]:
            rows += _fo_rows(d, sym, [d])
        for d in month_end:
            rows += _fo_rows(d, sym, [d], fut=[d])
    flagged, _ = sc.flag_schedule(sc.expiry_table(pd.DataFrame(rows)))
    cal = sc.week_calendar(days, flagged)
    rng = np.random.default_rng(seed)
    donors = [f"d{j:02d}" for j in range(19)]
    cols = [sc.name_key(sc.TREATED[k][0]) for k in sc.TREATED] + [sc.name_key(sc.NIFTY_NAME)] + donors
    common = rng.normal(0, 0.4, len(days))
    load = rng.uniform(0.5, 1.5, len(cols))
    v = common[:, None] * load[None, :] + rng.normal(0, 0.3, (len(days), len(cols)))
    park = pd.DataFrame(v, index=days, columns=cols)
    bn = sc.name_key("Nifty Bank")
    wed_weekly = (days.weekday == 2) & (days <= pd.Timestamp("2024-11-13"))
    park.loc[wed_weekly, bn] += -effect_bn  # premium while the weekly exists; gone afterwards
    dl = {"parkinson": park, "garman_klass": park + rng.normal(0, 0.1, park.shape)}
    return dl, cal, flagged, days, donors


def test_full_analysis_on_synthetic_panel_detects_planted_removal():
    dl, cal, flagged, days, donors = _synthetic_study(effect_bn=-1.5, seed=11)
    grid = np.round(np.arange(-3.0, 1.01, 0.1), 1)
    res, keep = sc.analyse_all(dl, cal, flagged, days, days, donors, grid=grid)
    assert len(res) == 3 * 10 + 1
    assert (res.loc[~res["analysis"].str.startswith("S6"), "J"] == 19).all()
    assert (res.loc[res["analysis"].str.startswith("S6"), "J"] == 18).all()
    prim = res[res["analysis"] == "primary"].iloc[0]
    W = sc.windows_for("BN", flagged)
    assert prim["n_pre"] == len(sc.usable_weeks(cal, 2, *W["pre"], symbols=sc.cal_symbols("BN")))
    assert prim["n_post"] == len(sc.usable_weeks(cal, 2, *W["post"], symbols=sc.cal_symbols("BN")))
    assert prim["effect"] < -1.0 and prim["level_gap_pre"] > 1.0
    assert prim["mde80_ratio_test_log_points"] == pytest.approx(1.93 * prim["rmspe_pre"])
    decision = sc.make_decision(res, [keep[k]["placebo"] for k in sc.TREATED])
    assert decision["primary"]["verdict"] == "support"
    assert decision["primary"]["p_ratio"] == pytest.approx(1 / 20)
    for k in sc.SECONDARIES:
        assert decision["secondaries"][k]["holm_p"] >= 0.1 - 1e-12
        assert decision["secondaries"][k]["smallest_possible_holm_p"] == pytest.approx(0.1)
    # S8 uses only Thursday and Friday as the baseline; S5 drops Nifty and Sensex days
    s8 = res[(res["analysis"] == "S8 baseline Thursday and Friday only") & (res["treated"] == "BN")].iloc[0]
    assert s8["n_pre"] <= prim["n_pre"] and s8["effect"] < -1.0


def test_analyse_refuses_without_phase2_flag():
    with pytest.raises(SystemExit):
        sc.main(["analyse", "--full", "nowhere"])


def test_analyse_refuses_other_data(tmp_path):
    (tmp_path / "index_close_full").mkdir()
    (tmp_path / "fo_index").mkdir()
    with pytest.raises(SystemExit, match="fingerprint"):
        sc.main(["analyse", "--phase2", "--full", str(tmp_path), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_data_fingerprint_tracks_content(tmp_path):
    for sub in ("index_close_full", "fo_index"):
        (tmp_path / sub).mkdir()
    (tmp_path / "index_close_full" / "ind_close_all_02012024.csv").write_text("a\n")
    (tmp_path / "fo_index" / "fo_20240102.csv.gz").write_bytes(b"b")
    (tmp_path / "fo_index" / "notes.txt").write_text("ignored")
    f1 = sc.data_fingerprint(tmp_path)
    assert (f1["n_index_files"], f1["n_fo_files"]) == (1, 1)
    assert sc.data_fingerprint(tmp_path)["sha256"] == f1["sha256"]
    (tmp_path / "index_close_full" / "ind_close_all_02012024.csv").write_text("A\n")
    assert sc.data_fingerprint(tmp_path)["sha256"] != f1["sha256"]


def test_crosscheck_local_finds_a_changed_price(tmp_path):
    def write(d, rows):
        d.mkdir(parents=True)
        pd.DataFrame(rows, columns=["Index Name", "Index Date", "Open Index Value", "High Index Value",
                                    "Low Index Value", "Closing Index Value"]
                     ).to_csv(d / "ind_close_all_02012024.csv", index=False)
    write(tmp_path / "index_close", [["Nifty 50", "02-01-2024", 100, 101, 99, 100.5],
                                     ["Nifty Bank", "02-01-2024", 200, 202, 198, 201]])
    write(tmp_path / "full", [["Nifty 50", "02-01-2024", 100, 101, 99, 100.5],
                              ["Nifty Bank", "02-01-2024", 200, 202.5, 198, 201],
                              ["Nifty Auto", "02-01-2024", 10, 11, 9, 10]])
    fo = pd.DataFrame(_fo_rows("2024-01-03", "BANKNIFTY", ["2024-01-03"]))
    out = sc.crosscheck_local(tmp_path, sc.load_index_panel(tmp_path / "full"), fo, fo)
    assert out["index_rows_local"] == 2 and out["index_rows_found_in_download"] == 2
    assert out["index_rows_identical"] == 1
    assert out["BANKNIFTY_expiries_identical"] and out["BANKNIFTY_expiries_download"] == 1


def test_sensex_rule_days_and_holiday_rollback(tmp_path):
    cal = tmp_path / "cal.csv"
    pd.DataFrame({"fact_type": ["weekday_regime", "weekday_regime", "expiry_date"],
                  "date": ["2024-01-01", "2024-01-22", "2024-01-05"], "date_end": ["2024-01-19", "", ""],
                  "weekday": ["Fri", "Tue", "Fri"]}).to_csv(cal, index=False)
    days = pd.bdate_range("2024-01-01", "2024-02-02").drop(pd.Timestamp("2024-01-12"))  # Friday holiday
    got = sc.sensex_rule_days(days, cal)
    assert [d.date().isoformat() for d in got] == ["2024-01-05", "2024-01-11", "2024-01-19", "2024-01-23",
                                                   "2024-01-30"]


def test_thu_fri_only_baseline():
    days = pd.bdate_range("2024-01-08", "2024-01-12")
    ex = sc.thu_fri_only_exclusions(days, 2)  # Wednesday target
    assert list(ex.weekday) == [0, 1]
    daily = pd.DataFrame({"a": [5.0, 7.0, 3.0, 1.0, 2.0]}, index=days)
    out = sc.week_relative(daily, 2, [days[0]], exclude_days=ex)
    assert out.iloc[0]["a"] == pytest.approx(3.0 - 1.5)
    assert list(sc.thu_fri_only_exclusions(days, 0).weekday) == [1, 2]
