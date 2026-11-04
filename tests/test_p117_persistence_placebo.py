"""Tests for P117 (experiments/p117_persistence_placebo.py). SYNTHETIC data only (no raw data file is read).

Covers: the gap of the actual pair; the rule for a pair straddling a calendar change and the three placebo sets;
the fast test for two pairs sharing a trading day against a brute-force check; the greedy day-disjoint selection;
the placebo p, the day-disjoint p, the separate-span count and the conditional p on hand-made numbers; the
decision rules; the half-life cap; the daily EGARCH level joined over a data hole (and unchanged without one); the
bootstrap with an explicit seed against P114's bootstrap, draw for draw; and an end-to-end dry run of every stage.
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex.volatility import daily_egarch_variance

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p117_persistence_placebo.py"
_spec = importlib.util.spec_from_file_location("p117_persistence_placebo", _PATH)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)
p114 = ex.p114
rep = ex.rep


def bdays(n, start="2018-01-01"):
    return pd.bdate_range(start, periods=n)


def test_actual_rows_and_gap():
    d = bdays(400)
    before = (str(d[100].date()), str(d[163].date()))
    after = (str(d[300].date()), str(d[358].date()))
    act = ex.actual_rows(d, before, after, 64, 59)
    assert act["start_row"] == 100 and act["after_row"] == 300
    assert act["gap"] == 300 - 164                       # grid days strictly between the two windows
    with pytest.raises(ValueError):
        ex.actual_rows(d, before, (str(d[300].date()), str(d[357].date())), 64, 59)


def test_straddle_rule():
    first = pd.DatetimeIndex(["2023-01-02", "2023-01-02", "2023-03-01", "2022-01-03"])
    last = pd.DatetimeIndex(["2023-02-28", "2023-03-01", "2023-06-01", "2023-02-28"])
    s = ex.straddles(first, last, "2023-03-01")
    # ends the day before the change: no; ends on its first day: yes; starts on it: no; spans it: no (ends before)
    assert list(s) == [False, True, False, False]


def test_sets_and_actual_pair():
    d = bdays(700)
    gap, Lb, La = 50, 30, 25
    changes = {"w": str(d[150].date()), "a": str(d[400].date()), "b": str(d[430].date()), "c": str(d[600].date())}
    sets = {"primary": ("a", "b"), "clean": tuple(changes), "literal": ("b",)}
    s_act = 380
    P = ex.enumerate_pairs(d, gap, Lb, La, history=60, changes=changes, sets=sets, actual_start=s_act)
    A = P[P["is_actual"]]
    assert len(A) == 1 and A["straddles_a"].iloc[0] and A["straddles_b"].iloc[0]
    for s in sets:
        assert not P.loc[P["is_actual"], f"in_{s}"].any()
        assert (P.loc[P[f"in_{s}"], "start_row"] >= 60).all()
    prim, clean, lit = (set(P.loc[P[f"in_{s}"], "start_row"]) for s in ("primary", "clean", "literal"))
    assert clean <= prim <= lit
    span = Lb + gap + La
    for _, r in P[P["in_primary"]].iterrows():
        for c in ("a", "b"):
            cd = pd.Timestamp(changes[c])
            assert not (d[r["start_row"]] < cd <= d[r["start_row"] + span - 1])
    for _, r in P[P["in_clean"]].iterrows():
        for c in changes.values():
            assert not (d[r["start_row"]] < pd.Timestamp(c) <= d[r["start_row"] + span - 1])
    assert (P["after_row"] == P["start_row"] + Lb + gap).all()
    assert (P["after_last"] == d[P["start_row"] + span - 1]).all()


@pytest.mark.parametrize("Lb,gap,La", [(64, 258, 59), (30, 10, 40), (5, 0, 5), (24, 16, 20)])
def test_pairs_share_days_matches_brute_force(Lb, gap, La):
    span = Lb + gap + La
    u0 = set(ex.pair_days(0, Lb, gap, La).tolist())
    for dd in range(0, span + 10):
        brute = bool(u0 & set(ex.pair_days(dd, Lb, gap, La).tolist()))
        assert bool(ex.pairs_share_days(dd, Lb, gap, La)) == brute, dd


def test_disjoint_selection_is_disjoint_and_greedy():
    Lb, gap, La = 20, 40, 15
    starts = np.arange(0, 400, 3)
    for ph in (0, 7, 30):
        sel = ex.disjoint_selection(starts, Lb, gap, La, ph)
        used = [set(ex.pair_days(s, Lb, gap, La).tolist()) for s in sel]
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                assert not (used[i] & used[j])
        allu = set().union(*used) if used else set()
        for s in starts:              # greedy: every pair left out (at or after the phase) clashes with a kept one
            if s >= ph and s not in sel:
                assert set(ex.pair_days(s, Lb, gap, La).tolist()) & allu
    # with a gap, pairs Lb apart interleave and share no day
    sel = ex.disjoint_selection(np.arange(0, 200), Lb, gap, La, 0)
    assert sel[:3] == [0, 20, 40]


def test_placebo_p_counts_ties_and_drops_nan():
    r = ex.placebo_p([-3.0, -2.0, -1.0, 0.0, 1.0, np.nan], -2.0)
    assert r["n"] == 5 and r["n_dropped"] == 1 and r["k"] == 2
    assert r["p"] == pytest.approx(3 / 6) and r["share"] == pytest.approx(2 / 5)
    assert ex.placebo_p([0.0, 1.0], -5.0)["p"] == pytest.approx(1 / 3)


def test_disjoint_p_and_separate_spans():
    Lb, gap, La = 10, 20, 8
    starts = np.arange(0, 300)
    delta = np.linspace(-1, 1, 300)
    lo = ex.disjoint_p(starts, delta, -5.0, Lb, gap, La)        # actual below every placebo: smallest attainable p
    assert lo["p"] == pytest.approx(lo["p_min_attainable"]) and lo["k_mean"] == 0
    hi = ex.disjoint_p(starts, delta, 5.0, Lb, gap, La)          # above every placebo: p = 1
    assert hi["p"] == pytest.approx(1.0)
    assert 1 <= lo["n_min"] <= lo["n_mean"] <= lo["n_max"]
    sp = ex.separate_spans(starts, Lb + gap + La)
    assert sp["max"] == int(np.ceil(300 / (Lb + gap + La))) and sp["min"] >= sp["max"] - 1


def test_conditional_p():
    rng = np.random.default_rng(1)
    x = rng.normal(5, 1, 500)
    y = 2.0 - 0.8 * x + rng.normal(0, 0.1, 500)
    r = ex.conditional_p(y, x, d_act=2.0 - 0.8 * 7.0 - 1.0, x_act=7.0)
    assert r["slope"] == pytest.approx(-0.8, abs=0.03) and r["resid_act"] == pytest.approx(-1.0, abs=0.05)
    assert r["p"] == pytest.approx(1 / 501)
    r2 = ex.conditional_p(y, x, d_act=2.0 - 0.8 * 7.0, x_act=7.0)    # on the line: an ordinary residual
    assert 0.2 < r2["p"] < 0.8


def test_decision_rules():
    assert ex.verdict(0.05, True) == "the fall is unusual"
    assert ex.verdict(0.0501, True) == "unclear"
    assert ex.verdict(0.10, True) == "unclear"
    assert ex.verdict(0.1001, True) == "the fall is within the normal range"
    assert ex.verdict(0.001, False).startswith("no verdict")
    assert ex.did_verdict(0.01) == "the fall is larger than the control's"
    assert ex.did_verdict(0.2) == "the fall is not clearly larger than the control's"
    assert ex.did_verdict(0.07) == "unclear"
    assert ex.boot_p_below0([-1, -1, -1, 1]) == pytest.approx(2 / 5)


def test_log_hl_cap():
    v = ex.log_hl([np.inf, 1e9, 10.0])
    assert v[0] == pytest.approx(np.log(1e5)) and v[1] == pytest.approx(np.log(1e5)) and v[2] == pytest.approx(np.log(10))


def _toy_grid(with_hole: bool):
    rng = np.random.default_rng(7)
    n = 160
    d = bdays(n, "2021-01-04")
    if with_hole:
        d = d[:60].append(d[100:] + pd.Timedelta(days=60))
    y = rng.standard_t(6, len(d)) * 0.01
    y[0] = np.nan
    if with_hole:
        y[60] = 0.25                                   # the return across the hole: must be left out
    return {"days": pd.DatetimeIndex(d), "trim": "toy|hole" if with_hole else "toy", "toy": {"daily": y}}


def test_egarch_joined_over_hole():
    rep._EGARCH_CACHE.clear()
    G = _toy_grid(True)
    last = len(G["days"]) - 1
    hf, method = ex.egarch_level_joined(G, "toy", last)
    assert method.endswith("joined over a data hole")
    rows = np.arange(1, last + 1)
    keep = rows != 60
    h, _ = daily_egarch_variance(G["toy"]["daily"][rows[keep]])
    assert np.array_equal(hf[rows[keep]], h) and np.isnan(hf[60]) and np.isnan(hf[0])
    rep._EGARCH_CACHE.clear()
    G0 = _toy_grid(False)
    a = ex.egarch_level_joined(G0, "toy", len(G0["days"]) - 1)[0]
    rep._EGARCH_CACHE.clear()
    b = rep.daily_variance(G0, "toy", len(G0["days"]) - 1)[0]
    assert np.array_equal(a, b, equal_nan=True)
    rep._EGARCH_CACHE.clear()


def test_boot_draws_match_p114(tmp_path):
    rep._EGARCH_CACHE.clear()
    p114._HAR_CACHE.clear()
    bars = p114.synthetic_bars(70, 3)
    G = rep.build_grids(bars, p114.TRIM)
    fw = p114.fit_window(G, "nifty", np.arange(40, 64), models=ex.MODELS)
    st = ex.store_of(fw)
    key = ("before_inferred", "nifty")
    pickle.dump({key: st}, open(tmp_path / "fits.pkl", "wb"))
    p114.boot_key((key[0], key[1], str(tmp_path), 8, ["ref", "F"]))
    theirs = pd.read_csv(tmp_path / "boot_before_inferred_nifty.csv", float_precision="round_trip")
    mine = ex.boot_draws(st, p114.SEED_INDEX[key], 8)
    for m in ex.MODELS:
        a = theirs[theirs["model"] == m].sort_values("draw")
        b = mine[mine["model"] == m].sort_values("draw")
        assert list(a["draw"]) == list(b["draw"])
        for q in ("alpha", "beta", "hl_irf"):
            assert np.array_equal(a[q].to_numpy(), b[q].to_numpy())
    part = ex.boot_draws(st, p114.SEED_INDEX[key], 8, chunks=[0], per_limit=1)
    assert np.array_equal(part["hl_irf"].to_numpy(), mine[mine["draw"] == 0]["hl_irf"].to_numpy())
    rep._EGARCH_CACHE.clear()
    p114._HAR_CACHE.clear()


def test_dry_run_all_stages(tmp_path):
    w, o = str(tmp_path / "work"), str(tmp_path / "out")
    ex.main(["windows", "--index", "nifty", "--synthetic", "--work", w])
    ex.main(["windows", "--index", "banknifty", "--synthetic", "--work", w])
    ex.main(["windows", "--index", "nifty", "--synthetic", "--work", w])        # resumes: nothing left to fit
    ex.main(["control", "--synthetic", "--work", w])
    ex.main(["analyze", "--synthetic", "--work", w, "--out", o, "--plan-commit", "test"])
    t = pd.read_csv(Path(o) / "tests.csv")
    assert list(t.columns) == ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness",
                               "p_method", "p_adjusted", "adjust_method"]
    assert t["p_raw"].between(0, 1).all() and (t["p_adjusted"] >= t["p_raw"] - 1e-12).all()
    assert set(t["role"]) == {"primary", "secondary", "exploratory"}
    assert t["test_id"].str.startswith("P_").sum() == 2 and t["test_id"].str.startswith("D_").sum() == 4
    v = pd.read_csv(Path(o) / "verdict.csv")
    allowed = {"the fall is unusual", "the fall is within the normal range", "unclear",
               "no verdict: too few separate placebo periods", "the fall is larger than the control's",
               "the fall is not clearly larger than the control's", "not computed"}
    assert set(v["label"]) <= allowed
    win = pd.read_csv(Path(w) / "windows_nifty.csv")
    assert not win.duplicated(["kind", "start_row", "model"]).any()
    pp = pd.read_csv(Path(o) / "placebo_pairs.csv")
    assert pp["is_actual"].sum() == 2
    assert (Path(o) / "fig_placebo.png").exists() and (Path(o) / "did.csv").exists()
