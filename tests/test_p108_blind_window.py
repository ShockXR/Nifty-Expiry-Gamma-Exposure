"""Tests for the P108 blind window search. SYNTHETIC data only.

The search code lives in experiments/p108_blind_window/search.py and is
imported from its path, so these tests do not depend on the project package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p108_search", ROOT / "experiments/p108_blind_window/search.py")
S = importlib.util.module_from_spec(SPEC)
sys.modules["p108_search"] = S
SPEC.loader.exec_module(S)


# ---------------------------------------------------------------- synthetic data
def synthetic_bars(n_days: int = 150, seed: int = 7, start: str = "2023-06-01",
                   missing_share: float = 0.002) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SYNTHETIC one-minute bars for two correlated indices, 375 bars a day, prices to 2 decimals."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    minutes = np.arange(375)
    frames = {"nifty": [], "bank": []}
    level = {"nifty": 20000.0, "bank": 45000.0}
    for di, day in enumerate(days):
        z = rng.standard_t(4, size=(375, 2)) * np.array([2.5e-4, 3.5e-4])
        common = rng.standard_t(4, size=375) * 2.5e-4
        gap = rng.normal(0, 3e-3, size=2)
        for j, k in enumerate(("nifty", "bank")):
            r = 0.8 * common * (1.4 if k == "bank" else 1.0) + 0.6 * z[:, j]
            r[0] += gap[j]
            path = level[k] * np.exp(np.cumsum(r))
            level[k] = path[-1]
            close = np.round(path, 2)
            ts = day + pd.Timedelta(hours=9, minutes=15) + pd.to_timedelta(minutes, unit="min")
            df = pd.DataFrame({"timestamp": ts, "open": close, "high": close + 1.0,
                               "low": close - 1.0, "close": close})
            if di == 20:  # a short special session
                df = df.iloc[:105]
            drop = rng.random(len(df)) < missing_share
            drop[0] = False
            frames[k].append(df[~drop])
    return tuple(pd.concat(frames[k], ignore_index=True) for k in ("nifty", "bank"))


def independent_returns(df: pd.DataFrame, first: str, last: str, days, overnight: str, kind: str):
    """Independent (pandas) construction of a window's returns, used to check the search code."""
    d = df.copy()
    d["date"] = d["timestamp"].dt.normalize()
    tod = d["timestamp"].dt.strftime("%H:%M")
    d = d[(tod >= first) & (tod <= last) & d["date"].isin(days)].sort_values("timestamp")
    p = d.set_index("timestamp")["close"]
    if kind == "log":
        r = np.log(p).diff()
    else:
        r = p.pct_change()
    if overnight == "intraday":
        same = d["date"].to_numpy()[1:] == d["date"].to_numpy()[:-1]
        r = r.iloc[1:][same]
    else:
        r = r.iloc[1:]
    return r


# ---------------------------------------------------------------- unit tests
def test_moment_variants_match_scipy():
    rng = np.random.default_rng(1)
    x = rng.standard_t(3, size=5000) * 1e-3
    n = len(x)
    c = x - x.mean()
    v = S.stat_variants(n, (c ** 2).mean(), (c ** 3).mean(), (c ** 4).mean())
    assert v[("skew", "g1")] == pytest.approx(scipy.stats.skew(x, bias=True), rel=1e-10)
    assert v[("skew", "G1")] == pytest.approx(scipy.stats.skew(x, bias=False), rel=1e-10)
    assert v[("skew", "G1")] == pytest.approx(pd.Series(x).skew(), rel=1e-10)
    assert v[("kurt", "g2")] == pytest.approx(scipy.stats.kurtosis(x, bias=True), rel=1e-10)
    assert v[("kurt", "G2")] == pytest.approx(scipy.stats.kurtosis(x, bias=False), rel=1e-10)
    assert v[("kurt", "G2")] == pytest.approx(pd.Series(x).kurt(), rel=1e-10)
    assert v[("kurt", "g2+3")] == pytest.approx(scipy.stats.kurtosis(x, fisher=False), rel=1e-10)
    s = x.std(ddof=1)
    assert v[("skew", "b1")] == pytest.approx((c ** 3).mean() / s ** 3, rel=1e-10)
    assert v[("kurt", "b2")] == pytest.approx((c ** 4).mean() / s ** 4 - 3, rel=1e-10)
    assert v[("sd", "sample")] == pytest.approx(s, rel=1e-12)


def test_matches_rounding():
    assert S.matches(-0.0085846494, -0.008584649, 9)
    assert not S.matches(-0.0085846498, -0.008584649, 9)
    assert S.matches(-0.0085846498, -0.008584649, 9, loose=True)
    assert not S.matches(-0.0085846506, -0.008584649, 9, loose=True)
    assert S.matches(23039.0, 23039, 0)
    assert not S.matches(23040.0, 23039, 0)
    assert S.matches(-0.0000014, -0.000001, 6)
    assert not S.matches(-0.0000016, -0.000001, 6)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_anderson_matches_scipy():
    rng = np.random.default_rng(3)
    x = rng.standard_t(5, size=3000)
    ours = S.anderson_variants(x)["ddof1"]
    assert ours == pytest.approx(scipy.stats.anderson(x, dist="norm").statistic, rel=1e-9)


def test_definitions_and_names_round_trip():
    defs = S.all_definitions()
    assert len(defs) == 17 * 3 * 2 * 2
    for d in defs:
        assert S.definition_from_name(d.name) == d
    d = S.Definition(15, 360, "concat", "log", "all")
    assert d.session_label == "0930-1529"
    assert S.Definition(0, 360, "concat", "log", "all").session_label == "0915-1514"


@pytest.fixture(scope="module")
def synth():
    dfn, dfb = synthetic_bars()
    days, mats = S.build_matrices(dfn, dfb, fields=("open", "high", "low", "close"))
    return dfn, dfb, days, mats


@pytest.mark.parametrize("overnight", ["concat", "intraday", "concat_pre"])
@pytest.mark.parametrize("kind", ["log", "simple"])
def test_window_table_equals_direct(synth, overnight, kind):
    """SYNTHETIC: prefix-sum window moments equal a direct computation on the assembled series."""
    dfn, dfb, days, mats = synth
    d = S.Definition(15, 360, overnight, kind, "all")
    dl = S.day_level(days, mats, d)
    tbl = S.window_table(S.day_aggregates(dl), overnight, lengths=(40, 64))
    for i in (0, 37, len(tbl["a"]) - 1):
        a, b = int(tbl["a"][i]), int(tbl["b"][i])
        ser = S.window_series(dl["days"], days, mats, d, a, b)
        for k, df in (("nifty", dfn), ("bank", dfb)):
            x = ser[k].to_numpy()
            assert tbl[k]["n"][i] == len(x)
            assert tbl[k]["mean"][i] == pytest.approx(x.mean(), rel=1e-9, abs=1e-15)
            assert tbl[k]["m2"][i] == pytest.approx(x.var(), rel=1e-9)
            c = x - x.mean()
            assert tbl[k]["m3"][i] == pytest.approx((c ** 3).mean(), rel=1e-7)
            assert tbl[k]["m4"][i] == pytest.approx((c ** 4).mean(), rel=1e-9)
            assert tbl[k]["min"][i] == x.min() and tbl[k]["max"][i] == x.max()
            if overnight != "concat_pre":
                # a second, fully independent construction with pandas
                win_days = dl["days"][a:b + 1]
                y = independent_returns(df, "09:30", "15:29", win_days, overnight, kind).to_numpy()
                # log(a/b) and log(a) - log(b) differ by about 1e-15, far below the 1e-9 printed precision
                np.testing.assert_allclose(np.sort(x), np.sort(y), rtol=0, atol=1e-13)
        j = pd.concat([ser["nifty"], ser["bank"]], axis=1, join="inner")
        assert tbl["corr"][i] == pytest.approx(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1], rel=1e-9)


def planted_printed(x_n: np.ndarray, x_b: np.ndarray, corr: float) -> dict:
    """Round SYNTHETIC window statistics the way the paper prints them (Table 1 precision)."""
    out = {}
    for k, x, kdec in (("nifty", x_n, 8), ("bank", x_b, 7)):
        out[k] = {
            "mean": (round(x.mean(), 6), 6),
            "sd": (round(x.std(ddof=1), 9), 9),
            "min": (round(x.min(), 9), 9),
            "max": (round(x.max(), 9), 9),
            "skew": (round(scipy.stats.skew(x, bias=False), 9), 9),
            "kurt": (round(scipy.stats.kurtosis(x, bias=False), kdec), kdec),
            "n": (len(x), 0),
            "ad": (round(scipy.stats.anderson(x, dist="norm").statistic, 3), 3),
        }
    out["corr"] = (round(corr, 3), 3)
    return out


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_plant_and_recover(synth):
    """SYNTHETIC: plant a 64-day window under one definition and check the search finds it."""
    dfn, dfb, days, mats = synth
    planted = S.Definition(15, 360, "concat", "log", "all")
    dl = S.day_level(days, mats, planted)
    a, b = 30, 93
    win_days = dl["days"][a:b + 1]
    # printed values built with independent pandas code, not the search machinery
    xn = independent_returns(dfn, "09:30", "15:29", win_days, "concat", "log")
    xb = independent_returns(dfb, "09:30", "15:29", win_days, "concat", "log")
    j = pd.concat([xn, xb], axis=1, join="inner")
    printed = planted_printed(xn.to_numpy(), xb.to_numpy(), float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]))

    candidates = [planted,
                  S.Definition(0, 375, "concat", "log", "all"),
                  S.Definition(14, 360, "concat", "log", "all"),
                  S.Definition(15, 360, "intraday", "log", "all"),
                  S.Definition(15, 360, "concat", "simple", "all"),
                  S.Definition(15, 360, "concat_pre", "log", "all")]
    tops = []
    for d in candidates:
        ranked, _ = S.score_definition(days, mats, d, printed, lengths=tuple(range(40, 101)))
        tops.append(ranked.iloc[0])
    best = sorted(tops, key=lambda r: (-r.E, r.D))[0]
    assert best.definition == planted.name
    assert (best.start, best.end) == (win_days[0].strftime("%Y-%m-%d"), win_days[-1].strftime("%Y-%m-%d"))
    assert best.E == 15
    # D cannot reach zero: the correlation is printed to 3 decimals (rounding error up to 5e-4)
    assert best.D < 2e-4
    # the runner-up inside the planted definition is clearly worse
    ranked, _ = S.score_definition(days, mats, planted, printed, lengths=tuple(range(40, 101)))
    r1, rd = S.runners_up(ranked)
    assert r1.E <= 12 and r1.D > best.D
    # stage 2 (direct recomputation with Anderson-Darling) agrees
    row = S.stage2_row(days, mats, dl, planted, a, b, printed, ohlc=mats)
    assert row["E2"] == 17
    assert row["nifty_skew_variant"] == "G1" and row["nifty_kurt_variant"] == "G2"


def test_anchor_search_finds_planted_extreme(synth):
    dfn, dfb, days, mats = synth
    d = S.Definition(0, 375, "concat", "log", "all")
    dl = S.day_level(days, mats, d)
    x = dl[("intra", "bank")]
    i, jj = np.unravel_index(np.nanargmin(x), x.shape)
    printed = {k: {"min": (-9.0, 9), "max": (9.0, 9)} for k in S.INDICES}
    printed["bank"]["min"] = (round(float(x[i, jj]), 9), 9)
    rows = S.extreme_anchors(dl, d, printed)
    hits = [r for r in rows if r["index"] == "bank" and r["item"] == "min" and r["exact"]]
    assert any(r["date"] == dl["days"][i].strftime("%Y-%m-%d") for r in hits)


def test_run_end_to_end_synthetic(tmp_path):
    """SYNTHETIC: the full pipeline runs and writes its files (scores are meaningless here)."""
    dfn, dfb = synthetic_bars(n_days=80, start="2023-10-02")
    raw = tmp_path / "raw"
    raw.mkdir()
    dfn.to_csv(raw / "nifty_1min.csv", index=False)
    dfb.to_csv(raw / "banknifty_1min.csv", index=False)
    out = tmp_path / "out"
    defs = [S.Definition(15, 360, "concat", "log", "all"), S.Definition(0, 375, "intraday", "simple", "full")]
    res = S.run(raw, out, definitions=defs, lengths=(40, 41), top_defs=2, top_per_def=2, adf_top=1,
                label="SYNTHETIC")
    for f in ("best_by_definition.csv", "top_candidates.csv", "stated_windows.csv", "extreme_anchors.csv",
              "run_info.json", "all_windows_winning_definition.csv.gz", "after_window_check.csv"):
        assert (out / f).exists(), f
    assert len(res["best"]) == 2


def test_posthoc_one_day_dropped_equals_direct(synth):
    """SYNTHETIC: the post hoc 'one day left out' sums equal a direct pandas construction."""
    spec = importlib.util.spec_from_file_location("p108_posthoc", ROOT / "experiments/p108_blind_window/posthoc.py")
    P = importlib.util.module_from_spec(spec)
    sys.modules["p108_posthoc"] = P
    spec.loader.exec_module(P)
    dfn, dfb, days, mats = synth
    d = S.Definition(15, 360, "concat", "log", "all")
    dl = S.day_level(days, mats, d)
    # dummy printed values: only the window statistics are compared here
    printed = {k: {"mean": (0.0, 6), "sd": (3e-4, 9), "min": (-0.01, 9), "max": (0.01, 9), "skew": (0.1, 9),
                   "kurt": (5.0, 7), "n": (1, 0)} for k in S.INDICES}
    printed["corr"] = (0.5, 3)
    r = P.one_day_dropped(days, mats, d, printed, span=45)
    for _, row in r.sample(5, random_state=1).iterrows():
        win = dl["days"][(dl["days"] >= row.start) & (dl["days"] <= row.end)]
        win = win[win != pd.Timestamp(row.dropped_day)]
        assert len(win) == 44
        xs = {}
        for k, df in (("nifty", dfn), ("bank", dfb)):
            x = independent_returns(df, "09:30", "15:29", win, "concat", "log")
            xs[k] = x
            v = x.to_numpy()
            assert row[f"{k}_n"] == len(v)
            assert row[f"{k}_min"] == pytest.approx(v.min(), abs=1e-13)
            assert row[f"{k}_max"] == pytest.approx(v.max(), abs=1e-13)
            assert row[f"{k}_mean"] == pytest.approx(v.mean(), rel=1e-8, abs=1e-15)
            sd = v.std(ddof=1) if row.sd_variant == "sample" else v.std()
            assert row[f"{k}_sd"] == pytest.approx(sd, rel=1e-9)
        j = pd.concat([xs["nifty"], xs["bank"]], axis=1, join="inner")
        assert row["corr"] == pytest.approx(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1], rel=1e-9)
