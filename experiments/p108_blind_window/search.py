#!/usr/bin/env python
"""P108: blind search for the calendar window behind the paper's "before" sample.

The paper (Chhimwal, Pandey and Pandey 2025, Table 1) prints descriptive
statistics of one-minute Nifty 50 and Nifty Bank returns for a "before" window.
This script scores every window of consecutive trading days in the raw minute
data against those printed numbers, under a pre-registered set of return
definitions, and reports which window fits best and by how much.

It is written from scratch for study P108 (it imports nothing from the
project package) and follows docs/prereg/P108_BLIND_WINDOW.md.

Run on REAL DATA:
    OMP_NUM_THREADS=1 python experiments/p108_blind_window/search.py \
        --raw data/raw --out results/p108_blind_window

The tests (tests/test_p108_blind_window.py) run the same functions on
SYNTHETIC data only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.stats

HERE = Path(__file__).resolve().parent
PRINTED_CSV = HERE / "printed_values.csv"

OPEN_MINUTE = 9 * 60 + 15  # 09:15 IST, the first regular-session bar
FULL_BARS = 375  # 09:15 to 15:29 inclusive
INDICES = ("nifty", "bank")
LENGTHS = tuple(range(40, 101))  # window lengths in trading days (pre-registered)
FULL_DAY_SHARE = 0.9  # "full" calendar: a day needs at least 90% of the session's bars

# Statistic formula variants (pre-registered). Population central moments m2, m3, m4.
SD_VARIANTS = ("sample", "population")
SKEW_VARIANTS = ("g1", "G1", "b1")
KURT_VARIANTS = ("g2", "G2", "b2", "g2+3", "G2+3", "b2+3")

STAT_ITEMS = ("mean", "sd", "min", "max", "skew", "kurt", "n")


# --------------------------------------------------------------------------
# Return definitions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Definition:
    offset: int  # first bar, in minutes after 09:15
    nbars: int  # 375 (whole session) or 360
    overnight: str  # "concat", "intraday" or "concat_pre"
    kind: str  # "log" or "simple"
    calendar: str  # "all" or "full"

    @property
    def name(self) -> str:
        return f"{self.session_label}|{self.overnight}|{self.kind}|{self.calendar}"

    @property
    def session_label(self) -> str:
        start = OPEN_MINUTE + self.offset
        end = start + self.nbars - 1
        return f"{start // 60:02d}{start % 60:02d}-{end // 60:02d}{end % 60:02d}"


def all_definitions() -> list[Definition]:
    sessions = [(0, FULL_BARS)] + [(k, 360) for k in range(16)]
    out = []
    for off, nb in sessions:
        for ov in ("concat", "intraday", "concat_pre"):
            for kind in ("log", "simple"):
                for cal in ("all", "full"):
                    out.append(Definition(off, nb, ov, kind, cal))
    return out


def definition_from_name(name: str) -> Definition:
    sess, ov, kind, cal = name.split("|")
    s, e = sess.split("-")
    start = int(s[:2]) * 60 + int(s[2:])
    end = int(e[:2]) * 60 + int(e[2:])
    return Definition(start - OPEN_MINUTE, end - start + 1, ov, kind, cal)


def ret(cur, prev, kind: str):
    if kind == "log":
        return np.log(cur / prev)
    return cur / prev - 1.0


# --------------------------------------------------------------------------
# Printed values
# --------------------------------------------------------------------------
def load_printed(path: Path = PRINTED_CSV, window: str = "before") -> dict:
    """Return {'nifty': {stat: (value, decimals)}, 'bank': {...}, 'corr': (value, decimals)}."""
    df = pd.read_csv(path, dtype={"printed": str})
    tag = "B" if window == "before" else "A"
    out: dict = {"nifty": {}, "bank": {}}
    for _, r in df.iterrows():
        parts = str(r["id"]).split("_")
        if parts[0] != "T1" or parts[1] != tag:
            continue
        if parts[2] == "corr":
            out["corr"] = (float(r["value"]), int(r["decimals"]))
            continue
        idx = {"N": "nifty", "B": "bank"}[parts[2]]
        out[idx][parts[3]] = (float(r["value"]), int(r["decimals"]))
    return out


def matches(computed, printed: float, decimals: int, loose: bool = False):
    """True where `computed` rounds to `printed` at the printed number of decimals.

    loose=True accepts anything within one unit of the last printed digit
    (this also covers truncation instead of rounding).
    """
    half = (1.0 if loose else 0.5) * 10.0 ** (-decimals)
    return np.abs(np.asarray(computed, dtype=float) - printed) <= half * (1 + 1e-9) + 1e-15


# --------------------------------------------------------------------------
# Loading raw minute bars
# --------------------------------------------------------------------------
def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def build_matrices(dfn: pd.DataFrame, dfb: pd.DataFrame, fields=("close",)):
    """Put each index's regular-session bars into a (days x 375) matrix.

    Days are the dates on which both files have at least one bar between
    09:15 and 15:29. Bars outside that range (for example evening special
    sessions) are dropped.
    """
    prepared = {}
    for key, df in (("nifty", dfn), ("bank", dfb)):
        ts = df["timestamp"]
        minute = ts.dt.hour.to_numpy() * 60 + ts.dt.minute.to_numpy() - OPEN_MINUTE
        keep = (minute >= 0) & (minute < FULL_BARS)
        d = df.loc[keep].copy()
        d["minute"] = minute[keep]
        d["date"] = d["timestamp"].dt.normalize()
        prepared[key] = d
    days = np.array(sorted(set(prepared["nifty"]["date"]) & set(prepared["bank"]["date"])),
                    dtype="datetime64[ns]")
    day_pos = pd.Index(days)
    mats: dict = {}
    for key, d in prepared.items():
        d = d[d["date"].isin(day_pos)]
        rows = day_pos.get_indexer(d["date"])
        cols = d["minute"].to_numpy()
        for f in fields:
            m = np.full((len(days), FULL_BARS), np.nan)
            m[rows, cols] = d[f].to_numpy(dtype=float)
            mats[(key, f)] = m
    return pd.DatetimeIndex(days), mats


# --------------------------------------------------------------------------
# Day-level building blocks for one definition
# --------------------------------------------------------------------------
def _ffill_rows(p: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(p), np.arange(p.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = p[np.arange(p.shape[0])[:, None], idx]
    return out


def _first_last(p: np.ndarray):
    fin = np.isfinite(p)
    any_ = fin.any(axis=1)
    first_idx = np.argmax(fin, axis=1)
    last_idx = p.shape[1] - 1 - np.argmax(fin[:, ::-1], axis=1)
    r = np.arange(p.shape[0])
    first = np.where(any_, p[r, first_idx], np.nan)
    last = np.where(any_, p[r, last_idx], np.nan)
    return first, last


def day_level(days: pd.DatetimeIndex, mats: dict, d: Definition) -> dict:
    """Returns per day for one definition: intraday return matrices and overnight returns."""
    sub = {k: mats[(k, "close")][:, d.offset:d.offset + d.nbars] for k in INDICES}
    cnt = {k: np.isfinite(sub[k]).sum(axis=1) for k in INDICES}
    keep = (cnt["nifty"] > 0) & (cnt["bank"] > 0)
    if d.calendar == "full":
        need = int(np.ceil(FULL_DAY_SHARE * d.nbars))
        keep &= (cnt["nifty"] >= need) & (cnt["bank"] >= need)
    out = {"days": days[keep], "keep": keep, "bars": {k: cnt[k][keep] for k in INDICES}}
    for k in INDICES:
        p = sub[k][keep]
        f = _ffill_rows(p)
        cur, prev = p[:, 1:], f[:, :-1]
        ok = np.isfinite(cur) & np.isfinite(prev)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, ret(cur, prev, d.kind), np.nan)
        first, last = _first_last(p)
        with np.errstate(invalid="ignore", divide="ignore"):
            on = np.concatenate([[np.nan], ret(first[1:], last[:-1], d.kind)])
        out[("intra", k)] = r
        out[("on", k)] = on
    return out


def _sums(x: np.ndarray, axis=1) -> dict:
    fin = np.isfinite(x)
    z = np.where(fin, x, 0.0)
    return {
        "n": fin.sum(axis=axis).astype(float),
        "s1": z.sum(axis=axis), "s2": (z ** 2).sum(axis=axis),
        "s3": (z ** 3).sum(axis=axis), "s4": (z ** 4).sum(axis=axis),
        "min": np.where(fin, x, np.inf).min(axis=axis),
        "max": np.where(fin, x, -np.inf).max(axis=axis),
    }


def day_aggregates(dl: dict) -> dict:
    """Per-day sums for intraday and overnight parts, per index and for aligned pairs."""
    agg = {}
    for k in INDICES:
        agg[("intra", k)] = _sums(dl[("intra", k)])
        agg[("on", k)] = _sums(dl[("on", k)][:, None])
    for part in ("intra", "on"):
        x = dl[(part, "nifty")]
        y = dl[(part, "bank")]
        if part == "on":
            x, y = x[:, None], y[:, None]
        both = np.isfinite(x) & np.isfinite(y)
        xz, yz = np.where(both, x, 0.0), np.where(both, y, 0.0)
        agg[(part, "pair")] = {"n": both.sum(1).astype(float), "sx": xz.sum(1), "sy": yz.sum(1),
                               "sxx": (xz ** 2).sum(1), "syy": (yz ** 2).sum(1), "sxy": (xz * yz).sum(1)}
    return agg


# --------------------------------------------------------------------------
# Window statistics for every window
# --------------------------------------------------------------------------
def _prefix(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a)])


def _range_sum(pref: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sum over days lo..hi inclusive (empty when hi < lo)."""
    return np.where(hi >= lo, pref[np.maximum(hi + 1, lo)] - pref[lo], 0.0)


def _rolling_ext(x: np.ndarray, length: int, fn) -> np.ndarray:
    if length <= 0:
        return None
    return fn(np.lib.stride_tricks.sliding_window_view(x, length), axis=1)


def window_table(agg: dict, overnight: str, lengths=LENGTHS) -> dict:
    """Moments of every window [a, a+L-1] for each L, from per-day sums."""
    n_days = len(agg[("intra", "nifty")]["n"])
    a_all, b_all = [], []
    for L in lengths:
        if L > n_days:
            continue
        a = np.arange(0, n_days - L + 1)
        a_all.append(a)
        b_all.append(a + L - 1)
    if not a_all:
        return {"a": np.array([], int), "b": np.array([], int)}
    a = np.concatenate(a_all)
    b = np.concatenate(b_all)
    if overnight == "concat":
        on_lo = a + 1
    elif overnight == "concat_pre":
        on_lo = a.copy()
    else:
        on_lo = b + 1  # empty range
    tbl: dict = {"a": a, "b": b, "L": b - a + 1}
    for k in INDICES:
        ia, oa = agg[("intra", k)], agg[("on", k)]
        S = {}
        for s in ("n", "s1", "s2", "s3", "s4"):
            S[s] = _range_sum(_prefix(ia[s]), a, b) + _range_sum(_prefix(oa[s]), on_lo, b)
        mn = np.full(len(a), np.inf)
        mx = np.full(len(a), -np.inf)
        pos = 0
        for L in lengths:
            if L > n_days:
                continue
            m = n_days - L + 1
            sl = slice(pos, pos + m)
            mn[sl] = _rolling_ext(ia["min"], L, np.min)
            mx[sl] = _rolling_ext(ia["max"], L, np.max)
            if overnight != "intraday":
                if overnight == "concat":
                    on_min = _rolling_ext(oa["min"][1:], L - 1, np.min) if L > 1 else None
                    on_max = _rolling_ext(oa["max"][1:], L - 1, np.max) if L > 1 else None
                else:
                    on_min = _rolling_ext(oa["min"], L, np.min)
                    on_max = _rolling_ext(oa["max"], L, np.max)
                if on_min is not None:
                    mn[sl] = np.minimum(mn[sl], on_min[:m])
                    mx[sl] = np.maximum(mx[sl], on_max[:m])
            pos += m
        N = S["n"]
        mu = S["s1"] / N
        e2, e3, e4 = S["s2"] / N, S["s3"] / N, S["s4"] / N
        m2 = e2 - mu ** 2
        m3 = e3 - 3 * mu * e2 + 2 * mu ** 3
        m4 = e4 - 4 * mu * e3 + 6 * mu ** 2 * e2 - 3 * mu ** 4
        tbl[k] = {"n": N, "mean": mu, "m2": m2, "m3": m3, "m4": m4, "min": mn, "max": mx}
    P = {}
    for s in ("n", "sx", "sy", "sxx", "syy", "sxy"):
        P[s] = (_range_sum(_prefix(agg[("intra", "pair")][s]), a, b)
                + _range_sum(_prefix(agg[("on", "pair")][s]), on_lo, b))
    n = P["n"]
    cov = P["sxy"] / n - (P["sx"] / n) * (P["sy"] / n)
    vx = P["sxx"] / n - (P["sx"] / n) ** 2
    vy = P["syy"] / n - (P["sy"] / n) ** 2
    tbl["corr"] = cov / np.sqrt(vx * vy)
    return tbl


# --------------------------------------------------------------------------
# Statistic variants and the score
# --------------------------------------------------------------------------
def stat_variants(N, m2, m3, m4) -> dict:
    N = np.asarray(N, dtype=float)
    g1 = m3 / m2 ** 1.5
    g2 = m4 / m2 ** 2 - 3.0
    G1 = g1 * np.sqrt(N * (N - 1)) / (N - 2)
    G2 = ((N + 1) * g2 + 6) * (N - 1) / ((N - 2) * (N - 3))
    b1 = g1 * ((N - 1) / N) ** 1.5
    b2 = (g2 + 3) * ((N - 1) / N) ** 2 - 3
    return {
        ("sd", "sample"): np.sqrt(m2 * N / (N - 1)),
        ("sd", "population"): np.sqrt(m2),
        ("skew", "g1"): g1, ("skew", "G1"): G1, ("skew", "b1"): b1,
        ("kurt", "g2"): g2, ("kurt", "G2"): G2, ("kurt", "b2"): b2,
        ("kurt", "g2+3"): g2 + 3, ("kurt", "G2+3"): G2 + 3, ("kurt", "b2+3"): b2 + 3,
    }


def _err(item: str, c, p):
    c = np.asarray(c, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if item in ("sd", "kurt"):
            e = np.where(c > 0, np.log(np.abs(c) / abs(p)), 10.0)
        elif item in ("min", "max"):
            e = c / p - 1.0
        elif item == "skew":
            e = (c - p) / abs(p)
        elif item == "corr":
            e = c - p
        else:
            raise ValueError(item)
    return np.where(np.isfinite(e), e, 10.0)


VARIANTS = {"sd": SD_VARIANTS, "skew": SKEW_VARIANTS, "kurt": KURT_VARIANTS}


def score_table(tbl: dict, printed: dict) -> pd.DataFrame:
    """Exact-match count E (and loose E1) and distance D for every window."""
    nwin = len(tbl["a"])
    cols: dict = {"a": tbl["a"], "b": tbl["b"], "L": tbl["L"]}
    E = np.zeros(nwin, int)
    E1 = np.zeros(nwin, int)
    sq = np.zeros(nwin)
    nerr = 0
    var = {k: stat_variants(tbl[k]["n"], tbl[k]["m2"], tbl[k]["m3"], tbl[k]["m4"]) for k in INDICES}
    # items with formula variants: choose the variant (same for both indices) with most matches, then least error
    for item, vlist in VARIANTS.items():
        keys, chosen_vals = [], []
        for v in vlist:
            m = np.zeros(nwin, int)
            se = np.zeros(nwin)
            for k in INDICES:
                p, dec = printed[k][item]
                c = var[k][(item, v)]
                m += matches(c, p, dec)
                se += _err(item, c, p) ** 2
            keys.append(m * 1e9 - se)
        best = np.argmax(np.vstack(keys), axis=0)
        cols[f"{item}_variant"] = np.array(vlist, dtype=object)[best]
        for k in INDICES:
            p, dec = printed[k][item]
            c = np.choose(best, [var[k][(item, v)] for v in vlist])
            cols[f"{k}_{item}"] = c
            ok = matches(c, p, dec)
            cols[f"{k}_{item}_ok"] = ok
            E += ok
            E1 += matches(c, p, dec, loose=True)
            sq += _err(item, c, p) ** 2
            nerr += 1
    for k in INDICES:
        for item in ("mean", "min", "max", "n"):
            p, dec = printed[k][item]
            c = tbl[k][item]
            cols[f"{k}_{item}"] = c
            ok = matches(c, p, dec)
            cols[f"{k}_{item}_ok"] = ok
            E += ok
            E1 += matches(c, p, dec, loose=True)
            if item in ("min", "max"):
                sq += _err(item, c, p) ** 2
                nerr += 1
    p, dec = printed["corr"]
    cols["corr"] = tbl["corr"]
    ok = matches(tbl["corr"], p, dec)
    cols["corr_ok"] = ok
    E += ok
    E1 += matches(tbl["corr"], p, dec, loose=True)
    sq += _err("corr", tbl["corr"], p) ** 2
    nerr += 1
    cols["E"] = E
    cols["E1"] = E1
    cols["D"] = np.sqrt(sq / nerr)
    cols["n_extreme_ok"] = sum(cols[f"{k}_{it}_ok"].astype(int) for k in INDICES for it in ("min", "max"))
    return pd.DataFrame(cols)


def rank_windows(sc: pd.DataFrame) -> pd.DataFrame:
    return sc.sort_values(["E", "D"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def score_definition(days, mats, d: Definition, printed: dict, lengths=LENGTHS):
    dl = day_level(days, mats, d)
    agg = day_aggregates(dl)
    tbl = window_table(agg, d.overnight, lengths)
    sc = score_table(tbl, printed)
    cal = dl["days"]
    sc.insert(0, "start", cal[sc["a"].to_numpy()].strftime("%Y-%m-%d"))
    sc.insert(1, "end", cal[sc["b"].to_numpy()].strftime("%Y-%m-%d"))
    sc.insert(0, "definition", d.name)
    return rank_windows(sc), dl


def overlap_share(a1, b1, a2, b2) -> float:
    inter = max(0, min(b1, b2) - max(a1, a2) + 1)
    return inter / (b1 - a1 + 1)


def runners_up(ranked: pd.DataFrame) -> tuple:
    """Best window differing from the winner, and best window sharing < 50% of the winner's days."""
    w = ranked.iloc[0]
    diff = ranked.iloc[1:2] if len(ranked) > 1 else ranked.iloc[0:0]
    distinct = None
    for i in range(1, len(ranked)):
        r = ranked.iloc[i]
        if overlap_share(w.a, w.b, r.a, r.b) < 0.5:
            distinct = r
            break
    return (diff.iloc[0] if len(diff) else None), distinct


# --------------------------------------------------------------------------
# Direct (stage 2) computation for a single window, from raw prices
# --------------------------------------------------------------------------
def window_series(dl_days: pd.DatetimeIndex, days: pd.DatetimeIndex, mats: dict, d: Definition,
                  a: int, b: int) -> dict:
    """Assemble each index's return series for window [a, b] directly, as a researcher would.

    concat: join the window's session prices into one series and difference it.
    intraday: difference within each day only.
    concat_pre: as concat, plus the first day's overnight return.
    Returns timestamps too, so that the two indices can be aligned for the correlation.
    """
    pos = days.get_indexer(dl_days)
    out = {}
    for k in INDICES:
        sub = mats[(k, "close")][:, d.offset:d.offset + d.nbars]
        prices, stamps, dayid = [], [], []
        for j in range(a, b + 1):
            row = sub[pos[j]]
            fin = np.isfinite(row)
            prices.append(row[fin])
            stamps.append(dl_days[j] + pd.to_timedelta(OPEN_MINUTE + d.offset + np.flatnonzero(fin), unit="min"))
            dayid.append(np.full(fin.sum(), j))
        p = np.concatenate(prices)
        ts = np.concatenate([s.values for s in stamps])
        did = np.concatenate(dayid)
        r = ret(p[1:], p[:-1], d.kind)
        rts = ts[1:]
        if d.overnight == "intraday":
            same = did[1:] == did[:-1]
            r, rts = r[same], rts[same]
        elif d.overnight == "concat_pre" and a > 0:
            prev_row = sub[pos[a - 1]]
            prev_last = prev_row[np.isfinite(prev_row)][-1]
            r = np.concatenate([[ret(p[0], prev_last, d.kind)], r])
            rts = np.concatenate([[ts[0]], rts])
        out[k] = pd.Series(r, index=pd.DatetimeIndex(rts))
    return out


def anderson_variants(x: np.ndarray) -> dict:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    out = {}
    for ddof in (1, 0):
        z = (x - x.mean()) / x.std(ddof=ddof)
        logcdf = scipy.stats.norm.logcdf(z)
        logsf = scipy.stats.norm.logsf(z)
        i = np.arange(1, n + 1)
        a2 = -n - np.sum((2 * i - 1) / n * (logcdf + logsf[::-1]))
        out[f"ddof{ddof}"] = a2
        out[f"ddof{ddof}_adj"] = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
    return out


def direct_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    c = x - mu
    m2, m3, m4 = (c ** 2).mean(), (c ** 3).mean(), (c ** 4).mean()
    out = {"n": n, "mean": mu, "min": x.min(), "max": x.max()}
    for (item, v), val in stat_variants(n, m2, m3, m4).items():
        out[f"{item}:{v}"] = float(val)
    return out


def adf_report(x: np.ndarray) -> dict:
    import warnings

    from statsmodels.tsa.stattools import adfuller
    warnings.simplefilter("ignore", FutureWarning)  # adfuller return-type notice only
    out = {}
    try:
        out["adf_c_aic"] = float(adfuller(x, regression="c", autolag="AIC")[0])
    except Exception as exc:  # pragma: no cover
        out["adf_c_aic"] = f"error: {exc}"
    try:
        k = int(np.floor((len(x) - 1) ** (1 / 3)))
        out["adf_ct_fixed"] = float(adfuller(x, maxlag=k, regression="ct", autolag=None)[0])
    except Exception as exc:  # pragma: no cover
        out["adf_ct_fixed"] = f"error: {exc}"
    return out


def stage2_row(days, mats, dl, d: Definition, a: int, b: int, printed: dict, ohlc=None,
               with_adf: bool = False) -> dict:
    ser = window_series(dl["days"], days, mats, d, a, b)
    row = {"definition": d.name, "start": dl["days"][a].strftime("%Y-%m-%d"),
           "end": dl["days"][b].strftime("%Y-%m-%d"), "L": b - a + 1}
    E2 = 0
    for k in INDICES:
        x = ser[k].to_numpy()
        st = direct_stats(x)
        row[f"{k}_n"] = st["n"]
        for item in ("mean", "min", "max", "n"):
            p, dec = printed[k][item]
            row[f"{k}_{item}"] = st[item]
            row[f"{k}_{item}_ok"] = bool(matches(st[item], p, dec))
        for item, vlist in VARIANTS.items():
            p, dec = printed[k][item]
            vals = {v: st[f"{item}:{v}"] for v in vlist}
            okv = {v: bool(matches(vals[v], p, dec)) for v in vlist}
            bestv = min(vlist, key=lambda v: (not okv[v], abs(_err(item, vals[v], p))))
            row[f"{k}_{item}"] = vals[bestv]
            row[f"{k}_{item}_variant"] = bestv
            row[f"{k}_{item}_ok"] = okv[bestv]
        ad = anderson_variants(x)
        p, dec = printed[k]["ad"]
        bestad = min(ad, key=lambda v: (not bool(matches(ad[v], p, dec)), abs(ad[v] - p)))
        row[f"{k}_ad"] = ad[bestad]
        row[f"{k}_ad_variant"] = bestad
        row[f"{k}_ad_ok"] = bool(matches(ad[bestad], p, dec))
        if with_adf:
            for name, val in adf_report(x).items():
                row[f"{k}_{name}"] = val
        if ohlc is not None:
            row[f"{k}_rs_median"] = rs_median(ohlc, dl, days, d, a, b, k)
        E2 += sum(row[f"{k}_{it}_ok"] for it in ("mean", "sd", "min", "max", "skew", "kurt", "n", "ad"))
    joined = pd.concat([ser["nifty"], ser["bank"]], axis=1, join="inner")
    corr = float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])
    p, dec = printed["corr"]
    row["corr"] = corr
    row["corr_ok"] = bool(matches(corr, p, dec))
    E2 += row["corr_ok"]
    row["E2"] = int(E2)
    win_days = dl["days"][a:b + 1]
    row["n_wednesdays"] = int((win_days.weekday == 2).sum())
    row["n_thursdays"] = int((win_days.weekday == 3).sum())
    return row


def rs_median(ohlc: dict, dl, days, d: Definition, a: int, b: int, k: str) -> float:
    """Median per-minute Rogers-Satchell volatility over the window's session bars (report only)."""
    pos = days.get_indexer(dl["days"][a:b + 1])
    sl = slice(d.offset, d.offset + d.nbars)
    o, h, lo, c = (ohlc[(k, f)][pos, sl] for f in ("open", "high", "low", "close"))
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = np.log(h / c) * np.log(h / o) + np.log(lo / c) * np.log(lo / o)
    rs = rs[np.isfinite(rs)]
    return float(np.median(np.sqrt(np.maximum(rs, 0.0))))


# --------------------------------------------------------------------------
# Anchor search: where in the data do the printed extremes occur?
# --------------------------------------------------------------------------
def extreme_anchors(dl: dict, d: Definition, printed: dict) -> list[dict]:
    rows = []
    for k in INDICES:
        for item in ("min", "max"):
            p, dec = printed[k][item]
            for part in ("intra", "on"):
                x = dl[(part, k)]
                if part == "on" and d.overnight == "intraday":
                    continue
                if x.ndim == 1:
                    hit = np.flatnonzero(matches(np.nan_to_num(x, nan=9.0), p, dec, loose=True))
                    for i in hit:
                        rows.append({"definition": d.name, "index": k, "item": item, "part": "overnight",
                                     "date": dl["days"][i].strftime("%Y-%m-%d"),
                                     "bar_time": d.session_label[:4], "value": float(x[i]),
                                     "exact": bool(matches(x[i], p, dec))})
                else:
                    hi, hj = np.nonzero(matches(np.nan_to_num(x, nan=9.0), p, dec, loose=True))
                    for i, j in zip(hi, hj):
                        mins = OPEN_MINUTE + d.offset + j + 1
                        rows.append({"definition": d.name, "index": k, "item": item, "part": "intraday",
                                     "date": dl["days"][i].strftime("%Y-%m-%d"),
                                     "bar_time": f"{mins // 60:02d}{mins % 60:02d}", "value": float(x[i, j]),
                                     "exact": bool(matches(x[i, j], p, dec))})
    return rows


# --------------------------------------------------------------------------
# Stated windows
# --------------------------------------------------------------------------
def stated_windows(cal: pd.DatetimeIndex) -> dict:
    """Index pairs (a, b) in a definition's calendar for the windows the paper's text implies."""
    out = {}
    nov = np.flatnonzero((cal >= "2023-11-01") & (cal <= "2024-02-29"))
    if len(nov):
        a, b = int(nov[0]), int(nov[-1])
        out["stated_nov1_feb29"] = (a, b)
        if a + 63 < len(cal):
            out["stated_first64_from_nov1"] = (a, a + 63)
        if b - 63 >= 0:
            out["stated_last64_to_feb29"] = (b - 63, b)
    return out


def lookup(ranked: pd.DataFrame, a: int, b: int):
    hit = ranked.index[(ranked["a"] == a) & (ranked["b"] == b)]
    if len(hit) == 0:
        return None, None
    return int(hit[0]) + 1, ranked.loc[hit[0]]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SUMMARY_COLS = ["definition", "start", "end", "L", "E", "E1", "D", "n_extreme_ok",
                "nifty_n", "bank_n", "nifty_mean", "bank_mean", "nifty_sd", "bank_sd",
                "nifty_min", "nifty_max", "bank_min", "bank_max", "nifty_skew", "bank_skew",
                "nifty_kurt", "bank_kurt", "corr", "sd_variant", "skew_variant", "kurt_variant"]


def run(raw: Path, out: Path, definitions=None, lengths=LENGTHS, top_defs: int = 10,
        top_per_def: int = 20, adf_top: int = 3, label: str = "REAL DATA", plan_commit: str = "",
        write_all_windows: bool = True) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    printed = load_printed(window="before")
    printed_after = load_printed(window="after")
    dfn = load_bars(raw / "nifty_1min.csv")
    dfb = load_bars(raw / "banknifty_1min.csv")
    days, mats = build_matrices(dfn, dfb, fields=("open", "high", "low", "close"))
    definitions = definitions or all_definitions()

    best_rows, anchors, stated_rows, ranked_store = [], [], [], {}
    for d in definitions:
        ranked, dl = score_definition(days, mats, d, printed, lengths)
        w = ranked.iloc[0]
        r1, rd = runners_up(ranked)
        row = {c: w[c] for c in SUMMARY_COLS}
        row["runner_up"] = f"{r1.start}..{r1.end}" if r1 is not None else ""
        row["runner_up_E"] = int(r1.E) if r1 is not None else np.nan
        row["runner_up_D"] = float(r1.D) if r1 is not None else np.nan
        row["distinct_runner_up"] = f"{rd.start}..{rd.end}" if rd is not None else ""
        row["distinct_runner_up_E"] = int(rd.E) if rd is not None else np.nan
        row["distinct_runner_up_D"] = float(rd.D) if rd is not None else np.nan
        row["n_windows"] = len(ranked)
        row["n_windows_with_best_E"] = int((ranked["E"] == w.E).sum())
        best_rows.append(row)
        anchors.extend(extreme_anchors(dl, d, printed))
        for name, (a, b) in stated_windows(dl["days"]).items():
            rank, r = lookup(ranked, a, b)
            srow = {"definition": d.name, "stated_variant": name,
                    "start": dl["days"][a].strftime("%Y-%m-%d"), "end": dl["days"][b].strftime("%Y-%m-%d"),
                    "L": b - a + 1, "rank": rank, "n_windows": len(ranked)}
            if r is not None:
                for c in SUMMARY_COLS[4:]:
                    srow[c] = r[c]
            stated_rows.append(srow)
        ranked_store[d.name] = ranked.head(top_per_def).copy()
        ranked_store[d.name + "#E_hist"] = ranked["E"].value_counts().sort_index()

    best = pd.DataFrame(best_rows).sort_values(["E", "D"], ascending=[False, True], kind="mergesort")
    best = best.reset_index(drop=True)
    best.insert(0, "def_rank", np.arange(1, len(best) + 1))
    best.to_csv(out / "best_by_definition.csv", index=False)
    pd.DataFrame(anchors).drop_duplicates().to_csv(out / "extreme_anchors.csv", index=False)
    pd.DataFrame(stated_rows).to_csv(out / "stated_windows.csv", index=False)

    # Stage 2: direct recomputation (plus Anderson-Darling) for the top windows of the top definitions
    top_names = list(best["definition"].head(top_defs))
    s2 = []
    win_def = definition_from_name(top_names[0])
    for rank_d, name in enumerate(top_names):
        d = definition_from_name(name)
        dl = day_level(days, mats, d)
        top = ranked_store[name]
        for i, r in top.iterrows():
            row = stage2_row(days, mats, dl, d, int(r.a), int(r.b), printed, ohlc=mats,
                             with_adf=(rank_d == 0 and i < adf_top))
            row.update({"def_rank": rank_d + 1, "stage1_rank": i + 1, "stage1_E": int(r.E), "stage1_D": float(r.D)})
            s2.append(row)
    # stated windows under the winning definition
    dl_w = day_level(days, mats, win_def)
    for name, (a, b) in stated_windows(dl_w["days"]).items():
        row = stage2_row(days, mats, dl_w, win_def, a, b, printed, ohlc=mats, with_adf=True)
        row.update({"def_rank": 1, "stage1_rank": np.nan, "stated_variant": name})
        s2.append(row)
    s2 = pd.DataFrame(s2)
    s2.to_csv(out / "top_candidates.csv", index=False)

    # all windows for the winning definition, and the E distribution (null reference)
    if write_all_windows:
        full, _ = score_definition(days, mats, win_def, printed, lengths)
        keep_cols = ["start", "end", "L", "E", "E1", "D", "n_extreme_ok"] + \
            [c for c in full.columns if c.endswith("_ok") or c.startswith(("nifty_", "bank_"))] + ["corr"]
        keep_cols = list(dict.fromkeys(keep_cols))
        full[keep_cols].to_csv(out / "all_windows_winning_definition.csv.gz", index=False,
                               float_format="%.10g", compression="gzip")
        top_all = pd.concat([ranked_store[n] for n in best["definition"]], ignore_index=True)
        top_all[["definition"] + [c for c in top_all.columns if c != "definition"]].to_csv(
            out / "top20_by_definition.csv.gz", index=False, float_format="%.10g", compression="gzip")
    hist = pd.concat({n: ranked_store[n + "#E_hist"] for n in top_names}, axis=1).fillna(0).astype(int)
    hist.index.name = "E"
    hist.to_csv(out / "E_distribution_top_definitions.csv")

    # Secondary (corroboration only): the after window, winning definition
    ranked_after, dla = score_definition(days, mats, win_def, printed_after, lengths)
    ranked_after.head(20)[SUMMARY_COLS].to_csv(out / "after_window_check.csv", index=False)

    info = {
        "label": label,
        "study": "P108 blind window search",
        "plan_commit": plan_commit,
        "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "run_seconds": round(time.time() - t0, 1),
        "inputs": {p.name: sha256(p) for p in (raw / "nifty_1min.csv", raw / "banknifty_1min.csv", PRINTED_CSV)},
        "n_calendar_days_common": int(len(days)),
        "calendar_first": str(days[0].date()), "calendar_last": str(days[-1].date()),
        "n_definitions": len(definitions), "lengths": [min(lengths), max(lengths)],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
        "winning_definition": top_names[0],
    }
    try:
        import statsmodels
        info["versions"]["statsmodels"] = statsmodels.__version__
    except ImportError:  # pragma: no cover
        pass
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    return {"best": best, "stage2": s2, "info": info, "after": ranked_after}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/p108_blind_window")
    ap.add_argument("--plan-commit", default="")
    args = ap.parse_args(argv)
    res = run(Path(args.raw), Path(args.out), plan_commit=args.plan_commit)
    print(res["best"].head(15).to_string())
    print(res["info"])


if __name__ == "__main__":
    main()
