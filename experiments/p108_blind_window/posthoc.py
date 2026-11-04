#!/usr/bin/env python
"""P108 POST HOC analyses (REAL DATA). Not part of the registered plan; run after seeing phase 2 results.

Why: the registered search (search.py) put the before window in late November 2022 to
February 2023 under the 09:15-15:14 concatenated log-return definition. Both printed
maxima were reproduced exactly, but both printed minima were not: in our data both
minima fall on the same overnight return (27 Jan 2023 15:14 close to 30 Jan 2023 09:15
close) and differ slightly from the printed values. These analyses ask why, and whether
the window's exact endpoints can be pinned down once that one difference is allowed for.

PH1  Which pairs of our prices reproduce the printed minima exactly? (bars of the last
     hour of 27 Jan 2023 against the first ten minutes of 30 Jan 2023, all four price fields),
     and, over the whole history, does "previous day's 15:15 open to 09:15 close" match
     the printed minima on any other day?
PH2  Patch: replace the 27 Jan 2023 15:14 close of each index by that bar's next open
     (15:15 open), the price PH1 points to, then rerun the registered window scoring
     for the winning definition over every window.
PH3  Alternative without a one-day patch: use the next bar's open as the last price of
     every day, and score all windows.
PH4  Stage 2 (direct recomputation with Anderson-Darling and ADF) for the top windows of PH2.

    OMP_NUM_THREADS=1 python experiments/p108_blind_window/posthoc.py --raw data/raw --out results/p108_blind_window
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search as S  # noqa: E402

WIN_DEF = "0915-1514|concat|log|all"
PATCH_DAY = "2023-01-27"
PATCH_MINUTE = 15 * 60 + 14 - S.OPEN_MINUTE  # column of the 15:14 bar


def ph1(dfs: dict, printed: dict) -> pd.DataFrame:
    rows = []
    for k, df in dfs.items():
        s = df.set_index("timestamp")
        p, dec = printed[k]["min"]
        a = s.loc["2023-01-27 14:30":"2023-01-27 15:29"]
        b = s.loc["2023-01-30 09:15":"2023-01-30 09:25"]
        for ta, ra in a.iterrows():
            for fa in ("open", "high", "low", "close"):
                for tb, rb in b.iterrows():
                    for fb in ("open", "high", "low", "close"):
                        r = float(np.log(rb[fb] / ra[fa]))
                        if S.matches(r, p, dec, loose=True):
                            rows.append({"check": "pair_search_27_to_30_Jan_2023", "index": k,
                                         "from": f"{ta} {fa}", "to": f"{tb} {fb}", "value": r,
                                         "exact": bool(S.matches(r, p, dec))})
        # whole history: previous day's 15:15 open to this day's 09:15 close
        s = s.copy()
        s["date"] = s.index.normalize()
        o1515 = s[s.index.strftime("%H:%M") == "15:15"].set_index("date")["open"]
        c0915 = s[s.index.strftime("%H:%M") == "09:15"].set_index("date")["close"]
        dates = c0915.index.intersection(o1515.index)
        prev = o1515.reindex(dates).shift(1)
        r = np.log(c0915.reindex(dates) / prev)
        hit = r[S.matches(r.fillna(9).to_numpy(), p, dec, loose=True)]
        for d0, v in hit.items():
            rows.append({"check": "history_prev1515open_to_0915close", "index": k, "from": "previous day 15:15 open",
                         "to": f"{d0.date()} 09:15 close", "value": float(v), "exact": bool(S.matches(v, p, dec))})
        # whole history, the registered definition: previous day's 15:14 close to 09:15 close
        c1514 = s[s.index.strftime("%H:%M") == "15:14"].set_index("date")["close"]
        dates = c0915.index.intersection(c1514.index)
        r = np.log(c0915.reindex(dates) / c1514.reindex(dates).shift(1))
        near = r[(r - p).abs() < 2e-4]
        for d0, v in near.items():
            rows.append({"check": "history_prev1514close_to_0915close_within_2e-4", "index": k,
                         "from": "previous day 15:14 close", "to": f"{d0.date()} 09:15 close", "value": float(v),
                         "exact": bool(S.matches(v, p, dec))})
    return pd.DataFrame(rows)


def patched_mats(mats: dict, days: pd.DatetimeIndex) -> tuple[dict, list]:
    out = {key: m.copy() for key, m in mats.items()}
    i = days.get_loc(pd.Timestamp(PATCH_DAY))
    log = []
    for k in S.INDICES:
        old = out[(k, "close")][i, PATCH_MINUTE]
        new = mats[(k, "open")][i, PATCH_MINUTE + 1]
        out[(k, "close")][i, PATCH_MINUTE] = new
        log.append({"index": k, "day": PATCH_DAY, "bar": "15:14", "old_close": old, "new_close_from_15:15_open": new})
    return out, log


def next_open_last_price(mats: dict, d: S.Definition) -> dict:
    """Every day's last session price replaced by the next bar's open (when that bar exists)."""
    out = {key: m.copy() for key, m in mats.items()}
    last = d.offset + d.nbars - 1
    if last + 1 >= S.FULL_BARS:
        return out
    for k in S.INDICES:
        nxt = mats[(k, "open")][:, last + 1]
        ok = np.isfinite(nxt) & np.isfinite(out[(k, "close")][:, last])
        out[(k, "close")][ok, last] = nxt[ok]
    return out


def one_day_dropped(days, mats, d: S.Definition, printed: dict, span: int = 65) -> pd.DataFrame:
    """PH5: every run of `span` consecutive days with one interior day left out (concat only).

    The overnight return then runs from the last price before the gap to the first price after it.
    """
    assert d.overnight == "concat"
    dl = S.day_level(days, mats, d)
    agg = S.day_aggregates(dl)
    keep = dl["keep"]
    n_days = len(dl["days"])
    first, last, newon = {}, {}, {}
    for k in S.INDICES:
        sub = mats[(k, "close")][:, d.offset:d.offset + d.nbars][keep]
        first[k], last[k] = S._first_last(sub)
        with np.errstate(invalid="ignore", divide="ignore"):
            nv = np.full(n_days, np.nan)
            nv[1:-1] = S.ret(first[k][2:], last[k][:-2], d.kind)
        newon[k] = nv
    A, J = [], []
    for j in range(1, span - 1):
        a = np.arange(0, n_days - span + 1)
        A.append(a)
        J.append(np.full(len(a), j))
    a = np.concatenate(A)
    jj = np.concatenate(J)
    b = a + span - 1
    kk = a + jj
    tbl = {"a": a, "b": b, "L": np.full(len(a), span - 1)}

    def rmin(x, start, length, fn, fill):
        out = np.full(len(start), fill)
        ok = length > 0
        for L in np.unique(length[ok]):
            sel = ok & (length == L)
            view = fn(np.lib.stride_tricks.sliding_window_view(x, int(L)), axis=1)
            out[sel] = view[start[sel]]
        return out

    for k in S.INDICES:
        ia, oa = agg[("intra", k)], agg[("on", k)]
        S_ = {}
        nv = newon[k]
        nvz = np.where(np.isfinite(nv), nv, 0.0)
        nvn = np.isfinite(nv).astype(float)
        for s, p in (("n", 0), ("s1", 1), ("s2", 2), ("s3", 3), ("s4", 4)):
            pi = S._prefix(ia[s])
            po = S._prefix(oa[s])
            intra = pi[b + 1] - pi[a] - ia[s][kk]
            on = po[b + 1] - po[a + 1] - oa[s][kk] - oa[s][kk + 1]
            add = nvn[kk] if s == "n" else nvz[kk] ** p
            S_[s] = intra + on + add
        mn = np.minimum.reduce([
            rmin(ia["min"], a, jj, np.min, np.inf),
            rmin(oa["min"], a + 1, jj - 1, np.min, np.inf),
            rmin(ia["min"], kk + 1, span - 1 - jj, np.min, np.inf),
            rmin(oa["min"], kk + 2, span - 2 - jj, np.min, np.inf),
            np.where(np.isfinite(nv[kk]), nv[kk], np.inf)])
        mx = np.maximum.reduce([
            rmin(ia["max"], a, jj, np.max, -np.inf),
            rmin(oa["max"], a + 1, jj - 1, np.max, -np.inf),
            rmin(ia["max"], kk + 1, span - 1 - jj, np.max, -np.inf),
            rmin(oa["max"], kk + 2, span - 2 - jj, np.max, -np.inf),
            np.where(np.isfinite(nv[kk]), nv[kk], -np.inf)])
        N = S_["n"]
        mu = S_["s1"] / N
        e2, e3, e4 = S_["s2"] / N, S_["s3"] / N, S_["s4"] / N
        tbl[k] = {"n": N, "mean": mu, "m2": e2 - mu ** 2, "m3": e3 - 3 * mu * e2 + 2 * mu ** 3,
                  "m4": e4 - 4 * mu * e3 + 6 * mu ** 2 * e2 - 3 * mu ** 4, "min": mn, "max": mx}
    P = {}
    for s in ("n", "sx", "sy", "sxx", "syy", "sxy"):
        ip, op = agg[("intra", "pair")][s], agg[("on", "pair")][s]
        pi, po = S._prefix(ip), S._prefix(op)
        P[s] = pi[b + 1] - pi[a] - ip[kk] + po[b + 1] - po[a + 1] - op[kk] - op[kk + 1]
    x, y = newon["nifty"][kk], newon["bank"][kk]
    both = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(both, x, 0.0), np.where(both, y, 0.0)
    P["n"] += both
    P["sx"] += x
    P["sy"] += y
    P["sxx"] += x * x
    P["syy"] += y * y
    P["sxy"] += x * y
    n = P["n"]
    cov = P["sxy"] / n - (P["sx"] / n) * (P["sy"] / n)
    tbl["corr"] = cov / np.sqrt((P["sxx"] / n - (P["sx"] / n) ** 2) * (P["syy"] / n - (P["sy"] / n) ** 2))
    sc = S.score_table(tbl, printed)
    cal = dl["days"]
    sc.insert(0, "start", cal[a].strftime("%Y-%m-%d"))
    sc.insert(1, "end", cal[b].strftime("%Y-%m-%d"))
    sc.insert(2, "dropped_day", cal[kk].strftime("%Y-%m-%d"))
    return S.rank_windows(sc)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/p108_blind_window")
    args = ap.parse_args(argv)
    t0 = time.time()
    raw, out = Path(args.raw), Path(args.out)
    printed = S.load_printed(window="before")
    dfn = S.load_bars(raw / "nifty_1min.csv")
    dfb = S.load_bars(raw / "banknifty_1min.csv")
    days, mats = S.build_matrices(dfn, dfb, fields=("open", "high", "low", "close"))
    d = S.definition_from_name(WIN_DEF)

    r1 = ph1({"nifty": dfn, "bank": dfb}, printed)
    r1.to_csv(out / "posthoc_ph1_minimum_pairs.csv", index=False)

    cols = S.SUMMARY_COLS[1:] + [c for c in ("nifty_mean_ok", "nifty_sd_ok", "nifty_min_ok", "nifty_max_ok",
                                              "nifty_skew_ok", "nifty_kurt_ok", "nifty_n_ok", "bank_mean_ok",
                                              "bank_sd_ok", "bank_min_ok", "bank_max_ok", "bank_skew_ok",
                                              "bank_kurt_ok", "bank_n_ok", "corr_ok")]
    pm, plog = patched_mats(mats, days)
    ranked2, dl2 = S.score_definition(days, pm, d, printed)
    ranked2.head(30)[["definition"] + cols].to_csv(out / "posthoc_ph2_patched_top30.csv", index=False)
    ranked2["E"].value_counts().sort_index().rename("n_windows").to_csv(out / "posthoc_ph2_E_distribution.csv")
    # PH7: the "constraint" view. Windows (patched data) that reproduce N and all four extremes exactly,
    # with their mean and correlation flags, and the unpatched equivalent (both maxima exact, N exact,
    # and the window minimum being the 30 Jan 2023 overnight return).
    feas = ranked2[(ranked2["n_extreme_ok"] == 4) & ranked2["nifty_n_ok"] & ranked2["bank_n_ok"]]
    feas[["start", "end", "L", "E", "D", "nifty_mean", "bank_mean", "nifty_mean_ok", "bank_mean_ok", "corr",
          "corr_ok", "nifty_sd", "bank_sd", "nifty_skew", "bank_skew", "nifty_kurt", "bank_kurt"]].to_csv(
        out / "posthoc_ph7_feasible_windows_patched.csv", index=False)
    ranked0, _ = S.score_definition(days, mats, d, printed)
    f0 = ranked0[ranked0["nifty_max_ok"] & ranked0["bank_max_ok"] & ranked0["nifty_n_ok"] & ranked0["bank_n_ok"]
                 & np.isclose(ranked0["nifty_min"], -0.008533547824, atol=1e-12)
                 & np.isclose(ranked0["bank_min"], -0.017135337059, atol=1e-12)]
    f0[["start", "end", "L", "E", "D", "nifty_mean", "bank_mean", "nifty_mean_ok", "bank_mean_ok", "corr",
        "corr_ok"]].to_csv(out / "posthoc_ph7_feasible_windows_unpatched.csv", index=False)
    print("PH7 patched feasible:\n", feas[["start", "end", "L", "E", "D", "nifty_mean_ok", "bank_mean_ok", "corr"]])
    print("PH7 unpatched feasible:\n", f0[["start", "end", "L", "E", "D", "nifty_mean_ok", "bank_mean_ok", "corr"]])

    res3 = []
    for name in (WIN_DEF, "0915-1514|concat|log|full"):
        d3 = S.definition_from_name(name)
        ranked3, _ = S.score_definition(days, next_open_last_price(mats, d3), d3, printed)
        top = ranked3.head(10)[["definition"] + cols].copy()
        top.insert(1, "variant", "last price of every day = next bar open")
        res3.append(top)
    pd.concat(res3).to_csv(out / "posthoc_ph3_next_open_top10.csv", index=False)

    s2 = []
    for i, r in ranked2.head(10).iterrows():
        row = S.stage2_row(days, pm, dl2, d, int(r.a), int(r.b), printed, ohlc=pm, with_adf=(i < 3))
        row.update({"stage1_rank": i + 1, "stage1_E": int(r.E), "stage1_D": float(r.D)})
        s2.append(row)
    pd.DataFrame(s2).to_csv(out / "posthoc_ph4_patched_stage2.csv", index=False)

    r5 = one_day_dropped(days, pm, d, printed)
    r5.head(30)[["start", "end", "dropped_day"] + cols[2:]].to_csv(out / "posthoc_ph5_one_day_dropped_top30.csv",
                                                                  index=False)
    r5["E"].value_counts().sort_index().rename("n_configurations").to_csv(out / "posthoc_ph5_E_distribution.csv")

    printed_after = S.load_printed(window="after")
    res6 = []
    for name in ("0915-1514|concat|log|full", WIN_DEF):
        ranked6, _ = S.score_definition(days, mats, S.definition_from_name(name), printed_after)
        res6.append(ranked6.head(10)[["definition"] + cols])
    pd.concat(res6).to_csv(out / "posthoc_ph6_after_window_full_calendar.csv", index=False)

    info = {"label": "REAL DATA, POST HOC", "definition": WIN_DEF, "patch": plog,
            "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "run_seconds": round(time.time() - t0, 1)}
    (out / "posthoc_run_info.json").write_text(json.dumps(info, indent=2, default=float))
    print(r1.to_string())
    print(ranked2.head(10)[["start", "end", "L", "E", "E1", "D", "nifty_sd", "bank_sd", "nifty_skew", "bank_skew",
                            "nifty_kurt", "bank_kurt", "corr", "skew_variant", "kurt_variant"]].to_string())
    print(pd.concat(res3)[["definition", "start", "end", "L", "E", "D"]].to_string())
    print(pd.DataFrame(s2)[["start", "end", "E2", "nifty_ad", "bank_ad", "nifty_ad_ok", "bank_ad_ok"]].to_string())
    print(r5.head(10)[["start", "end", "dropped_day", "E", "D", "nifty_sd", "bank_sd", "nifty_skew", "bank_skew",
                       "nifty_kurt", "bank_kurt", "corr"]].to_string())
    print(r5["E"].value_counts().sort_index().to_string())
    print(pd.concat(res6)[["definition", "start", "end", "L", "E", "D", "nifty_n", "n_extreme_ok"]].to_string())


if __name__ == "__main__":
    main()
