"""P117 POST HOC (decided after the registered results were seen; descriptive only, nothing is added to tests.csv).

The registered run showed that every reference fit of a Nifty window containing 24 February 2021 has its
persistence at the upper bound, and so do all of the fixed model's bound fits. On that day NSE halted trading
(a technical outage) and the Nifty minute file fills the halt with 288 flat bars, so the day passes the grid's
300-bar rule. The event study (`experiments/expiry_event_study.py`) already drops days with more than 20% flat bars;
the replication's two-index grid drops this day because Bank Nifty has only 54 bars on it; P117's one-index grid
(P114's rolling-window grid) keeps it.

This script, reading only committed P117 outputs and the raw minute bars:

1. lists the grid days with more than 20% flat bars (the event study's threshold) for Nifty and Bank Nifty; here a
   flat bar is a minute whose close equals the previous minute's close (the event study counts minutes whose high
   equals their low; on the days this script flags, both counts are above 20%);
2. for every index, model and set, splits the placebo pairs whose change is at or below the actual change into
   those that contain such a day, those that straddle another calendar change (and which), and the rest, with the
   range of their start dates;
3. recomputes the registered placebo share and p with the pairs that contain such a day left out (descriptive);
4. counts, for every index, model and kind of window, the fits whose half-life is at the persistence bound (above
   50,000 minutes), how many of them contain such a day, and the fits that did not converge or were not computed;
5. describes the primary set's pairs in three groups by the calendar changes they straddle: the Nifty weekly launch
   (before windows in 2018 and early 2019), none (the clean set, 2019 to 2023), and the pairs whose before window
   starts on or after CPP's event b (2024 and 2025; they straddle later changes): medians of the half-lives and of
   the change, and how many pairs have a change at or below the actual one.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_posthoc_flat_days.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p117_persistence_placebo", ROOT / "experiments" / "p117_persistence_placebo.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "p117_persistence_placebo"
FLAT_SHARE = 0.20            # the event study's rule: a day with more than 20% flat bars is dropped
BOUND_HL = 50_000.0          # a half-life above this is "at the persistence bound" (0.99999 gives about 69,000)


def flat_days(k: str) -> pd.DataFrame:
    bars = ex.rp.load_minute_bars(ROOT / "data" / "raw" / f"{k}_1min.csv", "2017-01-01", ex.DATA_END[k])
    G = ex.p114.index_grid(bars, k)
    C = G[k]["close"]
    r = np.diff(np.log(C), axis=1)
    n = np.isfinite(C).sum(axis=1)
    flat = (r == 0).sum(axis=1)
    d = pd.DataFrame({"index": k, "date": G["days"], "bars": n, "flat_bars": flat, "flat_share": flat / n})
    return d[d["flat_share"] > FLAT_SHARE]


def main():
    label = ex.LABEL_REAL
    pairs = pd.read_csv(OUT / "placebo_pairs.csv", parse_dates=["before_first", "before_last", "after_first", "after_last"])
    summ = pd.read_csv(OUT / "placebo_summary.csv")
    fd = pd.concat([flat_days(k) for k in ex.TREATED], ignore_index=True)
    fd.insert(0, "data", label)
    fd.to_csv(OUT / "posthoc_flat_days.csv", index=False)
    print("POST HOC. Grid days with more than 20% flat bars:")
    print(fd.to_string(index=False))
    # every change except CPP's event b: the primary and clean sets never straddle a, the literal set may
    others = [c for c in ex.CHANGES if c != "b"]
    rows, prow = [], []
    for k in ex.TREATED:
        P = pairs[pairs["index"] == k].copy()
        bad = pd.DatetimeIndex(fd.loc[fd["index"] == k, "date"])
        has_flat = np.zeros(len(P), bool)
        for d in bad:
            has_flat |= (((P["before_first"] <= d) & (P["before_last"] >= d)) | ((P["after_first"] <= d) & (P["after_last"] >= d))).to_numpy()
        P["contains_flat_day"] = has_flat
        for m in ex.MODELS:
            for s in ("primary", "clean", "literal"):
                r = summ[(summ["index"] == k) & (summ["model"] == m) & (summ["set"] == s)]
                if r.empty:
                    continue
                r = r.iloc[0]
                d_act = r["delta_actual"]
                S = P[P[f"in_{s}"] & P[f"delta_{m}"].notna()]
                K = S[S[f"delta_{m}"] <= d_act]
                cat = np.where(K["contains_flat_day"], "contains a day with more than 20% flat bars",
                               np.where(K[[f"straddles_{c}" for c in others]].any(axis=1), "straddles another calendar change",
                                        "straddles no calendar change"))
                for c in ("contains a day with more than 20% flat bars", "straddles another calendar change",
                          "straddles no calendar change"):
                    g = K[cat == c]
                    rows.append({"data": label, "index": k, "model": m, "set": s, "delta_actual": d_act,
                                 "category": c, "pairs_at_or_below_actual": len(g),
                                 "before_at_bound": int((g[f"hl_before_{m}"] > BOUND_HL).sum()),
                                 "changes_straddled": ", ".join(f"{x} {int(g[f'straddles_{x}'].sum())}" for x in others
                                                                if g[f"straddles_{x}"].any()),
                                 "first_before_start": str(g["before_first"].min().date()) if len(g) else "",
                                 "last_before_start": str(g["before_first"].max().date()) if len(g) else ""})
                T = S[~S["contains_flat_day"]]
                pp = ex.placebo_p(T[f"delta_{m}"], d_act)
                prow.append({"data": label, "index": k, "model": m, "set": s, "delta_actual": d_act,
                             "n_pairs_registered": int(r["n_pairs"]), "p_registered": r["p_placebo"],
                             "n_pairs_with_flat_day": int(S["contains_flat_day"].sum()),
                             "n_pairs_without": pp["n"], "k_at_or_below_without": pp["k"],
                             "share_at_or_below_without": pp["share"], "p_without_flat_day_pairs": pp["p"],
                             "windows_at_bound_in_set_pairs": int((S[f"hl_before_{m}"] > BOUND_HL).sum() + (S[f"hl_after_{m}"] > BOUND_HL).sum())})
    br = pd.DataFrame(rows)
    pw = pd.DataFrame(prow)
    br.to_csv(OUT / "posthoc_extreme_pairs.csv", index=False)
    pw.to_csv(OUT / "posthoc_without_flat_days.csv", index=False)
    brow = []
    for k in ex.TREATED:
        f = OUT / f"windows_{k}.csv"
        if not f.exists():
            continue
        W = pd.read_csv(f, parse_dates=["first_day", "last_day"])
        bad = pd.DatetimeIndex(fd.loc[fd["index"] == k, "date"])
        flat = np.zeros(len(W), bool)
        for d in bad:
            flat |= ((W["first_day"] <= d) & (W["last_day"] >= d)).to_numpy()
        W["contains_flat_day"] = flat
        for m in ex.MODELS:
            for kind in ("before", "after"):
                g = W[(W["model"] == m) & (W["kind"] == kind)]
                bnd = g[g["hl_irf"] > BOUND_HL]
                other = bnd[~bnd["contains_flat_day"]]
                brow.append({"data": label, "index": k, "model": m, "kind": kind, "windows": len(g),
                             "not_computed": int(g["hl_irf"].isna().sum()),
                             "not_converged": int((g["converged"].astype(str) != "True").sum()),
                             "contain_flat_day": int(g["contains_flat_day"].sum()),
                             "at_bound": len(bnd), "at_bound_containing_flat_day": int(bnd["contains_flat_day"].sum()),
                             "at_bound_other_first_start": str(other["first_day"].min().date()) if len(other) else "",
                             "at_bound_other_last_start": str(other["first_day"].max().date()) if len(other) else ""})
    bw = pd.DataFrame(brow)
    bw.to_csv(OUT / "posthoc_bound_windows.csv", index=False)
    grow = []
    b_day = pd.Timestamp(ex.CHANGES["b"])
    for k in ex.TREATED:
        P = pairs[(pairs["index"] == k) & pairs["in_primary"]]
        bad = pd.DatetimeIndex(fd.loc[fd["index"] == k, "date"])
        has_flat = np.zeros(len(P), bool)
        for d in bad:
            has_flat |= (((P["before_first"] <= d) & (P["before_last"] >= d)) | ((P["after_first"] <= d) & (P["after_last"] >= d))).to_numpy()
        groups = {"straddles the Nifty weekly launch": P["straddles_nifty_weekly"].to_numpy(),
                  "straddles no calendar change (the clean set)": P["in_clean"].to_numpy(),
                  "before window starts on or after event b": (P["before_first"] >= b_day).to_numpy()}
        for m in ex.MODELS:
            r = summ[(summ["index"] == k) & (summ["model"] == m) & (summ["set"] == "primary")]
            if r.empty:
                continue
            d_act = r.iloc[0]["delta_actual"]
            for gname, g in groups.items():
                G = P[g]
                x = G[f"delta_{m}"].dropna()
                grow.append({"data": label, "index": k, "model": m, "set": "primary", "group": gname, "pairs": len(G),
                             "pairs_with_flat_day": int(has_flat[g].sum()),
                             "first_before_start": str(G["before_first"].min().date()) if len(G) else "",
                             "last_before_start": str(G["before_first"].max().date()) if len(G) else "",
                             "last_after_end": str(G["after_last"].max().date()) if len(G) else "",
                             "changes_straddled": ", ".join(f"{c} {int(G[f'straddles_{c}'].sum())}" for c in ex.CHANGES
                                                            if G[f"straddles_{c}"].any()),
                             "median_hl_before": float(G[f"hl_before_{m}"].median()),
                             "median_hl_after": float(G[f"hl_after_{m}"].median()),
                             "median_delta": float(x.median()), "share_with_fall": float((x < 0).mean()),
                             "min_delta": float(x.min()), "delta_actual": d_act,
                             "pairs_at_or_below_actual": int((x <= d_act).sum()),
                             "share_at_or_below_actual": float((x <= d_act).mean())})
    gw = pd.DataFrame(grow)
    gw.to_csv(OUT / "posthoc_pairs_by_period.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nPOST HOC. Placebo pairs with a change at or below the actual change, by kind:")
    print(br.to_string(index=False))
    print("\nPOST HOC (descriptive, not a registered test). Placebo share and p without the pairs containing such a day:")
    print(pw.to_string(index=False))
    print("\nPOST HOC. Window fits at the persistence bound (half-life above 50,000 minutes):")
    print(bw.to_string(index=False))
    print("\nPOST HOC. Primary-set pairs by the calendar changes they straddle:")
    print(gw.to_string(index=False))


if __name__ == "__main__":
    main()
