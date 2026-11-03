"""P117 phase-1 checks (run before the plan was committed). Nothing here computes a placebo statistic, a half-life
of a new window, or anything about Fin Nifty's volatility. It reports:

1. the one-index grids (days with at least 300 bars from 09:15 to 15:14), the actual windows on them and the gap;
2. the number of placebo pairs in each registered set, how many of them share no trading day (greedy, every
   phase) and how many have wholly separate spans (dates only);
3. that P114's committed rolling-window fits are reproduced by this study's window code (public numbers);
4. that P114's committed bootstrap draws are reproduced by this study's bootstrap with explicit seeds (first draws
   of chunk 0 for each treated key; public numbers);
5. Fin Nifty's minute bars on the actual windows (dates only), its data hole, and its expiry calendar from the
   F&O bhavcopies (expiry dates only).

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_plan_checks.py
"""

from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p117_persistence_placebo", ROOT / "experiments" / "p117_persistence_placebo.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)
p114, rep, rp = ex.p114, ex.rep, ex.rp
RAW = ROOT / "data" / "raw"


def main():
    print("P117 phase-1 checks. REAL DATA dates and committed P114 numbers only; no new statistic is computed.")
    print("\n1-2. Grids, the actual pair, and the placebo pairs (dates only)")
    grids = {}
    for k in ("nifty", "banknifty", "finnifty"):
        bars = rp.load_minute_bars(RAW / f"{k}_1min.csv", "2017-01-01", ex.DATA_END[k])
        grids[k] = p114.index_grid(bars, k)
        d = grids[k]["days"]
        print(f"  {k}: {len(d)} grid days, {d[0].date()} to {d[-1].date()}")
    for k in ex.TREATED:
        d = grids[k]["days"]
        act = ex.actual_rows(d)
        P = ex.enumerate_pairs(d, act["gap"], actual_start=act["start_row"])
        span = ex.L_BEFORE + act["gap"] + ex.L_AFTER
        print(f"  {k}: actual before rows {act['start_row']}..{act['start_row'] + 63}, after rows {act['after_row']}.."
              f"{act['after_row'] + 58}, gap {act['gap']} grid days, span {span}; the actual pair straddles: "
              + ", ".join(c for c in ex.CHANGES if P.loc[P['is_actual'], f'straddles_{c}'].iloc[0]))
        for s in ex.SETS:
            S = P[P[f"in_{s}"]]
            st = S["start_row"].to_numpy()
            ns = [len(ex.disjoint_selection(st, ex.L_BEFORE, act["gap"], ex.L_AFTER, ph)) for ph in range(span)]
            sp = ex.separate_spans(st, span)
            print(f"    {s}: {len(S)} pairs, before windows from {S['before_first'].min().date()} to "
                  f"{S['before_first'].max().date()}, last after window ends {S['after_last'].max().date()}; "
                  f"day-disjoint pairs per phase min {min(ns)}, mean {np.mean(ns):.2f}, max {max(ns)}; wholly separate "
                  f"spans {sp['min']} to {sp['max']}; meets the minimum: {len(S) >= ex.MIN_PAIRS and np.mean(ns) >= ex.MIN_DISJOINT}")
        S = P[P["in_primary"]]
        other = [c for c in ex.CHANGES if c not in ("a", "b")]
        print("    primary-set pairs that straddle another change: "
              + "; ".join(f"{c} {int(S[f'straddles_{c}'].sum())}" for c in other)
              + f"; none of them {int((~S[[f'straddles_{c}' for c in other]].any(axis=1)).sum())}; pairs after event b "
              f"{int((S['before_first'] >= pd.Timestamp(ex.CHANGES['b'])).sum())}")
        need = P[P["is_actual"] | P[[f"in_{s}" for s in ex.SETS]].any(axis=1)]
        print(f"    windows to fit: {need['start_row'].nunique()} before and {need['after_row'].nunique()} after")

    print("\n3. P114's committed rolling windows through this study's window code (REAL DATA, committed numbers)")
    roll = pd.read_csv(ROOT / "results" / "p114_persistence_fixed_model" / "rolling.csv", float_precision="round_trip")
    for k, wns in (("nifty", (0, 22, 43)), ("banknifty", (0, 37))):
        G = grids[k]
        rep._EGARCH_CACHE.clear()
        p114._HAR_CACHE.clear()
        for wn in wns:
            r0 = roll[(roll["index"] == k) & (roll["window_no"] == wn)]
            s0 = int(np.searchsorted(G["days"], pd.Timestamp(r0["start"].iloc[0])))
            rows = ex.window_rows(G, k, np.arange(s0, s0 + 64), "before", "check")
            for r in rows:
                c = r0[r0["model"] == r["model"]].iloc[0]
                same = all(r[q] == c[q] for q in ("alpha", "beta", "gam_x", "delta", "nu", "hl_irf"))
                print(f"  {k} window {wn} ({r0['start'].iloc[0]}) {r['model']}: half-life {r['hl_irf']:.4f} "
                      f"(committed {c['hl_irf']:.4f}); six values {'identical' if same else 'DIFFERENT'}")

    print("\n4. P114's committed bootstrap draws through this study's bootstrap (REAL DATA, committed numbers)")
    rep._EGARCH_CACHE.clear()
    p114._HAR_CACHE.clear()
    G2 = rep.build_grids(p114.load_bars(RAW, p114.DATA_END_MAIN), p114.TRIM)
    pb = pd.read_csv(ROOT / "results" / "p114_persistence_fixed_model" / "bootstrap_draws.csv", float_precision="round_trip")
    for (w, k) in ex.DID_KEYS:
        x0, x1 = ex.ACTUAL["before" if w == "before_inferred" else "after"]
        d = G2["days"][(G2["days"] >= pd.Timestamp(x0)) & (G2["days"] <= pd.Timestamp(x1))]
        fw = p114.fit_window(G2, k, rep.period_rows(G2, d), models=ex.MODELS)
        bd = ex.boot_draws(ex.store_of(fw), p114.SEED_INDEX[(w, k)], ex.N_BOOT, chunks=[0], per_limit=2)
        n_same = n_all = 0
        for _, r in bd.iterrows():
            c = pb[(pb["window"] == w) & (pb["index"] == k) & (pb["model"] == r["model"]) & (pb["draw"] == r["draw"])].iloc[0]
            for q in ("alpha", "beta", "gam_x", "delta", "hl_irf"):
                n_all += 1
                n_same += int(float(f"{r[q]:.12g}") == c[q])
        print(f"  {w} {k}: {len(d)} days; first 2 draws, ref and F: {n_same} of {n_all} values identical to the 12 "
              "significant digits of the committed file")

    print("\n5. Fin Nifty: minute bars on the actual windows, data holes, expiry calendar (dates only)")
    dn, dc = grids["nifty"]["days"], grids["finnifty"]["days"]
    for w in ("before", "after"):
        x0, x1 = map(pd.Timestamp, ex.ACTUAL[w])
        a = dn[(dn >= x0) & (dn <= x1)]
        b = dc[(dc >= x0) & (dc <= x1)]
        print(f"  {w} window: Nifty {len(a)} days, Fin Nifty {len(b)} days, identical dates: {a.equals(b)}")
    step = np.r_[0, np.diff(dc.values).astype("timedelta64[D]").astype(int)]
    for i in np.flatnonzero(step > ex.HOLE_DAYS):
        print(f"  data hole: {dc[i - 1].date()} to {dc[i].date()} ({step[i]} calendar days)")
    print(f"  Fin Nifty grid days before the actual before window: {int((dc < pd.Timestamp(ex.ACTUAL['before'][0])).sum())}")
    rows = []
    for f in sorted(glob.glob(str(RAW / "finnifty" / "bhavcopy" / "*.csv.gz"))):
        df = pd.read_csv(f)
        day = pd.Timestamp(Path(f).name[3:11])
        if "SYMBOL" in df.columns:
            df = df[df["SYMBOL"].str.strip() == "FINNIFTY"]
            pairs = set(zip(df["INSTRUMENT"].str.strip(), pd.to_datetime(df["EXPIRY_DT"].str.strip(), format="%d-%b-%Y")))
        else:
            df = df[df["TckrSymb"].str.strip() == "FINNIFTY"]
            pairs = {("FUTIDX" if i == "IDF" else "OPTIDX", pd.Timestamp(e)) for i, e in zip(df["FinInstrmTp"], df["XpryDt"])}
        rows += [(i, e, day) for i, e in pairs]
    x = pd.DataFrame(rows, columns=["ins", "expiry", "date"])
    last = x.groupby(["ins", "expiry"])["date"].max().reset_index()
    last = last[last["expiry"] == last["date"]]
    fut = set(last.loc[last["ins"] == "FUTIDX", "expiry"])
    c = pd.DataFrame({"expiry": sorted(set(last["expiry"]))})
    c["monthly"] = c["expiry"].isin(fut)
    c["weekday"] = c["expiry"].dt.day_name()
    print(f"  bhavcopies {x['date'].min().date()} to {x['date'].max().date()}: {len(c)} Fin Nifty expiries")
    c["period"] = np.where(c["expiry"] <= pd.Timestamp("2024-11-19"), "to 19 Nov 2024",
                           np.where(c["expiry"] <= pd.Timestamp("2024-12-31"), "20 Nov to 31 Dec 2024",
                                    np.where(c["expiry"] <= pd.Timestamp("2025-08-31"), "Jan to Aug 2025", "from Sep 2025")))
    for (p, m), g in c.groupby(["period", "monthly"], sort=False):
        print(f"    {p}, {'monthly' if m else 'weekly'}: {len(g)} expiries; weekdays {g['weekday'].value_counts().to_dict()}")


if __name__ == "__main__":
    sys.exit(main())
