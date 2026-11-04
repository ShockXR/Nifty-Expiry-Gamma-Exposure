"""P114 (and P117's actual pair): refit ref and F on the before_inferred and after windows with flagged days removed
from the grid (REAL DATA, post hoc, descriptive).

The windows themselves (Nov 2022 onward) contain no flagged day. The flagged days enter only through the history that
the daily level is estimated on: F's X1 HAR level is a regression on daily realized variance (sum of squared
one-minute returns) from the first grid day (2017) to the window's last day; ref's daily EGARCH uses daily
close-to-close returns over the same history. Variants: registered (all days), minus >20% days, minus >5% days.
Grids: the two-index grid of P114's fit stage (data to 2024-06-30) and the one-index Nifty grid of P114's rolling
stage and P117 (data to 2026-11-02).

    PYTHONPATH=$REPO/src OMP_NUM_THREADS=1 python recheck_p114.py
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT = Path(__file__).resolve().parents[2] / "results" / "dq1_flat_bars"
spec = importlib.util.spec_from_file_location("p114", REPO / "experiments" / "p114_persistence_fixed_model.py")
p114 = importlib.util.module_from_spec(spec)
sys.modules["p114"] = p114
spec.loader.exec_module(p114)
rep, rp = p114.rep, p114.rp

FL = pd.read_csv(OUT / "flat_all_days.csv.gz", parse_dates=["date"])
FL["share360"] = FL["flat_minutes_first360"] / FL["returns"].clip(upper=360)


def flagged(k, thr):
    d = FL[(FL["index"] == k) & (FL["share360"] > thr)]["date"]
    return set(d)


def drop_days(bars, days):
    return bars[~bars["date"].isin(pd.DatetimeIndex(sorted(days)))].reset_index(drop=True)


def clear():
    p114._HAR_CACHE.clear()
    rep._EGARCH_CACHE.clear()


rows = []
t0 = time.time()
WIN = {w: p114.WINDOWS[w] for w in ("before_inferred", "after")}
# two-index grid (P114 fit stage)
bars2 = p114.load_bars(RAW, p114.DATA_END_MAIN)
for vname, thr in (("registered (all days)", None), ("minus >20% flat days", 0.20), ("minus >5% flat days", 0.05)):
    drop = set() if thr is None else flagged("nifty", thr) | flagged("banknifty", thr)
    b = {k: drop_days(v, drop) for k, v in bars2.items()}
    clear()
    G = rep.build_grids(b, p114.TRIM)
    for w, (a, z) in WIN.items():
        days = G["days"][(G["days"] >= pd.Timestamp(a)) & (G["days"] <= pd.Timestamp(z))]
        for k in p114.INDICES:
            r = rep.period_rows(G, days)
            fw = p114.fit_window(G, k, r, models=("ref", "F"))
            dropped_in_hist = sorted(d.strftime("%Y-%m-%d") for d in drop if d <= days.max())
            for m in ("ref", "F"):
                rr = fw[m]["row"]
                rows.append({"data": "REAL DATA (post hoc recheck, descriptive)", "grid": "two-index (P114 fit stage)",
                             "variant": vname, "days_dropped": ";".join(dropped_in_hist), "window": w, "index": k,
                             "model": m, "n_days": len(days), "alpha": rr["alpha"], "beta": rr["beta"],
                             "hl_irf": rr["hl_irf"], "loglik": rr["loglik"]})
            print(f"{vname} two-index {w} {k} done {time.time() - t0:.0f}s", flush=True)
# one-index Nifty grid (P114 rolling stage, P117)
bars1 = rp.load_minute_bars(RAW / "nifty_1min.csv", "2017-01-01", p114.DATA_END_ROLL["nifty"])
for vname, thr in (("registered (all days)", None), ("minus >20% flat days", 0.20), ("minus >5% flat days", 0.05)):
    drop = set() if thr is None else flagged("nifty", thr)
    clear()
    G = p114.index_grid(drop_days(bars1, drop), "nifty")
    for w, (a, z) in WIN.items():
        days = G["days"][(G["days"] >= pd.Timestamp(a)) & (G["days"] <= pd.Timestamp(z))]
        r = rep.period_rows(G, days)
        fw = p114.fit_window(G, "nifty", r, models=("ref", "F"))
        for m in ("ref", "F"):
            rr = fw[m]["row"]
            rows.append({"data": "REAL DATA (post hoc recheck, descriptive)", "grid": "one-index Nifty (P114 rolling, P117)",
                         "variant": vname, "days_dropped": ";".join(sorted(d.strftime("%Y-%m-%d") for d in drop if d <= days.max())),
                         "window": w, "index": "nifty", "model": m, "n_days": len(days), "alpha": rr["alpha"],
                         "beta": rr["beta"], "hl_irf": rr["hl_irf"], "loglik": rr["loglik"]})
        print(f"{vname} one-index {w} done {time.time() - t0:.0f}s", flush=True)
R = pd.DataFrame(rows)
# fall: log(hl_after) - log(hl_before), and the fall factor hl_before / hl_after
piv = R.pivot_table(index=["grid", "variant", "index", "model"], columns="window", values="hl_irf").reset_index()
piv["delta_log_hl"] = np.log(piv["after"]) - np.log(piv["before_inferred"])
piv["fall_factor"] = piv["before_inferred"] / piv["after"]
# change in equivalent persistence (P114): rho = 0.5 ** (1 / half-life), after minus before
piv["d_rho_irf"] = 0.5 ** (1 / piv["after"]) - 0.5 ** (1 / piv["before_inferred"])
R.to_csv(OUT / "recheck_p114_fits.csv", index=False)
piv.to_csv(OUT / "recheck_p114_fall.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(R.drop(columns="data").to_string(index=False))
print(piv.to_string(index=False))
