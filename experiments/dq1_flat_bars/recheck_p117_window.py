"""Refit P117 Nifty placebo windows that contain 24 Feb 2021, with and without that day (REAL DATA, post hoc).

Same code path as P117's windows stage (p114.index_grid on Nifty bars 2017-01-01 to 2026-11-02, then
p114.fit_window with models ref and F). 'without' drops 24 Feb 2021 from the grid, so the window keeps its other
63 (before) or 58 (after) days; a second variant extends the window by one day so it keeps 64 or 59 days.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT = Path(__file__).resolve().parents[2] / "results" / "dq1_flat_bars"
spec = importlib.util.spec_from_file_location("p114", REPO / "experiments" / "p114_persistence_fixed_model.py")
p114 = importlib.util.module_from_spec(spec); sys.modules["p114"] = p114; spec.loader.exec_module(p114)
rp = p114.rp
DAY = pd.Timestamp("2021-02-24")
W = pd.read_csv(REPO / "results" / "p117_persistence_placebo" / "windows_nifty.csv", parse_dates=["first_day", "last_day"])
# four windows: before windows starting 2021-01-11 and 2020-11-24 (day at the end), after windows likewise
picks = [("before", "2021-01-11"), ("before", "2020-12-17"), ("after", "2021-01-11"), ("after", "2021-02-03")]
bars = rp.load_minute_bars(RAW / "nifty_1min.csv", "2017-01-01", "2026-11-02")
p114._HAR_CACHE.clear(); p114.rep._EGARCH_CACHE.clear()
G = p114.index_grid(bars, "nifty")
G2 = p114.index_grid(bars[bars["date"] != DAY].reset_index(drop=True), "nifty")
rows = []
for kind, first in picks:
    L = 64 if kind == "before" else 59
    s = int(np.searchsorted(G["days"], pd.Timestamp(first)))
    days = G["days"][s:s + L]
    com = W[(W["kind"] == kind) & (W["first_day"] == pd.Timestamp(first))]
    for variant, GG, rr in (("registered (with 24 Feb 2021)", G, np.arange(s, s + L)),
                            ("without 24 Feb 2021, same dates", G2, np.where(G2["days"].isin(days))[0]),
                            ("without 24 Feb 2021, extended by one day", G2,
                             np.arange(int(np.searchsorted(G2["days"], pd.Timestamp(first))),
                                       int(np.searchsorted(G2["days"], pd.Timestamp(first))) + L))):
        fw = p114.fit_window(GG, "nifty", rr, models=("ref", "F"))
        for m in ("ref", "F"):
            r = fw[m]["row"]
            c = com[com["model"] == m]
            rows.append({"data": "REAL DATA (post hoc recheck, descriptive)", "kind": kind, "variant": variant,
                         "first_day": str(GG["days"][rr[0]].date()), "last_day": str(GG["days"][rr[-1]].date()),
                         "n_days": len(rr), "model": m, "alpha": r["alpha"], "beta": r["beta"], "hl_irf": r["hl_irf"],
                         "at_bound_gt_50000": bool(r["hl_irf"] > 50000),
                         "committed_hl_irf": float(c["hl_irf"].iloc[0]) if len(c) else np.nan})
        print(kind, first, variant, "done", flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT / "recheck_p117_windows.csv", index=False)
pd.set_option("display.width", 250)
print(R.drop(columns="data").to_string(index=False))
