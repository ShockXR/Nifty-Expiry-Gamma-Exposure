"""P117 placebo p without the pairs that contain a flagged day, at the 20% and the 5% screens (REAL DATA, descriptive).

Reads the committed results/p117_persistence_placebo/placebo_pairs.csv and placebo_summary.csv, and this folder's
flat_all_days.csv.gz. P117's grid is the 360-minute grid (09:15 to 15:14) with a 300-bar rule, so a day is flagged
here by its flat share on the first 360 slots (flat_minutes_first360 / returns), on days with at least 300 bars.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[2] / "results" / "dq1_flat_bars"
P = REPO / "results" / "p117_persistence_placebo"
pairs = pd.read_csv(P / "placebo_pairs.csv", parse_dates=["before_first", "before_last", "after_first", "after_last"])
summ = pd.read_csv(P / "placebo_summary.csv")
A = pd.read_csv(OUT / "flat_all_days.csv.gz", parse_dates=["date"])
A["share360"] = A["flat_minutes_first360"] / A["returns"].clip(upper=360)
rows = []
for thr in (0.20, 0.05):
    for k in ("nifty", "banknifty"):
        bad = A[(A["index"] == k) & (A["bars"] >= 300) & (A["share360"] > thr) & (A["date"].dt.weekday < 5)]["date"]
        Pk = pairs[pairs["index"] == k]
        has = np.zeros(len(Pk), bool)
        for d in bad:
            has |= (((Pk["before_first"] <= d) & (Pk["before_last"] >= d)) | ((Pk["after_first"] <= d) & (Pk["after_last"] >= d))).to_numpy()
        for m in ("ref", "F"):
            for s in ("primary", "clean", "literal"):
                r = summ[(summ["index"] == k) & (summ["model"] == m) & (summ["set"] == s)].iloc[0]
                S = Pk[Pk[f"in_{s}"].astype(bool) & Pk[f"delta_{m}"].notna()]
                hs = has[(Pk[f"in_{s}"].astype(bool) & Pk[f"delta_{m}"].notna()).to_numpy()]
                x = S.loc[~hs, f"delta_{m}"].to_numpy()
                kk = int((x <= r["delta_actual"]).sum())
                rows.append({"data": "REAL DATA (P117 committed placebo pairs; post hoc, descriptive)",
                             "screen": f">{int(thr * 100)}% flat (360-minute grid)", "index": k, "model": m, "set": s,
                             "flagged_days_on_grid": ";".join(d.strftime("%Y-%m-%d") for d in bad),
                             "p_registered": round(float(r["p_placebo"]), 4), "n_pairs": int(r["n_pairs"]),
                             "pairs_with_flagged_day": int(hs.sum()), "n_without": len(x), "k_without": kk,
                             "p_without": round((1 + kk) / (1 + len(x)), 4)})
R = pd.DataFrame(rows)
R.to_csv(OUT / "recheck_p117.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(R.drop(columns="data").to_string(index=False))
