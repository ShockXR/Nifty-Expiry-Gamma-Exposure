"""Summarise results/power_study.csv into rejection rates and median gains (E4)."""

import numpy as np
import pandas as pd

d = pd.read_csv("results/power_study.csv")
rows = []
for key, g in d.groupby(["kappa", "noise", "target", "subset"]):
    rec = dict(zip(["kappa", "noise", "target", "subset"], key))
    rec["reps"] = len(g)
    rec["median_test_days"] = int(g["n_days_test"].median())
    rec["hybrid_vs_mcsgarch_pct"] = float(np.median(100 * (1 - g["qlike_base"] / g["qlike_mcsgarch"])))
    rec["har_vs_mcsgarch_pct"] = float(np.median(100 * (1 - g["qlike_har"] / g["qlike_mcsgarch"])))
    for s in ("us", "participant", "oracle"):
        rec[f"median_gain_pct_{s}"] = float(g[f"gain_pct_{s}"].median())
        rec[f"reject_rate_{s}"] = float((g[f"dm_p_{s}"] < 0.05).mean())
    rows.append(rec)
out = pd.DataFrame(rows)
out.to_csv("results/power_study_summary.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(out.round(2).to_string())
