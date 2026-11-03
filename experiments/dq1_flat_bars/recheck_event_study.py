"""Event study: R0 own-expiry premium with the >5% flat days removed from the week baselines (REAL DATA, post hoc).

The event study already drops days with more than 20% flat bars. Days between 5% and 20% that it keeps, inside its
long sample (from 4 Mar 2019): Nifty 13 Mar 2020, 23 Mar 2020, 7 Mar 2022; Bank Nifty 13 Mar 2020, 23 Mar 2020
(Bank Nifty 7 Mar 2022 is already dropped, under 370 bars). None is an expiry day, so they enter only the baseline
(mean of the other days of their week). This rebuilds the weekly value of those weeks without the day, with the
event study's own week_relative, Series and level_summary (same seeds, 4,999 draws), from the committed
results/event_study/daily_measures.csv.gz, weekly_relative.csv.gz and week_flags.csv.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[2] / "results" / "dq1_flat_bars"
spec = importlib.util.spec_from_file_location("es", REPO / "experiments" / "expiry_event_study.py")
es = importlib.util.module_from_spec(spec); sys.modules["es"] = es; spec.loader.exec_module(es)
E = REPO / "results" / "event_study"
rel = pd.read_csv(E / "weekly_relative.csv.gz", parse_dates=["week", "target_date"])
flags = pd.read_csv(E / "week_flags.csv", parse_dates=["week"]).set_index("week")
dm = pd.read_csv(E / "daily_measures.csv.gz", parse_dates=["date"])
prem = pd.read_csv(E / "premium_by_regime.csv", parse_dates=["from_week", "to_week"])
NAME = {"N": "Nifty 50", "BN": "Bank Nifty"}
FLAG = {"N": ["2020-03-13", "2020-03-23", "2022-03-07"], "BN": ["2020-03-13", "2020-03-23", "2022-03-07"]}
MEAS = ["rv5", "rv1", "rs", "rv_first_hour", "rv_last_hour", "rv_last30"]
REG = ("R0 both Thu (2019-03 to a)", es.LONG_START, pd.Timestamp("2023-09-04"))
rows = []
for k in ("N", "BN"):
    d = dm[dm["index_name"] == NAME[k]].set_index("date")
    present = [pd.Timestamp(x) for x in FLAG[k] if pd.Timestamp(x) in d.index]
    for m in MEAS:
        r_old = rel[(rel["index"] == k) & (rel["target"] == "own") & (rel["measure"] == m)].set_index("week")
        r_new = r_old.copy()
        for day in present:
            w = es.week_start(pd.DatetimeIndex([day]))[0]
            if w not in r_old.index:
                continue
            t = r_old.loc[w, "target_date"]
            if t == day:
                r_new = r_new.drop(index=w)
                continue
            v = d[m].drop(index=day)
            wr = es.week_relative(v, pd.Series([t], index=[w]), log=True)
            if w in wr.index:
                r_new.loc[w, ["rel", "n_other"]] = [wr.loc[w, "rel"], wr.loc[w, "n_other"]]
            else:
                r_new = r_new.drop(index=w)
        for sample in ("clean_weekly", "all"):
            for lab, rr in (("committed", r_old), ("without >5% flat days", r_new)):
                tab = rr.reset_index().assign(index=k, target="own", measure=m)
                y = es.Series(tab, flags, (k, "own"), None, sample).values(m)
                x = y[(y.index >= REG[1]) & (y.index < REG[2])].dropna()
                s = es.level_summary(x.to_numpy(), n_boot=4999, seed=es.seed_of(k, "own", m, REG[0], sample))
                c = prem[(prem["index"] == k) & (prem["target"] == "own") & (prem["measure"] == m)
                         & (prem["regime"] == REG[0]) & (prem["sample"] == sample)]
                rows.append({"data": "REAL DATA (event study committed files; post hoc recheck)", "index": k,
                             "measure": m, "sample": sample, "regime": REG[0], "version": lab,
                             "days_removed": ";".join(x_.strftime("%Y-%m-%d") for x_ in present) if lab != "committed" else "",
                             "n_weeks": int(s["n"]) if "n" in s else len(x), "mean": s["mean"], "boot_lo": s["boot_lo"],
                             "boot_hi": s["boot_hi"], "hac_p": s["hac_p"],
                             "committed_mean_in_premium_by_regime": float(c["mean"].iloc[0]) if len(c) else np.nan,
                             "committed_hac_p": float(c["hac_p"].iloc[0]) if len(c) else np.nan})
R = pd.DataFrame(rows)
R.to_csv(OUT / "recheck_event_study.csv", index=False)
pd.set_option("display.width", 250)
print(R.drop(columns=["data", "regime"]).to_string(index=False))
