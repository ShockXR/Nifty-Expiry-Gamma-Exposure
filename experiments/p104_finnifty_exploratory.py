"""P104 phase 2: exploratory looks that the plan did not ask for (labelled exploratory in
docs/THIRD_INDEX_FORECAST.md). REAL DATA. Written after the registered results were seen; no verdict depends on it.

Two questions about the benchmark that came up when reading the registered outputs:

1. mcsGARCH's daily level falls back to a simple EWMA variance when the EGARCH fit fails (unchanged code). On
   Fin Nifty this happened in the first 3 of 29 refits. Does the hybrid's gain differ between the test days
   forecast with the fallback and the others? (P91 made the same split for Bank Nifty.)
2. On 5 June 2024, the day after the general-election result (4 June 2024, Fin Nifty fell about 7%), mcsGARCH's
   first forecasts of the day were about 135 times the realized variance. What does the comparison look like
   without that one day, under QLIKE and under MSE, and how large was the same forecast on Nifty and Bank Nifty?

Reads results/pipeline_nse_finnifty/forecasts_<target>.csv and mcsgarch_params.csv, and for the 5 June
comparison the committed Nifty v2 and Bank Nifty P91 forecasts (not rerun). Fits nothing; scores with the
pipeline's own functions (daily_losses, diebold_mariano with alternative="greater"), as
experiments/p104_finnifty_summary.py does.

    python experiments/p104_finnifty_exploratory.py

Writes results/pipeline_nse_finnifty/p104/exploratory.csv (one-sided p-values, all exploratory) and
results/pipeline_nse_finnifty/p104/june5_2024.csv (descriptive, no p-values).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.evaluation import daily_losses, diebold_mariano, mse, qlike

ROOT = Path(__file__).resolve().parents[1]
FN = ROOT / "results" / "pipeline_nse_finnifty"
OTHERS = {"Nifty (v2 forecasts)": ROOT / "results" / "pipeline_nse_v2",
          "Bank Nifty (P91 forecasts)": ROOT / "results" / "pipeline_nse_banknifty",
          "Fin Nifty": FN}
OUT = FN / "p104"
TARGETS = ["next15", "next30", "last_hour"]
DAY = "2024-06-05"
LOSS_FN = {"qlike": qlike, "mse": mse}


def compare(pred: pd.DataFrame, loss: str, model="hybrid_base", ref="mcsgarch") -> dict:
    """As compare() in p104_finnifty_summary.py, for either loss."""
    fn = LOSS_FN[loss]
    l_ref, l_mod = float(np.mean(fn(pred["y"], pred[ref]))), float(np.mean(fn(pred["y"], pred[model])))
    L = daily_losses(pred, [ref, model], loss=loss)
    dm = diebold_mariano(L[ref], L[model], alternative="greater")
    return {"loss": loss, "days": int(pred["date"].nunique()), "anchors": len(pred), "loss_mcsgarch": l_ref,
            "loss_hybrid_base": l_mod, "change_pct": 100 * (l_mod / l_ref - 1), "dm_stat": dm["stat"],
            "p_one_sided": dm["p"]}


def block_method(target: str) -> pd.Series:
    """daily_method of the mcsGARCH refit that forecast each test day index t (refits every 21 days)."""
    p = pd.read_csv(FN / "mcsgarch_params.csv")
    p = p[(p["target"] == target) & (p["refit_every"] == 21)].sort_values("start")
    return p.set_index("start")["daily_method"]


def main():
    rows, june = [], []
    for t in TARGETS:
        f = pd.read_csv(FN / f"forecasts_{t}.csv", usecols=["t", "date", "minute", "y", "mcsgarch", "hybrid_base"])
        meth = block_method(t)
        starts = meth.index.to_numpy()
        f["daily_method"] = meth.to_numpy()[np.searchsorted(starts, f["t"].to_numpy(), side="right") - 1]
        for m, g in f.groupby("daily_method"):
            rows.append({"analysis": "mcsGARCH daily level by refit type", "target": t,
                         "group": f"days forecast with daily_method = {m}",
                         "first_day": g["date"].min(), "last_day": g["date"].max(), **compare(g, "qlike")})
        g = f[f["date"] != DAY]
        for loss in ("qlike", "mse"):
            rows.append({"analysis": "without 5 June 2024", "target": t, "group": "all test days except 5 June 2024",
                         "first_day": g["date"].min(), "last_day": g["date"].max(), **compare(g, loss)})
        for name, folder in OTHERS.items():
            o = pd.read_csv(folder / f"forecasts_{t}.csv", usecols=["date", "minute", "y", "mcsgarch", "hybrid_base"])
            d = o[o["date"] == DAY].sort_values("minute").iloc[0]
            se = (o["y"] - o["mcsgarch"]) ** 2
            june.append({"index": name, "target": t, "date": DAY, "minute": int(d["minute"]), "realized": d["y"],
                         "mcsgarch": d["mcsgarch"], "hybrid_base": d["hybrid_base"],
                         "mcsgarch_over_realized": d["mcsgarch"] / d["y"],
                         "hybrid_over_realized": d["hybrid_base"] / d["y"],
                         "share_of_mcsgarch_squared_error": float(se.loc[d.name] / se.sum()),
                         "largest_mcsgarch_over_realized_any_day": float((o["mcsgarch"] / o["y"]).max())})
    OUT.mkdir(parents=True, exist_ok=True)
    ex, jn = pd.DataFrame(rows), pd.DataFrame(june)
    ex.to_csv(OUT / "exploratory.csv", index=False, lineterminator="\n")
    jn.to_csv(OUT / "june5_2024.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 220)
    print("REAL DATA. P104 exploratory looks (not in the plan).")
    print(ex.drop(columns=["first_day", "last_day"]).to_string(index=False))
    print(jn.to_string(index=False))


if __name__ == "__main__":
    main()
