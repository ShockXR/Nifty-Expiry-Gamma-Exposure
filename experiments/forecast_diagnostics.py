"""Where do the forecasting gains in R1 come from? REAL DATA.

Answers review items M1, M2, m9 and the last-hour comparison in M6
(docs/INDEPENDENT_REVIEW.md), from the committed forecast files only:

* MSE next to QLIKE, and why they disagree (a few huge opening-anchor errors);
* QLIKE gain against mcsGARCH at the 09:15 anchor and at all later anchors;
* QLIKE gain by expiry regime;
* a trimmed QLIKE (top 1% of anchor losses removed) and the median loss;
* the share of anchors on which each model is closer than mcsGARCH (in QLIKE);
* the hybrid against HAR for the last hour (Diebold-Mariano, two-sided).

    python experiments/forecast_diagnostics.py --run results/pipeline_nse_v2

Writes results/forecast_diagnostics/*.csv and results/forecast_diagnostics_stdout.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.evaluation import daily_losses, diebold_mariano, mse, qlike

MODELS = ["har", "vix", "implied", "linear_base", "lgbm_base", "hybrid_base", "hybrid_gex_participant", "hybrid_gex_gross"]
REGIME_NAMES = {0: "multi_weekly", 1: "single_weekly_thu", 2: "single_weekly_tue"}


def pct(a, b):
    return 100 * (a / b - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse_v2")
    ap.add_argument("--out", default="results/forecast_diagnostics")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, worst, extra = [], [], []
    for target in ("next15", "next30", "last_hour"):
        f = pd.read_csv(Path(a.run) / f"forecasts_{target}.csv", parse_dates=["date"])
        f["regime_name"] = f["regime"].map(REGIME_NAMES).fillna(f["regime"].astype(str))
        q = {m: qlike(f["y"], f[m]) for m in ["mcsgarch"] + MODELS}
        e = {m: mse(f["y"], f[m]) for m in ["mcsgarch"] + MODELS}
        cut = np.quantile(q["mcsgarch"], 0.99)
        subsets = {"all anchors": np.ones(len(f), bool)}
        if target != "last_hour":
            subsets["09:15 anchor only"] = (f["minute"] == 0).to_numpy()
            subsets["excluding 09:15 anchor"] = (f["minute"] != 0).to_numpy()
        for r, name in REGIME_NAMES.items():
            subsets[f"regime {name}"] = (f["regime_name"] == name).to_numpy()
        subsets["expiry days"] = f["is_expiry"].to_numpy(bool)
        for sname, m in subsets.items():
            for mod in MODELS:
                qm, qb = q[mod][m], q["mcsgarch"][m]
                keep = qb <= cut  # trimmed: drop anchors in mcsGARCH's top 1% of losses (same anchors for every model)
                rows.append({"target": target, "subset": sname, "model": mod, "anchors": int(m.sum()),
                             "qlike_pct_vs_mcsgarch": pct(qm.mean(), qb.mean()),
                             "qlike_trimmed_pct_vs_mcsgarch": pct(qm[keep].mean(), qb[keep].mean()),
                             "median_qlike_pct_vs_mcsgarch": pct(np.median(qm), np.median(qb)),
                             "share_anchors_better_than_mcsgarch": float((qm < qb).mean()),
                             "mse": float(e[mod][m].mean()), "mse_mcsgarch": float(e["mcsgarch"][m].mean())})
        # the largest squared errors of mcsGARCH
        top = f.assign(sq_err=e["mcsgarch"]).nlargest(10, "sq_err")
        share = top["sq_err"].sum() / e["mcsgarch"].sum()
        for r in top.itertuples():
            worst.append({"target": target, "date": r.date.date(), "minute_after_0915": r.minute, "realized": r.y,
                          "mcsgarch": r.mcsgarch, "hybrid_base": r.hybrid_base, "lgbm_base": r.lgbm_base, "har": r.har,
                          "share_of_mcsgarch_total_mse_top10": share})
        # tail share of QLIKE
        for mod in ["mcsgarch"] + MODELS:
            qq = np.sort(q[mod])[::-1]
            extra.append({"target": target, "model": mod, "top1pct_share_of_qlike": float(qq[: max(1, len(qq) // 100)].sum() / qq.sum())})
        if target == "last_hour":
            L = daily_losses(f, ["har", "hybrid_base", "lgbm_base", "mcsgarch"])
            pre = L.index < pd.Timestamp("2026-08-03")
            exp_days = set(f.loc[f["is_expiry"], "date"])
            for sname, m in (("all days", np.ones(len(L), bool)), ("before 3 Aug 2026", pre),
                             ("expiry days", L.index.isin(exp_days))):
                for alt in ("hybrid_base", "lgbm_base"):
                    t = diebold_mariano(L.loc[m, alt], L.loc[m, "har"])
                    extra.append({"target": target, "model": f"{alt} vs har ({sname})", "dm_two_sided_p": t["p"],
                                  "mean_loss_diff": t["mean_diff"], "n_days": t["n"]})
    res, wst, ext = pd.DataFrame(rows), pd.DataFrame(worst), pd.DataFrame(extra)
    res.to_csv(out / "gains_by_subset.csv", index=False)
    wst.to_csv(out / "largest_mcsgarch_errors.csv", index=False)
    ext.to_csv(out / "tails_and_last_hour_tests.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 400, "display.precision", 4):
        text = "\n".join(["REAL DATA. Diagnostics of the R1 forecasting contest (from results/pipeline_nse_v2/forecasts_*.csv).", "",
                          "== QLIKE gain vs mcsGARCH by subset (negative = better than mcsGARCH)",
                          res[res["model"].isin(["har", "lgbm_base", "hybrid_base"])].to_string(index=False), "",
                          "== the 10 largest mcsGARCH squared errors per target", wst.to_string(index=False), "",
                          "== QLIKE tail shares and last-hour DM tests", ext.to_string(index=False)])
    (out.parent / "forecast_diagnostics_stdout.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
