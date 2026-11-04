"""P104 phase 2: the registered hypotheses and breakdowns (docs/prereg/P104_THIRD_INDEX_FORECAST.md). REAL DATA.

Fin Nifty (Nifty Financial Services), the third index. A copy of experiments/p91_banknifty_summary.py with the
paths changed. Reads the outputs of the registered command (results/pipeline_nse_finnifty/) and, for the
comparison on the same dates, the committed Nifty v2 forecasts (results/pipeline_nse_v2/forecasts_<target>.csv)
and Bank Nifty P91 forecasts (results/pipeline_nse_banknifty/forecasts_<target>.csv), neither rerun. It fits
nothing. Two pieces are computed here because the pipeline does not write them (prereg Sections 6 and 7): the
weekly against monthly-only split, and Nifty and Bank Nifty scored on the Fin Nifty test dates. Both use the
pipeline's own functions (daily_losses, diebold_mariano with alternative="greater").

    python experiments/p104_finnifty_summary.py

Writes results/pipeline_nse_finnifty/p104/*.csv and prints a summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.evaluation import daily_losses, diebold_mariano, holm, qlike

ROOT = Path(__file__).resolve().parents[1]
BN = ROOT / "results" / "pipeline_nse_finnifty"  # the index under test (name kept from the P91 script)
NV2 = ROOT / "results" / "pipeline_nse_v2"
BNK = ROOT / "results" / "pipeline_nse_banknifty"
OUT = BN / "p104"
TARGETS = ["next15", "next30", "last_hour"]
MONTHLY_ONLY_START = pd.Timestamp("2024-11-20")  # first day of the pipeline's second regime (prereg Section 6)
ALPHA = 0.05


def tests(target):
    return pd.read_csv(BN / f"tests_{target}.csv")


def sb(target):
    return pd.read_csv(BN / f"scoreboard_{target}.csv")


def dm_row(target, subset, model="hybrid_base", loss="qlike"):
    t = tests(target)
    r = t[(t["subset"] == subset) & (t["loss"] == loss) & (t["comparison"] == f"mcsgarch vs {model}")]
    if r.empty:
        return None
    return r.iloc[0]


def sb_row(target, group, model):
    s = sb(target)
    r = s[(s["group"] == group) & (s["model"] == model)]
    return None if r.empty else r.iloc[0]


def compare(pred: pd.DataFrame, model="hybrid_base", ref="mcsgarch") -> dict:
    """Score `model` against `ref` exactly as the pipeline does (anchor-mean QLIKE; DM on daily means)."""
    q_ref, q_mod = float(np.mean(qlike(pred["y"], pred[ref]))), float(np.mean(qlike(pred["y"], pred[model])))
    L = daily_losses(pred, [ref, model])
    dm = diebold_mariano(L[ref], L[model], alternative="greater")
    return {"days": int(pred["date"].nunique()), "anchors": len(pred), "qlike_mcsgarch": q_ref, "qlike_model": q_mod,
            "change_pct": 100 * (q_mod / q_ref - 1), "dm_stat": dm["stat"], "p_one_sided": dm["p"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.load(open(BN / "run_info.json"))
    rows = []

    # H1 (primary), H2, H3 ------------------------------------------------------------------------------------
    def hyp(name, target, subset):
        r, a, b = dm_row(target, subset), sb_row(target, subset, "mcsgarch"), sb_row(target, subset, "hybrid_base")
        return {"hypothesis": name, "target": target, "subset": subset, "days": int(r["n"]),
                "qlike_mcsgarch": a["qlike"], "qlike_hybrid_base": b["qlike"], "change_pct": b["qlike_vs_mcsgarch_pct"],
                "dm_stat": r["stat"], "p_one_sided": r["p"], "p_reversed": 1 - r["p"]}

    h1 = [hyp("H1", t, "all") for t in ("next15", "next30")]
    for r, ph in zip(h1, holm([r["p_one_sided"] for r in h1])):
        r["p_holm"] = ph
    h3 = [hyp("H3", t, "expiry") for t in ("next15", "next30")]
    for r, ph in zip(h3, holm([r["p_one_sided"] for r in h3])):
        r["p_holm"] = ph
    h2 = hyp("H2", "last_hour", "all")
    h2["p_holm"] = np.nan
    h3x = hyp("H3 (reported, no hypothesis)", "last_hour", "expiry")
    h3x["p_holm"] = np.nan
    rows = h1 + [h2] + h3 + [h3x]
    hyp_df = pd.DataFrame(rows)

    n_ok = sum(r["p_holm"] <= ALPHA for r in h1)
    if n_ok == 2:
        h1_verdict = "Replicated"
    elif n_ok == 1:
        h1_verdict = "Partly replicated (" + ", ".join(r["target"] for r in h1 if r["p_holm"] <= ALPHA) + ")"
    else:
        if any(r["change_pct"] > 0 and r["p_reversed"] <= ALPHA for r in h1):
            h1_verdict = "Not replicated: reversed"
        elif all(r["change_pct"] < 0 for r in h1):
            h1_verdict = "Not replicated: same direction, not significant"
        else:
            h1_verdict = "Not replicated"
    h2_verdict = "success" if h2["p_one_sided"] <= ALPHA else "not supported"
    h3_verdict = "success" if all(r["p_holm"] <= ALPHA for r in h3) else "not supported"
    hyp_df["verdict"] = [h1_verdict] * 2 + [h2_verdict] + [h3_verdict] * 2 + ["(none)"]
    hyp_df.to_csv(OUT / "hypotheses.csv", index=False)

    # H4: GEX ablation in the hybrid ----------------------------------------------------------------------------
    abl_rows = []
    for t in TARGETS:
        a = pd.read_csv(BN / f"ablation_{t}.csv")
        keep = a[(a["family"].isin(["hybrid: base vs base+GEX", "hybrid: placebo vs real (same columns)"]))
                 & (a["group"].isin(["all", "expiry"]))]
        abl_rows.append(keep.assign(target=t))
    abl = pd.concat(abl_rows, ignore_index=True)
    abl.to_csv(OUT / "h4_ablation_hybrid.csv", index=False)
    main_cells = abl[(abl["family"] == "hybrid: base vs base+GEX") & (abl["group"] == "all") & (abl["label"] != "placebo")]
    h4_fail = bool((main_cells["p_holm_in_family"] <= ALPHA).any())
    h4_verdict = "prediction failed" if h4_fail else "prediction held (no GEX set improves the hybrid after Holm)"

    # Scoreboard against mcsGARCH for every model (Section 7 item 3) --------------------------------------------
    sbr = []
    for t in TARGETS:
        s, tt = sb(t), tests(t)
        for g in ("all", "expiry", "non_expiry", "multi_weekly", "single_weekly_thu", "single_weekly_tue",
                  "development", "holdout", "open_0915", "excl_0915"):
            for _, r in s[s["group"] == g].iterrows():
                if r["model"] == "mcsgarch":
                    continue
                q = tt[(tt["subset"] == g) & (tt["comparison"] == f"mcsgarch vs {r['model']}")]
                pq = q.loc[q["loss"] == "qlike", "p"]
                pm = q.loc[q["loss"] == "mse", "p"]
                sbr.append({"target": t, "group": g, "model": r["model"], "days": r["days"],
                            "qlike_vs_mcsgarch_pct": r["qlike_vs_mcsgarch_pct"],
                            "p_qlike": pq.iloc[0] if len(pq) else np.nan,
                            "qlike_trim_vs_mcsgarch_pct": r["qlike_trim_vs_mcsgarch_pct"],
                            "mse_vs_mcsgarch_pct": r["mse_vs_mcsgarch_pct"], "p_mse": pm.iloc[0] if len(pm) else np.nan})
    pd.DataFrame(sbr).to_csv(OUT / "scoreboard_vs_mcsgarch.csv", index=False)

    # Weekly period against monthly-only period (Section 6 item 4) ----------------------------------------------
    per = []
    for t in TARGETS:
        p = pd.read_csv(BN / f"forecasts_{t}.csv", parse_dates=["date"])
        for name, mask in (("weekly period (to 19 Nov 2024)", p["date"] < MONTHLY_ONLY_START),
                           ("monthly-only period (from 20 Nov 2024)", p["date"] >= MONTHLY_ONLY_START)):
            for sub_name, sub in (("all", p[mask]), ("expiry", p[mask & p["is_expiry"].astype(bool)])):
                if sub["date"].nunique() < 10:
                    per.append({"target": t, "period": name, "subset": sub_name, "days": int(sub["date"].nunique()),
                                "note": "fewer than 10 days: no test (same rule as the pipeline)"})
                    continue
                per.append({"target": t, "period": name, "subset": sub_name, **compare(sub)})
    pd.DataFrame(per).to_csv(OUT / "period_split.csv", index=False)

    # Nifty and Bank Nifty on the same dates (Section 7 item 5) -------------------------------------------------
    bn_dates = set(pd.read_csv(BN / "forecasts_next15.csv", usecols=["date"], parse_dates=["date"])["date"])
    nf = []
    for t in TARGETS:
        n = pd.read_csv(NV2 / f"forecasts_{t}.csv", parse_dates=["date"])
        n = n[n["date"].isin(bn_dates)]
        for sub_name, sub in (("all", n), ("expiry (Nifty's own expiry days)", n[n["is_expiry"].astype(bool)])):
            nf.append({"index": "Nifty (v2 forecasts, same dates)", "target": t, "subset": sub_name, **compare(sub)})
        k = pd.read_csv(BNK / f"forecasts_{t}.csv", parse_dates=["date"])
        k = k[k["date"].isin(bn_dates)]
        for sub_name, sub in (("all", k), ("expiry (Bank Nifty's own expiry days)", k[k["is_expiry"].astype(bool)])):
            nf.append({"index": "Bank Nifty (P91 forecasts, same dates)", "target": t, "subset": sub_name,
                       **compare(sub)})
        b = pd.read_csv(BN / f"forecasts_{t}.csv", parse_dates=["date"])
        for sub_name, sub in (("all", b), ("expiry (Fin Nifty's own expiry days)", b[b["is_expiry"].astype(bool)])):
            nf.append({"index": "Fin Nifty", "target": t, "subset": sub_name, **compare(sub)})
    pd.DataFrame(nf).to_csv(OUT / "other_indices_same_dates.csv", index=False)

    # Model Confidence Set membership ----------------------------------------------------------------------------
    mcs_rows = []
    for t in TARGETS:
        for g in ("all", "expiry"):
            for suf, loss in (("", "qlike"), ("_mse", "mse")):
                f = BN / f"mcs_{t}_{g}{suf}.csv"
                if not f.exists():
                    continue
                m = pd.read_csv(f, index_col=0)
                mcs_rows.append({"target": t, "subset": g, "loss": loss, "models_in_set": int(m["in_mcs"].sum()),
                                 "hybrid_base_in": bool(m.loc["hybrid_base", "in_mcs"]),
                                 "mcsgarch_in": bool(m.loc["mcsgarch", "in_mcs"]),
                                 "members": " ".join(m.index[m["in_mcs"]])})
    pd.DataFrame(mcs_rows).to_csv(OUT / "mcs_membership.csv", index=False)

    # Data checks of prereg Section 2, recomputed from the run's own files -------------------------------------
    dm = pd.read_csv(BN / "daily_market.csv", index_col=0, parse_dates=True)
    test = dm.index[150:]
    tl = dm.loc[test]
    gex = pd.read_csv(BN / "daily_gex_panel.csv", index_col=0)
    checks = {
        "days": len(dm), "first_day": str(dm.index[0].date()), "last_day": str(dm.index[-1].date()),
        "run_info_days": info["days"],
        "last_training_day": str(dm.index[149].date()), "first_test_day": str(test[0].date()), "test_days": len(test),
        "test_days_in_forecasts": {t: int(pd.read_csv(BN / f"forecasts_{t}.csv", usecols=["date"])["date"].nunique())
                                   for t in TARGETS},
        "forecasts_per_day": {t: round(len(pd.read_csv(BN / f"forecasts_{t}.csv", usecols=["date"]))
                                       / int(pd.read_csv(BN / f"forecasts_{t}.csv", usecols=["date"])["date"].nunique()), 3)
                              for t in TARGETS},
        "bhavcopy_missing_days": int(dm["bhavcopy_missing"].sum()),
        "vix_missing_days": int(dm["vix"].isna().sum()),
        "futures_value_missing_days": int(dm[[c for c in dm if c.endswith("_futures_traded_value")][0]].isna().sum()),
        "test_days_by_regime": tl["regime"].value_counts().to_dict(),
        "expiry_test_days": int(tl["is_expiry"].sum()),
        "monthly_expiry_test_days": int((tl["is_expiry"] & tl["is_monthly_expiry"]).sum()),
        "expiry_test_days_weekly_period": int(tl.loc[test < MONTHLY_ONLY_START, "is_expiry"].sum()),
        "expiry_test_days_monthly_only": int(tl.loc[test >= MONTHLY_ONLY_START, "is_expiry"].sum()),
        "flip_participant_missing": int(gex["flip_participant"].isna().sum()),
        "gex_panel_missing_other_columns": int(gex.drop(columns=["flip_participant"]).isna().sum().sum()),
        "development_days": int((test < pd.Timestamp("2025-07-01")).sum()),
        "holdout_days": int((test >= pd.Timestamp("2025-07-01")).sum()),
    }
    json.dump(checks, open(OUT / "data_checks.json", "w"), indent=2, default=str)

    # Print -----------------------------------------------------------------------------------------------------
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("REAL DATA. P104 Fin Nifty replication (third index).")
    print(f"H1 verdict: {h1_verdict}")
    print(hyp_df.to_string(index=False))
    print(f"H4 verdict: {h4_verdict}")
    print(abl[["target", "family", "label", "group", "days", "change_pct", "p_one_sided", "p_holm_in_family"]].to_string(index=False))
    print(pd.DataFrame(per).to_string(index=False))
    print(pd.DataFrame(nf).to_string(index=False))
    print(pd.DataFrame(mcs_rows).drop(columns=["members"]).to_string(index=False))
    print(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    main()
