"""Summaries of the development runs behind docs/MODEL_ITERATIONS_REAL.md (REAL DATA).

The development runs used the real data cut at 30 June 2025 (the hold-out
period from 1 July 2025 was never computed during development). Their raw
forecasts are large and live outside the repository; this script turns them
into the small tables in results/pipeline_nse_v2/development/.

    python experiments/v2_development_summary.py --dev-dir <folder with dev runs> \
        --baseline results/pipeline_nse --out results/pipeline_nse_v2/development

Expected sub-folders of --dev-dir (each holds forecasts_<target>.csv):
    dev2         all models, single seed (also last_hour_pooled)
    dev_seed1/2  a subset of LightGBM models with seeds 1 and 2 (next30)
    dev_s3       3-seed averages, default hyperparameters (next30)
    dev_s3tune   3-seed averages, tuned inside each training window (next30)
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.evaluation import ablation_table, daily_losses, diebold_mariano, scoreboard, subgroups

DEV_END = pd.Timestamp("2025-06-30")
META = ["t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y"]


def load(path):
    p = pd.read_csv(path, parse_dates=["date"])
    return p[p["date"] <= DEV_END].sort_values(["t", "minute"]).reset_index(drop=True)


def vs_mcs(p, groups, run, target):
    models = [c for c in p.columns if c not in META]
    sb = scoreboard(p, models, groups)
    rows = []
    for g, mask in groups.items():
        L = daily_losses(p[mask], models)
        for m in models:
            dm = diebold_mariano(L["mcsgarch"], L[m], alternative="greater") if m != "mcsgarch" else {"p": np.nan}
            rows.append({"group": g, "model": m, "dm_p_vs_mcsgarch": dm["p"]})
    out = sb.merge(pd.DataFrame(rows), on=["group", "model"])
    return out.assign(run=run, target=target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--baseline", default="results/pipeline_nse")
    ap.add_argument("--out", default="results/pipeline_nse_v2/development")
    a = ap.parse_args()
    D, out = Path(a.dev_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. scoreboards on the development window: the first real-data run (baseline) and dev2
    frames = []
    for run, folder, targets in (("baseline_v1", Path(a.baseline), ["next15", "next30", "last_hour"]),
                                 ("dev2", D / "dev2", ["next15", "next30", "last_hour", "last_hour_pooled"])):
        for t in targets:
            f = folder / f"forecasts_{t}.csv"
            if f.exists():
                p = load(f)
                frames.append(vs_mcs(p, subgroups(p), run, t))
    pd.concat(frames).to_csv(out / "dev_scoreboards.csv", index=False)

    # 2. GEX ablations on the development window (single seed)
    abl = []
    for t in ["next15", "next30", "last_hour", "last_hour_pooled"]:
        f = D / "dev2" / f"forecasts_{t}.csv"
        if not f.exists():
            continue
        p = load(f)
        pairs = []
        for kind in ("linear", "hybrid", "hybridiv"):
            for s in ("gross", "us", "participant", "client", "placebo"):
                pairs.append((f"{kind}: base vs base+GEX", s, f"{kind}_base", f"{kind}_gex_{s}"))
            for s in ("us", "participant", "client"):
                pairs.append((f"{kind}: unsigned vs signed", s, f"{kind}_gex_gross", f"{kind}_gex_{s}"))
        pairs += [("v1 (first run): base vs base+GEX", "all v1 GEX", "hybrid_v1", "hybrid_v1gex"),
                  ("base upgrades", "implied+calendar block (hybrid)", "hybrid_v1", "hybrid_base"),
                  ("base upgrades", "implied+calendar block (linear)", "linear_v1", "linear_base"),
                  ("base upgrades", "implied+calendar block (lgbm)", "lgbm_v1", "lgbm_base"),
                  ("base upgrades", "implied daily component (hybrid to hybridiv)", "hybrid_base", "hybridiv_base")]
        g = subgroups(p)
        g = {k: v for k, v in g.items() if k in ("all", "expiry", "non_expiry")}
        abl.append(ablation_table(p, pairs, g).assign(target=t))
    pd.concat(abl).to_csv(out / "dev_ablation_single_seed.csv", index=False)

    # 3. seed noise (next30): same model, seeds 0, 1, 2
    runs = {0: D / "dev2", 1: D / "dev_seed1", 2: D / "dev_seed2"}
    if all((r / "forecasts_next30.csv").exists() for r in runs.values()):
        P = {s: load(r / "forecasts_next30.csv") for s, r in runs.items()}
        models = [c for c in P[1].columns if c not in META and c != "mcsgarch"]
        L = {s: daily_losses(p, ["mcsgarch"] + models) for s, p in P.items()}
        rows = []
        for m in models:
            for s in runs:
                rows.append({"model": m, "comparison": f"seed {s}", "qlike_vs_mcsgarch_pct": 100 * (L[s][m].mean() / L[s]["mcsgarch"].mean() - 1)})
            for s1, s2 in itertools.combinations(runs, 2):
                rows.append({"model": m, "comparison": f"seed {s1} vs seed {s2}",
                             "abs_qlike_change_pct": abs(100 * (L[s2][m].mean() / L[s1][m].mean() - 1)),
                             "dm_p_two_sided": diebold_mariano(L[s1][m], L[s2][m])["p"]})
        for kind in ("hybrid", "hybridiv"):
            for st in ("participant", "us"):
                for s in runs:
                    d = diebold_mariano(L[s][f"{kind}_base"], L[s][f"{kind}_gex_{st}"], alternative="greater")
                    rows.append({"model": f"{kind}_gex_{st} vs {kind}_base", "comparison": f"GEX ablation, seed {s}",
                                 "change_pct": 100 * (L[s][f"{kind}_gex_{st}"].mean() / L[s][f"{kind}_base"].mean() - 1),
                                 "p_one_sided": d["p"]})
        pd.DataFrame(rows).to_csv(out / "dev_seed_noise_next30.csv", index=False)

    # 4. tuning inside the training window versus defaults (3-seed averages, next30)
    fa, fb = D / "dev_s3" / "forecasts_next30.csv", D / "dev_s3tune" / "forecasts_next30.csv"
    if fa.exists() and fb.exists():
        A, B = load(fa), load(fb)
        models = [c for c in A.columns if c not in META]
        LA, LB = daily_losses(A, models), daily_losses(B, models)
        rows = []
        for m in models:
            rows.append({"model": m, "default_vs_mcsgarch_pct": 100 * (LA[m].mean() / LA["mcsgarch"].mean() - 1),
                         "tuned_vs_mcsgarch_pct": 100 * (LB[m].mean() / LB["mcsgarch"].mean() - 1),
                         "tuned_vs_default_pct": 100 * (LB[m].mean() / LA[m].mean() - 1),
                         "dm_p_two_sided": diebold_mariano(LA[m], LB[m])["p"] if m != "mcsgarch" else np.nan})
        for st in ("participant", "us"):
            for lab, LL in (("default", LA), ("tuned", LB)):
                d = diebold_mariano(LL["hybrid_base"], LL[f"hybrid_gex_{st}"], alternative="greater")
                rows.append({"model": f"hybrid_gex_{st} vs hybrid_base ({lab}, 3 seeds)",
                             "tuned_vs_default_pct": np.nan,
                             "ablation_change_pct": 100 * (LL[f"hybrid_gex_{st}"].mean() / LL["hybrid_base"].mean() - 1),
                             "ablation_p_one_sided": d["p"]})
        pd.DataFrame(rows).to_csv(out / "dev_tuning_next30.csv", index=False)
    # 5. checks of the implied panel against realized variance (development window, from the final run's panels)
    res = out.parent
    if (res / "daily_implied_panel.csv").exists() and (res / "daily_market.csv").exists():
        ip = pd.read_csv(res / "daily_implied_panel.csv", parse_dates=["publish_date"]).set_index("publish_date")
        dm = pd.read_csv(res / "daily_market.csv", parse_dates=["date"]).set_index("date")
        df = pd.DataFrame({"rv": dm["rv_day"], "is_expiry": dm["is_expiry"].astype(bool),
                           "imp_prev": (ip["iv_near"] ** 2 / 252).shift(1),            # chain of day t-1
                           "vix_prev": (dm["vix"] ** 2 / 252).shift(1),
                           "straddle_ratio": ip["straddle_var_near"] / ip["w_near"],
                           "iv_m21": ip["iv_m21"], "vix": dm["vix"]}).dropna()
        dev = df[(df.index <= DEV_END) & (df.index >= "2023-08-10")]
        rows = []
        for name, sub in (("all days", dev), ("expiry days", dev[dev["is_expiry"]]), ("non-expiry days", dev[~dev["is_expiry"]])):
            rows.append({"sample": name, "days": len(sub),
                         "corr_log_rv_log_implied_prev": np.corrcoef(np.log(sub["rv"]), np.log(sub["imp_prev"]))[0, 1],
                         "corr_log_rv_log_vix_prev": np.corrcoef(np.log(sub["rv"]), np.log(sub["vix_prev"]))[0, 1],
                         "mean_rv_over_implied": sub["rv"].mean() / sub["imp_prev"].mean()})
        for lab, lo, hi in (("2023-08-10 to 2024-02-29", "2023-08-10", "2024-02-29"),
                            ("2024-03-01 to 2024-11-19", "2024-03-01", "2024-11-19"),
                            ("2024-11-20 to 2025-06-30", "2024-11-20", "2025-06-30")):
            sub = dev[(dev.index >= lo) & (dev.index <= hi)]
            for e in (False, True):
                s2 = sub[sub["is_expiry"] == e]
                rows.append({"sample": f"{lab}, {'expiry' if e else 'non-expiry'} days", "days": len(s2),
                             "mean_rv_over_implied": s2["rv"].mean() / s2["imp_prev"].mean()})
        rows.append({"sample": "all development days: straddle variance / iv_near^2 T (mean)",
                     "days": len(dev), "mean_rv_over_implied": np.nan, "value": dev["straddle_ratio"].mean()})
        rows.append({"sample": "all development days: corr(iv_m21, India VIX)", "days": len(dev),
                     "value": np.corrcoef(dev["iv_m21"], dev["vix"])[0, 1]})
        pd.DataFrame(rows).to_csv(out / "dev_implied_checks.csv", index=False)
    print("written to", out)


if __name__ == "__main__":
    main()
