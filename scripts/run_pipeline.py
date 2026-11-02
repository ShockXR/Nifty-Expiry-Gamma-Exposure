"""One command from raw data (or a synthetic market) to every output file.

    python scripts/run_pipeline.py --source synthetic --days 500 --kappa 0.5
    python scripts/run_pipeline.py --source nse --raw data/raw --out results/pipeline_nse_v2

Outputs go to results/pipeline_<source>/ (or --out):
    daily_market.csv             one row per day: labels, realized variance pieces, VIX
    daily_gex_panel.csv          GEX under each sign rule (all strikes and near the money), flip, walls, max pain
    daily_implied_panel.csv      option-implied quantities from each day's published chain (all expiries)
    forecasts_<target>.csv       every test anchor: realized value and every model's forecast
    scoreboard_<target>.csv      QLIKE and MSE by model and subgroup (expiry, regime, development/hold-out)
    tests_<target>.csv           Diebold-Mariano: each model vs mcsGARCH, by subgroup
    ablation_<target>.csv        the GEX ablation: same model with and without each GEX set, by subgroup,
                                 one-sided DM p-values and Holm-adjusted p-values within each family
    mcs_<target>_<subset>.csv    Model Confidence Set p-values (daily QLIKE)
    mcsgarch_params.csv          parameters from every walk-forward refit (paper's daily component and implied one)
    feature_importance_<target>.csv, linear_coefficients_<target>.csv, hyperparameters_<target>.csv
    seed_noise_<target>.csv      (with --seed-study) LightGBM forecast noise across random seeds
    figures/*.png
    run_info.json                configuration, data source label, versions, timing
Synthetic runs are labelled SYNTHETIC in run_info.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from niftygex import plots  # noqa: E402
from niftygex.evaluation import (ablation_table, daily_losses, diebold_mariano, holm,  # noqa: E402
                                 model_confidence_set, scoreboard, subgroups)
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel  # noqa: E402
from niftygex.forecast import WalkForwardConfig, run_walkforward  # noqa: E402
from niftygex.implied import build_implied_panel  # noqa: E402
from niftygex.volatility import McsGarch, daily_egarch_variance  # noqa: E402

TARGETS = {"next15": dict(horizon=15, step=15), "next30": dict(horizon=30, step=15),
           "last_hour": dict(horizon=60, anchors=[315])}
SIGN_SETS = ["gross", "us", "participant", "client", "placebo"]
SIGNED = ["us", "participant", "client"]
# Chosen on the development window only (see docs/MODEL_ITERATIONS_REAL.md): the paper-anchored
# hybrid with the full version 2 base. hybridiv and linear are reported as robustness families.
HEADLINE_KIND = "hybrid"


def load_market(a):
    if a.source == "synthetic":
        from niftygex.simulate import SimConfig, simulate_market
        return simulate_market(SimConfig(n_days=a.days, kappa=a.kappa, seed=a.seed,
                                         participant_noise=a.participant_noise))
    from niftygex.data import build_market_from_nse
    raw = Path(a.raw)
    official = None
    if (raw / "index_daily.csv").exists():  # NSE's own list of trading days (from download_nse.py)
        idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
        official = idx.loc[idx["index"].str.lower() == "nifty 50", "date"]
    return build_market_from_nse(raw / "nifty_1min.csv", raw / "bhavcopy",
                                 raw / "participant_oi" if (raw / "participant_oi").exists() else None,
                                 raw / "india_vix.csv" if (raw / "india_vix.csv").exists() else None,
                                 raw / "nifty_fut_1min.csv" if (raw / "nifty_fut_1min.csv").exists() else None,
                                 oi_unit=a.oi_unit, lot_size=a.lot_size, r=a.rate,
                                 start=a.start, end=a.end, min_coverage=a.min_coverage,
                                 official_days=official, use_file_expiries=True)


def write_daily_market(m, path):
    """One row per day: expiry labels, realized variance pieces, VIX and the chain's spot and forward."""
    R = np.asarray(m["R"], float)
    lab = m["labels"]
    C = np.asarray(m["close"], float)
    last_close = np.array([row[np.isfinite(row)][-1] if np.isfinite(row).any() else np.nan for row in C])
    first_open = np.asarray(m["open"], float)[:, 0]
    df = pd.DataFrame({
        "is_expiry": lab["is_expiry"].to_numpy(), "is_monthly_expiry": lab["is_monthly_expiry"].to_numpy(),
        "regime": lab["regime"].to_numpy(), "weekday": lab["weekday"].to_numpy(),
        "trading_days_to_expiry": lab["trading_days_to_expiry"].to_numpy(),
        "open": first_open, "close_last_bar": last_close,
        "rv_day": (R ** 2).sum(axis=1), "rv_first_hour": (R[:, :60] ** 2).sum(axis=1),
        "rv_last_hour": (R[:, -60:] ** 2).sum(axis=1),
        "minutes_with_bars": np.isfinite(C).sum(axis=1),
        "vix": np.asarray(m["vix"], float),
        "chain_spot": [c.attrs.get("spot") for c in m["chains"]],
        "chain_expiry": [c.attrs.get("expiry") for c in m["chains"]],
        "bhavcopy_missing": [bool(c.attrs.get("missing", False)) for c in m["chains"]],
    }, index=pd.DatetimeIndex(m["days"], name="date"))
    if m.get("futures") is not None:
        df["nifty_futures_traded_value"] = m["futures"]["fut_value"].reindex(df.index).to_numpy()
    df.to_csv(path)


def build_models(has_oracle: bool, full: bool = True):
    models = ["naive_last", "naive_prevday", "vix", "implied", "har", "mcsgarch", "mcsgarch_iv",
              "linear_v1", "lgbm_v1", "hybrid_v1", "hybrid_v1gex"]
    sets = list(SIGN_SETS) + (["oracle"] if has_oracle else [])
    kinds = ("linear", "lgbm", "hybrid", "hybridiv") if full else ("linear", "hybridiv")
    for kind in kinds:
        models.append(f"{kind}_base")
        models += [f"{kind}_gex_{s}" for s in sets]
    return models


def ablation_pairs(models):
    pairs = []
    for kind in ("linear", "lgbm", "hybrid", "hybridiv"):
        if f"{kind}_base" not in models:
            continue
        for s in SIGN_SETS + ["oracle"]:
            if f"{kind}_gex_{s}" in models:
                pairs.append((f"{kind}: base vs base+GEX", s, f"{kind}_base", f"{kind}_gex_{s}"))
        for s in SIGNED + ["oracle"]:
            if f"{kind}_gex_{s}" in models and f"{kind}_gex_gross" in models:
                pairs.append((f"{kind}: unsigned vs signed", s, f"{kind}_gex_gross", f"{kind}_gex_{s}"))
        # same columns, real days versus other days: controls for the cost of extra columns
        if f"{kind}_gex_placebo" in models and f"{kind}_gex_us" in models:
            pairs.append((f"{kind}: placebo vs real (same columns)", "us", f"{kind}_gex_placebo", f"{kind}_gex_us"))
    if "hybrid_v1gex" in models:
        pairs.append(("v1 (first run): base vs base+GEX", "all v1 GEX", "hybrid_v1", "hybrid_v1gex"))
    for b, a, lab in (("hybrid_v1", "hybrid_base", "implied+calendar block"),
                      ("linear_v1", "linear_base", "implied+calendar block"),
                      ("lgbm_v1", "lgbm_base", "implied+calendar block"),
                      ("hybrid_base", "hybridiv_base", "implied daily component")):
        if b in models and a in models:
            pairs.append(("base upgrades", lab, b, a))
    return pairs


def seed_study(m, ds, base_cfg: WalkForwardConfig, seeds, models):
    """Run the LightGBM models with different seeds; report spread and seed-vs-seed DM tests."""
    preds = {}
    for s in seeds:
        cfg = WalkForwardConfig(**{**base_cfg.__dict__, "models": ["mcsgarch"] + models, "seed": s, "n_seeds": 1})
        p, _ = run_walkforward(m, ds, cfg)
        preds[s] = p
    rows = []
    ref = preds[seeds[0]]
    for mdl in models:
        L = {s: daily_losses(p.assign(**{mdl: p[mdl]}), [mdl])[mdl] for s, p in preds.items()}
        q = {s: float(L[s].mean()) for s in seeds}
        for s in seeds:
            rows.append({"model": mdl, "seed": s, "qlike": q[s],
                         "qlike_vs_mcsgarch_pct": 100 * (q[s] / daily_losses(ref, ["mcsgarch"])["mcsgarch"].mean() - 1)})
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1:]:
                dm = diebold_mariano(L[s1], L[s2])
                rows.append({"model": mdl, "seed": f"{s1} vs {s2}", "qlike": np.nan,
                             "abs_change_pct": abs(100 * (q[s2] / q[s1] - 1)), "dm_p_two_sided": dm["p"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "nse"], default="synthetic")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--participant-noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--oi-unit", default="keep_units",
                    help="NSE files report OI in shares; keep_units keeps shares and uses lot size 1 (exact)")
    ap.add_argument("--start", default="2023-01-01", help="first day of real data to use")
    ap.add_argument("--end", default=None, help="last day of real data to use")
    ap.add_argument("--holdout-start", default="2025-07-01",
                    help="first day of the hold-out period (reported separately; never used for design choices)")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="drop days whose minute bars cover less than this share of the session")
    ap.add_argument("--rate", type=float, default=0.065, help="risk-free rate for Black-76")
    ap.add_argument("--lot-size", type=float, default=None)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--tune", action="store_true", help="tune LightGBM inside each training window")
    ap.add_argument("--tune-every", type=int, default=3)
    ap.add_argument("--n-seeds", type=int, default=1, help="LightGBM: average log forecasts over this many seeds")
    ap.add_argument("--seed-study", type=int, default=0, help="also rerun LightGBM models with this many seeds")
    ap.add_argument("--lean", action="store_true", help="fewer model families (faster, for tests)")
    ap.add_argument("--min-train", type=int, default=150)
    ap.add_argument("--block", type=int, default=21)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out or f"results/pipeline_{a.source}")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = load_market(a)
    label = ("SYNTHETIC (simulated market, not real data)" if a.source == "synthetic" else
             "REAL DATA: NSE bhavcopy, participant OI and India VIX; third-party 1-minute Nifty bars "
             "validated against NSE daily values")
    holdout = a.holdout_start if a.source == "nse" else None
    print(f"[{label}] {len(m['days'])} days loaded", flush=True)

    write_daily_market(m, out / "daily_market.csv")
    panel = build_gex_panel(m)
    panel.to_csv(out / "daily_gex_panel.csv")
    implied = build_implied_panel(m)
    implied.to_csv(out / "daily_implied_panel.csv")
    print(f"panels built ({time.time() - t0:.0f}s)", flush=True)
    plots.gex_timeseries(panel, out / "figures" / "gex_by_sign_rule.png",
                         f"Net GEX / 20-day median gross GEX, by sign rule ({'synthetic' if a.source == 'synthetic' else 'NSE'})")

    # Full-sample mcsGARCH for the diurnal figure and a parameter table like the paper's Table 2
    is_exp = m["labels"]["is_expiry"].to_numpy()
    h, _ = daily_egarch_variance(m["R"].sum(axis=1))
    full = McsGarch(expiry_diurnal=True, smooth=5).fit(m["R"], h, is_exp, compute_tstats=True)
    plots.diurnal_curves(full.s, full.s_expiry, out / "figures" / "diurnal_curves.png",
                         "mcsGARCH diurnal variance curve (in-sample)")
    pd.DataFrame([{k: v for k, v in full.params.items() if k != "tstats"} |
                  {f"z_{k}": v for k, v in full.params["tstats"].items()}]).to_csv(out / "mcsgarch_full_sample.csv", index=False)

    has_oracle = "dealer_next" in m
    models = build_models(has_oracle, full=not a.lean)
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle" or has_oracle}
    cfg = WalkForwardConfig(models=models, gex_sets=sets, tune=a.tune, tune_every=a.tune_every,
                            n_seeds=a.n_seeds, seed=a.seed, min_train=a.min_train, block=a.block)
    pairs = ablation_pairs(models)
    params_all, timing = [], {}
    for name in a.targets:
        t1 = time.time()
        ds = build_dataset(m, panel, implied=implied, **TARGETS[name])
        pred, info = run_walkforward(m, ds, cfg)
        timing[name] = round(time.time() - t1, 1)
        present = [x for x in models if x in pred]
        pred.to_csv(out / f"forecasts_{name}.csv", index=False)
        groups = subgroups(pred, holdout_start=holdout)
        sb = scoreboard(pred, present, groups)
        sb.to_csv(out / f"scoreboard_{name}.csv", index=False)
        for g in ("all", "expiry"):
            plots.scoreboard_bars(sb, g, out / "figures" / f"scoreboard_{name}_{g}.png",
                                  f"{name}, {g} days: out-of-sample QLIKE vs mcsGARCH")
        tests = []
        for subset, mask in groups.items():
            sub = pred[mask]
            if sub["date"].nunique() < 10:
                continue
            L = daily_losses(sub, present)
            for mdl in present:
                if mdl != "mcsgarch":
                    tests.append({"subset": subset, "comparison": f"mcsgarch vs {mdl}",
                                  **diebold_mariano(L["mcsgarch"], L[mdl], alternative="greater")})
            if subset in ("all", "expiry"):
                try:
                    mcs = model_confidence_set(L[present], reps=1000)
                    mcs.assign(subset=subset).to_csv(out / f"mcs_{name}_{subset}.csv")
                except Exception as exc:  # small samples can make the MCS bootstrap fail
                    print(f"MCS skipped for {name}/{subset}: {exc}")
        pd.DataFrame(tests).to_csv(out / f"tests_{name}.csv", index=False)
        abl = ablation_table(pred, pairs, groups)
        if not abl.empty:
            abl["p_holm_in_family"] = np.nan
            for (fam, grp), idx in abl.groupby(["family", "group"]).groups.items():
                if "placebo" in fam:
                    continue
                keep = [i for i in idx if abl.loc[i, "label"] != "placebo"]
                abl.loc[keep, "p_holm_in_family"] = holm(abl.loc[keep, "p_one_sided"].to_numpy())
        abl.to_csv(out / f"ablation_{name}.csv", index=False)
        if info["importance"]:
            imp = pd.concat(info["importance"], axis=1)
            fi = {}
            for mdl in sorted({c.split("@")[0] for c in imp.columns}):
                fi[mdl] = imp[[c for c in imp.columns if c.split("@")[0] == mdl]].mean(axis=1)
            pd.DataFrame(fi).to_csv(out / f"feature_importance_{name}.csv")
        if info["linear_coef"]:
            lc = pd.concat(info["linear_coef"], axis=1)
            rows = []
            for mdl in sorted({c.split("@")[0] for c in lc.columns}):
                sub = lc[[c for c in lc.columns if c.split("@")[0] == mdl]]
                for feat, vals in sub.iterrows():
                    v = vals.dropna()
                    rows.append({"model": mdl, "feature": feat, "mean_coef": v.mean(), "share_positive": (v > 0).mean(),
                                 "folds": len(v), "last_fold_coef": v.iloc[-1] if len(v) else np.nan})
            pd.DataFrame(rows).to_csv(out / f"linear_coefficients_{name}.csv", index=False)
        pd.DataFrame(info["hyper"]).to_csv(out / f"hyperparameters_{name}.csv", index=False)
        params_all.append(pd.DataFrame(info["mcs_params"]).assign(target=name, daily_component="egarch (paper)"))
        if info["mcs_iv_params"]:
            params_all.append(pd.DataFrame(info["mcs_iv_params"]).assign(target=name, daily_component="implied"))
        if a.seed_study > 1:
            lgbm_models = [x for x in ("lgbm_base", "hybrid_base", "hybridiv_base", f"{HEADLINE_KIND}_gex_us") if x in models]
            seeds = [a.seed + k for k in range(a.seed_study)]
            seed_study(m, ds, cfg, seeds, lgbm_models).to_csv(out / f"seed_noise_{name}.csv", index=False)
        print(f"{name}: {len(pred)} test anchors, {pred['date'].nunique()} test days ({timing[name]:.0f}s)", flush=True)
    pd.concat(params_all).to_csv(out / "mcsgarch_params.csv", index=False)

    import lightgbm, arch, scipy  # noqa: E401
    json.dump({"source": label, "args": vars(a), "seconds": round(time.time() - t0, 1), "target_seconds": timing,
               "models": models, "gex_sets": sets, "headline_kind": HEADLINE_KIND,
               "holdout_start": holdout,
               "protocol": ("Design choices were made on walk-forward test days before the hold-out start only; "
                            "the hold-out period was evaluated once with the final configuration.") if holdout else None,
               "days": [str(pd.Timestamp(m["days"][0]).date()), str(pd.Timestamp(m["days"][-1]).date()), len(m["days"])],
               "config": m.get("config"), "python": sys.version.split()[0],
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
                            "arch": arch.__version__, "scipy": scipy.__version__}},
              open(out / "run_info.json", "w"), indent=2, default=str)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
