"""One command from raw data (or a synthetic market) to every output file.

    python scripts/run_pipeline.py --source synthetic --days 500 --kappa 0.5
    python scripts/run_pipeline.py --source nse --raw data/raw

Outputs go to results/pipeline_<source>/:
    daily_gex_panel.csv          GEX under each sign rule, flip, walls, max pain
    forecasts_<target>.csv       every test anchor: realized value and every model's forecast
    scoreboard_<target>.csv      QLIKE and MSE by model, overall / expiry / non-expiry
    tests_<target>.csv           Diebold-Mariano: each model vs mcsGARCH, and the GEX ablations
    mcs_<target>.csv             Model Confidence Set p-values (daily QLIKE)
    mcsgarch_params.csv          parameters from every walk-forward refit
    feature_importance_<target>.csv
    figures/*.png
    run_info.json                configuration, data source label, versions
Synthetic runs are labelled SYNTHETIC in run_info.json and in every file name's folder.
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
from niftygex.evaluation import daily_losses, diebold_mariano, model_confidence_set, scoreboard  # noqa: E402
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel  # noqa: E402
from niftygex.forecast import WalkForwardConfig, run_walkforward  # noqa: E402
from niftygex.volatility import McsGarch, daily_egarch_variance  # noqa: E402

TARGETS = {"next15": dict(horizon=15, step=15), "next30": dict(horizon=30, step=15),
           "last_hour": dict(horizon=60, anchors=[315])}


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
    df.to_csv(path)


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
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="drop days whose minute bars cover less than this share of the session")
    ap.add_argument("--rate", type=float, default=0.065, help="risk-free rate for Black-76")
    ap.add_argument("--lot-size", type=float, default=None)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out or f"results/pipeline_{a.source}")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = load_market(a)
    label = ("SYNTHETIC (simulated market, not real data)" if a.source == "synthetic" else
             "REAL DATA: NSE bhavcopy, participant OI and India VIX; third-party 1-minute Nifty bars "
             "validated against NSE daily values")
    print(f"[{label}] {len(m['days'])} days loaded", flush=True)

    write_daily_market(m, out / "daily_market.csv")
    panel = build_gex_panel(m)
    panel.to_csv(out / "daily_gex_panel.csv")
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

    models = ["naive_last", "naive_prevday", "vix", "har", "mcsgarch", "lgbm_base", "lgbm_gex",
              "hybrid_base", "hybrid_gex", "hybrid_gex_us", "hybrid_gex_participant"]
    sets = {"us": GEX_SETS["us"], "participant": GEX_SETS["participant"]}
    if "dealer_next" in m:
        models.append("hybrid_gex_oracle")
        sets["oracle"] = GEX_SETS["oracle"]
    cfg = WalkForwardConfig(models=models, gex_sets=sets, tune=a.tune, seed=a.seed)
    params_all = []
    for name in a.targets:
        ds = build_dataset(m, panel, **TARGETS[name])
        pred, info = run_walkforward(m, ds, cfg)
        pred.to_csv(out / f"forecasts_{name}.csv", index=False)
        sb = scoreboard(pred, models)
        sb.to_csv(out / f"scoreboard_{name}.csv", index=False)
        for g in ("all", "expiry"):
            plots.scoreboard_bars(sb, g, out / "figures" / f"scoreboard_{name}_{g}.png",
                                  f"{name}, {g} days: out-of-sample QLIKE vs mcsGARCH")
        tests = []
        for subset, mask in (("all", np.ones(len(pred), bool)), ("expiry", pred["is_expiry"].to_numpy(bool))):
            L = daily_losses(pred[mask], models)
            for mdl in models:
                if mdl != "mcsgarch":
                    tests.append({"subset": subset, "comparison": f"mcsgarch vs {mdl}",
                                  **diebold_mariano(L["mcsgarch"], L[mdl], alternative="greater")})
            for base, alts in (("lgbm_base", ["lgbm_gex"]), ("hybrid_base", [x for x in models if x.startswith("hybrid_gex")])):
                for alt in alts:
                    tests.append({"subset": subset, "comparison": f"GEX ablation: {base} vs {alt}",
                                  **diebold_mariano(L[base], L[alt], alternative="greater")})
            try:
                mcs = model_confidence_set(L[models], reps=1000)
                mcs.assign(subset=subset).to_csv(out / f"mcs_{name}_{subset}.csv")
            except Exception as exc:  # small samples can make the MCS bootstrap fail
                print(f"MCS skipped for {name}/{subset}: {exc}")
        pd.DataFrame(tests).to_csv(out / f"tests_{name}.csv", index=False)
        if info["importance"]:
            pd.concat(info["importance"], axis=1).mean(axis=1).sort_values(ascending=False).to_csv(
                out / f"feature_importance_{name}.csv", header=["mean_gain_share"])
        params_all.append(pd.DataFrame(info["mcs_params"]).assign(target=name))
        print(f"{name}: {len(pred)} test anchors", flush=True)
    pd.concat(params_all).to_csv(out / "mcsgarch_params.csv", index=False)

    import lightgbm, arch, scipy  # noqa: E401
    json.dump({"source": label, "args": vars(a), "seconds": round(time.time() - t0, 1),
               "config": m.get("config"), "python": sys.version.split()[0],
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
                            "arch": arch.__version__, "scipy": scipy.__version__}},
              open(out / "run_info.json", "w"), indent=2, default=str)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
