"""E4: Can the walk-forward GEX ablation detect a dealer-gamma effect, and how
much does the sign rule matter? SYNTHETIC DATA ONLY.

For each cell (kappa = strength of the true gamma channel, participant_noise =
how badly the participant file measures dealer positions) and each
replication, simulate a market, run the forecast ladder walk-forward, and test
whether adding GEX features lowers out-of-sample QLIKE (one-sided
Diebold-Mariano on daily average losses).

Usage: python experiments/power_study.py --reps 8 --days 500 --kappas 0 0.5 1.0 --noises 0.1 0.5 --out results/power_study.csv
"""

from __future__ import annotations

import argparse
import itertools
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from niftygex.evaluation import daily_losses, diebold_mariano, qlike
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel
from niftygex.forecast import WalkForwardConfig, run_walkforward
from niftygex.simulate import SimConfig, simulate_market

SETS = ["us", "participant", "oracle"]
MODELS = ["mcsgarch", "har", "hybrid_base"] + [f"hybrid_gex_{s}" for s in SETS]


BLOCK = 42


def one_run(args):
    kappa, noise, days, rep = args[:4]
    block = args[4] if len(args) > 4 else BLOCK
    seed = 1000 * rep + int(kappa * 100) + int(noise * 1000) * 7
    t0 = time.time()
    m = simulate_market(SimConfig(n_days=days, kappa=kappa, participant_noise=noise, seed=seed))
    panel = build_gex_panel(m)
    cfg = WalkForwardConfig(models=MODELS, gex_sets={s: GEX_SETS[s] for s in SETS}, seed=seed, block=block)
    rows = []
    for target, kw in (("next30", dict(horizon=30, step=15)), ("last_hour", dict(horizon=60, anchors=[315]))):
        ds = build_dataset(m, panel, **kw)
        pred, _ = run_walkforward(m, ds, cfg)
        for subset, mask in (("all", np.ones(len(pred), bool)), ("expiry", pred["is_expiry"].to_numpy(bool))):
            sub = pred[mask]
            L = daily_losses(sub, MODELS)
            base = qlike(sub["y"], sub["hybrid_base"]).mean()
            rec = {"kappa": kappa, "noise": noise, "days": days, "rep": rep, "seed": seed,
                   "target": target, "subset": subset, "n_days_test": len(L),
                   "qlike_mcsgarch": qlike(sub["y"], sub["mcsgarch"]).mean(),
                   "qlike_har": qlike(sub["y"], sub["har"]).mean(), "qlike_base": base}
            for s in SETS:
                col = f"hybrid_gex_{s}"
                rec[f"qlike_{s}"] = qlike(sub["y"], sub[col]).mean()
                rec[f"gain_pct_{s}"] = 100 * (1 - rec[f"qlike_{s}"] / base)
                dm = diebold_mariano(L["hybrid_base"], L[col], alternative="greater")
                rec[f"dm_stat_{s}"] = dm["stat"]
                rec[f"dm_p_{s}"] = dm["p"]
            rows.append(rec)
    for r in rows:
        r["seconds"] = time.time() - t0
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--noises", type=float, nargs="+", default=[0.1, 0.4])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--out", default="results/power_study.csv")
    a = ap.parse_args()
    jobs = list(itertools.product(a.kappas, a.noises, [a.days], range(a.reps), [a.block]))
    rows = []
    with Pool(a.procs) as pool:
        for k, res in enumerate(pool.imap_unordered(one_run, jobs)):
            rows.extend(res)
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"{k + 1}/{len(jobs)} done", flush=True)


if __name__ == "__main__":
    main()
