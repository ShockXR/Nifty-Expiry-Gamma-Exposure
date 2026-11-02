"""E1: do the engines estimate what they claim, at realistic sizes? SYNTHETIC DATA.

E1a  mcsGARCH parameter recovery at the paper's sample size (64 days) and at
     250 and 500 days, with the paper's Nifty "before" estimates as the truth
     (alpha 0.026, beta 0.971, nu 8.9). Daily variance known (isolates the
     intraday step) and estimated by EGARCH (the full two-step procedure).
E1b  Speed and accuracy of the vectorised implied-vol solver on a full chain.

Usage: python experiments/estimator_validation.py --reps 40
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from niftygex import greeks  # noqa: E402
from niftygex.simulate import SimConfig, simulate_market  # noqa: E402
from niftygex.volatility import McsGarch, daily_egarch_variance  # noqa: E402

OUT = Path("results/estimator_validation")
OUT.mkdir(parents=True, exist_ok=True)
TRUE = {"alpha": 0.026, "beta": 0.971, "nu": 8.9}


def e1a(reps: int):
    rows = []
    for days in (64, 250, 500):
        for r in range(reps):
            m = simulate_market(SimConfig(n_days=days, kappa=0.0, seed=10_000 + r, alpha=TRUE["alpha"],
                                          beta=TRUE["beta"], nu=TRUE["nu"]))
            ex = m["labels"]["is_expiry"].to_numpy()
            for daily in ("known", "egarch"):
                if daily == "known":
                    h = m["h_true"]
                    method = "known"
                else:
                    h, info = daily_egarch_variance(m["R"].sum(axis=1))
                    method = info["method"]
                t0 = time.time()
                mod = McsGarch().fit(m["R"], h, ex)
                s = mod.s / mod.s.sum()
                st = m["s_true"] / m["s_true"].sum()
                rows.append({"days": days, "rep": r, "daily": daily, "daily_method": method,
                             "alpha": mod.params["alpha"], "beta": mod.params["beta"], "nu": mod.params["nu"],
                             "persistence": mod.params["alpha"] + mod.params["beta"],
                             "diurnal_corr": float(np.corrcoef(s, st)[0, 1]),
                             "diurnal_mape": float(np.mean(np.abs(s / st - 1))),
                             "fit_seconds": time.time() - t0})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e1a_mcsgarch_recovery.csv", index=False)
    g = df.groupby(["days", "daily"])
    summ = pd.DataFrame({
        "bias_alpha": g["alpha"].mean() - TRUE["alpha"], "rmse_alpha": g["alpha"].apply(lambda s: np.sqrt(np.mean((s - TRUE["alpha"]) ** 2))),
        "bias_beta": g["beta"].mean() - TRUE["beta"], "rmse_beta": g["beta"].apply(lambda s: np.sqrt(np.mean((s - TRUE["beta"]) ** 2))),
        "median_nu": g["nu"].median(), "true_persistence": TRUE["alpha"] + TRUE["beta"],
        "mean_persistence": g["persistence"].mean(), "diurnal_corr": g["diurnal_corr"].mean(),
        "diurnal_mape": g["diurnal_mape"].mean(), "fit_seconds": g["fit_seconds"].mean(),
        "share_ewma_fallback": g["daily_method"].apply(lambda s: float(np.mean(s == "ewma"))),
        "reps": g.size(),
    })
    summ.to_csv(OUT / "e1a_summary.csv")
    return summ


def e1b():
    rng = np.random.default_rng(0)
    rows = []
    for n in (162, 2_000, 50_000):
        F = 24000.0
        K = F * np.exp(rng.uniform(-0.08, 0.08, n))
        T = rng.choice([1 / 365, 3 / 365, 6 / 365], n)
        sig = rng.uniform(0.08, 0.4, n)
        call = rng.random(n) < 0.5
        px = greeks.price(F, K, T, 0.065, sig, call)
        t0 = time.time()
        iv = greeks.implied_vol(px, F, K, T, 0.065, call)
        dt = time.time() - t0
        ok = np.isfinite(iv)
        intrinsic = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0)) * np.exp(-0.065 * T)
        tick = (px - intrinsic) >= 0.05  # NSE tick size: below it the price carries no vol information
        rows.append({"n_options": n, "seconds": dt, "share_solved": ok.mean(),
                     "share_with_time_value_ge_tick": tick.mean(),
                     "max_abs_error_ge_tick": float(np.max(np.abs(iv[ok & tick] - sig[ok & tick]))),
                     "max_abs_error_all_solved": float(np.max(np.abs(iv[ok] - sig[ok])))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e1b_iv_solver.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    print(e1b().to_string())
    print(e1a(a.reps).to_string())
