"""E2 and E3: how reliable are the paper's tests at its sample sizes? SYNTHETIC DATA.

E2  Size and power of before/after tests on per-minute volatility and volume.
    Under a "no change" null, both periods come from the same process. The
    minute-level Mann-Whitney test (the paper's approach) is compared with
    tests that respect within-day dependence.
E3  Sampling uncertainty and setting-sensitivity of the Diebold-Yilmaz total
    connectedness at sample sizes like the paper's sub-samples (for example
    one trading hour on about four monthly expiry days).

Usage: python experiments/inference_checks.py [--reps 400]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex import connectedness as cn
from niftygex import inference, realized

OUT = Path("results/inference_checks")
OUT.mkdir(parents=True, exist_ok=True)
M = 375


def fast_days(rng, D, scale=1.0, rho2=None, k=8):
    """Vectorised minute bars for D days: returns RS variance and volume (D x M).

    Daily log-variance follows a stationary AR(1) (phi 0.9), the diurnal curve
    is U-shaped and a GARCH(1,1)-t runs within each day. ``scale`` multiplies
    the variance (used for the power runs). With ``rho2`` a second, correlated
    index is generated as well (for E3).
    """
    phi, sd = 0.9, 0.35
    lv = np.empty(D)
    lv[0] = rng.normal(0, sd)
    for t in range(1, D):
        lv[t] = phi * lv[t - 1] + rng.normal(0, sd * np.sqrt(1 - phi**2))
    i = np.arange(M)
    s = 1 + 2.5 * np.exp(-i / 15) + 0.7 * np.exp(-(M - 1 - i) / 25)
    s = s / s.mean()
    base = (0.11**2 / 252 / M) * scale
    a, b, nu = 0.03, 0.95, 7.0
    q = np.ones(D)
    var = np.empty((D, M))
    zt = rng.standard_t(nu, (D, M)) * np.sqrt((nu - 2) / nu)
    for j in range(M):
        var[:, j] = base * np.exp(lv) * s[j] * q
        q = (1 - a - b) + a * q * zt[:, j] ** 2 + b * q
    sub1 = rng.standard_normal((D, M, k))
    out = {}
    paths = [sub1]
    if rho2 is not None:
        paths.append(rho2 * sub1 + np.sqrt(1 - rho2**2) * rng.standard_normal((D, M, k)))
    for n, sub in enumerate(paths):
        steps = sub * np.sqrt(var[..., None] / k)
        path = np.cumsum(steps, axis=2)
        c = path[..., -1]
        h = np.maximum(path.max(axis=2), 0)
        l = np.minimum(path.min(axis=2), 0)
        rs = realized.rogers_satchell(np.ones_like(c), np.exp(h), np.exp(l), np.exp(c))
        out[f"rs{n}"] = rs
        out[f"ret{n}"] = c
    day_factor = rng.normal(0, 0.3, D)[:, None]
    out["volume"] = np.exp(12 + day_factor + 0.5 * np.log(var / base) + rng.normal(0, 0.4, (D, M)))
    return out


def e2(reps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    designs = {"all days (64 vs 59)": (64, 59), "one weekday (13 vs 12)": (13, 12)}
    for dname, (d1, d2) in designs.items():
        for effect in (1.0, 1.10, 1.25):
            for r in range(reps):
                b = fast_days(rng, d1)
                a = fast_days(rng, d2, scale=effect)
                for var in ("rs0", "volume"):
                    x, y = b[var], a[var]
                    if var == "volume" and effect != 1.0:
                        y = y * effect ** 0.5
                    res = {
                        "minute_mwu": inference.minute_level_mwu(x, y)["p"],
                        "day_mwu": inference.day_level_mwu(x, y)["p"],
                        "hac_log": inference.hac_mean_difference(np.maximum(x, 1e-20), np.maximum(y, 1e-20))["p"],
                    }
                    if var == "rs0" and r < min(reps, 200):
                        res["day_bootstrap"] = inference.day_cluster_bootstrap(x, y, n_boot=300, seed=r)["p"]
                    for test, p in res.items():
                        rows.append({"design": dname, "variance_ratio": effect, "variable": var, "rep": r, "test": test, "p": p})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e2_tests_raw.csv", index=False)
    summ = (df.assign(reject=df["p"] < 0.05)
              .groupby(["design", "variable", "variance_ratio", "test"])["reject"]
              .agg(["mean", "count"]).rename(columns={"mean": "rejection_rate_5pct", "count": "reps"}))
    summ["mc_se"] = np.sqrt(summ["rejection_rate_5pct"] * (1 - summ["rejection_rate_5pct"]) / summ["reps"])
    summ.to_csv(OUT / "e2_size_power.csv")
    return summ


def e3(seed: int = 1, n_boot: int = 300):
    rng = np.random.default_rng(seed)
    rows = []
    # Volatility proxies for two correlated indices: log of per-minute RS variance.
    for label, days, minutes_used in (("one hour on 4 expiry days", 4, 60),
                                      ("full session on 13 expiry days", 13, M),
                                      ("full session on 64 days", 64, M)):
        sims = fast_days(rng, days, rho2=0.85)
        lo = M - minutes_used
        Y = np.column_stack([np.log(np.maximum(sims[f"rs{n}"][:, lo:], 1e-20)).ravel() for n in (0, 1)])
        segs = np.repeat(np.arange(days), minutes_used)
        for p in (1, 2, 5, 10):
            for H in (5, 10, 20, 50, 100):
                c = cn.connectedness(Y, p=p, H=H)
                rows.append({"sample": label, "n_obs": len(Y), "p": p, "H": H,
                             "total_dy": c.total, "total_paper_scale": c.paper_scale_total})
        boot = cn.block_bootstrap_total(Y, p=2, H=10, n_boot=n_boot, block=20, segments=segs, seed=seed)
        rows.append({"sample": label, "n_obs": len(Y), "p": 2, "H": 10, "bootstrap": True,
                     "total_dy": cn.connectedness(Y, 2, 10).total,
                     "boot_ci_low": float(np.percentile(boot, 2.5)), "boot_ci_high": float(np.percentile(boot, 97.5)),
                     "boot_sd": float(boot.std())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e3_dy_sensitivity.csv", index=False)
    grid = df[df.get("bootstrap").isna()] if "bootstrap" in df else df
    summ = grid.groupby("sample")["total_dy"].agg(["min", "max", "median"])
    ci = df[df.get("bootstrap") == True][["sample", "total_dy", "boot_ci_low", "boot_ci_high", "boot_sd"]]  # noqa: E712
    summ = summ.join(ci.set_index("sample"), rsuffix="_p2H10")
    summ.to_csv(OUT / "e3_summary.csv")
    return summ


def cross_share_example():
    """Can the off-diagonal share exceed 50% (paper Table 3 last hour: 52.79)?"""
    rows = []
    for rho in (0.5, 0.85, 0.95):
        A = np.zeros((1, 2, 2))
        Sigma = np.array([[1, rho], [rho, 1.0]])
        rows.append({"case": f"no dynamics, rho={rho}", "cross_share_pct": 100 * cn.gfevd(A, Sigma, 10)[0, 1]})
    for lead in (0.3, 0.6, 0.9):
        A = np.array([[[0.2, lead], [0.0, 0.5]]])
        Sigma = np.array([[1, 0.85], [0.85, 1.0]])
        rows.append({"case": f"series 2 leads series 1 by {lead}, rho=0.85", "cross_share_pct": 100 * cn.gfevd(A, Sigma, 10)[0, 1]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e3_cross_share_example.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    print(cross_share_example().to_string())
    print(e3().to_string())
    print(e2(a.reps).to_string())
