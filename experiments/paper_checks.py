"""Checks on the published numbers and methods of Chhimwal, Pandey and Pandey (2025).

E6  Arithmetic checks on published table values (deterministic).
E5  ARMA(1,1) mean on minute returns: are the large, opposite-signed AR and
    MA estimates a sign of near-cancellation? (SYNTHETIC Monte Carlo)

Only short numeric values from the paper's tables are used here, as inputs
to the checks. Usage: python experiments/paper_checks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.connectedness import table_from_paper
from niftygex.volatility import fit_arma11

OUT = Path("results/paper_checks")
OUT.mkdir(parents=True, exist_ok=True)


def arma11_acf1(phi, theta):
    """First autocorrelation of an ARMA(1,1): (1 + phi theta)(phi + theta) / (1 + 2 phi theta + theta^2)."""
    return (1 + phi * theta) * (phi + theta) / (1 + 2 * phi * theta + theta**2)


def e6_table_arithmetic():
    rows = []
    # Table 2 AR(1) and MA(1) estimates with their reported z statistics.
    t2 = {
        "Nifty 50, before": (-0.054778, -0.25788, 0.078178, 0.36870, "**"),
        "Bank Nifty, before": (0.619437, 5.17573, -0.642318, -5.50709, "***"),
        "Nifty 50, after": (-0.627945, -2.7907, 0.638569, 2.8719, "***"),
        "Bank Nifty, after": (0.750097, 13.3985, -0.792505, -15.4559, "***"),
    }
    for k, (phi, zphi, th, zth, stars) in t2.items():
        rows.append({
            "check": "Table 2 ARMA(1,1)", "series": k, "phi": phi, "theta": th,
            "phi_plus_theta": phi + th, "implied_acf1": arma11_acf1(phi, th),
            "z_phi": zphi, "z_theta": zth, "stars_reported": stars,
            "stars_consistent_with_z": (abs(zphi) > 1.96) if stars == "**" else (abs(zphi) > 2.576),
        })
    # Table 2 persistence and the unit-mean restriction omega = 1 - alpha - beta.
    t2v = {
        "Nifty 50, before": (0.002319, 0.026481, 0.971477),
        "Bank Nifty, before": (0.004263, 0.032328, 0.963935),
        "Nifty 50, after": (0.010345, 0.049363, 0.941048),
        "Bank Nifty, after": (0.018692, 0.053195, 0.929680),
    }
    for k, (om, a, b) in t2v.items():
        rows.append({"check": "Table 2 intraday GARCH", "series": k, "alpha_plus_beta": a + b,
                     "omega": om, "one_minus_alpha_minus_beta": 1 - a - b,
                     "implied_unconditional_q2": om / (1 - a - b),
                     "half_life_minutes": np.log(0.5) / np.log(a + b)})
    # Tables 3 and 4: DY scale.
    for name, tbl in {
        "T3 weekly+monthly, before": [[57.29, 42.71], [41.29, 58.71]],
        "T3 last hour monthly, before": [[47.21, 52.79], [52.43, 47.57]],
        "T4 P1 weekly+monthly Nifty expiry, after": [[61.57, 38.43], [37.95, 62.05]],
    }.items():
        c = table_from_paper(tbl)
        rows.append({"check": "DY total scale", "series": name, "paper_scale_total": c.paper_scale_total,
                     "dy_standard_total": c.total})
    # Tables 5 and 6: Mann-Whitney statistics versus their maximum possible values.
    t56 = [
        ("T5 volume all, Nifty", 444392222.5, 23039, 21463),
        ("T5 volume all, Bank Nifty", 538810985.5, 23039, 21463),
        ("T5 volume Wed, Nifty", 19165172.0, 4680, 4320),
        ("T5 volume Wed, Bank Nifty", 21613964.5, 4680, 4320),
        ("T5 volume Thu, Nifty", 16357754.5, 4680, 4680),
        ("T5 volume Thu, Bank Nifty", 21478045.0, 4680, 4680),
        ("T6 vol all, Nifty", 483967708.5, 23038, 21462),
        ("T6 vol all, Bank Nifty", 500287483.5, 23038, 21462),
        ("T6 vol Wed, Nifty", 18633563.0, 4679, 4319),
        ("T6 vol Wed, Bank Nifty", 207455847.5, 4680, 4319),
        ("T6 vol Thu, Nifty", 156272524.5, 4679, 4679),
        ("T6 vol Thu, Bank Nifty", 2174645383.0, 4679, 4679),
    ]
    for name, w, n1, n2 in t56:
        max_u = n1 * n2
        max_w = n1 * n2 + max(n1, n2) * (max(n1, n2) + 1) / 2
        rows.append({"check": "MWW statistic bound", "series": name, "reported": w, "n1": n1, "n2": n2,
                     "max_U": max_u, "max_rank_sum_W": max_w, "within_possible_range": w <= max_w})
    # Sample sizes versus a 375-minute session.
    for name, n, days_guess in (("Before, all minutes", 23039, None), ("After, all minutes", 21239, None),
                                ("Wednesdays before", 4680, None), ("Wednesdays after", 4320, None)):
        rows.append({"check": "observations per day", "series": name, "n": n,
                     "days_if_375_min": n / 375, "days_if_360_min": n / 360})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e6_table_arithmetic.csv", index=False)
    return df


def e5_arma_near_cancellation(n=23039, reps=200, seed=0):
    """Fit ARMA(1,1) to series with tiny true autocorrelation and look at the estimates."""
    rng = np.random.default_rng(seed)
    rows = []
    for true_rho in (0.0, -0.02):
        for r in range(reps):
            # GARCH-t noise so the volatility clustering resembles minute data
            nu, a, b = 7.0, 0.03, 0.95
            z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
            e = np.empty(n)
            q = 1.0
            for i in range(n):
                e[i] = np.sqrt(q) * z[i]
                q = (1 - a - b) + a * e[i] ** 2 + b * q
            # MA(1) with first autocorrelation true_rho: theta / (1 + theta^2) = rho
            th = (1 - np.sqrt(1 - 4 * true_rho**2)) / (2 * true_rho) if true_rho != 0 else 0.0
            x = e.copy()
            x[1:] += th * e[:-1]
            p, _ = fit_arma11(x * 1e-4)
            rows.append({"true_acf1": true_rho, "rep": r, "phi": p["phi"], "theta": p["theta"],
                         "phi_plus_theta": p["phi"] + p["theta"], "implied_acf1": arma11_acf1(p["phi"], p["theta"])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e5_arma_near_cancellation.csv", index=False)
    summ = df.groupby("true_acf1").agg(
        share_abs_phi_gt_0_5=("phi", lambda s: float(np.mean(np.abs(s) > 0.5))),
        median_abs_phi=("phi", lambda s: float(np.median(np.abs(s)))),
        median_abs_phi_plus_theta=("phi_plus_theta", lambda s: float(np.median(np.abs(s)))),
        corr_phi_theta=("phi", lambda s: float(np.corrcoef(s, df.loc[s.index, "theta"])[0, 1])),
        mean_implied_acf1=("implied_acf1", "mean"),
        sd_implied_acf1=("implied_acf1", "std"),
    )
    summ.to_csv(OUT / "e5_summary.csv")
    return summ


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(e6_table_arithmetic().to_string())
    print(e5_arma_near_cancellation())
