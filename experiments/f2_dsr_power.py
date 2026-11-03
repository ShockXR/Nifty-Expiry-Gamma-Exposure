"""POST HOC (REAL DATA inputs): could the registered DSR verdict of P90 ever have come out the other way?

The F2 economics study (P90, docs/F2_ECONOMICS.md) judged the primary rule P by the Deflated Sharpe
Ratio (DSR) over N = 824 trials and found DSR = 1.4e-12. A referee asked what the probability of that
verdict was if the in-sample effect were real. This script answers with the normal approximation that
the DSR itself uses (Bailey and Lopez de Prado 2012, 2014):

* the estimated daily Sharpe ratio over T days is approximately normal around the true Sharpe ratio
  SR with variance (1 - skew SR + (kurt - 1) / 4 SR^2) / (T - 1);
* the rule "passes" when DSR > 0.95, i.e. when the estimated Sharpe ratio exceeds a cutoff c that
  solves PSR(c; SR_0) = 0.95, where SR_0 is the Sharpe ratio the luckiest of N worthless trials would
  show (results/f2_economics/dsr.csv).

For several "true" Sharpe ratios (the test period's own gross and net values, and multiples of the
gross value) it reports the probability of passing, and the true annualised Sharpe ratio needed for a
50% and an 80% chance. Skewness and kurtosis are held at the test period's values for rule P.

    python experiments/f2_dsr_power.py

Writes results/f2_economics/dsr_power_posthoc.csv. Reads only committed result files.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "f2_economics"
ANNUAL = 252


def psr_den(sr: float, skew: float, kurt: float) -> float:
    """Variance factor of the Sharpe ratio estimate (kurt is not excess kurtosis)."""
    return 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2


def psr(sr_hat: float, T: int, skew: float, kurt: float, sr_star: float) -> float:
    den = psr_den(sr_hat, skew, kurt)
    if den <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr_hat - sr_star) * math.sqrt(T - 1) / math.sqrt(den)))


def pass_cutoff(T: int, skew: float, kurt: float, sr0: float, level: float = 0.95) -> float:
    """Smallest estimated daily Sharpe ratio with PSR(.; sr0) >= level."""
    f = lambda c: psr(c, T, skew, kurt, sr0) - level  # noqa: E731
    return float(optimize.brentq(f, sr0, sr0 + 5.0))


def prob_pass(sr_true: float, T: int, skew: float, kurt: float, cutoff: float) -> float:
    """P(estimated Sharpe > cutoff) when the true daily Sharpe ratio is sr_true (normal approximation)."""
    sd = math.sqrt(max(psr_den(sr_true, skew, kurt), 1e-12) / (T - 1))
    return float(stats.norm.sf((cutoff - sr_true) / sd))


def sr_needed(prob: float, T: int, skew: float, kurt: float, cutoff: float) -> float:
    f = lambda s: prob_pass(s, T, skew, kurt, cutoff) - prob  # noqa: E731
    return float(optimize.brentq(f, -1.0, cutoff + 2.0))


def main() -> None:
    dsr = pd.read_csv(RES / "dsr.csv")
    row = dsr[(dsr["rule"] == "P") & (dsr["primary_N"])].iloc[0]
    T, skew, kurt, sr0 = int(row["T"]), float(row["skew"]), float(row["kurt"]), float(row["sr0_daily"])
    pnl = pd.read_csv(RES / "daily_pnl.csv")
    gross, net = pnl["P_gross"], pnl["P_net"]
    sr_gross = float(gross.mean() / gross.std(ddof=1))
    sr_net = float(net.mean() / net.std(ddof=1))
    cutoff = pass_cutoff(T, skew, kurt, sr0)
    scen = [("test period, net of registered costs (1 bp slippage per side)", sr_net),
            ("test period, gross (no costs at all)", sr_gross),
            ("twice the test-period gross Sharpe ratio", 2 * sr_gross),
            ("five times the test-period gross Sharpe ratio", 5 * sr_gross)]
    rows = []
    for name, s in scen:
        rows.append({"scenario": name, "sr_true_daily": s, "sr_true_annual": s * math.sqrt(ANNUAL),
                     "prob_dsr_above_0_95": prob_pass(s, T, skew, kurt, cutoff)})
    for p in (0.5, 0.8):
        s = sr_needed(p, T, skew, kurt, cutoff)
        rows.append({"scenario": f"true Sharpe ratio needed for a {int(p * 100)}% chance of passing",
                     "sr_true_daily": s, "sr_true_annual": s * math.sqrt(ANNUAL), "prob_dsr_above_0_95": p})
    out = pd.DataFrame(rows)
    out["T_days"], out["N_trials"], out["sr0_daily"], out["sr0_annual"] = T, int(row["N"]), sr0, sr0 * math.sqrt(ANNUAL)
    out["pass_cutoff_daily"], out["pass_cutoff_annual"] = cutoff, cutoff * math.sqrt(ANNUAL)
    out["skew"], out["kurt"] = skew, kurt
    # the mean edge per trade (bp) that each daily Sharpe ratio implies, holding the gross volatility and trade frequency fixed
    sd_day = float(gross.std(ddof=1))
    trades_per_day = float((pnl["P_w"] != 0).mean())
    out["edge_bp_per_trade_implied"] = out["sr_true_daily"] * sd_day / trades_per_day
    out.to_csv(RES / "dsr_power_posthoc.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("POST HOC, REAL DATA inputs (results/f2_economics/dsr.csv, daily_pnl.csv)")
        print(out[["scenario", "sr_true_annual", "prob_dsr_above_0_95", "edge_bp_per_trade_implied"]].to_string(index=False))
        print(f"cutoff: annualised Sharpe {cutoff * math.sqrt(ANNUAL):.2f} (SR_0 {sr0 * math.sqrt(ANNUAL):.2f}), T = {T}")


if __name__ == "__main__":
    main()
