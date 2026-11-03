"""P107 POST HOC: how much do a few extreme days move the cross-index correlations and the effective number?

Not in the registered plan (docs/prereg/P107_POOLED_THREE_INDEX.md). Written after the registered results were seen,
because the 30-minute Pearson correlations of the daily loss differentials were far below their Spearman (rank)
counterparts and had wide bootstrap intervals, which points at a few influential days. Descriptive only: no p-value
is computed. Done by an AI agent (Claude) for Shreyan Thanki.

Input: results/p107_pooled_test/daily_losses.csv (written by experiments/p107_pooled_test.py, REAL DATA).
Outputs: results/p107_pooled_test/posthoc_influence.csv and posthoc_extreme_days.csv.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p107_posthoc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p107_pooled_test as pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "p107_pooled_test"
ORDER = ["Nifty", "Bank Nifty", "Fin Nifty"]


def winsorize_columns(D: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    """Clip each column at its own lo and hi quantiles."""
    D = np.asarray(D, float)
    ql, qh = np.quantile(D, lo, axis=0), np.quantile(D, hi, axis=0)
    return np.clip(D, ql, qh)


def extreme_order(D: np.ndarray) -> np.ndarray:
    """Row positions sorted from the most to the least extreme day: largest absolute standardized value in any column."""
    D = np.asarray(D, float)
    z = (D - D.mean(axis=0)) / D.std(axis=0, ddof=1)
    return np.argsort(-np.abs(z).max(axis=1), kind="stable")


def drop_extreme(D: np.ndarray, k: int) -> np.ndarray:
    """D without its k most extreme rows (``extreme_order``), keeping the date order of the rest."""
    if k == 0:
        return np.asarray(D, float)
    drop = set(extreme_order(D)[:k].tolist())
    return np.asarray(D, float)[[i for i in range(len(D)) if i not in drop]]


def dependence_row(D: np.ndarray, lag: int) -> dict:
    R = np.corrcoef(D, rowvar=False)
    e = pt.effective_numbers(R, pt.long_run_cov(D, lag))
    return {"corr_nifty_banknifty": R[0, 1], "corr_nifty_finnifty": R[0, 2], "corr_banknifty_finnifty": R[1, 2],
            "kish": e["kish"], "design_lr": e["design_lr"], "nifty_equiv_equal": e["nifty_equiv_equal"]}


def main():
    daily = pd.read_csv(OUT / "daily_losses.csv")
    daily["diff"] = daily["qlike_mcsgarch"] - daily["qlike_hybrid_base"]
    rows, ext = [], []
    for t in pt.TARGETS:
        w = daily[daily.target == t].pivot(index="date", columns="index", values="diff")[ORDER]
        D = w.to_numpy()
        lag = pt.dk_lag(len(D))     # the registered rule, 5 for 604 dates (kept for the shorter samples too)
        variants = {"all days (registered)": D, "winsorized at each index's 1st and 99th percentile": winsorize_columns(D)}
        for k in (1, 3, 6, 12):
            variants[f"without the {k} most extreme day{'s' if k > 1 else ''}"] = drop_extreme(D, k)
        for name, X in variants.items():
            rows.append({"data": "REAL DATA", "role": "post hoc", "target": t, "benchmark": "mcsgarch", "variant": name,
                         "days": len(X), "lag": lag, **dependence_row(X, lag)})
        # rank (Spearman) version of Kish, for comparison with the Pearson one
        Rs = w.corr(method="spearman").to_numpy()
        rows.append({"data": "REAL DATA", "role": "post hoc", "target": t, "benchmark": "mcsgarch",
                     "variant": "Spearman rank correlations (Kish only)", "days": len(D), "lag": np.nan,
                     "corr_nifty_banknifty": Rs[0, 1], "corr_nifty_finnifty": Rs[0, 2], "corr_banknifty_finnifty": Rs[1, 2],
                     "kish": pt.effective_numbers(Rs)["kish"], "design_lr": np.nan, "nifty_equiv_equal": np.nan})
        order = extreme_order(D)
        z = (D - D.mean(axis=0)) / D.std(axis=0, ddof=1)
        for rank, i in enumerate(order[:6], start=1):
            ext.append({"data": "REAL DATA", "role": "post hoc", "target": t, "rank": rank, "date": w.index[i],
                        **{f"diff_{c}": D[i, j] for j, c in enumerate(ORDER)},
                        "max_abs_z": float(np.abs(z[i]).max())})
    pd.DataFrame(rows).to_csv(OUT / "posthoc_influence.csv", index=False)
    pd.DataFrame(ext).to_csv(OUT / "posthoc_extreme_days.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(rows).drop(columns=["data", "role", "benchmark"]).round(3).to_string())
        print(pd.DataFrame(ext).drop(columns=["data", "role"]).round(4).to_string())


if __name__ == "__main__":
    main()
