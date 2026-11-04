"""Referee response, forecasting: Giacomini and White (2006) tests (referee priority 5).

POST HOC: done after the referee report, in response to it. REAL DATA: reads the
committed forecasts of the final run (results/pipeline_nse_v2/forecasts_*.csv) and
the information-matched mcsGARCH forecasts built by experiments/referee_forecast.py
(results/referee_forecast/mcsgarchx_forecasts_*.csv). Nothing is refitted.

For hybrid_base against mcsGARCH and against the best information-matched mcsGARCH
of each horizon (the one with the lowest mean QLIKE on all days in
results/referee_forecast/fair_benchmarks.csv), at the three horizons, on the daily
average QLIKE difference d_t = L_benchmark,t - L_hybrid,t (positive: hybrid better):

* unconditional test (instrument: a constant), Newey-West variance;
* conditional test C1 (a constant and d_{t-1});
* conditional test C2 (a constant, d_{t-1}, an expiry-day dummy and the expiry-regime
  dummies), both with Giacomini and White's one-step variance (1/n) sum Z_t Z_t',
  which is valid because Z_t = h_{t-1} d_t is a martingale difference under their
  null, and with a Newey-West variance as a check;
* slope tests: does the advantage vary with the conditions? OLS of d_t on the
  instruments, Wald tests that the slopes are zero (all of them, the lag alone, the
  expiry dummy alone, the regime dummies), Newey-West covariance;
* the mean advantage on expiry and other days and in each regime (descriptive).

Every instrument is known before day t: d_{t-1} at the close of day t-1, the expiry
calendar and the regime in advance. Holm-adjusted p-values are given over the family
of the main tests; the Newey-West versions of C1 and C2 are a robustness check.

    PYTHONPATH=src python experiments/referee_forecast_gw.py > results/referee_forecast_gw_stdout.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from niftygex.evaluation import REGIMES, daily_losses, holm

TARGETS = ["next15", "next30", "last_hour"]
BENCH = "mcsgarch"
HEADLINE = "hybrid_base"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def nw_lag(n: int) -> int:
    """The bandwidth rule of the pipeline's DM test (niftygex.evaluation.diebold_mariano)."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def newey_west(Z: np.ndarray, lag: int | None = None, center: bool = True) -> np.ndarray:
    """Newey-West (Bartlett kernel) long-run covariance of the rows of Z (n x q), divided by n."""
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n = len(Z)
    if lag is None:
        lag = nw_lag(n)
    Zc = Z - Z.mean(axis=0) if center else Z
    S = Zc.T @ Zc / n
    for k in range(1, lag + 1):
        G = Zc[k:].T @ Zc[:-k] / n
        S += (1 - k / (lag + 1)) * (G + G.T)
    return S


def gw_test(d: np.ndarray, H: np.ndarray, variance: str = "gw", lag: int | None = None) -> dict:
    """Giacomini and White (2006) test of H0: E[d_t | F_{t-1}] = 0, with instruments H (n x q)
    whose row t is known at t-1. Statistic n Zbar' Omega^-1 Zbar, chi-square with q df,
    Z_t = H_t d_t. variance "gw": Omega = (1/n) sum Z_t Z_t' (their one-step estimator);
    "nw": Newey-West, centred. With H a column of ones and "nw" this is the squared DM
    statistic without the small-sample factor."""
    d = np.asarray(d, float)
    H = np.asarray(H, float)
    if H.ndim == 1:
        H = H[:, None]
    Z = H * d[:, None]
    n, q = Z.shape
    zbar = Z.mean(axis=0)
    om = Z.T @ Z / n if variance == "gw" else newey_west(Z, lag, center=True)
    stat = float(n * zbar @ np.linalg.solve(om, zbar))
    return {"stat": stat, "df": q, "p": float(chi2.sf(stat, q)), "n": n}


def slope_wald(d: np.ndarray, H: np.ndarray, test_cols: list[int], lag: int | None = None) -> dict:
    """OLS of d on H (first column a constant); Wald test that the coefficients in test_cols
    are zero, with the Newey-West (HAC) covariance of the coefficients."""
    d = np.asarray(d, float)
    H = np.asarray(H, float)
    n, q = H.shape
    A = np.linalg.inv(H.T @ H / n)
    b = A @ (H.T @ d / n)
    u = d - H @ b
    S = newey_west(H * u[:, None], lag, center=False)
    V = A @ S @ A / n
    R = np.zeros((len(test_cols), q))
    for i, c in enumerate(test_cols):
        R[i, c] = 1.0
    rb = R @ b
    stat = float(rb @ np.linalg.solve(R @ V @ R.T, rb))
    return {"stat": stat, "df": len(test_cols), "p": float(chi2.sf(stat, len(test_cols))), "coef": b,
            "se": np.sqrt(np.diag(V)), "n": n}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def day_table(run: Path, out: Path, target: str, bench_matched: str) -> pd.DataFrame:
    """Daily average QLIKE of mcsGARCH, the matched benchmark and the hybrid, with the
    expiry flag and the regime of each test day."""
    f = pd.read_csv(run / f"forecasts_{target}.csv", parse_dates=["date"])
    if bench_matched not in f:
        x = pd.read_csv(out / f"mcsgarchx_forecasts_{target}.csv", parse_dates=["date"])
        assert (f["t"].to_numpy() == x["t"].to_numpy()).all() and (f["minute"].to_numpy() == x["minute"].to_numpy()).all()
        f[bench_matched] = x[bench_matched].to_numpy()
    L = daily_losses(f, [BENCH, bench_matched, HEADLINE], "qlike")
    cal = f.groupby("date").agg(is_expiry=("is_expiry", "first"), regime=("regime", "first"), t=("t", "first"))
    return L.join(cal)


def instruments(day: pd.DataFrame, d: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Rows 1..n-1: constant, d_{t-1}, expiry dummy, regime dummies (base: the first regime present)."""
    e = day["is_expiry"].to_numpy(bool).astype(float)
    reg = day["regime"].to_numpy(int)
    present = sorted(set(reg))
    cols = [np.ones(len(d) - 1), d[:-1], e[1:]]
    names = ["const", "d_lag", "expiry"]
    for k in present[1:]:
        cols.append((reg[1:] == k).astype(float))
        names.append(f"regime_{REGIMES[k]}")
    return np.column_stack(cols), names, d[1:]


def run_pair(day: pd.DataFrame, bench: str, label: str, target: str) -> tuple[list[dict], list[dict]]:
    lb = day[bench].to_numpy(float)
    lh = day[HEADLINE].to_numpy(float)
    scale = 100.0 / lb.mean()        # d in points of the benchmark's mean daily QLIKE
    d_all = (lb - lh) * scale
    H, names, d = instruments(day, d_all)
    n = len(d)
    rows = []

    def add(test, res, family, h_names):
        rows.append({"target": target, "benchmark": label, "benchmark_model": bench, "test": test,
                     "instruments": " + ".join(h_names), "family": family, "n": res["n"], "df": res["df"],
                     "stat": res["stat"], "p": res["p"]})

    add("unconditional", gw_test(d, H[:, :1], "nw"), "main", names[:1])
    add("conditional C1", gw_test(d, H[:, :2], "gw"), "main", names[:2])
    add("conditional C2", gw_test(d, H, "gw"), "main", names)
    add("conditional C1 (Newey-West)", gw_test(d, H[:, :2], "nw"), "robustness", names[:2])
    add("conditional C2 (Newey-West)", gw_test(d, H, "nw"), "robustness", names)
    s1 = slope_wald(d, H[:, :2], [1])
    add("slope C1: lag", s1, "main", names[:2])
    s2 = slope_wald(d, H, list(range(1, H.shape[1])))
    add("slope C2: all slopes", s2, "main", names)
    add("slope C2: expiry", slope_wald(d, H, [2]), "main", names)
    reg_cols = [i for i, nm in enumerate(names) if nm.startswith("regime_")]
    add("slope C2: regimes", slope_wald(d, H, reg_cols), "main", names)
    for r in rows:
        r["mean_d_points"] = float(d.mean())
    coefs = [{"target": target, "benchmark": label, "benchmark_model": bench, "regression": "C2", "term": nm,
              "coef_points": float(c), "se_points": float(s)} for nm, c, s in zip(names, s2["coef"], s2["se"])]
    coefs += [{"target": target, "benchmark": label, "benchmark_model": bench, "regression": "C1", "term": nm,
               "coef_points": float(c), "se_points": float(s)} for nm, c, s in zip(names[:2], s1["coef"], s1["se"])]
    return rows, coefs


def ewma_fallback_starts(run: Path) -> list[int]:
    """First days of the refits in which the paper's daily EGARCH fell back to an EWMA."""
    p = run / "mcsgarch_params.csv"
    if not p.exists():
        return []
    mp = pd.read_csv(p)
    mp = mp[(mp["target"] == "next15") & (mp["refit_every"] == 21) & (mp["daily_component"] == "egarch (paper)")]
    return sorted(mp.loc[mp["daily_method"] == "ewma", "start"].astype(int))


def group_margins(day: pd.DataFrame, bench: str, label: str, target: str, ewma_starts=()) -> list[dict]:
    """Hybrid's margin in % of the benchmark's mean QLIKE within each group (descriptive)."""
    e = day["is_expiry"].to_numpy(bool)
    reg = day["regime"].to_numpy(int)
    groups = {"all": np.ones(len(day), bool), "expiry": e, "non_expiry": ~e}
    for k in sorted(set(reg)):
        groups[REGIMES[k]] = reg == k
        groups[f"{REGIMES[k]}_expiry"] = (reg == k) & e
        groups[f"{REGIMES[k]}_non_expiry"] = (reg == k) & ~e
    if len(ewma_starts):
        tt = day["t"].to_numpy()
        ew = np.any([(tt >= s) & (tt < s + 21) for s in ewma_starts], axis=0)
        groups[f"{REGIMES[0]}_excl_ewma_fallback"] = (reg == 0) & ~ew
    out = []
    for g, m in groups.items():
        lb = day.loc[m, bench].mean()
        lh = day.loc[m, HEADLINE].mean()
        out.append({"target": target, "benchmark": label, "benchmark_model": bench, "group": g, "days": int(m.sum()),
                    "hybrid_pct_vs_benchmark": 100 * (lh / lb - 1),
                    "first_date": str(day.index[m].min().date()), "last_date": str(day.index[m].max().date())})
    return out


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/pipeline_nse_v2")
    ap.add_argument("--out", default="results/referee_forecast")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    fb = pd.read_csv(out / "fair_benchmarks.csv")
    best = fb[(fb.subset == "all") & (fb.benchmark == "best_matched_mcsgarch")].set_index("target")["benchmark_model"]

    print("REAL DATA. POST HOC (referee priority 5). Giacomini and White (2006) tests of hybrid_base against mcsGARCH "
          "and the best information-matched mcsGARCH, daily average QLIKE.")
    print("d_t = benchmark loss - hybrid loss, in points of the benchmark's mean daily QLIKE (positive: hybrid better).")
    print("Instruments (all known before day t): C1 = constant + d_{t-1}; C2 = C1 + expiry-day dummy + regime dummies "
          "(base: multi_weekly; regimes are consecutive periods: multi_weekly to 2024-11-19, single_weekly_thu to "
          "2025-08-29, single_weekly_tue from 2025-09-01).")
    rows, coefs, margins = [], [], []
    ewma_starts = ewma_fallback_starts(run)
    for target in TARGETS:
        day = day_table(run, out, target, best[target])
        for bench, label in ((BENCH, "mcsgarch"), (best[target], "best_matched_mcsgarch")):
            r, c = run_pair(day, bench, label, target)
            rows += r
            coefs += c
            margins += group_margins(day, bench, label, target, ewma_starts)
    res = pd.DataFrame(rows)
    main_mask = res["family"] == "main"
    res["p_holm_main_family"] = np.nan
    res.loc[main_mask, "p_holm_main_family"] = holm(res.loc[main_mask, "p"].to_numpy())
    res.to_csv(out / "gw_tests.csv", index=False)
    cf = pd.DataFrame(coefs)
    cf.to_csv(out / "gw_regression_coefficients.csv", index=False)
    gm = pd.DataFrame(margins)
    gm.to_csv(out / "gw_group_margins.csv", index=False)

    print(f"\np-values: {int(main_mask.sum())} in the main family (Holm-adjusted over all of them), "
          f"{int((~main_mask).sum())} Newey-West robustness versions; {len(res)} in all.")
    for target in TARGETS:
        for label in ("mcsgarch", "best_matched_mcsgarch"):
            s = res[(res.target == target) & (res.benchmark == label)]
            bm = s["benchmark_model"].iloc[0]
            print(f"\n{target} vs {bm} ({label}); {s['n'].iloc[0]} days; mean d {s['mean_d_points'].iloc[0]:+.2f} points")
            for r in s.itertuples():
                holm_txt = f"  Holm {fmt_p(r.p_holm_main_family)}" if r.family == "main" else ""
                print(f"  {r.test:30s} chi2({r.df}) = {r.stat:7.2f}  p {fmt_p(r.p):>6s}{holm_txt}")
            c = cf[(cf.target == target) & (cf.benchmark == label) & (cf.regression == "C2")]
            print("  C2 coefficients (points, HAC se): " + ", ".join(
                f"{t} {v:+.2f} ({se:.2f})" for t, v, se in zip(c.term, c.coef_points, c.se_points)))
            g = gm[(gm.target == target) & (gm.benchmark == label)].set_index("group")
            print("  hybrid vs benchmark by group (% of the group's benchmark QLIKE): " + ", ".join(
                f"{k} {g.loc[k, 'hybrid_pct_vs_benchmark']:+.1f}% ({g.loc[k, 'days']})"
                for k in ["expiry", "non_expiry"] + [x for x in REGIMES if x in g.index]
                + [x for x in g.index if x.endswith("_excl_ewma_fallback")]))
    print(f"\nwrote {out / 'gw_tests.csv'}, {out / 'gw_regression_coefficients.csv'}, {out / 'gw_group_margins.csv'}")


if __name__ == "__main__":
    main()
