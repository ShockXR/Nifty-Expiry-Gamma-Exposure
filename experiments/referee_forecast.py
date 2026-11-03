"""Referee response, forecasting part (REFEREE_REPORT.md M4, M5 and minor comment 15).

POST HOC: everything here was done after the referee report, in response to it.
It reads the committed forecasts of the final run (results/pipeline_nse_v2/) and
does not change the pipeline or its results. REAL DATA (NSE files and validated
1-minute Nifty bars), except the tests in tests/test_referee_forecast.py, which
use synthetic data only.

Sections (all run by default, in this order):

1. spa        Hansen (2005) test for Superior Predictive Ability and White (2000)
              Reality Check of "the best of the 31 challengers beats mcsGARCH" on
              daily average QLIKE, with the stationary bootstrap of Politis and
              Romano (1994); per horizon on all days, expiry days and the days
              before the closing auction. Also single-step and step-down max-t
              adjusted p-values for every challenger (Romano and Wolf 2005).
2. mcsgarchx  mcsGARCH-X: the paper's mcsGARCH whose daily component is an
              AR(1)-EGARCH(1,1)-X with log India VIX in the log-variance equation
              (and a version that also has the straddle-implied variance), plus
              mcsGARCH with VIX as its daily level. Same 21-day walk-forward as the
              pipeline. mcsGARCH and mcsgarch_iv are rebuilt first and checked
              against the committed forecasts. Cached in mcsgarchx_forecasts_*.csv.
3. bench      the hybrid against the information-matched benchmarks (mcsgarch_iv,
              linear_base and the mcsGARCH-X family), DM tests.
4. noise      estimator noise: seed noise (seed_noise_next30.csv), the spread across
              hybrid variants that differ only by feature blocks the ablation found
              uninformative, the DM p-values with that noise added, the seed noise
              that would be needed to lose significance, and the cost of the rerun
              that a full bootstrap over seeds needs.
5. mcs        Model Confidence Set: what it keeps, the top-ranked model, the rank of
              hybrid_base; recomputed with 10,000 replications and two block lengths.
6. holdout    development window and hold-out: plain LightGBM, tuned LightGBM and
              the hybrid, with DM tests of LightGBM against the hybrid.
7. mse        the ranking under MSE, from the committed scoreboards and tests.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/referee_forecast.py \
        > results/referee_forecast_stdout.txt

Writes results/referee_forecast/*.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2
from scipy.stats import t as student_t

from niftygex.evaluation import daily_losses, diebold_mariano, model_confidence_set, subgroups

TARGETS = ["next15", "next30", "last_hour"]
SUBSETS = ["all", "expiry", "pre_auction"]
HOLDOUT_START = "2025-07-01"
AUCTION_START = "2026-08-03"
BENCH = "mcsgarch"
HEADLINE = "hybrid_base"
SQRT2_OV_PI = np.sqrt(2.0 / np.pi)


# ---------------------------------------------------------------------------
# Stationary bootstrap, SPA, Reality Check, max-t adjusted p-values
# ---------------------------------------------------------------------------


def stationary_bootstrap_indices(n: int, reps: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis and Romano (1994) stationary bootstrap: (reps x n) indices into 0..n-1.

    Each resample starts at a uniform index; at every later step it starts a new
    block at a uniform index with probability q = 1 / mean_block and otherwise
    continues with the next observation (wrapping round at the end), so block
    lengths are geometric with mean ``mean_block``.
    """
    q = 1.0 / float(mean_block)
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    jump = rng.random((reps, n)) < q
    fresh = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        cont = idx[:, t - 1] + 1
        cont[cont == n] = 0
        idx[:, t] = np.where(jump[:, t], fresh[:, t], cont)
    return idx


def bootstrap_means(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Column means of x (n x m) in every resample: (reps x m), via counts (no copy of the data)."""
    reps, n = idx.shape
    counts = np.bincount((idx + n * np.arange(reps)[:, None]).ravel(), minlength=reps * n).reshape(reps, n)
    return counts @ np.asarray(x, float).reshape(n, -1) / n


def sb_kernel_variance(d: np.ndarray, mean_block: float) -> np.ndarray:
    """Long-run variance of sqrt(n) * mean(d) per column, with the stationary-bootstrap kernel.

    omega^2 = gamma_0 + 2 sum_{i=1}^{n-1} kappa(n, i) gamma_i, with
    kappa(n, i) = ((n - i)/n) (1 - q)^i + (i/n) (1 - q)^(n - i), q = 1/mean_block
    (Politis and Romano 1994; the estimator used in Hansen 2005).
    """
    d = np.asarray(d, float)
    if d.ndim == 1:
        d = d[:, None]
    n = d.shape[0]
    q = 1.0 / float(mean_block)
    dc = d - d.mean(axis=0)
    v = (dc * dc).sum(axis=0) / n
    for i in range(1, n):
        kappa = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
        if kappa < 1e-14:
            continue
        v = v + 2 * kappa * (dc[:-i] * dc[i:]).sum(axis=0) / n
    return v


def stepdown_maxt_pvalues(t_obs: np.ndarray, t_boot: np.ndarray) -> np.ndarray:
    """Step-down max-t adjusted p-values (Romano and Wolf 2005), monotone in the ordering.

    ``t_boot`` (reps x m) holds centred bootstrap t statistics. The model with the
    largest observed t is compared with the maximum over all models, the next one
    with the maximum over the models not yet passed, and so on.
    """
    t_obs = np.asarray(t_obs, float)
    order = np.argsort(-t_obs)
    out = np.empty(len(t_obs))
    prev = 0.0
    for r, k in enumerate(order):
        pk = float((t_boot[:, order[r:]].max(axis=1) >= t_obs[k]).mean())
        prev = max(prev, pk)
        out[k] = prev
    return out


def spa_test(d: np.ndarray, reps: int = 10000, mean_block: float = 10.0, seed: int = 0,
             idx: np.ndarray | None = None) -> dict:
    """Hansen's SPA test and White's Reality Check for H0: no model beats the benchmark.

    ``d`` is (n x m): benchmark loss minus model loss for each of m models, so a
    positive mean means the model is better. Returns

    * ``p_spa_l``, ``p_spa_c``, ``p_spa_u``: Hansen's (2005) lower, consistent and
      upper p-values for the studentized statistic T = max(0, max_k sqrt(n) dbar_k / omega_k);
      the consistent one is the recommended test;
    * ``p_rc``: White's (2000) Reality Check (statistic max_k sqrt(n) dbar_k, not
      studentized, every model recentred);
    * ``t``: each model's studentized statistic; ``p_single`` and ``p_stepdown``:
      max-t adjusted p-values (every model recentred, so the family-wise error rate
      over the m comparisons is controlled).
    """
    d = np.asarray(d, float)
    n, m = d.shape
    dbar = d.mean(axis=0)
    omega = np.sqrt(np.maximum(sb_kernel_variance(d, mean_block), 1e-300))
    tstat = np.sqrt(n) * dbar / omega
    T_spa = max(float(tstat.max()), 0.0)
    if idx is None:
        idx = stationary_bootstrap_indices(n, reps, mean_block, np.random.default_rng(seed))
    dstar = bootstrap_means(d, idx)
    thr = np.sqrt(2 * np.log(np.log(n)))
    centres = {"l": np.maximum(dbar, 0.0), "c": np.where(tstat >= -thr, dbar, 0.0), "u": dbar}
    out = {"n": n, "m": m, "T_spa": T_spa, "best": int(np.argmax(tstat)), "best_by_mean": int(np.argmax(dbar))}
    for k, c in centres.items():
        Tstar = np.maximum((np.sqrt(n) * (dstar - c) / omega).max(axis=1), 0.0)
        out[f"p_spa_{k}"] = float((Tstar > T_spa).mean())
    V = np.sqrt(n) * dbar.max()
    Vstar = (np.sqrt(n) * (dstar - dbar)).max(axis=1)
    out["rc_stat"] = float(V)
    out["p_rc"] = float((Vstar > V).mean())
    Z = np.sqrt(n) * (dstar - dbar) / omega
    zmax = Z.max(axis=1)
    out["t"] = tstat
    out["dbar"] = dbar
    out["p_single"] = np.array([(zmax >= tk).mean() for tk in tstat])
    out["p_stepdown"] = stepdown_maxt_pvalues(tstat, Z)
    out["n_excluded_consistent"] = int((tstat < -thr).sum())
    return out


# ---------------------------------------------------------------------------
# DM with estimator noise added
# ---------------------------------------------------------------------------


def dm_with_extra_noise(dbar: float, se: float, n: int, extra_sd: float) -> tuple[float, float]:
    """One-sided DM statistic and p-value when an independent noise term with standard
    deviation ``extra_sd`` (same units as dbar) is added to the sampling standard error ``se``.

    Uses the same Student t(n - 1) reference as niftygex.evaluation.diebold_mariano.
    """
    tot = np.sqrt(se**2 + extra_sd**2)
    stat = dbar / tot
    return float(stat), float(student_t.sf(stat, max(n - 1, 1)))


def breakeven_noise_sd(dbar: float, se: float, n: int, alpha: float) -> float:
    """Extra noise SD at which the one-sided p-value of dm_with_extra_noise reaches ``alpha``."""
    tc = student_t.isf(alpha, max(n - 1, 1))
    if dbar <= 0:
        return 0.0
    v = (dbar / tc) ** 2 - se**2
    return float(np.sqrt(v)) if v > 0 else 0.0


def sd_upper_bound(sd: float, df: int, level: float = 0.95) -> float:
    """One-sided upper confidence bound for a standard deviation estimated with ``df`` degrees of freedom."""
    return float(sd * np.sqrt(df / chi2.ppf(1 - level, df)))


# ---------------------------------------------------------------------------
# AR(1)-EGARCH(1,1)-X with Student t errors (the daily component of mcsGARCH-X)
# ---------------------------------------------------------------------------


def arch_backcast(resid: np.ndarray) -> float:
    """The arch package's backcast for EGARCH: log of an EWMA (0.94) mean of the first 75 squared residuals."""
    tau = min(75, len(resid))
    w = 0.94 ** np.arange(tau)
    w = w / w.sum()
    return float(np.log(np.sum(resid[:tau] ** 2 * w)))


def ols_ar1_backcast(y: np.ndarray) -> float:
    """arch's backcast as arch computes it: from the residuals of an OLS AR(1) fit of y
    (arch's starting values), not from the residuals at the estimated parameters."""
    y = np.asarray(y, float)
    Z = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    b, *_ = np.linalg.lstsq(Z, y[1:], rcond=None)
    return arch_backcast(y[1:] - Z @ b)


def egarchx_filter(params: np.ndarray, y: np.ndarray, X: np.ndarray | None, backcast: float | None = None):
    """Conditional variances of an AR(1)-EGARCH(1,1)-X for the whole series y.

    params = (mu, phi, omega, alpha, gamma, beta, delta_1..delta_k, nu). For t >= 1,
    with e_t = y_t - mu - phi y_{t-1} and z_t = e_t / sigma_t,

        log sigma2_t = omega + alpha (|z_{t-1}| - sqrt(2/pi)) + gamma z_{t-1}
                       + beta log sigma2_{t-1} + delta' X_t,

    the arch package's EGARCH (Nelson 1991) parametrisation plus exogenous terms.
    X_t must be known before day t opens (the caller lags it). The first modelled
    day uses the backcast for log sigma2_0 and no shock term, as arch does; by
    default the backcast is arch's (ols_ar1_backcast of y).
    Returns (sigma2, resid, lnsigma2) for days 1..n-1.
    """
    y = np.asarray(y, float)
    k = 0 if X is None else X.shape[1]
    mu, phi, omega, alpha, gamma, beta = (float(v) for v in params[:6])
    delta = np.asarray(params[6:6 + k], float)
    e = y[1:] - mu - phi * y[:-1]
    xb = (X[1:] @ delta) if k else np.zeros(len(e))
    bc = ols_ar1_backcast(y) if backcast is None else float(backcast)
    el, xl = e.tolist(), xb.tolist()
    lns = [0.0] * len(el)
    prev_ln = bc
    v = omega + beta * prev_ln + xl[0]          # first modelled day: no lagged shock (as in arch)
    for t in range(len(el)):
        if t > 0:
            v = omega + beta * prev_ln + xl[t] + alpha * (abs(prev_z) - SQRT2_OV_PI) + gamma * prev_z
        if v > 700.0:
            v = 700.0
        elif v < -700.0:
            v = -700.0
        lns[t] = v
        prev_ln = v
        prev_z = el[t] / math.exp(0.5 * v)
    lns = np.asarray(lns)
    return np.exp(lns), e, lns


def egarchx_loglik(params: np.ndarray, y: np.ndarray, X: np.ndarray | None, backcast: float | None = None) -> float:
    """Student t log-likelihood (standardised t, as in arch) of the AR(1)-EGARCH(1,1)-X."""
    k = 0 if X is None else X.shape[1]
    nu = params[6 + k]
    s2, e, lns = egarchx_filter(params, y, X, backcast)
    if not (np.all(np.isfinite(s2)) and np.all(s2 > 0) and nu > 2):
        return -1e300
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    ll = c - 0.5 * lns - (nu + 1) / 2 * np.log1p(e * e / ((nu - 2) * s2))
    val = float(np.sum(ll))
    return val if np.isfinite(val) else -1e300


def fit_egarchx(y: np.ndarray, X: np.ndarray | None, starts: list[np.ndarray], backcast: float | None = None) -> dict:
    """Maximum likelihood fit from several starting points; returns the best.

    The backcast is held fixed during the fit (arch's default: from an OLS AR(1) fit of y).
    """
    k = 0 if X is None else X.shape[1]
    bc = ols_ar1_backcast(y) if backcast is None else float(backcast)
    bounds = ([(-5, 5), (-0.99, 0.99), (-15, 15), (-2, 3), (-2, 2), (0.0, 0.9999)]
              + [(-5, 5)] * k + [(2.05, 300)])
    best = None
    for s in starts:
        s = np.clip(np.asarray(s, float), [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(lambda p: -egarchx_loglik(p, y, X, bc), s, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000})
        ll = -float(res.fun)
        if best is None or ll > best["loglik"]:
            best = {"params": res.x.copy(), "loglik": ll, "converged": bool(res.success)}
    best["backcast"] = bc
    return best


def egarchx_daily_variance(daily_returns: np.ndarray, X: np.ndarray | None, train_size: int,
                           start_params: np.ndarray, extra_starts=(), scale: float = 100.0) -> tuple[np.ndarray, dict]:
    """One-step-ahead daily variance from an AR(1)-EGARCH(1,1)-X fitted on the first train_size days.

    Mirrors niftygex.volatility.daily_egarch_variance: fit on the training days, then
    hold the parameters fixed and filter the whole series, so h_t uses information up
    to the close of day t-1 only. Two differences, both to keep it strictly real-time:
    the recursion's starting value (the backcast) comes from the training days in both
    steps (arch's ``fix``, which the paper's EGARCH uses, recomputes it from an OLS fit
    on the whole series), and the sanity check looks at the training days only (the
    paper's check also looks at h on every later day of the sample).
    ``start_params`` are the paper's EGARCH estimates on the same days (mu, phi, omega,
    alpha, gamma, beta, nu); delta starts at zero and at a second point where X carries
    half of the persistence; ``extra_starts`` (same layout, delta = 0 added) are tried too.
    Returns (h, info); h is None if the fit fails the sanity checks.
    """
    y = np.asarray(daily_returns, float) * scale
    k = 0 if X is None else X.shape[1]
    Xs = None
    if k:
        mu_x = X[1:train_size].mean(axis=0)
        Xs = X - mu_x          # centred on the training mean: only omega changes
    p0 = np.asarray(start_params, float)
    s1 = np.concatenate([p0[:6], np.zeros(k), p0[6:7]])
    s2 = np.concatenate([p0[:5], [0.5], np.full(k, 0.5 / max(k, 1)), p0[6:7]])
    s2[2] = 0.5 * np.log(np.var(y[:train_size]))
    starts = [s1, s2] if k else [s1]
    starts += [np.concatenate([np.asarray(e, float)[:6], np.zeros(k), np.asarray(e, float)[6:7]]) for e in extra_starts]
    bc = ols_ar1_backcast(y[:train_size])
    fit = fit_egarchx(y[:train_size], None if Xs is None else Xs[:train_size], starts, bc)
    p = fit["params"]
    s2full, _, _ = egarchx_filter(p, y, Xs, bc)
    h = np.concatenate([[s2full[0]], s2full])
    info = {"loglik": fit["loglik"], "converged": fit["converged"],
            **{n: float(v) for n, v in zip(["mu", "phi", "omega", "alpha", "gamma", "beta"], p[:6])},
            **{f"delta_{j + 1}": float(p[6 + j]) for j in range(k)}, "nu": float(p[6 + k])}
    v = np.var(y[:train_size])
    ok = bool(np.all(np.isfinite(p)) and abs(p[1]) < 1 and 0 <= p[5] < 1 and np.all(h[:train_size] > 0)
              and np.all(np.abs(np.log(h[:train_size]) - np.log(v)) < 5))
    info["ok"] = ok
    return (h / scale**2 if ok else None), info


def realtime_guard(h: np.ndarray, fallback: np.ndarray, y_train: np.ndarray, start: int, end: int,
                   scale: float = 100.0) -> tuple[np.ndarray, int | None]:
    """From the first test day in [start, end) whose daily variance is 5 log points or more away
    from the training variance (the paper's sanity bound), use ``fallback`` instead. Each day's
    decision uses only that day's forecast, so the rule could run in real time."""
    v = np.var(np.asarray(y_train, float) * scale)
    bad = np.abs(np.log(h[start:end] * scale**2) - np.log(v)) >= 5
    if not bad.any():
        return h, None
    j = start + int(np.argmax(bad))
    out = h.copy()
    out[j:] = fallback[j:]
    return out, j


def lagged_log_daily_variance(level_annual: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """X_t = log(level_{t-1}^2 / 252 * scale^2): an annualised volatility (VIX, implied vol)
    known at the close of day t-1, as a daily variance in the units of the EGARCH. X_0 = X_1."""
    v = np.log(np.asarray(level_annual, float) ** 2 / 252 * scale**2)
    out = np.empty_like(v)
    out[1:] = v[:-1]
    out[0] = out[1]
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_returns(raw: Path, start="2023-01-01", end="2026-11-02", min_coverage=0.95):
    """The (days x minutes) matrix of 1-minute log returns exactly as run_pipeline.py builds it."""
    from niftygex.data import load_minute_bars, to_session_grid

    bars = load_minute_bars(raw / "nifty_1min.csv")
    bars = bars[(bars.index >= pd.Timestamp(start)) & (bars.index < pd.Timestamp(end) + pd.Timedelta(days=1))]
    grid = to_session_grid(bars, minutes=375)
    keep = np.isfinite(grid["close"]).mean(axis=1) >= min_coverage
    idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
    official = idx.loc[idx["index"].str.lower() == "nifty 50", "date"]
    keep &= grid["days"].isin(pd.DatetimeIndex(official).normalize())
    return np.nan_to_num(grid["returns"][keep], nan=0.0), grid["days"][keep]


def load_forecasts(run: Path, target: str) -> pd.DataFrame:
    return pd.read_csv(run / f"forecasts_{target}.csv", parse_dates=["date"])


def masks_for(f: pd.DataFrame) -> dict:
    return subgroups(f, holdout_start=HOLDOUT_START, auction_start=AUCTION_START)


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Section 1: SPA and Reality Check
# ---------------------------------------------------------------------------


def section_spa(run: Path, out: Path, models: list, reps: int, blocks: list, main_block: float):
    print("\n== 1. SPA test and Reality Check: does the best of the 31 challengers beat mcsGARCH? "
          "(daily average QLIKE; REAL DATA; post hoc)")
    challengers = [m for m in models if m != BENCH]
    rows, per_model = [], []
    for target in TARGETS:
        f = load_forecasts(run, target)
        g = masks_for(f)
        for subset in SUBSETS:
            sub = f[g[subset]]
            L = daily_losses(sub, models, "qlike")
            d = (L[[BENCH]].to_numpy() - L[challengers].to_numpy())
            n = d.shape[0]
            for b in blocks:
                seed = 20261103 + 100 * TARGETS.index(target) + 10 * SUBSETS.index(subset)
                r = spa_test(d, reps=reps, mean_block=b, seed=seed)
                best = challengers[r["best"]]
                rows.append({"target": target, "subset": subset, "days": n, "n_challengers": len(challengers),
                             "mean_block": b, "reps": reps,
                             "best_by_t": best, "best_by_t_pct_vs_mcsgarch": pct(L[best].mean(), L[BENCH].mean()),
                             "best_by_mean": challengers[r["best_by_mean"]],
                             "best_by_mean_pct_vs_mcsgarch": pct(L[challengers[r["best_by_mean"]]].mean(), L[BENCH].mean()),
                             "T_spa": r["T_spa"], "p_spa_consistent": r["p_spa_c"], "p_spa_lower": r["p_spa_l"],
                             "p_spa_upper": r["p_spa_u"], "rc_stat": r["rc_stat"], "p_reality_check": r["p_rc"],
                             "models_dropped_by_consistent_rule": r["n_excluded_consistent"]})
                if b == main_block:
                    for j, mdl in enumerate(challengers):
                        dm = diebold_mariano(L[BENCH], L[mdl], alternative="greater")
                        per_model.append({"target": target, "subset": subset, "model": mdl, "days": n,
                                          "pct_vs_mcsgarch": pct(L[mdl].mean(), L[BENCH].mean()),
                                          "t_stat_sb_kernel": r["t"][j], "p_dm_one_sided": dm["p"],
                                          "p_maxt_single_step": r["p_single"][j],
                                          "p_maxt_stepdown": r["p_stepdown"][j]})
    spa = pd.DataFrame(rows)
    spa.to_csv(out / "spa_reality_check.csv", index=False)
    pm = pd.DataFrame(per_model)
    pm.to_csv(out / "spa_adjusted_pvalues_by_model.csv", index=False)
    main = spa[spa.mean_block == main_block]
    print(f"mean block {main_block:g} days, {reps} stationary-bootstrap replications, 31 challengers")
    print(f"{'target':10s} {'subset':12s} {'days':>5s} {'best (by t)':24s} {'vs mcsGARCH':>11s} "
          f"{'SPA_c':>7s} {'SPA_l':>7s} {'SPA_u':>7s} {'RC':>7s}")
    for r in main.itertuples():
        print(f"{r.target:10s} {r.subset:12s} {r.days:5d} {r.best_by_t:24s} {r.best_by_t_pct_vs_mcsgarch:+10.1f}% "
              f"{fmt_p(r.p_spa_consistent):>7s} {fmt_p(r.p_spa_lower):>7s} {fmt_p(r.p_spa_upper):>7s} "
              f"{fmt_p(r.p_reality_check):>7s}")
    print("block-length sensitivity (SPA_c / RC):")
    for (target, subset), s in spa.groupby(["target", "subset"], sort=False):
        print(f"  {target:10s} {subset:12s} " + "  ".join(
            f"b={r.mean_block:g}: {fmt_p(r.p_spa_consistent)}/{fmt_p(r.p_reality_check)}" for r in s.itertuples()))
    print("hybrid_base: raw DM p, then max-t adjusted p over the 31 challengers (single step, step-down):")
    for r in pm[pm.model == HEADLINE].itertuples():
        print(f"  {r.target:10s} {r.subset:12s} {r.pct_vs_mcsgarch:+6.1f}%  DM {fmt_p(r.p_dm_one_sided)}  "
              f"single {fmt_p(r.p_maxt_single_step)}  step-down {fmt_p(r.p_maxt_stepdown)}")
    cnt = pm.assign(sig=pm.p_maxt_stepdown <= 0.05).groupby(["target", "subset"], sort=False)["sig"].sum()
    print("challengers that beat mcsGARCH at 5% after the step-down adjustment:",
          ", ".join(f"{t}/{s}: {int(v)}" for (t, s), v in cnt.items()))
    return spa, pm


# ---------------------------------------------------------------------------
# Section 2: mcsGARCH-X
# ---------------------------------------------------------------------------


KEEP_COLS = ["t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y", "mcsgarch", "mcsgarch_iv"]
NEW_COLS = ["mcsgarch_rebuilt", "mcsgarch_iv_rebuilt", "mcsgarch_realtime", "mcsgarch_vix", "mcsgarchx_vix",
            "mcsgarchx_vix_iv"]


def section_mcsgarchx(run: Path, raw: Path, out: Path, rebuild: bool):
    from arch import arch_model

    from niftygex.volatility import McsGarch, daily_egarch_variance

    files = {t: out / f"mcsgarchx_forecasts_{t}.csv" for t in TARGETS}
    print("\n== 2. mcsGARCH-X: the paper's model with India VIX (and the straddle) in its daily "
          "EGARCH (REAL DATA; post hoc)")
    if not rebuild and all(p.exists() for p in files.values()):
        print("using cached files", ", ".join(str(p) for p in files.values()))
        return {t: pd.read_csv(p, parse_dates=["date"]) for t, p in files.items()}
    t0 = time.time()
    R, days = load_returns(raw)
    T = len(days)
    dm = pd.read_csv(run / "daily_market.csv", parse_dates=["date"])
    assert (pd.DatetimeIndex(dm["date"]) == days).all(), "days differ from the committed daily_market.csv"
    rv_err = float(np.max(np.abs((R**2).sum(axis=1) / dm["rv_day"].to_numpy() - 1)))
    print(f"returns rebuilt: {T} days; max relative difference of daily RV against daily_market.csv {rv_err:.1e}")
    is_exp = dm["is_expiry"].to_numpy(bool)
    vix = dm["vix"].to_numpy(float)
    ip = pd.read_csv(run / "daily_implied_panel.csv", parse_dates=["publish_date"]).set_index("publish_date")
    iv_near = ip["iv_near"].reindex(days).to_numpy(float)
    # implied variance per day for day t from the chain of t-1, as forecast.py builds mcsgarch_iv
    logiv = pd.Series(np.r_[np.nan, np.log(np.maximum(iv_near[:-1] ** 2 / 252, 1e-14))]).ffill().bfill().to_numpy()
    h_iv = np.exp(logiv)
    X_vix = lagged_log_daily_variance(vix)[:, None]
    X_iv = (logiv + np.log(100.0**2))[:, None]
    X_both = np.column_stack([X_vix[:, 0], X_iv[:, 0]])
    h_vixlevel = np.exp(X_vix[:, 0]) / 100.0**2
    daily = R.sum(axis=1)

    fc = {t: load_forecasts(run, t) for t in TARGETS}
    for t in TARGETS:
        for c in NEW_COLS:
            fc[t][c] = np.nan
    prow = []
    y = daily * 100.0
    for start in range(150, T, 21):
        tb = time.time()
        end = min(start + 21, T)
        h, hinfo = daily_egarch_variance(daily, train_size=start)
        row = {"start": start, "end": end, "paper_daily_method": hinfo["method"]}
        # The paper's fallback rule checks h on every day of the sample, later days included.
        # mcsgarch_realtime applies the same bound only to days already seen (training days,
        # and test days one at a time), so it shows what that look-ahead did.
        h_rt = h
        tfit = arch_model(y[:start], mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t",
                          rescale=False).fit(disp="off", show_warning=False)
        tp = tfit.params
        p_t = np.array([tp["Const"], tp["y[1]"], tp["omega"], tp["alpha[1]"], tp["gamma[1]"], tp["beta[1]"], tp["nu"]])
        if hinfo["method"] != "egarch_t":
            ht = np.asarray(arch_model(y, mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
                            .fix(tfit.params.values).conditional_volatility, float) ** 2
            ht[0] = ht[1]
            lv = np.log(np.var(y[:start]))
            bad = np.abs(np.log(ht) - lv) >= 5
            first_bad = int(np.argmax(bad)) if bad.any() else None
            row["paper_t_fit_first_bad_day"] = first_bad
            row["paper_t_fit_first_bad_date"] = str(days[first_bad].date()) if first_bad is not None else ""
            row["paper_fallback_triggered_by_later_days_only"] = bool(first_bad is None or first_bad >= end)
            if abs(p_t[1]) < 1 and 0 <= p_t[5] < 1 and not bad[:start].any():
                h_rt, j = realtime_guard(ht / 100.0**2, h, daily[:start], start, end)
                row["realtime_daily_method"] = "egarch_t" + (f" (EWMA from day {j})" if j is not None else "")
            else:
                row["realtime_daily_method"] = hinfo["method"]
        else:
            row["realtime_daily_method"] = hinfo["method"]
        variants = {"mcsgarch_rebuilt": h, "mcsgarch_iv_rebuilt": h_iv, "mcsgarch_vix": h_vixlevel}
        if h_rt is not h:
            variants["mcsgarch_realtime"] = h_rt
        # start the X fits at the paper's EGARCH estimates on the same days (arch, Student t)
        p0 = p_t
        if hinfo["method"].startswith("egarch"):
            # the paper's EGARCH h comes from arch's fix(), whose backcast uses an OLS fit on all days;
            # the same parameters with the training-days backcast give the strictly real-time version
            s2nl, _, _ = egarchx_filter(p0, y, None, ols_ar1_backcast(y[:start]))
            h_nl = np.r_[s2nl[0], s2nl] / 100.0**2
            row["paper_h_backcast_max_rel_diff_test_days"] = float(np.max(np.abs(h_nl[start:end] / h[start:end] - 1)))
        p_alt = p0.copy()
        p_alt[5] = 0.5
        p_alt[2] = 0.5 * np.log(np.var(y[:start]))
        base = fit_egarchx(y[:start], None, [p0, p_alt])
        xfits = {}
        for name, X in (("mcsgarchx_vix", X_vix), ("mcsgarchx_vix_iv", X_both)):
            xfits[name] = egarchx_daily_variance(daily, X, start, p0, extra_starts=[base["params"]])
            # the restricted model again, from the X optimum without delta, so that the LR compares optima
            px = np.array([xfits[name][1][c] for c in ("mu", "phi", "omega", "alpha", "gamma", "beta", "nu")])
            b2 = fit_egarchx(y[:start], None, [px])
            if b2["loglik"] > base["loglik"]:
                base = b2
        row["loglik_egarch_own"] = base["loglik"]
        for name, X in (("mcsgarchx_vix", X_vix), ("mcsgarchx_vix_iv", X_both)):
            hx, info = xfits[name]
            row.update({f"{name}_{k}": v for k, v in info.items()})
            row[f"{name}_lr_vs_egarch"] = 2 * (info["loglik"] - base["loglik"])
            if hx is None:      # failed fit: use the (real-time) paper daily variance for this block
                hx = h_rt
                row[f"{name}_daily_method"] = f"fallback to paper ({row['realtime_daily_method']})"
            else:
                hx, j = realtime_guard(hx, h_rt, daily[:start], start, end)
                row[f"{name}_daily_method"] = "egarchx_t" + (f" (paper from day {j})" if j is not None else "")
            variants[name] = hx
        tf = time.time()
        for name, hv in variants.items():
            mcs = McsGarch(expiry_diurnal=True, smooth=5)
            mcs.fit(R[:start], hv[:start], is_exp[:start])
            filt = mcs.filter(R, hv, is_exp)
            row[f"{name}_intraday_alpha"] = mcs.params["alpha"]
            row[f"{name}_intraday_beta"] = mcs.params["beta"]
            for t in TARGETS:
                f = fc[t]
                sel = (f["t"] >= start) & (f["t"] < end)
                f.loc[sel, name] = mcs.forecast_variance(filt, hv, f.loc[sel, "t"].to_numpy(), f.loc[sel, "minute"].to_numpy(),
                                                         f.loc[sel, "horizon"].to_numpy())
        if "mcsgarch_realtime" not in variants:
            for t in TARGETS:
                sel = (fc[t]["t"] >= start) & (fc[t]["t"] < end)
                fc[t].loc[sel, "mcsgarch_realtime"] = fc[t].loc[sel, "mcsgarch_rebuilt"]
        row["seconds_daily_fits"] = round(tf - tb, 2)
        row["mcsgarch_fits"] = len(variants)
        row["seconds_mcsgarch_fits"] = round(time.time() - tf, 2)
        prow.append(row)
        print(f"  block {start}: paper daily {hinfo['method']} (real-time rule: {row['realtime_daily_method']}); "
              f"VIX-X {row['mcsgarchx_vix_daily_method']}, LR {row['mcsgarchx_vix_lr_vs_egarch']:.1f}, delta "
              f"{row['mcsgarchx_vix_delta_1']:.2f}; VIX+IV-X {row['mcsgarchx_vix_iv_daily_method']}, LR "
              f"{row['mcsgarchx_vix_iv_lr_vs_egarch']:.1f}; {time.time() - tb:.1f}s", flush=True)
    params = pd.DataFrame(prow)
    params.to_csv(out / "mcsgarchx_params.csv", index=False)
    for t in TARGETS:
        fc[t][KEEP_COLS + NEW_COLS].to_csv(files[t], index=False)
        e1 = np.max(np.abs(fc[t]["mcsgarch_rebuilt"] / fc[t]["mcsgarch"] - 1))
        e2 = np.max(np.abs(fc[t]["mcsgarch_iv_rebuilt"] / fc[t]["mcsgarch_iv"] - 1))
        print(f"  {t}: rebuilt mcsGARCH vs committed, max relative difference {e1:.1e}; mcsgarch_iv {e2:.1e}")
    secs = time.time() - t0
    print(f"mcsGARCH-X walk-forward done in {secs:.0f}s ({len(prow)} refits; per refit up to 6 mcsGARCH fits "
          f"and 5 daily EGARCH fits; mcsGARCH fits {params['seconds_mcsgarch_fits'].sum():.0f}s in all, "
          f"{params['seconds_mcsgarch_fits'].sum() / params['mcsgarch_fits'].sum():.2f}s each)")
    json.dump({"seconds": round(secs, 1), "refits": len(prow),
               "note": "POST HOC, REAL DATA. mcsGARCH-X built in experiments/referee_forecast.py; pipeline code unchanged."},
              open(out / "mcsgarchx_run_info.json", "w"), indent=2)
    return {t: fc[t][KEEP_COLS + NEW_COLS] for t in TARGETS}


# ---------------------------------------------------------------------------
# Section 3: information-matched benchmarks
# ---------------------------------------------------------------------------

MATCHED = ["mcsgarch_iv", "mcsgarch_vix", "mcsgarchx_vix", "mcsgarchx_vix_iv"]


def section_bench(run: Path, out: Path, mx: dict):
    print("\n== 3. The hybrid against information-matched benchmarks (QLIKE; one-sided DM p that the "
          "hybrid is better; REAL DATA; post hoc)")
    rows = []
    subsets = SUBSETS + ["development", "holdout", "no_ewma_fallback"]
    # test days forecast by the two refits in which the paper's daily EGARCH fell back to an EWMA
    mp = pd.read_csv(run / "mcsgarch_params.csv")
    mp = mp[(mp["target"] == "next15") & (mp["refit_every"] == 21) & (mp["daily_component"] == "egarch (paper)")]
    ewma_starts = sorted(mp.loc[mp["daily_method"] == "ewma", "start"].astype(int))
    for target in TARGETS:
        f = load_forecasts(run, target)
        x = mx[target]
        assert (f["t"].to_numpy() == x["t"].to_numpy()).all() and (f["minute"].to_numpy() == x["minute"].to_numpy()).all()
        for c in ("mcsgarch_realtime", "mcsgarch_vix", "mcsgarchx_vix", "mcsgarchx_vix_iv"):
            f[c] = x[c].to_numpy()
        g = masks_for(f)
        tt = f["t"].to_numpy()
        g["no_ewma_fallback"] = ~np.any([(tt >= s) & (tt < s + 21) for s in ewma_starts], axis=0)
        if target == TARGETS[0]:
            print(f"refits with the EWMA fallback start at days {ewma_starts}; "
                  f"{f.loc[~g['no_ewma_fallback'], 'date'].nunique()} test days are left out in 'no_ewma_fallback'")
        for subset in subsets:
            sub = f[g[subset]]
            mods = [BENCH, "mcsgarch_realtime", "mcsgarch_iv", "mcsgarch_vix", "mcsgarchx_vix", "mcsgarchx_vix_iv", "linear_base", HEADLINE]
            L = daily_losses(sub, mods, "qlike")
            best_matched = min(MATCHED, key=lambda m: L[m].mean())
            for bench in [BENCH, "mcsgarch_realtime"] + MATCHED + ["linear_base", "best_matched_mcsgarch"]:
                b = best_matched if bench == "best_matched_mcsgarch" else bench
                dmh = diebold_mariano(L[b], L[HEADLINE], alternative="greater")
                dmb = diebold_mariano(L[BENCH], L[b], alternative="greater") if b != BENCH else {"p": np.nan}
                rows.append({"target": target, "subset": subset, "days": len(L), "benchmark": bench,
                             "benchmark_model": b,
                             "benchmark_pct_vs_mcsgarch": pct(L[b].mean(), L[BENCH].mean()),
                             "p_benchmark_beats_mcsgarch": dmb["p"],
                             "hybrid_pct_vs_benchmark": pct(L[HEADLINE].mean(), L[b].mean()),
                             "p_hybrid_beats_benchmark": dmh["p"], "dm_stat": dmh["stat"]})
    res = pd.DataFrame(rows)
    res.to_csv(out / "fair_benchmarks.csv", index=False)
    show = ["mcsgarch", "mcsgarch_realtime", "mcsgarch_iv", "mcsgarch_vix", "mcsgarchx_vix", "mcsgarchx_vix_iv", "linear_base",
            "best_matched_mcsgarch"]
    print("benchmark against mcsGARCH (QLIKE %, one-sided p that it beats mcsGARCH):")
    for target in TARGETS:
        for subset in SUBSETS:
            s = res[(res.target == target) & (res.subset == subset)].set_index("benchmark")
            print(f"  {target:10s} {subset:12s} " + "  ".join(
                f"{b}: {s.loc[b, 'benchmark_pct_vs_mcsgarch']:+.1f}% ({fmt_p(s.loc[b, 'p_benchmark_beats_mcsgarch'])})"
                for b in show[1:7]))
    print("hybrid against each benchmark (QLIKE %, one-sided p):")
    for target in TARGETS:
        for subset in subsets:
            s = res[(res.target == target) & (res.subset == subset)].set_index("benchmark")
            print(f"  {target:10s} {subset:12s} " + "  ".join(
                f"{b}: {s.loc[b, 'hybrid_pct_vs_benchmark']:+.1f}% ({fmt_p(s.loc[b, 'p_hybrid_beats_benchmark'])})"
                for b in show) + f"  [best matched = {s.loc['best_matched_mcsgarch', 'benchmark_model']}]")
    return res


# ---------------------------------------------------------------------------
# Section 4: estimator noise
# ---------------------------------------------------------------------------

VARIANTS = ["hybrid_base", "hybrid_gex_gross", "hybrid_gex_us", "hybrid_gex_participant", "hybrid_gex_client",
            "hybrid_gex_placebo"]


def lgbm_cost_estimate(run: Path) -> pd.DataFrame:
    """Upper bound on the cost of rerunning hybrid_base with other seeds, from the final run's own timing.

    LightGBM work in each block is taken as proportional to boosting rounds x training rows:
    tuning pass (default grid, early stopping, 80% of rows, best_iter + 50 rounds), then
    n_seeds final fits of best_iter rounds on all rows; the tuned model (lgbmt) is counted
    with 9 grid points whenever it was re-tuned. All of a target's run time is charged to
    LightGBM, so the result is an upper bound.
    """
    info = json.load(open(run / "run_info.json"))
    n_seeds = int(info["args"]["n_seeds"])
    rows = []
    for target in TARGETS:
        hp = pd.read_csv(run / f"hyperparameters_{target}.csv")
        hp = hp[hp["iters"].notna()].copy()
        hp["train_rows"] = hp["start"]  # rows are proportional to training days (same anchors per day)
        tune_pts = np.where(hp["model"].str.startswith("lgbmt") & hp["retuned"].astype(bool), 9, 1)
        hp["work"] = hp["train_rows"] * (tune_pts * 0.8 * (hp["iters"] + 50) + n_seeds * hp["iters"])
        total = float(hp["work"].sum())
        secs = float(info["target_seconds"][target])
        hb = hp[hp["model"] == HEADLINE]
        single = float((hb["train_rows"] * (0.8 * (hb["iters"] + 50) + hb["iters"])).sum())
        share = {m: float(hp.loc[hp["model"] == m, "work"].sum()) / total for m in (HEADLINE, "lgbm_base", "lgbmt_base")}
        rows.append({"target": target, "target_seconds_final_run": secs, "lgbm_models_in_run": hp["model"].nunique(),
                     "hybrid_share_of_lgbm_work": share[HEADLINE],
                     "seconds_per_5seed_hybrid_run_upper": share[HEADLINE] * secs,
                     "seconds_per_single_seed_hybrid_run_upper": single / total * secs,
                     "seconds_per_5seed_lgbm_base_run_upper": share["lgbm_base"] * secs,
                     "seconds_per_5seed_lgbmt_base_run_upper": share["lgbmt_base"] * secs})
    return pd.DataFrame(rows)


def section_noise(run: Path, out: Path, mcsx_seconds: float | None):
    print("\n== 4. Estimator noise (seeds and refits) and the DM p-values (REAL DATA; post hoc)")
    sn = pd.read_csv(run / "seed_noise_next30.csv")
    single = sn[sn["qlike"].notna()].copy()
    single["seed"] = single["seed"].astype(int)
    sd_rows = []
    for mdl, s in single.groupby("model", sort=False):
        sd_rows.append({"model": mdl, "seeds": ",".join(map(str, s["seed"])), "n": len(s),
                        "margin_min": s["qlike_vs_mcsgarch_pct"].min(), "margin_max": s["qlike_vs_mcsgarch_pct"].max(),
                        "sd_pp": float(s["qlike_vs_mcsgarch_pct"].std(ddof=1))})
    sdt = pd.DataFrame(sd_rows)
    pooled = float(np.sqrt((sdt["sd_pp"] ** 2).mean()))
    pooled_df = int((sdt["n"] - 1).sum())
    hyb_sd = float(sdt.loc[sdt.model == HEADLINE, "sd_pp"].iloc[0])
    sdt = pd.concat([sdt, pd.DataFrame([{"model": "pooled (4 models)", "n": int(sdt["n"].sum()), "sd_pp": pooled,
                                         "sd_pp_upper95": sd_upper_bound(pooled, pooled_df)}])], ignore_index=True)
    sdt.loc[sdt.model == HEADLINE, "sd_pp_upper95"] = sd_upper_bound(hyb_sd, 2)
    sdt.to_csv(out / "seed_noise_summary.csv", index=False)
    up_pooled = sd_upper_bound(pooled, pooled_df)
    print(f"seed noise at 30 minutes, all days (seed_noise_next30.csv, single seeds 7, 8, 9): hybrid_base margins "
          f"{', '.join(f'{v:.2f}' for v in single.loc[single.model == HEADLINE, 'qlike_vs_mcsgarch_pct'])}% "
          f"(SD {hyb_sd:.2f} points); pooled over the 4 models SD {pooled:.2f} points ({pooled_df} df, "
          f"95% upper bound {up_pooled:.2f})")

    rows, env = [], []
    for target in TARGETS:
        f = load_forecasts(run, target)
        g = masks_for(f)
        for subset in SUBSETS + ["holdout"]:
            sub = f[g[subset]]
            L = daily_losses(sub, [BENCH] + VARIANTS, "qlike")
            n = len(L)
            l0 = float(L[BENCH].mean())
            dm = diebold_mariano(L[BENCH], L[HEADLINE], alternative="greater")
            dbar = dm["mean_diff"]
            se = dbar / dm["stat"]                     # HAC standard error incl. the HLN factor
            margin = pct(L[HEADLINE].mean(), l0)
            anchors_ratio = len(load_forecasts(run, "next30")) / max(len(sub), 1)
            # spread of the hybrid variants (same model, same seeds; only uninformative feature blocks differ)
            vm = np.array([pct(L[v].mean(), l0) for v in VARIANTS])
            vp = np.array([diebold_mariano(L[BENCH], L[v], alternative="greater")["p"] for v in VARIANTS])
            var_sd = float(vm.std(ddof=1))
            scen = {"none": 0.0,
                    "seed SD next30 pooled, 5-seed average": pooled / np.sqrt(5),
                    "seed SD next30 pooled, as one seed": pooled,
                    "seed SD next30 95% upper bound, as one seed": up_pooled,
                    "next30 upper bound scaled by sqrt(anchors), as one seed": up_pooled * np.sqrt(anchors_ratio),
                    "spread of the 6 hybrid variants": var_sd}
            row = {"target": target, "subset": subset, "days": n, "anchors": len(sub), "hybrid_pct_vs_mcsgarch": margin,
                   "dm_se_pp": 100 * se / l0, "dm_p": dm["p"]}
            for k, sd_pp in scen.items():
                stat, p = dm_with_extra_noise(dbar, se, n, sd_pp / 100 * l0)
                row[f"extra_sd_pp[{k}]"] = sd_pp
                row[f"p[{k}]"] = p
            for a in (0.05, 0.01):
                row[f"breakeven_sd_pp_p{a}"] = 100 * breakeven_noise_sd(dbar, se, n, a) / l0
            rows.append(row)
            env.append({"target": target, "subset": subset, "days": n,
                        **{f"{v}_pct": m for v, m in zip(VARIANTS, vm)}, **{f"{v}_p": p for v, p in zip(VARIANTS, vp)},
                        "variant_margin_min": vm.min(), "variant_margin_max": vm.max(), "variant_sd_pp": var_sd,
                        "variant_p_max": vp.max()})
    res = pd.DataFrame(rows)
    res.to_csv(out / "seed_adjusted_dm.csv", index=False)
    pd.DataFrame(env).to_csv(out / "refit_envelope.csv", index=False)
    print(f"{'target':10s} {'subset':12s} {'margin':>7s} {'DM SE':>6s} {'DM p':>7s} {'p+seed5':>8s} "
          f"{'p+seed1':>8s} {'p+UB':>7s} {'p+UBscl':>8s} {'p+var':>7s} {'b/e 5%':>7s} {'b/e 1%':>7s}")
    for d in res.to_dict("records"):
        vals = [d[c] for c in res.columns if c.startswith("p[")]
        print(f"{d['target']:10s} {d['subset']:12s} {d['hybrid_pct_vs_mcsgarch']:+6.1f}% {d['dm_se_pp']:6.2f} "
              f"{fmt_p(vals[0]):>7s} {fmt_p(vals[1]):>8s} {fmt_p(vals[2]):>8s} {fmt_p(vals[3]):>7s} {fmt_p(vals[4]):>8s} "
              f"{fmt_p(vals[5]):>7s} {d['breakeven_sd_pp_p0.05']:7.2f} {d['breakeven_sd_pp_p0.01']:7.2f}")
    print("(margin and SDs in points of mcsGARCH's QLIKE; b/e = extra estimator-noise SD at which p reaches 5% or 1%)")
    e = pd.DataFrame(env)
    print("hybrid variants (base and 5 GEX/placebo blocks): margin range and largest DM p")
    for r in e.itertuples():
        print(f"  {r.target:10s} {r.subset:12s} {r.variant_margin_min:+.1f}% to {r.variant_margin_max:+.1f}%, "
              f"SD {r.variant_sd_pp:.2f}, largest p {fmt_p(r.variant_p_max)}")
    cost = lgbm_cost_estimate(run)
    info = json.load(open(run / "run_info.json"))
    prep = float(info["seconds"]) - sum(float(v) for v in info["target_seconds"].values())
    fit_s = np.nan
    mp = out / "mcsgarchx_params.csv"
    if mp.exists():
        q = pd.read_csv(mp)
        fit_s = float(q["seconds_mcsgarch_fits"].sum() / q["mcsgarch_fits"].sum())
    refits, k_configs = 38, 20
    # one configuration = the final run's settings with another cfg.seed: tuning pass plus 5 seeds,
    # for hybrid_base, lgbm_base and lgbmt_base; run_walkforward refits mcsGARCH and mcsgarch_iv in every block
    cost["mcsgarch_fit_seconds_measured"] = fit_s
    cost["mcsgarch_seconds_per_target_run"] = refits * 2 * fit_s
    cost["seconds_per_config_upper"] = (cost["seconds_per_5seed_hybrid_run_upper"] + cost["seconds_per_5seed_lgbm_base_run_upper"]
                                        + cost["seconds_per_5seed_lgbmt_base_run_upper"] + cost["mcsgarch_seconds_per_target_run"])
    cost["data_preparation_seconds_upper"] = prep
    if mcsx_seconds is not None:
        cost["mcsgarchx_walkforward_seconds_measured"] = mcsx_seconds
    cost.to_csv(out / "seed_rerun_cost.csv", index=False)
    print("cost of rerunning hybrid_base (upper bounds from run_info.json and hyperparameters_*.csv):")
    for r in cost.itertuples():
        print(f"  {r.target:10s} one 5-seed hybrid run <= {r.seconds_per_5seed_hybrid_run_upper:.0f}s, one single-seed run "
              f"<= {r.seconds_per_single_seed_hybrid_run_upper:.0f}s (hybrid = {100 * r.hybrid_share_of_lgbm_work:.1f}% of "
              f"the run's LightGBM work; target took {r.target_seconds_final_run:.0f}s); lgbm_base <= "
              f"{r.seconds_per_5seed_lgbm_base_run_upper:.0f}s, lgbmt_base <= {r.seconds_per_5seed_lgbmt_base_run_upper:.0f}s; "
              f"mcsGARCH refits {r.mcsgarch_seconds_per_target_run:.0f}s")
    tot = k_configs * cost["seconds_per_config_upper"].sum() + prep
    print(f"  {k_configs} seed configurations x 3 targets x (hybrid, lgbm_base, lgbmt_base): <= {tot / 3600:.1f} hours on one "
          f"core (data preparation <= {prep:.0f}s; mcsGARCH fit measured at {fit_s:.2f}s); hybrid alone "
          f"<= {(k_configs * (cost['seconds_per_5seed_hybrid_run_upper'] + cost['mcsgarch_seconds_per_target_run']).sum() + prep) / 3600:.1f} hours")
    return res, cost


# ---------------------------------------------------------------------------
# Section 5: Model Confidence Set
# ---------------------------------------------------------------------------


def mcs_rank(df: pd.DataFrame, model: str) -> tuple[int, int]:
    """(elimination rank, tie-aware rank): 1 = the last model left; the tie-aware rank is
    1 + the number of models with a strictly higher MCS p-value."""
    d = df.reset_index(drop=True)
    pos = int(d.index[d["Model name"] == model][0])
    elim_rank = len(d) - pos
    p = float(d.loc[pos, "Pvalue"])
    return elim_rank, int((d["Pvalue"] > p + 1e-12).sum()) + 1


def section_mcs(run: Path, out: Path, models: list, reps: int, blocks: list):
    print("\n== 5. Model Confidence Set, 10% level, daily QLIKE (REAL DATA; committed files, then recomputed)")
    rows = []
    for target in TARGETS:
        for subset in SUBSETS:
            d = pd.read_csv(run / f"mcs_{target}_{subset}.csv")
            kept = d[d["in_mcs"]]
            er, tr = mcs_rank(d, HEADLINE)
            pv = d.set_index("Model name")["Pvalue"]
            inm = d.set_index("Model name")["in_mcs"]
            rows.append({"target": target, "subset": subset, "source": "committed (1,000 reps, block 5)",
                         "kept": len(kept), "of": len(d), "top_ranked": d.iloc[-1]["Model name"],
                         "hybrid_base_p": float(pv[HEADLINE]), "hybrid_base_rank": er, "hybrid_base_rank_ties": tr,
                         **{f"{m}_in": bool(inm[m]) for m in ("mcsgarch", "har", "vix", "mcsgarch_iv", "lgbm_base")},
                         **{f"{m}_p": float(pv[m]) for m in ("mcsgarch", "har", "vix", "mcsgarch_iv", "lgbm_base")},
                         "kept_models": " ".join(kept["Model name"])})
    # recompute: first with the pipeline's settings (reproduction check), then more replications
    for target in TARGETS:
        f = load_forecasts(run, target)
        g = masks_for(f)
        for subset in SUBSETS:
            L = daily_losses(f[g[subset]], models, "qlike")
            for r_, b in [(1000, 5)] + [(reps, bb) for bb in blocks]:
                d = model_confidence_set(L[models], reps=r_, block=b, seed=0).reset_index()
                d = d.rename(columns={d.columns[0]: "Model name"})
                if (r_, b) == (1000, 5):
                    c = pd.read_csv(run / f"mcs_{target}_{subset}.csv").set_index("Model name")["Pvalue"]
                    diff = float((d.set_index("Model name")["Pvalue"] - c.reindex(d["Model name"]).to_numpy()).abs().max())
                    print(f"  {target} {subset}: recomputed with the pipeline's settings, max |p - committed p| = {diff:.1e}")
                kept = d[d["in_mcs"]]
                er, tr = mcs_rank(d, HEADLINE)
                pv = d.set_index("Model name")["Pvalue"]
                inm = d.set_index("Model name")["in_mcs"]
                rows.append({"target": target, "subset": subset, "source": f"recomputed ({r_:,} reps, block {b})",
                             "kept": len(kept), "of": len(d), "top_ranked": d.iloc[-1]["Model name"],
                             "hybrid_base_p": float(pv[HEADLINE]), "hybrid_base_rank": er, "hybrid_base_rank_ties": tr,
                             **{f"{m}_in": bool(inm[m]) for m in ("mcsgarch", "har", "vix", "mcsgarch_iv", "lgbm_base")},
                             **{f"{m}_p": float(pv[m]) for m in ("mcsgarch", "har", "vix", "mcsgarch_iv", "lgbm_base")},
                             "kept_models": " ".join(kept["Model name"])})
    res = pd.DataFrame(rows)
    res.to_csv(out / "mcs_summary.csv", index=False)
    # simple ranking by mean daily QLIKE
    rk = []
    for target in TARGETS:
        f = load_forecasts(run, target)
        g = masks_for(f)
        for subset in SUBSETS:
            L = daily_losses(f[g[subset]], models, "qlike").mean().sort_values()
            rk.append({"target": target, "subset": subset, "lowest_mean_qlike": L.index[0],
                       "hybrid_base_rank_by_mean": int(list(L.index).index(HEADLINE)) + 1,
                       "mcsgarch_rank_by_mean": int(list(L.index).index(BENCH)) + 1, "models": len(L)})
    rk = pd.DataFrame(rk)
    rk.to_csv(out / "rank_by_mean_qlike.csv", index=False)
    for r in res.itertuples():
        print(f"  {r.target:10s} {r.subset:12s} {r.source:34s} keeps {r.kept:2d}/{r.of}; top {r.top_ranked:24s} "
              f"hybrid_base p {r.hybrid_base_p:.3f} rank {r.hybrid_base_rank} (ties {r.hybrid_base_rank_ties}); "
              f"mcsGARCH {'in' if r.mcsgarch_in else 'out'} (p {r.mcsgarch_p:.3f}), HAR {'in' if r.har_in else 'out'} "
              f"(p {r.har_p:.3f}), VIX {'in' if r.vix_in else 'out'} (p {r.vix_p:.3f})")
    print("rank by mean daily QLIKE (1 = lowest):")
    for r in rk.itertuples():
        print(f"  {r.target:10s} {r.subset:12s} lowest: {r.lowest_mean_qlike:24s} hybrid_base {r.hybrid_base_rank_by_mean}"
              f"/{r.models}, mcsGARCH {r.mcsgarch_rank_by_mean}/{r.models}")
    return res, rk


# ---------------------------------------------------------------------------
# Section 6: hold-out
# ---------------------------------------------------------------------------


def section_holdout(run: Path, out: Path):
    print("\n== 6. Development window and hold-out (QLIKE vs mcsGARCH; DM; REAL DATA)")
    mods = ["lgbm_base", "lgbmt_base", HEADLINE, "linear_base"]
    periods = ["development", "holdout", "holdout_pre_auction", "development_expiry", "holdout_expiry", "all"]
    rows, pair = [], []
    for target in TARGETS:
        f = load_forecasts(run, target)
        g = masks_for(f)
        for per in periods:
            sub = f[g[per]]
            L = daily_losses(sub, [BENCH] + mods, "qlike")
            for m in mods:
                dm = diebold_mariano(L[BENCH], L[m], alternative="greater")
                rows.append({"target": target, "period": per, "days": len(L), "model": m,
                             "pct_vs_mcsgarch": pct(L[m].mean(), L[BENCH].mean()), "p_beats_mcsgarch": dm["p"]})
            for lg in ("lgbm_base", "lgbmt_base"):
                two = diebold_mariano(L[HEADLINE], L[lg])
                one = diebold_mariano(L[HEADLINE], L[lg], alternative="greater")
                pair.append({"target": target, "period": per, "days": len(L), "lightgbm": lg,
                             "lightgbm_pct_vs_hybrid": pct(L[lg].mean(), L[HEADLINE].mean()),
                             "dm_stat": two["stat"], "p_two_sided": two["p"], "p_lightgbm_better_one_sided": one["p"]})
    t1, t2 = pd.DataFrame(rows), pd.DataFrame(pair)
    t1.to_csv(out / "holdout_table.csv", index=False)
    t2.to_csv(out / "holdout_lightgbm_vs_hybrid.csv", index=False)
    for target in TARGETS:
        for per in periods[:3]:
            s = t1[(t1.target == target) & (t1.period == per)].set_index("model")
            print(f"  {target:10s} {per:20s} " + "  ".join(
                f"{m}: {s.loc[m, 'pct_vs_mcsgarch']:+.1f}% ({fmt_p(s.loc[m, 'p_beats_mcsgarch'])})" for m in mods))
    print("LightGBM against the hybrid (negative = LightGBM lower loss; two-sided DM p):")
    for r in t2[t2.period.isin(["development", "holdout", "holdout_pre_auction", "all"])].itertuples():
        print(f"  {r.target:10s} {r.period:20s} {r.lightgbm:10s} {r.lightgbm_pct_vs_hybrid:+6.1f}%  p {fmt_p(r.p_two_sided)}")
    return t1, t2


# ---------------------------------------------------------------------------
# Section 7: MSE
# ---------------------------------------------------------------------------


def section_mse(run: Path, out: Path):
    print("\n== 7. The ranking under MSE (committed scoreboards, tests and MCS files; REAL DATA)")
    rows = []
    for target in TARGETS:
        sb = pd.read_csv(run / f"scoreboard_{target}.csv")
        ts = pd.read_csv(run / f"tests_{target}.csv")
        for subset in ("all", "pre_auction", "excl_0915"):
            s = sb[sb.group == subset].copy()
            if s.empty:
                continue
            s["mse_rank"] = s["mse"].rank(method="min").astype(int)
            s["qlike_rank"] = s["qlike"].rank(method="min").astype(int)
            tt = ts[(ts.subset == subset) & (ts.loss == "mse")].copy()
            tt["model"] = tt["comparison"].str.replace("mcsgarch vs ", "", regex=False)
            s = s.merge(tt[["model", "p"]], on="model", how="left")
            mcs_file = run / f"mcs_{target}_{subset}_mse.csv"
            kept = int(pd.read_csv(mcs_file)["in_mcs"].sum()) if mcs_file.exists() else np.nan
            for r in s.itertuples():
                rows.append({"target": target, "subset": subset, "model": r.model, "mse_rank": r.mse_rank,
                             "qlike_rank": r.qlike_rank, "mse_pct_vs_mcsgarch": r.mse_vs_mcsgarch_pct,
                             "qlike_pct_vs_mcsgarch": r.qlike_vs_mcsgarch_pct, "p_mse_beats_mcsgarch": r.p,
                             "mse_mcs_kept": kept, "models": len(s)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "mse_ranking.csv", index=False)
    show = ["har", "vix", "mcsgarch_iv", "linear_base", "lgbm_base", "lgbmt_base", HEADLINE, "mcsgarch"]
    for target in TARGETS:
        s = res[(res.target == target) & (res.subset == "all")].set_index("model")
        print(f"  {target:10s} all days: MSE MCS keeps {int(s['mse_mcs_kept'].iloc[0])}/{len(s)}; " + "  ".join(
            f"{m}: rank {s.loc[m, 'mse_rank']} ({s.loc[m, 'mse_pct_vs_mcsgarch']:+.1f}%, p {fmt_p(s.loc[m, 'p_mse_beats_mcsgarch'])})"
            for m in show))
    return res


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse_v2")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/referee_forecast")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--block", type=float, default=10.0, help="main mean block length (days) for SPA and RC")
    ap.add_argument("--rebuild-mcsgarchx", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="sections to skip: spa mcsgarchx bench noise mcs holdout mse")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.load(open(run / "run_info.json"))
    models = info["models"]
    t0 = time.time()
    print("REAL DATA. POST HOC analyses in response to the referee report (M4, M5, minor 15). Inputs: committed files in "
          f"{run}; outputs in {out}. {len(models)} models, benchmark {BENCH}, headline {HEADLINE}.")
    if "spa" not in a.skip:
        section_spa(run, out, models, a.reps, [5.0, 10.0, 20.0], a.block)
    mx, mx_secs = None, None
    if "mcsgarchx" not in a.skip:
        mx = section_mcsgarchx(run, Path(a.raw), out, a.rebuild_mcsgarchx)
        ri = out / "mcsgarchx_run_info.json"
        mx_secs = json.load(open(ri))["seconds"] if ri.exists() else None
    if "bench" not in a.skip and mx is not None:
        section_bench(run, out, mx)
    if "noise" not in a.skip:
        section_noise(run, out, mx_secs)
    if "mcs" not in a.skip:
        section_mcs(run, out, models, a.reps, [5, 10])
    if "holdout" not in a.skip:
        section_holdout(run, out)
    if "mse" not in a.skip:
        section_mse(run, out)
    print(f"\ndone in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
