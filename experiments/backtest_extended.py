#!/usr/bin/env python
"""Extended out-of-sample backtests computed from the COMMITTED forecast files.

Everything here is REAL DATA (the committed walk-forward forecasts of the real NSE run in
results/pipeline_nse_v2 and the paper-as-printed daily losses in results/p116_benchmark_as_printed).
No raw data is read and nothing is re-estimated. The script was written and run by an AI agent
(Agent A of the backtest round) as a check on the headline claim that the hybrid (mcsGARCH times a
LightGBM correction, column hybrid_base) beats our mcsGARCH (column mcsgarch) and the paper-as-
printed mcsGARCH (mcs_cpp, daily losses only).

Items (one csv each in --out):
 1. conditional gains by regime (VIX tercile, realized-variance tercile, expiry, expiry weekday,
    weekday, net GEX sign, closeness to the gamma flip) with the repo's DM test, and a Giacomini-
    White conditional predictive ability test with these variables as instruments (the GW code is
    reused from experiments/referee_forecast_gw.py; only the instruments are new);
 2. Mincer-Zarnowitz calibration, forecast encompassing, and the distribution of y / f;
 3. alternative loss functions (MSE on log variance, MSE on variance, QLIKE on the volatility
    scale, Patton's family with b = -2, -1, 0, 1);
 4. Hansen's SPA test (stationary bootstrap) for "no model in the ladder beats mcsGARCH", and for
    "no model beats the hybrid";
 5. horizon consistency (30-minute forecast against the sum of the two 15-minute forecasts, and the
    last hour against the sums of the shorter windows);
 6. rolling 63-day gain series, the longest run of losing days, and the drawdown of the cumulative
    gain.

Loss: QLIKE = y / f - log(y / f) - 1, averaged within a day and then across days. Gain in % is
100 (1 - mean loss of the hybrid / mean loss of the benchmark): positive means the hybrid is better.
DM: the repo's niftygex.evaluation.diebold_mariano (Newey-West, Harvey-Leybourne-Newbold factor,
Student t), one-sided p for "the hybrid has the lower loss" unless stated otherwise.

Run:  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/backtest_extended.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

try:
    from niftygex.evaluation import REGIMES, diebold_mariano
except ImportError:  # the test file imports this module without PYTHONPATH=src
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from niftygex.evaluation import REGIMES, diebold_mariano

TARGETS = ["next15", "next30", "last_hour"]
BENCH = "mcsgarch"
HEAD = "hybrid_base"
PAPER = "mcs_cpp"
NON_MODEL = {"t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y"}
EPS = 1e-16


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def qlike(y, f):
    y = np.maximum(np.asarray(y, float), EPS)
    f = np.maximum(np.asarray(f, float), EPS)
    return y / f - np.log(y / f) - 1


def mse_var(y, f):
    return (np.asarray(y, float) - np.asarray(f, float)) ** 2


def mse_log(y, f):
    y = np.maximum(np.asarray(y, float), EPS)
    f = np.maximum(np.asarray(f, float), EPS)
    return (np.log(y) - np.log(f)) ** 2


def qlike_vol(y, f):
    """QLIKE applied to the square roots (volatility scale)."""
    return qlike(np.sqrt(np.maximum(np.asarray(y, float), 0)), np.sqrt(np.maximum(np.asarray(f, float), 0)))


def patton(y, f, b: float):
    """Patton (2011) family of robust losses; b = -2 is QLIKE, b = 0 is MSE / 2."""
    y = np.maximum(np.asarray(y, float), EPS)
    f = np.maximum(np.asarray(f, float), EPS)
    if b == -2:
        return y / f - np.log(y / f) - 1
    if b == -1:
        return f - y + y * np.log(y / f)
    return (y ** (b + 2) - f ** (b + 2)) / ((b + 1) * (b + 2)) - f ** (b + 1) * (y - f) / (b + 1)


LOSSES = {
    "qlike": qlike,
    "mse_log_variance": mse_log,
    "mse_variance": mse_var,
    "qlike_volatility_scale": qlike_vol,
    "patton_b_-2 (QLIKE)": lambda y, f: patton(y, f, -2),
    "patton_b_-1": lambda y, f: patton(y, f, -1),
    "patton_b_0 (MSE/2)": lambda y, f: patton(y, f, 0),
    "patton_b_1": lambda y, f: patton(y, f, 1),
}


def daily_mean_loss(df: pd.DataFrame, models, fn=qlike) -> pd.DataFrame:
    L = pd.DataFrame({m: fn(df["y"].to_numpy(), df[m].to_numpy()) for m in models})
    L["date"] = df["date"].to_numpy()
    return L.groupby("date").mean()


def gain_pct(lb, lh) -> float:
    return float(100 * (1 - np.mean(lh) / np.mean(lb)))


def dm_row(lb, lh) -> dict:
    """DM of benchmark loss minus hybrid loss; one-sided p that the hybrid is better."""
    r = diebold_mariano(lb, lh, alternative="greater")
    return {"n_days": r["n"], "gain_pct": gain_pct(lb, lh), "dm_stat": r["stat"], "dm_p_one_sided": r["p"]}


# ---------------------------------------------------------------------------
# Stationary bootstrap and Hansen's SPA
# ---------------------------------------------------------------------------


def stationary_bootstrap_indices(n: int, draws: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: draws x n index matrix, blocks of geometric length
    (mean mean_block), wrapping around the end of the sample."""
    q = 1.0 / mean_block
    new = rng.random((draws, n)) < q
    new[:, 0] = True
    starts = rng.integers(0, n, size=(draws, n))
    pos = np.arange(n)[None, :]
    block_start = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    return (np.take_along_axis(starts, block_start, axis=1) + (pos - block_start)) % n


def sb_variance(d: np.ndarray, mean_block: float) -> np.ndarray:
    """Long-run variance of each column of d implied by the stationary bootstrap (Hansen 2005)."""
    d = np.asarray(d, float)
    n = d.shape[0]
    q = 1.0 / mean_block
    dc = d - d.mean(axis=0)
    g0 = (dc * dc).sum(axis=0) / n
    out = g0.copy()
    for i in range(1, n):
        k = (n - i) / n * (1 - q) ** i + i / n * (1 - q) ** (n - i)
        if k < 1e-12:
            break
        out += 2 * k * (dc[i:] * dc[:-i]).sum(axis=0) / n
    return np.maximum(out, 1e-300)


def spa_test(loss_bench: np.ndarray, loss_alts: np.ndarray, draws: int = 5000, mean_block: float = 10.0,
             seed: int = 0) -> dict:
    """Hansen (2005) test for superior predictive ability.

    H0: no alternative has lower expected loss than the benchmark. d_{k,t} = L_bench,t - L_k,t.
    Returns the studentized statistic and three p-values (consistent, lower, upper), plus White's
    reality-check p (non-studentized, no recentring) from the same draws."""
    d = np.asarray(loss_bench, float)[:, None] - np.asarray(loss_alts, float)
    n, m = d.shape
    dbar = d.mean(axis=0)
    omega = np.sqrt(sb_variance(d, mean_block))
    tstat = np.sqrt(n) * dbar / omega
    t_spa = max(float(tstat.max()), 0.0)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, draws, mean_block, rng)
    dstar = d[idx].mean(axis=1)                       # draws x m
    z = np.sqrt(n) * (dstar - dbar) / omega           # recentred, studentized
    thresh = np.sqrt(2 * np.log(np.log(n)))
    mu_c = np.where(tstat > -thresh, 0.0, -tstat)     # consistent: keep only models not too poor
    mu_l = np.where(tstat < 0, -tstat, 0.0)           # lower: all poor models pushed to zero
    mu_u = np.zeros(m)                                # upper: every model's mean set to zero
    out = {"n": n, "n_models": m, "t_spa": t_spa, "best_model_index": int(tstat.argmax()),
           "best_t": float(tstat.max())}
    for name, mu in (("p_consistent", mu_c), ("p_lower", mu_l), ("p_upper", mu_u)):
        tb = np.maximum((z - mu).max(axis=1), 0.0)
        out[name] = float((tb >= t_spa).mean())
    rc = float((np.sqrt(n) * dbar).max())
    rc_b = (np.sqrt(n) * (dstar - dbar)).max(axis=1)
    out["p_reality_check"] = float((rc_b >= rc).mean())
    out["tstat"] = tstat
    return out


# ---------------------------------------------------------------------------
# HAC regressions with scores summed within a day
# ---------------------------------------------------------------------------


def nw_lag(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def hac_ols(X: np.ndarray, y: np.ndarray, day: np.ndarray) -> dict:
    """OLS with a covariance that sums the scores within each day (anchors of one day share a
    shock) and then applies Newey-West across days with the repo's bandwidth rule."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    S = pd.DataFrame(X * u[:, None]).groupby(np.asarray(day)).sum().to_numpy()   # days x k
    nd = len(S)
    lag = nw_lag(nd)
    Om = S.T @ S
    for k in range(1, lag + 1):
        G = S[k:].T @ S[:-k]
        Om += (1 - k / (lag + 1)) * (G + G.T)
    V = XtX_inv @ Om @ XtX_inv
    return {"coef": b, "se": np.sqrt(np.diag(V)), "V": V, "n": len(y), "n_days": nd}


def wald(coef, V, R, r) -> tuple[float, float]:
    R = np.atleast_2d(R)
    diff = R @ coef - r
    stat = float(diff @ np.linalg.solve(R @ V @ R.T, diff))
    return stat, float(chi2.sf(stat, len(diff)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_forecasts(run: Path, target: str) -> pd.DataFrame:
    f = pd.read_csv(run / f"forecasts_{target}.csv", parse_dates=["date"])
    return f.sort_values(["t", "minute"]).reset_index(drop=True)


def day_conditioning(run: Path, test_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Day-level conditioning variables, every one known at the close of the previous day."""
    dm = pd.read_csv(run / "daily_market.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    gx = pd.read_csv(run / "daily_gex_panel.csv", parse_dates=["publish_date"]).set_index("publish_date")
    dm["vix_prev"] = dm["vix"].shift(1)
    dm["rv_prev"] = dm["rv_day"].shift(1)
    dm["prev_date"] = dm["date"].shift(1)
    dm["close_prev"] = dm["close_last_bar"].shift(1)
    g = gx.reindex(dm["prev_date"])
    dm["net_gex_us_prev"] = g["net_gex_us_convention"].to_numpy()
    dm["net_gex_part_prev"] = g["net_gex_participant"].to_numpy()
    dm["flip_us_prev"] = g["flip_us_convention"].to_numpy()
    dm["flip_part_prev"] = g["flip_participant"].to_numpy()
    dm["flip_dist_us_pct"] = 100 * np.abs(dm["close_prev"] - dm["flip_us_prev"]) / dm["close_prev"]
    dm["flip_dist_part_pct"] = 100 * np.abs(dm["close_prev"] - dm["flip_part_prev"]) / dm["close_prev"]
    dm = dm.set_index("date").reindex(test_days)
    dm["vix_tercile"] = pd.qcut(dm["vix_prev"], 3, labels=["low", "mid", "high"]).astype(str)
    dm["rv_tercile"] = pd.qcut(dm["rv_prev"], 3, labels=["low", "mid", "high"]).astype(str)
    wd = dm.index.weekday
    dm["weekday_name"] = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[wd]
    dm["gex_us_sign"] = np.where(dm["net_gex_us_prev"].isna(), "nan", np.where(dm["net_gex_us_prev"] >= 0, "positive", "negative"))
    dm["gex_part_sign"] = np.where(dm["net_gex_part_prev"].isna(), "nan", np.where(dm["net_gex_part_prev"] >= 0, "positive", "negative"))
    dm["near_flip_us"] = np.where(dm["flip_dist_us_pct"].isna(), "nan", np.where(dm["flip_dist_us_pct"] <= 0.5, "within_0.5pct", "farther"))
    dm["near_flip_part"] = np.where(dm["flip_dist_part_pct"].isna(), "nan", np.where(dm["flip_dist_part_pct"] <= 0.5, "within_0.5pct", "farther"))
    dm["flip_dist_us_tercile"] = pd.qcut(dm["flip_dist_us_pct"], 3, labels=["near", "mid", "far"]).astype(str)
    return dm


def load_paper(p116: Path, target: str, index: str = "nifty") -> pd.Series:
    """Daily QLIKE of the paper-as-printed model; an all-NaN series when the file is not committed."""
    p = p116 / f"daily_losses_{index}_{target}.csv"
    if not p.exists():
        return pd.Series(dtype=float, name=PAPER)
    d = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    return d[PAPER]


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def item1_regimes(day: pd.DataFrame, cond: pd.DataFrame, target: str) -> list[dict]:
    """day: daily losses with columns BENCH, HEAD, PAPER (PAPER may hold NaN), is_expiry, regime."""
    e = day["is_expiry"].to_numpy(bool)
    reg = np.array(REGIMES)[day["regime"].to_numpy(int)]
    wdn = cond["weekday_name"].to_numpy()
    cells = {("all", "all"): np.ones(len(day), bool)}
    for lvl in ["low", "mid", "high"]:
        cells[("vix_prev_tercile", lvl)] = cond["vix_tercile"].to_numpy() == lvl
        cells[("rv_prev_tercile", lvl)] = cond["rv_tercile"].to_numpy() == lvl
    cells[("expiry", "expiry_day")] = e
    cells[("expiry", "non_expiry_day")] = ~e
    for w in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        m = e & (wdn == w)
        if m.sum() >= 5:
            cells[("expiry_weekday", f"expiry_on_{w}")] = m
    for r in REGIMES:
        cells[("expiry_regime", f"{r}_expiry")] = e & (reg == r)
        cells[("expiry_regime", f"{r}_non_expiry")] = ~e & (reg == r)
    for w in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        cells[("weekday", w)] = wdn == w
    for s in ["positive", "negative"]:
        cells[("net_gex_us_convention_prev_sign", s)] = cond["gex_us_sign"].to_numpy() == s
        cells[("net_gex_participant_prev_sign", s)] = cond["gex_part_sign"].to_numpy() == s
    for s in ["within_0.5pct", "farther"]:
        cells[("prev_close_vs_flip_us", s)] = cond["near_flip_us"].to_numpy() == s
        cells[("prev_close_vs_flip_participant", s)] = cond["near_flip_part"].to_numpy() == s
    for s in ["near", "mid", "far"]:
        cells[("prev_close_distance_to_flip_us_tercile", s)] = cond["flip_dist_us_tercile"].to_numpy() == s
    rows = []
    for (fam, lvl), m in cells.items():
        for bench in (BENCH, PAPER):
            lb = day.loc[m, bench].to_numpy(float)
            lh = day.loc[m, HEAD].to_numpy(float)
            ok = np.isfinite(lb) & np.isfinite(lh)
            if ok.sum() < 3:
                continue
            r = dm_row(lb[ok], lh[ok])
            r.update({"target": target, "family": fam, "cell": lvl, "benchmark": bench,
                      "mean_loss_benchmark": float(lb[ok].mean()), "mean_loss_hybrid": float(lh[ok].mean()),
                      "share_days_hybrid_better": float((lh[ok] < lb[ok]).mean())})
            rows.append(r)
    return rows


def item1_gw(day: pd.DataFrame, cond: pd.DataFrame, target: str, gw) -> tuple[list[dict], list[dict]]:
    """Giacomini-White tests with the new conditioning variables as instruments, reusing
    gw_test and slope_wald from experiments/referee_forecast_gw.py."""
    rows, coefs = [], []
    for bench in (BENCH, PAPER):
        if day[bench].isna().all():
            continue
        lb = day[bench].to_numpy(float)
        lh = day[HEAD].to_numpy(float)
        d_all = (lb - lh) * 100.0 / np.nanmean(lb)       # points of the benchmark's mean daily QLIKE
        e = day["is_expiry"].to_numpy(bool).astype(float)
        fam = {}
        fam["vix_tercile"] = ([(cond["vix_tercile"] == l).to_numpy(float) for l in ("mid", "high")], ["vix_mid", "vix_high"])
        fam["rv_tercile"] = ([(cond["rv_tercile"] == l).to_numpy(float) for l in ("mid", "high")], ["rv_mid", "rv_high"])
        fam["expiry"] = ([e], ["expiry"])
        fam["weekday"] = ([(cond["weekday_name"] == w).to_numpy(float) for w in ("Tue", "Wed", "Thu", "Fri")], ["Tue", "Wed", "Thu", "Fri"])
        fam["gex_us_sign"] = ([np.where(cond["gex_us_sign"] == "nan", np.nan, (cond["gex_us_sign"] == "negative").astype(float))], ["gex_us_negative"])
        fam["gex_participant_sign"] = ([np.where(cond["gex_part_sign"] == "nan", np.nan, (cond["gex_part_sign"] == "negative").astype(float))], ["gex_part_negative"])
        fam["near_flip_us"] = ([np.where(cond["near_flip_us"] == "nan", np.nan, (cond["near_flip_us"] == "within_0.5pct").astype(float))], ["near_flip_us"])
        fam["flip_distance_tercile"] = ([(cond["flip_dist_us_tercile"] == l).to_numpy(float) for l in ("mid", "far")], ["flip_dist_mid", "flip_dist_far"])
        all_cols, all_names = [], []
        for k, (cols, names) in fam.items():
            all_cols += cols
            all_names += names
        fam["all_new_instruments"] = (all_cols, all_names)
        for k, (cols, names) in [("none (C1: constant + lagged d)", ([], []))] + list(fam.items()):
            H = np.column_stack([np.ones(len(d_all) - 1), d_all[:-1]] + [c[1:] for c in cols])
            names_all = ["const", "d_lag"] + names
            d = d_all[1:]
            ok = np.isfinite(H).all(axis=1) & np.isfinite(d)
            Hk, dk = H[ok], d[ok]
            g = gw.gw_test(dk, Hk, "gw")
            row = {"target": target, "benchmark": bench, "instrument_family": k, "instruments": " + ".join(names_all),
                   "n_days": g["n"], "gw_df": g["df"], "gw_chi2": g["stat"], "gw_p": g["p"]}
            if cols:
                s = gw.slope_wald(dk, Hk, list(range(2, Hk.shape[1])))
                row.update({"slope_wald_df": s["df"], "slope_wald_chi2": s["stat"], "slope_wald_p": s["p"]})
                for nm, c, se in zip(names_all, s["coef"], s["se"]):
                    coefs.append({"target": target, "benchmark": bench, "instrument_family": k, "term": nm,
                                  "coef_points": float(c), "se_points": float(se)})
            else:
                s = gw.slope_wald(dk, Hk, [1])
                row.update({"slope_wald_df": s["df"], "slope_wald_chi2": s["stat"], "slope_wald_p": s["p"]})
            rows.append(row)
    return rows, coefs


def item2_calibration(f: pd.DataFrame, target: str) -> tuple[list[dict], list[dict], list[dict]]:
    day = f["date"].to_numpy()
    y = f["y"].to_numpy(float)
    # days whose mean realized variance is in the top 1% (the June 2024 election days and a few others)
    ymean = pd.Series(y).groupby(day).mean()
    top = ymean[ymean >= ymean.quantile(0.99)].index
    keep = ~np.isin(day, top)
    mz, enc, ratio = [], [], []
    for m in (HEAD, BENCH, "lgbm_base", "har"):
        fx = f[m].to_numpy(float)
        for scale, yy, ff, sel in (("log", np.log(y), np.log(fx), slice(None)), ("variance", y, fx, slice(None)),
                                   ("variance_excl_top1pct_days", y[keep], fx[keep], keep)):
            X = np.column_stack([np.ones(len(ff)), ff])
            r = hac_ols(X, yy, day[sel])
            st, p = wald(r["coef"], r["V"], np.eye(2), np.array([0.0, 1.0]))
            st1, p1 = wald(r["coef"], r["V"], np.array([[0.0, 1.0]]), np.array([1.0]))
            mz.append({"target": target, "model": m, "scale": scale, "n_obs": r["n"], "n_days": r["n_days"],
                       "intercept": r["coef"][0], "intercept_se": r["se"][0], "slope": r["coef"][1], "slope_se": r["se"][1],
                       "wald_joint_0_1_chi2": st, "wald_joint_0_1_p": p, "wald_slope_1_chi2": st1, "wald_slope_1_p": p1,
                       "r2": 1 - np.var(yy - X @ r["coef"]) / np.var(yy)})
        q = y / fx
        ratio.append({"target": target, "model": m, "mean_y_over_f": q.mean(), "p10": np.quantile(q, 0.1),
                      "median": np.median(q), "p90": np.quantile(q, 0.9), "share_y_over_f_above_1": (q > 1).mean(),
                      "mean_log_y_over_f": np.log(q).mean()})
    fh, fm = f[HEAD].to_numpy(float), f[BENCH].to_numpy(float)
    for scale, yy, a, b, sel in (("variance", y, fh, fm, slice(None)),
                                 ("variance_excl_top1pct_days", y[keep], fh[keep], fm[keep], keep),
                                 ("log", np.log(y), np.log(fh), np.log(fm), slice(None))):
        # y = w f_h + (1 - w) f_m + e  <=>  y - f_m = w (f_h - f_m) + e  (with an intercept)
        X = np.column_stack([np.ones(len(yy)), a - b])
        r = hac_ols(X, yy - b, day[sel])
        w, se = r["coef"][1], r["se"][1]
        s0, p0 = wald(r["coef"], r["V"], np.array([[0.0, 1.0]]), np.array([0.0]))
        s1, p1 = wald(r["coef"], r["V"], np.array([[0.0, 1.0]]), np.array([1.0]))
        enc.append({"target": target, "scale": scale, "weight_on_hybrid": w, "se": se, "intercept": r["coef"][0],
                    "wald_w_eq_0_chi2": s0, "p_w_eq_0 (hybrid adds nothing to mcsgarch)": p0,
                    "wald_w_eq_1_chi2": s1, "p_w_eq_1 (mcsgarch adds nothing to hybrid)": p1, "n_obs": r["n"]})
    # QLIKE-optimal combination weight on a grid (daily average QLIKE)
    grid = np.round(np.arange(0.0, 1.51, 0.01), 2)
    best, bestw, lh_daily = np.inf, np.nan, None
    for w in grid:
        fc = w * fh + (1 - w) * fm
        Ld = pd.Series(qlike(y, fc)).groupby(day).mean()
        if Ld.mean() < best:
            best, bestw, lh_daily = Ld.mean(), w, Ld
    L1 = pd.Series(qlike(y, fh)).groupby(day).mean()
    L0 = pd.Series(qlike(y, fm)).groupby(day).mean()
    r_best = diebold_mariano(L1.to_numpy(), lh_daily.to_numpy(), alternative="greater")
    enc.append({"target": target, "scale": "qlike_grid", "weight_on_hybrid": bestw, "se": np.nan, "intercept": 0.0,
                "wald_w_eq_0_chi2": np.nan, "p_w_eq_0 (hybrid adds nothing to mcsgarch)": np.nan,
                "wald_w_eq_1_chi2": np.nan, "p_w_eq_1 (mcsgarch adds nothing to hybrid)": np.nan, "n_obs": len(y),
                "combination_gain_over_hybrid_pct": gain_pct(L1, lh_daily), "combination_gain_over_mcsgarch_pct": gain_pct(L0, lh_daily),
                "dm_combination_vs_hybrid_p_one_sided": r_best["p"]})
    return mz, enc, ratio


def item3_losses(f: pd.DataFrame, target: str) -> list[dict]:
    rows = []
    for name, fn in LOSSES.items():
        L = daily_mean_loss(f, [BENCH, HEAD, "lgbm_base", "har"], fn)
        r = dm_row(L[BENCH].to_numpy(), L[HEAD].to_numpy())
        r.update({"target": target, "loss": name, "mean_loss_mcsgarch": L[BENCH].mean(), "mean_loss_hybrid": L[HEAD].mean(),
                  "share_days_hybrid_better": float((L[HEAD] < L[BENCH]).mean()),
                  "gain_vs_lgbm_base_pct": gain_pct(L["lgbm_base"], L[HEAD]), "gain_vs_har_pct": gain_pct(L["har"], L[HEAD])})
        rows.append(r)
    return rows


def item4_spa(day_all: pd.DataFrame, target: str, draws: int, block: float, seed: int) -> tuple[list[dict], list[dict]]:
    L = day_all.dropna(axis=1, how="all").dropna()
    rows, per_model = [], []
    for bench, label in ((BENCH, "mcsgarch"), (HEAD, "hybrid_base")):
        alts = [c for c in L.columns if c != bench]
        res = spa_test(L[bench].to_numpy(), L[alts].to_numpy(), draws, block, seed)
        gains = np.array([gain_pct(L[bench], L[a]) for a in alts])
        rows.append({"target": target, "benchmark": bench, "n_days": res["n"], "n_alternatives": res["n_models"],
                     "share_alternatives_with_lower_mean_loss": float((gains > 0).mean()),
                     "best_alternative_by_t": alts[res["best_model_index"]], "its_gain_pct": float(gains[res["best_model_index"]]),
                     "best_alternative_by_gain": alts[int(gains.argmax())], "best_gain_pct": float(gains.max()),
                     "t_spa": res["t_spa"], "spa_p_consistent": res["p_consistent"], "spa_p_lower": res["p_lower"],
                     "spa_p_upper": res["p_upper"], "reality_check_p": res["p_reality_check"], "draws": draws,
                     "mean_block_days": block, "seed": seed})
        if bench == BENCH:
            for a, g, t in zip(alts, gains, res["tstat"]):
                per_model.append({"target": target, "model": a, "gain_vs_mcsgarch_pct": float(g), "studentized_t": float(t)})
    return rows, per_model


def item5_horizon(f15: pd.DataFrame, f30: pd.DataFrame, f60: pd.DataFrame, target_pair: str) -> list[dict]:
    rows = []
    for m in (HEAD, BENCH, "lgbm_base", "har"):
        p15 = f15.pivot(index="date", columns="minute", values=m)
        y15 = f15.pivot(index="date", columns="minute", values="y")
        p30 = f30.pivot(index="date", columns="minute", values=m)
        y30 = f30.pivot(index="date", columns="minute", values="y")
        a30 = p30.columns.to_numpy()
        # 30-minute window at anchor a = 15-minute windows at a and a + 15
        s15 = p15[a30].to_numpy() + p15[a30 + 15].to_numpy()
        ys = y15[a30].to_numpy() + y15[a30 + 15].to_numpy()
        assert np.allclose(ys, y30.to_numpy(), rtol=1e-9)
        direct = p30.to_numpy()
        yy = y30.to_numpy()
        Ld = qlike(yy, direct).mean(axis=1)
        # (i) sum of the two 15-minute forecasts: the second one is made 15 minutes later, so it has
        #     more information than the direct 30-minute forecast (not a fair horse race);
        # (ii) 2 x the first 15-minute forecast: same information set, assumes a flat variance.
        for lab, alt in (("next30 vs sum of two next15 (second made 15 min later)", s15),
                         ("next30 vs 2 x next15 at the same anchor (same information)", 2 * p15[a30].to_numpy())):
            Ls = qlike(yy, alt).mean(axis=1)
            ratio = (direct / alt).ravel()
            r = diebold_mariano(Ls, Ld, alternative="two-sided")
            rows.append({"pair": lab, "model": m, "n_days": len(Ld), "n_windows": direct.size,
                         "qlike_direct": Ld.mean(), "qlike_sum_of_shorter": Ls.mean(),
                         "gain_of_direct_over_sum_pct": gain_pct(Ls, Ld), "dm_stat": r["stat"], "dm_p_two_sided": r["p"],
                         "ratio_direct_over_sum_median": np.median(ratio), "ratio_p10": np.quantile(ratio, 0.1),
                         "ratio_p90": np.quantile(ratio, 0.9), "corr_log_direct_log_sum": np.corrcoef(np.log(direct.ravel()), np.log(alt.ravel()))[0, 1]})
        # last hour (anchor 315, 60 minutes) = next30 at 315 + 345 = next15 at 315, 330, 345, 360
        y60 = f60.set_index("date")["y"]
        d60 = f60.set_index("date")[m]
        s30 = p30[315] + p30[345]
        s15b = p15[315] + p15[330] + p15[345] + p15[360]
        assert np.allclose((y30[315] + y30[345]).to_numpy(), y60.reindex(y30.index).to_numpy(), rtol=1e-9)
        for lab, s in (("last_hour vs sum of two next30 (second made 30 min later)", s30),
                       ("last_hour vs sum of four next15 (later ones made later)", s15b),
                       ("last_hour vs 2 x next30 at 315 (same information)", 2 * p30[315]),
                       ("last_hour vs 4 x next15 at 315 (same information)", 4 * p15[315])):
            Ld = qlike(y60.reindex(s.index).to_numpy(), d60.reindex(s.index).to_numpy())
            Ls = qlike(y60.reindex(s.index).to_numpy(), s.to_numpy())
            ratio = (d60.reindex(s.index) / s).to_numpy()
            r = diebold_mariano(Ls, Ld, alternative="two-sided")
            rows.append({"pair": lab, "model": m, "n_days": len(Ld), "n_windows": len(Ld),
                         "qlike_direct": Ld.mean(), "qlike_sum_of_shorter": Ls.mean(),
                         "gain_of_direct_over_sum_pct": gain_pct(Ls, Ld), "dm_stat": r["stat"], "dm_p_two_sided": r["p"],
                         "ratio_direct_over_sum_median": np.median(ratio), "ratio_p10": np.quantile(ratio, 0.1),
                         "ratio_p90": np.quantile(ratio, 0.9), "corr_log_direct_log_sum": np.corrcoef(np.log(d60.reindex(s.index)), np.log(s))[0, 1]})
    # cross check: which 30-minute forecast is best, the hybrid's own or the summed hybrid 15s, against mcsGARCH's
    return rows


def longest_run(flag: np.ndarray) -> tuple[int, int, int]:
    """Length, start index and end index of the longest run of True."""
    best = (0, -1, -1)
    i = 0
    n = len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            if j - i + 1 > best[0]:
                best = (j - i + 1, i, j)
            i = j + 1
        else:
            i += 1
    return best


def item6_rolling(day: pd.DataFrame, target: str, window: int = 63) -> tuple[pd.DataFrame, list[dict]]:
    series, summ = [], []
    dates = day.index
    for bench in (BENCH, PAPER):
        lb = day[bench].to_numpy(float)
        lh = day[HEAD].to_numpy(float)
        ok = np.isfinite(lb) & np.isfinite(lh)
        lb, lh, dd = lb[ok], lh[ok], dates[ok]
        n = len(lb)
        if n < window + 1:
            continue
        cb = np.convolve(lb, np.ones(window), "valid") / window
        ch = np.convolve(lh, np.ones(window), "valid") / window
        g = 100 * (1 - ch / cb)
        for k, v in enumerate(g):
            series.append({"target": target, "benchmark": bench, "window_end": dd[k + window - 1].date(), "gain_pct": v})
        lose = lh > lb
        L, i0, i1 = longest_run(lose)
        W, w0, w1 = longest_run(~lose)
        d = (lb - lh) * 100 / lb.mean()           # daily gain in points of the benchmark's mean loss
        cum = np.cumsum(d)
        peak = np.maximum.accumulate(cum)
        dd_pts = peak - cum
        k = int(dd_pts.argmax())
        summ.append({"target": target, "benchmark": bench, "window_days": window, "n_windows": len(g),
                     "rolling_gain_min_pct": g.min(), "rolling_gain_min_window_end": dd[int(g.argmin()) + window - 1].date(),
                     "rolling_gain_median_pct": np.median(g), "rolling_gain_max_pct": g.max(),
                     "share_windows_positive": float((g > 0).mean()),
                     "longest_losing_run_days": L, "losing_run_start": dd[i0].date() if L else None, "losing_run_end": dd[i1].date() if L else None,
                     "longest_winning_run_days": W, "winning_run_start": dd[w0].date() if W else None, "winning_run_end": dd[w1].date() if W else None,
                     "share_days_hybrid_better": float((~lose).mean()),
                     "max_drawdown_of_cumulative_gain_points": float(dd_pts.max()), "drawdown_trough_date": dd[k].date(),
                     "cumulative_gain_end_points": float(cum[-1]), "n_days": n})
    return pd.DataFrame(series), summ


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_gw_module():
    p = Path(__file__).resolve().with_name("referee_forecast_gw.py")
    spec = importlib.util.spec_from_file_location("referee_forecast_gw", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="nifty", choices=["nifty", "banknifty", "finnifty"],
                    help="which index's committed run to read; sets the defaults of --run and --out")
    ap.add_argument("--run", default=None, help="results folder of the run (default results/pipeline_nse_v2 for nifty, "
                                                "results/pipeline_nse_<index> otherwise)")
    ap.add_argument("--p116", default="results/p116_benchmark_as_printed")
    ap.add_argument("--out", default=None, help="default results/backtest_extended for nifty, results/backtest_extended/<index> otherwise")
    ap.add_argument("--draws", type=int, default=5000, help="stationary-bootstrap draws for the SPA test")
    ap.add_argument("--block", type=float, default=10.0, help="mean block length in days")
    ap.add_argument("--seed", type=int, default=9_700_001)
    ap.add_argument("--skip", default="", help="comma-separated item numbers to skip, e.g. 4,5")
    a = ap.parse_args(argv)
    if a.run is None:
        a.run = "results/pipeline_nse_v2" if a.index == "nifty" else f"results/pipeline_nse_{a.index}"
    if a.out is None:
        a.out = "results/backtest_extended" if a.index == "nifty" else f"results/backtest_extended/{a.index}"
    run, p116, out = Path(a.run), Path(a.p116), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    t0 = time.time()
    gw = load_gw_module()
    print(f"REAL DATA ({a.index}, run {run}), computed by an AI agent from the committed forecasts. Hybrid = hybrid_base; "
          "benchmarks = mcsgarch (our mcsGARCH) and mcs_cpp (paper as printed, daily losses only, where committed).")

    fc = {t: load_forecasts(run, t) for t in TARGETS}
    models = [c for c in fc["next15"].columns if c not in NON_MODEL]
    days, conds = {}, {}
    regime_rows, gw_rows, gw_coefs, mz_rows, enc_rows, ratio_rows, loss_rows, spa_rows, spa_models = [], [], [], [], [], [], [], [], []
    roll_series, roll_summ = [], []
    for t in TARGETS:
        f = fc[t]
        L = daily_mean_loss(f, models)
        cal = f.groupby("date").agg(is_expiry=("is_expiry", "first"), regime=("regime", "first"), t=("t", "first"))
        day = L.join(cal)
        paper = load_paper(p116, t, a.index)
        day[PAPER] = paper.reindex(day.index).to_numpy()
        if day[PAPER].isna().all():
            print(f"  no paper-as-printed daily losses committed for {a.index} {t}: mcs_cpp rows skipped")
        days[t] = day
        cond = day_conditioning(run, day.index)
        conds[t] = cond
        okp = day[PAPER].notna()
        print(f"\n[{t}] {len(day)} test days {day.index.min().date()} to {day.index.max().date()}; "
              f"hybrid gain vs mcsgarch {gain_pct(day[BENCH], day[HEAD]):.2f}%"
              + (f", vs mcs_cpp {gain_pct(day.loc[okp, PAPER], day.loc[okp, HEAD]):.2f}%" if okp.any() else ""))
        if "1" not in skip:
            regime_rows += item1_regimes(day, cond, t)
            r, c = item1_gw(day, cond, t, gw)
            gw_rows += r
            gw_coefs += c
            pd.DataFrame(regime_rows).to_csv(out / "regime_gains.csv", index=False)
            pd.DataFrame(gw_rows).to_csv(out / "gw_tests.csv", index=False)
            pd.DataFrame(gw_coefs).to_csv(out / "gw_slope_coefficients.csv", index=False)
        if "2" not in skip:
            mz, enc, ratio = item2_calibration(f, t)
            mz_rows += mz
            enc_rows += enc
            ratio_rows += ratio
            pd.DataFrame(mz_rows).to_csv(out / "calibration_mincer_zarnowitz.csv", index=False)
            pd.DataFrame(enc_rows).to_csv(out / "encompassing.csv", index=False)
            pd.DataFrame(ratio_rows).to_csv(out / "ratio_y_over_f.csv", index=False)
        if "3" not in skip:
            loss_rows += item3_losses(f, t)
            pd.DataFrame(loss_rows).to_csv(out / "alternative_losses.csv", index=False)
        if "4" not in skip:
            r, pm = item4_spa(day[models + [PAPER]], t, a.draws, a.block, a.seed)
            spa_rows += r
            spa_models += pm
            pd.DataFrame(spa_rows).to_csv(out / "spa.csv", index=False)
            pd.DataFrame(spa_models).to_csv(out / "spa_per_model.csv", index=False)
        if "6" not in skip:
            s, m = item6_rolling(day, t)
            roll_series.append(s)
            roll_summ += m
            pd.concat(roll_series).to_csv(out / "rolling_gain_series.csv", index=False)
            pd.DataFrame(roll_summ).to_csv(out / "rolling_gain_summary.csv", index=False)
        print(f"  done in {time.time() - t0:.0f}s")
    if "5" not in skip:
        h = pd.DataFrame(item5_horizon(fc["next15"], fc["next30"], fc["last_hour"], "all"))
        h.to_csv(out / "horizon_consistency.csv", index=False)

    # ----- summary -----
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.4g}")
    if regime_rows:
        rg = pd.DataFrame(regime_rows)
        print("\n1. Gain of the hybrid by regime (REAL DATA; gain %, one-sided DM p):")
        for t in TARGETS:
            s = rg[(rg.target == t) & (rg.benchmark == BENCH)]
            print(f"  {t} vs mcsgarch:")
            for r in s.itertuples():
                print(f"    {r.family:38s} {r.cell:26s} n={r.n_days:4d} gain={r.gain_pct:6.2f}%  DM={r.dm_stat:6.2f} p={r.dm_p_one_sided:.3g}")
        gwt = pd.DataFrame(gw_rows)
        print("\n   Giacomini-White (instruments known at t-1); slope Wald = do the new instruments predict the daily gain?")
        for r in gwt[gwt.benchmark == BENCH].itertuples():
            print(f"    {r.target:10s} {r.instrument_family:32s} GW chi2({r.gw_df})={r.gw_chi2:7.2f} p={r.gw_p:.3g}   "
                  f"slope chi2({r.slope_wald_df})={r.slope_wald_chi2:6.2f} p={r.slope_wald_p:.3g}")
    if mz_rows:
        print("\n2. Mincer-Zarnowitz (HAC, scores summed within day):")
        print(pd.DataFrame(mz_rows)[["target", "model", "scale", "intercept", "slope", "slope_se", "wald_joint_0_1_p", "wald_slope_1_p", "r2"]].to_string(index=False))
        print("\n   Encompassing (y = w f_hybrid + (1-w) f_mcsgarch):")
        print(pd.DataFrame(enc_rows).to_string(index=False))
        print("\n   Distribution of y / f:")
        print(pd.DataFrame(ratio_rows).to_string(index=False))
    if loss_rows:
        print("\n3. Alternative losses (gain of hybrid over mcsgarch, one-sided DM p):")
        print(pd.DataFrame(loss_rows)[["target", "loss", "n_days", "gain_pct", "dm_stat", "dm_p_one_sided", "share_days_hybrid_better", "gain_vs_lgbm_base_pct"]].to_string(index=False))
    if spa_rows:
        print(f"\n4. Hansen SPA (stationary bootstrap, {a.draws} draws, mean block {a.block} days, seed {a.seed}):")
        print(pd.DataFrame(spa_rows).drop(columns=["draws", "mean_block_days", "seed"]).to_string(index=False))
    if "5" not in skip:
        print("\n5. Horizon consistency (direct forecast vs the sum of the shorter windows):")
        print(h[["pair", "model", "qlike_direct", "qlike_sum_of_shorter", "gain_of_direct_over_sum_pct", "dm_p_two_sided", "ratio_direct_over_sum_median", "ratio_p10", "ratio_p90"]].to_string(index=False))
    if roll_summ:
        print("\n6. Rolling 63-day gain and runs:")
        print(pd.DataFrame(roll_summ).to_string(index=False))
    info = {"script": "experiments/backtest_extended.py", "index": a.index, "run": str(run), "p116": str(p116), "seed": a.seed, "draws": a.draws,
            "mean_block_days": a.block, "seconds": round(time.time() - t0, 1), "numpy": np.__version__, "pandas": pd.__version__,
            "n_models_in_ladder": len(models), "models": models, "finished_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    print(f"\nWrote {out}/ in {info['seconds']}s")


if __name__ == "__main__":
    main()
