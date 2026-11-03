"""P107: one pooled test of the forecasting hybrid over Nifty, Bank Nifty and Fin Nifty.

Registered plan: docs/prereg/P107_POOLED_THREE_INDEX.md. Results: results/p107_pooled_test/.
Write-up: docs/P107_POOLED_THREE_INDEX.md. Done by an AI agent (Claude) for Shreyan Thanki.

The hybrid (`hybrid_base`: mcsGARCH times a LightGBM correction) beat mcsGARCH at 15 and 30 minutes on each of
three NSE indices, each tested on its own (Nifty v2, Bank Nifty P91, Fin Nifty P104). The three samples share
dates, news, India VIX and NSE's participant file, so they are not independent. This script

* restricts the committed forecasts of the three runs to their common test dates,
* forms each index's DAILY mean QLIKE loss differential d_it = QLIKE(benchmark) - QLIKE(hybrid_base)
  (positive = the hybrid is better), exactly as the project's DM test does for one index,
* pools the three with Driscoll-Kraay (1998) standard errors, which allow any correlation between the indices on
  the same day and serial correlation over neighbouring days,
* adds a joint stationary bootstrap over dates, a Wald test that the three gains are equal, the pooled last hour,
  pooled expiry days and the pooled edge over `mcsgarch_iv`,
* and measures how correlated the three indices' daily differentials are and how many independent indices they are
  worth ("effective number of indices").

Nothing is refitted: the inputs are the committed files `forecasts_next15.csv`, `forecasts_next30.csv` and
`forecasts_last_hour.csv` of `results/pipeline_nse_v2`, `results/pipeline_nse_banknifty` and
`results/pipeline_nse_finnifty` (REAL DATA).

Usage, from the repository root:

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p107_pooled_test.py count   # phase 1: dates only, no loss
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p107_pooled_test.py run     # phase 2: every registered output

The pure functions at the top (Driscoll-Kraay, Wald, effective numbers, stationary bootstrap) are unit-tested on
SYNTHETIC data in tests/test_p107_pooled_test.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import t as student_t

from niftygex.evaluation import daily_losses, diebold_mariano, holm

ROOT = Path(__file__).resolve().parents[1]

# (key, display name, committed pipeline folder). The order fixes the index order everywhere (Nifty first).
INDICES = [
    ("nifty", "Nifty", "results/pipeline_nse_v2"),
    ("banknifty", "Bank Nifty", "results/pipeline_nse_banknifty"),
    ("finnifty", "Fin Nifty", "results/pipeline_nse_finnifty"),
]
TARGETS = ["next15", "next30", "last_hour"]
TARGET_LABEL = {"next15": "next 15 minutes", "next30": "next 30 minutes", "last_hour": "last hour (14:30 to 15:30)"}
BENCH = "mcsgarch"
MODEL = "hybrid_base"
MATCHED = "mcsgarch_iv"
MODELS = [BENCH, MODEL, MATCHED]

SEED = 107
N_DRAWS = 9_999
MEAN_BLOCK = 10           # registered mean block length of the stationary bootstrap (trading days)
ROBUST_BLOCKS = (5, 20)   # robustness block lengths
SHIFT_PLACEBO = {"banknifty": 21, "finnifty": 42}   # circular shifts (trading days) for the placebo correlation
OUT = ROOT / "results" / "p107_pooled_test"


# ---------------------------------------------------------------------------------------------------------------
# Pure statistics (tested on synthetic data)
# ---------------------------------------------------------------------------------------------------------------

def dk_lag(n_periods: int) -> int:
    """Bandwidth rule: floor(4 (T/100)^(2/9)), the rule of Hoechle (2007, Stata xtscc) and of the project's DM test."""
    return int(np.floor(4 * (n_periods / 100) ** (2 / 9)))


def bartlett_weights(lag: int) -> np.ndarray:
    """Bartlett (Newey-West) kernel weights w_j = 1 - j/(lag + 1), j = 0..lag."""
    return 1.0 - np.arange(lag + 1) / (lag + 1.0)


def kernel_sum(H: np.ndarray, lag: int) -> np.ndarray:
    """S = sum_t h_t h_t' + sum_{j=1..lag} w_j sum_t (h_t h_{t-j}' + h_{t-j} h_t') for a (T x k) array H.

    No division by T. With H the per-period sums of the regression scores this is the middle of the
    Driscoll-Kraay sandwich.
    """
    H = np.asarray(H, float)
    if H.ndim == 1:
        H = H[:, None]
    w = bartlett_weights(lag)
    S = H.T @ H
    for j in range(1, min(lag, H.shape[0] - 1) + 1):
        G = H[j:].T @ H[:-j]
        S = S + w[j] * (G + G.T)
    return S


def driscoll_kraay_cov(X, resid, time_code, lag: int) -> np.ndarray:
    """Driscoll and Kraay (1998) covariance of OLS coefficients in a (possibly unbalanced) panel.

    ``time_code`` holds integer period codes 0..T-1 for each observation; every period must hold at least one
    observation. The scores x_it * u_it are summed over the units observed in each period, and a Newey-West
    (Bartlett) kernel with ``lag`` lags is applied to those period sums: V = (X'X)^-1 S (X'X)^-1. No small-sample
    factor (as Stata's xtscc and statsmodels' ``hac-groupsum`` with ``use_correction=False``).
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    u = np.asarray(resid, float)
    tc = np.asarray(time_code, int)
    T = int(tc.max()) + 1
    H = np.zeros((T, X.shape[1]))
    np.add.at(H, tc, X * u[:, None])
    S = kernel_sum(H, lag)
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


def panel_ols_dk(y, X, time_code, lag: int | None = None) -> dict:
    """Pooled OLS of y on X with the Driscoll-Kraay covariance. Default lag: ``dk_lag`` of the number of periods."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    T = int(np.asarray(time_code).max()) + 1
    lag = dk_lag(T) if lag is None else int(lag)
    return {"beta": beta, "cov": driscoll_kraay_cov(X, u, time_code, lag), "T": T, "lag": lag}


def index_means_dk(D, observed=None, lag: int | None = None) -> dict:
    """Per-index means of a (dates x indices) panel and their Driscoll-Kraay covariance.

    This is OLS of d_it on one dummy per index, with the Driscoll-Kraay covariance, written out directly: each
    index's mean is taken over its own observed dates, the residuals are d_it minus that mean (zero where index i is
    not observed), and the kernel runs over the dates on which at least one index is observed (rows with no
    observation are dropped, so for an unbalanced subset the time axis is the union of the indices' dates).
    For a balanced panel and equal weights, a' V a is exactly the Newey-West variance of the mean of the
    cross-sectional average series (tested).
    """
    D = np.asarray(D, float)
    if D.ndim == 1:
        D = D[:, None]
    obs = np.isfinite(D) if observed is None else (np.asarray(observed, bool) & np.isfinite(D))
    keep = obs.any(axis=1)
    D, obs = D[keep], obs[keep]
    T = D.shape[0]
    n = obs.sum(axis=0).astype(float)
    if np.any(n == 0):
        raise ValueError("an index has no observation")
    means = np.where(obs, D, 0.0).sum(axis=0) / n
    U = np.where(obs, D - means, 0.0)
    lag = dk_lag(T) if lag is None else int(lag)
    S = kernel_sum(U, lag)
    return {"means": means, "cov": S / np.outer(n, n), "T": T, "lag": lag, "n": n.astype(int), "S": S}


def combo_test(means, cov, a, n_periods: int, alternative: str = "greater", hln: bool = True) -> dict:
    """t test of the linear combination a'mu of the index means.

    Same small-sample treatment as the project's DM test (niftygex.evaluation.diebold_mariano): the statistic is
    multiplied by the Harvey-Leybourne-Newbold factor sqrt((T - 1)/T) for one-step forecasts and compared with a
    Student t with T - 1 degrees of freedom, T = number of dates. With one index this is that DM test exactly.
    """
    a = np.asarray(a, float)
    est = float(a @ np.asarray(means, float))
    se = float(np.sqrt(max(a @ np.asarray(cov, float) @ a, 1e-300)))
    stat = est / se
    T = int(n_periods)
    if hln:
        stat *= np.sqrt((T - 1) / T)
    df = max(T - 1, 1)
    if alternative == "greater":
        p = student_t.sf(stat, df)
    elif alternative == "less":
        p = student_t.cdf(stat, df)
    else:
        p = 2 * student_t.sf(abs(stat), df)
    return {"estimate": est, "se": se, "stat": float(stat), "p": float(p), "df": df}


def contrast_matrix(n_idx: int) -> np.ndarray:
    """(N-1) x N matrix of successive differences mu_1 - mu_2, mu_2 - mu_3, ..."""
    R = np.zeros((n_idx - 1, n_idx))
    for i in range(n_idx - 1):
        R[i, i], R[i, i + 1] = 1.0, -1.0
    return R


def wald_equal(means, cov, n_periods: int, hln: bool = True) -> dict:
    """Wald test that all index means are equal, with the cross-index HAC (Driscoll-Kraay) covariance.

    W = (R mu)' (R V R')^-1 (R mu), multiplied by (T - 1)/T (the square of the HLN factor) so that with two indices
    F = W/q equals the square of the two-sided t test of the difference; F = W/q is compared with F(q, T - 1).
    """
    m = np.asarray(means, float)
    V = np.asarray(cov, float)
    R = contrast_matrix(m.size)
    q = R.shape[0]
    Rm = R @ m
    W = float(Rm @ np.linalg.solve(R @ V @ R.T, Rm))
    T = int(n_periods)
    if hln:
        W *= (T - 1) / T
    F = W / q
    return {"W": W, "F": F, "q": q, "df2": T - 1, "p": float(f_dist.sf(F, q, T - 1)),
            "range": float(m.max() - m.min())}


def long_run_cov(D: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West long-run covariance Omega of a balanced (T x N) panel (S / T of ``kernel_sum``)."""
    D = np.asarray(D, float)
    U = D - D.mean(axis=0)
    return kernel_sum(U, lag) / D.shape[0]


def effective_numbers(R: np.ndarray | None = None, omega: np.ndarray | None = None) -> dict:
    """Effective number of independent indices.

    From a correlation matrix R (N x N):
      * kish:          N^2 / (1'R1) = N / (1 + (N - 1) mean off-diagonal correlation), the design effect of an
                       equally weighted mean of equally noisy series (Kish 1965);
      * participation: (sum lambda)^2 / sum lambda^2 = N^2 / sum rho_ij^2 (eigenvalue participation ratio,
                       Bretherton et al. 1999);
      * nyholt:        1 + (N - 1) (1 - Var(lambda) / N), Var with N - 1 in the denominator (Nyholt 2004);
      * li_ji:         sum f(|lambda|), f(x) = 1[x >= 1] + (x - floor(x)) (Li and Ji 2005).
    From a long-run covariance Omega (N x N):
      * design_lr:     N tr(Omega) / (1' Omega 1): N times the variance of the equally weighted pooled mean if the
                       indices were independent, divided by its actual (Driscoll-Kraay) variance. This is the
                       registered headline: it is the number of independent indices that the pooled test is worth.
      * kish_lr:       kish on the long-run correlation matrix.
      * nifty_equiv_equal: Omega_11 / (a' Omega a), a = 1/N: how many "Nifty samples" of precision the equally
                       weighted pooled mean has (index 1 = Nifty).
      * nifty_equiv_gls:   Omega_11 * 1' Omega^-1 1: the same for the best (GLS) weights if the three gains were equal
                           (pseudo-inverse, so that a singular Omega gives the limit).
    """
    out = {}
    if R is not None:
        R = np.asarray(R, float)
        N = R.shape[0]
        lam = np.sort(np.linalg.eigvalsh(R))[::-1]
        out["kish"] = float(N ** 2 / R.sum())
        out["participation"] = float(lam.sum() ** 2 / np.sum(lam ** 2))
        out["nyholt"] = float(1 + (N - 1) * (1 - np.var(lam, ddof=1) / N)) if N > 1 else 1.0
        al = np.round(np.abs(lam), 10)       # rounding: 2.9999999999999996 must count as 3
        out["li_ji"] = float(np.sum((al >= 1).astype(float) + (al - np.floor(al))))
        for k, v in enumerate(lam):
            out[f"eigenvalue_{k + 1}"] = float(v)
    if omega is not None:
        Om = np.asarray(omega, float)
        N = Om.shape[0]
        one = np.ones(N)
        a = one / N
        out["design_lr"] = float(N * np.trace(Om) / Om.sum())
        sd = np.sqrt(np.diag(Om))
        Rl = Om / np.outer(sd, sd)
        out["kish_lr"] = float(N ** 2 / Rl.sum())
        out["nifty_equiv_equal"] = float(Om[0, 0] / (a @ Om @ a))
        out["nifty_equiv_gls"] = float(Om[0, 0] * (one @ np.linalg.pinv(Om) @ one))
    return out


def stationary_bootstrap_indices(n: int, reps: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis and Romano (1994) stationary bootstrap: a (reps x n) array of date positions in 0..n-1.

    Each draw starts at a uniform position; at every later step it starts a new block at a uniform position with
    probability 1/mean_block and otherwise moves to the next date (wrapping round at the end), so block lengths are
    geometric with mean ``mean_block``. The same positions are used for all three indices (joint resampling).
    """
    q = 1.0 / float(mean_block)
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    new = rng.random((reps, n)) < q
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_p_greater(boot_stat: np.ndarray, est: float) -> float:
    """One-sided bootstrap p for H0: mean <= 0 against mean > 0, from the recentred bootstrap distribution:
    p = (1 + #{b: stat*_b - est >= est}) / (B + 1). The smallest possible value is 1/(B + 1)."""
    b = np.asarray(boot_stat, float)
    return float((1 + np.sum(b - est >= est)) / (b.size + 1))


def boot_index_means(D: np.ndarray, idx: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """Means of every index column over each bootstrap draw of dates: (reps x N)."""
    D = np.asarray(D, float)
    out = np.empty((idx.shape[0], D.shape[1]))
    for s in range(0, idx.shape[0], chunk):
        out[s:s + chunk] = D[idx[s:s + chunk]].mean(axis=1)
    return out


def boot_dependence(D: np.ndarray, idx: np.ndarray, lag: int, chunk: int = 500) -> dict:
    """For each bootstrap draw: the Pearson correlation matrix and the Newey-West long-run covariance of the
    resampled (T x N) panel. Returns arrays (reps x N x N)."""
    D = np.asarray(D, float)
    reps, T = idx.shape
    N = D.shape[1]
    corr = np.empty((reps, N, N))
    omega = np.empty((reps, N, N))
    w = bartlett_weights(lag)
    for s in range(0, reps, chunk):
        X = D[idx[s:s + chunk]]                       # c x T x N
        X = X - X.mean(axis=1, keepdims=True)
        G0 = np.einsum("ctn,ctm->cnm", X, X)
        sd = np.sqrt(np.einsum("cnn->cn", G0))
        corr[s:s + chunk] = G0 / (sd[:, :, None] * sd[:, None, :])
        S = G0.copy()
        for j in range(1, lag + 1):
            G = np.einsum("ctn,ctm->cnm", X[:, j:], X[:, :-j])
            S += w[j] * (G + np.transpose(G, (0, 2, 1)))
        omega[s:s + chunk] = S / T
    return {"corr": corr, "omega": omega}


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel_path(path: Path) -> str:
    """Path relative to the repository root when it lies inside it (so that run_info.json has no machine paths)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def common_dates(dirs: dict, targets=TARGETS) -> list:
    """Dates present in every index's forecast file, for every target (reads the date column only)."""
    sets = []
    for t in targets:
        for d in dirs.values():
            sets.append(set(pd.read_csv(Path(d) / f"forecasts_{t}.csv", usecols=["date"])["date"]))
    return sorted(set.intersection(*sets))


def count_only(dirs: dict) -> dict:
    """Phase-1 counts: common dates, anchors and expiry days. Reads date, minute, is_expiry and regime only."""
    out = {}
    common = common_dates(dirs)
    out["common_dates"] = len(common)
    out["first"], out["last"] = common[0], common[-1]
    exp = {}
    for t in TARGETS:
        mins = {}
        for k, d in dirs.items():
            f = pd.read_csv(Path(d) / f"forecasts_{t}.csv", usecols=["date", "minute", "is_expiry", "regime"])
            out[f"{k}_{t}_test_days"] = int(f["date"].nunique())
            f = f[f["date"].isin(common)]
            mins[k] = f.groupby("date")["minute"].apply(tuple)
            out[f"{k}_{t}_anchors"] = int(len(f))
            e = f.groupby("date")["is_expiry"].first().astype(bool)
            exp[(k, t)] = set(e[e].index)
            out[f"{k}_expiry_days"] = int(e.sum())
            out["regime_counts"] = {int(r): int(c) for r, c in f.groupby("date")["regime"].first().value_counts().sort_index().items()}
        keys = list(dirs)
        out[f"same_anchor_minutes_{t}"] = bool(all((mins[keys[0]] == mins[k]).all() for k in keys[1:]))
    E = {k: exp[(k, "next15")] for k in dirs}
    keys = list(dirs)
    out["expiry_union_dates"] = len(set.union(*E.values()))
    out["expiry_index_days"] = int(sum(len(v) for v in E.values()))
    out["expiry_all_three"] = len(set.intersection(*E.values()))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            out[f"expiry_both_{keys[i]}_{keys[j]}"] = len(E[keys[i]] & E[keys[j]])
    return out


def bounds_from_committed(ref_file: Path) -> pd.DataFrame:
    """What the committed per-index results on the common dates already imply for the pooled t statistic.

    Reads P104's committed table (one row per index, target and subset: both mean QLIKEs and the DM statistic). For
    each index the standard error of its mean difference is s_i = mean_diff_i / t_i. For the equally weighted pooled
    mean, the pooled t is
      * at least sum(mu) / sum(s) (perfectly correlated indices; this is never below the smallest per-index t), and
      * sum(mu) / sqrt(sum(s^2)) if the indices were independent;
    and the design-effect effective number equals 3 (t_pooled / t_independent)^2. No new loss is computed.
    For expiry days the three indices use different day sets, so the bounds are only approximate there.
    """
    ref = pd.read_csv(ref_file)
    ref["sub"] = np.where(ref["subset"] == "all", "all", "expiry (own days)")
    order = {"Nifty (v2 forecasts, same dates)": 0, "Bank Nifty (P91 forecasts, same dates)": 1, "Fin Nifty": 2}
    ref["order"] = ref["index"].map(order)
    rows = []
    for (target, subset), g in ref.groupby(["target", "sub"], sort=False):
        g = g.sort_values("order")
        mu = (g["qlike_mcsgarch"] - g["qlike_model"]).to_numpy()
        tt = g["dm_stat"].to_numpy()
        s = mu / tt
        rows.append({"target": target, "subset": subset,
                     "t_nifty": tt[0], "t_banknifty": tt[1], "t_finnifty": tt[2], "min_t": tt.min(),
                     "pooled_t_if_perfectly_correlated": mu.sum() / s.sum(),
                     "pooled_t_if_independent": mu.sum() / np.sqrt(np.sum(s ** 2)),
                     "approximate": subset != "all"})
    return pd.DataFrame(rows)


def load_daily(dirs: dict, target: str, dates: list) -> dict:
    """Daily mean QLIKE of mcsGARCH, hybrid_base and mcsgarch_iv and the expiry flag, per index, on ``dates``."""
    out = {}
    minutes = {}
    for k, d in dirs.items():
        f = pd.read_csv(Path(d) / f"forecasts_{target}.csv", usecols=["date", "minute", "is_expiry", "y", *MODELS])
        f = f[f["date"].isin(dates)].sort_values(["date", "minute"]).reset_index(drop=True)
        minutes[k] = f.groupby("date")["minute"].apply(tuple)
        L = daily_losses(f, MODELS, "qlike")
        L["is_expiry"] = f.groupby("date")["is_expiry"].first().astype(bool)
        L["anchors"] = f.groupby("date").size()
        out[k] = L.loc[dates]
    keys = list(dirs)
    for k in keys[1:]:
        if not (minutes[keys[0]] == minutes[k]).all():
            raise ValueError(f"anchor minutes differ between {keys[0]} and {k} for target {target}")
    return out


def diff_matrix(daily: dict, bench: str, model: str = MODEL) -> np.ndarray:
    """(T x N) matrix of d_it = QLIKE(bench) - QLIKE(model), positive = model better."""
    return np.column_stack([daily[k][bench].to_numpy() - daily[k][model].to_numpy() for k in daily])


def expiry_mask(daily: dict) -> np.ndarray:
    return np.column_stack([daily[k]["is_expiry"].to_numpy(bool) for k in daily])


def bench_level(daily: dict, bench: str, mask: np.ndarray | None = None) -> np.ndarray:
    """Each index's mean daily QLIKE of the benchmark (over its own observed days)."""
    out = []
    for i, k in enumerate(daily):
        v = daily[k][bench].to_numpy()
        out.append(float(v[mask[:, i]].mean()) if mask is not None else float(v.mean()))
    return np.array(out)


# ---------------------------------------------------------------------------------------------------------------
# The registered analysis
# ---------------------------------------------------------------------------------------------------------------

def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def script_differs() -> bool | None:
    """True if this script has uncommitted changes relative to HEAD."""
    try:
        r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__).resolve().relative_to(ROOT))],
                           cwd=ROOT, capture_output=True)
        return r.returncode != 0
    except Exception:  # pragma: no cover
        return None


def dk_method(T: int, lag: int) -> str:
    return (f"Driscoll-Kraay t (Bartlett kernel, lag {lag}, T = {T} dates, HLN factor sqrt((T-1)/T), "
            f"Student t with {T - 1} df)")


def run(dirs: dict, out: Path, label: str = "REAL DATA", draws: int = N_DRAWS, plan_commit: str | None = None,
        verbose: bool = True) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dirs)
    names = {k: n for k, n, _ in INDICES} if set(keys) == {k for k, _, _ in INDICES} else {k: k for k in keys}
    N = len(keys)
    log_lines: list[str] = []

    def log(s=""):
        log_lines.append(s)
        if verbose:
            print(s)

    dates = common_dates(dirs)
    T_all = len(dates)
    daily = {t: load_daily(dirs, t, dates) for t in TARGETS}
    log(f"{label}. P107 pooled test over {', '.join(names[k] for k in keys)}: {T_all} common dates, "
        f"{dates[0]} to {dates[-1]}.")

    tests: list[dict] = []

    def add(test_id, hypothesis, role, sample, estimate, statistic, p_raw, sidedness, p_method, family=None):
        tests.append({"test_id": test_id, "hypothesis": hypothesis, "role": role, "sample": sample,
                      "estimate": estimate, "statistic": statistic, "p_raw": p_raw, "sidedness": sidedness,
                      "p_method": p_method, "family": family})

    sample_all = (f"{T_all} common test dates, {dates[0]} to {dates[-1]}, all {N} indices "
                  f"({N * T_all:,} index-days)")
    a_eq = np.full(N, 1.0 / N)
    pooled_rows = []     # descriptive table of every pooled comparison
    se_rows = []
    per_index_rows = []
    boot_rows = []

    # --- per-index numbers (descriptive; the per-index tests against mcsGARCH are already committed) -----------
    for t in TARGETS:
        dl = daily[t]
        emask = expiry_mask(dl)
        for bench in (BENCH, MATCHED):
            for sub in ("all", "expiry"):
                for i, k in enumerate(keys):
                    L = dl[k] if sub == "all" else dl[k][emask[:, i]]
                    dm = diebold_mariano(L[bench], L[MODEL], alternative="greater")
                    per_index_rows.append({
                        "data": label, "target": t, "benchmark": bench, "subset": sub, "index": names[k],
                        "days": dm["n"], "qlike_benchmark": float(L[bench].mean()), "qlike_hybrid": float(L[MODEL].mean()),
                        "mean_diff": dm["mean_diff"], "change_pct": 100 * (L[MODEL].mean() / L[bench].mean() - 1),
                        "dm_stat": dm["stat"], "dm_lag": dk_lag(dm["n"]),
                        "dm_p": dm["p"]})
    per_index = pd.DataFrame(per_index_rows)

    # check against P104's committed same-dates table (REAL DATA only)
    check_rows = []
    ref_file = ROOT / "results/pipeline_nse_finnifty/p104/other_indices_same_dates.csv"
    if label == "REAL DATA" and ref_file.exists():
        ref = pd.read_csv(ref_file)
        name_map = {"Nifty (v2 forecasts, same dates)": "Nifty", "Bank Nifty (P91 forecasts, same dates)": "Bank Nifty",
                    "Fin Nifty": "Fin Nifty"}
        for r in ref.to_dict("records"):
            sub = "all" if r["subset"] == "all" else "expiry"
            nm = name_map[r["index"]]
            mine = per_index[(per_index.target == r["target"]) & (per_index.benchmark == BENCH) &
                             (per_index.subset == sub) & (per_index["index"] == nm)].iloc[0]
            check_rows.append({"index": nm, "target": r["target"], "subset": sub,
                               "days_p104": r["days"], "days_p107": int(mine["days"]),
                               "change_pct_p104": r["change_pct"], "change_pct_p107": float(mine["change_pct"]),
                               "dm_stat_p104": r["dm_stat"], "dm_stat_p107": float(mine["dm_stat"]),
                               "match": bool(r["days"] == mine["days"] and abs(r["dm_stat"] - mine["dm_stat"]) < 1e-8
                                             and abs(r["change_pct"] - mine["change_pct"]) < 1e-8)})
        chk = pd.DataFrame(check_rows)
        chk.to_csv(out / "check_against_p104.csv", index=False)
        log(f"Check against p104/other_indices_same_dates.csv: {int(chk.match.sum())} of {len(chk)} rows match.")
        if not chk.match.all():   # plan Section 9: stop before any pooled number is written
            raise RuntimeError("per-index numbers do not match P104's committed same-dates table; see "
                               "check_against_p104.csv")

    # --- helper: one pooled DK comparison ---------------------------------------------------------------------
    def pooled(t, bench, sub="all", weights="equal", lag=None, cols=None):
        dl = daily[t]
        D = diff_matrix(dl, bench)
        mask = expiry_mask(dl) if sub == "expiry" else np.ones_like(D, bool)
        c = list(range(N)) if cols is None else cols
        D, mask = D[:, c], mask[:, c]
        res = index_means_dk(D, observed=mask, lag=lag)
        lev = bench_level({keys[j]: dl[keys[j]] for j in c}, bench, mask)
        if weights == "scaled":
            a = 1.0 / (len(c) * lev)
        else:
            a = np.full(len(c), 1.0 / len(c))
        test = combo_test(res["means"], res["cov"], a, res["T"])
        pct = -100 * res["means"].sum() / lev.sum() if weights == "equal" else -100 * float(a @ res["means"])
        return res, test, lev, pct, D, mask

    # --- P1 primary: pooled raw QLIKE differential, mcsGARCH minus hybrid, 15 and 30 minutes ----------------
    log("\nPrimary (P1): pooled mean daily QLIKE differential, mcsGARCH minus hybrid_base, Driscoll-Kraay.")
    prim = {}
    for t in ("next15", "next30"):
        res, test, lev, pct, D, _ = pooled(t, BENCH)
        prim[t] = (res, test, lev, pct, D)
        add(f"P1_{t}", f"Pooled over the three indices, the hybrid has a lower mean daily QLIKE than mcsGARCH for the "
            f"{TARGET_LABEL[t]} (estimate: equally weighted mean of the three indices' mean daily QLIKE differences, "
            f"mcsGARCH minus hybrid; positive = hybrid better)", "primary", sample_all, test["estimate"], test["stat"],
            test["p"], "one-sided (hybrid better)", dk_method(res["T"], res["lag"]), family="P1")
        log(f"  {t}: pooled diff {test['estimate']:.6f} (pooled change {pct:+.2f}%), DK se {test['se']:.6f}, "
            f"t {test['stat']:.2f}, p {test['p']:.3g}, lag {res['lag']}, T {res['T']}")
        pooled_rows.append({"test_id": f"P1_{t}", "target": t, "benchmark": BENCH, "subset": "all", "weights": "equal",
                            "indices": "all three", "T": res["T"], "lag": res["lag"],
                            **{f"mean_diff_{names[k]}": res["means"][i] for i, k in enumerate(keys)},
                            **{f"bench_qlike_{names[k]}": lev[i] for i, k in enumerate(keys)},
                            "pooled_diff": test["estimate"], "pooled_change_pct": pct, "dk_se": test["se"],
                            "stat": test["stat"]})
        # SE comparison: DK, independence across indices, naive iid over all index-days
        V = res["cov"]
        se_dk = test["se"]
        se_ind = float(np.sqrt(np.sum(np.diag(V)) / N ** 2))
        flat = D.ravel()
        se_iid = float(flat.std(ddof=1) / np.sqrt(flat.size))
        se_rows.append({"target": t, "pooled_diff": test["estimate"], "se_driscoll_kraay": se_dk,
                        "se_if_indices_independent": se_ind, "se_naive_iid_index_days": se_iid,
                        "t_driscoll_kraay_no_hln": test["estimate"] / se_dk,
                        "t_if_indices_independent": test["estimate"] / se_ind,
                        "t_naive_iid": test["estimate"] / se_iid,
                        "variance_ratio_dk_over_independent": se_dk ** 2 / se_ind ** 2,
                        "variance_ratio_dk_over_naive": se_dk ** 2 / se_iid ** 2})

    # --- S1a joint stationary bootstrap ------------------------------------------------------------------------
    log(f"\nSecondary (S1a): joint stationary bootstrap over dates, {draws} draws.")
    boot_idx = {}
    for b in (MEAN_BLOCK, *ROBUST_BLOCKS):
        boot_idx[b] = stationary_bootstrap_indices(T_all, draws, b, np.random.default_rng([SEED, b]))
    boot_store = {}
    for t in ("next15", "next30"):
        res, test, lev, pct, D = prim[t]
        Lb = np.column_stack([daily[t][k][BENCH].to_numpy() for k in keys])
        for b in (MEAN_BLOCK, *ROBUST_BLOCKS):
            mb = boot_index_means(D, boot_idx[b])
            lb = boot_index_means(Lb, boot_idx[b])
            pooled_b = mb @ a_eq
            pct_b = -100 * mb.sum(axis=1) / lb.sum(axis=1)
            p = bootstrap_p_greater(pooled_b, test["estimate"])
            role = "secondary" if b == MEAN_BLOCK else "robustness"
            tid = f"S1a_{t}" if b == MEAN_BLOCK else f"R3_block{b}_{t}"
            add(tid, f"Pooled over the three indices, the hybrid has a lower mean daily QLIKE than mcsGARCH for the "
                f"{TARGET_LABEL[t]}, tested with a joint bootstrap over dates (mean block {b} days)", role, sample_all,
                test["estimate"], float(test["estimate"] / pooled_b.std(ddof=1)), p, "one-sided (hybrid better)",
                f"joint stationary bootstrap over dates (all three indices resampled together; mean block {b} days, "
                f"{draws} draws, numpy default_rng([{SEED}, {b}])); p = (1 + #(m* - m >= m)) / (B + 1); "
                f"statistic = estimate / bootstrap sd",
                family="S1a" if b == MEAN_BLOCK else None)
            boot_rows.append({"target": t, "mean_block": b, "draws": draws, "pooled_diff": test["estimate"],
                              "boot_sd": float(pooled_b.std(ddof=1)),
                              "ci95_lo": float(np.quantile(pooled_b, 0.025)), "ci95_hi": float(np.quantile(pooled_b, 0.975)),
                              "pooled_change_pct": pct, "change_pct_ci95_lo": float(np.quantile(pct_b, 0.025)),
                              "change_pct_ci95_hi": float(np.quantile(pct_b, 0.975)), "p_one_sided": p,
                              "draws_below_zero": int(np.sum(pooled_b <= 0))})
            if b == MEAN_BLOCK:
                boot_store[t] = mb
            log(f"  {t}, block {b}: boot sd {pooled_b.std(ddof=1):.6f} (DK se {test['se']:.6f}), p {p:.4g}")

    # --- S1b homogeneity --------------------------------------------------------------------------------------
    log("\nSecondary (S1b): Wald test that the three mean differentials are equal.")
    for t in ("next15", "next30"):
        res = prim[t][0]
        w = wald_equal(res["means"], res["cov"], res["T"])
        add(f"S1b_{t}", f"The three indices' mean daily QLIKE gains of the hybrid over mcsGARCH for the {TARGET_LABEL[t]} "
            f"differ (null: all three equal; estimate: largest minus smallest index mean difference)", "secondary",
            sample_all, w["range"], w["F"], w["p"], "non-directional (Wald)",
            f"Wald F({w['q']}, {w['df2']}) with the Driscoll-Kraay (cross-index HAC) covariance of the three index "
            f"means (lag {res['lag']}), W times (T-1)/T", family="S1b")
        log(f"  {t}: means {np.round(res['means'], 6)}, F {w['F']:.3f}, p {w['p']:.4g}")

    # --- S1c pooled last hour ----------------------------------------------------------------------------------
    log("\nSecondary (S1c): pooled last hour.")
    res, test, lev, pct, D, _ = pooled("last_hour", BENCH)
    add("S1c_last_hour", "Pooled over the three indices, the hybrid has a lower mean QLIKE than mcsGARCH in the last "
        "hour (14:30 to 15:30) (estimate: equally weighted mean of the three mean differences)", "secondary",
        sample_all, test["estimate"], test["stat"], test["p"], "one-sided (hybrid better)", dk_method(res["T"], res["lag"]),
        family="S1c")
    pooled_rows.append({"test_id": "S1c_last_hour", "target": "last_hour", "benchmark": BENCH, "subset": "all",
                        "weights": "equal", "indices": "all three", "T": res["T"], "lag": res["lag"],
                        **{f"mean_diff_{names[k]}": res["means"][i] for i, k in enumerate(keys)},
                        **{f"bench_qlike_{names[k]}": lev[i] for i, k in enumerate(keys)},
                        "pooled_diff": test["estimate"], "pooled_change_pct": pct, "dk_se": test["se"], "stat": test["stat"]})
    log(f"  last hour: pooled change {pct:+.2f}%, t {test['stat']:.2f}, p {test['p']:.3g}")

    # --- S1d pooled expiry days --------------------------------------------------------------------------------
    log("\nSecondary (S1d): pooled expiry days (each index's own expiry days, equal weight per index).")
    for t in ("next15", "next30", "last_hour"):
        res, test, lev, pct, D, mask = pooled(t, BENCH, sub="expiry")
        n_by = ", ".join(f"{names[k]} {res['n'][i]}" for i, k in enumerate(keys))
        samp = (f"each index's own expiry days among the {T_all} common dates ({n_by}; {int(res['n'].sum())} "
                f"index-days on {res['T']} dates)")
        if t == "last_hour":
            tid, role, fam = "E4_expiry_last_hour", "exploratory", None
        else:
            tid, role, fam = f"S1d_{t}", "secondary", "S1d"
        add(tid, f"Pooled over the three indices, on each index's own expiry days, the hybrid has a lower mean QLIKE "
            f"than mcsGARCH for the {TARGET_LABEL[t]} (estimate: equally weighted mean of the three indices' mean "
            f"expiry-day differences)", role, samp, test["estimate"], test["stat"], test["p"],
            "one-sided (hybrid better)", dk_method(res["T"], res["lag"]) + "; unbalanced panel, index-dummy means",
            family=fam)
        pooled_rows.append({"test_id": tid, "target": t, "benchmark": BENCH, "subset": "expiry", "weights": "equal",
                            "indices": "all three", "T": res["T"], "lag": res["lag"],
                            **{f"mean_diff_{names[k]}": res["means"][i] for i, k in enumerate(keys)},
                            **{f"bench_qlike_{names[k]}": lev[i] for i, k in enumerate(keys)},
                            "pooled_diff": test["estimate"], "pooled_change_pct": pct, "dk_se": test["se"],
                            "stat": test["stat"]})
        log(f"  {t}: n {res['n']}, T {res['T']}, lag {res['lag']}, pooled change {pct:+.2f}%, t {test['stat']:.2f}, "
            f"p {test['p']:.3g}")
        if t != "last_hour":
            # R4: equal weight per index-day (pooled OLS on a constant over all expiry index-days)
            dl = daily[t]
            Dfull = diff_matrix(dl, BENCH)
            m_ = expiry_mask(dl)
            keep = m_.any(axis=1)
            Dk, mk = Dfull[keep], m_[keep]
            tt, ii = np.nonzero(mk)
            r4 = panel_ols_dk(Dk[tt, ii], np.ones(tt.size), tt)
            c4 = combo_test(r4["beta"], r4["cov"], [1.0], r4["T"])
            add(f"R4_{t}", f"As S1d for the {TARGET_LABEL[t]}, but every expiry index-day weighted equally (pooled "
                f"mean over all expiry index-days)", "robustness", samp, c4["estimate"], c4["stat"], c4["p"],
                "one-sided (hybrid better)", dk_method(r4["T"], r4["lag"]) + "; pooled OLS on a constant")

    # --- S1e pooled edge over mcsgarch_iv ----------------------------------------------------------------------
    log("\nSecondary (S1e): pooled edge over mcsgarch_iv.")
    for t in ("next15", "next30", "last_hour"):
        res, test, lev, pct, D, _ = pooled(t, MATCHED)
        if t == "last_hour":
            tid, role, fam = "E3_iv_last_hour", "exploratory", None
        else:
            tid, role, fam = f"S1e_{t}", "secondary", "S1e"
        add(tid, f"Pooled over the three indices, the hybrid has a lower mean daily QLIKE than mcsgarch_iv (mcsGARCH "
            f"with an option-implied daily level) for the {TARGET_LABEL[t]} (estimate: equally weighted mean of the "
            f"three mean differences, mcsgarch_iv minus hybrid)", role, sample_all, test["estimate"], test["stat"],
            test["p"], "one-sided (hybrid better)", dk_method(res["T"], res["lag"]), family=fam)
        pooled_rows.append({"test_id": tid, "target": t, "benchmark": MATCHED, "subset": "all", "weights": "equal",
                            "indices": "all three", "T": res["T"], "lag": res["lag"],
                            **{f"mean_diff_{names[k]}": res["means"][i] for i, k in enumerate(keys)},
                            **{f"bench_qlike_{names[k]}": lev[i] for i, k in enumerate(keys)},
                            "pooled_diff": test["estimate"], "pooled_change_pct": pct, "dk_se": test["se"],
                            "stat": test["stat"]})
        log(f"  {t}: pooled change vs mcsgarch_iv {pct:+.2f}%, t {test['stat']:.2f}, p {test['p']:.3g}")

    # --- robustness: scaled differentials and a doubled bandwidth -----------------------------------------------
    log("\nRobustness: scaled differentials (R1), doubled bandwidth (R2).")
    for t in ("next15", "next30"):
        res, test, lev, pct, D, _ = pooled(t, BENCH, weights="scaled")
        add(f"R1_{t}", f"As P1 for the {TARGET_LABEL[t]}, with each index's differences divided by its own mcsGARCH mean "
            f"QLIKE (estimate: average proportional gain over the three indices)", "robustness", sample_all,
            test["estimate"], test["stat"], test["p"], "one-sided (hybrid better)",
            dk_method(res["T"], res["lag"]) + "; weights 1/(3 x mean mcsGARCH QLIKE of the index), treated as fixed")
        pooled_rows.append({"test_id": f"R1_{t}", "target": t, "benchmark": BENCH, "subset": "all", "weights": "scaled",
                            "indices": "all three", "T": res["T"], "lag": res["lag"],
                            **{f"mean_diff_{names[k]}": res["means"][i] for i, k in enumerate(keys)},
                            **{f"bench_qlike_{names[k]}": lev[i] for i, k in enumerate(keys)},
                            "pooled_diff": test["estimate"], "pooled_change_pct": pct, "dk_se": test["se"],
                            "stat": test["stat"]})
        log(f"  R1 {t}: average proportional gain {100 * test['estimate']:.2f}%, t {test['stat']:.2f}, p {test['p']:.3g}")
        lag2 = 2 * dk_lag(T_all)
        res, test, lev, pct, D, _ = pooled(t, BENCH, lag=lag2)
        add(f"R2_{t}", f"As P1 for the {TARGET_LABEL[t]}, with twice the registered bandwidth", "robustness", sample_all,
            test["estimate"], test["stat"], test["p"], "one-sided (hybrid better)", dk_method(res["T"], res["lag"]))
        log(f"  R2 {t}: lag {lag2}, t {test['stat']:.2f}, p {test['p']:.3g}")

    # --- exploratory --------------------------------------------------------------------------------------------
    log("\nExploratory.")
    if N == 3:
        for t in ("next15", "next30"):
            res, test, lev, pct, D, _ = pooled(t, BENCH, cols=[1, 2])
            add(f"E1_{t}", f"Pooled over Bank Nifty and Fin Nifty only (the two indices the hybrid was not designed on), "
                f"the hybrid has a lower mean daily QLIKE than mcsGARCH for the {TARGET_LABEL[t]}", "exploratory",
                f"{T_all} common test dates, Bank Nifty and Fin Nifty ({2 * T_all:,} index-days)", test["estimate"],
                test["stat"], test["p"], "one-sided (hybrid better)", dk_method(res["T"], res["lag"]))
            log(f"  E1 {t}: Bank+Fin pooled change {pct:+.2f}%, t {test['stat']:.2f}, p {test['p']:.3g}")
    for t in TARGETS:
        for i, k in enumerate(keys):
            r = per_index[(per_index.target == t) & (per_index.benchmark == MATCHED) & (per_index.subset == "all") &
                          (per_index["index"] == names[k])].iloc[0]
            add(f"E2_{k}_{t}", f"On {names[k]} alone, the hybrid has a lower mean daily QLIKE than mcsgarch_iv for the "
                f"{TARGET_LABEL[t]} (estimate: mean daily difference, mcsgarch_iv minus hybrid)", "exploratory",
                f"{T_all} common test dates, {names[k]} only", float(r["mean_diff"]), float(r["dm_stat"]),
                float(r["dm_p"]), "one-sided (hybrid better)",
                f"Diebold-Mariano (niftygex.evaluation.diebold_mariano: Newey-West lag {int(r['dm_lag'])}, HLN, "
                f"Student t)")
    dl = daily["last_hour"]
    resl = index_means_dk(diff_matrix(dl, BENCH))
    w = wald_equal(resl["means"], resl["cov"], resl["T"])
    add("E5_homogeneity_last_hour", "The three indices' mean QLIKE gains of the hybrid over mcsGARCH in the last hour "
        "differ (null: all three equal; estimate: largest minus smallest index mean difference)", "exploratory",
        sample_all, w["range"], w["F"], w["p"], "non-directional (Wald)",
        f"Wald F({w['q']}, {w['df2']}) with the Driscoll-Kraay covariance of the three index means (lag {resl['lag']}), "
        f"W times (T-1)/T")
    log(f"  E5 last-hour homogeneity: F {w['F']:.3f}, p {w['p']:.3g}")

    # --- Holm within the registered families -------------------------------------------------------------------
    tdf = pd.DataFrame(tests)
    tdf["p_adjusted"] = np.nan
    tdf["adjust_method"] = "none"
    fam_text = {"P1": "Holm over 15 and 30 minutes (primary family P1)",
                "S1a": "Holm over 15 and 30 minutes (family S1a)", "S1b": "Holm over 15 and 30 minutes (family S1b)",
                "S1d": "Holm over 15 and 30 minutes (family S1d)", "S1e": "Holm over 15 and 30 minutes (family S1e)",
                "S1c": "none (single test)"}
    for fam, g in tdf.groupby("family"):
        if fam == "S1c":
            tdf.loc[g.index, "p_adjusted"] = g["p_raw"].to_numpy()
            tdf.loc[g.index, "adjust_method"] = fam_text[fam]
            continue
        tdf.loc[g.index, "p_adjusted"] = holm(g["p_raw"].to_numpy())
        tdf.loc[g.index, "adjust_method"] = fam_text[fam]
    tdf["data"] = label
    cols = ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method", "data"]
    tdf[cols].to_csv(out / "tests.csv", index=False)

    # --- correlations and effective numbers -------------------------------------------------------------------
    log("\nCorrelation of daily loss differentials and effective number of indices.")
    corr_rows, neff_rows = [], []
    lag_all = dk_lag(T_all)
    for t in TARGETS:
        for bench in (BENCH, MATCHED):
            D = diff_matrix(daily[t], bench)
            R = np.corrcoef(D, rowvar=False)
            Rs = pd.DataFrame(D).corr(method="spearman").to_numpy()
            Om = long_run_cov(D, lag_all)
            sd = np.sqrt(np.diag(Om))
            Rl = Om / np.outer(sd, sd)
            Dsh = D.copy()
            for i, k in enumerate(keys):
                if k in SHIFT_PLACEBO:
                    Dsh[:, i] = np.roll(D[:, i], SHIFT_PLACEBO[k])
            Rsh = np.corrcoef(Dsh, rowvar=False)
            Omsh = long_run_cov(Dsh, lag_all)
            Llev = np.column_stack([daily[t][k][bench].to_numpy() for k in keys])
            Rlev = np.corrcoef(Llev, rowvar=False)
            main = (t in ("next15", "next30") and bench == BENCH)
            bd = None
            if main or (t == "last_hour" and bench == BENCH):
                bd = boot_dependence(D, boot_idx[MEAN_BLOCK], lag_all)
            for kind, M in (("pearson", R), ("spearman", Rs), ("long_run", Rl), ("pearson_placebo_shift", Rsh),
                            ("pearson_benchmark_loss_level", Rlev)):
                for i in range(N):
                    for j in range(i + 1, N):
                        row = {"data": label, "target": t, "benchmark": bench, "kind": kind,
                               "index_a": names[keys[i]], "index_b": names[keys[j]], "value": float(M[i, j]),
                               "ci95_lo": np.nan, "ci95_hi": np.nan}
                        if bd is not None and kind in ("pearson", "long_run"):
                            if kind == "pearson":
                                v = bd["corr"][:, i, j]
                            else:
                                o = bd["omega"]
                                v = o[:, i, j] / np.sqrt(o[:, i, i] * o[:, j, j])
                            row["ci95_lo"], row["ci95_hi"] = float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
                        corr_rows.append(row)
            # effective numbers
            sets = {"all three": list(range(N))}
            if N == 3:
                sets.update({f"{names[keys[i]]} + {names[keys[j]]}": [i, j] for i in range(N) for j in range(i + 1, N)})
            for sname, c in sets.items():
                e = effective_numbers(R[np.ix_(c, c)], Om[np.ix_(c, c)])
                if c[0] != 0:     # "Nifty equivalents" only for sets that contain Nifty (index 0)
                    e.pop("nifty_equiv_equal", None)
                    e.pop("nifty_equiv_gls", None)
                eb = None
                if bd is not None and sname == "all three":
                    eb = {}
                    for r_ in range(bd["corr"].shape[0]):
                        er = effective_numbers(bd["corr"][r_], bd["omega"][r_])
                        for kk, vv in er.items():
                            eb.setdefault(kk, []).append(vv)
                for kk, vv in e.items():
                    row = {"data": label, "target": t, "benchmark": bench, "set": sname, "measure": kk, "value": vv,
                           "ci95_lo": np.nan, "ci95_hi": np.nan, "n_indices": len(c)}
                    if eb is not None and kk in eb:
                        arr = np.asarray(eb[kk])
                        row["ci95_lo"], row["ci95_hi"] = float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))
                    neff_rows.append(row)
            # what each index adds when it is added last (three-index value minus the pair without it)
            if N == 3:
                whole = effective_numbers(R, Om)
                for i, k in enumerate(keys):
                    c = [j for j in range(N) if j != i]
                    part = effective_numbers(R[np.ix_(c, c)], Om[np.ix_(c, c)])
                    for kk in ("design_lr", "kish"):
                        neff_rows.append({"data": label, "target": t, "benchmark": bench,
                                          "set": f"added by {names[k]} to the other two", "measure": kk,
                                          "value": whole[kk] - part[kk], "ci95_lo": np.nan, "ci95_hi": np.nan,
                                          "n_indices": N})
            # bootstrap-variance version of the design effect (point estimate), main comparison only
            if main:
                mb = boot_store[t]
                v_pool = float(np.var(mb @ a_eq, ddof=1))
                v_ind = float(np.sum(np.var(mb, axis=0, ddof=1)) / N ** 2)
                neff_rows.append({"data": label, "target": t, "benchmark": bench, "set": "all three",
                                  "measure": "design_bootstrap", "value": N * v_ind / v_pool, "ci95_lo": np.nan,
                                  "ci95_hi": np.nan, "n_indices": N})
            # placebo: shifted dates
            ep = effective_numbers(Rsh, Omsh)
            for kk in ("kish", "participation", "design_lr"):
                neff_rows.append({"data": label, "target": t, "benchmark": bench, "set": "all three, placebo shift",
                                  "measure": kk, "value": ep[kk], "ci95_lo": np.nan, "ci95_hi": np.nan, "n_indices": N})
            if main or t == "last_hour" and bench == BENCH:
                log(f"  {t} vs {bench}: Pearson {np.round(R[np.triu_indices(N, 1)], 3)}, long-run "
                    f"{np.round(Rl[np.triu_indices(N, 1)], 3)}, N_eff(design, long-run) "
                    f"{effective_numbers(R, Om)['design_lr']:.3f}, Kish {effective_numbers(R)['kish']:.3f}")
    corr_df = pd.DataFrame(corr_rows)
    neff_df = pd.DataFrame(neff_rows)
    corr_df.to_csv(out / "correlations.csv", index=False)
    neff_df.to_csv(out / "effective_n.csv", index=False)

    # --- other outputs -----------------------------------------------------------------------------------------
    pd.DataFrame(pooled_rows).assign(data=label).to_csv(out / "pooled_estimates.csv", index=False)
    pd.DataFrame(se_rows).assign(data=label).to_csv(out / "se_comparison.csv", index=False)
    pd.DataFrame(boot_rows).assign(data=label).to_csv(out / "bootstrap.csv", index=False)
    per_index.drop(columns=["dm_p"]).to_csv(out / "per_index.csv", index=False)
    long = []
    for t in TARGETS:
        for k in keys:
            L = daily[t][k]
            long.append(pd.DataFrame({"data": label, "target": t, "index": names[k], "date": L.index,
                                      "is_expiry": L["is_expiry"].to_numpy(),
                                      "anchors": L["anchors"].to_numpy(),
                                      "qlike_mcsgarch": L[BENCH].to_numpy(), "qlike_hybrid_base": L[MODEL].to_numpy(),
                                      "qlike_mcsgarch_iv": L[MATCHED].to_numpy()}))
    pd.concat(long).to_csv(out / "daily_losses.csv", index=False, float_format="%.10g")

    # --- verdicts ----------------------------------------------------------------------------------------------
    def fam_p(prefix):
        return tdf[tdf.family == prefix].set_index("test_id")["p_adjusted"]

    verdict = []
    p1 = fam_p("P1")
    n_ok = int((p1 <= 0.05).sum())
    verdict.append({"family": "P1 primary", "rule": "both Holm p <= 0.05: confirmed; one: partly; none: not confirmed",
                    "verdict": {2: "pooled gain confirmed", 1: "pooled gain partly confirmed", 0: "pooled gain not confirmed"}[n_ok]})
    s = fam_p("S1a")
    verdict.append({"family": "S1a bootstrap", "rule": "both Holm p <= 0.05: the bootstrap agrees",
                    "verdict": "bootstrap agrees" if (s <= 0.05).all() else
                    ("bootstrap agrees at one horizon" if (s <= 0.05).any() else "bootstrap does not agree")})
    s = fam_p("S1b")
    verdict.append({"family": "S1b homogeneity", "rule": "a Holm p <= 0.05 at a horizon: the gains differ there",
                    "verdict": "the gains differ at " + ", ".join(i.split("_")[1] for i in s.index[s <= 0.05])
                    if (s <= 0.05).any() else "no evidence that the gains differ (equality not rejected)"})
    s = fam_p("S1c")
    verdict.append({"family": "S1c last hour", "rule": "p <= 0.05: pooled last-hour gain supported",
                    "verdict": "pooled last-hour gain supported" if (s <= 0.05).all() else "pooled last-hour gain not supported"})
    for fam, name in (("S1d", "pooled expiry-day gain"), ("S1e", "pooled edge over mcsgarch_iv")):
        s = fam_p(fam)
        k = int((s <= 0.05).sum())
        verdict.append({"family": f"{fam}", "rule": "both Holm p <= 0.05: supported; one: partly; none: not supported",
                        "verdict": {2: f"{name} supported", 1: f"{name} partly supported (at "
                                    + ", ".join(i.split("_")[1] for i in s.index[s <= 0.05]) + ")",
                                    0: f"{name} not supported"}[k]})
    pd.DataFrame(verdict).assign(data=label).to_csv(out / "verdicts.csv", index=False)
    for v in verdict:
        log(f"Verdict {v['family']}: {v['verdict']}")

    # --- run info ----------------------------------------------------------------------------------------------
    import scipy
    info = {
        "study": "P107 pooled three-index test of the forecasting hybrid",
        "label": label,
        "prereg_file": "docs/prereg/P107_POOLED_THREE_INDEX.md",
        "plan_commit": plan_commit,
        "code_commit_at_run": git_head() if label == "REAL DATA" else None,
        "script_sha256": sha256(Path(__file__)),
        "script_differs_from_code_commit": script_differs() if label == "REAL DATA" else None,
        "command": "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p107_pooled_test.py run",
        "inputs": {f"{names[k]}/{t}": {"file": rel_path(Path(dirs[k]) / f"forecasts_{t}.csv"),
                                       "sha256": sha256(Path(dirs[k]) / f"forecasts_{t}.csv")}
                   for k in keys for t in TARGETS},
        "common_dates": {"n": T_all, "first": dates[0], "last": dates[-1]},
        "seeds": {"bootstrap": {str(b): [SEED, b] for b in (MEAN_BLOCK, *ROBUST_BLOCKS)}},
        "bootstrap_draws": draws,
        "mean_block": MEAN_BLOCK,
        "robust_blocks": list(ROBUST_BLOCKS),
        "dk_lag_all_days": lag_all,
        "placebo_shifts_trading_days": SHIFT_PLACEBO,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
        "run_seconds": round(time.time() - t0, 1),
        "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import statsmodels
        info["versions"]["statsmodels"] = statsmodels.__version__
    except Exception:  # pragma: no cover
        pass
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (out / "stdout.txt").write_text("\n".join(log_lines) + "\n")
    return {"tests": tdf, "verdicts": verdict, "info": info}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["count", "run"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--plan-commit", default=None)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    args = ap.parse_args(argv)
    dirs = {k: ROOT / d for k, _, d in INDICES}
    if args.mode == "count":
        ph1 = Path(args.out) / "phase1"
        ph1.mkdir(parents=True, exist_ok=True)
        c = count_only(dirs)
        c["label"] = "REAL DATA (calendar counts only: date, minute, is_expiry and regime columns; no loss)"
        c["inputs_sha256"] = {f"{k}/forecasts_{t}.csv": sha256(Path(d) / f"forecasts_{t}.csv")
                              for k, d in dirs.items() for t in TARGETS}
        (ph1 / "counts.json").write_text(json.dumps(c, indent=2, default=str) + "\n")
        b = bounds_from_committed(ROOT / "results/pipeline_nse_finnifty/p104/other_indices_same_dates.csv")
        b.insert(0, "data", "REAL DATA (arithmetic on the committed p104/other_indices_same_dates.csv; no new loss)")
        b.to_csv(ph1 / "bounds_from_committed.csv", index=False)
        print(json.dumps(c, indent=2, default=str))
        print(b.to_string())
        return
    run(dirs, Path(args.out), plan_commit=args.plan_commit, draws=args.draws)


if __name__ == "__main__":
    main()
