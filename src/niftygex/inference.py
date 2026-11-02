"""Before-versus-after tests for expiry-day effects, with dependence-aware inference.

The 2025 paper compares per-minute volume and per-minute Rogers-Satchell
volatility before and after an expiry change with a Mann-Whitney-Wilcoxon
test that treats every minute as an independent draw. Minutes within a day
share the same daily volatility level and volatility clusters from minute to
minute, so they are not independent. This module provides that test and
three alternatives that respect the dependence:

* ``day_level_mwu``: aggregate to one number per day first, then test.
* ``hac_mean_difference``: regress the per-minute variable on an "after"
  dummy with Newey-West (HAC) standard errors.
* ``day_cluster_bootstrap``: resample whole days and bootstrap the
  difference in medians (or means).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def minute_level_mwu(before: np.ndarray, after: np.ndarray) -> dict:
    """Mann-Whitney U on pooled minutes (the paper's approach)."""
    b = np.ravel(before)
    a = np.ravel(after)
    b, a = b[np.isfinite(b)], a[np.isfinite(a)]
    res = mannwhitneyu(b, a, alternative="two-sided")
    return {"stat": float(res.statistic), "p": float(res.pvalue), "n_before": b.size, "n_after": a.size,
            "max_U": float(b.size) * a.size}


def day_level_mwu(before: np.ndarray, after: np.ndarray, agg=np.nanmedian) -> dict:
    """Aggregate each day (row) first, then Mann-Whitney on daily values."""
    b = agg(np.atleast_2d(before), axis=1)
    a = agg(np.atleast_2d(after), axis=1)
    res = mannwhitneyu(b, a, alternative="two-sided")
    return {"stat": float(res.statistic), "p": float(res.pvalue), "n_before": b.size, "n_after": a.size}


def hac_mean_difference(before: np.ndarray, after: np.ndarray, maxlags: int | None = None, log: bool = True) -> dict:
    """OLS of y on [1, after] with Newey-West standard errors.

    With ``log=True`` the variable is log transformed first (useful for
    volume and volatility, which are skewed). The default lag length is the
    number of minutes in one day, which covers within-day clustering.
    """
    import statsmodels.api as sm

    b = np.atleast_2d(before)
    a = np.atleast_2d(after)
    y = np.concatenate([b.ravel(), a.ravel()])
    d = np.concatenate([np.zeros(b.size), np.ones(a.size)])
    ok = np.isfinite(y) & ((y > 0) if log else True)
    y = np.log(y[ok]) if log else y[ok]
    X = sm.add_constant(d[ok])
    L = b.shape[1] if maxlags is None else maxlags
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    return {"diff": float(fit.params[1]), "se": float(fit.bse[1]), "p": float(fit.pvalues[1])}


def day_cluster_bootstrap(
    before: np.ndarray,
    after: np.ndarray,
    stat=np.nanmedian,
    n_boot: int = 2000,
    seed: int | None = 0,
) -> dict:
    """Bootstrap the difference stat(after) - stat(before) by resampling whole days.

    The two-sided p-value is from the percentile of zero in the bootstrap
    distribution of the centred statistic.
    """
    rng = np.random.default_rng(seed)
    b = np.atleast_2d(before)
    a = np.atleast_2d(after)
    obs = stat(a) - stat(b)
    draws = np.empty(n_boot)
    for k in range(n_boot):
        bb = b[rng.integers(0, len(b), len(b))]
        aa = a[rng.integers(0, len(a), len(a))]
        draws[k] = stat(aa) - stat(bb)
    centred = draws - draws.mean()
    p = float(np.mean(np.abs(centred) >= abs(obs)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p": p}
