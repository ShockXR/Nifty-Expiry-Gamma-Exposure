"""Realized volatility measures from 1-minute bars.

All inputs are per-minute arrays for one or more days. Returns are log returns.

* Rogers-Satchell (1991) per-bar variance, which the 2025 paper uses for its
  volatility tests. It is unbiased for a driftless or drifting Brownian motion
  within the bar that is observed continuously, and uses open, high, low and
  close. With discretely sampled prices the observed high and low are inside
  the true ones, so the estimate is biased DOWN: measured on simulated paths
  (OUR CHECK, coordinator's measurement), the mean estimate is 0.79, 0.93 and
  0.99 of the true variance at 60, 600 and 6,000 sub-steps per bar.
* Realized variance at a chosen sampling interval (sum of squared returns).
* Bipower variation (Barndorff-Nielsen and Shephard 2004), which is robust to
  jumps; RV minus BV estimates the jump part.
* Bulk volume classification (Easley, Lopez de Prado and O'Hara 2012) as an
  order-flow proxy when trade signs are not available.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t

MU1 = np.sqrt(2.0 / np.pi)  # E|Z| for standard normal Z


def rogers_satchell(o, h, l, c):
    """Per-bar Rogers-Satchell variance: ln(H/C) ln(H/O) + ln(L/C) ln(L/O).

    Unbiased only for continuously observed prices; biased down for discretely sampled high and low (see the
    module docstring for the measured size).
    """
    o, h, l, c = (np.asarray(a, float) for a in (o, h, l, c))
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def realized_variance(returns, k: int = 1, axis: int = -1):
    """Sum of squared k-minute returns. ``returns`` are 1-minute log returns.

    With k > 1 the 1-minute returns are first summed into non-overlapping
    k-minute returns (a trailing remainder is kept as a shorter last block).
    """
    r = np.asarray(returns, float)
    if k == 1:
        return np.nansum(r**2, axis=axis)
    r = np.moveaxis(r, axis, -1)
    n = r.shape[-1]
    idx = np.arange(0, n, k)
    agg = np.add.reduceat(np.nan_to_num(r), idx, axis=-1)
    return np.sum(agg**2, axis=-1)


def bipower_variation(returns, axis: int = -1):
    """(pi/2) * sum |r_i| |r_{i-1}|, scaled by n/(n-1) for the lost term."""
    r = np.abs(np.moveaxis(np.asarray(returns, float), axis, -1))
    n = r.shape[-1]
    bv = np.nansum(r[..., 1:] * r[..., :-1], axis=-1) / MU1**2
    return bv * n / (n - 1)


def jump_share(returns, axis: int = -1):
    """max(RV - BV, 0) / RV, the share of variance attributed to jumps."""
    rv = realized_variance(returns, axis=axis)
    bv = bipower_variation(returns, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rv > 0, np.maximum(rv - bv, 0.0) / rv, 0.0)


def bulk_volume_imbalance(price_change, volume, sigma, dof: float | None = None):
    """Bulk volume classification: signed volume = V * (2 * CDF(dP / sigma) - 1).

    ``sigma`` is the standard deviation of bar price changes (for example a
    rolling estimate). With ``dof`` set, a Student-t CDF is used instead of the
    normal, as suggested for heavy tailed data.
    """
    z = np.asarray(price_change, float) / np.asarray(sigma, float)
    cdf = student_t.cdf(z, dof) if dof else norm.cdf(z)
    return np.asarray(volume, float) * (2.0 * cdf - 1.0)
