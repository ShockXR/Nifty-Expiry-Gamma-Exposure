"""Black-76 prices, greeks and implied volatility for Nifty index options.

Nifty options are European and cash settled, and are usually priced off the
futures price F, so Black (1976) is the natural model. All functions are
vectorised over numpy arrays.

Conventions
-----------
F      forward (futures) price
K      strike
T      time to expiry in years (must be > 0)
r      continuously compounded risk-free rate
sigma  annualised implied volatility
is_call  boolean array, True for calls

Greeks are with respect to F. For an index option, the spot gamma is
gamma_F * (dF/dS)**2, and dF/dS = exp((r - q) T) is within a fraction of a
percent of 1 for weekly options, so ``spot_gamma`` applies that factor
explicitly when the caller supplies the carry.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _pdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _d1d2(F, K, T, sigma):
    F, K, T, sigma = (np.asarray(a, dtype=float) for a in (F, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(F, K, T, r, sigma, is_call):
    """Black-76 option price."""
    d1, d2 = _d1d2(F, K, T, sigma)
    disc = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    call = disc * (F * ndtr(d1) - K * ndtr(d2))
    put = disc * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def delta(F, K, T, r, sigma, is_call):
    d1, _ = _d1d2(F, K, T, sigma)
    disc = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    return np.where(is_call, disc * ndtr(d1), disc * (ndtr(d1) - 1.0))


def gamma(F, K, T, r, sigma):
    """Second derivative of price with respect to F. Same for calls and puts."""
    d1, _ = _d1d2(F, K, T, sigma)
    disc = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    return disc * _pdf(d1) / (np.asarray(F, float) * sigma * np.sqrt(T))


def vega(F, K, T, r, sigma):
    """Derivative of price with respect to sigma (per 1.00 of vol, not per 1%)."""
    d1, _ = _d1d2(F, K, T, sigma)
    disc = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    return disc * np.asarray(F, float) * _pdf(d1) * np.sqrt(T)


def vanna(F, K, T, r, sigma):
    """Derivative of delta with respect to sigma. Same for calls and puts."""
    d1, d2 = _d1d2(F, K, T, sigma)
    disc = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    return -disc * _pdf(d1) * d2 / sigma


def charm(F, K, T, r, sigma, is_call):
    """Change in delta per year of calendar time passing (dDelta/dt = -dDelta/dT).

    Multiply by 1/365 for the change per calendar day. Charm is what makes
    dealer hedges move on expiry day even when the index does not.
    """
    d1, d2 = _d1d2(F, K, T, sigma)
    T = np.asarray(T, float)
    r = np.asarray(r, float)
    disc = np.exp(-r * T)
    common = disc * _pdf(d1) * d2 / (2.0 * T)
    call = r * disc * ndtr(d1) + common
    put = r * disc * (ndtr(d1) - 1.0) + common
    return np.where(is_call, call, put)


def spot_gamma(F, K, T, r, sigma, carry=0.0):
    """Gamma with respect to spot S, where F = S exp(carry * T).

    carry = r - dividend yield. On expiry week the factor is close to 1.
    """
    factor = np.exp(np.asarray(carry, float) * np.asarray(T, float))
    return gamma(F, K, T, r, sigma) * factor**2


def implied_vol(
    target,
    F,
    K,
    T,
    r,
    is_call,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-10,
    max_iter: int = 100,
):
    """Vectorised Black-76 implied volatility.

    Newton steps are taken inside a bracket [lo, hi] that is shrunk with every
    evaluation, and a bisection step is used whenever Newton would leave the
    bracket or vega is tiny. This keeps the solver safe for deep in- or
    out-of-the-money options on expiry day, where vega goes to zero.

    Returns NaN where the price violates no-arbitrage bounds.
    """
    target, F, K, T, r = np.broadcast_arrays(*(np.asarray(a, float) for a in (target, F, K, T, r)))
    is_call = np.broadcast_to(np.asarray(is_call, bool), target.shape)
    disc = np.exp(-r * T)
    intrinsic = np.where(is_call, disc * np.maximum(F - K, 0.0), disc * np.maximum(K - F, 0.0))
    upper = np.where(is_call, disc * F, disc * K)
    valid = (target > intrinsic) & (target < upper) & (T > 0)

    lo_arr = np.full(target.shape, lo)
    hi_arr = np.full(target.shape, hi)
    # Brenner and Subrahmanyam style starting point, clipped into the bracket.
    with np.errstate(divide="ignore", invalid="ignore"):
        guess = np.sqrt(2.0 * np.pi / np.where(T > 0, T, 1.0)) * target / (disc * F)
    sig = np.clip(np.nan_to_num(guess, nan=0.2), lo * 2, hi / 2)
    sig = np.where(valid, sig, 0.2)

    for _ in range(max_iter):
        p = price(F, K, T, r, sig, is_call)
        diff = p - target
        # Price is increasing in sigma, so shrink the bracket.
        hi_arr = np.where(diff > 0, sig, hi_arr)
        lo_arr = np.where(diff <= 0, sig, lo_arr)
        v = vega(F, K, T, r, sig)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = sig - diff / v
        use_newton = (v > 1e-12) & (newton > lo_arr) & (newton < hi_arr)
        new_sig = np.where(use_newton, newton, 0.5 * (lo_arr + hi_arr))
        done = np.abs(new_sig - sig) < tol
        sig = np.where(valid, new_sig, sig)
        if np.all(done | ~valid):
            break
    return np.where(valid, sig, np.nan)
