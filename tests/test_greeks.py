import numpy as np
import pytest

from niftygex import greeks

F, K, T, r = 24000.0, np.array([23000, 23500, 24000, 24500, 25000.0]), 5 / 365, 0.065
SIG = np.array([0.18, 0.15, 0.13, 0.12, 0.125])


def test_put_call_parity():
    c = greeks.price(F, K, T, r, SIG, True)
    p = greeks.price(F, K, T, r, SIG, False)
    assert np.allclose(c - p, np.exp(-r * T) * (F - K), atol=1e-8)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call):
    dF, ds, dT = 0.5, 1e-5, 1e-7
    p = lambda f=F, s=SIG, t=T: greeks.price(f, K, t, r, s, is_call)  # noqa: E731
    d = greeks.delta(F, K, T, r, SIG, is_call)
    assert np.allclose(d, (p(F + dF) - p(F - dF)) / (2 * dF), atol=1e-6)
    g = greeks.gamma(F, K, T, r, SIG)
    assert np.allclose(g, (p(F + dF) - 2 * p() + p(F - dF)) / dF**2, rtol=1e-3, atol=1e-8)
    v = greeks.vega(F, K, T, r, SIG)
    assert np.allclose(v, (p(s=SIG + ds) - p(s=SIG - ds)) / (2 * ds), rtol=1e-5)
    vn = greeks.vanna(F, K, T, r, SIG)
    dd = (greeks.delta(F, K, T, r, SIG + ds, is_call) - greeks.delta(F, K, T, r, SIG - ds, is_call)) / (2 * ds)
    assert np.allclose(vn, dd, rtol=1e-4, atol=1e-8)
    ch = greeks.charm(F, K, T, r, SIG, is_call)
    dT_num = -(greeks.delta(F, K, T + dT, r, SIG, is_call) - greeks.delta(F, K, T - dT, r, SIG, is_call)) / (2 * dT)
    assert np.allclose(ch, dT_num, rtol=1e-4, atol=1e-6)


def test_implied_vol_roundtrip_including_expiry_day():
    rng = np.random.default_rng(0)
    n = 5000
    Ks = 24000 * np.exp(rng.uniform(-0.08, 0.08, n))
    Ts = rng.choice([0.5 / 365 / 6, 1 / 365, 3 / 365, 30 / 365], n)
    sig = rng.uniform(0.07, 0.6, n)
    calls = rng.random(n) < 0.5
    px = greeks.price(F, Ks, Ts, r, sig, calls)
    iv = greeks.implied_vol(px, F, Ks, Ts, r, calls)
    # Options worth essentially nothing beyond intrinsic carry no vol information.
    intrinsic = np.where(calls, np.maximum(F - Ks, 0), np.maximum(Ks - F, 0)) * np.exp(-r * Ts)
    informative = (px - intrinsic) > 1e-6 * F
    ok = informative & np.isfinite(iv)
    assert ok.sum() > 0.6 * n
    assert np.nanmax(np.abs(iv[ok] - sig[ok])) < 1e-6


def test_implied_vol_rejects_arbitrage_prices():
    iv = greeks.implied_vol(np.array([-1.0, 1e9]), F, 24000.0, T, r, True)
    assert np.all(np.isnan(iv))
