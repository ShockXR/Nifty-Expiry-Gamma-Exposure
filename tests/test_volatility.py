import numpy as np
import pytest

from niftygex.simulate import SimConfig, simulate_market
from niftygex.volatility import McsGarch, fit_garch11, _garch_q2


@pytest.fixture(scope="module")
def market():
    return simulate_market(SimConfig(n_days=120, kappa=0.0, seed=11))


def test_garch_recursion_matches_loop():
    rng = np.random.default_rng(0)
    x2 = rng.random(50)
    q = _garch_q2(x2, 0.1, 0.05, 0.9, 1.0)
    ref = [1.0]
    for n in range(1, 50):
        ref.append(0.1 + 0.05 * x2[n - 1] + 0.9 * ref[-1])
    assert np.allclose(q, ref)


def test_garch11_recovers_parameters():
    rng = np.random.default_rng(5)
    n, om, a, b, nu = 60000, 0.05, 0.05, 0.9, 6.0
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    x = np.empty(n)
    q = 1.0
    for i in range(n):
        x[i] = np.sqrt(q) * z[i]
        q = om + a * x[i] ** 2 + b * q
    p = fit_garch11(x)
    assert abs(p["alpha"] - a) < 0.01 and abs(p["beta"] - b) < 0.02 and abs(p["nu"] - nu) < 1.0


def test_mcsgarch_recovers_simulated_parameters(market):
    m = market
    mod = McsGarch().fit(m["R"], m["h_true"], m["labels"]["is_expiry"].to_numpy())
    p = mod.params
    assert abs(p["alpha"] - 0.03) < 0.01
    assert abs(p["beta"] - 0.95) < 0.02
    assert abs(p["nu"] - 7.0) < 1.5
    # diurnal shape: estimated and true curves are proportional up to noise
    s = mod.s / mod.s.sum()
    st = m["s_true"] / m["s_true"].sum()
    assert np.corrcoef(s, st)[0, 1] > 0.85


def test_forecast_is_causal_and_sums_components(market):
    m = market
    R, h = m["R"], m["h_true"]
    mod = McsGarch().fit(R, h)
    f1 = mod.filter(R, h)
    R2 = R.copy()
    R2[50, 200:] = 0.05  # change the future of day 50
    f2 = mod.filter(R2, h)
    a = mod.forecast_variance(f1, h, [50], [200], 30)
    b = mod.forecast_variance(f2, h, [50], [200], 30)
    assert np.isclose(a, b)  # data from minute 200 on cannot affect a forecast made at 200
    # horizon 1: h_t * s_i * q2
    one = mod.forecast_variance(f1, h, [10], [100], 1)[0]
    assert np.isclose(one, h[10] * f1["S"][10, 100] * f1["q2"][10, 100])


def test_expiry_diurnal_captures_late_session_bump(market):
    m = market
    ex = m["labels"]["is_expiry"].to_numpy()
    mod = McsGarch(expiry_diurnal=True, smooth=15).fit(m["R"], m["h_true"], ex)
    ratio = mod.s_expiry[-20:].mean() / mod.s[-20:].mean()
    true_ratio = m["s_true_expiry"][-20:].mean() / m["s_true"][-20:].mean()
    assert ratio > 1.15 and abs(ratio - true_ratio) < 0.5
