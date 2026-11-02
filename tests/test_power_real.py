"""Fast synthetic checks for experiments/power_real.py (no real data needed).

They check the numerical helpers against statsmodels, the resampling and sign-error generators, and the
simulation engine itself: with no effect the tests reject about 5% of the time (size), and power rises with
the size of the effect and with the sample size.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import statsmodels.api as sm

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "power_real.py"
_spec = importlib.util.spec_from_file_location("power_real", _PATH)
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

LAMS = np.array([0.0, 1.0, 2.0, 3.0, 4.0])


def _ar1(n, rho, rng, burn=50):
    e = rng.normal(size=n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = rho * x[t - 1] + e[t]
    return x[burn:]


def _reg_spec(n=300, seed=0, rho=0.3, tested="level", binary=False):
    """Synthetic day-level design: one control, an expiry dummy every fifth day, a persistent GEX-like regressor
    and AR(1) noise. There is no GEX effect in the data; est is set to one standard error, so lambda counts
    standard errors."""
    rng = np.random.default_rng(seed)
    x = _ar1(n, 0.5, rng)
    g = _ar1(n, 0.8, rng)
    if binary:
        g = (g < 0).astype(float)
    exp = (np.arange(n) % 5 == 4).astype(float)
    mult = rng.normal(size=n) if binary else exp
    C = np.column_stack([np.ones(n), x, exp])
    y = 0.5 + 0.3 * x + 0.2 * exp + _ar1(n, rho, rng)
    sp = pr.make_reg_spec("syn", y, C, g, mult, tested, binary, np.abs(g) + rng.random(n) * 1e-9,
                          {"all days": np.arange(n), "second half": np.arange(n // 2, n)})
    sp["est"] = sp["se"]
    return sp


def _cell(**kw):
    c = {"scheme": "residual", "m": 1.0, "source": "all days", "pattern": "none", "flip": 0.0, "block": 10,
         "lambdas": LAMS}
    c.update(kw)
    return c


def _power(spec, reps, seed, stat=None, **kw):
    acc = pr.simulate_cell(spec, _cell(**kw), reps, np.random.default_rng(seed))
    stat = stat or next(iter(acc))
    return acc[stat]["rej"] / reps


# ---------------------------------------------------------------- numerical helpers


def test_ols_hac_batch_matches_statsmodels():
    rng = np.random.default_rng(1)
    n = 250
    X = np.column_stack([np.ones(n), rng.normal(size=n), _ar1(n, 0.7, rng)])
    Y = np.column_stack([X @ np.array([0.2, 1.0, -0.5]) + _ar1(n, 0.4, rng) for _ in range(3)])
    for j in (1, 2):
        b, se = pr.ols_hac_batch(X, Y, j, lags=5)
        for k in range(Y.shape[1]):
            ref = sm.OLS(Y[:, k], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            assert np.isclose(b[k], ref.params[j])
            assert np.isclose(se[k], ref.bse[j], rtol=1e-10)


def test_mean_hac_batch_matches_statsmodels():
    rng = np.random.default_rng(2)
    D = np.column_stack([_ar1(120, 0.5, rng) + 0.3, _ar1(120, 0.1, rng)])
    mu, se = pr.mean_hac_batch(D, lags=4)
    for k in range(2):
        ref = sm.OLS(D[:, k], np.ones((120, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        assert np.isclose(mu[k], ref.params[0])
        assert np.isclose(se[k], ref.bse[0], rtol=1e-10)


def test_stationary_indices_runs_and_range():
    rng = np.random.default_rng(3)
    idx = pr.stationary_indices(50, 20000, 8.0, rng)
    assert len(idx) == 20000 and idx.min() >= 0 and idx.max() < 50
    cont = idx[1:] == (idx[:-1] + 1) % 50
    mean_run = len(idx) / (1 + (~cont).sum())
    assert 6.5 < mean_run < 10.0  # geometric runs with mean 8 (plus rare accidental continuations)
    assert len(pr.stationary_indices(50, 0, 8.0, rng)) == 0


def test_block_signs_constant_within_blocks():
    s = pr.block_signs(200, 10, np.random.default_rng(4))
    assert set(np.unique(s)) <= {-1.0, 1.0}
    changes = np.flatnonzero(s[1:] != s[:-1]) + 1
    assert np.all(np.diff(changes) % 10 == 0)  # sign changes only on block boundaries


def test_flip_mask_patterns():
    rng = np.random.default_rng(5)
    n = 50000
    assert abs(pr.flip_mask(n, 0.25, "iid", rng).mean() - 0.25) < 0.01
    sp = pr.flip_mask(n, 0.25, "spells", rng)
    assert abs(sp.mean() - 0.25) < 0.03
    starts = np.flatnonzero(sp[1:] & ~sp[:-1])
    assert 15 < sp.sum() / max(len(starts), 1) < 25  # spells last about 20 days
    mag = np.arange(100.0)[::-1]
    sm_ = pr.flip_mask(100, 0.1, "smallest", rng, mag)
    assert sm_.sum() == 10 and np.all(mag[sm_] < 10)
    assert not pr.flip_mask(10, 0.0, "iid", rng).any()


def test_measured_flips_sign_or_indicator():
    v = np.array([0.5, -1.0, 2.0])
    f = np.array([True, False, True])
    assert np.allclose(pr.measured(v, f, binary=False), [-0.5, -1.0, -2.0])
    assert np.allclose(pr.measured(np.array([1.0, 0.0, 0.0]), f, binary=True), [0.0, 0.0, 1.0])


def test_mde_and_multiplier_helpers():
    lams = np.array([0.0, 0.5, 1.0, 2.0])
    assert np.isclose(pr.mde_from_curve(lams, np.array([0.05, 0.3, 0.7, 0.9])), 1.5)
    assert np.isnan(pr.mde_from_curve(lams, np.array([0.05, 0.1, 0.2, 0.5])))
    # power at the target reaches 80% between 1.5x and 2x
    m, how = pr.mult_needed([1, 1.5, 2, 3], [0.5, 0.7, 0.9, 0.99], [1.4, 1.1, 0.95, 0.8], 1.0)
    assert np.isclose(m, 1.75) and how.startswith("simulated")
    # already enough: scaled down with the square-root rule
    m, how = pr.mult_needed([1, 1.5, 2, 3], [0.9, 0.95, 0.99, 1.0], [0.8, 0.65, 0.57, 0.46], 1.0)
    assert np.isclose(m, 0.64) and how.startswith("current sample")
    # beyond the grid: extrapolated from the MDE at 3x
    m, how = pr.mult_needed([1, 1.5, 2, 3], [0.2, 0.3, 0.4, 0.6], [2.0, 1.7, 1.4, 1.2], 1.0)
    assert np.isclose(m, 3 * 1.2 ** 2) and how.startswith("extrapolated")


# ---------------------------------------------------------------- the simulation engine


@pytest.mark.parametrize("scheme", ["residual", "wild"])
def test_regression_size_near_nominal(scheme):
    sp = _reg_spec(seed=10)
    p0 = _power(sp, 800, 11, scheme=scheme, lambdas=np.array([0.0]))[0]
    assert 0.025 <= p0 <= 0.085  # 5% nominal; Monte Carlo standard error is 0.8 points


def test_regression_power_rises_with_delta_and_sample():
    sp = _reg_spec(seed=12)
    p1 = _power(sp, 500, 13)
    assert np.all(np.diff(p1) > -0.03)  # monotone up to Monte Carlo noise
    assert p1[-1] > 0.85 and p1[1] < 0.5  # 4 standard errors: high power; 1 standard error: low
    p2 = _power(sp, 500, 14, m=2.0)
    p3 = _power(sp, 500, 15, m=3.0)
    assert p2[2] > p1[2] + 0.05 and p3[2] > p2[2]
    # extending from a sub-sample of days works too
    p2b = _power(sp, 300, 16, m=2.0, source="second half")
    assert p2b[2] > p1[2] - 0.05


def test_interaction_and_binary_designs_run():
    sp = _reg_spec(seed=17, tested="interaction", binary=True)
    p = _power(sp, 300, 18)
    assert 0.0 <= p[0] <= 0.1 and p[-1] > 0.8


def test_sign_errors_reduce_power():
    sp = _reg_spec(seed=19)
    lams = np.array([0.0, 3.0])
    p_ok = _power(sp, 500, 20, lambdas=lams)
    p_25 = _power(sp, 500, 20, lambdas=lams, pattern="iid", flip=0.25)
    p_50 = _power(sp, 500, 20, lambdas=lams, pattern="iid", flip=0.5)
    assert p_25[1] < p_ok[1] - 0.15
    assert p_50[1] < 0.15  # half the signs wrong: the regressor carries no signal
    acc = pr.simulate_cell(sp, _cell(lambdas=lams, pattern="iid", flip=0.9), 300, np.random.default_rng(21))
    assert acc["hac_t"]["rej_right"][1] < 0.05 * 300  # mostly reversed sign: rejections have the wrong sign


def test_r2_engine_size_and_power():
    # Block length 1 (plain resampling of weeks): the Wilcoxon test should hold its 5% level.
    rng = np.random.default_rng(31)
    sp = pr.make_r2_spec("syn", rng.normal(size=80) * 0.4)
    sp["est"] = sp["se"]
    for scheme in ("residual", "wild"):
        acc = pr.simulate_cell(sp, _cell(block=1, source="same regime", scheme=scheme), 800,
                               np.random.default_rng(23))
        pw = acc["wilcoxon"]["rej"] / 800
        pt = acc["hac_t_mean"]["rej"] / 800
        assert 0.02 <= pw[0] <= 0.085
        assert 0.02 <= pt[0] <= 0.11  # Newey-West t with 80 weeks rejects a little too often, a known effect
        assert np.all(np.diff(pw) > -0.03) and pw[-1] > 0.85
    # Blocks of 4 weeks on a longer series, and a doubled sample
    sp = pr.make_r2_spec("syn", np.random.default_rng(32).normal(size=400) * 0.4)
    sp["est"] = sp["se"]
    acc = pr.simulate_cell(sp, _cell(block=4, source="same regime"), 800, np.random.default_rng(23))
    assert 0.025 <= acc["hac_t_mean"]["rej"][0] / 800 <= 0.09
    p2 = pr.simulate_cell(sp, _cell(block=4, source="same regime", m=2.0), 400, np.random.default_rng(24))
    assert p2["wilcoxon"]["rej"][2] / 400 > acc["wilcoxon"]["rej"][2] / 800 + 0.1


def test_forecast_engine_size_and_power():
    rng = np.random.default_rng(25)
    n, ntr = 400, 200
    x = _ar1(n, 0.6, rng)
    g = _ar1(n, 0.9, rng) * 0.02
    exp = (np.arange(n) % 5 == 4).astype(float)
    X0 = np.column_stack([np.ones(n), x, exp])
    y = 0.1 + 0.5 * x + 0.3 * exp + _ar1(n, 0.2, rng) * 0.5
    h = -3.0 * g + 8.0 * g * exp  # effect shape, as in the real study
    sp = pr.make_fc_spec(y, X0, g, exp, h, ntr, {"all days": np.arange(n)})
    acc = pr.simulate_cell(sp, _cell(lambdas=np.array([0.0, 0.5, 1.0])), 400, np.random.default_rng(26))
    p = acc["dm_one_sided"]["rej"] / 400
    assert p[0] <= 0.08  # nested models: the one-sided test is at most nominal
    assert p[2] > p[1] > p[0]
