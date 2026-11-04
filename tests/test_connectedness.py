import numpy as np

from niftygex import connectedness as cn


def test_paper_scale_vs_standard_scale():
    # Table 3 of Chhimwal, Pandey and Pandey (2025), weekly and monthly expiry, before.
    t = cn.table_from_paper([[57.29, 42.71], [41.29, 58.71]])
    assert np.isclose(t.paper_scale_total, 84.00)
    assert np.isclose(t.total, 42.00)


def test_gfevd_independent_series_has_no_spillover():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(20000, 2))
    c = cn.connectedness(Y, p=1, H=10)
    assert c.total < 0.5


def test_gfevd_impact_horizon_matches_closed_form():
    # VAR with no dynamics: H=1 generalized FEVD has closed form rho^2 / (1 + rho^2).
    rho = 0.8
    A = np.zeros((1, 2, 2))
    Sigma = np.array([[1, rho], [rho, 1.0]])
    th = cn.gfevd(A, Sigma, H=1)
    assert np.allclose(th[0, 1], rho**2 / (1 + rho**2))


def test_var_recovers_coefficients_and_directional_spillover():
    rng = np.random.default_rng(3)
    T = 50000
    Y = np.zeros((T, 2))
    e = rng.normal(size=(T, 2))
    for t in range(1, T):
        Y[t, 0] = 0.5 * Y[t - 1, 0] + e[t, 0]
        Y[t, 1] = 0.6 * Y[t - 1, 0] + 0.2 * Y[t - 1, 1] + e[t, 1]
    fit = cn.fit_var(Y, 1)
    assert np.allclose(fit.A[0], [[0.5, 0.0], [0.6, 0.2]], atol=0.02)
    c = cn.connectedness_from_fit(fit, H=10)
    assert c.net[0] > 10 and c.net[1] < -10  # series 0 transmits to series 1
