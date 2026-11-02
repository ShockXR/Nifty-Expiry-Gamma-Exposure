import numpy as np

from niftygex import realized


def test_rogers_satchell_unbiased_for_brownian_bars():
    # Fine monitoring (20,000 steps per bar). With coarse monitoring the bar's
    # high and low are understated and RS is biased low; see ITERATION_LOG.md.
    rng = np.random.default_rng(1)
    n, k, sig = 1000, 20000, 0.001
    steps = rng.normal(0, sig / np.sqrt(k), (n, k))
    path = np.cumsum(steps, axis=1)
    o = np.ones(n)
    c = np.exp(path[:, -1])
    h = np.exp(np.maximum(path.max(axis=1), 0))
    l = np.exp(np.minimum(path.min(axis=1), 0))
    rs = realized.rogers_satchell(o, h, l, c)
    assert abs(rs.mean() / sig**2 - 1) < 0.05


def test_rv_and_bipower_agree_without_jumps_and_split_with_jump():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.001, (2000, 375))
    rv = realized.realized_variance(r)
    bv = realized.bipower_variation(r)
    assert abs(np.mean(bv / rv) - 1) < 0.01
    r[:, 100] += 0.01  # add a jump of 10 sd
    js = realized.jump_share(r)
    assert np.median(js) > 0.15


def test_rv_k_minute_aggregation():
    r = np.arange(10, dtype=float)
    assert np.isclose(realized.realized_variance(r, k=5), (0 + 1 + 2 + 3 + 4) ** 2 + (5 + 6 + 7 + 8 + 9) ** 2)


def test_bulk_volume_signs():
    v = realized.bulk_volume_imbalance(np.array([2.0, -2.0, 0.0]), np.array([100.0, 100, 100]), 1.0)
    assert v[0] > 90 and v[1] < -90 and v[2] == 0
