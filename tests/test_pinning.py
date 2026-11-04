"""Fast unit tests for the helpers of experiments/pinning_study.py (no real data needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex import gex

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "pinning_study.py"
_spec = importlib.util.spec_from_file_location("pinning_study", _PATH)
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


# ---------------------------------------------------------------- strike distance


def test_dist_to_grid_values():
    x = np.array([22000.0, 22012.0, 22037.5, 22049.95, 22025.0, 17992.15])
    d = ps.dist_to_grid(x, 50.0)
    assert np.allclose(d, [0.0, 12.0, 12.5, 0.05, 25.0, 7.85])
    assert np.all((d >= 0) & (d <= 25.0))
    assert np.allclose(ps.nearest_grid([22012.0, 22038.0], 50.0), [22000.0, 22050.0])


def test_bank_nifty_grid_and_bp():
    assert ps.dist_to_grid(44327.8, 100.0) == pytest.approx(27.8)
    assert ps.dist_to_grid(44372.2, 100.0) == pytest.approx(27.8)


def test_placebo_strikes_are_half_grid_shift():
    rng = np.random.default_rng(0)
    x = rng.uniform(20000, 26000, 1000)
    for g in (50.0, 100.0):
        d, dp = ps.dist_to_grid(x, g), ps.placebo_dist(x, g)
        # a close sitting on a strike is as far as possible from the shifted (placebo) strikes
        assert np.allclose(d + dp, g / 2)
        assert np.allclose(ps.nearest_grid(x, g, offset=g / 2) % g, g / 2)


def test_signed_move_and_change_in_distance():
    assert ps.signed_move_toward(10.0, 15.0, 20.0) == pytest.approx(5.0)
    assert ps.signed_move_toward(10.0, 5.0, 20.0) == pytest.approx(-5.0)
    assert ps.signed_move_toward(30.0, 25.0, 20.0) == pytest.approx(5.0)
    assert ps.signed_move_toward(20.0, 25.0, 20.0) == 0.0  # starting on the strike: no direction
    assert ps.change_in_distance(10.0, 18.0, 20.0) == pytest.approx(-8.0)
    assert ps.change_in_distance(18.0, 30.0, 20.0) == pytest.approx(8.0)


def test_signed_move_has_zero_mean_without_pinning():
    # random-walk moves, independent of where the strike is: mean "move toward" is ~0
    # while the change in absolute distance is positive (the Jensen effect the design avoids)
    rng = np.random.default_rng(1)
    p0 = rng.uniform(22000, 23000, 200_000)
    p1 = p0 + rng.normal(0, 30, p0.size)
    K = ps.nearest_grid(p0, 50.0)
    mt = ps.signed_move_toward(p0, p1, K)
    assert abs(mt.mean()) < 4 * mt.std() / np.sqrt(mt.size)
    assert ps.change_in_distance(p0, p1, K).mean() > 1.0


# ---------------------------------------------------------------- chain selection (no look-ahead)


def _table(date, expiries, oi=100.0):
    rows = []
    for e in expiries:
        for k in (21950.0, 22000.0, 22050.0):
            for c in (True, False):
                rows.append({"date": pd.Timestamp(date), "expiry": pd.Timestamp(e), "strike": k, "is_call": c,
                             "oi": oi, "settle": 10.0, "underlying": np.nan})
    return pd.DataFrame(rows)


def test_prev_trading_day_is_strictly_before():
    days = pd.DatetimeIndex(["2025-01-06", "2025-01-07", "2025-01-08"])
    assert ps.prev_trading_day(days, pd.Timestamp("2025-01-08")) == pd.Timestamp("2025-01-07")
    assert ps.prev_trading_day(days, pd.Timestamp("2025-01-06")) is None
    # a non-trading date: previous trading day before it
    assert ps.prev_trading_day(days, pd.Timestamp("2025-01-09")) == pd.Timestamp("2025-01-08")
    # skip a day with no file
    assert ps.prev_trading_day(days, pd.Timestamp("2025-01-08"), available={pd.Timestamp("2025-01-06")}) == \
        pd.Timestamp("2025-01-06")


def test_chain_before_never_uses_same_day_file():
    days = pd.DatetimeIndex(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"])
    E = pd.Timestamp("2025-01-09")
    bhav = {d: _table(d, [E, pd.Timestamp("2025-01-16")], oi=float(i + 1)) for i, d in enumerate(days)}
    ch, cdate, stale = ps.chain_before(bhav, days, E, E)
    assert cdate == pd.Timestamp("2025-01-08") and cdate < E and stale == 0
    assert set(pd.to_datetime(ch["expiry"])) == {E}
    assert (ch["oi"] == 3.0).all()  # the 8 Jan file, not the 9 Jan one
    # changing the expiry-day file must not change what the study sees for that day
    bhav2 = dict(bhav)
    bhav2[E] = _table(E, [E], oi=1e9)
    ch2, _, _ = ps.chain_before(bhav2, days, E, E)
    pd.testing.assert_frame_equal(ch.reset_index(drop=True), ch2.reset_index(drop=True))


def test_chain_before_missing_file_and_placebo_day():
    days = pd.DatetimeIndex(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"])
    E = pd.Timestamp("2025-01-09")
    bhav = {d: _table(d, [E]) for d in days if d != pd.Timestamp("2025-01-08")}
    ch, cdate, stale = ps.chain_before(bhav, days, E, E)
    assert cdate == pd.Timestamp("2025-01-07") and stale == 1
    # placebo day (one day before expiry) uses the same contract from the day before it
    ch, cdate, _ = ps.chain_before(bhav, days, pd.Timestamp("2025-01-07"), E)
    assert cdate == pd.Timestamp("2025-01-06") and set(pd.to_datetime(ch["expiry"])) == {E}


def test_chain_before_handles_holiday_redated_contract():
    # 27 Jun 2023: the Nifty contract was still listed for 29 Jun but expired on 28 Jun
    days = pd.DatetimeIndex(["2023-06-26", "2023-06-27", "2023-06-28"])
    bhav = {pd.Timestamp("2023-06-27"): _table("2023-06-27", ["2023-06-29", "2023-07-06"]),
            pd.Timestamp("2023-06-28"): _table("2023-06-28", ["2023-06-28", "2023-07-06"])}
    ch, cdate, _ = ps.chain_before(bhav, days, pd.Timestamp("2023-06-28"), pd.Timestamp("2023-06-28"))
    assert cdate == pd.Timestamp("2023-06-27")
    assert set(pd.to_datetime(ch["expiry"])) == {pd.Timestamp("2023-06-29")}
    assert "redated_from" in ch.attrs


# ---------------------------------------------------------------- targets and dealer gamma


def _chain():
    K = np.array([21900.0, 21950.0, 22000.0, 22050.0, 22100.0])
    rows = []
    for k in K:
        rows.append({"strike": k, "is_call": True, "oi": 1000.0 if k != 22050 else 5000.0, "iv": 0.15})
        rows.append({"strike": k, "is_call": False, "oi": 1000.0 if k != 21950 else 3000.0, "iv": 0.15})
    return pd.DataFrame(rows)


def test_strike_targets():
    c = _chain()
    t = ps.strike_targets(c, F=22010.0, T=1 / 252)
    assert t["K_maxoi"] == 22050.0
    assert t["K_maxpain"] == gex.max_pain(c)
    assert t["K_maxgamma"] in (22000.0, 22050.0)
    assert 0 < t["maxoi_share"] < 1


def test_dealer_gamma_signs_and_consistency_with_package():
    c = _chain()
    w_us = ps.rule_weights(c, "us_convention", None)
    # only calls at a strike: dealers long gamma there under the US rule; only puts: short
    calls = c.assign(oi=np.where(c["is_call"], c["oi"], 0.0))
    puts = c.assign(oi=np.where(~c["is_call"], c["oi"], 0.0))
    assert ps.dealer_gamma_at(calls, [22000.0], ps.TAU_SETTLE, w_us)[0] > 0
    assert ps.dealer_gamma_at(puts, [22000.0], ps.TAU_SETTLE, w_us)[0] < 0
    # at the forward with the chain's own T it equals the package's net GEX
    F, T = 22010.0, 1 / 252
    ours = ps.dealer_gamma_at(c, [F], T, w_us)[0]
    pkg = gex.strike_gex(c, F, 1.0, T, ps.RATE, 0.0, "us_convention", forward=F)["gex"].sum()
    assert ours == pytest.approx(pkg, rel=1e-10)
    gross = ps.dealer_gamma_at(c, [F], T, ps.rule_weights(c, "gross", None))[0]
    assert gross > abs(ours)


def test_participant_rules_use_t_minus_1_table():
    tbl = pd.DataFrame({"option_index_call_long": [60.0, 40.0, 100.0], "option_index_call_short": [50.0, 50.0, 100.0],
                        "option_index_put_long": [30.0, 70.0, 100.0], "option_index_put_short": [60.0, 40.0, 100.0]},
                       index=["Client", "Pro", "TOTAL"])
    c = _chain()
    w_pro = ps.rule_weights(c, "participant_pro", tbl)
    w_cm = ps.rule_weights(c, "client_mirror", tbl)
    assert np.allclose(w_pro[c["is_call"].to_numpy()], -0.1) and np.allclose(w_pro[~c["is_call"].to_numpy()], 0.3)
    assert np.allclose(w_cm[c["is_call"].to_numpy()], -0.1) and np.allclose(w_cm[~c["is_call"].to_numpy()], 0.3)
    assert np.isnan(ps.rule_weights(c, "participant_pro", None)).all()


# ---------------------------------------------------------------- inference helpers


def test_permutation_and_sign_flip_tests():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 25, 200), rng.uniform(0, 25, 700)
    _, p_null = ps.perm_test_diff(x, y, rng, n=2000)
    assert p_null > 0.01
    diff, p_alt = ps.perm_test_diff(x * 0.6, y, rng, n=2000)
    assert diff < 0 and p_alt < 0.001
    # strata: labels shuffled within weekday keeps the test valid
    sx, sy = rng.integers(0, 5, x.size), rng.integers(0, 5, y.size)
    _, p_s = ps.perm_test_diff(x, y, rng, n=2000, strata_x=sx, strata_y=sy)
    assert 0 < p_s <= 1
    m, p = ps.sign_flip_test(rng.normal(0, 1, 300), rng, n=2000)
    assert p > 0.01
    m, p = ps.sign_flip_test(rng.normal(0.5, 1, 300), rng, n=2000)
    assert m > 0 and p < 0.001


def test_uniform_mean_test_and_holm():
    rng = np.random.default_rng(3)
    assert ps.uniform_mean_test(rng.uniform(0, 25, 200), 25.0, rng, n=2000) > 0.01
    assert ps.uniform_mean_test(rng.uniform(0, 15, 200), 25.0, rng, n=2000) < 0.001
    assert np.allclose(ps.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(ps.holm([0.2, 0.9]), [0.4, 0.9])
    assert np.allclose(ps.holm([0.5, 0.9]), [1.0, 1.0])


def test_power_check_size_is_nominal():
    rng = np.random.default_rng(4)
    pw = ps.power_check([("t", 100, 300)], 50.0, rng, reps=500)
    size = pw[(pw["effect"] == 0)]["power_5pct"]
    assert (size < 0.1).all()
    big = pw[(pw["effect"] == 0.6)]["power_5pct"]
    assert (big > 0.9).all()
