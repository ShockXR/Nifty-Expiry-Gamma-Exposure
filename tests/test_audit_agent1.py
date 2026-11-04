"""Independent audit tests (validation round, November 2026).

Invariants of the option maths (put-call parity, gamma, vega, implied-vol round trip), GEX sign rules and
units, the expiry calendar on known dates, forecast-loss properties (QLIKE, Diebold-Mariano, Holm), and the
no-look-ahead property of the feature builder and the walk-forward ladder: forecasts for days before a cut
must not change when the data after the cut is altered. All data are synthetic and seeded; the file runs in
well under a minute (set OMP_NUM_THREADS=1).
"""

import copy

import numpy as np
import pandas as pd
import pytest

from niftygex import evaluation as ev
from niftygex import gex, greeks, implied, inference, realized
from niftygex.data import to_session_grid
from niftygex.expiry_calendar import label_days, weekly_expiries
from niftygex.features import build_dataset, build_gex_panel
from niftygex.forecast import WalkForwardConfig, run_walkforward
from niftygex.simulate import SimConfig, simulate_market
from niftygex.volatility import _ewma_variance, daily_egarch_variance

R_RATE = 0.065


def _grid():
    F = 24000.0
    K = F * np.array([0.9, 0.97, 1.0, 1.03, 1.1])
    T = np.array([1.0, 3.0, 5.0, 21.0, 63.0]) / 252.0
    sig = np.array([0.08, 0.15, 0.3, 0.8])
    Kg, Tg, Sg = np.meshgrid(K, T, sig, indexing="ij")
    return F, Kg.ravel(), Tg.ravel(), Sg.ravel()


# --------------------------------------------------------------------------------------------
# Black-76 maths
# --------------------------------------------------------------------------------------------


def test_put_call_parity_holds_on_a_grid():
    F, K, T, s = _grid()
    c = greeks.price(F, K, T, R_RATE, s, True)
    p = greeks.price(F, K, T, R_RATE, s, False)
    assert np.allclose(c - p, np.exp(-R_RATE * T) * (F - K), rtol=0, atol=1e-6)
    assert np.all(c >= -1e-9) and np.all(p >= -1e-9)


def test_gamma_is_positive_and_equals_the_finite_difference_for_calls_and_puts():
    F, K, T, s = _grid()
    g = greeks.gamma(F, K, T, R_RATE, s)
    assert np.all(g > 0)
    h = F * 1e-4  # small against sigma * sqrt(T) * F (about 120 points at the shortest maturity and lowest vol)
    for is_call in (True, False):
        up, mid, dn = (greeks.price(F + d, K, T, R_RATE, s, is_call) for d in (h, 0.0, -h))
        fd = (up - 2 * mid + dn) / h**2
        ok = g > 1e-8
        assert ok.sum() > 50
        assert np.allclose(fd[ok], g[ok], rtol=2e-3, atol=1e-9)


def test_delta_vega_vanna_charm_match_finite_differences():
    F, K, T, s = _grid()
    for is_call in (True, False):
        h = F * 1e-5
        fd = (greeks.price(F + h, K, T, R_RATE, s, is_call) - greeks.price(F - h, K, T, R_RATE, s, is_call)) / (2 * h)
        assert np.allclose(fd, greeks.delta(F, K, T, R_RATE, s, is_call), atol=1e-6)
        # charm = - d delta / d T
        dT = T * 1e-5
        d_up = greeks.delta(F, K, T + dT, R_RATE, s, is_call)
        d_dn = greeks.delta(F, K, T - dT, R_RATE, s, is_call)
        assert np.allclose(-(d_up - d_dn) / (2 * dT), greeks.charm(F, K, T, R_RATE, s, is_call), atol=1e-4, rtol=1e-4)
    ds = s * 1e-5
    v_fd = (greeks.price(F, K, T, R_RATE, s + ds, True) - greeks.price(F, K, T, R_RATE, s - ds, True)) / (2 * ds)
    assert np.allclose(v_fd, greeks.vega(F, K, T, R_RATE, s), rtol=1e-5, atol=1e-6)
    assert np.all(greeks.vega(F, K, T, R_RATE, s) > 0)
    vn_fd = (greeks.delta(F, K, T, R_RATE, s + ds, True) - greeks.delta(F, K, T, R_RATE, s - ds, True)) / (2 * ds)
    assert np.allclose(vn_fd, greeks.vanna(F, K, T, R_RATE, s), atol=1e-6)


def test_spot_gamma_applies_the_carry_factor_squared():
    F, K, T, s = 24000.0, 24100.0, 3 / 252, 0.15
    base = greeks.gamma(F, K, T, R_RATE, s)
    assert greeks.spot_gamma(F, K, T, R_RATE, s, carry=0.0) == pytest.approx(base)
    assert greeks.spot_gamma(F, K, T, R_RATE, s, carry=0.06) == pytest.approx(base * np.exp(2 * 0.06 * T))


def test_implied_vol_round_trip_on_a_grid():
    F, K, T, s = _grid()
    for is_call in (True, False):
        p = greeks.price(F, K, T, R_RATE, s, is_call)
        iv = greeks.implied_vol(p, F, K, T, R_RATE, is_call)
        tv = p - np.exp(-R_RATE * T) * np.maximum((F - K) if is_call else (K - F), 0.0)
        informative = (tv > 1e-6 * F) & (greeks.vega(F, K, T, R_RATE, s) > 1e-3)
        assert informative.sum() > 40
        assert np.allclose(iv[informative], s[informative], atol=1e-7)


def test_implied_vol_is_nan_outside_no_arbitrage_bounds_and_for_bad_maturity():
    F, K, T = 100.0, 100.0, 0.05
    disc = np.exp(-R_RATE * T)
    bad = np.array([-1.0, 0.0, disc * F, disc * F + 1.0, np.nan])
    assert np.isnan(greeks.implied_vol(bad, F, K, T, R_RATE, True)).all()
    itm = disc * 20.0  # deep ITM call priced at its intrinsic value or below
    assert np.isnan(greeks.implied_vol(np.array([itm - 0.01, itm]), F, 80.0, T, R_RATE, True)).all()
    assert np.isnan(greeks.implied_vol(np.array([1.0]), F, K, 0.0, R_RATE, True)).all()


def test_implied_vol_is_monotone_in_price():
    F, K, T = 100.0, 101.0, 0.04
    prices = greeks.price(F, K, T, R_RATE, np.linspace(0.05, 1.0, 40), True)
    iv = greeks.implied_vol(prices, F, K, T, R_RATE, True)
    assert np.all(np.diff(iv) > 0)


def test_implied_vol_clamps_at_the_search_bracket_edges_documented():
    # Documented limitation (reports/CODE_AUDIT.md, A1-A-006): a price that needs sigma outside [1e-4, 5] returns the
    # bracket edge, not NaN. Real Nifty implied vols are far inside the bracket, so no committed number is affected.
    F, K, T = 100.0, 100.0, 0.05
    hi = greeks.implied_vol(greeks.price(F, K, T, R_RATE, 6.0, True), F, K, T, R_RATE, True)
    lo = greeks.implied_vol(greeks.price(F, K, T, R_RATE, 5e-5, True), F, K, T, R_RATE, True)
    assert float(hi) == pytest.approx(5.0, abs=1e-6) and float(lo) == pytest.approx(1e-4, abs=1e-9)


def test_degenerate_option_inputs_raise_or_give_nan():
    # Changed after the validation audit (A3-09, A3-13): T = 0 used to give NaN gamma and a net GEX of 0.0
    # with only a RuntimeWarning; the GEX engine now refuses it. IV = 0 is treated as missing (NaN) with a
    # GexInputWarning. The greeks functions themselves still return NaN.
    chain = pd.DataFrame({"strike": [95.0, 105.0], "is_call": [True, False], "oi": [10.0, 10.0], "iv": [0.2, 0.2]})
    with pytest.raises(ValueError, match="T"):
        gex.strike_gex(chain, 100.0, 1.0, 0.0)
    with pytest.warns(gex.GexInputWarning, match="IV 0"):
        assert np.isnan(gex.strike_gex(chain.assign(iv=0.0), 100.0, 1.0, 0.02)["gex"]).all()
    assert np.isnan(greeks.price(100.0, 100.0, 0.0, 0.0, 0.2, True))


# --------------------------------------------------------------------------------------------
# GEX: units and sign rules
# --------------------------------------------------------------------------------------------


def _chain(spot=24000.0, n=9, oi=1000.0):
    K = spot * (1 + 0.01 * np.arange(-n // 2 + 1, n // 2 + 1))
    return pd.DataFrame({
        "strike": np.r_[K, K], "is_call": np.r_[np.ones(len(K), bool), np.zeros(len(K), bool)],
        "oi": np.full(2 * len(K), oi), "iv": np.full(2 * len(K), 0.15)})


def test_gross_gex_follows_the_documented_formula_and_scales_linearly():
    spot, lot, T = 24000.0, 75.0, 3 / 252
    ch = _chain(spot).iloc[[3]]
    out = gex.strike_gex(ch, spot, lot, T, R_RATE)
    g = greeks.gamma(spot, ch["strike"].iloc[0], T, R_RATE, 0.15)
    assert out["gross_gex"].iloc[0] == pytest.approx(g * ch["oi"].iloc[0] * lot * spot**2 * 0.01, rel=1e-12)
    twice_oi = gex.strike_gex(ch.assign(oi=ch["oi"] * 2), spot, lot, T, R_RATE)
    assert twice_oi["gross_gex"].iloc[0] == pytest.approx(2 * out["gross_gex"].iloc[0])
    # keeping open interest in shares with lot size 1 equals open interest in contracts with the real lot size
    shares = gex.strike_gex(ch.assign(oi=ch["oi"] * lot), spot, 1.0, T, R_RATE)
    assert shares["gross_gex"].iloc[0] == pytest.approx(out["gross_gex"].iloc[0])


def test_sign_rules_flip_as_documented():
    spot, lot, T = 24000.0, 75.0, 3 / 252
    ch = _chain(spot)
    net = {r: gex.strike_gex(ch, spot, lot, T, R_RATE, rule=r)["gex"].sum() for r in ("us_convention", "long_all", "short_all")}
    gross = gex.strike_gex(ch, spot, lot, T, R_RATE, rule="long_all")["gross_gex"].sum()
    assert net["long_all"] == pytest.approx(gross) and net["short_all"] == pytest.approx(-gross)
    # equal call and put open interest at the same strikes and vols: the US convention nets to (almost) zero
    assert abs(net["us_convention"]) < 1e-9 * gross
    calls_only = ch[ch["is_call"]]
    puts_only = ch[~ch["is_call"]]
    assert gex.strike_gex(calls_only, spot, lot, T, R_RATE)["gex"].sum() > 0
    assert gex.strike_gex(puts_only, spot, lot, T, R_RATE)["gex"].sum() < 0
    # swapping the call and put labels negates the US-convention net GEX
    swapped = ch.assign(is_call=~ch["is_call"])
    big_calls = ch.assign(oi=np.where(ch["is_call"], 3000.0, 1000.0))
    big_puts = big_calls.assign(is_call=~big_calls["is_call"])
    a = gex.strike_gex(big_calls, spot, lot, T, R_RATE)["gex"].sum()
    b = gex.strike_gex(big_puts, spot, lot, T, R_RATE)["gex"].sum()
    assert a > 0 and b == pytest.approx(-a, rel=1e-12)
    assert swapped["is_call"].sum() == (~ch["is_call"]).sum()


def test_participant_rules_need_weights_and_unknown_rule_raises():
    with pytest.raises(ValueError):
        gex.sign_weights(np.array([True, False]), "participant")
    with pytest.raises(ValueError):
        gex.sign_weights(np.array([True]), "nonsense")
    w = gex.sign_weights(np.array([True, False]), "client_mirror", gex.ParticipantWeights(call=-0.2, put=0.4))
    assert w.tolist() == [-0.2, 0.4]


def test_participant_weights_mirror_flips_sign_and_is_clipped():
    tbl = pd.DataFrame(
        {"option_index_call_long": [300.0, 100.0, 400.0], "option_index_call_short": [100.0, 300.0, 400.0],
         "option_index_put_long": [50.0, 150.0, 200.0], "option_index_put_short": [150.0, 50.0, 200.0]},
        index=["Pro", "Client", "TOTAL"])
    a = gex.ParticipantWeights.from_participant_oi(tbl, "Pro")
    b = gex.ParticipantWeights.from_participant_oi(tbl, "Pro", mirror=True)
    assert a.call == pytest.approx(0.5) and a.put == pytest.approx(-0.5)
    assert (b.call, b.put) == (-a.call, -a.put)
    tbl.loc["Pro", "option_index_call_long"] = 5000.0  # a share above one is clipped, not passed on
    assert gex.ParticipantWeights.from_participant_oi(tbl, "Pro").call == 1.0
    tbl.loc["TOTAL", ["option_index_call_long", "option_index_put_long"]] = 0.0  # empty market: weight 0, no division error
    z = gex.ParticipantWeights.from_participant_oi(tbl, "Pro")
    assert (z.call, z.put) == (0.0, 0.0)


def test_gamma_flip_is_a_root_of_net_gex_and_nan_when_the_sign_never_changes():
    spot, lot, T = 24000.0, 75.0, 3 / 252
    K = np.array([23500.0, 23800.0, 24000.0, 24200.0, 24500.0])
    ch = pd.DataFrame({"strike": np.r_[K, K], "is_call": np.r_[np.ones(5, bool), np.zeros(5, bool)],
                       "oi": np.r_[[0, 0, 500, 3000, 3000], [3000, 3000, 500, 0, 0]].astype(float), "iv": 0.14})
    # puts below and calls above spot: net GEX is a mix of signs and changes sign somewhere inside the grid
    ch.loc[ch["is_call"] & (ch["strike"] <= 24000), "oi"] = 100.0
    flip = gex.gamma_flip(ch, spot + 150.0, lot, T, R_RATE, width=0.05)
    if np.isfinite(flip):
        assert gex.net_gex_at(ch, flip, lot, T, R_RATE) == pytest.approx(0.0, abs=1e-6 * abs(
            gex.strike_gex(ch, flip, lot, T, R_RATE, rule="long_all")["gross_gex"].sum()))
    assert np.isnan(gex.gamma_flip(ch, spot, lot, T, R_RATE, rule="long_all"))
    assert np.isnan(gex.gamma_flip(ch, spot, lot, T, R_RATE, rule="short_all"))


def test_oi_walls_and_max_pain_on_a_hand_computed_chain():
    ch = pd.DataFrame({"strike": [100.0, 110.0, 120.0, 100.0, 110.0, 120.0],
                       "is_call": [True, True, True, False, False, False],
                       "oi": [10.0, 50.0, 5.0, 40.0, 10.0, 5.0], "iv": 0.2})
    walls = gex.oi_walls(ch)
    assert walls["call_wall"] == 110.0 and walls["put_wall"] == 100.0 and walls["max_oi_strike"] == 110.0
    # payouts at settlement 100 / 110 / 120: 100 -> 50*10+5*20=600 ; 110 -> 10*10+10*... computed by brute force
    best = min(ch["strike"].unique(), key=lambda s: (np.where(ch["is_call"], np.maximum(s - ch["strike"], 0),
                                                          np.maximum(ch["strike"] - s, 0)) * ch["oi"]).sum())
    assert gex.max_pain(ch) == best


# --------------------------------------------------------------------------------------------
# realized measures
# --------------------------------------------------------------------------------------------


def test_realized_measures_basic_identities():
    rng = np.random.default_rng(1)
    r = rng.standard_normal((4, 375)) * 5e-4
    assert np.allclose(realized.realized_variance(r), (r**2).sum(axis=1))
    k5 = realized.realized_variance(r, k=5)
    assert np.allclose(k5, (r.reshape(4, 75, 5).sum(axis=2) ** 2).sum(axis=1))
    bv = realized.bipower_variation(r)
    assert bv.shape == (4,) and np.all(bv > 0)
    js = realized.jump_share(r)
    assert np.all((js >= 0) & (js <= 1))
    jumpy = r.copy()
    jumpy[:, 100] += 0.02
    assert np.all(realized.jump_share(jumpy) > 0.5)
    assert np.all(realized.bipower_variation(jumpy) < 2 * bv + 1e-6)  # robust to one jump
    assert realized.jump_share(np.zeros(10)) == 0.0


def test_rogers_satchell_and_bulk_volume_ranges():
    o, c = 100.0, 100.5
    h, l = 101.0, 99.5
    assert realized.rogers_satchell(o, h, l, c) > 0
    assert realized.rogers_satchell(o, o, o, o) == 0.0
    v = np.array([100.0, 100.0, 100.0])
    imb = realized.bulk_volume_imbalance(np.array([-10.0, 0.0, 10.0]), v, 1.0)
    assert imb[1] == pytest.approx(0.0) and imb[0] == pytest.approx(-imb[2]) and np.all(np.abs(imb) <= v)


# --------------------------------------------------------------------------------------------
# forecast evaluation
# --------------------------------------------------------------------------------------------


def test_qlike_is_minimised_at_the_sample_mean_and_homogeneous():
    rng = np.random.default_rng(2)
    sigma2 = 4e-6
    y = sigma2 * rng.standard_normal(4000) ** 2  # realized-variance proxies with E[y] = sigma2
    grid = sigma2 * np.linspace(0.5, 1.6, 2201)
    loss = np.array([ev.qlike(y, f).mean() for f in grid])
    assert grid[loss.argmin()] == pytest.approx(y.mean(), rel=2e-3)  # the minimiser is exactly the mean of y
    assert abs(y.mean() / sigma2 - 1) < 0.1
    assert np.all(ev.qlike(y, 2 * y) > 0) and ev.qlike(3.0, 3.0) == 0.0
    assert np.allclose(ev.qlike(7 * y, 7 * y[::-1]), ev.qlike(y, y[::-1]))  # scale free
    assert ev.qlike(2.0, 1.0) != ev.qlike(1.0, 2.0)  # asymmetric: under-forecasting is punished more
    assert ev.qlike(2.0, 1.0) > ev.qlike(1.0, 2.0)
    assert ev.mse(2.0, 1.0) == 1.0


def test_diebold_mariano_sign_symmetry_and_degenerate_cases():
    rng = np.random.default_rng(3)
    base = rng.exponential(1.0, 400)
    better = base * 0.9 + rng.normal(0, 0.02, 400)
    g = ev.diebold_mariano(base, better, alternative="greater")  # model 2 has lower loss
    assert g["stat"] > 0 and g["p"] < 0.01
    assert ev.diebold_mariano(better, base, alternative="greater")["p"] > 0.99
    two = ev.diebold_mariano(base, better)
    assert two["p"] == pytest.approx(2 * min(g["p"], 1 - g["p"]), rel=1e-9)
    assert ev.diebold_mariano(base, base)["p"] == 1.0
    assert ev.diebold_mariano(base, better, lag=0)["n"] == 400


def test_diebold_mariano_returns_nan_instead_of_crashing_on_empty_input():
    out = ev.diebold_mariano(np.array([]), np.array([]))
    assert out["n"] == 0 and np.isnan(out["p"]) and np.isnan(out["stat"])
    out = ev.diebold_mariano(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert out["n"] == 0 and np.isnan(out["p"])


def test_holm_matches_a_hand_computation_and_keeps_nan():
    p = np.array([0.01, 0.04, 0.03, np.nan])
    out = ev.holm(p)
    assert out[0] == pytest.approx(0.03) and out[2] == pytest.approx(0.06) and out[1] == pytest.approx(0.06)
    assert np.isnan(out[3])
    assert np.all(out[np.isfinite(out)] >= p[np.isfinite(p)])


def test_trimmed_mean_removes_only_the_upper_tail():
    x = np.r_[np.ones(99), 1000.0]
    assert ev.trimmed_mean(x, 0.01) == pytest.approx(1.0)
    assert ev.trimmed_mean(x, 0.0) == pytest.approx(x.mean())
    assert np.isnan(ev.trimmed_mean(np.array([]), 0.0))


def test_day_cluster_bootstrap_is_seeded_and_resamples_whole_days():
    rng = np.random.default_rng(4)
    before, after = rng.normal(0, 1, (30, 20)), rng.normal(0.5, 1, (30, 20))
    a = inference.day_cluster_bootstrap(before, after, n_boot=200, seed=5)
    b = inference.day_cluster_bootstrap(before, after, n_boot=200, seed=5)
    c = inference.day_cluster_bootstrap(before, after, n_boot=200, seed=6)
    assert a == b and a != c
    assert a["ci_low"] < a["diff"] < a["ci_high"]


# --------------------------------------------------------------------------------------------
# expiry calendar (documented rule, checked on known dates; trading days built by us, so OUR CHECK)
# --------------------------------------------------------------------------------------------


def _days(start, end, holidays=()):
    d = pd.bdate_range(start, end)
    return d.difference(pd.DatetimeIndex(holidays))


def test_weekly_expiry_weekday_switches_from_thursday_to_tuesday():
    ex = weekly_expiries(_days("2025-08-04", "2025-09-30"))
    thu = [d for d in ex if d <= pd.Timestamp("2025-08-31")]
    tue = [d for d in ex if d >= pd.Timestamp("2025-09-01")]
    assert all(d.dayofweek == 3 for d in thu) and thu[-1] == pd.Timestamp("2025-08-28")
    assert all(d.dayofweek == 1 for d in tue) and tue[0] == pd.Timestamp("2025-09-02")
    assert len(thu) == 4 and len(tue) == 5


def test_holiday_on_the_scheduled_day_moves_expiry_to_the_previous_trading_day():
    hol = [pd.Timestamp("2025-04-10")]  # a Thursday
    ex = weekly_expiries(_days("2025-04-01", "2025-04-30", hol))
    assert pd.Timestamp("2025-04-09") in ex and pd.Timestamp("2025-04-10") not in ex
    ex2 = weekly_expiries(_days("2025-09-22", "2025-10-10", [pd.Timestamp("2025-09-30")]))  # a Tuesday holiday
    assert pd.Timestamp("2025-09-29") in ex2


def test_regime_boundaries_and_days_to_expiry():
    days = _days("2024-11-11", "2025-09-12")
    lab = label_days(days)
    assert lab.loc["2024-11-19", "regime"] == "multi_weekly" and lab.loc["2024-11-20", "regime"] == "single_weekly_thu"
    assert lab.loc["2025-08-29", "regime"] == "single_weekly_thu" and lab.loc["2025-09-01", "regime"] == "single_weekly_tue"
    exp = lab[lab["is_expiry"]]
    assert (lab.loc[exp.index, "trading_days_to_expiry"] == 0).all()
    wed = lab.loc["2025-07-02"]  # Wednesday: next expiry is Thursday 3 July
    assert wed["trading_days_to_expiry"] == 1 and wed["next_expiry"] == pd.Timestamp("2025-07-03")
    monthly = lab[lab["is_monthly_expiry"]].index
    assert pd.Timestamp("2025-07-31") in monthly and pd.Timestamp("2025-07-24") not in monthly


@pytest.mark.xfail(strict=True, reason="A1-A-004: days after the last in-sample expiry get no days-to-expiry, and "
                                       "features.build_dataset turns that NaN into 0 (= 'expiry day')")
def test_days_after_the_last_in_sample_expiry_still_get_a_days_to_expiry():
    days = _days("2025-08-18", "2025-09-11")  # ends on a Thursday; the next Tuesday expiry is outside the sample
    lab = label_days(days)
    assert lab["trading_days_to_expiry"].notna().all()


# --------------------------------------------------------------------------------------------
# implied-variance panel
# --------------------------------------------------------------------------------------------


def test_expiry_summary_recovers_the_generating_vol_and_forward():
    F, T_days, sig = 24000.0, 5, 0.18
    T = T_days / 252
    K = F * np.exp(np.linspace(-0.08, 0.08, 33))
    rows = []
    for k in K:
        for is_call in (True, False):
            rows.append({"strike": k, "is_call": is_call, "settle": float(greeks.price(F, k, T, R_RATE, sig, is_call)),
                         "oi": 1000.0, "volume": 10.0})
    out = implied.expiry_summary(pd.DataFrame(rows), T_days, R_RATE, spot_hint=F)
    assert out["F"] == pytest.approx(F, rel=2e-3)  # parity forward from the strike with the smallest |C - P|
    assert out["iv_atm"] == pytest.approx(sig, abs=1e-3)
    assert out["straddle_var"] == pytest.approx(sig**2 * T, rel=0.02)  # normal approximation of the straddle
    assert out["call_oi"] == out["put_oi"]


def test_trading_day_counter_excludes_start_includes_end():
    days = pd.bdate_range("2025-01-01", "2025-01-31").difference(pd.DatetimeIndex(["2025-01-27"]))
    c = implied.TradingDayCounter(days)
    assert c("2025-01-24", "2025-01-28") == 1  # Mon 27 is a holiday, so only Tue 28 lies in (Fri 24, Tue 28]
    assert c("2025-01-02", "2025-01-03") == 1
    assert c("2025-01-30", "2025-02-04") == 1 + 2  # 31 Jan, then 3 Feb and 4 Feb beyond the known days


# --------------------------------------------------------------------------------------------
# no look-ahead: features and walk-forward forecasts
# --------------------------------------------------------------------------------------------

CUT = 75  # first perturbed day; the first walk-forward block is days 60 to 74


def _market_pair(seed=21, n_days=90):
    m = simulate_market(SimConfig(n_days=n_days, kappa=0.0, seed=seed))
    m2 = copy.deepcopy(m)
    rng = np.random.default_rng(99)
    # a clear but moderate change: with factor 2 the EGARCH sanity check flips to EWMA, see the xfail test below
    m2["R"][CUT:] = m2["R"][CUT:] * 1.5
    for k in ("open", "high", "low", "close"):
        m2[k][CUT:] = m2[k][CUT:] * rng.uniform(0.7, 1.3)
    m2["volume"][CUT:] = m2["volume"][CUT:] * 5.0
    m2["vix"] = np.asarray(m2["vix"], float).copy()
    m2["vix"][CUT:] *= 3.0
    for t in range(CUT, n_days):
        ch = m2["chains"][t].copy()
        ch["oi"] = ch["oi"] * rng.uniform(0.2, 5.0, len(ch))
        ch.attrs.update(m["chains"][t].attrs)
        m2["chains"][t] = ch
    if m2["participant"] is not None:
        m2["participant"] = m2["participant"].copy()
        m2["participant"].iloc[CUT:] = -m2["participant"].iloc[CUT:]
    m2["dealer_next"] = m2["dealer_next"].copy()
    m2["dealer_next"].iloc[CUT:] = -m2["dealer_next"].iloc[CUT:]
    return m, m2


def test_features_for_days_before_the_cut_do_not_depend_on_later_days():
    m, m2 = _market_pair()
    a = build_dataset(m, build_gex_panel(m), horizon=30, step=45)
    b = build_dataset(m2, build_gex_panel(m2), horizon=30, step=45)
    assert not np.allclose(a["y"][a["t"] >= CUT], b["y"][b["t"] >= CUT])  # the perturbation is real
    pa, pb = a[a["t"] < CUT].reset_index(drop=True), b[b["t"] < CUT].reset_index(drop=True)
    pd.testing.assert_frame_equal(pa, pb, check_exact=True)
    # Day CUT itself uses days before it for everything that is not an own-day minute feature
    own_day = ["y", "log_rv_back_h", "log_rv_back_15", "log_rv_back_60", "log_rv_today_pm", "log_vol15_rel", "ofi15",
               "abs_ret_today_sd", "dist_flip_us", "dist_flip_participant", "dist_max_oi", "dist_max_pain",
               "dist_call_wall", "dist_put_wall", "log_rv_prevday_same"]
    ca, cb = a[a["t"] == CUT].reset_index(drop=True), b[b["t"] == CUT].reset_index(drop=True)
    day_level = [c for c in ca.columns if c not in own_day and c in cb.columns and ca[c].dtype.kind == "f"
                 and not c.startswith("pl_")]
    for c in day_level:
        if c.startswith(("gex_", "gross_", "atm_", "oi_change")):
            continue  # these use the chain of day CUT-1 (unchanged) but are scaled by a spot-relative level
        assert np.allclose(ca[c].to_numpy(), cb[c].to_numpy(), equal_nan=True), c


def test_walk_forward_forecasts_for_a_block_do_not_depend_on_days_after_it():
    m, m2 = _market_pair()
    cfg = WalkForwardConfig(min_train=60, block=15, models=["naive_last", "har", "mcsgarch", "hybrid_base"])
    a, info_a = run_walkforward(m, build_dataset(m, build_gex_panel(m), horizon=30, step=45), cfg)
    b, info_b = run_walkforward(m2, build_dataset(m2, build_gex_panel(m2), horizon=30, step=45), cfg)
    first_a, first_b = a[a["t"] < CUT].reset_index(drop=True), b[b["t"] < CUT].reset_index(drop=True)
    assert first_a["t"].min() == 60 and first_a["t"].max() == CUT - 1
    assert info_a["mcs_params"][0]["daily_method"] == info_b["mcs_params"][0]["daily_method"] == "egarch_t"
    # Day-level inputs and the naive forecast are exactly unchanged. The fitted models differ by at most about 1e-4 in
    # relative terms (A1-A-008): arch's EGARCH filter with fixed parameters returns slightly different variances for
    # early days when the series passed to it is longer. Tolerance 1e-3 leaves a 10-fold margin over what was measured.
    pd.testing.assert_frame_equal(first_a[["t", "minute", "y", "naive_last"]], first_b[["t", "minute", "y", "naive_last"]],
                                  check_exact=True)
    for col in ("har", "mcsgarch", "hybrid_base"):
        assert np.allclose(first_a[col], first_b[col], rtol=1e-3, atol=0), col
    assert not np.allclose(a[a["t"] >= CUT]["mcsgarch"], b[b["t"] >= CUT]["mcsgarch"], rtol=1e-2)  # later blocks do react


def test_walk_forward_is_deterministic_for_a_fixed_seed():
    m = simulate_market(SimConfig(n_days=80, kappa=0.0, seed=5))
    ds = build_dataset(m, build_gex_panel(m), horizon=30, step=45)
    cfg = WalkForwardConfig(min_train=60, block=10, models=["har", "mcsgarch", "lgbm_base", "linear_base"])
    p1, _ = run_walkforward(m, ds, cfg)
    p2, _ = run_walkforward(m, ds, cfg)
    pd.testing.assert_frame_equal(p1, p2, check_exact=True)
    m_again = simulate_market(SimConfig(n_days=80, kappa=0.0, seed=5))
    assert np.array_equal(m["R"], m_again["R"])
    assert not np.array_equal(m["R"], simulate_market(SimConfig(n_days=80, kappa=0.0, seed=6))["R"])


def test_daily_egarch_variance_uses_only_earlier_days_once_fitted():
    rng = np.random.default_rng(7)
    daily = rng.standard_normal(300) * 0.008
    h1, info = daily_egarch_variance(daily, train_size=200)
    assert info["method"].startswith("egarch")
    d2 = daily.copy()
    d2[250:] *= 1.8  # a clear rise that stays inside the sanity band of the fit
    h2, info2 = daily_egarch_variance(d2, train_size=200)
    assert info2["method"] == info["method"]
    assert np.allclose(h1[:251], h2[:251], rtol=2e-2)  # h_t for t <= 250 uses returns up to t - 1 (plus arch's tiny end effect)
    assert not np.allclose(h1[251:], h2[251:])


@pytest.mark.xfail(strict=True, reason="A1-A-002 (disclosed in docs/MODEL_ITERATIONS_REAL.md): _egarch_ok checks the filtered "
                                       "variance of ALL days, so a later shock can switch earlier days from EGARCH to EWMA")
def test_egarch_versus_ewma_choice_does_not_depend_on_days_after_the_training_window():
    rng = np.random.default_rng(7)
    daily = rng.standard_normal(300) * 0.008
    _, info = daily_egarch_variance(daily, train_size=200)
    shocked = daily.copy()
    shocked[250:] *= 30  # variance x900, more than the 5 log-point band of _egarch_ok
    _, info2 = daily_egarch_variance(shocked, train_size=200)
    assert info2["method"] == info["method"]


def test_ewma_variance_does_not_depend_on_data_after_the_training_window_in_the_normal_case():
    rng = np.random.default_rng(8)
    y = rng.standard_normal(250) * 0.8
    base = _ewma_variance(y, n_init=200)
    shocked = y.copy()
    shocked[230:] *= 2.0  # an ordinary rise in volatility after the cut: the floor (1% of the mean square) is inactive
    assert np.allclose(base[:230], _ewma_variance(shocked, n_init=200)[:230])


@pytest.mark.xfail(strict=True, reason="A1-A-003: the EWMA fallback floors h at 1% of the mean square of the WHOLE "
                                       "series, which includes days after the training window")
def test_ewma_variance_floor_does_not_look_ahead():
    y = np.r_[np.random.default_rng(0).standard_normal(100) * 0.01, np.ones(5) * 50.0]
    assert np.allclose(_ewma_variance(y[:100], n_init=100), _ewma_variance(y, n_init=100)[:100])


# --------------------------------------------------------------------------------------------
# missing minute bars
# --------------------------------------------------------------------------------------------


def test_missing_minute_bar_loses_the_move_across_the_gap_documented():
    # Documented limitation (A1-A-005): a missing bar gives NaN for its own return AND the next one, because the
    # next return is measured against the missing close; build_market later zero-fills them, so the price move
    # across the gap never enters realized variance.
    idx = pd.date_range("2025-01-06 09:15", periods=10, freq="min")
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0 + np.arange(10) * 0.1, "volume": 1.0},
                        index=idx).drop(idx[4])
    grid = to_session_grid(bars, minutes=10)
    r = grid["returns"][0]
    assert np.isnan(r[4]) and np.isnan(r[5]) and np.isfinite(np.delete(r, [4, 5])).all()
    assert np.isnan(grid["close"][0, 4]) and np.isnan(grid["volume"][0, 4])


def test_one_missing_bar_blanks_the_minute_features_for_the_rest_of_that_day_documented():
    # Documented limitation (A1-A-005): close and volume are NaN for a missing bar and are cumulated, so every anchor
    # after the gap on that day has NaN order-flow, volume and distance-moved features. Other days are unaffected.
    m, _ = _market_pair()
    m["close"][10, 100] = np.nan
    m["volume"][10, 100] = np.nan
    ds = build_dataset(m, build_gex_panel(m), horizon=30, step=15)
    day = ds[ds["t"] == 10]
    after, before = day[day["minute"] > 100], day[day["minute"] <= 90]
    assert after[["ofi15", "log_vol15_rel", "abs_ret_today_sd"]].isna().all().all()
    assert not before[["ofi15", "log_vol15_rel"]].isna().any().any()
    assert not ds[ds["t"] == 11][["ofi15", "log_vol15_rel"]].isna().any().any()
