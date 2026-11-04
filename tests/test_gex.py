import numpy as np
import pandas as pd
import pytest

from niftygex import gex, greeks


def make_chain(spot=24000.0):
    strikes = np.arange(23000, 25001, 100.0)
    call_oi = np.exp(-0.5 * ((strikes - 24300) / 300) ** 2) * 1e5
    put_oi = np.exp(-0.5 * ((strikes - 23700) / 300) ** 2) * 1e5
    return pd.DataFrame({
        "strike": np.r_[strikes, strikes],
        "is_call": np.r_[np.ones(len(strikes), bool), np.zeros(len(strikes), bool)],
        "oi": np.r_[call_oi, put_oi],
        "iv": 0.13,
    })


def test_gex_matches_hand_formula():
    ch = make_chain()
    t = gex.strike_gex(ch, 24000.0, 75, 3 / 365, rule="long_all")
    g = greeks.gamma(24000.0, ch["strike"], 3 / 365, 0.065, 0.13)
    expected = np.sum(g * ch["oi"] * 75 * 24000.0**2 * 0.01)
    assert np.isclose(t["gex"].sum(), expected)


def test_gex_is_rupee_delta_change_for_one_percent_move():
    # Finite difference: change in total delta (in rupees) when spot moves 0.1%.
    ch = make_chain()
    S, T = 24000.0, 3 / 365
    def rupee_delta(s):
        d = greeks.delta(s, ch["strike"].to_numpy(), T, 0.065, 0.13, ch["is_call"].to_numpy())
        return np.sum(d * ch["oi"].to_numpy() * 75) * s
    dS = 0.0001 * S
    # d(delta notional)/dS * S * 1% ignoring the second-order delta * dS term
    num = (rupee_delta(S + dS) - rupee_delta(S - dS)) / (2 * dS)
    d = greeks.delta(S, ch["strike"].to_numpy(), T, 0.065, 0.13, ch["is_call"].to_numpy())
    delta_term = np.sum(d * ch["oi"].to_numpy() * 75)
    gamma_part = (num - delta_term) * S * 0.01
    assert np.isclose(gamma_part, gex.strike_gex(ch, S, 75, T, rule="long_all")["gex"].sum(), rtol=1e-4)


def test_sign_rules_and_bounds():
    ch = make_chain()
    kw = dict(spot=24000.0, lot_size=75, T=3 / 365)
    up = gex.strike_gex(ch, rule="long_all", **kw)["gex"].sum()
    lo = gex.strike_gex(ch, rule="short_all", **kw)["gex"].sum()
    us = gex.strike_gex(ch, rule="us_convention", **kw)["gex"].sum()
    pw = gex.ParticipantWeights(call=-0.3, put=-0.2)
    pa = gex.strike_gex(ch, rule="participant", participant=pw, **kw)["gex"].sum()
    assert lo < us < up and lo < pa < up and np.isclose(up, -lo)
    with pytest.raises(ValueError):
        gex.strike_gex(ch, rule="participant", **kw)


def test_gamma_flip_between_put_and_call_mass():
    ch = make_chain()
    flip = gex.gamma_flip(ch, 24000.0, 75, 3 / 365, rule="us_convention")
    # Symmetric call and put OI around 24000 -> flip near 24000 under the US rule.
    assert abs(flip - 24000.0) < 60
    assert np.isnan(gex.gamma_flip(ch, 24000.0, 75, 3 / 365, rule="long_all"))


def test_walls_and_max_pain():
    ch = make_chain()
    w = gex.oi_walls(ch)
    assert w["call_wall"] == 24300 and w["put_wall"] == 23700
    assert 23700 <= gex.max_pain(ch) <= 24300


def test_participant_weights_from_table():
    tbl = pd.DataFrame({
        "option_index_call_long": [600.0, 50, 150, 200, 1000],
        "option_index_call_short": [500.0, 0, 200, 300, 1000],
        "option_index_put_long": [700.0, 20, 130, 150, 1000],
        "option_index_put_short": [400.0, 0, 300, 300, 1000],
    }, index=["Client", "DII", "FII", "Pro", "TOTAL"])
    pw = gex.ParticipantWeights.from_participant_oi(tbl, "Pro")
    assert np.isclose(pw.call, -0.1) and np.isclose(pw.put, -0.15)
    cm = gex.ParticipantWeights.from_participant_oi(tbl, "Client", mirror=True)
    assert np.isclose(cm.call, -0.1) and np.isclose(cm.put, -0.3)
