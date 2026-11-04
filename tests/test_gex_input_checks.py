"""Input checks of the GEX engine (validation audit A3, reports/stress/a3_gex_stress_cases.csv).

Each test names the A3 case it covers. Before these checks the engine accepted
every one of these inputs and returned a normal looking number (often exactly
0.0). SYNTHETIC chains only.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from niftygex import gex
from niftygex.gex import GexInputWarning

SPOT, LOT, T = 24000.0, 75.0, 3 / 365


def _chain() -> pd.DataFrame:
    strikes = np.arange(22800.0, 25250.0, 100.0)
    rows = []
    for k in strikes:
        for is_call in (True, False):
            otm = (k >= SPOT) if is_call else (k <= SPOT)
            oi = 4.0e4 * np.exp(-((k - SPOT) / 600.0) ** 2) * (1.6 if otm else 0.6)
            rows.append(dict(strike=k, is_call=is_call, oi=oi, iv=0.12 + 0.1 * abs(np.log(k / SPOT))))
    return pd.DataFrame(rows)


def _clean_net() -> float:
    return gex.net_gex_at(_chain(), SPOT, LOT, T)


def test_clean_chain_gives_no_warning_and_the_same_numbers_as_the_formula():
    c = _chain()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = gex.strike_gex(c, SPOT, LOT, T)
        row = gex.daily_summary(c, SPOT, LOT, T)
    assert np.isfinite(row["net_gex_us_convention"]) and row["net_gex_us_convention"] == pytest.approx(t["gex"].sum())
    # the returned table keeps the caller's columns and dtypes
    assert list(t.columns[:4]) == ["strike", "is_call", "oi", "iv"] and t["is_call"].dtype == bool


def test_returned_table_keeps_integer_strikes_and_open_interest():
    c = _chain()
    c["strike"] = c["strike"].astype(int)
    c["oi"] = c["oi"].round().astype(int)
    t = gex.strike_gex(c, SPOT, LOT, T)
    assert t["strike"].dtype == c["strike"].dtype and t["oi"].dtype == c["oi"].dtype


@pytest.mark.parametrize("col", ["iv", "oi", "strike", "is_call"])
def test_missing_column_names_the_schema(col):  # A3-33, A3-34
    with pytest.raises(ValueError, match="missing column"):
        gex.strike_gex(_chain().drop(columns=col), SPOT, LOT, T)


def test_upper_case_column_is_reported_as_missing():  # A3-34
    with pytest.raises(ValueError, match="lower case"):
        gex.strike_gex(_chain().rename(columns={"oi": "OI"}), SPOT, LOT, T)


def test_option_type_strings_are_refused():  # A3-28
    c = _chain()
    c["is_call"] = np.where(c["is_call"], "CE", "PE")
    with pytest.raises(ValueError, match="CE"):
        gex.strike_gex(c, SPOT, LOT, T)
    with pytest.raises(ValueError, match="is_call"):
        gex.daily_summary(c, SPOT, LOT, T)


def test_option_type_as_one_and_zero_still_works():  # A3-29
    c = _chain()
    c["is_call"] = c["is_call"].astype(int)
    assert gex.net_gex_at(c, SPOT, LOT, T) == pytest.approx(_clean_net())
    c["is_call"] = c["is_call"] * 2
    with pytest.raises(ValueError, match="other numbers"):
        gex.net_gex_at(c, SPOT, LOT, T)


def test_negative_open_interest_is_refused():  # A3-03
    c = _chain()
    c.loc[c.index[::7], "oi"] *= -1
    with pytest.raises(ValueError, match="negative open interest"):
        gex.strike_gex(c, SPOT, LOT, T)


def test_negative_iv_is_refused():  # A3-14
    c = _chain()
    c["iv"] = -c["iv"]
    with pytest.raises(ValueError, match="negative implied volatility"):
        gex.strike_gex(c, SPOT, LOT, T)


@pytest.mark.parametrize("name,kw", [
    ("spot", dict(spot=0.0)), ("spot", dict(spot=-SPOT)), ("spot", dict(spot=np.nan)),  # A3-17, A3-18
    ("lot_size", dict(lot_size=0.0)), ("lot_size", dict(lot_size=-75.0)),  # A3-19, A3-20
    ("T", dict(T=0.0)), ("T", dict(T=-1 / 365)), ("T", dict(T=np.inf)),  # A3-09, A3-10
])
def test_nonpositive_or_nonfinite_scalars_are_refused(name, kw):
    args = dict(spot=SPOT, lot_size=LOT, T=T) | kw
    for fn in (gex.strike_gex, gex.net_gex_at, gex.gamma_flip, gex.daily_summary):
        with pytest.raises(ValueError, match=name):
            fn(_chain(), args["spot"], args["lot_size"], args["T"])


def test_one_second_to_expiry_is_still_allowed():  # A3-12 (model property, not an input error)
    assert np.isfinite(gex.net_gex_at(_chain(), SPOT, LOT, 1 / (365 * 24 * 3600)))


def test_participant_weights_outside_unit_interval_are_refused():  # A3-32
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        gex.net_gex_at(_chain(), SPOT, LOT, T, rule="participant", participant=gex.ParticipantWeights(2.0, -3.0))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        gex.daily_summary(_chain(), SPOT, LOT, T, participant=gex.ParticipantWeights(0.1, 1.5))
    # weights on the edges are fine
    assert np.isfinite(gex.net_gex_at(_chain(), SPOT, LOT, T, rule="participant",
                                      participant=gex.ParticipantWeights(1.0, -1.0)))


def test_all_open_interest_missing_gives_nan_not_zero():  # A3-05
    c = _chain()
    c["oi"] = np.nan
    with pytest.warns(GexInputWarning, match="missing oi"):
        net = gex.net_gex_at(c, SPOT, LOT, T)
    assert np.isnan(net)
    with pytest.warns(GexInputWarning):
        row = gex.daily_summary(c, SPOT, LOT, T)
    assert np.isnan(row["net_gex_us_convention"]) and np.isnan(row["gross_gex"])
    assert np.isnan(row["max_pain"]) and np.isnan(row["call_wall"])


def test_all_iv_missing_gives_nan_not_zero():  # A3-06
    c = _chain()
    c["iv"] = np.nan
    with pytest.warns(GexInputWarning, match="missing iv"):
        assert np.isnan(gex.net_gex_at(c, SPOT, LOT, T))


def test_iv_zero_is_treated_as_missing_with_a_warning():  # A3-13
    c = _chain()
    c["iv"] = 0.0
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        net = gex.net_gex_at(c, SPOT, LOT, T)
    assert np.isnan(net)
    msgs = [str(w.message) for w in rec]
    assert any("IV 0" in m for m in msgs)
    assert not any(issubclass(w.category, RuntimeWarning) for w in rec)  # no more divide-by-zero noise


@pytest.mark.parametrize("col,frac", [("oi", 0.2), ("iv", 0.25)])
def test_partly_missing_rows_warn_with_the_count(col, frac):  # A3-04, A3-07
    c = _chain()
    idx = c.sample(frac=frac, random_state=1).index
    c.loc[idx, col] = np.nan
    with pytest.warns(GexInputWarning, match=rf"{len(idx)} of {len(c)} rows have missing {col}"):
        net = gex.net_gex_at(c, SPOT, LOT, T)
    assert np.isfinite(net)  # the remaining rows are still summed, but no longer silently


def test_stale_tiny_iv_warns():  # A3-16
    c = _chain()
    c.loc[c.index[:5], "iv"] = 1e-4
    with pytest.warns(GexInputWarning, match="stale"):
        gex.strike_gex(c, SPOT, LOT, T)


def test_duplicated_rows_warn():  # A3-27
    c = _chain()
    c = pd.concat([c, c.iloc[20:40]], ignore_index=True)
    with pytest.warns(GexInputWarning, match="20 duplicated"):
        gex.strike_gex(c, SPOT, LOT, T)


def test_empty_chain_warns_and_daily_summary_is_nan():  # A3-08, A3-38, A3-39
    c = _chain().iloc[:0]
    with pytest.warns(GexInputWarning, match="empty"):
        assert np.isnan(gex.net_gex_at(c, SPOT, LOT, T))
    with pytest.warns(GexInputWarning):
        row = gex.daily_summary(c, SPOT, LOT, T)
    assert np.isnan(row["net_gex_us_convention"]) and np.isnan(row["gross_gex"])
    with pytest.warns(GexInputWarning):
        assert np.isnan(gex.max_pain(c))  # used to crash with "argmin of an empty sequence"
    with pytest.warns(GexInputWarning):
        assert np.isnan(gex.gamma_flip(c, SPOT, LOT, T))


def test_zero_open_interest_gives_nan_walls_and_max_pain_not_the_first_strike():  # A3-02, A3-37
    c = _chain()
    c["oi"] = 0.0
    walls = gex.oi_walls(c)
    assert all(np.isnan(v) for v in walls.values())
    assert np.isnan(gex.max_pain(c))
    assert gex.net_gex_at(c, SPOT, LOT, T) == 0.0  # zero open interest really is zero exposure


def test_walls_and_max_pain_unchanged_on_a_clean_chain():
    c = _chain()
    walls = gex.oi_walls(c)
    calls = c[c["is_call"]].groupby("strike")["oi"].sum()
    puts = c[~c["is_call"]].groupby("strike")["oi"].sum()
    assert walls["call_wall"] == calls.idxmax() and walls["put_wall"] == puts.idxmax()
    assert np.isfinite(gex.max_pain(c))


def test_nonpositive_strike_is_refused():
    c = _chain()
    c.loc[0, "strike"] = 0.0
    with pytest.raises(ValueError, match="strike"):
        gex.strike_gex(c, SPOT, LOT, T)


def test_validate_chain_returns_a_checked_copy_and_leaves_the_input_alone():
    c = _chain()
    c["is_call"] = c["is_call"].astype(int)
    c.loc[0, "iv"] = 0.0
    before = c.copy()
    with pytest.warns(GexInputWarning):
        clean = gex.validate_chain(c, SPOT, LOT, T)
    pd.testing.assert_frame_equal(c, before)
    assert clean["is_call"].dtype == bool and np.isnan(clean.loc[0, "iv"])


def test_total_helper():
    assert gex.total([1.0, np.nan, 2.0]) == 3.0
    assert np.isnan(gex.total([np.nan, np.nan]))
    assert gex.total([]) == 0.0
