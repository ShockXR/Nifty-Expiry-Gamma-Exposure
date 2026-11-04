"""Input checks of the GEX engine: the gaps left by PR #4, written by an AI agent (Agent B).

All chains here are SYNTHETIC, built like reports/stress/a3_gex_stress.py (same strikes, IV smile and
seed 9_800_000). PR #4 (tests/test_gex_input_checks.py) already covers most of the 41 stress cases.
This file keeps only what that file does not: the stress cases PR #4 still accepted silently and the
small fixes added on top of it (marked "Agent B" in src/niftygex/gex.py), plus a case-by-case check
that every one of the 24 originally silent stress cases now gives a signal (error, NaN or warning),
except the two unit mistakes the engine cannot see.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from niftygex import gex as G

S, LOT, T0, R = 24000.0, 75.0, 3 / 365, 0.065


def chain(seed=9_800_000, step=50, lo=0.9, hi=1.1):
    rng = np.random.default_rng(seed)
    K = np.arange(round(S * lo / step) * step, S * hi + 1, step)
    rows = []
    for k in K:
        iv = 0.14 + 0.5 * (k / S - 1) ** 2 + 0.02 * (k < S)
        for c in (True, False):
            oi = float(rng.integers(100, 20000)) * (1 + 3 * np.exp(-((k - S) / 400) ** 2))
            rows.append(dict(strike=float(k), is_call=c, oi=oi, iv=iv))
    return pd.DataFrame(rows)


C = chain()


def summary(c):
    return G.daily_summary(c, S, LOT, T0, R, with_flip=False)


def _with_nan(col, every):
    c = C.copy()
    c.loc[c.index[::every], col] = np.nan
    return c


# ---------- gaps in PR #4, fixed on top of it ----------

def test_agentb_iv_above_cap_is_refused():
    with pytest.raises(ValueError, match="above 5.0"):
        G.net_gex_at(C.assign(iv=6.0), S, LOT, T0, R)
    with pytest.raises(ValueError, match="above 5.0"):
        G.net_gex_at(C.assign(iv=np.inf), S, LOT, T0, R)
    assert np.isfinite(G.net_gex_at(C.assign(iv=G.IV_MAX), S, LOT, T0, R))  # A3-15: 500 percent still allowed


def test_agentb_infinite_open_interest_is_refused():
    with pytest.raises(ValueError, match="infinite open interest"):
        G.net_gex_at(C.assign(oi=np.inf), S, LOT, T0, R)


def test_agentb_atm_gross_gex_is_nan_not_zero_when_all_oi_missing():  # A3-05, the near-the-money column
    with pytest.warns(G.GexInputWarning):
        row = summary(C.assign(oi=np.nan))
    assert np.isnan(row["gross_gex_atm1pct"])


def test_agentb_calls_only_chain_warns():  # A3-26
    with pytest.warns(G.GexInputWarning, match="only calls"):
        summary(C[C.is_call])
    with pytest.warns(G.GexInputWarning, match="only puts"):
        summary(C[~C.is_call])


def test_agentb_no_strike_near_spot_warns():  # A3-25
    with pytest.warns(G.GexInputWarning, match="within 2%"):
        summary(C[(C.strike < S * 0.98) | (C.strike > S * 1.02)])


def test_agentb_clean_chain_still_raises_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        row = G.daily_summary(C, S, LOT, T0, R)
    assert row["gross_gex_atm1pct"] == float(
        G.strike_gex(C, S, LOT, T0, R, rule="long_all").query("abs(strike / @S - 1) <= 0.01")["gross_gex"].sum())


# ---------- the 24 originally silent stress cases, one by one ----------

def _raises(f):
    try:
        f()
    except ValueError:
        return True
    return False


def _signal(f):
    """'error', 'nan' or 'warning' when the engine gives a visible signal, else 'silent'."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            v = f()
        except ValueError:
            return "error"
    if isinstance(v, dict):
        v = [v[k] for k in ("gross_gex", "net_gex_us_convention", "call_wall", "put_wall", "max_pain")]
    if np.isnan(np.asarray(v, float)).any():
        return "nan"
    return "warning" if any(issubclass(x.category, G.GexInputWarning) for x in w) else "silent"


SILENT_BEFORE = {
    "A3-02 all OI zero (walls, max pain)": lambda: summary(C.assign(oi=0.0)),
    "A3-03 negative OI": lambda: summary(C.assign(oi=np.where(C.index % 7 == 0, -C.oi, C.oi))),
    "A3-04 20 percent NaN OI": lambda: summary(_with_nan("oi", 5)),
    "A3-05 all OI NaN": lambda: summary(C.assign(oi=np.nan)),
    "A3-06 all IV NaN": lambda: summary(C.assign(iv=np.nan)),
    "A3-07 25 percent NaN IV": lambda: summary(_with_nan("iv", 4)),
    "A3-08 empty chain": lambda: summary(C.iloc[0:0]),
    "A3-09 T = 0": lambda: G.daily_summary(C, S, LOT, 0.0, R),
    "A3-10 T < 0": lambda: G.daily_summary(C, S, LOT, -1 / 365, R),
    "A3-13 IV = 0": lambda: summary(C.assign(iv=0.0)),
    "A3-14 IV negative": lambda: summary(C.assign(iv=-0.2)),
    "A3-16 IV 0.0001": lambda: summary(C.assign(iv=1e-4)),
    "A3-17 spot 0": lambda: G.daily_summary(C, 0.0, LOT, T0, R),
    "A3-18 spot negative": lambda: G.daily_summary(C, -S, LOT, T0, R),
    "A3-19 lot size 0": lambda: G.daily_summary(C, S, 0.0, T0, R),
    "A3-20 lot size negative": lambda: G.daily_summary(C, S, -75.0, T0, R),
    "A3-25 no strike within 2 percent": lambda: summary(C[(C.strike < S * 0.98) | (C.strike > S * 1.02)]),
    "A3-26 calls only": lambda: summary(C[C.is_call]),
    "A3-27 duplicated rows": lambda: summary(pd.concat([C, C.iloc[90:110]])),
    "A3-28 option type CE/PE": lambda: summary(C.assign(is_call=np.where(C.is_call, "CE", "PE"))),
    "A3-32 weights outside [-1, 1]": lambda: G.daily_summary(C, S, LOT, T0, R, participant=G.ParticipantWeights(2.0, -3.0)),
    "A3-37 daily_summary on zero-OI chain": lambda: G.daily_summary(C.assign(oi=0.0), S, LOT, T0, R),
}


@pytest.mark.parametrize("name", list(SILENT_BEFORE))
def test_originally_silent_case_now_gives_a_signal(name):
    assert _signal(SILENT_BEFORE[name]) != "silent"


@pytest.mark.parametrize("name,f", [
    ("A3-21 OI in shares (x75)", lambda: summary(C.assign(oi=C.oi * 75))),
    ("A3-22 lot size 1", lambda: G.daily_summary(C, S, 1.0, T0, R, with_flip=False)),
])
def test_unit_mistakes_remain_invisible_to_the_engine(name, f):
    # documented limitation (risk R-A5): a unit error gives a valid looking chain; the pipeline's
    # scripts/check_oi_units.py is the guard. This test pins the limitation so a future fix shows up.
    assert _signal(f) == "silent"
