"""Gamma exposure (GEX) by strike and in aggregate, under different dealer-signing rules.

Definitions
-----------
For one option series k with spot gamma ``g_k`` (per 1 point of the index),
open interest ``oi_k`` in contracts and lot size ``L``, the rupee change in
the holder's delta notional for a 1% move in the index S is

    gex_k = g_k * oi_k * L * S**2 * 0.01

(derivation: delta changes by g_k * dS per unit, dS = 0.01 S, and the delta
is worth S rupees per unit). Net dealer GEX is

    GEX = sum_k  w_k * gex_k

where w_k in [-1, 1] is the dealers' net share of the open interest in that
series (+1 means dealers hold all of it long). Public data never shows w_k
directly, which is the "sign problem". This module keeps several rules side
by side so every result can be reported under each of them.

Sign rules
----------
``us_convention``  w = +1 for calls, -1 for puts. The usual assumption in US
                   GEX work (customers sell calls and buy puts). It may be
                   wrong for India, where retail clients are large net buyers
                   of both calls and puts.
``participant``    w for calls and puts taken from NSE participant-wise open
                   interest for a chosen group (default "Pro"). That file
                   aggregates all index options, not only Nifty, and is not
                   by strike, so the same weight is used at every strike.
``client_mirror``  dealers are assumed to take the opposite side of the
                   "Client" category.
``long_all`` / ``short_all``  upper and lower bounds (+1 or -1 everywhere).

Units: GEX is in rupees per 1% move. Divide by 1e7 for crore.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import greeks

SIGN_RULES = ("us_convention", "participant", "client_mirror", "long_all", "short_all")


@dataclass
class ParticipantWeights:
    """Dealers' net share of index call and put open interest, each in [-1, 1]."""

    call: float
    put: float

    @classmethod
    def from_participant_oi(cls, table: pd.DataFrame, group: str = "Pro", mirror: bool = False):
        """Build weights from a participant-wise OI table (see data.load_participant_oi).

        weight = (group long - group short) / total open contracts, where the
        total is the market-wide long side (which equals the short side).
        That is the group's net position as a share of all open contracts.
        With ``mirror=True`` the sign of the group's position is flipped
        (dealers take the opposite side of that group).
        """
        row = table.loc[group]
        tot = table.loc["TOTAL"] if "TOTAL" in table.index else table.drop(index=[group]).sum() + row
        call_oi = float(tot["option_index_call_long"])
        put_oi = float(tot["option_index_put_long"])
        wc = (row["option_index_call_long"] - row["option_index_call_short"]) / call_oi if call_oi else 0.0
        wp = (row["option_index_put_long"] - row["option_index_put_short"]) / put_oi if put_oi else 0.0
        s = -1.0 if mirror else 1.0
        return cls(call=float(np.clip(s * wc, -1, 1)), put=float(np.clip(s * wp, -1, 1)))


def sign_weights(is_call, rule: str, participant: ParticipantWeights | None = None) -> np.ndarray:
    is_call = np.asarray(is_call, bool)
    if rule == "us_convention":
        return np.where(is_call, 1.0, -1.0)
    if rule == "long_all":
        return np.ones(is_call.shape)
    if rule == "short_all":
        return -np.ones(is_call.shape)
    if rule in ("participant", "client_mirror"):
        if participant is None:
            raise ValueError(f"rule {rule!r} needs ParticipantWeights")
        return np.where(is_call, participant.call, participant.put)
    raise ValueError(f"unknown sign rule {rule!r}; choose from {SIGN_RULES}")


def strike_gex(
    chain: pd.DataFrame,
    spot: float,
    lot_size: float,
    T: float,
    r: float = 0.065,
    carry: float = 0.0,
    rule: str = "us_convention",
    participant: ParticipantWeights | None = None,
    forward: float | None = None,
) -> pd.DataFrame:
    """GEX per option series.

    ``chain`` needs columns: strike, is_call, oi (contracts), iv (annualised).
    Returns the chain with added columns gamma, gross_gex, weight, gex.
    """
    F = spot * np.exp(carry * T) if forward is None else forward
    out = chain.copy()
    g = greeks.spot_gamma(F, out["strike"].to_numpy(float), T, r, out["iv"].to_numpy(float), carry)
    out["gamma"] = g
    out["gross_gex"] = g * out["oi"].to_numpy(float) * lot_size * spot**2 * 0.01
    out["weight"] = sign_weights(out["is_call"].to_numpy(bool), rule, participant)
    out["gex"] = out["weight"] * out["gross_gex"]
    return out


def gex_by_strike(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a strike_gex table to one row per strike (calls and puts summed)."""
    agg = table.groupby("strike").agg(gex=("gex", "sum"), gross_gex=("gross_gex", "sum"), oi=("oi", "sum"))
    calls = table[table["is_call"]].groupby("strike")["oi"].sum().rename("call_oi")
    puts = table[~table["is_call"]].groupby("strike")["oi"].sum().rename("put_oi")
    return agg.join(calls).join(puts).fillna(0.0)


def net_gex_at(
    chain: pd.DataFrame,
    spot: float,
    lot_size: float,
    T: float,
    r: float = 0.065,
    carry: float = 0.0,
    rule: str = "us_convention",
    participant: ParticipantWeights | None = None,
) -> float:
    """Net GEX if the index were at ``spot``, holding IVs and OI fixed."""
    return float(strike_gex(chain, spot, lot_size, T, r, carry, rule, participant)["gex"].sum())


def gamma_flip(
    chain: pd.DataFrame,
    spot: float,
    lot_size: float,
    T: float,
    r: float = 0.065,
    carry: float = 0.0,
    rule: str = "us_convention",
    participant: ParticipantWeights | None = None,
    width: float = 0.05,
    n_grid: int = 201,
) -> float:
    """Index level closest to ``spot`` where net GEX changes sign.

    Searches a grid of +/- ``width`` around spot and refines the root with
    Brent's method. Returns NaN when net GEX keeps one sign over the grid,
    which is common under the long_all and short_all bounds.
    """
    grid = spot * np.linspace(1 - width, 1 + width, n_grid)
    vals = np.array([net_gex_at(chain, s, lot_size, T, r, carry, rule, participant) for s in grid])
    sign_change = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if sign_change.size == 0:
        return float("nan")
    roots = []
    f = lambda s: net_gex_at(chain, s, lot_size, T, r, carry, rule, participant)  # noqa: E731
    for j in sign_change:
        roots.append(brentq(f, grid[j], grid[j + 1], xtol=1e-6 * spot))
    roots = np.array(roots)
    return float(roots[np.argmin(np.abs(roots - spot))])


def oi_walls(chain: pd.DataFrame) -> dict:
    """Strikes with the largest call OI, put OI and total OI."""
    calls = chain[chain["is_call"]].groupby("strike")["oi"].sum()
    puts = chain[~chain["is_call"]].groupby("strike")["oi"].sum()
    total = calls.add(puts, fill_value=0.0)
    return {
        "call_wall": float(calls.idxmax()) if len(calls) else np.nan,
        "put_wall": float(puts.idxmax()) if len(puts) else np.nan,
        "max_oi_strike": float(total.idxmax()) if len(total) else np.nan,
    }


def max_pain(chain: pd.DataFrame) -> float:
    """Strike at which the total intrinsic value paid to option holders is smallest."""
    strikes = np.sort(chain["strike"].unique())
    K = chain["strike"].to_numpy(float)
    oi = chain["oi"].to_numpy(float)
    is_call = chain["is_call"].to_numpy(bool)
    payout = []
    for s in strikes:
        intrinsic = np.where(is_call, np.maximum(s - K, 0), np.maximum(K - s, 0))
        payout.append(np.sum(intrinsic * oi))
    return float(strikes[int(np.argmin(payout))])


def daily_summary(
    chain: pd.DataFrame,
    spot: float,
    lot_size: float,
    T: float,
    r: float = 0.065,
    carry: float = 0.0,
    participant: ParticipantWeights | None = None,
    client_weights: ParticipantWeights | None = None,
    with_flip: bool = True,
) -> dict:
    """One row of the daily GEX panel: net GEX under every available rule plus levels."""
    row: dict = {"spot": spot, "T": T}
    gross = strike_gex(chain, spot, lot_size, T, r, carry, "long_all")
    row["gross_gex"] = float(gross["gross_gex"].sum())
    near = gross[np.abs(gross["strike"] / spot - 1) <= 0.01]
    row["gross_gex_atm1pct"] = float(near["gross_gex"].sum())
    rules = [("us_convention", None)]
    if participant is not None:
        rules.append(("participant", participant))
    if client_weights is not None:
        rules.append(("client_mirror", client_weights))
    for rule, pw in rules:
        t = strike_gex(chain, spot, lot_size, T, r, carry, rule, pw)
        row[f"net_gex_{rule}"] = float(t["gex"].sum())
        if with_flip:
            row[f"flip_{rule}"] = gamma_flip(chain, spot, lot_size, T, r, carry, rule, pw)
    row.update(oi_walls(chain))
    row["max_pain"] = max_pain(chain)
    return row
