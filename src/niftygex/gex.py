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

Input checks
------------
Every public function checks its inputs (``validate_chain``) so that bad
data cannot pass as a quiet, normal looking number. It raises ``ValueError``
for input that has no sensible meaning: a missing column, option types given
as strings such as "CE"/"PE", negative open interest, negative implied
volatility, a spot, lot size or time to expiry that is not a positive
finite number, and participant weights outside [-1, 1]. It warns
(``GexInputWarning``) for input that can be real but changes the answer:
missing (NaN) open interest or IV, IV of exactly 0 (treated as missing, as
on the NSE option chain page for untraded strikes), IV below 1 percent (a
stale quote), duplicated series and an empty chain. A total over rows that
are all missing is NaN, not 0, so an empty download never looks like a
neutral day. The engine cannot check units: open interest must be in
contracts and ``lot_size`` in shares per contract (open interest in shares
with lot size 75 inflates GEX 75 times).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import greeks

SIGN_RULES = ("us_convention", "participant", "client_mirror", "long_all", "short_all")
REQUIRED_COLUMNS = ("strike", "is_call", "oi", "iv")
STALE_IV = 0.01  # an annualised IV below 1 percent is not a real Nifty quote
# Added in the 4 November 2026 review of PR #4: upper bound for an annualised IV (5.0 = 500 percent). It equals the top of
# the bracket of greeks.implied_vol, so every IV that solver can return still passes.
IV_MAX = 5.0


class GexInputWarning(UserWarning):
    """Input that the engine can use but that changes the answer (see "Input checks")."""


def _check_positive(name: str, value) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number, got {value!r}") from None
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


def _option_type_as_bool(col: pd.Series) -> np.ndarray:
    """``is_call`` as booleans. Accepts True/False and 1/0; refuses strings such as "CE"/"PE"."""
    if pd.api.types.is_bool_dtype(col):
        return col.to_numpy(bool)
    if pd.api.types.is_numeric_dtype(col):
        v = col.to_numpy(float)
        if np.isin(v, (0.0, 1.0)).all():
            return v.astype(bool)
        raise ValueError("is_call must be True/False or 1/0; found other numbers")
    vals = col.to_numpy(object)
    if all(isinstance(x, (bool, np.bool_)) for x in vals):
        return vals.astype(bool)
    example = next((x for x in vals if not isinstance(x, (bool, np.bool_))), None)
    raise ValueError(
        f"is_call must be True/False or 1/0, got values like {example!r}. Every non-empty string counts as "
        "True, so all options would be treated as calls. Convert first, for example "
        "chain['is_call'] = chain['option_type'].eq('CE')"
    )


def _check_weights(participant) -> None:
    if participant is None:
        return
    for side in ("call", "put"):
        w = float(getattr(participant, side))
        if math.isfinite(w) and not -1.0 <= w <= 1.0:
            raise ValueError(f"participant weight for {side}s is {w}; a dealer share must lie in [-1, 1]")


def validate_chain(
    chain: pd.DataFrame,
    spot: float | None = None,
    lot_size: float | None = None,
    T: float | None = None,
    participant: "ParticipantWeights | None" = None,
) -> pd.DataFrame:
    """Check a chain and its scalars; return a clean copy (see "Input checks" in the module docstring).

    The copy has ``is_call`` as booleans, numeric strike, oi and iv, and IV
    of exactly 0 replaced by NaN. Raises ``ValueError`` or warns with
    ``GexInputWarning``.
    """
    if not isinstance(chain, pd.DataFrame):
        raise ValueError("chain must be a pandas DataFrame")
    missing = [c for c in REQUIRED_COLUMNS if c not in chain.columns]
    if missing:
        raise ValueError(f"chain is missing column(s) {missing}; it needs {list(REQUIRED_COLUMNS)} "
                         f"(lower case), got {list(chain.columns)}")
    for name, value in (("spot", spot), ("lot_size", lot_size), ("T (time to expiry in years)", T)):
        if value is not None:
            _check_positive(name, value)
    _check_weights(participant)
    out = chain.copy()
    out["is_call"] = _option_type_as_bool(out["is_call"])
    for c in ("strike", "oi", "iv"):
        try:
            out[c] = pd.to_numeric(out[c]).astype(float)
        except (TypeError, ValueError):
            raise ValueError(f"column {c!r} must be numeric") from None
    if len(out) == 0:
        warnings.warn("empty option chain: there is no open interest to measure", GexInputWarning, stacklevel=3)
        return out
    oi, iv, k = out["oi"].to_numpy(), out["iv"].to_numpy(), out["strike"].to_numpy()
    if (oi < 0).any():
        raise ValueError(f"{int((oi < 0).sum())} row(s) have negative open interest")
    if (iv < 0).any():
        raise ValueError(f"{int((iv < 0).sum())} row(s) have negative implied volatility")
    # Added in the 4 November 2026 review of PR #4: infinite OI, and IV above IV_MAX (or infinite), used to give a silent NaN or a huge number
    if np.isinf(oi).any():
        raise ValueError(f"{int(np.isinf(oi).sum())} row(s) have infinite open interest")
    if (iv > IV_MAX).any():
        raise ValueError(f"{int((iv > IV_MAX).sum())} row(s) have implied volatility above {IV_MAX} "
                         f"({IV_MAX:.0%} a year), the upper limit of the IV solver")
    if not (np.isfinite(k) & (k > 0)).all():
        raise ValueError("every strike must be a positive finite number")
    zero_iv = iv == 0
    if zero_iv.any():
        out.loc[zero_iv, "iv"] = np.nan
        warnings.warn(f"{int(zero_iv.sum())} of {len(out)} rows have IV 0, treated as missing", GexInputWarning,
                      stacklevel=3)
    stale = (iv > 0) & (iv < STALE_IV)
    if stale.any():
        warnings.warn(f"{int(stale.sum())} of {len(out)} rows have IV below {STALE_IV:.0%} (stale quote?); "
                      "their gamma is huge and dominates the total", GexInputWarning, stacklevel=3)
    for c in ("oi", "iv"):
        n_nan = int(out[c].isna().sum())
        if n_nan and c == "iv":
            n_nan -= int(zero_iv.sum())
        if n_nan:
            warnings.warn(f"{n_nan} of {len(out)} rows have missing {c}; totals leave them out "
                          "(NaN if every row is missing)", GexInputWarning, stacklevel=3)
    n_dup = int(out.duplicated(["strike", "is_call"]).sum())
    if n_dup:
        warnings.warn(f"{n_dup} duplicated (strike, option type) rows: their open interest is counted twice",
                      GexInputWarning, stacklevel=3)
    return out


def total(values) -> float:
    """Sum that skips NaN but is NaN when every value is missing (so an all-NaN chain is not 0)."""
    v = np.asarray(values, float)
    if v.size and np.isnan(v).all():
        return float("nan")
    return float(np.nansum(v))


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
    Inputs are checked with ``validate_chain`` (raises or warns); the
    returned table keeps the caller's columns and dtypes.
    """
    clean = validate_chain(chain, spot, lot_size, T, participant)
    return _gex_table(chain, clean, spot, lot_size, T, r, carry, rule, participant, forward)


def _gex_table(chain, clean, spot, lot_size, T, r, carry, rule, participant, forward=None) -> pd.DataFrame:
    """strike_gex on a chain already checked by validate_chain (``clean`` is its checked copy)."""
    if forward is not None:
        _check_positive("forward", forward)
    F = spot * np.exp(carry * T) if forward is None else forward
    out = chain.copy()
    g = greeks.spot_gamma(F, clean["strike"].to_numpy(float), T, r, clean["iv"].to_numpy(float), carry)
    out["gamma"] = g
    out["gross_gex"] = g * clean["oi"].to_numpy(float) * lot_size * spot**2 * 0.01
    out["weight"] = sign_weights(clean["is_call"].to_numpy(bool), rule, participant)
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
    """Net GEX if the index were at ``spot``, holding IVs and OI fixed (NaN if no row has data)."""
    t = strike_gex(chain, spot, lot_size, T, r, carry, rule, participant)
    return total(t["gex"]) if len(t) else float("nan")


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
    which is common under the long_all and short_all bounds, and also when
    any row has missing open interest or IV (``validate_chain`` warns then).
    """
    chain = validate_chain(chain, spot, lot_size, T, participant)
    if len(chain) == 0:
        return float("nan")
    K = chain["strike"].to_numpy(float)
    iv = chain["iv"].to_numpy(float)
    oi = chain["oi"].to_numpy(float)
    w = sign_weights(chain["is_call"].to_numpy(bool), rule, participant)

    def f(s):
        s = np.atleast_1d(np.asarray(s, float))[:, None]
        g = greeks.spot_gamma(s * np.exp(carry * T), K[None, :], T, r, iv[None, :], carry)
        return (g * (w * oi)[None, :]).sum(axis=1) * lot_size * s[:, 0] ** 2 * 0.01

    grid = spot * np.linspace(1 - width, 1 + width, n_grid)
    vals = f(grid)
    sign_change = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if sign_change.size == 0:
        return float("nan")
    roots = np.array([brentq(lambda x: f(x)[0], grid[j], grid[j + 1], xtol=1e-6 * spot) for j in sign_change])
    return float(roots[np.argmin(np.abs(roots - spot))])


def oi_walls(chain: pd.DataFrame) -> dict:
    """Strikes with the largest call OI, put OI and total OI (NaN when that side has no open interest)."""
    chain = validate_chain(chain)
    calls = chain[chain["is_call"]].groupby("strike")["oi"].sum()
    puts = chain[~chain["is_call"]].groupby("strike")["oi"].sum()
    both = calls.add(puts, fill_value=0.0)

    def wall(oi: pd.Series) -> float:
        # with no positive open interest every strike ties and idxmax would just return the first one
        return float(oi.idxmax()) if len(oi) and oi.max() > 0 else np.nan

    return {"call_wall": wall(calls), "put_wall": wall(puts), "max_oi_strike": wall(both)}


def max_pain(chain: pd.DataFrame) -> float:
    """Strike at which the total intrinsic value paid to option holders is smallest.

    NaN for an empty chain or one with no positive open interest (every strike would tie).
    Rows with missing open interest are left out.
    """
    chain = validate_chain(chain)
    chain = chain[chain["oi"].notna()]
    if len(chain) == 0 or not chain["oi"].max() > 0:
        return float("nan")
    strikes = np.sort(chain["strike"].unique())
    K = chain["strike"].to_numpy(float)
    oi = chain["oi"].to_numpy(float)
    is_call = chain["is_call"].to_numpy(bool)
    payout = []
    for s in strikes:
        intrinsic = np.where(is_call, np.maximum(s - K, 0), np.maximum(K - s, 0))
        payout.append(np.sum(intrinsic * oi))
    return float(strikes[int(np.argmin(payout))])


def _check_coverage(clean: pd.DataFrame, spot: float) -> None:
    """Added in the 4 November 2026 review of PR #4: warn about chains that are valid but cannot give a representative GEX."""
    if len(clean) == 0:
        return
    n_calls = int(clean["is_call"].sum())
    if n_calls in (0, len(clean)):
        warnings.warn(f"chain holds only {'puts' if n_calls == 0 else 'calls'}: net GEX ignores the other side",
                      GexInputWarning, stacklevel=3)
    if not (np.abs(clean["strike"].to_numpy() / spot - 1) <= 0.02).any():
        warnings.warn("no strike within 2% of spot: the near-the-money gamma that dominates GEX is missing",
                      GexInputWarning, stacklevel=3)


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
    """One row of the daily GEX panel: net GEX under every available rule plus levels.

    Totals are NaN (not 0) for an empty chain or one where every row is missing.
    """
    row: dict = {"spot": spot, "T": T}
    clean = validate_chain(chain, spot, lot_size, T)
    _check_coverage(clean, spot)  # Added in the 4 November 2026 review of PR #4
    _check_weights(participant)
    _check_weights(client_weights)
    empty = len(clean) == 0
    gross = _gex_table(chain, clean, spot, lot_size, T, r, carry, "long_all", None)
    row["gross_gex"] = float("nan") if empty else total(gross["gross_gex"])
    near = gross[np.abs(gross["strike"] / spot - 1) <= 0.01]
    # Fix from the 4 November 2026 review of PR #4: total() (not .sum()) so an all-missing chain gives NaN here too, not 0.0
    row["gross_gex_atm1pct"] = float("nan") if empty else total(near["gross_gex"])
    rules = [("us_convention", None)]
    if participant is not None:
        rules.append(("participant", participant))
    if client_weights is not None:
        rules.append(("client_mirror", client_weights))
    for rule, pw in rules:
        t = _gex_table(chain, clean, spot, lot_size, T, r, carry, rule, pw)
        row[f"net_gex_{rule}"] = float("nan") if empty else total(t["gex"])
        if with_flip:
            row[f"flip_{rule}"] = gamma_flip(chain, spot, lot_size, T, r, carry, rule, pw)
    row.update(oi_walls(chain))
    row["max_pain"] = max_pain(chain)
    return row
