"""Loaders for NSE F&O bhavcopy, participant-wise open interest and 1-minute bars.

These loaders were written against the published NSE file layouts. They have
been tested on small fixture files in the same layouts (see tests/fixtures),
not yet on live downloads, because the cloud environment used to build this
prototype cannot reach nseindia.com. Check them on a real file first, and in
particular confirm whether open interest is reported in contracts or in
units (shares); see ``oi_in_units``.

Two bhavcopy layouts exist:

* Legacy (to early July 2024): columns INSTRUMENT, SYMBOL, EXPIRY_DT,
  STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS,
  VAL_INLAKH, OPEN_INT, CHG_IN_OI, TIMESTAMP.
* UDiFF (from July 2024): columns TradDt, FinInstrmTp, TckrSymb, XpryDt,
  StrkPric, OptnTp, ClsPric, SttlmPric, UndrlygPric, OpnIntrst,
  ChngInOpnIntrst, TtlTradgVol, NewBrdLotQty and others.

Both are mapped to one tidy schema:

    date, symbol, expiry, strike, is_call, close, settle, oi, chg_oi,
    volume, underlying, lot_size
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OPTION_SCHEMA = ["date", "symbol", "expiry", "strike", "is_call", "close", "settle", "oi", "chg_oi",
                 "volume", "underlying", "lot_size"]


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]


def load_bhavcopy(path, symbol: str = "NIFTY") -> pd.DataFrame:
    """Read one F&O bhavcopy (legacy or UDiFF) and return Nifty index options."""
    df = _clean_cols(pd.read_csv(path))
    if "INSTRUMENT" in df.columns:
        df = df[(df["INSTRUMENT"].str.strip() == "OPTIDX") & (df["SYMBOL"].str.strip() == symbol)]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["TIMESTAMP"], format="mixed", dayfirst=True),
                "symbol": df["SYMBOL"].str.strip(),
                "expiry": pd.to_datetime(df["EXPIRY_DT"], format="mixed", dayfirst=True),
                "strike": df["STRIKE_PR"].astype(float),
                "is_call": df["OPTION_TYP"].str.strip().eq("CE"),
                "close": df["CLOSE"].astype(float),
                "settle": df["SETTLE_PR"].astype(float),
                "oi": df["OPEN_INT"].astype(float),
                "chg_oi": df["CHG_IN_OI"].astype(float),
                "volume": df["CONTRACTS"].astype(float),
                "underlying": np.nan,
                "lot_size": np.nan,
            }
        )
    elif "FinInstrmTp" in df.columns:
        df = df[(df["FinInstrmTp"].str.strip() == "IDO") & (df["TckrSymb"].str.strip() == symbol)]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["TradDt"]),
                "symbol": df["TckrSymb"].str.strip(),
                "expiry": pd.to_datetime(df["XpryDt"]),
                "strike": df["StrkPric"].astype(float),
                "is_call": df["OptnTp"].str.strip().eq("CE"),
                "close": df["ClsPric"].astype(float),
                "settle": df["SttlmPric"].astype(float),
                "oi": df["OpnIntrst"].astype(float),
                "chg_oi": df["ChngInOpnIntrst"].astype(float),
                "volume": df["TtlTradgVol"].astype(float),
                "underlying": df["UndrlygPric"].astype(float) if "UndrlygPric" in df else np.nan,
                "lot_size": df["NewBrdLotQty"].astype(float) if "NewBrdLotQty" in df else np.nan,
            }
        )
    else:
        raise ValueError(f"{path}: unrecognised bhavcopy layout")
    return out[OPTION_SCHEMA].reset_index(drop=True)


def oi_in_units(chain: pd.DataFrame, lot_size: float) -> pd.DataFrame:
    """Convert OI from units (shares) to contracts. Call only if the file reports units."""
    out = chain.copy()
    out["oi"] = out["oi"] / lot_size
    out["chg_oi"] = out["chg_oi"] / lot_size
    return out


_PARTICIPANT_MAP = {
    "Client Type": "client_type",
    "Future Index Long": "future_index_long",
    "Future Index Short": "future_index_short",
    "Future Stock Long": "future_stock_long",
    "Future Stock Short": "future_stock_short",
    "Option Index Call Long": "option_index_call_long",
    "Option Index Put Long": "option_index_put_long",
    "Option Index Call Short": "option_index_call_short",
    "Option Index Put Short": "option_index_put_short",
    "Option Stock Call Long": "option_stock_call_long",
    "Option Stock Put Long": "option_stock_put_long",
    "Option Stock Call Short": "option_stock_call_short",
    "Option Stock Put Short": "option_stock_put_short",
    "Total Long Contracts": "total_long",
    "Total Short Contracts": "total_short",
}


def load_participant_oi(path) -> pd.DataFrame:
    """Read an NSE participant-wise OI CSV (fao_participant_oi_DDMMYYYY.csv).

    The file starts with a one-line title, then a header row. Returns a table
    indexed by client type (Client, DII, FII, Pro, TOTAL).
    """
    raw = pd.read_csv(path, header=None, dtype=str)
    hdr_row = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Client Type")][0]
    df = _clean_cols(pd.read_csv(path, skiprows=hdr_row))
    df = df.rename(columns={k: v for k, v in _PARTICIPANT_MAP.items() if k in df.columns})
    df["client_type"] = df["client_type"].astype(str).str.strip()
    df = df.set_index("client_type")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    return df


def load_minute_bars(path, tz_naive: bool = True) -> pd.DataFrame:
    """Read 1-minute OHLCV bars with a timestamp column (vendor-neutral).

    Accepts columns named like timestamp/datetime/date, open, high, low,
    close and optional volume (case-insensitive).
    """
    df = _clean_cols(pd.read_csv(path))
    df.columns = [c.lower() for c in df.columns]
    tcol = next(c for c in ("timestamp", "datetime", "date", "time") if c in df.columns)
    ts = pd.to_datetime(df[tcol])
    if tz_naive and getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    out = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]].astype(float)
    out.index = ts
    return out.sort_index()


def to_session_grid(bars: pd.DataFrame, start: str = "09:15", minutes: int = 375) -> dict:
    """Align bars onto a (days x minutes) grid for the 09:15 to 15:30 session.

    Returns dict with arrays open, high, low, close, volume (days x minutes),
    the list of days, and the 1-minute log returns (first minute uses the
    bar's own open, so overnight gaps are excluded).
    """
    bars = bars.copy()
    bars["day"] = bars.index.normalize()
    t0 = pd.to_timedelta(start + ":00")
    bars["slot"] = ((bars.index - bars["day"] - t0) / pd.Timedelta(minutes=1)).astype(int)
    bars = bars[(bars["slot"] >= 0) & (bars["slot"] < minutes)]
    days = pd.DatetimeIndex(sorted(bars["day"].unique()))
    pos = {d: i for i, d in enumerate(days)}
    grid = {}
    for col in ("open", "high", "low", "close", "volume"):
        arr = np.full((len(days), minutes), np.nan)
        if col in bars:
            arr[bars["day"].map(pos).to_numpy(), bars["slot"].to_numpy()] = bars[col].to_numpy()
        grid[col] = arr
    c = grid["close"]
    prev = np.concatenate([grid["open"][:, :1], c[:, :-1]], axis=1)
    grid["returns"] = np.log(c / prev)
    grid["days"] = days
    return grid
