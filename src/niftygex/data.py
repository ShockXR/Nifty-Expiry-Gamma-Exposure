"""Loaders for NSE F&O bhavcopy, participant-wise open interest and 1-minute bars.

These loaders were first written against the published NSE file layouts and
tested on synthetic files. On 2 November 2026 they were run on real files
from the NSE archive server (January 2023 to October 2026). Checked facts:

* Open interest (OPEN_INT, OpnIntrst) is reported in units (shares), in
  both layouts. On 30 Oct 2025 the sum over all index options of OI divided
  by each row's lot size equals the "Option Index Call Long" TOTAL of the
  participant-wise OI file exactly (4,230,409 contracts); index futures
  match exactly on 10 Jan 2024 too.
* Lot sizes can differ between expiries on the same day (for example 75 and
  65 for NIFTY in late 2025), so any conversion must be per row. The
  simplest exact choice is to keep OI in shares and use a lot size of 1
  (``oi_unit="keep_units"``); GEX is then in rupees per 1% move directly.

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


ALL_SCHEMA = OPTION_SCHEMA + ["instrument", "value"]


def load_bhavcopy_all(path, symbols=("NIFTY", "BANKNIFTY")) -> pd.DataFrame:
    """Read one F&O bhavcopy (legacy or UDiFF): index options AND index futures.

    Returns the tidy schema plus ``instrument`` ("OPT" or "FUT") and
    ``value`` (traded value in rupees). Checked on real files around the
    July 2024 layout switch: futures traded value and contract counts are on
    the same scale in both layouts (legacy VAL_INLAKH is in lakh rupees,
    UDiFF TtlTrfVal in rupees; CONTRACTS and TtlTradgVol are both contracts).
    """
    df = _clean_cols(pd.read_csv(path))
    syms = set(symbols)
    if "INSTRUMENT" in df.columns:
        inst = df["INSTRUMENT"].astype(str).str.strip()
        sym = df["SYMBOL"].astype(str).str.strip()
        df = df[inst.isin(["OPTIDX", "FUTIDX"]) & sym.isin(syms)]
        inst = df["INSTRUMENT"].astype(str).str.strip()
        opt_typ = df["OPTION_TYP"].astype(str).str.strip()
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["TIMESTAMP"], format="mixed", dayfirst=True),
                "symbol": df["SYMBOL"].astype(str).str.strip(),
                "expiry": pd.to_datetime(df["EXPIRY_DT"], format="mixed", dayfirst=True),
                "strike": df["STRIKE_PR"].astype(float),
                "is_call": opt_typ.eq("CE"),
                "close": df["CLOSE"].astype(float),
                "settle": df["SETTLE_PR"].astype(float),
                "oi": df["OPEN_INT"].astype(float),
                "chg_oi": df["CHG_IN_OI"].astype(float),
                "volume": df["CONTRACTS"].astype(float),
                "underlying": np.nan,
                "lot_size": np.nan,
                "instrument": np.where(inst.eq("OPTIDX"), "OPT", "FUT"),
                "value": df["VAL_INLAKH"].astype(float) * 1e5 if "VAL_INLAKH" in df else np.nan,
            }
        )
    elif "FinInstrmTp" in df.columns:
        inst = df["FinInstrmTp"].astype(str).str.strip()
        sym = df["TckrSymb"].astype(str).str.strip()
        df = df[inst.isin(["IDO", "IDF"]) & sym.isin(syms)]
        inst = df["FinInstrmTp"].astype(str).str.strip()
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["TradDt"]),
                "symbol": df["TckrSymb"].astype(str).str.strip(),
                "expiry": pd.to_datetime(df["XpryDt"]),
                "strike": pd.to_numeric(df["StrkPric"], errors="coerce").astype(float),
                "is_call": df["OptnTp"].astype(str).str.strip().eq("CE"),
                "close": df["ClsPric"].astype(float),
                "settle": df["SttlmPric"].astype(float),
                "oi": df["OpnIntrst"].astype(float),
                "chg_oi": df["ChngInOpnIntrst"].astype(float),
                "volume": df["TtlTradgVol"].astype(float),
                "underlying": df["UndrlygPric"].astype(float) if "UndrlygPric" in df else np.nan,
                "lot_size": df["NewBrdLotQty"].astype(float) if "NewBrdLotQty" in df else np.nan,
                "instrument": np.where(inst.eq("IDO"), "OPT", "FUT"),
                "value": df["TtlTrfVal"].astype(float) if "TtlTrfVal" in df else np.nan,
            }
        )
    else:
        raise ValueError(f"{path}: unrecognised bhavcopy layout")
    # UDiFF files are not ordered by strike (found on real files): sort so that any
    # neighbour-based step (IV gap filling, interpolation) sees strikes in order.
    out = out.sort_values(["symbol", "instrument", "expiry", "is_call", "strike"], kind="stable")
    return out[ALL_SCHEMA].reset_index(drop=True)


def load_bhavcopy(path, symbol: str = "NIFTY") -> pd.DataFrame:
    """Read one F&O bhavcopy (legacy or UDiFF) and return Nifty index options."""
    out = load_bhavcopy_all(path, symbols=(symbol,))
    return out[out["instrument"] == "OPT"][OPTION_SCHEMA].reset_index(drop=True)


def bhavcopy_files(directory) -> list:
    """All bhavcopy files in a folder (plain .csv or gzip .csv.gz), sorted by name."""
    d = Path(directory)
    return sorted(list(d.glob("*.csv")) + list(d.glob("*.csv.gz")))


def actual_expiries(bhav: dict, last_day: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Expiry dates that really happened, read from the bhavcopies themselves.

    ``bhav`` maps trade date to a tidy option table (load_bhavcopy). A contract
    expiry E counts if E is itself a trade date in the files and contracts
    with expiry E trade on E. This picks up holiday shifts (for example the
    Wednesday 25 Jan 2023 expiry before Republic Day) without a holiday list.
    """
    dates = sorted(bhav)
    have = set(dates)
    out = set()
    for d in dates:
        exps = pd.to_datetime(bhav[d]["expiry"]).dt.normalize().unique()
        if pd.Timestamp(d) in set(exps) and pd.Timestamp(d) in have:
            out.add(pd.Timestamp(d))
    return pd.DatetimeIndex(sorted(out))


def holiday_adjusted_expiries(bhav: dict) -> dict:
    """Map listed expiry dates that turned out to be exchange holidays to the real expiry day.

    When an expiry date becomes a holiday after the contract is listed, NSE
    moves the expiry to the previous trading day, but the bhavcopies before the
    change can still carry the old date (for example contracts listed for
    Thursday 29 June 2023 that really expired on Wednesday 28 June). Counting
    business days to the listed date then overstates the time to expiry.
    A listed date inside the span of the files that is not itself a trade
    date is mapped to the last trade date before it. Dates after the last
    file are left alone (their holidays are unknown).
    """
    dates = pd.DatetimeIndex(sorted(pd.Timestamp(d).normalize() for d in bhav))
    if len(dates) == 0:
        return {}
    have = set(dates)
    out = {}
    for df in bhav.values():
        for e in pd.to_datetime(df["expiry"]).dt.normalize().unique():
            e = pd.Timestamp(e)
            if e in out or e in have or e > dates[-1] or e < dates[0]:
                continue
            before = dates[dates < e]
            if len(before):
                out[e] = before[-1]
    return out


def parity_forward(chain: pd.DataFrame, T: float, r: float) -> float:
    """Forward index level for the chain's expiry from put-call parity.

    Uses the strike where call and put settlement prices are closest (the
    same step as the VIX methodology): F = K + exp(rT) (C - P).
    Returns NaN if no strike has both a call and a put price.
    """
    c = chain[chain["is_call"]].set_index("strike")["settle"]
    p = chain[~chain["is_call"]].set_index("strike")["settle"]
    both = pd.concat([c, p], axis=1, keys=["c", "p"]).dropna()
    both = both[(both["c"] > 0) & (both["p"] > 0)]
    if both.empty:
        return float("nan")
    k = (both["c"] - both["p"]).abs().idxmin()
    return float(k + np.exp(r * T) * (both.loc[k, "c"] - both.loc[k, "p"]))


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
    with open(path, encoding="utf-8-sig") as fh:
        lines = fh.readlines()
    hdr_row = next(k for k, ln in enumerate(lines) if ln.strip().strip('"').startswith("Client Type"))
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


# ---------------------------------------------------------------------------
# Build a "market" dict (see features.py) from NSE-layout files
# ---------------------------------------------------------------------------


def _participant_weights_row(tbl: pd.DataFrame) -> dict:
    tot_c = float(tbl.loc["TOTAL", "option_index_call_long"])
    tot_p = float(tbl.loc["TOTAL", "option_index_put_long"])
    out = {}
    for grp, key in (("Pro", "pro"), ("Client", "client")):
        r = tbl.loc[grp]
        out[f"{key}_call"] = (r["option_index_call_long"] - r["option_index_call_short"]) / tot_c
        out[f"{key}_put"] = (r["option_index_put_long"] - r["option_index_put_short"]) / tot_p
    return out


def build_market_from_nse(
    bars_path,
    bhavcopy_dir,
    participant_dir=None,
    vix_path=None,
    futures_bars_path=None,
    symbol: str = "NIFTY",
    lot_size: float | None = None,
    oi_unit: str = "units",
    r: float = 0.065,
    minutes: int = 375,
    start=None,
    end=None,
    min_coverage: float = 0.0,
    official_days=None,
    use_file_expiries: bool = False,
) -> dict:
    """Assemble the market dict from raw NSE-layout files.

    * ``bars_path``: 1-minute index bars (CSV, see load_minute_bars).
    * ``bhavcopy_dir``: one bhavcopy CSV per trading day, any file name; the
      trade date is read from inside the file.
    * ``participant_dir``: participant-wise OI CSVs, file names ending in
      DDMMYYYY.csv (the NSE naming), optional.
    * ``vix_path``: CSV with columns date and close (India VIX level in
      percent, as NSE publishes it), optional.
    * ``futures_bars_path``: 1-minute futures bars, used for volume.
    * ``oi_unit``: "units" if the bhavcopy reports OI in shares (divide by the
      lot size), "contracts" if already in contracts, "keep_units" to keep OI
      in shares and use a lot size of 1 (exact when lot sizes change; this is
      the right choice for real NSE files, which report shares).
    * ``start``/``end``: keep only days in this range.
    * ``min_coverage``: drop days whose minute bars cover less than this
      share of the session (removes Muhurat and other short special sessions).
    * ``official_days``: if given, keep only these trading days (for example
      the dates in NSE's daily index file).
    * ``use_file_expiries``: label expiry days from the bhavcopies (dates on
      which a contract really expired) instead of the weekday rule.

    For each day t the chain used is the nearest weekly expiry strictly after
    t, i.e. the contract that will be live on day t+1 (or expiring on it).
    """
    from .expiry_calendar import label_days

    bars = load_minute_bars(bars_path)
    if start is not None:
        bars = bars[bars.index >= pd.Timestamp(start)]
    if end is not None:
        bars = bars[bars.index < pd.Timestamp(end) + pd.Timedelta(days=1)]
    grid = to_session_grid(bars, minutes=minutes)
    keep = np.isfinite(grid["close"]).mean(axis=1) >= min_coverage
    if official_days is not None:
        keep &= grid["days"].isin(pd.DatetimeIndex(official_days).normalize())
    if not keep.all():
        for k in ("open", "high", "low", "close", "volume", "returns"):
            grid[k] = grid[k][keep]
        grid["days"] = grid["days"][keep]
    days = grid["days"]
    if futures_bars_path is not None:
        fg = to_session_grid(load_minute_bars(futures_bars_path), minutes=minutes)
        vol = pd.DataFrame(fg["volume"], index=fg["days"]).reindex(days).to_numpy()
        grid["volume"] = np.nan_to_num(vol, nan=0.0) + 1.0
    if not np.isfinite(grid["volume"]).any():
        grid["volume"] = np.ones_like(grid["close"])
    # bhavcopy, keyed by trade date
    bhav = {}
    other_bhav = {}  # other index (BANKNIFTY) options: only their expiry dates are kept
    fut_rows = {}
    lo, hi = days.min() - pd.Timedelta(days=10), days.max() + pd.Timedelta(days=10)
    for f in bhavcopy_files(bhavcopy_dir):
        stamp = "".join(ch for ch in Path(f).name if ch.isdigit())[-8:]
        try:
            fd = pd.to_datetime(stamp, format="%Y%m%d")
            if not (lo <= fd <= hi):
                continue
        except ValueError:
            pass
        full = load_bhavcopy_all(f, symbols=(symbol, "BANKNIFTY"))
        df = full[(full["instrument"] == "OPT") & (full["symbol"] == symbol)][OPTION_SCHEMA].reset_index(drop=True)
        if len(df):
            dkey = pd.Timestamp(df["date"].iloc[0]).normalize()
            bhav[dkey] = df
            other = full[(full["instrument"] == "OPT") & (full["symbol"] == "BANKNIFTY")]
            if len(other):
                other_bhav[dkey] = other[["date", "expiry"]]
            fu = full[(full["instrument"] == "FUT") & (full["symbol"] == symbol)].sort_values("expiry")
            if len(fu):
                fut_rows[dkey] = {"fut_value": float(fu["value"].sum()), "fut_oi": float(fu["oi"].sum()),
                                  "fut_contracts": float(fu["volume"].sum()),
                                  "fut_near_settle": float(fu["settle"].iloc[0]),
                                  "fut_near_expiry": fu["expiry"].iloc[0]}
    expiries = actual_expiries(bhav) if use_file_expiries else None
    labels = label_days(days, expiries=expiries)
    eff = holiday_adjusted_expiries(bhav)
    chains = []
    lot_used = []
    pos = {d: i for i, d in enumerate(days)}
    for t, d in enumerate(days):
        df = bhav.get(d)
        spot = float(grid["close"][t, np.isfinite(grid["close"][t])][-1])
        if df is None:
            chains.append(chains[-1].copy() if chains else pd.DataFrame(columns=["strike", "is_call", "oi", "settle"]))
            chains[-1].attrs.update({"spot": spot, "T": 1 / 252, "date": d, "missing": True})
            lot_used.append(lot_used[-1] if lot_used else (lot_size or np.nan))
            continue
        df = df.assign(expiry=df["expiry"].map(lambda e: eff.get(pd.Timestamp(e).normalize(), e)))
        nxt = df.loc[df["expiry"] > d, "expiry"].min()
        sel = df[df["expiry"] == nxt].copy()
        if oi_unit == "keep_units":
            ls = 1.0
        elif oi_unit == "units":
            if lot_size is None and sel["lot_size"].notna().all():
                # per-row conversion: lot sizes can differ across expiries on one day
                sel["oi"] = sel["oi"] / sel["lot_size"]
                sel["chg_oi"] = sel["chg_oi"] / sel["lot_size"]
                ls = float(sel["lot_size"].median())
            else:
                ls = lot_size if lot_size is not None else float(sel["lot_size"].dropna().iloc[0])
                sel = oi_in_units(sel, ls)
        else:
            ls = lot_size if lot_size is not None else float(sel["lot_size"].dropna().iloc[0])
        if nxt in pos:
            n_td = pos[nxt] - t
        else:
            n_td = int(np.busday_count(d.date(), pd.Timestamp(nxt).date()))
        ch = sel[["strike", "is_call", "oi", "settle"]].reset_index(drop=True)
        under = sel["underlying"].dropna()
        T_ch = max(n_td, 1) / 252
        fwd = parity_forward(ch, T_ch, r) if oi_unit == "keep_units" else float("nan")
        if np.isfinite(fwd) and abs(fwd / spot - 1) < 0.05:
            ch_spot = fwd  # forward for this expiry (pass carry=0 to build_gex_panel)
        else:
            ch_spot = float(under.iloc[0]) if len(under) else spot
        ch.attrs.update({"spot": ch_spot, "T": T_ch, "date": d, "expiry": nxt, "spot_index": spot})
        chains.append(ch)
        lot_used.append(ls)

    participant = None
    if participant_dir is not None:
        rows = {}
        for f in sorted(Path(participant_dir).glob("*.csv")):
            stamp = f.stem[-8:]
            try:
                d = pd.to_datetime(stamp, format="%d%m%Y")
            except ValueError:
                continue
            rows[d] = _participant_weights_row(load_participant_oi(f))
        participant = pd.DataFrame.from_dict(rows, orient="index").sort_index().reindex(days).ffill()

    vix = np.full(len(days), np.nan)
    if vix_path is not None:
        v = pd.read_csv(vix_path)
        v.columns = [c.strip().lower() for c in v.columns]
        # ISO dates (YYYY-MM-DD, as download_nse.py writes them) must not be read day-first:
        # pandas would swap day and month (2023-01-02 would become 1 February).
        ds = v["date"].astype(str)
        iso = ds.str.match(r"^\d{4}-\d{2}-\d{2}")
        dates = pd.to_datetime(ds.where(iso), format="ISO8601").fillna(pd.to_datetime(ds.where(~iso), dayfirst=True))
        v = v.set_index(dates)["close"].astype(float) / 100.0
        vix = v.reindex(days).ffill().to_numpy()
    if not np.isfinite(vix).all():
        # Fallback: annualised realized volatility of the day (flagged in attrs).
        rv = np.nansum(grid["returns"] ** 2, axis=1)
        vix = np.where(np.isfinite(vix), vix, np.sqrt(rv * 252))

    lots = np.asarray(lot_used, float)
    # Slim copy of every NIFTY option series published each day (all expiries), for
    # option-implied features (term structure, skew, put/call ratios). Keyed by publish date.
    option_tables = {}
    for d in days:
        df = bhav.get(d)
        if df is None:
            continue
        keep_cols = ["expiry", "strike", "is_call", "settle", "close", "oi", "volume"]
        sub = df[keep_cols].copy()
        # same holiday mapping as the chains: a contract listed for a holiday expires the day before
        sub["expiry"] = pd.to_datetime(sub["expiry"]).dt.normalize().map(lambda e: eff.get(e, e))
        sub = sub[sub["expiry"] > d]
        option_tables[d] = sub.reset_index(drop=True)
    futures = pd.DataFrame.from_dict(fut_rows, orient="index").reindex(days) if fut_rows else None
    other_expiries = actual_expiries(other_bhav) if other_bhav else pd.DatetimeIndex([])
    return {
        "source": "nse",
        "days": days,
        "labels": labels,
        "R": np.nan_to_num(grid["returns"], nan=0.0),
        "open": grid["open"], "high": grid["high"], "low": grid["low"], "close": grid["close"],
        "volume": grid["volume"],
        "chains": chains,
        "participant": participant,
        "vix": vix,
        "lot_size": 1.0 if oi_unit == "keep_units" else (float(np.nanmedian(lots)) if np.isfinite(lots).any() else 75.0),
        "r": r,
        "missing_bhavcopy_days": [c.attrs["date"] for c in chains if c.attrs.get("missing")],
        "option_tables": option_tables,
        "futures": futures,
        "other_index_expiries": other_expiries,
        "trading_days_all": pd.DatetimeIndex(sorted(bhav)),
        "expiry_source": "bhavcopy" if use_file_expiries else "weekday rule",
    }
