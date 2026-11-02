"""Expiry-day event study on REAL data: Nifty 50 and Bank Nifty, 2019 to 2026.

Question
--------
Does an index's own expiry day carry extra volatility (and volume)? How did
that "expiry premium" change at each NSE/SEBI rule change, and does removing
an index's weekly expiry remove its premium relative to an index that kept
its weekly? The last question is answered with a difference-in-differences
(DiD) that uses Nifty as the control for Bank Nifty at SEBI's 20 Nov 2024
measures.

Design (details and the reasons for each choice are in docs/EVENT_STUDY.md)
------
1. Expiry dates are read from the NSE F&O bhavcopies themselves (an expiry
   happened on day d if option contracts with expiry d trade on d; it is a
   monthly expiry if an index future also expires that day). Schedule
   changes (weekday moves, the end of Bank Nifty weeklies) are detected
   from these dates, not typed in. Before the first bhavcopy (20 Oct 2022)
   a Thursday rule with holiday roll-back is used and validated on the
   overlap.
2. Daily measures from 1-minute bars: realized variance (5-minute returns,
   the primary measure; 1-minute; first hour, last hour, last 30 minutes),
   summed per-minute Rogers-Satchell, absolute overnight return, and the
   realized correlation of the two indices' 5-minute returns. Daily
   derivatives turnover per index from the bhavcopies.
3. Week-relative measure: log value on a target day of a calendar week
   minus the mean log value on the other trading days of that week. Shocks
   common to the whole week cancel, so slow changes in the general level of
   volatility do not masquerade as expiry effects (experiment E2).
4. Inference uses weeks as the unit: Welch and Mann-Whitney tests on weekly
   values, Newey-West (HAC) regressions, a moving-block bootstrap of weeks,
   randomization inference over placebo event dates, pre-trend checks, and a
   day-level DiD regression with index-by-week fixed effects and standard
   errors clustered by week.

    OMP_NUM_THREADS=1 python experiments/expiry_event_study.py

Writes results/event_study/*.csv, *.png, summary_tables.md and
results/event_study_stdout.txt. Every number is REAL DATA.
"""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = {"N": "NIFTY", "BN": "BANKNIFTY"}
NAMES = {"N": "Nifty 50", "BN": "Bank Nifty"}
WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
SESSION_START_MIN = 9 * 60 + 15
N_MIN = 375
LONG_START = pd.Timestamp("2019-03-04")  # long sample start (see DESIGN_LOG)
PRIMARY = "rv5"
LOG_MEASURES = ["rv5", "rv1", "rs", "rv_first_hour", "rv_midday", "rv_last_hour", "rv_last30",
                "opt_premium", "opt_notional", "fut_notional"]
LEVEL_MEASURES = ["abs_overnight_bp", "corr5_z"]
MEASURES = LOG_MEASURES + LEVEL_MEASURES
# Diagnostics of how the premium depends on the sampling frequency (premium table only)
DIAG_MEASURES = ["rv1_over_rv5", "rs_over_rv5"]
ALL_MEASURES = MEASURES + DIAG_MEASURES
MEASURE_LABEL = {
    "rv5": "realized variance, 5-min returns, whole day (primary)",
    "rv1": "realized variance, 1-min returns, whole day",
    "rs": "Rogers-Satchell, summed over 1-min bars",
    "rv_first_hour": "realized variance, 1-min, 09:15 to 10:15",
    "rv_midday": "realized variance, 1-min, 10:15 to 14:30",
    "rv_last_hour": "realized variance, 1-min, 14:30 to 15:30",
    "rv_last30": "realized variance, 1-min, 15:00 to 15:30",
    "opt_premium": "option premium turnover (Rs)",
    "opt_notional": "option notional turnover (Rs)",
    "fut_notional": "futures turnover (Rs)",
    "abs_overnight_bp": "absolute overnight return (bp, level difference)",
    "corr5_z": "Fisher z of 5-min return correlation, Nifty vs Bank Nifty (level difference)",
    "rv1_over_rv5": "ratio of 1-min to 5-min realized variance (diagnostic)",
    "rs_over_rv5": "ratio of Rogers-Satchell to 5-min realized variance (diagnostic)",
}
DESIGN_LOG: list[str] = []


def log_design(msg: str) -> None:
    DESIGN_LOG.append(msg)


def seed_of(*parts) -> int:
    """Deterministic seed from labels (Python's hash() changes between runs)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2**31)


# ---------------------------------------------------------------------------
# 1. Bhavcopy: expiries, schedule changes, daily turnover
# ---------------------------------------------------------------------------


def read_bhavcopy_file(path) -> pd.DataFrame:
    """One F&O bhavcopy (legacy or UDiFF) -> NIFTY/BANKNIFTY index options and futures.

    Columns: date, symbol, inst ('opt'/'fut'), expiry, strike, close, contracts,
    notional (Rs; for options NSE reports (strike + premium) x quantity), lot
    (NaN in the legacy layout, which does not report it).
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "INSTRUMENT" in df.columns:
        out = pd.DataFrame({
            "date": pd.to_datetime(df["TIMESTAMP"].astype(str).str.strip(), format="%d-%b-%Y"),
            "symbol": df["SYMBOL"].astype(str).str.strip(),
            "inst": df["INSTRUMENT"].astype(str).str.strip().map({"OPTIDX": "opt", "FUTIDX": "fut"}),
            "expiry": pd.to_datetime(df["EXPIRY_DT"].astype(str).str.strip(), format="%d-%b-%Y"),
            "strike": pd.to_numeric(df["STRIKE_PR"], errors="coerce"),
            "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
            "contracts": pd.to_numeric(df["CONTRACTS"], errors="coerce"),
            "notional": pd.to_numeric(df["VAL_INLAKH"], errors="coerce") * 1e5,
            "lot": np.nan,
        })
    elif "FinInstrmTp" in df.columns:
        out = pd.DataFrame({
            "date": pd.to_datetime(df["TradDt"]),
            "symbol": df["TckrSymb"].astype(str).str.strip(),
            "inst": df["FinInstrmTp"].astype(str).str.strip().map({"IDO": "opt", "IDF": "fut"}),
            "expiry": pd.to_datetime(df["XpryDt"]),
            "strike": pd.to_numeric(df["StrkPric"], errors="coerce"),
            "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
            "contracts": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
            "notional": pd.to_numeric(df["TtlTrfVal"], errors="coerce"),
            "lot": pd.to_numeric(df["NewBrdLotQty"], errors="coerce"),
        })
    else:
        raise ValueError(f"{path}: unrecognised bhavcopy layout")
    keep = out["symbol"].isin(SYMBOLS.values()) & out["inst"].notna()
    return out[keep].reset_index(drop=True)


def lot_history(bhav: pd.DataFrame, reported) -> pd.DataFrame:
    """Runs of the daily median option lot size per symbol. ``reported`` marks
    rows whose lot size is printed in the file (UDiFF); the rest are inferred."""
    b = bhav.assign(reported=np.asarray(reported, bool))
    b = b[(b["inst"] == "opt") & (b["contracts"] > 0) & b["lot"].notna()]
    d = b.groupby(["symbol", "date"]).agg(lot=("lot", "median"), reported=("reported", "mean")).reset_index()
    d["lot"] = d["lot"].round().astype(int)
    d["source"] = np.where(d["reported"] > 0.5, "printed in file (UDiFF)", "inferred from turnover (legacy layout)")
    rows = []
    for sym, g in d.groupby("symbol"):
        g = g.sort_values("date")
        run = ((g["lot"] != g["lot"].shift()) | (g["source"] != g["source"].shift())).cumsum()
        for _, r in g.groupby(run):
            rows.append({"symbol": sym, "lot": int(r["lot"].iloc[0]), "from": r["date"].iloc[0].date(),
                         "to": r["date"].iloc[-1].date(), "trade_dates": len(r), "source": r["source"].iloc[0]})
    return pd.DataFrame(rows)


def load_all_bhavcopy(dirs) -> pd.DataFrame:
    files = []
    for d in dirs:
        files += sorted(Path(d).glob("fo_*.csv*"))
    parts = [read_bhavcopy_file(f) for f in files]
    return pd.concat(parts, ignore_index=True)


def fill_legacy_lots(bhav: pd.DataFrame) -> pd.DataFrame:
    """Legacy files omit the lot size; recover it from turnover / (contracts x price).

    For options NSE turnover is (strike + premium) x quantity, so
    notional / (contracts x (strike + close)) is the lot size up to the gap
    between the close and the traded prices (checked: 1st to 99th
    percentile within 0.3 of an integer). The estimate is rounded and the
    median is taken per symbol, trade date and expiry.
    """
    b = bhav.copy()
    miss = b["lot"].isna() & (b["contracts"] > 0)
    price = np.where(b["inst"] == "opt", b["strike"] + b["close"], b["close"])
    est = np.round(b["notional"] / (b["contracts"] * price))
    b["lot_est"] = np.where(miss, est, np.nan)
    med = b.groupby(["symbol", "date", "expiry"])["lot_est"].transform("median")
    b["lot"] = b["lot"].fillna(med)
    return b.drop(columns="lot_est")


def expiry_table(bhav: pd.DataFrame) -> pd.DataFrame:
    """Actual expiries per symbol: option contracts with expiry d trade on d.

    kind is 'monthly' when an index future of the same symbol also expires
    on d (futures exist only for monthly contracts), otherwise 'weekly'.
    """
    rows = []
    for sym, g in bhav.groupby("symbol"):
        opt = g[g["inst"] == "opt"]
        fut_exp = set(g.loc[g["inst"] == "fut", "expiry"])
        exp = sorted(set(opt.loc[opt["expiry"] == opt["date"], "expiry"]))
        for e in exp:
            rows.append({"symbol": sym, "expiry": pd.Timestamp(e), "weekday": WD[pd.Timestamp(e).weekday()]
                         if pd.Timestamp(e).weekday() < 5 else "Sat/Sun",
                         "kind": "monthly" if e in fut_exp else "weekly", "source": "bhavcopy"})
    return pd.DataFrame(rows)


def detect_schedule_changes(exp: pd.DataFrame, confirm: int = 3, stop_gap_days: int = 21,
                            data_end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find weekday changes and the end of a weekly series from actual expiry dates.

    Per (symbol, kind) the expiries are walked in order. A different weekday
    starts a new regime only if it is confirmed by ``confirm`` consecutive
    expiries on that same new weekday; an isolated different weekday is a
    holiday shift. A weekly series that stops for more than
    ``stop_gap_days`` (with data continuing) is reported as stopped.

    Returns (changes, flagged) where flagged is ``exp`` with columns
    'scheduled_weekday' and 'shifted' added.
    """
    changes = []
    flagged = []
    for (sym, kind), g in exp.sort_values("expiry").groupby(["symbol", "kind"], sort=False):
        g = g.sort_values("expiry").reset_index(drop=True)
        wds = [d.weekday() for d in g["expiry"]]
        n = len(g)
        first = wds[: min(confirm, n)]
        cur = max(set(first), key=first.count)
        sched = []
        i = 0
        while i < n:
            if wds[i] != cur:
                nxt = wds[i: i + confirm]
                if len(nxt) == confirm and len(set(nxt)) == 1:
                    changes.append({"symbol": sym, "kind": kind, "change": "weekday",
                                    "last_old_expiry": g["expiry"].iloc[i - 1] if i > 0 else pd.NaT,
                                    "old_weekday": WD[cur], "first_new_expiry": g["expiry"].iloc[i],
                                    "new_weekday": WD[wds[i]]})
                    cur = wds[i]
            sched.append(cur)
            i += 1
        g["scheduled_weekday"] = [WD[s] for s in sched]
        g["shifted"] = [WD[w] != WD[s] if w < 5 else True for w, s in zip(wds, sched)]
        flagged.append(g)
        if kind == "weekly" and n > 1:
            gaps = g["expiry"].diff().dt.days
            for k in np.where(gaps > stop_gap_days)[0]:
                changes.append({"symbol": sym, "kind": kind, "change": "gap", "last_old_expiry": g["expiry"].iloc[k - 1],
                                "old_weekday": WD[sched[k - 1]], "first_new_expiry": g["expiry"].iloc[k],
                                "new_weekday": WD[sched[k]]})
            end = data_end if data_end is not None else exp["expiry"].max()
            if (end - g["expiry"].iloc[-1]).days > stop_gap_days:
                changes.append({"symbol": sym, "kind": kind, "change": "stopped",
                                "last_old_expiry": g["expiry"].iloc[-1], "old_weekday": WD[sched[-1]],
                                "first_new_expiry": pd.NaT, "new_weekday": "none"})
    ch = pd.DataFrame(changes)
    if len(ch):
        ch = ch.sort_values(["symbol", "kind", "last_old_expiry"]).reset_index(drop=True)
    return ch, pd.concat(flagged, ignore_index=True) if flagged else exp.assign(scheduled_weekday=[], shifted=[])


def rule_expiries(trading_days, weekday: int, start, end, symbol: str) -> pd.DataFrame:
    """Rule-based expiries: scheduled weekday each calendar week, rolled back to the
    last trading day on or before it within the same week; the last expiry of each
    calendar month is the monthly one."""
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_days).normalize())))
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    day_set = set(days)
    rows, sched = [], []
    mondays = pd.date_range(days.min() - pd.Timedelta(days=days.min().weekday()), days.max(), freq="7D")
    for mon in mondays:
        s = mon + pd.Timedelta(days=weekday)
        d = s
        while d >= mon and d not in day_set:
            d -= pd.Timedelta(days=1)
        if d >= mon and d in day_set:
            rows.append(d)
            sched.append(s)
    e = pd.DataFrame({"symbol": symbol, "expiry": pd.DatetimeIndex(rows), "sched": pd.DatetimeIndex(sched)})
    # the monthly contract is the last scheduled weekday of the month; a holiday roll-back
    # can move the 1st-of-month weekly into the previous month, so group by scheduled date
    last = e.groupby(e["sched"].dt.to_period("M"))["sched"].transform("max")
    e["kind"] = np.where(e["sched"] == last, "monthly", "weekly")
    e = e.drop(columns="sched")
    e["weekday"] = [WD[d.weekday()] for d in e["expiry"]]
    e["scheduled_weekday"] = WD[weekday]
    e["shifted"] = e["weekday"] != WD[weekday]
    e["source"] = "rule (Thursday, holiday roll-back)"
    return e


def daily_turnover(bhav: pd.DataFrame) -> pd.DataFrame:
    """Per symbol and day: option premium and notional turnover, futures turnover (Rs)."""
    b = bhav.copy()
    b["shares"] = b["contracts"] * b["lot"]
    is_opt = b["inst"] == "opt"
    b["premium"] = np.where(is_opt, b["notional"] - b["strike"] * b["shares"], np.nan)
    b["zero_dte"] = is_opt & (b["expiry"] == b["date"])
    g = b.groupby(["symbol", "date"])
    out = pd.DataFrame({
        "opt_premium": b[is_opt].groupby(["symbol", "date"])["premium"].sum().clip(lower=0),
        "opt_notional": b[is_opt].groupby(["symbol", "date"])["notional"].sum(),
        "opt_contracts": b[is_opt].groupby(["symbol", "date"])["contracts"].sum(),
        "fut_notional": b[~is_opt].groupby(["symbol", "date"])["notional"].sum(),
        "opt_premium_0dte": b[b["zero_dte"]].groupby(["symbol", "date"])["premium"].sum(),
    })
    out["opt_premium_0dte"] = out["opt_premium_0dte"].fillna(0.0)
    out["share_premium_0dte"] = out["opt_premium_0dte"] / out["opt_premium"]
    _ = g
    return out.reset_index()


# ---------------------------------------------------------------------------
# 2. Minute bars -> daily realized measures
# ---------------------------------------------------------------------------


def minute_grid(path) -> dict:
    """1-minute bars on a (days x 375) grid for the 09:15 to 15:30 session."""
    b = pd.read_csv(path)
    ts = pd.to_datetime(b["timestamp"])
    slot = (ts.dt.hour * 60 + ts.dt.minute - SESSION_START_MIN).to_numpy()
    keep = (slot >= 0) & (slot < N_MIN)
    ts, slot, b = ts[keep], slot[keep], b[keep]
    day = ts.dt.normalize()
    days = pd.DatetimeIndex(sorted(day.unique()))
    row = pd.Series(np.arange(len(days)), index=days)[day].to_numpy()
    grid = {"days": days}
    for c in ("open", "high", "low", "close"):
        a = np.full((len(days), N_MIN), np.nan)
        a[row, slot] = b[c].to_numpy(float)
        grid[c] = a
    return grid


def daily_from_grid(grid: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Daily measures, 15-minute bucket variances and 5-minute returns per day."""
    o, h, l, c = grid["open"], grid["high"], grid["low"], grid["close"]
    n_bars = np.isfinite(c).sum(axis=1)
    flat = np.nanmean(np.where(np.isfinite(c), (h == l).astype(float), np.nan), axis=1)
    # bars with high == low in the last 20 minutes (15:10 to 15:29): a closing auction freezes the index
    flat_last20 = (h[:, 355:] == l[:, 355:]).sum(axis=1)
    first_open = pd.DataFrame(o).bfill(axis=1).to_numpy()[:, 0]
    cff = pd.DataFrame(c).ffill(axis=1).to_numpy()
    prev = np.concatenate([first_open[:, None], cff[:, :-1]], axis=1)
    prev = np.where(np.isfinite(prev), prev, first_open[:, None])
    r1 = np.where(np.isfinite(c), np.log(c / prev), 0.0)
    rs = np.where(np.isfinite(c), np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o), 0.0)
    c5 = cff[:, 4::5]
    p5 = np.concatenate([first_open[:, None], c5[:, :-1]], axis=1)
    r5 = np.log(c5 / p5)
    last_close = cff[:, -1]
    overnight = np.log(first_open / np.concatenate([[np.nan], last_close[:-1]]))
    sq = r1 ** 2
    df = pd.DataFrame({
        "rv1": sq.sum(axis=1),
        "rv5": np.nansum(r5 ** 2, axis=1),
        "rs": rs.sum(axis=1),
        "rv_first_hour": sq[:, :60].sum(axis=1),
        "rv_midday": sq[:, 60:315].sum(axis=1),
        "rv_last_hour": sq[:, 315:].sum(axis=1),
        "rv_last30": sq[:, 345:].sum(axis=1),
        "abs_overnight_bp": np.abs(overnight) * 1e4,
        "n_bars": n_bars,
        "flat_share": flat,
        "close_last": last_close,
        "flat_bars_last20": flat_last20,
    }, index=grid["days"])
    buckets = sq.reshape(len(sq), 25, 15).sum(axis=2)
    return df, buckets, r5


def detect_closing_auction(daily_df: pd.DataFrame, min_flat: int = 8, min_days: int = 10):
    """First day from which every day has a frozen close (>= ``min_flat`` flat bars in
    the last 20 minutes), if that run lasts at least ``min_days`` days; else None.

    From 3 Aug 2026 NSE sets the official close in a closing auction: the index
    is frozen for about 13 minutes before 15:28 to 15:29 and then jumps to the
    auction price.
    """
    flag = (daily_df["flat_bars_last20"] >= min_flat).to_numpy()
    days = daily_df.index
    tail_all = np.flip(np.logical_and.accumulate(np.flip(flag)))
    if not tail_all.any():
        return None
    k = int(np.argmax(tail_all))
    return days[k] if (len(days) - k) >= min_days else None


def valid_day_mask(days: pd.DatetimeIndex, n_bars, flat_share, official: set, official_start) -> np.ndarray:
    ok = (days.weekday < 5) & (np.asarray(n_bars) >= 370) & (np.asarray(flat_share) <= 0.2)
    in_off = np.array([(d < official_start) or (d in official) for d in days])
    return ok & in_off


# ---------------------------------------------------------------------------
# 3. Week-relative construction
# ---------------------------------------------------------------------------


def week_start(d) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(d)
    return (d - pd.to_timedelta(d.weekday, unit="D")).normalize()


def week_relative(values: pd.Series, targets: pd.Series, log: bool = True, min_other: int = 2,
                  exclude=None) -> pd.DataFrame:
    """Target-day value minus the mean of the other days of the same calendar week.

    ``values``: daily values indexed by trading day (only valid days; NaN
    allowed). ``targets``: target date per week, indexed by week start
    (Monday). With ``log=True`` logs are taken first (non-positive -> NaN).
    A week is kept if the target day has a finite value and at least
    ``min_other`` other days do. ``exclude``: dates never used in the
    baseline (for example the other index's expiry days).
    Returns a frame indexed by week start with columns rel, target_date, n_other.
    """
    v = values.astype(float)
    if log:
        v = np.log(v.where(v > 0))
    wk = week_start(v.index)
    frame = pd.DataFrame({"v": v.to_numpy(), "week": wk}, index=v.index)
    rows = {}
    tgt = targets.dropna()
    groups = {k: g for k, g in frame.groupby("week")}
    for w, t in tgt.items():
        g = groups.get(pd.Timestamp(w))
        if g is None or pd.Timestamp(t) not in g.index:
            continue
        vt = g.loc[pd.Timestamp(t), "v"]
        others = g.drop(index=pd.Timestamp(t))["v"].dropna()
        if exclude is not None:
            others = others[~others.index.isin(list(exclude))]
        if not np.isfinite(vt) or len(others) < min_other:
            continue
        rows[pd.Timestamp(w)] = {"rel": float(vt - others.mean()), "target_date": pd.Timestamp(t),
                                 "n_other": int(len(others))}
    out = pd.DataFrame.from_dict(rows, orient="index")
    if out.empty:
        out = pd.DataFrame(columns=["rel", "target_date", "n_other"])
    out.index.name = "week"
    return out.sort_index()


def weekday_targets(days: pd.DatetimeIndex, weekday: int) -> pd.Series:
    d = pd.DatetimeIndex(days)
    d = d[d.weekday == weekday]
    return pd.Series(d, index=week_start(d))


def date_targets(days: pd.DatetimeIndex, dates) -> pd.Series:
    """One target per week: the date from ``dates`` that falls in that week (weeks
    with two such dates are dropped)."""
    d = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates)) & set(pd.DatetimeIndex(days))))
    s = pd.Series(d, index=week_start(d))
    return s[~s.index.duplicated(keep=False)]


# ---------------------------------------------------------------------------
# 4. Inference on weekly values
# ---------------------------------------------------------------------------


def mbb_means(x: np.ndarray, block: int, n_boot: int, rng) -> np.ndarray:
    """Circular block bootstrap distribution of the mean of x (in time order).

    Blocks of ``block`` consecutive weeks are drawn with wrap-around
    (Politis and Romano's circular version of the moving-block bootstrap),
    so every week has the same chance of being drawn and the bootstrap
    distribution is centred on the sample mean. (The plain moving-block
    version under-weights the first and last weeks, which shifted intervals
    noticeably at 15 to 20 weeks per side; see the design log.)
    """
    n = len(x)
    if n == 0:
        return np.full(n_boot, np.nan)
    # at least four blocks per sample, otherwise every draw is a rotation of the same weeks
    b = max(1, min(block, n // 4))
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % n).reshape(n_boot, nb * b)[:, :n]
    return x[idx].mean(axis=1)


def hac_post(pre: np.ndarray, post: np.ndarray, lags: int = 4) -> tuple[float, float, float]:
    """OLS of y on [1, post] with Newey-West standard errors; returns (diff, se, p)."""
    import statsmodels.api as sm

    y = np.concatenate([pre, post])
    X = sm.add_constant(np.concatenate([np.zeros(len(pre)), np.ones(len(post))]))
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    return float(fit.params[1]), float(fit.bse[1]), float(fit.pvalues[1])


def hac_mean(x: np.ndarray, lags: int = 4) -> tuple[float, float, float]:
    import statsmodels.api as sm

    fit = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    return float(fit.params[0]), float(fit.bse[0]), float(fit.pvalues[0])


def welch_t(pre: np.ndarray, post: np.ndarray) -> float:
    if len(pre) < 2 or len(post) < 2:
        return np.nan
    se = np.sqrt(pre.var(ddof=1) / len(pre) + post.var(ddof=1) / len(post))
    return float((post.mean() - pre.mean()) / se) if se > 0 else np.nan


def compare(pre, post, n_boot: int = 4999, block: int = 4, lags: int = 4, seed: int = 0) -> dict:
    """Pre versus post on weekly values: means, Welch, Mann-Whitney, HAC, block bootstrap."""
    pre = np.asarray(pre, float)
    post = np.asarray(post, float)
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    out = {"n_pre": len(pre), "n_post": len(post), "mean_pre": np.nan, "mean_post": np.nan, "diff": np.nan,
           "boot_lo": np.nan, "boot_hi": np.nan, "boot_p": np.nan, "hac_se": np.nan, "hac_p": np.nan,
           "welch_t": np.nan, "welch_p": np.nan, "mwu_p": np.nan}
    if len(pre) < 3 or len(post) < 3:
        return out
    rng = np.random.default_rng(seed)
    diff = post.mean() - pre.mean()
    draws = mbb_means(post, block, n_boot, rng) - mbb_means(pre, block, n_boot, rng)
    _, se, p = hac_post(pre, post, lags)
    t = welch_t(pre, post)
    out.update({
        "mean_pre": pre.mean(), "mean_post": post.mean(), "diff": diff,
        "boot_lo": float(np.percentile(draws, 2.5)), "boot_hi": float(np.percentile(draws, 97.5)),
        "boot_p": float((1 + np.sum(np.abs(draws - draws.mean()) >= abs(diff))) / (1 + n_boot)),
        "hac_se": se, "hac_p": p, "welch_t": t,
        "welch_p": float(stats.ttest_ind(post, pre, equal_var=False).pvalue),
        "mwu_p": float(stats.mannwhitneyu(post, pre, alternative="two-sided").pvalue),
    })
    return out


def level_summary(x, n_boot: int = 4999, block: int = 4, lags: int = 4, seed: int = 0) -> dict:
    """Mean of weekly values with a block-bootstrap CI, HAC p (H0: mean 0) and Wilcoxon p."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = {"n_weeks": len(x), "mean": np.nan, "boot_lo": np.nan, "boot_hi": np.nan, "hac_p": np.nan,
           "wilcoxon_p": np.nan, "share_positive": np.nan, "median": np.nan}
    if len(x) < 5:
        return out
    rng = np.random.default_rng(seed)
    draws = mbb_means(x, block, n_boot, rng)
    m, se, p = hac_mean(x, lags)
    out.update({"mean": m, "boot_lo": float(np.percentile(draws, 2.5)), "boot_hi": float(np.percentile(draws, 97.5)),
                "hac_p": p, "wilcoxon_p": float(stats.wilcoxon(x).pvalue), "share_positive": float((x > 0).mean()),
                "median": float(np.median(x))})
    return out


def placebo_event_indices(n_weeks: int, event_idx, W: int, blocked_idx, valid_start: int = 0) -> np.ndarray:
    """Week positions k usable as placebo events for windows of W weeks each side.

    A placebo window [k - W, k + W) must lie inside the grid (and after
    ``valid_start``), must not contain any real event (``blocked_idx``)
    strictly inside it, and k itself must not be a real event.
    """
    blocked = np.asarray(sorted(set(int(b) for b in blocked_idx)), int)
    ks = []
    for k in range(valid_start + W, n_weeks - W + 1):
        if k in set(blocked.tolist()):
            continue
        if np.any((blocked > k - W) & (blocked < k + W)):
            continue
        ks.append(k)
    _ = event_idx
    return np.asarray(ks, int)


def placebo_t_stats(values: np.ndarray, ks: np.ndarray, W: int, min_n: int = 5) -> np.ndarray:
    """Welch t of (post - pre) means at each placebo position, NaN-aware."""
    out = np.full(len(ks), np.nan)
    for j, k in enumerate(ks):
        pre = values[k - W:k]
        post = values[k:k + W]
        pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
        if len(pre) >= min_n and len(post) >= min_n:
            out[j] = welch_t(pre, post)
    return out


def n_independent(ks: np.ndarray, W: int) -> int:
    """Greedy count of placebo positions whose windows (2W weeks) do not overlap."""
    n, last = 0, -10**9
    for k in np.sort(np.asarray(ks, int)):
        if k - last >= 2 * W:
            n += 1
            last = k
    return n


def randomization_p(t_obs: float, t_placebo: np.ndarray, n_indep: int | None = None,
                    min_indep: int = 3) -> tuple[float, int]:
    """Share of placebo |t| at least as large as the actual |t| (with the +1 correction).

    Overlapping placebo windows are strongly dependent, so the reference
    distribution is only as rich as the number of non-overlapping windows;
    below ``min_indep`` of them no p-value is reported.
    """
    tp = t_placebo[np.isfinite(t_placebo)]
    if not np.isfinite(t_obs) or len(tp) < 10 or (n_indep is not None and n_indep < min_indep):
        return np.nan, len(tp)
    return float((1 + np.sum(np.abs(tp) >= abs(t_obs))) / (1 + len(tp))), len(tp)


def pretrend(weeks: pd.DatetimeIndex, y: np.ndarray, lags: int = 4) -> tuple[float, float]:
    """Slope of y on time (per year) in the pre window, HAC p-value."""
    import statsmodels.api as sm

    ok = np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan
    tyr = (pd.DatetimeIndex(weeks)[ok] - pd.DatetimeIndex(weeks)[ok][0]).days.to_numpy() / 365.25
    fit = sm.OLS(y[ok], sm.add_constant(tyr)).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    return float(fit.params[1]), float(fit.pvalues[1])


def did_day_regression(daily_logs: dict, targets: dict, weeks_pre, weeks_post) -> dict:
    """Day-level DiD with index-by-week fixed effects, SE clustered by week.

    y_{i,d} = a_{i,w} + b1 T_{i,d} + b2 T x Post + b3 T x A + b4 T x A x Post + e,
    where T marks the target day of index i and A is the treated index (the
    first key of ``daily_logs``). b4 is the DiD.
    """
    import statsmodels.api as sm

    keys = list(daily_logs)
    wk_pre, wk_post = set(pd.DatetimeIndex(weeks_pre)), set(pd.DatetimeIndex(weeks_post))
    rows = []
    for i, k in enumerate(keys):
        s = daily_logs[k].dropna()
        wk = week_start(s.index)
        tset = set(pd.DatetimeIndex(targets[k]))
        for d, w, v in zip(s.index, wk, s.to_numpy()):
            if w in wk_pre or w in wk_post:
                rows.append({"y": v, "idx": i, "week": w, "T": float(d in tset), "post": float(w in wk_post)})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"did": np.nan, "se": np.nan, "p": np.nan, "n_days": 0}
    # keep only index-weeks that contain a target day and at least two other days
    grp = df.groupby(["idx", "week"])
    keep = grp["T"].transform("sum").eq(1) & grp["T"].transform("size").ge(3)
    df = df[keep]
    # weeks present for both indices
    both = df.groupby("week")["idx"].nunique()
    df = df[df["week"].isin(both[both == 2].index)].reset_index(drop=True)
    if df.empty:
        return {"did": np.nan, "se": np.nan, "p": np.nan, "n_days": 0}
    A = (df["idx"] == 0).astype(float)
    X = pd.DataFrame({"T": df["T"], "T_post": df["T"] * df["post"], "T_A": df["T"] * A,
                      "T_A_post": df["T"] * A * df["post"]})
    fe = pd.get_dummies(df["idx"].astype(str) + "_" + df["week"].dt.strftime("%Y%m%d"), dtype=float)
    X = pd.concat([X, fe], axis=1)
    groups = pd.factorize(df["week"])[0]
    fit = sm.OLS(df["y"].to_numpy(), X.to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    j = list(X.columns).index("T_A_post")
    return {"did": float(fit.params[j]), "se": float(fit.bse[j]), "p": float(fit.pvalues[j]),
            "n_days": int(len(df)), "n_weeks": int(df["week"].nunique())}


# ---------------------------------------------------------------------------
# 5. Assemble the data
# ---------------------------------------------------------------------------


def build_data(raw: Path):
    t0 = time.time()
    lines = []
    idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
    official = set(idx.loc[idx["index"].str.lower() == "nifty 50", "date"])
    official_start = min(official)
    lines.append(f"Official NSE trading days in index_daily.csv: {len(official)} "
                 f"({official_start.date()} to {max(official).date()}).")

    # bhavcopy
    dirs = [raw / "extra_2022" / "bhavcopy", raw / "bhavcopy"]
    bhav_raw = load_all_bhavcopy(dirs)
    bhav = fill_legacy_lots(bhav_raw)
    lots = lot_history(bhav, bhav_raw["lot"].notna().to_numpy())
    bdays = set(bhav["date"])
    lines.append(f"Bhavcopy files read: {len(bdays)} trade dates, {min(bdays).date()} to {max(bdays).date()}, "
                 f"{len(bhav):,} NIFTY/BANKNIFTY rows.")
    miss_b = sorted(d.date() for d in official if d not in bdays)
    extra_b = sorted(d.date() for d in bdays if d >= official_start and d not in official)
    lines.append(f"Official days without a bhavcopy: {miss_b}; bhavcopy days not official: {extra_b}")
    lot_missing = bhav["lot"].isna() & (bhav["contracts"] > 0)
    lines.append(f"Rows with trades but no lot size after inference: {int(lot_missing.sum())}")
    lines.append("Option lot size (daily median over traded contracts; runs):\n" + lots.to_string(index=False))

    # expiries and schedule
    exp_b = expiry_table(bhav)
    changes, exp_flag = detect_schedule_changes(exp_b, data_end=max(bdays))

    # minute bars
    grids, daily, buckets, r5 = {}, {}, {}, {}
    for k, fname in (("N", "nifty_1min.csv"), ("BN", "banknifty_1min.csv")):
        g = minute_grid(raw / fname)
        df, bk, rr = daily_from_grid(g)
        ok = valid_day_mask(df.index, df["n_bars"], df["flat_share"], official, official_start)
        dropped = df.index[~ok]
        lines.append(f"{NAMES[k]} minute bars: {len(df)} days {df.index.min().date()} to {df.index.max().date()}; "
                     f"dropped {len(dropped)} (weekend sessions, <370 bars, >20% flat bars or not an official day): "
                     f"{[d.date().isoformat() for d in dropped]}")
        daily[k] = df[ok]
        buckets[k] = pd.DataFrame(bk[ok], index=df.index[ok])
        r5[k] = pd.DataFrame(rr[ok], index=df.index[ok])
    # realized correlation of the two indices on common days (Fisher z)
    common = daily["N"].index.intersection(daily["BN"].index)
    a, b = r5["N"].loc[common].to_numpy(), r5["BN"].loc[common].to_numpy()
    rho = np.nansum(a * b, axis=1) / np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
    z = pd.Series(np.arctanh(np.clip(rho, -0.999, 0.999)), index=common)
    for k in daily:
        daily[k]["corr5_z"] = z.reindex(daily[k].index)
        daily[k]["rv1_over_rv5"] = daily[k]["rv1"] / daily[k]["rv5"]
        daily[k]["rs_over_rv5"] = daily[k]["rs"] / daily[k]["rv5"]

    # turnover
    turn = daily_turnover(bhav)
    for k, sym in SYMBOLS.items():
        t = turn[turn["symbol"] == sym].set_index("date")
        for c in ("opt_premium", "opt_notional", "fut_notional", "opt_contracts", "share_premium_0dte"):
            daily[k][c] = t[c].reindex(daily[k].index)

    # calendar: rule before the first bhavcopy, bhavcopy after
    first_b = min(bdays)
    trading = daily["N"].index.union(daily["BN"].index)
    exp_all = []
    for k, sym in SYMBOLS.items():
        rule = rule_expiries(trading, 3, LONG_START, first_b - pd.Timedelta(days=1), sym)
        # validate the rule on the overlap where the schedule was still Thursday
        cut = {"NIFTY": pd.Timestamp("2025-08-31"), "BANKNIFTY": pd.Timestamp("2023-08-31")}[sym]
        rv = rule_expiries(trading, 3, first_b, cut, sym)
        bb = exp_flag[(exp_flag["symbol"] == sym) & (exp_flag["expiry"] <= cut)]
        same = set(rv["expiry"]) == set(bb["expiry"])
        mon_rule = set(rv.loc[rv["kind"] == "monthly", "expiry"])
        mon_b = set(bb.loc[bb["kind"] == "monthly", "expiry"])
        lines.append(f"Rule check {sym} {first_b.date()} to {cut.date()}: rule {len(rv)} expiries, bhavcopy {len(bb)}; "
                     f"identical dates: {same}; rule-only {sorted(d.date().isoformat() for d in set(rv['expiry']) - set(bb['expiry']))}; "
                     f"bhavcopy-only {sorted(d.date().isoformat() for d in set(bb['expiry']) - set(rv['expiry']))}; "
                     f"monthly labels identical: {mon_rule == mon_b} (differences: "
                     f"{sorted(d.date().isoformat() for d in mon_rule ^ mon_b)})")
        exp_all += [rule, exp_flag[exp_flag["symbol"] == sym]]
    expiries = pd.concat(exp_all, ignore_index=True).sort_values(["symbol", "expiry"]).reset_index(drop=True)
    auction = detect_closing_auction(daily["N"])
    if auction is not None:
        n_auc = int((daily["N"].index >= auction).sum())
        before = int((daily["N"].loc[:auction - pd.Timedelta(days=1), "flat_bars_last20"] >= 8).sum())
        lines.append(f"Closing auction detected in the Nifty bars from {auction.date()} ({n_auc} days, every one with "
                     f">= 8 flat bars in the last 20 minutes; {before} valid days before it had that many).")
    lines.append(f"Data assembled in {time.time() - t0:.1f}s.")
    return {"daily": daily, "buckets": buckets, "expiries": expiries, "changes": changes, "lines": lines,
            "bhav_days": bdays, "first_bhav": first_b, "auction_start": auction, "lots": lots}


def define_events(changes: pd.DataFrame, auction_start=None) -> pd.DataFrame:
    """Turn detected schedule changes into event weeks (Monday of the first treated week)."""
    def pick(sym, kind, change, new_wd=None):
        c = changes[(changes["symbol"] == sym) & (changes["kind"] == kind) & (changes["change"] == change)]
        if new_wd is not None:
            c = c[c["new_weekday"] == new_wd]
        if len(c) != 1:
            raise RuntimeError(f"expected one {change} for {sym} {kind} {new_wd}, found {len(c)}:\n{c}")
        return c.iloc[0]

    rows = []
    c = pick("BANKNIFTY", "weekly", "weekday", "Wed")
    rows.append({"event": "a", "description": "Bank Nifty weekly expiry moves Thursday to Wednesday",
                 "last_old_expiry": c["last_old_expiry"], "first_new_expiry": c["first_new_expiry"],
                 "event_week": week_start([c["first_new_expiry"]])[0]})
    c = pick("BANKNIFTY", "monthly", "weekday", "Wed")
    rows.append({"event": "b", "description": "Bank Nifty monthly expiry moves Thursday to Wednesday (paper's event)",
                 "last_old_expiry": c["last_old_expiry"], "first_new_expiry": c["first_new_expiry"],
                 "event_week": week_start([c["last_old_expiry"] + pd.Timedelta(days=7)])[0]})
    c = pick("BANKNIFTY", "weekly", "stopped")
    rows.append({"event": "c", "description": "Bank Nifty weekly options stop (SEBI measures of 20 Nov 2024); Nifty keeps weeklies",
                 "last_old_expiry": c["last_old_expiry"], "first_new_expiry": pd.NaT,
                 "event_week": week_start([c["last_old_expiry"] + pd.Timedelta(days=7)])[0]})
    c = pick("BANKNIFTY", "monthly", "weekday", "Thu")
    rows.append({"event": "c2", "description": "Bank Nifty monthly expiry moves Wednesday to last Thursday",
                 "last_old_expiry": c["last_old_expiry"], "first_new_expiry": c["first_new_expiry"],
                 "event_week": week_start([c["first_new_expiry"]])[0]})
    c = pick("NIFTY", "weekly", "weekday", "Tue")
    rows.append({"event": "d", "description": "Nifty weekly expiry moves Thursday to Tuesday (monthlies of both indices to Tuesday)",
                 "last_old_expiry": c["last_old_expiry"], "first_new_expiry": c["first_new_expiry"],
                 "event_week": week_start([c["first_new_expiry"]])[0]})
    if auction_start is not None:
        rows.append({"event": "e", "description": "Closing auction sets the official close (frozen index before 15:29; "
                     "found in the minute bars, not the bhavcopy)", "last_old_expiry": pd.NaT,
                     "first_new_expiry": pd.NaT, "event_week": week_start([auction_start])[0]})
    return pd.DataFrame(rows)


def week_flags(expiries: pd.DataFrame, daily: dict) -> pd.DataFrame:
    """Per calendar week: expiries of each index, monthly / shifted flags, day counts."""
    weeks = week_start(daily["N"].index.union(daily["BN"].index)).unique().sort_values()
    f = pd.DataFrame(index=weeks)
    f.index.name = "week"
    e = expiries.copy()
    e["week"] = week_start(e["expiry"])
    for k, sym in SYMBOLS.items():
        ek = e[e["symbol"] == sym]
        f[f"{k}_expiries"] = ek.groupby("week")["expiry"].apply(lambda s: ";".join(d.strftime("%Y-%m-%d") for d in s))
        f[f"{k}_monthly"] = ek.groupby("week")["kind"].apply(lambda s: (s == "monthly").any()).reindex(weeks).fillna(False).astype(bool)
        f[f"{k}_shifted"] = ek.groupby("week")["shifted"].apply(lambda s: bool(s.any())).reindex(weeks).fillna(False).astype(bool)
        f[f"{k}_n_expiries"] = ek.groupby("week").size().reindex(weeks).fillna(0).astype(int)
        f[f"{k}_days"] = pd.Series(1, index=daily[k].index).groupby(week_start(daily[k].index)).sum().reindex(weeks).fillna(0).astype(int)
    f["monthly_any"] = f["N_monthly"] | f["BN_monthly"]
    f["shifted_any"] = f["N_shifted"] | f["BN_shifted"]
    f["clean_weekly"] = ~f["monthly_any"] & ~f["shifted_any"]
    f["clean_monthly"] = f["monthly_any"] & ~f["shifted_any"]
    f["regular"] = ~f["shifted_any"]
    return f


def build_relative(daily: dict, expiries: pd.DataFrame) -> pd.DataFrame:
    """Long table of week-relative values: index x target x measure x week."""
    rows = []
    for k, sym in SYMBOLS.items():
        days = daily[k].index
        tg = {WD[i]: weekday_targets(days, i) for i in range(5)}
        tg["own"] = date_targets(days, expiries.loc[expiries["symbol"] == sym, "expiry"])
        for tname, targets in tg.items():
            for m in ALL_MEASURES:
                if m not in daily[k] or daily[k][m].notna().sum() == 0:
                    continue
                wr = week_relative(daily[k][m], targets, log=m not in LEVEL_MEASURES)
                if wr.empty:
                    continue
                wr = wr.reset_index()
                wr["index"] = k
                wr["target"] = tname
                wr["measure"] = m
                rows.append(wr)
    out = pd.concat(rows, ignore_index=True)
    return out[["index", "target", "measure", "week", "target_date", "rel", "n_other"]]


# ---------------------------------------------------------------------------
# 6. Analyses
# ---------------------------------------------------------------------------


class Series:
    """Weekly series built from the long relative table: rel(A) or rel(A) - rel(B)."""

    def __init__(self, rel: pd.DataFrame, flags: pd.DataFrame, a, b=None, sample: str = "clean_weekly",
                 grid_start=LONG_START):
        self.rel, self.flags, self.a, self.b, self.sample = rel, flags, a, b, sample
        self.grid = flags.index[flags.index >= grid_start]

    def values(self, measure: str) -> pd.Series:
        def one(spec):
            idx, tgt = spec
            r = self.rel[(self.rel["index"] == idx) & (self.rel["target"] == tgt) & (self.rel["measure"] == measure)]
            return r.set_index("week")["rel"]
        y = one(self.a)
        if self.b is not None:
            y = y - one(self.b)
        y = y.reindex(self.grid)
        if self.sample != "all":
            y = y.where(self.flags[self.sample].reindex(self.grid).fillna(False).astype(bool))
        return y

    def label(self) -> str:
        s = f"{self.a[0]} {self.a[1]}"
        if self.b is not None:
            s += f" minus {self.b[0]} {self.b[1]}"
        return s


def window_split(y: pd.Series, event_week, W, lo=None, hi=None, pre_start=None, pre_end=None, post_start=None, post_end=None):
    """Pre: [event - W weeks, event) above lo; post: [event, event + W weeks) below hi.
    Explicit date windows override W (used for the paper's own window)."""
    wk = y.index
    if pre_start is not None:
        pre = y[(wk >= pre_start) & (wk <= pre_end)]
        post = y[(wk >= post_start) & (wk <= post_end)]
        return pre, post
    e = pd.Timestamp(event_week)
    p0 = e - pd.Timedelta(weeks=W)
    p1 = e + pd.Timedelta(weeks=W)
    if lo is not None:
        p0 = max(p0, pd.Timestamp(lo))
    if hi is not None:
        p1 = min(p1, pd.Timestamp(hi))
    return y[(wk >= p0) & (wk < e)], y[(wk >= e) & (wk < p1)]


def run_event_tests(rel, flags, events, specs, windows, blocked_weeks, n_boot, measures=MEASURES):
    out = []
    for sp in specs:
        ser = Series(rel, flags, sp["a"], sp.get("b"), sp.get("sample", "clean_weekly"))
        ev_week = events.set_index("event").loc[sp["event"], "event_week"]
        for m in measures:
            if m in sp.get("skip_measures", []):
                continue
            if m == "corr5_z" and sp.get("b") is not None and sp["b"][1] == sp["a"][1]:
                continue  # the correlation is one number per day: identical for both indices
            y = ser.values(m)
            if y.notna().sum() < 6:
                continue
            grid_vals = y.to_numpy()
            grid_weeks = y.index
            for wname, W in windows:
                if wname == "paper window":
                    if sp["event"] != "b":
                        continue
                    pre, post = window_split(y, ev_week, None, pre_start=pd.Timestamp("2023-10-30"),
                                             pre_end=pd.Timestamp("2024-02-26"), post_start=pd.Timestamp("2024-03-04"),
                                             post_end=pd.Timestamp("2024-05-27"))
                    Wp = 17
                else:
                    pre, post = window_split(y, ev_week, W, sp.get("lo"), sp.get("hi"))
                    Wp = W
                res = compare(pre.to_numpy(), post.to_numpy(), n_boot=n_boot,
                              seed=seed_of(sp["id"], m, wname))
                # randomization inference over placebo event dates (studentized)
                blocked_idx = [int(np.searchsorted(grid_weeks, b)) for b in blocked_weeks]
                ks = placebo_event_indices(len(grid_weeks), None, Wp, blocked_idx)
                tp = placebo_t_stats(grid_vals, ks, Wp)
                ks_ok = ks[np.isfinite(tp)]
                n_ind = n_independent(ks_ok, Wp)
                p_ri, n_pl = randomization_p(res["welch_t"], tp, n_ind)
                slope, p_tr = pretrend(pre.index, pre.to_numpy())
                row = {"spec_id": sp["id"], "event": sp["event"], "comparison": sp["label"], "series": ser.label(),
                       "sample": ser.sample, "measure": m, "window": wname,
                       "pre_first_week": pre.dropna().index.min(), "pre_last_week": pre.dropna().index.max(),
                       "post_first_week": post.dropna().index.min(), "post_last_week": post.dropna().index.max()}
                row.update(res)
                row.update({"ratio_post_over_pre": float(np.exp(res["diff"])) if m in LOG_MEASURES and np.isfinite(res["diff"]) else np.nan,
                            "ri_p": p_ri, "ri_n_placebo": n_pl, "ri_n_independent": n_ind,
                            "mde80": 2.8 * res["hac_se"] if np.isfinite(res["hac_se"]) else np.nan,
                            "pretrend_slope_per_year": slope, "pretrend_p": p_tr})
                out.append(row)
    return pd.DataFrame(out)


def premium_by_regime(rel, flags, regimes, n_boot):
    out = []
    for k in SYMBOLS:
        for tgt in ["own"] + WD:
            for sample in ("clean_weekly", "all"):
                if tgt != "own" and sample == "all":
                    continue
                for m in ALL_MEASURES:
                    ser = Series(rel, flags, (k, tgt), None, sample)
                    y = ser.values(m)
                    for rname, lo, hi in regimes:
                        x = y[(y.index >= lo) & (y.index < hi)].dropna()
                        if len(x) < 5:
                            continue
                        s = level_summary(x.to_numpy(), n_boot=n_boot, seed=seed_of(k, tgt, m, rname, sample))
                        row = {"index": k, "target": tgt, "sample": sample, "measure": m, "regime": rname,
                               "from_week": x.index.min(), "to_week": x.index.max()}
                        row.update(s)
                        if m not in LEVEL_MEASURES:
                            row.update({"ratio": np.exp(s["mean"]), "ratio_lo": np.exp(s["boot_lo"]),
                                        "ratio_hi": np.exp(s["boot_hi"])})
                        out.append(row)
    return pd.DataFrame(out)


def diurnal_profiles(buckets: dict, expiries, flags, regimes, targets_by_regime) -> pd.DataFrame:
    """Week-relative log variance per 15-minute bucket on the target day, by regime."""
    rows = []
    for k in SYMBOLS:
        bk = buckets[k]
        logb = np.log(bk.where(bk > 0))
        wk = week_start(bk.index)
        for rname, lo, hi in regimes:
            for tlabel, wd in targets_by_regime[(k, rname)]:
                if wd == "own":
                    tg = date_targets(bk.index, expiries.loc[expiries["symbol"] == SYMBOLS[k], "expiry"])
                else:
                    tg = weekday_targets(bk.index, WD.index(wd))
                tg = tg[(tg.index >= lo) & (tg.index < hi)]
                okw = flags["clean_weekly"].reindex(tg.index).fillna(False).astype(bool)
                tg = tg[okw.to_numpy()]
                vals = []
                for w, t in tg.items():
                    m = wk == w
                    if t not in bk.index or m.sum() < 3:
                        continue
                    g = logb[m]
                    others = g.drop(index=t)
                    vals.append(g.loc[t].to_numpy() - others.mean(axis=0).to_numpy())
                if len(vals) < 5:
                    continue
                v = np.array(vals)
                mean = np.nanmean(v, axis=0)
                se = np.nanstd(v, axis=0, ddof=1) / np.sqrt(np.isfinite(v).sum(axis=0))
                for j in range(25):
                    t0 = SESSION_START_MIN + 15 * j
                    rows.append({"index": k, "regime": rname, "target": tlabel, "bucket": j,
                                 "start": f"{t0 // 60:02d}:{t0 % 60:02d}", "mean_rel": mean[j], "se": se[j],
                                 "ratio": np.exp(mean[j]), "n_weeks": len(v)})
    return pd.DataFrame(rows)


def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN p-values are left as NaN)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def rel_weekday(daily: dict, k: str, m: str, wd: str, exclude=None, min_other: int = 2) -> pd.Series:
    tg = weekday_targets(daily[k].index, WD.index(wd))
    wr = week_relative(daily[k][m], tg, log=m not in LEVEL_MEASURES, min_other=min_other, exclude=exclude)
    return wr["rel"] if len(wr) else pd.Series(dtype=float)


def robustness_event_c(daily, expiries, flags, events, n_boot, measures=("rv5", "rs", "rv_last30", "opt_premium")):
    """Event c (Bank Nifty loses its weekly): the main DiD and its two parts under
    alternative choices. Windows are bounded by events a and d as in the main table."""
    ev = events.set_index("event")["event_week"]
    a_, c_, d_ = ev["a"], ev["c"], ev["d"]
    exp_n = set(expiries.loc[expiries["symbol"] == "NIFTY", "expiry"])
    exp_bn = set(expiries.loc[expiries["symbol"] == "BANKNIFTY", "expiry"])
    grid = flags.index[flags.index >= LONG_START]
    bn_post_expiry = (grid >= c_) & (flags["BN_n_expiries"].reindex(grid).to_numpy() > 0)
    variants = {
        "main": {},
        "baseline excludes the other index's expiry days": {"excl": True},
        "weeks with at least 3 other trading days": {"min_other": 3},
        "weekly values winsorized at 2.5 and 97.5 percent": {"wins": True},
        "bootstrap blocks of 2 weeks": {"block": 2},
        "bootstrap blocks of 8 weeks": {"block": 8},
        "HAC with 8 lags": {"lags": 8},
        "monthly-expiry weeks included (post-c weeks with any BN expiry dropped)": {"sample": "regular"},
    }
    rows = []
    for m in measures:
        for vname, o in variants.items():
            A = rel_weekday(daily, "BN", m, "Wed", exclude=exp_n if o.get("excl") else None,
                            min_other=o.get("min_other", 2)).reindex(grid)
            B = rel_weekday(daily, "N", m, "Thu", exclude=exp_bn if o.get("excl") else None,
                            min_other=o.get("min_other", 2)).reindex(grid)
            if o.get("sample") == "regular":
                mask = flags["regular"].reindex(grid).to_numpy() & ~bn_post_expiry
            else:
                mask = flags["clean_weekly"].reindex(grid).to_numpy()
            for sname, y in (("DiD main: BN Wed minus Nifty Thu", A - B), ("BN Wed", A), ("Nifty Thu", B)):
                y = y.where(mask)
                if o.get("wins"):
                    lo_q, hi_q = np.nanpercentile(y.to_numpy(), [2.5, 97.5])
                    y = y.clip(lo_q, hi_q)
                for wname, W in (("6 months", 26), ("12 months", 52)):
                    pre, post = window_split(y, c_, W, a_, d_)
                    r = compare(pre.to_numpy(), post.to_numpy(), n_boot=n_boot, block=o.get("block", 4),
                                lags=o.get("lags", 4), seed=seed_of("rob", m, vname, sname, wname))
                    rows.append({"measure": m, "series": sname, "variant": vname, "window": wname, **r})
    return pd.DataFrame(rows)


def own_premium_by_halfyear(rel, flags, events, n_boot, measures=("rv5", "rs", "rv_first_hour", "rv_last30", "opt_premium")):
    """Expiry-weekday premium per half-year: Nifty Thursday (Tuesday after d),
    Bank Nifty Thursday (Wednesday after a; former expiry weekday after c)."""
    ev = events.set_index("event")["event_week"]
    rows = []
    for k in SYMBOLS:
        for m in measures:
            parts = []
            for tgt, lo, hi in ((("Thu", LONG_START, ev["d"]), ("Tue", ev["d"], pd.Timestamp("2100-01-01"))) if k == "N"
                                else (("Thu", LONG_START, ev["a"]), ("Wed", ev["a"], pd.Timestamp("2100-01-01")))):
                y = Series(rel, flags, (k, tgt), None, "clean_weekly").values(m)
                y = y[(y.index >= lo) & (y.index < hi)]
                parts.append(pd.DataFrame({"val": y, "weekday": tgt}))
            y = pd.concat(parts).dropna(subset=["val"])
            half = pd.Index(y.index.year.astype(str) + np.where(y.index.month <= 6, "H1", "H2"))
            if "e" in ev:
                half = pd.Index(np.where(y.index >= ev["e"], "closing auction (from e)", half))
            for h, g in y.groupby(half.to_numpy()):
                s = level_summary(g["val"].to_numpy(), n_boot=n_boot, seed=seed_of("half", k, m, h))
                rows.append({"index": k, "measure": m, "half_year": h, "weekday": "/".join(sorted(set(g["weekday"]))),
                             "own_expiry": not (k == "BN" and g.index.min() >= ev["c"]), **s})
    return pd.DataFrame(rows)


PAPER_WINDOWS = {
    # as stated in the paper
    "stated (Nov 2023 to Feb 2024 vs Mar to May 2024)": ("2023-11-01", "2024-02-29", "2024-03-01", "2024-05-31"),
    # the 64-day before window that docs/REPLICATION_NSE.md finds matches the paper's Table 1
    "inferred by REPLICATION_NSE.md (14 Nov 2022 to 10 Feb 2023 vs Mar to May 2024)":
        ("2022-11-14", "2023-02-10", "2024-03-01", "2024-05-31"),
}


def event_b_raw_vs_relative(daily, rel, flags, measures=("rs", "rv5")) -> pd.DataFrame:
    """The paper's windows: raw daily levels (what a before/after comparison of
    levels sees) next to the week-relative premium (what is specific to that weekday)."""
    rows = []
    for wname, (a0, a1, b0, b1) in PAPER_WINDOWS.items():
        rows += _raw_vs_relative_one(daily, rel, flags, measures, wname, pd.Timestamp(a0), pd.Timestamp(a1),
                                     pd.Timestamp(b0), pd.Timestamp(b1))
    return pd.DataFrame(rows)


def _raw_vs_relative_one(daily, rel, flags, measures, wname, pre0, pre1, post0, post1) -> list:
    rows = []
    for k in SYMBOLS:
        for m in measures:
            lv = np.log(daily[k][m].where(daily[k][m] > 0))
            for wd in ("Wed", "Thu", "all days"):
                sel = lv if wd == "all days" else lv[lv.index.weekday == WD.index(wd)]
                a = sel[(sel.index >= pre0) & (sel.index <= pre1)].dropna().to_numpy()
                b = sel[(sel.index >= post0) & (sel.index <= post1)].dropna().to_numpy()
                _, _, p_hac = hac_post(a, b, lags=5)
                row = {"window": wname, "index": k, "measure": m, "weekday": wd, "n_days_pre": len(a), "n_days_post": len(b),
                       "raw_mean_log_pre": a.mean(), "raw_mean_log_post": b.mean(), "raw_diff": b.mean() - a.mean(),
                       "raw_ratio": float(np.exp(b.mean() - a.mean())),
                       "raw_day_mwu_p": float(stats.mannwhitneyu(b, a).pvalue), "raw_day_hac_p": p_hac}
                if wd != "all days":
                    y = Series(rel, flags, (k, wd), None, "regular").values(m)
                    ya = y[(y.index >= pre0 - pd.Timedelta(days=2)) & (y.index <= pre1)].dropna().to_numpy()
                    yb = y[(y.index >= post0 - pd.Timedelta(days=4)) & (y.index <= post1)].dropna().to_numpy()
                    c = compare(ya, yb, n_boot=1999, seed=seed_of("braw", k, m, wd))
                    row.update({"rel_weeks_pre": c["n_pre"], "rel_weeks_post": c["n_post"], "rel_mean_pre": c["mean_pre"],
                                "rel_mean_post": c["mean_post"], "rel_diff": c["diff"], "rel_boot_lo": c["boot_lo"],
                                "rel_boot_hi": c["boot_hi"], "rel_hac_p": c["hac_p"], "rel_mwu_p": c["mwu_p"]})
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------

# Categorical slots (validated reference palette): colour follows the weekday / regime.
WD_COL = {"Thu": "#2a78d6", "Wed": "#eb6834", "Tue": "#1baf7a"}
REG_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c23b8a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {"rv5": "whole-day realized variance (5-minute returns)",
         "rv_last30": "realized variance in the last 30 minutes (15:00 to 15:30, 1-minute returns)",
         "opt_premium": "option premium turnover (rupees)", "rs": "Rogers-Satchell variance (sum over 1-minute bars)"}


def _style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)


def fig_weekly_premium(rel, flags, events, path, measure=PRIMARY):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    panels = [("N", "Nifty expiry: Thu until d, Tue after"), ("BN", "Bank Nifty expiry: Thu until a, Wed until c, none after")]
    for ax, (k, note) in zip(axes, panels):
        _style(ax)
        for tgt, lab in (("Thu", "Thursday"), ("Wed", "Wednesday"), ("Tue", "Tuesday")):
            y = Series(rel, flags, (k, tgt), None, "clean_weekly").values(measure)
            sm = y.rolling(26, min_periods=10).mean()
            ax.plot(sm.index, sm.to_numpy(), color=WD_COL[tgt], lw=2, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        for _, e in events.iterrows():
            ax.axvline(e["event_week"], color=INK2, lw=0.8, ls=(0, (3, 3)))
            ax.text(e["event_week"], 1.01, f" {e['event']}", transform=ax.get_xaxis_transform(),
                    fontsize=9, color=INK2, va="bottom")
        ax.set_ylabel("log value: day minus\nrest of its week", color=INK2, fontsize=9)
        ax.set_title(f"{NAMES[k]} ({note})", fontsize=10.5, loc="left", color=INK, pad=14)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3)
    axes[-1].set_xlabel("week (26-week rolling mean; weeks with a monthly or holiday-shifted expiry excluded)", color=INK2, fontsize=9)
    fig.suptitle(f"Weekday premium in {SHORT.get(measure, measure)}, REAL DATA",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_did_c(rel, flags, events, path, measures=("rv5", "rv_last30", "opt_premium")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = events.set_index("event")
    e = ev.loc["c", "event_week"]
    lo, hi = ev.loc["a", "event_week"], ev.loc["d", "event_week"]
    fig, axes = plt.subplots(1, len(measures), figsize=(4.2 * len(measures), 4.4))
    for ax, m in zip(np.atleast_1d(axes), measures):
        _style(ax)
        for spec, lab, col in ((("BN", "Wed"), None, WD_COL["Wed"]), (("N", "Thu"), None, WD_COL["Thu"])):
            y = Series(rel, flags, spec, None, "clean_weekly").values(m)
            y = y[(y.index >= lo) & (y.index < hi)]
            rel_wk = ((y.index - e).days // 7)
            b = (np.floor(rel_wk / 8) * 8).astype(int)
            g = y.groupby(b).agg(["mean", "std", "count"])
            g = g[g["count"] >= 3]
            se = g["std"] / np.sqrt(g["count"])
            name = "Bank Nifty Wednesday (expiry until c)" if spec[0] == "BN" else "Nifty Thursday (expiry throughout)"
            off = 3.4 if spec[0] == "BN" else 4.6
            ax.errorbar(g.index + off, g["mean"], yerr=1.96 * se, color=col, marker="o", ms=5, lw=2, capsize=2,
                        elinewidth=1.2, label=name)
        ax.axvline(0, color=INK2, ls=(0, (3, 3)), lw=0.9)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("weeks relative to event c, 8-week bins", color=INK2, fontsize=9)
        ax.set_ylabel("log value: day minus rest of its week", color=INK2, fontsize=9)
        ax.set_title(SHORT.get(m, m).replace(" (", "\n("), fontsize=9.5, loc="left", color=INK)
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Event c (Bank Nifty weeklies stop, Nifty keeps them): bin means with 95% intervals, REAL DATA",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_diurnal(prof, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    regs = list(dict.fromkeys(prof["regime"]))
    for ax, k in zip(axes, SYMBOLS):
        _style(ax)
        p = prof[prof["index"] == k]
        for (reg, tgt), g in p.groupby(["regime", "target"], sort=False):
            col = REG_COL[regs.index(reg) % len(REG_COL)]
            ls = "-" if tgt.startswith("own") else (0, (4, 2))
            ax.plot(g["bucket"], g["mean_rel"], color=col, ls=ls, lw=2,
                    label=f"{reg}: {tgt} (n={int(g['n_weeks'].iloc[0])})")
        ax.axhline(0, color=INK2, lw=0.8)
        ticks = list(range(0, 25, 4))
        ax.set_xticks(ticks)
        ax.set_xticklabels([p["start"].iloc[j] for j in ticks] if len(p) else ticks)
        ax.set_title(f"{NAMES[k]}", fontsize=10.5, loc="left", color=INK)
        ax.set_xlabel("15-minute bucket start (IST)", color=INK2, fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    axes[0].set_ylabel("log variance in the bucket: expiry (or former\nexpiry) weekday minus rest of week", color=INK2, fontsize=9)
    fig.suptitle("Where in the day the expiry-weekday premium sits (solid: own expiry day, dashed: former expiry weekday), REAL DATA",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_placebo(panels, title, path):
    """panels: list of (subtitle, t_obs, t_placebo)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 3.9), sharey=True)
    for ax, (sub, t_obs, t_placebo) in zip(np.atleast_1d(axes), panels):
        _style(ax)
        tp = t_placebo[np.isfinite(t_placebo)]
        ax.hist(tp, bins=30, color="#2a78d6", edgecolor="white", linewidth=1)
        ax.axvline(t_obs, color="#eb6834", lw=2, label=f"actual event c: t = {t_obs:.2f}")
        ax.set_xlabel("Welch t (after minus before) at placebo dates", color=INK2, fontsize=9)
        ax.set_title(sub, fontsize=9.5, loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("placebo dates", color=INK2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Summary tables (markdown, every number from the CSVs written above)
# ---------------------------------------------------------------------------


def _ci(m, lo, hi, nd=2):
    if not np.isfinite(m):
        return ""
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def summary_markdown(events, changes, prem, et, did_reg, rob, braw, vol, half, cal) -> str:
    L = ["# Event study summary tables (REAL DATA)", "",
         "Generated by experiments/expiry_event_study.py. Values are week-relative log differences "
         "(expiry or former expiry weekday minus the mean of the other days of the same week) unless stated; "
         "brackets are 95% circular block bootstrap intervals (blocks of 4 weeks). exp(value) is the ratio.", ""]
    L += ["## Events found in the bhavcopies", "", _md(events.assign(
        last_old_expiry=events["last_old_expiry"].dt.strftime("%Y-%m-%d"),
        first_new_expiry=events["first_new_expiry"].dt.strftime("%Y-%m-%d").fillna("none"),
        event_week=events["event_week"].dt.strftime("%Y-%m-%d"))), ""]
    # premium by regime
    tg = {("N", "R0 both Thu (2019-03 to a)"): ["own"], ("N", "R0v both Thu, bhavcopy-verified"): ["own"],
          ("N", "R1 BN Wed, N Thu (a to c)"): ["own"], ("N", "R2 BN no weekly, N Thu (c to d)"): ["own"],
          ("N", "R3 N Tue, BN no weekly (d to e)"): ["own", "Thu"],
          ("N", "R3+ after d, incl. closing-auction days"): ["own", "Thu"],
          ("N", "R4 closing auction only (after e)"): ["own", "Thu"],
          ("BN", "R0 both Thu (2019-03 to a)"): ["own"], ("BN", "R0v both Thu, bhavcopy-verified"): ["own"],
          ("BN", "R1 BN Wed, N Thu (a to c)"): ["own"], ("BN", "R2 BN no weekly, N Thu (c to d)"): ["Wed"],
          ("BN", "R3 N Tue, BN no weekly (d to e)"): ["Wed"]}
    rows = []
    for (k, reg), tgts in tg.items():
        for t in tgts:
            r = {"index": NAMES[k], "regime": reg, "day": "own expiry day" if t == "own" else f"former expiry weekday ({t})"}
            for m in ("rv5", "rs", "rv_first_hour", "rv_last_hour", "rv_last30", "opt_premium", "corr5_z"):
                q = prem[(prem["index"] == k) & (prem["regime"] == reg) & (prem["target"] == t) &
                         (prem["sample"] == "clean_weekly") & (prem["measure"] == m)]
                r[m] = (_ci(q["mean"].iloc[0], q["boot_lo"].iloc[0], q["boot_hi"].iloc[0]) +
                        f" (p {_p(q['hac_p'].iloc[0])}, n {int(q['n_weeks'].iloc[0])})") if len(q) else ""
            rows.append(r)
    L += ["## Expiry-day premium by regime (weeks without monthly or holiday-shifted expiries)", "",
          "HAC p for mean = 0, n = weeks.", "", _md(pd.DataFrame(rows)), ""]
    rows = []
    for (k, reg), tgts in tg.items():
        for t in tgts:
            r = {"index": NAMES[k], "regime": reg, "day": t}
            for m in ("rv1_over_rv5", "rs_over_rv5", "rv1"):
                q = prem[(prem["index"] == k) & (prem["regime"] == reg) & (prem["target"] == t) &
                         (prem["sample"] == "clean_weekly") & (prem["measure"] == m)]
                r[m] = (_ci(q["mean"].iloc[0], q["boot_lo"].iloc[0], q["boot_hi"].iloc[0]) +
                        f" (p {_p(q['hac_p'].iloc[0])})") if len(q) else ""
            rows.append(r)
    L += ["## Frequency diagnostics (post hoc): premium in the ratio of 1-minute or Rogers-Satchell to 5-minute variance",
          "", _md(pd.DataFrame(rows)), ""]
    # every weekday, last 30 minutes (post hoc view: which weekdays carry an end-of-day premium in each regime)
    regs = ["R0v both Thu, bhavcopy-verified", "R1 BN Wed, N Thu (a to c)", "R2 BN no weekly, N Thu (c to d)",
            "R3 N Tue, BN no weekly (d to e)"]
    rows = []
    for k in SYMBOLS:
        for reg in regs:
            r = {"index": NAMES[k], "regime": reg}
            for t in ("Mon", "Tue", "Wed", "Thu", "Fri"):
                q = prem[(prem["index"] == k) & (prem["regime"] == reg) & (prem["target"] == t) &
                         (prem["sample"] == "clean_weekly") & (prem["measure"] == "rv_last30")]
                r[t] = (_ci(q["mean"].iloc[0], q["boot_lo"].iloc[0], q["boot_hi"].iloc[0]) +
                        f" (p {_p(q['hac_p'].iloc[0])}, n {int(q['n_weeks'].iloc[0])})") if len(q) else ""
            rows.append(r)
    L += ["## Every weekday, last 30 minutes (post hoc view)", "",
          "Week-relative log premium of rv_last30 for each weekday; HAC p for mean = 0, n = weeks. "
          "Bank Nifty bars end on 22 Apr 2026, so its R3 row stops there.", "", _md(pd.DataFrame(rows)), ""]

    def evt(ids, measures, windows=("3 months", "6 months", "12 months")):
        rows = []
        for sid in ids:
            for m in measures:
                for w in windows:
                    q = et[(et["spec_id"] == sid) & (et["measure"] == m) & (et["window"] == w)]
                    if not len(q):
                        continue
                    q = q.iloc[0]
                    rows.append({"comparison": q["comparison"], "measure": m, "window": w,
                                 "weeks pre/post": f"{int(q['n_pre'])}/{int(q['n_post'])}",
                                 "before": f"{q['mean_pre']:+.2f}" if np.isfinite(q["mean_pre"]) else "",
                                 "after": f"{q['mean_post']:+.2f}" if np.isfinite(q["mean_post"]) else "",
                                 "change [95% CI]": _ci(q["diff"], q["boot_lo"], q["boot_hi"]),
                                 "HAC p": _p(q["hac_p"]), "MWU p": _p(q["mwu_p"]), "placebo p": _p(q["ri_p"]),
                                 "MDE80": f"{q['mde80']:.2f}" if np.isfinite(q["mde80"]) else "",
                                 "pre-trend p": _p(q["pretrend_p"])})
        return _md(pd.DataFrame(rows))

    L += ["## Event c: Bank Nifty loses its weekly expiry, Nifty keeps it", "",
          "MDE80 = 2.8 x HAC standard error (smallest change detectable with 80% power at 5%). "
          "Placebo p: share of placebo event dates (no rule change inside the window) with a larger |t|; "
          "n/a when fewer than 3 non-overlapping placebo windows exist.", "",
          evt(["c_DiD_main", "c_BN_Wed", "c_N_Thu", "c_DiD_sameday", "c_DiD_Fri"], ["rv5"]), "",
          evt(["c_DiD_main", "c_BN_Wed", "c_N_Thu"], ["rs", "rv_last_hour", "rv_last30", "opt_premium", "corr5_z"],
              ("6 months", "12 months")), ""]
    q = did_reg[did_reg["spec_id"].isin(["c_DiD_main", "c_DiD_sameday"])]
    q = q[q["measure"].isin(["rv5", "rs", "rv_last30", "opt_premium"])]
    L += ["### Day-level DiD regression with index-by-week fixed effects (SE clustered by week)", "",
          _md(q.assign(did=q["did"].map(lambda x: f"{x:+.3f}"), se=q["se"].map(lambda x: f"{x:.3f}"),
                       p=q["p"].map(_p))[["spec_id", "measure", "window", "did", "se", "p", "n_days", "n_weeks"]]), ""]
    r = rob[rob["measure"].isin(["rv5", "rv_last30"])]
    L += ["### Robustness of event c", "",
          _md(pd.DataFrame({"measure": r["measure"], "series": r["series"], "variant": r["variant"], "window": r["window"],
                            "weeks": r["n_pre"].astype(int).astype(str) + "/" + r["n_post"].astype(int).astype(str),
                            "change [95% CI]": [_ci(a, b, c) for a, b, c in zip(r["diff"], r["boot_lo"], r["boot_hi"])],
                            "HAC p": r["hac_p"].map(_p), "MWU p": r["mwu_p"].map(_p)})), ""]
    L += ["## Event a: Bank Nifty weekly expiry moves Thursday to Wednesday", "",
          evt(["a_BN_own", "a_BN_Wed", "a_BN_Thu", "a_N_Thu", "a_N_Wed", "a_DiD_Wed", "a_BN_switch", "a_N_Fri", "a_BN_Fri"],
              ["rv5"], ("6 months", "12 months")), "",
          evt(["a_BN_Wed", "a_BN_Thu", "a_N_Wed", "a_DiD_Wed"], ["rs", "rv_last30", "opt_premium", "corr5_z"], ("6 months",)), ""]
    L += ["## Event b: Bank Nifty monthly expiry moves to Wednesday (the paper's event)", "",
          evt(["b_BN_Wed_m", "b_BN_Thu_m", "b_N_Thu_m", "b_N_Wed_m", "b_N_Thu_all", "b_N_Wed_all", "b_BN_Wed_all",
               "b_BN_Thu_all"], ["rv5", "rs"], ("6 months", "paper window")), "",
          "### The paper's window: raw daily levels versus the week-relative premium", "",
          _md(pd.DataFrame({"window": braw["window"].str.split(" \\(").str[0], "index": braw["index"].map(NAMES),
                            "measure": braw["measure"], "day": braw["weekday"],
                            "days pre/post": braw["n_days_pre"].astype(str) + "/" + braw["n_days_post"].astype(str),
                            "raw change in mean log": braw["raw_diff"].map(lambda x: f"{x:+.2f}"),
                            "raw ratio": braw["raw_ratio"].map(lambda x: f"{x:.2f}"),
                            "raw day-level MWU p": braw["raw_day_mwu_p"].map(_p),
                            "week-relative change [95% CI]": [_ci(a, b, c) for a, b, c in
                                                             zip(braw.get("rel_diff", np.nan), braw.get("rel_boot_lo", np.nan),
                                                                 braw.get("rel_boot_hi", np.nan))],
                            "week-relative HAC p": braw.get("rel_hac_p", pd.Series(np.nan, index=braw.index)).map(_p),
                            "week-relative MWU p": braw.get("rel_mwu_p", pd.Series(np.nan, index=braw.index)).map(_p)})), "",
          "Windows: " + "; ".join(PAPER_WINDOWS), ""]
    L += ["## Event d: Nifty weekly expiry moves Thursday to Tuesday", "",
          evt(["d_N_Thu", "d_N_Tue", "d_N_switch", "d_BN_Thu", "d_BN_Tue", "d_DiD_Thu", "d_N_Fri"], ["rv5"],
              ("6 months", "12 months")), "",
          evt(["d_N_Thu", "d_N_Tue", "d_N_switch", "d_DiD_Thu"], ["rs", "rv_last_hour", "rv_last30", "opt_premium"],
              ("6 months", "12 months")), "",
          "### Event d with the closing-auction days (event e onward) kept in the after window", "",
          evt(["d_N_Tue_incl_auction", "d_N_Thu_incl_auction", "d_N_switch_incl_auction"],
              ["rv5", "rs", "rv_last_hour", "rv_last30"], ("12 months",)), ""]
    h = half[half["measure"].isin(["rv5", "rv_last30", "opt_premium"])]
    L += ["## Expiry-weekday premium by half-year", "",
          "Nifty: Thursday, Tuesday after d. Bank Nifty: Thursday before a, Wednesday after (a former expiry weekday after c).",
          "", _md(h.assign(value=[_ci(a, b, c) for a, b, c in zip(h["mean"], h["boot_lo"], h["boot_hi"])])
                  .pivot_table(index="half_year", columns=["index", "measure"], values="value", aggfunc="first")
                  .reset_index().set_axis(["half year"] + [f"{NAMES[a]} {b}" for a, b in
                                                            h.pivot_table(index="half_year", columns=["index", "measure"],
                                                                          values="mean").columns], axis=1).fillna("")), ""]
    L += ["## Calibration: how often do tests reject for the negative-control weekday (Friday)?", "",
          "Friday was not targeted by any rule change. Shares of rows (all measures of that group; measures from the "
          "same days are strongly correlated, so these are not independent tests).", "",
          _md(cal.round(3).fillna("")), ""]
    L += ["## Option premium turnover (Rs crore), medians per day", "",
          _md(vol.assign(index=vol["index"].map(NAMES)).round(3)), ""]
    _ = changes
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------


def fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "event_study"))
    ap.add_argument("--boot", type=int, default=4999)
    a = ap.parse_args()
    t0 = time.time()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    log_design("Primary outcome fixed before any result was seen: whole-day realized variance from 5-minute returns "
               "(rv5), 6-month windows, the DiD at event c. Everything else is secondary.")
    log_design("Expiry dates come from the bhavcopies (an expiry happened on d if options expiring on d trade on d; "
               "monthly if an index future also expires on d). Before the first bhavcopy (20 Oct 2022) a Thursday rule "
               "with holiday roll-back is used; it reproduces every bhavcopy expiry date on the overlap (see stdout).")
    log_design("Changed: the rule first labelled the last expiry *date* of a month as monthly, which mislabels a "
               "1st-of-month Thursday holiday rolled back into the previous month (found by the overlap check, e.g. "
               "30 Apr 2025). The rule now uses the last *scheduled* Thursday of the month.")
    log_design("Long sample starts 4 Mar 2019 because Nifty weekly options are generally reported to start in "
               "February 2019 (not verifiable from our files). Pre-Oct-2022 data are used only for event a's "
               "12-month pre-window (Sep to Oct 2022), placebo dates and the regime R0 description.")
    log_design("Weeks with a monthly expiry of either index are excluded from the main samples: event b changes "
               "only monthly weeks (analysed separately), and after event c Bank Nifty still has a monthly expiry, "
               "which would put a Bank Nifty expiry inside the 'no weekly' weeks. Weeks where any expiry was moved "
               "by a holiday are also excluded (target weekday is then a holiday or a double expiry day).")
    log_design("Volume is measured as turnover in rupees (premium and notional), not contracts, because lot sizes "
               "changed several times (read from the UDiFF files; inferred for the legacy layout).")
    log_design("Changed: the moving-block bootstrap under-weights the first and last weeks of a window; with 15 to "
               "20 weeks per side this visibly shifted intervals (e.g. a difference of -0.01 had an interval of "
               "-0.26 to 0.67 in a trial run). Replaced by the circular block bootstrap (blocks of 4 weeks).")
    log_design("Changed: with very few weeks (event b's monthly weeks: 3 to 7 per side) a 4-week block equals the "
               "whole sample, so every circular draw had the same mean and the interval collapsed to a point. The "
               "block length is now capped at a quarter of the sample (iid resampling for 7 weeks or fewer).")
    log_design("Changed: randomization inference over placebo dates uses the Welch t statistic (studentized, so "
               "periods with different noise are comparable) and is reported only when the placebo windows contain "
               "at least 3 non-overlapping windows. With 12-month windows only about 2 fit in the 2019 to 2023 "
               "quiet period and the placebo t statistics had a standard deviation near 0.6 (too narrow), which made "
               "p-values look far too small.")
    log_design("Friday was planned as a negative-control weekday because no NSE rule change in this study targets "
               "it. It turned out not to be clean: Friday premia moved at events a and d about as much as treated "
               "weekdays, and BSE's Sensex weekly options are reported to have expired on Fridays in 2023 and 2024 "
               "(BSE files are not in this repository, so this is unverified). The calibration table reports how "
               "often each test rejects for Friday; we use it to judge which tests are too liberal.")
    log_design("Added (data fact from the coordinator, checked in the bars): from 3 Aug 2026 NSE sets the official "
               "close in a closing auction, so the Nifty minute index is frozen for about 13 minutes before 15:28 to "
               "15:29 and then jumps. The script detects this (every day from then on has >= 8 flat bars in the last "
               "20 minutes) and treats it as event e: regime R3 and event d's windows stop at e, and the "
               "auction days are reported separately (R4) and in 'kept' variants. Bank Nifty bars end in April 2026, "
               "before e.")
    log_design("The daily return correlation is one number per day, so it is not used in DiDs that compare the "
               "same weekday of both indices (it would cancel exactly).")
    log_design("Changed: a table splitting the extra variance into first-hour and last-hour shares was dropped "
               "because the whole-day excess is close to zero, so the shares were ratios of noise. Replaced by "
               "log premia per part of the day (first hour, 10:15 to 14:30, last hour, last 30 minutes).")
    log_design("Added after seeing results (post hoc, labelled as such): the frequency diagnostics rv1/rv5 and "
               "RS/rv5, because the per-minute Rogers-Satchell premium differed from the 5-minute RV premium.")
    D = build_data(raw)
    daily, expiries, changes = D["daily"], D["expiries"], D["changes"]
    lines = ["REAL DATA. Expiry-day event study, Nifty 50 and Bank Nifty.", ""] + D["lines"]

    events = define_events(changes, D["auction_start"])
    flags = week_flags(expiries, daily)
    rel = build_relative(daily, expiries)

    # Save the data products
    changes.to_csv(out / "schedule_changes_detected.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    expiries.to_csv(out / "expiry_calendar.csv", index=False)
    D["lots"].to_csv(out / "lot_sizes.csv", index=False)
    gz = {"method": "gzip", "mtime": 0}  # byte-identical reruns
    pd.concat({NAMES[k]: v for k, v in daily.items()}, names=["index_name", "date"]).to_csv(
        out / "daily_measures.csv.gz", compression=gz)
    rel.to_csv(out / "weekly_relative.csv.gz", index=False, compression=gz)
    flags.to_csv(out / "week_flags.csv")

    ev = events.set_index("event")["event_week"]
    blocked = list(ev.to_numpy())
    data_end_bn = daily["BN"].index.max()
    lines += ["", "== Schedule changes detected from the bhavcopies", changes.to_string(index=False),
              "", "== Events used (event_week = Monday of the first treated week)", events.to_string(index=False)]

    # Regimes
    regimes = [
        ("R0 both Thu (2019-03 to a)", LONG_START, ev["a"]),
        ("R0v both Thu, bhavcopy-verified", pd.Timestamp("2022-10-17"), ev["a"]),
        ("R1 BN Wed, N Thu (a to c)", ev["a"], ev["c"]),
        ("R2 BN no weekly, N Thu (c to d)", ev["c"], ev["d"]),
        ("R3 N Tue, BN no weekly (d to e)", ev["d"], ev.get("e", pd.Timestamp("2100-01-01"))),
    ]
    if "e" in ev:
        regimes += [("R3+ after d, incl. closing-auction days", ev["d"], pd.Timestamp("2100-01-01")),
                    ("R4 closing auction only (after e)", ev["e"], pd.Timestamp("2100-01-01"))]
    prem = premium_by_regime(rel, flags, regimes, a.boot)
    prem.to_csv(out / "premium_by_regime.csv", index=False)

    # Event tests
    windows = [("3 months", 13), ("6 months", 26), ("12 months", 52), ("paper window", None)]
    a_, b_, c_, d_ = ev["a"], ev["b"], ev["c"], ev["d"]
    e_ = ev.get("e", None)
    specs = [
        # (a) weekly separation; monthly weeks excluded so (b) is irrelevant; post bounded by (c)
        {"id": "a_BN_own", "event": "a", "label": "BN own expiry day (Thu then Wed)", "a": ("BN", "own"), "hi": c_},
        {"id": "a_BN_Wed", "event": "a", "label": "BN Wednesday", "a": ("BN", "Wed"), "hi": c_},
        {"id": "a_BN_Thu", "event": "a", "label": "BN Thursday", "a": ("BN", "Thu"), "hi": c_},
        {"id": "a_N_Thu", "event": "a", "label": "Nifty Thursday (own expiry)", "a": ("N", "Thu"), "hi": c_},
        {"id": "a_N_Wed", "event": "a", "label": "Nifty Wednesday (BN expiry after a)", "a": ("N", "Wed"), "hi": c_},
        {"id": "a_DiD_Wed", "event": "a", "label": "DiD: BN Wed minus Nifty Wed", "a": ("BN", "Wed"), "b": ("N", "Wed"), "hi": c_},
        {"id": "a_BN_switch", "event": "a", "label": "BN Wed minus BN Thu (does the premium follow the expiry?)",
         "a": ("BN", "Wed"), "b": ("BN", "Thu"), "hi": c_},
        {"id": "a_N_Fri", "event": "a", "label": "Negative control: Nifty Friday (no NSE change)", "a": ("N", "Fri"), "hi": c_},
        {"id": "a_BN_Fri", "event": "a", "label": "Negative control: BN Friday (no NSE change)", "a": ("BN", "Fri"), "hi": c_},
        # (b) monthly move: only monthly-expiry weeks are treated
        {"id": "b_BN_Wed_m", "event": "b", "label": "BN Wednesday, monthly weeks", "a": ("BN", "Wed"), "sample": "clean_monthly", "lo": a_, "hi": c_},
        {"id": "b_BN_Thu_m", "event": "b", "label": "BN Thursday, monthly weeks", "a": ("BN", "Thu"), "sample": "clean_monthly", "lo": a_, "hi": c_},
        {"id": "b_N_Thu_m", "event": "b", "label": "Nifty Thursday, monthly weeks", "a": ("N", "Thu"), "sample": "clean_monthly", "lo": a_, "hi": c_},
        {"id": "b_N_Wed_m", "event": "b", "label": "Nifty Wednesday, monthly weeks", "a": ("N", "Wed"), "sample": "clean_monthly", "lo": a_, "hi": c_},
        {"id": "b_N_Thu_all", "event": "b", "label": "Nifty Thursday, all regular weeks", "a": ("N", "Thu"), "sample": "regular", "lo": a_, "hi": c_},
        {"id": "b_N_Wed_all", "event": "b", "label": "Nifty Wednesday, all regular weeks", "a": ("N", "Wed"), "sample": "regular", "lo": a_, "hi": c_},
        {"id": "b_BN_Wed_all", "event": "b", "label": "BN Wednesday, all regular weeks", "a": ("BN", "Wed"), "sample": "regular", "lo": a_, "hi": c_},
        {"id": "b_BN_Thu_all", "event": "b", "label": "BN Thursday, all regular weeks", "a": ("BN", "Thu"), "sample": "regular", "lo": a_, "hi": c_},
        # (c) BN weekly removed; Nifty keeps weekly. Pre bounded by (a), post by (d)
        {"id": "c_BN_Wed", "event": "c", "label": "BN Wednesday (expiry until c)", "a": ("BN", "Wed"), "lo": a_, "hi": d_},
        {"id": "c_N_Thu", "event": "c", "label": "Nifty Thursday (expiry throughout)", "a": ("N", "Thu"), "lo": a_, "hi": d_},
        {"id": "c_DiD_main", "event": "c", "label": "DiD main: BN Wed minus Nifty Thu", "a": ("BN", "Wed"), "b": ("N", "Thu"), "lo": a_, "hi": d_},
        {"id": "c_DiD_sameday", "event": "c", "label": "DiD same weekday: BN Wed minus Nifty Wed", "a": ("BN", "Wed"), "b": ("N", "Wed"), "lo": a_, "hi": d_},
        {"id": "c_N_Wed", "event": "c", "label": "Nifty Wednesday (BN expiry until c)", "a": ("N", "Wed"), "lo": a_, "hi": d_},
        {"id": "c_BN_Thu", "event": "c", "label": "BN Thursday (Nifty expiry day)", "a": ("BN", "Thu"), "lo": a_, "hi": d_},
        {"id": "c_DiD_Fri", "event": "c", "label": "Negative control: BN Fri minus Nifty Fri (no NSE change)", "a": ("BN", "Fri"), "b": ("N", "Fri"), "lo": a_, "hi": d_},
        # (d) Nifty Thu -> Tue. Pre bounded by (c)
        {"id": "d_N_Thu", "event": "d", "label": "Nifty Thursday (former expiry)", "a": ("N", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_N_Tue", "event": "d", "label": "Nifty Tuesday (new expiry)", "a": ("N", "Tue"), "lo": c_, "hi": e_},
        {"id": "d_N_switch", "event": "d", "label": "Nifty Tue minus Nifty Thu (does the premium follow the expiry?)",
         "a": ("N", "Tue"), "b": ("N", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_BN_Thu", "event": "d", "label": "BN Thursday (Nifty expiry until d)", "a": ("BN", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_BN_Tue", "event": "d", "label": "BN Tuesday (Nifty expiry after d)", "a": ("BN", "Tue"), "lo": c_, "hi": e_},
        {"id": "d_DiD_Thu", "event": "d", "label": "DiD: Nifty Thu minus BN Thu", "a": ("N", "Thu"), "b": ("BN", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_N_Fri", "event": "d", "label": "Negative control: Nifty Friday (no NSE change)", "a": ("N", "Fri"), "lo": c_, "hi": e_},
        # (d) again, keeping the closing-auction days (from event e) in the post window
        {"id": "d_N_Tue_incl_auction", "event": "d", "label": "Nifty Tuesday (new expiry), closing-auction days kept",
         "a": ("N", "Tue"), "lo": c_},
        {"id": "d_N_Thu_incl_auction", "event": "d", "label": "Nifty Thursday (former expiry), closing-auction days kept",
         "a": ("N", "Thu"), "lo": c_},
        {"id": "d_N_switch_incl_auction", "event": "d", "label": "Nifty Tue minus Nifty Thu, closing-auction days kept",
         "a": ("N", "Tue"), "b": ("N", "Thu"), "lo": c_},
    ]
    et = run_event_tests(rel, flags, events, specs, windows, blocked, a.boot)
    cal = []
    et_neg = et["spec_id"].str.contains("_Fri")
    vol_m = et["measure"].isin(["opt_premium", "opt_notional", "fut_notional"])
    for wname in ("3 months", "6 months", "12 months"):
        for grp, msk in (("negative control (Friday)", et_neg), ("treated or spillover weekday", ~et_neg)):
            for mg, mm in (("minute-bar measures", ~vol_m), ("turnover measures", vol_m)):
                q = et[(et["window"] == wname) & msk & mm]
                if not len(q):
                    continue
                cal.append({"window": wname, "group": grp, "measures": mg, "rows": len(q),
                            "share_hac_p_below_0.05": float((q["hac_p"] < 0.05).mean()),
                            "share_mwu_p_below_0.05": float((q["mwu_p"] < 0.05).mean()),
                            "share_boot_ci_excludes_0": float(((q["boot_lo"] > 0) | (q["boot_hi"] < 0)).mean()),
                            "rows_with_placebo_p": int(q["ri_p"].notna().sum()),
                            "share_placebo_p_below_0.05": float((q["ri_p"] < 0.05).sum() / max(q["ri_p"].notna().sum(), 1))
                            if q["ri_p"].notna().any() else np.nan})
    cal = pd.DataFrame(cal)
    cal.to_csv(out / "negative_control_calibration.csv", index=False)
    et["bh_q_within_window"] = np.nan
    for wname, g in et.groupby("window"):
        et.loc[g.index, "bh_q_within_window"] = bh_qvalues(g["hac_p"].to_numpy())
    et.to_csv(out / "event_tests.csv", index=False)

    # Day-level DiD regressions with index-by-week fixed effects (event c and a)
    did_rows = []
    for spec_id, A, B, tA, tB, lo, hi, e in (
        ("c_DiD_main", "BN", "N", "Wed", "Thu", a_, d_, c_),
        ("c_DiD_sameday", "BN", "N", "Wed", "Wed", a_, d_, c_),
        ("a_DiD_Wed", "BN", "N", "Wed", "Wed", None, c_, a_),
        ("d_DiD_Thu", "N", "BN", "Thu", "Thu", c_, e_, d_),
    ):
        for m in LOG_MEASURES:
            for wname, W in (("3 months", 13), ("6 months", 26), ("12 months", 52)):
                p0 = e - pd.Timedelta(weeks=W)
                p1 = e + pd.Timedelta(weeks=W)
                if lo is not None:
                    p0 = max(p0, lo)
                if hi is not None:
                    p1 = min(p1, hi)
                cw = flags.index[flags["clean_weekly"]]
                wpre = cw[(cw >= p0) & (cw < e)]
                wpost = cw[(cw >= e) & (cw < p1)]
                logs = {A: np.log(daily[A][m].where(daily[A][m] > 0)), B: np.log(daily[B][m].where(daily[B][m] > 0))}
                if A == B:
                    continue
                tg = {A: daily[A].index[daily[A].index.weekday == WD.index(tA)],
                      B: daily[B].index[daily[B].index.weekday == WD.index(tB)]}
                r = did_day_regression(logs, tg, wpre, wpost)
                did_rows.append({"spec_id": spec_id, "measure": m, "window": wname, **r})
    did_reg = pd.DataFrame(did_rows)
    did_reg.to_csv(out / "did_day_regressions.csv", index=False)

    # Diurnal profiles and where the excess variance sits
    dregs = [
        ("R0 (2019-03 to a)", LONG_START, a_),
        ("R1 (a to c)", a_, c_),
        ("R2 (c to d)", c_, d_),
        ("R3 (d to e)", d_, e_ if e_ is not None else pd.Timestamp("2100-01-01")),
        ("R4 (closing auction, after e)", e_ if e_ is not None else pd.Timestamp("2100-01-01"), pd.Timestamp("2100-01-01")),
    ]
    tbr = {
        ("N", "R0 (2019-03 to a)"): [("own Thu", "Thu")], ("N", "R1 (a to c)"): [("own Thu", "Thu")],
        ("N", "R2 (c to d)"): [("own Thu", "Thu")], ("N", "R3 (d to e)"): [("own Tue", "Tue"), ("former Thu", "Thu")],
        ("N", "R4 (closing auction, after e)"): [("own Tue", "Tue")], ("BN", "R4 (closing auction, after e)"): [],
        ("BN", "R0 (2019-03 to a)"): [("own Thu", "Thu")], ("BN", "R1 (a to c)"): [("own Wed", "Wed")],
        ("BN", "R2 (c to d)"): [("former Wed", "Wed")], ("BN", "R3 (d to e)"): [("former Wed", "Wed")],
    }
    prof = diurnal_profiles(D["buckets"], expiries, flags, dregs, tbr)
    prof.to_csv(out / "diurnal_profiles.csv", index=False)
    braw = event_b_raw_vs_relative(daily, rel, flags)
    braw.to_csv(out / "event_b_paper_window_raw_vs_relative.csv", index=False)
    rob = robustness_event_c(daily, expiries, flags, events, a.boot)
    rob.to_csv(out / "robustness_event_c.csv", index=False)
    half = own_premium_by_halfyear(rel, flags, events, a.boot)
    half.to_csv(out / "own_premium_by_halfyear.csv", index=False)

    # Volume descriptives: share of option premium turnover in the expiring contract on expiry days
    vol_rows = []
    for k in SYMBOLS:
        d = daily[k]
        own = set(expiries.loc[expiries["symbol"] == SYMBOLS[k], "expiry"])
        for rname, lo, hi in regimes[1:]:
            if rname.startswith("R3+"):
                continue
            sel = d[(d.index >= lo) & (d.index < hi) & d["opt_premium"].notna()]
            is_e = sel.index.isin(list(own))
            vol_rows.append({"index": k, "regime": rname, "days": len(sel), "expiry_days": int(is_e.sum()),
                             "median_opt_premium_cr_expiry": sel.loc[is_e, "opt_premium"].median() / 1e7,
                             "median_opt_premium_cr_other": sel.loc[~is_e, "opt_premium"].median() / 1e7,
                             "median_share_0dte_on_expiry": sel.loc[is_e, "share_premium_0dte"].median()})
    vol = pd.DataFrame(vol_rows)
    vol.to_csv(out / "volume_descriptives.csv", index=False)
    (out / "summary_tables.md").write_text(summary_markdown(events, changes, prem, et, did_reg, rob, braw, vol, half, cal))

    # Figures
    fig_weekly_premium(rel, flags, events, out / "fig1_weekday_premium_timeline_rv5.png", PRIMARY)
    fig_weekly_premium(rel, flags, events, out / "fig1b_weekday_premium_timeline_last30.png", "rv_last30")
    fig_weekly_premium(rel, flags, events, out / "fig1c_weekday_premium_timeline_premium_turnover.png", "opt_premium")
    fig_did_c(rel, flags, events, out / "fig2_event_c_did.png")
    fig_diurnal(prof, out / "fig3_diurnal_premium.png")
    panels, pl_rows = [], []
    for m in (PRIMARY, "rv_last30"):
        main_row = et[(et["spec_id"] == "c_DiD_main") & (et["measure"] == m) & (et["window"] == "6 months")].iloc[0]
        ser = Series(rel, flags, ("BN", "Wed"), ("N", "Thu"), "clean_weekly").values(m)
        gw = ser.index
        ks = placebo_event_indices(len(gw), None, 26, [int(np.searchsorted(gw, b)) for b in blocked])
        tp = placebo_t_stats(ser.to_numpy(), ks, 26)
        pl_rows.append(pd.DataFrame({"measure": m, "placebo_week": gw[ks], "welch_t": tp}))
        panels.append((SHORT[m], main_row["welch_t"], tp))
    pd.concat(pl_rows).to_csv(out / "placebo_c_did_main_6m.csv", index=False)
    fig_placebo(panels, "Event c, DiD (Bank Nifty Wednesday minus Nifty Thursday), 6-month windows: placebo dates "
                "with no rule change inside the window, REAL DATA", out / "fig4_placebo_c.png")

    # Text summary
    key = et[(et["measure"] == PRIMARY) & (et["window"].isin(["6 months"]))]
    cols = ["spec_id", "n_pre", "n_post", "mean_pre", "mean_post", "diff", "boot_lo", "boot_hi", "hac_p", "mwu_p", "ri_p",
            "ri_n_placebo", "pretrend_p"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        lines += ["", "== Premium by regime (primary measure, own/former expiry weekday, clean weekly weeks)",
                  prem[(prem["measure"] == PRIMARY) & (prem["sample"] == "clean_weekly")][
                      ["index", "target", "regime", "n_weeks", "mean", "boot_lo", "boot_hi", "hac_p", "ratio"]].to_string(index=False),
                  "", "== Event tests, primary measure, 6-month windows", key[cols].to_string(index=False),
                  "", "== Day-level DiD regressions (index-by-week FE, SE clustered by week), primary measure",
                  did_reg[did_reg["measure"] == PRIMARY].to_string(index=False),
                  "", "== Robustness, event c, primary measure",
                  rob[rob["measure"] == PRIMARY][["series", "variant", "window", "n_pre", "n_post", "diff", "boot_lo",
                                                  "boot_hi", "hac_p", "mwu_p"]].to_string(index=False),
                  "", "== Negative-control calibration (share of rows rejecting at 5%)", cal.round(3).to_string(index=False),
                  "", "== Volume descriptives", pd.DataFrame(vol_rows).to_string(index=False)]
    lines += ["", "== Design log"] + [f"- {s}" for s in DESIGN_LOG]
    lines += ["", f"Total run time {time.time() - t0:.0f}s."]
    text = "\n".join(lines)
    (out.parent / "event_study_stdout.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
