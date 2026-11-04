"""Referee response on the expiry evidence (comments M8, M9, M10). POST HOC. REAL DATA.

Everything in this script was written after the referee report of 3 November
2026 and after every result it revisits had been seen. It is a set of checks
requested by the referee, not a new pre-specified test. Every number it
writes comes from NSE files (bhavcopies, index files), third-party minute bars
(validated in earlier studies) or reported BSE calendars (not verified: BSE's
servers are blocked from this machine). The only SYNTHETIC data are the
simulated noisy prices used to check the noise-robust estimators (Section 4),
and they are labelled as such.

What it does
------------
1. M9, clean baseline. Builds a day-level calendar of every index-option
   expiry on NSE and BSE. NSE (Nifty, Bank Nifty, FinNifty, Midcap Nifty,
   Nifty Next 50): from NSE's full F&O bhavcopies (the repository's own
   bhavcopies are filtered to Nifty and Bank Nifty, so the full files are
   fetched once from NSE's archive with --fetch-nse and reduced to
   data/external/nse_all_index_contracts_by_expiry.csv.gz; an expiry happened
   on d if option contracts expiring on d traded on d, and it is monthly if a
   future of the same index expired on d). BSE (Sensex, Bankex, Sensex 50):
   reported weekday regimes (data/external/sensex_expiry_calendar.csv and
   other_weekly_expiries.csv) turned into dates with the holiday roll-back
   rule; NOT verified. Counts the days with no expiry of any index per week
   and regime, and recomputes R2 (docs/REAL_DATA_RESULTS.md) and the event
   study's premia with the baseline restricted to no-expiry days. Also
   baseline-free two-day differences, a pooled regression with a dummy for
   every index's expiry, and a table of calendar changes with sources.
2. M8, placebo inference. For every row of results/event_study/event_tests.csv:
   the number of non-overlapping placebo windows next to the placebo p, a
   randomization p over non-overlapping windows only (its floor is
   1 / (windows + 1)), a prediction-interval t test against those windows
   (t distribution with windows - 1 degrees of freedom) and a wild cluster
   restricted bootstrap (clusters of 4 consecutive weeks, Webb weights; few
   clusters). The Friday negative control is recomputed without Fridays that
   were an expiry day of any index, and every event window is checked for a
   weekday free of every expiry.
3. M10, noise. End-of-day variance (last 30 minutes, last hour) from 1-minute
   returns, 2- and 5-minute subsampled returns (averaged over offsets), a
   Parzen realized kernel and a pre-averaging estimator (both implemented
   here and checked on simulated noisy prices), with the first-order
   autocorrelation of 1-minute returns. Days from 3 August 2026 (closing
   auction: index frozen about 15:16 to 15:28) are kept apart.
4. M8 bullets: comparison type, window and pre-trend p of each event's key
   rows, a trend-adjusted change, and event d's analogue of event a's
   headline comparison.

    OMP_NUM_THREADS=1 python experiments/referee_expiry.py --fetch-nse   # once, about 15 minutes
    OMP_NUM_THREADS=1 python experiments/referee_expiry.py               # about 10 minutes, deterministic

Writes results/referee_expiry/*.csv, summary.md and
results/referee_expiry_stdout.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NSE_EXTRACT = "nse_all_index_contracts_by_expiry.csv.gz"
NSE_EXTRACT_SOURCE = "nse_all_index_contracts_by_expiry_source.json"
NSE_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYIT", "NIFTYFPI"]  # every index with options in the files
BSE_SYMBOLS = ["SENSEX", "BANKEX", "SENSEX50"]
ALL_SYMBOLS = NSE_SYMBOLS + BSE_SYMBOLS
AUCTION = pd.Timestamp("2026-08-03")
EOD_MEASURES = ["rv_last_hour", "rv_last30"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


es = _load("expiry_event_study", ROOT / "experiments" / "expiry_event_study.py")
sc = _load("sensex_confound", ROOT / "experiments" / "sensex_confound.py")
rdl = _load("real_day_level", ROOT / "experiments" / "real_day_level.py")
WD = es.WD

# ---------------------------------------------------------------------------
# 0. Full NSE F&O bhavcopies (every index symbol), fetched from NSE's archive
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}


def _bhav_urls(d: pd.Timestamp) -> list[str]:
    arch = "https://nsearchives.nseindia.com"
    udiff = f"{arch}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    legacy = f"{arch}/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"
    return [udiff, legacy] if d >= pd.Timestamp("2024-07-08") else [legacy, udiff]


def index_rows_from_full_bhavcopy(raw: bytes) -> pd.DataFrame:
    """Every index option and index future row of one full F&O bhavcopy, aggregated per
    (date, symbol, instrument, expiry): number of series, volume traded (contracts in the
    legacy layout, NSE's TtlTradgVol in UDiFF), turnover (Rs) and open interest. Works for
    the legacy and the UDiFF layout."""
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    if "INSTRUMENT" in df.columns:
        d = df[df["INSTRUMENT"].str.strip().isin(["OPTIDX", "FUTIDX"])]
        out = pd.DataFrame({
            "date": pd.to_datetime(d["TIMESTAMP"].str.strip(), format="%d-%b-%Y"),
            "symbol": d["SYMBOL"].str.strip(),
            "inst": d["INSTRUMENT"].str.strip().map({"OPTIDX": "opt", "FUTIDX": "fut"}),
            "expiry": pd.to_datetime(d["EXPIRY_DT"].str.strip(), format="%d-%b-%Y"),
            "contracts": pd.to_numeric(d["CONTRACTS"], errors="coerce"),
            "turnover_rs": pd.to_numeric(d["VAL_INLAKH"], errors="coerce") * 1e5,
            "open_interest": pd.to_numeric(d["OPEN_INT"], errors="coerce"),
        })
        out["actual_expiry"] = out["expiry"]
    else:
        d = df[df["FinInstrmTp"].str.strip().isin(["IDO", "IDF"])]
        out = pd.DataFrame({
            "date": pd.to_datetime(d["TradDt"].str.strip()),
            "symbol": d["TckrSymb"].str.strip(),
            "inst": d["FinInstrmTp"].str.strip().map({"IDO": "opt", "IDF": "fut"}),
            "expiry": pd.to_datetime(d["XpryDt"].str.strip()),
            "contracts": pd.to_numeric(d["TtlTradgVol"], errors="coerce"),
            "turnover_rs": pd.to_numeric(d["TtlTrfVal"], errors="coerce"),
            "open_interest": pd.to_numeric(d["OpnIntrst"], errors="coerce"),
        })
        act = d["FininstrmActlXpryDt"].str.strip() if "FininstrmActlXpryDt" in d else d["XpryDt"].str.strip()
        out["actual_expiry"] = pd.to_datetime(act, errors="coerce")
    g = out.groupby(["date", "symbol", "inst", "expiry", "actual_expiry"], dropna=False)
    return g.agg(n_series=("contracts", "size"), contracts=("contracts", "sum"), turnover_rs=("turnover_rs", "sum"),
                 open_interest=("open_interest", "sum")).reset_index()


def _fetch_one(d: pd.Timestamp, pause: float = 0.3):
    import requests

    s = requests.Session()
    for url in _bhav_urls(d):
        for k in range(4):
            try:
                r = s.get(url, headers=_HEADERS, timeout=40)
            except requests.RequestException:
                time.sleep(2 ** (k + 1))
                continue
            if r.status_code == 404:
                break
            if r.status_code == 200 and r.content:
                time.sleep(pause)
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    raw = z.read(z.namelist()[0])
                rows = index_rows_from_full_bhavcopy(raw)
                meta = {"date": d.strftime("%Y-%m-%d"), "url": url, "zip_bytes": len(r.content),
                        "zip_sha256": hashlib.sha256(r.content).hexdigest(), "index_rows": int(len(rows))}
                return rows, meta
            time.sleep(2 ** (k + 1))
    time.sleep(pause)
    return None, {"date": d.strftime("%Y-%m-%d"), "url": None, "status": "not found (holiday or not published)"}


def fetch_nse_full(raw: Path, ext: Path, start: str, end: str, workers: int = 3) -> None:
    """Download the full F&O bhavcopy for every weekday with Nifty minute bars or a filtered
    bhavcopy, keep only index options and futures (all symbols), aggregate per expiry and
    write data/external/nse_all_index_contracts_by_expiry.csv.gz plus a source file with the
    URL, size and SHA-256 of every zip. The zips themselves are not stored."""
    bars = pd.read_csv(raw / "nifty_1min.csv", usecols=["timestamp"])
    days = set(pd.to_datetime(bars["timestamp"]).dt.normalize().unique())
    for sub in ("bhavcopy", "extra_2022/bhavcopy"):
        for f in (raw / sub).glob("fo_*.csv.gz"):
            days.add(pd.Timestamp(f.name[3:11]))
    days = sorted(d for d in days if pd.Timestamp(start) <= d <= pd.Timestamp(end) and d.weekday() < 5)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_fetch_one, days))
    frames = [r for r, _ in res if r is not None]
    metas = [m for _, m in res]
    out = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol", "inst", "expiry"])
    ext.mkdir(parents=True, exist_ok=True)
    out.to_csv(ext / NSE_EXTRACT, index=False, compression={"method": "gzip", "mtime": 0})
    src = {"what": "Index option and index future rows of NSE's full F&O bhavcopies, every index symbol, aggregated "
                   "per trade date, symbol, instrument and expiry. Built by experiments/referee_expiry.py --fetch-nse.",
           "source": "https://nsearchives.nseindia.com (legacy fo{DD}{MON}{YYYY}bhav.csv.zip and UDiFF "
                     "BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip)",
           "fetched_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
           "days_requested": len(days), "days_found": len(frames), "seconds": round(time.time() - t0),
           "files": metas}
    (ext / NSE_EXTRACT_SOURCE).write_text(json.dumps(src, indent=1))
    print(f"fetched {len(frames)} of {len(days)} days in {time.time() - t0:.0f}s; {len(out)} rows")


# ---------------------------------------------------------------------------
# 1. Calendar of every index-option expiry (NSE verified, BSE reported)
# ---------------------------------------------------------------------------


def nse_expiries_from_extract(ext: pd.DataFrame, symbols=NSE_SYMBOLS, traded_only: bool = True) -> pd.DataFrame:
    """Expiries per NSE symbol: option contracts expiring on d appear (and, by default, trade)
    in d's bhavcopy. kind = 'monthly' if a future of the same symbol expires on d and no later
    future of that symbol expires in the same calendar month (some indices listed weekly
    futures in 2021 to 2023). Only futures that stay listed up to their expiry count: when NSE
    moved monthly expiries (January 2025), the long-dated futures on the old dates were
    withdrawn and relisted on the new ones. Also returns the expiring options' turnover and
    contracts on d."""
    e = ext[ext["symbol"].isin(symbols)]
    opt = e[(e["inst"] == "opt") & (e["expiry"] == e["date"])]
    if traded_only:
        opt = opt[opt["contracts"] > 0]
    bdays = np.array(sorted(ext["date"].unique()), dtype="datetime64[ns]")
    fut = {}
    for sym, g in e[e["inst"] == "fut"].groupby("symbol"):
        life = g.groupby("expiry")["date"].max()
        exps = pd.DatetimeIndex(life.index)
        k = np.searchsorted(bdays, exps.values, side="right") - 1
        last_file_day = pd.DatetimeIndex(bdays[np.clip(k, 0, None)])
        ran = life.to_numpy() >= last_file_day.values
        fx = pd.Series(sorted(exps[ran]))
        fut[sym] = set(fx.groupby(fx.dt.to_period("M")).max()) if len(fx) else set()
    rows = []
    for (sym, d), g in opt.groupby(["symbol", "date"]):
        d = pd.Timestamp(d)
        rows.append({"symbol": sym, "expiry": d,
                     "weekday": WD[d.weekday()] if d.weekday() < 5 else "Sat/Sun",
                     "kind": "monthly" if d in fut.get(sym, set()) else "weekly",
                     "exchange": "NSE", "status": "verified (NSE bhavcopy)",
                     "expiring_option_contracts": float(g["contracts"].sum()),
                     "expiring_option_turnover_rs_cr": float(g["turnover_rs"].sum()) / 1e7})
    return pd.DataFrame(rows, columns=["symbol", "expiry", "weekday", "kind", "exchange", "status",
                                       "expiring_option_contracts", "expiring_option_turnover_rs_cr"])


def bse_expiries(full_days, sensex_cal: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """BSE expiry days from reported weekday regimes with the reported holiday roll-back rule.

    Sensex: the regimes of sensex_expiry_calendar.csv (as in experiments/sensex_confound.py).
    Bankex and Sensex 50: the BSE rows of other_weekly_expiries.csv. Bankex monthly contracts
    in the weeks after its weekly series stopped (Nov and Dec 2024) are an assumption (last
    Monday of the month); from January 2025 BSE's monthly contracts are reported to expire on
    the same weekday as Sensex, so they add no new days and are not listed."""
    rows = []
    sx = sc.sensex_rule_expiries(full_days, sc.load_regimes(sensex_cal))
    for _, r in sx.iterrows():
        rows.append({"symbol": "SENSEX", "expiry": r["expiry"], "weekday": r["weekday"], "kind": "weekly",
                     "exchange": "BSE", "status": "reported (weekday regime plus holiday rule; not verified)"})
    for _, o in other[other["exchange"] == "BSE"].iterrows():
        reg = [(pd.Timestamp(o["start"]), pd.Timestamp(o["end"]), WD.index(o["weekday"]))]
        for _, r in sc.sensex_rule_expiries(full_days, reg).iterrows():
            rows.append({"symbol": o["index"], "expiry": r["expiry"], "weekday": r["weekday"], "kind": "weekly",
                         "exchange": "BSE", "status": "reported (weekday regime plus holiday rule; not verified)"})
    td = pd.DatetimeIndex(sorted(full_days))
    for m in ("2024-11", "2024-12"):
        mon = td[(td.to_period("M") == pd.Period(m)) & (td.weekday == 0)]
        if len(mon):
            rows.append({"symbol": "BANKEX", "expiry": mon.max(), "weekday": "Mon", "kind": "monthly",
                         "exchange": "BSE", "status": "assumed (last Monday of the month; not sourced)"})
    out = pd.DataFrame(rows, columns=["symbol", "expiry", "weekday", "kind", "exchange", "status"])
    return out.drop_duplicates(["symbol", "expiry"]).sort_values(["symbol", "expiry"]).reset_index(drop=True)


def day_calendar(days, exp: pd.DataFrame, symbols=ALL_SYMBOLS) -> pd.DataFrame:
    """One row per trading day: a 0/1 column per index (1 = that index has an option expiry
    that day), the number of expiring indices, and whether any NSE / BSE / any index expires."""
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(days))))
    cal = pd.DataFrame(index=days)
    cal.index.name = "date"
    cal["weekday"] = [WD[d.weekday()] if d.weekday() < 5 else "Sat/Sun" for d in days]
    cal["week"] = es.week_start(days)
    for s in symbols:
        dset = set(exp.loc[exp["symbol"] == s, "expiry"])
        cal[s] = days.isin(list(dset)).astype(int)
    cal["n_expiring"] = cal[list(symbols)].sum(axis=1)
    nse = [s for s in symbols if s not in BSE_SYMBOLS]
    bse = [s for s in symbols if s in BSE_SYMBOLS]
    cal["any_nse"] = cal[nse].sum(axis=1) > 0
    cal["any_bse"] = (cal[bse].sum(axis=1) > 0) if bse else False
    cal["any_expiry"] = cal["n_expiring"] > 0
    cal["expiring"] = [";".join(s for s in symbols if row[s]) for _, row in cal[list(symbols)].iterrows()]
    return cal


def week_counts(cal: pd.DataFrame, full_days) -> pd.DataFrame:
    """Per calendar week: full trading days, days with any expiry, days with no expiry of any
    index, and the same counting only NSE expiries (the verified part of the calendar)."""
    c = cal[cal.index.isin(list(full_days))]
    g = c.groupby("week")
    out = pd.DataFrame({
        "trading_days": g.size(),
        "days_any_expiry": g["any_expiry"].sum().astype(int),
        "days_no_expiry": (~c["any_expiry"]).groupby(c["week"]).sum().astype(int),
        "days_no_nse_expiry": (~c["any_nse"]).groupby(c["week"]).sum().astype(int),
        "no_expiry_weekdays": c[~c["any_expiry"]].groupby("week")["weekday"].apply(lambda s: ";".join(s)),
    })
    out["no_expiry_weekdays"] = out["no_expiry_weekdays"].fillna("")
    return out


def label_regimes(weeks, cuts) -> np.ndarray:
    """cuts: list of (name, start, end) with end exclusive; weeks outside every cut get ''."""
    w = pd.DatetimeIndex(weeks)
    lab = np.array([""] * len(w), dtype=object)
    for name, lo, hi in cuts:
        lab[(w >= pd.Timestamp(lo)) & (w < pd.Timestamp(hi))] = name
    return lab


def summarise_weeks(wc: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    rows = []
    for reg, g in wc[wc[regime_col] != ""].groupby(regime_col, sort=False):
        rows.append({"regime": reg, "weeks": len(g), "first_week": g.index.min().date(), "last_week": g.index.max().date(),
                     "mean_trading_days": g["trading_days"].mean(),
                     "mean_no_expiry_days": g["days_no_expiry"].mean(),
                     "weeks_with_0_no_expiry_days": int((g["days_no_expiry"] == 0).sum()),
                     "weeks_with_1": int((g["days_no_expiry"] == 1).sum()),
                     "weeks_with_2_or_more": int((g["days_no_expiry"] >= 2).sum()),
                     "mean_no_nse_expiry_days": g["days_no_nse_expiry"].mean(),
                     "weeks_with_2_or_more_no_nse_expiry_days": int((g["days_no_nse_expiry"] >= 2).sum()),
                     "most_common_no_expiry_weekdays": g["no_expiry_weekdays"].replace("", "none").value_counts().index[0]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Clean-baseline R2 and week-relative helpers
# ---------------------------------------------------------------------------


def r2_week_relative(dm: pd.DataFrame, col: str, min_other: int = 2) -> pd.DataFrame:
    """real_day_level.week_relative with a choice of the minimum number of baseline days.
    With min_other = 2 it is identical to the published function."""
    d = dm.copy()
    d["lv"] = np.log(d[col].where(d[col] > 0))
    d["week"] = d.index.to_period("W-SUN")
    rows = []
    for wk, g in d.groupby("week"):
        e = g[g["is_expiry"]]
        o = g[~g["is_expiry"]]
        if len(e) != 1 or len(o) < min_other:
            continue
        rows.append({"week": str(wk), "date": e.index[0], "regime": e["regime"].iloc[0],
                     "weekday": int(e["weekday"].iloc[0]), "rel": float(e["lv"].iloc[0] - o["lv"].mean()),
                     "n_other": len(o)})
    return pd.DataFrame(rows, columns=["week", "date", "regime", "weekday", "rel", "n_other"])


def r2_summary(x: np.ndarray) -> dict:
    """R2's summary of weekly values (iid bootstrap with seed 0 and 4000 draws, Wilcoxon p)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = {"weeks": len(x), "mean_rel": np.nan, "pct": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
           "wilcoxon_p": np.nan}
    if len(x) < 5:
        return out
    boot = np.random.default_rng(0).choice(x, (4000, len(x))).mean(axis=1)
    out.update({"mean_rel": x.mean(), "pct": 100 * (np.exp(x.mean()) - 1), "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)), "wilcoxon_p": float(stats.wilcoxon(x).pvalue)})
    return out


def r2_table(dm: pd.DataFrame, measures, variants: dict, periods: dict) -> pd.DataFrame:
    """R2 premia for each baseline variant. variants: name -> (dates removed from the
    baseline, min_other). periods: name -> function of the week table returning a mask."""
    rows = []
    for vname, (excl, min_other) in variants.items():
        ex = set(pd.DatetimeIndex(list(excl)))
        drop = dm.index.isin(list(ex)) & ~dm["is_expiry"].astype(bool)
        d = dm[~drop]
        for col in measures:
            wr = r2_week_relative(d, col, min_other)
            for pname, sel in periods.items():
                m = sel(wr) if len(wr) else np.zeros(0, bool)
                s = r2_summary(wr.loc[m, "rel"].to_numpy() if len(wr) else np.array([]))
                rows.append({"baseline": vname, "measure": col, "period": pname, **s,
                             "mean_baseline_days": float(wr.loc[m, "n_other"].mean()) if len(wr) and np.any(m) else np.nan})
    return pd.DataFrame(rows)


def two_day_difference(values: pd.Series, wd_a: int, wd_b: int) -> pd.Series:
    """Per week: log value on weekday a minus log value on weekday b (no baseline at all)."""
    v = np.log(values.where(values > 0)).dropna()
    wk = es.week_start(v.index)
    df = pd.DataFrame({"v": v.to_numpy(), "week": wk, "wd": v.index.weekday})
    a = df[df["wd"] == wd_a].set_index("week")["v"]
    b = df[df["wd"] == wd_b].set_index("week")["v"]
    return (a - b).dropna()


# ---------------------------------------------------------------------------
# 3. Inference with few effective units
# ---------------------------------------------------------------------------


def nonoverlap_sets(ks, W: int) -> list[np.ndarray]:
    """All greedy sets of non-overlapping placebo positions (windows of 2W weeks), one per
    starting phase 0..2W-1. Phase 0 is the set counted by es.n_independent."""
    ks = np.sort(np.asarray(ks, int))
    if len(ks) == 0:
        return []
    sets, seen = [], set()
    for phase in range(2 * W):
        chosen, last = [], -10**9
        for k in ks:
            if k < ks[0] + phase:
                continue
            if k - last >= 2 * W:
                chosen.append(k)
                last = k
        key = tuple(chosen)
        if chosen and key not in seen:
            seen.add(key)
            sets.append(np.asarray(chosen, int))
    return sets


def randomization_p_nonoverlap(t_obs: float, t_placebo) -> float:
    """Share of non-overlapping placebo |t| at least as large as |t_obs|, with the +1
    correction. Its floor is 1 / (q + 1) for q placebo windows."""
    tp = np.asarray(t_placebo, float)
    tp = tp[np.isfinite(tp)]
    if not np.isfinite(t_obs) or len(tp) == 0:
        return np.nan
    return float((1 + np.sum(np.abs(tp) >= abs(t_obs))) / (1 + len(tp)))


def prediction_t_p(t_obs: float, t_placebo) -> float:
    """Two-sided p of t_obs as a new draw from the normal population of the q placebo t
    statistics: (t_obs - mean) / (sd * sqrt(1 + 1/q)) ~ t with q - 1 degrees of freedom.
    Valid with few units if the placebo statistics are roughly normal; needs q >= 3."""
    tp = np.asarray(t_placebo, float)
    tp = tp[np.isfinite(tp)]
    q = len(tp)
    if not np.isfinite(t_obs) or q < 3:
        return np.nan
    sd = tp.std(ddof=1)
    if sd <= 0:
        return np.nan
    z = (t_obs - tp.mean()) / (sd * np.sqrt(1 + 1 / q))
    return float(2 * stats.t.sf(abs(z), q - 1))


_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster_p(pre, post, block: int = 4, n_boot: int = 1999, seed: int = 0) -> tuple[float, int]:
    """Wild cluster restricted bootstrap p for the post-minus-pre difference in means.

    Regression y = a + b post; clusters are runs of ``block`` consecutive available weeks
    within each side (no cluster straddles the event). The null b = 0 is imposed, residuals
    are flipped by cluster with Webb's six-point weights, and the CR1 t statistic is compared.
    Returns (p, number of clusters). With fewer than about 10 clusters this is the few-cluster
    case: the test can be conservative or liberal, and the p is coarse."""
    pre = np.asarray(pre, float)
    post = np.asarray(post, float)
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    if len(pre) < 3 or len(post) < 3:
        return np.nan, 0
    y = np.concatenate([pre, post])
    x = np.concatenate([np.zeros(len(pre)), np.ones(len(post))])
    cl = np.concatenate([np.arange(len(pre)) // block, 1000 + np.arange(len(post)) // block])
    _, cl = np.unique(cl, return_inverse=True)
    G = int(cl.max() + 1)
    if G < 3:
        return np.nan, G
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)

    def tstat(Y):  # Y: (B, n)
        beta = Y @ X @ XtX_inv.T            # (B, 2)
        e = Y - beta @ X.T                  # (B, n)
        S = np.zeros((Y.shape[0], G, 2))
        for g in range(G):
            m = cl == g
            S[:, g, :] = e[:, m] @ X[m]
        meat = np.einsum("bgi,bgj->bij", S, S)
        n, k = len(y), 2
        adj = G / (G - 1) * (n - 1) / (n - k)
        V = adj * np.einsum("ij,bjk,kl->bil", XtX_inv, meat, XtX_inv)
        return beta[:, 1] / np.sqrt(np.maximum(V[:, 1, 1], 1e-300))

    t_obs = tstat(y[None, :])[0]
    y0 = np.full_like(y, y.mean())
    e0 = y - y0
    rng = np.random.default_rng(seed)
    w = rng.choice(_WEBB, size=(n_boot, G))
    Ystar = y0[None, :] + w[:, cl] * e0[None, :]
    t_b = tstat(Ystar)
    return float((1 + np.sum(np.abs(t_b) >= abs(t_obs))) / (1 + n_boot)), G


def trend_adjusted_change(pre: pd.Series, post: pd.Series, lags: int = 4) -> tuple[float, float, float]:
    """Level shift at the event net of one linear time trend fitted to both sides:
    y = a + b t + c post (t in years); returns (c, HAC se, HAC p)."""
    import statsmodels.api as sm

    pre, post = pre.dropna(), post.dropna()
    if len(pre) < 5 or len(post) < 5:
        return np.nan, np.nan, np.nan
    y = np.concatenate([pre.to_numpy(), post.to_numpy()])
    wk = pre.index.append(post.index)
    t = (wk - wk[0]).days.to_numpy() / 365.25
    X = sm.add_constant(np.column_stack([t, np.r_[np.zeros(len(pre)), np.ones(len(post))]]))
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    return float(fit.params[2]), float(fit.bse[2]), float(fit.pvalues[2])


# ---------------------------------------------------------------------------
# 4. Noise-robust end-of-day variance
# ---------------------------------------------------------------------------


def window_log_prices(close: np.ndarray, first_slot: int, last_slot: int) -> np.ndarray:
    """Log prices from the close of the bar before ``first_slot`` to the close of
    ``last_slot`` (days x (last - first + 2)), forward-filled within the day."""
    cff = pd.DataFrame(close).ffill(axis=1).to_numpy()
    return np.log(cff[:, first_slot - 1:last_slot + 1])


def rv_from_logp(lp: np.ndarray) -> np.ndarray:
    r = np.diff(lp, axis=1)
    return np.nansum(r ** 2, axis=1)


def rv_subsampled(lp: np.ndarray, k: int) -> np.ndarray:
    """Average over the k offsets of the realized variance of k-minute returns. Each offset's
    sum is scaled by n / (minutes covered) so that every offset covers the whole window."""
    n = lp.shape[1] - 1
    vals = []
    for o in range(k):
        idx = np.arange(o, n + 1, k)
        if len(idx) < 2:
            continue
        cover = idx[-1] - idx[0]
        r = np.diff(lp[:, idx], axis=1)
        vals.append(np.nansum(r ** 2, axis=1) * n / cover)
    return np.mean(vals, axis=0)


def parzen(x: np.ndarray) -> np.ndarray:
    x = np.abs(np.asarray(x, float))
    return np.where(x <= 0.5, 1 - 6 * x ** 2 + 6 * x ** 3, np.where(x <= 1, 2 * (1 - x) ** 3, 0.0))


def realized_kernel(r: np.ndarray, H: int) -> np.ndarray:
    """Realized kernel with Parzen weights (Barndorff-Nielsen, Hansen, Lunde and Shephard
    2008, 2009): gamma_0 + 2 sum_{h=1}^{H} k(h / (H + 1)) gamma_h, with gamma_h the sum of
    r_j r_{j-h} inside the window. Non-negative by construction (Parzen kernel)."""
    r = np.nan_to_num(np.atleast_2d(np.asarray(r, float)))
    out = (r ** 2).sum(axis=1)
    for h in range(1, H + 1):
        out = out + 2 * parzen(h / (H + 1)) * (r[:, h:] * r[:, :-h]).sum(axis=1)
    return out


def pre_averaged(r: np.ndarray, kn: int) -> np.ndarray:
    """Pre-averaging estimator (Jacod, Li, Mykland, Podolskij and Vetter 2009) with
    g(x) = min(x, 1 - x) and window kn, bias-corrected for noise with the window's own
    realized variance, and divided by (1 - c) where c = psi1 / (2 kn^2 psi2) removes the
    finite-sample loss of integrated variance that the noise correction causes. Can be
    negative on a single day."""
    r = np.nan_to_num(np.atleast_2d(np.asarray(r, float)))
    n = r.shape[1]
    j = np.arange(1, kn)
    g = np.minimum(j / kn, 1 - j / kn)
    gfull = np.minimum(np.arange(kn + 1) / kn, 1 - np.arange(kn + 1) / kn)
    psi1 = kn * np.sum(np.diff(gfull) ** 2)
    psi2 = np.sum(g ** 2) / kn
    nwin = n - kn + 2
    Y = np.zeros((r.shape[0], nwin))
    for i in range(nwin):
        seg = r[:, i:i + kn - 1]
        Y[:, i] = seg @ g[:seg.shape[1]]
    c = psi1 / (2 * kn ** 2 * psi2)
    raw = n / nwin / (kn * psi2) * (Y ** 2).sum(axis=1) - c * (r ** 2).sum(axis=1)
    return raw / (1 - c)


def autocorr1(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per day: first-order autocorrelation of returns in the window (sum r_j r_{j-1} / sum r_j^2),
    with its numerator and denominator (for pooled values)."""
    r = np.nan_to_num(np.atleast_2d(np.asarray(r, float)))
    num = (r[:, 1:] * r[:, :-1]).sum(axis=1)
    den = (r ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, np.nan)
    return rho, num, den


EOD_WINDOWS = {"last30": (345, 374), "last_hour": (315, 374), "midday30": (165, 194)}
ESTIMATORS = ["rv1", "rvss2", "rvss5", "rk", "pav"]
RK_H = 3
PAV_KN = {"last30": 6, "last_hour": 8, "midday30": 6}


def eod_measures(close: np.ndarray, days, windows=EOD_WINDOWS) -> pd.DataFrame:
    """Daily end-of-day variance with each estimator and 1-minute autocorrelation, per window.
    midday30 (12:00 to 12:30) is a reference window for the autocorrelation."""
    out = pd.DataFrame(index=pd.DatetimeIndex(days))
    for w, (a, b) in windows.items():
        lp = window_log_prices(close, a, b)
        r = np.diff(lp, axis=1)
        out[f"{w}_rv1"] = rv_from_logp(lp)
        out[f"{w}_rvss2"] = rv_subsampled(lp, 2)
        out[f"{w}_rvss5"] = rv_subsampled(lp, 5)
        out[f"{w}_rk"] = realized_kernel(r, RK_H)
        out[f"{w}_rk_H1"] = realized_kernel(r, 1)
        out[f"{w}_rk_H5"] = realized_kernel(r, 5)
        out[f"{w}_pav"] = pre_averaged(r, PAV_KN[w])
        rho, num, den = autocorr1(r)
        out[f"{w}_rho1"], out[f"{w}_ac_num"], out[f"{w}_ac_den"] = rho, num, den
        out[f"{w}_zero_returns"] = (np.abs(r) < 1e-12).sum(axis=1)
    return out


def simulate_noisy_windows(n_days: int, seed: int, noise_sd: float = 0.0, n_stocks: int = 0,
                           update_prob: float = 1.0, minutes: int = 30, sigma_day: float = 0.01) -> tuple[np.ndarray, float]:
    """SYNTHETIC. Log prices at 1-minute closes over a window of ``minutes`` minutes
    (minutes + 1 prices) built from a 1-second efficient price with constant volatility
    (``sigma_day`` per 375-minute day). Either iid noise (``noise_sd`` in log units) is added
    to each 1-minute close, or (``n_stocks`` > 0) the observed price is an equal-weight index
    of stale stock prices: each stock follows the common efficient price plus its own
    martingale part and updates with probability ``update_prob`` per second (last-trade
    staleness). Returns (log prices, true integrated variance of the efficient price)."""
    rng = np.random.default_rng(seed)
    sec = 60
    n_sec = minutes * sec + sec  # one extra minute before the window starts
    s2 = sigma_day ** 2 / (375 * sec)
    if n_stocks == 0:
        eff = np.cumsum(rng.normal(0, np.sqrt(s2), (n_days, n_sec)), axis=1)
        obs = eff[:, sec - 1::sec]
        obs = obs + rng.normal(0, noise_sd, obs.shape) if noise_sd > 0 else obs
        return obs, s2 * minutes * sec
    common = np.cumsum(rng.normal(0, np.sqrt(s2), (n_days, n_sec)), axis=1)
    idx = np.zeros((n_days, n_sec))
    for _ in range(n_stocks):
        own = np.cumsum(rng.normal(0, np.sqrt(s2), (n_days, n_sec)), axis=1)
        p = common + own
        upd = rng.random((n_days, n_sec)) < update_prob
        upd[:, 0] = True
        last = np.where(upd, np.arange(n_sec)[None, :], 0)
        last = np.maximum.accumulate(last, axis=1)
        idx += np.take_along_axis(p, last, axis=1)
    idx /= n_stocks
    obs = idx[:, sec - 1::sec]
    # integrated variance of the efficient index (common part plus the average of the own parts)
    return obs, s2 * (1 + 1 / n_stocks) * minutes * sec


def estimator_bias_table(n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC check of every estimator over a 30-minute window: mean estimate / true IV."""
    scen = {"no noise": {}, "iid noise, small (1-min RV bias about +10%)": {"noise_sd": 1.15e-4},
            "iid noise, large (1-min RV bias about +50%)": {"noise_sd": 2.58e-4},
            "stale index (20 stocks, updates every 30 s on average)": {"n_stocks": 20, "update_prob": 1 / 30}}
    rows = []
    for j, (name, kw) in enumerate(scen.items()):
        lp, iv = simulate_noisy_windows(n_days, seed + j, **kw)
        r = np.diff(lp, axis=1)
        est = {"rv1": rv_from_logp(lp), "rvss2": rv_subsampled(lp, 2), "rvss5": rv_subsampled(lp, 5),
               "rk": realized_kernel(r, RK_H), "rk_H1": realized_kernel(r, 1), "rk_H5": realized_kernel(r, 5),
               "pav": pre_averaged(r, PAV_KN["last30"])}
        rho, _, _ = autocorr1(r)
        for k, v in est.items():
            rows.append({"scenario": name, "estimator": k, "days": n_days, "mean_over_true": float(np.mean(v) / iv),
                         "sd_over_true": float(np.std(v) / iv), "share_negative": float(np.mean(v < 0)),
                         "mean_rho1": float(np.nanmean(rho)), "data": "SYNTHETIC"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Event tests with the extra inference columns
# ---------------------------------------------------------------------------

WINDOWS = [("3 months", 13), ("6 months", 26), ("12 months", 52), ("paper window", None)]


def event_specs(ev: pd.Series) -> list[dict]:
    """The comparisons of experiments/expiry_event_study.py (main), copied exactly."""
    a_, c_, d_ = ev["a"], ev["c"], ev["d"]
    e_ = ev.get("e", None)
    return [
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
        {"id": "b_BN_Wed_m", "event": "b", "label": "BN Wednesday, monthly weeks", "a": ("BN", "Wed"), "sample": "clean_monthly", "lo": a_, "hi": c_},
        {"id": "b_BN_Thu_m", "event": "b", "label": "BN Thursday, monthly weeks", "a": ("BN", "Thu"), "sample": "clean_monthly", "lo": a_, "hi": c_},
        {"id": "b_N_Thu_m", "event": "b", "label": "Nifty Thursday, monthly weeks", "a": ("N", "Thu"), "sample": "clean_monthly", "lo": a_, "hi": c_},
        {"id": "b_N_Wed_m", "event": "b", "label": "Nifty Wednesday, monthly weeks", "a": ("N", "Wed"), "sample": "clean_monthly", "lo": a_, "hi": c_},
        {"id": "b_N_Thu_all", "event": "b", "label": "Nifty Thursday, all regular weeks", "a": ("N", "Thu"), "sample": "regular", "lo": a_, "hi": c_},
        {"id": "b_N_Wed_all", "event": "b", "label": "Nifty Wednesday, all regular weeks", "a": ("N", "Wed"), "sample": "regular", "lo": a_, "hi": c_},
        {"id": "b_BN_Wed_all", "event": "b", "label": "BN Wednesday, all regular weeks", "a": ("BN", "Wed"), "sample": "regular", "lo": a_, "hi": c_},
        {"id": "b_BN_Thu_all", "event": "b", "label": "BN Thursday, all regular weeks", "a": ("BN", "Thu"), "sample": "regular", "lo": a_, "hi": c_},
        {"id": "c_BN_Wed", "event": "c", "label": "BN Wednesday (expiry until c)", "a": ("BN", "Wed"), "lo": a_, "hi": d_},
        {"id": "c_N_Thu", "event": "c", "label": "Nifty Thursday (expiry throughout)", "a": ("N", "Thu"), "lo": a_, "hi": d_},
        {"id": "c_DiD_main", "event": "c", "label": "DiD main: BN Wed minus Nifty Thu", "a": ("BN", "Wed"), "b": ("N", "Thu"), "lo": a_, "hi": d_},
        {"id": "c_DiD_sameday", "event": "c", "label": "DiD same weekday: BN Wed minus Nifty Wed", "a": ("BN", "Wed"), "b": ("N", "Wed"), "lo": a_, "hi": d_},
        {"id": "c_N_Wed", "event": "c", "label": "Nifty Wednesday (BN expiry until c)", "a": ("N", "Wed"), "lo": a_, "hi": d_},
        {"id": "c_BN_Thu", "event": "c", "label": "BN Thursday (Nifty expiry day)", "a": ("BN", "Thu"), "lo": a_, "hi": d_},
        {"id": "c_DiD_Fri", "event": "c", "label": "Negative control: BN Fri minus Nifty Fri (no NSE change)", "a": ("BN", "Fri"), "b": ("N", "Fri"), "lo": a_, "hi": d_},
        {"id": "d_N_Thu", "event": "d", "label": "Nifty Thursday (former expiry)", "a": ("N", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_N_Tue", "event": "d", "label": "Nifty Tuesday (new expiry)", "a": ("N", "Tue"), "lo": c_, "hi": e_},
        {"id": "d_N_switch", "event": "d", "label": "Nifty Tue minus Nifty Thu (does the premium follow the expiry?)",
         "a": ("N", "Tue"), "b": ("N", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_BN_Thu", "event": "d", "label": "BN Thursday (Nifty expiry until d)", "a": ("BN", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_BN_Tue", "event": "d", "label": "BN Tuesday (Nifty expiry after d)", "a": ("BN", "Tue"), "lo": c_, "hi": e_},
        {"id": "d_DiD_Thu", "event": "d", "label": "DiD: Nifty Thu minus BN Thu", "a": ("N", "Thu"), "b": ("BN", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_N_Fri", "event": "d", "label": "Negative control: Nifty Friday (no NSE change)", "a": ("N", "Fri"), "lo": c_, "hi": e_},
        {"id": "d_N_Tue_incl_auction", "event": "d", "label": "Nifty Tuesday (new expiry), closing-auction days kept",
         "a": ("N", "Tue"), "lo": c_},
        {"id": "d_N_Thu_incl_auction", "event": "d", "label": "Nifty Thursday (former expiry), closing-auction days kept",
         "a": ("N", "Thu"), "lo": c_},
        {"id": "d_N_switch_incl_auction", "event": "d", "label": "Nifty Tue minus Nifty Thu, closing-auction days kept",
         "a": ("N", "Tue"), "b": ("N", "Thu"), "lo": c_},
    ]


def comparison_type(spec_id: str) -> str:
    if "switch" in spec_id:
        return "switch (new expiry weekday minus old, same index)"
    if "DiD" in spec_id:
        return "difference-in-differences (two indices)"
    if "_Fri" in spec_id:
        return "negative control (Friday)"
    if spec_id.startswith("b_"):
        return "single weekday, monthly weeks" if spec_id.endswith("_m") else "single weekday, all regular weeks"
    return "single weekday"


def series_tests(y: pd.Series, ev_week, W, lo, hi, blocked_weeks, n_boot: int, seed: int, paper: bool = False,
                 wcb_boot: int = 1999, with_boot: bool = True) -> dict:
    """Every test of one before/after comparison on a weekly series (grid of weeks, NaN where
    not in the sample): the event study's tests plus the few-unit inference of Section 3."""
    if paper:
        pre, post = es.window_split(y, ev_week, None, pre_start=pd.Timestamp("2023-10-30"), pre_end=pd.Timestamp("2024-02-26"),
                                    post_start=pd.Timestamp("2024-03-04"), post_end=pd.Timestamp("2024-05-27"))
        Wp = 17
    else:
        pre, post = es.window_split(y, ev_week, W, lo, hi)
        Wp = W
    a, b = pre.to_numpy(), post.to_numpy()
    if with_boot:
        res = es.compare(a, b, n_boot=n_boot, seed=seed)
    else:
        a2, b2 = a[np.isfinite(a)], b[np.isfinite(b)]
        res = {"n_pre": len(a2), "n_post": len(b2), "diff": np.nan, "welch_t": np.nan, "hac_p": np.nan, "hac_se": np.nan,
               "mwu_p": np.nan, "boot_lo": np.nan, "boot_hi": np.nan, "mean_pre": np.nan, "mean_post": np.nan}
        if len(a2) >= 3 and len(b2) >= 3:
            _, se, p = es.hac_post(a2, b2)
            res.update({"diff": b2.mean() - a2.mean(), "welch_t": es.welch_t(a2, b2), "hac_p": p, "hac_se": se,
                        "mean_pre": a2.mean(), "mean_post": b2.mean(),
                        "mwu_p": float(stats.mannwhitneyu(b2, a2, alternative="two-sided").pvalue)})
    grid_vals, grid_weeks = y.to_numpy(), y.index
    blocked_idx = [int(np.searchsorted(grid_weeks, pd.Timestamp(bw))) for bw in blocked_weeks]
    ks = es.placebo_event_indices(len(grid_weeks), None, Wp, blocked_idx)
    tp = es.placebo_t_stats(grid_vals, ks, Wp)
    ok = np.isfinite(tp)
    ks_ok, tp_ok = ks[ok], tp[ok]
    n_ind = es.n_independent(ks_ok, Wp)
    p_ri, n_pl = es.randomization_p(res["welch_t"], tp, n_ind)
    sets = nonoverlap_sets(ks_ok, Wp)
    tmap = dict(zip(ks_ok.tolist(), tp_ok.tolist()))
    p_no = [randomization_p_nonoverlap(res["welch_t"], [tmap[k] for k in s]) for s in sets]
    p_pr = [prediction_t_p(res["welch_t"], [tmap[k] for k in s]) for s in sets]
    greedy = sets[0] if sets else np.array([], int)
    wcr, G = wild_cluster_p(a, b, n_boot=wcb_boot, seed=seed)
    slope, p_tr = es.pretrend(pre.index, a)
    ta, ta_se, ta_p = trend_adjusted_change(pre, post)
    res.update({
        "ri_p": p_ri, "ri_n_placebo": n_pl, "ri_n_independent": n_ind,
        "placebo_first_week": grid_weeks[ks_ok.min() - Wp].date() if len(ks_ok) else None,
        "placebo_last_week": grid_weeks[min(ks_ok.max() + Wp - 1, len(grid_weeks) - 1)].date() if len(ks_ok) else None,
        "nonoverlap_window_centres": ";".join(grid_weeks[k].strftime("%Y-%m-%d") for k in greedy),
        "nonoverlap_q": len(greedy), "nonoverlap_floor": 1 / (len(greedy) + 1) if len(greedy) else np.nan,
        "nonoverlap_p": p_no[0] if p_no else np.nan,
        "nonoverlap_p_median_over_phases": float(np.nanmedian(p_no)) if p_no and np.isfinite(p_no).any() else np.nan,
        "nonoverlap_phases": len(sets),
        "prediction_t_p": p_pr[0] if p_pr else np.nan,
        "prediction_t_p_median_over_phases": float(np.nanmedian(p_pr)) if p_pr and np.isfinite(p_pr).any() else np.nan,
        "wild_cluster_p": wcr, "wild_cluster_G": G,
        "pretrend_slope_per_year": slope, "pretrend_p": p_tr,
        "trend_adjusted_change": ta, "trend_adjusted_se": ta_se, "trend_adjusted_p": ta_p,
        "pre_first_week": pre.dropna().index.min(), "post_last_week": post.dropna().index.max()})
    return res


def event_rows(rel, flags, events, specs, windows, blocked_weeks, measures, n_boot, with_boot=True,
               wcb_boot: int = 1999) -> pd.DataFrame:
    """es.run_event_tests with the extra inference columns (same series, windows and seeds)."""
    evw = events.set_index("event")["event_week"]
    out = []
    for sp in specs:
        ser = es.Series(rel, flags, sp["a"], sp.get("b"), sp.get("sample", "clean_weekly"))
        for m in measures:
            if m == "corr5_z" and sp.get("b") is not None and sp["b"][1] == sp["a"][1]:
                continue
            y = ser.values(m)
            if y.notna().sum() < 6:
                for wname, _ in windows:
                    if wname == "paper window" and sp["event"] != "b":
                        continue
                    out.append({"spec_id": sp["id"], "event": sp["event"], "comparison": sp["label"],
                                "comparison_type": comparison_type(sp["id"]), "measure": m, "window": wname,
                                "n_pre": int(y[y.index < evw.loc[sp["event"]]].notna().sum()),
                                "n_post": int(y[y.index >= evw.loc[sp["event"]]].notna().sum()),
                                "note": "fewer than 6 usable weeks in the whole grid"})
                continue
            for wname, W in windows:
                if wname == "paper window" and sp["event"] != "b":
                    continue
                r = series_tests(y, evw.loc[sp["event"]], W, sp.get("lo"), sp.get("hi"), blocked_weeks, n_boot,
                                 es.seed_of(sp["id"], m, wname), paper=(wname == "paper window"), with_boot=with_boot,
                                 wcb_boot=wcb_boot)
                out.append({"spec_id": sp["id"], "event": sp["event"], "comparison": sp["label"],
                            "comparison_type": comparison_type(sp["id"]), "measure": m, "window": wname, **r})
    return pd.DataFrame(out)


def calibration_shares(df: pd.DataFrame, label: str) -> dict:
    def share(x):
        x = x.dropna().astype(float)
        return float(x.mean()) if len(x) else np.nan
    return {"group": label, "rows": len(df), "rows_estimable": int(df["diff"].notna().sum()) if "diff" in df else 0,
            "share_hac_p_below_0.05": share((df["hac_p"] < 0.05).where(df["hac_p"].notna())),
            "share_mwu_p_below_0.05": share((df["mwu_p"] < 0.05).where(df["mwu_p"].notna())),
            "share_boot_ci_excludes_0": share(((df["boot_lo"] > 0) | (df["boot_hi"] < 0)).where(df["boot_lo"].notna())),
            "rows_with_placebo_p": int(df["ri_p"].notna().sum()),
            "share_placebo_p_below_0.05": share((df["ri_p"] < 0.05).where(df["ri_p"].notna())),
            "share_nonoverlap_p_below_0.05": share((df["nonoverlap_p"] < 0.05).where(df["nonoverlap_p"].notna())),
            "share_prediction_t_p_below_0.05": share((df["prediction_t_p"] < 0.05).where(df["prediction_t_p"].notna())),
            "share_wild_cluster_p_below_0.05": share((df["wild_cluster_p"] < 0.05).where(df["wild_cluster_p"].notna())),
            "median_wild_cluster_G": float(df["wild_cluster_G"].median()) if "wild_cluster_G" in df and df["wild_cluster_G"].notna().any() else np.nan}


def placebo_summary(pl: pd.DataFrame) -> pd.DataFrame:
    """Per window: how many independent placebo windows the published p rests on, and how often each p rejects."""
    rows = []
    for wname, g in pl.groupby("window", sort=False):
        def share(col, cut=0.05, g=g):
            x = g[col].dropna()
            return float((x < cut).mean()) if len(x) else np.nan
        rows.append({
            "window": wname, "rows": len(g), "rows_estimable": int(g["diff"].notna().sum()),
            "ri_n_independent_min": g["ri_n_independent"].min(), "ri_n_independent_median": g["ri_n_independent"].median(),
            "ri_n_independent_max": g["ri_n_independent"].max(),
            "rows_with_placebo_p": int(g["ri_p"].notna().sum()), "share_placebo_p_below_0.05": share("ri_p"),
            "nonoverlap_q_min": g["nonoverlap_q"].min(), "nonoverlap_q_median": g["nonoverlap_q"].median(),
            "nonoverlap_q_max": g["nonoverlap_q"].max(), "nonoverlap_floor_median": g["nonoverlap_floor"].median(),
            "rows_with_nonoverlap_p": int(g["nonoverlap_p"].notna().sum()),
            "share_nonoverlap_p_below_0.05": share("nonoverlap_p"), "share_nonoverlap_p_below_0.10": share("nonoverlap_p", 0.10),
            "rows_with_prediction_t_p": int(g["prediction_t_p"].notna().sum()), "share_prediction_t_p_below_0.05": share("prediction_t_p"),
            "rows_with_wild_cluster_p": int(g["wild_cluster_p"].notna().sum()), "share_wild_cluster_p_below_0.05": share("wild_cluster_p"),
            "median_wild_cluster_G": g["wild_cluster_G"].median(),
            "share_hac_p_below_0.05": share("hac_p")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Calendar changes (the table the referee asked for)
# ---------------------------------------------------------------------------

SEBI_2024 = ("SEBI circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132, 1 Oct 2024 (listed as Verified in docs/LITERATURE_REVIEW.md); "
             "phase dates as summarised by Business Standard, 3 Oct 2024")
SEBI_2024_URL = ("https://www.business-standard.com/amp/markets/capital-market-news/"
                 "sebi-tightens-norms-for-equity-derivatives-trading-124100300354_1.html")
OTHER_URL = "data/external/other_weekly_expiries.csv"
SX_URL = "data/external/sensex_expiry_calendar.csv"


def calendar_change_table(nse_changes: pd.DataFrame, ev: pd.Series, first_listing: dict | None = None) -> pd.DataFrame:
    """Every calendar or rule change from 2023 to 2026 that touches the weekdays compared in the
    expiry studies. NSE rows take their dates from the bhavcopy detection (verified); BSE and
    SEBI rows are reported (sources given). 'inside_windows' lists the event windows (6 or 12
    months each side, bounded by neighbouring events as in the event study) that contain it."""
    def det(sym, kind, change, new_wd=None):
        c = nse_changes[(nse_changes["symbol"] == sym) & (nse_changes["kind"] == kind) & (nse_changes["change"] == change)]
        if new_wd is not None:
            c = c[c["new_weekday"] == new_wd]
        if len(c) == 0:
            return None, None
        r = c.iloc[0]
        return r["last_old_expiry"], r["first_new_expiry"]

    rows = []

    def add(date, exch, index, change, status, source, url, in_study=False, det_last=None):
        rows.append({"date": pd.Timestamp(date) if date is not None and pd.notna(date) else pd.NaT, "exchange": exch,
                     "index": index, "change": change, "status": status,
                     "last_old_expiry": pd.Timestamp(det_last).date() if det_last is not None and pd.notna(det_last) else None,
                     "in_published_event_list": in_study, "source": source, "source_url": url})

    nse_src = "NSE F&O bhavcopies (full files; this script's detection)"
    lo, fn = det("BANKNIFTY", "weekly", "weekday", "Wed")
    add(fn, "NSE", "Bank Nifty", "weekly expiry Thursday to Wednesday (event a)", "verified (NSE bhavcopy)", nse_src, "", True, lo)
    lo, fn = det("BANKNIFTY", "monthly", "weekday", "Wed")
    add(fn, "NSE", "Bank Nifty", "monthly expiry Thursday to Wednesday (event b)", "verified (NSE bhavcopy)", nse_src, "", True, lo)
    lo, fn = det("BANKNIFTY", "weekly", "stopped")
    add(lo, "NSE", "Bank Nifty", "last weekly expiry; weeklies stop (event c)", "verified (NSE bhavcopy)", nse_src, "", True, lo)
    lo, fn = det("BANKNIFTY", "monthly", "weekday", "Thu")
    add(fn, "NSE", "Bank Nifty", "monthly expiry Wednesday to last Thursday (event c2)", "verified (NSE bhavcopy)", nse_src, "", True, lo)
    lo, fn = det("NIFTY", "weekly", "weekday", "Tue")
    add(fn, "NSE", "Nifty 50", "weekly expiry Thursday to Tuesday; NSE monthlies to Tuesday (event d)", "verified (NSE bhavcopy)",
        nse_src + "; NSE circular 108/2025", "", True, lo)
    names = {"FINNIFTY": "Nifty Financial Services", "MIDCPNIFTY": "Nifty Midcap Select", "NIFTYNXT50": "Nifty Next 50",
             "NIFTY": "Nifty 50", "NIFTYFPI": "NIFTYFPI (new index options, Aug 2026)"}
    for _, r in nse_changes.iterrows():
        sym = r["symbol"]
        if sym not in names or (sym == "NIFTY" and r["kind"] == "weekly"):
            continue
        last = r["last_old_expiry"]
        if pd.notna(last) and last < pd.Timestamp("2022-10-01"):
            continue
        if r["change"] == "weekday":
            note = " (two weeks before event a)" if (sym == "MIDCPNIFTY" and r["new_weekday"] == "Mon" and r["kind"] == "weekly") else ""
            if pd.notna(last) and WD[pd.Timestamp(last).weekday()] != r["old_weekday"]:
                note += f" (the last expiry before the change, {pd.Timestamp(last).date()}, was a {WD[pd.Timestamp(last).weekday()]})"
            add(r["first_new_expiry"], "NSE", names[sym], f"{r['kind']} expiry {r['old_weekday']} to {r['new_weekday']}{note}",
                "verified (NSE bhavcopy)", nse_src, "", sym == "NIFTY", last)
        elif r["change"] == "stopped":
            add(last, "NSE", names[sym], "last weekly expiry; weeklies stop (part of the SEBI package at event c)",
                "verified (NSE bhavcopy)", nse_src, "", False, last)
    for sym, name in (("NIFTYNXT50", "Nifty Next 50"), ("NIFTYFPI", "NIFTYFPI")):
        first = first_listing.get(sym) if first_listing else None
        if first is not None:
            add(first, "NSE", name, "index options listed (monthly expiries only)", "verified (NSE bhavcopy)", nse_src, "", False)
    add("2023-05-15", "BSE", "Sensex and Bankex", "weekly options relaunched; first weekly expiry Friday 19 May 2023",
        "reported", "Business Standard / PTI, 19 May 2023 (via sensex_expiry_calendar.csv)",
        "https://www.business-standard.com/companies/news/sensex-bankex-derivative-contracts-witness-record-rs-528-crore-turnover-123051901107_1.html")
    add("2023-10-16", "BSE", "Bankex", "weekly expiry Friday to Monday", "reported", "Business Standard, 30 Aug 2023 (via other_weekly_expiries.csv)",
        "https://www.business-standard.com/amp/markets/news/bse-moves-bankex-expiry-to-monday-from-friday-to-be-effective-from-oct-16-123083000569_1.html")
    add("2024-11-14", "BSE", "Sensex 50", "last weekly expiry; weeklies stop", "reported",
        "Outlook Business / Angel One reports of the BSE notice of Oct 2024 (via other_weekly_expiries.csv)", OTHER_URL)
    add("2024-11-18", "BSE", "Bankex", "last weekly expiry; weeklies stop", "reported",
        "Outlook Business / Angel One reports of the BSE notice of Oct 2024 (via other_weekly_expiries.csv)", OTHER_URL)
    add("2024-11-20", "SEBI", "all index options", "phase 1: one weekly benchmark expiry per exchange, minimum contract value "
        "Rs 15 lakh, extra 2% margin on short options on expiry day", "verified (circular listed in LITERATURE_REVIEW.md)",
        SEBI_2024, SEBI_2024_URL, True)
    add("2025-01-07", "BSE", "Sensex", "weekly expiry Friday to Tuesday (last Friday 3 Jan 2025)", "reported",
        "BSE circular of 28 Nov 2024 as reported (via sensex_expiry_calendar.csv)", SX_URL)
    add("2025-02-01", "SEBI", "all index options", "phase 2: upfront collection of option premium from buyers; no calendar-spread "
        "margin benefit for contracts expiring that day", "reported (secondary summary; circular not re-read)", SEBI_2024, SEBI_2024_URL)
    add("2025-04-01", "SEBI", "all index options", "phase 3: intraday monitoring of position limits in index derivatives",
        "reported (secondary summary; circular not re-read)", SEBI_2024, SEBI_2024_URL)
    add("2025-05-26", "SEBI", "all equity derivatives", "circular: each exchange's expiries on Tuesday or Thursday",
        "reported", "SEBI circular SEBI/HO/MRD/MRD-TPD-1/P/CIR/2025/76 (via sensex_expiry_calendar.csv)", SX_URL)
    add("2025-09-04", "BSE", "Sensex", "weekly expiry Tuesday to Thursday (same week as event d)", "reported",
        "news summaries of the BSE circular and SEBI approval (via sensex_expiry_calendar.csv)", SX_URL)
    add("2026-08-03", "NSE", "cash market close", "closing auction sets the official close (event e)", "detected in minute bars",
        "experiments/expiry_event_study.py", "", True)
    t = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    bounds = {"a": (None, ev["c"]), "b": (ev["a"], ev["c"]), "c": (ev["a"], ev["d"]), "d": (ev["c"], ev.get("e"))}
    inside = []
    for d in t["date"]:
        hits = []
        for e, (lo, hi) in bounds.items():
            if pd.notna(d) and es.week_start([d])[0] == ev[e]:
                hits.append(f"{e} event week")
            for wn, W in (("6m", 26), ("12m", 52)):
                p0 = ev[e] - pd.Timedelta(weeks=W)
                p1 = ev[e] + pd.Timedelta(weeks=W)
                if lo is not None:
                    p0 = max(p0, lo)
                if hi is not None:
                    p1 = min(p1, hi)
                if pd.notna(d) and p0 <= es.week_start([d])[0] < p1 and es.week_start([d])[0] != ev[e]:
                    hits.append(f"{e} {wn}")
        inside.append(", ".join(hits))
    t["inside_event_windows"] = inside
    t["date"] = t["date"].dt.date
    return t


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------


def _ci(m, lo, hi, nd=2):
    return es._ci(m, lo, hi, nd) if np.isfinite(m) else "n/a"


def _p(p):
    return es._p(p)


def _md(df: pd.DataFrame) -> str:
    return es._md(df.astype(object).where(df.notna(), ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--ext", default=str(ROOT / "data" / "external"))
    ap.add_argument("--out", default=str(ROOT / "results" / "referee_expiry"))
    ap.add_argument("--boot", type=int, default=4999)
    ap.add_argument("--fetch-nse", action="store_true", help="download NSE's full F&O bhavcopies (about 15 minutes) and stop")
    ap.add_argument("--end", default="2026-11-02", help="last raw-data date read by the event-study loader (the committed "
                    "run's snapshot ended on 2026-11-02); 'none' reads everything on disk")
    a = ap.parse_args()
    t0 = time.time()
    raw, ext_dir, out = Path(a.raw), Path(a.ext), Path(a.out)
    if a.fetch_nse:
        fetch_nse_full(raw, ext_dir, "2019-01-01", "2026-11-02")
        return
    out.mkdir(parents=True, exist_ok=True)
    L = ["REAL DATA (BSE calendar reported, not verified). Referee response on the expiry evidence: M8, M9, M10. POST HOC.", ""]

    # ------------------------------------------------------------------ data, exactly as in the event study
    D = es.build_data(raw, None if str(a.end).lower() == "none" else pd.Timestamp(a.end))
    daily, expiries = D["daily"], D["expiries"]
    events = es.define_events(D["changes"], D["auction_start"])
    ev = events.set_index("event")["event_week"]
    flags = es.week_flags(expiries, daily)
    blocked = list(ev.to_numpy())
    rel_pub = es.build_relative(daily, expiries)
    pub_rel = pd.read_csv(ROOT / "results" / "event_study" / "weekly_relative.csv.gz", parse_dates=["week", "target_date"])
    chk = rel_pub.merge(pub_rel, on=["index", "target", "measure", "week"], suffixes=("", "_pub"))
    L += [f"Event-study data rebuilt: {len(rel_pub)} weekly values; matched to the committed weekly_relative.csv.gz on "
          f"{len(chk)} of {len(pub_rel)} rows, largest difference {np.nanmax(np.abs(chk['rel'] - chk['rel_pub'])):.2e}."]

    # ------------------------------------------------------------------ 1. calendar of every index expiry
    src = json.loads((ext_dir / NSE_EXTRACT_SOURCE).read_text())
    ext = pd.read_csv(ext_dir / NSE_EXTRACT, parse_dates=["date", "expiry", "actual_expiry"])
    unknown = set(ext.loc[ext["inst"] == "opt", "symbol"]) - set(NSE_SYMBOLS)
    if unknown:
        raise RuntimeError(f"index option symbols not in NSE_SYMBOLS: {unknown}")
    nse_exp = nse_expiries_from_extract(ext)
    nse_listed = nse_expiries_from_extract(ext, traded_only=False)
    L += ["", "== 1. Calendar of every index-option expiry",
          f"Full NSE bhavcopies: {src['days_found']} trade dates found of {src['days_requested']} weekdays requested "
          f"({ext['date'].min().date()} to {ext['date'].max().date()}), fetched {src['fetched_utc']} from {src['source']}.",
          "Index option symbols in the files: " + ", ".join(f"{s} ({ext.loc[(ext['symbol'] == s) & (ext['inst'] == 'opt'), 'date'].min().date()} "
                                                        f"to {ext.loc[(ext['symbol'] == s) & (ext['inst'] == 'opt'), 'date'].max().date()})"
                                                        for s in sorted(ext.loc[ext['inst'] == 'opt', 'symbol'].unique())),
          f"Expiry days with traded contracts: {len(nse_exp)}; with listed contracts (traded or not): {len(nse_listed)} "
          f"(differences: {sorted(set(map(tuple, nse_listed[['symbol', 'expiry']].astype(str).values)) - set(map(tuple, nse_exp[['symbol', 'expiry']].astype(str).values)))})"]
    # cross-check Nifty and Bank Nifty against the event study's own calendar (filtered bhavcopies)
    for sym in ("NIFTY", "BANKNIFTY"):
        mine = set(nse_exp.loc[(nse_exp["symbol"] == sym) & (nse_exp["expiry"] >= D["first_bhav"]), "expiry"])
        theirs = set(expiries.loc[(expiries["symbol"] == sym) & (expiries["source"] == "bhavcopy"), "expiry"])
        mk = set(nse_exp.loc[(nse_exp["symbol"] == sym) & (nse_exp["kind"] == "monthly") & (nse_exp["expiry"] >= D["first_bhav"]), "expiry"])
        tk = set(expiries.loc[(expiries["symbol"] == sym) & (expiries["source"] == "bhavcopy") & (expiries["kind"] == "monthly"), "expiry"])
        L.append(f"Check {sym} against the event study's calendar from {D['first_bhav'].date()}: {len(mine)} vs {len(theirs)} dates, "
                 f"identical: {mine == theirs}; monthly labels identical: {mk == tk}")
        rule = expiries[(expiries["symbol"] == sym) & (expiries["source"] != "bhavcopy")]
        mine_pre = set(nse_exp.loc[(nse_exp["symbol"] == sym) & (nse_exp["expiry"] >= es.LONG_START) & (nse_exp["expiry"] < D["first_bhav"]), "expiry"])
        rset = set(rule.loc[rule["expiry"] >= es.LONG_START, "expiry"])
        L.append(f"  before {D['first_bhav'].date()} the event study used a Thursday rule: {len(rset)} rule dates vs {len(mine_pre)} "
                 f"bhavcopy dates, identical: {rset == mine_pre}; rule-only {sorted(d.date().isoformat() for d in rset - mine_pre)}; "
                 f"bhavcopy-only {sorted(d.date().isoformat() for d in mine_pre - rset)}")
    nse_changes, _ = es.detect_schedule_changes(nse_exp.assign(source="bhavcopy"), data_end=ext["date"].max())
    nse_changes.to_csv(out / "nse_schedule_changes_detected.csv", index=False)
    L += ["NSE schedule changes detected from the full bhavcopies (three expiries on a new weekday confirm a change):",
          nse_changes.to_string(index=False)]
    # how much trading each index's expiry carried (expiring options' notional turnover on the expiry day)
    act = nse_exp[nse_exp["expiry"] >= pd.Timestamp("2022-10-17")].copy()
    act["half_year"] = act["expiry"].dt.year.astype(str) + np.where(act["expiry"].dt.month <= 6, "H1", "H2")
    nifty_med = act[act["symbol"] == "NIFTY"].groupby("half_year")["expiring_option_turnover_rs_cr"].median()
    activity = (act.groupby(["symbol", "half_year"])
                .agg(expiries=("expiry", "size"), weekdays=("weekday", lambda s: ";".join(f"{k} {v}" for k, v in s.value_counts().items())),
                     median_expiring_turnover_rs_cr=("expiring_option_turnover_rs_cr", "median"),
                     median_expiring_contracts=("expiring_option_contracts", "median")).reset_index())
    activity["relative_to_nifty_expiry_median"] = activity["median_expiring_turnover_rs_cr"] / activity["half_year"].map(nifty_med)
    activity.to_csv(out / "nse_index_expiry_activity.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L += ["Expiry-day notional turnover of the expiring options, NSE indices (median per half-year, Rs crore):",
              activity.round(4).to_string(index=False)]

    full_days = daily["N"].index.union(daily["BN"].index)
    full_days = full_days[full_days >= es.LONG_START]
    # trading days without a bhavcopy in NSE's archive: assume a weekly expiry of an index whose
    # expiries fall on that weekday in the weeks just before and after
    no_file = sorted(set(full_days) - set(ext["date"]))
    fills = []
    for d in no_file:
        for sym, g in nse_exp[nse_exp["kind"] == "weekly"].groupby("symbol"):
            near = g[(g["expiry"] - d).abs() <= pd.Timedelta(days=8)]
            if len(near) >= 2 and (near["expiry"].dt.weekday == d.weekday()).all() and \
                    (near["expiry"] < d).any() and (near["expiry"] > d).any():
                fills.append({"symbol": sym, "expiry": d, "weekday": WD[d.weekday()], "kind": "weekly", "exchange": "NSE",
                              "status": "assumed (no bhavcopy for this day in NSE's archive; neighbouring weeks expire on this weekday)"})
    L.append(f"Trading days with minute bars but no bhavcopy in NSE's archive: {[d.date().isoformat() for d in no_file]}; "
             f"assumed expiries added: {[(f['symbol'], f['expiry'].date().isoformat()) for f in fills]}")
    if fills:
        nse_exp = pd.concat([nse_exp, pd.DataFrame(fills)], ignore_index=True).sort_values(["symbol", "expiry"]).reset_index(drop=True)
    sx_cal = pd.read_csv(ROOT / "data" / "external" / "sensex_expiry_calendar.csv", dtype=str, keep_default_na=False)
    other = pd.read_csv(ROOT / "data" / "external" / "other_weekly_expiries.csv")
    bse_exp = bse_expiries(daily["N"].index, sx_cal, other)
    sx_pub = pd.read_csv(ROOT / "results" / "sensex_confound" / "sensex_expiry_days_used.csv", parse_dates=["expiry"])
    L.append(f"Sensex dates identical to results/sensex_confound/sensex_expiry_days_used.csv: "
             f"{set(bse_exp.loc[bse_exp['symbol'] == 'SENSEX', 'expiry']) == set(sx_pub['expiry'])}")
    # the reported NSE rows of other_weekly_expiries.csv against the bhavcopies
    for _, o in other[other["exchange"] == "NSE"].iterrows():
        reg = [(pd.Timestamp(o["start"]), pd.Timestamp(o["end"]), WD.index(o["weekday"]))]
        rule = set(sc.sensex_rule_expiries(daily["N"].index, reg)["expiry"])
        lo_, hi_ = max(pd.Timestamp(o["start"]), ext["date"].min()), pd.Timestamp(o["end"])
        seen = set(nse_exp.loc[(nse_exp["symbol"] == o["index"]) & (nse_exp["expiry"] >= lo_) & (nse_exp["expiry"] <= hi_), "expiry"])
        cmpd = sc.compare_dates(rule, seen, lo_, hi_)
        L.append(f"Reported {o['index']} {o['weekday']} {o['start']} to {o['end']} vs bhavcopy: rule {cmpd['n_rule']}, bhavcopy "
                 f"{cmpd['n_other']}, both {cmpd['n_both']}; rule-only {cmpd['rule_only'] or 'none'}; bhavcopy-only {cmpd['other_only'] or 'none'}")
    all_exp = pd.concat([nse_exp, bse_exp], ignore_index=True)
    all_exp = all_exp[all_exp["expiry"] >= es.LONG_START].sort_values(["expiry", "exchange", "symbol"]).reset_index(drop=True)
    all_exp.to_csv(out / "all_index_expiries.csv", index=False)
    cal_days = full_days.union(pd.DatetimeIndex(ext["date"].unique()))
    cal_days = cal_days[(cal_days >= es.LONG_START) & (cal_days.weekday < 5)]
    cal = day_calendar(cal_days, all_exp)
    cal["full_session"] = cal.index.isin(full_days)
    cal.to_csv(out / "all_index_expiry_calendar.csv")
    all_dates = set(all_exp["expiry"])
    nse_dates = set(nse_exp["expiry"])
    sx_dates = set(bse_exp.loc[bse_exp["symbol"] == "SENSEX", "expiry"])

    wc = week_counts(cal, full_days)
    wc["clean_weekly"] = flags["clean_weekly"].reindex(wc.index).fillna(False).astype(bool)
    e_ = ev.get("e", pd.Timestamp("2100-01-01"))
    es_reg = [("R0 (Mar 2019 to Oct 2022, rule-based NSE calendar in the event study)", es.LONG_START, pd.Timestamp("2022-10-17")),
              ("R0v (Oct 2022 to a)", pd.Timestamp("2022-10-17"), ev["a"]), ("R1 (a to c)", ev["a"], ev["c"]),
              ("R2 (c to d)", ev["c"], ev["d"]), ("R3 (d to e)", ev["d"], e_), ("R4 (closing auction, from e)", e_, pd.Timestamp("2100-01-01"))]
    r2_reg = [("multi_weekly (Jan 2023 to 19 Nov 2024)", pd.Timestamp("2023-01-02"), pd.Timestamp("2024-11-18")),
              ("single_weekly_thu (20 Nov 2024 to 31 Aug 2025)", pd.Timestamp("2024-11-18"), pd.Timestamp("2025-09-01")),
              ("single_weekly_tue (from 1 Sep 2025)", pd.Timestamp("2025-09-01"), pd.Timestamp("2100-01-01"))]
    seg = [("2019-03 to 2020-12 (Nifty, Bank Nifty Thursday only)", es.LONG_START, pd.Timestamp("2021-01-04")),
           ("2021-01 to 2022-10 (FinNifty weeklies start)", pd.Timestamp("2021-01-04"), pd.Timestamp("2022-10-17")),
           ("17 Oct 2022 to 14 May 2023", pd.Timestamp("2022-10-17"), pd.Timestamp("2023-05-15")),
           ("15 May to 20 Aug 2023 (Sensex, Bankex relaunched on Friday)", pd.Timestamp("2023-05-15"), pd.Timestamp("2023-08-21")),
           ("21 Aug to 3 Sep 2023 (Midcap moved to Monday)", pd.Timestamp("2023-08-21"), ev["a"]),
           ("4 Sep 2023 to 17 Nov 2024 (event a to c)", ev["a"], ev["c"]),
           ("18 Nov 2024 to 5 Jan 2025 (Sensex on Friday)", ev["c"], pd.Timestamp("2025-01-06")),
           ("6 Jan to 31 Aug 2025 (Sensex on Tuesday)", pd.Timestamp("2025-01-06"), ev["d"]),
           ("1 Sep 2025 to 2 Aug 2026 (Nifty Tue, Sensex Thu)", ev["d"], e_),
           ("3 Aug 2026 on (closing auction)", e_, pd.Timestamp("2100-01-01"))]
    wc["event_study_regime"] = label_regimes(wc.index, es_reg)
    wc["r2_regime"] = label_regimes(wc.index, r2_reg)
    wc["calendar_segment"] = label_regimes(wc.index, seg)
    wc.to_csv(out / "no_expiry_days_by_week.csv")
    # how much each NSE index's expiry carried in each calendar segment and on each weekday
    a2 = nse_exp[nse_exp["expiry"] >= pd.Timestamp("2022-10-17")].copy()
    a2["calendar_segment"] = label_regimes(es.week_start(a2["expiry"]), seg)
    nmed = a2[a2["symbol"] == "NIFTY"].groupby("calendar_segment")["expiring_option_turnover_rs_cr"].median()
    act_seg = (a2.groupby(["calendar_segment", "symbol", "weekday"], sort=False)
               .agg(expiries=("expiry", "size"), first=("expiry", "min"), last=("expiry", "max"),
                    median_expiring_turnover_rs_cr=("expiring_option_turnover_rs_cr", "median")).reset_index())
    act_seg["relative_to_nifty_expiry_median_in_segment"] = act_seg["median_expiring_turnover_rs_cr"] / act_seg["calendar_segment"].map(nmed)
    act_seg["first"], act_seg["last"] = act_seg["first"].dt.date, act_seg["last"].dt.date
    act_seg.to_csv(out / "nse_index_expiry_activity_by_segment.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L += ["Expiring options' notional turnover by calendar segment and weekday (median, Rs crore; relative to Nifty's median "
              "expiry in the same segment):", act_seg.round(5).to_string(index=False)]
    summ = []
    for col in ("event_study_regime", "r2_regime", "calendar_segment"):
        for sample, w in (("all weeks", wc), ("clean weekly weeks", wc[wc["clean_weekly"]])):
            s = summarise_weeks(w, col)
            s.insert(0, "sample", sample)
            s.insert(0, "grouping", col)
            summ.append(s)
    summ = pd.concat(summ, ignore_index=True)
    summ.to_csv(out / "no_expiry_days_by_regime.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L += ["Days with no expiry of any index (NSE verified, BSE reported), per week:",
              summ[summ["sample"] == "all weeks"][["grouping", "regime", "weeks", "mean_no_expiry_days", "weeks_with_0_no_expiry_days",
                                                    "weeks_with_1", "weeks_with_2_or_more", "mean_no_nse_expiry_days",
                                                    "most_common_no_expiry_weekdays"]].round(2).to_string(index=False)]
    changes_tab = calendar_change_table(nse_changes, ev, {s: ext.loc[(ext["symbol"] == s) & (ext["inst"] == "opt"), "date"].min() for s in ("NIFTYNXT50", "NIFTYFPI")})
    changes_tab.to_csv(out / "calendar_changes.csv", index=False)

    # ------------------------------------------------------------------ 2a. R2 with a clean baseline
    dm = pd.read_csv(ROOT / "results" / "pipeline_nse_v2" / "daily_market.csv", index_col=0, parse_dates=True)
    r2_pub = pd.read_csv(ROOT / "results" / "real_day_level" / "r2_expiry_effect_by_regime.csv")
    mine = []
    for col in ("rv_day", "rv_first_hour", "rv_last_hour"):
        wr = r2_week_relative(dm, col, 2)
        for reg in rdl.REGIMES:
            mine.append(wr.loc[wr["regime"] == reg, "rel"].mean())
    pub_means = r2_pub[r2_pub["regime"].isin(rdl.REGIMES)]["mean_rel"].to_numpy()
    L += ["", "== 2. M9: R2 and the event study with the baseline restricted to days with no expiry of any index",
          f"R2 rebuilt from results/pipeline_nse_v2/daily_market.csv reproduces the published means: {np.allclose(mine, pub_means)}"]
    variants = {
        "published: all days that are not Nifty expiry days": (set(), 2),
        "Sensex days removed (as in SENSEX_CONFOUND.md)": (sx_dates, 2),
        "no NSE expiry of any index (verified), at least 2 days": (nse_dates, 2),
        "no expiry of any index, NSE or BSE, at least 2 days": (all_dates, 2),
        "no expiry of any index, NSE or BSE, at least 1 day": (all_dates, 1),
    }
    periods = {
        "multi_weekly": lambda w: w["regime"] == "multi_weekly",
        "multi_weekly, 2 Jan to 14 May 2023": lambda w: (w["regime"] == "multi_weekly") & (w["date"] < pd.Timestamp("2023-05-15")),
        "multi_weekly, 15 May to 20 Aug 2023": lambda w: (w["regime"] == "multi_weekly") & (w["date"] >= pd.Timestamp("2023-05-15")) & (w["date"] < pd.Timestamp("2023-08-21")),
        "multi_weekly, 21 Aug 2023 to 19 Nov 2024": lambda w: (w["regime"] == "multi_weekly") & (w["date"] >= pd.Timestamp("2023-08-21")),
        "single_weekly_thu": lambda w: w["regime"] == "single_weekly_thu",
        "single_weekly_tue": lambda w: w["regime"] == "single_weekly_tue",
        "single_weekly_tue, before 3 Aug 2026": lambda w: (w["regime"] == "single_weekly_tue") & (w["date"] < AUCTION),
        "single_weekly_tue, closing-auction days": lambda w: (w["regime"] == "single_weekly_tue") & (w["date"] >= AUCTION),
    }
    r2c = r2_table(dm, ["rv_day", "rv_last_hour"], variants, periods)
    r2c.to_csv(out / "r2_clean_baseline.csv", index=False)
    # minor comment 16: Tuesday against Thursday single-weekly regime, with each baseline
    pw = []
    for vname, (excl, min_other) in variants.items():
        drop = dm.index.isin(list(excl)) & ~dm["is_expiry"].astype(bool)
        for col in ("rv_day", "rv_last_hour"):
            wr = r2_week_relative(dm[~drop], col, min_other)
            xa = wr.loc[wr["regime"] == "single_weekly_thu", "rel"].dropna()
            xb = wr.loc[wr["regime"] == "single_weekly_tue", "rel"].dropna()
            if len(xa) >= 5 and len(xb) >= 5:
                pw.append({"baseline": vname, "measure": col, "weeks_thu": len(xa), "weeks_tue": len(xb),
                           "tue_minus_thu": xb.mean() - xa.mean(), "mannwhitney_p": float(stats.mannwhitneyu(xa, xb).pvalue),
                           "bonferroni_p_18_tests": min(1.0, 18 * float(stats.mannwhitneyu(xa, xb).pvalue))})
    pw = pd.DataFrame(pw)
    pw.to_csv(out / "r2_tue_vs_thu_by_baseline.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L += [r2c[["baseline", "measure", "period", "weeks", "mean_rel", "pct", "ci95_low", "ci95_high", "wilcoxon_p",
                   "mean_baseline_days"]].round(4).to_string(index=False),
              "Tuesday minus Thursday single-weekly regime (minor comment 16):", pw.round(4).to_string(index=False)]

    # ------------------------------------------------------------------ 2b. event study with a clean baseline
    MS = ["rv5", "rs", "rv_last_hour", "rv_last30"]
    targets = {k: {**{WD[i]: es.weekday_targets(daily[k].index, i) for i in range(5)},
                   "own": es.date_targets(daily[k].index, expiries.loc[expiries["symbol"] == es.SYMBOLS[k], "expiry"])}
               for k in ("N", "BN")}
    rel_clean = sc.relative_table(daily, targets, MS, exclude={"N": all_dates, "BN": all_dates})
    rel_nse = sc.relative_table(daily, targets, MS, exclude={"N": nse_dates, "BN": nse_dates})
    prem_regs = [r for r in es_reg if not r[0].startswith("R0 (")]
    prem = []
    for k in ("N", "BN"):
        for tgt in ["own"] + WD:
            for m in MS:
                for bname, rt in (("published", rel_pub), ("no NSE expiry (verified)", rel_nse), ("no expiry of any index", rel_clean)):
                    y = es.Series(rt, flags, (k, tgt), None, "clean_weekly").values(m)
                    for rname, lo, hi in prem_regs:
                        x = y[(y.index >= lo) & (y.index < hi)].dropna()
                        s = es.level_summary(x.to_numpy(), n_boot=a.boot, seed=es.seed_of(k, tgt, m, rname, "clean_weekly"))
                        prem.append({"index": k, "target": tgt, "measure": m, "regime": rname, "baseline": bname, **s})
    prem = pd.DataFrame(prem)
    prem.to_csv(out / "event_premium_by_regime_clean_baseline.csv", index=False)
    specs = event_specs(ev)
    et_win = [w for w in WINDOWS if w[0] != "paper window"]
    et_clean = es.run_event_tests(rel_clean, flags, events, specs, et_win, blocked, a.boot, measures=MS)
    et_pub = pd.read_csv(ROOT / "results" / "event_study" / "event_tests.csv")
    keyc = ["spec_id", "measure", "window"]
    vals = ["n_pre", "n_post", "diff", "boot_lo", "boot_hi", "hac_p", "ri_p", "ri_n_independent", "pretrend_p"]
    base = et_pub[et_pub["measure"].isin(MS) & et_pub["window"].isin([w for w, _ in et_win])][keyc + ["event", "comparison"] + vals]
    cmp_clean = base.merge(et_clean[keyc + vals], on=keyc, how="left", suffixes=("", "_clean"))
    for c in ("n_pre_clean", "n_post_clean"):
        cmp_clean[c] = cmp_clean[c].fillna(0).astype(int)
    cmp_clean["estimable_clean"] = cmp_clean["diff_clean"].notna()
    cmp_clean.to_csv(out / "event_tests_clean_baseline.csv", index=False)
    est = cmp_clean.groupby(["event", "window"]).agg(rows=("spec_id", "size"), estimable_published=("diff", lambda s: int(s.notna().sum())),
                                                      estimable_clean=("estimable_clean", "sum")).reset_index()
    L += ["Event-study rows (rv5, RS, last hour, last 30 minutes) estimable with at least 3 weeks per side:",
          est.to_string(index=False)]

    # baseline-free two-day differences (switch comparisons)
    tdd_rows = []
    for k, wa, wb, e, lo, hi, lab in (("BN", 2, 3, "a", None, ev["c"], "BN Wed minus BN Thu"),
                                       ("N", 1, 3, "d", ev["c"], ev.get("e"), "Nifty Tue minus Nifty Thu"),
                                       ("N", 2, 3, "a", None, ev["c"], "Nifty Wed minus Nifty Thu"),
                                       ("BN", 1, 3, "d", ev["c"], ev.get("e"), "BN Tue minus BN Thu")):
        for m in ("rv5", "rv_last_hour", "rv_last30"):
            y = two_day_difference(daily[k][m], wa, wb).reindex(flags.index[flags.index >= es.LONG_START])
            y = y.where(flags["clean_weekly"].reindex(y.index).fillna(False).astype(bool))
            for wname, W in (("6 months", 26), ("12 months", 52)):
                r = series_tests(y, ev[e], W, lo, hi, blocked, a.boot, es.seed_of("tdd", k, wa, wb, m, wname))
                tdd_rows.append({"event": e, "series": lab, "measure": m, "window": wname, **r})
    tdd = pd.DataFrame(tdd_rows)
    tdd.to_csv(out / "two_day_raw_differences.csv", index=False)

    # pooled regression with a dummy for every index's expiry
    cw = set(flags.index[flags["clean_weekly"]])
    reg_rows = []
    for k in ("N", "BN"):
        d = daily[k]
        d = d[(d.index >= pd.Timestamp("2022-10-17")) & (d.index < e_)]
        p = pd.DataFrame(index=d.index)
        p["week"] = es.week_start(d.index)
        p = p[p["week"].isin(cw)]
        own = "NIFTY" if k == "N" else "BANKNIFTY"
        is_own = cal[own].reindex(p.index).fillna(0).astype(bool).to_numpy()
        if k == "N":
            p["own_thu_to_d"] = (is_own & (p.index < ev["d"])).astype(float)
            p["own_tue_from_d"] = (is_own & (p.index >= ev["d"])).astype(float)
        else:
            p["own_thu_to_a"] = (is_own & (p.index < ev["a"])).astype(float)
            p["own_wed_a_to_c"] = (is_own & (p.index >= ev["a"])).astype(float)
        others = [s for s in ALL_SYMBOLS if s != own]
        for s in others:
            p[f"x_{s}"] = (cal[s].reindex(p.index).fillna(0).astype(bool).to_numpy() & ~is_own).astype(float)
        for i in range(1, 5):
            p[f"wd_{WD[i]}"] = (p.index.weekday == i).astype(float)
        own_cols = [c for c in p.columns if c.startswith("own")]
        x_cols = [c for c in p.columns if c.startswith("x_")]
        wd_cols = [c for c in p.columns if c.startswith("wd_")]
        for m in ("rv5", "rv_last_hour", "rv_last30"):
            p["y"] = np.log(d[m].reindex(p.index).where(lambda s: s > 0))
            for spec, xs in (("own expiry only, weekday effects", own_cols + wd_cols),
                             ("all index dummies, weekday effects", own_cols + x_cols + wd_cols),
                             ("all index dummies, no weekday effects", own_cols + x_cols)):
                tab, n, nw = sc.within_week_ols(p, "y", xs)
                for _, t in tab.iterrows():
                    if t["term"].startswith("wd_"):
                        continue
                    reg_rows.append({"index": k, "measure": m, "spec": spec, "term": t["term"], "coef": t["coef"], "se": t["se"],
                                     "p": t["p"], "n_days": n, "n_weeks": nw})
    pooled = pd.DataFrame(reg_rows)
    pooled.to_csv(out / "pooled_all_index_regression.csv", index=False)

    # ------------------------------------------------------------------ 3. M8: placebo inference on every published row
    allrows = event_rows(rel_pub, flags, events, specs, WINDOWS, blocked, es.MEASURES, a.boot, with_boot=False, wcb_boot=999)
    m_ = allrows.merge(et_pub[keyc + ["diff", "ri_p", "ri_n_independent", "boot_lo", "boot_hi", "hac_p", "mwu_p"]],
                       on=keyc, how="left", suffixes=("", "_pub"))
    same_diff = np.nanmax(np.abs(m_["diff"] - m_["diff_pub"]))
    both = m_["ri_p"].notna() | m_["ri_p_pub"].notna()
    same_ri = np.allclose(m_.loc[both, "ri_p"].fillna(-1), m_.loc[both, "ri_p_pub"].fillna(-1))
    same_ind = bool((m_["ri_n_independent"].fillna(-1) == m_["ri_n_independent_pub"].fillna(-1)).all())
    m_["boot_lo"], m_["boot_hi"] = m_["boot_lo_pub"], m_["boot_hi_pub"]
    keep = ["spec_id", "event", "comparison", "comparison_type", "measure", "window", "n_pre", "n_post", "diff", "boot_lo", "boot_hi",
            "hac_p", "mwu_p", "ri_p", "ri_n_placebo", "ri_n_independent", "placebo_first_week", "placebo_last_week",
            "nonoverlap_window_centres", "nonoverlap_q", "nonoverlap_floor", "nonoverlap_p", "nonoverlap_p_median_over_phases",
            "nonoverlap_phases", "prediction_t_p", "prediction_t_p_median_over_phases", "wild_cluster_p", "wild_cluster_G",
            "pretrend_p", "trend_adjusted_change", "trend_adjusted_p"]
    pl = m_[keep]
    pl.to_csv(out / "placebo_inference_all_rows.csv", index=False)
    L += ["", "== 3. M8: placebo inference",
          f"Every row of results/event_study/event_tests.csv recomputed: {len(pl)} rows; largest difference in the change "
          f"{same_diff:.1e}; placebo p identical: {same_ri}; number of non-overlapping windows identical: {same_ind}."]
    with pd.option_context("display.width", 250):
        L.append(pl.groupby("window")["ri_n_independent"].describe().to_string())
    psum = placebo_summary(pl)
    psum.to_csv(out / "placebo_inference_summary.csv", index=False)
    head_keys = [("a_BN_switch", "rv_last30"), ("a_BN_switch", "rv_last_hour"), ("a_BN_switch", "rv5"),
                 ("a_BN_Wed", "rv_last30"), ("a_N_Fri", "rv5"), ("c_DiD_main", "rv5"),
                 ("d_N_Tue", "rv_last30"), ("d_N_Tue", "rv_last_hour"), ("d_N_Thu", "rv_last30"), ("d_N_switch", "rv_last30")]
    hk = pd.DataFrame(head_keys, columns=["spec_id", "measure"])
    ph = hk.merge(pl[pl["window"].isin(["3 months", "6 months", "12 months"])], on=["spec_id", "measure"], how="left")
    ph.to_csv(out / "placebo_inference_headline_rows.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L += ["Placebo inference, summary over every row (shares among rows where the p-value exists):",
              psum.round(3).to_string(index=False),
              "Headline rows:",
              ph[["spec_id", "measure", "window", "diff", "hac_p", "ri_p", "ri_n_independent", "nonoverlap_q", "nonoverlap_floor",
                  "nonoverlap_p", "prediction_t_p", "wild_cluster_p", "wild_cluster_G", "pretrend_p"]].round(4).to_string(index=False)]

    # negative control: published Friday rows, then without expiry Fridays
    minute_m = [m for m in es.MEASURES if m not in ("opt_premium", "opt_notional", "fut_notional")]
    fri_specs = [s for s in specs if "_Fri" in s["id"]]
    neg_pub = event_rows(rel_pub, flags, events, fri_specs, et_win, blocked, minute_m, a.boot)
    fri_ok = {k: es.weekday_targets(daily[k].index, 4) for k in ("N", "BN")}
    fri_ok = {k: v[~v.isin(list(all_dates))] for k, v in fri_ok.items()}
    fri_ok_sx = {k: es.weekday_targets(daily[k].index, 4) for k in ("N", "BN")}
    fri_ok_sx = {k: v[~v.isin(list(sx_dates))] for k, v in fri_ok_sx.items()}
    neg_var = {}
    for vname, tg, excl in (("Sensex expiry Fridays dropped", fri_ok_sx, None),
                            ("Fridays with any index expiry dropped", fri_ok, None),
                            ("Fridays with any index expiry dropped, baseline without expiry days", fri_ok, all_dates)):
        rt = sc.relative_table(daily, {k: {"Fri": tg[k]} for k in ("N", "BN")}, minute_m,
                               exclude=None if excl is None else {"N": excl, "BN": excl})
        neg_var[vname] = event_rows(rt, flags, events, fri_specs, et_win, blocked, minute_m, a.boot)
    cal_rows = []
    trt = pl[~pl["spec_id"].str.contains("_Fri") & pl["measure"].isin(minute_m)]
    for wname, _ in et_win:
        cal_rows.append({"window": wname, "variant": "published Friday rows", **calibration_shares(neg_pub[neg_pub["window"] == wname], "negative control (Friday)")})
        for vname, df in neg_var.items():
            cal_rows.append({"window": wname, "variant": vname, **calibration_shares(df[df["window"] == wname], "negative control (Friday)")})
        tw = trt[trt["window"] == wname].copy()
        cal_rows.append({"window": wname, "variant": "treated or spillover weekdays (published rows)", **calibration_shares(tw, "treated")})
    calib = pd.DataFrame(cal_rows)
    calib.to_csv(out / "negative_control_recomputed.csv", index=False)
    pubcal = pd.read_csv(ROOT / "results" / "event_study" / "negative_control_calibration.csv")
    for wname, _ in et_win:
        q = neg_pub[neg_pub["window"] == wname]
        pc = pubcal[(pubcal["window"] == wname) & (pubcal["group"] == "negative control (Friday)") &
                    (pubcal["measures"] == "minute-bar measures")].iloc[0]
        L.append(f"Published Friday rows, {wname}, recomputed with the event study's own share definition: {len(q)} rows, HAC "
                 f"{(q['hac_p'] < 0.05).mean():.3f} (published {pc['share_hac_p_below_0.05']:.3f}), bootstrap "
                 f"{((q['boot_lo'] > 0) | (q['boot_hi'] < 0)).mean():.3f} (published {pc['share_boot_ci_excludes_0']:.3f}), placebo "
                 f"{(q['ri_p'] < 0.05).sum() / max(q['ri_p'].notna().sum(), 1):.3f} (published {pc['share_placebo_p_below_0.05']})")
    neg_all = pd.concat([neg_pub.assign(variant="published Friday rows")] + [v.assign(variant=k) for k, v in neg_var.items()], ignore_index=True)
    neg_all.to_csv(out / "negative_control_rows.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L += ["Negative-control calibration (shares of rows with p < 0.05):", calib.round(3).to_string(index=False)]

    # is any weekday free of every expiry inside each event window?
    free_rows = []
    bounds = {"a": (None, ev["c"]), "b": (ev["a"], ev["c"]), "c": (ev["a"], ev["d"]), "d": (ev["c"], ev.get("e"))}
    fd = cal[cal["full_session"]]
    for e, (lo, hi) in bounds.items():
        for wname, W in et_win:
            p0, p1 = ev[e] - pd.Timedelta(weeks=W), ev[e] + pd.Timedelta(weeks=W)
            p0 = max(p0, lo) if lo is not None else p0
            p1 = min(p1, hi) if hi is not None else p1
            for side, s0, s1 in (("before", p0, ev[e]), ("after", ev[e], p1)):
                for sample in ("all weeks", "clean weekly weeks"):
                    for i, wd in enumerate(WD):
                        dd = fd[(fd["week"] >= s0) & (fd["week"] < s1) & (fd.index.weekday == i)]
                        if sample == "clean weekly weeks":
                            dd = dd[flags["clean_weekly"].reindex(dd["week"]).fillna(False).astype(bool).to_numpy()]
                        free_rows.append({"event": e, "window": wname, "side": side, "sample": sample, "weekday": wd,
                                          "weeks": len(dd), "weeks_weekday_free_of_every_expiry": int((~dd["any_expiry"]).sum()),
                                          "weeks_weekday_free_of_nse_expiry": int((~dd["any_nse"]).sum()),
                                          "indices_expiring_on_it": ";".join(sorted(set(";".join(dd["expiring"]).split(";")) - {""}))})
    free = pd.DataFrame(free_rows)
    free["free_in_every_week"] = (free["weeks"] > 0) & (free["weeks_weekday_free_of_every_expiry"] == free["weeks"])
    free.to_csv(out / "weekday_free_of_expiry.csv", index=False)
    fs_rows = []
    for (e, wname), g in free[free["sample"] == "all weeks"].groupby(["event", "window"], sort=False):
        row = {"event": e, "window": wname}
        for side in ("before", "after"):
            gs = g[g["side"] == side]
            row[f"weekdays_free_in_every_week_{side}"] = ",".join(gs.loc[gs["free_in_every_week"], "weekday"]) or "none"
            row[f"weeks_free_by_weekday_{side}"] = ", ".join(f"{r.weekday} {r.weeks_weekday_free_of_every_expiry}/{r.weeks}"
                                                             for r in gs.itertuples())
        fs_rows.append(row)
    fs = pd.DataFrame(fs_rows)
    fs.to_csv(out / "weekday_free_summary.csv", index=False)
    L += ["Weekdays free of every index expiry in every week of the window (all weeks):", fs.to_string(index=False)]

    # ------------------------------------------------------------------ 4. M10: noise-robust end-of-day variance
    sim = estimator_bias_table()
    sim.to_csv(out / "noise_estimator_simulation.csv", index=False)
    L += ["", "== 4. M10: noise-robust end-of-day variance",
          "SYNTHETIC check of the estimators (mean estimate / true integrated variance, 400 simulated 30-minute windows):",
          sim.pivot(index="estimator", columns="scenario", values="mean_over_true").round(3).to_string()]
    daily2, eods = {}, {}
    for k, fname in (("N", "nifty_1min.csv"), ("BN", "banknifty_1min.csv")):
        g = es.minute_grid(raw / fname)
        eod = eod_measures(g["close"], g["days"])
        eod = eod.reindex(daily[k].index)
        eods[k] = eod
        daily2[k] = daily[k].join(eod)
        for w in ("last30", "last_hour"):
            for e2 in ("rvss5", "rk"):
                daily2[k][f"{w}_rv1_over_{e2}"] = daily2[k][f"{w}_rv1"] / daily2[k][f"{w}_{e2}"]
        L.append(f"{es.NAMES[k]}: last-30 rv1 equals the event study's rv_last30 (largest relative difference "
                 f"{np.nanmax(np.abs(daily2[k]['last30_rv1'] / daily2[k]['rv_last30'] - 1)):.1e}); last hour "
                 f"{np.nanmax(np.abs(daily2[k]['last_hour_rv1'] / daily2[k]['rv_last_hour'] - 1)):.1e}; days with a non-positive "
                 f"pre-averaging value (last 30 min): {int((daily2[k]['last30_pav'] <= 0).sum())} of {len(daily2[k])}")
    out_eod = pd.concat({es.NAMES[k]: v for k, v in eods.items()}, names=["index_name", "date"])
    out_eod.to_csv(out / "noise_robust_daily.csv.gz", compression={"method": "gzip", "mtime": 0})
    # R2 (Nifty) with every estimator
    dm2 = dm.join(eods["N"], how="left")
    r2_rel_check = np.nanmax(np.abs(dm2["last_hour_rv1"] / dm2["rv_last_hour"] - 1))
    nm = [f"{w}_{e2}" for w in ("last30", "last_hour") for e2 in ESTIMATORS]
    per_n = {k: v for k, v in periods.items() if not k.startswith("multi_weekly,")}
    r2n = r2_table(dm2, nm, {"published: all days that are not Nifty expiry days": (set(), 2),
                             "no expiry of any index, NSE or BSE, at least 2 days": (all_dates, 2)}, per_n)
    r2n.to_csv(out / "noise_robust_r2.csv", index=False)
    L.append(f"R2 last hour rebuilt from the minute bars vs the pipeline's rv_last_hour: largest relative difference {r2_rel_check:.1e}")
    # event study with every estimator
    nm_ratio = [f"{w}_rv1_over_{e2}" for w in ("last30", "last_hour") for e2 in ("rvss5", "rk")]
    rel_noise = sc.relative_table(daily2, targets, nm + nm_ratio)
    noise_specs = [s for s in specs if s["id"] in ("a_BN_switch", "a_BN_Wed", "a_BN_Thu", "c_BN_Wed", "c_DiD_main", "d_N_Tue",
                                                     "d_N_Thu", "d_N_switch", "d_DiD_Thu")]
    et_noise = es.run_event_tests(rel_noise, flags, events, noise_specs, [("6 months", 26), ("12 months", 52)], blocked, a.boot,
                                  measures=nm)
    et_noise.to_csv(out / "noise_robust_event_tests.csv", index=False)
    pn = []
    for k in ("N", "BN"):
        for m in nm + nm_ratio:
            y = es.Series(rel_noise, flags, (k, "own"), None, "clean_weekly").values(m)
            for rname, lo, hi in prem_regs:
                x = y[(y.index >= lo) & (y.index < hi)].dropna()
                s = es.level_summary(x.to_numpy(), n_boot=a.boot, seed=es.seed_of(k, "own", m, rname, "noise"))
                pn.append({"index": k, "target": "own", "measure": m, "regime": rname, **s})
    pn = pd.DataFrame(pn)
    pn.to_csv(out / "noise_robust_premium_by_regime.csv", index=False)
    # first-order autocorrelation of 1-minute returns
    ac_rows = []
    for k in ("N", "BN"):
        d = daily2[k]
        own = "NIFTY" if k == "N" else "BANKNIFTY"
        c2 = cal.reindex(d.index)
        is_own = c2[own].fillna(0).astype(bool)
        any_e = c2["any_expiry"].fillna(False).astype(bool)
        groups = {"own expiry day": is_own, "other index expiry only": any_e & ~is_own, "no expiry of any index": ~any_e,
                  "all days other than own expiry": ~is_own}
        for rname, lo, hi in prem_regs:
            inreg = (d.index >= lo) & (d.index < hi)
            for w in ("last30", "last_hour", "midday30"):
                ref = None
                for gname, gm in groups.items():
                    sel = inreg & gm.to_numpy()
                    x = d.loc[sel, f"{w}_rho1"].dropna()
                    row = {"index": k, "regime": rname, "window": w, "group": gname, "days": len(x),
                           "mean_rho1": x.mean() if len(x) else np.nan,
                           "ci95_low": x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 2 else np.nan,
                           "ci95_high": x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 2 else np.nan,
                           "pooled_rho1": d.loc[sel, f"{w}_ac_num"].sum() / d.loc[sel, f"{w}_ac_den"].sum() if sel.any() else np.nan,
                           "mean_zero_returns": d.loc[sel, f"{w}_zero_returns"].mean() if sel.any() else np.nan}
                    if gname == "own expiry day":
                        ref = x
                    elif ref is not None and len(ref) > 2 and len(x) > 2:
                        row["welch_p_own_minus_this_group"] = float(stats.ttest_ind(ref, x, equal_var=False).pvalue)
                        row["own_minus_this_group"] = ref.mean() - x.mean()
                    ac_rows.append(row)
    ac = pd.DataFrame(ac_rows)
    ac.to_csv(out / "autocorrelation_1min_returns.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L += ["First-order autocorrelation of 1-minute returns, last 30 minutes, Nifty:",
              ac[(ac["index"] == "N") & (ac["window"] == "last30")][["regime", "group", "days", "mean_rho1", "ci95_low", "ci95_high",
                                                                       "pooled_rho1", "own_minus_this_group", "welch_p_own_minus_this_group"]].round(3).to_string(index=False)]

    # ------------------------------------------------------------------ 5. M8 bullets: pre-trends, comparison choice
    key_ids = ["a_BN_switch", "a_BN_Wed", "a_BN_Thu", "a_DiD_Wed", "b_BN_Wed_m", "c_DiD_main", "c_BN_Wed", "d_N_switch",
               "d_N_Tue", "d_N_Thu", "d_DiD_Thu"]
    pt = pl[pl["spec_id"].isin(key_ids) & pl["measure"].isin(["rv5", "rv_last_hour", "rv_last30"]) &
            pl["window"].isin(["3 months", "6 months", "12 months"])].copy()
    pt = pt.merge(et_pub[keyc + ["boot_lo", "boot_hi"]], on=keyc, how="left", suffixes=("_x", ""))
    pt = pt[["event", "spec_id", "comparison_type", "measure", "window", "n_pre", "n_post", "diff", "boot_lo", "boot_hi", "hac_p",
             "ri_p", "ri_n_independent", "nonoverlap_p", "prediction_t_p", "wild_cluster_p", "pretrend_p", "trend_adjusted_change",
             "trend_adjusted_p"]]
    pt.to_csv(out / "pretrends_and_comparisons.csv", index=False)
    sxc = pd.read_csv(ROOT / "results" / "sensex_confound" / "iii_event_tests_with_sensex_control.csv")
    an = []
    for sid in ("a_BN_switch", "d_N_switch"):
        for m in ("rv_last30", "rv_last_hour"):
            r = pl[(pl["spec_id"] == sid) & (pl["measure"] == m) & (pl["window"] == "6 months")].iloc[0]
            pb = et_pub[(et_pub["spec_id"] == sid) & (et_pub["measure"] == m) & (et_pub["window"] == "6 months")].iloc[0]
            an.append({"spec_id": sid, "measure": m, "version": "published (6 months)", "weeks": f"{pb['n_pre']}/{pb['n_post']}",
                       "diff": pb["diff"], "lo": pb["boot_lo"], "hi": pb["boot_hi"], "hac_p": pb["hac_p"], "placebo_p": pb["ri_p"],
                       "pretrend_p": pb["pretrend_p"], "prediction_t_p": r["prediction_t_p"], "wild_cluster_p": r["wild_cluster_p"]})
            s = sxc[(sxc["spec_id"] == sid) & (sxc["measure"] == m)]
            if len(s):
                s = s.iloc[0]
                an.append({"spec_id": sid, "measure": m, "version": "Sensex days out of the baselines (SENSEX_CONFOUND.md)",
                           "weeks": f"{s['n_pre_sensex_control']}/{s['n_post_sensex_control']}", "diff": s["diff_sensex_control"],
                           "lo": s["boot_lo_sensex_control"], "hi": s["boot_hi_sensex_control"], "hac_p": s["hac_p_sensex_control"],
                           "placebo_p": s["ri_p_sensex_control"], "pretrend_p": s["pretrend_p_sensex_control"]})
            c = cmp_clean[(cmp_clean["spec_id"] == sid) & (cmp_clean["measure"] == m) & (cmp_clean["window"] == "6 months")].iloc[0]
            an.append({"spec_id": sid, "measure": m, "version": "baseline without any index's expiry days",
                       "weeks": f"{c['n_pre_clean']}/{c['n_post_clean']}", "diff": c["diff_clean"], "lo": c["boot_lo_clean"],
                       "hi": c["boot_hi_clean"], "hac_p": c["hac_p_clean"], "placebo_p": c["ri_p_clean"], "pretrend_p": c["pretrend_p_clean"]})
            lab = "BN Wed minus BN Thu" if sid == "a_BN_switch" else "Nifty Tue minus Nifty Thu"
            t2 = tdd[(tdd["series"] == lab) & (tdd["measure"] == m) & (tdd["window"] == "6 months")].iloc[0]
            an.append({"spec_id": sid, "measure": m, "version": "raw two-day difference, no baseline", "weeks": f"{t2['n_pre']}/{t2['n_post']}",
                       "diff": t2["diff"], "lo": t2["boot_lo"], "hi": t2["boot_hi"], "hac_p": t2["hac_p"], "placebo_p": t2["ri_p"],
                       "pretrend_p": t2["pretrend_p"], "prediction_t_p": t2["prediction_t_p"], "wild_cluster_p": t2["wild_cluster_p"]})
            mm = "last30_rk" if m == "rv_last30" else "last_hour_rk"
            n2 = et_noise[(et_noise["spec_id"] == sid) & (et_noise["measure"] == mm) & (et_noise["window"] == "6 months")].iloc[0]
            an.append({"spec_id": sid, "measure": m, "version": "realized kernel instead of 1-minute RV", "weeks": f"{n2['n_pre']}/{n2['n_post']}",
                       "diff": n2["diff"], "lo": n2["boot_lo"], "hi": n2["boot_hi"], "hac_p": n2["hac_p"], "placebo_p": n2["ri_p"],
                       "pretrend_p": n2["pretrend_p"]})
    an = pd.DataFrame(an)
    an.to_csv(out / "switch_comparison_a_vs_d.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L += ["", "== 5. M8: pre-trends and the event-d analogue of event a's headline (switch, 6 months)", an.round(4).to_string(index=False)]

    # ------------------------------------------------------------------ summary.md
    md = ["# Referee response, expiry evidence: summary tables (REAL DATA unless marked SYNTHETIC)", "",
          "Generated by `experiments/referee_expiry.py` (post hoc, in response to the referee report of 3 November 2026). "
          "Explained in `docs/REFEREE_RESPONSE_EXPIRY.md`. NSE expiries are verified from NSE's full F&O bhavcopies; BSE expiries "
          "(Sensex, Bankex, Sensex 50) are reported, not verified.", ""]
    md += ["## No-expiry days per week", "",
           _md(summ[summ["sample"] == "all weeks"][["grouping", "regime", "weeks", "mean_no_expiry_days", "weeks_with_0_no_expiry_days",
                                                    "weeks_with_1", "weeks_with_2_or_more", "mean_no_nse_expiry_days"]].round(2)), ""]
    md += ["## Calendar changes", "", _md(changes_tab[["date", "exchange", "index", "change", "status", "inside_event_windows"]]), ""]
    md += ["## How much each NSE index's expiry carried (expiring options' notional turnover, median per segment and weekday)", "",
           _md(act_seg.round(5)), ""]
    r2s = r2c.copy()
    r2s["value"] = [f"{_ci(m, lo, hi)} ({w} wk, p {_p(p)})" if np.isfinite(m) else f"n/a ({w} wk)"
                    for m, lo, hi, w, p in zip(r2s["mean_rel"], r2s["ci95_low"], r2s["ci95_high"], r2s["weeks"], r2s["wilcoxon_p"])]
    md += ["## R2 with a clean baseline (log points; R2's iid bootstrap interval; Wilcoxon p)", "",
           _md(r2s.pivot_table(index=["measure", "period"], columns="baseline", values="value", aggfunc="first").reset_index()), ""]
    k6 = cmp_clean[(cmp_clean["window"] == "6 months") & cmp_clean["spec_id"].isin(key_ids)].copy()
    k6["published"] = [f"{_ci(d, lo, hi)}, p {_p(p)}, placebo {_p(q)} ({a_}/{b_} wk)" for d, lo, hi, p, q, a_, b_ in
                       zip(k6["diff"], k6["boot_lo"], k6["boot_hi"], k6["hac_p"], k6["ri_p"], k6["n_pre"], k6["n_post"])]
    k6["clean baseline"] = [f"{_ci(d, lo, hi)}, p {_p(p)}, placebo {_p(q)} ({a_}/{b_} wk)" if np.isfinite(d) else f"not estimable ({a_}/{b_} wk)"
                            for d, lo, hi, p, q, a_, b_ in zip(k6["diff_clean"], k6["boot_lo_clean"], k6["boot_hi_clean"], k6["hac_p_clean"],
                                                              k6["ri_p_clean"], k6["n_pre_clean"], k6["n_post_clean"])]
    md += ["## Event tests, 6 months: published against the clean baseline", "",
           _md(k6[["spec_id", "measure", "published", "clean baseline"]]), ""]
    md += ["## Placebo inference over every published row", "", _md(psum.round(3)), "",
           "## Placebo inference, headline rows", "",
           _md(ph[["spec_id", "measure", "window", "diff", "hac_p", "ri_p", "ri_n_independent", "nonoverlap_q", "nonoverlap_floor",
                   "nonoverlap_p", "prediction_t_p", "wild_cluster_p", "wild_cluster_G", "pretrend_p", "trend_adjusted_change",
                   "trend_adjusted_p"]].round(4)), ""]
    md += ["## Negative control recomputed (share of rows with p < 0.05)", "", _md(calib.round(3)), ""]
    md += ["## Weekdays free of every expiry in every week of each event window", "", _md(fs), ""]
    md += ["## Estimator check on SIMULATED prices (SYNTHETIC): mean estimate / true variance", "",
           _md(sim.pivot(index="estimator", columns="scenario", values="mean_over_true").round(3).reset_index()), ""]
    r2ns = r2n[r2n["period"].isin(["single_weekly_tue, before 3 Aug 2026", "single_weekly_tue, closing-auction days"])].copy()
    r2ns["value"] = [f"{_ci(m, lo, hi)} ({w} wk, p {_p(p)})" if np.isfinite(m) else f"n/a ({w} wk)"
                     for m, lo, hi, w, p in zip(r2ns["mean_rel"], r2ns["ci95_low"], r2ns["ci95_high"], r2ns["weeks"], r2ns["wilcoxon_p"])]
    md += ["## R2 Tuesday regime with noise-robust estimators", "",
           _md(r2ns.pivot_table(index=["baseline", "period"], columns="measure", values="value", aggfunc="first").reset_index()), ""]
    en = et_noise.copy()
    en["value"] = [f"{_ci(d, lo, hi)}, p {_p(p)}, placebo {_p(q)}" for d, lo, hi, p, q in
                   zip(en["diff"], en["boot_lo"], en["boot_hi"], en["hac_p"], en["ri_p"])]
    md += ["## Event tests with noise-robust estimators", "",
           _md(en.pivot_table(index=["spec_id", "window"], columns="measure", values="value", aggfunc="first").reset_index()), ""]
    acs = ac[ac["window"].isin(["last30", "midday30"])].copy()
    acs["value"] = [f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({n} d)" if np.isfinite(lo) else (f"{m:+.3f} ({n} d)" if n else "n/a")
                    for m, lo, hi, n in zip(acs["mean_rho1"], acs["ci95_low"], acs["ci95_high"], acs["days"])]
    md += ["## First-order autocorrelation of 1-minute returns", "",
           _md(acs.pivot_table(index=["index", "window", "regime"], columns="group", values="value", aggfunc="first").reset_index()), ""]
    md += ["## Event d's analogue of event a's headline (switch), 6 months", "", _md(an.round(3)), ""]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    L += ["", f"Total run time {time.time() - t0:.0f}s."]
    text = "\n".join(L)
    (out.parent / "referee_expiry_stdout.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
