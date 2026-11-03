"""P96: did losing the weekly expiry remove the expiry-weekday volatility premium? (synthetic control)

Pre-registration: docs/prereg/P96_SYNTHETIC_CONTROL.md. Every choice below is
fixed there; read it first. This file is self-contained (numpy, pandas, scipy;
requests only for ``fetch``; matplotlib only for the phase 2 figure).

Subcommands
-----------
fetch     Download the full NSE files this study needs from NSE's archive server
          into --out. The copies in data/raw/ were filtered on download (index
          file: Nifty 50, Nifty Bank and India VIX only; F&O bhavcopy: NIFTY and
          BANKNIFTY only), so they cannot supply donors or the Fin Nifty and
          Midcap Select calendars. Index files are saved unmodified; bhavcopies
          are cut to index futures and index options rows (all symbols).
coverage  Phase 1. Which indices exist on every day, which have listed index
          derivatives, the donor pool after the rule, each treated index's
          expiry calendar and the number of usable weeks. Prices are read only to
          check that they exist and are internally consistent (High > Low, Open
          and Close inside the range) and that the downloaded files agree with
          the filtered copies in data/raw/. Prints the SHA-256 fingerprint of the
          downloaded files. No volatility, variance, premium, fit or effect is
          computed.
power     Phase 1. Power of the design on SIMULATED data (no real data read).
analyse   Phase 2 only. The pre-registered analysis on real data.

    OMP_NUM_THREADS=1 python experiments/synthetic_control.py fetch --out DIR
    OMP_NUM_THREADS=1 python experiments/synthetic_control.py coverage --full DIR --out DIR2
    OMP_NUM_THREADS=1 python experiments/synthetic_control.py power --J 18 --T0 40 --T1 20
    OMP_NUM_THREADS=1 python experiments/synthetic_control.py analyse --full DIR   # phase 2
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
MASTER_SEED = 20261102
WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# ---------------------------------------------------------------------------
# 0. Fixed design (pre-registration, sections 2 to 10)
# ---------------------------------------------------------------------------

# SEBI circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132 of 1 Oct 2024: weekly index options
# on one benchmark index per exchange from 20 Nov 2024.
CIRCULAR_DATE = pd.Timestamp("2024-10-01")
EFFECTIVE_DATE = pd.Timestamp("2024-11-20")

PRE_START = pd.Timestamp("2023-09-04")   # Monday of Bank Nifty's first Wednesday weekly (event a)
PRE_END = pd.Timestamp("2024-09-27")     # Friday before the week of the circular
PRE_END_ALT = pd.Timestamp("2024-07-26")  # sensitivity: before SEBI's consultation paper of 30 Jul 2024
POST_WEEKS = 26                           # primary post window, calendar weeks from the post start
POST_END_LONG = pd.Timestamp("2025-08-29")  # sensitivity: to the week before Nifty moved to Tuesday
FAKE_TREATMENT = pd.Timestamp("2024-04-01")  # placebo in time: fake post = 1 Apr to 27 Sep 2024 (26 weeks)
STUDY_START, STUDY_END = PRE_START, POST_END_LONG  # donors must be complete on every trading day here

# One-hour Diwali (Muhurat) sessions on weekdays, listed by NSE as trading days.
MUHURAT_DAYS = (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-10-21"))

TREATED = {  # key: (index name in NSE's file, derivatives symbol, old weekly expiry weekday, 0 = Monday)
    "BN": ("Nifty Bank", "BANKNIFTY", 2),
    "FIN": ("Nifty Financial Services", "FINNIFTY", 1),
    "MID": ("Nifty Midcap Select", "MIDCPNIFTY", 0),
}
PRIMARY, SECONDARIES = "BN", ("FIN", "MID")
NIFTY_NAME = "Nifty 50"  # not a donor (weekly options); its Wednesday is a descriptive check only
CAL_SYMBOLS = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")  # their monthly or shifted weeks are dropped
DERIV_SYMBOL_TO_INDEX = {  # NSE index-derivative symbol -> index name in ind_close_all
    "NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank", "FINNIFTY": "Nifty Financial Services",
    "MIDCPNIFTY": "Nifty Midcap Select", "NIFTYNXT50": "Nifty Next 50",
}
ALPHA = 0.05
MIN_OTHER = 2  # other valid trading days needed in a week (as in the event study)
# Reported (not verified) Sensex weekly expiry weekdays, used only by sensitivity S5.
SENSEX_CAL = ROOT / "data" / "external" / "sensex_expiry_calendar.csv"
# SHA-256 fingerprint of the downloaded files (index_close_full/ and fo_index/) as
# printed by ``coverage`` in phase 1 and recorded in the pre-registration. ``analyse``
# stops if the files it is given have another fingerprint.
# Confidence-set grid (section 9 of the pre-registration): -3.00 to +3.00 log points.
CI_GRID = np.round(np.arange(-3.0, 3.0001, 0.01), 2)
# MDE80 of the primary ratio test as a multiple of the pre-period RMSPE, from the phase 1
# simulations (``power``, 400 panels per effect size, J = 19 and each unit's own T0 and T1).
MDE80_PER_RMSPE = {"BN": 1.93, "FIN": 1.96, "MID": 2.23}
PREREG_DATA_SHA256 = "49e64d24c9ddeddfc74351e4ee9eb00edf8877a3af61f911d8bdea69c791d223"

# Donor rule, step D3: name patterns (lower case, matched as regular expressions on the
# whole name). Order matters only for the reason printed.
NON_EQUITY = [r"\bvix\b", r"g-sec", r"bharat bond", r"1d rate", r"\btr\b", r"inverse", r"leverage",
              r"futures", r"arbitrage", r"dividend points", r"\busd\b", r"reits", r"invits"]
BROAD = [r"^nifty ?50$", r"^nifty next (50|100)$", r"^nifty ?(100|200|500)$", r"midcap ?\d", r"smallcap ?\d",
         r"microcap", r"largemidcap", r"midsmallcap", r"total market", r"midcap select", r"sme emerge",
         r"multicap", r"fpi 150", r"^nifty500 largemidsmall"]
STRATEGY = [r"alpha", r"beta", r"low[ -]volatility", r"quality", r"momentum", r"value", r"equal[ -]weight",
            r"equal-cap", r"\besg\b", r"dividend", r"liquid", r"growth sectors", r"multifactor", r"flexicap",
            r"top \d+", r"shariah", r"\d+% cap"]
# Step D5: overlap with the constituents of ANY treated index, judged from the name (banks and
# financials for Bank Nifty and Fin Nifty; the midcap and small-cap universe for Midcap Select).
# One donor pool serves all three treated units, because the rule change hit all three at once:
# a donor that holds many constituents of any of them is itself exposed to the treatment.
OVERLAP = [r"bank", r"financ", r"nbfc", r"insurance", r"capital market", r"services sector",
           r"midcap", r"midsmall", r"smallcap"]

# ---------------------------------------------------------------------------
# 1. Fetch (NSE archive). Writes only into --out.
# ---------------------------------------------------------------------------

ARCHIVE = "https://nsearchives.nseindia.com"
UDIFF_START = pd.Timestamp("2024-07-08")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}


def _bhav_urls(d: pd.Timestamp) -> list[str]:
    udiff = f"{ARCHIVE}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    legacy = f"{ARCHIVE}/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"
    return [udiff, legacy] if d >= UDIFF_START else [legacy, udiff]


def _get(session, url: str, retries: int = 4) -> bytes | None:
    import requests
    for k in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=40)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** (k + 1))
    return None


def filter_index_derivatives(raw: bytes) -> pd.DataFrame:
    """All index futures and index options rows of a full F&O bhavcopy, NSE's own columns."""
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    if "INSTRUMENT" in df.columns:
        keep = df["INSTRUMENT"].str.strip().isin(["OPTIDX", "FUTIDX"])
    else:
        keep = df["FinInstrmTp"].str.strip().isin(["IDO", "IDF"])
    return df[keep]


def _fetch_day(d: pd.Timestamp, out: Path, what: str, pause: float) -> tuple | None:
    import requests
    s = requests.Session()
    if what == "index":
        f = out / "index_close_full" / f"ind_close_all_{d:%d%m%Y}.csv"
        if f.exists():
            return None
        c = _get(s, f"{ARCHIVE}/content/indices/ind_close_all_{d:%d%m%Y}.csv")
        time.sleep(pause)
        if c is None:
            return ("index", d.date().isoformat())
        f.write_bytes(c)
        return None
    f = out / "fo_index" / f"fo_{d:%Y%m%d}.csv.gz"
    if f.exists():
        return None
    c = None
    for url in _bhav_urls(d):
        c = _get(s, url)
        if c is not None:
            break
    time.sleep(pause)
    if c is None:
        return ("fo", d.date().isoformat())
    with zipfile.ZipFile(io.BytesIO(c)) as z:
        raw = z.read(z.namelist()[0])
    filter_index_derivatives(raw).to_csv(f, index=False, compression={"method": "gzip", "mtime": 0})
    return None


def official_days(raw: Path) -> pd.DatetimeIndex:
    """NSE trading days: dates of the Nifty 50 rows in data/raw/index_daily.csv."""
    idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
    return pd.DatetimeIndex(sorted(idx.loc[idx["index"].str.lower() == "nifty 50", "date"].unique()))


def cmd_fetch(a) -> None:
    out = Path(a.out)
    (out / "index_close_full").mkdir(parents=True, exist_ok=True)
    (out / "fo_index").mkdir(parents=True, exist_ok=True)
    days = official_days(Path(a.raw))
    jobs = []
    if a.what in ("index", "both"):
        jobs += [(d, "index") for d in days]
    if a.what in ("fo", "both"):
        lo, hi = pd.Timestamp(a.fo_start), pd.Timestamp(a.fo_end)
        jobs += [(d, "fo") for d in days if lo <= d <= hi]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(lambda j: _fetch_day(j[0], out, j[1], a.pause), jobs))
    failed = [r for r in res if r is not None]
    (out / f"fetch_failures_{a.what}.json").write_text(json.dumps(failed, indent=1))
    print(f"fetch: {len(jobs)} jobs, {len(failed)} failed, {time.time() - t0:.0f}s; failures: {failed[:20]}")


# ---------------------------------------------------------------------------
# 2. Reading NSE files
# ---------------------------------------------------------------------------

INDEX_COLS = {"Index Name": "name", "Open Index Value": "open", "High Index Value": "high",
              "Low Index Value": "low", "Closing Index Value": "close"}


def name_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def load_index_panel(index_dir: Path) -> pd.DataFrame:
    """Long table date, key, name, open, high, low, close from ind_close_all_DDMMYYYY.csv files.

    The date comes from the file name (the 'Index Date' column is month-first in some
    April 2023 files; see scripts/download_nse.py). Non-numeric prices ('-') become NaN.
    """
    frames = []
    for f in sorted(Path(index_dir).glob("ind_close_all_*.csv")):
        part = pd.read_csv(f, dtype=str)
        part.columns = [c.strip() for c in part.columns]
        part = part.rename(columns=INDEX_COLS)[list(INDEX_COLS.values())]
        part["date"] = pd.to_datetime(f.stem[-8:], format="%d%m%Y")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    df["name"] = df["name"].astype(str).str.strip()
    df["key"] = df["name"].map(name_key)
    return df.drop_duplicates(["date", "key"])[["date", "key", "name", "open", "high", "low", "close"]]


def read_fo_file(path) -> pd.DataFrame:
    """One (filtered) F&O bhavcopy -> date, symbol, inst ('opt'/'fut'), expiry, contracts."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    if "INSTRUMENT" in df.columns:
        out = pd.DataFrame({
            "date": pd.to_datetime(df["TIMESTAMP"].str.strip(), format="%d-%b-%Y"),
            "symbol": df["SYMBOL"].str.strip(),
            "inst": df["INSTRUMENT"].str.strip().map({"OPTIDX": "opt", "FUTIDX": "fut"}),
            "expiry": pd.to_datetime(df["EXPIRY_DT"].str.strip(), format="%d-%b-%Y"),
            "contracts": pd.to_numeric(df["CONTRACTS"], errors="coerce"),
        })
    else:
        out = pd.DataFrame({
            "date": pd.to_datetime(df["TradDt"].str.strip()),
            "symbol": df["TckrSymb"].str.strip(),
            "inst": df["FinInstrmTp"].str.strip().map({"IDO": "opt", "IDF": "fut"}),
            "expiry": pd.to_datetime(df["XpryDt"].str.strip()),
            "contracts": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        })
    return out[out["inst"].notna()]


def load_fo(dirs) -> pd.DataFrame:
    files = []
    for d in dirs:
        files += sorted(Path(d).glob("fo_*.csv*"))
    if not files:
        return pd.DataFrame(columns=["date", "symbol", "inst", "expiry", "contracts"])
    parts = [read_fo_file(f) for f in files]
    fo = pd.concat(parts, ignore_index=True)
    # one row per (date, symbol, inst, expiry): enough for calendars
    return (fo.groupby(["date", "symbol", "inst", "expiry"], as_index=False)["contracts"].sum())


def data_fingerprint(full: Path) -> dict:
    """SHA-256 over the downloaded files: for every file in index_close_full/ and fo_index/
    (sorted by relative path), the line '<relative path> <sha256 of its bytes>'; the
    fingerprint is the SHA-256 of those lines. Also the file counts."""
    full = Path(full)
    files = sorted(list((full / "index_close_full").glob("ind_close_all_*.csv"))
                   + list((full / "fo_index").glob("fo_*.csv*")))
    lines = [f"{f.relative_to(full).as_posix()} {hashlib.sha256(f.read_bytes()).hexdigest()}" for f in files]
    return {"sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
            "n_index_files": sum(f.parent.name == "index_close_full" for f in files),
            "n_fo_files": sum(f.parent.name == "fo_index" for f in files)}


def crosscheck_local(raw: Path, panel: pd.DataFrame, fo_full: pd.DataFrame, fo_local: pd.DataFrame) -> dict:
    """Do the downloaded files agree with the filtered copies already in data/raw/?

    Index: every (date, index) row of data/raw/index_close/ (Nifty 50, Nifty Bank, India VIX)
    is looked up in the downloaded panel and its Open, High, Low and Close compared (exact
    equality of the printed values). F&O: the NIFTY and BANKNIFTY expiry tables built from
    the two sets of bhavcopies are compared over the days both cover."""
    local = load_index_panel(Path(raw) / "index_close")
    m = local.merge(panel, on=["date", "key"], how="left", suffixes=("_local", "_full"))
    px = ["open", "high", "low", "close"]
    found = m[[f"{c}_full" for c in px]].notna().any(axis=1)
    same = np.ones(len(m), bool)
    for c in px:
        a, b = m[f"{c}_local"].to_numpy(), m[f"{c}_full"].to_numpy()
        same &= (a == b) | (np.isnan(a) & np.isnan(b))
    lo, hi = fo_full["date"].min(), fo_full["date"].max()
    both = sorted(set(fo_full["date"]) & set(fo_local["date"]))
    out = {"index_rows_local": int(len(m)), "index_rows_found_in_download": int(found.sum()),
           "index_rows_identical": int(same.sum()), "fo_days_in_both": len(both)}
    for s in ("NIFTY", "BANKNIFTY"):
        e = [set(expiry_table(f[(f["symbol"] == s) & f["date"].isin(both)])["expiry"])
             for f in (fo_full, fo_local)]
        out[f"{s}_expiries_download"], out[f"{s}_expiries_local"] = len(e[0]), len(e[1])
        out[f"{s}_expiries_identical"] = e[0] == e[1]
    out["fo_span"] = f"{lo.date()} to {hi.date()}"
    return out


# ---------------------------------------------------------------------------
# 3. Expiry calendar (from the bhavcopies)
# ---------------------------------------------------------------------------


def expiry_table(fo: pd.DataFrame) -> pd.DataFrame:
    """Actual expiries: options with expiry d traded (contracts > 0) on d. 'monthly' when an
    index future of the same symbol also expires on d, else 'weekly'."""
    rows = []
    for sym, g in fo.groupby("symbol"):
        opt = g[(g["inst"] == "opt") & (g["expiry"] == g["date"]) & (g["contracts"] > 0)]
        fut = set(g.loc[g["inst"] == "fut", "expiry"])
        for e in sorted(set(opt["expiry"])):
            rows.append({"symbol": sym, "expiry": pd.Timestamp(e), "wd": pd.Timestamp(e).weekday(),
                         "kind": "monthly" if e in fut else "weekly"})
    return pd.DataFrame(rows, columns=["symbol", "expiry", "wd", "kind"])


def flag_schedule(exp: pd.DataFrame, confirm: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scheduled weekday per (symbol, kind) series; an expiry on another weekday is a holiday
    shift unless ``confirm`` consecutive expiries confirm a new weekday (then it is a change).
    Returns (exp with 'sched_wd' and 'shifted', table of weekday changes)."""
    out, changes = [], []
    for (sym, kind), g in exp.sort_values("expiry").groupby(["symbol", "kind"]):
        g = g.sort_values("expiry").reset_index(drop=True)
        wds = g["wd"].tolist()
        first = wds[:confirm]
        cur = max(set(first), key=first.count)
        sched = []
        for i, w in enumerate(wds):
            if w != cur:
                nxt = wds[i:i + confirm]
                if len(nxt) == confirm and len(set(nxt)) == 1:
                    changes.append({"symbol": sym, "kind": kind, "last_old": g["expiry"].iloc[i - 1] if i else pd.NaT,
                                    "old_wd": WD[cur] if cur < 5 else str(cur),
                                    "first_new": g["expiry"].iloc[i], "new_wd": WD[w] if w < 5 else str(w)})
                    cur = w
            sched.append(cur)
        g["sched_wd"] = sched
        g["shifted"] = g["wd"] != g["sched_wd"]
        out.append(g)
    flagged = pd.concat(out, ignore_index=True) if out else exp.assign(sched_wd=[], shifted=[])
    return flagged, pd.DataFrame(changes)


def week_start(d) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(d)
    return (d - pd.to_timedelta(d.weekday, unit="D")).normalize()


def last_weekly(flagged: pd.DataFrame, symbol: str) -> pd.Timestamp:
    w = flagged[(flagged["symbol"] == symbol) & (flagged["kind"] == "weekly")]
    return w["expiry"].max()


def post_window(flagged: pd.DataFrame, key: str, weeks: int | None = POST_WEEKS) -> tuple:
    """Post window of a treated index: from the Monday after the week of its last weekly
    expiry, for ``weeks`` calendar weeks (to the Friday), or to POST_END_LONG if weeks is None."""
    lw = last_weekly(flagged, TREATED[key][1])
    start = week_start([lw])[0] + pd.Timedelta(days=7)
    end = POST_END_LONG if weeks is None else start + pd.Timedelta(weeks=weeks) - pd.Timedelta(days=3)
    return start, end


def trading_days(official: pd.DatetimeIndex) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(official)
    return d[(d.weekday < 5) & ~d.isin(list(MUHURAT_DAYS))]


def week_calendar(days: pd.DatetimeIndex, flagged: pd.DataFrame) -> pd.DataFrame:
    """Per calendar week: number of trading days, which weekdays trade, and for each of NIFTY,
    BANKNIFTY, FINNIFTY and MIDCPNIFTY whether the week holds a monthly expiry or a
    holiday-shifted expiry of that symbol (columns monthly_SYM, shifted_SYM)."""
    weeks = week_start(days).unique().sort_values()
    cal = pd.DataFrame(index=weeks)
    cal.index.name = "week"
    ws = pd.Series(week_start(days), index=days)
    cal["n_days"] = ws.value_counts().reindex(weeks).fillna(0).astype(int)
    for i, w in enumerate(WD):
        cal[f"has_{w}"] = cal.index.isin(ws[days.weekday == i].values)
    for s in CAL_SYMBOLS:
        e = flagged[flagged["symbol"] == s]
        cal[f"monthly_{s}"] = cal.index.isin(week_start(e.loc[e["kind"] == "monthly", "expiry"]))
        cal[f"shifted_{s}"] = cal.index.isin(week_start(e.loc[e["shifted"], "expiry"]))
    return cal


def clean_mask(cal: pd.DataFrame, symbols) -> pd.Series:
    """True for weeks without a monthly or holiday-shifted expiry of any of ``symbols``."""
    bad = np.zeros(len(cal), bool)
    for s in symbols:
        bad |= cal[f"monthly_{s}"].to_numpy() | cal[f"shifted_{s}"].to_numpy()
    return pd.Series(~bad, index=cal.index)


def usable_weeks(cal: pd.DataFrame, weekday: int, lo, hi, symbols=("NIFTY",), min_other: int = MIN_OTHER) -> pd.DatetimeIndex:
    """Weeks with Monday in [lo, hi] that are clean for ``symbols`` (Nifty and the treated
    index), whose target weekday is a trading day and that have at least ``min_other`` other
    trading days."""
    c = cal[(cal.index >= pd.Timestamp(lo)) & (cal.index <= pd.Timestamp(hi))]
    ok = clean_mask(c, symbols) & c[f"has_{WD[weekday]}"] & (c["n_days"] - 1 >= min_other)
    return c.index[ok.to_numpy()]


def cal_symbols(key: str) -> tuple:
    """Symbols whose monthly or shifted weeks are dropped for treated unit ``key``."""
    return ("NIFTY", TREATED[key][1]) if key in TREATED else ("NIFTY",)


# ---------------------------------------------------------------------------
# 4. Donor rule (pre-registration section 3)
# ---------------------------------------------------------------------------


def classify_name(name: str) -> str:
    """'non-equity', 'broad', 'strategy' or 'sector/theme' from the index name (step D3)."""
    k = name_key(name)
    for cat, pats in (("non-equity", NON_EQUITY), ("broad", BROAD), ("strategy", STRATEGY)):
        if any(re.search(p, k) for p in pats):
            return cat
    return "sector/theme"


def complete_and_valid(panel: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per index key: days present with all four prices, days with High > Low, days with Open and
    Close inside [Low, High]; 'complete' if all three equal the number of days (step D1, D2)."""
    p = panel[panel["date"].isin(days)].copy()
    px = p[["open", "high", "low", "close"]]
    p["has_prices"] = px.notna().all(axis=1) & (px > 0).all(axis=1)
    p["hl_ok"] = p["has_prices"] & (p["high"] > p["low"])
    p["in_range"] = p["hl_ok"] & p["open"].between(p["low"], p["high"]) & p["close"].between(p["low"], p["high"])
    g = p.groupby("key")
    out = pd.DataFrame({"name": g["name"].last(), "days_present": g["date"].nunique(),
                        "days_with_prices": g["has_prices"].sum(), "days_high_gt_low": g["hl_ok"].sum(),
                        "days_open_close_in_range": g["in_range"].sum()})
    n = len(days)
    out["present_all_days"] = out["days_present"] == n
    out["complete"] = (out["days_with_prices"] == n) & (out["days_high_gt_low"] == n)
    out["consistent"] = out["complete"] & (out["days_open_close_in_range"] == n)
    return out


def donor_table(valid: pd.DataFrame, deriv_names: set[str]) -> pd.DataFrame:
    """Apply steps D3 to D5 to the output of complete_and_valid (steps D1, D2). One row per
    index: category, listed index derivatives, name overlap, and the final 'donor' flag."""
    t = valid.copy()
    t["category"] = t["name"].map(classify_name)
    t["has_derivatives"] = t.index.isin({name_key(n) for n in deriv_names})
    t["overlap"] = np.asarray(t.index.map(lambda key: any(re.search(p, key) for p in OVERLAP)), bool)
    t["donor"] = t["complete"] & (t["category"] == "sector/theme") & ~t["has_derivatives"] & ~t["overlap"]
    return t


# ---------------------------------------------------------------------------
# 5. Outcome (phase 2): week-relative log range-based variance
# ---------------------------------------------------------------------------

LOG4LN2 = np.log(4.0 * np.log(2.0))


def log_parkinson(high, low) -> np.ndarray:
    """log of Parkinson (1980) variance (ln(H/L))^2 / (4 ln 2); NaN unless High > Low > 0."""
    h, lo = np.asarray(high, float), np.asarray(low, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(h / lo)
        out = 2.0 * np.log(r) - LOG4LN2
    return np.where((lo > 0) & (h > lo), out, np.nan)


def log_garman_klass(open_, high, low, close) -> np.ndarray:
    """log of Garman and Klass (1980) variance 0.5 (ln H/L)^2 - (2 ln 2 - 1) (ln C/O)^2."""
    o, h, lo, c = (np.asarray(x, float) for x in (open_, high, low, close))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 0.5 * np.log(h / lo) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
        out = np.log(v)
    return np.where((lo > 0) & (h > lo) & (v > 0), out, np.nan)


def week_relative(daily: pd.DataFrame, weekday: int, weeks, min_other: int = MIN_OTHER,
                  exclude_days=None) -> pd.DataFrame:
    """Weeks x units: value on the target weekday minus the mean over the other trading days of
    the same calendar week. ``daily``: trading days x units (log variance). ``exclude_days``:
    dates never used in the baseline. A unit-week is NaN unless the target day and at least
    ``min_other`` baseline days are finite."""
    d = daily.copy()
    wk = week_start(d.index)
    excl = set(pd.DatetimeIndex(exclude_days)) if exclude_days is not None else set()
    rows = {}
    for w in pd.DatetimeIndex(weeks):
        sel = d[wk == w]
        tgt = sel[sel.index.weekday == weekday]
        if len(tgt) != 1:
            continue
        base = sel[(sel.index.weekday != weekday) & ~sel.index.isin(list(excl))]
        n_ok = base.notna().sum()
        val = tgt.iloc[0] - base.mean()
        val[n_ok < min_other] = np.nan
        rows[w] = val
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "week"
    return out.reindex(columns=daily.columns)


# ---------------------------------------------------------------------------
# 6. Synthetic control (Abadie, Diamond and Hainmueller 2010)
# ---------------------------------------------------------------------------


def simplex_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """argmin ||y - X w||^2 subject to w >= 0 and sum(w) = 1. X: (T, J), y: (T,).

    Start from a non-negative least squares solution with the adding-up constraint as a
    heavily weighted extra row, then polish with SLSQP; the better of the two is kept.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    T, J = X.shape
    if J == 1:
        return np.ones(1)
    scale = max(float(np.abs(X).max()), float(np.abs(y).max()), 1e-12)
    M = 1e3 * scale * np.sqrt(T)
    w0, _ = optimize.nnls(np.vstack([X, M * np.ones((1, J))]), np.concatenate([y, [M]]), maxiter=100 * J)
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(J, 1.0 / J)

    def f(w):
        r = y - X @ w
        return 0.5 * float(r @ r)

    def g(w):
        return -(X.T @ (y - X @ w))

    res = optimize.minimize(f, w0, jac=g, method="SLSQP", bounds=[(0.0, 1.0)] * J,
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(J)}],
                            options={"ftol": 1e-14, "maxiter": 1000})
    w1 = np.clip(res.x, 0.0, None)
    w1 = w1 / w1.sum() if w1.sum() > 0 else w0
    w = w1 if f(w1) <= f(w0) else w0
    w[w < 1e-10] = 0.0
    return w / w.sum()


def sc_fit(y: np.ndarray, X: np.ndarray, pre: np.ndarray, post: np.ndarray, demean: bool = True) -> dict:
    """Synthetic control for one treated series y (T,) from donors X (T, J).

    demean=True (primary): each series is centred on its own pre-period mean before the
    weights are fitted (Doudchenko and Imbens 2016; Ferman and Pinto 2021), so the estimate is
    the change in the gap from the pre-period to the post-period. demean=False: classic
    synthetic control, estimate = mean post-period gap.
    """
    y, X = np.asarray(y, float), np.asarray(X, float)
    pre, post = np.asarray(pre, bool), np.asarray(post, bool)
    if demean:
        yc, Xc = y - y[pre].mean(), X - X[pre].mean(axis=0)
    else:
        yc, Xc = y, X
    w = simplex_weights(Xc[pre], yc[pre])
    gap = yc - Xc @ w
    r_pre = float(np.sqrt(np.mean(gap[pre] ** 2)))
    r_post = float(np.sqrt(np.mean(gap[post] ** 2)))
    raw_gap = y - X @ w
    return {"w": w, "gap": gap, "effect": float(gap[post].mean()), "pre_gap_mean": float(gap[pre].mean()),
            "rmspe_pre": r_pre, "rmspe_post": r_post, "ratio": r_post / r_pre if r_pre > 0 else np.inf,
            "level_gap_pre": float(raw_gap[pre].mean()), "level_gap_post": float(raw_gap[post].mean())}


def placebo_in_space(Y: pd.DataFrame, treated: str, donors: list, pre, post, demean: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fit the treated unit on all donors and each donor on the other donors (the treated unit is
    never a donor). Returns (one row per unit, the treated unit's fit).

    p_ratio: share of the J + 1 units whose post/pre RMSPE ratio is at least the treated one's
    (Abadie et al. 2010). p_neg: share whose standardised effect (effect / pre RMSPE) is at most
    the treated one's (one-sided, prediction: negative)."""
    rows, fits = [], {}
    for u in [treated] + list(donors):
        pool = [d for d in donors if d != u]
        fit = sc_fit(Y[u].to_numpy(), Y[pool].to_numpy(), pre, post, demean)
        fits[u] = fit
        rows.append({"unit": u, "treated": u == treated, "effect": fit["effect"], "rmspe_pre": fit["rmspe_pre"],
                     "rmspe_post": fit["rmspe_post"], "ratio": fit["ratio"],
                     "std_effect": fit["effect"] / fit["rmspe_pre"] if fit["rmspe_pre"] > 0 else np.nan,
                     "n_donors_used": int((fit["w"] > 0).sum())})
    tab = pd.DataFrame(rows)
    t = tab.iloc[0]
    tab.attrs["p_ratio"] = float((tab["ratio"] >= t["ratio"] - 1e-12).mean())
    tab.attrs["p_neg"] = float((tab["std_effect"] <= t["std_effect"] + 1e-12).mean())
    tab.attrs["p_pos"] = float((tab["std_effect"] >= t["std_effect"] - 1e-12).mean())
    tab.attrs["n_units"] = len(tab)
    return tab, fits[treated]


def test_inversion(Y: pd.DataFrame, treated: str, donors: list, pre, post, placebo: pd.DataFrame,
                   grid: np.ndarray, demean: bool = True, alpha: float = ALPHA) -> pd.DataFrame:
    """Firpo and Possebom (2018) style confidence set for a constant effect a: subtract a from
    the treated unit's post-period outcomes, refit, and recompute its RMSPE-ratio p-value
    against the unchanged placebo ratios. The set is every a with p(a) > alpha."""
    post = np.asarray(post, bool)
    other = placebo.loc[~placebo["treated"], "ratio"].to_numpy()
    y0 = Y[treated].to_numpy()
    X = Y[donors].to_numpy()
    rows = []
    for a in grid:
        fit = sc_fit(y0 - a * post, X, pre, post, demean)
        p = (1 + np.sum(other >= fit["ratio"] - 1e-12)) / (len(other) + 1)
        rows.append({"a": float(a), "ratio": fit["ratio"], "p": float(p), "in_set": bool(p > alpha)})
    return pd.DataFrame(rows)


def confidence_set(inv: pd.DataFrame) -> tuple[float, float, bool]:
    s = inv.loc[inv["in_set"], "a"]
    if s.empty:
        return np.nan, np.nan, False
    contiguous = bool(inv.loc[s.index.min():s.index.max(), "in_set"].all())
    return float(s.min()), float(s.max()), contiguous


def hac_mean_diff(d: np.ndarray, post: np.ndarray, lags: int = 4) -> tuple[float, float, float]:
    """OLS of d on [1, post] with Newey-West (Bartlett) standard errors: (coef, se, two-sided p)."""
    d, post = np.asarray(d, float), np.asarray(post, float)
    X = np.column_stack([np.ones_like(d), post])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ d
    u = d - X @ b
    xu = X * u[:, None]
    S = xu.T @ xu
    for L in range(1, lags + 1):
        wgt = 1.0 - L / (lags + 1.0)
        G = xu[L:].T @ xu[:-L]
        S += wgt * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(V[1, 1]))
    tval = b[1] / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(tval), df=len(d) - 2)) if np.isfinite(tval) else np.nan
    return float(b[1]), se, p


def simple_did(Y: pd.DataFrame, treated: str, donors: list, pre, post) -> dict:
    """Cross-check: treated minus the equal-weight donor average; DiD = post mean - pre mean.
    HAC p (two-sided) and an in-space rank p (one-sided, negative) where each donor is compared
    with the average of the other donors."""
    pre, post = np.asarray(pre, bool), np.asarray(post, bool)
    keep = pre | post

    def did_of(u, pool):
        d = Y[u].to_numpy() - Y[pool].to_numpy().mean(axis=1)
        return float(d[post].mean() - d[pre].mean()), d

    est, d = did_of(treated, donors)
    coef, se, p_hac = hac_mean_diff(d[keep], post[keep])
    plac = [did_of(u, [v for v in donors if v != u])[0] for u in donors]
    p_rank = (1 + np.sum(np.asarray(plac) <= est + 1e-12)) / (len(donors) + 1)
    return {"did": est, "hac_se": se, "hac_p": p_hac, "rank_p_neg": float(p_rank), "n_pre": int(pre.sum()),
            "n_post": int(post.sum())}


def pooled_test(tables: list) -> dict:
    """Joint one-sided test over several treated units (in the spirit of Cavallo, Galiani, Noy and
    Pantano 2013). Statistic: the mean over treated units of the standardised effect
    (effect / pre RMSPE). Placebo distribution: the same mean for every combination that takes
    one donor placebo from each unit's table. p = share of all combinations, the real one
    included, whose mean is at most the real one."""
    real = np.mean([t.loc[t["treated"], "std_effect"].iloc[0] for t in tables])
    plac = [t.loc[~t["treated"], "std_effect"].to_numpy() for t in tables]
    grid = np.zeros(1)
    for v in plac:
        grid = (grid[:, None] + v[None, :]).ravel()
    grid = grid / len(plac)
    p = (1 + np.sum(grid <= real + 1e-12)) / (len(grid) + 1)
    return {"mean_std_effect": float(real), "p_neg": float(p), "n_combinations": int(len(grid))}


def holm(pvals: dict) -> dict:
    """Holm (1979) step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def verdict(effect: float, p: float, level_pre: float | None = None, ci: tuple | None = None,
            alpha: float = ALPHA) -> str:
    """Decision rule of the pre-registration, section 7."""
    if p <= alpha and effect < 0:
        return "support"
    if p <= alpha and effect > 0:
        return "evidence against"
    if level_pre is not None and level_pre <= 0:
        return "no support (no pre-period premium over the synthetic control to remove)"
    if level_pre is not None and ci is not None and np.isfinite(ci[0]) and -level_pre < ci[0]:
        return "no support (removal of the whole pre-period premium is ruled out)"
    return "no support (too imprecise to tell)"


# ---------------------------------------------------------------------------
# 7. Power on simulated data (phase 1; no real data)
# ---------------------------------------------------------------------------


def simulate_panel(rng, J: int, T0: int, T1: int, effect: float = 0.0, factor_sd: float = 0.5,
                   noise_sd: float = 1.0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Weekly outcomes for a treated unit 'T' and donors 'D0'..: y = loading x common factor +
    noise, effect added to the treated unit's post weeks. Units: noise standard deviations."""
    T = T0 + T1
    f = rng.normal(0, factor_sd, T)
    lam = rng.uniform(0.5, 1.5, J + 1)
    Y = lam[None, :] * f[:, None] + rng.normal(0, noise_sd, (T, J + 1))
    post = np.arange(T) >= T0
    Y[post, 0] += effect
    cols = ["T"] + [f"D{j}" for j in range(J)]
    return pd.DataFrame(Y, columns=cols), ~post, post


def power_curve(J: int, T0: int, T1: int, effects, n_sim: int = 400, seed: int = 0, alpha: float = ALPHA) -> pd.DataFrame:
    rows = []
    for k, eff in enumerate(effects):
        rng = np.random.default_rng([MASTER_SEED, 96, seed, k])
        rej_ratio = rej_neg = 0
        rmspe = []
        for _ in range(n_sim):
            Y, pre, post = simulate_panel(rng, J, T0, T1, -abs(eff))
            donors = [c for c in Y.columns if c != "T"]
            tab, fit = placebo_in_space(Y, "T", donors, pre, post, demean=True)
            rej_ratio += (tab.attrs["p_ratio"] <= alpha) and fit["effect"] < 0
            rej_neg += tab.attrs["p_neg"] <= alpha
            rmspe.append(fit["rmspe_pre"])
        rows.append({"effect_sd_units": -abs(eff), "J": J, "T0": T0, "T1": T1, "n_sim": n_sim,
                     "power_ratio_test": rej_ratio / n_sim, "power_signed_test": rej_neg / n_sim,
                     "median_rmspe_pre": float(np.median(rmspe))})
    return pd.DataFrame(rows)


def mde(pc: pd.DataFrame, col: str, target: float = 0.8) -> float:
    """Smallest |effect| (noise SD units) at which ``col`` reaches ``target``, by linear
    interpolation between grid points; NaN if the grid never reaches it."""
    x, y = pc["effect_sd_units"].abs().to_numpy(), pc[col].to_numpy()
    o = np.argsort(x)
    x, y = x[o], y[o]
    for i in range(1, len(x)):
        if y[i] >= target > y[i - 1]:
            return float(x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return float(x[0]) if len(y) and y[0] >= target else np.nan


def brownian_noise_floor(n_days: int = 20000, n_steps: int = 2000, n_other: int = 4, seed: int = 7) -> dict:
    """Standard deviation of the daily log Parkinson variance, and of the week-relative value
    (target day minus the mean of ``n_other`` other days), when the index is a driftless
    Brownian motion with constant volatility observed at ``n_steps`` points a day. SIMULATED;
    real data can only be noisier (volatility changes within the week, jumps)."""
    rng = np.random.default_rng([MASTER_SEED, 96, seed])
    x = np.cumsum(rng.normal(0, 1 / np.sqrt(n_steps), (n_days, n_steps)), axis=1)
    x = np.concatenate([np.zeros((n_days, 1)), x], axis=1)
    lp = log_parkinson(np.exp(x.max(axis=1)), np.exp(x.min(axis=1)))
    sd_day = float(np.std(lp, ddof=1))
    k = n_days // (n_other + 1)
    blocks = lp[: k * (n_other + 1)].reshape(k, n_other + 1)
    rel = blocks[:, 0] - blocks[:, 1:].mean(axis=1)
    return {"sd_daily_log_parkinson": sd_day, "sd_week_relative": float(np.std(rel, ddof=1)),
            "bias_of_log_parkinson": float(lp.mean())}


def cmd_power(a) -> None:
    effects = [float(x) for x in a.effects.split(",")]
    fl = brownian_noise_floor()
    print(f"Noise floor under a constant-volatility Brownian motion (SIMULATED): SD of daily log Parkinson "
          f"{fl['sd_daily_log_parkinson']:.3f}; SD of the week-relative value with 4 other days "
          f"{fl['sd_week_relative']:.3f} log points")
    t0 = time.time()
    pc = power_curve(a.J, a.T0, a.T1, effects, n_sim=a.n_sim)
    print(f"Power of the primary test on SIMULATED data (J={a.J} donors, T0={a.T0} pre weeks, "
          f"T1={a.T1} post weeks, alpha={ALPHA}, smallest attainable p = 1/{a.J + 1} = {1 / (a.J + 1):.4f}); "
          f"{time.time() - t0:.0f}s")
    print(pc.to_string(index=False))
    r0 = float(pc.loc[pc["effect_sd_units"].abs().idxmin(), "median_rmspe_pre"])
    for col in ("power_ratio_test", "power_signed_test"):
        m = mde(pc, col)
        print(f"MDE80 {col}: {m:.2f} noise SD = {m / r0:.2f} x the median pre-period RMSPE under no effect ({r0:.3f})")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        pc.to_csv(Path(a.out) / f"power_J{a.J}_T0{a.T0}_T1{a.T1}.csv", index=False)


# ---------------------------------------------------------------------------
# 8. Coverage (phase 1)
# ---------------------------------------------------------------------------


def build_calendar(raw: Path, full: Path):
    official = official_days(raw)
    days = trading_days(official)
    fo_local = load_fo([raw / "extra_2022" / "bhavcopy", raw / "bhavcopy"])
    fo_full = load_fo([full / "fo_index"])
    # NIFTY and BANKNIFTY from the local files (whole span); other symbols from the full files
    fo = pd.concat([fo_local[fo_local["symbol"].isin(["NIFTY", "BANKNIFTY"])],
                    fo_full[~fo_full["symbol"].isin(["NIFTY", "BANKNIFTY"])]], ignore_index=True)
    exp = expiry_table(fo)
    flagged, changes = flag_schedule(exp)
    cal = week_calendar(days, flagged)
    return {"official": official, "days": days, "fo_full": fo_full, "fo_local": fo_local, "flagged": flagged,
            "changes": changes, "cal": cal}


def windows_for(key: str, flagged: pd.DataFrame) -> dict:
    ps, pe = post_window(flagged, key)
    _, pe_long = post_window(flagged, key, None)
    return {"pre": (PRE_START, PRE_END), "post": (ps, pe), "post_long": (ps, pe_long),
            "pre_alt": (PRE_START, PRE_END_ALT), "fake_pre": (PRE_START, FAKE_TREATMENT - pd.Timedelta(days=1)),
            "fake_post": (FAKE_TREATMENT, PRE_END)}


def cmd_coverage(a) -> None:
    raw, full = Path(a.raw), Path(a.full)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    C = build_calendar(raw, full)
    days, flagged, cal, fo_full = C["days"], C["flagged"], C["cal"], C["fo_full"]
    study_days = days[(days >= STUDY_START) & (days <= STUDY_END)]
    lines.append(f"Official NSE trading days (index_daily.csv): {len(C['official'])}, "
                 f"{C['official'].min().date()} to {C['official'].max().date()}; Monday to Friday without Muhurat "
                 f"sessions: {len(days)}. Muhurat days dropped: {[d.date().isoformat() for d in MUHURAT_DAYS if d in C['official']]}.")
    lines.append(f"Study window for donor completeness: {STUDY_START.date()} to {STUDY_END.date()}, {len(study_days)} trading days.")

    # index files
    panel = load_index_panel(full / "index_close_full")
    file_days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    miss = [d.date().isoformat() for d in C["official"] if d not in set(file_days)]
    lines.append(f"Full index files read: {len(file_days)} days, {file_days.min().date()} to {file_days.max().date()}; "
                 f"official days without a file: {miss}")
    spell = panel.groupby("key")["name"].agg(lambda s: sorted(set(s)))
    multi = spell[spell.map(len) > 1]
    lines.append(f"Distinct index names (case and spacing folded): {panel['key'].nunique()}; "
                 f"names spelled more than one way: {dict(multi)}")
    lines.append(f"Index names present on the first and the last study day: "
                 f"{panel.loc[panel['date'] == study_days.min(), 'key'].nunique()} and "
                 f"{panel.loc[panel['date'] == study_days.max(), 'key'].nunique()}")

    # derivatives
    fo_win = fo_full[(fo_full["date"] >= STUDY_START) & (fo_full["date"] <= STUDY_END)]
    sym = fo_win.groupby("symbol").agg(first=("date", "min"), last=("date", "max"), days=("date", "nunique"))
    fo_days = set(fo_win["date"])
    lines.append(f"F&O bhavcopies (full, index rows) read: {fo_full['date'].nunique()} days, "
                 f"{fo_full['date'].min().date()} to {fo_full['date'].max().date()}; study trading days with a "
                 f"bhavcopy: {sum(d in fo_days for d in study_days)} of {len(study_days)}; other bhavcopy days in the "
                 f"study window: {sorted(d.date().isoformat() for d in fo_days - set(study_days))}")
    fp = data_fingerprint(full)
    lines.append(f"Data fingerprint (SHA-256 of the downloaded files): {fp['sha256']} "
                 f"({fp['n_index_files']} index files, {fp['n_fo_files']} F&O files)")
    xc = crosscheck_local(raw, panel, fo_full, C["fo_local"])
    lines.append(f"Downloaded files against the filtered copies in data/raw/: {xc}")
    sx = sensex_rule_days(C["days"])
    sx = sx[(sx >= STUDY_START) & (sx <= STUDY_END)]
    lines.append(f"Reported Sensex weekly expiry days in the study window (rule, for S5 only): {len(sx)}; weekdays "
                 f"{pd.Series(sx.weekday).map(lambda x: WD[x]).value_counts().to_dict()}")
    lines.append("Index-derivative symbols in the study window:\n" + sym.to_string())
    unknown = sorted(set(sym.index) - set(DERIV_SYMBOL_TO_INDEX))
    if unknown:
        raise SystemExit(f"unmapped index-derivative symbols {unknown}: add them to DERIV_SYMBOL_TO_INDEX")
    deriv_names = {DERIV_SYMBOL_TO_INDEX[s] for s in sym.index}

    # donor rule
    valid = complete_and_valid(panel, study_days)
    tab = donor_table(valid, deriv_names)
    n0 = len(tab)
    n1 = int(tab["present_all_days"].sum())
    n2 = int(tab["complete"].sum())
    n2b = int(tab["consistent"].sum())
    cats = tab[tab["complete"]].groupby("category").size().to_dict()
    lines.append(f"Donor rule counts: D0 names seen on any study day {n0}; D1 present on all {len(study_days)} "
                 f"days {n1}; D2 all four prices on every day and High > Low on every day {n2} "
                 f"(of which Open and Close inside [Low, High] on every day: {n2b}); D3 categories among D2: {cats}; "
                 f"D3 sector/theme with index derivatives: "
                 f"{sorted(tab.loc[tab['complete'] & (tab['category'] == 'sector/theme') & tab['has_derivatives'], 'name'])}")
    dn = tab[tab["donor"]]
    ov = tab[tab["complete"] & (tab["category"] == "sector/theme") & ~tab["has_derivatives"] & tab["overlap"]]
    lines.append(f"D5 removed for name overlap with a treated index: {sorted(ov['name'])}")
    lines.append(f"Donor pool (all three treated units): J = {len(dn)}: {sorted(dn['name'])}")
    lines.append("Category of every index that passed D2: " + "; ".join(
        f"{c}: {sorted(tab.loc[tab['complete'] & (tab['category'] == c), 'name'])}"
        for c in ("broad", "strategy", "non-equity")))
    tab.sort_values(["category", "name"]).to_csv(out / "donor_table.csv")
    incomplete = tab[~tab["complete"]].sort_values("days_present", ascending=False)
    incomplete[["name", "category", "days_present", "days_with_prices", "days_high_gt_low"]].to_csv(out / "incomplete_indices.csv")

    # treated units present?
    for k, (nm, s, wd) in TREATED.items():
        r = tab.loc[name_key(nm)] if name_key(nm) in tab.index else None
        lines.append(f"Treated {k} {nm}: complete on all study days: {bool(r is not None and r['complete'])}")
    rn = tab.loc[name_key(NIFTY_NAME)] if name_key(NIFTY_NAME) in tab.index else None
    lines.append(f"Nifty 50 (descriptive check only) complete: {bool(rn is not None and rn['complete'])}")

    # calendars
    lines.append("Schedule changes detected (3 consecutive expiries on a new weekday):\n" + C["changes"].to_string(index=False))
    for s in CAL_SYMBOLS:
        f = flagged[(flagged["symbol"] == s) & (flagged["expiry"] >= PRE_START - pd.Timedelta(days=60))
                    & (flagged["expiry"] <= STUDY_END)]
        wk = f[f["kind"] == "weekly"]
        lines.append(f"{s}: expiries {f['expiry'].min().date()} to {f['expiry'].max().date()} in the files read; weekly "
                     f"{len(wk)} (weekdays {wk['wd'].map(lambda x: WD[x]).value_counts().to_dict()}), monthly "
                     f"{int((f['kind'] == 'monthly').sum())} (weekdays {f.loc[f['kind'] == 'monthly', 'wd'].map(lambda x: WD[x]).value_counts().to_dict()}), "
                     f"holiday-shifted {int(f['shifted'].sum())}: {[d.date().isoformat() for d in f.loc[f['shifted'], 'expiry']]}; "
                     f"last weekly {last_weekly(flagged, s).date() if s != 'NIFTY' else 'n/a (kept)'}")
    flagged.to_csv(out / "expiry_calendar.csv", index=False)
    cal.to_csv(out / "week_calendar.csv")

    # week counts per treated unit
    rows = []
    for k, (nm, s, wd) in TREATED.items():
        W = windows_for(k, flagged)
        sched = flagged[(flagged["symbol"] == s) & (flagged["kind"] == "weekly")]
        exp_days = set(sched["expiry"])
        for wname, (lo, hi) in W.items():
            uw = usable_weeks(cal, wd, lo, hi, cal_symbols(k))
            allw = cal[(cal.index >= lo) & (cal.index <= hi)]
            clean = clean_mask(allw, cal_symbols(k))
            tgt_days = [w + pd.Timedelta(days=wd) for w in uw]
            n_exp = sum(d in exp_days for d in tgt_days)
            rows.append({"treated": k, "weekday": WD[wd], "window": wname, "from": lo.date(), "to": hi.date(),
                         "calendar_weeks": len(allw), "monthly_or_shifted_weeks": int((~clean).sum()),
                         "target_day_not_trading": int((clean & ~allw[f"has_{WD[wd]}"]).sum()),
                         "usable_weeks": len(uw), "usable_weeks_with_own_weekly_expiry_on_target_day": n_exp,
                         "usable_weeks_if_all_four_symbols_cleaned": len(usable_weeks(cal, wd, lo, hi, CAL_SYMBOLS))})
    wc = pd.DataFrame(rows)
    wc.to_csv(out / "week_counts.csv", index=False)
    lines.append("Usable weeks per treated unit and window (calendar only):\n" + wc.to_string(index=False))
    # the Nifty-Wednesday descriptive check uses the BN weeks
    text = "\n".join(lines)
    (out / "coverage.txt").write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------------------
# 9. Analysis (phase 2 only; refuses to run unless --phase2 is given)
# ---------------------------------------------------------------------------


def daily_log_variance(panel: pd.DataFrame, keys: list, days: pd.DatetimeIndex, measure: str) -> pd.DataFrame:
    p = panel[panel["key"].isin(keys) & panel["date"].isin(days)]
    if measure == "parkinson":
        v = log_parkinson(p["high"], p["low"])
    elif measure == "garman_klass":
        v = log_garman_klass(p["open"], p["high"], p["low"], p["close"])
    else:
        raise ValueError(measure)
    wide = p.assign(v=v).pivot(index="date", columns="key", values="v")
    return wide.reindex(index=days, columns=keys)


def run_unit(daily: pd.DataFrame, cal: pd.DataFrame, key: str, treated_col: str, donors: list, W: dict,
             post_name: str = "post", pre_name: str = "pre", demean: bool = True, min_other: int = MIN_OTHER,
             exclude_days=None, grid=None) -> dict:
    wd = TREATED[key][2]
    pre_w = usable_weeks(cal, wd, *W[pre_name], symbols=cal_symbols(key), min_other=min_other)
    post_w = usable_weeks(cal, wd, *W[post_name], symbols=cal_symbols(key), min_other=min_other)
    weeks = pre_w.append(post_w)
    Y = week_relative(daily[[treated_col] + donors], wd, weeks, min_other=min_other, exclude_days=exclude_days)
    Y = Y.dropna(how="any")
    pre = Y.index.isin(pre_w)
    post = Y.index.isin(post_w)
    tab, fit = placebo_in_space(Y, treated_col, donors, pre, post, demean)
    res = {"Y": Y, "pre": pre, "post": post, "placebo": tab, "fit": fit, "did": simple_did(Y, treated_col, donors, pre, post)}
    if grid is not None:
        inv = test_inversion(Y, treated_col, donors, pre, post, tab, grid, demean)
        res["inversion"] = inv
        res["ci"] = confidence_set(inv)
    return res


def summary_row(label: str, key: str, r: dict) -> dict:
    t = r["placebo"]
    ci = r.get("ci", (np.nan, np.nan, False))
    inv = r.get("inversion")
    edge = bool(inv is not None and np.isfinite(ci[0])
                and (ci[0] <= inv["a"].min() + 1e-9 or ci[1] >= inv["a"].max() - 1e-9))
    mult = MDE80_PER_RMSPE.get(key) if label in ("primary", "secondary") else None
    return {"analysis": label, "treated": key, "n_pre": int(r["pre"].sum()), "n_post": int(r["post"].sum()),
            "J": len(t) - 1, "min_p": 1.0 / len(t), "effect": r["fit"]["effect"],
            "level_gap_pre": r["fit"]["level_gap_pre"], "level_gap_post": r["fit"]["level_gap_post"],
            "rmspe_pre": r["fit"]["rmspe_pre"], "rmspe_post": r["fit"]["rmspe_post"], "ratio": r["fit"]["ratio"],
            "p_ratio": t.attrs["p_ratio"], "p_neg": t.attrs["p_neg"], "ci_low": ci[0], "ci_high": ci[1],
            "ci_contiguous": ci[2], "ci_at_grid_edge": edge, "did": r["did"]["did"], "did_hac_p": r["did"]["hac_p"],
            "did_rank_p_neg": r["did"]["rank_p_neg"], "n_donors_with_weight": int((r["fit"]["w"] > 0).sum()),
            "mde80_ratio_test_log_points": mult * r["fit"]["rmspe_pre"] if mult else np.nan}


def make_decision(res: pd.DataFrame, placebo_tables: list) -> dict:
    """Pre-registration sections 7 and 8: the primary verdict, the secondaries with Holm, the
    joint test, and the placebo-in-time flag."""
    prim = res[res["analysis"] == "primary"].iloc[0]
    sec = res[res["analysis"] == "secondary"].set_index("treated")
    hp = holm(sec["p_ratio"].to_dict())
    pt = res[res["analysis"].str.startswith("placebo in time") & (res["treated"] == PRIMARY)].iloc[0]
    v = verdict(prim["effect"], prim["p_ratio"], prim["level_gap_pre"], (prim["ci_low"], prim["ci_high"]))
    if v == "support" and pt["p_ratio"] <= ALPHA:
        v = "support, but the placebo in time also rejects (not reliable)"
    return {
        "primary": {"treated": PRIMARY, "effect": prim["effect"], "p_ratio": prim["p_ratio"], "min_p": prim["min_p"],
                    "p_signed_one_sided": prim["p_neg"], "ci_95": [prim["ci_low"], prim["ci_high"]],
                    "pre_period_premium_over_synthetic": prim["level_gap_pre"], "did_equal_weights": prim["did"],
                    "did_hac_p": prim["did_hac_p"], "placebo_in_time_p_ratio": pt["p_ratio"],
                    "placebo_in_time_effect": pt["effect"], "verdict": v},
        "secondaries": {k: {"effect": sec.loc[k, "effect"], "p_ratio": sec.loc[k, "p_ratio"], "holm_p": hp[k],
                            "smallest_possible_holm_p": min(1.0, len(sec) * sec.loc[k, "min_p"]),
                            "verdict": verdict(sec.loc[k, "effect"], hp[k])} for k in sec.index},
        "joint_test_all_three": pooled_test(placebo_tables),
    }


def fig_gaps(keep: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(keep), figsize=(5 * len(keep), 3.6), sharey=True)
    for ax, (k, r) in zip(np.atleast_1d(axes), keep.items()):
        Y, donors = r["Y"], list(r["Y"].columns[1:])
        for u in donors:
            pool = [d for d in donors if d != u]
            g = sc_fit(Y[u].to_numpy(), Y[pool].to_numpy(), r["pre"], r["post"])["gap"]
            ax.plot(Y.index, g, color="#bbbbbb", lw=0.7)
        ax.plot(Y.index, r["fit"]["gap"], color="#c23b8a", lw=1.8, label=f"{TREATED[k][0]} ({WD[TREATED[k][2]]})")
        ax.axvline(EFFECTIVE_DATE, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{TREATED[k][0]}: gap to its synthetic control", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analyse_all(dl: dict, cal: pd.DataFrame, flagged: pd.DataFrame, days: pd.DatetimeIndex,
                study_days: pd.DatetimeIndex, donors: list, grid=CI_GRID) -> tuple[pd.DataFrame, dict]:
    """Every pre-registered run (sections 7 to 10) for the three treated indices. ``dl``: daily
    log variance by measure ('parkinson', 'garman_klass'), trading days x index keys (the three
    treated indices, Nifty 50 and the donors). Returns (one row per run, the primary runs)."""
    rows, keep = [], {}
    for k in TREATED:
        tk = name_key(TREATED[k][0])
        W = windows_for(k, flagged)
        r = run_unit(dl["parkinson"], cal, k, tk, donors, W, grid=grid)
        keep[k] = r
        rows.append(summary_row("primary" if k == PRIMARY else "secondary", k, r))
        sens_runs = {
            "S1 long post window": dict(post_name="post_long"),
            "S2 Garman-Klass": dict(daily=dl["garman_klass"]),
            "S3 classic synthetic control (no demeaning)": dict(demean=False),
            "S4 pre-period ends 26 Jul 2024": dict(pre_name="pre_alt"),
            "S5 baseline without Nifty and Sensex expiry days": dict(exclude_days=nifty_sensex_days(flagged, days)),
            "S7 at least 3 other days": dict(min_other=3),
            "S8 baseline Thursday and Friday only": dict(exclude_days=thu_fri_only_exclusions(study_days, TREATED[k][2])),
            "placebo in time (fake date 1 Apr 2024)": dict(pre_name="fake_pre", post_name="fake_post"),
        }
        for label, kw in sens_runs.items():
            d = kw.pop("daily", dl["parkinson"])
            rows.append(summary_row(label, k, run_unit(d, cal, k, tk, donors, W, **kw)))
        top = donors[int(np.argmax(r["fit"]["w"]))]
        rows.append(summary_row(f"S6 without the largest-weight donor ({top})", k,
                                run_unit(dl["parkinson"], cal, k, tk, [d for d in donors if d != top], W)))
        if k == PRIMARY:
            rn = run_unit(dl["parkinson"], cal, k, name_key(NIFTY_NAME), donors, W)
            rows.append(summary_row("descriptive: Nifty 50 Wednesday as a fake treated unit", "NIFTY", rn))
    return pd.DataFrame(rows), keep


def cmd_analyse(a) -> None:
    if not a.phase2:
        raise SystemExit("analyse computes outcomes on real data; it runs only in phase 2 (pass --phase2).")
    raw, full = Path(a.raw), Path(a.full)
    fp = data_fingerprint(full)
    if fp["sha256"] != PREREG_DATA_SHA256 and not a.accept_new_data:
        raise SystemExit(f"data fingerprint {fp['sha256']} differs from the pre-registered {PREREG_DATA_SHA256}; "
                         "pass --accept-new-data only if the change is reported as a deviation.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    C = build_calendar(raw, full)
    days, flagged, cal = C["days"], C["flagged"], C["cal"]
    study_days = days[(days >= STUDY_START) & (days <= STUDY_END)]
    panel = load_index_panel(full / "index_close_full")
    fo_win = C["fo_full"][(C["fo_full"]["date"] >= STUDY_START) & (C["fo_full"]["date"] <= STUDY_END)]
    deriv_names = {DERIV_SYMBOL_TO_INDEX[s] for s in fo_win["symbol"].unique()}
    tab = donor_table(complete_and_valid(panel, study_days), deriv_names)
    donors = sorted(tab.index[tab["donor"]])
    units = [name_key(TREATED[k][0]) for k in TREATED] + [name_key(NIFTY_NAME)] + donors
    dl = {m: daily_log_variance(panel, units, study_days, m) for m in ("parkinson", "garman_klass")}
    res, keep = analyse_all(dl, cal, flagged, days, study_days, donors)
    res.to_csv(out / "results.csv", index=False)
    decision = make_decision(res, [keep[k]["placebo"] for k in TREATED])
    (out / "decision.json").write_text(json.dumps(decision, indent=1, default=float))
    for k, r in keep.items():
        r["Y"].assign(pre=r["pre"], post=r["post"], gap=r["fit"]["gap"]).to_csv(out / f"weekly_{k}.csv")
        r["placebo"].to_csv(out / f"placebo_space_{k}.csv", index=False)
        r["inversion"].to_csv(out / f"test_inversion_{k}.csv", index=False)
        pd.Series(r["fit"]["w"], index=r["Y"].columns[1:], name="weight").to_csv(out / f"weights_{k}.csv")
    tab.to_csv(out / "donor_table.csv")
    info = {"donors": [tab.loc[d, "name"] for d in donors], "J": len(donors), "n_index_files": int(panel["date"].nunique()),
            "data_fingerprint": fp, "prereg_fingerprint": PREREG_DATA_SHA256,
            "study_days": len(study_days), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__, "seconds": round(time.time() - t0, 1)}
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    try:
        fig_gaps(keep, out / "fig_gaps.png")
    except Exception as exc:  # the figure is a convenience; never fail the run on it
        print(f"figure skipped: {exc}")
    print(res.to_string(index=False))
    print(json.dumps(decision, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f}s")


def sensex_rule_days(days: pd.DatetimeIndex, cal_path: Path = SENSEX_CAL) -> pd.DatetimeIndex:
    """Reported Sensex weekly expiry days: inside each 'weekday_regime' row of
    data/external/sensex_expiry_calendar.csv, every scheduled weekday, moved back to the
    previous trading day when it is not one (the reported holiday rule; the same rule as
    experiments/sensex_confound.py). Reported, not verified (BSE files are not available)."""
    cal = pd.read_csv(cal_path, dtype=str, keep_default_na=False)
    td = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(days))))
    tdset = set(td)
    out = set()
    for _, r in cal[cal["fact_type"] == "weekday_regime"].iterrows():
        start = pd.Timestamp(r["date"])
        end = pd.Timestamp(r["date_end"]) if r["date_end"] else td.max()
        wd = WD.index(r["weekday"])
        s = start + pd.Timedelta(days=(wd - start.weekday()) % 7)
        while s <= min(end, td.max()):
            d = s
            while d not in tdset and d > s - pd.Timedelta(days=6):
                d -= pd.Timedelta(days=1)
            if d in tdset:
                out.add(d)
            s += pd.Timedelta(days=7)
    return pd.DatetimeIndex(sorted(out))


def nifty_sensex_days(flagged: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """S5: days with a Nifty expiry (bhavcopy) or a reported Sensex weekly expiry."""
    n = flagged.loc[flagged["symbol"] == "NIFTY", "expiry"]
    return pd.DatetimeIndex(sorted(set(n) | set(sensex_rule_days(days))))


def thu_fri_only_exclusions(days: pd.DatetimeIndex, weekday: int) -> pd.DatetimeIndex:
    """S8: baseline days other than Thursday and Friday (never the target day). For each
    treated index this removes the old expiry weekdays of the other two treated indices
    (Monday, Tuesday, Wednesday), which lost their weeklies on the same date."""
    d = pd.DatetimeIndex(days)
    return d[(d.weekday != weekday) & ~d.weekday.isin([3, 4])]


# ---------------------------------------------------------------------------
# 10. Command line
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    f.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    f.add_argument("--what", choices=["index", "fo", "both"], default="both")
    f.add_argument("--fo-start", default="2023-08-01")
    f.add_argument("--fo-end", default="2025-08-29")
    f.add_argument("--workers", type=int, default=3)
    f.add_argument("--pause", type=float, default=0.3)
    c = sub.add_parser("coverage")
    c.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    c.add_argument("--full", required=True, help="folder written by fetch")
    c.add_argument("--out", required=True)
    p = sub.add_parser("power")
    p.add_argument("--J", type=int, required=True)
    p.add_argument("--T0", type=int, required=True)
    p.add_argument("--T1", type=int, required=True)
    p.add_argument("--effects", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    p.add_argument("--n-sim", type=int, default=400)
    p.add_argument("--out", default="")
    an = sub.add_parser("analyse")
    an.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    an.add_argument("--full", required=True)
    an.add_argument("--out", default=str(ROOT / "results" / "synthetic_control"))
    an.add_argument("--phase2", action="store_true")
    an.add_argument("--accept-new-data", action="store_true",
                    help="run although the data fingerprint differs from the pre-registered one (a deviation)")
    a = ap.parse_args(argv)
    {"fetch": cmd_fetch, "coverage": cmd_coverage, "power": cmd_power, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
