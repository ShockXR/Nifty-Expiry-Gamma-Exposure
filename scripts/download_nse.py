"""Download NSE F&O bhavcopy, participant-wise OI and daily index closes.

Tested on 2 November 2026 from a cloud machine that can reach the NSE file
archives (nsearchives.nseindia.com) but not the main website
(www.nseindia.com). None of the files below need the main website or its
cookies.

    python scripts/download_nse.py --start 2023-01-01 --end 2026-10-31 --out data/raw

What it saves under --out:

* bhavcopy/fo_YYYYMMDD.csv.gz: the day's F&O bhavcopy, kept in NSE's own
  layout but filtered to index options and index futures of the symbols in
  --symbols (default NIFTY and BANKNIFTY). Full files are about 6 MB a day;
  the filtered ones are about 0.1 MB.
* participant_oi/fao_participant_oi_DDMMYYYY.csv: participant-wise OI, as published.
* index_close/ind_close_all_DDMMYYYY.csv: the day's official index values
  (open, high, low, close for every NSE index, including India VIX),
  filtered to the indices in --indices.
* india_vix.csv and index_daily.csv: built from index_close at the end.

Days that fail are listed at the end; weekends and exchange holidays are
expected to fail (NSE returns 404). The script is polite: a few requests at a
time, browser-like headers, retries with back-off, and it skips files already
on disk, so it can be stopped and restarted.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}
UDIFF_START = pd.Timestamp("2024-07-08")
ARCHIVE = "https://nsearchives.nseindia.com"


def bhavcopy_urls(d: pd.Timestamp) -> list[str]:
    """Candidate URLs, newest layout first (NSE served both for a while around July 2024)."""
    udiff = f"{ARCHIVE}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    legacy = f"{ARCHIVE}/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"
    return [udiff, legacy] if d >= UDIFF_START else [legacy, udiff]


def participant_url(d: pd.Timestamp) -> str:
    return f"{ARCHIVE}/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv"


def index_close_url(d: pd.Timestamp) -> str:
    return f"{ARCHIVE}/content/indices/ind_close_all_{d:%d%m%Y}.csv"


def fetch(session: requests.Session, url: str, retries: int = 4) -> bytes | None:
    for k in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** (k + 1))
    return None


def filter_bhavcopy(raw: bytes, symbols: set[str]) -> pd.DataFrame:
    """Keep index options and index futures for the given symbols, in NSE's own columns."""
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    if "INSTRUMENT" in df.columns:  # legacy layout
        keep = df["INSTRUMENT"].str.strip().isin(["OPTIDX", "FUTIDX"]) & df["SYMBOL"].str.strip().isin(symbols)
    else:  # UDiFF layout
        keep = df["FinInstrmTp"].str.strip().isin(["IDO", "IDF"]) & df["TckrSymb"].str.strip().isin(symbols)
    return df[keep]


def download_day(d: pd.Timestamp, out: Path, symbols: set[str], indices: set[str], pause: float) -> list[tuple]:
    s = requests.Session()
    failed = []
    bfile = out / "bhavcopy" / f"fo_{d:%Y%m%d}.csv.gz"
    if not bfile.exists():
        content = None
        for url in bhavcopy_urls(d):
            content = fetch(s, url)
            if content is not None:
                break
        if content is None:
            failed.append(("bhavcopy", d.date()))
        else:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                raw = z.read(z.namelist()[0])
            filter_bhavcopy(raw, symbols).to_csv(bfile, index=False, compression={"method": "gzip", "mtime": 0})
        time.sleep(pause)
    pfile = out / "participant_oi" / f"fao_participant_oi_{d:%d%m%Y}.csv"
    if not pfile.exists():
        content = fetch(s, participant_url(d))
        if content is None:
            failed.append(("participant", d.date()))
        else:
            pfile.write_bytes(content)
        time.sleep(pause)
    ifile = out / "index_close" / f"ind_close_all_{d:%d%m%Y}.csv"
    if not ifile.exists():
        content = fetch(s, index_close_url(d))
        if content is None:
            failed.append(("index_close", d.date()))
        else:
            df = pd.read_csv(io.BytesIO(content), dtype=str)
            df.columns = [c.strip() for c in df.columns]
            df = df[df["Index Name"].str.strip().str.upper().isin(indices)].drop_duplicates()
            df.to_csv(ifile, index=False)
        time.sleep(pause)
    return failed


def build_daily_tables(out: Path) -> None:
    """Combine the per-day index files into india_vix.csv and index_daily.csv."""
    files = sorted((out / "index_close").glob("ind_close_all_*.csv"))
    if not files:
        return
    frames = []
    for f in files:
        part = pd.read_csv(f, dtype=str)
        # The trade date is taken from the file name (ind_close_all_DDMMYYYY.csv). The "Index Date"
        # column inside is not reliable: some files from April 2023 write it month-first
        # (6 April 2023 appears as 04-06-2023), which day-first parsing turns into 4 June.
        part["date"] = pd.to_datetime(f.stem[-8:], format="%d%m%Y")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    names = {"Index Name": "index", "Open Index Value": "open", "High Index Value": "high",
             "Low Index Value": "low", "Closing Index Value": "close", "Volume": "volume",
             "Turnover (Rs. Cr.)": "turnover_cr"}
    df = df.rename(columns=names)
    cols = [c for c in names.values() if c in df.columns and c != "index"]
    for c in cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    df["index"] = df["index"].str.strip()
    tidy = df[["date", "index"] + cols].drop_duplicates(["date", "index"]).sort_values(["index", "date"])
    tidy.to_csv(out / "index_daily.csv", index=False)
    vix = tidy[tidy["index"].str.upper() == "INDIA VIX"][["date", "close"]]
    vix.to_csv(out / "india_vix.csv", index=False)
    print(f"index_daily.csv: {len(tidy)} rows; india_vix.csv: {len(vix)} days")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--symbols", nargs="+", default=["NIFTY", "BANKNIFTY"])
    ap.add_argument("--indices", nargs="+", default=["NIFTY 50", "NIFTY BANK", "INDIA VIX"])
    ap.add_argument("--workers", type=int, default=3, help="days fetched at the same time (keep this small)")
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between requests per worker")
    a = ap.parse_args()
    out = Path(a.out)
    for sub in ("bhavcopy", "participant_oi", "index_close"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    days = list(pd.bdate_range(a.start, a.end))
    symbols, indices = set(a.symbols), {x.upper() for x in a.indices}
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for k, res in enumerate(ex.map(lambda d: download_day(d, out, symbols, indices, a.pause), days), 1):
            failed.extend(res)
            if k % 50 == 0:
                print(f"{k}/{len(days)} days, {time.time() - t0:.0f}s, {len(failed)} misses", flush=True)
    build_daily_tables(out)
    pd.DataFrame(failed, columns=["file", "date"]).to_csv(out / "download_misses.csv", index=False)
    print(f"done; {len(failed)} files not found (weekends and holidays are expected to fail):")
    for f in failed[:30]:
        print("  ", *f)


if __name__ == "__main__":
    main()
