"""Download free 1-minute index bars for Nifty 50 and Nifty Bank from public GitHub datasets.

NSE does not publish free historical 1-minute index bars, and broker APIs need
an account. Two community datasets on GitHub (both MIT licensed) cover what
this project needs:

* Nifty 50: github.com/technovusin/nifty50-historical-data, 1-minute candles
  from April 2017, one file per year to 2025 and one per month from 2026,
  updated daily. Columns Epoch, Timestamp (IST), Open, High, Low, Close
  (Volume only in 2026 files).
* Nifty Bank: github.com/sandeepkapri/BankNifty-Data, 1-minute candles from
  January 2015 to 22 April 2026 in one file. Columns Instrument, Date
  (DD-MM-YYYY), Time, Open, High, Low, Close.

These are third-party files, not NSE's. scripts/validate_minute_bars.py
checks every day against NSE's official daily open, high, low and close
(index_daily.csv from download_nse.py) before anything uses them.

    python scripts/fetch_minute_bars.py --out data/raw

Writes data/raw/nifty_1min.csv and data/raw/banknifty_1min.csv with columns
timestamp, open, high, low, close (IST, no time zone), plus a small
provenance file minute_bars_source.json (URLs, git commit, row counts).
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import pandas as pd
import requests

NIFTY_REPO = "technovusin/nifty50-historical-data"
BANK_REPO = "sandeepkapri/BankNifty-Data"
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def head_commit(repo: str) -> str:
    out = subprocess.run(["git", "ls-remote", f"https://github.com/{repo}", "HEAD"],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.split()[0] if out.stdout else "HEAD"


def get_csv(repo: str, ref: str, path: str) -> pd.DataFrame | None:
    r = requests.get(RAW.format(repo=repo, ref=ref, path=path), timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))


def nifty_paths(first_year: int, last_year: int, last_month: int) -> list[str]:
    paths = []
    for y in range(first_year, min(last_year, 2025) + 1):
        paths.append(f"1min/{y}/NIFTY50_1min_{y}.csv")
    for y in range(max(first_year, 2026), last_year + 1):
        for m in range(1, 13 if y < last_year else last_month + 1):
            paths.append(f"1min/{y}/NIFTY50_1min_{y}-{m:02d}.csv")
    return paths


def fetch_nifty(first_year: int, today: pd.Timestamp, ref: str | None = None) -> tuple[pd.DataFrame, dict]:
    ref = ref or head_commit(NIFTY_REPO)
    frames = []
    for p in nifty_paths(first_year, today.year, today.month):
        df = get_csv(NIFTY_REPO, ref, p)
        if df is not None:
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"timestamp": pd.to_datetime(df["Timestamp"]),
                        "open": df["Open"], "high": df["High"], "low": df["Low"], "close": df["Close"]})
    return out, {"repo": NIFTY_REPO, "commit": ref, "files": len(frames)}


def fetch_bank(ref: str | None = None) -> tuple[pd.DataFrame, dict]:
    ref = ref or head_commit(BANK_REPO)
    df = get_csv(BANK_REPO, ref, "bank-nifty-1m-data.csv")
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d-%m-%Y %H:%M:%S")
    out = pd.DataFrame({"timestamp": ts, "open": df["Open"], "high": df["High"], "low": df["Low"], "close": df["Close"]})
    return out, {"repo": BANK_REPO, "commit": ref, "files": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--first-year", type=int, default=2017)
    ap.add_argument("--nifty-ref", default=None, help="git commit of the Nifty dataset to use (default: latest); "
                    "pin it to reproduce a past run exactly (see data/MANIFEST.json)")
    ap.add_argument("--bank-ref", default=None, help="git commit of the Bank Nifty dataset to use (default: latest)")
    ap.add_argument("--until", default=None, help="drop bars after this date (for example 2026-11-02)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prov = {}
    for name, (df, info) in {"nifty": fetch_nifty(a.first_year, pd.Timestamp.today(), a.nifty_ref),
                             "banknifty": fetch_bank(a.bank_ref)}.items():
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        df = df[df["timestamp"].dt.year >= a.first_year]
        if a.until:
            df = df[df["timestamp"] < pd.Timestamp(a.until) + pd.Timedelta(days=1)]
        df.to_csv(out / f"{name}_1min.csv", index=False)
        info.update(rows=len(df), first=str(df["timestamp"].min()), last=str(df["timestamp"].max()),
                    days=int(df["timestamp"].dt.normalize().nunique()))
        prov[name] = info
        print(name, info, flush=True)
    json.dump(prov, open(out / "minute_bars_source.json", "w"), indent=2)


if __name__ == "__main__":
    main()
