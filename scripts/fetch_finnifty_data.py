"""Download and pin the raw data for the Fin Nifty replication (P104, docs/prereg/P104_THIRD_INDEX_FORECAST.md).

Three pieces, all written under --raw (git-ignored; hashes go to data/MANIFEST.csv):

* finnifty_1min.csv: 1-minute Nifty Financial Services (FINNIFTY) index bars, columns
  timestamp, open, high, low, close (IST, no time zone), like nifty_1min.csv. Source: the 17
  files data/FINNIFTY_part1.csv ... part17.csv of the public GitHub repository
  ayush-roy-21/EigenTrade at commit 21fc3e72cb6b54f55af2ef9815de6a7c140ac0f8 (fetched from
  raw.githubusercontent.com). The repository states no licence and does not say where the bars
  came from, so they are a third-party dataset of unknown origin; scripts/validate_minute_bars.py
  checks every day against NSE's official daily high and low before anything uses them. Each
  downloaded file must have the SHA-256 recorded below, otherwise the script stops. The volume
  column of the source (zero before 2025, filled on some later days) is dropped, so that the
  bars carry the same columns as the Nifty and Bank Nifty bars.
  finnifty_1min_source.json records the repository, commit, files, hashes and row counts.
* finnifty/index_daily_finnifty.csv: NSE's official daily values of "Nifty Financial Services"
  (date, index, open, high, low, close), read from the daily index files already downloaded for
  P96 (nse_full/index_close_full/ind_close_all_DDMMYYYY.csv; date taken from the file name, see
  download_nse.py). Its dates are the official trading days of the run.
* finnifty/bhavcopy/fo_YYYYMMDD.csv.gz: for every official day from --start to --bhav-end, the
  day's F&O bhavcopy filtered to index options and index futures (all index symbols, NSE's own
  columns), as in nse_full/fo_index. Days already in nse_full/fo_index (1 August 2023 to
  29 August 2025, downloaded for P96) are copied from there; the others are downloaded from
  NSE's archive (the same URLs as download_nse.py).

    python scripts/fetch_finnifty_data.py --raw data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_nse import bhavcopy_urls, fetch, filter_bhavcopy  # noqa: E402

REPO = "ayush-roy-21/EigenTrade"
COMMIT = "21fc3e72cb6b54f55af2ef9815de6a7c140ac0f8"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
# SHA-256 of each source file at COMMIT (checked on 3 November 2026).
FILES = {
    "data/FINNIFTY_part1.csv": "daa29645ad574e4849a1a2e77bb67c6d4bedd55ad37df43313de35eb8582cd7d",
    "data/FINNIFTY_part2.csv": "c6f9093b95a2d3ff58ffc54b37053a8db83640e88aaa9e7f827365de1215ab8a",
    "data/FINNIFTY_part3.csv": "446867456cda69806197c5d61425d04a6c5947422147be97a199cbdf022bca90",
    "data/FINNIFTY_part4.csv": "7a20d1e3c2e94e210dd396fe5a224e20bcae6f29e2d3410b73d10d7eb19a4136",
    "data/FINNIFTY_part5.csv": "992ed10e9adb4daebd181defbc4961d819bf370e0c6e4fa5562777cf2a5eb4bb",
    "data/FINNIFTY_part6.csv": "2090e134def7c589246811582e3add2859a3647767af1865bcb5b13b980075e2",
    "data/FINNIFTY_part7.csv": "e00225f9ce21dc3775a37c833dcaa4aa76df3a79aa55f50fd0a64c7faa2023b3",
    "data/FINNIFTY_part8.csv": "40e6c226bf6a53274cba995f62c81bb17dfd9d2e4ba592d907875d8883a20e54",
    "data/FINNIFTY_part9.csv": "30fa771f708772bd4db769c773059bef698e3c1ccc3f6c7d29894e801d6ec20c",
    "data/FINNIFTY_part10.csv": "a8c87d92ba22bfe55df3e8852190a4c93e7eeac8f3bf2398d13e98ceb896d893",
    "data/FINNIFTY_part11.csv": "bde480ece7e5ed7fbd9ce48000cb9793b83dcde829860092de3928423cfa56ba",
    "data/FINNIFTY_part12.csv": "b078ab278660d505b9ff13a8793bd24ef106cb87b10b000750059acfff1ec4f7",
    "data/FINNIFTY_part13.csv": "b02c0f1e2efb6e69d5b9383e4e3588fb34dae12ddc52e8bb1f7b063109cbb926",
    "data/FINNIFTY_part14.csv": "e006f8e56ec7a793ccd8374d1fbec992723c7d2fd96d75aaebab0c7ed6236ce0",
    "data/FINNIFTY_part15.csv": "480df553a02269deddca2ccc6a4de4d44814fe2d564dd9a1d205cb512d05d7a4",
    "data/FINNIFTY_part16.csv": "0c18c6c1e4b70b9e4c1d11947a0f5b9cd5ec38b2e9257a64e982cb82ece45148",
    "data/FINNIFTY_part17.csv": "98d93030847f7bee98f4b9e4c93a6fffbeb9c2758cfda9b31d04f0241993b004",
}
INDEX_NAME = "Nifty Financial Services"
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}


def parse_bars(content: bytes) -> pd.DataFrame:
    """One source file to timestamp, open, high, low, close. Dates are written ="DD-MM-YY" (an Excel export)."""
    df = pd.read_csv(io.BytesIO(content))
    if set(df["symbol"].astype(str).str.strip()) != {"FINNIFTY"}:
        raise ValueError("unexpected symbol in source file")
    date = df["date"].astype(str).str.replace("=", "", regex=False).str.replace('"', "", regex=False)
    ts = pd.to_datetime(date + " " + df["time"].astype(str), format="%d-%m-%y %H:%M:%S")
    return pd.DataFrame({"timestamp": ts, "open": df["open"].astype(float), "high": df["high"].astype(float),
                         "low": df["low"].astype(float), "close": df["close"].astype(float)})


def fetch_bars(raw: Path) -> None:
    frames, info = [], []
    for path, want in FILES.items():
        r = requests.get(RAW_URL.format(repo=REPO, ref=COMMIT, path=path), timeout=300)
        r.raise_for_status()
        got = hashlib.sha256(r.content).hexdigest()
        if got != want:
            raise SystemExit(f"{path}: SHA-256 {got} is not the pinned {want}")
        frames.append(parse_bars(r.content))
        info.append({"path": path, "bytes": len(r.content), "sha256": got})
    df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(raw / "finnifty_1min.csv", index=False, lineterminator="\n")
    prov = {"repo": REPO, "commit": COMMIT, "files": info, "rows": len(df),
            "first": str(df["timestamp"].min()), "last": str(df["timestamp"].max()),
            "days": int(df["timestamp"].dt.normalize().nunique()),
            "note": "no licence stated in the source repository; origin of the bars not stated; volume column dropped"}
    (raw / "finnifty_1min_source.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"finnifty_1min.csv: {len(df)} rows, {prov['first']} to {prov['last']}, {prov['days']} days")


def official_table(raw: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((raw / "nse_full" / "index_close_full").glob("ind_close_all_*.csv")):
        part = pd.read_csv(f, dtype=str)
        part.columns = [c.strip() for c in part.columns]
        part = part[part["Index Name"].str.strip().str.lower() == INDEX_NAME.lower()]
        if part.empty:
            continue
        r = part.iloc[0]
        rows.append({"date": pd.to_datetime(f.stem[-8:], format="%d%m%Y"), "index": INDEX_NAME,
                     **{k: pd.to_numeric(str(r[c]).replace(",", ""), errors="coerce") for k, c in
                        (("open", "Open Index Value"), ("high", "High Index Value"),
                         ("low", "Low Index Value"), ("close", "Closing Index Value"))}})
    out = pd.DataFrame(rows).sort_values("date")
    (raw / "finnifty").mkdir(parents=True, exist_ok=True)
    out.to_csv(raw / "finnifty" / "index_daily_finnifty.csv", index=False, lineterminator="\n",
               date_format="%Y-%m-%d")
    print(f"index_daily_finnifty.csv: {len(out)} days, {out['date'].min().date()} to {out['date'].max().date()}")
    return out


def bhav_day(d: pd.Timestamp, raw: Path, pause: float):
    dest = raw / "finnifty" / "bhavcopy" / f"fo_{d:%Y%m%d}.csv.gz"
    if dest.exists():
        return None
    src = raw / "nse_full" / "fo_index" / f"fo_{d:%Y%m%d}.csv.gz"
    if src.exists():
        shutil.copyfile(src, dest)
        return None
    s = requests.Session()
    content = None
    for url in bhavcopy_urls(d):
        content = fetch(s, url)
        if content is not None:
            break
    time.sleep(pause)
    if content is None:
        return d.date().isoformat()
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        rawcsv = z.read(z.namelist()[0])
    filter_bhavcopy(rawcsv, INDEX_SYMBOLS).to_csv(dest, index=False, compression={"method": "gzip", "mtime": 0})
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--start", default="2023-01-02")
    ap.add_argument("--bhav-end", default="2026-02-06",
                    help="last bhavcopy day: 10 calendar days after the last bar day (the loader reads files "
                         "up to 10 days past the sample, as for Bank Nifty, whose folder ran past its last bar)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.5)
    ap.add_argument("--skip-bars", action="store_true")
    a = ap.parse_args()
    raw = Path(a.raw)
    if not a.skip_bars:
        fetch_bars(raw)
    off = official_table(raw)
    (raw / "finnifty" / "bhavcopy").mkdir(parents=True, exist_ok=True)
    days = [d for d in off["date"] if pd.Timestamp(a.start) <= d <= pd.Timestamp(a.bhav_end)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        failed = [x for x in ex.map(lambda d: bhav_day(d, raw, a.pause), days) if x]
    print(f"bhavcopy: {len(days)} official days, {len(failed)} not found, {time.time() - t0:.0f}s; {failed[:20]}")


if __name__ == "__main__":
    main()
