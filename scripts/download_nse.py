"""Download NSE F&O bhavcopy and participant-wise OI files for a date range.

Run this on your own machine (the build environment cannot reach NSE).
NOT TESTED against live NSE servers from the build environment: URL patterns
were correct as of 2025 and may change. Failed days are listed at the end.

    python scripts/download_nse.py --start 2023-01-01 --end 2026-10-31 --out data/raw
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}
UDIFF_START = pd.Timestamp("2024-07-08")


def bhavcopy_url(d: pd.Timestamp) -> str:
    if d >= UDIFF_START:
        return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    return f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"


def participant_url(d: pd.Timestamp) -> str:
    return f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv"


def fetch(session, url, retries=3):
    for k in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "bhavcopy").mkdir(parents=True, exist_ok=True)
    (out / "participant_oi").mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.get("https://www.nseindia.com", headers=HEADERS, timeout=20)  # sets cookies
    failed = []
    for d in pd.bdate_range(a.start, a.end):
        bfile = out / "bhavcopy" / f"fo_{d:%Y%m%d}.csv"
        if not bfile.exists():
            content = fetch(s, bhavcopy_url(d))
            if content is None:
                failed.append(("bhavcopy", d.date()))
            else:
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    bfile.write_bytes(z.read(z.namelist()[0]))
            time.sleep(1)
        pfile = out / "participant_oi" / f"fao_participant_oi_{d:%d%m%Y}.csv"
        if not pfile.exists():
            content = fetch(s, participant_url(d))
            if content is None:
                failed.append(("participant", d.date()))
            else:
                pfile.write_bytes(content)
            time.sleep(1)
    print(f"done; {len(failed)} files not found (holidays are expected to fail):")
    for f in failed[:50]:
        print("  ", *f)


if __name__ == "__main__":
    main()
