"""Check what unit bhavcopy open interest is in, against NSE's participant-wise OI file.

For each date, downloads the FULL UDiFF F&O bhavcopy (all index option
symbols, not only NIFTY) and the participant-wise OI file, then compares:

* all: sum over index options of OpnIntrst / NewBrdLotQty (each row's own lot size)
* excl_expiring: the same, leaving out contracts that expire on that date
* participant: the TOTAL "Option Index Call Long" (or Put Long) row, in contracts

    python scripts/check_oi_units.py --dates 2024-05-15 2024-07-05 2025-10-30

Writes results/data_validation/oi_units_check.csv.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
A = "https://nsearchives.nseindia.com"
DEFAULT = ["2024-01-10", "2024-05-15", "2024-07-05", "2024-07-08", "2024-10-10",
           "2025-10-28", "2025-10-30", "2026-10-27", "2026-10-29"]


def bhavcopy(d: pd.Timestamp) -> pd.DataFrame:
    r = requests.get(f"{A}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip", headers=H, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [c.strip() for c in df.columns]
    return df


def participant(d: pd.Timestamp) -> pd.DataFrame:
    r = requests.get(f"{A}/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv", headers=H, timeout=60)
    r.raise_for_status()
    lines = r.text.splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.strip().strip('"').startswith("Client Type"))
    t = pd.read_csv(io.StringIO("\n".join(lines[k:])))
    t.columns = [c.strip() for c in t.columns]
    return t.set_index(t.columns[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=DEFAULT)
    ap.add_argument("--out", default="results/data_validation/oi_units_check.csv")
    a = ap.parse_args()
    rows = []
    for ds in a.dates:
        d = pd.Timestamp(ds)
        df, p = bhavcopy(d), participant(d)
        o = df[df["FinInstrmTp"] == "IDO"].copy()
        o["contracts"] = o["OpnIntrst"] / o["NewBrdLotQty"]
        expiring = pd.to_datetime(o["XpryDt"]) == d
        for typ, col in (("CE", "Option Index Call Long"), ("PE", "Option Index Put Long")):
            s = o["OptnTp"] == typ
            rows.append({"date": ds, "type": typ, "all": round(o.loc[s, "contracts"].sum()),
                         "excl_expiring": round(o.loc[s & ~expiring, "contracts"].sum()),
                         "participant": int(p.loc["TOTAL", col]),
                         "expiring_symbols": " ".join(sorted(o.loc[expiring, "TckrSymb"].unique()))})
    r = pd.DataFrame(rows)
    r["diff_excl_expiring"] = r["excl_expiring"] - r["participant"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(a.out, index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
