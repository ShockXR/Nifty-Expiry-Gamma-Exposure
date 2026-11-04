"""Compare the two outage days' minute bars with NSE's official daily index values (REAL DATA, post hoc).

The local official files start on 20 October 2022, so the flat-bar screen could not check the two NSE outage
days it found (10 July 2017 and 24 February 2021). This script downloads NSE's daily index file for each day
from the public archive (archives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv) into
data/raw/dq1_official/ (git-ignored, like all raw data) and compares the official open, high, low and close
with the 09:15 to 15:29 minute bars, as scripts/validate_minute_bars.py does.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/dq1_flat_bars/official_check.py
"""
import urllib.request
from pathlib import Path

import pandas as pd

from niftygex.data import load_minute_bars

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
DL = RAW / "dq1_official"
OUT = REPO / "results" / "dq1_flat_bars"
LABEL = "REAL DATA (third-party 1-minute bars; NSE official daily index values from the NSE archive)"
DAYS = ["2017-07-10", "2021-02-24"]
INDICES = {"nifty": ("nifty_1min.csv", "Nifty 50"), "banknifty": ("banknifty_1min.csv", "Nifty Bank")}
URL = "https://archives.nseindia.com/content/indices/ind_close_all_{}.csv"


def official(day: pd.Timestamp) -> pd.DataFrame:
    tag = day.strftime("%d%m%Y")
    path = DL / f"ind_close_all_{tag}.csv"
    if not path.exists():
        DL.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL.format(tag), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    o = pd.read_csv(path)
    o.columns = [c.strip() for c in o.columns]
    o["Index Name"] = o["Index Name"].str.strip()
    return o


def main() -> None:
    rows = []
    for k, (f, name) in INDICES.items():
        bars = load_minute_bars(RAW / f)
        for d in DAYS:
            day = pd.Timestamp(d)
            o = official(day)
            o = o[o["Index Name"] == name].iloc[0]
            x = bars[bars.index.normalize() == day].between_time("09:15", "15:29")
            off = {c: float(o[f"{c.title()} Index Value"]) for c in ["open", "high", "low"]}
            off["close"] = float(o["Closing Index Value"])
            ours = {"open": float(x["open"].iloc[0]), "high": float(x["high"].max()),
                    "low": float(x["low"].min()), "close": float(x["close"].iloc[-1])}
            row = {"data": LABEL, "index": k, "date": d, "bars_0915_1529": len(x)}
            for c in ["open", "high", "low", "close"]:
                row[f"bars_{c}"] = ours[c]
                row[f"official_{c}"] = off[c]
                row[f"diff_{c}_bp"] = round(1e4 * (ours[c] / off[c] - 1), 1)
            row["high_low_within_5bp"] = abs(row["diff_high_bp"]) <= 5 and abs(row["diff_low_bp"]) <= 5
            rows.append(row)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "official_check.csv", index=False)
    print(res.drop(columns="data").to_string(index=False))


if __name__ == "__main__":
    main()
