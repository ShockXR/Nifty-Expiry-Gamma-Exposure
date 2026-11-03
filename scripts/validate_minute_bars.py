"""Check third-party 1-minute bars against NSE's official daily index values.

The minute bars come from community datasets on GitHub (see
fetch_minute_bars.py), so before any result uses them we compare, day by
day, what the bars imply with what NSE itself published (index_daily.csv,
from download_nse.py):

* high: highest 1-minute high of the day vs NSE's day high
* low: lowest 1-minute low vs NSE's day low
* open: first bar's open vs NSE's open
* close: last bar's close vs NSE's close. These are NOT expected to match
  exactly: NSE's official close is an average over the last 30 minutes of
  trading, not the last trade. The difference is reported for information.

It also lists official trading days with no bars, bar days that are not
official trading days, and days with fewer bars than a full session.

    python scripts/validate_minute_bars.py --raw data/raw --start 2023-01-01

Writes results/data_validation/: one CSV per index with every day, and
summary.csv. The Fin Nifty bars of P104 (scripts/fetch_finnifty_data.py) are
checked against NSE's "Nifty Financial Services" values with

    python scripts/validate_minute_bars.py --indices "Nifty Financial Services" --out results/data_validation_finnifty
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# index name in the official file -> (bar file, official daily file), both under --raw
PAIRS = {"Nifty 50": ("nifty_1min.csv", "index_daily.csv"), "Nifty Bank": ("banknifty_1min.csv", "index_daily.csv"),
         "Nifty Financial Services": ("finnifty_1min.csv", "finnifty/index_daily_finnifty.csv")}
DEFAULT_INDICES = ["Nifty 50", "Nifty Bank"]


def daily_from_bars(path: Path, start: str) -> pd.DataFrame:
    b = pd.read_csv(path, parse_dates=["timestamp"])
    b = b[b["timestamp"] >= pd.Timestamp(start)]
    t = b["timestamp"].dt.time
    b = b[(t >= pd.Timestamp("09:15").time()) & (t <= pd.Timestamp("15:29").time())]
    g = b.groupby(b["timestamp"].dt.normalize())
    return pd.DataFrame({"open_bars": g["open"].first(), "high_bars": g["high"].max(), "low_bars": g["low"].min(),
                         "close_bars": g["close"].last(), "n_bars": g.size()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--out", default="results/data_validation")
    ap.add_argument("--tol-bp", type=float, default=5.0, help="tolerance in basis points for high/low/open")
    ap.add_argument("--indices", nargs="+", default=DEFAULT_INDICES, choices=list(PAIRS))
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for index in a.indices:
        fname, off_file = PAIRS[index]
        off = pd.read_csv(raw / off_file, parse_dates=["date"])
        o = off[off["index"].str.lower() == index.lower()].set_index("date")[["open", "high", "low", "close"]]
        o = o[o.index >= pd.Timestamp(a.start)]
        bars = daily_from_bars(raw / fname, a.start)
        last_bar_day = bars.index.max()
        o_cov = o[o.index <= last_bar_day]
        df = o_cov.join(bars, how="outer")
        for k in ("open", "high", "low", "close"):
            df[f"{k}_diff_bp"] = 1e4 * (df[f"{k}_bars"] / df[k] - 1)
        df["official_day_without_bars"] = df["n_bars"].isna() & df["close"].notna()
        df["bar_day_not_official"] = df["close"].isna() & df["n_bars"].notna()
        df["short_session"] = df["n_bars"] < 370
        ok = df["n_bars"].notna() & df["close"].notna()
        good = df[ok]
        within = lambda k: float((good[f"{k}_diff_bp"].abs() <= a.tol_bp).mean())
        df.to_csv(out / f"minute_vs_official_{index.replace(' ', '_').lower()}.csv", index_label="date")
        summary.append({
            "index": index, "bar_file": fname,
            "first_day": str(good.index.min().date()), "last_day": str(good.index.max().date()),
            "official_days": int(o_cov.shape[0]), "days_compared": int(ok.sum()),
            "official_days_without_bars": int(df["official_day_without_bars"].sum()),
            "bar_days_not_official": int(df["bar_day_not_official"].sum()),
            "days_under_370_bars": int((good["n_bars"] < 370).sum()),
            "median_bars_per_day": float(good["n_bars"].median()),
            f"high_within_{a.tol_bp:g}bp": within("high"), f"low_within_{a.tol_bp:g}bp": within("low"),
            f"open_within_{a.tol_bp:g}bp": within("open"),
            "high_median_abs_bp": float(good["high_diff_bp"].abs().median()),
            "low_median_abs_bp": float(good["low_diff_bp"].abs().median()),
            "close_median_abs_bp": float(good["close_diff_bp"].abs().median()),
            "close_p95_abs_bp": float(good["close_diff_bp"].abs().quantile(0.95)),
            "worst_high_or_low_bp": float(np.nanmax(np.abs(good[["high_diff_bp", "low_diff_bp"]].to_numpy()))),
        })
    s = pd.DataFrame(summary)
    s.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(s.T)


if __name__ == "__main__":
    main()
