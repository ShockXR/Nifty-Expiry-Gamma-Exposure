"""Flat-bar screen of every index minute file (REAL DATA, read only).

Session: the project's own loader (niftygex.data.load_minute_bars + to_session_grid), 09:15 to 15:30,
375 one-minute slots. One-minute return as the forecasting pipeline defines it: log(close/prev), prev =
previous slot's close (the bar's own open for 09:15). A flat return is an exact zero. Shares are flat
returns over finite returns. Official NSE daily high/low (where a local file has them) are compared as in
scripts/validate_minute_bars.py (09:15 to 15:29 bars, 5 bp tolerance).

    PYTHONPATH=$REPO/src OMP_NUM_THREADS=1 python screen.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.data import load_minute_bars, to_session_grid

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT = Path(__file__).resolve().parents[2] / "results" / "dq1_flat_bars"
LABEL = "REAL DATA (third-party 1-minute bars; NSE official daily values where available)"
FILES = {"nifty": "nifty_1min.csv", "banknifty": "banknifty_1min.csv", "finnifty": "finnifty_1min.csv",
         "sensex": "sensex_1min.csv"}
OFFICIAL = {"nifty": [("index_daily.csv", "nifty 50"), ("extra_2022/index_daily.csv", "nifty 50")],
            "banknifty": [("index_daily.csv", "nifty bank"), ("extra_2022/index_daily.csv", "nifty bank")],
            "finnifty": [("finnifty/index_daily_finnifty.csv", "nifty financial services")],
            "sensex": []}
TOL_BP = 5.0


def longest_run(mask_row):
    best = cur = 0
    for v in mask_row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def official(k):
    frames = []
    for f, name in OFFICIAL[k]:
        o = pd.read_csv(RAW / f, parse_dates=["date"])
        frames.append(o[o["index"].str.lower() == name][["date", "open", "high", "low", "close"]])
    if not frames:
        return None
    o = pd.concat(frames).drop_duplicates("date").set_index("date").sort_index()
    return o


def screen(k):
    bars = load_minute_bars(RAW / FILES[k])
    G = to_session_grid(bars)
    R, C, H, L = G["returns"], G["close"], G["high"], G["low"]
    fin = np.isfinite(R)
    flat = fin & (R == 0)
    d = pd.DataFrame({
        "index": k, "date": G["days"], "weekday": G["days"].day_name().str[:3],
        "bars": np.isfinite(C).sum(axis=1), "returns": fin.sum(axis=1), "flat_minutes": flat.sum(axis=1),
        "longest_flat_run_min": [longest_run(r) for r in flat],
        "flat_minutes_last20": flat[:, 355:].sum(axis=1),
        "flat_minutes_first360": flat[:, :360].sum(axis=1),
        "high_eq_low_bars": (np.isfinite(C) & (H == L)).sum(axis=1),
    })
    d["flat_share"] = d["flat_minutes"] / d["returns"].where(d["returns"] > 0)
    d["flat_share_excl_last20"] = (d["flat_minutes"] - d["flat_minutes_last20"]) / d["returns"].where(d["returns"] > 0)
    # first and last slot of the longest run, as clock times
    starts = []
    for r in flat:
        best = cur = 0; bs = s = -1
        for j, v in enumerate(r):
            if v:
                if cur == 0:
                    s = j
                cur += 1
                if cur > best:
                    best, bs = cur, s
            else:
                cur = 0
        starts.append("" if best == 0 else
                      f"{(pd.Timestamp('09:15') + pd.Timedelta(minutes=bs)).strftime('%H:%M')} to "
                      f"{(pd.Timestamp('09:15') + pd.Timedelta(minutes=bs + best - 1)).strftime('%H:%M')}")
    d["longest_run_clock"] = starts
    # official high/low
    o = official(k)
    hb = np.nanmax(np.where(np.isfinite(H), H, -np.inf), axis=1)
    lb = np.nanmin(np.where(np.isfinite(L), L, np.inf), axis=1)
    d["bars_high"], d["bars_low"] = hb, lb
    if o is not None:
        oo = o.reindex(G["days"])
        d["official_high"], d["official_low"] = oo["high"].to_numpy(), oo["low"].to_numpy()
        d["high_diff_bp"] = 1e4 * (d["bars_high"] / d["official_high"] - 1)
        d["low_diff_bp"] = 1e4 * (d["bars_low"] / d["official_low"] - 1)
        has = d["official_high"].notna()
        d["official_check"] = np.where(~has, "no official file for this day",
                                       np.where((d["high_diff_bp"].abs() > TOL_BP) | (d["low_diff_bp"].abs() > TOL_BP),
                                                "DISAGREES (>5 bp)", "agrees (within 5 bp)"))
        first_off = o.index.min()
    else:
        d["official_high"] = d["official_low"] = d["high_diff_bp"] = d["low_diff_bp"] = np.nan
        d["official_check"] = "no official BSE file in data/raw"
        first_off = None
    d.insert(0, "data", LABEL)
    return d, first_off


def main():
    alld, cov = [], []
    for k in FILES:
        d, first_off = screen(k)
        alld.append(d)
        cov.append({"index": k, "first_day": str(d["date"].min().date()), "last_day": str(d["date"].max().date()),
                    "days": len(d), "median_flat_share": float(d["flat_share"].median()),
                    "days_over_5pct": int((d["flat_share"] > 0.05).sum()),
                    "days_over_20pct": int((d["flat_share"] > 0.20).sum()),
                    "official_from": str(first_off.date()) if first_off is not None else "none"})
    A = pd.concat(alld, ignore_index=True)
    A.to_csv(OUT / "flat_all_days.csv.gz", index=False)
    f5 = A[A["flat_share"] > 0.05]
    f5.to_csv(OUT / "flat_days_over_5pct.csv", index=False)
    f20 = A[A["flat_share"] > 0.20]
    f20.to_csv(OUT / "flat_days_over_20pct.csv", index=False)
    pd.DataFrame(cov).to_csv(OUT / "coverage.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
    cols = ["index", "date", "weekday", "bars", "returns", "flat_minutes", "flat_share", "longest_flat_run_min",
            "longest_run_clock", "flat_minutes_last20", "flat_minutes_first360", "high_eq_low_bars", "high_diff_bp",
            "low_diff_bp", "official_check"]
    print(LABEL)
    print(pd.DataFrame(cov).to_string(index=False))
    print("\n>20% flat:")
    print(f20[cols].to_string(index=False))
    print("\n>5% flat:")
    print(f5[cols].to_string(index=False))


if __name__ == "__main__":
    main()
