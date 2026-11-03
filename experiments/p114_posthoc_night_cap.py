"""P114, POST HOC (added after the registered results were seen; not in the plan; no p-values; changes no
registered number or verdict).

The registered half-life of a 12:15 shock cannot exceed about 178.5 minutes when most of what is left at the close
is removed overnight (the "night cap", plan Section 2 and weakness 5). This script reads the registered bootstrap
draws and reports, for every before-after comparison of the reference (ref), the fixed model (F) and F with
leverage (F4):

* how many draws of each window have a response still above half of its first value at the last minute of the
  shock's day (half-life of at least SHOCK_TO_CLOSE - 1 = 178 minutes), and the largest half-life among the draws.
  For F and F4 such a draw reaches half only through the overnight fade, so its half-life sits at the night cap
  (about 178 to 179 minutes) whatever the persistence within the day. For the reference (no overnight fade) a
  half-life above 178 minutes is not capped: it only says that the response outlasts the day;
* a measure that has no night cap: the share of the first response still there at the close of the shock's day
  (`share_close`, already computed for every fit and draw by the registered code), with its 95% percentile interval
  in each window and for the paired change after minus before (draws paired by draw number, as in the registered
  bootstrap).

    PYTHONPATH=src python experiments/p114_posthoc_night_cap.py [--out results/p114_persistence_fixed_model]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = 360
SHOCK_MINUTE = 180
SHOCK_TO_CLOSE = M - 1 - SHOCK_MINUTE          # 179 steps from the shock to the last minute of the day
CAP_FROM = SHOCK_TO_CLOSE - 1                  # half-life at or above 178: half point reached only overnight
BEFORE = ["before_inferred", "before_stated", "before_p108"]


def pct(a, q):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if len(a) else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "p114_persistence_fixed_model"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    boot = pd.read_csv(out / "bootstrap_draws.csv", float_precision="round_trip")
    fits = pd.read_csv(out / "fits.csv", float_precision="round_trip")
    label = str(fits["data"].iloc[0])
    rows = []
    for w in BEFORE:
        for k in ("nifty", "banknifty"):
            for m in ("ref", "F", "F4"):
                db = boot[(boot["window"] == w) & (boot["index"] == k) & (boot["model"] == m)].sort_values("draw")
                da = boot[(boot["window"] == "after") & (boot["index"] == k) & (boot["model"] == m)].sort_values("draw")
                if db.empty or da.empty:
                    continue
                pb = fits[(fits["window"] == w) & (fits["index"] == k) & (fits["model"] == m)].iloc[0]
                pa = fits[(fits["window"] == "after") & (fits["index"] == k) & (fits["model"] == m)].iloc[0]
                n = min(len(db), len(da))
                sb = db["share_close"].to_numpy()[:n]
                sa = da["share_close"].to_numpy()[:n]
                d = sa - sb
                rows.append({
                    "data": label, "role": "post hoc (descriptive, no p-value)", "before": w, "index": k, "model": m,
                    "boot_draws": n,
                    "draws_above_half_at_close_before": int(np.sum(db["hl_irf"].to_numpy()[:n] >= CAP_FROM)),
                    "draws_above_half_at_close_after": int(np.sum(da["hl_irf"].to_numpy()[:n] >= CAP_FROM)),
                    "max_hl_irf_draws_before": float(np.nanmax(db["hl_irf"].to_numpy()[:n])),
                    "max_hl_irf_draws_after": float(np.nanmax(da["hl_irf"].to_numpy()[:n])),
                    "point_above_half_at_close_before": bool(pb["hl_irf"] >= CAP_FROM),
                    "point_above_half_at_close_after": bool(pa["hl_irf"] >= CAP_FROM),
                    "share_close_before": pb["share_close"], "share_close_before_lo": pct(sb, 2.5),
                    "share_close_before_hi": pct(sb, 97.5),
                    "share_close_after": pa["share_close"], "share_close_after_lo": pct(sa, 2.5),
                    "share_close_after_hi": pct(sa, 97.5),
                    "d_share_close": pa["share_close"] - pb["share_close"], "d_share_close_lo": pct(d, 2.5),
                    "d_share_close_hi": pct(d, 97.5),
                    "share_draws_fell_share_close": float(np.mean(d[~np.isnan(d)] < 0)),
                })
    res = pd.DataFrame(rows)
    res.to_csv(out / "posthoc_night_cap.csv", index=False)
    print(res.to_string())


if __name__ == "__main__":
    main()
