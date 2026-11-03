#!/usr/bin/env python
"""P108 phase 3, POST HOC (REAL DATA): why do the two independent searches pick different endpoints?

Written after the blind answer was committed and after reading the earlier search's result
(14 Nov 2022 to 10 Feb 2023). It rescores the 16 "feasible" 64-day windows (those reproducing
N and all four extremes once the 27 Jan 2023 day-end price is corrected, see posthoc.py PH2/PH7)
with the statistic formulas fixed to the ones the after window points to (SD with N in the
denominator, adjusted skewness G1, adjusted EXCESS kurtosis G2), instead of letting each window
pick its best formula. It reports, for each window, the error on every printed number, so that
one can see which printed numbers favour which endpoints.

    OMP_NUM_THREADS=1 python experiments/p108_blind_window/phase3_compare.py --raw data/raw --out results/p108_blind_window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc as PH  # noqa: E402
import search as S  # noqa: E402


def fixed_variant_table(tbl: dict, printed: dict) -> pd.DataFrame:
    cols = {"a": tbl["a"], "b": tbl["b"], "L": tbl["L"]}
    sq = np.zeros(len(tbl["a"]))
    for k in S.INDICES:
        v = S.stat_variants(tbl[k]["n"], tbl[k]["m2"], tbl[k]["m3"], tbl[k]["m4"])
        vals = {"sd": v[("sd", "population")], "skew": v[("skew", "G1")], "kurt": v[("kurt", "G2")],
                "min": tbl[k]["min"], "max": tbl[k]["max"]}
        for item, c in vals.items():
            p, dec = printed[k][item]
            cols[f"{k}_{item}"] = c
            e = S._err(item, c, p)
            cols[f"{k}_{item}_relerr"] = e
            sq += e ** 2
        for item in ("mean", "n"):
            p, dec = printed[k][item]
            cols[f"{k}_{item}"] = tbl[k][item]
            cols[f"{k}_{item}_ok"] = S.matches(tbl[k][item], p, dec)
    p, dec = printed["corr"]
    cols["corr"] = tbl["corr"]
    cols["corr_err"] = tbl["corr"] - p
    sq += (tbl["corr"] - p) ** 2
    cols["D_fixed"] = np.sqrt(sq / 11)
    return pd.DataFrame(cols)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/p108_blind_window")
    args = ap.parse_args(argv)
    raw, out = Path(args.raw), Path(args.out)
    printed = S.load_printed(window="before")
    dfn = S.load_bars(raw / "nifty_1min.csv")
    dfb = S.load_bars(raw / "banknifty_1min.csv")
    days, mats = S.build_matrices(dfn, dfb, fields=("open", "high", "low", "close"))
    d = S.definition_from_name(PH.WIN_DEF)
    frames = []
    for label, m in (("patched", PH.patched_mats(mats, days)[0]), ("unpatched", mats)):
        dl = S.day_level(days, m, d)
        tbl = S.window_table(S.day_aggregates(dl), d.overnight, lengths=(64,))
        t = fixed_variant_table(tbl, printed)
        cal = dl["days"]
        t.insert(0, "start", cal[t["a"].to_numpy()].strftime("%Y-%m-%d"))
        t.insert(1, "end", cal[t["b"].to_numpy()].strftime("%Y-%m-%d"))
        t.insert(0, "prices", label)
        # the feasible set: N and both maxima exact, minima equal to the 30 Jan 2023 overnight return
        on_min = {"patched": (-0.008584649, -0.017256735), "unpatched": (-0.008533548, -0.017135337)}[label]
        feas = (t["nifty_n_ok"] & t["bank_n_ok"]
                & S.matches(t["nifty_max"], *printed["nifty"]["max"]) & S.matches(t["bank_max"], *printed["bank"]["max"])
                & S.matches(t["nifty_min"], on_min[0], 9) & S.matches(t["bank_min"], on_min[1], 9))
        t = t[feas].copy()
        t["both_means_ok"] = t["nifty_mean_ok"] & t["bank_mean_ok"]
        t["rank_D_fixed"] = t["D_fixed"].rank().astype(int)
        frames.append(t.sort_values("D_fixed"))
    res = pd.concat(frames, ignore_index=True)
    res.drop(columns=["a", "b"]).to_csv(out / "phase3_feasible_windows_fixed_formulas.csv", index=False,
                                         float_format="%.10g")
    show = ["prices", "start", "end", "D_fixed", "rank_D_fixed", "both_means_ok", "nifty_mean", "bank_mean",
            "nifty_sd_relerr", "bank_sd_relerr", "nifty_skew_relerr", "bank_skew_relerr", "nifty_kurt_relerr",
            "bank_kurt_relerr", "corr_err"]
    pd.set_option("display.width", 250)
    print(res[show].to_string())


if __name__ == "__main__":
    main()
