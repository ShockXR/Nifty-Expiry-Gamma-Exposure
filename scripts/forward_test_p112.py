"""Forward test P112 (docs/prereg/P112_FORWARD_MCSGARCH_FIXES.md): count the qualifying days, and run the
registered test once, at its stopping point. REAL DATA unless the input says otherwise.

NEW CODE, written on 4 November 2026 by an AI agent (Claude), after the plan was registered and before any of its
test days (from Wednesday 4 November 2026) existed. It adds no statistic of its own: the test is the one the plan
names (P109's one-sided Diebold-Mariano on daily mean QLIKE, Newey-West variance, HLN factor, Student t(n-1),
niftygex.evaluation.diebold_mariano), with Holm over the two horizons.

    # how many valid test days exist, against the 250 the plan needs (computes NO loss, NO forecast)
    PYTHONPATH=src python scripts/forward_test_p112.py count --raw data/live/forward
    # the test itself: refuses unless the stopping point is reached AND the flag is given
    PYTHONPATH=src python scripts/forward_test_p112.py run --raw data/live/forward \\
        --forecasts <per-anchor forecasts> --i-confirm-stopping-point

Qualifying test day (plan section 3): a Nifty trading day (a row for "Nifty 50" in index_daily.csv) on or after
4 November 2026 whose one-minute bars exist and whose daily high and low from the bars are within 0.05% (5 bp)
of NSE's official high and low (the rule of scripts/validate_minute_bars.py).

Stopping point (plan section 3): the 250th qualifying day, or 31 December 2027, whichever comes first. The date
cap counts as reached only when the official index file runs to 31 December 2027 or later. If the cap comes
first with fewer than 120 qualifying days, the test is underpowered: descriptive only, no verdict.

What `run` needs and what does not exist yet: `--forecasts` is a CSV with one row per (date, minute, target)
for target in {next15, next30} and columns y, mcsgarch, combination, mcs_cpp, hybrid, made by continuing the
pinned walk-forward of P109 (`experiments/p109_mcsgarch_extensions.py` at 7f0b5c7 / f081138), P116 (`mcs_cpp`
at cf6c9d4) and the v2 pipeline (hybrid_base) forward in time, refitting every 21 trading days. The code that
produces that file for days after 2 November 2026 is NOT written yet: p109's `run` stage is pinned to the
committed Nifty run ending on 2 November 2026 and stops if the number of days differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.evaluation import daily_losses, diebold_mariano

PLAN = "docs/prereg/P112_FORWARD_MCSGARCH_FIXES.md"
FIRST_TEST_DAY = pd.Timestamp("2026-11-04")
STOP_DAYS = 250
CAP = pd.Timestamp("2027-12-31")
MIN_DAYS_FOR_VERDICT = 120
TOL_BP = 5.0                     # 0.05%
ALPHA = 0.05
TARGETS = ("next15", "next30")
MODELS = ("mcsgarch", "combination", "mcs_cpp", "hybrid")
OPEN_MINUTE = 0                  # the 09:15 anchor (minute 0), left out in S3
REPS, MEAN_BLOCK, BOOT_SEED = 2000, 10, 112


# ---------------------------------------------------------------- qualifying days (no outcome)

def official_nifty(raw: Path) -> pd.DataFrame:
    idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
    x = idx[idx["index"].str.upper() == "NIFTY 50"].set_index("date").sort_index()
    return x[~x.index.duplicated(keep="last")]


def bar_high_low(raw: Path, first: pd.Timestamp) -> pd.DataFrame:
    b = pd.read_csv(raw / "nifty_1min.csv", parse_dates=["timestamp"])
    b = b[b["timestamp"] >= first]
    b["date"] = b["timestamp"].dt.normalize()
    return b.groupby("date").agg(high=("high", "max"), low=("low", "min"), n_bars=("close", "size"))


def qualifying_days(raw: Path) -> pd.DataFrame:
    """One row per Nifty trading day from 4 November 2026: bars present, high/low gaps in bp, valid flag."""
    off = official_nifty(raw)
    off = off[off.index >= FIRST_TEST_DAY]
    bars = bar_high_low(raw, FIRST_TEST_DAY).reindex(off.index)
    t = pd.DataFrame(index=off.index)
    t["n_bars"] = bars["n_bars"].fillna(0).astype(int)
    t["high_diff_bp"] = 1e4 * (bars["high"] / off["high"] - 1)
    t["low_diff_bp"] = 1e4 * (bars["low"] / off["low"] - 1)
    t["valid"] = (t["n_bars"] > 0) & (t["high_diff_bp"].abs() <= TOL_BP) & (t["low_diff_bp"].abs() <= TOL_BP)
    return t


def stopping(days: pd.DataFrame, last_official_day) -> dict:
    valid = days.index[days["valid"].to_numpy()]
    in_cap = valid[valid <= CAP]
    info = {"rule": f"{STOP_DAYS}th valid test day from {FIRST_TEST_DAY.date()}, or {CAP.date()}, whichever first",
            "trading_days_since_start": int((days.index <= CAP).sum()), "valid_test_days": int(len(in_cap)),
            "required_valid_days": STOP_DAYS,
            "last_official_day": None if last_official_day is None else str(pd.Timestamp(last_official_day).date())}
    if len(in_cap) >= STOP_DAYS:
        return {**info, "reached": True, "stop_day": str(in_cap[STOP_DAYS - 1].date()), "reason": f"{STOP_DAYS} valid days",
                "underpowered": False}
    if last_official_day is not None and pd.Timestamp(last_official_day) >= CAP:
        return {**info, "reached": True, "stop_day": str(CAP.date()), "reason": "date cap reached",
                "underpowered": len(in_cap) < MIN_DAYS_FOR_VERDICT}
    return {**info, "reached": False, "stop_day": None, "reason": "stopping point not reached", "underpowered": None}


def test_days(raw: Path) -> tuple[pd.DataFrame, dict]:
    off = official_nifty(raw)
    days = qualifying_days(raw)
    return days, stopping(days, off.index.max() if len(off) else None)


# ---------------------------------------------------------------- the registered test (only at the stopping point)

def holm(p: dict) -> dict:
    k = sorted(p, key=lambda x: p[x])
    m = len(k)
    out, run = {}, 0.0
    for i, h in enumerate(k):
        run = max(run, min(1.0, (m - i) * p[h]))
        out[h] = run
    return out


def dm_one_sided(D: pd.DataFrame, worse: str, better: str) -> dict:
    """p: `better` has lower loss; q: the opposite one-sided p. Estimate in points of `worse`'s mean QLIKE."""
    g = diebold_mariano(D[worse], D[better], alternative="greater")
    q = diebold_mariano(D[worse], D[better], alternative="less")["p"]
    return {"n_days": g["n"], "t": g["stat"], "p": g["p"], "p_opposite": q,
            "gain_points": float(100 * (D[worse].mean() - D[better].mean()) / D[worse].mean())}


def family(D: dict, worse: str, better: str) -> dict:
    r = {t: dm_one_sided(D[t], worse, better) for t in TARGETS}
    adj = holm({t: r[t]["p"] for t in TARGETS})
    for t in TARGETS:
        r[t]["holm_p"] = adj[t]
        r[t]["verdict"] = ("contradicted" if r[t]["p_opposite"] <= ALPHA else
                           "confirmed" if adj[t] <= ALPHA else "not confirmed")
    return r


def recovered_share(D: pd.DataFrame, rng) -> tuple[float, float, float]:
    """S2 as P109's recovered_share: (L_mcs - L_comb) / (L_mcs - L_hybrid), stationary bootstrap of days."""
    x = D[["mcsgarch", "hybrid", "combination"]].to_numpy(float)
    n = len(x)
    share = (x[:, 0].mean() - x[:, 2].mean()) / (x[:, 0].mean() - x[:, 1].mean())
    p = 1.0 / MEAN_BLOCK
    s = np.empty(REPS)
    for b in range(REPS):
        idx = np.empty(n, int)
        idx[0] = rng.integers(n)
        for i in range(1, n):
            idx[i] = rng.integers(n) if rng.random() < p else (idx[i - 1] + 1) % n
        m = x[idx].mean(axis=0)
        s[b] = (m[0] - m[2]) / (m[0] - m[1])
    return float(share), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def analyse(fc: pd.DataFrame, valid_days: pd.DatetimeIndex) -> dict:
    fc = fc.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    fc = fc[fc["date"].isin(valid_days)]
    D, D_no_open = {}, {}
    for t in TARGETS:
        x = fc[fc["target"] == t]
        if x.empty or x[list(MODELS) + ["y"]].isna().any().any():
            raise SystemExit(f"forecasts for {t} are missing or incomplete on the test days")
        D[t] = daily_losses(x, list(MODELS), "qlike")
        D_no_open[t] = daily_losses(x[x["minute"] != OPEN_MINUTE], list(MODELS), "qlike")
    rng = np.random.default_rng(BOOT_SEED)
    out = {"test_days": {t: int(len(D[t])) for t in TARGETS},
           "H1_combination_vs_mcsgarch": family(D, "mcsgarch", "combination"),
           "H1b_combination_vs_mcs_cpp": family(D, "mcs_cpp", "combination"),
           "secondary": {}}
    for t in TARGETS:
        two = lambda w, b: diebold_mariano(D[t][w], D[t][b], alternative="two-sided")
        out["secondary"][t] = {
            "S1_hybrid_vs_combination_two_sided": two("combination", "hybrid"),
            "S2_share_of_hybrid_gain_recovered": dict(zip(("estimate", "ci_low", "ci_high"), recovered_share(D[t], rng))),
            "S3_H1_without_09_15": dm_one_sided(D_no_open[t], "mcsgarch", "combination"),
            "S4_hybrid_vs_mcs_cpp": dm_one_sided(D[t], "mcs_cpp", "hybrid"),
            "S5_mcsgarch_vs_mcs_cpp_two_sided": two("mcsgarch", "mcs_cpp")}
    return out


# ---------------------------------------------------------------- command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["count", "run"])
    ap.add_argument("--raw", default="data/raw", help="folder with nifty_1min.csv and index_daily.csv")
    ap.add_argument("--forecasts", help="run: per-anchor forecasts on the forward days (see the docstring)")
    ap.add_argument("--i-confirm-stopping-point", action="store_true",
                    help="run: required; confirms this is the single registered run at the stopping point")
    ap.add_argument("--out", default="results/forward_test_p112")
    a = ap.parse_args(argv)
    raw = Path(a.raw)
    missing = [n for n in ("nifty_1min.csv", "index_daily.csv") if not (raw / n).exists()]
    if missing:
        print(f"Missing in {raw}: {missing}")
        return 2
    days, st = test_days(raw)
    print(f"P112 forward test ({PLAN}). Counts only: no forecast and no loss is computed in this step.")
    print(json.dumps({k: v for k, v in st.items()}, indent=2))
    if a.mode == "count":
        if not st["reached"]:
            print("stopping point not reached")
        else:
            print("stopping point reached: run once with `run --i-confirm-stopping-point` and the pinned code")
        return 0
    if not st["reached"]:
        print(f"REFUSED: stopping point not reached ({st['valid_test_days']} of {STOP_DAYS} valid days, cap "
              f"{CAP.date()}). The plan allows one run, at the stopping point; nothing was computed.")
        return 3
    if not a.i_confirm_stopping_point:
        print("REFUSED: the stopping point is reached, but the single registered run needs "
              "--i-confirm-stopping-point. Nothing was computed.")
        return 3
    if not a.forecasts:
        print("REFUSED: --forecasts is required (see the docstring). Nothing was computed.")
        return 2
    stop = pd.Timestamp(st["stop_day"])
    valid = days.index[days["valid"].to_numpy() & (days.index <= stop)]
    res = analyse(pd.read_csv(a.forecasts), valid)
    if st["underpowered"]:  # plan section 3: fewer than 120 valid days at the cap, no verdict
        for fam in ("H1_combination_vs_mcsgarch", "H1b_combination_vs_mcs_cpp"):
            for t in TARGETS:
                res[fam][t]["verdict"] = "underpowered: descriptive only, no verdict"
    res = {"plan": PLAN, "stopping": st, "underpowered_descriptive_only": bool(st["underpowered"]), **res}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    days.to_csv(out / "test_days.csv", index_label="date")
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
