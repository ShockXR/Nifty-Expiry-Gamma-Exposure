"""Forward test of H1 and H2, pre-registered in docs/prereg/FORWARD_TEST.md. REAL DATA.

Runs the confirmatory tests that were fixed on 2 November 2026, before any of the data
existed, on days from 3 November 2026 on. Every day of the forward test is a closing-auction
day (from 3 Aug 2026 the index is frozen from 15:16 to 15:28 and the 15:29 bar is the auction
price; the 15:15 bar is a single price equal to the frozen level).

H1  F2 (R5 of docs/REAL_DATA_RESULTS.md): in the binary regression of
    experiments/momentum_gex.py, r_last = a + b r_rod + c r_rod x neg + d neg, the Pro-rule
    interaction c (r_rod_x_neg) is negative. One-sided, HAC (Newey-West) 5 lags.
    Primary r_last: 14:59 to 15:29 bar closes (ends at the auction price).
    Secondary H1b: 14:59 to 15:15 bar closes (stops at the freeze).
H2  The Tuesday expiry-day last-hour premium (R2): week-relative log realized variance of
    the last hour of continuous trading, 14:15 to 15:15 bar closes, is positive. One-sided
    Wilcoxon signed-rank test. Secondary H2b: Sensex expiry days left out of the baseline.

Holm across H1 and H2. Each is evaluated once, at its own stopping point:
    H1: 130 days with negative Pro GEX at the previous close, or 31 Dec 2029, whichever first.
    H2: 100 counting Tuesday-expiry weeks, or 31 Dec 2028, or the first Nifty weekly expiry
        that is not on Tuesday (or the trading day before a Tuesday holiday), whichever first.
Before a stopping point is reached the script reports only counts of days (no peeking).

    # the forward test, once the raw files for 1 Sep 2026 to END are in data/live/forward
    python scripts/forward_test.py --start 2026-11-03 --end END --raw data/live/forward
    # software check on PAST data only (not the forward test)
    python scripts/forward_test.py --dry-run --start 2025-01-01 --end 2026-07-31 --raw data/raw
    # the power arithmetic of the pre-registration (reads only committed result files)
    python scripts/forward_test.py --power

Writes results/forward_test/<start>_<end>/ (a dry run: results/forward_test/dryrun_<start>_<end>/)
with summary.json and summary.md, plus h1_days.csv and h2_weeks.csv for each test evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import momentum_gex as mg  # noqa: E402
import real_day_level as rdl  # noqa: E402

PREREG = "docs/prereg/FORWARD_TEST.md"
FORWARD_START = pd.Timestamp("2026-11-03")
LAST_IN_SAMPLE_DAY = pd.Timestamp("2026-11-02")
AUCTION_START = pd.Timestamp("2026-08-03")

H1_STOP_NEG_DAYS = 130
H1_CAP = pd.Timestamp("2029-12-31")
H1_MIN_EACH = 30  # at least 30 negative-GEX days and 30 other days, or "not evaluable"
H2_STOP_WEEKS = 100
H2_CAP = pd.Timestamp("2028-12-31")
H2_MIN_WEEKS = 20
ALPHA = 0.05
CAP_TOLERANCE_DAYS = 7  # data must reach within 7 calendar days of a cap date
LEAD_IN_DAYS = 45  # calendar days before --start loaded for GEX at the previous close
MIN_LEAD_IN_TRADING_DAYS = 20
TUE_REGIME = "single_weekly_tue"
BAR_TOL_BP = 5.0
MAX_BAD_BAR_SHARE = 0.02

COMMITTED_RUNS = ["results/pipeline_nse_v2", "results/pipeline_nse"]  # first one that exists is used for checks

# Slots are minutes after 09:15 (bar start times), as in experiments/momentum_gex.py.
SLOT_1415, SLOT_1429, SLOT_1459, SLOT_1515, SLOT_1529 = 300, 314, 344, 360, 374
N_SLOTS = 375

DOWNLOAD_HELP = """The raw files are missing or do not cover the window. Fetch them into a folder that is
not data/raw (data/raw is hashed in data/MANIFEST.csv and must not change), for example:

    python scripts/download_nse.py --start 2026-09-01 --end {end} --out data/live/forward
    python scripts/fetch_minute_bars.py --out data/live/forward --first-year 2026 --until {end}

then rerun with --raw data/live/forward. (data/live/ is ignored by git.)"""


# ---------------------------------------------------------------- inputs


def build_inputs(raw: Path, first: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """daily_market and daily_gex_panel built from raw files with the pipeline's own functions and its
    --source nse settings (scripts/run_pipeline.py: load_market, write_daily_market; niftygex.features:
    build_gex_panel), without the forecasting step."""
    import tempfile

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_pipeline as rp  # noqa: E402  (heavy imports: only when building)
    from niftygex.features import build_gex_panel

    a = types.SimpleNamespace(source="nse", raw=str(raw), oi_unit="keep_units", lot_size=None, rate=0.065,
                              start=str(first.date()), end=str(end.date()), min_coverage=0.95)
    m = rp.load_market(a)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "daily_market.csv"
        rp.write_daily_market(m, p)
        dm = pd.read_csv(p, index_col=0, parse_dates=True)
    panel = build_gex_panel(m)
    panel.index = pd.DatetimeIndex(panel.index)
    return dm, panel


def committed_run(root: Path = ROOT) -> Path | None:
    for r in COMMITTED_RUNS:
        if (root / r / "daily_gex_panel.csv").exists():
            return root / r
    return None


def load_inputs(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dm = pd.read_csv(run / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(run / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    return dm, panel


def read_minute_grid(bars_path: Path, days: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """1-minute open, high, low and close on a (day x slot) grid, slot = minutes after 09:15,
    built the way experiments/momentum_gex.daily_returns builds its grid."""
    b = pd.read_csv(bars_path, parse_dates=["timestamp"])
    b = b[(b["timestamp"] >= days.min() - pd.Timedelta(days=7)) & (b["timestamp"] < days.max() + pd.Timedelta(days=1))]
    b["day"] = b["timestamp"].dt.normalize()
    b["slot"] = ((b["timestamp"] - b["day"]) / pd.Timedelta(minutes=1) - 555).astype(int)
    b = b[(b["slot"] >= 0) & (b["slot"] < N_SLOTS)]
    out = {}
    for col in ("open", "high", "low", "close"):
        piv = b.pivot_table(index="day", columns="slot", values=col, aggfunc="last")
        out[col] = piv.reindex(index=days, columns=range(N_SLOTS))
    return out


def official_nifty(raw: Path) -> pd.DataFrame | None:
    p = raw / "index_daily.csv"
    if not p.exists():
        return None
    idx = pd.read_csv(p, parse_dates=["date"])
    return idx[idx["index"].str.lower() == "nifty 50"].set_index("date")[["open", "high", "low", "close"]]


def sensex_expiry_days(trading_days: pd.DatetimeIndex) -> set:
    """Sensex weekly expiry days by the rule of experiments/sensex_confound.py (reported weekday regimes in
    data/external/sensex_expiry_calendar.csv, moved back to the previous trading day on a holiday)."""
    spec = importlib.util.spec_from_file_location("sensex_confound", ROOT / "experiments" / "sensex_confound.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    cal = pd.read_csv(sc.CAL_PATH, dtype=str, keep_default_na=False)
    sx = sc.sensex_rule_expiries(sorted(trading_days), sc.load_regimes(cal))
    return set(pd.DatetimeIndex(sx["expiry"]))


# ---------------------------------------------------------------- H1


def h1_table(close: pd.DataFrame, panel: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per trading day: r_rod, r_last (14:59 to 15:29), r_last_1515 (14:59 to 15:15), all in bp, and the
    Pro-rule GEX at the previous close with its sign dummy, exactly as experiments/momentum_gex.py."""
    c = close.reindex(days)
    prev_close = c[SLOT_1529].shift(1)  # previous trading day's 15:29 bar close
    t = pd.DataFrame(index=days)
    t["r_rod_bp"] = np.log(c[SLOT_1459] / prev_close) * 1e4
    t["r_last_bp"] = np.log(c[SLOT_1529] / c[SLOT_1459]) * 1e4
    t["r_last_1515_bp"] = np.log(c[SLOT_1515] / c[SLOT_1459]) * 1e4
    scale = panel["gross_gex"].rolling(20, min_periods=5).median()
    g = (panel[mg.RULES["participant_pro"]] / scale).shift(1).reindex(days)  # known at the previous close
    t["gex_pro_prev"] = g
    t["neg"] = (g < 0).astype(float).where(g.notna())  # a day with no GEX is missing, not "non-negative"
    return t


def h1_sample(tab: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Days t with start <= t <= end and a complete primary observation. Inputs from day t-1 (GEX and the
    previous close) are known before day t opens; no outcome from outside the window enters."""
    w = tab[(tab.index >= start) & (tab.index <= end)]
    return w.dropna(subset=["r_rod_bp", "r_last_bp", "neg"]).sort_index()


def h1_stopping(sample: pd.DataFrame, end: pd.Timestamp, forward: bool, data_last_day: pd.Timestamp,
                target: int = H1_STOP_NEG_DAYS, cap: pd.Timestamp = H1_CAP) -> dict:
    s = sample[sample.index <= cap] if forward else sample
    cum = (s["neg"] == 1).cumsum()
    info = {"rule": f"{target} days with negative Pro GEX at the previous close, or {cap.date()}, whichever first",
            "observations_so_far": int(len(s)), "negative_gex_days_so_far": int(cum.iloc[-1]) if len(cum) else 0,
            "other_days_so_far": int((s["neg"] == 0).sum())}
    hit = cum.index[cum.to_numpy() >= target]
    if len(hit):
        return {**info, "reached": True, "stop_day": hit[0], "reason": f"{target} negative-GEX days reached"}
    if forward:
        if end >= cap:
            if data_last_day >= cap - pd.Timedelta(days=CAP_TOLERANCE_DAYS):
                return {**info, "reached": True, "stop_day": cap, "reason": f"date cap {cap.date()} reached"}
            return {**info, "reached": False, "stop_day": None,
                    "reason": f"--end is past the cap but the data end on {data_last_day.date()}; download through {cap.date()}"}
        return {**info, "reached": False, "stop_day": None, "reason": "stopping point not reached"}
    return {**info, "reached": True, "stop_day": end,
            "reason": "window end (dry run: past data, the count rule was not reached inside the window)"}


def h1_test(s: pd.DataFrame, ycol: str) -> dict:
    """R5's binary regression on the sample, HAC 5 lags (momentum_gex.fit); one-sided p-values from z."""
    X = pd.DataFrame({"r_rod": s["r_rod_bp"], "neg": s["neg"], "r_rod_x_neg": s["r_rod_bp"] * s["neg"]})
    ok = X.notna().all(axis=1) & s[ycol].notna()
    n_neg, n_oth = int((X.loc[ok, "neg"] == 1).sum()), int((X.loc[ok, "neg"] == 0).sum())
    base = {"outcome": ycol, "n": int(ok.sum()), "negative_gex_days": n_neg, "other_days": n_oth}
    if n_neg < 2 or n_oth < 2:
        return {**base, "estimated": False}
    f = mg.fit(s[ycol], X)
    c, se = float(f.params["r_rod_x_neg"]), float(f.bse["r_rod_x_neg"])
    z = c / se
    return {**base, "estimated": True, "coef_r_rod_x_neg": c, "se_hac": se, "z": z,
            "p_predicted": float(norm.cdf(z)), "p_opposite": float(norm.sf(z)),
            "p_two_sided_statsmodels": float(f.pvalues["r_rod_x_neg"]), "r2": float(f.rsquared),
            "coefficients": {k: float(v) for k, v in f.params.items()}}


def momentum_only(s: pd.DataFrame) -> dict:
    f = mg.fit(s["r_last_bp"], s[["r_rod_bp"]])
    return {"slope_r_rod": float(f.params["r_rod_bp"]), "se_hac": float(f.bse["r_rod_bp"]),
            "p_two_sided": float(f.pvalues["r_rod_bp"]), "n": int(f.nobs)}


# ---------------------------------------------------------------- H2


def rv_between(close: pd.DataFrame, first_slot: int, last_slot: int) -> pd.Series:
    """Sum of squared 1-minute log returns from the close of bar first_slot to the close of bar last_slot.
    Missing if any of those bar closes is missing."""
    c = close.loc[:, first_slot:last_slot]
    lr = np.log(c).diff(axis=1).iloc[:, 1:]
    return (lr ** 2).sum(axis=1).where(c.notna().all(axis=1))


def is_tuesday_rule_expiry(day: pd.Timestamp, trading_days: set) -> bool:
    """True if day is a Tuesday, or the last trading day before a Tuesday that is not a trading day
    (the holiday roll-back). Days after the end of the data count as not trading."""
    if day.weekday() == 1:
        return True
    next_tue = day + pd.Timedelta(days=(1 - day.weekday()) % 7)
    d = day + pd.Timedelta(days=1)
    while d <= next_tue:
        if d in trading_days:
            return False
        d += pd.Timedelta(days=1)
    return True


def h2_weeks(rv: pd.Series, dm: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, exclude=None) -> pd.DataFrame:
    """R2's week-relative measure (real_day_level.week_relative) built from days inside [start, end] only.
    A day without a valid value is dropped first, so a week counts only if its expiry day and at least two
    other days of the week inside the window have values. exclude: non-expiry days to leave out of the baseline."""
    d = dm.loc[(dm.index >= start) & (dm.index <= end), ["is_expiry", "regime", "weekday"]].copy()
    d["is_expiry"] = d["is_expiry"].astype(bool)
    d["val"] = rv.reindex(d.index)
    d = d[d["val"].notna() & (d["val"] > 0)]
    if exclude is not None:
        d = d[~(d.index.isin(pd.DatetimeIndex(sorted(exclude))) & ~d["is_expiry"])]
    cols = ["week", "date", "regime", "weekday", "rel", "n_other", "week_friday", "tuesday_rule"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    wr = rdl.week_relative(d, "val")
    if wr.empty:
        return pd.DataFrame(columns=cols)
    wr["date"] = pd.to_datetime(wr["date"])
    wr["week_friday"] = wr["date"] - pd.to_timedelta(wr["date"].dt.weekday, unit="D") + pd.Timedelta(days=4)
    trading = set(dm.index)
    wr["tuesday_rule"] = [is_tuesday_rule_expiry(x, trading) for x in wr["date"]]
    return wr[cols].sort_values("date").reset_index(drop=True)


def expiry_rule_change(dm: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    """First Nifty expiry day in the window (Tuesday regime) that breaks the Tuesday rule, or None."""
    trading = set(dm.index)
    m = (dm.index >= start) & (dm.index <= end) & dm["is_expiry"].astype(bool).to_numpy() & (dm["regime"] == TUE_REGIME).to_numpy()
    for d in dm.index[m]:
        if not is_tuesday_rule_expiry(d, trading):
            return d
    return None


def h2_stopping(weeks: pd.DataFrame, change_day, end: pd.Timestamp, forward: bool, data_last_day: pd.Timestamp,
                target: int = H2_STOP_WEEKS, cap: pd.Timestamp = H2_CAP) -> tuple[dict, pd.DataFrame]:
    w = weeks[(weeks["regime"] == TUE_REGIME) & weeks["tuesday_rule"].astype(bool) & (weeks["week_friday"] <= end)]
    if change_day is not None:
        w = w[w["date"] < change_day - pd.Timedelta(days=change_day.weekday())]
    if forward:
        w = w[w["week_friday"] <= cap]
    w = w.sort_values("date")
    info = {"rule": f"{target} counting Tuesday-expiry weeks, or weeks ending by {cap.date()}, or a change of the "
                    "expiry weekday, whichever first", "counting_weeks_so_far": int(len(w)),
            "expiry_rule_change_day": change_day}
    if len(w) >= target:
        sel = w.iloc[:target]
        return {**info, "reached": True, "stop_day": sel["week_friday"].iloc[-1], "reason": f"{target} weeks reached"}, sel
    if change_day is not None:
        return {**info, "reached": True, "stop_day": change_day,
                "reason": f"Nifty weekly expiry on {change_day.date()} breaks the Tuesday rule"}, w
    if forward:
        if end >= cap:
            if data_last_day >= cap - pd.Timedelta(days=CAP_TOLERANCE_DAYS):
                return {**info, "reached": True, "stop_day": cap, "reason": f"date cap {cap.date()} reached"}, w
            return {**info, "reached": False, "stop_day": None,
                    "reason": f"--end is past the cap but the data end on {data_last_day.date()}; download through {cap.date()}"}, w.iloc[:0]
        return {**info, "reached": False, "stop_day": None, "reason": "stopping point not reached"}, w.iloc[:0]
    return {**info, "reached": True, "stop_day": end,
            "reason": "window end (dry run: past data, the count rule was not reached inside the window)"}, w


def h2_test(x) -> dict:
    x = np.asarray(x, float)
    out = {"weeks": int(len(x))}
    if len(x) < 2:
        return {**out, "estimated": False}
    boot = np.random.default_rng(0).choice(x, (4000, len(x))).mean(axis=1)  # as real_day_level.r2_regimes
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {**out, "estimated": True, "mean_rel": float(x.mean()), "pct_premium": 100 * (np.exp(x.mean()) - 1),
            "median_rel": float(np.median(x)), "ci95_rel": [float(lo), float(hi)],
            "ci95_pct": [100 * (np.exp(lo) - 1), 100 * (np.exp(hi) - 1)], "share_weeks_positive": float((x > 0).mean()),
            "p_predicted": float(wilcoxon(x, alternative="greater").pvalue),
            "p_opposite": float(wilcoxon(x, alternative="less").pvalue),
            "p_two_sided": float(wilcoxon(x).pvalue),
            "t_test_mean_p_greater": float(ttest_1samp(x, 0.0, alternative="greater").pvalue)}


# ---------------------------------------------------------------- decisions


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def classify(results: dict, alpha: float = ALPHA) -> dict:
    """Pre-registered verdicts for the confirmatory family {H1, H2} (Holm, m = 2 always).

    results[h] = {"status": "evaluated" | "not evaluable" | "not reached" | "data check failed", "p_predicted",
    "p_opposite"}. A failed data check is treated like a stopping point not yet reached: the test is rerun later.
    A hypothesis that is not evaluable counts as p = 1 in Holm. When the other hypothesis has not reached
    its stopping point: p <= alpha/2 is confirmed at once (Holm cannot undo it), p in (alpha/2, alpha] is
    pending (confirmed only if the other ends with p <= alpha/2)."""
    out = {}
    for h, r in results.items():
        other = next(k for k in results if k != h)
        o = results[other]
        if r["status"] == "not reached":
            out[h] = {"verdict": "stopping point not reached", "holm_p": None}
            continue
        if r["status"] == "not evaluable":
            out[h] = {"verdict": "not evaluable", "holm_p": None}
            continue
        if r["status"] == "data check failed":
            out[h] = {"verdict": "not run: minute bars failed the check against NSE; rerun with a corrected source",
                      "holm_p": None}
            continue
        p, q = r["p_predicted"], r["p_opposite"]
        if o["status"] in ("not reached", "data check failed"):
            adj = None
        else:
            adj = holm({h: p, other: o["p_predicted"] if o["status"] == "evaluated" else 1.0})[h]
        if q < alpha:
            verdict = "contradict"
        elif p > alpha:
            verdict = "fail to confirm"
        elif adj is None:
            verdict = "confirm" if p <= alpha / 2 else "pending: confirmed only if the other hypothesis ends with p <= 0.025"
        else:
            verdict = "confirm" if adj <= alpha else "fail to confirm"
        out[h] = {"verdict": verdict, "holm_p": adj}
    return out


def classify_secondary(p: float, q: float, alpha: float = ALPHA) -> str:
    if q < alpha:
        return "opposite direction (secondary, unadjusted)"
    if p <= alpha:
        return "supports (secondary, unadjusted; cannot change a verdict)"
    return "does not support (secondary, unadjusted)"


# ---------------------------------------------------------------- the analysis


def analyse(dm: pd.DataFrame, panel: pd.DataFrame, grid: dict, start: pd.Timestamp, end: pd.Timestamp,
            forward: bool, sensex_days: set | None = None) -> dict:
    """Everything that depends on the data. Returns a dict with counts, stopping information, tests and
    verdicts, plus the day and week tables under the keys '_h1_days' and '_h2_weeks'."""
    days = dm.index
    data_last_day = days.max()
    res: dict = {"window": {"start": start, "end": end, "trading_days_in_window": int(((days >= start) & (days <= end)).sum()),
                            "last_day_in_data": data_last_day}}
    # H1
    tab = h1_table(grid["close"], panel, days)
    s1 = h1_sample(tab, start, end)
    st1 = h1_stopping(s1, end, forward, data_last_day)
    res["h1_stopping"] = st1
    h1 = {"status": "not reached"}
    if st1["reached"]:
        smp = s1[s1.index <= st1["stop_day"]]
        t = h1_test(smp, "r_last_bp")
        ok = t.get("estimated", False) and t["negative_gex_days"] >= H1_MIN_EACH and t["other_days"] >= H1_MIN_EACH
        h1 = {"status": "evaluated" if ok else "not evaluable", **t}
        if ok:
            res["h1b"] = h1_test(smp, "r_last_1515_bp")
            if res["h1b"].get("estimated"):
                res["h1b"]["classification"] = classify_secondary(res["h1b"]["p_predicted"], res["h1b"]["p_opposite"])
            res["h1_descriptive"] = {"momentum_only": momentum_only(smp)}
        res["_h1_days"] = smp
    res["h1"] = h1
    # H2
    rv = rv_between(grid["close"], SLOT_1415, SLOT_1515)
    weeks = h2_weeks(rv, dm, start, end)
    change = expiry_rule_change(dm, start, end)
    st2, sel = h2_stopping(weeks, change, end, forward, data_last_day)
    res["h2_stopping"] = st2
    h2 = {"status": "not reached"}
    if st2["reached"]:
        t = h2_test(sel["rel"])
        ok = t.get("estimated", False) and t["weeks"] >= H2_MIN_WEEKS
        h2 = {"status": "evaluated" if ok else "not evaluable", **t}
        tbl = sel[["week", "date", "weekday", "n_other", "rel"]].rename(columns={"rel": "rel_1415_1515", "n_other": "n_other_1415_1515"})
        if ok:
            if sensex_days is not None:
                wb = h2_weeks(rv, dm, start, end, exclude=sensex_days)
                wb = wb[wb["date"].isin(sel["date"])]
                res["h2b"] = h2_test(wb["rel"])
                if res["h2b"].get("estimated"):
                    res["h2b"]["classification"] = classify_secondary(res["h2b"]["p_predicted"], res["h2b"]["p_opposite"])
                tbl = tbl.merge(wb[["date", "rel", "n_other"]].rename(columns={"rel": "rel_sensex_out", "n_other": "n_other_sensex_out"}),
                                on="date", how="left")
            wo = h2_weeks(dm["rv_last_hour"], dm, start, end)
            wo = wo[wo["date"].isin(sel["date"])]
            res["h2_descriptive"] = {"r2_original_last_hour_1430_1529": h2_test(wo["rel"])}
            tbl = tbl.merge(wo[["date", "rel"]].rename(columns={"rel": "rel_r2_original_1430_1529"}), on="date", how="left")
        res["_h2_weeks"] = tbl
    res["h2"] = h2
    res["decisions"] = classify({"H1": h1, "H2": h2})
    # consistency: the 14:30 to 15:29 measure rebuilt from the grid equals the pipeline's rv_last_hour
    w = (days >= start) & (days <= end)
    rv_orig = rv_between(grid["close"], SLOT_1429, SLOT_1529)[w]
    both = rv_orig.notna() & dm.loc[w, "rv_last_hour"].notna()
    res["check_rv_last_hour_matches_pipeline_max_rel_diff"] = float(
        (rv_orig[both] / dm.loc[w, "rv_last_hour"][both] - 1).abs().max()) if both.any() else None
    return res


# ---------------------------------------------------------------- data checks


def bar_check(grid: dict, official: pd.DataFrame | None, days: pd.DatetimeIndex) -> dict:
    """Minute bars against NSE's official daily high, low and open (as scripts/validate_minute_bars.py)."""
    if official is None or len(days) == 0:
        return {"checked": False}
    hi, lo = grid["high"].loc[days].max(axis=1), grid["low"].loc[days].min(axis=1)
    op = grid["open"].loc[days, 0]
    o = official.reindex(days)
    dh, dl, do = (1e4 * (hi / o["high"] - 1)).abs(), (1e4 * (lo / o["low"] - 1)).abs(), (1e4 * (op / o["open"] - 1)).abs()
    comp = o["high"].notna() & hi.notna()
    bad = comp & ((dh > BAR_TOL_BP) | (dl > BAR_TOL_BP))
    n = int(comp.sum())
    return {"checked": True, "days_compared": n, "days_without_official_values": int((~comp).sum()),
            "high_or_low_off_by_more_than_5bp": int(bad.sum()),
            "open_off_by_more_than_5bp": int((comp & (do > BAR_TOL_BP)).sum()),
            "share_bad": float(bad.sum() / n) if n else None,
            "bad_days": [str(d.date()) for d in days[bad.to_numpy()]][:50],
            "passes": bool(n > 0 and bad.sum() / n <= MAX_BAD_BAR_SHARE)}


def auction_check(grid: dict, official: pd.DataFrame | None, days: pd.DatetimeIndex) -> dict:
    """Counts of days with the auction structure seen from 3 Aug 2026 (pre-registration, section 2)."""
    d = days[days >= AUCTION_START]
    if len(d) == 0:
        return {"auction_days_in_window": 0}
    c, h, lo = grid["close"].loc[d], grid["high"].loc[d], grid["low"].loc[d]
    fr = list(range(SLOT_1515 + 1, SLOT_1529 - 1))  # 15:16 to 15:27
    frozen = (c[fr].eq(c[SLOT_1515], axis=0).all(axis=1) & (h[fr + [SLOT_1515]].to_numpy() == lo[fr + [SLOT_1515]].to_numpy()).all(axis=1))
    cont = list(range(SLOT_1415, SLOT_1515))  # 14:15 to 15:14
    flat_early = (h[cont].to_numpy() == lo[cont].to_numpy()).any(axis=1)
    out = {"auction_days_in_window": int(len(d)),
           "days_15_15_to_15_27_frozen_at_the_15_15_close": int(frozen.sum()),
           "days_with_a_flat_bar_between_14_15_and_15_14": int(flat_early.sum())}
    if official is not None:
        off = official["close"].reindex(d)
        out["days_15_29_close_within_0_5bp_of_official_close"] = int(((c[SLOT_1529] / off - 1).abs() < 5e-5).sum())
    return out


def reference_checks(res: dict, start: pd.Timestamp, end: pd.Timestamp, root: Path = ROOT) -> dict:
    """For the documented dry-run window, compare with numbers already published from the same days:
    FALSIFICATION T4 (Pro binary, 2025 onward, before the auction) and the closing-auction R2 split
    (Tuesday regime before 3 Aug 2026, R2's original last hour)."""
    if (start, end) != (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-07-31")):
        return {"applies": False}
    out = {"applies": True}
    t4p, r2p = root / "results/falsification/t4_preregistered_f2.csv", root / "results/closing_auction/r2_tuesday_split.csv"
    if t4p.exists() and res["h1"].get("estimated"):
        t4 = pd.read_csv(t4p)
        r = t4[(t4["sample"] == "pre_auction") & (t4["rule"] == "participant_pro")].iloc[0]
        out["h1_vs_falsification_t4_pre_auction"] = {
            "published_coef": float(r["coef_2025_onward"]), "here": res["h1"]["coef_r_rod_x_neg"],
            "published_p_one_sided": float(r["p_one_sided"]), "here_p": res["h1"]["p_predicted"],
            "published_days": int(r["n_2025_onward"]), "here_days": res["h1"]["n"],
            "published_negative_days": int(r["neg_days_2025_onward"]), "here_negative_days": res["h1"]["negative_gex_days"],
            "match": bool(abs(float(r["coef_2025_onward"]) - res["h1"]["coef_r_rod_x_neg"]) < 1e-8
                          and int(r["n_2025_onward"]) == res["h1"]["n"])}
    d = res.get("h2_descriptive", {}).get("r2_original_last_hour_1430_1529", {})
    if r2p.exists() and d.get("estimated"):
        r2 = pd.read_csv(r2p)
        r = r2[(r2["measure"] == "rv_last_hour") & (r2["sample"] == "Tuesday regime, before 3 Aug 2026")].iloc[0]
        out["h2_original_last_hour_vs_closing_auction_r2_split"] = {
            "published_pct": float(r["pct_expiry_vs_week"]), "here_pct": d["pct_premium"],
            "published_p_two_sided": float(r["p_signed_rank"]), "here_p_two_sided": d["p_two_sided"],
            "published_weeks": int(r["weeks"]), "here_weeks": d["weeks"],
            "match": bool(abs(float(r["pct_expiry_vs_week"]) - d["pct_premium"]) < 1e-8 and int(r["weeks"]) == d["weeks"])}
    return out


def lead_in_check(dm: pd.DataFrame, start: pd.Timestamp) -> int:
    return int((dm.index < start).sum())


# ---------------------------------------------------------------- provenance


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def folder_digest(folder: Path, first: pd.Timestamp, end: pd.Timestamp, fmt: str, from_stem: bool) -> dict:
    """SHA-256 over (name, file hash) of the files whose date stamp lies in [first, end]."""
    if not folder.exists():
        return {"files": 0, "sha256": None}
    h, n = hashlib.sha256(), 0
    for f in sorted(folder.iterdir()):
        name = f.name
        digits = "".join(ch for ch in (f.stem.split(".")[0] if from_stem else name) if ch.isdigit())[-8:]
        try:
            d = pd.to_datetime(digits, format=fmt)
            if not (first <= d <= end):
                continue
        except ValueError:
            pass
        h.update(name.encode())
        h.update((sha256(f) or "").encode())
        n += 1
    return {"files": n, "sha256": h.hexdigest()}


def git_head() -> dict:
    """The commit checked out, read from .git without running git (None if not available)."""
    g = ROOT / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return {"ref": None, "commit": head}
        ref = head.split(" ", 1)[1]
        if (g / ref).exists():
            return {"ref": ref, "commit": (g / ref).read_text().strip()}
        pr = g / "packed-refs"
        if pr.exists():
            for line in pr.read_text().splitlines():
                if line.endswith(" " + ref):
                    return {"ref": ref, "commit": line.split()[0]}
        return {"ref": ref, "commit": None}
    except OSError:
        return {"ref": None, "commit": None}


CODE_FILES = ["scripts/forward_test.py", "experiments/momentum_gex.py", "experiments/real_day_level.py",
              "experiments/sensex_confound.py", "experiments/expiry_event_study.py", "scripts/run_pipeline.py",
              "data/external/sensex_expiry_calendar.csv", PREREG]


def code_fingerprint() -> dict:
    files = CODE_FILES + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src" / "niftygex").glob("*.py"))
    return {f: sha256(ROOT / f) for f in files}


# ---------------------------------------------------------------- power arithmetic (pre-registration, section 5)


def one_sided_power(effect: float, se: float, alpha: float) -> float:
    """Normal approximation: P(z > z_{1-alpha}) when z ~ N(|effect| / se, 1)."""
    return float(norm.cdf(abs(effect) / se - norm.ppf(1 - alpha)))


def h1_se(n_neg: float, n_other: float, se_in: float, n_neg_in: float, n_other_in: float) -> float:
    """SE of the interaction (a difference of two slopes) scales with sqrt(1/n_neg + 1/n_other)."""
    return se_in * np.sqrt((1 / n_neg + 1 / n_other) / (1 / n_neg_in + 1 / n_other_in))


def h2_se(n_weeks: float, se_in: float, n_in: float) -> float:
    return se_in * np.sqrt(n_in / n_weeks)


def power_inputs(root: Path = ROOT) -> dict:
    reg = pd.read_csv(root / "results/momentum_gex/regressions.csv")
    r = reg[(reg["rule"] == "participant_pro") & (reg["spec"] == "binary") & (reg["subset"] == "all")
            & (reg["term"] == "r_rod_x_neg")].iloc[0]
    gs = pd.read_csv(root / "results/momentum_gex/gamma_states.csv")
    g = gs[(gs["rule"] == "participant_pro") & (gs["subset"] == "all")].set_index("state")["days"]
    t4 = pd.read_csv(root / "results/falsification/t4_preregistered_f2.csv")
    t4 = t4[t4["rule"] == "participant_pro"].set_index("sample")
    run = committed_run(root)
    dm, panel = load_inputs(run)
    scale = panel["gross_gex"].rolling(20, min_periods=5).median()
    gp = (panel[mg.RULES["participant_pro"]] / scale).shift(1).reindex(dm.index)
    tue = gp[(dm.index >= pd.Timestamp("2025-09-01")) & gp.notna()]
    r2 = pd.read_csv(root / "results/real_day_level/r2_expiry_effect_by_regime.csv")
    r2 = r2[(r2["measure"] == "rv_last_hour") & (r2["regime"] == TUE_REGIME)].iloc[0]
    pr = pd.read_csv(root / "results/power_real/headline.csv")
    pr = pr[(pr["test"] == "R2_tue_last_hour") & (pr["stat"] == "wilcoxon")].iloc[0]
    per_year = dm.groupby(dm.index.year).size()
    return {
        "h1_coef": float(r["coef"]), "h1_se_in": float(r["se_hac"]), "h1_n_in": int(r["n"]),
        "h1_neg_in": int(g["GEX < 0"]), "h1_other_in": int(g["GEX >= 0"]),
        "t4_full_n": int(t4.loc["full", "n_2025_onward"]), "t4_full_neg": int(t4.loc["full", "neg_days_2025_onward"]),
        "t4_full_se": float(t4.loc["full", "se_hac_2025_onward"]),
        "t4_pre_n": int(t4.loc["pre_auction", "n_2025_onward"]), "t4_pre_neg": int(t4.loc["pre_auction", "neg_days_2025_onward"]),
        "t4_pre_se": float(t4.loc["pre_auction", "se_hac_2025_onward"]),
        "tue_n": int(len(tue)), "tue_neg": int((tue < 0).sum()),
        "days_2023": int(per_year.get(2023, 0)), "days_2024": int(per_year.get(2024, 0)), "days_2025": int(per_year.get(2025, 0)),
        "days_3nov_31dec_2025": int(((dm.index >= "2025-11-03") & (dm.index <= "2025-12-31")).sum()),
        "h2_mean": float(r2["mean_rel"]), "h2_weeks_in": int(r2["weeks"]),
        "h2_ci": (float(r2["ci95_low"]), float(r2["ci95_high"])),
        "h2_mde80_wild": float(pr["mde80_wild"]), "h2_mde80_residual": float(pr["mde80_residual"]),
        "h2_analytic_2p8_se": float(pr["analytic_mde80_2p8_se"]),
        "run_folder_for_rates": str(run.relative_to(root)),
    }


def power_report(root: Path = ROOT) -> tuple[str, dict]:
    x = power_inputs(root)
    z95, z975, z80 = norm.ppf(0.95), norm.ppf(0.975), norm.ppf(0.80)
    se_h1 = lambda nn, no: h1_se(nn, no, x["h1_se_in"], x["h1_neg_in"], x["h1_other_in"])  # noqa: E731
    k = x["h1_se_in"] / np.sqrt(1 / x["h1_neg_in"] + 1 / x["h1_other_in"])
    eff = abs(x["h1_coef"])
    rate_full = x["t4_full_neg"] / x["t4_full_n"]
    rate_tue = x["tue_neg"] / x["tue_n"]
    rate_auc = (x["t4_full_neg"] - x["t4_pre_neg"]) / (x["t4_full_n"] - x["t4_pre_n"])
    per_year = 245  # 2023 and 2024 had 245 trading days in daily_market.csv, 2025 had 247
    to_2027 = x["days_3nov_31dec_2025"] + per_year
    to_2029 = x["days_3nov_31dec_2025"] + 3 * per_year
    rows1 = []

    def add1(label, nn, no):
        se = se_h1(nn, no)
        rows1.append({"scenario": label, "n_neg": nn, "n_other": no, "se": se,
                      **{f"power_{lab}_{a}": one_sided_power(e, se, a) for lab, e in (("est", eff), ("half", eff / 2))
                         for a in (0.05, 0.025)}})
    add1(f"stop at 130, rate {rate_tue:.1%} (Tuesday regime)", 130, 130 * (1 - rate_tue) / rate_tue)
    add1(f"stop at 130, rate {rate_full:.1%} (2025 onward)", 130, 130 * (1 - rate_full) / rate_full)
    add1(f"cap 31 Dec 2029, rate {rate_auc:.1%} (auction days)", to_2029 * rate_auc, to_2029 * (1 - rate_auc))
    add1(f"NOT CHOSEN: cap 31 Dec 2027, rate {rate_tue:.1%}", to_2027 * rate_tue, to_2027 * (1 - rate_tue))
    add1(f"NOT CHOSEN: stop at 100, rate {rate_tue:.1%}", 100, 100 * (1 - rate_tue) / rate_tue)
    se_eff = x["h2_mde80_wild"] / (z975 + z80)
    se_boot = (x["h2_ci"][1] - x["h2_ci"][0]) / (2 * z975)
    rows2 = []
    for n in (100, 52, H2_MIN_WEEKS):
        se = h2_se(n, se_eff, x["h2_weeks_in"])
        rows2.append({"weeks": n, "se": se, **{f"power_{lab}_{a}": one_sided_power(e, se, a)
                                                for lab, e in (("est", x["h2_mean"]), ("half", x["h2_mean"] / 2))
                                                for a in (0.05, 0.025)}})
    L = ["POWER ARITHMETIC OF docs/prereg/FORWARD_TEST.md (normal approximation; inputs read from committed files)",
         f"one-sided power = Phi(|effect| / SE - z), z = {z95:.4f} (alpha 0.05) or {z975:.4f} (alpha 0.025)", "",
         "H1 inputs:",
         f"  R5 Pro binary r_rod_x_neg: coef {x['h1_coef']:.6f}, HAC SE {x['h1_se_in']:.6f}, n {x['h1_n_in']} "
         "(results/momentum_gex/regressions.csv)",
         f"  negative / other days in sample: {x['h1_neg_in']} / {x['h1_other_in']} (results/momentum_gex/gamma_states.csv)",
         f"  SE(n_neg, n_other) = k sqrt(1/n_neg + 1/n_other), k = {x['h1_se_in']:.6f} / sqrt(1/{x['h1_neg_in']} + 1/{x['h1_other_in']}) = {k:.4f}",
         f"  check on 2025 onward (FALSIFICATION T4): {x['t4_full_neg']} / {x['t4_full_n'] - x['t4_full_neg']} days -> formula "
         f"{se_h1(x['t4_full_neg'], x['t4_full_n'] - x['t4_full_neg']):.4f}, actual {x['t4_full_se']:.4f}; before the auction "
         f"{x['t4_pre_neg']} / {x['t4_pre_n'] - x['t4_pre_neg']} -> {se_h1(x['t4_pre_neg'], x['t4_pre_n'] - x['t4_pre_neg']):.4f}, "
         f"actual {x['t4_pre_se']:.4f}",
         f"  share of days with negative Pro GEX: 2025 onward {x['t4_full_neg']}/{x['t4_full_n']} = {rate_full:.3f}; "
         f"Tuesday regime {x['tue_neg']}/{x['tue_n']} = {rate_tue:.3f} ({x['run_folder_for_rates']}); auction days "
         f"{x['t4_full_neg'] - x['t4_pre_neg']}/{x['t4_full_n'] - x['t4_pre_n']} = {rate_auc:.3f}",
         f"  trading days: {x['days_2023']} (2023), {x['days_2024']} (2024), {x['days_2025']} (2025); {per_year} a year assumed; "
         f"{x['days_3nov_31dec_2025']} from 3 Nov to 31 Dec 2025 used for the rest of 2026; to 31 Dec 2027: {to_2027}, "
         f"to 31 Dec 2029: {to_2029}",
         f"  effect at the estimate {eff:.4f}, at half {eff / 2:.4f}", ""]
    L.append(f"  {'scenario':<48} {'n_neg':>7} {'n_other':>8} {'SE':>8}  est a.05  est a.025  half a.05  half a.025")
    for r in rows1:
        L.append(f"  {r['scenario']:<48} {r['n_neg']:7.1f} {r['n_other']:8.1f} {r['se']:8.5f}  {r['power_est_0.05']:8.0%}"
                 f"  {r['power_est_0.025']:9.0%}  {r['power_half_0.05']:9.0%}  {r['power_half_0.025']:10.0%}")
    L += ["", "H2 inputs:",
          f"  R2 Tuesday last hour: mean {x['h2_mean']:.6f}, {x['h2_weeks_in']} weeks, bootstrap 95% interval "
          f"{x['h2_ci'][0]:.5f} to {x['h2_ci'][1]:.5f} (results/real_day_level/r2_expiry_effect_by_regime.csv)",
          f"  simulated MDE80 of the signed-rank test: residual {x['h2_mde80_residual']:.6f}, wild {x['h2_mde80_wild']:.6f}; "
          f"textbook 2.8 SE {x['h2_analytic_2p8_se']:.6f} (results/power_real/headline.csv)",
          f"  effective SE = less favourable MDE80 / (z_0.975 + z_0.80) = {x['h2_mde80_wild']:.6f} / {z975 + z80:.4f} = {se_eff:.5f} "
          f"(plain SE from the bootstrap interval: {se_boot:.5f}; from 2.8 SE: {x['h2_analytic_2p8_se'] / 2.8:.5f})",
          f"  SE(n) = {se_eff:.5f} sqrt({x['h2_weeks_in']} / n); effect at the estimate {x['h2_mean']:.4f}, at half {x['h2_mean'] / 2:.4f}", "",
          f"  {'weeks':>7} {'SE':>8}  est a.05  est a.025  half a.05  half a.025"]
    for r in rows2:
        L.append(f"  {r['weeks']:7d} {r['se']:8.5f}  {r['power_est_0.05']:8.1%}  {r['power_est_0.025']:9.1%}"
                 f"  {r['power_half_0.05']:9.0%}  {r['power_half_0.025']:10.0%}")
    return "\n".join(L), {"inputs": x, "h1": rows1, "h2": rows2, "k": k, "h2_se_eff": se_eff}


# ---------------------------------------------------------------- output


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def write_markdown(path: Path, s: dict) -> None:
    dry = not s["is_forward_test"]
    L = []
    if dry:
        L += [f"# DRY RUN on past data, {s['window']['start']} to {s['window']['end']}: software check only", "",
              "> **This is NOT the forward test.** Every day in this window was already used to find and study the "
              "effects (REAL_DATA_RESULTS R2 and R5, FALSIFICATION T4), so nothing here is evidence for or against H1 "
              "or H2. It only shows that `scripts/forward_test.py` runs end to end, reproduces published numbers and "
              "applies the pre-registered decision rules. The date caps do not apply to past data: each test uses the "
              "whole window unless its count rule (130 negative-GEX days, 100 weeks) is reached inside it.", ""]
    else:
        L += [f"# Forward test, {s['window']['start']} to {s['window']['end']}", "",
              f"Pre-registered in `{PREREG}`. The analysis must be run with the code at the commit that adds that file.", ""]
    L += [f"Mode: {s['mode']}. Code checked out: {s['code']['git_head'].get('commit') or 'unknown'}. "
          f"Inputs: {s['inputs']['source']}.", "", "## Verdicts (pre-registered decision rules)", "",
          "| Hypothesis | Stopping point | Sample | Estimate | One-sided p, predicted direction | Holm p | One-sided p, opposite direction | Verdict |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for h, key in (("H1", "h1"), ("H2", "h2")):
        r, st, dec = s[key], s[f"{key}_stopping"], s["decisions"][h]
        if r["status"] == "not reached":
            samp = (f"{st['observations_so_far']} days, {st['negative_gex_days_so_far']} negative-GEX" if h == "H1"
                    else f"{st['counting_weeks_so_far']} counting weeks")
            L.append(f"| {h} | not reached ({st['reason']}) | {samp} (counts only) | not computed | not computed | | | {dec['verdict']} |")
            continue
        if h == "H1":
            samp = f"{r['n']} days ({r['negative_gex_days']} negative GEX)"
            est = f"r_rod_x_neg {_f(r.get('coef_r_rod_x_neg'))} (HAC SE {_f(r.get('se_hac'))})"
        else:
            samp = f"{r['weeks']} weeks"
            est = f"mean {_f(r.get('mean_rel'))} log points ({_f(r.get('pct_premium'), 3)}% more variance)"
        L.append(f"| {h} | {st['stop_day']} ({st['reason']}) | {samp} | {est} | {_f(r.get('p_predicted'))} | "
                 f"{_f(dec['holm_p'])} | {_f(r.get('p_opposite'))} | {'DRY RUN: ' if dry else ''}{dec['verdict']} |")
    L += ["", "Confirm: Holm p at most 0.05. Contradict: opposite-direction p below 0.05. Otherwise fail to confirm. "
          "Not evaluable: H1 with fewer than 30 negative-GEX or 30 other days, H2 with fewer than 20 weeks.", ""]
    if "h1b" in s or "h2b" in s:
        L += ["## Secondary tests (not in the Holm family; cannot change a verdict)", "",
              "| Test | Sample | Estimate | p, predicted | p, opposite | Reading |", "| --- | --- | --- | --- | --- | --- |"]
        if "h1b" in s and s["h1b"].get("estimated"):
            r = s["h1b"]
            L.append(f"| H1b: r_last 14:59 to 15:15 (stops at the freeze) | {r['n']} days | r_rod_x_neg {_f(r['coef_r_rod_x_neg'])} "
                     f"(SE {_f(r['se_hac'])}) | {_f(r['p_predicted'])} | {_f(r['p_opposite'])} | {r['classification']} |")
        if "h2b" in s and s["h2b"].get("estimated"):
            r = s["h2b"]
            L.append(f"| H2b: Sensex expiry days out of the baseline | {r['weeks']} weeks | mean {_f(r['mean_rel'])} "
                     f"({_f(r['pct_premium'], 3)}%) | {_f(r['p_predicted'])} | {_f(r['p_opposite'])} | {r['classification']} |")
        L.append("")
    if "h1_descriptive" in s or "h2_descriptive" in s:
        L += ["## Descriptive only (no test, no decision)", ""]
        if "h1_descriptive" in s:
            m = s["h1_descriptive"]["momentum_only"]
            L.append(f"* H1 sample, momentum only: slope of r_last on r_rod {_f(m['slope_r_rod'])} (HAC SE {_f(m['se_hac'])}), {m['n']} days.")
        if "h2" in s and s["h2"].get("estimated"):
            r = s["h2"]
            L.append(f"* H2 (14:15 to 15:15): {_f(r['pct_premium'], 3)}% more variance on the expiry day, bootstrap 95% interval "
                     f"{_f(r['ci95_pct'][0], 3)}% to {_f(r['ci95_pct'][1], 3)}%; expiry day higher in {_f(100 * r['share_weeks_positive'], 3)}% "
                     f"of weeks; t-test of the mean, one-sided p {_f(r['t_test_mean_p_greater'])}; two-sided signed-rank p {_f(r['p_two_sided'])}.")
        if "h2_descriptive" in s:
            r = s["h2_descriptive"]["r2_original_last_hour_1430_1529"]
            if r.get("estimated"):
                L.append(f"* H2 weeks, R2's original last hour (bars 14:30 to 15:29; on auction days it includes the frozen "
                         f"minutes and the auction jump): {_f(r['pct_premium'], 3)}% ({r['weeks']} weeks), two-sided signed-rank p "
                         f"{_f(r['p_two_sided'])}.")
        L.append("")
    L += ["## Data and software checks", ""]
    for k, v in s["checks"].items():
        L.append(f"* {k}: {json.dumps(_clean(v))}")
    L += ["", "Files: `summary.json` (everything above, plus code and data hashes)"
          + (", `h1_days.csv`" if s["h1"]["status"] != "not reached" else "")
          + (", `h2_weeks.csv`" if s["h2"]["status"] != "not reached" else "") + "."]
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--raw", default="data/raw", help="folder with nifty_1min.csv, bhavcopy/, participant_oi/, index_daily.csv")
    ap.add_argument("--run", default=None, help="use daily_market.csv and daily_gex_panel.csv from this folder (made by "
                    "scripts/run_pipeline.py) instead of building them from --raw")
    ap.add_argument("--dry-run", action="store_true", help="software check on past data (end on or before 2026-11-02)")
    ap.add_argument("--out-root", default="results/forward_test")
    ap.add_argument("--power", action="store_true", help="print the power arithmetic of the pre-registration and stop")
    a = ap.parse_args(argv)
    if a.power:
        print(power_report()[0])
        return 0
    if not a.start or not a.end:
        ap.error("--start and --end are required")
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    forward = not a.dry_run
    if end < start:
        ap.error("--end is before --start")
    if forward and start != FORWARD_START:
        ap.error(f"the pre-registered forward test starts on {FORWARD_START.date()}; other windows only with --dry-run on past data")
    if not forward and end > LAST_IN_SAMPLE_DAY:
        ap.error(f"a dry run may use past data only (end on or before {LAST_IN_SAMPLE_DAY.date()})")
    raw = Path(a.raw)
    first = start - pd.Timedelta(days=LEAD_IN_DAYS)
    need = ["nifty_1min.csv", "index_daily.csv"] + ([] if a.run else ["bhavcopy", "participant_oi", "india_vix.csv"])
    missing = [n for n in need if not (raw / n).exists()]
    if missing:
        print(f"Missing in {raw}: {missing}\n" + DOWNLOAD_HELP.format(end=end.date()))
        return 2
    if a.run:
        dm, panel = load_inputs(Path(a.run))
        source = f"{a.run}/daily_market.csv and daily_gex_panel.csv"
    else:
        dm, panel = build_inputs(raw, first, end)
        source = f"built from {raw} with scripts/run_pipeline.py load_market and niftygex.features.build_gex_panel"
    dm = dm[(dm.index >= first) & (dm.index <= end)]
    panel = panel[(panel.index >= first) & (panel.index <= end)]
    if lead_in_check(dm, start) < MIN_LEAD_IN_TRADING_DAYS or dm.index.max() < start:
        print(f"The data must cover at least {MIN_LEAD_IN_TRADING_DAYS} trading days before {start.date()} and the window "
              f"itself; found {lead_in_check(dm, start)} before it, last day {dm.index.max().date() if len(dm) else None}.\n"
              + DOWNLOAD_HELP.format(end=end.date()))
        return 2
    grid = read_minute_grid(raw / "nifty_1min.csv", dm.index)
    official = official_nifty(raw)
    sx = sensex_expiry_days(dm.index)
    res = analyse(dm, panel, grid, start, end, forward, sensex_days=sx)

    wdays = dm.index[(dm.index >= start) & (dm.index <= end)]
    ref = mg.daily_returns(raw / "nifty_1min.csv", dm.index)
    tab = h1_table(grid["close"], panel, dm.index)
    checks = {
        "minute_bars_vs_nse_official": bar_check(grid, official, wdays),
        "closing_auction_structure": auction_check(grid, official, wdays),
        "r_rod_and_r_last_equal_momentum_gex_daily_returns_max_abs_diff_bp": float(np.nanmax(np.abs(np.r_[
            (tab["r_rod_bp"] - ref["r_rod"] * 1e4).to_numpy(), (tab["r_last_bp"] - ref["r_last"] * 1e4).to_numpy()]))),
        "rv_last_hour_rebuilt_from_bars_vs_pipeline_max_rel_diff": res.pop("check_rv_last_hour_matches_pipeline_max_rel_diff"),
        "lead_in_trading_days_before_start": lead_in_check(dm, start),
    }
    if not forward:
        checks["dry_run_reproduces_published_numbers"] = reference_checks(res, start, end)
    committed = committed_run()
    if committed is not None and not a.run:
        cdm, cpanel = load_inputs(committed)
        j = panel.join(cpanel, rsuffix="_committed", how="inner")
        jd = dm.join(cdm[["is_expiry"]], rsuffix="_committed", how="inner")
        if len(j):
            checks["rebuilt_panel_vs_committed_run"] = {
                "committed_run": str(committed.relative_to(ROOT)), "overlapping_days": int(len(j)),
                "max_rel_diff_net_gex_participant": float((j["net_gex_participant"] / j["net_gex_participant_committed"] - 1).abs().max()),
                "sign_mismatches_net_gex_participant": int((np.sign(j["net_gex_participant"]) != np.sign(j["net_gex_participant_committed"])).sum()),
                "expiry_label_mismatches": int((jd["is_expiry"].astype(bool) != jd["is_expiry_committed"].astype(bool)).sum())}
    bars_ok = checks["minute_bars_vs_nse_official"].get("passes", True)
    if forward and not bars_ok:
        for h, key in (("H1", "h1"), ("H2", "h2")):
            if res[key]["status"] != "not reached":
                res[key] = {"status": "data check failed", "reason": "minute bars failed the check against NSE (more than 2% of days)"}
        res["decisions"] = classify({"H1": res["h1"], "H2": res["h2"]})
        res.pop("_h1_days", None), res.pop("_h2_weeks", None)

    if forward:
        out = Path(a.out_root) / f"{start.date()}_{end.date()}"
        label = "FORWARD TEST (pre-registered in docs/prereg/FORWARD_TEST.md)"
    else:
        out = Path(a.out_root) / f"dryrun_{start.date()}_{end.date()}"
        label = ("DRY RUN ON PAST DATA: a software check, NOT the forward test. These days were already used to find "
                 "the effects; nothing here is evidence.")
    out.mkdir(parents=True, exist_ok=True)
    import scipy
    import statsmodels
    summary = {
        "label": label, "is_forward_test": forward, "mode": "forward test" if forward else "dry run on past data",
        "preregistration": PREREG,
        "stopping_rules": {"H1": f"{H1_STOP_NEG_DAYS} negative-GEX days or {H1_CAP.date()}", "H2": f"{H2_STOP_WEEKS} weeks, "
                           f"{H2_CAP.date()}, or an expiry-weekday change", "applied": forward},
        **res,
        "checks": checks,
        "inputs": {"source": source, "raw": str(raw),
                   "sha256_nifty_1min": sha256(raw / "nifty_1min.csv"), "sha256_index_daily": sha256(raw / "index_daily.csv"),
                   "bhavcopy": folder_digest(raw / "bhavcopy", first, end, "%Y%m%d", from_stem=False) if not a.run else None,
                   "participant_oi": folder_digest(raw / "participant_oi", first, end, "%d%m%Y", from_stem=True) if not a.run else None,
                   "minute_bars_source": json.loads((raw / "minute_bars_source.json").read_text())
                   if (raw / "minute_bars_source.json").exists() else None},
        "code": {"git_head": git_head(), "sha256": code_fingerprint()},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
    }
    if "_h1_days" in res:
        res["_h1_days"].to_csv(out / "h1_days.csv", index_label="date")
    if "_h2_weeks" in res:
        res["_h2_weeks"].to_csv(out / "h2_weeks.csv", index=False)
    if forward and res["h1"]["status"] != "not reached" and res["h2"]["status"] != "not reached":
        (out / "inputs").mkdir(exist_ok=True)
        dm.to_csv(out / "inputs" / "daily_market.csv")
        panel.to_csv(out / "inputs" / "daily_gex_panel.csv")
    clean = _clean(summary)
    (out / "summary.json").write_text(json.dumps(clean, indent=2) + "\n")
    write_markdown(out / "summary.md", clean)
    print(label)
    print((out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
