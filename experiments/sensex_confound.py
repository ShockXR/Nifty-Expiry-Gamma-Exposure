"""Is BSE's Sensex weekly expiry a confounder for the NSE expiry studies? REAL DATA.

Question
--------
docs/EVENT_STUDY.md (finding 6, Section 5.6) found end-of-day variance premia
on weekdays when no NSE index expires, and those weekdays match the weekday on
which BSE's Sensex weekly options are reported to expire: Friday from the May
2023 relaunch to 3 Jan 2025, Tuesday from January to August 2025, Thursday
from September 2025. This script (i) measures the end-of-day premium of Nifty
and Bank Nifty on Sensex expiry days that are not NSE expiry days, (ii) asks
whether the premium moves when Sensex moves, with placebo dates as in the
event study, and (iii) reruns the key NSE results (R2 in
docs/REAL_DATA_RESULTS.md and the event study) with a Sensex control.

Data
----
* NSE minute bars, bhavcopies and the NSE expiry calendar: built exactly as in
  experiments/expiry_event_study.py (its functions are imported, not copied).
* Sensex calendar: data/external/sensex_expiry_calendar.csv. BSE's own servers
  are blocked from the machine this was built on, so no row is "verified":
  the weekday regimes come from news and broker summaries of BSE and SEBI
  circulars, and 73 dates (Jan 2024 to Jun 2025) come from third-party
  Sensex option prices (data/external/sensex_nearest_expiry_quantresearch.csv).
  The script turns the reported weekday regimes into dates with the
  holiday roll-back rule (previous NSE trading day) and checks them against
  the 73 dates seen in option prices.
* Sensex 1-minute bars (technovusin/sensex-historical-data, same provider as
  the Nifty bars), only for a calendar check: does the Sensex index itself
  carry an end-of-day premium on the rule's Sensex expiry days?

Pre-stated main test (written into this docstring before the script was first
run; the all-weekday table of EVENT_STUDY.md Section 5.6 had already been
seen, so this is a sharper check of a pattern noticed after the fact, not an
independent discovery)
------------------------------------------------------------------------------
Natural experiment S1. BSE moved the Sensex weekly expiry from Friday to
Tuesday: the last Friday expiry was 3 Jan 2025 and the first Tuesday expiry
7 Jan 2025, so the event week is the week of Monday 6 Jan 2025. No NSE weekly
expiry changed between event c (week of 18 Nov 2024) and event d (week of
1 Sep 2025), and Nifty's own weekly expiry stayed on Thursday.

  Outcome: for Nifty, the week-relative log variance of the last 30 minutes
    (rv_last30) on Tuesday minus the same on Friday of the same week, built
    exactly as in the event study (other days of the week as the baseline),
    in "clean weekly" weeks (no monthly expiry of Nifty or Bank Nifty, no
    holiday shift of an NSE expiry).
  Windows: 26 weeks before the event week and 26 weeks from it. The pre
    window starts in July 2024 and therefore contains event c. Tuesday minus
    Friday is not affected by a change on Wednesday or Thursday (for a full
    week it equals 1.25 times the raw log difference between Tuesday and
    Friday), and event c changed Wednesday (Bank Nifty's weekly) and contract
    terms on all days.
  H1: the change (post minus pre) is positive: the end-of-day premium moves
    from Friday to Tuesday with Sensex.
  Inference: Newey-West (HAC, 4 lags) p-value, circular block-bootstrap 95%
    interval (4-week blocks), Mann-Whitney p, and a placebo p from
    randomization over placebo event weeks (Welch t, the event study's
    machinery). Placebo windows may not contain a change of Tuesday's or
    Friday's status: the Sensex relaunch (S0), S1, event d (= S2, Sensex to
    Thursday and Nifty to Tuesday) and event e (closing auction) are blocked.
  Decision rule: H1 is supported if the change is positive and both the HAC
    p and the placebo p are below 0.05 (the event study showed that HAC alone
    rejects too often for before/after comparisons). The result is reported
    whatever it is.

Secondary analyses (fixed before the first run; every one is reported)
-----------------------------------------------------------------------
  Calendar check (3 tests): Sensex minus Nifty week-relative last-30-minute
    log variance on the rule's Sensex expiry days, per Sensex regime.
  (i) Levels (12 tests): end-of-day premium (last 30 minutes, last hour) of
    Nifty and Bank Nifty on Sensex expiry days that are not NSE expiry days,
    per Sensex regime (Friday, Tuesday, Thursday), clean weekly weeks,
    baseline = the other days of the week that are not NSE expiry days.
    Levels mix a Sensex effect with ordinary weekday differences.
  (ii) Changes (13 tests): S1 for Bank Nifty (last 30 minutes) and for both
    indices in the last hour and the whole day (rv5) (5); the relaunch S0
    (Friday, last 30 minutes, both indices, post window bounded by event a,
    all events blocked for placebos) (2); Wednesdays on which Sensex expired
    because Thursday was a holiday, against the other Wednesdays of the
    Thursday regime (permutation test, both indices) (2); a pooled day-level
    regression with week and weekday fixed effects, NSE expiry dummies and a
    Sensex dummy (both indices; the Sensex coefficient for the last 30
    minutes and the last hour) (4).
  (iii) NSE results with a Sensex control (re-estimates, not new
    hypotheses): R2 rerun with Sensex days removed from the baseline (9 regime
    means and 9 pairwise comparisons); the event study's key 6-month tests
    (primary DiD at c, event a switch, event d Tuesday, Thursday, switch and
    DiD) for rv5, last hour and last 30 minutes with Sensex days removed from
    the baselines (18); the own-expiry premium by regime with the same
    control (18); the NSE expiry coefficients of the pooled regression with
    and without the Sensex dummy.

    OMP_NUM_THREADS=1 python experiments/sensex_confound.py [--fetch-sensex]

--fetch-sensex downloads the Sensex minute bars to data/raw/sensex_1min.csv
(about 25 MB from raw.githubusercontent.com). Writes
results/sensex_confound/*.csv, summary.md and
results/sensex_confound_stdout.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


es = _load("expiry_event_study", ROOT / "experiments" / "expiry_event_study.py")

WD = es.WD
CAL_PATH = ROOT / "data" / "external" / "sensex_expiry_calendar.csv"
OBS_PATH = ROOT / "data" / "external" / "sensex_nearest_expiry_quantresearch.csv"
SENSEX_REPO = "technovusin/sensex-historical-data"
MEASURES = ["rv5", "rv_last_hour", "rv_last30"]
LABEL = {"rv5": "whole day (rv5)", "rv_last_hour": "last hour", "rv_last30": "last 30 minutes"}
S1_WEEK = pd.Timestamp("2025-01-06")  # week of the first Tuesday Sensex expiry (7 Jan 2025)
BN_SAMPLE_START = pd.Timestamp("2022-10-17")  # first week with bhavcopy-verified NSE expiries


# ---------------------------------------------------------------------------
# 1. Sensex minute bars (optional download)
# ---------------------------------------------------------------------------


def sensex_paths(today: pd.Timestamp) -> list[str]:
    paths = [f"1min/{y}/SENSEX_1min_{y}.csv" for y in (2023, 2024, 2025)]
    for y in range(2026, today.year + 1):
        for m in range(1, 13 if y < today.year else today.month + 1):
            paths.append(f"1min/{y}/SENSEX_1min_{y}-{m:02d}.csv")
    return paths


def fetch_sensex_bars(out_csv: Path, today: pd.Timestamp, ref: str = "main") -> dict:
    """Download the Sensex 1-minute CSVs and save them in the repo's minute-bar layout."""
    import requests

    frames, files = [], []
    for p in sensex_paths(today):
        r = requests.get(f"https://raw.githubusercontent.com/{SENSEX_REPO}/{ref}/{p}", timeout=120)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        files.append({"path": p, "bytes": len(r.content), "sha256": hashlib.sha256(r.content).hexdigest()})
        frames.append(pd.read_csv(io.BytesIO(r.content)))
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"timestamp": pd.to_datetime(df["Timestamp"]), "open": df["Open"], "high": df["High"],
                        "low": df["Low"], "close": df["Close"]})
    out = out.drop_duplicates("timestamp").sort_values("timestamp")
    out.to_csv(out_csv, index=False)
    meta = {"repo": SENSEX_REPO, "ref": ref, "fetched_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
            "files": files, "rows": int(len(out)), "first": str(out["timestamp"].min()),
            "last": str(out["timestamp"].max())}
    out_csv.with_name("sensex_1min_source.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# 2. Sensex expiry calendar from the reported weekday regimes
# ---------------------------------------------------------------------------


def load_regimes(cal: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """(start, end, weekday) for each 'weekday_regime' row; an empty end means open-ended."""
    r = cal[cal["fact_type"] == "weekday_regime"]
    out = []
    for _, row in r.iterrows():
        end = pd.Timestamp(row["date_end"]) if isinstance(row["date_end"], str) and row["date_end"] else pd.Timestamp("2100-01-01")
        out.append((pd.Timestamp(row["date"]), end, WD.index(row["weekday"])))
    return sorted(out)


def sensex_rule_expiries(trading_days, regimes) -> pd.DataFrame:
    """One expiry per scheduled weekday inside each regime, moved back to the previous
    trading day when the scheduled day is not a trading day (the reported holiday rule).

    Scheduled days after the last trading day are not generated.
    """
    td = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_days))))
    tdset = set(td)
    rows = []
    for start, end, wd in regimes:
        s = start + pd.Timedelta(days=(wd - start.weekday()) % 7)
        stop = min(end, td.max())
        while s <= stop:
            d = s
            while d not in tdset and d > s - pd.Timedelta(days=6):
                d -= pd.Timedelta(days=1)
            if d in tdset:
                rows.append({"expiry": d, "scheduled": s, "weekday": WD[d.weekday()] if d.weekday() < 5 else str(d.weekday()),
                             "scheduled_weekday": WD[wd], "shifted": d != s, "regime_weekday": WD[wd]})
            s += pd.Timedelta(days=7)
    out = pd.DataFrame(rows, columns=["expiry", "scheduled", "weekday", "scheduled_weekday", "shifted", "regime_weekday"])
    return out.drop_duplicates("expiry").sort_values("expiry").reset_index(drop=True)


def compare_dates(rule_dates, other_dates, lo, hi) -> dict:
    """Agreement of two date lists inside [lo, hi]."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    a = {pd.Timestamp(d) for d in rule_dates if lo <= pd.Timestamp(d) <= hi}
    b = {pd.Timestamp(d) for d in other_dates if lo <= pd.Timestamp(d) <= hi}
    return {"from": lo.date(), "to": hi.date(), "n_rule": len(a), "n_other": len(b), "n_both": len(a & b),
            "rule_only": ";".join(sorted(d.date().isoformat() for d in a - b)),
            "other_only": ";".join(sorted(d.date().isoformat() for d in b - a))}


# ---------------------------------------------------------------------------
# 3. Week-relative tables and tests (event-study functions underneath)
# ---------------------------------------------------------------------------


def relative_table(daily: dict, targets: dict, measures, exclude: dict | None = None) -> pd.DataFrame:
    """Long table (index, target, measure, week, target_date, rel, n_other), like
    es.build_relative, for chosen targets. ``targets[k][name]`` is a Series of
    target dates indexed by week; ``exclude[k]`` holds dates never used in a
    baseline (the target day itself is always kept)."""
    rows = []
    for k, tg in targets.items():
        for tname, t in tg.items():
            for m in measures:
                if m not in daily[k]:
                    continue
                wr = es.week_relative(daily[k][m], t, log=True, exclude=None if exclude is None else exclude.get(k))
                if wr.empty:
                    continue
                wr = wr.reset_index()
                wr["index"], wr["target"], wr["measure"] = k, tname, m
                rows.append(wr)
    out = pd.concat(rows, ignore_index=True)
    return out[["index", "target", "measure", "week", "target_date", "rel", "n_other"]]


def event_test(y: pd.Series, event_week, W: int, blocked_weeks, n_boot: int, seed: int, lo=None, hi=None) -> dict:
    """Before/after test on a weekly series, exactly as es.run_event_tests does it."""
    pre, post = es.window_split(y, event_week, W, lo, hi)
    res = es.compare(pre.to_numpy(), post.to_numpy(), n_boot=n_boot, seed=seed)
    grid, vals = y.index, y.to_numpy()
    blocked_idx = [int(np.searchsorted(grid, pd.Timestamp(b))) for b in blocked_weeks]
    ks = es.placebo_event_indices(len(grid), None, W, blocked_idx)
    tp = es.placebo_t_stats(vals, ks, W)
    n_ind = es.n_independent(ks[np.isfinite(tp)], W)
    p_ri, n_pl = es.randomization_p(res["welch_t"], tp, n_ind)
    slope, p_tr = es.pretrend(pre.index, pre.to_numpy())
    res.update({"ri_p": p_ri, "ri_n_placebo": n_pl, "ri_n_independent": n_ind, "pretrend_p": p_tr,
                "pre_first_week": pre.dropna().index.min(), "post_last_week": post.dropna().index.max()})
    return res


def permutation_mean_diff(treated, control, n_perm: int = 9999, seed: int = 0) -> tuple[float, float]:
    """Difference in means (treated minus control) and its two-sided permutation p."""
    x = np.asarray(treated, float)
    z = np.asarray(control, float)
    x, z = x[np.isfinite(x)], z[np.isfinite(z)]
    if len(x) == 0 or len(z) == 0:
        return np.nan, np.nan
    obs = x.mean() - z.mean()
    pool = np.concatenate([x, z])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pool)
        hits += abs(p[:len(x)].mean() - p[len(x):].mean()) >= abs(obs) - 1e-15
    return float(obs), float((1 + hits) / (1 + n_perm))


def within_week_ols(df: pd.DataFrame, y: str, xs: list[str], week: str = "week"):
    """OLS of y on xs with week fixed effects (within transformation), SE clustered by week."""
    import statsmodels.api as sm

    d = df[[y, week] + xs].dropna()
    d = d[d.groupby(week)[y].transform("size") >= 2]
    dm = d[[y] + xs] - d.groupby(week)[[y] + xs].transform("mean")
    keep = [c for c in xs if dm[c].abs().sum() > 1e-12]
    fit = sm.OLS(dm[y].to_numpy(), dm[keep].to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d[week])[0]})
    return pd.DataFrame({"term": keep, "coef": fit.params, "se": fit.bse, "p": fit.pvalues}), int(len(d)), int(d[week].nunique())


def r2_with_exclusion(rdl, dm: pd.DataFrame, exclude_dates) -> pd.DataFrame:
    """R2 of docs/REAL_DATA_RESULTS.md (real_day_level.r2_regimes), with the given
    non-expiry days removed before the week-relative values are built."""
    ex = set(pd.DatetimeIndex(exclude_dates))
    drop = dm.index.isin(list(ex)) & ~dm["is_expiry"].astype(bool)
    return rdl.r2_regimes(dm[~drop])


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------


fmt_ci, fmt_p = es._ci, es._p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "sensex_confound"))
    ap.add_argument("--boot", type=int, default=4999)
    ap.add_argument("--fetch-sensex", action="store_true")
    ap.add_argument("--sensex-ref", default="22e57b89519f54187db7087f648b3c3299a55693",
                    help="git commit of technovusin/sensex-historical-data (pinned; the files used have the hashes in sensex_1min_source.json)")
    ap.add_argument("--until", default="2026-11-02", help="last month of monthly Sensex files to fetch")
    ap.add_argument("--fetch-only", action="store_true", help="download the Sensex bars and stop")
    a = ap.parse_args()
    t0 = time.time()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["REAL DATA. Sensex expiry as a confounder of the NSE expiry studies.", ""]

    if a.fetch_sensex:
        meta = fetch_sensex_bars(raw / "sensex_1min.csv", pd.Timestamp(a.until), ref=a.sensex_ref)
        lines.append(f"Downloaded Sensex minute bars: {meta['rows']} rows, {meta['first']} to {meta['last']}.")
        if a.fetch_only:
            print(lines[-1])
            return

    # --- NSE data, exactly as in the event study
    D = es.build_data(raw)
    daily, expiries = D["daily"], D["expiries"]
    events = es.define_events(D["changes"], D["auction_start"])
    flags = es.week_flags(expiries, daily)
    ev = events.set_index("event")["event_week"]
    auction = D["auction_start"] if D["auction_start"] is not None else pd.Timestamp("2100-01-01")
    lines += D["lines"]
    idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
    official = set(idx.loc[idx["index"].str.lower() == "nifty 50", "date"])
    official_start = min(official)

    # --- Sensex calendar
    cal = pd.read_csv(CAL_PATH, dtype=str, keep_default_na=False)
    regimes = load_regimes(cal)
    # Official days with a full session. Special short sessions (Muhurat) are in NSE's list but are not
    # days on which a weekly contract expires (Sensex expired on Thu 31 Oct 2024, before the Fri 1 Nov
    # 2024 Muhurat session, in the option prices).
    full = sorted(d for d in official if d in set(daily["N"].index))
    sx = sensex_rule_expiries(full, regimes)
    sx_dates = set(sx["expiry"])
    obs = pd.read_csv(OBS_PATH, parse_dates=["trade_date", "expiry_in_symbol"])
    obs_exp = set(obs["expiry_in_symbol"])
    obs_on_day = set(obs.loc[obs["trade_date"] == obs["expiry_in_symbol"], "trade_date"])
    rep = cal[(cal["fact_type"] == "expiry_date") & ~cal["source"].str.startswith("Akashmishra06")]
    rep_dates = set(pd.to_datetime(rep["date"]))
    chk = pd.DataFrame([
        {"check": "rule vs expiry dates in option symbols", **compare_dates(sx_dates, obs_exp, obs["trade_date"].min(), obs["trade_date"].max())},
        {"check": "rule vs dates with option prices on the expiry day", **compare_dates(
            sx_dates & set(obs["trade_date"]), obs_on_day, obs["trade_date"].min(), obs["trade_date"].max())},
        {"check": "individually reported dates found in the rule", **compare_dates(sx_dates, rep_dates, "2023-05-01", "2026-12-31")},
    ])
    chk.loc[2, "rule_only"] = "n/a (only a few dates are reported one by one)"
    chk.loc[2, "n_rule"] = np.nan
    chk["n_rule"] = chk["n_rule"].astype("Int64")
    chk.to_csv(out / "calendar_check.csv", index=False)
    sx_out = sx.copy()
    sx_out["seen_in_option_symbols"] = np.where(sx_out["expiry"] <= obs["trade_date"].max(),
                                                sx_out["expiry"].isin(list(obs_exp)) & (sx_out["expiry"] >= obs["trade_date"].min()), np.nan)
    sx_out["individually_reported"] = sx_out["expiry"].isin(list(rep_dates))
    sx_out["nifty_expiry_same_day"] = sx_out["expiry"].isin(list(expiries.loc[expiries["symbol"] == "NIFTY", "expiry"]))
    sx_out["banknifty_expiry_same_day"] = sx_out["expiry"].isin(list(expiries.loc[expiries["symbol"] == "BANKNIFTY", "expiry"]))
    sx_out["basis"] = "rule: reported weekday regime plus holiday roll-back (NOT individually sourced unless flagged)"
    sx_out.to_csv(out / "sensex_expiry_days_used.csv", index=False)
    lines += ["", "== Sensex calendar (rule from reported regimes, checked against third-party option prices)",
              f"Regimes: {[(s.date().isoformat(), e.date().isoformat(), WD[w]) for s, e, w in regimes]}",
              f"Rule expiries: {len(sx)} from {sx['expiry'].min().date()} to {sx['expiry'].max().date()}, "
              f"{int(sx['shifted'].sum())} moved by a holiday: "
              f"{[d.date().isoformat() for d in sx.loc[sx['shifted'], 'expiry']]}",
              f"Official NSE days treated as no-expiry special sessions (no full Nifty session): "
              f"{sorted(d.date().isoformat() for d in official - set(full))}",
              chk.to_string(index=False)]

    # Sensex bars (calendar check only)
    sx_path = raw / "sensex_1min.csv"
    if sx_path.exists():
        g = es.minute_grid(sx_path)
        df, _, _ = es.daily_from_grid(g)
        ok = es.valid_day_mask(df.index, df["n_bars"], df["flat_share"], official, official_start)
        daily["SX"] = df[ok]
        lines.append(f"Sensex minute bars: {int(ok.sum())} valid days {df.index[ok].min().date()} to "
                     f"{df.index[ok].max().date()} (dropped {int((~ok).sum())}).")
    else:
        lines.append("Sensex minute bars not found (run with --fetch-sensex); calendar check skipped.")

    nse_exp = {"N": set(expiries.loc[expiries["symbol"] == "NIFTY", "expiry"]),
               "BN": set(expiries.loc[expiries["symbol"] == "BANKNIFTY", "expiry"])}
    nse_any = nse_exp["N"] | nse_exp["BN"]
    sx_only = sorted(sx_dates - nse_any)

    # targets and week-relative tables
    targets = {}
    for k in ("N", "BN"):
        days = daily[k].index
        targets[k] = {WD[i]: es.weekday_targets(days, i) for i in range(5)}
        targets[k]["own"] = es.date_targets(days, sorted(nse_exp[k]))
        targets[k]["SXonly"] = es.date_targets(days, sx_only)
    rel = relative_table(daily, targets, MEASURES)                                    # as in the event study
    rel_x_nse = relative_table(daily, {k: {"SXonly": targets[k]["SXonly"]} for k in ("N", "BN")}, MEASURES,
                               exclude={"N": nse_any, "BN": nse_any})                  # baseline without NSE expiry days
    rel_x_sx = relative_table(daily, targets, MEASURES, exclude={"N": sx_dates, "BN": sx_dates})  # Sensex control
    sx_periods = [("Fri", pd.Timestamp("2023-05-15"), S1_WEEK), ("Tue", S1_WEEK, ev["d"]), ("Thu", ev["d"], ev["e"] if "e" in ev else auction)]
    all_tests = []

    def add_test(family, name, est, lo, hi, p, n, extra=None, main=False):
        row = {"family": family, "test": name, "estimate": est, "ci_lo": lo, "ci_hi": hi, "p": p, "n": n, "main": main}
        row.update(extra or {})
        all_tests.append(row)

    # ------------------------------------------------------------------ main test
    blocked_s = [pd.Timestamp("2023-05-15"), S1_WEEK, ev["d"]] + ([ev["e"]] if "e" in ev else [])
    blocked_all = sorted(set(list(ev.to_numpy()) + blocked_s))
    ser = es.Series(rel, flags, ("N", "Tue"), ("N", "Fri"), "clean_weekly").values("rv_last30")
    r = event_test(ser, S1_WEEK, 26, blocked_s, a.boot, es.seed_of("main", "S1", "N"))
    r_all = event_test(ser, S1_WEEK, 26, blocked_all, a.boot, es.seed_of("main", "S1", "N"))
    supported = bool(r["diff"] > 0 and r["hac_p"] < 0.05 and np.isfinite(r["ri_p"]) and r["ri_p"] < 0.05)
    add_test("main", "S1: Nifty Tue minus Fri, last 30 minutes, 6 months", r["diff"], r["boot_lo"], r["boot_hi"],
             r["hac_p"], f"{r['n_pre']}/{r['n_post']}",
             {"mwu_p": r["mwu_p"], "placebo_p": r["ri_p"], "placebo_n": r["ri_n_placebo"],
              "placebo_independent": r["ri_n_independent"], "placebo_p_all_events_blocked": r_all["ri_p"],
              "pretrend_p": r["pretrend_p"], "mean_pre": r["mean_pre"], "mean_post": r["mean_post"],
              "mde80": 2.8 * r["hac_se"]}, main=True)
    lines += ["", "== MAIN PRE-STATED TEST (S1: Sensex Friday -> Tuesday, week of 6 Jan 2025)",
              f"Nifty, Tuesday minus Friday, week-relative log rv_last30, clean weekly weeks, 26 weeks each side: "
              f"pre mean {r['mean_pre']:+.3f} ({r['n_pre']} weeks, from {r['pre_first_week'].date()}), "
              f"post mean {r['mean_post']:+.3f} ({r['n_post']} weeks, to {r['post_last_week'].date()})",
              f"change {r['diff']:+.3f}, bootstrap 95% [{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}], HAC p {r['hac_p']:.4f}, "
              f"Mann-Whitney p {r['mwu_p']:.4f}, placebo p {r['ri_p']:.4f} ({r['ri_n_placebo']} placebo weeks, "
              f"{r['ri_n_independent']} non-overlapping), placebo p with every event blocked {r_all['ri_p']:.4f}, "
              f"pre-trend p {r['pretrend_p']:.3f}",
              f"Minimum detectable change at 80% power (2.8 x HAC standard error): {2.8 * r['hac_se']:.3f}",
              f"Implied Sensex-day effect on Nifty's last 30 minutes (change / 2.5, full weeks): {r['diff'] / 2.5:+.3f} log points",
              f"Decision (H1 needs change > 0, HAC p < 0.05 and placebo p < 0.05): {'SUPPORTED' if supported else 'NOT SUPPORTED'}"]

    # ------------------------------------------------------------------ calendar check with Sensex bars
    fp_rows = []
    if "SX" in daily:
        sx_days = daily["SX"].index
        tg_sx = {"SX": {WD[i]: es.weekday_targets(sx_days, i) for i in range(5)},
                 "N": {WD[i]: targets["N"][WD[i]] for i in range(5)}}
        tg_sx["SX"]["sxday"] = es.date_targets(sx_days, sorted(sx_dates))
        tg_sx["N"]["sxday"] = es.date_targets(daily["N"].index, sorted(sx_dates))
        rel_sx = relative_table(daily, tg_sx, ["rv_last30"])
        flags_sx = flags.copy()
        for wd in WD + ["sxday"]:
            d = es.Series(rel_sx, flags_sx, ("SX", wd), ("N", wd), "clean_weekly").values("rv_last30")
            own = es.Series(rel_sx, flags_sx, ("SX", wd), None, "clean_weekly").values("rv_last30")
            # descriptive sub-periods added after the first run (no tests): before the relaunch, and the
            # parts of each regime that the option prices do not cover
            extra = [("pre-relaunch (no Sensex weekly)", pd.Timestamp("2023-01-02"), pd.Timestamp("2023-05-15")),
                     ("Fri, 2023 only (no option-price dates)", pd.Timestamp("2023-05-15"), pd.Timestamp("2024-01-01")),
                     ("Tue, Jun to Aug 2025 (no option-price dates)", pd.Timestamp("2025-06-02"), ev["d"])]
            for pname, lo, hi in sx_periods + extra:
                x = d[(d.index >= lo) & (d.index < hi)].dropna()
                xo = own[(own.index >= lo) & (own.index < hi)].dropna()
                if wd == "sxday" and (pname, lo, hi) in extra:
                    continue
                if wd == "sxday" and (pname, lo, hi) in sx_periods:
                    s = es.level_summary(x.to_numpy(), n_boot=a.boot, seed=es.seed_of("fp", pname))
                    add_test("calendar check", f"Sensex minus Nifty on Sensex expiry days, {pname} regime, last 30 minutes",
                             s["mean"], s["boot_lo"], s["boot_hi"], s["hac_p"], s["n_weeks"])
                fp_rows.append({"target": wd, "sensex_regime": pname, "weeks": len(x),
                                "sensex_minus_nifty_mean": x.mean() if len(x) else np.nan,
                                "sensex_own_mean": xo.mean() if len(xo) else np.nan})
        fp = pd.DataFrame(fp_rows)
        fp.to_csv(out / "calendar_check_sensex_bars.csv", index=False)
        piv = fp.pivot(index="sensex_regime", columns="target", values="sensex_minus_nifty_mean")[WD + ["sxday"]]
        piv = piv.reindex([p[0] for p in extra[:1] + sx_periods[:1] + extra[1:2] + sx_periods[1:2] + extra[2:] + sx_periods[2:]])
        lines += ["", "== Calendar check: Sensex minus Nifty, week-relative log rv_last30 (means, clean weekly weeks)",
                  piv.round(3).to_string()]

    # ------------------------------------------------------------------ (i) levels
    lvl = []
    for k in ("N", "BN"):
        for m in ("rv_last30", "rv_last_hour"):
            y = es.Series(rel_x_nse, flags, (k, "SXonly"), None, "clean_weekly").values(m)
            for pname, lo, hi in sx_periods:
                x = y[(y.index >= lo) & (y.index < hi)].dropna()
                if len(x) < 5:
                    continue
                s = es.level_summary(x.to_numpy(), n_boot=a.boot, seed=es.seed_of("lvl", k, m, pname))
                lvl.append({"index": k, "measure": m, "sensex_regime": pname, **s,
                            "from_week": x.index.min().date(), "to_week": x.index.max().date()})
                add_test("(i) level", f"{es.NAMES[k]}, Sensex-only days, {pname} regime, {LABEL[m]}",
                         s["mean"], s["boot_lo"], s["boot_hi"], s["hac_p"], s["n_weeks"], {"wilcoxon_p": s["wilcoxon_p"]})
    lvl = pd.DataFrame(lvl)
    lvl.to_csv(out / "i_levels_sensex_only_days.csv", index=False)

    # ------------------------------------------------------------------ (ii) changes
    chg = []
    for k, m in (("BN", "rv_last30"), ("N", "rv_last_hour"), ("BN", "rv_last_hour"), ("N", "rv5"), ("BN", "rv5")):
        y = es.Series(rel, flags, (k, "Tue"), (k, "Fri"), "clean_weekly").values(m)
        rr = event_test(y, S1_WEEK, 26, blocked_s, a.boot, es.seed_of("S1", k, m))
        chg.append({"event": "S1", "series": f"{k} Tue minus Fri", "measure": m, **rr})
        add_test("(ii) change", f"S1: {es.NAMES[k]} Tue minus Fri, {LABEL[m]}, 6 months", rr["diff"], rr["boot_lo"],
                 rr["boot_hi"], rr["hac_p"], f"{rr['n_pre']}/{rr['n_post']}", {"placebo_p": rr["ri_p"], "mwu_p": rr["mwu_p"]})
    chg.insert(0, {"event": "S1", "series": "N Tue minus Fri", "measure": "rv_last30", **r})
    s0 = pd.Timestamp("2023-05-15")
    for k in ("N", "BN"):
        y = es.Series(rel, flags, (k, "Fri"), None, "clean_weekly").values("rv_last30")
        rr = event_test(y, s0, 26, blocked_all, a.boot, es.seed_of("S0", k), hi=ev["a"])
        chg.append({"event": "S0", "series": f"{k} Fri", "measure": "rv_last30", **rr})
        add_test("(ii) change", f"S0 relaunch: {es.NAMES[k]} Friday, last 30 minutes", rr["diff"], rr["boot_lo"],
                 rr["boot_hi"], rr["hac_p"], f"{rr['n_pre']}/{rr['n_post']}", {"placebo_p": rr["ri_p"], "mwu_p": rr["mwu_p"]})
    chg = pd.DataFrame(chg)
    chg.to_csv(out / "ii_change_tests.csv", index=False)

    # holiday-moved Sensex Wednesdays in the Thursday regime
    hol = []
    wed_sx = sorted(d for d in sx_dates if d.weekday() == 2 and d >= ev["d"] and d < auction)
    for k in ("N", "BN"):
        days = daily[k].index
        wed = days[(days.weekday == 2) & (days >= ev["d"]) & (days < auction)]
        wed = wed[~wed.isin(list(nse_any))]
        wr = es.week_relative(daily[k]["rv_last30"], es.weekday_targets(wed, 2), exclude=nse_any)
        is_sx = wr["target_date"].isin(wed_sx)
        diff, p = permutation_mean_diff(wr.loc[is_sx, "rel"], wr.loc[~is_sx, "rel"], seed=es.seed_of("wed", k))
        hol.append({"index": k, "sensex_wednesdays": ";".join(d.date().isoformat() for d in wr.loc[is_sx, "target_date"]),
                    "n_sensex": int(is_sx.sum()), "n_other": int((~is_sx).sum()),
                    "mean_sensex": wr.loc[is_sx, "rel"].mean(), "mean_other": wr.loc[~is_sx, "rel"].mean(),
                    "diff": diff, "perm_p": p})
        add_test("(ii) change", f"Holiday-moved Sensex Wednesdays vs other Wednesdays, {es.NAMES[k]}, last 30 minutes",
                 diff, np.nan, np.nan, p, f"{int(is_sx.sum())}/{int((~is_sx).sum())}")
    hol = pd.DataFrame(hol)
    hol.to_csv(out / "ii_holiday_wednesdays.csv", index=False)

    # pooled day-level regression with week and weekday fixed effects
    cw = set(flags.index[flags["clean_weekly"]])
    reg_rows = []
    for k in ("N", "BN"):
        d = daily[k]
        d = d[(d.index >= BN_SAMPLE_START) & (d.index < auction)]
        p = pd.DataFrame(index=d.index)
        p["week"] = es.week_start(d.index)
        p = p[p["week"].isin(cw)]
        for i in range(1, 5):
            p[f"wd_{WD[i]}"] = (p.index.weekday == i).astype(float)
        own, oth = nse_exp[k], nse_exp["BN" if k == "N" else "N"]
        is_own = p.index.isin(list(own))
        is_oth = p.index.isin(list(oth)) & ~is_own
        if k == "N":
            p["own_thu_to_d"] = (is_own & (p.index < ev["d"])).astype(float)
            p["own_tue_from_d"] = (is_own & (p.index >= ev["d"])).astype(float)
            p["other_nse"] = is_oth.astype(float)
        else:
            p["own_thu_to_a"] = (is_own & (p.index < ev["a"])).astype(float)
            p["own_wed_a_to_c"] = (is_own & (p.index >= ev["a"])).astype(float)
            p["other_nse_thu_to_d"] = (is_oth & (p.index < ev["d"])).astype(float)
            p["other_nse_tue_from_d"] = (is_oth & (p.index >= ev["d"])).astype(float)
        p["sensex"] = p.index.isin(list(sx_dates)).astype(float)
        base = [c for c in p.columns if c.startswith(("wd_", "own", "other"))]
        for m in MEASURES:
            p["y"] = np.log(d[m].reindex(p.index).where(lambda s: s > 0))
            for spec, xs in (("without Sensex dummy", base), ("with Sensex dummy", base + ["sensex"])):
                tab, n, nw = within_week_ols(p, "y", xs)
                for _, t in tab.iterrows():
                    if t["term"].startswith("wd_"):
                        continue
                    reg_rows.append({"index": k, "measure": m, "spec": spec, "term": t["term"], "coef": t["coef"],
                                     "se": t["se"], "p": t["p"], "n_days": n, "n_weeks": nw})
                    if t["term"] == "sensex" and m in ("rv_last30", "rv_last_hour"):
                        add_test("(ii) change", f"Pooled regression, Sensex dummy, {es.NAMES[k]}, {LABEL[m]}",
                                 t["coef"], t["coef"] - 1.96 * t["se"], t["coef"] + 1.96 * t["se"], t["p"], n)
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(out / "ii_iii_pooled_regression.csv", index=False)

    # ------------------------------------------------------------------ (iii) NSE results with a Sensex control
    # R2 (docs/REAL_DATA_RESULTS.md)
    rdl = _load("real_day_level", ROOT / "experiments" / "real_day_level.py")
    dm = pd.read_csv(ROOT / "results" / "pipeline_nse_v2" / "daily_market.csv", index_col=0, parse_dates=True)
    r2_orig = rdl.r2_regimes(dm)
    r2_ctrl = r2_with_exclusion(rdl, dm, sx_dates)
    pub = pd.read_csv(ROOT / "results" / "real_day_level" / "r2_expiry_effect_by_regime.csv")
    rep_ok = np.allclose(pub["mean_rel"].to_numpy(float), r2_orig["mean_rel"].to_numpy(float))
    r2 = r2_orig.merge(r2_ctrl, on=["measure", "regime"], suffixes=("", "_sensex_control"))
    r2.to_csv(out / "iii_r2_with_sensex_control.csv", index=False)
    for _, t in r2_ctrl.iterrows():
        pp = t["wilcoxon_p_rel_eq_0"] if pd.notna(t.get("wilcoxon_p_rel_eq_0")) else t.get("mannwhitney_p")
        add_test("(iii) re-estimate", f"R2 with Sensex days out of the baseline: {t['measure']}, {t['regime']}",
                 t["mean_rel"], t.get("ci95_low", np.nan), t.get("ci95_high", np.nan), pp, t["weeks"])

    # Event study key tests, Sensex days out of the baselines
    a_, c_, d_ = ev["a"], ev["c"], ev["d"]
    e_ = ev["e"] if "e" in ev else None
    specs = [
        {"id": "c_DiD_main", "event": "c", "label": "DiD main: BN Wed minus Nifty Thu", "a": ("BN", "Wed"), "b": ("N", "Thu"), "lo": a_, "hi": d_},
        {"id": "a_BN_switch", "event": "a", "label": "BN Wed minus BN Thu", "a": ("BN", "Wed"), "b": ("BN", "Thu"), "hi": c_},
        {"id": "d_N_Tue", "event": "d", "label": "Nifty Tuesday (new expiry)", "a": ("N", "Tue"), "lo": c_, "hi": e_},
        {"id": "d_N_Thu", "event": "d", "label": "Nifty Thursday (former expiry)", "a": ("N", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_N_switch", "event": "d", "label": "Nifty Tue minus Nifty Thu", "a": ("N", "Tue"), "b": ("N", "Thu"), "lo": c_, "hi": e_},
        {"id": "d_DiD_Thu", "event": "d", "label": "DiD: Nifty Thu minus BN Thu", "a": ("N", "Thu"), "b": ("BN", "Thu"), "lo": c_, "hi": e_},
    ]
    blocked_es = list(ev.to_numpy())
    et_orig = es.run_event_tests(rel, flags, events, specs, [("6 months", 26)], blocked_es, a.boot, measures=MEASURES)
    et_ctrl = es.run_event_tests(rel_x_sx, flags, events, specs, [("6 months", 26)], blocked_es, a.boot, measures=MEASURES)
    pub_et = pd.read_csv(ROOT / "results" / "event_study" / "event_tests.csv")
    chk_et = et_orig.merge(pub_et, on=["spec_id", "measure", "window"], suffixes=("", "_pub"))
    et_rep = float(np.nanmax(np.abs(chk_et["diff"] - chk_et["diff_pub"]))) if len(chk_et) else np.nan
    cols = ["spec_id", "measure", "n_pre", "n_post", "diff", "boot_lo", "boot_hi", "hac_p", "ri_p", "pretrend_p"]
    et = et_orig[cols].merge(et_ctrl[cols], on=["spec_id", "measure"], suffixes=("", "_sensex_control"))
    et.to_csv(out / "iii_event_tests_with_sensex_control.csv", index=False)
    for _, t in et.iterrows():
        add_test("(iii) re-estimate", f"Event study {t['spec_id']}, {LABEL[t['measure']]}, Sensex days out of baselines",
                 t["diff_sensex_control"], t["boot_lo_sensex_control"], t["boot_hi_sensex_control"],
                 t["hac_p_sensex_control"], f"{t['n_pre_sensex_control']}/{t['n_post_sensex_control']}",
                 {"placebo_p": t["ri_p_sensex_control"]})

    # Own-expiry premium by regime, Sensex days out of the baseline
    regs = [("R0v", BN_SAMPLE_START, a_), ("R1", a_, c_), ("R2", c_, d_), ("R3", d_, e_ if e_ is not None else auction)]
    own_rows = []
    for k in ("N", "BN"):
        for m in MEASURES:
            for tag, rt in (("as published", rel), ("Sensex days out of baseline", rel_x_sx)):
                y = es.Series(rt, flags, (k, "own"), None, "clean_weekly").values(m)
                for rname, lo, hi in regs:
                    x = y[(y.index >= lo) & (y.index < hi)].dropna()
                    if len(x) < 5:
                        continue
                    s = es.level_summary(x.to_numpy(), n_boot=a.boot, seed=es.seed_of(k, "own", m, rname, "clean_weekly"))
                    own_rows.append({"index": k, "measure": m, "regime": rname, "version": tag, **s})
                    if tag != "as published":
                        add_test("(iii) re-estimate", f"Own-expiry premium, {es.NAMES[k]} {rname}, {LABEL[m]}, Sensex days out of baseline",
                                 s["mean"], s["boot_lo"], s["boot_hi"], s["hac_p"], s["n_weeks"])
    own_df = pd.DataFrame(own_rows)
    own_df.to_csv(out / "iii_own_premium_with_sensex_control.csv", index=False)

    # ------------------------------------------------------------------ multiple testing and summary
    tests = pd.DataFrame(all_tests)
    tests["bh_q_within_family"] = np.nan
    for fam, g in tests.groupby("family"):
        tests.loc[g.index, "bh_q_within_family"] = es.bh_qvalues(pd.to_numeric(g["p"], errors="coerce").to_numpy())
    tests.to_csv(out / "all_tests.csv", index=False)
    counts = tests["family"].value_counts().to_dict()

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        lines += ["", "== (i) Levels on Sensex expiry days that are not NSE expiry days (baseline without NSE expiry days)",
                  lvl[["index", "measure", "sensex_regime", "n_weeks", "mean", "boot_lo", "boot_hi", "hac_p", "wilcoxon_p"]].to_string(index=False),
                  "", "== (ii) Change tests", chg[["event", "series", "measure", "n_pre", "n_post", "diff", "boot_lo", "boot_hi", "hac_p", "mwu_p",
                                                  "ri_p", "ri_n_independent", "pretrend_p"]].to_string(index=False),
                  "", "== (ii) Holiday-moved Sensex Wednesdays", hol.to_string(index=False),
                  "", "== (ii)/(iii) Pooled regression (week and weekday fixed effects, SE clustered by week)",
                  reg.to_string(index=False),
                  "", f"== (iii) R2 rerun (reproduces the published R2 table: {rep_ok})",
                  r2[["measure", "regime", "weeks", "mean_rel", "wilcoxon_p_rel_eq_0", "mannwhitney_p", "weeks_sensex_control",
                      "mean_rel_sensex_control", "ci95_low_sensex_control", "ci95_high_sensex_control",
                      "wilcoxon_p_rel_eq_0_sensex_control", "mannwhitney_p_sensex_control"]].to_string(index=False),
                  "", f"== (iii) Event-study key tests, 6 months (largest difference from event_tests.csv as published: {et_rep:.2e})",
                  et.to_string(index=False),
                  "", "== (iii) Own-expiry premium by regime", own_df[["index", "measure", "regime", "version", "n_weeks", "mean", "boot_lo",
                                                                   "boot_hi", "hac_p"]].to_string(index=False),
                  "", f"== Tests run: {len(tests)} p-values ({counts}); one main test.",
                  f"Total run time {time.time() - t0:.0f}s."]

    # summary.md
    md = ["# Sensex expiry confound: summary tables (REAL DATA)", "",
          "Generated by `experiments/sensex_confound.py`. Explained in `docs/SENSEX_CONFOUND.md`.", "",
          "## Main pre-stated test", "",
          f"S1 (Sensex weekly Friday to Tuesday, week of 6 Jan 2025). Nifty, Tuesday minus Friday, last 30 minutes, "
          f"26 weeks each side: change {fmt_ci(r['diff'], r['boot_lo'], r['boot_hi'])}, HAC p {fmt_p(r['hac_p'])}, "
          f"placebo p {fmt_p(r['ri_p'])} ({r['ri_n_independent']} non-overlapping placebo windows), "
          f"Mann-Whitney p {fmt_p(r['mwu_p'])}, weeks {r['n_pre']}/{r['n_post']}, minimum detectable change "
          f"(80% power) {2.8 * r['hac_se']:.2f}. Decision: "
          f"{'supported' if supported else 'not supported'}.", "",
          "## Calendar check", "", es._md(chk.astype(object).where(chk.notna(), "n/a")), "",
          "## All tests", "", "| family | test | estimate [95% interval] | p | BH q (family) | n |", "| --- | --- | --- | --- | --- | --- |"]
    for _, t in tests.iterrows():
        md.append(f"| {t['family']} | {t['test']} | {fmt_ci(t['estimate'], t['ci_lo'], t['ci_hi']) if np.isfinite(t['ci_lo']) else f'{t.estimate:+.2f}'} "
                  f"| {fmt_p(float(t['p']))} | {fmt_p(float(t['bh_q_within_family']))} | {t['n']} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    text = "\n".join(lines)
    (out.parent / "sensex_confound_stdout.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
