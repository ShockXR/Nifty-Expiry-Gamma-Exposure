"""P117: is the fall in intraday persistence larger than the changes seen between comparable windows that do not
straddle the expiry change, and larger than the change in a control index? (registered plan:
docs/prereg/P117_PERSISTENCE_PLACEBO.md)

P114 found that the fall in the half-life of an intraday volatility shock from the inferred before window
(14 November 2022 to 10 February 2023, 64 days) to the after window (1 March to 31 May 2024, 59 days) survives a
fixed mcsGARCH for Nifty, but that the before window is the most persistent of 44 rolling Nifty windows since
November 2022. This script asks whether the fall is unusual:

    placebo   every pair of windows with the same lengths (64 and 59 trading days) and the same gap between them
              (258 trading days) as the actual pair, sliding one trading day at a time over the data, keeping the
              pairs that do not straddle the expiry changes inside the actual pair (primary set), or no calendar
              change at all (clean set), or not CPP's event b only (literal set); statistic: the change in the log
              half-life, after minus before; p = (1 + placebo pairs with a fall at least as large) / (1 + pairs)
    DiD       the change for Nifty (or Bank Nifty) minus the change for Fin Nifty (whose expiry calendar did not
              change between November 2022 and November 2024) over the actual windows, with P114's day-block
              bootstrap (Fin Nifty resampled on the same days as the treated index in P114's draws)

Models (from P114, unchanged): ``ref`` the replication's mcsGARCH (Table 2) and ``F`` the fixed model (X1 HAR daily
level + X2 overnight decay + X3 today's running mean). Persistence measure: P114's primary, the half-life of the
impulse response to one standardised shock at 12:15 (for ``ref`` it equals ln(0.5) / ln(alpha + beta)).

Stages (from the repository root, raw data in data/raw):

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py inputs
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py windows --index nifty
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py windows --index banknifty
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py control
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py analyze --plan-commit <hash>

Each stage writes to --work (default data/processed/p117, git-ignored); ``windows`` resumes where it stopped.
``analyze`` writes results/p117_persistence_placebo/. ``--synthetic`` runs every stage on a simulated three-index
market (labelled SYNTHETIC). Tests: tests/test_p117_persistence_placebo.py (synthetic only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from niftygex import replication as rp  # noqa: E402
from niftygex.evaluation import holm  # noqa: E402
from niftygex.volatility import daily_egarch_variance, ext_persistence_summary, fit_garch11, fit_garch_ext  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# P114's script (unchanged) gives the grids, the window fit (reference and fixed model), the daily levels and the
# bootstrap step; it loads the replication's script, whose daily EGARCH cache it shares.
p114 = _load("p114_persistence_fixed_model", ROOT / "experiments" / "p114_persistence_fixed_model.py")
rep = p114.rep

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

LABEL_REAL = "REAL DATA (third-party 1-minute bars validated against NSE official daily values)"
LABEL_SYN = "SYNTHETIC (simulated three-index market, not real data)"
TREATED = ("nifty", "banknifty")
CONTROL = "finnifty"
INDEX_NAME = {"nifty": "Nifty 50", "banknifty": "Bank Nifty", "finnifty": "Fin Nifty"}
MODELS = ("ref", "F")
MODEL_LABEL = {"ref": "reference: the replication's mcsGARCH (Table 2)",
               "F": "fixed model F: X1 HAR daily level + X2 overnight decay + X3 today's running mean"}
L_BEFORE, L_AFTER = 64, 59
ACTUAL = {"before": ("2022-11-14", "2023-02-10"), "after": ("2024-03-01", "2024-05-31")}
HISTORY_MIN = 250                          # grid days of data before the first day of a pair's before window
DATA_END = {"nifty": "2026-11-02", "banknifty": "2026-04-22", "finnifty": "2026-01-27"}
HL_CAP = p114.HL_CAP                       # 100,000 minutes: an infinite half-life is capped here for logs
# Expiry-calendar changes of Nifty or Bank Nifty (and the closing auction), from the project's event table
# (results/event_study/events.csv): the first day on which the old schedule no longer applied, i.e. the day after
# the last expiry on the old schedule; for the Nifty weekly launch the event study's long-sample start (4 March
# 2019; weeklies are reported to have started in February 2019), and for the closing auction its first day.
CHANGES = {
    "nifty_weekly": "2019-03-04",          # Nifty weekly options (reported February 2019); event study LONG_START
    "a": "2023-08-25",                     # Bank Nifty weekly Thu -> Wed (last old expiry 24 Aug 2023)
    "b": "2024-03-01",                     # Bank Nifty monthly Thu -> Wed, CPP's event (last old expiry 29 Feb 2024)
    "c": "2024-11-14",                     # Bank Nifty weeklies stop (last weekly 13 Nov 2024; SEBI package)
    "c2": "2024-12-25",                    # Bank Nifty monthly Wed -> last Thu (last old expiry 24 Dec 2024)
    "d": "2025-08-22",                     # Nifty weekly Thu -> Tue, monthlies to Tue (last old weekly 21 Aug 2025)
    "e": "2026-08-03",                     # closing auction sets the official close
}
SETS = {   # name -> the changes a pair must not straddle
    "primary": ("a", "b"),                 # the two changes that lie between the actual windows
    "clean": tuple(CHANGES),               # no Nifty or Bank Nifty calendar change at all
    "literal": ("b",),                     # CPP's event only (the task's literal definition)
}
SET_LABEL = {"primary": "primary set: pairs that straddle neither event a nor event b",
             "clean": "clean set: pairs that straddle no Nifty or Bank Nifty calendar change",
             "literal": "literal set: pairs that do not straddle CPP's event b"}
MIN_PAIRS = 200                            # minimum placebo pairs for a verdict
MIN_DISJOINT = 8.0                         # minimum mean number of day-disjoint pairs (over phases) for a verdict
P_UNUSUAL, P_NORMAL = 0.05, 0.10           # decision thresholds (on Holm-adjusted p)
QUANTILES = (0, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 100)
# DiD bootstrap: P114's draws of the treated index (results/p114_persistence_fixed_model/bootstrap_draws.csv);
# the control is resampled with the treated index's seeds (P114's SEED_INDEX), so the same days are drawn.
DID_KEYS = [("before_inferred", "nifty"), ("after", "nifty"), ("before_inferred", "banknifty"), ("after", "banknifty")]
N_BOOT = p114.N_BOOT
BOOT_CHUNKS = p114.BOOT_CHUNKS
BLOCK_DAYS = p114.BLOCK_DAYS
HOLE_DAYS = 10                             # a step of more than 10 calendar days between grid days is a data hole

SYNTH = dict(n_days=240, seed=117, L_before=24, L_after=20, gap=16, history=40, stride=9, n_boot=6,
             before_row=150, changes_row={"nifty_weekly": 45, "a": 178, "b": 186, "c": 225},
             hole=(20, 32), check_draws=2)


# ---------------------------------------------------------------------------
# Pairs of windows
# ---------------------------------------------------------------------------


def actual_rows(days: pd.DatetimeIndex, before=ACTUAL["before"], after=ACTUAL["after"],
                L_b=L_BEFORE, L_a=L_AFTER) -> dict:
    """Grid rows of the actual pair and the gap between its windows (grid days strictly between them)."""
    b0, b1, a0, a1 = map(pd.Timestamp, (*before, *after))
    s = int(np.searchsorted(days, b0))
    sa = int(np.searchsorted(days, a0))
    if days[s] != b0 or days[s + L_b - 1] != b1 or days[sa] != a0 or days[sa + L_a - 1] != a1:
        raise ValueError("the actual windows do not have the registered lengths on this grid")
    return {"start_row": s, "after_row": sa, "gap": sa - (s + L_b)}


def straddles(first_day, last_day, change) -> np.ndarray:
    """A pair straddles a change if its before window starts before the change and its after window ends on or
    after the change's first day."""
    c = pd.Timestamp(change)
    return (pd.DatetimeIndex(first_day) < c) & (pd.DatetimeIndex(last_day) >= c)


def enumerate_pairs(days: pd.DatetimeIndex, gap: int, L_b: int = L_BEFORE, L_a: int = L_AFTER,
                    history: int = HISTORY_MIN, changes: dict = CHANGES, sets: dict = SETS,
                    actual_start: int | None = None, stride: int = 1) -> pd.DataFrame:
    """Every pair of windows (before: L_b grid days from start_row; after: L_a days starting gap days after the
    before window ends) that fits in the grid with ``history`` grid days before it, every ``stride`` days, with the
    changes it straddles and its set memberships. The actual pair (if given) is added and flagged."""
    T = len(days)
    span = L_b + gap + L_a
    starts = list(range(history, T - span + 1, stride))
    if actual_start is not None and actual_start not in starts:
        starts.append(actual_start)
    s = np.array(sorted(starts), int)
    d = pd.DataFrame({"start_row": s, "after_row": s + L_b + gap})
    d["before_first"] = days[s]
    d["before_last"] = days[s + L_b - 1]
    d["after_first"] = days[s + L_b + gap]
    d["after_last"] = days[s + span - 1]
    for name, c in changes.items():
        d[f"straddles_{name}"] = straddles(d["before_first"], d["after_last"], c)
    d["is_actual"] = d["start_row"] == (-1 if actual_start is None else actual_start)
    d["history_ok"] = d["start_row"] >= history
    for name, avoid in sets.items():
        ok = d["history_ok"] & ~d["is_actual"]
        for c in avoid:
            ok &= ~d[f"straddles_{c}"]
        d[f"in_{name}"] = ok
    return d


def pair_days(start_row: int, L_b: int, gap: int, L_a: int) -> np.ndarray:
    """The grid rows a pair uses (its before and after windows)."""
    return np.r_[np.arange(start_row, start_row + L_b), np.arange(start_row + L_b + gap, start_row + L_b + gap + L_a)]


def placebo_p(delta, d_act: float) -> dict:
    """One-sided placebo p for 'the actual fall is unusually large': (1 + placebo pairs with a change at or below
    the actual change) / (1 + placebo pairs). Pairs without a value are dropped and counted."""
    x = np.asarray(delta, float)
    n_nan = int(np.isnan(x).sum())
    x = x[~np.isnan(x)]
    k = int(np.sum(x <= d_act))
    return {"n": len(x), "n_dropped": n_nan, "k": k, "share": k / len(x) if len(x) else np.nan,
            "p": (1 + k) / (1 + len(x))}


def pairs_share_days(d, L_b: int, gap: int, L_a: int):
    """Whether two pairs whose starts are d >= 0 grid days apart share a grid day: their before windows overlap
    (d < L_b), their after windows overlap (d < L_a), or the later pair's before window overlaps the earlier pair's
    after window (gap < d < L_b + gap + L_a). (The later pair's after window always lies after the earlier pair's
    before window.)"""
    d = np.abs(np.asarray(d))
    return (d < max(L_b, L_a)) | ((d > gap) & (d < L_b + gap + L_a))


def disjoint_selection(starts, L_b: int, gap: int, L_a: int, phase: int) -> list:
    """Greedy day-disjoint selection: walk the pairs in order of their start and keep a pair if it shares no grid
    day with any pair kept so far, starting at the first pair that starts at least ``phase`` days after the first."""
    st = np.sort(np.asarray(starts, int))
    if len(st) == 0:
        return []
    span = L_b + gap + L_a
    keep = []
    for s in st:
        if s < st[0] + phase:
            continue
        recent = [k for k in keep[-span:] if s - k < span]
        if not recent or not pairs_share_days(s - np.array(recent), L_b, gap, L_a).any():
            keep.append(int(s))
    return keep


def disjoint_p(starts, delta, d_act: float, L_b: int, gap: int, L_a: int) -> dict:
    """Conservative placebo p that counts only day-disjoint pairs: for every phase 0 .. span - 1, the greedy
    day-disjoint selection gives p_phase = (1 + kept pairs with a change at or below the actual) / (1 + kept
    pairs); the p is the mean over phases. Pairs without a value are left out before the selection."""
    st = np.asarray(starts, int)
    x = np.asarray(delta, float)
    ok = ~np.isnan(x)
    st, x = st[ok], x[ok]
    val = dict(zip(st.tolist(), x.tolist()))
    span = L_b + gap + L_a
    ps, ns, ks = [], [], []
    for ph in range(span):
        sel = disjoint_selection(st, L_b, gap, L_a, ph)
        k = int(sum(val[s] <= d_act for s in sel))
        ps.append((1 + k) / (1 + len(sel)))
        ns.append(len(sel))
        ks.append(k)
    return {"p": float(np.mean(ps)), "n_mean": float(np.mean(ns)), "n_min": int(np.min(ns)),
            "n_max": int(np.max(ns)), "k_mean": float(np.mean(ks)), "p_min_attainable": float(np.mean(1.0 / (1 + np.array(ns))))}


def separate_spans(starts, span: int) -> dict:
    """How many pairs whose whole spans (first day of the before window to the last day of the after window) do
    not overlap at all can be found, over all phases (greedy)."""
    st = np.sort(np.asarray(starts, int))
    out = []
    for ph in range(span):
        c, nxt = 0, (st[0] + ph if len(st) else 0)
        for s in st:
            if s >= nxt:
                c += 1
                nxt = s + span
        out.append(c)
    return {"min": int(np.min(out)) if out else 0, "max": int(np.max(out)) if out else 0}


def conditional_p(delta, x_before, d_act: float, x_act: float) -> dict:
    """Conditional placebo: OLS of the change on the before window's log half-life over the placebo pairs; the
    actual pair's residual is compared with the placebo residuals, p = (1 + residuals at or below) / (1 + pairs).
    A low p means the after window's half-life is unusually short given how persistent the before window was."""
    y = np.asarray(delta, float)
    x = np.asarray(x_before, float)
    ok = ~(np.isnan(y) | np.isnan(x))
    y, x = y[ok], x[ok]
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    e_act = d_act - (b[0] + b[1] * x_act)
    k = int(np.sum(e <= e_act))
    return {"intercept": float(b[0]), "slope": float(b[1]), "resid_act": float(e_act), "n": len(y), "k": k,
            "p": (1 + k) / (1 + len(y)), "resid_sd": float(np.std(e, ddof=2))}


def verdict(p_holm: float, meets_min: bool) -> str:
    """Registered decision rule for one model and index (placebo tests)."""
    if not meets_min:
        return "no verdict: too few separate placebo periods"
    if not np.isfinite(p_holm):
        return "not computed"
    if p_holm <= P_UNUSUAL:
        return "the fall is unusual"
    if p_holm > P_NORMAL:
        return "the fall is within the normal range"
    return "unclear"


def did_verdict(p_holm: float) -> str:
    """Registered decision rule for the difference-in-differences."""
    if not np.isfinite(p_holm):
        return "not computed"
    if p_holm <= P_UNUSUAL:
        return "the fall is larger than the control's"
    if p_holm > P_NORMAL:
        return "the fall is not clearly larger than the control's"
    return "unclear"


def log_hl(hl) -> np.ndarray:
    return np.log(np.minimum(np.asarray(hl, float), HL_CAP))


def boot_p_below0(d) -> float:
    """One-sided bootstrap p for 'the statistic is below 0': (1 + draws at or above 0) / (1 + draws) (P114)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    return float((1 + np.sum(d >= 0)) / (1 + len(d)))


# ---------------------------------------------------------------------------
# Data and grids
# ---------------------------------------------------------------------------


def synthetic_market() -> dict:
    """SYNTHETIC: three simulated indices (P114's simulator); the third gets a data hole like Fin Nifty's."""
    n = SYNTH["n_days"]
    two = p114.synthetic_bars(n, SYNTH["seed"], beta_change_day=SYNTH["changes_row"]["b"])
    third = p114.synthetic_bars(n, SYNTH["seed"] + 1, beta_change_day=10 ** 6)["nifty"]
    days = pd.DatetimeIndex(sorted(third["date"].unique()))
    h0, h1 = SYNTH["hole"]
    third = third[~third["date"].isin(days[h0:h1])].reset_index(drop=True)
    return {"nifty": two["nifty"], "banknifty": two["banknifty"], "finnifty": third}


def load_index(a, k: str) -> pd.DataFrame:
    if a.synthetic:
        return synthetic_market()[k]
    return rp.load_minute_bars(Path(a.raw) / f"{k}_1min.csv", "2017-01-01", DATA_END[k])


def synthetic_dates(days: pd.DatetimeIndex) -> dict:
    """SYNTHETIC: the actual windows and the change dates on the simulated grid."""
    s = SYNTH["before_row"]
    sa = s + SYNTH["L_before"] + SYNTH["gap"]
    before = (str(days[s].date()), str(days[s + SYNTH["L_before"] - 1].date()))
    after = (str(days[sa].date()), str(days[sa + SYNTH["L_after"] - 1].date()))
    changes = {c: str(days[r].date()) for c, r in SYNTH["changes_row"].items()}
    return {"before": before, "after": after, "changes": changes}


def design(a, days: pd.DatetimeIndex) -> dict:
    """Window lengths, actual windows, history, stride and change dates (registered, or SYNTHETIC)."""
    if a.synthetic:
        sd = synthetic_dates(days)
        return {"L_b": SYNTH["L_before"], "L_a": SYNTH["L_after"], "history": SYNTH["history"],
                "stride": SYNTH["stride"], "before": sd["before"], "after": sd["after"], "changes": sd["changes"],
                "sets": {"primary": ("a", "b"), "clean": tuple(sd["changes"]), "literal": ("b",)}}
    return {"L_b": L_BEFORE, "L_a": L_AFTER, "history": HISTORY_MIN, "stride": 1, "before": ACTUAL["before"],
            "after": ACTUAL["after"], "changes": CHANGES, "sets": SETS}


def egarch_level_joined(G: dict, k: str, last_row: int, hole_days: int = HOLE_DAYS):
    """The replication's daily AR(1)-EGARCH(1,1)-t level for an index whose grid has a data hole (Fin Nifty:
    February to July 2022): the daily return that spans the hole is left out and the series on both sides is joined;
    the day after the hole gets no level (NaN; it is in no window). Stored in the replication's cache, so that
    ``rep.fit_period`` (the reference fit) uses it. Without a hole this is exactly ``rep.daily_variance``."""
    key = (G["trim"], k, last_row)
    if key in rep._EGARCH_CACHE:
        return rep._EGARCH_CACHE[key]
    days = G["days"]
    step = np.r_[0, np.diff(days.values).astype("timedelta64[D]").astype(int)]
    rows = np.arange(1, last_row + 1)
    keep = step[rows] <= hole_days
    if keep.all():
        return rep.daily_variance(G, k, last_row)
    y = G[k]["daily"][rows[keep]]
    h, info = daily_egarch_variance(y)
    hf = np.full(len(days), np.nan)
    hf[rows[keep]] = h
    rep._EGARCH_CACHE[key] = (hf, info["method"] + "|joined over a data hole")
    return rep._EGARCH_CACHE[key]


WINDOW_COLS = ("omega", "alpha", "beta", "gam_x", "delta", "nu", "loglik", "converged", "ab", "hl_ab", "hl_irf",
               "rho_irf", "share_close", "n_obs", "n_days")


def window_rows(G: dict, k: str, rows: np.ndarray, kind: str, label: str) -> list:
    """P114's fit (reference and F) of one window, as table rows."""
    t0 = time.time()
    try:
        fw = p114.fit_window(G, k, rows, models=MODELS)
    except Exception as exc:  # noqa: BLE001 - registered: a window whose fit raises is recorded as not computed
        return [{"data": label, "index": k, "kind": kind, "start_row": int(rows[0]),
                 "first_day": str(G["days"][rows[0]].date()), "last_day": str(G["days"][rows[-1]].date()),
                 "model": m, **{c: np.nan for c in WINDOW_COLS}, "daily_method": f"error: {type(exc).__name__}: {exc}"[:200],
                 "fit_seconds": round(time.time() - t0, 2)} for m in MODELS]
    out = []
    for m in MODELS:
        r = fw[m]["row"]
        out.append({"data": label, "index": k, "kind": kind, "start_row": int(rows[0]),
                    "first_day": str(G["days"][rows[0]].date()), "last_day": str(G["days"][rows[-1]].date()),
                    "model": m, **{c: r[c] for c in WINDOW_COLS},
                    "daily_method": fw["_ref_fit"]["daily_method"], "fit_seconds": round(time.time() - t0, 2)})
    return out


# ---------------------------------------------------------------------------
# Stage inputs
# ---------------------------------------------------------------------------


def stage_inputs(a):
    """SHA-256 of the raw files this study reads, checked against data/MANIFEST.csv."""
    import hashlib
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "data" / "MANIFEST.csv").set_index("path")["sha256"].to_dict()
    res = {}
    for f in ("nifty_1min.csv", "banknifty_1min.csv", "minute_bars_source.json", "finnifty_1min.csv",
              "finnifty_1min_source.json"):
        h = hashlib.sha256()
        with open(Path(a.raw) / f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        res[f] = {"sha256": h.hexdigest(), "manifest": man.get(f), "matches_manifest": man.get(f) == h.hexdigest()}
    json.dump(res, open(work / "inputs.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    if not all(v["matches_manifest"] for v in res.values()):
        raise SystemExit("raw files differ from data/MANIFEST.csv")


# ---------------------------------------------------------------------------
# Stage windows: fit every window that a placebo pair (or the actual pair) needs
# ---------------------------------------------------------------------------


def stage_windows(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    label = LABEL_SYN if a.synthetic else LABEL_REAL
    k = a.index
    t0 = time.time()
    rep._EGARCH_CACHE.clear()
    p114._HAR_CACHE.clear()
    Gi = p114.index_grid(load_index(a, k), k)
    days = Gi["days"]
    ds = design(a, days)
    act = actual_rows(days, ds["before"], ds["after"], ds["L_b"], ds["L_a"])
    pairs = enumerate_pairs(days, act["gap"], ds["L_b"], ds["L_a"], ds["history"], ds["changes"], ds["sets"],
                            actual_start=act["start_row"], stride=ds["stride"])
    pairs.insert(0, "index", k)
    need = pairs[pairs["is_actual"] | pairs[[f"in_{s}" for s in ds["sets"]]].any(axis=1)]
    pairs.to_csv(work / f"pairs_{k}.csv", index=False)
    jobs = [("before", act["start_row"]), ("after", act["after_row"])]
    jobs += sorted({("before", int(s)) for s in need["start_row"]} | {("after", int(s)) for s in need["after_row"]},
                   key=lambda j: (j[1], j[0]))
    jobs = list(dict.fromkeys(jobs))
    path = work / f"windows_{k}.csv"
    done = set()
    if path.exists():
        old = pd.read_csv(path)
        done = set(zip(old["kind"], old["start_row"].astype(int)))
    todo = [j for j in jobs if j not in done]
    print(f"[{label}] P117 windows {k}: {len(days)} grid days, gap {act['gap']}, {len(pairs)} pairs enumerated "
          f"({', '.join(f'{s} {int(pairs[f'in_{s}'].sum())}' for s in ds['sets'])}); {len(jobs)} windows needed, "
          f"{len(todo)} to fit", flush=True)
    buf = []
    for n, (kind, s) in enumerate(todo):
        L = ds["L_b"] if kind == "before" else ds["L_a"]
        buf += window_rows(Gi, k, np.arange(s, s + L), kind, label)
        if len(buf) >= 100 or n == len(todo) - 1:
            pd.DataFrame(buf).to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.17g")
            buf = []
        if (n + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {n + 1} of {len(todo)} windows ({el / 60:.1f} min, about {el / (n + 1) * (len(todo) - n - 1) / 60:.0f}"
                  f" min left)", flush=True)
    json.dump({"index": k, "grid_days": len(days), "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
               **act, "L_before": ds["L_b"], "L_after": ds["L_a"], "windows_needed": len(jobs),
               "windows_fitted_this_run": len(todo), "seconds": round(time.time() - t0, 1)},
              open(work / f"windows_{k}_info.json", "w"), indent=2)
    # P114's committed rolling windows that coincide with a before window here must be reproduced exactly
    if not a.synthetic:
        roll = pd.read_csv(ROOT / "results" / "p114_persistence_fixed_model" / "rolling.csv", float_precision="round_trip")
        mine = pd.read_csv(path, float_precision="round_trip")
        mine = mine[mine["kind"] == "before"]
        chk = []
        for _, r in roll[roll["index"] == k].iterrows():
            m = mine[(mine["first_day"] == r["start"]) & (mine["model"] == r["model"])]
            if m.empty:
                continue
            m = m.iloc[0]
            for q in ("alpha", "beta", "gam_x", "delta", "nu", "hl_irf"):
                chk.append({"index": k, "window_start": r["start"], "model": r["model"], "quantity": q,
                            "this_study": m[q], "p114_committed": r[q], "identical": bool(m[q] == r[q])})
        chk = pd.DataFrame(chk)
        f = work / "reproduction_p114_rolling.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        if len(old):
            old = old[old["index"] != k]
        pd.concat([old, chk], ignore_index=True).to_csv(f, index=False, float_format="%.17g")
        if len(chk):
            print(f"  P114 rolling windows reproduced: {int(chk['identical'].sum())} of {len(chk)} values identical "
                  f"({chk['window_start'].nunique()} windows)", flush=True)
    print(f"windows stage {k} done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Stage control: Fin Nifty on the actual windows, and its day-block bootstrap paired with P114's draws
# ---------------------------------------------------------------------------


def boot_draws(st: dict, seed_index: int, n_boot: int, models=MODELS, chunks=None, per_limit=None) -> pd.DataFrame:
    """P114's bootstrap (boot_key) for one window, with an explicit seed index: circular 5-day blocks drawn with
    seeds 1000 * seed_index + chunk, the diurnal pattern re-estimated and every model refitted on each resample.
    Same random stream as P114, so the same seed index draws the same days for a window of the same length."""
    valid, M = st["valid"], st["M"]
    eps = st["ref"]["eps"]
    T = len(eps)
    n_blocks = int(np.ceil(T / BLOCK_DAYS))
    per = int(np.ceil(n_boot / BOOT_CHUNKS))
    out = []
    for c in (range(BOOT_CHUNKS) if chunks is None else chunks):
        rng = np.random.default_rng(1000 * seed_index + c)
        for b in range(per if per_limit is None else min(per, per_limit)):
            stt = rng.integers(0, T, n_blocks)
            idx = ((stt[:, None] + np.arange(BLOCK_DAYS)[None, :]) % T).ravel()[:T]
            e, v = eps[idx], valid[idx]
            for name in models:
                hh = st[name]["h"][idx]
                x = p114.resample_x(e, hh, v)
                _, decay, lev, todayrv = p114.MODELS[name]
                if name == "ref":
                    p = fit_garch11(x.ravel(), weights=v.ravel().astype(float))
                else:
                    p = fit_garch_ext(x, e < 0, v.astype(float), decay, lev, todayrv)
                s = ext_persistence_summary(p, M, p114.SHOCK_MINUTE)
                out.append({"model": name, "draw": c * per + b, "chunk": c, "seed": 1000 * seed_index + c,
                            "alpha": p["alpha"], "beta": p["beta"], "gam_x": p.get("gam_x", 0.0),
                            "delta": p.get("delta", 1.0), "nu": p.get("nu"), "converged": p.get("converged"),
                            "ab": s["ab"], "hl_irf": s["hl_irf"], "rho_irf": s["rho_irf"]})
    return pd.DataFrame(out)


def store_of(fw: dict) -> dict:
    return {"valid": fw["_valid"], "M": fw["_M"],
            **{m: {"params": fw[m]["params"], "eps": fw[m]["eps"], "h": fw[m]["h"]} for m in MODELS}}


def stage_control(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    label = LABEL_SYN if a.synthetic else LABEL_REAL
    t0 = time.time()
    n_boot = SYNTH["n_boot"] if a.synthetic else a.n_boot
    rep._EGARCH_CACHE.clear()
    p114._HAR_CACHE.clear()
    # the treated indices on the replication's two-index grid (P114's windows), refitted as a check
    if a.synthetic:
        mk = synthetic_market()
        bars2 = {k: mk[k] for k in TREATED}
        fin_bars = mk[CONTROL]
    else:
        bars2 = p114.load_bars(Path(a.raw), p114.DATA_END_MAIN)
        fin_bars = load_index(a, CONTROL)
    G2 = rep.build_grids(bars2, p114.TRIM)
    ds = design(a, G2["days"])
    wins = {"before_inferred": ds["before"], "after": ds["after"]}
    wdays = {w: G2["days"][(G2["days"] >= pd.Timestamp(x0)) & (G2["days"] <= pd.Timestamp(x1))] for w, (x0, x1) in wins.items()}
    fits, stores = [], {}
    for w, d in wdays.items():
        for k in TREATED:
            fw = p114.fit_window(G2, k, rep.period_rows(G2, d), models=MODELS)
            stores[(w, k)] = store_of(fw)
            for m in MODELS:
                fits.append({"data": label, "window": w, "index": k, "grid": "two-index (replication)",
                             "first_day": str(d.min().date()), "last_day": str(d.max().date()), "model": m,
                             **{c: fw[m]["row"][c] for c in WINDOW_COLS}, "daily_method": fw["_ref_fit"]["daily_method"]})
    # the control: Fin Nifty's own grid, the same dates, daily EGARCH joined over its data hole
    rep._EGARCH_CACHE.clear()
    p114._HAR_CACHE.clear()
    Gc = p114.index_grid(fin_bars, CONTROL)
    step = np.r_[0, np.diff(Gc["days"].values).astype("timedelta64[D]").astype(int)]
    holes = [(str(Gc["days"][i - 1].date()), str(Gc["days"][i].date())) for i in np.flatnonzero(step > HOLE_DAYS)]
    for w, d in wdays.items():
        rows = np.flatnonzero(Gc["days"].isin(d))
        if len(rows) != len(d) or not np.array_equal(Gc["days"][rows], d) or np.any(np.diff(rows) != 1):
            raise SystemExit(f"Fin Nifty lacks some days of the {w} window, or has extra days inside it")
        egarch_level_joined(Gc, CONTROL, int(rows[-1]))
        fw = p114.fit_window(Gc, CONTROL, rows, models=MODELS)
        stores[(w, CONTROL)] = store_of(fw)
        for m in MODELS:
            fits.append({"data": label, "window": w, "index": CONTROL, "grid": "Fin Nifty's own (same dates)",
                         "first_day": str(d.min().date()), "last_day": str(d.max().date()), "model": m,
                         **{c: fw[m]["row"][c] for c in WINDOW_COLS}, "daily_method": fw["_ref_fit"]["daily_method"],
                         "history_days": int(rows[0])})
    fits = pd.DataFrame(fits)
    fits.to_csv(work / "control_fits.csv", index=False, float_format="%.17g")
    print(f"[{label}] P117 control: Fin Nifty grid {len(Gc['days'])} days ({Gc['days'][0].date()} to "
          f"{Gc['days'][-1].date()}), data holes {holes}; fits done ({time.time() - t0:.0f}s)", flush=True)
    # check: P114's committed point fits and its first bootstrap draws are reproduced by this code
    if not a.synthetic:
        pf = pd.read_csv(ROOT / "results" / "p114_persistence_fixed_model" / "fits.csv", float_precision="round_trip")
        pb = pd.read_csv(ROOT / "results" / "p114_persistence_fixed_model" / "bootstrap_draws.csv", float_precision="round_trip")
        chk = []
        for _, r in fits[fits["index"].isin(TREATED)].iterrows():
            c = pf[(pf["window"] == r["window"]) & (pf["index"] == r["index"]) & (pf["model"] == r["model"])].iloc[0]
            for q in ("alpha", "beta", "gam_x", "delta", "nu", "hl_irf"):
                chk.append({"check": "P114 point fit", "window": r["window"], "index": r["index"], "model": r["model"],
                            "draw": -1, "quantity": q, "this_study": r[q], "p114_committed": c[q],
                            "identical": bool(r[q] == c[q])})
        for (w, k) in DID_KEYS:
            d = boot_draws(stores[(w, k)], p114.SEED_INDEX[(w, k)], N_BOOT, chunks=[0], per_limit=a.check_draws)
            for _, r in d.iterrows():
                c = pb[(pb["window"] == w) & (pb["index"] == k) & (pb["model"] == r["model"]) & (pb["draw"] == r["draw"])].iloc[0]
                for q in ("alpha", "beta", "gam_x", "delta", "hl_irf"):
                    chk.append({"check": "P114 bootstrap draw (committed file has 12 significant digits)", "window": w,
                                "index": k, "model": r["model"], "draw": int(r["draw"]), "quantity": q, "this_study": r[q],
                                "p114_committed": c[q], "identical": bool(float(f"{r[q]:.12g}") == c[q])})
        chk = pd.DataFrame(chk)
        chk.to_csv(work / "reproduction_p114_control.csv", index=False, float_format="%.17g")
        print(f"  P114 reproduced: {int(chk['identical'].sum())} of {len(chk)} values identical", flush=True)
    # the control's bootstrap, with each treated index's seeds (the same resampled days as P114's draws)
    for (w, k) in DID_KEYS:
        path = work / f"boot_{CONTROL}_{w}_seeds_{k}.csv"
        if path.exists():
            continue
        tb = time.time()
        seed_index = p114.SEED_INDEX[(w, k)] if not a.synthetic else (2 if k == "nifty" else 3) + (2 if w == "after" else 0)
        d = boot_draws(stores[(w, CONTROL)], seed_index, n_boot)
        d.insert(0, "seed_index", seed_index)
        d.insert(0, "seeds_of", k)
        d.insert(0, "index", CONTROL)
        d.insert(0, "window", w)
        d.to_csv(path.with_suffix(".tmp"), index=False, float_format="%.17g")
        os.replace(path.with_suffix(".tmp"), path)
        print(f"  bootstrap {CONTROL} {w} with {k}'s seeds: {len(d)} rows ({time.time() - tb:.0f}s)", flush=True)
    if a.synthetic:   # SYNTHETIC: the treated draws are made here (on real data they are P114's committed draws)
        for (w, k) in DID_KEYS:
            path = work / f"boot_{k}_{w}.csv"
            if path.exists():
                continue
            seed_index = (2 if k == "nifty" else 3) + (2 if w == "after" else 0)
            d = boot_draws(stores[(w, k)], seed_index, n_boot)
            d.insert(0, "index", k)
            d.insert(0, "window", w)
            d.to_csv(path, index=False, float_format="%.17g")
    json.dump({"seconds": round(time.time() - t0, 1), "finnifty_grid_days": len(Gc["days"]), "holes": holes},
              open(work / "control_info.json", "w"), indent=2)
    print(f"control stage done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Stage analyze
# ---------------------------------------------------------------------------


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pair_table(pairs: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """One row per pair with each model's half-lives and the change in log half-life (after minus before)."""
    d = pairs.copy()
    for m in MODELS:
        w = win[win["model"] == m]
        hb = w[w["kind"] == "before"].set_index("start_row")["hl_irf"]
        ha = w[w["kind"] == "after"].set_index("start_row")["hl_irf"]
        d[f"hl_before_{m}"] = d["start_row"].map(hb)
        d[f"hl_after_{m}"] = d["after_row"].map(ha)
        d[f"delta_{m}"] = log_hl(d[f"hl_after_{m}"]) - log_hl(d[f"hl_before_{m}"])
    return d


def stage_analyze(a):
    work = Path(a.work)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    label = LABEL_SYN if a.synthetic else LABEL_REAL
    tests, summ, allpairs, verdicts = [], [], [], []

    def add(test_id, hyp, role, family, sample, est, stat, p, side, method):
        tests.append({"test_id": test_id, "hypothesis": hyp, "role": role, "family": family, "sample": sample,
                      "estimate": est, "statistic": stat, "p_raw": p, "sidedness": side, "p_method": method})

    # ---- placebo tests
    info = {}
    for k in TREATED:
        pf, wf = work / f"pairs_{k}.csv", work / f"windows_{k}.csv"
        if not (pf.exists() and wf.exists()):
            print(f"  {k}: no windows; skipped", flush=True)
            continue
        pairs = pd.read_csv(pf, parse_dates=["before_first", "before_last", "after_first", "after_last"])
        win = pd.read_csv(wf, float_precision="round_trip")
        info[k] = json.load(open(work / f"windows_{k}_info.json"))
        gap, L_b, L_a = info[k]["gap"], info[k]["L_before"], info[k]["L_after"]
        P = pair_table(pairs, win)
        sets = [c[3:] for c in P.columns if c.startswith("in_")]
        keep = P["is_actual"] | P[[f"in_{s}" for s in sets]].any(axis=1)
        allpairs.append(P[keep])
        A = P[P["is_actual"]].iloc[0]
        for m in MODELS:
            d_act = float(A[f"delta_{m}"])
            for s in sets:
                S = P[P[f"in_{s}"]]
                pp = placebo_p(S[f"delta_{m}"], d_act)
                dj = disjoint_p(S["start_row"], S[f"delta_{m}"], d_act, L_b, gap, L_a)
                sp = separate_spans(S["start_row"][S[f"delta_{m}"].notna()], L_b + gap + L_a)
                cp = conditional_p(S[f"delta_{m}"], log_hl(S[f"hl_before_{m}"]), d_act, float(log_hl(A[f"hl_before_{m}"])))
                x = S[f"delta_{m}"].dropna().to_numpy()
                hb = log_hl(S[f"hl_before_{m}"].dropna())
                ha = log_hl(S[f"hl_after_{m}"].dropna())
                e = {"data": label, "index": k, "model": m, "set": s, "set_label": SET_LABEL.get(s, s),
                     "hl_before_actual": A[f"hl_before_{m}"], "hl_after_actual": A[f"hl_after_{m}"],
                     "delta_actual": d_act, "fall_factor_actual": float(np.exp(-d_act)),
                     "n_pairs": pp["n"], "n_pairs_dropped": pp["n_dropped"], "k_at_or_below": pp["k"],
                     "share_at_or_below": pp["share"], "p_placebo": pp["p"],
                     "first_before_start": str(S["before_first"].min().date()) if len(S) else "",
                     "last_after_end": str(S["after_last"].max().date()) if len(S) else "",
                     "disjoint_p": dj["p"], "disjoint_n_mean": dj["n_mean"], "disjoint_n_min": dj["n_min"],
                     "disjoint_n_max": dj["n_max"], "disjoint_k_mean": dj["k_mean"],
                     "disjoint_p_min_attainable": dj["p_min_attainable"],
                     "separate_spans_min": sp["min"], "separate_spans_max": sp["max"],
                     "meets_minimum": bool(pp["n"] >= MIN_PAIRS and dj["n_mean"] >= MIN_DISJOINT),
                     "cond_intercept": cp["intercept"], "cond_slope": cp["slope"], "cond_resid_actual": cp["resid_act"],
                     "cond_resid_sd": cp["resid_sd"], "cond_k": cp["k"], "cond_p": cp["p"],
                     "rank_before_actual_pct": float(np.mean(hb <= float(log_hl(A[f'hl_before_{m}']))) * 100) if len(hb) else np.nan,
                     "rank_after_actual_pct": float(np.mean(ha <= float(log_hl(A[f'hl_after_{m}']))) * 100) if len(ha) else np.nan,
                     "share_pairs_with_fall": float(np.mean(x < 0)) if len(x) else np.nan,
                     **{f"q{q:g}": (float(np.percentile(x, q)) if len(x) else np.nan) for q in QUANTILES}}
                summ.append(e)
    summ = pd.DataFrame(summ)
    smp = {"nifty": "Nifty 50", "banknifty": "Bank Nifty"}

    def srow(k, m, s):
        r = summ[(summ["index"] == k) & (summ["model"] == m) & (summ["set"] == s)]
        return r.iloc[0] if len(r) else None

    hyp = ("The actual change in the log half-life of a 12:15 shock (after minus before) under the {ml} is lower "
           "(a larger fall) than the changes between placebo pairs of windows with the same lengths and gap ({sl}). "
           "Estimate: actual change in log half-life.")
    meth = ("placebo in time: every pair of 64- and 59-day windows 258 trading days apart, one-day steps, {sl}; "
            "p = (1 + placebo pairs with a change at or below the actual) / (1 + placebo pairs); statistic = share of "
            "placebo pairs at or below the actual")
    fam = {"P": ("primary", [("nifty", "primary")], "primary family P (Nifty, primary set; reference and F)"),
           "S1": ("secondary", [("banknifty", "primary")], "secondary S1 (Bank Nifty, primary set)"),
           "S2": ("secondary", [("nifty", "clean"), ("banknifty", "clean")], "secondary S2 (clean set)"),
           "S3": ("secondary", [("nifty", "literal"), ("banknifty", "literal")], "secondary S3 (literal set: only event b excluded)")}
    for f, (role, keys, fl) in fam.items():
        for k, s in keys:
            for m in MODELS:
                r = srow(k, m, s)
                if r is None:
                    continue
                add(f"{f}_{k}_{s}_{m}", hyp.format(ml=MODEL_LABEL[m], sl=SET_LABEL[s]), role, fl,
                    f"{smp[k]}, actual pair 14 Nov 2022 to 10 Feb 2023 against 1 Mar to 31 May 2024; {r['n_pairs']} placebo pairs "
                    f"({r['first_before_start']} to {r['last_after_end']})", r["delta_actual"], r["share_at_or_below"],
                    r["p_placebo"], "one-sided (actual fall larger)", meth.format(sl=SET_LABEL[s]))
    for k in TREATED:
        for m in MODELS:
            r = srow(k, m, "primary")
            if r is None:
                continue
            add(f"S4_{k}_primary_{m}_disjoint", hyp.format(ml=MODEL_LABEL[m], sl=SET_LABEL["primary"]) +
                " Counting only day-disjoint placebo pairs.", "secondary", "secondary S4 (day-disjoint count, primary set)",
                f"{smp[k]}, primary set; on average {r['disjoint_n_mean']:.2f} day-disjoint pairs per phase",
                r["delta_actual"], r["disjoint_k_mean"], r["disjoint_p"], "one-sided (actual fall larger)",
                "for each phase 0 .. 380, greedy selection of placebo pairs that share no trading day; p_phase = (1 + "
                "selected pairs at or below the actual) / (1 + selected pairs); p = mean over phases; statistic = mean "
                "number of selected pairs at or below the actual")
            add(f"S5_{k}_primary_{m}_conditional",
                f"Given the before window's log half-life, the after window's half-life under the {MODEL_LABEL[m]} is "
                "shorter than placebo pairs predict (residual of the change after an OLS fit on the before window's log "
                "half-life over the primary set). Estimate: the actual pair's residual.", "secondary",
                "secondary S5 (conditional on the before window, primary set)", f"{smp[k]}, primary set, {r['n_pairs']} pairs",
                r["cond_resid_actual"], r["cond_slope"], r["cond_p"], "one-sided (residual lower)",
                "OLS of the placebo changes on the before window's log half-life; p = (1 + placebo residuals at or below the "
                "actual residual) / (1 + placebo pairs); statistic = OLS slope")

    # ---- difference-in-differences against Fin Nifty
    did_rows = []
    cf = work / "control_fits.csv"
    if cf.exists():
        fits = pd.read_csv(cf, float_precision="round_trip")
        if a.synthetic:
            tb = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in sorted(work.glob("boot_nifty_*.csv"))
                            + sorted(work.glob("boot_banknifty_*.csv"))], ignore_index=True)
        else:
            tb = pd.read_csv(ROOT / "results" / "p114_persistence_fixed_model" / "bootstrap_draws.csv", float_precision="round_trip")
        cb = {f.name: pd.read_csv(f, float_precision="round_trip") for f in sorted(work.glob(f"boot_{CONTROL}_*.csv"))}

        def pt(w, k, m):
            return fits[(fits["window"] == w) & (fits["index"] == k) & (fits["model"] == m)].iloc[0]

        for k in TREATED:
            for m in MODELS:
                dT = float(log_hl(pt("after", k, m)["hl_irf"]) - log_hl(pt("before_inferred", k, m)["hl_irf"]))
                dC = float(log_hl(pt("after", CONTROL, m)["hl_irf"]) - log_hl(pt("before_inferred", CONTROL, m)["hl_irf"]))
                e = {"data": label, "treated": k, "control": CONTROL, "model": m,
                     "hl_before_treated": pt("before_inferred", k, m)["hl_irf"], "hl_after_treated": pt("after", k, m)["hl_irf"],
                     "hl_before_control": pt("before_inferred", CONTROL, m)["hl_irf"],
                     "hl_after_control": pt("after", CONTROL, m)["hl_irf"],
                     "delta_treated": dT, "delta_control": dC, "did": dT - dC}
                bT = tb[(tb["index"] == k) & (tb["model"] == m)]
                cbb = cb.get(f"boot_{CONTROL}_before_inferred_seeds_{k}.csv")
                cba = cb.get(f"boot_{CONTROL}_after_seeds_{k}.csv")
                if len(bT) and cbb is not None and cba is not None:
                    tbb = bT[bT["window"] == "before_inferred"].set_index("draw")["hl_irf"]
                    tba = bT[bT["window"] == "after"].set_index("draw")["hl_irf"]
                    ccb = cbb[cbb["model"] == m].set_index("draw")["hl_irf"]
                    cca = cba[cba["model"] == m].set_index("draw")["hl_irf"]
                    idx = tbb.index.intersection(tba.index).intersection(ccb.index).intersection(cca.index)
                    DT = log_hl(tba[idx]) - log_hl(tbb[idx])
                    DC = log_hl(cca[idx]) - log_hl(ccb[idx])
                    D = DT - DC
                    e.update({"boot_draws": len(idx), "boot_draws_not_computed": int(np.isnan(D).sum()),
                              "did_lo": p114.pct(D, 2.5), "did_hi": p114.pct(D, 97.5), "share_draws_below0": float(np.mean(D[~np.isnan(D)] < 0)),
                              "p_did": boot_p_below0(D), "delta_control_lo": p114.pct(DC, 2.5),
                              "delta_control_hi": p114.pct(DC, 97.5), "p_control_fell": boot_p_below0(DC),
                              "share_control_draws_fell": float(np.mean(DC[~np.isnan(DC)] < 0)),
                              "delta_treated_lo": p114.pct(DT, 2.5), "delta_treated_hi": p114.pct(DT, 97.5)})
                did_rows.append(e)
        did = pd.DataFrame(did_rows)
        for _, r in did.iterrows():
            if "p_did" not in r or not np.isfinite(r.get("p_did", np.nan)):
                continue
            add(f"D_{r['treated']}_{r['model']}_did",
                f"The change in the log half-life of a 12:15 shock under the {MODEL_LABEL[r['model']]} is lower (a larger "
                f"fall) for {smp[r['treated']]} than for Fin Nifty, whose expiry calendar did not change, over the same "
                "windows (difference-in-differences). Estimate: treated change minus control change.", "secondary",
                "secondary D (difference-in-differences against Fin Nifty)",
                f"{smp[r['treated']]} minus Fin Nifty, 14 Nov 2022 to 10 Feb 2023 against 1 Mar to 31 May 2024 (the same days)",
                r["did"], r["share_draws_below0"], r["p_did"], "one-sided (treated fell more)",
                "day-block bootstrap: P114's 400 draws of the treated index; Fin Nifty refitted on the same resampled days "
                "(same seeds); before and after paired by draw number; p = (1 + draws with DiD >= 0) / (1 + draws)")
        for _, r in did[did["treated"] == "nifty"].iterrows():
            if "p_control_fell" not in r or not np.isfinite(r.get("p_control_fell", np.nan)):
                continue
            add(f"E1_finnifty_{r['model']}_change",
                f"Fin Nifty's half-life of a 12:15 shock under the {MODEL_LABEL[r['model']]} is shorter after than before "
                "(the same windows). Estimate: change in log half-life.", "exploratory", "exploratory E1 (the control's own change)",
                "Fin Nifty, 14 Nov 2022 to 10 Feb 2023 against 1 Mar to 31 May 2024", r["delta_control"],
                r["share_control_draws_fell"], r["p_control_fell"], "one-sided (half-life fell)",
                "day-block bootstrap (resampled on Nifty's days in P114's draws); p = (1 + draws with no fall) / (1 + draws)")
        did.to_csv(out / "did.csv", index=False)

    # ---- Holm within families, verdicts
    T = pd.DataFrame(tests)
    T["p_adjusted"] = np.nan
    T["adjust_method"] = ""
    for f, g in T.groupby("family"):
        if g["role"].iloc[0] == "exploratory":
            T.loc[g.index, "p_adjusted"] = g["p_raw"]
            T.loc[g.index, "adjust_method"] = "none (exploratory)"
            continue
        T.loc[g.index, "p_adjusted"] = holm(g["p_raw"].to_numpy())
        T.loc[g.index, "adjust_method"] = f"Holm over {len(g)} rows: {f}"
    for _, r in T.iterrows():
        tid = r["test_id"]
        if tid.startswith(("P_", "S1_", "S2_", "S3_")):
            k, s, m = tid.split("_")[1], tid.split("_")[2], tid.split("_")[3]
            mm = bool(srow(k, m, s)["meets_minimum"])
            verdicts.append({"test_id": tid, "index": k, "set": s, "model": m, "estimate": r["estimate"],
                             "p_raw": r["p_raw"], "p_adjusted": r["p_adjusted"], "meets_minimum": mm,
                             "label": verdict(r["p_adjusted"], mm)})
        elif tid.startswith(("S4_", "S5_")):
            k, m = tid.split("_")[1], tid.split("_")[3]
            mm = bool(srow(k, m, "primary")["meets_minimum"])
            verdicts.append({"test_id": tid, "index": k, "set": "primary", "model": m,
                             "estimate": r["estimate"], "p_raw": r["p_raw"], "p_adjusted": r["p_adjusted"],
                             "meets_minimum": mm, "label": verdict(r["p_adjusted"], mm)})
        elif tid.startswith("D_"):
            verdicts.append({"test_id": tid, "index": tid.split("_")[1], "set": "actual windows", "model": tid.split("_")[2],
                             "estimate": r["estimate"], "p_raw": r["p_raw"], "p_adjusted": r["p_adjusted"],
                             "meets_minimum": True, "label": did_verdict(r["p_adjusted"])})
    cols = ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"]
    T[cols].to_csv(out / "tests.csv", index=False)
    pd.DataFrame(verdicts).to_csv(out / "verdict.csv", index=False)
    summ.to_csv(out / "placebo_summary.csv", index=False)
    if allpairs:
        AP = pd.concat(allpairs, ignore_index=True)
        AP.insert(0, "data", label)
        AP.to_csv(out / "placebo_pairs.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")
    for k in TREATED:
        f = work / f"windows_{k}.csv"
        if f.exists():
            w = pd.read_csv(f, float_precision="round_trip")
            w.to_csv(out / f"windows_{k}.csv", index=False, float_format="%.10g")
    for f in ("control_fits.csv", "reproduction_p114_rolling.csv", "reproduction_p114_control.csv"):
        if (work / f).exists():
            pd.read_csv(work / f, float_precision="round_trip").to_csv(out / f, index=False, float_format="%.17g")
    cb = sorted(work.glob(f"boot_{CONTROL}_*.csv"))
    if cb:
        pd.concat([pd.read_csv(f, float_precision="round_trip") for f in cb], ignore_index=True).to_csv(
            out / "control_bootstrap_draws.csv", index=False, float_format="%.17g")
    make_figures(out, label, summ, allpairs, gap=next(iter(info.values()))["gap"] if info else None)
    ri = {"data": label, "plan_commit": a.plan_commit, "analysis_commit": git_head(), "analysis_seconds": round(time.time() - t0, 1),
          "windows": info, "control": json.load(open(work / "control_info.json")) if (work / "control_info.json").exists() else None,
          "inputs": json.load(open(work / "inputs.json")) if (work / "inputs.json").exists() else None,
          "settings": {"L_before": L_BEFORE, "L_after": L_AFTER, "history_min": HISTORY_MIN, "changes": CHANGES,
                       "sets": {k: list(v) for k, v in SETS.items()}, "min_pairs": MIN_PAIRS, "min_disjoint": MIN_DISJOINT,
                       "thresholds": [P_UNUSUAL, P_NORMAL], "hl_cap": HL_CAP, "n_boot": N_BOOT},
          "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
    json.dump(ri, open(out / "run_info.json", "w"), indent=2, default=str)
    print(T[["test_id", "estimate", "p_raw", "p_adjusted"]].to_string(), flush=True)
    print(pd.DataFrame(verdicts)[["test_id", "label"]].to_string(), flush=True)
    print(f"analyze done in {time.time() - t0:.1f}s", flush=True)


def make_figures(out: Path, label: str, summ: pd.DataFrame, allpairs: list, gap=None):
    if not allpairs or summ.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    AP = pd.concat(allpairs, ignore_index=True)
    ks = [k for k in TREATED if (AP["index"] == k).any()]
    fig, axes = plt.subplots(len(ks), 2, figsize=(11, 3.6 * len(ks)), squeeze=False)
    for i, k in enumerate(ks):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            P = AP[AP["index"] == k]
            x = P.loc[P["in_primary"], f"delta_{m}"].dropna()
            xc = P.loc[P["in_clean"], f"delta_{m}"].dropna()
            d_act = float(P.loc[P["is_actual"], f"delta_{m}"].iloc[0])
            bins = np.linspace(min(x.min(), xc.min() if len(xc) else x.min(), d_act), max(x.max(), 0.5, d_act), 50)
            ax.hist(x, bins=bins, color="#2a78d6", alpha=0.55, label=f"primary set ({len(x)} pairs)")
            if len(xc):
                ax.hist(xc, bins=bins, histtype="step", color="#1b1b1b", lw=1.2, label=f"clean set ({len(xc)} pairs)")
            r = summ[(summ["index"] == k) & (summ["model"] == m) & (summ["set"] == "primary")].iloc[0]
            ax.axvline(d_act, color="#d6452a", lw=2, label=f"actual pair: {d_act:+.2f} (placebo p {r['p_placebo']:.3f})")
            ax.axvline(r["q5"], color="#7a7a7a", lw=1, ls="--", label="5% quantile of the primary set")
            ax.axvline(0, color="#bdbdbd", lw=0.8)
            ax.set_title(f"{INDEX_NAME[k]}, {'reference model' if m == 'ref' else 'fixed model F'}")
            ax.set_xlabel("change in log half-life of a 12:15 shock, after minus before")
            ax.set_ylabel("placebo pairs")
            ax.legend(fontsize=7.5)
    fig.suptitle(f"Placebo changes between windows {gap} trading days apart, and the actual change ({label.split(' (')[0]})")
    fig.tight_layout()
    fig.savefig(out / "fig_placebo.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(len(ks), 1, figsize=(11, 3.3 * len(ks)), squeeze=False)
    for i, k in enumerate(ks):
        ax = axes[i, 0]
        P = AP[AP["index"] == k]
        for m, col in (("ref", "#2a78d6"), ("F", "#eb6834")):
            S = P[P["in_literal"]]
            ax.scatter(S["before_first"], S[f"delta_{m}"], s=3, color=col, alpha=0.6,
                       label=f"{'reference' if m == 'ref' else 'fixed model F'}: placebo pairs")
            A = P[P["is_actual"]]
            ax.scatter(A["before_first"], A[f"delta_{m}"], s=60, color=col, edgecolor="k", marker="*",
                       label=f"{'reference' if m == 'ref' else 'fixed model F'}: actual pair")
        ax.axhline(0, color="#bdbdbd", lw=0.8)
        ax.set_title(f"{INDEX_NAME[k]}: change by first day of the before window (pairs straddling event a are drawn too)")
        ax.set_ylabel("change in log half-life")
        ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_placebo_time.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("inputs", "windows", "control", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--work", default=None)
        p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
        p.add_argument("--synthetic", action="store_true", help="simulated three-index market (dry run)")
        if name == "windows":
            p.add_argument("--index", choices=TREATED, required=True)
        if name == "control":
            p.add_argument("--n-boot", type=int, default=N_BOOT)
            p.add_argument("--check-draws", type=int, default=3, help="P114 draws per key recomputed as a check")
        if name == "analyze":
            p.add_argument("--out", default=None)
            p.add_argument("--plan-commit", default=None)
    a = ap.parse_args(argv)
    if a.work is None:
        a.work = str(ROOT / "data" / "processed" / ("p117_synthetic" if a.synthetic else "p117"))
    if getattr(a, "out", "x") is None:
        a.out = str(ROOT / "results" / ("p117_persistence_placebo_synthetic" if a.synthetic else "p117_persistence_placebo"))
    if a.synthetic and a.stage == "control":
        a.check_draws = SYNTH["check_draws"]
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    {"inputs": stage_inputs, "windows": stage_windows, "control": stage_control, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
