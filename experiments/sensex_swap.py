"""P92: does the expiring index become the net transmitter of volatility spillovers on its own expiry day? REAL DATA.

The binding design is the pre-registration docs/prereg/P92_SENSEX_SWAP.md, committed to git before any outcome of
this study was computed. This docstring is a summary of it.

Natural experiment. On 1 September 2025 NSE and BSE swapped expiry weekdays. From 1 January 2025 to 31 August 2025
the BSE Sensex weekly expired on Tuesday and the NSE Nifty weekly on Thursday; from 1 September 2025 Nifty expires on
Tuesday and Sensex on Thursday. Holding the weekday fixed, only the identity of the expiring index changes.

Data (REAL DATA, nothing simulated here; synthetic checks are in tests/test_sensex_swap.py):
  * data/raw/nifty_1min.csv (technovusin/nifty50-historical-data, pinned commit in minute_bars_source.json) and
    data/raw/sensex_1min.csv (technovusin/sensex-historical-data, pinned commit in sensex_1min_source.json);
  * Nifty expiry days from NSE bhavcopies: results/pipeline_nse_v2/daily_market.csv (is_expiry, is_monthly_expiry,
    regime);
  * Sensex expiry days from the reported weekday regimes in data/external/sensex_expiry_calendar.csv with the
    reported holiday roll-back rule (experiments/sensex_confound.py functions, imported, not copied).

Minute volatility: exactly as in experiments/connectedness_gamma.py (its functions are imported): 360-bar session
09:15 to 15:14, days on which both indices have at least 300 of the 360 bars, mcsGARCH conditional volatility
sqrt(h * s * q) fitted per NSE expiry regime (single_weekly_thu from 21 Nov 2024, single_weekly_tue from 1 Sep 2025),
ARMA(1,1) mean, Student t, daily variance from an AR(1)-EGARCH(1,1)-t on the daily returns of the common days.

Outcome per day d: a VAR(2) on that day's minutes only (lags never cross days), generalized FEVD at horizon 10
(Diebold and Yilmaz 2012). NET_d = (Nifty to Sensex share) minus (Sensex to Nifty share), in percentage points
(positive: Nifty is the net transmitter). TOT_d = the sum of the two cross shares (paper scale, 0 to 200).

Primary test H1 (one-sided): delta = [NET(Tue, after) - NET(Tue, before)] - [NET(Thu, after) - NET(Thu, before)]
> 0, cell means of NET_d, before = 1 Jan 2025 to 31 Aug 2025, after = 1 Sep 2025 to 2 Nov 2026 (the last day with
both bars in the pinned files). Confirmatory p-value: block permutation of the period label over calendar weeks
(blocks of 4 consecutive weeks, the Tuesday and Thursday of a week always move together, weekdays never move),
9,999 draws, p = (1 + #{delta* >= delta}) / (1 + draws). Reported with it: the same delta as the interaction
coefficient of a day-level OLS with weekday and period effects and Newey-West (5 lags) standard errors.
Secondary family (Holm): S1 = H1 on the last hour (14:15 to 15:14); S2 = H2, total connectedness on either index's
expiry day against other days (day-level regression with weekday and Sensex-period effects, HAC, two-sided);
S3 = H3, the same NET test around the Sensex move from Friday to Tuesday (week of 6 Jan 2025), Tuesday against
Friday, one-sided (delta < 0). Placebos (uncorrected, two-sided): PL1 Monday against Wednesday around 1 Sep 2025
(non-expiry days only); PL2 and PL3 Tuesday against Thursday at fake swap dates inside the before window
(5 May 2025) and inside the pre-auction after window (16 Feb 2026). Robustness rows are reported, never decisive.

Usage:

    OMP_NUM_THREADS=1 python experiments/sensex_swap.py --coverage   # phase 1: calendar and data coverage only, no outcome
    OMP_NUM_THREADS=1 python experiments/sensex_swap.py              # phase 2: the full pre-registered analysis

Writes results/sensex_swap/ (coverage.csv, calendar_days.csv, excluded_days.csv, calendar_check.csv, mde.csv,
mde_proxy.csv, run_info_coverage.json, and in phase 2 day_values.csv, cell_means.csv, tests.csv, robustness.csv, h2_regression.csv,
mcsgarch_fits.csv, verdict.json, run_info.json) and results/sensex_swap_stdout.txt (phase 1:
results/sensex_swap_coverage_stdout.txt).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    """Import an experiment script as a module (registered, so dataclasses work); reuse it if already loaded."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cg = _load("connectedness_gamma", ROOT / "experiments" / "connectedness_gamma.py")
sc = _load("sensex_confound", ROOT / "experiments" / "sensex_confound.py")

from niftygex import replication as rp  # noqa: E402

LABEL = "REAL DATA"
RAW = ROOT / "data" / "raw"
RUN = ROOT / "results" / "pipeline_nse_v2"
CAL_PATH = ROOT / "data" / "external" / "sensex_expiry_calendar.csv"
SX_USED = ROOT / "results" / "sensex_confound" / "sensex_expiry_days_used.csv"
CG_DAYS = ROOT / "results" / "connectedness_gamma" / "sample_days.csv"
PREREG = ROOT / "docs" / "prereg" / "P92_SENSEX_SWAP.md"
OUT = ROOT / "results" / "sensex_swap"
LOG = ROOT / "results" / "sensex_swap_stdout.txt"
LOG_COVERAGE = ROOT / "results" / "sensex_swap_coverage_stdout.txt"

INDICES = ("nifty", "sensex")  # order matters: table[1, 0] = Nifty to Sensex, table[0, 1] = Sensex to Nifty
TRIM = cg.TRIM  # "360_drop_last15": bars 09:15 to 15:14
P_MAIN, H_MAIN = cg.P_MAIN, cg.H_MAIN  # VAR(2), horizon 10
PMAX_AIC, H_SENS = cg.PMAX_AIC, cg.H_SENS  # sensitivity: lag by AIC (max 10), horizon 30
HAC_LAGS = cg.HAC_LAGS  # 5
MIN_BARS = 300  # of 360, both indices (the connectedness study's rule)
LAST_HOUR = ("14:15", "15:14")  # the last 60 bars of the 360-bar session
MIN_LAST_HOUR = 50  # of 60 minutes with both series valid
SEED, STUDY = 20261102, 92
N_PERM = 9999
BLOCK_WEEKS = 4
ALPHA, POWER = 0.05, 0.80
PREREG_MDE = 4.90  # points of delta; fixed in docs/prereg/P92_SENSEX_SWAP.md section 6.4 (used by the decision rule)

SAMPLE_START = pd.Timestamp("2024-11-21")  # first day of NSE's single_weekly_thu regime in daily_market.csv
END = pd.Timestamp("2026-11-02")  # last day with both Nifty and Sensex bars in the pinned files
SWAP = pd.Timestamp("2025-09-01")  # Nifty to Tuesday, Sensex to Thursday
SENSEX_TUE_WEEK = pd.Timestamp("2025-01-06")  # week of the first Tuesday Sensex expiry (7 Jan 2025)
AUCTION_START = pd.Timestamp("2026-08-03")  # closing auction on NSE and BSE (minutes 15:16 to 15:28 frozen)
WDN = ["Mon", "Tue", "Wed", "Thu", "Fri"]
PROXY_REGIMES = ("single_weekly_thu", "single_weekly_tue")


# ---------------------------------------------------------------------------
# Test specifications (fixed in the pre-registration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DidSpec:
    """A weekday difference-in-differences: [y(a, after) - y(a, before)] - [y(b, after) - y(b, before)].

    ``status`` gives the expected expiry status of each cell, as ((weekday, after, status), ...), status one of
    "N" (only Nifty expires), "S" (only Sensex expires), "none" (neither). A day on the cell weekday enters only if
    its actual status matches. ``after`` is 1 for days in calendar weeks starting on or after ``split`` (a Monday).
    """

    test_id: str
    family: str
    what: str
    wd_a: int
    wd_b: int
    lo: str
    split: str
    hi: str
    status: tuple
    alternative: str  # "greater", "less" or "two-sided"
    outcome: str = "net_day"
    seed_key: int = 0

    def expected(self, wd: int, after: int) -> str:
        for w, a, s in self.status:
            if w == wd and a == after:
                return s
        raise KeyError((wd, after))


def _swap_status(a_before, a_after, b_before, b_after, wd_a, wd_b):
    return ((wd_a, 0, a_before), (wd_a, 1, a_after), (wd_b, 0, b_before), (wd_b, 1, b_after))


END_S = END.strftime("%Y-%m-%d")
P1 = DidSpec("P1", "primary", "H1: Tue minus Thu NET, after minus before the 1 Sep 2025 swap (whole day)",
             1, 3, "2025-01-01", "2025-09-01", END_S, _swap_status("S", "N", "N", "S", 1, 3), "greater", "net_day", 1)
S1 = DidSpec("S1", "secondary", "H1 on the last hour (14:15 to 15:14)",
             1, 3, "2025-01-01", "2025-09-01", END_S, _swap_status("S", "N", "N", "S", 1, 3), "greater", "net_lh", 2)
S3 = DidSpec("S3", "secondary", "H3: Tue minus Fri NET, after minus before the Sensex move to Tuesday (week of 6 Jan 2025)",
             1, 4, "2024-11-21", "2025-01-06", "2025-08-31", _swap_status("none", "S", "S", "none", 1, 4), "less", "net_day", 3)
PL1 = DidSpec("PL1", "placebo", "Mon minus Wed NET around 1 Sep 2025 (days on which no index expires)",
              0, 2, "2025-01-01", "2025-09-01", END_S, _swap_status("none", "none", "none", "none", 0, 2), "two-sided", "net_day", 4)
PL2 = DidSpec("PL2", "placebo", "Tue minus Thu NET at a fake swap on 5 May 2025 (inside the before window)",
              1, 3, "2025-01-01", "2025-05-05", "2025-08-31", _swap_status("S", "S", "N", "N", 1, 3), "two-sided", "net_day", 5)
PL3 = DidSpec("PL3", "placebo", "Tue minus Thu NET at a fake swap on 16 Feb 2026 (after window, before the auction)",
              1, 3, "2025-09-01", "2026-02-16", "2026-07-31", _swap_status("N", "N", "S", "S", 1, 3), "two-sided", "net_day", 6)
DID_TESTS = (P1, S1, S3, PL1, PL2, PL3)
P1_NO_AUCTION = DidSpec("R_auction", "robustness", "H1 without closing-auction days (after window ends 31 Jul 2026)",
                        1, 3, "2025-01-01", "2025-09-01", "2026-07-31", P1.status, "greater", "net_day", 15)


# ---------------------------------------------------------------------------
# Calendars and cells
# ---------------------------------------------------------------------------


def week_start(days) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(days)
    return days - pd.to_timedelta(days.weekday, unit="D")


def sensex_expiries(trading_days, cal_path: Path | None = None) -> pd.DataFrame:
    """Sensex weekly expiry days from the reported weekday regimes and holiday rule (sensex_confound functions)."""
    cal = pd.read_csv(cal_path or CAL_PATH, dtype=str, keep_default_na=False)
    return sc.sensex_rule_expiries(trading_days, sc.load_regimes(cal))


def last_per_month(dates) -> set:
    """The last date of each calendar month (used as the Sensex monthly expiry: an approximation, see the prereg)."""
    s = pd.Series(pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates)))))
    if s.empty:
        return set()
    return set(s.groupby([s.dt.year, s.dt.month]).max())


def build_calendar(days, nifty: pd.DataFrame, sx: pd.DataFrame) -> pd.DataFrame:
    """One row per sample day: weekday, week, expiry flags, status, regimes.

    ``nifty`` is indexed by date with boolean columns is_expiry, is_monthly_expiry and a regime column
    (daily_market.csv); ``sx`` is the Sensex rule table (expiry, shifted).
    """
    days = pd.DatetimeIndex(days)
    df = pd.DataFrame(index=days)
    df.index.name = "date"
    df["weekday"] = days.weekday
    df["week"] = week_start(days)
    nx = nifty.reindex(days)
    df["nifty_expiry"] = nx["is_expiry"].fillna(False).astype(bool).to_numpy()
    df["nifty_monthly"] = nx["is_monthly_expiry"].fillna(False).astype(bool).to_numpy()
    sched = np.where(days >= SWAP, 1, 3)
    df["nifty_shifted"] = df["nifty_expiry"] & (df["weekday"].to_numpy() != sched)
    sx_days = set(pd.DatetimeIndex(sx["expiry"]))
    df["sensex_expiry"] = days.isin(list(sx_days))
    df["sensex_shifted"] = days.isin(list(pd.DatetimeIndex(sx.loc[sx["shifted"].astype(bool), "expiry"])))
    df["sensex_monthly_rule"] = days.isin(list(last_per_month(sx_days)))
    df["status"] = [status_of(n, s) for n, s in zip(df["nifty_expiry"], df["sensex_expiry"])]
    df["either_expiry"] = df["nifty_expiry"] | df["sensex_expiry"]
    df["nse_regime"] = nx["regime"].astype(str).to_numpy()
    df["sensex_period"] = sensex_period(df["week"])
    df["auction_day"] = days >= AUCTION_START
    return df


def status_of(nifty_exp: bool, sensex_exp: bool) -> str:
    return {(True, False): "N", (False, True): "S", (False, False): "none", (True, True): "both"}[(bool(nifty_exp), bool(sensex_exp))]


def sensex_period(weeks) -> np.ndarray:
    w = pd.DatetimeIndex(weeks)
    return np.where(w < SENSEX_TUE_WEEK, "sensex_fri", np.where(w < SWAP, "sensex_tue", "sensex_thu"))


def select_cells(cal: pd.DataFrame, spec: DidSpec, drop_mask: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Days entering ``spec`` (columns is_a, after, week, cell) and the days on a cell weekday left out (with reason)."""
    split = pd.Timestamp(spec.split)
    if split.weekday() != 0:
        raise ValueError("split must be a Monday")
    lo, hi = pd.Timestamp(spec.lo), pd.Timestamp(spec.hi)
    c = cal[(cal.index >= lo) & (cal.index <= hi) & cal["weekday"].isin([spec.wd_a, spec.wd_b])].copy()
    c["after"] = (c["week"] >= split).astype(int)
    c["is_a"] = (c["weekday"] == spec.wd_a).astype(int)
    c["expected"] = [spec.expected(w, a) for w, a in zip(c["weekday"], c["after"])]
    c["cell"] = [f"{WDN[w]} {'after' if a else 'before'}" for w, a in zip(c["weekday"], c["after"])]
    ok = c["status"] == c["expected"]
    reason = np.where(~ok, "status " + c["status"].astype(str) + ", expected " + c["expected"].astype(str), "")
    if drop_mask is not None:
        dm_ = drop_mask.reindex(c.index).fillna(False).astype(bool).to_numpy()
        reason = np.where(ok & dm_, "dropped by this check", reason)
        ok = ok & ~dm_
    excl = c.loc[~ok, ["weekday", "cell", "status", "expected"]].assign(reason=reason[~ok.to_numpy()], test=spec.test_id)
    return c[ok], excl


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def did_many(y: np.ndarray, is_a: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Difference in differences of cell means for each row of ``after`` (draws x days of 0/1); NaN if a cell is empty."""
    y = np.asarray(y, float)
    A = np.atleast_2d(np.asarray(after, float))
    a = np.asarray(is_a, float)[None, :]
    b = 1.0 - a
    Bf = 1.0 - A

    def mean(mask):
        s = mask.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(s > 0, (mask * y[None, :]).sum(axis=1) / np.where(s > 0, s, 1.0), np.nan)

    return (mean(A * a) - mean(Bf * a)) - (mean(A * b) - mean(Bf * b))


def week_codes(week) -> tuple[np.ndarray, pd.DatetimeIndex]:
    w = pd.DatetimeIndex(week)
    uniq = pd.DatetimeIndex(sorted(set(w)))
    return np.searchsorted(uniq.values, w.values), uniq


def week_label_draws(week_after: np.ndarray, block: int, n: int, rng) -> np.ndarray:
    """(n x weeks) period labels: blocks of ``block`` consecutive weeks reordered at random (cg.block_permutation)."""
    W = len(week_after)
    g = np.zeros(W, int)
    out = np.empty((n, W), dtype=np.int8)
    for i in range(n):
        out[i] = week_after[cg.block_permutation(g, block, rng)]
    return out


def perm_p(obs: float, draws: np.ndarray, alternative: str) -> float:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if not np.isfinite(obs) or len(d) == 0:
        return np.nan
    if alternative == "greater":
        k = np.sum(d >= obs - 1e-12)
    elif alternative == "less":
        k = np.sum(d <= obs + 1e-12)
    else:
        k = np.sum(np.abs(d) >= abs(obs) - 1e-12)
    return float((1 + k) / (1 + len(d)))


def did_perm_test(y, is_a, after, week, alternative: str, n_perm: int, block: int, rng, how: str = "weeks") -> dict:
    """DiD of cell means with its block-permutation p-value.

    how = "weeks": period labels move by calendar week in blocks of ``block`` weeks, both weekdays of a week together.
    how = "within_weekday": labels move separately within each weekday, in blocks of ``block`` consecutive days of
    that weekday (the Tuesday and Thursday of a week can then get different labels).
    """
    y, is_a, after = np.asarray(y, float), np.asarray(is_a, int), np.asarray(after, int)
    ok = np.isfinite(y)
    y, is_a, after, week = y[ok], is_a[ok], after[ok], pd.DatetimeIndex(week)[ok]
    obs = float(did_many(y, is_a, after)[0])
    if how == "weeks":
        code, uniq = week_codes(week)
        wa = np.zeros(len(uniq), np.int8)
        wa[code] = after
        draws_lab = week_label_draws(wa, block, n_perm, rng)[:, code]
    elif how == "within_weekday":
        draws_lab = np.empty((n_perm, len(y)), np.int8)
        for i in range(n_perm):
            draws_lab[i] = after[cg.block_permutation(is_a, block, rng)]
    else:
        raise ValueError(how)
    draws = did_many(y, is_a, draws_lab)
    fin = draws[np.isfinite(draws)]
    return {"estimate": obs, "perm_p": perm_p(obs, draws, alternative),
            "perm_p_greater": perm_p(obs, draws, "greater"), "perm_p_less": perm_p(obs, draws, "less"),
            "perm_null_mean": float(fin.mean()) if len(fin) else np.nan, "perm_null_sd": float(fin.std(ddof=1)) if len(fin) > 1 else np.nan,
            "perm_draws_used": int(len(fin)), "n_days": int(len(y))}


def did_hac(y, is_a, after, alternative: str, lags: int = HAC_LAGS) -> dict:
    """OLS of y on [1, a, after, a x after] (days in date order), Newey-West errors; the interaction is the DiD."""
    import statsmodels.api as sm

    y, is_a, after = np.asarray(y, float), np.asarray(is_a, float), np.asarray(after, float)
    ok = np.isfinite(y)
    X = np.column_stack([np.ones(ok.sum()), is_a[ok], after[ok], is_a[ok] * after[ok]])
    res = sm.OLS(y[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    b, se = float(res.params[3]), float(res.bse[3])
    z = b / se if se > 0 else np.nan
    p = {"greater": norm.sf(z), "less": norm.cdf(z), "two-sided": 2 * norm.sf(abs(z))}[alternative]
    return {"hac_coef": b, "hac_se": se, "hac_p": float(p), "hac_ci95_lo": b - 1.959964 * se, "hac_ci95_hi": b + 1.959964 * se,
            "hac_upper95_one_sided": b + 1.644854 * se, "hac_lower95_one_sided": b - 1.644854 * se}


def paired_week_did(y, is_a, after, week) -> tuple[float, pd.DataFrame]:
    """Week fixed-effects version: D_w = y(a) - y(b) in weeks with exactly one a-day and one b-day;
    the DiD is mean(D_w, after) - mean(D_w, before)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "a": np.asarray(is_a, int), "after": np.asarray(after, int),
                       "week": pd.DatetimeIndex(week)}).dropna(subset=["y"])
    cnt = df.groupby(["week", "a"]).size().unstack(fill_value=0)
    good = cnt.index[(cnt.get(1, 0) == 1) & (cnt.get(0, 0) == 1)]
    d = df[df["week"].isin(good)]
    wk = d.pivot_table(index="week", columns="a", values="y", aggfunc="first")
    out = pd.DataFrame({"D": wk[1] - wk[0], "after": d.groupby("week")["after"].first()})
    est = float(out.loc[out["after"] == 1, "D"].mean() - out.loc[out["after"] == 0, "D"].mean())
    return est, out


def paired_week_perm(y, is_a, after, week, alternative, n_perm, block, rng) -> dict:
    est, wk = paired_week_did(y, is_a, after, week)
    D, lab = wk["D"].to_numpy(), wk["after"].to_numpy().astype(np.int8)
    L = week_label_draws(lab, block, n_perm, rng).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (L * D).sum(1) / L.sum(1) - ((1 - L) * D).sum(1) / (1 - L).sum(1)
    return {"estimate": est, "perm_p": perm_p(est, draws, alternative), "perm_null_mean": float(np.nanmean(draws)),
            "perm_draws_used": int(np.isfinite(draws).sum()), "n_days": int(2 * len(wk)), "n_weeks": int(len(wk))}


def net_from_values(v: np.ndarray) -> np.ndarray:
    """NET from cg.conn_values output [total, Sensex to Nifty, Nifty to Sensex] (last axis)."""
    v = np.asarray(v, float)
    return v[..., 2] - v[..., 1]


def per_day_values(ds, H: int, min_valid: int | None = None, Y=None) -> np.ndarray:
    """(days x 3) [total, Sensex to Nifty, Nifty to Sensex] from one VAR per day (cg.conn_values on each day's cross products)."""
    out = np.array([cg.conn_values(ds, ds.F[d], H) for d in range(ds.F.shape[0])])
    if min_valid is not None and Y is not None:
        nvalid = np.isfinite(Y).all(axis=2).sum(axis=1)
        out[nvalid < min_valid] = np.nan
    return out


def pooled_cell_did(ds, H: int, pos: np.ndarray, is_a, after) -> float:
    """CPP-style version: one pooled VAR per cell (all its days stacked, lags never crossing days), then the DiD of NET."""
    is_a, after = np.asarray(is_a, int), np.asarray(after, int)
    W = np.zeros((4, ds.F.shape[0]))
    for r, (a, f) in enumerate(((1, 1), (1, 0), (0, 1), (0, 0))):
        np.add.at(W[r], pos[(is_a == a) & (after == f)], 1.0)
    S = ds.sums(W)
    net = [net_from_values(cg.conn_values(ds, S[r], H)) for r in range(4)]
    return float((net[0] - net[1]) - (net[2] - net[3]))


def expiry_regression(y, E, weekday, period, lags: int = HAC_LAGS, weekday_effects: bool = True) -> dict:
    """H2: y on an expiry dummy, weekday dummies (Monday base) and Sensex-period dummies (first period base), HAC.
    ``weekday_effects=False`` drops the weekday dummies (the descriptive, CPP-like comparison)."""
    import statsmodels.api as sm

    y = np.asarray(y, float)
    X, names = design_h2(E, weekday, period, weekday_effects)
    ok = np.isfinite(y)
    res = sm.OLS(y[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    j = names.index("either_expiry")
    b, se = float(res.params[j]), float(res.bse[j])
    return {"estimate": b, "hac_se": se, "hac_p": float(res.pvalues[j]), "hac_ci95_lo": b - 1.959964 * se,
            "hac_ci95_hi": b + 1.959964 * se, "n_days": int(ok.sum()), "n_expiry_days": int(np.asarray(E)[ok].sum()),
            "r2": float(res.rsquared), "params": dict(zip(names, map(float, res.params)))}


def design_h2(E, weekday, period, weekday_effects: bool = True) -> tuple[np.ndarray, list]:
    E = np.asarray(E, float)
    wd = np.asarray(weekday)
    per = np.asarray(period)
    cols, names = [np.ones(len(E)), E], ["const", "either_expiry"]
    for w in range(1, 5) if weekday_effects else ():
        cols.append((wd == w).astype(float))
        names.append(f"wd_{WDN[w]}")
    for p in ("sensex_tue", "sensex_thu"):
        cols.append((per == p).astype(float))
        names.append(f"period_{p}")
    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# Minimum detectable effect (design only; the variance comes from another study's published file)
# ---------------------------------------------------------------------------


def z_mde(alpha: float = ALPHA, power: float = POWER, one_sided: bool = True) -> float:
    return float(norm.ppf(1 - alpha if one_sided else 1 - alpha / 2) + norm.ppf(power))


def mde_did(n_cells, sigma: float, alpha: float = ALPHA, power: float = POWER, one_sided: bool = True) -> float:
    """Smallest DiD detectable with the given power: z * sigma * sqrt(sum 1/n), days independent with variance sigma^2."""
    n = np.asarray(n_cells, float)
    return z_mde(alpha, power, one_sided) * sigma * float(np.sqrt(np.sum(1.0 / n)))


def mde_regression(X: np.ndarray, j: int, sigma: float, alpha: float = ALPHA, power: float = POWER, one_sided: bool = False) -> float:
    return z_mde(alpha, power, one_sided) * sigma * float(np.sqrt(np.linalg.inv(X.T @ X)[j, j]))


def proxy_variance(path: Path | None = None, regimes=PROXY_REGIMES) -> dict:
    """Day-to-day spread of NET and total connectedness from the Nifty / Bank Nifty connectedness study's published
    per-day file (results/connectedness_gamma/sample_days.csv): residual SD after removing regime x weekday means,
    the lag-one-week autocorrelation of those residuals (same weekday), and the Tue-Thu correlation within a week."""
    path = Path(path or CG_DAYS)
    sd = pd.read_csv(path, parse_dates=["date"])
    sd = sd[sd["regime"].isin(regimes)].copy()
    sd["net"] = sd["day_nifty_to_bn"] - sd["day_bn_to_nifty"]
    sd["wd"] = sd["date"].dt.weekday
    sd["week"] = week_start(sd["date"])
    out = {"source": _rel(path), "regimes": ";".join(regimes),
           "first": str(sd["date"].min().date()), "last": str(sd["date"].max().date()), "n_days": int(len(sd))}
    for col, key in (("net", "net"), ("day_total_conn", "total")):
        g = sd.groupby(["regime", "wd"])[col]
        r = sd[col] - g.transform("mean")
        out[f"sigma_{key}"] = float(np.sqrt((r ** 2).sum() / (len(sd) - g.ngroups)))
        sd[f"r_{key}"] = r
        pairs = []
        for _, gg in sd.sort_values("date").groupby("wd"):
            x = gg.set_index("week")[f"r_{key}"]
            prev = x.reindex(x.index - pd.Timedelta(days=7))
            pairs.append(pd.DataFrame({"x": x.to_numpy(), "x_prev": prev.to_numpy()}))
        pp = pd.concat(pairs).dropna()
        out[f"rho_week_{key}"] = float(np.corrcoef(pp["x"], pp["x_prev"])[0, 1])
        tt = sd[sd["wd"].isin([1, 3])].pivot_table(index="week", columns="wd", values=f"r_{key}").dropna()
        out[f"corr_tue_thu_{key}"] = float(np.corrcoef(tt[1], tt[3])[0, 1])
        out[f"n_pairs_week_{key}"], out[f"n_weeks_tue_thu_{key}"] = int(len(pp)), int(len(tt))
    return out


def design_effect(rho: float) -> float:
    """Variance inflation of a mean of an AR(1) series with lag-one correlation rho (1 when rho <= 0)."""
    return float((1 + rho) / (1 - rho)) if rho > 0 else 1.0


# ---------------------------------------------------------------------------
# Verdict (decision rule of the pre-registration, section 6)
# ---------------------------------------------------------------------------


def primary_verdict(estimate, p_greater, p_less, hac_upper95, mde, placebo: dict) -> dict:
    """Primary verdict and design-check flags. ``placebo`` maps PL1, PL2, PL3 to (estimate, two-sided p)."""
    if np.isfinite(estimate) and estimate > 0 and p_greater < ALPHA:
        v = "SUPPORTED"
    elif np.isfinite(estimate) and estimate < 0 and p_less < ALPHA:
        v = "EVIDENCE AGAINST (the expiring index becomes the net receiver)"
    elif np.isfinite(hac_upper95) and hac_upper95 < mde:
        v = "NOT SUPPORTED (effects as large as the pre-registered MDE are ruled out)"
    else:
        v = "NOT SUPPORTED (inconclusive: effects as large as the pre-registered MDE are not ruled out)"
    flags = []
    for k in ("PL2", "PL3"):
        _e, p = placebo.get(k, (np.nan, np.nan))
        if np.isfinite(p) and p < ALPHA:
            flags.append(f"DESIGN CHECK FAILED: {k} (a fake swap date) has p = {p:.3g}; the Tue minus Thu gap moves when nothing changes")
    e1, p1 = placebo.get("PL1", (np.nan, np.nan))
    if np.isfinite(p1) and p1 < ALPHA:
        if np.isfinite(estimate) and np.sign(e1) == np.sign(estimate):
            flags.append(f"PL1 (Mon minus Wed) moves in the same direction (p = {p1:.3g}): the shift is not confined to the expiry day itself")
        else:
            flags.append(f"PL1 (Mon minus Wed) moves in the opposite direction (p = {p1:.3g}): a warning about the design")
    return {"verdict": v, "design_flags": flags}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def build_pair_grids(bars: dict, trim: str, keys=INDICES, min_bars: int = MIN_BARS) -> dict:
    """cg.build_grids with the two index names as an argument (the same code; cg.build_grids is fixed to Nifty and
    Bank Nifty). Days on which both indices have at least ``min_bars`` bars; continuous log returns; daily returns."""
    grids = {k: rp.minute_grid(bars[k], trim) for k in keys}
    days = None
    for k in keys:
        ok = pd.DatetimeIndex(grids[k]["days"][np.isfinite(grids[k]["close"]).sum(axis=1) >= min_bars])
        days = ok if days is None else days.intersection(ok)
    out = {"days": days, "minutes": grids[keys[0]]["minutes"]}
    for k in keys:
        g = rp.minute_grid(bars[k], trim, days=days)
        r = rp.continuous_log_returns(g["close"], overnight=True)
        last = pd.Series(np.log(g["close"][:, ::-1]).tolist()).map(lambda row: next((x for x in row if np.isfinite(x)), np.nan))
        out[k] = {"r": r, "daily": np.r_[np.nan, np.diff(last.to_numpy())], "close": g["close"]}
    return out


def volatility_inputs(G: dict, sample_days: pd.DatetimeIndex, groups: dict, fit_fn=None) -> dict:
    """mcsGARCH conditional and intraday volatility on ``sample_days``, fitted separately on each group of days.

    ``groups`` maps a fit name to the sample days fitted together (the connectedness study fits each NSE expiry
    regime on its own). ``fit_fn`` defaults to cg.fit_regime (tests pass a stand-in)."""
    fit_fn = fit_fn or cg.fit_regime
    rowpos = {d: i for i, d in enumerate(G["days"])}
    spos = {d: i for i, d in enumerate(sample_days)}
    M = len(G["minutes"])
    cond = {k: np.full((len(sample_days), M), np.nan) for k in INDICES}
    intra = {k: np.full((len(sample_days), M), np.nan) for k in INDICES}
    rows_out = []
    for name, gdays in groups.items():
        gdays = pd.DatetimeIndex(sorted(gdays))
        rows = np.array([rowpos[d] for d in gdays])
        where = np.array([spos[d] for d in gdays])
        for k in INDICES:
            f = fit_fn(G, k, rows)
            cond[k][where] = f["cond_vol"]
            intra[k][where] = f["intraday_vol"]
            p = f.get("params", {})
            rows_out.append({"data": LABEL, "fit": name, "index": k, "first": str(gdays.min().date()), "last": str(gdays.max().date()),
                             "days_fitted": len(rows), **{q: p.get(q, np.nan) for q in ("alpha", "beta", "nu", "loglik")},
                             "persistence": p.get("alpha", np.nan) + p.get("beta", np.nan), "daily_variance_method": f.get("daily_method", "")})
    return {"cond": cond, "intra": intra, "fits": pd.DataFrame(rows_out)}


def stack_pair(src: dict) -> np.ndarray:
    """(days x minutes x 2) with each series divided by one global constant (connectedness does not change)."""
    return np.stack([src[k] / np.nanmedian(src[k]) for k in INDICES], axis=2)


def window_only(Y: np.ndarray, minutes, start: str, end: str) -> np.ndarray:
    """Copy of Y with every minute outside [start, end] set to NaN (a day's VAR then uses only the window)."""
    m = np.array([start <= t <= end for t in minutes])
    Z = Y.copy()
    Z[:, ~m, :] = np.nan
    return Z


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _rel(path: Path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


class _Tee:
    """Copy stdout to a log file for the duration of a ``with`` block (stdout is restored afterwards)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        self.old = sys.stdout
        self.tee = cg.Tee(self.path)
        sys.stdout = self.tee
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stdout = self.old
        self.tee.f.close()
        return False


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _fmt(df: pd.DataFrame, cols=None) -> str:
    d = df[cols] if cols else df
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 70):
        return d.to_string(index=False, float_format=lambda x: f"{x:.4g}")


# ---------------------------------------------------------------------------
# Phase 1: calendar and data coverage (no outcome is computed)
# ---------------------------------------------------------------------------


def load_calendar_inputs():
    """Bars (counts only), the sample days, the calendar and the checks. Computes no return, volatility or connectedness."""
    bars = {k: rp.load_minute_bars(RAW / f"{k}_1min.csv", start="2023-01-01") for k in INDICES}
    grids = {k: rp.minute_grid(bars[k], TRIM) for k in INDICES}
    nb = {k: pd.Series(np.isfinite(grids[k]["close"]).sum(axis=1), index=grids[k]["days"]) for k in INDICES}
    both = nb["nifty"].index.intersection(nb["sensex"].index)
    last_both = max(d for d in both if nb["nifty"][d] >= MIN_BARS and nb["sensex"][d] >= MIN_BARS)
    if last_both < END:
        raise SystemExit(f"bars end {last_both.date()}, before the pre-registered end {END.date()}")
    trading = pd.DatetimeIndex([d for d in both if nb["nifty"][d] >= MIN_BARS and nb["sensex"][d] >= MIN_BARS and d <= END])
    dm = pd.read_csv(RUN / "daily_market.csv", index_col=0, parse_dates=True)
    sx = sensex_expiries(trading)
    days = trading[(trading >= SAMPLE_START) & (trading <= END)]
    missing_dm = days.difference(dm.index)
    if len(missing_dm):
        raise SystemExit(f"days without a Nifty calendar row: {list(missing_dm.date)}")
    cal = build_calendar(days, dm, sx)
    cal["n_bars_nifty"] = nb["nifty"].reindex(days).to_numpy()
    cal["n_bars_sensex"] = nb["sensex"].reindex(days).to_numpy()
    cal["missing_minutes_nifty"] = 360 - cal["n_bars_nifty"]
    cal["missing_minutes_sensex"] = 360 - cal["n_bars_sensex"]
    dropped = sorted(set(nb["nifty"].index).union(nb["sensex"].index))
    dropped = [d for d in dropped if SAMPLE_START <= d <= END and d not in set(days)]
    info = {"bars": bars, "nb": nb, "dm": dm, "sx": sx, "trading": trading, "days": days, "dropped": dropped}
    return cal, info


def calendar_checks(cal: pd.DataFrame, info: dict) -> pd.DataFrame:
    rows = []
    # Sensex rule dates against the dates the Sensex confound study used (same rule, computed there)
    if SX_USED.exists():
        used = pd.read_csv(SX_USED, parse_dates=["expiry"])
        a = set(cal.index[cal["sensex_expiry"]])
        lo, hi = cal.index.min(), min(cal.index.max(), used["expiry"].max())
        b = {d for d in used["expiry"] if lo <= d <= hi}
        a = {d for d in a if d <= hi}
        rows.append({"check": "Sensex rule here vs results/sensex_confound/sensex_expiry_days_used.csv", "from": lo.date(), "to": hi.date(),
                     "n_here": len(a), "n_other": len(b), "n_both": len(a & b),
                     "here_only": ";".join(sorted(d.date().isoformat() for d in a - b)),
                     "other_only": ";".join(sorted(d.date().isoformat() for d in b - a))})
    # Nifty bhavcopy expiries against the weekday rule of src/niftygex/expiry_calendar.py
    from niftygex.expiry_calendar import weekly_expiries

    rule = set(weekly_expiries(info["trading"]))
    a = set(cal.index[cal["nifty_expiry"]])
    b = {d for d in rule if cal.index.min() <= d <= cal.index.max()}
    rows.append({"check": "Nifty expiries from bhavcopies (daily_market.csv) vs weekday rule with holiday roll-back", "from": cal.index.min().date(),
                 "to": cal.index.max().date(), "n_here": len(a), "n_other": len(b), "n_both": len(a & b),
                 "here_only": ";".join(sorted(d.date().isoformat() for d in a - b)), "other_only": ";".join(sorted(d.date().isoformat() for d in b - a))})
    # individually reported Sensex dates inside the sample
    calf = pd.read_csv(CAL_PATH, dtype=str, keep_default_na=False)
    rep = pd.to_datetime(calf.loc[calf["fact_type"] == "expiry_date", "date"])
    rep = {d for d in rep if cal.index.min() <= d <= cal.index.max()}
    a = set(cal.index[cal["sensex_expiry"]])
    rows.append({"check": "individually reported Sensex dates (calendar file) found among the rule dates", "from": cal.index.min().date(),
                 "to": cal.index.max().date(), "n_here": np.nan, "n_other": len(rep), "n_both": len(a & rep),
                 "here_only": "n/a", "other_only": ";".join(sorted(d.date().isoformat() for d in rep - a))})
    return pd.DataFrame(rows)


def coverage_tables(cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, excl = [], []
    monthly = cal["nifty_monthly"] | cal["sensex_monthly_rule"]
    specs = list(DID_TESTS) + [P1_NO_AUCTION]
    for spec in specs:
        inc, ex = select_cells(cal, spec)
        excl.append(ex)
        for cell, g in inc.groupby("cell", sort=False):
            rows.append({"test": spec.test_id, "cell": cell, "expected_status": g["expected"].iloc[0], "n_days": len(g),
                         "n_weeks": g["week"].nunique(), "first": g.index.min().date(), "last": g.index.max().date(),
                         "n_monthly_expiry_either": int(monthly.reindex(g.index).sum()), "n_auction_days": int(g["auction_day"].sum()),
                         "n_days_missing_any_minute": int(((g["missing_minutes_nifty"] > 0) | (g["missing_minutes_sensex"] > 0)).sum()),
                         "missing_minutes_total_nifty": int(g["missing_minutes_nifty"].sum()),
                         "missing_minutes_total_sensex": int(g["missing_minutes_sensex"].sum()),
                         "n_excluded_on_this_weekday": int((ex["cell"] == cell).sum())})
    # H2: days by Sensex period and expiry status
    for per, g in cal.groupby("sensex_period", sort=False):
        for lab, m in (("either expiry", g["either_expiry"]), ("Nifty expiry", g["nifty_expiry"]), ("Sensex expiry", g["sensex_expiry"]),
                       ("holiday-shifted expiry (either)", g["nifty_shifted"] | g["sensex_shifted"]), ("no expiry", ~g["either_expiry"])):
            rows.append({"test": "S2 (H2)", "cell": f"{per}: {lab}", "expected_status": "", "n_days": int(m.sum()),
                         "n_weeks": g.loc[m, "week"].nunique(), "first": g.index[m].min().date() if m.any() else None,
                         "last": g.index[m].max().date() if m.any() else None,
                         "n_monthly_expiry_either": int(monthly.reindex(g.index[m]).sum()), "n_auction_days": int(g.loc[m, "auction_day"].sum()),
                         "n_days_missing_any_minute": int(((g.loc[m, "missing_minutes_nifty"] > 0) | (g.loc[m, "missing_minutes_sensex"] > 0)).sum()),
                         "missing_minutes_total_nifty": int(g.loc[m, "missing_minutes_nifty"].sum()),
                         "missing_minutes_total_sensex": int(g.loc[m, "missing_minutes_sensex"].sum()), "n_excluded_on_this_weekday": 0})
    cov = pd.DataFrame(rows)
    cov.insert(0, "data", LABEL)
    ex = pd.concat(excl).reset_index()
    return cov, ex


def mde_table(cal: pd.DataFrame, proxy: dict) -> pd.DataFrame:
    rows = []
    sig = proxy["sigma_net"]
    deff = design_effect(proxy["rho_week_net"])
    for spec in (P1, S1, S3, PL1, PL2, PL3):
        inc, _ = select_cells(cal, spec)
        n = inc.groupby("cell", sort=False).size()
        one = spec.alternative != "two-sided"
        alphas = [ALPHA] if spec.family in ("primary", "placebo") else [ALPHA, ALPHA / 3]
        for a in alphas:
            m_sig = mde_did(n.to_numpy(), 1.0, a, POWER, one)
            proxy_ok = spec.outcome == "net_day"
            rows.append({"test": spec.test_id, "outcome": spec.outcome, "alpha": a, "sided": "one" if one else "two", "power": POWER,
                         "cells": "; ".join(f"{c} {k}" for c, k in n.items()), "mde_in_sigma_units": m_sig,
                         "sigma_proxy": sig if proxy_ok else np.nan,
                         "mde_points_independent_days": m_sig * sig if proxy_ok else np.nan,
                         "mde_points_with_design_effect": m_sig * sig * np.sqrt(deff) if proxy_ok else np.nan,
                         "mde_points_sigma_half": m_sig * sig * 0.5 if proxy_ok else np.nan,
                         "mde_points_sigma_double": m_sig * sig * 2.0 if proxy_ok else np.nan,
                         "note": "" if proxy_ok else "no published per-day last-hour values exist, so no proxy sigma"})
    X, names = design_h2(cal["either_expiry"], cal["weekday"], cal["sensex_period"])
    for a in (ALPHA, ALPHA / 3):
        m_sig = mde_regression(X, names.index("either_expiry"), 1.0, a, POWER, one_sided=False)
        rows.append({"test": "S2", "outcome": "tot_day", "alpha": a, "sided": "two", "power": POWER,
                     "cells": f"{len(cal)} days, {int(cal['either_expiry'].sum())} expiry days", "mde_in_sigma_units": m_sig,
                     "sigma_proxy": proxy["sigma_total"], "mde_points_independent_days": m_sig * proxy["sigma_total"],
                     "mde_points_with_design_effect": m_sig * proxy["sigma_total"] * np.sqrt(design_effect(proxy["rho_week_total"])),
                     "mde_points_sigma_half": m_sig * proxy["sigma_total"] * 0.5, "mde_points_sigma_double": m_sig * proxy["sigma_total"] * 2.0,
                     "note": "regression on the calendar design matrix (expiry dummy, weekday and Sensex-period dummies)"})
    out = pd.DataFrame(rows)
    out.insert(0, "data", "DESIGN (cell counts from REAL calendars; sigma from the published Nifty / Bank Nifty file)")
    return out


def run_coverage():
    OUT.mkdir(parents=True, exist_ok=True)
    with _Tee(LOG_COVERAGE):
        _run_coverage()


def _run_coverage():
    print(f"{LABEL}. P92 phase 1: calendar and data coverage only. No return, volatility, connectedness or other outcome is computed.")
    cal, info = load_calendar_inputs()
    print(f"Sample days (both indices >= {MIN_BARS} of 360 bars, {SAMPLE_START.date()} to {END.date()}): {len(cal)}; "
          f"left out (fewer bars, special sessions): {[d.date().isoformat() for d in info['dropped']]}")
    print(f"Days with any missing minute (09:15 to 15:14): Nifty {int((cal['missing_minutes_nifty'] > 0).sum())}, "
          f"Sensex {int((cal['missing_minutes_sensex'] > 0).sum())}; total missing minutes Nifty {int(cal['missing_minutes_nifty'].sum())}, "
          f"Sensex {int(cal['missing_minutes_sensex'].sum())}")
    chk = calendar_checks(cal, info)
    chk.to_csv(OUT / "calendar_check.csv", index=False)
    print("\nCalendar checks (calendar_check.csv):")
    print(_fmt(chk))
    cov, ex = coverage_tables(cal)
    cov.to_csv(OUT / "coverage.csv", index=False)
    ex.to_csv(OUT / "excluded_days.csv", index=False)
    out_cal = cal.copy()
    for spec in list(DID_TESTS) + [P1_NO_AUCTION]:
        inc, exx = select_cells(cal, spec)
        out_cal[f"cell_{spec.test_id}"] = ""
        out_cal.loc[inc.index, f"cell_{spec.test_id}"] = inc["cell"]
        out_cal.loc[exx.index, f"cell_{spec.test_id}"] = "excluded: " + exx["reason"]
    out_cal.insert(0, "data", LABEL)
    out_cal.reset_index().to_csv(OUT / "calendar_days.csv", index=False)
    print("\nCells (coverage.csv):")
    print(_fmt(cov.drop(columns="data")))
    print("\nDays on a cell weekday that were left out (excluded_days.csv):")
    print(_fmt(ex) if len(ex) else "none")
    shifted = cal[cal["nifty_shifted"] | cal["sensex_shifted"]]
    print("\nHoliday-shifted expiry days in the sample:")
    print(_fmt(shifted.reset_index()[["date", "weekday", "nifty_expiry", "nifty_shifted", "sensex_expiry", "sensex_shifted", "nifty_monthly"]]))
    proxy = proxy_variance()
    pd.DataFrame([proxy]).to_csv(OUT / "mde_proxy.csv", index=False)
    print("\nVariance proxy from results/connectedness_gamma/sample_days.csv (Nifty / Bank Nifty, published) (mde_proxy.csv):")
    print(json.dumps(proxy, indent=1))
    mde = mde_table(cal, proxy)
    mde.to_csv(OUT / "mde.csv", index=False)
    print("\nMinimum detectable effects (mde.csv):")
    print(_fmt(mde.drop(columns="data")))
    meta = {"data": LABEL, "phase": "1 (coverage only, no outcome)", "sample_start": str(cal.index.min().date()), "sample_end": str(cal.index.max().date()),
            "n_days": int(len(cal)), "pre_registered_mde_points": PREREG_MDE,
            "input_sha256_16": {_rel(p): _sha(p) for p in (RAW / "nifty_1min.csv", RAW / "sensex_1min.csv", RUN / "daily_market.csv", CAL_PATH, CG_DAYS)}}
    (OUT / "run_info_coverage.json").write_text(json.dumps(meta, indent=2))
    print("\nInputs (run_info_coverage.json): " + json.dumps(meta["input_sha256_16"]))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Phase 2: the pre-registered analysis
# ---------------------------------------------------------------------------


def run_full(n_perm: int = N_PERM):
    OUT.mkdir(parents=True, exist_ok=True)
    with _Tee(LOG), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        _run_full(n_perm)


def _run_full(n_perm: int):
    t0 = time.time()
    prereg_commit = _git("log", "-1", "--format=%H %cI", "--", str(PREREG))
    print(f"{LABEL}. P92: does the expiring index become the net transmitter on its expiry day? Pre-registration commit: "
          f"{prereg_commit or 'NOT FOUND IN GIT'}")
    cal, _info = load_calendar_inputs()
    days = pd.DatetimeIndex(cal.index)

    # ------------------------------------------------ minute volatility, exactly as the connectedness study builds it
    bars = {k: rp.load_minute_bars(RAW / f"{k}_1min.csv", start="2017-01-01") for k in INDICES}
    G = build_pair_grids(bars, TRIM)
    assert set(days) <= set(G["days"])
    regime = cal["nse_regime"]
    groups = {r: days[(regime == r).to_numpy()] for r in pd.unique(regime)}
    vol = volatility_inputs(G, days, groups)
    vol_single = volatility_inputs(G, days, {"one fit, whole sample": days})
    fits = pd.concat([vol["fits"].assign(use="main"), vol_single["fits"].assign(use="robustness: single fit")], ignore_index=True)
    fits.to_csv(OUT / "mcsgarch_fits.csv", index=False)
    print("\nmcsGARCH fits (mcsgarch_fits.csv):")
    print(_fmt(fits, ["use", "fit", "index", "first", "last", "days_fitted", "alpha", "beta", "persistence", "nu", "daily_variance_method"]))

    Y = stack_pair(vol["cond"])
    Y_intra = stack_pair(vol["intra"])
    Y_single = stack_pair(vol_single["cond"])
    Y_lh = window_only(Y, G["minutes"], *LAST_HOUR)
    Ys, segs = [], []
    for d in range(len(days)):
        ok = np.all(np.isfinite(Y[d]), axis=1)
        Ys.append(Y[d][ok])
        segs.append(d * 1_000_000 + np.cumsum(~ok)[ok])
    p_aic = rp.var_lag_aic(np.vstack(Ys), np.concatenate(segs), PMAX_AIC)
    ds = {"main": cg.day_stats(Y, P_MAIN), "lh": cg.day_stats(Y_lh, P_MAIN), "intra": cg.day_stats(Y_intra, P_MAIN),
          "single": cg.day_stats(Y_single, P_MAIN), "aic": cg.day_stats(Y, p_aic)}
    v = {"day": per_day_values(ds["main"], H_MAIN), "lh": per_day_values(ds["lh"], H_MAIN, MIN_LAST_HOUR, Y_lh),
         "intra": per_day_values(ds["intra"], H_MAIN), "single": per_day_values(ds["single"], H_MAIN), "aic": per_day_values(ds["aic"], H_SENS)}
    # check: the per-day value equals the replication's connectedness function on that day alone
    d0 = int(np.where(cal["status"].to_numpy() != "none")[0][0])
    ok0 = np.all(np.isfinite(Y[d0]), axis=1)
    ref = rp.connectedness_segments(Y[d0][ok0], np.cumsum(~ok0)[ok0], P_MAIN, H_MAIN)
    chk = float(np.max(np.abs(v["day"][d0] - [ref.paper_scale_total, ref.table[0, 1], ref.table[1, 0]])))
    print(f"\nVAR lag by AIC on all sample days: {p_aic}. Check, per-day value vs replication.connectedness_segments: max diff {chk:.2e}.")

    dv = cal.copy()
    for key, arr in v.items():
        dv[f"tot_{key}"] = arr[:, 0]
        dv[f"s2n_{key}"] = arr[:, 1]
        dv[f"n2s_{key}"] = arr[:, 2]
        dv[f"net_{key}"] = net_from_values(arr)
    dv["n_minutes_both_valid"] = np.isfinite(Y).all(axis=2).sum(axis=1)
    dv["n_minutes_both_valid_last_hour"] = np.isfinite(Y_lh).all(axis=2).sum(axis=1)
    out_dv = dv.copy()
    out_dv.insert(0, "data", LABEL)
    out_dv.reset_index().to_csv(OUT / "day_values.csv", index=False)

    # ------------------------------------------------ the DiD tests
    def run_spec(spec: DidSpec, outcome=None, drop=None, how="weeks", block=BLOCK_WEEKS, key=None):
        inc, _ = select_cells(dv, spec, drop)
        y = inc[outcome or spec.outcome].to_numpy(float)
        rng = np.random.default_rng([SEED, STUDY, key or spec.seed_key])
        r = did_perm_test(y, inc["is_a"], inc["after"], inc["week"], spec.alternative, n_perm, block, rng, how)
        r.update(did_hac(y, inc["is_a"].to_numpy(), inc["after"].to_numpy(), spec.alternative))
        r["n_cells"] = "; ".join(f"{c} {int(np.isfinite(g[outcome or spec.outcome]).sum())}" for c, g in inc.groupby("cell", sort=False))
        return r, inc

    trows, cells = [], []
    for spec in DID_TESTS:
        r, inc = run_spec(spec)
        trows.append({"data": LABEL, "test": spec.test_id, "family": spec.family, "what": spec.what, "outcome": spec.outcome,
                      "alternative": spec.alternative, "block_weeks": BLOCK_WEEKS, **r})
        for c, g in inc.groupby("cell", sort=False):
            yy = g[spec.outcome].to_numpy(float)
            yy = yy[np.isfinite(yy)]
            cells.append({"data": LABEL, "test": spec.test_id, "cell": c, "expected_status": g["expected"].iloc[0], "n_days": len(yy),
                          "mean": yy.mean() if len(yy) else np.nan, "sd": yy.std(ddof=1) if len(yy) > 1 else np.nan,
                          "median": np.median(yy) if len(yy) else np.nan,
                          "mean_tot": g[spec.outcome.replace("net", "tot")].mean(), "mean_n2s": g[spec.outcome.replace("net", "n2s")].mean(),
                          "mean_s2n": g[spec.outcome.replace("net", "s2n")].mean()})
    # H2
    h2 = expiry_regression(dv["tot_day"], dv["either_expiry"], dv["weekday"], dv["sensex_period"])
    params = h2.pop("params")
    trows.append({"data": LABEL, "test": "S2", "family": "secondary", "what": "H2: total connectedness, either index's expiry day minus other days "
                  "(weekday and Sensex-period effects, HAC)", "outcome": "tot_day", "alternative": "two-sided",
                  **h2, "perm_p": np.nan})
    h2d = expiry_regression(dv["tot_day"], dv["either_expiry"], dv["weekday"], dv["sensex_period"], weekday_effects=False)
    params_d = h2d.pop("params")
    pd.DataFrame([{"data": LABEL, "spec": spec_, "term": k, "coef": b} for spec_, pr in
                  (("H2 (weekday and Sensex-period effects)", params), ("descriptive: Sensex-period effects only, no weekday effects", params_d))
                  for k, b in pr.items()]).to_csv(OUT / "h2_regression.csv", index=False)
    tests = pd.DataFrame(trows)
    tests["p_used"] = np.where(tests["test"] == "S2", tests["hac_p"], tests["perm_p"])
    sec = tests["family"] == "secondary"
    tests["holm_p_secondary"] = np.nan
    tests.loc[sec, "holm_p_secondary"] = cg.holm(tests.loc[sec, "p_used"].to_numpy())
    tests.to_csv(OUT / "tests.csv", index=False)
    pd.DataFrame(cells).to_csv(OUT / "cell_means.csv", index=False)

    # ------------------------------------------------ robustness of the primary (reported, never decisive)
    monthly = dv["nifty_monthly"] | dv["sensex_monthly_rule"]
    rob = []

    def add(name, r):
        rob.append({"data": LABEL, "check": name, **r})

    for k, blk in ((11, 1), (12, 2), (13, 8)):
        add(f"block of {blk} week(s)", run_spec(P1, block=blk, key=k)[0])
    add("labels moved separately within each weekday (blocks of 4 same-weekday days)", run_spec(P1, how="within_weekday", key=14)[0])
    add("closing-auction days left out (after window ends 31 Jul 2026)", run_spec(P1_NO_AUCTION, key=15)[0])
    add("monthly expiry days of either index left out", run_spec(P1, drop=monthly, key=16)[0])
    add("input = intraday mcsGARCH part sqrt(q) only", run_spec(P1, outcome="net_intra", key=17)[0])
    add(f"VAR({p_aic}) by AIC, horizon {H_SENS}", run_spec(P1, outcome="net_aic", key=18)[0])
    add("one mcsGARCH fit over the whole sample", run_spec(P1, outcome="net_single", key=19)[0])
    inc, _ = select_cells(dv, P1)
    rng = np.random.default_rng([SEED, STUDY, 21])
    add("week fixed effects (paired Tue minus Thu in weeks with both)", paired_week_perm(inc["net_day"], inc["is_a"], inc["after"], inc["week"],
                                                                                          "greater", n_perm, BLOCK_WEEKS, rng))
    rk = inc["net_day"].rank().to_numpy()
    add("ranks of NET instead of values", did_perm_test(rk, inc["is_a"], inc["after"], inc["week"], "greater", n_perm, BLOCK_WEEKS,
                                                         np.random.default_rng([SEED, STUDY, 22])))
    for k, col, alt, nm in ((23, "n2s_day", "greater", "Nifty to Sensex share alone (one half of NET; H1 implies a rise)"),
                            (24, "s2n_day", "less", "Sensex to Nifty share alone (other half of NET; H1 implies a fall)"),
                            (25, "tot_day", "two-sided", "total connectedness (descriptive)")):
        add(nm, run_spec(replace(P1, alternative=alt), outcome=col, key=k)[0] | {"alternative_used": alt})
    # CPP-style pooled cells
    pos = np.searchsorted(days.values, inc.index.values)
    obs = pooled_cell_did(ds["main"], H_MAIN, pos, inc["is_a"].to_numpy(), inc["after"].to_numpy())
    code, uniq = week_codes(inc["week"])
    wa = np.zeros(len(uniq), np.int8)
    wa[code] = inc["after"].to_numpy()
    L = week_label_draws(wa, BLOCK_WEEKS, n_perm, np.random.default_rng([SEED, STUDY, 20]))[:, code]
    draws = np.array([pooled_cell_did(ds["main"], H_MAIN, pos, inc["is_a"].to_numpy(), L[i]) for i in range(n_perm)])
    add("pooled VAR per cell (CPP style), NET of the pooled fits", {"estimate": obs, "perm_p": perm_p(obs, draws, "greater"),
                                                                     "perm_null_mean": float(np.nanmean(draws)), "perm_draws_used": int(np.isfinite(draws).sum()),
                                                                     "n_days": int(len(inc))})
    add("H2 descriptive: expiry minus other days without weekday effects (CPP-like; confounded with weekday)",
        {**h2d, "alternative_used": "two-sided"})
    robust = pd.DataFrame(rob)
    robust.to_csv(OUT / "robustness.csv", index=False)

    # ------------------------------------------------ verdict
    mde_tab = mde_table(cal, proxy_variance())
    mde_tab.to_csv(OUT / "mde.csv", index=False)
    mde_p = PREREG_MDE  # the pre-registered number, whatever the proxy file now gives
    inc_p1, _ = select_cells(dv, P1)
    sd_obs = float(np.sqrt(inc_p1.groupby("cell")["net_day"].var(ddof=1).mul(inc_p1.groupby("cell")["net_day"].count() - 1).sum()
                           / (inc_p1["net_day"].notna().sum() - 4)))
    n_obs = inc_p1.groupby("cell")["net_day"].count().to_numpy()
    t = tests.set_index("test")
    pv = {k: (float(t.loc[k, "estimate"]), float(t.loc[k, "perm_p"])) for k in ("PL1", "PL2", "PL3")}
    verdict = primary_verdict(float(t.loc["P1", "estimate"]), float(t.loc["P1", "perm_p_greater"]), float(t.loc["P1", "perm_p_less"]),
                              float(t.loc["P1", "hac_upper95_one_sided"]), mde_p, pv)
    verdict.update({"primary_estimate": float(t.loc["P1", "estimate"]), "primary_perm_p": float(t.loc["P1", "perm_p"]),
                    "primary_hac_p": float(t.loc["P1", "hac_p"]), "pre_registered_mde_points": mde_p,
                    "observed_within_cell_sd_net": sd_obs, "mde_implied_by_observed_sd (not decisive)": mde_did(n_obs, sd_obs),
                    "secondary_holm": {k: float(t.loc[k, "holm_p_secondary"]) for k in ("S1", "S2", "S3")}})
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))

    info_json = {"data": LABEL, "prereg_commit": prereg_commit, "seed": SEED, "study": STUDY,
                 "seed_rule": "numpy.random.default_rng([20261102, 92, key]); keys in tests.csv / robustness.csv order: "
                              "P1 1, S1 2, S3 3, PL1 4, PL2 5, PL3 6, blocks 11-13, within weekday 14, auction 15, monthly 16, intraday 17, "
                              "AIC 18, single fit 19, pooled 20, week FE 21, ranks 22, n2s 23, s2n 24, total 25",
                 "n_perm": n_perm, "block_weeks": BLOCK_WEEKS, "var_p_main": P_MAIN, "H_main": H_MAIN, "var_p_aic": int(p_aic), "H_sensitivity": H_SENS,
                 "session_rule": TRIM, "last_hour": LAST_HOUR, "min_last_hour": MIN_LAST_HOUR, "sample_start": str(days.min().date()),
                 "sample_end": str(days.max().date()), "n_days": int(len(days)), "per_day_check_max_abs_diff": chk,
                 "timing_s": round(time.time() - t0, 1),
                 "input_sha256_16": {_rel(p): _sha(p) for p in (RAW / "nifty_1min.csv", RAW / "sensex_1min.csv", RUN / "daily_market.csv",
                                                                                  CAL_PATH, CG_DAYS)}}
    (OUT / "run_info.json").write_text(json.dumps(info_json, indent=2, default=str))

    print("\nCell means (cell_means.csv):")
    print(_fmt(pd.DataFrame(cells).drop(columns="data")))
    print("\nTests (tests.csv):")
    print(_fmt(tests, ["test", "family", "outcome", "alternative", "estimate", "perm_p", "hac_p", "hac_ci95_lo", "hac_ci95_hi", "holm_p_secondary",
                       "n_cells"]))
    print("\nRobustness of the primary (robustness.csv; not decisive):")
    print(_fmt(robust, ["check", "estimate", "perm_p", "hac_p", "n_days"]))
    print("\nVerdict (verdict.json):")
    print(json.dumps(verdict, indent=2))
    print(f"\nTiming: {time.time() - t0:.1f} s")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true", help="phase 1: calendars and data coverage only, no outcome")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    a = ap.parse_args()
    if a.coverage:
        run_coverage()
    else:
        run_full(a.n_perm)


if __name__ == "__main__":
    main()
