"""Replicate the empirical tables of Chhimwal, Pandey and Pandey (2025) on REAL data.

Paper: "Effect of multiple index derivative expiry on volatility, volume, and
connectedness: a tale of two stock indices in India", Review of Derivatives
Research 28(3). Every number this script writes is computed from REAL DATA:
third-party 1-minute Nifty 50 and Bank Nifty bars (data/raw/*_1min.csv),
validated here against NSE's official daily index values, plus NSE's daily
F&O bhavcopy. The paper's printed values appear only in columns named
``paper_printed`` (comparison targets typed in from its tables).

Usage (about 5 to 10 minutes on 4 cores):

    OMP_NUM_THREADS=1 python experiments/replicate_paper.py
    OMP_NUM_THREADS=1 python experiments/replicate_paper.py --trim 375_full   # other session rule as primary
    OMP_NUM_THREADS=1 python experiments/replicate_paper.py --quick           # fewer bootstrap draws

Writes results/replication_nse/*.csv, two PNG figures, and
results/replication_nse_stdout.txt. Deterministic: all random draws are seeded.

Three samples are analysed:

* ``after``            1 March to 31 May 2024 (as the paper states).
* ``before_stated``    1 November 2023 to 29 February 2024 (as the paper states).
* ``before_inferred``  the 64-trading-day window between July 2022 and
                       February 2024 whose return statistics best match the
                       paper's Table 1 (found by ``fingerprint_search``).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
from scipy.stats import mannwhitneyu

from niftygex import replication as rp
from niftygex.inference import day_cluster_bootstrap, day_level_mwu, hac_mean_difference, minute_level_mwu
from niftygex.realized import rogers_satchell
from niftygex.volatility import McsGarch, daily_egarch_variance, fit_garch11

RAW = Path("data/raw")
OUT = Path("results/replication_nse")
LOG = Path("results/replication_nse_stdout.txt")
LABEL = "REAL DATA (third-party 1-minute bars validated against NSE official daily values)"
INDICES = ("nifty", "banknifty")
OFFICIAL_NAMES = {"Nifty 50": "nifty", "Nifty Bank": "banknifty", "India VIX": "vix"}
SYMBOL = {"nifty": "NIFTY", "banknifty": "BANKNIFTY"}

AFTER = ("2024-03-01", "2024-05-31")
STATED_BEFORE = ("2023-11-01", "2024-02-29")
SEARCH_RANGE = ("2022-07-01", "2024-02-29")
DATA_RANGE = ("2022-10-20", "2024-06-30")  # range validated and used for calendars
PAPER_DAYS = {"before": 64, "after": 59}
PAPER_WEEKDAY_DAYS = {("before", "wed"): 13, ("before", "thu"): 13, ("after", "wed"): 12, ("after", "thu"): 13}

# ---------------------------------------------------------------------------
# Paper (printed) comparison targets
# ---------------------------------------------------------------------------

P_T1 = {
    ("before", "nifty"): dict(n=23039, mean=-0.000001, sd=0.000360947, min=-0.008584649, max=0.006221108, skew=-0.524960403, excess_kurtosis=64.74526352, anderson_darling=649.191, adf=-23.05),
    ("before", "banknifty"): dict(n=23039, mean=-0.000002, sd=0.000522211, min=-0.017256735, max=0.012023448, skew=-1.357725832, excess_kurtosis=125.1117068, anderson_darling=790.061, adf=-20.69261),
    ("after", "nifty"): dict(n=21239, mean=0.000001, sd=0.000355, min=-0.007086, max=0.009408, skew=0.53, excess_kurtosis=75.88, anderson_darling=492.399, adf=-29.626103),
    ("after", "banknifty"): dict(n=21239, mean=0.000002, sd=0.000443, min=-0.008699, max=0.009905, skew=0.64, excess_kurtosis=57.3, anderson_darling=466.153, adf=-145.6132),
}
P_T1_CORR = {"before": 0.854, "after": 0.818}
P_T2 = {
    ("before", "nifty"): dict(phi=-0.054778, theta=0.078178, omega=0.002319, alpha=0.026481, beta=0.971477, shape=8.864628, loglik=158616.4, aic=-13.769),
    ("before", "banknifty"): dict(phi=0.619437, theta=-0.642318, omega=0.004263, alpha=0.032328, beta=0.963935, shape=6.70796, loglik=148500.0, aic=-13.095),
    ("after", "nifty"): dict(phi=-0.627945, theta=0.638569, omega=0.010345, alpha=0.049363, beta=0.941048, shape=6.983731, loglik=145063.0, aic=-13.659),
    ("after", "banknifty"): dict(phi=0.750097, theta=-0.792505, omega=0.018692, alpha=0.053195, beta=0.929680, shape=5.483723, loglik=139758.5, aic=-13.160),
}
# Tables 3 and 4: [[Nifty own, Nifty from Bank Nifty], [Bank Nifty from Nifty, Bank Nifty own]] and the printed total.
P_T34 = {
    ("before", "union", "all_expiry"): ([[57.29, 42.71], [41.29, 58.71]], 84.00),
    ("before", "union", "monthly"): ([[63.27, 36.73], [37.97, 62.03]], 74.70),
    ("before", "union", "monthly_first_hour"): ([[50.33, 49.67], [49.75, 50.25]], 99.42),
    ("before", "union", "monthly_mid_hour"): ([[66.26, 33.74], [34.90, 65.10]], 68.64),
    ("before", "union", "monthly_last_hour"): ([[47.21, 52.79], [52.43, 47.57]], 105.22),
    ("after", "nifty", "all_expiry"): ([[61.57, 38.43], [37.95, 62.05]], 76.38),
    ("after", "nifty", "monthly"): ([[63.21, 36.79], [37.79, 62.21]], 74.58),
    ("after", "nifty", "monthly_first_hour"): ([[60.86, 40.14], [40.03, 60.97]], 80.18),
    ("after", "nifty", "monthly_mid_hour"): ([[66.32, 33.68], [35.13, 64.87]], 68.81),
    ("after", "nifty", "monthly_last_hour"): ([[70.83, 29.17], [37.27, 62.73]], 66.44),
    ("after", "bn", "all_expiry"): ([[62.55, 37.45], [35.57, 64.43]], 73.02),
    ("after", "bn", "monthly"): ([[62.87, 37.13], [35.61, 64.39]], 72.74),
    ("after", "bn", "monthly_first_hour"): ([[62.44, 37.56], [34.41, 65.59]], 70.97),
    ("after", "bn", "monthly_mid_hour"): ([[66.03, 33.97], [32.57, 67.43]], 66.54),
    ("after", "bn", "monthly_last_hour"): ([[61.19, 38.81], [35.62, 64.38]], 74.43),
}
# Tables 5 and 6: (n_before, median_before, n_after, median_after, printed statistic, printed p).
P_T5 = {
    ("all", "nifty"): (23039, 478000, 21463, 634000, 444392222.5, 0.000),
    ("all", "banknifty"): (23039, 439000, 21463, 380000, 538810985.5, 0.000),
    ("wed", "nifty"): (4680, 483000, 4320, 576000, 19165172.0, 0.000),
    ("wed", "banknifty"): (4680, 405000, 4320, 444000, 21613964.5, 0.000),
    ("thu", "nifty"): (4680, 469000, 4680, 774000, 16357754.5, 0.000),
    ("thu", "banknifty"): (4680, 425000, 4680, 411000, 21478045.0, 0.0011),
}
P_T5_TEXT_PCT = {("all", "nifty"): 32, ("wed", "nifty"): 19, ("thu", "nifty"): 61,
                 ("all", "banknifty"): -13, ("wed", "banknifty"): 9, ("thu", "banknifty"): -3}
P_T6 = {
    ("all", "nifty"): (23038, 0.00024, 21462, 0.00026, 483967708.5, 0.000),
    ("all", "banknifty"): (23038, 0.00034, 21462, 0.00034, 500287483.5, 0.087),
    ("wed", "nifty"): (4679, 0.00023, 4319, 0.00023, 18633563.0, 0.061),
    ("wed", "banknifty"): (4680, 0.00033, 4319, 0.00036, 207455847.5, 0.000),
    ("thu", "nifty"): (4679, 0.00024, 4679, 0.00027, 156272524.5, 0.000),
    ("thu", "banknifty"): (4679, 0.00034, 4679, 0.00034, 2174645383.0, 0.079),
}


class Tee:
    """Write everything printed to the console and to the stdout file."""

    def __init__(self, path: Path):
        self.f = open(path, "w")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def note(msg: str):
    """Progress and timing messages go to stderr only, so the stdout file stays deterministic."""
    print(msg, file=sys.__stderr__, flush=True)


def header(title: str):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_official() -> pd.DataFrame:
    """NSE official daily index values.

    Built from the per-day NSE files (ind_close_all_DDMMYYYY.csv) with the
    date taken from the FILE NAME: some April 2023 files write the date
    month-first inside the file, so parsing the in-file date misdates them.
    data/raw/index_daily.csv (rebuilt the same way) only fills dates for which
    no per-day file exists.
    """
    frames = []
    files = sorted((RAW / "extra_2022" / "index_close").glob("ind_close_all_*.csv")) + sorted((RAW / "index_close").glob("ind_close_all_*.csv"))
    if files:
        parts = []
        for f in files:
            d = pd.read_csv(f, dtype=str)
            d["date"] = pd.to_datetime(f.stem.split("_")[-1], format="%d%m%Y")
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        names = {"Index Name": "index", "Open Index Value": "open", "High Index Value": "high", "Low Index Value": "low",
                 "Closing Index Value": "close", "Volume": "volume", "Turnover (Rs. Cr.)": "turnover_cr"}
        df = df.rename(columns=names)
        for c in ("open", "high", "low", "close", "volume", "turnover_cr"):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
        df["index"] = df["index"].str.strip()
        frames.append(df[["date", "index", "open", "high", "low", "close", "volume", "turnover_cr"]])
    p = RAW / "index_daily.csv"
    if p.exists():
        try:
            frames.append(pd.read_csv(p, parse_dates=["date"]))
        except Exception as e:  # noqa: BLE001 - keep going with the per-day files
            note(f"could not read {p}: {e}")
    off = pd.concat(frames, ignore_index=True)
    off["index"] = off["index"].str.strip()
    off["key"] = off["index"].map(OFFICIAL_NAMES)
    off = off.dropna(subset=["key"]).drop_duplicates(["date", "key"]).sort_values(["key", "date"])
    return off.reset_index(drop=True)


def load_bhavcopy(start, end):
    """Contract rows (for the expiry calendar) and day-level F&O volume, legacy layout only."""
    files = sorted((RAW / "extra_2022" / "bhavcopy").glob("fo_*.csv.gz")) + sorted((RAW / "bhavcopy").glob("fo_*.csv.gz"))
    s, e = pd.Timestamp(start).strftime("%Y%m%d"), pd.Timestamp(end).strftime("%Y%m%d")
    files = [f for f in files if s <= f.name[3:11] <= e]
    contracts, volume, skipped = [], [], 0
    for f in files:
        d = rp.read_legacy_bhavcopy(f)
        if d is None:
            skipped += 1
            continue
        d = d[d["SYMBOL"].isin(SYMBOL.values())]
        contracts.append(d[["SYMBOL", "INSTRUMENT", "expiry", "date"]].drop_duplicates())
        volume.append(rp.daily_fo_volume(d))
    return pd.concat(contracts, ignore_index=True), pd.concat(volume, ignore_index=True), len(files), skipped


# ---------------------------------------------------------------------------
# Validation of minute bars against official daily values
# ---------------------------------------------------------------------------


def validate(bars: dict, off: pd.DataFrame, start, end) -> pd.DataFrame:
    rows = []
    for k in INDICES:
        b = bars[k]
        b = b[(b["date"] >= start) & (b["date"] <= end)]
        reg = b[(b["tod"] >= "09:15") & (b["tod"] <= "15:29")]
        g = reg.groupby("date").agg(m_open=("open", "first"), m_high=("high", "max"), m_low=("low", "min"),
                                    m_close=("close", "last"), n_bars=("close", "size"))
        allb = b.groupby("date").size().rename("n_bars_any")
        o = off[(off["key"] == k) & (off["date"] >= start) & (off["date"] <= end)].set_index("date")
        j = g.join(allb, how="outer").join(o[["open", "high", "low", "close"]], how="outer")
        for a, c in (("m_open", "open"), ("m_high", "high"), ("m_low", "low"), ("m_close", "close")):
            j[f"diff_{c}_bps"] = 1e4 * (j[a] / j[c] - 1)
        j["index"] = k
        j["weekday"] = j.index.day_name()
        j["in_official"] = j["close"].notna()
        j["in_minute_bars"] = j["n_bars_any"].notna()
        j["flag_ohl_gt_5bps"] = (j[["diff_open_bps", "diff_high_bps", "diff_low_bps"]].abs() > 5).any(axis=1)
        j.index.name = "date"
        rows.append(j.reset_index())
    v = pd.concat(rows, ignore_index=True)
    v = v.rename(columns={"open": "official_open", "high": "official_high", "low": "official_low", "close": "official_close"})
    return v


def validation_summary(v: pd.DataFrame, periods: dict) -> pd.DataFrame:
    rows = []
    spans = {"whole_range": (pd.Timestamp(DATA_RANGE[0]), pd.Timestamp(DATA_RANGE[1]))}
    spans.update({p: (d.min(), d.max()) for p, d in periods.items()})
    for name, (a, b) in spans.items():
        for k in INDICES:
            x = v[(v["index"] == k) & (v["date"] >= a) & (v["date"] <= b)]
            both = x[x["in_official"] & x["in_minute_bars"] & x["m_close"].notna()]
            rows.append({
                "data": LABEL, "span": name, "start": a.date(), "end": b.date(), "index": k,
                "official_days": int(x["in_official"].sum()),
                "minute_bar_days_weekday": int((x["in_minute_bars"] & (pd.to_datetime(x["date"]).dt.weekday < 5)).sum()),
                "official_days_missing_in_minute_bars": ";".join(pd.to_datetime(x.loc[x["in_official"] & ~x["in_minute_bars"], "date"]).dt.strftime("%Y-%m-%d")),
                "minute_days_not_in_official": ";".join(pd.to_datetime(x.loc[~x["in_official"] & x["in_minute_bars"], "date"]).dt.strftime("%Y-%m-%d")),
                "days_compared": len(both),
                "days_with_fewer_than_375_bars": ";".join(pd.to_datetime(both.loc[both["n_bars"] < 375, "date"]).dt.strftime("%Y-%m-%d")),
                "open_median_abs_bps": both["diff_open_bps"].abs().median(), "open_max_abs_bps": both["diff_open_bps"].abs().max(),
                "high_median_abs_bps": both["diff_high_bps"].abs().median(), "high_max_abs_bps": both["diff_high_bps"].abs().max(),
                "low_median_abs_bps": both["diff_low_bps"].abs().median(), "low_max_abs_bps": both["diff_low_bps"].abs().max(),
                "close_median_abs_bps": both["diff_close_bps"].abs().median(), "close_max_abs_bps": both["diff_close_bps"].abs().max(),
                "days_open_high_low_off_by_more_than_5bps": ";".join(pd.to_datetime(both.loc[both["flag_ohl_gt_5bps"], "date"]).dt.strftime("%Y-%m-%d")),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grids, returns and daily variance
# ---------------------------------------------------------------------------


def build_grids(bars: dict, trim: str, overnight: bool = True) -> dict:
    """(days x minutes) grids on the days where both indices have at least 300 bars in the session."""
    grids = {k: rp.minute_grid(bars[k], trim) for k in INDICES}
    days = None
    for k in INDICES:
        ok = pd.DatetimeIndex(grids[k]["days"][np.isfinite(grids[k]["close"]).sum(axis=1) >= 300])
        days = ok if days is None else days.intersection(ok)
    out = {"days": days, "minutes": grids["nifty"]["minutes"], "trim": trim, "overnight": overnight}
    for k in INDICES:
        g = rp.minute_grid(bars[k], trim, days=days)
        r = rp.continuous_log_returns(g["close"], overnight=overnight)
        last = pd.Series(np.log(g["close"][:, ::-1]).tolist()).map(lambda row: next((x for x in row if np.isfinite(x)), np.nan))
        daily = np.r_[np.nan, np.diff(last.to_numpy())]
        rs = rogers_satchell(g["open"], g["high"], g["low"], g["close"])
        out[k] = {"open": g["open"], "high": g["high"], "low": g["low"], "close": g["close"], "r": r, "daily": daily,
                  "rs_vol": np.sqrt(np.maximum(rs, 0.0))}
    return out


def period_rows(G: dict, days: pd.DatetimeIndex) -> np.ndarray:
    return np.where(G["days"].isin(days))[0]


def window_returns(G: dict, k: str, rows: np.ndarray) -> np.ndarray:
    """Returns of a window as the paper counts them: the window's very first return is dropped."""
    R = G[k]["r"][rows].copy()
    R[0, 0] = np.nan
    return R


_EGARCH_CACHE: dict = {}


def daily_variance(G: dict, k: str, last_row: int) -> np.ndarray:
    """AR(1)-EGARCH(1,1)-t one-step-ahead daily variance, fitted on all days up to ``last_row`` (in-sample per period)."""
    key = (G["trim"], k, last_row)
    if key not in _EGARCH_CACHE:
        y = G[k]["daily"][1 : last_row + 1]
        h, info = daily_egarch_variance(y)
        hf = np.full(len(G["days"]), np.nan)
        hf[1 : last_row + 1] = h
        _EGARCH_CACHE[key] = (hf, info["method"])
    return _EGARCH_CACHE[key]


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------


def describe(r: np.ndarray, full: bool) -> dict:
    r = r[np.isfinite(r)]
    d = {"n": len(r), "mean": r.mean(), "sd": r.std(ddof=1), "min": r.min(), "max": r.max(),
         "skew": stats.skew(r), "excess_kurtosis": stats.kurtosis(r)}
    if full:
        from statsmodels.tsa.stattools import adfuller

        d["anderson_darling"] = stats.anderson(r, dist="norm").statistic
        d["adf"] = adfuller(r, autolag="AIC")[0]
    return d


def fingerprint_score(st: dict, target: dict) -> float:
    return (abs(np.log(st["sd"] / target["sd"])) + abs(st["min"] - target["min"]) / abs(target["min"])
            + abs(st["max"] - target["max"]) / abs(target["max"])
            + abs(st["excess_kurtosis"] - target["excess_kurtosis"]) / abs(target["excess_kurtosis"]))


def fingerprint_search(G: dict, n_days: int = 64) -> pd.DataFrame:
    """Score every window of ``n_days`` consecutive trading days against the paper's Table 1 'before' column."""
    days = G["days"]
    a, b = pd.Timestamp(SEARCH_RANGE[0]), pd.Timestamp(SEARCH_RANGE[1])
    rows = []
    for s in range(len(days) - n_days + 1):
        d0, d1 = days[s], days[s + n_days - 1]
        if d0 < a or d1 > b:
            continue
        rec = {"start": d0.date(), "end": d1.date(), "score": 0.0}
        for k in INDICES:
            st = describe(window_returns(G, k, np.arange(s, s + n_days)).ravel(), full=False)
            rec.update({f"{k}_{x}": st[x] for x in ("n", "sd", "min", "max", "skew", "excess_kurtosis")})
            rec["score"] += fingerprint_score(st, P_T1[("before", k)])
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def table1(G: dict, periods: dict, full: bool) -> pd.DataFrame:
    rows = []
    for p, days in periods.items():
        side = "after" if p == "after" else "before"
        rr = period_rows(G, days)
        R = {k: window_returns(G, k, rr) for k in INDICES}
        for k in INDICES:
            st = describe(R[k].ravel(), full)
            for s, v in st.items():
                rows.append({"data": LABEL, "period": p, "index": k, "statistic": s, "ours": v,
                             "paper_printed": P_T1[(side, k)].get(s, np.nan)})
        a, b = R["nifty"].ravel(), R["banknifty"].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        rows.append({"data": LABEL, "period": p, "index": "both", "statistic": "correlation",
                     "ours": np.corrcoef(a[ok], b[ok])[0, 1], "paper_printed": P_T1_CORR[side]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 2: mcsGARCH
# ---------------------------------------------------------------------------


def arma_robust_z(r: np.ndarray, mp: dict) -> dict:
    """Sandwich (heteroskedasticity-robust) z statistics for the two-step ARMA(1,1) conditional least squares fit."""
    rl = np.concatenate([[r.mean()], r[:-1]])

    def resid(p):
        return lfilter([1.0], [1.0, p[2]], r - p[0] - p[1] * rl)

    p0 = np.array([mp["c"], mp["phi"], mp["theta"]])
    e0 = resid(p0)
    J = np.empty((len(r), 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = 1e-6 * max(abs(p0[j]), 1e-3)
        J[:, j] = (resid(p0 + d) - resid(p0 - d)) / (2 * d[j])
    A = np.linalg.inv(J.T @ J)
    cov = A @ ((J * (e0**2)[:, None]).T @ J) @ A
    return dict(zip(("c", "phi", "theta"), p0 / np.sqrt(np.diag(cov))))


def fit_period(G: dict, k: str, rows: np.ndarray) -> dict:
    """mcsGARCH on one index and one sample.

    Workaround for a bug in McsGarch.fit: if a whole minute-of-day column has
    no valid return (as when every first-minute return is removed), its
    diurnal factor is 0, the normalised residual there is 0/0 = NaN, and the
    intraday GARCH silently returns its starting values. Such columns are
    dropped before fitting and put back as NaN afterwards.
    """
    R = window_returns(G, k, rows)
    keep = np.isfinite(R).any(axis=0)
    Rk = R[:, keep]
    hf, method = daily_variance(G, k, int(rows[-1]))
    h = hf[rows]
    mg = McsGarch(mean="arma11", dist="t", smooth=1).fit(Rk, h, compute_tstats=True)
    filt = mg.filter(Rk, h)
    valid_k = np.isfinite(Rk)
    S = filt["S"]
    ll_r = mg.params["loglik"] - 0.5 * float(np.sum(np.log(h[:, None] * S)[valid_k]))
    n = int(valid_k.sum())
    cond_vol = np.full(R.shape, np.nan)
    intraday_vol = np.full(R.shape, np.nan)
    cond_vol[:, keep] = np.sqrt(h[:, None] * S * filt["q2"])
    intraday_vol[:, keep] = np.sqrt(filt["q2"])
    return {"mg": mg, "R": R, "Rk": Rk, "keep": keep, "h": h, "filt": filt, "valid": np.isfinite(R), "valid_k": valid_k,
            "ll_returns": ll_r, "n": n, "aic_per_obs": (-2 * ll_r + 2 * 7) / n, "daily_method": method,
            "cond_vol": cond_vol, "intraday_vol": intraday_vol}


def table2_rows(fits: dict, full: bool) -> pd.DataFrame:
    rows = []
    for (p, k), f in fits.items():
        side = "after" if p == "after" else "before"
        mg = f["mg"]
        par = mg.params
        z = par.get("tstats", {})
        zm = arma_robust_z(np.where(f["valid_k"], f["Rk"], 0.0).ravel(), mg.mean_params) if full else {}
        P = par["alpha"] + par["beta"]
        vals = [("constant", mg.mean_params["c"], zm.get("c")), ("phi_ar1", mg.mean_params["phi"], zm.get("phi")),
                ("theta_ma1", mg.mean_params["theta"], zm.get("theta")), ("omega", par["omega"], z.get("omega")),
                ("alpha", par["alpha"], z.get("alpha")), ("beta", par["beta"], z.get("beta")), ("shape_nu", par["nu"], z.get("nu")),
                ("phi_plus_theta", mg.mean_params["phi"] + mg.mean_params["theta"], None),
                ("persistence_alpha_plus_beta", P, None), ("half_life_minutes", float(rp.half_life(P)), None),
                ("unconditional_q", par["omega"] / (1 - P), None), ("loglik_returns_scale", f["ll_returns"], None),
                ("aic_per_obs", f["aic_per_obs"], None), ("n_obs", f["n"], None)]
        pp = P_T2[(side, k)]
        ppP = pp["alpha"] + pp["beta"]
        printed = {"constant": np.nan, "phi_ar1": pp["phi"], "theta_ma1": pp["theta"], "omega": pp["omega"], "alpha": pp["alpha"],
                   "beta": pp["beta"], "shape_nu": pp["shape"], "phi_plus_theta": pp["phi"] + pp["theta"],
                   "persistence_alpha_plus_beta": ppP, "half_life_minutes": float(rp.half_life(ppP)),
                   "unconditional_q": pp["omega"] / (1 - ppP), "loglik_returns_scale": pp["loglik"], "aic_per_obs": pp["aic"],
                   "n_obs": P_T1[(side, k)]["n"]}
        for name, v, zz in vals:
            rows.append({"data": LABEL, "period": p, "index": k, "parameter": name, "ours": v,
                         "ours_z": np.nan if zz is None else float(zz), "paper_printed": printed[name],
                         "daily_variance_method": f["daily_method"]})
    return pd.DataFrame(rows)


def _boot_persistence_worker(args):
    eps, h, valid, block, n, seed = args
    rng = np.random.default_rng(seed)
    T = len(h)
    n_blocks = int(np.ceil(T / block))
    out = []
    for _ in range(n):
        st = rng.integers(0, T, n_blocks)
        idx = ((st[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        e, hh, v = eps[idx], h[idx], valid[idx]
        s = np.where(v, e**2 / hh[:, None], 0.0).sum(axis=0) / np.maximum(v.sum(axis=0), 1)
        x = np.where(v, e / np.sqrt(hh[:, None] * s), 0.0)
        f = fit_garch11(x.ravel(), weights=v.ravel().astype(float))
        out.append((f["alpha"], f["beta"], f["nu"], f["omega"]))
    return np.array(out)


HL_GRID = np.geomspace(4, 6000, 45)


def _profile_worker(args):
    x, w, params = args
    return rp.profile_half_life(x, params, HL_GRID, weights=w)


def _joint_lr_worker(args):
    """LR test of equal persistence in two samples (each with its own omega, alpha share and nu)."""
    xb, wb, pb, xa, wa, pa = args

    def joint(loghl):
        return -(rp.profile_half_life(xb, pb, [np.exp(loghl)], weights=wb)["loglik"].iloc[0]
                 + rp.profile_half_life(xa, pa, [np.exp(loghl)], weights=wa)["loglik"].iloc[0])

    hb, ha = float(rp.half_life(pb["alpha"] + pb["beta"])), float(rp.half_life(pa["alpha"] + pa["beta"]))
    lo, hi = np.log(min(hb, ha)) - 0.5, np.log(max(hb, ha)) + 0.5
    res = minimize_scalar(joint, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    lr = 2 * (pb["loglik"] + pa["loglik"] + res.fun)
    return {"common_half_life": float(np.exp(res.x)), "lr_stat": max(lr, 0.0), "p_value": float(stats.chi2.sf(max(lr, 0.0), 1))}


def persistence_inference(fits: dict, pool, n_boot: int, block: int = 5):
    keys = list(fits)
    chunks = 4
    tasks = []
    for i, key in enumerate(keys):
        f = fits[key]
        for c in range(chunks):
            tasks.append((f["filt"]["eps"], f["h"], f["valid_k"], block, int(np.ceil(n_boot / chunks)), 1000 * i + c))
    boots = list(pool.map(_boot_persistence_worker, tasks))
    prof_tasks = [(fits[k]["filt"]["x"].ravel(), fits[k]["valid_k"].ravel().astype(float), fits[k]["mg"].params) for k in keys]
    profs = list(pool.map(_profile_worker, prof_tasks))
    rows, draws = [], {}
    for i, key in enumerate(keys):
        p, k = key
        side = "after" if p == "after" else "before"
        f = fits[key]
        par = f["mg"].params
        b = np.vstack(boots[i * chunks : (i + 1) * chunks])[:n_boot]
        Pb = b[:, 0] + b[:, 1]
        draws[key] = Pb
        names, cov = rp.garch11_cov(f["filt"]["x"].ravel(), par, f["valid_k"].ravel().astype(float))
        dm = rp.persistence_delta(par, names, cov)
        prof = profs[i]
        ci = rp.profile_ci(prof, par["loglik"])
        pp = P_T2[(side, k)]
        rows.append({
            "data": LABEL, "period": p, "index": k, "alpha": par["alpha"], "beta": par["beta"],
            "persistence": dm["persistence"], "se_persistence_delta": dm["se_persistence"],
            "half_life_min": dm["half_life"], "se_half_life_delta": dm["se_half_life_delta"],
            "half_life_profile_ci95_lo": ci["lo"], "half_life_profile_ci95_hi": ci["hi"],
            "profile_ci_hits_grid_edge": bool(ci["lo_open"] or ci["hi_open"]),
            "boot_draws": len(Pb), "boot_block_days": block,
            "persistence_boot_ci95_lo": np.percentile(Pb, 2.5), "persistence_boot_ci95_hi": np.percentile(Pb, 97.5),
            "half_life_boot_ci95_lo": float(rp.half_life(np.percentile(Pb, 2.5))),
            "half_life_boot_ci95_hi": float(rp.half_life(np.percentile(Pb, 97.5))),
            "half_life_boot_median": float(rp.half_life(np.median(Pb))),
            "paper_printed_persistence": pp["alpha"] + pp["beta"],
            "paper_printed_half_life_min": float(rp.half_life(pp["alpha"] + pp["beta"])),
        })
    return pd.DataFrame(rows), draws


def persistence_change(fits: dict, draws: dict, pool) -> pd.DataFrame:
    tasks, meta = [], []
    for before in ("before_stated", "before_inferred"):
        for k in INDICES:
            fb, fa = fits[(before, k)], fits[("after", k)]
            tasks.append((fb["filt"]["x"].ravel(), fb["valid_k"].ravel().astype(float), fb["mg"].params,
                          fa["filt"]["x"].ravel(), fa["valid_k"].ravel().astype(float), fa["mg"].params))
            meta.append((before, k))
    lrs = list(pool.map(_joint_lr_worker, tasks))
    rows = []
    for (before, k), lr in zip(meta, lrs):
        Pb, Pa = draws[(before, k)], draws[("after", k)]
        n = min(len(Pb), len(Pa))
        d = Pa[:n] - Pb[:n]
        pb = fits[(before, k)]["mg"].params
        pa = fits[("after", k)]["mg"].params
        rows.append({"data": LABEL, "before": before, "index": k,
                     "persistence_before": pb["alpha"] + pb["beta"], "persistence_after": pa["alpha"] + pa["beta"],
                     "half_life_before": float(rp.half_life(pb["alpha"] + pb["beta"])),
                     "half_life_after": float(rp.half_life(pa["alpha"] + pa["beta"])),
                     "diff_persistence_after_minus_before": (pa["alpha"] + pa["beta"]) - (pb["alpha"] + pb["beta"]),
                     "diff_boot_ci95_lo": np.percentile(d, 2.5), "diff_boot_ci95_hi": np.percentile(d, 97.5),
                     "share_boot_draws_after_less_persistent": float(np.mean(d < 0)),
                     "lr_equal_persistence_stat": lr["lr_stat"], "lr_p_value": lr["p_value"],
                     "common_half_life_under_null": lr["common_half_life"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables 3 and 4: connectedness
# ---------------------------------------------------------------------------

FAMILIES = {"union": ("nifty", "banknifty"), "nifty": ("nifty",), "bn": ("banknifty",)}
SUBSAMPLES = ("all_expiry", "monthly", "monthly_first_hour", "monthly_mid_hour", "monthly_last_hour")
INPUTS = ("cond_vol", "intraday_vol", "returns", "abs_returns", "rs_vol")
PRIMARY = {"input": "cond_vol", "p": "2", "H": 10, "drop_first_minute": False}


def expiry_sets(cal: pd.DataFrame) -> dict:
    out = {}
    for k in INDICES:
        c = cal[cal["symbol"] == SYMBOL[k]]
        out[(k, "all")] = set(c["expiry"])
        out[(k, "monthly")] = set(c.loc[c["is_monthly"], "expiry"])
    return out


def subsample_days(days: pd.DatetimeIndex, fam: str, sub: str, ex: dict) -> np.ndarray:
    kind = "all" if sub == "all_expiry" else "monthly"
    s = set().union(*[ex[(k, kind)] for k in FAMILIES[fam]])
    return np.asarray(days.isin(list(s)))


def minute_slice(sub: str, M: int) -> slice:
    if sub.endswith("first_hour"):
        return slice(0, 60)
    if sub.endswith("mid_hour"):
        return slice(M // 2 - 30, M // 2 + 30)
    if sub.endswith("last_hour"):
        return slice(M - 60, M)
    return slice(0, M)


def stack_series(arrs, day_mask, msl, drop_first_minute=False):
    """Stack two (days x minutes) arrays into (rows x 2) with segment ids that break at days and at gaps."""
    cols = []
    for a in arrs:
        x = a[day_mask][:, msl]
        if drop_first_minute and (msl.start in (None, 0)):
            x = x[:, 1:]
        cols.append(x)
    nd, m = cols[0].shape
    Y = np.column_stack([c.ravel() for c in cols])
    seg = np.repeat(np.arange(nd), m)
    ok = np.all(np.isfinite(Y), axis=1)
    seg = seg * 1_000_000 + np.cumsum(~ok)
    return Y[ok], seg[ok], nd


def period_inputs(G, fits, p, rows):
    out = {}
    for name in INPUTS:
        arrs = []
        for k in INDICES:
            f = fits[(p, k)]
            if name == "cond_vol":
                a = np.where(f["valid"], f["cond_vol"], np.nan)
            elif name == "intraday_vol":
                a = np.where(f["valid"], f["intraday_vol"], np.nan)
            elif name == "returns":
                a = f["R"]
            elif name == "abs_returns":
                a = np.abs(f["R"])
            else:
                a = G[k]["rs_vol"][rows]
            arrs.append(a)
        out[name] = arrs
    return out


def conn_record(Y, seg, p, H):
    c = rp.connectedness_segments(Y, seg, p, H)
    t = c.table
    return {"nifty_own": t[0, 0], "nifty_from_bn": t[0, 1], "bn_from_nifty": t[1, 0], "bn_own": t[1, 1],
            "paper_scale_total": c.paper_scale_total, "dy_total": c.total, "net_nifty_to_minus_from": c.net[0]}


def _conn_boot_worker(args):
    Y, seg, p, H, n, seed = args
    return rp.segment_bootstrap_connectedness(Y, seg, p, H, n_boot=n, seed=seed)


def connectedness_tables(G, fits, periods, ex, pool, n_boot, full=True):
    M = len(G["minutes"])
    grid_rows, prim_rows, boot_tasks, boot_meta = [], [], [], []
    for p, days in periods.items():
        rows = period_rows(G, days)
        d = G["days"][rows]
        inputs = period_inputs(G, fits, p, rows)
        subs = [("all_days", "all_days")] + [(fam, sub) for fam in FAMILIES for sub in SUBSAMPLES]
        for fam, sub in subs:
            day_mask = np.ones(len(d), bool) if fam == "all_days" else subsample_days(d, fam, sub, ex)
            msl = minute_slice(sub, M)
            n_days = int(day_mask.sum())
            if n_days == 0:
                continue
            side = "after" if p == "after" else "before"
            printed = P_T34.get((side, fam, sub))
            base = {"data": LABEL, "trim": G["trim"], "period": p, "family": fam, "subsample": sub, "n_days": n_days,
                    "days": ";".join(d[day_mask].strftime("%Y-%m-%d")) if fam != "all_days" else ""}
            configs = []
            for inp in (INPUTS if full else ("cond_vol",)):
                for pl in (("1", "2", "5", "aic") if full else ("2",)):
                    for H in ((10, 30) if full else (10,)):
                        configs.append((inp, pl, H, False))
            configs.append(("cond_vol", "2", 10, True))
            aic_cache = {}
            for inp, pl, H, dfm in configs:
                Y, seg, _ = stack_series(inputs[inp], day_mask, msl, dfm)
                if pl == "aic":
                    if (inp, dfm) not in aic_cache:
                        aic_cache[(inp, dfm)] = rp.var_lag_aic(Y, seg, 10)
                    pv = aic_cache[(inp, dfm)]
                else:
                    pv = int(pl)
                rec = dict(base, input=inp, p_rule=pl, p_used=pv, H=H, drop_first_minute=dfm, n_obs=len(Y))
                rec.update(conn_record(Y, seg, pv, H))
                if printed is not None:
                    rec["paper_printed_total"] = printed[1]
                    rec["paper_printed_nifty_from_bn"] = printed[0][0][1]
                    rec["paper_printed_bn_from_nifty"] = printed[0][1][0]
                grid_rows.append(rec)
                is_primary = (inp, pl, H, dfm) == (PRIMARY["input"], PRIMARY["p"], PRIMARY["H"], PRIMARY["drop_first_minute"])
                if is_primary:
                    prim_rows.append(rec)
                    if full:
                        boot_tasks.append((Y, seg, pv, H, n_boot, 7919 + len(boot_tasks)))
                        boot_meta.append((p, fam, sub))
    grid = pd.DataFrame(grid_rows)
    prim = pd.DataFrame(prim_rows)
    draws = {}
    if full:
        res = list(pool.map(_conn_boot_worker, boot_tasks))
        for meta, r in zip(boot_meta, res):
            draws[meta] = r
        prim["boot_ci95_lo"] = [np.nanpercentile(draws[(r.period, r.family, r.subsample)], 2.5) for r in prim.itertuples()]
        prim["boot_ci95_hi"] = [np.nanpercentile(draws[(r.period, r.family, r.subsample)], 97.5) for r in prim.itertuples()]
    return grid, prim, draws


def connectedness_changes(prim: pd.DataFrame, draws: dict) -> pd.DataFrame:
    rows = []
    look = prim.set_index(["period", "family", "subsample"])["paper_scale_total"]
    dayset = prim.set_index(["period", "family", "subsample"])["days"]
    for before in ("before_stated", "before_inferred"):
        pairs = [(("all_days", "all_days"), ("all_days", "all_days"), "all days, all minutes")]
        for sub in SUBSAMPLES:
            for fa in ("nifty", "bn"):
                pairs.append((("union", sub), (fa, sub), f"paper comparison: Table 3 vs Table 4 panel {'I' if fa == 'nifty' else 'II'}"))
                pairs.append(((fa, sub), (fa, sub), "like for like: same index's expiry days before and after"))
        seen = set()
        for (fb, sb), (fa, sa), what in pairs:
            kb, ka = (before, fb, sb), ("after", fa, sa)
            if kb not in draws or ka not in draws or (kb, ka) in seen:
                continue
            if fb != "union" and fb != "all_days" and dayset.get((before, "union", sb)) == dayset.get(kb):
                continue  # same days as the paper comparison row (every family coincides when both indices expire together)
            seen.add((kb, ka))
            n = min(len(draws[kb]), len(draws[ka]))
            d = draws[ka][:n] - draws[kb][:n]
            pb = P_T34.get(("before", fb, sb))
            pa = P_T34.get(("after", fa, sa))
            rows.append({"data": LABEL, "before": before, "before_family": fb, "after_family": fa, "subsample": sb, "comparison": what,
                         "total_before": look[kb], "total_after": look[ka], "diff_after_minus_before": look[ka] - look[kb],
                         "diff_boot_ci95_lo": np.nanpercentile(d, 2.5), "diff_boot_ci95_hi": np.nanpercentile(d, 97.5),
                         "paper_printed_diff": (pa[1] - pb[1]) if (pa and pb) else np.nan})
    return pd.DataFrame(rows)


def grid_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """Range of the paper-scale total over the sensitivity grid, and the configuration closest to the printed value."""
    rows = []
    g0 = grid[~grid["drop_first_minute"].astype(bool)]
    for key, g in g0.groupby(["period", "family", "subsample"], sort=False):
        prim = g[(g["input"] == PRIMARY["input"]) & (g["p_rule"] == PRIMARY["p"]) & (g["H"] == PRIMARY["H"])]
        dfm = grid[grid["drop_first_minute"].astype(bool) & (grid["period"] == key[0]) & (grid["family"] == key[1]) & (grid["subsample"] == key[2])]
        rec = {"data": LABEL, "period": key[0], "family": key[1], "subsample": key[2], "n_configs": len(g),
               "primary_total": prim["paper_scale_total"].iloc[0] if len(prim) else np.nan,
               "primary_total_without_first_minute": dfm["paper_scale_total"].iloc[0] if len(dfm) else np.nan,
               "grid_min": g["paper_scale_total"].min(), "grid_max": g["paper_scale_total"].max(),
               "grid_max_cross_share": g[["nifty_from_bn", "bn_from_nifty"]].max().max()}
        for inp in INPUTS:
            gi = g[g["input"] == inp]["paper_scale_total"]
            rec[f"min_{inp}"], rec[f"max_{inp}"] = gi.min(), gi.max()
        if "paper_printed_total" in g and g["paper_printed_total"].notna().any():
            t = g["paper_printed_total"].iloc[0]
            j = (g["paper_scale_total"] - t).abs().idxmin()
            rec.update({"paper_printed_total": t, "closest_total": g.loc[j, "paper_scale_total"],
                        "closest_config": f"{g.loc[j, 'input']}, p={g.loc[j, 'p_used']} ({g.loc[j, 'p_rule']}), H={g.loc[j, 'H']}",
                        "printed_inside_grid_range": bool(rec["grid_min"] <= t <= rec["grid_max"])})
        rows.append(rec)
    return pd.DataFrame(rows)


def rolling_connectedness(G, fits, periods, window=360, step=30):
    rows = []
    for p, days in periods.items():
        rows_idx = period_rows(G, days)
        arrs = [np.where(fits[(p, k)]["valid"], fits[(p, k)]["cond_vol"], np.nan) for k in INDICES]
        Y = np.column_stack([a.ravel() for a in arrs])
        ts = (G["days"][rows_idx].values[:, None] + pd.to_timedelta([f"{m}:00" for m in G["minutes"]]).values[None, :]).ravel()
        ok = np.all(np.isfinite(Y), axis=1)
        Y, ts = Y[ok], ts[ok]
        dayid = np.repeat(np.arange(len(rows_idx)), len(G["minutes"]))[ok]
        for e in range(window, len(Y) + 1, step):
            c = rp.connectedness_segments(Y[e - window : e], dayid[e - window : e], 2, 10)
            rows.append({"data": LABEL, "period": p, "end_time": pd.Timestamp(ts[e - 1]), "paper_scale_total": c.paper_scale_total,
                         "nifty_from_bn": c.table[0, 1], "bn_from_nifty": c.table[1, 0], "net_nifty": c.net[0]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables 5 and 6: before/after tests
# ---------------------------------------------------------------------------


def day_masks(days: pd.DatetimeIndex, ex: dict) -> dict:
    wd = days.weekday
    return {"all": np.ones(len(days), bool), "wed": np.asarray(wd == 2), "thu": np.asarray(wd == 3),
            "nifty_expiry": np.asarray(days.isin(list(ex[("nifty", "all")]))),
            "bn_expiry": np.asarray(days.isin(list(ex[("banknifty", "all")])))}


def week_rel_test(vb: pd.Series, tb: pd.Series, va: pd.Series, ta: pd.Series) -> dict:
    rb = rp.week_relative(vb, tb)["rel_log"].to_numpy()
    ra = rp.week_relative(va, ta)["rel_log"].to_numpy()
    out = {"weekrel_n_before": len(rb), "weekrel_n_after": len(ra)}
    if len(rb) >= 3 and len(ra) >= 3:
        out.update({"weekrel_mean_before": rb.mean(), "weekrel_mean_after": ra.mean(), "weekrel_diff": ra.mean() - rb.mean(),
                    "weekrel_p_mwu": mannwhitneyu(rb, ra, alternative="two-sided").pvalue,
                    "weekrel_p_welch": stats.ttest_ind(ra, rb, equal_var=False).pvalue})
    return out


def table6(G, periods, ex, n_boot, full=True) -> pd.DataFrame:
    rows = []
    M = len(G["minutes"])
    for before in ("before_stated", "before_inferred"):
        rb, ra = period_rows(G, periods[before]), period_rows(G, periods["after"])
        db, da = G["days"][rb], G["days"][ra]
        mb, ma = day_masks(db, ex), day_masks(da, ex)
        for k in INDICES:
            Vb, Va = G[k]["rs_vol"][rb], G[k]["rs_vol"][ra]
            dayb = pd.Series(np.nanmedian(Vb, axis=1), index=db)
            daya = pd.Series(np.nanmedian(Va, axis=1), index=da)
            for sub in mb:
                b, a = Vb[mb[sub]], Va[ma[sub]]
                if len(b) == 0 or len(a) == 0:
                    continue
                mw = minute_level_mwu(b, a)
                rec = {"data": LABEL, "trim": G["trim"], "before": before, "index": k, "subsample": sub,
                       "n_days_before": len(b), "n_days_after": len(a), "n_min_before": mw["n_before"], "n_min_after": mw["n_after"],
                       "median_before": np.nanmedian(b), "median_after": np.nanmedian(a),
                       "pct_change_median": 100 * (np.nanmedian(a) / np.nanmedian(b) - 1),
                       "minute_mwu_U_before": mw["stat"], "minute_mwu_prob_after_gt_before": 1 - mw["stat"] / mw["max_U"],
                       "minute_mwu_p": mw["p"]}
                if full:
                    dl = day_level_mwu(b, a)
                    hac = hac_mean_difference(b, a, maxlags=M, log=True)
                    bs = day_cluster_bootstrap(b, a, n_boot=n_boot, seed=11)
                    rec.update({"day_mwu_p": dl["p"], "hac_diff_mean_log": hac["diff"], "hac_se": hac["se"], "hac_p": hac["p"],
                                "boot_median_diff": bs["diff"], "boot_ci95_lo": bs["ci_low"], "boot_ci95_hi": bs["ci_high"], "boot_p": bs["p"]})
                    if sub != "all":
                        rec.update(week_rel_test(dayb, pd.Series(mb[sub], index=db), daya, pd.Series(ma[sub], index=da)))
                pk = P_T6.get((sub, k))
                if pk:
                    rec.update({"paper_printed_n_before": pk[0], "paper_printed_median_before": pk[1], "paper_printed_n_after": pk[2],
                                "paper_printed_median_after": pk[3], "paper_printed_p": pk[5]})
                rows.append(rec)
    return pd.DataFrame(rows)


VOLUME_MEASURES = {
    "index_constituent_volume_shares": "official index file: shares traded in the index's constituents",
    "index_constituent_turnover_cr": "official index file: value traded in the constituents (Rs crore)",
    "options_premium_turnover_cr": "bhavcopy: index options premium turnover (Rs crore)",
    "options_notional_turnover_cr": "bhavcopy: index options notional turnover (Rs crore)",
    "options_quantity_units": "bhavcopy: index options contracts x lot size",
    "options_contracts": "bhavcopy: index options contracts (NOT comparable across a lot-size change)",
    "futures_notional_turnover_cr": "bhavcopy: index futures notional turnover (Rs crore)",
    "futures_quantity_units": "bhavcopy: index futures contracts x lot size",
}


def daily_volume_table(off: pd.DataFrame, fo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in INDICES:
        o = off[off["key"] == k].set_index("date")
        f = fo[fo["symbol"] == SYMBOL[k]].set_index("date")
        d = pd.DataFrame({
            "index_constituent_volume_shares": o["volume"], "index_constituent_turnover_cr": o["turnover_cr"],
        }).join(pd.DataFrame({
            "options_premium_turnover_cr": f["opt_premium_cr"], "options_notional_turnover_cr": f["opt_notional_cr"],
            "options_quantity_units": f["opt_qty"], "options_contracts": f["opt_contracts"],
            "futures_notional_turnover_cr": f["fut_notional_cr"], "futures_quantity_units": f["fut_qty"], "lot_size": f["lot_size"],
        }), how="outer")
        d["index"] = k
        d.index.name = "date"
        rows.append(d.reset_index())
    return pd.concat(rows, ignore_index=True)


def table5(vol: pd.DataFrame, periods, ex, n_boot) -> pd.DataFrame:
    rows = []
    for before in ("before_stated", "before_inferred"):
        db, da = periods[before], periods["after"]
        mb, ma = day_masks(db, ex), day_masks(da, ex)
        for k in INDICES:
            v = vol[vol["index"] == k].set_index("date")
            lot_b = v["lot_size"].reindex(db)
            lot_a = v["lot_size"].reindex(da)
            for meas, desc in VOLUME_MEASURES.items():
                sb, sa = v[meas].reindex(db), v[meas].reindex(da)
                for sub in mb:
                    b = sb[mb[sub]].to_numpy(float)
                    a = sa[ma[sub]].to_numpy(float)
                    b, a = b[np.isfinite(b) & (b > 0)], a[np.isfinite(a) & (a > 0)]
                    if len(b) < 3 or len(a) < 3:
                        continue
                    hac = hac_mean_difference(b.reshape(-1, 1), a.reshape(-1, 1), maxlags=5, log=True)
                    bs = day_cluster_bootstrap(b.reshape(-1, 1), a.reshape(-1, 1), n_boot=n_boot, seed=13)
                    rec = {"data": LABEL + "; day-level F&O and index volume, NOT per-minute volume", "before": before, "index": k,
                           "measure": meas, "measure_description": desc, "subsample": sub, "n_days_before": len(b), "n_days_after": len(a),
                           "median_before": np.median(b), "median_after": np.median(a), "pct_change_median": 100 * (np.median(a) / np.median(b) - 1),
                           "day_mwu_p": mannwhitneyu(b, a, alternative="two-sided").pvalue,
                           "hac_diff_mean_log": hac["diff"], "hac_p": hac["p"], "boot_ci95_lo": bs["ci_low"], "boot_ci95_hi": bs["ci_high"], "boot_p": bs["p"],
                           "lot_sizes_before": ";".join(str(int(x)) for x in sorted(lot_b.dropna().unique())),
                           "lot_sizes_after": ";".join(str(int(x)) for x in sorted(lot_a.dropna().unique()))}
                    if sub != "all":
                        rec.update(week_rel_test(sb.dropna()[sb.dropna() > 0], pd.Series(mb[sub], index=db), sa.dropna()[sa.dropna() > 0], pd.Series(ma[sub], index=da)))
                    pk = P_T5.get((sub, k))
                    if pk:
                        rec.update({"paper_printed_per_minute_median_before": pk[1], "paper_printed_per_minute_median_after": pk[3],
                                    "paper_printed_pct_change_from_medians": 100 * (pk[3] / pk[1] - 1), "paper_printed_p": pk[5]})
                        if meas == "index_constituent_volume_shares":
                            rec["ours_median_daily_volume_over_360_before"] = np.median(b) / 360
                            rec["ours_median_daily_volume_over_360_after"] = np.median(a) / 360
                    rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Checks on the printed numbers themselves
# ---------------------------------------------------------------------------


def printed_checks() -> pd.DataFrame:
    rows = []
    for (side, k), pp in P_T2.items():
        n_impl = -2 * pp["loglik"] / pp["aic"]
        rows.append({"check": "Table 2: N implied by printed LL and AIC (N = -2 LL / AIC)", "item": f"{side} {k}",
                     "value": n_impl, "compare_with": P_T1[(side, k)]["n"], "days_of_360_minutes": n_impl / 360,
                     "consistent": abs(n_impl - P_T1[(side, k)]["n"]) < 100})
    for (side, fam, sub), (m, tot) in P_T34.items():
        for i, name in enumerate(("Nifty 50 row", "Bank Nifty row")):
            s = sum(m[i])
            rows.append({"check": "Tables 3 and 4: printed row sums to 100", "item": f"{side} {fam} {sub} {name}", "value": s,
                         "compare_with": 100.0, "consistent": abs(s - 100) < 0.02})
        rows.append({"check": "Tables 3 and 4: printed total equals sum of the two cross shares", "item": f"{side} {fam} {sub}",
                     "value": m[0][1] + m[1][0], "compare_with": tot, "consistent": abs(m[0][1] + m[1][0] - tot) < 0.02})
    for key, (n1, m1, n2, m2, w, pv) in P_T5.items():
        pct = 100 * (m2 / m1 - 1)
        rows.append({"check": "Table 5: % change implied by printed medians vs % quoted in the text", "item": f"{key[0]} {key[1]}",
                     "value": pct, "compare_with": P_T5_TEXT_PCT[key], "consistent": abs(pct - P_T5_TEXT_PCT[key]) < 1.0})
    for tab, dct in (("Table 5", P_T5), ("Table 6", P_T6)):
        for key, (n1, m1, n2, m2, w, pv) in dct.items():
            for reading, u in (("W read as U of the before sample", w), ("W read as rank sum of the before sample", w - n1 * (n1 + 1) / 2)):
                auc = u / (n1 * n2)
                z = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                p = 2 * stats.norm.sf(abs(z))
                in_range = 0 <= auc <= 1
                p_ok = (p < 0.0005) if pv < 0.0005 else (abs(np.log10(max(p, 1e-300)) - np.log10(pv)) < 0.5)
                direction_ok = (m1 == m2) or ((auc < 0.5) == (m1 < m2))
                rows.append({"check": f"{tab}: printed statistic vs printed p-value and medians ({reading})", "item": f"{key[0]} {key[1]}",
                             "value": p, "compare_with": pv, "implied_prob_before_gt_after": auc, "statistic_in_possible_range": in_range,
                             "p_matches": p_ok, "direction_matches_medians": direction_ok,
                             "consistent": bool(in_range and p_ok)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {"before_inferred": "#2a78d6", "before_stated": "#eb6834", "after": "#1baf7a"}
STYLES = {"before_inferred": "-", "before_stated": "--", "after": ":"}
NAMES = {"before_inferred": "before, inferred window", "before_stated": "before, stated window", "after": "after (Mar to May 2024)"}
INDEX_NAMES = {"nifty": "Nifty 50", "banknifty": "Bank Nifty"}


def plot_diurnal(G, fits, periods, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = pd.to_datetime(["2000-01-01 " + m for m in G["minutes"]])
    for ax, k in zip(axes, INDICES):
        for p in periods:
            s = fits[(p, k)]["mg"].s
            xk = x[fits[(p, k)]["keep"]]
            sm = pd.Series(s[1:]).rolling(15, center=True, min_periods=8).mean().to_numpy()
            ax.plot(xk[1:], np.sqrt(sm / np.median(sm)), STYLES[p], color=COLORS[p], lw=1.8, label=NAMES[p])
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{INDEX_NAMES[k]}: intraday volatility pattern (mcsGARCH diurnal factor)", fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("volatility relative to the day's median minute")
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f"{LABEL}. Session {G['trim']}; first minute (overnight gap) not shown; 15-minute centred average for display.",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rolling(roll: pd.DataFrame, path):
    ps = list(dict.fromkeys(roll["period"]))
    fig, axes = plt.subplots(len(ps), 2, figsize=(12, 2.6 * len(ps)), squeeze=False, sharey="col")
    for i, p in enumerate(ps):
        r = roll[roll["period"] == p]
        xi = np.arange(len(r))
        for j, (col, lab) in enumerate((("paper_scale_total", "total (paper scale)"), ("net_nifty", "net, Nifty 50 (to minus from)"))):
            ax = axes[i, j]
            ax.plot(xi, r[col].to_numpy(), color=COLORS[p], lw=0.8)
            ax.set_title(f"{NAMES[p]}: rolling {lab}", fontsize=9)
            ticks = np.linspace(0, len(r) - 1, 5).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([r["end_time"].iloc[t].strftime("%d %b %y") for t in ticks], fontsize=7)
            ax.grid(True, color="#e6e6e6", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if col == "net_nifty":
                ax.axhline(0, color="#888888", lw=0.6)
    fig.text(0.01, 0.005, f"{LABEL}. VAR(2), H = 10, input mcsGARCH conditional volatility, 360-minute window every 30 minutes (trading time).",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fmt(df: pd.DataFrame, cols=None, n=None) -> str:
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    if n:
        d = d.head(n)
    return d.to_string(index=False, float_format=lambda v: f"{v:.6g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trim", default="360_drop_last15", choices=list(rp.SESSION_RULES))
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap draws (for development)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    n_pboot, n_cboot, n_dboot = (40, 40, 200) if a.quick else (400, 300, 2000)
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(LOG)
    t0 = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)

    print(f"Replication of Chhimwal, Pandey and Pandey (2025) on {LABEL}.")
    print("Numbers labelled paper_printed are the paper's printed values, typed in as comparison targets.")
    print(f"Primary session rule: {a.trim} {rp.SESSION_RULES[a.trim]}; bootstrap draws: persistence {n_pboot}, connectedness {n_cboot}, medians {n_dboot}.")

    # ---------------- data
    bars = {k: rp.load_minute_bars(RAW / f"{k}_1min.csv", "2017-01-01", DATA_RANGE[1]) for k in INDICES}
    off = load_official()
    contracts, fo, n_files, n_skipped = load_bhavcopy(DATA_RANGE[0], "2024-07-05")
    cal = rp.expiry_calendar(contracts)
    cal = cal[(cal["expiry"] >= DATA_RANGE[0]) & (cal["expiry"] <= DATA_RANGE[1])]
    cal.assign(data="NSE F&O bhavcopy (official)").to_csv(OUT / "expiry_calendar.csv", index=False)
    ex = expiry_sets(cal)
    note(f"data loaded {time.time() - t0:.0f}s")

    # ---------------- grids, windows
    G = build_grids(bars, a.trim)
    search = fingerprint_search(G) if a.trim == "360_drop_last15" else fingerprint_search(build_grids(bars, "360_drop_last15"))
    best = search.iloc[0]
    sb, sa = pd.Timestamp(STATED_BEFORE[0]), pd.Timestamp(STATED_BEFORE[1])
    periods = {
        "before_stated": G["days"][(G["days"] >= sb) & (G["days"] <= sa)],
        "before_inferred": G["days"][(G["days"] >= pd.Timestamp(best["start"])) & (G["days"] <= pd.Timestamp(best["end"]))],
        "after": G["days"][(G["days"] >= AFTER[0]) & (G["days"] <= AFTER[1])],
    }
    search.assign(data=LABEL).to_csv(OUT / "fingerprint_search.csv", index=False)

    header("1. Data and validation")
    print(f"Minute bars: {RAW}/nifty_1min.csv and banknifty_1min.csv (third-party, provenance in minute_bars_source.json).")
    print(f"Official NSE daily index values: {len(off)} index-days loaded. Bhavcopy: {n_files} files read, {n_skipped} in the newer layout skipped.")
    v = validate(bars, off, pd.Timestamp(DATA_RANGE[0]), pd.Timestamp(DATA_RANGE[1]))
    v.to_csv(OUT / "validation_by_day.csv", index=False, float_format="%.8g")
    vs = validation_summary(v, periods)
    vs.to_csv(OUT / "validation.csv", index=False)
    print(fmt(vs, ["span", "index", "official_days", "days_compared", "open_median_abs_bps", "high_median_abs_bps", "low_median_abs_bps",
                   "close_median_abs_bps", "high_max_abs_bps", "low_max_abs_bps", "days_open_high_low_off_by_more_than_5bps"]))

    header("2. Expiry calendar from the bhavcopy (an expiry = last trading day of a contract; monthly = futures expiry)")
    for k in INDICES:
        c = cal[(cal["symbol"] == SYMBOL[k])]
        for p, days in periods.items():
            cc = c[(c["expiry"] >= days.min()) & (c["expiry"] <= days.max())]
            print(f"{k:9s} {p:16s} expiries {len(cc):2d} (monthly {int(cc['is_monthly'].sum())}): "
                  + ", ".join(f"{d:%d %b %y} {d:%a}{'*' if m else ''}" for d, m in zip(cc['expiry'], cc['is_monthly'])))
    print("(* = monthly)")

    header("3. Windows and day counts (paper: 64 days before, 59 after; 13 and 12 Wednesdays, 13 and 13 Thursdays)")
    print(f"Fingerprint search over all {len(search)} windows of 64 consecutive trading days from {SEARCH_RANGE[0]} to {SEARCH_RANGE[1]},")
    print("scored on SD, minimum, maximum and excess kurtosis of both indices against Table 1's 'before' column. Best windows:")
    print(fmt(search, ["rank", "start", "end", "score", "nifty_sd", "nifty_min", "nifty_max", "nifty_excess_kurtosis",
                       "banknifty_sd", "banknifty_min", "banknifty_max", "banknifty_excess_kurtosis"], n=8))
    in_stated = search[(pd.to_datetime(search["start"]) >= sb)]
    if len(in_stated):
        print(f"Best window that starts on or after {STATED_BEFORE[0]}: rank {int(in_stated.iloc[0]['rank'])}, "
              f"{in_stated.iloc[0]['start']} to {in_stated.iloc[0]['end']}, score {in_stated.iloc[0]['score']:.3f} (best overall {best['score']:.3f}).")
    dc = []
    for p, days in periods.items():
        side = "after" if p == "after" else "before"
        rr = period_rows(G, days)
        dc.append({"data": LABEL, "period": p, "start": days.min().date(), "end": days.max().date(), "trading_days": len(days),
                   "paper_days": PAPER_DAYS[side], "wednesdays": int((days.weekday == 2).sum()), "paper_wednesdays": PAPER_WEEKDAY_DAYS[(side, "wed")],
                   "thursdays": int((days.weekday == 3).sum()), "paper_thursdays": PAPER_WEEKDAY_DAYS[(side, "thu")],
                   "nifty_expiry_days": int(days.isin(list(ex[("nifty", "all")])).sum()), "bn_expiry_days": int(days.isin(list(ex[("banknifty", "all")])).sum()),
                   "nifty_monthly_expiries": int(days.isin(list(ex[("nifty", "monthly")])).sum()),
                   "bn_monthly_expiries": int(days.isin(list(ex[("banknifty", "monthly")])).sum()),
                   "n_returns_nifty": int(np.isfinite(window_returns(G, "nifty", rr)).sum()),
                   "n_returns_banknifty": int(np.isfinite(window_returns(G, "banknifty", rr)).sum()),
                   "paper_n_returns": P_T1[(side, "nifty")]["n"]})
    dc = pd.DataFrame(dc)
    dc.to_csv(OUT / "day_counts.csv", index=False)
    print(fmt(dc.drop(columns="data")))

    header("4. Which 360 minutes? Table 1 statistics under each session rule and return definition")
    sri = []
    for trim in rp.SESSION_RULES:
        for overnight in (True, False):
            Gt = G if (trim == a.trim and overnight) else build_grids(bars, trim, overnight)
            for p, days in periods.items():
                side = "after" if p == "after" else "before"
                rr = period_rows(Gt, days)
                for k in INDICES:
                    st = describe(window_returns(Gt, k, rr).ravel(), full=False)
                    sri.append({"data": LABEL, "trim": trim, "overnight_in_first_minute": overnight, "period": p, "index": k,
                                **{x: st[x] for x in ("n", "sd", "min", "max", "skew", "excess_kurtosis")},
                                "fingerprint_score": fingerprint_score(st, P_T1[(side, k)]),
                                **{f"paper_printed_{x}": P_T1[(side, k)][x] for x in ("n", "sd", "min", "max", "skew", "excess_kurtosis")}})
    sri = pd.DataFrame(sri)
    sri.to_csv(OUT / "session_rule_inference.csv", index=False)
    print("Fingerprint score (lower = closer to the paper's Table 1; sum over the four statistics) by rule:")
    print(sri.pivot_table(index=["trim", "overnight_in_first_minute"], columns=["period", "index"], values="fingerprint_score").round(3).to_string())

    # ---------------- Table 1
    header("Table 1. Descriptive statistics of 1-minute log returns")
    t1 = table1(G, periods, full=True)
    t1.to_csv(OUT / "table1_descriptive.csv", index=False)
    print(t1.pivot_table(index="statistic", columns=["period", "index"], values="ours", sort=False).to_string(float_format=lambda v: f"{v:.6g}"))
    print("paper (printed):")
    print(t1[t1["period"] != "before_stated"].pivot_table(index="statistic", columns=["period", "index"], values="paper_printed", sort=False).to_string(float_format=lambda v: f"{v:.6g}"))
    note(f"table 1 done {time.time() - t0:.0f}s")

    # ---------------- Table 2
    header("Table 2. ARMA(1,1)-mcsGARCH(1,1)-t on 1-minute returns (daily variance from AR(1)-EGARCH(1,1)-t)")
    fits = {(p, k): fit_period(G, k, period_rows(G, days)) for p, days in periods.items() for k in INDICES}
    t2 = table2_rows(fits, full=True)
    t2.to_csv(OUT / "table2_mcsgarch.csv", index=False)
    print(t2.pivot_table(index="parameter", columns=["period", "index"], values="ours", sort=False).to_string(float_format=lambda v: f"{v:.6g}"))
    print("our z statistics (GARCH part: numerical Hessian; ARMA part: robust sandwich):")
    print(t2.pivot_table(index="parameter", columns=["period", "index"], values="ours_z", sort=False).dropna(how="all").round(2).to_string())
    print("paper (printed):")
    print(t2[t2["period"] != "before_stated"].pivot_table(index="parameter", columns=["period", "index"], values="paper_printed", sort=False).to_string(float_format=lambda v: f"{v:.6g}"))

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pers, pdraws = persistence_inference(fits, pool, n_pboot)
        pers.to_csv(OUT / "table2_persistence.csv", index=False)
        print("\nIntraday persistence (alpha + beta) and half-life in minutes, with uncertainty:")
        print(fmt(pers, ["period", "index", "persistence", "se_persistence_delta", "half_life_min", "se_half_life_delta",
                         "half_life_profile_ci95_lo", "half_life_profile_ci95_hi", "half_life_boot_ci95_lo", "half_life_boot_ci95_hi",
                         "paper_printed_half_life_min"]))
        pchg = persistence_change(fits, pdraws, pool)
        pchg.to_csv(OUT / "table2_persistence_change.csv", index=False)
        print("\nDid persistence fall after the change? (bootstrap of the difference; likelihood-ratio test of equal persistence)")
        print(fmt(pchg, ["before", "index", "half_life_before", "half_life_after", "diff_persistence_after_minus_before",
                         "diff_boot_ci95_lo", "diff_boot_ci95_hi", "lr_equal_persistence_stat", "lr_p_value"]))
        note(f"table 2 done {time.time() - t0:.0f}s")

        # window sensitivity of Table 2
        ws = []
        for _, w in search.head(6).iterrows():
            days = G["days"][(G["days"] >= pd.Timestamp(w["start"])) & (G["days"] <= pd.Timestamp(w["end"]))]
            for k in INDICES:
                f = fit_period(G, k, period_rows(G, days))
                par = f["mg"].params
                P = par["alpha"] + par["beta"]
                ws.append({"data": LABEL, "fingerprint_rank": int(w["rank"]), "start": w["start"], "end": w["end"], "index": k,
                           "omega": par["omega"], "alpha": par["alpha"], "beta": par["beta"], "shape_nu": par["nu"], "persistence": P,
                           "half_life_min": float(rp.half_life(P)), "loglik_returns_scale": f["ll_returns"],
                           "paper_printed_half_life_min": float(rp.half_life(P_T2[("before", k)]["alpha"] + P_T2[("before", k)]["beta"]))})
        ws = pd.DataFrame(ws)
        ws.to_csv(OUT / "table2_window_sensitivity.csv", index=False)
        print("\nHow much does the 'before' half-life move if the 64-day window shifts by a few days? (top fingerprint windows)")
        print(fmt(ws, ["fingerprint_rank", "start", "end", "index", "alpha", "beta", "shape_nu", "half_life_min", "paper_printed_half_life_min"]))

        # ---------------- Tables 3 and 4
        header("Tables 3 and 4. Diebold-Yilmaz connectedness on sub-samples")
        grid, prim, cdraws = connectedness_tables(G, fits, periods, ex, pool, n_cboot, full=True)
        grid.drop(columns="days").to_csv(OUT / "tables34_connectedness_grid.csv", index=False, float_format="%.6g")
        prim.to_csv(OUT / "tables34_connectedness.csv", index=False)
        print(f"Primary configuration: input {PRIMARY['input']}, VAR({PRIMARY['p']}), horizon {PRIMARY['H']}; VAR lags never cross a day or hour boundary.")
        print("Totals are on the paper's scale (sum of the two cross shares); the standard DY index is half of this.")
        print("Intervals: bootstrap resampling whole days (or whole hours for the one-hour sub-samples).")
        show = prim[((prim["period"] != "before_inferred") | prim["family"].isin(["all_days", "union"]))
                    & ((prim["period"] != "after") | (prim["family"] != "union"))]
        print(fmt(show, ["period", "family", "subsample", "n_days", "n_obs", "nifty_from_bn", "bn_from_nifty", "paper_scale_total",
                         "boot_ci95_lo", "boot_ci95_hi", "paper_printed_nifty_from_bn", "paper_printed_bn_from_nifty", "paper_printed_total"]))
        print("(before_inferred: Nifty and Bank Nifty expired on the same days, so the union, Nifty and Bank Nifty rows coincide.)")
        chg = connectedness_changes(prim, cdraws)
        chg.to_csv(OUT / "tables34_changes.csv", index=False)
        print("\nChange in total connectedness, after minus before (paper scale), with whole-day (or whole-hour) bootstrap 95% intervals:")
        print(fmt(chg, ["before", "before_family", "after_family", "subsample", "total_before", "total_after", "diff_after_minus_before",
                        "diff_boot_ci95_lo", "diff_boot_ci95_hi", "paper_printed_diff"]))
        gs = grid_summary(grid)
        gs.to_csv(OUT / "tables34_grid_summary.csv", index=False)
        print(f"\nSensitivity grid ({len(INPUTS)} input series x VAR lags 1, 2, 5, AIC x horizons 10, 30): range of the paper-scale total,")
        print("the configuration closest to the printed value, and the primary total without the first (overnight) minute:")
        print(fmt(gs[gs["paper_printed_total"].notna()] if "paper_printed_total" in gs else gs,
                  ["period", "family", "subsample", "grid_min", "grid_max", "grid_max_cross_share", "primary_total",
                   "primary_total_without_first_minute", "paper_printed_total", "closest_total", "closest_config"]))
        note(f"tables 3 and 4 done {time.time() - t0:.0f}s")

    roll = rolling_connectedness(G, fits, {p: periods[p] for p in ("before_inferred", "before_stated", "after")})
    roll.to_csv(OUT / "fig1_rolling_connectedness.csv", index=False, float_format="%.6g")
    plot_rolling(roll, OUT / "fig_rolling_connectedness.png")
    plot_diurnal(G, fits, periods, OUT / "fig_diurnal.png")

    # ---------------- Table 5
    header("Table 5 (replacement). Volume: the paper's per-minute volume is not available; day-level measures instead")
    vol = daily_volume_table(off, fo)
    t5 = table5(vol, periods, ex, n_dboot)
    t5.to_csv(OUT / "table5_volume_daylevel.csv", index=False)
    print(fmt(t5[t5["measure"].isin(["index_constituent_volume_shares", "options_premium_turnover_cr", "futures_notional_turnover_cr"])],
              ["before", "index", "measure", "subsample", "n_days_before", "n_days_after", "pct_change_median", "day_mwu_p", "hac_p",
               "boot_p", "weekrel_diff", "weekrel_p_mwu", "paper_printed_pct_change_from_medians", "paper_printed_p"]))
    note(f"table 5 done {time.time() - t0:.0f}s")

    # ---------------- Table 6
    header("Table 6. Rogers-Satchell volatility per minute (square root), before vs after")
    t6 = table6(G, periods, ex, n_dboot, full=True)
    t6.to_csv(OUT / "table6_volatility.csv", index=False)
    print(fmt(t6, ["before", "index", "subsample", "n_min_before", "n_min_after", "median_before", "median_after", "pct_change_median",
                   "minute_mwu_p", "day_mwu_p", "hac_p", "boot_p", "weekrel_diff", "weekrel_p_mwu",
                   "paper_printed_median_before", "paper_printed_median_after", "paper_printed_p"]))
    note(f"table 6 done {time.time() - t0:.0f}s")

    # ---------------- session-rule and overnight sensitivity
    header("Sensitivity: session rule and the overnight return (core numbers only)")
    sens = []
    for trim in rp.SESSION_RULES:
        for overnight in (True, False):
            if trim == a.trim and overnight:
                Gt, ft = G, fits
            else:
                Gt = build_grids(bars, trim, overnight)
                pt = {p: Gt["days"][Gt["days"].isin(days)] for p, days in periods.items()}
                ft = {(p, k): fit_period(Gt, k, period_rows(Gt, pt[p])) for p in pt for k in INDICES}
            base = {"data": LABEL, "trim": trim, "overnight_in_first_minute": overnight}
            for (p, k), f in ft.items():
                par = f["mg"].params
                P = par["alpha"] + par["beta"]
                sens.append(dict(base, table="2", period=p, index=k, item="half_life_min", ours=float(rp.half_life(P))))
                sens.append(dict(base, table="2", period=p, index=k, item="alpha", ours=par["alpha"]))
                sens.append(dict(base, table="2", period=p, index=k, item="beta", ours=par["beta"]))
            pt = {p: Gt["days"][Gt["days"].isin(days)] for p, days in periods.items()}
            _, pr, _ = connectedness_tables(Gt, ft, pt, ex, None, 0, full=False)
            for r in pr[(pr["drop_first_minute"] == False) & (pr["family"].isin(["union", "nifty", "bn", "all_days"]))].itertuples():  # noqa: E712
                sens.append(dict(base, table="3/4", period=r.period, index=f"{r.family}", item=f"total {r.subsample}", ours=r.paper_scale_total))
            if overnight:
                t6s = table6(Gt, pt, ex, 0, full=False)
                for r in t6s[t6s["subsample"].isin(["all", "wed", "thu"])].itertuples():
                    sens.append(dict(base, table="6", period=r.before, index=r.index, item=f"{r.subsample} pct_change_median", ours=r.pct_change_median))
                    sens.append(dict(base, table="6", period=r.before, index=r.index, item=f"{r.subsample} minute_mwu_p", ours=r.minute_mwu_p))
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "sensitivity_session_rule.csv", index=False)
    piv = sens[sens["item"] == "half_life_min"].pivot_table(index=["trim", "overnight_in_first_minute"], columns=["period", "index"], values="ours")
    print("Intraday half-life (minutes) by session rule and treatment of the overnight return:")
    print(piv.round(1).to_string())
    note(f"sensitivity done {time.time() - t0:.0f}s")

    header("Checks on the paper's printed numbers (arithmetic only)")
    pc = printed_checks()
    pc.to_csv(OUT / "paper_printed_checks.csv", index=False)
    simple = pc[~pc["check"].str.contains("printed statistic")]
    print(fmt(simple[~simple["consistent"].astype(bool)], ["check", "item", "value", "compare_with"]))
    w = pc[pc["check"].str.contains("printed statistic")].copy()
    w["table"] = w["check"].str[:7]
    w["reading"] = np.where(w["check"].str.contains("rank sum"), "rank_sum", "U")
    wp = w.pivot_table(index=["table", "item"], columns="reading", values=["consistent", "direction_matches_medians"], aggfunc="first")
    print("\nTables 5 and 6: is the printed statistic possible and does it give the printed p-value? (consistent, by reading)")
    print("and does the implied direction agree with the printed medians? (direction_matches_medians; a soft check only)")
    print(wp.to_string())

    print(f"\nWrote {len(list(OUT.glob('*.csv')))} CSV files and {len(list(OUT.glob('*.png')))} figures to {OUT}/.")
    note(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
