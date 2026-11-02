"""Which market-structure variable explains when expiry-day pinning appears and disappears? (P87, REAL DATA)

This script runs the pre-registration docs/prereg/P87_PINNING_MECHANISM.md, committed to git as
e50a77506692aaf9b53cfca6c3cff6010308d3f7 (2 Nov 2026, 22:39 UTC) before this script was written.
Implementation choices the pre-registration left open, and any departures from it, are listed in
DESIGN_LOG and written to design_log.csv.

Stage A (no outcome data): candidate variables C1 to C6, A1 and the variants C1g, C1w, C2d, built
from the previous day's bhavcopy, the previous official close and the expiry calendars. Written to
candidates.csv and candidate_checks.csv before any outcome is joined.
Stage B: outcome y = distance of NSE's official close to the nearest strike divided by the strike
spacing (50 Nifty, 100 Bank Nifty). Primary test (C1), secondary family (C2, C3, C5, C6, Holm),
between-regime prediction, placebos P1 to P5, exploratory checks, C4 and A1 descriptions.

Model, for each candidate z (standardized by the mean and SD of that index's expiry days):

    y = a[cell] + b[cell] E + d z + gamma E z + m[index] M + n[index] E M + error

cell = index x regime x closing auction; E = expiry day; M = monthly contract (expires with the
index futures). OLS, standard errors clustered by calendar week (CR1), t with (weeks - 1) df.

    OMP_NUM_THREADS=1 python experiments/pinning_mechanism.py --raw data/raw --out results/pinning_mechanism
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from niftygex import data  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = _load_module("pinning_study")  # imported, not copied

LABEL = "REAL DATA"
PREREG_PATH = "docs/prereg/P87_PINNING_MECHANISM.md"
PREREG_COMMIT = "e50a77506692aaf9b53cfca6c3cff6010308d3f7"
GRID = {"NIFTY": 50.0, "BANKNIFTY": 100.0}
SEEDS = {"primary": 8701, "secondary": 8702, "bootstrap": 8703, "placebo": 8704}
N_PERM = 10_000
N_BOOT = 2_000
ENFORCEMENT_START = pd.Timestamp("2025-07-04")  # first trading day after SEBI's interim order (reported 3 Jul 2025)
CAS_START = ps.CAS_START                        # closing auction from 3 Aug 2026
TESTED = ["C1", "C2", "C3", "C5", "C6"]
SECONDARY = ["C2", "C3", "C5", "C6"]
VARIANTS = ["C1g", "C1w", "C2d"]
FALSIFIABLE_BY_SIGN = ["C1", "C2", "C5", "C6"]  # pre-registration section 6
NAMES = {
    "C1": "OI concentration at the two strikes around the previous close (primary)",
    "C1g": "C1 variant: gamma-weighted concentration (30 minutes left)",
    "C1w": "C1 variant: window of two strike spacings either side",
    "C2": "expiring contract's share of the index's option turnover, previous day",
    "C2d": "C2 variant: same share on the day itself (can be caused by pinning)",
    "C3": "contract size, log of lot size times previous close",
    "C4": "weekly index option expiries in the calendar week, NSE and BSE",
    "C5": "log of expiring OI over next-expiry OI, previous day",
    "C6": "log of strike spacing over typical daily move (20-day RMS)",
}
SHORT = {"C1": "C1 OI near the money\n(primary)", "C2": "C2 volume share\n(previous day)", "C3": "C3 contract size",
         "C5": "C5 expiring OI vs\nnext-expiry OI", "C6": "C6 strike spacing vs\ntypical daily move"}
CELL_ORDER = ["NIFTY:multi_weekly:pre", "NIFTY:single_weekly_thu:pre", "NIFTY:single_weekly_tue:pre",
              "NIFTY:single_weekly_tue:auction", "BANKNIFTY:weekly_options:pre", "BANKNIFTY:monthly_only:pre",
              "BANKNIFTY:monthly_only:auction"]
EXPECTED_EXPIRIES = dict(zip(CELL_ORDER, [98, 41, 48, 13, 98, 21, 3]))  # stated in the pre-registration
CONTRASTS = [("Nifty, multi-weekly to Thursday", "NIFTY:multi_weekly:pre", "NIFTY:single_weekly_thu:pre"),
             ("Nifty, Thursday to Tuesday", "NIFTY:single_weekly_thu:pre", "NIFTY:single_weekly_tue:pre"),
             ("Bank Nifty, weekly to monthly", "BANKNIFTY:weekly_options:pre", "BANKNIFTY:monthly_only:pre")]
WEEKDAY = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}

DESIGN_LOG = [
    ("implementation", "Lot size before July 2024 (C3)",
     "Greatest common divisor of the contract's positive OI values; on a day where it disagrees with the "
     "turnover rule round(median(value / (contracts x (strike + premium)))) the turnover rule is used, and if the "
     "divisor rule agreed with NewBrdLotQty on fewer than 99% of UDiFF contract-days the turnover rule is used "
     "throughout. The pre-registration said the divisor is 'checked against' the turnover rule without saying which "
     "wins on disagreement."),
    ("implementation", "C4 calendar for other indices",
     "data/external/other_weekly_expiries.csv was built on 2 Nov 2026 from web search summaries of NSE and BSE "
     "notices (FINNIFTY, MIDCPNIFTY, BANKEX, SENSEX50); every row is 'reported', none verified (BSE servers and "
     "tradingqna.com not reachable). Scheduled weekly dates are rolled back to the previous NSE trading day; the "
     "last scheduled date of each calendar month is the monthly contract and is not counted. Sensex dates come from "
     "results/sensex_confound/sensex_expiry_days_used.csv with the last Sensex expiry of each month taken as monthly. "
     "Nifty and Bank Nifty weekly dates are their bhavcopy expiry days that are not index futures expiry days."),
    ("implementation", "Contract used on t-1",
     "pinning_study.chain_before; if the t-1 file lists the contract under a holiday re-dated expiry, that listing's "
     "turnover, OI and lot size are used for C2, C3 and C5."),
    ("implementation", "C2d on the day itself",
     "Turnover of the contract expiring on E in day t's own bhavcopy (or a listing within 3 days of E) over all "
     "options of the index that day."),
    ("implementation", "Placebo P2 size comparison",
     "'At least as large as with the close' compares |gamma at 14:30| with |gamma| of the main model on the full "
     "sample; gamma with the official close on the same minute-bar sample is also reported."),
    ("implementation", "Holiday-shifted expiry days (exploratory check only)",
     "An expiry day whose next calendar day is a weekday without trading, and whose weekday differs from the most "
     "common expiry weekday of that index within 35 days either side."),
    ("added output", "decisions.csv, descriptive_c4_a1.csv, run_info.json",
     "Summary files requested with the task; they add no analysis that changes a decision."),
    ("added output", "settlement_check.csv",
     "The pre-registered rerun of the check that every expiring option's settlement price equals the official close "
     "(section 8), by cell."),
    ("post hoc, descriptive", "C4 'consistent or inconsistent' rule",
     "The pre-registration did not define it. Rule used: C4 is consistent with a contrast if the change in the mean "
     "weekly count on expiry days points the way the mechanism needs (fewer weeklies with more pinning); a change "
     "smaller than 0.5 weeklies counts as no change, which cannot match a real change in pinning."),
]


# ---------------------------------------------------------------------------
# Candidate helpers (unit-tested in tests/test_pinning_mechanism.py)
# ---------------------------------------------------------------------------


def bracket_strikes(S: float, g: float, n_each_side: int = 1) -> np.ndarray:
    """The n strikes at or below S and the n strikes above it on the grid of spacing g."""
    lo = g * np.floor(S / g)
    low = [lo - g * j for j in range(n_each_side)]
    high = [lo + g * (j + 1) for j in range(n_each_side)]
    return np.array(sorted(low + high))


def near_money_share(strike, oi, S: float, g: float, n_each_side: int = 1) -> float:
    """Share of total OI sitting at the strikes bracketing S (C1; C1w with n_each_side = 2)."""
    strike, oi = np.asarray(strike, float), np.asarray(oi, float)
    tot = np.nansum(oi)
    if not np.isfinite(S) or tot <= 0:
        return np.nan
    K = bracket_strikes(S, g, n_each_side)
    m = np.isclose(strike[:, None], K[None, :], atol=1e-6).any(axis=1)
    return float(np.nansum(oi[m]) / tot)


def lot_from_gcd(oi) -> float:
    """Greatest common divisor of the positive OI values (OI is reported in shares, a multiple of the lot)."""
    v = np.asarray(oi, float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) == 0:
        return np.nan
    return float(np.gcd.reduce(np.round(v).astype(np.int64)))


def lot_from_turnover(value, contracts, strike, premium) -> float:
    """Lot size implied by notional turnover = contracts x lot x (strike + premium) (median over traded rows)."""
    value, contracts, strike, premium = (np.asarray(a, float) for a in (value, contracts, strike, premium))
    m = np.isfinite(value) & (contracts > 0) & ((strike + premium) > 0)
    if not m.any():
        return np.nan
    return float(np.round(np.median(value[m] / (contracts[m] * (strike[m] + premium[m])))))


def choose_lot(lot_new: float, lot_gcd: float, lot_turn: float, gcd_ok: bool = True) -> float:
    """C3 lot rule: NewBrdLotQty when present; else the divisor rule, unless it disagrees with the turnover rule."""
    if np.isfinite(lot_new):
        return float(lot_new)
    if gcd_ok and np.isfinite(lot_gcd) and (not np.isfinite(lot_turn) or lot_gcd == lot_turn):
        return float(lot_gcd)
    return float(lot_turn) if np.isfinite(lot_turn) else np.nan


def safe_log_ratio(a: float, b: float) -> float:
    return float(np.log(a / b)) if (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0) else np.nan


def rms_recent_returns(closes: pd.Series, t, n: int = 20) -> float:
    """Root mean square of the last n daily log changes of ``closes`` that end strictly before t."""
    s = closes.sort_index()
    r = np.log(s).diff().dropna()
    r = r[r.index < pd.Timestamp(t)]
    if len(r) < n:
        return np.nan
    return float(np.sqrt(np.mean(r.iloc[-n:].to_numpy() ** 2)))


def spacing_ratio(g: float, S: float, rms: float) -> float:
    """C6: log of strike spacing over the typical daily move S x rms."""
    return float(np.log(g / (S * rms))) if (np.isfinite(S) and np.isfinite(rms) and S > 0 and rms > 0) else np.nan


def scheduled_weekly_dates(start, end, weekday: int, trading_days) -> pd.DatetimeIndex:
    """The scheduled weekday of every calendar week from start to end, rolled back to the previous trading day."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    tset = set(pd.DatetimeIndex(trading_days).normalize())
    w = start - pd.Timedelta(days=start.weekday())
    out = []
    while w <= end:
        d = w + pd.Timedelta(days=weekday)
        k = 0
        while d not in tset and k < 7:
            d -= pd.Timedelta(days=1)
            k += 1
        if d in tset and start <= d <= end:
            out.append(d)
        w += pd.Timedelta(days=7)
    return pd.DatetimeIndex(sorted(set(out)))


def product_weekly_dates(periods: pd.DataFrame, trading_days) -> pd.DatetimeIndex:
    """Weekly (non-monthly) expiry dates of one index from its weekday periods (columns start, end, weekday).

    The last scheduled date of each calendar month, under the weekday in force at the month's end, is the
    monthly contract and is removed.
    """
    p = periods.sort_values("start")
    weekly = []
    for _, r in p.iterrows():
        weekly += list(scheduled_weekly_dates(r["start"], r["end"], WEEKDAY[r["weekday"]], trading_days))
    weekly = sorted(set(weekly))
    monthly = set()
    for y, m in sorted({(d.year, d.month) for d in weekly}):
        m0 = pd.Timestamp(year=y, month=m, day=1)
        m1 = m0 + pd.offsets.MonthEnd(0)
        rr = p[p["start"] <= m1].iloc[-1]
        cand = scheduled_weekly_dates(m0, m1, WEEKDAY[rr["weekday"]], trading_days)
        if len(cand):
            monthly.add(cand[-1])
    return pd.DatetimeIndex([d for d in weekly if d not in monthly])


def weekly_count(days, weekly_sets: dict) -> pd.Series:
    """C4: for each day, the number of weekly expiries (all products) in its Monday-to-Sunday week."""
    days = pd.DatetimeIndex(days)
    monday = days - pd.to_timedelta(days.weekday, unit="D")
    all_w = pd.DatetimeIndex(sorted(d for s in weekly_sets.values() for d in s))
    wk = pd.Series(all_w - pd.to_timedelta(all_w.weekday, unit="D")).value_counts()
    return pd.Series([int(wk.get(m, 0)) for m in monday], index=days)


# ---------------------------------------------------------------------------
# Regression helpers (unit-tested)
# ---------------------------------------------------------------------------


def build_design(df: pd.DataFrame, zcol: str, ecol: str = "E", cellcol: str = "cell", ecellcol: str | None = None,
                 monthly: bool = True) -> pd.DataFrame:
    """Columns: a[cell], b[cell or ecell] x E, m[index] x M, n[index] x E x M, then d (z) and gamma (E x z)."""
    ecellcol = ecellcol or cellcol
    e = df[ecol].to_numpy(float)
    cols = {}
    for c in sorted(df[cellcol].unique()):
        cols[f"a[{c}]"] = (df[cellcol] == c).to_numpy(float)
    for c in sorted(df[ecellcol].unique()):
        cols[f"b[{c}]"] = (df[ecellcol] == c).to_numpy(float) * e
    if monthly:
        for s in sorted(df["symbol"].unique()):
            ms = (df["symbol"] == s).to_numpy(float) * df["M"].to_numpy(float)
            cols[f"m[{s}]"] = ms
            cols[f"n[{s}]"] = ms * e
    z = df[zcol].to_numpy(float)
    cols["d"] = z
    cols["gamma"] = e * z
    return pd.DataFrame(cols, index=df.index)


def prune_collinear(X: pd.DataFrame, keep=("d", "gamma")) -> pd.DataFrame:
    """Drop all-zero and collinear control columns (never d or gamma)."""
    ctrl = [c for c in X.columns if c not in keep and np.abs(X[c].to_numpy()).sum() > 0]
    kept, M = [], np.empty((len(X), 0))
    for c in ctrl + list(keep):
        cand = np.column_stack([M, X[c].to_numpy(float)])
        if np.linalg.matrix_rank(cand) > M.shape[1]:
            M, kept = cand, kept + [c]
        elif c in keep:
            raise ValueError(f"{c} is collinear with the fixed effects")
    return X[kept]


def ols_cluster(X: np.ndarray, y: np.ndarray, groups) -> dict:
    """OLS with CR1 cluster-robust covariance; inference with t(G - 1)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    codes, _ = pd.factorize(pd.Series(groups).astype(str))
    G = int(codes.max()) + 1
    S = np.zeros((G, k))
    np.add.at(S, codes, X * u[:, None])
    meat = S.T @ S
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return {"beta": beta, "se": se, "df": G - 1, "n": n, "k": k, "G": G}


def fit_model(df: pd.DataFrame, zcol: str, ycol: str = "y", ecol: str = "E", cellcol: str = "cell",
              ecellcol: str | None = None, monthly: bool = True, cluster: str = "week") -> dict:
    sub = df.dropna(subset=[zcol, ycol])
    X = prune_collinear(build_design(sub, zcol, ecol, cellcol, ecellcol, monthly))
    y = sub[ycol].to_numpy(float)
    r = ols_cluster(X.to_numpy(), y, sub[cluster].to_numpy())
    cols = list(X.columns)
    out = {"n_days": int(len(sub)), "n_expiry": int(sub[ecol].sum()), "n_clusters": r["G"]}
    tcrit = stats.t.ppf(0.975, r["df"])
    for name in ("gamma", "d"):
        j = cols.index(name)
        b, s = float(r["beta"][j]), float(r["se"][j])
        p = float(2 * stats.t.sf(abs(b / s), r["df"])) if s > 0 else np.nan
        out.update({name: b, f"{name}_se": s, f"{name}_ci_low": b - tcrit * s, f"{name}_ci_high": b + tcrit * s,
                    f"{name}_p": p})
    out["_X"], out["_y"], out["_sub"] = X, y, sub
    return out


def perm_gamma_p(fit: dict, zcol: str, ecol: str, strata_col: str, rng, n: int = N_PERM, batch: int = 1000) -> float:
    """Permutation p for gamma: z shuffled among days of the same stratum (cell x expiry status).

    Uses the Frisch-Waugh-Lovell form: the controls are partialled out once and each permuted gamma comes
    from a 2 x 2 regression of the residualized outcome on the residualized z and E x z.
    """
    X, y, sub = fit["_X"], fit["_y"], fit["_sub"]
    W = X.drop(columns=["d", "gamma"]).to_numpy(float)
    Q, _ = np.linalg.qr(W)

    def resid(V):
        return V - Q @ (Q.T @ V)

    yr = resid(y)
    z = sub[zcol].to_numpy(float)
    e = sub[ecol].to_numpy(float)
    strata = [np.where(sub[strata_col].to_numpy() == s)[0] for s in pd.unique(sub[strata_col])]

    def gam(Z):
        R1, R2 = resid(Z), resid(e[:, None] * Z)
        a, b, c = (R1 * R1).sum(0), (R1 * R2).sum(0), (R2 * R2).sum(0)
        d1, d2 = R1.T @ yr, R2.T @ yr
        return (a * d2 - b * d1) / (a * c - b * b)

    obs = float(gam(z[:, None])[0])
    null = []
    done = 0
    while done < n:
        B = min(batch, n - done)
        Z = np.empty((len(z), B))
        for idx in strata:
            perm = rng.permuted(np.tile(np.arange(len(idx)), (B, 1)), axis=1)
            Z[idx, :] = z[idx][perm].T
        null.append(gam(Z))
        done += B
    null = np.concatenate(null)
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (n + 1))


def cell_gaps(cell, E, y) -> pd.Series:
    """Observed expiry gap per cell: mean y on expiry days minus mean y on other days."""
    d = pd.DataFrame({"cell": cell, "E": E, "y": y})
    m = d.groupby(["cell", "E"])["y"].mean().unstack()
    return (m.get(1.0, m.get(1)) - m.get(0.0, m.get(0))).rename("gap")


def predicted_gaps(cell, E, z, d: float, gamma: float) -> pd.Series:
    """Gap per cell implied by the candidate: d x (mean z on expiry days - mean z on other days) + gamma x mean z on expiry days."""
    t = pd.DataFrame({"cell": cell, "E": E, "z": z})
    m = t.groupby(["cell", "E"])["z"].mean().unstack()
    ze, zo = m.get(1.0, m.get(1)), m.get(0.0, m.get(0))
    return (d * (ze - zo) + gamma * ze).rename("pred")


def share_explained(pred: np.ndarray, obs: np.ndarray, w: np.ndarray) -> float:
    pred, obs, w = (np.asarray(a, float) for a in (pred, obs, w))
    return float(np.sum(w * pred * obs) / np.sum(w * obs ** 2))


# ---------------------------------------------------------------------------
# Raw bhavcopy scan (turnover, OI totals, lot sizes, futures expiry days)
# ---------------------------------------------------------------------------


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]


def scan_bhavcopies(raw: Path):
    recs, fut_days = [], {s: set() for s in GRID}
    for f in data.bhavcopy_files(raw / "bhavcopy"):
        df = _clean(pd.read_csv(f))
        if "INSTRUMENT" in df.columns:
            date = pd.to_datetime([df["TIMESTAMP"].iloc[0]], format="mixed", dayfirst=True)[0].normalize()
            inst, sym = df["INSTRUMENT"].astype(str).str.strip(), df["SYMBOL"].astype(str).str.strip()
            exp = pd.to_datetime(df["EXPIRY_DT"], format="mixed", dayfirst=True).dt.normalize()
            is_opt, is_fut = inst.eq("OPTIDX"), inst.eq("FUTIDX")
            tab = pd.DataFrame({"value": df["VAL_INLAKH"].astype(float) * 1e5, "contracts": df["CONTRACTS"].astype(float),
                                "strike": df["STRIKE_PR"].astype(float), "premium": df["CLOSE"].astype(float),
                                "oi": df["OPEN_INT"].astype(float), "lot_new": np.nan})
        elif "FinInstrmTp" in df.columns:
            date = pd.to_datetime(df["TradDt"].iloc[0]).normalize()
            inst, sym = df["FinInstrmTp"].astype(str).str.strip(), df["TckrSymb"].astype(str).str.strip()
            exp = pd.to_datetime(df["XpryDt"]).dt.normalize()
            is_opt, is_fut = inst.eq("IDO"), inst.eq("IDF")
            tab = pd.DataFrame({"value": df["TtlTrfVal"].astype(float), "contracts": df["TtlTradgVol"].astype(float),
                                "strike": df["StrkPric"].astype(float), "premium": df["ClsPric"].astype(float),
                                "oi": df["OpnIntrst"].astype(float), "lot_new": df["NewBrdLotQty"].astype(float)})
        else:
            raise ValueError(f"{f}: unrecognised layout")
        tab["expiry"] = exp
        for s in GRID:
            if (is_fut & sym.eq(s) & exp.eq(date)).any():
                fut_days[s].add(date)
            o = tab[is_opt & sym.eq(s)]
            for e, g in o.groupby("expiry"):
                ln = g["lot_new"].dropna()
                recs.append({"date": date, "symbol": s, "expiry": pd.Timestamp(e), "turnover": float(np.nansum(g["value"])),
                             "oi_total": float(np.nansum(g["oi"])), "n_rows": len(g),
                             "lot_gcd": lot_from_gcd(g["oi"]),
                             "lot_turnover": lot_from_turnover(g["value"], g["contracts"], g["strike"], g["premium"]),
                             "lot_new": float(ln.mode().iloc[0]) if len(ln) else np.nan})
    con = pd.DataFrame(recs)
    return con, {s: pd.DatetimeIndex(sorted(v)) for s, v in fut_days.items()}


# ---------------------------------------------------------------------------
# Stage A: candidates (no outcomes)
# ---------------------------------------------------------------------------


DESIGN_COLS = ["symbol", "is_expiry", "is_monthly_expiry", "k", "expiry", "regime", "closing_auction",
               "chain_date", "chain_redated", "close_prev"]


def official_extended(raw: Path, official: dict) -> dict:
    """Official closes with the late-2022 history (for the 20-day RMS of C6)."""
    out = {}
    ex = pd.read_csv(raw / "extra_2022" / "index_daily.csv", parse_dates=["date"])
    for s, v in ps.SYMBOLS.items():
        old = ex[ex["index"].str.lower() == v["index"].lower()].set_index("date")["close"]
        s_all = pd.concat([old[old.index < official[s].index.min()], official[s]]).sort_index()
        out[s] = s_all[~s_all.index.duplicated(keep="last")]
    return out


def weekly_sets_all(raw: Path, expiries: dict, fut_days: dict, trading_days) -> tuple[dict, pd.DataFrame]:
    sets = {}
    for s in GRID:
        sets[s] = pd.DatetimeIndex([d for d in expiries[s] if d not in set(fut_days[s])])
    sx = pd.read_csv(ROOT / "results/sensex_confound/sensex_expiry_days_used.csv", parse_dates=["expiry"])["expiry"]
    sx = pd.DatetimeIndex(sorted(sx))
    last = pd.Series(sx, index=sx).groupby([sx.year, sx.month]).max()
    sets["SENSEX"] = pd.DatetimeIndex([d for d in sx if d not in set(last.values)])
    other = pd.read_csv(ROOT / "data/external/other_weekly_expiries.csv", parse_dates=["start", "end"])
    for idx, p in other.groupby("index"):
        sets[idx] = product_weekly_dates(p, trading_days)
    summ = pd.DataFrame([{"product": k, "n_weekly_dates": len(v), "first": str(v.min().date()) if len(v) else "",
                          "last": str(v.max().date()) if len(v) else ""} for k, v in sets.items()])
    return sets, summ


def holiday_shifted(expiry_days: pd.DatetimeIndex, trading_days) -> set:
    tset = set(pd.DatetimeIndex(trading_days))
    out = set()
    ex = pd.DatetimeIndex(sorted(expiry_days))
    for E in ex:
        nxt = E + pd.Timedelta(days=1)
        if nxt.weekday() >= 5 or nxt in tset:
            continue
        near = ex[(abs((ex - E).days) <= 35) & (ex != E)]
        if len(near) and E.weekday() != pd.Series(near.weekday).mode().iloc[0]:
            out.add(E)
    return out


def build_candidates(sym, panel, official, official_ext, bhav_sym, con_idx, totals, fut_days, c4, trading_days):
    g = GRID[sym]
    days = pd.DatetimeIndex(sorted(set(official[sym].index) & set(bhav_sym))).normalize()
    design = panel[DESIGN_COLS].copy()
    rows = []
    for t, r in design.iterrows():
        E, S = pd.Timestamp(r["expiry"]), float(r["close_prev"])
        row = {"date": t, "symbol": sym}
        ch, cdate, _ = ps.chain_before(bhav_sym, days, t, E)
        if ch is None or not len(ch):
            rows.append(row)
            continue
        ce = pd.Timestamp(ch["expiry"].iloc[0]).normalize()
        tot = ch.groupby("strike")["oi"].sum()
        row["C1"] = near_money_share(tot.index, tot.values, S, g, 1)
        row["C1w"] = near_money_share(tot.index, tot.values, S, g, 2)
        # C1g: gross gamma at S with 30 minutes left, relative to all OI sitting at S
        n_td = int(days.searchsorted(E) - days.searchsorted(cdate))
        T1 = max(n_td, 1) / ps.TRADING_DAYS
        und = ch["underlying"].dropna()
        fb = float(und.iloc[0]) if len(und) else float(official[sym][cdate])
        c = ps.prepare_chain(ch, T1, ps.RATE, fb)
        w = np.ones(len(c))
        num = float(ps.dealer_gamma_at(c, [S], ps.TAU_SETTLE, w)[0])
        c_at = c.copy()
        c_at["strike"] = S
        den = float(ps.dealer_gamma_at(c_at, [S], ps.TAU_SETTLE, w)[0])
        row["C1g"] = num / den if den > 0 else np.nan
        # C2, C3, C5 from the raw scan of the t-1 file
        key = (cdate, sym, ce)
        if key in con_idx.index:
            cr = con_idx.loc[key]
            row["C2"] = cr["turnover"] / totals[(cdate, sym)] if totals.get((cdate, sym), 0) > 0 else np.nan
            row.update({"lot_new": cr["lot_new"], "lot_gcd": cr["lot_gcd"], "lot_turnover": cr["lot_turnover"]})
            later = con_idx.loc[(cdate, sym)]
            later = later[later.index > ce]
            row["C5"] = safe_log_ratio(cr["oi_total"], later["oi_total"].iloc[0]) if len(later) else np.nan
        # C2d: the same share on day t itself
        if (t, sym) in totals:
            todays = con_idx.loc[(t, sym)]
            cand = [e for e in todays.index if e == E] or sorted(e for e in todays.index if abs((e - E).days) <= 3 and e >= t)
            if cand and totals[(t, sym)] > 0:
                row["C2d"] = todays.loc[cand[0], "turnover"] / totals[(t, sym)]
        row["C6"] = spacing_ratio(g, S, rms_recent_returns(official_ext[sym], t))
        row.update({"chain_expiry_used": ce, "chain_date": cdate})
        rows.append(row)
    cand = pd.DataFrame(rows).set_index("date")
    out = design.drop(columns=["chain_date"]).join(cand.drop(columns=["symbol"]), how="left")
    out["E"] = out["is_expiry"].astype(float)
    out["M"] = out["expiry"].isin(fut_days[sym]).astype(float)
    out["C4"] = c4.reindex(out.index).astype(float)
    out["A1"] = (out.index >= ENFORCEMENT_START).astype(float)
    out["cell"] = sym + ":" + out["regime"] + ":" + np.where(out["closing_auction"], "auction", "pre")
    idx = pd.DatetimeIndex(out.index)
    out["week"] = (idx - pd.to_timedelta(idx.weekday, unit="D")).strftime("%Y-%m-%d")
    out["month"] = idx.strftime("%Y-%m")
    out["half"] = idx.year.astype(str) + np.where(idx.month <= 6, "H1", "H2")
    out["cellhalf"] = out["cell"] + ":" + out["half"]
    out["F"] = (out["k"] == 1).astype(float)  # day before expiry (placebo P1)
    shifted = holiday_shifted(pd.DatetimeIndex(out.index[out["is_expiry"]]), trading_days)
    out["holiday_shifted_expiry"] = out.index.isin(list(shifted)) & out["is_expiry"]
    return out


def finalize_lots(cands: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ud = cands[cands["lot_new"].notna()]
    agree_gcd = float((ud["lot_gcd"] == ud["lot_new"]).mean()) if len(ud) else np.nan
    agree_turn = float((ud["lot_turnover"] == ud["lot_new"]).mean()) if len(ud) else np.nan
    gcd_ok = bool(agree_gcd >= 0.99)
    cands = cands.copy()
    cands["lot"] = [choose_lot(a, b, c, gcd_ok) for a, b, c in zip(cands["lot_new"], cands["lot_gcd"], cands["lot_turnover"])]
    cands["contract_value_rs"] = cands["lot"] * cands["close_prev"]
    cands["C3"] = np.log(cands["contract_value_rs"])
    leg = cands[cands["lot_new"].isna() & cands["lot_gcd"].notna()]
    info = {"udiff_contract_days": len(ud), "agree_gcd_vs_NewBrdLotQty": agree_gcd,
            "agree_turnover_vs_NewBrdLotQty": agree_turn, "gcd_rule_used_for_legacy_days": gcd_ok,
            "legacy_contract_days": len(leg),
            "legacy_days_gcd_disagrees_with_turnover_rule": int((leg["lot_gcd"] != leg["lot_turnover"]).sum())}
    return cands, info


def standardize(c: pd.DataFrame, cols) -> pd.DataFrame:
    c = c.copy()
    for col in cols:
        c[f"z_{col}"] = np.nan
        for s in c["symbol"].unique():
            m = c["symbol"] == s
            ref = c.loc[m & (c["E"] == 1), col].dropna()
            c.loc[m, f"z_{col}"] = (c.loc[m, col] - ref.mean()) / ref.std(ddof=1)
    return c


# ---------------------------------------------------------------------------
# Stage B helpers
# ---------------------------------------------------------------------------


def row_from_fit(fit: dict, **extra) -> dict:
    r = {k: v for k, v in fit.items() if not k.startswith("_")}
    r.update(extra)
    return r


def between_regime(df: pd.DataFrame, fits: dict, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each cell's expiry gap from the within-cell estimates; compare changes across regimes."""
    obs_gap = cell_gaps(df["cell"], df["E"], df["y"])
    obs = np.array([obs_gap[b] - obs_gap[a] for _, a, b in CONTRASTS])
    preds, cellrows = {}, []
    for cnd in TESTED:
        zc = f"z_{cnd}"
        sub = df.dropna(subset=[zc])
        pg = predicted_gaps(sub["cell"], sub["E"], sub[zc], fits[cnd]["d"], fits[cnd]["gamma"])
        preds[cnd] = np.array([pg[b] - pg[a] for _, a, b in CONTRASTS])
        zm = sub.groupby(["cell", "E"])[zc].mean().unstack()
        for cell in CELL_ORDER:
            cellrows.append({"row_type": "cell", "candidate": cnd, "cell": cell, "mean_z_expiry": zm.loc[cell, 1.0],
                             "mean_z_other": zm.loc[cell, 0.0], "predicted_gap": pg[cell], "observed_gap": obs_gap[cell]})
    # bootstrap: resample calendar weeks within cells
    pos = np.arange(len(df))
    blocks = {}
    for (cell, wk), ix in df.reset_index(drop=True).groupby(["cell", "week"]).indices.items():
        blocks.setdefault(cell, []).append(ix)
    cell_list = list(blocks)
    prepared = {}
    for cnd in TESTED:
        zc = f"z_{cnd}"
        mask = df[[zc, "y"]].notna().all(axis=1).to_numpy()
        fit = fits[cnd]
        X = fit["_X"].to_numpy(float)
        inv = np.full(len(df), -1)
        inv[pos[mask]] = np.arange(mask.sum())
        prepared[cnd] = (mask, X, inv, list(fit["_X"].columns))
    cell_arr, E_arr, y_arr = df["cell"].to_numpy(), df["E"].to_numpy(float), df["y"].to_numpy(float)
    boot_obs = np.empty((N_BOOT, len(CONTRASTS)))
    boot_pred = {c: np.empty((N_BOOT, len(CONTRASTS))) for c in TESTED}
    for b in range(N_BOOT):
        rows = np.concatenate([blocks[c][j] for c in cell_list for j in rng.integers(0, len(blocks[c]), len(blocks[c]))])
        og = cell_gaps(cell_arr[rows], E_arr[rows], y_arr[rows])
        boot_obs[b] = [og.get(bb, np.nan) - og.get(aa, np.nan) for _, aa, bb in CONTRASTS]
        for cnd in TESTED:
            mask, X, inv, cols = prepared[cnd]
            r = rows[mask[rows]]
            Xi, yi = X[inv[r]], y_arr[r]
            beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]
            dd, gg = beta[cols.index("d")], beta[cols.index("gamma")]
            pg = predicted_gaps(cell_arr[r], E_arr[r], df[f"z_{cnd}"].to_numpy(float)[r], dd, gg)
            boot_pred[cnd][b] = [pg.get(bb, np.nan) - pg.get(aa, np.nan) for _, aa, bb in CONTRASTS]
    w = 1.0 / np.nanvar(boot_obs, axis=0, ddof=1)
    rows_out = []
    for cnd in TESTED:
        s = share_explained(preds[cnd], obs, w)
        sb = np.array([share_explained(boot_pred[cnd][b], boot_obs[b], w) for b in range(N_BOOT)])
        for k, (name, a, bb) in enumerate(CONTRASTS):
            rows_out.append({"row_type": "contrast", "candidate": cnd, "contrast": name, "cell_from": a, "cell_to": bb,
                             "observed_change": obs[k], "observed_ci_low": np.nanquantile(boot_obs[:, k], 0.025),
                             "observed_ci_high": np.nanquantile(boot_obs[:, k], 0.975), "weight": w[k],
                             "predicted_change": preds[cnd][k],
                             "predicted_ci_low": np.nanquantile(boot_pred[cnd][:, k], 0.025),
                             "predicted_ci_high": np.nanquantile(boot_pred[cnd][:, k], 0.975)})
        rows_out.append({"row_type": "share_explained", "candidate": cnd, "contrast": "all three", "share_explained_s": s,
                         "s_ci_low": np.nanquantile(sb, 0.025), "s_ci_high": np.nanquantile(sb, 0.975),
                         "pred_sign_ok_nifty_multi_to_thu": bool(preds[cnd][0] < 0),
                         "pred_sign_ok_nifty_thu_to_tue": bool(preds[cnd][1] > 0)})
    return pd.DataFrame(rows_out), pd.DataFrame(cellrows)


def figure(prim_sec: pd.DataFrame, plac: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("main: official close", None, "#1f4e79", "o"), ("P1 fake expiry (day before)", "P1a", "#8c8c8c", "s"),
              ("P2 price at 14:30", "P2", "#c55a11", "^"), ("P3 fake strikes (g/4)", "P3", "#548235", "D"),
              ("P5 half-year expiry effects", "P5", "#7030a0", "v")]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ys = np.arange(len(TESTED))[::-1]
    for j, (lab, key, col, mk) in enumerate(series):
        off = (j - 2) * 0.13
        for i, cnd in enumerate(TESTED):
            if key is None:
                r = prim_sec[prim_sec["candidate"] == cnd].iloc[0]
            else:
                r = plac[(plac["candidate"] == cnd) & (plac["placebo"] == key)].iloc[0]
            g, lo, hi = r["gamma"] * 50, r["gamma_ci_low"] * 50, r["gamma_ci_high"] * 50
            ax.errorbar(g, ys[i] + off, xerr=[[g - lo], [hi - g]], fmt=mk, color=col, ms=5, capsize=2,
                        label=lab if i == 0 else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvspan(-1.23, 1.23, color="#dddddd", alpha=0.5, lw=0, label="pre-registered MDE (1.2 Nifty points)")
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT[c] for c in TESTED], fontsize=8)
    ax.set_xlabel("gamma: extra change in distance on expiry days per 1 SD of the candidate\n"
                  "(Nifty-point equivalents: share of strike spacing x 50; negative = more pinning)", fontsize=8)
    ax.set_title("REAL DATA: within-regime slopes of expiry-day distance to the nearest strike (95% intervals)", fontsize=9)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_within_slopes.png", dpi=130)
    plt.close(fig)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(d: Path) -> dict:
    files = sorted(x for x in d.iterdir() if x.is_file())
    lines = "\n".join(f"{x.name}:{sha256_file(x)}" for x in files)
    return {"n_files": len(files), "sha256_of_name_hash_list": hashlib.sha256(lines.encode()).hexdigest()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/pinning_mechanism")
    ap.add_argument("--stdout", default="results/pinning_mechanism_stdout.txt")
    ap.add_argument("--stage-a-only", action="store_true", help="build and check the candidates only (no outcomes)")
    a = ap.parse_args()
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"{LABEL}. P87 pinning mechanism. Pre-registration {PREREG_PATH}, commit {PREREG_COMMIT}."]

    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    def table(name, df):
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 4,
                               "display.max_colwidth", 60):
            log("")
            log(f"== {name}")
            log(df.to_string(index=False))

    # ------------------------------------------------------------ inputs
    official, bhav, ptables = ps.load_inputs(raw)
    log(f"Loaded {len(bhav['NIFTY'])} NIFTY and {len(bhav['BANKNIFTY'])} BANKNIFTY bhavcopy days ({time.time() - t0:.0f}s)")
    con, fut_days = scan_bhavcopies(raw)
    con_idx = con.set_index(["date", "symbol", "expiry"]).sort_index()
    totals = con.groupby(["date", "symbol"])["turnover"].sum().to_dict()
    log(f"Scanned bhavcopies for turnover, OI and lot sizes: {len(con)} contract-days ({time.time() - t0:.0f}s)")
    official_ext = official_extended(raw, official)
    trading_days = pd.DatetimeIndex(sorted(set(official["NIFTY"].index) | set(bhav["NIFTY"])))
    panels, expiries = {}, {}
    for sym, cfg in ps.SYMBOLS.items():
        closes = ps.minute_closes(raw, cfg["bars"]) if (raw / cfg["bars"]).exists() else None
        panels[sym], expiries[sym], _ = ps.build_day_panel(sym, official[sym], bhav[sym], ptables, closes)
        log(f"{sym}: day panel {len(panels[sym])} days ({time.time() - t0:.0f}s)")
    wsets, wsumm = weekly_sets_all(raw, expiries, fut_days, trading_days)
    all_days = pd.DatetimeIndex(sorted(set(panels["NIFTY"].index) | set(panels["BANKNIFTY"].index)))
    c4 = weekly_count(all_days, wsets)

    # ------------------------------------------------------------ stage A: candidates only
    cands = pd.concat([build_candidates(s, panels[s], official, official_ext, bhav[s], con_idx, totals, fut_days, c4,
                                        trading_days) for s in GRID])
    cands, lotinfo = finalize_lots(cands)
    cands = standardize(cands, TESTED + ["C4"] + VARIANTS)
    cands.index.name = "date"
    cand_cols = (["symbol", "cell", "E", "M", "k", "F", "expiry", "chain_date", "chain_expiry_used", "chain_redated",
                  "holiday_shifted_expiry", "close_prev", "lot_new", "lot_gcd", "lot_turnover", "lot", "contract_value_rs",
                  "A1", "week", "month", "half", "cellhalf"] + TESTED + ["C4"] + VARIANTS
                 + [f"z_{c}" for c in TESTED + ["C4"] + VARIANTS])
    cands[cand_cols].to_csv(out / "candidates.csv")
    checks = []
    for cell in CELL_ORDER:
        n = int(((cands["cell"] == cell) & (cands["E"] == 1)).sum())
        checks.append({"check": f"expiry days in cell {cell}", "value": n, "expected": EXPECTED_EXPIRIES[cell],
                       "ok": n == EXPECTED_EXPIRIES[cell]})
    checks.append({"check": "index-days in panel", "value": len(cands), "expected": 1878, "ok": len(cands) == 1878})
    for k_, v in lotinfo.items():
        checks.append({"check": f"lot size: {k_}", "value": v})
    e = cands[cands["E"] == 1]
    for s in GRID:
        es = e[e["symbol"] == s]
        checks.append({"check": f"{s}: expiry days where futures-based monthly flag M equals pinning study's monthly label",
                       "value": f"{int((es['M'] == es['is_monthly_expiry'].astype(float)).sum())} of {len(es)}"})
        checks.append({"check": f"{s}: monthly expiry days (M = 1)", "value": int(es["M"].sum())})
        checks.append({"check": f"{s}: holiday-shifted expiry days (exploratory definition)",
                       "value": int(es["holiday_shifted_expiry"].sum())})
        checks.append({"check": f"{s}: days using a holiday re-dated contract",
                       "value": int(cands.loc[cands["symbol"] == s, "chain_redated"].fillna(False).astype(bool).sum())})
    for c in TESTED + ["C4"] + VARIANTS + ["A1"]:
        checks.append({"check": f"missing {c} (days)", "value": int(cands[c].isna().sum())})
    for _, r in wsumm.iterrows():
        checks.append({"check": f"C4 source: weekly expiry dates of {r['product']}", "value": int(r["n_weekly_dates"]),
                       "note": f"{r['first']} to {r['last']}"})
    for c in TESTED + ["C4"]:
        for cell in CELL_ORDER:
            q = cands[cands["cell"] == cell]
            for lab, m in (("expiry", q["E"] == 1), ("other", q["E"] == 0)):
                checks.append({"check": f"{c} {cell} {lab} days: within-cell SD of z / mean of raw value",
                               "value": round(float(q.loc[m, f"z_{c}"].std(ddof=1)), 4),
                               "note": f"mean raw {q.loc[m, c].mean():.5g}; n {int(q.loc[m, c].notna().sum())}"})
    for cell in CELL_ORDER:
        q = cands[(cands["cell"] == cell) & (cands["E"] == 1)]
        checks.append({"check": f"{cell} expiry days: mean lot size / mean contract value (Rs lakh) / mean weekly count",
                       "value": round(float(q["lot"].mean()), 2),
                       "note": f"contract value {q['contract_value_rs'].mean() / 1e5:.2f} lakh; weekly count {q['C4'].mean():.2f}"})
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out / "candidate_checks.csv", index=False)
    table("candidate_checks (stage A, written before any outcome was joined)", checks_df)
    log(f"Stage A done ({time.time() - t0:.0f}s)")
    if a.stage_a_only:
        return 0

    # ------------------------------------------------------------ stage B: outcomes
    outc = []
    for s in GRID:
        p, g = panels[s], GRID[s]
        o = pd.DataFrame(index=p.index)
        o["symbol"] = s
        o["y"] = p["d"] / g
        o["y_quarter"] = ps.dist_to_grid(p["close"], g, offset=g / 4) / g
        o["half_hit"] = (p["d_placebo"] <= g / 10).astype(float)
        o["y_1430"] = ps.dist_to_grid(p["p1429"], g) / g if "p1429" in p else np.nan
        o["y_1500"] = ps.dist_to_grid(p["p1459"], g) / g if "p1459" in p else np.nan
        outc.append(o)
    outc = pd.concat(outc).reset_index().rename(columns={"index": "date"})
    df = cands.reset_index().merge(outc, on=["date", "symbol"], how="left", validate="1:1")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["strata"] = df["cell"] + ":" + df["E"].astype(int).astype(str)

    # pre-registered data check (section 8): settlement price of every expiring option equals the official close
    sett = []
    for s in GRID:
        for E in expiries[s]:
            if E not in panels[s].index:
                continue
            ch = bhav[s][E]
            ch = ch[pd.to_datetime(ch["expiry"]).dt.normalize() == E]
            ok = bool(len(ch) and np.all(np.abs(ch["settle"].to_numpy(float) - official[s][E]) < 0.005))
            sett.append({"symbol": s, "cell": cands.loc[(cands.index == E) & (cands["symbol"] == s), "cell"].iloc[0],
                         "expiry": E, "all_settlement_prices_equal_official_close": ok})
    sett = pd.DataFrame(sett)
    sett_sum = sett.groupby("cell")["all_settlement_prices_equal_official_close"].agg(["sum", "count"]).reset_index()
    sett_sum.columns = ["cell", "expiry_days_settlement_equals_close", "expiry_days"]
    sett_sum.to_csv(out / "settlement_check.csv", index=False)
    table("settlement price equals official close on expiry days (pre-registered check)", sett_sum)

    # primary and secondary
    fits = {c: fit_model(df, f"z_{c}") for c in TESTED}
    rng_p, rng_s = np.random.default_rng(SEEDS["primary"]), np.random.default_rng(SEEDS["secondary"])
    perm = {"C1": perm_gamma_p(fits["C1"], "z_C1", "E", "strata", rng_p)}
    for c in SECONDARY:
        perm[c] = perm_gamma_p(fits[c], f"z_{c}", "E", "strata", rng_s)
    rows = []
    for c in TESTED:
        f = fits[c]
        rows.append(row_from_fit(f, candidate=c, description=NAMES[c], role="primary" if c == "C1" else "secondary",
                                 p_perm_within_cell=perm[c], gamma_nifty_points_per_sd=f["gamma"] * 50,
                                 gamma_banknifty_points_per_sd=f["gamma"] * 100))
    ps_df = pd.DataFrame(rows)
    sec = ps_df["role"] == "secondary"
    ps_df.loc[sec, "p_holm"] = ps.holm(ps_df.loc[sec, "gamma_p"].to_numpy())
    ps_df["p_decision"] = np.where(ps_df["role"] == "primary", ps_df["gamma_p"], ps_df["p_holm"])
    ps_df["cond1_within_regime"] = (ps_df["gamma"] < 0) & (ps_df["p_decision"] < 0.05)
    ps_df["wrong_sign_significant"] = (ps_df["gamma"] > 0) & (ps_df["p_decision"] < 0.05)
    ps_df[ps_df["role"] == "primary"].to_csv(out / "primary_test.csv", index=False)
    ps_df[ps_df["role"] == "secondary"].to_csv(out / "secondary_tests.csv", index=False)
    table("primary and secondary tests", ps_df)
    log(f"Main tests done ({time.time() - t0:.0f}s)")

    # placebos
    rng_pl = np.random.default_rng(SEEDS["placebo"])
    plac = []
    nonexp = df[df["E"] == 0].copy()
    nonexp["strataF"] = nonexp["cell"] + ":" + nonexp["F"].astype(int).astype(str)
    for c in TESTED:
        zc, g_main = f"z_{c}", fits[c]["gamma"]
        sgn = np.sign(g_main)
        f1 = fit_model(nonexp, zc, ecol="F")
        p1 = perm_gamma_p(f1, zc, "F", "strataF", rng_pl)
        fl = bool(np.sign(f1["gamma"]) == sgn and f1["gamma_p"] < 0.05)
        plac.append(row_from_fit(f1, candidate=c, placebo="P1a", description="non-expiry days; day before expiry as fake E",
                                 p_perm_within_cell=p1, flag_rule="same sign as gamma and p < 0.05", flagged=fl))
        fm = fits[c]
        fl = bool(np.sign(fm["d"]) == sgn and fm["d_p"] < 0.05)
        plac.append({"candidate": c, "placebo": "P1b", "description": "slope d on ordinary days in the main model",
                     "gamma": fm["d"], "gamma_se": fm["d_se"], "gamma_ci_low": fm["d_ci_low"], "gamma_ci_high": fm["d_ci_high"],
                     "gamma_p": fm["d_p"], "n_days": fm["n_days"], "flag_rule": "same sign as gamma and p < 0.05", "flagged": fl})
        for key, ycol, desc in (("P2", "y_1430", "price at 14:30 (14:29 bar close) instead of the official close"),
                                ("P2b", "y_1500", "price at 15:00 (14:59 bar close); reported only"),
                                ("P3", "y_quarter", "mean distance to strikes shifted by g/4"),
                                ("P4", "half_hit", "share of closes within g/10 of a half-way point")):
            fp = fit_model(df, zc, ycol=ycol)
            pp = perm_gamma_p(fp, zc, "E", "strata", rng_pl)
            extra = {}
            if key == "P2":
                fl = bool(np.sign(fp["gamma"]) == sgn and fp["gamma_p"] < 0.05 and abs(fp["gamma"]) >= abs(g_main))
                same = fit_model(df[df["y_1430"].notna()], zc)
                extra = {"gamma_close_same_sample": same["gamma"], "gamma_close_same_sample_p": same["gamma_p"]}
                rule = "same sign, p < 0.05 and |gamma| at least |main gamma|"
            elif key == "P2b":
                fl, rule = False, "none (reported only)"
            elif key == "P3":
                fl, rule = bool(fp["gamma_p"] < 0.05), "p < 0.05"
            else:
                fl, rule = bool(fp["gamma"] > 0 and fp["gamma_p"] < 0.05), "gamma > 0 and p < 0.05"
            plac.append(row_from_fit(fp, candidate=c, placebo=key, description=desc, p_perm_within_cell=pp,
                                     flag_rule=rule, flagged=fl, **extra))
        f5 = fit_model(df, zc, ecellcol="cellhalf")
        fl = not (np.sign(f5["gamma"]) == sgn and abs(f5["gamma"]) >= 0.5 * abs(g_main))
        plac.append(row_from_fit(f5, candidate=c, placebo="P5", description="separate expiry effect per cell and half-year",
                                 flag_rule="not (same sign and at least half the size)", flagged=bool(fl)))
    plac_df = pd.DataFrame(plac)
    plac_df.to_csv(out / "placebos.csv", index=False)
    table("placebos", plac_df.drop(columns=["description"]))
    log(f"Placebos done ({time.time() - t0:.0f}s)")

    # between-regime prediction
    rng_b = np.random.default_rng(SEEDS["bootstrap"])
    btw, btw_cells = between_regime(df, fits, rng_b)
    btw_all = pd.concat([btw, btw_cells], ignore_index=True)
    btw_all.to_csv(out / "between_regime.csv", index=False)
    table("between-regime prediction", btw)
    log(f"Between-regime bootstrap done ({time.time() - t0:.0f}s)")

    # decisions
    dec = []
    for c in TESTED:
        r = ps_df[ps_df["candidate"] == c].iloc[0]
        flags = plac_df[(plac_df["candidate"] == c) & plac_df["flagged"].astype(bool)]["placebo"].tolist()
        sh = btw[(btw["candidate"] == c) & (btw["row_type"] == "share_explained")].iloc[0]
        c1, c2 = bool(r["cond1_within_regime"]), len(flags) == 0
        c3 = bool(sh["pred_sign_ok_nifty_multi_to_thu"] and sh["pred_sign_ok_nifty_thu_to_tue"]
                  and sh["share_explained_s"] >= 0.5)
        if c1 and c2 and c3:
            label = "explains when pinning appears and disappears"
        elif c1 and c2:
            label = "matters day to day but not for the regime pattern"
        elif c1:
            label = "suspect (a placebo flags)"
        else:
            label = "not supported"
        if bool(r["wrong_sign_significant"]):
            label += ("; falsified (significant wrong sign)" if c in FALSIFIABLE_BY_SIGN
                      else "; significant wrong sign (not a pre-stated falsifier for this candidate)")
        dec.append({"candidate": c, "role": r["role"], "gamma": r["gamma"], "p_used": r["p_decision"],
                    "cond1_within_regime": c1, "placebo_flags": ",".join(flags) or "none", "cond2_no_flag": c2,
                    "share_explained_s": sh["share_explained_s"], "cond3_between_regime": c3, "decision": label})
    dec_df = pd.DataFrame(dec)
    dec_df.to_csv(out / "decisions.csv", index=False)
    table("decisions (pre-registered rule)", dec_df)

    # exploratory
    rob = []
    specs = [("Nifty only", lambda d: d[d["symbol"] == "NIFTY"], {}),
             ("Bank Nifty only", lambda d: d[d["symbol"] == "BANKNIFTY"], {}),
             ("no monthly controls", lambda d: d, {"monthly": False}),
             ("drop closing-auction days (from 3 Aug 2026)", lambda d: d[~d["cell"].str.endswith("auction")], {}),
             ("drop holiday-shifted expiry days", lambda d: d[~d["holiday_shifted_expiry"].astype(bool)], {}),
             ("drop days using a re-dated contract", lambda d: d[~d["chain_redated"].fillna(False).astype(bool)], {}),
             ("cluster by calendar month", lambda d: d, {"cluster": "month"})]
    for c in TESTED:
        for name, sel, kw in specs:
            f = fit_model(sel(df), f"z_{c}", **kw)
            rob.append(row_from_fit(f, candidate=c, spec=name, label="exploratory"))
    for c in VARIANTS:
        f = fit_model(df, f"z_{c}")
        rob.append(row_from_fit(f, candidate=c, spec=NAMES[c], label="exploratory"))
    rob_df = pd.DataFrame(rob)
    rob_df.to_csv(out / "robustness.csv", index=False)
    table("exploratory checks", rob_df[["candidate", "spec", "n_days", "n_expiry", "gamma", "gamma_ci_low",
                                         "gamma_ci_high", "gamma_p"]])

    # descriptive: C4 and A1
    desc = []
    f4 = fit_model(df, "z_C4")
    desc.append(row_from_fit(f4, item="C4 within-cell gamma (descriptive, not counted)"))
    m4 = df[df["E"] == 1].groupby("cell")["C4"].mean()
    for name, a_, b_ in CONTRASTS:
        dc4 = m4[b_] - m4[a_]
        og = cell_gaps(df["cell"], df["E"], df["y"])
        dobs = og[b_] - og[a_]
        needed = "fewer weeklies" if dobs < 0 else "more weeklies"
        cons = (abs(dc4) >= 0.5) and (np.sign(dc4) == np.sign(dobs))
        desc.append({"item": f"C4 consistency (post hoc rule): {name}", "mean_C4_expiry_from": m4[a_],
                     "mean_C4_expiry_to": m4[b_], "change_C4": dc4, "observed_gap_change": dobs,
                     "mechanism_needs": needed, "consistent": bool(cons)})
    q = df[df["cell"] == "NIFTY:single_weekly_thu:pre"]
    for lab, m in (("before 4 Jul 2025", q["A1"] == 0), ("from 4 Jul 2025", q["A1"] == 1)):
        qq = q[m]
        desc.append({"item": f"A1: Nifty Thursday regime, {lab}", "n_expiry": int(qq["E"].sum()),
                     "n_other": int((qq["E"] == 0).sum()), "mean_y_expiry": qq.loc[qq["E"] == 1, "y"].mean(),
                     "mean_y_other": qq.loc[qq["E"] == 0, "y"].mean(),
                     "gap": qq.loc[qq["E"] == 1, "y"].mean() - qq.loc[qq["E"] == 0, "y"].mean(),
                     "gap_nifty_points": 50 * (qq.loc[qq["E"] == 1, "y"].mean() - qq.loc[qq["E"] == 0, "y"].mean())})
    desc_df = pd.DataFrame(desc)
    desc_df.to_csv(out / "descriptive_c4_a1.csv", index=False)
    table("descriptive C4 and A1", desc_df)

    pd.DataFrame(DESIGN_LOG, columns=["type", "item", "detail"]).to_csv(out / "design_log.csv", index=False)
    figure(ps_df, plac_df, out)

    # headline
    r = ps_df[ps_df["role"] == "primary"].iloc[0]
    log("")
    log("== HEADLINE (REAL DATA)")
    log(f"Primary (C1): gamma = {r['gamma']:.4f} of the strike spacing per SD [{r['gamma_ci_low']:.4f}, "
        f"{r['gamma_ci_high']:.4f}] = {r['gamma_nifty_points_per_sd']:.2f} Nifty points per SD; week-clustered p = "
        f"{r['gamma_p']:.4f}; permutation p = {r['p_perm_within_cell']:.4f}; {r['n_days']} days, {r['n_expiry']} expiry days, "
        f"{r['n_clusters']} weeks.")
    for _, d in dec_df.iterrows():
        log(f"  {d['candidate']}: {d['decision']} (gamma {d['gamma']:.4f}, p used {d['p_used']:.4f}, flags {d['placebo_flags']}, "
            f"s = {d['share_explained_s']:.2f})")
    run_s = time.time() - t0
    log(f"Total run time {run_s:.0f}s.")

    root = ROOT
    hashes = {str(p.resolve().relative_to(ROOT.resolve())) if p.resolve().is_relative_to(ROOT.resolve()) else str(p):
              sha256_file(p) for p in [raw / "index_daily.csv", raw / "extra_2022" / "index_daily.csv",
                                               raw / "nifty_1min.csv", raw / "banknifty_1min.csv",
                                               root / "results/sensex_confound/sensex_expiry_days_used.csv",
                                               root / "data/external/sensex_expiry_calendar.csv",
                                               root / "data/external/other_weekly_expiries.csv",
                                               root / PREREG_PATH, HERE / "pinning_mechanism.py",
                                               HERE / "pinning_study.py"] if p.exists()}
    hashes["data/raw/bhavcopy (directory)"] = sha256_dir(raw / "bhavcopy")
    hashes["data/raw/participant_oi (directory)"] = sha256_dir(raw / "participant_oi")
    info = {"label": LABEL, "prereg": PREREG_PATH, "prereg_commit": PREREG_COMMIT, "started_utc": started,
            "run_seconds": round(run_s, 1), "seeds": SEEDS, "n_permutations": N_PERM, "n_bootstrap": N_BOOT,
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "inputs_sha256": hashes}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    Path(a.stdout).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
