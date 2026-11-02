"""RQ4 (tracker item P42): does Nifty / Bank Nifty volatility connectedness depend on the dealer gamma regime?

REAL DATA: validated third-party 1-minute Nifty 50 and Bank Nifty bars (data/raw/*_1min.csv), the NSE-based
daily GEX panel and day file written by the pipeline (results/pipeline_nse_v2/daily_gex_panel.csv and
daily_market.csv). Nothing here is simulated; the synthetic checks live in tests/test_connectedness_gamma.py.

Idea. Option dealers who are short gamma hedge by trading with the market (buy after rises, sell after falls).
If both indices are hedged that way, a shock to one could spread to the other more strongly. Dealers who are
long gamma trade against the market, which should damp that. So connectedness (how much of each index's
volatility forecast error comes from shocks to the other index, Diebold and Yilmaz 2012) might be higher after
short-gamma evenings. The paper (Chhimwal, Pandey and Pandey 2025) does not ask this.

PRE-STATED DESIGN (written before the script was first run on the real data; reported whatever it shows)
--------------------------------------------------------------------------------------------------------
Sample: every trading day from 3 Jan 2023 to the last day with Bank Nifty bars (22 Apr 2026) on which both
indices have at least 300 of the 360 bars, GEX exists at the previous close, and India VIX exists at the
previous close. The closing auction (from 3 Aug 2026; minutes 15:16 to 15:28 frozen) falls after the last
Bank Nifty bar, so no auction day can enter; the 360-bar session 09:15 to 15:14 (the replication's rule)
also leaves out the last 15 minutes. The script counts and prints auction days anyway.

Volatility input (the replication's primary): mcsGARCH conditional volatility of each index, sqrt(h * s * q),
fitted in-sample per expiry regime (the regime plays the role of the replication's "period"), ARMA(1,1) mean,
Student t, daily variance h from an AR(1)-EGARCH(1,1)-t fitted on daily returns since 2017 up to the regime's
last day. VAR lags never cross from one day to the next (each day is its own segment), as in the replication.

Gamma state (known at the previous close): the scale-free share = net GEX / gross GEX of the same front chain.
Day t gets the share published at the close of the previous trading day.
  * Primary rule: Pro participant sign (column net_gex_participant). Short gamma = share < 0.
  * Secondary rules: US convention and Client mirror (same split at zero), and gross gamma (no sign): top
    versus bottom tercile of log gross GEX at the previous close, terciles formed within each expiry regime.

PRIMARY TEST (one test, no multiple-testing correction):
  Statistic: regime-stratified difference in total connectedness, short-gamma days minus long-gamma days.
  Within each expiry regime (multi_weekly, single_weekly_thu, single_weekly_tue) the short days are pooled
  into one segment-aware VAR(2), the long days into another; total connectedness on the paper's scale (the
  sum of the two cross shares, 0 to 200) at horizon H = 10 minutes; the regime differences are averaged with
  weights equal to each regime's share of the sample days (a regime enters only if both states have at
  least 5 days in it). Stratifying by regime is needed because short-gamma days under the Pro rule are
  concentrated in 2023 to 2024 (the multi-weekly regime), and connectedness is known to move between regimes.
  p-value: two-sided permutation test that moves the state labels in blocks of 20 consecutive trading days,
  blocks reordered at random within each regime (block edges start at a random offset), 5,000 draws,
  p = (1 + #{|T* - mean T*| >= |T - mean T*|}) / (1 + draws). This keeps the day-to-day persistence of the
  gamma state, which a day-by-day shuffle would destroy.
  Interval: 95% percentile interval from a regime-stratified moving-block bootstrap of whole days (blocks of
  20 trading days drawn within each regime), 2,000 draws.
  Hypothesis from hedging theory: the difference is positive. Significance level 0.05, two-sided.

SECONDARY FAMILY (Holm correction across all 13):
  S1 Pro, Nifty to Bank Nifty (share of Bank Nifty's forecast error variance from Nifty shocks)
  S2 Pro, Bank Nifty to Nifty
  S3 US convention rule, total;  S4 Client-mirror rule, total
  S5 gross gamma, top minus bottom tercile, total (a sign-free control)
  S6 Pro, total, stratified by regime x VIX tercile (VIX at the previous close; terciles over the sample)
  S7 Pro, day-level regression: per-day connectedness (one VAR(2) per day, H = 10) on the short-gamma
     dummy, log VIX at the previous close, regime dummies and a Nifty expiry-day dummy; Newey-West
     (5 lags) standard error shown, permutation p used for Holm
  S8 Pro, Nifty expiry days only;  S9 Pro, all other days
  S10 to S12 Pro within each regime (multi_weekly, single_weekly_thu, single_weekly_tue)
  S13 Pro, sensitivity setting: VAR lag chosen by AIC (maximum 10) on all sample days pooled, horizon 30
  Each uses the same permutation scheme and bootstrap as the primary test.

ROBUSTNESS (reported, not corrected): pooled difference without stratification (with the within-regime and
the whole-sample block permutation), block lengths 1, 10 and 40 days, and the intraday mcsGARCH component
sqrt(q) as the input (no daily level and no time-of-day shape).

POST-HOC CHECKS (added after the first full run, because the gross-gamma control S5 was the only Holm
survivor and its top tercile is mostly Nifty expiry days; not pre-stated, not corrected, own seeds):
  E1 gross gamma, top minus bottom tercile, on days that are not Nifty expiry days;
  E2 Nifty expiry days minus other days (regime-stratified), to show the expiry-day difference directly;
  E3 gross gamma, top minus bottom tercile, stratified by regime x VIX tercile (is it the volatility level?).

Seeds: master seed 20261102. Permutation draw i of variant v uses numpy.random.default_rng([20261102, 1, v, i]),
bootstrap draw i uses default_rng([20261102, 2, v, i]), so results do not depend on the number of workers.

Usage (about 1.5 minutes on 4 cores):

    OMP_NUM_THREADS=1 python experiments/connectedness_gamma.py
    OMP_NUM_THREADS=1 python experiments/connectedness_gamma.py --quick     # 400 permutations, 200 bootstraps

Writes results/connectedness_gamma/*.csv (run_summary.csv holds the run-level numbers), fig_connectedness_gamma.png,
run_info.json and results/connectedness_gamma_stdout.txt.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex import replication as rp
from niftygex.connectedness import VarFit, connectedness_from_fit
from niftygex.volatility import McsGarch, daily_egarch_variance

RAW = Path("data/raw")
RUN = Path("results/pipeline_nse_v2")
OUT = Path("results/connectedness_gamma")
LOG = Path("results/connectedness_gamma_stdout.txt")
LABEL = "REAL DATA"
INDICES = ("nifty", "banknifty")
TRIM = "360_drop_last15"
SAMPLE_START = pd.Timestamp("2023-01-02")
AUCTION_START = pd.Timestamp("2026-08-03")
REGIMES = ("multi_weekly", "single_weekly_thu", "single_weekly_tue")
RULES = {"pro": "net_gex_participant", "us": "net_gex_us_convention", "client_mirror": "net_gex_client_mirror"}
SEED = 20261102
P_MAIN, H_MAIN = 2, 10
PMAX_AIC, H_SENS = 10, 30
BLOCK = 20
MIN_DAYS = 5
HAC_LAGS = 5
N_PERM, N_BOOT = 5000, 2000

# ---------------------------------------------------------------------------
# Small helpers (tested in tests/test_connectedness_gamma.py)
# ---------------------------------------------------------------------------


def gamma_states(panel: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Gamma state of each day in ``days``, known at the previous close.

    ``panel`` is indexed by publish date (the close at which the chain was observed) on the Nifty trading
    calendar. Each measure is shifted by one row of the panel (the previous trading day) and then read on
    ``days``, so a day missing from ``days`` (for example a day without Bank Nifty bars) never shifts the
    alignment of the next day.
    """
    out = pd.DataFrame(index=panel.index)
    for rule, col in RULES.items():
        out[f"share_{rule}"] = (panel[col] / panel["gross_gex"]).shift(1)
    out["log_gross_prev"] = np.log(panel["gross_gex"]).shift(1)
    out["publish_date_used"] = pd.Series(panel.index, index=panel.index).shift(1)
    return out.reindex(days)


def tercile_codes(x: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """0, 1, 2 for the bottom, middle and top tercile of ``x`` (within each group if given); NaN stays NaN."""
    if groups is None:
        return pd.qcut(x, 3, labels=False).astype(float)
    out = pd.Series(np.nan, index=x.index)
    for g in pd.unique(groups):
        m = groups == g
        out[m] = pd.qcut(x[m], 3, labels=False).astype(float)
    return out


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN entries are left out and stay NaN)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * p[i]))
        out[i] = running
    return out


def block_permutation(groups: np.ndarray, block: int, rng) -> np.ndarray:
    """A permutation of positions 0..n-1 that reorders blocks of consecutive positions within each group.

    ``groups`` must label contiguous runs (here: expiry regimes in time order). In each group the block
    edges start at a random offset, the blocks are put in a random order, and position i receives the
    label that was at position perm[i]. With block = 1 this is an ordinary shuffle within groups.
    """
    groups = np.asarray(groups)
    n = len(groups)
    perm = np.arange(n)
    for g in pd.unique(groups):
        idx = np.where(groups == g)[0]
        m = len(idx)
        if m <= 1:
            continue
        if block <= 1:
            perm[idx] = idx[rng.permutation(m)]
            continue
        off = int(rng.integers(0, min(block, m)))
        edges = [0] + list(range(off if off > 0 else block, m, block)) + [m]
        edges = sorted(set(edges))
        pieces = [idx[a:b] for a, b in zip(edges[:-1], edges[1:])]
        order = rng.permutation(len(pieces))
        perm[idx] = np.concatenate([pieces[j] for j in order])
    return perm


def block_bootstrap_weights(groups: np.ndarray, block: int, rng) -> np.ndarray:
    """Day multiplicities from a moving-block bootstrap of blocks of consecutive days drawn within each group."""
    groups = np.asarray(groups)
    w = np.zeros(len(groups))
    for g in pd.unique(groups):
        idx = np.where(groups == g)[0]
        m = len(idx)
        L = max(1, min(block, m))
        nb = int(np.ceil(m / L))
        starts = rng.integers(0, m - L + 1, nb)
        pick = (starts[:, None] + np.arange(L)[None, :]).ravel()[:m]
        np.add.at(w, idx[pick], 1.0)
    return w


def centered_perm_p(real: float, draws: np.ndarray) -> tuple[float, float]:
    """Two-sided permutation p-value centred on the mean of the permutation distribution, and that mean."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if not np.isfinite(real) or len(d) == 0:
        return np.nan, np.nan
    m = float(d.mean())
    return float((1 + np.sum(np.abs(d - m) >= abs(real - m) - 1e-12)) / (1 + len(d))), m


# ---------------------------------------------------------------------------
# VAR connectedness from per-day cross products
# ---------------------------------------------------------------------------


@dataclass
class DayStats:
    """Per-day OLS cross products of a segment-aware VAR(p): sums over any set of days give the pooled fit.

    Pooling days in ``rp.fit_var_segments`` (each day its own segment, segments also broken at missing
    minutes) gives exactly X'X = sum of the days' X'X, and the same for X'Z and Z'Z. So a pooled VAR for any
    set of days, with any multiplicity (a bootstrap draw), costs one small linear solve.
    """

    p: int
    N: int
    F: np.ndarray  # (D, k*k + k*N + N*N + 1)

    @property
    def k(self) -> int:
        return 1 + self.p * self.N

    def sums(self, W: np.ndarray) -> np.ndarray:
        return np.atleast_2d(W) @ self.F

    def unpack(self, row: np.ndarray):
        k, N = self.k, self.N
        a = k * k
        b = a + k * N
        c = b + N * N
        return row[:a].reshape(k, k), row[a:b].reshape(k, N), row[b:c].reshape(N, N), row[c]


def day_stats(Yd: np.ndarray, p: int) -> DayStats:
    """Cross products for each day of ``Yd`` (days x minutes x N); rows with a NaN start a new segment."""
    D, M, N = Yd.shape
    k = 1 + p * N
    F = np.zeros((D, k * k + k * N + N * N + 1))
    for d in range(D):
        y = Yd[d]
        ok = np.all(np.isfinite(y), axis=1)
        if ok.sum() <= p:
            continue
        seg = np.cumsum(~ok)[ok]
        X, Z = rp.var_design(y[ok], seg, p)
        F[d] = np.concatenate([(X.T @ X).ravel(), (X.T @ Z).ravel(), (Z.T @ Z).ravel(), [len(Z)]])
    return DayStats(p=p, N=N, F=F)


def fit_from_sums(ds: DayStats, row: np.ndarray) -> VarFit | None:
    """The VAR that ``rp.fit_var_segments`` would give on the days whose cross products were summed into ``row``."""
    XtX, XtZ, ZtZ, n = ds.unpack(row)
    k, N, p = ds.k, ds.N, ds.p
    if n <= k + 1:
        return None
    try:
        B = np.linalg.solve(XtX, XtZ)
    except np.linalg.LinAlgError:
        return None
    UtU = ZtZ - XtZ.T @ B
    Sigma = (UtU + UtU.T) / 2 / max(n - k, 1)
    if not np.all(np.isfinite(Sigma)) or np.any(np.diag(Sigma) <= 0):
        return None
    A = np.stack([B[1 + (l - 1) * N : 1 + l * N].T for l in range(1, p + 1)])
    return VarFit(A=A, c=B[0], Sigma=Sigma, resid=np.empty((0, N)), p=p)


def conn_values(ds: DayStats, row: np.ndarray, H: int) -> np.ndarray:
    """[paper-scale total, Nifty from Bank Nifty (BN to Nifty), Bank Nifty from Nifty (Nifty to BN)]."""
    fit = fit_from_sums(ds, row)
    if fit is None:
        return np.full(3, np.nan)
    c = connectedness_from_fit(fit, H)
    return np.array([c.paper_scale_total, c.table[0, 1], c.table[1, 0]])


# ---------------------------------------------------------------------------
# Contrasts: a statistic = stratified difference between two states
# ---------------------------------------------------------------------------


@dataclass
class Contrast:
    name: str
    label_key: str  # which label vector (permuted together) defines the two states: 1 = state A, 0 = state B
    subset: np.ndarray  # bool (D,)
    strata: np.ndarray  # int (D,)
    stats_key: str  # which DayStats
    H: int
    state_a: str = "short gamma"
    state_b: str = "long gamma"
    included: list = field(default_factory=list)  # strata used (fixed from the real labels)
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_a: dict = field(default_factory=dict)
    n_b: dict = field(default_factory=dict)

    def setup(self, labels: dict, min_days: int = MIN_DAYS):
        lab = labels[self.label_key]
        inc, w = [], []
        for s in np.unique(self.strata[self.subset]):
            m = self.subset & (self.strata == s)
            na, nb = int(np.sum(m & (lab == 1))), int(np.sum(m & (lab == 0)))
            self.n_a[int(s)], self.n_b[int(s)] = na, nb
            if na >= min_days and nb >= min_days:
                inc.append(int(s))
                w.append(na + nb)
        self.included = inc
        self.weights = np.asarray(w, float) / max(sum(w), 1)
        return self

    def cell_masks(self, lab: np.ndarray, w: np.ndarray) -> np.ndarray:
        """(2 * strata, D) weight rows: stratum s state A, stratum s state B, ..."""
        rows = []
        for s in self.included:
            m = self.subset & (self.strata == s)
            rows.append(w * (m & (lab == 1)))
            rows.append(w * (m & (lab == 0)))
        return np.array(rows) if rows else np.zeros((0, len(w)))

    def evaluate(self, stats: dict, labels: dict, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stratified differences [total, BN to Nifty, Nifty to BN] and the per-cell values (strata x 2 x 3)."""
        if not self.included:
            return np.full(3, np.nan), np.zeros((0, 2, 3))
        ds = stats[self.stats_key]
        W = self.cell_masks(labels[self.label_key], w)
        S = ds.sums(W)
        vals = np.array([conn_values(ds, S[i], self.H) for i in range(len(S))]).reshape(len(self.included), 2, 3)
        diff = np.einsum("s,sk->k", self.weights, vals[:, 0, :] - vals[:, 1, :])
        return diff, vals


# ---------------------------------------------------------------------------
# Parallel draws (globals are set before the pool forks)
# ---------------------------------------------------------------------------

_G: dict = {}


def _apply_perm(labels: dict, perm: np.ndarray) -> dict:
    return {k: v[perm] for k, v in labels.items()}


def _reg_coef(y: np.ndarray, Xo: np.ndarray, dummy: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), dummy, Xo])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(b[1])


def _draw_task(args):
    kind, variant, block, group_key, i0, i1, names, with_reg = args
    contrasts = [c for c in _G["contrasts"] if c.name in names]
    groups = _G["groups"][group_key]
    D = len(groups)
    out = []
    for i in range(i0, i1):
        if kind == "perm":
            rng = np.random.default_rng([SEED, 1, variant, i])
            lab = _apply_perm(_G["labels"], block_permutation(groups, block, rng))
            w = np.ones(D)
        else:
            rng = np.random.default_rng([SEED, 2, variant, i])
            lab = _G["labels"]
            w = block_bootstrap_weights(groups, block, rng)
        rec = {}
        for c in contrasts:
            diff, vals = c.evaluate(_G["stats"], lab, w)
            rec[c.name] = (diff, vals if kind == "boot" else None)
        if with_reg:
            r = _G["reg"]
            d = lab["pro"].astype(float)
            ok = r["ok"]
            rec["__reg_vix"] = _reg_coef(r["y"][ok], r["X_vix"][ok], d[ok])
            rec["__reg_novix"] = _reg_coef(r["y"][ok], r["X_novix"][ok], d[ok])
        out.append(rec)
    return out


def run_draws(pool, kind, variant, block, group_key, n, names, with_reg=False, chunk=100):
    tasks = [(kind, variant, block, group_key, a, min(a + chunk, n), names, with_reg) for a in range(0, n, chunk)]
    res = pool.map(_draw_task, tasks) if pool is not None else map(_draw_task, tasks)
    flat = [r for part in res for r in part]
    out = {}
    for nm in names:
        out[nm] = {"diff": np.array([r[nm][0] for r in flat])}
        if kind == "boot":
            out[nm]["vals"] = np.array([r[nm][1] for r in flat])
    if with_reg:
        out["__reg_vix"] = np.array([r["__reg_vix"] for r in flat])
        out["__reg_novix"] = np.array([r["__reg_novix"] for r in flat])
    return out


# ---------------------------------------------------------------------------
# Data and volatility inputs (small helpers copied from experiments/replicate_paper.py)
# ---------------------------------------------------------------------------


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def build_grids(bars: dict, trim: str) -> dict:
    """(days x minutes) grids on the days where both indices have at least 300 bars (as in replicate_paper.py)."""
    grids = {k: rp.minute_grid(bars[k], trim) for k in INDICES}
    days = None
    for k in INDICES:
        ok = pd.DatetimeIndex(grids[k]["days"][np.isfinite(grids[k]["close"]).sum(axis=1) >= 300])
        days = ok if days is None else days.intersection(ok)
    out = {"days": days, "minutes": grids["nifty"]["minutes"]}
    for k in INDICES:
        g = rp.minute_grid(bars[k], trim, days=days)
        r = rp.continuous_log_returns(g["close"], overnight=True)
        last = pd.Series(np.log(g["close"][:, ::-1]).tolist()).map(lambda row: next((x for x in row if np.isfinite(x)), np.nan))
        out[k] = {"r": r, "daily": np.r_[np.nan, np.diff(last.to_numpy())]}
    return out


def fit_regime(G: dict, k: str, rows: np.ndarray) -> dict:
    """mcsGARCH on one index over one regime (the replication's fit_period, without t statistics)."""
    R = G[k]["r"][rows].copy()
    R[0, 0] = np.nan
    keep = np.isfinite(R).any(axis=0)
    last = int(rows[-1])
    y = G[k]["daily"][1 : last + 1]
    hh, info = daily_egarch_variance(y)
    hf = np.full(len(G["days"]), np.nan)
    hf[1 : last + 1] = hh
    h = hf[rows]
    mg = McsGarch(mean="arma11", dist="t", smooth=1).fit(R[:, keep], h)
    filt = mg.filter(R[:, keep], h)
    cond, intra = np.full(R.shape, np.nan), np.full(R.shape, np.nan)
    cond[:, keep] = np.sqrt(h[:, None] * filt["S"] * filt["q2"])
    intra[:, keep] = np.sqrt(filt["q2"])
    valid = np.isfinite(R)
    return {"cond_vol": np.where(valid, cond, np.nan), "intraday_vol": np.where(valid, intra, np.nan),
            "params": mg.params, "daily_method": info["method"]}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _fmt(df: pd.DataFrame, cols=None) -> str:
    d = df[cols] if cols else df
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 60):
        return d.to_string(index=False, float_format=lambda x: f"{x:.4g}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    n_perm, n_boot = (400, 200) if a.quick else (N_PERM, N_BOOT)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(LOG)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    timing = {}

    print(f"{LABEL}. RQ4 (P42): Nifty / Bank Nifty volatility connectedness by dealer gamma state.")
    print(f"Seed {SEED}; permutations {n_perm}; bootstrap draws {n_boot}; block {BLOCK} trading days; "
          f"VAR({P_MAIN}), H={H_MAIN}; sensitivity VAR(AIC, max {PMAX_AIC}), H={H_SENS}; min days per state per stratum {MIN_DAYS}.")

    # -------------------------------------------------------------- data
    bars = {k: rp.load_minute_bars(RAW / f"{k}_1min.csv", start="2017-01-01") for k in INDICES}
    G = build_grids(bars, TRIM)
    dm = pd.read_csv(RUN / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(RUN / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    bn_last = bars["banknifty"]["date"].max()
    timing["load_s"] = time.time() - t0

    cand = G["days"][(G["days"] >= SAMPLE_START) & (G["days"] <= bn_last) & G["days"].isin(dm.index)]
    st = gamma_states(panel, cand)
    vix_prev = np.log(dm["vix"]).shift(1).reindex(cand)
    keep = st[[f"share_{r}" for r in RULES] + ["log_gross_prev"]].notna().all(axis=1) & vix_prev.notna()
    days = cand[keep.to_numpy()]
    st, vix_prev = st.loc[days], vix_prev.loc[days]
    dmd = dm.loc[days]
    regime = dmd["regime"].astype(str)
    assert set(regime) <= set(REGIMES), set(regime)
    reg_code = regime.map({r: i for i, r in enumerate(REGIMES)}).to_numpy()
    assert np.all(np.diff(reg_code) >= 0), "regimes must be contiguous in time"
    n_auction = int((days >= AUCTION_START).sum())
    print(f"\nSample: {days.min().date()} to {days.max().date()}, {len(days)} days (Bank Nifty bars end {bn_last.date()}); "
          f"Nifty expiry days {int(dmd['is_expiry'].sum())}; closing-auction days (from {AUCTION_START.date()}) in sample: {n_auction}.")
    print("Days per regime: " + ", ".join(f"{r} {int((regime == r).sum())}" for r in REGIMES))

    # -------------------------------------------------------------- volatility inputs, per regime
    t1 = time.time()
    rowpos = {d: i for i, d in enumerate(G["days"])}
    cond = {k: np.full((len(days), len(G["minutes"])), np.nan) for k in INDICES}
    intra = {k: np.full_like(cond[k], np.nan) for k in INDICES}
    fit_rows = []
    for r in REGIMES:
        sel = np.where(regime.to_numpy() == r)[0]
        rd = days[sel]
        # fit on every trading day of the regime inside the sample window (also days dropped for missing states)
        span = cand[(cand >= rd.min()) & (cand <= rd.max())]
        span = span[dm.loc[span, "regime"].astype(str).to_numpy() == r]
        rows = np.array([rowpos[d] for d in span])
        where = np.searchsorted(span, rd)
        for k in INDICES:
            f = fit_regime(G, k, rows)
            cond[k][sel] = f["cond_vol"][where]
            intra[k][sel] = f["intraday_vol"][where]
            p = f["params"]
            fit_rows.append({"data": LABEL, "regime": r, "index": k, "days_fitted": len(rows), "alpha": p["alpha"], "beta": p["beta"],
                             "persistence": p["alpha"] + p["beta"], "half_life_min": float(rp.half_life(p["alpha"] + p["beta"])),
                             "nu": p["nu"], "loglik_scaled": p["loglik"], "daily_variance_method": f["daily_method"]})
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(OUT / "mcsgarch_fits.csv", index=False)
    timing["mcsgarch_s"] = time.time() - t1
    print("\nmcsGARCH per regime (mcsgarch_fits.csv):")
    print(_fmt(fits, ["regime", "index", "days_fitted", "alpha", "beta", "half_life_min", "nu", "daily_variance_method"]))

    # scale each series by one global constant (connectedness does not change; it helps the linear algebra)
    def stack3(src):
        sc = {k: np.nanmedian(src[k]) for k in INDICES}
        return np.stack([src[k] / sc[k] for k in INDICES], axis=2)

    Y_main, Y_intra = stack3(cond), stack3(intra)

    # -------------------------------------------------------------- labels
    labels = {rule: (st[f"share_{rule}"].to_numpy() < 0).astype(int) for rule in RULES}
    gross_t = tercile_codes(st["log_gross_prev"], regime).to_numpy()
    labels["gross"] = np.where(gross_t == 2, 1, np.where(gross_t == 0, 0, -1))
    labels["expiry"] = dmd["is_expiry"].astype(int).to_numpy()  # used only by the post-hoc check E2
    vix_t = tercile_codes(vix_prev).to_numpy().astype(int)
    is_exp = dmd["is_expiry"].astype(bool).to_numpy()
    all_days = np.ones(len(days), bool)

    # -------------------------------------------------------------- VAR cross products
    t2 = time.time()
    Ys, segs = [], []
    for d in range(len(days)):
        y = Y_main[d]
        ok = np.all(np.isfinite(y), axis=1)
        Ys.append(y[ok])
        segs.append(d * 1_000_000 + np.cumsum(~ok)[ok])
    p_aic = rp.var_lag_aic(np.vstack(Ys), np.concatenate(segs), PMAX_AIC)
    stats = {"main": day_stats(Y_main, P_MAIN), "sens": day_stats(Y_main, p_aic), "intra": day_stats(Y_intra, P_MAIN)}
    timing["cross_products_s"] = time.time() - t2
    print(f"\nVAR lag chosen by AIC (max {PMAX_AIC}) on all {len(days)} days pooled: p = {p_aic} (used with H = {H_SENS}).")

    # check: the fast pooled fit equals the replication's connectedness_segments on the stacked raw data
    m_chk = labels["pro"] == 1
    raw3 = np.stack([cond[k] for k in INDICES], axis=2)
    Yc, sc_ = [], []
    for d in np.where(m_chk)[0]:
        ok = np.all(np.isfinite(raw3[d]), axis=1)
        Yc.append(raw3[d][ok])
        sc_.append(d * 1_000_000 + np.cumsum(~ok)[ok])
    ref = rp.connectedness_segments(np.vstack(Yc), np.concatenate(sc_), P_MAIN, H_MAIN)
    fast = conn_values(stats["main"], stats["main"].sums(m_chk.astype(float))[0], H_MAIN)
    chk = float(np.max(np.abs(fast - np.array([ref.paper_scale_total, ref.table[0, 1], ref.table[1, 0]]))))
    print(f"Check: pooled short-gamma (Pro) days, fast cross-product VAR vs replication.connectedness_segments: "
          f"largest absolute difference {chk:.2e} connectedness points.")

    # -------------------------------------------------------------- contrasts
    rs, vs = reg_code, reg_code * 3 + vix_t
    C = [
        Contrast("P0_pro_total", "pro", all_days, rs, "main", H_MAIN),
        Contrast("S3_us_total", "us", all_days, rs, "main", H_MAIN),
        Contrast("S4_client_mirror_total", "client_mirror", all_days, rs, "main", H_MAIN),
        Contrast("S5_gross_top_vs_bottom", "gross", all_days, rs, "main", H_MAIN, "top gross tercile", "bottom gross tercile"),
        Contrast("S6_pro_vix_stratified", "pro", all_days, vs, "main", H_MAIN),
        Contrast("S8_pro_expiry_days", "pro", is_exp, rs, "main", H_MAIN),
        Contrast("S9_pro_other_days", "pro", ~is_exp, rs, "main", H_MAIN),
        Contrast("S10_pro_multi_weekly", "pro", reg_code == 0, rs, "main", H_MAIN),
        Contrast("S11_pro_single_weekly_thu", "pro", reg_code == 1, rs, "main", H_MAIN),
        Contrast("S12_pro_single_weekly_tue", "pro", reg_code == 2, rs, "main", H_MAIN),
        Contrast("S13_pro_sensitivity", "pro", all_days, rs, "sens", H_SENS),
        Contrast("R_pro_pooled", "pro", all_days, np.zeros(len(days), int), "main", H_MAIN),
        Contrast("R_pro_intraday_input", "pro", all_days, rs, "intra", H_MAIN),
    ]
    # POST-HOC checks, added after the first full run (not pre-stated, not in any family, uncorrected)
    E = [
        Contrast("E1_gross_top_vs_bottom_other_days", "gross", ~is_exp, rs, "main", H_MAIN, "top gross tercile", "bottom gross tercile"),
        Contrast("E2_expiry_vs_other_days", "expiry", all_days, rs, "main", H_MAIN, "Nifty expiry day", "other day"),
        Contrast("E3_gross_top_vs_bottom_vix_stratified", "gross", all_days, vs, "main", H_MAIN, "top gross tercile", "bottom gross tercile"),
    ]
    for c in E:
        c.setup(labels)
    for c in C:
        c.setup(labels)
    names_main = [c.name for c in C]
    real = {c.name: c.evaluate(stats, labels, np.ones(len(days))) for c in C}

    # -------------------------------------------------------------- day-level proxies and regression
    import statsmodels.api as sm

    ds = stats["main"]
    per_day = np.array([conn_values(ds, ds.F[d], H_MAIN) for d in range(len(days))])
    reg_d = (reg_code[:, None] == np.array([1, 2])[None, :]).astype(float)
    X_novix = np.column_stack([reg_d, is_exp.astype(float)])
    X_vix = np.column_stack([X_novix, vix_prev.to_numpy()])
    y_day = per_day[:, 0]
    ok_reg = np.isfinite(y_day)
    _G.update({"labels": labels, "stats": stats, "contrasts": C + E,
               "groups": {"regime": reg_code, "whole": np.zeros(len(days), int)},
               "reg": {"y": y_day, "X_vix": X_vix, "X_novix": X_novix, "ok": ok_reg}})
    reg_rows = []
    for spec, Xo in (("regime + expiry dummies", X_novix), ("regime + expiry dummies + log VIX(t-1)", X_vix)):
        X = sm.add_constant(np.column_stack([labels["pro"].astype(float), Xo])[ok_reg])
        res = sm.OLS(y_day[ok_reg], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        reg_rows.append({"data": LABEL, "outcome": "per-day total connectedness (paper scale), VAR(2), H=10, cond_vol",
                         "spec": spec, "n_days": int(ok_reg.sum()), "coef_short_gamma_pro": res.params[1],
                         "hac_se": res.bse[1], "hac_p": res.pvalues[1], "ci95_lo": res.conf_int()[1, 0], "ci95_hi": res.conf_int()[1, 1],
                         "coef_log_vix_prev": res.params[-1] if "VIX" in spec else np.nan,
                         "hac_p_log_vix_prev": res.pvalues[-1] if "VIX" in spec else np.nan, "r2": res.rsquared,
                         "mean_outcome": float(np.mean(y_day[ok_reg]))})

    # -------------------------------------------------------------- permutation and bootstrap draws
    t3 = time.time()
    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers) if a.workers > 1 else None
    try:
        perm = run_draws(pool, "perm", 0, BLOCK, "regime", n_perm, names_main, with_reg=True)
        boot = run_draws(pool, "boot", 0, BLOCK, "regime", n_boot, names_main)
        rob = {}
        for v, blk in ((1, 1), (2, 10), (3, 40)):
            rob[("perm", blk)] = run_draws(pool, "perm", v, blk, "regime", n_perm, ["P0_pro_total"])
            rob[("boot", blk)] = run_draws(pool, "boot", v, blk, "regime", n_boot, ["P0_pro_total"])
        rob[("perm_whole", BLOCK)] = run_draws(pool, "perm", 4, BLOCK, "whole", n_perm, ["R_pro_pooled"])
        # the post-hoc checks get their own seeds (variant 5), so nothing above changes
        eperm = run_draws(pool, "perm", 5, BLOCK, "regime", n_perm, [c.name for c in E])
        eboot = run_draws(pool, "boot", 5, BLOCK, "regime", n_boot, [c.name for c in E])
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    timing["draws_s"] = time.time() - t3

    # -------------------------------------------------------------- tables
    def summarise(name, k, perm_draws, boot_draws, real_val):
        p, mnull = centered_perm_p(real_val, perm_draws)
        b = boot_draws[np.isfinite(boot_draws)]
        return {"estimate": real_val, "boot_ci95_lo": np.percentile(b, 2.5) if len(b) else np.nan,
                "boot_ci95_hi": np.percentile(b, 97.5) if len(b) else np.nan, "boot_median": np.median(b) if len(b) else np.nan,
                "perm_p": p, "perm_null_mean": mnull,
                "perm_draws_used": int(np.isfinite(perm_draws).sum()), "boot_draws_used": int(len(b))}

    cmap = {c.name: c for c in C}
    test_spec = [
        ("P0", "primary", "P0_pro_total", 0, "Pro rule, all days: total connectedness, short minus long gamma (regime-stratified)"),
        ("S1", "secondary", "P0_pro_total", 2, "Pro rule, all days: Nifty to Bank Nifty share, short minus long"),
        ("S2", "secondary", "P0_pro_total", 1, "Pro rule, all days: Bank Nifty to Nifty share, short minus long"),
        ("S3", "secondary", "S3_us_total", 0, "US convention rule: total, short minus long"),
        ("S4", "secondary", "S4_client_mirror_total", 0, "Client-mirror rule: total, short minus long"),
        ("S5", "secondary", "S5_gross_top_vs_bottom", 0, "Gross gamma (no sign): total, top minus bottom tercile"),
        ("S6", "secondary", "S6_pro_vix_stratified", 0, "Pro rule: total, stratified by regime x VIX tercile"),
        ("S7", "secondary", "__reg_vix", 0, "Pro rule: per-day connectedness on short dummy + log VIX(t-1) + regime + expiry"),
        ("S8", "secondary", "S8_pro_expiry_days", 0, "Pro rule, Nifty expiry days: total, short minus long"),
        ("S9", "secondary", "S9_pro_other_days", 0, "Pro rule, other days: total, short minus long"),
        ("S10", "secondary", "S10_pro_multi_weekly", 0, "Pro rule, multi_weekly regime: total, short minus long"),
        ("S11", "secondary", "S11_pro_single_weekly_thu", 0, "Pro rule, single_weekly_thu regime: total, short minus long"),
        ("S12", "secondary", "S12_pro_single_weekly_tue", 0, "Pro rule, single_weekly_tue regime: total, short minus long"),
        ("S13", "secondary", "S13_pro_sensitivity", 0, f"Pro rule, VAR({p_aic}) by AIC, H={H_SENS}: total, short minus long"),
    ]
    trows = []
    for tid, fam, nm, j, what in test_spec:
        if nm == "__reg_vix":
            rr = reg_rows[1]
            p, mnull = centered_perm_p(rr["coef_short_gamma_pro"], perm["__reg_vix"])
            rec = {"estimate": rr["coef_short_gamma_pro"], "boot_ci95_lo": np.nan, "boot_ci95_hi": np.nan,
                   "hac_ci95_lo": rr["ci95_lo"], "hac_ci95_hi": rr["ci95_hi"], "hac_p": rr["hac_p"], "perm_p": p,
                   "perm_null_mean": mnull, "perm_draws_used": int(np.isfinite(perm["__reg_vix"]).sum()), "boot_draws_used": 0,
                   "strata_used": "all days (regression)", "n_days_state_a": int(np.sum(labels["pro"][ok_reg] == 1)),
                   "n_days_state_b": int(np.sum(labels["pro"][ok_reg] == 0))}
        else:
            c = cmap[nm]
            rec = summarise(nm, j, perm[nm]["diff"][:, j], boot[nm]["diff"][:, j], real[nm][0][j])
            rec.update({"strata_used": ";".join(_stratum_name(c, s) for s in c.included),
                        "n_days_state_a": int(sum(c.n_a[s] for s in c.included)), "n_days_state_b": int(sum(c.n_b[s] for s in c.included)),
                        "n_days_dropped_small_strata": int(sum(c.n_a[s] + c.n_b[s] for s in c.n_a if s not in c.included))})
        rec = {"data": LABEL, "test": tid, "family": fam, "contrast": what, "measure": ["total", "bn_to_nifty", "nifty_to_bn"][j], **rec}
        trows.append(rec)
    tests = pd.DataFrame(trows)
    sec = tests["family"] == "secondary"
    tests["holm_p_secondary"] = np.nan
    tests.loc[sec, "holm_p_secondary"] = holm(tests.loc[sec, "perm_p"].to_numpy())
    tests.to_csv(OUT / "tests.csv", index=False)

    # levels per state and stratum, with bootstrap intervals
    lrows = []
    for c in C:
        vals_real = real[c.name][1]
        vb = boot[c.name]["vals"]
        for si, s in enumerate(c.included):
            for a_i, (state, n_s) in enumerate(((c.state_a, c.n_a[s]), (c.state_b, c.n_b[s]))):
                for j, meas in enumerate(("total", "bn_to_nifty", "nifty_to_bn")):
                    x = vb[:, si, a_i, j]
                    x = x[np.isfinite(x)]
                    lrows.append({"data": LABEL, "contrast": c.name, "stratum": _stratum_name(c, s), "stratum_weight": c.weights[si],
                                  "state": state, "n_days": n_s, "measure": meas, "value": vals_real[si, a_i, j],
                                  "boot_ci95_lo": np.percentile(x, 2.5) if len(x) else np.nan,
                                  "boot_ci95_hi": np.percentile(x, 97.5) if len(x) else np.nan})
    levels = pd.DataFrame(lrows)
    levels.to_csv(OUT / "state_connectedness.csv", index=False)

    # robustness
    rrows = []
    base = real["P0_pro_total"][0][0]
    for blk in (1, 10, 40):
        rec = summarise("P0", 0, rob[("perm", blk)]["P0_pro_total"]["diff"][:, 0], rob[("boot", blk)]["P0_pro_total"]["diff"][:, 0], base)
        rrows.append({"data": LABEL, "check": f"primary statistic, block length {blk} day(s)", **rec})
    for nm, what, pd_ in (("R_pro_pooled", "pooled difference (no stratification), within-regime block permutation", perm["R_pro_pooled"]["diff"][:, 0]),
                          ("R_pro_pooled", "pooled difference (no stratification), whole-sample block permutation", rob[("perm_whole", BLOCK)]["R_pro_pooled"]["diff"][:, 0]),
                          ("R_pro_intraday_input", "primary statistic on the intraday mcsGARCH component sqrt(q)", perm["R_pro_intraday_input"]["diff"][:, 0])):
        rec = summarise(nm, 0, pd_, boot[nm]["diff"][:, 0], real[nm][0][0])
        rrows.append({"data": LABEL, "check": what, **rec})
    rrows.append({"data": LABEL, "check": f"closing-auction days (from {AUCTION_START.date()}) in the sample", "estimate": float(n_auction)})
    robust = pd.DataFrame(rrows)
    robust.to_csv(OUT / "robustness.csv", index=False)

    erows = []
    for c in E:
        rv = c.evaluate(stats, labels, np.ones(len(days)))
        rec = summarise(c.name, 0, eperm[c.name]["diff"][:, 0], eboot[c.name]["diff"][:, 0], rv[0][0])
        lv_ = {f"value_{st_}_{_stratum_name(c, s_)}": rv[1][si, ai, 0] for si, s_ in enumerate(c.included)
               for ai, st_ in enumerate(("a", "b"))}
        erows.append({"data": LABEL, "status": "POST-HOC (added after the first full run; not pre-stated, uncorrected)",
                      "check": c.name, "state_a": c.state_a, "state_b": c.state_b,
                      "n_days_state_a": int(sum(c.n_a[s_] for s_ in c.included)), "n_days_state_b": int(sum(c.n_b[s_] for s_ in c.included)),
                      **rec, **lv_})
    explo = pd.DataFrame(erows)
    explo.to_csv(OUT / "posthoc_checks.csv", index=False)

    regdf = pd.DataFrame(reg_rows)
    for i, key in enumerate(("__reg_novix", "__reg_vix")):
        regdf.loc[i, "perm_p"], regdf.loc[i, "perm_null_mean"] = centered_perm_p(regdf.loc[i, "coef_short_gamma_pro"], perm[key])
    regdf.to_csv(OUT / "day_level_regression.csv", index=False)

    # per-day file
    pdf = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "regime": regime.to_numpy(), "is_nifty_expiry": is_exp,
                        "gex_publish_date_used": pd.DatetimeIndex(st["publish_date_used"]).strftime("%Y-%m-%d"),
                        "log_vix_prev": vix_prev.to_numpy(), "vix_tercile": vix_t,
                        **{f"share_{r}_prev": st[f"share_{r}"].to_numpy() for r in RULES},
                        **{f"short_gamma_{r}": labels[r] for r in RULES},
                        "log_gross_gex_prev": st["log_gross_prev"].to_numpy(), "gross_tercile_within_regime": gross_t,
                        "trading_days_to_expiry": dmd["trading_days_to_expiry"].to_numpy(),
                        "n_minutes_both_valid": np.isfinite(Y_main).all(axis=2).sum(axis=1),
                        "day_total_conn": per_day[:, 0], "day_bn_to_nifty": per_day[:, 1], "day_nifty_to_bn": per_day[:, 2]})
    pdf.insert(0, "data", LABEL)
    pdf.to_csv(OUT / "sample_days.csv", index=False)

    # counts table
    crow = []
    for r_i, r in enumerate(REGIMES + ("all",)):
        m = all_days if r == "all" else reg_code == r_i
        for e_name, e_m in (("all days", all_days), ("Nifty expiry days", is_exp), ("other days", ~is_exp)):
            mm = m & e_m
            rec = {"data": LABEL, "regime": r, "days": e_name, "n_days": int(mm.sum())}
            for rule in RULES:
                rec[f"short_{rule}"] = int(np.sum(labels[rule][mm] == 1))
            crow.append(rec)
    counts = pd.DataFrame(crow)
    counts.to_csv(OUT / "state_counts.csv", index=False)
    # what the gross-gamma terciles pick out: days to the next Nifty expiry, and overlap with the Pro state
    comp = pdf.assign(gross_tercile=pdf["gross_tercile_within_regime"].map({0.0: "bottom", 1.0: "middle", 2.0: "top"}))
    gcomp = comp.groupby("gross_tercile").agg(n_days=("date", "size"), share_nifty_expiry_day=("is_nifty_expiry", "mean"),
                                              mean_trading_days_to_expiry=("trading_days_to_expiry", "mean"),
                                              share_short_gamma_pro=("short_gamma_pro", "mean"),
                                              mean_day_total_conn=("day_total_conn", "mean")).reset_index()
    gcomp.insert(0, "data", LABEL)
    gcomp.to_csv(OUT / "gross_tercile_composition.csv", index=False)

    timing["total_s"] = time.time() - t0
    info = {"data": LABEL, "seed": SEED, "perm_seed_rule": "default_rng([20261102, 1, variant, draw])",
            "boot_seed_rule": "default_rng([20261102, 2, variant, draw])",
            "variants": {"0": f"main draws, block {BLOCK}", "1": "block 1", "2": "block 10", "3": "block 40", "4": "whole-sample permutation",
                         "5": "post-hoc checks E1 to E3"},
            "n_perm": n_perm, "n_boot": n_boot, "block_days": BLOCK, "min_days_per_state_per_stratum": MIN_DAYS,
            "var_p_main": P_MAIN, "H_main": H_MAIN, "var_p_sensitivity_aic": int(p_aic), "H_sensitivity": H_SENS,
            "session_rule": TRIM, "sample_start": str(days.min().date()), "sample_end": str(days.max().date()), "n_days": int(len(days)),
            "closing_auction_days_in_sample": n_auction, "fast_vs_replication_check_max_abs_diff": chk,
            "workers": a.workers, "timing_seconds": {k: round(v, 1) for k, v in timing.items()},
            "input_sha256_16": {p.name: _sha(p) for p in (RUN / "daily_market.csv", RUN / "daily_gex_panel.csv",
                                                          RAW / "nifty_1min.csv", RAW / "banknifty_1min.csv")}}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    flat = {k: v for k, v in info.items() if not isinstance(v, dict)}
    flat.update({f"timing_{k}": round(v, 1) for k, v in timing.items()})
    flat.update({f"n_days_{r}": int((regime == r).sum()) for r in REGIMES})
    flat["n_nifty_expiry_days"] = int(is_exp.sum())
    pd.DataFrame({"key": list(flat), "value": [str(v) for v in flat.values()]}).to_csv(OUT / "run_summary.csv", index=False)

    # -------------------------------------------------------------- print
    print("\nGamma-state counts (state_counts.csv):")
    print(_fmt(counts))
    print("\nConnectedness per state and stratum, total (state_connectedness.csv, paper scale 0 to 200):")
    lv = levels[(levels["measure"] == "total") & levels["contrast"].isin(["P0_pro_total", "R_pro_pooled", "S3_us_total", "S4_client_mirror_total", "S5_gross_top_vs_bottom"])]
    print(_fmt(lv, ["contrast", "stratum", "stratum_weight", "state", "n_days", "value", "boot_ci95_lo", "boot_ci95_hi"]))
    print("\nDirectional shares, Pro rule (state_connectedness.csv):")
    lv = levels[(levels["contrast"] == "P0_pro_total") & (levels["measure"] != "total")]
    print(_fmt(lv, ["stratum", "state", "n_days", "measure", "value", "boot_ci95_lo", "boot_ci95_hi"]))
    print("\nPrimary and secondary tests (tests.csv). estimate = state A minus state B; p two-sided:")
    print(_fmt(tests, ["test", "measure", "contrast", "estimate", "boot_ci95_lo", "boot_ci95_hi", "boot_median", "perm_p", "perm_null_mean", "holm_p_secondary",
                       "n_days_state_a", "n_days_state_b"]))
    print("\nDay-level regression (day_level_regression.csv):")
    print(_fmt(regdf, ["spec", "n_days", "coef_short_gamma_pro", "hac_se", "hac_p", "perm_p", "coef_log_vix_prev", "hac_p_log_vix_prev", "r2", "mean_outcome"]))
    print("\nRobustness (robustness.csv):")
    print(_fmt(robust, ["check", "estimate", "boot_ci95_lo", "boot_ci95_hi", "boot_median", "perm_p", "perm_null_mean"]))
    print("\nPOST-HOC checks, not pre-stated (posthoc_checks.csv):")
    print(_fmt(explo, ["check", "n_days_state_a", "n_days_state_b", "estimate", "boot_ci95_lo", "boot_ci95_hi", "boot_median", "perm_p", "perm_null_mean"]))
    print("\nWhat the gross-gamma terciles pick out (gross_tercile_composition.csv):")
    print(_fmt(gcomp.drop(columns="data")))
    print("\nVIX-stratified cells, Pro rule (state_connectedness.csv, contrast S6):")
    lv = levels[(levels["contrast"] == "S6_pro_vix_stratified") & (levels["measure"] == "total")]
    print(_fmt(lv, ["stratum", "stratum_weight", "state", "n_days", "value", "boot_ci95_lo", "boot_ci95_hi"]))

    make_figure(levels, tests)
    print(f"\nTiming (seconds): {json.dumps({k: round(v, 1) for k, v in timing.items()})}")
    sys.stdout.flush()


def _stratum_name(c: Contrast, s: int) -> str:
    if c.name in ("S6_pro_vix_stratified", "E3_gross_top_vs_bottom_vix_stratified"):
        return f"{REGIMES[s // 3]} | VIX tercile {s % 3 + 1}"
    if c.name == "R_pro_pooled":
        return "all regimes pooled"
    return REGIMES[s]


def make_figure(levels: pd.DataFrame, tests: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#d9d8d4"
    blue, orange = "#2a78d6", "#eb6834"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.35]})
    lv = levels[(levels["contrast"] == "P0_pro_total") & (levels["measure"] == "total")]
    strata = list(dict.fromkeys(lv["stratum"]))
    for si, s in enumerate(strata):
        for off, (state, col) in zip((-0.12, 0.12), (("short gamma", blue), ("long gamma", orange))):
            r = lv[(lv["stratum"] == s) & (lv["state"] == state)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax1.plot([si + off, si + off], [r["boot_ci95_lo"], r["boot_ci95_hi"]], color=col, lw=2, solid_capstyle="round")
            ax1.plot(si + off, r["value"], "o", ms=8, color=col, markeredgecolor="white", markeredgewidth=1.5,
                     label=state if si == 0 else None)
            ax1.annotate(f"{r['n_days']} d", (si + off, r["boot_ci95_lo"]), textcoords="offset points", xytext=(0, -12),
                         ha="center", fontsize=8, color=muted)
    ax1.set_xticks(range(len(strata)))
    ax1.set_xticklabels([s.replace("_", " ") for s in strata], fontsize=9, color=ink)
    ax1.set_ylabel("Total connectedness (paper scale, 0 to 200)", color=ink)
    ax1.set_title("Pro rule: connectedness by gamma state\nwithin each expiry regime (95% bootstrap intervals)", fontsize=10, color=ink, loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="best")
    ax1.grid(axis="y", color=grid, lw=0.8)
    ax1.set_axisbelow(True)

    t = tests[tests["test"] != "S7"].iloc[::-1].reset_index(drop=True)
    for i, r in t.iterrows():
        col = blue if r["test"] == "P0" else ink
        ax2.plot([r["boot_ci95_lo"], r["boot_ci95_hi"]], [i, i], color=col, lw=2, solid_capstyle="round")
        ax2.plot(r["estimate"], i, "o", ms=8 if r["test"] == "P0" else 7, color=col, markeredgecolor="white", markeredgewidth=1.2)
        pp = r["perm_p"] if r["test"] == "P0" else r["holm_p_secondary"]
        ax2.annotate(("p " if r["test"] == "P0" else "Holm p ") + f"{pp:.2g}", (1.01, i), xycoords=("axes fraction", "data"),
                     fontsize=8, color=muted, va="center")
    ax2.axvline(0, color=muted, lw=1)
    ax2.set_yticks(range(len(t)))
    ax2.set_yticklabels([f"{r['test']}  {_short(r)}" for _, r in t.iterrows()], fontsize=8, color=ink)
    ax2.set_xlabel("Difference in connectedness points (short minus long gamma; S5: top minus bottom gross tercile)", color=ink)
    ax2.set_title("Primary (P0) and secondary differences, 95% bootstrap intervals", fontsize=10, color=ink, loc="left")
    ax2.grid(axis="x", color=grid, lw=0.8)
    ax2.set_axisbelow(True)
    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=muted)
    fig.text(0.01, 0.01, "REAL DATA: NSE Nifty 50 and Bank Nifty 1-minute bars, Jan 2023 to Apr 2026; GEX at the previous close. "
             "Source: results/connectedness_gamma/state_connectedness.csv and tests.csv", fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.03, 0.97, 1))
    fig.savefig(OUT / "fig_connectedness_gamma.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def _short(r) -> str:
    m = {"P0": "Pro, total", "S1": "Pro, Nifty to BN", "S2": "Pro, BN to Nifty", "S3": "US rule, total", "S4": "Client mirror, total",
         "S5": "Gross gamma top vs bottom", "S6": "Pro, regime x VIX strata", "S8": "Pro, expiry days", "S9": "Pro, other days",
         "S10": "Pro, multi weekly", "S11": "Pro, single weekly Thu", "S12": "Pro, single weekly Tue", "S13": "Pro, AIC lag, H 30"}
    return m.get(r["test"], r["test"])


if __name__ == "__main__":
    main()
