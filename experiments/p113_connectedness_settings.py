"""P113: which connectedness setting reproduces the paper's 15 printed totals (Tables 3 and 4)?

Registered plan: docs/prereg/P113_CONNECTEDNESS_SETTINGS.md (committed and pushed before any
connectedness number of this study was computed). Write-up: docs/P113_CONNECTEDNESS_SETTINGS.md.

Every number this script writes is REAL DATA (third-party 1-minute Nifty 50 and Bank Nifty bars,
the same data and the same mcsGARCH code as experiments/replicate_paper.py), except the printed
targets, which are the paper's numbers as re-keyed blind in results/replication_nse/rekey_blind.csv.

The grid (Section 6 of the plan), per before window:
    5 decompositions x 8 input series x 8 horizons x 17 (construction, VAR lag) pairs = 5,440.

Usage (one process, about 3 minutes):

    PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 python experiments/p113_connectedness_settings.py

Reads data/raw/ (via the replication's loaders), results/replication_nse/expiry_calendar.csv,
results/replication_nse/rekey_blind.csv and results/replication_nse/tables34_connectedness.csv
(baseline check). Writes results/p113_connectedness_settings/.
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
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex import replication as rp

REPO = Path(__file__).resolve().parents[1]
OUT = Path("results/p113_connectedness_settings")
LABEL = "REAL DATA (third-party 1-minute bars validated against NSE official daily values)"
PLAN = "docs/prereg/P113_CONNECTEDNESS_SETTINGS.md"


def _load_replication():
    """Import experiments/replicate_paper.py as a module (its main() is not run)."""
    if "replicate_paper" in sys.modules:
        return sys.modules["replicate_paper"]
    spec = importlib.util.spec_from_file_location("replicate_paper", REPO / "experiments" / "replicate_paper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replicate_paper"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# The registered grid
# ---------------------------------------------------------------------------

WINDOWS = {"A": ("2022-11-14", "2023-02-10"), "B": ("2022-11-22", "2023-02-20"), "ST": ("2023-11-01", "2024-02-29")}
AFTER = ("2024-03-01", "2024-05-31")
DECOMPS = ("GR", "GU", "GC", "CN", "CB")
INPUTS = ("cvar", "cvol", "logvar", "qvol", "ret", "absret", "sqret", "rsvol")
HS = (1, 2, 5, 10, 11, 20, 50, 100)
LAGS_SC = ("1", "2", "3", "5", "10", "aic", "bic")
LAGS_R = ("1", "2", "5")
CONSTR = ("S", "C", "R")
ROLL_W = 200
PMAX = 10
BASELINE = {"D": "GR", "I": "cvol", "C": "S", "L": "2", "H": 10}
N_PERM = 1000
N_BOOT = 999
SEED = 113

SUBS = ("all_expiry", "monthly", "monthly_first_hour", "monthly_mid_hour", "monthly_last_hour")
# The 15 target cells: (cell id, side, family used for the days, sub-sample).
CELLS = [(f"T3.{i + 1}", "before", "union", s) for i, s in enumerate(SUBS)] + \
        [(f"T4.I.{i + 1}", "after", "nifty", s) for i, s in enumerate(SUBS)] + \
        [(f"T4.II.{i + 1}", "after", "bn", s) for i, s in enumerate(SUBS)]
CELL_IDS = [c[0] for c in CELLS]
OUTLIERS = ("T3.3", "T3.5", "T4.I.3")
FOLDS = {"Table 3": CELL_IDS[:5], "Table 4 Panel I": CELL_IDS[5:10], "Table 4 Panel II": CELL_IDS[10:]}
ALT_TARGETS = {"T4.I.3": 78.17, "T4.II.3": 71.97}


def spec_list() -> pd.DataFrame:
    """All 5,440 specifications in the registered tie-break order (D, I, C, L, H)."""
    rows = []
    for d in DECOMPS:
        for i in INPUTS:
            for c in CONSTR:
                for lag in (LAGS_R if c == "R" else LAGS_SC):
                    for h in HS:
                        rows.append({"D": d, "I": i, "C": c, "L": lag, "H": h})
    df = pd.DataFrame(rows)
    df["order"] = np.arange(len(df))
    df["spec"] = df["D"] + "|" + df["I"] + "|" + df["C"] + "|p=" + df["L"] + "|H=" + df["H"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Decompositions, batched over many VAR fits (checked against niftygex.connectedness in the tests)
# ---------------------------------------------------------------------------


def ma_batch(A: np.ndarray, Hmax: int) -> np.ndarray:
    """Moving-average matrices for a batch of 2-variable VARs. A: (B, p, N, N) -> Psi (B, Hmax, N, N)."""
    Bn, p, N, _ = A.shape
    Psi = np.zeros((Bn, Hmax, N, N))
    Psi[:, 0] = np.eye(N)
    for h in range(1, Hmax):
        acc = np.zeros((Bn, N, N))
        for lag in range(1, min(h, p) + 1):
            acc += A[:, lag - 1] @ Psi[:, h - lag]
        Psi[:, h] = acc
    return Psi


def decomp_batch(A: np.ndarray, Sigma: np.ndarray, Hs=HS) -> np.ndarray:
    """Cross shares in percent for every decomposition and horizon.

    Returns an array (B, len(DECOMPS), len(Hs), 2): [..., 0] = Nifty from Bank Nifty (row 0, column 1),
    [..., 1] = Bank Nifty from Nifty (row 1, column 0). Horizon H sums h = 0, ..., H - 1.
    NaN where a decomposition cannot be computed (Sigma not positive definite).
    """
    A = np.asarray(A, float)
    Sigma = np.asarray(Sigma, float)
    Bn = A.shape[0]
    Hmax = max(Hs)
    hidx = np.array(Hs) - 1
    Psi = ma_batch(A, Hmax)
    PS = Psi @ Sigma[:, None]  # (B, Hmax, 2, 2)
    diag = np.diagonal(Sigma, axis1=1, axis2=2)  # (B, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.cumsum(PS**2, axis=1)[:, hidx] / diag[:, None, None, :]
        den = np.cumsum((PS * Psi).sum(-1), axis=1)[:, hidx]  # (B, nH, 2): (Psi Sigma Psi')_ii
        raw = num / den[..., None]
        out = np.full((Bn, len(DECOMPS), len(Hs), 2), np.nan)
        for k, d in enumerate(DECOMPS):
            if d == "GR":
                th = raw / raw.sum(-1, keepdims=True)
            elif d == "GU":
                th = raw
            elif d == "GC":
                th = raw / raw.sum(-2, keepdims=True)
            else:
                ok = (diag > 0).all(1) & (np.linalg.det(Sigma) > 0)
                S = Sigma.copy()
                S[~ok] = np.eye(2)
                if d == "CN":
                    L = np.linalg.cholesky(S)
                    PB = Psi @ L[:, None]
                    th = np.cumsum(PB**2, axis=1)[:, hidx] / den[..., None]
                else:  # CB: Bank Nifty first
                    Lp = np.linalg.cholesky(S[:, ::-1, ::-1])
                    PB = Psi @ Lp[:, ::-1, :][:, None]
                    th = (np.cumsum(PB**2, axis=1)[:, hidx] / den[..., None])[..., ::-1]
                th = np.where(ok[:, None, None, None], th, np.nan)
            out[:, k, :, 0] = 100 * th[..., 0, 1]
            out[:, k, :, 1] = 100 * th[..., 1, 0]
    return out


# ---------------------------------------------------------------------------
# VAR fits: segmented, concatenated, rolling
# ---------------------------------------------------------------------------


def var_lag_ic(Y, seg, pmax: int, criterion: str) -> int:
    """VAR lag with the smallest AIC (the replication's rule) or BIC, all orders on the same rows."""
    if criterion == "aic":
        return rp.var_lag_aic(Y, seg, pmax)
    best, best_p = np.inf, 1
    for p in range(1, pmax + 1):
        fit = rp.fit_var_segments(Y, seg, p, min_pos=pmax)
        T, N = fit.resid.shape
        S = fit.resid.T @ fit.resid / T
        sign, logdet = np.linalg.slogdet(S)
        ic = logdet + np.log(T) * p * N * N / T if sign > 0 else np.inf
        if ic < best:
            best, best_p = ic, p
    return best_p


def lag_value(rule: str, Y, seg) -> int:
    return var_lag_ic(Y, seg, PMAX, rule) if rule in ("aic", "bic") else int(rule)


def fit_cross(Y, seg, rule: str):
    """One VAR on (Y, seg) under a lag rule -> (cross shares (len(DECOMPS), len(HS), 2), lag used)."""
    p = lag_value(rule, Y, seg)
    if len(Y) < 3 * (1 + 2 * p):
        return np.full((len(DECOMPS), len(HS), 2), np.nan), p
    fit = rp.fit_var_segments(Y, seg, p)
    return decomp_batch(fit.A[None], fit.Sigma[None])[0], p


def rolling_cross(Yj: np.ndarray, ends: np.ndarray, p: int, w: int = ROLL_W) -> np.ndarray:
    """Cross shares of VAR(p) fits in rolling windows of w rows of the joined series Yj, one per end row.

    The window ending at row e holds rows e - w + 1, ..., e; the VAR uses targets e - w + 1 + p, ..., e
    with lags inside the window (as fit_var on Yj[e - w + 1 : e + 1]). Computed from cumulative
    cross-products. The series is first divided by its standard deviation (the shares do not change).
    Returns (len(ends), len(DECOMPS), len(HS), 2); ends with e < w - 1 are NaN.
    """
    Yj = np.asarray(Yj, float)
    Yj = Yj / Yj.std(axis=0)
    T, N = Yj.shape
    k = 1 + N * p
    X = np.full((T, k), np.nan)
    X[:, 0] = 1.0
    for lag in range(1, p + 1):
        X[lag:, 1 + (lag - 1) * N : 1 + lag * N] = Yj[:-lag]
    X[:p] = 0.0
    Z = Yj.copy()
    Z[:p] = 0.0  # rows before p are never targets
    cxx = np.concatenate([np.zeros((1, k, k)), np.cumsum(X[:, :, None] * X[:, None, :], axis=0)])
    cxy = np.concatenate([np.zeros((1, k, N)), np.cumsum(X[:, :, None] * Z[:, None, :], axis=0)])
    cyy = np.concatenate([np.zeros((1, N, N)), np.cumsum(Z[:, :, None] * Z[:, None, :], axis=0)])
    ends = np.asarray(ends)
    out = np.full((len(ends), len(DECOMPS), len(HS), 2), np.nan)
    ok = ends >= w - 1
    e = ends[ok]
    lo = e - w + 1 + p  # first target row
    XtX = cxx[e + 1] - cxx[lo]
    XtY = cxy[e + 1] - cxy[lo]
    YtY = cyy[e + 1] - cyy[lo]
    Bc = np.linalg.solve(XtX, XtY)
    UU = YtY - np.einsum("bki,bkj->bij", Bc, XtY)
    Sigma = UU / (w - p - k)
    A = np.stack([Bc[:, 1 + (lag - 1) * N : 1 + lag * N, :].transpose(0, 2, 1) for lag in range(1, p + 1)], axis=1)
    res = np.empty((len(e), len(DECOMPS), len(HS), 2))
    for s in range(0, len(e), 2000):
        res[s : s + 2000] = decomp_batch(A[s : s + 2000], Sigma[s : s + 2000])
    out[ok] = res
    return out


# ---------------------------------------------------------------------------
# Data: windows, mcsGARCH fits, input series, cells
# ---------------------------------------------------------------------------


def input_arrays(G, fit_n, fit_b, rows) -> dict:
    """The 8 input series as (days x minutes) arrays for both indices."""
    out = {}
    for name in INPUTS:
        arrs = []
        for k, f in (("nifty", fit_n), ("banknifty", fit_b)):
            v = f["valid"]
            if name == "cvar":
                a = np.where(v, f["cond_vol"] ** 2, np.nan)
            elif name == "cvol":
                a = np.where(v, f["cond_vol"], np.nan)
            elif name == "logvar":
                a = np.where(v, 2 * np.log(f["cond_vol"]), np.nan)
            elif name == "qvol":
                a = np.where(v, f["intraday_vol"], np.nan)
            elif name == "ret":
                a = f["R"]
            elif name == "absret":
                a = np.abs(f["R"])
            elif name == "sqret":
                a = f["R"] ** 2
            else:
                a = G[k]["rs_vol"][rows]
            arrs.append(a)
        out[name] = arrs
    return out


def cell_mask(rpp, days, fam, sub, ex, M) -> np.ndarray:
    """(days x minutes) boolean mask of a sub-sample."""
    dm = np.ones(len(days), bool) if fam == "all_days" else rpp.subsample_days(days, fam, sub, ex)
    mm = np.zeros(M, bool)
    mm[rpp.minute_slice(sub, M)] = True
    return dm[:, None] & mm[None, :]


def stacked(rpp, arrs, days, fam, sub, ex, M, constr):
    """(Y, seg, day id) for the segmented (S) or concatenated (C) construction of one cell."""
    dm = np.ones(len(days), bool) if fam == "all_days" else rpp.subsample_days(days, fam, sub, ex)
    Y, seg, _ = rpp.stack_series(arrs, dm, rpp.minute_slice(sub, M))
    day = seg // 1_000_000
    if constr == "C":
        seg = np.zeros(len(Y), dtype=seg.dtype)
    return Y, seg, day


def joined(arrs):
    """The whole window's series joined day after day; rows with a missing value in either index dropped."""
    Yfull = np.column_stack([a.ravel() for a in arrs])
    ok = np.all(np.isfinite(Yfull), axis=1)
    pos = np.full(len(Yfull), -1)
    pos[ok] = np.arange(ok.sum())
    return Yfull[ok], pos


# ---------------------------------------------------------------------------
# Fit criterion, ranking, guards
# ---------------------------------------------------------------------------


def fit_metrics(T: np.ndarray, tgt: np.ndarray) -> dict:
    """T: (n_spec, 15) totals; tgt: (15,). NaN anywhere makes MAD15 infinite (ranked last)."""
    e = np.abs(T - tgt[None, :])
    bad = ~np.isfinite(e).all(1)
    e0 = np.where(np.isfinite(e), e, 0.0)
    mad = np.where(bad, np.inf, e0.mean(1))
    k2 = np.where(bad, 0, (e0 <= 2).sum(1))
    oi = [CELL_IDS.index(c) for c in OUTLIERS]
    out3 = np.where(bad, np.inf, e0[:, oi].max(1))
    mad10a = np.where(bad, np.inf, e0[:, 5:].mean(1))
    within2_after = ~bad & (e0[:, 5:] <= 2).all(1)
    return {"MAD15": mad, "K2": k2, "OUT3": out3, "MAD10a": mad10a, "after_all_within2": within2_after}


def rank_specs(specs: pd.DataFrame, m: dict) -> np.ndarray:
    """Indices of specs in registered ranking order: MAD15, then K2 (desc), OUT3, fixed order."""
    return np.lexsort((specs["order"].to_numpy(), m["OUT3"], -m["K2"], m["MAD15"]))


def crit_b(T, tgt, out_pos) -> np.ndarray:
    """Specs meeting criterion (b): the cells at out_pos within 5 points and MAD15 <= 5."""
    e = np.abs(T - tgt[None, :])
    fin = np.isfinite(e).all(1)
    return fin & (e[:, out_pos].max(1) <= 5.0) & (np.where(fin, e.mean(1), np.inf) <= 5.0)


def crit_c(T, tgt) -> np.ndarray:
    """Specs meeting criterion (c): MAD10a <= 1 and all 10 after cells within 2 points."""
    e = np.abs(T[:, 5:] - tgt[None, 5:])
    fin = np.isfinite(e).all(1)
    return fin & (np.where(fin, e.mean(1), np.inf) <= 1.0) & (e.max(1) <= 2.0)


def permutation_guard(T: np.ndarray, tgt: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Shuffle reference (plan Section 8): best MAD15 and criteria (b) and (c) on shuffled targets."""
    rng = np.random.default_rng(seed)
    fin = np.isfinite(T).all(1)
    Tf = T[fin]
    m = fit_metrics(T, tgt)
    best = float(np.min(m["MAD15"]))
    out_pos = np.array([CELL_IDS.index(c) for c in OUTLIERS])
    obs_b = bool(crit_b(T, tgt, out_pos).any())
    obs_c = bool(crit_c(T, tgt).any())
    best_perm, b_perm, c_perm, best10_perm = [], [], [], []
    for _ in range(n_perm):
        perm = rng.permutation(15)
        t = tgt[perm]
        e = np.abs(Tf - t[None, :])
        mad = e.mean(1)
        best_perm.append(mad.min())
        # criterion (b): the cells that now hold the three outlying values
        pos = np.array([int(np.where(perm == CELL_IDS.index(c))[0][0]) for c in OUTLIERS])
        b_perm.append(bool(((e[:, pos].max(1) <= 5.0) & (mad <= 5.0)).any()))
        # criterion (c): the 10 after targets shuffled among the 10 after cells
        pa = 5 + rng.permutation(10)
        ta = tgt.copy()
        ta[5:] = tgt[pa]
        ea = np.abs(Tf[:, 5:] - ta[None, 5:])
        best10_perm.append(ea.mean(1).min())
        c_perm.append(bool(((ea.mean(1) <= 1.0) & (ea.max(1) <= 2.0)).any()))
    best_perm = np.array(best_perm)
    best10 = float(np.min(m["MAD10a"]))
    return {
        "best_mad15": best, "p_best_mad15": (1 + int((best_perm <= best).sum())) / (n_perm + 1),
        "perm_best_mad15_median": float(np.median(best_perm)), "perm_best_mad15_p05": float(np.quantile(best_perm, 0.05)),
        "obs_b": obs_b, "p_b": (1 + int(np.sum(b_perm))) / (n_perm + 1) if obs_b else np.nan, "perm_share_b": float(np.mean(b_perm)),
        "best_mad10a": best10, "p_best_mad10a": (1 + int((np.array(best10_perm) <= best10).sum())) / (n_perm + 1),
        "obs_c": obs_c, "p_c": (1 + int(np.sum(c_perm))) / (n_perm + 1) if obs_c else np.nan, "perm_share_c": float(np.mean(c_perm)),
        "perm_best_mad10a_median": float(np.median(best10_perm)),
    }


def loto(T: np.ndarray, tgt: np.ndarray, specs: pd.DataFrame) -> pd.DataFrame:
    """Leave one table out: choose the best spec on 10 cells, score it on the 5 held-out cells."""
    rows = []
    e = np.abs(T - tgt[None, :])
    fin = np.isfinite(e).all(1)
    for fold, cells in FOLDS.items():
        ho = np.array([CELL_IDS.index(c) for c in cells])
        tr = np.setdiff1d(np.arange(15), ho)
        mad_tr = np.where(fin, e[:, tr].mean(1), np.inf)
        k2_tr = np.where(fin, (e[:, tr] <= 2).sum(1), 0)
        j = np.lexsort((specs["order"].to_numpy(), -k2_tr, mad_tr))[0]
        rows.append({"held_out": fold, "chosen_spec": specs["spec"].iloc[j], "train_mad10": mad_tr[j],
                     "held_out_mad5": e[j, ho].mean(), "held_out_errors": ";".join(f"{x:.2f}" for x in (T[j, ho] - tgt[ho])),
                     "held_out_mad5_median_spec": float(np.median(np.where(fin, e[:, ho].mean(1), np.nan)[fin]))})
    df = pd.DataFrame(rows)
    df["pooled_held_out_mad15"] = df["held_out_mad5"].mean()
    return df


def verdict(T, tgt, specs, guard, lt) -> dict:
    """The registered decision rule (plan Section 9)."""
    m = fit_metrics(T, tgt)
    j = rank_specs(specs, m)[0]
    held = float(lt["pooled_held_out_mad15"].iloc[0])
    out_pos = [CELL_IDS.index(c) for c in OUTLIERS]
    e = np.abs(T[j] - tgt)
    R = (m["MAD15"][j] <= 2.0) and (m["K2"][j] >= 12) and (e[out_pos].max() <= 2.0) and held <= 3.0 and guard["p_best_mad15"] <= 0.05
    a = (m["MAD15"][j] <= 4.0) and (m["K2"][j] >= 8) and held <= 5.0 and guard["p_best_mad15"] <= 0.05
    b = guard["obs_b"] and guard["p_b"] <= 0.05
    c = guard["obs_c"] and guard["p_c"] <= 0.05
    v = "reproduced" if R else ("partly reproduced" if (a or b or c) else "not reproduced")
    return {"best_spec": specs["spec"].iloc[j], "best_MAD15": m["MAD15"][j], "best_K2": int(m["K2"][j]), "best_OUT3": m["OUT3"][j],
            "best_MAD10a": m["MAD10a"][j], "held_out_MAD15": held, "p_best_mad15": guard["p_best_mad15"],
            "n_specs_meeting_b": int(crit_b(T, tgt, np.array(out_pos)).sum()), "p_b": guard["p_b"],
            "n_specs_meeting_c": int(crit_c(T, tgt).sum()), "p_c": guard["p_c"],
            "rule_reproduced": bool(R), "rule_partly_a": bool(a), "rule_partly_b": bool(b), "rule_partly_c": bool(c), "verdict": v}


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot_p(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    n = len(d)
    return float(min(1.0, 2 * min((1 + (d <= 0).sum()) / (n + 1), (1 + (d >= 0).sum()) / (n + 1))))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def read_targets(rpp) -> pd.DataFrame:
    rk = pd.read_csv("results/replication_nse/rekey_blind.csv", dtype=str)
    rk = rk[rk["table"].isin(["Table 3", "Table 4"])].copy()
    rk["block"] = rk["row_label"].str.split(" | ", regex=False).str[0].str.strip()
    rk["row"] = rk["row_label"].str.split(" | ", regex=False).str[1].str.strip()
    rk["panel"] = np.where(rk["column_label"].str.startswith("Panel II"), "II", np.where(rk["column_label"].str.startswith("Panel I"), "I", ""))
    rk["col"] = rk["column_label"].str.replace(r"^Panel I+: ", "", regex=True).str.strip()

    def sub_of(b):
        b = b.lower()
        if b.startswith("weekly and monthly"):
            return "all_expiry"
        if b.startswith("monthly expiry"):
            return "monthly"
        if b.startswith("1st"):
            return "monthly_first_hour"
        if b.startswith("mid"):
            return "monthly_mid_hour"
        if b.startswith("last"):
            return "monthly_last_hour"
        raise ValueError(b)

    rk["sub"] = rk["block"].map(sub_of)
    val = lambda s: float(s.replace(",", "").replace("*", ""))
    rows = []
    for cid, side, fam, sub in CELLS:
        tb = "Table 3" if side == "before" else "Table 4"
        pn = "" if side == "before" else ("I" if fam == "nifty" else "II")
        g = rk[(rk["table"] == tb) & (rk["panel"] == pn) & (rk["sub"] == sub)]
        get = lambda r, c: val(g[(g["row"] == r) & (g["col"] == c)]["printed_value"].iloc[0])
        rec = {"cell": cid, "side": side, "family": fam, "subsample": sub,
               "printed_total": get("To", "From"), "printed_nifty_from_bn": get("Nifty 50", "Bank Nifty"),
               "printed_bn_from_nifty": get("Bank Nifty", "Nifty 50"), "printed_nifty_own": get("Nifty 50", "Nifty 50"),
               "printed_bn_own": get("Bank Nifty", "Bank Nifty"), "source": "results/replication_nse/rekey_blind.csv"}
        key = ("before", "union", sub) if side == "before" else ("after", fam, sub)
        (m, tot) = rpp.P_T34[key]
        rec["matches_P_T34"] = bool(np.isclose(tot, rec["printed_total"]) and np.allclose(m, [[rec["printed_nifty_own"], rec["printed_nifty_from_bn"]], [rec["printed_bn_from_nifty"], rec["printed_bn_own"]]]))
        rec["alt_target_total"] = ALT_TARGETS.get(cid, rec["printed_total"])
        rows.append(rec)
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rpp = _load_replication()
    log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)

    # ---------------- targets
    tg = read_targets(rpp)
    tg.to_csv(OUT / "targets.csv", index=False)
    assert tg["matches_P_T34"].all(), "re-keyed targets differ from P_T34"
    tgt = tg["printed_total"].to_numpy(float)
    tgt_alt = tg["alt_target_total"].to_numpy(float)
    pr01 = tg["printed_nifty_from_bn"].to_numpy(float)
    pr10 = tg["printed_bn_from_nifty"].to_numpy(float)
    log("targets read (15 totals, 30 cross shares)")

    # ---------------- data, windows, mcsGARCH fits (the replication's code)
    bars = {k: rp.load_minute_bars(rpp.RAW / f"{k}_1min.csv", "2017-01-01", rpp.DATA_RANGE[1]) for k in rpp.INDICES}
    cal = pd.read_csv("results/replication_nse/expiry_calendar.csv", parse_dates=["expiry"])
    cal["is_monthly"] = cal["is_monthly"].astype(str).str.lower().eq("true")
    ex = rpp.expiry_sets(cal)
    G = rpp.build_grids(bars, "360_drop_last15")
    M = len(G["minutes"])
    fp = pd.read_csv("results/replication_nse/fingerprint_search.csv").iloc[0]
    assert (fp["start"], fp["end"]) == WINDOWS["A"], "window A is not the replication's inferred window"
    periods = {w: G["days"][(G["days"] >= pd.Timestamp(s)) & (G["days"] <= pd.Timestamp(e))] for w, (s, e) in WINDOWS.items()}
    periods["after"] = G["days"][(G["days"] >= pd.Timestamp(AFTER[0])) & (G["days"] <= pd.Timestamp(AFTER[1]))]
    fits, inputs = {}, {}
    for w, days in periods.items():
        rows = rpp.period_rows(G, days)
        fits[w] = {k: rpp.fit_period(G, k, rows) for k in rpp.INDICES}
        inputs[w] = input_arrays(G, fits[w]["nifty"], fits[w]["banknifty"], rows)
        log(f"mcsGARCH fitted: window {w} ({len(days)} days, {days.min():%Y-%m-%d} to {days.max():%Y-%m-%d})")

    # ---------------- baseline check: the replication's main setting through the replication's own code
    committed = pd.read_csv("results/replication_nse/tables34_connectedness.csv")
    bl = []
    pmap = {"A": "before_inferred", "ST": "before_stated", "after": "after"}
    for w, pname in pmap.items():
        days = periods[w]
        for fam in ("all_days", "union", "nifty", "bn"):
            for sub in (("all_days",) if fam == "all_days" else SUBS):
                dm = np.ones(len(days), bool) if fam == "all_days" else rpp.subsample_days(days, fam, sub, ex)
                Y, seg, _ = rpp.stack_series(inputs[w]["cvol"], dm, rpp.minute_slice(sub, M))
                rec = rpp.conn_record(Y, seg, 2, 10)
                mine, _ = fit_cross(Y, seg, "2")
                c = committed[(committed["period"] == pname) & (committed["family"] == fam) & (committed["subsample"] == sub)].iloc[0]
                dset = ";".join(days[dm].strftime("%Y-%m-%d")) if fam != "all_days" else ""
                bl.append({"window": w, "family": fam, "subsample": sub, "n_obs": len(Y), "committed_n_obs": int(c["n_obs"]),
                           "total": rec["paper_scale_total"], "committed_total": c["paper_scale_total"],
                           "abs_diff_total": abs(rec["paper_scale_total"] - c["paper_scale_total"]),
                           "abs_diff_cross": max(abs(rec["nifty_from_bn"] - c["nifty_from_bn"]), abs(rec["bn_from_nifty"] - c["bn_from_nifty"])),
                           "batched_minus_package": abs(mine[DECOMPS.index("GR"), HS.index(10)].sum() - rec["paper_scale_total"]),
                           "days_match": (dset == (c["days"] if isinstance(c["days"], str) else ""))})
    bl = pd.DataFrame(bl)
    bl.to_csv(OUT / "baseline_check.csv", index=False, float_format="%.12g")
    log(f"baseline check: max |diff| total {bl['abs_diff_total'].max():.2e}, cross {bl['abs_diff_cross'].max():.2e}, "
        f"batched vs package {bl['batched_minus_package'].max():.2e}, days match {bl['days_match'].all()}, n_obs match {(bl['n_obs'] == bl['committed_n_obs']).all()}")
    baseline_ok = bool((bl["abs_diff_total"] <= 1e-9 * np.maximum(1, bl["committed_total"].abs())).all() and bl["days_match"].all()
                       and (bl["n_obs"] == bl["committed_n_obs"]).all() and (bl["batched_minus_package"] <= 1e-9).all())

    # ---------------- the grid: every (window, cell, input, construction, lag) -> shares for all D and H
    specs = spec_list()
    cell_plan = {w: [c for c in CELLS if c[1] == "before"] for w in WINDOWS}
    cell_plan["after"] = [c for c in CELLS if c[1] == "after"]
    store = {}  # (window, cell, I, C, L) -> (shares (D, H, 2), p_used, n_obs)
    for w, cells in cell_plan.items():
        days = periods[w]
        for inp in INPUTS:
            arrs = inputs[w][inp]
            for cid, side, fam, sub in cells:
                for constr in ("S", "C"):
                    Y, seg, _ = stacked(rpp, arrs, days, fam, sub, ex, M, constr)
                    for lag in LAGS_SC:
                        sh, p = fit_cross(Y, seg, lag)
                        store[(w, cid, inp, constr, lag)] = (sh, p, len(Y))
            # rolling: one pass per lag over the whole window's joined series
            Yj, pos = joined(arrs)
            masks = {cid: cell_mask(rpp, days, fam, sub, ex, M).ravel() for cid, side, fam, sub in cells}
            ends = np.unique(np.concatenate([pos[m & (pos >= 0)] for m in masks.values()]))
            for lag in LAGS_R:
                rc = rolling_cross(Yj, ends, int(lag))
                for cid, m in masks.items():
                    sel = np.isin(ends, pos[m & (pos >= 0)])
                    vals = rc[sel]
                    store[(w, cid, inp, "R", lag)] = (np.nanmean(vals, axis=0) if np.isfinite(vals).any() else np.full(vals.shape[1:], np.nan),
                                                      int(lag), int(np.isfinite(vals[:, 0, 0, 0]).sum()))
            log(f"grid: window {w}, input {inp} done")

    # ---------------- assemble per-window matrices of totals (n_spec x 15) and cross shares
    dI = {d: i for i, d in enumerate(DECOMPS)}
    hI = {h: i for i, h in enumerate(HS)}

    def matrices(w):
        T = np.full((len(specs), 15), np.nan)
        C01 = np.full((len(specs), 15), np.nan)
        C10 = np.full((len(specs), 15), np.nan)
        P = np.zeros((len(specs), 15), int)
        Nn = np.zeros((len(specs), 15), int)
        for j, (cid, side, fam, sub) in enumerate(CELLS):
            ww = w if side == "before" else "after"
            for r in specs.itertuples():
                sh, p, n = store[(ww, cid, r.I, r.C, r.L)]
                x = sh[dI[r.D], hI[r.H]]
                C01[r.Index, j], C10[r.Index, j] = x[0], x[1]
                P[r.Index, j], Nn[r.Index, j] = p, n
        T = C01 + C10
        return T, C01, C10, P, Nn

    mats = {w: matrices(w) for w in WINDOWS}
    log("matrices assembled")

    # grid_cells.csv.gz: every specification and cell
    gc = []
    for w in WINDOWS:
        T, C01, C10, P, Nn = mats[w]
        for j, cid in enumerate(CELL_IDS):
            if CELLS[j][1] == "after" and w != "A":
                continue  # after cells are the same in every window; written once
            gc.append(pd.DataFrame({"window": w if CELLS[j][1] == "before" else "after", "cell": cid, "spec": specs["spec"],
                                    "D": specs["D"], "I": specs["I"], "C": specs["C"], "L": specs["L"], "H": specs["H"],
                                    "p_used": P[:, j], "n_obs": Nn[:, j], "nifty_from_bn": C01[:, j], "bn_from_nifty": C10[:, j],
                                    "total": T[:, j], "printed_total": tgt[j]}))
    gc = pd.concat(gc, ignore_index=True)
    gc.insert(0, "data", "REAL DATA")
    gc.to_csv(OUT / "grid_cells.csv.gz", index=False, float_format="%.4f")

    # spec_fit.csv: every specification's fit, per window
    sf = []
    for w in WINDOWS:
        T, C01, C10, _, _ = mats[w]
        m = fit_metrics(T, tgt)
        cs = np.abs(np.concatenate([C01 - pr01, C10 - pr10], axis=1))
        d = specs.copy()
        d.insert(0, "window", w)
        for k in ("MAD15", "K2", "OUT3", "MAD10a"):
            d[k] = m[k]
        d["CS30"] = np.where(np.isfinite(cs).all(1), cs.mean(1), np.nan)
        for j, cid in enumerate(CELL_IDS):
            d[f"err_{cid}"] = T[:, j] - tgt[j]
        d["rank"] = np.empty(len(d), int)
        d.loc[rank_specs(specs, m), "rank"] = np.arange(1, len(d) + 1)
        sf.append(d)
    sf = pd.concat(sf, ignore_index=True)
    sf.insert(0, "data", "REAL DATA")
    sf.to_csv(OUT / "spec_fit.csv", index=False, float_format="%.4f")
    n_nan = {w: int((~np.isfinite(mats[w][0]).all(1)).sum()) for w in WINDOWS}
    log(f"specifications with a missing cell: {n_nan}")

    # ---------------- primary analysis on window A
    TA = mats["A"][0]
    mA = fit_metrics(TA, tgt)
    order = rank_specs(specs, mA)
    best_j = order[0]
    best = specs.iloc[best_j]
    n_distinct = int(len(np.unique(np.round(np.where(np.isfinite(TA), TA, -999), 2), axis=0)))
    guard = {w: permutation_guard(mats[w][0], tgt, a.n_perm) for w in WINDOWS}
    lt = {w: loto(mats[w][0], tgt, specs) for w in WINDOWS}
    pd.concat([d.assign(window=w) for w, d in lt.items()]).to_csv(OUT / "loto.csv", index=False, float_format="%.4f")
    pd.DataFrame([dict(window=w, **g) for w, g in guard.items()]).to_csv(OUT / "permutation.csv", index=False, float_format="%.6g")
    log(f"window A: best {best['spec']} MAD15 {mA['MAD15'][best_j]:.2f}; {n_distinct} distinct result vectors")

    # verdicts: windows, the window-as-dimension grid, subsets
    vd = []
    for w in WINDOWS:
        v = verdict(mats[w][0], tgt, specs, guard[w], lt[w])
        vd.append({"analysis": f"window {w}", "grid": "all 5,440", "n_specs": len(specs), **v})
    Tall = np.vstack([mats[w][0] for w in WINDOWS])
    specs_all = pd.concat([specs.assign(spec=specs["spec"] + f"|W={w}", order=specs["order"] + i * len(specs)) for i, w in enumerate(WINDOWS)], ignore_index=True)
    g_all = permutation_guard(Tall, tgt, a.n_perm)
    l_all = loto(Tall, tgt, specs_all)
    vd.append({"analysis": "window as a dimension", "grid": "all 16,320", "n_specs": len(specs_all), **verdict(Tall, tgt, specs_all, g_all, l_all)})
    subsets = {
        "(i) paper's written method: GR, cvar or cvol, S or C": (specs["D"] == "GR") & specs["I"].isin(["cvar", "cvol"]) & specs["C"].isin(["S", "C"]),
        "(ii) all GR (Diebold-Yilmaz 2012)": specs["D"] == "GR",
        **{f"(iii) decomposition {d}": specs["D"] == d for d in DECOMPS},
        **{f"(iv) construction {c}": specs["C"] == c for c in CONSTR},
    }
    sub_best = {}
    for name, msk in subsets.items():
        msk = msk.to_numpy()
        sp = specs[msk].reset_index(drop=True)
        T = TA[msk]
        g = permutation_guard(T, tgt, a.n_perm)
        v = verdict(T, tgt, sp, g, loto(T, tgt, sp))
        vd.append({"analysis": "window A, subset", "grid": name, "n_specs": int(msk.sum()), **v})
        sub_best[name] = v["best_spec"]
    vd = pd.DataFrame(vd)
    vd.insert(0, "data", "REAL DATA")
    vd.to_csv(OUT / "verdict.csv", index=False, float_format="%.6g")
    vd[vd["analysis"] == "window A, subset"].to_csv(OUT / "subsets.csv", index=False, float_format="%.6g")
    log("verdicts:\n" + vd[["analysis", "grid", "best_spec", "best_MAD15", "best_K2", "best_OUT3", "held_out_MAD15", "p_best_mad15", "verdict"]].to_string())

    # best.csv: the top 25 on window A with their 15 cells; near_ties.csv
    top = sf[sf["window"] == "A"].sort_values("rank").head(25)
    top.to_csv(OUT / "best.csv", index=False, float_format="%.4f")
    nt = sf[(sf["window"] == "A") & (sf["MAD15"] <= mA["MAD15"][best_j] + 0.5)].sort_values("rank")
    nt.to_csv(OUT / "near_ties.csv", index=False, float_format="%.4f")

    # by_dimension.csv
    bd = []
    sA = sf[sf["window"] == "A"]
    for dim in ("D", "I", "C", "L", "H"):
        for val, g in sA.groupby(dim, sort=False):
            gb = g.sort_values("rank").iloc[0]
            bd.append({"data": "REAL DATA", "window": "A", "dimension": dim, "value": val, "n_specs": len(g), "best_MAD15": gb["MAD15"],
                       "best_spec": gb["spec"], "best_K2": gb["K2"], "median_MAD15": g["MAD15"].replace(np.inf, np.nan).median(),
                       "n_in_near_tie_set": int((g["MAD15"] <= mA["MAD15"][best_j] + 0.5).sum())})
    pd.DataFrame(bd).to_csv(OUT / "by_dimension.csv", index=False, float_format="%.4f")

    # reach.csv (S6): per decomposition and cell, the largest total and smaller cross share
    rc_rows = []
    for w in WINDOWS:
        T, C01, C10, _, _ = mats[w]
        for d in DECOMPS + ("all",):
            msk = np.ones(len(specs), bool) if d == "all" else (specs["D"] == d).to_numpy()
            for j, cid in enumerate(CELL_IDS):
                if CELLS[j][1] == "after" and w != "A":
                    continue
                t = T[msk, j]
                lo = np.minimum(C01[msk, j], C10[msk, j])
                rc_rows.append({"data": "REAL DATA", "window": w if CELLS[j][1] == "before" else "after", "decomposition": d, "cell": cid,
                                "printed_total": tgt[j], "n_specs": int(msk.sum()), "max_total": np.nanmax(t), "min_total": np.nanmin(t),
                                "max_smaller_cross_share": np.nanmax(lo), "n_within_2": int((np.abs(t - tgt[j]) <= 2).sum()),
                                "n_within_5": int((np.abs(t - tgt[j]) <= 5).sum()), "n_total_above_100": int((t > 100).sum())})
    pd.DataFrame(rc_rows).to_csv(OUT / "reach.csv", index=False, float_format="%.4f")

    # windows.csv (S4)
    wrows = []
    for w in WINDOWS:
        T = mats[w][0]
        m = fit_metrics(T, tgt)
        j = rank_specs(specs, m)[0]
        wrows.append({"data": "REAL DATA", "window": w, "dates": f"{WINDOWS[w][0]} to {WINDOWS[w][1]}", "n_days": len(periods[w]),
                      "best_spec": specs["spec"].iloc[j], "MAD15": m["MAD15"][j], "K2": m["K2"][j], "OUT3": m["OUT3"][j],
                      "baseline_MAD15": m["MAD15"][int(specs[(specs.D == "GR") & (specs.I == "cvol") & (specs.C == "S") & (specs.L == "2") & (specs.H == 10)].index[0])],
                      **{f"best_{cid}": T[j, k] for k, cid in enumerate(CELL_IDS)}})
    pd.DataFrame(wrows).to_csv(OUT / "windows.csv", index=False, float_format="%.4f")

    # alt_targets.csv (S5)
    ma = fit_metrics(TA, tgt_alt)
    ja = rank_specs(specs, ma)[0]
    pd.DataFrame([{"data": "REAL DATA", "window": "A", "targets": "alternative (T4.I.3 = 78.17, T4.II.3 = 71.97)", "best_spec": specs["spec"].iloc[ja],
                   "MAD15": ma["MAD15"][ja], "K2": ma["K2"][ja], "OUT3": ma["OUT3"][ja],
                   "same_best_as_printed_targets": bool(ja == best_j), "printed_targets_best_MAD15": mA["MAD15"][best_j]}]
                 ).to_csv(OUT / "alt_targets.csv", index=False, float_format="%.4f")

    # ---------------- secondaries S1 to S3 for the best spec, and the best spec of the paper's written method
    reported = {"overall best": best_j}
    jw = specs.index[specs["spec"] == sub_best["(i) paper's written method: GR, cvar or cvol, S or C"]][0]
    if jw != best_j:
        reported["best of the paper's written method"] = int(jw)
    ch_rows, dir_rows, bs_rows, test_rows = [], [], [], []
    T, C01, C10, _, _ = mats["A"]
    pr_net = pr10 - pr01
    for role_name, j in reported.items():
        sp = specs.iloc[j]
        # S1: changes
        for k, sub in enumerate(SUBS):
            for pan, off in (("Panel I", 5), ("Panel II", 10)):
                ours = T[j, off + k] - T[j, k]
                printed = tgt[off + k] - tgt[k]
                ch_rows.append({"data": "REAL DATA", "specification": sp["spec"], "reported_as": role_name, "subsample": sub, "panel": pan,
                                "before_total": T[j, k], "after_total": T[j, off + k], "change": ours, "printed_change": printed,
                                "abs_diff": abs(ours - printed), "same_sign": bool(np.sign(ours) == np.sign(printed))})
        # S2: direction per cell
        for k, cid in enumerate(CELL_IDS):
            net = C10[j, k] - C01[j, k]
            dir_rows.append({"data": "REAL DATA", "specification": sp["spec"], "reported_as": role_name, "cell": cid,
                             "net_nifty": net, "printed_net_nifty": pr_net[k], "same_sign": bool(np.sign(net) == np.sign(pr_net[k]))})
        # S2: all days of each window
        for w in ("A", "after"):
            days = periods[w]
            arrs = inputs[w][sp["I"]]
            if sp["C"] == "R":
                Yj, pos = joined(arrs)
                rc = rolling_cross(Yj, np.arange(len(Yj)), int(sp["L"]))
                x = np.nanmean(rc[:, dI[sp["D"]], hI[sp["H"]]], axis=0)
            else:
                Y, seg, _ = stacked(rpp, arrs, days, "all_days", "all_days", ex, M, sp["C"])
                sh, _ = fit_cross(Y, seg, sp["L"])
                x = sh[dI[sp["D"]], hI[sp["H"]]]
            dir_rows.append({"data": "REAL DATA", "specification": sp["spec"], "reported_as": role_name, "cell": f"all days, window {w}",
                             "net_nifty": x[1] - x[0], "printed_net_nifty": np.nan, "same_sign": np.nan, "total": x.sum()})
        # S3: day-level bootstrap of every cell
        draws = {}
        for k, (cid, side, fam, sub) in enumerate(CELLS):
            w = "A" if side == "before" else "after"
            days = periods[w]
            arrs = inputs[w][sp["I"]]
            rng = np.random.default_rng(SEED + 1000 * len(bs_rows) + k)
            out = np.full(a.n_boot, np.nan)
            if sp["C"] == "R":
                Yj, pos = joined(arrs)
                msk = cell_mask(rpp, days, fam, sub, ex, M)
                ends_by_day = [pos[np.where(msk[d] & (pos.reshape(msk.shape)[d] >= 0))[0] + d * M] for d in range(len(days)) if msk[d].any()]
                ends_all = np.concatenate(ends_by_day)
                rc = rolling_cross(Yj, ends_all, int(sp["L"]))[:, dI[sp["D"]], hI[sp["H"]]].sum(-1)
                per_day, s0 = [], 0
                for e_d in ends_by_day:
                    per_day.append(rc[s0 : s0 + len(e_d)])
                    s0 += len(e_d)
                for b in range(a.n_boot):
                    pick = rng.integers(0, len(per_day), len(per_day))
                    out[b] = np.nanmean(np.concatenate([per_day[i] for i in pick]))
            else:
                Y, seg, day = stacked(rpp, arrs, days, fam, sub, ex, M, "S")
                ud = np.unique(day)
                pieces = [np.where(day == u)[0] for u in ud]
                for b in range(a.n_boot):
                    pick = rng.integers(0, len(pieces), len(pieces))
                    idx = np.concatenate([pieces[i] for i in pick])
                    if sp["C"] == "C":
                        sb = np.zeros(len(idx), int)
                    else:
                        sb = np.concatenate([seg[pieces[i]] % 1_000_000 + 1_000_000 * n for n, i in enumerate(pick)])
                    sh, _ = fit_cross(Y[idx], sb, sp["L"])
                    out[b] = sh[dI[sp["D"]], hI[sp["H"]]].sum()
            draws[cid] = out
            bs_rows.append({"data": "REAL DATA", "specification": sp["spec"], "reported_as": role_name, "cell": cid, "point": T[j, k],
                            "boot_ci95_lo": np.nanpercentile(out, 2.5), "boot_ci95_hi": np.nanpercentile(out, 97.5),
                            "n_draws_finite": int(np.isfinite(out).sum()), "printed_total": tgt[k]})
        stm = [("1", "All expiry days: Panel I total minus Table 3 total (connectedness falls on Nifty expiry days)", "T4.I.1", "T3.1", None, None),
               ("2", "All expiry days: Panel II total minus Table 3 total (falls on Bank Nifty expiry days)", "T4.II.1", "T3.1", None, None),
               ("3", "Last hour of monthly expiry days: Panel I minus Table 3", "T4.I.5", "T3.5", None, None),
               ("4", "Last hour of monthly expiry days: Panel II minus Table 3", "T4.II.5", "T3.5", None, None),
               ("5", "Last-hour change minus all-expiry change, Panel I (the fall is larger in the last hour)", "T4.I.5", "T3.5", "T4.I.1", "T3.1"),
               ("6", "Last-hour change minus all-expiry change, Panel II", "T4.II.5", "T3.5", "T4.II.1", "T3.1")]
        rows_t = []
        for sid, hyp, a1, b1, a2, b2 in stm:
            ia, ib = CELL_IDS.index(a1), CELL_IDS.index(b1)
            est = T[j, ia] - T[j, ib]
            d = draws[a1] - draws[b1]
            if a2:
                ia2, ib2 = CELL_IDS.index(a2), CELL_IDS.index(b2)
                est -= T[j, ia2] - T[j, ib2]
                d = d - (draws[a2] - draws[b2])
            printed = (tgt[ia] - tgt[ib]) - ((tgt[CELL_IDS.index(a2)] - tgt[CELL_IDS.index(b2)]) if a2 else 0)
            rows_t.append({"sid": sid, "hyp": hyp, "est": est, "lo": np.nanpercentile(d, 2.5), "hi": np.nanpercentile(d, 97.5),
                           "p": boot_p(d), "printed": printed})
        padj = holm([r["p"] for r in rows_t])
        tag = "best" if role_name == "overall best" else "paper_method_best"
        for r, pa in zip(rows_t, padj):
            test_rows.append({"test_id": f"S3_{tag}_{r['sid']}",
                              "hypothesis": f"{r['hyp']}. Under specification {sp['spec']} ({role_name}, window A). Estimate: our difference in points of the paper's scale (printed: {r['printed']:.2f}); 95% day-level bootstrap interval {r['lo']:.2f} to {r['hi']:.2f}.",
                              "role": "secondary", "sample": f"before {WINDOWS['A'][0]} to {WINDOWS['A'][1]}; after {AFTER[0]} to {AFTER[1]}; expiry days from the NSE bhavcopy",
                              "estimate": r["est"], "statistic": r["est"], "p_raw": r["p"], "sidedness": "two-sided",
                              "p_method": f"day-level bootstrap ({a.n_boot} draws per cell, whole days or whole hour blocks resampled, before and after independent): p = min(1, 2 min((1 + #draws <= 0)/(n + 1), (1 + #draws >= 0)/(n + 1)))",
                              "p_adjusted": pa, "adjust_method": f"Holm over 6 rows: S3 statements for {role_name}"})
    pd.DataFrame(ch_rows).to_csv(OUT / "changes.csv", index=False, float_format="%.4f")
    pd.DataFrame(dir_rows).to_csv(OUT / "direction.csv", index=False, float_format="%.4f")
    pd.DataFrame(bs_rows).to_csv(OUT / "bootstrap.csv", index=False, float_format="%.4f")

    # ---------------- tests.csv
    gA = guard["A"]
    prim = [
        {"test_id": "P1_best_fit_shuffle", "hypothesis": f"The best of the 5,440 specifications ({best['spec']}) fits the 15 printed totals in their true cells better than the same grid fits the totals shuffled among the 15 cells (window A). Estimate: best MAD15 in points; median best MAD15 over shuffles {gA['perm_best_mad15_median']:.2f}.",
         "estimate": gA["best_mad15"], "statistic": gA["best_mad15"], "p_raw": gA["p_best_mad15"]},
        {"test_id": "P2_outlier_cells_shuffle", "hypothesis": f"Some specification brings the three outlying cells (99.42, 105.22, 80.18) within 5 points with MAD15 <= 5 more often than when the totals are shuffled (window A). Observed: {'yes' if gA['obs_b'] else 'no'}; share of shuffles in which some specification does it: {gA['perm_share_b']:.3f}. Estimate: number of specifications meeting it on the true targets.",
         "estimate": float(crit_b(TA, tgt, np.array([CELL_IDS.index(c) for c in OUTLIERS])).sum()), "statistic": float(crit_b(TA, tgt, np.array([CELL_IDS.index(c) for c in OUTLIERS])).sum()), "p_raw": gA["p_b"]},
        {"test_id": "P3_after_cells_shuffle", "hypothesis": f"Some specification reproduces the 10 after cells (MAD <= 1, all within 2 points) better than when the 10 after totals are shuffled among the after cells. Observed: {'yes' if gA['obs_c'] else 'no'}. Estimate: best MAD over the 10 after cells; median over shuffles {gA['perm_best_mad10a_median']:.2f}; p for the best MAD10a {gA['p_best_mad10a']:.4f}.",
         "estimate": gA["best_mad10a"], "statistic": gA["best_mad10a"], "p_raw": gA["p_c"]},
    ]
    for r in prim:
        r.update({"role": "primary", "sample": f"window A ({WINDOWS['A'][0]} to {WINDOWS['A'][1]}) and after ({AFTER[0]} to {AFTER[1]}), 15 printed totals of Tables 3 and 4",
                  "sidedness": "one-sided (a fit at least as good on shuffled targets)",
                  "p_method": f"permutation: {a.n_perm} random reassignments of the printed totals to the cells, the whole grid searched again each time; p = (1 + count)/(n + 1); NaN when the criterion is not met on the true targets (no test)"})
    pp = np.array([r["p_raw"] for r in prim], float)
    padj = np.full(3, np.nan)
    okp = np.isfinite(pp)
    padj[okp] = holm(pp[okp])
    for r, x in zip(prim, padj):
        r["p_adjusted"] = x
        r["adjust_method"] = f"Holm over the {int(okp.sum())} primary rows with a p-value"
    tests = pd.DataFrame(prim + test_rows)[["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method", "p_adjusted", "adjust_method"]]
    tests.to_csv(OUT / "tests.csv", index=False)

    # ---------------- run info
    info = {"study": "P113", "label": LABEL, "plan": PLAN,
            "plan_commit": subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", PLAN], capture_output=True, text=True).stdout.strip(),
            "inputs_sha256": {p: sha256(Path(p)) for p in ("data/raw/nifty_1min.csv", "data/raw/banknifty_1min.csv",
                                                            "results/replication_nse/expiry_calendar.csv", "results/replication_nse/rekey_blind.csv",
                                                            "results/replication_nse/tables34_connectedness.csv")},
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "n_specs_per_window": len(specs), "n_distinct_result_vectors_window_A": n_distinct, "n_specs_with_missing_cell": n_nan,
            "baseline_reproduces_committed": baseline_ok, "n_perm": a.n_perm, "n_boot": a.n_boot, "seed": SEED,
            "windows": WINDOWS, "after": AFTER, "rolling_window_minutes": ROLL_W, "run_seconds": round(time.time() - t0, 1)}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    log(f"done: baseline reproduces committed = {baseline_ok}; best {best['spec']}")


if __name__ == "__main__":
    main()
