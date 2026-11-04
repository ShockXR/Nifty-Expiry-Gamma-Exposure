"""P114: does the fall in intraday persistence survive when the intraday model includes P109's fixes?
(registered plan: docs/prereg/P114_PERSISTENCE_FIXED_MODEL.md)

The replication (experiments/replicate_paper.py, Table 2) finds that the intraday persistence alpha + beta of
mcsGARCH falls from the paper's "before" sample to its "after" sample: the half-life of an intraday volatility
shock falls from 359 to 72 minutes for Nifty on the inferred before window. P109 found that mcsGARCH is
misspecified (no running level inside the day, no overnight reset, a stale daily level). A GARCH(1,1) without a
slowly moving level can show alpha + beta near 1 because the day's level drifts. This script refits the
replication's mcsGARCH (the reference, exactly as in Table 2) and the same model with P109's fixes on each
window, and asks whether the fall in persistence survives.

    ref   the replication's ARMA(1,1)-mcsGARCH(1,1)-t (replicate_paper.fit_period), daily AR(1)-EGARCH level
    F     the fixed model (primary): X1 HAR daily level + X2 overnight decay + X3 today's running mean (GARCH-X)
    F4    F plus X4 (leverage), the likelihood part of P109's combination (secondary)
    X1, X2, X3   one fix at a time (exploratory, point estimates only)

Persistence measures (niftygex.volatility.ext_persistence_summary): alpha + beta, and the half-life of the model's
impulse response to one standardised shock at 12:15 (minute 180 of the 360-minute session), by the exact
recursion; its equivalent persistence is 0.5 ** (1 / half-life).

Stages (from the repository root, raw data in data/raw):

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py fit
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py boot --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py lr
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py rolling
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py analyze --plan-commit <hash>

Each stage writes to --work (default data/processed/p114, git-ignored); a finished bootstrap key is not rerun.
``analyze`` writes results/p114_persistence_fixed_model/. ``--synthetic`` runs every stage on a simulated
two-index market (labelled SYNTHETIC). Tests: tests/test_p114_persistence_fixed_model.py (synthetic only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from niftygex import replication as rp  # noqa: E402
from niftygex.evaluation import holm  # noqa: E402
from niftygex.volatility import (IRF_MAX_STEPS, McsGarchExt, equivalent_persistence,  # noqa: E402
                                 ext_impulse_response, ext_persistence_summary, fit_garch11, fit_garch_ext,
                                 fit_garch_ext_fixed, har_daily_variance)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The replication's own code builds the grids, the windows, the daily EGARCH level and the reference fit, so that
# the reference is Table 2 exactly (checked in the fit stage).
rep = _load("replicate_paper", ROOT / "experiments" / "replicate_paper.py")

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

LABEL_REAL = "REAL DATA (third-party 1-minute bars validated against NSE official daily values)"
LABEL_SYN = "SYNTHETIC (simulated two-index market, not real data)"
INDICES = ("nifty", "banknifty")
INDEX_NAME = {"nifty": "Nifty 50", "banknifty": "Bank Nifty"}
TRIM = "360_drop_last15"                     # the replication's primary session rule, 09:15 to 15:14
DATA_END_MAIN = "2024-06-30"                 # the replication's data range (rep.DATA_RANGE[1])
WINDOWS = {                                  # inclusive first and last day
    "before_inferred": ("2022-11-14", "2023-02-10"),   # the replication's inferred window (primary)
    "before_stated": ("2023-11-01", "2024-02-29"),     # the paper's stated window (secondary)
    "after": ("2024-03-01", "2024-05-31"),             # the paper's after window
    "before_p108": ("2022-11-22", "2023-02-20"),       # P108's blind window (robustness)
}
BEFORE_WINDOWS = ["before_inferred", "before_stated", "before_p108"]
# the replication's bootstrap seeds: 1000 * i + chunk, with i the position of the key in its fits dict
SEED_INDEX = {("before_stated", "nifty"): 0, ("before_stated", "banknifty"): 1,
              ("before_inferred", "nifty"): 2, ("before_inferred", "banknifty"): 3,
              ("after", "nifty"): 4, ("after", "banknifty"): 5,
              ("before_p108", "nifty"): 6, ("before_p108", "banknifty"): 7}
N_BOOT = 400
BOOT_CHUNKS = 4
BLOCK_DAYS = 5
SHOCK_MINUTE = 180                           # 12:15, the middle of the 360-minute session
OTHER_SHOCKS = {"1015": 60, "1415": 300}

# model -> (daily level, overnight decay X2, leverage X4, today's running mean X3)
MODELS = {
    "ref": ("egarch", False, False, False),
    "F": ("har", True, False, True),
    "F4": ("har", True, True, True),
    "X1": ("har", False, False, False),
    "X2": ("egarch", True, False, False),
    "X3": ("egarch", False, False, True),
}
MODEL_LABEL = {
    "ref": "reference: the replication's mcsGARCH (Table 2)",
    "F": "fixed model: X1 HAR daily level + X2 overnight decay + X3 today's running mean",
    "F4": "fixed model plus X4 leverage (P109's combination without its forecast calibration)",
    "X1": "X1 alone (HAR daily level)", "X2": "X2 alone (overnight decay)", "X3": "X3 alone (today's running mean)",
}
# bootstrap: ref and F on every key; F4 on the inferred and after windows. Order of computation is registered.
BOOT_ORDER = [("before_inferred", "nifty"), ("after", "nifty"), ("before_inferred", "banknifty"),
              ("after", "banknifty"), ("before_stated", "nifty"), ("before_stated", "banknifty"),
              ("before_p108", "nifty"), ("before_p108", "banknifty")]
F4_KEYS = {("before_inferred", "nifty"), ("after", "nifty"), ("before_inferred", "banknifty"),
           ("after", "banknifty")}
# likelihood-ratio tests: (before window, index, model, measure)
LR_TESTS = [("before_inferred", "nifty", "F", "irf"), ("before_inferred", "nifty", "F", "ab"),
            ("before_inferred", "banknifty", "F", "irf"), ("before_inferred", "banknifty", "F", "ab"),
            ("before_stated", "nifty", "F", "irf"), ("before_stated", "banknifty", "F", "irf")]
# reproduction of the replication's likelihood-ratio statistics with this script's profile code (a check)
LR_CHECKS = [("before_inferred", "nifty", "ref", "ab"), ("before_inferred", "banknifty", "ref", "ab"),
             ("before_stated", "nifty", "ref", "ab"), ("before_stated", "banknifty", "ref", "ab")]
# rolling windows (exploratory)
ROLL_FIRST = "2022-11-14"
ROLL_LEN = 64
ROLL_STEP = 21
ROLL_NONOVERLAP_EVERY = 4                    # windows 0, 4, 8, ... do not overlap (84 days apart)
DATA_END_ROLL = {"nifty": "2026-11-02", "banknifty": "2026-04-22"}
HL_CAP = float(IRF_MAX_STEPS)                # a half-life that is infinite is capped here for logs and ratios

SYNTH = dict(n_days=420, start="2021-01-04", seed=114, n_boot=8, roll_first=250, roll_len=40, roll_step=20,
             windows={"before_inferred": (250, 314), "before_stated": (190, 250), "after": (340, 399),
                      "before_p108": (255, 319)})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def synthetic_bars(n_days: int, seed: int, beta_change_day: int = 330) -> dict:
    """SYNTHETIC: one-minute bars for two indices from an mcsGARCH-type model with a GARCH-X intraday part,
    an overnight decay and a drifting daily level. The intraday beta falls at ``beta_change_day``."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    M = 375
    i = np.arange(M)
    s = 1.0 + 1.8 * np.exp(-i / 25.0) + 0.6 * np.exp(-(M - 1 - i) / 30.0)
    s = s / s.mean()
    out = {}
    for j, k in enumerate(INDICES):
        om, al, gx, de, nu = 0.02, 0.04, 0.015, 0.2, 7.0
        lv = np.zeros(n_days)                  # log daily level: AR(1), so the level drifts
        for t in range(1, n_days):
            lv[t] = 0.95 * lv[t - 1] + 0.12 * rng.standard_normal()
        h = (6e-4 if j == 0 else 9e-4) ** 2 / M * np.exp(lv)
        z = rng.standard_t(nu, size=(n_days, M)) / np.sqrt(nu / (nu - 2))
        r = np.zeros((n_days, M))
        q = 1.0
        for t in range(n_days):
            be = 0.93 if t < beta_change_day else 0.88
            pers = al + be + gx
            qbar = om / (1 - pers)
            run = 0.0
            for m in range(M):
                x = np.sqrt(q) * z[t, m]
                r[t, m] = np.sqrt(h[t] * s[m]) * x
                run += x * x
                q = om + al * x * x + gx * run / (m + 1) + be * q
            q = qbar + de * (q - qbar)
        gap = np.sqrt(0.3 * h * M) * rng.standard_normal(n_days)
        inc = r.copy()
        inc[:, 0] += gap                       # the first bar's close carries the overnight gap
        logc = np.log(10000.0 * (1 + j)) + np.cumsum(inc.ravel()).reshape(n_days, M)
        close = np.exp(logc)
        openp = np.exp(logc - r)
        ts = (days.values[:, None] + pd.to_timedelta("09:15:00").to_timedelta64()
              + (np.arange(M) * np.timedelta64(1, "m"))[None, :])
        df = pd.DataFrame({"timestamp": pd.DatetimeIndex(ts.ravel()), "open": openp.ravel(),
                           "high": np.maximum(openp, close).ravel(), "low": np.minimum(openp, close).ravel(),
                           "close": close.ravel()})
        df["date"] = df["timestamp"].dt.normalize()
        df["tod"] = df["timestamp"].dt.strftime("%H:%M")
        out[k] = df
    return out


def load_bars(raw: Path, end: str) -> dict:
    return {k: rp.load_minute_bars(raw / f"{k}_1min.csv", "2017-01-01", end) for k in INDICES}


def windows_for(G: dict, synthetic: bool) -> dict:
    if synthetic:
        return {w: G["days"][a:b] for w, (a, b) in SYNTH["windows"].items()}
    return {w: G["days"][(G["days"] >= pd.Timestamp(a)) & (G["days"] <= pd.Timestamp(b))]
            for w, (a, b) in WINDOWS.items()}


def daily_rv(G: dict, k: str) -> np.ndarray:
    """Daily realized variance for the HAR level (X1): the sum of squared one-minute log returns in the session,
    with the day's first bar measured from its own open, so the overnight gap is left out (as in P109)."""
    C, O = G[k]["close"], G[k]["open"]
    r = np.array(G[k]["r"], float, copy=True)
    fin = np.isfinite(C)
    first = np.argmax(fin, axis=1)
    rows = np.arange(len(C))
    r[rows, first] = np.log(C[rows, first] / O[rows, first])
    return np.nansum(r**2, axis=1)


_HAR_CACHE: dict = {}


def har_level(G: dict, k: str, last_row: int) -> np.ndarray:
    """X1: one-step-ahead daily level from a log-HAR of daily realized variance, fitted on all days from the start
    of the data up to ``last_row`` (in sample for each window, as the replication's daily EGARCH)."""
    key = (G["trim"], k, last_row)
    if key not in _HAR_CACHE:
        rv = daily_rv(G, k)
        h, info = har_daily_variance(rv[: last_row + 1])
        hf = np.full(len(G["days"]), np.nan)
        hf[: last_row + 1] = h
        _HAR_CACHE[key] = (hf, info["params"])
    return _HAR_CACHE[key]


def index_grid(bars: pd.DataFrame, k: str, trim: str = TRIM) -> dict:
    """The replication's grid (rep.build_grids) for one index only: the days on which that index has at least
    300 bars in the session (the replication also requires the other index; used for the rolling windows, which
    run past the end of the Bank Nifty data)."""
    g0 = rp.minute_grid(bars, trim)
    ok = pd.DatetimeIndex(g0["days"][np.isfinite(g0["close"]).sum(axis=1) >= 300])
    g = rp.minute_grid(bars, trim, days=ok)
    r = rp.continuous_log_returns(g["close"], overnight=True)
    last = pd.Series(np.log(g["close"][:, ::-1]).tolist()).map(lambda row: next((x for x in row if np.isfinite(x)), np.nan))
    daily = np.r_[np.nan, np.diff(last.to_numpy())]
    return {"days": ok, "minutes": g["minutes"], "trim": f"{trim}|{k}-only", "overnight": True,
            k: {"open": g["open"], "high": g["high"], "low": g["low"], "close": g["close"], "r": r, "daily": daily}}


# ---------------------------------------------------------------------------
# Fitting one window
# ---------------------------------------------------------------------------


def drift_measures(x: np.ndarray, valid: np.ndarray) -> dict:
    """Registered drift measure: m_t = mean over the day's valid minutes of the squared normalised residual x^2
    that the intraday GARCH sees; drift = standard deviation over the window's days of log m_t; ac1 = the lag-1
    autocorrelation of log m_t across the window's days."""
    v = valid.astype(float)
    m = np.sum(np.where(valid, x * x, 0.0), axis=1) / np.maximum(v.sum(axis=1), 1)
    lm = np.log(m[m > 0])
    return {"drift_sd_log_daily_mean_x2": float(np.std(lm, ddof=1)),
            "drift_ac1_log_daily_mean_x2": float(np.corrcoef(lm[1:], lm[:-1])[0, 1])}


def summary_row(params: dict, M: int) -> dict:
    s = ext_persistence_summary(params, M, SHOCK_MINUTE)
    for lab, m in OTHER_SHOCKS.items():
        s[f"hl_irf_{lab}"] = ext_persistence_summary(params, M, m)["hl_irf"]
    return s


def fit_window(G: dict, k: str, rows: np.ndarray, models=tuple(MODELS)) -> dict:
    """Every model on one window and index, in sample, as the replication fits Table 2."""
    ref = rep.fit_period(G, k, rows)
    out = {"_ref_fit": ref}
    Rk, keep, valid = ref["Rk"], ref["keep"], ref["valid_k"]
    M = Rk.shape[1]
    hh = {"egarch": ref["h"], "har": har_level(G, k, int(rows[-1]))[0][rows]}
    for name in models:
        t0 = time.time()
        daily, decay, lev, todayrv = MODELS[name]
        if name == "ref":
            mdl, filt, h = ref["mg"], ref["filt"], ref["h"]
        else:
            h = hh[daily]
            mdl = McsGarchExt(mean="arma11", dist="t", smooth=1, overnight_decay=decay, leverage=lev,
                              garchx_today=todayrv).fit(Rk, h)
            filt = mdl.filter(Rk, h)
        p = {kk: v for kk, v in mdl.params.items() if kk != "tstats"}
        p.setdefault("gam_l", 0.0)
        p.setdefault("gam_x", 0.0)
        p.setdefault("delta", 1.0)
        row = {"model": name, "M": M, "n_obs": int(valid.sum()), "n_days": len(rows), "kept_minutes": int(keep.sum()),
               **{kk: p.get(kk) for kk in ("omega", "alpha", "beta", "gam_l", "gam_x", "delta", "nu", "loglik",
                                           "converged")},
               "arma_phi": mdl.mean_params.get("phi"), "arma_theta": mdl.mean_params.get("theta"),
               **summary_row(p, M), **drift_measures(filt["x"], valid), "fit_seconds": round(time.time() - t0, 2)}
        out[name] = {"row": row, "params": p, "eps": filt["eps"], "h": h, "x": filt["x"]}
    out["_valid"] = valid
    out["_M"] = M
    return out


# ---------------------------------------------------------------------------
# Stage fit: point estimates and the reproduction check of Table 2
# ---------------------------------------------------------------------------


def stage_fit(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    label = LABEL_SYN if a.synthetic else LABEL_REAL
    bars = synthetic_bars(SYNTH["n_days"], SYNTH["seed"]) if a.synthetic else load_bars(Path(a.raw), DATA_END_MAIN)
    G = rep.build_grids(bars, TRIM)
    wins = windows_for(G, a.synthetic)
    print(f"[{label}] P114 fit stage: {len(G['days'])} days in the grid; windows: "
          + "; ".join(f"{w} {d.min().date()} to {d.max().date()} ({len(d)} days)" for w, d in wins.items()), flush=True)
    rows, store = [], {}
    for w, days in wins.items():
        for k in INDICES:
            fw = fit_window(G, k, rep.period_rows(G, days))
            for name in MODELS:
                rows.append({"data": label, "window": w, "index": k, "start": str(days.min().date()),
                             "end": str(days.max().date()), **fw[name]["row"]})
            store[(w, k)] = {"valid": fw["_valid"], "M": fw["_M"],
                             **{name: {"params": fw[name]["params"], "eps": fw[name]["eps"], "h": fw[name]["h"]}
                                for name in ("ref", "F", "F4")},
                             "x": {name: fw[name]["x"] for name in MODELS}}
            # the ARMA residuals do not depend on the variance model
            for name in ("F", "F4"):
                assert np.array_equal(store[(w, k)][name]["eps"], store[(w, k)]["ref"]["eps"])
            print(f"  {w} {k}: done ({time.time() - t0:.0f}s)", flush=True)
    fits = pd.DataFrame(rows)
    fits.to_csv(work / "fits.csv", index=False, float_format="%.17g")
    with open(work / "fits.pkl", "wb") as fh:
        pickle.dump(store, fh)
    # reproduction check: the reference must equal the replication's Table 2 (committed) on its three windows
    if not a.synthetic:
        t2 = pd.read_csv(ROOT / "results" / "replication_nse" / "table2_mcsgarch.csv", float_precision="round_trip")
        chk = []
        name_map = {"omega": "omega", "alpha": "alpha", "beta": "beta", "nu": "shape_nu",
                    "arma_phi": "phi_ar1", "arma_theta": "theta_ma1"}
        for w in ("before_stated", "before_inferred", "after"):
            for k in INDICES:
                r = fits[(fits["window"] == w) & (fits["index"] == k) & (fits["model"] == "ref")].iloc[0]
                for mine, theirs in name_map.items():
                    c = t2[(t2["period"] == w) & (t2["index"] == k) & (t2["parameter"] == theirs)]["ours"].iloc[0]
                    chk.append({"check": "Table 2 reference fit", "window": w, "index": k, "quantity": theirs,
                                "this_study": r[mine], "committed": c, "identical": bool(r[mine] == c),
                                "abs_diff": abs(r[mine] - c)})
        chk = pd.DataFrame(chk)
        chk.to_csv(work / "reproduction_table2.csv", index=False, float_format="%.17g")
        print(f"Table 2 reproduction: {int(chk['identical'].sum())} of {len(chk)} values identical; largest "
              f"absolute difference {chk['abs_diff'].max():.3g}", flush=True)
    json.dump({"seconds": round(time.time() - t0, 1), "grid_days": len(G["days"]),
               "windows": {w: [str(d.min().date()), str(d.max().date()), len(d)] for w, d in wins.items()}},
              open(work / "fit_info.json", "w"), indent=2)
    print(f"fit stage done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage boot: the replication's day-block bootstrap, every model on the same resampled days
# ---------------------------------------------------------------------------


def resample_x(e, hh, v):
    """The replication's bootstrap step: the diurnal pattern re-estimated on the resampled days, then the
    normalised residuals (rep._boot_persistence_worker)."""
    s = np.where(v, e**2 / hh[:, None], 0.0).sum(axis=0) / np.maximum(v.sum(axis=0), 1)
    return np.where(v, e / np.sqrt(hh[:, None] * s), 0.0)


def boot_key(args):
    """All bootstrap draws of one (window, index) key: the replication's resampled day indices (same seeds, same
    random stream), and on each resample the reference GARCH(1,1) (exactly the replication's draw) and the
    fixed models refitted with their own daily level."""
    w, k, work, n_boot, models = args
    path = Path(work) / f"boot_{w}_{k}.csv"
    if path.exists():
        return w, k, 0.0
    t0 = time.time()
    st = pickle.load(open(Path(work) / "fits.pkl", "rb"))[(w, k)]
    valid, M = st["valid"], st["M"]
    eps = st["ref"]["eps"]
    T = len(eps)
    n_blocks = int(np.ceil(T / BLOCK_DAYS))
    per = int(np.ceil(n_boot / BOOT_CHUNKS))
    out = []
    for c in range(BOOT_CHUNKS):
        rng = np.random.default_rng(1000 * SEED_INDEX[(w, k)] + c)
        for b in range(per):
            stt = rng.integers(0, T, n_blocks)
            idx = ((stt[:, None] + np.arange(BLOCK_DAYS)[None, :]) % T).ravel()[:T]
            e, v = eps[idx], valid[idx]
            draw = c * per + b
            for name in models:
                tf = time.time()
                hh = st[name]["h"][idx]
                x = resample_x(e, hh, v)
                daily, decay, lev, todayrv = MODELS[name]
                if name == "ref":
                    p = fit_garch11(x.ravel(), weights=v.ravel().astype(float))
                else:
                    p = fit_garch_ext(x, e < 0, v.astype(float), decay, lev, todayrv)
                s = ext_persistence_summary(p, M, SHOCK_MINUTE)
                out.append({"window": w, "index": k, "model": name, "draw": draw, "chunk": c, "seed": 1000 * SEED_INDEX[(w, k)] + c,
                            **{kk: p.get(kk, 0.0 if kk in ("gam_l", "gam_x") else (1.0 if kk == "delta" else np.nan))
                               for kk in ("omega", "alpha", "beta", "gam_l", "gam_x", "delta", "nu", "loglik")},
                            "converged": p.get("converged"), "ab": s["ab"], "total": s["total"], "hl_irf": s["hl_irf"],
                            "rho_irf": s["rho_irf"], "share_close": s["share_close"],
                            "seconds": round(time.time() - tf, 3)})
    df = pd.DataFrame(out)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)
    return w, k, round(time.time() - t0, 1)


def stage_boot(a):
    work = Path(a.work)
    n_boot = SYNTH["n_boot"] if a.synthetic else a.n_boot
    jobs = []
    for key in BOOT_ORDER:
        models = ["ref", "F"] + (["F4"] if key in F4_KEYS else [])
        if a.keys and f"{key[0]}:{key[1]}" not in a.keys:
            continue
        jobs.append((key[0], key[1], str(work), n_boot, models))
    t0 = time.time()
    print(f"P114 bootstrap: {len(jobs)} keys, {n_boot} draws each, workers {a.workers}", flush=True)
    if a.workers == 1:
        res = map(boot_key, jobs)
    else:
        pool = mp.get_context("fork").Pool(a.workers, maxtasksperchild=1)
        res = pool.imap(boot_key, jobs)
    with open(work / "boot_timing.csv", "a") as fh:
        for w, k, secs in res:
            fh.write(f"{w},{k},{secs}\n")
            print(f"  {w} {k}: {secs:.0f}s (elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    if a.workers != 1:
        pool.close()
        pool.join()
    print(f"bootstrap done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Stage lr: likelihood-ratio tests of equal persistence (profile likelihood over a common value)
# ---------------------------------------------------------------------------


def normalised_grid(st: dict, name: str) -> np.ndarray:
    """The (days x minutes) normalised residuals of the full-sample fit of a model (as fitted)."""
    return st["x"][name]


def joint_lr(Xb, vb, pb, Xa, va, pa, flags, measure, M, shock=SHOCK_MINUTE) -> dict:
    """LR test that two samples share one persistence value (each keeps its own other parameters). measure "ab":
    alpha + beta; "irf": the impulse-response half-life. The common value is searched on the log half-life
    scale over [log min - 0.5, log max + 0.5] (as the replication's _joint_lr_worker)."""
    decay, todayrv = flags
    sb = ext_persistence_summary(pb, M, shock)
    sa = ext_persistence_summary(pa, M, shock)
    key = "hl_irf" if measure == "irf" else "hl_ab"
    hb, ha = min(sb[key], HL_CAP), min(sa[key], HL_CAP)
    best = {"b": pb["loglik"], "a": pa["loglik"]}
    trace = []

    def prof(X, v, p, loghl, side):
        hl = float(np.exp(loghl))
        value = hl if measure == "irf" else 0.5 ** (1.0 / hl)
        ll, q, ok = fit_garch_ext_fixed(X, v, decay, todayrv, measure, value, M=M, shock_minute=shock, start=p)
        if np.isfinite(ll):
            best[side] = max(best[side], ll)
        return ll if np.isfinite(ll) else -1e300

    def joint(loghl):
        lb = prof(Xb, vb, pb, loghl, "b")
        la = prof(Xa, va, pa, loghl, "a")
        trace.append((float(np.exp(loghl)), lb, la))
        return -(lb + la)

    lo, hi = np.log(min(hb, ha)) - 0.5, np.log(max(hb, ha)) + 0.5
    res = minimize_scalar(joint, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    lr = 2 * (best["b"] + best["a"] + res.fun)
    return {"common_half_life": float(np.exp(res.x)), "lr_stat": max(float(lr), 0.0),
            "p_value": float(stats.chi2.sf(max(lr, 0.0), 1)), "loglik_before": pb["loglik"],
            "loglik_after": pa["loglik"], "loglik_before_best": best["b"], "loglik_after_best": best["a"],
            "loglik_joint": float(-res.fun), "n_profile_evaluations": len(trace)}


def lr_job(args):
    w, k, model, measure, work = args
    t0 = time.time()
    store = pickle.load(open(Path(work) / "fits.pkl", "rb"))
    sb, sa = store[(w, k)], store[("after", k)]
    M = sb["M"]
    _, decay, lev, todayrv = MODELS[model]
    assert not lev, "the profile likelihood has no leverage term"
    pb, pa = dict(sb[model]["params"]), dict(sa[model]["params"])
    Xb, Xa = normalised_grid(sb, model), normalised_grid(sa, model)
    r = joint_lr(Xb, sb["valid"].astype(float), pb, Xa, sa["valid"].astype(float), pa, (decay, todayrv), measure, M)
    return {"before": w, "index": k, "model": model, "measure": measure, **r, "seconds": round(time.time() - t0, 1)}


def stage_lr(a):
    work = Path(a.work)
    jobs = [(w, k, m, me, str(work)) for (w, k, m, me) in LR_TESTS + LR_CHECKS]
    t0 = time.time()
    out = []
    for j in jobs:
        r = lr_job(j)
        out.append(r)
        print(f"  LR {j[0]} {j[1]} {j[2]} {j[3]}: LR {r['lr_stat']:.3f}, p {r['p_value']:.3g} ({r['seconds']:.0f}s)",
              flush=True)
        pd.DataFrame(out).to_csv(work / "lr.csv", index=False, float_format="%.17g")
    print(f"LR stage done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Stage rolling: persistence and drift in rolling 64-day windows (exploratory)
# ---------------------------------------------------------------------------


def stage_rolling(a):
    work = Path(a.work)
    t0 = time.time()
    label = LABEL_SYN if a.synthetic else LABEL_REAL
    rep._EGARCH_CACHE.clear()
    rows = []
    for k in INDICES:
        if a.synthetic:
            bars = synthetic_bars(SYNTH["n_days"], SYNTH["seed"])[k]
            first_pos, L, step = SYNTH["roll_first"], SYNTH["roll_len"], SYNTH["roll_step"]
        else:
            bars = rp.load_minute_bars(Path(a.raw) / f"{k}_1min.csv", "2017-01-01", DATA_END_ROLL[k])
            L, step = ROLL_LEN, ROLL_STEP
        Gi = index_grid(bars, k)
        T = len(Gi["days"])
        if not a.synthetic:
            first_pos = int(np.searchsorted(Gi["days"], pd.Timestamp(ROLL_FIRST)))
        starts = list(range(first_pos, T - L + 1, step))
        print(f"[{label}] rolling {k}: {T} days, {len(starts)} windows of {L} days every {step}", flush=True)
        for n, s0 in enumerate(starts):
            r = np.arange(s0, s0 + L)
            fw = fit_window(Gi, k, r, models=("ref", "F"))
            d = Gi["days"][r]
            for name in ("ref", "F"):
                rows.append({"data": label, "index": k, "window_no": n, "start": str(d.min().date()),
                             "end": str(d.max().date()), "non_overlapping_subset": n % ROLL_NONOVERLAP_EVERY == 0,
                             **fw[name]["row"]})
            pd.DataFrame(rows).to_csv(work / "rolling.csv", index=False, float_format="%.17g")
        print(f"  {k} done ({time.time() - t0:.0f}s)", flush=True)
    rep._EGARCH_CACHE.clear()
    print(f"rolling stage done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Stage analyze
# ---------------------------------------------------------------------------


def pct(a, q):
    """Percentile over the draws in which the measure could be computed (non-finite draws are dropped and
    counted in persistence_change.csv)."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if len(a) else np.nan


def capped_log_hl(hl):
    return np.log(np.minimum(np.asarray(hl, float), HL_CAP))


def verdict_rule(d_point: float, lo: float, hi: float) -> str:
    """Registered decision rule on the change after minus before of the equivalent persistence (negative =
    shocks die out faster after, the paper's direction)."""
    if hi < 0:
        return "survives"
    if d_point < 0:
        return "weakened"
    if lo > 0:
        return "reversed"
    return "gone"


def size_label(kept: float) -> str:
    if not np.isfinite(kept):
        return "not defined"
    if kept >= 0.5:
        return "keeps most of the fall (kept share at least one half)"
    if kept > 0:
        return "keeps less than half of the fall"
    return "keeps none of the fall"


def boot_p_fell(d: np.ndarray) -> float:
    """One-sided bootstrap p for 'persistence fell': (1 + number of draws with no fall) / (1 + draws)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    return float((1 + np.sum(d >= 0)) / (1 + len(d)))


def stage_analyze(a):
    work = Path(a.work)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    label = LABEL_SYN if a.synthetic else LABEL_REAL
    fits = pd.read_csv(work / "fits.csv", float_precision="round_trip")
    boot = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in sorted(work.glob("boot_*.csv")) if not p.name.startswith("boot_timing")],
                     ignore_index=True) if list(work.glob("boot_*_*.csv")) else pd.DataFrame()
    lr = pd.read_csv(work / "lr.csv", float_precision="round_trip") if (work / "lr.csv").exists() else pd.DataFrame()
    roll = pd.read_csv(work / "rolling.csv", float_precision="round_trip") if (work / "rolling.csv").exists() else pd.DataFrame()

    def pt(w, k, m):
        return fits[(fits["window"] == w) & (fits["index"] == k) & (fits["model"] == m)].iloc[0]

    def dr(w, k, m):
        if boot.empty:
            return None
        d = boot[(boot["window"] == w) & (boot["index"] == k) & (boot["model"] == m)].sort_values("draw")
        return d if len(d) else None

    # ---- persistence by model, window and index
    prow = []
    for _, r in fits.iterrows():
        d = dr(r["window"], r["index"], r["model"])
        e = {"data": label, **{c: r[c] for c in ("window", "index", "start", "end", "model")},
             "model_label": MODEL_LABEL[r["model"]],
             **{c: r[c] for c in ("alpha", "beta", "gam_x", "gam_l", "delta", "nu", "omega", "loglik", "ab", "total",
                                  "hl_ab", "hl_irf", "rho_irf", "share_close", "share_next_open", "hl_irf_1015",
                                  "hl_irf_1415", "drift_sd_log_daily_mean_x2", "drift_ac1_log_daily_mean_x2",
                                  "n_obs", "n_days", "converged")}}
        if d is not None:
            e.update({"boot_draws": len(d), "ab_boot_lo": pct(d["ab"], 2.5), "ab_boot_hi": pct(d["ab"], 97.5),
                      "hl_irf_boot_lo": float(np.exp(pct(capped_log_hl(d["hl_irf"]), 2.5))),
                      "hl_irf_boot_hi": float(np.exp(pct(capped_log_hl(d["hl_irf"]), 97.5))),
                      "hl_irf_boot_median": float(np.exp(pct(capped_log_hl(d["hl_irf"]), 50))),
                      "rho_irf_boot_lo": pct(d["rho_irf"], 2.5), "rho_irf_boot_hi": pct(d["rho_irf"], 97.5),
                      "boot_share_converged": float(d["converged"].astype(bool).mean())})
        prow.append(e)
    pers = pd.DataFrame(prow)
    pers.to_csv(out / "persistence.csv", index=False)

    # ---- change from before to after, with the kept share
    crow = {}
    for w in BEFORE_WINDOWS:
        for k in INDICES:
            for m in ("ref", "F", "F4", "X1", "X2", "X3"):
                b, af = pt(w, k, m), pt("after", k, m)
                e = {"data": label, "before": w, "index": k, "model": m, "model_label": MODEL_LABEL[m],
                     "hl_irf_before": b["hl_irf"], "hl_irf_after": af["hl_irf"],
                     "rho_irf_before": b["rho_irf"], "rho_irf_after": af["rho_irf"],
                     "d_rho_irf": af["rho_irf"] - b["rho_irf"],
                     "ab_before": b["ab"], "ab_after": af["ab"], "d_ab": af["ab"] - b["ab"],
                     "hl_ab_before": b["hl_ab"], "hl_ab_after": af["hl_ab"],
                     "fall_factor_irf": min(b["hl_irf"], HL_CAP) / min(af["hl_irf"], HL_CAP),
                     "log_fall_irf": float(capped_log_hl(b["hl_irf"]) - capped_log_hl(af["hl_irf"]))}
                db, da = dr(w, k, m), dr("after", k, m)
                if db is not None and da is not None:
                    n = min(len(db), len(da))
                    d_rho = da["rho_irf"].to_numpy()[:n] - db["rho_irf"].to_numpy()[:n]
                    d_ab = da["ab"].to_numpy()[:n] - db["ab"].to_numpy()[:n]
                    lf = capped_log_hl(db["hl_irf"].to_numpy()[:n]) - capped_log_hl(da["hl_irf"].to_numpy()[:n])
                    e.update({"boot_draws": n, "boot_draws_not_computed": int(np.sum(np.isnan(d_rho))),
                              "d_rho_irf_lo": pct(d_rho, 2.5), "d_rho_irf_hi": pct(d_rho, 97.5),
                              "share_draws_fell_irf": float(np.mean(d_rho[~np.isnan(d_rho)] < 0)), "p_boot_fell_irf": boot_p_fell(d_rho),
                              "d_ab_lo": pct(d_ab, 2.5), "d_ab_hi": pct(d_ab, 97.5),
                              "share_draws_fell_ab": float(np.mean(d_ab[~np.isnan(d_ab)] < 0)), "p_boot_fell_ab": boot_p_fell(d_ab),
                              "log_fall_irf_lo": pct(lf, 2.5), "log_fall_irf_hi": pct(lf, 97.5),
                              "_lf_draws": lf})
                crow[(w, k, m)] = e
    # kept share against the reference (paired draws: the same resampled days for both models)
    for (w, k, m), e in crow.items():
        ref = crow[(w, k, "ref")]
        e["kept_share_log_fall"] = e["log_fall_irf"] / ref["log_fall_irf"] if ref["log_fall_irf"] != 0 else np.nan
        e["kept_share_d_rho"] = e["d_rho_irf"] / ref["d_rho_irf"] if ref["d_rho_irf"] != 0 else np.nan
        e["kept_share_d_ab"] = e["d_ab"] / ref["d_ab"] if ref["d_ab"] != 0 else np.nan
        if "_lf_draws" in e and "_lf_draws" in ref:
            ks = e["_lf_draws"] / ref["_lf_draws"]
            dl = e["_lf_draws"] - ref["_lf_draws"]
            e.update({"kept_share_log_fall_lo": pct(ks, 2.5), "kept_share_log_fall_hi": pct(ks, 97.5),
                      "log_fall_minus_ref_lo": pct(dl, 2.5), "log_fall_minus_ref_hi": pct(dl, 97.5),
                      "log_fall_minus_ref": e["log_fall_irf"] - ref["log_fall_irf"]})
    for e in crow.values():
        e.pop("_lf_draws", None)
    chg = pd.DataFrame(list(crow.values()))
    if not lr.empty:
        for _, r in lr.iterrows():
            m = (chg["before"] == r["before"]) & (chg["index"] == r["index"]) & (chg["model"] == r["model"])
            chg.loc[m, f"lr_{r['measure']}_stat"] = r["lr_stat"]
            chg.loc[m, f"lr_{r['measure']}_p"] = r["p_value"]
            chg.loc[m, f"lr_{r['measure']}_common_half_life"] = r["common_half_life"]
    chg.to_csv(out / "persistence_change.csv", index=False)

    # ---- tests.csv
    tests = []

    def add(test_id, hyp, role, family, sample, est, stat, p, side, method):
        tests.append({"test_id": test_id, "hypothesis": hyp, "role": role, "sample": sample, "estimate": est,
                      "statistic": stat, "p_raw": p, "sidedness": side, "p_method": method, "_family": family})

    BM = (f"day-block bootstrap as in the replication: {N_BOOT if not a.synthetic else SYNTH['n_boot']} resamples "
          f"of {BLOCK_DAYS}-day circular blocks per window, diurnal pattern and intraday model re-estimated each "
          "time; before and after draws independent, paired by draw number; p = (1 + draws with no fall) / (1 + draws)")
    LM = ("likelihood-ratio test of a common value in the two windows (each keeps its other parameters), profile "
          "likelihood, chi-square(1)")
    WN = {"before_inferred": "inferred before window (14 Nov 2022 to 10 Feb 2023)",
          "before_stated": "stated before window (1 Nov 2023 to 29 Feb 2024)",
          "before_p108": "P108 blind window (22 Nov 2022 to 20 Feb 2023)"}

    def sample(w, k):
        return f"{INDEX_NAME[k]}, {WN[w]} against after window (1 Mar to 31 May 2024), one-minute returns 09:15 to 15:14"

    def c(w, k, m):
        return crow[(w, k, m)]

    def lrrow(w, k, m, me):
        if lr.empty:
            return None
        r = lr[(lr["before"] == w) & (lr["index"] == k) & (lr["model"] == m) & (lr["measure"] == me)]
        return r.iloc[0] if len(r) else None

    HYP_IRF = ("Under the {m} model, the half-life of the intraday impulse response to one standardised shock at 12:15 "
               "is shorter after than before (the paper's direction). Estimate: change after minus before of the "
               "equivalent persistence 0.5 ** (1 / half-life).")
    HYP_AB = ("Under the {m} model, alpha + beta is lower after than before. Estimate: change after minus before.")
    if ("before_inferred", "nifty", "F") in crow and "p_boot_fell_irf" in c("before_inferred", "nifty", "F"):
        e = c("before_inferred", "nifty", "F")
        add("P1_nifty_inferred_F_irf_boot", HYP_IRF.format(m="fixed (X1 + X2 + X3)"), "primary", "P", sample("before_inferred", "nifty"),
            e["d_rho_irf"], e["share_draws_fell_irf"], e["p_boot_fell_irf"], "one-sided (persistence fell)", BM)

    def add_boot(tid, fam, role, w, k, m, measure):
        e = c(w, k, m)
        if f"p_boot_fell_{measure}" not in e or not np.isfinite(e.get(f"p_boot_fell_{measure}", np.nan)):
            return
        hyp = (HYP_IRF if measure == "irf" else HYP_AB).format(m=MODEL_LABEL[m])
        est = e["d_rho_irf"] if measure == "irf" else e["d_ab"]
        add(tid, hyp, role, fam, sample(w, k), est, e[f"share_draws_fell_{measure}"], e[f"p_boot_fell_{measure}"],
            "one-sided (persistence fell)", BM)

    def add_lr(tid, fam, role, w, k, m, measure):
        r = lrrow(w, k, m, measure)
        if r is None:
            return
        e = c(w, k, m)
        hyp = ("Under the {m} model, the impulse-response half-life (shock at 12:15) differs between the windows "
               "(likelihood-ratio test of equal half-life). Estimate: change after minus before of the equivalent "
               "persistence." if measure == "irf" else
               "Under the {m} model, alpha + beta differs between the windows (likelihood-ratio test of equal alpha + "
               "beta). Estimate: change after minus before.").format(m=MODEL_LABEL[m])
        add(tid, hyp, role, fam, sample(w, k), e["d_rho_irf"] if measure == "irf" else e["d_ab"], r["lr_stat"],
            r["p_value"], "two-sided (chi-square(1), as the replication)", LM)

    add_lr("S1_nifty_inferred_F_irf_lr", "S1", "secondary", "before_inferred", "nifty", "F", "irf")
    add_boot("S1_nifty_inferred_F_ab_boot", "S1", "secondary", "before_inferred", "nifty", "F", "ab")
    add_lr("S1_nifty_inferred_F_ab_lr", "S1", "secondary", "before_inferred", "nifty", "F", "ab")
    add_boot("S2_bank_inferred_F_irf_boot", "S2", "secondary", "before_inferred", "banknifty", "F", "irf")
    add_lr("S2_bank_inferred_F_irf_lr", "S2", "secondary", "before_inferred", "banknifty", "F", "irf")
    add_boot("S2_bank_inferred_F_ab_boot", "S2", "secondary", "before_inferred", "banknifty", "F", "ab")
    add_lr("S2_bank_inferred_F_ab_lr", "S2", "secondary", "before_inferred", "banknifty", "F", "ab")
    add_boot("S3_nifty_stated_F_irf_boot", "S3", "secondary", "before_stated", "nifty", "F", "irf")
    add_lr("S3_nifty_stated_F_irf_lr", "S3", "secondary", "before_stated", "nifty", "F", "irf")
    add_boot("S3_bank_stated_F_irf_boot", "S3", "secondary", "before_stated", "banknifty", "F", "irf")
    add_lr("S3_bank_stated_F_irf_lr", "S3", "secondary", "before_stated", "banknifty", "F", "irf")
    add_boot("S4_nifty_inferred_F4_irf_boot", "S4", "secondary", "before_inferred", "nifty", "F4", "irf")
    add_boot("S4_bank_inferred_F4_irf_boot", "S4", "secondary", "before_inferred", "banknifty", "F4", "irf")
    add_boot("R1_nifty_p108_F_irf_boot", "R1", "robustness", "before_p108", "nifty", "F", "irf")
    add_boot("R1_bank_p108_F_irf_boot", "R1", "robustness", "before_p108", "banknifty", "F", "irf")

    # exploratory: rolling windows, drift against persistence
    rc = []
    if not roll.empty:
        for k in INDICES:
            rr = roll[roll["index"] == k]
            if rr.empty:
                continue
            ref = rr[rr["model"] == "ref"].set_index("window_no")
            fm = rr[rr["model"] == "F"].set_index("window_no")
            drift = ref["drift_sd_log_daily_mean_x2"]
            for m, frame, col, what in (("ref", ref, "hl_ab", "the reference's half-life ln(0.5)/ln(alpha + beta)"),
                                        ("F", fm, "hl_irf", "the fixed model's impulse-response half-life")):
                y = capped_log_hl(frame[col])
                rho_all = stats.spearmanr(drift, y).statistic
                sub = frame["non_overlapping_subset"].astype(bool)
                res = stats.spearmanr(drift[sub], y[sub.to_numpy()])
                rc.append({"data": label, "index": k, "model": m, "y": col, "n_windows": len(frame),
                           "spearman_all_windows": rho_all, "n_non_overlapping": int(sub.sum()),
                           "spearman_non_overlapping": res.statistic, "p_non_overlapping": res.pvalue})
                add(f"E1_{k}_{m}_drift_vs_halflife", f"Across rolling 64-day windows, the drift measure (SD of the log daily "
                    f"mean of x^2 under the reference model) is rank-correlated with {what}. Estimate: Spearman rho on "
                    "the non-overlapping windows.", "exploratory", "E", f"{INDEX_NAME[k]}, rolling 64-day windows from "
                    f"{rr['start'].min()} to {rr['end'].max()}, every {ROLL_NONOVERLAP_EVERY}th window (non-overlapping)",
                    res.statistic, res.statistic, res.pvalue, "two-sided", "Spearman rank correlation, exact or t "
                    "approximation (scipy), non-overlapping windows only")
    rcdf = pd.DataFrame(rc)
    rcdf.to_csv(out / "rolling_correlations.csv", index=False)
    if not roll.empty:
        roll.to_csv(out / "rolling.csv", index=False)

    tdf = pd.DataFrame(tests)
    fam_how = {"P": None, "S1": "secondary S1 (Nifty, inferred window, fixed model: other test or measure)",
               "S2": "secondary S2 (Bank Nifty, inferred window, fixed model)",
               "S3": "secondary S3 (stated before window, fixed model, impulse-response half-life)",
               "S4": "secondary S4 (fixed model with leverage, inferred window)",
               "R1": "robustness R1 (P108's blind window, fixed model)", "E": "exploratory"}
    tdf["p_adjusted"] = np.nan
    tdf["adjust_method"] = ""
    for fam, how in fam_how.items():
        msk = tdf["_family"] == fam
        if not msk.any():
            continue
        if fam == "P":
            tdf.loc[msk, "p_adjusted"] = tdf.loc[msk, "p_raw"]
            tdf.loc[msk, "adjust_method"] = "none (one primary test)"
        elif fam == "E":
            tdf.loc[msk, "adjust_method"] = "none (exploratory)"
        else:
            tdf.loc[msk, "p_adjusted"] = holm(tdf.loc[msk, "p_raw"].to_numpy(float))
            tdf.loc[msk, "adjust_method"] = f"Holm over {int(msk.sum())} rows: {how}"
    cols = ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"]
    tdf[cols].to_csv(out / "tests.csv", index=False)

    # ---- verdict
    vrows = []
    for w in BEFORE_WINDOWS:
        for k in INDICES:
            for m in ("F", "F4"):
                e = c(w, k, m)
                if "d_rho_irf_lo" not in e:
                    continue
                v = verdict_rule(e["d_rho_irf"], e["d_rho_irf_lo"], e["d_rho_irf_hi"])
                vrows.append({"data": label, "before": w, "index": k, "model": m,
                              "role": "primary" if (w, k, m) == ("before_inferred", "nifty", "F") else "secondary",
                              "verdict": v, "size": size_label(e.get("kept_share_log_fall", np.nan)),
                              "d_rho_irf": e["d_rho_irf"], "d_rho_irf_lo": e["d_rho_irf_lo"],
                              "d_rho_irf_hi": e["d_rho_irf_hi"], "hl_irf_before": e["hl_irf_before"],
                              "hl_irf_after": e["hl_irf_after"], "kept_share_log_fall": e.get("kept_share_log_fall"),
                              "kept_share_log_fall_lo": e.get("kept_share_log_fall_lo"),
                              "kept_share_log_fall_hi": e.get("kept_share_log_fall_hi")})
    pd.DataFrame(vrows).to_csv(out / "verdict.csv", index=False)

    # ---- impulse responses for a figure and a table
    irf = []
    for w in ("before_inferred", "after", "before_stated"):
        for k in INDICES:
            for m in ("ref", "F", "F4"):
                r = pt(w, k, m)
                p = {"alpha": r["alpha"], "beta": r["beta"], "gam_l": r["gam_l"], "gam_x": r["gam_x"], "delta": r["delta"]}
                resp = ext_impulse_response(p, int(r["M"]), SHOCK_MINUTE, n_steps=3 * int(r["M"]))
                for j in range(len(resp)):
                    irf.append({"data": label, "window": w, "index": k, "model": m, "minutes_after_shock": j + 1,
                                "response_share_of_first": resp[j] / resp[0]})
    pd.DataFrame(irf).to_csv(out / "irf_curves.csv", index=False, float_format="%.8g")
    if not boot.empty:
        boot.to_csv(out / "bootstrap_draws.csv", index=False, float_format="%.12g")
    fits.to_csv(out / "fits.csv", index=False)
    if not lr.empty:
        lr.to_csv(out / "lr.csv", index=False)
    for f in ("reproduction_table2.csv",):
        if (work / f).exists():
            pd.read_csv(work / f, float_precision="round_trip").to_csv(out / f, index=False, float_format="%.17g")
    make_figures(out, label)
    # ---- reproduction of the replication's bootstrap and LR (reference model)
    if not a.synthetic:
        reproduction_checks(out, boot, lr)
    info = {"label": label, "plan_commit": a.plan_commit, "analysis_commit": git_head(), "python": sys.version,
            "numpy": np.__version__, "pandas": pd.__version__, "seconds_analyze": round(time.time() - t0, 1),
            "settings": {"windows": WINDOWS, "n_boot": N_BOOT, "block_days": BLOCK_DAYS, "chunks": BOOT_CHUNKS,
                         "shock_minute": SHOCK_MINUTE, "other_shocks": OTHER_SHOCKS, "models": MODELS,
                         "roll": [ROLL_FIRST, ROLL_LEN, ROLL_STEP, ROLL_NONOVERLAP_EVERY], "hl_cap": HL_CAP}}
    for f in ("fit_info.json", "inputs.json"):
        if (work / f).exists():
            info[f.replace(".json", "")] = json.load(open(work / f))
    for f in ("boot_timing.csv",):
        if (work / f).exists():
            info["boot_timing"] = pd.read_csv(work / f, header=None, names=["window", "index", "seconds"]).to_dict("records")
    json.dump(info, open(out / "run_info.json", "w"), indent=2, default=str)
    print(f"analyze done in {time.time() - t0:.0f}s; verdict:")
    print(pd.DataFrame(vrows).to_string() if vrows else "(no bootstrap)")


def reproduction_checks(out: Path, boot: pd.DataFrame, lr: pd.DataFrame):
    """The reference bootstrap draws and LR statistics against the replication's committed Table 2 files."""
    t2p = pd.read_csv(ROOT / "results" / "replication_nse" / "table2_persistence.csv", float_precision="round_trip")
    t2c = pd.read_csv(ROOT / "results" / "replication_nse" / "table2_persistence_change.csv", float_precision="round_trip")
    rows = []
    for _, r in t2p.iterrows():
        d = boot[(boot["window"] == r["period"]) & (boot["index"] == r["index"]) & (boot["model"] == "ref")]
        if d.empty:
            continue
        for q, col in ((2.5, "persistence_boot_ci95_lo"), (97.5, "persistence_boot_ci95_hi")):
            v = pct(d["ab"], q)
            rows.append({"check": "reference bootstrap percentile of alpha + beta", "window": r["period"],
                         "index": r["index"], "quantity": col, "this_study": v, "committed": r[col],
                         "identical": bool(v == r[col]), "abs_diff": abs(v - r[col])})
    for _, r in t2c.iterrows():
        db = boot[(boot["window"] == r["before"]) & (boot["index"] == r["index"]) & (boot["model"] == "ref")].sort_values("draw")
        da = boot[(boot["window"] == "after") & (boot["index"] == r["index"]) & (boot["model"] == "ref")].sort_values("draw")
        if db.empty or da.empty:
            continue
        n = min(len(db), len(da))
        d = da["ab"].to_numpy()[:n] - db["ab"].to_numpy()[:n]
        for q, col in ((2.5, "diff_boot_ci95_lo"), (97.5, "diff_boot_ci95_hi")):
            v = pct(d, q)
            rows.append({"check": "reference bootstrap interval of the change in alpha + beta", "window": r["before"],
                         "index": r["index"], "quantity": col, "this_study": v, "committed": r[col],
                         "identical": bool(v == r[col]), "abs_diff": abs(v - r[col])})
        if not lr.empty:
            l = lr[(lr["before"] == r["before"]) & (lr["index"] == r["index"]) & (lr["model"] == "ref")]
            if len(l):
                rows.append({"check": "reference LR statistic of equal alpha + beta (this study's profile code)",
                             "window": r["before"], "index": r["index"], "quantity": "lr_equal_persistence_stat",
                             "this_study": l["lr_stat"].iloc[0], "committed": r["lr_equal_persistence_stat"],
                             "identical": bool(l["lr_stat"].iloc[0] == r["lr_equal_persistence_stat"]),
                             "abs_diff": abs(l["lr_stat"].iloc[0] - r["lr_equal_persistence_stat"])})
    pd.DataFrame(rows).to_csv(out / "reproduction_bootstrap_lr.csv", index=False, float_format="%.17g")


def make_figures(out: Path, label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = out / "irf_curves.csv"
    if f.exists():
        d = pd.read_csv(f)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        sty = {("ref", "before_inferred"): ("#2a78d6", "--"), ("ref", "after"): ("#1baf7a", "--"),
               ("F", "before_inferred"): ("#2a78d6", "-"), ("F", "after"): ("#1baf7a", "-")}
        for ax, k in zip(axes, INDICES):
            for (m, w), (col, ls) in sty.items():
                s = d[(d["index"] == k) & (d["model"] == m) & (d["window"] == w)]
                ax.plot(s["minutes_after_shock"], s["response_share_of_first"], color=col, ls=ls,
                        label=f"{'reference' if m == 'ref' else 'fixed model'}, {'before (inferred)' if w != 'after' else 'after'}")
            ax.axhline(0.5, color="grey", lw=0.6)
            ax.axvline(359 - SHOCK_MINUTE, color="grey", lw=0.6, ls=":")
            ax.set_title(INDEX_NAME[k])
            ax.set_xlabel("trading minutes after a shock at 12:15 (dotted line: the close)")
            ax.set_xlim(0, 540)
        axes[0].set_ylabel("response, share of the first minute's")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Impulse response of intraday variance to one shock ({label.split(' (')[0]})")
        fig.tight_layout()
        fig.savefig(out / "fig_irf.png", dpi=120)
        plt.close(fig)
    f = out / "rolling.csv"
    if f.exists():
        d = pd.read_csv(f)
        fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex="col")
        for j, k in enumerate(INDICES):
            r = d[(d["index"] == k) & (d["model"] == "ref")]
            fm = d[(d["index"] == k) & (d["model"] == "F")]
            if r.empty:
                continue
            x = pd.to_datetime(r["start"])
            axes[0, j].plot(x, np.minimum(r["hl_ab"], HL_CAP), label="reference: ln(0.5)/ln(alpha + beta)")
            axes[0, j].plot(pd.to_datetime(fm["start"]), np.minimum(fm["hl_irf"], HL_CAP), label="fixed model: impulse-response half-life")
            axes[0, j].set_yscale("log")
            axes[0, j].set_title(INDEX_NAME[k])
            axes[0, j].set_ylabel("half-life, minutes")
            axes[0, j].legend(fontsize=8)
            axes[1, j].plot(x, r["drift_sd_log_daily_mean_x2"], color="#eb6834")
            axes[1, j].set_ylabel("drift: SD of log daily mean x^2")
            axes[1, j].set_xlabel("first day of the 64-day window")
        fig.suptitle(f"Rolling 64-day windows ({label.split(' (')[0]})")
        fig.tight_layout()
        fig.savefig(out / "fig_rolling.png", dpi=120)
        plt.close(fig)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def stage_inputs(a):
    """SHA-256 of the raw files this study reads, checked against data/MANIFEST.csv."""
    import hashlib
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "data" / "MANIFEST.csv").set_index("path")["sha256"].to_dict()
    res = {}
    for f in ("nifty_1min.csv", "banknifty_1min.csv", "minute_bars_source.json"):
        h = hashlib.sha256()
        with open(Path(a.raw) / f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        res[f] = {"sha256": h.hexdigest(), "manifest": man.get(f), "matches_manifest": man.get(f) == h.hexdigest()}
    json.dump(res, open(work / "inputs.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    if not all(v["matches_manifest"] for v in res.values()):
        raise SystemExit("raw files differ from data/MANIFEST.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("inputs", "fit", "boot", "lr", "rolling", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--work", default=None)
        p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
        p.add_argument("--synthetic", action="store_true", help="simulated two-index market (dry run)")
        if name == "boot":
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--n-boot", type=int, default=N_BOOT)
            p.add_argument("--keys", nargs="+", default=None, help="window:index keys to run (default all)")
        if name == "analyze":
            p.add_argument("--out", default=None)
            p.add_argument("--plan-commit", default=None)
    a = ap.parse_args(argv)
    if a.work is None:
        a.work = str(ROOT / "data" / "processed" / ("p114_synthetic" if a.synthetic else "p114"))
    if getattr(a, "out", "x") is None:
        a.out = str(ROOT / "results" / ("p114_persistence_fixed_model_synthetic" if a.synthetic else "p114_persistence_fixed_model"))
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    {"inputs": stage_inputs, "fit": stage_fit, "boot": stage_boot, "lr": stage_lr, "rolling": stage_rolling,
     "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
