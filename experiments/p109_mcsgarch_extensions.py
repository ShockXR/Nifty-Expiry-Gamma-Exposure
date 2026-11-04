"""P109: can a few parametric extensions of mcsGARCH recover the forecasting hybrid's gain?
(registered plan: docs/prereg/P109_MCSGARCH_EXTENSIONS.md)

P106 found that the hybrid (mcsGARCH times exp(a LightGBM correction)) gets all of its gain over mcsGARCH from
mcsGARCH's own kind of information: realized-volatility history and the time of day. This script fits five
pre-specified parametric extensions of mcsGARCH, each aimed at one suspected misspecification, and their
combination, with the walk-forward refit schedule of the committed runs (refit every 21 trading days on all
earlier days, the same test days), and compares them with mcsGARCH and with the hybrid's committed forecasts.

    x1_har          daily level from a log-HAR of daily realized variance (instead of the daily EGARCH)
    x2_overnight    the intraday component decays toward its average over the night (estimated delta)
    x3_todayrv      GARCH-X: today's running realized variance (normalised) in the intraday recursion
    x4_leverage     an asymmetric (GJR) term in the intraday recursion
    x5_calibration  mcsGARCH times a factor for each anchor minute, the training-window mean of y / forecast
    combination     all five together
    comb_no_x<k>    the combination without extension k (exploratory, Nifty only)

Stages:

    # 1. forecasts (REAL DATA), one invocation per index, 2 worker processes
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index NIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index BANKNIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index FINNIFTY --workers 2
    # 2. losses, tests, tables, verdict
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py analyze --plan-commit <hash>

``check`` compares the rebuilt mcsGARCH (and y) with the committed forecasts and computes no loss.
``--synthetic`` (run, check and analyze) uses a simulated market, with the hybrid fitted by the pipeline's own
walk-forward on it, for a dry run of every step; its output is labelled SYNTHETIC. Per-block forecasts go to
--work (default data/processed/p109, git-ignored); a finished block is not rerun. ``analyze`` writes
results/p109_mcsgarch_extensions/. Tests: tests/test_p109_mcsgarch_extensions.py (synthetic data only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from niftygex.evaluation import REGIMES, daily_losses, diebold_mariano  # noqa: E402
from niftygex.features import CLOSING_AUCTION_START  # noqa: E402
from niftygex.volatility import (McsGarchExt, arma11_mean_square_forecast, daily_egarch_variance,  # noqa: E402
                                 har_daily_variance)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# P106's module brings the pipeline's market loader (rp), the stationary bootstrap (rf), the Newey-West and
# EWMA-fallback helpers (gw), the tests.csv collector, the share interval and the input hashing, so that
# nothing here is a second implementation.
gd = _load("p106_gain_decomposition", ROOT / "experiments" / "p106_gain_decomposition.py")
rp, rf, gw = gd.rp, gd.rf, gd.gw

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

MIN_TRAIN = 150
BLOCK = 21                       # mcsGARCH is refitted before every block of 21 test days, as in the committed runs
EXPIRY_DIURNAL = True            # the committed WalkForwardConfig: separate expiry-day diurnal curve
DIURNAL_SMOOTH = 5               # ... smoothed with a centred 5-minute moving average

# Each variant: daily component, the three flags of the intraday recursion, time-of-day calibration.
SPECS = {
    "mcsgarch":       dict(daily="egarch", decay=False, lev=False, todayrv=False, cal=False),
    "x1_har":         dict(daily="har",    decay=False, lev=False, todayrv=False, cal=False),
    "x2_overnight":   dict(daily="egarch", decay=True,  lev=False, todayrv=False, cal=False),
    "x3_todayrv":     dict(daily="egarch", decay=False, lev=False, todayrv=True,  cal=False),
    "x4_leverage":    dict(daily="egarch", decay=False, lev=True,  todayrv=False, cal=False),
    "x5_calibration": dict(daily="egarch", decay=False, lev=False, todayrv=False, cal=True),
    "combination":    dict(daily="har",    decay=True,  lev=True,  todayrv=True,  cal=True),
    "comb_no_x1":     dict(daily="egarch", decay=True,  lev=True,  todayrv=True,  cal=True),
    "comb_no_x2":     dict(daily="har",    decay=False, lev=True,  todayrv=True,  cal=True),
    "comb_no_x3":     dict(daily="har",    decay=True,  lev=True,  todayrv=False, cal=True),
    "comb_no_x4":     dict(daily="har",    decay=True,  lev=False, todayrv=True,  cal=True),
    "comb_no_x5":     dict(daily="har",    decay=True,  lev=True,  todayrv=True,  cal=False),
}
EXTENSIONS = ["x1_har", "x2_overnight", "x3_todayrv", "x4_leverage", "x5_calibration"]
LEAVE_ONE_OUT = ["comb_no_x1", "comb_no_x2", "comb_no_x3", "comb_no_x4", "comb_no_x5"]
NIFTY_VARIANTS = ["mcsgarch"] + EXTENSIONS + ["combination"] + LEAVE_ONE_OUT
OTHER_VARIANTS = ["mcsgarch"] + EXTENSIONS + ["combination"]
LABELS = {
    "mcsgarch": "mcsGARCH (as committed)",
    "x1_har": "X1 daily level from a log-HAR of daily realized variance",
    "x2_overnight": "X2 overnight decay of the intraday component",
    "x3_todayrv": "X3 today's realized variance in the intraday recursion (GARCH-X)",
    "x4_leverage": "X4 intraday leverage (GJR) term",
    "x5_calibration": "X5 time-of-day calibration factor",
    "combination": "combination of X1 to X5",
    "hybrid": "the hybrid (hybrid_base, committed forecasts)",
    "mcsinfo": "the hybrid that sees only mcsGARCH's information (P106 mcsinfo)",
    **{f"comb_no_x{k}": f"the combination without X{k}" for k in range(1, 6)},
}

TARGET_LABEL = {"next15": "next 15 min", "next30": "next 30 min", "last_hour": "last hour"}
PRIMARY_TARGETS = ["next15", "next30"]
ALPHA = 0.05
REPS = 10000
MEAN_BLOCK = 10.0
BOOT_SEED = 109
MOST = 0.5                        # "most of the gain": a share of one half or more

INDEXES = gd.INDEXES             # committed run folders and last days (Nifty, Bank Nifty, Fin Nifty)
SYNTH = dict(n_days=200, start="2024-06-03", seed=3, min_train=60, block=20, n_seeds=2,
             targets={"next15": dict(horizon=15, step=45), "next30": dict(horizon=30, step=45)})
ID_COLS = ["target", "t", "date", "minute", "horizon", "y"]


# Optional switches of a variant's spec (added for study P116, docs/prereg/P116_BENCHMARK_AS_PRINTED.md). No P109
# variant sets them, so every P109 variant gets the defaults below and runs exactly as before:
#   expiry_diurnal  separate diurnal curve for expiry days (default EXPIRY_DIURNAL, as committed)
#   smooth          centred moving-average window of the diurnal curve (default DIURNAL_SMOOTH, as committed)
#   mean            intraday conditional mean of McsGarch: "zero" (as committed) or "arma11" (the paper's)
#   mean_sq         add the model's expected squared conditional mean over the window to the variance forecast
#                   (niftygex.volatility.arma11_mean_square_forecast); default False. Not part of the fit key:
#                   the same fit gives the forecast with and without it.
SPEC_DEFAULTS = {"expiry_diurnal": EXPIRY_DIURNAL, "smooth": DIURNAL_SMOOTH, "mean": "zero", "mean_sq": False}


def spec_value(variant: str, key: str):
    return SPECS[variant].get(key, SPEC_DEFAULTS[key])


def fit_key(variant: str) -> tuple:
    s = SPECS[variant]
    return (s["daily"], s["decay"], s["lev"], s["todayrv"], spec_value(variant, "expiry_diurnal"),
            spec_value(variant, "smooth"), spec_value(variant, "mean"))


def target_specs(synthetic: bool) -> dict:
    """target -> (horizon, step, evaluation minutes or None), as scripts/run_pipeline.py builds them."""
    if synthetic:
        return {k: (v["horizon"], v["step"], None) for k, v in SYNTH["targets"].items()}
    return {k: (v["horizon"], v["step"], rp.EVAL_MINUTES.get(k)) for k, v in rp.TARGETS.items()}


def dataset_rows(T: int, M: int, end: int, horizon: int, step: int) -> pd.DataFrame:
    """The (t, minute, horizon) rows that features.build_dataset makes for days 1 .. end-1, in its order
    (sorted by day, then anchor), so that forecast_variance sees exactly the arrays run_walkforward gives it."""
    anchors = list(range(0, M - horizon + 1, step))
    t = np.repeat(np.arange(1, end), len(anchors))
    i = np.tile(np.array(anchors), end - 1)
    j = np.minimum(i + horizon, M)
    return pd.DataFrame({"t": t, "minute": i, "horizon": j - i})


def realized_target(R: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """y = sum of squared one-minute returns over [i, i + horizon), computed as features.build_dataset does."""
    T = R.shape[0]
    cum = np.concatenate([np.zeros((T, 1)), np.cumsum(R**2, axis=1)], axis=1)
    t, i = rows["t"].to_numpy(), rows["minute"].to_numpy()
    return cum[t, i + rows["horizon"].to_numpy()] - cum[t, i]


def calibration_factors(y: np.ndarray, f: np.ndarray, minute: np.ndarray, train: np.ndarray) -> dict:
    """X5: for each anchor minute, the QLIKE-optimal constant factor on the training rows, mean(y / f)."""
    out = {}
    for m in np.unique(minute[train]):
        sel = train & (minute == m)
        out[int(m)] = float(np.mean(y[sel] / f[sel]))
    return out


# ---------------------------------------------------------------------------
# Stage 1: forecasts, one job per refit block
# ---------------------------------------------------------------------------

_STATE: dict = {}   # filled in the parent before the workers are forked


def block_file(work: Path, index: str, start: int) -> Path:
    return Path(work) / f"{index}_block{start:04d}.csv"


def run_block(job):
    """Refit every variant on days 0 .. start-1 and forecast every target's rows of days start .. end-1.
    Mirrors run_walkforward: the daily EGARCH and mcsGARCH fitted on R[:start], filtered over all days."""
    index, start, work = job
    st = _STATE
    R, is_exp, days = st["R"], st["is_exp"], st["days"]
    T, M = R.shape
    end = min(start + st["block"], T)
    t0 = time.time()
    daily = R.sum(axis=1)
    h = {}
    h["egarch"], info_eg = daily_egarch_variance(daily, train_size=start)
    h["har"], info_har = har_daily_variance((R**2).sum(axis=1), train_size=start)
    fits: dict = {}
    for v in st["variants"]:
        fits.setdefault(fit_key(v), []).append(v)
    rows = {}
    for tname, (hz, step, ev) in st["targets"].items():
        d = dataset_rows(T, M, end, hz, step)
        d["y"] = realized_target(R, d)
        rows[tname] = d
    out = {tname: {} for tname in rows}
    params = []
    fit_seconds = {}
    for key, vs in fits.items():
        dkind, decay, lev, todayrv, exp_d, smooth, mean = key
        tf = time.time()
        mdl = McsGarchExt(expiry_diurnal=exp_d, smooth=smooth, mean=mean, overnight_decay=decay,
                          leverage=lev, garchx_today=todayrv)
        mdl.fit(R[:start], h[dkind][:start], is_exp[:start])
        filt = mdl.filter(R, h[dkind], is_exp)
        fit_seconds["+".join(vs)] = round(time.time() - tf, 2)
        p = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v)
             for k, v in mdl.params.items() if k != "tstats"}
        for tname, d in rows.items():
            f = mdl.forecast_variance(filt, h[dkind], d["t"].to_numpy(), d["minute"].to_numpy(), d["horizon"].to_numpy())
            msq = None
            if any(spec_value(v, "mean_sq") for v in vs):   # P116 only; no P109 variant sets mean_sq
                msq = arma11_mean_square_forecast(mdl, R, filt, h[dkind], d["t"].to_numpy(), d["minute"].to_numpy(),
                                                  d["horizon"].to_numpy())
            f_var = f
            train = d["t"].to_numpy() < start
            for v in vs:
                f = f_var + msq if spec_value(v, "mean_sq") else f_var
                if SPECS[v]["cal"]:
                    c = calibration_factors(d["y"].to_numpy(), f, d["minute"].to_numpy(), train)
                    out[tname][v] = f * d["minute"].map(c).to_numpy(float)
                    params.append({"index": index, "start": start, "variant": v, "target": tname, "kind": "calibration",
                                   **{f"c_{m}": val for m, val in c.items()}})
                else:
                    out[tname][v] = f
        for v in vs:
            params.append({"index": index, "start": start, "variant": v, "target": "all", "kind": "model",
                           "daily": dkind, "daily_method": (info_eg if dkind == "egarch" else info_har)["method"],
                           **({f"har_{k}": val for k, val in info_har["params"].items()} if dkind == "har" else {}),
                           **{k: p.get(k) for k in ("omega", "alpha", "beta", "gam_l", "gam_x", "delta", "nu",
                                                    "persistence", "qbar", "loglik", "nobs", "converged", "start")
                              if k in p and k != "start"},
                           "ext_start_used": mdl.params.get("start"),   # which registered start won (extended fits)
                           # P116 switches and the fitted ARMA(1,1) mean, only for variants that set a switch
                           **({"expiry_diurnal": exp_d, "smooth": smooth, "mean": mean,
                               "mean_sq": spec_value(v, "mean_sq"),
                               **{f"arma_{k}": float(val) for k, val in mdl.mean_params.items()}}
                              if any(k in SPECS[v] for k in SPEC_DEFAULTS) else {})})
    frames = []
    for tname, d in rows.items():
        ev = st["targets"][tname][2]
        keep = d["t"].to_numpy() >= start
        if ev is not None:
            keep &= d["minute"].isin(ev).to_numpy()
        f = d.loc[keep, ["t", "minute", "horizon", "y"]].copy()
        f.insert(0, "target", tname)
        f.insert(2, "date", pd.DatetimeIndex(days)[f["t"].to_numpy()].strftime("%Y-%m-%d"))
        for v in st["variants"]:
            f[v] = out[tname][v][keep]
        frames.append(f)
    res = pd.concat(frames, ignore_index=True)
    path = block_file(work, index, start)
    tmp = path.with_suffix(".tmp")
    res.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)
    secs = round(time.time() - t0, 1)
    pd.DataFrame(params).to_csv(Path(work) / f"{index}_block{start:04d}_params.csv", index=False)
    with open(Path(work) / f"{index}_timing.csv", "a") as fh:
        fh.write(f"{index},{start},{end - start},{secs},{json.dumps(fit_seconds).replace(',', ';')},{os.getpid()}\n")
    return index, start, end - start, secs


def synthetic_market():
    from niftygex.simulate import SimConfig, simulate_market
    return simulate_market(SimConfig(n_days=SYNTH["n_days"], start=SYNTH["start"], seed=SYNTH["seed"], kappa=0.0))


def synthetic_committed(m: dict, work: Path):
    """SYNTHETIC stand-in for the committed run: the pipeline's own walk-forward (mcsgarch and hybrid_base) on
    the simulated market, with SYNTH's small settings, written as work/SYNTH_committed_<target>.csv."""
    from niftygex.features import GEX_SETS, build_dataset, build_gex_panel
    from niftygex.forecast import WalkForwardConfig, run_walkforward
    from niftygex.implied import build_implied_panel
    panel, implied = build_gex_panel(m), build_implied_panel(m)
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle"}
    for tname, spec in SYNTH["targets"].items():
        ds = build_dataset(m, panel, implied=implied, **spec)
        cfg = WalkForwardConfig(models=["mcsgarch", "hybrid_base"], gex_sets=sets, min_train=SYNTH["min_train"],
                                block=SYNTH["block"], mcs_refit_every=SYNTH["block"], n_seeds=SYNTH["n_seeds"], seed=7)
        pred, _ = run_walkforward(m, ds, cfg)
        pred["date"] = pd.DatetimeIndex(pred["date"]).strftime("%Y-%m-%d")
        pred.to_csv(Path(work) / f"SYNTH_committed_{tname}.csv", index=False, float_format="%.17g")


def committed_path(index: str, target: str, work: Path) -> Path:
    if index == "SYNTH":
        return Path(work) / f"SYNTH_committed_{target}.csv"
    return INDEXES[index]["run"] / f"forecasts_{target}.csv"


def stage_run(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    index = "SYNTH" if a.synthetic else a.index
    label = "SYNTHETIC (simulated market, not real data)" if a.synthetic else "REAL DATA"
    t0 = time.time()
    if a.synthetic:
        m = synthetic_market()
        min_train, block = SYNTH["min_train"], SYNTH["block"]
        if not all(committed_path("SYNTH", t, work).exists() for t in SYNTH["targets"]):
            synthetic_committed(m, work)
    else:
        info = json.load(open(INDEXES[index]["run"] / "run_info.json"))
        ns = argparse.Namespace(source="nse", raw=a.raw, symbol=index, oi_unit=gd.PIPELINE_SETTINGS["oi_unit"],
                                lot_size=None, rate=gd.PIPELINE_SETTINGS["rate"], start=gd.PIPELINE_SETTINGS["start"],
                                end=INDEXES[index]["end"], min_coverage=gd.PIPELINE_SETTINGS["min_coverage"])
        m = rp.load_market(ns)
        last = str(pd.Timestamp(m["days"][-1]).date())
        bad = gd.check_run_settings(index, info, last)
        if bad or len(m["days"]) != info["days"][2]:
            raise SystemExit(f"settings differ from the committed run: {bad}, days {len(m['days'])} vs {info['days'][2]}")
        min_train, block = gd.PIPELINE_SETTINGS["min_train"], gd.PIPELINE_SETTINGS["block"]
        hp = work / f"inputs_{index}.json"
        if not hp.exists():
            hp.write_text(json.dumps(gd.input_hashes(Path(a.raw), index), indent=2))
    R = m["R"]
    T = R.shape[0]
    variants = a.variants or (NIFTY_VARIANTS if index in ("NIFTY", "SYNTH") else OTHER_VARIANTS)
    starts = list(range(min_train, T, block))
    if a.blocks:
        starts = [s for s in starts if s in set(a.blocks)]
    jobs = [(index, s, str(work)) for s in starts if not block_file(work, index, s).exists()]
    print(f"[{label}] {getattr(a, 'study', 'P109 mcsGARCH extensions')}, stage 1, {index}: {len(m['days'])} days, "
          f"{len(starts)} refit blocks, "
          f"{len(jobs)} to run; variants {variants}; workers {a.workers} (market loaded in {time.time() - t0:.0f}s)",
          flush=True)
    _STATE.update(R=R, is_exp=m["labels"]["is_expiry"].to_numpy(), days=m["days"], block=block,
                  variants=variants, targets=target_specs(a.synthetic))
    tim = work / f"{index}_timing.csv"
    if not tim.exists():
        tim.write_text("index,start,days,seconds,fit_seconds,pid\n")
    done = 0

    def report(res):
        ix, s, nd, secs = res
        print(f"  [{done}/{len(jobs)}] {ix} block {s} ({nd} days): {secs:.0f}s (elapsed {(time.time() - t0) / 60:.1f} min)",
              flush=True)

    if a.workers == 1:
        for j in jobs:
            done += 1
            report(run_block(j))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=a.workers, maxtasksperchild=1) as pool:
            for res in pool.imap(run_block, jobs, chunksize=1):
                done += 1
                report(res)
    print(f"stage 1 ({index}) done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Assembling the work files; reproduction check (no losses)
# ---------------------------------------------------------------------------


def assemble(work: Path, index: str) -> pd.DataFrame:
    files = sorted(Path(work).glob(f"{index}_block[0-9][0-9][0-9][0-9].csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([gd.read_csv_exact(f) for f in files], ignore_index=True)


def merged(work: Path, index: str, target: str) -> pd.DataFrame:
    """This study's forecasts of one target joined with the committed run's identifiers, y, mcsgarch and
    hybrid_base (renamed committed_mcsgarch, hybrid). Rows: the committed run's test anchors."""
    a = assemble(work, index)
    if a.empty:
        return a
    a = a[a["target"] == target].drop(columns="target")
    c = gd.read_csv_exact(committed_path(index, target, work))
    c["date"] = pd.to_datetime(c["date"]).dt.strftime("%Y-%m-%d")
    keep = ["t", "date", "minute", "horizon", "is_expiry", "regime", "y", "mcsgarch", "hybrid_base"]
    c = c[[k for k in keep if k in c]].rename(columns={"y": "committed_y", "mcsgarch": "committed_mcsgarch",
                                                       "hybrid_base": "hybrid"})
    out = c.merge(a.drop(columns=["date", "horizon"]), on=["t", "minute"], how="left", validate="one_to_one")
    return out.sort_values(["t", "minute"]).reset_index(drop=True)


def reproduction_check(w: pd.DataFrame, label: str) -> dict:
    """mcsGARCH and y rebuilt by this script against the committed forecasts (forecasts only, no loss)."""
    ok = w["mcsgarch"].notna().to_numpy()
    out = {"check": label, "rows_committed": len(w), "rows_rebuilt": int(ok.sum())}
    for name, (x, y) in {"y": ("y", "committed_y"), "mcsgarch": ("mcsgarch", "committed_mcsgarch")}.items():
        a, b = w.loc[ok, x].to_numpy(float), w.loc[ok, y].to_numpy(float)
        out[f"{name}_n_identical"] = int(np.sum(a == b))
        out[f"{name}_max_abs_rel_diff"] = float(np.max(np.abs(a / b - 1))) if ok.any() else np.nan
    return out


def stage_check(a):
    work = Path(a.work)
    rows = []
    indexes = ["SYNTH"] if a.synthetic else a.indexes
    for index in indexes:
        for t in (SYNTH["targets"] if index == "SYNTH" else rp.TARGETS):
            if not committed_path(index, t, work).exists():
                continue
            w = merged(work, index, t)
            if w.empty:
                continue
            r = reproduction_check(w, f"{index} {t}")
            rows.append(r)
            print(r, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

DM_METHOD = "Diebold-Mariano on daily mean QLIKE, Newey-West, HLN factor, Student t(n-1)"


def day_frame(w: pd.DataFrame, models: list, minutes=None) -> pd.DataFrame:
    """Daily mean QLIKE of each model (over the anchors in ``minutes``, all if None) with the day's labels."""
    x = w if minutes is None else w[w["minute"].isin(minutes)]
    counts = x.groupby("date").size()
    if counts.nunique() != 1:
        raise ValueError("days have different numbers of anchors")
    L = daily_losses(x, models, "qlike")
    cal = x.groupby("date").agg(t=("t", "first"), is_expiry=("is_expiry", "first"), regime=("regime", "first"))
    return L.join(cal)


def points(num: float, den: float) -> float:
    return 100.0 * num / den


def gain(D: pd.DataFrame, m: str) -> float:
    """Gain over mcsGARCH in points of mcsGARCH's mean daily QLIKE (positive: better than mcsGARCH)."""
    return points(D["mcsgarch"].mean() - D[m].mean(), D["mcsgarch"].mean())


def recovered_share(D: pd.DataFrame, m: str, ref: str = "hybrid", reps: int = REPS, rng=None):
    """Share of the reference's gain over mcsGARCH that model m recovers, (L_mcs - L_m) / (L_mcs - L_ref),
    with a 95% stationary-bootstrap percentile interval (days resampled in blocks of mean length 10)."""
    rng = rng if rng is not None else np.random.default_rng(BOOT_SEED)
    x = np.column_stack([D["mcsgarch"].to_numpy(), D[ref].to_numpy(), D[m].to_numpy()])
    share = (x[:, 0].mean() - x[:, 2].mean()) / (x[:, 0].mean() - x[:, 1].mean())
    idx = rf.stationary_bootstrap_indices(len(x), reps, MEAN_BLOCK, rng)
    mb = rf.bootstrap_means(x, idx)
    s = (mb[:, 0] - mb[:, 2]) / (mb[:, 0] - mb[:, 1])
    return float(share), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def sample_text(index: str, target: str, D: pd.DataFrame, what: str) -> str:
    d0, d1 = pd.Timestamp(D.index.min()).date(), pd.Timestamp(D.index.max()).date()
    lab = "Simulated market" if index == "SYNTH" else INDEXES[index]["label"]
    return f"{lab}, {TARGET_LABEL[target]}, {what}, {len(D)} test days {d0} to {d1}"


def primary_verdict(p_holm: dict, share: dict) -> dict:
    """The registered decision rule (plan Section 4) applied to the two primary rows.

    share[target] = (estimate, ci_low, ci_high)."""
    beats = {t: bool(np.isfinite(p_holm[t]) and p_holm[t] < ALPHA) for t in PRIMARY_TARGETS}
    pt = {t: share[t][0] for t in PRIMARY_TARGETS}
    lo = {t: share[t][1] for t in PRIMARY_TARGETS}
    hi = {t: share[t][2] for t in PRIMARY_TARGETS}
    if all(beats.values()) and all(lo[t] >= MOST for t in PRIMARY_TARGETS):
        v = "supported: recovers most of the gain (interval above one half at both horizons)"
    elif all(beats.values()) and all(pt[t] >= MOST for t in PRIMARY_TARGETS):
        v = "supported in the point estimate only: recovers most of the gain, but an interval reaches below one half"
    elif all(hi[t] < MOST for t in PRIMARY_TARGETS):
        v = ("contradicted: recovers less than half of the gain (interval below one half at both horizons)"
             + ("; it still beats mcsGARCH" if any(beats.values()) else "; it does not beat mcsGARCH"))
    elif any(beats.values()):
        v = "inconclusive: beats mcsGARCH, but the share recovered is not clearly above or below one half"
    else:
        v = "inconclusive: does not beat mcsGARCH, and the share recovered is not clearly below one half"
    return {"verdict": v, **{f"beats_mcsgarch_{t}": beats[t] for t in PRIMARY_TARGETS},
            **{f"share_{t}": pt[t] for t in PRIMARY_TARGETS}, **{f"share_low_{t}": lo[t] for t in PRIMARY_TARGETS},
            **{f"share_high_{t}": hi[t] for t in PRIMARY_TARGETS}}


# ---------------------------------------------------------------------------
# Stage 2: analysis
# ---------------------------------------------------------------------------


def stage_analyze(a):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    synth = a.synthetic
    label = "SYNTHETIC (simulated market, not real data)" if synth else "REAL DATA"
    nifty = "SYNTH" if synth else "NIFTY"
    targets = list(SYNTH["targets"]) if synth else list(rp.TARGETS)
    reps = a.reps
    print(f"[{label}] P109 mcsGARCH extensions, stage 2 (docs/prereg/P109_MCSGARCH_EXTENSIONS.md)", flush=True)
    Rw = gd.Rows()
    gain_rows, share_rows, anchor_rows, regime_rows, repro_rows, verdict_rows = [], [], [], [], [], []
    rng_counter = [0]

    def rng():
        rng_counter[0] += 1
        return np.random.default_rng([BOOT_SEED, rng_counter[0]])

    def dm_row(test_id, hyp, role, family, D, index, target, what, worse, better, est, extra=None):
        dm = diebold_mariano(D[worse], D[better], alternative="greater")
        Rw.add(test_id, hyp, role, sample_text(index, target, D, what), est, dm["stat"], dm["p"],
               f"one-sided ({better} has lower loss than {worse})", DM_METHOD, family, index=index, target=target,
               subset=what, worse=worse, better=better, days=dm["n"], **(extra or {}))
        return dm

    W, D = {}, {}
    for index in ([nifty] if synth else ["NIFTY", "BANKNIFTY", "FINNIFTY"]):
        variants = NIFTY_VARIANTS if index in ("NIFTY", "SYNTH") else OTHER_VARIANTS
        for t in (targets if index in ("NIFTY", "SYNTH") else PRIMARY_TARGETS):
            if not committed_path(index, t, work).exists():
                continue
            w = merged(work, index, t)
            if w.empty or w[variants].isna().any().any():
                print(f"  {index} {t}: work files missing or incomplete; not computed", flush=True)
                continue
            rc = reproduction_check(w, f"{index} {t} vs {committed_path(index, t, work).relative_to(ROOT) if not synth else 'synthetic run'}")
            repro_rows.append(rc)
            if rc["y_n_identical"] != len(w) or rc["mcsgarch_n_identical"] != len(w):
                # registered check (plan Section 6): the rebuilt benchmark must equal the committed one exactly
                print(f"  WARNING {index} {t}: rebuilt y or mcsGARCH differs from the committed forecasts: {rc}", flush=True)
                if not a.allow_mismatch:
                    raise SystemExit("reproduction check failed; rerun with --allow-mismatch to report anyway")
            w["y"] = w["committed_y"]          # losses use the committed target (identical when the check passes)
            W[(index, t)] = w
            D[(index, t)] = day_frame(w, ["hybrid"] + variants)
    if (nifty, PRIMARY_TARGETS[0] if not synth else targets[0]) not in D:
        raise SystemExit("no finished work files for the primary index")

    # P106's mcsinfo hybrid and P99's seed-averaged hybrid (Nifty, committed daily losses)
    if not synth:
        for t in targets:
            if ("NIFTY", t) not in D:
                continue
            d = D[("NIFTY", t)]
            p106 = ROOT / "results" / "p106_gain_decomposition" / f"daily_losses_nifty_{t}_s7.csv"
            if p106.exists():
                x = pd.read_csv(p106, index_col="date", float_precision="round_trip")
                x.index = pd.to_datetime(x.index).strftime("%Y-%m-%d")
                d["mcsinfo"] = x["mcsinfo"].reindex(d.index)
                repro_rows.append({"check": f"NIFTY {t}: P106 daily QLIKE of mcsGARCH vs this study's",
                                   "rows_committed": len(x), "rows_rebuilt": int(d["mcsinfo"].notna().sum()),
                                   "mcsgarch_daily_max_abs_rel_diff": float(np.max(np.abs(
                                       x["mcsgarch"].reindex(d.index).to_numpy() / d["mcsgarch"].to_numpy() - 1)))})
            p99 = ROOT / "results" / "seed_bootstrap" / f"daily_losses_{t}.csv"
            if p99.exists():
                x = pd.read_csv(p99, index_col="date", float_precision="round_trip")
                x.index = pd.to_datetime(x.index).strftime("%Y-%m-%d")
                cols = [c for c in x if c.startswith("qlike_hybrid_s")]
                d["hybrid_seedavg"] = x[cols].mean(axis=1).reindex(d.index)
                repro_rows.append({"check": f"NIFTY {t}: P99 seed-7 hybrid daily QLIKE vs this study's hybrid",
                                   "rows_committed": len(x), "rows_rebuilt": int(d["hybrid_seedavg"].notna().sum()),
                                   "hybrid_daily_max_abs_rel_diff": float(np.max(np.abs(
                                       x["qlike_hybrid_s7"].reindex(d.index).to_numpy() / d["hybrid"].to_numpy() - 1))),
                                   "seed_sets_averaged": len(cols)})

    def gains_and_shares(index, t, d, models, subset="all days", refs=("hybrid",)):
        for m in models:
            row = {"index": index, "target": t, "subset": subset, "model": m, "days": len(d),
                   "qlike_mean": d[m].mean(), "gain_points": gain(d, m),
                   "pct_vs_mcsgarch": 100 * (d[m].mean() / d["mcsgarch"].mean() - 1)}
            gain_rows.append(row)
            for ref in refs:
                if ref not in d or m == ref or d[ref].isna().any():
                    continue
                sh, lo, hi = recovered_share(d, m, ref, reps=reps, rng=rng())
                share_rows.append({"index": index, "target": t, "subset": subset, "model": m, "reference": ref,
                                   "reference_gain_points": gain(d, ref), "model_gain_points": gain(d, m),
                                   "share": sh, "ci_low": lo, "ci_high": hi})

    # ---------------- primary and the Nifty secondaries
    share_primary = {}
    for t in targets:
        if (nifty, t) not in D:
            continue
        d = D[(nifty, t)]
        refs = ("hybrid", "mcsinfo", "hybrid_seedavg") if not synth else ("hybrid",)
        gains_and_shares(nifty, t, d, ["hybrid"] + NIFTY_VARIANTS[1:] + [c for c in ("mcsinfo", "hybrid_seedavg") if c in d],
                         refs=refs)
        if t in PRIMARY_TARGETS or (synth and t in targets):
            dm_row(f"P1_{t}_combination", f"The combination of the five extensions has lower mean QLIKE than mcsGARCH"
                   f" ({TARGET_LABEL[t]}, all days). Estimate: its gain over mcsGARCH, points of mcsGARCH's QLIKE.",
                   "primary", "primary", d, nifty, t, "all days", "mcsgarch", "combination", gain(d, "combination"))
            sh = [r for r in share_rows if r["index"] == nifty and r["target"] == t and r["model"] == "combination"
                  and r["reference"] == "hybrid" and r["subset"] == "all days"][0]
            share_primary[t] = (sh["share"], sh["ci_low"], sh["ci_high"])
            for x in EXTENSIONS:
                dm_row(f"S1_{t}_{x}", f"{LABELS[x]} alone has lower mean QLIKE than mcsGARCH ({TARGET_LABEL[t]}, all"
                       " days). Estimate: gain over mcsGARCH, points.", "secondary", "S1_single_extensions", d, nifty, t,
                       "all days", "mcsgarch", x, gain(d, x))
            dm_row(f"S2_{t}_hybrid_vs_combination", f"The hybrid has lower mean QLIKE than the combination"
                   f" ({TARGET_LABEL[t]}, all days). Estimate: the hybrid's gain minus the combination's, points.",
                   "secondary", "S2_hybrid_vs_combination", d, nifty, t, "all days", "combination", "hybrid",
                   gain(d, "hybrid") - gain(d, "combination"))
            # S4: the 09:15 anchor and the other anchors
            w = W[(nifty, t)]
            for sname, mins in (("09:15 anchor only", [0]), ("without the 09:15 anchor",
                                                             sorted(set(w["minute"]) - {0}))):
                dd = day_frame(w, ["hybrid"] + NIFTY_VARIANTS, minutes=mins)
                key = "0915" if mins == [0] else "rest"
                dm_row(f"S4_{t}_{key}", f"The combination has lower mean QLIKE than mcsGARCH ({TARGET_LABEL[t]},"
                       f" {sname}). Estimate: gain over mcsGARCH, points of mcsGARCH's QLIKE on these anchors.",
                       "secondary", "S4_anchor", dd, nifty, t, sname, "mcsgarch", "combination", gain(dd, "combination"))
                gains_and_shares(nifty, t, dd, ["hybrid"] + NIFTY_VARIANTS[1:], subset=sname)
                for x in EXTENSIONS:
                    dm_row(f"E2_{t}_{key}_{x}", f"{LABELS[x]} alone has lower mean QLIKE than mcsGARCH"
                           f" ({TARGET_LABEL[t]}, {sname}).", "exploratory", "E2_singles_by_anchor", dd, nifty, t, sname,
                           "mcsgarch", x, gain(dd, x))
            # robustness: without the EWMA-fallback days, before the closing auction
            if not synth:
                ew = gw.ewma_fallback_starts(INDEXES["NIFTY"]["run"])
                tt = d["t"].to_numpy()
                ewm = np.any([(tt >= s0) & (tt < s0 + BLOCK) for s0 in ew], axis=0) if ew else np.zeros(len(d), bool)
                pre = pd.to_datetime(d.index).to_numpy() < np.datetime64(pd.Timestamp(CLOSING_AUCTION_START))
                for fam, sname, mask in (("R1_excl_ewma", "without the 42 EWMA-fallback days", ~ewm),
                                         ("R2_pre_auction", "before the closing auction", pre)):
                    dd = d[mask]
                    dm_row(f"{fam.split('_')[0]}_{t}_combination", f"As P1_{t}_combination, {sname}.", "robustness", fam,
                           dd, nifty, t, sname, "mcsgarch", "combination", gain(dd, "combination"))
                    gains_and_shares(nifty, t, dd, ["hybrid", "combination"], subset=sname)
                if "hybrid_seedavg" in d and d["hybrid_seedavg"].notna().all():
                    dm_row(f"R3_{t}_seedavg_hybrid_vs_combination", f"As S2_{t}, with the hybrid's daily QLIKE averaged"
                           " over P99's 21 seed sets.", "robustness", "R3_seed_averaged_hybrid", d, nifty, t,
                           "all days, seed-averaged hybrid", "combination", "hybrid_seedavg",
                           gain(d, "hybrid_seedavg") - gain(d, "combination"))
            # exploratory: leave one extension out of the combination; expiry days; mcsinfo hybrid
            for k, v in enumerate(LEAVE_ONE_OUT, start=1):
                dm_row(f"E1_{t}_{v}", f"Removing X{k} from the combination raises its mean QLIKE ({TARGET_LABEL[t]}, all"
                       " days). Estimate: the rise, points of mcsGARCH's QLIKE.", "exploratory", "E1_leave_one_out", d,
                       nifty, t, "all days", v, "combination", points(d[v].mean() - d["combination"].mean(),
                                                                      d["mcsgarch"].mean()))
            de = d[d["is_expiry"].to_numpy(bool)]
            dm_row(f"E3_{t}_expiry_combination", f"The combination beats mcsGARCH on expiry days ({TARGET_LABEL[t]}).",
                   "exploratory", "E3_expiry", de, nifty, t, "expiry days", "mcsgarch", "combination",
                   gain(de, "combination"))
            dm_row(f"E3_{t}_expiry_hybrid_vs_combination", f"The hybrid beats the combination on expiry days"
                   f" ({TARGET_LABEL[t]}).", "exploratory", "E3_expiry", de, nifty, t, "expiry days", "combination",
                   "hybrid", gain(de, "hybrid") - gain(de, "combination"))
            gains_and_shares(nifty, t, de, ["hybrid"] + NIFTY_VARIANTS[1:], subset="expiry days")
            if "mcsinfo" in d and d["mcsinfo"].notna().all():
                dm_row(f"E5_{t}_mcsinfo_vs_combination", f"P106's mcsinfo hybrid beats the combination ({TARGET_LABEL[t]}).",
                       "exploratory", "E5_mcsinfo", d, nifty, t, "all days", "combination", "mcsinfo",
                       gain(d, "mcsinfo") - gain(d, "combination"))
        else:   # last hour
            dm_row(f"S3_{t}_combination", "The combination has lower mean QLIKE than mcsGARCH (last hour, 14:30 forecast)."
                   " Estimate: gain over mcsGARCH, points.", "secondary", "S3_last_hour", d, nifty, t, "all days",
                   "mcsgarch", "combination", gain(d, "combination"))
            dm_row(f"S3_{t}_hybrid_vs_combination", "The hybrid has lower mean QLIKE than the combination (last hour)."
                   " Estimate: the hybrid's gain minus the combination's, points.", "secondary", "S3_last_hour", d, nifty,
                   t, "all days", "combination", "hybrid", gain(d, "hybrid") - gain(d, "combination"))
            for x in EXTENSIONS:
                dm_row(f"E4_{t}_{x}", f"{LABELS[x]} alone beats mcsGARCH (last hour).", "exploratory", "E4_last_hour",
                       d, nifty, t, "all days", "mcsgarch", x, gain(d, x))
            for k, v in enumerate(LEAVE_ONE_OUT, start=1):
                dm_row(f"E1_{t}_{v}", f"Removing X{k} from the combination raises its mean QLIKE (last hour).",
                       "exploratory", "E1_leave_one_out", d, nifty, t, "all days", v, "combination",
                       points(d[v].mean() - d["combination"].mean(), d["mcsgarch"].mean()))
        # descriptive: gain by anchor minute and by regime
        w = W[(nifty, t)]
        for mnt in sorted(set(w["minute"])):
            dd = day_frame(w, ["hybrid"] + NIFTY_VARIANTS, minutes=[mnt])
            anchor_rows.append({"index": nifty, "target": t, "minute": mnt,
                                "clock": (pd.Timestamp("2000-01-01 09:15") + pd.Timedelta(minutes=int(mnt))).strftime("%H:%M"),
                                "days": len(dd), **{f"{m}_pct_vs_mcsgarch": 100 * (dd[m].mean() / dd["mcsgarch"].mean() - 1)
                                                    for m in ["hybrid"] + NIFTY_VARIANTS[1:]},
                                "mcsgarch_qlike_mean": dd["mcsgarch"].mean(),
                                "hybrid_reduction_sum": float((dd["mcsgarch"] - dd["hybrid"]).sum()),
                                "combination_reduction_sum": float((dd["mcsgarch"] - dd["combination"]).sum())})
        tot = [r for r in anchor_rows if r["target"] == t]
        for r in tot:   # each anchor's share of the total QLIKE reduction over mcsGARCH (all anchors, all days)
            r["hybrid_reduction_share"] = r["hybrid_reduction_sum"] / sum(z["hybrid_reduction_sum"] for z in tot)
            r["combination_reduction_share"] = (r["combination_reduction_sum"]
                                                / sum(z["combination_reduction_sum"] for z in tot))
        for k, name in enumerate(REGIMES):
            m_ = d["regime"].to_numpy() == k
            if m_.any():
                gains_and_shares(nifty, t, d[m_], ["hybrid", "combination"] + EXTENSIONS, subset=f"regime {name}")

    # ---------------- other indices (S5, S6; E4b singles)
    for index, fam in (("BANKNIFTY", "S5_banknifty"), ("FINNIFTY", "S6_finnifty")):
        for t in PRIMARY_TARGETS:
            if (index, t) not in D:
                continue
            d = D[(index, t)]
            gains_and_shares(index, t, d, ["hybrid"] + OTHER_VARIANTS[1:])
            pre = fam.split("_")[0]
            dm_row(f"{pre}_{t}_combination", f"The combination has lower mean QLIKE than mcsGARCH on"
                   f" {INDEXES[index]['label']} ({TARGET_LABEL[t]}, nothing retuned). Estimate: gain, points.",
                   "secondary", fam, d, index, t, "all days", "mcsgarch", "combination", gain(d, "combination"))
            dm_row(f"{pre}_{t}_hybrid_vs_combination", f"The hybrid has lower mean QLIKE than the combination on"
                   f" {INDEXES[index]['label']} ({TARGET_LABEL[t]}). Estimate: difference of gains, points.",
                   "secondary", fam, d, index, t, "all days", "combination", "hybrid",
                   gain(d, "hybrid") - gain(d, "combination"))
            for x in EXTENSIONS:
                dm_row(f"E6_{index.lower()}_{t}_{x}", f"{LABELS[x]} alone beats mcsGARCH on {INDEXES[index]['label']}"
                       f" ({TARGET_LABEL[t]}).", "exploratory", "E6_other_index_singles", d, index, t, "all days",
                       "mcsgarch", x, gain(d, x))

    # ---------------- adjust, verdict, write
    families = {"primary": "the two primary rows (combination vs mcsGARCH, 15 and 30 minutes)",
                "S1_single_extensions": "secondary S1", "S2_hybrid_vs_combination": "secondary S2",
                "S3_last_hour": "secondary S3", "S4_anchor": "secondary S4", "S5_banknifty": "secondary S5",
                "S6_finnifty": "secondary S6", "R1_excl_ewma": "robustness R1", "R2_pre_auction": "robustness R2",
                "R3_seed_averaged_hybrid": "robustness R3", "E1_leave_one_out": None, "E2_singles_by_anchor": None,
                "E3_expiry": None, "E4_last_hour": None, "E5_mcsinfo": None, "E6_other_index_singles": None}
    tests = Rw.frame(families)
    detail = pd.DataFrame(Rw.detail)
    prim = tests[tests["_family"] == "primary"].set_index("test_id")["p_adjusted"]
    ptargets = PRIMARY_TARGETS if not synth else targets
    if all(f"P1_{t}_combination" in prim.index for t in ptargets) and all(t in share_primary for t in ptargets):
        pv = primary_verdict({t: prim[f"P1_{t}_combination"] for t in ptargets}, share_primary) \
            if ptargets == PRIMARY_TARGETS else {"verdict": "synthetic dry run (targets differ)"}
        verdict_rows.append({"study": "P109", **pv})
    # which extension: S1 rows, single shares, leave-one-out
    ext_rows = []
    for x in EXTENSIONS:
        r = {"extension": x, "label": LABELS[x]}
        for t in ptargets:
            s1 = tests[tests["test_id"] == f"S1_{t}_{x}"]
            sh = [z for z in share_rows if z["index"] == nifty and z["target"] == t and z["model"] == x
                  and z["reference"] == "hybrid" and z["subset"] == "all days"]
            e1 = tests[tests["test_id"] == f"E1_{t}_comb_no_x{x[1]}"]
            r.update({f"{t}_gain_points": gain(D[(nifty, t)], x), f"{t}_share_of_hybrid_gain": sh[0]["share"] if sh else np.nan,
                      f"{t}_share_ci_low": sh[0]["ci_low"] if sh else np.nan,
                      f"{t}_share_ci_high": sh[0]["ci_high"] if sh else np.nan,
                      f"{t}_S1_p_holm": float(s1["p_adjusted"].iloc[0]) if len(s1) else np.nan,
                      f"{t}_loo_rise_points": float(e1["estimate"].iloc[0]) if len(e1) else np.nan,
                      f"{t}_loo_p_raw": float(e1["p_raw"].iloc[0]) if len(e1) else np.nan})
        r["beats_mcsgarch_both_horizons"] = bool(all(r.get(f"{t}_S1_p_holm", 1) < ALPHA for t in ptargets))
        ext_rows.append(r)
    tests_out = tests.drop(columns="_family")
    tests_out.to_csv(out / "tests.csv", index=False)
    detail.to_csv(out / "tests_detail.csv", index=False)
    pd.DataFrame(verdict_rows).to_csv(out / "verdict.csv", index=False)
    pd.DataFrame(ext_rows).to_csv(out / "extensions.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(out / "gains.csv", index=False)
    pd.DataFrame(share_rows).to_csv(out / "shares.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(out / "gain_by_anchor.csv", index=False)
    pd.DataFrame(repro_rows).to_csv(out / "reproduction_check.csv", index=False)
    for (index, t), d in D.items():
        d.to_csv(out / f"daily_losses_{index.lower()}_{t}.csv")
    # parameters of every refit, and their summary
    pfiles = sorted(work.glob("*_block[0-9][0-9][0-9][0-9]_params.csv"))
    if pfiles:
        P = pd.concat([pd.read_csv(f) for f in pfiles], ignore_index=True)
        P.to_csv(out / "params_by_refit.csv", index=False)
        mp_ = P[P["kind"] == "model"]
        num = [c for c in mp_.columns if c not in ("index", "start", "variant", "target", "kind", "daily", "daily_method")
               and pd.api.types.is_numeric_dtype(mp_[c]) and mp_[c].notna().any()]
        summ = mp_.groupby(["index", "variant"])[num].agg(["mean", "min", "max"])
        summ.columns = [f"{a}_{b}" for a, b in summ.columns]
        summ["refits"] = mp_.groupby(["index", "variant"]).size()
        summ.to_csv(out / "params_summary.csv")
        cp = P[(P["kind"] == "calibration")]
        if len(cp):
            ccols = [c for c in cp.columns if c.startswith("c_")]
            cs = cp.groupby(["index", "variant", "target"])[ccols].agg(["mean", "min", "max"])
            cs = cs.stack(level=0, future_stack=True).reset_index().rename(columns={"level_3": "minute"})
            cs["minute"] = cs["minute"].str[2:].astype(int)
            cs.dropna(subset=["mean"]).sort_values(["index", "variant", "target", "minute"]).to_csv(
                out / "calibration_factors.csv", index=False)
    timing = []
    for f in sorted(work.glob("*_timing.csv")):
        timing.append(pd.read_csv(f))
    if timing:
        pd.concat(timing, ignore_index=True).to_csv(out / "timing.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"\n[{label}] verdict:")
    print(pd.DataFrame(verdict_rows).T.to_string())
    print(f"\n[{label}] tests (primary and secondary):")
    print(tests_out[tests_out["role"].isin(["primary", "secondary", "robustness"])][
        ["test_id", "estimate", "statistic", "p_raw", "p_adjusted"]].to_string(index=False))
    print(f"\n[{label}] shares of the hybrid's gain (all days):")
    sr = pd.DataFrame(share_rows)
    print(sr[(sr["subset"] == "all days")].to_string(index=False))
    print(f"\n[{label}] extensions:")
    print(pd.DataFrame(ext_rows).T.to_string())
    print(f"\n[{label}] reproduction checks:")
    print(pd.DataFrame(repro_rows).to_string(index=False))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    import arch
    import scipy
    inputs = {}
    for ix in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
        p = work / f"inputs_{ix}.json"
        if p.exists():
            inputs[ix] = json.loads(p.read_text())
    wall = {}
    for ix in ("NIFTY", "BANKNIFTY", "FINNIFTY", "SYNTH"):
        files = list(work.glob(f"{ix}_block[0-9][0-9][0-9][0-9].csv"))
        tim = work / f"{ix}_timing.csv"
        if files and tim.exists():
            tt = pd.read_csv(tim)
            wall[ix] = {"blocks": len(files), "seconds_jobs_total": float(tt["seconds"].sum()),
                        "first_block_written_utc": pd.Timestamp(min(f.stat().st_mtime for f in files), unit="s").strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "last_block_written_utc": pd.Timestamp(max(f.stat().st_mtime for f in files), unit="s").strftime("%Y-%m-%dT%H:%M:%SZ")}
    json.dump({"source": label + ("" if synth else ": NSE 1-minute index bars as in the committed runs; the hybrid's "
                                  "forecasts are the committed ones"),
               "study": "P109 mcsGARCH extensions (docs/prereg/P109_MCSGARCH_EXTENSIONS.md)",
               "plan_commit": a.plan_commit, "analysis_commit": commit,
               "commands": ["PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index NIFTY --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index BANKNIFTY --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index FINNIFTY --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py analyze --plan-commit <plan commit>"],
               "specs": SPECS, "min_train": MIN_TRAIN, "block": BLOCK, "expiry_diurnal": EXPIRY_DIURNAL,
               "diurnal_smooth": DIURNAL_SMOOTH, "synthetic_settings": SYNTH if synth else None,
               "bootstrap": {"reps": reps, "mean_block_days": MEAN_BLOCK, "seed": BOOT_SEED}, "alpha": ALPHA,
               "most_threshold": MOST, "input_sha256": inputs, "stage1": wall,
               "seconds_stage2": round(time.time() - t0, 1), "python": sys.version.split()[0],
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "arch": arch.__version__,
                            "scipy": scipy.__version__}},
              open(out / "run_info.json", "w"), indent=2, default=str)
    print(f"\nwritten to {out} ({time.time() - t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("run", "check", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--work", default=None)
        p.add_argument("--synthetic", action="store_true", help="simulated market and small settings (dry run)")
        if name == "run":
            p.add_argument("--index", choices=list(INDEXES), default="NIFTY")
            p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
            p.add_argument("--variants", nargs="+", default=None)
            p.add_argument("--blocks", nargs="+", type=int, default=None)
            p.add_argument("--workers", type=int, default=2)
        if name == "check":
            p.add_argument("--indexes", nargs="+", default=["NIFTY"])
        if name == "analyze":
            p.add_argument("--out", default=None)
            p.add_argument("--reps", type=int, default=REPS)
            p.add_argument("--plan-commit", default=None)
            p.add_argument("--allow-mismatch", action="store_true",
                           help="report even if the rebuilt mcsGARCH differs from the committed one")
    a = ap.parse_args(argv)
    if a.work is None:
        a.work = str(ROOT / "data" / "processed" / ("p109_synthetic" if a.synthetic else "p109"))
    if a.stage == "analyze" and a.out is None:
        a.out = str(ROOT / "data" / "processed" / "p109_synthetic_results") if a.synthetic else \
            str(ROOT / "results" / "p109_mcsgarch_extensions")
    if a.stage == "run":
        stage_run(a)
    elif a.stage == "check":
        stage_check(a)
    else:
        stage_analyze(a)


if __name__ == "__main__":
    main()
