"""P106: where does the forecasting hybrid's gain come from? (registered plan: docs/prereg/P106_GAIN_DECOMPOSITION.md)

The hybrid (``hybrid_base``: mcsGARCH times exp(a LightGBM correction)) beats mcsGARCH by 5.8% and 7.6%
(QLIKE, next 15 and 30 minutes) on Nifty, and an information-matched mcsGARCH (``mcsgarch_iv``) by 4.8% and
5.9%, the last edge only in the periods since SEBI's November 2024 cut in weekly options. This script

1. refits the hybrid without each of five feature groups (leave-one-group-out), plus three combined or
   split variants, with the pipeline's own walk-forward (``run_walkforward``, same seeds, refits, tuning and
   test days), and tests whether removing each group raises QLIKE (one-sided Diebold-Mariano); and
2. tests whether each group's contribution differs between the multi-weekly period (test days to 19
   November 2024) and the period since 20 November 2024 (two-sided, HAC delta method on period ratios).

Stages:

    # 1. forecasts (REAL DATA), one invocation per index; about 1.5 hours in all on 2 cores
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index NIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index BANKNIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index FINNIFTY --workers 2
    # 2. losses, tests, tables, verdicts
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py analyze --plan-commit <hash>

``check`` compares the full hybrid (seed 7) in the work files with the committed forecasts and computes no
loss. ``--synthetic`` (run and analyze) uses a simulated market and small settings, for a dry run of every
step; its output is labelled SYNTHETIC. Per-variant forecasts go to --work (default data/processed/p106,
git-ignored); a finished job is not rerun. ``analyze`` writes results/p106_gain_decomposition/.
Tests: tests/test_p106_gain_decomposition.py (synthetic data only).
"""

from __future__ import annotations

import argparse
import hashlib
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
from scipy.stats import spearmanr  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from niftygex.evaluation import REGIMES, daily_losses, diebold_mariano, holm, subgroups  # noqa: E402
from niftygex.features import (BASE_FEATURES, CLOSING_AUCTION_START, GEX_SETS, IMPLIED_FEATURES,  # noqa: E402
                               build_dataset, build_gex_panel)
from niftygex.forecast import WalkForwardConfig, run_walkforward  # noqa: E402
from niftygex.implied import build_implied_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The pipeline's market loader and targets, the stationary bootstrap of the referee response (M4.2) and its
# Newey-West helper (priority 5), so that nothing here is a second implementation.
rp = _load("run_pipeline", ROOT / "scripts" / "run_pipeline.py")
rf = _load("referee_forecast", ROOT / "experiments" / "referee_forecast.py")
gw = _load("referee_forecast_gw", ROOT / "experiments" / "referee_forecast_gw.py")

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

# The four columns that run_walkforward adds to the version 2 base set for every "base" model.
EXTRA_V2 = ["log_mcs", "log_s_window", "log_mcs_iv", "log_implied_window"]
HYBRID_COLUMNS = list(BASE_FEATURES) + EXTRA_V2          # the 41 columns of hybrid_base, in order

# Five groups; every column of the hybrid is in exactly one (checked by a test).
GROUPS = {
    # everything computed from the option chain published at the previous close
    "implied": list(IMPLIED_FEATURES) + ["log_mcs_iv", "log_implied_window"],
    # India VIX at the previous close
    "vix": ["log_vix"],
    # time of day: where the window sits in the session, and mcsGARCH's diurnal share of the window
    "intraday": ["minute_frac", "mins_left", "exp_x_minute", "log_s_window"],
    # expiry calendar, expiry regime and other dated rules known in advance
    "calendar": ["is_expiry", "tdte", "regime", "weekday", "is_monthly_expiry", "day_before_expiry",
                 "bn_expiry_only", "closing_auction"],
    # everything computed from past one-minute index returns, including mcsGARCH's own forecast
    "realized": ["log_rv_back_h", "log_rv_back_15", "log_rv_back_60", "log_rv_today_pm", "log_rv_d1", "log_rv_w",
                 "log_rv_m", "log_rv_prevday_same", "jump_share_d1", "log_vol15_rel", "ofi15", "log_mcs"],
}
PRIMARY_GROUPS = list(GROUPS)
GROUP_LABELS = {"implied": "option-implied (previous evening's chain)", "vix": "India VIX",
                "intraday": "time of day", "calendar": "calendar and regime",
                "realized": "realized-variance history (incl. mcsGARCH's forecast)",
                "optionmkt": "option-implied and VIX together",
                "mcsinfo": "everything mcsGARCH does not see (implied, VIX, calendar except the expiry flag)",
                "implied_level": "implied variance level part of the implied group",
                "implied_shape": "implied shape and positioning part of the implied group"}
IMPLIED_LEVEL = ["log_impvar_day", "log_impvar_to_exp", "log_iv_m21", "log_fwdvar_day", "vrp_w", "abs_ret_today_sd",
                 "log_mcs_iv", "log_implied_window"]
IMPLIED_SHAPE = ["exp_premium", "term_slope", "skew_near", "smile_near", "skew_m21", "log_pcr_oi_near",
                 "log_pcr_vol_near", "log_pcr_oi_all"]
DROP_SETS = {**GROUPS,
             "optionmkt": GROUPS["implied"] + GROUPS["vix"],
             "mcsinfo": GROUPS["implied"] + GROUPS["vix"] + [c for c in GROUPS["calendar"] if c != "is_expiry"],
             "implied_level": IMPLIED_LEVEL, "implied_shape": IMPLIED_SHAPE}
VARIANTS = ["full"] + list(DROP_SETS)                    # 10 variants
REPEAT_VARIANTS = ["full"] + PRIMARY_GROUPS + ["optionmkt", "mcsinfo"]     # 8, with the repeat seeds
OTHER_INDEX_VARIANTS = ["full"] + PRIMARY_GROUPS                           # 6, Bank Nifty and Fin Nifty

TARGETS = list(rp.TARGETS)                               # next15, next30, last_hour
PRIMARY_TARGETS = ["next15", "next30"]
HEADLINE_SEED = 7
REPEAT_SEEDS = [101, 102, 103, 104]                      # P99's first four new ensembles
ALL_SEEDS = [HEADLINE_SEED] + REPEAT_SEEDS
N_SEEDS = 5
CHUNK = 5                                                # variants per walk-forward job (they share mcsGARCH)
ALPHA = 0.05
REPS = 10000
MEAN_BLOCK = 10.0
BOOT_SEED = 106
PERIOD_SPLIT = "2024-11-20"                             # SEBI's cut: first day of the single-weekly regime

INDEXES = {
    "NIFTY": {"run": ROOT / "results" / "pipeline_nse_v2", "end": "2026-11-02", "label": "Nifty"},
    "BANKNIFTY": {"run": ROOT / "results" / "pipeline_nse_banknifty", "end": "2026-04-22", "label": "Bank Nifty"},
    "FINNIFTY": {"run": ROOT / "results" / "pipeline_nse_finnifty", "end": "2026-01-27", "label": "Fin Nifty"},
}
P99_DIR = ROOT / "results" / "seed_bootstrap"
# Settings shared by all three committed runs (checked against each run_info.json).
PIPELINE_SETTINGS = dict(source="nse", start="2023-01-01", min_coverage=0.95, rate=0.065, oi_unit="keep_units",
                         lot_size=None, tune=False, tune_every=3, n_seeds=N_SEEDS, seed=HEADLINE_SEED, min_train=150,
                         block=21, mcs_refit_every=21, holdout_start="2025-07-01")
SYNTH_SETTINGS = dict(n_days=200, start="2024-06-03", seed=3, min_train=60, block=20, n_seeds=2,
                      targets=["next15", "next30"], seeds=[7, 101], horizon_step=45)
ID_COLS = ["t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y"]


def model_name(variant: str) -> str:
    return "hybrid_base" if variant == "full" else f"hybrid_drop_{variant}"


def hybrid_columns(variant: str) -> list[str]:
    """The columns the variant's LightGBM correction sees (forecast.model_columns, in that order)."""
    drop = set(DROP_SETS.get(variant, []))
    return [c for c in HYBRID_COLUMNS if c not in drop]


# ---------------------------------------------------------------------------
# Configuration and job plan
# ---------------------------------------------------------------------------


def make_config(seed: int, eval_minutes, gex_sets: dict, variants, **over) -> WalkForwardConfig:
    """The committed runs' WalkForwardConfig (scripts/run_pipeline.py main) with another base seed, the drop
    sets of this study and only mcsGARCH, the implied-level mcsGARCH and the hybrids of ``variants``."""
    kw = dict(models=["mcsgarch", "mcsgarch_iv"] + [model_name(v) for v in variants], gex_sets=gex_sets,
              tune=PIPELINE_SETTINGS["tune"], tune_every=PIPELINE_SETTINGS["tune_every"],
              n_seeds=PIPELINE_SETTINGS["n_seeds"], seed=seed, min_train=PIPELINE_SETTINGS["min_train"],
              block=PIPELINE_SETTINGS["block"], mcs_refit_every=PIPELINE_SETTINGS["mcs_refit_every"],
              drop_sets={k: list(v) for k, v in DROP_SETS.items()})
    kw.update(over)
    cfg = WalkForwardConfig(**kw)
    cfg.eval_minutes = eval_minutes
    return cfg


def chunks(seq, size: int):
    seq = list(seq)
    k = int(np.ceil(len(seq) / size))
    # near-equal chunks, in order
    return [tuple(x) for x in np.array_split(np.array(seq, dtype=object), k)] if seq else []


def job_plan(index: str, targets=None, seeds=None, variants=None, chunk: int = CHUNK) -> list[tuple]:
    """(index, target, seed, variants) in the registered priority order. Nifty: every variant at every target
    with the headline seed, then the repeat seeds at 15 and 30 minutes; other indices: the full hybrid and the
    five groups at 15 and 30 minutes with the headline seed. Optional filters keep the order."""
    plan = []
    if index == "NIFTY" or index == "SYNTH":
        tlist = TARGETS if index == "NIFTY" else SYNTH_SETTINGS["targets"]
        for t in tlist:
            plan += [(index, t, HEADLINE_SEED, c) for c in chunks(VARIANTS, chunk)]
        rseeds = REPEAT_SEEDS if index == "NIFTY" else SYNTH_SETTINGS["seeds"][1:]
        for s in rseeds:
            for t in (PRIMARY_TARGETS if index == "NIFTY" else SYNTH_SETTINGS["targets"]):
                plan += [(index, t, s, c) for c in chunks(REPEAT_VARIANTS, chunk)]
    else:
        for t in PRIMARY_TARGETS:
            plan += [(index, t, HEADLINE_SEED, c) for c in chunks(OTHER_INDEX_VARIANTS, chunk)]
    out = []
    for (ix, t, s, vs) in plan:
        if targets is not None and t not in targets:
            continue
        if seeds is not None and s not in seeds:
            continue
        if variants is not None:
            vs = tuple(v for v in vs if v in variants)
        if vs:
            out.append((ix, t, s, vs))
    return out


def check_run_settings(index: str, run_info: dict, raw_end: str | None) -> list[str]:
    """Differences between PIPELINE_SETTINGS (plus symbol and end) and a committed run's arguments."""
    a = run_info["args"]
    bad = [f"{k}: here {v!r}, committed {a.get(k)!r}" for k, v in PIPELINE_SETTINGS.items() if a.get(k) != v]
    if a.get("symbol", "NIFTY") != index:
        bad.append(f"symbol: committed {a.get('symbol', 'NIFTY')}")
    if a.get("end") not in (None, INDEXES[index]["end"]):
        bad.append(f"end: committed {a.get('end')}")
    if run_info["days"][1] != raw_end:
        bad.append(f"last day: committed {run_info['days'][1]}, here {raw_end}")
    return bad


# ---------------------------------------------------------------------------
# Stage 1: forecasts
# ---------------------------------------------------------------------------

_STATE: dict = {}   # filled in the parent before the workers are forked


def work_file(work: Path, index: str, target: str, seed: int, variant: str) -> Path:
    return Path(work) / f"{index}_{target}_s{seed}_{variant}.csv"


def job_done(work: Path, job: tuple) -> bool:
    ix, t, s, vs = job
    return all(work_file(work, ix, t, s, v).exists() for v in vs)


def run_job(job):
    (index, target, seed, variants), work = job[:4], job[4]
    m, ds, sets = _STATE["market"], _STATE["datasets"][target], _STATE["sets"]
    t0 = time.time()
    cfg = _STATE.get("config_fn", make_config)(seed, _STATE["eval_minutes"].get(target), sets, variants)
    pred, _ = run_walkforward(m, ds, cfg)
    for v in variants:
        path = work_file(Path(work), index, target, seed, v)
        tmp = path.with_suffix(".tmp")
        out = pred[ID_COLS + ["mcsgarch", "mcsgarch_iv", model_name(v)]].rename(columns={model_name(v): "forecast"})
        out.to_csv(tmp, index=False)
        os.replace(tmp, path)
    secs = round(time.time() - t0, 1)
    with open(Path(work) / "timing.csv", "a") as fh:
        fh.write(f"{index},{target},{seed},{'+'.join(variants)},{len(variants)},{secs},{len(pred)},{os.getpid()}\n")
    return index, target, seed, variants, secs, len(pred)


def sha256_file(path: Path) -> str:
    """SHA-256 as in data/MANIFEST.csv (scripts/data_manifest.py: of the decompressed content for .gz files,
    whose headers carry a time stamp)."""
    return _load("data_manifest", ROOT / "scripts" / "data_manifest.py").sha256(Path(path))


def input_hashes(raw: Path, index: str) -> dict:
    """SHA-256 of every raw input of one index (directories: one digest over the sorted per-file digests),
    and how many files agree with data/MANIFEST.csv."""
    spec = rp.SYMBOLS[index]
    files = [spec["bars"], spec.get("official_file", "index_daily.csv"), "india_vix.csv", spec["futures_bars"]]
    dirs = [spec.get("bhavcopy", "bhavcopy"), "participant_oi"]
    man_path = ROOT / "data" / "MANIFEST.csv"
    man = pd.read_csv(man_path).set_index("path")["sha256"].to_dict() if man_path.exists() else {}
    out, agree, differ, absent = {}, 0, 0, 0
    for f in files:
        p = raw / f
        if p.exists():
            d = sha256_file(p)
            out[f] = d
            if f in man:
                agree += man[f] == d
                differ += man[f] != d
            else:
                absent += 1
    for dname in dirs:
        p = raw / dname
        if not p.exists():
            continue
        lines = []
        for q in sorted(x for x in p.rglob("*") if x.is_file()):
            rel = str(q.relative_to(raw))
            d = sha256_file(q)
            lines.append(f"{rel}:{d}")
            if rel in man:
                agree += man[rel] == d
                differ += man[rel] != d
            else:
                absent += 1
        out[dname + "/ (" + str(len(lines)) + " files)"] = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    out["manifest_check"] = {"files_matching_manifest": int(agree), "files_differing": int(differ),
                             "files_not_in_manifest": int(absent)}
    return out


def synthetic_market():
    from niftygex.simulate import SimConfig, simulate_market
    return simulate_market(SimConfig(n_days=SYNTH_SETTINGS["n_days"], start=SYNTH_SETTINGS["start"],
                                     seed=SYNTH_SETTINGS["seed"], kappa=0.0))


def synthetic_config(seed, eval_minutes, sets, variants):
    return make_config(seed, eval_minutes, sets, variants, min_train=SYNTH_SETTINGS["min_train"],
                       block=SYNTH_SETTINGS["block"], mcs_refit_every=SYNTH_SETTINGS["block"],
                       n_seeds=SYNTH_SETTINGS["n_seeds"])


def stage_run(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    index = "SYNTH" if a.synthetic else a.index
    plan = job_plan(index, targets=a.targets, seeds=a.seeds, variants=a.variants, chunk=a.chunk)
    jobs = [j for j in plan if not job_done(work, j)]
    label = "SYNTHETIC (simulated market, not real data)" if a.synthetic else "REAL DATA"
    print(f"[{label}] P106 gain decomposition, stage 1, {index}: {len(jobs)} jobs to run "
          f"({len(plan) - len(jobs)} already in {work}); workers {a.workers}", flush=True)
    if not jobs:
        return
    t0 = time.time()
    if a.synthetic:
        m = synthetic_market()
        _STATE["config_fn"] = synthetic_config
        targets = {"next15": dict(horizon=15, step=SYNTH_SETTINGS["horizon_step"]),
                   "next30": dict(horizon=30, step=SYNTH_SETTINGS["horizon_step"])}
    else:
        info = json.load(open(INDEXES[index]["run"] / "run_info.json"))
        ns = argparse.Namespace(source="nse", raw=a.raw, symbol=index, oi_unit=PIPELINE_SETTINGS["oi_unit"],
                                lot_size=None, rate=PIPELINE_SETTINGS["rate"], start=PIPELINE_SETTINGS["start"],
                                end=INDEXES[index]["end"], min_coverage=PIPELINE_SETTINGS["min_coverage"])
        m = rp.load_market(ns)
        last = str(pd.Timestamp(m["days"][-1]).date())
        bad = check_run_settings(index, info, last)
        if bad or len(m["days"]) != info["days"][2]:
            raise SystemExit(f"settings differ from the committed run: {bad}, days {len(m['days'])} vs {info['days'][2]}")
        targets = rp.TARGETS
        hp = work / f"inputs_{index}.json"
        if not hp.exists():
            hp.write_text(json.dumps(input_hashes(Path(a.raw), index), indent=2))
    print(f"market loaded: {len(m['days'])} days, {m['days'][0]} to {m['days'][-1]} ({time.time() - t0:.0f}s)", flush=True)
    panel = build_gex_panel(m)
    implied = build_implied_panel(m)
    has_oracle = "dealer_next" in m
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle" or has_oracle}
    need = sorted({j[1] for j in jobs})
    datasets = {t: build_dataset(m, panel, implied=implied, **targets[t]) for t in need}
    print(f"panels and datasets built ({time.time() - t0:.0f}s)", flush=True)
    _STATE.update(market=m, datasets=datasets, sets=sets,
                  eval_minutes={} if a.synthetic else dict(rp.EVAL_MINUTES))
    if not (work / "timing.csv").exists():
        (work / "timing.csv").write_text("index,target,seed,variants,n_variants,seconds,rows,pid\n")
    done = 0
    args = [j + (str(work),) for j in jobs]

    def report(res):
        ix, target, seed, vs, secs, n = res
        print(f"  [{done}/{len(jobs)}] {ix} {target} seed {seed} {'+'.join(vs)}: {n} forecasts, {secs:.0f}s "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)

    if a.workers == 1:      # in this process (same results; used by the tests)
        for j in args:
            done += 1
            report(run_job(j))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=a.workers, maxtasksperchild=1) as pool:
            for res in pool.imap(run_job, args, chunksize=1):
                done += 1
                report(res)
    print(f"stage 1 ({index}) done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Reading the work files; reproduction check (no losses)
# ---------------------------------------------------------------------------


def read_csv_exact(path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def load_wide(work: Path, index: str, target: str, seed: int, variants) -> pd.DataFrame:
    """Identifiers, y, mcsgarch, mcsgarch_iv and one column per variant (named by the variant). Raises if the
    identifiers, y or either benchmark differ between the variants' files."""
    wide = None
    for v in variants:
        f = read_csv_exact(work_file(work, index, target, seed, v)).sort_values(["t", "minute"]).reset_index(drop=True)
        if wide is None:
            wide = f[ID_COLS + ["mcsgarch", "mcsgarch_iv"]].copy()
        else:
            for c in ID_COLS + ["mcsgarch", "mcsgarch_iv"]:
                if not f[c].equals(wide[c]):
                    raise ValueError(f"{index} {target} seed {seed}: column {c} differs between variants")
        wide[v] = f["forecast"].to_numpy()
    return wide


def reproduction_check(wide: pd.DataFrame, committed: pd.DataFrame, label: str) -> dict:
    """The full hybrid (seed 7) and both benchmarks against the committed forecasts (forecasts only)."""
    a = wide.set_index(["date", "minute"])
    b = committed.set_index(["date", "minute"])
    common = a.index.intersection(b.index)
    out = {"check": label, "rows_rerun": len(a), "rows_committed": len(b), "rows_matched": len(common)}
    for name, (ca, cb) in {"y": ("y", "y"), "mcsgarch": ("mcsgarch", "mcsgarch"),
                           "mcsgarch_iv": ("mcsgarch_iv", "mcsgarch_iv"), "hybrid_base": ("full", "hybrid_base")}.items():
        if ca not in a or cb not in b:
            continue
        x, y = a.loc[common, ca].to_numpy(float), b.loc[common, cb].to_numpy(float)
        out[f"{name}_n_identical"] = int(np.sum(x == y))
        out[f"{name}_max_abs_rel_diff"] = float(np.max(np.abs(x / y - 1)))
    return out


def p99_daily_check(d: pd.DataFrame, p99: pd.DataFrame, seed: int, label: str) -> dict:
    """The full hybrid's daily QLIKE with a repeat seed set against P99's committed daily losses.
    Both are aligned on the date written as YYYY-MM-DD text, whatever type each index has (fix after the
    plan commit: the registered version aligned a text index with a datetime index and so matched no day)."""
    def text_index(x):
        return pd.Index(pd.to_datetime(x.index).strftime("%Y-%m-%d"))
    a = d.set_axis(text_index(d))
    b = p99.set_axis(text_index(p99))
    x = a["full"].reindex(b.index).to_numpy(float)
    y = b[f"qlike_hybrid_s{seed}"].to_numpy(float)
    m = a["mcsgarch"].reindex(b.index).to_numpy(float)
    my = b["qlike_mcsgarch"].to_numpy(float)
    ok = np.isfinite(x)
    return {"check": label, "rows_rerun": len(a), "rows_committed": len(b), "rows_matched": int(ok.sum()),
            "hybrid_base_n_identical": int(np.sum(x[ok] == y[ok])),
            "hybrid_base_daily_qlike_max_abs_rel_diff": float(np.max(np.abs(x[ok] / y[ok] - 1))) if ok.any() else np.nan,
            "mcsgarch_n_identical": int(np.sum(m[ok] == my[ok])),
            "mcsgarch_daily_qlike_max_abs_rel_diff": float(np.max(np.abs(m[ok] / my[ok] - 1))) if ok.any() else np.nan}


def stage_check(a):
    work = Path(a.work)
    rows = []
    for index in a.indexes:
        for t in (a.targets or TARGETS):
            if not work_file(work, index, t, HEADLINE_SEED, "full").exists():
                continue
            wide = load_wide(work, index, t, HEADLINE_SEED, ["full"])
            committed = read_csv_exact(INDEXES[index]["run"] / f"forecasts_{t}.csv")
            r = reproduction_check(wide, committed, f"{index} {t} seed {HEADLINE_SEED}")
            rows.append(r)
            print(r, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def day_frame(wide: pd.DataFrame, variants) -> pd.DataFrame:
    """Daily average QLIKE of mcsgarch, mcsgarch_iv and every variant, with the day's calendar labels.
    Every day must have the same number of anchors, so the mean of daily means is the anchor mean."""
    counts = wide.groupby("date").size()
    if counts.nunique() != 1:
        raise ValueError("days have different numbers of anchors")
    L = daily_losses(wide, ["mcsgarch", "mcsgarch_iv"] + list(variants), "qlike")
    cal = wide.groupby("date").agg(t=("t", "first"), is_expiry=("is_expiry", "first"), regime=("regime", "first"))
    return L.join(cal)


def points(num_mean: float, den_mean: float) -> float:
    return 100.0 * num_mean / den_mean


def ratio_var(num: np.ndarray, den: np.ndarray, lag: int | None = None) -> tuple[float, float]:
    """r = 100 mean(num) / mean(den) and the variance of r by the delta method with a Newey-West long-run
    variance (Bartlett kernel, the DM test's bandwidth rule) of the influence series."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    md = den.mean()
    r = 100.0 * num.mean() / md
    psi = 100.0 * (num - (r / 100.0) * den) / md
    lrv = float(gw.newey_west(psi, lag)[0, 0])
    return r, lrv / n


def period_diff_test(num: np.ndarray, den: np.ndarray, p2: np.ndarray) -> dict:
    """Two-sided test of H0: the ratio 100 E[num] / E[den] is the same in period 1 (~p2) and period 2 (p2).
    Delta method on each period's ratio, Newey-West variance within each period, periods independent;
    Student t with n - 2 degrees of freedom."""
    p2 = np.asarray(p2, bool)
    r1, v1 = ratio_var(num[~p2], den[~p2])
    r2, v2 = ratio_var(num[p2], den[p2])
    d = r2 - r1
    se = float(np.sqrt(v1 + v2))
    n = len(num)
    stat = d / se if se > 0 else np.nan
    p = float(2 * student_t.sf(abs(stat), n - 2)) if np.isfinite(stat) else np.nan
    return {"r1": r1, "r2": r2, "diff": d, "se": se, "stat": stat, "p": p, "n1": int((~p2).sum()), "n2": int(p2.sum())}


def period_diff_bootstrap(num: np.ndarray, den: np.ndarray, p2: np.ndarray, reps: int = REPS,
                          mean_block: float = MEAN_BLOCK, rng: np.random.Generator | None = None) -> dict:
    """Stationary bootstrap of days, separately within each period; two-sided p for the difference of
    period ratios: (1 + #{|D* - D| >= |D|}) / (reps + 1); also the 2.5% and 97.5% points of D*."""
    rng = rng if rng is not None else np.random.default_rng(BOOT_SEED)
    p2 = np.asarray(p2, bool)
    res = []
    for mask in (~p2, p2):
        x = np.column_stack([num[mask], den[mask]])
        idx = rf.stationary_bootstrap_indices(len(x), reps, mean_block, rng)
        mb = rf.bootstrap_means(x, idx)
        res.append(100.0 * mb[:, 0] / mb[:, 1])
    dstar = res[1] - res[0]
    d = period_diff_test(num, den, p2)["diff"]
    return {"diff": d, "p": float((1 + np.sum(np.abs(dstar - d) >= abs(d))) / (reps + 1)),
            "q025": float(np.quantile(dstar, 0.025)), "q975": float(np.quantile(dstar, 0.975))}


def share_ci(l_mcs: np.ndarray, l_full: np.ndarray, l_var: np.ndarray, reps: int = REPS,
             mean_block: float = MEAN_BLOCK, rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Share of the hybrid's gain over mcsGARCH carried by a group, (mean l_var - mean l_full) /
    (mean l_mcs - mean l_full), with a 95% stationary-bootstrap percentile interval."""
    rng = rng if rng is not None else np.random.default_rng(BOOT_SEED)
    x = np.column_stack([l_mcs, l_full, l_var])
    share = (x[:, 2].mean() - x[:, 1].mean()) / (x[:, 0].mean() - x[:, 1].mean())
    idx = rf.stationary_bootstrap_indices(len(x), reps, mean_block, rng)
    mb = rf.bootstrap_means(x, idx)
    s = (mb[:, 2] - mb[:, 1]) / (mb[:, 0] - mb[:, 1])
    return float(share), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def contribution(D: pd.DataFrame, variant: str, mask=None) -> dict:
    """Group contribution in points of mcsGARCH's mean daily QLIKE over the days in ``mask`` (positive: the
    group lowers the hybrid's loss) and the one-sided DM test that removing it raises the loss."""
    d = D if mask is None else D[np.asarray(mask, bool)]
    c = points(d[variant].mean() - d["full"].mean(), d["mcsgarch"].mean())
    dm = diebold_mariano(d[variant], d["full"], alternative="greater")
    return {"estimate": c, "stat": dm["stat"], "p": dm["p"], "days": dm["n"],
            "margin_full": points(d["full"].mean() - d["mcsgarch"].mean(), d["mcsgarch"].mean()),
            "margin_variant": points(d[variant].mean() - d["mcsgarch"].mean(), d["mcsgarch"].mean())}


def period_mask(D: pd.DataFrame, split: str = PERIOD_SPLIT) -> np.ndarray:
    """True on days of period 2 (from the cut on). Uses the date; the analysis checks that it equals the
    pipeline's regime label (regime code 1 or 2)."""
    return pd.to_datetime(D.index).to_numpy() >= np.datetime64(pd.Timestamp(split))


def seed_label(n_agree: int, k: int, p_adj: float, alpha: float = ALPHA) -> str:
    """The registered verdict for one primary row."""
    if not np.isfinite(p_adj):
        return "not computed"
    if p_adj >= alpha:
        return "not supported"
    return "supported" if n_agree == k else "seed-fragile"


# ---------------------------------------------------------------------------
# Stage 2: analysis
# ---------------------------------------------------------------------------

DM_METHOD = "Diebold-Mariano on daily mean QLIKE, Newey-West, HLN factor, Student t(n-1)"
DELTA_METHOD = "difference of period ratios, delta method with Newey-West variance within each period, Student t(n-2)"
BOOT_METHOD = "stationary bootstrap of days within each period, mean block 10, 10,000 draws"


class Rows:
    """Collects tests.csv rows (the 11 registered columns) and a detail table."""

    def __init__(self):
        self.rows, self.detail = [], []

    def add(self, test_id, hypothesis, role, sample, estimate, statistic, p_raw, sidedness, p_method, family,
            **detail):
        self.rows.append({"test_id": test_id, "hypothesis": hypothesis, "role": role, "sample": sample,
                          "estimate": estimate, "statistic": statistic, "p_raw": p_raw, "sidedness": sidedness,
                          "p_method": p_method, "p_adjusted": np.nan, "adjust_method": "", "_family": family})
        self.detail.append({"test_id": test_id, "family": family, **detail})

    def frame(self, families: dict) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        for fam, how in families.items():
            m = df["_family"] == fam
            if not m.any():
                continue
            if how is None:
                df.loc[m, "adjust_method"] = "none (no adjustment)"
                df.loc[m, "p_adjusted"] = df.loc[m, "p_raw"]
            else:
                df.loc[m, "p_adjusted"] = holm(df.loc[m, "p_raw"].to_numpy(float))
                df.loc[m, "adjust_method"] = f"Holm over {int(m.sum())} rows: {how}"
        return df


TARGET_LABEL = {"next15": "next 15 min", "next30": "next 30 min", "last_hour": "last hour"}


def sample_text(index: str, target: str, D: pd.DataFrame, what: str, seed: str) -> str:
    d0, d1 = pd.Timestamp(D.index.min()).date(), pd.Timestamp(D.index.max()).date()
    return f"{INDEXES.get(index, {'label': index})['label']}, {TARGET_LABEL[target]}, {what}, {len(D)} test days {d0} to {d1}, {seed}"


def stage_analyze(a):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    synth = a.synthetic
    label = "SYNTHETIC (simulated market, not real data)" if synth else "REAL DATA"
    nifty = "SYNTH" if synth else "NIFTY"
    targets = SYNTH_SETTINGS["targets"] if synth else TARGETS
    rseeds = SYNTH_SETTINGS["seeds"][1:] if synth else REPEAT_SEEDS
    seeds_all = [HEADLINE_SEED] + rseeds
    reps = a.reps
    print(f"[{label}] P106 gain decomposition, stage 2 (docs/prereg/P106_GAIN_DECOMPOSITION.md)", flush=True)
    R = Rows()
    contrib_rows, period_rows, seed_rows, repro_rows, regime_rows, share_rows = [], [], [], [], [], []

    # ---------------- Nifty: daily losses per target and seed
    D = {}      # (target, seed) -> day frame
    for t in targets:
        for s in seeds_all:
            vs = VARIANTS if s == HEADLINE_SEED else REPEAT_VARIANTS
            if not all(work_file(work, nifty, t, s, v).exists() for v in vs):
                continue
            wide = load_wide(work, nifty, t, s, vs)
            D[(t, s)] = day_frame(wide, vs)
            if s == HEADLINE_SEED and not synth:
                committed = read_csv_exact(INDEXES["NIFTY"]["run"] / f"forecasts_{t}.csv")
                repro_rows.append(reproduction_check(wide, committed, f"NIFTY {t} seed 7 vs results/pipeline_nse_v2"))
            if s != HEADLINE_SEED and not synth and (P99_DIR / f"daily_losses_{t}.csv").exists():
                p99 = pd.read_csv(P99_DIR / f"daily_losses_{t}.csv", index_col="date", parse_dates=["date"],
                                  float_precision="round_trip")
                repro_rows.append(p99_daily_check(D[(t, s)], p99, s,
                                                  f"NIFTY {t} seed {s} full hybrid daily QLIKE vs P99 daily_losses_{t}.csv"))
    if (targets[0], HEADLINE_SEED) not in D:
        raise SystemExit("no finished headline-seed work files for the primary index")
    # period labels: the date split must equal the pipeline's regime labels
    for (t, s), d in D.items():
        p2 = period_mask(d)
        if not np.array_equal(p2, d["regime"].to_numpy() != 0):
            raise ValueError(f"{t}: the period split by date differs from the pipeline's regime labels")
    # EWMA-fallback test days (Nifty only; referee response, side finding 1)
    ew_starts = [] if synth else gw.ewma_fallback_starts(INDEXES["NIFTY"]["run"])

    # ---------------- Part 1: leave-one-group-out, all days
    def part1_rows(d, t, index, seed_txt, role, family, variants, prefix, subset_name="all days", mask=None,
                   hyp_extra=""):
        for v in variants:
            c = contribution(d, v, mask)
            sub = d if mask is None else d[np.asarray(mask, bool)]
            R.add(f"{prefix}_{t}_{v}", f"Removing the {GROUP_LABELS[v]} group from the hybrid raises its mean QLIKE"
                  f" ({TARGET_LABEL[t]}, {subset_name}){hyp_extra}. Estimate: the rise, in points of mcsGARCH's QLIKE.",
                  role, sample_text(index, t, sub, subset_name, seed_txt), c["estimate"], c["stat"], c["p"],
                  "one-sided (removal raises loss)", DM_METHOD, family, index=index, target=t, variant=v,
                  subset=subset_name, seed=seed_txt, days=c["days"], margin_full_pct=c["margin_full"],
                  margin_variant_pct=c["margin_variant"])
            contrib_rows.append({"index": index, "target": t, "subset": subset_name, "seed": seed_txt, "variant": v,
                                 "days": c["days"], "contribution_points": c["estimate"], "dm_stat": c["stat"],
                                 "p_one_sided": c["p"], "hybrid_margin_vs_mcsgarch_pct": c["margin_full"],
                                 "variant_margin_vs_mcsgarch_pct": c["margin_variant"]})

    for t in targets:
        d = D[(t, HEADLINE_SEED)]
        if t in PRIMARY_TARGETS:
            part1_rows(d, t, nifty, "seed set 7 (headline)", "primary", "primary", PRIMARY_GROUPS, "P1")
            part1_rows(d, t, nifty, "seed set 7 (headline)", "secondary", "S3_option_and_mcsinfo",
                       ["optionmkt"], "S3")
            # the mcsGARCH-information hybrid against mcsGARCH, and the full hybrid against it
            dm = diebold_mariano(d["mcsgarch"], d["mcsinfo"], alternative="greater")
            R.add(f"S3_{t}_mcsinfo_vs_mcsgarch", f"A hybrid that sees only mcsGARCH's information has lower mean QLIKE"
                  f" than mcsGARCH ({TARGET_LABEL[t]}, all days). Estimate: its margin over mcsGARCH, % (negative = better).",
                  "secondary", sample_text(nifty, t, d, "all days", "seed set 7 (headline)"),
                  points(d["mcsinfo"].mean() - d["mcsgarch"].mean(), d["mcsgarch"].mean()), dm["stat"], dm["p"],
                  "one-sided (hybrid better)", DM_METHOD, "S3_option_and_mcsinfo", index=nifty, target=t,
                  variant="mcsinfo", subset="all days", days=dm["n"])
            part1_rows(d, t, nifty, "seed set 7 (headline)", "secondary", "S3_option_and_mcsinfo", ["mcsinfo"], "S3",
                       hyp_extra=" (the value of the information mcsGARCH lacks)")
            part1_rows(d, t, nifty, "seed set 7 (headline)", "exploratory", "E1_implied_split",
                       ["implied_level", "implied_shape"], "E1")
            # robustness: without the 09:15 anchor, before the auction (rebuilt from the anchor-level files)
            wide = load_wide(work, nifty, t, HEADLINE_SEED, VARIANTS)
            for sname, keep in (("without the 09:15 anchor", wide["minute"].to_numpy() != 0),
                                ("before the closing auction", pd.to_datetime(wide["date"]).to_numpy()
                                 < np.datetime64(pd.Timestamp(CLOSING_AUCTION_START)))):
                if keep.all() or not keep.any():
                    continue
                dd = day_frame(wide[keep], VARIANTS)
                fam = "R2_excl_0915" if sname.startswith("without") else "R3_pre_auction"
                part1_rows(dd, t, nifty, "seed set 7 (headline)", "robustness", fam, PRIMARY_GROUPS,
                           fam.split("_")[0], subset_name=sname)
            # exploratory: expiry days
            part1_rows(d, t, nifty, "seed set 7 (headline)", "exploratory", "E2_expiry", PRIMARY_GROUPS, "E2",
                       subset_name="expiry days", mask=d["is_expiry"].to_numpy(bool))
            # shares of the gain over mcsGARCH
            for v in VARIANTS[1:]:
                rng = np.random.default_rng([BOOT_SEED, targets.index(t), VARIANTS.index(v)])
                sh, lo, hi = share_ci(d["mcsgarch"].to_numpy(), d["full"].to_numpy(), d[v].to_numpy(), reps=reps, rng=rng)
                share_rows.append({"index": nifty, "target": t, "variant": v, "share_of_gain": sh,
                                   "share_ci_low": lo, "share_ci_high": hi})
        else:   # last hour: secondary (groups), exploratory (combined)
            part1_rows(d, t, nifty, "seed set 7 (headline)", "secondary", "S4_last_hour", PRIMARY_GROUPS, "S4")
            part1_rows(d, t, nifty, "seed set 7 (headline)", "exploratory", "E4_last_hour_combined",
                       ["optionmkt", "mcsinfo", "implied_level", "implied_shape"], "E4")
            for v in VARIANTS[1:]:
                rng = np.random.default_rng([BOOT_SEED, targets.index(t), VARIANTS.index(v)])
                sh, lo, hi = share_ci(d["mcsgarch"].to_numpy(), d["full"].to_numpy(), d[v].to_numpy(), reps=reps, rng=rng)
                share_rows.append({"index": nifty, "target": t, "variant": v, "share_of_gain": sh,
                                   "share_ci_low": lo, "share_ci_high": hi})

    # ---------------- seed repeats (part 1): the contribution in each seed set, and seed-pooled DM
    seed_info = {}
    for t in PRIMARY_TARGETS if not synth else SYNTH_SETTINGS["targets"]:
        have = [s for s in seeds_all if (t, s) in D]
        for v in REPEAT_VARIANTS[1:]:
            est = {s: points(D[(t, s)][v].mean() - D[(t, s)]["full"].mean(), D[(t, s)]["mcsgarch"].mean()) for s in have}
            vals = np.array(list(est.values()))
            seed_info[("P1", t, v)] = est
            seed_rows.append({"part": "1 (all days)", "target": t, "variant": v, "seeds": " ".join(map(str, have)),
                              "k": len(have), **{f"estimate_s{s}": e for s, e in est.items()},
                              "mean": vals.mean(), "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                              "min": vals.min(), "max": vals.max(), "n_positive": int((vals > 0).sum())})
        if len(have) > 1:
            pooled = {c: sum(D[(t, s)][c] for s in have) / len(have) for c in ["mcsgarch"] + REPEAT_VARIANTS}
            pdf = pd.DataFrame(pooled)
            part1_rows(pdf, t, nifty, f"daily QLIKE averaged over seed sets {', '.join(map(str, have))}",
                       "robustness", "R1_seed_pooled", PRIMARY_GROUPS, "R1")

    # ---------------- placebo: the full hybrid against itself with another seed set
    for t in PRIMARY_TARGETS if not synth else SYNTH_SETTINGS["targets"]:
        d7 = D[(t, HEADLINE_SEED)]
        for s in rseeds:
            if (t, s) not in D:
                continue
            ds_ = D[(t, s)]
            dm = diebold_mariano(ds_["full"], d7["full"], alternative="greater")
            R.add(f"PL1_{t}_s{s}", f"The full hybrid with seed set 7 has lower mean QLIKE than the same hybrid with seed"
                  f" set {s} ({TARGET_LABEL[t]}, all days; a placebo: only the seeds differ). Estimate: points of mcsGARCH's QLIKE.",
                  "placebo", sample_text(nifty, t, d7, "all days", f"seed sets 7 and {s}"),
                  points(ds_["full"].mean() - d7["full"].mean(), d7["mcsgarch"].mean()), dm["stat"], dm["p"],
                  "one-sided (seed set 7 better)", DM_METHOD, "PL1_seed_placebo", index=nifty, target=t, seed=s)
            p2 = period_mask(d7)
            pt = period_diff_test((ds_["full"] - d7["full"]).to_numpy(), d7["mcsgarch"].to_numpy(), p2)
            R.add(f"PL2_{t}_s{s}", f"The seed-only difference (seed set {s} minus seed set 7) is the same in the two periods"
                  f" ({TARGET_LABEL[t]}; a placebo for the period tests). Estimate: period 2 minus period 1, points.",
                  "placebo", sample_text(nifty, t, d7, "period 1 vs period 2", f"seed sets 7 and {s}"),
                  pt["diff"], pt["stat"], pt["p"], "two-sided", DELTA_METHOD, "PL2_seed_placebo_periods",
                  index=nifty, target=t, seed=s)

    # ---------------- Part 2: the two periods
    def period_block(d, t, seed_txt, role_groups, fam_groups, prefix, mask_days=None, subset="all days",
                     with_edge=True, boot=False, rng_key=0, record=True):
        dd = d if mask_days is None else d[np.asarray(mask_days, bool)]
        p2 = period_mask(dd)
        mcs = dd["mcsgarch"].to_numpy()
        stats = {}
        stats["G"] = period_diff_test((dd["mcsgarch"] - dd["full"]).to_numpy(), mcs, p2)
        stats["B"] = period_diff_test((dd["mcsgarch_iv"] - dd["mcsgarch"]).to_numpy(), mcs, p2)
        stats["E"] = period_diff_test((dd["mcsgarch_iv"] - dd["full"]).to_numpy(), mcs, p2)
        for v in [x for x in VARIANTS[1:] if x in dd]:
            stats[v] = period_diff_test((dd[v] - dd["full"]).to_numpy(), mcs, p2)
        # the hybrid against mcsgarch_iv in % of mcsgarch_iv's QLIKE, the sign convention of gw_group_margins.csv
        # (negative = the hybrid is better)
        e_pct = {k: 100 * (dd.loc[m, "full"].mean() / dd.loc[m, "mcsgarch_iv"].mean() - 1) for k, m in (("1", ~p2), ("2", p2))}
        for k, st in (stats.items() if record else []):
            period_rows.append({"target": t, "seed": seed_txt, "subset": subset, "quantity": k,
                                "period1_points": st["r1"], "period2_points": st["r2"], "difference": st["diff"],
                                "se": st["se"], "t": st["stat"], "p_two_sided": st["p"], "days_p1": st["n1"],
                                "days_p2": st["n2"],
                                **({"hybrid_vs_mcsgarch_iv_pct_p1": e_pct["1"], "hybrid_vs_mcsgarch_iv_pct_p2": e_pct["2"]}
                                   if k == "E" else {})})
        for v in role_groups:
            if v not in stats:
                continue
            st = stats[v]
            R.add(f"{prefix}_{t}_{v}", f"The contribution of the {GROUP_LABELS[v]} group to the hybrid differs between the"
                  f" multi-weekly period (to 19 Nov 2024) and the period since 20 Nov 2024 ({TARGET_LABEL[t]}, {subset})."
                  " Estimate: period 2 minus period 1, points of each period's mcsGARCH QLIKE.",
                  fam_groups[0], sample_text(nifty, t, dd, f"{subset}, {st['n1']} + {st['n2']} days", seed_txt),
                  st["diff"], st["stat"], st["p"], "two-sided", DELTA_METHOD, fam_groups[1], index=nifty, target=t,
                  variant=v, subset=subset, seed=seed_txt, period1=st["r1"], period2=st["r2"])
            if boot:
                rng = np.random.default_rng([BOOT_SEED, 7, targets.index(t), VARIANTS.index(v), rng_key])
                b = period_diff_bootstrap((dd[v] - dd["full"]).to_numpy(), mcs, p2, reps=reps, rng=rng)
                R.add(f"R5_{t}_{v}", f"As {prefix}_{t}_{v}, with a day-block bootstrap instead of the delta method.",
                      "robustness", sample_text(nifty, t, dd, f"{subset}, {st['n1']} + {st['n2']} days", seed_txt),
                      b["diff"], np.nan, b["p"], "two-sided", BOOT_METHOD, "R5_period_bootstrap", index=nifty, target=t,
                      variant=v, ci_low=b["q025"], ci_high=b["q975"])
        if with_edge:
            for k, txt in (("E", "The hybrid's edge over the information-matched mcsGARCH (mcsgarch_iv)"),
                           ("G", "The hybrid's gain over mcsGARCH"),
                           ("B", "The extra loss of mcsgarch_iv over mcsGARCH (the benchmark's own change)")):
                st = stats[k]
                if with_edge == "exploratory":
                    tid, role, fam = f"E5_{t}_{k}", "exploratory", "E5_last_hour_periods"
                elif subset == "all days":
                    tid, role = f"S{1 if k == 'E' else 2}_{t}_{k}", "secondary"
                    fam = "S1_edge_periods" if k == "E" else "S2_decomposition"
                else:
                    tid, role, fam = f"R6_{t}_{k}_{subset.replace(' ', '_')}", "robustness", "R6_edge_robustness"
                R.add(tid, f"{txt} differs between the multi-weekly period and the period since 20 Nov 2024"
                      f" ({TARGET_LABEL[t]}, {subset}). Estimate: period 2 minus period 1, points of each period's mcsGARCH QLIKE.",
                      role, sample_text(nifty, t, dd, f"{subset}, {st['n1']} + {st['n2']} days", seed_txt),
                      st["diff"], st["stat"], st["p"], "two-sided", DELTA_METHOD, fam, index=nifty, target=t,
                      quantity=k, subset=subset, period1=st["r1"], period2=st["r2"])
        return stats

    for t in targets:
        d = D[(t, HEADLINE_SEED)]
        if t in PRIMARY_TARGETS or synth:
            st = period_block(d, t, "seed set 7 (headline)", PRIMARY_GROUPS, ("primary", "primary"), "P2", boot=True)
            # share of the change in the edge that is the benchmark's own change, with a bootstrap interval
            p2 = period_mask(d)
            rng = np.random.default_rng([BOOT_SEED, 9, targets.index(t)])
            mats = []
            for mask in (~p2, p2):
                x = d.loc[mask, ["mcsgarch", "mcsgarch_iv", "full"]].to_numpy()
                idx = rf.stationary_bootstrap_indices(len(x), reps, MEAN_BLOCK, rng)
                mats.append(rf.bootstrap_means(x, idx))
            def _b(m):
                return 100 * (m[:, 1] - m[:, 0]) / m[:, 0]
            def _e(m):
                return 100 * (m[:, 1] - m[:, 2]) / m[:, 0]
            sh = (_b(mats[1]) - _b(mats[0])) / (_e(mats[1]) - _e(mats[0]))
            period_rows.append({"target": t, "seed": "seed set 7 (headline)", "subset": "all days",
                                "quantity": "share of the change in the edge that is the benchmark's (dB/dE)",
                                "difference": st["B"]["diff"] / st["E"]["diff"],
                                "ci_low": float(np.quantile(sh, 0.025)), "ci_high": float(np.quantile(sh, 0.975))})
            period_block(d, t, "seed set 7 (headline)", ["optionmkt", "mcsinfo"], ("secondary", "S3b_combined_periods"),
                         "S3b", with_edge=False, record=False)
            period_block(d, t, "seed set 7 (headline)", ["implied_level", "implied_shape"],
                         ("exploratory", "E3_implied_split_periods"), "E3", with_edge=False, record=False)
            # robustness: period 2 without the auction days; period 1 without the EWMA-fallback days
            pre = pd.to_datetime(d.index).to_numpy() < np.datetime64(pd.Timestamp(CLOSING_AUCTION_START))
            if not pre.all():
                period_block(d, t, "seed set 7 (headline)", PRIMARY_GROUPS, ("robustness", "R7_periods_pre_auction"),
                             "R7", mask_days=pre, subset="before the closing auction")
            if ew_starts:
                tt = d["t"].to_numpy()
                ew = np.any([(tt >= s0) & (tt < s0 + PIPELINE_SETTINGS["block"]) for s0 in ew_starts], axis=0)
                period_block(d, t, "seed set 7 (headline)", PRIMARY_GROUPS, ("robustness", "R8_periods_excl_ewma"), "R8",
                             mask_days=~ew, subset="without the EWMA-fallback days")
            # seed repeats of the period differences, and the seed-pooled version
            have = [s for s in seeds_all if (t, s) in D]
            for v in REPEAT_VARIANTS[1:]:
                est = {}
                for s in have:
                    ds_ = D[(t, s)]
                    est[s] = period_diff_test((ds_[v] - ds_["full"]).to_numpy(), ds_["mcsgarch"].to_numpy(),
                                              period_mask(ds_))["diff"]
                vals = np.array(list(est.values()))
                seed_info[("P2", t, v)] = est
                seed_rows.append({"part": "2 (period 2 minus period 1)", "target": t, "variant": v,
                                  "seeds": " ".join(map(str, have)), "k": len(have),
                                  **{f"estimate_s{s}": e for s, e in est.items()}, "mean": vals.mean(),
                                  "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan, "min": vals.min(),
                                  "max": vals.max(), "n_positive": int((vals > 0).sum())})
            if len(have) > 1:
                pooled = pd.DataFrame({c: sum(D[(t, s)][c] for s in have) / len(have)
                                       for c in ["mcsgarch", "mcsgarch_iv"] + REPEAT_VARIANTS})
                pooled["t"] = d["t"]
                period_block(pooled, t, f"daily QLIKE averaged over seed sets {', '.join(map(str, have))}",
                             PRIMARY_GROUPS, ("robustness", "R9_periods_seed_pooled"), "R9", with_edge=False)
        else:
            period_block(d, t, "seed set 7 (headline)", PRIMARY_GROUPS, ("exploratory", "E5_last_hour_periods"), "E5",
                         with_edge="exploratory")
        # descriptive: the three regimes
        for k, name in enumerate(REGIMES):
            m = d["regime"].to_numpy() == k
            if not m.any():
                continue
            dd = d[m]
            mc = dd["mcsgarch"].mean()
            row = {"target": t, "regime": name, "days": int(m.sum()),
                   "hybrid_gain_points": points(mc - dd["full"].mean(), mc),
                   "mcsgarch_iv_extra_loss_points": points(dd["mcsgarch_iv"].mean() - mc, mc),
                   "edge_over_mcsgarch_iv_points": points(dd["mcsgarch_iv"].mean() - dd["full"].mean(), mc),
                   "hybrid_vs_mcsgarch_iv_pct": 100 * (dd["full"].mean() / dd["mcsgarch_iv"].mean() - 1)}
            for v in VARIANTS[1:]:
                row[f"contribution_{v}"] = points(dd[v].mean() - dd["full"].mean(), mc)
            regime_rows.append(row)

    # ---------------- other indices (secondary)
    other_rank = []
    if not synth:
        for index in ("BANKNIFTY", "FINNIFTY"):
            for t in PRIMARY_TARGETS:
                if not all(work_file(work, index, t, HEADLINE_SEED, v).exists() for v in OTHER_INDEX_VARIANTS):
                    continue
                wide = load_wide(work, index, t, HEADLINE_SEED, OTHER_INDEX_VARIANTS)
                committed = read_csv_exact(INDEXES[index]["run"] / f"forecasts_{t}.csv")
                repro_rows.append(reproduction_check(wide, committed, f"{index} {t} seed 7 vs {INDEXES[index]['run'].name}"))
                d = day_frame(wide, OTHER_INDEX_VARIANTS)
                fam = "S5_banknifty" if index == "BANKNIFTY" else "S6_finnifty"
                part1_rows(d, t, index, "seed set 7 (headline)", "secondary", fam, PRIMARY_GROUPS, fam.split("_")[0])
                for v in PRIMARY_GROUPS:
                    rng = np.random.default_rng([BOOT_SEED, 3, PRIMARY_TARGETS.index(t), VARIANTS.index(v),
                                                 1 if index == "BANKNIFTY" else 2])
                    sh, lo, hi = share_ci(d["mcsgarch"].to_numpy(), d["full"].to_numpy(), d[v].to_numpy(), reps=reps, rng=rng)
                    share_rows.append({"index": index, "target": t, "variant": v, "share_of_gain": sh,
                                       "share_ci_low": lo, "share_ci_high": hi})
                nif = {v: points(D[(t, HEADLINE_SEED)][v].mean() - D[(t, HEADLINE_SEED)]["full"].mean(),
                                 D[(t, HEADLINE_SEED)]["mcsgarch"].mean()) for v in PRIMARY_GROUPS}
                oth = {v: points(d[v].mean() - d["full"].mean(), d["mcsgarch"].mean()) for v in PRIMARY_GROUPS}
                rho = spearmanr([nif[v] for v in PRIMARY_GROUPS], [oth[v] for v in PRIMARY_GROUPS]).statistic
                other_rank.append({"index": index, "target": t, "spearman_rank_corr_with_nifty": float(rho),
                                   **{f"nifty_{v}": nif[v] for v in PRIMARY_GROUPS},
                                   **{f"{index.lower()}_{v}": oth[v] for v in PRIMARY_GROUPS}})

    # ---------------- adjust, verdicts, write
    families = {
        "primary": "all primary rows (part 1 and part 2, 15 and 30 minutes)",
        "S1_edge_periods": "secondary S1", "S2_decomposition": "secondary S2", "S3_option_and_mcsinfo": "secondary S3",
        "S3b_combined_periods": "secondary S3b", "S4_last_hour": "secondary S4", "S5_banknifty": "secondary S5",
        "S6_finnifty": "secondary S6", "R1_seed_pooled": "robustness R1", "R2_excl_0915": "robustness R2",
        "R3_pre_auction": "robustness R3", "R5_period_bootstrap": "robustness R5", "R6_edge_robustness": "robustness R6",
        "R7_periods_pre_auction": "robustness R7", "R8_periods_excl_ewma": "robustness R8",
        "R9_periods_seed_pooled": "robustness R9", "PL1_seed_placebo": "placebo PL1",
        "PL2_seed_placebo_periods": "placebo PL2", "E1_implied_split": None, "E2_expiry": None,
        "E3_implied_split_periods": None, "E4_last_hour_combined": None, "E5_last_hour_periods": None}
    tests = R.frame(families)
    detail = pd.DataFrame(R.detail)
    verdicts = []
    for _, r in tests[tests["_family"] == "primary"].iterrows():
        part, t, v = r["test_id"].split("_", 2)[0], r["test_id"].split("_")[1], r["test_id"].split("_", 2)[2]
        est = seed_info.get((part, t, v), {})
        head = r["estimate"]
        sign = np.sign(head)
        agree = int(sum(np.sign(e) == sign for e in est.values())) if part == "P2" else int(sum(e > 0 for e in est.values()))
        sd = float(np.std(list(est.values()), ddof=1)) if len(est) > 1 else np.nan
        lab = seed_label(agree, len(est), r["p_adjusted"])
        if part == "P1" and head <= 0 and r["p_adjusted"] < ALPHA:
            lab = "not supported"
        verdicts.append({"test_id": r["test_id"], "part": part, "target": t, "group": v, "estimate_seed7": head,
                         "p_raw": r["p_raw"], "p_holm": r["p_adjusted"], "seed_sets": len(est),
                         "seed_sets_agreeing": agree, "seed_sd": sd,
                         "within_seed_noise": bool(np.isfinite(sd) and abs(head) < 2 * sd),
                         "direction": ("P2 > P1" if head > 0 else "P2 < P1") if part == "P2" else "",
                         "verdict": lab})
    verdicts = pd.DataFrame(verdicts)
    tests_out = tests.drop(columns="_family")
    tests_out.to_csv(out / "tests.csv", index=False)
    detail.to_csv(out / "tests_detail.csv", index=False)
    verdicts.to_csv(out / "verdict.csv", index=False)
    pd.DataFrame(contrib_rows).to_csv(out / "contributions.csv", index=False)
    pd.DataFrame(share_rows).to_csv(out / "shares.csv", index=False)
    pd.DataFrame(period_rows).to_csv(out / "periods.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(out / "regimes.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "seed_repeats.csv", index=False)
    pd.DataFrame(other_rank).to_csv(out / "other_index_ranking.csv", index=False)
    pd.DataFrame(repro_rows).to_csv(out / "reproduction_check.csv", index=False)
    for (t, s), d in D.items():
        d.to_csv(out / f"daily_losses_{nifty.lower()}_{t}_s{s}.csv")
    if (work / "timing.csv").exists():
        pd.read_csv(work / "timing.csv").to_csv(out / "timing.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\n[{label}] primary rows and verdicts:")
    print(verdicts.to_string(index=False))
    print(f"\n[{label}] contributions (points of mcsGARCH's QLIKE; positive = the group helps):")
    print(pd.DataFrame(contrib_rows).to_string(index=False))
    print(f"\n[{label}] shares of the gain:")
    print(pd.DataFrame(share_rows).to_string(index=False))
    print(f"\n[{label}] periods:")
    print(pd.DataFrame(period_rows).to_string(index=False))
    print(f"\n[{label}] seed repeats:")
    print(pd.DataFrame(seed_rows).to_string(index=False))
    print(f"\n[{label}] reproduction checks:")
    print(pd.DataFrame(repro_rows).to_string(index=False))
    if other_rank:
        print(pd.DataFrame(other_rank).to_string(index=False))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    import arch
    import lightgbm
    import scipy
    inputs = {}
    for ix in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
        p = work / f"inputs_{ix}.json"
        if p.exists():
            inputs[ix] = json.loads(p.read_text())
    timing = pd.read_csv(work / "timing.csv") if (work / "timing.csv").exists() else pd.DataFrame()
    # stage-1 wall clock per index (added after the plan commit): from the input-hash file, written when a run
    # starts, to the last work file written
    wall = {}
    for ix in ("NIFTY", "BANKNIFTY", "FINNIFTY", "SYNTH"):
        files = list(work.glob(f"{ix}_*.csv"))
        start_file = work / f"inputs_{ix}.json"
        if files:
            end = max(f.stat().st_mtime for f in files)
            start = start_file.stat().st_mtime if start_file.exists() else None
            wall[ix] = {"start_utc": (pd.Timestamp(start, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ") if start else None),
                        "end_utc": pd.Timestamp(end, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "minutes": round((end - start) / 60, 1) if start else None}
    json.dump({"source": label + ("" if synth else ": NSE bhavcopy, participant OI and India VIX; third-party 1-minute "
                                  "index bars validated against NSE daily values"),
               "study": "P106 gain decomposition (docs/prereg/P106_GAIN_DECOMPOSITION.md)",
               "plan_commit": a.plan_commit, "analysis_commit": commit,
               "commands": ["PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index NIFTY --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index BANKNIFTY --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index FINNIFTY --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py analyze --plan-commit <plan commit>"],
               "committed_runs": {k: str(v["run"].relative_to(ROOT)) for k, v in INDEXES.items()},
               "pipeline_settings": PIPELINE_SETTINGS, "synthetic_settings": SYNTH_SETTINGS if synth else None,
               "groups": GROUPS, "drop_sets": DROP_SETS, "hybrid_columns": HYBRID_COLUMNS,
               "seeds": {"headline_base_seed": HEADLINE_SEED, "repeat_base_seeds": rseeds,
                         "lgbm_seeds": {str(s): [s + 1000 * k for k in range(N_SEEDS)] for s in seeds_all},
                         "note": "the base seed also seeds the tuning pass that picks the number of boosting rounds"},
               "period_split": PERIOD_SPLIT, "bootstrap": {"reps": reps, "mean_block_days": MEAN_BLOCK, "seed": BOOT_SEED},
               "alpha": ALPHA, "input_sha256": inputs,
               "seconds_stage1_jobs": float(timing["seconds"].sum()) if len(timing) else None,
               "stage1_wall_clock": wall,
               "seconds_stage2": round(time.time() - t0, 1), "python": sys.version.split()[0],
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
                            "arch": arch.__version__, "scipy": scipy.__version__}},
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
            p.add_argument("--targets", nargs="+", default=None)
            p.add_argument("--seeds", nargs="+", type=int, default=None)
            p.add_argument("--variants", nargs="+", default=None)
            p.add_argument("--chunk", type=int, default=CHUNK)
            p.add_argument("--workers", type=int, default=2)
        if name == "check":
            p.add_argument("--indexes", nargs="+", default=["NIFTY"])
            p.add_argument("--targets", nargs="+", default=None)
        if name == "analyze":
            p.add_argument("--out", default=None)
            p.add_argument("--reps", type=int, default=REPS)
            p.add_argument("--plan-commit", default=None)
    a = ap.parse_args(argv)
    if a.work is None:
        a.work = str(ROOT / "data" / "processed" / ("p106_synthetic" if a.synthetic else "p106"))
    if a.stage == "analyze" and a.out is None:
        a.out = str(ROOT / "data" / "processed" / "p106_synthetic_results") if a.synthetic else \
            str(ROOT / "results" / "p106_gain_decomposition")
    if a.stage == "run":
        stage_run(a)
    elif a.stage == "check":
        stage_check(a)
    else:
        stage_analyze(a)


if __name__ == "__main__":
    main()
