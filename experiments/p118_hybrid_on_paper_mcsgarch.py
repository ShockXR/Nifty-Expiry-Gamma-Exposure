"""P118: the hybrid built on mcsGARCH as the paper specifies it (registered plan:
docs/prereg/P118_HYBRID_ON_PAPER_MCSGARCH.md). Done by an AI agent (Claude) for Shreyan Thanki.

The committed hybrid (``hybrid_base``) is mcs_ours times exp(a LightGBM correction learnt from 41 inputs), where
mcs_ours is our implementation of mcsGARCH (separate expiry-day diurnal curve, 5-minute smoothing, zero mean).
P116 refitted mcsGARCH as Chhimwal, Pandey and Pandey (2025) specify it (mcs_cpp: one diurnal curve, no smoothing,
ARMA(1,1) mean, Student t) and found it forecasts Nifty better than mcs_ours. This script builds the identical
hybrid recipe on mcs_cpp (``hybrid_cpp``): the same 41 inputs, LightGBM settings, 21-day walk-forward refits and
seed sets, with mcs_cpp's forecast in place of mcs_ours's both as the multiplicative base and as the input
``log_mcs`` (``hybridalt_base`` of niftygex.forecast with ``alt_base`` = the paper's specification). It asks
(1) how much better hybrid_cpp is than mcs_cpp, and (2) whether the learnt correction absorbs the choice of base,
i.e. whether hybrid_cpp and the committed hybrid are equivalent within 0.5 point of mcs_cpp's QLIKE.

Job kinds (one walk-forward per index, target, seed set and kind; the pipeline's run_walkforward):

    repro     mcsgarch, mcsgarch_iv, mcsgarch_alt (= mcs_cpp) and hybrid_base: the reproduction check of phase 1
    cpp       mcsgarch, mcsgarch_iv, mcsgarch_alt and hybridalt_base (= hybrid_cpp)
    cpp_base  as cpp, plus hybrid_base (a reproduction check of the committed hybrid inside the study run)
    cppall    every mcsGARCH of the run as the paper specifies it (mcsgarch, mcsgarch_iv and so the inputs
              log_mcs, log_s_window, log_mcs_iv, log_implied_window), with hybrid_base on it (exploratory E1)

Stages:

    # phase 1: reproduction of the committed hybrid and of P116's mcs_cpp (forecasts only, no loss)
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index NIFTY --phase repro --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py check --indexes NIFTY
    # phase 2: the study (REAL DATA), then the analysis
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index NIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index BANKNIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index FINNIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py analyze --plan-commit <hash>

``--synthetic`` runs every stage on two simulated markets with small settings (labelled SYNTHETIC). Work files go
to --work (default data/processed/p118, git-ignored); a finished job is not rerun. ``analyze`` writes
results/p118_hybrid_on_paper_mcsgarch/. Tests: tests/test_p118_hybrid_on_paper_mcsgarch.py (synthetic data only).
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
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel  # noqa: E402
from niftygex.forecast import run_walkforward  # noqa: E402
from niftygex.implied import build_implied_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    """Load a script as a module under its own name, registered in sys.modules (as P116 does)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# P106: the committed runs' WalkForwardConfig (make_config), settings check, work-file names, input hashes, the
# market loader (rp), the stationary bootstrap (rf) and the tests.csv collector (Rows).
gd = _load("p118_copy_of_p106_gain_decomposition", ROOT / "experiments" / "p106_gain_decomposition.py")
rp, rf = gd.rp, gd.rf
# P110: TOST, non-inferiority and the delta-method TOST, with the DM test's variance (tested there).
st = _load("p118_copy_of_p110_gain_stress_test", ROOT / "experiments" / "p110_gain_stress_test.py")
# P116: the paper's specification of mcsGARCH (mcs_cpp) and, through its copy of P109's driver, the per-block
# refits that produced P116's mcs_cpp forecasts (used for the synthetic cross-check) and the daily-loss frames.
p116 = _load("p118_copy_of_p116_benchmark_as_printed", ROOT / "experiments" / "p116_benchmark_as_printed.py")
ex = p116.ex

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

# mcs_cpp exactly as P116 fitted it: McsGarch(mean="arma11", dist="t", expiry_diurnal=False, smooth=1); its
# forecast of y is the variance part plus the expected squared conditional mean (P116's mean_sq=True).
CPP_SPEC = dict(mean="arma11", dist="t", expiry_diurnal=False, smooth=1)
_p116_cpp = p116.P116_SPECS["mcs_cpp"]
assert (_p116_cpp["expiry_diurnal"], _p116_cpp["smooth"], _p116_cpp["mean"], _p116_cpp["mean_sq"]) == \
    (CPP_SPEC["expiry_diurnal"], CPP_SPEC["smooth"], CPP_SPEC["mean"], True)

KIND_MODELS = {
    "repro": ["mcsgarch", "mcsgarch_iv", "mcsgarch_alt", "hybrid_base"],
    "cpp": ["mcsgarch", "mcsgarch_iv", "mcsgarch_alt", "hybridalt_base"],
    "cpp_base": ["mcsgarch", "mcsgarch_iv", "mcsgarch_alt", "hybrid_base", "hybridalt_base"],
    "cppall": ["mcsgarch", "mcsgarch_iv", "hybrid_base"],
}
KIND_SETTINGS = {
    "repro": dict(alt_base=CPP_SPEC),
    "cpp": dict(alt_base=CPP_SPEC),
    "cpp_base": dict(alt_base=CPP_SPEC),
    "cppall": dict(expiry_diurnal=False, diurnal_smooth=1, mcs_mean="arma11"),
}
STUDY_KINDS = ("cpp", "cpp_base")          # the job kinds that hold hybrid_cpp

HEADLINE_SEED = 7
REPEAT_SEEDS = [101, 102, 103, 104]        # P99's first four new seed sets, as P106 and P110
ALL_SEEDS = [HEADLINE_SEED] + REPEAT_SEEDS
NIFTY_TARGETS = ["next15", "next30", "last_hour"]
PRIMARY_TARGETS = ["next15", "next30"]
OTHER_INDEXES = {"BANKNIFTY": "S4_banknifty", "FINNIFTY": "S5_finnifty"}
MARGIN = 0.5                               # points of mcs_cpp's mean daily QLIKE (equivalence margin of H2)
ALPHA = 0.05
REPS = 10000
MEAN_BLOCK = 10.0
BOOT_SEED = 118
INDEXES = gd.INDEXES
P106_RESULTS = ROOT / "results" / "p106_gain_decomposition"
P109_RESULTS = ROOT / "results" / "p109_mcsgarch_extensions"
P110_RESULTS = ROOT / "results" / "p110_gain_stress_test"
P116_RESULTS = ROOT / "results" / "p116_benchmark_as_printed"
P116_REF = ROOT / "data" / "processed" / "p116_ref"      # copy of P116's git-ignored block files (forecasts)
TARGET_LABEL = {"next15": "next 15 min", "next30": "next 30 min", "last_hour": "last hour"}
LABEL = {"NIFTY": "Nifty", "BANKNIFTY": "Bank Nifty", "FINNIFTY": "Fin Nifty", "SYNTH": "synthetic A (stands in for Nifty)",
         "SYNTHB": "synthetic B (stands in for Bank Nifty)"}
SYNTH = dict(n_days=120, start="2024-03-01", market_seeds={"SYNTH": 3, "SYNTHB": 4}, min_train=60, block=20,
             n_seeds=1, targets=["next15", "next30"], seeds=[7, 101], horizon_step=45)
SYNTH_OTHER = {"SYNTHB": "S4_banknifty"}
ID_COLS = ["t", "date", "minute", "horizon", "is_expiry", "regime", "y"]
DM_METHOD = "Diebold-Mariano on daily mean QLIKE, Newey-West, HLN factor, Student t(n-1)"
TOST_METHOD = ("two one-sided tests (TOST) on the daily difference in points of mcs_cpp's QLIKE, margin {d} point(s), "
               "DM variance (Newey-West, HLN factor, Student t(n-1)); p = larger one-sided p (P110's tost)")
TOST_DELTA = ("TOST on the ratio 100 mean(hybrid - hybrid_cpp) / mean(mcs_cpp), delta-method Newey-West variance, "
              "HLN factor, Student t(n-1), margin {d} point(s) (P110's tost_delta_method)")


def is_synth(index: str) -> bool:
    return index.startswith("SYNTH")


def model_label(index: str) -> str:
    return LABEL.get(index, index)


# ---------------------------------------------------------------------------
# Job plan and stage 1 (forecasts)
# ---------------------------------------------------------------------------


def targets_of(index: str) -> list:
    if is_synth(index):
        return list(SYNTH["targets"])
    return list(NIFTY_TARGETS) if index == "NIFTY" else list(PRIMARY_TARGETS)


def seeds_of(index: str) -> list:
    return list(SYNTH["seeds"]) if is_synth(index) else list(ALL_SEEDS)


def is_main_index(index: str) -> bool:
    return index in ("NIFTY", "SYNTH")


def job_plan(index: str, phase: str = "study", seeds=None, targets=None, kinds=None) -> list[tuple]:
    """(index, target, seed, kind) in the registered priority order.

    Phase "repro": the main index, seed set 7, every target, kind repro.
    Phase "study", main index: hybrid_cpp with seed set 7 at every target; then seed sets 101 to 104 at 15 and 30
    minutes (the first of them at 15 minutes also refits the committed hybrid, kind cpp_base); then the exploratory
    cppall with seed set 7 at every target. Other indices: seed set 7 at 15 and 30 minutes (kind cpp_base), then
    seed sets 101 to 104 (kind cpp). Optional filters keep the order."""
    tl = targets_of(index)
    sl = seeds_of(index)
    plan = []
    if phase == "repro":
        plan = [(index, t, sl[0], "repro") for t in tl]
    elif is_main_index(index):
        plan += [(index, t, sl[0], "cpp") for t in tl]
        for k, s in enumerate(sl[1:]):
            for t in [x for x in tl if x in PRIMARY_TARGETS]:
                plan.append((index, t, s, "cpp_base" if (k == 0 and t == "next15") else "cpp"))
        plan += [(index, t, sl[0], "cppall") for t in tl]
    else:
        tl = [x for x in tl if x in PRIMARY_TARGETS]
        plan += [(index, t, sl[0], "cpp_base") for t in tl]
        for s in sl[1:]:
            plan += [(index, t, s, "cpp") for t in tl]
    out = []
    for (ix, t, s, k) in plan:
        if seeds is not None and s not in seeds:
            continue
        if targets is not None and t not in targets:
            continue
        if kinds is not None and k not in kinds:
            continue
        out.append((ix, t, s, k))
    return out


def job_file(work: Path, index: str, target: str, seed: int, kind: str) -> Path:
    return Path(work) / f"{index}_{target}_s{seed}_{kind}.csv"


def make_job_config(index: str, seed: int, kind: str, eval_minutes, sets: dict):
    """The committed runs' WalkForwardConfig (P106's make_config) with this job's models and settings."""
    over = dict(models=list(KIND_MODELS[kind]), **KIND_SETTINGS[kind])
    if is_synth(index):
        over.update(min_train=SYNTH["min_train"], block=SYNTH["block"], mcs_refit_every=SYNTH["block"],
                    n_seeds=SYNTH["n_seeds"])
    return gd.make_config(seed, eval_minutes, sets, ["full"], **over)


_STATE: dict = {}   # filled in the parent before the workers are forked


def run_job(job):
    (index, target, seed, kind), work = job[:4], Path(job[4])
    t0 = time.time()
    t_start = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg = make_job_config(index, seed, kind, _STATE["eval_minutes"].get(target), _STATE["sets"])
    pred, info = run_walkforward(_STATE["market"], _STATE["datasets"][target], cfg)
    pred = pred.copy()
    pred["date"] = pd.DatetimeIndex(pred["date"]).strftime("%Y-%m-%d")
    cols = ID_COLS + [m for m in KIND_MODELS[kind] if m in pred]
    path = job_file(work, index, target, seed, kind)
    tmp = path.with_suffix(".tmp")
    pred[cols].to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)
    params = pd.DataFrame(info.get("mcs_alt_params") or info["mcs_params"])
    params.insert(0, "which", "mcsgarch_alt" if info.get("mcs_alt_params") else "mcsgarch")
    params.to_csv(path.with_name(path.stem + "_params.csv"), index=False)
    secs = round(time.time() - t0, 1)
    t_end = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(work / "timing.csv", "a") as fh:
        fh.write(f"{index},{target},{seed},{kind},{'+'.join(KIND_MODELS[kind])},{secs},{len(pred)},{os.getpid()},"
                 f"{t_start},{t_end}\n")
    return index, target, seed, kind, secs, len(pred)


def synthetic_market(index: str):
    from niftygex.simulate import SimConfig, simulate_market
    return simulate_market(SimConfig(n_days=SYNTH["n_days"], start=SYNTH["start"],
                                     seed=SYNTH["market_seeds"][index], kappa=0.0))


def synthetic_targets() -> dict:
    return {t: dict(horizon=int(t[4:]), step=SYNTH["horizon_step"]) for t in SYNTH["targets"]}


def synthetic_reference(m: dict, index: str, work: Path, datasets: dict, sets: dict):
    """SYNTHETIC stand-ins for the committed inputs of the real study:
    the committed run (mcsgarch, mcsgarch_iv, hybrid_base with every seed set; the pipeline's own walk-forward),
    written as work/<index>_committed_<target>_s<seed>.csv; and P116's mcs_cpp, made by P116's own driver (its copy
    of P109's run_block) into work/p116_ref_synth/."""
    for t, ds in datasets.items():
        for s in SYNTH["seeds"]:
            p = Path(work) / f"{index}_committed_{t}_s{s}.csv"
            if p.exists():
                continue
            cfg = gd.make_config(s, None, sets, ["full"], min_train=SYNTH["min_train"], block=SYNTH["block"],
                                 mcs_refit_every=SYNTH["block"], n_seeds=SYNTH["n_seeds"])
            pred, _ = run_walkforward(m, ds, cfg)
            pred = pred.copy()
            pred["date"] = pd.DatetimeIndex(pred["date"]).strftime("%Y-%m-%d")
            pred.to_csv(p, index=False, float_format="%.17g")
    ref = Path(work) / "p116_ref_synth"
    ref.mkdir(parents=True, exist_ok=True)
    R = m["R"]
    ex._STATE.update(R=R, is_exp=m["labels"]["is_expiry"].to_numpy(), days=m["days"], block=SYNTH["block"],
                     variants=["mcsgarch", "mcs_cpp"],
                     targets={t: (int(t[4:]), SYNTH["horizon_step"], None) for t in SYNTH["targets"]})
    for start in range(SYNTH["min_train"], R.shape[0], SYNTH["block"]):
        if not ex.block_file(ref, index, start).exists():
            ex.run_block((index, start, str(ref)))


def stage_run(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    index = a.index
    synth = is_synth(index)
    plan = job_plan(index, a.phase, seeds=a.seeds, targets=a.targets, kinds=a.kinds)
    jobs = [j for j in plan if not job_file(work, *j).exists()]
    label = "SYNTHETIC (simulated market, not real data)" if synth else "REAL DATA"
    print(f"[{label}] P118 hybrid on the paper's mcsGARCH, stage 1 ({a.phase}), {index}: {len(jobs)} jobs to run "
          f"({len(plan) - len(jobs)} already in {work}); workers {a.workers}", flush=True)
    if not jobs and not synth:
        return
    t0 = time.time()
    if synth:
        m = synthetic_market(index)
        targets = synthetic_targets()
        eval_minutes = {}
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
        targets = rp.TARGETS
        eval_minutes = dict(rp.EVAL_MINUTES)
        hp = work / f"inputs_{index}.json"
        if not hp.exists():
            hp.write_text(json.dumps(gd.input_hashes(Path(a.raw), index), indent=2))
    print(f"market loaded: {len(m['days'])} days, {m['days'][0]} to {m['days'][-1]} ({time.time() - t0:.0f}s)", flush=True)
    panel = build_gex_panel(m)
    implied = build_implied_panel(m)
    has_oracle = "dealer_next" in m
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle" or has_oracle}
    need = sorted({j[1] for j in jobs} | (set(targets) if synth else set()))
    datasets = {t: build_dataset(m, panel, implied=implied, **targets[t]) for t in need}
    print(f"panels and datasets built ({time.time() - t0:.0f}s)", flush=True)
    if synth:
        synthetic_reference(m, index, work, datasets, sets)
    _STATE.update(market=m, datasets=datasets, sets=sets, eval_minutes=eval_minutes)
    if not (work / "timing.csv").exists():
        (work / "timing.csv").write_text("index,target,seed,kind,models,seconds,rows,pid,start_utc,end_utc\n")
    done = 0
    args = [j + (str(work),) for j in jobs]

    def report(res):
        ix, target, seed, kind, secs, n = res
        print(f"  [{done}/{len(jobs)}] {ix} {target} seed {seed} {kind}: {n} forecasts, {secs:.0f}s "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)

    if a.workers == 1:
        for j in args:
            done += 1
            report(run_job(j))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=a.workers, maxtasksperchild=1) as pool:
            for res in pool.imap(run_job, args, chunksize=1):
                done += 1
                report(res)
    print(f"stage 1 ({index}, {a.phase}) done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Reading the work files; reproduction checks (forecasts only, no loss)
# ---------------------------------------------------------------------------


def read_sorted(path: Path) -> pd.DataFrame:
    f = gd.read_csv_exact(path)
    f["date"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
    return f.sort_values(["t", "minute"]).reset_index(drop=True)


def committed_path(index: str, target: str, work: Path, seed: int = HEADLINE_SEED) -> Path:
    if is_synth(index):
        return Path(work) / f"{index}_committed_{target}_s{seed}.csv"
    return INDEXES[index]["run"] / f"forecasts_{target}.csv"


def committed(index: str, target: str, work: Path, seed: int = HEADLINE_SEED) -> pd.DataFrame:
    """The committed run's test anchors with y, mcsgarch (mcs_ours), mcsgarch_iv and hybrid_base (seed set 7)."""
    c = read_sorted(committed_path(index, target, work, seed))
    return c[ID_COLS + ["mcsgarch", "mcsgarch_iv", "hybrid_base"]]


def p116_reference(index: str, target: str, work: Path, ref_dir: Path) -> pd.DataFrame | None:
    """P116's mcs_cpp forecasts (t, minute, mcs_cpp) of one index and target, or None if not on disk."""
    ref = Path(work) / "p116_ref_synth" if is_synth(index) else Path(ref_dir)
    files = sorted(ref.glob(f"{index}_block[0-9][0-9][0-9][0-9].csv"))
    if not files:
        return None
    a = pd.concat([gd.read_csv_exact(f) for f in files], ignore_index=True)
    a = a[a["target"] == target][["t", "minute", "mcs_cpp"]]
    return a.sort_values(["t", "minute"]).reset_index(drop=True)


def _same(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = int(np.sum(a == b))
    rel = float(np.max(np.abs(a / b - 1))) if len(a) else np.nan
    return {"n_identical": n, "max_abs_rel_diff": rel}


def forecast_checks(work: Path, index: str, target: str, ref_dir: Path = P116_REF) -> list[dict]:
    """Every job file of one index and target against the committed forecasts, P116's mcs_cpp and each other.
    Forecasts and targets only; no loss is computed."""
    rows = []
    cpath = committed_path(index, target, work)
    if not cpath.exists():
        return rows
    c = committed(index, target, work)
    ref = p116_reference(index, target, work, ref_dir)
    alt7 = None
    for s in seeds_of(index):
        for kind in KIND_MODELS:
            p = job_file(work, index, target, s, kind)
            if not p.exists():
                continue
            f = read_sorted(p)
            base = {"index": index, "target": target, "seed": s, "kind": kind, "rows_job": len(f),
                    "rows_committed": len(c)}
            same_rows = (len(f) == len(c) and np.array_equal(f["t"].to_numpy(), c["t"].to_numpy())
                         and np.array_equal(f["minute"].to_numpy(), c["minute"].to_numpy())
                         and f["date"].equals(c["date"]))
            rows.append({**base, "check": "rows (t, minute, date) identical to the committed run's", "column": "rows",
                         "n_identical": len(f) if same_rows else 0, "max_abs_rel_diff": 0.0 if same_rows else np.nan,
                         "ok": bool(same_rows)})
            if not same_rows:
                continue
            pairs = [("y", c["y"], "committed y")]
            if kind != "cppall":
                pairs += [("mcsgarch", c["mcsgarch"], "committed mcsgarch (mcs_ours)"),
                          ("mcsgarch_iv", c["mcsgarch_iv"], "committed mcsgarch_iv")]
            if "hybrid_base" in f and kind != "cppall" and s == HEADLINE_SEED:
                pairs.append(("hybrid_base", c["hybrid_base"], "committed hybrid_base (seed set 7)"))
            if "mcsgarch_alt" in f:
                if ref is not None:
                    r = f[["t", "minute"]].merge(ref, on=["t", "minute"], how="left")
                    pairs.append(("mcsgarch_alt", r["mcs_cpp"], "P116's mcs_cpp forecasts"))
                if s == HEADLINE_SEED and alt7 is None:
                    alt7 = f["mcsgarch_alt"].to_numpy()
                elif alt7 is not None:
                    pairs.append(("mcsgarch_alt", alt7, "mcsgarch_alt of the seed-7 job (deterministic)"))
            if kind == "cppall":
                if alt7 is not None:
                    pairs.append(("mcsgarch", alt7, "mcsgarch_alt (mcs_cpp) of the seed-7 job"))
                if ref is not None:
                    r = f[["t", "minute"]].merge(ref, on=["t", "minute"], how="left")
                    pairs.append(("mcsgarch", r["mcs_cpp"], "P116's mcs_cpp forecasts"))
            for col, other, what in pairs:
                res = _same(f[col].to_numpy(), np.asarray(other, float))
                rows.append({**base, "check": f"{col} vs {what}", "column": col, **res,
                             "ok": bool(res["n_identical"] == len(f))})
    return rows


def stage_check(a):
    work = Path(a.work)
    indexes = ["SYNTH", "SYNTHB"] if a.synthetic else a.indexes
    rows = []
    for index in indexes:
        for t in targets_of(index):
            r = forecast_checks(work, index, t, Path(a.p116_ref))
            rows.extend(r)
    df = pd.DataFrame(rows)
    if not df.empty:
        pd.set_option("display.width", 220)
        print(df[["index", "target", "seed", "kind", "check", "rows_job", "n_identical", "max_abs_rel_diff", "ok"]]
              .to_string(index=False), flush=True)
        df.to_csv(work / "forecast_checks.csv", index=False)
        print(f"all identical: {bool(df['ok'].all())}", flush=True)
    return df


# ---------------------------------------------------------------------------
# Daily losses
# ---------------------------------------------------------------------------


def study_file(work: Path, index: str, target: str, seed: int) -> Path | None:
    for kind in STUDY_KINDS:
        p = job_file(work, index, target, seed, kind)
        if p.exists():
            return p
    return None


def committed_seed_losses(index: str, target: str, seed: int, work: Path, w: pd.DataFrame) -> pd.DataFrame | None:
    """Daily QLIKE of mcs_ours and of the committed hybrid with one repeat seed set: P106's committed files (Nifty),
    P110's (Bank Nifty, Fin Nifty), or, on a simulated market, the stand-in run. Columns mcsgarch, full."""
    if is_synth(index):
        p = committed_path(index, target, work, seed)
        if not p.exists():
            return None
        c = read_sorted(p)
        L = daily_losses(c, ["mcsgarch", "hybrid_base"], "qlike").rename(columns={"hybrid_base": "full"})
        return L
    src = P106_RESULTS if index == "NIFTY" else P110_RESULTS
    p = src / f"daily_losses_{index.lower()}_{target}_s{seed}.csv"
    if not p.exists():
        return None
    x = pd.read_csv(p, index_col="date", float_precision="round_trip")
    x.index = pd.to_datetime(x.index).strftime("%Y-%m-%d")
    return x[["mcsgarch", "full"]]


def build_frames(work: Path, index: str, target: str):
    """Anchor-level frame (seed set 7) and the day frame of one index and target, with the loss-level checks.

    Day-frame columns: mcs_ours, mcs_cpp, hybrid_s<seed> (the committed recipe), hybrid_cpp_s<seed>, hybrid and
    hybrid_cpp (their daily QLIKE averaged over the seed sets present for both), hybrid_cpp_all (cppall, seed 7),
    combination (P109, committed), t, is_expiry, regime. Returns (w, D, checks, seeds) or None."""
    if not committed_path(index, target, work).exists():
        return None
    p7 = study_file(work, index, target, HEADLINE_SEED)
    if p7 is None:
        return None
    c = committed(index, target, work)
    w = c[ID_COLS].copy()
    w["mcs_ours"] = c["mcsgarch"].to_numpy()
    w[f"hybrid_s{HEADLINE_SEED}"] = c["hybrid_base"].to_numpy()
    f7 = read_sorted(p7)
    w["mcs_cpp"] = f7["mcsgarch_alt"].to_numpy()
    w[f"hybrid_cpp_s{HEADLINE_SEED}"] = f7["hybridalt_base"].to_numpy()
    rebuilt_base = {}
    seeds = [HEADLINE_SEED]
    for s in seeds_of(index)[1:]:
        p = study_file(work, index, target, s)
        if p is None:
            continue
        f = read_sorted(p)
        w[f"hybrid_cpp_s{s}"] = f["hybridalt_base"].to_numpy()
        if "hybrid_base" in f:
            rebuilt_base[s] = f["hybrid_base"].to_numpy()
        seeds.append(s)
    pall = job_file(work, index, target, HEADLINE_SEED, "cppall")
    if pall.exists():
        w["hybrid_cpp_all"] = read_sorted(pall)["hybrid_base"].to_numpy()
    models = [m for m in w.columns if m not in ID_COLS]
    D = ex.day_frame(w, models)
    checks = []
    # the committed hybrid with the repeat seed sets (committed daily losses), aligned on the same days
    have = [HEADLINE_SEED]
    for s in seeds[1:]:
        x = committed_seed_losses(index, target, s, work, w)
        if x is None:
            continue
        x = x.reindex(D.index)
        res = _same(x["mcsgarch"].to_numpy(), D["mcs_ours"].to_numpy())
        checks.append({"index": index, "target": target, "seed": s,
                       "check": f"committed daily QLIKE of mcsGARCH (seed set {s} file) vs this study's mcs_ours",
                       "rows_job": len(D), **res, "ok": bool(res["n_identical"] == len(D))})
        D[f"hybrid_s{s}"] = x["full"].to_numpy()
        if s in rebuilt_base:
            ws = w[ID_COLS].copy()
            ws["rebuilt"] = rebuilt_base[s]
            Lr = daily_losses(ws, ["rebuilt"], "qlike").reindex(D.index)
            res = _same(Lr["rebuilt"].to_numpy(), D[f"hybrid_s{s}"].to_numpy())
            checks.append({"index": index, "target": target, "seed": s,
                           "check": f"daily QLIKE of the committed hybrid rebuilt here (seed set {s}) vs the committed file",
                           "rows_job": len(D), **res, "ok": bool(res["n_identical"] == len(D))})
        have.append(s)
    both = [s for s in seeds if s in have]
    D["hybrid"] = sum(D[f"hybrid_s{s}"] for s in both) / len(both)
    D["hybrid_cpp"] = sum(D[f"hybrid_cpp_s{s}"] for s in both) / len(both)
    # P116's committed daily QLIKE of mcs_ours, mcs_cpp and the hybrid on the same days
    if not is_synth(index):
        p = P116_RESULTS / f"daily_losses_{index.lower()}_{target}.csv"
        if p.exists():
            x = pd.read_csv(p, index_col="date", float_precision="round_trip")
            x.index = pd.to_datetime(x.index).strftime("%Y-%m-%d")
            x = x.reindex(D.index)
            for mine, theirs in (("mcs_ours", "mcs_ours"), ("mcs_cpp", "mcs_cpp"), (f"hybrid_s{HEADLINE_SEED}", "hybrid")):
                res = _same(x[theirs].to_numpy(), D[mine].to_numpy())
                checks.append({"index": index, "target": target, "seed": HEADLINE_SEED,
                               "check": f"P116 committed daily QLIKE of {theirs} vs this study's {mine}",
                               "rows_job": len(D), **res, "ok": bool(res["n_identical"] == len(D))})
        p = P109_RESULTS / f"daily_losses_{index.lower()}_{target}.csv"
        if p.exists():
            x = pd.read_csv(p, index_col="date", float_precision="round_trip")
            x.index = pd.to_datetime(x.index).strftime("%Y-%m-%d")
            x = x.reindex(D.index)
            res = _same(x["mcsgarch"].to_numpy(), D["mcs_ours"].to_numpy())
            checks.append({"index": index, "target": target, "seed": HEADLINE_SEED,
                           "check": "P109 committed daily QLIKE of mcsGARCH vs this study's mcs_ours (for the combination)",
                           "rows_job": len(D), **res, "ok": bool(res["n_identical"] == len(D))})
            D["combination"] = x["combination"].to_numpy()
    return w, D, checks, both


# ---------------------------------------------------------------------------
# Statistics (pure functions, tested on synthetic data)
# ---------------------------------------------------------------------------


def gain_pct(D: pd.DataFrame, m: str, bench: str) -> float:
    """Model m's gain over a benchmark in % of the benchmark's mean daily QLIKE (positive: m is better)."""
    return 100.0 * (D[bench].mean() - D[m].mean()) / D[bench].mean()


def e_points(D: pd.DataFrame, hyb: str = "hybrid", hcpp: str = "hybrid_cpp") -> float:
    """E: hybrid_cpp's gain over the committed hybrid in points of mcs_cpp's mean daily QLIKE (positive: the hybrid
    built on mcs_cpp is better). Equals gain_pct(hcpp, mcs_cpp) - gain_pct(hyb, mcs_cpp)."""
    return 100.0 * (D[hyb].mean() - D[hcpp].mean()) / D["mcs_cpp"].mean()


def b_points(D: pd.DataFrame) -> float:
    """B: mcs_cpp's gain over mcs_ours in points of mcs_cpp's mean daily QLIKE (the difference of the two bases)."""
    return 100.0 * (D["mcs_ours"].mean() - D["mcs_cpp"].mean()) / D["mcs_cpp"].mean()


def absorbed_share(D: pd.DataFrame, hyb: str = "hybrid", hcpp: str = "hybrid_cpp") -> float:
    """A = 1 - E / B: the share of the difference between the bases that the correction absorbs (1: the two hybrids
    are equally good; 0: the hybrids differ by as much as their bases)."""
    return 1.0 - e_points(D, hyb, hcpp) / b_points(D)


def pts_series(D: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """Daily 100 x (loss of a - loss of b) / mean daily loss of mcs_cpp: its mean is the difference in points."""
    return 100.0 * (D[a] - D[b]).to_numpy(float) / D["mcs_cpp"].mean()


def boot_intervals(D: pd.DataFrame, hyb: str, hcpp: str, reps: int = REPS, rng=None) -> dict:
    """95% stationary-bootstrap percentile intervals of E, B, A and of the gain of hybrid_cpp over mcs_cpp minus the
    gain of the committed hybrid over mcs_ours (days resampled jointly, mean block 10)."""
    rng = rng if rng is not None else np.random.default_rng(BOOT_SEED)
    x = np.column_stack([D["mcs_ours"].to_numpy(), D["mcs_cpp"].to_numpy(), D[hyb].to_numpy(), D[hcpp].to_numpy()])
    idx = rf.stationary_bootstrap_indices(len(x), reps, MEAN_BLOCK, rng)
    mb = rf.bootstrap_means(x, idx)
    o, c, h, hc = mb[:, 0], mb[:, 1], mb[:, 2], mb[:, 3]
    e = 100 * (h - hc) / c
    b = 100 * (o - c) / c
    a = 1 - e / b
    own = 100 * (c - hc) / c - 100 * (o - h) / o

    def q(v):
        return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))

    return {"E": q(e), "B": q(b), "A": q(a), "own_gain_diff": q(own)}


def test_row(D: pd.DataFrame, kind: str, hyb: str = "hybrid", hcpp: str = "hybrid_cpp", delta: float = MARGIN) -> dict:
    """Estimate, statistic and p of one registered row on a day frame.
    h1: hcpp has lower mean QLIKE than mcs_cpp (one-sided DM); estimate: its gain over mcs_cpp, %.
    tost: hcpp and hyb are equivalent within delta points of mcs_cpp's QLIKE (P110's tost); estimate E.
    diff: hcpp and hyb have different mean QLIKE (two-sided DM, positive statistic: hcpp has lower loss); estimate E.
    tost_delta: as tost with P110's delta-method variance of the ratio.
    vs_comb: hcpp has lower mean QLIKE than P109's combination (one-sided); estimate in points of mcs_cpp's QLIKE.
    vs_ours: hcpp has lower mean QLIKE than mcs_ours (one-sided); estimate: its gain over mcs_ours, %."""
    if kind == "h1":
        dm = diebold_mariano(D["mcs_cpp"], D[hcpp], alternative="greater")
        return {"estimate": gain_pct(D, hcpp, "mcs_cpp"), "stat": dm["stat"], "p": dm["p"], "n": dm["n"]}
    if kind == "tost":
        r = st.tost(pts_series(D, hyb, hcpp), delta)
        return {**r, "estimate": e_points(D, hyb, hcpp)}
    if kind == "tost_delta":
        r = st.tost_delta_method((D[hyb] - D[hcpp]).to_numpy(float), D["mcs_cpp"].to_numpy(float), delta)
        return r
    if kind == "diff":
        dm = diebold_mariano(D[hyb], D[hcpp], alternative="two-sided")
        return {"estimate": e_points(D, hyb, hcpp), "stat": dm["stat"], "p": dm["p"], "n": dm["n"]}
    if kind == "vs_comb":
        dm = diebold_mariano(D["combination"], D[hcpp], alternative="greater")
        return {"estimate": 100.0 * (D["combination"].mean() - D[hcpp].mean()) / D["mcs_cpp"].mean(),
                "stat": dm["stat"], "p": dm["p"], "n": dm["n"]}
    if kind == "vs_ours":
        dm = diebold_mariano(D["mcs_ours"], D[hcpp], alternative="greater")
        return {"estimate": gain_pct(D, hcpp, "mcs_ours"), "stat": dm["stat"], "p": dm["p"], "n": dm["n"]}
    raise ValueError(kind)


def seed_estimates(D: pd.DataFrame, seeds) -> dict:
    """Per seed set: hybrid_cpp's gain over mcs_cpp (%), the committed hybrid's gain over mcs_cpp (%), and E."""
    out = {}
    for s in seeds:
        out[s] = {"h1": gain_pct(D, f"hybrid_cpp_s{s}", "mcs_cpp"), "hyb_gain": gain_pct(D, f"hybrid_s{s}", "mcs_cpp"),
                  "E": e_points(D, f"hybrid_s{s}", f"hybrid_cpp_s{s}")}
    return out


def seed_agree(kind: str, est: float, per_seed: dict, delta: float = MARGIN) -> int:
    """How many seed sets point the way the row's alternative does (the registered seed rule, as P110)."""
    if kind == "h1":
        return int(sum(v["h1"] > 0 for v in per_seed.values()))
    if kind == "tost":
        return int(sum(abs(v["E"]) < delta for v in per_seed.values()))
    if kind == "diff":
        return int(sum(np.sign(v["E"]) == np.sign(est) and v["E"] != 0 for v in per_seed.values()))
    raise ValueError(kind)


ROW_PASS = {"h1": "supported", "tost": "equivalent", "tost_delta": "equivalent", "diff": "different"}
ROW_FAIL = {"h1": "not supported", "tost": "not shown", "tost_delta": "not shown", "diff": "no significant difference"}


def row_label(kind: str, p_adj: float, n_agree: int, k: int, alpha: float = ALPHA) -> str:
    """A row passes if its Holm-adjusted p is below alpha and every seed set agrees; "seed-fragile" if the p passes
    but a seed set disagrees (P110's rule)."""
    if not np.isfinite(p_adj):
        return "not computed"
    if p_adj >= alpha:
        return ROW_FAIL[kind]
    return ROW_PASS[kind] if n_agree == k else "seed-fragile"


def horizon_outcome(tost_label: str, diff_label: str, e: float) -> str:
    """The registered outcome at one horizon (plan Section 6), from the H2 (TOST) and S0 (difference) row labels and
    E (positive: the hybrid built on mcs_cpp is better)."""
    if tost_label == "equivalent":
        out = "correction absorbs the base"
        if diff_label == "different":
            better = "mcs_cpp" if e > 0 else "mcs_ours"
            out += f" (a small difference, within the {MARGIN}-point margin, favours the hybrid built on {better})"
        return out
    if diff_label == "different":
        out = "base matters, cpp better" if e > 0 else "base matters, ours better"
        if abs(e) <= MARGIN:
            out += f" (the point estimate is within the {MARGIN}-point margin, but equivalence is not shown)"
        return out
    if "seed-fragile" in (tost_label, diff_label):
        return "inconclusive (a test passed its Holm threshold but not every seed set agreed)"
    return "inconclusive (neither equivalence nor a difference is shown)"


def outcome_name(text: str) -> str:
    return text.split(" (")[0]


def p118_verdict(labels: dict, e: dict, targets=PRIMARY_TARGETS) -> dict:
    """labels[t] = {"h1": label, "tost": label, "diff": label}; e[t] = E at target t."""
    per = {t: horizon_outcome(labels[t]["tost"], labels[t]["diff"], e[t]) for t in targets}
    names = [outcome_name(per[t]) for t in targets]
    if len(set(names)) == 1:
        verdict = names[0]
    else:
        verdict = "mixed: " + "; ".join(f"{outcome_name(per[t])} at {TARGET_LABEL[t]}" for t in targets)
    beats = {t: labels[t]["h1"] == "supported" for t in targets}
    if all(beats.values()):
        h1 = "hybrid_cpp beats mcs_cpp at every primary horizon"
    elif any(beats.values()):
        h1 = "hybrid_cpp beats mcs_cpp at " + ", ".join(TARGET_LABEL[t] for t in targets if beats[t]) + " only"
    else:
        h1 = "hybrid_cpp is not shown to beat mcs_cpp at any primary horizon"
    return {"verdict": verdict, "h1": h1, **{f"outcome_{t}": per[t] for t in targets},
            **{f"h1_{t}": labels[t]["h1"] for t in targets}, **{f"h2_tost_{t}": labels[t]["tost"] for t in targets},
            **{f"s0_diff_{t}": labels[t]["diff"] for t in targets}}


# ---------------------------------------------------------------------------
# Stage 2: analysis
# ---------------------------------------------------------------------------

HYP = {
    "h1": "hybrid_cpp (the hybrid recipe built on mcs_cpp) has lower mean QLIKE than mcs_cpp ({t}{x}). Estimate: its "
          "gain over mcs_cpp, % of mcs_cpp's QLIKE.",
    "tost": "hybrid_cpp and the committed hybrid (built on mcs_ours) are equivalent: their mean QLIKE differ by less "
            "than {d} point(s) of mcs_cpp's QLIKE ({t}{x}). Estimate: E, hybrid_cpp's gain over the committed hybrid, "
            "points (positive: hybrid_cpp better).",
    "tost_delta": "As the TOST row, with the delta-method variance of the ratio ({t}{x}). Estimate: E, points.",
    "diff": "hybrid_cpp and the committed hybrid have different mean QLIKE ({t}{x}). Estimate: E, hybrid_cpp's gain "
            "over the committed hybrid, points of mcs_cpp's QLIKE (positive: hybrid_cpp better).",
    "vs_comb": "hybrid_cpp has lower mean QLIKE than P109's combination ({t}{x}). Estimate: the combination's loss "
               "minus hybrid_cpp's, points of mcs_cpp's QLIKE.",
    "vs_ours": "hybrid_cpp has lower mean QLIKE than mcs_ours ({t}{x}). Estimate: its gain over mcs_ours, % of "
               "mcs_ours's QLIKE.",
}
SIDED = {"h1": "one-sided (hybrid_cpp has lower loss than mcs_cpp)", "tost": "two one-sided tests (equivalence)",
         "tost_delta": "two one-sided tests (equivalence)",
         "diff": "two-sided (d = loss of the committed hybrid minus loss of hybrid_cpp; positive statistic: hybrid_cpp "
                 "has lower loss)", "vs_comb": "one-sided (hybrid_cpp has lower loss than the combination)",
         "vs_ours": "one-sided (hybrid_cpp has lower loss than mcs_ours)"}


def method_of(kind: str, delta: float = MARGIN) -> str:
    if kind == "tost":
        return TOST_METHOD.format(d=delta)
    if kind == "tost_delta":
        return TOST_DELTA.format(d=delta)
    return DM_METHOD


def sample_text(index: str, target: str, D: pd.DataFrame, what: str, seeds: str) -> str:
    d0, d1 = pd.Timestamp(D.index.min()).date(), pd.Timestamp(D.index.max()).date()
    return f"{model_label(index)}, {TARGET_LABEL[target]}, {what}, {len(D)} test days {d0} to {d1}, {seeds}"


def seeds_text(seeds) -> str:
    if len(seeds) == 1:
        return f"seed set {seeds[0]}"
    return "daily QLIKE averaged over seed sets " + ", ".join(str(s) for s in seeds)


def stage_analyze(a):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    synth = a.synthetic
    label = "SYNTHETIC (simulated market, not real data)" if synth else "REAL DATA"
    main = "SYNTH" if synth else "NIFTY"
    others = SYNTH_OTHER if synth else OTHER_INDEXES
    print(f"[{label}] P118 hybrid on the paper's mcsGARCH, stage 2 (docs/prereg/P118_HYBRID_ON_PAPER_MCSGARCH.md)",
          flush=True)
    Rw = gd.Rows()
    repro, loss_rows, seed_rows, bound_rows, anchor_rows, verdict_rows, absorb_rows = ([] for _ in range(7))
    counter = [0]

    def rng():
        counter[0] += 1
        return np.random.default_rng([BOOT_SEED, counter[0]])

    # ---------------- forecasts: reproduction checks (refuse to go on unless every one is identical)
    for index in [main] + list(others):
        for t in targets_of(index):
            repro.extend(forecast_checks(work, index, t, Path(a.p116_ref)))
    # ---------------- daily losses
    data = {}
    for index in [main] + list(others):
        for t in targets_of(index):
            r = build_frames(work, index, t)
            if r is None:
                print(f"  {index} {t}: work files missing; not computed", flush=True)
                continue
            w, D, checks, seeds = r
            repro.extend(checks)
            data[(index, t)] = (w, D, seeds)
    R = pd.DataFrame(repro)
    if R.empty or not R["ok"].all():
        bad = R[~R["ok"]] if not R.empty else R
        print(bad.to_string(), flush=True)
        if not a.allow_mismatch:
            raise SystemExit("reproduction check failed; rerun with --allow-mismatch to report anyway")
    if not all((main, t) in data for t in PRIMARY_TARGETS):
        raise SystemExit("no finished work files for the primary targets")

    def add(tid, kind, D, index, t, what, role, family, hyb="hybrid", hcpp="hybrid_cpp", seeds=(HEADLINE_SEED,),
            per_seed=None, delta=MARGIN, hyp_extra=""):
        r = test_row(D, kind, hyb, hcpp, delta)
        k = len(per_seed) if per_seed else 0
        n_agree = seed_agree(kind, r["estimate"], per_seed, delta) if per_seed and kind in ("h1", "tost", "diff") else np.nan
        Rw.add(tid, HYP[kind].format(t=TARGET_LABEL[t], x=hyp_extra, d=delta), role,
               sample_text(index, t, D, what, seeds_text(seeds)), r["estimate"], r["stat"], r["p"], SIDED[kind],
               method_of(kind, delta), family, index=index, target=t, subset=what, kind=kind, hyb=hyb, hcpp=hcpp,
               seeds="+".join(str(s) for s in seeds), days=len(D), margin=delta if kind.startswith("tost") else np.nan,
               se=r.get("se", np.nan), ci90_low=r.get("ci90_low", np.nan), ci90_high=r.get("ci90_high", np.nan),
               bound=r.get("bound", np.nan), seed_sets=k, seed_sets_agreeing=n_agree)
        if kind.startswith("tost"):
            bound_rows.append({"test_id": tid, "index": index, "target": t, "subset": what, "seeds": seeds_text(seeds),
                               "estimate_points": r["estimate"], "se": r["se"], "ci90_low": r["ci90_low"],
                               "ci90_high": r["ci90_high"], "smallest_passing_margin": r["bound"], "p_raw": r["p"],
                               "margin": delta})
        return r

    def trio(prefix, D, index, t, what, role, family, seeds, per_seed):
        for kind in ("h1", "tost", "diff"):
            add(f"{prefix}_{kind}", kind, D, index, t, what, role, family, seeds=seeds, per_seed=per_seed)

    def losses(index, t, D, subset, seeds):
        models = [m for m in ["mcs_ours", "mcs_cpp", "hybrid", "hybrid_cpp", "hybrid_cpp_all", "combination"]
                  + [f"hybrid_s{s}" for s in seeds] + [f"hybrid_cpp_s{s}" for s in seeds] if m in D]
        for m in models:
            loss_rows.append({"index": index, "target": t, "subset": subset, "model": m, "days": len(D),
                              "seeds_in_average": "+".join(str(s) for s in seeds) if m in ("hybrid", "hybrid_cpp") else "",
                              "qlike_mean": D[m].mean(), "gain_pct_vs_mcs_cpp": gain_pct(D, m, "mcs_cpp"),
                              "gain_pct_vs_mcs_ours": gain_pct(D, m, "mcs_ours"),
                              "points_vs_hybrid_of_mcs_cpp": 100.0 * (D["hybrid"].mean() - D[m].mean()) / D["mcs_cpp"].mean()})

    def absorb(index, t, D, subset, hyb, hcpp, seeds):
        ci = boot_intervals(D, hyb, hcpp, a.reps, rng())
        own = gain_pct(D, hcpp, "mcs_cpp") - gain_pct(D, hyb, "mcs_ours")
        absorb_rows.append({"index": index, "target": t, "subset": subset, "seeds": seeds_text(seeds), "days": len(D),
                            "hybrid_cpp_gain_vs_mcs_cpp_pct": gain_pct(D, hcpp, "mcs_cpp"),
                            "hybrid_gain_vs_mcs_cpp_pct": gain_pct(D, hyb, "mcs_cpp"),
                            "hybrid_gain_vs_mcs_ours_pct": gain_pct(D, hyb, "mcs_ours"),
                            "hybrid_cpp_gain_vs_mcs_ours_pct": gain_pct(D, hcpp, "mcs_ours"),
                            "E_points": e_points(D, hyb, hcpp), "E_ci_low": ci["E"][0], "E_ci_high": ci["E"][1],
                            "B_points": b_points(D), "B_ci_low": ci["B"][0], "B_ci_high": ci["B"][1],
                            "A_absorbed_share": absorbed_share(D, hyb, hcpp), "A_ci_low": ci["A"][0],
                            "A_ci_high": ci["A"][1], "own_base_gain_diff_points": own,
                            "own_base_gain_diff_ci_low": ci["own_gain_diff"][0],
                            "own_base_gain_diff_ci_high": ci["own_gain_diff"][1]})

    # ---------------- Nifty (or the main synthetic market)
    per_seed_main = {}
    for t in targets_of(main):
        if (main, t) not in data:
            continue
        w, D, seeds = data[(main, t)]
        exp_mask = D["is_expiry"].to_numpy(bool)
        losses(main, t, D, "all days", seeds)
        losses(main, t, D[exp_mask], "expiry days", seeds)
        losses(main, t, D[~exp_mask], "non-expiry days", seeds)
        for k, name in enumerate(REGIMES):
            m_ = D["regime"].to_numpy() == k
            if m_.any():
                losses(main, t, D[m_], f"regime {name}", seeds)
        ps = seed_estimates(D, seeds)
        for s, v in ps.items():
            seed_rows.append({"index": main, "target": t, "subset": "all days", "seed": s,
                              "hybrid_cpp_gain_vs_mcs_cpp_pct": v["h1"], "hybrid_gain_vs_mcs_cpp_pct": v["hyb_gain"],
                              "E_points": v["E"]})
        if t in PRIMARY_TARGETS:
            per_seed_main[t] = ps
            add(f"P_{t}_h1", "h1", D, main, t, "all days", "primary", "primary", seeds=seeds, per_seed=ps)
            add(f"P_{t}_tost", "tost", D, main, t, "all days", "primary", "primary", seeds=seeds, per_seed=ps)
            add(f"S0_{t}_diff", "diff", D, main, t, "all days", "secondary", "S0_difference", seeds=seeds, per_seed=ps)
            absorb(main, t, D, "all days", "hybrid", "hybrid_cpp", seeds)
            absorb(main, t, D, "all days", f"hybrid_s{HEADLINE_SEED}", f"hybrid_cpp_s{HEADLINE_SEED}", [HEADLINE_SEED])
            # S2 expiry days, E2 non-expiry days (seed-averaged, per-seed agreement within the subset)
            for sub, mask, fam, role, pre in (("expiry days", exp_mask, "S2_expiry_days", "secondary", "S2"),
                                              ("non-expiry days", ~exp_mask, "E2_nonexpiry", "exploratory", "E2")):
                Ds = D[mask]
                pss = seed_estimates(Ds, seeds)
                trio(f"{pre}_{t}_{'expiry' if pre == 'S2' else 'nonexpiry'}", Ds, main, t, sub, role, fam, seeds, pss)
                absorb(main, t, Ds, sub, "hybrid", "hybrid_cpp", seeds)
            if "combination" in D and D["combination"].notna().all():
                add(f"S3_{t}_vs_combination", "vs_comb", D, main, t, "all days", "secondary", "S3_combination",
                    seeds=seeds)
            # R1: seed set 7 alone; R2: delta-method TOST on the primary sample
            h7, c7 = f"hybrid_s{HEADLINE_SEED}", f"hybrid_cpp_s{HEADLINE_SEED}"
            for kind in ("h1", "tost", "diff"):
                add(f"R1_{t}_seed7_{kind}", kind, D, main, t, "all days", "robustness", "R1_seed7", hyb=h7, hcpp=c7)
            add(f"R2_{t}_tost_delta", "tost_delta", D, main, t, "all days", "robustness", "R2_delta_method",
                seeds=seeds)
            # PL1 (placebo): hybrid_cpp with each repeat seed set against hybrid_cpp with seed set 7
            for s in seeds[1:]:
                Dp = D.copy()
                Dp["_a"], Dp["_b"] = Dp[f"hybrid_cpp_s{HEADLINE_SEED}"], Dp[f"hybrid_cpp_s{s}"]
                r = st.tost(pts_series(Dp, "_a", "_b"), MARGIN)
                Rw.add(f"PL1_{t}_s{s}", f"Placebo: hybrid_cpp with seed set {s} and with seed set 7 are equivalent "
                       f"within {MARGIN} point of mcs_cpp's QLIKE ({TARGET_LABEL[t]}). Only the seeds differ. Estimate: "
                       f"the seed-{s} model's gain over the seed-7 model, points.", "placebo",
                       sample_text(main, t, D, "all days", f"seed sets 7 and {s}"), r["estimate"], r["stat"], r["p"],
                       SIDED["tost"], TOST_METHOD.format(d=MARGIN), "PL1_seeds", index=main, target=t,
                       subset="all days", kind="tost", hyb=f"hybrid_cpp_s{HEADLINE_SEED}", hcpp=f"hybrid_cpp_s{s}",
                       seeds=f"7+{s}", days=len(D), margin=MARGIN, se=r["se"], ci90_low=r["ci90_low"],
                       ci90_high=r["ci90_high"], bound=r["bound"], seed_sets=2, seed_sets_agreeing=np.nan)
            # PL0 (placebo, known before the plan was committed): the committed hybrid, seed set s vs 7
            for s in seeds[1:]:
                Dp = D.copy()
                Dp["_a"], Dp["_b"] = Dp[f"hybrid_s{HEADLINE_SEED}"], Dp[f"hybrid_s{s}"]
                r = st.tost(pts_series(Dp, "_a", "_b"), MARGIN)
                Rw.add(f"PL0_{t}_s{s}", f"Placebo, computed during planning on committed losses (known before the plan "
                       f"commit): the committed hybrid with seed set {s} and with seed set 7 are equivalent within "
                       f"{MARGIN} point of mcs_cpp's QLIKE ({TARGET_LABEL[t]}). Estimate: points.", "placebo",
                       sample_text(main, t, D, "all days", f"seed sets 7 and {s}"), r["estimate"], r["stat"], r["p"],
                       SIDED["tost"], TOST_METHOD.format(d=MARGIN), "PL0_committed_seeds", index=main, target=t,
                       subset="all days", kind="tost", hyb=f"hybrid_s{HEADLINE_SEED}", hcpp=f"hybrid_s{s}",
                       seeds=f"7+{s}", days=len(D), margin=MARGIN, se=r["se"], ci90_low=r["ci90_low"],
                       ci90_high=r["ci90_high"], bound=r["bound"], seed_sets=2, seed_sets_agreeing=np.nan)
            # E3: the 09:15 anchor and the other anchors (seed set 7; anchor-level forecasts)
            for sname, mins, key in (("09:15 anchor only", [0], "0915"),
                                     ("without the 09:15 anchor", sorted(set(w["minute"]) - {0}), "rest")):
                dd = ex.day_frame(w, ["mcs_ours", "mcs_cpp", h7, c7], minutes=mins)
                add(f"E3_{t}_{key}_diff", "diff", dd, main, t, sname, "exploratory", "E3_anchor", hyb=h7, hcpp=c7)
            add(f"E4_{t}_vs_ours", "vs_ours", D, main, t, "all days", "exploratory", "E4_vs_ours", seeds=seeds)
        else:   # last hour: seed set 7 (the committed hybrid has no repeat seed sets there)
            h7, c7 = f"hybrid_s{HEADLINE_SEED}", f"hybrid_cpp_s{HEADLINE_SEED}"
            for kind in ("h1", "tost", "diff"):
                add(f"S1_{t}_{kind}", kind, D, main, t, "all days", "secondary", "S1_last_hour", hyb=h7, hcpp=c7)
            add(f"E4_{t}_vs_ours", "vs_ours", D, main, t, "all days", "exploratory", "E4_vs_ours", hcpp=c7)
            absorb(main, t, D, "all days", h7, c7, [HEADLINE_SEED])
        # E1: hybrid_cpp_all (every mcsGARCH input as the paper specifies it), seed set 7
        if "hybrid_cpp_all" in D:
            h7, c7 = f"hybrid_s{HEADLINE_SEED}", f"hybrid_cpp_s{HEADLINE_SEED}"
            add(f"E1_{t}_all_h1", "h1", D, main, t, "all days", "exploratory", "E1_cpp_all", hcpp="hybrid_cpp_all",
                hyp_extra="; hybrid_cpp_all: every mcsGARCH input as the paper specifies it")
            add(f"E1_{t}_all_vs_hybrid_cpp", "diff", D, main, t, "all days", "exploratory", "E1_cpp_all", hyb=c7,
                hcpp="hybrid_cpp_all", hyp_extra="; here hybrid_cpp_all against hybrid_cpp (seed set 7)")
            add(f"E1_{t}_all_vs_hybrid", "diff", D, main, t, "all days", "exploratory", "E1_cpp_all", hyb=h7,
                hcpp="hybrid_cpp_all", hyp_extra="; here hybrid_cpp_all against the committed hybrid (seed set 7)")
        # gain by anchor (descriptive, seed set 7)
        for mnt in sorted(set(w["minute"])):
            cols = ["mcs_ours", "mcs_cpp", f"hybrid_s{HEADLINE_SEED}", f"hybrid_cpp_s{HEADLINE_SEED}"] + \
                (["hybrid_cpp_all"] if "hybrid_cpp_all" in w else [])
            dd = ex.day_frame(w, cols, minutes=[mnt])
            anchor_rows.append({"index": main, "target": t, "minute": mnt,
                                "clock": (pd.Timestamp("2000-01-01 09:15") + pd.Timedelta(minutes=int(mnt))).strftime("%H:%M"),
                                "days": len(dd), **{f"{c}_qlike_mean": dd[c].mean() for c in cols},
                                **{f"{c}_gain_pct_vs_mcs_cpp": gain_pct(dd, c, "mcs_cpp") for c in cols if c != "mcs_cpp"},
                                "hybrid_s7_gain_pct_vs_mcs_ours": gain_pct(dd, f"hybrid_s{HEADLINE_SEED}", "mcs_ours"),
                                "E_points_seed7": e_points(dd, f"hybrid_s{HEADLINE_SEED}", f"hybrid_cpp_s{HEADLINE_SEED}")})

    # ---------------- other indices (S4 Bank Nifty, S5 Fin Nifty)
    per_seed_other = {}
    for index, fam in others.items():
        for t in PRIMARY_TARGETS:
            if (index, t) not in data:
                continue
            w, D, seeds = data[(index, t)]
            exp_mask = D["is_expiry"].to_numpy(bool)
            losses(index, t, D, "all days", seeds)
            losses(index, t, D[exp_mask], "expiry days", seeds)
            ps = seed_estimates(D, seeds)
            per_seed_other[(index, t)] = ps
            for s, v in ps.items():
                seed_rows.append({"index": index, "target": t, "subset": "all days", "seed": s,
                                  "hybrid_cpp_gain_vs_mcs_cpp_pct": v["h1"], "hybrid_gain_vs_mcs_cpp_pct": v["hyb_gain"],
                                  "E_points": v["E"]})
            trio(f"{fam.split('_')[0]}_{t}", D, index, t, "all days", "secondary", fam, seeds, ps)
            absorb(index, t, D, "all days", "hybrid", "hybrid_cpp", seeds)
            absorb(index, t, D, "all days", f"hybrid_s{HEADLINE_SEED}", f"hybrid_cpp_s{HEADLINE_SEED}", [HEADLINE_SEED])

    # ---------------- adjust, verdict, write
    families = {"primary": "the four primary rows (H1 and H2 at 15 and 30 minutes, Nifty, all days)",
                "S0_difference": "secondary S0 (the difference test of the decision rule)",
                "S1_last_hour": "secondary S1 (last hour)", "S2_expiry_days": "secondary S2 (expiry days)",
                "S3_combination": "secondary S3 (hybrid_cpp vs P109's combination)",
                "S4_banknifty": "secondary S4 (Bank Nifty)", "S5_finnifty": "secondary S5 (Fin Nifty)",
                "R1_seed7": "robustness R1 (seed set 7 alone)", "R2_delta_method": "robustness R2 (delta method)",
                "PL1_seeds": "placebo PL1 (hybrid_cpp, seed sets)",
                "PL0_committed_seeds": "placebo PL0 (the committed hybrid, seed sets; computed during planning)",
                "E1_cpp_all": None, "E2_nonexpiry": None, "E3_anchor": None, "E4_vs_ours": None}
    tests = Rw.frame(families)
    detail = pd.DataFrame(Rw.detail)
    tests["_label"] = ""
    dmap = detail.set_index("test_id")
    for i, row in tests.iterrows():
        dr = dmap.loc[row["test_id"]]
        if dr["kind"] in ROW_PASS:
            k = int(dr["seed_sets"]) if dr["seed_sets"] else 1
            n_ag = dr["seed_sets_agreeing"]
            n_ag = k if not np.isfinite(n_ag) else int(n_ag)
            tests.at[i, "_label"] = row_label(dr["kind"], float(row["p_adjusted"]), n_ag, k)
    detail = detail.merge(tests[["test_id", "p_adjusted", "_label"]].rename(columns={"_label": "row_verdict"}),
                          on="test_id")
    lab = tests.set_index("test_id")["_label"]
    padj = tests.set_index("test_id")["p_adjusted"]

    def verdict_for(index, pre_h1, pre_tost, pre_diff, role, seeds_by_t):
        labels, es = {}, {}
        for t in PRIMARY_TARGETS:
            labels[t] = {"h1": lab[pre_h1.format(t=t)], "tost": lab[pre_tost.format(t=t)],
                         "diff": lab[pre_diff.format(t=t)]}
        rows = {"study": "P118", "index": index, "role": role}
        for t in PRIMARY_TARGETS:
            D = data[(index, t)][1]
            es[t] = e_points(D)
        v = p118_verdict(labels, es)
        rows.update(v)
        for t in PRIMARY_TARGETS:
            D = data[(index, t)][1]
            ab = [r for r in absorb_rows if r["index"] == index and r["target"] == t and r["subset"] == "all days"
                  and r["seeds"] == seeds_text(seeds_by_t[t])][0]
            rows.update({f"seeds_{t}": seeds_text(seeds_by_t[t]),
                         f"hybrid_cpp_gain_vs_mcs_cpp_pct_{t}": gain_pct(D, "hybrid_cpp", "mcs_cpp"),
                         f"hybrid_gain_vs_mcs_cpp_pct_{t}": gain_pct(D, "hybrid", "mcs_cpp"),
                         f"hybrid_gain_vs_mcs_ours_pct_{t}": gain_pct(D, "hybrid", "mcs_ours"),
                         f"E_points_{t}": es[t], f"E_ci_low_{t}": ab["E_ci_low"], f"E_ci_high_{t}": ab["E_ci_high"],
                         f"B_points_{t}": b_points(D), f"A_absorbed_share_{t}": ab["A_absorbed_share"],
                         f"A_ci_low_{t}": ab["A_ci_low"], f"A_ci_high_{t}": ab["A_ci_high"],
                         f"p_h1_holm_{t}": float(padj[pre_h1.format(t=t)]),
                         f"p_tost_holm_{t}": float(padj[pre_tost.format(t=t)]),
                         f"p_diff_holm_{t}": float(padj[pre_diff.format(t=t)])})
        return rows

    verdict_rows.append(verdict_for(main, "P_{t}_h1", "P_{t}_tost", "S0_{t}_diff", "primary (decision rule)",
                                    {t: data[(main, t)][2] for t in PRIMARY_TARGETS}))
    for index, fam in others.items():
        if all((index, t) in data for t in PRIMARY_TARGETS):
            pre = fam.split("_")[0]
            verdict_rows.append(verdict_for(index, pre + "_{t}_h1", pre + "_{t}_tost", pre + "_{t}_diff",
                                            "secondary (same rule, applied descriptively)",
                                            {t: data[(index, t)][2] for t in PRIMARY_TARGETS}))

    tests_out = tests.drop(columns=["_family", "_label"])
    tests_out.to_csv(out / "tests.csv", index=False)
    detail.to_csv(out / "tests_detail.csv", index=False)
    pd.DataFrame(verdict_rows).to_csv(out / "verdict.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(out / "losses.csv", index=False)
    pd.DataFrame(absorb_rows).to_csv(out / "absorption.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "seed_repeats.csv", index=False)
    pd.DataFrame(bound_rows).to_csv(out / "equivalence_bounds.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(out / "gain_by_anchor.csv", index=False)
    R.to_csv(out / "reproduction_check.csv", index=False)
    for (index, t), (_, D, _) in data.items():
        D.to_csv(out / f"daily_losses_{index.lower()}_{t}.csv")
    tim = work / "timing.csv"
    if tim.exists():
        pd.read_csv(tim).to_csv(out / "timing.csv", index=False)
    pf = sorted(work.glob("*_s*_*_params.csv"))
    if pf:
        P = []
        for f in pf:
            x = pd.read_csv(f)
            x.insert(0, "job", f.stem[: -len("_params")])
            P.append(x)
        pd.concat(P, ignore_index=True).to_csv(out / "params_by_job.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"\n[{label}] verdict:")
    print(pd.DataFrame(verdict_rows).T.to_string())
    print(f"\n[{label}] tests (primary, secondary, robustness, placebo):")
    show = tests[tests["role"].isin(["primary", "secondary", "robustness", "placebo"])]
    print(show[["test_id", "estimate", "statistic", "p_raw", "p_adjusted", "_label"]].to_string(index=False))
    print(f"\n[{label}] absorption (E, B, A):")
    print(pd.DataFrame(absorb_rows).to_string(index=False))
    print(f"\n[{label}] reproduction checks: {len(R)} rows, all identical: {bool(R['ok'].all())}")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    import lightgbm
    import scipy
    inputs = {}
    for ix in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
        p = work / f"inputs_{ix}.json"
        if p.exists():
            inputs[ix] = json.loads(p.read_text())
    wall = {}
    if tim.exists():
        tt = pd.read_csv(tim)
        for ix, g in tt.groupby("index"):
            wall[ix] = {"jobs": int(len(g)), "seconds_jobs_total": float(g["seconds"].sum()),
                        "first_job_started_utc": str(g["start_utc"].min()), "last_job_ended_utc": str(g["end_utc"].max()),
                        "by_kind_seconds": {k: float(v) for k, v in g.groupby("kind")["seconds"].sum().items()}}
    json.dump({"source": label + ("" if synth else ": NSE 1-minute index bars as in the committed runs; the committed "
                                  "hybrid's forecasts (seed set 7) and daily losses (seed sets 101 to 104, P106 and "
                                  "P110) are the committed ones; mcs_cpp is refitted inside every job and checked "
                                  "against P116; P109's combination enters through its committed daily losses"),
               "study": "P118 hybrid on the paper's mcsGARCH (docs/prereg/P118_HYBRID_ON_PAPER_MCSGARCH.md)",
               "plan_commit": a.plan_commit, "analysis_commit": commit,
               "cpp_spec": CPP_SPEC, "kind_models": KIND_MODELS, "kind_settings": KIND_SETTINGS,
               "pipeline_settings": gd.PIPELINE_SETTINGS, "seeds": ALL_SEEDS if not synth else SYNTH["seeds"],
               "margin_points": MARGIN, "alpha": ALPHA,
               "seeds_used": {f"{ix}_{t}": v[2] for (ix, t), v in data.items()},
               "synthetic_settings": SYNTH if synth else None,
               "bootstrap": {"reps": a.reps, "mean_block_days": MEAN_BLOCK, "seed": BOOT_SEED},
               "input_sha256": inputs, "stage1": wall, "seconds_stage2": round(time.time() - t0, 1),
               "python": sys.version.split()[0],
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
                            "scipy": scipy.__version__}},
              open(out / "run_info.json", "w"), indent=2, default=str)
    print(f"\nwritten to {out} ({time.time() - t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("run", "check", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--work", default=None)
        p.add_argument("--synthetic", action="store_true", help="simulated markets and small settings (dry run)")
        p.add_argument("--p116-ref", default=str(P116_REF))
        if name == "run":
            p.add_argument("--index", choices=list(INDEXES) + ["SYNTH", "SYNTHB"], default="NIFTY")
            p.add_argument("--phase", choices=["repro", "study"], default="study")
            p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
            p.add_argument("--seeds", nargs="+", type=int, default=None)
            p.add_argument("--targets", nargs="+", default=None)
            p.add_argument("--kinds", nargs="+", default=None)
            p.add_argument("--workers", type=int, default=2)
        if name == "check":
            p.add_argument("--indexes", nargs="+", default=["NIFTY"])
        if name == "analyze":
            p.add_argument("--out", default=None)
            p.add_argument("--reps", type=int, default=REPS)
            p.add_argument("--plan-commit", default=None)
            p.add_argument("--allow-mismatch", action="store_true",
                           help="report even if a reproduction check fails")
    a = ap.parse_args(argv)
    if a.work is None:
        a.work = str(ROOT / "data" / "processed" / ("p118_synthetic" if a.synthetic else "p118"))
    if a.stage == "run" and a.synthetic and not is_synth(a.index):
        a.index = "SYNTH"
    if a.stage == "analyze" and a.out is None:
        a.out = str(ROOT / "data" / "processed" / "p118_synthetic_results") if a.synthetic else \
            str(ROOT / "results" / "p118_hybrid_on_paper_mcsgarch")
    if a.stage == "run":
        stage_run(a)
    elif a.stage == "check":
        return stage_check(a)
    else:
        stage_analyze(a)


if __name__ == "__main__":
    main()
