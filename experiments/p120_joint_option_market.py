"""P120: is the hybrid's gain free of option-market information when the option-implied and VIX inputs are removed
together?

Registered plan: docs/prereg/P120_JOINT_OPTION_MARKET.md. Results: results/p120_joint_option_market/.
Write-up: docs/P120_JOINT_OPTION_MARKET.md. Done by an AI agent (Claude) for Shreyan Thanki.

P106 removed one input group at a time and, on Nifty only, the option-implied and VIX groups together (the
``optionmkt`` drop set), testing only whether the removal raises the loss. P110 ran equivalence tests per group on
Bank Nifty and Fin Nifty but never ``optionmkt``. This script

1. fits ``optionmkt`` on Bank Nifty and Fin Nifty at 15 and 30 minutes with seed sets 7 and 101 to 104 (P106's code
   path, nothing retuned; the full hybrid is P110's, not refitted), the full hybrid and ``optionmkt`` for the last
   hour (Bank Nifty and Fin Nifty with all five seed sets, Nifty with 101 to 104), and one Nifty ``optionmkt`` job
   (15 minutes, seed set 7) as a reproduction check of P106's work file;
2. tests, on each index and horizon, whether the hybrid without the two groups is within 1 point of mcsGARCH's QLIKE
   of the full hybrid (two one-sided tests, as P110), with the registered companion, secondary, robustness and
   exploratory rows. Nifty at 15 and 30 minutes uses P106's committed daily losses.

Stages:

    # 1. forecasts (REAL DATA), one invocation per index, at most 2 worker processes
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index BANKNIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index FINNIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index NIFTY --workers 2
    # 2. losses, tests, tables, verdicts
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py analyze --plan-commit <hash>

On real data both stages refuse to run unless the registered plan is in the git history (and ``analyze`` unless
--plan-commit names a commit that contains the plan and is an ancestor of the checked-out commit).
``--synthetic`` (run and analyze) uses three simulated markets and small settings, for a dry run of every step; its
output is labelled SYNTHETIC. Work files go to --work (default data/processed/p120, git-ignored); a finished job is
not rerun. Tests: tests/test_p120_joint_option_market.py (synthetic data only).
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
from scipy.stats import t as student_t  # noqa: E402

from niftygex.evaluation import diebold_mariano  # noqa: E402
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel  # noqa: E402
from niftygex.implied import build_implied_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    """Load a script as a module under its own name, registered in sys.modules (as P116 and P118 do), so that the
    worker pool can pickle its functions."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# P106: drop sets, the committed runs' configuration (make_config), the job function (run_job), work files, settings
# check, input hashes, day frames, the share interval, the tests.csv collector (Rows) and the reproduction checks.
gd = _load("p120_copy_of_p106_gain_decomposition", ROOT / "experiments" / "p106_gain_decomposition.py")
# P110: the DM pieces, TOST, the delta-method TOST, the points series, the pooled test (P107's Driscoll-Kraay
# covariance) and the June and top-loss masks.
st = _load("p120_copy_of_p110_gain_stress_test", ROOT / "experiments" / "p110_gain_stress_test.py")

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

PLAN = "docs/prereg/P120_JOINT_OPTION_MARKET.md"
STUDY_INDEXES = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
NEW_INDEXES = ["BANKNIFTY", "FINNIFTY"]       # optionmkt never run before
TARGETS = ["next15", "next30"]
LAST = "last_hour"
VARIANT = "optionmkt"                          # P106's DROP_SETS["optionmkt"] = implied + vix groups
HEADLINE_SEED = gd.HEADLINE_SEED               # 7
NEW_SEEDS = list(gd.REPEAT_SEEDS)              # 101, 102, 103, 104
SEEDS = [HEADLINE_SEED] + NEW_SEEDS
MARGIN = st.MARGIN                             # 1 point of mcsGARCH's QLIKE
MARGIN_SENS = st.MARGIN_SENS                   # 0.5 point
ALPHA = 0.05
REPS = 10000
MEAN_BLOCK = 10.0
BOOT_SEED = 120
P106_RESULTS = ROOT / "results" / "p106_gain_decomposition"
P110_RESULTS = ROOT / "results" / "p110_gain_stress_test"
P99_DIR = ROOT / "results" / "seed_bootstrap"
LABEL_REAL = ("REAL DATA: NSE bhavcopy, participant OI and India VIX; third-party 1-minute index bars validated "
              "against NSE daily values")
LABEL_SYNTH = "SYNTHETIC (simulated market, not real data)"
LABEL = {"NIFTY": "Nifty", "BANKNIFTY": "Bank Nifty", "FINNIFTY": "Fin Nifty",
         "SYNTHA": "synthetic A (stands in for Bank Nifty)", "SYNTHB": "synthetic B (stands in for Fin Nifty)",
         "SYNTHC": "synthetic C (stands in for Nifty)"}
TARGET_LABEL = {"next15": "next 15 min", "next30": "next 30 min", "last_hour": "last hour"}
SYNTH = dict(n_days=200, start="2024-03-01", market_seeds={"SYNTHA": 3, "SYNTHB": 4, "SYNTHC": 5}, min_train=60,
             block=20, n_seeds=1, seeds=[7, 101], horizon_step=45)
SYNTH_INDEXES = ["SYNTHC", "SYNTHA", "SYNTHB"]   # in the roles of Nifty, Bank Nifty, Fin Nifty
SYNTH_VARIANTS = ("full", "implied", "vix", VARIANT)


def is_synth(index: str) -> bool:
    return index.startswith("SYNTH")


# ---------------------------------------------------------------------------
# The plan must be in the git history
# ---------------------------------------------------------------------------


def _git(args, root=ROOT):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def plan_in_history(root: Path = ROOT, plan: str = PLAN) -> str | None:
    """The last commit that touched the plan file, or None if the plan is not in the git history."""
    try:
        r = _git(["log", "-1", "--format=%H", "--", plan], root)
    except OSError:
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def check_plan_commit(commit: str | None, root: Path = ROOT, plan: str = PLAN) -> str:
    """Refuse (SystemExit) unless ``commit`` contains the plan file and is an ancestor of the checked-out commit.
    Returns the full hash."""
    if not commit:
        raise SystemExit("--plan-commit is required on real data (the commit that added the registered plan)")
    r = _git(["rev-parse", "--verify", f"{commit}^{{commit}}"], root)
    if r.returncode != 0:
        raise SystemExit(f"--plan-commit {commit} is not a commit of this repository")
    full = r.stdout.strip()
    if _git(["cat-file", "-e", f"{full}:{plan}"], root).returncode != 0:
        raise SystemExit(f"--plan-commit {commit} does not contain {plan}")
    if _git(["merge-base", "--is-ancestor", full, "HEAD"], root).returncode != 0:
        raise SystemExit(f"--plan-commit {commit} is not an ancestor of the checked-out commit")
    return full


def require_plan(root: Path = ROOT, plan: str = PLAN) -> str:
    c = plan_in_history(root, plan)
    if c is None:
        raise SystemExit(f"the registered plan {plan} is not in the git history: commit and push it before any "
                         "real-data run")
    return c


# ---------------------------------------------------------------------------
# Job plan and stage 1 (forecasts)
# ---------------------------------------------------------------------------


def job_plan(index: str, seeds=None, targets=None) -> list[tuple]:
    """(index, target, seed, variants), in the registered order: the 15- and 30-minute jobs (seed set 7, then 101 to
    104), then the last-hour jobs. Bank Nifty and Fin Nifty: ``optionmkt`` at 15 and 30 minutes (the full hybrid is
    P110's) and the full hybrid and ``optionmkt`` for the last hour. Nifty: one reproduction job (15 minutes, seed
    set 7) and the last hour with seed sets 101 to 104. Synthetic markets have no committed runs, so there every
    variant the analysis needs is fitted. Optional filters keep the order."""
    plan = []
    if is_synth(index):
        for s in SYNTH["seeds"]:
            plan += [(index, t, s, SYNTH_VARIANTS) for t in TARGETS]
        plan += [(index, LAST, s, ("full", VARIANT)) for s in SYNTH["seeds"]]
    elif index == "NIFTY":
        plan.append((index, "next15", HEADLINE_SEED, (VARIANT,)))
        plan += [(index, LAST, s, ("full", VARIANT)) for s in NEW_SEEDS]
    elif index in NEW_INDEXES:
        for s in SEEDS:
            plan += [(index, t, s, (VARIANT,)) for t in TARGETS]
        plan += [(index, LAST, s, ("full", VARIANT)) for s in SEEDS]
    else:
        raise ValueError(index)
    return [j for j in plan if (seeds is None or j[2] in seeds) and (targets is None or j[1] in targets)]


def synthetic_market(index: str):
    from niftygex.simulate import SimConfig, simulate_market
    return simulate_market(SimConfig(n_days=SYNTH["n_days"], start=SYNTH["start"],
                                     seed=SYNTH["market_seeds"][index], kappa=0.0))


def synthetic_config(seed, eval_minutes, sets, variants):
    return gd.make_config(seed, eval_minutes, sets, variants, min_train=SYNTH["min_train"], block=SYNTH["block"],
                          mcs_refit_every=SYNTH["block"], n_seeds=SYNTH["n_seeds"])


def synthetic_targets() -> dict:
    h = SYNTH["horizon_step"]
    return {"next15": dict(horizon=15, step=h), "next30": dict(horizon=30, step=h), LAST: dict(horizon=60, step=h)}


def stage_run(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    index = a.index
    synth = is_synth(index)
    if not synth:
        require_plan()
    plan = job_plan(index, seeds=a.seeds, targets=a.targets)
    jobs = [j for j in plan if not gd.job_done(work, j)]
    label = LABEL_SYNTH if synth else "REAL DATA"
    print(f"[{label}] P120 joint option market, stage 1, {index}: {len(jobs)} jobs to run "
          f"({len(plan) - len(jobs)} already in {work}); workers {a.workers}", flush=True)
    if not jobs:
        return
    t0 = time.time()
    if synth:
        m = synthetic_market(index)
        gd._STATE["config_fn"] = synthetic_config
        targets = synthetic_targets()
    else:
        gd._STATE.pop("config_fn", None)
        info = json.load(open(gd.INDEXES[index]["run"] / "run_info.json"))
        ns = argparse.Namespace(source="nse", raw=a.raw, symbol=index, oi_unit=gd.PIPELINE_SETTINGS["oi_unit"],
                                lot_size=None, rate=gd.PIPELINE_SETTINGS["rate"], start=gd.PIPELINE_SETTINGS["start"],
                                end=gd.INDEXES[index]["end"], min_coverage=gd.PIPELINE_SETTINGS["min_coverage"])
        m = gd.rp.load_market(ns)
        last = str(pd.Timestamp(m["days"][-1]).date())
        bad = gd.check_run_settings(index, info, last)
        if bad or len(m["days"]) != info["days"][2]:
            raise SystemExit(f"settings differ from the committed run: {bad}, days {len(m['days'])} vs {info['days'][2]}")
        targets = gd.rp.TARGETS
        hp = work / f"inputs_{index}.json"
        if not hp.exists():
            hp.write_text(json.dumps(gd.input_hashes(Path(a.raw), index), indent=2))
    print(f"market loaded: {len(m['days'])} days, {m['days'][0]} to {m['days'][-1]} ({time.time() - t0:.0f}s)", flush=True)
    panel = build_gex_panel(m)
    implied = build_implied_panel(m)
    has_oracle = "dealer_next" in m
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle" or has_oracle}
    need = sorted({j[1] for j in jobs})
    datasets = {t: build_dataset(m, panel, implied=implied, **targets[t]) for t in need}
    print(f"panels and datasets built ({time.time() - t0:.0f}s)", flush=True)
    # the last hour is scored on the 14:30 window only (minute 315), in the synthetic dry run too
    gd._STATE.update(market=m, datasets=datasets, sets=sets, eval_minutes=dict(gd.rp.EVAL_MINUTES))
    if not (work / "timing.csv").exists():
        (work / "timing.csv").write_text("index,target,seed,variants,n_variants,seconds,rows,pid\n")
    args = [j + (str(work),) for j in jobs]
    done = 0

    def report(res):
        ix, target, seed, vs, secs, n = res
        print(f"  [{done}/{len(jobs)}] {ix} {target} seed {seed} {'+'.join(vs)}: {n} forecasts, {secs:.0f}s "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)

    if a.workers == 1:
        for j in args:
            done += 1
            report(gd.run_job(j))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(a.workers, 2), maxtasksperchild=1) as pool:
            for res in pool.imap(gd.run_job, args, chunksize=1):
                done += 1
                report(res)
    print(f"stage 1 ({index}) done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Statistics (pure functions, tested on synthetic data)
# ---------------------------------------------------------------------------


def beyond_margin(d, delta: float = MARGIN, side: str = "above") -> dict:
    """One-sided test that the mean of the daily series d (points) lies beyond the margin: side "above", H0 mu <= delta
    against mu > delta ("worse by at least delta without the option market"); side "below", H0 mu >= -delta against
    mu < -delta (the mirror). The DM test's variance, HLN factor and Student t(n - 1) (P110's dm_parts); equals the
    project's DM test of d - delta ("greater") or d + delta ("less")."""
    q = st.dm_parts(d)
    if side == "above":
        stat = (q["mean"] - delta) / q["se"] * q["hln"]
        p = float(student_t.sf(stat, q["df"]))
    elif side == "below":
        stat = (q["mean"] + delta) / q["se"] * q["hln"]
        p = float(student_t.cdf(stat, q["df"]))
    else:
        raise ValueError(side)
    return {"estimate": q["mean"], "se": q["se"], "stat": float(stat), "p": p, "n": q["n"]}


def joint_estimate(D: pd.DataFrame, v: str = VARIANT) -> float:
    """J = 100 x (L_v - L_full) / L_mcs, in points (positive: removing the inputs raises the loss)."""
    return 100.0 * (D[v].mean() - D["full"].mean()) / D["mcsgarch"].mean()


def interaction_series(D: pd.DataFrame) -> np.ndarray:
    """Daily 100 x (L_optionmkt - L_implied - L_vix + L_full) / mean L_mcs: its mean is J - C_implied - C_vix."""
    return 100.0 * (D[VARIANT] - D["implied"] - D["vix"] + D["full"]).to_numpy(float) / D["mcsgarch"].mean()


def seed_average(frames: list, cols=None) -> pd.DataFrame:
    """Daily QLIKE averaged over seed sets, for every variant column present in all frames (P110's seed_average
    averages only P110's seven variants, so it would leave ``optionmkt`` at the first seed set's values). mcsGARCH
    (and mcsgarch_iv where present) and the days must be identical in every seed set."""
    base = frames[0]
    bench = [c for c in ("mcsgarch", "mcsgarch_iv") if all(c in f for f in frames)]
    for f in frames[1:]:
        if not f.index.equals(base.index):
            raise ValueError("the days differ between seed sets")
        for c in bench:
            if not np.array_equal(f[c].to_numpy(), base[c].to_numpy()):
                raise ValueError(f"{c} differs between seed sets")
    known = list(gd.VARIANTS)
    cols = cols if cols is not None else [c for c in known if all(c in f for f in frames)]
    out = base.copy()
    for c in [c for c in known if c in out and c not in cols]:
        out = out.drop(columns=c)
    for c in cols:
        out[c] = sum(f[c] for f in frames) / len(frames)
    return out


def margin_text(delta: float) -> str:
    return f"{delta:g} point"


def primary_verdict(p_eq: float, p_worse: float, p_better: float, seed_est, delta: float = MARGIN,
                    alpha: float = ALPHA) -> tuple[str, str]:
    """The registered verdict of an H1 (or S3a) row and its sub-label: "equivalent within 1 point" if the Holm TOST p
    is below alpha and every seed set's J lies inside (-delta, delta); "worse by at least 1 point" if the Holm p of
    the companion test is below alpha and every seed set's J is above delta; the mirror "better by at least 1 point
    without them"; otherwise "not shown either way" (sub-label "seed-fragile" if a Holm p passed but the seed rule
    failed)."""
    est = list(seed_est)
    if not np.isfinite(p_eq):
        return "not computed", ""
    m = margin_text(delta)
    if p_eq < alpha:
        return (f"equivalent within {m}", "") if all(-delta < e < delta for e in est) else \
            ("not shown either way", "seed-fragile")
    if np.isfinite(p_worse) and p_worse < alpha:
        return (f"worse by at least {m}", "") if all(e > delta for e in est) else \
            ("not shown either way", "seed-fragile")
    if np.isfinite(p_better) and p_better < alpha:
        return (f"better by at least {m} without them", "") if all(e < -delta for e in est) else \
            ("not shown either way", "seed-fragile")
    return "not shown either way", ""


def margin_only_verdict(p_eq: float, seed_est, delta: float, alpha: float = ALPHA) -> tuple[str, str]:
    """S1 (0.5 point): "equivalent within 0.5 point" or "not shown" (sub-label "seed-fragile")."""
    if not np.isfinite(p_eq):
        return "not computed", ""
    if p_eq >= alpha:
        return "not shown", ""
    return (f"equivalent within {margin_text(delta)}", "") if all(-delta < e < delta for e in seed_est) else \
        ("not shown", "seed-fragile")


def difference_verdict(p: float, estimate: float, seed_est, alpha: float = ALPHA) -> tuple[str, str]:
    """S2 and S3b: "different from zero" if the Holm p is below alpha and every seed set has the estimate's sign."""
    if not np.isfinite(p):
        return "not computed", ""
    if p >= alpha:
        return "no difference shown", ""
    return ("different from zero", "") if all(np.sign(e) == np.sign(estimate) for e in seed_est) else \
        ("no difference shown", "seed-fragile")


def within_seed_noise(estimate: float, seed_est) -> bool:
    vals = np.asarray(list(seed_est), float)
    return bool(len(vals) > 1 and abs(estimate) < 2 * np.std(vals, ddof=1))


# ---------------------------------------------------------------------------
# Daily-loss frames
# ---------------------------------------------------------------------------


def read_daily(path: Path) -> pd.DataFrame:
    D = pd.read_csv(path, index_col="date", float_precision="round_trip")
    D.index = pd.to_datetime(D.index)
    return D


def frame_from_work(work: Path, index: str, target: str, seed: int, variants) -> pd.DataFrame:
    wide = gd.load_wide(work, index, target, seed, list(variants))
    D = gd.day_frame(wide, list(variants))
    D.index = pd.to_datetime(D.index)
    return D


def have_work(work: Path, index: str, target: str, seed: int, variants) -> bool:
    return all(gd.work_file(work, index, target, seed, v).exists() for v in variants)


def compare_forecast_files(a_path: Path, b_path: Path, label: str, forecast: bool = True) -> dict:
    """Row-by-row comparison of two work files (identifiers, y, mcsgarch, mcsgarch_iv and, if asked, the forecast)."""
    a = gd.read_csv_exact(a_path).sort_values(["t", "minute"]).reset_index(drop=True)
    b = gd.read_csv_exact(b_path).sort_values(["t", "minute"]).reset_index(drop=True)
    out = {"check": label, "rows_here": len(a), "rows_reference": len(b), "same_row_count": bool(len(a) == len(b))}
    if len(a) != len(b):
        return {**out, "identical": False}
    ok = True
    for c in ["date", "minute", "y", "mcsgarch", "mcsgarch_iv"]:
        same = bool(np.array_equal(a[c].to_numpy(), b[c].to_numpy()))
        out[f"{c}_identical"] = same
        ok &= same
    if forecast:
        x, y = a["forecast"].to_numpy(float), b["forecast"].to_numpy(float)
        out["forecast_n_identical"] = int(np.sum(x == y))
        out["forecast_max_abs_rel_diff"] = float(np.max(np.abs(x / y - 1)))
        ok &= bool(np.all(x == y))
    out["identical"] = bool(ok)
    return out


def compare_daily(a: pd.DataFrame, b: pd.DataFrame, cols, label: str) -> dict:
    """Day-by-day comparison of daily QLIKE columns of two frames (same days required)."""
    out = {"check": label, "rows_here": len(a), "rows_reference": len(b),
           "same_days": bool(a.index.equals(b.index))}
    ok = out["same_days"]
    if ok:
        for c in cols:
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            out[f"{c}_n_identical"] = int(np.sum(x == y))
            out[f"{c}_max_abs_rel_diff"] = float(np.max(np.abs(x / y - 1)))
            ok &= bool(np.all(x == y))
    out["identical"] = bool(ok)
    return out


def load_frames(a, synth: bool, repro: list) -> dict:
    """(index, target, seed) -> day frame with mcsgarch, full, optionmkt (and implied, vix where available)."""
    work = Path(a.work)
    F = {}
    if synth:
        for ix in SYNTH_INDEXES:
            for t in TARGETS:
                for s in SYNTH["seeds"]:
                    if have_work(work, ix, t, s, SYNTH_VARIANTS):
                        F[(ix, t, s)] = frame_from_work(work, ix, t, s, SYNTH_VARIANTS)
            for s in SYNTH["seeds"]:
                if have_work(work, ix, LAST, s, ("full", VARIANT)):
                    F[(ix, LAST, s)] = frame_from_work(work, ix, LAST, s, ("full", VARIANT))
        return F
    p106_work, p110_work = Path(a.p106_work), Path(a.p110_work)
    # ---- Nifty, 15 and 30 minutes: P106's committed daily losses (no refit); checked against P106's work files
    for t in TARGETS:
        for s in SEEDS:
            D = read_daily(P106_RESULTS / f"daily_losses_nifty_{t}_s{s}.csv")
            F[("NIFTY", t, s)] = D
            if have_work(p106_work, "NIFTY", t, s, ("full", VARIANT)):
                W = frame_from_work(p106_work, "NIFTY", t, s, ("full", VARIANT))
                repro.append(compare_daily(W, D, ["mcsgarch", "full", VARIANT],
                                           f"NIFTY {t} seed {s}: daily QLIKE from P106's work files vs P106's committed "
                                           f"daily_losses_nifty_{t}_s{s}.csv"))
            else:
                repro.append({"check": f"NIFTY {t} seed {s}: P106 work files not on disk ({p106_work}); not run",
                              "identical": np.nan})
    # the rerun Nifty optionmkt job against P106's work file
    mine = gd.work_file(work, "NIFTY", "next15", HEADLINE_SEED, VARIANT)
    ref = gd.work_file(p106_work, "NIFTY", "next15", HEADLINE_SEED, VARIANT)
    if mine.exists() and ref.exists():
        repro.append(compare_forecast_files(mine, ref, "NIFTY next15 seed 7 optionmkt: rerun here vs P106's work file"))
    else:
        repro.append({"check": "NIFTY next15 seed 7 optionmkt rerun vs P106's work file: a file is missing; not run",
                      "identical": np.nan})
    # ---- Nifty last hour: seed set 7 from P106's committed file, 101 to 104 from this study's runs
    F[("NIFTY", LAST, HEADLINE_SEED)] = read_daily(P106_RESULTS / "daily_losses_nifty_last_hour_s7.csv")
    p99 = None
    if (P99_DIR / "daily_losses_last_hour.csv").exists():
        p99 = pd.read_csv(P99_DIR / "daily_losses_last_hour.csv", index_col="date", parse_dates=["date"],
                          float_precision="round_trip")
    for s in NEW_SEEDS:
        if have_work(work, "NIFTY", LAST, s, ("full", VARIANT)):
            D = frame_from_work(work, "NIFTY", LAST, s, ("full", VARIANT))
            F[("NIFTY", LAST, s)] = D
            if p99 is not None:
                r = gd.p99_daily_check(D, p99, s, f"NIFTY last_hour seed {s} full hybrid daily QLIKE vs P99 "
                                                  "daily_losses_last_hour.csv")
                r["identical"] = bool(r["hybrid_base_n_identical"] == r["rows_committed"] == r["rows_matched"]
                                      and r["mcsgarch_n_identical"] == r["rows_committed"])
                repro.append(r)
    # ---- Bank Nifty and Fin Nifty
    for ix in NEW_INDEXES:
        for t in TARGETS:
            for s in SEEDS:
                if not have_work(work, ix, t, s, (VARIANT,)):
                    continue
                N = frame_from_work(work, ix, t, s, (VARIANT,))
                ref_path = P110_RESULTS / f"daily_losses_{ix.lower()}_{t}_s{s}.csv"
                R = read_daily(ref_path)
                chk = compare_daily(N, R, ["mcsgarch", "mcsgarch_iv"],
                                    f"{ix} {t} seed {s}: mcsGARCH daily QLIKE here vs P110's committed {ref_path.name}")
                repro.append(chk)
                if not chk["identical"] and not a.allow_mismatch:
                    raise SystemExit(f"{ix} {t} seed {s}: mcsGARCH differs from P110's committed daily losses; rerun "
                                     "with --allow-mismatch to report anyway")
                D = R.copy()
                D[VARIANT] = N[VARIANT].reindex(D.index)
                F[(ix, t, s)] = D
                # row-by-row against P110's full work file, and P110's full daily QLIKE recomputed from it
                pf = gd.work_file(p110_work, ix, t, s, "full")
                if pf.exists():
                    repro.append(compare_forecast_files(gd.work_file(work, ix, t, s, VARIANT), pf,
                                                        f"{ix} {t} seed {s}: identifiers, y, mcsgarch, mcsgarch_iv of the "
                                                        "optionmkt work file vs P110's full work file", forecast=False))
                    W = frame_from_work(p110_work, ix, t, s, ("full",))
                    repro.append(compare_daily(W, R, ["mcsgarch", "full"],
                                               f"{ix} {t} seed {s}: full hybrid daily QLIKE from P110's work file vs "
                                               f"P110's committed {ref_path.name}"))
                else:
                    repro.append({"check": f"{ix} {t} seed {s}: P110's work file not on disk ({p110_work}); not run",
                                  "identical": np.nan})
        for s in SEEDS:
            if not have_work(work, ix, LAST, s, ("full", VARIANT)):
                continue
            F[(ix, LAST, s)] = frame_from_work(work, ix, LAST, s, ("full", VARIANT))
            if s == HEADLINE_SEED:
                wide = gd.load_wide(work, ix, LAST, s, ["full"])
                committed = gd.read_csv_exact(gd.INDEXES[ix]["run"] / "forecasts_last_hour.csv")
                r = gd.reproduction_check(wide, committed, f"{ix} last_hour seed 7 full hybrid vs "
                                                           f"{gd.INDEXES[ix]['run'].name}/forecasts_last_hour.csv")
                r["identical"] = bool(r["rows_rerun"] == r["rows_committed"] == r["rows_matched"]
                                      and all(r[f"{c}_n_identical"] == r["rows_matched"]
                                              for c in ("y", "mcsgarch", "mcsgarch_iv", "hybrid_base")))
                repro.append(r)
    return F


# ---------------------------------------------------------------------------
# Stage 2: analysis
# ---------------------------------------------------------------------------

DM_METHOD = st.DM_METHOD
DM_TWO = "Diebold-Mariano on daily mean QLIKE, Newey-West, HLN factor, Student t(n-1), two-sided"
BEYOND_METHOD = ("one-sided test that J lies beyond the margin ({d} point(s)) on the daily difference in points of "
                 "mcsGARCH's QLIKE; DM variance (Newey-West, HLN factor, Student t(n-1))")
INTER_METHOD = ("two-sided DM test (Newey-West, HLN factor, Student t(n-1)) of the daily series 100 x (optionmkt - "
                "implied - vix + full) / mean mcsGARCH QLIKE against zero")

HYP = {
    "tost": ("The hybrid without the option-implied and VIX groups together (optionmkt) has mean QLIKE within {d} "
             "point(s) of mcsGARCH's QLIKE of the full hybrid's ({t}). Estimate: J, points (positive: the option "
             "market helps)."),
    "worse": ("The hybrid without the option-implied and VIX groups together is worse than the full hybrid by at least "
              "{d} point(s) of mcsGARCH's QLIKE ({t}). Estimate: J, points."),
    "better": ("Mirror, for completeness: the hybrid without the option-implied and VIX groups together is better than "
               "the full hybrid by at least {d} point(s) of mcsGARCH's QLIKE ({t}). Estimate: J, points."),
    "diff": ("Removing the option-implied and VIX groups together changes the hybrid's mean QLIKE at all ({t}). "
             "Estimate: J, points."),
    "inter": ("The option-implied and VIX groups stand in for each other: removing both changes the loss by a different "
              "amount than the sum of removing each ({t}). Estimate: J - C_implied - C_vix, points (positive: "
              "substitutes)."),
}
SIDED = {"tost": "two one-sided tests (equivalence)", "worse": "one-sided (J above the margin)",
         "better": "one-sided (J below minus the margin)", "diff": "two-sided", "inter": "two-sided"}


def sample_text(index: str, t: str, D: pd.DataFrame, what: str, synth: bool) -> str:
    d0, d1 = pd.Timestamp(D.index.min()).date(), pd.Timestamp(D.index.max()).date()
    pre = "SYNTHETIC: " if synth else "REAL DATA: "
    return f"{pre}{LABEL.get(index, index)}, {TARGET_LABEL[t]}, {len(D)} test days {d0} to {d1}, {what}"


def row_stats(D: pd.DataFrame, kind: str, delta: float = MARGIN, method: str = "dm") -> dict:
    """Estimate, statistic and p of one row on a day frame (columns mcsgarch, full, optionmkt; implied and vix for the
    interaction)."""
    J = joint_estimate(D)
    if kind == "tost":
        if method == "delta":
            r = st.tost_delta_method((D[VARIANT] - D["full"]).to_numpy(float), D["mcsgarch"].to_numpy(float), delta)
        else:
            r = st.tost(st.points_series(D, VARIANT, "full"), delta)
        return {**r, "estimate": r["estimate"] if method == "delta" else J}
    if kind in ("worse", "better"):
        r = beyond_margin(st.points_series(D, VARIANT, "full"), delta, "above" if kind == "worse" else "below")
        return {**r, "estimate": J}
    if kind == "diff":
        dm = diebold_mariano(D[VARIANT], D["full"], alternative="two-sided")
        return {"estimate": J, "stat": dm["stat"], "p": dm["p"], "n": dm["n"]}
    if kind == "inter":
        x = interaction_series(D)
        dm = diebold_mariano(x, np.zeros_like(x), alternative="two-sided")
        return {"estimate": float(x.mean()), "stat": dm["stat"], "p": dm["p"], "n": dm["n"]}
    raise ValueError(kind)


def stage_analyze(a):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    synth = a.synthetic
    label = LABEL_SYNTH if synth else LABEL_REAL
    plan_commit = None
    if not synth:
        require_plan()
        plan_commit = check_plan_commit(a.plan_commit)
        if _git(["diff", "--quiet", plan_commit, "--", PLAN]).returncode != 0:
            print(f"note: {PLAN} differs from its version at the plan commit (any change is a deviation)", flush=True)
    print(f"[{label}] P120 joint option market, stage 2 ({PLAN})", flush=True)
    indexes = SYNTH_INDEXES if synth else STUDY_INDEXES
    seeds_all = SYNTH["seeds"] if synth else SEEDS
    new_seeds = seeds_all[1:]
    reps = a.reps
    repro = []
    F = load_frames(a, synth, repro)

    def avg(ix, t, seeds):
        if not all((ix, t, s) in F for s in seeds):
            return None
        return seed_average([F[(ix, t, s)] for s in seeds])

    A5 = {(ix, t): avg(ix, t, seeds_all) for ix in indexes for t in TARGETS + [LAST]}
    A4 = {(ix, t): avg(ix, t, new_seeds) for ix in indexes for t in TARGETS + [LAST]}
    for (ix, t), D in A5.items():
        if D is not None:
            repro.append({"check": f"{ix} {t}: mcsGARCH daily QLIKE identical in all {len(seeds_all)} seed sets",
                          "rows_here": len(D), "identical": True})
    R = gd.Rows()
    verd_rows, bound_rows, contrib_rows, share_rows, pool_rows, seed_rows = [], [], [], [], [], []
    seed_est = {}            # test_id -> {seed set: estimate}
    what5 = f"daily QLIKE averaged over seed sets {', '.join(map(str, seeds_all))}"
    what4 = f"daily QLIKE averaged over the new seed sets {', '.join(map(str, new_seeds))}"

    def add(tid, kind, D, ix, t, what, role, family, delta=MARGIN, method="dm", hyp_extra=""):
        r = row_stats(D, kind, delta, method)
        meth = {"tost": (st.TOST_DELTA if method == "delta" else st.TOST_METHOD).format(d=delta),
                "worse": BEYOND_METHOD.format(d=delta), "better": BEYOND_METHOD.format(d=delta),
                "diff": DM_TWO, "inter": INTER_METHOD}[kind]
        R.add(tid, HYP[kind].format(d=f"{delta:g}", t=f"{TARGET_LABEL[t]}{hyp_extra}"), role,
              sample_text(ix, t, D, what, synth), r["estimate"], r["stat"], r["p"], SIDED[kind], meth, family,
              data=label, index=ix, target=t, kind=kind, margin=delta if kind in ("tost", "worse", "better") else np.nan,
              days=len(D), se=r.get("se", np.nan), ci90_low=r.get("ci90_low", np.nan),
              ci90_high=r.get("ci90_high", np.nan), p_upper=r.get("p_upper", np.nan), p_lower=r.get("p_lower", np.nan),
              smallest_passing_margin=r.get("bound", np.nan))
        return r

    def seeds_of(ix, t, fn):
        return {s: fn(F[(ix, t, s)]) for s in seeds_all if (ix, t, s) in F}

    cells = [(ix, t) for ix in indexes for t in TARGETS]
    cells = [(ix, t) for ix, t in cells if A5[(ix, t)] is not None]
    last_cells = [(ix, LAST) for ix in indexes if A5[(ix, LAST)] is not None]
    gain = {}
    for ix, t in cells + last_cells:
        D = A5[(ix, t)]
        gain[(ix, t)] = 100 * (D["mcsgarch"].mean() - D["full"].mean()) / D["mcsgarch"].mean()

    # ---------------- H1, H1w, H1b (primary), S1, S2, robustness, exploratory: the six 15- and 30-minute cells
    for ix, t in cells:
        D = A5[(ix, t)]
        J_s = seeds_of(ix, t, joint_estimate)
        for kind, fam, role, delta in (("tost", "H1", "primary", MARGIN), ("worse", "H1w", "primary (companion)", MARGIN),
                                       ("better", "H1b", "primary (companion)", MARGIN),
                                       ("tost", "S1_margin_half", "secondary", MARGIN_SENS),
                                       ("diff", "S2_two_sided", "secondary", MARGIN)):
            tid = f"{fam.split('_')[0]}_{ix}_{t}"
            r = add(tid, kind, D, ix, t, what5, role, fam, delta)
            seed_est[tid] = J_s
            if kind == "tost":
                bound_rows.append({"data": label, "test_id": tid, "index": ix, "target": t, "sample": "five-seed average",
                                   "margin": delta, "estimate_points": r["estimate"], "se_points": r["se"],
                                   "ci90_low": r["ci90_low"], "ci90_high": r["ci90_high"],
                                   "smallest_margin_that_passes": r["bound"], "hybrid_gain_points": gain[(ix, t)],
                                   "bound_as_share_of_gain": r["bound"] / gain[(ix, t)], "p_tost": r["p"],
                                   "p_margin_1": row_stats(D, "tost", MARGIN)["p"],
                                   "p_margin_0.5": row_stats(D, "tost", MARGIN_SENS)["p"]})
        # robustness R1 to R5: the H1 TOST in other samples or with another variance
        samples = [("R1_seed7", F.get((ix, t, HEADLINE_SEED)), f"seed set {HEADLINE_SEED} alone", "dm"),
                   ("R2_new_seeds", A4[(ix, t)], what4, "dm"),
                   ("R3_delta_method", D, what5, "delta")]
        jm = st.date_mask(D, st.INFLUENCE_DATES)
        if not jm.all():
            samples.append(("R4_without_june_2024", D[jm], f"{what5}, without 4 and 5 June 2024", "dm"))
        keep = st.top_loss_days(D)
        samples.append(("R5_without_top_loss_days", D[keep], f"{what5}, without the {int((~keep).sum())} days with the "
                        "largest mcsGARCH daily QLIKE", "dm"))
        for fam, DD, what, meth in samples:
            if DD is None:
                continue
            add(f"{fam.split('_')[0]}_{ix}_{t}", "tost", DD, ix, t, what, "robustness", fam, MARGIN, meth)
        # exploratory E1 (interaction) and E2 (expiry days)
        if all(c in D for c in ("implied", "vix")):
            tid = f"E1_{ix}_{t}"
            add(tid, "inter", D, ix, t, what5, "exploratory", "E1_interaction")
            seed_est[tid] = seeds_of(ix, t, lambda x: float(interaction_series(x).mean()))
        ex = D["is_expiry"].astype(bool).to_numpy()
        if ex.sum() > 10:
            add(f"E2_{ix}_{t}", "tost", D[ex], ix, t, f"{what5}, expiry days only", "exploratory", "E2_expiry_days",
                MARGIN)

    # ---------------- S3: the last hour
    for ix, t in last_cells:
        D = A5[(ix, t)]
        J_s = seeds_of(ix, t, joint_estimate)
        for kind, fam, role in (("tost", "S3a_last_hour_tost", "secondary"), ("worse", "S3aw_last_hour_worse", "secondary"),
                                ("better", "S3ab_last_hour_better", "secondary"), ("diff", "S3b_last_hour_dm", "secondary")):
            tid = f"{fam.split('_')[0]}_{ix}_{t}"
            r = add(tid, kind, D, ix, t, what5, role, fam, MARGIN)
            seed_est[tid] = J_s
            if kind == "tost":
                bound_rows.append({"data": label, "test_id": tid, "index": ix, "target": t, "sample": "five-seed average",
                                   "margin": MARGIN, "estimate_points": r["estimate"], "se_points": r["se"],
                                   "ci90_low": r["ci90_low"], "ci90_high": r["ci90_high"],
                                   "smallest_margin_that_passes": r["bound"], "hybrid_gain_points": gain[(ix, t)],
                                   "bound_as_share_of_gain": r["bound"] / gain[(ix, t)], "p_tost": r["p"],
                                   "p_margin_1": r["p"], "p_margin_0.5": row_stats(D, "tost", MARGIN_SENS)["p"]})

    # ---------------- S4: pooled over the three indices (Driscoll-Kraay), TOST at 1 point
    for t in TARGETS:
        frames = [A5.get((ix, t)) for ix in indexes]
        if any(f is None for f in frames):
            continue
        dates = sorted(set.intersection(*[set(f.index) for f in frames]))
        X = st.pooled_series(frames, "tost", VARIANT, dates)
        r = st.pooled_test(X, "tost", MARGIN)
        names = " + ".join(LABEL[ix] for ix in indexes)
        tid = f"S4_pooled_{t}"
        R.add(tid, HYP["tost"].format(d=f"{MARGIN:g}", t=f"{TARGET_LABEL[t]}, pooled over {names}"), "secondary",
              f"{'SYNTHETIC' if synth else 'REAL DATA'}: {names}, {TARGET_LABEL[t]}, {len(dates)} common test days "
              f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}, {what5}", r["estimate"], r["stat"],
              r["p"], SIDED["tost"], st.POOL_METHOD + f"; margin {MARGIN:g} point", "S4_pooled", data=label, target=t,
              kind="tost", margin=MARGIN, days=len(dates), se=r["se"], ci90_low=r["ci90_low"], ci90_high=r["ci90_high"],
              p_upper=r["p_upper"], p_lower=r["p_lower"], smallest_passing_margin=r["bound"])
        pool_rows.append({"data": label, "test_id": tid, "pool": names, "target": t, "days": len(dates),
                          "estimate_points": r["estimate"], "se": r["se"], "stat": r["stat"], "p": r["p"],
                          "dk_lag": r["lag"], "ci90_low": r["ci90_low"], "ci90_high": r["ci90_high"],
                          "smallest_margin_that_passes": r["bound"],
                          **{f"estimate_{ix.lower()}": e for ix, e in zip(indexes, r["per_index"])}})

    # ---------------- adjust
    families = {
        "H1": "primary H1, TOST at 1 point, the six index and horizon cells",
        "H1w": "primary companion H1w, worse by at least 1 point, the six cells",
        "H1b": "primary companion H1b, better by at least 1 point (mirror), the six cells",
        "S1_margin_half": "secondary S1, TOST at 0.5 point", "S2_two_sided": "secondary S2, two-sided DM",
        "S3a_last_hour_tost": "secondary S3a, last hour, TOST at 1 point",
        "S3aw_last_hour_worse": "secondary S3a companion, last hour, worse by at least 1 point",
        "S3ab_last_hour_better": "secondary S3a companion, last hour, better by at least 1 point (mirror)",
        "S3b_last_hour_dm": "secondary S3b, last hour, two-sided DM", "S4_pooled": "secondary S4, pooled TOST at 1 point",
        "R1_seed7": "robustness R1", "R2_new_seeds": "robustness R2", "R3_delta_method": "robustness R3",
        "R4_without_june_2024": "robustness R4", "R5_without_top_loss_days": "robustness R5",
        "E1_interaction": None, "E2_expiry_days": None}
    tests = R.frame(families)
    padj = dict(zip(tests["test_id"], tests["p_adjusted"]))
    det = {d["test_id"]: d for d in R.detail}

    # ---------------- verdicts
    def vrow(tid, verdict, sub, fam):
        r = tests.loc[tests["test_id"] == tid].iloc[0]
        d = det[tid]
        est = seed_est.get(tid, {})
        vals = np.array(list(est.values()), float)
        return {"data": label, "test_id": tid, "family": fam, "index": d.get("index", ""), "target": d.get("target", ""),
                "estimate": r["estimate"], "se": d.get("se", np.nan), "ci90_low": d.get("ci90_low", np.nan),
                "ci90_high": d.get("ci90_high", np.nan), "p_raw": r["p_raw"], "p_holm": r["p_adjusted"],
                "seed_sets": len(est), "seed_min": float(vals.min()) if len(vals) else np.nan,
                "seed_max": float(vals.max()) if len(vals) else np.nan,
                "seed_sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "within_seed_noise": within_seed_noise(r["estimate"], vals) if len(vals) else False,
                "verdict": verdict, "sub_label": sub}

    for ix, t in cells:
        J_s = list(seed_est[f"H1_{ix}_{t}"].values())
        v, sub = primary_verdict(padj[f"H1_{ix}_{t}"], padj[f"H1w_{ix}_{t}"], padj[f"H1b_{ix}_{t}"], J_s, MARGIN)
        row = vrow(f"H1_{ix}_{t}", v, sub, "H1")
        row.update({"p_holm_worse": padj[f"H1w_{ix}_{t}"], "p_holm_better": padj[f"H1b_{ix}_{t}"],
                    "seed_sets_inside_margin": int(sum(-MARGIN < e < MARGIN for e in J_s)),
                    "seed_sets_above_margin": int(sum(e > MARGIN for e in J_s)),
                    **{f"R{k}_holm": padj.get(f"R{k}_{ix}_{t}", np.nan) for k in range(1, 6)},
                    **{f"R{k}_passes": bool(padj.get(f"R{k}_{ix}_{t}", np.nan) < ALPHA) for k in range(1, 6)}})
        verd_rows.append(row)
        v, sub = margin_only_verdict(padj[f"S1_{ix}_{t}"], J_s, MARGIN_SENS)
        row = vrow(f"S1_{ix}_{t}", v, sub, "S1_margin_half")
        row["seed_sets_inside_margin"] = int(sum(-MARGIN_SENS < e < MARGIN_SENS for e in J_s))
        verd_rows.append(row)
        est = tests.loc[tests["test_id"] == f"S2_{ix}_{t}", "estimate"].iloc[0]
        v, sub = difference_verdict(padj[f"S2_{ix}_{t}"], est, J_s)
        row = vrow(f"S2_{ix}_{t}", v, sub, "S2_two_sided")
        row["seed_sets_same_sign"] = int(sum(np.sign(e) == np.sign(est) for e in J_s))
        verd_rows.append(row)
    for ix, t in last_cells:
        J_s = list(seed_est[f"S3a_{ix}_{t}"].values())
        v, sub = primary_verdict(padj[f"S3a_{ix}_{t}"], padj[f"S3aw_{ix}_{t}"], padj[f"S3ab_{ix}_{t}"], J_s, MARGIN)
        row = vrow(f"S3a_{ix}_{t}", v, sub, "S3a_last_hour_tost")
        row.update({"p_holm_worse": padj[f"S3aw_{ix}_{t}"], "p_holm_better": padj[f"S3ab_{ix}_{t}"],
                    "seed_sets_inside_margin": int(sum(-MARGIN < e < MARGIN for e in J_s))})
        verd_rows.append(row)
        est = tests.loc[tests["test_id"] == f"S3b_{ix}_{t}", "estimate"].iloc[0]
        v, sub = difference_verdict(padj[f"S3b_{ix}_{t}"], est, J_s)
        row = vrow(f"S3b_{ix}_{t}", v, sub, "S3b_last_hour_dm")
        row["seed_sets_same_sign"] = int(sum(np.sign(e) == np.sign(est) for e in J_s))
        verd_rows.append(row)
    for t in TARGETS:
        tid = f"S4_pooled_{t}"
        if tid in padj:
            verd_rows.append(vrow(tid, "equivalent within 1 point (pooled)" if padj[tid] < ALPHA else "not shown (pooled)",
                                  "", "S4_pooled"))
    for ix, t in cells:
        if f"E1_{ix}_{t}" in padj:
            verd_rows.append(vrow(f"E1_{ix}_{t}", "exploratory (no verdict)", "", "E1_interaction"))
    verdicts = pd.DataFrame(verd_rows)

    # ---------------- descriptive: contributions in every sample, seed repeats, shares of the gain
    for ix, t in cells + last_cells:
        samples = [("five-seed average", A5[(ix, t)]), ("new-seed average", A4[(ix, t)])] + \
                  [(f"seed set {s}", F.get((ix, t, s))) for s in seeds_all]
        for sname, D in samples:
            if D is None:
                continue
            mc = D["mcsgarch"].mean()
            row = {"data": label, "index": ix, "target": t, "sample": sname, "days": len(D),
                   "J_optionmkt_points": joint_estimate(D),
                   "hybrid_gain_points": 100 * (mc - D["full"].mean()) / mc,
                   "optionmkt_gain_points": 100 * (mc - D[VARIANT].mean()) / mc}
            for v in ("implied", "vix", "mcsinfo"):
                if v in D:
                    row[f"C_{v}_points"] = 100 * (D[v].mean() - D["full"].mean()) / mc
            if "implied" in D and "vix" in D:
                row["interaction_points"] = row["J_optionmkt_points"] - row["C_implied_points"] - row["C_vix_points"]
            contrib_rows.append(row)
        D = A5[(ix, t)]
        rng = np.random.default_rng([BOOT_SEED, indexes.index(ix), (TARGETS + [LAST]).index(t)])
        sh, lo, hi = gd.share_ci(D["mcsgarch"].to_numpy(), D["full"].to_numpy(), D[VARIANT].to_numpy(), reps=reps, rng=rng)
        share_rows.append({"data": label, "index": ix, "target": t, "sample": "five-seed average", "variant": VARIANT,
                           "share_of_gain": sh, "share_ci_low": lo, "share_ci_high": hi, "reps": reps})
    for tid, est in seed_est.items():
        vals = np.array(list(est.values()), float)
        seed_rows.append({"data": label, "test_id": tid, **{f"estimate_s{s}": e for s, e in est.items()},
                          "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                          "min": float(vals.min()), "max": float(vals.max())})

    # ---------------- write
    cols = ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"]
    tests[cols].to_csv(out / "tests.csv", index=False)
    pd.DataFrame(R.detail).to_csv(out / "tests_detail.csv", index=False)
    verdicts.to_csv(out / "verdict.csv", index=False)
    pd.DataFrame(bound_rows).to_csv(out / "equivalence_bounds.csv", index=False)
    pd.DataFrame(contrib_rows).to_csv(out / "contributions.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "seed_repeats.csv", index=False)
    pd.DataFrame(share_rows).to_csv(out / "shares.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(out / "pooled.csv", index=False)
    rp_df = pd.DataFrame(repro)
    rp_df.insert(0, "data", label)
    rp_df.to_csv(out / "reproduction_check.csv", index=False)
    written = []
    for (ix, t, s), D in sorted(F.items()):
        new = synth or ix in NEW_INDEXES or (ix == "NIFTY" and t == LAST and s != HEADLINE_SEED)
        if not new:
            continue
        keep_cols = [c for c in ["mcsgarch", "mcsgarch_iv", "full", VARIANT, "implied", "vix", "t", "is_expiry", "regime"]
                     if c in D]
        DD = D[keep_cols].copy()
        DD.index = pd.to_datetime(DD.index).strftime("%Y-%m-%d")
        DD.index.name = "date"
        DD.insert(0, "data", label)
        path = out / f"daily_losses_{ix.lower()}_{t}_s{s}.csv"
        DD.to_csv(path)
        written.append(path.name)
    timing = pd.read_csv(work / "timing.csv") if (work / "timing.csv").exists() else pd.DataFrame()
    if len(timing):
        timing.insert(0, "data", label)
        timing.to_csv(out / "timing.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 70)
    print(f"\n[{label}] verdicts:")
    print(verdicts.drop(columns=["data"]).to_string(index=False))
    print(f"\n[{label}] equivalence bounds:")
    print(pd.DataFrame(bound_rows).drop(columns=["data"]).to_string(index=False))
    print(f"\n[{label}] contributions (points of mcsGARCH's QLIKE):")
    print(pd.DataFrame(contrib_rows).drop(columns=["data"]).to_string(index=False))
    print(f"\n[{label}] pooled:")
    print(pd.DataFrame(pool_rows).drop(columns=["data"], errors="ignore").to_string(index=False))
    print(f"\n[{label}] shares of the gain:")
    print(pd.DataFrame(share_rows).drop(columns=["data"]).to_string(index=False))
    print(f"\n[{label}] reproduction checks:")
    print(rp_df.drop(columns=["data"]).to_string(index=False))
    print(f"\n[{label}] every row (id, estimate, statistic, p, adjusted p):")
    print(tests[["test_id", "estimate", "statistic", "p_raw", "p_adjusted"]].to_string(index=False))

    commit = _git(["rev-parse", "HEAD"]).stdout.strip() or None
    import arch
    import lightgbm
    import scipy
    inputs = {}
    for ix in (SYNTH_INDEXES if synth else STUDY_INDEXES):
        p = work / f"inputs_{ix}.json"
        if p.exists():
            inputs[ix] = json.loads(p.read_text())
    wall = {}
    for ix in (SYNTH_INDEXES if synth else STUDY_INDEXES):
        files = list(work.glob(f"{ix}_*.csv"))
        start_file = work / f"inputs_{ix}.json"
        if files:
            end = max(f.stat().st_mtime for f in files)
            start = start_file.stat().st_mtime if start_file.exists() else None
            wall[ix] = {"start_utc": (pd.Timestamp(start, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ") if start else None),
                        "end_utc": pd.Timestamp(end, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "minutes": round((end - start) / 60, 1) if start else None}
    ref_files = {}
    if not synth:
        for p in sorted(list(P106_RESULTS.glob("daily_losses_nifty_*.csv")) + list(P110_RESULTS.glob("daily_losses_*.csv"))
                        + [P99_DIR / "daily_losses_last_hour.csv"]):
            if p.exists():
                ref_files[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    json.dump({"source": label, "data": label,
               "study": f"P120 joint option market ({PLAN})",
               "plan_commit": plan_commit, "analysis_commit": commit,
               "commands": [
                   "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index BANKNIFTY --workers 2",
                   "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index FINNIFTY --workers 2",
                   "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index NIFTY --workers 2",
                   "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py analyze --plan-commit <plan commit>"],
               "variant": VARIANT, "drop_set": gd.DROP_SETS[VARIANT], "pipeline_settings": gd.PIPELINE_SETTINGS,
               "synthetic_settings": SYNTH if synth else None,
               "seeds": {"all": seeds_all, "primary_sample": what5,
                         "lgbm_seeds": {str(s): [s + 1000 * k for k in range(gd.N_SEEDS)] for s in seeds_all}},
               "margin_points": MARGIN, "sensitivity_margin_points": MARGIN_SENS, "alpha": ALPHA,
               "bootstrap": {"reps": reps, "mean_block_days": MEAN_BLOCK, "seed": BOOT_SEED},
               "influence_dates": st.INFLUENCE_DATES, "top_share": st.TOP_SHARE,
               "reference_files_sha256": ref_files, "daily_loss_files_written": written, "input_sha256": inputs,
               "seconds_stage1_jobs": float(timing["seconds"].sum()) if len(timing) else None,
               "stage1_wall_clock": wall, "seconds_stage2": round(time.time() - t0, 1),
               "python": sys.version.split()[0],
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
                            "arch": arch.__version__, "scipy": scipy.__version__}},
              open(out / "run_info.json", "w"), indent=2, default=str)
    print(f"\nwritten to {out} ({time.time() - t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("run", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--work", default=None)
        p.add_argument("--synthetic", action="store_true", help="simulated markets and small settings (dry run)")
        if name == "run":
            p.add_argument("--index", type=str.upper, choices=STUDY_INDEXES + SYNTH_INDEXES, default="BANKNIFTY")
            p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
            p.add_argument("--targets", nargs="+", default=None)
            p.add_argument("--seeds", nargs="+", type=int, default=None)
            p.add_argument("--workers", type=int, default=2, help="worker processes (at most 2)")
        if name == "analyze":
            p.add_argument("--out", default=None)
            p.add_argument("--reps", type=int, default=REPS)
            p.add_argument("--plan-commit", default=None)
            p.add_argument("--p106-work", default=str(ROOT / "data" / "processed" / "p106"),
                           help="P106's git-ignored work files (Nifty checks)")
            p.add_argument("--p110-work", default=str(ROOT / "data" / "processed" / "p110"),
                           help="P110's git-ignored work files (Bank Nifty and Fin Nifty checks)")
            p.add_argument("--allow-mismatch", action="store_true",
                           help="report even if mcsGARCH differs from P110's committed daily losses")
    a = ap.parse_args(argv)
    if a.work is None:
        a.work = str(ROOT / "data" / "processed" / ("p120_synthetic" if a.synthetic else "p120"))
    if a.stage == "run":
        if a.synthetic != is_synth(a.index):
            raise SystemExit("--synthetic goes with the SYNTH* indices and only with them")
        stage_run(a)
    else:
        if a.out is None:
            a.out = str(ROOT / "data" / "processed" / "p120_synthetic_results") if a.synthetic else \
                str(ROOT / "results" / "p120_joint_option_market")
        stage_analyze(a)


if __name__ == "__main__":
    main()
