"""P99: seed bootstrap of the forecasting hybrid (registered plan: docs/prereg/P99_SEED_BOOTSTRAP.md).

The referee (docs/REFEREE_RESPONSE_FORECAST.md, M4.3) said the Diebold-Mariano p-values of the
headline treat the fitted hybrid as fixed, although LightGBM's random seeds move it. This script
reruns the Nifty v2 walk-forward of ``hybrid_base`` (mcsGARCH times a LightGBM correction, LightGBM
averaged over 5 seeds) with K = 20 new, non-overlapping 5-seed ensembles, at all three targets, with
the same data and settings as the headline run (results/pipeline_nse_v2/run_info.json). Only the
models the hybrid needs are fitted: mcsGARCH (deterministic) and hybrid_base; run_walkforward also
fits the implied-level mcsGARCH internally because two of the hybrid's features come from it.

Two stages:

    # 1. forecasts (REAL DATA): one walk-forward per (target, base seed); about 1 to 2 hours on 2 cores
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py run --raw data/raw --workers 2 \
        > results/seed_bootstrap/run_stdout.txt 2>&1
    # 2. losses, Diebold-Mariano tests, the seed bootstrap and the registered verdict
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py analyze \
        > results/seed_bootstrap/analyze_stdout.txt 2>&1

``check`` compares the work files of the headline base seed (7) with the committed v2 forecasts and
computes no loss (it was used before the plan was committed). Per-job forecasts go to --work
(default data/processed/seed_bootstrap, git-ignored); a finished job is not rerun. The analysis
writes results/seed_bootstrap/*.csv and run_info.json. Tests: tests/test_seed_bootstrap.py
(synthetic data only).
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

from niftygex.evaluation import daily_losses, diebold_mariano, qlike, subgroups  # noqa: E402
from niftygex.features import CLOSING_AUCTION_START, GEX_SETS, build_dataset, build_gex_panel  # noqa: E402
from niftygex.forecast import WalkForwardConfig, run_walkforward  # noqa: E402
from niftygex.implied import build_implied_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The pipeline's own market loader and target definitions, and the stationary bootstrap of the
# referee response's SPA test (M4.2), so that nothing here is a second implementation.
rp = _load("run_pipeline", ROOT / "scripts" / "run_pipeline.py")
rf = _load("referee_forecast", ROOT / "experiments" / "referee_forecast.py")

HEADLINE_RUN = ROOT / "results" / "pipeline_nse_v2"
TARGETS = list(rp.TARGETS)                      # next15, next30, last_hour
MODELS = ["mcsgarch", "hybrid_base"]
HEADLINE_SEED = 7                               # the v2 run's --seed
SEED_STUDY_SEEDS = (7, 8, 9)                    # single seeds of the v2 run's seed study (next30)
NEW_SEEDS = list(range(101, 121))               # K = 20 new base seeds (registered)
N_SEEDS = 5                                     # LightGBM fits per ensemble, seeds base + 1000 * s
END = "2026-11-02"
HOLDOUT_START = "2025-07-01"
SUBSETS = ["all", "expiry", "pre_auction", "holdout"]   # all and expiry are registered; the others descriptive
VERDICT_TARGETS = ("next15", "next30")
ALPHA = 0.05
REPS = 10000
MEAN_BLOCK = 10.0
SENS_BLOCKS = (5.0, 20.0)
BOOT_SEED = 99
# Settings of the headline run that this study must share (checked against its run_info.json).
PIPELINE_SETTINGS = dict(source="nse", symbol="NIFTY", start="2023-01-01", min_coverage=0.95, rate=0.065,
                         oi_unit="keep_units", lot_size=None, tune=False, tune_every=3, n_seeds=N_SEEDS,
                         seed=HEADLINE_SEED, min_train=150, block=21, mcs_refit_every=21,
                         holdout_start=HOLDOUT_START)


# ---------------------------------------------------------------------------
# Seeds and configuration
# ---------------------------------------------------------------------------


def lgbm_seeds(base: int, n_seeds: int = N_SEEDS) -> list[int]:
    """LightGBM seeds of one ensemble, as in forecast._lgbm_fit_predict (the tuning pass uses ``base``)."""
    return [base + 1000 * s for s in range(n_seeds)]


def seeds_disjoint(bases, used=(HEADLINE_SEED,) + SEED_STUDY_SEEDS, n_seeds: int = N_SEEDS) -> bool:
    """True if no LightGBM seed of any ensemble in ``bases`` repeats, or appears in the ensembles
    (5-seed for the headline, single for the seed study) of the seeds in ``used``."""
    seen = set()
    for b in sorted(set(used)):
        seen.update(lgbm_seeds(b, n_seeds if b == HEADLINE_SEED else 1))
    for b in bases:
        s = set(lgbm_seeds(b, n_seeds))
        if s & seen:
            return False
        seen |= s
    return True


def make_config(seed: int, eval_minutes, gex_sets: dict, models=MODELS, **over) -> WalkForwardConfig:
    """The headline run's WalkForwardConfig (scripts/run_pipeline.py main) with another base seed and
    only ``models``."""
    kw = dict(models=list(models), gex_sets=gex_sets, tune=PIPELINE_SETTINGS["tune"],
              tune_every=PIPELINE_SETTINGS["tune_every"], n_seeds=PIPELINE_SETTINGS["n_seeds"], seed=seed,
              min_train=PIPELINE_SETTINGS["min_train"], block=PIPELINE_SETTINGS["block"],
              mcs_refit_every=PIPELINE_SETTINGS["mcs_refit_every"])
    kw.update(over)
    cfg = WalkForwardConfig(**kw)
    cfg.eval_minutes = eval_minutes
    return cfg


def check_headline_settings(run_info: dict, raw_end: str | None) -> list[str]:
    """Differences between PIPELINE_SETTINGS and the headline run's arguments (empty list = none)."""
    a = run_info["args"]
    bad = [f"{k}: here {v!r}, headline {a.get(k)!r}" for k, v in PIPELINE_SETTINGS.items()
           if k != "symbol" and a.get(k) != v]
    if a.get("symbol", "NIFTY") != "NIFTY":
        bad.append("symbol")
    if run_info["days"][1] != raw_end:
        bad.append(f"last day: headline {run_info['days'][1]}, here {raw_end}")
    return bad


# ---------------------------------------------------------------------------
# Stage 1: forecasts
# ---------------------------------------------------------------------------

_STATE: dict = {}   # filled in the parent before the workers are forked


def work_file(work: Path, target: str, seed: int) -> Path:
    return work / f"forecasts_{target}_s{seed}.csv"


def run_job(job):
    target, seed, work = job
    m, ds, sets = _STATE["market"], _STATE["datasets"][target], _STATE["sets"]
    t0 = time.time()
    cfg = make_config(seed, rp.EVAL_MINUTES.get(target), sets)
    pred, _ = run_walkforward(m, ds, cfg)
    keep = ["t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y"] + MODELS
    path = work_file(Path(work), target, seed)
    tmp = path.with_suffix(".tmp")
    pred[keep].to_csv(tmp, index=False)
    os.replace(tmp, path)
    secs = round(time.time() - t0, 1)
    with open(Path(work) / "timing.csv", "a") as fh:
        fh.write(f"{target},{seed},{secs},{len(pred)},{os.getpid()}\n")
    return target, seed, secs, len(pred)


def stage_run(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    seeds = a.seeds
    jobs = [(t, s, str(work)) for s in seeds for t in a.targets if not work_file(work, t, s).exists()]
    # headline seed first (its reproduction can be checked early), then the longest targets first
    order = {t: i for i, t in enumerate(TARGETS)}
    jobs.sort(key=lambda j: (j[1] != HEADLINE_SEED, order[j[0]], j[1]))
    print(f"[REAL DATA] P99 seed bootstrap, stage 1: {len(jobs)} jobs to run "
          f"({len(seeds) * len(a.targets) - len(jobs)} already in {work}); workers {a.workers}", flush=True)
    if not jobs:
        return
    t0 = time.time()
    info = json.load(open(HEADLINE_RUN / "run_info.json"))
    ns = argparse.Namespace(source="nse", raw=a.raw, symbol="NIFTY", oi_unit=PIPELINE_SETTINGS["oi_unit"],
                            lot_size=None, rate=PIPELINE_SETTINGS["rate"], start=PIPELINE_SETTINGS["start"],
                            end=a.end, min_coverage=PIPELINE_SETTINGS["min_coverage"])
    m = rp.load_market(ns)
    last = str(pd.Timestamp(m["days"][-1]).date())
    bad = check_headline_settings(info, last)
    if bad or len(m["days"]) != info["days"][2]:
        raise SystemExit(f"settings differ from the headline run: {bad}, days {len(m['days'])} vs {info['days'][2]}")
    print(f"market loaded: {len(m['days'])} days, {m['days'][0]} to {m['days'][-1]} ({time.time() - t0:.0f}s)", flush=True)
    panel = build_gex_panel(m)
    implied = build_implied_panel(m)
    has_oracle = "dealer_next" in m
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle" or has_oracle}
    datasets = {t: build_dataset(m, panel, implied=implied, **rp.TARGETS[t]) for t in sorted({j[0] for j in jobs})}
    print(f"panels and datasets built ({time.time() - t0:.0f}s)", flush=True)
    _STATE.update(market=m, datasets=datasets, sets=sets)
    if not (work / "timing.csv").exists():
        (work / "timing.csv").write_text("target,seed,seconds,rows,pid\n")
    ctx = mp.get_context("fork")
    done = 0
    with ctx.Pool(processes=a.workers, maxtasksperchild=1) as pool:
        for target, seed, secs, n in pool.imap_unordered(run_job, jobs, chunksize=1):
            done += 1
            print(f"  [{done}/{len(jobs)}] {target} seed {seed}: {n} forecasts, {secs:.0f}s "
                  f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"stage 1 done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Reading the work files; reproduction check (no losses)
# ---------------------------------------------------------------------------


ID_COLS = ["t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y"]


def read_csv_exact(path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def load_wide(work: Path, target: str, seeds) -> pd.DataFrame:
    """One frame per target: identifiers, y, mcsgarch and one hybrid_base column per base seed
    (``hyb_<seed>``). Raises if y, the identifiers or mcsGARCH differ between runs."""
    wide, mcs_dev = None, 0.0
    for s in seeds:
        f = read_csv_exact(work_file(work, target, s)).sort_values(["t", "minute"]).reset_index(drop=True)
        if wide is None:
            wide = f[ID_COLS + ["mcsgarch"]].copy()
        else:
            for c in ID_COLS:
                if not f[c].equals(wide[c]):
                    raise ValueError(f"{target}: column {c} differs between base seeds {seeds[0]} and {s}")
            mcs_dev = max(mcs_dev, float(np.max(np.abs(f["mcsgarch"] / wide["mcsgarch"] - 1))))
        wide[f"hyb_{s}"] = f["hybrid_base"].to_numpy()
    wide.attrs["mcsgarch_max_rel_diff_across_seeds"] = mcs_dev
    return wide


def reproduction_check(wide: pd.DataFrame, committed: pd.DataFrame, target: str, seed: int = HEADLINE_SEED) -> dict:
    """Compare the rerun of the headline base seed with the committed v2 forecasts (forecasts only)."""
    a = wide.set_index(["date", "minute"])
    b = committed.set_index(["date", "minute"])
    common = a.index.intersection(b.index)
    out = {"target": target, "rows_rerun": len(a), "rows_committed": len(b), "rows_matched": len(common)}
    pairs = {"y": ("y", "y"), "mcsgarch": ("mcsgarch", "mcsgarch"), "hybrid_base": (f"hyb_{seed}", "hybrid_base")}
    for name, (ca, cb) in pairs.items():
        x, y = a.loc[common, ca].to_numpy(float), b.loc[common, cb].to_numpy(float)
        out[f"{name}_n_identical"] = int(np.sum(x == y))
        out[f"{name}_max_abs_rel_diff"] = float(np.max(np.abs(x / y - 1)))
    out["mcsgarch_max_rel_diff_across_seeds"] = wide.attrs.get("mcsgarch_max_rel_diff_across_seeds", np.nan)
    return out


def stage_check(a):
    work = Path(a.work)
    rows = []
    for t in a.targets:
        if not work_file(work, t, HEADLINE_SEED).exists():
            print(f"{t}: no work file for seed {HEADLINE_SEED}")
            continue
        seeds = [s for s in [HEADLINE_SEED] + NEW_SEEDS if work_file(work, t, s).exists()]
        wide = load_wide(work, t, seeds)
        committed = read_csv_exact(HEADLINE_RUN / f"forecasts_{t}.csv")
        r = reproduction_check(wide, committed, t)
        rows.append(r)
        print(r, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 2: losses, DM tests, seed bootstrap, verdict
# ---------------------------------------------------------------------------


def ensemble_rows(wide: pd.DataFrame, seeds, groups: dict, target: str) -> list[dict]:
    """Per base seed and subset: QLIKE margin over mcsGARCH (as in scoreboard_*.csv: anchor means) and
    the one-sided DM test on daily average QLIKE (as in tests_*.csv)."""
    rows = []
    cols = [f"hyb_{s}" for s in seeds]
    for subset, mask in groups.items():
        sub = wide[mask]
        if sub["date"].nunique() < 10:
            continue
        L = daily_losses(sub, ["mcsgarch"] + cols, "qlike")
        q0 = float(np.mean(qlike(sub["y"], sub["mcsgarch"])))
        for s, c in zip(seeds, cols):
            q = float(np.mean(qlike(sub["y"], sub[c])))
            dm = diebold_mariano(L["mcsgarch"], L[c], alternative="greater")
            rows.append({"target": target, "subset": subset, "base_seed": s,
                         "role": "headline (reproduction)" if s == HEADLINE_SEED else "new",
                         "lgbm_seeds": " ".join(map(str, lgbm_seeds(s))), "days": dm["n"], "anchors": len(sub),
                         "qlike_mcsgarch": q0, "qlike_hybrid": q, "margin_pct": 100 * (q / q0 - 1),
                         "mean_daily_diff": dm["mean_diff"], "dm_stat": dm["stat"], "dm_p": dm["p"],
                         "share_days_hybrid_better": float((L[c] < L["mcsgarch"]).mean())})
    return rows


def seed_bootstrap(l_bench: np.ndarray, l_models: np.ndarray, reps: int = REPS, mean_block: float = MEAN_BLOCK,
                   rng: np.random.Generator | None = None, idx: np.ndarray | None = None,
                   ens: np.ndarray | None = None) -> dict:
    """Bootstrap over days and seeds for H0: the hybrid is not better than the benchmark.

    ``l_bench`` (n,) is the benchmark's daily loss, ``l_models`` (n x K) the daily loss of each of K
    ensembles of the hybrid. Each draw resamples days with the stationary bootstrap (mean block
    ``mean_block``) and picks ONE ensemble k at random; its mean gain d*_b = mean(l_bench - l_k) over
    the resampled days varies with both the days and the seeds. The observed gain is the grand mean
    dbar over days and ensembles.

    * ``p``: null-imposed one-sided p, (1 + #{d*_b - dbar >= dbar}) / (reps + 1);
    * ``p_percentile``: share of draws with no gain, (1 + #{d*_b <= 0}) / (reps + 1);
    * ``margin_q*``: percentiles of the draw's QLIKE margin, 100 (mean l_k / mean l_bench - 1).

    With K = 1 this is a plain stationary bootstrap of one model's daily loss differential.
    ``idx`` and ``ens`` can be passed to reuse the same draws.
    """
    l_bench = np.asarray(l_bench, float)
    l_models = np.asarray(l_models, float)
    if l_models.ndim == 1:
        l_models = l_models[:, None]
    n, K = l_models.shape
    rng = rng if rng is not None else np.random.default_rng(BOOT_SEED)
    if idx is None:
        idx = rf.stationary_bootstrap_indices(n, reps, mean_block, rng)
    reps = idx.shape[0]
    if ens is None:
        ens = rng.integers(0, K, reps)
    means = rf.bootstrap_means(np.column_stack([l_bench, l_models]), idx)      # reps x (K + 1)
    mb = means[:, 0]
    mk = means[np.arange(reps), 1 + ens]
    dstar = mb - mk
    dbar = float(l_bench.mean() - l_models.mean())
    margin = 100 * (mk / mb - 1)
    gains = l_bench.mean() - l_models.mean(axis=0)
    return {"n": n, "K": K, "reps": reps, "mean_block": mean_block, "dbar": dbar,
            "margin_pct_grand": 100 * (l_models.mean() / l_bench.mean() - 1),
            "p": float((1 + np.sum(dstar - dbar >= dbar)) / (reps + 1)),
            "p_percentile": float((1 + np.sum(dstar <= 0)) / (reps + 1)),
            "sd_draw": float(dstar.std(ddof=1)),
            "sd_between_ensembles": float(gains.std(ddof=1)) if K > 1 else 0.0,
            "margin_q025": float(np.quantile(margin, 0.025)), "margin_q500": float(np.quantile(margin, 0.5)),
            "margin_q950": float(np.quantile(margin, 0.95)), "margin_q975": float(np.quantile(margin, 0.975)),
            "share_draws_no_gain": float(np.mean(dstar <= 0))}


def summarize(ens: pd.DataFrame) -> pd.DataFrame:
    """Distribution over the NEW ensembles per target and subset, with the headline seed beside it."""
    rows = []
    for (t, sub), g in ens.groupby(["target", "subset"], sort=False):
        new = g[g["role"] == "new"]
        head = g[g["role"] != "new"]
        r = {"target": t, "subset": sub, "K": len(new), "days": int(new["days"].iloc[0]),
             "margin_min": new["margin_pct"].min(), "margin_median": new["margin_pct"].median(),
             "margin_max": new["margin_pct"].max(), "margin_mean": new["margin_pct"].mean(),
             "margin_sd": new["margin_pct"].std(ddof=1),
             "dm_p_max": new["dm_p"].max(), "dm_p_median": new["dm_p"].median(),
             "dm_stat_min": new["dm_stat"].min(),
             "n_margin_negative": int((new["margin_pct"] < 0).sum()), "n_p_below_alpha": int((new["dm_p"] < ALPHA).sum()),
             "worst_seed_by_margin": int(new.loc[new["margin_pct"].idxmax(), "base_seed"]),
             "worst_seed_by_p": int(new.loc[new["dm_p"].idxmax(), "base_seed"])}
        if len(head):
            r["headline_margin"] = float(head["margin_pct"].iloc[0])
            r["headline_dm_p"] = float(head["dm_p"].iloc[0])
            r["headline_rank_among_new_by_margin"] = int((new["margin_pct"] < r["headline_margin"]).sum()) + 1
        rows.append(r)
    return pd.DataFrame(rows)


def rule_label(margins, ps, alpha: float = ALPHA) -> str:
    """The registered rule for one cell: every ensemble better (margin < 0) with one-sided DM p < alpha."""
    margins, ps = np.asarray(margins, float), np.asarray(ps, float)
    if np.any(~np.isfinite(margins)) or np.any(~np.isfinite(ps)):
        return "not computed"
    if np.all(margins < 0) and np.all(ps < alpha):
        return "holds"
    if np.any(margins >= 0):
        return "sign flips"
    return "weakened"


def verdict(ens: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    """Primary verdict (next15 and next30, all days) and the registered secondary checks."""
    new = ens[ens["role"] == "new"]
    rows = []
    for t in TARGETS:
        for sub in ("all", "expiry"):
            g = new[(new.target == t) & (new.subset == sub)]
            lab = rule_label(g["margin_pct"], g["dm_p"]) if len(g) else "not computed"
            b = boot[(boot.target == t) & (boot.subset == sub) & (boot.mean_block == MEAN_BLOCK)]
            bp = float(b["p"].iloc[0]) if len(b) else np.nan
            kind = "primary" if (t in VERDICT_TARGETS and sub == "all") else "secondary"
            rows.append({"check": f"{t}, {sub} days", "kind": kind, "rule": "every ensemble: margin < 0 and DM p < 0.05",
                         "K": len(g), "worst_margin": g["margin_pct"].max() if len(g) else np.nan,
                         "max_dm_p": g["dm_p"].max() if len(g) else np.nan, "result": lab,
                         "seed_bootstrap_p": bp, "seed_bootstrap_below_alpha": bool(bp < ALPHA)})
    v = pd.DataFrame(rows)
    prim = v[v["kind"] == "primary"]
    if (prim["result"] == "not computed").any():
        overall = "NOT COMPUTED"
    elif (prim["result"] == "holds").all():
        overall = "HOLDS"
    elif (prim["result"] == "sign flips").any():
        overall = "FAILS (sign flips)"
    else:
        overall = "FAILS (weakened)"
    v = pd.concat([v, pd.DataFrame([{"check": "PRIMARY VERDICT (next15 and next30, all days)", "kind": "primary",
                                     "result": overall}])], ignore_index=True)
    return v


def daily_loss_table(wide: pd.DataFrame, seeds) -> pd.DataFrame:
    cols = [f"hyb_{s}" for s in seeds]
    L = daily_losses(wide, ["mcsgarch"] + cols, "qlike")
    meta = wide.groupby("date").agg(is_expiry=("is_expiry", "first"), anchors=("y", "size"))
    out = meta.join(L.rename(columns={"mcsgarch": "qlike_mcsgarch", **{c: f"qlike_hybrid_s{c[4:]}" for c in cols}}))
    out.index.name = "date"
    return out


def stage_analyze(a):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [HEADLINE_SEED] + NEW_SEEDS
    assert seeds_disjoint(NEW_SEEDS), "new ensembles share LightGBM seeds"
    t0 = time.time()
    ens_rows, boot_rows, rep_rows = [], [], []
    print("[REAL DATA] P99 seed bootstrap, stage 2 (docs/prereg/P99_SEED_BOOTSTRAP.md)")
    for ti, t in enumerate(TARGETS):
        wide = load_wide(work, t, seeds)
        committed = read_csv_exact(HEADLINE_RUN / f"forecasts_{t}.csv")
        rep = reproduction_check(wide, committed, t)
        groups = subgroups(wide, holdout_start=HOLDOUT_START, auction_start=CLOSING_AUCTION_START)
        groups = {k: groups[k] for k in SUBSETS if k in groups}
        rows = ensemble_rows(wide, seeds, groups, t)
        # the headline seed's statistics against the committed scoreboard and tests
        sb = pd.read_csv(HEADLINE_RUN / f"scoreboard_{t}.csv")
        ts = pd.read_csv(HEADLINE_RUN / f"tests_{t}.csv")
        for sub in ("all", "expiry"):
            r = next(x for x in rows if x["subset"] == sub and x["base_seed"] == HEADLINE_SEED)
            m0 = float(sb[(sb.group == sub) & (sb.model == "hybrid_base")]["qlike_vs_mcsgarch_pct"].iloc[0])
            p0 = float(ts[(ts.subset == sub) & (ts.loss == "qlike") & (ts.comparison == "mcsgarch vs hybrid_base")]["p"].iloc[0])
            rep[f"margin_{sub}_rerun"], rep[f"margin_{sub}_committed"] = r["margin_pct"], m0
            rep[f"dm_p_{sub}_rerun"], rep[f"dm_p_{sub}_committed"] = r["dm_p"], p0
        rep_rows.append(rep)
        ens_rows += rows
        daily_loss_table(wide, seeds).to_csv(out / f"daily_losses_{t}.csv")
        for si, (sub, mask) in enumerate(groups.items()):
            L = daily_losses(wide[mask], ["mcsgarch"] + [f"hyb_{s}" for s in seeds], "qlike")
            lb, lnew = L["mcsgarch"].to_numpy(), L[[f"hyb_{s}" for s in NEW_SEEDS]].to_numpy()
            for bi, blk in enumerate((MEAN_BLOCK,) + SENS_BLOCKS):
                rng = np.random.default_rng([BOOT_SEED, ti, si, bi])
                idx = rf.stationary_bootstrap_indices(len(lb), a.reps, blk, rng)
                b = seed_bootstrap(lb, lnew, mean_block=blk, rng=rng, idx=idx)
                # the same day draws for the headline ensemble alone (seed noise left out)
                h = seed_bootstrap(lb, L[f"hyb_{HEADLINE_SEED}"].to_numpy(), mean_block=blk, idx=idx,
                                   ens=np.zeros(len(idx), int))
                boot_rows.append({"target": t, "subset": sub, "registered": blk == MEAN_BLOCK and sub in ("all", "expiry"),
                                  **b, "p_day_only_headline": h["p"], "margin_q950_day_only_headline": h["margin_q950"]})
        print(f"{t}: reproduction {rep['hybrid_base_n_identical']}/{rep['rows_matched']} hybrid forecasts identical "
              f"(max rel diff {rep['hybrid_base_max_abs_rel_diff']:.2e}); mcsGARCH across 21 runs max rel diff "
              f"{rep['mcsgarch_max_rel_diff_across_seeds']:.2e} ({time.time() - t0:.0f}s)", flush=True)
    ens = pd.DataFrame(ens_rows)
    boot = pd.DataFrame(boot_rows)
    summ = summarize(ens)
    bmain = boot[boot.mean_block == MEAN_BLOCK].set_index(["target", "subset"])
    summ["seed_bootstrap_p"] = [bmain.loc[(r.target, r.subset), "p"] for r in summ.itertuples()]
    summ["seed_bootstrap_p_percentile"] = [bmain.loc[(r.target, r.subset), "p_percentile"] for r in summ.itertuples()]
    ver = verdict(ens, boot)
    ens.to_csv(out / "ensembles.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    boot.to_csv(out / "bootstrap.csv", index=False)
    ver.to_csv(out / "verdict.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(out / "reproduction_check.csv", index=False)
    timing = pd.read_csv(work / "timing.csv")
    timing = timing.drop_duplicates(["target", "seed"], keep="last").sort_values(["seed", "target"])
    timing[["target", "seed", "seconds", "rows"]].to_csv(out / "timing.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nPer target and subset, over the K = 20 new 5-seed ensembles (margin = QLIKE % vs mcsGARCH; "
          "negative = hybrid better):")
    print(summ[["target", "subset", "days", "margin_min", "margin_median", "margin_max", "margin_sd", "dm_p_max",
                "n_p_below_alpha", "headline_margin", "headline_dm_p", "seed_bootstrap_p"]].to_string(index=False))
    print("\nSeed bootstrap (all blocks):")
    print(boot[["target", "subset", "mean_block", "p", "p_percentile", "margin_q025", "margin_q950", "margin_q975",
                "sd_draw", "sd_between_ensembles", "p_day_only_headline"]].to_string(index=False))
    print("\nVerdict:")
    print(ver[["check", "kind", "K", "worst_margin", "max_dm_p", "result", "seed_bootstrap_p"]].to_string(index=False))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    import lightgbm, arch, scipy  # noqa: E401
    head_info = json.load(open(HEADLINE_RUN / "run_info.json"))
    json.dump({"source": "REAL DATA: " + head_info["source"].removeprefix("REAL DATA: "),
               "study": "P99 seed bootstrap of hybrid_base (docs/prereg/P99_SEED_BOOTSTRAP.md)",
               "plan_commit": a.plan_commit, "analysis_commit": commit,
               "commands": ["PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py run --raw data/raw --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py analyze"],
               "headline_run": str(HEADLINE_RUN.relative_to(ROOT)), "end": END, "days": head_info["days"],
               "models_fitted": MODELS + ["(internal) mcsgarch_iv, for two hybrid features"],
               "pipeline_settings": PIPELINE_SETTINGS, "headline_seed": HEADLINE_SEED,
               "new_base_seeds": NEW_SEEDS, "lgbm_seeds": {str(s): lgbm_seeds(s) for s in seeds},
               "seed_note": "the base seed also seeds the tuning pass that picks the number of boosting rounds",
               "subsets": SUBSETS, "registered_subsets": ["all", "expiry"], "alpha": ALPHA,
               "bootstrap": {"reps": a.reps, "mean_block_days": MEAN_BLOCK, "sensitivity_blocks": list(SENS_BLOCKS),
                             "rng": f"numpy default_rng([{BOOT_SEED}, target index, subset index, block index])",
                             "draw": "stationary bootstrap of days and one of the K ensembles drawn uniformly per draw"},
               "dm_test": head_info["dm_test"], "seconds_stage1_jobs": float(timing["seconds"].sum()),
               "python": sys.version.split()[0],
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
                            "arch": arch.__version__, "scipy": scipy.__version__}},
              open(out / "run_info.json", "w"), indent=2, default=str)
    print(f"\nwritten to {out} ({time.time() - t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("run", "check", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--work", default=str(ROOT / "data" / "processed" / "seed_bootstrap"))
        p.add_argument("--targets", nargs="+", default=TARGETS)
        if name == "run":
            p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
            p.add_argument("--end", default=END)
            p.add_argument("--seeds", nargs="+", type=int, default=[HEADLINE_SEED] + NEW_SEEDS)
            p.add_argument("--workers", type=int, default=2)
        if name == "analyze":
            p.add_argument("--out", default=str(ROOT / "results" / "seed_bootstrap"))
            p.add_argument("--reps", type=int, default=REPS)
            p.add_argument("--plan-commit", default=None)
    a = ap.parse_args(argv)
    if a.stage == "run":
        stage_run(a)
    elif a.stage == "check":
        stage_check(a)
    else:
        stage_analyze(a)


if __name__ == "__main__":
    main()
