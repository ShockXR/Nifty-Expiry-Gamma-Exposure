"""P110: stress test of P106's three claims about the forecasting hybrid's gain, on Bank Nifty and Fin Nifty.

Registered plan: docs/prereg/P110_GAIN_STRESS_TEST.md. Results: results/p110_gain_stress_test/.
Write-up: docs/P110_GAIN_STRESS_TEST.md. Done by an AI agent (Claude) for Shreyan Thanki.

P106 refitted the hybrid (mcsGARCH times a LightGBM correction, ``hybrid_base``) without each of five input
groups. On Nifty it used five seed sets; on Bank Nifty and Fin Nifty only seed set 7, and it never ran the
``mcsinfo`` variant (only what mcsGARCH itself sees) there. Its "adds nothing" claims rest on failures to reject.
This script

1. refits, on Bank Nifty and Fin Nifty at 15 and 30 minutes, the full hybrid, the five leave-one-group-out
   variants and ``mcsinfo`` with seed sets 7, 101, 102, 103 and 104 (P106's code path, nothing retuned);
2. tests P106's three claims with the registered families: the realized history and the time of day carry the
   gain (one-sided Diebold-Mariano), the option-implied, VIX and calendar groups add nothing (two one-sided
   tests of equivalence, margin 1 point of mcsGARCH's QLIKE), and the ``mcsinfo`` hybrid gets the gain (it beats
   mcsGARCH, and it is non-inferior to the full hybrid within 1 point);
3. adds secondary, robustness, placebo and exploratory rows (pooled two- and three-index versions with
   Driscoll-Kraay errors, a 0.5-point margin, seed set 7 alone, all five seed sets, a delta-method version,
   without 4 and 5 June 2024, without the 1% of days with the largest mcsGARCH loss, a Nifty re-analysis).

Stages:

    # 1. forecasts (REAL DATA), one invocation per index, one process
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p110_gain_stress_test.py run --index BANKNIFTY
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p110_gain_stress_test.py run --index FINNIFTY
    # 2. losses, tests, tables, verdicts
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p110_gain_stress_test.py analyze --plan-commit <hash>

``--synthetic`` (run and analyze) uses three simulated markets and small settings, for a dry run of every step;
its output is labelled SYNTHETIC. Work files go to --work (default data/processed/p110, git-ignored); a finished
job is not rerun. Tests: tests/test_p110_gain_stress_test.py (synthetic data only).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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

from niftygex.evaluation import diebold_mariano  # noqa: E402
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel  # noqa: E402
from niftygex.implied import build_implied_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# P106's run and analysis code (groups, drop sets, configuration, work files, contribution, ratio variance,
# period test, share interval) and P107's Driscoll-Kraay pooled means, so that nothing here is a second version.
gd = _load("p106_gain_decomposition", ROOT / "experiments" / "p106_gain_decomposition.py")
pt = _load("p107_pooled_test", ROOT / "experiments" / "p107_pooled_test.py")

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

STUDY_INDEXES = ["BANKNIFTY", "FINNIFTY"]
REFERENCE_INDEX = "NIFTY"            # P106's committed daily losses; nothing is refitted on Nifty
VARIANTS = ["full"] + list(gd.PRIMARY_GROUPS) + ["mcsinfo"]   # 7: full, implied, vix, intraday, calendar, realized, mcsinfo
CARRIERS = ["realized", "intraday"]
NULL_GROUPS = ["implied", "vix", "calendar"]
TARGETS = ["next15", "next30"]
HEADLINE_SEED = gd.HEADLINE_SEED     # 7
NEW_SEEDS = list(gd.REPEAT_SEEDS)    # 101, 102, 103, 104
ALL_SEEDS = [HEADLINE_SEED] + NEW_SEEDS
MARGIN = 1.0                         # equivalence and non-inferiority margin, points of mcsGARCH's QLIKE
MARGIN_SENS = 0.5                    # sensitivity margin (secondary)
ALPHA = 0.05
REPS = 10000
MEAN_BLOCK = 10.0
BOOT_SEED = 110
INFLUENCE_DATES = ["2024-06-04", "2024-06-05"]   # general-election result day and the day after
TOP_SHARE = 0.01                     # robustness: drop this share of days with the largest mcsGARCH daily QLIKE
CHUNK = 7                            # all seven variants in one walk-forward job (they share the mcsGARCH fits)
P106_RESULTS = ROOT / "results" / "p106_gain_decomposition"
LABEL = {"BANKNIFTY": "Bank Nifty", "FINNIFTY": "Fin Nifty", "NIFTY": "Nifty",
         "SYNTHA": "synthetic A", "SYNTHB": "synthetic B", "SYNTHC": "synthetic C (stands in for Nifty)"}
TARGET_LABEL = {"next15": "next 15 min", "next30": "next 30 min"}
SYNTH = dict(n_days=200, start="2024-03-01", market_seeds={"SYNTHA": 3, "SYNTHB": 4, "SYNTHC": 5}, min_train=60,
             block=20, n_seeds=1, targets=["next15", "next30"], seeds=[7, 101], horizon_step=45)
SYNTH_STUDY = ["SYNTHA", "SYNTHB"]
SYNTH_REFERENCE = "SYNTHC"

# The primary rows for one index and target: (family, kind, variant)
PRIMARY_SPECS = ([("F1", "carrier", v) for v in CARRIERS] + [("F2", "tost", v) for v in NULL_GROUPS]
                 + [("F3", "beats", "mcsinfo"), ("F3", "ni", "mcsinfo")])
KIND_SUFFIX = {"carrier": "", "tost": "", "adds": "", "beats": "_beats_mcsgarch", "ni": "_noninferior",
               "full_beats": "_full_beats_mcsinfo"}


def primary_id(index: str, target: str, fam: str, kind: str, v: str) -> str:
    return f"{fam}_{index}_{target}_{v}{KIND_SUFFIX[kind]}"


# ---------------------------------------------------------------------------
# Job plan and stage 1 (forecasts)
# ---------------------------------------------------------------------------


def is_synth(index: str) -> bool:
    return index.startswith("SYNTH")


def job_plan(index: str, seeds=None, targets=None, variants=None, chunk: int = CHUNK) -> list[tuple]:
    """(index, target, seed, variants): seed set 7 first, then 101 to 104; within a seed set 15 then 30 minutes;
    all seven variants in one job. Optional filters keep the order."""
    seeds_l = SYNTH["seeds"] if is_synth(index) else ALL_SEEDS
    targets_l = SYNTH["targets"] if is_synth(index) else TARGETS
    plan = []
    for s in seeds_l:
        for t in targets_l:
            plan += [(index, t, s, c) for c in gd.chunks(VARIANTS, chunk)]
    out = []
    for (ix, t, s, vs) in plan:
        if seeds is not None and s not in seeds:
            continue
        if targets is not None and t not in targets:
            continue
        if variants is not None:
            vs = tuple(v for v in vs if v in variants)
        if vs:
            out.append((ix, t, s, vs))
    return out


def synthetic_market(index: str):
    from niftygex.simulate import SimConfig, simulate_market
    return simulate_market(SimConfig(n_days=SYNTH["n_days"], start=SYNTH["start"],
                                     seed=SYNTH["market_seeds"][index], kappa=0.0))


def synthetic_config(seed, eval_minutes, sets, variants):
    return gd.make_config(seed, eval_minutes, sets, variants, min_train=SYNTH["min_train"], block=SYNTH["block"],
                          mcs_refit_every=SYNTH["block"], n_seeds=SYNTH["n_seeds"])


def stage_run(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    index = a.index
    synth = is_synth(index)
    plan = job_plan(index, seeds=a.seeds, targets=a.targets, variants=a.variants, chunk=a.chunk)
    jobs = [j for j in plan if not gd.job_done(work, j)]
    label = "SYNTHETIC (simulated market, not real data)" if synth else "REAL DATA"
    print(f"[{label}] P110 stress test, stage 1, {index}: {len(jobs)} jobs to run "
          f"({len(plan) - len(jobs)} already in {work}); one process", flush=True)
    if not jobs:
        return
    t0 = time.time()
    if synth:
        m = synthetic_market(index)
        gd._STATE["config_fn"] = synthetic_config
        targets = {"next15": dict(horizon=15, step=SYNTH["horizon_step"]),
                   "next30": dict(horizon=30, step=SYNTH["horizon_step"])}
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
    gd._STATE.update(market=m, datasets=datasets, sets=sets, eval_minutes={} if synth else dict(gd.rp.EVAL_MINUTES))
    if not (work / "timing.csv").exists():
        (work / "timing.csv").write_text("index,target,seed,variants,n_variants,seconds,rows,pid\n")
    for k, j in enumerate(jobs, 1):
        ix, target, seed, vs, secs, n = gd.run_job(j + (str(work),))
        print(f"  [{k}/{len(jobs)}] {ix} {target} seed {seed} {'+'.join(vs)}: {n} forecasts, {secs:.0f}s "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"stage 1 ({index}) done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Statistics (pure functions, tested on synthetic data)
# ---------------------------------------------------------------------------


def dm_parts(d) -> dict:
    """The pieces of the project's DM test (niftygex.evaluation.diebold_mariano) for a daily series d: its mean,
    the Newey-West standard error of the mean (Bartlett kernel, bandwidth floor(4 (n/100)^(2/9))), the
    Harvey-Leybourne-Newbold factor for one-step forecasts and the Student t degrees of freedom."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    return {"mean": float(d.mean()), "se": float(np.sqrt(max(lrv, 1e-300) / n)), "hln": float(np.sqrt((n - 1) / n)),
            "df": max(n - 1, 1), "n": int(n)}


def _tost_from(mean: float, se: float, hln: float, df: int, delta: float) -> dict:
    """Two one-sided tests of H0: |mu| >= delta against H1: -delta < mu < delta, each a t test with the HLN factor.
    p = the larger of the two one-sided p-values; the statistic reported is that of the binding (larger-p) side.
    The 90% interval is the one whose inclusion in (-delta, delta) is equivalent to rejection at 5%."""
    t_up = (mean - delta) / se * hln         # H0: mu >= delta; reject for small values
    t_lo = (mean + delta) / se * hln         # H0: mu <= -delta; reject for large values
    p_up = float(student_t.cdf(t_up, df))
    p_lo = float(student_t.sf(t_lo, df))
    half = float(student_t.ppf(0.95, df) * se / hln)
    up_binds = p_up >= p_lo
    return {"estimate": mean, "se": se, "stat": t_up if up_binds else t_lo, "p": max(p_up, p_lo), "p_upper": p_up,
            "p_lower": p_lo, "ci90_low": mean - half, "ci90_high": mean + half, "bound": abs(mean) + half,
            "df": df, "binding": "upper" if up_binds else "lower"}


def tost(d, delta: float = MARGIN) -> dict:
    """TOST on the mean of a daily series d (points), with the DM test's variance and small-sample treatment.
    Equals diebold_mariano(d - delta, 0, 'less') and diebold_mariano(d + delta, 0, 'greater') (tested)."""
    q = dm_parts(d)
    return {**_tost_from(q["mean"], q["se"], q["hln"], q["df"], delta), "n": q["n"]}


def noninferiority(d, delta: float = MARGIN) -> dict:
    """One-sided test of H0: mu >= delta against H1: mu < delta (the upper half of the TOST)."""
    r = tost(d, delta)
    return {"estimate": r["estimate"], "se": r["se"], "stat": (r["estimate"] - delta) / r["se"] * dm_parts(d)["hln"],
            "p": r["p_upper"], "ci95_upper": r["ci90_high"], "n": r["n"], "df": r["df"]}


def tost_delta_method(num, den, delta: float = MARGIN) -> dict:
    """TOST on the ratio r = 100 mean(num) / mean(den) with the delta-method variance (P106's ratio_var:
    Newey-West variance of the influence series, the DM bandwidth), HLN factor and Student t(n - 1)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    r, v = gd.ratio_var(num, den)
    return {**_tost_from(r, float(np.sqrt(v)), float(np.sqrt((n - 1) / n)), max(n - 1, 1), delta), "n": n}


def points_series(D: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """Daily 100 x (loss of a - loss of b) / mean daily loss of mcsGARCH: its mean is the difference in points."""
    return 100.0 * (D[a] - D[b]).to_numpy(float) / D["mcsgarch"].mean()


def row_estimate(D: pd.DataFrame, kind: str, v: str) -> float:
    """The registered estimate of a row, in points of mcsGARCH's mean daily QLIKE.
    carrier, adds, tost: removing group v raises the loss by this much (positive: the group helps);
    beats: the variant's margin over mcsGARCH (negative: the variant is better);
    ni, full_beats: the variant's extra loss over the full hybrid (positive: the full hybrid is better)."""
    m = D["mcsgarch"].mean()
    if kind in ("carrier", "adds", "tost"):
        return 100.0 * (D[v].mean() - D["full"].mean()) / m
    if kind == "beats":
        return 100.0 * (D[v].mean() - m) / m
    if kind in ("ni", "full_beats"):
        return 100.0 * (D[v].mean() - D["full"].mean()) / m
    raise ValueError(kind)


def row_test(D: pd.DataFrame, kind: str, v: str, delta: float = MARGIN, method: str = "dm") -> dict:
    """Estimate, statistic and p of one row on a day frame (columns mcsgarch, full, the variants)."""
    est = row_estimate(D, kind, v)
    if method == "delta":
        if kind not in ("tost", "ni"):
            raise ValueError("delta method only for tost and ni rows")
        r = tost_delta_method((D[v] - D["full"]).to_numpy(float), D["mcsgarch"].to_numpy(float), delta)
        if kind == "ni":
            st = (r["estimate"] - delta) / r["se"] * np.sqrt((r["n"] - 1) / r["n"])
            return {"estimate": r["estimate"], "stat": float(st), "p": r["p_upper"], "n": r["n"], "se": r["se"]}
        return {**r, "p": r["p"], "stat": r["stat"]}
    if kind in ("carrier", "adds"):
        dm = diebold_mariano(D[v], D["full"], alternative="greater")
        return {"estimate": est, "stat": dm["stat"], "p": dm["p"], "n": dm["n"]}
    if kind == "full_beats":
        dm = diebold_mariano(D[v], D["full"], alternative="greater")
        return {"estimate": est, "stat": dm["stat"], "p": dm["p"], "n": dm["n"]}
    if kind == "beats":
        dm = diebold_mariano(D["mcsgarch"], D[v], alternative="greater")
        return {"estimate": est, "stat": dm["stat"], "p": dm["p"], "n": dm["n"]}
    if kind == "tost":
        r = tost(points_series(D, v, "full"), delta)
        return {**r, "estimate": est}
    if kind == "ni":
        r = noninferiority(points_series(D, v, "full"), delta)
        return {**r, "estimate": est}
    raise ValueError(kind)


def agrees(kind: str, est: float, delta: float = MARGIN) -> bool:
    """Whether one seed set's estimate points the way the row's alternative hypothesis does."""
    if kind in ("carrier", "adds", "full_beats"):
        return est > 0
    if kind == "tost":
        return -delta < est < delta
    if kind == "ni":
        return est < delta
    if kind == "beats":
        return est < 0
    raise ValueError(kind)


VERDICT_PASS = {"carrier": "supported", "tost": "equivalent", "ni": "non-inferior", "beats": "supported"}
VERDICT_FAIL = {"carrier": "not supported", "tost": "not shown", "ni": "not shown", "beats": "not supported"}


def row_verdict(kind: str, p_adj: float, n_agree: int, k: int, alpha: float = ALPHA) -> str:
    """The registered verdict of a primary row: passes if the Holm-adjusted p is below alpha and every seed set
    agrees; seed-fragile if the p passes but a seed set disagrees."""
    if not np.isfinite(p_adj):
        return "not computed"
    if p_adj >= alpha:
        return VERDICT_FAIL[kind]
    return VERDICT_PASS[kind] if n_agree == k else "seed-fragile"


def pooled_test(X: np.ndarray, kind: str, delta: float = MARGIN) -> dict:
    """Equal-weight pooled version of a row over the columns (indices) of X, a balanced (dates x indices) panel of
    daily series in points, with P107's Driscoll-Kraay covariance (index_means_dk), the HLN factor and Student
    t(T - 1). kind: carrier/adds/full_beats (H1: mean > 0), beats (H1: mean < 0), tost, ni."""
    r = pt.index_means_dk(X)
    N = X.shape[1]
    a = np.ones(N) / N
    est = float(a @ r["means"])
    se = float(np.sqrt(max(a @ r["cov"] @ a, 1e-300)))
    T = r["T"]
    hln, df = float(np.sqrt((T - 1) / T)), T - 1
    out = {"estimate": est, "se": se, "T": T, "lag": r["lag"], "per_index": [float(x) for x in r["means"]]}
    if kind in ("carrier", "adds", "full_beats"):
        st = est / se * hln
        return {**out, "stat": st, "p": float(student_t.sf(st, df))}
    if kind == "beats":
        st = -est / se * hln
        return {**out, "stat": st, "p": float(student_t.sf(st, df))}
    tt = _tost_from(est, se, hln, df, delta)
    if kind == "tost":
        return {**out, **tt}
    if kind == "ni":
        st = (est - delta) / se * hln
        return {**out, "stat": st, "p": tt["p_upper"], "ci95_upper": tt["ci90_high"]}
    raise ValueError(kind)


def pooled_series(frames: list, kind: str, v: str, dates) -> np.ndarray:
    """(dates x indices) panel of each index's daily series in points of its own mcsGARCH QLIKE over ``dates``."""
    cols = []
    for D in frames:
        d = D.loc[dates]
        if kind == "beats":
            cols.append(points_series(d, v, "mcsgarch"))
        else:
            cols.append(points_series(d, v, "full"))
    return np.column_stack(cols)


def seed_average(frames: list) -> pd.DataFrame:
    """Daily QLIKE averaged over seed sets (mcsGARCH and mcsgarch_iv are deterministic: checked equal)."""
    base = frames[0]
    for f in frames[1:]:
        for c in ("mcsgarch", "mcsgarch_iv"):
            if not np.array_equal(f[c].to_numpy(), base[c].to_numpy()) or not f.index.equals(base.index):
                raise ValueError(f"{c} or the days differ between seed sets")
    out = base.copy()
    for c in [c for c in VARIANTS if c in base]:
        out[c] = sum(f[c] for f in frames) / len(frames)
    return out


def top_loss_days(D: pd.DataFrame, share: float = TOP_SHARE) -> np.ndarray:
    """Mask that is False on the floor(share x days) days with the largest mcsGARCH daily QLIKE."""
    k = int(np.floor(share * len(D)))
    keep = np.ones(len(D), bool)
    if k > 0:
        keep[np.argsort(-D["mcsgarch"].to_numpy(float), kind="stable")[:k]] = False
    return keep


def date_mask(D: pd.DataFrame, dates) -> np.ndarray:
    """False on the listed dates."""
    idx = pd.to_datetime(D.index)
    return ~idx.isin(pd.to_datetime(list(dates)))


def claim_verdicts(rows: pd.DataFrame, index: str, targets) -> dict:
    """The registered claim-level verdicts for one index (Section 5 of the plan), from the primary verdicts and the
    S1 and S2 rows (columns test_id, verdict, p_adjusted, seed_sets_agreeing, seed_sets)."""
    r = rows.set_index("test_id")

    def get(fam, t, v, kind):
        return r.loc[primary_id(index, t, fam, kind, v)]

    # claim 1: realized history and time of day carry the gain
    ok = {v: [get("F1", t, v, "carrier")["verdict"] == "supported" for t in targets] for v in CARRIERS}
    if all(all(x) for x in ok.values()):
        c1 = "survives"
    elif any(not any(x) for x in ok.values()):
        c1 = "does not survive"
    else:
        c1 = "partly survives"
    # claim 2: the option-implied, VIX and calendar groups add nothing
    cells = {}
    for v in NULL_GROUPS:
        for t in targets:
            eq = get("F2", t, v, "tost")["verdict"] == "equivalent"
            s1 = get("S1", t, v, "adds")
            adds = bool(s1["p_adjusted"] < ALPHA and s1["seed_sets_agreeing"] == s1["seed_sets"])
            cells[(v, t)] = ("contradicted" if adds and not eq else "adds a little (under the margin)" if adds
                             else "shown" if eq else "not shown")
    vals = list(cells.values())
    if "contradicted" in vals:
        c2 = "contradicted"
    elif "adds a little (under the margin)" in vals:
        c2 = "qualified: a group adds a little, under the margin"
    elif all(x == "shown" for x in vals):
        c2 = "survives (shown by equivalence)"
    else:
        c2 = "survives only weakly (no contribution found, equivalence not shown for all rows)"
    # claim 3: the mcsinfo hybrid gets the whole gain
    beats = [get("F3", t, "mcsinfo", "beats")["verdict"] == "supported" for t in targets]
    ni = [get("F3", t, "mcsinfo", "ni")["verdict"] == "non-inferior" for t in targets]
    lacks = []
    for t in targets:
        s2 = get("S2", t, "mcsinfo", "full_beats")
        lacks.append(bool(s2["p_adjusted"] < ALPHA and s2["seed_sets_agreeing"] == s2["seed_sets"]))
    if not any(beats) or any(lk and not n for lk, n in zip(lacks, ni)):
        c3 = "contradicted"
    elif all(beats) and all(ni) and not any(lacks):
        c3 = "survives"
    else:
        c3 = "partly survives"
    overall = "all three survive" if (c1 == "survives" and c2.startswith("survives") and c3 == "survives") else \
        "not all three survive"
    return {"index": index, "claim1_carriers": c1, "claim2_adds_nothing": c2, "claim3_mcsinfo": c3,
            "overall": overall, **{f"claim2_{v}_{t}": c for (v, t), c in cells.items()},
            **{f"claim3_{t}": f"beats mcsGARCH: {b}; non-inferior: {n}; full significantly better: {lk}"
               for t, b, n, lk in zip(targets, beats, ni, lacks)}}


# ---------------------------------------------------------------------------
# Stage 2: analysis
# ---------------------------------------------------------------------------

DM_METHOD = "Diebold-Mariano on daily mean QLIKE, Newey-West, HLN factor, Student t(n-1)"
TOST_METHOD = ("two one-sided tests (TOST) on the daily difference in points of mcsGARCH's QLIKE, margin {d} point(s), "
               "DM variance (Newey-West, HLN factor, Student t(n-1)); p = larger one-sided p")
NI_METHOD = ("non-inferiority test (one-sided) on the daily difference in points of mcsGARCH's QLIKE, margin {d} "
             "point(s), DM variance (Newey-West, HLN factor, Student t(n-1))")
TOST_DELTA = ("TOST on the ratio 100 mean(variant - full) / mean(mcsGARCH), delta-method Newey-West variance, HLN factor, "
              "Student t(n-1), margin {d} point(s)")
NI_DELTA = "non-inferiority on the same ratio, delta-method Newey-West variance, HLN factor, Student t(n-1), margin {d}"
POOL_METHOD = ("equal-weight mean over indices of each index's daily series in points of its own mcsGARCH QLIKE; "
               "Driscoll-Kraay covariance (P107 index_means_dk), HLN factor, Student t(T-1)")
DELTA_PERIOD = "difference of period ratios, delta method with Newey-West variance within each period, Student t(n-2)"

HYP = {
    "carrier": "Removing the {g} group from the hybrid raises its mean QLIKE ({t}). Estimate: the rise, points of mcsGARCH's QLIKE.",
    "adds": "Removing the {g} group from the hybrid raises its mean QLIKE, i.e. the group adds something ({t}). "
            "Estimate: the rise, points of mcsGARCH's QLIKE.",
    "tost": "The {g} group adds nothing: removing it changes the hybrid's mean QLIKE by less than {d} point(s) of "
            "mcsGARCH's QLIKE in either direction ({t}). Estimate: the change, points.",
    "beats": "The mcsinfo hybrid (only what mcsGARCH sees) has lower mean QLIKE than mcsGARCH ({t}). Estimate: its "
             "margin over mcsGARCH, points (negative = better).",
    "ni": "The mcsinfo hybrid is non-inferior to the full hybrid: its mean QLIKE is less than {d} point(s) of "
          "mcsGARCH's QLIKE above the full hybrid's ({t}). Estimate: its extra loss, points.",
    "full_beats": "The full hybrid has lower mean QLIKE than the mcsinfo hybrid, i.e. the information mcsGARCH lacks "
                  "(option market, VIX, calendar except the expiry flag) adds something ({t}). Estimate: points.",
}
SIDED = {"carrier": "one-sided (removal raises loss)", "adds": "one-sided (removal raises loss)",
         "tost": "two one-sided tests (equivalence)", "beats": "one-sided (variant better than mcsGARCH)",
         "ni": "one-sided (non-inferiority)", "full_beats": "one-sided (full hybrid better)"}


def method_text(kind: str, delta: float, method: str = "dm") -> str:
    if method == "delta":
        return (TOST_DELTA if kind == "tost" else NI_DELTA).format(d=delta)
    if kind == "tost":
        return TOST_METHOD.format(d=delta)
    if kind == "ni":
        return NI_METHOD.format(d=delta)
    return DM_METHOD


def hyp_text(kind: str, v: str, t: str, delta: float = MARGIN, extra: str = "") -> str:
    return HYP[kind].format(g=gd.GROUP_LABELS.get(v, v), t=f"{TARGET_LABEL[t]}{extra}", d=delta)


def sample_text(index: str, t: str, D: pd.DataFrame, what: str) -> str:
    d0, d1 = pd.Timestamp(D.index.min()).date(), pd.Timestamp(D.index.max()).date()
    return f"{LABEL.get(index, index)}, {TARGET_LABEL[t]}, {len(D)} test days {d0} to {d1}, {what}"


def load_index_frames(work: Path, index: str, targets, seeds) -> dict:
    """(target, seed) -> day frame of one index (daily QLIKE of mcsgarch, mcsgarch_iv and the 7 variants)."""
    out = {}
    for t in targets:
        for s in seeds:
            if not all(gd.work_file(work, index, t, s, v).exists() for v in VARIANTS):
                continue
            wide = gd.load_wide(work, index, t, s, VARIANTS)
            D = gd.day_frame(wide, VARIANTS)
            D.index = pd.to_datetime(D.index)
            out[(t, s)] = D
    return out


def load_reference_frames(work: Path, synth: bool, targets, seeds) -> dict:
    """Nifty: P106's committed daily losses (no refit). Synthetic: a third simulated market's work files."""
    if synth:
        return load_index_frames(work, SYNTH_REFERENCE, targets, seeds)
    out = {}
    for t in targets:
        for s in seeds:
            p = P106_RESULTS / f"daily_losses_nifty_{t}_s{s}.csv"
            if p.exists():
                D = pd.read_csv(p, index_col="date", float_precision="round_trip")
                D.index = pd.to_datetime(D.index)
                out[(t, s)] = D
    return out


def stage_analyze(a):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    synth = a.synthetic
    label = "SYNTHETIC (simulated market, not real data)" if synth else "REAL DATA"
    study = SYNTH_STUDY if synth else STUDY_INDEXES
    ref = SYNTH_REFERENCE if synth else REFERENCE_INDEX
    targets = SYNTH["targets"] if synth else TARGETS
    seeds_all = SYNTH["seeds"] if synth else ALL_SEEDS
    new_seeds = seeds_all[1:]
    reps = a.reps
    print(f"[{label}] P110 stress test, stage 2 (docs/prereg/P110_GAIN_STRESS_TEST.md)", flush=True)
    R = gd.Rows()
    repro, contrib_rows, seed_rows, bound_rows, share_rows, period_rows, pool_rows, rank_rows = ([] for _ in range(8))
    seed_est = {}            # test_id -> {seed: estimate}

    # ---------------- daily losses
    F = {ix: load_index_frames(work, ix, targets, seeds_all) for ix in study}
    for ix in study:
        missing = [(t, s) for t in targets for s in seeds_all if (t, s) not in F[ix]]
        if missing:
            raise SystemExit(f"{ix}: work files missing for {missing}")
    NREF = load_reference_frames(work, synth, targets, seeds_all)
    avg = {ix: {t: seed_average([F[ix][(t, s)] for s in new_seeds]) for t in targets} for ix in study}
    avg5 = {ix: {t: seed_average([F[ix][(t, s)] for s in seeds_all]) for t in targets} for ix in study}
    ref_avg = {t: seed_average([NREF[(t, s)] for s in new_seeds]) for t in targets
               if all((t, s) in NREF for s in new_seeds)}

    # ---------------- reproduction checks (no loss in the first two)
    if not synth:
        for ix in study:
            for t in targets:
                wide = gd.load_wide(work, ix, t, HEADLINE_SEED, VARIANTS)
                committed = gd.read_csv_exact(gd.INDEXES[ix]["run"] / f"forecasts_{t}.csv")
                repro.append(gd.reproduction_check(wide, committed, f"{ix} {t} seed 7 full hybrid vs {gd.INDEXES[ix]['run'].name}"))
                refdir = Path(a.p106_work)
                for v in [v for v in VARIANTS if v != "mcsinfo"]:
                    pf = gd.work_file(refdir, ix, t, HEADLINE_SEED, v)
                    if not pf.exists():
                        continue
                    old = gd.read_csv_exact(pf).sort_values(["t", "minute"]).reset_index(drop=True)
                    same = {c: bool(np.array_equal(old[c].to_numpy(), wide[c].to_numpy()))
                            for c in ["date", "minute", "y", "mcsgarch", "mcsgarch_iv"]}
                    fx = old["forecast"].to_numpy(float)
                    repro.append({"check": f"{ix} {t} seed 7 {v} vs P106 work file", "rows_rerun": len(wide),
                                  "rows_committed": len(old), "rows_matched": int(len(old) == len(wide)),
                                  "forecast_n_identical": int(np.sum(fx == wide[v].to_numpy(float))),
                                  "forecast_max_abs_rel_diff": float(np.max(np.abs(fx / wide[v].to_numpy(float) - 1))),
                                  **{f"{c}_identical": x for c, x in same.items()}})
        # seed-7 contributions against P106's committed S5 and S6 rows
        p106 = pd.read_csv(P106_RESULTS / "tests.csv").set_index("test_id")
        for ix, fam in (("BANKNIFTY", "S5"), ("FINNIFTY", "S6")):
            for t in targets:
                for v in gd.PRIMARY_GROUPS:
                    c = gd.contribution(F[ix][(t, HEADLINE_SEED)], v)
                    row = p106.loc[f"{fam}_{t}_{v}"]
                    repro.append({"check": f"{ix} {t} seed 7 {v}: contribution and DM statistic vs P106 tests.csv {fam}_{t}_{v}",
                                  "estimate_here": c["estimate"], "estimate_p106": row["estimate"],
                                  "stat_here": c["stat"], "stat_p106": row["statistic"],
                                  "max_abs_diff": float(max(abs(c["estimate"] - row["estimate"]), abs(c["stat"] - row["statistic"])))})
    # mcsGARCH and the days are the same in every seed set (seed_average checks); record it
    for ix in study:
        for t in targets:
            repro.append({"check": f"{ix} {t}: mcsgarch and mcsgarch_iv daily QLIKE identical in all {len(seeds_all)} seed sets",
                          "rows_rerun": len(avg5[ix][t]), "ok": True})

    # ---------------- primary rows (F1 to F3) on the new-seed average, and their seed repeats
    def add_row(tid, kind, v, D, ix, t, what, role, family, delta=MARGIN, method="dm", hyp_extra=""):
        st = row_test(D, kind, v, delta, method)
        R.add(tid, hyp_text(kind, v, t, delta, hyp_extra), role, sample_text(ix, t, D, what), st["estimate"],
              st["stat"], st["p"], SIDED[kind], method_text(kind, delta, method), family, index=ix, target=t,
              variant=v, kind=kind, margin=delta if kind in ("tost", "ni") else np.nan, days=len(D),
              se=st.get("se", np.nan), ci90_low=st.get("ci90_low", np.nan), ci90_high=st.get("ci90_high", np.nan),
              ci95_upper=st.get("ci95_upper", np.nan), p_upper=st.get("p_upper", np.nan),
              p_lower=st.get("p_lower", np.nan))
        return st

    what_new = f"daily QLIKE averaged over the new seed sets {', '.join(map(str, new_seeds))}"
    for ix in study:
        for t in targets:
            D = avg[ix][t]
            for fam, kind, v in PRIMARY_SPECS:
                tid = primary_id(ix, t, fam, kind, v)
                st = add_row(tid, kind, v, D, ix, t, what_new, "primary", fam)
                seed_est[tid] = {s: row_estimate(F[ix][(t, s)], kind, v) for s in seeds_all}
                if kind in ("tost", "ni"):
                    gain = 100 * (D["mcsgarch"].mean() - D["full"].mean()) / D["mcsgarch"].mean()
                    bound_rows.append({"test_id": tid, "index": ix, "target": t, "variant": v, "kind": kind,
                                       "sample": "new-seed average", "estimate_points": st["estimate"],
                                       "se_points": st["se"], "ci90_low": st.get("ci90_low", np.nan),
                                       "ci90_high": st.get("ci90_high", st.get("ci95_upper")),
                                       "smallest_margin_that_passes": (st["bound"] if kind == "tost"
                                                                       else st["ci95_upper"]),
                                       "hybrid_gain_points": gain,
                                       "bound_as_share_of_gain": (st["bound"] if kind == "tost" else st["ci95_upper"]) / gain,
                                       "p_margin_1": st["p"],
                                       "p_margin_0.5": row_test(D, kind, v, MARGIN_SENS)["p"]})
            # secondary S1 (adds something) and S2 (the full hybrid beats mcsinfo)
            for v in NULL_GROUPS:
                tid = primary_id(ix, t, "S1", "adds", v)
                add_row(tid, "adds", v, D, ix, t, what_new, "secondary", "S1_adds_something")
                seed_est[tid] = {s: row_estimate(F[ix][(t, s)], "adds", v) for s in seeds_all}
            tid = primary_id(ix, t, "S2", "full_beats", "mcsinfo")
            add_row(tid, "full_beats", "mcsinfo", D, ix, t, what_new, "secondary", "S2_full_beats_mcsinfo")
            seed_est[tid] = {s: row_estimate(F[ix][(t, s)], "full_beats", "mcsinfo") for s in seeds_all}
            # S3: the 0.5-point margin
            for v in NULL_GROUPS:
                add_row(f"S3_{ix}_{t}_{v}", "tost", v, D, ix, t, what_new, "secondary", "S3_margin_half", MARGIN_SENS)
            add_row(f"S3_{ix}_{t}_mcsinfo_noninferior", "ni", "mcsinfo", D, ix, t, what_new, "secondary",
                    "S3_margin_half", MARGIN_SENS)

            # robustness R1 to R5 (the 28 primary rows in other samples or with another variance)
            masks = {"R4": (date_mask(D, INFLUENCE_DATES), "without 4 and 5 June 2024"),
                     "R5": (top_loss_days(D), f"without the {int(np.floor(TOP_SHARE * len(D)))} days with the largest "
                                              "mcsGARCH daily QLIKE")}
            for fam, kind, v in PRIMARY_SPECS:
                pid = primary_id(ix, t, fam, kind, v)
                add_row(f"R1_{pid}", kind, v, F[ix][(t, HEADLINE_SEED)], ix, t, "seed set 7 alone (P106's sample; "
                        "mcsinfo is new)", "robustness", "R1_seed7")
                add_row(f"R2_{pid}", kind, v, avg5[ix][t], ix, t, "daily QLIKE averaged over all five seed sets",
                        "robustness", "R2_five_seeds")
                if kind in ("tost", "ni"):
                    add_row(f"R3_{pid}", kind, v, D, ix, t, what_new, "robustness", "R3_delta_method", method="delta")
                for rk, (mk, txt) in masks.items():
                    if mk.all():
                        continue
                    fam_name = "R4_without_june_2024" if rk == "R4" else "R5_without_top_loss_days"
                    add_row(f"{rk}_{pid}", kind, v, D[mk], ix, t, f"{what_new}, {txt}", "robustness", fam_name)

            # placebo: the full hybrid against itself with another seed set
            d7 = F[ix][(t, HEADLINE_SEED)]
            for s in new_seeds:
                ds_ = F[ix][(t, s)]
                dd = pd.DataFrame({"mcsgarch": d7["mcsgarch"], "full": d7["full"], "seedfull": ds_["full"]})
                r1 = tost(points_series(dd, "seedfull", "full"), MARGIN)
                R.add(f"PL1_{ix}_{t}_s{s}", f"Placebo: the full hybrid with seed set {s} and with seed set 7 differ by less "
                      f"than {MARGIN} point in either direction ({TARGET_LABEL[t]}). Estimate: seed {s} minus seed 7, points.",
                      "placebo", sample_text(ix, t, dd, f"seed sets 7 and {s}"), r1["estimate"], r1["stat"], r1["p"],
                      SIDED["tost"], method_text("tost", MARGIN), "PL1_seed_placebo_tost", index=ix, target=t, seed=s)
                dm = diebold_mariano(ds_["full"], d7["full"], alternative="greater")
                R.add(f"PL2_{ix}_{t}_s{s}", f"Placebo: the full hybrid with seed set 7 has lower mean QLIKE than with seed set {s} "
                      f"({TARGET_LABEL[t]}; only the seeds differ). Estimate: points.", "placebo",
                      sample_text(ix, t, dd, f"seed sets 7 and {s}"), r1["estimate"], dm["stat"], dm["p"],
                      "one-sided (seed set 7 better)", DM_METHOD, "PL2_seed_placebo_dm", index=ix, target=t, seed=s)

            # exploratory E1: the period change of the mcsinfo contribution (P106's S3b on Nifty)
            p2 = gd.period_mask(D)
            if p2.any() and (~p2).any():
                mcs = D["mcsgarch"].to_numpy(float)
                st = gd.period_diff_test((D["mcsinfo"] - D["full"]).to_numpy(float), mcs, p2)
                R.add(f"E1_{ix}_{t}_mcsinfo_periods", "The value to the hybrid of the information mcsGARCH lacks (mcsinfo minus "
                      f"full) differs between the period to 19 Nov 2024 and the period since 20 Nov 2024 ({TARGET_LABEL[t]}). "
                      "Estimate: period 2 minus period 1, points of each period's mcsGARCH QLIKE.", "exploratory",
                      sample_text(ix, t, D, f"{what_new}, {st['n1']} + {st['n2']} days"), st["diff"], st["stat"], st["p"],
                      "two-sided", DELTA_PERIOD, "E1_mcsinfo_periods", index=ix, target=t, period1=st["r1"],
                      period2=st["r2"])
                for q, num in [("G (hybrid's gain over mcsGARCH)", D["mcsgarch"] - D["full"]),
                               ("B (mcsgarch_iv's extra loss over mcsGARCH)", D["mcsgarch_iv"] - D["mcsgarch"])] + \
                              [(f"contribution of {v}", D[v] - D["full"]) for v in VARIANTS[1:]]:
                    st = gd.period_diff_test(num.to_numpy(float), mcs, p2)
                    period_rows.append({"index": ix, "target": t, "sample": "new-seed average", "quantity": q,
                                        "period1_points": st["r1"], "period2_points": st["r2"], "difference": st["diff"],
                                        "se": st["se"], "t": st["stat"], "p_two_sided": st["p"], "days_p1": st["n1"],
                                        "days_p2": st["n2"]})

            # descriptive: contributions in every sample, shares of the gain
            samples = [("new-seed average", D), ("seed set 7", F[ix][(t, HEADLINE_SEED)]), ("five-seed average", avg5[ix][t])] + \
                      [(f"seed set {s}", F[ix][(t, s)]) for s in new_seeds]
            for sname, dd in samples:
                mc = dd["mcsgarch"].mean()
                for v in VARIANTS[1:]:
                    contrib_rows.append({"index": ix, "target": t, "sample": sname, "variant": v, "days": len(dd),
                                         "contribution_points": 100 * (dd[v].mean() - dd["full"].mean()) / mc,
                                         "hybrid_margin_vs_mcsgarch_pct": 100 * (dd["full"].mean() / mc - 1),
                                         "variant_margin_vs_mcsgarch_pct": 100 * (dd[v].mean() / mc - 1)})
            for k, v in enumerate(VARIANTS[1:]):
                rng = np.random.default_rng([BOOT_SEED, study.index(ix), targets.index(t), k])
                sh, lo, hi = gd.share_ci(D["mcsgarch"].to_numpy(), D["full"].to_numpy(), D[v].to_numpy(), reps=reps, rng=rng)
                share_rows.append({"index": ix, "target": t, "sample": "new-seed average", "variant": v,
                                   "share_of_gain": sh, "share_ci_low": lo, "share_ci_high": hi})

    # ---------------- S5: Nifty re-analysis (P106's committed daily losses, new-seed average)
    for t in targets:
        if t not in ref_avg:
            continue
        D = ref_avg[t]
        for v in NULL_GROUPS:
            st = add_row(f"S5_{ref}_{t}_{v}", "tost", v, D, ref, t, what_new + " (P106's committed files)", "secondary",
                         "S5_nifty_reanalysis")
            gain = 100 * (D["mcsgarch"].mean() - D["full"].mean()) / D["mcsgarch"].mean()
            bound_rows.append({"test_id": f"S5_{ref}_{t}_{v}", "index": ref, "target": t, "variant": v, "kind": "tost",
                               "sample": "new-seed average", "estimate_points": st["estimate"], "se_points": st["se"],
                               "ci90_low": st["ci90_low"], "ci90_high": st["ci90_high"],
                               "smallest_margin_that_passes": st["bound"], "hybrid_gain_points": gain,
                               "bound_as_share_of_gain": st["bound"] / gain, "p_margin_1": st["p"],
                               "p_margin_0.5": row_test(D, "tost", v, MARGIN_SENS)["p"]})
        st = add_row(f"S5_{ref}_{t}_mcsinfo_noninferior", "ni", "mcsinfo", D, ref, t, what_new + " (P106's committed files)",
                     "secondary", "S5_nifty_reanalysis")
        gain = 100 * (D["mcsgarch"].mean() - D["full"].mean()) / D["mcsgarch"].mean()
        bound_rows.append({"test_id": f"S5_{ref}_{t}_mcsinfo_noninferior", "index": ref, "target": t, "variant": "mcsinfo",
                           "kind": "ni", "sample": "new-seed average", "estimate_points": st["estimate"],
                           "se_points": st["se"], "ci90_low": np.nan, "ci90_high": st["ci95_upper"],
                           "smallest_margin_that_passes": st["ci95_upper"], "hybrid_gain_points": gain,
                           "bound_as_share_of_gain": st["ci95_upper"] / gain, "p_margin_1": st["p"],
                           "p_margin_0.5": row_test(D, "ni", "mcsinfo", MARGIN_SENS)["p"]})
        mc = D["mcsgarch"].mean()
        for v in VARIANTS[1:]:
            contrib_rows.append({"index": ref, "target": t, "sample": "new-seed average (P106's committed files)", "variant": v,
                                 "days": len(D), "contribution_points": 100 * (D[v].mean() - D["full"].mean()) / mc,
                                 "hybrid_margin_vs_mcsgarch_pct": 100 * (D["full"].mean() / mc - 1),
                                 "variant_margin_vs_mcsgarch_pct": 100 * (D[v].mean() / mc - 1)})

    # ---------------- S4: pooled two- and three-index versions (Driscoll-Kraay)
    pool_specs = [("carrier", v) for v in CARRIERS] + [("tost", v) for v in NULL_GROUPS] + \
                 [("beats", "mcsinfo"), ("ni", "mcsinfo")]
    pools = {"S4a": ([ref] + study, "S4a_pooled_three_index"), "S4b": (list(study), "S4b_pooled_two_index")}
    for pk, (members, fam) in pools.items():
        for t in targets:
            frames = [ref_avg.get(t) if ix == ref else avg[ix][t] for ix in members]
            if any(f is None for f in frames):
                continue
            dates = sorted(set.intersection(*[set(f.index) for f in frames]))
            for kind, v in pool_specs:
                X = pooled_series(frames, kind, v, dates)
                st = pooled_test(X, kind, MARGIN)
                tid = f"{pk}_{t}_{v}{KIND_SUFFIX[kind]}"
                names = " + ".join(LABEL[ix] for ix in members)
                R.add(tid, hyp_text(kind, v, t, MARGIN, f", pooled over {names}"), "secondary",
                      f"{names}, {TARGET_LABEL[t]}, {len(dates)} common test days {pd.Timestamp(dates[0]).date()} to "
                      f"{pd.Timestamp(dates[-1]).date()}, {what_new}", st["estimate"], st["stat"], st["p"], SIDED[kind],
                      POOL_METHOD + ("" if kind in ("carrier", "beats") else f"; margin {MARGIN} point"), fam,
                      target=t, variant=v, kind=kind, days=len(dates))
                pool_rows.append({"test_id": tid, "pool": names, "target": t, "variant": v, "kind": kind,
                                  "days": len(dates), "estimate_points": st["estimate"], "se": st["se"],
                                  "stat": st["stat"], "p": st["p"], "dk_lag": st["lag"],
                                  "ci90_low": st.get("ci90_low", np.nan), "ci90_high": st.get("ci90_high", np.nan),
                                  **{f"estimate_{ix.lower()}": e for ix, e in zip(members, st["per_index"])}})

    # ---------------- adjust, verdicts, claims
    families = {
        "F1": "primary F1, the realized history and the time of day carry the gain",
        "F2": "primary F2, the option-implied, VIX and calendar groups add nothing (equivalence, 1 point)",
        "F3": "primary F3, the mcsinfo hybrid gets the gain",
        "S1_adds_something": "secondary S1", "S2_full_beats_mcsinfo": "secondary S2", "S3_margin_half": "secondary S3",
        "S4a_pooled_three_index": "secondary S4a", "S4b_pooled_two_index": "secondary S4b",
        "S5_nifty_reanalysis": "secondary S5", "R1_seed7": "robustness R1", "R2_five_seeds": "robustness R2",
        "R3_delta_method": "robustness R3", "R4_without_june_2024": "robustness R4",
        "R5_without_top_loss_days": "robustness R5", "PL1_seed_placebo_tost": "placebo PL1",
        "PL2_seed_placebo_dm": "placebo PL2", "E1_mcsinfo_periods": None}
    tests = R.frame(families)
    # R6: the 28 primary p-values adjusted together (one Holm over all primary rows)
    prim = tests[tests["role"] == "primary"]
    r6 = prim.copy()
    r6["test_id"] = "R6_" + r6["test_id"]
    r6["role"] = "robustness"
    r6["p_adjusted"] = gd.holm(r6["p_raw"].to_numpy(float))
    r6["adjust_method"] = f"Holm over {len(r6)} rows: robustness R6, all primary rows of F1, F2 and F3 together"
    r6["_family"] = "R6_one_family"
    tests = pd.concat([tests, r6], ignore_index=True)

    verdicts = []
    for _, r in tests[tests["role"].isin(["primary", "secondary"]) & tests["test_id"].str.match(r"^(F|S1|S2)")].iterrows():
        tid = r["test_id"]
        det = R.detail[[d["test_id"] for d in R.detail].index(tid)]
        kind = det["kind"]
        est = seed_est.get(tid, {})
        agree = int(sum(agrees(kind, e) for e in est.values()))
        vals = np.array(list(est.values()))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        vk = kind if kind in VERDICT_PASS else "carrier"
        verdicts.append({"test_id": tid, "family": r["_family"], "index": det["index"], "target": det["target"],
                         "variant": det["variant"], "kind": kind, "estimate": r["estimate"], "p_raw": r["p_raw"],
                         "p_holm": r["p_adjusted"], "seed_sets": len(est), "seed_sets_agreeing": agree, "seed_sd": sd,
                         "seed_min": float(vals.min()), "seed_max": float(vals.max()),
                         "within_seed_noise": bool(kind in ("carrier", "adds", "full_beats") and np.isfinite(sd)
                                                   and abs(r["estimate"]) < 2 * sd),
                         "verdict": row_verdict(vk, r["p_adjusted"], agree, len(est)),
                         "verdict_if_one_family_R6": (row_verdict(vk, float(r6.loc[r6["test_id"] == "R6_" + tid, "p_adjusted"].iloc[0]),
                                                                  agree, len(est)) if r["role"] == "primary" else "")})
        seed_rows.append({"test_id": tid, "kind": kind, **{f"estimate_s{s}": e for s, e in est.items()},
                          "mean": float(vals.mean()), "sd": sd, "min": float(vals.min()), "max": float(vals.max()),
                          "n_agreeing": agree})
    verdicts = pd.DataFrame(verdicts)
    vv = verdicts.rename(columns={"p_holm": "p_adjusted"})
    claims = pd.DataFrame([claim_verdicts(vv, ix, targets) for ix in study])

    # descriptive: ranking of the five contributions against Nifty's
    for ix in study:
        for t in targets:
            if t not in ref_avg:
                continue
            nd, od = ref_avg[t], avg[ix][t]
            nif = [row_estimate(nd, "carrier", v) for v in gd.PRIMARY_GROUPS]
            oth = [row_estimate(od, "carrier", v) for v in gd.PRIMARY_GROUPS]
            rank_rows.append({"index": ix, "target": t, "sample": "new-seed average",
                              "spearman_rank_corr_with_nifty": float(spearmanr(nif, oth).statistic),
                              **{f"nifty_{v}": x for v, x in zip(gd.PRIMARY_GROUPS, nif)},
                              **{f"{ix.lower()}_{v}": x for v, x in zip(gd.PRIMARY_GROUPS, oth)}})

    # ---------------- write
    tests_out = tests.drop(columns="_family")
    cols = ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"]
    tests_out[cols].to_csv(out / "tests.csv", index=False)
    pd.DataFrame(R.detail).to_csv(out / "tests_detail.csv", index=False)
    verdicts.to_csv(out / "verdict.csv", index=False)
    claims.to_csv(out / "claims.csv", index=False)
    pd.DataFrame(bound_rows).to_csv(out / "equivalence_bounds.csv", index=False)
    pd.DataFrame(contrib_rows).to_csv(out / "contributions.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "seed_repeats.csv", index=False)
    pd.DataFrame(share_rows).to_csv(out / "shares.csv", index=False)
    pd.DataFrame(period_rows).to_csv(out / "periods.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(out / "pooled.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(out / "other_index_ranking.csv", index=False)
    pd.DataFrame(repro).to_csv(out / "reproduction_check.csv", index=False)
    for ix in study:
        for (t, s), d in F[ix].items():
            d.to_csv(out / f"daily_losses_{ix.lower()}_{t}_s{s}.csv")
    if (work / "timing.csv").exists():
        pd.read_csv(work / "timing.csv").to_csv(out / "timing.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)
    print(f"\n[{label}] primary and S1/S2 rows with verdicts:")
    print(verdicts.drop(columns=["family"]).to_string(index=False))
    print(f"\n[{label}] claim-level verdicts:")
    print(claims.T.to_string())
    print(f"\n[{label}] equivalence bounds:")
    print(pd.DataFrame(bound_rows).to_string(index=False))
    print(f"\n[{label}] pooled:")
    print(pd.DataFrame(pool_rows).to_string(index=False))
    print(f"\n[{label}] reproduction checks:")
    print(pd.DataFrame(repro).to_string(index=False))
    print(f"\n[{label}] other rows (id, estimate, p, Holm p):")
    print(tests_out.loc[~tests_out["role"].isin(["primary"]), ["test_id", "estimate", "p_raw", "p_adjusted"]].to_string(index=False))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    import arch
    import lightgbm
    import scipy
    inputs = {}
    for ix in study:
        p = work / f"inputs_{ix}.json"
        if p.exists():
            inputs[ix] = json.loads(p.read_text())
    timing = pd.read_csv(work / "timing.csv") if (work / "timing.csv").exists() else pd.DataFrame()
    wall = {}
    for ix in study:
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
        for t in targets:
            for s in seeds_all:
                p = P106_RESULTS / f"daily_losses_nifty_{t}_s{s}.csv"
                if p.exists():
                    ref_files[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    json.dump({"source": label + ("" if synth else ": NSE bhavcopy, participant OI and India VIX; third-party 1-minute "
                                  "index bars validated against NSE daily values"),
               "study": "P110 stress test of P106's claims (docs/prereg/P110_GAIN_STRESS_TEST.md)",
               "plan_commit": a.plan_commit, "analysis_commit": commit,
               "commands": ["PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p110_gain_stress_test.py run --index BANKNIFTY",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p110_gain_stress_test.py run --index FINNIFTY",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p110_gain_stress_test.py analyze --plan-commit <plan commit>"],
               "study_indexes": study, "reference_index": ref, "variants": VARIANTS,
               "drop_sets": {v: gd.DROP_SETS[v] for v in VARIANTS[1:]}, "pipeline_settings": gd.PIPELINE_SETTINGS,
               "synthetic_settings": SYNTH if synth else None,
               "seeds": {"headline": HEADLINE_SEED, "new": new_seeds, "primary_sample": "daily QLIKE averaged over the new seed sets",
                         "lgbm_seeds": {str(s): [s + 1000 * k for k in range(gd.N_SEEDS)] for s in seeds_all}},
               "margin_points": MARGIN, "sensitivity_margin_points": MARGIN_SENS, "alpha": ALPHA,
               "bootstrap": {"reps": reps, "mean_block_days": MEAN_BLOCK, "seed": BOOT_SEED},
               "influence_dates": INFLUENCE_DATES, "top_share": TOP_SHARE,
               "nifty_reference_files_sha256": ref_files, "input_sha256": inputs,
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
            p.add_argument("--index", choices=STUDY_INDEXES + SYNTH_STUDY + [SYNTH_REFERENCE], default="BANKNIFTY")
            p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
            p.add_argument("--targets", nargs="+", default=None)
            p.add_argument("--seeds", nargs="+", type=int, default=None)
            p.add_argument("--variants", nargs="+", default=None)
            p.add_argument("--chunk", type=int, default=CHUNK)
        if name == "analyze":
            p.add_argument("--out", default=None)
            p.add_argument("--reps", type=int, default=REPS)
            p.add_argument("--plan-commit", default=None)
            p.add_argument("--p106-work", default=str(ROOT / "data" / "processed" / "p106_ref"),
                           help="P106's seed-7 Bank Nifty and Fin Nifty work files, for the reproduction check")
    a = ap.parse_args(argv)
    if a.work is None:
        a.work = str(ROOT / "data" / "processed" / ("p110_synthetic" if a.synthetic else "p110"))
    if a.stage == "run":
        if a.synthetic != is_synth(a.index):
            raise SystemExit("--synthetic goes with the SYNTH* indices and only with them")
        stage_run(a)
    else:
        if a.out is None:
            a.out = str(ROOT / "data" / "processed" / "p110_synthetic_results") if a.synthetic else \
                str(ROOT / "results" / "p110_gain_stress_test")
        stage_analyze(a)


if __name__ == "__main__":
    main()
