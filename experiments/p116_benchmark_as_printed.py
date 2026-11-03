"""P116: how large is the hybrid's gain over mcsGARCH as Chhimwal, Pandey and Pandey (2025) specify it?
(registered plan: docs/prereg/P116_BENCHMARK_AS_PRINTED.md)

The forecasting benchmark of the committed runs (``mcs_ours``) is mcsGARCH with three choices that the paper does
not make: a separate diurnal curve for expiry days, a centred 5-minute moving average of the diurnal curve, and a
zero intraday mean. The paper (its equations 1 to 9) has one diurnal curve s_i, the plain average of eps^2 / sigma^2
by minute, and an ARMA(1,1) intraday mean with Student-t errors; the replication of its Table 2 fits
``McsGarch(mean="arma11", dist="t", smooth=1)``. This script refits, with the walk-forward of the committed runs
(refit every 21 trading days on all earlier days, the same test days, anchors and targets), mcsGARCH as the paper
specifies it (``mcs_cpp``) and every other combination of the three switches (a 2 x 2 x 2 factorial), and compares
them with the committed hybrid and P109's committed combination.

    mcsgarch          mcs_ours: the committed benchmark (expiry curve on, smoothing 5, zero mean), rebuilt and
                      checked against the committed forecasts
    mcs_cpp           one diurnal curve, no smoothing, ARMA(1,1) mean; forecast = variance part + expected
                      squared conditional mean (niftygex.volatility.arma11_mean_square_forecast)
    mcs_cpp_varonly   the same fit, forecast without the squared conditional mean (exploratory)
    ours_no_expiry    mcs_ours with one diurnal curve            (one switch set to the paper's choice)
    ours_no_smooth    mcs_ours without smoothing                 (one switch set to the paper's choice)
    ours_arma         mcs_ours with the ARMA(1,1) mean           (one switch set to the paper's choice)
    cpp_expiry        mcs_cpp with the expiry-day curve          (one switch set to ours)
    cpp_smooth        mcs_cpp with smoothing 5                   (one switch set to ours)
    cpp_zero_mean     mcs_cpp with a zero mean                   (one switch set to ours)

Stages (the walk-forward, the market loader, the reproduction check and the block files are P109's, reused):

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index NIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index BANKNIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index FINNIFTY --workers 2
    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py analyze --plan-commit <hash>

``check`` compares the rebuilt mcs_ours (and y) with the committed forecasts and computes no loss. ``--synthetic``
runs every stage on a simulated market (labelled SYNTHETIC). Block files go to --work (default
data/processed/p116, git-ignored). ``analyze`` writes results/p116_benchmark_as_printed/. Tests:
tests/test_p116_benchmark_as_printed.py (synthetic data only).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
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

from niftygex.evaluation import REGIMES, diebold_mariano  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    """Load a script as a module under its own name, registered in sys.modules so that the worker pool can
    pickle its functions by reference (the workers are forked and share the registered copy)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# P109's module (this study's own copy, under its own name) brings the walk-forward per refit block (run_block),
# the market loader and the settings check (stage_run), the reproduction check (merged, reproduction_check,
# stage_check), the daily losses (day_frame), and through P106 the tests.csv collector and the stationary bootstrap.
ex = _load("p116_copy_of_p109_mcsgarch_extensions", ROOT / "experiments" / "p109_mcsgarch_extensions.py")
gd, rf = ex.gd, ex.rf

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

_BASE = dict(daily="egarch", decay=False, lev=False, todayrv=False, cal=False)
P116_SPECS = {
    "mcs_cpp":         dict(_BASE, expiry_diurnal=False, smooth=1, mean="arma11", mean_sq=True),
    "mcs_cpp_varonly": dict(_BASE, expiry_diurnal=False, smooth=1, mean="arma11", mean_sq=False),
    "ours_no_expiry":  dict(_BASE, expiry_diurnal=False, smooth=5, mean="zero", mean_sq=False),
    "ours_no_smooth":  dict(_BASE, expiry_diurnal=True, smooth=1, mean="zero", mean_sq=False),
    "ours_arma":       dict(_BASE, expiry_diurnal=True, smooth=5, mean="arma11", mean_sq=True),
    "cpp_expiry":      dict(_BASE, expiry_diurnal=True, smooth=1, mean="arma11", mean_sq=True),
    "cpp_smooth":      dict(_BASE, expiry_diurnal=False, smooth=5, mean="arma11", mean_sq=True),
    "cpp_zero_mean":   dict(_BASE, expiry_diurnal=False, smooth=1, mean="zero", mean_sq=False),
}
ex.SPECS.update(P116_SPECS)     # this module's own copy of P109's module; P109's own runs are not affected

# work-file column -> model name in this study's outputs ("mcsgarch" is P109's spec, the committed benchmark)
VARIANTS = ["mcsgarch"] + list(P116_SPECS)
OURS, CPP, HYB, COMB = "mcs_ours", "mcs_cpp", "hybrid", "combination"
RENAME = {"mcsgarch": OURS}
FROM_OURS = ["ours_no_expiry", "ours_no_smooth", "ours_arma"]
FROM_CPP = ["cpp_expiry", "cpp_smooth", "cpp_zero_mean"]
FACTORS = ["expiry_curve", "smoothing", "mean"]
# level of each factor: 1 = our choice (expiry curve, smoothing 5, zero mean), 0 = the paper's choice
FACTORIAL = {
    OURS: (1, 1, 1), "ours_no_expiry": (0, 1, 1), "ours_no_smooth": (1, 0, 1), "ours_arma": (1, 1, 0),
    "cpp_expiry": (1, 0, 0), "cpp_smooth": (0, 1, 0), "cpp_zero_mean": (0, 0, 1), CPP: (0, 0, 0),
}
MODELS = [HYB, OURS, CPP, "mcs_cpp_varonly"] + FROM_OURS + FROM_CPP
LABELS = {
    OURS: "mcs_ours (the committed benchmark: expiry-day curve, 5-minute smoothing, zero mean)",
    CPP: "mcs_cpp (as the paper specifies: one diurnal curve, no smoothing, ARMA(1,1) mean)",
    "mcs_cpp_varonly": "mcs_cpp without the expected squared conditional mean in its forecast",
    "ours_no_expiry": "mcs_ours with one diurnal curve for all days",
    "ours_no_smooth": "mcs_ours without smoothing of the diurnal curve",
    "ours_arma": "mcs_ours with the ARMA(1,1) intraday mean",
    "cpp_expiry": "mcs_cpp with a separate expiry-day curve",
    "cpp_smooth": "mcs_cpp with the 5-minute smoothing",
    "cpp_zero_mean": "mcs_cpp with a zero intraday mean",
    HYB: "the hybrid (hybrid_base, committed forecasts, built on mcs_ours)",
    COMB: "P109's combination (committed daily losses)",
}

PRIMARY_TARGETS = ["next15", "next30"]
TARGET_LABEL = ex.TARGET_LABEL
ALPHA = 0.05
MARGIN = 0.5                    # points: "no material difference" if the two gains differ by at most this
REPS = 10000
MEAN_BLOCK = 10.0
BOOT_SEED = 116
INDEXES = ex.INDEXES
P109_DIR = ROOT / "results" / "p109_mcsgarch_extensions"
DM_METHOD = ex.DM_METHOD


# ---------------------------------------------------------------------------
# Stage 1 and the check: P109's code with this study's variants
# ---------------------------------------------------------------------------


def stage_run(a):
    a.variants = a.variants or VARIANTS
    a.study = "P116 benchmark as printed"
    ex.stage_run(a)


def stage_check(a):
    return ex.stage_check(a)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def pct(num: float, den: float) -> float:
    return 100.0 * num / den


def gain_pct(D: pd.DataFrame, m: str, bench: str) -> float:
    """Model m's gain over a benchmark in % of the benchmark's mean daily QLIKE (positive: m is better)."""
    return pct(D[bench].mean() - D[m].mean(), D[bench].mean())


def delta_points(D: pd.DataFrame) -> float:
    """The hybrid's gain over mcs_cpp minus its gain over mcs_ours, percentage points (positive: the headline,
    measured against mcs_ours, understates the gain over the paper's specification)."""
    return gain_pct(D, HYB, CPP) - gain_pct(D, HYB, OURS)


def delta_interval(D: pd.DataFrame, reps: int = REPS, rng=None):
    """95% stationary-bootstrap percentile interval of delta_points (days resampled jointly, mean block 10)."""
    rng = rng if rng is not None else np.random.default_rng(BOOT_SEED)
    x = np.column_stack([D[HYB].to_numpy(), D[OURS].to_numpy(), D[CPP].to_numpy()])
    idx = rf.stationary_bootstrap_indices(len(x), reps, MEAN_BLOCK, rng)
    mb = rf.bootstrap_means(x, idx)
    d = 100 * ((mb[:, 2] - mb[:, 0]) / mb[:, 2] - (mb[:, 1] - mb[:, 0]) / mb[:, 1])
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def shapley(L: dict, ref: float) -> dict:
    """Shapley split of L[mcs_cpp] - L[mcs_ours] over the three switches, in points of ``ref``.

    L maps each factorial model name to its mean daily QLIKE. The value of a set of switches set to our choice is
    L[mcs_cpp] - L[that model]; each switch gets its marginal value averaged over the 3! orders in which the
    switches can be set. The three parts add up to L[mcs_cpp] - L[mcs_ours] exactly."""
    by_level = {lv: m for m, lv in FACTORIAL.items()}

    def v(levels):
        return L[CPP] - L[by_level[tuple(levels)]]

    out = {f: 0.0 for f in FACTORS}
    perms = list(itertools.permutations(range(3)))
    for order in perms:
        lv = [0, 0, 0]
        for k in order:
            before = v(lv)
            lv[k] = 1
            out[FACTORS[k]] += (v(lv) - before) / len(perms)
    return {f: pct(val, ref) for f, val in out.items()}


def horizon_outcome(p_holm_h2: float, delta: float, ci: tuple) -> str:
    """The registered outcome at one horizon (plan Section 5)."""
    lo, hi = ci
    if np.isfinite(p_holm_h2) and p_holm_h2 < ALPHA:
        out = "headline understated" if delta > 0 else "headline overstated"
        if abs(delta) <= MARGIN:
            out += f" (by less than the {MARGIN}-point margin: a clear but immaterial difference)"
        return out
    if abs(delta) <= MARGIN:
        if lo >= -MARGIN and hi <= MARGIN:
            return f"no material difference (the 95% interval of the change lies inside +-{MARGIN} points)"
        return (f"no material difference (point estimate; the 95% interval of the change reaches beyond +-{MARGIN}"
                " points)")
    return "inconclusive (the change exceeds the margin but is not significant)"


def outcome_name(text: str) -> str:
    return text.split(" (")[0]


def p116_verdict(p_h1: dict, p_h2: dict, delta: dict, ci: dict, targets=PRIMARY_TARGETS) -> dict:
    """The registered decision rule over the primary targets. p_h1, p_h2: Holm-adjusted p of H1 and H2."""
    per = {t: horizon_outcome(p_h2[t], delta[t], ci[t]) for t in targets}
    names = {outcome_name(per[t]) for t in targets}
    if len(names) == 1:
        verdict = names.pop()
    else:
        verdict = "mixed: " + "; ".join(f"{outcome_name(per[t])} at {TARGET_LABEL[t]}" for t in targets)
    beats = {t: bool(np.isfinite(p_h1[t]) and p_h1[t] < ALPHA) for t in targets}
    if all(beats.values()):
        h1 = "the hybrid beats mcs_cpp at every primary horizon"
    elif any(beats.values()):
        h1 = "the hybrid beats mcs_cpp at " + ", ".join(TARGET_LABEL[t] for t in targets if beats[t]) + " only"
    else:
        h1 = "the hybrid does not beat mcs_cpp at any primary horizon"
    return {"verdict": verdict, "h1": h1, **{f"outcome_{t}": per[t] for t in targets},
            **{f"hybrid_beats_mcs_cpp_{t}": beats[t] for t in targets}}


# ---------------------------------------------------------------------------
# Stage 2: analysis
# ---------------------------------------------------------------------------


def load_index(work: Path, index: str, t: str, allow_mismatch: bool, synth: bool):
    """Daily losses of every model of one index and target, with the reproduction check; None if not run."""
    if not ex.committed_path(index, t, work).exists():
        return None, None, None
    w = ex.merged(work, index, t)
    if w.empty or w[VARIANTS].isna().any().any():
        print(f"  {index} {t}: work files missing or incomplete; not computed", flush=True)
        return None, None, None
    rel = "synthetic run" if synth else ex.committed_path(index, t, work).relative_to(ROOT)
    rc = ex.reproduction_check(w, f"{index} {t}: rebuilt mcs_ours and y vs {rel}")
    if rc["y_n_identical"] != len(w) or rc["mcsgarch_n_identical"] != len(w):
        print(f"  WARNING {index} {t}: rebuilt y or mcs_ours differs from the committed forecasts: {rc}", flush=True)
        if not allow_mismatch:
            raise SystemExit("reproduction check failed; rerun with --allow-mismatch to report anyway")
    w["y"] = w["committed_y"]
    w = w.rename(columns=RENAME)
    D = ex.day_frame(w, MODELS)
    p109 = P109_DIR / f"daily_losses_{index.lower()}_{t}.csv"
    extra = []
    if not synth and p109.exists():
        x = pd.read_csv(p109, index_col="date", float_precision="round_trip")
        x.index = pd.to_datetime(x.index).strftime("%Y-%m-%d")
        D[COMB] = x[COMB].reindex(D.index)
        extra.append({"check": f"{index} {t}: P109 committed daily QLIKE of mcsGARCH and the hybrid vs this study's",
                      "rows_committed": len(x), "rows_rebuilt": int(D[COMB].notna().sum()),
                      "mcsgarch_daily_max_abs_rel_diff": float(np.max(np.abs(
                          x["mcsgarch"].reindex(D.index).to_numpy() / D[OURS].to_numpy() - 1))),
                      "hybrid_daily_max_abs_rel_diff": float(np.max(np.abs(
                          x["hybrid"].reindex(D.index).to_numpy() / D[HYB].to_numpy() - 1)))})
    return w, D, [rc] + extra


def stage_analyze(a):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    synth = a.synthetic
    label = "SYNTHETIC (simulated market, not real data)" if synth else "REAL DATA"
    nifty = "SYNTH" if synth else "NIFTY"
    print(f"[{label}] P116 benchmark as printed, stage 2 (docs/prereg/P116_BENCHMARK_AS_PRINTED.md)", flush=True)
    Rw = gd.Rows()
    loss_rows, attr_rows, anchor_rows, repro_rows, verdict_rows = [], [], [], [], []
    boot_counter = [0]

    def rng():
        boot_counter[0] += 1
        return np.random.default_rng([BOOT_SEED, boot_counter[0]])

    def dm_row(test_id, hyp, role, family, D, index, target, what, l1, l2, est, sided):
        """sided "one": l2 has lower loss than l1 (one-sided). "two": two-sided on d = l1 - l2."""
        dm = diebold_mariano(D[l1], D[l2], alternative="greater" if sided == "one" else "two-sided")
        side = (f"one-sided ({l2} has lower loss than {l1})" if sided == "one"
                else f"two-sided (d = loss of {l1} minus loss of {l2}; positive statistic: {l2} has lower loss)")
        Rw.add(test_id, hyp, role, ex.sample_text(index, target, D, what), est, dm["stat"], dm["p"], side, DM_METHOD,
               family, index=index, target=target, subset=what, l1=l1, l2=l2, days=dm["n"])
        return dm

    def h1_h2(prefix, role, family, D, index, t, what):
        dm_row(f"{prefix}_hybrid_vs_cpp", f"The hybrid has lower mean QLIKE than mcs_cpp ({TARGET_LABEL[t]}, {what})."
               " Estimate: the hybrid's gain over mcs_cpp, % of mcs_cpp's QLIKE.", role, family, D, index, t, what,
               CPP, HYB, gain_pct(D, HYB, CPP), "one")
        dm_row(f"{prefix}_ours_vs_cpp", f"mcs_ours and mcs_cpp have different mean QLIKE ({TARGET_LABEL[t]}, {what})."
               " Estimate: mcs_ours's gain over mcs_cpp, % of mcs_cpp's QLIKE (positive: our extension makes the"
               " benchmark better).", role, family, D, index, t, what, CPP, OURS, gain_pct(D, OURS, CPP), "two")

    def losses(index, t, D, subset):
        models = [m for m in MODELS + [COMB] if m in D and D[m].notna().all()]
        for m in models:
            loss_rows.append({"index": index, "target": t, "subset": subset, "model": m, "label": LABELS[m],
                              "days": len(D), "qlike_mean": D[m].mean(),
                              "loss_diff_vs_mcs_ours": D[OURS].mean() - D[m].mean(),
                              "gain_pct_vs_mcs_ours": gain_pct(D, m, OURS),
                              "loss_diff_vs_mcs_cpp": D[CPP].mean() - D[m].mean(),
                              "gain_pct_vs_mcs_cpp": gain_pct(D, m, CPP)})
        L = {m: D[m].mean() for m in FACTORIAL}
        sh = shapley(L, L[CPP])
        r = {"index": index, "target": t, "subset": subset, "days": len(D),
             "total_points": pct(L[CPP] - L[OURS], L[CPP]),
             **{f"shapley_{f}_points": sh[f] for f in FACTORS},
             **{f"from_ours_{v}_points": pct(L[v] - L[OURS], L[CPP]) for v in FROM_OURS},
             **{f"from_cpp_{v}_points": pct(L[CPP] - L[v], L[CPP]) for v in FROM_CPP},
             "hybrid_gain_vs_mcs_ours_pct": gain_pct(D, HYB, OURS), "hybrid_gain_vs_mcs_cpp_pct": gain_pct(D, HYB, CPP),
             "delta_points": delta_points(D)}
        attr_rows.append(r)

    targets_n = list(ex.SYNTH["targets"]) if synth else list(ex.rp.TARGETS)
    data = {}
    for index in ([nifty] if synth else ["NIFTY", "BANKNIFTY", "FINNIFTY"]):
        for t in (targets_n if index in ("NIFTY", "SYNTH") else PRIMARY_TARGETS):
            w, D, rc = load_index(work, index, t, a.allow_mismatch, synth)
            if D is None:
                continue
            repro_rows.extend(rc)
            data[(index, t)] = (w, D)
    if not all((nifty, t) in data for t in PRIMARY_TARGETS):
        raise SystemExit("no finished work files for the primary targets")

    # ---------------- primary (H1, H2 at 15 and 30 minutes, Nifty, all days) and the Nifty secondaries
    deltas, cis = {}, {}
    for t in targets_n:
        if (nifty, t) not in data:
            continue
        w, D = data[(nifty, t)]
        exp_mask = D["is_expiry"].to_numpy(bool)
        losses(nifty, t, D, "all days")
        losses(nifty, t, D[exp_mask], "expiry days")
        losses(nifty, t, D[~exp_mask], "non-expiry days")
        for k, name in enumerate(REGIMES):
            m_ = D["regime"].to_numpy() == k
            if m_.any():
                losses(nifty, t, D[m_], f"regime {name}")
        if t in PRIMARY_TARGETS:
            h1_h2(f"P_{t}", "primary", "primary", D, nifty, t, "all days")
            deltas[t] = delta_points(D)
            cis[t] = delta_interval(D, a.reps, rng())
            h1_h2(f"S2_{t}_expiry", "secondary", "S2_expiry_days", D[exp_mask], nifty, t, "expiry days")
            if COMB in D and D[COMB].notna().all():
                dm_row(f"S3_{t}_combination_vs_cpp", f"P109's combination has lower mean QLIKE than mcs_cpp"
                       f" ({TARGET_LABEL[t]}, all days). Estimate: its gain over mcs_cpp, % of mcs_cpp's QLIKE.",
                       "secondary", "S3_combination", D, nifty, t, "all days", CPP, COMB, gain_pct(D, COMB, CPP), "one")
            h1_h2(f"E5_{t}_nonexpiry", "exploratory", "E5_nonexpiry", D[~exp_mask], nifty, t, "non-expiry days")
            for sname, mins, key in (("09:15 anchor only", [0], "0915"),
                                     ("without the 09:15 anchor", sorted(set(w["minute"]) - {0}), "rest")):
                dd = ex.day_frame(w, MODELS, minutes=mins)
                h1_h2(f"E4_{t}_{key}", "exploratory", "E4_anchor", dd, nifty, t, sname)
                losses(nifty, t, dd, sname)
        else:   # last hour
            h1_h2(f"S1_{t}", "secondary", "S1_last_hour", D, nifty, t, "all days")
        for v in FROM_OURS:
            dm_row(f"E1_{nifty.lower()}_{t}_{v}", f"{LABELS[v]} and mcs_ours have different mean QLIKE"
                   f" ({TARGET_LABEL[t]}). Estimate: the change in loss when this one switch is set to the paper's"
                   " choice, % of mcs_ours's QLIKE (positive: the paper's choice is worse).", "exploratory",
                   "E1_one_switch_from_ours", D, nifty, t, "all days", v, OURS, pct(D[v].mean() - D[OURS].mean(),
                                                                                    D[OURS].mean()), "two")
        for v in FROM_CPP:
            dm_row(f"E2_{nifty.lower()}_{t}_{v}", f"{LABELS[v]} and mcs_cpp have different mean QLIKE"
                   f" ({TARGET_LABEL[t]}). Estimate: the gain from setting this one switch to our choice, % of"
                   " mcs_cpp's QLIKE (positive: our choice is better).", "exploratory", "E2_one_switch_from_cpp", D,
                   nifty, t, "all days", CPP, v, gain_pct(D, v, CPP), "two")
        dm_row(f"E3_{t}_varonly", f"mcs_cpp with and without the expected squared conditional mean in its forecast have"
               f" different mean QLIKE ({TARGET_LABEL[t]}). Estimate: % change of mcs_cpp's QLIKE when the term is"
               " left out (positive: leaving it out is worse).", "exploratory", "E3_mean_term", D, nifty, t,
               "all days", "mcs_cpp_varonly", CPP, pct(D["mcs_cpp_varonly"].mean() - D[CPP].mean(), D[CPP].mean()),
               "two")
        # gain by anchor (descriptive)
        for mnt in sorted(set(w["minute"])):
            dd = ex.day_frame(w, MODELS, minutes=[mnt])
            anchor_rows.append({"index": nifty, "target": t, "minute": mnt,
                                "clock": (pd.Timestamp("2000-01-01 09:15") + pd.Timedelta(minutes=int(mnt))).strftime("%H:%M"),
                                "days": len(dd), **{f"{m}_qlike_mean": dd[m].mean() for m in [HYB, OURS, CPP]},
                                **{f"{m}_pct_vs_mcs_cpp": 100 * (dd[m].mean() / dd[CPP].mean() - 1)
                                   for m in [HYB, OURS] + FROM_OURS + FROM_CPP},
                                "hybrid_pct_vs_mcs_ours": 100 * (dd[HYB].mean() / dd[OURS].mean() - 1)})

    # ---------------- other indices (S4 Bank Nifty, S5 Fin Nifty; E1, E2 there)
    other_delta, other_ci = {}, {}
    for index, fam in (("BANKNIFTY", "S4_banknifty"), ("FINNIFTY", "S5_finnifty")):
        for t in PRIMARY_TARGETS:
            if (index, t) not in data:
                continue
            w, D = data[(index, t)]
            exp_mask = D["is_expiry"].to_numpy(bool)
            losses(index, t, D, "all days")
            losses(index, t, D[exp_mask], "expiry days")
            h1_h2(f"{fam.split('_')[0]}_{t}", "secondary", fam, D, index, t, "all days")
            other_delta[(index, t)] = delta_points(D)
            other_ci[(index, t)] = delta_interval(D, a.reps, rng())
            for v in FROM_OURS:
                dm_row(f"E1_{index.lower()}_{t}_{v}", f"{LABELS[v]} and mcs_ours have different mean QLIKE on"
                       f" {INDEXES[index]['label']} ({TARGET_LABEL[t]}).", "exploratory", "E1_one_switch_from_ours", D,
                       index, t, "all days", v, OURS, pct(D[v].mean() - D[OURS].mean(), D[OURS].mean()), "two")
            for v in FROM_CPP:
                dm_row(f"E2_{index.lower()}_{t}_{v}", f"{LABELS[v]} and mcs_cpp have different mean QLIKE on"
                       f" {INDEXES[index]['label']} ({TARGET_LABEL[t]}).", "exploratory", "E2_one_switch_from_cpp", D,
                       index, t, "all days", CPP, v, gain_pct(D, v, CPP), "two")

    # ---------------- adjust, verdict, write
    families = {"primary": "the four primary rows (H1 and H2 at 15 and 30 minutes, Nifty, all days)",
                "S1_last_hour": "secondary S1 (last hour)", "S2_expiry_days": "secondary S2 (expiry days)",
                "S3_combination": "secondary S3 (P109's combination vs mcs_cpp)",
                "S4_banknifty": "secondary S4 (Bank Nifty)", "S5_finnifty": "secondary S5 (Fin Nifty)",
                "E1_one_switch_from_ours": None, "E2_one_switch_from_cpp": None, "E3_mean_term": None,
                "E4_anchor": None, "E5_nonexpiry": None}
    tests = Rw.frame(families)
    detail = pd.DataFrame(Rw.detail)
    padj = tests.set_index("test_id")["p_adjusted"]
    p_h1 = {t: float(padj[f"P_{t}_hybrid_vs_cpp"]) for t in PRIMARY_TARGETS}
    p_h2 = {t: float(padj[f"P_{t}_ours_vs_cpp"]) for t in PRIMARY_TARGETS}
    v = p116_verdict(p_h1, p_h2, deltas, cis)
    rows = {"study": "P116", "index": nifty, "role": "primary (decision rule)", **v}
    for t in PRIMARY_TARGETS:
        D = data[(nifty, t)][1]
        rows.update({f"hybrid_gain_vs_mcs_ours_pct_{t}": gain_pct(D, HYB, OURS),
                     f"hybrid_gain_vs_mcs_cpp_pct_{t}": gain_pct(D, HYB, CPP),
                     f"delta_points_{t}": deltas[t], f"delta_ci_low_{t}": cis[t][0], f"delta_ci_high_{t}": cis[t][1],
                     f"mcs_ours_gain_vs_mcs_cpp_pct_{t}": gain_pct(D, OURS, CPP),
                     f"p_h1_holm_{t}": p_h1[t], f"p_h2_holm_{t}": p_h2[t]})
    verdict_rows.append(rows)
    # the same rule applied, descriptively, to each other index (its own S-family Holm p)
    for index, pre in (("BANKNIFTY", "S4"), ("FINNIFTY", "S5")):
        if not all((index, t) in data for t in PRIMARY_TARGETS):
            continue
        ph1 = {t: float(padj[f"{pre}_{t}_hybrid_vs_cpp"]) for t in PRIMARY_TARGETS}
        ph2 = {t: float(padj[f"{pre}_{t}_ours_vs_cpp"]) for t in PRIMARY_TARGETS}
        dl = {t: other_delta[(index, t)] for t in PRIMARY_TARGETS}
        cl = {t: other_ci[(index, t)] for t in PRIMARY_TARGETS}
        vv = p116_verdict(ph1, ph2, dl, cl)
        rr = {"study": "P116", "index": index, "role": "secondary (same rule, applied descriptively)", **vv}
        for t in PRIMARY_TARGETS:
            D = data[(index, t)][1]
            rr.update({f"hybrid_gain_vs_mcs_ours_pct_{t}": gain_pct(D, HYB, OURS),
                       f"hybrid_gain_vs_mcs_cpp_pct_{t}": gain_pct(D, HYB, CPP), f"delta_points_{t}": dl[t],
                       f"delta_ci_low_{t}": cl[t][0], f"delta_ci_high_{t}": cl[t][1],
                       f"mcs_ours_gain_vs_mcs_cpp_pct_{t}": gain_pct(D, OURS, CPP),
                       f"p_h1_holm_{t}": ph1[t], f"p_h2_holm_{t}": ph2[t]})
        verdict_rows.append(rr)

    tests_out = tests.drop(columns="_family")
    tests_out.to_csv(out / "tests.csv", index=False)
    detail.to_csv(out / "tests_detail.csv", index=False)
    pd.DataFrame(verdict_rows).to_csv(out / "verdict.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(out / "losses.csv", index=False)
    pd.DataFrame(attr_rows).to_csv(out / "attribution.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(out / "gain_by_anchor.csv", index=False)
    pd.DataFrame(repro_rows).to_csv(out / "reproduction_check.csv", index=False)
    for (index, t), (_, D) in data.items():
        D.to_csv(out / f"daily_losses_{index.lower()}_{t}.csv")
    pfiles = sorted(work.glob("*_block[0-9][0-9][0-9][0-9]_params.csv"))
    if pfiles:
        P = pd.concat([pd.read_csv(f) for f in pfiles], ignore_index=True)
        P = P[P["kind"] == "model"].copy()
        P["variant"] = P["variant"].replace(RENAME)
        keep = [c for c in ["index", "start", "variant", "daily_method", "expiry_diurnal", "smooth", "mean", "mean_sq",
                            "arma_c", "arma_phi", "arma_theta", "omega", "alpha", "beta", "nu", "loglik", "nobs",
                            "converged"] if c in P]
        P = P[keep].sort_values(["index", "start", "variant"])
        P.to_csv(out / "params_by_refit.csv", index=False)
        num = [c for c in ["arma_c", "arma_phi", "arma_theta", "omega", "alpha", "beta", "nu"] if c in P]
        P["persistence"] = P["alpha"] + P["beta"]
        summ = P.groupby(["index", "variant"])[num + ["persistence"]].agg(["mean", "min", "max"])
        summ.columns = [f"{x}_{y}" for x, y in summ.columns]
        summ["refits"] = P.groupby(["index", "variant"]).size()
        summ["converged_all"] = P.groupby(["index", "variant"])["converged"].all()
        summ.to_csv(out / "params_summary.csv")
    timing = [pd.read_csv(f) for f in sorted(work.glob("*_timing.csv"))]
    if timing:
        pd.concat(timing, ignore_index=True).to_csv(out / "timing.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"\n[{label}] verdict:")
    print(pd.DataFrame(verdict_rows).T.to_string())
    print(f"\n[{label}] tests (primary and secondary):")
    print(tests_out[tests_out["role"].isin(["primary", "secondary"])][
        ["test_id", "estimate", "statistic", "p_raw", "p_adjusted"]].to_string(index=False))
    print(f"\n[{label}] mean QLIKE, all days:")
    lr = pd.DataFrame(loss_rows)
    print(lr[lr["subset"] == "all days"][["index", "target", "model", "qlike_mean", "gain_pct_vs_mcs_ours",
                                         "gain_pct_vs_mcs_cpp"]].to_string(index=False))
    print(f"\n[{label}] attribution (points of mcs_cpp's QLIKE):")
    print(pd.DataFrame(attr_rows).to_string(index=False))
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
                                  "forecasts are the committed ones; P109's combination enters through its committed "
                                  "daily losses"),
               "study": "P116 benchmark as printed (docs/prereg/P116_BENCHMARK_AS_PRINTED.md)",
               "plan_commit": a.plan_commit, "analysis_commit": commit,
               "commands": ["PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index NIFTY --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index BANKNIFTY --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index FINNIFTY --workers 2",
                            "PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py analyze --plan-commit <plan commit>"],
               "specs": {RENAME.get(k, k): ex.SPECS[k] for k in VARIANTS}, "factorial": FACTORIAL,
               "min_train": ex.MIN_TRAIN, "block": ex.BLOCK, "margin_points": MARGIN, "alpha": ALPHA,
               "synthetic_settings": ex.SYNTH if synth else None,
               "bootstrap": {"reps": a.reps, "mean_block_days": MEAN_BLOCK, "seed": BOOT_SEED},
               "input_sha256": inputs, "stage1": wall, "seconds_stage2": round(time.time() - t0, 1),
               "python": sys.version.split()[0],
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
                           help="report even if the rebuilt mcs_ours differs from the committed one")
    a = ap.parse_args(argv)
    if a.work is None:
        a.work = str(ROOT / "data" / "processed" / ("p116_synthetic" if a.synthetic else "p116"))
    if a.stage == "analyze" and a.out is None:
        a.out = str(ROOT / "data" / "processed" / "p116_synthetic_results") if a.synthetic else \
            str(ROOT / "results" / "p116_benchmark_as_printed")
    if a.stage == "run":
        stage_run(a)
    elif a.stage == "check":
        return stage_check(a)
    else:
        stage_analyze(a)


if __name__ == "__main__":
    main()
