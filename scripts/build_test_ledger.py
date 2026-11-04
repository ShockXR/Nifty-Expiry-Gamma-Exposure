#!/usr/bin/env python
"""Project-wide test ledger (P89).

Reads the hypothesis tests of every real-data study in ``results/`` (one small
adapter function per study, each naming the CSV files and columns it uses) and
writes:

* ``results/test_ledger.csv``: one row per test with its raw p, role,
  pre-specification status and three adjusted p-values: Benjamini-Yekutieli
  (BY) over the whole project, BY within the study and Holm within the study.
* ``results/test_ledger_not_counted.csv``: p-values and outputs that are
  reported somewhere but deliberately not counted as tests (duplicates,
  alternative p-values of a counted test, placebo draws, simulations, figures
  or text without a p-value), each with a reason.
* ``results/test_ledger_checks.csv``: every adjusted p-value that a study
  reports itself, recomputed from its own raw p-values and compared.
* ``docs/TEST_LEDGER.md``: the plain-English summary (unless ``--no-doc``).

Rules followed throughout:

* A p-value is never invented. A result reported only as a figure, a
  confidence interval or text is listed as "not counted", never as a row.
* The raw p is two-sided unless the study stated a one-sided test in advance.
  A one-sided p that was not pre-stated is converted to two-sided as
  min(1, 2 min(p, 1 - p)). Wald, likelihood-ratio and chi-square tests are not
  directional and are used as reported.
* The same test reported in two files is counted once (see each adapter).
* Folders that appear later (``pinning_mechanism``, ``f2_economics``,
  ``pipeline_nse_v2``) are picked up automatically; ``power_real`` is
  excluded unless it contains p-values that are not already in the ledger.
* The forecasting contest is three studies: R1 is the first run
  (``results/pipeline_nse``, archived), R1V2 the current contest
  (``results/pipeline_nse_v2``) and R1V2DEV its development runs
  (``results/pipeline_nse_v2/development``). Different forecasts are different
  tests, so they are not deduplicated against each other. BNFC is the same
  pipeline on Bank Nifty (P91, ``results/pipeline_nse_banknifty``) and FNFC on
  Fin Nifty (P104, ``results/pipeline_nse_finnifty``).
* Pre-registered studies P92 (``sensex_swap``), P96 (``synthetic_control``) and
  P91 (``pipeline_nse_banknifty``), P104 (``pipeline_nse_finnifty``) and the post hoc referee responses
  (``referee_f2``, ``referee_forecast``, ``referee_expiry``) have their own
  adapters; a referee row that repeats a test already in the ledger is not
  counted again.

No git command is run. Commit times are read from the reflogs (``logs/HEAD`` and
``logs/refs/heads/**`` of the repository, followed through a worktree's
``.git`` file to the main repository).

Usage::

    python scripts/build_test_ledger.py [--root PATH] [--no-doc]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Multiple-testing adjustments (NaN-aware; NaN in, NaN out, NaN not counted).
# ----------------------------------------------------------------------------


def p_adjust(p, method: str) -> np.ndarray:
    """Adjusted p-values for 'holm', 'bh', 'by' or 'bonferroni'.

    Missing values are ignored (they do not count towards m) and stay missing.
    Matches statsmodels.stats.multitest.multipletests ('holm', 'fdr_bh',
    'fdr_by', 'bonferroni') on the non-missing values.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    q = p[ok]
    order = np.argsort(q, kind="mergesort")
    s = q[order]
    i = np.arange(1, m + 1, dtype=float)
    if method == "holm":
        adj = np.maximum.accumulate(np.minimum(1.0, (m - i + 1.0) * s))
    elif method in ("bh", "by"):
        c = float(np.sum(1.0 / i)) if method == "by" else 1.0
        adj = np.minimum.accumulate((m * c * s / i)[::-1])[::-1]
        adj = np.minimum(adj, 1.0)
    elif method == "bonferroni":
        adj = np.minimum(1.0, m * s)
    else:
        raise ValueError(f"unknown method {method}")
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def p_by(p):
    return p_adjust(p, "by")


def p_holm(p):
    return p_adjust(p, "holm")


def p_bh(p):
    return p_adjust(p, "bh")


def two_sided_from_one_sided(p: float) -> float:
    """Two-sided p from a one-sided p of a symmetric statistic."""
    if p is None or not np.isfinite(p):
        return float("nan")
    return float(min(1.0, 2.0 * min(p, 1.0 - p)))


# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------

PS_YES = "yes"
PS_PARTLY = "partly"
PS_STATED = "stated, not time-stamped"
PS_NO = "no"
PS_UNKNOWN = "unknown (study not yet documented)"

SIDE_TWO = "two-sided"
SIDE_ONE_PRE = "one-sided (pre-stated)"
SIDE_ONE_STATED = "one-sided (stated by the study, not time-stamped)"
SIDE_CONVERTED = "two-sided (doubled from a one-sided p that was not pre-stated)"
SIDE_NONDIR = "not directional (Wald, LR or chi-square), as reported"
SIDE_DOUBLED = ("two-sided equivalent min(1, 2p) of a one-sided max-type p (SPA or Reality Check) that was not "
                "pre-stated")

ROLES = ["primary", "headline", "secondary", "robustness", "placebo",
         "exploratory", "post hoc", "unlabelled"]

STRATEGY_NOTE = ("RESEARCH_STRATEGY_PANDEY.md (RQ1, H1) was in the repository when the session started "
                 "(reflog 19:53 UTC) and the first real-data commit is 20:28 UTC; confirm with git log")
COMMITTED_WITH_RESULTS = "study committed to git together with its results"

STUDIES = {
    "R1": ("Forecasting contest, first run (archived)", "docs/REAL_DATA_RESULTS.md (R1, what the first run said); results/pipeline_nse/"),
    "R1V2": ("Forecasting contest, current run (model study, RQ1)", "docs/REAL_DATA_RESULTS.md (R1); docs/MODEL_ITERATIONS_REAL.md"),
    "R1V2DEV": ("Forecasting contest, development runs (older protocol)", "docs/MODEL_ITERATIONS_REAL.md (Section 3)"),
    "R1b": ("Forecast diagnostics: tails and last hour", "docs/REAL_DATA_RESULTS.md (R1)"),
    "AUCTION": ("Closing-auction check (from 3 Aug 2026)", "docs/REAL_DATA_RESULTS.md (R0)"),
    "R2": ("Expiry-day volatility premium by regime", "docs/REAL_DATA_RESULTS.md (R2)"),
    "R3": ("GEX and next-day realized variance (F1)", "docs/REAL_DATA_RESULTS.md (R3)"),
    "R4": ("Day-level pinning toward max-OI and max-pain strikes", "docs/REAL_DATA_RESULTS.md (R4)"),
    "R5": ("End-of-day reversal and GEX (F2)", "docs/REAL_DATA_RESULTS.md (R5)"),
    "REPL": ("Replication of the paper on NSE data", "docs/REPLICATION_NSE.md"),
    "PIN": ("Expiry-day pinning study", "docs/PINNING_STUDY.md"),
    "EVENT": ("Expiry-schedule event study", "docs/EVENT_STUDY.md"),
    "FALS": ("Falsification of F1 and F2", "docs/FALSIFICATION.md"),
    "DSIGN": ("Dealer sign estimated from market responses", "docs/DEALER_SIGN.md"),
    "INTRA": ("Intraday gamma (P81) and follow-ups", "docs/INTRADAY_GAMMA.md"),
    "CONN": ("Connectedness by gamma regime (RQ4)", "docs/CONNECTEDNESS_GAMMA.md"),
    "SENSEX": ("BSE Sensex expiry as a confounder", "docs/SENSEX_CONFOUND.md"),
    "F2ECON": ("Economic size of F2 after costs (P90)", "docs/F2_ECONOMICS.md"),
    "PINMECH": ("Pinning mechanism (P87)", "docs/PINNING_MECHANISM.md"),
    "BNFC": ("Bank Nifty forecasting contest (P91)", "docs/BANKNIFTY_FORECAST.md; docs/prereg/P91_BANKNIFTY_FORECAST.md"),
    "SWAP": ("Nifty and Sensex expiry-weekday swap, spillovers (P92)", "docs/SENSEX_SWAP.md; docs/prereg/P92_SENSEX_SWAP.md"),
    "SYNTH": ("Loss of weekly expiries, synthetic control (P96)", "docs/SYNTHETIC_CONTROL.md; docs/prereg/P96_SYNTHETIC_CONTROL.md"),
    "REFFC": ("Referee response: forecasting (post hoc)", "docs/REFEREE_RESPONSE_FORECAST.md"),
    "REFEXP": ("Referee response: expiry premia and event study (post hoc)", "docs/REFEREE_RESPONSE_EXPIRY.md"),
    "REFF2": ("Referee response: F2 inference and positions (post hoc)", "docs/REFEREE_RESPONSE_F2.md"),
    "P103ECON": ("Economic value of the forecasting hybrid (P103)", "docs/P103_ECONOMIC_VALUE.md; docs/prereg/P103_ECONOMIC_VALUE.md"),
    "P99SEED": ("Seed bootstrap of the forecasting hybrid (P99)", "docs/SEED_BOOTSTRAP.md; docs/prereg/P99_SEED_BOOTSTRAP.md"),
    "FNFC": ("Fin Nifty forecasting contest, third index (P104)", "docs/THIRD_INDEX_FORECAST.md; docs/prereg/P104_THIRD_INDEX_FORECAST.md"),
    "POOL3": ("Pooled three-index test of the forecasting hybrid (P107)", "docs/P107_POOLED_THREE_INDEX.md; docs/prereg/P107_POOLED_THREE_INDEX.md"),
    "GAIN6": ("Where the forecasting hybrid's gain comes from: feature-group ablation and the November 2024 periods (P106)",
              "docs/P106_GAIN_DECOMPOSITION.md; docs/prereg/P106_GAIN_DECOMPOSITION.md"),
    "MCSX": ("Five parametric fixes to mcsGARCH against the forecasting hybrid (P109)",
             "docs/P109_MCSGARCH_EXTENSIONS.md; docs/prereg/P109_MCSGARCH_EXTENSIONS.md"),
    "CONNSET": ("Which connectedness setting reproduces the paper's printed totals (P113)",
                "docs/P113_CONNECTEDNESS_SETTINGS.md; docs/prereg/P113_CONNECTEDNESS_SETTINGS.md"),
    "PERSFIX": ("Does the paper's fall in intraday persistence survive P109's fixes to mcsGARCH (P114)",
                "docs/P114_PERSISTENCE_FIXED_MODEL.md; docs/prereg/P114_PERSISTENCE_FIXED_MODEL.md"),
    "BENCHCPP": ("The forecasting hybrid against mcsGARCH as the paper specifies it (P116)",
                 "docs/P116_BENCHMARK_AS_PRINTED.md; docs/prereg/P116_BENCHMARK_AS_PRINTED.md"),
    "GAINSTRESS": ("P106's three claims stress-tested on Bank Nifty and Fin Nifty (P110)",
                   "docs/P110_GAIN_STRESS_TEST.md; docs/prereg/P110_GAIN_STRESS_TEST.md"),
    "PERSPLAC": ("Is the paper's fall in intraday persistence larger than in placebo windows (P117)",
                 "docs/P117_PERSISTENCE_PLACEBO.md; docs/prereg/P117_PERSISTENCE_PLACEBO.md"),
    "HYBCPP": ("The same hybrid built on mcsGARCH as the paper specifies it (P118)",
               "docs/P118_HYBRID_ON_PAPER_MCSGARCH.md; docs/prereg/P118_HYBRID_ON_PAPER_MCSGARCH.md"),
}

PREREG = {
    "pinning_mechanism": ("docs/prereg/P87_PINNING_MECHANISM.md", "Pre-registration P87"),
    "f2_economics": ("docs/prereg/P90_F2_ECONOMICS.md", "Pre-registration P90"),
    "pipeline_nse_banknifty": ("docs/prereg/P91_BANKNIFTY_FORECAST.md", "P91 pre-registration"),
    "sensex_swap": ("docs/prereg/P92_SENSEX_SWAP.md", "Pre-registration P92"),
    "synthetic_control": ("docs/prereg/P96_SYNTHETIC_CONTROL.md", "P96 pre-registration"),
    "p103_economic_value": ("docs/prereg/P103_ECONOMIC_VALUE.md", "P103 registered plan"),
    "seed_bootstrap": ("docs/prereg/P99_SEED_BOOTSTRAP.md", "P99 registered plan"),
    "pipeline_nse_finnifty": ("docs/prereg/P104_THIRD_INDEX_FORECAST.md", "P104 registered plan"),
    "p107_pooled_test": ("docs/prereg/P107_POOLED_THREE_INDEX.md", "P107 registered plan"),
    "p106_gain_decomposition": ("docs/prereg/P106_GAIN_DECOMPOSITION.md", "P106 registered plan"),
    "p109_mcsgarch_extensions": ("docs/prereg/P109_MCSGARCH_EXTENSIONS.md", "P109 registered plan"),
    "p113_connectedness_settings": ("docs/prereg/P113_CONNECTEDNESS_SETTINGS.md", "P113 registered plan"),
    "p114_persistence_fixed_model": ("docs/prereg/P114_PERSISTENCE_FIXED_MODEL.md", "P114 registered plan"),
    "p116_benchmark_as_printed": ("docs/prereg/P116_BENCHMARK_AS_PRINTED.md", "P116 registered plan"),
    "p110_gain_stress_test": ("docs/prereg/P110_GAIN_STRESS_TEST.md", "P110 registered plan"),
    "p117_persistence_placebo": ("docs/prereg/P117_PERSISTENCE_PLACEBO.md", "P117 registered plan"),
    "p118_hybrid_on_paper_mcsgarch": ("docs/prereg/P118_HYBRID_ON_PAPER_MCSGARCH.md", "P118 registered plan"),
}

COLUMNS = [
    "study", "study_name", "test_id", "hypothesis", "outcome", "sample", "sign_rule",
    "estimate", "statistic_name", "statistic", "p_reported", "p_raw", "sidedness", "p_method",
    "alt_p", "role", "pre_specified", "pre_spec_note", "source_file", "source_key",
    "p_study_adjusted", "study_adjust_method", "p_study_adjusted_recomputed", "study_adjust_check",
    "p_by_project", "p_by_study", "p_holm_study", "survives_by_project_5pct",
    "survives_holm_study_5pct", "doc", "note",
]

MEASURES = {
    "rv5": "whole-day realized variance (5-minute returns)",
    "rv1": "whole-day realized variance (1-minute returns)",
    "rs": "whole-day Rogers-Satchell variance",
    "rv_day": "whole-day realized variance",
    "rv_first_hour": "first-hour realized variance",
    "rv_midday": "midday realized variance",
    "rv_last_hour": "last-hour realized variance",
    "rv_last30": "last-30-minute realized variance",
    "opt_premium": "option premium turnover",
    "opt_notional": "option notional turnover",
    "fut_notional": "futures notional turnover",
    "abs_overnight_bp": "absolute overnight return",
    "corr5_z": "Nifty and Bank Nifty 5-minute return correlation (Fisher z)",
    "rv1_over_rv5": "ratio of 1-minute to 5-minute realized variance",
    "rs_over_rv5": "ratio of Rogers-Satchell to 5-minute realized variance",
}
TARGETS = {"next15": "realized variance over the next 15 minutes",
           "next30": "realized variance over the next 30 minutes",
           "last_hour": "realized variance over the last hour"}
RULES = {"us_convention": "US convention", "participant_pro": "NSE Pro participant",
         "client_mirror": "mirror of NSE Client", "us": "US convention", "pro": "NSE Pro participant",
         "cmirror": "mirror of NSE Client", "none": "", "gross": "gross (unsigned)"}

DASHES = {"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-"}


def clean(s) -> str:
    if s is None or (isinstance(s, float) and not np.isfinite(s)):
        return ""
    s = str(s)
    for k, v in DASHES.items():
        s = s.replace(k, v)
    return s


def num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def fmt(x, digits=4) -> str:
    v = num(x)
    if not np.isfinite(v):
        return ""
    return f"{v:.{digits}g}"


def _int(x) -> str:
    v = num(x)
    return str(int(v)) if np.isfinite(v) else "n/a"


def alt(**kw) -> str:
    parts = []
    for k, v in kw.items():
        if isinstance(v, str):
            if v:
                parts.append(f"{k}={v}")
            continue
        if np.isfinite(num(v)):
            parts.append(f"{k}={fmt(v)}")
    return "; ".join(parts)


def key(*parts) -> str:
    return "|".join(clean(p) for p in parts)


# ----------------------------------------------------------------------------
# Context: rows, not-counted items, cross-checks, files consumed
# ----------------------------------------------------------------------------


class Ctx:
    def __init__(self, root):
        self.root = Path(root)
        self.rows: list[dict] = []
        self.not_counted: list[dict] = []
        self.checks: list[dict] = []
        self.consumed: set[str] = set()
        self.warnings: list[str] = []

    # files
    def res(self, rel: str) -> Path:
        return self.root / "results" / rel

    def has(self, rel: str) -> bool:
        return self.res(rel).exists()

    def read(self, rel: str) -> pd.DataFrame:
        self.consumed.add(rel)
        return pd.read_csv(self.res(rel))

    # records
    def add(self, study, hypothesis, outcome, sample, p_reported, *, role, pre_specified,
            source_file, source_key, sidedness=SIDE_TWO, sign_rule="", estimate=float("nan"),
            statistic_name="", statistic=float("nan"), p_method="", alt_p="", pre_spec_note="",
            note="", p_study_adjusted=float("nan"), study_adjust_method="", family=""):
        p_rep = num(p_reported)
        if not np.isfinite(p_rep):
            return None
        if not (-1e-12 <= p_rep <= 1 + 1e-12):
            self.warn(f"{study}: p outside [0, 1] in {source_file} ({source_key}); skipped")
            return None
        if sidedness == SIDE_CONVERTED:
            p_raw = two_sided_from_one_sided(p_rep)
        elif sidedness == SIDE_DOUBLED:
            p_raw = float(min(1.0, 2.0 * p_rep))
        else:
            p_raw = p_rep
        if p_rep == 0.0:
            note = (note + "; " if note else "") + "reported p is exactly 0 (below the method's resolution or floating-point underflow); kept as 0"
        row = dict(
            study=study, study_name=STUDIES.get(study, (study, ""))[0], test_id="",
            hypothesis=hypothesis, outcome=outcome, sample=sample, sign_rule=sign_rule,
            estimate=num(estimate), statistic_name=statistic_name, statistic=num(statistic),
            p_reported=p_rep, p_raw=p_raw, sidedness=sidedness, p_method=p_method, alt_p=alt_p,
            role=role, pre_specified=pre_specified, pre_spec_note=pre_spec_note,
            source_file=f"results/{source_file}", source_key=source_key,
            p_study_adjusted=num(p_study_adjusted), study_adjust_method=study_adjust_method,
            p_study_adjusted_recomputed=float("nan"), study_adjust_check="",
            doc=STUDIES.get(study, ("", ""))[1], note=note, _family=family,
        )
        self.rows.append(row)
        return row

    def skip(self, study, source_file, what, n, reason):
        if not n:
            return
        self.not_counted.append(dict(study=study, source_file=f"results/{source_file}" if source_file else "",
                                     what=what, n_items=n, reason=reason))

    def warn(self, msg):
        self.warnings.append(msg)
        print("WARNING:", msg, file=sys.stderr)

    def check(self, study, family, source_file, method, p_raw, p_study, note="", tol=1e-6):
        """Recompute a study-reported adjustment and record the comparison."""
        p_raw = np.asarray(p_raw, float)
        p_study = np.asarray(p_study, float)
        rec = p_adjust(p_raw, method)
        both = np.isfinite(rec) & np.isfinite(p_study)
        diff = np.abs(rec[both] - p_study[both])
        n_mis = int((diff > tol).sum())
        self.checks.append(dict(
            study=study, family=family, source_file=f"results/{source_file}", method=method,
            m_tests=int(np.isfinite(p_raw).sum()), n_compared=int(both.sum()), n_mismatch=n_mis,
            max_abs_diff=float(diff.max()) if diff.size else float("nan"),
            result="match" if n_mis == 0 and both.sum() > 0 else ("MISMATCH" if n_mis else "nothing to compare"),
            note=note))
        return rec


def _set(row, k, v):
    if row is not None:
        row[k] = v


def dedup_exact(rows: list[dict], cols=("p_reported", "estimate"), digits=12, label="") -> list[dict]:
    """Drop rows whose (p, estimate) are identical to an earlier row (same test reported twice)."""
    seen = {}
    out = []
    dropped = []
    for r in rows:
        k = tuple(round(num(r[c]), digits) if np.isfinite(num(r[c])) else "nan" for c in cols)
        if k in seen and all(x != "nan" for x in k):
            dropped.append((r, seen[k]))
            continue
        seen[k] = r
        out.append(r)
    return out, dropped


# ----------------------------------------------------------------------------
# Adapters. Each one says which CSV and which columns it reads.
# ----------------------------------------------------------------------------


def adapt_r1(ctx: Ctx, study="R1", folder="pipeline_nse", pre_spec_ok=True):
    """R1 forecasting contest.

    Files: results/<folder>/tests_{next15,next30,last_hour}.csv.
    Columns: subset (all / expiry), comparison ("mcsgarch vs X": X has lower
    QLIKE than mcsGARCH; "GEX ablation: A vs B": B, with GEX, has lower QLIKE
    than A), mean_diff, stat (Diebold-Mariano with the Harvey-Leybourne-Newbold
    correction), p (one-sided, alternative "greater"), n (test days).
    One-sided Diebold-Mariano tests were pre-stated (strategy H1).
    MCS files mcs_*.csv are set-level p-values that overlap these DM tests: not counted.
    """
    files = sorted(ctx.res(folder).glob("tests_*.csv"))
    rows_before = len(ctx.rows)
    for f in files:
        rel = f"{folder}/{f.name}"
        target = f.stem.replace("tests_", "")
        d = ctx.read(rel)
        for _, r in d.iterrows():
            comp = clean(r["comparison"])
            subset = clean(r["subset"])
            expiry = subset == "expiry"
            samp = f"{'expiry' if expiry else 'all'} test days (n = {int(num(r['n']))}), horizon {target}"
            if comp.startswith("GEX ablation"):
                a, b = [s.strip() for s in comp.split(":", 1)[1].split(" vs ")]
                hyp = f"Adding GEX features ({b}) lowers forecast loss against the same model without them ({a})"
                rule = {"hybrid_gex_us": "US convention", "hybrid_gex_participant": "Pro and Client participant",
                        }.get(b, "all three rules")
                role = "headline" if expiry else "secondary"
            else:
                b = comp.split(" vs ")[1].strip()
                hyp = f"{b} forecasts better than mcsGARCH (our implementation of the paper's model)"
                rule = ""
                role = "headline" if (expiry and b.startswith(("hybrid", "lgbm"))) else "secondary"
            if pre_spec_ok and expiry and role == "headline":
                ps, psn = PS_YES, STRATEGY_NOTE
            elif pre_spec_ok:
                ps, psn = PS_PARTLY, "RQ1/H1 named expiry days; the all-days comparison and model list were set later"
            else:
                ps, psn = PS_NO, "models built after the first contest results"
            ctx.add(study, hyp, f"QLIKE loss of forecasts of {TARGETS.get(target, target)}", samp, r["p"],
                    role=role, pre_specified=ps, pre_spec_note=psn, source_file=rel,
                    source_key=key(f.name, subset, comp), sidedness=SIDE_ONE_PRE if pre_spec_ok else SIDE_CONVERTED,
                    sign_rule=rule, estimate=r["mean_diff"], statistic_name="Diebold-Mariano (HLN) t",
                    statistic=r["stat"], p_method="Diebold-Mariano, one-sided (model better)")
    mcs = sorted(ctx.res(folder).glob("mcs_*.csv"))
    for f in mcs:
        ctx.consumed.add(f"{folder}/{f.name}")
    if mcs:
        ctx.skip(study, f"{folder}/mcs_*.csv", "Model Confidence Set p-values", len(mcs),
                 "set-level p-values (already adjusted for the model set) that answer the same question as the DM tests")
    return len(ctx.rows) - rows_before


# ----------------------------------------------------------------------------
# R1V2: the current forecasting contest (results/pipeline_nse_v2) and its development runs
# ----------------------------------------------------------------------------

V2_LOSS = {"qlike": "QLIKE", "mse": "MSE (squared error)"}
V2_RULE = {"gross": "gross (unsigned)", "us": "US convention", "participant": "NSE Pro participant",
           "client": "mirror of NSE Client", "placebo": "placebo (random earlier day)"}
V2_SET = {"gross": "unsigned gross gamma", "us": "US-convention signed GEX", "participant": "Pro-participant signed GEX",
          "client": "client-mirror signed GEX", "placebo": "placebo GEX from a random earlier day"}
V2_FAMILY = {"linear": "ridge log-HAR", "hybrid": "hybrid", "hybridiv": "hybrid on the implied-level mcsGARCH",
             "lgbm": "LightGBM", "lgbmt": "tuned LightGBM"}
V2_FAMILY_SHORT = {**V2_FAMILY, "hybridiv": "implied-level hybrid"}
V2_RULE_SHORT = {"us": "US", "participant": "Pro", "client": "client mirror", "gross": "gross"}
V2_MODEL = {
    "mcsgarch": "mcsGARCH refitted every 21 days, our implementation of the paper's model",
    "naive_last": "naive, last value", "naive_prevday": "naive, previous day", "vix": "India VIX model",
    "implied": "ATM straddle model", "har": "HAR", "mcsgarch_r63": "mcsGARCH refitted every 63 days",
    "mcsgarch_iv": "mcsGARCH with the straddle-implied daily level",
    "linear_v1": "ridge log-HAR, first-run base", "lgbm_v1": "LightGBM, first-run base",
    "hybrid_v1": "hybrid, first-run base", "hybrid_v1gex": "hybrid, first-run base and first-run GEX features",
    "linear_base": "ridge log-HAR, new base", "lgbm_base": "LightGBM, new base", "lgbmt_base": "tuned LightGBM, new base",
    "hybrid_base": "hybrid of mcsGARCH and LightGBM, new base", "hybridiv_base": "hybrid on the implied-level mcsGARCH, new base",
}
V2_SUBSET = {
    "all": "all test days", "expiry": "expiry days", "non_expiry": "non-expiry days",
    "day_before_expiry": "days before an expiry", "open_0915": "09:15 forecast only",
    "excl_0915": "all test days without the 09:15 forecast", "excl_0915_expiry": "expiry days without the 09:15 forecast",
    "multi_weekly": "several-weeklies regime, to 19 Nov 2024",
    "single_weekly_thu": "one Thursday weekly, 20 Nov 2024 to 31 Aug 2025",
    "single_weekly_tue": "Tuesday weekly, from 1 Sep 2025",
    "pre_auction": "test days before the closing auction began on 3 Aug 2026",
    "auction_period": "closing-auction period, from 3 Aug 2026",
    "pre_auction_excl_0915": "test days before 3 Aug 2026 without the 09:15 forecast",
    "development": "development window, test days to 30 Jun 2025", "holdout": "hold-out, 1 Jul 2025 onwards",
    "holdout_pre_auction": "hold-out days before 3 Aug 2026",
}
V2_PS_H1 = ("H1 in RESEARCH_STRATEGY_PANDEY.md (in git before the first real-data run) stated a one-sided Diebold-Mariano "
            "test of the same model without and with GEX on expiry-day test days; the v2 models, GEX sets and losses "
            "were set after the first contest")
V2_PS_NO = "v2 models and comparisons set after the first contest (MODEL_ITERATIONS_REAL.md)"
V2_PS_RQ1 = ("RQ1 asked in general terms whether models beat mcsGARCH on expiry days, but it stated no test, and the "
             "hybrid was made the headline after it had won the first contest; " + V2_PS_NO)
V2_PS_DEV = "development run, used to choose the final design (MODEL_ITERATIONS_REAL.md, Section 3)"
V2_DM = "Diebold-Mariano on daily mean losses (Newey-West, HLN, Student t), one-sided as reported ({})"
V2_DM_DEV = "Diebold-Mariano on daily mean losses, development protocol (normal p-values), one-sided as reported ({})"
V2_TESTS_COLS = {"subset", "loss", "comparison", "mean_diff", "stat", "p", "n"}
V2_ABL_COLS = {"family", "label", "group", "base", "alt", "days", "change_pct", "dm_stat", "p_one_sided"}
V2_P_IS_ONE = ("one-sided p is exactly 1 in the file (1 to machine precision), so the doubled two-sided p is 0; the true "
               "value is too small to represent: the second model is far worse")


def v2_model(name) -> str:
    """Plain-English name of a v2 forecasting model code."""
    name = clean(name)
    if name in V2_MODEL:
        return V2_MODEL[name]
    m = re.match(r"(linear|hybridiv|hybrid|lgbmt|lgbm)_gex_(\w+)$", name)
    if m:
        return f"{V2_FAMILY[m.group(1)]}, new base + {V2_SET.get(m.group(2), m.group(2))}"
    return name


def v2_rule(name) -> str:
    m = re.search(r"_gex_(\w+)$", clean(name))
    return V2_RULE.get(m.group(1), m.group(1)) if m else ""


def v2_subset(s) -> str:
    """Plain-English label of a v2 subset code (subset or group column)."""
    s = clean(s)
    if s in V2_SUBSET:
        return V2_SUBSET[s]
    for suf, lab in (("_non_expiry", "non-expiry days"), ("_expiry", "expiry days")):
        if s.endswith(suf) and s[: -len(suf)] in V2_SUBSET:
            return f"{V2_SUBSET[s[: -len(suf)]]}, {lab} only"
    return s


def _v2_obs(x: float, better_if_positive: bool, yes: str, no: str) -> str:
    if not np.isfinite(x) or x == 0:
        return "no difference"
    return yes if (x > 0) == better_if_positive else no


def _v2_add(ctx: Ctx, study, hyp, outcome, sample, p, side, note="", **kw):
    if side == SIDE_CONVERTED and num(p) >= 1.0:
        note = (note + "; " if note else "") + V2_P_IS_ONE
    return ctx.add(study, hyp, outcome, sample, p, sidedness=side, note=note, **kw)


def _v2_vs_mcsgarch(comp: str, gain: float, better_if_positive=True, desc_a=None, desc_b=None):
    """Hypothesis text for 'a vs b' (b against the reference a), with the observed direction."""
    a, b = [s.strip() for s in clean(comp).split(" vs ", 1)]
    ref = "mcsGARCH" if a == "mcsgarch" else a
    obs = _v2_obs(gain, better_if_positive, f"{b} better", f"{b} worse")
    return a, b, (f"{b} and {ref} differ in forecast loss (observed: {obs}); "
                  f"{b} = {desc_b or v2_model(b)}, {ref} = {desc_a or v2_model(a)}")


def _v2_ablation(family, label, subset, base, alt_model, change, dev=False):
    """Hypothesis, role, pre-specification and sidedness of one row of an ablation file (QLIKE or MSE).

    H1 rows (the same model without and with a GEX set, on expiry-day test days, any model family and loss)
    are 'partly' pre-specified and keep the one-sided p; every other row is doubled to two-sided."""
    fam, lab, subset = clean(family), clean(label), clean(subset)
    head = fam.split(":")[0].strip()
    first_run = fam.startswith("v1")
    famdesc = "first-run hybrid" if first_run else V2_FAMILY_SHORT.get(head, head)
    if fam.endswith("base vs base+GEX"):
        kind = "placebo" if lab == "placebo" else "gex"
    elif fam.endswith("unsigned vs signed"):
        kind = "sign"
    elif "placebo vs real" in fam:
        kind = "real"
    else:
        kind = "other"
    h1 = kind == "gex" and subset == "expiry"
    pair = f"{base} vs {alt_model}"
    obs = _v2_obs(change, False, "lower", "higher")
    if kind == "placebo":
        hyp = f"Placebo GEX (from a random earlier day) changes the {famdesc}'s forecast loss (observed: {obs}; {pair})"
    elif kind == "gex":
        what = "the first run's GEX features" if first_run else V2_SET.get(lab, lab)
        hyp = (f"Adding {what} lowers the {famdesc}'s forecast loss (H1; {pair})" if h1 else
               f"Adding {what} changes the {famdesc}'s forecast loss (observed: {obs}; {pair})")
    elif kind == "sign":
        hyp = (f"{famdesc[0].upper() + famdesc[1:]}: signed GEX ({V2_RULE_SHORT.get(lab, lab)}) and unsigned gross gamma give "
               f"different forecast loss (observed: signed {obs}; {pair})")
    elif kind == "real":
        hyp = (f"{famdesc[0].upper() + famdesc[1:]}: real GEX ({V2_RULE_SHORT.get(lab, lab)}) and the same columns from a random "
               f"earlier day differ in loss (observed: real {obs}; {pair})")
    elif lab.startswith("implied daily component"):
        hyp = f"Building the hybrid on the implied-level mcsGARCH changes its forecast loss (observed: {obs}; {pair})"
    elif lab.startswith("implied+calendar"):
        hyp = f"Adding the option-implied and calendar block changes {base}'s forecast loss (observed: {obs}; {pair})"
    else:
        hyp = f"{fam}, {lab}: {alt_model} and {base} differ in forecast loss (observed: {alt_model} {obs})"
    if kind == "placebo":
        role = "placebo"
    elif dev:
        role = "exploratory"
    elif subset not in ("all", "expiry"):
        role = "robustness"
    elif fam == "hybrid: base vs base+GEX" and subset == "expiry":
        role = "headline"
    else:
        role = "secondary"
    if h1:
        return hyp, role, PS_PARTLY, V2_PS_H1 + ("; " + V2_PS_DEV if dev else ""), SIDE_ONE_PRE
    return hyp, role, PS_NO, V2_PS_DEV if dev else V2_PS_NO, SIDE_CONVERTED


def _v2_holm_check(ctx: Ctx, study, rel, d: pd.DataFrame):
    """Recompute p_holm_in_family (Holm within one model family and one subset, placebo rows excluded) and
    record one cross-check per file. Returns the recomputed values, aligned with d."""
    if "p_holm_in_family" not in d.columns:
        return None
    rec = pd.Series(np.nan, index=d.index)
    n_fam = 0
    for _, g in d.groupby(["family", "group"], sort=False):
        h = g["p_holm_in_family"].notna()
        if h.any():
            n_fam += 1
            rec.loc[g.index[h]] = p_holm(g.loc[h, "p_one_sided"].values)
    a, b = rec.to_numpy(float), pd.to_numeric(d["p_holm_in_family"], errors="coerce").to_numpy(float)
    both = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a[both] - b[both])
    n_mis = int((diff > 1e-6).sum())
    ctx.checks.append(dict(
        study=study, family=f"Holm within each ablation family ({n_fam} families: one model family and one subset, placebo excluded)",
        source_file=f"results/{rel}", method="holm", m_tests=int(np.isfinite(b).sum()), n_compared=int(both.sum()),
        n_mismatch=n_mis, max_abs_diff=float(diff.max()) if diff.size else float("nan"),
        result="match" if n_mis == 0 and both.sum() > 0 else ("MISMATCH" if n_mis else "nothing to compare"),
        note="one-sided p-values, as the study uses them"))
    return rec


def v2_shape_ok(ctx: Ctx, folder="pipeline_nse_v2") -> bool:
    tests = sorted(ctx.res(folder).glob("tests_*.csv"))
    return bool(tests) and all(V2_TESTS_COLS.issubset(pd.read_csv(f, nrows=0).columns) for f in tests)


def adapt_r1v2(ctx: Ctx, study="R1V2", folder="pipeline_nse_v2"):
    """R1V2: the current forecasting contest (the model study; docs/MODEL_ITERATIONS_REAL.md).

    tests_{next15,next30,last_hour}.csv: subset (28 day groups, 24 for the last hour), loss (qlike or mse),
      comparison ("mcsgarch vs X"), mean_diff (mcsGARCH's mean daily loss minus X's: positive means X is
      better; the estimate), stat (Diebold-Mariano t with the HLN correction), p (one-sided, "X has lower
      loss"; Student t with n - 1 df), n (test days).
    ablation_{target}.csv (QLIKE) and ablation_{target}_mse.csv (MSE): family, label, group, base, alt, days,
      change_pct (percent change in loss from base to alt: negative means alt is better; the estimate),
      dm_stat, p_one_sided ("alt has lower loss"), p_holm_in_family (QLIKE only: Holm within one model family
      and one subset, placebo excluded; recomputed and cross-checked).
    seed_noise_next30.csv: dm_p_two_sided between single-seed reruns of the identical model.
    mcs_*.csv: Model Confidence Set p-values, not counted (as for R1).

    Roles: headline = hybrid_base against mcsGARCH (QLIKE, all days and expiry days) and the hybrid's GEX
    ablation on expiry days (QLIKE; RQ1/H1). Secondary = every other model against mcsGARCH and every
    other ablation comparison, QLIKE, all days and expiry days. Robustness = every other subset and every
    MSE test. Placebo = adding placebo GEX (a random earlier day), and two seeds of one model.
    Pre-specification: the v2 models were built after the first contest, so 'no', except the H1 rows
    (the same model without and with a GEX set, on expiry-day test days, any model family or loss): H1 in
    RESEARCH_STRATEGY_PANDEY.md stated that one-sided test before any real data, so these rows are 'partly'
    pre-specified (direction stated, specification later), the label the ledger uses for every study whose
    question was in the strategy but whose specification came later, and they keep their one-sided p. Every
    other one-sided p is doubled to two-sided, as for every study without a prior one-sided statement.
    Duplicates: an ablation row that compares the same two models on the same days under the same loss as
    a tests_*.csv row is counted once (under tests_*.csv); in the files of 2 Nov 2026 no ablation row does,
    because every tests_*.csv row has mcsGARCH on one side. Rows with identical p and DM statistic are
    counted once. R1 (the first run) is a different set of forecasts and is not deduplicated against.
    """
    rows = []
    tested = set()
    for f in sorted(ctx.res(folder).glob("tests_*.csv")):
        rel = f"{folder}/{f.name}"
        target = f.stem.replace("tests_", "")
        d = ctx.read(rel)
        for _, r in d.iterrows():
            comp, subset, loss = clean(r["comparison"]), clean(r["subset"]), clean(r["loss"]).lower()
            md = num(r["mean_diff"])
            a, b, hyp = _v2_vs_mcsgarch(comp, md)
            tested.add((target, loss, subset, frozenset((a, b))))
            if loss != "qlike" or subset not in ("all", "expiry"):
                role = "robustness"
            else:
                role = "headline" if b == "hybrid_base" else "secondary"
            psn = V2_PS_RQ1 if (role == "headline" and subset == "expiry") else V2_PS_NO
            row = _v2_add(ctx, study, hyp, f"{V2_LOSS.get(loss, loss)} loss of forecasts of {TARGETS.get(target, target)}",
                          f"{v2_subset(subset)} (n = {_int(r['n'])} days), horizon {target}", r["p"], SIDE_CONVERTED,
                          role=role, pre_specified=PS_NO, pre_spec_note=psn, source_file=rel,
                          source_key=key(f.name, loss, subset, comp), sign_rule=v2_rule(b), estimate=md,
                          statistic_name="Diebold-Mariano (HLN) t", statistic=r["stat"],
                          p_method=V2_DM.format("model better than mcsGARCH"))
            if row is not None:
                rows.append(ctx.rows.pop())
    n_dup_tests = 0
    for f in sorted(ctx.res(folder).glob("ablation_*.csv")):
        rel = f"{folder}/{f.name}"
        m = re.match(r"ablation_(.+?)(_mse)?$", f.stem)
        target, loss = m.group(1), ("mse" if m.group(2) else "qlike")
        d = ctx.read(rel)
        if not V2_ABL_COLS.issubset(d.columns):
            ctx.warn(f"{rel}: unexpected columns; not read")
            continue
        rec = _v2_holm_check(ctx, study, rel, d)
        for idx, r in d.iterrows():
            subset, base, alt_model = clean(r["group"]), clean(r["base"]), clean(r["alt"])
            if (target, loss, subset, frozenset((base, alt_model))) in tested:
                n_dup_tests += 1
                continue
            hyp, role, ps, psn, side = _v2_ablation(r["family"], r["label"], subset, base, alt_model, num(r["change_pct"]))
            if loss != "qlike" and role in ("headline", "secondary"):
                role = "robustness"
            lab = clean(r["label"])
            has_holm = rec is not None and np.isfinite(num(r.get("p_holm_in_family")))
            row = _v2_add(ctx, study, hyp, f"{V2_LOSS[loss]} loss of forecasts of {TARGETS.get(target, target)}",
                          f"{v2_subset(subset)} (n = {_int(r['days'])} days), horizon {target}", r["p_one_sided"], side,
                          role=role, pre_specified=ps, pre_spec_note=psn, source_file=rel,
                          source_key=key(f.name, r["family"], lab, subset, base, alt_model),
                          sign_rule=V2_RULE.get(lab, ""), estimate=r["change_pct"],
                          statistic_name="Diebold-Mariano (HLN) t", statistic=r["dm_stat"],
                          p_method=V2_DM.format("second model better"),
                          p_study_adjusted=r["p_holm_in_family"] if has_holm else float("nan"),
                          study_adjust_method="Holm within the ablation family (model family and subset, one-sided p)" if has_holm else "")
            if row is not None:
                if has_holm:
                    row["p_study_adjusted_recomputed"] = rec[idx]
                rows.append(ctx.rows.pop())
    rel = f"{folder}/seed_noise_next30.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["dm_p_two_sided"].notna()].iterrows():
            row = ctx.add(study, f"Two single-seed runs of the identical {r['model']} (seeds {r['seed']}) forecast equally well "
                          "(a check of the DM test: no difference exists by design)",
                          f"QLIKE loss of forecasts of {TARGETS['next30']}", "all test days, single-seed reruns", r["dm_p_two_sided"],
                          role="placebo", pre_specified=PS_NO, pre_spec_note="single-seed reruns decided after the development seed study (Iteration 7)",
                          source_file=rel, source_key=key("seed_noise_next30.csv", r["model"], r["seed"]),
                          estimate=r["abs_change_pct"], statistic_name="absolute change in QLIKE (%)",
                          p_method="Diebold-Mariano, two-sided")
            if row is not None:
                rows.append(ctx.rows.pop())
        ctx.skip(study, rel, "single-seed scores", int(d["dm_p_two_sided"].isna().sum()), "a model's score for one seed, no p-value")
    kept, dropped = dedup_exact(rows, cols=("p_reported", "statistic"))
    ctx.rows.extend(kept)
    ctx.skip(study, f"{folder}/ablation_*.csv", "ablation rows that repeat a tests_*.csv comparison", n_dup_tests,
             "same two models, days and loss as a counted tests_*.csv row (counted once there)")
    ctx.skip(study, f"{folder}/", "rows repeating another row (same p and DM statistic)", len(dropped), "same test (counted once)")
    mcs = sorted(ctx.res(folder).glob("mcs_*.csv"))
    for f in mcs:
        ctx.consumed.add(f"{folder}/{f.name}")
    ctx.skip(study, f"{folder}/mcs_*.csv", "Model Confidence Set p-values (QLIKE and MSE)", len(mcs),
             "set-level p-values (already adjusted for the model set) that answer the same question as the DM tests")
    desc = [f for pat in ("scoreboard_*.csv", "gain_by_anchor_*.csv") for f in sorted(ctx.res(folder).glob(pat))]
    ctx.skip(study, f"{folder}/scoreboard_*.csv, gain_by_anchor_*.csv", "scoreboards and gains by anchor time", len(desc),
             "losses relative to mcsGARCH, no p-value; the DM tests behind them are in tests_*.csv")
    return len(kept)


def adapt_r1v2_dev(ctx: Ctx, study="R1V2DEV", folder="pipeline_nse_v2/development"):
    """R1V2DEV: the development runs of the model study (docs/MODEL_ITERATIONS_REAL.md, Section 3).

    The data were cut at 30 June 2025 and the protocol was older (mcsGARCH refitted every 63 days, normal
    DM p-values, single seeds). Counted, as exploratory, because their p-values were looked at and used to
    choose the final design (model families, seeds, tuning, last-hour training), and leaving them out would
    hide part of the search. They are a separate study so that R1V2's own Holm and BY stay about the final
    contest; they count in the project-wide BY.
    dev_scoreboards.csv: run (baseline_v1 = the first run's forecasts on the development days; dev2),
      target (last_hour_pooled = last-hour models trained on every 60-minute window), group, model, days,
      qlike_vs_mcsgarch_pct (estimate), dm_p_vs_mcsgarch (one-sided, model better than mcsGARCH).
    dev_ablation_single_seed.csv: as the final ablation files, plus target; no Holm column.
    dev_seed_noise_next30.csv: dm_p_two_sided between seeds of the identical model (placebo), p_one_sided
      of the GEX ablation for each seed. dev_tuning_next30.csv: dm_p_two_sided (tuned against default
      hyperparameters), ablation_p_one_sided (3-seed GEX ablation, default or tuned).
    Sidedness and pre-specification as for R1V2 (H1 rows on expiry days keep their one-sided p). Rows with
    the same p and estimate (the same forecasts tested twice) are counted once.
    """
    base_dir = ctx.res(folder)
    if not base_dir.exists():
        return 0
    rows = []
    run_desc = {"baseline_v1": "first run's forecasts", "dev2": "development run dev2, single seed"}

    def _keep(row):
        if row is not None:
            rows.append(ctx.rows.pop())

    rel = f"{folder}/dev_scoreboards.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["dm_p_vs_mcsgarch"].notna()].iterrows():
            tgt = clean(r["target"])
            tbase = tgt.replace("_pooled", "")
            hz = f"{tbase}, models trained on every 60-minute window" if tgt.endswith("_pooled") else tbase
            pct = num(r["qlike_vs_mcsgarch_pct"])
            v1 = clean(r["run"]) == "baseline_v1"
            _, b, hyp = _v2_vs_mcsgarch(f"mcsgarch vs {clean(r['model'])}", pct, better_if_positive=False,
                                        desc_a="mcsGARCH refitted every 63 days, our implementation of the paper's model",
                                        desc_b=f"the first run's {clean(r['model'])} (results/pipeline_nse)" if v1 else None)
            _keep(_v2_add(ctx, study, hyp, f"QLIKE loss of forecasts of {TARGETS.get(tbase, tbase)}",
                          f"development window, {v2_subset(r['group'])} (n = {_int(r['days'])} days), horizon {hz}; "
                          f"{run_desc.get(clean(r['run']), clean(r['run']))}", r["dm_p_vs_mcsgarch"], SIDE_CONVERTED,
                          role="exploratory", pre_specified=PS_NO, pre_spec_note=V2_PS_DEV, source_file=rel,
                          source_key=key(r["run"], tgt, r["group"], b), sign_rule=v2_rule(b), estimate=pct,
                          statistic_name="QLIKE relative to mcsGARCH (%)", p_method=V2_DM_DEV.format("model better than mcsGARCH")))
        ctx.skip(study, rel, "mcsGARCH rows", int(d["dm_p_vs_mcsgarch"].isna().sum()), "the reference model itself, no test")
    rel = f"{folder}/dev_ablation_single_seed.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            subset, base, alt_model, tgt = clean(r["group"]), clean(r["base"]), clean(r["alt"]), clean(r["target"])
            tbase = tgt.replace("_pooled", "")
            hyp, role, ps, psn, side = _v2_ablation(r["family"], r["label"], subset, base, alt_model, num(r["change_pct"]), dev=True)
            _keep(_v2_add(ctx, study, hyp, f"QLIKE loss of forecasts of {TARGETS.get(tbase, tbase)}",
                          f"development window, {v2_subset(subset)} (n = {_int(r['days'])} days), horizon {tgt}; single seed",
                          r["p_one_sided"], side, role=role, pre_specified=ps, pre_spec_note=psn, source_file=rel,
                          source_key=key(tgt, r["family"], r["label"], subset, base, alt_model),
                          sign_rule=V2_RULE.get(clean(r["label"]), ""), estimate=r["change_pct"],
                          statistic_name="Diebold-Mariano t", statistic=r["dm_stat"], p_method=V2_DM_DEV.format("second model better")))
    rel = f"{folder}/dev_seed_noise_next30.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        outc = f"QLIKE loss of forecasts of {TARGETS['next30']}"
        samp = "development window, all test days, horizon next30"
        for _, r in d.iterrows():
            comp = clean(r["comparison"])
            if np.isfinite(num(r["dm_p_two_sided"])):
                _keep(ctx.add(study, f"Two single-seed runs of the identical {r['model']} ({comp}) forecast equally well "
                              "(a check of the DM test: no difference exists by design)", outc, samp, r["dm_p_two_sided"],
                              role="placebo", pre_specified=PS_NO, pre_spec_note=V2_PS_DEV, source_file=rel,
                              source_key=key(r["model"], comp), estimate=r["abs_qlike_change_pct"],
                              statistic_name="absolute change in QLIKE (%)", p_method="Diebold-Mariano, two-sided"))
            elif np.isfinite(num(r["p_one_sided"])):
                alt_model, base = [s.strip() for s in clean(r["model"]).split(" vs ", 1)]
                _keep(_v2_add(ctx, study, f"Adding GEX changes the forecast loss (observed: "
                              f"{_v2_obs(num(r['change_pct']), False, 'lower', 'higher')}; {base} vs {alt_model}, {comp})", outc, samp, r["p_one_sided"],
                              SIDE_CONVERTED, role="exploratory", pre_specified=PS_NO, pre_spec_note=V2_PS_DEV,
                              source_file=rel, source_key=key(r["model"], comp), sign_rule=v2_rule(alt_model),
                              estimate=r["change_pct"], statistic_name="percent change in QLIKE",
                              p_method=V2_DM_DEV.format("model with GEX better")))
        ctx.skip(study, rel, "single-seed scores", int((d["dm_p_two_sided"].isna() & d["p_one_sided"].isna()).sum()),
                 "a model's score for one seed, no p-value")
    rel = f"{folder}/dev_tuning_next30.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        outc = f"QLIKE loss of forecasts of {TARGETS['next30']}"
        samp = "development window, all test days, horizon next30; 3 seeds"
        for _, r in d.iterrows():
            mdl = clean(r["model"])
            if np.isfinite(num(r["dm_p_two_sided"])):
                _keep(ctx.add(study, f"Tuning {mdl} inside the training window changes its forecast loss (observed: "
                              f"{_v2_obs(num(r['tuned_vs_default_pct']), False, 'lower', 'higher')})", outc, samp,
                              r["dm_p_two_sided"], role="exploratory", pre_specified=PS_NO, pre_spec_note=V2_PS_DEV,
                              source_file=rel, source_key=key(mdl, "tuned vs default"), sign_rule=v2_rule(mdl),
                              estimate=r["tuned_vs_default_pct"], statistic_name="percent change in QLIKE, tuned vs default",
                              p_method="Diebold-Mariano, two-sided"))
            elif np.isfinite(num(r["ablation_p_one_sided"])):
                m = re.match(r"(\S+) vs (\S+) \((.+)\)", mdl)
                alt_model, base, how = m.groups() if m else (mdl, "", "")
                _keep(_v2_add(ctx, study, f"Adding GEX changes the forecast loss (observed: "
                              f"{_v2_obs(num(r['ablation_change_pct']), False, 'lower', 'higher')}; {base} vs {alt_model}, {how})", outc, samp,
                              r["ablation_p_one_sided"], SIDE_CONVERTED, role="exploratory", pre_specified=PS_NO,
                              pre_spec_note=V2_PS_DEV, source_file=rel, source_key=key(mdl, "ablation"),
                              sign_rule=v2_rule(alt_model), estimate=r["ablation_change_pct"],
                              statistic_name="percent change in QLIKE", p_method=V2_DM_DEV.format("model with GEX better")))
    kept, dropped = dedup_exact(rows, cols=("p_reported", "estimate"))
    ctx.rows.extend(kept)
    ctx.skip(study, f"{folder}/", "rows repeating another development row (same p and estimate)", len(dropped),
             "the same forecasts tested twice (models that are not trained, under both last-hour training schemes; "
             "the seed-0 GEX ablation, which is also the single-seed ablation): counted once")
    for f in sorted(base_dir.glob("*.csv")):
        rel = f"{folder}/{f.name}"
        if rel not in ctx.consumed:
            ctx.consumed.add(rel)
            ctx.skip(study, rel, f.name, 1, "descriptive checks (correlations, ratios), no p-value")
    return len(kept)


def adapt_r1b(ctx: Ctx):
    """Forecast diagnostics. File: results/forecast_diagnostics/tails_and_last_hour_tests.csv.
    Columns: target, model ("A vs B (sample)"), dm_two_sided_p, mean_loss_diff, n_days.
    Rows without dm_two_sided_p are tail shares (descriptive): not tests.
    """
    # The current file reads the second run; the first run's version (which holds the withdrawn HAR
    # last-hour claim) is archived in results/archive_v1_first_run/ and counted too, labelled as such.
    for prefix, label in (("", ""), ("archive_v1_first_run/", " (first run, archived)")):
        rel = prefix + "forecast_diagnostics/tails_and_last_hour_tests.csv"
        if not ctx.has(rel):
            continue
        d = ctx.read(rel)
        t = d[d["dm_two_sided_p"].notna()]
        for _, r in t.iterrows():
            m = re.match(r"(.+?) vs (.+?) \((.+)\)", clean(r["model"]))
            a, b, s = m.groups() if m else (clean(r["model"]), "", "")
            ctx.add("R1b", f"{a} and {b} forecast the last hour equally well (QLIKE){label}",
                    f"QLIKE loss of forecasts of {TARGETS.get(clean(r['target']), clean(r['target']))}",
                    f"{s} (n = {int(num(r['n_days']))})", r["dm_two_sided_p"], role="post hoc",
                    pre_specified=PS_NO, pre_spec_note="diagnostic run after the R1 results",
                    source_file=rel, source_key=key(r["target"], r["model"]), estimate=r["mean_loss_diff"],
                    statistic_name="mean loss difference", p_method="Diebold-Mariano, two-sided")
        ctx.skip("R1b", rel, "tail-share rows (top 1% share of QLIKE)", int(d["dm_two_sided_p"].isna().sum()),
                 "descriptive shares, no p-value")
        for other in ("gains_by_subset.csv", "largest_mcsgarch_errors.csv"):
            if ctx.has(f"{prefix}forecast_diagnostics/{other}"):
                ctx.consumed.add(f"{prefix}forecast_diagnostics/{other}")
                ctx.skip("R1b", f"{prefix}forecast_diagnostics/{other}", "descriptive table", 1, "no p-value")


def _same_num(a: float, b: float, rel=1e-9, tol=1e-12) -> bool:
    """Equal up to floating-point noise (the same test computed by two scripts)."""
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= rel * max(abs(a), abs(b)) + tol)


def _contest_dm_tests(ctx: Ctx):
    """(target, subset, comparison) -> [(study, folder, p, mean_diff, n)] for the QLIKE DM tests of the
    first run (R1, results/pipeline_nse) and the current contest (R1V2, results/pipeline_nse_v2)."""
    out = {}
    for study, folder in (("R1", "pipeline_nse"), ("R1V2", "pipeline_nse_v2")):
        if not ctx.res(folder).exists():
            continue
        for f in sorted(ctx.res(folder).glob("tests_*.csv")):
            d = pd.read_csv(f)
            if not {"subset", "comparison", "p", "mean_diff", "n"}.issubset(d.columns):
                continue
            if "loss" in d.columns:
                d = d[d["loss"].astype(str).str.lower() == "qlike"]
            t = f.stem.replace("tests_", "")
            for _, r in d.iterrows():
                out.setdefault((t, clean(r["subset"]), clean(r["comparison"])), []).append(
                    (study, folder, num(r["p"]), num(r["mean_diff"]), num(r["n"])))
    return out


def adapt_auction(ctx: Ctx):
    """Closing-auction check. Rows on all days duplicate R1 (or R1V2), R2 and R5 and are dropped.

    r1_dm_tests.csv: target, sample, group, comparison, mean_diff, stat, p (one-sided, as R1). Each row is
      matched against the QLIKE DM tests of the first run (results/pipeline_nse/tests_*.csv, R1) and of the
      current contest (results/pipeline_nse_v2/tests_*.csv, R1V2, subsets all, expiry, pre_auction and
      pre_auction_expiry): a row with the same target, days, comparison, n, p and mean_diff is the same test
      and is counted once, under R1 or R1V2. The script reads the v2 forecasts since 2 Nov 2026, so after a
      rerun its rows before 3 Aug 2026 repeat R1V2's pre_auction rows; the committed file still holds the
      first run's numbers, whose all-days rows repeat R1.
    r2_tuesday_split.csv: measure, sample, pct_expiry_vs_week, p_signed_rank (two-sided Wilcoxon).
    r5_without_auction.csv: sample, spec, rule, term, coef, p (HAC, two-sided).
    """
    for rel in ("closing_auction/r1_dm_tests.csv", "archive_v1_first_run/closing_auction/r1_dm_tests.csv"):
        if not ctx.has(rel):
            continue
        d = ctx.read(rel)
        look = _contest_dm_tests(ctx)
        dups = {}
        for _, r in d.iterrows():
            comp, samp, grp = clean(r["comparison"]), clean(r["sample"]), clean(r["group"])
            before = samp.startswith("before")
            sub = ("pre_auction" if before else "all") if grp == "all" else ("pre_auction_expiry" if before else "expiry")
            hit = next((h for h in look.get((clean(r["target"]), sub, comp), [])
                        if _same_num(h[2], num(r["p"])) and _same_num(h[3], num(r["mean_diff"])) and h[4] == num(r["n"])), None)
            if hit is not None:
                k = (samp, hit[0], hit[1])
                dups[k] = dups.get(k, 0) + 1
                continue
            if not before:
                ctx.warn(f"{rel}: a row on all test days matches no contest test ({clean(r['target'])}, {grp}, {comp}); counted under AUCTION")
            if comp.startswith("GEX ablation"):
                a, b = [s.strip() for s in comp.split(":", 1)[1].split(" vs ")]
                hyp = f"Adding GEX features ({b}) lowers forecast loss against {a}" + (", before the closing auction" if before else "")
            else:
                b = comp.split(" vs ")[1].strip()
                hyp = f"{b} forecasts better than mcsGARCH" + (", before the closing auction" if before else "")
            ctx.add("AUCTION", hyp, f"QLIKE loss of forecasts of {TARGETS.get(clean(r['target']), r['target'])}",
                    f"{r['group']} test days{' before 3 Aug 2026' if before else ''} (n = {int(num(r['n']))})", r["p"],
                    role="robustness", pre_specified=PS_NO,
                    pre_spec_note="added after the closing auction was flagged; one-sided test as pre-stated for R1",
                    source_file=rel, source_key=key(r["target"], r["sample"], r["group"], comp),
                    sidedness=SIDE_ONE_PRE, estimate=r["mean_diff"], statistic_name="Diebold-Mariano (HLN) t",
                    statistic=r["stat"], p_method="Diebold-Mariano, one-sided (model better)")
        for (samp, study, folder), n in dups.items():
            where = "" if study == "R1" else (", subsets all and expiry" if not samp.startswith("before") else
                                              ", subsets pre_auction and pre_auction_expiry")
            ctx.skip("AUCTION", rel, f"rows on {samp}" if samp.startswith("all") else f"rows on test days {samp}", n,
                     f"identical to results/{folder}/tests_*.csv{where} (counted under {study})")
    rel = "closing_auction/r2_tuesday_split.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        keep = d[~d["sample"].str.endswith(", all")]
        for _, r in keep.iterrows():
            ctx.add("AUCTION", f"Tuesday expiry days differ from the rest of their week in {MEASURES.get(r['measure'], r['measure'])}",
                    f"log({r['measure']} on expiry day / rest of week)", f"{r['sample']} ({int(num(r['weeks']))} weeks)",
                    r["p_signed_rank"], role="robustness", pre_specified=PS_NO,
                    pre_spec_note="split added after the closing auction was flagged", source_file=rel,
                    source_key=key(r["measure"], r["sample"]), estimate=r["pct_expiry_vs_week"],
                    statistic_name="percent difference", p_method="Wilcoxon signed-rank, two-sided")
        ctx.skip("AUCTION", rel, "'Tuesday regime, all' rows", int(d["sample"].str.endswith(", all").sum()),
                 "same test as results/real_day_level/r2_expiry_effect_by_regime.csv (counted under R2)")
    rel = "closing_auction/r5_without_auction.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        keep = d[d["sample"].str.startswith("before")]
        for _, r in keep.iterrows():
            if r["term"] == "r_rod":
                hyp = "The last 30 minutes reverse (or continue) the rest-of-day return, before the closing auction"
            else:
                hyp = f"The end-of-day reversal is stronger after negative GEX ({RULES.get(r['rule'], r['rule'])}), before the closing auction"
            ctx.add("AUCTION", hyp, "last-30-minute return", f"days before 3 Aug 2026 (n = {int(num(r['n']))}); {r['spec']}",
                    r["p"], role="robustness", pre_specified=PS_NO, pre_spec_note="added after the closing auction was flagged",
                    source_file=rel, source_key=key(r["sample"], r["spec"], r["rule"], r["term"]),
                    sign_rule=RULES.get(clean(r["rule"]), clean(r["rule"])), estimate=r["coef"],
                    statistic_name=f"coefficient on {r['term']}", p_method="OLS with HAC standard errors, two-sided")
        ctx.skip("AUCTION", rel, "'all days' rows", int((~d["sample"].str.startswith("before")).sum()),
                 "identical to results/momentum_gex/regressions.csv (counted under R5)")
    if ctx.has("closing_auction/auction_evidence.csv"):
        ctx.consumed.add("closing_auction/auction_evidence.csv")
        ctx.skip("AUCTION", "closing_auction/auction_evidence.csv", "evidence that the auction exists", 1, "descriptive, no p-value")


def adapt_r2(ctx: Ctx):
    """R2. File: results/real_day_level/r2_expiry_effect_by_regime.csv.
    Per-regime rows: wilcoxon_p_rel_eq_0 (two-sided, mean log ratio of expiry day to rest of week = 0).
    Regime-difference rows ("A minus B"): mannwhitney_p (two-sided).
    """
    rel = "real_day_level/r2_expiry_effect_by_regime.csv"
    if not ctx.has(rel):
        return
    d = ctx.read(rel)
    for _, r in d.iterrows():
        meas = MEASURES.get(clean(r["measure"]), clean(r["measure"]))
        reg = clean(r["regime"])
        if " minus " in reg:
            ctx.add("R2", f"The expiry-day premium in {meas} differs between regimes ({reg})",
                    f"log({r['measure']} on expiry day / rest of week)", f"regimes {reg}", r["mannwhitney_p"],
                    role="secondary", pre_specified=PS_NO, pre_spec_note="regime comparison not stated in advance",
                    source_file=rel, source_key=key(r["measure"], reg), estimate=r["mean_rel"],
                    statistic_name="difference in mean log ratio", p_method="Mann-Whitney, two-sided")
        else:
            ctx.add("R2", f"Expiry days have a different {meas} from the rest of their week ({reg})",
                    f"log({r['measure']} on expiry day / rest of week)", f"regime {reg} ({clean(r['weeks'])} weeks)",
                    r["wilcoxon_p_rel_eq_0"], role="headline", pre_specified=PS_PARTLY,
                    pre_spec_note="RQ3 (new regimes) stated in RESEARCH_STRATEGY_PANDEY.md; measures and test chosen later",
                    source_file=rel, source_key=key(r["measure"], reg), estimate=r["mean_rel"],
                    statistic_name="mean log ratio", p_method="Wilcoxon signed-rank, two-sided")


def adapt_r3(ctx: Ctx):
    """R3. Files: results/real_day_level/r3_gex_regressions.csv (columns controls, spec, term, coef,
    se_hac, t, p, n_days; only the GEX terms gex_l1 and gex_l1_x_exp are tests, other terms are controls)
    and r3_robustness.csv (rule, variant, term, coef, se_hac, p, n_days, p_holm, p_bh; Holm and BH
    over the 30 GEX-term rows). The per_sd GEX rows of r3_robustness repeat the regime x expiry rows of
    r3_gex_regressions (identical p) and are not counted twice.
    """
    rel = "real_day_level/r3_gex_regressions.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        g = d[d["term"].isin(["gex_l1", "gex_l1_x_exp"])]
        for _, r in g.iterrows():
            rule = clean(r["spec"]).replace("gex_", "")
            headline = clean(r["controls"]) != "none"
            hyp = ("Previous evening's GEX predicts the day's realized variance on ordinary days"
                   if r["term"] == "gex_l1" else
                   "The GEX effect on realized variance differs on expiry days (GEX x expiry)")
            row = ctx.add("R3", hyp, "log daily realized variance", f"all days (n = {int(num(r['n_days']))}); controls: {r['controls']}",
                    r["p"], role="headline" if headline else "secondary", pre_specified=PS_PARTLY,
                    pre_spec_note="H2 (hedging sign) and H3 stated in RESEARCH_STRATEGY_PANDEY.md; specification chosen later",
                    source_file=rel, source_key=key(r["controls"], r["spec"], r["term"]),
                    sign_rule=RULES.get(rule, rule), estimate=r["coef"], statistic_name="t (HAC)", statistic=r["t"],
                    p_method="OLS with HAC standard errors, two-sided")
            if headline:
                _set(row, "_h18", ("F1", rule, "regime x expiry", clean(r["term"])))
        ctx.skip("R3", rel, "control and intercept coefficients", int(len(d) - len(g)), "not hypotheses of the study")
    rel = "real_day_level/r3_robustness.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        gex = d["term"].isin(["gex_l1", "gex_l1_x_exp"])
        rec_h = ctx.check("R3", "30 GEX-term rows of r3_robustness", rel, "holm", d.loc[gex, "p"], d.loc[gex, "p_holm"])
        rec_b = ctx.check("R3", "30 GEX-term rows of r3_robustness", rel, "bh", d.loc[gex, "p"], d.loc[gex, "p_bh"])
        hmap = dict(zip(d.index[gex], rec_h))
        for idx, r in d.iterrows():
            if gex[idx] and clean(r["variant"]) == "per_sd":
                continue
            term = clean(r["term"])
            hyp = {"gex_l1": "Previous evening's GEX predicts realized variance on ordinary days",
                   "gex_l1_x_exp": "The GEX effect differs on expiry days (GEX x expiry)",
                   }.get(term, "The total GEX effect on expiry days (GEX + GEX x expiry) is not zero")
            row = ctx.add("R3", f"{hyp}; GEX scaled as '{r['variant']}'", "log daily realized variance",
                          f"all days (n = {int(num(r['n_days']))})", r["p"], role="robustness", pre_specified=PS_NO,
                          pre_spec_note="robustness variants chosen after the first R3 run",
                          source_file=rel, source_key=key(r["rule"], r["variant"], term),
                          sign_rule=RULES.get(clean(r["rule"]), clean(r["rule"])), estimate=r["coef"],
                          statistic_name="coefficient", p_method="OLS with HAC standard errors, two-sided",
                          p_study_adjusted=r["p_holm"] if gex[idx] else float("nan"),
                          study_adjust_method="Holm over the 30 GEX-term rows of r3_robustness" if gex[idx] else "")
            if row is not None and gex[idx]:
                row["p_study_adjusted_recomputed"] = hmap[idx]
                if clean(r["variant"]) == "share":
                    row["_h18"] = ("F1", clean(r["rule"]), "robustness: share", term)
        ctx.skip("R3", rel, "per_sd GEX-term rows", int((gex & (d["variant"] == "per_sd")).sum()),
                 "identical p to the regime x expiry rows of r3_gex_regressions.csv (counted once)")


def adapt_r4(ctx: Ctx):
    """R4. File: results/real_day_level/r4_pinning.csv; rows with mannwhitney_p (two-sided)."""
    rel = "real_day_level/r4_pinning.csv"
    if not ctx.has(rel):
        return
    d = ctx.read(rel)
    for _, r in d[d["mannwhitney_p"].notna()].iterrows():
        tgt = {"max_oi_strike": "the strike with the most open interest", "max_pain": "the max-pain strike"}.get(
            clean(r["target"]), clean(r["target"]))
        ctx.add("R4", f"On expiry days the index moves toward {tgt} more than on other days",
                "change in distance to the target strike (bp)", clean(r["group"]), r["mannwhitney_p"],
                role="exploratory", pre_specified=PS_NO, pre_spec_note="first look, not stated in advance",
                source_file=rel, source_key=key(r["target"], r["group"]), estimate=r["mean_change_bp"],
                statistic_name="mean change (bp)", p_method="Mann-Whitney, two-sided")
    ctx.skip("R4", rel, "descriptive rows", int(d["mannwhitney_p"].isna().sum()), "group summaries without a p-value")


def adapt_r5(ctx: Ctx):
    """R5. Files: results/momentum_gex/regressions.csv (spec, rule, subset, term, coef, se_hac, p, n,
    p_holm_family, p_bh_family, family_size). Tests: the 36 interaction terms (r_rod_x_neg, r_rod_x_gex;
    family_size 36) and the momentum-only r_rod rows; other coefficients are controls.
    results/momentum_gex/stability.csv (rule, part, r_rod_x_neg, p): sub-period re-estimates.
    gamma_states.csv has t statistics only: not counted.
    """
    rel = "momentum_gex/regressions.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        fam = d["family_size"].notna()
        rec_h = ctx.check("R5", "36 interaction terms (family_size 36)", rel, "holm", d.loc[fam, "p"], d.loc[fam, "p_holm_family"])
        ctx.check("R5", "36 interaction terms (family_size 36)", rel, "bh", d.loc[fam, "p"], d.loc[fam, "p_bh_family"])
        hmap = dict(zip(d.index[fam], rec_h))
        mom = (d["spec"] == "momentum only") & (d["term"] == "r_rod")
        for idx, r in d[fam | mom].iterrows():
            subset = clean(r["subset"])
            if mom[idx]:
                hyp = "The last 30 minutes reverse (or continue) the rest-of-day return"
                role, rule = "secondary", ""
            else:
                rule = RULES.get(clean(r["rule"]), clean(r["rule"]))
                hyp = ("The end-of-day reversal is stronger after negative GEX" if r["term"] == "r_rod_x_neg"
                       else "The end-of-day reversal changes with the level of GEX")
                headline = subset == "all" and clean(r["spec"]) in ("binary", "binary + VIX and size controls")
                role = "headline" if headline else "secondary"
            row = ctx.add("R5", hyp, "last-30-minute return", f"{subset} days (n = {int(num(r['n']))}); {r['spec']}",
                          r["p"], role=role, pre_specified=PS_NO,
                          pre_spec_note="R5 was added after the first real-data run; no prior statement",
                          source_file=rel, source_key=key(r["spec"], r["rule"], subset, r["term"]),
                          sign_rule=rule, estimate=r["coef"], statistic_name=f"coefficient on {r['term']}",
                          p_method="OLS with HAC standard errors, two-sided",
                          p_study_adjusted=r["p_holm_family"] if fam[idx] else float("nan"),
                          study_adjust_method="Holm over the 36 interaction terms" if fam[idx] else "")
            if row is not None and fam[idx]:
                row["p_study_adjusted_recomputed"] = hmap[idx]
                if role == "headline":
                    row["_h18"] = ("F2", clean(r["rule"]), "binary, all days" if clean(r["spec"]) == "binary" else "controls, all days", "r_rod_x_neg")
        ctx.skip("R5", rel, "control coefficients (const, neg, regime terms, VIX terms, r_rod in interaction models)",
                 int(len(d) - (fam | mom).sum()), "not hypotheses of the study")
    rel = "momentum_gex/stability.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["p"].notna()].iterrows():
            ctx.add("R5", f"The reversal is stronger after negative GEX, within {r['part']}", "last-30-minute return",
                    f"{r['part']} ({int(num(r['days']))} days, {int(num(r['neg_days']))} negative)", r["p"],
                    role="robustness", pre_specified=PS_NO, source_file=rel, source_key=key(r["rule"], r["part"]),
                    sign_rule=RULES.get(clean(r["rule"]), clean(r["rule"])), estimate=r["r_rod_x_neg"],
                    statistic_name="coefficient on r_rod_x_neg", p_method="OLS with HAC standard errors, two-sided")
        ctx.skip("R5", rel, "sub-period without negative-GEX days", int(d["p"].isna().sum()), "no p-value")
    if ctx.has("momentum_gex/gamma_states.csv"):
        ctx.consumed.add("momentum_gex/gamma_states.csv")
        ctx.skip("R5", "momentum_gex/gamma_states.csv", "gamma-state summaries", 1, "t statistics only, no p-value")


def adapt_repl(ctx: Ctx):
    """Replication of the paper. Main p: day-level Mann-Whitney (day_mwu_p); HAC and bootstrap p are
    alternative p-values of the same comparison (alt_p). The week-relative change (weekrel_p_mwu, with
    weekrel_p_welch as alternative) is a separate hypothesis.
    Files: table5_volume_daylevel.csv, table6_volatility.csv (minute_mwu_p is the paper's minute-level
    test, kept as an alternative and flagged), table2_persistence_change.csv (lr_p_value).
    paper_printed_p is the paper's own p-value: not ours, not counted.
    """
    rel = "replication_nse/table5_volume_daylevel.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            paper_measure = clean(r["measure"]) == "index_constituent_volume_shares"
            sub = clean(r["subsample"])
            samp = f"{r['index']}, {sub}, before = {r['before']} ({int(num(r['n_days_before']))} vs {int(num(r['n_days_after']))} days)"
            ctx.add("REPL", f"{r['index']} {r['measure_description']} changed after the expiry-day change ({sub} days)",
                    clean(r["measure"]), samp, r["day_mwu_p"],
                    role=("headline" if sub == "all" else "secondary") if paper_measure else "exploratory",
                    pre_specified=PS_PARTLY if paper_measure else PS_NO,
                    pre_spec_note="H4 stated in RESEARCH_STRATEGY_PANDEY.md; measures and windows chosen later" if paper_measure else "additional measure",
                    source_file=rel, source_key=key("level", r["before"], r["index"], r["measure"], sub),
                    estimate=r["pct_change_median"], statistic_name="percent change in median",
                    p_method="day-level Mann-Whitney, two-sided",
                    alt_p=alt(hac_p=r["hac_p"], boot_p=r["boot_p"]) + ("; boot_p=0 means below 1/2000" if num(r["boot_p"]) == 0 else ""))
            if np.isfinite(num(r["weekrel_p_mwu"])):
                ctx.add("REPL", f"{r['index']} {r['measure_description']}: the {sub} day stands out more from its own week after the change",
                        f"week-relative log {r['measure']}", samp, r["weekrel_p_mwu"], role="exploratory", pre_specified=PS_NO,
                        pre_spec_note="our addition, not in the paper", source_file=rel,
                        source_key=key("weekrel", r["before"], r["index"], r["measure"], sub), estimate=r["weekrel_diff"],
                        statistic_name="change in week-relative log value", p_method="Mann-Whitney, two-sided",
                        alt_p=alt(weekrel_p_welch=r["weekrel_p_welch"]))
        ctx.skip("REPL", rel, "paper_printed_p", int(d["paper_printed_p"].notna().sum()), "the paper's own printed p-values, not our tests")
    rel = "replication_nse/table6_volatility.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            sub = clean(r["subsample"])
            samp = f"{r['index']}, {sub}, before = {r['before']} ({int(num(r['n_days_before']))} vs {int(num(r['n_days_after']))} days)"
            ctx.add("REPL", f"{r['index']} intraday volatility changed after the expiry-day change ({sub} days)",
                    "absolute 1-minute return (median)", samp, r["day_mwu_p"],
                    role="headline" if sub == "all" else "secondary", pre_specified=PS_PARTLY,
                    pre_spec_note="H4 stated in RESEARCH_STRATEGY_PANDEY.md; windows chosen later",
                    source_file=rel, source_key=key("level", r["before"], r["index"], sub),
                    estimate=r["pct_change_median"], statistic_name="percent change in median",
                    p_method="day-level Mann-Whitney, two-sided",
                    alt_p=alt(hac_p=r["hac_p"], boot_p=r["boot_p"], minute_mwu_p_invalid=r["minute_mwu_p"]),
                    note="minute_mwu_p is the paper's minute-level test, which ignores dependence; not used")
            if np.isfinite(num(r["weekrel_p_mwu"])):
                ctx.add("REPL", f"{r['index']} volatility on {sub} days stands out more from its own week after the change",
                        "week-relative log volatility", samp, r["weekrel_p_mwu"], role="exploratory", pre_specified=PS_NO,
                        pre_spec_note="our addition, not in the paper", source_file=rel,
                        source_key=key("weekrel", r["before"], r["index"], sub), estimate=r["weekrel_diff"],
                        statistic_name="change in week-relative log value", p_method="Mann-Whitney, two-sided",
                        alt_p=alt(weekrel_p_welch=r["weekrel_p_welch"]))
        ctx.skip("REPL", rel, "paper_printed_p", int(d["paper_printed_p"].notna().sum()), "the paper's own printed p-values, not our tests")
    rel = "replication_nse/table2_persistence_change.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            ctx.add("REPL", f"{r['index']} intraday volatility persistence (alpha + beta) changed after the expiry-day change",
                    "intraday GARCH persistence", f"{r['index']}, before = {r['before']}", r["lr_p_value"],
                    role="headline", pre_specified=PS_NO, pre_spec_note="persistence change is the paper's claim; not in H4",
                    source_file=rel, source_key=key(r["before"], r["index"]),
                    estimate=r["diff_persistence_after_minus_before"], statistic_name="LR statistic",
                    statistic=r["lr_equal_persistence_stat"], sidedness=SIDE_NONDIR, p_method="likelihood ratio")
    for f, why in [("tables34_changes.csv", "bootstrap intervals only, no p-value"),
                   ("tables34_connectedness.csv", "p_rule / p_used are VAR lag settings, not p-values"),
                   ("tables34_connectedness_grid.csv", "p_rule / p_used are VAR lag settings, not p-values"),
                   ("table2_mcsgarch.csv", "z statistics only (ours_z), no p-value"),
                   ("table1_descriptive.csv", "ADF and Anderson-Darling statistics only"),
                   ("paper_printed_checks.csv", "checks of the paper's arithmetic, not hypothesis tests"),
                   ("rekey_check.csv", "re-keying of printed numbers, not hypothesis tests")]:
        if ctx.has(f"replication_nse/{f}"):
            ctx.consumed.add(f"replication_nse/{f}")
            ctx.skip("REPL", f"replication_nse/{f}", f, 1, why)


def adapt_pin(ctx: Ctx):
    """Pinning study (10,000-draw permutation p-values, two-sided |diff|).

    t1_clustering.csv: per row four hypotheses: p_perm (expiry vs other days; p_perm_weekday_strata is an
      alternative), p_perm_share_within_tenth, p_expiry_vs_uniform, p_nonexpiry_vs_uniform (placebo).
    t1_regime_heterogeneity.csv: p_perm. t3_target_strikes.csv and t4_intraday_movement.csv: p_value.
    t4_intraday_by_regime.csv: p_perm (rows whose diff repeats a t1 row are dropped).
    t5_gamma_conditioning.csv: p_perm, p_holm_3rules (P2). t7_robustness.csv: p_value.
    P1 (Nifty, all) and P2 (Nifty C1 expiry rows) were stated in the script's docstring, not time-stamped.
    """
    pre_note = "written into the script's docstring before results, per the analysis log; " + COMMITTED_WITH_RESULTS
    rel = "pinning/t1_clustering.csv"
    t1_diffs = set()
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            sym, s = clean(r["symbol"]), clean(r["sample"])
            t1_diffs.add(round(num(r["diff_pts"]), 6))
            p1 = sym == "NIFTY" and s == "all"
            regime = sym == "NIFTY" and s in ("multi_weekly", "single_weekly_thu", "single_weekly_tue")
            role = "primary" if p1 else ("secondary" if regime else "exploratory")
            ps = PS_STATED if p1 else (PS_PARTLY if regime else PS_NO)
            psn = pre_note if p1 else ("the three-regime split was part of the brief" if regime else "")
            samp = f"{sym}, {s} ({int(num(r['n_expiry']))} expiry vs {int(num(r['n_nonexpiry']))} other days)"
            base = dict(source_file=rel, sign_rule="")
            ctx.add("PIN", f"{sym} closes nearer a strike on expiry days than on other days ({s})" + (" [P1]" if p1 else ""),
                    "distance from official close to nearest strike", samp, r["p_perm"], role=role,
                    pre_specified=ps, pre_spec_note=psn, source_key=key(sym, s, "p_perm"), estimate=r["diff_pts"],
                    statistic_name="difference in mean distance (points)", p_method="permutation, two-sided",
                    alt_p=alt(p_perm_weekday_strata=r["p_perm_weekday_strata"]), **base)
            ctx.add("PIN", f"{sym}: more expiry-day closes within a tenth of the strike spacing than on other days ({s})",
                    "share of closes within a tenth of the spacing", samp, r["p_perm_share_within_tenth"],
                    role="exploratory", pre_specified=PS_NO, source_key=key(sym, s, "share_within_tenth"),
                    estimate=num(r["share_within_tenth_expiry"]) - num(r["share_within_tenth_nonexpiry"]),
                    statistic_name="difference in share", p_method="permutation, two-sided", **base)
            ctx.add("PIN", f"{sym} expiry-day close distances differ from the no-pinning (uniform) benchmark ({s})",
                    "distance from official close to nearest strike", samp, r["p_expiry_vs_uniform"],
                    role="exploratory", pre_specified=PS_NO, source_key=key(sym, s, "expiry_vs_uniform"),
                    estimate=r["mean_dist_expiry_pts"], statistic_name="mean distance (points)",
                    p_method="test against uniform, two-sided", **base)
            ctx.add("PIN", f"{sym} non-expiry close distances differ from the uniform benchmark ({s})",
                    "distance from official close to nearest strike", samp, r["p_nonexpiry_vs_uniform"],
                    role="placebo", pre_specified=PS_NO, source_key=key(sym, s, "nonexpiry_vs_uniform"),
                    estimate=r["mean_dist_nonexpiry_pts"], statistic_name="mean distance (points)",
                    p_method="test against uniform, two-sided", **base)
    rel = "pinning/t1_regime_heterogeneity.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            ctx.add("PIN", f"{r['symbol']}: the expiry-day pinning gap differs between {r['regime_a']} and {r['regime_b']}",
                    "gap in distance to nearest strike", f"{r['symbol']}, {int(num(r['n_expiry_a']))} vs {int(num(r['n_expiry_b']))} expiry days",
                    r["p_perm"], role="exploratory", pre_specified=PS_NO, source_file=rel,
                    source_key=key(r["symbol"], r["regime_a"], r["regime_b"]), estimate=r["gap_a_minus_gap_b_pts"],
                    statistic_name="difference in gap (points)", p_method="permutation, two-sided")
    rel = "pinning/t3_target_strikes.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["p_value"].notna()].iterrows():
            g = clean(r["group"]).replace("test: ", "")
            ctx.add("PIN", f"{r['symbol']} {r['target']}: {g}", "hits of / moves toward the target strike",
                    f"{r['symbol']} ({clean(r['n'])})", r["p_value"], role="exploratory", pre_specified=PS_NO,
                    source_file=rel, source_key=key(r["symbol"], r["target"], g), estimate=r["estimate"],
                    statistic_name="estimate", p_method=f"{clean(r['test'])}, two-sided")
        ctx.skip("PIN", rel, "descriptive rows", int(d["p_value"].isna().sum()), "group summaries without a p-value")
    rel = "pinning/t4_intraday_movement.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["p_value"].notna()].iterrows():
            g = clean(r["group"]).replace("test: ", "")
            ctx.add("PIN", f"{r['symbol']} from {r['start']}: {g}", "intraday distance to / move toward nearest strike",
                    f"{r['symbol']}, start {r['start']}", r["p_value"], role="exploratory", pre_specified=PS_NO,
                    source_file=rel, source_key=key(r["symbol"], r["start"], g), estimate=r["estimate"],
                    statistic_name="estimate (points)", p_method="permutation or sign-flip, two-sided")
        ctx.skip("PIN", rel, "descriptive rows", int(d["p_value"].isna().sum()), "group summaries without a p-value")
    rel = "pinning/t4_intraday_by_regime.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        n_dup = 0
        for _, r in d.iterrows():
            if round(num(r["diff_pts"]), 6) in t1_diffs:
                n_dup += 1
                continue
            ctx.add("PIN", f"{r['symbol']} ({r['regime']}): {r['price']} is nearer a strike on expiry days",
                    "distance to nearest strike", f"{r['symbol']}, {r['regime']} ({int(num(r['n_expiry']))} vs {int(num(r['n_nonexpiry']))} days)",
                    r["p_perm"], role="exploratory", pre_specified=PS_NO, source_file=rel,
                    source_key=key(r["symbol"], r["regime"], r["price"]), estimate=r["diff_pts"],
                    statistic_name="difference (points)", p_method="permutation, two-sided")
        ctx.skip("PIN", rel, "rows identical to t1_clustering", n_dup, "same difference and test as a t1 row (counted once)")
    rel = "pinning/t5_gamma_conditioning.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        hol = d["p_holm_3rules"].notna()
        for sym in d.loc[hol, "symbol"].unique():
            m = hol & (d["symbol"] == sym)
            rec = ctx.check("PIN", f"{sym} C1 expiry rows, Holm over 3 rules", rel, "holm", d.loc[m, "p_perm"], d.loc[m, "p_holm_3rules"])
            for i, v in zip(d.index[m], rec):
                d.loc[i, "_rec"] = v
        for idx, r in d.iterrows():
            day = clean(r["day_type"])
            c1 = clean(r["conditioning"]).startswith("C1")
            p2 = c1 and r["symbol"] == "NIFTY" and day == "expiry"
            if p2:
                role, ps, psn = "primary", PS_STATED, pre_note + " [P2]"
            elif day in ("day before expiry", "all non-expiry"):
                role, ps, psn = "placebo", PS_NO, ""
            else:
                role, ps, psn = "exploratory", PS_NO, ""
            row = ctx.add("PIN", f"{r['symbol']} {day}: pinning is stronger when dealer gamma near the strike is positive ({r['conditioning']})" + (" [P2]" if p2 else ""),
                          "distance from close to nearest strike", f"{r['symbol']}, {day} ({_int(r['n_pos'])} positive, {_int(r['n_neg'])} negative days)",
                          r["p_perm"], role=role, pre_specified=ps, pre_spec_note=psn, source_file=rel,
                          source_key=key(r["symbol"], r["conditioning"], r["rule"], day),
                          sign_rule=RULES.get(clean(r["rule"]), clean(r["rule"])), estimate=r["diff_pos_minus_neg_pts"],
                          statistic_name="difference positive minus negative (points)", p_method="permutation, two-sided",
                          p_study_adjusted=r["p_holm_3rules"], study_adjust_method="Holm over 3 sign rules" if hol[idx] else "")
            if row is not None and hol[idx]:
                row["p_study_adjusted_recomputed"] = r["_rec"]
    rel = "pinning/t7_robustness.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["p_value"].notna()].iterrows():
            ctx.add("PIN", f"{r['symbol']}: {r['check']}", "distance from close to nearest strike", f"{r['symbol']} ({r['n']})",
                    r["p_value"], role="robustness", pre_specified=PS_NO, source_file=rel,
                    source_key=key(r["symbol"], r["check"]), estimate=r["estimate"], statistic_name="difference (points)",
                    p_method="permutation, two-sided")
    for f in sorted(ctx.res("pinning").glob("*.csv")):
        rel = f"pinning/{f.name}"
        if rel not in ctx.consumed:
            ctx.consumed.add(rel)
            ctx.skip("PIN", rel, f.name, 1, "descriptive table, power calculation or sign agreement; no hypothesis test")


EVENT_TABLE3 = {"rv5", "rs", "rv_first_hour", "rv_last30", "opt_premium"}


def adapt_event(ctx: Ctx):
    """Event study. Main p: HAC (hac_p), the p the study adjusts (bh_q_within_window = BH on hac_p
    within each window). Bootstrap, Welch, Mann-Whitney and placebo-date (ri_p) p-values are
    alternatives (alt_p). pretrend_p is a diagnostic: not counted.
    Files: event_tests.csv, premium_by_regime.csv (hac_p; wilcoxon_p alternative),
    own_premium_by_halfyear.csv, did_day_regressions.csv (p), robustness_event_c.csv (hac_p),
    event_b_paper_window_raw_vs_relative.csv (raw_day_mwu_p; rel_hac_p).
    """
    pre_note = "stated as fixed before any result was seen (EVENT_STUDY.md); " + COMMITTED_WITH_RESULTS
    rel = "event_study/event_tests.csv"
    paper_hac = set()
    if ctx.has(rel):
        d = ctx.read(rel)
        for w in d["window"].unique():
            m = d["window"] == w
            rec = ctx.check("EVENT", f"event_tests window '{w}' (BH on hac_p)", rel, "bh", d.loc[m, "hac_p"], d.loc[m, "bh_q_within_window"])
            d.loc[m, "_rec"] = rec
        for _, r in d.iterrows():
            spec, meas, win = clean(r["spec_id"]), clean(r["measure"]), clean(r["window"])
            if win == "paper window":
                paper_hac.add(round(num(r["hac_p"]), 12))
            if "_Fri" in spec:
                role, ps, psn = "placebo", PS_NO, "negative control (no rule change on Fridays)"
            elif spec == "c_DiD_main" and meas == "rv5" and win == "6 months":
                role, ps, psn = "primary", PS_STATED, pre_note
            elif (spec == "c_DiD_main" and meas == "rv5") or "incl_auction" in spec:
                role, ps, psn = "robustness", PS_NO, ""
            else:
                role, ps, psn = "exploratory", PS_NO, "the study labels only the primary test"
            row = ctx.add("EVENT", f"Event {r['event']}: {clean(r['comparison'])} changed in {MEASURES.get(meas, meas)}",
                    f"week-relative {meas}", f"{r['sample']} weeks, {win} ({clean(r['n_pre'])} vs {clean(r['n_post'])} weeks)",
                    r["hac_p"], role=role, pre_specified=ps, pre_spec_note=psn, source_file=rel,
                    source_key=key(spec, meas, win), estimate=r["diff"], statistic_name="change (log points)",
                    p_method="HAC (Newey-West), two-sided",
                    alt_p=alt(boot_p=r["boot_p"], welch_p=r["welch_p"], mwu_p=r["mwu_p"], placebo_ri_p=r["ri_p"]),
                    p_study_adjusted=r["bh_q_within_window"], study_adjust_method="BH within window (hac_p)")
            _set(row, "p_study_adjusted_recomputed", r["_rec"])
        ctx.skip("EVENT", rel, "pre-trend p-values (pretrend_p)", int(d["pretrend_p"].notna().sum()),
                 "diagnostic of the design, not a hypothesis of the study")
    rel = "event_study/premium_by_regime.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        d = pd.concat([d[d["target"] == "own"], d[d["target"] != "own"]])
        seen = set()
        n_dup = 0
        for _, r in d.iterrows():
            k = (r["index"], r["sample"], r["measure"], r["regime"], round(num(r["mean"]), 10), round(num(r["hac_p"]), 10))
            if k in seen:
                n_dup += 1
                continue
            seen.add(k)
            tgt, meas, samp = clean(r["target"]), clean(r["measure"]), clean(r["sample"])
            if meas in ("rv1_over_rv5", "rs_over_rv5"):
                role, psn = "post hoc", "frequency diagnostics added after seeing results (EVENT_STUDY.md 5.7)"
            elif tgt != "own":
                role, psn = "post hoc", "all-weekday end-of-day table added after seeing results (EVENT_STUDY.md 5.6)"
            elif samp == "clean_weekly" and meas in EVENT_TABLE3:
                role, psn = "secondary", "Table 3 of EVENT_STUDY.md"
            else:
                role, psn = "exploratory", ""
            day = "own expiry day" if tgt == "own" else tgt
            ctx.add("EVENT", f"{r['index']} {day} has a premium in {MEASURES.get(meas, meas)} over the rest of the week ({r['regime']})",
                    f"week-relative {meas}", f"{samp} weeks, {r['regime']} ({clean(r['n_weeks'])} weeks)", r["hac_p"],
                    role=role, pre_specified=PS_NO, pre_spec_note=psn, source_file=rel,
                    source_key=key(r["index"], tgt, samp, meas, r["regime"]), estimate=r["mean"],
                    statistic_name="mean week-relative log value", p_method="HAC (Newey-West), two-sided",
                    alt_p=alt(wilcoxon_p=r["wilcoxon_p"]))
        ctx.skip("EVENT", rel, "weekday rows identical to the own-expiry row", n_dup, "same weeks and values as an own-expiry row (counted once)")
    rel = "event_study/own_premium_by_halfyear.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["hac_p"].notna()].iterrows():
            meas = clean(r["measure"])
            ctx.add("EVENT", f"{r['index']} own expiry day has a premium in {MEASURES.get(meas, meas)} in {r['half_year']}",
                    f"week-relative {meas}", f"{r['half_year']}, {r['weekday']} ({clean(r['n_weeks'])} weeks)", r["hac_p"],
                    role="exploratory", pre_specified=PS_NO, source_file=rel,
                    source_key=key(r["index"], meas, r["half_year"], r["weekday"]), estimate=r["mean"],
                    statistic_name="mean week-relative log value", p_method="HAC (Newey-West), two-sided",
                    alt_p=alt(wilcoxon_p=r["wilcoxon_p"]))
    rel = "event_study/did_day_regressions.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            meas = clean(r["measure"])
            ctx.add("EVENT", f"{r['spec_id']}: difference-in-differences in {MEASURES.get(meas, meas)} (day-level regression)",
                    meas, f"{r['window']} ({clean(r['n_days'])} days, {clean(r['n_weeks'])} weeks)", r["p"],
                    role="robustness", pre_specified=PS_NO, source_file=rel, source_key=key(r["spec_id"], meas, r["window"]),
                    estimate=r["did"], statistic_name="DiD coefficient", p_method="OLS, week-clustered, two-sided")
    rel = "event_study/robustness_event_c.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        alts = {"main", "bootstrap blocks of 2 weeks", "bootstrap blocks of 8 weeks", "HAC with 8 lags"}
        for _, r in d[~d["variant"].isin(alts)].iterrows():
            meas = clean(r["measure"])
            ctx.add("EVENT", f"Event c, {r['series']}: change in {MEASURES.get(meas, meas)}; variant: {r['variant']}",
                    f"week-relative {meas}", f"{r['window']} ({clean(r['n_pre'])} vs {clean(r['n_post'])} weeks)", r["hac_p"],
                    role="robustness", pre_specified=PS_NO, source_file=rel,
                    source_key=key(r["variant"], r["series"], meas, r["window"]), estimate=r["diff"],
                    statistic_name="change (log points)", p_method="HAC (Newey-West), two-sided",
                    alt_p=alt(boot_p=r["boot_p"], welch_p=r["welch_p"], mwu_p=r["mwu_p"]))
        n_main = int((d["variant"] == "main").sum())
        n_alt = int(d["variant"].isin(alts - {"main"}).sum())
        ctx.skip("EVENT", rel, "'main' variant rows", n_main, "identical to event_tests.csv rows (counted once)")
        ctx.skip("EVENT", rel, "bootstrap-block and 8-lag HAC variants", n_alt,
                 "alternative intervals or standard errors for the same estimate (the bootstrap variants keep the same hac_p)")
    rel = "event_study/event_b_paper_window_raw_vs_relative.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        n_dup = 0
        for _, r in d.iterrows():
            meas = clean(r["measure"])
            samp = f"{r['index']} {r['weekday']}, {r['window']}"
            ctx.add("EVENT", f"Event b, paper's window: raw level of {MEASURES.get(meas, meas)} on {r['index']} {r['weekday']} changed",
                    f"log {meas} (raw level)", samp, r["raw_day_mwu_p"], role="exploratory", pre_specified=PS_NO,
                    source_file=rel, source_key=key("raw", r["window"], r["index"], meas, r["weekday"]),
                    estimate=r["raw_diff"], statistic_name="change in mean log level", p_method="day-level Mann-Whitney, two-sided",
                    alt_p=alt(raw_day_hac_p=r["raw_day_hac_p"]))
            if np.isfinite(num(r["rel_hac_p"])):
                if round(num(r["rel_hac_p"]), 12) in paper_hac:
                    n_dup += 1
                    continue
                ctx.add("EVENT", f"Event b, paper's window: week-relative {MEASURES.get(meas, meas)} on {r['index']} {r['weekday']} changed",
                        f"week-relative {meas}", samp, r["rel_hac_p"], role="exploratory", pre_specified=PS_NO,
                        source_file=rel, source_key=key("rel", r["window"], r["index"], meas, r["weekday"]),
                        estimate=r["rel_diff"], statistic_name="change (log points)", p_method="HAC, two-sided",
                        alt_p=alt(rel_mwu_p=r["rel_mwu_p"]))
        ctx.skip("EVENT", rel, "week-relative rows identical to event_tests paper-window rows", n_dup, "same hac_p (counted once)")
    for f, why in [("negative_control_calibration.csv", "shares of rejections across tests, not a test"),
                   ("placebo_c_did_main_6m.csv", "placebo draws for the primary test (summarised by its ri_p)")]:
        if ctx.has(f"event_study/{f}"):
            ctx.consumed.add(f"event_study/{f}")
            ctx.skip("EVENT", f"event_study/{f}", f, 1, why)


FALS_FINDING = {
    "F1": ("Previous evening's GEX predicts next-day realized variance (F1)", "next-day log realized variance"),
    "F2": ("The end-of-day reversal is stronger after negative GEX (F2)", "last-30-minute return"),
    "roll-off": ("Expiring gamma (roll-off) changes next-day volatility", "next-day log realized variance"),
}


def _fals_role(test: str, detail: str):
    t, dl = test, detail.lower()
    if t == "T1 permutation":
        return "robustness"
    if t == "T2 timing":
        return "placebo" if ("(lead)" in dl or dl.startswith("horse race lead")) else "robustness"
    if t.startswith("T3"):
        return "robustness"
    if t == "T4 split":
        return "robustness"
    if t == "T4 forecast":
        return "secondary"
    if t == "T4 pre-registered F2":
        return "primary_if_pro"
    if t == "T5 Bank Nifty":
        return "secondary"
    if t == "T6 roll-off":
        if dl.startswith("pseudo"):
            return "placebo"
        return "secondary" if dl.endswith("main") else "robustness"
    if t == "T6 real minus placebo":
        return "secondary"
    if t == "T6 roll-off IV":
        return "secondary" if dl == "main" else "robustness"
    return "exploratory"


def adapt_fals(ctx: Ctx):
    """Falsification. Counted: t7_family_B_all_tests_this_study.csv (742 tests: test, sample, finding,
    rule, detail, statistic, p, p_holm, p_bh), the study's own list of every test it ran. The per-test
    files t1 to t6 are its sources and are not counted again.
    T4 forecast p-values are one-sided ("GEX helps") without a prior statement: doubled.
    T4 pre-registered F2 is one-sided as stated by the study ("pre-registered style", designed after the
    full-sample results). Family A (78 rows) re-runs R3 and R5 with older code: used for cross-checks only.
    """
    rel = "falsification/t7_family_B_all_tests_this_study.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        rec_h = ctx.check("FALS", "family B (742 tests)", rel, "holm", d["p"], d["p_holm"])
        ctx.check("FALS", "family B (742 tests)", rel, "bh", d["p"], d["p_bh"])
        for (idx, r), rh in zip(d.iterrows(), rec_h):
            test, det, fnd = clean(r["test"]), clean(r["detail"]), clean(r["finding"])
            role = _fals_role(test, det)
            rule_raw = clean(r["rule"])
            if role == "primary_if_pro":
                role = "primary" if rule_raw.startswith("participant_pro") else "secondary"
            side = SIDE_TWO
            if test == "T4 forecast":
                side = SIDE_CONVERTED
            elif test == "T4 pre-registered F2":
                side = SIDE_ONE_STATED
            hyp0, outc = FALS_FINDING.get(fnd, (fnd, ""))
            rbase = rule_raw.split(" (")[0]
            ps_note = ("'pre-registered style': direction fixed on 2023 to 2024 data but designed after the full-sample results"
                       if test == "T4 pre-registered F2" else "falsification tests designed after the in-sample results")
            row = ctx.add("FALS", f"{test}: {hyp0}; {det}", outc, clean(r["sample"]), r["p"], role=role,
                          pre_specified=PS_NO, pre_spec_note=ps_note, source_file=rel,
                          source_key=key(test, r["sample"], fnd, rule_raw, det), sidedness=side,
                          sign_rule=RULES.get(rbase, rule_raw), statistic_name="statistic", statistic=r["statistic"],
                          p_method=test, p_study_adjusted=r["p_holm"], study_adjust_method="Holm over family B (742)")
            if row is not None:
                row["p_study_adjusted_recomputed"] = rh
                if side == SIDE_CONVERTED:
                    row["note"] = "study's p_holm/p_bh use the one-sided p"
    # Family A: cross-check only.
    rel = "falsification/t7_family_A_original_tests.csv"
    if ctx.has(rel):
        a = ctx.read(rel)
        ctx.check("FALS", "family A (78 original GEX-term tests)", rel, "holm", a["p"], a["p_holm"],
                  note="family A re-runs R3 and R5 with older code (n 932 for F2 against 929 now) and lists 6 pairs twice")
        ctx.check("FALS", "family A (78 original GEX-term tests)", rel, "bh", a["p"], a["p_bh"])
        dup = a.assign(_p=a["p"].round(10)).duplicated(subset=["finding", "rule", "term", "_p"], keep="first").sum()
        ctx.skip("FALS", rel, "family A (78 rows)", len(a),
                 f"re-estimates of R3 and R5 tests already counted under R3 and R5 (older code; {int(dup)} rows repeat another row's p)")
    rel = "falsification/t7_headline_adjusted.csv"
    if ctx.has(rel):
        h = ctx.read(rel)
        ctx.check("FALS", "18 headline tests (Holm within headline)", rel, "holm", h["p"], h["p_holm_within_headline"])
        ctx.check("FALS", "18 headline tests (BH within headline)", rel, "bh", h["p"], h["p_bh_within_headline"])
        ctx.skip("FALS", rel, "18 headline tests", len(h), "the same tests as R3 and R5 headline rows (counted there); used for the cross-check")
    rel = "falsification/t7_headline_permutation_adjusted.csv"
    if ctx.has(rel):
        h = ctx.read(rel)
        col = "p_two_sided_t"
        ctx.check("FALS", "12 headline permutation tests (Holm)", rel, "holm", h[col], h["p_holm_perm"], note=f"raw p column {col}")
        ctx.check("FALS", "12 headline permutation tests (BH)", rel, "bh", h[col], h["p_bh_perm"], note=f"raw p column {col}")
        ctx.skip("FALS", rel, "12 headline permutation p-values", len(h), "also in family B as T1 permutation rows (counted there)")
    if ctx.has("falsification/headline_in_sample.csv"):
        ctx.consumed.add("falsification/headline_in_sample.csv")
        ctx.skip("FALS", "falsification/headline_in_sample.csv", "headline re-estimates (4 regimes)", 36,
                 "near-duplicates of R3 and R5 headline tests; not separate hypotheses")
    for f in sorted(ctx.res("falsification").glob("*.csv*")):
        rel = f"falsification/{f.name}"
        if rel not in ctx.consumed:
            ctx.consumed.add(rel)
            if f.name.startswith(("t1_", "t2_", "t3_", "t4_", "t5_", "t6_")):
                ctx.skip("FALS", rel, f.name, 1, "source table of family B; its tests are counted once there (draw files are placebo draws, not tests)")
            else:
                ctx.skip("FALS", rel, f.name, 1, "data, chain recomputation or replication check; no hypothesis test")


def adapt_dsign(ctx: Ctx):
    """Dealer sign. Files and columns:
    direction_estimates.csv: wald_p (objective a or b; joint rows have no Wald p). Shares rows whose p equals
      the fixed US rule in fixed_rule_fits.csv are the same test; rows identical to the base fit are dropped.
    fixed_rule_fits.csv: p (theory-signed GEX term) and other_gex_p (the other GEX term), two-sided.
    oos_dm.csv: p_two_sided (p_est_better is its one-sided version); identical_forecasts rows have no test.
    oos_summary.csv: p_better_than_none (one-sided, not pre-stated: doubled).
    robustness_oos.csv: p_two_sided. kalman_params.csv: p_q_eq_0 (LR test of no drift).
    placebo_summary.csv: placebo_p of statistics; used as alternatives of the matching tests, and as rows
      only where no other p exists (joint LR for shares, Kalman share-of-days statistics).
    """
    pre_note = "primary set named in DEALER_SIGN.md; " + COMMITTED_WITH_RESULTS
    plc = {}
    rel = "dealer_sign/placebo_summary.csv"
    if ctx.has(rel):
        p = ctx.read(rel)
        plc = {clean(r["statistic"]): r for _, r in p.iterrows()}
    fixed_us = {}
    rel_f = "dealer_sign/fixed_rule_fits.csv"
    fx = ctx.read(rel_f) if ctx.has(rel_f) else pd.DataFrame()
    for _, r in fx.iterrows():
        if r["rule"] == "us":
            fixed_us[(clean(r["sample"]), clean(r["controls"]), clean(r["eq"]))] = round(num(r["p"]), 10)
    rel = "dealer_sign/direction_estimates.csv"
    us_dup = set()
    if ctx.has(rel):
        d = ctx.read(rel)
        seen = {}
        n_dup = 0
        for _, r in d[d["wald_p"].notna()].iterrows():
            fam, obj, samp, ctl = clean(r["family"]), clean(r["objective"]), clean(r["sample"]), clean(r["controls"])
            k = (fam, obj, round(num(r["wald_p"]), 10), num(r["angle_deg"]))
            if k in seen:
                n_dup += 1
                continue
            seen[k] = True
            if fam == "shares" and fixed_us.get((samp, ctl, obj)) == round(num(r["wald_p"]), 10):
                us_dup.add((samp, ctl, obj))
            a_or_b = ("expiry-day volatility outcome (a)" if obj == "a" else "end-of-day reversal outcome (b)")
            extra = ""
            if samp == "full sample" and ctl == "base":
                lr = plc.get(f"lr_{obj}_{fam}")
                tn = plc.get(f"theta_norm_{obj}_{fam}")
                extra = alt(placebo_lr_p=lr["placebo_p"] if lr is not None else float("nan"),
                            placebo_theta_norm_p=tn["placebo_p"] if tn is not None else float("nan"))
            ctx.add("DSIGN", f"An estimated dealer-sign direction ({fam}) explains the {a_or_b}",
                    "log realized variance (a) or last-30-minute return (b)", f"{samp}; controls {ctl}", r["wald_p"],
                    role="secondary" if (samp == "full sample" and ctl == "base") else "robustness",
                    pre_specified=PS_NO, source_file=rel, source_key=key(samp, ctl, fam, obj),
                    estimate=r["angle_deg"], statistic_name=f"Wald chi-square ({int(num(r['wald_df']))} df)",
                    statistic=r["wald"], sidedness=SIDE_NONDIR, p_method="Wald", alt_p=extra,
                    note="shares direction test equals the fixed US-rule test in fixed_rule_fits.csv" if (samp, ctl, obj) in us_dup else "")
        ctx.skip("DSIGN", rel, "rows identical to the base fit (same p and angle)", n_dup, "same test (counted once)")
        ctx.skip("DSIGN", rel, "joint rows", int(d["wald_p"].isna().sum()), "no Wald p (joint direction is summarised by the placebo LR test)")
    if len(fx):
        n_dup = 0
        for _, r in fx.iterrows():
            k3 = (clean(r["sample"]), clean(r["controls"]), clean(r["eq"]))
            rule = RULES.get(clean(r["rule"]), clean(r["rule"]))
            outc = "log realized variance" if r["eq"] == "a" else "last-30-minute return"
            if r["rule"] == "us" and k3 in us_dup:
                n_dup += 1
            else:
                ctx.add("DSIGN", f"Fixed rule: the theory-signed term ({r['signed_term']}) is not zero (outcome {r['eq']})",
                        outc, f"{r['sample']}; controls {r['controls']} (n = {clean(r['n'])})", r["p"], role="secondary",
                        pre_specified=PS_NO, source_file=rel_f, source_key=key(*k3, r["rule"], "signed"), sign_rule=rule,
                        estimate=r["coef"], statistic_name="coefficient", p_method="OLS with HAC standard errors, two-sided")
            ctx.add("DSIGN", f"Fixed rule: the other GEX term ({r['other_gex_term']}) is not zero (outcome {r['eq']})",
                    outc, f"{r['sample']}; controls {r['controls']} (n = {clean(r['n'])})", r["other_gex_p"], role="exploratory",
                    pre_specified=PS_NO, source_file=rel_f, source_key=key(*k3, r["rule"], "other"), sign_rule=rule,
                    estimate=r["other_gex_coef"], statistic_name="coefficient", p_method="OLS with HAC standard errors, two-sided")
        ctx.skip("DSIGN", rel_f, "US-rule signed-term rows", n_dup, "same test as the shares direction test (counted once there)")
    rel = "dealer_sign/oos_dm.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        prim = (d["primary"] == True) & (d["identical_forecasts"] == False)  # noqa: E712
        rec = ctx.check("DSIGN", "10 primary DM tests (Holm, two-sided)", rel, "holm", d.loc[prim, "p_two_sided"], d.loc[prim, "p_two_sided_holm_primary"])
        ctx.check("DSIGN", "10 primary DM tests (Holm, one-sided)", rel, "holm", d.loc[prim, "p_est_better"], d.loc[prim, "p_est_better_holm_primary"])
        rmap = dict(zip(d.index[prim], rec))
        for idx, r in d[d["identical_forecasts"] == False].iterrows():  # noqa: E712
            pr = bool(prim[idx])
            row = ctx.add("DSIGN", f"Out of sample, {r['estimated']} and the fixed {RULES.get(clean(r['fixed_rule']), r['fixed_rule'])} rule forecast equally well (outcome {r['eq']}, {r['variant']} template)",
                          "squared forecast error", f"{r['sample']} ({clean(r['n_days'])} days)", r["p_two_sided"],
                          role="primary" if pr else "exploratory", pre_specified=PS_STATED if pr else PS_NO,
                          pre_spec_note=pre_note if pr else "", source_file=rel,
                          source_key=key(r["sample"], r["eq"], r["variant"], r["estimated"], r["fixed_rule"]),
                          estimate=r["pct_est_vs_rule"], statistic_name="Diebold-Mariano t", statistic=r["dm_stat"],
                          p_method="Diebold-Mariano, two-sided", alt_p=alt(p_est_better_one_sided=r["p_est_better"]),
                          p_study_adjusted=r["p_two_sided_holm_primary"] if pr else float("nan"),
                          study_adjust_method="Holm over the 10 primary DM tests" if pr else "")
            if row is not None and pr:
                row["p_study_adjusted_recomputed"] = rmap[idx]
        ctx.skip("DSIGN", rel, "identical-forecast rows", int((d["identical_forecasts"] == True).sum()),  # noqa: E712
                 "estimated and fixed forecasts are identical: no test")
    rel = "dealer_sign/oos_summary.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        seen = set()
        n_dup = 0
        for _, r in d[d["p_better_than_none"].notna()].iterrows():
            k = (clean(r["sample"]), clean(r["eq"]), round(num(r["p_better_than_none"]), 10), round(num(r["pct_vs_none"]), 10))
            if k in seen:
                n_dup += 1
                continue
            seen.add(k)
            model = clean(r["model"])
            extra = ""
            st = plc.get(f"oos_pct_{r['eq']}_{model}_vs_none")
            if st is not None and r["sample"] == "all test days" and r["variant"] == "signed" and abs(num(st["real"]) - num(r["pct_vs_none"])) < 1e-6:
                extra = alt(placebo_p_one_sided=st["placebo_p"],
                            placebo_p_two_sided_equiv=two_sided_from_one_sided(num(st["placebo_p"])))
            ctx.add("DSIGN", f"Out of sample, the {model} GEX model beats the model without GEX (outcome {r['eq']}, {r['variant']} template)",
                    "squared forecast error", f"{r['sample']} ({clean(r['n_days'])} days)", r["p_better_than_none"],
                    role="exploratory", pre_specified=PS_NO, source_file=rel,
                    source_key=key(r["sample"], r["eq"], r["variant"], model), sidedness=SIDE_CONVERTED,
                    estimate=r["pct_vs_none"], statistic_name="Diebold-Mariano t", statistic=r["dm_stat_vs_none"],
                    p_method="Diebold-Mariano", alt_p=extra)
        ctx.skip("DSIGN", rel, "rows with identical forecasts (same p and gain)", n_dup, "same test (counted once)")
    rel = "dealer_sign/robustness_oos.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        keep = d[d["p_two_sided"].notna() & (d["identical_forecasts"] != True)]  # noqa: E712
        for _, r in keep.iterrows():
            ctx.add("DSIGN", f"Out of sample ({r['robustness']}): {r['comparison']} forecast equally well (outcome {r['eq']})",
                    "squared forecast error", f"{r['robustness']} ({clean(r['n_days'])} days)", r["p_two_sided"],
                    role="robustness", pre_specified=PS_NO, source_file=rel,
                    source_key=key(r["robustness"], r["comparison"], r["eq"]), estimate=r["pct"],
                    statistic_name="percent difference in loss", p_method="Diebold-Mariano, two-sided",
                    alt_p=alt(p_est_better_one_sided=r["p_est_better"]))
        ctx.skip("DSIGN", rel, "identical-forecast and 'last refit direction' rows", int(len(d) - len(keep)),
                 "no test (identical forecasts) or a direction estimate without a p-value")
    rel = "dealer_sign/kalman_params.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            st = None
            for k_, v in plc.items():
                if k_.startswith(f"kalman_drift_test_p_{r['eq']}_{r['family']}"):
                    st = v
            ctx.add("DSIGN", f"The Kalman dealer-weight path drifts over time ({r['family']}, outcome {r['eq']})",
                    "dealer weights (state)", "training 2023 to 2024", r["p_q_eq_0"], role="exploratory",
                    pre_specified=PS_NO, source_file=rel, source_key=key(r["family"], r["eq"]), estimate=r["q"],
                    statistic_name="LR statistic", statistic=r["lr_q_vs_0"], sidedness=SIDE_NONDIR,
                    p_method="likelihood ratio (q = 0 on the boundary)",
                    alt_p=alt(placebo_p=st["placebo_p"] if st is not None else float("nan")),
                    note="placebo fits reject q = 0 far too often for outcome (b) (DEALER_SIGN.md)" if r["eq"] == "b" else "")
    if plc:
        rel = "dealer_sign/placebo_summary.csv"
        used = 0
        for name, r in plc.items():
            if not np.isfinite(num(r["placebo_p"])):
                continue
            if name == "lr_joint_shares" or name.startswith("kalman_share_test_days"):
                ctx.add("DSIGN", f"Placebo-calibrated test: {name} is larger than under shuffled outcomes",
                        name, f"{int(num(r['n_perm']))} shuffles", r["placebo_p"], role="exploratory", pre_specified=PS_NO,
                        source_file=rel, source_key=key(name), estimate=r["real"], statistic_name=name,
                        p_method="placebo (shuffled outcome), upper tail")
                used += 1
        ctx.skip("DSIGN", rel, "placebo p-values attached to other rows or duplicated",
                 int(sum(np.isfinite(num(r["placebo_p"])) for r in plc.values())) - used,
                 "alternative p-values of counted tests (alt_p), or lr_joint_participants = lr_b_participants (same statistic and p)")
    for f in ("kalman_states.csv", "kalman_summary.csv", "lr_profiles.csv"):
        if ctx.has(f"dealer_sign/{f}"):
            ctx.consumed.add(f"dealer_sign/{f}")
            ctx.skip("DSIGN", f"dealer_sign/{f}", f, 1, "p_us_side columns are shares of probability mass, not p-values" if "kalman" in f else "likelihood profile, no test")
    for f in sorted(ctx.res("dealer_sign").glob("synthetic*.csv")):
        ctx.consumed.add(f"dealer_sign/{f.name}")
        ctx.skip("DSIGN", f"dealer_sign/{f.name}", f.name, 1, "synthetic-data check, not a test of real data")


INTRA_ROLE = {"headline": "headline", "secondary": "secondary", "exploratory": "exploratory",
              "robustness": "robustness", "placebo": "placebo", "followup_primary": "primary",
              "followup_secondary": "secondary", "followup_robustness": "robustness", "followup_placebo": "placebo"}


def adapt_intra(ctx: Ctx):
    """Intraday gamma. File: results/intraday_gamma/all_tests.csv (test_id, family, test, coef, p
    (two-sided), p_holm_headline, p_holm_within_family), the study's own list of tests. Other CSVs in the
    folder are its sources. Rows 120 and 121 repeat FH1 (118) and row 123 repeats 122: dropped.
    followup_summary.csv: p_holm_over_FH1_FH2_BN1 (cross-check).
    """
    rel = "intraday_gamma/all_tests.csv"
    if not ctx.has(rel):
        return
    d = ctx.read(rel)
    hd = d["family"] == "headline"
    rec_hd = ctx.check("INTRA", "7 headline tests (Holm)", rel, "holm", d.loc[hd, "p"], d.loc[hd, "p_holm_headline"])
    d["_rec_hd"] = np.nan
    d.loc[hd, "_rec_hd"] = rec_hd
    for fam in d["family"].unique():
        m = d["family"] == fam
        ctx.check("INTRA", f"family '{fam}' (Holm within family)", rel, "holm", d.loc[m, "p"], d.loc[m, "p_holm_within_family"])
    pre_note = "pre-registration stamped 21:36 UTC inside design_log.csv; " + COMMITTED_WITH_RESULTS + " (21:53 UTC)"
    rows = []
    for _, r in d.iterrows():
        fam = clean(r["family"])
        tid = int(num(r["test_id"]))
        test = clean(r["test"])
        if fam == "headline":
            ps, psn = PS_STATED, "headline tests stated before the first run, per INTRADAY_GAMMA.md; " + COMMITTED_WITH_RESULTS
        elif fam.startswith("followup") and fam != "followup_robustness" and "horse race" not in test:
            ps, psn = PS_STATED, pre_note
        else:
            ps, psn = PS_NO, ("added after the first run (design_log.csv)" if fam in ("placebo", "robustness") else "")
        n0 = len(ctx.rows)
        samp = ("Bank Nifty expiry days" if "Bank Nifty" in test or test.startswith("BN") else
                "Nifty, final half hour of expiry days" if ("final half hour" in test or test.startswith("FH")) else
                "Nifty intraday 15-minute panel (day type in the hypothesis)")
        row = ctx.add("INTRA", test, "realized variance over the next 5 or 15 minutes (log)", f"{samp}; family {fam}", r["p"],
                      role=INTRA_ROLE.get(fam, "unlabelled"), pre_specified=ps, pre_spec_note=psn, source_file=rel,
                      source_key=key(tid, fam), estimate=r["coef"], statistic_name="coefficient",
                      p_method=f"{clean(r['source'])}, two-sided",
                      p_study_adjusted=r["p_holm_headline"] if fam == "headline" else float("nan"),
                      study_adjust_method="Holm over the 7 headline tests" if fam == "headline" else "")
        if row is not None:
            if fam == "headline":
                row["p_study_adjusted_recomputed"] = r["_rec_hd"]
            rows.append(ctx.rows.pop())
    kept, dropped = dedup_exact(rows, cols=("p_reported", "estimate"))
    ctx.rows.extend(kept)
    ctx.skip("INTRA", rel, "rows repeating another row (same coefficient and p)", len(dropped),
             "e.g. test 120 and 121 equal FH1 (118); 123 equals 122: counted once")
    fs = "intraday_gamma/followup_summary.csv"
    if ctx.has(fs):
        f = ctx.read(fs)
        ctx.check("INTRA", "follow-up primaries FH1, FH2, BN1 (Holm)", fs, "holm", f["p_two_sided"], f["p_holm_over_FH1_FH2_BN1"])
    for f in sorted(ctx.res("intraday_gamma").glob("*.csv")):
        rel2 = f"intraday_gamma/{f.name}"
        if rel2 not in ctx.consumed:
            ctx.consumed.add(rel2)
            ctx.skip("INTRA", rel2, f.name, 1, "source or descriptive table; its tests are counted once in all_tests.csv")


def adapt_conn(ctx: Ctx):
    """Connectedness. tests.csv: test (P0, S1 to S13), family, contrast, estimate, perm_p (main,
    two-sided, pre-stated), hac_p (S7 only, alternative), holm_p_secondary (Holm over S1 to S13).
    robustness.csv (perm_p), posthoc_checks.csv (perm_p), day_level_regression.csv (row without the VIX
    control; the row with it repeats S7).
    """
    pre_note = "stated before the first run per CONNECTEDNESS_GAMMA.md, which says it was not committed to git first"
    rel = "connectedness_gamma/tests.csv"
    s7 = None
    if ctx.has(rel):
        d = ctx.read(rel)
        sec = d["family"] == "secondary"
        rec = ctx.check("CONN", "S1 to S13 (Holm)", rel, "holm", d.loc[sec, "perm_p"], d.loc[sec, "holm_p_secondary"])
        rmap = dict(zip(d.index[sec], rec))
        for idx, r in d.iterrows():
            if r["test"] == "S7":
                s7 = (round(num(r["perm_p"]), 10), round(num(r["hac_p"]), 10))
            row = ctx.add("CONN", f"{r['test']}: {clean(r['contrast'])}", f"connectedness ({r['measure']})",
                          f"{clean(r['n_days_state_a'])} vs {clean(r['n_days_state_b'])} days", r["perm_p"],
                          role="primary" if r["family"] == "primary" else "secondary", pre_specified=PS_STATED,
                          pre_spec_note=pre_note, source_file=rel, source_key=key(r["test"]),
                          sign_rule="NSE Pro participant" if "Pro" in clean(r["contrast"]) else "",
                          estimate=r["estimate"], statistic_name="difference", p_method="block permutation, two-sided",
                          alt_p=alt(hac_p=r["hac_p"]),
                          p_study_adjusted=r["holm_p_secondary"] if sec[idx] else float("nan"),
                          study_adjust_method="Holm over S1 to S13" if sec[idx] else "")
            if row is not None and sec[idx]:
                row["p_study_adjusted_recomputed"] = rmap[idx]
    rel = "connectedness_gamma/robustness.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["perm_p"].notna()].iterrows():
            ctx.add("CONN", f"Robustness: {clean(r['check'])}", "total connectedness", "", r["perm_p"], role="robustness",
                    pre_specified=PS_STATED, pre_spec_note=pre_note, source_file=rel, source_key=key(r["check"]),
                    estimate=r["estimate"], statistic_name="difference", p_method="block permutation, two-sided",
                    note="block-length rows re-test P0's statistic with another block length" if "block length" in clean(r["check"]) else "")
        ctx.skip("CONN", rel, "rows without perm_p", int(d["perm_p"].isna().sum()), "count of days, no test")
    rel = "connectedness_gamma/posthoc_checks.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["perm_p"].notna()].iterrows():
            ctx.add("CONN", f"Post hoc: {clean(r['check'])}", "total connectedness",
                    f"{clean(r['n_days_state_a'])} vs {clean(r['n_days_state_b'])} days", r["perm_p"], role="post hoc",
                    pre_specified=PS_NO, pre_spec_note="added after the first full run", source_file=rel,
                    source_key=key(r["check"]), estimate=r["estimate"], statistic_name="difference",
                    p_method="block permutation, two-sided")
    rel = "connectedness_gamma/day_level_regression.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        n_dup = 0
        for _, r in d.iterrows():
            k = (round(num(r["perm_p"]), 10), round(num(r["hac_p"]), 10))
            if s7 is not None and (k[0] == s7[0] or k[1] == s7[1]):
                n_dup += 1
                continue
            ctx.add("CONN", f"Per-day connectedness is higher on short-gamma (Pro) days ({clean(r['spec'])})", clean(r["outcome"]),
                    f"{clean(r['n_days'])} days", r["hac_p"], role="robustness", pre_specified=PS_NO,
                    pre_spec_note="not in the study's own count of 23 tests", source_file=rel, source_key=key(r["spec"]),
                    sign_rule="NSE Pro participant", estimate=r["coef_short_gamma_pro"], statistic_name="coefficient",
                    p_method="OLS with HAC standard errors, two-sided", alt_p=alt(perm_p=r["perm_p"]))
        ctx.skip("CONN", rel, "row repeating S7", n_dup, "same test as S7 (counted once)")
        ctx.skip("CONN", rel, "hac_p_log_vix_prev", int(d["hac_p_log_vix_prev"].notna().sum()), "control coefficient")


SENSEX_ROLE = {"main": "primary", "calendar check": "secondary", "(i) level": "secondary",
               "(ii) change": "secondary", "(iii) re-estimate": "robustness"}


def adapt_sensex(ctx: Ctx):
    """Sensex confound. File: results/sensex_confound/all_tests.csv (family, test, estimate, p (main,
    two-sided), mwu_p, placebo_p, placebo_p_all_events_blocked, wilcoxon_p (alternatives),
    pretrend_p (diagnostic), bh_q_within_family). The other CSVs in the folder are its sources.
    """
    rel = "sensex_confound/all_tests.csv"
    if not ctx.has(rel):
        return
    d = ctx.read(rel)
    for fam in d["family"].unique():
        m = d["family"] == fam
        rec = ctx.check("SENSEX", f"family '{fam}' (BH within family)", rel, "bh", d.loc[m, "p"], d.loc[m, "bh_q_within_family"])
        d.loc[m, "_rec"] = rec
    for _, r in d.iterrows():
        fam = clean(r["family"])
        main = fam == "main"
        row = ctx.add("SENSEX", clean(r["test"]), "week-relative or level realized variance", f"n = {clean(r['n'])}", r["p"],
                role=SENSEX_ROLE.get(fam, "unlabelled"), pre_specified=PS_STATED if main else PS_NO,
                pre_spec_note=("stated in the script docstring before the run; the idea came from a post hoc table; "
                               + COMMITTED_WITH_RESULTS) if main else "",
                source_file=rel, source_key=key(fam, r["test"]), estimate=r["estimate"], statistic_name="estimate",
                p_method="HAC or regression, two-sided",
                alt_p=alt(mwu_p=r["mwu_p"], placebo_p=r["placebo_p"], placebo_p_all_events_blocked=r["placebo_p_all_events_blocked"],
                          wilcoxon_p=r["wilcoxon_p"]),
                p_study_adjusted=r["bh_q_within_family"], study_adjust_method="BH within family")
        _set(row, "p_study_adjusted_recomputed", r["_rec"])
    ctx.skip("SENSEX", rel, "pre-trend p-values", int(d["pretrend_p"].notna().sum()), "diagnostic, not a hypothesis")
    for f in sorted(ctx.res("sensex_confound").glob("*.csv")):
        rel2 = f"sensex_confound/{f.name}"
        if rel2 not in ctx.consumed:
            ctx.consumed.add(rel2)
            ctx.skip("SENSEX", rel2, f.name, 1, "source table of all_tests.csv (tests counted once there) or calendar data")


# ----------------------------------------------------------------------------
# Folders that may appear later
# ----------------------------------------------------------------------------


def git_dirs(root: Path):
    """The repository directories whose reflogs hold this checkout's commits: ``.git`` itself, or, when ``.git``
    is a file (a worktree), the worktree's own git directory and the main repository's (``commondir``)."""
    g = Path(root) / ".git"
    if g.is_dir():
        return [g]
    out = []
    if g.is_file():
        txt = g.read_text(errors="replace").strip()
        if txt.startswith("gitdir:"):
            gd = Path(txt.split(":", 1)[1].strip())
            gd = gd if gd.is_absolute() else (Path(root) / gd)
            out.append(gd)
            cd = gd / "commondir"
            if cd.is_file():
                c = Path(cd.read_text(errors="replace").strip())
                out.append(c if c.is_absolute() else (gd / c))
    return [d for d in out if d.is_dir()]


def reflog_commits(root: Path):
    """(unix time, message) of every reflog entry in logs/HEAD and logs/refs/heads/** of the repository (read
    only; no git command). A commit made in another worktree is in its branch's reflog, so both are read."""
    files = []
    for d in git_dirs(Path(root)):
        files.append(d / "logs" / "HEAD")
        heads = d / "logs" / "refs" / "heads"
        if heads.is_dir():
            files.extend(sorted(p for p in heads.rglob("*") if p.is_file()))
    out, seen = [], set()
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text(errors="replace").splitlines():
            try:
                head, msg = line.split("\t", 1)
                parts = head.split()
                item = (int(parts[-2]), msg)
            except (ValueError, IndexError):
                continue
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def prereg_status(ctx: Ctx, folder: str, outcome=None):
    """PS_YES when the pre-registration's commit (found by its message in the reflogs) is older than the earliest
    result file. ``outcome``: glob patterns (relative to the folder) of the files that hold outcomes; files written
    before the outcomes on purpose (phase-1 coverage counts committed with the plan) are then left out."""
    if folder not in PREREG:
        return PS_UNKNOWN, ""
    doc, phrase = PREREG[folder]
    if not (ctx.root / doc).exists():
        return PS_UNKNOWN, f"{doc} not found"
    commits = [ts for ts, msg in reflog_commits(ctx.root) if phrase in msg]
    if outcome:
        files = sorted({p for pat in outcome for p in ctx.res(folder).glob(pat) if p.is_file()})
    else:
        files = [p for p in ctx.res(folder).rglob("*") if p.is_file()]
    what = "outcome file" if outcome else "result file"
    if not commits:
        return PS_STATED, f"{doc} exists, but no commit '{phrase}' was found in the reflogs"
    t_c = min(commits)
    t_r = min(p.stat().st_mtime for p in files) if files else float("inf")
    tc = datetime.fromtimestamp(t_c, timezone.utc).strftime("%d %b %Y %H:%M UTC")
    if not files:
        return PS_YES, f"{doc} committed {tc} per the reflogs; no {what}s yet"
    if t_c < t_r:
        tr = datetime.fromtimestamp(t_r, timezone.utc).strftime("%H:%M UTC")
        return PS_YES, (f"{doc} committed {tc} per the reflogs, before the earliest {what} ({tr}); "
                        "confirm with git log")
    return PS_STATED, f"{doc} committed {tc}, but {what}s are older"


P_COL = re.compile(r"(^|_)(p|pvalue|p_value)(_|$)|^Pvalue$")
P_COL_EXCLUDE = re.compile(r"holm|(^|_)bh(_|$)|(^|_)by(_|$)|(^|_)q(_|$)|adj|printed|published|pretrend|share|rows_with", re.I)


def p_like_columns(d: pd.DataFrame):
    cols = []
    for c in d.columns:
        if P_COL.search(c) and not P_COL_EXCLUDE.search(c):
            v = pd.to_numeric(d[c], errors="coerce")
            if v.notna().any() and v.min() >= 0 and v.max() <= 1:
                cols.append(c)
    return cols


def adapt_generic(ctx: Ctx, study: str, folder: str):
    """Fallback for a study without a dedicated adapter.

    Uses all_tests.csv or tests.csv when present, otherwise every CSV with a p-like column. The main p is
    'p', 'p_value', 'p_two_sided' or 'perm_p' if present, else the first p-like column; other p-like
    columns become alternatives. Role comes from a 'role' or 'family' column when present. Rows with the
    same (p, estimate) are counted once. A warning asks for a dedicated adapter.
    """
    base = ctx.res(folder)
    csvs = sorted(p for p in base.rglob("*.csv"))
    pref = [p for p in csvs if p.name in ("all_tests.csv", "tests.csv")]
    use = pref if pref else csvs
    ps_status, ps_note = prereg_status(ctx, folder)
    rows = []
    for f in use:
        rel = str(f.relative_to(ctx.res(""))).replace("\\", "/")
        d = ctx.read(rel)
        pcs = p_like_columns(d)
        if not pcs:
            continue
        main = next((c for c in ("p", "p_value", "p_two_sided", "perm_p", "p_one_sided") if c in pcs), pcs[0])
        est = next((c for c in ("estimate", "coef", "diff", "mean", "effect") if c in d.columns), None)
        txt_cols = [c for c in d.columns if d[c].dtype == object or str(d[c].dtype).startswith(("str", "string"))][:5]
        role_col = next((c for c in ("role", "family", "status") if c in d.columns), None)
        for _, r in d.iterrows():
            role = "unlabelled"
            if role_col:
                lab = clean(r[role_col]).lower()
                for k in ("primary", "headline", "secondary", "robustness", "placebo", "exploratory"):
                    if k in lab:
                        role = k
                        break
                if "post" in lab and "hoc" in lab:
                    role = "post hoc"
            if role in ("primary", "headline", "secondary"):
                ps, psn = ps_status, ps_note
            elif role in ("exploratory", "post hoc"):
                ps, psn = PS_NO, ""
            else:
                ps, psn = PS_UNKNOWN, ps_note
            side = SIDE_ONE_PRE if ("one_sided" in main and ps == PS_YES) else (SIDE_CONVERTED if "one_sided" in main else SIDE_TWO)
            desc = "; ".join(f"{c}={clean(r[c])}" for c in txt_cols if clean(r[c]))
            n0 = len(ctx.rows)
            ctx.add(study, desc or f.name, "", "", r[main], role=role, pre_specified=ps, pre_spec_note=psn,
                    source_file=rel, source_key=key(f.name, desc), sidedness=side,
                    estimate=r[est] if est else float("nan"), statistic_name=est or "", p_method=f"column {main}",
                    alt_p=alt(**{c: r[c] for c in pcs if c != main}),
                    note="read by the generic adapter; check roles and duplicates with a dedicated adapter")
            if len(ctx.rows) > n0:
                rows.append(ctx.rows.pop())
    kept, dropped = dedup_exact(rows)
    ctx.rows.extend(kept)
    if kept or dropped:
        ctx.skip(study, folder + "/", "rows repeating another row (same p and estimate)", len(dropped), "counted once")
    ctx.warn(f"{folder}: read by the generic adapter ({len(kept)} rows); write a dedicated adapter once the study is documented")
    for f in csvs:
        rel = str(f.relative_to(ctx.res(""))).replace("\\", "/")
        ctx.consumed.add(rel)


def adapt_f2econ(ctx: Ctx):
    """P90 F2 economics (pre-registered in docs/prereg/P90_F2_ECONOMICS.md).

    metrics.csv: sample, rule (P primary; S1, S2 secondary; D sensitivity), costs, hac_t, p_one_sided
    (HAC t of the mean daily net return, one-sided "mean > 0", stated in the pre-registration).
    cost_grid.csv: rule, costs, p_one_sided over the cost grid; rows identical to metrics.csv are dropped.
    auction_days.csv: rows identical to the auction-day rows of metrics.csv are dropped.
    The decision statistic of the study is the deflated Sharpe ratio, not a p-value (dsr.csv: not counted).
    Falls back to the generic adapter if the files have another shape.
    """
    folder = "f2_economics"
    need = {"sample", "rule", "costs", "hac_t", "p_one_sided"}
    if not ctx.has(f"{folder}/metrics.csv") or not need.issubset(pd.read_csv(ctx.res(f"{folder}/metrics.csv"), nrows=0).columns):
        return adapt_generic(ctx, "F2ECON", folder)
    ps_status, ps_note = prereg_status(ctx, folder)
    role_of = {"P": "primary", "S1": "secondary", "S2": "secondary", "D": "robustness"}
    rule_desc = {"P": "rule P (trade against the day after negative Pro GEX)", "S1": "rule S1 (sized by the move)",
                 "S2": "rule S2 (trade only when the forecast beats costs)", "D": "sensitivity D (entry one bar later)"}
    seen = set()
    rows = []
    for fname in ("metrics.csv", "cost_grid.csv", "auction_days.csv"):
        rel = f"{folder}/{fname}"
        if not ctx.has(rel):
            continue
        d = ctx.read(rel)
        for _, r in d.iterrows():
            rule = clean(r["rule"])
            samp = clean(r["sample"]) if "sample" in d.columns else "test"
            costs = clean(r["costs"])
            k = (rule, round(num(r["p_one_sided"]), 12), round(num(r["hac_t"]), 12))
            if not np.isfinite(num(r["p_one_sided"])) or k in seen:
                continue
            seen.add(k)
            test_base = samp == "test" and costs == "base"
            if test_base:
                role, ps, psn = role_of.get(rule, "unlabelled"), ps_status, ps_note
            elif samp == "test" or fname == "cost_grid.csv":
                role, ps, psn = "robustness", PS_NO, "cost level; the pre-registration says cost levels are not trials"
            else:
                role, ps, psn = "exploratory", PS_NO, f"sample '{samp}'"
            side = SIDE_ONE_PRE if ps_status == PS_YES else SIDE_ONE_STATED
            ctx.add("F2ECON", f"{rule_desc.get(rule, rule)} earns a positive mean net return ({costs} costs)",
                    "daily net return (bp of notional)", f"{samp} sample", r["p_one_sided"], role=role,
                    pre_specified=ps, pre_spec_note=psn, source_file=rel, source_key=key(fname, samp, rule, costs),
                    sidedness=side, estimate=r.get("mean_net_per_day", float("nan")), statistic_name="HAC t",
                    statistic=r["hac_t"], p_method="HAC t of the mean, one-sided (mean > 0)")
            rows.append(ctx.rows.pop())
    ctx.rows.extend(rows)
    for f in sorted(ctx.res(folder).glob("*.csv")):
        rel = f"{folder}/{f.name}"
        if rel not in ctx.consumed:
            ctx.consumed.add(rel)
            ctx.skip("F2ECON", rel, f.name, 1, "deflated Sharpe ratio, bootstrap, costs, trades or daily P&L; no further p-value")


def adapt_pinmech(ctx: Ctx):
    """P87 pinning mechanism (pre-registered in docs/prereg/P87_PINNING_MECHANISM.md).

    primary_test.csv and secondary_tests.csv: candidate, description, role, gamma (expiry-day slope),
    gamma_p (two-sided, week-clustered; the pre-registered test), p_perm_within_cell (a check that decides
    nothing: alternative), p_holm (Holm over the four secondaries; cross-checked). d_p is the ordinary-day
    slope, a nuisance term here (it is tested as placebo P1b).
    placebos.csv: candidate, placebo, description, gamma_p (pre-registered placebo table).
    robustness.csv: candidate, spec, label (the study's own label), gamma_p.
    decisions.csv repeats these p-values, descriptive_c4_a1.csv is descriptive by design ("not counted"),
    and between_regime.csv has intervals only: none of them is counted.
    Falls back to the generic adapter if the files have another shape.
    """
    folder = "pinning_mechanism"
    need = {"candidate", "gamma", "gamma_p"}
    if not ctx.has(f"{folder}/primary_test.csv") or not need.issubset(pd.read_csv(ctx.res(f"{folder}/primary_test.csv"), nrows=0).columns):
        return adapt_generic(ctx, "PINMECH", folder)
    ps_status, ps_note = prereg_status(ctx, folder)
    outc = "distance from official close to nearest strike, in strike spacings"
    for fname, role in (("primary_test.csv", "primary"), ("secondary_tests.csv", "secondary")):
        rel = f"{folder}/{fname}"
        if not ctx.has(rel):
            continue
        d = ctx.read(rel)
        rec = None
        if role == "secondary" and "p_holm" in d.columns:
            rec = ctx.check("PINMECH", "secondary family C2, C3, C5, C6 (Holm)", rel, "holm", d["gamma_p"], d["p_holm"])
        for i, (_, r) in enumerate(d.iterrows()):
            row = ctx.add("PINMECH", f"On expiry days a higher {clean(r.get('description', r['candidate']))} goes with a close nearer a strike ({r['candidate']}, gamma)",
                          outc, f"all index-days ({_int(r.get('n_days'))}, {_int(r.get('n_expiry'))} expiry days)", r["gamma_p"],
                          role=role, pre_specified=ps_status, pre_spec_note=ps_note, source_file=rel,
                          source_key=key(fname, r["candidate"]), estimate=r["gamma"], statistic_name="gamma (expiry x candidate)",
                          p_method="OLS, week-clustered (CR1), t with weeks - 1 df, two-sided",
                          alt_p=alt(p_perm_within_cell=r.get("p_perm_within_cell", float("nan"))),
                          p_study_adjusted=r.get("p_holm", float("nan")) if role == "secondary" else float("nan"),
                          study_adjust_method="Holm over the four secondaries" if role == "secondary" else "")
            if rec is not None:
                _set(row, "p_study_adjusted_recomputed", rec[i])
    rel = f"{folder}/placebos.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["gamma_p"].notna()].iterrows():
            ctx.add("PINMECH", f"Placebo {r['placebo']} for {r['candidate']}: {clean(r['description'])}", outc,
                    f"{_int(r.get('n_days'))} days", r["gamma_p"], role="placebo", pre_specified=ps_status,
                    pre_spec_note=ps_note + "; placebo table in the pre-registration", source_file=rel,
                    source_key=key(r["candidate"], r["placebo"]), estimate=r["gamma"], statistic_name="gamma or slope",
                    p_method="OLS, week-clustered, two-sided",
                    alt_p=alt(p_perm_within_cell=r.get("p_perm_within_cell", float("nan")),
                              gamma_close_same_sample_p=r.get("gamma_close_same_sample_p", float("nan"))))
    rel = f"{folder}/robustness.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["gamma_p"].notna()].iterrows():
            lab = clean(r.get("label", "")).lower()
            role = next((k for k in ("primary", "secondary", "robustness", "placebo", "exploratory") if k in lab), "robustness")
            ctx.add("PINMECH", f"{r['candidate']}: {clean(r['spec'])}", outc, f"{_int(r.get('n_days'))} days", r["gamma_p"],
                    role=role, pre_specified=PS_NO, pre_spec_note=f"labelled '{lab}' by the study", source_file=rel,
                    source_key=key(r["candidate"], r["spec"]), estimate=r["gamma"], statistic_name="gamma (expiry x candidate)",
                    p_method="OLS, week-clustered, two-sided")
    for f, why in (("decisions.csv", "repeats the primary and Holm-adjusted secondary p-values (p_used)"),
                   ("descriptive_c4_a1.csv", "C4 and A1 are descriptive by design; the study marks the C4 p-value 'not counted'"),
                   ("between_regime.csv", "predicted and observed regime gaps with intervals; no p-value"),
                   ("candidates.csv", "candidate variables (data)"), ("candidate_checks.csv", "data checks"),
                   ("design_log.csv", "design log")):
        if ctx.has(f"{folder}/{f}"):
            ctx.consumed.add(f"{folder}/{f}")
            ctx.skip("PINMECH", f"{folder}/{f}", f, 1, why)
    for f in sorted(ctx.res(folder).glob("*.csv")):
        rel = f"{folder}/{f.name}"
        if rel not in ctx.consumed:
            ctx.warn(f"{rel}: not read by the PINMECH adapter; check whether it holds tests")


P103_OUTCOME = ["utility_tests.csv", "trading_tests.csv", "trading_metrics.csv", "verdict.csv", "dsr.csv"]


def adapt_p103econ(ctx: Ctx):
    """P103: is the hybrid's forecasting gain worth anything economically? (docs/prereg/P103_ECONOMIC_VALUE.md)

    utility_tests.csv: test_id (H1 primary; S1 to S4 secondary, Holm over the four in p_holm_secondary,
      cross-checked; R1 to R4 robustness, all registered), window_set, subset, model, benchmark, fee_pct_year
      (estimate: realized-utility fee of the model over the benchmark, percent of wealth a year), dm_stat,
      p_one_sided (repository DM test on day-level utility losses, one-sided as registered).
    trading_tests.csv: T1 (tracking) and T2 (net utility at base-case costs) are the registered trading tests
      (secondary); T1-iv, T2-iv, T2-rebal and T2-grid are registered robustness versions. One-sided DM p.
    trading_metrics.csv: HAC t of each strategy's mean daily net return at each cost level (one-sided, mean > 0),
      a registered metric, counted as robustness (P90 counted its cost grid the same way).
    Not counted: fees, utilities of every model, QLIKE on the same windows, daily series, DSR, bootstrap of the
      Sharpe ratio, verdict, checks and the descriptive files added in phase 2.
    """
    folder = "p103_economic_value"
    need = {"test_id", "role", "window_set", "subset", "model", "benchmark", "fee_pct_year", "dm_stat", "p_one_sided"}
    rel_u = f"{folder}/utility_tests.csv"
    if not ctx.has(rel_u):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "P103ECON", folder)
        return
    if not need.issubset(pd.read_csv(ctx.res(rel_u), nrows=0).columns):
        adapt_generic(ctx, "P103ECON", folder)
        return
    ps, psn = prereg_status(ctx, folder, P103_OUTCOME)
    side = SIDE_ONE_PRE if ps == PS_YES else SIDE_ONE_STATED
    tgt = {"next15": "next 15 minutes", "next30": "next 30 minutes", "last_hour": "last hour"}
    rows = []
    u = ctx.read(rel_u)
    for _, r in u.iterrows():
        tid = clean(r["test_id"])
        role = "primary" if tid == "H1" else "secondary" if tid.startswith("S") else "robustness"
        hyp = (f"P103 {tid}: {clean(r['model'])} gives a higher realized utility (Bollerslev, Hood, Huss and Pedersen) "
               f"than {clean(r['benchmark'])}, {tgt.get(clean(r['target']), clean(r['target']))}, windows "
               f"{clean(r['window_set'])}")
        ctx.add("P103ECON", hyp, "day-level realized utility (volatility-timing investor)", clean(r["subset"]),
                r["p_one_sided"], role=role, pre_specified=ps, pre_spec_note=psn, source_file=rel_u,
                source_key=key("utility_tests.csv", tid, r["window_set"], r["subset"], r["benchmark"]), sidedness=side,
                estimate=r["fee_pct_year"], statistic_name="DM (HLN), fee in percent of wealth a year as estimate",
                statistic=r["dm_stat"], p_method="Diebold-Mariano on daily utility losses, one-sided",
                p_study_adjusted=r["p_holm_secondary"] if role == "secondary" else float("nan"),
                study_adjust_method="Holm over S1 to S4" if role == "secondary" else "")
        rows.append(ctx.rows.pop())
    secm = u["test_id"].astype(str).str.startswith("S")
    if secm.any():
        rec = ctx.check("P103ECON", "S1 to S4 (Holm)", rel_u, "holm", u.loc[secm, "p_one_sided"].to_numpy(float),
                        u.loc[secm, "p_holm_secondary"].to_numpy(float))
        for r_, v in zip([r for r in rows if r["role"] == "secondary"], rec):
            r_["p_study_adjusted_recomputed"] = v
    rel_t = f"{folder}/trading_tests.csv"
    if ctx.has(rel_t):
        t = ctx.read(rel_t)
        for _, r in t.iterrows():
            tid = clean(r["test_id"])
            role = "secondary" if tid in ("T1", "T2") else "robustness"
            what = ("tracks its volatility target more closely (daily squared log deviation)" if r["kind"] == "tracking"
                    else f"has a higher daily net realized utility ({clean(r['costs'])})")
            ctx.add("P103ECON", f"P103 {tid}: the vol-targeted rule sized by {clean(r['model'])} {what} than the one sized "
                    f"by {clean(r['benchmark'])}", clean(r["kind"]), "all test days", r["p_one_sided"], role=role,
                    pre_specified=ps, pre_spec_note=psn, source_file=rel_t,
                    source_key=key("trading_tests.csv", tid, r["benchmark"], r["costs"]), sidedness=side,
                    estimate=r["estimate"], statistic_name="DM (HLN)", statistic=r["dm_stat"],
                    p_method="Diebold-Mariano on daily differences, one-sided")
            rows.append(ctx.rows.pop())
    rel_m = f"{folder}/trading_metrics.csv"
    if ctx.has(rel_m):
        m = ctx.read(rel_m)
        for _, r in m.iterrows():
            ctx.add("P103ECON", f"P103: the vol-targeted rule sized by {clean(r['strategy'])} earns a positive mean daily net "
                    f"return ({clean(r['costs'])})", "daily net return (bp of capital)", "all test days", r["p_one_sided"],
                    role="robustness", pre_specified=ps, pre_spec_note=psn, source_file=rel_m,
                    source_key=key("trading_metrics.csv", r["strategy"], r["costs"]), sidedness=side,
                    estimate=r["mean_net_bp_day"], statistic_name="HAC t", statistic=r["hac_t"],
                    p_method="HAC t of the mean (Newey-West, 5 lags), one-sided (mean > 0)")
            rows.append(ctx.rows.pop())
    kept, dropped = dedup_exact(rows)
    ctx.rows.extend(kept)
    ctx.skip("P103ECON", f"{folder}/", "rows repeating another row (same p and estimate)", len(dropped), "counted once")
    _consume_rest(ctx, "P103ECON", folder, {
        r"utility_models\.csv|fee_grid\.csv|qlike_same_windows\.csv": "fees, utilities and QLIKE margins; no p-value",
        r"daily_utility\.csv|trading_daily\.csv": "daily series behind the tests; no p-value",
        r"dsr\.csv|bootstrap_sharpe\.csv|sizing_rules_33\.csv": "Deflated Sharpe Ratio, bootstrap intervals and the 33 sizing rules; no p-value",
        r"window_contributions\.csv|index_drift\.csv": "descriptive, added in phase 2; no p-value",
    }, "verdict, checks, run information or log; no p-value")


P99_OUTCOME = ["verdict.csv", "summary.csv", "bootstrap.csv", "ensembles.csv"]


def adapt_p99seed(ctx: Ctx):
    """P99: does the forecasting headline survive retraining LightGBM with new seeds? (docs/prereg/P99_SEED_BOOTSTRAP.md)

    verdict.csv: one row per target and subset (next15 and next30 on all days are primary; the other four are
      secondary). The registered rule is that every one of the 20 new 5-seed ensembles beats mcsGARCH with a
      one-sided DM p below 0.05, an intersection-union test whose p-value is the largest DM p (max_dm_p); the
      seed bootstrap p (seed_bootstrap_p, registered secondary S1) goes in alt_p.
    Not counted: the 20 ensembles' own DM p-values (ensembles.csv, the same hypothesis under other seeds; the
      verdict row carries them), the bootstrap and summary tables, the reproduction check of the headline seed,
      the comparison with the referee response (m43_comparison.csv, added after the results) and logs.
    """
    folder = "seed_bootstrap"
    rel = f"{folder}/verdict.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "P99SEED", folder)
        return
    v = ctx.read(rel)
    need = {"check", "kind", "K", "worst_margin", "max_dm_p", "seed_bootstrap_p"}
    if not need.issubset(v.columns):
        adapt_generic(ctx, "P99SEED", folder)
        return
    ps, psn = prereg_status(ctx, folder, P99_OUTCOME)
    side = SIDE_ONE_PRE if ps == PS_YES else SIDE_ONE_STATED
    for _, r in v.iterrows():
        if not np.isfinite(num(r["max_dm_p"])):
            continue  # the overall verdict line
        chk = clean(r["check"])
        tgt, _, sub = chk.partition(", ")
        role = "primary" if clean(r["kind"]) == "primary" else "secondary"
        ctx.add("P99SEED", f"P99: every one of {int(num(r['K']))} new 5-seed ensembles of the hybrid beats mcsGARCH "
                f"({chk}; intersection-union test, p = the largest one-sided DM p)",
                "QLIKE loss of forecasts of realized variance", f"{sub} (n = 792 test days or the expiry subset)",
                r["max_dm_p"], role=role, pre_specified=ps, pre_spec_note=psn, source_file=rel,
                source_key=key("verdict.csv", tgt, sub), sidedness=side, estimate=r["worst_margin"],
                statistic_name="worst margin of the 20 ensembles (QLIKE change, percent)",
                p_method="largest one-sided Diebold-Mariano (HLN) p over 20 ensembles",
                alt_p=alt(seed_bootstrap_p=r["seed_bootstrap_p"]))
    _consume_rest(ctx, "P99SEED", folder, {
        r"ensembles\.csv": "each ensemble's own DM test: the same hypothesis under other seeds; carried by the verdict rows",
        r"summary\.csv|bootstrap\.csv": "spread of margins and the seed bootstrap; the bootstrap p is the verdict rows' alt_p",
        r"reproduction_check\.csv": "check that the headline seed reproduces the committed forecasts; no p-value",
        r"m43_comparison\.csv": "comparison with the referee response, added after the results; no p-value",
        r"daily_losses_.*\.csv|timing\.csv": "daily losses behind the tests and run timing; no p-value",
    }, "run information or log; no p-value")


P107_OUTCOME = ["tests.csv", "verdicts.csv", "pooled_estimates.csv", "bootstrap.csv", "correlations.csv",
                "effective_n.csv", "per_index.csv", "daily_losses.csv"]
# Rows whose p-value is another inference for the same estimate as a counted row: they go in that row's alt_p.
P107_ALT = {"S1a": "joint_bootstrap_p", "R3_block5": "bootstrap_block5_p", "R3_block20": "bootstrap_block20_p",
            "R2": "dk_double_lag_p"}


def adapt_p107pool(ctx: Ctx):
    """P107: one pooled test of the hybrid over Nifty, Bank Nifty and Fin Nifty (docs/prereg/P107_POOLED_THREE_INDEX.md).

    tests.csv: one row per test, with the study's role and sidedness. Counted: P1 (primary, Driscoll-Kraay, Holm over
      two horizons), S1b to S1e (secondary), R1 and R4 (robustness: other estimates) and every exploratory row.
      Not counted separately: S1a (the joint bootstrap), R3 (other block lengths) and R2 (twice the bandwidth), which
      test the same estimate as P1 with another inference and go in P1's alt_p.
    Each index's own result was known before the plan, so the pooled test summarises evidence already counted in
    R1V2, BNFC and FNFC; it is counted here because it is a separate registered hypothesis.
    """
    folder = "p107_pooled_test"
    rel = f"{folder}/tests.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "POOL3", folder)
        return
    d = ctx.read(rel)
    need = {"test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"}
    if not need.issubset(d.columns):
        adapt_generic(ctx, "POOL3", folder)
        return
    ps, psn = prereg_status(ctx, folder, P107_OUTCOME)
    alts = {}
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        stem, _, hor = tid.rpartition("_")
        if stem in P107_ALT:
            alts.setdefault(hor, {})[P107_ALT[stem]] = r["p_raw"]
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        stem, _, hor = tid.rpartition("_")
        if stem in P107_ALT:
            continue
        role = clean(r["role"]) or "exploratory"
        sd = clean(r["sidedness"]).lower()
        if "one-sided" in sd:
            side = SIDE_ONE_PRE if (ps == PS_YES and role in ("primary", "secondary", "robustness")) else SIDE_ONE_STATED
        else:
            side = SIDE_NONDIR
        is_pre = role in ("primary", "secondary", "robustness")
        a = alt(**alts.get(hor, {})) if stem == "P1" else ""
        padj = num(r["p_adjusted"]) if role in ("primary", "secondary") else float("nan")
        ctx.add("POOL3", f"P107 {tid}: {clean(r['hypothesis'])}", "QLIKE loss of forecasts of realized variance, "
                "pooled over Nifty, Bank Nifty and Fin Nifty", clean(r["sample"]), r["p_raw"], role=role,
                pre_specified=ps if is_pre else PS_NO, pre_spec_note=psn if is_pre else "exploratory in the plan or post hoc",
                source_file=rel, source_key=key("tests.csv", tid), sidedness=side, estimate=r["estimate"],
                statistic_name="test statistic (Driscoll-Kraay t, Wald F or DM t, see p_method)",
                statistic=r["statistic"], p_method=clean(r["p_method"]), alt_p=a,
                p_study_adjusted=padj, study_adjust_method=clean(r["adjust_method"]) if np.isfinite(padj) else "")
    _consume_rest(ctx, "POOL3", folder, {
        r"verdicts\.csv|pooled_estimates\.csv|per_index\.csv|se_comparison\.csv|check_against_p104\.csv":
            "verdicts, estimates and checks behind tests.csv; no new p-value",
        r"bootstrap\.csv": "bootstrap draws summary; its p-values are the alt_p of the P1 rows",
        r"correlations\.csv|effective_n\.csv": "correlations and the effective number of indices: descriptive, no test",
        r"posthoc_.*\.csv": "post hoc influence diagnostics, no p-value",
        r"daily_losses\.csv": "daily losses behind the tests; no p-value",
    }, "run information, counts or log; no p-value")


P106_OUTCOME = ["tests.csv", "tests_detail.csv", "contributions.csv", "shares.csv", "verdict.csv", "periods.csv",
                "regimes.csv", "seed_repeats.csv", "daily_losses_*.csv"]


def adapt_p106gain(ctx: Ctx):
    """P106: where the hybrid's gain comes from (docs/prereg/P106_GAIN_DECOMPOSITION.md).

    tests.csv: one row per test, with the study's role and sidedness. Counted: P1 (leave-one-group-out, primary),
      P2 (period interaction, primary), S1 to S6 (secondary), R1 to R3 and R6 to R9 (robustness: other seeds,
      samples or definitions, so other estimates), PL1 and PL2 (placebos, registered) and every exploratory row.
      Not counted separately: R5, the stationary bootstrap of the same period differences as P2; its p goes in
      the matching P2 row's alt_p.
    """
    folder = "p106_gain_decomposition"
    rel = f"{folder}/tests.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "GAIN6", folder)
        return
    d = ctx.read(rel)
    need = {"test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"}
    if not need.issubset(d.columns):
        adapt_generic(ctx, "GAIN6", folder)
        return
    ps, psn = prereg_status(ctx, folder, P106_OUTCOME)
    alts = {}
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        if tid.startswith("R5_"):
            alts["P2_" + tid[3:]] = r["p_raw"]
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        if tid.startswith("R5_") and ("P2_" + tid[3:]) in set(d["test_id"].map(clean)):
            continue
        role = clean(r["role"]) or "exploratory"
        is_pre = role in ("primary", "secondary", "robustness", "placebo")
        sd = clean(r["sidedness"]).lower()
        if "one-sided" in sd:
            side = SIDE_ONE_PRE if (ps == PS_YES and is_pre) else SIDE_ONE_STATED
        else:
            side = SIDE_NONDIR
        a = alt(stationary_bootstrap_p=alts[tid]) if tid in alts else ""
        padj = num(r["p_adjusted"]) if role in ("primary", "secondary", "placebo") else float("nan")
        ctx.add("GAIN6", f"P106 {tid}: {clean(r['hypothesis'])}", "QLIKE loss of the forecasting hybrid with a "
                "feature group removed, or its change across the November 2024 cut", clean(r["sample"]), r["p_raw"],
                role=role, pre_specified=ps if is_pre else PS_NO,
                pre_spec_note=psn if is_pre else "exploratory in the plan or post hoc",
                source_file=rel, source_key=key("tests.csv", tid), sidedness=side, estimate=r["estimate"],
                statistic_name="test statistic (DM t or delta-method t, see p_method)", statistic=r["statistic"],
                p_method=clean(r["p_method"]), alt_p=a,
                p_study_adjusted=padj, study_adjust_method=clean(r["adjust_method"]) if np.isfinite(padj) else "")
    _consume_rest(ctx, "GAIN6", folder, {
        r"tests_detail\.csv|contributions\.csv|shares\.csv|verdict\.csv|periods\.csv|regimes\.csv|seed_repeats\.csv|"
        r"other_index_ranking\.csv|reproduction_check\.csv|timing\.csv":
            "estimates, shares, verdicts and checks behind tests.csv; no new p-value",
        r"daily_losses_.*\.csv": "daily losses behind the tests; no p-value",
    }, "run information, counts or log; no p-value")


P109_OUTCOME = ["tests.csv", "tests_detail.csv", "gains.csv", "shares.csv", "verdict.csv", "gain_by_anchor.csv",
                "params_by_refit.csv", "params_summary.csv", "calibration_factors.csv", "daily_losses_*.csv"]


def adapt_p109mcsx(ctx: Ctx):
    """P109: five pre-specified parametric extensions of mcsGARCH (docs/prereg/P109_MCSGARCH_EXTENSIONS.md).

    tests.csv: one row per test, all Diebold-Mariano on daily mean QLIKE with the study's role and one-sided
      direction. Every row is counted: P1 (primary, the combination against mcsGARCH), S1 to S6 (secondary),
      R1 to R3 (robustness: other samples or the seed-averaged hybrid, so other estimates) and E1 to E6
      (exploratory, including the leave-one-out of the combination).
    """
    folder = "p109_mcsgarch_extensions"
    rel = f"{folder}/tests.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "MCSX", folder)
        return
    d = ctx.read(rel)
    need = {"test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"}
    if not need.issubset(d.columns):
        adapt_generic(ctx, "MCSX", folder)
        return
    ps, psn = prereg_status(ctx, folder, P109_OUTCOME)
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        role = clean(r["role"]) or "exploratory"
        is_pre = role in ("primary", "secondary", "robustness")
        sd = clean(r["sidedness"]).lower()
        if "one-sided" in sd:
            side = SIDE_ONE_PRE if (ps == PS_YES and is_pre) else SIDE_ONE_STATED
        else:
            side = SIDE_NONDIR
        padj = num(r["p_adjusted"]) if role in ("primary", "secondary") else float("nan")
        ctx.add("MCSX", f"P109 {tid}: {clean(r['hypothesis'])}", "QLIKE loss of forecasts of realized variance: "
                "extended mcsGARCH, mcsGARCH or the hybrid", clean(r["sample"]), r["p_raw"],
                role=role, pre_specified=ps if is_pre else PS_NO,
                pre_spec_note=psn if is_pre else "exploratory in the plan or post hoc",
                source_file=rel, source_key=key("tests.csv", tid), sidedness=side, estimate=r["estimate"],
                statistic_name="Diebold-Mariano t (HLN)", statistic=r["statistic"],
                p_method=clean(r["p_method"]),
                p_study_adjusted=padj, study_adjust_method=clean(r["adjust_method"]) if np.isfinite(padj) else "")
    _consume_rest(ctx, "MCSX", folder, {
        r"tests_detail\.csv|gains\.csv|shares\.csv|verdict\.csv|gain_by_anchor\.csv|params_by_refit\.csv|"
        r"params_summary\.csv|calibration_factors\.csv|extensions\.csv|reproduction_check\.csv|timing\.csv":
            "estimates, shares, parameters and checks behind tests.csv; no new p-value",
        r"daily_losses_.*\.csv": "daily losses behind the tests; no p-value",
    }, "run information, counts or log; no p-value")



P113_OUTCOME = ["tests.csv", "verdict.csv", "best.csv", "spec_fit.csv", "grid_cells.csv.gz", "permutation.csv",
                "bootstrap.csv", "changes.csv", "reach.csv"]


def adapt_p113connset(ctx: Ctx):
    """P113: which connectedness setting reproduces the paper's 15 printed totals (docs/prereg/P113_CONNECTEDNESS_SETTINGS.md).

    tests.csv: P1 to P3 (primary) compare the grid's fit to the printed totals with fits to the same totals
      shuffled among the cells (permutation p); P2 and P3 have no p-value because no specification met their
      criteria, so they are not counted. S3 rows (secondary) are day-level bootstrap tests of the paper's
      connectedness changes under the best specification and under the best of the paper's written method.
    """
    folder = "p113_connectedness_settings"
    rel = f"{folder}/tests.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "CONNSET", folder)
        return
    d = ctx.read(rel)
    need = {"test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"}
    if not need.issubset(d.columns):
        adapt_generic(ctx, "CONNSET", folder)
        return
    ps, psn = prereg_status(ctx, folder, P113_OUTCOME)
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        role = clean(r["role"]) or "exploratory"
        is_pre = role in ("primary", "secondary", "robustness")
        sd = clean(r["sidedness"]).lower()
        if "one-sided" in sd:
            side = SIDE_ONE_PRE if (ps == PS_YES and is_pre) else SIDE_ONE_STATED
        else:
            side = SIDE_NONDIR
        padj = num(r["p_adjusted"]) if role in ("primary", "secondary") else float("nan")
        ctx.add("CONNSET", f"P113 {tid}: {clean(r['hypothesis'])}", "Diebold-Yilmaz connectedness totals (paper's "
                "scale) against the printed Tables 3 and 4", clean(r["sample"]), r["p_raw"],
                role=role, pre_specified=ps if is_pre else PS_NO,
                pre_spec_note=psn if is_pre else "exploratory in the plan or post hoc",
                source_file=rel, source_key=key("tests.csv", tid), sidedness=side, estimate=r["estimate"],
                statistic_name="fit (mean absolute miss) or change in total", statistic=r["statistic"],
                p_method=clean(r["p_method"]),
                p_study_adjusted=padj, study_adjust_method=clean(r["adjust_method"]) if np.isfinite(padj) else "")
    _consume_rest(ctx, "CONNSET", folder, {
        r"verdict\.csv|best\.csv|spec_fit\.csv|grid_cells\.csv\.gz|permutation\.csv|bootstrap\.csv|changes\.csv|"
        r"reach\.csv|by_dimension\.csv|near_ties\.csv|loto\.csv|subsets\.csv|windows\.csv|targets\.csv|"
        r"alt_targets\.csv|direction\.csv|baseline_check\.csv":
            "grid fits, bootstrap draws, targets and checks behind tests.csv; no new p-value",
    }, "run information, counts or log; no p-value")


P114_OUTCOME = ["tests.csv", "verdict.csv", "persistence.csv", "persistence_change.csv", "lr.csv", "fits.csv",
                "bootstrap_draws.csv", "irf_curves.csv", "rolling.csv", "rolling_correlations.csv"]


def adapt_p114persfix(ctx: Ctx):
    """P114: does the fall in intraday persistence survive P109's fixes (docs/prereg/P114_PERSISTENCE_FIXED_MODEL.md).

    tests.csv: P1 (primary, Nifty, inferred window, fixed model, day-block bootstrap of the change in the
      impulse-response persistence), S1 to S4 (secondary: other measures and tests, Bank Nifty, the stated window,
      the model with leverage), R1 (robustness: P108's window) and E1 (exploratory rank correlations across
      rolling windows). The likelihood-ratio rows are two-sided chi-square tests; the bootstrap rows are one-sided
      in the paper's direction. Every row is counted.
    """
    folder = "p114_persistence_fixed_model"
    rel = f"{folder}/tests.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "PERSFIX", folder)
        return
    d = ctx.read(rel)
    need = {"test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"}
    if not need.issubset(d.columns):
        adapt_generic(ctx, "PERSFIX", folder)
        return
    ps, psn = prereg_status(ctx, folder, P114_OUTCOME)
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        role = clean(r["role"]) or "exploratory"
        is_pre = role in ("primary", "secondary", "robustness")
        sd = clean(r["sidedness"]).lower()
        if "one-sided" in sd:
            side = SIDE_ONE_PRE if (ps == PS_YES and is_pre) else SIDE_ONE_STATED
        else:
            side = SIDE_NONDIR
        padj = num(r["p_adjusted"]) if role in ("primary", "secondary") else float("nan")
        ctx.add("PERSFIX", f"P114 {tid}: {clean(r['hypothesis'])}", "intraday persistence (impulse-response "
                "half-life or alpha + beta) of mcsGARCH with P109's fixes, before against after the expiry change",
                clean(r["sample"]), r["p_raw"],
                role=role, pre_specified=ps if is_pre else PS_NO,
                pre_spec_note=psn if is_pre else "exploratory in the plan or post hoc",
                source_file=rel, source_key=key("tests.csv", tid), sidedness=side, estimate=r["estimate"],
                statistic_name="bootstrap share, likelihood-ratio chi-square or Spearman rho (see p_method)",
                statistic=r["statistic"], p_method=clean(r["p_method"]),
                p_study_adjusted=padj, study_adjust_method=clean(r["adjust_method"]) if np.isfinite(padj) else "")
    _consume_rest(ctx, "PERSFIX", folder, {
        r"verdict\.csv|persistence\.csv|persistence_change\.csv|lr\.csv|fits\.csv|bootstrap_draws\.csv|"
        r"irf_curves\.csv|rolling\.csv|rolling_correlations\.csv|posthoc_night_cap\.csv|"
        r"reproduction_bootstrap_lr\.csv|reproduction_table2\.csv":
            "fits, bootstrap draws, impulse responses, rolling windows and reproduction checks behind tests.csv; "
            "no new p-value",
    }, "run information, counts or log; no p-value")


P116_OUTCOME = ["tests.csv", "tests_detail.csv", "losses.csv", "verdict.csv", "attribution.csv", "gain_by_anchor.csv",
                "daily_losses_*.csv"]


def adapt_p116benchcpp(ctx: Ctx):
    """P116: the hybrid and our mcsGARCH against mcsGARCH as the paper specifies it (docs/prereg/P116_BENCHMARK_AS_PRINTED.md).

    tests.csv: P (primary: the hybrid against mcs_cpp, one-sided; our mcsGARCH against mcs_cpp, two-sided; 15 and
      30 minutes, Holm over four), S1 to S5 (secondary: last hour, expiry days, P109's combination, Bank Nifty,
      Fin Nifty) and E1 to E5 (exploratory: single switches, anchors, non-expiry days). All are Diebold-Mariano
      tests on daily mean QLIKE. Every row is counted.
    """
    folder = "p116_benchmark_as_printed"
    rel = f"{folder}/tests.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "BENCHCPP", folder)
        return
    d = ctx.read(rel)
    need = {"test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"}
    if not need.issubset(d.columns):
        adapt_generic(ctx, "BENCHCPP", folder)
        return
    ps, psn = prereg_status(ctx, folder, P116_OUTCOME)
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        role = clean(r["role"]) or "exploratory"
        is_pre = role in ("primary", "secondary", "robustness")
        sd = clean(r["sidedness"]).lower()
        if "one-sided" in sd:
            side = SIDE_ONE_PRE if (ps == PS_YES and is_pre) else SIDE_ONE_STATED
        else:
            side = SIDE_NONDIR
        padj = num(r["p_adjusted"]) if role in ("primary", "secondary") else float("nan")
        ctx.add("BENCHCPP", f"P116 {tid}: {clean(r['hypothesis'])}", "QLIKE loss of forecasts of realized variance: "
                "the hybrid, our mcsGARCH or a variant against mcsGARCH as the paper specifies it",
                clean(r["sample"]), r["p_raw"],
                role=role, pre_specified=ps if is_pre else PS_NO,
                pre_spec_note=psn if is_pre else "exploratory in the plan or post hoc",
                source_file=rel, source_key=key("tests.csv", tid), sidedness=side, estimate=r["estimate"],
                statistic_name="Diebold-Mariano t (HLN)", statistic=r["statistic"], p_method=clean(r["p_method"]),
                p_study_adjusted=padj, study_adjust_method=clean(r["adjust_method"]) if np.isfinite(padj) else "")
    _consume_rest(ctx, "BENCHCPP", folder, {
        r"tests_detail\.csv|losses\.csv|verdict\.csv|attribution\.csv|gain_by_anchor\.csv|params_by_refit\.csv|"
        r"params_summary\.csv|reproduction_check\.csv|timing\.csv":
            "losses, attributions, parameters and checks behind tests.csv; no new p-value",
        r"daily_losses_.*\.csv": "daily losses behind the tests; no p-value",
    }, "run information, counts or log; no p-value")



P110_OUTCOME = ["tests.csv", "tests_detail.csv", "verdict.csv", "claims.csv", "contributions.csv", "shares.csv",
                "equivalence_bounds.csv", "pooled.csv", "periods.csv", "seed_repeats.csv", "daily_losses_*.csv"]


def adapt_p110gainstress(ctx: Ctx):
    """P110: P106's three claims stress-tested on Bank Nifty and Fin Nifty (docs/prereg/P110_GAIN_STRESS_TEST.md).

    tests.csv: F1 to F3 (primary: carriers of the gain, equivalence of the other groups within 1 point, the mcsinfo
      hybrid against mcsGARCH and non-inferior to the full hybrid), S1 to S3 (secondary, including pooled tests),
      R1 to R6 (robustness: single seed sets and other adjustments, so repeated estimates), PL (placebo: seed sets
      against each other), E and PH (exploratory and post hoc period tests). Diebold-Mariano, equivalence (two
      one-sided tests) and non-inferiority tests on daily mean QLIKE. Every row is counted.
    """
    folder = "p110_gain_stress_test"
    rel = f"{folder}/tests.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "GAINSTRESS", folder)
        return
    d = ctx.read(rel)
    need = {"test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"}
    if not need.issubset(d.columns):
        adapt_generic(ctx, "GAINSTRESS", folder)
        return
    ps, psn = prereg_status(ctx, folder, P110_OUTCOME)
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        role = clean(r["role"]) or "exploratory"
        is_pre = role in ("primary", "secondary", "robustness", "placebo")
        sd = clean(r["sidedness"]).lower()
        if "one-sided" in sd and "two one-sided" not in sd:
            side = SIDE_ONE_PRE if (ps == PS_YES and is_pre) else SIDE_ONE_STATED
        else:
            side = SIDE_NONDIR
        padj = num(r["p_adjusted"]) if role in ("primary", "secondary") else float("nan")
        ctx.add("GAINSTRESS", f"P110 {tid}: {clean(r['hypothesis'])}", "QLIKE loss of the hybrid with or without "
                "a group of inputs, or of the mcsinfo hybrid, against the full hybrid or mcsGARCH",
                clean(r["sample"]), r["p_raw"],
                role=role, pre_specified=ps if is_pre else PS_NO,
                pre_spec_note=psn if is_pre else "exploratory in the plan or post hoc",
                source_file=rel, source_key=key("tests.csv", tid), sidedness=side, estimate=r["estimate"],
                statistic_name="test statistic (DM t, equivalence or non-inferiority t, see p_method)",
                statistic=r["statistic"], p_method=clean(r["p_method"]),
                p_study_adjusted=padj, study_adjust_method=clean(r["adjust_method"]) if np.isfinite(padj) else "")
    _consume_rest(ctx, "GAINSTRESS", folder, {
        r"tests_detail\.csv|verdict\.csv|claims\.csv|contributions\.csv|shares\.csv|equivalence_bounds\.csv|"
        r"pooled\.csv|periods\.csv|seed_repeats\.csv|other_index_ranking\.csv|reproduction_check\.csv|"
        r"timing\.csv":
            "estimates, bounds, verdicts and checks behind tests.csv; no new p-value",
        r"daily_losses_.*\.csv": "daily losses behind the tests; no p-value",
    }, "run information, counts or log; no p-value")


P117_OUTCOME = ["tests.csv", "verdict.csv", "placebo_summary.csv", "placebo_pairs.csv", "windows_nifty.csv",
                "windows_banknifty.csv", "did.csv", "control_fits.csv", "control_bootstrap_draws.csv"]


def adapt_p117persplac(ctx: Ctx):
    """P117: is the fall in intraday persistence larger than the changes between placebo windows
    (docs/prereg/P117_PERSISTENCE_PLACEBO.md).

    tests.csv: P (primary: Nifty, primary placebo set, reference mcsGARCH and the fixed model F, Holm over two),
      S1 to S5 (secondary: Bank Nifty, the clean, literal and day-disjoint sets, the conditional placebo), D
      (secondary: difference-in-differences against Fin Nifty) and E1 (exploratory: Fin Nifty's own change). The
      placebo rows are one-sided rank p-values over placebo pairs of windows; the D and E1 rows are one-sided
      bootstrap p-values. Every row is counted. The post hoc flat-day files add no p-value.
    """
    folder = "p117_persistence_placebo"
    rel = f"{folder}/tests.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "PERSPLAC", folder)
        return
    d = ctx.read(rel)
    need = {"test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"}
    if not need.issubset(d.columns):
        adapt_generic(ctx, "PERSPLAC", folder)
        return
    ps, psn = prereg_status(ctx, folder, P117_OUTCOME)
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        role = clean(r["role"]) or "exploratory"
        is_pre = role in ("primary", "secondary", "robustness")
        sd = clean(r["sidedness"]).lower()
        if "one-sided" in sd:
            side = SIDE_ONE_PRE if (ps == PS_YES and is_pre) else SIDE_ONE_STATED
        else:
            side = SIDE_NONDIR
        padj = num(r["p_adjusted"]) if role in ("primary", "secondary") else float("nan")
        ctx.add("PERSPLAC", f"P117 {tid}: {clean(r['hypothesis'])}", "change in the log half-life of a 12:15 "
                "intraday volatility shock, before against after the expiry change, compared with placebo pairs of "
                "windows or with Fin Nifty",
                clean(r["sample"]), r["p_raw"],
                role=role, pre_specified=ps if is_pre else PS_NO,
                pre_spec_note=psn if is_pre else "exploratory in the plan or post hoc",
                source_file=rel, source_key=key("tests.csv", tid), sidedness=side, estimate=r["estimate"],
                statistic_name="share of placebo pairs at or below the actual, or bootstrap share (see p_method)",
                statistic=r["statistic"], p_method=clean(r["p_method"]),
                p_study_adjusted=padj, study_adjust_method=clean(r["adjust_method"]) if np.isfinite(padj) else "")
    _consume_rest(ctx, "PERSPLAC", folder, {
        r"verdict\.csv|placebo_summary\.csv|placebo_pairs\.csv|windows_nifty\.csv|windows_banknifty\.csv|did\.csv|"
        r"control_fits\.csv|control_bootstrap_draws\.csv|reproduction_p114_rolling\.csv|reproduction_p114_control\.csv":
            "window fits, placebo pairs, the control fits and draws and the reproduction checks behind tests.csv; "
            "no new p-value",
        r"posthoc_.*\.csv": "post hoc descriptive tables (flat-bar days, bound windows, pairs by period); no p-value",
    }, "run information, counts or log; no p-value")


P118_OUTCOME = ["tests.csv", "tests_detail.csv", "verdict.csv", "losses.csv", "absorption.csv", "seed_repeats.csv",
                "equivalence_bounds.csv", "gain_by_anchor.csv", "daily_losses_*.csv"]


def adapt_p118hybcpp(ctx: Ctx):
    """P118: the identical hybrid recipe built on mcsGARCH as the paper specifies it
    (docs/prereg/P118_HYBRID_ON_PAPER_MCSGARCH.md).

    tests.csv: P (primary: H1 hybrid_cpp beats mcs_cpp, one-sided DM; H2 hybrid_cpp and the committed hybrid are
      equivalent within 0.5 point, two one-sided tests; 15 and 30 minutes, Holm over four), S0 to S5 (secondary:
      the two-sided difference behind the decision rule, last hour, expiry days, P109's combination, Bank Nifty,
      Fin Nifty), R (robustness: seed set 7 alone, delta-method TOST), PL (placebo: seed sets of one model against
      each other) and E (exploratory). Diebold-Mariano and equivalence tests on daily mean QLIKE. Every row is
      counted; equivalence rows are reported as they are (not directional).
    """
    folder = "p118_hybrid_on_paper_mcsgarch"
    rel = f"{folder}/tests.csv"
    if not ctx.has(rel):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "HYBCPP", folder)
        return
    d = ctx.read(rel)
    need = {"test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw", "sidedness", "p_method",
            "p_adjusted", "adjust_method"}
    if not need.issubset(d.columns):
        adapt_generic(ctx, "HYBCPP", folder)
        return
    ps, psn = prereg_status(ctx, folder, P118_OUTCOME)
    for _, r in d.iterrows():
        tid = clean(r["test_id"])
        role = clean(r["role"]) or "exploratory"
        is_pre = role in ("primary", "secondary", "robustness", "placebo")
        sd = clean(r["sidedness"]).lower()
        if "one-sided" in sd and "two one-sided" not in sd:
            side = SIDE_ONE_PRE if (ps == PS_YES and is_pre) else SIDE_ONE_STATED
        else:
            side = SIDE_NONDIR
        padj = num(r["p_adjusted"]) if role in ("primary", "secondary") else float("nan")
        ctx.add("HYBCPP", f"P118 {tid}: {clean(r['hypothesis'])}", "QLIKE loss of forecasts of realized variance: "
                "the hybrid built on mcsGARCH as the paper specifies it, against that mcsGARCH or the committed hybrid",
                clean(r["sample"]), r["p_raw"],
                role=role, pre_specified=ps if is_pre else PS_NO,
                pre_spec_note=psn if is_pre else "exploratory in the plan or post hoc",
                source_file=rel, source_key=key("tests.csv", tid), sidedness=side, estimate=r["estimate"],
                statistic_name="test statistic (DM t or equivalence t, see p_method)", statistic=r["statistic"],
                p_method=clean(r["p_method"]),
                p_study_adjusted=padj, study_adjust_method=clean(r["adjust_method"]) if np.isfinite(padj) else "")
    _consume_rest(ctx, "HYBCPP", folder, {
        r"tests_detail\.csv|verdict\.csv|losses\.csv|absorption\.csv|seed_repeats\.csv|equivalence_bounds\.csv|"
        r"gain_by_anchor\.csv|params_by_job\.csv|reproduction_check\.csv|timing\.csv":
            "losses, absorbed shares, seed repeats, parameters and checks behind tests.csv; no new p-value",
        r"daily_losses_.*\.csv": "daily losses behind the tests; no p-value",
    }, "run information, counts or log; no p-value")


def adapt_power_real(ctx: Ctx):
    """results/power_real/: simulations calibrated on real data. Excluded, unless a p-value in it is not
    already in the ledger (then it would be a new test of data and is flagged for review)."""
    folder = "power_real"
    if not ctx.res(folder).exists():
        return
    known = np.array(sorted({round(num(r["p_reported"]), 12) for r in ctx.rows} |
                            {round(float(x), 12) for r in ctx.rows for x in re.findall(r"=([0-9.eE+-]+)", r["alt_p"]) if np.isfinite(num(x))}))
    unmatched = []
    for f in sorted(ctx.res(folder).glob("*.csv")):
        rel = f"{folder}/{f.name}"
        ctx.consumed.add(rel)
        d = pd.read_csv(f)
        if "p" in d.columns and f.name == "estimates.csv":
            for _, r in d.iterrows():
                v = num(r["p"])
                if not np.isfinite(v):
                    continue
                if known.size and np.min(np.abs(known - v) / np.maximum(v, 1e-300)) < 1e-4:
                    continue
                if known.size and np.min(np.abs(known - v)) < 5e-5:
                    continue
                unmatched.append(f"{clean(r.get('test', ''))} p={fmt(v)}")
    ctx.skip("POWER", f"{folder}/", "power study calibrated on real data", 1,
             "simulation; estimates.csv re-reports p-values of tests already in the ledger (p_hac_t is an alternative standard error)")
    if unmatched:
        ctx.skip("POWER", f"{folder}/estimates.csv", "p-values not matched to a ledger row", len(unmatched),
                 "review: " + "; ".join(unmatched[:12]))


# ----------------------------------------------------------------------------
# Pre-registered studies P92 (Sensex swap) and P96 (synthetic control)
# ----------------------------------------------------------------------------


def _consume_rest(ctx: Ctx, study: str, folder: str, reasons: dict, default: str):
    """Mark every file of a study folder that the adapter did not read as consumed, with one not-counted line per
    reason (reasons: {regular expression on the file name: reason})."""
    base = ctx.res(folder)
    if not base.exists():
        return
    groups = {}
    for f in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = str(f.relative_to(ctx.res(""))).replace("\\", "/")
        if rel in ctx.consumed:
            continue
        ctx.consumed.add(rel)
        why = next((v for k, v in reasons.items() if re.fullmatch(k, f.name)), default)
        groups.setdefault(why, []).append(str(f.relative_to(base)).replace("\\", "/"))
    for why, names in groups.items():
        what = ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
        ctx.skip(study, f"{folder}/" if len(names) > 1 else f"{folder}/{names[0]}", what, len(names), why)


SWAP_OUTCOME = ["tests.csv", "robustness.csv", "verdict.json", "cell_means.csv", "day_values.csv", "h2_regression.csv"]
SWAP_P1_ALT = {"block of 1 week(s)": "perm_p_block_1_week", "block of 2 week(s)": "perm_p_block_2_weeks",
               "block of 8 week(s)": "perm_p_block_8_weeks",
               "labels moved separately within each weekday (blocks of 4 same-weekday days)": "perm_p_within_weekday"}
SWAP_OUTCOME_DESC = {"net_day": "net spillover from Nifty to Sensex (NET), whole day",
                     "net_lh": "net spillover from Nifty to Sensex (NET), last hour",
                     "tot_day": "total connectedness of Nifty and Sensex, whole day"}


def adapt_swap(ctx: Ctx):
    """P92: the Nifty and Sensex expiry-weekday swap of 1 September 2025 (docs/prereg/P92_SENSEX_SWAP.md).

    tests.csv: test (P1 primary; S1 to S3 secondary; PL1 to PL3 placebo), alternative (greater, less or
      two-sided, as registered), estimate, perm_p (block permutation, 4-week blocks: one-sided in the registered
      direction for P1, S1 and S3, two-sided for the placebos), perm_p_greater, perm_p_less, hac_p (Newey-West,
      two-sided), p_used (the p the study decides on: perm_p, or hac_p for S2, a regression), holm_p_secondary
      (Holm over S1 to S3; cross-checked).
    robustness.csv: check, estimate, perm_p (one-sided as P1 unless alternative_used says otherwise), hac_p.
      The block lengths of 1, 2 and 8 weeks and the within-weekday permutation re-test P1's own estimate with
      another permutation scheme: alternative p-values of P1 (its alt_p), not new tests. Every other row changes
      the estimate and is counted as robustness. The two rows the plan calls description (total connectedness,
      and H2 without weekday effects) have p-values and are counted as exploratory.
    Not counted: h2_regression.csv (coefficients of S2's regression, controls included), cell means, per-day
      values, coverage, calendars, excluded days, MDE files, mcsGARCH fits, verdict.json (repeats tests.csv).
    """
    folder = "sensex_swap"
    rel = f"{folder}/tests.csv"
    need = {"test", "family", "alternative", "estimate", "perm_p", "hac_p", "p_used"}
    if not ctx.has(rel) or not need.issubset(pd.read_csv(ctx.res(rel), nrows=0).columns):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "SWAP", folder)
        return
    ps_status, ps_note = prereg_status(ctx, folder, SWAP_OUTCOME)
    side_pre = SIDE_ONE_PRE if ps_status == PS_YES else SIDE_ONE_STATED
    d = ctx.read(rel)
    rob_rel = f"{folder}/robustness.csv"
    rob = ctx.read(rob_rel) if ctx.has(rob_rel) else None
    p1_alt = {}
    if rob is not None:
        for _, r in rob.iterrows():
            if clean(r["check"]) in SWAP_P1_ALT:
                p1_alt[SWAP_P1_ALT[clean(r["check"])]] = r["perm_p"]
    sec = (d["family"] == "secondary") & d["holm_p_secondary"].notna()
    rmap = {}
    if sec.any():
        rec = ctx.check("SWAP", "secondaries S1 to S3 (Holm)", rel, "holm", d.loc[sec, "p_used"], d.loc[sec, "holm_p_secondary"])
        rmap = dict(zip(d.index[sec], rec))
    for idx, r in d.iterrows():
        test, fam, dirn = clean(r["test"]), clean(r["family"]), clean(r["alternative"]).lower()
        role = {"primary": "primary", "secondary": "secondary", "placebo": "placebo"}.get(fam, "unlabelled")
        use_perm = np.isfinite(num(r["perm_p"]))
        directional = dirn in ("greater", "less")
        if use_perm:
            p = r["perm_p"]
            side = side_pre if directional else SIDE_TWO
            pm = (f"block permutation (blocks of {_int(r.get('block_weeks'))} weeks), "
                  + (f"one-sided ({dirn})" if directional else "two-sided"))
            other = r["perm_p_less"] if dirn == "greater" else (r["perm_p_greater"] if dirn == "less" else float("nan"))
            extra = dict(p1_alt) if test == "P1" else {}
            altp = alt(hac_p_two_sided=r["hac_p"], perm_p_other_direction=other, **extra)
        else:
            p, side, pm, altp = r["hac_p"], SIDE_TWO, "OLS with Newey-West standard errors, two-sided", ""
        if not _same_num(num(p), num(r["p_used"])):
            ctx.warn(f"{rel}: {test} p_used differs from the p the ledger uses")
        pred = {"greater": " (registered prediction: positive)", "less": " (registered prediction: negative)"}.get(dirn, "")
        row = ctx.add("SWAP", f"{test}: {clean(r['what'])}{pred}", SWAP_OUTCOME_DESC.get(clean(r["outcome"]), clean(r["outcome"])),
                      f"{_int(r['n_days'])} days" + (f"; cells {clean(r['n_cells'])}" if clean(r.get("n_cells")) else ""), p,
                      role=role, pre_specified=ps_status, pre_spec_note=ps_note, source_file=rel,
                      source_key=key("tests.csv", test), sidedness=side, estimate=r["estimate"],
                      statistic_name="difference in differences of NET (points)" if use_perm else "expiry-day coefficient (points)",
                      p_method=pm, alt_p=altp,
                      note=("P1's other permutation schemes (block lengths, within weekday) are in alt_p, not counted"
                            if test == "P1" and p1_alt else ""),
                      p_study_adjusted=r["holm_p_secondary"] if idx in rmap else float("nan"),
                      study_adjust_method="Holm over S1 to S3" if idx in rmap else "")
        if idx in rmap:
            _set(row, "p_study_adjusted_recomputed", rmap[idx])
    if rob is not None:
        n_alt = 0
        for _, r in rob.iterrows():
            chk = clean(r["check"])
            if chk in SWAP_P1_ALT:
                n_alt += 1
                continue
            used = clean(r.get("alternative_used", "")).lower()
            descriptive = "descriptive" in chk.lower()
            use_perm = np.isfinite(num(r["perm_p"]))
            p = r["perm_p"] if use_perm else r["hac_p"]
            if descriptive or used == "two-sided" or not use_perm:
                side = SIDE_TWO
            else:
                side = side_pre
            dirn = used if used in ("greater", "less") else "greater"
            pm = (f"block permutation, one-sided ({dirn})" if side != SIDE_TWO else
                  ("block permutation, two-sided" if use_perm else "OLS with Newey-West standard errors, two-sided"))
            if descriptive:
                role, ps, psn = "exploratory", PS_NO, ("the plan reports this as description only (Sections 7 and 9); "
                                                       "counted because a p-value was reported and seen")
            else:
                role, ps, psn = "robustness", ps_status, ps_note + "; robustness of the primary (plan Section 9, never decisive)"
            ctx.add("SWAP", f"Robustness of P1: {chk}", "net spillover from Nifty to Sensex (NET), whole day",
                    f"{_int(r['n_days'])} days", p, role=role, pre_specified=ps, pre_spec_note=psn, source_file=rob_rel,
                    source_key=key("robustness.csv", chk), sidedness=side, estimate=r["estimate"],
                    statistic_name="estimate (points)", p_method=pm,
                    alt_p=alt(hac_p_two_sided=r["hac_p"]) if use_perm else "")
        ctx.skip("SWAP", rob_rel, "block lengths of 1, 2 and 8 weeks and the within-weekday permutation", n_alt,
                 "other permutation schemes for P1's own estimate: alternative p-values of P1 (in its alt_p)")
    _consume_rest(ctx, "SWAP", folder, {
        r"h2_regression\.csv": "coefficients of S2's regression (the expiry term is S2; weekday and period terms are controls)",
        r"verdict\.json": "verdict and Holm values, repeating tests.csv",
        r"(cell_means|day_values)\.csv": "cell means and per-day values behind the tests; no p-value",
        r"mde.*\.csv": "minimum detectable effects (design), no p-value",
    }, "coverage, calendars, excluded days, mcsGARCH fits or run information; no p-value")


SYNTH_INDEX = {"BN": "Bank Nifty (Wednesday)", "FIN": "Fin Nifty (Tuesday)", "MID": "Nifty Midcap Select (Monday)",
               "NIFTY": "Nifty 50 Wednesday (kept its weekly options)"}
SYNTH_OUTCOME = ["results.csv", "decision.json", "placebo_space_*.csv", "test_inversion_*.csv"]


def adapt_synth(ctx: Ctx):
    """P96: did losing the weekly expiry (November 2024) remove the expiry-weekday premium? Synthetic control
    (docs/prereg/P96_SYNTHETIC_CONTROL.md).

    results.csv: analysis (primary, secondary, S1 to S8 sensitivities, placebo in time, descriptive), treated
      (BN, FIN, MID; NIFTY descriptive), effect (post-period gap to the synthetic control minus the pre-period
      gap, log points), p_ratio (rank of the post/pre RMSPE ratio among the treated unit and its placebos: the
      registered test, not directional; main p), p_neg (one-sided rank p of the standardized effect, registered
      as "reported only": alternative), did (treated minus the plain donor average, post minus pre), did_hac_p
      (Newey-West, two-sided), did_rank_p_neg (one-sided rank p for a fall: alternative).
    decision.json: Holm over the two secondaries (cross-checked) and the joint test of all three (one-sided p for
      a negative mean standardized effect, registered as supplementary, not a decision input).
    Ratio-test rows: primary (BN), secondary (FIN, MID), robustness (S1 to S8), placebo (placebo in time),
      exploratory (Nifty 50's Wednesday, descriptive by design). Difference-in-differences rows: exploratory
      (registered as a cross-check, not as a hypothesis; reading it as a rise is post hoc).
    Rows with the same p and estimate are counted once (Midcap S7 equals its main run, and the classic synthetic
      control S3 leaves the DiD unchanged).
    Not counted: test_inversion_*.csv (the confidence set: ratio p over a grid of shifted effects),
      placebo_space_*.csv (placebo draws behind p_ratio), weights, weekly series, donor table, figure.
    """
    folder = "synthetic_control"
    rel = f"{folder}/results.csv"
    need = {"analysis", "treated", "effect", "p_ratio", "p_neg", "did", "did_hac_p"}
    if not ctx.has(rel) or not need.issubset(pd.read_csv(ctx.res(rel), nrows=0).columns):
        if ctx.res(folder).exists():
            adapt_generic(ctx, "SYNTH", folder)
        return
    ps_status, ps_note = prereg_status(ctx, folder, SYNTH_OUTCOME)
    side_pre = SIDE_ONE_PRE if ps_status == PS_YES else SIDE_ONE_STATED
    d = ctx.read(rel)
    dec = {}
    if ctx.has(f"{folder}/decision.json"):
        ctx.consumed.add(f"{folder}/decision.json")
        dec = json.loads(ctx.res(f"{folder}/decision.json").read_text())
    holm = {}
    secs = dec.get("secondaries", {})
    if secs:
        names = list(secs)
        rec = ctx.check("SYNTH", "secondaries Fin Nifty and Midcap Select (Holm)", f"{folder}/decision.json", "holm",
                        [secs[k].get("p_ratio") for k in names], [secs[k].get("holm_p") for k in names])
        holm = {k: (secs[k].get("holm_p"), rec[i]) for i, k in enumerate(names)}
    ratio_rows, did_rows = [], []
    for _, r in d.iterrows():
        an, tr = clean(r["analysis"]), clean(r["treated"])
        idx_desc = SYNTH_INDEX.get(tr, tr)
        if an == "primary":
            role = "primary"
        elif an == "secondary":
            role = "secondary"
        elif re.match(r"S\d", an):
            role = "robustness"
        elif an.startswith("placebo in time"):
            role = "placebo"
        elif an.startswith("descriptive"):
            role = "exploratory"
        else:
            role = "unlabelled"
        if role == "exploratory":
            ps, psn = PS_NO, ("descriptive only in the plan (Section 9); counted because a p-value was reported and seen")
        else:
            ps, psn = ps_status, ps_note + ("; sensitivity, plan Section 10 (none changes a verdict)" if role == "robustness" else "")
        samp = f"{_int(r['n_pre'])} pre and {_int(r['n_post'])} post weeks, {_int(r['J'])} donor placebos"
        label = an if an not in ("primary", "secondary") else f"{an} analysis"
        hy = (f"{idx_desc}: the old expiry weekday's volatility premium over its synthetic control changed after the "
              f"November 2024 cut ({label}; registered prediction: it fell)")
        n0 = len(ctx.rows)
        row = ctx.add("SYNTH", hy, "week-relative log Parkinson variance of the old expiry weekday", samp, r["p_ratio"],
                      role=role, pre_specified=ps, pre_spec_note=psn, source_file=rel, source_key=key("ratio", an, tr),
                      estimate=r["effect"], statistic_name="effect (log points; gap post minus gap pre)",
                      statistic=r["ratio"], p_method="rank of the post/pre RMSPE ratio among the treated unit and its "
                      "placebos in space (not directional; smallest possible p = 1/(J+1))",
                      alt_p=alt(p_signed_one_sided_fall=r["p_neg"]),
                      p_study_adjusted=holm[tr][0] if (role == "secondary" and tr in holm) else float("nan"),
                      study_adjust_method="Holm over Fin Nifty and Midcap Select" if (role == "secondary" and tr in holm) else "")
        if row is not None and role == "secondary" and tr in holm:
            row["p_study_adjusted_recomputed"] = holm[tr][1]
        if len(ctx.rows) > n0:
            ratio_rows.append(ctx.rows.pop())
        n0 = len(ctx.rows)
        ctx.add("SYNTH", f"{idx_desc}: the old expiry weekday moved against the plain average of the donors "
                f"(difference-in-differences cross-check; {label})", "week-relative log Parkinson variance of the old expiry weekday",
                samp, r["did_hac_p"], role="exploratory", pre_specified=PS_NO,
                pre_spec_note=("registered as a cross-check reported with a two-sided Newey-West p (plan Section 9), not as a "
                               "hypothesis; reading it as evidence of a rise is post hoc (SYNTHETIC_CONTROL.md Section 9)"),
                source_file=rel, source_key=key("did", an, tr), estimate=r["did"],
                statistic_name="DiD (log points)", p_method="Newey-West (4 lags), two-sided",
                alt_p=alt(did_rank_p_fall=r["did_rank_p_neg"]))
        if len(ctx.rows) > n0:
            did_rows.append(ctx.rows.pop())
    kept, dropped = dedup_exact(ratio_rows + did_rows)
    ctx.rows.extend(kept)
    ctx.skip("SYNTH", rel, "rows repeating another row (same p and estimate)", len(dropped),
             "same test (Midcap S7 equals its main run; the classic synthetic control S3 leaves the DiD unchanged); counted once")
    jt = dec.get("joint_test_all_three")
    if jt and np.isfinite(num(jt.get("p_neg"))):
        ctx.add("SYNTH", "Bank Nifty, Fin Nifty and Midcap Select together: the mean standardized effect is negative "
                "(the old expiry weekdays calmed down)", "standardized effect (effect / pre RMSPE), mean of three",
                f"{_int(jt.get('n_combinations'))} combinations of donor placebos", jt["p_neg"], role="secondary",
                pre_specified=ps_status, pre_spec_note=ps_note + "; supplementary test, not a decision input (plan Section 8)",
                source_file=f"{folder}/decision.json", source_key="joint_test_all_three", sidedness=side_pre,
                estimate=jt.get("mean_std_effect"), statistic_name="mean standardized effect",
                p_method="rank among all combinations of one donor placebo per index, one-sided (mean < 0)",
                note="not in the Holm family of the two secondaries")
    _consume_rest(ctx, "SYNTH", folder, {
        r"test_inversion_.*\.csv": "confidence set by test inversion: ratio p over a grid of shifted effects, not separate hypotheses",
        r"placebo_space_.*\.csv": "placebo draws (one per donor) behind p_ratio",
        r"weights_.*\.csv": "synthetic-control weights",
        r"weekly_.*\.csv": "weekly series of the treated index and donors (data)",
        r"donor_table\.csv": "donor pool checks (data)",
    }, "figure or run information; no p-value")


# ----------------------------------------------------------------------------
# Referee responses (post hoc): F2, forecasting, expiry premia
# ----------------------------------------------------------------------------

REF_PS = "post hoc: done after the referee report of 3 November 2026 ({}); nothing in it is pre-registered"
F2_NOT_ROBUST = ("newey_west_5", "driscoll_kraay", "within_block_shuffle")
F2_RULE = {"pro": "NSE Pro participant", "fii": "NSE FII", "fii_plus_pro": "NSE FII plus Pro", "client": "NSE Client",
           "client_mirror": "mirror of NSE Client", "us_convention": "US convention", "dii": "NSE DII",
           "short_all": "every category short"}
F2_SAMPLE = {"full": "full sample", "2023_2024": "2023 to 2024", "2025_onward": "2025 onward", "pre_auction": "before the closing auction"}


def _f2_robust(method: str) -> bool:
    return not clean(method).startswith(F2_NOT_ROBUST)


def _f2_pick(g: pd.DataFrame, pcol="p"):
    """Main p of one estimate reported under several inference methods: the largest p among the methods that
    respect the persistence of the state (the study's cautious reading; the maximum of valid p-values is a valid,
    conservative p-value). Newey-West with 5 lags, Driscoll-Kraay and the within-block shuffle are reported by
    the study for comparison only. Returns (p, method, robust methods, {other method: p})."""
    pm = [(clean(m), num(p)) for m, p in zip(g["method"], g[pcol]) if np.isfinite(num(p))]
    if not pm:
        return float("nan"), "", [], {}
    rob = [x for x in pm if _f2_robust(x[0])] or pm
    m_main, p_main = max(rob, key=lambda x: x[1])
    others = {m: p for m, p in pm if m != m_main}
    return p_main, m_main, [m for m, _ in rob], others


def _f2_nw(g: pd.DataFrame, pcol="p"):
    h = g[g["method"].map(clean) == "newey_west_5"]
    return num(h[pcol].iloc[0]) if len(h) else float("nan")


def adapt_ref_f2(ctx: Ctx):
    """Referee response on F2 (docs/REFEREE_RESPONSE_F2.md; experiments/referee_f2.py and referee_f2_positions.py).
    All post hoc, so every row is exploratory.

    One row per estimate. Each estimate is reported under several inference methods (Newey-West 5 lags,
    Driscoll-Kraay, Newey-West 60 lags fixed-b, CR1 and wild cluster bootstraps by month, episode and quarter,
    block permutations of the t and the coefficient, within-block shuffles). The main p is the largest p among the
    methods that respect the persistence of the state, the study's own cautious reading; every other p is in
    alt_p. Files:
    item1_unconditional_slope.csv (test, sample, coef, method, p_two_sided; p_one_sided_momentum as alt),
    item3_f2_inference.csv, item4_other_rules.csv, item5_alternatives.csv (rule, spec, sample or case, term,
      coef, method, p; rows "not estimable" have no p),
    item6_angle_equality.csv (per sample and block length: the bootstrap test of equal response angles, the
      minimum-distance J test (chi-square, 1 or 2 df) and the Wald test of theta_a = 0; the block lengths and df
      are alternatives; the Wald test is counted only where it differs from the J test),
    item7_minute_autocorrelation.csv (rows "expiry minus other": p_two_sided) and item7_junction.csv (p),
    item8_positions_f2.csv (category, position, form, sample, coef, method, p) and item8_horse_race.csv
      (horse_race, position, form, coef, method, p; joint Wald rows are not directional).
    The same estimate in two files is counted once (same coefficient, days and Newey-West p; item 3 first):
    simulated p-values differ slightly between files because each draws its own. The F2 rows re-estimate R5 on
    932 days (R5 now uses 929), so they are different estimates from R5's and are counted.
    Not counted: item 2 (shares, runs, autocorrelations of the sign: descriptive), item 6 bootstrap draws,
    input checks, item8_positions_describe.csv.
    """
    folder = "referee_f2"
    if not ctx.res(folder).exists():
        return
    psn = REF_PS.format("REFEREE_RESPONSE_F2.md")
    seen = {}
    n_dup = {}
    n_items = {"1-7": 0, "8": 0}
    n_pv = {"1-7": 0, "8": 0}

    def _key(coef, n, nw):
        return (round(num(coef), 10), _int(n), round(nw, 10) if np.isfinite(nw) else "nan")

    def _add_group(g, rel, item, hyp, samp, est, n, pcol="p", side=SIDE_TWO, stat_name="coefficient", alt_extra=None,
                   skey="", dedup=True):
        p, m_main, rob, others = _f2_pick(g, pcol)
        n_pv[item] += int(g[pcol].notna().sum())
        if not np.isfinite(p):
            return None
        if dedup:
            k = _key(est, n, _f2_nw(g, pcol))
            if k in seen:
                n_dup[(rel, seen[k])] = n_dup.get((rel, seen[k]), 0) + 1
                return None
            seen[k] = rel
        row = ctx.add("REFF2", hyp, "Nifty return over the last 30 minutes (r_last) and its reversal slope", samp, p,
                      role="exploratory", pre_specified=PS_NO, pre_spec_note=psn, source_file=rel, source_key=skey,
                      sidedness=side, estimate=est, statistic_name=stat_name,
                      p_method=(f"largest p among the methods that respect the persistence of the state ({', '.join(rob)}); "
                                f"here {m_main}; " + ("not directional" if side == SIDE_NONDIR else "two-sided")),
                      alt_p=alt(**{**others, **(alt_extra or {})}))
        if row is not None:
            n_items[item] += 1
        return row

    rel = f"{folder}/item1_unconditional_slope.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for (test, samp), g in d.groupby(["test", "sample"], sort=False):
            mom = g["p_one_sided_momentum"].dropna()
            _add_group(g, rel, "1-7", f"Referee item 1: {clean(test)}; tested against zero",
                       f"{F2_SAMPLE.get(clean(samp), clean(samp))} ({_int(g['n'].iloc[0])} days)", num(g["coef"].iloc[0]),
                       g["n"].iloc[0], pcol="p_two_sided", skey=key("item1", test, samp),
                       alt_extra={"p_one_sided_momentum_nw5": mom.iloc[0]} if len(mom) else None, dedup=False)
    for fname, item_lab, col in (("item3_f2_inference.csv", "item 3 (inference for a persistent state)", "sample"),
                                 ("item4_other_rules.csv", "item 4 (other categories as the dealer)", "sample"),
                                 ("item5_alternatives.csv", "item 5 (alternative specifications)", "case")):
        rel = f"{folder}/{fname}"
        if not ctx.has(rel):
            continue
        d = ctx.read(rel)
        n_ne = 0
        for gk, g in d.groupby(["rule", "spec", "outcome", col, "term"], sort=False):
            rule, spec, outc, sc, term = (clean(x) for x in gk)
            if g["p"].notna().sum() == 0:
                n_ne += 1
                continue
            t_txt = "" if term == "N_x_r_rod" else f", term {term}"
            hyp = (f"F2, referee {item_lab}: after a negative-GEX evening ({F2_RULE.get(rule, rule)} as the dealer) the "
                   f"end-of-day reversal slope differs ({spec} specification, outcome {outc}{t_txt}"
                   + (f"; {sc}" if col == "case" else "") + ")")
            _add_group(g, rel, "1-7", hyp,
                       f"{F2_SAMPLE.get(sc, sc)} ({_int(g['n'].iloc[0])} days, "
                       f"{_int(g['n_neg'].iloc[0])} negative)",
                       num(g["coef"].iloc[0]), g["n"].iloc[0], skey=key(fname, rule, spec, outc, sc, term))
        ctx.skip("REFF2", rel, "groups marked 'not estimable'", n_ne, "no p-value (fewer than 5 days in one sign state, or "
                 "net GEX below zero on every evening)")
    rel = f"{folder}/item6_angle_equality.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for samp, g in d.groupby("sample", sort=False):
            n_pv["1-7"] += 3 * len(g)
            blocks = [int(num(b)) for b in g["block_days"]]
            pb = [num(x) for x in g["p_equal_basic_bootstrap"]]
            i = int(np.nanargmax(pb))
            r = g.iloc[i]
            row = ctx.add("REFF2", "Referee item 6: the response angles of the two Pro states are equal (basic bootstrap of angle a minus angle b)",
                          "direction of the end-of-day response (angle, degrees)",
                          f"{clean(samp)} ({_int(r['n_days_a'])} and {_int(r['n_days_b'])} days)", pb[i], role="exploratory",
                          pre_specified=PS_NO, pre_spec_note=psn, source_file=rel, source_key=key("item6", samp, "angle"),
                          estimate=r["diff_a_minus_b"], statistic_name="angle a minus angle b (degrees)",
                          p_method=f"moving-block bootstrap, larger p of the two block lengths (here {blocks[i]} days), two-sided",
                          alt_p=alt(**{f"p_block_{b}": p for b, p in zip(blocks, pb) if b != blocks[i]}))
            n_items["1-7"] += row is not None
            cand = [(f"chi2_{df}_block_{b}", num(x[f"md_p_chi2_{df}"]), num(x["md_J"]))
                    for b, (_, x) in zip(blocks, g.iterrows()) for df in (1, 2)]
            nm, pj, jv = max(cand, key=lambda c: c[1])
            row = ctx.add("REFF2", "Referee item 6: one common response angle fits both Pro states (minimum-distance J test)",
                          "direction of the end-of-day response", clean(samp), pj, role="exploratory", pre_specified=PS_NO,
                          pre_spec_note=psn, source_file=rel, source_key=key("item6", samp, "md_J"), sidedness=SIDE_NONDIR,
                          statistic_name="J", statistic=jv,
                          p_method=f"minimum-distance J, largest p over 1 or 2 degrees of freedom and block lengths (here {nm})",
                          alt_p=alt(**{c[0]: c[1] for c in cand if c[0] != nm}))
            n_items["1-7"] += row is not None
            same = all(_same_num(num(x["wald_theta_a_zero_boot"]), num(x["md_J"]), rel=1e-5) for _, x in g.iterrows())
            if same:
                ctx.skip("REFF2", rel, f"Wald test of theta_a = 0, {clean(samp)}", len(g),
                         "same statistic as the J test in this sample (counted once there)")
                continue
            pw = [num(x) for x in g["wald_theta_a_zero_boot_p"]]
            i = int(np.nanargmax(pw))
            row = ctx.add("REFF2", "Referee item 6: the response of the negative-GEX state is zero (Wald test of theta_a = 0)",
                          "end-of-day response coefficients", clean(samp), pw[i], role="exploratory", pre_specified=PS_NO,
                          pre_spec_note=psn, source_file=rel, source_key=key("item6", samp, "wald_theta_a"), sidedness=SIDE_NONDIR,
                          statistic_name="Wald", statistic=g["wald_theta_a_zero_boot"].iloc[i],
                          p_method=f"bootstrap Wald, larger p of the two block lengths (here {blocks[i]} days)",
                          alt_p=alt(**{f"p_block_{b}": p for b, p in zip(blocks, pw) if b != blocks[i]}))
            n_items["1-7"] += row is not None
    rel = f"{folder}/item7_minute_autocorrelation.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["p_two_sided"].notna()].iterrows():
            n_pv["1-7"] += 1
            row = ctx.add("REFF2", f"Referee item 7: first-order autocorrelation of 1-minute returns in {clean(r['window'])} differs "
                          f"between expiry and other days ({clean(r['grouping'])}: {clean(r['group'])})",
                          "first-order autocorrelation of 1-minute returns", clean(r["group"]), r["p_two_sided"],
                          role="exploratory", pre_specified=PS_NO, pre_spec_note=psn, source_file=rel,
                          source_key=key("item7", r["window"], r["grouping"], r["group"]), estimate=r["ac1_pooled"],
                          statistic_name="difference in pooled autocorrelation", p_method="day bootstrap, two-sided")
            n_items["1-7"] += row is not None
        ctx.skip("REFF2", rel, "per-group autocorrelations", int(d["p_two_sided"].isna().sum()), "descriptive, no p-value")
    rel = f"{folder}/item7_junction.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        ccol = next(c for c in d.columns if c.startswith("corr"))
        for _, r in d.iterrows():
            n_pv["1-7"] += int(np.isfinite(num(r["p"])))
            row = ctx.add("REFF2", "Referee item 7: returns of the 14:58 to 14:59 and 14:59 to 15:00 bars are correlated",
                          "correlation of two consecutive 1-minute returns", f"{clean(r['days'])} ({_int(r['n'])} days)", r["p"],
                          role="exploratory", pre_specified=PS_NO, pre_spec_note=psn, source_file=rel,
                          source_key=key("item7_junction", r["days"]), estimate=r[ccol], statistic_name="correlation",
                          p_method="correlation test, two-sided")
            n_items["1-7"] += row is not None
    rel = f"{folder}/item8_positions_f2.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for gk, g in d.groupby(["category", "position", "form", "sample"], sort=False):
            cat, pos, form, samp = (clean(x) for x in gk)
            _add_group(g, rel, "8", f"F2, referee item 8: with the {cat} {pos.replace('_', ' ')} position as the state ({form}), the "
                       f"end-of-day reversal slope differs (F2's direction: {'negative' if form == 'dummy' else 'positive'} coefficient)",
                       f"{F2_SAMPLE.get(samp, samp)} ({_int(g['n'].iloc[0])} days)", num(g["coef"].iloc[0]), g["n"].iloc[0],
                       skey=key("item8_positions", cat, pos, form, samp))
    rel = f"{folder}/item8_horse_race.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for gk, g in d.groupby(["horse_race", "position", "form"], sort=False):
            hr, pos, form = (clean(x) for x in gk)
            joint = form == "joint"
            hyp = (f"Horse race '{hr}' (Pro, full sample): the raw position states jointly change the reversal slope (Wald)" if joint else
                   f"Horse race '{hr}' (Pro, full sample): the {pos.replace('_', ' ')} state ({form}) changes the reversal slope, "
                   "the other states held fixed")
            _add_group(g, rel, "8", hyp, f"full sample ({_int(g['n'].iloc[0])} days)",
                       float("nan") if joint else num(g["coef"].iloc[0]), g["n"].iloc[0], side=SIDE_NONDIR if joint else SIDE_TWO,
                       stat_name="Wald" if joint else "coefficient", skey=key("item8_horse_race", hr, pos, form), dedup=False)
    for (rel, first), n in n_dup.items():
        ctx.skip("REFF2", rel, "estimates already counted from another file", n,
                 f"same coefficient, days and Newey-West p as a row of {first} (counted once there)")
    ctx.ref_f2_counts = (dict(n_pv), dict(n_items))
    ctx.skip("REFF2", f"{folder}/item1 to item7", "p-values reported for items 1 to 7 (REFEREE_RESPONSE_F2.md says 598)",
             n_pv["1-7"], f"counted as {n_items['1-7']} tests: the rest are other inference methods for the same estimate "
             "(in alt_p) or the same estimate in another file")
    ctx.skip("REFF2", f"{folder}/item8_*", "p-values reported for item 8 (REFEREE_RESPONSE_F2.md says 381)", n_pv["8"],
             f"counted as {n_items['8']} tests: the rest are other inference methods for the same estimate (in alt_p) "
             "or the same estimate as an item 3 or item 4 row")
    _consume_rest(ctx, "REFF2", folder, {
        r"item2_.*\.csv": "shares, runs and autocorrelations of the Pro sign (descriptive, no p-value)",
        r"item6_angle_bootstrap_draws\.csv\.gz": "bootstrap draws behind the item 6 tests",
        r"item8_positions_describe\.csv": "description of the positions (no p-value)",
    }, "input checks or run information; no p-value")


REFFC_MODEL = {
    "mcsgarch_realtime": "mcsGARCH with a strictly real-time EWMA fallback",
    "mcsgarch_vix": "mcsGARCH with India VIX as its daily level",
    "mcsgarchx_vix": "mcsGARCH-X, daily EGARCH with log India VIX",
    "mcsgarchx_vix_iv": "mcsGARCH-X, daily EGARCH with log India VIX and the log straddle-implied variance",
}


def _v2_lookup(ctx: Ctx, folder="pipeline_nse_v2"):
    """(target, loss, subset, comparison) -> one-sided DM p of the current contest's tests_*.csv (QLIKE and MSE)."""
    out = {}
    for f in sorted(ctx.res(folder).glob("tests_*.csv")):
        d = pd.read_csv(f)
        if not V2_TESTS_COLS.issubset(d.columns):
            continue
        t = f.stem.replace("tests_", "")
        for loss, sub, comp, p in zip(d["loss"], d["subset"], d["comparison"], d["p"]):
            out[(t, clean(loss).lower(), clean(sub), clean(comp))] = num(p)
    return out


def _reffc_hyp_name(bm, bmm):
    if bm == "best_matched_mcsgarch":
        return f"the best information-matched mcsGARCH of the subset, here {REFFC_MODEL.get(bmm, v2_model(bmm))}"
    return REFFC_MODEL.get(bmm) or v2_model(bmm)


def adapt_ref_forecast(ctx: Ctx):
    """Referee response on the forecasting contest (docs/REFEREE_RESPONSE_FORECAST.md). All post hoc: exploratory.

    spa_reality_check.csv: target, subset, mean_block (5, 10, 20), p_spa_consistent (Hansen's SPA: does any
      challenger beat mcsGARCH; main p at block 10, the study's choice), p_spa_lower, p_spa_upper,
      p_reality_check and the other block lengths (alternatives). A max-type one-sided p that was not pre-stated
      is doubled as min(1, 2p) (a p near 1 does not mean a challenger is worse).
    fair_benchmarks.csv: each information-matched benchmark against mcsGARCH (p_benchmark_beats_mcsgarch) and the
      hybrid against each benchmark (p_hybrid_beats_benchmark, dm_stat); one-sided, doubled to two-sided.
    holdout_lightgbm_vs_hybrid.csv: p_two_sided (main), p_lightgbm_better_one_sided (alternative).
    gw_tests.csv: Giacomini-White tests, family main (42 tests; chi-square, not directional;
      p_holm_main_family cross-checked); the 12 Newey-West versions of C1 and C2 are alternatives of those rows.
    Rows that repeat a test of the current contest (R1V2: same models, days, loss and p) are not counted again:
      holdout_table.csv, mse_ranking.csv, refit_envelope.csv, the DM p of spa_adjusted_pvalues_by_model.csv and
      seed_adjusted_dm.csv, and the hybrid-against-mcsGARCH rows of fair_benchmarks.csv; any row that does not
      match is counted. Rows with the same p and estimate within this study are counted once (the "best matched"
      benchmark is one of the named ones).
    Not counted: step-down and single-step max-t adjusted p-values (adjustments), noise-adjusted DM p-values
      (alternatives of R1V2 tests), MCS p-values, GW regression coefficients and group margins, mcsGARCH-X
      parameters and likelihood ratios (no p-value), forecasts, seed summaries and ranks.
    """
    folder = "referee_forecast"
    if not ctx.res(folder).exists():
        return
    psn = REF_PS.format("REFEREE_RESPONSE_FORECAST.md")
    v2 = _v2_lookup(ctx)
    rows, dup = [], {}

    def keep(row):
        if row is not None:
            rows.append(ctx.rows.pop())

    def vs_ref(rel, t, loss, subset, ref, model, p, pct, n, skey, desc_a=None, desc_b=None, stat=float("nan")):
        """'model' against 'ref' with a one-sided p that model has the lower loss; pct: model's loss relative to ref (%)."""
        if not np.isfinite(num(p)):
            return
        if _same_num(v2.get((t, loss, subset, f"{ref} vs {model}"), float("nan")), num(p)):
            dup[rel] = dup.get(rel, 0) + 1
            return
        _, b, hyp = _v2_vs_mcsgarch(f"{ref} vs {model}", num(pct), better_if_positive=False, desc_a=desc_a, desc_b=desc_b)
        keep(_v2_add(ctx, "REFFC", hyp, f"{V2_LOSS.get(loss, loss)} loss of forecasts of {TARGETS.get(t, t)}",
                     f"{v2_subset(subset)} (n = {_int(n)} days), horizon {t}", p, SIDE_CONVERTED, role="exploratory",
                     pre_specified=PS_NO, pre_spec_note=psn, source_file=rel, source_key=skey, sign_rule=v2_rule(b),
                     estimate=pct, statistic_name=f"loss relative to {'mcsGARCH' if ref == 'mcsgarch' else ref} (%)",
                     statistic=stat, p_method=V2_DM.format("second model better")))

    rel = f"{folder}/spa_reality_check.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        n_alt = 0
        for (t, s), g in d.groupby(["target", "subset"], sort=False):
            m = g[g["mean_block"] == 10]
            r = m.iloc[0] if len(m) else g.iloc[len(g) // 2]
            altd = {"p_spa_lower": r["p_spa_lower"], "p_spa_upper": r["p_spa_upper"], "p_reality_check": r["p_reality_check"]}
            for _, o in g.iterrows():
                if num(o["mean_block"]) != num(r["mean_block"]):
                    b = _int(o["mean_block"])
                    altd[f"p_spa_block_{b}"] = o["p_spa_consistent"]
                    altd[f"p_reality_check_block_{b}"] = o["p_reality_check"]
                    n_alt += 1
            keep(ctx.add("REFFC", f"At least one of the {_int(r['n_challengers'])} challengers beats mcsGARCH (Hansen SPA, "
                         f"allowing for the search; best by t: {clean(r['best_by_t'])})", f"QLIKE loss of forecasts of {TARGETS.get(t, t)}",
                         f"{v2_subset(s)} (n = {_int(r['days'])} days), horizon {t}", r["p_spa_consistent"], role="exploratory",
                         pre_specified=PS_NO, pre_spec_note=psn, source_file=rel, source_key=key("spa_reality_check.csv", t, s),
                         sidedness=SIDE_DOUBLED, estimate=r["best_by_t_pct_vs_mcsgarch"],
                         statistic_name="SPA statistic (largest studentized loss difference)", statistic=r["T_spa"],
                         p_method=(f"Hansen SPA, consistent p, stationary bootstrap (mean block {_int(r['mean_block'])} days, "
                                   f"{_int(r['reps'])} replications)"), alt_p=alt(**altd)))
        ctx.skip("REFFC", rel, "SPA and Reality Check with blocks of 5 and 20 days", n_alt,
                 "the same test with another bootstrap block length (in alt_p of the block-10 row)")
    rel = f"{folder}/fair_benchmarks.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            t, s, bm, bmm = clean(r["target"]), clean(r["subset"]), clean(r["benchmark"]), clean(r["benchmark_model"])
            name = _reffc_hyp_name(bm, bmm)
            if bmm != "mcsgarch":
                vs_ref(rel, t, "qlike", s, "mcsgarch", bmm, r["p_benchmark_beats_mcsgarch"], r["benchmark_pct_vs_mcsgarch"],
                       r["days"], key("fair_benchmarks.csv", t, s, bm, "vs mcsgarch"), desc_b=name)
            vs_ref(rel, t, "qlike", s, bmm, "hybrid_base", r["p_hybrid_beats_benchmark"], r["hybrid_pct_vs_benchmark"],
                   r["days"], key("fair_benchmarks.csv", t, s, bm, "hybrid"), desc_a=name, stat=r["dm_stat"])
    rel = f"{folder}/holdout_table.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            vs_ref(rel, clean(r["target"]), "qlike", clean(r["period"]), "mcsgarch", clean(r["model"]), r["p_beats_mcsgarch"],
                   r["pct_vs_mcsgarch"], r["days"], key("holdout_table.csv", r["target"], r["period"], r["model"]))
    rel = f"{folder}/mse_ranking.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            vs_ref(rel, clean(r["target"]), "mse", clean(r["subset"]), "mcsgarch", clean(r["model"]), r["p_mse_beats_mcsgarch"],
                   r["mse_pct_vs_mcsgarch"], float("nan"), key("mse_ranking.csv", r["target"], r["subset"], r["model"]))
    rel = f"{folder}/refit_envelope.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        models = [c[:-2] for c in d.columns if c.endswith("_p") and c != "variant_p_max" and f"{c[:-2]}_pct" in d.columns]
        for _, r in d.iterrows():
            for m_ in models:
                vs_ref(rel, clean(r["target"]), "qlike", clean(r["subset"]), "mcsgarch", m_, r[f"{m_}_p"], r[f"{m_}_pct"],
                       r["days"], key("refit_envelope.csv", r["target"], r["subset"], m_))
        ctx.skip("REFFC", rel, "variant_p_max", int(d["variant_p_max"].notna().sum()), "the largest of the six p-values in the row, not a test")
    rel = f"{folder}/spa_adjusted_pvalues_by_model.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            vs_ref(rel, clean(r["target"]), "qlike", clean(r["subset"]), "mcsgarch", clean(r["model"]), r["p_dm_one_sided"],
                   r["pct_vs_mcsgarch"], r["days"], key("spa_adjusted_pvalues_by_model.csv", r["target"], r["subset"], r["model"]))
        ctx.skip("REFFC", rel, "single-step and step-down max-t adjusted p-values", int(d[["p_maxt_single_step", "p_maxt_stepdown"]].notna().sum().sum()),
                 "multiple-testing adjustments of the DM tests (Romano and Wolf), not new tests")
    rel = f"{folder}/seed_adjusted_dm.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            vs_ref(rel, clean(r["target"]), "qlike", clean(r["subset"]), "mcsgarch", "hybrid_base", r["dm_p"],
                   r["hybrid_pct_vs_mcsgarch"], r["days"], key("seed_adjusted_dm.csv", r["target"], r["subset"]))
        pc = [c for c in d.columns if c.startswith("p[")]
        ctx.skip("REFFC", rel, "noise-adjusted DM p-values", int(d[pc].notna().sum().sum()),
                 "the same estimate (hybrid against mcsGARCH, an R1V2 test) with seed noise added to the DM variance: alternatives")
    rel = f"{folder}/holdout_lightgbm_vs_hybrid.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            t, per, lg = clean(r["target"]), clean(r["period"]), clean(r["lightgbm"])
            pct = num(r["lightgbm_pct_vs_hybrid"])
            keep(ctx.add("REFFC", f"{lg} and the hybrid differ in forecast loss (observed: {lg} {_v2_obs(pct, False, 'better', 'worse')}); "
                         f"{lg} = {v2_model(lg)}", f"QLIKE loss of forecasts of {TARGETS.get(t, t)}",
                         f"{v2_subset(per)} (n = {_int(r['days'])} days), horizon {t}", r["p_two_sided"], role="exploratory",
                         pre_specified=PS_NO, pre_spec_note=psn, source_file=rel, source_key=key("holdout_lightgbm_vs_hybrid.csv", t, per, lg),
                         estimate=pct, statistic_name="loss relative to the hybrid (%)", statistic=r["dm_stat"],
                         p_method="Diebold-Mariano (HLN), two-sided", alt_p=alt(p_lightgbm_better_one_sided=r["p_lightgbm_better_one_sided"])))
    rel = f"{folder}/gw_tests.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        main = d[d["family"] == "main"]
        nw = d[d["family"] != "main"]
        rec = ctx.check("REFFC", "Giacomini-White main family (Holm over all main-family tests)", rel, "holm", main["p"], main["p_holm_main_family"])
        for i, (_, r) in enumerate(main.iterrows()):
            t, bm, test = clean(r["target"]), clean(r["benchmark_model"]), clean(r["test"])
            w = nw[(nw["target"] == r["target"]) & (nw["benchmark"] == r["benchmark"]) & (nw["test"].map(clean) == f"{test} (Newey-West)")]
            bname = "mcsGARCH" if bm == "mcsgarch" else bm
            if test == "unconditional":
                hyp = f"The hybrid's mean QLIKE advantage over {bname} is zero (Giacomini-White unconditional)"
            elif test.startswith("conditional"):
                hyp = f"The hybrid's advantage over {bname} cannot be predicted from {clean(r['instruments'])} (Giacomini-White {test})"
            else:
                hyp = f"Giacomini-White {test}: the hybrid's advantage over {bname} does not vary with {clean(r['instruments'])}"
            row = ctx.add("REFFC", hyp + f" ({clean(r['benchmark'])})", f"daily QLIKE difference, forecasts of {TARGETS.get(t, t)}",
                          f"all test days (n = {_int(r['n'])}), horizon {t}", r["p"], role="exploratory", pre_specified=PS_NO,
                          pre_spec_note=psn, source_file=rel, source_key=key("gw_tests.csv", t, r["benchmark"], test),
                          sidedness=SIDE_NONDIR, estimate=r["mean_d_points"], statistic_name=f"Wald chi-square ({_int(r['df'])} df)",
                          statistic=r["stat"], p_method="Giacomini-White Wald test",
                          alt_p=alt(p_newey_west=w["p"].iloc[0]) if len(w) else "",
                          p_study_adjusted=r["p_holm_main_family"], study_adjust_method="Holm over the main family")
            _set(row, "p_study_adjusted_recomputed", rec[i])
            keep(row)
        ctx.skip("REFFC", rel, "Newey-West versions of C1 and C2", len(nw),
                 "another variance for the same statistic (in alt_p of the C1 and C2 rows); the study counts 54 = 42 + 12")
    kept, dropped = dedup_exact(rows)
    ctx.rows.extend(kept)
    for rel_, n in dup.items():
        ctx.skip("REFFC", rel_, "rows that repeat a test of the current contest", n,
                 "same models, days, loss and p as an R1V2 row of results/pipeline_nse_v2/tests_*.csv (counted once there)")
    ctx.skip("REFFC", f"{folder}/", "rows repeating another row of this study (same p and estimate)", len(dropped),
             "the best-matched benchmark is one of the named benchmarks (counted once)")
    _consume_rest(ctx, "REFFC", folder, {
        r"mcs_summary\.csv": "Model Confidence Set p-values (set-level, already adjusted)",
        r"gw_(regression_coefficients|group_margins)\.csv": "coefficients and margins behind the GW tests (descriptive)",
        r"mcsgarchx_.*": "mcsGARCH-X forecasts, parameters and likelihood ratios (no p-value) and run information",
    }, "seed summaries, run costs or ranks; no p-value")


REFEXP_ALT = ("boot_p", "welch_p", "mwu_p", "ri_p", "nonoverlap_p", "prediction_t_p", "wild_cluster_p", "placebo_p")
REFEXP_TERM = {"own_thu_to_a": "own expiry on Thursday, before event a", "own_wed_a_to_c": "own expiry on Wednesday, events a to c",
               "own_thu_to_d": "own expiry on Thursday, before event d", "own_tue_from_d": "own expiry on Tuesday, from event d"}


class _Known:
    """Index of counted ledger rows by (p, estimate), to recognise a test that another file already reports."""

    def __init__(self, rows, studies):
        self.idx = {}
        for r in rows:
            p = num(r["p_reported"])
            if r["study"] in studies and np.isfinite(p):
                self.idx.setdefault(f"{p:.6g}", []).append((p, num(r["estimate"]), r["study"]))

    def find(self, p, est):
        p, est = num(p), num(est)
        for pp, ee, s in self.idx.get(f"{p:.6g}", []):
            if _same_num(pp, p) and (_same_num(ee, est) or not (np.isfinite(ee) or np.isfinite(est))):
                return s
        return None


def _refexp_meas(m):
    m = clean(m)
    if m in MEASURES:
        return MEASURES[m]
    if m.startswith("last30_"):
        return f"last-30-minute variance ({m[7:]} estimator)"
    return m


def adapt_ref_expiry(ctx: Ctx):
    """Referee response on the expiry premia and the event study (docs/REFEREE_RESPONSE_EXPIRY.md). All post hoc:
    exploratory. Friday rows say "Negative control (Friday)" in the hypothesis.

    Event-style files (event_tests_clean_baseline, negative_control_rows, placebo_inference_all_rows,
    placebo_inference_headline_rows, pretrends_and_comparisons, two_day_raw_differences, switch_comparison_a_vs_d,
    noise_robust_event_tests): main p hac_p (as in EVENT), estimate diff; boot, Welch, Mann-Whitney, placebo-date,
    non-overlap, prediction-interval and wild cluster p-values are alternatives. The clean-baseline columns of
    event_tests_clean_baseline (hac_p_clean, diff_clean, ri_p_clean) are new tests where estimable. The
    trend-adjusted change (trend_adjusted_change, trend_adjusted_p, HAC) is a new estimate and is counted.
    Premium files (event_premium_by_regime_clean_baseline, noise_robust_premium_by_regime): hac_p and mean,
    wilcoxon_p alternative. R2-style files (r2_clean_baseline, noise_robust_r2: wilcoxon_p and mean_rel;
    r2_tue_vs_thu_by_baseline: mannwhitney_p, with a Bonferroni over 18 cross-checked).
    autocorrelation_1min_returns.csv: welch_p_own_minus_this_group. pooled_all_index_regression.csv: p of the own
    expiry terms; the other indices' expiry dummies are controls.
    A row whose p and estimate equal a counted EVENT, R2, SENSEX or AUCTION row is that test again and is not
    counted; rows repeating each other within this study are counted once. pretrend_p is a diagnostic.
    Not counted: calendars, activity tables, shares of rejections (negative_control_recomputed,
    placebo_inference_summary), the noise-estimator simulation (SYNTHETIC) and the daily data.
    """
    folder = "referee_expiry"
    if not ctx.res(folder).exists():
        return
    psn = REF_PS.format("REFEREE_RESPONSE_EXPIRY.md")
    known = _Known(ctx.rows, {"EVENT", "R2", "SENSEX", "AUCTION"})
    rows, dups, n_na, n_alt_dup, n_pre = [], {}, {}, {}, {}

    def add(rel, hyp, outcome, samp, p, est, skey, stat_name, pm, altp="", **kw):
        if not np.isfinite(num(p)):
            n_na[rel] = n_na.get(rel, 0) + 1
            return "na"
        s = known.find(p, est)
        if s:
            dups[(rel, s)] = dups.get((rel, s), 0) + 1
            return "dup"
        n0 = len(ctx.rows)
        ctx.add("REFEXP", hyp, outcome, samp, p, role="exploratory", pre_specified=PS_NO, pre_spec_note=psn,
                source_file=rel, source_key=skey, estimate=est, statistic_name=stat_name, p_method=pm, alt_p=altp, **kw)
        if len(ctx.rows) > n0:
            rows.append(ctx.rows.pop())
            return rows[-1]
        return None

    def ev_text(r, what):
        spec, comp = clean(r.get("spec_id", "")), clean(r.get("comparison", "")) or clean(r.get("series", ""))
        fri = "_Fri" in spec or "negative control" in (comp + " " + clean(r.get("comparison_type", ""))).lower()
        comp = re.sub(r"^Negative control:\s*", "", comp)
        return ("Negative control (Friday): " if fri else "") + f"Event {clean(r['event'])}: {comp} {what} in {_refexp_meas(r['measure'])}"

    def alts(r, cols=REFEXP_ALT):
        return alt(**{c: r[c] for c in cols if c in r.index})

    def n_alts(r, cols=REFEXP_ALT):
        return sum(int(np.isfinite(num(r[c]))) for c in cols if c in r.index)

    def ev_file(fname, extra=""):
        rel = f"{folder}/{fname}"
        if not ctx.has(rel):
            return
        d = ctx.read(rel)
        for _, r in d.iterrows():
            win = clean(r.get("window", ""))
            var = clean(r.get("variant", "")) or clean(r.get("series", "")) if fname != "two_day_raw_differences.csv" else ""
            tail = f" ({var})" if var and var not in ("published Friday rows",) else ""
            samp = f"{win} ({clean(r.get('n_pre'))} vs {clean(r.get('n_post'))} weeks)"
            sk = key(fname, r.get("spec_id", r.get("series", "")), r["measure"], win, r.get("variant", ""))
            res = add(rel, ev_text(r, "changed") + tail + extra, f"week-relative {clean(r['measure'])}", samp, r["hac_p"],
                      r["diff"], sk, "change (log points)", "HAC (Newey-West), two-sided", alts(r))
            if res == "dup":
                n_alt_dup[rel] = n_alt_dup.get(rel, 0) + n_alts(r)
            if "trend_adjusted_p" in r.index:
                add(rel, ev_text(r, "shifted in level after a linear trend through both sides") + tail + extra,
                    f"week-relative {clean(r['measure'])}", samp, r["trend_adjusted_p"], r["trend_adjusted_change"],
                    sk + "|trend", "trend-adjusted level shift (log points)", "HAC, two-sided")
            if "pretrend_p" in r.index:
                n_pre[rel] = n_pre.get(rel, 0) + int(np.isfinite(num(r["pretrend_p"])))

    rel = f"{folder}/event_tests_clean_baseline.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            win, meas = clean(r["window"]), clean(r["measure"])
            sk = key("event_tests_clean_baseline.csv", r["spec_id"], meas, win)
            res = add(rel, ev_text(r, "changed") + " (published baseline)", f"week-relative {meas}",
                      f"{win} ({clean(r['n_pre'])} vs {clean(r['n_post'])} weeks)", r["hac_p"], r["diff"], sk + "|published",
                      "change (log points)", "HAC (Newey-West), two-sided", alt(placebo_ri_p=r["ri_p"]))
            if res == "dup":
                n_alt_dup[rel] = n_alt_dup.get(rel, 0) + int(np.isfinite(num(r["ri_p"])))
            if str(r["estimable_clean"]).lower() in ("true", "1", "1.0"):
                add(rel, ev_text(r, "changed") + " (baseline: days with no expiry of any index)", f"week-relative {meas}, clean baseline",
                    f"{win} ({clean(r['n_pre_clean'])} vs {clean(r['n_post_clean'])} weeks)", r["hac_p_clean"], r["diff_clean"],
                    sk + "|clean", "change (log points)", "HAC (Newey-West), two-sided", alt(placebo_ri_p=r["ri_p_clean"]))
            n_pre[rel] = n_pre.get(rel, 0) + int(np.isfinite(num(r["pretrend_p"]))) + int(np.isfinite(num(r["pretrend_p_clean"])))
        ctx.skip("REFEXP", rel, "clean-baseline columns of rows that are not estimable", int((d["estimable_clean"].astype(str).str.lower() != "true").sum()),
                 "no clean baseline in the window (no p-value used)")
    for fname in ("negative_control_rows.csv", "placebo_inference_all_rows.csv", "placebo_inference_headline_rows.csv",
                  "pretrends_and_comparisons.csv", "two_day_raw_differences.csv", "noise_robust_event_tests.csv"):
        ev_file(fname, " (raw two-day difference, no baseline)" if fname == "two_day_raw_differences.csv" else "")
    rel = f"{folder}/switch_comparison_a_vs_d.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            spec, meas, ver = clean(r["spec_id"]), clean(r["measure"]), clean(r["version"])
            res = add(rel, f"Event {spec[0]}: the expiry premium followed the switch ({spec}: new expiry weekday minus old) in "
                      f"{_refexp_meas(meas)}; version: {ver}", f"week-relative {meas}", f"{clean(r['weeks'])} weeks", r["hac_p"],
                      r["diff"], key("switch_comparison_a_vs_d.csv", spec, meas, ver), "change (log points)",
                      "HAC (Newey-West), two-sided", alts(r))
            if res == "dup":
                n_alt_dup[rel] = n_alt_dup.get(rel, 0) + n_alts(r)
            n_pre[rel] = n_pre.get(rel, 0) + int(np.isfinite(num(r["pretrend_p"])))
    for fname, has_base in (("event_premium_by_regime_clean_baseline.csv", True), ("noise_robust_premium_by_regime.csv", False)):
        rel = f"{folder}/{fname}"
        if not ctx.has(rel):
            continue
        d = ctx.read(rel)
        for _, r in d.iterrows():
            tgt, meas = clean(r["target"]), clean(r["measure"])
            base = clean(r["baseline"]) if has_base else ""
            add(rel, f"{r['index']} {'own expiry day' if tgt == 'own' else tgt} has a premium in {_refexp_meas(meas)} over the rest "
                f"of the week ({clean(r['regime'])}" + (f"; baseline: {base})" if base else ")"), f"week-relative {meas}",
                f"{clean(r['regime'])} ({_int(r['n_weeks'])} weeks)", r["hac_p"], r["mean"],
                key(fname, r["index"], tgt, meas, r["regime"], base), "mean week-relative log value",
                "HAC (Newey-West), two-sided", alt(wilcoxon_p=r["wilcoxon_p"]))
    for fname in ("r2_clean_baseline.csv", "noise_robust_r2.csv"):
        rel = f"{folder}/{fname}"
        if not ctx.has(rel):
            continue
        d = ctx.read(rel)
        for _, r in d.iterrows():
            meas = clean(r["measure"])
            add(rel, f"Expiry days have a different {_refexp_meas(meas)} from the baseline days of their week "
                f"({clean(r['period'])}; baseline: {clean(r['baseline'])})", f"log({meas} on expiry day / baseline days)",
                f"{clean(r['period'])} ({_int(r['weeks'])} weeks)", r["wilcoxon_p"], r["mean_rel"],
                key(fname, r["baseline"], meas, r["period"]), "mean log ratio", "Wilcoxon signed-rank, two-sided")
    rel = f"{folder}/r2_tue_vs_thu_by_baseline.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        if "bonferroni_p_18_tests" in d.columns:
            a = np.minimum(1.0, 18.0 * pd.to_numeric(d["mannwhitney_p"], errors="coerce").to_numpy(float))
            b = pd.to_numeric(d["bonferroni_p_18_tests"], errors="coerce").to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            diff = np.abs(a[both] - b[both])
            n_mis = int((diff > 1e-6).sum())
            ctx.checks.append(dict(study="REFEXP", family="Tuesday against Thursday regime (Bonferroni over 18, as the study states)",
                                   source_file=f"results/{rel}", method="bonferroni (m = 18)", m_tests=18, n_compared=int(both.sum()),
                                   n_mismatch=n_mis, max_abs_diff=float(diff.max()) if diff.size else float("nan"),
                                   result="match" if n_mis == 0 and both.sum() else ("MISMATCH" if n_mis else "nothing to compare"),
                                   note=f"the file lists {len(d)} of the 18 tests in the study's family"))
        for i, (_, r) in enumerate(d.iterrows()):
            meas = clean(r["measure"])
            row = add(rel, f"The expiry-day premium in {_refexp_meas(meas)} differs between the Tuesday and Thursday single-weekly "
                      f"regimes (baseline: {clean(r['baseline'])})", f"log({meas} on expiry day / baseline days)",
                      f"{clean(r['weeks_tue'])} Tuesday and {clean(r['weeks_thu'])} Thursday weeks", r["mannwhitney_p"],
                      r["tue_minus_thu"], key("r2_tue_vs_thu_by_baseline.csv", r["baseline"], meas), "difference in mean log ratio",
                      "Mann-Whitney, two-sided", p_study_adjusted=r.get("bonferroni_p_18_tests", float("nan")),
                      study_adjust_method="Bonferroni over 18 tests (the study's family)")
            if isinstance(row, dict):
                row["p_study_adjusted_recomputed"] = min(1.0, 18.0 * num(r["mannwhitney_p"]))
    rel = f"{folder}/autocorrelation_1min_returns.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d[d["welch_p_own_minus_this_group"].notna()].iterrows():
            add(rel, f"{r['index']}, {clean(r['window'])}: first-order autocorrelation of 1-minute returns differs between its own "
                f"expiry days and {clean(r['group'])} ({clean(r['regime'])})", "first-order autocorrelation of 1-minute returns",
                f"{clean(r['regime'])}", r["welch_p_own_minus_this_group"], r["own_minus_this_group"],
                key("autocorrelation_1min_returns.csv", r["index"], r["regime"], r["window"], r["group"]),
                "difference in mean autocorrelation", "Welch t, two-sided")
        ctx.skip("REFEXP", rel, "own-expiry rows", int(d["welch_p_own_minus_this_group"].isna().sum()), "reference group, no test")
    rel = f"{folder}/pooled_all_index_regression.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        own = d["term"].astype(str).str.startswith("own_")
        for _, r in d[own].iterrows():
            term, meas = clean(r["term"]), clean(r["measure"])
            add(rel, f"{r['index']}: {REFEXP_TERM.get(term, term)} raises {_refexp_meas(meas)} (pooled day-level regression; {clean(r['spec'])})",
                meas, f"{_int(r['n_days'])} days, {_int(r['n_weeks'])} weeks", r["p"], r["coef"],
                key("pooled_all_index_regression.csv", r["index"], meas, r["spec"], term), "coefficient",
                "OLS with week fixed effects, week-clustered, two-sided")
        ctx.skip("REFEXP", rel, "other indices' expiry dummies", int((~own).sum()), "control coefficients")
    kept, dropped = dedup_exact(rows)
    ctx.rows.extend(kept)
    for (rel_, s), n in dups.items():
        ctx.skip("REFEXP", rel_, f"rows that repeat a test of {s}", n, f"same p and estimate as a counted {s} row (counted once there)")
    for rel_, n in n_alt_dup.items():
        ctx.skip("REFEXP", rel_, "other p-values of those repeated tests", n,
                 "placebo-date, non-overlap, prediction-interval, wild cluster or other p-values of a test counted elsewhere (alternatives)")
    for rel_, n in n_na.items():
        ctx.skip("REFEXP", rel_, "rows without a p-value", n, "not estimable in that window or variant")
    for rel_, n in n_pre.items():
        ctx.skip("REFEXP", rel_, "pre-trend p-values (pretrend_p)", n, "diagnostic of the design, not a hypothesis")
    ctx.skip("REFEXP", f"{folder}/", "rows repeating another row of this study (same p and estimate)", len(dropped), "same test (counted once)")
    _consume_rest(ctx, "REFEXP", folder, {
        r"(negative_control_recomputed|placebo_inference_summary)\.csv": "shares of rejections across tests, not a test",
        r"noise_estimator_simulation\.csv": "simulation (SYNTHETIC)",
        r"noise_robust_daily\.csv\.gz": "daily data behind the noise-robust tests",
        r"summary\.md": "text summary",
    }, "calendar, expiry activity or weekday table; no p-value")


# ----------------------------------------------------------------------------
# P91: the forecasting pipeline on Bank Nifty (same layout as pipeline_nse_v2)
# ----------------------------------------------------------------------------

BNFC_FOLDER = "pipeline_nse_banknifty"
BNFC_OUTCOME = ["tests_*.csv", "ablation_*.csv", "p91/*.csv"]
BNFC_GEX_SETS = ("gross", "us", "participant", "client")
BNFC_REGIME_SPLIT = re.compile(r"^(multi_weekly|single_weekly_thu|single_weekly_tue)_(expiry|non_expiry)$")
BNFC_PS_REPORT = "registered to be reported without a hypothesis (plan Sections 4, 6 and 7); one-sided p doubled to two-sided"
BNFC_PS_EXPL = "labelled exploratory in BANKNIFTY_FORECAST.md (Section 10: not in the plan)"


def _bnfc_h(sk: str):
    """(code, role, hypothesis) of a registered hypothesis row of the Bank Nifty contest, from its source_key."""
    p = sk.split("|")
    if len(p) == 4 and p[1] == "qlike" and p[3] == "mcsgarch vs hybrid_base":
        t = p[0].replace("tests_", "").replace(".csv", "")
        if p[2] == "all" and t in ("next15", "next30"):
            return "H1", "primary", f"P91 H1: on Bank Nifty the hybrid has a lower QLIKE than mcsGARCH (horizon {t}, all test days)"
        if p[2] == "all" and t == "last_hour":
            return "H2", "secondary", "P91 H2: on Bank Nifty the hybrid has a lower QLIKE than mcsGARCH in the last hour (all test days)"
        if p[2] == "expiry" and t in ("next15", "next30"):
            return "H3", "secondary", f"P91 H3: on Bank Nifty expiry days the hybrid has a lower QLIKE than mcsGARCH (horizon {t})"
    if (len(p) == 6 and re.fullmatch(r"ablation_(next15|next30|last_hour)\.csv", p[0]) and p[1] == "hybrid: base vs base+GEX"
            and p[2] in BNFC_GEX_SETS and p[3] == "all"):
        return "H4", "secondary", (f"P91 H4: adding {V2_SET.get(p[2], p[2])} lowers the Bank Nifty hybrid's QLIKE "
                                   f"({p[0].replace('ablation_', '').replace('.csv', '')}, all test days; the registered prediction is no improvement)")
    return None


def _bnfc_exploratory(sk: str) -> bool:
    p = sk.split("|")
    if p[0].startswith("ablation_") and len(p) == 6:
        return p[3] not in ("all", "expiry")
    if p[0].startswith("tests_") and len(p) == 4:
        return bool(BNFC_REGIME_SPLIT.match(p[2]))
    return False


def adapt_bnfc(ctx: Ctx):
    """P91: the Nifty forecasting pipeline run unchanged on Bank Nifty (docs/prereg/P91_BANKNIFTY_FORECAST.md;
    write-up docs/BANKNIFTY_FORECAST.md). The files have the layout of results/pipeline_nse_v2 and are read by
    adapt_r1v2 (same columns, Holm cross-checks and deduplication); this adapter then sets the registered roles.

    H1 (primary): hybrid_base against mcsGARCH, QLIKE, all days, next 15 and next 30 minutes (Holm over the two,
      cross-checked against p91/hypotheses.csv). H2 (secondary): the same, last hour. H3 (secondary): the same on
      expiry days, next 15 and 30 minutes (Holm over two). H4 (secondary): the hybrid with each of the four GEX sets
      against the base hybrid, QLIKE, all days, three targets (Holm within target from the ablation file). These
      keep their one-sided p (direction stated in the plan) and get the plan's pre-specification status.
    Exploratory (BANKNIFTY_FORECAST.md Section 10): the ablation by anchor time and period (every ablation group
      other than all days and expiry days) and the regime split into expiry and other days.
    Everything else was registered to be reported without a hypothesis: roles as in R1V2 (a headline role there
      becomes secondary), not pre-specified as a test, one-sided p doubled to two-sided.
    p91/: hypotheses.csv and h4_ablation_hybrid.csv repeat the pipeline's p-values (cross-checked, not counted);
      scoreboard_vs_mcsgarch.csv repeats tests_*.csv; period_split.csv is counted where it is not a pipeline
      group (the pooled monthly-only period); nifty_same_dates.csv (Nifty v2 forecasts on the same dates, called a
      description by the plan) is counted as exploratory; mcs_membership.csv is a Model Confidence Set table.
    If the folder does not exist (run not finished), nothing is counted and a note is written.
    """
    folder = BNFC_FOLDER
    if not ctx.res(folder).exists():
        ctx.skip("BNFC", f"{folder}/", "P91 Bank Nifty forecasting contest", 1,
                 "registered run not finished: the folder does not exist yet; the adapter reads the plan's layout "
                 "(as pipeline_nse_v2) when it appears")
        return 0
    if not sorted(ctx.res(folder).glob("tests_*.csv")):
        ctx.skip("BNFC", f"{folder}/", "P91 Bank Nifty forecasting contest", 1,
                 "folder present but no tests_*.csv yet (run not finished?); nothing counted")
        ctx.warn(f"{folder}: no tests_*.csv; nothing counted")
        return 0
    if not v2_shape_ok(ctx, folder):
        adapt_generic(ctx, "BNFC", folder)
        return 0
    ps_status, ps_note = prereg_status(ctx, folder, BNFC_OUTCOME)
    side_pre = SIDE_ONE_PRE if ps_status == PS_YES else SIDE_ONE_STATED
    n0 = len(ctx.rows)
    adapt_r1v2(ctx, study="BNFC", folder=folder)
    new = [r for r in ctx.rows[n0:] if r["study"] == "BNFC"]
    holm = {}
    rel = f"{folder}/p91/hypotheses.csv"
    if ctx.has(rel):
        h = ctx.read(rel)
        for code in ("H1", "H3"):
            g = h[h["hypothesis"].map(clean) == code]
            if len(g) and "p_holm" in g.columns:
                rec = ctx.check("BNFC", f"{code}: Holm over the next 15 and next 30 minutes", rel, "holm", g["p_one_sided"], g["p_holm"])
                for (_, r), rc in zip(g.iterrows(), rec):
                    holm[(code, clean(r["target"]))] = (num(r["p_one_sided"]), num(r["p_holm"]), rc)
        ctx.skip("BNFC", rel, "registered summary of H1 to H3", len(h), "repeats tests_*.csv (counted there); Holm values cross-checked")
    for r in new:
        h = _bnfc_h(r["source_key"])
        if h:
            code, role, text = h
            r.update(role=role, hypothesis=text, pre_specified=ps_status, sidedness=side_pre, p_raw=r["p_reported"],
                     pre_spec_note=f"{code} of the plan (Section 4); " + ps_note)
            r["note"] = re.sub(r"(; )?" + re.escape(V2_P_IS_ONE), "", r["note"])
            t = r["source_key"].split("|")[0].replace("tests_", "").replace(".csv", "")
            if (code, t) in holm:
                p1, ph, rc = holm[(code, t)]
                if not _same_num(p1, r["p_reported"]):
                    ctx.warn(f"{rel}: {code} {t} p differs from tests_{t}.csv")
                r.update(p_study_adjusted=ph, p_study_adjusted_recomputed=rc,
                         study_adjust_method=f"Holm over the two {code} tests (next 15 and next 30 minutes)")
            continue
        if r["sidedness"] in (SIDE_ONE_PRE, SIDE_ONE_STATED):
            r["sidedness"] = SIDE_CONVERTED
            r["p_raw"] = two_sided_from_one_sided(r["p_reported"])
            if r["p_reported"] >= 1.0:
                r["note"] = (r["note"] + "; " if r["note"] else "") + V2_P_IS_ONE
        if _bnfc_exploratory(r["source_key"]):
            r.update(role="exploratory", pre_specified=PS_NO, pre_spec_note=BNFC_PS_EXPL)
        else:
            r.update(role="secondary" if r["role"] == "headline" else r["role"], pre_specified=PS_NO, pre_spec_note=BNFC_PS_REPORT)
    lookup = _v2_lookup(ctx, folder)
    known = {round(num(r["p_reported"]), 12) for r in new}
    rel = f"{folder}/p91/h4_ablation_hybrid.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        miss = int(sum(round(num(p), 12) not in known for p in d["p_one_sided"] if np.isfinite(num(p))))
        if miss:
            ctx.warn(f"{rel}: {miss} p-values not found in the ablation files")
        ctx.skip("BNFC", rel, "registered H4 table (hybrid ablation)", len(d), "repeats ablation_*.csv rows (counted there)")
    rel = f"{folder}/p91/scoreboard_vs_mcsgarch.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        n, miss = 0, 0
        for _, r in d.iterrows():
            for loss, pc in (("qlike", "p_qlike"), ("mse", "p_mse")):
                if np.isfinite(num(r[pc])):
                    n += 1
                    miss += not _same_num(lookup.get((clean(r["target"]), loss, clean(r["group"]), f"mcsgarch vs {clean(r['model'])}"),
                                                     float("nan")), num(r[pc]))
        if miss:
            ctx.warn(f"{rel}: {miss} p-values not found in tests_*.csv")
        ctx.skip("BNFC", rel, "registered scoreboard against mcsGARCH (QLIKE and MSE p-values)", n, "repeats tests_*.csv (counted there)")
    rows = []
    rel = f"{folder}/p91/period_split.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        n_dup = 0
        for _, r in d.iterrows():
            t, sub = clean(r["target"]), clean(r["subset"])
            if any(_same_num(v, num(r["p_one_sided"])) for (tt, ll, _, cc), v in lookup.items()
                   if tt == t and ll == "qlike" and cc == "mcsgarch vs hybrid_base"):
                n_dup += 1
                continue
            expl = sub != "all"
            _, b, hyp = _v2_vs_mcsgarch("mcsgarch vs hybrid_base", num(r["change_pct"]), better_if_positive=False)
            row = _v2_add(ctx, "BNFC", hyp, f"QLIKE loss of forecasts of {TARGETS.get(t, t)}",
                          f"{clean(r['period'])}, {'expiry days' if sub == 'expiry' else 'all test days'} (n = {_int(r['days'])} days), horizon {t}",
                          r["p_one_sided"], SIDE_CONVERTED, role="exploratory" if expl else "robustness", pre_specified=PS_NO,
                          pre_spec_note=BNFC_PS_EXPL if expl else BNFC_PS_REPORT, source_file=rel,
                          source_key=key("period_split.csv", t, r["period"], sub), estimate=r["change_pct"],
                          statistic_name="loss relative to mcsGARCH (%)", statistic=r["dm_stat"],
                          p_method=V2_DM.format("model better than mcsGARCH"))
            if row is not None:
                rows.append(ctx.rows.pop())
        ctx.skip("BNFC", rel, "periods that are a pipeline group", n_dup, "same test as a tests_*.csv row (counted there)")
    rel = f"{folder}/p91/nifty_same_dates.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        for _, r in d.iterrows():
            t, sub = clean(r["target"]), clean(r["subset"])
            _, b, hyp = _v2_vs_mcsgarch("mcsgarch vs hybrid_base", num(r["change_pct"]), better_if_positive=False)
            row = _v2_add(ctx, "BNFC", "Nifty on the Bank Nifty test dates: " + hyp, f"QLIKE loss of Nifty forecasts of {TARGETS.get(t, t)}",
                          f"{sub} (n = {_int(r['days'])} days), horizon {t}", r["p_one_sided"], SIDE_CONVERTED, role="exploratory",
                          pre_specified=PS_NO, pre_spec_note="the plan calls this a description, not a test (Section 7, item 5); "
                          "counted because a p-value was reported", source_file=rel, source_key=key("nifty_same_dates.csv", t, sub),
                          estimate=r["change_pct"], statistic_name="loss relative to mcsGARCH (%)", statistic=r["dm_stat"],
                          p_method=V2_DM.format("model better than mcsGARCH"))
            if row is not None:
                rows.append(ctx.rows.pop())
    kept, dropped = dedup_exact(rows, cols=("p_reported", "statistic"))
    ctx.rows.extend(kept)
    ctx.skip("BNFC", f"{folder}/p91/", "rows repeating another row (same p and DM statistic)", len(dropped), "same test (counted once)")
    _consume_rest(ctx, "BNFC", folder, {
        r"mcs_membership\.csv": "Model Confidence Set membership (set-level)",
        r"(scoreboard|gain_by_anchor)_.*\.csv": "losses relative to mcsGARCH, no p-value (tests in tests_*.csv)",
    }, "forecasts, daily panels, hyperparameters, coefficients, feature importance, mcsGARCH parameters, figures, data checks "
       "or run information; no p-value")
    return len(new) + len(kept)


# ----------------------------------------------------------------------------
# P104: the forecasting pipeline on a third index, Fin Nifty (same layout as pipeline_nse_v2)
# ----------------------------------------------------------------------------

FNFC_FOLDER = "pipeline_nse_finnifty"
FNFC_OUTCOME = ["tests_*.csv", "ablation_*.csv", "p104/*.csv"]
FNFC_PS_EXPL = "labelled exploratory in THIRD_INDEX_FORECAST.md (not in the plan)"


def _fnfc_h(sk: str):
    """(code, role, hypothesis) of a registered hypothesis row of the Fin Nifty contest (P104), from its source_key."""
    h = _bnfc_h(sk)
    if h is None:
        return None
    code, role, text = h
    return code, role, text.replace("P91 ", "P104 ").replace("Bank Nifty", "Fin Nifty")


def adapt_fnfc(ctx: Ctx):
    """P104: the Nifty forecasting pipeline run unchanged on Fin Nifty (docs/prereg/P104_THIRD_INDEX_FORECAST.md;
    write-up docs/THIRD_INDEX_FORECAST.md). A copy of adapt_bnfc (P91) with the study code, folder and summary
    sub-folder changed: the same hypotheses H1 to H4 with the same roles, the same exploratory groups (the ablation
    by anchor time and period, the regime split into expiry and other days) and the same treatment of the rest.
    p104/other_indices_same_dates.csv holds Nifty v2 and Bank Nifty P91 forecasts on the Fin Nifty test dates
    (counted as exploratory, as P91's Nifty rows) and Fin Nifty's own rows, which repeat tests_*.csv (cross-checked,
    not counted). p104/exploratory.csv (experiments/p104_finnifty_exploratory.py, written after the results were
    seen) is counted as exploratory. If the folder does not exist (run not finished), nothing is counted and a note
    is written.
    """
    folder, st = FNFC_FOLDER, "FNFC"
    if not ctx.res(folder).exists():
        ctx.skip(st, f"{folder}/", "P104 Fin Nifty forecasting contest", 1,
                 "registered run not finished: the folder does not exist yet; the adapter reads the plan's layout "
                 "(as pipeline_nse_v2) when it appears")
        return 0
    if not sorted(ctx.res(folder).glob("tests_*.csv")):
        ctx.skip(st, f"{folder}/", "P104 Fin Nifty forecasting contest", 1,
                 "folder present but no tests_*.csv yet (run not finished?); nothing counted")
        ctx.warn(f"{folder}: no tests_*.csv; nothing counted")
        return 0
    if not v2_shape_ok(ctx, folder):
        adapt_generic(ctx, st, folder)
        return 0
    ps_status, ps_note = prereg_status(ctx, folder, FNFC_OUTCOME)
    side_pre = SIDE_ONE_PRE if ps_status == PS_YES else SIDE_ONE_STATED
    n0 = len(ctx.rows)
    adapt_r1v2(ctx, study=st, folder=folder)
    new = [r for r in ctx.rows[n0:] if r["study"] == st]
    holm = {}
    rel = f"{folder}/p104/hypotheses.csv"
    if ctx.has(rel):
        h = ctx.read(rel)
        for code in ("H1", "H3"):
            g = h[h["hypothesis"].map(clean) == code]
            if len(g) and "p_holm" in g.columns:
                rec = ctx.check(st, f"{code}: Holm over the next 15 and next 30 minutes", rel, "holm", g["p_one_sided"], g["p_holm"])
                for (_, r), rc in zip(g.iterrows(), rec):
                    holm[(code, clean(r["target"]))] = (num(r["p_one_sided"]), num(r["p_holm"]), rc)
        ctx.skip(st, rel, "registered summary of H1 to H3", len(h), "repeats tests_*.csv (counted there); Holm values cross-checked")
    for r in new:
        h = _fnfc_h(r["source_key"])
        if h:
            code, role, text = h
            r.update(role=role, hypothesis=text, pre_specified=ps_status, sidedness=side_pre, p_raw=r["p_reported"],
                     pre_spec_note=f"{code} of the plan (Section 4); " + ps_note)
            r["note"] = re.sub(r"(; )?" + re.escape(V2_P_IS_ONE), "", r["note"])
            t = r["source_key"].split("|")[0].replace("tests_", "").replace(".csv", "")
            if (code, t) in holm:
                p1, ph, rc = holm[(code, t)]
                if not _same_num(p1, r["p_reported"]):
                    ctx.warn(f"{rel}: {code} {t} p differs from tests_{t}.csv")
                r.update(p_study_adjusted=ph, p_study_adjusted_recomputed=rc,
                         study_adjust_method=f"Holm over the two {code} tests (next 15 and next 30 minutes)")
            continue
        if r["sidedness"] in (SIDE_ONE_PRE, SIDE_ONE_STATED):
            r["sidedness"] = SIDE_CONVERTED
            r["p_raw"] = two_sided_from_one_sided(r["p_reported"])
            if r["p_reported"] >= 1.0:
                r["note"] = (r["note"] + "; " if r["note"] else "") + V2_P_IS_ONE
        if _bnfc_exploratory(r["source_key"]):
            r.update(role="exploratory", pre_specified=PS_NO, pre_spec_note=FNFC_PS_EXPL)
        else:
            r.update(role="secondary" if r["role"] == "headline" else r["role"], pre_specified=PS_NO, pre_spec_note=BNFC_PS_REPORT)
    lookup = _v2_lookup(ctx, folder)
    known = {round(num(r["p_reported"]), 12) for r in new}
    rel = f"{folder}/p104/h4_ablation_hybrid.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        miss = int(sum(round(num(p), 12) not in known for p in d["p_one_sided"] if np.isfinite(num(p))))
        if miss:
            ctx.warn(f"{rel}: {miss} p-values not found in the ablation files")
        ctx.skip(st, rel, "registered H4 table (hybrid ablation)", len(d), "repeats ablation_*.csv rows (counted there)")
    rel = f"{folder}/p104/scoreboard_vs_mcsgarch.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        n, miss = 0, 0
        for _, r in d.iterrows():
            for loss, pc in (("qlike", "p_qlike"), ("mse", "p_mse")):
                if np.isfinite(num(r[pc])):
                    n += 1
                    miss += not _same_num(lookup.get((clean(r["target"]), loss, clean(r["group"]), f"mcsgarch vs {clean(r['model'])}"),
                                                     float("nan")), num(r[pc]))
        if miss:
            ctx.warn(f"{rel}: {miss} p-values not found in tests_*.csv")
        ctx.skip(st, rel, "registered scoreboard against mcsGARCH (QLIKE and MSE p-values)", n, "repeats tests_*.csv (counted there)")
    rows = []
    rel = f"{folder}/p104/period_split.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        n_dup = 0
        for _, r in d.iterrows():
            t, sub = clean(r["target"]), clean(r["subset"])
            if "p_one_sided" not in d.columns or not np.isfinite(num(r["p_one_sided"])):
                continue
            if any(_same_num(v, num(r["p_one_sided"])) for (tt, ll, _, cc), v in lookup.items()
                   if tt == t and ll == "qlike" and cc == "mcsgarch vs hybrid_base"):
                n_dup += 1
                continue
            expl = sub != "all"
            _, b, hyp = _v2_vs_mcsgarch("mcsgarch vs hybrid_base", num(r["change_pct"]), better_if_positive=False)
            row = _v2_add(ctx, st, hyp, f"QLIKE loss of forecasts of {TARGETS.get(t, t)}",
                          f"{clean(r['period'])}, {'expiry days' if sub == 'expiry' else 'all test days'} (n = {_int(r['days'])} days), horizon {t}",
                          r["p_one_sided"], SIDE_CONVERTED, role="exploratory" if expl else "robustness", pre_specified=PS_NO,
                          pre_spec_note=FNFC_PS_EXPL if expl else BNFC_PS_REPORT, source_file=rel,
                          source_key=key("period_split.csv", t, r["period"], sub), estimate=r["change_pct"],
                          statistic_name="loss relative to mcsGARCH (%)", statistic=r["dm_stat"],
                          p_method=V2_DM.format("model better than mcsGARCH"))
            if row is not None:
                rows.append(ctx.rows.pop())
        ctx.skip(st, rel, "periods that are a pipeline group", n_dup, "same test as a tests_*.csv row (counted there)")
    rel = f"{folder}/p104/other_indices_same_dates.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        own = d[d["index"].map(clean) == "Fin Nifty"]
        miss = int(sum(not any(_same_num(v, num(p)) for v in lookup.values()) for p in own["p_one_sided"]))
        if miss:
            ctx.warn(f"{rel}: {miss} Fin Nifty p-values not found in tests_*.csv")
        ctx.skip(st, rel, "Fin Nifty's own rows (all days and expiry days)", len(own), "repeat tests_*.csv (counted there)")
        for _, r in d[d["index"].map(clean) != "Fin Nifty"].iterrows():
            t, sub, idx = clean(r["target"]), clean(r["subset"]), clean(r["index"])
            name = idx.split(" (")[0]
            _, b, hyp = _v2_vs_mcsgarch("mcsgarch vs hybrid_base", num(r["change_pct"]), better_if_positive=False)
            row = _v2_add(ctx, st, f"{name} on the Fin Nifty test dates: " + hyp,
                          f"QLIKE loss of {name} forecasts of {TARGETS.get(t, t)}",
                          f"{sub} (n = {_int(r['days'])} days), horizon {t}", r["p_one_sided"], SIDE_CONVERTED, role="exploratory",
                          pre_specified=PS_NO, pre_spec_note="the plan calls this a description, not a test (Section 7, item 5); "
                          "counted because a p-value was reported", source_file=rel,
                          source_key=key("other_indices_same_dates.csv", idx, t, sub),
                          estimate=r["change_pct"], statistic_name="loss relative to mcsGARCH (%)", statistic=r["dm_stat"],
                          p_method=V2_DM.format("model better than mcsGARCH"))
            if row is not None:
                rows.append(ctx.rows.pop())
    rel = f"{folder}/p104/exploratory.csv"
    if ctx.has(rel):
        # experiments/p104_finnifty_exploratory.py, written after the registered results were seen: the split by
        # mcsGARCH's daily-level fallback and the comparison without 5 June 2024 (QLIKE and MSE). All exploratory.
        d = ctx.read(rel)
        for _, r in d.iterrows():
            t, loss = clean(r["target"]), clean(r["loss"])
            _, b, hyp = _v2_vs_mcsgarch("mcsgarch vs hybrid_base", num(r["change_pct"]), better_if_positive=False)
            row = _v2_add(ctx, st, hyp, f"{loss.upper()} loss of forecasts of {TARGETS.get(t, t)}",
                          f"{clean(r['analysis'])}: {clean(r['group'])} (n = {_int(r['days'])} days), horizon {t}",
                          r["p_one_sided"], SIDE_CONVERTED, role="exploratory", pre_specified=PS_NO,
                          pre_spec_note=FNFC_PS_EXPL, source_file=rel,
                          source_key=key("exploratory.csv", r["analysis"], t, r["group"], loss),
                          estimate=r["change_pct"], statistic_name="loss relative to mcsGARCH (%)", statistic=r["dm_stat"],
                          p_method=V2_DM.format("model better than mcsGARCH"))
            if row is not None:
                rows.append(ctx.rows.pop())
    kept, dropped = dedup_exact(rows, cols=("p_reported", "statistic"))
    ctx.rows.extend(kept)
    ctx.skip(st, f"{folder}/p104/", "rows repeating another row (same p and DM statistic)", len(dropped), "same test (counted once)")
    _consume_rest(ctx, st, folder, {
        r"mcs_membership\.csv": "Model Confidence Set membership (set-level)",
        r"(scoreboard|gain_by_anchor)_.*\.csv": "losses relative to mcsGARCH, no p-value (tests in tests_*.csv)",
        r"june5_2024\.csv": "the 09:15 forecasts of 5 June 2024 on three indices (descriptive, no p-value)",
    }, "forecasts, daily panels, hyperparameters, coefficients, feature importance, mcsGARCH parameters, figures, data checks "
       "or run information; no p-value")
    return len(new) + len(kept)


EXCLUDED_TOP = {
    "pipeline_synthetic": "synthetic data (simulation of the pipeline)",
    "inference_checks": "simulation checks of inference methods",
    "estimator_validation": "simulation checks of estimators",
    "paper_checks": "arithmetic checks of the paper's printed numbers",
    "data_validation": "data validation against NSE official values (no hypothesis test)",
    "live": "live-data snapshots, no tests",
    "power_study.csv": "simulation (power study on synthetic data)",
    "power_study_summary.csv": "simulation (power study on synthetic data)",
}


def scan_uncovered(ctx: Ctx):
    """List every results CSV with a p-like column that no adapter read and no rule excluded."""
    found = []
    for f in sorted(ctx.res("").rglob("*.csv*")):
        rel = str(f.relative_to(ctx.res(""))).replace("\\", "/")
        top = rel.split("/")[0]
        if rel in ctx.consumed or top in EXCLUDED_TOP or top.startswith("test_ledger"):
            continue
        try:
            cols = pd.read_csv(f, nrows=50)
        except Exception:  # noqa: BLE001
            continue
        if p_like_columns(cols):
            found.append(rel)
    for top, why in EXCLUDED_TOP.items():
        if ctx.res(top).exists():
            ctx.skip("EXCLUDED", top, top, 1, why)
    for rel in found:
        ctx.skip("UNCOVERED", rel, "file with a p-value column not read by any adapter", 1, "review: add to an adapter or exclude")
        ctx.warn(f"uncovered file with p-values: results/{rel}")
    return found


# ----------------------------------------------------------------------------
# Assemble
# ----------------------------------------------------------------------------

ADAPTERS = [adapt_r1, adapt_r1b, adapt_auction, adapt_r2, adapt_r3, adapt_r4, adapt_r5, adapt_repl,
            adapt_pin, adapt_event, adapt_fals, adapt_dsign, adapt_intra, adapt_conn, adapt_sensex]

STUDY_ORDER = ["R1", "R1V2", "R1V2DEV", "R1b", "AUCTION", "R2", "R3", "R4", "R5", "REPL", "PIN", "EVENT", "FALS",
               "DSIGN", "INTRA", "CONN", "SENSEX", "F2ECON", "PINMECH", "BNFC", "SWAP", "SYNTH", "REFFC", "REFEXP", "REFF2",
               "P103ECON", "P99SEED", "FNFC", "POOL3", "GAIN6", "MCSX", "CONNSET", "PERSFIX", "BENCHCPP", "GAINSTRESS",
               "PERSPLAC", "HYBCPP"]

# Studies read after ADAPTERS and the later folders: the referee adapters compare with rows already counted.
LATE_ADAPTERS = [adapt_bnfc, adapt_swap, adapt_synth, adapt_ref_forecast, adapt_ref_expiry, adapt_ref_f2, adapt_p103econ,
                 adapt_p99seed, adapt_fnfc, adapt_p107pool, adapt_p106gain, adapt_p109mcsx,
                 adapt_p113connset, adapt_p114persfix, adapt_p116benchcpp,
                 adapt_p110gainstress, adapt_p117persplac, adapt_p118hybcpp]


def headline18_check(ctx: Ctx):
    """FALSIFICATION T7 'Holm within the 18 headline tests', recomputed from the current R3 and R5 p-values."""
    rel = "falsification/t7_headline_adjusted.csv"
    if not ctx.has(rel):
        return
    rows = {r["_h18"]: r for r in ctx.rows if "_h18" in r}
    h = pd.read_csv(ctx.res(rel))
    keys = [(clean(r["finding"]), clean(r["rule"]), clean(r["spec_detail"]), clean(r["term"])) for _, r in h.iterrows()]
    missing = [k for k in keys if k not in rows]
    if missing:
        ctx.warn(f"headline-18 check: no ledger row for {missing}")
        return
    rec = p_holm([rows[k]["p_reported"] for k in keys])
    n_mis, maxd = 0, 0.0
    for (_, r), k, rh in zip(h.iterrows(), keys, rec):
        dd = abs(rh - num(r["p_holm_within_headline"]))
        maxd = max(maxd, dd)
        n_mis += int(dd > 1e-6)
        row = rows[k]
        msg = (f"FALSIFICATION T7 Holm within the 18 headline tests: {fmt(r['p_holm_within_headline'])} "
               f"(study's p {fmt(r['p'])}, n {int(num(r['n']))}); recomputed from current p-values: {fmt(rh)}")
        row["note"] = (row["note"] + "; " if row["note"] else "") + msg
    ctx.checks.append(dict(study="FALS", family="18 headline tests, recomputed from the current R3 and R5 files",
                           source_file=f"results/{rel}", method="holm", m_tests=18, n_compared=18, n_mismatch=n_mis,
                           max_abs_diff=maxd, result="match" if n_mis == 0 else "MISMATCH",
                           note="the study's F2 rows used 932 days (older code); R5 now uses 929"))


def build(root, write=True, write_doc=True, quiet=False):
    t0 = time.time()
    ctx = Ctx(root)
    for a in ADAPTERS:
        a(ctx)
    # later folders
    if ctx.res("f2_economics").exists() and any(ctx.res("f2_economics").iterdir()):
        adapt_f2econ(ctx)
    if ctx.res("pinning_mechanism").exists() and any(ctx.res("pinning_mechanism").iterdir()):
        adapt_pinmech(ctx)
    if ctx.res("pipeline_nse_v2").exists():
        if v2_shape_ok(ctx):
            adapt_r1v2(ctx)
            adapt_r1v2_dev(ctx)
        else:
            adapt_generic(ctx, "R1V2", "pipeline_nse_v2")
    for a in LATE_ADAPTERS:
        a(ctx)
    adapt_power_real(ctx)
    headline18_check(ctx)
    uncovered = scan_uncovered(ctx)

    df = pd.DataFrame(ctx.rows)
    order = {s: i for i, s in enumerate(STUDY_ORDER)}
    df["_o"] = df["study"].map(lambda s: order.get(s, 99))
    df = df.sort_values(["_o"], kind="mergesort").drop(columns=["_o"]).reset_index(drop=True)
    df["test_id"] = [f"{s}-{i:04d}" for s, i in zip(df["study"], df.groupby("study").cumcount() + 1)]
    df["p_by_project"] = p_by(df["p_raw"].values)
    df["p_by_study"] = np.nan
    df["p_holm_study"] = np.nan
    for s, g in df.groupby("study"):
        df.loc[g.index, "p_by_study"] = p_by(g["p_raw"].values)
        df.loc[g.index, "p_holm_study"] = p_holm(g["p_raw"].values)
    df["survives_by_project_5pct"] = df["p_by_project"] <= 0.05
    df["survives_holm_study_5pct"] = df["p_holm_study"] <= 0.05
    # per-row study adjustment check
    a, b = df["p_study_adjusted"], df["p_study_adjusted_recomputed"]
    both = a.notna() & b.notna()
    df.loc[both, "study_adjust_check"] = np.where((a[both] - b[both]).abs() <= 1e-6, "match", "MISMATCH")
    df.loc[a.notna() & b.isna(), "study_adjust_check"] = "not recomputed"
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith(("str", "string")):
            df[c] = df[c].map(clean)
    df = df[COLUMNS]
    nc = pd.DataFrame(ctx.not_counted)
    ck = pd.DataFrame(ctx.checks)
    for t in (nc, ck):
        for c in t.columns:
            if t[c].dtype == object or str(t[c].dtype).startswith(("str", "string")):
                t[c] = t[c].map(clean)
    if write:
        out = Path(root) / "results"
        df.to_csv(out / "test_ledger.csv", index=False)
        nc.to_csv(out / "test_ledger_not_counted.csv", index=False)
        ck.to_csv(out / "test_ledger_checks.csv", index=False)
    if write_doc:
        write_markdown(Path(root), df, nc, ck, ctx)
    if not quiet:
        summarize(df, ck, ctx, uncovered, time.time() - t0)
    return df, nc, ck, ctx


def summarize(df, ck, ctx, uncovered, secs):
    pd.set_option("display.width", 220)
    print(f"Ledger: {len(df)} tests from {df['study'].nunique()} studies; m = {len(df)}, "
          f"c(m) = {np.sum(1.0 / np.arange(1, len(df) + 1)):.3f}")
    tab = pd.crosstab(df["study"], df["role"], margins=True)
    print(tab)
    print("Survive BY (project, 5%):", int(df["survives_by_project_5pct"].sum()),
          "| survive Holm within study (5%):", int(df["survives_holm_study_5pct"].sum()))
    print(df[df["survives_by_project_5pct"]][["test_id", "role", "p_raw", "p_by_project", "hypothesis"]].to_string(max_colwidth=90))
    print("Checks:", ck["result"].value_counts().to_dict())
    print(ck[ck["result"] != "match"].to_string(max_colwidth=80))
    print("Row-level mismatches:", int((df["study_adjust_check"] == "MISMATCH").sum()))
    print("Uncovered files:", uncovered)
    print(f"Done in {secs:.1f} s")


# ----------------------------------------------------------------------------
# docs/TEST_LEDGER.md
# ----------------------------------------------------------------------------


def _p(x):
    v = num(x)
    if not np.isfinite(v):
        return ""
    if v == 0:
        return "0"
    if v < 1e-4:
        return f"{v:.1e}"
    return f"{v:.3g}" if v < 0.01 else f"{v:.3f}"


def _md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for r in rows:
        out.append("| " + " | ".join(clean(x).replace("|", "/") for x in r) + " |")
    return "\n".join(out)


def _sel(df, study, pattern):
    rx = re.compile(pattern)
    m = (df["study"] == study) & df["source_key"].map(lambda x: bool(rx.search(x)))
    return df[m]


RECON = [
    (1, "The data hold up.", [],
     "No p-values: this is data validation (results/data_validation), not a hypothesis test. Nothing to correct."),
    (2, "The paper's model can be beaten out of sample at 15 and 30 minutes, even against an mcsGARCH that is also given India VIX or the option-implied variance; "
        "most of the gain comes at the open: the hybrid (mcsGARCH corrected by LightGBM) "
        "is 5.8%, 7.6% and 6.4% better at 15 minutes, 30 minutes and the last hour on all days and 6.6%, 6.2% and 7.9% "
        "better on expiry days (QLIKE; one-sided DM p at most 0.010 in all six cells). Post hoc, the 15- and 30-minute gains "
        "survive Hansen's SPA and hold against an mcsGARCH given VIX or the implied variance (4.8%, p 0.005; 5.9%, p below "
        "0.001). It replicates on Bank Nifty (P91, registered plan): 3.9% and 5.0% at 15 and 30 minutes (Holm p 5e-12 "
        "and 2e-11), 8.1% in the last hour.",
     [("hybrid against mcsGARCH, QLIKE, all days (15 min, 30 min, last hour)", "R1V2",
       r"^tests_(next15|next30|last_hour)\.csv\|qlike\|all\|mcsgarch vs hybrid_base$"),
      ("hybrid against mcsGARCH, QLIKE, expiry days (15 min, 30 min, last hour)", "R1V2",
       r"^tests_(next15|next30|last_hour)\.csv\|qlike\|expiry\|mcsgarch vs hybrid_base$"),
      ("post hoc (referee): Hansen SPA, some challenger beats mcsGARCH, all days", "REFFC",
       r"^spa_reality_check\.csv\|(next15|next30|last_hour)\|all$"),
      ("post hoc (referee): hybrid against each information-matched mcsGARCH, QLIKE, all days", "REFFC",
       r"^fair_benchmarks\.csv\|(next15|next30|last_hour)\|all\|(mcsgarch_iv|mcsgarch_vix|mcsgarchx_vix|mcsgarchx_vix_iv)\|hybrid$"),
      ("P91 Bank Nifty (registered): H1 (15 and 30 min) and H2 (last hour), all days", "BNFC",
       r"^tests_(next15|next30|last_hour)\.csv\|qlike\|all\|mcsgarch vs hybrid_base$"),
      ("P91 Bank Nifty (registered): H3, expiry days (15 and 30 min)", "BNFC",
       r"^tests_(next15|next30)\.csv\|qlike\|expiry\|mcsgarch vs hybrid_base$")],
     "The ledger doubles the one-sided p-values of the current contest and of the referee checks, because those "
     "comparisons were not stated before the data were seen; the P91 rows keep their one-sided p (stated in the plan). "
     "An information-matched benchmark that is the best in a subset appears once (under its own name)."),
    (3, "The last hour: the first run's HAR edge (about 11%) came mostly from the closing-auction days and "
        "was withdrawn. In the model study the hybrid is 6.4% better than mcsGARCH (p below 0.001), 7.0% before the "
        "auction (p = 0.002), and HAR is not better at all (+1.9%). The least secure of the three horizons.",
     [("hybrid against mcsGARCH, last hour, QLIKE: all days, before the auction, expiry days", "R1V2",
       r"^tests_last_hour\.csv\|qlike\|(all|pre_auction|expiry)\|mcsgarch vs hybrid_base$"),
      ("HAR against mcsGARCH, last hour, QLIKE: all days and before the auction", "R1V2",
       r"^tests_last_hour\.csv\|qlike\|(all|pre_auction)\|mcsgarch vs har$"),
      ("the withdrawn first-run claim: HAR against mcsGARCH, last hour, all days", "R1",
       r"^tests_last_hour\.csv\|all\|mcsgarch vs har$")],
     "The HAR rows are what the finding says is not there (HAR is now worse than mcsGARCH, and the first-run "
     "claim is withdrawn), so their not surviving is what it says. The HAR rows of the current contest are "
     "two-sided, so a small p there would mean HAR is clearly worse."),
    (4, "GEX features do not improve out-of-sample forecasts under any sign rule, and signed GEX is never better "
        "than unsigned gross gamma: adding GEX to the hybrid moves its loss by -0.4% to +0.1% at 15 and 30 minutes "
        "and makes the last hour 0.9% to 1.6% worse; no cell on all days, expiry days or before the auction survives Holm.",
     [("hybrid with and without each GEX set (gross, US, Pro, client), QLIKE: all days, expiry days, before the auction", "R1V2",
       r"^ablation_(next15|next30|last_hour)\.csv\|hybrid: base vs base\+GEX\|(gross|us|participant|client)\|(all|expiry|pre_auction)\|"),
      ("hybrid, signed GEX against unsigned gross gamma, QLIKE: all days, expiry days, before the auction", "R1V2",
       r"^ablation_(next15|next30|last_hour)\.csv\|hybrid: unsigned vs signed\|[^|]+\|(all|expiry|pre_auction)\|")],
     "This is a null finding, so nothing needs to survive; the check is that no ablation is significant. "
     "The expiry-day rows are H1 and keep their one-sided p; the others are doubled to two-sided."),
    (5, "Pro-based GEX: the expiry-day sign flip survives the falsification tests (Holm 0.020 among 18 headline tests) but not on Bank Nifty (p = 0.25).",
     [("R3, Pro rule, regime x expiry", "R3", r"^regime x expiry\|gex_participant_pro\|"),
      ("R3 robustness, Pro rule, share scaling (the Holm 0.020 row)", "R3", r"^participant_pro\|share\|gex_l1"),
      ("Bank Nifty replication, Pro rule, GEX x expiry", "FALS", r"^T5 Bank Nifty\|banknifty\|F1\|participant_pro\|gex_l1_x_exp, share")],
     "The Holm 0.020 quoted in the finding is within the 18 headline tests only."),
    (6, "Expiry-day premium: about +11% (p 0.045) in the multi-weekly regime, +21% (p 0.005) on Tuesdays, +43% in the Tuesday "
        "last hour (p 0.00005), about 63% with Sensex days out of the baseline. Post hoc, against days on which no NSE or BSE "
        "index option expires: Tuesday +26% (p 0.0009) and +63% in the last hour; the pre-November 2024 premium cannot be identified.",
     [("whole day, multi-weekly and Tuesday regimes", "R2", r"^rv_day\|(multi_weekly|single_weekly_tue)$"),
      ("last hour, Tuesday regime", "R2", r"^rv_last_hour\|single_weekly_tue$"),
      ("last hour, Tuesday regime, Sensex days out of the baseline", "SENSEX", r"R2 with Sensex days out of the baseline: rv_last_hour, single_weekly_tue$"),
      ("post hoc (referee): multi-weekly and Tuesday regimes, baselines without any NSE or BSE expiry", "REFEXP",
       r"^r2_clean_baseline\.csv\|no expiry of any index, NSE or BSE[^|]*\|(rv_day|rv_last_hour)\|(multi_weekly|single_weekly_tue)$")],
     "Referee rows that repeat an R2 or SENSEX row are counted once, under R2 or SENSEX."),
    (7, "F2: the end-of-day reversal is stronger after negative Pro GEX (Holm 0.016 among 78; out-of-sample test p 0.006; "
        "permutation p 0.0005 to 0.010). Post hoc, with inference built for the slow negative-Pro state, it holds over the full "
        "sample (largest robust p 0.012) and from 2025 (0.038) but not inside 2023 to 2024.",
     [("R5 in-sample, Pro rule, binary, all days", "R5", r"^binary\|participant_pro\|all\|r_rod_x_neg$"),
      ("'pre-registered style' out-of-sample test (one-sided as stated)", "FALS", r"^T4 pre-registered F2\|(full|pre_auction)\|F2\|participant_pro"),
      ("permutation tests, Pro rule, binary, full sample", "FALS", r"^T1 permutation\|full\|F2\|participant_pro\|x_neg_binary"),
      ("post hoc (referee): Pro rule, binary, largest p over the persistence-robust methods; full sample, 2023 to 2024, 2025 onward",
       "REFF2", r"^item3_f2_inference\.csv\|pro\|binary\|r_last\|(full|2023_2024|2025_onward)\|N_x_r_rod$")],
     "The 'Holm 0.016 among 78' comes from FALSIFICATION family A, which used an older run (932 days); see section 6. "
     "The referee rows are re-estimates on those 932 days and use, for each estimate, the largest p among the methods that "
     "respect the persistence of the state."),
    (8, "Dealer sign: outcome (a) points weakly to Pro (p 0.18), outcome (b) to the reverse of Pro (p 4e-6); out of sample no estimated sign beats the fixed rules after correction.",
     [("estimated direction, full sample, participants, outcome (a)", "DSIGN", r"^full sample\|base\|participants\|a$"),
      ("estimated direction, full sample, participants, outcome (b)", "DSIGN", r"^full sample\|base\|participants\|b$"),
      ("the 10 primary out-of-sample DM tests", "DSIGN", r"^all test days\|(a|b)\|signed\|est_(participants|shares)\|")],
     "For the out-of-sample part the finding is a null, so not surviving is what it says."),
    ("P91", "Bank Nifty forecasting (registered plan): the Nifty hybrid, run with nothing changed, beats mcsGARCH at 15 and 30 "
            "minutes (H1, Holm over two); H4 predicted that GEX does not help, and that prediction failed at 15 minutes.",
     [("H1, primary: 15 and 30 minutes, all days", "BNFC", r"^tests_(next15|next30)\.csv\|qlike\|all\|mcsgarch vs hybrid_base$"),
      ("H4: hybrid with each GEX set against the base hybrid, all days", "BNFC",
       r"^ablation_(next15|next30|last_hour)\.csv\|hybrid: base vs base\+GEX\|(gross|us|participant|client)\|all\|")],
     "H1 keeps its one-sided p (stated in the plan). H4 predicted no improvement; BANKNIFTY_FORECAST.md calls the "
     "prediction failed because four 15-minute cells have a Holm p (over the four sets) at most 0.05. Across the whole "
     "project none of the 12 H4 rows survives, so the failure rests on the plan's own Holm family, not on project-wide "
     "correction."),
    ("P92", "Nifty and Sensex expiry-weekday swap (registered plan): the expiring index becomes the net transmitter of "
            "spillovers on its own expiry day (primary P1, whole day). Not supported (+0.42 points, p 0.33); the last-hour "
            "secondary S1 has Holm 0.015, but a placebo at a fake swap date also moved (p 0.038).",
     [("P1, primary, whole day (one-sided, registered)", "SWAP", r"^tests\.csv\|P1$"),
      ("S1 to S3, secondary", "SWAP", r"^tests\.csv\|S[123]$"),
      ("placebos PL1 to PL3", "SWAP", r"^tests\.csv\|PL[123]$")],
     "The registered result is a null finding (not supported), so the check is that nothing survives that would contradict it."),
    ("P96", "Synthetic control of the November 2024 weekly-expiry cut (registered plan): Bank Nifty's old expiry "
            "Wednesday calmed down relative to a synthetic control (primary). No support (ratio p 0.95; the effect is positive).",
     [("primary: Bank Nifty, RMSPE-ratio test", "SYNTH", r"^ratio\|primary\|BN$"),
      ("secondaries: Fin Nifty and Midcap Select", "SYNTH", r"^ratio\|secondary\|(FIN|MID)$"),
      ("post hoc reading: difference-in-differences against the donor average, main runs", "SYNTH", r"^did\|(primary|secondary)\|")],
     "The registered result is a null finding (no support), so the check is that nothing survives that would contradict "
     "it. The DiD rows are the post hoc reading of a rise (exploratory)."),
    ("P103", "Economic value of the forecasting hybrid (registered plan): at 30 minutes the hybrid gives a higher "
             "realized utility than mcsGARCH (H1), but the fee is tiny (about 0.04% of wealth a year); against the "
             "information-matched mcsgarch_iv at 15 and 30 minutes it is not significant after Holm.",
     [("H1, primary: 30 minutes, non-overlapping windows, all days", "P103ECON", r"^utility_tests\.csv\|H1\|"),
      ("S1 to S4, secondary (Holm over four)", "P103ECON", r"^utility_tests\.csv\|S[1-4]\|"),
      ("T1 and T2, the registered trading tests", "P103ECON", r"^trading_tests\.csv\|T[12]\|")],
     "The verdict of P103 rests on the size of the fee and on T1, not only on whether a p-value survives; see "
     "docs/P103_ECONOMIC_VALUE.md."),
    ("P99", "Seed bootstrap of the forecasting hybrid (registered plan): every one of 20 new 5-seed ensembles beats "
            "mcsGARCH at 15 and 30 minutes on all days (primary). Holds; in the last hour of expiry days one ensemble "
            "has p 0.052 (weakened).",
     [("primary: 15 and 30 minutes, all days", "P99SEED", r"^verdict\.csv\|next(15|30)\|all"),
      ("secondary: last hour on all days, and expiry days", "P99SEED", r"^verdict\.csv\|(last_hour\|all|.*\|expiry)")],
     "The p of each row is the largest DM p over the 20 ensembles (an intersection-union test), so a row that "
     "survives means every ensemble's gain survives."),
    ("P104", "Fin Nifty forecasting, the third index (registered plan): the Nifty hybrid, run with nothing changed, beats "
             "mcsGARCH at 15 and 30 minutes (H1, Holm over two: replicated); the last hour (H2) and Fin Nifty expiry days "
             "(H3) are not supported; H4 predicted that GEX does not help, and that prediction held.",
     [("H1, primary: 15 and 30 minutes, all days", "FNFC", r"^tests_(next15|next30)\.csv\|qlike\|all\|mcsgarch vs hybrid_base$"),
      ("H2 and H3, secondary: last hour all days, expiry days at 15 and 30 minutes", "FNFC",
       r"^tests_(last_hour\.csv\|qlike\|all|(next15|next30)\.csv\|qlike\|expiry)\|mcsgarch vs hybrid_base$"),
      ("H4: hybrid with each GEX set against the base hybrid, all days", "FNFC",
       r"^ablation_(next15|next30|last_hour)\.csv\|hybrid: base vs base\+GEX\|(gross|us|participant|client)\|all\|")],
     "H1 to H3 keep their one-sided p (stated in the plan). H4 is a null prediction, so not surviving is what it says."),
    ("P107", "Pooled three-index test (registered plan): pooled over Nifty, Bank Nifty and Fin Nifty, the hybrid beats "
             "mcsGARCH at 15 and 30 minutes (primary, Driscoll-Kraay, Holm over two). It summarises evidence already "
             "counted in R1V2, BNFC and FNFC.",
     [("P1, primary: 15 and 30 minutes, all common dates", "POOL3", r"^tests\.csv\|P1_"),
      ("S1b to S1e, secondary: equal gains, last hour, expiry days, edge over mcsgarch_iv", "POOL3", r"^tests\.csv\|S1[b-e]_")],
     "The primary test could not fail once the three single-index results were known (the plan shows the bound); "
     "the informative outputs are the effective number of indices and the edge over mcsgarch_iv."),
    ("P106", "Feature-group ablation (registered plan): removing realized history or time of day raises the hybrid's "
             "loss; removing the option-implied, VIX or calendar group does not; no group's value changed "
             "significantly across the November 2024 cut (primary, Holm over 20).",
     [("P1, primary: one group removed, 15 and 30 minutes", "GAIN6", r"^tests\.csv\|P1_"),
      ("P2, primary: change across the November 2024 cut", "GAIN6", r"^tests\.csv\|P2_")],
     "P2 rows are expected not to survive; the implied group at 30 minutes is the closest (Holm 0.051 in the study)."),
    ("P109", "Five registered parametric fixes to mcsGARCH, combined, beat mcsGARCH at 15 and 30 minutes (primary, "
             "Holm over two) and recover the hybrid's gain.",
     [("P1, primary: the combination against mcsGARCH, 15 and 30 minutes", "MCSX", r"^tests\.csv\|P1_")],
     "The extensions were chosen after P106 on the same days, so this explains the gain rather than confirming it."),
    ("P113", "Some setting among 5,440 per before window reproduces the paper's 15 printed connectedness totals "
             "better than chance (primary P1; P2 and P3 have no p-value because no setting met their criteria).",
     [("P1, primary: best fit against shuffled totals", "CONNSET", r"^tests\.csv\|P1_")],
     "Verdict 'not reproduced': the best setting fits better than chance but misses the before last hour by 23 points."),
    ("P114", "Under mcsGARCH with P109's fixes (X1 to X3), Nifty's intraday persistence still falls from the inferred "
             "before window to the after window (primary, day-block bootstrap, one-sided).",
     [("P1, primary: Nifty, inferred window, impulse-response persistence", "PERSFIX", r"^tests\.csv\|P1_")],
     "Verdict 'survives' for Nifty on the inferred window only; Bank Nifty, the stated window and P108's window "
     "are weakened or gone, and the size of the fall is very uncertain."),
    ("P116", "The hybrid beats mcsGARCH as the paper specifies it, and our mcsGARCH differs from it, at 15 and 30 "
             "minutes on Nifty (primary, Holm over four).",
     [("P, primary: hybrid and our mcsGARCH against mcs_cpp, 15 and 30 minutes", "BENCHCPP", r"^tests\.csv\|P_")],
     "Verdict 'headline overstated': our extension made the Nifty benchmark worse, so the gain over the paper's "
     "specification (4.43% and 6.64%) is smaller than the headline (5.80% and 7.65%)."),
    ("P110", "P106's three claims hold on Bank Nifty and Fin Nifty: realized history and time of day carry the gain, "
             "the option-implied, VIX and calendar groups are equivalent to zero within 1 point, and the mcsinfo "
             "hybrid beats mcsGARCH and is non-inferior to the full hybrid (primary F1 to F3, Holm).",
     [("F1 to F3, primary: two indices, 15 and 30 minutes", "GAINSTRESS", r"^tests\.csv\|F[123]_")],
     "Claim 1 survives on both indices; claim 2 is qualified (calendar adds a little); claim 3 partly survives."),
    ("P117", "Nifty's fall in intraday persistence from the inferred before window to the after window is larger "
             "than the changes between placebo pairs of windows that straddle neither Bank Nifty move (primary, "
             "reference mcsGARCH and the fixed model F, Holm over two).",
     [("P, primary: Nifty, primary placebo set, reference and F", "PERSPLAC", r"^tests\.csv\|P_")],
     "Verdict 'the fall is within the normal range' under both models (placebo p 0.191 and 0.105, Holm 0.211); "
     "Fin Nifty, whose expiry did not move, fell as much."),
    ("P118", "The hybrid recipe built on mcsGARCH as the paper specifies it beats that mcsGARCH (H1), and it is "
             "equivalent to the committed hybrid within 0.5 point (H2), at 15 and 30 minutes on Nifty (primary, "
             "Holm over four).",
     [("P, primary: hybrid_cpp against mcs_cpp, and TOST against the committed hybrid, 15 and 30 minutes", "HYBCPP",
       r"^tests\.csv\|P_")],
     "H1 holds (5.06% and 7.09%); H2 is not shown and no difference is shown either, so the registered verdict on "
     "whether the base matters is 'inconclusive' at both horizons."),
]


def write_markdown(root: Path, df: pd.DataFrame, nc: pd.DataFrame, ck: pd.DataFrame, ctx: Ctx):
    m = len(df)
    cm = float(np.sum(1.0 / np.arange(1, m + 1)))
    n_sig = int((df["p_raw"] < 0.05).sum())
    surv = df[df["survives_by_project_5pct"]]
    holm_s = df[df["survives_holm_study_5pct"]]
    by_s = df[df["p_by_study"] <= 0.05]
    L = []
    w = L.append
    w("# Test ledger: every hypothesis test in the project, and what survives correction")
    w("")
    w("This page is generated by `scripts/build_test_ledger.py` from the result files. Rerun the script after "
      "any study changes; the numbers below then update. The full ledger is `results/test_ledger.csv` "
      f"({m} rows, one per test). Results that are reported but deliberately not counted are in "
      "`results/test_ledger_not_counted.csv`, and the recomputation of every adjusted p-value that a study "
      "reports itself is in `results/test_ledger_checks.csv`. All results are REAL DATA unless a study says otherwise.")
    w("")
    w("## 1. Why this ledger exists")
    w("")
    w("A p-value answers one question: if there were no effect, how often would a result at least this "
      "extreme turn up by chance? At the usual 5% line, one test in twenty with no real effect behind it "
      f"still comes out \"significant\". This project ran **{m} tests** across {df['study'].nunique()} studies. "
      f"Even if none of the effects were real, about {int(round(0.05 * m))} of them would be expected to "
      f"pass 5% by luck alone. In fact {n_sig} have a raw p below 0.05. The question for a reader is which "
      "of those would still stand if we account for how many shots we took.")
    w("")
    w("**Multiple-testing corrections** raise each p-value to reflect the number of tests. Two are used here:")
    w("")
    w("* **Benjamini-Yekutieli (BY)** controls the false discovery rate: among the results we call "
      "significant, the expected share of false ones stays below 5%. Unlike the better-known Benjamini-Hochberg "
      "method, BY is valid whatever the dependence between tests. That matters here, because many tests reuse "
      "the same days, the same GEX series and the same outcomes, so they are strongly correlated. The price is "
      f"that BY is conservative: it multiplies by an extra factor c(m) = 1 + 1/2 + ... + 1/m, which for m = {m} "
      f"is {cm:.2f}. With {m} tests, the smallest p-value must be below about "
      f"{0.05 / (m * cm):.1e} to survive, and the k-th smallest below about k x {0.05 / (m * cm):.1e}.")
    w("* **Holm** (within each study) controls the chance of even one false positive within that study. It "
      "is the standard, strict choice for a study's own family of tests.")
    w("")
    w("We report three adjusted values for every test: BY over the whole project (`p_by_project`), BY "
      "within its study (`p_by_study`) and Holm within its study (`p_holm_study`). Project-wide BY is the "
      "strictest reading: it treats the project as one search. Within-study values are the fairer reading "
      "for a study that was designed on its own. Neither replaces judgement: a correlated set of tests "
      "that all point the same way can be more convincing than any single p-value suggests, and BY ignores that.")
    w("")
    w("**How the ledger counts.** One row is one hypothesis: one estimate, in one sample, under one "
      "specification. Where a study reports several p-values for the same estimate (for example a HAC p, a "
      "bootstrap p and a placebo p), the one the study treats as main is used and the rest go in `alt_p`. "
      "Coefficients on control variables, pre-trend diagnostics and placebo draws are not tests. A test "
      "reported in two files is counted once. Raw p-values are two-sided unless the study stated a one-sided "
      "test in advance (column `sidedness`); a one-sided p without such a statement is doubled. No p-value "
      "is invented: a result shown only as a figure, an interval or text is listed as not counted.")
    w("")
    yes_st = sorted(set(df.loc[df["pre_specified"] == PS_YES, "study"]))
    w("**Pre-specified?** `yes` means a written statement of the test was committed to git before the "
      "run, judged from the commit times in the reflogs (`logs/HEAD` and `logs/refs/heads` of the repository; from a "
      "worktree, those of the main repository; no git command is run; confirm with git log). "
      "That holds for RESEARCH_STRATEGY_PANDEY.md (RQ1 and H1), which was in the repository before the first "
      "real-data run, and for the P87, P90, P91, P92, P96 and P104 pre-registrations in `docs/prereg/` when their commit is "
      "older than the study's result files. For P91, P92, P96 and P104 only the files that hold outcomes are compared, because "
      "P92's coverage counts were committed together with its plan on purpose. "
      f"Studies with `yes` rows now: {', '.join(yes_st) if yes_st else 'none'}. "
      "Every other study was committed together with its results, so a claim that a test was fixed in advance "
      "is marked `stated, not time-stamped`. `partly` means the direction was stated in advance but the exact "
      "specification was chosen later.")
    w("")
    # counts
    w("## 2. How many tests, by study and role")
    w("")
    roles = [r for r in ROLES if r in set(df["role"])]
    ct = pd.crosstab(df["study"], df["role"]).reindex(columns=roles, fill_value=0)
    ct = ct.reindex([s for s in STUDY_ORDER if s in ct.index] + [s for s in ct.index if s not in STUDY_ORDER])
    rows = []
    for s, r in ct.iterrows():
        g = df[df["study"] == s]
        rows.append([f"{s}: {STUDIES.get(s, (s,))[0]}"] + [str(int(r[c])) for c in roles] +
                     [str(len(g)), str(int((g["p_raw"] < 0.05).sum())), str(int(g["survives_holm_study_5pct"].sum())),
                      str(int((g["p_by_study"] <= 0.05).sum())), str(int(g["survives_by_project_5pct"].sum()))])
    tot = ["**all**"] + [str(int(ct[c].sum())) for c in roles] + [str(m), str(n_sig), str(len(holm_s)), str(len(by_s)), str(len(surv))]
    rows.append(tot)
    w(_md_table(["study"] + roles + ["tests", "raw p < 0.05", "Holm study", "BY study", "BY project"], rows))
    w("")
    w("The last three columns count tests that stay at or below 0.05 after each correction. Pre-specification: " +
      "; ".join(f"{k}: {v}" for k, v in df["pre_specified"].value_counts().items()) + ". "
      f"The event study alone contributes {int((df['study'] == 'EVENT').sum())} tests "
      f"({100 * (df['study'] == 'EVENT').mean():.0f}% of the ledger).")
    w("")
    con = df["study"].isin(["R1", "R1V2", "R1V2DEV"])
    if con.any():
        w("**The forecasting contest is three studies.** **R1** is the first run (`results/pipeline_nse/`), now "
          "archived: REAL_DATA_RESULTS.md mentions it only as history (finding 3 and \"What the first run said\"), and "
          "its last-hour HAR claim was withdrawn. **R1V2** is the current contest (`results/pipeline_nse_v2/`, the model study of "
          "MODEL_ITERATIONS_REAL.md), the one REAL_DATA_RESULTS.md reports: every model against mcsGARCH in up to 28 "
          "groups of days under two losses (QLIKE and MSE), every with-and-without comparison of the GEX ablation, "
          "and the seed checks. **R1V2DEV** holds the development runs of the model study (test days to 30 June 2025, "
          "with an older protocol: mcsGARCH refitted every 63 days, normal p-values, one random seed). All three are "
          "counted because all their p-values were seen: the first run's were published before they were replaced, "
          "and the development runs were used to choose the final design, so leaving either out would hide part of "
          "the search. They are different forecasts, so they are not merged, and R1V2DEV is kept apart from R1V2 so "
          "that the within-study corrections of R1V2 are about the final contest only. "
          f"Together they contribute {int(con.sum())} tests ({100 * con.mean():.0f}% of the ledger): "
          + ", ".join(f"{s_} {int((df['study'] == s_).sum())}" for s_ in ("R1", "R1V2", "R1V2DEV")) + ". "
          "Their p-values are one-sided (\"this model has lower loss\"). The ledger keeps them one-sided for the first "
          "run, as before, and for the rows that are H1 of RESEARCH_STRATEGY_PANDEY.md (the same model without and "
          "with GEX, on expiry days), and doubles all others to two-sided. So in R1V2 and R1V2DEV a small p can also "
          "mean that a model is clearly worse than the one it is compared with; the hypothesis text says which way "
          "the difference went.")
        w("")
    newst = [s_ for s_ in ("BNFC", "FNFC", "SWAP", "SYNTH", "REFFC", "REFEXP", "REFF2") if (df["study"] == s_).any()]
    if newst:
        cnt = {s_: int((df["study"] == s_).sum()) for s_ in newst}
        bn = df[df["study"] == "BNFC"]
        n_h = int((bn["pre_specified"] == PS_YES).sum()) if len(bn) else 0
        f2c = getattr(ctx, "ref_f2_counts", None)
        w("**Registered plans and referee responses.** " + ", ".join(f"{s_} {cnt[s_]}" for s_ in newst) + " tests. "
          "Of these, three studies have a registered analysis plan: **P91** (BNFC, the Nifty forecasting pipeline run unchanged on "
          "Bank Nifty), **P92** (SWAP, the Nifty and Sensex expiry-weekday swap) and **P96** (SYNTH, a synthetic control "
          "of the November 2024 cut in weekly expiries). Their primary, secondary, placebo and robustness tests carry the "
          "plan's status and role, and only the tests whose direction the plan states keep a one-sided p. BNFC has the "
          "same files as R1V2 and is counted the same way: every comparison the pipeline wrote is a test. "
          f"{n_h} of them are the plan's hypotheses (H1 primary; H2, H3 and H4 secondary); the rest were registered to "
          "be reported without a hypothesis (roles as in R1V2) or are labelled exploratory in BANKNIFTY_FORECAST.md, "
          "and their one-sided p-values are doubled. Rows that a plan calls descriptive but that report a p-value "
          "(P92's total connectedness and H2 without weekday effects, P96's Nifty 50 Wednesday and its "
          "difference-in-differences cross-checks, P91's Nifty on the same dates) are counted as exploratory, because "
          "the p-values were reported and seen.")
        if "FNFC" in cnt:
            fn = df[df["study"] == "FNFC"]
            w("")
            w(f"**P104** (FNFC, {cnt['FNFC']} tests) runs the same pipeline a third time, on Fin Nifty, under its own "
              "registered plan, and is counted exactly like BNFC: "
              f"{int((fn['pre_specified'] == PS_YES).sum())} rows are the plan's hypotheses H1 to H4; Nifty's and Bank "
              "Nifty's forecasts on the Fin Nifty test dates, and the two looks at the benchmark added after the results "
              "were seen (mcsGARCH's EWMA fallback, and the comparison without 5 June 2024), are counted as exploratory.")
        w("")
        w("The three **referee responses** (REFFC forecasting, REFEXP expiry premia, REFF2 the F2 reversal) were all "
          "done after the referee report, so every row is exploratory and none is pre-specified. A referee row that "
          "repeats a test already in the ledger (same estimate and p, for example a published event-study row, an R2 "
          "premium or a DM test of the current contest) is not counted again; the referee files' other p-values for "
          "such a test (placebo-date, prediction-interval, wild cluster) are listed as not counted. The trend-adjusted "
          "change of the expiry response is a new estimate for each event-study row and is counted. Hansen's SPA p is "
          "one-sided by construction (does any challenger beat mcsGARCH); it was not stated in advance, so it is doubled "
          "as min(1, 2p). The F2 files report each estimate under up to 17 inference methods; the ledger counts one row "
          "per estimate, with the largest p among the methods that respect the persistence of the Pro state as the main "
          "p (the study's own cautious reading; the largest of several valid p-values is itself valid) and the others in "
          "`alt_p`."
          + (f" So the {f2c[0]['1-7']} p-values of items 1 to 7 "
             + ("(as REFEREE_RESPONSE_F2.md says)" if f2c[0]['1-7'] == 598 else "(REFEREE_RESPONSE_F2.md says 598)")
             + f" are {f2c[1]['1-7']} tests, and the {f2c[0]['8']} of item 8 "
             + ("(as it says)" if f2c[0]['8'] == 381 else "(it says 381)")
             + f" are {f2c[1]['8']} tests." if f2c else ""))
        w("")
    if (df["study"] == "P103ECON").any():
        n103 = int((df["study"] == "P103ECON").sum())
        w(f"**P103** (P103ECON, {n103} tests, registered plan): the economic value of the forecasting hybrid. H1 is "
          "primary, S1 to S4 are secondary (Holm over the four, cross-checked), T1 and T2 are the registered trading "
          "tests (secondary), and every other row is a robustness check that the plan lists (other windows and subsets, "
          "the trading tests against mcsgarch_iv and at other cost levels, and each rule's mean net return at each cost "
          "level). All keep their one-sided p because the plan states the direction.")
        w("")
    # survivors
    w("## 3. What survives project-wide BY at 5%")
    w("")
    if len(surv) == 0:
        w("**Nothing.** No test in the project survives BY at 5% across all tests.")
        w("")
    else:
        cut = float(surv["p_raw"].max())
        ev = surv[surv["study"] == "EVENT"]
        ev_turn = ev[ev["hypothesis"].str.contains("turnover", case=False)]
        ev_plc = ev[ev["role"] == "placebo"]
        gex_studies = {"R3", "R5", "FALS", "DSIGN", "INTRA", "CONN", "REFF2"}
        contest = {"R1", "R1V2", "R1V2DEV", "BNFC", "FNFC", "REFFC"}
        gex_mask = (df["study"].isin(gex_studies) |
                    (df["study"].isin({"AUCTION", "PIN"}) & df["hypothesis"].str.contains("GEX|gamma", case=False, regex=True)) |
                    (df["study"].isin(contest) & df["hypothesis"].str.contains("GEX", regex=False)
                     & ~df["hypothesis"].str.contains("mcsGARCH", regex=False)))
        con_s = surv[surv["study"].isin(contest)]
        con_worse = con_s[con_s["hypothesis"].str.contains(r"observed: \S+ worse\)", regex=True)]
        gs = surv[gex_mask.loc[surv.index]]
        gs_claims = gs[~gs["role"].isin(["placebo"])]
        w(f"**{len(surv)} of {m} tests survive ({100 * len(surv) / m:.0f}%). That is more than \"very few\", but it "
          "says little about the project's gamma claims, for three reasons.**")
        w("")
        w(f"1. **The bar is set by the strong, unsurprising results.** BY steps up through the sorted p-values, so "
          f"the cut-off depends on how many very small p-values there are. Here it works out at a raw p of about "
          f"{cut:.1e}: every test at or below that survives and none above it does. The project contains hundreds of "
          f"near-certain effects: option and futures turnover jumping on expiry days ({len(ev_turn)} of the "
          f"{len(ev)} event-study survivors are turnover measures), forecasting models beating mcsGARCH at 15 and "
          f"30 minutes, simple forecasts (naive, VIX, the straddle alone) losing to it by a wide margin ({len(con_worse)} "
          f"of the {len(con_s)} forecasting-contest survivors are a model clearly worse than the one it is compared "
          "with, which counts as a rejection once a one-sided p is doubled to two-sided), Sensex calendar checks, "
          "volume changes in the replication. Each one makes room for the next. "
          "BY's 5% promise is about the whole list of survivors, not about any subset such as the GEX results, "
          "where false positives could be more concentrated.")
        w(f"2. **The event study's HAC p-values reject too often.** Its own negative-control check (Fridays, no rule "
          f"change) found HAC p below 0.05 in 20% to 46% of tests (EVENT_STUDY.md 6.2), and {len(ev_plc)} of those "
          f"Friday placebo rows survive here. The study itself trusts only results with a small placebo-date p, "
          "whose smallest possible value is about 0.005; no p-value that large can survive project-wide BY.")
        w(f"3. **Some survivors cut against a claim.** {len(gs)} survivors involve GEX, dealer gamma or the dealer "
          f"sign ({len(gs) - len(gs_claims)} of them placebo rows). Several are falsification results: gamma "
          "measured after the outcome predicting as well as gamma measured before it, or gross gamma with no sign "
          "predicting volatility. A surviving placebo is evidence against the original claim.")
        w("")
        ne = df[df["study"] != "EVENT"]
        ne_by = p_by(ne["p_raw"].values)
        w(f"Without the event study ({m - len(ne)} tests), {int((ne_by <= 0.05).sum())} of the remaining {len(ne)} tests "
          f"would survive BY, with a cut-off of about {ne.loc[ne_by <= 0.05, 'p_raw'].max():.1e} on the raw p.")
        w("")
        w("By study: " + ", ".join(f"{k} {v}" for k, v in surv["study"].value_counts().items()) + ". By role: " +
          ", ".join(f"{k} {v}" for k, v in surv["role"].value_counts().items()) + ".")
        w("")
        w("**Survivors that involve GEX, dealer gamma or the dealer sign:**")
        w("")
        w(_md_table(["test_id", "role", "hypothesis", "rule", "sample", "raw p", "Holm study", "BY project"],
                    [[r.test_id, r.role, r.hypothesis[:150], r.sign_rule, r.sample[:60], _p(r.p_raw), _p(r.p_holm_study), _p(r.p_by_project)]
                     for r in gs.itertuples()]))
        w("")
        w("Every survivor is marked with `*` in the table of section 7.1.")
        w("")
    # primary/headline
    w("## 4. Do the studies' own primary and headline claims survive?")
    w("")
    ph = df[df["role"].isin(["primary", "headline"])]
    rows = []
    for s in [x for x in STUDY_ORDER if x in set(ph["study"])]:
        g = ph[ph["study"] == s]
        for r in g.sort_values("p_raw").itertuples():
            rows.append([r.test_id, r.role, r.pre_specified, r.hypothesis[:110], r.sign_rule, r.sample[:50], _p(r.p_raw),
                         _p(r.p_holm_study), _p(r.p_by_project)])
    w(f"{len(ph)} tests carry a primary or headline role. {int(ph['survives_holm_study_5pct'].sum())} survive Holm "
      f"within their study and {int(ph['survives_by_project_5pct'].sum())} survive BY across the project.")
    w("")
    srows = []
    for s_ in [x for x in STUDY_ORDER if x in set(ph["study"])]:
        g = ph[ph["study"] == s_]
        srows.append([f"{s_}: {STUDIES.get(s_, (s_,))[0]}", ", ".join(f"{k} {v}" for k, v in g["role"].value_counts().items()),
                      ", ".join(f"{k} {v}" for k, v in g["pre_specified"].value_counts().items()),
                      str(int((g["p_raw"] < 0.05).sum())), str(int(g["survives_holm_study_5pct"].sum())),
                      str(int(g["survives_by_project_5pct"].sum()))])
    w(_md_table(["study", "primary / headline tests", "pre-specified", "raw p < 0.05", "Holm study", "BY project"], srows))
    w("")
    w("Every primary and headline test, smallest p first within each study:")
    w("")
    w(_md_table(["test_id", "role", "pre-specified", "hypothesis", "rule", "sample", "raw p", "Holm study", "BY project"], rows))
    w("")
    # reconciliation
    w("## 5. Reconciliation with the eight findings of REAL_DATA_RESULTS.md and the registered plans")
    w("")
    w("For each finding, and for the primary test of each registered analysis plan added since (P91, P92, P96, P99, P103, P104): the "
      "ledger rows that carry it, and whether it survives BY across the whole project. Post hoc referee checks are "
      "listed under the finding they test. REAL_DATA_RESULTS.md itself is not changed here.")
    w("")
    for num_, claim, groups, extra in RECON:
        w(f"**Finding {num_}.** {claim}" if isinstance(num_, int) else f"**{num_}.** {claim}")
        w("")
        if not groups:
            w(extra)
            w("")
            continue
        parts, allg, n_any = [], [], 0
        for label, s_, pat in groups:
            g = _sel(df, s_, pat)
            allg.append(g)
            if len(g) == 0:
                parts.append(f"{label}: no matching ledger row (check the selector in the script)")
                continue
            k = int(g["survives_by_project_5pct"].sum())
            n_any += int(k > 0)
            parts.append(f"{label}: {k} of {len(g)} rows survive (smallest raw p {_p(g['p_raw'].min())}, "
                         f"smallest BY project {_p(g['p_by_project'].min())}, smallest Holm within study {_p(g['p_holm_study'].min())})")
        g = pd.concat(allg)
        if len(g):
            w(_md_table(["test_id", "role", "hypothesis", "rule", "sample", "raw p", "Holm study", "BY project"],
                        [[r.test_id, r.role, r.hypothesis[:120], r.sign_rule, r.sample[:55], _p(r.p_raw), _p(r.p_holm_study), _p(r.p_by_project)]
                         for r in g.itertuples()]))
            w("")
        for x in parts:
            w(f"* {x}.")
        if "null finding" in extra:
            n_surv = int(g["survives_by_project_5pct"].sum())
            verdict = ("**Consistent with the finding:** none of these rows is significant, even before correction."
                       if (g["p_raw"] >= 0.05).all() else
                       "**Consistent with the finding after correction:** some raw p-values are below 0.05, but none survives."
                       if n_surv == 0 else
                       f"**Not consistent with the finding:** {n_surv} of these rows survive project-wide BY.")
            w("")
            w(verdict + " " + extra)
            w("")
            continue
        n_rows = int(g["survives_by_project_5pct"].sum())
        verdict = ("**Survives project-wide BY** (every row)." if n_rows == len(g) else
                   f"**Partly survives project-wide BY:** {n_rows} of {len(g)} rows (see the lines above)." if n_any else
                   "**Does not survive project-wide BY.**")
        w("")
        w(verdict + (" " + extra if extra else ""))
        w("")
    # mismatches
    w("## 6. Cross-checks against the studies' own adjusted p-values")
    w("")
    w(f"Every adjusted p-value that a study reports was recomputed from that study's own raw p-values "
      f"({len(ck)} families or variants, `results/test_ledger_checks.csv`). Results: " +
      ", ".join(f"{k} {v}" for k, v in ck["result"].value_counts().items()) + ".")
    w("")
    bad = ck[ck["result"] != "match"]
    if len(bad):
        w(_md_table(["study", "family", "method", "tests", "mismatches", "largest difference", "note"],
                    [[r.study, r.family, r.method, str(r.m_tests), str(r.n_mismatch), _p(r.max_abs_diff), r.note] for r in bad.itertuples()]))
        w("")
    w(MISMATCH_TEXT)
    w("")
    # not counted
    w("## 7. Tables")
    w("")
    w("### 7.1 Every test with raw p below 0.05")
    w("")
    w(f"{n_sig} tests. Sorted by study, then raw p. `BY project` is the project-wide Benjamini-Yekutieli value; "
      f"`*` marks the {len(surv)} at or below 0.05 (the survivors).")
    w("")
    sig = df[df["p_raw"] < 0.05].copy()
    sig["_o"] = sig["study"].map(lambda s: STUDY_ORDER.index(s) if s in STUDY_ORDER else 99)
    sig = sig.sort_values(["_o", "p_raw"])
    w(_md_table(["test_id", "role", "hypothesis", "rule", "sample", "raw p", "BY project"],
                [[r.test_id, r.role, r.hypothesis[:120], r.sign_rule, r.sample[:50], _p(r.p_raw),
                  _p(r.p_by_project) + (" *" if r.survives_by_project_5pct else "")] for r in sig.itertuples()]))
    w("")
    w("### 7.2 What is reported but not counted")
    w("")
    w("Each line is a group of p-values or outputs left out on purpose, with the reason. Placebo draws, "
      "alternative p-values of a counted test, duplicates across files, control coefficients, diagnostics, "
      "simulations and figures are not tests.")
    w("")
    w(_md_table(["study", "file", "what", "items", "reason"],
                [[r.study, r.source_file, r.what, str(r.n_items), r.reason] for r in nc.itertuples()]))
    w("")
    if ctx.warnings:
        w("### 7.3 Warnings from the last run")
        w("")
        for x in ctx.warnings:
            w(f"* {x}")
        w("")
    w("## 8. Column guide for `results/test_ledger.csv`")
    w("")
    w("`study`, `test_id`, `hypothesis` (short plain English), `outcome`, `sample`, `sign_rule`, `estimate`, "
      "`statistic_name`, `statistic`, `p_reported` (as in the source file), `p_raw` (two-sided unless "
      "`sidedness` says one-sided and pre-stated), `sidedness`, `p_method`, `alt_p` (other p-values of the same "
      "estimate, not counted), `role`, `pre_specified`, `pre_spec_note`, `source_file`, `source_key` (the "
      "row's identifying fields in that file), `p_study_adjusted` and `study_adjust_method` (the study's own "
      "adjusted p, where it reports one), `p_study_adjusted_recomputed` and `study_adjust_check`, `p_by_project`, "
      "`p_by_study`, `p_holm_study`, `survives_by_project_5pct`, `survives_holm_study_5pct`, `doc`, `note`.")
    w("")
    text = "\n".join(L)
    text = clean(text)
    (root / "docs" / "TEST_LEDGER.md").write_text(text)


MISMATCH_TEXT = (
    "**Reading the cross-checks.** A match means the study's adjusted p-values follow from its own raw "
    "p-values exactly as its documentation says. Known differences, all explained:\n\n"
    "* **FALSIFICATION family A and the 18 headline tests** were computed from an older run of R3 and R5. "
    "The study's F2 rows use 932 days (931 with controls); the current R5 file uses 929. The p-values move a "
    "little (for example Pro binary 0.000214 then, 0.000216 now), so the study's \"Holm 0.016 among 78\" and "
    "\"Holm 0.020 among 18\" are reproduced from its own file but differ slightly when recomputed from the "
    "current R3 and R5 files. The row-level values are in the `note` column of the R3 and R5 headline rows.\n"
    "* **Family A lists six tests twice** (the regime x expiry rows and the per-SD robustness rows of R3 have "
    "identical p-values; only the scale of the coefficient differs), so its \"78 tests\" are 72 distinct tests; its Holm values are therefore slightly "
    "conservative. The ledger counts each once.\n"
    "* **Event study BH** is applied by the study to `hac_p` within each window, across all 34 comparisons "
    "including the Friday negative controls; the recomputation uses the same set.\n"
    "* **Intraday gamma, Holm within family** counts rows that repeat each other (tests 120 and 121 repeat 118; "
    "123 repeats 122), so its within-family Holm values are slightly conservative.\n"
    "* **Falsification T4 forecast** p-values are one-sided; the study's Holm and BH use them as one-sided, "
    "while the ledger doubles them because the direction was not stated in advance.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--no-doc", action="store_true", help="do not rewrite docs/TEST_LEDGER.md")
    a = ap.parse_args(argv)
    build(a.root, write=True, write_doc=not a.no_doc)


if __name__ == "__main__":
    main()
