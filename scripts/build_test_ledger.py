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

No git command is run. Commit times are read from ``.git/logs/HEAD`` only.

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

ROLES = ["primary", "headline", "secondary", "robustness", "placebo",
         "exploratory", "post hoc", "unlabelled"]

STRATEGY_NOTE = ("RESEARCH_STRATEGY_PANDEY.md (RQ1, H1) was in the repository when the session started "
                 "(reflog 19:53 UTC) and the first real-data commit is 20:28 UTC; confirm with git log")
COMMITTED_WITH_RESULTS = "study committed to git together with its results"

STUDIES = {
    "R1": ("Forecasting contest on NSE data (RQ1)", "docs/REAL_DATA_RESULTS.md (R1)"),
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
    "R1V2": ("Forecasting contest, improved models (pipeline v2)", "docs/REAL_DATA_RESULTS.md"),
}

PREREG = {
    "pinning_mechanism": ("docs/prereg/P87_PINNING_MECHANISM.md", "Pre-registration P87"),
    "f2_economics": ("docs/prereg/P90_F2_ECONOMICS.md", "Pre-registration P90"),
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
        p_raw = two_sided_from_one_sided(p_rep) if sidedness == SIDE_CONVERTED else p_rep
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
                hyp = f"{b} forecasts better than mcsGARCH (the paper's model)"
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


def adapt_r1b(ctx: Ctx):
    """Forecast diagnostics. File: results/forecast_diagnostics/tails_and_last_hour_tests.csv.
    Columns: target, model ("A vs B (sample)"), dm_two_sided_p, mean_loss_diff, n_days.
    Rows without dm_two_sided_p are tail shares (descriptive): not tests.
    """
    rel = "forecast_diagnostics/tails_and_last_hour_tests.csv"
    if not ctx.has(rel):
        return
    d = ctx.read(rel)
    t = d[d["dm_two_sided_p"].notna()]
    for _, r in t.iterrows():
        m = re.match(r"(.+?) vs (.+?) \((.+)\)", clean(r["model"]))
        a, b, s = m.groups() if m else (clean(r["model"]), "", "")
        ctx.add("R1b", f"{a} and {b} forecast the last hour equally well (QLIKE)",
                f"QLIKE loss of forecasts of {TARGETS.get(clean(r['target']), clean(r['target']))}",
                f"{s} (n = {int(num(r['n_days']))})", r["dm_two_sided_p"], role="post hoc",
                pre_specified=PS_NO, pre_spec_note="diagnostic run after the R1 results",
                source_file=rel, source_key=key(r["target"], r["model"]), estimate=r["mean_loss_diff"],
                statistic_name="mean loss difference", p_method="Diebold-Mariano, two-sided")
    ctx.skip("R1b", rel, "tail-share rows (top 1% share of QLIKE)", int(d["dm_two_sided_p"].isna().sum()),
             "descriptive shares, no p-value")
    for other in ("gains_by_subset.csv", "largest_mcsgarch_errors.csv"):
        if ctx.has(f"forecast_diagnostics/{other}"):
            ctx.consumed.add(f"forecast_diagnostics/{other}")
            ctx.skip("R1b", f"forecast_diagnostics/{other}", "descriptive table", 1, "no p-value")


def adapt_auction(ctx: Ctx):
    """Closing-auction check. Rows on all days duplicate R1, R2 and R5 and are dropped.

    r1_dm_tests.csv: target, sample, group, comparison, mean_diff, stat, p (one-sided, as R1).
    r2_tuesday_split.csv: measure, sample, pct_expiry_vs_week, p_signed_rank (two-sided Wilcoxon).
    r5_without_auction.csv: sample, spec, rule, term, coef, p (HAC, two-sided).
    """
    rel = "closing_auction/r1_dm_tests.csv"
    if ctx.has(rel):
        d = ctx.read(rel)
        keep = d[d["sample"].str.startswith("before")]
        for _, r in keep.iterrows():
            comp = clean(r["comparison"])
            if comp.startswith("GEX ablation"):
                a, b = [s.strip() for s in comp.split(":", 1)[1].split(" vs ")]
                hyp = f"Adding GEX features ({b}) lowers forecast loss against {a}, before the closing auction"
            else:
                b = comp.split(" vs ")[1].strip()
                hyp = f"{b} forecasts better than mcsGARCH, before the closing auction"
            ctx.add("AUCTION", hyp, f"QLIKE loss of forecasts of {TARGETS.get(clean(r['target']), r['target'])}",
                    f"{r['group']} test days before 3 Aug 2026 (n = {int(num(r['n']))})", r["p"],
                    role="robustness", pre_specified=PS_NO,
                    pre_spec_note="added after the closing auction was flagged; one-sided test as pre-stated for R1",
                    source_file=rel, source_key=key(r["target"], r["sample"], r["group"], comp),
                    sidedness=SIDE_ONE_PRE, estimate=r["mean_diff"], statistic_name="Diebold-Mariano (HLN) t",
                    statistic=r["stat"], p_method="Diebold-Mariano, one-sided (model better)")
        ctx.skip("AUCTION", rel, "rows on all test days", int((~d["sample"].str.startswith("before")).sum()),
                 "identical to results/pipeline_nse/tests_*.csv (counted under R1)")
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


def reflog_commits(root: Path):
    f = root / ".git" / "logs" / "HEAD"
    out = []
    if not f.exists():
        return out
    for line in f.read_text(errors="replace").splitlines():
        try:
            head, msg = line.split("\t", 1)
            parts = head.split()
            out.append((int(parts[-2]), msg))
        except (ValueError, IndexError):
            continue
    return out


def prereg_status(ctx: Ctx, folder: str):
    if folder not in PREREG:
        return PS_UNKNOWN, ""
    doc, phrase = PREREG[folder]
    if not (ctx.root / doc).exists():
        return PS_UNKNOWN, f"{doc} not found"
    commits = [ts for ts, msg in reflog_commits(ctx.root) if phrase in msg]
    files = [p for p in ctx.res(folder).rglob("*") if p.is_file()]
    if not commits:
        return PS_STATED, f"{doc} exists, but no commit '{phrase}' was found in .git/logs/HEAD"
    t_c = min(commits)
    t_r = min(p.stat().st_mtime for p in files) if files else float("inf")
    tc = datetime.fromtimestamp(t_c, timezone.utc).strftime("%d %b %Y %H:%M UTC")
    if not files:
        return PS_YES, f"{doc} committed {tc} per .git/logs/HEAD; no result files yet"
    if t_c < t_r:
        tr = datetime.fromtimestamp(t_r, timezone.utc).strftime("%H:%M UTC")
        return PS_YES, (f"{doc} committed {tc} per .git/logs/HEAD, before the earliest result file ({tr}); "
                        "confirm with git log")
    return PS_STATED, f"{doc} committed {tc}, but result files are older"


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

STUDY_ORDER = ["R1", "R1b", "AUCTION", "R2", "R3", "R4", "R5", "REPL", "PIN", "EVENT", "FALS", "DSIGN",
               "INTRA", "CONN", "SENSEX", "F2ECON", "PINMECH", "R1V2"]


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
        if list(ctx.res("pipeline_nse_v2").glob("tests_*.csv")):
            n0 = len(ctx.rows)
            adapt_r1(ctx, study="R1V2", folder="pipeline_nse_v2", pre_spec_ok=False)
            new = ctx.rows[n0:]
            del ctx.rows[n0:]
            old = {(r["source_key"], round(r["p_reported"], 12)) for r in ctx.rows if r["study"] == "R1"}
            keep = [r for r in new if (r["source_key"], round(r["p_reported"], 12)) not in old]
            ctx.rows.extend(keep)
            ctx.skip("R1V2", "pipeline_nse_v2/tests_*.csv", "rows identical to R1", len(new) - len(keep), "counted under R1")
        else:
            adapt_generic(ctx, "R1V2", "pipeline_nse_v2")
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
    (2, "A hybrid of mcsGARCH and LightGBM beats mcsGARCH at 15 and 30 minutes, on all days and on expiry days (DM p of 0.001 or below).",
     [("hybrid against mcsGARCH, all days", "R1", r"^tests_next(15|30)\.csv\|all\|mcsgarch vs hybrid_base$"),
      ("hybrid against mcsGARCH, expiry days", "R1", r"^tests_next(15|30)\.csv\|expiry\|mcsgarch vs hybrid_base$")], ""),
    (3, "Last hour: HAR is about 11% better over all days (one-sided p = 0.015), but not before the closing auction (p = 0.11).",
     [("HAR against mcsGARCH, last hour, all days", "R1", r"^tests_last_hour\.csv\|all\|mcsgarch vs har$"),
      ("the same before 3 Aug 2026", "AUCTION", r"^last_hour\|before 3 Aug 2026\|all\|mcsgarch vs har$")],
     "The finding itself says the last-hour edge is not reliable."),
    (4, "GEX features do not improve out-of-sample forecasts.",
     [("GEX ablations on expiry days (3 horizons, 4 model pairs)", "R1", r"\|expiry\|GEX ablation")],
     "This is a null finding, so nothing needs to survive; the check is that no ablation is significant."),
    (5, "Pro-based GEX: the expiry-day sign flip survives the falsification tests (Holm 0.020 among 18 headline tests) but not on Bank Nifty (p = 0.25).",
     [("R3, Pro rule, regime x expiry", "R3", r"^regime x expiry\|gex_participant_pro\|"),
      ("R3 robustness, Pro rule, share scaling (the Holm 0.020 row)", "R3", r"^participant_pro\|share\|gex_l1"),
      ("Bank Nifty replication, Pro rule, GEX x expiry", "FALS", r"^T5 Bank Nifty\|banknifty\|F1\|participant_pro\|gex_l1_x_exp, share")],
     "The Holm 0.020 quoted in the finding is within the 18 headline tests only."),
    (6, "Expiry-day premium: about +11% (p 0.045) in the multi-weekly regime, +21% (p 0.005) on Tuesdays, +43% in the Tuesday last hour (p 0.00005), about 63% with Sensex days out of the baseline.",
     [("whole day, multi-weekly and Tuesday regimes", "R2", r"^rv_day\|(multi_weekly|single_weekly_tue)$"),
      ("last hour, Tuesday regime", "R2", r"^rv_last_hour\|single_weekly_tue$"),
      ("last hour, Tuesday regime, Sensex days out of the baseline", "SENSEX", r"R2 with Sensex days out of the baseline: rv_last_hour, single_weekly_tue$")], ""),
    (7, "F2: the end-of-day reversal is stronger after negative Pro GEX (Holm 0.016 among 78; out-of-sample test p 0.006; permutation p 0.0005 to 0.010).",
     [("R5 in-sample, Pro rule, binary, all days", "R5", r"^binary\|participant_pro\|all\|r_rod_x_neg$"),
      ("'pre-registered style' out-of-sample test (one-sided as stated)", "FALS", r"^T4 pre-registered F2\|(full|pre_auction)\|F2\|participant_pro"),
      ("permutation tests, Pro rule, binary, full sample", "FALS", r"^T1 permutation\|full\|F2\|participant_pro\|x_neg_binary")],
     "The 'Holm 0.016 among 78' comes from FALSIFICATION family A, which used an older run (932 days); see section 6."),
    (8, "Dealer sign: outcome (a) points weakly to Pro (p 0.18), outcome (b) to the reverse of Pro (p 4e-6); out of sample no estimated sign beats the fixed rules after correction.",
     [("estimated direction, full sample, participants, outcome (a)", "DSIGN", r"^full sample\|base\|participants\|a$"),
      ("estimated direction, full sample, participants, outcome (b)", "DSIGN", r"^full sample\|base\|participants\|b$"),
      ("the 10 primary out-of-sample DM tests", "DSIGN", r"^all test days\|(a|b)\|signed\|est_(participants|shares)\|")],
     "For the out-of-sample part the finding is a null, so not surviving is what it says."),
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
      "run, judged from the commit times in `.git/logs/HEAD` (no git command is run; confirm with git log). "
      "That holds for RESEARCH_STRATEGY_PANDEY.md (RQ1 and H1), which was in the repository before the first "
      "real-data run, and for the P87 and P90 pre-registrations in `docs/prereg/` when their commit is older "
      f"than the study's result files. Studies with `yes` rows now: {', '.join(yes_st) if yes_st else 'none'}. "
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
        gex_studies = {"R3", "R5", "FALS", "DSIGN", "INTRA", "CONN"}
        gex_mask = (df["study"].isin(gex_studies) |
                    (df["study"].isin({"AUCTION", "PIN"}) & df["hypothesis"].str.contains("GEX|gamma", case=False, regex=True)))
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
          "30 minutes, Sensex calendar checks, volume changes in the replication. Each one makes room for the next. "
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
    w("## 5. Reconciliation with the eight findings of REAL_DATA_RESULTS.md")
    w("")
    w("For each finding: the ledger rows that carry it, and whether it survives BY across the whole project. "
      "REAL_DATA_RESULTS.md itself is not changed here.")
    w("")
    for num_, claim, groups, extra in RECON:
        w(f"**Finding {num_}.** {claim}")
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
            verdict = ("**Consistent with the finding:** none of these rows is significant, even before correction."
                       if (g["p_raw"] >= 0.05).all() else
                       "**Consistent with the finding after correction:** some raw p-values are below 0.05, but none survives.")
            w(verdict)
            w("")
            continue
        verdict = ("**Survives project-wide BY.**" if n_any == len(groups) else
                   "**Partly survives project-wide BY** (see the lines above)." if n_any else
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
