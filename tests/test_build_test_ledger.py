"""Fast tests for scripts/build_test_ledger.py: BY and Holm against statsmodels, and adapter round trips
on tiny fake result folders (no real data needed)."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_test_ledger.py"
_spec = importlib.util.spec_from_file_location("build_test_ledger", _PATH)
btl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(btl)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("method,sm", [("by", "fdr_by"), ("holm", "holm"), ("bh", "fdr_bh")])
def test_adjustments_match_statsmodels(seed, method, sm):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 400))
    # a mix of uniform nulls, small signals and exact ties
    p = np.concatenate([rng.uniform(size=m), rng.uniform(0, 1e-4, size=m // 5), np.full(3, 0.01)])
    rng.shuffle(p)
    ours = btl.p_adjust(p, method)
    ref = multipletests(p, method=sm)[1]
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("method,sm", [("by", "fdr_by"), ("holm", "holm")])
def test_adjustments_ignore_missing(method, sm):
    rng = np.random.default_rng(42)
    p = rng.uniform(size=50)
    p[[3, 17, 40]] = np.nan
    ours = btl.p_adjust(p, method)
    ok = np.isfinite(p)
    assert np.isnan(ours[~ok]).all()
    np.testing.assert_allclose(ours[ok], multipletests(p[ok], method=sm)[1], rtol=1e-12)


def test_edge_cases():
    assert np.isnan(btl.p_by([np.nan, np.nan])).all()
    assert btl.p_holm([0.2])[0] == pytest.approx(0.2)
    assert btl.p_by([0.2])[0] == pytest.approx(0.2)  # c(1) = 1
    # BY is never below BH, and Holm is never below the raw p
    p = np.random.default_rng(1).uniform(size=100)
    assert (btl.p_by(p) >= btl.p_bh(p) - 1e-15).all()
    assert (btl.p_holm(p) >= p - 1e-15).all()


def test_two_sided_conversion():
    assert btl.two_sided_from_one_sided(0.01) == pytest.approx(0.02)
    assert btl.two_sided_from_one_sided(0.99) == pytest.approx(0.02)
    assert btl.two_sided_from_one_sided(0.5) == pytest.approx(1.0)
    assert np.isnan(btl.two_sided_from_one_sided(float("nan")))


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_connectedness_adapter_round_trip(tmp_path):
    """A fake connectedness_gamma/tests.csv goes in; the ledger rows, roles and the Holm cross-check come out."""
    p = [0.04, 0.01, 0.5, 0.2]
    sec_p = np.array(p[1:])
    tests = pd.DataFrame({
        "data": "fake", "test": ["P0", "S1", "S2", "S3"], "family": ["primary", "secondary", "secondary", "secondary"],
        "contrast": ["Pro rule: total, short minus long", "a", "b", "c"], "measure": "total",
        "estimate": [1.0, 2.0, 3.0, 4.0], "perm_p": p, "hac_p": [np.nan, np.nan, 0.3, np.nan],
        "n_days_state_a": 10, "n_days_state_b": 12,
        "holm_p_secondary": [np.nan] + list(multipletests(sec_p, method="holm")[1]),
    })
    _write(tmp_path / "results" / "connectedness_gamma" / "tests.csv", tests)
    ctx = btl.Ctx(tmp_path)
    btl.adapt_conn(ctx)
    assert len(ctx.rows) == 4
    rows = {r["source_key"]: r for r in ctx.rows}
    assert rows["P0"]["role"] == "primary" and rows["S1"]["role"] == "secondary"
    assert rows["P0"]["p_raw"] == pytest.approx(0.04)
    assert rows["P0"]["pre_specified"] == btl.PS_STATED
    assert "hac_p=0.3" in rows["S2"]["alt_p"]
    assert rows["S1"]["p_study_adjusted_recomputed"] == pytest.approx(rows["S1"]["p_study_adjusted"])
    assert ctx.checks[0]["result"] == "match"


def test_sensex_adapter_detects_mismatch(tmp_path):
    """A study-reported BH value that does not follow from its raw p is flagged."""
    df = pd.DataFrame({
        "family": ["main", "(i) level", "(i) level"], "test": ["S1 main", "x", "y"], "estimate": [0.4, 0.1, 0.2],
        "ci_lo": 0, "ci_hi": 1, "p": [0.15, 0.01, 0.04], "n": 30, "main": [True, False, False],
        "mwu_p": np.nan, "placebo_p": [0.09, np.nan, np.nan], "placebo_n": 10, "placebo_independent": 3,
        "placebo_p_all_events_blocked": np.nan, "pretrend_p": [0.5, np.nan, np.nan], "mean_pre": 0, "mean_post": 0,
        "mde80": 0, "wilcoxon_p": np.nan, "bh_q_within_family": [0.15, 0.02, 0.05],  # correct BH would be 0.04, 0.04
    })
    _write(tmp_path / "results" / "sensex_confound" / "all_tests.csv", df)
    ctx = btl.Ctx(tmp_path)
    btl.adapt_sensex(ctx)
    assert len(ctx.rows) == 3
    res = {c["family"]: c["result"] for c in ctx.checks}
    assert res["family 'main' (BH within family)"] == "match"
    assert res["family '(i) level' (BH within family)"] == "MISMATCH"
    assert [r["role"] for r in ctx.rows] == ["primary", "secondary", "secondary"]
    assert "placebo_p=0.09" in ctx.rows[0]["alt_p"]


def test_one_sided_not_prestated_is_doubled(tmp_path):
    ctx = btl.Ctx(tmp_path)
    r = ctx.add("X", "h", "o", "s", 0.03, role="exploratory", pre_specified=btl.PS_NO, source_file="f.csv",
                source_key="k", sidedness=btl.SIDE_CONVERTED)
    assert r["p_reported"] == pytest.approx(0.03) and r["p_raw"] == pytest.approx(0.06)
    r = ctx.add("X", "h", "o", "s", 0.03, role="primary", pre_specified=btl.PS_YES, source_file="f.csv",
                source_key="k2", sidedness=btl.SIDE_ONE_PRE)
    assert r["p_raw"] == pytest.approx(0.03)
    assert ctx.add("X", "h", "o", "s", float("nan"), role="primary", pre_specified=btl.PS_NO,
                   source_file="f.csv", source_key="k3") is None  # no p-value, no row


def test_generic_adapter_picks_up_new_folder_and_dedups(tmp_path):
    """A study folder that appears later (here a fake pinning_mechanism) is read without a dedicated adapter."""
    df = pd.DataFrame({"family": ["primary", "secondary", "secondary", "exploratory"],
                       "test": ["C1", "C2", "C2 again", "x"], "coef": [-0.1, 0.2, 0.2, 0.3],
                       "p": [0.02, 0.3, 0.3, 0.9], "perm_p": [0.03, np.nan, np.nan, np.nan]})
    _write(tmp_path / "results" / "pinning_mechanism" / "all_tests.csv", df)
    _write(tmp_path / "results" / "pinning_mechanism" / "other.csv", pd.DataFrame({"p": [0.5]}))
    ctx = btl.Ctx(tmp_path)
    btl.adapt_generic(ctx, "PINMECH", "pinning_mechanism")
    assert len(ctx.rows) == 3  # the repeated C2 row is counted once; other.csv is ignored when all_tests.csv exists
    assert [r["role"] for r in ctx.rows] == ["primary", "secondary", "exploratory"]
    assert "perm_p=0.03" in ctx.rows[0]["alt_p"]
    assert any("generic adapter" in w for w in ctx.warnings)


def test_full_build_on_fake_project(tmp_path):
    """End to end on two fake studies: ledger columns, project-wide BY and within-study Holm."""
    tests = pd.DataFrame({"subset": ["all", "expiry"], "comparison": ["mcsgarch vs har", "GEX ablation: hybrid_base vs hybrid_gex"],
                          "mean_diff": [0.1, 0.01], "stat": [3.0, 0.5], "p": [0.001, 0.3], "n": [100, 20]})
    _write(tmp_path / "results" / "pipeline_nse" / "tests_next15.csv", tests)
    conn = pd.DataFrame({"data": "fake", "test": ["P0"], "family": ["primary"], "contrast": ["c"], "measure": "total",
                         "estimate": [1.0], "perm_p": [0.04], "hac_p": [np.nan], "n_days_state_a": 1, "n_days_state_b": 1,
                         "holm_p_secondary": [np.nan]})
    _write(tmp_path / "results" / "connectedness_gamma" / "tests.csv", conn)
    (tmp_path / "docs").mkdir()
    df, nc, ck, ctx = btl.build(tmp_path, write=True, write_doc=True, quiet=True)
    assert list(df.columns) == btl.COLUMNS
    assert len(df) == 3
    np.testing.assert_allclose(df["p_by_project"], multipletests(df["p_raw"], method="fdr_by")[1])
    r1 = df[df["study"] == "R1"]
    np.testing.assert_allclose(r1["p_holm_study"], multipletests(r1["p_raw"], method="holm")[1])
    assert (tmp_path / "results" / "test_ledger.csv").exists()
    text = (tmp_path / "docs" / "TEST_LEDGER.md").read_text()
    assert "\u2013" not in text and "\u2014" not in text
    assert df.loc[df["study"] == "R1", "sidedness"].eq(btl.SIDE_ONE_PRE).all()


# ---------------------------------------------------------------------------------------------------------
# R1V2: the current forecasting contest (results/pipeline_nse_v2) and its development runs
# ---------------------------------------------------------------------------------------------------------


def _v2_tests():
    rows = [  # subset, loss, comparison, mean_diff, stat, p, n
        ("all", "qlike", "mcsgarch vs hybrid_base", 0.02, 3.5, 0.001, 792),
        ("expiry", "qlike", "mcsgarch vs har", -0.01, -1.3, 0.9, 169),
        ("pre_auction", "qlike", "mcsgarch vs hybrid_base", 0.02, 3.0, 0.002, 729),
        ("all", "mse", "mcsgarch vs hybrid_base", 1e-11, 1.1, 0.13, 792),
        ("holdout_expiry", "qlike", "mcsgarch vs naive_last", -0.5, -9.0, 1.0, 70),
    ]
    return pd.DataFrame(rows, columns=["subset", "loss", "comparison", "mean_diff", "stat", "p", "n"])


def _v2_ablation(holm=True):
    rows = [  # family, label, group, base, alt, change_pct, dm_stat, p_one_sided
        ("hybrid: base vs base+GEX", "us", "expiry", "hybrid_base", "hybrid_gex_us", 0.4, -1.1, 0.87),
        ("hybrid: base vs base+GEX", "gross", "expiry", "hybrid_base", "hybrid_gex_gross", -0.3, 1.2, 0.12),
        ("hybrid: base vs base+GEX", "placebo", "expiry", "hybrid_base", "hybrid_gex_placebo", 0.7, -2.5, 0.99),
        ("hybrid: base vs base+GEX", "us", "all", "hybrid_base", "hybrid_gex_us", -0.2, 0.6, 0.27),
        ("linear: base vs base+GEX", "us", "expiry", "linear_base", "linear_gex_us", 1.5, -1.2, 0.88),
        ("hybrid: unsigned vs signed", "us", "expiry", "hybrid_gex_gross", "hybrid_gex_us", 0.4, -1.0, 0.85),
        ("hybrid: base vs base+GEX", "us", "holdout", "hybrid_base", "hybrid_gex_us", -1.1, 2.4, 0.009),
    ]
    d = pd.DataFrame(rows, columns=["family", "label", "group", "base", "alt", "change_pct", "dm_stat", "p_one_sided"])
    d.insert(5, "days", d["group"].map({"expiry": 169, "all": 792, "holdout": 330}))
    if holm:
        d["p_holm_in_family"] = np.nan
        for _, g in d.groupby(["family", "group"]):
            h = g["label"] != "placebo"
            d.loc[g.index[h], "p_holm_in_family"] = multipletests(g.loc[h, "p_one_sided"], method="holm")[1]
    return d


def _write_v2(root, holm=True):
    v2 = root / "results" / "pipeline_nse_v2"
    _write(v2 / "tests_next15.csv", _v2_tests())
    _write(v2 / "ablation_next15.csv", _v2_ablation(holm))
    _write(v2 / "ablation_next15_mse.csv", _v2_ablation(holm=False).iloc[[0]].assign(change_pct=-3.3, dm_stat=1.04, p_one_sided=0.149))
    _write(v2 / "mcs_next15_all.csv", pd.DataFrame({"Model name": ["a"], "Pvalue": [0.5], "in_mcs": [True]}))
    return v2


def test_v2_adapter_labels_roles_and_sidedness(tmp_path):
    """Loss, subset label, role, pre-specification and sidedness of each kind of v2 row; Holm cross-check."""
    _write_v2(tmp_path)
    ctx = btl.Ctx(tmp_path)
    n = btl.adapt_r1v2(ctx)
    assert n == len(ctx.rows) == 5 + 7 + 1
    rows = {r["source_key"]: r for r in ctx.rows}
    head = rows["tests_next15.csv|qlike|all|mcsgarch vs hybrid_base"]
    assert head["role"] == "headline" and head["pre_specified"] == btl.PS_NO
    assert head["sidedness"] == btl.SIDE_CONVERTED and head["p_raw"] == pytest.approx(0.002)
    assert head["outcome"].startswith("QLIKE") and head["sample"].startswith("all test days (n = 792 days)")
    assert "observed: hybrid_base better" in head["hypothesis"]
    har = rows["tests_next15.csv|qlike|expiry|mcsgarch vs har"]
    assert har["role"] == "secondary" and har["p_raw"] == pytest.approx(0.2) and "observed: har worse" in har["hypothesis"]
    pre = rows["tests_next15.csv|qlike|pre_auction|mcsgarch vs hybrid_base"]
    assert pre["role"] == "robustness" and "before the closing auction" in pre["sample"]
    mse = rows["tests_next15.csv|mse|all|mcsgarch vs hybrid_base"]
    assert mse["role"] == "robustness" and mse["outcome"].startswith("MSE")  # MSE is never labelled QLIKE
    naive = rows["tests_next15.csv|qlike|holdout_expiry|mcsgarch vs naive_last"]
    assert naive["sample"].startswith("hold-out, 1 Jul 2025 onwards, expiry days only (n = 70 days)")
    assert naive["p_raw"] == 0.0 and "machine precision" in naive["note"]
    ab = {(r["source_key"].split("|")[0], r["source_key"].split("|")[1], r["source_key"].split("|")[2],
           r["source_key"].split("|")[3]): r for r in ctx.rows if r["source_key"].startswith("ablation")}
    h1 = ab[("ablation_next15.csv", "hybrid: base vs base+GEX", "us", "expiry")]
    assert h1["role"] == "headline" and h1["pre_specified"] == btl.PS_PARTLY
    assert h1["sidedness"] == btl.SIDE_ONE_PRE and h1["p_raw"] == pytest.approx(0.87)  # H1: one-sided, kept
    assert h1["sign_rule"] == "US convention" and "H1" in h1["hypothesis"]
    assert h1["p_study_adjusted_recomputed"] == pytest.approx(h1["p_study_adjusted"])
    assert ab[("ablation_next15.csv", "hybrid: base vs base+GEX", "gross", "expiry")]["role"] == "headline"
    plc = ab[("ablation_next15.csv", "hybrid: base vs base+GEX", "placebo", "expiry")]
    assert plc["role"] == "placebo" and plc["pre_specified"] == btl.PS_NO and plc["sidedness"] == btl.SIDE_CONVERTED
    alld = ab[("ablation_next15.csv", "hybrid: base vs base+GEX", "us", "all")]
    assert alld["role"] == "secondary" and alld["p_raw"] == pytest.approx(0.54)
    lin = ab[("ablation_next15.csv", "linear: base vs base+GEX", "us", "expiry")]
    assert lin["role"] == "secondary" and lin["pre_specified"] == btl.PS_PARTLY
    sign = ab[("ablation_next15.csv", "hybrid: unsigned vs signed", "us", "expiry")]
    assert sign["role"] == "secondary" and sign["pre_specified"] == btl.PS_NO and sign["p_raw"] == pytest.approx(0.3)
    assert ab[("ablation_next15.csv", "hybrid: base vs base+GEX", "us", "holdout")]["role"] == "robustness"
    m = ab[("ablation_next15_mse.csv", "hybrid: base vs base+GEX", "us", "expiry")]
    assert m["role"] == "robustness" and m["pre_specified"] == btl.PS_PARTLY and m["outcome"].startswith("MSE")
    holm = [c for c in ctx.checks if c["source_file"].endswith("ablation_next15.csv")]
    assert len(holm) == 1 and holm[0]["result"] == "match" and holm[0]["m_tests"] == 6
    assert any("Model Confidence Set" in x["what"] for x in ctx.not_counted)


def test_v2_ablation_row_repeating_a_tests_row_is_counted_once(tmp_path):
    v2 = _write_v2(tmp_path, holm=False)
    extra = pd.DataFrame([("odd family", "x", "all", "mcsgarch", "hybrid_base", 792, -5.8, 3.5, 0.001)],
                         columns=["family", "label", "group", "base", "alt", "days", "change_pct", "dm_stat", "p_one_sided"])
    _write(v2 / "ablation_next15.csv", pd.concat([_v2_ablation(holm=False), extra], ignore_index=True))
    ctx = btl.Ctx(tmp_path)
    assert btl.adapt_r1v2(ctx) == 5 + 7 + 1
    nc = [x for x in ctx.not_counted if "repeat a tests_*.csv comparison" in x["what"]]
    assert len(nc) == 1 and nc[0]["n_items"] == 1


def test_v2_and_first_run_both_counted(tmp_path):
    """R1 (first run) and R1V2 are different forecasts: a v2 row is not dropped because R1 has the same key."""
    v1 = pd.DataFrame({"subset": ["all"], "comparison": ["mcsgarch vs hybrid_base"], "mean_diff": [0.02],
                       "stat": [3.5], "p": [0.001], "n": [792]})
    _write(tmp_path / "results" / "pipeline_nse" / "tests_next15.csv", v1)
    _write_v2(tmp_path)
    df, nc, ck, ctx = btl.build(tmp_path, write=False, write_doc=False, quiet=True)
    assert (df["study"] == "R1").sum() == 1 and (df["study"] == "R1V2").sum() == 13
    assert not any("identical to R1" in str(x) for x in nc.get("what", []))


def test_v2_development_runs_counted_as_exploratory(tmp_path):
    dev = tmp_path / "results" / "pipeline_nse_v2" / "development"
    sb = pd.DataFrame({"group": ["all", "all", "all", "expiry"], "model": ["mcsgarch", "har", "har", "hybrid_base"],
                       "days": [462, 462, 462, 99], "qlike_vs_mcsgarch_pct": [0.0, 9.0, 9.0, -4.0],
                       "dm_p_vs_mcsgarch": [np.nan, 0.99, 0.99, 0.02], "run": "dev2",
                       "target": ["next15", "last_hour", "last_hour_pooled", "next15"]})
    _write(dev / "dev_scoreboards.csv", sb)
    ab = _v2_ablation(holm=False).iloc[[0, 2]].assign(target="next15")
    _write(dev / "dev_ablation_single_seed.csv", ab)
    sn = pd.DataFrame({"model": ["hybrid_base", "hybrid_base"], "comparison": ["seed 0", "seed 0 vs seed 1"],
                       "qlike_vs_mcsgarch_pct": [-7.8, np.nan], "abs_qlike_change_pct": [np.nan, 0.3],
                       "dm_p_two_sided": [np.nan, 0.18], "change_pct": np.nan, "p_one_sided": np.nan})
    _write(dev / "dev_seed_noise_next30.csv", sn)
    _write(dev / "dev_implied_checks.csv", pd.DataFrame({"sample": ["all days"], "value": [1.0]}))
    ctx = btl.Ctx(tmp_path)
    n = btl.adapt_r1v2_dev(ctx)
    assert n == len(ctx.rows) == 2 + 2 + 1  # the har row repeated under both last-hour schemes is counted once
    roles = sorted(r["role"] for r in ctx.rows)
    assert roles == ["exploratory", "exploratory", "exploratory", "placebo", "placebo"]
    h1 = [r for r in ctx.rows if "H1" in r["hypothesis"]]
    assert len(h1) == 1 and h1[0]["pre_specified"] == btl.PS_PARTLY and h1[0]["sidedness"] == btl.SIDE_ONE_PRE
    seed = [r for r in ctx.rows if "seed 0 vs seed 1" in r["hypothesis"]][0]
    assert seed["sidedness"] == btl.SIDE_TWO and seed["p_raw"] == pytest.approx(0.18)
    assert "63 days" in [r for r in ctx.rows if r["source_key"].endswith("|har")][0]["hypothesis"]
    assert "development/dev_implied_checks.csv" in "".join(x["source_file"] for x in ctx.not_counted)
    assert "pipeline_nse_v2/development/dev_scoreboards.csv" in ctx.consumed


def test_auction_rows_repeating_a_contest_test_are_counted_once(tmp_path):
    """closing_auction/r1_dm_tests.csv rows equal to an R1 or R1V2 test (up to rounding) are not counted again."""
    v1 = pd.DataFrame({"subset": ["all"], "comparison": ["mcsgarch vs har"], "mean_diff": [0.03], "stat": [2.2],
                       "p": [0.015], "n": [792]})
    _write(tmp_path / "results" / "pipeline_nse" / "tests_last_hour.csv", v1)
    v2 = pd.DataFrame({"subset": ["pre_auction"], "loss": ["qlike"], "comparison": ["mcsgarch vs hybrid_base"],
                       "mean_diff": [0.02], "stat": [3.0], "p": [0.00165], "n": [729]})
    _write(tmp_path / "results" / "pipeline_nse_v2" / "tests_last_hour.csv", v2)
    au = pd.DataFrame({"target": "last_hour", "sample": ["all test days", "before 3 Aug 2026", "before 3 Aug 2026"],
                       "group": "all", "comparison": ["mcsgarch vs har", "mcsgarch vs hybrid_base", "mcsgarch vs har"],
                       "mean_diff": [0.03, 0.02 + 1e-17, 0.01], "stat": [2.2, 3.0, 1.2],
                       "p": [0.015, 0.00165 + 1e-16, 0.11], "n": [792, 729, 729]})
    _write(tmp_path / "results" / "closing_auction" / "r1_dm_tests.csv", au)
    ctx = btl.Ctx(tmp_path)
    btl.adapt_auction(ctx)
    assert len(ctx.rows) == 1 and "har" in ctx.rows[0]["hypothesis"]
    reasons = " ".join(x["reason"] for x in ctx.not_counted)
    assert "counted under R1)" in reasons and "counted under R1V2)" in reasons and not ctx.warnings


# ---------------------------------------------------------------------------------------------------------
# Pre-registration status from the reflogs (also inside a git worktree), and the doubled max-type p
# ---------------------------------------------------------------------------------------------------------

T_COMMIT = 1_700_000_000  # Nov 2023: older than any file the tests write


def _reflog_line(ts, msg):
    return f"{'0' * 40} {'1' * 40} A Author <a@example.com> {ts} +0000\t{msg}\n"


def _fake_repo(root, phrase, ts=T_COMMIT, doc=None):
    """A plain checkout whose reflog (logs/HEAD) holds a commit with the given message."""
    logs = root / ".git" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "HEAD").write_text(_reflog_line(ts, f"commit: {phrase}"))
    if doc:
        (root / doc).parent.mkdir(parents=True, exist_ok=True)
        (root / doc).write_text("plan")


def test_doubled_max_type_p(tmp_path):
    """SPA-style one-sided max p: min(1, 2p), not 2 min(p, 1 - p) (a p near 1 is not evidence of the other direction)."""
    ctx = btl.Ctx(tmp_path)
    kw = dict(role="exploratory", pre_specified=btl.PS_NO, source_file="f.csv", sidedness=btl.SIDE_DOUBLED)
    assert ctx.add("X", "h", "o", "s", 0.03, source_key="a", **kw)["p_raw"] == pytest.approx(0.06)
    r = ctx.add("X", "h", "o", "s", 0.8, source_key="b", **kw)
    assert r["p_reported"] == pytest.approx(0.8) and r["p_raw"] == 1.0


def test_reflog_read_through_a_worktree(tmp_path):
    """A worktree's .git is a file; its commits are in the worktree's HEAD log and in the branch logs (nested
    branch names) of the main repository's git directory."""
    main_git = tmp_path / "main" / ".git"
    wt_git = main_git / "worktrees" / "wt"
    (wt_git / "logs").mkdir(parents=True)
    (wt_git / "commondir").write_text("../..\n")
    (wt_git / "logs" / "HEAD").write_text(_reflog_line(T_COMMIT + 5, "commit: work in the worktree"))
    branch_log = main_git / "logs" / "refs" / "heads" / "claude" / "thread"
    branch_log.parent.mkdir(parents=True)
    branch_log.write_text(_reflog_line(T_COMMIT, "commit: Pre-registration P92") + "not a reflog line\n")
    (main_git / "logs" / "HEAD").write_text(_reflog_line(T_COMMIT, "commit: Pre-registration P92"))
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {wt_git}\n")
    assert [p.resolve() for p in btl.git_dirs(wt)] == [wt_git.resolve(), main_git.resolve()]
    commits = btl.reflog_commits(wt)
    assert sorted(commits) == [(T_COMMIT, "commit: Pre-registration P92"), (T_COMMIT + 5, "commit: work in the worktree")]
    # the plan is older than the outcome files, but a run file written with the plan is older still
    (wt / "docs" / "prereg").mkdir(parents=True)
    (wt / "docs" / "prereg" / "P92_SENSEX_SWAP.md").write_text("plan")
    _write(wt / "results" / "sensex_swap" / "tests.csv", pd.DataFrame({"p": [0.5]}))
    _write(wt / "results" / "sensex_swap" / "coverage.csv", pd.DataFrame({"n": [1]}))
    os.utime(wt / "results" / "sensex_swap" / "coverage.csv", (T_COMMIT - 100, T_COMMIT - 100))
    ctx = btl.Ctx(wt)
    ps, note = btl.prereg_status(ctx, "sensex_swap", ["tests.csv"])
    assert ps == btl.PS_YES and "per the reflogs" in note and "outcome file" in note
    assert btl.prereg_status(ctx, "sensex_swap")[0] == btl.PS_STATED  # every result file counts without the globs
    assert btl.git_dirs(tmp_path / "nowhere") == []


# ---------------------------------------------------------------------------------------------------------
# P92 (sensex_swap) and P96 (synthetic_control)
# ---------------------------------------------------------------------------------------------------------


def _swap_tests():
    p_sec = [0.02, 0.04, 0.5]
    return pd.DataFrame({
        "test": ["P1", "S1", "S2", "S3", "PL1"], "family": ["primary", "secondary", "secondary", "secondary", "placebo"],
        "what": ["Tue minus Thu NET, after minus before", "last hour", "regression", "total", "fake date"],
        "outcome": ["net_day", "net_lh", "net_day", "tot_day", "net_day"],
        "alternative": ["greater", "greater", "two-sided", "less", "two-sided"],
        "estimate": [0.5, 0.7, 0.3, -0.2, 0.1], "block_weeks": 4,
        "perm_p": [0.33, 0.02, np.nan, 0.5, 0.7], "perm_p_greater": [0.33, 0.02, np.nan, 0.52, 0.35],
        "perm_p_less": [0.67, 0.98, np.nan, 0.5, 0.65], "hac_p": [0.4, 0.03, 0.04, 0.6, 0.8],
        "p_used": [0.33, 0.02, 0.04, 0.5, 0.7], "n_days": 300, "n_cells": "70/80/70/80",
        "holm_p_secondary": [np.nan] + list(multipletests(p_sec, method="holm")[1]) + [np.nan],
    })


def test_swap_adapter_roles_alternatives_and_holm(tmp_path):
    base = tmp_path / "results" / "sensex_swap"
    _write(base / "tests.csv", _swap_tests())
    _write(base / "robustness.csv", pd.DataFrame({
        "check": ["block of 1 week(s)", "drop September 2025", "total connectedness (descriptive)"],
        "estimate": [0.5, 0.4, 1.0], "perm_p": [0.31, 0.2, 0.05], "hac_p": [0.4, 0.3, 0.06], "n_days": 300,
        "alternative_used": ["greater", "greater", "two-sided"]}))
    _write(base / "h2_regression.csv", pd.DataFrame({"term": ["expiry"], "coef": [0.3], "p": [0.04]}))
    _write(base / "coverage.csv", pd.DataFrame({"n": [1]}))
    _fake_repo(tmp_path, "Pre-registration P92", doc="docs/prereg/P92_SENSEX_SWAP.md")
    ctx = btl.Ctx(tmp_path)
    btl.adapt_swap(ctx)
    rows = {r["source_key"]: r for r in ctx.rows}
    assert len(ctx.rows) == 7 and not ctx.warnings
    p1 = rows["tests.csv|P1"]
    assert p1["role"] == "primary" and p1["pre_specified"] == btl.PS_YES and p1["sidedness"] == btl.SIDE_ONE_PRE
    assert p1["p_raw"] == pytest.approx(0.33)  # registered one-sided: kept
    assert "perm_p_block_1_week=0.31" in p1["alt_p"] and "perm_p_other_direction=0.67" in p1["alt_p"]
    assert rows["tests.csv|S2"]["p_raw"] == pytest.approx(0.04) and "Newey-West" in rows["tests.csv|S2"]["p_method"]
    assert rows["tests.csv|PL1"]["role"] == "placebo" and rows["tests.csv|PL1"]["sidedness"] == btl.SIDE_TWO
    s1 = rows["tests.csv|S1"]
    assert s1["p_study_adjusted_recomputed"] == pytest.approx(s1["p_study_adjusted"])
    assert [c["result"] for c in ctx.checks] == ["match"]
    assert rows["robustness.csv|drop September 2025"]["role"] == "robustness"
    desc = rows["robustness.csv|total connectedness (descriptive)"]
    assert desc["role"] == "exploratory" and desc["pre_specified"] == btl.PS_NO and desc["sidedness"] == btl.SIDE_TWO
    nc = {x["what"]: x for x in ctx.not_counted}
    assert nc["block lengths of 1, 2 and 8 weeks and the within-weekday permutation"]["n_items"] == 1
    assert {"sensex_swap/h2_regression.csv", "sensex_swap/coverage.csv"} <= ctx.consumed


def test_synth_adapter_ratio_did_joint_and_dedup(tmp_path):
    base = tmp_path / "results" / "synthetic_control"
    rows = [  # analysis, treated, effect, ratio, p_ratio, p_neg, did, did_hac_p, did_rank_p_neg
        ("primary", "BN", -0.02, 1.1, 0.95, 0.4, 0.05, 0.01, 0.9),
        ("secondary", "FIN", 0.01, 1.5, 0.3, 0.6, 0.06, 0.2, 0.8),
        ("secondary", "MID", 0.03, 1.2, 0.6, 0.7, 0.07, 0.3, 0.7),
        ("S7 drop the last month", "MID", 0.03, 1.2, 0.6, 0.7, 0.07, 0.3, 0.7),  # same as MID's main run
        ("placebo in time (2023)", "BN", 0.002, 0.9, 0.5, 0.5, 0.01, 0.6, 0.4),
        ("descriptive", "NIFTY", 0.04, 2.0, 0.2, 0.9, 0.08, 0.05, 0.95),
    ]
    d = pd.DataFrame(rows, columns=["analysis", "treated", "effect", "ratio", "p_ratio", "p_neg", "did", "did_hac_p", "did_rank_p_neg"])
    _write(base / "results.csv", d.assign(n_pre=100, n_post=50, J=19))
    (base / "decision.json").write_text(
        '{"secondaries": {"FIN": {"p_ratio": 0.3, "holm_p": 0.6}, "MID": {"p_ratio": 0.6, "holm_p": 0.6}},'
        ' "joint_test_all_three": {"p_neg": 0.966, "n_combinations": 6859, "mean_std_effect": 0.2}}')
    _write(base / "placebo_space_BN.csv", pd.DataFrame({"donor": ["x"], "ratio": [1.0]}))
    ctx = btl.Ctx(tmp_path)
    btl.adapt_synth(ctx)
    assert len(ctx.rows) == 6 + 6 - 2 + 1
    rows = {r["source_key"]: r for r in ctx.rows}
    bn = rows["ratio|primary|BN"]
    assert bn["role"] == "primary" and bn["sidedness"] == btl.SIDE_TWO and bn["p_raw"] == pytest.approx(0.95)
    assert "p_signed_one_sided_fall=0.4" in bn["alt_p"]
    fin = rows["ratio|secondary|FIN"]
    assert fin["p_study_adjusted"] == pytest.approx(0.6) and fin["p_study_adjusted_recomputed"] == pytest.approx(0.6)
    assert rows["ratio|placebo in time (2023)|BN"]["role"] == "placebo"
    assert rows["ratio|descriptive|NIFTY"]["role"] == "exploratory" and rows["ratio|descriptive|NIFTY"]["pre_specified"] == btl.PS_NO
    assert "ratio|S7 drop the last month|MID" not in rows and "did|S7 drop the last month|MID" not in rows
    assert all(r["role"] == "exploratory" and r["pre_specified"] == btl.PS_NO for k, r in rows.items() if k.startswith("did|"))
    joint = rows["joint_test_all_three"]
    assert joint["role"] == "secondary" and joint["sidedness"] == btl.SIDE_ONE_STATED and joint["p_raw"] == pytest.approx(0.966)
    assert joint["pre_specified"] == btl.PS_UNKNOWN  # no plan in this fake project
    assert [c["result"] for c in ctx.checks] == ["match"]
    assert [x["n_items"] for x in ctx.not_counted if x["what"].startswith("rows repeating")] == [2]
    assert "synthetic_control/placebo_space_BN.csv" in ctx.consumed


# ---------------------------------------------------------------------------------------------------------
# Referee responses (post hoc): F2, forecasting, expiry premia
# ---------------------------------------------------------------------------------------------------------


def _f2_group(rule, sample, coef, nw5, others, n=932):
    methods = {"newey_west_5": nw5, **others}
    return pd.DataFrame({"rule": rule, "spec": "binary", "outcome": "r_last", "sample": sample, "term": "N_x_r_rod",
                         "coef": coef, "n": n, "n_neg": 300, "method": list(methods), "p": list(methods.values())})


def test_ref_f2_one_row_per_estimate(tmp_path):
    base = tmp_path / "results" / "referee_f2"
    robust = {"driscoll_kraay": 0.002, "wild_cluster_month": 0.012, "block_perm_t": 0.008}
    item3 = pd.concat([_f2_group("pro", "full", -0.5, 0.001, robust),
                       _f2_group("pro", "2023_2024", np.nan, np.nan, {"wild_cluster_month": np.nan}, n=400)])
    _write(base / "item3_f2_inference.csv", item3)
    item4 = pd.concat([_f2_group("pro", "full", -0.5, 0.001, {"wild_cluster_month": 0.011}),  # same estimate as item 3
                       _f2_group("fii", "full", 0.2, 0.3, {"wild_cluster_month": 0.45, "driscoll_kraay": 0.9})])
    _write(base / "item4_other_rules.csv", item4)
    _write(base / "item8_horse_race.csv", pd.DataFrame({
        "horse_race": "raw states", "position": "all", "form": "joint", "coef": np.nan, "n": 932,
        "method": ["newey_west_5", "wild_cluster_month"], "p": [0.01, 0.2]}))
    _write(base / "item2_sign_shares.csv", pd.DataFrame({"share": [0.4]}))
    ctx = btl.Ctx(tmp_path)
    btl.adapt_ref_f2(ctx)
    assert len(ctx.rows) == 3
    rows = {r["source_key"]: r for r in ctx.rows}
    f2 = rows["item3_f2_inference.csv|pro|binary|r_last|full|N_x_r_rod"]
    # Driscoll-Kraay (0.9 for fii) and Newey-West 5 are comparisons only, never the main p
    assert f2["p_raw"] == pytest.approx(0.012) and "wild_cluster_month" in f2["p_method"]
    assert "newey_west_5=0.001" in f2["alt_p"] and f2["role"] == "exploratory" and f2["pre_specified"] == btl.PS_NO
    assert rows["item4_other_rules.csv|fii|binary|r_last|full|N_x_r_rod"]["p_raw"] == pytest.approx(0.45)
    hr = rows["item8_horse_race|raw states|all|joint"]
    assert hr["sidedness"] == btl.SIDE_NONDIR and hr["p_raw"] == pytest.approx(0.2)
    assert ctx.ref_f2_counts == ({"1-7": 4 + 2 + 3, "8": 2}, {"1-7": 2, "8": 1})  # p-values seen, tests counted
    nc = {x["what"]: x for x in ctx.not_counted}
    assert nc["groups marked 'not estimable'"]["n_items"] == 1
    assert nc["estimates already counted from another file"]["n_items"] == 1
    assert "referee_f2/item2_sign_shares.csv" in ctx.consumed


def test_ref_forecast_spa_fair_benchmarks_and_gw(tmp_path):
    _write(tmp_path / "results" / "pipeline_nse_v2" / "tests_next15.csv", _v2_tests())
    base = tmp_path / "results" / "referee_forecast"
    _write(base / "spa_reality_check.csv", pd.DataFrame({
        "target": "next15", "subset": "all", "mean_block": [5, 10, 20], "p_spa_consistent": [0.02, 0.03, 0.04],
        "p_spa_lower": 0.01, "p_spa_upper": 0.05, "p_reality_check": [0.06, 0.07, 0.08], "n_challengers": 12,
        "best_by_t": "hybrid_base", "days": 792, "best_by_t_pct_vs_mcsgarch": -6.0, "T_spa": 3.1, "reps": 2000}))
    fb = pd.DataFrame({
        "target": "next15", "subset": "all", "benchmark": ["mcsgarch", "vix", "best_matched_mcsgarch"],
        "benchmark_model": ["mcsgarch", "mcsgarch_vix", "mcsgarch_vix"], "p_benchmark_beats_mcsgarch": [np.nan, 0.2, 0.2],
        "benchmark_pct_vs_mcsgarch": [0.0, -1.0, -1.0], "days": 792,
        "p_hybrid_beats_benchmark": [0.001, 0.01, 0.01],  # the first is the R1V2 test (same p): not counted again
        "hybrid_pct_vs_benchmark": [-5.0, -3.0, -3.0], "dm_stat": [3.5, 2.3, 2.3]})
    _write(base / "fair_benchmarks.csv", fb)
    gw = pd.DataFrame({
        "target": "next15", "benchmark": "mcsgarch", "benchmark_model": "mcsgarch",
        "test": ["unconditional", "conditional C1", "conditional C1 (Newey-West)"], "family": ["main", "main", "newey_west"],
        "instruments": ["constant", "constant and lagged difference", "constant and lagged difference"], "n": 792,
        "p": [0.01, 0.04, 0.05], "mean_d_points": [0.2, 0.21, 0.21], "df": [1, 2, 2], "stat": [6.6, 6.4, 6.0]})
    gw["p_holm_main_family"] = [0.02, 0.04, np.nan]
    _write(base / "gw_tests.csv", gw)
    _write(base / "mcs_summary.csv", pd.DataFrame({"model": ["a"], "p": [0.5]}))
    ctx = btl.Ctx(tmp_path)
    btl.adapt_ref_forecast(ctx)
    assert len(ctx.rows) == 1 + 2 + 2
    rows = {r["source_key"]: r for r in ctx.rows}
    spa = rows["spa_reality_check.csv|next15|all"]
    assert spa["sidedness"] == btl.SIDE_DOUBLED and spa["p_raw"] == pytest.approx(0.06)
    assert "p_spa_block_5=0.02" in spa["alt_p"] and "p_reality_check=0.07" in spa["alt_p"]
    vix = rows["fair_benchmarks.csv|next15|all|vix|vs mcsgarch"]
    assert vix["sidedness"] == btl.SIDE_CONVERTED and vix["p_raw"] == pytest.approx(0.4)
    assert rows["fair_benchmarks.csv|next15|all|vix|hybrid"]["p_raw"] == pytest.approx(0.02)
    c1 = rows["gw_tests.csv|next15|mcsgarch|conditional C1"]
    assert c1["sidedness"] == btl.SIDE_NONDIR and "p_newey_west=0.05" in c1["alt_p"]
    assert c1["p_study_adjusted_recomputed"] == pytest.approx(0.04)
    assert all(r["role"] == "exploratory" and r["pre_specified"] == btl.PS_NO for r in ctx.rows)
    assert [c["result"] for c in ctx.checks] == ["match"]
    nc = {(x["source_file"], x["what"]): x["n_items"] for x in ctx.not_counted}
    assert nc[("results/referee_forecast/fair_benchmarks.csv", "rows that repeat a test of the current contest")] == 1
    assert nc[("results/referee_forecast/", "rows repeating another row of this study (same p and estimate)")] == 2
    assert nc[("results/referee_forecast/spa_reality_check.csv", "SPA and Reality Check with blocks of 5 and 20 days")] == 2
    assert nc[("results/referee_forecast/gw_tests.csv", "Newey-West versions of C1 and C2")] == 1
    assert "referee_forecast/mcs_summary.csv" in ctx.consumed


def test_ref_expiry_repeats_and_trend_rows(tmp_path):
    ctx = btl.Ctx(tmp_path)
    ctx.add("EVENT", "published row", "o", "s", 0.0123, role="secondary", pre_specified=btl.PS_NO,
            source_file="event_study/tests.csv", source_key="e", estimate=0.05)
    base = tmp_path / "results" / "referee_expiry"
    _write(base / "negative_control_rows.csv", pd.DataFrame({
        "spec_id": ["a_Fri", "b_Thu"], "comparison": ["Negative control: Nifty Friday", "Nifty Thursday"],
        "event": ["a", "b"], "measure": "rv_day", "window": "8 weeks", "n_pre": 8, "n_post": 8,
        "hac_p": [0.0123, 0.04], "diff": [0.05, 0.1], "placebo_p": [0.1, 0.2],
        "trend_adjusted_p": [0.3, 0.2], "trend_adjusted_change": [0.02, 0.03], "pretrend_p": [0.5, 0.6]}))
    _write(base / "pooled_all_index_regression.csv", pd.DataFrame({
        "index": "NIFTY", "term": ["own_thu_to_a", "x_bn"], "measure": "rv_day", "spec": "week fixed effects",
        "n_days": 900, "n_weeks": 180, "p": [0.01, 0.001], "coef": [0.2, 0.1]}))
    _write(base / "expiry_calendar.csv", pd.DataFrame({"date": ["2024-01-04"]}))
    btl.adapt_ref_expiry(ctx)
    new = [r for r in ctx.rows if r["study"] == "REFEXP"]
    assert len(new) == 4  # the Friday HAC row repeats the EVENT row; its trend-adjusted test is new
    fri = [r for r in new if r["hypothesis"].startswith("Negative control (Friday): ")]
    assert len(fri) == 1 and "trend" in fri[0]["hypothesis"] and "Negative control:" not in fri[0]["hypothesis"]
    assert fri[0]["p_raw"] == pytest.approx(0.3)
    thu = [r for r in new if r["source_key"].endswith("b_Thu|rv_day|8 weeks|")][0]
    assert "placebo_p=0.2" in thu["alt_p"]
    assert all(r["role"] == "exploratory" and r["pre_specified"] == btl.PS_NO for r in new)
    nc = {x["what"]: x["n_items"] for x in ctx.not_counted}
    assert nc["rows that repeat a test of EVENT"] == 1 and nc["other p-values of those repeated tests"] == 1
    assert nc["pre-trend p-values (pretrend_p)"] == 2 and nc["other indices' expiry dummies"] == 1
    assert "referee_expiry/expiry_calendar.csv" in ctx.consumed


# ---------------------------------------------------------------------------------------------------------
# P91: the forecasting pipeline on Bank Nifty
# ---------------------------------------------------------------------------------------------------------


def test_bnfc_absent_folder_is_a_note_not_an_error(tmp_path):
    ctx = btl.Ctx(tmp_path)
    assert btl.adapt_bnfc(ctx) == 0
    assert not ctx.rows and not ctx.warnings
    assert len(ctx.not_counted) == 1 and "registered run not finished" in ctx.not_counted[0]["reason"]


def test_bnfc_on_a_copy_of_the_v2_layout(tmp_path):
    """The v2 files under pipeline_nse_banknifty: the plan's H1 to H4 keep their one-sided p and its status;
    everything else is doubled and not pre-specified; the regime split and other ablation groups are exploratory."""
    v2 = _write_v2(tmp_path)
    bn = tmp_path / "results" / "pipeline_nse_banknifty"
    bn.mkdir(parents=True)
    for f in v2.iterdir():
        (bn / f.name).write_bytes(f.read_bytes())
    t30 = _v2_tests().assign(mean_diff=lambda d: d["mean_diff"] * 1.1, stat=lambda d: d["stat"] * 1.1)
    t30.loc[0, "p"] = 0.004
    t30.loc[len(t30)] = ("multi_weekly_expiry", "qlike", "mcsgarch vs hybrid_base", 0.03, 2.0, 0.03, 60)
    _write(bn / "tests_next30.csv", t30)
    _write(bn / "p91" / "hypotheses.csv", pd.DataFrame({
        "hypothesis": ["H1", "H1"], "target": ["next15", "next30"], "p_one_sided": [0.001, 0.004], "p_holm": [0.002, 0.004]}))
    _write(bn / "daily_market.csv", pd.DataFrame({"date": ["2024-01-01"]}))
    _fake_repo(tmp_path, "P91 pre-registration", doc="docs/prereg/P91_BANKNIFTY_FORECAST.md")
    ctx = btl.Ctx(tmp_path)
    n = btl.adapt_bnfc(ctx)
    new = [r for r in ctx.rows if r["study"] == "BNFC"]
    assert n == len(new) == 5 + 6 + 7 + 1 and not ctx.warnings
    rows = {r["source_key"]: r for r in new}
    for t, p, ph in (("next15", 0.001, 0.002), ("next30", 0.004, 0.004)):
        h1 = rows[f"tests_{t}.csv|qlike|all|mcsgarch vs hybrid_base"]
        assert h1["role"] == "primary" and h1["pre_specified"] == btl.PS_YES and h1["sidedness"] == btl.SIDE_ONE_PRE
        assert h1["p_raw"] == pytest.approx(p) and h1["hypothesis"].startswith("P91 H1")
        assert h1["p_study_adjusted"] == pytest.approx(ph) and h1["p_study_adjusted_recomputed"] == pytest.approx(ph)
    har = rows["tests_next15.csv|qlike|expiry|mcsgarch vs har"]
    assert har["sidedness"] == btl.SIDE_CONVERTED and har["p_raw"] == pytest.approx(0.2) and har["pre_specified"] == btl.PS_NO
    assert rows["tests_next30.csv|qlike|multi_weekly_expiry|mcsgarch vs hybrid_base"]["role"] == "exploratory"
    ab = {tuple(r["source_key"].split("|")[:4]): r for r in new if r["source_key"].startswith("ablation_next15.csv|")}
    h4 = ab[("ablation_next15.csv", "hybrid: base vs base+GEX", "us", "all")]
    assert h4["role"] == "secondary" and h4["hypothesis"].startswith("P91 H4") and h4["p_raw"] == pytest.approx(0.27)
    assert h4["pre_specified"] == btl.PS_YES
    us_exp = ab[("ablation_next15.csv", "hybrid: base vs base+GEX", "us", "expiry")]  # an R1V2 headline, not a P91 hypothesis
    assert us_exp["role"] == "secondary" and us_exp["sidedness"] == btl.SIDE_CONVERTED and us_exp["p_raw"] == pytest.approx(0.26)
    assert ab[("ablation_next15.csv", "hybrid: base vs base+GEX", "us", "holdout")]["role"] == "exploratory"
    assert not any(r["role"] == "headline" for r in new)
    checks = {c["family"]: c["result"] for c in ctx.checks if c["study"] == "BNFC"}
    assert checks["H1: Holm over the next 15 and next 30 minutes"] == "match"
    assert "pipeline_nse_banknifty/daily_market.csv" in ctx.consumed


# ---------------------------------------------------------------------------------------------------------
# P104: the forecasting pipeline on a third index, Fin Nifty
# ---------------------------------------------------------------------------------------------------------


def test_fnfc_absent_folder_is_a_note_not_an_error(tmp_path):
    ctx = btl.Ctx(tmp_path)
    assert btl.adapt_fnfc(ctx) == 0
    assert not ctx.rows and not ctx.warnings
    assert len(ctx.not_counted) == 1 and "registered run not finished" in ctx.not_counted[0]["reason"]


def test_fnfc_on_a_copy_of_the_v2_layout(tmp_path):
    """As the BNFC test, under pipeline_nse_finnifty with p104/: the plan's hypotheses are labelled P104 and keep
    their one-sided p; the other indices on the same dates are exploratory; Fin Nifty's own rows are not counted."""
    v2 = _write_v2(tmp_path)
    fn = tmp_path / "results" / "pipeline_nse_finnifty"
    fn.mkdir(parents=True)
    for f in v2.iterdir():
        (fn / f.name).write_bytes(f.read_bytes())
    t30 = _v2_tests().assign(mean_diff=lambda d: d["mean_diff"] * 1.1, stat=lambda d: d["stat"] * 1.1)
    t30.loc[0, "p"] = 0.004
    t30.loc[len(t30)] = ("multi_weekly_expiry", "qlike", "mcsgarch vs hybrid_base", 0.03, 2.0, 0.03, 60)
    _write(fn / "tests_next30.csv", t30)
    _write(fn / "p104" / "hypotheses.csv", pd.DataFrame({
        "hypothesis": ["H1", "H1"], "target": ["next15", "next30"], "p_one_sided": [0.001, 0.004], "p_holm": [0.002, 0.004]}))
    own_p = float(_v2_tests().loc[0, "p"])
    _write(fn / "p104" / "other_indices_same_dates.csv", pd.DataFrame({
        "index": ["Nifty (v2 forecasts, same dates)", "Bank Nifty (P91 forecasts, same dates)", "Fin Nifty"],
        "target": ["next15"] * 3, "subset": ["all"] * 3, "days": [604] * 3, "anchors": [15100] * 3,
        "qlike_mcsgarch": [0.2] * 3, "qlike_model": [0.19] * 3, "change_pct": [-5.0, -4.0, -3.0],
        "dm_stat": [8.1, 6.6, 2.0], "p_one_sided": [1e-9, 2e-8, own_p]}))
    _write(fn / "p104" / "exploratory.csv", pd.DataFrame({
        "analysis": ["without 5 June 2024"] * 2, "target": ["next15"] * 2, "group": ["all test days except 5 June 2024"] * 2,
        "first_day": ["2023-08-10"] * 2, "last_day": ["2026-01-27"] * 2, "loss": ["qlike", "mse"], "days": [603] * 2,
        "anchors": [15075] * 2, "loss_mcsgarch": [0.25, 3e-11], "loss_hybrid_base": [0.24, 2.9e-11],
        "change_pct": [-4.8, -4.5], "dm_stat": [6.5, 1.6], "p_one_sided": [7e-11, 0.058]}))
    _write(fn / "p104" / "june5_2024.csv", pd.DataFrame({"index": ["Fin Nifty"], "mcsgarch_over_realized": [134.6]}))
    _write(fn / "daily_market.csv", pd.DataFrame({"date": ["2024-01-01"]}))
    _fake_repo(tmp_path, "P104 registered plan", doc="docs/prereg/P104_THIRD_INDEX_FORECAST.md")
    ctx = btl.Ctx(tmp_path)
    n = btl.adapt_fnfc(ctx)
    new = [r for r in ctx.rows if r["study"] == "FNFC"]
    assert n == len(new) == 5 + 6 + 7 + 1 + 2 + 2 and not ctx.warnings
    expl = [r for r in new if r["source_file"].endswith("p104/exploratory.csv")]
    assert len(expl) == 2 and all(r["role"] == "exploratory" and r["pre_specified"] == btl.PS_NO for r in expl)
    assert sorted(r["p_raw"] for r in expl) == pytest.approx([1.4e-10, 0.116])  # one-sided p doubled (not planned)
    assert "pipeline_nse_finnifty/p104/june5_2024.csv" in ctx.consumed
    rows = {r["source_key"]: r for r in new}
    for t, p, ph in (("next15", 0.001, 0.002), ("next30", 0.004, 0.004)):
        h1 = rows[f"tests_{t}.csv|qlike|all|mcsgarch vs hybrid_base"]
        assert h1["role"] == "primary" and h1["pre_specified"] == btl.PS_YES and h1["sidedness"] == btl.SIDE_ONE_PRE
        assert h1["p_raw"] == pytest.approx(p) and h1["hypothesis"].startswith("P104 H1") and "Fin Nifty" in h1["hypothesis"]
        assert h1["p_study_adjusted"] == pytest.approx(ph)
    assert rows["tests_next30.csv|qlike|multi_weekly_expiry|mcsgarch vs hybrid_base"]["role"] == "exploratory"
    others = [r for r in new if r["source_file"].endswith("other_indices_same_dates.csv")]
    assert len(others) == 2 and all(r["role"] == "exploratory" and r["pre_specified"] == btl.PS_NO for r in others)
    assert {r["hypothesis"].split(" on the")[0] for r in others} == {"Nifty", "Bank Nifty"}
    assert not any(r["role"] == "headline" for r in new)
    assert any("Fin Nifty's own rows" in c["what"] for c in ctx.not_counted if c["study"] == "FNFC")
    assert "pipeline_nse_finnifty/daily_market.csv" in ctx.consumed
