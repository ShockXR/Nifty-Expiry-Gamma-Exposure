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
