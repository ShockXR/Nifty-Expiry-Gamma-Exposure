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
