"""Tests for experiments/p120_joint_option_market.py (P120). SYNTHETIC data only (no raw data file is read).

Covers the drop set and the job plan, the "beyond the margin" tests (they equal the project's DM test with a shifted
null and complement P110's TOST), the seed average (which, unlike P110's, averages the optionmkt column), the verdict
rules, the check that the registered plan is in the git history (on a temporary repository), the refusals on real
data, and an end-to-end dry run of both stages on simulated markets.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex.evaluation import diebold_mariano

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p120_joint_option_market.py"
_spec = importlib.util.spec_from_file_location("p120_joint_option_market", _PATH)
pj = importlib.util.module_from_spec(_spec)
sys.modules["p120_joint_option_market"] = pj
_spec.loader.exec_module(pj)
gd, st = pj.gd, pj.st


def ar1(n, phi, mean, rng, df=5, scale=1.0):
    """AR(1) series with Student t errors (heavy tails) and the given mean."""
    e = rng.standard_t(df, n) * scale
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return mean + x


def toy_frame(n=300, seed=0, shift=0.0, base=None):
    """A day frame like gd.day_frame's: daily QLIKE of mcsgarch, mcsgarch_iv, full, implied, vix and optionmkt;
    optionmkt is ``shift`` points of mcsGARCH's mean QLIKE above full on average."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-05-01", periods=n)
    b = base if base is not None else 0.25 * np.exp(np.random.default_rng(99).normal(0, 0.3, n))
    D = pd.DataFrame({"mcsgarch": b, "mcsgarch_iv": b * 1.01}, index=idx)
    D["full"] = b * 0.95 + rng.normal(0, 0.002, n)
    D["implied"] = D["full"] + rng.normal(0, 0.002, n)
    D["vix"] = D["full"] + rng.normal(0, 0.002, n)
    D["optionmkt"] = D["full"] + shift * b.mean() / 100 + rng.normal(0, 0.002, n)
    D["is_expiry"] = (np.arange(n) % 5) == 0
    return D


# ---------------------------------------------------------------- design


def test_drop_set_seeds_and_settings():
    assert gd.DROP_SETS[pj.VARIANT] == gd.GROUPS["implied"] + gd.GROUPS["vix"]
    assert len(gd.DROP_SETS[pj.VARIANT]) == 17 and len(gd.hybrid_columns(pj.VARIANT)) == 24
    assert pj.SEEDS == [7, 101, 102, 103, 104] and pj.NEW_SEEDS == [101, 102, 103, 104]
    assert pj.MARGIN == 1.0 and pj.MARGIN_SENS == 0.5 and pj.ALPHA == 0.05
    assert pj.PLAN == "docs/prereg/P120_JOINT_OPTION_MARKET.md" and (gd.ROOT / pj.PLAN).exists()
    for ix in pj.STUDY_INDEXES:        # the committed runs' settings, as P106 checks them
        info = json.load(open(gd.INDEXES[ix]["run"] / "run_info.json"))
        assert gd.check_run_settings(ix, info, info["days"][1]) == []


def test_job_plan_order_and_coverage():
    bn = pj.job_plan("BANKNIFTY")
    assert len(bn) == 15
    assert bn[:2] == [("BANKNIFTY", "next15", 7, ("optionmkt",)), ("BANKNIFTY", "next30", 7, ("optionmkt",))]
    assert [j[2] for j in bn[:10]] == [7, 7, 101, 101, 102, 102, 103, 103, 104, 104]
    assert all(j[3] == ("optionmkt",) for j in bn[:10])                      # the full hybrid is P110's
    assert bn[10:] == [("BANKNIFTY", "last_hour", s, ("full", "optionmkt")) for s in pj.SEEDS]
    assert pj.job_plan("NIFTY") == [("NIFTY", "next15", 7, ("optionmkt",))] + \
        [("NIFTY", "last_hour", s, ("full", "optionmkt")) for s in [101, 102, 103, 104]]
    assert pj.job_plan("FINNIFTY", seeds=[7], targets=["last_hour"]) == [("FINNIFTY", "last_hour", 7, ("full", "optionmkt"))]
    syn = pj.job_plan("SYNTHA")
    assert all("implied" in j[3] and "vix" in j[3] for j in syn if j[1] != "last_hour")
    with pytest.raises(ValueError):
        pj.job_plan("SENSEX")


# ---------------------------------------------------------------- the tests


def test_beyond_margin_equals_shifted_dm_and_complements_the_tost():
    rng = np.random.default_rng(2)
    for mean in (-1.6, -0.5, 0.0, 0.4, 1.3, 2.2):
        d = ar1(660, 0.2, mean, rng, scale=1.5)
        up = pj.beyond_margin(d, 1.0, "above")
        lo = pj.beyond_margin(d, 1.0, "below")
        dm_up = diebold_mariano(d - 1.0, np.zeros_like(d), alternative="greater")
        dm_lo = diebold_mariano(d + 1.0, np.zeros_like(d), alternative="less")
        assert np.isclose(up["p"], dm_up["p"], rtol=1e-10) and np.isclose(up["stat"], dm_up["stat"], rtol=1e-10)
        assert np.isclose(lo["p"], dm_lo["p"], rtol=1e-10)
        tt = st.tost(d, 1.0)
        assert np.isclose(up["p"], 1 - tt["p_upper"], atol=1e-12) and np.isclose(lo["p"], 1 - tt["p_lower"], atol=1e-12)
        # "worse by at least 1 point" passes at 5% exactly when the 90% interval lies wholly above +1
        assert (up["p"] < 0.05) == (tt["ci90_low"] > 1.0)
        assert (lo["p"] < 0.05) == (tt["ci90_high"] < -1.0)
    with pytest.raises(ValueError):
        pj.beyond_margin(d, 1.0, "sideways")


def test_joint_estimate_interaction_and_points_series():
    D = toy_frame(shift=0.6)
    J = pj.joint_estimate(D)
    assert np.isclose(J, 100 * (D["optionmkt"].mean() - D["full"].mean()) / D["mcsgarch"].mean())
    assert np.isclose(st.points_series(D, "optionmkt", "full").mean(), J)
    assert abs(J - 0.6) < 0.2
    C = {v: 100 * (D[v].mean() - D["full"].mean()) / D["mcsgarch"].mean() for v in ("implied", "vix")}
    assert np.isclose(pj.interaction_series(D).mean(), J - C["implied"] - C["vix"])
    r = pj.row_stats(D, "tost", 1.0)
    assert r["estimate"] == J and np.isclose(r["p"], st.tost(st.points_series(D, "optionmkt", "full"), 1.0)["p"])
    dm = diebold_mariano(D["optionmkt"], D["full"], alternative="two-sided")
    assert pj.row_stats(D, "diff")["p"] == dm["p"]
    rd = pj.row_stats(D, "tost", 1.0, method="delta")
    assert np.isclose(rd["estimate"], J)


def test_seed_average_includes_optionmkt_unlike_p110s():
    D1, D2 = toy_frame(seed=1, shift=0.2), toy_frame(seed=2, shift=0.8)
    A = pj.seed_average([D1, D2])
    assert np.allclose(A["optionmkt"], (D1["optionmkt"] + D2["optionmkt"]) / 2)
    assert np.allclose(A["full"], (D1["full"] + D2["full"]) / 2) and A["mcsgarch"].equals(D1["mcsgarch"])
    # the five-seed estimate is the mean of the seed estimates (mcsGARCH is the same in every seed set)
    assert np.isclose(pj.joint_estimate(A), (pj.joint_estimate(D1) + pj.joint_estimate(D2)) / 2)
    # P110's seed_average leaves optionmkt at the first seed set's values: the reason for our own
    B = st.seed_average([D1, D2])
    assert B["optionmkt"].equals(D1["optionmkt"])
    D3 = D2.copy()
    D3.iloc[0, 0] += 1e-9
    with pytest.raises(ValueError):
        pj.seed_average([D1, D3])
    with pytest.raises(ValueError):
        pj.seed_average([D1, D2.iloc[1:]])


def test_verdict_rules():
    v = pj.primary_verdict
    assert v(0.01, 0.9, 0.9, [0.2, 0.4, 0.1, 0.3, 0.5]) == ("equivalent within 1 point", "")
    assert v(0.01, 0.9, 0.9, [0.2, 0.4, 1.1, 0.3, 0.5]) == ("not shown either way", "seed-fragile")
    assert v(0.30, 0.9, 0.9, [0.2, 0.4, 0.1, 0.3, 0.5]) == ("not shown either way", "")
    assert v(0.99, 0.01, 0.9, [1.5, 1.8, 1.2, 1.9, 1.4]) == ("worse by at least 1 point", "")
    assert v(0.99, 0.01, 0.9, [1.5, 0.8, 1.2, 1.9, 1.4]) == ("not shown either way", "seed-fragile")
    assert v(0.99, 0.9, 0.01, [-1.5, -1.8, -1.2, -1.9, -1.4]) == ("better by at least 1 point without them", "")
    assert v(np.nan, 0.9, 0.9, [0.1]) == ("not computed", "")
    assert v(0.01, 0.9, 0.9, [0.2, 0.45], delta=0.5) == ("equivalent within 0.5 point", "")
    assert pj.margin_only_verdict(0.01, [0.1, 0.2, -0.3], 0.5) == ("equivalent within 0.5 point", "")
    assert pj.margin_only_verdict(0.01, [0.1, 0.6, -0.3], 0.5) == ("not shown", "seed-fragile")
    assert pj.margin_only_verdict(0.2, [0.1, 0.2], 0.5) == ("not shown", "")
    assert pj.difference_verdict(0.01, 0.4, [0.2, 0.5, 0.1]) == ("different from zero", "")
    assert pj.difference_verdict(0.01, 0.4, [0.2, -0.5, 0.1]) == ("no difference shown", "seed-fragile")
    assert pj.difference_verdict(0.3, 0.4, [0.2, 0.5, 0.1]) == ("no difference shown", "")
    assert pj.within_seed_noise(0.1, [0.0, 0.3, -0.2]) and not pj.within_seed_noise(0.6, [0.55, 0.6, 0.65])


def test_compare_helpers(tmp_path):
    D = toy_frame(n=50)
    assert pj.compare_daily(D, D.copy(), ["mcsgarch", "full"], "same")["identical"]
    E = D.copy()
    E.iloc[3, E.columns.get_loc("full")] *= 1.0000001
    r = pj.compare_daily(E, D, ["mcsgarch", "full"], "one day differs")
    assert not r["identical"] and r["full_n_identical"] == 49 and r["mcsgarch_n_identical"] == 50
    assert not pj.compare_daily(D.iloc[1:], D, ["full"], "days differ")["identical"]
    f = pd.DataFrame({"t": [1, 1, 2], "minute": [0, 15, 0], "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                      "y": [1.0, 2.0, 3.0], "mcsgarch": [1.1, 2.1, 3.1], "mcsgarch_iv": [1.2, 2.2, 3.2],
                      "forecast": [0.9, 1.9, 2.9]})
    f.to_csv(tmp_path / "a.csv", index=False)
    f.iloc[::-1].to_csv(tmp_path / "b.csv", index=False)               # row order does not matter
    assert pj.compare_forecast_files(tmp_path / "a.csv", tmp_path / "b.csv", "same")["identical"]
    g = f.copy()
    g.loc[1, "forecast"] = 1.95
    g.to_csv(tmp_path / "c.csv", index=False)
    r = pj.compare_forecast_files(tmp_path / "a.csv", tmp_path / "c.csv", "forecast differs")
    assert not r["identical"] and r["forecast_n_identical"] == 2
    assert pj.compare_forecast_files(tmp_path / "a.csv", tmp_path / "c.csv", "ids only", forecast=False)["identical"]


# ---------------------------------------------------------------- the plan must be in the git history


def _git(root, *args):
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args], cwd=root,
                          capture_output=True, text=True, check=True)


def test_plan_in_history_and_plan_commit(tmp_path):
    root = tmp_path / "repo"
    (root / "docs" / "prereg").mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "README.txt").write_text("x")
    _git(root, "add", "README.txt")
    _git(root, "commit", "-q", "-m", "first")
    first = _git(root, "rev-parse", "HEAD").stdout.strip()
    assert pj.plan_in_history(root) is None
    with pytest.raises(SystemExit):
        pj.require_plan(root)
    (root / pj.PLAN).write_text("plan")                                     # on disk but not committed
    assert pj.plan_in_history(root) is None
    _git(root, "add", pj.PLAN)
    _git(root, "commit", "-q", "-m", "plan")
    plan_commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    assert pj.plan_in_history(root) == plan_commit == pj.require_plan(root)
    (root / "code.txt").write_text("y")
    _git(root, "add", "code.txt")
    _git(root, "commit", "-q", "-m", "code")
    assert pj.check_plan_commit(plan_commit[:7], root) == plan_commit       # an ancestor that contains the plan
    with pytest.raises(SystemExit):
        pj.check_plan_commit(first, root)                                   # does not contain the plan
    with pytest.raises(SystemExit):
        pj.check_plan_commit(None, root)
    with pytest.raises(SystemExit):
        pj.check_plan_commit("0123456789abcdef0123456789abcdef01234567", root)
    _git(root, "checkout", "-q", first)                                     # the plan commit is not an ancestor
    with pytest.raises(SystemExit):
        pj.check_plan_commit(plan_commit, root)


def test_real_data_stages_refuse_without_the_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(pj, "plan_in_history", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="not in the git history"):
        pj.main(["run", "--index", "banknifty", "--work", str(tmp_path / "w")])
    with pytest.raises(SystemExit, match="not in the git history"):
        pj.main(["analyze", "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    monkeypatch.setattr(pj, "plan_in_history", lambda *a, **k: "abc")
    with pytest.raises(SystemExit, match="--plan-commit is required"):
        pj.main(["analyze", "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        pj.main(["run", "--index", "SYNTHA", "--work", str(tmp_path / "w")])  # synthetic index without --synthetic


# ---------------------------------------------------------------- end to end


def test_end_to_end_dry_run(tmp_path, monkeypatch):
    monkeypatch.setitem(pj.SYNTH, "n_days", 150)
    monkeypatch.setitem(pj.SYNTH, "min_train", 50)
    monkeypatch.setitem(pj.SYNTH, "block", 30)
    work, out = tmp_path / "work", tmp_path / "out"
    for ix in pj.SYNTH_INDEXES:
        pj.main(["run", "--synthetic", "--index", ix, "--work", str(work), "--workers", "1"])
    pj.main(["analyze", "--synthetic", "--reps", "99", "--work", str(work), "--out", str(out)])
    tests = pd.read_csv(out / "tests.csv")
    p110_cols = list(pd.read_csv(gd.ROOT / "results" / "p110_gain_stress_test" / "tests.csv", nrows=1).columns)
    assert list(tests.columns) == p110_cols
    assert tests["test_id"].is_unique and tests["sample"].str.startswith("SYNTHETIC").all()
    h1 = tests[tests["test_id"].str.match(r"^H1_")]
    assert len(h1) == 6 and (h1["role"] == "primary").all()
    assert h1["adjust_method"].str.startswith("Holm over 6 rows").all()
    assert np.allclose(h1["p_adjusted"], gd.holm(h1["p_raw"].to_numpy()))
    for fam, k in (("H1w", 6), ("H1b", 6), ("S1", 6), ("S2", 6), ("S3a", 3), ("S3aw", 3), ("S3b", 3), ("S4", 2),
                   ("R1", 6), ("R2", 6), ("R3", 6), ("R4", 6), ("R5", 6)):
        rows = tests[tests["test_id"].str.startswith(fam + "_")]
        assert len(rows) == k and rows["adjust_method"].str.startswith(f"Holm over {k} rows").all(), fam
    assert (tests.loc[tests["test_id"].str.match(r"^E[12]_"), "adjust_method"] == "none (no adjustment)").all()
    v = pd.read_csv(out / "verdict.csv")
    allowed = {"equivalent within 1 point", "worse by at least 1 point", "better by at least 1 point without them",
               "not shown either way"}
    assert set(v.loc[v["family"] == "H1", "verdict"]) <= allowed and (v.loc[v["family"] == "H1", "seed_sets"] == 2).all()
    # an H1 estimate can be recomputed from the daily-loss files (average over the two synthetic seed sets)
    d7 = pd.read_csv(out / "daily_losses_syntha_next30_s7.csv", index_col="date")
    d101 = pd.read_csv(out / "daily_losses_syntha_next30_s101.csv", index_col="date")
    j = 50 * ((d7["optionmkt"] - d7["full"]).mean() + (d101["optionmkt"] - d101["full"]).mean()) / d7["mcsgarch"].mean()
    assert np.isclose(tests.loc[tests["test_id"] == "H1_SYNTHA_next30", "estimate"].iloc[0], j)
    # every results table carries the data label
    for f in out.glob("*.csv"):
        if f.name != "tests.csv":
            assert pd.read_csv(f, nrows=1)["data"].iloc[0].startswith("SYNTHETIC"), f.name
    info = json.loads((out / "run_info.json").read_text())
    assert info["source"].startswith("SYNTHETIC") and info["plan_commit"] is None
    pooled = pd.read_csv(out / "pooled.csv")
    assert len(pooled) == 2 and pooled["pool"].str.count(r"\+").eq(2).all()
