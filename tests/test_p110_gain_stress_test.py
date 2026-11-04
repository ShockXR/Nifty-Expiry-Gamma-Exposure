"""Tests for experiments/p110_gain_stress_test.py (P110). SYNTHETIC data only (no raw data file is read).

Covers the variants and the job plan, the equivalence and non-inferiority tests (they equal the project's DM test
with a shifted null, their size at the margin and their power), the delta-method and pooled versions, the row
estimates, seed agreement and verdict labels, the claim-level decision rules, the influence masks, and an
end-to-end dry run of both stages on simulated markets.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex.evaluation import diebold_mariano

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p110_gain_stress_test.py"
_spec = importlib.util.spec_from_file_location("p110_gain_stress_test", _PATH)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)
gd = st.gd


def ar1(n, phi, mean, rng, df=5, scale=1.0):
    """AR(1) series with Student t errors (heavy tails) and the given mean."""
    e = rng.standard_t(df, n) * scale
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return mean + x


# ---------------------------------------------------------------- design


def test_variants_groups_and_seeds():
    assert st.VARIANTS == ["full", "implied", "vix", "intraday", "calendar", "realized", "mcsinfo"]
    assert set(st.CARRIERS) | set(st.NULL_GROUPS) == set(gd.PRIMARY_GROUPS)
    assert not set(st.CARRIERS) & set(st.NULL_GROUPS)
    kept = set(gd.hybrid_columns("mcsinfo"))
    assert kept == set(gd.GROUPS["realized"]) | set(gd.GROUPS["intraday"]) | {"is_expiry"}
    assert st.NEW_SEEDS == [101, 102, 103, 104] and st.ALL_SEEDS[0] == 7
    assert st.MARGIN == 1.0 and st.MARGIN_SENS == 0.5
    assert st.STUDY_INDEXES == ["BANKNIFTY", "FINNIFTY"]
    # the three committed runs' settings, as P106 checks them
    for ix in st.STUDY_INDEXES:
        info = json.load(open(gd.INDEXES[ix]["run"] / "run_info.json"))
        assert gd.check_run_settings(ix, info, info["days"][1]) == []


def test_job_plan_order_and_coverage():
    plan = st.job_plan("BANKNIFTY")
    assert len(plan) == 10                                   # 5 seed sets x 2 targets, all 7 variants per job
    assert [j[2] for j in plan] == [7, 7, 101, 101, 102, 102, 103, 103, 104, 104]
    assert [j[1] for j in plan[:2]] == ["next15", "next30"]
    assert all(j[3] == tuple(st.VARIANTS) for j in plan)
    assert st.job_plan("FINNIFTY", seeds=[7], variants=["full", "mcsinfo"]) == [
        ("FINNIFTY", "next15", 7, ("full", "mcsinfo")), ("FINNIFTY", "next30", 7, ("full", "mcsinfo"))]
    assert all(len(j[3]) <= 4 for j in st.job_plan("BANKNIFTY", chunk=4))


# ---------------------------------------------------------------- the tests


def test_dm_parts_reproduce_the_project_dm_test():
    rng = np.random.default_rng(1)
    d = ar1(500, 0.3, 0.1, rng)
    q = st.dm_parts(d)
    dm = diebold_mariano(d, np.zeros_like(d), alternative="greater")
    assert np.isclose(q["mean"] / q["se"] * q["hln"], dm["stat"], rtol=1e-12)
    assert q["n"] == dm["n"] == 500 and q["df"] == 499


def test_tost_equals_two_shifted_dm_tests_and_the_90_percent_interval():
    rng = np.random.default_rng(2)
    for mean in (-0.8, -0.2, 0.0, 0.3, 0.9, 1.4):
        d = ar1(660, 0.2, mean, rng, scale=1.5)
        r = st.tost(d, 1.0)
        up = diebold_mariano(d - 1.0, np.zeros_like(d), alternative="less")
        lo = diebold_mariano(d + 1.0, np.zeros_like(d), alternative="greater")
        assert np.isclose(r["p_upper"], up["p"], rtol=1e-10) and np.isclose(r["p_lower"], lo["p"], rtol=1e-10)
        assert r["p"] == max(r["p_upper"], r["p_lower"])
        assert np.isclose(r["stat"], up["stat"] if r["binding"] == "upper" else lo["stat"])
        inside = (r["ci90_low"] > -1.0) and (r["ci90_high"] < 1.0)
        assert inside == (r["p"] < 0.05)
        assert np.isclose(r["bound"], max(abs(r["ci90_low"]), abs(r["ci90_high"])))
    ni = st.noninferiority(d, 1.0)
    assert np.isclose(ni["p"], st.tost(d, 1.0)["p_upper"]) and np.isclose(ni["ci95_upper"], st.tost(d, 1.0)["ci90_high"])


def test_tost_and_noninferiority_size_at_the_margin_and_power():
    rng = np.random.default_rng(3)
    reps, n = 400, 600
    rej_tost = rej_ni = rej_power = 0
    for _ in range(reps):
        # true mean exactly on the margin: H0 true; autocorrelated, heavy-tailed differences
        d = ar1(n, 0.3, 1.0, rng, scale=2.0)
        rej_tost += st.tost(d, 1.0)["p"] < 0.05
        rej_ni += st.noninferiority(d, 1.0)["p"] < 0.05
        # true mean 0 with a standard error near the real data's (about 0.1 to 0.4 points): should pass
        rej_power += st.tost(ar1(n, 0.3, 0.0, rng, scale=2.0), 1.0)["p"] < 0.05
    assert 0.01 <= rej_tost / reps <= 0.09
    assert 0.01 <= rej_ni / reps <= 0.09
    assert rej_power / reps > 0.9


def test_delta_method_tost_equals_dm_tost_when_the_scale_is_constant():
    rng = np.random.default_rng(4)
    num = ar1(604, 0.2, 0.002, rng, scale=0.01)
    den = np.full(604, 0.25)
    a = st.tost_delta_method(num, den, 1.0)
    b = st.tost(100 * num / den.mean(), 1.0)
    for k in ("estimate", "se", "p", "p_upper", "p_lower", "ci90_low", "ci90_high"):
        assert np.isclose(a[k], b[k], rtol=1e-9, atol=1e-12), k
    # with a varying scale the two differ, but only a little for a scale that varies little
    den2 = 0.25 * np.exp(rng.normal(0, 0.05, 604))
    c = st.tost_delta_method(num, den2, 1.0)
    assert abs(c["estimate"] - 100 * num.mean() / den2.mean()) < 1e-12


def toy_frame(n=300, seed=0, shifts=None):
    """A day frame like gd.day_frame's: daily QLIKE of mcsgarch, mcsgarch_iv and the 7 variants."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-05-01", periods=n)
    base = 0.25 * np.exp(rng.normal(0, 0.3, n))
    D = pd.DataFrame({"mcsgarch": base, "mcsgarch_iv": base * 1.01}, index=idx)
    shifts = shifts or {}
    for v in st.VARIANTS:
        D[v] = base * (1 - 0.05) + shifts.get(v, 0.0) * 0.25 / 100 + rng.normal(0, 0.002, n)
    return D


def test_row_estimates_signs_and_agreement():
    D = toy_frame(shifts={"realized": 2.0, "mcsinfo": 0.3})
    m = D["mcsgarch"].mean()
    assert np.isclose(st.row_estimate(D, "carrier", "realized"), 100 * (D["realized"].mean() - D["full"].mean()) / m)
    assert np.isclose(st.row_estimate(D, "beats", "mcsinfo"), 100 * (D["mcsinfo"].mean() - m) / m)
    assert st.row_estimate(D, "beats", "mcsinfo") < 0
    assert np.isclose(st.row_estimate(D, "ni", "mcsinfo"), st.row_estimate(D, "full_beats", "mcsinfo"))
    # the carrier row is P106's contribution and DM test exactly
    c = gd.contribution(D, "realized")
    r = st.row_test(D, "carrier", "realized")
    assert (r["estimate"], r["stat"], r["p"]) == (c["estimate"], c["stat"], c["p"])
    # the 'beats' row is P106's S3 (b) test: DM of mcsGARCH's loss against the variant's, positive when better
    dm = diebold_mariano(D["mcsgarch"], D["mcsinfo"], alternative="greater")
    assert st.row_test(D, "beats", "mcsinfo")["p"] == dm["p"] and dm["stat"] > 0
    assert st.agrees("carrier", 0.1) and not st.agrees("carrier", -0.1)
    assert st.agrees("tost", 0.99) and not st.agrees("tost", 1.0) and not st.agrees("tost", -1.2)
    assert st.agrees("ni", -3.0) and not st.agrees("ni", 1.5)
    assert st.agrees("beats", -0.1) and not st.agrees("beats", 0.0)


def test_verdict_labels():
    assert st.row_verdict("carrier", 0.01, 5, 5) == "supported"
    assert st.row_verdict("carrier", 0.01, 4, 5) == "seed-fragile"
    assert st.row_verdict("carrier", 0.05, 5, 5) == "not supported"
    assert st.row_verdict("tost", 0.001, 5, 5) == "equivalent"
    assert st.row_verdict("tost", 0.2, 5, 5) == "not shown"
    assert st.row_verdict("ni", 0.001, 5, 5) == "non-inferior"
    assert st.row_verdict("ni", 0.001, 3, 5) == "seed-fragile"
    assert st.row_verdict("beats", np.nan, 5, 5) == "not computed"


def test_pooled_test_reduces_to_the_single_index_and_average_series():
    rng = np.random.default_rng(6)
    x = ar1(604, 0.25, 0.2, rng)
    y = 0.5 * x + ar1(604, 0.25, -0.1, rng)
    one = st.pooled_test(x[:, None], "tost", 1.0)
    single = st.tost(x, 1.0)
    for k in ("estimate", "se", "p", "p_upper", "p_lower"):
        assert np.isclose(one[k], single[k], rtol=1e-10), k
    two = st.pooled_test(np.column_stack([x, y]), "tost", 1.0)
    avgser = st.tost((x + y) / 2, 1.0)
    for k in ("estimate", "se", "p"):
        assert np.isclose(two[k], avgser[k], rtol=1e-10), k
    car = st.pooled_test(np.column_stack([x, y]), "carrier")
    dm = diebold_mariano((x + y) / 2, np.zeros(604), alternative="greater")
    assert np.isclose(car["stat"], dm["stat"]) and np.isclose(car["p"], dm["p"])
    beats = st.pooled_test(np.column_stack([-x, -y]), "beats")
    assert np.isclose(beats["p"], car["p"])
    ni = st.pooled_test(np.column_stack([x, y]), "ni", 1.0)
    assert np.isclose(ni["p"], two["p_upper"])


def test_seed_average_masks_and_top_days():
    D1, D2 = toy_frame(seed=1), toy_frame(seed=2)
    D2[["mcsgarch", "mcsgarch_iv"]] = D1[["mcsgarch", "mcsgarch_iv"]]
    A = st.seed_average([D1, D2])
    assert np.allclose(A["full"], (D1["full"] + D2["full"]) / 2) and A["mcsgarch"].equals(D1["mcsgarch"])
    D3 = D2.copy()
    D3.iloc[0, 0] += 1e-9
    with pytest.raises(ValueError):
        st.seed_average([D1, D3])
    D = toy_frame(n=300)
    D.loc[pd.Timestamp("2024-06-04")] = D.iloc[0]          # make sure the dates exist
    D = D.sort_index()
    m = st.date_mask(D, st.INFLUENCE_DATES)
    assert (~m).sum() == 2 and not m[D.index.get_loc(pd.Timestamp("2024-06-04"))]
    keep = st.top_loss_days(D, 0.01)
    assert (~keep).sum() == 3
    assert D["mcsgarch"][~keep].min() >= D["mcsgarch"][keep].max()


def claim_rows(index, verdicts, s1=None, s2=None):
    """A verdict table with the given primary verdicts; S1 and S2 rows default to 'not significant'."""
    rows = []
    s1, s2 = s1 or {}, s2 or {}
    for t in ("next15", "next30"):
        for fam, kind, v in st.PRIMARY_SPECS:
            tid = st.primary_id(index, t, fam, kind, v)
            rows.append({"test_id": tid, "verdict": verdicts.get((t, v, kind), st.VERDICT_PASS[kind]),
                         "p_adjusted": 0.001, "seed_sets_agreeing": 5, "seed_sets": 5})
        for v in st.NULL_GROUPS:
            p = s1.get((t, v), 0.9)
            rows.append({"test_id": st.primary_id(index, t, "S1", "adds", v), "verdict": "", "p_adjusted": p,
                         "seed_sets_agreeing": 5, "seed_sets": 5})
        rows.append({"test_id": st.primary_id(index, t, "S2", "full_beats", "mcsinfo"), "verdict": "",
                     "p_adjusted": s2.get(t, 0.9), "seed_sets_agreeing": 5, "seed_sets": 5})
    return pd.DataFrame(rows)


def test_claim_level_decision_rules():
    t2 = ["next15", "next30"]
    c = st.claim_verdicts(claim_rows("BANKNIFTY", {}), "BANKNIFTY", t2)
    assert c["claim1_carriers"] == "survives" and c["claim2_adds_nothing"].startswith("survives (shown")
    assert c["claim3_mcsinfo"] == "survives" and c["overall"] == "all three survive"
    c = st.claim_verdicts(claim_rows("X", {("next15", "intraday", "carrier"): "not supported",
                                           ("next30", "intraday", "carrier"): "seed-fragile"}), "X", t2)
    assert c["claim1_carriers"] == "does not survive"
    c = st.claim_verdicts(claim_rows("X", {("next15", "intraday", "carrier"): "not supported"}), "X", t2)
    assert c["claim1_carriers"] == "partly survives"
    c = st.claim_verdicts(claim_rows("X", {("next30", "implied", "tost"): "not shown"}), "X", t2)
    assert c["claim2_adds_nothing"].startswith("survives only weakly")
    c = st.claim_verdicts(claim_rows("X", {}, s1={("next15", "calendar"): 0.01}), "X", t2)
    assert c["claim2_adds_nothing"].startswith("qualified")
    c = st.claim_verdicts(claim_rows("X", {("next15", "calendar", "tost"): "not shown"},
                                     s1={("next15", "calendar"): 0.01}), "X", t2)
    assert c["claim2_adds_nothing"] == "contradicted" and c["overall"] == "not all three survive"
    c = st.claim_verdicts(claim_rows("X", {("next15", "mcsinfo", "beats"): "not supported",
                                           ("next30", "mcsinfo", "beats"): "not supported"}), "X", t2)
    assert c["claim3_mcsinfo"] == "contradicted"
    c = st.claim_verdicts(claim_rows("X", {("next30", "mcsinfo", "ni"): "not shown"}, s2={"next30": 0.01}), "X", t2)
    assert c["claim3_mcsinfo"] == "contradicted"
    c = st.claim_verdicts(claim_rows("X", {("next30", "mcsinfo", "ni"): "not shown"}), "X", t2)
    assert c["claim3_mcsinfo"] == "partly survives"


# ---------------------------------------------------------------- end to end


def test_end_to_end_dry_run(tmp_path, monkeypatch):
    monkeypatch.setitem(st.SYNTH, "n_days", 170)
    monkeypatch.setitem(st.SYNTH, "min_train", 50)
    monkeypatch.setitem(st.SYNTH, "block", 30)
    monkeypatch.setitem(st.SYNTH, "targets", ["next30"])
    work, out = tmp_path / "work", tmp_path / "out"
    for ix in ("SYNTHA", "SYNTHB", "SYNTHC"):
        st.main(["run", "--synthetic", "--index", ix, "--work", str(work)])
    st.main(["analyze", "--synthetic", "--reps", "99", "--work", str(work), "--out", str(out)])
    tests = pd.read_csv(out / "tests.csv")
    p106_cols = list(pd.read_csv(gd.ROOT / "results" / "p106_gain_decomposition" / "tests.csv", nrows=1).columns)
    assert list(tests.columns) == p106_cols
    assert tests["test_id"].is_unique
    assert set(tests["role"]) <= {"primary", "secondary", "robustness", "placebo", "exploratory", "post hoc"}
    prim = tests[tests["role"] == "primary"]
    assert len(prim) == 14                                  # 7 rows x 2 synthetic indices x 1 target
    for fam, k in (("F1", 4), ("F2", 6), ("F3", 4)):
        rows = prim[prim["test_id"].str.startswith(fam + "_")]
        assert len(rows) == k and rows["adjust_method"].str.startswith(f"Holm over {k} rows").all()
    r6 = tests[tests["test_id"].str.startswith("R6_")]
    assert len(r6) == 14 and r6["adjust_method"].str.startswith("Holm over 14 rows").all()
    assert np.allclose(r6["p_raw"].to_numpy(), prim["p_raw"].to_numpy())
    v = pd.read_csv(out / "verdict.csv")
    assert set(v["seed_sets"]) == {2}
    claims = pd.read_csv(out / "claims.csv")
    assert list(claims["index"]) == ["SYNTHA", "SYNTHB"]
    info = json.loads((out / "run_info.json").read_text())
    assert info["source"].startswith("SYNTHETIC")
    # an equivalence estimate can be recomputed from the daily-loss files (average over the new seed set 101)
    d = pd.read_csv(out / "daily_losses_syntha_next30_s101.csv", index_col=0)
    row = tests[tests["test_id"] == "F2_SYNTHA_next30_vix"].iloc[0]
    assert np.isclose(row["estimate"], 100 * (d["vix"].mean() - d["full"].mean()) / d["mcsgarch"].mean())
    pooled = pd.read_csv(out / "pooled.csv")
    assert set(pooled["pool"]) == {"synthetic C (stands in for Nifty) + synthetic A + synthetic B",
                                   "synthetic A + synthetic B"}
