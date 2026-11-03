"""Tests for experiments/p106_gain_decomposition.py (P106). SYNTHETIC data only (no raw data file is read).

Covers the feature groups (every column of the hybrid in exactly one group), the new leave-columns-out code
path in niftygex.forecast (default unchanged, exact column lists, no interference between models fitted in
one walk-forward), the configuration (the committed runs' settings), the job plan, the work files and the
reproduction check, the statistics (contribution, DM, period-difference test and its size, bootstrap,
shares), the registered verdict labels, and an end-to-end dry run of both stages on a simulated market.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex.evaluation import daily_losses, diebold_mariano
from niftygex.features import BASE_FEATURES, GEX_SETS, build_dataset, build_gex_panel
from niftygex.forecast import WalkForwardConfig, model_columns, run_walkforward
from niftygex.implied import build_implied_panel
from niftygex.simulate import SimConfig, simulate_market

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p106_gain_decomposition.py"
_spec = importlib.util.spec_from_file_location("p106_gain_decomposition", _PATH)
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)

SETS = {k: v for k, v in GEX_SETS.items() if k != "oracle"}


@pytest.fixture(scope="module")
def synth():
    m = simulate_market(SimConfig(n_days=100, kappa=0.0, seed=5, start="2024-08-01"))
    ds = build_dataset(m, build_gex_panel(m), horizon=30, step=45, implied=build_implied_panel(m))
    return m, ds


def small_cfg(seed, variants, **kw):
    return gd.make_config(seed, None, SETS, variants, min_train=50, block=20, mcs_refit_every=20, n_seeds=2, **kw)


# ---------------------------------------------------------------- groups and the new code path


def test_every_hybrid_column_is_in_exactly_one_group():
    cols = gd.HYBRID_COLUMNS
    assert len(cols) == 41 and len(set(cols)) == 41
    assert cols[:len(BASE_FEATURES)] == list(BASE_FEATURES)
    members = [c for g in gd.GROUPS.values() for c in g]
    assert sorted(members) == sorted(cols)          # a partition: no column missing, none twice
    assert {len(v) for v in gd.GROUPS.values()} == {16, 1, 4, 8, 12}
    assert "vrp_w" in gd.GROUPS["implied"] and "abs_ret_today_sd" in gd.GROUPS["implied"]
    assert "log_mcs" in gd.GROUPS["realized"] and "log_s_window" in gd.GROUPS["intraday"]
    assert set(gd.IMPLIED_LEVEL) | set(gd.IMPLIED_SHAPE) == set(gd.GROUPS["implied"])
    assert not set(gd.IMPLIED_LEVEL) & set(gd.IMPLIED_SHAPE)
    assert set(gd.DROP_SETS["optionmkt"]) == set(gd.GROUPS["implied"]) | {"log_vix"}
    kept = set(gd.hybrid_columns("mcsinfo"))
    assert kept == set(gd.GROUPS["realized"]) | set(gd.GROUPS["intraday"]) | {"is_expiry"}
    assert len(gd.VARIANTS) == 10 and gd.VARIANTS[0] == "full"
    assert set(gd.REPEAT_VARIANTS) <= set(gd.VARIANTS) and set(gd.OTHER_INDEX_VARIANTS) <= set(gd.REPEAT_VARIANTS)


def test_drop_feature_sets_in_model_columns():
    cfg = WalkForwardConfig()
    assert cfg.drop_sets == {}                        # default unchanged: no drop sets
    extra = list(gd.EXTRA_V2)
    base = model_columns("base", cfg, extra, ["log_mcs", "log_s_window"])
    assert base == gd.HYBRID_COLUMNS
    cfg = WalkForwardConfig(drop_sets={k: list(v) for k, v in gd.DROP_SETS.items()})
    assert model_columns("base", cfg, extra, []) == base   # adding drop sets changes nothing for "base"
    for v in gd.DROP_SETS:
        cols = model_columns(f"drop_{v}", cfg, extra, [])
        assert cols == gd.hybrid_columns(v) == [c for c in base if c not in set(gd.DROP_SETS[v])]
    with pytest.raises(KeyError):
        model_columns("drop_nonexistent", cfg, extra, [])


def test_drop_variants_in_one_walkforward(synth):
    m, ds = synth
    variants = ["full", "implied", "realized"]
    pred, info = run_walkforward(m, ds, small_cfg(11, variants))
    alone, _ = run_walkforward(m, ds, small_cfg(11, ["full"]))
    # the full hybrid does not depend on which other variants are fitted with it
    for c in ("y", "mcsgarch", "mcsgarch_iv", "hybrid_base"):
        assert np.array_equal(pred[c].to_numpy(), alone[c].to_numpy()), c
    assert not np.array_equal(pred["hybrid_drop_implied"].to_numpy(), pred["hybrid_base"].to_numpy())
    # the importance table of each variant lists exactly its columns
    imp = {s.name.split("@")[0]: list(s.index) for s in info["importance"]}
    assert imp["hybrid_base"] == gd.hybrid_columns("full")
    assert imp["hybrid_drop_implied"] == gd.hybrid_columns("implied")
    assert imp["hybrid_drop_realized"] == gd.hybrid_columns("realized")
    # an empty drop set gives the full hybrid exactly
    cfg = small_cfg(11, ["full"])
    cfg.models = cfg.models + ["hybrid_drop_none"]
    cfg.drop_sets = {**cfg.drop_sets, "none": []}
    p2, _ = run_walkforward(m, ds, cfg)
    assert np.array_equal(p2["hybrid_drop_none"].to_numpy(), p2["hybrid_base"].to_numpy())


# ---------------------------------------------------------------- configuration and plan


def test_config_matches_the_committed_runs():
    cfg = gd.make_config(7, [315], SETS, ["full", "vix"])
    assert cfg.models == ["mcsgarch", "mcsgarch_iv", "hybrid_base", "hybrid_drop_vix"]
    assert (cfg.n_seeds, cfg.seed, cfg.min_train, cfg.block, cfg.mcs_refit_every, cfg.tune, cfg.tune_every) == \
        (5, 7, 150, 21, 21, False, 3)
    assert cfg.eval_minutes == [315] and cfg.lgbm_threads == 1
    for index, spec in gd.INDEXES.items():           # committed settings files, not data
        info = json.loads((spec["run"] / "run_info.json").read_text())
        assert gd.check_run_settings(index, info, info["days"][1]) == [], index
    info = json.loads((gd.INDEXES["NIFTY"]["run"] / "run_info.json").read_text())
    assert gd.check_run_settings("BANKNIFTY", info, info["days"][1])     # wrong symbol is caught
    bad = json.loads(json.dumps(info))
    bad["args"]["n_seeds"] = 3
    assert gd.check_run_settings("NIFTY", bad, info["days"][1])
    assert gd.check_run_settings("NIFTY", info, "2026-11-03")


def test_job_plan_order_and_coverage():
    plan = gd.job_plan("NIFTY")
    # headline seed first, all three targets, every variant exactly once per target
    head = [j for j in plan if j[2] == gd.HEADLINE_SEED]
    assert plan[:len(head)] == head
    for t in gd.TARGETS:
        vs = [v for j in head if j[1] == t for v in j[3]]
        assert vs == gd.VARIANTS
    rep = [j for j in plan if j[2] != gd.HEADLINE_SEED]
    assert {j[2] for j in rep} == set(gd.REPEAT_SEEDS) and {j[1] for j in rep} == set(gd.PRIMARY_TARGETS)
    for s in gd.REPEAT_SEEDS:
        for t in gd.PRIMARY_TARGETS:
            assert [v for j in rep if (j[1], j[2]) == (t, s) for v in j[3]] == gd.REPEAT_VARIANTS
    assert all(len(j[3]) <= gd.CHUNK for j in plan)
    assert sum(len(j[3]) for j in plan) == 3 * 10 + 4 * 2 * 8
    for ix in ("BANKNIFTY", "FINNIFTY"):
        p = gd.job_plan(ix)
        assert {j[1] for j in p} == set(gd.PRIMARY_TARGETS) and {j[2] for j in p} == {gd.HEADLINE_SEED}
        assert sum(len(j[3]) for j in p) == 2 * 6
    only = gd.job_plan("NIFTY", targets=["next15"], seeds=[7], variants=["full"])
    assert only == [("NIFTY", "next15", 7, ("full",))]


def test_repeat_seeds_are_p99s_and_disjoint_from_the_headline():
    assert gd.REPEAT_SEEDS == [101, 102, 103, 104] and gd.HEADLINE_SEED == 7
    used = {s + 1000 * k for s in gd.ALL_SEEDS for k in range(gd.N_SEEDS)}
    assert len(used) == 25


# ---------------------------------------------------------------- work files and the reproduction check


def test_work_files_round_trip_and_reproduction(synth, tmp_path, monkeypatch):
    m, ds = synth
    gd._STATE.update(market=m, datasets={"next30": ds}, sets=SETS, eval_minutes={},
                     config_fn=lambda seed, ev, sets, variants: small_cfg(seed, variants))
    (tmp_path / "timing.csv").write_text("index,target,seed,variants,n_variants,seconds,rows,pid\n")
    out = gd.run_job(("SYNTH", "next30", 7, ("full", "vix"), str(tmp_path)))
    assert out[0] == "SYNTH" and out[3] == ("full", "vix") and out[5] > 0
    assert gd.job_done(tmp_path, ("SYNTH", "next30", 7, ("full", "vix")))
    assert not gd.job_done(tmp_path, ("SYNTH", "next30", 7, ("full", "calendar")))
    wide = gd.load_wide(tmp_path, "SYNTH", "next30", 7, ["full", "vix"])
    direct, _ = run_walkforward(m, ds, small_cfg(7, ["full", "vix"]))
    direct.to_csv(tmp_path / "committed.csv", index=False)
    committed = gd.read_csv_exact(tmp_path / "committed.csv")
    r = gd.reproduction_check(wide, committed, "synthetic")
    assert r["rows_matched"] == len(direct) == r["hybrid_base_n_identical"] == r["mcsgarch_n_identical"]
    assert r["mcsgarch_iv_n_identical"] == len(direct) and r["hybrid_base_max_abs_rel_diff"] == 0.0
    assert np.array_equal(wide["vix"].to_numpy(), direct["hybrid_drop_vix"].to_numpy())
    committed.loc[2, "hybrid_base"] *= 1.001
    assert gd.reproduction_check(wide, committed, "x")["hybrid_base_n_identical"] == len(direct) - 1
    bad = gd.read_csv_exact(gd.work_file(tmp_path, "SYNTH", "next30", 7, "vix"))
    bad.loc[0, "mcsgarch"] *= 2
    bad.to_csv(gd.work_file(tmp_path, "SYNTH", "next30", 7, "vix"), index=False)
    with pytest.raises(ValueError):
        gd.load_wide(tmp_path, "SYNTH", "next30", 7, ["full", "vix"])
    # the date split equals the pipeline's regime labels
    D = gd.day_frame(wide, ["full"])
    assert np.array_equal(gd.period_mask(D), D["regime"].to_numpy() != 0)
    assert gd.period_mask(D).any() and (~gd.period_mask(D)).any()


# ---------------------------------------------------------------- statistics


def toy_days(n=300, gain=0.02, contrib=0.01, seed=0, p2_extra=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    base = np.exp(rng.normal(-1.5, 0.5, n))
    p2 = dates >= pd.Timestamp(gd.PERIOD_SPLIT)
    full = base * (1 - gain) + rng.normal(0, 0.01, n)
    var = full + contrib + p2_extra * p2 + rng.normal(0, 0.01, n)
    return pd.DataFrame({"mcsgarch": base, "mcsgarch_iv": base * 1.01, "full": full, "vix": var,
                         "regime": p2.astype(int), "is_expiry": False, "t": np.arange(n)}, index=dates)


def test_contribution_equals_the_definition_and_the_dm_test():
    d = toy_days()
    c = gd.contribution(d, "vix")
    assert np.isclose(c["estimate"], 100 * (d["vix"].mean() - d["full"].mean()) / d["mcsgarch"].mean())
    dm = diebold_mariano(d["vix"], d["full"], alternative="greater")
    assert np.isclose(c["p"], dm["p"]) and np.isclose(c["stat"], dm["stat"]) and c["p"] < 0.01
    assert np.isclose(c["margin_full"], 100 * (d["full"].mean() / d["mcsgarch"].mean() - 1))


def test_daily_frame_matches_anchor_means():
    rng = np.random.default_rng(1)
    dates = np.repeat(pd.bdate_range("2024-10-01", periods=40), 4)
    y = np.exp(rng.normal(-9, 1, len(dates)))
    f = pd.DataFrame({"t": np.repeat(np.arange(40), 4), "date": dates, "minute": np.tile([0, 45, 90, 135], 40),
                      "horizon": 30, "is_expiry": False, "tdte": 1, "regime": 0, "y": y,
                      "mcsgarch": y * np.exp(rng.normal(0, .5, len(y))), "mcsgarch_iv": y * 1.2,
                      "full": y * np.exp(rng.normal(0, .4, len(y)))})
    D = gd.day_frame(f, ["full"])
    L = daily_losses(f, ["mcsgarch", "full"])
    assert np.allclose(D["full"], L["full"])
    with pytest.raises(ValueError):
        gd.day_frame(f.iloc[1:], ["full"])     # unequal anchors per day are refused


def test_period_difference_test_known_cases():
    # constant denominator: the ratio test is a difference of means with Newey-West variances
    d = toy_days(n=400, p2_extra=0.0)
    num = (d["vix"] - d["full"]).to_numpy()
    den = np.full(len(d), 2.0)
    p2 = gd.period_mask(d)
    r = gd.period_diff_test(num, den, p2)
    assert np.isclose(r["r1"], 50 * num[~p2].mean()) and np.isclose(r["r2"], 50 * num[p2].mean())
    v1 = gd.gw.newey_west(50 * num[~p2])[0, 0] / (~p2).sum()
    v2 = gd.gw.newey_west(50 * num[p2])[0, 0] / p2.sum()
    assert np.isclose(r["se"], np.sqrt(v1 + v2))
    # a real shift between the periods is found, in the right direction
    d = toy_days(n=400, p2_extra=0.02, seed=3)
    r = gd.period_diff_test((d["vix"] - d["full"]).to_numpy(), d["mcsgarch"].to_numpy(), gd.period_mask(d))
    assert r["diff"] > 0 and r["p"] < 0.01
    b = gd.period_diff_bootstrap((d["vix"] - d["full"]).to_numpy(), d["mcsgarch"].to_numpy(), gd.period_mask(d),
                                 reps=499, rng=np.random.default_rng(0))
    assert np.isclose(b["diff"], r["diff"]) and b["p"] < 0.02 and b["q025"] < r["diff"] < b["q975"]


def test_period_difference_test_size_under_the_null():
    """AR(1) loss differentials with the same mean in both periods: about 5% rejections."""
    rng = np.random.default_rng(42)
    n, rej, reps = 400, 0, 400
    p2 = np.arange(n) >= 160
    for _ in range(reps):
        e = rng.standard_t(5, n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = 0.3 * x[i - 1] + e[i]
        den = np.exp(rng.normal(0, 0.3, n)) + 1
        num = 0.1 * x + 0.05 * den
        rej += gd.period_diff_test(num, den, p2)["p"] < 0.05
    assert 0.02 <= rej / reps <= 0.09


def test_share_of_gain_and_interval():
    d = toy_days(seed=4)
    sh, lo, hi = gd.share_ci(d["mcsgarch"].to_numpy(), d["full"].to_numpy(), d["vix"].to_numpy(), reps=499,
                             rng=np.random.default_rng(0))
    expect = (d["vix"].mean() - d["full"].mean()) / (d["mcsgarch"].mean() - d["full"].mean())
    assert np.isclose(sh, expect) and lo < sh < hi


def test_verdict_labels():
    assert gd.seed_label(5, 5, 0.01) == "supported"
    assert gd.seed_label(4, 5, 0.01) == "seed-fragile"
    assert gd.seed_label(5, 5, 0.05) == "not supported"
    assert gd.seed_label(0, 5, np.nan) == "not computed"


def test_holm_families_in_the_rows_collector():
    R = gd.Rows()
    for i, p in enumerate([0.01, 0.04, 0.03]):
        R.add(f"P1_x_{i}", "h", "primary", "s", 1.0, 1.0, p, "one-sided", "m", "primary")
    R.add("E1_x", "h", "exploratory", "s", 1.0, 1.0, 0.2, "one-sided", "m", "E1")
    df = R.frame({"primary": "test", "E1": None})
    assert list(df.columns[:11]) == ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw",
                                     "sidedness", "p_method", "p_adjusted", "adjust_method"]
    assert np.allclose(df["p_adjusted"].iloc[:3], [0.03, 0.06, 0.06])
    assert df["p_adjusted"].iloc[3] == 0.2 and df["adjust_method"].iloc[3].startswith("none")


# ---------------------------------------------------------------- end to end, both stages


def test_end_to_end_dry_run(tmp_path, monkeypatch):
    monkeypatch.setitem(gd.SYNTH_SETTINGS, "n_days", 150)
    monkeypatch.setitem(gd.SYNTH_SETTINGS, "start", "2024-07-01")
    monkeypatch.setitem(gd.SYNTH_SETTINGS, "min_train", 50)
    monkeypatch.setitem(gd.SYNTH_SETTINGS, "block", 25)
    monkeypatch.setitem(gd.SYNTH_SETTINGS, "n_seeds", 1)
    monkeypatch.setitem(gd.SYNTH_SETTINGS, "targets", ["next30"])
    work, out = tmp_path / "work", tmp_path / "out"
    gd.main(["run", "--synthetic", "--workers", "1", "--work", str(work)])
    gd.main(["analyze", "--synthetic", "--reps", "99", "--work", str(work), "--out", str(out)])
    tests = pd.read_csv(out / "tests.csv")
    assert tests["test_id"].is_unique
    assert set(tests["role"]) <= {"primary", "secondary", "robustness", "placebo", "exploratory", "post hoc"}
    prim = tests[tests["role"] == "primary"]
    assert len(prim) == 10                       # 5 groups x (part 1, part 2) x one synthetic target
    assert prim["adjust_method"].str.startswith("Holm over 10 rows").all()
    v = pd.read_csv(out / "verdict.csv")
    assert len(v) == 10 and set(v["seed_sets"]) == {2}
    info = json.loads((out / "run_info.json").read_text())
    assert info["source"].startswith("SYNTHETIC")
    # a contribution can be recomputed from the daily-loss file
    d = pd.read_csv(out / "daily_losses_synth_next30_s7.csv", index_col=0)
    row = tests[tests["test_id"] == "P1_next30_realized"].iloc[0]
    assert np.isclose(row["estimate"], 100 * (d["realized"].mean() - d["full"].mean()) / d["mcsgarch"].mean())
