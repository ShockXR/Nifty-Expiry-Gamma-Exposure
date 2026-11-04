"""Tests for experiments/seed_bootstrap.py (P99). SYNTHETIC data only (no raw data file is read).

Covers the seed bookkeeping (no LightGBM seed shared between ensembles or with the headline run),
the configuration (the headline run's settings), that fitting only mcsGARCH and hybrid_base gives the
same hybrid forecasts as a run with many other models, the work-file round trip and reproduction
check, the per-ensemble statistics (equal to the pipeline's scoreboard and DM test), the seed
bootstrap (against direct indexing, its size under the null and its response to seed noise) and the
registered decision rule.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex.evaluation import daily_losses, diebold_mariano, scoreboard
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel
from niftygex.forecast import run_walkforward
from niftygex.implied import build_implied_panel
from niftygex.simulate import SimConfig, simulate_market

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "seed_bootstrap.py"
_spec = importlib.util.spec_from_file_location("seed_bootstrap", _PATH)
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)

SETS = {k: v for k, v in GEX_SETS.items() if k != "oracle"}
_MAKE_CONFIG = sb.make_config


@pytest.fixture(scope="module")
def synth():
    m = simulate_market(SimConfig(n_days=80, kappa=0.0, seed=5))
    ds = build_dataset(m, build_gex_panel(m), horizon=30, step=45, implied=build_implied_panel(m))
    return m, ds


def small_cfg(seed, models=sb.MODELS):
    return _MAKE_CONFIG(seed, None, SETS, models=models, min_train=50, block=15, mcs_refit_every=15, n_seeds=2)


# ---------------------------------------------------------------- seeds and configuration


def test_seed_sets_do_not_overlap():
    assert sb.lgbm_seeds(7) == [7, 1007, 2007, 3007, 4007]
    assert len(sb.NEW_SEEDS) == 20 and sb.HEADLINE_SEED not in sb.NEW_SEEDS
    assert sb.seeds_disjoint(sb.NEW_SEEDS)
    assert not sb.seeds_disjoint([8])            # single seed 8 of the v2 seed study
    assert not sb.seeds_disjoint([101, 1101])    # 1101 is the second seed of ensemble 101
    assert not sb.seeds_disjoint([1007])         # the headline's second seed
    used = {x for b in sb.NEW_SEEDS for x in sb.lgbm_seeds(b)}
    assert len(used) == 100 and not used & {7, 8, 9, 1007, 2007, 3007, 4007}


def test_config_is_the_headline_runs_with_another_seed():
    cfg = sb.make_config(113, [315], SETS)
    assert cfg.models == ["mcsgarch", "hybrid_base"] and cfg.seed == 113 and cfg.eval_minutes == [315]
    assert (cfg.n_seeds, cfg.min_train, cfg.block, cfg.mcs_refit_every, cfg.tune, cfg.tune_every) == (5, 150, 21, 21, False, 3)
    assert cfg.lgbm_threads == 1
    info = json.loads((sb.HEADLINE_RUN / "run_info.json").read_text())   # committed settings file, not data
    assert sb.check_headline_settings(info, info["days"][1]) == []
    bad = json.loads(json.dumps(info))
    bad["args"]["n_seeds"] = 3
    assert sb.check_headline_settings(bad, info["days"][1]) and sb.check_headline_settings(info, "2026-11-03")


# ---------------------------------------------------------------- the reduced model list


def test_reduced_model_list_gives_the_same_hybrid_and_mcsgarch(synth):
    m, ds = synth
    small, _ = run_walkforward(m, ds, small_cfg(11))
    many = ["naive_last", "har", "mcsgarch", "mcsgarch_iv", "implied", "linear_base", "lgbm_base", "hybrid_base",
            "hybrid_gex_us", "hybridiv_base"]
    big, _ = run_walkforward(m, ds, small_cfg(11, models=many))
    assert list(small.columns[-2:]) == ["mcsgarch", "hybrid_base"]
    for c in ("y", "mcsgarch", "hybrid_base"):
        assert np.array_equal(small[c].to_numpy(), big[c].to_numpy()), c
    other, _ = run_walkforward(m, ds, small_cfg(12))
    assert np.array_equal(other["mcsgarch"].to_numpy(), small["mcsgarch"].to_numpy())   # deterministic benchmark
    assert not np.array_equal(other["hybrid_base"].to_numpy(), small["hybrid_base"].to_numpy())


def test_work_files_round_trip_and_reproduction_check(synth, tmp_path, monkeypatch):
    m, ds = synth
    monkeypatch.setattr(sb, "make_config", lambda seed, ev, sets, **kw: small_cfg(seed))
    sb._STATE.update(market=m, datasets={"next30": ds}, sets=SETS)
    (tmp_path / "timing.csv").write_text("target,seed,seconds,rows,pid\n")
    for s in (7, 101):
        target, seed, secs, n = sb.run_job(("next30", s, str(tmp_path)))
        assert (target, seed) == ("next30", s) and n > 0
    wide = sb.load_wide(tmp_path, "next30", [7, 101])
    assert wide.attrs["mcsgarch_max_rel_diff_across_seeds"] == 0.0
    # "committed" forecasts made directly, written and read like the pipeline's files
    direct, _ = run_walkforward(m, ds, small_cfg(7))
    direct.to_csv(tmp_path / "committed.csv", index=False)
    committed = sb.read_csv_exact(tmp_path / "committed.csv")
    r = sb.reproduction_check(wide, committed, "next30")
    assert r["rows_matched"] == len(direct) == r["hybrid_base_n_identical"] == r["mcsgarch_n_identical"]
    assert r["hybrid_base_max_abs_rel_diff"] == 0.0
    committed.loc[3, "hybrid_base"] *= 1.001
    r2 = sb.reproduction_check(wide, committed, "next30")
    assert r2["hybrid_base_n_identical"] == len(direct) - 1 and r2["hybrid_base_max_abs_rel_diff"] > 9e-4
    # runs that disagree on the realized values are refused
    bad = sb.read_csv_exact(sb.work_file(tmp_path, "next30", 101))
    bad.loc[0, "y"] *= 2
    bad.to_csv(sb.work_file(tmp_path, "next30", 101), index=False)
    with pytest.raises(ValueError):
        sb.load_wide(tmp_path, "next30", [7, 101])
    assert len(pd.read_csv(tmp_path / "timing.csv")) == 2


# ---------------------------------------------------------------- per-ensemble statistics


def synthetic_wide(n_days=60, anchors=4, seeds=(7, 101, 102), gain=0.1, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for t, d in enumerate(days):
        lvl = np.exp(rng.normal(0, 0.5))
        for a in range(anchors):
            rows.append({"t": t, "date": d, "minute": 15 * a, "horizon": 15, "is_expiry": t % 5 == 3, "tdte": 0,
                         "regime": 0, "y": lvl * rng.exponential(1.0)})
    w = pd.DataFrame(rows)
    w["mcsgarch"] = np.exp(rng.normal(0, 0.6, len(w)))
    for s in seeds:
        w[f"hyb_{s}"] = np.exp(rng.normal(0, 0.6 - gain, len(w)))
    return w


def test_ensemble_rows_equal_scoreboard_and_dm():
    w = synthetic_wide()
    groups = {"all": np.ones(len(w), bool), "expiry": w["is_expiry"].to_numpy(bool)}
    rows = pd.DataFrame(sb.ensemble_rows(w, [7, 101, 102], groups, "next15"))
    assert len(rows) == 6 and set(rows["role"]) == {"headline (reproduction)", "new"}
    for s in (7, 101, 102):
        p = w.rename(columns={f"hyb_{s}": "hybrid_base"})
        board = scoreboard(p, ["mcsgarch", "hybrid_base"], groups)
        for g in ("all", "expiry"):
            r = rows[(rows.base_seed == s) & (rows.subset == g)].iloc[0]
            ref = board[(board.group == g) & (board.model == "hybrid_base")]["qlike_vs_mcsgarch_pct"].iloc[0]
            assert r["margin_pct"] == pytest.approx(ref, rel=1e-12)
            L = daily_losses(p[groups[g]], ["mcsgarch", "hybrid_base"])
            assert r["dm_p"] == diebold_mariano(L["mcsgarch"], L["hybrid_base"], alternative="greater")["p"]
            assert r["days"] == len(L)


def test_daily_loss_table_rebuilds_the_anchor_margin():
    w = synthetic_wide(anchors=3)
    tab = sb.daily_loss_table(w, [7, 101])
    assert list(tab.columns) == ["is_expiry", "anchors", "qlike_mcsgarch", "qlike_hybrid_s7", "qlike_hybrid_s101"]
    wsum = lambda c: (tab[c] * tab["anchors"]).sum() / tab["anchors"].sum()  # noqa: E731
    rows = pd.DataFrame(sb.ensemble_rows(w, [101], {"all": np.ones(len(w), bool)}, "x"))
    assert 100 * (wsum("qlike_hybrid_s101") / wsum("qlike_mcsgarch") - 1) == pytest.approx(rows["margin_pct"].iloc[0], rel=1e-10)


# ---------------------------------------------------------------- the seed bootstrap


def test_bootstrap_with_one_ensemble_is_a_plain_day_bootstrap():
    rng = np.random.default_rng(3)
    n = 120
    lb = rng.exponential(1.0, n) + 0.05
    lm = lb - 0.05 + rng.normal(0, 0.3, n)
    idx = sb.rf.stationary_bootstrap_indices(n, 999, 10.0, np.random.default_rng(4))
    out = sb.seed_bootstrap(lb, lm, idx=idx, ens=np.zeros(999, int))
    d = lb - lm
    dstar = d[idx].mean(axis=1)
    assert out["p"] == pytest.approx((1 + np.sum(dstar - d.mean() >= d.mean())) / 1000)
    assert out["p_percentile"] == pytest.approx((1 + np.sum(dstar <= 0)) / 1000)
    assert out["dbar"] == pytest.approx(d.mean())
    # identical ensembles: the ensemble draw changes nothing
    same = sb.seed_bootstrap(lb, np.column_stack([lm] * 4), idx=idx, ens=np.random.default_rng(5).integers(0, 4, 999))
    assert same["p"] == pytest.approx(out["p"]) and same["sd_between_ensembles"] == 0.0


def test_bootstrap_direction_and_seed_noise():
    rng = np.random.default_rng(6)
    n, K = 300, 20
    lb = rng.exponential(1.0, n) + 0.5
    noise = rng.normal(0, 0.2, n)
    strong = lb[:, None] - 0.3 + noise[:, None] + rng.normal(0, 0.01, (n, K))
    assert sb.seed_bootstrap(lb, strong, reps=999, rng=np.random.default_rng(1))["p"] == pytest.approx(1 / 1000)
    worse = lb[:, None] + 0.3 + noise[:, None] + rng.normal(0, 0.01, (n, K))
    assert sb.seed_bootstrap(lb, worse, reps=999, rng=np.random.default_rng(1))["p"] > 0.99
    # a modest gain, without and with large seed-to-seed offsets: the offsets widen the draws and raise p
    quiet = lb[:, None] - 0.03 + noise[:, None]
    offsets = np.linspace(-0.06, 0.06, K)
    loud = quiet + offsets[None, :]
    a = sb.seed_bootstrap(lb, quiet, reps=2000, rng=np.random.default_rng(2))
    b = sb.seed_bootstrap(lb, loud, reps=2000, rng=np.random.default_rng(2))
    assert b["sd_draw"] > 1.5 * a["sd_draw"] and b["p"] > a["p"]
    assert b["dbar"] == pytest.approx(a["dbar"])
    assert b["margin_q025"] < a["margin_q025"] and b["margin_q975"] > a["margin_q975"]


def test_bootstrap_size_under_the_null():
    """With no true gain the 5% test rejects about 5% of the time (seeds fixed), and less with seed noise."""
    def rate(seed_sd):
        rej = 0
        for r in range(200):
            rng = np.random.default_rng(1000 + r)
            n, K = 150, 5
            lb = rng.exponential(1.0, n) + 1.0
            e = np.empty(n)                       # AR(1) differential with mean exactly zero in expectation
            e[0] = rng.normal()
            for i in range(1, n):
                e[i] = 0.3 * e[i - 1] + rng.normal()
            lm = lb[:, None] + 0.2 * e[:, None] + rng.normal(0, seed_sd, K)[None, :]
            rej += sb.seed_bootstrap(lb, lm, reps=499, mean_block=5.0, rng=rng)["p"] < 0.05
        return rej / 200
    assert 0.01 <= rate(0.0) <= 0.10
    assert rate(0.03) <= 0.10


# ---------------------------------------------------------------- the registered rule


def test_rule_label():
    assert sb.rule_label([-5, -6, -4], [0.001, 0.01, 0.04]) == "holds"
    assert sb.rule_label([-5, -6, -4], [0.001, 0.01, 0.06]) == "weakened"
    assert sb.rule_label([-5, 0.1, -4], [0.001, 0.6, 0.01]) == "sign flips"
    assert sb.rule_label([-5, -4], [0.01, np.nan]) == "not computed"


def make_ens(p_next30_worst=0.01, margin_next30_worst=-3.0):
    rows = []
    for t in sb.TARGETS:
        for sub in ("all", "expiry"):
            for s in [7] + sb.NEW_SEEDS:
                m, p = -6.0, 0.001
                if t == "next30" and sub == "all" and s == 120:
                    m, p = margin_next30_worst, p_next30_worst
                rows.append({"target": t, "subset": sub, "base_seed": s, "days": 100,
                             "role": "new" if s != 7 else "headline (reproduction)", "margin_pct": m, "dm_p": p,
                             "dm_stat": 3.0})
    return pd.DataFrame(rows)


def boot_for(ens):
    return pd.DataFrame([{"target": t, "subset": s, "mean_block": sb.MEAN_BLOCK, "p": 0.001}
                         for t in sb.TARGETS for s in ("all", "expiry")])


def test_verdict_and_summary():
    e = make_ens()
    v = sb.verdict(e, boot_for(e))
    assert v.iloc[-1]["result"] == "HOLDS"
    assert (v[v.kind == "primary"]["check"].str.startswith(("next15, all", "next30, all", "PRIMARY"))).all()
    e = make_ens(p_next30_worst=0.07)
    assert sb.verdict(e, boot_for(e)).iloc[-1]["result"] == "FAILS (weakened)"
    e = make_ens(p_next30_worst=0.7, margin_next30_worst=0.2)
    assert sb.verdict(e, boot_for(e)).iloc[-1]["result"] == "FAILS (sign flips)"
    s = sb.summarize(e).set_index(["target", "subset"])
    r = s.loc[("next30", "all")]
    assert r["K"] == 20 and r["margin_max"] == 0.2 and r["dm_p_max"] == 0.7 and r["worst_seed_by_margin"] == 120
    assert r["n_p_below_alpha"] == 19 and r["headline_margin"] == -6.0 and r["headline_rank_among_new_by_margin"] == 1


def test_m43_comparison_places_the_measured_spread_beside_the_assumptions():
    summ = pd.DataFrame({"target": ["next15", "next30"], "subset": ["all", "expiry"], "days": [100, 20],
                         "margin_sd": [0.1, 0.3]})
    adj = pd.DataFrame({"target": ["next30", "next15"], "subset": ["expiry", "all"], "dm_se_pp": [3.0, 1.0],
                        **{col: [0.5, 0.2] for col in sb.M43_SCENARIOS.values()}})
    out = sb.m43_comparison(summ, adj).set_index(["target", "subset"])
    assert out.loc[("next15", "all"), "measured_sd_20_reruns"] == 0.1
    assert out.loc[("next15", "all"), "measured_sd_over_dm_se"] == pytest.approx(0.1)
    assert out.loc[("next30", "expiry"), "measured_sd_over_dm_se"] == pytest.approx(0.1)
    assert out.loc[("next30", "expiry"), "pessimistic_sd_scaled"] == 0.5
