"""Tests for P118 (experiments/p118_hybrid_on_paper_mcsgarch.py and the optional alt_base / mcs_mean hook of
niftygex.forecast.run_walkforward). SYNTHETIC data only (no raw data file is read).

Covers: the defaults of WalkForwardConfig and the model names are unchanged; with the hook on, the committed models
(mcsgarch, mcsgarch_iv, hybrid_base) are bit-for-bit the same as without it; when the second mcsGARCH is the
committed specification, hybridalt_base equals hybrid_base exactly (so the base forecast is swapped consistently
as offset and input); hybridalt sees the same 41 inputs with log_mcs replaced in place; mcsgarch_alt equals direct
fits of the paper's specification and P116's own driver; the "cppall" settings give mcs_cpp as mcsgarch; no
look-ahead; the job plan; the statistics (E, B, absorbed share, the TOST row's units and sign); the seed rule and
the registered decision rule; and an end-to-end dry run of run, check and analyze on two simulated markets.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex import forecast as fc
from niftygex import volatility as vol
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel
from niftygex.implied import build_implied_panel
from niftygex.simulate import SimConfig, simulate_market

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p118_hybrid_on_paper_mcsgarch",
                                               _ROOT / "experiments" / "p118_hybrid_on_paper_mcsgarch.py")
p118 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p118)
gd = p118.gd

MIN_TRAIN, BLOCK = 60, 20


@pytest.fixture(scope="module")
def setup():
    m = simulate_market(SimConfig(n_days=100, kappa=0.0, seed=8, start="2024-08-01"))
    panel, implied = build_gex_panel(m), build_implied_panel(m)
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle"}
    ds = build_dataset(m, panel, implied=implied, horizon=30, step=45)
    return m, ds, sets


def _cfg(sets, models, **over):
    return gd.make_config(7, None, sets, ["full"], models=models, min_train=MIN_TRAIN, block=BLOCK,
                          mcs_refit_every=BLOCK, n_seeds=1, **over)


@pytest.fixture(scope="module")
def runs(setup):
    m, ds, sets = setup
    committed = fc.run_walkforward(m, ds, _cfg(sets, ["mcsgarch", "mcsgarch_iv", "hybrid_base"]))
    alt = fc.run_walkforward(m, ds, _cfg(sets, p118.KIND_MODELS["cpp_base"], alt_base=p118.CPP_SPEC))
    return committed, alt


# ---------------------------------------------------------------- the hook leaves the committed code unchanged


def test_defaults_and_names_unchanged():
    cfg = fc.WalkForwardConfig()
    assert cfg.mcs_mean == "zero" and cfg.alt_base is None
    assert cfg.expiry_diurnal is True and cfg.diurnal_smooth == 5
    for k in ("lgbm", "lgbmt", "hybrid", "hybridiv", "linear"):
        assert k in fc.LEARNED_KINDS
    assert fc.parse_model("hybrid_base") == ("hybrid", "base")
    assert fc.parse_model("hybridiv_gex_us") == ("hybridiv", "gex_us")
    assert fc.parse_model("hybridalt_base") == ("hybridalt", "base")
    assert fc.parse_model("mcsgarch_alt") == (None, None)


def test_alt_models_need_alt_base(setup):
    m, ds, sets = setup
    with pytest.raises(ValueError):
        fc.run_walkforward(m, ds, _cfg(sets, ["mcsgarch", "hybridalt_base"]))


def test_hook_leaves_committed_models_bit_identical(runs):
    (a, _), (b, _) = runs
    assert np.array_equal(a["t"].to_numpy(), b["t"].to_numpy())
    for c in ("y", "mcsgarch", "mcsgarch_iv", "hybrid_base"):
        assert np.array_equal(a[c].to_numpy(), b[c].to_numpy()), c
    # and the new forecasts are new
    assert not np.array_equal(b["mcsgarch_alt"].to_numpy(), b["mcsgarch"].to_numpy())
    assert not np.array_equal(b["hybridalt_base"].to_numpy(), b["hybrid_base"].to_numpy())


def test_alt_equal_to_committed_spec_gives_the_committed_hybrid(setup, runs):
    """With the committed specification as the second mcsGARCH, log_mcs_alt == log_mcs, so hybridalt_base must equal
    hybrid_base to the last bit: the base is swapped consistently, as offset and as input, and nothing else moves."""
    m, ds, sets = setup
    pred, _ = fc.run_walkforward(m, ds, _cfg(sets, ["mcsgarch", "mcsgarch_alt", "hybrid_base", "hybridalt_base"],
                                             alt_base=dict(expiry_diurnal=True, smooth=5)))
    assert np.array_equal(pred["mcsgarch_alt"].to_numpy(), pred["mcsgarch"].to_numpy())
    assert np.array_equal(pred["hybridalt_base"].to_numpy(), pred["hybrid_base"].to_numpy())
    assert np.array_equal(pred["hybrid_base"].to_numpy(), runs[0][0]["hybrid_base"].to_numpy())


def test_hybridalt_sees_the_same_41_inputs_with_log_mcs_replaced(runs):
    _, (pred, info) = runs
    imp = {s.name: list(s.index) for s in info["importance"]}
    starts = sorted({int(k.split("@")[1]) for k in imp})
    assert len(starts) >= 2
    for s in starts:
        base, alt = imp[f"hybrid_base@{s}"], imp[f"hybridalt_base@{s}"]
        assert base == gd.HYBRID_COLUMNS and len(base) == 41
        assert alt == ["log_mcs_alt" if c == "log_mcs" else c for c in base]
    assert "mcs_alt_params" in info and len(info["mcs_alt_params"]) == len(starts)
    assert all("mean_phi" in p for p in info["mcs_alt_params"])


def test_mcsgarch_alt_equals_direct_fits_and_p116_driver(setup, runs, tmp_path):
    m, ds, sets = setup
    _, (pred, _) = runs
    R, is_exp = m["R"], m["labels"]["is_expiry"].to_numpy()
    T, M = R.shape
    data = ds.sort_values(["t", "minute"]).reset_index(drop=True)
    for start in range(MIN_TRAIN, T, BLOCK):
        end = min(start + BLOCK, T)
        h, _ = vol.daily_egarch_variance(R.sum(axis=1), train_size=start)
        mdl = vol.McsGarch(**p118.CPP_SPEC).fit(R[:start], h[:start], is_exp[:start])
        filt = mdl.filter(R, h, is_exp)
        d = data[data["t"] < end]
        tt, ii, hh = d["t"].to_numpy(), d["minute"].to_numpy(), d["horizon"].to_numpy()
        f = mdl.forecast_variance(filt, h, tt, ii, hh) + vol.arma11_mean_square_forecast(mdl, R, filt, h, tt, ii, hh)
        test = tt >= start
        got = pred[(pred["t"] >= start) & (pred["t"] < end)]
        assert np.array_equal(got["mcsgarch_alt"].to_numpy(), f[test])
    # P116's own driver (its copy of P109's run_block) gives the same mcs_cpp forecasts, bit for bit
    ex = p118.ex
    ex._STATE.update(R=R, is_exp=is_exp, days=m["days"], block=BLOCK, variants=["mcsgarch", "mcs_cpp"],
                     targets={"next30": (30, 45, None)})
    for start in range(MIN_TRAIN, T, BLOCK):
        ex.run_block(("SYNTH", start, str(tmp_path)))
    ref = ex.assemble(tmp_path, "SYNTH").sort_values(["t", "minute"]).reset_index(drop=True)
    assert np.array_equal(ref["t"].to_numpy(), pred["t"].to_numpy())
    assert np.array_equal(ref["mcs_cpp"].to_numpy(), pred["mcsgarch_alt"].to_numpy())
    assert np.array_equal(ref["mcsgarch"].to_numpy(), pred["mcsgarch"].to_numpy())


def test_cppall_settings_give_mcs_cpp_as_mcsgarch(setup, runs):
    m, ds, sets = setup
    _, (alt, _) = runs
    pred, info = fc.run_walkforward(m, ds, _cfg(sets, p118.KIND_MODELS["cppall"], **p118.KIND_SETTINGS["cppall"]))
    assert np.array_equal(pred["mcsgarch"].to_numpy(), alt["mcsgarch_alt"].to_numpy())
    assert not np.array_equal(pred["mcsgarch_iv"].to_numpy(), alt["mcsgarch_iv"].to_numpy())
    assert not np.array_equal(pred["hybrid_base"].to_numpy(), alt["hybridalt_base"].to_numpy())
    assert np.isfinite(pred["hybrid_base"]).all() and (pred["hybrid_base"] > 0).all()


def test_no_look_ahead(setup, runs):
    """Changing every return after the first test block leaves that block's forecasts unchanged."""
    m, ds, sets = setup
    _, (base, _) = runs
    m2 = dict(m)
    R2 = m["R"].copy()
    cut = MIN_TRAIN + BLOCK
    R2[cut:] *= 3.0
    m2["R"] = R2
    pred, _ = fc.run_walkforward(m2, ds, _cfg(sets, p118.KIND_MODELS["cpp_base"], alt_base=p118.CPP_SPEC))
    a = base[base["t"] < cut]
    b = pred[pred["t"] < cut]
    for c in ("mcsgarch", "mcsgarch_iv", "mcsgarch_alt", "hybrid_base", "hybridalt_base"):
        assert np.array_equal(a[c].to_numpy(), b[c].to_numpy()), c
    assert not np.array_equal(base.loc[base["t"] >= cut, "mcsgarch_alt"].to_numpy(),
                              pred.loc[pred["t"] >= cut, "mcsgarch_alt"].to_numpy())


# ---------------------------------------------------------------- plan, statistics and the decision rule


def test_job_plan():
    rep = p118.job_plan("NIFTY", "repro")
    assert rep == [("NIFTY", t, 7, "repro") for t in ("next15", "next30", "last_hour")]
    nif = p118.job_plan("NIFTY")
    assert nif[:3] == [("NIFTY", t, 7, "cpp") for t in ("next15", "next30", "last_hour")]
    assert nif[3] == ("NIFTY", "next15", 101, "cpp_base")
    assert sum(j[3] == "cpp_base" for j in nif) == 1 and len(nif) == 3 + 8 + 3
    assert nif[-3:] == [("NIFTY", t, 7, "cppall") for t in ("next15", "next30", "last_hour")]
    assert not any(j[1] == "last_hour" and j[2] != 7 for j in nif)
    bn = p118.job_plan("BANKNIFTY")
    assert bn[:2] == [("BANKNIFTY", "next15", 7, "cpp_base"), ("BANKNIFTY", "next30", 7, "cpp_base")]
    assert len(bn) == 10 and {j[3] for j in bn[2:]} == {"cpp"}
    assert p118.job_plan("FINNIFTY", seeds=[7]) == [("FINNIFTY", t, 7, "cpp_base") for t in ("next15", "next30")]
    assert p118.job_plan("NIFTY", kinds=["cppall"], targets=["next15"]) == [("NIFTY", "next15", 7, "cppall")]
    # the paper's specification, as P116 fitted it
    assert p118.CPP_SPEC == dict(mean="arma11", dist="t", expiry_diurnal=False, smooth=1)
    assert p118.MARGIN == 0.5 and p118.ALL_SEEDS == [7, 101, 102, 103, 104]


def _frame(n=500, seed=0, e_shift=0.0, b_shift=0.03):
    rng = np.random.default_rng(seed)
    cpp = 1 + rng.random(n)
    return pd.DataFrame({"mcs_cpp": cpp, "mcs_ours": cpp + b_shift + 0.01 * rng.standard_normal(n),
                         "hybrid": cpp - 0.10 + 0.02 * rng.standard_normal(n),
                         "hybrid_cpp": cpp - 0.10 - e_shift + 0.02 * rng.standard_normal(n)})


def test_e_b_absorbed_share_and_units():
    D = _frame(e_shift=0.02)
    g = p118.gain_pct
    E = p118.e_points(D)
    assert E == pytest.approx(g(D, "hybrid_cpp", "mcs_cpp") - g(D, "hybrid", "mcs_cpp"), abs=1e-12)
    assert E > 0                                                   # hybrid_cpp has the lower loss
    assert p118.b_points(D) == pytest.approx(-g(D, "mcs_ours", "mcs_cpp"), abs=1e-12)
    assert p118.pts_series(D, "hybrid", "hybrid_cpp").mean() == pytest.approx(E, abs=1e-12)
    # A = 1 when the two hybrids are equally good, A = 0 when they differ by as much as their bases
    D2 = D.copy()
    D2["hybrid_cpp"] = D2["hybrid"]
    assert p118.absorbed_share(D2) == pytest.approx(1.0)
    D3 = D.copy()
    D3["hybrid_cpp"] = D3["hybrid"] - (D3["mcs_ours"].mean() - D3["mcs_cpp"].mean())
    assert p118.absorbed_share(D3) == pytest.approx(0.0, abs=1e-12)
    r = p118.test_row(D, "tost")
    q = p118.st.tost(p118.pts_series(D, "hybrid", "hybrid_cpp"), 0.5)
    assert r["estimate"] == pytest.approx(E) and r["p"] == q["p"] and r["ci90_low"] < E < r["ci90_high"]
    d = p118.test_row(D, "diff")
    assert d["estimate"] == pytest.approx(E) and d["stat"] > 0
    h = p118.test_row(D, "h1")
    assert h["estimate"] == pytest.approx(g(D, "hybrid_cpp", "mcs_cpp")) and h["p"] < 0.001
    ci = p118.boot_intervals(D, "hybrid", "hybrid_cpp", reps=300)
    assert ci["E"][0] < E < ci["E"][1]


def test_seed_rule_and_row_labels():
    ps = {7: {"h1": 4.0, "E": 0.2}, 101: {"h1": 4.5, "E": -0.1}, 102: {"h1": 3.9, "E": 0.6}}
    assert p118.seed_agree("h1", 4.1, ps) == 3
    assert p118.seed_agree("tost", 0.2, ps) == 2
    assert p118.seed_agree("diff", 0.2, ps) == 2
    assert p118.row_label("tost", 0.01, 3, 3) == "equivalent"
    assert p118.row_label("tost", 0.01, 2, 3) == "seed-fragile"
    assert p118.row_label("tost", 0.20, 3, 3) == "not shown"
    assert p118.row_label("diff", 0.01, 3, 3) == "different"
    assert p118.row_label("h1", 0.06, 3, 3) == "not supported"
    assert p118.row_label("h1", np.nan, 3, 3) == "not computed"


def test_decision_rule():
    ho = p118.horizon_outcome
    assert ho("equivalent", "no significant difference", 0.1) == "correction absorbs the base"
    assert ho("equivalent", "different", 0.2).startswith("correction absorbs the base (a small difference")
    assert "built on mcs_cpp" in ho("equivalent", "different", 0.2)
    assert ho("not shown", "different", 0.9) == "base matters, cpp better"
    assert ho("not shown", "different", -0.9) == "base matters, ours better"
    assert ho("not shown", "different", 0.4).startswith("base matters, cpp better (the point estimate is within")
    assert ho("not shown", "no significant difference", 0.3).startswith("inconclusive (neither")
    assert ho("seed-fragile", "no significant difference", 0.1).startswith("inconclusive (a test passed")
    assert ho("not shown", "seed-fragile", 0.8).startswith("inconclusive (a test passed")
    lab = {"h1": "supported", "tost": "equivalent", "diff": "no significant difference"}
    v = p118.p118_verdict({"next15": lab, "next30": lab}, {"next15": 0.1, "next30": -0.1})
    assert v["verdict"] == "correction absorbs the base" and v["h1"].startswith("hybrid_cpp beats mcs_cpp at every")
    lab2 = {"h1": "not supported", "tost": "not shown", "diff": "different"}
    v = p118.p118_verdict({"next15": lab, "next30": lab2}, {"next15": 0.1, "next30": 0.8})
    assert v["verdict"] == "mixed: correction absorbs the base at next 15 min; base matters, cpp better at next 30 min"
    assert v["h1"] == "hybrid_cpp beats mcs_cpp at next 15 min only"


# ---------------------------------------------------------------- end-to-end dry run


def test_end_to_end_dry_run(tmp_path, monkeypatch):
    monkeypatch.setitem(p118.SYNTH, "n_days", 100)
    work, out = tmp_path / "work", tmp_path / "out"
    for ix in ("SYNTH", "SYNTHB"):
        if ix == "SYNTH":
            p118.main(["run", "--synthetic", "--index", ix, "--phase", "repro", "--workers", "1", "--work", str(work)])
        p118.main(["run", "--synthetic", "--index", ix, "--workers", "1", "--work", str(work)])
    rc = p118.main(["check", "--synthetic", "--work", str(work)])
    assert len(rc) > 20 and rc["ok"].all()
    assert {"hybrid_base vs committed hybrid_base (seed set 7)", "mcsgarch_alt vs P116's mcs_cpp forecasts",
            "mcsgarch vs mcsgarch_alt (mcs_cpp) of the seed-7 job"} <= set(rc["check"])
    p118.main(["analyze", "--synthetic", "--reps", "100", "--work", str(work), "--out", str(out)])
    for f in ("tests.csv", "tests_detail.csv", "verdict.csv", "losses.csv", "absorption.csv", "seed_repeats.csv",
              "equivalence_bounds.csv", "gain_by_anchor.csv", "reproduction_check.csv", "timing.csv", "run_info.json",
              "params_by_job.csv", "daily_losses_synth_next15.csv", "daily_losses_synthb_next30.csv"):
        assert (out / f).exists(), f
    tests = pd.read_csv(out / "tests.csv")
    assert list(tests.columns) == ["test_id", "hypothesis", "role", "sample", "estimate", "statistic", "p_raw",
                                   "sidedness", "p_method", "p_adjusted", "adjust_method"]
    prim = tests[tests["role"] == "primary"]
    assert set(prim["test_id"]) == {"P_next15_h1", "P_next30_h1", "P_next15_tost", "P_next30_tost"}
    assert prim["adjust_method"].str.startswith("Holm over 4").all()
    assert tests["p_adjusted"].between(0, 1).all()
    assert (tests["p_adjusted"] >= tests["p_raw"] - 1e-15).all()
    assert set(tests["role"]) == {"primary", "secondary", "robustness", "placebo", "exploratory"}
    assert {"S0_next15_diff", "S4_next30_tost", "R1_next15_seed7_tost", "R2_next30_tost_delta", "PL1_next15_s101",
            "E1_next30_all_vs_hybrid", "E3_next15_0915_diff"} <= set(tests["test_id"])
    rc2 = pd.read_csv(out / "reproduction_check.csv")
    assert rc2["ok"].all()
    assert rc2["check"].str.contains("rebuilt here").any()          # the cpp_base job's hybrid at seed set 101
    v = pd.read_csv(out / "verdict.csv")
    assert list(v["role"]) == ["primary (decision rule)", "secondary (same rule, applied descriptively)"]
    D = pd.read_csv(out / "daily_losses_synth_next15.csv", index_col=0)
    assert np.allclose(D["hybrid_cpp"], (D["hybrid_cpp_s7"] + D["hybrid_cpp_s101"]) / 2, rtol=0, atol=1e-15)
    ab = pd.read_csv(out / "absorption.csv")
    row = ab[(ab["index"] == "SYNTH") & (ab["target"] == "next15") & (ab["subset"] == "all days")].iloc[0]
    assert row["E_points"] == pytest.approx(row["hybrid_cpp_gain_vs_mcs_cpp_pct"] - row["hybrid_gain_vs_mcs_cpp_pct"])
