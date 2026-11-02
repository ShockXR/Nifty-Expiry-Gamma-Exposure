"""Tests for the version 2 forecast ladder and evaluation helpers."""

import numpy as np
import pandas as pd

from niftygex.evaluation import ablation_table, holm, subgroups
from niftygex.features import GEX_SETS, build_dataset, build_gex_panel
from niftygex.forecast import WalkForwardConfig, _ridge_fit_predict, parse_model, run_walkforward
from niftygex.implied import build_implied_panel
from niftygex.simulate import SimConfig, simulate_market
from niftygex.volatility import McsGarch, daily_egarch_variance


def test_new_models_run_causally_on_synthetic_market():
    m = simulate_market(SimConfig(n_days=90, kappa=0.0, seed=21))
    panel = build_gex_panel(m)
    imp = build_implied_panel(m)
    assert imp["iv_near"].notna().all()
    ds = build_dataset(m, panel, horizon=30, step=45, implied=imp)
    models = ["mcsgarch", "mcsgarch_iv", "implied", "linear_base", "linear_gex_us", "hybrid_v1",
              "hybridiv_base", "hybridiv_gex_participant", "hybrid_gex_placebo"]
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle"}
    cfg = WalkForwardConfig(min_train=60, block=15, models=models, gex_sets=sets, n_seeds=2)
    pred, info = run_walkforward(m, ds, cfg)
    assert pred["t"].min() >= 60
    vals = pred[models].to_numpy()
    assert np.all(np.isfinite(vals)) and np.all(vals > 0)
    assert info["mcs_iv_params"] and all(p["start"] >= 60 for p in info["mcs_iv_params"])
    # forecasts for a block never change when later days are removed (no look-ahead)
    cut = int(pred["t"].max()) - 5
    m2 = dict(m)
    for k in ("R", "open", "high", "low", "close", "volume"):
        m2[k] = m[k][:cut]
    m2["days"] = m["days"][:cut]
    m2["labels"] = m["labels"].iloc[:cut]
    pred2, _ = run_walkforward(m2, ds[ds["t"] < cut], cfg)
    a = pred[pred["t"] < 75].sort_values(["t", "minute"])
    b = pred2[pred2["t"] < 75].sort_values(["t", "minute"])
    for c in ("mcsgarch", "mcsgarch_iv", "implied", "linear_base"):
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy(), rtol=1e-8), c


def test_feature_rows_do_not_change_when_later_days_are_removed():
    """Every feature of day t (GEX, implied, calendar, order flow, placebo) uses only days up to t."""
    m = simulate_market(SimConfig(n_days=70, kappa=0.0, seed=8))
    panel, imp = build_gex_panel(m), build_implied_panel(m)
    full = build_dataset(m, panel, horizon=30, step=45, implied=imp)
    cut = 50
    m2 = dict(m)
    for k, v in m.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(m["days"]):
            m2[k] = v[:cut]
        elif isinstance(v, (pd.DataFrame, pd.Series)) and len(v) == len(m["days"]):
            m2[k] = v.iloc[:cut]
        elif isinstance(v, list) and len(v) == len(m["days"]):
            m2[k] = v[:cut]
    m2["days"] = m["days"][:cut]
    short = build_dataset(m2, build_gex_panel(m2), horizon=30, step=45, implied=build_implied_panel(m2))
    a = full[full["t"] < cut - 1].reset_index(drop=True)   # the target of day cut-1 still needs its own bars
    b = short[short["t"] < cut - 1].reset_index(drop=True)
    assert len(a) == len(b) > 0
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    for c in num:
        assert np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-10, atol=1e-14, equal_nan=True), c


def test_dm_uses_student_t():
    from niftygex.evaluation import diebold_mariano
    from scipy.stats import t as student_t
    rng = np.random.default_rng(5)
    l1, l2 = rng.normal(1.0, 1, 40), rng.normal(0.7, 1, 40)
    r = diebold_mariano(l1, l2, alternative="greater", lag=0)
    assert np.isclose(r["p"], student_t.sf(r["stat"], 39))


def test_parse_model_names():
    assert parse_model("hybridiv_gex_us") == ("hybridiv", "gex_us")
    assert parse_model("hybrid_base") == ("hybrid", "base")
    assert parse_model("linear_v1") == ("linear", "v1")
    assert parse_model("mcsgarch_iv") == (None, None)


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "noise": rng.normal(size=n),
                       "regime": rng.integers(0, 3, n)})
    y = 0.5 * df["a"] - 0.3 * df["b"] + 0.2 * (df["regime"] == 2) + rng.normal(0, 0.3, n)
    pred, sm, hp, coef = _ridge_fit_predict(df.iloc[:2500], y.iloc[:2500].to_numpy(), df.iloc[2500:],
                                            ["a", "b", "noise", "regime"], (1e-4, 1e-2, 1.0))
    assert abs(coef["a"] - 0.5) < 0.03 and abs(coef["b"] + 0.3) < 0.03 and abs(coef["noise"]) < 0.03
    assert abs(coef["regime_2"] - 0.2) < 0.05
    assert np.corrcoef(pred, y.iloc[2500:])[0, 1] > 0.8


def test_vectorised_mcsgarch_forecast_matches_loop():
    m = simulate_market(SimConfig(n_days=40, kappa=0.0, seed=3))
    R, is_exp = m["R"], m["labels"]["is_expiry"].to_numpy()
    h, _ = daily_egarch_variance(R.sum(axis=1), train_size=30)
    mc = McsGarch(expiry_diurnal=True, smooth=5).fit(R[:30], h[:30], is_exp[:30])
    filt = mc.filter(R, h, is_exp)
    rng = np.random.default_rng(0)
    tt, ii, hh = rng.integers(1, 40, 200), rng.integers(0, 375, 200), rng.choice([15, 30, 60], 200)
    got = mc.forecast_variance(filt, h, tt, ii, hh)
    p = mc.params
    pers = p["alpha"] + p["beta"]
    qbar = p["omega"] / (1 - pers)
    want = []
    for t, i, hn in zip(tt, ii, hh):
        j = np.arange(i, min(i + hn, 375))
        want.append(h[t] * np.sum(filt["S"][t, j] * (qbar + pers ** (j - i) * (filt["q2"][t, i] - qbar))))
    assert np.allclose(got, want, rtol=1e-10)


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_ablation_table_and_subgroups():
    rng = np.random.default_rng(2)
    dates = np.repeat(pd.bdate_range("2025-01-01", periods=60), 4)
    y = rng.exponential(1.0, len(dates))
    pred = pd.DataFrame({"date": dates, "y": y, "is_expiry": np.tile([True, False, False, False], 60),
                         "tdte": 1, "regime": 1, "base": 1.0, "good": y * 0.9 + 0.1})
    g = subgroups(pred, holdout_start="2025-02-15")
    assert {"all", "expiry", "development", "holdout", "single_weekly_thu"} <= set(g)
    g2 = subgroups(pred, holdout_start="2025-02-15", auction_start="2025-03-10")
    assert g2["pre_auction"].sum() + g2["auction_period"].sum() == len(pred)
    assert g2["holdout_pre_auction"].sum() < g2["holdout"].sum()
    assert not np.any(g2["pre_auction"] & (pd.to_datetime(pred["date"]) >= "2025-03-10").to_numpy())
    t = ablation_table(pred, [("fam", "good", "base", "good")], {"all": g["all"]})
    assert t.loc[0, "change_pct"] < 0 and t.loc[0, "p_one_sided"] < 0.01


def test_scoreboard_trimmed_mse_and_anchor_groups():
    from niftygex.evaluation import gain_by_anchor, scoreboard, trimmed_mean
    assert np.isclose(trimmed_mean(np.r_[np.ones(99), 1000.0], 0.01), 1.0)
    rng = np.random.default_rng(3)
    dates = np.repeat(pd.bdate_range("2025-01-01", periods=30), 4)
    y = rng.exponential(1.0, len(dates))
    pred = pd.DataFrame({"date": dates, "y": y, "minute": np.tile([0, 15, 30, 45], 30),
                         "is_expiry": False, "mcsgarch": 1.0, "other": y * 0.5 + 0.5})
    g = subgroups(pred, auction_start="2025-01-20")
    assert g["open_0915"].sum() == 30 and g["excl_0915"].sum() == 90
    assert (g["pre_auction_excl_0915"] <= g["pre_auction"]).all()
    sb = scoreboard(pred, ["mcsgarch", "other"], g)
    row = sb[(sb.model == "other") & (sb.group == "all")].iloc[0]
    assert {"qlike_vs_mcsgarch_pct", "mse_vs_mcsgarch_pct", "qlike_trim_vs_mcsgarch_pct"} <= set(sb.columns)
    assert row["qlike_trim"] <= row["qlike"]
    ga = gain_by_anchor(pred, ["mcsgarch", "other"])
    assert set(ga["minute"]) == {0, 15, 30, 45}
    assert np.allclose(ga[ga.model == "mcsgarch"]["qlike_vs_ref_pct"], 0)
