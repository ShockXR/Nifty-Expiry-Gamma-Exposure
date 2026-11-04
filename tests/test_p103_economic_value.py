"""Fast unit tests for experiments/p103_economic_value.py (synthetic data only, no real files)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p103_economic_value.py"
_spec = importlib.util.spec_from_file_location("p103_economic_value", _PATH)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)


# ---------------------------------------------------------------- registered constants


def test_registered_constants_match_the_plan():
    assert (pe.SR_A, pe.GAMMA) == (0.4, 2.0)
    assert pe.TARGET == pytest.approx(0.20)
    assert (pe.CAP, pe.CONST_W, pe.N_PRIMARY) == (4.0, 2.0, 4)
    assert pe.N_WIN_YEAR == 3024
    assert pe.SIGMA_W == pytest.approx(0.20 / math.sqrt(3024))
    assert pe.SR_W == pytest.approx(0.4 / math.sqrt(3024))
    assert pe.ORIGINS["next30_primary"] == list(range(0, 331, 30))
    assert len(pe.ORIGINS["next30_primary"]) == 12
    assert pe.ORIGINS["next15_all"] == list(range(0, 361, 15))
    assert pe.ORIGINS["last_hour"] == [315]
    assert pe.STRATS == ["hybrid_base", "mcsgarch", "mcsgarch_iv", "constant"]
    assert (pe.MASTER_SEED, pe.STUDY_SEED) == (20261102, 103)


# ---------------------------------------------------------------- the utility formula


def test_bracket_hand_worked_values():
    # RV/F = 1: 1 - 1/2 = 1/2 (the maximum); RV/F = 4: 2 - 2 = 0; RV/F = 1/4: 1/2 - 1/8 = 3/8
    assert pe.bracket(1.0, 1.0) == pytest.approx(0.5)
    assert pe.bracket(4.0, 1.0) == pytest.approx(0.0)
    assert pe.bracket(1.0, 4.0) == pytest.approx(0.375)
    assert pe.bracket(0.0, 1.0) == pytest.approx(0.0)


def test_utility_loss_is_half_minus_bracket_and_nonnegative():
    rng = np.random.default_rng(0)
    y, f = rng.lognormal(0, 1, 1000), rng.lognormal(0, 1, 1000)
    np.testing.assert_allclose(pe.utility_loss(y, f), 0.5 - pe.bracket(y, f), atol=1e-12)
    assert (pe.utility_loss(y, f) >= -1e-15).all()
    assert (pe.bracket(y, f) <= 0.5 + 1e-15).all()


def test_annual_utility_of_a_perfect_forecast_is_sr2_over_2gamma():
    assert pe.annual_utility_pct(0.5) == pytest.approx(4.0)  # 100 x 0.16 / 2 x 0.5
    assert pe.max_utility_pct() == pytest.approx(4.0)
    assert pe.max_utility_pct(0.2, 1.0) == pytest.approx(2.0)


def test_annualisation_does_not_depend_on_the_window_length():
    # n windows a year, per-window Sharpe SR_a / sqrt(n): yearly utility = n x (SR_w^2 / gamma) x mean bracket
    rng = np.random.default_rng(1)
    b = pe.bracket(rng.lognormal(0, 0.5, 5000), 1.0)
    for n in (252, 3024, 6300):
        sr_w = pe.SR_A / math.sqrt(n)
        per_year = n * (sr_w ** 2 / pe.GAMMA) * b.mean() * 100
        assert per_year == pytest.approx(float(pe.annual_utility_pct(b.mean())))


def test_optimal_forecast_under_noisy_rv_is_slightly_above_the_mean():
    # Section 3.8 of the plan: maximising E[bracket] over F gives F = (E RV)^2 / (E sqrt RV)^2 >= E RV
    rng = np.random.default_rng(2)
    y = rng.chisquare(10, 200_000) / 10.0  # mean 1, noisy
    grid = np.linspace(0.8, 1.4, 601)
    best = grid[np.argmax([pe.bracket(y, f).mean() for f in grid])]
    theory = y.mean() ** 2 / np.sqrt(y).mean() ** 2
    assert best == pytest.approx(theory, abs=2e-3)
    assert theory > y.mean()


def test_size_labels():
    assert pe.size_label(0.04) == "tiny"
    assert pe.size_label(0.0999) == "tiny"
    assert pe.size_label(0.10) == "small"
    assert pe.size_label(0.49) == "small"
    assert pe.size_label(0.50) == "material"
    assert pe.size_label(float("nan")) == "tiny"


# ---------------------------------------------------------------- non-overlapping windows


def _pred(days=5, origins=range(0, 346, 15), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=days):
        for m in origins:
            rows.append({"date": d, "minute": m, "y": rng.lognormal(-12, 0.5)})
    return pd.DataFrame(rows)


def test_select_windows_keeps_only_registered_origins():
    p = _pred()
    s = pe.select_windows(p, pe.ORIGINS["next30_primary"])
    assert len(s) == 5 * 12
    assert sorted(s["minute"].unique()) == list(range(0, 331, 30))


def test_select_windows_refuses_a_day_with_a_missing_origin():
    p = _pred()
    p = p[~((p["date"] == p["date"].iloc[0]) & (p["minute"] == 60))]
    with pytest.raises(ValueError):
        pe.select_windows(p, pe.ORIGINS["next30_primary"])


def test_registered_window_sets_do_not_overlap_and_the_overlapping_one_does():
    for k in ("next30_primary", "next30_offset15", "next30_no0915"):
        assert pe.windows_do_not_overlap(pe.ORIGINS[k], 30)
        assert max(pe.ORIGINS[k]) + 30 <= 375
    assert pe.windows_do_not_overlap(pe.ORIGINS["next15_all"], 15)
    assert max(pe.ORIGINS["next15_all"]) + 15 == 375
    assert not pe.windows_do_not_overlap(pe.ORIGINS["next30_all"], 30)


def test_window_matrices_sum_minutes_in_each_window():
    rng = np.random.default_rng(3)
    R = rng.normal(0, 1e-3, (4, 375))
    lr, rv = pe.window_matrices(R, [0, 30], 30)
    np.testing.assert_allclose(lr[:, 1], R[:, 30:60].sum(axis=1))
    np.testing.assert_allclose(rv[:, 0], (R[:, :30] ** 2).sum(axis=1))


# ---------------------------------------------------------------- positions, turnover and costs


def test_positions_are_target_over_forecast_vol_and_capped():
    F = np.array([pe.SIGMA_W ** 2, (pe.SIGMA_W / 2) ** 2, (pe.SIGMA_W / 10) ** 2, (2 * pe.SIGMA_W) ** 2])
    np.testing.assert_allclose(pe.positions(F), [1.0, 2.0, 4.0, 0.5])


def test_turnover_hand_worked():
    tv, oc = pe.daily_turnover(np.array([[1.0, 2.0, 0.5], [2.0, 2.0, 2.0]]))
    np.testing.assert_allclose(tv, [1 + 1 + 1.5 + 0.5, 4.0])  # open, changes, close
    np.testing.assert_allclose(oc, [1.5, 4.0])


def test_cost_is_half_turnover_times_round_trip_for_any_leg_split():
    # Within a day the rule starts and ends flat, so units bought = units sold = turnover / 2, and charging the
    # buy-leg rate on buys and the sell-leg rate on sells gives exactly (turnover / 2) x (buy rate + sell rate).
    rng = np.random.default_rng(4)
    W = rng.uniform(0, 4, (50, 12))
    c_buy, c_sell = 0.9, 4.1  # bp per unit, e.g. stamp duty on buys, STT on sells
    full = np.concatenate([np.zeros((50, 1)), W, np.zeros((50, 1))], axis=1)
    dw = np.diff(full, axis=1)
    legs = np.clip(dw, 0, None).sum(axis=1) * c_buy + np.clip(-dw, 0, None).sum(axis=1) * c_sell
    tv, _ = pe.daily_turnover(W)
    np.testing.assert_allclose(pe.daily_cost_bp(tv, c_buy + c_sell), legs)


def test_round_trip_costs_reuse_p90_schedule():
    rt = lambda d: pe.fe.cost_components(pd.Timestamp(d), 1.0)["total"]  # noqa: E731
    assert rt("2024-01-15") == pytest.approx(4.26, abs=0.005)
    assert rt("2025-01-15") == pytest.approx(4.97, abs=0.005)
    assert rt("2026-05-01") == pytest.approx(7.97, abs=0.005)
    assert pe.fe.cost_components(pd.Timestamp("2025-01-15"), 0.0)["slippage"] == 0.0


# ---------------------------------------------------------------- tracking and net utility


def test_tracking_loss_is_zero_for_a_perfect_uncapped_forecast():
    rng = np.random.default_rng(5)
    Y = rng.lognormal(np.log(pe.SIGMA_W ** 2) + 0.3, 0.6, (20, 12))
    W = pe.SIGMA_W / np.sqrt(Y)  # perfect forecast, cap not binding by construction here
    assert (W < pe.CAP).all()
    np.testing.assert_allclose(pe.realized_strategy_vol(W, Y), pe.TARGET)
    np.testing.assert_allclose(pe.tracking_loss(W, Y), 0.0, atol=1e-20)
    np.testing.assert_allclose(pe.tracking_loss(2 * W, Y), math.log(2) ** 2)


def test_net_utility_of_a_perfect_forecast_without_costs_is_the_maximum():
    rng = np.random.default_rng(6)
    Y = rng.lognormal(np.log(pe.SIGMA_W ** 2) + 0.3, 0.6, (10, 12))
    W = pe.SIGMA_W / np.sqrt(Y)
    u = pe.net_utility_daily(W, Y, np.zeros(10))
    # each window: SR_w sigma_w - gamma/2 sigma_w^2 = SR_w^2 / (2 gamma) because sigma_w = SR_w / gamma
    np.testing.assert_allclose(u, 12 * pe.SR_W ** 2 / (2 * pe.GAMMA))
    assert u.mean() * pe.DAYS_YEAR * 100 == pytest.approx(pe.max_utility_pct())
    # costs come straight off, in bp of capital
    np.testing.assert_allclose(pe.net_utility_daily(W, Y, np.full(10, 5.0)), u - 5e-4)


# ---------------------------------------------------------------- synthetic sanity check of the comparison


def _synthetic_forecasts(days=250, noise_a=0.15, noise_b=0.45, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=days):
        level = rng.lognormal(-13, 0.4)
        for m in pe.ORIGINS["next30_primary"]:
            sig2 = level * (1.5 if m == 0 else 1.0)
            y = sig2 * rng.chisquare(30) / 30
            rows.append({"date": d, "minute": m, "y": y, "good": sig2 * rng.lognormal(0, noise_a),
                         "bad": sig2 * rng.lognormal(0, noise_b)})
    return pd.DataFrame(rows)


def test_better_forecast_earns_a_positive_fee_and_the_test_favours_it():
    p = _synthetic_forecasts()
    r = pe.utility_comparison(p, "good", "bad")
    assert r["fee_pct_year"] > 0
    assert r["p_one_sided"] < 0.01
    assert r["utility_model_pct_year"] < pe.max_utility_pct()
    back = pe.utility_comparison(p, "bad", "good")
    assert back["p_one_sided"] > 0.99
    assert back["fee_pct_year"] == pytest.approx(-r["fee_pct_year"])


def test_identical_forecasts_have_zero_fee():
    p = _synthetic_forecasts(days=40)
    p["copy"] = p["good"]
    r = pe.utility_comparison(p, "copy", "good")
    assert r["fee_pct_year"] == pytest.approx(0.0, abs=1e-12)
    assert r["fee_rs_per_crore_year"] == pytest.approx(0.0, abs=1e-6)
