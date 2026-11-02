"""Fast unit tests for experiments/f2_economics.py (synthetic data only, no real files)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "f2_economics.py"
_spec = importlib.util.spec_from_file_location("f2_economics", _PATH)
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)


# ---------------------------------------------------------------- PSR and DSR against hand-worked values


def test_psr_normal_returns_hand_worked():
    # SR = 0.1 per period, T = 101, skew 0, kurtosis 3:
    # denominator sqrt(1 - 0 + (3 - 1)/4 * 0.01) = sqrt(1.005) = 1.0024969
    # z = 0.1 * sqrt(100) / 1.0024969 = 0.9975093, Phi(0.9975093) = 0.8407413
    assert fe.psr(0.1, 101, 0.0, 3.0) == pytest.approx(0.8407413, abs=1e-6)


def test_psr_non_normal_returns_hand_worked():
    # skew -1, kurtosis 6: denominator sqrt(1 + 0.1 + 5/4 * 0.01) = sqrt(1.1125) = 1.0547512
    # z = 1 / 1.0547512 = 0.9480909, Phi(0.9480909) = 0.8284584 (lower than with normal returns)
    assert fe.psr(0.1, 101, -1.0, 6.0) == pytest.approx(0.8284584, abs=1e-6)
    assert fe.psr(0.1, 101, -1.0, 6.0) < fe.psr(0.1, 101, 0.0, 3.0)


def test_psr_at_its_own_sharpe_is_one_half():
    assert fe.psr(0.07, 250, 0.3, 5.0, sr_star=0.07) == pytest.approx(0.5)


def test_dsr_hand_worked():
    # N = 100, V = 0.0025 (sd of trial Sharpe ratios 0.05):
    # InvPhi(1 - 1/100) = 2.3263479, InvPhi(1 - 1/(100 e)) = 2.6802104
    # multiplier = 0.4227843 * 2.3263479 + 0.5772157 * 2.6802104 = 2.5306029; SR_0 = 0.05 * 2.5306029 = 0.1265301
    assert fe.expected_max_multiplier(100) == pytest.approx(2.5306029, abs=1e-6)
    assert fe.sr0(0.0025, 100) == pytest.approx(0.1265301, abs=1e-6)
    # selected SR = 0.2, T = 252, normal returns: z = (0.2 - 0.1265301) * sqrt(251) / sqrt(1.02) = 1.1525133
    # DSR = Phi(1.1525133) = 0.8754449
    assert fe.dsr(0.2, 252, 0.0, 3.0, 0.0025, 100) == pytest.approx(0.8754449, abs=1e-6)


def test_dsr_falls_as_trials_grow():
    vals = [fe.dsr(0.2, 252, 0.0, 3.0, 0.0025, n) for n in (10, 100, 1000)]
    assert vals[0] > vals[1] > vals[2]


def test_moments_two_point_distribution():
    # +1/-1 with equal weight: skew 0, kurtosis m4 / m2^2 = 1 / 1 = 1
    sk, ku = fe.moments([1.0, -1.0] * 50)
    assert sk == pytest.approx(0.0, abs=1e-12)
    assert ku == pytest.approx(1.0)


# ---------------------------------------------------------------- other statistics


def test_hac_t_matches_statsmodels_constant_regression():
    rng = np.random.default_rng(0)
    e = np.convolve(rng.normal(size=403), np.ones(4) / 2, mode="valid")[:400] + 0.1
    t, p = fe.hac_t(e, lags=5)
    ref = sm.OLS(e, np.ones(len(e))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    assert t == pytest.approx(ref.tvalues[0], rel=1e-10)
    assert p == pytest.approx(ref.pvalues[0] / 2 if ref.tvalues[0] > 0 else 1 - ref.pvalues[0] / 2, rel=1e-8)


def test_max_drawdown_hand_worked():
    # cumulative: 0, 1, 3, -1, 0, -2, 3 -> peak 3 after day 2, trough -2 after day 5: drawdown 5 over 3 days
    dates = pd.bdate_range("2025-01-01", periods=6)
    m = fe.max_drawdown([1, 2, -4, 1, -2, 5], dates)
    assert m["mdd"] == 5 and m["mdd_days"] == 3
    assert str(m["mdd_peak"]) == str(dates[1].date()) and str(m["mdd_trough"]) == str(dates[4].date())


def test_stationary_bootstrap_reproducible_and_in_range():
    a = fe.stationary_bootstrap_indices(50, 200, 5, np.random.default_rng([1, 2]))
    b = fe.stationary_bootstrap_indices(50, 200, 5, np.random.default_rng([1, 2]))
    assert (a == b).all() and a.min() >= 0 and a.max() < 50
    # mean block length is about 5: share of steps that are not "previous index + 1" is about 1/5
    breaks = (a[:, 1:] != (a[:, :-1] + 1) % 50).mean()
    assert 0.17 < breaks < 0.23


# ---------------------------------------------------------------- cost schedule by date


@pytest.mark.parametrize("date, stt, exch", [
    ("2023-03-31", 1.00, 0.400), ("2023-04-01", 1.25, 0.400), ("2024-09-30", 1.25, 0.400),
    ("2024-10-01", 2.00, 0.366), ("2026-03-31", 2.00, 0.366), ("2026-04-01", 5.00, 0.366)])
def test_cost_schedule_by_date(date, stt, exch):
    c = fe.cost_components(pd.Timestamp(date), slip_bp_side=1.0)
    brok = 40 / 1.5e6 * 1e4  # Rs 20 per order, two orders, Rs 15 lakh notional = 0.26667 bp
    gst = 0.18 * (brok + exch + 0.02)
    assert c["stt"] == stt and c["exchange"] == pytest.approx(exch)
    assert c["brokerage"] == pytest.approx(0.2666667, abs=1e-6)
    assert c["total"] == pytest.approx(stt + exch + 0.02 + 0.20 + brok + gst + 2.0)


def test_cost_totals_match_preregistration():
    tot = {d: round(fe.cost_components(pd.Timestamp(d))["total"], 2)
           for d in ("2023-02-01", "2024-02-01", "2025-06-02", "2026-05-04")}
    assert tot == {"2023-02-01": 4.01, "2024-02-01": 4.26, "2025-06-02": 4.97, "2026-05-04": 7.97}
    assert fe.cost_components(pd.Timestamp("2025-06-02"), slip_bp_side=3.0)["slippage"] == 6.0
    assert (fe.cost_frame(pd.bdate_range("2025-01-01", periods=3), None)["total"] == 0).all()


# ---------------------------------------------------------------- walk-forward has no look-ahead


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    r_rod = rng.normal(0, 60, n)
    neg = (rng.random(n) < 0.4).astype(float)
    r_last = -0.06 * r_rod * neg + rng.normal(0, 20, n)
    p1459 = 20000 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"r_rod_bp": r_rod, "r_last_bp": r_last, "neg": neg, "p_1459": p1459,
                         "p_1500": p1459 * 1.0001, "p_1529": p1459 * np.exp(r_last / 1e4),
                         "R_bp": (np.exp(r_last / 1e4) - 1) * 1e4, "R_D_bp": (np.exp(r_last / 1e4) / 1.0001 - 1) * 1e4},
                        index=idx)


def test_walk_forward_uses_only_earlier_days():
    d = _synthetic()
    test_idx = d.index[200:]
    wf = fe.walk_forward(d, test_idx, block=21, start=d.index[0])
    assert (wf["train_end"] < wf["block_start"]).all()
    assert (wf["n_train"] == [200 + 21 * k for k in range(len(wf))]).all()
    # change every outcome from the start of block 3 onward: blocks 1 to 3 must keep their parameters
    d2 = d.copy()
    cut = wf.loc[2, "block_start"]
    d2.loc[d2.index >= cut, ["r_last_bp", "r_rod_bp"]] *= -3.0
    wf2 = fe.walk_forward(d2, test_idx, block=21, start=d.index[0])
    cols = ["s", "a", "b", "c", "d"]
    assert np.allclose(wf.loc[:2, cols].to_numpy(float), wf2.loc[:2, cols].to_numpy(float))
    assert not np.allclose(wf.loc[3:, cols].to_numpy(float), wf2.loc[3:, cols].to_numpy(float))


def test_positions_do_not_use_the_same_day_outcome():
    d = _synthetic()
    test = d.iloc[200:]
    par = fe.params_by_day(fe.walk_forward(d, test.index, start=d.index[0]), test.index)
    cost = fe.cost_frame(test.index)["total"]
    t2 = test.copy()
    t2[["r_last_bp", "R_bp", "R_D_bp", "p_1529"]] = t2[["r_last_bp", "R_bp", "R_D_bp", "p_1529"]] * -5.0
    for rule in ("P", "S1", "S2", "D"):
        assert fe.positions(rule, test, par, cost).equals(fe.positions(rule, t2, par, cost))


def test_rules_trade_only_on_negative_gex_days_and_against_the_day():
    d = _synthetic()
    test = d.iloc[200:]
    par = fe.params_by_day(fe.walk_forward(d, test.index, start=d.index[0]), test.index)
    cost = fe.cost_frame(test.index)["total"]
    for rule in ("P", "S1", "S2", "D"):
        w = fe.positions(rule, test, par, cost)
        assert (w[test["neg"] != 1] == 0).all()
        assert (np.sign(w[w != 0]) == -np.sign(test.loc[w != 0, "r_rod_bp"])).all()
    assert fe.positions("S1", test, par, cost).abs().max() <= fe.CAP


def test_breakeven_cost_sets_mean_net_to_zero():
    d = _synthetic().iloc[200:]
    par = pd.DataFrame(1.0, index=d.index, columns=["s", "a", "b", "c", "d"])
    p = fe.run_rule("P", d, par, None)
    m = fe.metrics(p)
    c_star = m["breakeven_cost_rt"]
    net = p["w"] * d["R_bp"] - p["w"].abs() * c_star
    assert net.sum() == pytest.approx(0.0, abs=1e-8)
    assert math.isfinite(m["sr_annual"])
