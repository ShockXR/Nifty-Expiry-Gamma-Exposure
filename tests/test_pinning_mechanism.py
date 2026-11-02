"""Unit tests for experiments/pinning_mechanism.py. All data here are SYNTHETIC (hand-built chains and panels)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "pinning_mechanism.py"
_spec = importlib.util.spec_from_file_location("pinning_mechanism", _PATH)
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)


# ---------------------------------------------------------------- a hand-built chain (SYNTHETIC)

STRIKES = np.arange(21900.0, 22201.0, 50.0)            # 21900 ... 22200
OI_BY_STRIKE = np.array([100, 200, 300, 400, 500, 600, 700], float) * 75   # calls plus puts, in shares


def test_bracket_strikes():
    assert list(pm.bracket_strikes(22012.0, 50.0)) == [22000.0, 22050.0]
    assert list(pm.bracket_strikes(22012.0, 50.0, 2)) == [21950.0, 22000.0, 22050.0, 22100.0]
    # a previous close exactly on a strike: that strike and the next one up
    assert list(pm.bracket_strikes(22000.0, 50.0)) == [22000.0, 22050.0]
    assert list(pm.bracket_strikes(44327.8, 100.0)) == [44300.0, 44400.0]


def test_c1_near_money_share_on_hand_chain():
    total = OI_BY_STRIKE.sum()
    # S = 22012: strikes 22000 (OI 300*75) and 22050 (OI 400*75)
    assert pm.near_money_share(STRIKES, OI_BY_STRIKE, 22012.0, 50.0) == pytest.approx((300 + 400) * 75 / total)
    # wider window (C1w): 21950, 22000, 22050, 22100
    assert pm.near_money_share(STRIKES, OI_BY_STRIKE, 22012.0, 50.0, 2) == pytest.approx((200 + 300 + 400 + 500) * 75 / total)
    # calls and puts listed separately at the same strike are both counted
    k2 = np.concatenate([STRIKES, STRIKES])
    oi2 = np.concatenate([OI_BY_STRIKE / 2, OI_BY_STRIKE / 2])
    assert pm.near_money_share(k2, oi2, 22012.0, 50.0) == pytest.approx((300 + 400) * 75 / total)
    assert np.isnan(pm.near_money_share(STRIKES, np.zeros(7), 22012.0, 50.0))


def test_c3_lot_rules():
    oi = np.array([75 * 3, 75 * 10, 75 * 7, 0, np.nan])
    assert pm.lot_from_gcd(oi) == 75
    contracts = np.array([4, 10, 0, 2], float)
    strike = np.array([22000, 22050, 22100, 21950], float)
    prem = np.array([120.5, 80.0, 50.0, 200.0])
    value = contracts * 75 * (strike + prem)
    assert pm.lot_from_turnover(value, contracts, strike, prem) == 75
    # NewBrdLotQty wins when present; divisor rule otherwise; turnover rule when they disagree
    assert pm.choose_lot(65.0, 75.0, 75.0) == 65
    assert pm.choose_lot(np.nan, 25.0, 25.0) == 25
    assert pm.choose_lot(np.nan, 50.0, 25.0) == 25
    assert pm.choose_lot(np.nan, 25.0, 25.0, gcd_ok=False) == 25
    assert pm.choose_lot(np.nan, 75.0, 25.0, gcd_ok=False) == 25
    # contract value per lot (C3 before the log)
    assert np.log(75 * 22000.0) == pytest.approx(pm.np.log(pm.choose_lot(np.nan, 75.0, 75.0) * 22000.0))


def test_c2_turnover_share_and_c5_ratio():
    # SYNTHETIC day: three expiries' notional turnover; the expiring one is 9 of 12
    turn = pd.Series({"E1": 9.0e11, "E2": 2.0e11, "E3": 1.0e11})
    assert turn["E1"] / turn.sum() == pytest.approx(0.75)
    # C5: log of expiring OI over next-expiry OI
    assert pm.safe_log_ratio(300.0, 100.0) == pytest.approx(np.log(3.0))
    assert np.isnan(pm.safe_log_ratio(300.0, 0.0))
    assert np.isnan(pm.safe_log_ratio(np.nan, 1.0))


def test_c6_rms_uses_only_past_returns_and_spacing_ratio():
    days = pd.bdate_range("2024-01-01", periods=30)
    closes = pd.Series(20000.0 * np.exp(0.01 * np.arange(30)), index=days)
    closes.iloc[25:] *= 1.5          # a jump on day 25 and after must not enter the RMS for t = day 25
    rms = pm.rms_recent_returns(closes, days[25])
    assert rms == pytest.approx(0.01)
    assert np.isnan(pm.rms_recent_returns(closes, days[10]))    # fewer than 20 past returns
    assert pm.spacing_ratio(50.0, 20000.0, 0.01) == pytest.approx(np.log(0.25))


def test_c4_weekly_dates_holiday_rollback_monthly_and_count():
    days = pd.bdate_range("2024-01-01", "2024-02-29")
    days = days[days != pd.Timestamp("2024-01-23")]              # a Tuesday holiday
    per = pd.DataFrame({"start": [pd.Timestamp("2024-01-01")], "end": [pd.Timestamp("2024-02-29")], "weekday": ["Tue"]})
    sched = pm.scheduled_weekly_dates("2024-01-01", "2024-02-29", 1, days)
    assert pd.Timestamp("2024-01-22") in sched and pd.Timestamp("2024-01-23") not in sched
    weekly = pm.product_weekly_dates(per, days)
    # the last Tuesday of each month (30 Jan, 27 Feb) is the monthly contract, not a weekly
    assert pd.Timestamp("2024-01-30") not in weekly and pd.Timestamp("2024-02-27") not in weekly
    assert pd.Timestamp("2024-01-22") in weekly and pd.Timestamp("2024-02-20") in weekly
    other = pd.DatetimeIndex(["2024-01-25", "2024-01-26"])
    cnt = pm.weekly_count(pd.DatetimeIndex(["2024-01-22", "2024-01-26", "2024-01-29"]), {"A": weekly, "B": other})
    assert list(cnt) == [3, 3, 0]


# ---------------------------------------------------------------- regression helpers (SYNTHETIC panels)


def _panel(gamma_true, seed=0, n_weeks=200):
    rng = np.random.default_rng(seed)
    rows = []
    cells = ["NIFTY:a:pre", "NIFTY:b:pre", "BANKNIFTY:c:pre"]
    for w in range(n_weeks):
        cell = cells[w * len(cells) // n_weeks]
        sym = cell.split(":")[0]
        for d in range(5):
            E = float(d == 3)
            z = rng.normal()
            M = float(w % 4 == 3)
            y = 0.25 - 0.05 * E * (cell == "NIFTY:b:pre") + gamma_true * E * z + 0.01 * M * E + rng.normal(0, 0.12)
            rows.append({"symbol": sym, "cell": cell, "E": E, "M": M, "z_X": z, "y": y, "week": f"w{w}"})
    df = pd.DataFrame(rows)
    df["strata"] = df["cell"] + ":" + df["E"].astype(int).astype(str)
    return df


def test_primary_regression_recovers_planted_interaction():
    df = _panel(-0.06, seed=1)
    f = pm.fit_model(df, "z_X")
    assert f["gamma"] == pytest.approx(-0.06, abs=4 * f["gamma_se"])
    assert f["gamma_ci_low"] < -0.06 < f["gamma_ci_high"]
    assert f["gamma_p"] < 0.01
    assert abs(f["d"]) < 4 * f["d_se"]        # no slope planted on ordinary days
    assert f["n_clusters"] == 200
    f0 = pm.fit_model(_panel(0.0, seed=2), "z_X")
    assert abs(f0["gamma"]) < 4 * f0["gamma_se"]


def test_permutation_p_small_for_planted_effect_and_not_for_null():
    df = _panel(-0.08, seed=3)
    f = pm.fit_model(df, "z_X")
    p = pm.perm_gamma_p(f, "z_X", "E", "strata", np.random.default_rng(0), n=999)
    assert p < 0.01
    df0 = _panel(0.0, seed=4)
    f0 = pm.fit_model(df0, "z_X")
    p0 = pm.perm_gamma_p(f0, "z_X", "E", "strata", np.random.default_rng(0), n=999)
    assert p0 > 0.01


def test_cluster_se_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    df = _panel(-0.03, seed=5)
    X = pm.prune_collinear(pm.build_design(df, "z_X"))
    r = pm.ols_cluster(X.to_numpy(), df["y"].to_numpy(), df["week"].to_numpy())
    ref = sm.OLS(df["y"].to_numpy(), X.to_numpy()).fit(cov_type="cluster",
                                                       cov_kwds={"groups": pd.factorize(df["week"])[0]})
    assert np.allclose(r["beta"], ref.params)
    assert np.allclose(r["se"], ref.bse, rtol=1e-6)


def test_prune_collinear_keeps_gamma_and_drops_duplicates():
    df = _panel(0.0, seed=6)
    X = pm.build_design(df, "z_X")
    X["dup"] = X["d"] * 0 + X.iloc[:, 0]
    Xp = pm.prune_collinear(X)
    assert "gamma" in Xp.columns and "d" in Xp.columns and "dup" not in Xp.columns
    X2 = X.copy()
    X2["gamma"] = X2[[c for c in X2.columns if c.startswith("b[")]].sum(axis=1)   # gamma equal to E: collinear
    with pytest.raises(ValueError):
        pm.prune_collinear(X2.drop(columns=["dup"]))


def test_between_regime_helpers():
    cell = np.array(["A"] * 4 + ["B"] * 4)
    E = np.array([1, 0, 1, 0, 1, 0, 1, 0], float)
    y = np.array([0.1, 0.3, 0.2, 0.2, 0.3, 0.25, 0.3, 0.25])
    g = pm.cell_gaps(cell, E, y)
    assert g["A"] == pytest.approx(0.15 - 0.25) and g["B"] == pytest.approx(0.3 - 0.25)
    z = np.array([1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0])
    pg = pm.predicted_gaps(cell, E, z, d=0.0, gamma=-0.05)
    assert pg["A"] == pytest.approx(-0.05) and pg["B"] == pytest.approx(0.05)
    obs = np.array([-0.07, 0.11, 0.03])
    w = np.array([1.0, 2.0, 0.5])
    assert pm.share_explained(obs, obs, w) == pytest.approx(1.0)
    assert pm.share_explained(0.5 * obs, obs, w) == pytest.approx(0.5)


def test_quarter_shifted_grid_mean_is_g_over_4_under_symmetric_pull():
    """Placebo P3: any pull toward strikes that is equally likely from either side leaves the mean distance
    to the quarter-shifted grid at exactly g/4."""
    rng = np.random.default_rng(7)
    ps = pm.ps
    for g in (50.0, 100.0):
        K = g * rng.integers(400, 500, 5000)
        v = rng.uniform(0, g / 2, 5000) * rng.beta(0.5, 3, 5000)      # strongly pulled toward the strike
        closes = np.concatenate([K + v, K - v])                      # symmetric
        dq = ps.dist_to_grid(closes, g, offset=g / 4)
        assert dq.mean() == pytest.approx(g / 4, abs=1e-9)
        # whereas the distance to real strikes is far below g/4
        assert ps.dist_to_grid(closes, g).mean() < g / 8
