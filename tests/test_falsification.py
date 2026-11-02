"""Fast unit tests for the helpers of experiments/falsification.py (synthetic data only, no real files)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "falsification.py"
_spec = importlib.util.spec_from_file_location("falsification", _PATH)
fz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fz)


# ---------------------------------------------------------------- regression helpers


def test_ols_hac_matches_statsmodels():
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    e = np.convolve(rng.normal(size=n + 3), np.ones(4) / 2, mode="valid")[:n]  # autocorrelated errors
    y = X @ np.array([0.5, 1.0, -2.0]) + e
    b, se, t, p, r2 = fz.ols_hac(y, X, lags=5)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    assert np.allclose(b, ref.params)
    assert np.allclose(se, ref.bse, rtol=1e-10)
    assert np.allclose(p, ref.pvalues, rtol=1e-8)
    assert np.isclose(r2, ref.rsquared)


def test_fit_named_drops_collinear_regime_dummies():
    rng = np.random.default_rng(1)
    n = 120
    reg = np.repeat(["a", "b"], n // 2)
    X = pd.DataFrame({"x": rng.normal(size=n), "reg_a": (reg == "a").astype(float), "reg_b": (reg == "b").astype(float)})
    y = pd.Series(1.0 + 2.0 * X["x"] + rng.normal(size=n) * 0.1)
    res = fz.fit_named(y, X)
    # reg_a + reg_b equals the constant, so the later one is dropped and the fit stays well defined
    assert "reg_b" not in res.index and "reg_a" in res.index
    assert abs(res.loc["x", "coef"] - 2.0) < 0.05


def test_iv_recovers_effect_and_equals_ratio():
    rng = np.random.default_rng(2)
    n = 4000
    z = rng.normal(size=n)
    u = rng.normal(size=n)                      # confounder
    x = -1.0 * z + u + rng.normal(size=n) * 0.5  # endogenous regressor
    y = 0.7 * x + 2.0 * u + rng.normal(size=n)
    exog = np.ones((n, 1))
    r = fz.iv_2sls_hac(y, exog, x, z, lags=2)
    assert abs(r["beta_iv"] - 0.7) < 0.1
    assert np.isclose(r["beta_iv"], r["reduced_form_coef"] / r["first_stage_coef"])
    assert r["first_stage_F_hac"] > 100
    # OLS is biased by the confounder; IV is not
    b_ols = np.polyfit(x, y, 1)[0]
    assert abs(b_ols - 0.7) > 0.3


# ---------------------------------------------------------------- permutation helpers


def test_permute_within_groups_keeps_values_inside_groups():
    rng = np.random.default_rng(3)
    x = np.arange(30.0)
    groups = np.repeat([0, 1, 2], 10)
    out = fz.permute_within_groups(x, groups, rng)
    for g in range(3):
        assert sorted(out[groups == g]) == sorted(x[groups == g])
    assert not np.array_equal(out, x)
    assert np.array_equal(x, np.arange(30.0))  # input untouched


def test_circular_shift_is_a_rotation_inside_each_group():
    rng = np.random.default_rng(4)
    x = np.arange(12.0)
    groups = np.repeat(["a", "b"], 6)
    out = fz.circular_shift_within_groups(x, groups, rng)
    for g in ("a", "b"):
        seg, orig = out[groups == g], x[groups == g]
        assert any(np.array_equal(seg, np.roll(orig, k)) for k in range(1, 6))


def test_block_ids_and_perm_pvalues():
    assert list(fz.block_ids(7, 3)) == [0, 0, 0, 1, 1, 1, 2]
    draws = np.array([-3.0, -1.0, 0.0, 1.0, 2.0, 0.5, -0.5, 0.2, -0.2])
    pv = fz.perm_pvalues(2.5, draws, 2.5, draws)
    assert np.isclose(pv["p_two_sided_t"], (1 + 1) / (9 + 1))  # only |-3| >= 2.5
    assert np.isclose(pv["p_one_sided_t"], (1 + 0) / (9 + 1))  # nothing >= 2.5
    assert pv["n_valid"] == 9


# ---------------------------------------------------------------- expiring-gamma split


def _toy_book():
    days = pd.bdate_range("2024-01-01", "2024-01-19")  # three weeks, Monday to Friday
    thursdays = days[days.weekday == 3]
    rows = []
    for d in days:
        for k, e in enumerate(thursdays):
            if e > d:  # the loader keeps only contracts expiring after the publish date
                rows.append({"date": d, "expiry": e, "C": 10.0 * (k + 1) + d.day / 100, "P": 1.0 * (k + 1), "oi": 5.0})
    return days, thursdays, pd.DataFrame(rows)


def test_split_at_real_expiry_day():
    days, thursdays, tab = _toy_book()
    e = thursdays[1]  # 11 Jan 2024
    sp = fz.split_at(tab, days, [e])
    row = sp.loc[e]
    q = days[days.get_loc(e) - 1]
    assert row["prev_day"] == q and row["front_expires_on_event_day"]
    book_q = tab[tab["date"] == q]
    assert np.isclose(row["front_C"], book_q.loc[book_q["expiry"] == e, "C"].sum())
    assert np.isclose(row["rest_C"], book_q.loc[book_q["expiry"] > e, "C"].sum())
    # the surviving book at the close of e excludes the contract that expired on e
    book_e = tab[tab["date"] == e]
    assert np.isclose(row["post_C"], book_e.loc[book_e["expiry"] > e, "C"].sum())
    assert np.isclose(row["front_C"] + row["rest_C"], book_q["C"].sum())


def test_split_at_pseudo_day_keeps_front_alive():
    days, thursdays, tab = _toy_book()
    p = pd.Timestamp("2024-01-09")  # a Tuesday: nothing expires
    row = fz.split_at(tab, days, [p]).loc[p]
    assert not row["front_expires_on_event_day"]
    assert row["front_expiry"] == thursdays[1]
    # the front contract is still in the book after the pseudo day
    assert row["post_C"] > row["rest_C"]


# ---------------------------------------------------------------- placebo dates


def test_shifted_placebo_dates():
    days = pd.bdate_range("2024-01-01", "2024-02-09")
    exp = days[days.weekday == 3]
    plus2 = fz.shifted_placebo_dates(days, exp, 2)
    assert len(plus2) > 0 and not plus2.isin(exp).any()
    assert (plus2.weekday == 0).all()  # Thursday + 2 trading days = Monday
    # with the default exclusion, -1 (the day before expiry) never qualifies: the next day is an expiry day
    assert len(fz.shifted_placebo_dates(days, exp, -1)) == 0
    minus1 = fz.shifted_placebo_dates(days, exp, -1, exclude_adjacent=False)
    assert (minus1.weekday == 2).all() and len(minus1) == len(exp)


def test_weekday_placebo_dates_follow_regime():
    days = pd.bdate_range("2025-08-04", "2025-09-26")
    regime = np.where(days < pd.Timestamp("2025-09-01"), "single_weekly_thu", "single_weekly_tue")
    exp = days[((days.weekday == 3) & (days < pd.Timestamp("2025-09-01"))) |
               ((days.weekday == 1) & (days >= pd.Timestamp("2025-09-01")))]
    out = fz.weekday_placebo_dates(days, exp, regime)
    before, after = out[out < pd.Timestamp("2025-09-01")], out[out >= pd.Timestamp("2025-09-01")]
    assert (before.weekday == 1).all() and len(before) > 0
    assert (after.weekday == 3).all() and len(after) > 0
    assert not out.isin(exp).any()
