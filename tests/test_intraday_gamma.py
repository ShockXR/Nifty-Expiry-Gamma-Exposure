"""Fast unit tests for experiments/intraday_gamma.py on synthetic inputs (no real data needed)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "intraday_gamma.py"
_spec = importlib.util.spec_from_file_location("intraday_gamma", _PATH)
ig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ig)

YEAR_MIN = 375 * 252


def _hand_spot_gamma(S, K, T, sigma, r=0.065, carry=0.065):
    """Black-76 spot gamma written out with the math module only."""
    F = S * math.exp(carry * T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    gamma_f = math.exp(-r * T) * pdf / (F * sigma * math.sqrt(T))
    return gamma_f * math.exp(carry * T) ** 2


# ---------------------------------------------------------------- gamma at spot


def test_gamma_at_spot_matches_hand_calculation():
    S, T = 22013.0, 47.0 / YEAR_MIN
    K = np.array([21950.0, 22000.0, 22050.0])
    oc = np.array([1.0e6, 4.0e6, 2.5e6])
    op = np.array([3.0e6, 2.0e6, 0.5e6])
    ivc = np.array([0.13, 0.12, 0.115])
    ivp = np.array([0.14, 0.125, 0.12])
    g = ig.gamma_at_spot([S], [T], K, oc, op, ivc, ivp)
    call = sum(oc[i] * _hand_spot_gamma(S, K[i], T, ivc[i]) * S**2 * 0.01 for i in range(3))
    put = sum(op[i] * _hand_spot_gamma(S, K[i], T, ivp[i]) * S**2 * 0.01 for i in range(3))
    assert g["call"][0] == pytest.approx(call, rel=1e-10)
    assert g["put"][0] == pytest.approx(put, rel=1e-10)
    assert g["gross"][0] == pytest.approx(call + put, rel=1e-10)
    assert g["gross"][0] > 0
    # concentration: +/-0.5% of 22013 is 21902.9 to 22123.1, so all three strikes count
    assert g["conc"][0] == pytest.approx(1.0)
    # a strike 1% away is outside the band
    g2 = ig.gamma_at_spot([S], [T], np.append(K, 22300.0), np.append(oc, 1e6), np.append(op, 0.0),
                          np.append(ivc, 0.12), np.append(ivp, 0.12))
    far = 1e6 * _hand_spot_gamma(S, 22300.0, T, 0.12) * S**2 * 0.01
    assert g2["conc"][0] == pytest.approx((g2["gross"][0] - far) / g2["gross"][0], rel=1e-9)


def test_gamma_at_spot_vectorised_over_anchors():
    K = np.arange(21500.0, 22550.0, 50.0)
    oc = np.full(K.size, 1e6)
    S = np.array([21990.0, 22010.0, 22040.0])
    T = np.array([300.0, 120.0, 30.0]) / YEAR_MIN
    g = ig.gamma_at_spot(S, T, K, oc, oc, np.full(K.size, 0.12), np.full(K.size, 0.12))
    for j in range(3):
        one = ig.gamma_at_spot([S[j]], [T[j]], K, oc, oc, np.full(K.size, 0.12), np.full(K.size, 0.12))
        assert g["gross"][j] == pytest.approx(one["gross"][0], rel=1e-12)


def test_net_gex_sign_rules():
    call, put = np.array([10.0, 4.0]), np.array([3.0, 8.0])
    assert np.allclose(ig.net_gex(call, put, 1.0, -1.0), [7.0, -4.0])     # US convention
    assert np.allclose(ig.net_gex(call, put, 0.1, 0.05), [1.15, 0.8])     # participant-style weights
    assert np.allclose(ig.net_gex(call, put, 1.0, 1.0), call + put)       # long_all = gross


# ---------------------------------------------------------------- placebos


def test_half_strike_shift():
    K = np.array([22000.0, 22050.0, 22100.0])
    assert np.allclose(ig.shift_profile(K), [22025.0, 22075.0, 22125.0])
    assert np.allclose(ig.shift_profile(K, -25.0), [21975.0, 22025.0, 22075.0])
    # a uniform OI profile on the whole 50-point grid is unchanged by a full-strike shift, but not by a half shift
    grid = np.arange(19000.0, 25000.0, 50.0)
    oi = np.full(grid.size, 1e6)
    iv = np.full(grid.size, 0.12)
    S, T = np.array([22000.0]), np.array([10.0 / YEAR_MIN])
    real = ig.gamma_at_spot(S, T, grid, oi, oi, iv, iv)["gross"][0]
    full = ig.gamma_at_spot(S, T, ig.shift_profile(grid, 50.0), oi, oi, iv, iv)["gross"][0]
    half = ig.gamma_at_spot(S, T, ig.shift_profile(grid, 25.0), oi, oi, iv, iv)["gross"][0]
    assert full == pytest.approx(real, rel=1e-6)
    assert half < real  # with 10 minutes left the index sits on a real strike but between two shifted ones
    # with a whole day left the kernel is much wider than a strike, so the half shift barely matters
    Tday = np.array([1.0 / 252])
    r1 = ig.gamma_at_spot(S, Tday, grid, oi, oi, iv, iv)["gross"][0]
    h1 = ig.gamma_at_spot(S, Tday, ig.shift_profile(grid, 25.0), oi, oi, iv, iv)["gross"][0]
    assert h1 == pytest.approx(r1, rel=1e-3)


def test_recentre_shift_is_whole_strikes():
    sh = ig.recentre_shift(22013.0, 22540.0)
    assert sh == 550.0 and sh % 50 == 0
    assert ig.recentre_shift(22540.0, 22013.0) == -550.0
    assert ig.recentre_shift(18000.0, 18000.0) == 0.0


def test_permute_strike_oi_keeps_pairs_and_outside_band():
    rng = np.random.default_rng(0)
    K = np.arange(20000.0, 24050.0, 50.0)
    oc = rng.uniform(0, 1e6, K.size)
    op = rng.uniform(0, 1e6, K.size)
    pc, pp = ig.permute_strike_oi(K, oc, op, centre=22000.0, band=0.05, rng=rng)
    inside = np.abs(K / 22000.0 - 1) <= 0.05
    assert np.allclose(pc[~inside], oc[~inside]) and np.allclose(pp[~inside], op[~inside])
    assert pc[inside].sum() == pytest.approx(oc[inside].sum())
    assert not np.allclose(pc[inside], oc[inside])
    # (call, put) pairs move together
    pairs = set(zip(np.round(oc[inside], 6), np.round(op[inside], 6)))
    assert set(zip(np.round(pc[inside], 6), np.round(pp[inside], 6))) == pairs


# ---------------------------------------------------------------- clock and time floor


def test_time_to_expiry_and_floor():
    # 15:00 on expiry day: 30 minutes to 15:30
    assert ig.time_to_expiry(345, 0) == pytest.approx(30 / YEAR_MIN)
    # the 5-minute floor binds only in the last 5 minutes
    assert ig.time_to_expiry(372, 0, floor_minutes=5) == pytest.approx(5 / YEAR_MIN)
    assert ig.time_to_expiry(374, 0, floor_minutes=5) == pytest.approx(5 / YEAR_MIN)
    assert ig.time_to_expiry(369, 0, floor_minutes=5) == pytest.approx(6 / YEAR_MIN)
    # a 60-minute floor binds from 14:30
    assert ig.time_to_expiry(345, 0, floor_minutes=60) == pytest.approx(60 / YEAR_MIN)
    assert ig.time_to_expiry(300, 0, floor_minutes=60) == pytest.approx(75 / YEAR_MIN)
    # the day before expiry adds a full session
    assert ig.time_to_expiry(345, 1) == pytest.approx((30 + 375) / YEAR_MIN)
    # 09:30 with three sessions left
    assert ig.time_to_expiry(15, 3) == pytest.approx((360 + 3 * 375) / YEAR_MIN)
    # 'Asian' clock that ends at 15:10
    assert ig.time_to_expiry(345, 0, close_slot=ig.ASIAN_CLOSE_SLOT) == pytest.approx(10 / YEAR_MIN)
    # T is never zero, so gamma stays finite
    g = ig.gamma_at_spot([22000.0], [ig.time_to_expiry(374, 0)], [22000.0], [1e6], [1e6], [0.12], [0.12])
    assert np.isfinite(g["gross"]).all()
    assert ig.slot_clock(15) == "09:30" and ig.slot_clock(345) == "15:00" and ig.slot_clock(0) == "09:15"


# ---------------------------------------------------------------- realized variance windows


def test_minute_returns_and_rv_windows():
    closes = np.array([[101.0, 102.0, 101.0, 103.0]])
    r = ig.minute_returns([100.0], closes)
    assert r[0, 0] == pytest.approx(math.log(101 / 100))
    assert r[0, 3] == pytest.approx(math.log(103 / 101))
    ret = np.arange(1.0, 11.0) / 100  # 10 minutes
    assert ig.forward_rv(ret, 4, 3) == pytest.approx(np.sum(ret[4:7] ** 2))
    assert np.isnan(ig.forward_rv(ret, 8, 3))                       # runs past the end of the session
    assert np.isnan(ig.forward_rv(ret, 4, 3, last_ok_slot=5))       # closing-auction cut-off
    assert ig.backward_rv(ret, 6, 3) == pytest.approx(np.sum(ret[3:6] ** 2))
    # fewer minutes than the window: scale the available minutes up
    assert ig.backward_rv(ret, 2, 6) == pytest.approx(np.sum(ret[:2] ** 2) * 3)


# ---------------------------------------------------------------- no look-ahead


def _synthetic_record(seed=1, n_days_left=0):
    rng = np.random.default_rng(seed)
    K = np.arange(21000.0, 23050.0, 50.0)
    prof = pd.DataFrame({"strike": K, "oi_call": rng.uniform(1e5, 5e6, K.size), "oi_put": rng.uniform(1e5, 5e6, K.size),
                         "iv_call": np.full(K.size, 0.12), "iv_put": np.full(K.size, 0.13)})
    open0 = 22000.0
    closes = open0 * np.exp(np.cumsum(rng.normal(0, 3e-4, 375)))
    ret = ig.minute_returns([open0], closes[None, :])[0]
    return {"date": pd.Timestamp("2025-01-02"), "day_type": "expiry", "regime": "single_weekly_thu",
            "full_sessions": n_days_left, "cas_day": False, "K_maxoi": 22000.0, "S_prev": 21990.0, "F_prev": 21995.0,
            "scale_gross_prev_close": 1e12, "_prof": prof, "_closes": closes, "_ret": ret,
            "_w": {"us_convention": (1.0, -1.0), "participant_pro": (0.05, 0.1), "client_mirror": (0.2, -0.1)}}


def test_spot_at_anchor_uses_last_completed_bar():
    closes = np.arange(375, dtype=float)
    assert ig.spot_at_anchor(closes, 15) == 14.0  # the 09:29 bar closes at 09:30
    assert np.allclose(ig.spot_at_anchor(closes, ig.ANCHORS), ig.ANCHORS - 1)
    with pytest.raises(ValueError):
        ig.spot_at_anchor(closes, 0)


def test_no_look_ahead_in_anchor_values():
    rec = _synthetic_record()
    base = ig.anchor_rows(rec)
    m_idx = 40                       # anchor 12:50 (slot 215)
    m = int(ig.ANCHORS[m_idx])
    future = dict(rec)
    closes = rec["_closes"].copy()
    closes[m:] *= 1.02               # change every bar from the anchor minute on
    future["_closes"] = closes
    future["_ret"] = ig.minute_returns([rec["_closes"][0] / np.exp(rec["_ret"][0])], closes[None, :])[0]
    alt = ig.anchor_rows(future)
    known = ["spot", "gross", "gross_call", "gross_put", "conc", "rv_back15", "rv_back60", "dist_near",
             "dist_maxoi_pct", "ret_prev_pct", "net_us_convention", "net_participant_pro", "net_client_mirror",
             "gross_floor60", "gross_floor120", "gross_asian", "gross_shift_plus", "gross_shift_minus"]
    for c in known:
        assert np.allclose(base.loc[: m_idx, c], alt.loc[: m_idx, c], rtol=0, atol=0), c
        assert not np.allclose(base.loc[m_idx + 1:, c], alt.loc[m_idx + 1:, c]) or c in ("dist_near",), c
    # the outcome (next 15 minutes) is what is allowed to change at the anchor itself
    assert base.loc[m_idx, "rv15"] != alt.loc[m_idx, "rv15"]
    assert np.allclose(base.loc[: m_idx - 3, "rv15"], alt.loc[: m_idx - 3, "rv15"])


def test_day_gamma_floor_and_placebo_profile():
    rec = _synthetic_record()
    g5 = ig.day_gamma(rec)["gross"]
    g120 = ig.day_gamma(rec, floor=120)["gross"]
    late = ig.ANCHORS > 375 - 120
    assert np.allclose(g5[~late], g120[~late])
    assert not np.allclose(g5[late], g120[late])
    K = rec["_prof"]["strike"].to_numpy()
    gs = ig.day_gamma(rec, K=ig.shift_profile(K))["gross"]
    assert not np.allclose(gs, g5)


# ---------------------------------------------------------------- fixed-effects regression


def test_fe_ols_recovers_slope_with_day_and_minute_effects():
    rng = np.random.default_rng(3)
    days, mins = 120, 20
    d = np.repeat(np.arange(days), mins)
    m = np.tile(np.arange(mins), days)
    a_d = rng.normal(0, 2, days)[d]
    b_m = np.sin(np.arange(mins) / 3)[m]
    x = 0.8 * a_d + b_m + rng.normal(0, 1, d.size)     # x correlated with the day effect
    z = rng.normal(0, 1, d.size)
    y = 0.5 * x - 0.2 * z + a_d + 2 * b_m + rng.normal(0, 0.5, d.size)
    df = pd.DataFrame({"day_code": d, "slot": m, "y": y, "x": x, "z": z})
    res = ig.fe_ols(df, "y", ["x", "z"])
    assert res["coef"]["x"]["coef"] == pytest.approx(0.5, abs=0.03)
    assert res["coef"]["z"]["coef"] == pytest.approx(-0.2, abs=0.03)
    c = res["coef"]["x"]
    assert c["ci_low"] < 0.5 < c["ci_high"]
    assert c["se"] > 0 and c["se_twoway"] > 0 and c["se_dk"] > 0
    assert res["n_days"] == days and res["n_obs"] == days * mins
    # the fast path used by the placebo draws gives the same coefficient
    g1 = pd.factorize(df["day_code"], sort=True)[0]
    g2 = pd.factorize(df["slot"], sort=True)[0]
    Z = ig.demean_two_way(df[["y", "x", "z"]].to_numpy(float), g1, g2)
    b, t = ig.fast_coef(Z[:, 0], Z[:, 2:], Z[:, 1], g1, days)
    assert b == pytest.approx(c["coef"], rel=1e-8)


def test_demean_two_way_unbalanced():
    rng = np.random.default_rng(5)
    g1 = rng.integers(0, 15, 400)
    g2 = rng.integers(0, 7, 400)
    X = rng.normal(size=(400, 2)) + g1[:, None] * 0.3 + g2[:, None]
    Z = ig.demean_two_way(X, g1, g2)
    for g in (g1, g2):
        means = np.array([Z[g == k].mean(axis=0) for k in np.unique(g)])
        assert np.abs(means).max() < 1e-8


def test_holm():
    p = ig.holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(p, [0.04, 0.09, 0.09, 0.5])


# ---------------------------------------------------------------- follow-up 1: final half hour


def test_discrete_average_variance_formula():
    # A = mean of the next n values of a unit random walk: Var(A) = sum_ij min(i, j) / n^2 = (n + 1)(2n + 1) / (6n)
    for n in (1, 6, 13, 30):
        i = np.arange(1, n + 1)
        exact = np.minimum.outer(i, i).sum() / n**2
        assert exact == pytest.approx((n + 1) * (2 * n + 1) / (6 * n), rel=1e-12)


def test_settlement_gamma_matches_hand_calculation():
    S, n, pm = 22013.0, 12, 22040.0
    K = np.array([21950.0, 22000.0, 22050.0, 22100.0])
    oc = np.array([1.0e6, 4.0e6, 2.5e6, 0.7e6])
    op = np.array([3.0e6, 2.0e6, 0.5e6, 0.1e6])
    ivc, ivp = np.full(4, 0.12), np.full(4, 0.13)
    w = n / 30
    t_min = (n + 1) * (2 * n + 1) / (6 * n)          # 4.51 minutes
    for floor, t_used in ((1, t_min), (5, 5.0)):
        g = ig.settlement_gamma_at_spot([S], n, pm, K, oc, op, ivc, ivp, floor=floor)
        T = t_used / YEAR_MIN
        Ks = (K - (1 - w) * pm) / w
        call = sum(oc[i] * w * _hand_spot_gamma(S, Ks[i], T, ivc[i]) * S**2 * 0.01 for i in range(4))
        put = sum(op[i] * w * _hand_spot_gamma(S, Ks[i], T, ivp[i]) * S**2 * 0.01 for i in range(4))
        assert g["T_eff_minutes"][0] == pytest.approx(t_used)
        assert g["w"][0] == pytest.approx(w)
        assert g["call"][0] == pytest.approx(call, rel=1e-10)
        assert g["put"][0] == pytest.approx(put, rel=1e-10)


def test_settlement_gamma_full_window_and_certain_strikes():
    K = np.arange(21500.0, 22550.0, 50.0)
    oi = np.full(K.size, 1e6)
    iv = np.full(K.size, 0.12)
    # at 15:00 nothing is printed yet (w = 1): an option on the average of 30 closes = an option with 10.5 minutes left
    g = ig.settlement_gamma_at_spot([22010.0], 30, np.nan, K, oi, oi, iv, iv)
    t30 = 31 * 61 / 180
    ref = ig.gamma_at_spot([22010.0], [t30 / YEAR_MIN], K, oi, oi, iv, iv)
    assert g["T_eff_minutes"][0] == pytest.approx(t30)
    assert g["gross"][0] == pytest.approx(ref["gross"][0], rel=1e-12)
    # a strike so low that the printed part of the average already exceeds it carries no gamma (K* <= 0)
    g2 = ig.settlement_gamma_at_spot([22010.0], 3, 22000.0, np.array([100.0]), [1e6], [1e6], [0.12], [0.12])
    assert g2["gross"][0] == 0.0


def _fh_record(seed=1, day=0, grid=50.0):
    rec = _synthetic_record(seed)
    rec["date"] = pd.Timestamp("2025-01-02") + pd.Timedelta(days=7 * day)
    rec["grid"] = grid
    return rec


def test_fh_rows_timing_and_no_look_ahead():
    rec = _fh_record()
    base = ig.fh_rows(rec)
    assert len(base) == 25 and base["clock"].iloc[0] == "15:00" and base["clock"].iloc[-1] == "15:24"
    assert list(base["minutes_to_1530"]) == list(range(30, 5, -1))
    # on the clock to 15:30 at least 6 minutes are left, so the 1, 2 and 5 minute floors give the same gamma
    assert np.allclose(base["gross"], base["gross_floor2"]) and np.allclose(base["gross"], base["gross_floor5"])
    # the printed part of the settlement average: nothing at 15:00, the 15:00 bar's close at 15:01
    cl = rec["_closes"]
    assert base["settle_past_mean"].iloc[0] == base["spot"].iloc[0]
    assert base["settle_past_mean"].iloc[1] == pytest.approx(cl[345])
    assert base["settle_past_mean"].iloc[10] == pytest.approx(cl[345:355].mean())
    # change every bar from the anchor minute 15:10 on: nothing known at 15:10 or before may move
    m_idx = 10
    m = int(ig.FH_ANCHORS[m_idx])
    alt_rec = dict(rec)
    closes = cl.copy()
    closes[m:] *= 1.003
    alt_rec["_closes"] = closes
    alt_rec["_ret"] = ig.minute_returns([cl[0] / np.exp(rec["_ret"][0])], closes[None, :])[0]
    alt = ig.fh_rows(alt_rec)
    known = ["spot", "gross", "conc", "conc025", "gross_floor2", "gross_floor5", "settle_past_mean",
             "gross_settle_floor1", "gross_settle_floor2", "gross_settle_floor5", "rv_back5", "rv_back15", "rv_back60",
             "dist_near", "dist_maxoi_pct", "ret_prev_pct", "gross_shift_plus", "gross_shift_minus"]
    for c in known:
        assert np.allclose(base.loc[:m_idx, c], alt.loc[:m_idx, c], rtol=0, atol=0), c
    assert not np.allclose(base.loc[m_idx + 1:, "spot"], alt.loc[m_idx + 1:, "spot"])
    assert base.loc[m_idx, "rv5"] != alt.loc[m_idx, "rv5"]
    assert np.allclose(base.loc[:m_idx - 5, "rv5"], alt.loc[:m_idx - 5, "rv5"])


def test_grid_aware_half_strike_shift_for_bank_nifty():
    rec = _fh_record(grid=100.0)
    rows = ig.anchor_rows(rec)
    K = rec["_prof"]["strike"].to_numpy()
    assert np.allclose(rows["gross_shift_plus"], ig.day_gamma(rec, K=K + 50.0)["gross"])
    assert np.allclose(rows["gross_shift_minus"], ig.day_gamma(rec, K=K - 50.0)["gross"])
    assert rows["dist_near"].max() <= 50.0
    assert ig.recentre_shift(45120.0, 45390.0, 100.0) == 300.0


def test_followup_placebo_draws_on_synthetic_days():
    recs = [_fh_record(seed=s, day=s) for s in range(1, 9)]
    fh = ig.build_fh_panel(recs)
    ig._PLACEBO_CTX["synthetic"] = ig.prepare_placebo_ctx(recs, fh, y="y5", controls=ig.FH_CONTROLS,
                                                          anchors=ig.FH_ANCHORS, floor=ig.FH_FLOOR, grid=50.0)
    for kind in ("other_day", "permuted"):
        out = ig._placebo_draw((kind, "synthetic", 11))
        assert [o[2] for o in out] == ["log_gross", "conc"]
        assert all(np.isfinite(o[4]) and np.isfinite(o[5]) for o in out)
    # the same seed gives the same draw
    assert ig._placebo_draw(("permuted", "synthetic", 11)) == ig._placebo_draw(("permuted", "synthetic", 11))
    del ig._PLACEBO_CTX["synthetic"]


def test_one_sided_p():
    assert ig.one_sided_negative(-0.4, 0.04) == pytest.approx(0.02)
    assert ig.one_sided_negative(0.4, 0.04) == pytest.approx(0.98)
