"""Fast tests for experiments/dealer_sign.py on SYNTHETIC data where the true dealer weights are known.

The estimator should recover the weights when the hedging effect is strong,
and give uninformative estimates (wide sets, no significance, no
out-of-sample gain) when there is no effect. No real data are needed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex import gex, greeks

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "dealer_sign.py"
_spec = importlib.util.spec_from_file_location("dealer_sign", _PATH)
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def _estimate(df, family, ctrl=("base",)):
    d = ds.valid(df)
    return ds.estimate_direction([ds.gram_stats(*ds.design_arrays(d, eq, family, ctrl)) for eq in ("a", "b")])


def _wald_p(df, family, eq):
    d = ds.valid(df)
    return ds.hac_theta(*ds.design_arrays(d, eq, family, reduced=True))["wald_p"]


# ---------------------------------------------------------------- GEX is linear in the weights


def test_net_gex_is_linear_in_call_and_put_weights():
    """Any (w_call, w_put) GEX equals w_call * G_call + w_put * G_put, using the project's own strike_gex."""
    rng = np.random.default_rng(0)
    K = np.arange(21000, 23050, 50.0)
    chain = pd.DataFrame({"strike": np.r_[K, K], "is_call": np.r_[np.ones(K.size, bool), np.zeros(K.size, bool)],
                          "oi": rng.uniform(1e5, 5e6, 2 * K.size)})
    chain["iv"] = 0.12 + 0.3 * (chain["strike"] / 22000 - 1) ** 2
    spot, T = 22013.0, 3 / 252
    full = gex.strike_gex(chain, spot, 1.0, T, rule="long_all")
    G_call = full.loc[full["is_call"], "gross_gex"].sum()
    G_put = full.loc[~full["is_call"], "gross_gex"].sum()
    for wc, wp in ((1.0, -1.0), (0.03, -0.07), (-0.4, 0.9)):
        net = gex.strike_gex(chain, spot, 1.0, T, rule="participant",
                             participant=gex.ParticipantWeights(wc, wp))["gex"].sum()
        assert net == pytest.approx(wc * G_call + wp * G_put, rel=1e-10)
    us = gex.strike_gex(chain, spot, 1.0, T, rule="us_convention")["gex"].sum()
    assert us / (G_call + G_put) == pytest.approx(G_call / (G_call + G_put) - G_put / (G_call + G_put))
    assert np.all(greeks.spot_gamma(spot, K, T, 0.065, 0.12) > 0)


# ---------------------------------------------------------------- the LR profile


def test_lr_profile_is_scale_invariant_and_imposes_the_sign():
    rng = np.random.default_rng(1)
    n = 400
    Z = np.column_stack([np.ones(n), rng.standard_normal(n)])
    Bm = rng.standard_normal((n, 2))
    w_true = ds.unit(200.0)[0]
    y = Z @ [0.1, 0.5] - 0.4 * Bm @ w_true + rng.standard_normal(n)
    st = ds.gram_stats(y, Z, Bm)
    prof = ds.lr_profile(st)
    # only the direction matters: the angle grid is a grid of directions, so check the gain formula directly
    w = ds.unit(200.0)[0]
    for scale in (0.1, 1.0, 7.0):
        ws = scale * w
        gain = (ws @ st["b"]) ** 2 / (ws @ st["A"] @ ws)
        assert gain == pytest.approx((w @ st["b"]) ** 2 / (w @ st["A"] @ w))
    # the sign is identified: the opposite direction has the wrong sign and gets zero gain
    assert prof[200] > 10
    assert prof[20] == 0.0
    assert np.all(prof >= 0)
    assert abs(ds.circ_diff(ds.ANGLES[np.argmax(prof)], 200.0)) <= 8


def test_angle_sets_and_half_plane_midpoint():
    mask = np.zeros(360, bool)
    mask[350:] = True
    mask[:11] = True
    assert ds.angle_set_text(mask) == "350 to 10"
    assert ds.angle_set_text(np.ones(360, bool)) == "all angles"
    # a flat maximum over the half-plane w_call > w_put (226 to 44 degrees) has its midpoint at 315 (US convention)
    J = np.zeros(360)
    J[226:] = 3.0
    J[:45] = 3.0
    assert ds._best_angle_index(J) == 315
    J2 = np.zeros(360)
    J2[46:225] = 2.0
    assert ds._best_angle_index(J2) == 135


# ---------------------------------------------------------------- recovery when the effect is strong


@pytest.mark.parametrize("angle", [30.0, 150.0, 250.0])
def test_recovers_participant_mix_when_effect_is_strong(angle):
    df = ds.synthetic_frame(900, "participants", angle, snr_a=0.35, snr_b=0.35, seed=int(angle))
    est = _estimate(df, "participants")
    assert abs(ds.circ_diff(est["angle"], angle)) < 15
    assert est["set"][int(angle)]  # the truth is inside the 90% LR set
    assert not est["set"][int(angle + 180) % 360]  # the reverse sign is excluded
    assert est["set"].sum() < 90
    assert _wald_p(df, "participants", "a") < 1e-3 and _wald_p(df, "participants", "b") < 1e-3


def test_recovers_2d_call_put_shares_under_r3_scaling():
    """With the old scaling (net / 20-day median gross) the call and put sums are not collinear,
    so a full 2-D (w_call, w_put) direction is identified."""
    df = ds.synthetic_frame(900, "shares", 300.0, snr_a=0.35, snr_b=0.35, seed=5, scaling="r3")
    assert not ds.is_reduced(ds.valid(df), "shares")
    est = _estimate(df, "shares")
    assert abs(ds.circ_diff(est["angle"], 300.0)) < 20


@pytest.mark.parametrize("angle,side", [(290.0, 315.0), (120.0, 135.0), (10.0, 315.0)])
def test_scale_free_shares_reveal_only_the_side(angle, side):
    """s_call + s_put = 1, so only sign(w_call - w_put) is identified: the estimate is the middle of a half-plane."""
    df = ds.synthetic_frame(900, "shares", angle, snr_a=0.35, snr_b=0.35, seed=11)
    assert ds.is_reduced(ds.valid(df), "shares")
    est = _estimate(df, "shares")
    assert est["angle"] == side
    J = est["profile"]
    on_side = np.array([abs(ds.circ_diff(a, side)) < 80 for a in ds.ANGLES])
    assert np.ptp(J[on_side]) < 1e-6 * J.max()  # flat across the identified half-plane


# ---------------------------------------------------------------- uninformative when there is no effect


def _boot_width(df, family, seed):
    d = ds.valid(df)
    arrs = [ds.design_arrays(d, eq, family) for eq in ("a", "b")]
    est = ds.estimate_direction([ds.gram_stats(*a) for a in arrs])
    bt = ds.bootstrap_direction(arrs, 100, np.random.default_rng(seed))
    return est, ds.boot_bands(bt[:, 0], est["angle"])["angle_width"]


def test_no_effect_gives_uninformative_estimates():
    """No effect: wide bootstrap bands, no significant loading. (The profile-LR set is NOT a reliable guide here:
    with a zero effect the direction is not identified and that set can be spuriously narrow.)"""
    widths, small_p, lrs = [], 0, []
    for seed in range(6):
        df = ds.synthetic_frame(700, "participants", 60.0, snr_a=0.0, snr_b=0.0, seed=100 + seed)
        est, width = _boot_width(df, "participants", seed)
        widths.append(width)
        lrs.append(est["lr_max"])
        small_p += sum(_wald_p(df, "participants", eq) < 0.01 for eq in ("a", "b"))
    assert min(widths) > 120  # the 90% bootstrap band covers more than a third of the circle
    assert small_p <= 1
    strong, w_strong = _boot_width(ds.synthetic_frame(700, "participants", 60.0, 0.35, 0.35, seed=100), "participants", 0)
    assert w_strong < 40
    assert strong["lr_max"] > 3 * max(lrs)


def test_placebo_shuffle_keeps_rows_and_breaks_the_link():
    df = ds.valid(ds.synthetic_frame(500, "participants", 150.0, snr_a=0.4, snr_b=0.4, seed=3))
    dp = ds.permute_outcomes(df, np.random.default_rng(0))
    pd.testing.assert_frame_equal(dp[["x_pro", "x_cmirror", "s_call", "us"]], df[["x_pro", "x_cmirror", "s_call", "us"]])
    # each outcome row moves as a whole (HAR lags stay with their outcome)
    key = df[["y", "y_l1", "r_rod", "r_last"]].round(12).apply(tuple, axis=1)
    keyp = dp[["y", "y_l1", "r_rod", "r_last"]].round(12).apply(tuple, axis=1)
    assert set(key) == set(keyp) and not (key.to_numpy() == keyp.to_numpy()).all()
    real = ds.estimate_direction([ds.gram_stats(*ds.design_arrays(df, eq, "participants")) for eq in ("a", "b")])
    lr_pl = [ds.estimate_direction([ds.gram_stats(*ds.design_arrays(ds.permute_outcomes(df, np.random.default_rng(s)), eq,
                                                                     "participants")) for eq in ("a", "b")])["lr_max"]
             for s in range(20)]
    assert real["lr_max"] > max(lr_pl)
    assert np.median(lr_pl) < 6


# ---------------------------------------------------------------- state-space model (Kalman filter)


def test_kalman_tracks_slowly_drifting_weights():
    n = 900
    path = np.linspace(0.0, 90.0, n)
    df = ds.valid(ds.synthetic_frame(n, "participants", path, snr_a=0.5, snr_b=0.0, seed=7))
    r = ds.fit_tvp(df, "a", "participants", train_end=df.index[450])
    d = ds.tvp_direction(r["theta_f"], r["P_f"], np.random.default_rng(0), n_draw=200)
    truth = path[-len(df):]
    est_angle = np.degrees(np.arctan2(d["w2"], d["w1"])) % 360
    err = np.abs(ds.circ_diff(est_angle, truth))
    assert np.nanmedian(err[-200:]) < 30
    assert np.nanmedian(err[100:300]) < 35
    # the band covers the truth on most late days
    W = ds.unit(truth)
    inside = (d["w1_lo"] <= W[:, 0] + 1e-9) & (W[:, 0] <= d["w1_hi"] + 1e-9)
    assert inside.iloc[-200:].mean() > 0.6


def test_kalman_states_near_zero_without_effect():
    df = ds.valid(ds.synthetic_frame(700, "participants", 60.0, snr_a=0.0, snr_b=0.0, seed=21))
    r = ds.fit_tvp(df, "a", "participants", train_end=df.index[350])
    d = ds.tvp_direction(r["theta_f"], r["P_f"], np.random.default_rng(0), n_draw=50)
    wald = d["wald"].to_numpy()[~r["train"]]
    assert np.nanmean(wald > 4.605) < 0.25  # chi-square(2) 90% point


# ---------------------------------------------------------------- out-of-sample contest


def _oos(df):
    d = ds.valid(df)
    run = ds.oos_run(d, test_start=d.index[len(d) // 2], variants=("signed",))
    s, dm = ds.summarize_losses(ds.losses_from_run(d, run), {"all": lambda x: np.ones(len(x), bool)})
    return s.set_index(["eq", "model"]), dm, run


def test_oos_estimated_sign_beats_baseline_and_wrong_rules_when_effect_is_strong():
    s, dm, run = _oos(ds.synthetic_frame(900, "participants", 150.0, snr_a=0.4, snr_b=0.4, seed=3))
    for eq in ("a", "b"):
        assert s.loc[(eq, "est_participants"), "pct_vs_none"] < -4
        assert s.loc[(eq, "est_participants"), "p_better_than_none"] < 0.01
    # the Pro rule (0 degrees) is 150 degrees away from the truth: the estimated rule beats it
    row = dm[(dm.estimated == "est_participants") & (dm.fixed_rule == "pro")]
    assert (row["pct_est_vs_rule"] < 0).all() and (row["p_est_better"] < 0.05).all()
    path = run["path"][run["path"].family == "participants"]
    assert (np.abs(ds.circ_diff(path["angle_joint"], 150.0)) < 25).all()


def test_oos_no_gain_without_effect():
    s, dm, _ = _oos(ds.synthetic_frame(900, "participants", 150.0, snr_a=0.0, snr_b=0.0, seed=4))
    for eq in ("a", "b"):
        assert s.loc[(eq, "est_participants"), "pct_vs_none"] > -2
        assert s.loc[(eq, "est_participants"), "p_better_than_none"] > 0.05


def test_signed_fit_sets_a_wrong_sign_to_zero_and_holm():
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = 1.0 + 0.8 * X[:, 1] + rng.standard_normal(n) * 0.1  # positive slope: the wrong sign
    tr, te = np.arange(200), np.arange(200, n)
    p, _, c = ds.fit_predict(X, y, tr, te, signed_col=1)
    assert c == 0.0 and np.allclose(p, y[tr].mean())
    _, _, c_free = ds.fit_predict(X, y, tr, te)
    assert np.isnan(c_free)
    adj = ds.holm(np.array([0.01, 0.04, np.nan, 0.03]))
    assert adj[0] == pytest.approx(0.03) and adj[3] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[2])


def test_end_of_day_outcome_leaves_out_auction_days():
    df = ds.valid(ds.synthetic_frame(400, "participants", 30.0, seed=1))
    df.loc[df.index[-50:], "auction"] = 1.0
    y_b, Z_b, _ = ds.design_arrays(df, "b", "participants")
    y_a, _, _ = ds.design_arrays(df, "a", "participants")
    assert len(y_b) == len(df) - 50 and len(y_a) == len(df)
    y_r, _, _ = ds.design_arrays(df, "b", "participants", ("base", "auction_regime"))
    assert len(y_r) == len(df)
