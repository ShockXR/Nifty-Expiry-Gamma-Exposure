"""Tests for the option-implied block, its timing, and the new GEX measures."""

import numpy as np
import pandas as pd

from niftygex import greeks
from niftygex.data import build_market_from_nse
from niftygex.features import (BASE_FEATURES, GEX_SETS, IMPLIED_FEATURES, add_placebo, build_dataset,
                               build_gex_panel)
from niftygex.implied import TradingDayCounter, build_implied_panel, day_summary, expiry_summary
from niftygex.simulate import SimConfig, export_nse_layout, simulate_market


def _chain(F, T, r, vol_fn, strikes, traded=True):
    rows = []
    for k in strikes:
        for is_call in (True, False):
            v = vol_fn(np.log(k / F))
            rows.append({"strike": float(k), "is_call": is_call,
                         "settle": float(greeks.price(F, k, T, r, v, is_call)),
                         "oi": 1000.0, "volume": 10.0 if traded else 0.0})
    return pd.DataFrame(rows)


def test_expiry_summary_recovers_flat_vol_and_straddle_variance():
    F, n, r = 22000.0, 3, 0.065
    T = n / 252
    ch = _chain(F, T, r, lambda k: 0.15, np.arange(20500, 23501, 50))
    es = expiry_summary(ch, n, r, spot_hint=F)
    assert abs(es["F"] / F - 1) < 1e-6
    assert abs(es["iv_atm"] - 0.15) < 1e-4
    assert abs(es["straddle_var"] / (0.15**2 * T) - 1) < 0.01   # straddle-implied variance = sigma^2 T
    assert abs(es["skew"]) < 1e-4 and abs(es["smile"]) < 1e-4


def test_skew_sign_and_untraded_series_ignored():
    F, n, r = 22000.0, 5, 0.065
    T = n / 252
    smile = lambda k: 0.14 - 0.6 * k + 2.0 * k * k  # noqa: E731  (puts richer than calls)
    ch = _chain(F, T, r, smile, np.arange(20500, 23501, 50))
    es = expiry_summary(ch, n, r, spot_hint=F)
    assert es["skew"] > 0.05 and es["smile"] > 0
    # Untraded series with absurd (theoretical) settlement prices must not move anything.
    junk = _chain(F, T, r, lambda k: 0.40, np.arange(20525, 23500, 50), traded=False)
    es2 = expiry_summary(pd.concat([ch, junk], ignore_index=True), n, r, spot_hint=F)
    for key in ("F", "iv_atm", "skew", "smile"):
        assert np.isclose(es[key], es2[key], rtol=1e-9)


def test_day_summary_term_structure_and_forward_variance():
    d = pd.Timestamp("2025-01-06")
    days = pd.bdate_range("2025-01-01", "2025-03-31")
    counter = TradingDayCounter(days)
    r, F = 0.065, 23000.0
    parts = []
    for e, vol in ((pd.Timestamp("2025-01-09"), 0.12), (pd.Timestamp("2025-01-16"), 0.13),
                   (pd.Timestamp("2025-01-30"), 0.14), (pd.Timestamp("2025-02-27"), 0.15)):
        n = counter(d, e)
        c = _chain(F, n / 252, r, lambda k, v=vol: v, np.arange(21000, 25001, 50))
        c["expiry"] = e
        parts.append(c)
    row = day_summary(pd.concat(parts, ignore_index=True), d, counter, r, spot_hint=F)
    assert row["n_near"] == 3 and row["n_next"] == 8
    assert abs(row["iv_near"] - 0.12) < 1e-4 and abs(row["iv_next"] - 0.13) < 1e-4
    w1, w2 = 0.12**2 * 3 / 252, 0.13**2 * 8 / 252
    assert abs(row["fwdvar_day"] / ((w2 - w1) / 5) - 1) < 1e-3
    assert 0.14 < row["iv_m21"] < 0.15  # 21 trading days lies between the 18- and 38-day expiries


def test_trading_day_counter():
    days = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-10"])  # 9th is a holiday
    c = TradingDayCounter(days)
    assert c("2025-01-06", "2025-01-08") == 2
    assert c("2025-01-07", "2025-01-10") == 2
    assert c("2025-01-10", "2025-01-14") == 2  # beyond the data: weekdays


def _nse_like_market(tmp_path, n_days=14, seed=5):
    m = simulate_market(SimConfig(n_days=n_days, kappa=0.0, seed=seed))
    paths = export_nse_layout(m, tmp_path)
    return build_market_from_nse(paths["bars"], paths["bhavcopy"], paths["participant"], paths["vix"],
                                  oi_unit="keep_units")


def test_features_at_day_t_use_only_the_chain_published_at_t_minus_1(tmp_path):
    """Perturb everything published at the close of day k; rows for days <= k must not change."""
    mk = _nse_like_market(tmp_path)
    assert mk["option_tables"], "the loader must keep the full option tables"
    k = 6
    dk = mk["days"][k]

    def features(market):
        panel = build_gex_panel(market)
        imp = build_implied_panel(market)
        return build_dataset(market, panel, horizon=30, step=60, implied=imp)

    base = features(mk)
    pert = dict(mk)
    tabs = dict(mk["option_tables"])
    t2 = tabs[dk].copy()
    t2["settle"] *= 1.8          # implied vols jump
    t2["oi"] *= 3.0              # positioning jumps
    t2["volume"] *= 5.0
    tabs[dk] = t2
    pert["option_tables"] = tabs
    chains = list(mk["chains"])
    c2 = chains[k].copy()
    c2["settle"] *= 1.8
    c2["oi"] *= 3.0
    c2.attrs = dict(chains[k].attrs)
    chains[k] = c2
    pert["chains"] = chains
    after = features(pert)

    cols = [c for c in BASE_FEATURES + GEX_SETS["us"] + GEX_SETS["participant"] if c in base]
    early = base["t"] <= k
    pd.testing.assert_frame_equal(base.loc[early, cols].reset_index(drop=True),
                                  after.loc[early, cols].reset_index(drop=True))
    nxt = base["t"] == k + 1
    for c in ("log_impvar_day", "log_impvar_to_exp", "log_iv_m21", "gross_gex_rel"):
        assert not np.allclose(base.loc[nxt, c].to_numpy(), after.loc[nxt, c].to_numpy()), c
    # and the implied block of day k+2 again depends only on day k+1's chain
    nn = base["t"] == k + 2
    for c in IMPLIED_FEATURES:
        assert np.allclose(base.loc[nn, c].to_numpy(), after.loc[nn, c].to_numpy(), equal_nan=True), c


def test_placebo_columns_come_from_earlier_days():
    ds = pd.DataFrame({"t": np.repeat(np.arange(1, 41), 3), "minute": np.tile([0, 60, 120], 40)})
    ds["x"] = ds["t"] * 1000 + ds["minute"]
    out = add_placebo(ds.copy(), ["x"], seed=3)
    first = out["t"] == 1
    assert out.loc[first, "pl_x"].isna().all()           # no earlier day to draw from
    o = out[~first]
    src_day = o["pl_x"] // 1000
    assert np.all(src_day < o["t"])                       # always an earlier day, never the day itself
    assert np.all(o["pl_x"] % 1000 == o["minute"])        # same anchor minute
    assert o.groupby("t")["pl_x"].apply(lambda v: (v // 1000).nunique()).eq(1).all()  # one source day per day
    # the source of a day does not depend on how many days follow it
    out2 = add_placebo(ds[ds["t"] <= 25].copy(), ["x"], seed=3)
    assert np.allclose(out2["pl_x"].to_numpy(), out.loc[out["t"] <= 25, "pl_x"].to_numpy(), equal_nan=True)
