"""Daily GEX panel and the anchor-level forecasting dataset.

A "market" is a dict with these keys (the simulator and the NSE loaders both
produce it):

    days      DatetimeIndex of trading days
    labels    DataFrame from expiry_calendar.label_days
    R, open, high, low, close, volume   (days x minutes) arrays
    chains    list, one option chain per day as published at that day's close
              (columns strike, is_call, oi, settle; attrs spot and T)
    participant  DataFrame per publish day: pro_call, pro_put, client_call, client_put
    vix       India VIX (annualised, decimal) per publish day
    lot_size, r

Timing rule: anything published at the close of day t-1 (bhavcopy, the
participant file, VIX) is used only for forecasts on day t or later.
Intraday features at anchor (t, i) use minutes 0 to i-1 of day t only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import gex, greeks, realized

# Feature sets for the ablation. "gex_oracle" exists only in simulations.
GEX_LEVEL_FEATURES = ["gross_gex_rel", "atm_gamma_share", "dist_max_oi", "dist_max_pain",
                      "dist_call_wall", "dist_put_wall", "oi_change_rel"]
# Version 1 sets, kept so the first real-data run can be reproduced exactly.
GEX_SETS_V1 = {
    "us": ["gex_us", "dist_flip_us"] + GEX_LEVEL_FEATURES,
    "participant": ["gex_participant", "gex_client", "dist_flip_participant"] + GEX_LEVEL_FEATURES,
    "oracle": ["gex_oracle"] + GEX_LEVEL_FEATURES,
}

# Version 2: a clean ladder. "gross" holds only UNSIGNED gamma and OI-structure
# features; every signed set is "gross" plus the signed measures of one rule,
# so (signed set vs gross) isolates the sign and (gross vs base) isolates size.
GROSS_FEATURES = GEX_LEVEL_FEATURES + ["gross_gex_nm_rel", "gross_gex_depth"]
GEX_SETS = {
    "gross": list(GROSS_FEATURES),
    "us": GROSS_FEATURES + ["gex_us", "gex_us_nm", "gex_us_depth", "dist_flip_us"],
    "participant": GROSS_FEATURES + ["gex_participant", "gex_participant_nm", "gex_participant_depth",
                                     "dist_flip_participant"],
    "client": GROSS_FEATURES + ["gex_client", "gex_client_nm", "gex_client_depth"],
    "oracle": GROSS_FEATURES + ["gex_oracle"],
}
SIGNED_GEX_FEATURES = sorted({f for k in ("us", "participant", "client") for f in GEX_SETS[k]} - set(GROSS_FEATURES))

GEX_FEATURES = [
    "gex_us", "gex_participant", "gex_client", "gross_gex_rel", "atm_gamma_share",
    "dist_flip_us", "dist_flip_participant", "dist_max_oi", "dist_max_pain",
    "dist_call_wall", "dist_put_wall", "oi_change_rel",
]
ALL_GEX_FEATURES = sorted(set(GEX_FEATURES) | set(GROSS_FEATURES) | set(SIGNED_GEX_FEATURES))

# The placebo set: the "us" set's columns taken from a randomly chosen OTHER day
# (same anchor minute). It has the same marginal distribution as real GEX but
# no information about the day being forecast, so any "gain" it shows is noise.
PLACEBO_SOURCE = "us"
GEX_SETS["placebo"] = [f"pl_{c}" for c in GEX_SETS[PLACEBO_SOURCE]]


def build_gex_panel(market: dict, n_grid: int = 41, flip_width: float = 0.04, carry: float = 0.0) -> pd.DataFrame:
    """Daily GEX panel from the published chains.

    Implied vols are backed out of settlement prices with the Black-76 solver
    (the same path real bhavcopy data takes). The forward is
    spot * exp(carry * T); use carry = 0 for the simulator (which prices off
    spot) and roughly the risk-free rate minus dividend yield for real data,
    or pass the Nifty futures price as the chain's ``spot`` attribute.
    """
    rows = []
    lot, r = market["lot_size"], market["r"]
    part = market.get("participant")
    for t, ch in enumerate(market["chains"]):
        spot, T = float(ch.attrs["spot"]), float(ch.attrs["T"])
        # sort by strike explicitly: the IV gap filling below interpolates along row order
        c = ch[["strike", "is_call", "oi", "settle"]].sort_values(["is_call", "strike"], kind="stable").copy()
        c["iv"] = greeks.implied_vol(c["settle"].to_numpy(), spot * np.exp(carry * T),
                                     c["strike"].to_numpy(), T, r, c["is_call"].to_numpy())
        # Deep options with no time value carry no IV information: use the
        # nearest valid IV at the same side (only their tiny gamma is affected).
        c["iv"] = c.groupby("is_call")["iv"].transform(lambda s: s.interpolate(limit_direction="both"))
        c = c[c["oi"] > 0]
        pw = cw = None
        if part is not None:
            pr = part.iloc[t]
            pw = gex.ParticipantWeights(pr["pro_call"], pr["pro_put"])
            cw = gex.ParticipantWeights(-pr["client_call"], -pr["client_put"])
        row = gex.daily_summary(c, spot, lot, T, r, carry, pw, cw, with_flip=False)
        # Near-the-money gamma: strikes within one implied standard deviation of
        # the forward (at least 0.5%), where hedging flows are concentrated.
        iv_atm = _atm_iv(c, spot)
        band = max(iv_atm * np.sqrt(T), 0.005) if np.isfinite(iv_atm) else 0.01
        nm = c[np.abs(np.log(c["strike"] / spot)) <= band]
        row["atm_iv_chain"] = iv_atm
        row["gross_gex_nm"] = float(gex.strike_gex(nm, spot, lot, T, r, carry, "long_all")["gross_gex"].sum())
        for rule, w in (("us_convention", None), ("participant", pw), ("client_mirror", cw)):
            if rule != "us_convention" and w is None:
                continue
            row[f"net_gex_{rule}_nm"] = float(gex.strike_gex(nm, spot, lot, T, r, carry, rule, w)["gex"].sum())
        for rule, w in (("us_convention", None), ("participant", pw)):
            if rule == "participant" and w is None:
                continue
            row[f"flip_{rule}"] = gex.gamma_flip(c, spot, lot, T, r, carry, rule, w, width=flip_width, n_grid=n_grid)
        if "dealer_next" in market:  # simulator only: true dealer positions
            dn = market["dealer_next"].iloc[t]
            ow = gex.ParticipantWeights(dn["call"], dn["put"])
            row["net_gex_oracle"] = float(gex.strike_gex(c, spot, lot, T, r, carry, "participant", ow)["gex"].sum())
        row["total_oi"] = float(c["oi"].sum())
        rows.append(row)
    panel = pd.DataFrame(rows, index=market["days"][: len(rows)])
    panel.index.name = "publish_date"
    return panel


def _atm_iv(c: pd.DataFrame, F: float) -> float:
    """Straddle (call and put average) implied vol interpolated at the forward."""
    piv = c.pivot_table(index="strike", columns="is_call", values="iv", aggfunc="mean").reindex(columns=[False, True])
    st = piv.mean(axis=1, skipna=False).dropna()
    if len(st) < 2 or not (st.index.min() <= F <= st.index.max()):
        return float("nan")
    return float(np.interp(F, st.index.to_numpy(float), st.to_numpy(float)))


def _safe_log(x, floor=1e-14):
    return np.log(np.maximum(x, floor))


def build_dataset(market: dict, panel: pd.DataFrame, horizon: int, step: int = 15,
                  anchors: list | None = None, implied: pd.DataFrame | None = None,
                  placebo_seed: int = 12345) -> pd.DataFrame:
    """One row per (day, anchor minute) with target and features known at the anchor.

    ``horizon`` minutes ahead; anchors every ``step`` minutes from the open,
    or an explicit list (for example [315] with horizon 60 for the last hour).
    Target ``y`` is the realized variance (sum of squared 1-minute returns)
    over minutes [i, i + horizon).

    ``implied`` is the option-implied panel (implied.build_implied_panel),
    indexed by publish date. Day t uses the row published at the close of
    day t-1 (``implied.iloc[t-1]``), exactly like the GEX panel.
    """
    R = market["R"]
    T, M = R.shape
    C, O, V = market["close"], market["open"], market["volume"]
    lab = market["labels"]
    days = pd.DatetimeIndex(market["days"])
    is_exp = lab["is_expiry"].to_numpy()
    is_mexp = lab["is_monthly_expiry"].to_numpy() if "is_monthly_expiry" in lab else np.zeros(T, bool)
    tdte = lab["trading_days_to_expiry"].fillna(0).to_numpy()
    weekday = days.weekday.to_numpy()
    regime_code = pd.Categorical(lab["regime"], categories=["multi_weekly", "single_weekly_thu", "single_weekly_tue"]).codes
    other_exp = set(pd.DatetimeIndex(market.get("other_index_expiries", [])).normalize())
    bn_exp = np.array([d in other_exp for d in days], bool)
    if anchors is None:
        anchors = list(range(0, M - horizon + 1, step))

    r2 = R**2
    day_rv = r2.sum(axis=1)
    js = realized.jump_share(R)
    cum = np.concatenate([np.zeros((T, 1)), np.cumsum(r2, axis=1)], axis=1)
    vol_cum = np.concatenate([np.zeros((T, 1)), np.cumsum(V, axis=1)], axis=1)
    dC = np.concatenate([np.zeros((T, 1)), np.diff(np.log(C), axis=1)], axis=1)
    sig_dc = np.sqrt(np.median(r2))
    bvc = realized.bulk_volume_imbalance(dC, V, sig_dc)
    bvc_cum = np.concatenate([np.zeros((T, 1)), np.cumsum(bvc, axis=1)], axis=1)
    avg_day_vol = pd.Series(V.sum(axis=1)).shift(1).rolling(20, min_periods=1).mean().to_numpy()

    p = panel.reindex(days)
    gross_med = p["gross_gex"].rolling(20, min_periods=1).median().to_numpy()
    vix = np.asarray(market["vix"], float)
    fut = market.get("futures")
    fut_value = (fut["fut_value"].reindex(days).to_numpy(float) if fut is not None and "fut_value" in fut
                 else np.full(T, np.nan))
    ip = implied.reindex(days) if implied is not None else pd.DataFrame(index=days)

    def ipv(col, t_):
        return float(ip[col].iloc[t_]) if col in ip else np.nan

    rows = []
    for t in range(1, T):
        prev = t - 1
        rv_d1 = day_rv[prev]
        rv_w = day_rv[max(0, t - 5):t].mean()
        rv_m = day_rv[max(0, t - 22):t].mean()
        y_prevday = {i: cum[prev, min(i + horizon, M)] - cum[prev, i] for i in anchors}
        gp = p.iloc[prev]
        scale = gross_med[prev]
        depth = fut_value[prev]
        # option-implied block, all from the chain published at the close of t-1
        iv_near, iv_m21 = ipv("iv_near", prev), ipv("iv_m21", prev)
        impvar_day = iv_near**2 / 252
        fwdvar = ipv("fwdvar_day", prev)
        imp = {
            "log_impvar_day": _safe_log(impvar_day) if np.isfinite(impvar_day) else np.nan,
            "log_impvar_to_exp": _safe_log(ipv("w_near", prev)) if np.isfinite(ipv("w_near", prev)) else np.nan,
            "log_iv_m21": np.log(iv_m21) if np.isfinite(iv_m21) and iv_m21 > 0 else np.nan,
            "log_fwdvar_day": np.log(fwdvar) if np.isfinite(fwdvar) and fwdvar > 0 else np.nan,
            "exp_premium": np.log(impvar_day / fwdvar) if np.isfinite(fwdvar) and fwdvar > 0 else np.nan,
            "term_slope": np.log(iv_near / iv_m21) if np.isfinite(iv_m21) and iv_m21 > 0 else np.nan,
            "skew_near": ipv("skew_near", prev), "smile_near": ipv("smile_near", prev),
            "skew_m21": ipv("skew_m21", prev),
            "log_pcr_oi_near": np.log(ipv("pcr_oi_near", prev)) if ipv("pcr_oi_near", prev) > 0 else np.nan,
            "log_pcr_vol_near": np.log(ipv("pcr_vol_near", prev)) if ipv("pcr_vol_near", prev) > 0 else np.nan,
            "log_pcr_oi_all": np.log(ipv("pcr_oi_all", prev)) if ipv("pcr_oi_all", prev) > 0 else np.nan,
            "vrp_w": (np.log(impvar_day) - _safe_log(rv_w)) if np.isfinite(impvar_day) and impvar_day > 0 else np.nan,
        }
        cal = {
            "weekday": weekday[t], "is_monthly_expiry": bool(is_mexp[t]), "day_before_expiry": bool(tdte[t] == 1),
            "bn_expiry_only": bool(bn_exp[t] and not is_exp[t]),
        }
        sd_day = np.sqrt(impvar_day) if np.isfinite(impvar_day) and impvar_day > 0 else np.nan
        for i in anchors:
            j = min(i + horizon, M)
            y = cum[t, j] - cum[t, i]

            # backward-looking windows, spilling into the previous day at the open
            def back(k):
                if i >= k:
                    return cum[t, i] - cum[t, i - k]
                return cum[t, i] + (cum[prev, M] - cum[prev, M - (k - i)])
            spot_now = C[t, i - 1] if i > 0 else O[t, 0]
            vol15 = (vol_cum[t, i] - vol_cum[t, max(0, i - 15)]) / max(1, min(i, 15))
            ofi15 = (bvc_cum[t, i] - bvc_cum[t, max(0, i - 15)]) / max(1.0, vol_cum[t, i] - vol_cum[t, max(0, i - 15)])
            rows.append({
                "t": t, "date": days[t], "minute": i, "horizon": j - i,
                "y": y,
                "is_expiry": bool(is_exp[t]), "tdte": tdte[t], "regime": regime_code[t],
                "minute_frac": i / M, "mins_left": (M - i) / M,
                "exp_x_minute": float(is_exp[t]) * i / M,
                "log_rv_back_h": _safe_log(back(horizon)),
                "log_rv_back_15": _safe_log(back(15)),
                "log_rv_back_60": _safe_log(back(60)),
                "log_rv_today_pm": _safe_log(cum[t, i] / i) if i > 0 else _safe_log(day_rv[prev] / M),
                "log_rv_d1": _safe_log(rv_d1), "log_rv_w": _safe_log(rv_w), "log_rv_m": _safe_log(rv_m),
                "log_rv_prevday_same": _safe_log(y_prevday[i]),
                "jump_share_d1": js[prev],
                "log_vix": np.log(vix[prev]),
                "log_vol15_rel": np.log(max(vol15, 1e-9) * M / max(avg_day_vol[t], 1e-9)) if i > 0 else 0.0,
                "ofi15": ofi15 if i > 0 else 0.0,
                # how far the index has moved since the open, in implied daily standard deviations
                "abs_ret_today_sd": abs(np.log(spot_now / O[t, 0])) / sd_day,
                **imp, **cal,
                # GEX block (from the close of t-1)
                "gex_us": gp.get("net_gex_us_convention", np.nan) / scale,
                "gex_participant": gp.get("net_gex_participant", np.nan) / scale,
                "gex_client": gp.get("net_gex_client_mirror", np.nan) / scale,
                "gex_oracle": gp.get("net_gex_oracle", np.nan) / scale,
                "gex_us_nm": gp.get("net_gex_us_convention_nm", np.nan) / scale,
                "gex_participant_nm": gp.get("net_gex_participant_nm", np.nan) / scale,
                "gex_client_nm": gp.get("net_gex_client_mirror_nm", np.nan) / scale,
                "gex_us_depth": gp.get("net_gex_us_convention", np.nan) / depth,
                "gex_participant_depth": gp.get("net_gex_participant", np.nan) / depth,
                "gex_client_depth": gp.get("net_gex_client_mirror", np.nan) / depth,
                "gross_gex_depth": gp["gross_gex"] / depth,
                "gross_gex_rel": gp["gross_gex"] / scale,
                "gross_gex_nm_rel": gp.get("gross_gex_nm", np.nan) / scale,
                "atm_gamma_share": gp["gross_gex_atm1pct"] / gp["gross_gex"],
                "dist_flip_us": np.log(gp.get("flip_us_convention", np.nan) / spot_now),
                "dist_flip_participant": np.log(gp.get("flip_participant", np.nan) / spot_now),
                "dist_max_oi": abs(np.log(gp["max_oi_strike"] / spot_now)),
                "dist_max_pain": abs(np.log(gp["max_pain"] / spot_now)),
                "dist_call_wall": np.log(gp["call_wall"] / spot_now),
                "dist_put_wall": np.log(gp["put_wall"] / spot_now),
                "oi_change_rel": (gp["total_oi"] / p["total_oi"].iloc[prev - 1] - 1) if prev >= 1 else 0.0,
            })
    ds = pd.DataFrame(rows)
    return add_placebo(ds, GEX_SETS[PLACEBO_SOURCE], seed=placebo_seed)


def add_placebo(ds: pd.DataFrame, cols: list, seed: int = 12345) -> pd.DataFrame:
    """Add ``pl_<col>`` columns: each day's values replaced by those of another day.

    Days are permuted as whole blocks (no day keeps its own values) and
    matched on the anchor minute, so the placebo keeps the within-day shape
    and the marginal distribution of the real features but carries no
    information about the day being forecast.
    """
    rng = np.random.default_rng(seed)
    ts = np.sort(ds["t"].unique())
    if len(ts) < 2:
        for c in cols:
            ds[f"pl_{c}"] = np.nan
        return ds
    perm = rng.permutation(ts)
    for k in np.flatnonzero(perm == ts):  # remove fixed points by swapping with a neighbour
        j = (k + 1) % len(ts)
        perm[[k, j]] = perm[[j, k]]
    src = dict(zip(ts, perm))
    keyed = ds.set_index(["t", "minute"])[cols]
    idx = pd.MultiIndex.from_arrays([ds["t"].map(src).to_numpy(), ds["minute"].to_numpy()])
    vals = keyed.reindex(idx).to_numpy()
    for k, c in enumerate(cols):
        ds[f"pl_{c}"] = vals[:, k]
    return ds


# Version 1 base set (the first real-data run).
BASE_FEATURES_V1 = [
    "is_expiry", "tdte", "regime", "minute_frac", "mins_left",
    "log_rv_back_h", "log_rv_back_15", "log_rv_back_60", "log_rv_today_pm",
    "log_rv_d1", "log_rv_w", "log_rv_m", "log_rv_prevday_same", "jump_share_d1",
    "log_vix", "log_vol15_rel", "ofi15",
]
# Option-implied information, all from the chain published at the previous close.
IMPLIED_FEATURES = [
    "log_impvar_day", "log_impvar_to_exp", "log_iv_m21", "log_fwdvar_day", "exp_premium", "term_slope",
    "skew_near", "smile_near", "skew_m21", "log_pcr_oi_near", "log_pcr_vol_near", "log_pcr_oi_all", "vrp_w",
    "abs_ret_today_sd",
]
CALENDAR_FEATURES = ["weekday", "is_monthly_expiry", "day_before_expiry", "bn_expiry_only", "exp_x_minute"]
# Version 2 base set: v1 plus option-implied information and calendar structure.
BASE_FEATURES = BASE_FEATURES_V1 + IMPLIED_FEATURES + CALENDAR_FEATURES
