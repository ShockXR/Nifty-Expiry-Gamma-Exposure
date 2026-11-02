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
GEX_SETS = {
    "us": ["gex_us", "dist_flip_us"] + GEX_LEVEL_FEATURES,
    "participant": ["gex_participant", "gex_client", "dist_flip_participant"] + GEX_LEVEL_FEATURES,
    "oracle": ["gex_oracle"] + GEX_LEVEL_FEATURES,
}

GEX_FEATURES = [
    "gex_us", "gex_participant", "gex_client", "gross_gex_rel", "atm_gamma_share",
    "dist_flip_us", "dist_flip_participant", "dist_max_oi", "dist_max_pain",
    "dist_call_wall", "dist_put_wall", "oi_change_rel",
]


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
        c = ch[["strike", "is_call", "oi", "settle"]].copy()
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


def _safe_log(x, floor=1e-14):
    return np.log(np.maximum(x, floor))


def build_dataset(market: dict, panel: pd.DataFrame, horizon: int, step: int = 15,
                  anchors: list | None = None) -> pd.DataFrame:
    """One row per (day, anchor minute) with target and features known at the anchor.

    ``horizon`` minutes ahead; anchors every ``step`` minutes from the open,
    or an explicit list (for example [315] with horizon 60 for the last hour).
    Target ``y`` is the realized variance (sum of squared 1-minute returns)
    over minutes [i, i + horizon).
    """
    R = market["R"]
    T, M = R.shape
    C, O, V = market["close"], market["open"], market["volume"]
    lab = market["labels"]
    is_exp = lab["is_expiry"].to_numpy()
    tdte = lab["trading_days_to_expiry"].fillna(0).to_numpy()
    regime_code = pd.Categorical(lab["regime"], categories=["multi_weekly", "single_weekly_thu", "single_weekly_tue"]).codes
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

    p = panel.reindex(market["days"])
    gross_med = p["gross_gex"].rolling(20, min_periods=1).median().to_numpy()
    vix = np.asarray(market["vix"], float)

    rows = []
    for t in range(1, T):
        prev = t - 1
        rv_d1 = day_rv[prev]
        rv_w = day_rv[max(0, t - 5):t].mean()
        rv_m = day_rv[max(0, t - 22):t].mean()
        y_prevday = {i: cum[prev, min(i + horizon, M)] - cum[prev, i] for i in anchors}
        gp = p.iloc[prev]
        scale = gross_med[prev]
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
                "t": t, "date": market["days"][t], "minute": i, "horizon": j - i,
                "y": y,
                "is_expiry": bool(is_exp[t]), "tdte": tdte[t], "regime": regime_code[t],
                "minute_frac": i / M, "mins_left": (M - i) / M,
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
                # GEX block (from the close of t-1)
                "gex_us": gp.get("net_gex_us_convention", np.nan) / scale,
                "gex_participant": gp.get("net_gex_participant", np.nan) / scale,
                "gex_client": gp.get("net_gex_client_mirror", np.nan) / scale,
                "gex_oracle": gp.get("net_gex_oracle", np.nan) / scale,
                "gross_gex_rel": gp["gross_gex"] / scale,
                "atm_gamma_share": gp["gross_gex_atm1pct"] / gp["gross_gex"],
                "dist_flip_us": np.log(gp.get("flip_us_convention", np.nan) / spot_now),
                "dist_flip_participant": np.log(gp.get("flip_participant", np.nan) / spot_now),
                "dist_max_oi": abs(np.log(gp["max_oi_strike"] / spot_now)),
                "dist_max_pain": abs(np.log(gp["max_pain"] / spot_now)),
                "dist_call_wall": np.log(gp["call_wall"] / spot_now),
                "dist_put_wall": np.log(gp["put_wall"] / spot_now),
                "oi_change_rel": (gp["total_oi"] / p["total_oi"].iloc[prev - 1] - 1) if prev >= 1 else 0.0,
            })
    return pd.DataFrame(rows)


BASE_FEATURES = [
    "is_expiry", "tdte", "regime", "minute_frac", "mins_left",
    "log_rv_back_h", "log_rv_back_15", "log_rv_back_60", "log_rv_today_pm",
    "log_rv_d1", "log_rv_w", "log_rv_m", "log_rv_prevday_same", "jump_share_d1",
    "log_vix", "log_vol15_rel", "ofi15",
]
