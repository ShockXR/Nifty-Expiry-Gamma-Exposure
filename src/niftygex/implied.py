"""Option-implied information from the chain published at a day's close.

Everything here is computed from ONE day's bhavcopy (settlement prices, open
interest and volume of every listed Nifty option) and is used only for
forecasts on the NEXT trading day. Nothing intraday enters.

For each listed expiry E the module computes, with Black-76 and the
put-call-parity forward F of that expiry:

* ``n`` trading days from the publish day d to E (d excluded, E included),
  and T = n / 252, so implied variances are in "per trading day" time, which
  is the clock on which intraday realized variance accrues.
* ATM implied vol: the average of the call and put implied vols at each
  strike (the straddle vol, which is insensitive to small forward errors),
  interpolated linearly in strike at K = F.
* Total implied variance to expiry w = iv_atm^2 * T. This is the market's
  expected variance of log F from the close of d to the close of E, i.e.
  what the ATM straddle prices (to first order straddle = F sigma sqrt(T)
  sqrt(2/pi)). On the day before an expiry (n = 1) it is the market's own
  forecast of the expiry day's close-to-close variance.
* A skew proxy: (iv at one implied standard deviation below F minus iv at
  one standard deviation above F) / iv_atm, from out-of-the-money options
  (puts below F, calls above), with a smile curvature proxy alongside.
* Put/call open-interest and volume ratios.

The daily panel then combines expiries: nearest ("near"), second nearest
("next"), a constant-maturity 21-trading-day point (about one month,
interpolated linearly in total variance), the implied forward variance per
day between the near and next expiries, and term-structure slopes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import greeks

PANEL_COLUMNS = [
    "n_near", "iv_near", "w_near", "iv_next", "n_next", "w_next", "iv_m21", "fwdvar_day",
    "straddle_var_near", "skew_near", "smile_near", "skew_m21", "pcr_oi_near", "pcr_vol_near",
    "pcr_oi_all", "fwd_near",
]


class TradingDayCounter:
    """Counts trading sessions in (d, E] from a list of known trading days.

    Beyond the last known day, weekdays are counted (holidays unknown there;
    this only affects far expiries, never the near one inside the sample).
    """

    def __init__(self, trading_days):
        self.days = np.asarray(pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_days).normalize()))).values,
                               dtype="datetime64[D]")
        self.last = self.days[-1]

    def __call__(self, d, e) -> int:
        d64 = np.datetime64(pd.Timestamp(d).normalize().date(), "D")
        e64 = np.datetime64(pd.Timestamp(e).normalize().date(), "D")
        if e64 <= self.last:
            return int(np.searchsorted(self.days, e64, "right") - np.searchsorted(self.days, d64, "right"))
        inside = int(len(self.days) - np.searchsorted(self.days, d64, "right"))
        return inside + int(np.busday_count(self.last + 1, e64 + 1))


def _forward(sub: pd.DataFrame, T: float, r: float) -> float:
    c = sub[sub["is_call"]].groupby("strike")["settle"].mean()
    p = sub[~sub["is_call"]].groupby("strike")["settle"].mean()
    both = pd.concat([c, p], axis=1, keys=["c", "p"]).dropna()
    both = both[(both["c"] > 0) & (both["p"] > 0)]
    if both.empty:
        return float("nan")
    k = (both["c"] - both["p"]).abs().idxmin()
    return float(k + np.exp(r * T) * (both.loc[k, "c"] - both.loc[k, "p"]))


def expiry_summary(sub: pd.DataFrame, n_td: int, r: float, spot_hint: float | None = None,
                   max_abs_logm: float = 0.2) -> dict:
    """Implied quantities for one expiry. ``sub`` has strike, is_call, settle, oi, volume (optional iv)."""
    T = max(int(n_td), 1) / 252.0
    out = {"n": int(n_td), "T": T}
    calls_oi = float(sub.loc[sub["is_call"], "oi"].sum())
    puts_oi = float(sub.loc[~sub["is_call"], "oi"].sum())
    out["call_oi"], out["put_oi"] = calls_oi, puts_oi
    has_vol = "volume" in sub and sub["volume"].notna().any()
    if has_vol:
        out["call_vol"] = float(sub.loc[sub["is_call"], "volume"].sum())
        out["put_vol"] = float(sub.loc[~sub["is_call"], "volume"].sum())
        out["volume"] = out["call_vol"] + out["put_vol"]
        # Untraded series carry NSE's theoretical settlement price (checked on real
        # files: a nearly flat vol of about 15 to 17%), which would corrupt every
        # implied quantity of an illiquid expiry. Only series that traded are used.
        # (A file with no volume information at all, such as the simulator's
        # export, keeps every series.)
        if (sub["volume"] > 0).any():
            sub = sub[sub["volume"] > 0]
    F = _forward(sub, T, r) if ("settle" in sub and len(sub)) else float("nan")
    if not np.isfinite(F) or (spot_hint is not None and abs(F / spot_hint - 1) > 0.05):
        F = float(spot_hint) if spot_hint is not None else float("nan")
    out["F"] = F
    if not np.isfinite(F) or sub.empty:
        return out
    s = sub[np.abs(np.log(sub["strike"] / F)) <= max_abs_logm].copy()
    if "iv" not in s or s["iv"].isna().all():
        s["iv"] = greeks.implied_vol(s["settle"].to_numpy(float), F, s["strike"].to_numpy(float), T, r,
                                     s["is_call"].to_numpy(bool))
    piv = s.pivot_table(index="strike", columns="is_call", values="iv", aggfunc="mean")
    piv = piv.reindex(columns=[False, True])
    piv.columns = ["put", "call"]
    K = piv.index.to_numpy(float)
    straddle_iv = piv.mean(axis=1, skipna=False)  # both legs valid
    ok = straddle_iv.notna().to_numpy()
    below, above = K[ok & (K <= F)], K[ok & (K >= F)]
    if len(below) and len(above):
        k0, k1 = below.max(), above.min()
        v0, v1 = float(straddle_iv.loc[k0]), float(straddle_iv.loc[k1])
        iv_atm = v0 if k1 == k0 else v0 + (v1 - v0) * (F - k0) / (k1 - k0)
    else:
        iv_atm = float("nan")
    out["iv_atm"] = iv_atm
    # straddle-implied variance (normal approximation), as a cross-check of iv_atm^2 T
    px = s.pivot_table(index="strike", columns="is_call", values="settle", aggfunc="mean").reindex(columns=[False, True])
    px.columns = ["put", "call"]
    strad = (px["put"] + px["call"]).dropna()
    if len(strad) >= 2 and strad.index.min() <= F <= strad.index.max():
        st = float(np.interp(F, strad.index.to_numpy(float), strad.to_numpy(float)))
        out["straddle_var"] = (np.pi / 2) * (st * np.exp(r * T) / F) ** 2
    # OTM smile in log-moneyness: puts below F, calls above
    otm = pd.concat([piv.loc[piv.index < F, "put"], piv.loc[piv.index >= F, "call"]]).dropna().sort_index()
    if np.isfinite(iv_atm) and len(otm) >= 4:
        k = np.log(otm.index.to_numpy(float) / F)
        sd = iv_atm * np.sqrt(T)
        if k.min() <= -sd and k.max() >= sd:
            ivp = float(np.interp(-sd, k, otm.to_numpy(float)))
            ivc = float(np.interp(sd, k, otm.to_numpy(float)))
            out["skew"] = (ivp - ivc) / iv_atm
            out["smile"] = (ivp + ivc - 2 * iv_atm) / iv_atm
    return out


def day_summary(table: pd.DataFrame, publish_date, counter: TradingDayCounter, r: float,
                spot_hint: float | None = None, max_n: int = 70, min_volume_share: float = 0.005) -> dict:
    """Combine all listed expiries of one publish day into the panel row."""
    d = pd.Timestamp(publish_date).normalize()
    tb = table[table["expiry"] > d]
    row: dict = {}
    if tb.empty:
        return row
    exps = sorted(pd.DatetimeIndex(tb["expiry"].unique()))
    summ = []
    for e in exps:
        n = counter(d, e)
        if n > max_n and len(summ) >= 2:
            break
        es = expiry_summary(tb[tb["expiry"] == e], n, r, spot_hint)
        es["expiry"] = e
        summ.append(es)
    near = summ[0]
    # An expiry enters the term structure only if it is reasonably liquid.
    near_vol = near.get("volume", np.nan)
    for x in summ:
        x["liquid"] = bool(np.isfinite(x.get("iv_atm", np.nan)) and
                           (not np.isfinite(near_vol) or x.get("volume", 0) >= min_volume_share * near_vol))
    row["fwd_near"] = near.get("F", np.nan)
    row["n_near"] = near["n"]
    row["iv_near"] = near.get("iv_atm", np.nan)
    row["w_near"] = row["iv_near"] ** 2 * near["T"]
    row["straddle_var_near"] = near.get("straddle_var", np.nan)
    row["skew_near"] = near.get("skew", np.nan)
    row["smile_near"] = near.get("smile", np.nan)
    row["pcr_oi_near"] = near["put_oi"] / near["call_oi"] if near["call_oi"] > 0 else np.nan
    if near.get("call_vol", 0) > 0:
        row["pcr_vol_near"] = near["put_vol"] / near["call_vol"]
    tot_c = float(tb.loc[tb["is_call"], "oi"].sum())
    row["pcr_oi_all"] = float(tb.loc[~tb["is_call"], "oi"].sum()) / tot_c if tot_c > 0 else np.nan
    valid = [x for x in summ if x["liquid"]]
    if len(summ) >= 2 and summ[1]["liquid"]:
        nx = summ[1]
        row["n_next"] = nx["n"]
        row["iv_next"] = nx.get("iv_atm", np.nan)
        row["w_next"] = row["iv_next"] ** 2 * nx["T"]
        if nx["n"] > near["n"]:
            fv = (row["w_next"] - row["w_near"]) / (nx["n"] - near["n"])
            row["fwdvar_day"] = fv if fv > 0 else np.nan
    if valid:
        ns = np.array([x["n"] for x in valid], float)
        ws = np.array([x["iv_atm"] ** 2 * x["T"] for x in valid])
        target = 21.0
        if ns.max() >= target and ns.min() <= target:
            w21 = float(np.interp(target, ns, ws))
        elif ns.max() < target:
            w21 = ws[-1] / ns[-1] * target
        else:
            w21 = ws[0] / ns[0] * target
        row["iv_m21"] = np.sqrt(max(w21, 0) * 252 / target)
        m = valid[int(np.argmin(np.abs(ns - target)))]
        row["skew_m21"] = m.get("skew", np.nan)
    return row


def build_implied_panel(market: dict, r: float | None = None) -> pd.DataFrame:
    """One row per publish day (index = publish date) of option-implied quantities.

    Uses ``market["option_tables"]`` (every listed expiry, real data) when
    present; otherwise falls back to the single published chain per day
    (``market["chains"]``, as in the simulator), which gives the near-expiry
    quantities only.
    """
    r = market.get("r", 0.065) if r is None else r
    days = pd.DatetimeIndex(market["days"])
    tables = market.get("option_tables")
    counter = TradingDayCounter(market.get("trading_days_all", days) if len(market.get("trading_days_all", [])) else days)
    rows = {}
    for t, d in enumerate(days):
        ch = market["chains"][t]
        spot_hint = float(ch.attrs.get("spot", np.nan)) if len(ch.attrs) else None
        if tables is not None and d in tables:
            rows[d] = day_summary(tables[d], d, counter, r, spot_hint)
        elif tables is None and len(ch):
            n = max(int(round(float(ch.attrs["T"]) * 252)), 1)
            sub = ch.copy()
            if "volume" not in sub:
                sub["volume"] = np.nan
            es = expiry_summary(sub, n, r, spot_hint)
            rows[d] = {"fwd_near": es.get("F", np.nan), "n_near": es["n"], "iv_near": es.get("iv_atm", np.nan),
                       "w_near": es.get("iv_atm", np.nan) ** 2 * es["T"], "straddle_var_near": es.get("straddle_var", np.nan),
                       "skew_near": es.get("skew", np.nan), "smile_near": es.get("smile", np.nan),
                       "pcr_oi_near": es["put_oi"] / es["call_oi"] if es["call_oi"] > 0 else np.nan,
                       "pcr_oi_all": es["put_oi"] / es["call_oi"] if es["call_oi"] > 0 else np.nan}
    panel = pd.DataFrame.from_dict(rows, orient="index").reindex(days)
    for c in PANEL_COLUMNS:
        if c not in panel:
            panel[c] = np.nan
    # A missing bhavcopy day keeps the last published values (they were public then).
    panel = panel[PANEL_COLUMNS].ffill()
    panel.index.name = "publish_date"
    return panel
