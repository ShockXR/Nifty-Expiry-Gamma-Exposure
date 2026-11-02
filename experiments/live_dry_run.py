"""Live dry run for one Nifty expiry day: freeze forecasts before the open, score them after the close.

This is a DEMONSTRATION that the pipeline runs on live data, not evidence:
one day is one observation, and nothing about the models can be judged from
it. The point is that every forecast is written down and committed to git
before the market opens (the commit time is the proof), using only files
that existed at the previous close, and then scored mechanically.

    python experiments/live_dry_run.py freeze --day 2026-11-03     # before 09:15 IST
    python experiments/live_dry_run.py score  --day 2026-11-03     # after NSE publishes the day's files

What is frozen (all from data up to the previous trading day's close):

1. Day-ahead volatility. Realized variance (sum of squared 1-minute log
   returns, 09:15 to 15:29, no overnight gap) for the whole day and for the
   last hour (returns ending 14:30 to 15:29), from HAR-style regressions on
   all past days with dummies for expiry day, expiry regime and NSE's closing
   auction (from 3 Aug 2026). 80% intervals from the regression residuals.
2. The option market's own forecast: the expiring at-the-money straddle at
   the previous settlement implies an expected absolute close-to-close move
   (straddle price, in index points) and an implied variance.
3. A frozen 14:30 rule for the last hour: coefficients fitted now, applied at
   14:30 on the day to that day's variance so far.
4. Dealer gamma at the previous close under the three sign rules, the gamma
   flip, the call and put walls, max pain and the max open-interest strike,
   with what R3 and R5 say each state implies.
5. A frozen end-of-day rule (R5): predicted last-30-minute return given the
   14:59 rest-of-day return, under the Pro-rule GEX state of the previous
   evening.
6. Pinning: the base rates from the pinning study for the current regime
   (distance of the official close to the nearest 50-point strike, share
   within 5 points), and the named strikes.

Writes results/live/<day>/forecast.json and forecast.md (freeze), and
score.json and score.md (score).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from niftygex import greeks
from niftygex.data import load_bhavcopy, parity_forward

RAW, RUN, OUT = Path("data/raw"), Path("results/pipeline_nse"), Path("results/live")
AUCTION_START = pd.Timestamp("2026-08-03")
TUE_START = pd.Timestamp("2025-09-01")
THU_SINGLE_START = pd.Timestamp("2024-11-20")
R_RATE = 0.065


# ---------------------------------------------------------------- minute data
def minute_rv(bars: pd.DataFrame) -> pd.DataFrame:
    """Per day: rv_day, rv_to_1430 (returns ending 09:16 to 14:29), rv_last_hour (ending 14:30 to 15:29), and prices."""
    b = bars.copy()
    b["day"] = b["timestamp"].dt.normalize()
    b["slot"] = ((b["timestamp"] - b["day"]) / pd.Timedelta(minutes=1)).astype(int) - 555
    b = b[(b["slot"] >= 0) & (b["slot"] <= 374)]
    piv = b.pivot_table(index="day", columns="slot", values="close", aggfunc="last").reindex(columns=range(375))
    r = np.log(piv).diff(axis=1)  # column k = return ending at slot k (k >= 1)
    out = pd.DataFrame(index=piv.index)
    out["rv_day"] = (r.iloc[:, 1:] ** 2).sum(axis=1, min_count=300)
    out["rv_to_1430"] = (r.loc[:, 1:314] ** 2).sum(axis=1, min_count=250)
    out["rv_last_hour"] = (r.loc[:, 315:374] ** 2).sum(axis=1, min_count=50)
    out["p_1459"], out["p_1529"] = piv[344], piv[374]
    out["n_bars"] = piv.notna().sum(axis=1)
    return out


def official_closes() -> pd.Series:
    idx = pd.read_csv(RAW / "index_daily.csv", parse_dates=["date"])
    return idx[idx["index"].str.upper() == "NIFTY 50"].set_index("date")["close"].sort_index()


def vix_series() -> pd.Series:
    v = pd.read_csv(RAW / "india_vix.csv")
    v["date"] = pd.to_datetime(v["date"])
    return v.set_index("date")["close"].sort_index()


# ---------------------------------------------------------------- day panel and models
def day_panel(day: pd.Timestamp) -> pd.DataFrame:
    """History up to the day before `day`, plus one row for `day` with only predetermined columns."""
    dm = pd.read_csv(RUN / "daily_market.csv", index_col=0, parse_dates=True)
    bars = pd.read_csv(RAW / "nifty_1min.csv", parse_dates=["timestamp"])
    bars = bars[(bars["timestamp"] >= "2022-12-01") & (bars["timestamp"] < day)]
    rv = minute_rv(bars)
    d = pd.DataFrame(index=dm.index[dm.index < day])
    d = d.join(rv).join(dm[["is_expiry", "regime"]])
    d.loc[day, "is_expiry"] = True  # the target day is a known expiry (checked against the bhavcopy in freeze)
    d = d.sort_index()
    vix = vix_series()
    d["lvix_prev"] = np.log(vix.reindex(d.index).shift(1))
    for c in ("rv_day", "rv_last_hour"):
        lc = np.log(d[c])
        d[f"{c}_l1"] = lc.shift(1)
        d[f"{c}_m5"] = lc.shift(1).rolling(5).mean()
        d[f"{c}_m22"] = lc.shift(1).rolling(22).mean()
    d["expiry"] = d["is_expiry"].astype(float)
    d["tue"] = (d.index >= TUE_START).astype(float)
    d["thu_single"] = ((d.index >= THU_SINGLE_START) & (d.index < TUE_START)).astype(float)
    d["auction"] = (d.index >= AUCTION_START).astype(float)
    d["expiry_x_tue"] = d["expiry"] * d["tue"]
    d["expiry_x_thu_single"] = d["expiry"] * d["thu_single"]
    return d


def har_forecast(d: pd.DataFrame, day: pd.Timestamp, target: str) -> dict:
    cols = [f"{target}_l1", f"{target}_m5", f"{target}_m22", "lvix_prev", "expiry", "tue", "thu_single", "auction",
            "expiry_x_tue", "expiry_x_thu_single"]
    hist = d[d.index < day].dropna(subset=cols + [target])
    y = np.log(hist[target])
    f = sm.OLS(y, sm.add_constant(hist[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    x = sm.add_constant(d.loc[[day], cols], has_constant="add")
    mu = float(f.predict(x).iloc[0])
    res = f.resid.to_numpy()
    q10, q90 = np.quantile(res, [0.1, 0.9])
    smear = float(np.mean(np.exp(res)))  # Duan smearing: E[exp(error)], so the level forecast is unbiased
    return {"target": target, "n_train": int(f.nobs), "r2": float(f.rsquared),
            "log_forecast": mu, "forecast": float(np.exp(mu) * smear), "median_forecast": float(np.exp(mu)),
            "p10": float(np.exp(mu + q10)), "p90": float(np.exp(mu + q90)),
            "coefficients": {k: float(v) for k, v in f.params.items()}}


def rule_1430(d: pd.DataFrame, day: pd.Timestamp) -> dict:
    """Frozen coefficients: log last-hour RV on log RV up to 14:30 the same day plus predetermined terms."""
    cols = ["log_rv_to_1430", "rv_last_hour_l1", "lvix_prev", "expiry", "tue", "auction", "expiry_x_tue"]
    h = d[d.index < day].copy()
    h["log_rv_to_1430"] = np.log(h["rv_to_1430"])
    h = h.dropna(subset=cols + ["rv_last_hour"])
    f = sm.OLS(np.log(h["rv_last_hour"]), sm.add_constant(h[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    res = f.resid.to_numpy()
    pre = d.loc[day, ["rv_last_hour_l1", "lvix_prev", "expiry", "tue", "auction", "expiry_x_tue"]]
    return {"coefficients": {k: float(v) for k, v in f.params.items()}, "n_train": int(f.nobs), "r2": float(f.rsquared),
            "smearing": float(np.mean(np.exp(res))), "resid_q10": float(np.quantile(res, 0.1)),
            "resid_q90": float(np.quantile(res, 0.9)), "predetermined_inputs": {k: float(v) for k, v in pre.items()}}


def apply_1430(rule: dict, rv_to_1430: float) -> dict:
    c = rule["coefficients"]
    x = {"const": 1.0, "log_rv_to_1430": np.log(rv_to_1430), **rule["predetermined_inputs"]}
    mu = sum(c[k] * x[k] for k in c)
    return {"forecast": float(np.exp(mu) * rule["smearing"]), "p10": float(np.exp(mu + rule["resid_q10"])),
            "p90": float(np.exp(mu + rule["resid_q90"]))}


def r5_rule(day: pd.Timestamp, pro_gex_negative: bool) -> dict:
    """Frozen end-of-day rule from the R5 regression (binary Pro-rule spec), fitted on days before `day`."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import momentum_gex as mg
    dm = pd.read_csv(RUN / "daily_market.csv", index_col=0, parse_dates=True)
    dm = dm[dm.index < day]
    panel = pd.read_csv(RUN / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    d = dm.join(mg.daily_returns(RAW / "nifty_1min.csv", dm.index))
    scale = panel["gross_gex"].rolling(20, min_periods=5).median()
    d["g"] = (panel["net_gex_participant"] / scale).shift(1).reindex(d.index)
    d = d.dropna(subset=["r_rod", "r_last", "g"])
    neg = (d["g"] < 0).astype(float)
    X = pd.DataFrame({"r_rod": d["r_rod"] * 1e4, "neg": neg, "r_rod_x_neg": d["r_rod"] * 1e4 * neg})
    f = mg.fit(d["r_last"] * 1e4, X)
    p = f.params
    slope = float(p["r_rod"] + (p["r_rod_x_neg"] if pro_gex_negative else 0.0))
    icpt = float(p["const"] + (p["neg"] if pro_gex_negative else 0.0))
    return {"state": "Pro GEX < 0" if pro_gex_negative else "Pro GEX >= 0", "intercept_bp": icpt, "slope": slope,
            "n_train": int(f.nobs), "rule": "predicted last-30-minute return (bp) = intercept + slope x (rest-of-day return in bp)"}


# ---------------------------------------------------------------- option market
def straddle_view(prev: pd.Timestamp, day: pd.Timestamp) -> dict:
    ch = load_bhavcopy(RAW / "bhavcopy" / f"fo_{prev:%Y%m%d}.csv.gz", "NIFTY")
    sel = ch[ch["expiry"] == day]
    if sel.empty:
        raise SystemExit(f"No NIFTY contract expiring {day.date()} in the {prev.date()} bhavcopy; is it an expiry day?")
    T = 1 / 252
    F = parity_forward(sel[["strike", "is_call", "oi", "settle"]], T, R_RATE)
    piv = sel.pivot_table(index="strike", columns="is_call", values="settle")
    k = float((piv.index.to_series() - F).abs().idxmin())
    straddle = float(piv.loc[k, True] + piv.loc[k, False])
    iv = greeks.implied_vol(np.array([piv.loc[k, True]]), F, np.array([k]), T, R_RATE, np.array([True]))
    var_cc = (straddle / F) ** 2 * np.pi / 2  # straddle = F sigma sqrt(T) sqrt(2/pi) to first order
    return {"forward": F, "atm_strike": k, "straddle_points": straddle, "expected_abs_move_points": straddle,
            "implied_close_to_close_variance": float(var_cc), "atm_call_iv_annualised": float(np.asarray(iv).ravel()[0]),
            "note": "Call plus put at the at-the-money strike, at the previous settlement, for the contract expiring on the day. It prices the expected absolute distance between the day's official close (the settlement price) and that strike. Since 3 Aug 2026 the index's official close comes from a closing auction while F&O settlement prices come from the last half hour of F&O trading, so the previous official index close can differ from the forward (our reading of the two procedures)."}


def gex_state(prev: pd.Timestamp) -> dict:
    p = pd.read_csv(RUN / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    row = p.loc[prev]
    out = {k: float(row[k]) / 1e7 for k in ("gross_gex", "net_gex_us_convention", "net_gex_participant", "net_gex_client_mirror")}
    out = {f"{k}_crore_per_1pct": v for k, v in out.items()}
    for k in ("call_wall", "put_wall", "max_oi_strike", "max_pain", "flip_us_convention", "flip_participant", "spot"):
        out[k] = None if pd.isna(row[k]) else float(row[k])
    out["implications"] = {
        "R3 (in-sample, expiry days)": "Higher GEX the evening before went with a MORE volatile expiry day under the Pro and client-mirror rules; all three rules are positive tonight.",
        "R5 (in-sample)": "End-of-day reversal was concentrated after negative Pro GEX; tonight Pro GEX is "
                          + ("negative, so the reversal state applies." if row["net_gex_participant"] < 0 else "positive, so little follow-through either way is expected."),
        "Pinning study": "No dealer-gamma proxy explained pinning; in the Tuesday regime there was no pinning on average.",
    }
    return out


def pinning_base_rates() -> dict:
    t = pd.read_csv(Path("results/pinning/t1_clustering.csv"))
    r = t[(t["symbol"] == "NIFTY") & (t["sample"] == "single_weekly_tue")].iloc[0]
    return {"regime": "single_weekly_tue", "expiries": int(r["n_expiry"]),
            "mean_distance_to_nearest_50pt_strike_expiry": float(r["mean_dist_expiry_pts"]),
            "mean_distance_other_days": float(r["mean_dist_nonexpiry_pts"]),
            "share_within_5_points_expiry": float(r["share_within_tenth_expiry"]),
            "share_within_5_points_other_days": float(r["share_within_tenth_nonexpiry"]),
            "no_clustering_mean": 12.5, "no_clustering_share_within_5": 0.2,
            "forecast": "No pinning: distance to the nearest 50-point strike drawn from the same distribution as other days (mean about 12.5 points, 20% chance of being within 5 points)."}


# ---------------------------------------------------------------- freeze and score
def freeze(day: pd.Timestamp) -> None:
    dm = pd.read_csv(RUN / "daily_market.csv", index_col=0, parse_dates=True)
    prev = dm.index[dm.index < day].max()
    out = OUT / f"{day:%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    d = day_panel(day)
    gs = gex_state(prev)
    fc = {
        "day": str(day.date()), "information_set": f"files published up to the close of {prev.date()}",
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
        "label": "REAL DATA, live demonstration (n = 1, not evidence)",
        "previous_official_close": float(official_closes().loc[prev]),
        "previous_vix": float(vix_series().loc[prev]),
        "day_ahead": {t: har_forecast(d, day, t) for t in ("rv_day", "rv_last_hour")},
        "rule_1430": rule_1430(d, day),
        "option_market": straddle_view(prev, day),
        "gex_previous_close": gs,
        "rule_r5": r5_rule(day, gs["net_gex_participant_crore_per_1pct"] < 0),
        "pinning": pinning_base_rates(),
    }
    (out / "forecast.json").write_text(json.dumps(fc, indent=2))
    (out / "forecast.md").write_text(render_forecast(fc))
    print((out / "forecast.md").read_text())


def ann_vol(rv: float) -> float:
    return float(np.sqrt(rv * 252) * 100)


def render_forecast(fc: dict) -> str:
    da, om, g, pin, r5 = fc["day_ahead"], fc["option_market"], fc["gex_previous_close"], fc["pinning"], fc["rule_r5"]
    L = [f"# Live dry run: Nifty expiry on {fc['day']} (frozen forecasts)", "",
         f"**{fc['label']}.** Information set: {fc['information_set']}. Frozen at {fc['frozen_at_utc']} UTC; the git commit that adds this file is the timestamp that matters.", "",
         "## 1. Volatility, day ahead (HAR-style regressions on all past days)", "",
         "| Target | Forecast (variance) | As annualised vol | 80% interval (annualised vol) | Training days |", "| --- | --- | --- | --- | --- |"]
    for k, name in (("rv_day", "Whole day, 09:15 to 15:29"), ("rv_last_hour", "Last hour, 14:30 to 15:29")):
        x = da[k]
        scale = 1 if k == "rv_day" else 375 / 60
        L.append(f"| {name} | {x['forecast']:.3e} | {ann_vol(x['forecast'] * scale):.1f}% | {ann_vol(x['p10'] * scale):.1f}% to {ann_vol(x['p90'] * scale):.1f}% | {x['n_train']} |")
    L += ["", "(The last-hour row is scaled to a full-day rate so the two are comparable.)", "",
          "## 2. What the option market expects", "",
          f"Forward from put-call parity {om['forward']:,.1f}; at-the-money strike {om['atm_strike']:,.0f}; the expiring straddle costs {om['straddle_points']:.1f} points, so the market expects the official close to land on average about **{om['expected_abs_move_points']:.0f} points** away from {om['atm_strike']:,.0f}. The previous official close was {fc['previous_official_close']:,.2f}: since 3 August 2026 the index close comes from a closing auction, while option settlement prices come from F&O trading, which is the likely reason the forward and the official close differ. Implied variance to the close {om['implied_close_to_close_variance']:.3e} (includes the overnight gap, so it is not directly comparable with section 1).", "",
          "## 3. Dealer gamma at the previous close", "",
          "| Measure | Value |", "| --- | --- |",
          f"| Gross GEX (crore per 1% move) | {g['gross_gex_crore_per_1pct']:,.0f} |",
          f"| Net GEX, US convention | {g['net_gex_us_convention_crore_per_1pct']:,.0f} |",
          f"| Net GEX, Pro participant | {g['net_gex_participant_crore_per_1pct']:,.0f} |",
          f"| Net GEX, Client mirror | {g['net_gex_client_mirror_crore_per_1pct']:,.0f} |",
          f"| Gamma flip (US convention) | {g['flip_us_convention']:,.0f} |" if g["flip_us_convention"] else "| Gamma flip (US convention) | none |",
          f"| Call wall / put wall | {g['call_wall']:,.0f} / {g['put_wall']:,.0f} |",
          f"| Max open-interest strike / max pain | {g['max_oi_strike']:,.0f} / {g['max_pain']:,.0f} |", ""]
    for k, v in g["implications"].items():
        L.append(f"* **{k}:** {v}")
    L += ["", "## 4. Frozen rules applied during the day", "",
          f"* **14:30 rule (last hour):** log last-hour variance = frozen coefficients x (log variance from 09:15 to 14:30 today, yesterday's last hour, yesterday's VIX, regime and expiry terms); {fc['rule_1430']['n_train']} training days, R squared {fc['rule_1430']['r2']:.2f}.",
          f"* **End-of-day rule (R5), state {r5['state']}:** predicted last-30-minute return = {r5['intercept_bp']:.2f} bp + {r5['slope']:.4f} x (return from the previous close to 14:59, in bp); {r5['n_train']} training days.", "",
          "## 5. Pinning", "",
          f"In the Tuesday regime ({pin['expiries']} expiries) the official close was on average {pin['mean_distance_to_nearest_50pt_strike_expiry']:.1f} points from the nearest 50-point strike on expiry days and {pin['mean_distance_other_days']:.1f} on other days (12.5 means no clustering). Forecast: {pin['forecast']}", ""]
    return "\n".join(L)


def score(day: pd.Timestamp) -> None:
    out = OUT / f"{day:%Y-%m-%d}"
    fc = json.loads((out / "forecast.json").read_text())
    bars = pd.read_csv(RAW / "nifty_1min.csv", parse_dates=["timestamp"])
    bars = bars[(bars["timestamp"] >= day - pd.Timedelta(days=10)) & (bars["timestamp"] < day + pd.Timedelta(days=1))]
    rv = minute_rv(bars)
    if day not in rv.index:
        raise SystemExit(f"No minute bars for {day.date()} yet; run scripts/fetch_minute_bars.py first.")
    x = rv.loc[day]
    off = official_closes()
    if day not in off.index:
        raise SystemExit(f"No official close for {day.date()} yet; run scripts/download_nse.py for that day first.")
    close, prev_close = float(off.loc[day]), fc["previous_official_close"]
    prev_bar = rv.loc[rv.index < day, "p_1529"].iloc[-1]

    def ql(y, f):
        return float(y / f - np.log(y / f) - 1)

    res = {"day": fc["day"], "label": "REAL DATA, live demonstration (n = 1, not evidence)", "minute_bars": int(x["n_bars"])}
    for k in ("rv_day", "rv_last_hour"):
        f = fc["day_ahead"][k]
        res[k] = {"realized": float(x[k]), "forecast": f["forecast"], "p10": f["p10"], "p90": f["p90"],
                  "inside_80pct_interval": bool(f["p10"] <= x[k] <= f["p90"]), "qlike": ql(x[k], f["forecast"])}
    a = apply_1430(fc["rule_1430"], float(x["rv_to_1430"]))
    res["rule_1430"] = {**a, "rv_to_1430": float(x["rv_to_1430"]), "realized_last_hour": float(x["rv_last_hour"]),
                        "inside_80pct_interval": bool(a["p10"] <= x["rv_last_hour"] <= a["p90"]), "qlike": ql(x["rv_last_hour"], a["forecast"])}
    move = abs(close - fc["option_market"]["atm_strike"])
    res["option_market"] = {"expected_abs_move_points": fc["option_market"]["expected_abs_move_points"], "actual_abs_move_points": move,
                            "measured_from": "the at-the-money strike of the frozen straddle",
                            "realized_close_to_close_variance": float(np.log(close / prev_close) ** 2)}
    r_rod = 1e4 * np.log(x["p_1459"] / prev_bar)
    r_last = 1e4 * np.log(x["p_1529"] / x["p_1459"])
    r5 = fc["rule_r5"]
    pred = r5["intercept_bp"] + r5["slope"] * r_rod
    res["rule_r5"] = {"rest_of_day_bp": float(r_rod), "predicted_last30_bp": float(pred), "actual_last30_bp": float(r_last),
                      "same_sign": bool(np.sign(pred) == np.sign(r_last))}
    k_near = 50 * round(close / 50)
    g = fc["gex_previous_close"]
    res["pinning"] = {"official_close": close, "nearest_50pt_strike": k_near, "distance_points": abs(close - k_near),
                      "within_5_points": bool(abs(close - k_near) <= 5),
                      "distance_to_max_oi_strike": abs(close - g["max_oi_strike"]), "distance_to_max_pain": abs(close - g["max_pain"])}
    (out / "score.json").write_text(json.dumps(res, indent=2))
    L = [f"# Live dry run: Nifty expiry on {res['day']} (scored)", "", f"**{res['label']}.** Forecasts were frozen in `forecast.json` before the open.", "",
         "| Item | Forecast | Outcome | Note |", "| --- | --- | --- | --- |"]
    for k, name in (("rv_day", "Whole-day variance"), ("rv_last_hour", "Last-hour variance")):
        r = res[k]
        L.append(f"| {name} | {r['forecast']:.3e} (80%: {r['p10']:.2e} to {r['p90']:.2e}) | {r['realized']:.3e} | {'inside' if r['inside_80pct_interval'] else 'outside'} the interval; QLIKE {r['qlike']:.3f} |")
    r = res["rule_1430"]
    L.append(f"| Last hour, 14:30 rule | {r['forecast']:.3e} (80%: {r['p10']:.2e} to {r['p90']:.2e}) | {r['realized_last_hour']:.3e} | {'inside' if r['inside_80pct_interval'] else 'outside'}; QLIKE {r['qlike']:.3f} |")
    o = res["option_market"]
    L.append(f"| Distance of the official close from the straddle strike {fc['option_market']['atm_strike']:,.0f} | {o['expected_abs_move_points']:.0f} points (straddle price) | {o['actual_abs_move_points']:.0f} points | |")
    r = res["rule_r5"]
    L.append(f"| Last 30 minutes (R5 rule) | {r['predicted_last30_bp']:.1f} bp | {r['actual_last30_bp']:.1f} bp | rest of day {r['rest_of_day_bp']:.1f} bp; same sign: {r['same_sign']} |")
    p = res["pinning"]
    L.append(f"| Distance of close to nearest 50-point strike | about 12.5 points on average | {p['distance_points']:.2f} points ({p['nearest_50pt_strike']:,.0f}) | within 5 points: {p['within_5_points']} |")
    L += ["", "One day cannot confirm or refute any of these models. The value of this exercise is that the whole chain (download, features, frozen models, scoring) ran on live data with the forecasts committed before the outcome existed."]
    (out / "score.md").write_text("\n".join(L) + "\n")
    print((out / "score.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "score"])
    ap.add_argument("--day", required=True)
    a = ap.parse_args()
    day = pd.Timestamp(a.day).normalize()
    freeze(day) if a.mode == "freeze" else score(day)


if __name__ == "__main__":
    main()
