"""A3. GEX engine stress test (SYNTHETIC chains). Seeds: 9_800_000 + case. Each case records what the niftygex engine returns.
Usage: python a3_gex_stress.py <out_raw.csv>   (class is assigned in a3_classify.py after review)"""
import sys, warnings
import numpy as np, pandas as pd
from niftygex import gex as G, greeks as GR
ROWS = []
def case(name, fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try: obs, exc = fn(), ""
        except BaseException as e: obs, exc = "", f"{type(e).__name__}: {str(e)[:100]}"
        wn = sorted({f"{x.category.__name__}: {str(x.message)[:50]}" for x in w})
    ROWS.append(dict(case=name, observed=str(obs)[:260], exception=exc, warnings=" | ".join(wn)[:160]))

S, LOT, T0, R = 24000.0, 75.0, 3 / 365, 0.065
def chain(seed=9_800_000, step=50, lo=0.9, hi=1.1):
    rng = np.random.default_rng(seed); K = np.arange(round(S * lo / step) * step, S * hi + 1, step)
    rows = []
    for k in K:
        iv = 0.14 + 0.5 * (k / S - 1) ** 2 + 0.02 * (k < S)
        for c in (True, False):
            rows.append(dict(strike=float(k), is_call=c, oi=float(rng.integers(100, 20000)) * (1 + 3 * np.exp(-((k - S) / 400) ** 2)), iv=iv))
    return pd.DataFrame(rows)
C = chain()
def net(c, spot=S, lot=LOT, T=T0, rule="us_convention", **kw): return float(G.strike_gex(c, spot, lot, T, R, 0.0, rule, **kw)["gex"].sum())
def both(c, **kw): t = G.strike_gex(c, kw.pop("spot", S), kw.pop("lot", LOT), kw.pop("T", T0), R, 0.0, "us_convention"); return f"net_gex={t.gex.sum():.4g} gross={t.gross_gex.sum():.4g} n_nan_gex={int(t.gex.isna().sum())} n_inf={int(np.isinf(t.gex).sum())}"
case("control: clean chain", lambda: both(C) + f" flip={G.gamma_flip(C, S, LOT, T0, R):.1f} max_pain={G.max_pain(C):.0f}")
def zero_oi(): c = C.copy(); c["oi"] = 0.0; return both(c) + f" flip={G.gamma_flip(c, S, LOT, T0, R)} walls={G.oi_walls(c)} max_pain={G.max_pain(c)}"
case("all open interest zero", zero_oi)
def neg_oi(): c = C.copy(); c.loc[c.index[::7], "oi"] *= -1; return both(c)
case("14 percent of rows with negative open interest", neg_oi)
def nan_oi(): c = C.copy(); c.loc[c.index[::5], "oi"] = np.nan; return both(c) + f" | engine net via sum: {net(c):.4g} vs clean {net(C):.4g}"
case("20 percent of rows with NaN open interest", nan_oi)
def nan_oi_all(): c = C.copy(); c["oi"] = np.nan; return both(c) + f" | summary row: {G.daily_summary(c, S, LOT, T0, R, with_flip=False)['net_gex_us_convention']}"
case("all open interest NaN (empty download)", nan_oi_all)
def nan_iv_all(): c = C.copy(); c["iv"] = np.nan; return both(c)
case("all implied volatilities NaN (IV solver failed everywhere)", nan_iv_all)
def nan_iv_some(): c = C.copy(); c.loc[c.index[::4], "iv"] = np.nan; return both(c) + f" clean={net(C):.4g}"
case("25 percent IV NaN (solver failures)", nan_iv_some)
case("empty chain", lambda: both(C.iloc[0:0]))
case("expired options T = 0", lambda: both(C, T=0.0))
case("negative time to expiry T = -1 day", lambda: both(C, T=-1 / 365))
case("T = 1 minute (3e-6 years)", lambda: both(C, T=1 / (365 * 24 * 60)))
case("T = 1 second", lambda: both(C, T=1 / (365 * 24 * 3600)))
case("IV = 0 everywhere", lambda: both(C.assign(iv=0.0)))
case("IV negative", lambda: both(C.assign(iv=-0.2)))
case("IV 500 percent", lambda: both(C.assign(iv=5.0)))
case("IV 0.0001 (stale quote)", lambda: both(C.assign(iv=1e-4)))
case("spot = 0", lambda: both(C, spot=0.0))
case("spot negative", lambda: both(C, spot=-24000.0))
case("lot size 0", lambda: both(C, lot=0.0))
case("lot size negative", lambda: both(C, lot=-75.0))
def units(): a = net(C); b = net(C.assign(oi=C.oi * 75)); return f"net GEX with OI in shares (x75) / contracts = {b / a:.3f}; engine cannot tell the unit"
case("open interest in shares instead of contracts (x75)", units)
case("lot size 1 instead of 75", lambda: f"ratio vs correct = {net(C, lot=1.0) / net(C):.4f}")
def deep_otm(): c = chain(lo=0.5, hi=1.5, step=500); return both(c) + f" flip={G.gamma_flip(c, S, LOT, T0, R)}"
case("very wide strike range 50 percent to 150 percent of spot", deep_otm)
def only_otm(): c = C[C.strike > S * 1.05]; return both(c) + f" flip={G.gamma_flip(c, S, LOT, T0, R)}"
case("only deep OTM strikes (5 percent above spot)", only_otm)
def missing_atm(): c = C[(C.strike < S * 0.98) | (C.strike > S * 1.02)]; return both(c) + f" vs clean {net(C):.4g} flip={G.gamma_flip(c, S, LOT, T0, R):.1f} clean_flip={G.gamma_flip(C, S, LOT, T0, R):.1f}"
case("missing strikes: no strikes within 2 percent of spot", missing_atm)
def missing_puts(): c = C[C.is_call]; return both(c) + f" vs clean {net(C):.4g}"
case("all puts missing (calls only)", missing_puts)
def dup(): c = pd.concat([C, C.iloc[90:110]]); return both(c) + f" vs clean {net(C):.4g}"
case("20 duplicated rows near the money (double counted open interest)", dup)
def strings(): c = C.copy(); c["is_call"] = np.where(C.is_call, "CE", "PE"); return f"net with string option type = {net(c):.4g} vs correct {net(C):.4g}"
case("option type given as strings CE/PE instead of booleans", strings)
def intcall(): c = C.copy(); c["is_call"] = np.where(C.is_call, 1, 0); return f"net with 1/0 = {net(c):.4g} vs correct {net(C):.4g}"
case("option type given as 1/0 integers", intcall)
case("unknown sign rule", lambda: net(C, rule="foo"))
case("participant rule without weights", lambda: net(C, rule="participant"))
case("participant weights outside [-1, 1] (2, -3)", lambda: net(C, rule="participant", participant=G.ParticipantWeights(2.0, -3.0)) / net(C))
case("missing column iv", lambda: net(C.drop(columns="iv")))
case("column oi renamed OI", lambda: net(C.rename(columns={"oi": "OI"})))
case("strike given as string", lambda: net(C.assign(strike=C.strike.astype(str))))
case("flip search with all-calls chain (no sign change)", lambda: G.gamma_flip(C[C.is_call], S, LOT, T0, R))
case("daily_summary on zero chain", lambda: G.daily_summary(C.assign(oi=0.0), S, LOT, T0, R, with_flip=True))
case("oi_walls on empty chain", lambda: G.oi_walls(C.iloc[0:0]))
case("max_pain on empty chain", lambda: G.max_pain(C.iloc[0:0]))
# implied-vol solver
def iv_roundtrip():
    rng = np.random.default_rng(9_800_100); n = 20000
    F = np.full(n, S); K = S * np.exp(rng.uniform(-0.25, 0.25, n)); T = rng.choice([1 / 365 / 24, 1 / 365, 3 / 365, 30 / 365], n); sig = rng.uniform(0.05, 0.8, n); ic = rng.random(n) < 0.5
    p = GR.price(F, K, T, R, sig, ic); est = GR.implied_vol(p, F, K, T, R, ic)
    fail = np.isnan(est); tradable = p > 0.05  # 0.05 index points is the Nifty tick size
    bad = (~fail) & (np.abs(est - sig) > 1e-4)
    otm = np.where(ic, K > F, K < F); lowt = T < 2 / 365
    perr = np.abs(GR.price(F, K, T, R, np.where(fail, 0.2, est), ic) - p) / np.maximum(p, 1e-12)
    return (f"n=20000 NaN={int(fail.sum())} (price<=0.05: {int((fail & ~tradable).sum())}, price>0.05: {int((fail & tradable).sum())}); "
            f"NaN among OTM options with price>0.05: {int((fail & tradable & otm).sum())} of {int((tradable & otm).sum())} (ITM options: {int((fail & tradable & ~otm).sum())} of {int((tradable & ~otm).sum())}; ITM time value is below float resolution); IV error>1e-4 among solved: {int(bad.sum())} (of which price>0.05: {int((bad & tradable).sum())}); "
            f"max relative PRICE error of solved options with price>0.05: {perr[(~fail) & tradable].max():.1e}")
case("IV solver round trip, +-25 percent moneyness, 1 hour to 30 days", iv_roundtrip)
def iv_bad_prices():
    F, K, T = S, S, 3 / 365; ic = True
    out = {}
    for nm, p in [("below intrinsic", -5.0), ("zero", 0.0), ("above forward", 2 * S), ("NaN", np.nan), ("inf", np.inf)]:
        out[nm] = float(GR.implied_vol(p, F, K, T, R, ic))
    out["T=0"] = float(GR.implied_vol(100.0, F, K, 0.0, R, ic)); out["deep ITM at intrinsic"] = float(GR.implied_vol(0.9 * S - 0.0, F, 0.1 * S, T, R, ic))
    return str(out)
case("IV solver on impossible prices", iv_bad_prices)
pd.DataFrame(ROWS).to_csv(sys.argv[1], index=False); print(len(ROWS), "cases")
