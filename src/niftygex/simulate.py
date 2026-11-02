"""Synthetic Nifty-like market with a known dealer-gamma effect.

Everything this module produces is SYNTHETIC. It exists for two reasons:

1. Validate the estimators: data are generated from a model whose true
   parameters are known, so we can check that mcsGARCH, the Black-76 engine,
   the GEX engine and the forecast tests recover what was put in.
2. Design the real study: by switching the gamma effect on and off and
   making dealer positions more or less observable, we can measure how much
   data the real test needs and whether the test produces false positives.

Nothing here is evidence about the real Nifty market.

Data-generating process (per day t, minute i of M):

    r_{t,i} = sqrt(h_t * s_i(expiry_t) * q_{t,i} * m_{t,i}) * z_{t,i}

* h_t      daily open-to-close variance, EGARCH(1,1) in logs.
* s_i      U-shaped diurnal curve (sums to 1 over the day). On expiry days
           an extra late-session bump is added, matching the paper's finding
           that expiry days are more volatile.
* q_{t,i}  intraday GARCH(1,1) with unit unconditional mean, Student-t z.
* m_{t,i}  dealer hedging channel: exp(-kappa * clip(G_{t,i}, -3, 3)), where
           G is the dealers' true net GEX at the current index level and time
           to expiry, divided by a fixed reference size. Long-gamma dealers
           (G > 0) dampen variance, short-gamma dealers amplify it. kappa = 0
           switches the channel off.

Dealer positioning: dealers' net share of call and put OI (d_c, d_p in
[-1, 1]) follows a persistent AR(1). Its default mean is negative for both
calls and puts, a stylised version of retail clients in India being net
option buyers. The "participant" file the researcher sees is d plus noise;
the US convention (+1 calls, -1 puts) ignores d entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import greeks
from .expiry_calendar import label_days


@dataclass
class SimConfig:
    n_days: int = 500
    minutes: int = 375
    start: str = "2024-11-20"
    seed: int = 0
    spot0: float = 24000.0
    # daily EGARCH on open-to-close variance
    daily_vol_ann: float = 0.11
    eg_alpha: float = 0.12
    eg_gamma: float = -0.07
    eg_beta: float = 0.95
    # diurnal curve
    open_amp: float = 2.5
    open_tau: float = 15.0
    close_amp: float = 0.7
    close_tau: float = 25.0
    expiry_amp: float = 0.8
    expiry_tau: float = 40.0
    # intraday GARCH(1,1)
    alpha: float = 0.03
    beta: float = 0.95
    nu: float = 7.0
    # dealer gamma channel
    kappa: float = 0.0
    gex_update_every: int = 5
    dealer_mean_call: float = -0.25
    dealer_mean_put: float = -0.25
    dealer_rho: float = 0.9
    dealer_sd: float = 0.3
    participant_noise: float = 0.1
    # option chain
    lot_size: float = 75.0
    strike_step: float = 50.0
    n_strikes: int = 40
    r: float = 0.065
    vrp: float = 1.15
    # volume
    volume_base: float = 4000.0
    expiry_volume_boost: float = 0.4
    substeps: int = 8


def diurnal_curve(cfg: SimConfig, expiry: bool) -> np.ndarray:
    i = np.arange(cfg.minutes, dtype=float)
    M = cfg.minutes
    s = 1 + cfg.open_amp * np.exp(-i / cfg.open_tau) + cfg.close_amp * np.exp(-(M - 1 - i) / cfg.close_tau)
    if expiry:
        s = s + cfg.expiry_amp * np.exp(-(M - 1 - i) / cfg.expiry_tau)
    return s


def _oi_profile(rng, spot, strikes, n_days_to_exp, cfg):
    """Call and put OI (contracts) by strike for the nearest weekly expiry."""
    x = np.log(strikes / spot)
    build = 1.0 + 1.5 * (1 - min(n_days_to_exp, 5) / 5)  # OI builds into expiry
    base = 4e4 * build
    call = base * np.exp(-0.5 * ((x - 0.01) / 0.02) ** 2)
    put = base * np.exp(-0.5 * ((x + 0.01) / 0.02) ** 2)
    rounds = (np.mod(strikes, 500) == 0).astype(float)
    hundreds = (np.mod(strikes, 100) == 0).astype(float)
    call *= 1 + 1.5 * rounds + 0.3 * hundreds
    put *= 1 + 1.5 * rounds + 0.3 * hundreds
    call *= rng.lognormal(0, 0.25, len(strikes))
    put *= rng.lognormal(0, 0.25, len(strikes))
    return np.round(call), np.round(put)


def _smile(atm_iv, strikes, spot):
    x = np.log(strikes / spot)
    return atm_iv * np.clip(1 - 1.5 * x + 8 * x**2, 0.6, 3.0)


def simulate_market(cfg: SimConfig | None = None) -> dict:
    """Simulate a market. Returns a dict of numpy arrays and pandas tables."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    T, M = cfg.n_days, cfg.minutes
    days = pd.bdate_range(cfg.start, periods=T)
    labels = label_days(days)
    is_exp = labels["is_expiry"].to_numpy()
    tdte = labels["trading_days_to_expiry"].fillna(0).to_numpy().astype(int)

    s_norm = diurnal_curve(cfg, False)
    s_exp = diurnal_curve(cfg, True)
    s_norm_n = s_norm / s_norm.sum()
    s_exp_n = s_exp / s_norm.sum()  # expiry curve keeps the extra variance

    # daily EGARCH
    lvar_bar = np.log(cfg.daily_vol_ann**2 / 252)
    omega = (1 - cfg.eg_beta) * lvar_bar
    lh = np.empty(T)
    lh[0] = lvar_bar

    # dealer positions held during day t (fixed overnight at the close of t-1)
    dc = np.empty(T + 1)
    dp = np.empty(T + 1)
    dc[0], dp[0] = cfg.dealer_mean_call, cfg.dealer_mean_put
    innov_sd = cfg.dealer_sd * np.sqrt(1 - cfg.dealer_rho**2)
    for t in range(1, T + 1):
        dc[t] = cfg.dealer_mean_call + cfg.dealer_rho * (dc[t - 1] - cfg.dealer_mean_call) + rng.normal(0, innov_sd)
        dp[t] = cfg.dealer_mean_put + cfg.dealer_rho * (dp[t - 1] - cfg.dealer_mean_put) + rng.normal(0, innov_sd)
    dc, dp = np.clip(dc, -1, 1), np.clip(dp, -1, 1)

    # reference GEX size: gross GEX of a typical chain 3 days before expiry
    strikes0 = np.round(cfg.spot0 / cfg.strike_step) * cfg.strike_step + cfg.strike_step * np.arange(-cfg.n_strikes, cfg.n_strikes + 1)
    c0, p0 = _oi_profile(np.random.default_rng(12345), cfg.spot0, strikes0, 3, cfg)
    iv0 = _smile(cfg.daily_vol_ann * cfg.vrp, strikes0, cfg.spot0)
    g0 = greeks.spot_gamma(cfg.spot0, strikes0, 3.5 / 252, cfg.r, iv0)
    gex_ref = float(np.sum(g0 * (c0 + p0)) * cfg.lot_size * cfg.spot0**2 * 0.01)

    R = np.empty((T, M))
    O = np.empty((T, M))
    H = np.empty((T, M))
    L = np.empty((T, M))
    C = np.empty((T, M))
    V = np.empty((T, M))
    Mult = np.ones((T, M))
    G = np.zeros((T, M))
    q2 = 1.0
    z_prev = 0.0
    spot = cfg.spot0
    chains = []
    vix = np.empty(T)
    a, b = cfg.alpha, cfg.beta
    om_q = 1 - a - b
    t_scale = np.sqrt((cfg.nu - 2) / cfg.nu)

    def make_chain(spot_ref, days_to_exp, h_next):
        strikes = np.round(spot_ref / cfg.strike_step) * cfg.strike_step + cfg.strike_step * np.arange(-cfg.n_strikes, cfg.n_strikes + 1)
        call_oi, put_oi = _oi_profile(rng, spot_ref, strikes, days_to_exp, cfg)
        iv = _smile(np.sqrt(h_next * 252) * cfg.vrp, strikes, spot_ref)
        return pd.DataFrame({
            "strike": np.r_[strikes, strikes],
            "is_call": np.r_[np.ones(len(strikes), bool), np.zeros(len(strikes), bool)],
            "oi": np.r_[call_oi, put_oi],
            "iv": np.r_[iv, iv],
        })

    def next_lh(lh_prev, z):
        return omega + cfg.eg_alpha * (abs(z) - np.sqrt(2 / np.pi)) + cfg.eg_gamma * z + cfg.eg_beta * lh_prev

    active = make_chain(spot, tdte[0], np.exp(lh[0]))
    for t in range(T):
        h = np.exp(lh[t])
        s_day = s_exp_n if is_exp[t] else s_norm_n
        K_all = active["strike"].to_numpy(float)
        oi_all = active["oi"].to_numpy(float)
        iv_all = active["iv"].to_numpy(float)
        w_all = np.where(active["is_call"].to_numpy(bool), dc[t], dp[t])

        z = rng.standard_t(cfg.nu, M) * t_scale
        sub = rng.standard_normal((M, cfg.substeps))
        g_now = 0.0
        for i in range(M):
            if cfg.kappa != 0 and i % cfg.gex_update_every == 0:
                tau = max((tdte[t] + (M - i) / M) / 252, 5 / (M * 252))
                gam = greeks.spot_gamma(spot, K_all, tau, cfg.r, iv_all)
                g_now = float(np.sum(w_all * gam * oi_all) * cfg.lot_size * spot**2 * 0.01 / gex_ref)
            m = np.exp(-cfg.kappa * np.clip(g_now, -3, 3)) if cfg.kappa != 0 else 1.0
            var = h * s_day[i] * q2 * m
            ret = np.sqrt(var) * z[i]
            # intrabar path for high and low: substeps pinned to end at ret
            path = np.cumsum(sub[i]) * np.sqrt(var / cfg.substeps)
            path = path - (path[-1] - ret) * np.arange(1, cfg.substeps + 1) / cfg.substeps
            O[t, i] = spot
            H[t, i] = spot * np.exp(max(0.0, path.max()))
            L[t, i] = spot * np.exp(min(0.0, path.min()))
            spot = spot * np.exp(ret)
            C[t, i] = spot
            R[t, i] = ret
            Mult[t, i] = m
            G[t, i] = g_now
            V[t, i] = cfg.volume_base * s_day[i] * M * np.sqrt(q2 * m) * (1 + cfg.expiry_volume_boost * is_exp[t]) * rng.lognormal(0, 0.3)
            # intraday GARCH update (unit unconditional mean)
            q2 = om_q + a * q2 * z[i] ** 2 + b * q2
        z_prev = R[t].sum() / np.sqrt(h)
        lh_next = next_lh(lh[t], z_prev)
        if t + 1 < T:
            lh[t + 1] = lh_next
        # Close of day t: publish the chain that will be live on day t+1 (bhavcopy),
        # India VIX for the next day, and the participant file.
        dte_next = tdte[t + 1] if t + 1 < T else 4
        nxt = make_chain(spot, dte_next, np.exp(lh_next))
        tau_close = (dte_next + 1) / 252
        nxt["settle"] = greeks.price(spot, nxt["strike"].to_numpy(), tau_close, cfg.r, nxt["iv"].to_numpy(), nxt["is_call"].to_numpy())
        nxt.attrs.update({"spot": spot, "T": tau_close, "date": days[t]})
        chains.append(nxt)
        vix[t] = np.sqrt(np.exp(lh_next) * 252) * cfg.vrp * np.exp(rng.normal(0, 0.05))
        active = nxt

    # participant file published at the close of day t describes positions held on day t+1
    noise = cfg.participant_noise
    participant = pd.DataFrame({
        "pro_call": np.clip(dc[1:] + rng.normal(0, noise, T), -1, 1),
        "pro_put": np.clip(dp[1:] + rng.normal(0, noise, T), -1, 1),
        "client_call": np.clip(-0.8 * dc[1:] + rng.normal(0, noise * 1.5, T), -1, 1),
        "client_put": np.clip(-0.8 * dp[1:] + rng.normal(0, noise * 1.5, T), -1, 1),
    }, index=days)
    # true positions for the next day, known only to the simulator (oracle benchmark)
    dealer_next = pd.DataFrame({"call": dc[1:], "put": dp[1:]}, index=days)
    dc, dp = dc[:T], dp[:T]

    return {
        "config": asdict(cfg),
        "days": days,
        "labels": labels,
        "R": R, "open": O, "high": H, "low": L, "close": C, "volume": V,
        "h_true": np.exp(lh),
        "s_true": s_norm_n,
        "s_true_expiry": s_exp_n,
        "multiplier": Mult,
        "G_true": G,
        "dealer_call": dc,
        "dealer_put": dp,
        "participant": participant,
        "dealer_next": dealer_next,
        "chains": chains,
        "vix": vix,
        "gex_ref": gex_ref,
        "lot_size": cfg.lot_size,
        "r": cfg.r,
    }


def export_nse_layout(market: dict, out_dir) -> dict:
    """Write a simulated market to files in the NSE layouts that data.py reads.

    Used to test the real-data path end to end. The files are SYNTHETIC.
    Bhavcopy uses the UDiFF layout with OI in units (contracts x lot size).
    """
    from pathlib import Path

    out = Path(out_dir)
    (out / "bhavcopy").mkdir(parents=True, exist_ok=True)
    (out / "participant_oi").mkdir(parents=True, exist_ok=True)
    days = market["days"]
    M = market["R"].shape[1]
    lot = market["lot_size"]
    # 1-minute bars
    ts = np.concatenate([d + pd.Timedelta(hours=9, minutes=15) + pd.to_timedelta(np.arange(M), unit="min") for d in days])
    bars = pd.DataFrame({"timestamp": ts, "open": market["open"].ravel(), "high": market["high"].ravel(),
                         "low": market["low"].ravel(), "close": market["close"].ravel(),
                         "volume": market["volume"].ravel()})
    bars.to_csv(out / "nifty_1min.csv", index=False)
    labels = market["labels"]
    exp_dates = labels.index[labels["is_expiry"]]
    for t, (d, ch) in enumerate(zip(days, market["chains"])):
        later = exp_dates[exp_dates > d]
        expiry = later[0] if len(later) else d + pd.offsets.BDay(5)
        n = len(ch)
        udiff = pd.DataFrame({
            "TradDt": [d.strftime("%Y-%m-%d")] * n, "BizDt": d.strftime("%Y-%m-%d"), "Sgmt": "FO", "Src": "NSE",
            "FinInstrmTp": "IDO", "FinInstrmId": np.arange(n), "ISIN": "", "TckrSymb": "NIFTY", "SctySrs": "",
            "XpryDt": expiry.strftime("%Y-%m-%d"), "FininstrmActlXpryDt": expiry.strftime("%Y-%m-%d"),
            "StrkPric": ch["strike"].to_numpy(), "OptnTp": np.where(ch["is_call"], "CE", "PE"),
            "FinInstrmNm": "", "OpnPric": ch["settle"].to_numpy(), "HghPric": ch["settle"].to_numpy(),
            "LwPric": ch["settle"].to_numpy(), "ClsPric": ch["settle"].to_numpy(), "LastPric": ch["settle"].to_numpy(),
            "PrvsClsgPric": ch["settle"].to_numpy(), "UndrlygPric": ch.attrs["spot"], "SttlmPric": ch["settle"].to_numpy(),
            "OpnIntrst": ch["oi"].to_numpy() * lot, "ChngInOpnIntrst": 0, "TtlTradgVol": 0, "TtlTrfVal": 0,
            "TtlNbOfTxsExctd": 0, "SsnId": "F1", "NewBrdLotQty": lot, "Rmks": "", "Rsvd1": "", "Rsvd2": "",
            "Rsvd3": "", "Rsvd4": "",
        })
        # one extra row for a farther expiry and a futures row, which the loader must skip
        udiff.to_csv(out / "bhavcopy" / f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv", index=False)
        # participant OI file (title line, header, 5 rows)
        tot = float(ch["oi"].sum()) / 2
        pr = market["participant"].iloc[t]
        def side(wc, wp, base):
            return {"cl": max(wc, 0) * tot + base * tot, "cs": max(-wc, 0) * tot + base * tot,
                    "pl": max(wp, 0) * tot + base * tot, "ps": max(-wp, 0) * tot + base * tot}
        pro = side(pr["pro_call"], pr["pro_put"], 0.05)
        cli = side(pr["client_call"], pr["client_put"], 0.05)
        dii = side(0.0, 0.0, 0.02)
        fii = {}
        for leg_l, leg_s in (("cl", "cs"), ("pl", "ps")):
            longs = pro[leg_l] + cli[leg_l] + dii[leg_l]
            shorts = pro[leg_s] + cli[leg_s] + dii[leg_s]
            fl = max(0.05 * tot, tot - longs, shorts - longs + 0.05 * tot)
            fii[leg_l] = fl
            fii[leg_s] = longs + fl - shorts
        rows = []
        for name, g in (("Client", cli), ("DII", dii), ("FII", fii), ("Pro", pro)):
            rows.append([name, 0, 0, 0, 0, round(g["cl"]), round(g["pl"]), round(g["cs"]), round(g["ps"]), 0, 0, 0, 0, 0, 0])
        tot_row = ["TOTAL"] + [sum(r[k] for r in rows) for k in range(1, 15)]
        rows.append(tot_row)
        hdr = ["Client Type", "Future Index Long", "Future Index Short", "Future Stock Long", "Future Stock Short",
               "Option Index Call Long", "Option Index Put Long", "Option Index Call Short", "Option Index Put Short",
               "Option Stock Call Long", "Option Stock Put Long", "Option Stock Call Short", "Option Stock Put Short",
               "Total Long Contracts", "Total Short Contracts"]
        with open(out / "participant_oi" / f"fao_participant_oi_{d:%d%m%Y}.csv", "w") as fh:
            fh.write(f'"Participant wise Open Interest (no. of contracts) in Equity Derivatives as on {d:%b %d, %Y}"\n')
            pd.DataFrame(rows, columns=hdr).to_csv(fh, index=False)
    pd.DataFrame({"date": [d.strftime("%d-%m-%Y") for d in days], "close": market["vix"] * 100}).to_csv(out / "india_vix.csv", index=False)
    return {"bars": out / "nifty_1min.csv", "bhavcopy": out / "bhavcopy", "participant": out / "participant_oi",
            "vix": out / "india_vix.csv"}
