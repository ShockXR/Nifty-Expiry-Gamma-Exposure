"""Expiry-day pinning of Nifty and Bank Nifty to option strikes, and dealer gamma (REAL DATA).

Question
--------
Does the official close of Nifty 50 (and Nifty Bank, while it had weekly
options) end unusually close to listed option strikes on its own expiry
days, and is any such "pinning" stronger when option dealers are likely to
be net long gamma near that strike? (Avellaneda and Lipkin 2003 model;
Ni, Pearson and Poteshman 2005 for US stocks; Golez and Jackwerth 2012 for
S&P 500 futures.) The sign of dealer gamma is NOT observed on NSE: it is
inferred under three rules (US convention, Pro participant weights, mirror
of Client), and every gamma result is reported under each.

Pre-specified primary tests (fixed before any result was looked at)
------------------------------------------------------------------
P1  Nifty, all expiry days: mean distance (index points) from the official
    close to the nearest listed strike (50-point grid), expiry days minus
    non-expiry days. Two-sided permutation test over days.
P2  Nifty expiry days: the same distance, days where net dealer gamma near
    that strike is positive minus days where it is negative, under each of
    the three sign rules. Two-sided permutation tests, Holm-adjusted over
    the three rules.
Everything else (Bank Nifty, regimes, target strikes, intraday movement,
other conditioning variables, robustness) is secondary and exploratory, and
is labelled as such in the outputs.

No look-ahead
-------------
For a day t whose next Nifty expiry is E (E = t on expiry days), every
option quantity (OI, settlement prices, implied vols, max-OI / max-pain /
max-gamma strikes, dealer gamma) comes from the bhavcopy of the last trading
day strictly before t, for the contract expiring on E. That file is
published after the close of t-1. Participant weights also come from t-1.

Data (all REAL, read from data/raw)
-----------------------------------
* bhavcopy/fo_YYYYMMDD.csv.gz via niftygex.data.load_bhavcopy (OI in shares).
* participant_oi/*.csv via niftygex.data.load_participant_oi.
* index_daily.csv: NSE official closes (the settlement price of expiring
  index options; this script verifies that from the bhavcopies).
* nifty_1min.csv, banknifty_1min.csv: third-party 1-minute bars validated
  against NSE daily highs and lows (results/data_validation).

    OMP_NUM_THREADS=1 python experiments/pinning_study.py --raw data/raw --out results/pinning

Writes results/pinning/*.csv, results/pinning/*.png and results/pinning_stdout.txt.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from niftygex import data, gex, greeks  # noqa: E402
from niftygex.expiry_calendar import ExpiryRules, label_days  # noqa: E402

LABEL = "REAL DATA"
SYMBOLS = {
    "NIFTY": {"index": "Nifty 50", "bars": "nifty_1min.csv", "grid": 50.0},
    "BANKNIFTY": {"index": "Nifty Bank", "bars": "banknifty_1min.csv", "grid": 100.0},
}
RATE = 0.065                     # risk-free rate for Black-76, as in the pipeline
TRADING_DAYS = 252.0
SESSION_MINUTES = 375            # 09:15 to 15:29 bar starts
TAU_SETTLE = (30.0 / SESSION_MINUTES) / TRADING_DAYS   # 30 trading minutes, in years
RULES = ("us_convention", "participant_pro", "client_mirror")
BN_WEEKLY_END = pd.Timestamp("2024-11-20")  # Bank Nifty weekly options stop (SEBI, 20 Nov 2024)
# NSE closing auction session (CAS) live from 3 Aug 2026 (NSE circulars NSE/CMTR/73362 of 18 Mar 2026 and
# NSE/CMTR/75479 of 30 Jul 2026): closing prices of F&O stocks come from a 15:20 to 15:30 auction, and the
# 1-minute index is frozen from about 15:15 until the auction result is published.
CAS_START = pd.Timestamp("2026-08-03")
N_PERM = 10000
N_BOOT = 10000
SEED = 20261102

# Minute-bar slots (bar start minus 09:15)
SLOT_1400, SLOT_1429, SLOT_1459, SLOT_1500, SLOT_1529 = 285, 314, 344, 345, 374

DESIGN_LOG = [
    ("settlement price", "Use NSE's official close as the outcome, not the mean of 15:00 to 15:29 minute bars",
     "The bhavcopy settlement price of every expiring option equals the official close exactly on every "
     "expiry day (checked here), while the minute-bar mean differs from it by about 1 point (median)."),
    ("max-gamma strike", "Defined from gross (unsigned) gamma exposure",
     "A signed 'largest net GEX' strike depends on the unobserved sign; the sign enters only in the "
     "conditioning step, where all three rules are shown."),
    ("movement test", "Added the signed move toward the strike as the main movement measure",
     "The change in absolute distance has a positive mean under a pure random walk (Jensen's inequality) "
     "that grows with volatility, and expiry days are more volatile; the signed move has mean zero "
     "under no pinning whatever the volatility."),
    ("implied vols", "Interpolate missing IVs along strikes after sorting by strike",
     "UDiFF bhavcopy rows are not in strike order; interpolating in file order fills deep options with IVs of "
     "unrelated strikes. (data.load_bhavcopy now sorts rows; the script sorts again so it does not depend on that.)"),
    ("re-dated expiries", "If no contract with expiry E is listed on t-1, use the contract listed within 3 calendar "
     "days of E", "Holiday changes re-date contracts: on 27 Jun 2023 the Nifty contract was still listed for 29 Jun "
     "but expired on 28 Jun; Bank Nifty contracts were re-dated in Sep 2023 and Jan 2025."),
    ("closing auction", "Split the Tuesday regime at 3 Aug 2026 and restrict settlement-window bar measures "
     "(15:29 bar, 15:00 to 15:29 mean, intraday figure) to days before it",
     "NSE's closing auction session went live on 3 Aug 2026 (circular NSE/CMTR/75479); the official close then "
     "comes from an auction and the minute index is frozen from 15:15, so the 30-minute bar mean no longer "
     "approximates the close."),
    ("regime heterogeneity", "Added pairwise tests of whether the expiry effect differs between regimes, and a "
     "half-year table", "The first run showed opposite signs in the Thursday and Tuesday regimes; whether they "
     "really differ is a test, not something to read off two separate p-values."),
    ("gamma near a strike", "Dealer gamma at a strike is evaluated as if the index sat at that strike with "
     "30 trading minutes left (the settlement window)",
     "With one full day left, gamma at a level spreads over about +/-5 strikes; with 30 minutes left it "
     "weights the strike itself and its immediate neighbours, which is what 'near the strike' means."),
]


# ---------------------------------------------------------------------------
# Small helpers (unit-tested in tests/test_pinning.py)
# ---------------------------------------------------------------------------


def nearest_grid(x, grid: float, offset: float = 0.0):
    """Nearest point of the lattice {offset + j * grid} to x (vectorised)."""
    x = np.asarray(x, float)
    return offset + grid * np.round((x - offset) / grid)


def dist_to_grid(x, grid: float, offset: float = 0.0):
    """Distance from x to the nearest point of {offset + j * grid}; lies in [0, grid / 2]."""
    x = np.asarray(x, float)
    return np.abs(x - nearest_grid(x, grid, offset))


def placebo_dist(x, grid: float):
    """Distance to the nearest half-grid 'placebo strike' (strikes shifted by grid / 2)."""
    return dist_to_grid(x, grid, offset=grid / 2.0)


def signed_move_toward(p0, p1, K):
    """Move from p0 to p1 measured toward K (positive = toward). Zero when p0 sits on K."""
    p0, p1, K = (np.asarray(a, float) for a in (p0, p1, K))
    return (p1 - p0) * np.sign(K - p0)


def change_in_distance(p0, p1, K):
    """|p1 - K| - |p0 - K| (negative = ended closer to K)."""
    p0, p1, K = (np.asarray(a, float) for a in (p0, p1, K))
    return np.abs(p1 - K) - np.abs(p0 - K)


def prev_trading_day(days: pd.DatetimeIndex, t: pd.Timestamp, available=None):
    """Last day in ``days`` strictly before t (and in ``available`` if given). None if there is none."""
    pos = days.searchsorted(pd.Timestamp(t))  # first index with days[i] >= t
    for j in range(pos - 1, -1, -1):
        if available is None or days[j] in available:
            return days[j]
    return None


def chain_before(bhav: dict, days: pd.DatetimeIndex, t, expiry):
    """Option chain of the contract expiring on ``expiry`` from the last bhavcopy strictly before t.

    Returns (chain, chain_date, stale) where stale counts trading days skipped
    because a bhavcopy was missing. The chain never comes from day t itself
    or later, which is what makes every conditioning variable known before
    day t opens.
    """
    t = pd.Timestamp(t)
    d = prev_trading_day(days, t, available=bhav)
    if d is None:
        return None, None, None
    assert d < t
    stale = int(days.searchsorted(t) - days.searchsorted(d) - 1)
    full = bhav[d]
    exps = pd.to_datetime(full["expiry"]).dt.normalize()
    E = pd.Timestamp(expiry)
    ch = full[exps == E]
    if ch.empty:
        # holiday re-dating: the same contract may still carry its old expiry date on t-1
        cand = sorted(x for x in exps.unique() if x > d and abs((x - E).days) <= 3)
        if cand:
            ch = full[exps == cand[0]]
            ch.attrs["redated_from"] = cand[0]
    return ch, d, stale


def prepare_chain(ch: pd.DataFrame, T: float, r: float, fallback_forward: float) -> pd.DataFrame:
    """Sort by strike, back out Black-76 IVs from settlement prices, fill gaps along strikes."""
    c = ch[["strike", "is_call", "oi", "settle"]].copy()
    c = c.sort_values(["is_call", "strike"]).reset_index(drop=True)
    F = data.parity_forward(c, T, r)
    if not (np.isfinite(F) and abs(F / fallback_forward - 1) < 0.05):
        F = float(fallback_forward)
    iv = greeks.implied_vol(c["settle"].to_numpy(float), F, c["strike"].to_numpy(float), T, r,
                            c["is_call"].to_numpy(bool))
    c["iv_raw"] = iv
    c["iv"] = c.groupby("is_call")["iv_raw"].transform(lambda s: s.interpolate(limit_direction="both"))
    med = np.nanmedian(c["iv_raw"]) if np.isfinite(c["iv_raw"]).any() else 0.15
    c["iv"] = c["iv"].fillna(med)
    c.attrs.update({"F": float(F), "T": float(T), "iv_nan_share": float(np.isnan(iv).mean())})
    return c


def rule_weights(chain: pd.DataFrame, rule: str, ptable: pd.DataFrame | None) -> np.ndarray:
    """Dealer sign weight per option series under a sign rule (see niftygex.gex)."""
    is_call = chain["is_call"].to_numpy(bool)
    if rule == "gross":
        return np.ones(len(chain))
    if rule == "us_convention":
        return gex.sign_weights(is_call, "us_convention")
    if ptable is None:
        return np.full(len(chain), np.nan)
    if rule == "participant_pro":
        return gex.sign_weights(is_call, "participant", gex.ParticipantWeights.from_participant_oi(ptable, "Pro"))
    if rule == "client_mirror":
        return gex.sign_weights(is_call, "participant",
                                gex.ParticipantWeights.from_participant_oi(ptable, "Client", mirror=True))
    raise ValueError(rule)


def dealer_gamma_at(chain: pd.DataFrame, levels, tau: float, weights, r: float = RATE) -> np.ndarray:
    """Net dealer gamma exposure (rupees per 1% move; OI in shares) if the index stood at each level.

    sum_k w_k * OI_k * Gamma(level, K_k, tau, iv_k) * level^2 * 0.01, the
    package's GEX definition evaluated at a hypothetical index level.
    """
    lv = np.atleast_1d(np.asarray(levels, float))[:, None]
    K = chain["strike"].to_numpy(float)[None, :]
    iv = chain["iv"].to_numpy(float)[None, :]
    oi = chain["oi"].to_numpy(float)
    g = greeks.spot_gamma(lv, K, tau, r, iv)
    return (g * (np.asarray(weights, float) * oi)[None, :]).sum(axis=1) * lv[:, 0] ** 2 * 0.01


def strike_targets(chain: pd.DataFrame, F: float, T: float, r: float = RATE) -> dict:
    """Max-OI, max-pain and max-gross-gamma strikes of a chain (rows with OI > 0)."""
    c = chain[chain["oi"] > 0]
    if c.empty:
        return {"K_maxoi": np.nan, "K_maxpain": np.nan, "K_maxgamma": np.nan, "maxoi_share": np.nan}
    walls = gex.oi_walls(c)
    tab = gex.strike_gex(c, F, 1.0, T, r, 0.0, "long_all", forward=F)
    by = gex.gex_by_strike(tab)
    tot = c.groupby("strike")["oi"].sum()
    return {"K_maxoi": walls["max_oi_strike"], "K_maxpain": gex.max_pain(c),
            "K_maxgamma": float(by["gross_gex"].idxmax()), "maxoi_share": float(tot.max() / tot.sum())}


# ---------------------------------------------------------------------------
# Inference helpers (day-level)
# ---------------------------------------------------------------------------


def perm_test_diff(x, y, rng, n: int = N_PERM, strata_x=None, strata_y=None):
    """Two-sided permutation p-value for mean(x) - mean(y); optional strata (labels shuffled within)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = np.concatenate([x, y])
    lab = np.concatenate([np.ones(len(x), bool), np.zeros(len(y), bool)])
    obs = x.mean() - y.mean()
    if strata_x is None:
        strata = np.zeros(len(z), int)
    else:
        strata = pd.factorize(np.concatenate([np.asarray(strata_x), np.asarray(strata_y)]))[0]
    null = np.empty(n)
    done = 0
    while done < n:
        m = min(2000, n - done)
        L = np.tile(lab, (m, 1))
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            L[:, idx] = rng.permuted(L[:, idx], axis=1)
        nx = L.sum(axis=1)
        sx = (L * z).sum(axis=1)
        null[done:done + m] = sx / nx - (z.sum() - sx) / (len(z) - nx)
        done += m
    return float(obs), float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (n + 1))


def sign_flip_test(d, rng, n: int = N_PERM):
    """Two-sided randomization (sign-flip) p-value for mean(d) = 0, for paired differences."""
    d = np.asarray(d, float)
    obs = d.mean()
    s = rng.choice(np.array([-1.0, 1.0]), size=(n, len(d)))
    null = (s * d).mean(axis=1)
    return float(obs), float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (n + 1))


def boot_ci_diff(x, y, rng, n: int = N_BOOT, level: float = 0.95):
    x, y = np.asarray(x, float), np.asarray(y, float)
    bx = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    by = y[rng.integers(0, len(y), (n, len(y)))].mean(axis=1) if len(y) else np.zeros(n)
    a = (1 - level) / 2
    return tuple(np.quantile(bx - by, [a, 1 - a]))


def boot_ci_mean(x, rng, n: int = N_BOOT, level: float = 0.95):
    return boot_ci_diff(x, [], rng, n, level) if len(x) else (np.nan, np.nan)


def uniform_mean_test(d, half_grid: float, rng, n: int = N_PERM):
    """Monte Carlo two-sided p-value for mean(d) against d ~ Uniform(0, half_grid) (no clustering)."""
    d = np.asarray(d, float)
    null = rng.uniform(0, half_grid, (n, len(d))).mean(axis=1)
    mu = half_grid / 2
    return float((1 + np.sum(np.abs(null - mu) >= abs(d.mean() - mu) - 1e-12)) / (n + 1))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_inputs(raw: Path):
    idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
    official = {s: idx[idx["index"].str.lower() == v["index"].lower()].set_index("date")["close"].sort_index()
                for s, v in SYMBOLS.items()}
    bhav = {s: {} for s in SYMBOLS}
    for f in data.bhavcopy_files(raw / "bhavcopy"):
        for s in SYMBOLS:
            df = data.load_bhavcopy(f, s)
            if len(df):
                bhav[s][pd.Timestamp(df["date"].iloc[0]).normalize()] = df
    ptables = {}
    for f in sorted((raw / "participant_oi").glob("*.csv")):
        try:
            d = pd.to_datetime(f.stem[-8:], format="%d%m%Y")
        except ValueError:
            continue
        ptables[d] = data.load_participant_oi(f)
    return official, bhav, ptables


def minute_closes(raw: Path, fname: str, start: str = "2023-01-01"):
    """Bar closes on the 09:15 to 15:29 grid (days x 375), from niftygex.data loaders."""
    bars = data.load_minute_bars(raw / fname)
    bars = bars[bars.index >= pd.Timestamp(start)]
    g = data.to_session_grid(bars)
    return pd.DataFrame(g["close"], index=g["days"])


def build_day_panel(sym: str, official: pd.Series, bhav: dict, ptables: dict, closes: pd.DataFrame | None):
    """One row per official trading day: outcome distances, labels, and t-1 chain quantities."""
    grid = SYMBOLS[sym]["grid"]
    days = pd.DatetimeIndex(sorted(set(official.index) & set(bhav))).normalize()
    expiries = data.actual_expiries(bhav)
    lab = label_days(days, expiries=expiries)
    rules = ExpiryRules()
    rows, minute_rows = [], {}
    pt_dates = pd.DatetimeIndex(sorted(ptables))
    for i, t in enumerate(days):
        if i == 0:
            continue
        E = lab.loc[t, "next_expiry"]
        if pd.isna(E):
            continue
        close = float(official[t])
        close_prev = float(official[days[i - 1]])
        K_n = float(nearest_grid(close, grid))
        row = {"date": t, "symbol": sym, "close": close, "close_prev": close_prev,
               "is_expiry": bool(lab.loc[t, "is_expiry"]), "is_monthly_expiry": bool(lab.loc[t, "is_monthly_expiry"]),
               "k": int(lab.loc[t, "trading_days_to_expiry"]), "expiry": E, "weekday": int(t.weekday()),
               "regime": rules.regime(t), "K_near": K_n, "d": abs(close - K_n), "d_placebo": float(placebo_dist(close, grid)),
               "d100": float(dist_to_grid(close, 100.0)) if sym == "NIFTY" else np.nan}
        row["d_bp"] = row["d"] / close * 1e4
        # ---- chain from the previous trading day (no look-ahead)
        ch, cdate, stale = chain_before(bhav, days, t, E)
        if ch is not None and len(ch):
            n_td = int(days.searchsorted(pd.Timestamp(E)) - days.searchsorted(cdate))
            T1 = max(n_td, 1) / TRADING_DAYS
            und = ch["underlying"].dropna()
            fb = float(und.iloc[0]) if len(und) else float(official[cdate])
            c = prepare_chain(ch, T1, RATE, fb)
            F1 = c.attrs["F"]
            row.update({"chain_date": cdate, "chain_stale_days": stale, "T1_days": n_td, "F1": F1,
                        "chain_redated": "redated_from" in ch.attrs,
                        "iv_nan_share": c.attrs["iv_nan_share"], "chain_total_oi": float(c["oi"].sum())})
            row.update(strike_targets(c, F1, T1))
            tot = c.groupby("strike")["oi"].sum()
            row["oi_share_near"] = float(tot.get(K_n, 0.0) / tot.sum()) if tot.sum() > 0 else np.nan
            row["K_near_listed"] = bool(K_n in tot.index)
            pdate = cdate if cdate in ptables else (pt_dates[pt_dates <= cdate].max() if (pt_dates <= cdate).any() else None)
            ptab = ptables.get(pdate) if pdate is not None else None
            for rule in ("gross",) + RULES:
                w = rule_weights(c, rule, ptab)
                lv = [K_n, F1]
                for tk in ("K_maxoi", "K_maxgamma"):
                    lv.append(row[tk] if np.isfinite(row[tk]) else F1)
                g_near, _, g_maxoi, g_maxgam = dealer_gamma_at(c, lv, TAU_SETTLE, w)
                row[f"g_near_{rule}"] = g_near
                row[f"g_maxoi_{rule}"] = g_maxoi
                row[f"g_maxgamma_{rule}"] = g_maxgam
                row[f"g_day_{rule}"] = float(dealer_gamma_at(c, [F1], T1, w)[0])
        # ---- intraday prices from minute bars
        if closes is not None and t in closes.index:
            cl = closes.loc[t].to_numpy(float)
            win = cl[SLOT_1500:SLOT_1529 + 1]
            row.update({"p1429": cl[SLOT_1429], "p1459": cl[SLOT_1459], "p1529": cl[SLOT_1529],
                        "p1514": cl[SLOT_1459 + 15],
                        "mean30": float(np.nanmean(win)) if np.isfinite(win).any() else np.nan,
                        "n30": int(np.isfinite(win).sum())})
            minute_rows[t] = cl[SLOT_1400:SLOT_1529 + 1]
        rows.append(row)
    panel = pd.DataFrame(rows).set_index("date")
    if sym == "BANKNIFTY":
        panel["regime"] = np.where(panel.index < BN_WEEKLY_END, "weekly_options", "monthly_only")
    panel["closing_auction"] = panel.index >= CAS_START
    mins = pd.DataFrame.from_dict(minute_rows, orient="index") if minute_rows else None
    return panel, expiries, mins


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def describe_d(d, close, grid):
    d = np.asarray(d, float)
    close = np.asarray(close, float)
    return {"n": len(d), "mean_dist_pts": d.mean(), "mean_dist_bp": float(np.mean(d / close * 1e4)),
            "median_dist_pts": float(np.median(d)), "share_within_tenth_grid": float(np.mean(d <= grid / 10)),
            "share_within_twentieth_grid": float(np.mean(d <= grid / 20)),
            "placebo_share_within_tenth_grid": float(np.mean(d >= grid / 2 - grid / 10))}


def clustering_tests(panel: pd.DataFrame, sym: str, rng) -> pd.DataFrame:
    """T1: distance of the official close to the nearest strike, expiry vs non-expiry days."""
    grid = SYMBOLS[sym]["grid"]
    out = []
    samples = [("all", panel)] + [(r, panel[panel["regime"] == r]) for r in panel["regime"].unique()]
    if sym == "NIFTY":
        tue = panel[panel["regime"] == "single_weekly_tue"]
        samples += [("single_weekly_tue, before closing auction (to 31 Jul 2026)", tue[~tue["closing_auction"]]),
                    ("single_weekly_tue, closing auction (from 3 Aug 2026)", tue[tue["closing_auction"]]),
                    ("monthly expiries vs non-expiry", panel[panel["is_monthly_expiry"] | ~panel["is_expiry"]]),
                    ("weekly (non-monthly) expiries vs non-expiry", panel[~panel["is_monthly_expiry"] | ~panel["is_expiry"]])]
    for name, p in samples:
        e, o = p[p["is_expiry"]], p[~p["is_expiry"]]
        if len(e) < 5:
            continue
        de, do = describe_d(e["d"], e["close"], grid), describe_d(o["d"], o["close"], grid)
        diff, p_perm = perm_test_diff(e["d"], o["d"], rng)
        _, p_strat = perm_test_diff(e["d"], o["d"], rng, strata_x=e["weekday"], strata_y=o["weekday"])
        lo, hi = boot_ci_diff(e["d"], o["d"], rng)
        _, p_hit = perm_test_diff((e["d"] <= grid / 10).astype(float), (o["d"] <= grid / 10).astype(float), rng)
        bp_diff = de["mean_dist_bp"] - do["mean_dist_bp"]
        out.append({"symbol": sym, "sample": name, "grid_pts": grid, "uniform_null_mean_pts": grid / 4,
                    "first_day": str(p.index.min().date()), "last_day": str(p.index.max().date()),
                    "n_expiry": de["n"], "n_nonexpiry": do["n"],
                    "mean_dist_expiry_pts": de["mean_dist_pts"], "mean_dist_nonexpiry_pts": do["mean_dist_pts"],
                    "median_dist_expiry_pts": de["median_dist_pts"], "median_dist_nonexpiry_pts": do["median_dist_pts"],
                    "diff_pts": diff, "diff_ci95_low": lo, "diff_ci95_high": hi, "diff_bp": bp_diff,
                    "p_perm": p_perm, "p_perm_weekday_strata": p_strat, "p_perm_share_within_tenth": p_hit,
                    "p_expiry_vs_uniform": uniform_mean_test(e["d"], grid / 2, rng),
                    "p_nonexpiry_vs_uniform": uniform_mean_test(o["d"], grid / 2, rng),
                    "share_within_tenth_expiry": de["share_within_tenth_grid"],
                    "share_within_tenth_nonexpiry": do["share_within_tenth_grid"],
                    "placebo_share_tenth_expiry": de["placebo_share_within_tenth_grid"],
                    "placebo_share_tenth_nonexpiry": do["placebo_share_within_tenth_grid"],
                    "share_within_twentieth_expiry": de["share_within_twentieth_grid"],
                    "share_within_twentieth_nonexpiry": do["share_within_twentieth_grid"]})
    return pd.DataFrame(out)


def regime_heterogeneity(panel: pd.DataFrame, sym: str, rng, n: int = N_PERM) -> pd.DataFrame:
    """Does the expiry effect (expiry minus non-expiry mean distance) differ between two regimes?

    Statistic: gap_A - gap_B. Null: the same gap in both regimes. Regime
    labels are shuffled among expiry days of A and B and, separately, among
    non-expiry days of A and B, which keeps each day's expiry status fixed.
    """
    regs = [r for r in panel["regime"].unique() if panel.loc[panel["regime"] == r, "is_expiry"].sum() >= 5]
    out = []
    for i, a in enumerate(regs):
        for b in regs[i + 1:]:
            sub = panel[panel["regime"].isin([a, b])]
            e, o = sub[sub["is_expiry"]], sub[~sub["is_expiry"]]
            ea, oa = (e["regime"] == a).to_numpy(), (o["regime"] == a).to_numpy()
            de, do = e["d"].to_numpy(), o["d"].to_numpy()

            def gap(ma, mo):
                return (de[ma].mean() - do[mo].mean()) - (de[~ma].mean() - do[~mo].mean())

            obs = gap(ea, oa)
            null = np.array([gap(rng.permutation(ea), rng.permutation(oa)) for _ in range(n)])
            boot = []
            for _ in range(2000):
                ia = rng.integers(0, ea.sum(), ea.sum()); ib = rng.integers(0, (~ea).sum(), (~ea).sum())
                ja = rng.integers(0, oa.sum(), oa.sum()); jb = rng.integers(0, (~oa).sum(), (~oa).sum())
                boot.append((de[ea][ia].mean() - do[oa][ja].mean()) - (de[~ea][ib].mean() - do[~oa][jb].mean()))
            out.append({"symbol": sym, "regime_a": a, "regime_b": b, "n_expiry_a": int(ea.sum()), "n_expiry_b": int((~ea).sum()),
                        "gap_a_pts": de[ea].mean() - do[oa].mean(), "gap_b_pts": de[~ea].mean() - do[~oa].mean(),
                        "gap_a_minus_gap_b_pts": obs, "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "p_perm": float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (n + 1))})
    return pd.DataFrame(out)


def by_half_year(panel: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Descriptive: mean distance to the nearest strike by calendar half-year (no tests)."""
    p = panel.copy()
    p["half"] = p.index.year.astype(str) + np.where(p.index.month <= 6, " H1", " H2")
    out = []
    for h, g in p.groupby("half"):
        e, o = g[g["is_expiry"]], g[~g["is_expiry"]]
        out.append({"symbol": sym, "half_year": h, "n_expiry": len(e), "n_nonexpiry": len(o),
                    "mean_dist_expiry_pts": e["d"].mean() if len(e) else np.nan,
                    "se_expiry": e["d"].std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan,
                    "mean_dist_nonexpiry_pts": o["d"].mean(),
                    "share_within_tenth_expiry": float((e["d"] <= SYMBOLS[sym]["grid"] / 10).mean()) if len(e) else np.nan})
    return pd.DataFrame(out)


def profile_by_k(panel: pd.DataFrame, sym: str, rng) -> pd.DataFrame:
    """T2: mean distance to the nearest strike by trading days to expiry (0 = expiry day)."""
    grid = SYMBOLS[sym]["grid"]
    p = panel.copy()
    p["k_bucket"] = p["k"].clip(upper=5)
    out = []
    for k, g in p.groupby("k_bucket"):
        lo, hi = boot_ci_mean(g["d"], rng)
        out.append({"symbol": sym, "trading_days_to_expiry": "5+" if k == 5 else str(int(k)), "n": len(g),
                    "mean_dist_pts": g["d"].mean(), "ci95_low": lo, "ci95_high": hi,
                    "mean_dist_bp": float((g["d"] / g["close"] * 1e4).mean()),
                    "share_within_tenth_grid": float((g["d"] <= grid / 10).mean()), "uniform_null_mean_pts": grid / 4})
    return pd.DataFrame(out)


def target_tests(panel: pd.DataFrame, sym: str, rng) -> pd.DataFrame:
    """T3: pinning to the max-OI, max-pain and max-gamma strikes of the t-1 chain.

    Expiry days (k = 0) against (a) the day before expiry of the same
    contract (k = 1, paired by expiry cycle) and (b) all days with k >= 1.
    Placebo strikes: the target shifted by +/- half a grid step.
    """
    grid = SYMBOLS[sym]["grid"]
    half, band = grid / 2, grid / 5   # band: within 10 points for Nifty, 20 for Bank Nifty
    out = []
    p = panel.dropna(subset=["K_maxoi"]).copy()
    for tgt in ("K_maxoi", "K_maxpain", "K_maxgamma"):
        K = p[tgt]
        p["dist"] = (p["close"] - K).abs()
        p["hit"] = (p["dist"] <= band).astype(float)
        p["hit_placebo"] = 0.5 * (((p["close"] - K - half).abs() <= band).astype(float)
                                  + ((p["close"] - K + half).abs() <= band).astype(float))
        p["excess_hit"] = p["hit"] - p["hit_placebo"]
        p["move_toward"] = signed_move_toward(p["close_prev"], p["close"], K)
        p["dchange"] = change_in_distance(p["close_prev"], p["close"], K)
        p["dist_prev"] = (p["close_prev"] - K).abs()
        groups = {"expiry (k=0)": p[p["k"] == 0], "day before expiry (k=1)": p[p["k"] == 1],
                  "all non-expiry (k>=1)": p[p["k"] >= 1]}
        for gname, g in groups.items():
            out.append({"symbol": sym, "target": tgt, "group": gname, "n": len(g),
                        "mean_dist_prev_close_pts": g["dist_prev"].mean(), "mean_dist_close_pts": g["dist"].mean(),
                        "median_dist_close_pts": g["dist"].median(), "hit_rate": g["hit"].mean(),
                        "placebo_hit_rate": g["hit_placebo"].mean(), "excess_hit": g["excess_hit"].mean(),
                        "mean_move_toward_pts": g["move_toward"].mean(), "share_moved_toward": float((g["move_toward"] > 0).mean()),
                        "mean_change_in_dist_pts": g["dchange"].mean()})
        e = groups["expiry (k=0)"]
        # excess hit on expiry days against zero: sign-flip test on the per-day excess
        m, pe = sign_flip_test(e["excess_hit"], rng)
        lo, hi = boot_ci_mean(e["excess_hit"], rng)
        out.append({"symbol": sym, "target": tgt, "group": "test: expiry excess hit (strike minus placebo strikes) = 0",
                    "n": len(e), "estimate": m, "ci95_low": lo, "ci95_high": hi, "p_value": pe, "test": "sign-flip"})
        # paired by cycle: expiry day vs the day before, same contract
        e1 = groups["day before expiry (k=1)"].set_index("expiry")
        pair = e.set_index("expiry").join(e1[["excess_hit", "move_toward", "dist"]], rsuffix="_k1", how="inner")
        for col in ("excess_hit", "move_toward", "dist"):
            dd = pair[col] - pair[f"{col}_k1"]
            m, pp = sign_flip_test(dd, rng)
            lo, hi = boot_ci_mean(dd, rng)
            out.append({"symbol": sym, "target": tgt, "group": f"test: {col}, expiry minus day before (paired)",
                        "n": len(dd), "estimate": m, "ci95_low": lo, "ci95_high": hi, "p_value": pp, "test": "paired sign-flip"})
        o = groups["all non-expiry (k>=1)"]
        for col in ("excess_hit", "move_toward"):
            m, pp = perm_test_diff(e[col], o[col], rng)
            lo, hi = boot_ci_diff(e[col], o[col], rng)
            out.append({"symbol": sym, "target": tgt, "group": f"test: {col}, expiry minus all non-expiry",
                        "n": f"{len(e)} vs {len(o)}", "estimate": m, "ci95_low": lo, "ci95_high": hi, "p_value": pp,
                        "test": "permutation"})
    return pd.DataFrame(out)


def intraday_tests(panel: pd.DataFrame, sym: str, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T4: does the price move toward the nearest strike between 14:30 / 15:00 and the official close?

    The starting prices (14:30, 15:00) are in continuous trading in every
    regime. Measures inside the settlement window (15:29 bar, 15:00 to 15:29
    mean) are used only before the closing auction (3 Aug 2026), after which
    the minute index is frozen from about 15:15. Also checks how well each
    bar-based price approximates the official close.
    """
    grid = SYMBOLS[sym]["grid"]
    p = panel.dropna(subset=["p1429", "p1459"]).copy()
    pre = p[~p["closing_auction"] & (p["n30"] >= 29)]
    checks = []
    for era, q in (("before closing auction", pre), ("closing auction (from 3 Aug 2026)", p[p["closing_auction"]])):
        for name, col in (("mean of 15:00 to 15:29 bar closes", "mean30"), ("15:29 bar close (last bar)", "p1529"),
                          ("15:14 bar close (last continuous price under the auction)", "p1514"),
                          ("price at 15:00 (14:59 bar close)", "p1459")):
            err = (q[col] - q["close"]).abs().dropna()
            if len(err) == 0:
                continue
            checks.append({"symbol": sym, "era": era, "proxy": name, "n_days": len(err), "median_abs_err_pts": err.median(),
                           "p95_abs_err_pts": err.quantile(0.95), "share_within_5pts": float((err <= 5).mean())})
    out = []
    for start, col in (("14:30", "p1429"), ("15:00", "p1459")):
        K = nearest_grid(p[col], grid)
        p["mt"] = signed_move_toward(p[col], p["close"], K)
        p["dc"] = change_in_distance(p[col], p["close"], K)
        p["d0"] = np.abs(p[col] - K)
        e, o = p[p["is_expiry"]], p[~p["is_expiry"]]
        for g, x in (("expiry", e), ("non-expiry", o)):
            out.append({"symbol": sym, "start": start, "group": g, "n": len(x),
                        "mean_dist_at_start_pts": x["d0"].mean(),
                        "mean_move_toward_pts": x["mt"].mean(), "share_moved_toward": float((x["mt"] > 0).mean()),
                        "mean_change_in_dist_pts": x["dc"].mean(),
                        "mean_abs_move_pts": (x["close"] - x[col]).abs().mean()})
        for meas in ("mt", "dc"):
            m, pp = perm_test_diff(e[meas], o[meas], rng)
            lo, hi = boot_ci_diff(e[meas], o[meas], rng)
            out.append({"symbol": sym, "start": start,
                        "group": f"test: {'move toward' if meas == 'mt' else 'change in distance'}, expiry minus non-expiry",
                        "n": f"{len(e)} vs {len(o)}", "estimate": m, "ci95_low": lo, "ci95_high": hi, "p_value": pp})
        m, pp = sign_flip_test(e["mt"], rng)
        lo, hi = boot_ci_mean(e["mt"], rng)
        out.append({"symbol": sym, "start": start, "group": "test: move toward on expiry days = 0", "n": len(e),
                    "estimate": m, "ci95_low": lo, "ci95_high": hi, "p_value": pp})
    # distance of each price measure to its own nearest strike (where in the day does any pin appear?)
    for sample, q in (("all days with bars", p), ("before closing auction", pre)):
        for name, col in (("price at 14:30", "p1429"), ("price at 15:00", "p1459"), ("15:29 bar close", "p1529"),
                          ("mean of 15:00 to 15:29 bars", "mean30"), ("official close", "close")):
            if sample == "all days with bars" and col in ("p1529", "mean30"):
                continue
            qq = q.dropna(subset=[col])
            dd = dist_to_grid(qq[col], grid)
            ex = qq["is_expiry"].to_numpy()
            e, o = dd[ex], dd[~ex]
            m, pp = perm_test_diff(e, o, rng)
            lo, hi = boot_ci_diff(e, o, rng)
            out.append({"symbol": sym, "start": name, "group": f"test: distance to nearest strike, expiry minus non-expiry ({sample})",
                        "n": f"{len(e)} vs {len(o)}", "estimate": m, "ci95_low": lo, "ci95_high": hi, "p_value": pp,
                        "mean_expiry_pts": e.mean(), "mean_nonexpiry_pts": o.mean()})
    return pd.DataFrame(out), pd.DataFrame(checks)


def intraday_by_regime(panel: pd.DataFrame, sym: str, rng) -> pd.DataFrame:
    """Where in the afternoon does the expiry-day gap appear, regime by regime? (exploratory)"""
    grid = SYMBOLS[sym]["grid"]
    p = panel.dropna(subset=["p1429", "p1459"])
    out = []
    for reg, q in p.groupby("regime"):
        ex = q["is_expiry"].to_numpy()
        if ex.sum() < 5:
            continue
        for name, col in (("price at 14:30", "p1429"), ("price at 15:00", "p1459"), ("official close", "close")):
            dd = dist_to_grid(q[col], grid)
            m, pp = perm_test_diff(dd[ex], dd[~ex], rng)
            out.append({"symbol": sym, "regime": reg, "price": name, "n_expiry": int(ex.sum()), "n_nonexpiry": int((~ex).sum()),
                        "mean_expiry_pts": dd[ex].mean(), "mean_nonexpiry_pts": dd[~ex].mean(), "diff_pts": m, "p_perm": pp})
        K = nearest_grid(q["p1459"], grid)
        mt = signed_move_toward(q["p1459"], q["close"], K)
        m, pp = perm_test_diff(mt[ex], mt[~ex], rng)
        out.append({"symbol": sym, "regime": reg, "price": "move toward strike from 15:00 to official close",
                    "n_expiry": int(ex.sum()), "n_nonexpiry": int((~ex).sum()), "mean_expiry_pts": mt[ex].mean(),
                    "mean_nonexpiry_pts": mt[~ex].mean(), "diff_pts": m, "p_perm": pp})
    return pd.DataFrame(out)


def gamma_tests(panel: pd.DataFrame, sym: str, rng) -> pd.DataFrame:
    """T5: is pinning stronger when net dealer gamma near the strike is positive?

    C1 (primary for Nifty): sign of dealer gamma at the strike nearest to
       the close (t-1 OI, 30 minutes to expiry). Outcome: distance to that strike.
       Under no pinning the close is uniform within its strike cell whatever
       the sign, so both groups have mean grid/4.
    C2: sign of the day's net dealer GEX at the t-1 forward (whole chain,
       t-1 time to expiry). Fully predetermined. Outcome: distance to nearest strike.
    C3: gross (unsigned) concentration: share of chain OI at the nearest strike,
       above vs below its median. Sign-free.
    Each is computed on expiry days and, as placebo, on the day before expiry
    (k = 1, same contract) and on all non-expiry days; the difference in
    differences is reported with a permutation p-value.
    """
    grid = SYMBOLS[sym]["grid"]
    out = []
    p = panel.dropna(subset=["g_near_gross"]).copy()
    day_types = {"expiry": p[p["k"] == 0], "day before expiry": p[p["k"] == 1], "all non-expiry": p[p["k"] >= 1]}
    specs = []
    for rule in RULES:
        specs.append(("C1 sign at nearest strike", rule, f"g_near_{rule}"))
    for rule in RULES:
        specs.append(("C2 sign of day net GEX at forward", rule, f"g_day_{rule}"))
    specs.append(("C3 OI share at nearest strike above median", "none (unsigned)", "oi_share_near"))
    med_share = p.loc[p["k"] == 0, "oi_share_near"].median()
    for cond, rule, col in specs:
        res = {}
        for dt, g in day_types.items():
            g = g.dropna(subset=[col])
            if cond.startswith("C3"):
                pos = g[col] > med_share
            else:
                g = g[g[col] != 0]
                pos = g[col] > 0
            x, y = g.loc[pos, "d"], g.loc[~pos, "d"]
            row = {"symbol": sym, "conditioning": cond, "rule": rule, "day_type": dt,
                   "n_pos": len(x), "n_neg": len(y), "share_pos": float(pos.mean()) if len(g) else np.nan,
                   "mean_dist_pos_pts": x.mean() if len(x) else np.nan, "mean_dist_neg_pts": y.mean() if len(y) else np.nan}
            if len(x) >= 5 and len(y) >= 5:
                diff, pp = perm_test_diff(x, y, rng)
                lo, hi = boot_ci_diff(x, y, rng)
                rho, _ = stats.spearmanr(g[col], g["d"])
                row.update({"diff_pos_minus_neg_pts": diff, "ci95_low": lo, "ci95_high": hi, "p_perm": pp,
                            "spearman_signal_vs_dist": float(rho)})
                res[dt] = (x.to_numpy(), y.to_numpy())
            out.append(row)
        if "expiry" in res and "all non-expiry" in res:
            (xe, ye), (xo, yo) = res["expiry"], res["all non-expiry"]
            did = (xe.mean() - ye.mean()) - (xo.mean() - yo.mean())
            null = np.empty(N_PERM // 2)
            ze, zo = np.concatenate([xe, ye]), np.concatenate([xo, yo])
            for b in range(len(null)):
                pe_, po_ = rng.permutation(ze), rng.permutation(zo)
                null[b] = (pe_[:len(xe)].mean() - pe_[len(xe):].mean()) - (po_[:len(xo)].mean() - po_[len(xo):].mean())
            pdid = float((1 + np.sum(np.abs(null) >= abs(did) - 1e-12)) / (len(null) + 1))
            out.append({"symbol": sym, "conditioning": cond, "rule": rule, "day_type": "difference in differences (expiry minus non-expiry)",
                        "diff_pos_minus_neg_pts": did, "p_perm": pdid})
    df = pd.DataFrame(out)
    # Holm over the three rules for the primary C1 expiry-day test
    m = (df["conditioning"] == "C1 sign at nearest strike") & (df["day_type"] == "expiry") & df["p_perm"].notna()
    df.loc[m, "p_holm_3rules"] = holm(df.loc[m, "p_perm"].to_numpy())
    return df


def sign_agreement(panel: pd.DataFrame, sym: str) -> pd.DataFrame:
    e = panel[(panel["k"] == 0)].dropna(subset=["g_near_gross"])
    out = []
    for kind in ("near", "day"):
        s = pd.DataFrame({r: np.sign(e[f"g_{kind}_{r}"]) for r in RULES})
        for i, a in enumerate(RULES):
            for b in RULES[i + 1:]:
                out.append({"symbol": sym, "measure": f"g_{kind}", "rule_a": a, "rule_b": b,
                            "share_same_sign": float((s[a] == s[b]).mean())})
        for r in RULES:
            out.append({"symbol": sym, "measure": f"g_{kind}", "rule_a": r, "rule_b": "(share positive)",
                        "share_same_sign": float((s[r] > 0).mean())})
    return pd.DataFrame(out)


def weekday_swap(panel: pd.DataFrame, sym: str, rng) -> pd.DataFrame:
    """Robustness: the same weekday when it was and was not an expiry day."""
    out = []
    for wd, name in ((3, "Thursday"), (1, "Tuesday")):
        g = panel[panel["weekday"] == wd]
        e, o = g[g["is_expiry"]], g[~g["is_expiry"]]
        if len(e) < 5 or len(o) < 5:
            continue
        diff, pp = perm_test_diff(e["d"], o["d"], rng)
        lo, hi = boot_ci_diff(e["d"], o["d"], rng)
        out.append({"symbol": sym, "check": f"{name}s: expiry minus non-expiry", "n": f"{len(e)} vs {len(o)}",
                    "mean_expiry_pts": e["d"].mean(), "mean_nonexpiry_pts": o["d"].mean(), "estimate": diff,
                    "ci95_low": lo, "ci95_high": hi, "p_value": pp})
    if sym == "NIFTY":
        e, o = panel[panel["is_expiry"]], panel[~panel["is_expiry"]]
        diff, pp = perm_test_diff(e["d100"], o["d100"], rng)
        lo, hi = boot_ci_diff(e["d100"], o["d100"], rng)
        out.append({"symbol": sym, "check": "100-point grid (round hundreds): expiry minus non-expiry",
                    "n": f"{len(e)} vs {len(o)}", "mean_expiry_pts": e["d100"].mean(), "mean_nonexpiry_pts": o["d100"].mean(),
                    "estimate": diff, "ci95_low": lo, "ci95_high": hi, "p_value": pp})
        std = panel[panel["is_expiry"] & panel["weekday"].isin([1, 3])]
        diff, pp = perm_test_diff(std["d"], o["d"], rng)
        lo, hi = boot_ci_diff(std["d"], o["d"], rng)
        out.append({"symbol": sym, "check": "only scheduled-weekday expiries (no holiday shifts)", "n": f"{len(std)} vs {len(o)}",
                    "mean_expiry_pts": std["d"].mean(), "mean_nonexpiry_pts": o["d"].mean(), "estimate": diff,
                    "ci95_low": lo, "ci95_high": hi, "p_value": pp})
    return pd.DataFrame(out)


def power_check(designs, grid: float, rng, reps: int = 4000) -> pd.DataFrame:
    """T6: how large a pinning effect could the sample detect (two-sided 5%)?

    Two effect models on expiry days, non-expiry days uniform within their strike cell:
    'pinned share'   a share pi of expiry closes sit within grid/20 of a strike
    'pull'           every expiry close is pulled a fraction lam of the way to its strike
    ``designs``: list of (name, n1, n0); group 1 carries the effect, group 0 has none.
    Welch t-tests are used for speed (they agree closely with the permutation
    test at these sample sizes; checked in the stdout).
    """
    half = grid / 2
    out = []
    levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    for design, n1, n0 in designs:
        for model in ("pinned share", "pull"):
            for eff in levels:
                y = rng.uniform(0, half, (reps, n0))
                x = rng.uniform(0, half, (reps, n1))
                if model == "pinned share":
                    pinned = rng.random((reps, n1)) < eff
                    x = np.where(pinned, rng.uniform(0, grid / 20, (reps, n1)), x)
                else:
                    x = (1 - eff) * x
                t = stats.ttest_ind(x, y, axis=1, equal_var=False)
                out.append({"design": design, "grid_pts": grid, "n1": n1, "n0": n0, "model": model, "effect": eff,
                            "true_mean_shift_pts": float(np.mean(x) - half / 2), "power_5pct": float(np.mean(t.pvalue < 0.05))})
    return pd.DataFrame(out)


def mde_from_power(pw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (design, model), g in pw.groupby(["design", "model"]):
        g = g.sort_values("effect")
        above = g[g["power_5pct"] >= 0.8]
        if above.empty:
            rows.append({"design": design, "model": model, "grid_pts": g["grid_pts"].iloc[0], "n1": g["n1"].iloc[0],
                         "n0": g["n0"].iloc[0], "effect_80pct_power": np.nan, "mean_shift_pts_80pct": np.nan,
                         "note": "80% power not reached at effect 0.6"})
            continue
        j = above.index[0]
        i = g.index[g.index.get_loc(j) - 1] if g.index.get_loc(j) > 0 else j
        x0, x1, y0, y1 = g.loc[i, "effect"], g.loc[j, "effect"], g.loc[i, "power_5pct"], g.loc[j, "power_5pct"]
        eff = x1 if x1 == x0 else x0 + (0.8 - y0) * (x1 - x0) / (y1 - y0)
        s0, s1 = g.loc[i, "true_mean_shift_pts"], g.loc[j, "true_mean_shift_pts"]
        shift = s1 if x1 == x0 else s0 + (eff - x0) * (s1 - s0) / (x1 - x0)
        rows.append({"design": design, "model": model, "grid_pts": g["grid_pts"].iloc[0], "n1": g["n1"].iloc[0],
                     "n0": g["n0"].iloc[0], "effect_80pct_power": eff, "mean_shift_pts_80pct": shift})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figures(panels: dict, mins: dict, out: Path, halfyear: pd.DataFrame | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = panels["NIFTY"]
    grid = SYMBOLS["NIFTY"]["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    off = p["close"] - p["K_near"]
    bins = np.arange(-25, 25.01, 5)
    for ax, (name, m) in zip(axes, (("Nifty expiry days", p["is_expiry"]), ("Nifty non-expiry days", ~p["is_expiry"]))):
        x = off[m]
        ax.hist(x, bins=bins, weights=np.full(len(x), 1 / len(x)), color="#4C72B0" if "expiry days" == name[6:] else "#8C8C8C",
                edgecolor="white")
        ax.axhline(0.1, color="black", lw=1, ls="--", label="no clustering (uniform)")
        ax.set_title(f"{name} (n={len(x)})")
        ax.set_xlabel("official close minus nearest 50-point strike (points)")
        ax.set_ylabel("share of days")
        ax.set_ylim(0, 0.2)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("REAL DATA: where the Nifty official close lands inside its strike cell (Jan 2023 to Nov 2026)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_close_offset_histogram.png", dpi=130)
    plt.close(fig)

    m = mins.get("NIFTY")
    if m is not None:
        keep = p[~p["closing_auction"] & (p["n30"] >= 29)].index  # the index is frozen from 15:15 under the auction
        common = keep.intersection(m.index)
        mm = m.loc[common]
        dd = pd.DataFrame(dist_to_grid(mm.to_numpy(), grid), index=common)
        ex = p.loc[common, "is_expiry"].to_numpy()
        x = np.arange(dd.shape[1])
        labels = pd.date_range("2000-01-01 14:00", periods=dd.shape[1], freq="1min").strftime("%H:%M")
        fig, ax = plt.subplots(figsize=(9, 4))
        for flag, name, col in ((True, "expiry days", "#4C72B0"), (False, "non-expiry days", "#8C8C8C")):
            v = dd.to_numpy()[ex == flag]
            mu = np.nanmean(v, axis=0)
            se = np.nanstd(v, axis=0, ddof=1) / np.sqrt(np.isfinite(v).sum(axis=0))
            ax.plot(x, mu, color=col, label=f"{name}, 1-minute bar close (n={len(v)})")
            ax.fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.2)
            oc = p.loc[common, "d"].to_numpy()[ex == flag]
            ax.errorbar([len(x) + 3], [oc.mean()], yerr=[1.96 * oc.std(ddof=1) / np.sqrt(len(oc))], fmt="o", color=col)
        ax.axhline(grid / 4, color="black", ls="--", lw=1, label="no clustering (12.5 points)")
        ax.axvspan(SLOT_1500 - SLOT_1400, SLOT_1529 - SLOT_1400, color="orange", alpha=0.08, label="settlement window 15:00 to 15:30")
        ticks = list(range(0, len(x), 15)) + [len(x) + 3]
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks[:-1]] + ["official\nclose"])
        ax.set_ylabel("mean distance to nearest 50-point strike (points)")
        ax.set_title("REAL DATA: Nifty distance to the nearest strike in the last 90 minutes, Jan 2023 to Jul 2026 "
                     "(95% intervals)", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(out / "fig_intraday_distance.png", dpi=130)
        plt.close(fig)

    if halfyear is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, sym in zip(axes, SYMBOLS):
            h = halfyear[(halfyear["symbol"] == sym) & (halfyear["n_expiry"] >= 3)]
            x = np.arange(len(h))
            ax.errorbar(x, h["mean_dist_expiry_pts"], yerr=1.96 * h["se_expiry"], fmt="o", color="#4C72B0",
                        label="expiry days (95% interval)")
            ax.plot(x, h["mean_dist_nonexpiry_pts"], "s", color="#8C8C8C", label="non-expiry days")
            ax.axhline(SYMBOLS[sym]["grid"] / 4, color="black", ls="--", lw=1, label="no clustering")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{a}\n(n={n})" for a, n in zip(h["half_year"], h["n_expiry"])], fontsize=7)
            ax.set_ylabel(f"mean distance to nearest {SYMBOLS[sym]['grid']:g}-point strike")
            ax.set_title(f"{SYMBOLS[sym]['index']}", fontsize=10)
            ax.legend(fontsize=7)
        fig.suptitle("REAL DATA: expiry-day distance to the nearest strike by half-year (n = expiry days)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig_by_half_year.png", dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/pinning")
    ap.add_argument("--stdout", default="results/pinning_stdout.txt")
    a = ap.parse_args()
    t0 = time.time()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lines = [f"{LABEL}. Expiry-day pinning study. NSE bhavcopy, participant OI and official index closes; "
             "third-party 1-minute bars validated against NSE."]

    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    official, bhav, ptables = load_inputs(raw)
    log(f"Loaded {len(bhav['NIFTY'])} NIFTY and {len(bhav['BANKNIFTY'])} BANKNIFTY bhavcopy days, "
        f"{len(ptables)} participant files ({time.time() - t0:.0f}s)")

    panels, mins, checks, expiries = {}, {}, [], {}
    for sym, cfg in SYMBOLS.items():
        closes = minute_closes(raw, cfg["bars"]) if (raw / cfg["bars"]).exists() else None
        panel, ex, mn = build_day_panel(sym, official[sym], bhav[sym], ptables, closes)
        panels[sym], mins[sym], expiries[sym] = panel, mn, ex
        panel.to_csv(out / f"day_panel_{sym.lower()}.csv")
        # ---- data checks
        settle_ok = []
        for E in ex:
            c = bhav[sym][E]
            c = c[pd.to_datetime(c["expiry"]).dt.normalize() == E]
            settle_ok.append(bool(np.all(np.abs(c["settle"].to_numpy(float) - official[sym][E]) < 0.005)))
        e = panel[panel["is_expiry"]]
        checks += [
            {"symbol": sym, "check": "expiry days in bhavcopies (data.actual_expiries)", "value": len(ex)},
            {"symbol": sym, "check": "expiry days where every expiring option's settlement price equals the official close",
             "value": int(np.sum(settle_ok))},
            {"symbol": sym, "check": "trading days in panel", "value": len(panel)},
            {"symbol": sym, "check": "expiry days with a t-1 chain", "value": int(e["chain_date"].notna().sum())},
            {"symbol": sym, "check": "max stale days of t-1 chain (missing bhavcopy)", "value": float(panel["chain_stale_days"].max())},
            {"symbol": sym, "check": "days with a t-1 chain whose chain date is strictly before the day (no look-ahead)",
             "value": f"{int((panel['chain_date'] < panel.index).sum())} of {int(panel['chain_date'].notna().sum())}"},
            {"symbol": sym, "check": "days whose t-1 chain had to use a re-dated contract (holiday shift)",
             "value": int(panel["chain_redated"].fillna(False).astype(bool).sum())},
            {"symbol": sym, "check": "expiry days whose nearest strike to the close is listed in the t-1 chain",
             "value": int(e["K_near_listed"].sum())},
            {"symbol": sym, "check": "median share of options with no IV (no time value) on expiry-eve chains",
             "value": float(e["iv_nan_share"].median())},
        ]
        for (reg, wd), g in e.groupby(["regime", "weekday"]):
            checks.append({"symbol": sym, "check": f"expiry days, regime {reg}, weekday {wd} (0=Mon)", "value": len(g)})
        log(f"{sym}: {len(panel)} days, {len(ex)} expiry days; settlement = official close on {np.sum(settle_ok)}/{len(ex)}")

    # strike spacing near the money on expiry eve (is the grid really 50 / 100?)
    for sym, cfg in SYMBOLS.items():
        n_ok = n_all = 0
        for E in expiries[sym]:
            ch, cd, _ = chain_before(bhav[sym], pd.DatetimeIndex(sorted(bhav[sym])), E, E)
            if ch is None or not len(ch):
                continue
            k = np.sort(ch["strike"].unique())
            S = official[sym][cd]
            near = k[np.abs(k / S - 1) < 0.03]
            n_all += 1
            n_ok += int(len(near) > 1 and np.all(np.diff(near) == cfg["grid"]))
        checks.append({"symbol": sym, "check": f"expiry-eve chains whose strikes within 3% of spot are exactly {cfg['grid']:g} apart",
                       "value": f"{n_ok} of {n_all}"})

    t1 = pd.concat([clustering_tests(panels[s], s, rng) for s in SYMBOLS], ignore_index=True)
    t2 = pd.concat([profile_by_k(panels[s], s, rng) for s in SYMBOLS], ignore_index=True)
    t3 = pd.concat([target_tests(panels[s], s, rng) for s in SYMBOLS], ignore_index=True)
    t4s, t4c = zip(*[intraday_tests(panels[s], s, rng) for s in SYMBOLS])
    t4, t4c = pd.concat(t4s, ignore_index=True), pd.concat(t4c, ignore_index=True)
    t5 = pd.concat([gamma_tests(panels[s], s, rng) for s in SYMBOLS], ignore_index=True)
    t5b = pd.concat([sign_agreement(panels[s], s) for s in SYMBOLS], ignore_index=True)
    t7 = pd.concat([weekday_swap(panels[s], s, rng) for s in SYMBOLS], ignore_index=True)
    t1b = pd.concat([regime_heterogeneity(panels[s], s, rng) for s in SYMBOLS], ignore_index=True)
    t4b = pd.concat([intraday_by_regime(panels[s], s, rng) for s in SYMBOLS], ignore_index=True)
    t8 = pd.concat([by_half_year(panels[s], s) for s in SYMBOLS], ignore_index=True)

    # power, at the Nifty sample sizes (and C1 group sizes under the US rule as an example)
    pn = panels["NIFTY"]
    n_e, n_o = int(pn["is_expiry"].sum()), int((~pn["is_expiry"]).sum())
    c1 = t5[(t5["symbol"] == "NIFTY") & (t5["conditioning"] == "C1 sign at nearest strike") & (t5["day_type"] == "expiry")]
    designs = [("Nifty expiry vs non-expiry (P1)", n_e, n_o)]
    for _, r in c1.iterrows():
        designs.append((f"Nifty expiry days, positive vs negative gamma, {r['rule']} (P2)", int(r["n_pos"]), int(r["n_neg"])))
    for reg in ("multi_weekly", "single_weekly_thu", "single_weekly_tue"):
        q = pn[pn["regime"] == reg]
        designs.append((f"Nifty {reg} expiry vs non-expiry", int(q["is_expiry"].sum()), int((~q["is_expiry"]).sum())))
    pw = power_check(designs, SYMBOLS["NIFTY"]["grid"], rng)
    mde = mde_from_power(pw)
    # check: Welch t vs permutation p on the real P1 comparison
    e, o = pn[pn["is_expiry"]], pn[~pn["is_expiry"]]
    welch_p = float(stats.ttest_ind(e["d"], o["d"], equal_var=False).pvalue)

    checks_df = pd.DataFrame(checks)
    for name, df in (("data_checks", checks_df), ("t1_clustering", t1), ("t1_regime_heterogeneity", t1b),
                     ("t2_days_to_expiry_profile", t2),
                     ("t3_target_strikes", t3), ("t4_intraday_movement", t4), ("t4_close_proxy_check", t4c),
                     ("t4_intraday_by_regime", t4b), ("t8_by_half_year", t8),
                     ("t5_gamma_conditioning", t5), ("t5_sign_agreement", t5b), ("t6_power", pw), ("t6_power_mde", mde),
                     ("t7_robustness", t7),
                     ("design_log", pd.DataFrame(DESIGN_LOG, columns=["item", "change", "reason"]))):
        df.to_csv(out / f"{name}.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 4,
                               "display.max_colwidth", 70):
            log("")
            log(f"== {name}")
            log(df.to_string(index=False))
    log("")
    log(f"Welch t-test p for P1 (Nifty expiry vs non-expiry distance): {welch_p:.4f} "
        f"(permutation p: {t1.loc[(t1['symbol'] == 'NIFTY') & (t1['sample'] == 'all'), 'p_perm'].iloc[0]:.4f})")
    # ---- headline block (every number below is also in the CSV files)
    def t1row(sym, sample):
        return t1[(t1["symbol"] == sym) & (t1["sample"] == sample)].iloc[0]

    log("")
    log("== HEADLINE (REAL DATA)")
    r = t1row("NIFTY", "all")
    log(f"P1 Nifty, all {r['n_expiry']} expiry vs {r['n_nonexpiry']} other days: mean distance to nearest 50-pt strike "
        f"{r['mean_dist_expiry_pts']:.2f} vs {r['mean_dist_nonexpiry_pts']:.2f} pts (no clustering: 12.5); difference "
        f"{r['diff_pts']:.2f} pts [{r['diff_ci95_low']:.2f}, {r['diff_ci95_high']:.2f}], {r['diff_bp']:.2f} bp; "
        f"permutation p = {r['p_perm']:.4f}")
    for reg in ("multi_weekly", "single_weekly_thu", "single_weekly_tue"):
        r = t1row("NIFTY", reg)
        log(f"   {reg}: {r['mean_dist_expiry_pts']:.2f} vs {r['mean_dist_nonexpiry_pts']:.2f} pts, diff {r['diff_pts']:.2f} "
            f"[{r['diff_ci95_low']:.2f}, {r['diff_ci95_high']:.2f}], p = {r['p_perm']:.4f} (n = {r['n_expiry']} expiry days)")
    h = t1b[(t1b["symbol"] == "NIFTY") & (t1b["regime_a"] == "single_weekly_thu") & (t1b["regime_b"] == "single_weekly_tue")]
    if len(h):
        h = h.iloc[0]
        log(f"   Thursday single-weekly vs Tuesday regime, difference in expiry gap: {h['gap_a_minus_gap_b_pts']:.2f} pts, "
            f"p = {h['p_perm']:.4f}")
    c1n = t5[(t5["symbol"] == "NIFTY") & (t5["conditioning"] == "C1 sign at nearest strike") & (t5["day_type"] == "expiry")]
    for _, r in c1n.iterrows():
        log(f"P2 Nifty expiry days, {r['rule']}: positive {r['mean_dist_pos_pts']:.2f} (n={int(r['n_pos'])}) vs negative "
            f"{r['mean_dist_neg_pts']:.2f} (n={int(r['n_neg'])}) pts, diff {r['diff_pos_minus_neg_pts']:.2f}, p = {r['p_perm']:.3f}, "
            f"Holm p = {r['p_holm_3rules']:.3f}")
    r = t1row("BANKNIFTY", "weekly_options")
    log(f"Bank Nifty weekly-options era: {r['mean_dist_expiry_pts']:.2f} vs {r['mean_dist_nonexpiry_pts']:.2f} pts "
        f"(no clustering: 25), diff {r['diff_pts']:.2f} [{r['diff_ci95_low']:.2f}, {r['diff_ci95_high']:.2f}], "
        f"{r['diff_bp']:.2f} bp, p = {r['p_perm']:.4f} (n = {r['n_expiry']})")
    m = mde[mde["design"] == "Nifty expiry vs non-expiry (P1)"]
    for _, r in m.iterrows():
        log(f"Power (P1, {r['model']}): 80% power at effect {r['effect_80pct_power']:.3f}, i.e. a mean shift of "
            f"{r['mean_shift_pts_80pct']:.2f} pts")
    figures(panels, mins, out, t8)
    log(f"Figures written to {out}. Total run time {time.time() - t0:.0f}s.")
    Path(a.stdout).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
