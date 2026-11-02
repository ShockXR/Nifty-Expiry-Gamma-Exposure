"""Intraday dealer gamma at the current index level and next-minutes volatility (REAL DATA).

Question
--------
On a Nifty expiry day the open interest (OI) of the expiring contract is fixed
from the previous close, but the index moves and the clock runs down. The
gamma that option dealers must hedge therefore changes minute by minute in a
way that can be computed in real time from information known at that minute:
yesterday's OI and implied vols, today's index level so far, and the clock.
Does the volatility of the next 15 (or 30) minutes respond to this "gamma at
spot", holding fixed everything that is common to a day (day fixed effects)
and the normal intraday pattern (minute-of-day fixed effects)?

Unsigned (gross) gamma needs no assumption about who holds the options. If
dealers are net long gamma, more gamma at spot should mean calmer markets
(negative coefficient); if net short, more volatile markets (positive). So the
sign of the gross coefficient is itself evidence on the sign of dealer
positions, which NSE does not publish.

Design (all quantities REAL DATA, read from data/raw)
-----------------------------------------------------
* Chain: for day t with next expiry E (E = t on expiry days), the contract
  expiring on E from the bhavcopy of the last trading day strictly before t
  (pinning_study.chain_before). OI in shares. Implied vols from that day's
  settlement prices (Black-76, pinning_study.prepare_chain), held fixed per
  strike during day t (sticky strike).
* Anchors: every 5 minutes from 09:30 to 15:00 IST. The index level at an
  anchor is the close of the last completed 1-minute bar (bar starting one
  minute earlier), so nothing after the anchor is used.
* Time to expiry at an anchor, in trading years:
  (max(minutes to 15:30, floor) + 375 * full sessions left) / (375 * 252).
* Gross GEX at spot: sum over strikes and sides of OI * gamma * S^2 * 0.01
  (rupees per 1% move). Concentration: share of it from strikes within
  +/- 0.5% of spot. Signed: dealer weights per side under the US convention,
  NSE "Pro" participant weights and the mirror of NSE "Client" weights
  (niftygex.gex), from the previous day's participant file.
* Outcome: realized variance (sum of squared 1-minute log returns) over the
  next 15 and 30 minutes. From 3 Aug 2026 (closing auction, index frozen in
  the bars from 15:15) windows that reach past 15:14 are dropped.
* Regression: log RV(next 15) on log gross GEX at spot (or a signed version),
  with day and minute-of-day fixed effects, lagged 15- and 60-minute log RV
  and the absolute distance to the nearest 50-point strike. Standard errors
  clustered by day; checks with two-way (day, minute) clustering and
  Driscoll-Kraay (day as the time unit).
* Placebos: (a) OI profile shifted by half a strike (25 points); (b) the OI
  profile of another, randomly chosen day of the same type, re-centred on
  today's at-the-money strike; (c) strike-level OI randomly permuted across
  strikes within +/- 5% of the previous forward.
* Out of sample: HAR-style model of the last hour's variance on expiry days,
  with and without gross GEX at spot at 14:30; expanding window, refit every
  21 trading days, first 150 trading days for training only; QLIKE and a
  Diebold-Mariano test.

Follow-ups (pre-registered in design_log.csv on 2026-11-02 21:36 UTC)
-------------------------------------------------------------------
* Final half hour of Nifty expiry days before the closing auction: 1-minute
  anchors 15:00 to 15:24, next-5-minute RV, gamma with the true time to 15:30
  (1-minute floor; 2 and 5 as checks) and with the averaged settlement,
  concentration, distance controls and the move since the previous close.
* Bank Nifty replication of the concentration lead on expiry days with weekly
  options (Jan 2023 to Nov 2024), same specification as the Nifty test.

    OMP_NUM_THREADS=1 python experiments/intraday_gamma.py

Writes results/intraday_gamma/*.csv, *.png and results/intraday_gamma_stdout.txt.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from niftygex import data, evaluation, gex, greeks  # noqa: E402
from niftygex.expiry_calendar import ExpiryRules  # noqa: E402

_PS_PATH = Path(__file__).resolve().parent / "pinning_study.py"
_spec = importlib.util.spec_from_file_location("pinning_study", _PS_PATH)
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

LABEL = "REAL DATA"
RATE = 0.065                  # risk-free rate for Black-76, as elsewhere in the project
TRADING_DAYS = 252.0
SESSION = 375                 # 1-minute bars 09:15 to 15:29
GRID = 50.0                   # Nifty strike interval near the money
CLOSE_SLOT = 375              # 15:30 = 09:15 + 375 minutes
ASIAN_CLOSE_SLOT = 355        # 15:10: the official close averages 15:00 to 15:30 (variance-equivalent end)
ANCHORS = np.arange(15, 346, 5)   # bar slots of 09:30, 09:35, ..., 15:00
FLOOR_MAIN = 5                # minutes; never binds on this grid (the last anchor has 30 minutes left)
FLOOR_VARIANTS = (60, 120)
STRIKE_BAND = 0.10            # keep strikes within +/- 10% of the previous forward
CONC_BAND = 0.005             # concentration: strikes within +/- 0.5% of spot
PERM_BAND = 0.05              # placebo (c): permute OI across strikes within +/- 5% of the previous forward
SHIFT = GRID / 2              # placebo (a)
CAS_START = pd.Timestamp("2026-08-03")   # NSE closing auction; bars frozen from 15:15 (see pinning study)
CAS_LAST_SLOT = 359           # last clean bar (15:14) on closing-auction days
DAILY_LAST_SLOT = 359         # daily RV for the HAR model: 09:15 to 15:14 on every day (comparable across CAS)
SLOT_1430 = 315
OOS_TRAIN_DAYS = 150
OOS_REFIT = 21
DAY_TYPES = ("expiry", "day_before", "ordinary")
RULES = ("us_convention", "participant_pro", "client_mirror")
CONTROLS = ["lag15", "lag60", "dist_near"]
SEED = 20261103
N_DRAWS = 200
# follow-up 1: final half hour of expiry days before the closing auction
FH_ANCHORS = np.arange(345, 370)  # bar slots of 15:00, 15:01, ..., 15:24
FH_FLOOR = 1
FH_FLOOR_VARIANTS = (2, 5)
FH_HORIZON = 5
SETTLE_FIRST_SLOT = 345           # the official close averages the 30 one-minute closes of 15:00 to 15:29 bars
SETTLE_WINDOW = 30
FH_CONTROLS = ["lag5", "lag15", "lag60", "dist_near", "dist_maxoi_pct", "ret_prev_pct", "absret_prev_pct"]
FH_NAME = "final half hour (15:00 to 15:24), expiry days before the closing auction"
# follow-up 2: Bank Nifty replication of the concentration lead
BN_START = pd.Timestamp("2023-01-03")
BN_WEEKLY_END = pd.Timestamp("2024-11-20")   # Bank Nifty weekly options stopped (see pinning study)
BN_GRID = 100.0
BN_NAME = "Bank Nifty expiry days with weekly options (Jan 2023 to Nov 2024)"

DESIGN_LOG = [
    ("forward inside the day", "F = S * exp(r * T) with r = 6.5%, not the previous day's parity forward",
     "Parity forwards from settlement prices scatter by 10 to 20 points around the theoretical basis on "
     "short contracts; on expiry day the theoretical basis is under 6 points and goes to zero."),
    ("time floor", "Main floor 5 minutes; sensitivity with 60 and 120 minutes and with an 'Asian' clock "
     "that ends at 15:10", "The last anchor is 15:00 (30 minutes left), so a 5-minute floor never binds. The "
     "official close is a 15:00 to 15:30 average, whose variance equals that of a single price at about 15:10."),
    ("lagged 60-minute RV", "Before 10:15 the 60-minute lag is the RV since 09:15 scaled to 60 minutes",
     "Spilling into the previous afternoon would mix in the overnight gap and, from 3 Aug 2026, the "
     "closing-auction jump."),
    ("closing auction", "From 3 Aug 2026 outcome windows must end by 15:14",
     "The 1-minute index is frozen from 15:15 to about 15:28 and then jumps to the auction close."),
    ("signed regressors", "Net GEX at spot divided by gross GEX at the previous close (a day constant known "
     "before the open)", "A signed quantity has no logarithm; dividing by a pre-open day constant makes days "
     "with very different OI comparable without using any information from the day itself."),
    ("placebo (b) and (c)", "Only OI changes; today's implied vols are kept (interpolated at the placebo strikes)",
     "The placebo is meant to break the link to the real OI profile, not to change the volatility smile."),
    ("placebo (b) centring", "Another day's profile is moved by a whole number of 50-point strikes so that its "
     "previous-close forward sits on today's", "Keeping strikes on the real 50-point grid means a pure "
     "round-number effect would survive the placebo, while the real OI shape is lost."),
    ("leverage control (added after the first run)", "Robustness specs that add the signed and the absolute index "
     "move since the previous close", "Under the US convention net GEX at spot is call gamma minus put gamma, which "
     "rises mechanically when the index rises (calls sit above, puts below); volatility falls after rises (the "
     "leverage effect). Without this control a signed coefficient can reflect the leverage effect, not hedging. "
     "The headline specification was kept exactly as pre-specified."),
    ("expiry afternoon placebos (added after the first run)", "Placebos (a), (b) and (c) are also run on the expiry "
     "afternoon subsample (13:00 to 15:00)", "The first run showed a positive secondary coefficient there; a "
     "placebo is the natural check of whether it is specific to the real OI profile."),
    # ---- follow-up studies, written down on 2026-11-02 21:36 UTC before any follow-up code was written or run
    ("PRE-REGISTERED 2026-11-02 21:36 UTC: final half hour, sample and timing",
     "Nifty expiry days before 3 Aug 2026 (before the closing auction, so 15:00 to 15:29 trades normally). "
     "1-minute anchors 15:00 to 15:24 (25 per day); index level = close of the last completed bar. Outcome: log "
     "realized variance of the next 5 minutes (5 squared 1-minute log returns from the anchor minute on).",
     "Strike-level gamma only becomes sharp after 15:00 (gamma_width_expiry_day.csv), so this is where hedging "
     "should matter most."),
    ("PRE-REGISTERED 2026-11-02 21:36 UTC: final half hour, primary tests FH1 and FH2",
     "FH1: log gross GEX at spot (Black-76, minutes to 15:30, 1-minute floor). FH2: concentration (share of gross "
     "GEX from strikes within +/-0.5% of spot), same clock. Controls: log RV of the previous 5, 15 and 60 minutes, "
     "distance to the nearest 50-point strike, distance to the max-OI strike, signed and absolute move since the "
     "previous close. Day and minute-of-day fixed effects, clustering by day (two-way and Driscoll-Kraay "
     "reported). Two-sided p. FH1 has no predicted sign (negative = dealers net long, positive = net short); FH2 is "
     "predicted negative from the Nifty concentration lead.",
     "Same design as the main study, plus the leverage control that the main study showed is needed."),
    ("PRE-REGISTERED 2026-11-02 21:36 UTC: final half hour, secondary specifications",
     "Floors of 2 and 5 minutes (they cannot bind on the 15:30 clock, since 6 minutes remain at 15:24, and are "
     "reported anyway). Averaged-settlement gamma: the official close is the mean of the 30 one-minute closes of "
     "15:00 to 15:29; at an anchor with n future bars a share w = n/30 of that average is still open, so each "
     "option is w times an option on the future average with strike (K - (1 - w) x mean of the closes already "
     "printed in the window) / w and variance-equivalent time (n + 1)(2n + 1) / (6n) minutes; floors 1, 2 and 5. "
     "Concentration within +/-0.25%. Main-study controls only (lag 15, lag 60, distance to nearest strike).",
     "The averaged settlement is how NSE settles Nifty options before the closing auction; it makes gamma far "
     "sharper and moves it toward where the average is heading."),
    ("PRE-REGISTERED 2026-11-02 21:36 UTC: final half hour, placebos",
     "(a) OI profile shifted +25 and -25 points; (b) another final-half-hour sample day's OI profile re-centred on "
     "today's at-the-money strike; (c) strike-level OI permuted within +/-5%; 200 draws each; for FH1 and FH2.",
     "Same placebos as the main study."),
    ("PRE-REGISTERED 2026-11-02 21:36 UTC: Bank Nifty replication of the concentration lead (BN1)",
     "Bank Nifty expiry days with weekly options: expiry dates read from BANKNIFTY contracts in the same "
     "bhavcopies, 3 Jan 2023 to 19 Nov 2024 (weekly options stopped on 20 Nov 2024). Chain: the expiring BANKNIFTY "
     "contract from the previous trading day's bhavcopy; bars data/raw/banknifty_1min.csv; official closes "
     "'Nifty Bank' in index_daily.csv; 100-point strikes. Specification identical to the Nifty secondary test: "
     "log RV of the next 15 minutes on concentration (share within +/-0.5% of spot), controls log RV of the previous "
     "15 and 60 minutes and distance to the nearest (100-point) strike, day and minute-of-day fixed effects, anchors "
     "every 5 minutes 09:30 to 15:00, 5-minute floor, clustering by day. Predicted sign negative (Nifty -0.386). "
     "Primary p two-sided; the one-sided p (coefficient below zero) is also reported.",
     "An independent market with its own option book is the cleanest check of a lead found among many tests."),
    ("PRE-REGISTERED 2026-11-02 21:36 UTC: Bank Nifty secondary, and the follow-up family",
     "Secondary: the same with the leverage control; log gross GEX at spot; placebos (a) shifted +50 and -50 points "
     "(half a Bank Nifty strike), (b) and (c) as for Nifty, 200 draws. Multiple testing: Holm over the three "
     "follow-up primary tests FH1, FH2 and BN1.",
     "Keeps the follow-up family small and fixed in advance."),
]
PLACEBO_SAMPLES = {"expiry": ("expiry", 0), "day_before": ("day_before", 0), "ordinary": ("ordinary", 0),
                   "expiry afternoon (13:00 to 15:00)": ("expiry", 225)}


# ---------------------------------------------------------------------------
# Building blocks (unit-tested in tests/test_intraday_gamma.py)
# ---------------------------------------------------------------------------


def slot_clock(slot) -> str:
    """Clock time (IST) of a bar slot: slot 0 is 09:15."""
    m = 9 * 60 + 15 + int(slot)
    return f"{m // 60:02d}:{m % 60:02d}"


def time_to_expiry(slot, full_sessions, floor_minutes: float = FLOOR_MAIN, close_slot: int = CLOSE_SLOT):
    """Trading-year time to expiry at an anchor.

    (max(minutes from the anchor to the close, floor) + 375 * full sessions left) / (375 * 252).
    ``full_sessions`` counts the trading sessions after today up to and including the expiry day
    (0 on expiry day, 1 on the day before).
    """
    slot = np.asarray(slot, float)
    mins = np.maximum(close_slot - slot, floor_minutes)
    return (mins + SESSION * np.asarray(full_sessions, float)) / (SESSION * TRADING_DAYS)


def spot_at_anchor(closes, slot):
    """Index level known at an anchor: the close of the last completed bar (slot - 1).

    Bars are stamped with their start time, so the bar of slot s closes at the
    anchor time 09:15 + s minutes. Only bars before the anchor are used.
    """
    closes = np.asarray(closes, float)
    slot = np.asarray(slot, int)
    if np.any(slot < 1):
        raise ValueError("anchors must be after the first bar")
    return closes[..., slot - 1]


def minute_returns(open0, closes):
    """1-minute log returns on a (days x minutes) grid; the first uses the bar's own open (no overnight gap)."""
    closes = np.atleast_2d(np.asarray(closes, float))
    open0 = np.atleast_1d(np.asarray(open0, float))
    prev = np.concatenate([open0[:, None], closes[:, :-1]], axis=1)
    return np.log(closes / prev)


def forward_rv(returns, slot, h: int, last_ok_slot: int = SESSION - 1):
    """Realized variance over bars slot .. slot+h-1 (the h minutes after the anchor); NaN if past last_ok_slot."""
    r = np.asarray(returns, float)
    if slot + h - 1 > min(last_ok_slot, r.size - 1):
        return np.nan
    return float(np.sum(r[slot:slot + h] ** 2))


def backward_rv(returns, slot, h: int):
    """Realized variance over the h minutes before the anchor, scaled up if fewer than h minutes exist today."""
    r = np.asarray(returns, float)
    if slot >= h:
        return float(np.sum(r[slot - h:slot] ** 2))
    return float(np.sum(r[:slot] ** 2) * h / slot)


def gamma_at_spot(S, T, K, oi_call, oi_put, iv_call, iv_put, carry: float = RATE, r: float = RATE,
                  band: float = CONC_BAND) -> dict:
    """Gross gamma exposure at index level(s) S with time(s) to expiry T.

    For each level: sum over strikes of OI * Gamma(F, K, T, iv) * S^2 * 0.01
    for calls and for puts (OI in shares, so rupees per 1% move), with
    F = S * exp(carry * T) and the package's Black-76 spot gamma.
    Returns arrays ``call``, ``put``, ``gross`` (= call + put) and ``conc``
    (share of gross from strikes within +/- band of S).
    """
    S = np.atleast_1d(np.asarray(S, float))[:, None]
    T = np.broadcast_to(np.atleast_1d(np.asarray(T, float)), (S.shape[0],))[:, None]
    K = np.asarray(K, float)[None, :]
    F = S * np.exp(carry * T)
    gc = greeks.spot_gamma(F, K, T, r, np.asarray(iv_call, float)[None, :], carry)
    gp = greeks.spot_gamma(F, K, T, r, np.asarray(iv_put, float)[None, :], carry)
    scale = S ** 2 * 0.01
    ec = gc * np.asarray(oi_call, float)[None, :] * scale
    ep = gp * np.asarray(oi_put, float)[None, :] * scale
    call, put = ec.sum(axis=1), ep.sum(axis=1)
    gross = call + put
    near = np.abs(K / S - 1.0) <= band
    with np.errstate(invalid="ignore", divide="ignore"):
        conc = ((ec + ep) * near).sum(axis=1) / gross
    return {"call": call, "put": put, "gross": gross, "conc": conc}


def settlement_gamma_at_spot(S, n_future, past_mean, K, oi_call, oi_put, iv_call, iv_put, floor: float = FH_FLOOR,
                             window: int = SETTLE_WINDOW, carry: float = RATE, r: float = RATE) -> dict:
    """Gross gamma exposure at spot inside the averaged settlement window (final half hour of expiry day).

    The option pays max(A - K, 0) (or the put) on A, the mean of ``window`` one-minute closes. At an anchor
    with ``n_future`` of them still to come, a share w = n/window of A is open and the rest is the mean of the
    closes already printed (``past_mean``). So the option is w times an option on the mean of the future closes
    with strike K* = (K - (1 - w) * past_mean) / w, whose variance equals that of one price after
    (n + 1)(2n + 1) / (6n) minutes (discrete average of a random walk), floored at ``floor`` minutes.
    Its gamma with respect to the index is w * Gamma(S, K*, T_eff). Strikes with K* <= 0 are certain to
    finish in the money and carry no gamma.
    """
    S = np.atleast_1d(np.asarray(S, float))[:, None]
    n = np.broadcast_to(np.atleast_1d(np.asarray(n_future, float)), (S.shape[0],))[:, None]
    pm = np.broadcast_to(np.atleast_1d(np.asarray(past_mean, float)), (S.shape[0],))[:, None]
    K = np.asarray(K, float)[None, :]
    w = n / window
    Kstar = (K - np.where(w < 1.0, (1.0 - w) * pm, 0.0)) / w   # nothing printed yet when w = 1
    t_min = np.maximum((n + 1.0) * (2.0 * n + 1.0) / (6.0 * n), floor)
    T = t_min / (SESSION * TRADING_DAYS)
    ok = Kstar > 0
    Ks = np.where(ok, Kstar, 1.0)
    F = S * np.exp(carry * T)
    gc = np.where(ok, w * greeks.spot_gamma(F, Ks, T, r, np.asarray(iv_call, float)[None, :], carry), 0.0)
    gp = np.where(ok, w * greeks.spot_gamma(F, Ks, T, r, np.asarray(iv_put, float)[None, :], carry), 0.0)
    scale = S ** 2 * 0.01
    call = (gc * np.asarray(oi_call, float)[None, :] * scale).sum(axis=1)
    put = (gp * np.asarray(oi_put, float)[None, :] * scale).sum(axis=1)
    return {"call": call, "put": put, "gross": call + put, "T_eff_minutes": t_min[:, 0], "w": w[:, 0]}


def net_gex(call, put, w_call: float, w_put: float):
    """Signed (dealer) GEX from the call and put parts: w_call * call + w_put * put."""
    return w_call * np.asarray(call, float) + w_put * np.asarray(put, float)


def shift_profile(K, offset: float = SHIFT):
    """Placebo (a): move every strike of the OI profile by ``offset`` index points."""
    return np.asarray(K, float) + offset


def recentre_shift(F_other: float, F_target: float, grid: float = GRID) -> float:
    """Placebo (b): whole-strike shift that puts another day's at-the-money point on today's."""
    return float(grid * np.round((F_target - F_other) / grid))


def permute_strike_oi(K, oi_call, oi_put, centre: float, band: float, rng):
    """Placebo (c): permute the (call OI, put OI) pairs across the strikes within +/- band of centre.

    Strikes outside the band keep their OI; strikes and implied vols stay where they are.
    """
    K = np.asarray(K, float)
    oc, op = np.asarray(oi_call, float).copy(), np.asarray(oi_put, float).copy()
    idx = np.where(np.abs(K / centre - 1.0) <= band)[0]
    perm = idx[rng.permutation(idx.size)]
    oc[idx], op[idx] = oc[perm], op[perm]
    return oc, op


def holm(pvals):
    return ps.holm(pvals)


# ---------------------------------------------------------------------------
# Fixed-effects regression with clustered standard errors
# ---------------------------------------------------------------------------


def demean_two_way(X, g1, g2, tol: float = 1e-11, max_iter: int = 2000):
    """Remove two sets of fixed effects (integer codes g1, g2) by alternating projections."""
    X = np.array(X, float, copy=True)
    if X.ndim == 1:
        X = X[:, None]
    n1, n2 = np.bincount(g1), np.bincount(g2)
    scale = np.maximum(np.abs(X).max(axis=0), 1e-300)
    for _ in range(max_iter):
        m1 = np.column_stack([np.bincount(g1, X[:, j], minlength=n1.size) for j in range(X.shape[1])]) / n1[:, None]
        X -= m1[g1]
        m2 = np.column_stack([np.bincount(g2, X[:, j], minlength=n2.size) for j in range(X.shape[1])]) / n2[:, None]
        X -= m2[g2]
        if np.max(np.abs(m2) / scale) < tol:
            break
    return X


def _group_scores(Xu, groups, G):
    out = np.zeros((G, Xu.shape[1]))
    np.add.at(out, groups, Xu)
    return out


def fe_ols(df: pd.DataFrame, y: str, xs: list, day: str = "day_code", minute: str = "slot",
           dk_lag: int | None = None) -> dict:
    """OLS of y on xs with day and minute-of-day fixed effects.

    Returns coefficients with standard errors clustered by day (main), two-way
    clustered by day and minute-of-day (Cameron, Gelbach and Miller 2011), and
    Driscoll-Kraay (day as the time unit, Bartlett kernel). Day effects are
    nested in the day clusters and are not counted in the small-sample factor.
    """
    d = df.dropna(subset=[y] + xs)
    g1 = pd.factorize(d[day], sort=True)[0]
    g2 = pd.factorize(d[minute], sort=True)[0]
    Z = demean_two_way(d[[y] + xs].to_numpy(float), g1, g2)
    yy, X = Z[:, 0], Z[:, 1:]
    n, k = X.shape
    G1, G2 = g1.max() + 1, g2.max() + 1
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ yy
    u = yy - X @ beta
    k_eff = k + (G2 - 1)
    Xu = X * u[:, None]
    S1 = _group_scores(Xu, g1, G1)
    S2 = _group_scores(Xu, g2, G2)
    c1 = G1 / (G1 - 1) * (n - 1) / (n - k_eff)
    c2 = G2 / (G2 - 1) * (n - 1) / (n - k_eff)
    V1 = c1 * XtX_inv @ (S1.T @ S1) @ XtX_inv
    V2 = c2 * XtX_inv @ (S2.T @ S2) @ XtX_inv
    V0 = (n - 1) / (n - k_eff) * XtX_inv @ (Xu.T @ Xu) @ XtX_inv
    V12 = V1 + V2 - V0
    if np.any(np.diag(V12) <= 0):
        V12 = np.maximum(V1, V2)
    L = int(np.floor(4 * (G1 / 100) ** (2 / 9))) if dk_lag is None else dk_lag
    meat = S1.T @ S1
    for lag in range(1, L + 1):
        w = 1 - lag / (L + 1)
        Gl = S1[lag:].T @ S1[:-lag]
        meat += w * (Gl + Gl.T)
    Vdk = c1 * XtX_inv @ meat @ XtX_inv
    sst = float(yy @ yy)
    out = {"n_obs": int(n), "n_days": int(G1), "n_minutes": int(G2), "within_r2": 1 - float(u @ u) / sst if sst > 0 else np.nan,
           "dk_lag": L, "coef": {}}
    tq = stats.t.ppf(0.975, G1 - 1)
    for j, name in enumerate(xs):
        b = float(beta[j])
        se1, se12, sedk = (float(np.sqrt(max(V[j, j], 0.0))) for V in (V1, V12, Vdk))
        t1 = b / se1 if se1 > 0 else np.nan
        out["coef"][name] = {
            "coef": b, "se": se1, "ci_low": b - tq * se1, "ci_high": b + tq * se1, "t": t1,
            "p": float(2 * stats.t.sf(abs(t1), G1 - 1)),
            "se_twoway": se12, "p_twoway": float(2 * stats.t.sf(abs(b / se12), min(G1, G2) - 1)) if se12 > 0 else np.nan,
            "se_dk": sedk, "p_dk": float(2 * stats.t.sf(abs(b / sedk), G1 - 1)) if sedk > 0 else np.nan,
            "within_sd_x": float(np.std(X[:, j])),
        }
    return out


def fast_coef(y_dm, C_dm, x_dm, groups, G):
    """Coefficient and day-clustered t of x in a regression on [x, C] (all inputs already demeaned)."""
    X = np.column_stack([x_dm, C_dm])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y_dm
    u = y_dm - X @ beta
    S = _group_scores(X * u[:, None], groups, G)
    V = XtX_inv @ (S.T @ S) @ XtX_inv * G / (G - 1)
    return float(beta[0]), float(beta[0] / np.sqrt(V[0, 0]))


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def _load_bhav(f):
    df = data.load_bhavcopy(f, "NIFTY")
    if not len(df):
        return None, None
    return pd.Timestamp(df["date"].iloc[0]).normalize(), df


def _load_bhav_sym(args):
    f, sym = args
    df = data.load_bhavcopy(f, sym)
    if not len(df):
        return None, None
    return pd.Timestamp(df["date"].iloc[0]).normalize(), df


def load_banknifty(raw: Path, workers: int):
    """Bank Nifty inputs: official closes ('Nifty Bank'), BANKNIFTY option chains and 1-minute bars."""
    idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
    close = idx[idx["index"].str.lower() == "nifty bank"].set_index("date")["close"].sort_index()
    files = data.bhavcopy_files(raw / "bhavcopy")
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_load_bhav_sym, [(f, "BANKNIFTY") for f in files], chunksize=16)
    bhav = {d: df for d, df in res if d is not None}
    bars = data.load_minute_bars(raw / "banknifty_1min.csv")
    bars = bars[bars.index >= pd.Timestamp("2022-10-01")]
    return close, bhav, data.to_session_grid(bars)


def load_raw(raw: Path, workers: int):
    idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
    nifty = idx[idx["index"].str.lower() == "nifty 50"].set_index("date")["close"].sort_index()
    vix = idx[idx["index"].str.lower() == "india vix"].set_index("date")["close"].sort_index()
    files = data.bhavcopy_files(raw / "bhavcopy")
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_load_bhav, files, chunksize=16)
    bhav = {d: df for d, df in res if d is not None}
    ptables = {}
    for f in sorted((raw / "participant_oi").glob("*.csv")):
        try:
            d = pd.to_datetime(f.stem[-8:], format="%d%m%Y")
        except ValueError:
            continue
        ptables[d] = data.load_participant_oi(f)
    bars = data.load_minute_bars(raw / "nifty_1min.csv")
    bars = bars[bars.index >= pd.Timestamp("2022-10-01")]
    grid = data.to_session_grid(bars)
    return nifty, vix, bhav, ptables, grid


def chain_profile(c: pd.DataFrame, F_prev: float) -> pd.DataFrame:
    """Strike-level profile (one row per strike): call/put OI and implied vols, strikes within the band."""
    c = c[np.abs(c["strike"] / F_prev - 1) <= STRIKE_BAND]
    calls = c[c["is_call"]].set_index("strike")
    puts = c[~c["is_call"]].set_index("strike")
    K = np.union1d(calls.index.to_numpy(float), puts.index.to_numpy(float))
    prof = pd.DataFrame(index=pd.Index(K, name="strike"))
    prof["oi_call"] = calls["oi"].reindex(K).fillna(0.0).to_numpy()
    prof["oi_put"] = puts["oi"].reindex(K).fillna(0.0).to_numpy()
    prof["iv_call"] = calls["iv"].reindex(K).to_numpy()
    prof["iv_put"] = puts["iv"].reindex(K).to_numpy()
    prof["iv_call"] = prof["iv_call"].fillna(prof["iv_put"])
    prof["iv_put"] = prof["iv_put"].fillna(prof["iv_call"])
    return prof.reset_index()


def build_days(nifty, vix, bhav, ptables, grid, log, grid_pts: float = GRID, day_filter=None, label: str = "Nifty"):
    """One record per panel day: chain profile, timing, weights and the 1-minute returns.

    ``nifty`` is the official daily close of the index (Nifty 50, or Nifty Bank with ``grid_pts`` = 100);
    ``day_filter(date, day_type)`` keeps only the days for which it returns True.
    """
    gdays = pd.DatetimeIndex(grid["days"])
    C = pd.DataFrame(grid["close"], index=gdays)
    O = pd.DataFrame(grid["open"], index=gdays)
    cover = np.isfinite(C).mean(axis=1)
    Cf = C.ffill(axis=1)
    R = pd.DataFrame(minute_returns(O.iloc[:, 0].to_numpy(), Cf.to_numpy()), index=gdays)
    rv_daily = (R.iloc[:, :DAILY_LAST_SLOT + 1] ** 2).sum(axis=1, min_count=DAILY_LAST_SLOT + 1)

    official = pd.DatetimeIndex(nifty.index).normalize()
    full_bars = cover.reindex(official).fillna(0.0) >= 0.98
    special = official[~full_bars.to_numpy()]
    sess = official[full_bars.to_numpy()]
    # history for the HAR terms: full bar days before the first official day, then official full sessions
    hist = gdays[(cover.to_numpy() >= 0.98) & (gdays < official[0])].append(sess)
    bdays = pd.DatetimeIndex(sorted(bhav))
    expiries = data.actual_expiries(bhav)
    rules = ExpiryRules()
    pt_dates = pd.DatetimeIndex(sorted(ptables))
    records, checks = [], []
    for t in sess:
        if t <= bdays[0]:
            continue
        nxt = expiries[expiries >= t]
        if not len(nxt):
            continue
        E = nxt[0]
        n_full = int(np.sum((sess > t) & (sess <= E)))
        dtype = "expiry" if n_full == 0 else ("day_before" if n_full == 1 else "ordinary")
        if day_filter is not None and not day_filter(t, dtype):
            continue
        ch, cdate, stale = ps.chain_before(bhav, bdays, t, E)
        if ch is None or not len(ch) or cdate not in nifty.index:
            checks.append({"date": t, "dropped": "no chain"})
            continue
        n_prev = int(np.sum((sess > cdate) & (sess <= E)))
        T_prev = max(n_prev, 1) / TRADING_DAYS
        S_prev = float(nifty[cdate])
        und = ch["underlying"].dropna()
        fb = float(und.iloc[0]) if len(und) else S_prev
        c = ps.prepare_chain(ch, T_prev, RATE, fb)
        F_prev = c.attrs["F"]
        prof = chain_profile(c, F_prev)
        if prof.empty or prof[["oi_call", "oi_put"]].to_numpy().sum() <= 0:
            checks.append({"date": t, "dropped": "empty chain"})
            continue
        pdate = cdate if cdate in ptables else (pt_dates[pt_dates <= cdate].max() if (pt_dates <= cdate).any() else None)
        ptab = ptables.get(pdate) if pdate is not None else None
        w = {"us_convention": (1.0, -1.0)}
        if ptab is not None:
            pro = gex.ParticipantWeights.from_participant_oi(ptab, "Pro")
            cli = gex.ParticipantWeights.from_participant_oi(ptab, "Client", mirror=True)
            w["participant_pro"] = (pro.call, pro.put)
            w["client_mirror"] = (cli.call, cli.put)
        tot = prof["oi_call"] + prof["oi_put"]
        K_maxoi = float(prof.loc[tot.idxmax(), "strike"])
        Tp = np.array([T_prev])
        scale = gamma_at_spot([S_prev], Tp, prof["strike"], prof["oi_call"], prof["oi_put"],
                              prof["iv_call"], prof["iv_put"])["gross"][0]
        prev_sess = hist[hist < t]
        records.append({
            "date": t, "day_type": dtype, "regime": rules.regime(t), "expiry": E, "full_sessions": n_full,
            "chain_date": cdate, "chain_stale_days": stale, "chain_redated": "redated_from" in ch.attrs,
            "T_prev_days": n_prev, "S_prev": S_prev, "F_prev": F_prev, "n_strikes": len(prof),
            "total_oi": float(tot.sum()), "iv_nan_share": c.attrs["iv_nan_share"],
            "atm_iv": float(np.interp(F_prev, prof["strike"], 0.5 * (prof["iv_call"] + prof["iv_put"]))),
            "K_maxoi": K_maxoi, "scale_gross_prev_close": float(scale),
            "w_pro_call": w.get("participant_pro", (np.nan, np.nan))[0], "w_pro_put": w.get("participant_pro", (np.nan, np.nan))[1],
            "w_client_call": w.get("client_mirror", (np.nan, np.nan))[0], "w_client_put": w.get("client_mirror", (np.nan, np.nan))[1],
            "cas_day": bool(t >= CAS_START), "bar_coverage": float(cover[t]),
            "vix_prev": float(vix.get(cdate, np.nan)),
            "rv_d1": float(rv_daily.get(prev_sess[-1], np.nan)) if len(prev_sess) else np.nan,
            "rv_w": float(rv_daily.reindex(prev_sess[-5:]).mean()) if len(prev_sess) >= 5 else np.nan,
            "rv_m": float(rv_daily.reindex(prev_sess[-22:]).mean()) if len(prev_sess) >= 22 else np.nan,
            "grid": grid_pts,
            "_prof": prof, "_w": w, "_closes": Cf.loc[t].to_numpy(float), "_ret": R.loc[t].to_numpy(float),
        })
    if len(special) <= 12:
        left = ", ".join(str(d.date()) for d in special)
    else:
        left = (f"{len(special)} days, {', '.join(str(d.date()) for d in special[:3])}, ..., {special[-1].date()} "
                f"(the 1-minute bars end on {gdays.max().date()})")
    log(f"Official {label} days: {len(official)}; special or incomplete sessions left out: {left}")
    return records, expiries, checks


def day_gamma(rec: dict, floor: float = FLOOR_MAIN, close_slot: int = CLOSE_SLOT, K=None, oi_call=None,
              oi_put=None, iv_call=None, iv_put=None, anchors=ANCHORS, band: float = CONC_BAND) -> dict:
    """Gamma measures at every anchor of one day (optionally with a placebo profile)."""
    prof = rec["_prof"]
    K = prof["strike"].to_numpy(float) if K is None else K
    oc = prof["oi_call"].to_numpy(float) if oi_call is None else oi_call
    op = prof["oi_put"].to_numpy(float) if oi_put is None else oi_put
    ic = prof["iv_call"].to_numpy(float) if iv_call is None else iv_call
    ip = prof["iv_put"].to_numpy(float) if iv_put is None else iv_put
    S = spot_at_anchor(rec["_closes"], anchors)
    T = time_to_expiry(anchors, rec["full_sessions"], floor, close_slot)
    return gamma_at_spot(S, T, K, oc, op, ic, ip, band=band)


def anchor_rows(rec: dict) -> pd.DataFrame:
    """The panel rows of one day: outcomes, controls and all gamma measures."""
    r, cl = rec["_ret"], rec["_closes"]
    last_ok = CAS_LAST_SLOT if rec["cas_day"] else SESSION - 1
    S = spot_at_anchor(cl, ANCHORS)
    g = day_gamma(rec)
    out = pd.DataFrame({
        "date": rec["date"], "day_type": rec["day_type"], "regime": rec["regime"], "slot": ANCHORS,
        "clock": [slot_clock(s) for s in ANCHORS], "spot": S,
        "T_years": time_to_expiry(ANCHORS, rec["full_sessions"]),
        "rv15": [forward_rv(r, s, 15, last_ok) for s in ANCHORS],
        "rv30": [forward_rv(r, s, 30, last_ok) for s in ANCHORS],
        "rv_back15": [backward_rv(r, s, 15) for s in ANCHORS],
        "rv_back60": [backward_rv(r, s, 60) for s in ANCHORS],
        "gross": g["gross"], "gross_call": g["call"], "gross_put": g["put"], "conc": g["conc"],
    })
    grid = rec.get("grid", GRID)
    out["dist_near"] = ps.dist_to_grid(S, grid)
    out["dist_maxoi_pct"] = 100 * np.abs(np.log(S / rec["K_maxoi"]))
    out["ret_prev_pct"] = 100 * np.log(S / rec["S_prev"])
    out["absret_prev_pct"] = np.abs(out["ret_prev_pct"])
    sc = rec["scale_gross_prev_close"]
    for rule in RULES:
        wc, wp = rec["_w"].get(rule, (np.nan, np.nan))
        out[f"net_{rule}"] = net_gex(g["call"], g["put"], wc, wp) / sc
    for fl in FLOOR_VARIANTS:
        out[f"gross_floor{fl}"] = day_gamma(rec, floor=fl)["gross"]
    out["gross_asian"] = day_gamma(rec, close_slot=ASIAN_CLOSE_SLOT)["gross"]
    K = rec["_prof"]["strike"].to_numpy(float)
    gplus, gminus = day_gamma(rec, K=shift_profile(K, grid / 2)), day_gamma(rec, K=shift_profile(K, -grid / 2))
    out["gross_shift_plus"], out["conc_shift_plus"] = gplus["gross"], gplus["conc"]
    out["gross_shift_minus"], out["conc_shift_minus"] = gminus["gross"], gminus["conc"]
    return out


def fh_rows(rec: dict) -> pd.DataFrame:
    """Final-half-hour panel rows of one expiry day: 1-minute anchors 15:00 to 15:24, next-5-minute outcome."""
    r, cl = rec["_ret"], rec["_closes"]
    grid = rec.get("grid", GRID)
    A = FH_ANCHORS
    S = spot_at_anchor(cl, A)
    prof = rec["_prof"]
    K, oc, op = (prof[c].to_numpy(float) for c in ("strike", "oi_call", "oi_put"))
    ic, ip = prof["iv_call"].to_numpy(float), prof["iv_put"].to_numpy(float)
    g = day_gamma(rec, floor=FH_FLOOR, anchors=A)
    g25 = day_gamma(rec, floor=FH_FLOOR, anchors=A, band=0.0025)
    out = pd.DataFrame({
        "date": rec["date"], "day_type": rec["day_type"], "regime": rec["regime"], "slot": A,
        "clock": [slot_clock(a) for a in A], "spot": S, "minutes_to_1530": CLOSE_SLOT - A,
        "rv5": [forward_rv(r, a, FH_HORIZON) for a in A],
        "rv_back5": [backward_rv(r, a, 5) for a in A],
        "rv_back15": [backward_rv(r, a, 15) for a in A],
        "rv_back60": [backward_rv(r, a, 60) for a in A],
        "gross": g["gross"], "conc": g["conc"], "conc025": g25["conc"],
    })
    for fl in FH_FLOOR_VARIANTS:
        out[f"gross_floor{fl}"] = day_gamma(rec, floor=fl, anchors=A)["gross"]
    # averaged settlement: closes already printed in the window are bars SETTLE_FIRST_SLOT .. a-1 (known at a)
    n_future = SESSION - A
    past = np.array([np.mean(cl[SETTLE_FIRST_SLOT:a]) if a > SETTLE_FIRST_SLOT else S[i] for i, a in enumerate(A)])
    out["settle_past_mean"] = past
    for fl in (FH_FLOOR,) + FH_FLOOR_VARIANTS:
        gs = settlement_gamma_at_spot(S, n_future, past, K, oc, op, ic, ip, floor=fl)
        out[f"gross_settle_floor{fl}"] = gs["gross"]
    out["dist_near"] = ps.dist_to_grid(S, grid)
    out["dist_maxoi_pct"] = 100 * np.abs(np.log(S / rec["K_maxoi"]))
    out["ret_prev_pct"] = 100 * np.log(S / rec["S_prev"])
    out["absret_prev_pct"] = np.abs(out["ret_prev_pct"])
    gplus = day_gamma(rec, floor=FH_FLOOR, anchors=A, K=shift_profile(K, grid / 2))
    gminus = day_gamma(rec, floor=FH_FLOOR, anchors=A, K=shift_profile(K, -grid / 2))
    out["gross_shift_plus"], out["conc_shift_plus"] = gplus["gross"], gplus["conc"]
    out["gross_shift_minus"], out["conc_shift_minus"] = gminus["gross"], gminus["conc"]
    return out


def build_fh_panel(records) -> pd.DataFrame:
    fh = pd.concat([fh_rows(rec) for rec in records], ignore_index=True)
    for col in ["rv5", "rv_back5", "rv_back15", "rv_back60"]:
        fh[col] = fh[col].where(fh[col] > 0)
    fh["y5"] = np.log(fh["rv5"])
    fh["lag5"], fh["lag15"], fh["lag60"] = np.log(fh["rv_back5"]), np.log(fh["rv_back15"]), np.log(fh["rv_back60"])
    for col in [c for c in fh.columns if c.startswith("gross")]:
        fh["log_" + col] = np.log(fh[col].where(fh[col] > 0))
    fh["day_code"] = pd.factorize(fh["date"], sort=True)[0]
    return fh


def _anchor_rows_idx(i):
    return anchor_rows(_RECORDS[i])


_RECORDS: list = []
_PLACEBO_CTX: dict = {}


def build_panel(records, workers: int) -> pd.DataFrame:
    global _RECORDS
    _RECORDS = records
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_anchor_rows_idx, range(len(records)), chunksize=8)
    panel = pd.concat(parts, ignore_index=True)
    eps = 0.0
    for col in ["rv15", "rv30", "rv_back15", "rv_back60"]:
        panel[col] = panel[col].where(panel[col] > eps)
    panel["y15"] = np.log(panel["rv15"])
    panel["y30"] = np.log(panel["rv30"])
    panel["lag15"] = np.log(panel["rv_back15"])
    panel["lag60"] = np.log(panel["rv_back60"])
    for col in ["gross", "gross_floor60", "gross_floor120", "gross_asian", "gross_shift_plus", "gross_shift_minus"]:
        panel["log_" + col] = np.log(panel[col])
    panel["day_code"] = pd.factorize(panel["date"], sort=True)[0]
    return panel


# ---------------------------------------------------------------------------
# Placebo draws (b) other day's profile and (c) permuted OI
# ---------------------------------------------------------------------------


def _placebo_draw(args):
    """One draw of placebo (b) or (c) for one day type: returns coefficient and clustered t of the placebo."""
    kind, sample, seed = args
    ctx = _PLACEBO_CTX[sample]
    rng = np.random.default_rng(seed)
    recs = ctx["recs"]
    logs = []
    for i, rec in enumerate(recs):
        prof = rec["_prof"]
        K = prof["strike"].to_numpy(float)
        if kind == "other_day":
            j = rng.integers(0, len(recs) - 1)
            j = j + 1 if j >= i else j
            other = recs[j]
            po = other["_prof"]
            Kp = po["strike"].to_numpy(float) + recentre_shift(other["F_prev"], rec["F_prev"], ctx["grid"])
            ic = np.interp(Kp, K, prof["iv_call"].to_numpy(float))
            ip = np.interp(Kp, K, prof["iv_put"].to_numpy(float))
            g = day_gamma(rec, floor=ctx["floor"], anchors=ctx["anchors"], K=Kp, oi_call=po["oi_call"].to_numpy(float),
                          oi_put=po["oi_put"].to_numpy(float), iv_call=ic, iv_put=ip)
        else:
            oc, op = permute_strike_oi(K, prof["oi_call"], prof["oi_put"], rec["F_prev"], PERM_BAND, rng)
            g = day_gamma(rec, floor=ctx["floor"], anchors=ctx["anchors"], oi_call=oc, oi_put=op)
        with np.errstate(divide="ignore"):
            logs.append(np.column_stack([np.log(g["gross"]), g["conc"]]))
    X = np.concatenate(logs)[ctx["row_sel"]]
    if not np.isfinite(X).all():   # a placebo profile with no gamma at spot: report the draw as missing
        return [(kind, sample, reg, seed, np.nan, np.nan, np.nan, np.nan) for reg in ("log_gross", "conc")]
    X_dm = demean_two_way(X, ctx["g1"], ctx["g2"])
    out = []
    for j, reg in enumerate(("log_gross", "conc")):
        x_dm = X_dm[:, j]
        b, t = fast_coef(ctx["y_dm"], ctx["C_dm"], x_dm, ctx["g1"], ctx["G"])
        out.append((kind, sample, reg, seed, b, t, float(np.std(x_dm)),
                    float(np.corrcoef(x_dm, ctx["x_real_dm"][reg])[0, 1])))
    return out


def prepare_placebo_ctx(recs, panel, min_slot: int = 0, y: str = "y15", controls=CONTROLS, anchors=ANCHORS,
                        floor: float = FLOOR_MAIN, grid: float = GRID):
    """Everything a placebo draw needs: the sample's days, its rows, and y and controls with fixed effects removed."""
    dates = [r["date"] for r in recs]
    sub = panel[panel["date"].isin(dates)].sort_values(["date", "slot"]).reset_index(drop=True)
    assert len(sub) == len(recs) * len(anchors)
    cols = [y, "log_gross", "conc"] + list(controls)
    keep = sub[cols].notna().all(axis=1).to_numpy() & (sub["slot"] >= min_slot).to_numpy()
    s = sub[keep]
    g1 = pd.factorize(s["date"], sort=True)[0]
    g2 = pd.factorize(s["slot"], sort=True)[0]
    Z = demean_two_way(s[cols].to_numpy(float), g1, g2)
    return {"recs": recs, "row_sel": np.where(keep)[0], "g1": g1, "g2": g2, "G": int(g1.max() + 1),
            "y_dm": Z[:, 0], "x_real_dm": {"log_gross": Z[:, 1], "conc": Z[:, 2]}, "C_dm": Z[:, 3:],
            "anchors": anchors, "floor": floor, "grid": grid}


# ---------------------------------------------------------------------------
# Out-of-sample HAR test on expiry days
# ---------------------------------------------------------------------------


def oos_har(days_df: pd.DataFrame, sess_index: dict, target: str, extra: list, log) -> pd.DataFrame:
    """Expanding-window walk-forward on expiry days: HAR-style log model with and without ``extra``.

    The model is refit every OOS_REFIT trading days on all expiry days before the refit date; the
    first OOS_TRAIN_DAYS trading days are used for training only. Forecast = exp(fitted) times the
    mean of exp(residuals) in the training sample (Duan smearing). Returns one row per test day.
    """
    base = ["l_sofar", "l_last30", "l_rv_d1", "l_rv_w", "l_vix"]
    d = days_df.dropna(subset=[target] + base + extra).copy()
    d["k"] = d["date"].map(sess_index)
    refits = np.arange(OOS_TRAIN_DAYS, max(sess_index.values()) + 1, OOS_REFIT)
    rows = []
    for a in refits:
        test = d[(d["k"] >= a) & (d["k"] < a + OOS_REFIT)]
        train = d[d["k"] < a]
        if test.empty:
            continue
        preds = {}
        for name, cols in (("har", base), ("har_gamma", base + extra)):
            X = np.column_stack([np.ones(len(train))] + [train[c] for c in cols])
            beta, *_ = np.linalg.lstsq(X, np.log(train[target]), rcond=None)
            smear = float(np.mean(np.exp(np.log(train[target]) - X @ beta)))
            Xt = np.column_stack([np.ones(len(test))] + [test[c] for c in cols])
            preds[name] = np.exp(Xt @ beta) * smear
            if name == "har_gamma":
                preds["gamma_coef"] = np.full(len(test), beta[-1])
        rows.append(pd.DataFrame({"date": test["date"].to_numpy(), "y": test[target].to_numpy(),
                                  "har": preds["har"], "har_gamma": preds["har_gamma"],
                                  "gamma_coef_in_fit": preds["gamma_coef"], "n_train": len(train),
                                  "refit_k": a}))
    return pd.concat(rows, ignore_index=True)


def oos_summary(pred: pd.DataFrame, label: str) -> dict:
    lh = evaluation.qlike(pred["y"], pred["har"])
    lg = evaluation.qlike(pred["y"], pred["har_gamma"])
    dm2 = evaluation.diebold_mariano(lh, lg, alternative="two-sided")
    dm1 = evaluation.diebold_mariano(lh, lg, alternative="greater")
    d = np.asarray(lh) - np.asarray(lg)
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    return {"spec": label, "n_test_days": len(pred), "first_test_day": str(pd.Timestamp(pred["date"].min()).date()),
            "last_test_day": str(pd.Timestamp(pred["date"].max()).date()),
            "qlike_har": float(np.mean(lh)), "qlike_har_gamma": float(np.mean(lg)),
            "qlike_change_pct": 100 * (float(np.mean(lg)) / float(np.mean(lh)) - 1),
            "mean_loss_diff_har_minus_gamma": dm2["mean_diff"], "se_naive": se,
            "dm_stat": dm2["stat"], "p_dm_two_sided": dm2["p"], "p_dm_gamma_better": dm1["p"],
            "mde_80pct_power_pct_of_har_qlike": 100 * 2.8 * se / float(np.mean(lh)),
            "share_days_gamma_better": float(np.mean(d > 0))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def coef_row(res: dict, x: str, **labels) -> dict:
    c = res["coef"][x]
    row = dict(labels)
    row.update({"regressor": x, "coef": c["coef"], "ci95_low": c["ci_low"], "ci95_high": c["ci_high"],
                "se_cluster_day": c["se"], "t": c["t"], "p": c["p"], "se_twoway_day_minute": c["se_twoway"],
                "p_twoway": c["p_twoway"], "se_driscoll_kraay": c["se_dk"], "p_driscoll_kraay": c["p_dk"],
                "effect_per_within_sd": c["coef"] * c["within_sd_x"], "within_sd_x": c["within_sd_x"],
                "n_obs": res["n_obs"], "n_days": res["n_days"], "within_r2": res["within_r2"]})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/intraday_gamma")
    ap.add_argument("--stdout", default="results/intraday_gamma_stdout.txt")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    t_start = time.time()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))
        sys.stdout.flush()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    log(f"{LABEL}. Intraday dealer gamma at spot and next-minutes Nifty volatility.")
    log(f"Run started {time.strftime('%Y-%m-%d %H:%M:%S')}; workers={args.workers}; placebo draws={args.draws}")
    nifty, vix, bhav, ptables, grid = load_raw(raw, args.workers)
    log(f"Loaded {len(bhav)} NIFTY bhavcopies ({min(bhav).date()} to {max(bhav).date()}), "
        f"{len(ptables)} participant files, minute bars for {len(grid['days'])} days "
        f"({time.time() - t_start:.0f} s)")
    records, expiries, drop_checks = build_days(nifty, vix, bhav, ptables, grid, log)
    days_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in records])
    days_df.to_csv(out / "day_inputs.csv", index=False)
    log(f"Panel days: {len(records)} ({days_df['date'].min().date()} to {days_df['date'].max().date()}); "
        f"by type: {days_df['day_type'].value_counts().to_dict()}; expiry days in bhavcopies: {len(expiries)}")

    panel = build_panel(records, args.workers)
    log(f"Anchor panel: {len(panel)} rows, {len(ANCHORS)} anchors per day ({slot_clock(ANCHORS[0])} to "
        f"{slot_clock(ANCHORS[-1])}) ({time.time() - t_start:.0f} s)")
    derived = ["day_code", "y15", "y30", "lag15", "lag60"] + [c for c in panel.columns if c.startswith("log_")]
    panel.drop(columns=derived).to_csv(out / "anchor_panel.csv.gz", index=False, float_format="%.6g")

    # ---------------------------------------------------------------- data checks
    chk = [
        ("panel days", len(records)),
        ("expiry days", int((days_df["day_type"] == "expiry").sum())),
        ("days before expiry", int((days_df["day_type"] == "day_before").sum())),
        ("ordinary days", int((days_df["day_type"] == "ordinary").sum())),
        ("days dropped for a missing chain", len(drop_checks)),
        ("days whose chain is from the previous trading day (no look-ahead)",
         int((days_df["chain_date"] < days_df["date"]).sum())),
        ("days whose chain skipped a missing bhavcopy (stale > 0)", int((days_df["chain_stale_days"] > 0).sum())),
        ("days with a holiday re-dated contract", int(days_df["chain_redated"].sum())),
        ("closing-auction days (from 3 Aug 2026)", int(days_df["cas_day"].sum())),
        ("days without participant weights", int(days_df["w_pro_call"].isna().sum())),
        ("median strikes per chain (within +/-10%)", float(days_df["n_strikes"].median())),
        ("median share of chain rows with no IV from the price (filled)", float(days_df["iv_nan_share"].median())),
        ("anchor rows", len(panel)),
        ("rows with zero or missing next-15-min RV", int(panel["rv15"].isna().sum())),
        ("rows with zero or missing next-30-min RV (incl. closing-auction windows)", int(panel["rv30"].isna().sum())),
        ("rows with zero lagged 15-min RV", int(panel["rv_back15"].isna().sum())),
        ("rows where the 5-minute floor binds", int((CLOSE_SLOT - panel["slot"] < FLOOR_MAIN).sum())),
        ("minimum minutes to 15:30 on the anchor grid", int(CLOSE_SLOT - ANCHORS.max())),
    ]
    checks = pd.DataFrame(chk, columns=["check", "value"])
    checks.to_csv(out / "data_checks.csv", index=False)
    log("\n== data checks (data_checks.csv)")
    log(checks.to_string(index=False))
    pd.DataFrame(DESIGN_LOG, columns=["topic", "choice", "reason"]).to_csv(out / "design_log.csv", index=False)

    # ---------------------------------------------------------------- descriptives
    desc = []
    for dt in DAY_TYPES:
        s = panel[panel["day_type"] == dt]
        for sl in (15, 165, 285, 315, 345):
            a = s[s["slot"] == sl]
            desc.append({"day_type": dt, "clock": slot_clock(sl), "n_days": a["date"].nunique(),
                         "median_gross_gex_crore_per_1pct": float(np.nanmedian(a["gross"]) / 1e7),
                         "median_concentration_pm05pct": float(np.nanmedian(a["conc"])),
                         "median_rv15_x1e6": float(np.nanmedian(a["rv15"]) * 1e6),
                         "median_dist_near_pts": float(np.nanmedian(a["dist_near"])),
                         "sd_across_days_of_within_day_dev_log_gross": float(
                             s.assign(dm=s["log_gross"] - s.groupby("date")["log_gross"].transform("mean"))
                             .loc[lambda q: q["slot"] == sl, "dm"].std())})
    desc = pd.DataFrame(desc)
    desc.to_csv(out / "descriptives_by_time.csv", index=False)
    log("\n== descriptives by day type and time (descriptives_by_time.csv)")
    log(desc.to_string(index=False))
    # how wide is one option's gamma "bump" on expiry day? one standard deviation of the index move to expiry,
    # in index points: atm_iv * sqrt(T) * index level, with each expiry day's previous-close ATM implied vol
    ed = days_df[days_df["day_type"] == "expiry"]
    wrows = []
    for mins in (360, 240, 120, 60, 30, 20, 15, 10, 5):
        w = ed["atm_iv"] * np.sqrt(mins / (SESSION * TRADING_DAYS)) * ed["S_prev"]
        wrows.append({"minutes_to_1530": mins, "on_anchor_grid": mins >= CLOSE_SLOT - ANCHORS.max(),
                      "median_width_pts": float(w.median()), "p10_width_pts": float(w.quantile(0.1)),
                      "p90_width_pts": float(w.quantile(0.9)), "median_width_over_strike_spacing": float(w.median() / GRID)})
    width = pd.DataFrame(wrows)
    width.to_csv(out / "gamma_width_expiry_day.csv", index=False)
    log("\n== width of one option's gamma on expiry day, in index points (gamma_width_expiry_day.csv)")
    log(width.to_string(index=False))

    # ---------------------------------------------------------------- main regressions
    reg_rows = []
    tests = []   # every test that is run: family, name, p

    def run(sample_name, sub, y, x, controls=CONTROLS, family="secondary", spec="main", test_name=None):
        res = fe_ols(sub, y, [x] + list(controls))
        row = coef_row(res, x, sample=sample_name, outcome=y, spec=spec, family=family)
        reg_rows.append(row)
        tests.append({"family": family, "test": test_name or f"{sample_name} | {y} | {x} | {spec}",
                      "coef": row["coef"], "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"], "p": row["p"],
                      "source": "regressions.csv"})
        return res

    samples = {dt: panel[panel["day_type"] == dt] for dt in DAY_TYPES}
    for dt in DAY_TYPES:
        run(dt, samples[dt], "y15", "log_gross", family="headline", test_name=f"H: {dt}, log gross GEX at spot, next 15 min")
    for rule in RULES:
        run("expiry", samples["expiry"], "y15", f"net_{rule}", family="headline",
            test_name=f"H: expiry, signed GEX at spot ({rule}), next 15 min")
    for dt in DAY_TYPES:
        run(dt, samples[dt], "y30", "log_gross", spec="main")
        if dt != "expiry":
            for rule in RULES:
                run(dt, samples[dt], "y15", f"net_{rule}")
        run(dt, samples[dt], "y15", "conc", spec="concentration instead of level")
    exp = samples["expiry"]
    pm = exp[exp["slot"] >= 225]
    run("expiry afternoon (13:00 to 15:00)", pm, "y15", "log_gross")
    run("expiry afternoon (13:00 to 15:00)", pm, "y15", "conc", spec="concentration instead of level")
    for rule in RULES:
        run("expiry afternoon (13:00 to 15:00)", pm, "y15", f"net_{rule}")
    for reg in ("multi_weekly", "single_weekly_thu", "single_weekly_tue"):
        run(f"expiry, regime {reg}", exp[exp["regime"] == reg], "y15", "log_gross", family="exploratory")
    # robustness of the gross-GEX results
    pm_name = "expiry afternoon (13:00 to 15:00)"
    rob_samples = {dt: samples[dt] for dt in DAY_TYPES}
    rob_samples[pm_name] = pm
    lever = ["ret_prev_pct", "absret_prev_pct"]
    for dt, s in rob_samples.items():
        run(dt, s, "y15", "log_gross", controls=["lag15", "lag60"], family="robustness", spec="without distance-to-strike control")
        run(dt, s, "y15", "log_gross", controls=CONTROLS + ["dist_maxoi_pct"] + lever, family="robustness",
            spec="plus distance to max-OI strike, signed and absolute move since previous close")
        for fl in FLOOR_VARIANTS:
            run(dt, s, "y15", f"log_gross_floor{fl}", family="robustness", spec=f"time floor {fl} minutes")
        run(dt, s, "y15", "log_gross_asian", family="robustness", spec="clock ends 15:10 (averaged close)")
        # flexible distance control: 5-point distance bins x hour of day
        s2 = s.assign(hour=(s["slot"] + 15) // 60, dbin=np.minimum(s["dist_near"] // 5, 4).astype("Int64"))
        dums = pd.get_dummies(s2["dbin"].astype(str) + "_" + s2["hour"].astype(str), prefix="dxh", drop_first=True, dtype=float)
        s2 = pd.concat([s2, dums], axis=1)
        run(dt, s2, "y15", "log_gross", controls=["lag15", "lag60"] + list(dums.columns), family="robustness",
            spec="distance bins x hour dummies")
        noncas = s[~s["date"].ge(CAS_START)]
        run(dt, noncas, "y15", "log_gross", family="robustness", spec="before closing auction (to 31 Jul 2026)")
        # signed versions with the leverage control (added after the first run, see design log)
        for rule in RULES:
            run(dt, s, "y15", f"net_{rule}", controls=CONTROLS + lever, family="robustness",
                spec="signed, plus signed and absolute move since previous close")
        run(dt, s, "y15", "log_gross", controls=CONTROLS + lever, family="robustness",
            spec="plus signed and absolute move since previous close")
        run(dt, s, "y15", "conc", controls=CONTROLS + lever, family="robustness",
            spec="concentration, plus signed and absolute move since previous close")
    regs = pd.DataFrame(reg_rows)

    # within-day correlations of log gross GEX at spot with other anchor variables (after day and minute effects)
    corr_rows = []
    cvars = ["ret_prev_pct", "absret_prev_pct", "dist_near", "dist_maxoi_pct", "conc", "lag15",
             "log_gross_shift_plus", "net_us_convention", "net_participant_pro", "net_client_mirror"]
    for name, s in rob_samples.items():
        sub = s.dropna(subset=["log_gross"] + cvars)
        g1 = pd.factorize(sub["date"], sort=True)[0]
        g2 = pd.factorize(sub["slot"], sort=True)[0]
        Z = demean_two_way(sub[["log_gross"] + cvars].to_numpy(float), g1, g2)
        row = {"sample": name, "within_sd_log_gross": float(Z[:, 0].std())}
        for j, v in enumerate(cvars, start=1):
            row[f"corr_with_{v}"] = float(np.corrcoef(Z[:, 0], Z[:, j])[0, 1])
        corr_rows.append(row)
    wcorr = pd.DataFrame(corr_rows)
    wcorr.to_csv(out / "within_day_correlations.csv", index=False)
    log("\n== within-day correlations of log gross GEX at spot (after day and minute effects; within_day_correlations.csv)")
    log(wcorr.T.to_string())
    heads = {"log_gross": regs[(regs["spec"] == "main") & (regs["regressor"] == "log_gross")
                               & (regs["outcome"] == "y15")].set_index("sample"),
             "conc": regs[(regs["spec"] == "concentration instead of level") & (regs["regressor"] == "conc")
                          & (regs["outcome"] == "y15")].set_index("sample")}

    # ---------------------------------------------------------------- placebo (a): half-strike shift
    plac_rows = []

    def real_cols(name, reg):
        h = heads[reg]
        return {"real_coef": h.loc[name, "coef"], "real_ci95_low": h.loc[name, "ci95_low"],
                "real_ci95_high": h.loc[name, "ci95_high"], "real_p": h.loc[name, "p"], "real_t": h.loc[name, "t"]}

    for name, s in rob_samples.items():
        for reg, col, nm in (("log_gross", "log_gross_shift_plus", "(a) OI profile shifted +25 points"),
                             ("log_gross", "log_gross_shift_minus", "(a') OI profile shifted -25 points"),
                             ("conc", "conc_shift_plus", "(a) OI profile shifted +25 points"),
                             ("conc", "conc_shift_minus", "(a') OI profile shifted -25 points")):
            res = run(name, s, "y15", col, family="placebo", spec=nm)
            c = res["coef"][col]
            sub = s.dropna(subset=["y15", reg, col] + CONTROLS)
            g1 = pd.factorize(sub["date"], sort=True)[0]
            g2 = pd.factorize(sub["slot"], sort=True)[0]
            Z = demean_two_way(sub[[reg, col]].to_numpy(float), g1, g2)
            row = {"sample": name, "regressor": reg, "placebo": nm, "coef": c["coef"], "ci95_low": c["ci_low"],
                   "ci95_high": c["ci_high"],
                   "p": c["p"], "t": c["t"], "within_corr_with_real": float(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]),
                   "n_draws": np.nan, "share_draws_p_below_005": np.nan, "randomization_p": np.nan,
                   "source": "regressions.csv"}
            row.update(real_cols(name, reg))
            plac_rows.append(row)
        # horse race: real and shifted together
        res = fe_ols(s, "y15", ["log_gross", "log_gross_shift_plus"] + CONTROLS)
        for x in ("log_gross", "log_gross_shift_plus"):
            reg_rows.append(coef_row(res, x, sample=name, outcome="y15", spec="horse race: real and +25 shift together",
                                     family="placebo"))
            tests.append({"family": "placebo", "test": f"{name} | horse race (real and +25 shift together) | {x}",
                          "coef": res["coef"][x]["coef"], "ci95_low": res["coef"][x]["ci_low"],
                          "ci95_high": res["coef"][x]["ci_high"], "p": res["coef"][x]["p"], "source": "regressions.csv"})
    regs = pd.DataFrame(reg_rows)

    # ---------------------------------------------------------------- placebos (b) and (c): many draws
    global _PLACEBO_CTX
    _PLACEBO_CTX = {name: prepare_placebo_ctx([r for r in records if r["day_type"] == dt], panel, ms)
                    for name, (dt, ms) in PLACEBO_SAMPLES.items()}
    names = list(PLACEBO_SAMPLES)
    jobs = [(kind, name, SEED + 1000 * k + 17 * names.index(name) + 7 * (kind == "permuted"))
            for kind in ("other_day", "permuted") for name in names for k in range(args.draws)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        draws = pool.map(_placebo_draw, jobs, chunksize=4)
    draws = pd.DataFrame([r for d in draws for r in d],
                         columns=["placebo", "sample", "regressor", "seed", "coef", "t_cluster_day", "within_sd_x",
                                  "within_corr_with_real"])
    draws.to_csv(out / "placebo_draws.csv", index=False)
    log(f"\nPlacebo draws (b) and (c): {len(draws)} regressions in {time.time() - t0:.0f} s")
    for (kind, name, reg), g in draws.groupby(["placebo", "sample", "regressor"], sort=False):
        real_b, real_t = heads[reg].loc[name, "coef"], heads[reg].loc[name, "t"]
        nm = "(b) another day's OI profile, re-centred" if kind == "other_day" else "(c) strike-level OI permuted within +/-5%"
        p_rand = float((1 + np.sum(np.abs(g["t_cluster_day"]) >= abs(real_t))) / (len(g) + 1))
        G = _PLACEBO_CTX[name]["G"]
        row = {"sample": name, "regressor": reg, "placebo": nm, "coef": float(g["coef"].median()),
               "ci95_low": float(g["coef"].quantile(0.025)), "ci95_high": float(g["coef"].quantile(0.975)),
               "p": np.nan, "t": float(g["t_cluster_day"].median()),
               "within_corr_with_real": float(g["within_corr_with_real"].median()),
               "n_draws": len(g),
               "share_draws_p_below_005": float(np.mean(np.abs(g["t_cluster_day"]) >= stats.t.ppf(0.975, G - 1))),
               "randomization_p": p_rand, "source": "placebo_draws.csv"}
        row.update(real_cols(name, reg))
        plac_rows.append(row)
        tests.append({"family": "placebo", "test": f"{name} | {reg} | {nm} | randomization p (share of draws with |t| >= real |t|)",
                      "coef": float(g["coef"].median()), "ci95_low": float(g["coef"].quantile(0.025)),
                      "ci95_high": float(g["coef"].quantile(0.975)), "p": p_rand, "source": "placebo_summary.csv"})
    plac = pd.DataFrame(plac_rows)
    plac.to_csv(out / "placebo_summary.csv", index=False)

    # ---------------------------------------------------------------- out of sample (expiry days)
    sess_dates = pd.DatetimeIndex(sorted(days_df["date"]))
    sess_index = {d: i for i, d in enumerate(sess_dates)}
    e = days_df[days_df["day_type"] == "expiry"].copy()
    by_day = {r["date"]: r for r in records if r["day_type"] == "expiry"}
    rows = []
    for d0 in e["date"]:
        rec = by_day[d0]
        r_ = rec["_ret"]
        S1430 = spot_at_anchor(rec["_closes"], SLOT_1430)
        g = gamma_at_spot([S1430], time_to_expiry(SLOT_1430, 0), rec["_prof"]["strike"], rec["_prof"]["oi_call"],
                          rec["_prof"]["oi_put"], rec["_prof"]["iv_call"], rec["_prof"]["iv_put"])
        rows.append({"date": d0, "cas_day": rec["cas_day"],
                     "y_last_hour": float(np.sum(r_[SLOT_1430:SESSION] ** 2)),
                     "y_1430_1515": float(np.sum(r_[SLOT_1430:CAS_LAST_SLOT + 1] ** 2)),
                     "sofar": float(np.sum(r_[:SLOT_1430] ** 2)), "last30": float(np.sum(r_[SLOT_1430 - 30:SLOT_1430] ** 2)),
                     "gross_1430": float(g["gross"][0]), "scale": rec["scale_gross_prev_close"],
                     "rv_d1": rec["rv_d1"], "rv_w": rec["rv_w"], "vix_prev": rec["vix_prev"]})
    od = pd.DataFrame(rows)
    od["l_sofar"] = np.log(od["sofar"])
    od["l_last30"] = np.log(od["last30"])
    od["l_rv_d1"] = np.log(od["rv_d1"])
    od["l_rv_w"] = np.log(od["rv_w"])
    od["l_vix"] = np.log(od["vix_prev"])
    od["l_gross_1430"] = np.log(od["gross_1430"])
    od["l_gross_1430_rel"] = np.log(od["gross_1430"] / od["scale"])
    oos_rows, oos_preds = [], []
    specs = [("main: last hour 14:30 to 15:30, closing-auction days excluded, log gross GEX at 14:30",
              od[~od["cas_day"]], "y_last_hour", ["l_gross_1430"]),
             ("14:30 to 15:15, all expiry days, log gross GEX at 14:30", od, "y_1430_1515", ["l_gross_1430"]),
             ("last hour, closing-auction days excluded, log(gross GEX at 14:30 / gross at previous close)",
              od[~od["cas_day"]], "y_last_hour", ["l_gross_1430_rel"])]
    for k, (nm, dd, tgt, extra) in enumerate(specs):
        pr = oos_har(dd, sess_index, tgt, extra, log)
        pr["spec"] = nm
        oos_preds.append(pr)
        sm = oos_summary(pr, nm)
        oos_rows.append(sm)
        tests.append({"family": "headline" if k == 0 else "secondary", "test": f"{'H: ' if k == 0 else ''}OOS DM: {nm}",
                      "coef": sm["qlike_change_pct"], "ci95_low": np.nan, "ci95_high": np.nan,
                      "p": sm["p_dm_two_sided"], "source": "oos_summary.csv"})
    oos = pd.DataFrame(oos_rows)
    oos.to_csv(out / "oos_summary.csv", index=False)
    pd.concat(oos_preds).to_csv(out / "oos_forecasts.csv", index=False)
    od.to_csv(out / "oos_expiry_day_features.csv", index=False)

    # ---------------------------------------------------------------- follow-ups (pre-registered, see design log)
    hc = heads["conc"].loc["expiry"]
    run_followups(records, grid, raw, out, args, log, vix, ptables, tests,
                  {"coef": hc["coef"], "ci95_low": hc["ci95_low"], "ci95_high": hc["ci95_high"]})

    # ---------------------------------------------------------------- regressions table and multiple testing
    regs = pd.DataFrame(reg_rows)
    regs.to_csv(out / "regressions.csv", index=False)
    tests = pd.DataFrame(tests)
    tests["p_holm_headline"] = np.nan
    hmask = tests["family"] == "headline"
    tests.loc[hmask, "p_holm_headline"] = holm(tests.loc[hmask, "p"].to_numpy())
    tests["p_holm_within_family"] = np.nan
    for fam, idx in tests.groupby("family").groups.items():
        ok = tests.loc[idx, "p"].notna()
        tests.loc[ok[ok].index, "p_holm_within_family"] = holm(tests.loc[ok[ok].index, "p"].to_numpy())
    log(f"\nTests run, by family: {tests['family'].value_counts().to_dict()}")
    tests.insert(0, "test_id", np.arange(1, len(tests) + 1))
    tests.to_csv(out / "all_tests.csv", index=False)

    # ---------------------------------------------------------------- deciles for the figure (expiry afternoons)
    dec_rows = []
    for col, nm in (("log_gross", "real OI profile"), ("log_gross_shift_plus", "placebo: profile shifted +25 points")):
        sub = pm.dropna(subset=["y15", col]).copy()
        g1 = pd.factorize(sub["date"], sort=True)[0]
        g2 = pd.factorize(sub["slot"], sort=True)[0]
        Z = demean_two_way(sub[["y15", col]].to_numpy(float), g1, g2)
        sub["y_res"], sub["x_res"] = Z[:, 0], Z[:, 1]
        sub["dec_raw"] = pd.qcut(sub[col], 10, labels=False) + 1
        sub["dec_res"] = pd.qcut(sub["x_res"], 10, labels=False) + 1
        for dec, g in sub.groupby("dec_raw"):
            dec_rows.append({"profile": nm, "binning": "raw", "decile": int(dec), "n": len(g),
                             "mean_rv15_x1e6": float(g["rv15"].mean() * 1e6),
                             "se_rv15_x1e6": float(g["rv15"].std() / np.sqrt(len(g)) * 1e6),
                             "mean_y_res": np.nan, "se_y_res": np.nan})
        for dec, g in sub.groupby("dec_res"):
            dec_rows.append({"profile": nm, "binning": "within day and time of day", "decile": int(dec), "n": len(g),
                             "mean_rv15_x1e6": np.nan, "se_rv15_x1e6": np.nan,
                             "mean_y_res": float(g["y_res"].mean()),
                             "se_y_res": float(g["y_res"].std() / np.sqrt(len(g)))})
    deciles = pd.DataFrame(dec_rows)
    deciles.to_csv(out / "fig_deciles_expiry_afternoon.csv", index=False)
    make_figures(deciles, regs, plac, out)

    # ---------------------------------------------------------------- printed summary
    show = ["sample", "outcome", "spec", "regressor", "coef", "ci95_low", "ci95_high", "p", "p_twoway",
            "p_driscoll_kraay", "effect_per_within_sd", "n_obs", "n_days"]
    log("\n== headline regressions (regressions.csv, family = headline)")
    log(regs[regs["family"] == "headline"][show].to_string(index=False))
    log("\n== secondary and exploratory regressions (regressions.csv)")
    log(regs[regs["family"].isin(["secondary", "exploratory"])][show].to_string(index=False))
    log("\n== robustness (regressions.csv, family = robustness)")
    log(regs[regs["family"] == "robustness"][show].to_string(index=False))
    log("\n== placebos (placebo_summary.csv; for (b) and (c) coef is the median over draws and the interval is "
        "the 2.5% to 97.5% range of draws)")
    log(plac.to_string(index=False))
    log("\n== out of sample, expiry days (oos_summary.csv)")
    log(oos.T.to_string())
    log("\n== every test run, with Holm-adjusted p over the headline set (all_tests.csv)")
    log(tests.to_string(index=False))
    log(f"\nFinished in {time.time() - t_start:.0f} s")
    Path(args.stdout).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Follow-ups: final half hour of expiry days, and Bank Nifty replication
# ---------------------------------------------------------------------------


def one_sided_negative(coef: float, p_two: float) -> float:
    """One-sided p for the prediction 'coefficient below zero', from the two-sided t-test p."""
    return p_two / 2 if coef < 0 else 1 - p_two / 2


def run_followups(records, grid, raw, out, args, log, vix, ptables, tests, nifty_conc: dict):
    """The two pre-registered follow-ups. Writes followup_*.csv and fig3; appends its tests to ``tests``."""
    t_fu = time.time()
    rows, plac_rows, chk = [], [], []
    real = {}

    def run_f(sample, sub, y, x, controls, family, spec, test_name=None):
        res = fe_ols(sub, y, [x] + list(controls))
        row = coef_row(res, x, sample=sample, outcome=y, spec=spec, family=family)
        row["controls"] = " + ".join(controls)
        row["mde_80pct_power"] = 2.8 * row["se_cluster_day"]
        rows.append(row)
        tests.append({"family": family, "test": test_name or f"{sample} | {y} | {x} | {spec}", "coef": row["coef"],
                      "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"], "p": row["p"],
                      "source": "followup_regressions.csv"})
        return row

    def placebo_a(sample, sub, y, controls, half):
        for reg, col, nm in (("log_gross", "log_gross_shift_plus", f"(a) OI profile shifted +{half:.0f} points"),
                             ("log_gross", "log_gross_shift_minus", f"(a') OI profile shifted -{half:.0f} points"),
                             ("conc", "conc_shift_plus", f"(a) OI profile shifted +{half:.0f} points"),
                             ("conc", "conc_shift_minus", f"(a') OI profile shifted -{half:.0f} points")):
            row = run_f(sample, sub, y, col, controls, "followup_placebo", nm)
            d = sub.dropna(subset=[y, reg, col] + list(controls))
            Z = demean_two_way(d[[reg, col]].to_numpy(float), pd.factorize(d["date"], sort=True)[0],
                               pd.factorize(d["slot"], sort=True)[0])
            r = real[(sample, reg)]
            plac_rows.append({"sample": sample, "regressor": reg, "placebo": nm, "coef": row["coef"],
                              "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"], "p": row["p"], "t": row["t"],
                              "within_corr_with_real": float(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]), "n_draws": np.nan,
                              "n_draws_valid": np.nan, "share_draws_p_below_005": np.nan, "randomization_p": np.nan,
                              "source": "followup_regressions.csv", "real_coef": r["coef"], "real_ci95_low": r["ci95_low"],
                              "real_ci95_high": r["ci95_high"], "real_p": r["p"], "real_t": r["t"]})
        res = fe_ols(sub, y, ["log_gross", "log_gross_shift_plus"] + list(controls))
        for x in ("log_gross", "log_gross_shift_plus"):
            row = coef_row(res, x, sample=sample, outcome=y, spec=f"horse race: real and +{half:.0f} shift together",
                           family="followup_placebo")
            row["controls"] = " + ".join(controls)
            row["mde_80pct_power"] = 2.8 * row["se_cluster_day"]
            rows.append(row)
            tests.append({"family": "followup_placebo",
                          "test": f"{sample} | horse race (real and +{half:.0f} shift together) | {x}",
                          "coef": row["coef"], "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"],
                          "p": row["p"], "source": "followup_regressions.csv"})

    # ------------------------------------------------------------ (1) final half hour, Nifty expiry days
    fh_recs = [r for r in records if r["day_type"] == "expiry" and not r["cas_day"]]
    fh = build_fh_panel(fh_recs)
    derived = ["day_code", "y5", "lag5", "lag15", "lag60"] + [c for c in fh.columns if c.startswith("log_")]
    fh.drop(columns=derived).to_csv(out / "followup_final_half_hour_panel.csv.gz", index=False, float_format="%.6g")
    gdays = pd.DatetimeIndex(grid["days"])
    rawC = pd.DataFrame(grid["close"], index=gdays)
    fh_dates = pd.DatetimeIndex([r["date"] for r in fh_recs])
    R = np.vstack([r["_ret"] for r in fh_recs])
    n_eff = (SESSION - FH_ANCHORS).astype(float)
    t_eff = (n_eff + 1) * (2 * n_eff + 1) / (6 * n_eff)
    chk += [
        ("final half hour", "expiry days before the closing auction (sample days)", len(fh_recs)),
        ("final half hour", "first and last sample day", f"{fh_dates.min().date()} to {fh_dates.max().date()}"),
        ("final half hour", "anchor rows (25 per day)", len(fh)),
        ("final half hour", "rows with zero or missing next-5-minute RV", int(fh["rv5"].isna().sum())),
        ("final half hour", "rows with zero RV in the previous 5 minutes (lag dropped)", int(fh["rv_back5"].isna().sum())),
        ("final half hour", "missing 1-minute bars 15:00 to 15:29 on sample days (forward-filled)",
         int(rawC.loc[fh_dates].iloc[:, SETTLE_FIRST_SLOT:SESSION].isna().to_numpy().sum())),
        ("final half hour", "share of zero 1-minute returns, 15:00 to 15:29", float(np.mean(R[:, SETTLE_FIRST_SLOT:SESSION] == 0))),
        ("final half hour", "share of zero 1-minute returns, 14:15 to 14:59", float(np.mean(R[:, 300:SETTLE_FIRST_SLOT] == 0))),
        ("final half hour", "minimum minutes to 15:30 at an anchor (15:24)", int(CLOSE_SLOT - FH_ANCHORS.max())),
        ("final half hour", "rows where the 1, 2 or 5 minute floor binds on the clock to 15:30",
         int((CLOSE_SLOT - fh["slot"] < max(FH_FLOOR_VARIANTS)).sum())),
        ("final half hour", "averaged settlement: variance-equivalent minutes left at 15:00", float(t_eff[0])),
        ("final half hour", "averaged settlement: variance-equivalent minutes left at 15:24", float(t_eff[-1])),
        ("final half hour", "averaged settlement: anchors per day where the 2-minute floor binds", int(np.sum(t_eff < 2))),
        ("final half hour", "averaged settlement: anchors per day where the 5-minute floor binds (from 15:17)",
         int(np.sum(t_eff < 5))),
        ("final half hour", "rows with zero gross GEX at spot (clock to 15:30)", int((fh["gross"] <= 0).sum())),
        ("final half hour", "rows with zero gross GEX at spot (averaged settlement, floor 1)",
         int((fh["gross_settle_floor1"] <= 0).sum())),
    ]
    fd = []
    for a in (345, 351, 357, 363, 369):
        q = fh[fh["slot"] == a]
        fd.append({"study": "final half hour", "clock": slot_clock(a), "n_days": q["date"].nunique(),
                   "median_gross_gex_crore_per_1pct": float(q["gross"].median() / 1e7),
                   "median_gross_settle_floor1_crore_per_1pct": float(q["gross_settle_floor1"].median() / 1e7),
                   "median_concentration_pm05pct": float(q["conc"].median()),
                   "median_rv_next_x1e6": float(q["rv5"].median() * 1e6),
                   "median_dist_near_pts": float(q["dist_near"].median()),
                   "median_dist_maxoi_pct": float(q["dist_maxoi_pct"].median())})

    FH = FH_NAME
    real[(FH, "log_gross")] = run_f(FH, fh, "y5", "log_gross", FH_CONTROLS, "followup_primary", "FH1 main",
                                    "FH1: final half hour, log gross GEX at spot, next 5 min")
    real[(FH, "conc")] = run_f(FH, fh, "y5", "conc", FH_CONTROLS, "followup_primary", "FH2 main",
                               "FH2: final half hour, concentration within +/-0.5%, next 5 min")
    for fl in FH_FLOOR_VARIANTS:
        run_f(FH, fh, "y5", f"log_gross_floor{fl}", FH_CONTROLS, "followup_secondary", f"time floor {fl} minutes")
    for fl in (FH_FLOOR,) + FH_FLOOR_VARIANTS:
        run_f(FH, fh, "y5", f"log_gross_settle_floor{fl}", FH_CONTROLS, "followup_secondary",
              f"averaged-settlement gamma, floor {fl} minutes")
    run_f(FH, fh, "y5", "conc025", FH_CONTROLS, "followup_secondary", "concentration within +/-0.25%")
    run_f(FH, fh, "y5", "log_gross", CONTROLS, "followup_secondary", "main-study controls only")
    run_f(FH, fh, "y5", "conc", CONTROLS, "followup_secondary", "concentration, main-study controls only")
    placebo_a(FH, fh, "y5", FH_CONTROLS, GRID / 2)
    log(f"\nFinal half hour panel: {len(fh)} rows on {len(fh_recs)} expiry days ({time.time() - t_fu:.0f} s)")

    # ------------------------------------------------------------ (2) Bank Nifty expiry days with weekly options
    bn_close, bn_bhav, bn_grid = load_banknifty(raw, args.workers)
    bn_recs, bn_exp, bn_drop = build_days(
        bn_close, vix, bn_bhav, ptables, bn_grid, log, grid_pts=BN_GRID,
        day_filter=lambda t, d: d == "expiry" and BN_START <= t < BN_WEEKLY_END, label="Nifty Bank")
    bn = build_panel(bn_recs, args.workers)
    bn.drop(columns=["day_code", "y15", "y30", "lag15", "lag60"] + [c for c in bn.columns if c.startswith("log_")]) \
      .to_csv(out / "followup_banknifty_panel.csv.gz", index=False, float_format="%.6g")
    bdf = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in bn_recs])
    bdf.to_csv(out / "followup_banknifty_day_inputs.csv", index=False)
    in_win = bn_exp[(bn_exp >= BN_START) & (bn_exp < BN_WEEKLY_END)]
    missing = sorted(set(in_win) - set(bdf["date"]))
    chk += [
        ("Bank Nifty", "BANKNIFTY expiry dates in the bhavcopies, 3 Jan 2023 to 19 Nov 2024", len(in_win)),
        ("Bank Nifty", "sample days (expiry days with full 1-minute bars and a previous-day chain)", len(bn_recs)),
        ("Bank Nifty", "first and last sample day", f"{bdf['date'].min().date()} to {bdf['date'].max().date()}"),
        ("Bank Nifty", "expiry dates left out (incomplete bars or no chain)",
         ", ".join(str(pd.Timestamp(d).date()) for d in missing) or "none"),
        ("Bank Nifty", "sample days by weekday", str(bdf["date"].dt.day_name().value_counts().to_dict())),
        ("Bank Nifty", "days whose chain is from the previous trading day", int((bdf["chain_date"] < bdf["date"]).sum())),
        ("Bank Nifty", "days whose chain skipped a missing bhavcopy", int((bdf["chain_stale_days"] > 0).sum())),
        ("Bank Nifty", "days with a holiday re-dated contract", int(bdf["chain_redated"].sum())),
        ("Bank Nifty", "median strikes per chain (within +/-10%)", float(bdf["n_strikes"].median())),
        ("Bank Nifty", "median strike spacing near the money (points)",
         float(np.median([np.median(np.diff(r["_prof"]["strike"].to_numpy(float)[
             np.abs(r["_prof"]["strike"].to_numpy(float) / r["F_prev"] - 1) <= 0.01])) for r in bn_recs]))),
        ("Bank Nifty", "anchor rows", len(bn)),
        ("Bank Nifty", "rows with zero or missing next-15-minute RV", int(bn["rv15"].isna().sum())),
        ("Bank Nifty", "rows with zero lagged 15-minute RV", int(bn["rv_back15"].isna().sum())),
    ]
    for a in (15, 165, 285, 315, 345):
        q = bn[bn["slot"] == a]
        fd.append({"study": "Bank Nifty expiry days", "clock": slot_clock(a), "n_days": q["date"].nunique(),
                   "median_gross_gex_crore_per_1pct": float(q["gross"].median() / 1e7),
                   "median_gross_settle_floor1_crore_per_1pct": np.nan,
                   "median_concentration_pm05pct": float(q["conc"].median()),
                   "median_rv_next_x1e6": float(q["rv15"].median() * 1e6),
                   "median_dist_near_pts": float(q["dist_near"].median()),
                   "median_dist_maxoi_pct": float(q["dist_maxoi_pct"].median())})
    BN = BN_NAME
    lever = ["ret_prev_pct", "absret_prev_pct"]
    real[(BN, "conc")] = run_f(BN, bn, "y15", "conc", CONTROLS, "followup_primary",
                               "BN1 main (same as the Nifty concentration test)",
                               "BN1: Bank Nifty expiry days, concentration within +/-0.5%, next 15 min")
    run_f(BN, bn, "y15", "conc", CONTROLS + lever, "followup_secondary",
          "concentration, plus signed and absolute move since previous close")
    real[(BN, "log_gross")] = run_f(BN, bn, "y15", "log_gross", CONTROLS, "followup_secondary", "log gross GEX at spot")
    run_f(BN, bn, "y15", "log_gross", CONTROLS + lever, "followup_robustness",
          "log gross, plus signed and absolute move since previous close (not pre-registered)")
    placebo_a(BN, bn, "y15", CONTROLS, BN_GRID / 2)
    log(f"Bank Nifty panel: {len(bn)} rows on {len(bn_recs)} expiry days ({time.time() - t_fu:.0f} s)")

    # ------------------------------------------------------------ placebos (b) and (c) for both follow-ups
    _PLACEBO_CTX[FH] = prepare_placebo_ctx(fh_recs, fh, y="y5", controls=FH_CONTROLS, anchors=FH_ANCHORS,
                                           floor=FH_FLOOR, grid=GRID)
    _PLACEBO_CTX[BN] = prepare_placebo_ctx(bn_recs, bn, y="y15", controls=CONTROLS, anchors=ANCHORS,
                                           floor=FLOOR_MAIN, grid=BN_GRID)
    fu_names = [FH, BN]
    jobs = [(kind, name, SEED + 500000 + 1000 * k + 17 * i + 7 * (kind == "permuted"))
            for kind in ("other_day", "permuted") for i, name in enumerate(fu_names) for k in range(args.draws)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        dr = pool.map(_placebo_draw, jobs, chunksize=4)
    dr = pd.DataFrame([r for d in dr for r in d], columns=["placebo", "sample", "regressor", "seed", "coef",
                                                          "t_cluster_day", "within_sd_x", "within_corr_with_real"])
    dr.to_csv(out / "followup_placebo_draws.csv", index=False)
    log(f"Follow-up placebo draws (b) and (c): {len(dr)} regressions in {time.time() - t0:.0f} s")
    for (kind, name, reg), g in dr.groupby(["placebo", "sample", "regressor"], sort=False):
        r = real[(name, reg)]
        v = g.dropna(subset=["t_cluster_day"])
        nm = "(b) another day's OI profile, re-centred" if kind == "other_day" else "(c) strike-level OI permuted within +/-5%"
        p_rand = float((1 + np.sum(np.abs(v["t_cluster_day"]) >= abs(r["t"]))) / (len(v) + 1))
        G = _PLACEBO_CTX[name]["G"]
        plac_rows.append({"sample": name, "regressor": reg, "placebo": nm, "coef": float(v["coef"].median()),
                          "ci95_low": float(v["coef"].quantile(0.025)), "ci95_high": float(v["coef"].quantile(0.975)),
                          "p": np.nan, "t": float(v["t_cluster_day"].median()),
                          "within_corr_with_real": float(v["within_corr_with_real"].median()), "n_draws": len(g),
                          "n_draws_valid": len(v),
                          "share_draws_p_below_005": float(np.mean(np.abs(v["t_cluster_day"]) >= stats.t.ppf(0.975, G - 1))),
                          "randomization_p": p_rand, "source": "followup_placebo_draws.csv", "real_coef": r["coef"],
                          "real_ci95_low": r["ci95_low"], "real_ci95_high": r["ci95_high"], "real_p": r["p"],
                          "real_t": r["t"]})
        tests.append({"family": "followup_placebo",
                      "test": f"{name} | {reg} | {nm} | randomization p (share of draws with |t| >= real |t|)",
                      "coef": float(v["coef"].median()), "ci95_low": float(v["coef"].quantile(0.025)),
                      "ci95_high": float(v["coef"].quantile(0.975)), "p": p_rand,
                      "source": "followup_placebo_summary.csv"})

    regs = pd.DataFrame(rows)
    regs.to_csv(out / "followup_regressions.csv", index=False)
    plac = pd.DataFrame(plac_rows)
    plac.to_csv(out / "followup_placebo_summary.csv", index=False)
    checks = pd.DataFrame(chk, columns=["study", "check", "value"])
    checks.to_csv(out / "followup_data_checks.csv", index=False)
    fdesc = pd.DataFrame(fd)
    fdesc.to_csv(out / "followup_descriptives.csv", index=False)

    # ------------------------------------------------------------ pre-registered primary tests, with Holm over three
    prim = [("FH1", FH, "log_gross", None), ("FH2", FH, "conc", "negative"), ("BN1", BN, "conc", "negative")]
    srows = []
    for code, name, reg, pred in prim:
        r = real[(name, reg)]
        row = {"test": code, "sample": name, "outcome": r["outcome"], "regressor": reg, "controls": r["controls"],
               "predicted_sign": pred or "none", "coef": r["coef"], "ci95_low": r["ci95_low"],
               "ci95_high": r["ci95_high"], "se_cluster_day": r["se_cluster_day"], "t": r["t"], "p_two_sided": r["p"],
               "p_one_sided_predicted_negative": one_sided_negative(r["coef"], r["p"]) if pred else np.nan,
               "p_twoway": r["p_twoway"], "p_driscoll_kraay": r["p_driscoll_kraay"],
               "effect_per_within_sd": r["effect_per_within_sd"], "mde_80pct_power": r["mde_80pct_power"],
               "n_obs": r["n_obs"], "n_days": r["n_days"]}
        for pl, key in (("a_plus", "(a) OI profile shifted"), ("a_minus", "(a') OI profile shifted"),
                        ("b", "(b) another day"), ("c", "(c) strike-level")):
            q = plac[(plac["sample"] == name) & (plac["regressor"] == reg) & plac["placebo"].str.startswith(key)].iloc[0]
            row[f"placebo_{pl}_coef"] = q["coef"]
            row[f"placebo_{pl}_p" if pl.startswith("a") else f"placebo_{pl}_randomization_p"] = (
                q["p"] if pl.startswith("a") else q["randomization_p"])
        srows.append(row)
    summ = pd.DataFrame(srows)
    summ.insert(summ.columns.get_loc("p_twoway"), "p_holm_over_FH1_FH2_BN1", holm(summ["p_two_sided"].to_numpy()))
    summ["nifty_main_study_coef_same_spec"] = [np.nan, nifty_conc["coef"], nifty_conc["coef"]]
    summ["nifty_main_study_ci95"] = ["", f"{nifty_conc['ci95_low']:.3f} to {nifty_conc['ci95_high']:.3f}",
                                     f"{nifty_conc['ci95_low']:.3f} to {nifty_conc['ci95_high']:.3f}"]
    # power against an effect as large as the Nifty main-study estimate (normal approximation, 5% level)
    z = abs(nifty_conc["coef"]) / summ["se_cluster_day"]
    summ["power_vs_nifty_coef_two_sided"] = np.where(summ["regressor"] == "conc", stats.norm.cdf(z - stats.norm.ppf(0.975))
                                                     + stats.norm.cdf(-z - stats.norm.ppf(0.975)), np.nan)
    summ["power_vs_nifty_coef_one_sided"] = np.where(summ["regressor"] == "conc",
                                                     stats.norm.cdf(z - stats.norm.ppf(0.95)), np.nan)
    summ.to_csv(out / "followup_summary.csv", index=False)
    make_followup_figure(regs, plac, summ, nifty_conc, out)

    show = ["sample", "spec", "regressor", "coef", "ci95_low", "ci95_high", "p", "p_twoway", "p_driscoll_kraay",
            "mde_80pct_power", "n_obs", "n_days"]
    log("\n== FOLLOW-UPS (pre-registered 2026-11-02 21:36 UTC, see design_log.csv)")
    log("\n== follow-up data checks (followup_data_checks.csv)")
    log(checks.to_string(index=False))
    log("\n== follow-up descriptives (followup_descriptives.csv)")
    log(fdesc.to_string(index=False))
    log("\n== follow-up regressions (followup_regressions.csv)")
    log(regs[show].to_string(index=False))
    log("\n== follow-up placebos (followup_placebo_summary.csv)")
    log(plac.to_string(index=False))
    log("\n== follow-up primary tests FH1, FH2, BN1 with Holm over the three (followup_summary.csv)")
    log(summ.T.to_string())
    log(f"Follow-ups finished in {time.time() - t_fu:.0f} s")


def make_followup_figure(regs: pd.DataFrame, plac: pd.DataFrame, summ: pd.DataFrame, nifty_conc: dict, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [(FH_NAME, "log_gross", "FH1: final half hour\nlog gross GEX at spot"),
              (FH_NAME, "conc", "FH2: final half hour\nconcentration"),
              (BN_NAME, "conc", "BN1: Bank Nifty expiry days\nconcentration"),
              (BN_NAME, "log_gross", "Bank Nifty expiry days\nlog gross GEX (secondary)")]
    keys = [None, "(a) OI profile shifted +", "(b) another day", "(c) strike-level"]
    labels = ["real OI profile (95% CI)", "(a) shifted by half a strike (95% CI)",
              "(b) another day (2.5% to 97.5% of draws)", "(c) permuted (2.5% to 97.5% of draws)"]
    cols = ["#1f5fa8", "#c8742c", "#5a9e57", "#8c6bb1"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for i, (name, reg, _) in enumerate(groups):
        for j, (lab, key, col) in enumerate(zip(labels, keys, cols)):
            if key is None:
                r = regs[(regs["sample"] == name) & (regs["regressor"] == reg) & regs["family"].isin(
                    ["followup_primary", "followup_secondary"]) & regs["spec"].isin(
                    ["FH1 main", "FH2 main", "BN1 main (same as the Nifty concentration test)", "log gross GEX at spot"])].iloc[0]
            else:
                r = plac[(plac["sample"] == name) & (plac["regressor"] == reg) & plac["placebo"].str.startswith(key)].iloc[0]
            b, lo, hi = r["coef"], r["ci95_low"], r["ci95_high"]
            x = i + (j - 1.5) * 0.18
            ax.errorbar([x], [b], yerr=[[b - lo], [hi - b]], fmt="o", color=col, capsize=3, label=lab if i == 0 else None)
    ax.hlines(nifty_conc["coef"], 1.55, 2.45, colors="#444444", linestyles="dashed", lw=1)
    ax.annotate(f"Nifty main-study estimate {nifty_conc['coef']:.2f}", (2.0, nifty_conc["coef"]), fontsize=7,
                ha="center", va="top", xytext=(0, -4), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[2] for g in groups], fontsize=8)
    ax.set_ylabel("coefficient (log RV of the next 5 or 15 minutes)")
    ax.set_title("REAL DATA. Pre-registered follow-ups: final half hour of Nifty expiry days (next 5 min) and Bank Nifty\n"
                 "expiry days (next 15 min). Day and minute fixed effects; bars as in Figure 2", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig3_followups.png", dpi=150)
    plt.close(fig)


def make_figures(deciles: pd.DataFrame, regs: pd.DataFrame, plac: pd.DataFrame, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"real OI profile": "#1f5fa8", "placebo: profile shifted +25 points": "#c8742c"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for nm, c in colors.items():
        a = deciles[(deciles["profile"] == nm) & (deciles["binning"] == "raw")]
        axes[0].errorbar(a["decile"], a["mean_rv15_x1e6"], yerr=1.96 * a["se_rv15_x1e6"], marker="o", color=c,
                         label=nm, capsize=2)
        b = deciles[(deciles["profile"] == nm) & (deciles["binning"] != "raw")]
        axes[1].errorbar(b["decile"], b["mean_y_res"], yerr=1.96 * b["se_y_res"], marker="o", color=c, label=nm, capsize=2)
    axes[0].set_title("Raw: mean next-15-minute RV by decile of gross GEX at spot", fontsize=10)
    axes[0].set_xlabel("decile of gross GEX at spot (1 = lowest)")
    axes[0].set_ylabel("realized variance x 1e6")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title("Within day and time of day: log RV residual by decile", fontsize=10)
    axes[1].set_xlabel("decile of log gross GEX at spot, after removing day and time-of-day means")
    axes[1].set_ylabel("log next-15-minute RV, residual")
    for ax in axes:
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
    fig.suptitle("REAL DATA. Nifty expiry afternoons (anchors 13:00 to 15:00). Bars: 95% intervals (anchors treated as "
                 "independent, so too narrow)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_deciles_expiry_afternoon.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    groups = list(PLACEBO_SAMPLES)
    head = regs[(regs["spec"] == "main") & (regs["regressor"] == "log_gross") & (regs["outcome"] == "y15")
                & regs["sample"].isin(groups)].set_index("sample")
    labels = ["real OI profile (95% CI)", "(a) shifted +25 (95% CI)", "(b) another day (2.5% to 97.5% of draws)",
              "(c) permuted (2.5% to 97.5% of draws)"]
    keys = [None, "(a) OI profile shifted +25 points", "(b) another day's OI profile, re-centred",
            "(c) strike-level OI permuted within +/-5%"]
    cols = ["#1f5fa8", "#c8742c", "#5a9e57", "#8c6bb1"]
    for i, name in enumerate(groups):
        for j, (lab, key, col) in enumerate(zip(labels, keys, cols)):
            if key is None:
                b, lo, hi = head.loc[name, ["coef", "ci95_low", "ci95_high"]]
            else:
                r = plac[(plac["sample"] == name) & (plac["placebo"] == key) & (plac["regressor"] == "log_gross")].iloc[0]
                b, lo, hi = r["coef"], r["ci95_low"], r["ci95_high"]
            x = i + (j - 1.5) * 0.18
            ax.errorbar([x], [b], yerr=[[b - lo], [hi - b]], fmt="o", color=col, capsize=3, label=lab if i == 0 else None)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["expiry days", "day before expiry", "ordinary days", "expiry afternoons\n(13:00 to 15:00)"])
    ax.set_ylabel("coefficient on log gross GEX at spot")
    ax.set_title("REAL DATA. Next-15-minute log RV on log gross GEX at spot (day and minute fixed effects).\n"
                 "Bars for the real profile and (a): 95% CI clustered by day; for (b) and (c): spread of placebo draws", fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig2_coefficients_and_placebos.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
