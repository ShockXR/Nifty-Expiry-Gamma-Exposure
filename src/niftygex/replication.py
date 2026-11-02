"""Helpers for replicating Chhimwal, Pandey and Pandey (2025) on real NSE data.

Used by ``experiments/replicate_paper.py`` and tested in
``tests/test_replication.py``. Everything here is plain data handling or small
statistics that the existing modules do not already provide:

* session trimming and a (days x minutes) grid of 1-minute bars;
* continuous log returns (the overnight gap sits in the first minute of a day);
* the true expiry calendar from the F&O bhavcopy;
* day-level volume measures from the bhavcopy;
* a week-relative comparison (a day of interest versus the rest of its week);
* a VAR fit whose lags never cross a segment boundary, VAR lag choice by AIC,
  and a bootstrap of connectedness that resamples whole segments (days or hours);
* standard errors and a profile likelihood for intraday GARCH persistence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from niftygex.connectedness import Connectedness, VarFit, connectedness_from_fit
from niftygex.volatility import _garch_q2, _norm_loglik, _t_loglik

# ---------------------------------------------------------------------------
# Session trimming and minute grids
# ---------------------------------------------------------------------------

# Inclusive bar start times (HH:MM). NSE's regular session is 09:15 to 15:30,
# so the bars run from 09:15 to 15:29 (375 bars).
SESSION_RULES = {
    "360_drop_last15": ("09:15", "15:14"),
    "360_drop_first15": ("09:30", "15:29"),
    "375_full": ("09:15", "15:29"),
}


def session_minutes(rule: str) -> list[str]:
    """The bar start times kept by a session rule, as 'HH:MM' strings."""
    start, end = SESSION_RULES[rule]
    t = pd.date_range(f"2000-01-01 {start}", f"2000-01-01 {end}", freq="1min")
    return [x.strftime("%H:%M") for x in t]


def load_minute_bars(path, start=None, end=None) -> pd.DataFrame:
    """Read a timestamp/open/high/low/close CSV and add date and time-of-day columns."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.normalize()
    df["tod"] = df["timestamp"].dt.strftime("%H:%M")
    return df


def trim_session(bars: pd.DataFrame, rule: str = "360_drop_last15", weekdays_only: bool = True) -> pd.DataFrame:
    """Keep the bars inside the session rule; optionally drop weekend special sessions."""
    start, end = SESSION_RULES[rule]
    keep = (bars["tod"] >= start) & (bars["tod"] <= end)
    if weekdays_only:
        keep &= bars["date"].dt.weekday < 5
    return bars[keep]


def minute_grid(bars: pd.DataFrame, rule: str, days=None) -> dict:
    """Pivot trimmed bars to (days x minutes) arrays, NaN where a bar is missing.

    Returns dict with keys open, high, low, close (arrays), days (DatetimeIndex)
    and minutes (list of 'HH:MM').
    """
    mins = session_minutes(rule)
    b = trim_session(bars, rule)
    if days is not None:
        b = b[b["date"].isin(pd.DatetimeIndex(days))]
    out = {}
    for col in ("open", "high", "low", "close"):
        p = b.pivot(index="date", columns="tod", values=col).reindex(columns=mins)
        if days is not None:
            p = p.reindex(pd.DatetimeIndex(days))
        out[col] = p.to_numpy(float)
        out["days"] = pd.DatetimeIndex(p.index)
    out["minutes"] = mins
    return out


def continuous_log_returns(close: np.ndarray, overnight: bool = True) -> np.ndarray:
    """1-minute log returns on a (days x minutes) close grid, read row by row.

    The first minute of each day is measured from the previous day's last
    close, so it contains the overnight gap (this is what a 64 x 360 - 1 =
    23,039 count implies). The very first return is NaN. A missing bar gives a
    NaN return and the next return spans the gap. With ``overnight=False`` the
    first minute of every day is set to NaN instead.
    """
    lc = np.log(np.asarray(close, float))
    flat = lc.ravel()
    filled = pd.Series(flat).ffill().to_numpy()
    r = np.full_like(flat, np.nan)
    r[1:] = filled[1:] - filled[:-1]
    r[~np.isfinite(flat)] = np.nan
    r = r.reshape(lc.shape)
    if not overnight:
        r[:, 0] = np.nan
    return r


# ---------------------------------------------------------------------------
# Bhavcopy: expiry calendar and daily volume
# ---------------------------------------------------------------------------


def read_legacy_bhavcopy(path) -> pd.DataFrame | None:
    """Read one filtered NSE F&O bhavcopy in the legacy layout (before 8 July 2024).

    Returns None for files in the newer UDiFF layout, which this replication
    does not need.
    """
    df = pd.read_csv(path)
    if "INSTRUMENT" not in df.columns:
        return None
    name = Path(path).name
    df["date"] = pd.Timestamp(name.split("_")[-1][:8])
    df["expiry"] = pd.to_datetime(df["EXPIRY_DT"].str.strip(), format="%d-%b-%Y")
    df["SYMBOL"] = df["SYMBOL"].str.strip()
    df["INSTRUMENT"] = df["INSTRUMENT"].str.strip()
    return df


def expiry_calendar(contracts: pd.DataFrame) -> pd.DataFrame:
    """Expiry days per symbol from contract-level bhavcopy rows.

    ``contracts`` needs columns SYMBOL, INSTRUMENT, expiry and date (trade
    date). A date is an expiry day for a symbol when some contract of that
    symbol has that date as EXPIRY_DT and trades (appears) for the last time on
    it. Weekdays are never assumed, so holiday shifts and contracts whose
    expiry was re-labelled (for example Bank Nifty March 2024 futures first
    listed for Thursday 28 March and later realigned to Wednesday 27 March)
    are handled. An expiry after the last loaded date cannot be confirmed and
    is left out.

    A monthly expiry is a date on which an index futures contract (FUTIDX) of
    that symbol expires. Index futures are monthly only. The simpler rule "last
    expiry of a calendar month" fails when a weekly expiry follows the monthly
    one in the same month (Bank Nifty: monthly Thursday 25 January 2024, then
    a weekly on Wednesday 31 January 2024).
    """
    last = contracts.groupby(["SYMBOL", "INSTRUMENT", "expiry"])["date"].max().reset_index()
    last = last[last["expiry"] == last["date"]]
    rows = []
    for sym, g in last.groupby("SYMBOL"):
        fut = set(g.loc[g["INSTRUMENT"] == "FUTIDX", "expiry"])
        for d in sorted(set(g["expiry"])):
            rows.append({"symbol": sym, "expiry": d, "is_monthly": d in fut, "weekday": d.day_name()})
    return pd.DataFrame(rows)


def infer_lot_size(fut_rows: pd.DataFrame) -> float:
    """Lot size implied by futures rows: VAL_INLAKH * 1e5 / (CONTRACTS * price), rounded to a multiple of 5."""
    f = fut_rows[fut_rows["CONTRACTS"] > 0]
    if f.empty:
        return np.nan
    px = f[["OPEN", "HIGH", "LOW", "CLOSE"]].mean(axis=1)
    raw = np.median(f["VAL_INLAKH"] * 1e5 / (f["CONTRACTS"] * px))
    return float(5 * np.round(raw / 5))


def daily_fo_volume(day: pd.DataFrame) -> pd.DataFrame:
    """Day-level F&O activity per symbol from one legacy bhavcopy day.

    For options VAL_INLAKH is notional, (strike + premium) x quantity, so the
    premium turnover is VAL - strike x quantity. Quantity = contracts x lot
    size, with the lot size inferred from that day's futures. Contract counts
    are not comparable across a lot-size change; quantity, notional and
    premium are.
    """
    rows = []
    for sym, g in day.groupby("SYMBOL"):
        fut = g[g["INSTRUMENT"] == "FUTIDX"]
        opt = g[g["INSTRUMENT"] == "OPTIDX"]
        lot = infer_lot_size(fut)
        opt_notional = opt["VAL_INLAKH"].sum() * 1e5
        opt_qty = opt["CONTRACTS"].sum() * lot
        prem = opt_notional - float((opt["STRIKE_PR"] * opt["CONTRACTS"]).sum() * lot)
        rows.append({
            "date": g["date"].iloc[0], "symbol": sym, "lot_size": lot,
            "opt_contracts": float(opt["CONTRACTS"].sum()), "opt_qty": opt_qty,
            "opt_notional_cr": opt_notional / 1e7, "opt_premium_cr": prem / 1e7,
            "fut_contracts": float(fut["CONTRACTS"].sum()), "fut_qty": fut["CONTRACTS"].sum() * lot,
            "fut_notional_cr": fut["VAL_INLAKH"].sum() / 100.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Week-relative comparison
# ---------------------------------------------------------------------------


def week_relative(values: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Log of a target day's value minus the mean log of the other days of its week.

    ``values`` (positive) and ``target`` (bool) are indexed by trading date.
    Weeks run Monday to Sunday. One row per target day that has at least one
    non-target day in its week. This removes anything common to the whole
    week (for example a volatile month), so a before/after comparison of these
    numbers asks only whether the target day stands out from its own week more
    or less than before.
    """
    v = pd.Series(np.log(values.astype(float)), index=pd.DatetimeIndex(values.index))
    t = target.reindex(v.index).fillna(False).astype(bool)
    wk = v.index.to_period("W-SUN")
    rows = []
    for w in np.unique(wk):
        m = wk == w
        tv, ov = v[m & t.values], v[m & ~t.values]
        ov = ov[np.isfinite(ov)]
        for d, x in tv.items():
            if np.isfinite(x) and len(ov) > 0:
                rows.append({"week": str(w), "date": d, "rel_log": float(x - ov.mean()), "n_other": int(len(ov))})
    return pd.DataFrame(rows, columns=["week", "date", "rel_log", "n_other"])


# ---------------------------------------------------------------------------
# VAR and connectedness on segmented samples
# ---------------------------------------------------------------------------


def _position_in_segment(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    start = np.r_[True, seg[1:] != seg[:-1]]
    idx = np.arange(len(seg))
    first = np.maximum.accumulate(np.where(start, idx, 0))
    return idx - first


def var_design(Y, seg, p: int, min_pos: int | None = None):
    """Regressors and targets for a VAR(p) using only rows whose p lags lie in the same segment.

    Segments must be contiguous runs of equal ids (a new id starts a new run).
    ``min_pos`` (default p) lets several lag orders be compared on one sample.
    """
    Y = np.asarray(Y, float)
    pos = _position_in_segment(seg)
    rows = np.where(pos >= (p if min_pos is None else min_pos))[0]
    X = np.hstack([np.ones((len(rows), 1))] + [Y[rows - l] for l in range(1, p + 1)])
    return X, Y[rows]


def fit_var_segments(Y, seg, p: int, min_pos: int | None = None) -> VarFit:
    """OLS VAR(p) whose lags never cross a segment boundary (for example the gap between two expiry days)."""
    X, Z = var_design(Y, seg, p, min_pos)
    N = Z.shape[1]
    B, *_ = np.linalg.lstsq(X, Z, rcond=None)
    U = Z - X @ B
    dof = max(len(Z) - X.shape[1], 1)
    Sigma = U.T @ U / dof
    A = np.stack([B[1 + (l - 1) * N : 1 + l * N].T for l in range(1, p + 1)])
    return VarFit(A=A, c=B[0], Sigma=Sigma, resid=U, p=p)


def var_lag_aic(Y, seg, pmax: int = 10) -> int:
    """VAR lag order with the smallest AIC, all orders fitted on the same rows."""
    best, best_p = np.inf, 1
    for p in range(1, pmax + 1):
        fit = fit_var_segments(Y, seg, p, min_pos=pmax)
        T, N = fit.resid.shape
        S = fit.resid.T @ fit.resid / T
        sign, logdet = np.linalg.slogdet(S)
        aic = logdet + 2 * p * N * N / T if sign > 0 else np.inf
        if aic < best:
            best, best_p = aic, p
    return best_p


def connectedness_segments(Y, seg, p: int, H: int) -> Connectedness:
    return connectedness_from_fit(fit_var_segments(Y, seg, p), H)


def segment_bootstrap_connectedness(Y, seg, p: int, H: int, n_boot: int = 200, seed: int = 0) -> np.ndarray:
    """Bootstrap of the paper-scale total (sum of off-diagonal shares) by resampling whole segments.

    A segment is a contiguous run of equal ids, for example one trading day,
    or one trading hour on one expiry day. Segments are drawn with
    replacement, as many as there are, and each keeps its own minute-by-minute
    dynamics; VAR lags never link two drawn segments. Resampling short blocks
    of minutes instead was tried first and biased the total downward, because
    conditional volatility is far more persistent than a 30-minute block.
    With only three or four segments (one hour on each monthly expiry) the
    interval is coarse, which honestly reflects how little data there is.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, float)
    seg = np.asarray(seg)
    starts = np.r_[0, np.where(seg[1:] != seg[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(seg)]
    pieces = [(a, b) for a, b in zip(starts, ends) if b - a > p + 1]
    K = len(pieces)
    if K == 0:
        return np.full(n_boot, np.nan)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, K, K)
        idx = np.concatenate([np.arange(*pieces[j]) for j in pick])
        bseg = np.concatenate([np.full(pieces[j][1] - pieces[j][0], n) for n, j in enumerate(pick)])
        out[i] = connectedness_segments(Y[idx], bseg, p, H).paper_scale_total
    return out


# ---------------------------------------------------------------------------
# Intraday GARCH persistence: covariance, delta method, profile likelihood
# ---------------------------------------------------------------------------


def half_life(persistence):
    """Minutes for a volatility shock to halve: ln(0.5) / ln(alpha + beta)."""
    P = np.asarray(persistence, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(P < 1, np.log(0.5) / np.log(P), np.inf)


def garch11_cov(x, params: dict, weights=None, dist: str = "t", eps: float = 1e-4):
    """Inverse observed information for (omega, alpha, beta[, nu]) from a numerical Hessian.

    Same construction as ``volatility.garch11_tstats``, which returns only z
    statistics; the full matrix is needed for the standard error of alpha + beta.
    """
    x = np.asarray(x, float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    x2 = x * x
    q0 = params["q0"]
    names = ["omega", "alpha", "beta"] + (["nu"] if dist == "t" else [])
    p0 = np.array([params[n] for n in names])

    def ll(p):
        q2 = _garch_q2(x2, p[0], p[1], p[2], q0)
        out = _t_loglik(x, q2, p[3]) if dist == "t" else _norm_loglik(x, q2)
        return np.sum(w * out)

    k = len(p0)
    Hm = np.zeros((k, k))
    step = eps * np.maximum(np.abs(p0), 1e-3)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = step[i]
            ej[j] = step[j]
            Hm[i, j] = Hm[j, i] = (ll(p0 + ei + ej) - ll(p0 + ei - ej) - ll(p0 - ei + ej) + ll(p0 - ei - ej)) / (
                4 * step[i] * step[j]
            )
    try:
        cov = np.linalg.inv(-Hm)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    return names, cov


def persistence_delta(params: dict, names, cov) -> dict:
    """alpha + beta and the half-life with delta-method standard errors."""
    ia, ib = names.index("alpha"), names.index("beta")
    P = params["alpha"] + params["beta"]
    var = cov[ia, ia] + cov[ib, ib] + 2 * cov[ia, ib]
    se_P = float(np.sqrt(var)) if var > 0 else np.nan
    hl = float(half_life(P))
    dhl = -np.log(0.5) / (P * np.log(P) ** 2) if P < 1 else np.nan
    return {"persistence": P, "se_persistence": se_P, "half_life": hl, "se_half_life_delta": abs(dhl) * se_P}


def _fit_fixed_persistence(x, x2, w, P, q0, start, dist):
    def nll(th):
        omega = np.exp(th[0])
        share = 1.0 / (1.0 + np.exp(-th[1]))
        nu = 2.05 + np.exp(th[2]) if dist == "t" else np.inf
        q2 = _garch_q2(x2, omega, P * share, P * (1 - share), q0)
        ll = _t_loglik(x, q2, nu) if dist == "t" else _norm_loglik(x, q2)
        v = -np.sum(w * ll)
        return v if np.isfinite(v) else 1e300

    res = minimize(nll, start, method="L-BFGS-B")
    return -res.fun, res.x


def profile_half_life(x, params: dict, hl_grid, weights=None, dist: str = "t") -> pd.DataFrame:
    """Profile log-likelihood of the intraday GARCH over a grid of half-lives.

    At each half-life HL the persistence is fixed at 0.5 ** (1 / HL) and the
    likelihood is maximised over omega, the alpha share and nu. Starting values
    keep the fitted unconditional variance and alpha share.
    """
    x = np.asarray(x, float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    x2 = x * x
    q0 = params["q0"]
    P_hat = params["alpha"] + params["beta"]
    qbar = params["omega"] / max(1e-12, 1 - P_hat)
    share = params["alpha"] / P_hat
    nu0 = np.log(max(params["nu"] - 2.05, 1e-3)) if dist == "t" else 0.0
    rows = []
    for hl in hl_grid:
        P = 0.5 ** (1.0 / hl)
        start = np.array([np.log(qbar * (1 - P)), np.log(share / (1 - share)), nu0])
        ll, _ = _fit_fixed_persistence(x, x2, w, P, q0, start, dist)
        rows.append({"half_life": float(hl), "persistence": P, "loglik": ll})
    return pd.DataFrame(rows)


def profile_ci(prof: pd.DataFrame, ll_max: float, level: float = 0.95) -> dict:
    """Likelihood-ratio interval for the half-life from a profile on a grid.

    The interval is the set of grid half-lives whose LR statistic is below the
    chi-square(1) critical value, with linear interpolation at the edges. If
    the set reaches the end of the grid the bound is reported as open.
    """
    crit = chi2.ppf(level, 1)
    pr = prof.sort_values("half_life").reset_index(drop=True)
    lr = 2 * (max(ll_max, pr["loglik"].max()) - pr["loglik"].to_numpy())
    hl = np.log(pr["half_life"].to_numpy())
    inside = lr <= crit
    if not inside.any():
        return {"lo": np.nan, "hi": np.nan, "lo_open": False, "hi_open": False}
    i0, i1 = np.where(inside)[0][[0, -1]]

    def interp(a, b):
        if lr[a] == lr[b]:
            return np.exp(hl[a])
        f = (crit - lr[a]) / (lr[b] - lr[a])
        return float(np.exp(hl[a] + f * (hl[b] - hl[a])))

    lo = interp(i0 - 1, i0) if i0 > 0 else float(np.exp(hl[0]))
    hi = interp(i1, i1 + 1) if i1 < len(hl) - 1 else float(np.exp(hl[-1]))
    return {"lo": lo, "hi": hi, "lo_open": i0 == 0, "hi_open": i1 == len(hl) - 1}
