"""Trying to break two in-sample findings, plus a quasi-causal gamma roll-off test. REAL DATA.

F1 (experiments/real_day_level.py, r3_regressions; docs/REAL_DATA_RESULTS.md R3): today's log realized
    variance on yesterday's log RV, its 5-day mean, log VIX, an expiry dummy, yesterday's net GEX (scaled by
    the 20-day median gross GEX) and GEX x expiry, with regime x expiry controls, Newey-West (5 lags).
    Claim: GEX < 0 coefficient on ordinary days, positive interaction on expiry days.
    After an independent review, F1's main GEX measure here is the scale-free share net GEX / gross GEX of the
    same chain; the original ratio (net / 20-day median gross, about four times larger on expiry eves) is kept
    as the secondary version. Every F1 table has a "measure" column ("share" or "ratio").
F2 (experiments/momentum_gex.py; R5): last-30-minute return on the rest-of-day return, with an interaction
    with 1[GEX_{t-1} < 0]. Claim: under the Pro rule the end-of-day reversal is concentrated on negative-GEX days.

Tests (numbering as in docs/FALSIFICATION.md):
    T1  permutation placebos (shuffle GEX within regime, within 20-day blocks, within regime x expiry status,
        circular shift within regime, random re-signing), 2,000 draws each
    T2  timing placebos (GEX published on day t, t+1, t+2 instead of t-1; horse race of t-1 against t+1)
    T3  placebo sign rules (dealer weights for calls and puts drawn from [-1, 1]^2, a grid, and every angle),
        gross GEX and log total open interest as regressors
    T4  out-of-sample split (2023 to 2024 against 2025 onward, both directions) and a pre-registered style
        test of F2 on the later period only
    T5  Bank Nifty as a second market (same machinery: build_market_from_nse and build_gex_panel)
    T6  quasi-causal gamma roll-off test: the expiring contract's gamma leaves the market on expiry day for
        a calendar reason; real expiry days against pseudo-expiry days; an instrumental-variable version
    T7  multiple testing (Holm and Benjamini-Hochberg)

Closing auction: from 3 Aug 2026 NSE sets the official close in a closing auction; in the minute bars the index
is frozen from 15:16 to 15:28 and the 15:29 bar jumps to the auction price. Every headline test is therefore
run on two samples: "full" (2 Jan 2023 to 2 Nov 2026, with the auction period as its own regime wherever regime
controls or regime groups are used) and "pre_auction" (2 Jan 2023 to 31 Jul 2026).

    OMP_NUM_THREADS=1 python experiments/falsification.py            # about 5 to 10 minutes on 4 cores
    OMP_NUM_THREADS=1 python experiments/falsification.py --perms 100 --skip-banknifty   # quick check

Writes results/falsification/*.csv, results/falsification/run_info.json and results/falsification_stdout.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import momentum_gex as mg  # noqa: E402
import real_day_level as rdl  # noqa: E402

from niftygex import gex, greeks  # noqa: E402
from niftygex.data import (_participant_weights_row, bhavcopy_files, build_market_from_nse,  # noqa: E402
                           holiday_adjusted_expiries, load_bhavcopy, load_participant_oi, parity_forward)
from niftygex.features import build_gex_panel  # noqa: E402

SEED = 20261102
LAGS = 5
R_RATE = 0.065
RULES = dict(rdl.RULES)  # us_convention / participant_pro / client_mirror -> panel column
AUCTION_START = pd.Timestamp("2026-08-03")
PRE_AUCTION_END = pd.Timestamp("2026-07-31")
SPLIT_A_END = pd.Timestamp("2024-12-31")
REGIMES4 = ["multi_weekly", "single_weekly_thu", "single_weekly_tue", "closing_auction"]
THURSDAY_REGIMES = ("multi_weekly", "single_weekly_thu")
SAMPLES = {"full": None, "pre_auction": PRE_AUCTION_END}
SCHEMES = ("shuffle_within_regime", "shuffle_within_20day_blocks", "shuffle_within_regime_x_expiry",
           "circular_shift_within_regime", "random_resign")
# F1's GEX regressor. "share" (net / gross GEX of the same chain, in [-1, 1]) is scale-free and is the main
# version; "ratio" (net / 20-day median gross GEX) is the original R3 measure, kept as secondary. The ratio is
# mechanically about four times larger on expiry eves (the front chain's gamma peaks the day before expiry).
F1_MEASURES = {"share": "net GEX / gross GEX of the same chain (main, scale-free)",
               "ratio": "net GEX / 20-day median gross GEX (original R3)"}
F2_MEASURE = "sign of net GEX (scale irrelevant)"
F1_STATS = ("coef_gex_l1", "t_gex_l1", "coef_gex_l1_x_exp", "t_gex_l1_x_exp")
F2_STATS = ("coef_x_neg_binary", "t_x_neg_binary", "coef_x_neg_controls", "t_x_neg_controls")


# ---------------------------------------------------------------------------
# Small statistical helpers (tested in tests/test_falsification.py)
# ---------------------------------------------------------------------------


def _nw_meat(s: np.ndarray, lags: int) -> np.ndarray:
    S = s.T @ s
    for lag in range(1, min(lags, len(s) - 1) + 1):
        G = s[lag:].T @ s[:-lag]
        S += (1.0 - lag / (lags + 1.0)) * (G + G.T)
    return S


def ols_hac(y, X, lags: int = LAGS):
    """OLS with Newey-West (Bartlett) covariance and normal p-values.

    Matches statsmodels OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": lags}) (no small-sample
    correction), which is what real_day_level.py and momentum_gex.py use. X must include the constant.
    Returns (coef, se, t, p, r2).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    V = XtX_inv @ _nw_meat(X * u[:, None], lags) @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = b / se
    p = 2.0 * stats.norm.sf(np.abs(t))
    yc = y - y.mean()
    r2 = 1.0 - (u @ u) / (yc @ yc) if (yc @ yc) > 0 else np.nan
    return b, se, t, p, r2


def iv_2sls_hac(y, exog, endog, inst, lags: int = LAGS) -> dict:
    """Just-identified 2SLS with Newey-West covariance. ``exog`` includes the constant.

    Returns the IV coefficient, its HAC standard error and p-value, the first-stage coefficient on the
    instrument with its HAC F statistic (t squared), and the reduced-form coefficient with its p-value.
    """
    y, exog, endog, inst = (np.asarray(a, float) for a in (y, exog, endog, inst))
    W = np.column_stack([exog, inst])
    b1, _, t1, _, r2_1 = ols_hac(endog, W, lags)
    xhat = W @ b1
    Xh = np.column_stack([exog, xhat])
    X = np.column_stack([exog, endog])
    A = np.linalg.inv(Xh.T @ Xh)
    beta = A @ (Xh.T @ y)
    u = y - X @ beta
    V = A @ _nw_meat(Xh * u[:, None], lags) @ A
    se = float(np.sqrt(V[-1, -1]))
    brf, _, trf, prf, _ = ols_hac(y, W, lags)
    return {"beta_iv": float(beta[-1]), "se_iv": se, "p_iv": float(2 * stats.norm.sf(abs(beta[-1] / se))),
            "first_stage_coef": float(b1[-1]), "first_stage_F_hac": float(t1[-1] ** 2), "first_stage_r2": float(r2_1),
            "reduced_form_coef": float(brf[-1]), "reduced_form_p": float(prf[-1]), "n": int(len(y))}


def permute_within_groups(x, groups, rng) -> np.ndarray:
    """Return a copy of ``x`` with values shuffled among positions that share a group label."""
    x = np.asarray(x)
    out = x.copy()
    groups = np.asarray(groups)
    for gval in pd.unique(groups):
        idx = np.flatnonzero(groups == gval)
        out[idx] = x[rng.permutation(idx)]
    return out


def circular_shift_within_groups(x, groups, rng) -> np.ndarray:
    """Rotate ``x`` by a random non-zero offset inside each group (keeps the series' own autocorrelation)."""
    x = np.asarray(x)
    out = x.copy()
    groups = np.asarray(groups)
    for gval in pd.unique(groups):
        idx = np.flatnonzero(groups == gval)
        if len(idx) > 1:
            out[idx] = np.roll(x[idx], int(rng.integers(1, len(idx))))
    return out


def block_ids(n: int, size: int = 20) -> np.ndarray:
    """Consecutive blocks of ``size`` observations: 0,0,...,0,1,1,..."""
    return np.arange(n) // size


def perm_pvalues(real_t: float, draws_t: np.ndarray, real_b: float, draws_b: np.ndarray) -> dict:
    """Permutation p-values with the +1 correction (Phipson and Smyth 2010)."""
    t = draws_t[np.isfinite(draws_t)]
    b = draws_b[np.isfinite(draws_b)]
    n = len(t)
    if n == 0 or not np.isfinite(real_t):
        return {"n_valid": n, "p_two_sided_t": np.nan, "p_one_sided_t": np.nan, "p_two_sided_coef": np.nan}
    one = (np.sum(t <= real_t) if real_t < 0 else np.sum(t >= real_t))
    return {"n_valid": n, "p_two_sided_t": (1 + np.sum(np.abs(t) >= abs(real_t))) / (n + 1),
            "p_one_sided_t": (1 + one) / (n + 1),
            "p_two_sided_coef": (1 + np.sum(np.abs(b) >= abs(real_b))) / (len(b) + 1)}


def shifted_placebo_dates(days, expiry_days, shift: int, exclude_adjacent: bool = True) -> pd.DatetimeIndex:
    """Pseudo-expiry days: each real expiry day moved by ``shift`` trading days.

    A pseudo day p is dropped if p itself is a real expiry day, and (with ``exclude_adjacent``) if the
    trading day before or after p is one, so that neither the pre-event day nor the outcome day is an
    expiry day. Pseudo days without a trading day on both sides are dropped.
    """
    days = pd.DatetimeIndex(days)
    exp = set(pd.DatetimeIndex(expiry_days))
    out = set()
    for i in np.flatnonzero(days.isin(list(exp))):
        j = i + shift
        if j < 1 or j >= len(days) - 1:
            continue
        p = days[j]
        if p in exp:
            continue
        if exclude_adjacent and (days[j - 1] in exp or days[j + 1] in exp):
            continue
        out.add(p)
    return pd.DatetimeIndex(sorted(out))


def weekday_placebo_dates(days, expiry_days, regime) -> pd.DatetimeIndex:
    """Pseudo-expiry days on the other regime's expiry weekday.

    Tuesdays while Nifty expired on Thursday, Thursdays once it expired on Tuesday. Days that are real
    expiry days, or next to one, are dropped (as in shifted_placebo_dates).
    """
    days = pd.DatetimeIndex(days)
    regime = pd.Series(regime, index=days)
    exp = set(pd.DatetimeIndex(expiry_days))
    out = []
    for j in range(1, len(days) - 1):
        p = days[j]
        want = 1 if regime.iloc[j] in THURSDAY_REGIMES else 3
        if p.weekday() != want or p in exp or days[j - 1] in exp or days[j + 1] in exp:
            continue
        out.append(p)
    return pd.DatetimeIndex(out)


def split_at(tab: pd.DataFrame, days, event_days) -> pd.DataFrame:
    """Split the gamma book around each event day p (previous trading day q).

    ``tab``: one row per (date, expiry) with columns date, expiry, C, P (gross call and put GEX of the
    contracts with that expiry, published at the close of ``date``) and optionally oi.
    For each p: ``front`` = contracts with the nearest expiry on or after p, published at q (on a real
    expiry day these expire on p: the gamma that is about to vanish); ``rest`` = all later expiries at q;
    ``post`` = all contracts expiring after p, published at the close of p (the book that survives p).
    """
    days = pd.DatetimeIndex(days)
    pos = {d: i for i, d in enumerate(days)}
    by_date = {d: g for d, g in tab.groupby("date")}
    has_oi = "oi" in tab.columns
    rows = []
    for p in pd.DatetimeIndex(event_days):
        i = pos.get(p)
        if i is None or i == 0:
            continue
        q = days[i - 1]
        gq, gp = by_date.get(q), by_date.get(p)
        if gq is None or gp is None:
            continue
        live = gq[gq["expiry"] >= p]
        if live.empty:
            continue
        front = live["expiry"].min()
        f, rest, post = live[live["expiry"] == front], live[live["expiry"] > front], gp[gp["expiry"] > p]
        row = {"event_day": p, "prev_day": q, "front_expiry": front, "front_expires_on_event_day": bool(front == p),
               "front_C": f["C"].sum(), "front_P": f["P"].sum(), "rest_C": rest["C"].sum(), "rest_P": rest["P"].sum(),
               "post_C": post["C"].sum(), "post_P": post["P"].sum()}
        if has_oi:
            row.update({"front_oi": f["oi"].sum(), "rest_oi": rest["oi"].sum()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("event_day") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def regime4(dm: pd.DataFrame) -> pd.Series:
    """The three expiry regimes, plus the closing-auction period (from 3 Aug 2026) as a fourth."""
    r = dm["regime"].astype(str).copy()
    r[dm.index >= AUCTION_START] = "closing_auction"
    return r


def participant_weights(pdir: Path, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily Pro and Client net shares of index call and put OI, read and forward-filled as in data.py."""
    rows = {}
    for f in sorted(Path(pdir).glob("*.csv")):
        try:
            d = pd.to_datetime(f.stem[-8:], format="%d%m%Y")
        except ValueError:
            continue
        rows[d] = _participant_weights_row(load_participant_oi(f))
    return pd.DataFrame.from_dict(rows, orient="index").sort_index().reindex(days).ffill()


def rule_weights(pw: pd.DataFrame) -> dict:
    """(w_call, w_put) per publish date for each sign rule, plus the gross (long_all) bound."""
    idx = pw.index
    one = pd.Series(1.0, index=idx)
    return {"us_convention": (one, -one), "participant_pro": (pw["pro_call"], pw["pro_put"]),
            "client_mirror": (-pw["client_call"], -pw["client_put"]), "gross": (one, one)}


def _file_date(path) -> pd.Timestamp:
    return pd.to_datetime("".join(ch for ch in Path(path).name if ch.isdigit())[-8:], format="%Y%m%d")


def _load_task(args):
    path, symbol = args
    return _file_date(path), load_bhavcopy(path, symbol)


_G: dict = {}


def _init_expiry_worker(pos, eff):
    _G["pos"], _G["eff"] = pos, eff


def expiry_gex_rows(df: pd.DataFrame, d: pd.Timestamp, spot: float, pos: dict, eff: dict, r: float = R_RATE) -> list:
    """Gross call and put GEX of every expiry listed after ``d`` (same steps as build_market_from_nse and
    build_gex_panel: parity forward, Black-76 IV from settlement prices, neighbouring-strike fill, OI > 0,
    gex.strike_gex with lot size 1 because OI is in shares)."""
    df = df.copy()
    listed = pd.to_datetime(df["expiry"]).dt.normalize()
    df["expiry"] = listed.map(lambda e: eff.get(e, e))
    df["remapped"] = df["expiry"] != listed
    rows = []
    for e, sel in df[df["expiry"] > d].groupby("expiry", sort=True):
        n_td = pos[e] - pos[d] if (e in pos and d in pos) else int(np.busday_count(d.date(), pd.Timestamp(e).date()))
        T = max(n_td, 1) / 252
        ch = sel[["strike", "is_call", "oi", "settle"]].reset_index(drop=True)
        fwd = parity_forward(ch, T, r)
        if np.isfinite(fwd) and abs(fwd / spot - 1) < 0.05:
            F = fwd
        else:
            under = sel["underlying"].dropna()
            F = float(under.iloc[0]) if len(under) else spot
        c = ch.copy()
        c["iv"] = greeks.implied_vol(c["settle"].to_numpy(), F, c["strike"].to_numpy(), T, r, c["is_call"].to_numpy())
        c["iv"] = c.groupby("is_call")["iv"].transform(lambda s: s.interpolate(limit_direction="both"))
        c = c[c["oi"] > 0]
        row = {"date": d, "expiry": pd.Timestamp(e), "n_td": n_td, "T": T, "forward": F, "remapped": bool(sel["remapped"].any()),
               "C": 0.0, "P": 0.0, "oi_call": 0.0, "oi_put": 0.0, "n_series": len(c), "n_gamma_nan": 0}
        if len(c):
            t = gex.strike_gex(c, F, 1.0, T, r, 0.0, "long_all")
            gg = t["gross_gex"].to_numpy(float)
            isc = t["is_call"].to_numpy(bool)
            row.update({"C": float(np.nansum(gg[isc])), "P": float(np.nansum(gg[~isc])),
                        "oi_call": float(c.loc[c["is_call"], "oi"].sum()), "oi_put": float(c.loc[~c["is_call"], "oi"].sum()),
                        "n_gamma_nan": int((~np.isfinite(gg)).sum())})
        rows.append(row)
    return rows


def _expiry_task(args):
    d, df, spot = args
    return expiry_gex_rows(df, d, spot, _G["pos"], _G["eff"])


def chain_table(raw: Path, symbol: str, days: pd.DatetimeIndex, close: pd.Series, workers: int) -> pd.DataFrame:
    """Per (publish date, expiry) gross call and put GEX for every listed expiry of ``symbol``."""
    files = [f for f in bhavcopy_files(raw / "bhavcopy")
             if days.min() - pd.Timedelta(days=10) <= _file_date(f) <= days.max() + pd.Timedelta(days=10)]
    ctx = _mp_context()
    with ctx.Pool(workers) as pool:
        loaded = pool.map(_load_task, [(f, symbol) for f in files], chunksize=16)
    bhav = {d: df for d, df in loaded if len(df)}
    eff = holiday_adjusted_expiries({d: df[["expiry"]] for d, df in bhav.items()})
    pos = {d: i for i, d in enumerate(days)}
    tasks = [(d, bhav[d], float(close[d])) for d in days if d in bhav]
    with ctx.Pool(workers, initializer=_init_expiry_worker, initargs=(pos, eff)) as pool:
        parts = pool.map(_expiry_task, tasks, chunksize=8)
    tab = pd.DataFrame([r for part in parts for r in part])
    tab["oi"] = tab["oi_call"] + tab["oi_put"]
    tab.attrs["remapped_expiries"] = {str(k.date()): str(v.date()) for k, v in eff.items()}
    return tab


def front_from_table(tab: pd.DataFrame) -> pd.DataFrame:
    """The nearest expiry strictly after each publish date (the chain the daily panel uses)."""
    first = tab.sort_values(["date", "expiry"]).groupby("date").head(1).set_index("date")
    tot = tab.groupby("date")[["C", "P", "oi"]].sum().add_prefix("all_")
    return first[["expiry", "C", "P", "oi", "remapped"]].join(tot)


def _mp_context():
    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover
        return mp.get_context()


# ---------------------------------------------------------------------------
# F1 and F2 designs (same construction as real_day_level.py and momentum_gex.py)
# ---------------------------------------------------------------------------


def f1_frame(dm: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    d = dm.join(panel, how="left")
    d["regime4"] = regime4(dm)
    d["y"] = np.log(d["rv_day"])
    d["y_l1"] = d["y"].shift(1)
    d["y_w"] = d["y"].shift(1).rolling(5).mean()
    d["lvix_l1"] = np.log(d["vix"]).shift(1)
    d["exp"] = d["is_expiry"].astype(float)
    d["scale"] = d["gross_gex"].rolling(20, min_periods=5).median()
    return d


def f1_controls(d: pd.DataFrame, regime_col: str = "regime4", regimes: bool = True) -> pd.DataFrame:
    X = d[["y_l1", "y_w", "lvix_l1", "exp"]].copy()
    if regimes:
        for reg in REGIMES4[1:]:
            ind = (d[regime_col] == reg).astype(float)
            X[f"reg_{reg}"] = ind
            X[f"exp_x_{reg}"] = ind * d["exp"]
    return X


def independent_columns(A: np.ndarray) -> list:
    """Indices of columns kept by a left-to-right pass that drops any column that is a linear combination of
    the columns kept before it (for example a full set of regime dummies next to the constant)."""
    if np.linalg.matrix_rank(A) == A.shape[1]:
        return list(range(A.shape[1]))
    keep: list = []
    for j in range(A.shape[1]):
        if np.linalg.matrix_rank(A[:, keep + [j]]) == len(keep) + 1:
            keep.append(j)
    return keep


def fit_named(y: pd.Series, X: pd.DataFrame, lags: int = LAGS) -> pd.DataFrame:
    """OLS + HAC on named columns. Rows with missing values are dropped, and so are constant columns and
    columns that are exact linear combinations of earlier ones (put the terms of interest last)."""
    X = X.astype(float)
    ok = X.notna().all(axis=1) & y.notna()
    Xo = X[ok]
    keep = [c for c in Xo.columns if Xo[c].std() > 0]
    A = np.column_stack([np.ones(int(ok.sum()))] + [Xo[c].to_numpy() for c in keep])
    idx = independent_columns(A)
    keep = [keep[j - 1] for j in idx if j > 0]
    A = A[:, idx]
    b, se, t, p, r2 = ols_hac(y[ok].to_numpy(float), A, lags)
    res = pd.DataFrame({"coef": b, "se_hac": se, "t": t, "p": p}, index=pd.Index(["const"] + keep, name="term"))
    res["n"] = int(ok.sum())
    res["r2"] = r2
    return res


def get(res: pd.DataFrame, term: str, field: str = "coef") -> float:
    return float(res.loc[term, field]) if term in res.index else np.nan


def calendar_controls(d: pd.DataFrame) -> pd.DataFrame:
    """Dummies for trading days to the next expiry (1, 2, 3; base 4 or more; 0 is the expiry dummy) and for
    the weekday of day t (base Monday). Both are known from the calendar before the day starts."""
    out = pd.DataFrame(index=d.index)
    tdte = d["trading_days_to_expiry"]
    for k in (1, 2, 3):
        out[f"tdte_{k}"] = (tdte == k).astype(float)
    for k in range(1, 5):
        out[f"weekday_{k}"] = (d["weekday"] == k).astype(float)
    return out


def f1_fit(d, g, mask, regimes=True, extra: dict | None = None, calendar: bool = False) -> pd.DataFrame:
    X = f1_controls(d, regimes=regimes)
    if calendar:
        X = X.join(calendar_controls(d))
    X["gex_l1"] = g
    X["gex_l1_x_exp"] = g * d["exp"]
    for k, v in (extra or {}).items():
        X[k] = v
    return fit_named(d.loc[mask, "y"], X[mask])


def f2_frame(dm: pd.DataFrame, bars: Path) -> pd.DataFrame:
    d = dm.join(mg.daily_returns(bars, dm.index))
    d["regime4"] = regime4(dm)
    d = d.dropna(subset=["r_rod", "r_last"])
    d["r_rod_bp"], d["r_last_bp"] = d["r_rod"] * 1e4, d["r_last"] * 1e4
    d["lvix_prev"] = np.log(d["vix"]).shift(1)
    d["lvix_prev"] -= d["lvix_prev"].mean()
    return d


def f2_X(d: pd.DataFrame, g: pd.Series, spec: str, regime_col: str = "regime4", missing_as_nonneg: bool = True,
         prefix: str = "") -> pd.DataFrame:
    """F2 design. ``missing_as_nonneg`` codes days with unknown GEX as non-negative, exactly as momentum_gex.py
    does ((g < 0) is False for NaN); timing placebos switch it off so that unknown lead GEX drops the day."""
    r = d["r_rod_bp"]
    neg = (g < 0).astype(float)
    if not missing_as_nonneg:
        neg = neg.where(g.notna())
    if spec == "continuous":
        return pd.DataFrame({"r_rod": r, f"{prefix}gex": g, f"{prefix}r_rod_x_gex": r * g})
    X = pd.DataFrame({"r_rod": r, f"{prefix}neg": neg, f"{prefix}r_rod_x_neg": r * neg})
    if spec == "controls":
        X["lvix_prev"] = d["lvix_prev"]
        X["r_rod_x_lvix"] = r * d["lvix_prev"]
        X["r_rod_x_abs_r_rod"] = r * r.abs()
    elif spec == "regime_slopes":
        for reg in ("single_weekly_thu", "single_weekly_tue", "closing_auction"):
            ind = (d[regime_col] == reg).astype(float)
            X[f"reg_{reg}"] = ind
            X[f"r_rod_x_{reg}"] = r * ind
    return X


def f2_fit(d, g, spec, mask, regime_col="regime4", missing_as_nonneg=True) -> pd.DataFrame:
    X = f2_X(d, g, spec, regime_col, missing_as_nonneg)
    return fit_named(d.loc[mask, "r_last_bp"], X[mask])


# ---------------------------------------------------------------------------
# Permutation workers
# ---------------------------------------------------------------------------


def _draw(scheme: str, g: np.ndarray, groups, rng) -> np.ndarray:
    if scheme == "random_resign":
        return g * rng.choice(np.array([-1.0, 1.0]), size=len(g))
    if scheme == "circular_shift_within_regime":
        return circular_shift_within_groups(g, groups, rng)
    return permute_within_groups(g, groups, rng)


def f1_stat(pk: dict, g: np.ndarray) -> tuple:
    X = np.column_stack([pk["A"], g, g * pk["e"]])
    b, _, t, _, _ = ols_hac(pk["y"], X)
    return b[-2], t[-2], b[-1], t[-1]


def f2_stat(pk: dict, g: np.ndarray) -> tuple:
    neg = (g < 0).astype(float)  # NaN -> 0, as in momentum_gex.py
    if neg.sum() < 5 or (1 - neg).sum() < 5:
        return (np.nan,) * 4
    r = pk["r"]
    X = np.column_stack([np.ones(len(r)), r, neg, r * neg])
    b, _, t, _, _ = ols_hac(pk["y"], X)
    m = pk["zmask"]
    b2, _, t2, _, _ = ols_hac(pk["y"][m], np.column_stack([X[m], pk["Z"][m]]))
    return b[3], t[3], b2[3], t2[3]


def _perm_task(task):
    key, finding, scheme, seed, B, pk = task
    rng = np.random.default_rng(seed)
    fn = f1_stat if finding == "F1" else f2_stat
    groups = pk["groups"].get(scheme)
    out = np.empty((B, 4))
    for b in range(B):
        out[b] = fn(pk, _draw(scheme, pk["g"], groups, rng))
    return key, out


def _groups(regime: np.ndarray, exp: np.ndarray) -> dict:
    return {"shuffle_within_regime": regime, "circular_shift_within_regime": regime,
            "shuffle_within_20day_blocks": block_ids(len(regime), 20),
            "shuffle_within_regime_x_expiry": np.array([f"{a}|{int(b)}" for a, b in zip(regime, exp)])}


# ---------------------------------------------------------------------------
# Registry of every test reported (for T7)
# ---------------------------------------------------------------------------


class Registry:
    def __init__(self):
        self.rows: list = []

    def add(self, test: str, sample: str, finding: str, rule: str, detail: str, p: float, stat: float = np.nan):
        if p is None or not np.isfinite(p):
            return
        self.rows.append({"test": test, "sample": sample, "finding": finding, "rule": rule, "detail": detail,
                          "statistic": stat, "p": float(p)})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def adjust(p: pd.Series) -> pd.DataFrame:
    p = p.astype(float)
    return pd.DataFrame({"p_holm": multipletests(p, method="holm")[1], "p_bh": multipletests(p, method="fdr_bh")[1]},
                        index=p.index)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


def sample_mask(index: pd.DatetimeIndex, end) -> np.ndarray:
    return np.ones(len(index), bool) if end is None else (index <= end)


def original_family(dm, panel, d2, gs2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every GEX-term p-value the two original scripts compute (the 'garden' of the original study)."""
    r3 = rdl.r3_regressions(dm, panel)
    f1 = r3[r3["term"].isin(["gex_l1", "gex_l1_x_exp"])].assign(finding="F1", rule=lambda x: x["spec"].str[4:])
    f1 = f1.rename(columns={"controls": "spec_detail"})[["finding", "rule", "spec_detail", "term", "coef", "p", "n_days"]]
    f1 = f1.rename(columns={"n_days": "n"})
    rows = []
    subsets = {"all": np.ones(len(d2), bool), "non_expiry": ~d2["is_expiry"].to_numpy(bool),
               "expiry": d2["is_expiry"].to_numpy(bool)}
    for rule in RULES:
        g = gs2[rule].shift(1).reindex(d2.index)
        for sname, m in subsets.items():
            for spec, term in (("binary", "r_rod_x_neg"), ("regime_slopes", "r_rod_x_neg"),
                               ("controls", "r_rod_x_neg"), ("continuous", "r_rod_x_gex")):
                X = f2_X(d2, g, spec, regime_col="regime")
                if "neg" in X and X.loc[m, "neg"].nunique() < 2:
                    continue
                res = fit_named(d2.loc[m, "r_last_bp"], X[m])
                rows.append({"finding": "F2", "rule": rule, "spec_detail": f"{spec}, {sname} days", "term": term,
                             "coef": get(res, term), "p": get(res, term, "p"), "n": int(res["n"].iloc[0])})
    return f1.reset_index(drop=True), pd.DataFrame(rows)


def t1_permutations(d1, f1m, d2, gs2, B, workers, reg: Registry):
    tasks, real = [], {}
    i = 0
    for sname, end in SAMPLES.items():
        for rule in RULES:
            pk1s = {}
            for mname, gs1 in f1m.items():
                # F1 sample: rows with every regressor present
                g1 = gs1[rule].shift(1)
                m1 = sample_mask(d1.index, end)
                Xc = f1_controls(d1)
                ok = m1 & Xc.notna().all(axis=1).to_numpy() & g1.notna().to_numpy() & d1["y"].notna().to_numpy()
                Xc = Xc[ok]
                keep = [c for c in Xc.columns if Xc[c].std() > 0]
                pk1s[mname] = {"y": d1.loc[ok, "y"].to_numpy(float),
                               "A": np.column_stack([np.ones(ok.sum()), Xc[keep].to_numpy(float)]),
                               "e": d1.loc[ok, "exp"].to_numpy(float), "g": g1[ok].to_numpy(float),
                               "groups": _groups(d1.loc[ok, "regime4"].to_numpy(), d1.loc[ok, "exp"].to_numpy())}
                real[(sname, "F1", rule, mname)] = (f1_stat(pk1s[mname], pk1s[mname]["g"]), int(ok.sum()))
            # F2 sample: every day with returns; GEX missing on 3 early days is coded non-negative, as in the original
            m2 = sample_mask(d2.index, end)
            dd = d2[m2]
            g2 = gs2[rule].shift(1).reindex(d2.index)[m2]
            r = dd["r_rod_bp"].to_numpy(float)
            Z = np.column_stack([dd["lvix_prev"], r * dd["lvix_prev"], r * np.abs(r)])
            pk2 = {"y": dd["r_last_bp"].to_numpy(float), "r": r, "g": g2.to_numpy(float), "Z": Z,
                   "zmask": np.isfinite(Z).all(axis=1),
                   "groups": _groups(dd["regime4"].to_numpy(), dd["is_expiry"].to_numpy(bool))}
            real[(sname, "F2", rule, F2_MEASURE)] = (f2_stat(pk2, pk2["g"]), int(len(dd)))
            for scheme in SCHEMES:
                for finding, mname, pk in [("F1", m, pk1s[m]) for m in f1m] + [("F2", F2_MEASURE, pk2)]:
                    tasks.append(((sname, finding, rule, scheme, mname, i), finding, scheme, [SEED, i], B, pk))
                    i += 1
    with _mp_context().Pool(workers) as pool:
        results = pool.map(_perm_task, tasks, chunksize=1)
    summ, draws = [], []
    for (sname, finding, rule, scheme, mname, i), out in results:
        names = F1_STATS if finding == "F1" else F2_STATS
        (vals, n) = real[(sname, finding, rule, mname)]
        df = pd.DataFrame(out, columns=names)
        df.insert(0, "draw", np.arange(len(df)))
        for k, v in (("sample", sname), ("finding", finding), ("measure", mname), ("rule", rule), ("scheme", scheme),
                     ("seed", f"[{SEED}, {i}]")):
            df.insert(0, k, v)
        draws.append(df)
        for j in (0, 2):
            coef_name, t_name = names[j], names[j + 1]
            pv = perm_pvalues(vals[j + 1], out[:, j + 1], vals[j], out[:, j])
            term = coef_name.replace("coef_", "")
            summ.append({"sample": sname, "finding": finding, "measure": mname, "rule": rule, "term": term, "scheme": scheme,
                         "seed": f"[{SEED}, {i}]", "draws": B, "n_obs": n, "real_coef": vals[j], "real_t": vals[j + 1],
                         "perm_coef_mean": np.nanmean(out[:, j]), "perm_coef_sd": np.nanstd(out[:, j]),
                         "perm_coef_q025": np.nanpercentile(out[:, j], 2.5), "perm_coef_q975": np.nanpercentile(out[:, j], 97.5),
                         "perm_t_sd": np.nanstd(out[:, j + 1]), **pv})
            reg.add("T1 permutation", sname, finding, rule, f"{term}, {scheme}, {mname}", pv["p_two_sided_t"], vals[j + 1])
    return pd.DataFrame(summ), pd.concat(draws, ignore_index=True)


def t2_timing(d1, f1m, d2, gs2, reg: Registry) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    offsets = {-1: "t-1 (real)", 0: "t (same evening)", 1: "t+1 (lead)", 2: "t+2 (lead)"}
    for sname, end in SAMPLES.items():
        m1, m2 = sample_mask(d1.index, end), sample_mask(d2.index, end)
        for rule in RULES:
            for off, lab in offsets.items():
                for mname, gs1 in f1m.items():
                    g = gs1[rule].shift(-off)
                    res = f1_fit(d1, g, m1)
                    for term in ("gex_l1", "gex_l1_x_exp"):
                        rows.append({"sample": sname, "finding": "F1", "measure": mname, "rule": rule, "gex_published": lab,
                                     "spec": "single", "term": term, "coef": get(res, term), "t": get(res, term, "t"),
                                     "p": get(res, term, "p"), "n": int(res["n"].iloc[0])})
                        if off != -1:
                            reg.add("T2 timing", sname, "F1", rule, f"{term}, GEX at {lab}, {mname}", get(res, term, "p"),
                                    get(res, term, "t"))
                g2 = gs2[rule].shift(-off).reindex(d2.index)
                for spec in ("binary", "controls"):
                    res = f2_fit(d2, g2, spec, m2, missing_as_nonneg=(off == -1))
                    rows.append({"sample": sname, "finding": "F2", "measure": F2_MEASURE, "rule": rule, "gex_published": lab, "spec": spec,
                                 "term": "r_rod_x_neg", "coef": get(res, "r_rod_x_neg"), "t": get(res, "r_rod_x_neg", "t"),
                                 "p": get(res, "r_rod_x_neg", "p"), "n": int(res["n"].iloc[0])})
                    if off != -1:
                        reg.add("T2 timing", sname, "F2", rule, f"r_rod_x_neg {spec}, GEX at {lab}",
                                get(res, "r_rod_x_neg", "p"), get(res, "r_rod_x_neg", "t"))
            # horse race: GEX known before the outcome (t-1) against GEX published after it (t+1)
            for mname, gs1 in f1m.items():
                gl, gf = gs1[rule].shift(1), gs1[rule].shift(-1)
                res = f1_fit(d1, gl, m1, extra={"lead_gex": gf, "lead_gex_x_exp": gf * d1["exp"]})
                for term in ("gex_l1", "gex_l1_x_exp", "lead_gex", "lead_gex_x_exp"):
                    rows.append({"sample": sname, "finding": "F1", "measure": mname, "rule": rule,
                                 "gex_published": "t-1 and t+1 together", "spec": "horse race", "term": term,
                                 "coef": get(res, term), "t": get(res, term, "t"), "p": get(res, term, "p"),
                                 "n": int(res["n"].iloc[0])})
                    reg.add("T2 timing", sname, "F1", rule, f"horse race {term}, {mname}", get(res, term, "p"), get(res, term, "t"))
            gl2, gf2 = gs2[rule].shift(1).reindex(d2.index), gs2[rule].shift(-1).reindex(d2.index)
            X = f2_X(d2, gl2, "binary")
            lead = f2_X(d2, gf2, "binary", missing_as_nonneg=False, prefix="lead_")
            X = X.join(lead[["lead_neg", "lead_r_rod_x_neg"]])
            res = fit_named(d2.loc[m2, "r_last_bp"], X[m2])
            for term in ("r_rod_x_neg", "lead_r_rod_x_neg"):
                rows.append({"sample": sname, "finding": "F2", "measure": F2_MEASURE, "rule": rule, "gex_published": "t-1 and t+1 together",
                             "spec": "horse race (binary)", "term": term, "coef": get(res, term), "t": get(res, term, "t"),
                             "p": get(res, term, "p"), "n": int(res["n"].iloc[0])})
                reg.add("T2 timing", sname, "F2", rule, f"horse race {term}", get(res, term, "p"), get(res, term, "t"))
    acf = []
    for (mname, gs1), rule in [(m, r) for m in f1m.items() for r in RULES]:
        s = gs1[rule].dropna()
        neg = (s < 0).astype(int)
        acf.append({"measure": mname, "rule": rule, **{f"autocorr_lag{k}": s.autocorr(k) for k in (1, 2, 5, 20)},
                    "share_days_negative": float(neg.mean()), "share_days_sign_same_as_previous_day": float((neg == neg.shift(1)).iloc[1:].mean())})
    return pd.DataFrame(rows), pd.DataFrame(acf)


def _cloud_point(d1, d2, C, P, scale1, scale2, wc, wp, m1, m2) -> dict:
    net = wc * C + wp * P
    out = {}
    for mname, den in (("share", C + P), ("ratio", scale1)):
        g1 = (net / den).shift(1).reindex(d1.index)
        if g1[m1].std() > 0:
            r1 = f1_fit(d1, g1, m1)
            vals = (get(r1, "gex_l1", "t"), get(r1, "gex_l1_x_exp", "t"), get(r1, "gex_l1"), get(r1, "gex_l1_x_exp"))
        else:  # a constant regressor (for example a fixed rule on a chain held fixed) has no slope
            vals = (np.nan,) * 4
        for k, v in zip(("t_gex_l1", "t_gex_l1_x_exp", "coef_gex_l1", "coef_gex_l1_x_exp"), vals):
            out[f"{mname}_{k}"] = v
    g2 = (net / scale2).shift(1).reindex(d2.index)
    out["neg_days"] = int((g2[m2] < 0).sum())
    minority = min(out["neg_days"], int(m2.sum()) - out["neg_days"])
    for spec in ("binary", "controls"):
        if minority < 20:
            out[f"t_x_neg_{spec}"] = np.nan
            continue
        r2 = f2_fit(d2, g2, spec, m2)
        out[f"t_x_neg_{spec}"] = get(r2, "r_rod_x_neg", "t")
    return out


def t3_sign_cloud(d1, d2, panel, front, pw, reg: Registry, seed: int):
    """Placebo dealer weights. Net GEX = w_call * C + w_put * P, with C and P the gross call and put GEX of
    the front chain recomputed from the bhavcopies (front_from_table)."""
    C = front["C"].reindex(panel.index)
    P = front["P"].reindex(panel.index)
    gross = C + P
    scale = gross.rolling(20, min_periods=5).median()
    rng = np.random.default_rng([seed, 3])
    configs = [("angle", float(np.cos(np.radians(a))), float(np.sin(np.radians(a))), a) for a in range(0, 360)]
    configs += [("uniform", float(a), float(b), np.nan) for a, b in rng.uniform(-1, 1, size=(2000, 2))]
    grid = np.round(np.arange(-1, 1.0001, 0.1), 10)
    configs += [("grid", float(a), float(b), np.nan) for a in grid for b in grid if not (a == 0 and b == 0)]
    wts = rule_weights(pw)
    real = {k: wts[k] for k in ("us_convention", "participant_pro", "client_mirror")}
    rows, real_rows = [], []
    for sname, end in SAMPLES.items():
        m1, m2 = sample_mask(d1.index, end), sample_mask(d2.index, end)
        for kind, wc, wp, ang in configs:
            rows.append({"sample": sname, "kind": kind, "w_call": wc, "w_put": wp,
                         "angle_deg": ang if np.isfinite(ang) else float(np.degrees(np.arctan2(wp, wc)) % 360),
                         **_cloud_point(d1, d2, C, P, scale, scale, wc, wp, m1, m2)})
        for rule, (wc, wp) in real.items():
            wc_t, wp_t = wc.reindex(panel.index), wp.reindex(panel.index)
            mc, mp_ = float(wc_t.mean()), float(wp_t.mean())
            # chain held at its average shape relative to the scale, so only the participant weights move
            Cf, Pf = scale * float((C / scale).mean()), scale * float((P / scale).mean())
            for version, a, b, cc, pp in (("daily weights (as published)", wc_t, wp_t, C, P),
                                          ("weights fixed at sample mean, chain daily", mc, mp_, C, P),
                                          ("weights daily, chain fixed at average shape", wc_t, wp_t, Cf, Pf)):
                real_rows.append({"sample": sname, "rule": rule, "version": version, "mean_w_call": mc, "mean_w_put": mp_,
                                  "angle_deg": float(np.degrees(np.arctan2(mp_, mc)) % 360),
                                  **_cloud_point(d1, d2, cc, pp, scale, scale, a, b, m1, m2)})
    cloud = pd.DataFrame(rows)
    realdf = pd.DataFrame(real_rows)
    summ = []
    for sname in SAMPLES:
        for kind in ("uniform", "grid", "angle"):
            c = cloud[(cloud["sample"] == sname) & (cloud["kind"] == kind)]
            for stat in ("share_t_gex_l1", "share_t_gex_l1_x_exp", "ratio_t_gex_l1", "ratio_t_gex_l1_x_exp",
                         "t_x_neg_binary", "t_x_neg_controls"):
                x = c[stat].dropna()
                row = {"sample": sname, "kind": kind, "stat": stat, "draws_with_estimate": len(x), "draws_total": len(c),
                       "share_abs_t_above_1.96": float((x.abs() > 1.96).mean()) if len(x) else np.nan,
                       "share_t_below_-1.96": float((x < -1.96).mean()) if len(x) else np.nan,
                       "share_t_above_1.96": float((x > 1.96).mean()) if len(x) else np.nan,
                       "median_abs_t": float(x.abs().median()) if len(x) else np.nan,
                       "max_abs_t": float(x.abs().max()) if len(x) else np.nan}
                for rule in real:
                    rr = realdf[(realdf["sample"] == sname) & (realdf["rule"] == rule) & (realdf["version"] == "daily weights (as published)")]
                    v = float(rr[stat].iloc[0])
                    row[f"{rule}_t"] = v
                    ok = bool(len(x)) and np.isfinite(v)
                    k = int((x.abs() >= abs(v)).sum()) if ok else 0
                    row[f"{rule}_share_of_draws_with_larger_abs_t"] = k / len(x) if ok else np.nan
                    # rank p-value with the +1 correction: the real rule counted as one more member of the cloud
                    row[f"{rule}_placebo_rank_p"] = (1 + k) / (len(x) + 1) if ok else np.nan
                summ.append(row)
                if kind == "uniform":
                    for rule in real:
                        reg.add("T3 sign-rule cloud", sname, "F1" if "gex" in stat else "F2", rule,
                                f"{stat}: rank of |t| among placebo weight pairs ((1 + k) / (n + 1))",
                                row[f"{rule}_placebo_rank_p"], row[f"{rule}_t"])
    return cloud, realdf, pd.DataFrame(summ)


def t3_gross_oi(d1, d2, panel, reg: Registry) -> pd.DataFrame:
    """Unsigned quantities in place of net GEX: is it really gross gamma or open interest doing the work?"""
    scale = panel["gross_gex"].rolling(20, min_periods=5).median()
    cands = {"log gross GEX (demeaned)": np.log(panel["gross_gex"]) - np.log(panel["gross_gex"]).mean(),
             "gross GEX / 20-day median": panel["gross_gex"] / scale,
             "log total OI (demeaned)": np.log(panel["total_oi"]) - np.log(panel["total_oi"]).mean()}
    rows = []
    for sname, end in SAMPLES.items():
        m1, m2 = sample_mask(d1.index, end), sample_mask(d2.index, end)
        for name, s in cands.items():
            g1 = s.shift(1).reindex(d1.index)
            res = f1_fit(d1, g1, m1)
            for term in ("gex_l1", "gex_l1_x_exp"):
                rows.append({"sample": sname, "finding": "F1", "regressor": name, "spec": "F1 with regime controls",
                             "term": term.replace("gex", "x"), "coef": get(res, term), "t": get(res, term, "t"),
                             "p": get(res, term, "p"), "n": int(res["n"].iloc[0])})
                reg.add("T3 gross and OI", sname, "F1", name, term, get(res, term, "p"), get(res, term, "t"))
            g2 = s.shift(1).reindex(d2.index)
            low = g2 < g2[m2].median()  # "low" days play the role of negative-GEX days
            z = (g2 - g2[m2].mean()) / g2[m2].std()
            for spec, gg, term, lab in (("binary", np.where(low, -1.0, 1.0), "r_rod_x_neg", "r_rod x 1[below median]"),
                                        ("controls", np.where(low, -1.0, 1.0), "r_rod_x_neg", "r_rod x 1[below median], VIX and size controls"),
                                        ("continuous", z, "r_rod_x_gex", "r_rod x standardized value")):
                gser = pd.Series(gg, index=d2.index).where(g2.notna())
                res = f2_fit(d2, gser, spec, m2)
                rows.append({"sample": sname, "finding": "F2", "regressor": name, "spec": lab, "term": term,
                             "coef": get(res, term), "t": get(res, term, "t"), "p": get(res, term, "p"), "n": int(res["n"].iloc[0])})
                reg.add("T3 gross and OI", sname, "F2", name, lab, get(res, term, "p"), get(res, term, "t"))
    return pd.DataFrame(rows)


def t3_calendar(d1, f1m, d2, gs2, panel, reg: Registry) -> pd.DataFrame:
    """F1 and F2 with calendar controls added. The front chain's gross gamma rises mechanically as its expiry
    approaches, so any GEX built from it carries the position in the weekly cycle; if the weekly cycle itself
    moves volatility (or the late-day reversal), GEX can look predictive for a calendar reason."""
    scale = panel["gross_gex"].rolling(20, min_periods=5).median()
    regs1 = {f"{k} ({m})": gs1[k].shift(1) for m, gs1 in f1m.items() for k in RULES}
    regs1["gross GEX / 20-day median"] = (panel["gross_gex"] / scale).shift(1).reindex(d1.index)
    rows = []
    cal2 = calendar_controls(d2)
    for sname, end in SAMPLES.items():
        m1, m2 = sample_mask(d1.index, end), sample_mask(d2.index, end)
        for name, g in regs1.items():
            for calendar in (False, True):
                res = f1_fit(d1, g, m1, calendar=calendar)
                for term in ("gex_l1", "gex_l1_x_exp"):
                    rows.append({"sample": sname, "finding": "F1", "regressor": name,
                                 "spec": "regime x expiry + calendar controls" if calendar else "regime x expiry controls",
                                 "term": term, "coef": get(res, term), "t": get(res, term, "t"), "p": get(res, term, "p"),
                                 "n": int(res["n"].iloc[0])})
                    if calendar:
                        reg.add("T3 calendar controls", sname, "F1", name, term, get(res, term, "p"), get(res, term, "t"))
        for rule in RULES:
            g2 = gs2[rule].shift(1).reindex(d2.index)
            for spec in ("binary", "controls"):
                X = f2_X(d2, g2, spec)
                for c in cal2.columns:
                    X[c] = cal2[c]
                    X[f"r_rod_x_{c}"] = d2["r_rod_bp"] * cal2[c]
                X = X[[c for c in X.columns if c not in ("neg", "r_rod_x_neg")] + ["neg", "r_rod_x_neg"]]
                res = fit_named(d2.loc[m2, "r_last_bp"], X[m2])
                rows.append({"sample": sname, "finding": "F2", "regressor": rule,
                             "spec": f"{spec} + calendar dummies and their r_rod slopes", "term": "r_rod_x_neg",
                             "coef": get(res, "r_rod_x_neg"), "t": get(res, "r_rod_x_neg", "t"), "p": get(res, "r_rod_x_neg", "p"),
                             "n": int(res["n"].iloc[0])})
                reg.add("T3 calendar controls", sname, "F2", rule, f"r_rod_x_neg {spec}", get(res, "r_rod_x_neg", "p"),
                        get(res, "r_rod_x_neg", "t"))
    return pd.DataFrame(rows)


def _dm_test(loss_base, loss_alt, lags=LAGS) -> tuple[float, float]:
    """Diebold-Mariano: mean of (base loss - alternative loss) with a HAC standard error; one-sided p for
    'the alternative (with GEX) has lower loss'."""
    dl = np.asarray(loss_base, float) - np.asarray(loss_alt, float)
    b, se, t, _, _ = ols_hac(dl, np.ones((len(dl), 1)), lags)
    return float(t[0]), float(stats.norm.sf(t[0]))


def t4_out_of_sample(d1, f1m, d2, gs2, reg: Registry):
    split_rows, fc_rows, pre_rows = [], [], []
    for sname, end in SAMPLES.items():
        mA1 = d1.index <= SPLIT_A_END
        mB1 = (d1.index > SPLIT_A_END) & sample_mask(d1.index, end)
        mA2 = d2.index <= SPLIT_A_END
        mB2 = (d2.index > SPLIT_A_END) & sample_mask(d2.index, end)
        periods = {"2023 to 2024": (mA1, mA2), "2025 onward": (mB1, mB2)}
        for rule in RULES:
            g2 = gs2[rule].shift(1).reindex(d2.index)
            for pname, (p1, p2) in periods.items():
                for mname, gs1 in f1m.items():
                    g1 = gs1[rule].shift(1)
                    for regimes in (True, False):
                        res = f1_fit(d1, g1, p1, regimes=regimes)
                        for term in ("gex_l1", "gex_l1_x_exp"):
                            split_rows.append({"sample": sname, "finding": "F1", "measure": mname, "rule": rule, "period": pname,
                                               "spec": "regime x expiry controls" if regimes else "no regime controls",
                                               "term": term, "coef": get(res, term), "se_hac": get(res, term, "se_hac"),
                                               "p": get(res, term, "p"), "n": int(res["n"].iloc[0])})
                            if regimes:
                                reg.add("T4 split", sname, "F1", rule, f"{term}, {pname}, {mname}", get(res, term, "p"),
                                        get(res, term, "t"))
                for spec in ("binary", "controls"):
                    res = f2_fit(d2, g2, spec, p2)
                    split_rows.append({"sample": sname, "finding": "F2", "measure": F2_MEASURE, "rule": rule, "period": pname, "spec": spec,
                                       "term": "r_rod_x_neg", "coef": get(res, "r_rod_x_neg"),
                                       "se_hac": get(res, "r_rod_x_neg", "se_hac"), "p": get(res, "r_rod_x_neg", "p"),
                                       "n": int(res["n"].iloc[0]), "neg_days": int((g2[p2] < 0).sum())})
                    reg.add("T4 split", sname, "F2", rule, f"r_rod_x_neg {spec}, {pname}", get(res, "r_rod_x_neg", "p"),
                            get(res, "r_rod_x_neg", "t"))
            # Forecasts with frozen coefficients (no regime dummies: the later regimes do not exist in 2023 to 2024)
            for (trn, tst, lab) in ((periods["2023 to 2024"], periods["2025 onward"], "fit 2023 to 2024, test 2025 onward"),
                                    (periods["2025 onward"], periods["2023 to 2024"], "fit 2025 onward, test 2023 to 2024")):
                for mname, gs1 in f1m.items():
                    g1 = gs1[rule].shift(1)
                    X0 = f1_controls(d1, regimes=False)
                    X1 = X0.assign(gex_l1=g1, gex_l1_x_exp=g1 * d1["exp"])
                    ok = X1.notna().all(axis=1) & d1["y"].notna()
                    tr, te = trn[0] & ok.to_numpy(), tst[0] & ok.to_numpy()
                    losses = {}
                    for name, X in (("base", X0), ("gex", X1)):
                        A = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
                        b = np.linalg.lstsq(A[tr], d1["y"].to_numpy()[tr], rcond=None)[0]
                        losses[name] = (d1["y"].to_numpy()[te] - A[te] @ b) ** 2
                    t, p = _dm_test(losses["base"], losses["gex"])
                    fc_rows.append({"sample": sname, "finding": "F1", "measure": mname, "rule": rule, "design": lab,
                                    "test_days": int(te.sum()), "mse_without_gex": losses["base"].mean(),
                                    "mse_with_gex": losses["gex"].mean(),
                                    "pct_change_mse": 100 * (losses["gex"].mean() / losses["base"].mean() - 1),
                                    "dm_t": t, "dm_p_one_sided_gex_better": p})
                    reg.add("T4 forecast", sname, "F1", rule, f"{lab}, {mname}", p, t)
                X0 = pd.DataFrame({"r_rod": d2["r_rod_bp"]})
                X1 = f2_X(d2, g2, "binary")
                tr, te = trn[1], tst[1]
                losses = {}
                for name, X in (("base", X0), ("gex", X1)):
                    A = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
                    b = np.linalg.lstsq(A[tr], d2["r_last_bp"].to_numpy()[tr], rcond=None)[0]
                    losses[name] = (d2["r_last_bp"].to_numpy()[te] - A[te] @ b) ** 2
                t, p = _dm_test(losses["base"], losses["gex"])
                fc_rows.append({"sample": sname, "finding": "F2", "measure": F2_MEASURE, "rule": rule, "design": lab, "test_days": int(te.sum()),
                                "mse_without_gex": losses["base"].mean(), "mse_with_gex": losses["gex"].mean(),
                                "pct_change_mse": 100 * (losses["gex"].mean() / losses["base"].mean() - 1),
                                "dm_t": t, "dm_p_one_sided_gex_better": p})
                reg.add("T4 forecast", sname, "F2", rule, lab, p, t)
            # Pre-registered style test of F2: the hypothesis (sign and specification) is fixed from 2023 to 2024,
            # then tested once, one-sided, on 2025 onward only. Primary: Pro rule, binary spec, all days.
            resA = f2_fit(d2, g2, "binary", mA2)
            resB = f2_fit(d2, g2, "binary", mB2)
            cA, cB, tB = get(resA, "r_rod_x_neg"), get(resB, "r_rod_x_neg"), get(resB, "r_rod_x_neg", "t")
            direction = "negative" if cA < 0 else "positive"
            p_one = stats.norm.cdf(tB) if cA < 0 else stats.norm.sf(tB)
            pre_rows.append({"sample": sname, "rule": rule, "primary": rule == "participant_pro",
                             "hypothesis_from_2023_2024": f"r_rod_x_neg is {direction} (2023 to 2024 estimate {cA:.4f})",
                             "coef_2025_onward": cB, "se_hac_2025_onward": get(resB, "r_rod_x_neg", "se_hac"), "t": tB,
                             "p_one_sided": p_one, "n_2025_onward": int(resB["n"].iloc[0]),
                             "neg_days_2025_onward": int((g2[mB2] < 0).sum()),
                             "passes_at_5pct": bool(p_one < 0.05 and np.sign(cB) == np.sign(cA))})
            reg.add("T4 pre-registered F2", sname, "F2", rule, "binary, 2025 onward, one-sided", p_one, tB)
    return pd.DataFrame(split_rows), pd.DataFrame(fc_rows), pd.DataFrame(pre_rows)


def market_frame(m: dict) -> pd.DataFrame:
    """Day-level frame from a market dict (the same columns write_daily_market in run_pipeline.py writes)."""
    R = np.asarray(m["R"], float)
    lab = m["labels"]
    C = np.asarray(m["close"], float)
    return pd.DataFrame({
        "is_expiry": lab["is_expiry"].to_numpy(), "is_monthly_expiry": lab["is_monthly_expiry"].to_numpy(),
        "regime": lab["regime"].to_numpy(), "weekday": lab["weekday"].to_numpy(),
        "trading_days_to_expiry": lab["trading_days_to_expiry"].to_numpy(),
        "close_last_bar": [row[np.isfinite(row)][-1] if np.isfinite(row).any() else np.nan for row in C],
        "rv_day": (R ** 2).sum(axis=1), "minutes_with_bars": np.isfinite(C).sum(axis=1),
        "vix": np.asarray(m["vix"], float)}, index=pd.DatetimeIndex(m["days"], name="date"))


def t5_banknifty(raw: Path, out: Path, reg: Registry, nifty_d2: pd.DataFrame | None = None, nifty_gs2: dict | None = None):
    idx = pd.read_csv(raw / "index_daily.csv", parse_dates=["date"])
    official = idx.loc[idx["index"].str.lower() == "nifty bank", "date"]
    m = build_market_from_nse(raw / "banknifty_1min.csv", raw / "bhavcopy", raw / "participant_oi", raw / "india_vix.csv",
                              None, symbol="BANKNIFTY", oi_unit="keep_units", r=R_RATE, start="2023-01-01",
                              min_coverage=0.95, official_days=official, use_file_expiries=True)
    panel = build_gex_panel(m)
    dm = market_frame(m)
    panel.to_csv(out / "banknifty_gex_panel.csv")
    dm.to_csv(out / "banknifty_daily_market.csv")
    d1 = f1_frame(dm, panel)
    d2 = f2_frame(dm, raw / "banknifty_1min.csv")
    scale2 = panel["gross_gex"].rolling(20, min_periods=5).median()
    rows = []
    m1, m2 = np.ones(len(d1), bool), np.ones(len(d2), bool)
    for rule, col in RULES.items():
        for mname, den in (("share", d1["gross_gex"]), ("ratio", d1["scale"])):
            g1 = (panel[col] / den).shift(1)
            for regimes in (True, False):
                res = f1_fit(d1, g1, m1, regimes=regimes)
                for term in ("gex_l1", "gex_l1_x_exp"):
                    rows.append({"finding": "F1", "measure": mname, "rule": rule,
                                 "spec": "regime x expiry controls" if regimes else "no regime controls",
                                 "term": term, "coef": get(res, term), "se_hac": get(res, term, "se_hac"), "t": get(res, term, "t"),
                                 "p": get(res, term, "p"), "n": int(res["n"].iloc[0])})
                    if regimes:
                        reg.add("T5 Bank Nifty", "banknifty", "F1", rule, f"{term}, {mname}", get(res, term, "p"), get(res, term, "t"))
        g2 = (panel[col] / scale2).shift(1).reindex(d2.index)
        for spec in ("binary", "controls"):
            res = f2_fit(d2, g2, spec, m2)
            rows.append({"finding": "F2", "measure": F2_MEASURE, "rule": rule, "spec": spec, "term": "r_rod_x_neg", "coef": get(res, "r_rod_x_neg"),
                         "se_hac": get(res, "r_rod_x_neg", "se_hac"), "t": get(res, "r_rod_x_neg", "t"),
                         "p": get(res, "r_rod_x_neg", "p"), "n": int(res["n"].iloc[0]), "neg_days": int((g2 < 0).sum())})
            reg.add("T5 Bank Nifty", "banknifty", "F2", rule, f"r_rod_x_neg {spec}", get(res, "r_rod_x_neg", "p"),
                    get(res, "r_rod_x_neg", "t"))
    res = fit_named(d2["r_last_bp"], d2[["r_rod_bp"]])
    rows.append({"finding": "F2", "rule": "none", "spec": "momentum only", "term": "r_rod", "coef": get(res, "r_rod_bp"),
                 "se_hac": get(res, "r_rod_bp", "se_hac"), "t": get(res, "r_rod_bp", "t"), "p": get(res, "r_rod_bp", "p"),
                 "n": int(res["n"].iloc[0])})
    info = {"days": len(dm), "first_day": str(dm.index.min().date()), "last_day": str(dm.index.max().date()),
            "expiry_days": int(dm["is_expiry"].sum()), "f2_days": len(d2),
            "expiry_days_by_regime": dm.loc[dm["is_expiry"], "regime"].value_counts().to_dict()}
    if nifty_d2 is not None:
        # How independent is the second market? Same participant weights, highly correlated index.
        common = d2.index.intersection(nifty_d2.index)
        info["days_in_common_with_nifty_f2"] = len(common)
        info["corr_r_rod_nifty_banknifty"] = float(np.corrcoef(d2.loc[common, "r_rod_bp"], nifty_d2.loc[common, "r_rod_bp"])[0, 1])
        info["corr_r_last_nifty_banknifty"] = float(np.corrcoef(d2.loc[common, "r_last_bp"], nifty_d2.loc[common, "r_last_bp"])[0, 1])
        for rule, col in RULES.items():
            gb = (panel[col] / scale2).shift(1).reindex(common)
            gn = nifty_gs2[rule].shift(1).reindex(common)
            ok = gb.notna() & gn.notna()
            info[f"share_days_same_gex_sign_{rule}"] = float(((gb[ok] < 0) == (gn[ok] < 0)).mean())
    return pd.DataFrame(rows), dm, info


def qc_frame(sp: pd.DataFrame, w: tuple, scale_all: pd.Series, day: pd.DataFrame) -> pd.DataFrame:
    """Event-level variables for the roll-off test. All GEX terms are divided by the 20-day median of total
    (all-expiry) gross GEX at the pre-event close, and use the sign weights of that close."""
    days = day.index
    pos = {d: i for i, d in enumerate(days)}
    q = pd.DatetimeIndex(sp["prev_day"])
    wc, wp = (s.reindex(q).to_numpy(float) for s in w)
    s = scale_all.reindex(q).to_numpy(float)
    f = pd.DataFrame(index=sp.index)
    f["prev_day"] = q
    f["shock"] = (wc * sp["front_C"].to_numpy() + wp * sp["front_P"].to_numpy()) / s
    f["rest"] = (wc * sp["rest_C"].to_numpy() + wp * sp["rest_P"].to_numpy()) / s
    f["post"] = (wc * sp["post_C"].to_numpy() + wp * sp["post_P"].to_numpy()) / s
    f["total_before"] = f["shock"] + f["rest"]
    f["d_gex"] = f["post"] - f["total_before"]
    nxt = [days[pos[p] + 1] if pos[p] + 1 < len(days) else pd.NaT for p in sp.index]
    f["next_day"] = nxt
    y = day["y"]
    f["y_next"] = y.reindex(pd.DatetimeIndex(nxt)).to_numpy()
    f["y_event"] = y.reindex(sp.index).to_numpy()
    f["y_prev"] = y.reindex(q).to_numpy()
    f["ybar_prev"] = y.rolling(5).mean().reindex(q).to_numpy()
    f["lvix_prev"] = np.log(day["vix"]).reindex(q).to_numpy()
    for reg in REGIMES4[1:]:
        f[f"reg_{reg}"] = (day["regime4"].reindex(sp.index) == reg).astype(float).to_numpy()
    f["monthly"] = day["is_monthly_expiry"].reindex(sp.index).astype(float).to_numpy()
    f["front_expires_on_event_day"] = sp["front_expires_on_event_day"].to_numpy()
    return f


QC_CONTROLS = ["y_prev", "ybar_prev", "lvix_prev", "reg_single_weekly_thu", "reg_single_weekly_tue", "reg_closing_auction"]


def qc_fit(f: pd.DataFrame, extra: list = ()) -> pd.DataFrame:
    X = f[["shock", "rest"] + QC_CONTROLS + list(extra)]
    return fit_named(f["y_next"], X)


def qc_iv(f: pd.DataFrame, extra: list = ()) -> dict:
    cols = ["total_before"] + QC_CONTROLS + list(extra)
    ok = f[cols + ["y_next", "d_gex", "shock"]].notna().all(axis=1)
    X = f.loc[ok, cols]
    keep = [c for c in cols if X[c].std() > 0]
    exog = np.column_stack([np.ones(int(ok.sum()))] + [X[c].to_numpy(float) for c in keep])
    return iv_2sls_hac(f.loc[ok, "y_next"], exog, f.loc[ok, "d_gex"], f.loc[ok, "shock"])


def t6_quasi_causal(tab: pd.DataFrame, day: pd.DataFrame, weights: dict, reg: Registry, market: str = "nifty",
                    samples: dict = SAMPLES):
    days = day.index
    tot = tab.groupby("date")[["C", "P"]].sum().sum(axis=1).reindex(days)
    scale_all = tot.rolling(20, min_periods=5).median()
    expiry_days = days[day["is_expiry"].to_numpy(bool)]
    event_sets = {"real expiry days": (expiry_days, "")}
    for k in (-2, -1, 1, 2):
        clean = k in (-2, 2)
        event_sets[f"pseudo: expiry {k:+d} trading days"] = (
            shifted_placebo_dates(days, expiry_days, k, exclude_adjacent=clean),
            "" if clean else ("contaminated: the outcome day is the real expiry day" if k == -1 else
                              "contaminated: the pre-event day is the real expiry day"))
    event_sets["pseudo: other regime's expiry weekday"] = (weekday_placebo_dates(days, expiry_days, day["regime4"]), "")
    splits = {name: split_at(tab, days, ev) for name, (ev, _) in event_sets.items()}
    main_rows, iv_rows, ev_out = [], [], []
    for sname, end in samples.items():
        for name, (ev, note) in event_sets.items():
            sp = splits[name]
            if sp.empty:
                continue
            for rule, w in weights.items():
                f = qc_frame(sp, w, scale_all, day)
                f = f[f["next_day"].notna() & (f.index <= (end if end is not None else days.max()))]
                if end is not None:
                    f = f[f["next_day"] <= end]
                specs = {"main": []}
                if name == "real expiry days":
                    specs |= {"plus expiry-day RV (post-treatment control)": ["y_event"], "plus monthly-expiry dummy": ["monthly"]}
                for spec, extra in specs.items():
                    res = qc_fit(f, extra)
                    sd = float(f["shock"].std())
                    main_rows.append({"market": market, "sample": sname, "events": name, "note": note, "rule": rule, "spec": spec,
                                      "n_events": int(res["n"].iloc[0]),
                                      "share_front_expires": float(f["front_expires_on_event_day"].mean()),
                                      "shock_mean": float(f["shock"].mean()), "shock_sd": sd,
                                      "coef_shock": get(res, "shock"), "se_shock": get(res, "shock", "se_hac"),
                                      "p_shock": get(res, "shock", "p"), "coef_shock_per_sd": get(res, "shock") * sd,
                                      "se_shock_per_sd": get(res, "shock", "se_hac") * sd,
                                      "coef_rest": get(res, "rest"), "p_rest": get(res, "rest", "p"), "r2": float(res["r2"].iloc[0])})
                    reg.add("T6 roll-off", f"{market} {sname}", "roll-off", rule, f"{name}, {spec}", get(res, "shock", "p"),
                            get(res, "shock", "t"))
                if name == "real expiry days":
                    for spec, extra in (("main", []), ("plus expiry-day RV (post-treatment control)", ["y_event"])):
                        iv = qc_iv(f, extra)
                        iv_rows.append({"market": market, "sample": sname, "rule": rule, "spec": spec, **iv})
                        reg.add("T6 roll-off IV", f"{market} {sname}", "roll-off", rule, spec, iv["p_iv"])
                    if sname == "full":
                        ev_out.append(f.assign(rule=rule, market=market))
    main = pd.DataFrame(main_rows)
    # Real minus placebo (coefficients per 1 SD of the shock); the two sets share weeks, and treating them as
    # independent ignores a covariance that is most likely positive, which makes this z-test conservative.
    diffs = []
    for (sname, rule), g in main[main["spec"] == "main"].groupby(["sample", "rule"]):
        r = g[g["events"] == "real expiry days"]
        if r.empty:
            continue
        r = r.iloc[0]
        for _, pl in g[g["events"] != "real expiry days"].iterrows():
            z = (r["coef_shock_per_sd"] - pl["coef_shock_per_sd"]) / np.hypot(r["se_shock_per_sd"], pl["se_shock_per_sd"])
            diffs.append({"market": market, "sample": sname, "rule": rule, "placebo": pl["events"], "note": pl["note"],
                          "real_per_sd": r["coef_shock_per_sd"], "placebo_per_sd": pl["coef_shock_per_sd"],
                          "difference": r["coef_shock_per_sd"] - pl["coef_shock_per_sd"], "z": z,
                          "p_two_sided": 2 * stats.norm.sf(abs(z))})
            if not pl["note"]:
                reg.add("T6 real minus placebo", f"{market} {sname}", "roll-off", rule, pl["events"], 2 * stats.norm.sf(abs(z)), z)
    counts = {name: int(len(ev)) for name, (ev, _) in event_sets.items()}
    return main, pd.DataFrame(iv_rows), pd.DataFrame(diffs), (pd.concat(ev_out) if ev_out else pd.DataFrame()), counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fmt(df: pd.DataFrame, cols=None, n=None) -> str:
    x = df if cols is None else df[cols]
    if n is not None:
        x = x.head(n)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 400,
                           "display.precision", 4, "display.max_colwidth", 70):
        return x.to_string(index=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/falsification")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--skip-banknifty", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    run, raw, out = Path(a.run), Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def say(s=""):
        log.append(s)
        print(s, flush=True)

    reg = Registry()
    dm = pd.read_csv(run / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(run / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    d1 = f1_frame(dm, panel)
    d2 = f2_frame(dm, raw / "nifty_1min.csv")
    # F1 regressors, published at the close of each day (shifted by one day before use)
    f1m = {"share": {k: d1[c] / d1["gross_gex"] for k, c in RULES.items()},
           "ratio": {k: d1[c] / d1["scale"] for k, c in RULES.items()}}
    gs1 = f1m["ratio"]
    scale2 = panel["gross_gex"].rolling(20, min_periods=5).median()
    gs2 = {k: panel[c] / scale2 for k, c in RULES.items()}
    pw = participant_weights(raw / "participant_oi", dm.index)
    say("REAL DATA. Falsification tests for F1 (GEX and next-day volatility) and F2 (GEX and end-of-day reversal).")
    say(f"Nifty days {dm.index.min().date()} to {dm.index.max().date()} ({len(dm)} days, {int(dm['is_expiry'].sum())} expiry days); "
        f"closing-auction days from {AUCTION_START.date()}: {int((dm.index >= AUCTION_START).sum())}.")
    say(f"Seed {SEED}; permutations per cell {a.perms}; Newey-West lags {LAGS}; workers {a.workers}.")

    # ---- 0. replication and headline numbers on both samples
    f1o, f2o = original_family(dm, panel, d2, gs2)
    fam = pd.concat([f1o, f2o], ignore_index=True)
    rep = []
    try:
        r3_disk = pd.read_csv(run.parent / "real_day_level" / "r3_gex_regressions.csv")
        mom_disk = pd.read_csv(run.parent / "momentum_gex" / "regressions.csv")
        for _, r in f1o.iterrows():
            x = r3_disk[(r3_disk["controls"] == r["spec_detail"]) & (r3_disk["spec"] == f"gex_{r['rule']}") & (r3_disk["term"] == r["term"])]
            if len(x):
                rep.append({"finding": "F1", "rule": r["rule"], "spec": r["spec_detail"], "term": r["term"], "coef_here": r["coef"],
                            "coef_on_disk": float(x["coef"].iloc[0]), "p_here": r["p"], "p_on_disk": float(x["p"].iloc[0])})
        specmap = {"binary": "binary", "regime_slopes": "binary + regime slopes", "controls": "binary + VIX and size controls",
                   "continuous": "continuous"}
        for _, r in f2o.iterrows():
            spec, sub = r["spec_detail"].split(", ")
            x = mom_disk[(mom_disk["spec"] == specmap[spec]) & (mom_disk["rule"] == r["rule"]) & (mom_disk["subset"] == sub.replace(" days", ""))
                         & (mom_disk["term"] == r["term"])]
            if len(x):
                rep.append({"finding": "F2", "rule": r["rule"], "spec": r["spec_detail"], "term": r["term"], "coef_here": r["coef"],
                            "coef_on_disk": float(x["coef"].iloc[0]), "p_here": r["p"], "p_on_disk": float(x["p"].iloc[0])})
    except FileNotFoundError as exc:
        say(f"(replication check skipped: {exc})")
    rep = pd.DataFrame(rep)
    if len(rep):
        rep["abs_diff_coef"] = (rep["coef_here"] - rep["coef_on_disk"]).abs()
        rep.to_csv(out / "replication_check.csv", index=False)
        say(f"Replication of the original scripts' GEX terms: {len(rep)} coefficients compared, largest absolute difference "
            f"{rep['abs_diff_coef'].max():.2e}.")
    # sanity: the fast HAC routine equals statsmodels
    import statsmodels.api as sm
    g = gs1["participant_pro"].shift(1)
    X = f1_controls(d1).assign(gex_l1=g, gex_l1_x_exp=g * d1["exp"])
    ok = X.notna().all(axis=1) & d1["y"].notna()
    smf = sm.OLS(d1.loc[ok, "y"], sm.add_constant(X[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})
    mine = f1_fit(d1, g, np.ones(len(d1), bool))
    assert np.allclose(smf.bse["gex_l1_x_exp"], mine.loc["gex_l1_x_exp", "se_hac"], rtol=1e-8), "HAC mismatch"

    head = []
    for sname, end in SAMPLES.items():
        m1, m2 = sample_mask(d1.index, end), sample_mask(d2.index, end)
        for rule in RULES:
            for mname, gsm in f1m.items():
                res = f1_fit(d1, gsm[rule].shift(1), m1)
                for term in ("gex_l1", "gex_l1_x_exp"):
                    head.append({"sample": sname, "finding": "F1", "measure": mname, "rule": rule,
                                 "spec": "regime x expiry controls (auction = own regime)", "term": term, "coef": get(res, term),
                                 "se_hac": get(res, term, "se_hac"), "p": get(res, term, "p"), "n": int(res["n"].iloc[0])})
            g2 = gs2[rule].shift(1).reindex(d2.index)
            for spec in ("binary", "controls"):
                res = f2_fit(d2, g2, spec, m2)
                head.append({"sample": sname, "finding": "F2", "measure": F2_MEASURE, "rule": rule, "spec": spec, "term": "r_rod_x_neg",
                             "coef": get(res, "r_rod_x_neg"), "se_hac": get(res, "r_rod_x_neg", "se_hac"),
                             "p": get(res, "r_rod_x_neg", "p"), "n": int(res["n"].iloc[0])})
    head = pd.DataFrame(head)
    head.to_csv(out / "headline_in_sample.csv", index=False)
    say("\n== Headline in-sample coefficients (full sample: auction period as its own regime in F1's controls)")
    say(_fmt(head))

    # ---- T1 permutations
    t = time.time()
    perm, draws = t1_permutations(d1, f1m, d2, gs2, a.perms, a.workers, reg)
    perm.to_csv(out / "t1_permutation_summary.csv", index=False)
    draws.to_csv(out / "t1_permutation_draws.csv.gz", index=False)
    say(f"\n== T1 permutation placebos ({time.time() - t:.0f}s); p = share of placebo |t| at least the real |t|")
    say(_fmt(perm, ["sample", "finding", "measure", "rule", "term", "scheme", "real_coef", "real_t", "perm_coef_q025", "perm_coef_q975",
                    "perm_t_sd", "p_two_sided_t", "p_two_sided_coef"]))

    # ---- T2 timing
    tim, acf = t2_timing(d1, f1m, d2, gs2, reg)
    tim.to_csv(out / "t2_timing_placebo.csv", index=False)
    acf.to_csv(out / "t2_gex_persistence.csv", index=False)
    say("\n== T2 timing placebos")
    say(_fmt(tim))
    say(_fmt(acf))

    # ---- chain recomputation (all expiries) for T3 and T6
    t = time.time()
    tab = chain_table(raw, "NIFTY", dm.index, dm["close_last_bar"], a.workers)
    tab.to_csv(out / "chain_gex_by_expiry_nifty.csv.gz", index=False)
    front = front_from_table(tab)
    Cp = (panel["gross_gex"] + panel["net_gex_us_convention"]) / 2
    Pp = (panel["gross_gex"] - panel["net_gex_us_convention"]) / 2
    chk = pd.DataFrame({"C_panel": Cp, "P_panel": Pp}).join(front[["expiry", "C", "P", "remapped"]])
    chk["rel_diff_C"] = (chk["C"] / chk["C_panel"] - 1).abs()
    chk["rel_diff_P"] = (chk["P"] / chk["P_panel"] - 1).abs()
    chk["chain_expiry_panel"] = pd.to_datetime(dm["chain_expiry"])
    chk.to_csv(out / "chain_recompute_check.csv")
    bad = chk[(chk[["rel_diff_C", "rel_diff_P"]].max(axis=1) > 1e-6)]
    pro_check = (pw["pro_call"] * Cp + pw["pro_put"] * Pp) / panel["net_gex_participant"] - 1
    say(f"\n== Chain recomputation ({time.time() - t:.0f}s): {len(tab)} (day, expiry) rows. Front chain matches the panel within "
        f"1e-6 on {len(chk) - len(bad)} of {len(chk)} days; mismatching days: {', '.join(str(x.date()) for x in bad.index[:12])}"
        f"{' ...' if len(bad) > 12 else ''}. Holiday-remapped listed expiries: {tab.attrs['remapped_expiries']}. "
        f"Pro rule rebuilt from C, P and the participant file: largest relative error {pro_check.abs().max():.2e}.")

    # ---- T3 placebo sign rules
    t = time.time()
    cloud, realdf, csum = t3_sign_cloud(d1, d2, panel, front, pw, reg, SEED)
    cloud.to_csv(out / "t3_sign_rule_cloud.csv", index=False)
    realdf.to_csv(out / "t3_real_rules_in_cloud.csv", index=False)
    csum.to_csv(out / "t3_sign_rule_cloud_summary.csv", index=False)
    gro = t3_gross_oi(d1, d2, panel, reg)
    gro.to_csv(out / "t3_gross_and_oi.csv", index=False)
    cal = t3_calendar(d1, f1m, d2, gs2, panel, reg)
    cal.to_csv(out / "t3_calendar_controls.csv", index=False)
    say(f"\n== T3 placebo sign rules ({time.time() - t:.0f}s)")
    say(_fmt(csum))
    say(_fmt(realdf))
    say(_fmt(gro))
    say(_fmt(cal))

    # ---- T4 out of sample
    spl, fc, pre = t4_out_of_sample(d1, f1m, d2, gs2, reg)
    spl.to_csv(out / "t4_split_estimates.csv", index=False)
    fc.to_csv(out / "t4_oos_forecasts.csv", index=False)
    pre.to_csv(out / "t4_preregistered_f2.csv", index=False)
    say("\n== T4 out-of-sample split")
    say(_fmt(spl))
    say(_fmt(fc))
    say(_fmt(pre))

    # ---- T5 Bank Nifty
    bn_info = {}
    if not a.skip_banknifty:
        t = time.time()
        bn, bn_dm, bn_info = t5_banknifty(raw, out, reg, d2, gs2)
        bn.to_csv(out / "t5_banknifty.csv", index=False)
        say(f"\n== T5 Bank Nifty ({time.time() - t:.0f}s): {bn_info}")
        say(_fmt(bn))

    # ---- T6 quasi-causal roll-off
    t = time.time()
    day = d1[["y", "vix", "regime4", "is_monthly_expiry", "is_expiry"]].copy()
    weights = {k: tuple(s.reindex(dm.index) for s in v) for k, v in rule_weights(pw).items()}
    qc, qiv, qdiff, qev, qcounts = t6_quasi_causal(tab, day, weights, reg)
    if not a.skip_banknifty:
        bn_tab = chain_table(raw, "BANKNIFTY", bn_dm.index, bn_dm["close_last_bar"], a.workers)
        bn_tab.to_csv(out / "chain_gex_by_expiry_banknifty.csv.gz", index=False)
        bday = bn_dm.assign(y=np.log(bn_dm["rv_day"]), regime4=regime4(bn_dm))
        bw = {k: tuple(s.reindex(bn_dm.index) for s in v)
              for k, v in rule_weights(participant_weights(raw / "participant_oi", bn_dm.index)).items()}
        bqc, bqiv, bqdiff, _, bcounts = t6_quasi_causal(bn_tab, bday, bw, reg, market="banknifty", samples={"full": None})
        qc, qiv, qdiff = (pd.concat([x, y], ignore_index=True) for x, y in ((qc, bqc), (qiv, bqiv), (qdiff, bqdiff)))
        qcounts = {"nifty": qcounts, "banknifty": bcounts}
    qc.to_csv(out / "t6_rolloff_regressions.csv", index=False)
    qiv.to_csv(out / "t6_rolloff_iv.csv", index=False)
    qdiff.to_csv(out / "t6_real_minus_placebo.csv", index=False)
    qev.to_csv(out / "t6_event_table_nifty.csv")
    say(f"\n== T6 gamma roll-off ({time.time() - t:.0f}s). Event counts: {qcounts}")
    say(_fmt(qc, ["market", "sample", "events", "rule", "spec", "n_events", "share_front_expires", "shock_mean", "shock_sd",
                  "coef_shock", "p_shock", "coef_shock_per_sd", "coef_rest", "p_rest"]))
    say(_fmt(qiv))
    say(_fmt(qdiff))

    # ---- T7 multiple testing
    famA = fam.copy()
    if hasattr(rdl, "r3_robustness"):  # the scale-free variants added to real_day_level.py after the review
        rob = rdl.r3_robustness(dm, panel)
        rob = rob[rob["term"].isin(["gex_l1", "gex_l1_x_exp"])]
        famA = pd.concat([famA, pd.DataFrame({"finding": "F1", "rule": rob["rule"], "spec_detail": "robustness: " + rob["variant"],
                                              "term": rob["term"], "coef": rob["coef"], "p": rob["p"], "n": rob["n_days"]})],
                         ignore_index=True)
    famA["family"] = "A: every GEX-term test in the original scripts (R3, R3 robustness, R5)"
    famA = famA.join(adjust(famA["p"]))
    hl = famA[((famA["finding"] == "F1") & famA["spec_detail"].isin(["regime x expiry", "robustness: share"]))
              | ((famA["finding"] == "F2") & famA["spec_detail"].isin(["binary, all days", "controls, all days"]))].copy()
    hl = hl.join(adjust(hl["p"]).add_suffix("_within_headline"))
    pr = perm[(perm["scheme"] == "shuffle_within_regime") & (perm["sample"] == "full")].copy()
    pr = pr[(pr["finding"] == "F1") | (pr["term"].isin(["x_neg_binary", "x_neg_controls"]))]
    pr = pr[pr["measure"] != "ratio"].copy()  # main versions: the share for F1, the sign for F2
    pr = pr.join(adjust(pr["p_two_sided_t"]).add_suffix("_perm"))
    famB = reg.frame()
    famB = famB.join(adjust(famB["p"]))
    famA.to_csv(out / "t7_family_A_original_tests.csv", index=False)
    hl.to_csv(out / "t7_headline_adjusted.csv", index=False)
    pr.to_csv(out / "t7_headline_permutation_adjusted.csv", index=False)
    famB.to_csv(out / "t7_family_B_all_tests_this_study.csv", index=False)
    say(f"\n== T7 multiple testing. Family A (original scripts' GEX tests): {len(famA)} tests. "
        f"Family B (tests reported by this script): {len(famB)} tests.")
    say(_fmt(hl, ["finding", "rule", "spec_detail", "term", "coef", "p", "p_holm", "p_bh", "p_holm_within_headline",
                  "p_bh_within_headline"]))
    say(_fmt(pr, ["finding", "measure", "rule", "term", "real_t", "p_two_sided_t", "p_holm_perm", "p_bh_perm"]))
    say(f"Family B: {int((famB['p'] < 0.05).sum())} of {len(famB)} raw p < 0.05; {int((famB['p_bh'] < 0.05).sum())} "
        f"with Benjamini-Hochberg q < 0.05; {int((famB['p_holm'] < 0.05).sum())} with Holm < 0.05.")
    say(_fmt(famB[famB["p_bh"] < 0.05].sort_values("p")))

    info = {"label": "REAL DATA", "seed": SEED, "permutation_seeds": f"numpy default_rng([{SEED}, task_index]); the task index is in each row of t1_permutation_*.csv",
            "sign_cloud_seed": f"numpy default_rng([{SEED}, 3])", "permutations": a.perms, "newey_west_lags": LAGS,
            "workers": a.workers, "seconds": round(time.time() - t0, 1),
            "inputs": {"daily_market.csv": _sha(run / "daily_market.csv"), "daily_gex_panel.csv": _sha(run / "daily_gex_panel.csv")},
            "samples": {"full": f"{dm.index.min().date()} to {dm.index.max().date()}, closing-auction days (from {AUCTION_START.date()}) are their own regime",
                        "pre_auction": f"{dm.index.min().date()} to {PRE_AUCTION_END.date()}"},
            "split": f"A = to {SPLIT_A_END.date()}, B = after", "banknifty": bn_info, "tests_family_A": len(famA),
            "tests_family_B": len(famB), "python": sys.version.split()[0],
            "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    say(f"\nDone in {time.time() - t0:.0f}s.")
    (out.parent / "falsification_stdout.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
