"""F2 under the referee's questions (POST HOC, in response to the referee report of 3 Nov 2026). REAL DATA.

F2 (docs/REAL_DATA_RESULTS.md R5 and finding 7; docs/FALSIFICATION.md) is the finding that Nifty's
last-30-minute return reverses the rest-of-day return more after evenings on which GEX signed by NSE's
"Pro" participant positions was negative:

    r_last = a + b r_rod + d N + c (r_rod x N) + e,    N = 1[Pro-signed GEX at the previous close < 0]

r_rod runs from the previous trading day's 15:29 bar close to today's 14:59 bar close (overnight gap
included) and r_last from the 14:59 bar close to the 15:29 bar close, in basis points (bp), from the
validated 1-minute Nifty bars. F2 says c < 0. Every analysis below was specified after the referee report
(REFEREE_REPORT.md, points M2.2, M2.4, M2.5, M3, M10 and M13) and after every F2 result in the repository had
been seen. Nothing here is pre-registered, and the p-values are not adjusted for the further tests this
script adds.

Inputs (all committed or raw, read only): results/pipeline_nse_v2/daily_market.csv and daily_gex_panel.csv
(the v2 panels), data/raw/nifty_1min.csv, data/raw/participant_oi/*.csv,
data/external/sensex_expiry_calendar.csv and data/external/sebi_phases.csv (dates and sources of SEBI's
phased measures; no primary document could be fetched from this container, see the file).

Items (numbering as in docs/REFEREE_RESPONSE_F2.md):
  1  The unconditional slope b (no participant proxy). Under "dealers short all options" (the short_all rule
     in src/niftygex/gex.py, net GEX = minus gross GEX every evening) hedging theory predicts b > 0
     (end-of-day momentum). Inference: Newey-West (5 lags), Driscoll-Kraay (for one time series this is
     Newey-West on the regression scores with the lag rule floor(4 (T/100)^(2/9))), Newey-West with 60 lags
     and fixed-b p-values, clustered by calendar month (CR1, t with G-1 df) with a wild cluster bootstrap
     (restricted, Webb weights), and a 60-day moving-block bootstrap. Also the size version of short_all:
     does the slope rise with the size of gross gamma?
  2  What the Pro sign is: negative-GEX evenings by year, quarter and SEBI phase, run lengths,
     autocorrelation of the indicator and effective numbers of independent episodes, for Pro, FII, DII,
     Client (and Client mirror, FII plus Pro, US convention); plus the categories' raw position states.
  3  F2's interaction c under inference fit for a persistent state: clustered by month and by episode (a run
     of evenings with the same sign), with wild cluster bootstraps; Driscoll-Kraay; Newey-West 60 with
     fixed-b; and block permutations of the indicator with 60- and 120-day blocks (whole blocks are moved,
     so the runs inside them are kept). Full sample, 2023 to 2024, 2025 onward.
  4  F2 with other categories as the dealer proxy: FII, FII plus Pro, DII, Client mirror (and Client, US
     convention), plus short_all.
  5  Alternative explanations: the last 30 minutes split at the 15:15 bar close (broker auto square-off;
     the pre-auction closing-price averaging window), the 14:59 to 15:00 bar dropped (stale prints), month
     and quarter fixed effects and slopes, a January to March interaction, SEBI phase dummies (1 Feb 2025,
     1 Apr 2025, 3 Jul 2025) and Sensex expiry days.
  6  Dealer-sign angles (docs/DEALER_SIGN.md): a joint block bootstrap test that the angles revealed by
     outcome (a) and outcome (b) are equal.
  7  Microstructure noise: first-order autocorrelation of 1-minute Nifty returns in 14:59 to 15:29 on expiry
     and other days, by period, and the correlation across the 14:59 junction.

    OMP_NUM_THREADS=1 python experiments/referee_f2.py            # about 2 to 4 minutes
    OMP_NUM_THREADS=1 python -m pytest -q tests/test_referee_f2.py

Writes results/referee_f2/*.csv, results/referee_f2/run_info.json and results/referee_f2_stdout.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize, stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dealer_sign as dsg  # noqa: E402
import momentum_gex as mg  # noqa: E402

SEED = 20261103
LAGS = 5
SLOT_1459, SLOT_1500, SLOT_1515, SLOT_1529 = 344, 345, 360, 374  # bar labels, minutes after 09:15
N_SLOTS = 375
AUCTION_START = pd.Timestamp("2026-08-03")
SPLIT_A_END = pd.Timestamp("2024-12-31")
SAMPLES = {"full": (None, None), "2023_2024": (None, SPLIT_A_END), "2025_onward": (pd.Timestamp("2025-01-01"), None)}
CATEGORIES = ("pro", "fii", "dii", "client")
# Sign rules: net GEX = w_call x C + w_put x P, with (w_call, w_put) from the participant file (net position of a
# category as a share of all open index-option contracts) or fixed.
RULES = {
    "pro": {"pro": 1.0},
    "fii": {"fii": 1.0},
    "fii_plus_pro": {"fii": 1.0, "pro": 1.0},
    "dii": {"dii": 1.0},
    "client": {"client": 1.0},
    "client_mirror": {"client": -1.0},
    "us_convention": "fixed",
    "short_all": "fixed",
}
FIXED = {"us_convention": (1.0, -1.0), "short_all": (-1.0, -1.0), "long_all": (1.0, 1.0)}
REGIMES4 = ("multi_weekly", "single_weekly_thu", "single_weekly_tue", "closing_auction")
WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# =====================================================================================
# Statistical helpers (tested on synthetic data in tests/test_referee_f2.py)
# =====================================================================================


def nw_scalar(s: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) long-run variance sum of a score series s (no division by T)."""
    s = np.asarray(s, float)
    v = float(s @ s)
    for lag in range(1, min(lags, len(s) - 1) + 1):
        v += 2.0 * (1.0 - lag / (lags + 1.0)) * float(s[lag:] @ s[:-lag])
    return v


def dk_lag(T: int) -> int:
    """Driscoll-Kraay (Hoechle 2007) default lag: floor(4 (T/100)^(2/9))."""
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def ols_focus(y, X, j: int, lags: int = LAGS) -> dict:
    """OLS with the Newey-West standard error of coefficient j (no small-sample correction, as statsmodels HAC).

    The HAC variance of b_j is the long-run variance of h_t u_t with h = X (X'X)^-1 e_j, which is the (j, j)
    element of the usual sandwich. Returns coef, se, t, p (normal) and the pieces reused by other methods.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    h = X @ XtX_inv[:, j]
    se = float(np.sqrt(nw_scalar(h * u, lags)))
    t = float(b[j] / se) if se > 0 else np.nan
    return {"coef": float(b[j]), "se": se, "t": t, "p": float(2 * stats.norm.sf(abs(t))), "b": b, "u": u, "h": h,
            "XtX_inv": XtX_inv, "n": len(y), "k": X.shape[1]}


def cluster_se(h, u, groups, k: int) -> tuple[float, int]:
    """CR1 cluster-robust standard error of one coefficient (Stata/statsmodels small-sample factor)."""
    codes = pd.factorize(np.asarray(groups))[0]
    G = int(codes.max()) + 1
    n = len(u)
    S = np.bincount(codes, weights=np.asarray(h) * np.asarray(u), minlength=G)
    if G < 2:
        return np.nan, G
    return float(np.sqrt(G / (G - 1) * (n - 1) / (n - k) * (S @ S))), G


def webb_weights(shape, rng) -> np.ndarray:
    """Webb (2014) six-point weights, recommended for the wild cluster bootstrap with few clusters."""
    vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    return vals[rng.integers(0, 6, size=shape)]


def wild_cluster_p(y, X, j: int, groups, B: int, rng, chunk: int = 1000) -> dict:
    """Wild cluster restricted bootstrap (WCR, Cameron, Gelbach and Miller 2008; MacKinnon and Webb 2017).

    The null b_j = 0 is imposed, restricted residuals are multiplied by one Webb weight per cluster, and the
    CR1 t-statistic is recomputed on every draw. Two-sided p = (1 + #{|t*| >= |t|}) / (1 + B).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    codes = pd.factorize(np.asarray(groups))[0]
    G = int(codes.max()) + 1
    if G < 2:
        return {"t": np.nan, "p": np.nan, "G": G}
    XtX_inv = np.linalg.inv(X.T @ X)
    Pm = XtX_inv @ X.T
    h = X @ XtX_inv[:, j]
    c = G / (G - 1) * (n - 1) / (n - k)
    b = Pm @ y
    u = y - X @ b
    S = np.bincount(codes, weights=h * u, minlength=G)
    t0 = b[j] / np.sqrt(c * (S @ S))
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    yr = Xr @ br
    ur = y - yr
    ind = np.zeros((G, n))
    ind[codes, np.arange(n)] = 1.0
    count, done = 0, 0
    while done < B:
        m = min(chunk, B - done)
        W = webb_weights((G, m), rng)
        Ys = yr[:, None] + ur[:, None] * W[codes]
        Bs = Pm @ Ys
        R = Ys - X @ Bs
        Sg = ind @ (h[:, None] * R)
        ts = Bs[j] / np.sqrt(c * (Sg ** 2).sum(axis=0))
        count += int(np.sum(np.abs(ts) >= abs(t0)))
        done += m
    return {"t": float(t0), "p": (1 + count) / (1 + B), "G": G}


def fixed_b_null(T: int, M: int, n_sim: int, rng, chunk: int = 2000) -> np.ndarray:
    """Null distribution of |t| for a Bartlett-kernel HAC t-test with bandwidth M at sample size T, simulated
    from i.i.d. normal scores (the fixed-b limit of Kiefer and Vogelsang 2005 is the same for any single
    linear restriction in a regression)."""
    out = []
    done = 0
    while done < n_sim:
        m = min(chunk, n_sim - done)
        e = rng.standard_normal((m, T))
        mu = e.mean(axis=1)
        dd = e - mu[:, None]
        lrv = (dd * dd).sum(axis=1)
        for lag in range(1, min(M, T - 1) + 1):
            lrv += 2.0 * (1.0 - lag / (M + 1.0)) * (dd[:, lag:] * dd[:, :-lag]).sum(axis=1)
        out.append(np.abs(np.sqrt(T) * mu / np.sqrt(lrv / T)))
        done += m
    return np.concatenate(out)


def moving_block_indices(n: int, block: int, rng) -> np.ndarray:
    """Row positions of one moving-block bootstrap sample of length n."""
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def block_relocate(x, block: int, rng) -> np.ndarray:
    """Block permutation that keeps runs: rotate by a random offset, cut into consecutive blocks of ``block``
    observations and put the blocks in random order (each block keeps its internal order)."""
    x = np.asarray(x)
    n = len(x)
    xr = np.roll(x, int(rng.integers(0, n)))
    pieces = [xr[i:i + block] for i in range(0, n, block)]
    order = rng.permutation(len(pieces))
    return np.concatenate([pieces[i] for i in order])


def within_block_shuffle(x, block: int, rng) -> np.ndarray:
    """Shuffle observations inside consecutive blocks (the T1 scheme of docs/FALSIFICATION.md, longer blocks)."""
    x = np.asarray(x)
    out = x.copy()
    for i in range(0, len(x), block):
        out[i:i + block] = x[i:i + block][rng.permutation(min(block, len(x) - i))]
    return out


def runs_of(ind) -> list[tuple[int, int, float]]:
    """Runs of equal values in a sequence without missing values: (start position, end position, value)."""
    a = np.asarray(ind, float)
    if len(a) == 0:
        return []
    cut = np.flatnonzero(np.diff(a) != 0) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut - 1, [len(a) - 1]])
    return [(int(s), int(e), float(a[s])) for s, e in zip(starts, ends)]


def episode_ids(ind) -> np.ndarray:
    """Cluster label for each observation: the number of the run of equal values it belongs to."""
    a = np.asarray(ind, float)
    if len(a) == 0:
        return np.zeros(0, int)
    return np.concatenate([[0], np.cumsum(np.diff(a) != 0)]).astype(int)


def acf(x, lag: int) -> float:
    x = np.asarray(x, float)
    if lag >= len(x) - 2:
        return np.nan
    a, b = x[lag:], x[:-lag]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def n_eff_ar1(x) -> float:
    """Effective sample size of a persistent series under an AR(1) approximation: n (1 - rho1) / (1 + rho1)."""
    r = acf(x, 1)
    return float(len(x) * (1 - r) / (1 + r)) if np.isfinite(r) else np.nan


def iat_sokal(x, c: float = 5.0) -> tuple[float, int]:
    """Integrated autocorrelation time tau = 1 + 2 sum_{k<=K} rho_k with Sokal's window (smallest K >= c tau(K)).
    n / tau is an effective number of independent observations."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or x.std() == 0:
        return np.nan, 0
    d = x - x.mean()
    f = np.fft.rfft(d, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    rho = ac / ac[0]
    tau = 1.0
    for K in range(1, n):
        tau = 1.0 + 2.0 * rho[1:K + 1].sum()
        if K >= c * tau:
            return float(max(tau, 1.0)), K
    return float(max(tau, 1.0)), n - 1


def circ_diff(a, b):
    """a - b wrapped to (-180, 180]."""
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def arc_contains_zero(lo: float, hi: float) -> bool:
    """True when the arc from lo to hi (degrees, lo <= hi, not wrapped) contains 0 or a multiple of 360."""
    return bool(np.floor(hi / 360.0) * 360.0 >= lo)


def angle_of(theta) -> float:
    """Revealed direction (degrees) for unrestricted loadings theta: the theory-signed direction is -theta."""
    return float(np.degrees(np.arctan2(-theta[1], -theta[0])) % 360.0)


def theta_from_arrays(y, Z, Bm) -> np.ndarray:
    st = dsg.gram_stats(y, Z, Bm)
    return np.linalg.solve(st["A"], st["b"])


def same_direction_md(theta_a, theta_b, V, grid_step: float = 0.1) -> dict:
    """Minimum-distance test that two loading vectors point the same way: -theta_a = la u(phi),
    -theta_b = lb u(phi), la, lb >= 0. J = min (theta - h)' V^-1 (theta - h), compared with chi-square(1)."""
    th = -np.concatenate([np.asarray(theta_a, float), np.asarray(theta_b, float)])
    L = np.linalg.cholesky(np.linalg.inv(V))
    best = (np.inf, np.nan)
    for phi in np.arange(0.0, 360.0, grid_step):
        u = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
        Mx = np.zeros((4, 2))
        Mx[:2, 0], Mx[2:, 1] = u, u
        lam, rn = optimize.nnls(L.T @ Mx, L.T @ th)
        if rn ** 2 < best[0]:
            best = (rn ** 2, phi)
    return {"J": float(best[0]), "p": float(stats.chi2.sf(best[0], 1)), "phi_common": float(best[1])}


def pooled_pairs_stats(R: np.ndarray, cols: list[int]) -> pd.DataFrame:
    """Per-day sums for the first-order autocorrelation of 1-minute returns inside a window.

    R: days x minutes returns; cols: the window's return columns in time order. Pairs (r_j, r_j+1) with
    both values present. Returns per-day n, sx, sy, sxx, syy, sxy and the day's own autocorrelation."""
    A = R[:, cols[:-1]]
    Bv = R[:, cols[1:]]
    ok = np.isfinite(A) & np.isfinite(Bv)
    A0, B0 = np.where(ok, A, 0.0), np.where(ok, Bv, 0.0)
    n = ok.sum(axis=1)
    out = pd.DataFrame({"n": n, "sx": A0.sum(1), "sy": B0.sum(1), "sxx": (A0 ** 2).sum(1), "syy": (B0 ** 2).sum(1),
                        "sxy": (A0 * B0).sum(1)})
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = out["sx"] / n, out["sy"] / n
        cov = out["sxy"] / n - mx * my
        vx, vy = out["sxx"] / n - mx ** 2, out["syy"] / n - my ** 2
        out["ac1_day"] = cov / np.sqrt(vx * vy)
    return out


def pooled_corr(s: pd.DataFrame | np.ndarray) -> float:
    """Pearson correlation of all pooled pairs from per-day sums (columns n, sx, sy, sxx, syy, sxy)."""
    v = np.asarray(s, float)
    n, sx, sy, sxx, syy, sxy = v[..., 0], v[..., 1], v[..., 2], v[..., 3], v[..., 4], v[..., 5]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        return cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))


def boot_pooled_corr(sums: np.ndarray, B: int, rng) -> np.ndarray:
    """Day bootstrap of the pooled correlation: resample days with replacement (multinomial counts)."""
    nd = len(sums)
    cnt = rng.multinomial(nd, np.full(nd, 1.0 / nd), size=B)
    return pooled_corr(cnt @ sums)


# =====================================================================================
# Data
# =====================================================================================


def minute_closes(bars_path: Path, days: pd.DatetimeIndex) -> np.ndarray:
    """days x 375 matrix of 1-minute bar closes (09:15 to 15:29 bar labels), the same reading as momentum_gex."""
    b = pd.read_csv(bars_path, parse_dates=["timestamp"])
    b = b[(b["timestamp"] >= days.min() - pd.Timedelta(days=7)) & (b["timestamp"] < days.max() + pd.Timedelta(days=1))]
    day = b["timestamp"].dt.normalize()
    slot = ((b["timestamp"] - day) / pd.Timedelta(minutes=1) - 555).astype(int)
    piv = pd.DataFrame({"day": day, "slot": slot, "close": b["close"]}).pivot_table(
        index="day", columns="slot", values="close", aggfunc="last")
    piv = piv.reindex(index=days, columns=range(N_SLOTS))
    return piv.to_numpy(float)


def rule_gex(panel: pd.DataFrame, nets: pd.DataFrame) -> pd.DataFrame:
    """Net GEX on each evening under every sign rule, from the chain's call and put gamma (C, P) in the v2 panel
    and the participant file's net shares (forward-filled to trading days, as src/niftygex/data.py does)."""
    C = (panel["gross_gex"] + panel["net_gex_us_convention"]) / 2
    P = (panel["gross_gex"] - panel["net_gex_us_convention"]) / 2
    nt = nets.reindex(panel.index).ffill()
    out = pd.DataFrame(index=panel.index)
    for rule, spec in RULES.items():
        if spec == "fixed":
            wc, wp = FIXED[rule]
            out[rule] = wc * C + wp * P
        else:
            wc = sum(s * nt[f"{cat}_call"] for cat, s in spec.items())
            wp = sum(s * nt[f"{cat}_put"] for cat, s in spec.items())
            out[rule] = wc * C + wp * P
    out["gross"] = panel["gross_gex"]
    return out


def sensex_expiry_days(cal_path: Path, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sensex weekly expiry days from the reported weekday regimes, moved back to the previous trading day when
    the scheduled day is not one (the rule of experiments/sensex_confound.py)."""
    cal = pd.read_csv(cal_path, dtype=str, keep_default_na=False)
    reg = cal[cal["fact_type"] == "weekday_regime"]
    td = pd.DatetimeIndex(sorted(trading_days))
    tdset = set(td)
    out = set()
    for _, row in reg.iterrows():
        start = pd.Timestamp(row["date"])
        end = pd.Timestamp(row["date_end"]) if row["date_end"] else pd.Timestamp("2100-01-01")
        wd = WD.index(row["weekday"])
        s = start + pd.Timedelta(days=(wd - start.weekday()) % 7)
        while s <= min(end, td.max()):
            d = s
            while d not in tdset and d > s - pd.Timedelta(days=6):
                d -= pd.Timedelta(days=1)
            if d in tdset:
                out.add(d)
            s += pd.Timedelta(days=7)
    return pd.DatetimeIndex(sorted(out))


def sebi_dates(path: Path) -> dict:
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    t = t[t["fact_type"].isin(["sebi_phase", "sebi_order"])]
    names = {"2025-02-01": "sebi_feb2025", "2025-04-01": "sebi_apr2025", "2025-07-03": "sebi_jul2025"}
    return {names[d]: pd.Timestamp(d) for d in t["date"] if d in names}


def build_frame(dm: pd.DataFrame, gex: pd.DataFrame, closes: np.ndarray, sensex: pd.DatetimeIndex,
                sebi: dict) -> pd.DataFrame:
    """One row per trading day with the F2 outcome pieces, the evening-before signs and calendar variables."""
    c = pd.DataFrame(closes, index=dm.index)
    prev = c[SLOT_1529].shift(1)
    d = pd.DataFrame(index=dm.index)
    lg = lambda a, b: 1e4 * np.log(a / b)  # noqa: E731
    d["r_rod"] = lg(c[SLOT_1459], prev)
    d["r_last"] = lg(c[SLOT_1529], c[SLOT_1459])
    d["r_1459_1515"] = lg(c[SLOT_1515], c[SLOT_1459])
    d["r_1515_1529"] = lg(c[SLOT_1529], c[SLOT_1515])
    d["r_1459_1500"] = lg(c[SLOT_1500], c[SLOT_1459])
    d["r_1500_1529"] = lg(c[SLOT_1529], c[SLOT_1500])
    d["is_expiry"] = dm["is_expiry"].astype(bool)
    d["regime"] = dm["regime"].astype(str)
    d["regime4"] = np.where(d.index >= AUCTION_START, "closing_auction", d["regime"])
    d["lvix_prev"] = np.log(dm["vix"]).shift(1)  # on the full calendar, before any day is dropped
    for rule in RULES:
        g = gex[rule].shift(1).reindex(d.index)
        d[f"gex_{rule}"] = g
        d[f"neg_{rule}"] = (g < 0).astype(float).where(g.notna())
    gr = gex["gross"]
    d["gross_rel_l1"] = (gr / gr.rolling(20, min_periods=5).median()).shift(1).reindex(d.index)
    d["month"] = d.index.to_period("M").astype(str)
    d["yq"] = d.index.to_period("Q").astype(str)
    d["year"] = d.index.year
    d["qoy"] = d.index.quarter
    d["moy"] = d.index.month
    d["janmar"] = (d.index.month <= 3).astype(float)
    for k, v in sebi.items():
        d[k] = (d.index >= v).astype(float)
    d["sensex_expiry"] = d.index.isin(sensex).astype(float)
    d["pre_auction"] = d.index < AUCTION_START
    return d


def sample_mask(idx: pd.DatetimeIndex, name: str) -> np.ndarray:
    lo, hi = SAMPLES[name]
    m = np.ones(len(idx), bool)
    if lo is not None:
        m &= idx >= lo
    if hi is not None:
        m &= idx <= hi
    return m


# =====================================================================================
# F2 designs and inference
# =====================================================================================


def dummies(s: pd.Series, prefix: str) -> pd.DataFrame:
    D = pd.get_dummies(s.astype(str), prefix=prefix, dtype=float)
    return D.iloc[:, 1:]  # drop the first level


def f2_parts(d: pd.DataFrame, spec: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(Z, M, focus names) for X(N) = [Z, N x M]. Z holds every column that does not depend on the sign
    indicator N; M the columns multiplied by N. The F2 coefficient is the one on N x r_rod."""
    r = d["r_rod"]
    Z = pd.DataFrame({"const": 1.0, "r_rod": r}, index=d.index)
    M = pd.DataFrame({"N": 1.0, "N_x_r_rod": r}, index=d.index)
    focus = ["N_x_r_rod"]

    def add_slopes(D: pd.DataFrame):
        for col in D:
            Z[col] = D[col]
            Z[f"r_rod_x_{col}"] = r * D[col]

    lv = d["lvix_prev"] - d["lvix_prev"].mean()
    if spec in ("controls", "all_controls"):
        Z["lvix_prev"] = lv
        Z["r_rod_x_lvix"] = r * lv
        Z["r_rod_x_abs_r_rod"] = r * r.abs()
    if spec == "yq_fe":
        Z = Z.join(dummies(d["yq"], "yq"))
    if spec == "ym_fe":
        Z = Z.join(dummies(d["month"], "ym"))
    if spec == "qoy_slopes":
        add_slopes(dummies(d["qoy"], "q"))
    if spec == "moy_slopes":
        add_slopes(dummies(d["moy"], "m"))
    if spec == "yq_slopes":
        add_slopes(dummies(d["yq"], "yq"))
    if spec == "janmar":
        J = d["janmar"]
        Z["janmar"], Z["r_rod_x_janmar"] = J, r * J
        M = pd.DataFrame({"N_x_notJM": 1 - J, "N_x_r_rod_notJM": r * (1 - J), "N_x_JM": J, "N_x_r_rod_JM": r * J},
                         index=d.index)
        focus = ["N_x_r_rod_notJM", "N_x_r_rod_JM"]
    if spec == "janmar_diff":
        J = d["janmar"]
        Z["janmar"], Z["r_rod_x_janmar"] = J, r * J
        M["N_x_JM"], M["N_x_r_rod_x_JM"] = J, r * J
        focus = ["N_x_r_rod_x_JM"]
    if spec in ("sebi", "sebi_regime", "all_controls"):
        add_slopes(d[[c for c in d if c.startswith("sebi_")]])
    if spec in ("sebi_regime", "all_controls"):
        add_slopes(pd.DataFrame({f"reg_{k}": (d["regime4"] == k).astype(float) for k in REGIMES4[1:]}, index=d.index))
    if spec in ("sensex", "all_controls"):
        add_slopes(d[["sensex_expiry"]])
    if spec == "all_controls":
        # Year-quarter level effects are left out here: the 1 Apr 2025 phase starts a quarter and would be collinear.
        add_slopes(dummies(d["qoy"], "q"))
    return Z, M, focus


def _drop_constant(Z: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in Z if c == "const" or Z[c].std() > 0]
    return Z[keep]


def perm_draws(y, Z, M, j_focus, neg, scheme: str, block: int, B: int, rng) -> np.ndarray:
    """(coef, HAC t) of the focus term over B permutations of the sign indicator."""
    out = np.full((B, 2), np.nan)
    n = len(neg)
    for b in range(B):
        npm = block_relocate(neg, block, rng) if scheme == "relocate" else within_block_shuffle(neg, block, rng)
        s = npm.sum()
        if s < 5 or s > n - 5:
            continue
        X = np.column_stack([Z, npm[:, None] * M])
        XtX = X.T @ X
        if np.linalg.cond(XtX) > 1e12:
            continue
        f = ols_focus(y, X, j_focus)
        out[b] = f["coef"], f["t"]
    return out


def perm_p(real: float, draws: np.ndarray) -> float:
    d = draws[np.isfinite(draws)]
    return (1 + np.sum(np.abs(d) >= abs(real))) / (1 + len(d)) if len(d) else np.nan


def infer_f2(d: pd.DataFrame, rule: str, spec: str, sample: np.ndarray, y_col: str, methods: tuple, cfg: dict,
             seed_key: list, fixed_b_cache: dict) -> list[dict]:
    """Every requested inference method for the F2 interaction(s) of one rule, specification and sample."""
    base = d[sample & d[f"neg_{rule}"].notna().to_numpy()]
    episodes = pd.Series(episode_ids(base[f"neg_{rule}"].to_numpy()), index=base.index)
    Z, M, focus = f2_parts(base, spec)
    ok = base[y_col].notna() & Z.notna().all(axis=1) & M.notna().all(axis=1)
    base, Z, M = base[ok], _drop_constant(Z[ok]), M[ok]
    neg = base[f"neg_{rule}"].to_numpy(float)
    y = base[y_col].to_numpy(float)
    n_neg = int(neg.sum())
    ep = episodes[ok].to_numpy()
    head = {"rule": rule, "spec": spec, "outcome": y_col, "n": len(y), "n_neg": n_neg,
            "n_neg_episodes": int(sum(1 for _, _, v in runs_of(neg) if v == 1)),
            "n_episodes": int(len(np.unique(ep))), "n_months": int(base["month"].nunique()),
            "n_janmar": int(base["janmar"].sum()), "n_neg_janmar": int((neg * base["janmar"].to_numpy(float)).sum())}
    if n_neg < 5 or n_neg > len(y) - 5:
        return [{**head, "term": f, "method": "not estimable", "detail": "fewer than 5 days in one sign state"}
                for f in focus]
    Zv, Mv = Z.to_numpy(float), M.to_numpy(float)
    X = np.column_stack([Zv, neg[:, None] * Mv])
    names = list(Z.columns) + list(M.columns)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return [{**head, "term": f, "method": "not estimable", "detail": "collinear design"} for f in focus]
    rows = []
    for fi, fname in enumerate(focus):
        j = names.index(fname)
        f5 = ols_focus(y, X, j)
        r0 = {**head, "term": fname, "coef": f5["coef"]}

        def add(method, se=np.nan, stat=np.nan, p=np.nan, n_cl=np.nan, detail=""):
            rows.append({**r0, "method": method, "se": se, "stat": stat, "p": p, "n_clusters": n_cl, "detail": detail})

        if "hac5" in methods:
            add("newey_west_5", f5["se"], f5["t"], f5["p"], detail="Bartlett, 5 lags, normal p (as published)")
        if "dk" in methods:
            L = dk_lag(len(y))
            fd = ols_focus(y, X, j, lags=L)
            add("driscoll_kraay", fd["se"], fd["t"], fd["p"], detail=f"one time series: Newey-West on the scores, lag {L} = floor(4 (T/100)^(2/9))")
        if "nw60" in methods:
            f60 = ols_focus(y, X, j, lags=60)
            key = (len(y), 60)
            if key not in fixed_b_cache:
                fixed_b_cache[key] = fixed_b_null(len(y), 60, cfg["n_fixedb"], np.random.default_rng([SEED, 7, len(y)]))
            pfb = (1 + np.sum(fixed_b_cache[key] >= abs(f60["t"]))) / (1 + len(fixed_b_cache[key]))
            add("newey_west_60_fixed_b", f60["se"], f60["t"], pfb, detail=f"Bartlett, 60 lags (b = {60 / len(y):.3f}); fixed-b p by simulation; normal p {f60['p']:.4g}")
        for ci_, (cl_name, groups) in enumerate((("month", base["month"].to_numpy()), ("episode", ep),
                                                  ("quarter", base["yq"].to_numpy()))):
            if f"cluster_{cl_name}" in methods:
                se, G = cluster_se(f5["h"], f5["u"], groups, X.shape[1])
                t = f5["coef"] / se
                add(f"cluster_{cl_name}", se, t, float(2 * stats.t.sf(abs(t), G - 1)), G, f"CR1, t with {G - 1} df")
            if f"wild_{cl_name}" in methods:
                w = wild_cluster_p(y, X, j, groups, cfg["n_wild"], np.random.default_rng(seed_key + [fi, 1 + ci_]))
                add(f"wild_cluster_{cl_name}", np.nan, w["t"], w["p"], w["G"], f"restricted wild cluster bootstrap, Webb weights, {cfg['n_wild']} draws")
        for meth in methods:
            if not meth.startswith("perm_"):
                continue
            scheme, block = meth.split("_")[1], int(meth.split("_")[2])
            j_m = list(M.columns).index(fname)
            draws = perm_draws(y, Zv, Mv, Zv.shape[1] + j_m, neg, scheme, block, cfg["n_perm"],
                               np.random.default_rng(seed_key + [fi, 10 + block + (0 if scheme == "relocate" else 1000)]))
            n_valid = int(np.isfinite(draws[:, 0]).sum())
            label = "block_permutation" if scheme == "relocate" else "within_block_shuffle"
            band = (f"placebo coef 2.5 to 97.5%: {np.nanpercentile(draws[:, 0], 2.5):.4f} to "
                    f"{np.nanpercentile(draws[:, 0], 97.5):.4f}") if n_valid else "no valid draws"
            add(f"{label}_{block}_t", np.nan, f5["t"], perm_p(f5["t"], draws[:, 1]), np.ceil(len(y) / block),
                f"{n_valid} valid draws; statistic |HAC t|; {band}")
            add(f"{label}_{block}_coef", np.nan, f5["coef"], perm_p(f5["coef"], draws[:, 0]), np.ceil(len(y) / block),
                f"{n_valid} valid draws; statistic |coef|; {band}")
    return rows


# =====================================================================================
# Items
# =====================================================================================


def item1_unconditional(d: pd.DataFrame, cfg: dict, fixed_b_cache: dict) -> pd.DataFrame:
    """The plain slope b of r_last on r_rod: the short_all (constant sign) test of hedging theory."""
    rows = []
    masks = {"full": np.ones(len(d), bool), "pre_auction": d["pre_auction"].to_numpy(),
             "2023_2024": sample_mask(d.index, "2023_2024"), "2025_onward": sample_mask(d.index, "2025_onward"),
             "expiry days": d["is_expiry"].to_numpy(), "non-expiry days": ~d["is_expiry"].to_numpy()}
    for k, (sname, m) in enumerate(masks.items()):
        s = d[m]
        y, X = s["r_last"].to_numpy(), np.column_stack([np.ones(m.sum()), s["r_rod"].to_numpy()])
        f5 = ols_focus(y, X, 1)
        base = {"test": "unconditional slope b (short_all: theory predicts b > 0)", "sample": sname, "n": len(y),
                "coef": f5["coef"]}

        def add(method, se=np.nan, stat=np.nan, p=np.nan, p_mom=np.nan, lo=np.nan, hi=np.nan, n_cl=np.nan, detail=""):
            rows.append({**base, "method": method, "se": se, "stat": stat, "p_two_sided": p,
                         "p_one_sided_momentum": p_mom, "ci95_lo": lo, "ci95_hi": hi, "n_clusters": n_cl, "detail": detail})

        add("newey_west_5", f5["se"], f5["t"], f5["p"], stats.norm.sf(f5["t"]), f5["coef"] - 1.96 * f5["se"],
            f5["coef"] + 1.96 * f5["se"], detail="as published in R5")
        L = dk_lag(len(y))
        fd = ols_focus(y, X, 1, lags=L)
        add("driscoll_kraay", fd["se"], fd["t"], fd["p"], stats.norm.sf(fd["t"]), detail=f"Newey-West on the scores, lag {L}")
        f60 = ols_focus(y, X, 1, lags=60)
        key = (len(y), 60)
        if key not in fixed_b_cache:
            fixed_b_cache[key] = fixed_b_null(len(y), 60, cfg["n_fixedb"], np.random.default_rng([SEED, 7, len(y)]))
        pfb = (1 + np.sum(fixed_b_cache[key] >= abs(f60["t"]))) / (1 + len(fixed_b_cache[key]))
        add("newey_west_60_fixed_b", f60["se"], f60["t"], pfb, detail=f"normal p {f60['p']:.4g}")
        se, G = cluster_se(f5["h"], f5["u"], s["month"].to_numpy(), 2)
        t = f5["coef"] / se
        add("cluster_month", se, t, 2 * stats.t.sf(abs(t), G - 1), stats.t.sf(t, G - 1), n_cl=G, detail="CR1, t with G-1 df")
        w = wild_cluster_p(y, X, 1, s["month"].to_numpy(), cfg["n_wild"], np.random.default_rng([SEED, 1, k]))
        add("wild_cluster_month", stat=w["t"], p=w["p"], n_cl=w["G"], detail="restricted, Webb weights")
        rng = np.random.default_rng([SEED, 2, k])
        bs = np.empty(cfg["n_boot"])
        for b in range(cfg["n_boot"]):
            idx = moving_block_indices(len(y), 60, rng)
            Xb = X[idx]
            bs[b] = np.linalg.lstsq(Xb, y[idx], rcond=None)[0][1]
        dev = bs - f5["coef"]
        add("block_bootstrap_60", float(bs.std(ddof=1)), f5["coef"] / bs.std(ddof=1),
            (1 + np.sum(np.abs(dev) >= abs(f5["coef"]))) / (1 + len(bs)), (1 + np.sum(dev >= f5["coef"])) / (1 + len(bs)),
            float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
            detail=f"moving blocks of 60 days, {cfg['n_boot']} draws; percentile interval; basic-bootstrap p")
    # The size version of short_all: does the slope rise with the size of gross gamma (more short gamma)?
    s = d[d["gross_rel_l1"].notna()]
    for k, (sname, m) in enumerate({"full": np.ones(len(s), bool), "pre_auction": s["pre_auction"].to_numpy()}.items()):
        ss = s[m]
        z = (ss["gross_rel_l1"] - ss["gross_rel_l1"].mean()) / ss["gross_rel_l1"].std()
        X = np.column_stack([np.ones(len(ss)), ss["r_rod"], z, ss["r_rod"] * z])
        y = ss["r_last"].to_numpy()
        f5 = ols_focus(y, X, 3)
        base = {"test": "short_all size: slope change per 1 sd of gross GEX / 20-day median (theory predicts > 0)",
                "sample": sname, "n": len(y), "coef": f5["coef"]}
        rows.append({**base, "method": "newey_west_5", "se": f5["se"], "stat": f5["t"], "p_two_sided": f5["p"],
                     "p_one_sided_momentum": stats.norm.sf(f5["t"]), "ci95_lo": f5["coef"] - 1.96 * f5["se"],
                     "ci95_hi": f5["coef"] + 1.96 * f5["se"], "detail": f"gross GEX at the previous close; sd {ss['gross_rel_l1'].std():.3f}"})
        se, G = cluster_se(f5["h"], f5["u"], ss["month"].to_numpy(), 4)
        t = f5["coef"] / se
        rows.append({**base, "method": "cluster_month", "se": se, "stat": t, "p_two_sided": 2 * stats.t.sf(abs(t), G - 1),
                     "p_one_sided_momentum": stats.t.sf(t, G - 1), "n_clusters": G, "detail": "CR1"})
        w = wild_cluster_p(y, X, 3, ss["month"].to_numpy(), cfg["n_wild"], np.random.default_rng([SEED, 3, k]))
        rows.append({**base, "method": "wild_cluster_month", "stat": w["t"], "p_two_sided": w["p"], "n_clusters": w["G"],
                     "detail": "restricted, Webb weights"})
    return pd.DataFrame(rows)


def item2_tables(gex: pd.DataFrame, nets: pd.DataFrame, sebi: dict, trading_days: pd.DatetimeIndex) -> dict:
    """Negative-GEX evenings, runs, autocorrelation and effective episodes for each rule; raw position states."""
    rules = ["pro", "fii", "dii", "client", "client_mirror", "fii_plus_pro", "us_convention"]
    ev = pd.DataFrame({r: (gex[r] < 0).astype(float).where(gex[r].notna()) for r in rules}, index=gex.index)
    by_year, by_q, by_seg, runs_rows, summ, acf_rows = [], [], [], [], [], []
    seg_edges = [pd.Timestamp("1900-01-01")] + sorted(sebi.values()) + [pd.Timestamp("2100-01-01")]
    seg_names = ["before 1 Feb 2025", "1 Feb to 31 Mar 2025", "1 Apr to 2 Jul 2025", "from 3 Jul 2025"]
    periods = {"all evenings": np.ones(len(ev), bool), "2023 to 2024": ev.index <= SPLIT_A_END,
               "2025 onward": ev.index > SPLIT_A_END}
    for y in sorted(ev.index.year.unique()):
        periods[str(y)] = ev.index.year == y
    for r in rules:
        s = ev[r].dropna()
        for y, g in s.groupby(s.index.year):
            by_year.append({"rule": r, "year": y, "evenings": len(g), "negative": int(g.sum()), "share_negative": g.mean()})
        for q, g in s.groupby(s.index.to_period("Q").astype(str)):
            by_q.append({"rule": r, "quarter": q, "evenings": len(g), "negative": int(g.sum()), "share_negative": g.mean()})
        for k in range(len(seg_names)):
            g = s[(s.index >= seg_edges[k]) & (s.index < seg_edges[k + 1])]
            by_seg.append({"rule": r, "segment": seg_names[k], "from": g.index.min().date() if len(g) else None,
                           "to": g.index.max().date() if len(g) else None, "evenings": len(g),
                           "negative": int(g.sum()), "share_negative": g.mean() if len(g) else np.nan})
        for st, en, v in runs_of(s.to_numpy()):
            runs_rows.append({"rule": r, "sign": "negative" if v == 1 else "non-negative", "start": s.index[st].date(),
                              "end": s.index[en].date(), "evenings": en - st + 1})
        for pname, pm in periods.items():
            x = s[pm[ev[r].notna().to_numpy()]].to_numpy()
            if len(x) == 0:
                continue
            rr = runs_of(x)
            neg_len = np.array([e - b + 1 for b, e, v in rr if v == 1])
            tau, K = iat_sokal(x)
            top3 = np.sort(neg_len)[::-1][:3].sum() if len(neg_len) else 0
            summ.append({"rule": r, "period": pname, "evenings": len(x), "negative": int(x.sum()),
                         "share_negative": x.mean(), "sign_switches": int(np.sum(np.diff(x) != 0)),
                         "negative_runs": len(neg_len), "negative_runs_5plus": int(np.sum(neg_len >= 5)),
                         "negative_runs_10plus": int(np.sum(neg_len >= 10)),
                         "neg_run_mean": neg_len.mean() if len(neg_len) else np.nan,
                         "neg_run_median": float(np.median(neg_len)) if len(neg_len) else np.nan,
                         "neg_run_max": int(neg_len.max()) if len(neg_len) else 0,
                         "share_of_negatives_in_3_longest_runs": top3 / x.sum() if x.sum() else np.nan,
                         "acf1": acf(x, 1), "n_eff_ar1": n_eff_ar1(x), "iat_sokal": tau, "iat_window": K,
                         "n_eff_iat": len(x) / tau if np.isfinite(tau) else np.nan})
            for lag in (1, 2, 5, 10, 20, 60):
                acf_rows.append({"rule": r, "period": pname, "lag": lag, "acf": acf(x, lag)})
    # Raw position states of each category (as the referee tabulated): net short both index calls and puts, etc.
    nt = nets[nets.index.isin(trading_days)]
    pos = []
    for cat in CATEGORIES:
        c, p = nt[f"{cat}_call"], nt[f"{cat}_put"]
        state = np.select([(c < 0) & (p < 0), (c > 0) & (p > 0), (c < 0) & (p > 0), (c > 0) & (p < 0)],
                          ["short calls, short puts", "long calls, long puts", "short calls, long puts",
                           "long calls, short puts"], default="zero in one side")
        st = pd.Series(state, index=nt.index)
        for y, g in st.groupby(st.index.year):
            vc = g.value_counts(normalize=True)
            row = {"category": cat, "year": y, "days": len(g), "mean_net_call_share": c[g.index].mean(),
                   "mean_net_put_share": p[g.index].mean()}
            for k in ["short calls, short puts", "long calls, long puts", "short calls, long puts",
                      "long calls, short puts", "zero in one side"]:
                row[k] = float(vc.get(k, 0.0))
            pos.append(row)
    return {"by_year": pd.DataFrame(by_year), "by_quarter": pd.DataFrame(by_q), "by_segment": pd.DataFrame(by_seg),
            "runs": pd.DataFrame(runs_rows), "summary": pd.DataFrame(summ), "acf": pd.DataFrame(acf_rows),
            "positions": pd.DataFrame(pos)}


def item6_angles(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint block bootstrap of the dealer-sign angles of outcomes (a) and (b) (participants family)."""
    rows, draws_out = [], []
    samples = {"full sample": frame, "training 2023 to 2024": frame[frame.index < dsg.TEST_START]}
    for si, (sname, sub) in enumerate(samples.items()):
        ya, Za, Ba = dsg.design_arrays(sub, "a", "participants")
        yb, Zb, Bb = dsg.design_arrays(sub, "b", "participants")
        mb = dsg.eq_rows(sub, "b")
        posb = np.cumsum(mb) - 1
        th_a, th_b = theta_from_arrays(ya, Za, Ba), theta_from_arrays(yb, Zb, Bb)
        grid_a = dsg.estimate_direction([dsg.gram_stats(ya, Za, Ba)])["angle"]
        grid_b = dsg.estimate_direction([dsg.gram_stats(yb, Zb, Bb)])["angle"]
        ang_a, ang_b = angle_of(th_a), angle_of(th_b)
        dhat = float(circ_diff(ang_a, ang_b))
        for bi, block in enumerate((10, 60)):
            rng = np.random.default_rng([SEED, 6, si, bi])
            T = np.full((cfg["n_angle_boot"], 4), np.nan)
            n = len(sub)
            for r in range(cfg["n_angle_boot"]):
                idx = moving_block_indices(n, block, rng)
                ib = posb[idx[mb[idx]]]
                try:
                    T[r, :2] = theta_from_arrays(ya[idx], Za[idx], Ba[idx])
                    T[r, 2:] = theta_from_arrays(yb[ib], Zb[ib], Bb[ib])
                except np.linalg.LinAlgError:
                    continue
            T = T[np.isfinite(T).all(axis=1)]
            aa = np.degrees(np.arctan2(-T[:, 1], -T[:, 0])) % 360
            ab = np.degrees(np.arctan2(-T[:, 3], -T[:, 2])) % 360
            dstar = circ_diff(aa, ab)
            dev = circ_diff(dstar, dhat)
            V = np.cov(T.T)
            md = same_direction_md(th_a, th_b, V)
            corr = np.corrcoef(T.T)
            rows.append({"sample": sname, "block_days": block, "n_days_a": len(ya), "n_days_b": len(yb),
                         "angle_a_grid": grid_a, "angle_b_grid": grid_b, "angle_a": ang_a, "angle_b": ang_b,
                         "diff_a_minus_b": dhat, "boot_draws": len(T),
                         "boot_sd_diff": float(np.sqrt(np.mean(dev ** 2))),
                         "diff_ci90_lo": float(circ_diff(dhat - np.percentile(dev, 95), 0)),
                         "diff_ci90_hi": float(circ_diff(dhat - np.percentile(dev, 5), 0)),
                         "diff_ci95_lo": float(circ_diff(dhat - np.percentile(dev, 97.5), 0)),
                         "diff_ci95_hi": float(circ_diff(dhat - np.percentile(dev, 2.5), 0)),
                         "zero_in_ci95": bool(arc_contains_zero(dhat - np.percentile(dev, 97.5), dhat - np.percentile(dev, 2.5))),
                         "p_equal_basic_bootstrap": (1 + np.sum(np.abs(dev) >= abs(dhat))) / (1 + len(dev)),
                         "share_draws_same_side": float(np.mean(np.abs(dstar) < 90)),
                         "share_draws_within_45deg": float(np.mean(np.abs(dstar) < 45)),
                         "share_a_within_90_of_estimate": float(np.mean(np.abs(circ_diff(aa, ang_a)) < 90)),
                         "share_b_within_90_of_estimate": float(np.mean(np.abs(circ_diff(ab, ang_b)) < 90)),
                         "md_J": md["J"], "md_p_chi2_1": md["p"], "md_p_chi2_2": float(stats.chi2.sf(md["J"], 2)),
                         "md_common_angle": md["phi_common"],
                         "wald_theta_a_zero_boot": float(th_a @ np.linalg.solve(V[:2, :2], th_a)),
                         "wald_theta_a_zero_boot_p": float(stats.chi2.sf(th_a @ np.linalg.solve(V[:2, :2], th_a), 2)),
                         "max_abs_corr_theta_a_theta_b": float(np.abs(corr[:2, 2:]).max()),
                         "theta_a": f"({th_a[0]:.3f}, {th_a[1]:.3f})", "theta_b": f"({th_b[0]:.3f}, {th_b[1]:.3f})"})
            draws_out.append(pd.DataFrame({"sample": sname, "block_days": block, "angle_a": aa, "angle_b": ab,
                                           "diff": dstar}))
    return pd.DataFrame(rows), pd.concat(draws_out, ignore_index=True)


def item7_noise(d_all: pd.DataFrame, closes: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First-order autocorrelation of 1-minute returns in the last 30 minutes (and a reference window)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.log(closes[:, 1:] / closes[:, :-1])  # column j-1 = return into bar j
    win = {"14:59 to 15:29 (30 returns)": list(range(SLOT_1500 - 1, SLOT_1529)),
           "reference 09:15 to 14:59 (344 returns)": list(range(0, SLOT_1459))}
    sums = {w: pooled_pairs_stats(R, cols) for w, cols in win.items()}
    groupings = {"all pre-auction days": pd.Series("all", index=d_all.index).where(d_all["pre_auction"]),
                 "regime": d_all["regime4"],
                 "half": pd.Series(np.where(d_all.index <= SPLIT_A_END, "2023 to 2024", "2025 onward"), index=d_all.index).where(d_all["pre_auction"]),
                 "Pro sign (pre-auction)": d_all["neg_pro"].map({1.0: "negative Pro evening", 0.0: "non-negative Pro evening"}).where(d_all["pre_auction"])}
    rng = np.random.default_rng([SEED, 7])
    rows, junc = [], []
    exp = d_all["is_expiry"].to_numpy()
    for wname, S in sums.items():
        cols6 = ["n", "sx", "sy", "sxx", "syy", "sxy"]
        Sv = S[cols6].to_numpy(float)
        for gname, gs in groupings.items():
            for gval in [v for v in pd.unique(gs.dropna())]:
                gm = (gs == gval).to_numpy()
                res = {}
                for dtype, dm_ in (("expiry", gm & exp), ("other", gm & ~exp)):
                    keep = dm_ & (S["n"].to_numpy() > 1)
                    sv = Sv[keep]
                    if len(sv) < 3:
                        continue
                    est = float(pooled_corr(sv.sum(axis=0)))
                    bt = boot_pooled_corr(sv, cfg["n_noise_boot"], rng)
                    res[dtype] = (est, bt)
                    rows.append({"window": wname, "grouping": gname, "group": gval, "day_type": dtype,
                                 "days": int(keep.sum()), "pairs": int(sv[:, 0].sum()), "ac1_pooled": est,
                                 "ci95_lo": float(np.percentile(bt, 2.5)), "ci95_hi": float(np.percentile(bt, 97.5)),
                                 "ac1_mean_of_days": float(np.nanmean(S["ac1_day"].to_numpy()[keep])),
                                 "noise_var_share_iid_model": float(-2 * est) if est < 0 else 0.0,
                                 # Under i.i.d. noise, 1-minute RV = efficient variance / (1 - noise share), so the
                                 # log RV is overstated by -log(1 - noise share) relative to a noise-free measure.
                                 "log_rv_overstatement_iid_model": float(-np.log1p(2 * est)) if est < 0 else 0.0})
                if len(res) == 2:
                    diff = res["expiry"][0] - res["other"][0]
                    db = res["expiry"][1] - res["other"][1]
                    rows.append({"window": wname, "grouping": gname, "group": gval, "day_type": "expiry minus other",
                                 "ac1_pooled": diff, "ci95_lo": float(np.percentile(db, 2.5)),
                                 "ci95_hi": float(np.percentile(db, 97.5)),
                                 "p_two_sided": (1 + np.sum(np.abs(db - diff) >= abs(diff))) / (1 + len(db))})
    # The junction between r_rod and r_last: last minute before 14:59 bar close and first minute after it.
    x, y = R[:, SLOT_1459 - 1], R[:, SLOT_1500 - 1]
    for gname, gm in (("all pre-auction days", d_all["pre_auction"].to_numpy()),
                      ("pre-auction expiry days", d_all["pre_auction"].to_numpy() & exp),
                      ("pre-auction other days", d_all["pre_auction"].to_numpy() & ~exp),
                      ("closing-auction days", ~d_all["pre_auction"].to_numpy())):
        ok = gm & np.isfinite(x) & np.isfinite(y)
        rr = stats.pearsonr(x[ok], y[ok])
        junc.append({"days": gname, "n": int(ok.sum()),
                     "corr_r(14:58 to 14:59 bar)_r(14:59 to 15:00 bar)": float(rr.statistic), "p": float(rr.pvalue)})
    return pd.DataFrame(rows), pd.DataFrame(junc)


# =====================================================================================
# Main
# =====================================================================================


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse_v2")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--external", default="data/external")
    ap.add_argument("--out", default="results/referee_f2")
    ap.add_argument("--n-perm", type=int, default=4999)
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--n-wild", type=int, default=4999)
    ap.add_argument("--n-fixedb", type=int, default=20000)
    ap.add_argument("--n-angle-boot", type=int, default=4999)
    ap.add_argument("--n-noise-boot", type=int, default=1999)
    a = ap.parse_args()
    cfg = {"n_perm": a.n_perm, "n_boot": a.n_boot, "n_wild": a.n_wild, "n_fixedb": a.n_fixedb,
           "n_angle_boot": a.n_angle_boot, "n_noise_boot": a.n_noise_boot}
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        lines.append(s)
        print(s, flush=True)

    def show(title, df, cols=None):
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 400,
                               "display.max_colwidth", 70, "display.precision", 4):
            say(f"== {title}")
            say((df[cols] if cols else df).to_string(index=False))
            say()

    run, raw, ext = Path(a.run), Path(a.raw), Path(a.external)
    dm = pd.read_csv(run / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(run / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    nets = dsg.participant_nets(raw / "participant_oi")
    gex = rule_gex(panel, nets)
    closes = minute_closes(raw / "nifty_1min.csv", dm.index)
    sensex = sensex_expiry_days(ext / "sensex_expiry_calendar.csv", dm.index)
    sebi = sebi_dates(ext / "sebi_phases.csv")
    d_all = build_frame(dm, gex, closes, sensex, sebi)
    d = d_all.dropna(subset=["r_rod", "r_last"]).copy()

    say("REAL DATA. POST HOC analyses in response to the referee report (3 Nov 2026) on F2. Nothing here is pre-registered.")
    say(f"Inputs: {run}/daily_market.csv and daily_gex_panel.csv (v2 panels), {raw}/nifty_1min.csv, {raw}/participant_oi, "
        f"{ext}/sensex_expiry_calendar.csv, {ext}/sebi_phases.csv.")
    # ---------------------------------------------------------------- checks
    rets = mg.daily_returns(raw / "nifty_1min.csv", dm.index)
    chk = [{"check": "r_rod equals experiments/momentum_gex.py (bp, max abs diff)",
            "value": float((d["r_rod"] - rets["r_rod"].reindex(d.index) * 1e4).abs().max())},
           {"check": "r_last equals experiments/momentum_gex.py (bp, max abs diff)",
            "value": float((d["r_last"] - rets["r_last"].reindex(d.index) * 1e4).abs().max())},
           {"check": "Pro rule rebuilt from C, P and the participant file vs panel net_gex_participant (max abs diff / gross)",
            "value": float(((gex["pro"] - panel["net_gex_participant"]).abs() / panel["gross_gex"]).max())},
           {"check": "Client mirror rebuilt vs panel net_gex_client_mirror (max abs diff / gross)",
            "value": float(((gex["client_mirror"] - panel["net_gex_client_mirror"]).abs() / panel["gross_gex"]).max())},
           {"check": "split 14:59 to 15:15 plus 15:15 to 15:29 minus r_last (bp, max abs diff)",
            "value": float((d["r_1459_1515"] + d["r_1515_1529"] - d["r_last"]).abs().max())},
           {"check": "F2 days (r_rod and r_last present)", "value": len(d)},
           {"check": "F2 days with a Pro sign", "value": int(d["neg_pro"].notna().sum())},
           {"check": "F2 days with a negative Pro evening", "value": int(d["neg_pro"].sum())},
           {"check": "Sensex expiry days (rule) among F2 days", "value": int(d["sensex_expiry"].sum())},
           {"check": "days missing the 15:15 bar among F2 days", "value": int(d["r_1459_1515"].isna().sum())},
           {"check": "days missing the 15:00 bar among F2 days", "value": int(d["r_1459_1500"].isna().sum())}]
    sx_old = Path("results/sensex_confound/sensex_expiry_days_used.csv")
    if sx_old.exists():
        old = set(pd.to_datetime(pd.read_csv(sx_old)["expiry"]))
        old = {x for x in old if dm.index.min() <= x <= dm.index.max()}
        chk.append({"check": "Sensex days: rule here vs results/sensex_confound/sensex_expiry_days_used.csv (days in one only)",
                    "value": len(old ^ set(sensex))})
    chk = pd.DataFrame(chk)
    chk.to_csv(out / "input_checks.csv", index=False)
    show("input checks", chk)
    say(f"SEBI dates used (data/external/sebi_phases.csv): {', '.join(f'{k} {v.date()}' for k, v in sebi.items())}")
    say()
    fixed_b_cache: dict = {}

    # ---------------------------------------------------------------- item 1
    it1 = item1_unconditional(d, cfg, fixed_b_cache)
    it1.to_csv(out / "item1_unconditional_slope.csv", index=False)
    show("ITEM 1. Unconditional slope of r_last on r_rod (bp per bp). Under short_all hedging theory predicts b > 0", it1,
         ["test", "sample", "n", "coef", "method", "se", "stat", "p_two_sided", "p_one_sided_momentum", "ci95_lo", "ci95_hi", "n_clusters"])
    say(f"[{time.time() - t0:.0f}s] item 1 done")

    # ---------------------------------------------------------------- item 2
    it2 = item2_tables(gex, nets, sebi, dm.index)
    for k, v in it2.items():
        v.to_csv(out / f"item2_{k}.csv", index=False)
    show("ITEM 2. Negative-GEX evenings by year (evening = publish date of the GEX used the next day)",
         it2["by_year"].pivot(index="year", columns="rule", values="share_negative").reset_index())
    q = it2["by_quarter"]
    show("ITEM 2. Negative Pro evenings by quarter", q[q.rule == "pro"])
    show("ITEM 2. Negative-GEX evenings by SEBI phase", it2["by_segment"])
    show("ITEM 2. Runs, autocorrelation and effective numbers of episodes", it2["summary"][it2["summary"]["period"].isin(
        ["all evenings", "2023 to 2024", "2025 onward"])])
    rn = it2["runs"]
    show("ITEM 2. The 12 longest negative Pro runs", rn[(rn.rule == "pro") & (rn.sign == "negative")].sort_values(
        "evenings", ascending=False).head(12))
    show("ITEM 2. Raw position states by category and year (share of days)", it2["positions"])
    say(f"[{time.time() - t0:.0f}s] item 2 done")

    # ---------------------------------------------------------------- item 3
    all_m = ("hac5", "dk", "nw60", "cluster_month", "wild_month", "cluster_episode", "wild_episode",
             "cluster_quarter", "wild_quarter", "perm_relocate_60", "perm_relocate_120", "perm_within_60",
             "perm_within_120")
    rows = []
    for si, sname in enumerate(SAMPLES):
        for spi, spec in enumerate(("binary", "controls")):
            new = infer_f2(d, "pro", spec, sample_mask(d.index, sname), "r_last", all_m, cfg, [SEED, 3, si, spi],
                           fixed_b_cache)
            for r in new:
                r["sample"] = sname
            rows += new
    it3 = pd.DataFrame(rows)
    it3.to_csv(out / "item3_f2_inference.csv", index=False)
    show("ITEM 3. F2 interaction (Pro rule) under inference for a persistent state", it3,
         ["sample", "spec", "n", "n_neg", "n_neg_episodes", "coef", "method", "se", "stat", "p", "n_clusters"])
    say(f"[{time.time() - t0:.0f}s] item 3 done")

    # ---------------------------------------------------------------- item 4
    m4 = ("hac5", "cluster_month", "wild_month", "cluster_episode", "wild_episode", "perm_relocate_60")
    rows = []
    for ri, rule in enumerate(["pro", "fii", "fii_plus_pro", "dii", "client_mirror", "client", "us_convention", "short_all"]):
        for si, sname in enumerate(SAMPLES):
            for spi, spec in enumerate(("binary", "controls")):
                new = infer_f2(d, rule, spec, sample_mask(d.index, sname), "r_last", m4, cfg, [SEED, 4, ri, si, spi],
                               fixed_b_cache)
                for r in new:
                    r["sample"] = sname
                    if rule == "short_all" and r["method"] == "not estimable":
                        r["detail"] = "net GEX < 0 on every evening: the interaction is not identified; see item 1"
                rows += new
    it4 = pd.DataFrame(rows)
    it4.to_csv(out / "item4_other_rules.csv", index=False)
    show("ITEM 4. F2 under other dealer proxies", it4,
         ["rule", "sample", "spec", "n", "n_neg", "n_neg_episodes", "coef", "method", "stat", "p", "n_clusters", "detail"])
    say(f"[{time.time() - t0:.0f}s] item 4 done")

    # ---------------------------------------------------------------- item 5
    m5 = ("hac5", "cluster_month", "wild_month", "cluster_episode", "wild_episode", "perm_relocate_60")
    full = np.ones(len(d), bool)
    pre = d["pre_auction"].to_numpy()
    cases = [("baseline r_last (14:59 to 15:29)", "binary", "r_last", full),
             ("14:59 to 15:15 only", "binary", "r_1459_1515", full),
             ("15:15 to 15:29 only", "binary", "r_1515_1529", full),
             ("first bar dropped: 15:00 to 15:29", "binary", "r_1500_1529", full),
             ("first bar only: 14:59 to 15:00", "binary", "r_1459_1500", full),
             ("baseline r_last, pre-auction", "binary", "r_last", pre),
             ("14:59 to 15:15 only, pre-auction", "binary", "r_1459_1515", pre),
             ("15:15 to 15:29 only, pre-auction", "binary", "r_1515_1529", pre),
             ("first bar dropped: 15:00 to 15:29, pre-auction", "binary", "r_1500_1529", pre),
             ("first bar only: 14:59 to 15:00, pre-auction", "binary", "r_1459_1500", pre),
             ("VIX and move-size controls (as in R5)", "controls", "r_last", full),
             ("year-quarter fixed effects (levels)", "yq_fe", "r_last", full),
             ("year-month fixed effects (levels)", "ym_fe", "r_last", full),
             ("quarter-of-year levels and slopes", "qoy_slopes", "r_last", full),
             ("month-of-year levels and slopes", "moy_slopes", "r_last", full),
             ("year-quarter levels and slopes (within-quarter F2 only)", "yq_slopes", "r_last", full),
             ("January to March: F2 outside and inside", "janmar", "r_last", full),
             ("January to March: difference inside minus outside", "janmar_diff", "r_last", full),
             ("SEBI phases (1 Feb, 1 Apr, 3 Jul 2025) levels and slopes", "sebi", "r_last", full),
             ("SEBI phases plus expiry-regime levels and slopes", "sebi_regime", "r_last", full),
             ("Sensex expiry day level and slope", "sensex", "r_last", full),
             ("Sensex expiry days dropped", "binary", "r_last", d["sensex_expiry"].to_numpy() == 0),
             ("all controls together", "all_controls", "r_last", full)]
    rows = []
    for ci, (label, spec, ycol, m) in enumerate(cases):
        new = infer_f2(d, "pro", spec, m, ycol, m5, cfg, [SEED, 5, ci], fixed_b_cache)
        for r in new:
            r["case"] = label
        rows += new
    it5 = pd.DataFrame(rows)
    it5.to_csv(out / "item5_alternatives.csv", index=False)
    show("ITEM 5. Alternative explanations (Pro rule)", it5,
         ["case", "term", "n", "n_neg", "coef", "method", "stat", "p", "n_clusters"])
    say(f"[{time.time() - t0:.0f}s] item 5 done")

    # ---------------------------------------------------------------- item 6
    rets_ds = rets
    frame = dsg.valid(dsg.build_frame(dm, panel, nets, rets_ds))
    it6, draws6 = item6_angles(frame, cfg)
    it6.to_csv(out / "item6_angle_equality.csv", index=False)
    draws6.to_csv(out / "item6_angle_bootstrap_draws.csv.gz", index=False)
    show("ITEM 6. Joint block bootstrap: are the dealer-sign angles of outcome (a) and outcome (b) equal?", it6)
    say(f"[{time.time() - t0:.0f}s] item 6 done")

    # ---------------------------------------------------------------- item 7
    d7 = d_all.copy()
    it7, junc = item7_noise(d7, closes, cfg)
    it7.to_csv(out / "item7_minute_autocorrelation.csv", index=False)
    junc.to_csv(out / "item7_junction.csv", index=False)
    show("ITEM 7. First-order autocorrelation of 1-minute Nifty returns", it7)
    show("ITEM 7. Correlation across the 14:59 junction", junc)
    say(f"[{time.time() - t0:.0f}s] item 7 done")

    n_p = (int(it1["p_two_sided"].notna().sum()) + sum(int(t["p"].notna().sum()) for t in (it3, it4, it5))
           + 3 * len(it6) + int(it7["p_two_sided"].notna().sum()) + len(junc))
    say(f"p-values reported by this script (all post hoc, none adjusted for this count): {n_p}")
    info = {"seed": SEED, **cfg, "seconds": round(time.time() - t0, 1), "n_p_values_reported": n_p,
            "inputs_sha256_16": {str(p): _sha(p) for p in (run / "daily_market.csv", run / "daily_gex_panel.csv",
                                                          raw / "nifty_1min.csv", ext / "sensex_expiry_calendar.csv",
                                                          ext / "sebi_phases.csv")},
            "participant_files": len(list((raw / "participant_oi").glob("*.csv"))),
            "status": "post hoc, in response to the referee report; not pre-registered"}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    say(f"Done in {time.time() - t0:.0f}s.")
    (out.parent / "referee_f2_stdout.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
