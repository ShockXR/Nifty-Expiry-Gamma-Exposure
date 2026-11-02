"""P82: estimate the dealer sign from data instead of fixing it, then test it out of sample. REAL DATA.

The problem
-----------
Net dealer gamma exposure (GEX) needs the dealers' net share of each option,
which NSE does not publish. The project so far used three fixed guesses
(src/niftygex/gex.py): "us_convention" (dealers long calls, short puts),
"participant" (dealers = NSE's Pro category) and "client_mirror" (dealers take
the opposite side of NSE's Client category). R3 (calmer next day after high
Pro GEX) fits hedging theory under the Pro sign; R5 (end-of-day reversal after
negative Pro GEX) fits it only with the Pro sign reversed.

GEX measure
-----------
Net GEX is linear in the dealer weights. With G_call and G_put the gamma sums
over calls and puts of the end-of-day chain (rupees per 1% move, from
results/pipeline_nse/daily_gex_panel.csv: gross = G_call + G_put, US-convention
net = G_call - G_put), every candidate is SCALE-FREE: net GEX divided by gross
GEX of the same chain, a number in [-1, 1]. Log gross GEX (relative to its
20-day median) is a control in every model. (The earlier R3 and R5 measure,
net divided by the 20-day median of gross, is mechanically about 4 times larger
on expiry eves; it is kept only as a labelled comparison.)

Two families of dealer weights:

* "shares":       GEX(w) / gross = w_call * s_call + w_put * s_put, with
  s_call = G_call / gross, s_put = G_put / gross (s_call + s_put = 1).
  US convention is w = (1, -1). Because s_call + s_put = 1, the part
  (w_call + w_put)/2 is a constant and drops out against the intercept: this
  family can only reveal the SIGN of w_call - w_put ("are dealers longer
  calls than puts, as the US convention says, or the reverse?").
* "participants": GEX(w) / gross = a * Pro + b * ClientMirror (each the rule's
  scale-free GEX). Pro rule = (1, 0), Client-mirror rule = (0, 1), and
  (-1, 1) is FII plus DII (the four categories sum to zero). A 2-D direction.

Identification (the maintained hypothesis)
------------------------------------------
Participant OI alone cannot say who hedges. The weights are identified only
by assuming hedging theory is right:

  (a) on ordinary (non-expiry) days, next-day log realized variance, beyond a
      HAR baseline with India VIX and log gross GEX, FALLS when dealer gamma
      is higher;
  (b) the end-of-day momentum slope (last-30-minute return on the
      rest-of-day return) RISES when dealer gamma is lower (negative
      coefficient on rest-of-day return x GEX).

Only the direction of w is identified (its length is absorbed by the
regression coefficient), so w is a unit vector (cos phi, sin phi), estimated
on a 1-degree grid by a likelihood-ratio (LR) objective with the theory sign
imposed. The result is a "revealed sign under a maintained hypothesis", not
an observation. Outcome (b) leaves out the 63 closing-auction days (from
3 Aug 2026), on which the last 30 minutes contain 13 frozen minutes and an
auction jump.

Steps: (1) direction estimates on 2023 to 2024 and the full sample, with LR
confidence sets and block-bootstrap bands; (2) a 250-day rolling window and a
time-varying-parameter state-space model (Kalman filter, statsmodels MLE) for
drift; (3) an out-of-sample contest from 1 Jan 2025 with an expanding window
refitted every 21 trading days: estimated-sign GEX against the three fixed
rules, QLIKE for variance, squared error for the last-30-minute return,
Diebold-Mariano tests, sign agreement; (4) robustness: auction days, expiry
regime and days-to-expiry controls, open interest and gross-level controls,
the old R3 scaling; (5) a placebo that shuffles outcome days against the GEX
series; (6) a semi-synthetic validation (simulated outcomes on the real
regressors with known weights).

    python experiments/dealer_sign.py

Writes results/dealer_sign/ and results/dealer_sign_stdout.txt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm
from statsmodels.tsa.statespace.mlemodel import MLEModel

from niftygex import data as nd
from niftygex.evaluation import diebold_mariano, qlike

AUCTION_START = pd.Timestamp("2026-08-03")
TEST_START = pd.Timestamp("2025-01-01")
REFIT_EVERY = 21
ROLL_WINDOW = 250
BLOCK = 10
ANGLES = np.arange(360.0)
LR_CRIT_90 = float(chi2.ppf(0.90, 1))
FAMILIES = {"shares": ("s_call", "s_put"), "participants": ("x_pro", "x_cmirror")}
COMPONENT_NAMES = {"shares": ("w_call", "w_put"), "participants": ("a_pro", "b_client_mirror")}
REFERENCE = {
    "shares": {"US convention": 315.0, "reverse US": 135.0},
    "participants": {"Pro": 0.0, "Client mirror": 90.0, "FII+DII": 135.0, "reverse Pro": 180.0, "Client": 270.0},
}
FIXED_RULES = {"us": "us", "pro": "x_pro", "cmirror": "x_cmirror"}  # rule name -> GEX column
OUTCOME_COLS = ["rv", "y", "y_l1", "y_w5", "y_m22", "lvix_l1", "exp", "regime", "auction", "dtte", "r_rod", "r_last"]
NEEDED = ["y", "y_l1", "y_w5", "y_m22", "lvix_l1", "r_rod", "r_last", "s_call", "s_put", "us", "x_pro", "x_cmirror",
          "l_gross_rel", "gross_rel", "l_oi_rel"]
R3_COLUMNS = {"s_call": "g_call_m", "s_put": "g_put_m", "us": "us_m", "x_pro": "x_pro_m", "x_cmirror": "x_cmirror_m"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


# =====================================================================================
# Data
# =====================================================================================


def participant_nets(directory) -> pd.DataFrame:
    """Net index-option position of each category as a share of market OI, one row per file date.

    For category k and side s (call or put): (long - short) / TOTAL long, the
    same definition as gex.ParticipantWeights.from_participant_oi.
    """
    rows = {}
    for f in sorted(Path(directory).glob("*.csv")):
        try:
            d = pd.to_datetime(f.stem[-8:], format="%d%m%Y")
        except ValueError:
            continue
        t = nd.load_participant_oi(f)
        tc, tp = float(t.loc["TOTAL", "option_index_call_long"]), float(t.loc["TOTAL", "option_index_put_long"])
        r = {}
        for g in ("Client", "DII", "FII", "Pro"):
            r[f"{g.lower()}_call"] = (t.loc[g, "option_index_call_long"] - t.loc[g, "option_index_call_short"]) / tc
            r[f"{g.lower()}_put"] = (t.loc[g, "option_index_put_long"] - t.loc[g, "option_index_put_short"]) / tp
        rows[d] = r
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def build_frame(dm: pd.DataFrame, panel: pd.DataFrame, nets: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """One row per outcome day t+1: outcomes of that day, GEX inputs published at the close of day t.

    Scale-free GEX inputs (net / gross of the same chain): s_call, s_put, us,
    x_pro, x_cmirror, x_fii, x_client. Controls: l_gross_rel = log(gross / its
    20-day median), gross_rel (the ratio itself), l_oi_rel (same for total OI).
    The earlier R3 and R5 scaling (net / 20-day median gross) is kept in the
    *_m columns for comparison. All GEX inputs are lagged one trading day.
    """
    gross = panel["gross_gex"]
    med = gross.rolling(20, min_periods=5).median()
    gc = (gross + panel["net_gex_us_convention"]) / 2
    gp = (gross - panel["net_gex_us_convention"]) / 2
    nt = nets.reindex(panel.index).ffill()
    oi = panel["total_oi"]
    pub = pd.DataFrame({
        "s_call": gc / gross, "s_put": gp / gross, "us": panel["net_gex_us_convention"] / gross,
        "x_pro": panel["net_gex_participant"] / gross, "x_cmirror": panel["net_gex_client_mirror"] / gross,
        "x_fii": (nt["fii_call"] * gc + nt["fii_put"] * gp) / gross,
        "x_client": (nt["client_call"] * gc + nt["client_put"] * gp) / gross,
        "gross_rel": gross / med, "l_gross_rel": np.log(gross / med),
        "l_oi_rel": np.log(oi / oi.rolling(20, min_periods=5).median()),
        "g_call_m": gc / med, "g_put_m": gp / med, "us_m": panel["net_gex_us_convention"] / med,
        "x_pro_m": panel["net_gex_participant"] / med, "x_cmirror_m": panel["net_gex_client_mirror"] / med,
        "pro_call": nt["pro_call"], "pro_put": nt["pro_put"], "client_call": nt["client_call"],
        "client_put": nt["client_put"], "fii_call": nt["fii_call"], "fii_put": nt["fii_put"],
        "dii_call": nt["dii_call"], "dii_put": nt["dii_put"],
    }, index=panel.index)
    out = pd.DataFrame(index=dm.index)
    out.index.name = "date"
    y = np.log(dm["rv_day"].where(dm["rv_day"] > 0))
    out["rv"] = dm["rv_day"]
    out["y"] = y
    out["y_l1"] = y.shift(1)
    out["y_w5"] = y.shift(1).rolling(5).mean()
    out["y_m22"] = y.shift(1).rolling(22).mean()
    out["lvix_l1"] = np.log(dm["vix"]).shift(1)
    out["exp"] = dm["is_expiry"].astype(float)
    out["regime"] = dm["regime"]
    out["auction"] = (out.index >= AUCTION_START).astype(float)
    dtte = dm["trading_days_to_expiry"].copy()
    miss = dtte.isna()
    if miss.any():
        ce = pd.to_datetime(dm.loc[miss, "chain_expiry"])
        dtte[miss] = [np.busday_count(a.date(), b.date()) for a, b in zip(dm.index[miss], ce)]
    out["dtte"] = dtte
    out["r_rod"] = rets["r_rod"].reindex(dm.index) * 1e4
    out["r_last"] = rets["r_last"].reindex(dm.index) * 1e4
    return out.join(pub.shift(1))


def valid(df: pd.DataFrame) -> pd.DataFrame:
    ok = np.isfinite(df[NEEDED].to_numpy(float)).all(axis=1) & (df["rv"] > 0).to_numpy()
    return df[ok].copy()


def r3_scaled(df: pd.DataFrame) -> pd.DataFrame:
    """The same frame with the GEX inputs replaced by the earlier R3/R5 scaling (net / 20-day median gross)."""
    out = df.copy()
    for new, old in R3_COLUMNS.items():
        out[new] = df[old]
    return out


def permute_outcomes(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Placebo: shuffle whole outcome rows (outcomes, their HAR lags, VIX, expiry and auction flags) across days.

    The GEX inputs stay in calendar order, so any link between GEX and outcomes
    is broken while the HAR and momentum structure inside each row is kept.
    """
    out = df.copy()
    cols = [c for c in OUTCOME_COLS if c in df]
    perm = rng.permutation(len(df))
    out[cols] = df[cols].iloc[perm].to_numpy()
    for c in cols:
        if c != "regime":
            out[c] = out[c].astype(float)
    return out


# =====================================================================================
# Designs
# =====================================================================================


def eq_rows(df: pd.DataFrame, eq: str, ctrl: tuple = ("base",)) -> np.ndarray:
    """Rows used by an equation. The end-of-day outcome (b) leaves out closing-auction days
    unless they are modelled as their own regime ("auction_regime")."""
    if eq == "b" and "auction_regime" not in ctrl:
        return (df["auction"].to_numpy(float) == 0)
    return np.ones(len(df), bool)


def controls(df: pd.DataFrame, eq: str, ctrl: tuple = ("base",)) -> pd.DataFrame:
    """Baseline regressors (no dealer sign) for equation a (log RV, HAR) or b (last 30 minutes, momentum).

    Both include log gross GEX relative to its 20-day median (and its
    interaction with the expiry dummy in a, with the rest-of-day return in b).
    """
    e = df["exp"].to_numpy(float)
    rr = df["r_rod"].to_numpy(float)
    lg = df["l_gross_rel"].to_numpy(float)
    c = {"const": np.ones(len(df))}
    if eq == "a":
        for k in ("y_l1", "y_w5", "y_m22", "lvix_l1"):
            c[k] = df[k].to_numpy(float)
        c.update({"exp": e, "l_gross_rel": lg, "l_gross_rel_x_exp": lg * e, "auction": df["auction"].to_numpy(float)})
    else:
        c.update({"r_rod": rr, "l_gross_rel": lg, "r_rod_x_l_gross_rel": rr * lg})
    if "regime" in ctrl:
        for r in ("single_weekly_thu", "single_weekly_tue"):
            ind = (df["regime"] == r).to_numpy(float)
            c[f"reg_{r}"] = ind
            c[f"reg_{r}_x_{'exp' if eq == 'a' else 'r_rod'}"] = ind * (e if eq == "a" else rr)
        if eq == "a":
            for k in (1, 2, 3):
                c[f"dtte_{k}"] = (df["dtte"].to_numpy(float) == k).astype(float)
    if "auction_regime" in ctrl:
        au = df["auction"].to_numpy(float)
        if eq == "a":
            c["auction_x_exp"] = au * e
        else:
            c.update({"auction": au, "auction_x_r_rod": au * rr})
    if "gross_oi" in ctrl:
        g, lo = df["gross_rel"].to_numpy(float), df["l_oi_rel"].to_numpy(float)
        if eq == "a":
            c.update({"gross_rel_x_ord": g * (1 - e), "gross_rel_x_exp": g * e, "l_oi_rel": lo})
        else:
            c.update({"gross_rel": g, "r_rod_x_gross_rel": rr * g, "l_oi_rel": lo, "r_rod_x_l_oi_rel": rr * lo})
    return pd.DataFrame(c, index=df.index)


def multiplier(df: pd.DataFrame, eq: str) -> np.ndarray:
    """The variable that GEX multiplies in the theory-signed term: (1 - expiry) for a, r_rod for b."""
    return (1 - df["exp"].to_numpy(float)) if eq == "a" else df["r_rod"].to_numpy(float)


def outcome(df: pd.DataFrame, eq: str) -> np.ndarray:
    return df["y" if eq == "a" else "r_last"].to_numpy(float)


def is_reduced(df: pd.DataFrame, family: str) -> bool:
    """True when the family's two basis series sum to a constant (scale-free shares: s_call + s_put = 1).

    Then only w_call - w_put is identified, through the single series us = s_call - s_put.
    """
    B = df[list(FAMILIES[family])].to_numpy(float)
    s = B.sum(axis=1)
    return bool(np.allclose(s, s[0], atol=1e-9))


def reduced_basis(df: pd.DataFrame, family: str) -> np.ndarray:
    """The identified part of the family: us (1 column) when reduced, else the two basis series."""
    if is_reduced(df, family):
        return df[["us"]].to_numpy(float)
    return df[list(FAMILIES[family])].to_numpy(float)


def design_arrays(df: pd.DataFrame, eq: str, family: str, ctrl: tuple = ("base",), reduced: bool = False):
    """(y, Z, Bm) for the direction estimator, on the equation's rows.

    Z: baseline controls plus the family's basis entered freely (times the
    expiry dummy in a, as levels in b), so the direction is identified only
    from the theory-signed term. Bm: the two basis series times the
    multiplier, so GEX(w) x multiplier = Bm @ w. With reduced=True, Bm is the
    identified part only (us x multiplier for the scale-free shares family).
    """
    d = df[eq_rows(df, eq, ctrl)]
    Bfree = reduced_basis(d, family)
    B = Bfree if reduced else d[list(FAMILIES[family])].to_numpy(float)
    Z = controls(d, eq, ctrl).to_numpy(float)
    free = Bfree * d["exp"].to_numpy(float)[:, None] if eq == "a" else Bfree
    Z = np.column_stack([Z, free])
    return outcome(d, eq), Z, B * multiplier(d, eq)[:, None]


def usable(Z: np.ndarray, min_nonzero: int = 15) -> np.ndarray:
    """Columns worth estimating in a sample: the constant, and others with enough non-zero values."""
    nz = (np.abs(Z) > 0).sum(axis=0)
    sd = Z.std(axis=0)
    keep = (nz >= min_nonzero) & (sd > 0)
    keep[0] = True
    return keep


# =====================================================================================
# Direction estimator (grid over angles, theory sign imposed)
# =====================================================================================


def unit(angles_deg) -> np.ndarray:
    a = np.deg2rad(np.atleast_1d(np.asarray(angles_deg, float)))
    return np.column_stack([np.cos(a), np.sin(a)])


def gram_stats(y, Z, Bm) -> dict:
    """Sufficient statistics after partialling out Z (Frisch-Waugh): A = R'R, b = R'r_y, SSE0."""
    Z = Z[:, usable(Z)]
    Y = np.column_stack([y, Bm])
    coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    R = Y - Z @ coef
    ry, rB = R[:, 0], R[:, 1:]
    return {"A": rB.T @ rB, "b": rB.T @ ry, "sse0": float(ry @ ry), "n": int(len(y))}


def lr_profile(st: dict, angles=ANGLES, sign: float = -1.0) -> np.ndarray:
    """Gaussian LR gain of adding GEX(w) x multiplier, coefficient restricted to the theory sign.

    For each angle phi, w = (cos phi, sin phi). The unrestricted coefficient
    is (w'b)/(w'Aw); if it has the wrong sign the restricted fit puts it at
    zero and the gain is 0. The profile is unchanged when w is scaled by a
    positive number, which is why only the direction is identified.
    """
    W = unit(angles)
    wb = W @ st["b"]
    wAw = np.einsum("ki,ij,kj->k", W, st["A"], W)
    tol = 1e-9 * max(float(np.trace(st["A"])), 1e-300)
    gain = np.where((sign * wb > 0) & (wAw > tol), wb**2 / np.maximum(wAw, tol), 0.0)
    gain = np.minimum(gain, st["sse0"] * (1 - 1e-12))
    return st["n"] * np.log(st["sse0"] / (st["sse0"] - gain))


def _runs(mask: np.ndarray) -> list:
    """Contiguous runs of True on the circle as (start, end) index pairs (end may be < start when wrapping)."""
    n = len(mask)
    if mask.all():
        return [(0, n - 1)]
    if not mask.any():
        return []
    k = int(np.argmin(mask))
    runs, cur = [], None
    for j in range(1, n + 1):
        i = (k + j) % n
        if mask[i] and cur is None:
            cur = [i, i]
        elif mask[i]:
            cur[1] = i
        elif cur is not None:
            runs.append(tuple(cur))
            cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def angle_set_text(mask: np.ndarray, angles=ANGLES) -> str:
    if mask.all():
        return "all angles"
    if not mask.any():
        return "none"
    return "; ".join(f"{angles[a]:.0f} to {angles[b]:.0f}" for a, b in _runs(mask))


def _best_angle_index(J: np.ndarray, rtol: float = 1e-6) -> int:
    """Argmax of the profile; when the maximum is a flat run (a family that identifies only a half-plane),
    the middle of that run."""
    m = float(J.max())
    k = int(np.argmax(J))
    if m <= 0:
        return k
    tie = J >= m - rtol * max(1.0, m)
    if tie.sum() <= 1:
        return k
    n = len(J)
    for a, b in _runs(tie):
        length = (b - a) % n + 1
        if (k - a) % n < length:
            return int((a + (length - 1) // 2) % n)
    return k


def estimate_direction(stats: list, angles=ANGLES, level_crit: float = LR_CRIT_90) -> dict:
    """Best angle of the summed LR profiles, and the profile-LR confidence set."""
    prof = [lr_profile(s, angles) for s in stats]
    J = np.sum(prof, axis=0)
    k = _best_angle_index(J)
    return {"angle": float(angles[k]), "lr_max": float(J[k]), "set": J >= J[k] - level_crit,
            "profile": J, "parts": prof, "part_angles": [float(angles[_best_angle_index(p)]) for p in prof]}


def block_indices(n: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, max(n - block, 1), int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap_direction(arrs: list, n_boot: int, rng, block: int = BLOCK) -> np.ndarray:
    """Moving-block bootstrap of the angle: columns = joint, then each equation's own best angle.

    Blocks are drawn on each equation's own rows (the equations can have
    different rows, for example when the end-of-day outcome leaves out the
    closing-auction days).
    """
    out = np.full((n_boot, 1 + len(arrs)), np.nan)
    for r in range(n_boot):
        st = []
        for y, Z, Bm in arrs:
            idx = block_indices(len(y), block, rng)
            st.append(gram_stats(y[idx], Z[idx], Bm[idx]))
        est = estimate_direction(st)
        out[r, 0] = est["angle"]
        for j, p in enumerate(est["parts"]):
            out[r, 1 + j] = est["part_angles"][j] if p.max() > 0 else np.nan
    return out


def circ_diff(a, b):
    """a - b wrapped to (-180, 180]."""
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def boot_bands(boot_angles: np.ndarray, center: float, q=(5, 95)) -> dict:
    """Bootstrap 5 to 95% bands of the angle and of each unit-weight component, and the share of draws
    within 90 degrees of the estimate (the estimated sign is the same)."""
    a = boot_angles[np.isfinite(boot_angles)]
    out = {"angle_lo": np.nan, "angle_hi": np.nan, "angle_width": np.nan, "w1_lo": np.nan, "w1_hi": np.nan,
           "w2_lo": np.nan, "w2_hi": np.nan, "share_same_side": np.nan, "n_boot_valid": int(a.size)}
    if a.size == 0:
        return out
    d = circ_diff(a, center)
    W = unit(a)
    out.update({"angle_lo": float((center + np.percentile(d, q[0], method="nearest")) % 360),
                "angle_hi": float((center + np.percentile(d, q[1], method="nearest")) % 360),
                "angle_width": float(np.percentile(d, q[1], method="nearest") - np.percentile(d, q[0], method="nearest")),
                "w1_lo": float(np.percentile(W[:, 0], q[0], method="nearest")), "w1_hi": float(np.percentile(W[:, 0], q[1], method="nearest")),
                "w2_lo": float(np.percentile(W[:, 1], q[0], method="nearest")), "w2_hi": float(np.percentile(W[:, 1], q[1], method="nearest")),
                "share_same_side": float((np.abs(d) < 90).mean())})
    return out


def hac_theta(y, Z, Bm, lags: int = 5) -> dict:
    """Unrestricted OLS loading theta on the identified GEX columns, HAC Wald test of theta = 0."""
    Z = Z[:, usable(Z)]
    X = np.column_stack([Z, Bm])
    k = Bm.shape[1]
    f = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    th = np.asarray(f.params[-k:])
    V = np.asarray(f.cov_params())[-k:, -k:]
    W = float(th @ np.linalg.solve(V, th))
    out = {"theta1": th[0], "se1": float(np.sqrt(V[0, 0])), "theta2": np.nan, "se2": np.nan,
           "wald": W, "wald_df": k, "wald_p": float(chi2.sf(W, k)), "n": int(f.nobs)}
    if k == 2:
        out.update({"theta2": th[1], "se2": float(np.sqrt(V[1, 1]))})
    return out


def direction_table(df: pd.DataFrame, label: str, ctrl: tuple, rng, n_boot: int) -> tuple:
    """Direction estimates for one sample: per family, objectives a, b and joint."""
    rows, profs = [], []
    for fam in FAMILIES:
        arrs = [design_arrays(df, eq, fam, ctrl) for eq in ("a", "b")]
        st = [gram_stats(*a) for a in arrs]
        est = estimate_direction(st)
        red = is_reduced(df, fam)
        boot = bootstrap_direction(arrs, n_boot, rng) if n_boot else None
        objs = {"joint": (est["profile"], est["angle"])}
        for j, eq in enumerate(("a", "b")):
            objs[eq] = (est["parts"][j], est["part_angles"][j])
        for obj, (p, ang) in objs.items():
            s = p >= p.max() - LR_CRIT_90
            w = unit(ang)[0]
            row = {"sample": label, "controls": "+".join(ctrl), "family": fam,
                   "identified": "sign of w_call - w_put only" if red else "2-D direction",
                   "objective": obj, "n_days_a": len(arrs[0][0]), "n_days_b": len(arrs[1][0]),
                   "angle_deg": ang, COMPONENT_NAMES[fam][0]: w[0], COMPONENT_NAMES[fam][1]: w[1],
                   "lr_max": float(p.max()), "lr_set_90": angle_set_text(s), "lr_set_width_deg": int(s.sum())}
            for name, ref in REFERENCE[fam].items():
                row[f"in_set: {name}"] = bool(s[int(round(ref)) % 360])
            if obj != "joint":
                h = hac_theta(*design_arrays(df, obj, fam, ctrl, reduced=True))
                row.update({k: h[k] for k in ("theta1", "se1", "theta2", "se2", "wald", "wald_df", "wald_p")})
            if boot is not None:
                col = {"joint": 0, "a": 1, "b": 2}[obj]
                row.update({f"boot_{k}": v for k, v in boot_bands(boot[:, col], ang).items()})
            rows.append(row)
            profs.append(pd.DataFrame({"sample": label, "controls": "+".join(ctrl), "family": fam, "objective": obj,
                                       "angle_deg": ANGLES, "lr": p}))
    return pd.DataFrame(rows), pd.concat(profs, ignore_index=True)


# =====================================================================================
# Time-varying weights: state-space model, Kalman filter
# =====================================================================================


class TVPRegression(MLEModel):
    """y_t = x_t' delta + z_t' theta_t + e_t, theta_t = theta_{t-1} + eta_t (random walk).

    delta: fixed coefficients (states with no noise, exact diffuse start);
    theta: the GEX loadings, which drift with variance q per step.
    e_t ~ N(0, sigma2), eta_t ~ N(0, q I). Parameters (sigma2, q) by maximum likelihood.
    """

    def __init__(self, endog, x_fixed, z_tvp):
        x_fixed, z_tvp = np.asarray(x_fixed, float), np.asarray(z_tvp, float)
        self.k_f, self.k_v = x_fixed.shape[1], z_tvp.shape[1]
        k = self.k_f + self.k_v
        super().__init__(np.asarray(endog, float), k_states=k, k_posdef=self.k_v, initialization="diffuse")
        self["design"] = np.column_stack([x_fixed, z_tvp]).T[None, :, :]
        self["transition"] = np.eye(k)
        self["selection"] = np.vstack([np.zeros((self.k_f, self.k_v)), np.eye(self.k_v)])
        self._vy = float(np.nanvar(endog))

    @property
    def param_names(self):
        return ["sigma2", "q"]

    @property
    def start_params(self):
        return np.array([0.8 * self._vy, 1e-4 * self._vy])

    def transform_params(self, unconstrained):
        return np.asarray(unconstrained, float) ** 2

    def untransform_params(self, constrained):
        return np.sqrt(np.asarray(constrained, float))

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self["obs_cov", 0, 0] = params[0]
        self["state_cov"] = np.eye(self.k_v) * params[1]


def fit_tvp(df: pd.DataFrame, eq: str, family: str, train_end=TEST_START, ctrl: tuple = ("base",)) -> dict:
    """MLE on the equation's rows before ``train_end``, then filter and smooth all its rows with those parameters.

    The filtered state at day t uses data up to t only, so its one-step-ahead
    forecasts after ``train_end`` are out of sample. The GEX regressors (the
    identified part only) are scaled by their training standard deviation, so
    one q fits both, and the loadings are reported back in original units.
    """
    rows = eq_rows(df, eq, ctrl)
    y, Z, Bm = design_arrays(df, eq, family, ctrl, reduced=True)
    idx = df.index[rows]
    tr = idx < train_end
    Z = Z[:, usable(Z[tr])]
    sd = Bm[tr].std(axis=0)
    zs = Bm / sd
    m_tr = TVPRegression(y[tr], Z[tr], zs[tr])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_tr = m_tr.fit(disp=False, maxiter=500)
        with m_tr.fix_params({"q": 0.0}):
            res0 = m_tr.fit(disp=False)
    lr_q = max(2 * (res_tr.llf - res0.llf), 0.0)
    m = TVPRegression(y, Z, zs)
    filt = m.filter(res_tr.params)
    smo = m.smooth(res_tr.params)
    k0 = Z.shape[1]
    S = np.outer(sd, sd)[None]
    return {"params": dict(zip(m.param_names, res_tr.params)), "llf_train": float(res_tr.llf),
            "llf_q0": float(res0.llf), "lr_q": float(lr_q), "p_q0": float(0.5 * chi2.sf(lr_q, 1)) if lr_q > 0 else 1.0,
            "theta_f": filt.filtered_state[k0:].T / sd, "P_f": filt.filtered_state_cov[k0:, k0:].transpose(2, 0, 1) / S,
            "theta_s": smo.smoothed_state[k0:].T / sd, "P_s": smo.smoothed_state_cov[k0:, k0:].transpose(2, 0, 1) / S,
            "forecast": np.asarray(filt.forecasts[0]), "y": y, "index": idx, "rows": rows, "train": tr}


def tvp_direction(theta: np.ndarray, P: np.ndarray, rng, n_draw: int = 1000) -> pd.DataFrame:
    """Revealed direction from a filtered loading theta_t (direction = -theta_t), with 5 to 95% bands.

    Two loadings: unit vector -theta/|theta|, bands from draws of N(theta_t, P_t).
    One loading (scale-free shares family, theta on us): the side, w = (1, -1)/sqrt 2
    when theta < 0 (US side), and p_us_side = P(theta_t < 0) = Phi(-theta/se).
    """
    n, k = theta.shape
    out = {c: np.full(n, np.nan) for c in ("w1", "w2", "w1_lo", "w1_hi", "w2_lo", "w2_hi", "wald", "p_us_side")}
    if k == 1:
        se = np.sqrt(P[:, 0, 0])
        okk = np.isfinite(se) & (se > 0) & np.isfinite(theta[:, 0])
        s = -np.sign(theta[okk, 0])
        out["w1"][okk], out["w2"][okk] = s / np.sqrt(2), -s / np.sqrt(2)
        out["wald"][okk] = (theta[okk, 0] / se[okk]) ** 2
        out["p_us_side"][okk] = norm.cdf(-theta[okk, 0] / se[okk])
        return pd.DataFrame(out)
    eps = rng.standard_normal((n_draw, 2))
    for t in range(n):
        Pt = 0.5 * (P[t] + P[t].T)
        if not np.all(np.isfinite(Pt)) or not np.all(np.isfinite(theta[t])):
            continue
        ev, V = np.linalg.eigh(Pt)
        if ev.max() <= 0:
            continue
        ev = np.maximum(ev, 1e-12 * ev.max())
        L = V * np.sqrt(ev)
        d = -(theta[t][None, :] + eps @ L.T)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        c = -theta[t] / np.linalg.norm(theta[t])
        out["w1"][t], out["w2"][t] = c
        out["w1_lo"][t], out["w1_hi"][t] = np.percentile(d[:, 0], [5, 95])
        out["w2_lo"][t], out["w2_hi"][t] = np.percentile(d[:, 1], [5, 95])
        out["wald"][t] = float(theta[t] @ np.linalg.solve((V * ev) @ V.T, theta[t]))
    return pd.DataFrame(out)


# =====================================================================================
# Out-of-sample contest
# =====================================================================================


def template_columns(df: pd.DataFrame, eq: str, g: np.ndarray):
    """The two GEX columns of the forecasting template and the index of the theory-signed one.

    a: GEX x (1 - expiry) [signed, theory says negative] and GEX x expiry [free].
    b: GEX [free] and r_rod x GEX [signed, theory says negative].
    """
    e = df["exp"].to_numpy(float)
    if eq == "a":
        return np.column_stack([g * (1 - e), g * e]), 0
    return np.column_stack([g, df["r_rod"].to_numpy(float) * g]), 1


def fit_predict(X, y, tr, te, signed_col=None, sign=-1.0):
    """OLS on rows tr, prediction on rows te. With signed_col, a coefficient of the wrong sign is set to 0."""
    keep = usable(X[tr])
    cols = np.where(keep)[0]
    b, *_ = np.linalg.lstsq(X[np.ix_(tr, cols)], y[tr], rcond=None)
    coef_signed = np.nan
    if signed_col is not None and keep[signed_col]:
        j = int(np.where(cols == signed_col)[0][0])
        coef_signed = float(b[j])
        if sign * b[j] < 0:
            cols = cols[cols != signed_col]
            b, *_ = np.linalg.lstsq(X[np.ix_(tr, cols)], y[tr], rcond=None)
            coef_signed = 0.0
    resid = y[tr] - X[np.ix_(tr, cols)] @ b
    return X[np.ix_(te, cols)] @ b, resid, coef_signed


def oos_run(df: pd.DataFrame, ctrl: tuple = ("base",), test_start=TEST_START, refit_every: int = REFIT_EVERY,
            n_boot: int = 0, rng=None, variants=("signed", "free")) -> dict:
    """Expanding-window contest. At each refit: estimate the joint direction for both families on all
    earlier days, then fit every candidate's forecasting regression and predict the next block.

    Candidates: none (baseline, which already has log gross GEX), the three
    fixed rules (us, pro, cmirror), est_<family> (direction re-estimated at
    every refit) and est_<family>_frozen (direction fixed at the first
    estimate, i.e. on data before ``test_start``). Variant "signed" imposes
    the theory sign on the GEX coefficient (a wrong sign is set to 0);
    "free" does not, and then a rule and its reverse forecast identically.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(df)
    test_idx = np.where(df.index >= test_start)[0]
    blocks = [test_idx[i:i + refit_every] for i in range(0, len(test_idx), refit_every)]
    masks = {eq: eq_rows(df, eq, ctrl) for eq in ("a", "b")}
    base = {eq: controls(df, eq, ctrl).to_numpy(float) for eq in ("a", "b")}
    ys = {eq: outcome(df, eq) for eq in ("a", "b")}
    Bfam = {fam: df[list(cols)].to_numpy(float) for fam, cols in FAMILIES.items()}
    fixed = {k: df[c].to_numpy(float) for k, c in FIXED_RULES.items()}
    pred = {}
    gex_test = {f"est_{fam}": np.full(n, np.nan) for fam in FAMILIES}
    path, frozen = [], {}
    for bi, blk in enumerate(blocks):
        tr_all = np.arange(blk[0])
        G = {"none": None, **fixed}
        for fam in FAMILIES:
            arrs = [design_arrays(df.iloc[tr_all], eq, fam, ctrl) for eq in ("a", "b")]
            est = estimate_direction([gram_stats(*a) for a in arrs])
            w = unit(est["angle"])[0]
            if bi == 0:
                frozen[fam] = w
            G[f"est_{fam}"] = Bfam[fam] @ w
            G[f"est_{fam}_frozen"] = Bfam[fam] @ frozen[fam]
            gex_test[f"est_{fam}"][blk] = G[f"est_{fam}"][blk]
            row = {"refit_date": df.index[blk[0]], "train_days": len(tr_all), "family": fam,
                   "angle_joint": est["angle"], "w1": w[0], "w2": w[1], "lr_joint": est["lr_max"],
                   "lr_set_width_deg": int(est["set"].sum()), "angle_a": est["part_angles"][0],
                   "angle_b": est["part_angles"][1], "lr_a": float(est["parts"][0].max()),
                   "lr_b": float(est["parts"][1].max())}
            if n_boot:
                row.update(boot_bands(bootstrap_direction(arrs, n_boot, rng)[:, 0], est["angle"]))
            path.append(row)
        for eq in ("a", "b"):
            tr = tr_all[masks[eq][tr_all]]
            for name, g in G.items():
                for var in variants:
                    if g is None:
                        X, sc = base[eq], None
                    else:
                        tc, sc = template_columns(df, eq, g)
                        X = np.column_stack([base[eq], tc])
                        sc = base[eq].shape[1] + sc
                    p, resid, cs = fit_predict(X, ys[eq], tr, blk, sc if var == "signed" else None)
                    key = (eq, var, name)
                    if key not in pred:
                        pred[key] = {"pred": np.full(n, np.nan), "smear": np.full(n, np.nan), "coef": []}
                    pred[key]["pred"][blk] = p
                    pred[key]["smear"][blk] = float(np.mean(np.exp(resid))) if eq == "a" else np.nan
                    pred[key]["coef"].append(cs)
    return {"pred": pred, "path": pd.DataFrame(path), "test_idx": test_idx, "gex_test": gex_test, "frozen": frozen,
            "masks": masks}


def losses_from_run(df: pd.DataFrame, run: dict, extra: dict | None = None) -> pd.DataFrame:
    """Daily losses on the test days: QLIKE on realized variance (a), squared error in bp^2 (b).

    Outcome b is scored only on its own rows (closing-auction days are left out in the main run).
    """
    rows = []
    rv, rl = df["rv"].to_numpy(float), df["r_last"].to_numpy(float)
    items = dict(run["pred"])
    if extra:
        items.update(extra)
    for (eq, var, name), v in items.items():
        ti = run["test_idx"][run["masks"][eq][run["test_idx"]]]
        p = v["pred"][ti]
        if eq == "a":
            L = qlike(rv[ti], np.exp(p) * v["smear"][ti])
        else:
            L = (rl[ti] - p) ** 2
        rows.append(pd.DataFrame({"date": df.index[ti], "eq": eq, "variant": var, "model": name, "loss": L}))
    return pd.concat(rows, ignore_index=True)


def summarize_losses(L: pd.DataFrame, samples: dict) -> tuple:
    """Mean losses, % vs the no-dealer-sign baseline, and Diebold-Mariano tests (HLN correction, daily losses)."""
    summ, dm = [], []
    for sname, mask_fn in samples.items():
        Ls = L[mask_fn(L["date"])]
        for (eq, var), g in Ls.groupby(["eq", "variant"]):
            wide = g.pivot(index="date", columns="model", values="loss")
            if "none" not in wide:
                base_l = Ls[(Ls["eq"] == eq) & (Ls.model == "none") & (Ls.variant == "signed")].set_index("date")["loss"]
                wide["none"] = base_l.reindex(wide.index)
            for m in wide.columns:
                d = diebold_mariano(wide["none"], wide[m], alternative="greater") if m != "none" else {"p": np.nan, "stat": np.nan}
                summ.append({"sample": sname, "eq": eq, "variant": var, "model": m, "n_days": int(wide[m].notna().sum()),
                             "mean_loss": float(wide[m].mean()),
                             "pct_vs_none": float(100 * (wide[m].mean() / wide["none"].mean() - 1)),
                             "dm_stat_vs_none": d["stat"], "p_better_than_none": d["p"]})
            for est in [c for c in wide.columns if c.startswith("est_") or c.startswith("kalman_")]:
                for rule in FIXED_RULES:
                    if rule not in wide:
                        continue
                    diff = (wide[rule] - wide[est]).abs().max()
                    one = diebold_mariano(wide[rule], wide[est], alternative="greater")
                    two = diebold_mariano(wide[rule], wide[est])
                    same = bool(diff <= 1e-9 * max(1.0, float(wide[rule].abs().mean())))
                    dm.append({"sample": sname, "eq": eq, "variant": var, "estimated": est, "fixed_rule": rule,
                               "n_days": one["n"], "identical_forecasts": same,
                               "mean_loss_est": float(wide[est].mean()), "mean_loss_rule": float(wide[rule].mean()),
                               "pct_est_vs_rule": float(100 * (wide[est].mean() / wide[rule].mean() - 1)),
                               "dm_stat": np.nan if same else one["stat"],
                               "p_est_better": np.nan if same else one["p"],
                               "p_two_sided": np.nan if same else two["p"]})
    return pd.DataFrame(summ), pd.DataFrame(dm)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment; NaN entries (identical forecasts, no test) are left out."""
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (len(ok) - i) * p[k])
        adj[k] = min(running, 1.0)
    return adj


def sign_agreement(df: pd.DataFrame, run: dict) -> pd.DataFrame:
    """Share of days on which the estimated-sign GEX and each fixed rule's GEX have the same sign."""
    rows = []
    ti = run["test_idx"]
    tr = np.arange(ti[0])
    for fam in FAMILIES:
        g_test = run["gex_test"][f"est_{fam}"][ti]
        g_train = df[list(FAMILIES[fam])].to_numpy(float)[tr] @ run["frozen"][fam]
        for rule, col in FIXED_RULES.items():
            r_all = df[col].to_numpy(float)
            for period, g, r, dates in (("training 2023 to 2024 (direction from that period)", g_train, r_all[tr], df.index[tr]),
                                        ("test, expanding refits", g_test, r_all[ti], df.index[ti])):
                agree = np.sign(g) == np.sign(r)
                rows.append({"family": fam, "fixed_rule": rule, "period": period, "quarter": "all", "n_days": len(g),
                             "share_same_sign": float(agree.mean()), "share_est_negative": float((g < 0).mean()),
                             "share_rule_negative": float((r < 0).mean()), "corr": float(np.corrcoef(g, r)[0, 1])})
                q = pd.PeriodIndex(dates, freq="Q").astype(str)
                for qq in pd.unique(q):
                    m = q == qq
                    rows.append({"family": fam, "fixed_rule": rule, "period": period, "quarter": qq, "n_days": int(m.sum()),
                                 "share_same_sign": float(agree[m].mean()), "share_est_negative": float((g[m] < 0).mean()),
                                 "share_rule_negative": float((r[m] < 0).mean())})
    return pd.DataFrame(rows)


def rolling_directions(df: pd.DataFrame, window: int, step: int, n_boot: int, rng, ctrl=("base",)) -> pd.DataFrame:
    rows = []
    for end in range(window, len(df) + 1, step):
        sub = df.iloc[end - window:end]
        for fam in FAMILIES:
            arrs = [design_arrays(sub, eq, fam, ctrl) for eq in ("a", "b")]
            est = estimate_direction([gram_stats(*a) for a in arrs])
            w = unit(est["angle"])[0]
            row = {"window_end": sub.index[-1], "window_start": sub.index[0], "family": fam, "angle_joint": est["angle"],
                   "w1": w[0], "w2": w[1], "lr_joint": est["lr_max"], "lr_set_width_deg": int(est["set"].sum()),
                   "angle_a": est["part_angles"][0], "angle_b": est["part_angles"][1],
                   "lr_a": float(est["parts"][0].max()), "lr_b": float(est["parts"][1].max())}
            if n_boot:
                row.update(boot_bands(bootstrap_direction(arrs, n_boot, rng)[:, 0], est["angle"]))
            rows.append(row)
    return pd.DataFrame(rows)


def fixed_rule_fits(df: pd.DataFrame, label: str, ctrl=("base",)) -> pd.DataFrame:
    """In-sample template fits of each fixed rule (HAC, 5 lags): the theory-signed coefficient and its size."""
    rows = []
    for eq in ("a", "b"):
        d = df[eq_rows(df, eq, ctrl)]
        Z = controls(d, eq, ctrl).to_numpy(float)
        Z = Z[:, usable(Z)]
        y = outcome(d, eq)
        base_f = sm.OLS(y, Z).fit()
        for rule, col in FIXED_RULES.items():
            tc, sc = template_columns(d, eq, d[col].to_numpy(float))
            f = sm.OLS(y, np.column_stack([Z, tc])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            j = Z.shape[1] + sc
            snr = abs(f.params[j]) * np.std(tc[:, sc]) / np.sqrt(f.mse_resid)
            rows.append({"sample": label, "controls": "+".join(ctrl), "eq": eq, "rule": rule,
                         "signed_term": "GEX x (1 - expiry)" if eq == "a" else "r_rod x GEX",
                         "coef": float(f.params[j]), "se_hac": float(f.bse[j]), "p": float(f.pvalues[j]),
                         "theory_sign_ok": bool(f.params[j] < 0), "signal_to_noise": float(snr),
                         "other_gex_term": "GEX x expiry" if eq == "a" else "GEX level",
                         "other_gex_coef": float(f.params[Z.shape[1] + 1 - sc]),
                         "other_gex_p": float(f.pvalues[Z.shape[1] + 1 - sc]),
                         "r2": float(f.rsquared), "r2_baseline": float(base_f.rsquared), "n": int(f.nobs)})
    return pd.DataFrame(rows)


# =====================================================================================
# Semi-synthetic validation and synthetic frames (for tests)
# =====================================================================================


def semi_synthetic(df: pd.DataFrame, family: str, snrs, n_rep: int, rng, ctrl=("base",)) -> pd.DataFrame:
    """Simulated outcomes on the REAL regressors with known dealer weights (synthetic outcomes).

    For each replication: draw a true angle uniformly; outcome = baseline fit
    (real data, no dealer-sign term) + an effect along GEX(w_true) x multiplier
    with the theory sign + real residuals shuffled across days. The effect is
    the part of GEX(w_true) x multiplier not explained by the controls,
    scaled so that its sd / residual sd equals ``snr``. For the scale-free
    shares family only the side (sign of w_call - w_put) can be recovered,
    so true angles within 10 degrees of 45 or 225 (w_call = w_put) are
    redrawn.
    """
    arrs = {eq: design_arrays(df, eq, family, ctrl) for eq in ("a", "b")}
    red = is_reduced(df, family)
    fit0, e0, Zu = {}, {}, {}
    for eq, (y, Z, Bm) in arrs.items():
        Zu[eq] = Z[:, usable(Z)]
        c, *_ = np.linalg.lstsq(Zu[eq], y, rcond=None)
        fit0[eq], e0[eq] = Zu[eq] @ c, y - Zu[eq] @ c
    rows = []
    for snr in snrs:
        for r in range(n_rep):
            while True:
                ang = float(rng.uniform(0, 360))
                if not red or min(abs(circ_diff(ang, 45)), abs(circ_diff(ang, 225))) > 10:
                    break
            w = unit(ang)[0]
            st = []
            for eq in ("a", "b"):
                y, Z, Bm = arrs[eq]
                sig = Bm @ w
                cs, *_ = np.linalg.lstsq(Zu[eq], sig, rcond=None)
                sig = sig - Zu[eq] @ cs
                ysim = fit0[eq] - snr * np.std(e0[eq]) / np.std(sig) * sig + rng.permutation(e0[eq])
                st.append(gram_stats(ysim, Z, Bm))
            est = estimate_direction(st)
            err = float(abs(circ_diff(est["angle"], ang)))
            same_side = (np.sign(np.cos(np.deg2rad(est["angle"])) - np.sin(np.deg2rad(est["angle"]))) ==
                         np.sign(w[0] - w[1])) if red else err < 90
            rows.append({"family": family, "identified": "side only" if red else "2-D direction", "snr": snr, "rep": r,
                         "true_angle": ang, "est_angle": est["angle"], "abs_error_deg": np.nan if red else err,
                         "correct_side": bool(same_side), "covered_90": bool(est["set"][int(round(ang)) % 360]),
                         "lr_max": est["lr_max"], "set_width_deg": int(est["set"].sum()),
                         "abs_error_a": np.nan if red else float(abs(circ_diff(est["part_angles"][0], ang))),
                         "abs_error_b": np.nan if red else float(abs(circ_diff(est["part_angles"][1], ang)))})
    return pd.DataFrame(rows)


def synthetic_frame(n: int = 700, family: str = "participants", angle=30.0, snr_a: float = 0.0, snr_b: float = 0.0,
                    seed: int = 0, scaling: str = "scale_free") -> pd.DataFrame:
    """A fully synthetic frame with the same columns as the real one and KNOWN dealer weights.

    ``angle`` is the true direction in degrees (a number, or an array of length
    n for weights that drift). GEX(w) = B @ w with B the family's basis
    (scale-free, or the R3 scaling when scaling="r3", in which the shares
    family is a full 2-D direction). Outcome a: HAR-type log RV with an effect
    beta * GEX(w) on non-expiry days, beta < 0. Outcome b: last-30-minute
    return with slope (-0.03 + c * GEX(w)) on the rest-of-day return, c < 0.
    The effect sizes are signal sd / noise sd.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    pos = np.arange(n) % 5
    exp = (pos == 3).astype(float)
    dtte = ((3 - pos) % 5).astype(float)

    def ar(phi, sd):
        x = np.zeros(n)
        e = rng.standard_normal(n) * sd
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        return x

    gross_rel = (0.6 + 0.6 * (4 - dtte) / 4) * np.exp(ar(0.7, 0.15))
    s_call = np.clip(0.55 + ar(0.9, 0.08 * np.sqrt(1 - 0.81)), 0.2, 0.8)
    s_put = 1 - s_call
    pro_c, pro_p = ar(0.95, 0.01), ar(0.95, 0.01) + 0.005
    cm_c, cm_p = ar(0.95, 0.01) + 0.01, ar(0.95, 0.01) + 0.08
    x_pro, x_cm = pro_c * s_call + pro_p * s_put, cm_c * s_call + cm_p * s_put
    cols = {"s_call": s_call, "s_put": s_put, "us": s_call - s_put, "x_pro": x_pro, "x_cmirror": x_cm,
            "g_call_m": s_call * gross_rel, "g_put_m": s_put * gross_rel, "us_m": (s_call - s_put) * gross_rel,
            "x_pro_m": x_pro * gross_rel, "x_cmirror_m": x_cm * gross_rel}
    src = {k: cols[R3_COLUMNS[k]] if scaling == "r3" else cols[k] for k in R3_COLUMNS}
    B = np.column_stack([src[c] for c in FAMILIES[family]])
    W = unit(np.broadcast_to(np.asarray(angle, float), (n,)))
    G = (B * W).sum(axis=1)
    G = G - G.mean()
    na = rng.standard_normal(n) * 0.45
    sig_a = G * (1 - exp)
    beta = -snr_a * 0.45 / np.std(sig_a) if snr_a else 0.0
    lvix = -2.0 + ar(0.97, 0.05)
    y = np.zeros(n)
    y[0] = -9.5
    for t in range(1, n):
        w5 = y[max(0, t - 5):t].mean()
        y[t] = (-9.5 + 0.35 * (y[t - 1] + 9.5) + 0.3 * (w5 + 9.5) + 0.8 * (lvix[t - 1] + 2.0) + 0.25 * exp[t]
                + beta * sig_a[t] + na[t])
    ys = pd.Series(y, index=idx)
    r_rod = rng.standard_normal(n) * 60.0
    sig_b = r_rod * G
    c = -snr_b * 15.0 / np.std(sig_b) if snr_b else 0.0
    r_last = -0.03 * r_rod + c * sig_b + rng.standard_normal(n) * 15.0
    out = pd.DataFrame({
        "rv": np.exp(y), "y": y, "y_l1": ys.shift(1).to_numpy(), "y_w5": ys.shift(1).rolling(5).mean().to_numpy(),
        "y_m22": ys.shift(1).rolling(22).mean().to_numpy(), "lvix_l1": pd.Series(lvix).shift(1).to_numpy(),
        "exp": exp, "regime": "multi_weekly", "auction": 0.0, "dtte": dtte, "r_rod": r_rod, "r_last": r_last,
        **cols, "x_fii": x_cm - x_pro, "x_client": -x_cm, "gross_rel": gross_rel, "l_gross_rel": np.log(gross_rel),
        "l_oi_rel": ar(0.8, 0.05),
    }, index=pd.DatetimeIndex(idx, name="date"))
    return r3_scaled(out) if scaling == "r3" else out


# =====================================================================================
# Figures
# =====================================================================================


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _hline(ax, yv, name, ls="--"):
    ax.axhline(yv, color=INK, linestyle=ls, linewidth=1.0)
    ax.annotate(name, xy=(1.0, yv), xycoords=("axes fraction", "data"), xytext=(3, 0), textcoords="offset points",
                fontsize=7, color=INK, va="center")


def _vlines(ax):
    ax.axvline(TEST_START, color=MUTED, linewidth=0.8)
    ax.axvline(AUCTION_START, color=MUTED, linewidth=0.8)


def plot_weights(path_exp: pd.DataFrame, roll: pd.DataFrame, out: Path):
    """Three panels: shares family (share of bootstrap draws on the US side); participants a_pro and b_cm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True)
    pe = {f: path_exp[path_exp.family == f].sort_values("refit_date") for f in FAMILIES}
    ro = {f: roll[roll.family == f].sort_values("window_end") for f in FAMILIES}
    ax = axes[0]
    for d, x, col, lab, kind in ((ro["shares"], "window_end", ORANGE, "rolling 250-day window (end date)", "line"),
                                 (pe["shares"], "refit_date", BLUE, "expanding window, out-of-sample refits", "step")):
        us_side = (d["w1"] > d["w2"]).to_numpy()
        p_us = np.where(us_side, d["share_same_side"], 1 - d["share_same_side"])
        if kind == "step":
            ax.step(d[x], p_us, where="post", color=col, linewidth=2, label=lab)
        else:
            ax.plot(d[x], p_us, color=col, linewidth=2, label=lab)
    _hline(ax, 1.0, "US convention")
    _hline(ax, 0.0, "reverse US")
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("shares family: share of bootstrap draws with w_call > w_put (the US side); only this sign is identified",
                 fontsize=9, color=INK, loc="left")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=1, fontsize=8, frameon=False)
    for i, comp in enumerate((0, 1)):
        ax = axes[1 + i]
        k = f"w{comp + 1}"
        d = ro["participants"]
        ax.fill_between(d["window_end"], d[f"{k}_lo"], d[f"{k}_hi"], color=ORANGE, alpha=0.15, linewidth=0)
        ax.plot(d["window_end"], d[k], color=ORANGE, linewidth=2)
        d = pe["participants"]
        ax.fill_between(d["refit_date"], d[f"{k}_lo"], d[f"{k}_hi"], color=BLUE, alpha=0.18, linewidth=0, step="post")
        ax.step(d["refit_date"], d[k], where="post", color=BLUE, linewidth=2)
        _hline(ax, 1.0 if comp == 0 else 0.0, "Pro rule")
        _hline(ax, 0.0 if comp == 0 else 1.0, "Client-mirror rule", ls=":")
        ax.set_ylim(-1.08, 1.08)
        ax.set_title(f"participants family: {COMPONENT_NAMES['participants'][comp]} (unit direction, 90% bootstrap band)",
                     fontsize=9, color=INK, loc="left")
    for ax in axes:
        _vlines(ax)
        _style(ax)
    fig.suptitle("REAL DATA. Revealed dealer weights, joint fit of (a) variance and (b) momentum under hedging theory.\n"
                 "Grey vertical lines: start of the test period (1 Jan 2025) and the closing auction (3 Aug 2026).",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kalman(kstates: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True)
    for eq, col, lab in (("a", BLUE, "(a) variance equation"), ("b", ORANGE, "(b) momentum equation")):
        s = kstates[(kstates.family == "shares") & (kstates["eq"] == eq)]
        axes[0].plot(s["date"], s["p_us_side"], color=col, linewidth=1.6, label=lab)
        for i, comp in enumerate((0, 1)):
            k = f"w{comp + 1}"
            s = kstates[(kstates.family == "participants") & (kstates["eq"] == eq)]
            axes[1 + i].fill_between(s["date"], s[f"{k}_lo"], s[f"{k}_hi"], color=col, alpha=0.13, linewidth=0)
            axes[1 + i].plot(s["date"], s[k], color=col, linewidth=1.6, label=lab)
    _hline(axes[0], 1.0, "US convention")
    _hline(axes[0], 0.0, "reverse US")
    axes[0].axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("shares family: probability that w_call > w_put (US side), from the filtered loading",
                      fontsize=9, color=INK, loc="left")
    for i, comp in enumerate((0, 1)):
        _hline(axes[1 + i], 1.0 if comp == 0 else 0.0, "Pro rule")
        _hline(axes[1 + i], 0.0 if comp == 0 else 1.0, "Client-mirror rule", ls=":")
        axes[1 + i].set_ylim(-1.08, 1.08)
        axes[1 + i].set_title(f"participants family: {COMPONENT_NAMES['participants'][comp]} (unit direction, 90% band)",
                              fontsize=9, color=INK, loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=1, fontsize=8, frameon=False)
    for ax in axes:
        _vlines(ax)
        _style(ax)
    fig.suptitle("REAL DATA. Kalman-filtered revealed direction (one-sided: uses data up to each day).\n"
                 "Parameters estimated on 2023 to 2024; grey lines: start of the test period, closing auction.",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_profiles(prof: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("training 2023 to 2024", "base"), ("full sample", "base"), ("full sample, R3 scaling (comparison)", "base")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for i, fam in enumerate(FAMILIES):
        for j, (smp, ctl) in enumerate(panels):
            ax = axes[i, j]
            p = prof[(prof.family == fam) & (prof["sample"] == smp) & (prof.controls == ctl)]
            for obj, col, lab, lw, al in (("joint", AQUA, "joint (a) + (b)", 5, 0.45), ("a", BLUE, "(a) variance", 1.8, 1.0),
                                          ("b", ORANGE, "(b) momentum", 1.8, 1.0)):
                q = p[p.objective == obj]
                ax.plot(q["angle_deg"], q["lr"], color=col, linewidth=lw, alpha=al, label=lab, solid_capstyle="butt")
            top = max(float(p["lr"].max()) if len(p) else 1.0, 1.0)
            refs = dict(REFERENCE[fam])
            if fam == "shares" and "R3" in smp:
                refs["gross gamma"] = 45.0
            for name, ang in refs.items():
                ax.axvline(ang, color=MUTED, linewidth=0.8, linestyle="--")
                ax.text(ang + 2, top * 1.0, name, fontsize=6.5, color=INK, rotation=90, va="top")
            ax.set_xlim(0, 359)
            ax.set_xticks(range(0, 360, 45))
            ax.set_title(f"{fam}, {smp}", fontsize=8.5, color=INK, loc="left")
            ax.set_xlabel("direction angle (degrees)", fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel("LR gain, theory sign imposed", fontsize=8, color=MUTED)
            _style(ax)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("REAL DATA. How well each dealer direction fits hedging theory (higher = better; 0 = wrong sign).\n"
                 "Shares: angles 226 to 44 (through 315) mean w_call > w_put.", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


# =====================================================================================
# Main
# =====================================================================================


def _load_momentum_returns(bars: Path, days: pd.DatetimeIndex) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("momentum_gex", Path(__file__).with_name("momentum_gex.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.daily_returns(bars, days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/dealer_sign")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-perm-kalman", type=int, default=20)
    ap.add_argument("--n-synth", type=int, default=200)
    ap.add_argument("--seed", type=int, default=82)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    lines = []

    def say(s=""):
        lines.append(s)
        print(s, flush=True)

    def table(name, df, cols=None):
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 500,
                               "display.precision", 4):
            say(f"== {name}")
            say((df[cols] if cols else df).to_string(index=False))
            say()

    dm = pd.read_csv(Path(a.run) / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(Path(a.run) / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    nets = participant_nets(Path(a.raw) / "participant_oi")
    rets = _load_momentum_returns(Path(a.raw) / "nifty_1min.csv", dm.index)
    frame = build_frame(dm, panel, nets, rets)
    frame.to_csv(out / "frame.csv")
    df = valid(frame)
    train = df[df.index < TEST_START]
    pre_auction = df[df.index < AUCTION_START]
    say("REAL DATA. P82: dealer sign estimated from data (revealed sign under a maintained hypothesis).")
    say(f"Days with all inputs: {len(df)} of {len(frame)} ({df.index.min().date()} to {df.index.max().date()}); "
        f"expiry days {int(df['exp'].sum())}; closing-auction days {int(df['auction'].sum())} (left out of outcome b). "
        f"Training days (before {TEST_START.date()}): {len(train)}; test days: {int((df.index >= TEST_START).sum())} "
        f"(outcome b: {int(((df.index >= TEST_START) & (df['auction'] == 0)).sum())}).")
    gc = frame["s_call"] * panel["gross_gex"].shift(1).reindex(frame.index)
    gp = frame["s_put"] * panel["gross_gex"].shift(1).reindex(frame.index)
    pro_lin = (frame["pro_call"] * gc + frame["pro_put"] * gp) / (gc + gp)
    cm_lin = -(frame["client_call"] * gc + frame["client_put"] * gp) / (gc + gp)
    say(f"Check (GEX is linear in the weights): panel Pro GEX / gross minus (pro_call*s_call + pro_put*s_put), max abs "
        f"{(pro_lin - frame['x_pro']).abs().max():.1e}; Client mirror {(cm_lin - frame['x_cmirror']).abs().max():.1e}. "
        f"Four categories' net shares sum to zero: max abs "
        f"{nets[[c for c in nets if c.endswith('_call')]].sum(axis=1).abs().max():.1e} (calls), "
        f"{nets[[c for c in nets if c.endswith('_put')]].sum(axis=1).abs().max():.1e} (puts).")
    say(f"Call share of gross gamma s_call: mean {df['s_call'].mean():.3f}, sd {df['s_call'].std():.3f}. Scale-free "
        f"GEX / gross, sd over days: US {df['us'].std():.3f}, Pro {df['x_pro'].std():.4f}, Client mirror "
        f"{df['x_cmirror'].std():.4f}; corr(Pro, Client mirror) {df[['x_pro', 'x_cmirror']].corr().iloc[0, 1]:.3f}.")
    by = df.groupby("dtte")["gross_rel"].mean()
    say("Mean gross GEX / 20-day median by trading days to expiry of the outcome day: "
        + ", ".join(f"{int(k)}: {v:.2f}" for k, v in by.items()) + " (why the old R3 scaling is not used).")
    say()

    # ------------------------------------------------------------------ 1. direction estimates
    est_rows, prof_rows = [], []
    configs = [("training 2023 to 2024", train, ("base",), a.n_boot), ("full sample", df, ("base",), a.n_boot),
               ("full sample, no auction days", pre_auction, ("base",), 0),
               ("full sample", df, ("base", "auction_regime"), 0),
               ("training 2023 to 2024", train, ("base", "regime"), 0), ("full sample", df, ("base", "regime"), 0),
               ("training 2023 to 2024", train, ("base", "gross_oi"), 0), ("full sample", df, ("base", "gross_oi"), 0),
               ("training 2023 to 2024, R3 scaling (comparison)", r3_scaled(train), ("base",), 0),
               ("full sample, R3 scaling (comparison)", r3_scaled(df), ("base",), 0)]
    for label, sub, ctrl, nb in configs:
        e, p = direction_table(sub, label, ctrl, rng, nb)
        est_rows.append(e)
        prof_rows.append(p)
    est = pd.concat(est_rows, ignore_index=True)
    prof = pd.concat(prof_rows, ignore_index=True)
    est.to_csv(out / "direction_estimates.csv", index=False)
    prof.to_csv(out / "lr_profiles.csv", index=False)
    table("direction estimates (angle in degrees; 90% profile-LR set; shares family: only the side is identified)", est,
          ["sample", "controls", "family", "objective", "n_days_a", "n_days_b", "angle_deg", "lr_max", "lr_set_90",
           "lr_set_width_deg"])
    table("block-bootstrap bands (base controls); share_same_side = share of draws within 90 degrees of the estimate",
          est[est["boot_angle_lo"].notna()] if "boot_angle_lo" in est else est,
          ["sample", "family", "objective", "angle_deg", "boot_angle_lo", "boot_angle_hi", "boot_w1_lo", "boot_w1_hi",
           "boot_w2_lo", "boot_w2_hi", "boot_share_same_side"])
    table("unrestricted loadings theta (direction = -theta; shares: theta on US-convention GEX), HAC Wald test of theta = 0",
          est[est.objective != "joint"], ["sample", "controls", "family", "objective", "theta1", "se1", "theta2", "se2",
                                          "wald_df", "wald_p"])
    incols = ["sample", "controls", "family", "objective"] + [c for c in est.columns if c.startswith("in_set")]
    table("which reference directions are inside the 90% LR set", est, incols)
    frf = pd.concat([fixed_rule_fits(train, "training 2023 to 2024"), fixed_rule_fits(df, "full sample"),
                     fixed_rule_fits(df, "full sample", ("base", "regime"))], ignore_index=True)
    frf.to_csv(out / "fixed_rule_fits.csv", index=False)
    table("fixed rules in the forecasting template (in-sample, HAC; theory says coef < 0)", frf,
          ["sample", "controls", "eq", "rule", "coef", "se_hac", "p", "theory_sign_ok", "signal_to_noise",
           "other_gex_term", "other_gex_coef", "other_gex_p", "r2", "r2_baseline", "n"])
    say(f"[{time.time() - t0:.0f}s] direction estimates done")

    # ------------------------------------------------------------------ 2. drift: rolling window and Kalman
    roll = rolling_directions(df, ROLL_WINDOW, REFIT_EVERY, a.n_boot, rng)
    roll.to_csv(out / "weights_path_rolling.csv", index=False)
    kp, ks, kal_pred = [], [], {}
    for fam in FAMILIES:
        for eq in ("a", "b"):
            r = fit_tvp(df, eq, fam)
            kp.append({"family": fam, "eq": eq, "n_loadings": r["theta_f"].shape[1], "sigma2": r["params"]["sigma2"],
                       "q": r["params"]["q"], "llf_train": r["llf_train"], "llf_train_q0": r["llf_q0"],
                       "lr_q_vs_0": r["lr_q"], "p_q_eq_0": r["p_q0"]})
            dirf = tvp_direction(r["theta_f"], r["P_f"], rng)
            dirs = tvp_direction(r["theta_s"], r["P_s"], rng)
            k = r["theta_f"].shape[1]
            s = pd.DataFrame({"date": r["index"], "family": fam, "eq": eq, "theta1": r["theta_f"][:, 0],
                              "se1": np.sqrt(r["P_f"][:, 0, 0]),
                              "theta2": r["theta_f"][:, 1] if k == 2 else np.nan,
                              "se2": np.sqrt(r["P_f"][:, 1, 1]) if k == 2 else np.nan,
                              "theta1_smoothed": r["theta_s"][:, 0],
                              "theta2_smoothed": r["theta_s"][:, 1] if k == 2 else np.nan})
            s = pd.concat([s, dirf.set_index(s.index), dirs.add_suffix("_smoothed").set_index(s.index)], axis=1)
            ks.append(s)
            resid_tr = (r["y"] - r["forecast"])[r["train"]][60:]
            full_pred = np.full(len(df), np.nan)
            full_pred[r["rows"]] = r["forecast"]
            smear = float(np.mean(np.exp(resid_tr))) if eq == "a" else np.nan
            kal_pred[(eq, "free", f"kalman_{fam}")] = {"pred": full_pred, "smear": np.full(len(df), smear)}
    kp, ks = pd.DataFrame(kp), pd.concat(ks, ignore_index=True)
    kp.to_csv(out / "kalman_params.csv", index=False)
    ks.to_csv(out / "kalman_states.csv", index=False)
    table("Kalman time-varying loadings: MLE on 2023 to 2024 (q = per-day drift variance of the scaled loadings)", kp)
    ksum = []
    for (fam, eq), g in ks.groupby(["family", "eq"]):
        k = 1 if fam == "shares" else 2
        for per, m in (("training", g["date"] < TEST_START), ("test", g["date"] >= TEST_START)):
            h = g[m & g["wald"].notna()]
            row = {"family": fam, "eq": eq, "period": per, "days": len(h),
                   "median_wald_theta0": float(h["wald"].median()),
                   "share_days_wald_gt_90pct": float((h["wald"] > chi2.ppf(0.9, k)).mean()),
                   "w1_first": float(h["w1"].iloc[0]), "w1_last": float(h["w1"].iloc[-1]),
                   "w2_first": float(h["w2"].iloc[0]), "w2_last": float(h["w2"].iloc[-1])}
            if fam == "shares":
                row.update({"p_us_side_first": float(h["p_us_side"].iloc[0]), "p_us_side_last": float(h["p_us_side"].iloc[-1]),
                            "p_us_side_mean": float(h["p_us_side"].mean())})
            ksum.append(row)
    ksum = pd.DataFrame(ksum)
    ksum.to_csv(out / "kalman_summary.csv", index=False)
    table("Kalman filtered direction: strength (Wald of theta_t = 0) and drift", ksum)
    say(f"[{time.time() - t0:.0f}s] rolling and Kalman done")

    # ------------------------------------------------------------------ 3. out of sample
    samples = {"all test days": lambda d: np.ones(len(d), bool),
               "test days before the closing auction": lambda d: (d < AUCTION_START).to_numpy()}
    run = oos_run(df, ("base",), n_boot=a.n_boot, rng=rng)
    run["path"].to_csv(out / "weights_path_expanding.csv", index=False)
    L = losses_from_run(df, run, kal_pred)
    L.to_csv(out / "oos_losses.csv", index=False)
    summ, dmt = summarize_losses(L, samples)
    prim = (dmt["sample"] == "all test days") & (dmt["variant"] == "signed") & dmt["estimated"].isin(
        ["est_shares", "est_participants"])
    dmt["primary"] = prim
    dmt["p_est_better_holm_primary"] = np.nan
    dmt.loc[prim, "p_est_better_holm_primary"] = holm(dmt.loc[prim, "p_est_better"].to_numpy())
    dmt["p_two_sided_holm_primary"] = np.nan
    dmt.loc[prim, "p_two_sided_holm_primary"] = holm(dmt.loc[prim, "p_two_sided"].to_numpy())
    summ.to_csv(out / "oos_summary.csv", index=False)
    dmt.to_csv(out / "oos_dm.csv", index=False)
    table("expanding-window direction at each out-of-sample refit (joint objective)", run["path"],
          ["refit_date", "family", "angle_joint", "w1", "w2", "angle_lo", "angle_hi", "share_same_side", "angle_a",
           "angle_b", "lr_a", "lr_b"])
    table("out-of-sample mean loss (a: QLIKE of realized variance; b: squared error, bp^2), % vs baseline without dealer sign",
          summ, ["sample", "eq", "variant", "model", "n_days", "mean_loss", "pct_vs_none", "p_better_than_none"])
    table("Diebold-Mariano: estimated-sign GEX vs fixed rules, theory-signed template (primary: all test days; Holm over "
          "the tests that exist)", dmt[dmt["variant"] == "signed"],
          ["sample", "eq", "estimated", "fixed_rule", "n_days", "identical_forecasts", "pct_est_vs_rule", "dm_stat",
           "p_est_better", "p_two_sided", "p_est_better_holm_primary", "p_two_sided_holm_primary"])
    table("Diebold-Mariano, free-coefficient template (sign-free: a rule and its reverse forecast identically)",
          dmt[(dmt["variant"] == "free") & (dmt["sample"] == "all test days")],
          ["eq", "estimated", "fixed_rule", "identical_forecasts", "pct_est_vs_rule", "p_est_better", "p_two_sided"])
    coef_rows = []
    for (eq, var, name), v in run["pred"].items():
        if var == "signed" and name != "none":
            cs = np.asarray(v["coef"], float)
            coef_rows.append({"eq": eq, "model": name, "refits": len(cs), "share_refits_theory_sign": float((cs < 0).mean()),
                              "share_refits_set_to_zero": float((cs == 0).mean())})
    coef_rows = pd.DataFrame(coef_rows)
    coef_rows.to_csv(out / "oos_signed_coef_share.csv", index=False)
    table("theory-signed template: share of refits in which the GEX term had the theory sign (else it was set to 0)",
          coef_rows)
    agr = sign_agreement(df, run)
    agr.to_csv(out / "sign_agreement.csv", index=False)
    table("sign agreement between estimated-sign GEX and the fixed rules", agr[agr.quarter == "all"],
          ["family", "fixed_rule", "period", "n_days", "share_same_sign", "share_est_negative", "share_rule_negative", "corr"])
    say(f"[{time.time() - t0:.0f}s] out-of-sample done")

    # ------------------------------------------------------------------ 4. robustness
    rob_rows = []
    for name, sub, ctrl in (("auction days left out of both outcomes", pre_auction, ("base",)),
                            ("auction days kept in outcome b as their own regime", df, ("base", "auction_regime")),
                            ("expiry-regime and days-to-expiry controls", df, ("base", "regime")),
                            ("open-interest and gross-level controls", df, ("base", "gross_oi")),
                            ("old R3 scaling (net / 20-day median gross)", r3_scaled(df), ("base",))):
        rr = oos_run(sub, ctrl, variants=("signed",))
        s2, d2 = summarize_losses(losses_from_run(sub, rr), {"all test days": samples["all test days"]})
        for _, r in s2[s2.model.isin(["est_shares", "est_participants", "us", "pro", "cmirror"])].iterrows():
            rob_rows.append({"robustness": name, "comparison": f"{r['model']} vs none", "eq": r["eq"],
                             "n_days": r["n_days"], "pct": r["pct_vs_none"], "p_est_better": r["p_better_than_none"]})
        for _, r in d2[d2.estimated.isin(["est_shares", "est_participants"])].iterrows():
            rob_rows.append({"robustness": name, "comparison": f"{r['estimated']} vs {r['fixed_rule']}", "eq": r["eq"],
                             "n_days": r["n_days"], "pct": r["pct_est_vs_rule"], "p_est_better": r["p_est_better"],
                             "p_two_sided": r["p_two_sided"], "identical_forecasts": r["identical_forecasts"]})
        for _, r in rr["path"].groupby("family").tail(1).iterrows():
            rob_rows.append({"robustness": name, "comparison": f"last refit direction, {r['family']}", "eq": "joint",
                             "n_days": r["train_days"], "angle_deg": r["angle_joint"], "angle_a": r["angle_a"],
                             "angle_b": r["angle_b"]})
    rob = pd.DataFrame(rob_rows)
    rob.to_csv(out / "robustness_oos.csv", index=False)
    table("robustness, out of sample (theory-signed template); pct = % loss difference, negative = first model better", rob)
    say(f"[{time.time() - t0:.0f}s] robustness done")

    # ------------------------------------------------------------------ 5. placebo: shuffled outcomes
    real = {}
    for fam in FAMILIES:
        r = est[(est["sample"] == "full sample") & (est.controls == "base") & (est.family == fam)].set_index("objective")
        for obj in ("joint", "a", "b"):
            real[f"lr_{obj}_{fam}"] = r.loc[obj, "lr_max"]
    for eq in ("a", "b"):
        s = summ[(summ["sample"] == "all test days") & (summ["eq"] == eq) & (summ.variant == "signed")].set_index("model")
        for fam in FAMILIES:
            real[f"oos_pct_{eq}_est_{fam}_vs_none"] = s.loc[f"est_{fam}", "pct_vs_none"]
    pl = []
    for k in range(a.n_perm):
        dp = permute_outcomes(df, rng)
        row = {"perm": k}
        for fam in FAMILIES:
            e = estimate_direction([gram_stats(*design_arrays(dp, eq, fam)) for eq in ("a", "b")])
            row.update({f"lr_joint_{fam}": e["lr_max"], f"angle_joint_{fam}": e["angle"],
                        f"set_width_{fam}": int(e["set"].sum())})
            for j, eq in enumerate(("a", "b")):
                row[f"lr_{eq}_{fam}"] = float(e["parts"][j].max())
                str_ = gram_stats(*design_arrays(dp, eq, fam, reduced=True))
                th = np.linalg.solve(str_["A"], str_["b"])
                row[f"theta_norm_{eq}_{fam}"] = float(np.linalg.norm(th))
        rp = oos_run(dp, ("base",), variants=("signed",))
        sp, _ = summarize_losses(losses_from_run(dp, rp), {"all": samples["all test days"]})
        sp = sp.set_index(["eq", "model"])
        for eq in ("a", "b"):
            for fam in FAMILIES:
                row[f"oos_pct_{eq}_est_{fam}_vs_none"] = sp.loc[(eq, f"est_{fam}"), "pct_vs_none"]
        pl.append(row)
    pl = pd.DataFrame(pl)
    pl.to_csv(out / "placebo_draws.csv", index=False)
    psum = []
    for key, val in real.items():
        x = pl[key].to_numpy(float)
        more = x >= val if key.startswith("lr_") else x <= val  # OOS: more negative = bigger gain
        psum.append({"statistic": key, "real": val, "placebo_median": float(np.median(x)),
                     "placebo_q05": float(np.percentile(x, 5)), "placebo_q95": float(np.percentile(x, 95)),
                     "placebo_p": float((1 + more.sum()) / (1 + len(x))), "n_perm": len(x)})
    for fam in FAMILIES:
        for eq in ("a", "b"):
            sub = est[(est["sample"] == "full sample") & (est.controls == "base") & (est.family == fam)
                      & (est.objective == eq)].iloc[0]
            val = float(np.hypot(sub["theta1"], 0.0 if np.isnan(sub["theta2"]) else sub["theta2"]))
            x = pl[f"theta_norm_{eq}_{fam}"].to_numpy(float)
            psum.append({"statistic": f"theta_norm_{eq}_{fam}", "real": val, "placebo_median": float(np.median(x)),
                         "placebo_q05": float(np.percentile(x, 5)), "placebo_q95": float(np.percentile(x, 95)),
                         "placebo_p": float((1 + (x >= val).sum()) / (1 + len(x))), "n_perm": len(x)})
        w = pl[f"set_width_{fam}"]
        psum.append({"statistic": f"lr_set_width_deg_{fam}", "real": float(est[(est["sample"] == "full sample") & (
            est.controls == "base") & (est.family == fam) & (est.objective == "joint")]["lr_set_width_deg"].iloc[0]),
            "placebo_median": float(w.median()), "placebo_q05": float(w.quantile(0.05)),
            "placebo_q95": float(w.quantile(0.95)), "placebo_p": np.nan, "n_perm": len(w)})
    kpl = []
    for k in range(a.n_perm_kalman):
        dp = permute_outcomes(df, rng)
        for fam in FAMILIES:
            for eq in ("a", "b"):
                r = fit_tvp(dp, eq, fam)
                kk = r["theta_f"].shape[1]
                d = tvp_direction(r["theta_f"], r["P_f"], rng, n_draw=10)
                tm = ~r["train"]
                kpl.append({"perm": k, "family": fam, "eq": eq, "q": r["params"]["q"], "p_q_eq_0": r["p_q0"],
                            "median_wald_test": float(np.nanmedian(d["wald"].to_numpy()[tm])),
                            "share_days_wald_gt_90pct": float((d["wald"].to_numpy()[tm] > chi2.ppf(0.9, kk)).mean())})
    kpl = pd.DataFrame(kpl)
    kpl.to_csv(out / "placebo_kalman.csv", index=False)
    for (fam, eq), g in (kpl.groupby(["family", "eq"]) if len(kpl) else []):
        rr_ = ksum[(ksum.family == fam) & (ksum["eq"] == eq) & (ksum.period == "test")].iloc[0]
        x = g["share_days_wald_gt_90pct"]
        psum.append({"statistic": f"kalman_share_test_days_wald_gt_90pct_{eq}_{fam}", "real": rr_["share_days_wald_gt_90pct"],
                     "placebo_median": float(x.median()), "placebo_q05": float(x.quantile(0.05)),
                     "placebo_q95": float(x.quantile(0.95)),
                     "placebo_p": float((1 + (x >= rr_["share_days_wald_gt_90pct"]).sum()) / (1 + len(x))), "n_perm": len(x)})
        real_p = float(kp[(kp.family == fam) & (kp["eq"] == eq)]["p_q_eq_0"].iloc[0])
        psum.append({"statistic": f"kalman_drift_test_p_{eq}_{fam} (LR test of q = 0; placebo_p = share of placebo "
                                  f"fits with a p-value at least as small; share of placebo fits with p < 0.05: "
                                  f"{(g['p_q_eq_0'] < 0.05).mean():.2f})",
                     "real": real_p, "placebo_median": float(g["p_q_eq_0"].median()),
                     "placebo_q05": float(g["p_q_eq_0"].quantile(0.05)), "placebo_q95": float(g["p_q_eq_0"].quantile(0.95)),
                     "placebo_p": float((1 + (g["p_q_eq_0"] <= real_p).sum()) / (1 + len(g))), "n_perm": len(g)})
    psum = pd.DataFrame(psum)
    psum.to_csv(out / "placebo_summary.csv", index=False)
    table("placebo: outcome days shuffled against the GEX series (placebo_p = share of draws at least as extreme as real)",
          psum)
    say(f"[{time.time() - t0:.0f}s] placebo done")

    # ------------------------------------------------------------------ 6. semi-synthetic validation
    syn = pd.concat([semi_synthetic(df, fam, (0.0, 0.05, 0.10, 0.20), a.n_synth, rng) for fam in FAMILIES],
                    ignore_index=True)
    syn.to_csv(out / "synthetic_validation_draws.csv", index=False)
    ss = []
    for fam, g in syn.groupby("family"):
        crit = float(g[g.snr == 0]["lr_max"].quantile(0.95))
        for snr, h in g.groupby("snr"):
            ss.append({"family": fam, "identified": h["identified"].iloc[0], "snr": snr, "reps": len(h),
                       "detect_rate_5pct": float((h["lr_max"] > crit).mean()),
                       "share_correct_side": float(h["correct_side"].mean()),
                       "median_abs_error_deg": float(h["abs_error_deg"].median()),
                       "share_error_lt_30deg": float((h["abs_error_deg"] < 30).mean()) if h["abs_error_deg"].notna().any() else np.nan,
                       "coverage_90_lr_set": float(h["covered_90"].mean()),
                       "median_set_width_deg": float(h["set_width_deg"].median()),
                       "median_abs_error_a_only": float(h["abs_error_a"].median()),
                       "median_abs_error_b_only": float(h["abs_error_b"].median()), "null_lr_crit_95": crit})
    ss = pd.DataFrame(ss)
    ss.to_csv(out / "synthetic_validation.csv", index=False)
    table("SYNTHETIC outcomes on the real regressors with known weights (true angle uniform; snr = signal sd / noise sd)", ss)
    nulls = []
    for k in range(5):  # under no effect the correct-side and coverage rates should be about 0.5 and 0.9: check other seeds
        for fam in FAMILIES:
            h = semi_synthetic(df, fam, (0.0,), a.n_synth, np.random.default_rng(1000 + k))
            nulls.append({"family": fam, "seed": 1000 + k, "reps": len(h), "share_correct_side": float(h["correct_side"].mean()),
                          "coverage_90_lr_set": float(h["covered_90"].mean()),
                          "binomial_se_of_share": float(np.sqrt(0.25 / len(h)))})
    nulls = pd.DataFrame(nulls)
    nulls.to_csv(out / "synthetic_validation_null_seeds.csv", index=False)
    table("SYNTHETIC, no effect, other seeds (share_correct_side should be near 0.5)", nulls)
    real_snr = frf[(frf["sample"] == "full sample") & (frf.controls == "base")][["eq", "rule", "signal_to_noise"]]
    table("for scale: in-sample signal-to-noise of the fixed rules on REAL DATA (full sample)", real_snr)

    # ------------------------------------------------------------------ figures and info
    plot_weights(run["path"], roll, out / "figures" / "dealer_weights_over_time.png")
    plot_kalman(ks, out / "figures" / "dealer_weights_kalman.png")
    plot_profiles(prof, out / "figures" / "lr_profiles.png")
    info = {"source": "REAL DATA: results/pipeline_nse (NSE bhavcopy chains), NSE participant-wise OI, validated 1-minute bars",
            "seconds": round(time.time() - t0, 1), "args": vars(a), "test_start": str(TEST_START.date()),
            "auction_start": str(AUCTION_START.date()), "refit_every": REFIT_EVERY, "rolling_window": ROLL_WINDOW,
            "bootstrap_block": BLOCK, "angles": "0 to 359 degrees, step 1",
            "gex_measure": "scale-free: net / gross GEX of the same chain; log(gross / 20-day median) as a control",
            "outcome_b_rows": "closing-auction days (from 2026-08-03) left out"}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    say(f"[{time.time() - t0:.0f}s] finished. Files in {out}/")
    (out.parent / f"{out.name}_stdout.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
