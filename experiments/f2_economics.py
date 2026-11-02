"""P90: how big is finding F2 in economic terms, after realistic costs? REAL DATA.

Pre-registered in docs/prereg/P90_F2_ECONOMICS.md, committed as
236e3dba89ea8abd6bb6ec275e3e4210460c6dc9 (2 Nov 2026, 22:36 UTC) before this script was written or run.
This is a research evaluation of the economic size of a statistical finding, not trading advice.

F2 (docs/REAL_DATA_RESULTS.md R5): Nifty's last-30-minute return (14:59 to 15:29 bar closes) tends to reverse
the rest-of-day return (previous 15:29 bar close to 14:59), only after evenings when GEX under NSE's Pro
participant rule was negative. This script turns that into trading rules and charges realistic costs:

    P   (primary)  if Pro GEX at the previous close < 0: w = -sign(r_rod) at the 14:59 bar close,
                   exit at the 15:29 bar close; flat otherwise. No parameters.
    S1  (sized)    w = clip(-r_rod / s, -3, 3) on the same days, s = training median |r_rod| on negative-GEX days.
    S2  (filter)   trade like P only when the R5 regression fitted on the training window forecasts a gain
                   larger than the round-trip cost on that date.
    D   (delay)    P entered one minute later, at the 15:00 bar close (sensitivity, not a decision input).

Walk-forward: training 2 Jan 2023 to 31 Dec 2024; test from the first trading day of 2025 to 31 Jul 2026 in
blocks of 21 trading days; S1 and S2 are refitted before each block on all earlier days (expanding window).
Closing-auction days (from 3 Aug 2026) are a separate descriptive sample. Index bars stand in for near-month
futures. Decision (pre-stated): F2 is "economically meaningful" if the Deflated Sharpe Ratio of P (test period,
base-case costs, daily series, N = 824 trials) is above 0.95.

    OMP_NUM_THREADS=1 python experiments/f2_economics.py
    OMP_NUM_THREADS=1 python -m pytest -q tests/test_f2_economics.py

Writes results/f2_economics/*.csv, two figures, run_info.json and results/f2_economics_stdout.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import statsmodels  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import momentum_gex as mg  # noqa: E402

PREREG_FILE = "docs/prereg/P90_F2_ECONOMICS.md"
PREREG_COMMIT = "236e3dba89ea8abd6bb6ec275e3e4210460c6dc9"

TRAIN_START = pd.Timestamp("2023-01-02")
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")  # first trading day of 2025 is the first row on or after this date
TEST_END = pd.Timestamp("2026-07-31")
AUCTION_START = pd.Timestamp("2026-08-03")
BLOCK = 21
LAGS = 5
ANNUAL = 252
CAP = 3.0
SLOT_1500 = 345  # minutes after 09:15

# ---- costs, bp of notional (sources in the pre-registration, section 4)
SLIP_BASE = 1.0  # bp per side
SLIP_GRID = (0.0, 0.5, 1.0, 2.0, 3.0)
STT_SCHEDULE = ((pd.Timestamp("1900-01-01"), 1.00), (pd.Timestamp("2023-04-01"), 1.25),
                (pd.Timestamp("2024-10-01"), 2.00), (pd.Timestamp("2026-04-01"), 5.00))  # sell leg only
EXCH_SCHEDULE = ((pd.Timestamp("1900-01-01"), 0.200), (pd.Timestamp("2024-10-01"), 0.183))  # per side, incl. IPFT
SEBI_BP_SIDE = 0.01
STAMP_BP = 0.20  # buy leg only
GST_RATE = 0.18
BROKERAGE_RS_PER_ORDER = 20.0
NOTIONAL_RS = 1.5e6

# ---- inference
N_BOOT = 10_000
BOOT_BLOCK = 5
MASTER_SEED = 20261102
STUDY_SEED = 90
RULES = ("P", "S1", "S2", "D")
RULE_IDS = {"P": 0, "S1": 1, "S2": 2, "D": 3}
RULE_NAMES = {"P": "P: primary (against the day)", "S1": "S1: sized by the move",
              "S2": "S2: trade only if forecast beats cost", "D": "D: P entered at 15:00"}
N_FAMILY_A, N_FAMILY_B, N_THIS, N_PLACEBO_RULES = 78, 742, 4, 2800
N_PRIMARY = N_FAMILY_A + N_FAMILY_B + N_THIS  # 824
N_SENS = (N_FAMILY_A + N_THIS, N_PRIMARY, N_PRIMARY + N_PLACEBO_RULES)  # 82, 824, 3624
DSR_THRESHOLD = 0.95
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def _rate(schedule, date) -> float:
    date = pd.Timestamp(date)
    val = schedule[0][1]
    for start, r in schedule:
        if date >= start:
            val = r
    return val


def cost_components(date, slip_bp_side: float = SLIP_BASE) -> dict:
    """Round-trip cost in bp of notional for one unit traded on ``date`` (one buy and one sell order)."""
    stt = _rate(STT_SCHEDULE, date)
    exch = 2.0 * _rate(EXCH_SCHEDULE, date)
    sebi = 2.0 * SEBI_BP_SIDE
    brok = 2.0 * BROKERAGE_RS_PER_ORDER / NOTIONAL_RS * 1e4
    gst = GST_RATE * (brok + exch + sebi)
    statutory = stt + exch + sebi + STAMP_BP + brok + gst
    slip = 2.0 * slip_bp_side
    return {"stt": stt, "exchange": exch, "sebi": sebi, "stamp": STAMP_BP, "brokerage": brok, "gst": gst,
            "statutory": statutory, "slippage": slip, "total": statutory + slip}


def cost_frame(dates, slip_bp_side: float | None = SLIP_BASE) -> pd.DataFrame:
    """Per-date cost components. ``slip_bp_side=None`` means gross (every component zero)."""
    dates = pd.DatetimeIndex(dates)
    if slip_bp_side is None:
        cols = ["stt", "exchange", "sebi", "stamp", "brokerage", "gst", "statutory", "slippage", "total"]
        return pd.DataFrame(0.0, index=dates, columns=cols)
    return pd.DataFrame([cost_components(d, slip_bp_side) for d in dates], index=dates)


def cost_schedule_table() -> pd.DataFrame:
    starts = sorted({s for s, _ in STT_SCHEDULE + EXCH_SCHEDULE if s.year > 2000} | {TRAIN_START})
    rows = []
    for i, s in enumerate(starts):
        end = starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(starts) else pd.NaT
        c = cost_components(s, SLIP_BASE)
        rows.append({"from": s.date(), "to": end.date() if pd.notna(end) else "open", **{k: round(v, 4) for k, v in c.items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics (tested in tests/test_f2_economics.py)
# ---------------------------------------------------------------------------


def nw_lrv(x, lags: int = LAGS) -> float:
    """Newey-West long-run variance of x (Bartlett kernel), as statsmodels HAC without small-sample correction."""
    u = np.asarray(x, float) - np.mean(x)
    T = len(u)
    s = u @ u
    for lag in range(1, min(lags, T - 1) + 1):
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * (u[lag:] @ u[:-lag])
    return s / T


def hac_t(x, lags: int = LAGS) -> tuple[float, float]:
    """HAC t-statistic of the mean and its one-sided p-value for mean > 0."""
    x = np.asarray(x, float)
    lrv = nw_lrv(x, lags)
    if lrv <= 0:
        return np.nan, np.nan
    t = x.mean() / math.sqrt(lrv / len(x))
    return t, float(stats.norm.sf(t))


def sharpe(x) -> float:
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return x.mean() / sd if sd > 0 else np.nan


def moments(x) -> tuple[float, float]:
    """Skewness and kurtosis (normal = 3, not excess) as moment estimators."""
    u = np.asarray(x, float) - np.mean(x)
    m2 = np.mean(u ** 2)
    if m2 <= 0:
        return np.nan, np.nan
    return float(np.mean(u ** 3) / m2 ** 1.5), float(np.mean(u ** 4) / m2 ** 2)


def psr(sr: float, T: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey and Lopez de Prado 2012); sr per period, kurt not excess."""
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(T - 1) / math.sqrt(den)))


def expected_max_multiplier(N: int) -> float:
    """(1 - gamma) InvPhi(1 - 1/N) + gamma InvPhi(1 - 1/(N e)) of Bailey and Lopez de Prado (2014)."""
    return float((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / N) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (N * math.e)))


def sr0(V: float, N: int) -> float:
    return math.sqrt(V) * expected_max_multiplier(N)


def dsr(sr: float, T: int, skew: float, kurt: float, V: float, N: int) -> float:
    """Deflated Sharpe Ratio = PSR evaluated at SR_0."""
    return psr(sr, T, skew, kurt, sr0(V, N))


def max_drawdown(x, dates=None) -> dict:
    """Largest fall of the cumulative sum of x from an earlier peak (cumulative sum starts at 0)."""
    x = np.asarray(x, float)
    cum = np.concatenate([[0.0], np.cumsum(x)])
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    i = int(np.argmax(dd))
    j = int(np.argmax(cum[: i + 1]))
    out = {"mdd": float(dd[i]), "mdd_days": i - j}
    if dates is not None:
        dates = pd.DatetimeIndex(dates)
        out["mdd_peak"] = dates[j - 1].date() if j > 0 else "start"
        out["mdd_trough"] = dates[i - 1].date() if i > 0 else "start"
    return out


def stationary_bootstrap_indices(T: int, B: int, mean_block: float, rng) -> np.ndarray:
    """Politis and Romano (1994): blocks of geometric length (mean ``mean_block``) with wrap-around."""
    p = 1.0 / mean_block
    idx = np.empty((B, T), dtype=np.int32)
    idx[:, 0] = rng.integers(0, T, B)
    for t in range(1, T):
        new = rng.random(B) < p
        idx[:, t] = np.where(new, rng.integers(0, T, B), (idx[:, t - 1] + 1) % T)
    return idx


def bootstrap_sharpe(x, rng, B: int = N_BOOT, mean_block: float = BOOT_BLOCK) -> np.ndarray:
    x = np.asarray(x, float)
    idx = stationary_bootstrap_indices(len(x), B, mean_block, rng)
    xs = x[idx]
    sd = xs.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, xs.mean(axis=1) / sd, np.nan) * math.sqrt(ANNUAL)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def bar_close_at(bars_path: Path, days: pd.DatetimeIndex, slot: int) -> pd.Series:
    """Close of the 1-minute bar ``slot`` minutes after 09:15, built exactly like momentum_gex.daily_returns."""
    b = pd.read_csv(bars_path, parse_dates=["timestamp"])
    b = b[(b["timestamp"] >= days.min() - pd.Timedelta(days=7)) & (b["timestamp"] < days.max() + pd.Timedelta(days=1))]
    b["day"] = b["timestamp"].dt.normalize()
    b["slot"] = ((b["timestamp"] - b["day"]) / pd.Timedelta(minutes=1) - 555).astype(int)
    b = b[b["slot"] == slot]
    return b.groupby("day")["close"].last().reindex(days)


def load_frame(run: Path, raw: Path) -> pd.DataFrame:
    """R5's day frame (same construction as momentum_gex.main) plus the 15:00 bar and trade returns."""
    dm = pd.read_csv(run / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(run / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    d = dm.join(mg.daily_returns(raw / "nifty_1min.csv", dm.index))
    scale = panel["gross_gex"].rolling(20, min_periods=5).median()
    d["gex_pro"] = (panel[mg.RULES["participant_pro"]] / scale).shift(1).reindex(d.index)  # known at previous close
    d["p_1500"] = bar_close_at(raw / "nifty_1min.csv", dm.index, SLOT_1500)
    d = d.dropna(subset=["r_rod", "r_last"]).copy()
    d["r_rod_bp"], d["r_last_bp"] = d["r_rod"] * 1e4, d["r_last"] * 1e4
    d["neg"] = (d["gex_pro"] < 0).astype(float).where(d["gex_pro"].notna())
    d["R_bp"] = (d["p_1529"] / d["p_1459"] - 1.0) * 1e4
    d["R_D_bp"] = (d["p_1529"] / d["p_1500"] - 1.0) * 1e4
    return d


def replicate_r5(d: pd.DataFrame, regressions_csv: Path) -> dict:
    """Re-estimate R5's Pro binary interaction (all days) and compare with the file on disk."""
    X = pd.DataFrame({"r_rod": d["r_rod_bp"], "neg": d["neg"], "r_rod_x_neg": d["r_rod_bp"] * d["neg"]})
    f = mg.fit(d["r_last_bp"], X)
    ref = pd.read_csv(regressions_csv)
    ref = ref[(ref["spec"] == "binary") & (ref["rule"] == "participant_pro") & (ref["subset"] == "all")
              & (ref["term"] == "r_rod_x_neg")].iloc[0]
    out = {"coef": float(f.params["r_rod_x_neg"]), "p": float(f.pvalues["r_rod_x_neg"]), "n": int(f.nobs),
           "coef_on_disk": float(ref["coef"]), "p_on_disk": float(ref["p"]), "n_on_disk": int(ref["n"])}
    out["abs_diff_coef"] = abs(out["coef"] - out["coef_on_disk"])
    out["ok"] = bool(out["abs_diff_coef"] <= 1e-8 and out["n"] == out["n_on_disk"])
    return out


# ---------------------------------------------------------------------------
# Walk-forward and rules
# ---------------------------------------------------------------------------


def fit_params(train: pd.DataFrame) -> dict:
    """S1 scale and the S2 regression (R5 binary specification, OLS point estimates) on a training window."""
    tr = train[train["neg"].notna()]
    s = float(tr.loc[tr["neg"] == 1, "r_rod_bp"].abs().median())
    X = sm.add_constant(pd.DataFrame({"r_rod": tr["r_rod_bp"], "neg": tr["neg"], "r_rod_x_neg": tr["r_rod_bp"] * tr["neg"]}))
    b = sm.OLS(tr["r_last_bp"], X).fit().params
    return {"s": s, "a": float(b["const"]), "b": float(b["r_rod"]), "c": float(b["r_rod_x_neg"]), "d": float(b["neg"]),
            "n_train": int(len(tr)), "n_train_neg": int((tr["neg"] == 1).sum())}


def walk_forward(d: pd.DataFrame, test_index: pd.DatetimeIndex, block: int = BLOCK, start=TRAIN_START) -> pd.DataFrame:
    """Expanding-window refits before each block of ``block`` test days; training uses only days before the block."""
    rows = []
    for k in range(0, len(test_index), block):
        days = test_index[k:k + block]
        train = d[(d.index >= start) & (d.index < days[0])]
        rows.append({"block": k // block + 1, "block_start": days[0], "block_end": days[-1], "block_days": len(days),
                     "train_start": train.index.min(), "train_end": train.index.max(), **fit_params(train)})
    return pd.DataFrame(rows)


def params_by_day(wf: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index, columns=["s", "a", "b", "c", "d"], dtype=float)
    for _, r in wf.iterrows():
        m = (index >= r["block_start"]) & (index <= r["block_end"])
        out.loc[m, ["s", "a", "b", "c", "d"]] = r[["s", "a", "b", "c", "d"]].to_numpy(float)
    return out


def positions(rule: str, d: pd.DataFrame, par: pd.DataFrame | None, cost_total: pd.Series) -> pd.Series:
    """Position in units of notional, using only information available at the entry time of day t."""
    neg = (d["neg"] == 1).to_numpy()
    sgn = np.sign(d["r_rod_bp"].to_numpy())
    w = np.zeros(len(d))
    if rule in ("P", "D"):
        w = np.where(neg, -sgn, 0.0)
        if rule == "D":
            w = np.where(d["p_1500"].notna().to_numpy(), w, 0.0)
    elif rule == "S1":
        w = np.where(neg, np.clip(-d["r_rod_bp"].to_numpy() / par["s"].to_numpy(float), -CAP, CAP), 0.0)
    elif rule == "S2":
        p = par
        g = -sgn * (p["a"].to_numpy(float) + p["d"].to_numpy(float)
                    + (p["b"].to_numpy(float) + p["c"].to_numpy(float)) * d["r_rod_bp"].to_numpy())
        w = np.where(neg & (g > cost_total.to_numpy(float)), -sgn, 0.0)
    else:
        raise ValueError(rule)
    return pd.Series(w + 0.0, index=d.index)


def pnl(rule: str, d: pd.DataFrame, w: pd.Series, costs: pd.DataFrame) -> pd.DataFrame:
    R = d["R_D_bp"] if rule == "D" else d["R_bp"]
    R = R.fillna(0.0)
    aw = w.abs()
    out = pd.DataFrame({"w": w, "R_bp": R.where(w != 0, np.nan), "gross": w * R})
    for k in ("stt", "exchange", "sebi", "stamp", "brokerage", "gst", "statutory", "slippage", "total"):
        out[f"cost_{k}"] = aw * costs[k]
    out["net"] = out["gross"] - out["cost_total"]
    return out


def run_rule(rule: str, d: pd.DataFrame, par: pd.DataFrame | None, slip: float | None) -> pd.DataFrame:
    costs = cost_frame(d.index, slip)
    w = positions(rule, d, par, costs["total"])
    return pnl(rule, d, w, costs)


def metrics(p: pd.DataFrame, rng=None) -> dict:
    net, gross, w = p["net"].to_numpy(), p["gross"].to_numpy(), p["w"].to_numpy()
    tr = w != 0
    T, n_tr, units = len(net), int(tr.sum()), float(np.abs(w).sum())
    sr = sharpe(net)
    t, pval = hac_t(net)
    sk, ku = moments(net)
    out = {"days": T, "trades": n_tr, "units_traded": units,
           "mean_gross_per_trade": gross[tr].mean() if n_tr else np.nan,
           "mean_net_per_trade": net[tr].mean() if n_tr else np.nan,
           "median_net_per_trade": float(np.median(net[tr])) if n_tr else np.nan,
           "mean_net_per_day": net.mean(), "median_net_per_day": float(np.median(net)),
           "mean_cost_per_trade": p["cost_total"].to_numpy()[tr].mean() if n_tr else np.nan,
           "hit_rate": float((net[tr] > 0).mean()) if n_tr else np.nan,
           "sr_daily": sr, "sr_annual": sr * math.sqrt(ANNUAL), "hac_t": t, "p_one_sided": pval,
           **max_drawdown(net, p.index), "skew": sk, "kurt": ku, "psr0": psr(sr, T, sk, ku, 0.0)}
    out["breakeven_cost_rt"] = gross.sum() / units if units else np.nan
    stat_unit = p["cost_statutory"].sum() / units if units else np.nan
    out["mean_statutory_per_unit"] = stat_unit
    out["breakeven_slip_per_side"] = (out["breakeven_cost_rt"] - stat_unit) / 2 if units else np.nan
    if rng is not None:
        bs = bootstrap_sharpe(net, rng)
        out.update({"boot_sr_annual_lo": float(np.nanpercentile(bs, 2.5)), "boot_sr_annual_hi": float(np.nanpercentile(bs, 97.5)),
                    "boot_share_sr_le_0": float(np.nanmean(bs <= 0)), "boot_B": N_BOOT, "boot_mean_block": BOOT_BLOCK})
    return out


def family_a_z(path: Path) -> np.ndarray:
    a = pd.read_csv(path)
    p = a["p"].clip(lower=1e-300).to_numpy(float)
    return np.sign(a["coef"].to_numpy(float)) * stats.norm.isf(p / 2)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COL = {"P": "#2a78d6", "S1": "#eb6834", "S2": "#1baf7a", "D": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_equity(daily: dict, gross_p: pd.Series, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURF)
    _style(ax)
    ax.plot(gross_p.index, gross_p.cumsum(), color=COL["P"], lw=1.5, ls="--", label="P before costs")
    for r in RULES:
        c = daily[r]["net"].cumsum()
        ax.plot(c.index, c, color=COL[r], lw=2, label=RULE_NAMES[r] + ", net")
        ax.annotate(r, (c.index[-1], c.iloc[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(pd.Timestamp("2026-04-01"), color=INK2, lw=0.8, ls=":")
    ax.text(pd.Timestamp("2026-04-04"), ax.get_ylim()[1], "STT 0.02% to 0.05%", color=INK2, fontsize=8, va="top")
    ax.set_ylabel("cumulative profit and loss, bp of notional", color=INK, fontsize=10)
    ax.set_title("F2 trading rules, test period 2025 to 31 Jul 2026, base-case costs (REAL DATA)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_grid(grid: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = grid[grid["slippage_per_side"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF)
    for ax, col, lab in ((axes[0], "sr_annual", "annualised Sharpe ratio, net"),
                         (axes[1], "mean_net_per_trade", "mean net bp per trade")):
        _style(ax)
        for r in RULES:
            x = g[g["rule"] == r]
            ax.plot(x["slippage_per_side"], x[col], color=COL[r], lw=2, marker="o", ms=5, label=RULE_NAMES[r])
            if ax is axes[0]:  # direct labels on one panel only (the end points overlap on the other)
                ax.annotate(r, (x["slippage_per_side"].iloc[-1], x[col].iloc[-1]), xytext=(5, 0), textcoords="offset points",
                            color=INK, fontsize=9, va="center")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(SLIP_BASE, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("slippage per side, bp (statutory costs by date always included; dotted = base case)", color=INK, fontsize=8)
        ax.set_ylabel(lab, color=INK, fontsize=10)
        ax.set_xticks(list(SLIP_GRID))
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.suptitle("Cost stress grid, test period 2025 to 31 Jul 2026 (REAL DATA)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse_v2")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/f2_economics")
    a = ap.parse_args()
    t0 = time.time()
    run, raw, out = Path(a.run), Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg_csv = Path("results/momentum_gex/regressions.csv")
    fam_a_csv = Path("results/falsification/t7_family_A_original_tests.csv")
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    d = load_frame(run, raw)
    rep = replicate_r5(d, reg_csv)
    say("REAL DATA. P90: economic size of F2 after realistic costs. Research evaluation, not trading advice.")
    say(f"Pre-registration: {PREREG_FILE}, commit {PREREG_COMMIT}.")
    say(f"Replication of R5 Pro binary interaction: {rep['coef']:.10f} (p {rep['p']:.6f}, n {rep['n']}) vs file "
        f"{rep['coef_on_disk']:.10f} (n {rep['n_on_disk']}); abs diff {rep['abs_diff_coef']:.2e}; ok = {rep['ok']}")
    if not rep["ok"]:
        raise SystemExit("R5 replication failed; stopping before any trading metric (pre-registered check).")

    test = d[(d.index >= TEST_START) & (d.index <= TEST_END)]
    train = d[(d.index >= TRAIN_START) & (d.index <= TRAIN_END)]
    auction = d[d.index >= AUCTION_START]
    say(f"Days: all {len(d)}, training {len(train)} ({train.index.min().date()} to {train.index.max().date()}), "
        f"test {len(test)} ({test.index.min().date()} to {test.index.max().date()}), auction {len(auction)}.")
    say(f"Negative Pro GEX days: training {int((train['neg'] == 1).sum())}, test {int((test['neg'] == 1).sum())}, "
        f"auction {int((auction['neg'] == 1).sum())}; missing GEX in test: {int(test['neg'].isna().sum())}; "
        f"missing 15:00 bar in test: {int(test['p_1500'].isna().sum())}.")

    # cost schedule
    cs = cost_schedule_table()
    cs.to_csv(out / "cost_schedule.csv", index=False)
    say("")
    say("== cost schedule, bp per round trip, base case (1 bp slippage per side)")
    say(cs.to_string(index=False))

    # walk-forward
    wf = walk_forward(d, test.index)
    wf.to_csv(out / "walkforward_params.csv", index=False)
    par_test = params_by_day(wf, test.index)
    assert (wf["train_end"] < wf["block_start"]).all()
    say("")
    say(f"== walk-forward: {len(wf)} blocks of up to {BLOCK} test days, expanding window from {TRAIN_START.date()}")
    say(wf[["block", "block_start", "block_end", "train_end", "n_train", "n_train_neg", "s", "a", "b", "c", "d"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # in-sample and auction parameters (descriptive samples only)
    par_train = pd.DataFrame([fit_params(train)] * len(train), index=train.index)[["s", "a", "b", "c", "d"]]
    pre_auction = d[d.index < AUCTION_START]
    par_auct = pd.DataFrame([fit_params(pre_auction)] * len(auction), index=auction.index)[["s", "a", "b", "c", "d"]]

    # base case, test period, with bootstrap
    rows, daily, trades = [], {}, []
    for r in RULES:
        p = run_rule(r, test, par_test, SLIP_BASE)
        daily[r] = p
        rng = np.random.default_rng([MASTER_SEED, STUDY_SEED, RULE_IDS[r]])
        rows.append({"sample": "test", "rule": r, "costs": "base", **metrics(p, rng)})
        trades.append(_trade_rows("test", r, test, p))
    for sample, frame, par in (("training (in-sample context)", train, par_train),
                               ("auction days (not tradable as measured)", auction, par_auct)):
        for r in RULES:
            p = run_rule(r, frame, par, SLIP_BASE)
            rows.append({"sample": sample, "rule": r, "costs": "base", **metrics(p)})
            trades.append(_trade_rows(sample, r, frame, p))
    met = pd.DataFrame(rows)

    # DSR
    z_a = family_a_z(fam_a_csv)
    tm = met[(met["sample"] == "test")].set_index("rule")
    z_this = tm.loc[list(RULES), "hac_t"].to_numpy(float)
    T = int(tm.loc["P", "days"])
    sr_trials = np.concatenate([z_a, z_this]) / math.sqrt(T)
    V = float(np.var(sr_trials, ddof=1))
    drows = []
    for r in RULES:
        x = tm.loc[r]
        for N in N_SENS:
            s0 = sr0(V, N)
            drows.append({"rule": r, "N": N, "primary_N": N == N_PRIMARY, "T": int(x["days"]), "V": V,
                          "sd_trial_t": math.sqrt(V * T), "multiplier": expected_max_multiplier(N),
                          "sr0_daily": s0, "sr0_annual": s0 * math.sqrt(ANNUAL), "sr_daily": x["sr_daily"],
                          "sr_annual": x["sr_annual"], "skew": x["skew"], "kurt": x["kurt"], "psr0": x["psr0"],
                          "dsr": dsr(x["sr_daily"], int(x["days"]), x["skew"], x["kurt"], V, N)})
    dtab = pd.DataFrame(drows)
    dtab.to_csv(out / "dsr.csv", index=False)
    met["dsr_N824"] = met.apply(lambda q: dtab[(dtab["rule"] == q["rule"]) & (dtab["N"] == N_PRIMARY)]["dsr"].iloc[0]
                                if q["sample"] == "test" else np.nan, axis=1)
    met.to_csv(out / "metrics.csv", index=False)
    met[met["sample"].str.startswith("auction")].to_csv(out / "auction_days.csv", index=False)
    pd.concat(trades, ignore_index=True).to_csv(out / "trades.csv", index=False)

    # bootstrap table
    bt = met[met["sample"] == "test"][["rule", "sr_annual", "boot_sr_annual_lo", "boot_sr_annual_hi", "boot_share_sr_le_0",
                                         "boot_B", "boot_mean_block"]].copy()
    bt["seed"] = [f"[{MASTER_SEED}, {STUDY_SEED}, {RULE_IDS[r]}]" for r in bt["rule"]]
    bt.to_csv(out / "bootstrap_sharpe.csv", index=False)

    # daily P&L file
    dp = pd.DataFrame(index=test.index)
    dp["neg_pro_gex"], dp["r_rod_bp"], dp["R_bp"], dp["R_D_bp"] = test["neg"], test["r_rod_bp"], test["R_bp"], test["R_D_bp"]
    for r in RULES:
        for k in ("w", "gross", "cost_total", "net"):
            dp[f"{r}_{k}"] = daily[r][k]
        dp[f"{r}_cum_net"] = daily[r]["net"].cumsum()
    dp.index.name = "date"
    dp.to_csv(out / "daily_pnl.csv")

    # cost stress grid (S2 re-filtered at each level; break-even for S2 at base-case trades is in metrics.csv)
    grows = []
    for r in RULES:
        for slip in (None,) + SLIP_GRID:
            p = run_rule(r, test, par_test, slip)
            m = metrics(p)
            grows.append({"rule": r, "costs": "gross" if slip is None else f"statutory + {slip} bp per side",
                          "slippage_per_side": slip, **{k: m[k] for k in ("trades", "mean_gross_per_trade", "mean_net_per_trade",
                                                                           "mean_net_per_day", "hit_rate", "sr_annual", "hac_t",
                                                                           "p_one_sided", "mdd", "psr0")}})
    grid = pd.DataFrame(grows)
    grid.to_csv(out / "cost_grid.csv", index=False)

    fig_equity(daily, run_rule("P", test, par_test, None)["gross"], out / "fig_equity_curve.png")
    fig_grid(grid, out / "fig_cost_grid.png")

    # report
    show = ["rule", "days", "trades", "mean_gross_per_trade", "mean_cost_per_trade", "mean_net_per_trade",
            "median_net_per_trade", "mean_net_per_day", "median_net_per_day", "hit_rate", "sr_annual", "hac_t",
            "p_one_sided", "mdd", "mdd_days", "breakeven_cost_rt", "breakeven_slip_per_side", "psr0"]
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    say("")
    say("== test period, base-case costs (bp of notional; flat days count as 0 in per-day figures)")
    say(met[met["sample"] == "test"][show + ["boot_sr_annual_lo", "boot_sr_annual_hi", "dsr_N824"]].to_string(index=False, float_format=fmt))
    for sample in ("training (in-sample context)", "auction days (not tradable as measured)"):
        say("")
        say(f"== {sample}, base-case costs")
        say(met[met["sample"] == sample][show].to_string(index=False, float_format=fmt))
    say("")
    say(f"== Deflated Sharpe Ratio: V = {V:.3e} from {len(sr_trials)} trial t-statistics (78 family A + 4 here) "
        f"scaled by sqrt(T), T = {T}; spread of trial t-statistics sqrt(V T) = {math.sqrt(V * T):.3f}")
    say(dtab[["rule", "N", "multiplier", "sr0_annual", "sr_annual", "psr0", "dsr"]].to_string(index=False, float_format=fmt))
    say("")
    say("== cost stress grid (test period)")
    say(grid.drop(columns=["slippage_per_side"]).to_string(index=False, float_format=fmt))

    prim = dtab[(dtab["rule"] == "P") & (dtab["N"] == N_PRIMARY)].iloc[0]
    pm = tm.loc["P"]
    if prim["dsr"] > DSR_THRESHOLD:
        verdict = "economically meaningful"
    elif pm["mean_net_per_day"] <= 0:
        verdict = "not economically meaningful (no net edge)"
    else:
        verdict = "not economically meaningful (positive but not distinguishable from selection and noise)"
    say("")
    say(f"== PRE-REGISTERED VERDICT (rule P, test period, base-case costs, N = {N_PRIMARY}): DSR = {prim['dsr']:.4f} "
        f"vs threshold {DSR_THRESHOLD} -> F2 is {verdict}.")
    leads = [r for r in ("S1", "S2", "D") if dtab[(dtab["rule"] == r) & (dtab["N"] == N_PRIMARY)]["dsr"].iloc[0] > DSR_THRESHOLD]
    say(f"Secondary rules with DSR > {DSR_THRESHOLD} at N = {N_PRIMARY}: {leads if leads else 'none'}.")

    info = {"study": "P90 F2 economics", "label": "REAL DATA", "prereg_file": PREREG_FILE, "prereg_commit": PREREG_COMMIT,
            "verdict": verdict, "dsr_primary": float(prim["dsr"]), "seeds": {r: [MASTER_SEED, STUDY_SEED, i] for r, i in RULE_IDS.items()},
            "bootstrap": {"B": N_BOOT, "mean_block": BOOT_BLOCK, "method": "stationary (Politis and Romano 1994), percentile"},
            "trials": {"N_primary": N_PRIMARY, "N_sensitivity": list(N_SENS), "V": V, "T": T},
            "samples": {"train": [str(TRAIN_START.date()), str(TRAIN_END.date())], "test": [str(test.index.min().date()), str(test.index.max().date())],
                        "auction_from": str(AUCTION_START.date())},
            "costs": {"slippage_base_bp_per_side": SLIP_BASE, "grid": list(SLIP_GRID), "notional_rs": NOTIONAL_RS,
                      "brokerage_rs_per_order": BROKERAGE_RS_PER_ORDER},
            "replication_r5": rep,
            "input_sha256_16": {str(p): sha16(p) for p in (run / "daily_market.csv", run / "daily_gex_panel.csv",
                                                           raw / "nifty_1min.csv", reg_csv, fam_a_csv)},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "statsmodels": statsmodels.__version__, "scipy": scipy.__version__},
            "runtime_seconds": round(time.time() - t0, 1)}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    (out.parent / "f2_economics_stdout.txt").write_text("\n".join(log) + "\n")


def _trade_rows(sample: str, rule: str, frame: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    m = p["w"] != 0
    t = p[m].copy()
    t.insert(0, "sample", sample)
    t.insert(1, "rule", rule)
    t["r_rod_bp"] = frame.loc[m, "r_rod_bp"]
    t["is_expiry"] = frame.loc[m, "is_expiry"]
    t["is_monthly_expiry"] = frame.loc[m, "is_monthly_expiry"]
    t.index.name = "date"
    return t.reset_index()


if __name__ == "__main__":
    main()
