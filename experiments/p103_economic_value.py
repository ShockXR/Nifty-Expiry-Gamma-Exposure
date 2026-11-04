"""P103: is the hybrid's forecasting gain worth anything economically? REAL DATA.

Registered analysis plan: docs/prereg/P103_ECONOMIC_VALUE.md, committed as
77b435aec493b7e0acf1eab90e51490bb9377f17 (3 Nov 2026, 03:38 UTC) before this script was written or run.
This is a research evaluation of the economic size of a statistical finding, not trading advice.

Part A (realized utility; Bollerslev, Hood, Huss and Pedersen 2018). A mean-variance investor with a constant
Sharpe ratio SR and risk aversion gamma who holds x = SR / (gamma sqrt(F)) of wealth in Nifty over a window with
forecast variance F and realized variance RV earns the realized utility (SR^2 / gamma) [sqrt(RV/F) - RV/(2F)].
Annualised in percent of wealth: 100 (SR_a^2 / gamma) x mean bracket, whatever the window length. The fee of
the hybrid over a benchmark is the difference. Day-level average brackets, one-sided Diebold-Mariano test
(repository function). H1: 30 minutes, origins 0, 30, ..., 330, hybrid_base vs mcsgarch. S1 to S4 (Holm): 15
minutes and the last hour vs mcsgarch, 15 and 30 minutes vs mcsgarch_iv.

Part B (trading). Long Nifty over the same 30-minute windows, w = min(4, sigma*_w / sqrt(F)), sigma* = 20% a
year = SR/gamma, flat outside 09:15 to 15:15, index returns standing in for futures, P90's costs on every change
in position. Tracking test T1, net-utility test T2, Sharpe ratios, Deflated Sharpe Ratio with N = 4.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p103_economic_value.py
    PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p103_economic_value.py

Writes results/p103_economic_value/ (CSV tables, run_info.json, stdout.txt).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import f2_economics as fe  # noqa: E402  (P90: costs, PSR, DSR, HAC t, drawdown, bootstrap)

from niftygex.evaluation import diebold_mariano, holm, qlike, subgroups  # noqa: E402

PREREG_FILE = "docs/prereg/P103_ECONOMIC_VALUE.md"
PREREG_COMMIT = "77b435aec493b7e0acf1eab90e51490bb9377f17"

# ---- Part A
SR_A, GAMMA = 0.4, 2.0                      # annualised Sharpe ratio and relative risk aversion (primary)
FEE_GRID_SR = (0.2, 0.4, 0.6, 0.8)
FEE_GRID_GAMMA = (1.0, 2.0, 5.0, 10.0)
TINY, SMALL = 0.10, 0.50                    # size labels, percent of wealth a year
RS_PER_CRORE = 1e7
ORIGINS = {
    "next30_primary": list(range(0, 331, 30)),     # H1, S4, Part B
    "next30_offset15": list(range(15, 346, 30)),   # R1
    "next30_all": list(range(0, 346, 15)),         # R2 (overlapping)
    "next30_no0915": list(range(30, 331, 30)),     # R3
    "next15_all": list(range(0, 361, 15)),         # S1, S3
    "last_hour": [315],                            # S2
}
WINDOW_TARGET = {k: ("next15" if k.startswith("next15") else "last_hour" if k == "last_hour" else "next30") for k in ORIGINS}
FULL_ORIGINS = {"next15": list(range(0, 361, 15)), "next30": list(range(0, 346, 15)), "last_hour": [315]}
HORIZON = {"next15": 15, "next30": 30, "last_hour": 60}
HOLDOUT_START = "2025-07-01"
AUCTION_START = "2026-08-03"
SUBSETS = ["expiry", "non_expiry", "pre_auction", "development", "holdout",
           "multi_weekly", "single_weekly_thu", "single_weekly_tue"]

# ---- Part B
DAYS_YEAR = 252
WIN_PER_DAY = 12
N_WIN_YEAR = DAYS_YEAR * WIN_PER_DAY        # 3024
TARGET = SR_A / GAMMA                       # 20% a year
SIGMA_W = TARGET / math.sqrt(N_WIN_YEAR)    # per 30-minute window
SR_W = SR_A / math.sqrt(N_WIN_YEAR)
CAP = 4.0
CONST_W = 2.0
STRATS = ["hybrid_base", "mcsgarch", "mcsgarch_iv", "constant"]
STRAT_IDS = {s: i for i, s in enumerate(STRATS)}
SLIP_BASE = 1.0
SLIP_GRID = (None, 0.0, 0.5, 1.0, 2.0, 3.0)  # None = gross
TRACK_BAND = (0.8, 1.25)
N_BOOT, BOOT_BLOCK = 10_000, 5
MASTER_SEED, STUDY_SEED = 20261102, 103
DSR_THRESHOLD = 0.95
N_PRIMARY = 4

LABEL = "REAL DATA"


# ---------------------------------------------------------------------------
# Part A: realized utility (tested in tests/test_p103_economic_value.py)
# ---------------------------------------------------------------------------


def bracket(y, f):
    """sqrt(RV/F) - RV/(2F): realized utility per unit of SR^2/gamma; at most 1/2, reached at F = RV."""
    x = np.asarray(y, float) / np.asarray(f, float)
    return np.sqrt(x) - 0.5 * x


def utility_loss(y, f):
    """1/2 - bracket = (1/2)(1 - sqrt(RV/F))^2 >= 0."""
    x = np.asarray(y, float) / np.asarray(f, float)
    return 0.5 * (1.0 - np.sqrt(x)) ** 2


def annual_utility_pct(mean_bracket, sr: float = SR_A, gamma: float = GAMMA):
    """Annualised realized utility in percent of wealth: 100 (SR_a^2 / gamma) x mean bracket (any window length)."""
    return 100.0 * sr ** 2 / gamma * np.asarray(mean_bracket, float)


def max_utility_pct(sr: float = SR_A, gamma: float = GAMMA) -> float:
    return 100.0 * sr ** 2 / (2.0 * gamma)


def size_label(fee_pct: float) -> str:
    if not np.isfinite(fee_pct) or fee_pct < TINY:
        return "tiny"
    return "small" if fee_pct < SMALL else "material"


def select_windows(pred: pd.DataFrame, origins) -> pd.DataFrame:
    """Rows whose origin is in ``origins``; every day must have every origin exactly once."""
    origins = sorted(origins)
    sel = pred[pred["minute"].isin(origins)]
    per_day = sel.groupby("date")["minute"].apply(lambda m: sorted(m.tolist()))
    bad = per_day[per_day.map(lambda m: m != origins)]
    if len(bad) or sel["date"].nunique() != pred["date"].nunique():
        raise ValueError(f"{len(bad)} days without exactly the origins {origins}")
    return sel


def windows_do_not_overlap(origins, horizon: int) -> bool:
    o = sorted(origins)
    return all(b - a >= horizon for a, b in zip(o, o[1:]))


def daily_mean(sel: pd.DataFrame, values) -> pd.Series:
    return pd.Series(np.asarray(values, float), index=sel.index).groupby(sel["date"].to_numpy()).mean()


def utility_comparison(sel: pd.DataFrame, model: str, bench: str, sr: float = SR_A, gamma: float = GAMMA) -> dict:
    """Fee of ``model`` over ``bench`` and the one-sided DM test on day-level utility losses."""
    bm = daily_mean(sel, bracket(sel["y"], sel[model]))
    bb = daily_mean(sel, bracket(sel["y"], sel[bench]))
    dm = diebold_mariano(0.5 - bb.to_numpy(), 0.5 - bm.to_numpy(), alternative="greater")
    u_m, u_b = float(annual_utility_pct(bm.mean(), sr, gamma)), float(annual_utility_pct(bb.mean(), sr, gamma))
    fee = u_m - u_b
    u_max = max_utility_pct(sr, gamma)
    return {"model": model, "benchmark": bench, "n_days": int(len(bm)), "n_windows": int(len(sel)),
            "mean_bracket_model": float(bm.mean()), "mean_bracket_benchmark": float(bb.mean()),
            "utility_model_pct_year": u_m, "utility_benchmark_pct_year": u_b, "utility_max_pct_year": u_max,
            "fee_pct_year": fee, "fee_rs_per_crore_year": fee / 100.0 * RS_PER_CRORE,
            "share_of_shortfall_closed": fee / (u_max - u_b) if u_max != u_b else np.nan,
            "dm_mean_diff": dm["mean_diff"], "dm_stat": dm["stat"], "p_one_sided": dm["p"], "dm_n": dm["n"],
            "share_days_model_better": float((bm.to_numpy() > bb.to_numpy()).mean())}


# ---------------------------------------------------------------------------
# Part B: the trading rule (tested)
# ---------------------------------------------------------------------------


def positions(F, sigma_w: float = SIGMA_W, cap: float = CAP) -> np.ndarray:
    """Units of notional per unit of capital: min(cap, sigma_w / sqrt(F))."""
    return np.minimum(cap, sigma_w / np.sqrt(np.asarray(F, float)))


def daily_turnover(W) -> tuple[np.ndarray, np.ndarray]:
    """(total, opening plus closing) turnover per day for positions W (days x windows), flat before and after."""
    W = np.asarray(W, float)
    oc = np.abs(W[:, 0]) + np.abs(W[:, -1])
    reb = np.abs(np.diff(W, axis=1)).sum(axis=1)
    return oc + reb, oc


def daily_cost_bp(turnover, rt_cost_bp) -> np.ndarray:
    """Units bought = units sold = turnover / 2 within a day, so cost = turnover / 2 x round-trip cost."""
    return 0.5 * np.asarray(turnover, float) * np.asarray(rt_cost_bp, float)


def realized_strategy_vol(W, Y) -> np.ndarray:
    """Annualised realized volatility of the strategy each day: sqrt(252 x sum_k w_k^2 y_k)."""
    return np.sqrt(DAYS_YEAR * (np.asarray(W, float) ** 2 * np.asarray(Y, float)).sum(axis=1))


def tracking_loss(W, Y, target: float = TARGET) -> np.ndarray:
    s = realized_strategy_vol(W, Y)
    with np.errstate(divide="ignore"):
        return np.where(s > 0, np.log(s / target) ** 2, np.nan)


def net_utility_daily(W, Y, cost_bp, sr_w: float = SR_W, gamma: float = GAMMA, returns=None) -> np.ndarray:
    """Daily realized utility (fraction of wealth) net of costs; constant-Sharpe expected return unless ``returns``."""
    W, Y = np.asarray(W, float), np.asarray(Y, float)
    ret = sr_w * np.sqrt(Y) if returns is None else np.asarray(returns, float)
    return (W * ret - 0.5 * gamma * W ** 2 * Y).sum(axis=1) - np.asarray(cost_bp, float) / 1e4


def window_matrices(R: np.ndarray, origins, horizon: int):
    """Window sums of 1-minute log returns and of their squares, days x windows (R is days x 375 minutes)."""
    lr = np.column_stack([R[:, m:m + horizon].sum(axis=1) for m in origins])
    rv = np.column_stack([(R[:, m:m + horizon] ** 2).sum(axis=1) for m in origins])
    return lr, rv


def sharpe_annual(x) -> float:
    return fe.sharpe(x) * math.sqrt(DAYS_YEAR)


def strategy_metrics(gross_bp, cost_bp, W, Y, Rw, turnover, oc) -> dict:
    net = np.asarray(gross_bp) - np.asarray(cost_bp)
    t, p = fe.hac_t(net)
    sk, ku = fe.moments(net)
    sr = fe.sharpe(net)
    s = realized_strategy_vol(W, Y)
    tl = tracking_loss(W, Y)
    wr = (W * Rw).ravel()
    u_net = net_utility_daily(W, Y, cost_bp)
    u_real = net_utility_daily(W, Y, cost_bp, returns=Rw)
    return {"days": len(net), "mean_net_bp_day": float(net.mean()), "median_net_bp_day": float(np.median(net)),
            "mean_gross_bp_day": float(np.mean(gross_bp)), "mean_cost_bp_day": float(np.mean(cost_bp)),
            "ann_mean_net_pct": float(net.mean() * DAYS_YEAR / 100), "ann_vol_net_pct": float(net.std(ddof=1) * math.sqrt(DAYS_YEAR) / 100),
            "sr_daily": sr, "sr_annual": sr * math.sqrt(DAYS_YEAR), "hac_t": t, "p_one_sided": p,
            **fe.max_drawdown(net), "skew": sk, "kurt": ku, "psr0": fe.psr(sr, len(net), sk, ku, 0.0),
            "turnover_per_day": float(np.mean(turnover)), "turnover_open_close_per_day": float(np.mean(oc)),
            "turnover_rebalancing_per_day": float(np.mean(turnover - oc)),
            "cost_pct_year": float(np.mean(cost_bp) * DAYS_YEAR / 100),
            "mean_position": float(np.mean(W)), "share_windows_at_cap": float(np.mean(W >= CAP - 1e-12)),
            "realized_vol_rv_pct": float(math.sqrt(DAYS_YEAR * np.mean((W ** 2 * Y).sum(axis=1))) * 100),
            "realized_vol_returns_pct": float(np.std(wr, ddof=1) * math.sqrt(N_WIN_YEAR) * 100),
            "target_vol_pct": TARGET * 100,
            "rmsld": float(math.sqrt(np.nanmean(tl))), "mean_daily_vol_pct": float(np.mean(s) * 100),
            "share_days_in_band": float(np.mean((s >= TRACK_BAND[0] * TARGET) & (s <= TRACK_BAND[1] * TARGET))),
            "net_utility_pct_year": float(np.mean(u_net) * DAYS_YEAR * 100),
            "net_utility_realized_returns_pct_year": float(np.mean(u_real) * DAYS_YEAR * 100)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_forecasts(run: Path) -> dict:
    out = {}
    for tgt in ("next15", "next30", "last_hour"):
        d = pd.read_csv(run / f"forecasts_{tgt}.csv", parse_dates=["date"])
        out[tgt] = d.sort_values(["date", "minute"]).reset_index(drop=True)
    return out


def model_columns(pred: pd.DataFrame) -> list[str]:
    fixed = {"t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y"}
    return [c for c in pred.columns if c not in fixed]


def session_grid(bars_path: Path, days: pd.DatetimeIndex) -> dict:
    """The pipeline's session grid (niftygex.data.to_session_grid) from 10 days before the first test day."""
    from niftygex.data import load_minute_bars, to_session_grid
    b = load_minute_bars(bars_path)
    b = b[(b.index >= days.min() - pd.Timedelta(days=10)) & (b.index < days.max() + pd.Timedelta(days=1))]
    return to_session_grid(b)


def minute_returns(grid: dict, days: pd.DatetimeIndex) -> np.ndarray:
    """1-minute log returns of the test days, days x 375, NaN set to 0 (as in the pipeline's market dict)."""
    pos = pd.Index(grid["days"])
    missing = days.difference(pos)
    if len(missing):
        raise SystemExit(f"STOP: {len(missing)} test days have no minute bars, e.g. {missing[:3].tolist()}")
    return np.nan_to_num(grid["returns"][pos.get_indexer(days)], nan=0.0)


def index_drift(grid: dict, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Descriptive (added in phase 2): mean Nifty log return per day in parts of the day, bp, with a plain t."""
    pos = pd.Index(grid["days"])
    i = pos.get_indexer(days)
    R = np.nan_to_num(grid["returns"], nan=0.0)
    C = grid["close"]
    last = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else np.nan for r in C])
    parts = {"overnight: previous day's last bar close to the 09:15 bar's open": np.log(grid["open"][i, 0] / last[i - 1]),
             "09:15 to 15:15 (the windows traded in Part B)": R[i, :360].sum(axis=1),
             "of which the first minute (09:15 bar, open to close)": R[i, 0],
             "15:15 to the last bar's close (not traded)": R[i, 360:].sum(axis=1)}
    rows = []
    for k, x in parts.items():
        x = x[np.isfinite(x)] * 1e4
        rows.append({"label": LABEL, "part_of_day": k, "days": len(x), "mean_bp_per_day": x.mean(),
                     "sd_bp": x.std(ddof=1), "t_iid": x.mean() / x.std(ddof=1) * math.sqrt(len(x)),
                     "sum_log_return": x.sum() / 1e4})
    return pd.DataFrame(rows)


def pivot(sel: pd.DataFrame, col: str, days: pd.DatetimeIndex, origins) -> np.ndarray:
    return sel.pivot(index="date", columns="minute", values=col).reindex(index=days, columns=origins).to_numpy(float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse_v2")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/p103_economic_value")
    a = ap.parse_args()
    t0 = time.time()
    run, raw, out = Path(a.run), Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(s=""):
        print(s, flush=True)
        log.append(str(s))

    fmt = lambda v: f"{v:.6g}"  # noqa: E731
    say(f"{LABEL}. P103: economic value of the forecasting hybrid. Research evaluation, not trading advice.")
    say(f"Registered plan: {PREREG_FILE}, commit {PREREG_COMMIT}.")

    fc = load_forecasts(run)
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        say(f"check {name}: {'ok' if ok else 'FAILED'} ({detail})")
        if not ok:
            pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)
            raise SystemExit(f"STOP (registered stop rule): {name}")

    # ---- stop rules: origins, positive forecasts, published QLIKE margins
    days = pd.DatetimeIndex(sorted(fc["next30"]["date"].unique()))
    for tgt, d in fc.items():
        n_days = d["date"].nunique()
        try:
            select_windows(d, FULL_ORIGINS[tgt])
            ok = n_days == len(days) == 792
        except ValueError as e:
            ok = False
            say(str(e))
        check(f"origins_{tgt}", ok, f"{n_days} days, {len(FULL_ORIGINS[tgt])} origins each")
        mc = model_columns(d)
        vals = d[mc].to_numpy(float)
        check(f"forecasts_positive_{tgt}", np.isfinite(vals).all() and (vals > 0).all(),
              f"{len(mc)} model columns, min {np.nanmin(vals):.3e}")
        sb = pd.read_csv(run / f"scoreboard_{tgt}.csv")
        pub = float(sb[(sb["group"] == "all") & (sb["model"] == "hybrid_base")]["qlike_vs_mcsgarch_pct"].iloc[0])
        mine = 100 * (qlike(d["y"], d["hybrid_base"]).mean() / qlike(d["y"], d["mcsgarch"]).mean() - 1)
        check(f"published_qlike_margin_{tgt}", abs(mine - pub) < 1e-9, f"recomputed {mine:.10f}% vs published {pub:.10f}%")
    for k, o in ORIGINS.items():
        if k not in ("next30_all",):
            check(f"no_overlap_{k}", windows_do_not_overlap(o, HORIZON[WINDOW_TARGET[k]]), f"origins {o[0]} to {o[-1]}")

    # ---- minute returns, rebuilt y
    grid = session_grid(raw / "nifty_1min.csv", days)
    R = minute_returns(grid, days)
    o30 = ORIGINS["next30_primary"]
    sel30 = select_windows(fc["next30"], o30)
    LR, RVb = window_matrices(R, o30, 30)
    Y = pivot(sel30, "y", days, o30)
    rel = np.abs(RVb - Y) / np.maximum(Y, 1e-300)
    check("y_rebuilt_from_bars", np.nanmax(rel) <= 1e-9 and np.isfinite(Y).all(),
          f"{Y.size} windows, largest relative difference {np.nanmax(rel):.2e}")
    Rw = np.expm1(LR)
    pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)

    # =======================================================================
    # Part A
    # =======================================================================
    groups = {}
    regs = []

    def comp(role, test_id, window_set, model, bench, subset="all", note=""):
        d = fc[WINDOW_TARGET[window_set]]
        sel = select_windows(d, ORIGINS[window_set])
        if subset != "all":
            key = (window_set, subset)
            if key not in groups:
                groups[key] = subgroups(sel, holdout_start=HOLDOUT_START, auction_start=AUCTION_START)
            sel = sel[groups[key][subset]]
        r = utility_comparison(sel, model, bench)
        row = {"test_id": test_id, "role": role, "target": WINDOW_TARGET[window_set], "window_set": window_set,
               "origins": " ".join(map(str, ORIGINS[window_set])), "subset": subset, **r, "note": note}
        regs.append(row)
        return row

    comp("primary", "H1", "next30_primary", "hybrid_base", "mcsgarch")
    comp("secondary", "S1", "next15_all", "hybrid_base", "mcsgarch")
    comp("secondary", "S2", "last_hour", "hybrid_base", "mcsgarch")
    comp("secondary", "S3", "next15_all", "hybrid_base", "mcsgarch_iv")
    comp("secondary", "S4", "next30_primary", "hybrid_base", "mcsgarch_iv")
    for bench in ("mcsgarch", "mcsgarch_iv"):
        comp("robustness", "R1", "next30_offset15", "hybrid_base", bench, note="other non-overlapping offset")
        comp("robustness", "R2", "next30_all", "hybrid_base", bench, note="all 24 overlapping origins, day averages")
        comp("robustness", "R3", "next30_no0915", "hybrid_base", bench, note="primary windows without 09:15")
    for sub in SUBSETS:
        comp("robustness", "R4-H1", "next30_primary", "hybrid_base", "mcsgarch", subset=sub)
        comp("robustness", "R4-S4", "next30_primary", "hybrid_base", "mcsgarch_iv", subset=sub)
    ut = pd.DataFrame(regs)
    sec = ut["role"] == "secondary"
    ut["p_holm_secondary"] = np.nan
    ut.loc[sec, "p_holm_secondary"] = holm(ut.loc[sec, "p_one_sided"].to_numpy())
    ut["supported"] = ""  # robustness rows carry no decision
    prim = ut["role"] == "primary"
    ut.loc[prim, "supported"] = np.where(ut.loc[prim, "p_one_sided"] < 0.05, "yes", "no")
    ut.loc[sec, "supported"] = np.where(ut.loc[sec, "p_holm_secondary"] < 0.05, "yes", "no")
    ut.insert(0, "label", LABEL)
    ut.to_csv(out / "utility_tests.csv", index=False)

    # R5: every model
    mrows = []
    for ws in ("next15_all", "next30_primary", "last_hour"):
        d = select_windows(fc[WINDOW_TARGET[ws]], ORIGINS[ws])
        bref = bracket(d["y"], d["mcsgarch"]).mean()
        qref = qlike(d["y"], d["mcsgarch"]).mean()
        for m in model_columns(d):
            b = bracket(d["y"], d[m]).mean()
            mrows.append({"label": LABEL, "window_set": ws, "model": m, "mean_bracket": b,
                          "utility_pct_year": float(annual_utility_pct(b)),
                          "fee_vs_mcsgarch_pct_year": float(annual_utility_pct(b - bref)),
                          "qlike": float(qlike(d["y"], d[m]).mean()),
                          "qlike_vs_mcsgarch_pct": float(100 * (qlike(d["y"], d[m]).mean() / qref - 1))})
    um = pd.DataFrame(mrows)
    um["utility_rank"] = um.groupby("window_set")["utility_pct_year"].rank(ascending=False).astype(int)
    um["qlike_rank"] = um.groupby("window_set")["qlike"].rank().astype(int)
    um.to_csv(out / "utility_models.csv", index=False)

    # R6: fee grid and QLIKE on the same windows
    reg = ut[ut["role"].isin(["primary", "secondary"])]
    grows, qrows = [], []
    for _, r in reg.iterrows():
        for sr in FEE_GRID_SR:
            for g in FEE_GRID_GAMMA:
                fee = float(annual_utility_pct(r["mean_bracket_model"] - r["mean_bracket_benchmark"], sr, g))
                grows.append({"label": LABEL, "test_id": r["test_id"], "sr_annual": sr, "gamma": g,
                              "primary_values": sr == SR_A and g == GAMMA, "target_vol_pct": 100 * sr / g,
                              "utility_max_pct_year": max_utility_pct(sr, g), "fee_pct_year": fee,
                              "fee_rs_per_crore_year": fee / 100 * RS_PER_CRORE, "size_label": size_label(fee)})
        d = select_windows(fc[r["target"]], ORIGINS[r["window_set"]])
        qm, qb = qlike(d["y"], d[r["model"]]).mean(), qlike(d["y"], d[r["benchmark"]]).mean()
        qrows.append({"label": LABEL, "test_id": r["test_id"], "window_set": r["window_set"], "model": r["model"],
                      "benchmark": r["benchmark"], "n_windows": len(d), "qlike_model": qm, "qlike_benchmark": qb,
                      "qlike_margin_pct": 100 * (qm / qb - 1),
                      "utility_loss_margin_pct": 100 * ((0.5 - r["mean_bracket_model"]) / (0.5 - r["mean_bracket_benchmark"]) - 1)})
    pd.DataFrame(grows).to_csv(out / "fee_grid.csv", index=False)
    pd.DataFrame(qrows).to_csv(out / "qlike_same_windows.csv", index=False)

    # daily brackets (everything in Part A can be recomputed from these)
    drows = []
    for ws in ORIGINS:
        d = select_windows(fc[WINDOW_TARGET[ws]], ORIGINS[ws])
        dd = pd.DataFrame({m: daily_mean(d, bracket(d["y"], d[m])) for m in ("hybrid_base", "mcsgarch", "mcsgarch_iv")})
        dd.index.name = "date"
        dd = dd.reset_index()
        dd.insert(0, "window_set", ws)
        drows.append(dd)
    pd.concat(drows, ignore_index=True).to_csv(out / "daily_utility.csv", index=False)

    h1 = ut[ut["test_id"] == "H1"].iloc[0]
    say("")
    say("== Part A: realized utility, SR 0.4, gamma 2 (percent of wealth a year; perfect forecast 4%)")
    show = ["test_id", "role", "window_set", "subset", "model", "benchmark", "n_days", "utility_model_pct_year",
            "utility_benchmark_pct_year", "fee_pct_year", "fee_rs_per_crore_year", "share_of_shortfall_closed",
            "dm_stat", "p_one_sided", "p_holm_secondary", "share_days_model_better"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        say(ut[show].to_string(index=False, float_format=fmt))
        say("")
        say("== R6: QLIKE and utility-loss margins on the same windows")
        say(pd.DataFrame(qrows).to_string(index=False, float_format=fmt))
        say("")
        say("== R5: utility rank vs QLIKE rank (top 8 by utility)")
        for ws, g in um.groupby("window_set"):
            say(f"-- {ws}")
            say(g.sort_values("utility_rank").head(8)[["model", "utility_pct_year", "fee_vs_mcsgarch_pct_year",
                                                       "qlike_vs_mcsgarch_pct", "utility_rank", "qlike_rank"]]
                .to_string(index=False, float_format=fmt))

    # =======================================================================
    # Part B
    # =======================================================================
    rt = {s: np.array([fe.cost_components(dd, s)["total"] for dd in days]) if s is not None else np.zeros(len(days))
          for s in SLIP_GRID}
    stat_cost = np.array([fe.cost_components(dd, SLIP_BASE)["statutory"] for dd in days])
    all_models = model_columns(sel30)
    W = {m: positions(pivot(sel30, m, days, o30)) for m in all_models}
    W["constant"] = np.full_like(Y, CONST_W)
    gross = {m: 1e4 * (W[m] * Rw).sum(axis=1) for m in W}
    turn = {m: daily_turnover(W[m]) for m in W}

    mrows, daily_rows = [], []
    for s in STRATS:
        tv, oc = turn[s]
        for slip in SLIP_GRID:
            c = daily_cost_bp(tv, rt[slip])
            mrows.append({"label": LABEL, "strategy": s, "costs": "gross" if slip is None else f"statutory + {slip:g} bp per side",
                          "slippage_per_side": slip, "base_case": slip == SLIP_BASE,
                          **strategy_metrics(gross[s], c, W[s], Y, Rw, tv, oc)})
        c = daily_cost_bp(tv, rt[SLIP_BASE])
        daily_rows.append(pd.DataFrame({"date": days.date, "strategy": s, "gross_bp": gross[s], "turnover": tv,
                                        "turnover_open_close": oc, "cost_bp": c, "net_bp": gross[s] - c,
                                        "realized_vol": realized_strategy_vol(W[s], Y), "tracking_loss": tracking_loss(W[s], Y),
                                        "net_utility": net_utility_daily(W[s], Y, c),
                                        "gross_utility": net_utility_daily(W[s], Y, np.zeros(len(days)))}))
    tm = pd.DataFrame(mrows)
    pd.concat(daily_rows, ignore_index=True).to_csv(out / "trading_daily.csv", index=False)

    # ---- trading tests
    def ttest(test_id, kind, model, bench, slip=SLIP_BASE, rebal_only=False, role="robustness"):
        if kind == "tracking":
            lb, lm = tracking_loss(W[bench], Y), tracking_loss(W[model], Y)
            est = float(math.sqrt(np.nanmean(lm)) - math.sqrt(np.nanmean(lb)))
            est_name = "RMSLD model minus benchmark"
        else:
            def cost(m):
                tv, oc = turn[m]
                return daily_cost_bp(tv - oc if rebal_only else tv, rt[slip])
            ub, um_ = net_utility_daily(W[bench], Y, cost(bench)), net_utility_daily(W[model], Y, cost(model))
            lb, lm = -ub, -um_
            est = float((um_.mean() - ub.mean()) * DAYS_YEAR * 100)
            est_name = "net utility difference, percent of wealth a year"
        dm = diebold_mariano(lb, lm, alternative="greater")
        return {"label": LABEL, "test_id": test_id, "role": role, "kind": kind, "model": model, "benchmark": bench,
                "costs": ("rebalancing trades only, " if rebal_only else "") + ("gross" if slip is None else f"statutory + {slip:g} bp per side")
                if kind == "net_utility" else "not applicable (tracking uses positions only)",
                "estimate": est, "estimate_name": est_name, "dm_mean_diff": dm["mean_diff"], "dm_stat": dm["stat"],
                "p_one_sided": dm["p"], "n_days": dm["n"], "share_days_model_better": float(np.nanmean(lm < lb))}

    trows = [ttest("T1", "tracking", "hybrid_base", "mcsgarch", role="registered"),
             ttest("T2", "net_utility", "hybrid_base", "mcsgarch", role="registered"),
             ttest("T1-iv", "tracking", "hybrid_base", "mcsgarch_iv"),
             ttest("T2-iv", "net_utility", "hybrid_base", "mcsgarch_iv"),
             ttest("T2-rebal", "net_utility", "hybrid_base", "mcsgarch", rebal_only=True)]
    for slip in SLIP_GRID:
        if slip != SLIP_BASE:
            trows.append(ttest("T2-grid", "net_utility", "hybrid_base", "mcsgarch", slip=slip))
    tt = pd.DataFrame(trows)
    tt.to_csv(out / "trading_tests.csv", index=False)

    # ---- DSR
    base = tm[tm["base_case"]].set_index("strategy")
    T = len(days)
    sr4 = base.loc[STRATS, "sr_daily"].to_numpy(float)
    V4 = float(np.var(sr4, ddof=1))
    net33 = {m: gross[m] - daily_cost_bp(turn[m][0], rt[SLIP_BASE]) for m in W}
    sr33 = np.array([fe.sharpe(net33[m]) for m in W])
    V33 = float(np.var(sr33, ddof=1))
    dsr_rows = []
    for s in STRATS:
        x = base.loc[s]
        for N, V, vname in ((N_PRIMARY, V4, "sample variance of the 4 registered strategies"),
                            (33, V33, "sample variance of the 33 sizing rules"),
                            (N_PRIMARY, 1.0 / T, "1/T (pure noise)"), (33, 1.0 / T, "1/T (pure noise)")):
            s0 = fe.sr0(V, N)
            dsr_rows.append({"label": LABEL, "strategy": s, "N": N, "V": V, "V_source": vname,
                             "primary": N == N_PRIMARY and V == V4, "T": T, "multiplier": fe.expected_max_multiplier(N),
                             "sr0_annual": s0 * math.sqrt(DAYS_YEAR), "sr_annual": x["sr_annual"], "skew": x["skew"],
                             "kurt": x["kurt"], "psr0": x["psr0"], "dsr": fe.dsr(x["sr_daily"], T, x["skew"], x["kurt"], V, N)})
    dtab = pd.DataFrame(dsr_rows)
    dtab.to_csv(out / "dsr.csv", index=False)
    tm["dsr_N4"] = tm.apply(lambda q: dtab[(dtab["strategy"] == q["strategy"]) & dtab["primary"]]["dsr"].iloc[0]
                            if q["base_case"] else np.nan, axis=1)

    # ---- bootstrap of the Sharpe ratios (base-case costs)
    brows = []
    for s in STRATS:
        rng = np.random.default_rng([MASTER_SEED, STUDY_SEED, STRAT_IDS[s]])
        bs = fe.bootstrap_sharpe(net33[s], rng, B=N_BOOT, mean_block=BOOT_BLOCK)
        brows.append({"label": LABEL, "series": s, "sr_annual": sharpe_annual(net33[s]),
                      "lo_2_5": float(np.nanpercentile(bs, 2.5)), "hi_97_5": float(np.nanpercentile(bs, 97.5)),
                      "share_draws_le_0": float(np.nanmean(bs <= 0)), "B": N_BOOT, "mean_block": BOOT_BLOCK,
                      "seed": f"[{MASTER_SEED}, {STUDY_SEED}, {STRAT_IDS[s]}]"})
    rng = np.random.default_rng([MASTER_SEED, STUDY_SEED, 4])
    idx = fe.stationary_bootstrap_indices(T, N_BOOT, BOOT_BLOCK, rng)
    a_, b_ = net33["hybrid_base"][idx], net33["mcsgarch"][idx]
    diff = (a_.mean(1) / a_.std(1, ddof=1) - b_.mean(1) / b_.std(1, ddof=1)) * math.sqrt(DAYS_YEAR)
    brows.append({"label": LABEL, "series": "hybrid_base minus mcsgarch (paired)",
                  "sr_annual": sharpe_annual(net33["hybrid_base"]) - sharpe_annual(net33["mcsgarch"]),
                  "lo_2_5": float(np.percentile(diff, 2.5)), "hi_97_5": float(np.percentile(diff, 97.5)),
                  "share_draws_le_0": float(np.mean(diff <= 0)), "B": N_BOOT, "mean_block": BOOT_BLOCK,
                  "seed": f"[{MASTER_SEED}, {STUDY_SEED}, 4]"})
    bt = pd.DataFrame(brows)
    bt.to_csv(out / "bootstrap_sharpe.csv", index=False)
    tm = tm.merge(bt[bt["series"].isin(STRATS)][["series", "lo_2_5", "hi_97_5"]].rename(
        columns={"series": "strategy", "lo_2_5": "boot_sr_lo", "hi_97_5": "boot_sr_hi"}), on="strategy", how="left")
    tm.loc[~tm["base_case"], ["boot_sr_lo", "boot_sr_hi"]] = np.nan
    tm.to_csv(out / "trading_metrics.csv", index=False)

    # 33 sizing rules (DSR sensitivity), base case; the last two columns are descriptive, added in phase 2
    def smooth(m):
        if m == "constant":
            return 0.0
        return float(np.abs(np.diff(np.log(pivot(sel30, m, days, o30)), axis=1)).mean())
    pd.DataFrame({"label": LABEL, "rule": list(W), "sr_annual_net_base": sr33 * math.sqrt(DAYS_YEAR),
                  "sr_annual_gross": [sharpe_annual(gross[m]) for m in W],
                  "turnover_per_day": [float(turn[m][0].mean()) for m in W],
                  "rebalancing_turnover_per_day": [float((turn[m][0] - turn[m][1]).mean()) for m in W],
                  "mean_abs_change_log_forecast": [smooth(m) for m in W],
                  "rmsld": [float(math.sqrt(np.nanmean(tracking_loss(W[m], Y)))) for m in W],
                  "gross_utility_pct_year": [float(net_utility_daily(W[m], Y, np.zeros(len(days))).mean() * DAYS_YEAR * 100)
                                             for m in W],
                  "net_utility_pct_year_base": [float(net_utility_daily(W[m], Y, daily_cost_bp(turn[m][0], rt[SLIP_BASE])).mean()
                                                      * DAYS_YEAR * 100) for m in W]}).to_csv(
        out / "sizing_rules_33.csv", index=False)

    # descriptive, added in phase 2 (not in the plan): where the gross result comes from
    crow = []
    for s in STRATS:
        for k, m in enumerate(o30):
            crow.append({"label": LABEL, "strategy": s, "window_origin_minute": m, "mean_position": float(W[s][:, k].mean()),
                         "share_at_cap": float((W[s][:, k] >= CAP - 1e-12).mean()),
                         "mean_index_return_bp": float(Rw[:, k].mean() * 1e4),
                         "mean_gross_contribution_bp_day": float((W[s][:, k] * Rw[:, k]).mean() * 1e4)})
    wc = pd.DataFrame(crow)
    wc.to_csv(out / "window_contributions.csv", index=False)
    drift = index_drift(grid, days)
    drift.to_csv(out / "index_drift.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        say("")
        say(f"== Part B: vol-targeted long Nifty, 12 windows a day, target {TARGET:.0%}, cap {CAP:g}, base-case costs")
        cols = ["strategy", "mean_gross_bp_day", "mean_cost_bp_day", "mean_net_bp_day", "sr_annual", "boot_sr_lo",
                "boot_sr_hi", "hac_t", "p_one_sided", "turnover_per_day", "turnover_rebalancing_per_day", "cost_pct_year",
                "mean_position", "share_windows_at_cap", "realized_vol_rv_pct", "realized_vol_returns_pct", "rmsld",
                "share_days_in_band", "net_utility_pct_year", "net_utility_realized_returns_pct_year", "mdd", "psr0", "dsr_N4"]
        say(tm[tm["base_case"]][cols].to_string(index=False, float_format=fmt))
        say("")
        say("== cost grid: annualised Sharpe ratio and net utility")
        say(tm[["strategy", "costs", "mean_net_bp_day", "sr_annual", "net_utility_pct_year"]].to_string(index=False, float_format=fmt))
        say("")
        say("== trading tests")
        say(tt.drop(columns=["label"]).to_string(index=False, float_format=fmt))
        say("")
        say("== Deflated Sharpe Ratio")
        say(dtab.drop(columns=["label"]).to_string(index=False, float_format=fmt))
        say("")
        say("== bootstrap of the Sharpe ratio")
        say(bt.drop(columns=["label"]).to_string(index=False, float_format=fmt))
        say("")
        say("== descriptive (added in phase 2): Nifty's drift by part of the day, and gross contribution by window")
        say(drift.drop(columns=["label"]).to_string(index=False, float_format=fmt))
        say(wc.pivot(index="window_origin_minute", columns="strategy", values="mean_gross_contribution_bp_day")
            .join(wc[wc["strategy"] == "constant"].set_index("window_origin_minute")["mean_index_return_bp"])
            .to_string(float_format=fmt))
        say(pd.read_csv(out / "sizing_rules_33.csv").drop(columns=["label"]).sort_values("turnover_per_day")
            .to_string(index=False, float_format=fmt))

    # =======================================================================
    # Verdicts (Section 5 of the plan)
    # =======================================================================
    h1_ok = bool(h1["p_one_sided"] < 0.05)
    fee = float(h1["fee_pct_year"])
    lab = size_label(fee)
    t1 = tt[tt["test_id"] == "T1"].iloc[0]
    t2 = tt[tt["test_id"] == "T2"].iloc[0]
    if t1["p_one_sided"] < 0.05:
        part_b = "survives implementation" if t2["p_one_sided"] < 0.05 else "better risk control, not worth its trading costs"
    else:
        part_b = "no implementation gain"
    dsr_h = float(dtab[(dtab["strategy"] == "hybrid_base") & dtab["primary"]]["dsr"].iloc[0])
    standalone = "a standalone economically meaningful strategy" if dsr_h > DSR_THRESHOLD else "not a standalone strategy"
    if not h1_ok:
        overall = "no economic value shown"
    elif lab != "tiny" and part_b == "survives implementation":
        overall = "economically meaningful"
    else:
        why = []
        if lab == "tiny":
            why.append("the fee is tiny")
        if part_b != "survives implementation":
            why.append(f"Part B: {part_b}")
        overall = "detectable but not economically meaningful (" + "; ".join(why) + ")"
    s4 = ut[ut["test_id"] == "S4"].iloc[0]
    secondary = {r["test_id"]: bool(r["p_holm_secondary"] < 0.05) for _, r in ut[sec].iterrows()}
    verdict = pd.DataFrame([
        {"item": "H1 (30 minutes, hybrid_base vs mcsgarch, realized utility)", "statistic": h1["p_one_sided"],
         "result": "supported" if h1_ok else "not supported"},
        {"item": "fee at SR 0.4, gamma 2, percent of wealth a year", "statistic": fee, "result": lab},
        {"item": "fee in rupees a year per Rs 1 crore", "statistic": fee / 100 * RS_PER_CRORE, "result": lab},
        *[{"item": f"{k} (Holm)", "statistic": float(ut.loc[ut['test_id'] == k, 'p_holm_secondary'].iloc[0]),
           "result": "supported" if v else "not supported"} for k, v in secondary.items()],
        {"item": "T1 tracking, one-sided p", "statistic": t1["p_one_sided"], "result": "pass" if t1["p_one_sided"] < 0.05 else "fail"},
        {"item": "T2 net utility at base-case costs, one-sided p", "statistic": t2["p_one_sided"],
         "result": "pass" if t2["p_one_sided"] < 0.05 else "fail"},
        {"item": "Part B verdict", "statistic": np.nan, "result": part_b},
        {"item": "standalone DSR of the hybrid rule (N = 4)", "statistic": dsr_h, "result": standalone},
        {"item": "S4 information-matched check", "statistic": float(s4["p_holm_secondary"]),
         "result": "passes" if secondary["S4"] else "fails: value over mcsGARCH not shown to come from the method"},
        {"item": "OVERALL", "statistic": np.nan, "result": overall}])
    verdict.insert(0, "label", LABEL)
    verdict.to_csv(out / "verdict.csv", index=False)
    say("")
    say("== REGISTERED VERDICTS")
    say(verdict.drop(columns=["label"]).to_string(index=False, float_format=fmt))

    info = {"study": "P103 economic value of the forecasting hybrid", "label": LABEL, "prereg_file": PREREG_FILE,
            "prereg_commit": PREREG_COMMIT, "overall_verdict": overall, "part_b_verdict": part_b,
            "h1_p": float(h1["p_one_sided"]), "fee_pct_year": fee, "size_label": lab, "dsr_hybrid_N4": dsr_h,
            "parameters": {"sr_annual": SR_A, "gamma": GAMMA, "target_vol": TARGET, "cap": CAP, "constant_position": CONST_W,
                           "windows_per_day": WIN_PER_DAY, "windows_per_year": N_WIN_YEAR, "origins": ORIGINS,
                           "slippage_base_bp_per_side": SLIP_BASE, "slippage_grid": [s for s in SLIP_GRID if s is not None]},
            "seeds": {s: [MASTER_SEED, STUDY_SEED, i] for s, i in STRAT_IDS.items()} | {"paired_difference": [MASTER_SEED, STUDY_SEED, 4]},
            "bootstrap": {"B": N_BOOT, "mean_block": BOOT_BLOCK, "method": "stationary (Politis and Romano 1994), percentile"},
            "dsr": {"N_primary": N_PRIMARY, "V_primary": V4, "V_33": V33, "T": T},
            "days": {"n": int(T), "first": str(days.min().date()), "last": str(days.max().date())},
            "input_sha256_16": {str(p): fe.sha16(p) for p in [run / f"forecasts_{t}.csv" for t in fc]
                                + [run / f"scoreboard_{t}.csv" for t in fc] + [raw / "nifty_1min.csv"]},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__},
            "runtime_seconds": round(time.time() - t0, 1)}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    (out / "stdout.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
