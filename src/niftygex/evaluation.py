"""Forecast losses and tests: QLIKE, MSE, Diebold-Mariano, Model Confidence Set.

QLIKE(y, f) = y / f - log(y / f) - 1 and MSE(y, f) = (y - f)^2 are the two
losses that rank variance forecasts consistently even when the target is a
noisy realized-variance proxy (Patton 2011).

Inference is done on DAILY AVERAGE losses. Anchors within one day share the
same daily volatility shock, so treating them as independent would overstate
significance (the same issue as minute-level Wilcoxon tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def qlike(y, f):
    y = np.maximum(np.asarray(y, float), 1e-16)
    f = np.maximum(np.asarray(f, float), 1e-16)
    return y / f - np.log(y / f) - 1


def mse(y, f):
    return (np.asarray(y, float) - np.asarray(f, float)) ** 2


LOSSES = {"qlike": qlike, "mse": mse}


def daily_losses(pred: pd.DataFrame, models: list[str], loss: str = "qlike") -> pd.DataFrame:
    fn = LOSSES[loss]
    L = pd.DataFrame({m: fn(pred["y"], pred[m]) for m in models})
    L["date"] = pred["date"].to_numpy()
    return L.groupby("date").mean()


def diebold_mariano(l1, l2, lag: int | None = None, alternative: str = "two-sided") -> dict:
    """DM test on a loss differential d = l1 - l2 with a Newey-West variance.

    alternative "greater" tests whether model 2 has lower loss (d > 0).
    Uses the Harvey, Leybourne and Newbold (1997) small-sample correction.
    """
    d = np.asarray(l1, float) - np.asarray(l2, float)
    d = d[np.isfinite(d)]
    n = d.size
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        lrv += 2 * w * (dc[k:] @ dc[:-k]) / n
    se = np.sqrt(max(lrv, 1e-300) / n)
    stat = d.mean() / se
    h = 1
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    if alternative == "greater":
        p = 1 - norm.cdf(stat)
    elif alternative == "less":
        p = norm.cdf(stat)
    else:
        p = 2 * (1 - norm.cdf(abs(stat)))
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p": float(p), "n": int(n)}


def model_confidence_set(daily_loss: pd.DataFrame, size: float = 0.1, reps: int = 2000,
                         block: int = 5, seed: int = 0) -> pd.DataFrame:
    """Hansen, Lunde and Nason (2011) MCS on daily losses (arch implementation)."""
    from arch.bootstrap import MCS

    mcs = MCS(daily_loss.dropna(), size=size, reps=reps, block_size=block, method="R", seed=seed)
    mcs.compute()
    out = mcs.pvalues.copy()
    out["in_mcs"] = out.index.isin(mcs.included)
    return out


def scoreboard(pred: pd.DataFrame, models: list[str], groups: dict | None = None) -> pd.DataFrame:
    """Mean QLIKE and MSE for every model, overall and by subgroup.

    QLIKE is also shown relative to mcsGARCH when that model is present.
    """
    groups = groups or {
        "all": np.ones(len(pred), bool),
        "expiry": pred["is_expiry"].to_numpy(bool),
        "non_expiry": ~pred["is_expiry"].to_numpy(bool),
    }
    rows = []
    for g, mask in groups.items():
        sub = pred[mask]
        if sub.empty:
            continue
        for m in models:
            rows.append({"group": g, "model": m, "n": len(sub), "days": int(sub["date"].nunique()) if "date" in sub else np.nan,
                         "qlike": float(np.mean(qlike(sub["y"], sub[m]))),
                         "mse": float(np.mean(mse(sub["y"], sub[m])))})
    sb = pd.DataFrame(rows)
    if "mcsgarch" in models:
        ref = sb[sb.model == "mcsgarch"].set_index("group")["qlike"]
        sb["qlike_vs_mcsgarch_pct"] = 100 * (sb["qlike"] / sb["group"].map(ref) - 1)
    return sb


REGIMES = ["multi_weekly", "single_weekly_thu", "single_weekly_tue"]


def subgroups(pred: pd.DataFrame, holdout_start=None) -> dict:
    """Boolean masks for the standard breakdowns.

    all, expiry, non_expiry, day_before_expiry; each regime (codes 0, 1, 2 as
    in features.build_dataset) and regime x expiry; and, if ``holdout_start``
    is given, the development and hold-out periods (each also by expiry).
    """
    e = pred["is_expiry"].to_numpy(bool)
    g = {"all": np.ones(len(pred), bool), "expiry": e, "non_expiry": ~e}
    if "tdte" in pred:
        g["day_before_expiry"] = pred["tdte"].to_numpy() == 1
    if "regime" in pred:
        rc = pred["regime"].to_numpy()
        for k, name in enumerate(REGIMES):
            if (rc == k).any():
                g[f"{name}"] = rc == k
                g[f"{name}_expiry"] = (rc == k) & e
                g[f"{name}_non_expiry"] = (rc == k) & ~e
    if holdout_start is not None:
        dates = pd.to_datetime(pred["date"]).to_numpy()
        ho = dates >= np.datetime64(pd.Timestamp(holdout_start))
        for name, m in (("development", ~ho), ("holdout", ho)):
            if m.any():
                g[name] = m
                g[f"{name}_expiry"] = m & e
    return g


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = idx.size
    adj = np.maximum.accumulate([(m - r) * p[k] for r, k in enumerate(order)])
    out[order] = np.minimum(adj, 1.0)
    return out


def ablation_table(pred: pd.DataFrame, pairs: list[tuple], groups: dict, loss: str = "qlike") -> pd.DataFrame:
    """Clean with/without comparisons.

    ``pairs`` holds (family, label, base_model, alt_model). For every pair and
    group the table gives both mean losses, the percentage change from
    adding the features (negative = the extra features help) and a one-sided
    Diebold-Mariano test (HAC variance, Harvey-Leybourne-Newbold correction)
    on DAILY average losses that the alternative has lower loss.
    """
    rows = []
    fn = LOSSES[loss]
    for gname, mask in groups.items():
        sub = pred[mask]
        if sub["date"].nunique() < 10:
            continue
        models = sorted({m for _, _, b, a in pairs for m in (b, a) if m in sub})
        L = daily_losses(sub, models, loss)
        for fam, label, b, a in pairs:
            if b not in sub or a not in sub:
                continue
            lb, la = float(np.mean(fn(sub["y"], sub[b]))), float(np.mean(fn(sub["y"], sub[a])))
            dm = diebold_mariano(L[b], L[a], alternative="greater")
            rows.append({"family": fam, "label": label, "group": gname, "base": b, "alt": a,
                         "days": dm["n"], f"{loss}_base": lb, f"{loss}_alt": la,
                         "change_pct": 100 * (la / lb - 1), "dm_stat": dm["stat"], "p_one_sided": dm["p"]})
    return pd.DataFrame(rows)
