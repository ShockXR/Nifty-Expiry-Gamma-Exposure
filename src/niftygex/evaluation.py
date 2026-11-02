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
from scipy.stats import t as student_t


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
    Uses the Harvey, Leybourne and Newbold (1997) small-sample correction and,
    as they recommend, a Student t distribution with n - 1 degrees of freedom.
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
    df = max(n - 1, 1)
    if alternative == "greater":
        p = student_t.sf(stat, df)
    elif alternative == "less":
        p = student_t.cdf(stat, df)
    else:
        p = 2 * student_t.sf(abs(stat), df)
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


def trimmed_mean(x, trim: float = 0.01) -> float:
    """Mean after removing the largest ``trim`` share of values (upper tail only).

    QLIKE is bounded below by zero, so only the upper tail is trimmed. Each
    model's own largest losses are removed.
    """
    x = np.sort(np.asarray(x, float))
    k = int(np.floor(trim * x.size))
    return float(np.mean(x[: x.size - k])) if x.size - k > 0 else np.nan


def scoreboard(pred: pd.DataFrame, models: list[str], groups: dict | None = None, trim: float = 0.01) -> pd.DataFrame:
    """Mean QLIKE and MSE for every model, overall and by subgroup.

    Also the trimmed-mean QLIKE (each model's largest ``trim`` share of anchor
    losses removed) as a robustness check against a few huge losses. When
    mcsGARCH is present, every loss is also shown relative to it in percent.
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
            ql = qlike(sub["y"], sub[m])
            rows.append({"group": g, "model": m, "n": len(sub), "days": int(sub["date"].nunique()) if "date" in sub else np.nan,
                         "qlike": float(np.mean(ql)), "mse": float(np.mean(mse(sub["y"], sub[m]))),
                         "qlike_trim": trimmed_mean(ql, trim)})
    sb = pd.DataFrame(rows)
    if "mcsgarch" in models and len(sb):
        for col in ("qlike", "mse", "qlike_trim"):
            ref = sb[sb.model == "mcsgarch"].set_index("group")[col]
            sb[f"{col}_vs_mcsgarch_pct"] = 100 * (sb[col] / sb["group"].map(ref) - 1)
    return sb


def gain_by_anchor(pred: pd.DataFrame, models: list[str], ref: str = "mcsgarch", masks: dict | None = None) -> pd.DataFrame:
    """QLIKE and MSE of every model relative to ``ref`` at each anchor minute."""
    masks = masks or {"all": np.ones(len(pred), bool)}
    rows = []
    for g, mask in masks.items():
        sub = pred[mask]
        for minute, s in sub.groupby("minute"):
            lr, mr = np.mean(qlike(s["y"], s[ref])), np.mean(mse(s["y"], s[ref]))
            for m in models:
                rows.append({"group": g, "minute": int(minute), "model": m, "n": len(s),
                             "qlike_vs_ref_pct": 100 * (np.mean(qlike(s["y"], s[m])) / lr - 1),
                             "mse_vs_ref_pct": 100 * (np.mean(mse(s["y"], s[m])) / mr - 1)})
    return pd.DataFrame(rows)


REGIMES = ["multi_weekly", "single_weekly_thu", "single_weekly_tue"]


def subgroups(pred: pd.DataFrame, holdout_start=None, auction_start=None) -> dict:
    """Boolean masks for the standard breakdowns.

    all, expiry, non_expiry, day_before_expiry; each regime (codes 0, 1, 2 as
    in features.build_dataset) and regime x expiry; and, if ``holdout_start``
    is given, the development and hold-out periods (each also by expiry); and,
    if ``auction_start`` is given, the days before and after NSE's closing
    auction began (each also by expiry).
    """
    e = pred["is_expiry"].to_numpy(bool)
    g = {"all": np.ones(len(pred), bool), "expiry": e, "non_expiry": ~e}
    if "minute" in pred:
        op = pred["minute"].to_numpy() == 0
        if op.any() and (~op).any():   # the 09:15 anchor carries the overnight news
            g["open_0915"] = op
            g["excl_0915"] = ~op
            g["excl_0915_expiry"] = ~op & e
    if "tdte" in pred:
        g["day_before_expiry"] = pred["tdte"].to_numpy() == 1
    if "regime" in pred:
        rc = pred["regime"].to_numpy()
        for k, name in enumerate(REGIMES):
            if (rc == k).any():
                g[f"{name}"] = rc == k
                g[f"{name}_expiry"] = (rc == k) & e
                g[f"{name}_non_expiry"] = (rc == k) & ~e
    dates = pd.to_datetime(pred["date"]).to_numpy()
    if auction_start is not None:
        pre = dates < np.datetime64(pd.Timestamp(auction_start))
        if pre.any() and (~pre).any():
            g["pre_auction"] = pre
            g["pre_auction_expiry"] = pre & e
            g["pre_auction_non_expiry"] = pre & ~e
            g["auction_period"] = ~pre
            g["auction_period_expiry"] = ~pre & e
            if "excl_0915" in g:
                g["pre_auction_excl_0915"] = pre & g["excl_0915"]
    if holdout_start is not None:
        ho = dates >= np.datetime64(pd.Timestamp(holdout_start))
        for name, m in (("development", ~ho), ("holdout", ho)):
            if m.any():
                g[name] = m
                g[f"{name}_expiry"] = m & e
        if auction_start is not None and ho.any():
            hp = ho & (dates < np.datetime64(pd.Timestamp(auction_start)))
            if hp.any() and hp.sum() < ho.sum():
                g["holdout_pre_auction"] = hp
                g["holdout_pre_auction_expiry"] = hp & e
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
