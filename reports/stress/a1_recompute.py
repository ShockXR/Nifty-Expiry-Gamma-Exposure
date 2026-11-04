"""A1. Independent recomputation of the headline from the COMMITTED forecast files. Does not import niftygex.
Usage: python -I a1_recompute.py <repo_root> <out_dir>.   Bootstrap seed 9_600_001 (stationary bootstrap, mean block 10 days, 5000 draws)."""
import sys, json
import numpy as np, pandas as pd
from scipy import stats

root, out = sys.argv[1], sys.argv[2]
R = root + "/results/pipeline_nse_v2/"; P = root + "/results/p116_benchmark_as_printed/"
HZ = {"next15": 15, "next30": 30, "last_hour": 60}
rows, rob = [], []

def ql(y, f):
    y = np.maximum(np.asarray(y, float), 1e-16); f = np.maximum(np.asarray(f, float), 1e-16)
    return y / f - np.log(y / f) - 1

def nw_dm(d, lag=None, hln=True):
    """DM statistic of mean(d)=0 with Bartlett (Newey-West) variance; HLN small-sample factor and t(n-1) p-value (two-sided, one-sided)."""
    d = np.asarray(d, float); n = d.size
    if lag is None: lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dc = d - d.mean(); lrv = dc @ dc / n
    for k in range(1, lag + 1): lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    if hln: stat *= np.sqrt((n - 1) / n)  # h=1: (n+1-2h+h(h-1)/n)/n = (n-1)/n
    return stat, stats.t.sf(stat, n - 1), 2 * stats.t.sf(abs(stat), n - 1), lag

def stat_boot(lh, lm, B=5000, seed=9_600_001, mean_block=10):
    rng = np.random.default_rng(seed); n = len(lh); p = 1 / mean_block; g = np.empty(B)
    for b in range(B):
        idx = np.empty(n, int); i = 0; cur = rng.integers(n)
        for i in range(n):
            if i > 0 and rng.random() >= p: cur = (cur + 1) % n
            elif i > 0: cur = rng.integers(n)
            idx[i] = cur
        g[b] = 100 * (1 - lh[idx].mean() / lm[idx].mean())
    return np.quantile(g, [0.025, 0.975])

def gain(lh, lm): return 100 * (1 - np.mean(lh) / np.mean(lm))

def add(**kw): rows.append(kw)

for tgt, h in HZ.items():
    F = pd.read_csv(R + f"forecasts_{tgt}.csv"); sb = pd.read_csv(R + f"scoreboard_{tgt}.csv"); sb = sb[sb.group == "all"].set_index("model")
    # ---------------- integrity
    add(item="rows", horizon=tgt, value=len(F)); add(item="distinct days", horizon=tgt, value=F.date.nunique())
    add(item="duplicate (date, minute) rows", horizon=tgt, value=int(F.duplicated(["date", "minute"]).sum()))
    dts = pd.to_datetime(F.date); add(item="dates nondecreasing in file order", horizon=tgt, value=bool(dts.is_monotonic_increasing))
    add(item="t nondecreasing in file order", horizon=tgt, value=bool(F.t.is_monotonic_increasing))
    add(item="minutes increasing within each day", horizon=tgt, value=bool(F.groupby("date").minute.apply(lambda s: s.is_monotonic_increasing).all()))
    add(item="max(minute + horizon) (session is 375 minutes)", horizon=tgt, value=int((F.minute + F.horizon).max()))
    add(item="first and last test day", horizon=tgt, value=f"{F.date.min()} to {F.date.max()}")
    models = [c for c in F.columns if c not in ("t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y")]
    add(item="NaN forecasts (all models)", horizon=tgt, value=int(F[models].isna().sum().sum()))
    add(item="non-positive forecasts (all models)", horizon=tgt, value=int((F[models] <= 0).sum().sum()))
    add(item="non-positive or NaN realized y", horizon=tgt, value=int(((F.y <= 0) | F.y.isna()).sum()))
    # ---------------- QLIKE per model rebuilt, versus the committed scoreboard
    Q = {m: ql(F.y, F[m]) for m in models}
    worst = max(abs(Q[m].mean() - sb.loc[m, "qlike"]) for m in models)
    add(item="max |my mean QLIKE - committed scoreboard| over all models", horizon=tgt, value=worst, reference=0.0, status="VERIFIED" if worst < 1e-9 else "FAILED")
    # ---------------- headline: hybrid_base vs mcsgarch
    lh, lm = Q["hybrid_base"], Q["mcsgarch"]
    g = gain(lh, lm); add(item="gain of hybrid_base over mcsgarch (our implementation), pct", horizon=tgt, value=g,
                          reference=-sb.loc["hybrid_base", "qlike_vs_mcsgarch_pct"], status="VERIFIED" if abs(g + sb.loc["hybrid_base", "qlike_vs_mcsgarch_pct"]) < 1e-6 else "FAILED")
    dayh = pd.Series(lh).groupby(F.date.values).mean(); daym = pd.Series(lm).groupby(F.date.values).mean()
    d = (daym - dayh).to_numpy()
    for lag in (None, 5, 10, 21):
        s, p1, p2, lg = nw_dm(d, lag)
        add(item=f"DM hybrid_base vs mcsgarch day level, NW lag {lg}: stat / one-sided p", horizon=tgt, value=f"{s:.3f} / {p1:.2e}")
    # naive iid t-test and the share of days won
    add(item="share of days hybrid_base beats mcsgarch", horizon=tgt, value=float((d > 0).mean()))
    add(item="sign test p (two-sided)", horizon=tgt, value=float(stats.binomtest(int((d > 0).sum()), len(d), 0.5).pvalue))
    ci = stat_boot(dayh.to_numpy(), daym.to_numpy()); add(item="stationary-bootstrap 95% CI of the gain, pct", horizon=tgt, value=f"{ci[0]:.2f} to {ci[1]:.2f}")
    # ---------------- other baselines
    for b in ["naive_last", "naive_prevday", "har", "vix", "implied", "mcsgarch", "mcsgarch_iv", "lgbm_base", "linear_base", "hybrid_v1"]:
        lb = pd.Series(Q[b]).groupby(F.date.values).mean().to_numpy(); s, p1, _, _ = nw_dm(lb - dayh.to_numpy())
        add(item=f"hybrid_base gain over {b}, pct / DM one-sided p", horizon=tgt, value=f"{gain(Q['hybrid_base'], Q[b]):.2f} / {p1:.1e}")
    # ---------------- P116 paper-specified benchmark (day level files only)
    dl = pd.read_csv(P + f"daily_losses_nifty_{tgt}.csv").set_index("date")
    add(item="P116 daily file: days equal forecast file days", horizon=tgt, value=bool(set(dl.index) == set(F.date.unique())))
    cmp_ = (dl["hybrid"] - dayh.reindex(dl.index)).abs().max(); add(item="max |P116 hybrid daily QLIKE - my daily QLIKE of hybrid_base|", horizon=tgt, value=cmp_, reference=0.0, status="VERIFIED" if cmp_ < 1e-9 else "FAILED")
    cmp2 = (dl["mcs_ours"] - daym.reindex(dl.index)).abs().max(); add(item="max |P116 mcs_ours daily QLIKE - my daily QLIKE of mcsgarch|", horizon=tgt, value=cmp2, reference=0.0, status="VERIFIED" if cmp2 < 1e-9 else "FAILED")
    gc = gain(dl["hybrid"].to_numpy(), dl["mcs_cpp"].to_numpy())
    add(item="gain of hybrid over mcs_cpp (paper-specified), pct", horizon=tgt, value=gc)
    dc = (dl["mcs_cpp"] - dl["hybrid"]).to_numpy(); s, p1, _, lg = nw_dm(dc)
    add(item="DM hybrid vs mcs_cpp day level: stat / one-sided p", horizon=tgt, value=f"{s:.3f} / {p1:.2e}")
    ci2 = stat_boot(dl["hybrid"].to_numpy(), dl["mcs_cpp"].to_numpy()); add(item="bootstrap 95% CI gain vs mcs_cpp, pct", horizon=tgt, value=f"{ci2[0]:.2f} to {ci2[1]:.2f}")
    add(item="mcs_ours versus mcs_cpp (negative = our benchmark is worse), pct", horizon=tgt, value=gain(dl["mcs_ours"].to_numpy(), dl["mcs_cpp"].to_numpy()))
    # ---------------- outlier and trimming robustness (observation level, hybrid_base vs mcsgarch)
    top = np.argsort(-(Q["mcsgarch"]))[:3]
    for i in top: add(item=f"largest mcsgarch QLIKE obs {F.date[i]} minute {F.minute[i]}", horizon=tgt, value=f"mcsgarch {Q['mcsgarch'][i]:.1f}, hybrid {Q['hybrid_base'][i]:.1f}, y {F.y[i]:.2e}, f_mcs {F.mcsgarch[i]:.2e}, f_hyb {F.hybrid_base[i]:.2e}")
    topd = np.argsort(-np.abs(Q["mcsgarch"] - Q["hybrid_base"]))[:1]
    for i in topd: add(item="largest |loss difference| obs", horizon=tgt, value=f"{F.date[i]} minute {F.minute[i]} diff {Q['mcsgarch'][i]-Q['hybrid_base'][i]:.2f}")
    # MSE view: the 09:15 forecast of 5 June 2024 (mcsgarch is about 20 times too high that morning)
    se = {m: (F.y.to_numpy() - F[m].to_numpy()) ** 2 for m in ["mcsgarch", "hybrid_base", "har", "vix", "lgbm_base"]}
    mk = ((F.date == "2024-06-05") & (F.minute == 0)).to_numpy()
    for m_, v in se.items():
        add(item=f"MSE {m_}: all obs / without 2024-06-05 09:15 / share of total squared error from that obs", horizon=tgt,
            value=f"{v.mean():.3e} / {v[~mk].mean():.3e} / {v[mk].sum() / v.sum():.3f}")
    add(item="QLIKE of that obs: mcsgarch / hybrid_base (QLIKE barely penalises an over-forecast)", horizon=tgt, value=f"{Q['mcsgarch'][mk].sum():.2f} / {Q['hybrid_base'][mk].sum():.2f}")
    add(item="MSE gain of hybrid_base over mcsgarch, pct (all / without that obs)", horizon=tgt, value=f"{100*(1-se['hybrid_base'].mean()/se['mcsgarch'].mean()):.1f} / {100*(1-se['hybrid_base'][~mk].mean()/se['mcsgarch'][~mk].mean()):.1f}")
    add(item="MSE rank: hybrid_base vs har (all / without that obs), ratio of MSEs", horizon=tgt, value=f"{se['hybrid_base'].mean()/se['har'].mean():.2f} / {se['hybrid_base'][~mk].mean()/se['har'][~mk].mean():.2f}")
    mask = ~((F.date == "2024-06-05") & (F.minute == 0)).to_numpy()
    rob.append(dict(horizon=tgt, scheme="drop only the 2024-06-05 09:15 observation", n=int(mask.sum()), gain_pct=gain(lh[mask], lm[mask])))
    m2 = (F.date != "2024-06-05").to_numpy(); rob.append(dict(horizon=tgt, scheme="drop the whole day 2024-06-05", n=int(m2.sum()), gain_pct=gain(lh[m2], lm[m2])))
    rob.append(dict(horizon=tgt, scheme="all observations", n=len(F), gain_pct=g))
    for pct in (1, 5):
        k = int(np.ceil(len(F) * pct / 100))
        for name, key in [("joint: top by max(loss_hybrid, loss_mcs)", np.maximum(lh, lm)), ("top by |loss difference|", np.abs(lh - lm)), ("top by mcsgarch loss only", lm), ("top by hybrid loss only", lh)]:
            keep = np.ones(len(F), bool); keep[np.argsort(-key)[:k]] = False
            rob.append(dict(horizon=tgt, scheme=f"drop top {pct}% {name}", n=int(keep.sum()), gain_pct=gain(lh[keep], lm[keep])))
        srt = lambda x: np.sort(x)[: len(x) - k].mean()
        rob.append(dict(horizon=tgt, scheme=f"each model trimmed at its own top {pct}%", n=len(F) - k, gain_pct=100 * (1 - srt(lh) / srt(lm))))
    # ---------------- subperiods and rolling windows
    dates = pd.to_datetime(F.date); days = np.sort(F.date.unique())
    half = days[len(days) // 2]
    for name, msk in [("first half of test days", (F.date < half).to_numpy()), ("second half of test days", (F.date >= half).to_numpy()),
                      ("development (before 2025-07-01)", (dates < "2025-07-01").to_numpy()), ("holdout (from 2025-07-01)", (dates >= "2025-07-01").to_numpy())]:
        dd = (pd.Series(lm[msk]).groupby(F.date.values[msk]).mean() - pd.Series(lh[msk]).groupby(F.date.values[msk]).mean()).to_numpy()
        s, p1, _, _ = nw_dm(dd)
        rob.append(dict(horizon=tgt, scheme=name, n=int(msk.sum()), gain_pct=gain(lh[msk], lm[msk]), dm_stat=s, one_sided_p=p1, days=len(dd)))
    for yr in sorted(dates.dt.year.unique()):
        msk = (dates.dt.year == yr).to_numpy(); dd = (pd.Series(lm[msk]).groupby(F.date.values[msk]).mean() - pd.Series(lh[msk]).groupby(F.date.values[msk]).mean()).to_numpy()
        s, p1, _, _ = nw_dm(dd) if len(dd) > 30 else (np.nan, np.nan, 0, 0)
        rob.append(dict(horizon=tgt, scheme=f"calendar year {yr}", n=int(msk.sum()), gain_pct=gain(lh[msk], lm[msk]), dm_stat=s, one_sided_p=p1, days=len(dd)))
    W, S = 126, 63; rg = []
    for st in range(0, len(days) - W + 1, S):
        wd = days[st:st + W]; msk = F.date.isin(wd).to_numpy(); rg.append(gain(lh[msk], lm[msk]))
    rob.append(dict(horizon=tgt, scheme=f"rolling windows of {W} days step {S}: min / median / max gain, count positive", n=len(rg), gain_pct=np.median(rg),
                    note=f"min {min(rg):.2f} max {max(rg):.2f} positive {sum(x > 0 for x in rg)} of {len(rg)}"))
    # ---------------- closing auction (index frozen 15:16 to 15:28 from 2026-08-03; session minute 361 to 373)
    lastm = F.minute + F.horizon
    overlap = ((lastm > 361) & (dates >= "2026-08-03")).to_numpy()
    rob.append(dict(horizon=tgt, scheme="exclude windows that overlap the frozen index (days from 2026-08-03, window end > minute 361)", n=int((~overlap).sum()), gain_pct=gain(lh[~overlap], lm[~overlap]), note=f"{int(overlap.sum())} obs removed"))
    if overlap.sum() > 0:
        rob.append(dict(horizon=tgt, scheme="only windows that overlap the frozen index", n=int(overlap.sum()), gain_pct=gain(lh[overlap], lm[overlap])))
    pre = (dates < "2026-08-03").to_numpy(); rob.append(dict(horizon=tgt, scheme="only days before 2026-08-03", n=int(pre.sum()), gain_pct=gain(lh[pre], lm[pre])))
    # expiry vs non-expiry
    for nm, msk in [("expiry days", F.is_expiry.astype(bool).to_numpy()), ("non-expiry days", ~F.is_expiry.astype(bool).to_numpy())]:
        rob.append(dict(horizon=tgt, scheme=nm, n=int(msk.sum()), gain_pct=gain(lh[msk], lm[msk])))
    # excluding the 09:15 anchor
    if tgt != "last_hour":
        msk = (F.minute > 0).to_numpy(); rob.append(dict(horizon=tgt, scheme="exclude the 09:15 anchor", n=int(msk.sum()), gain_pct=gain(lh[msk], lm[msk])))
pd.DataFrame(rows).to_csv(out + "/a1_recompute_summary.csv", index=False)
pd.DataFrame(rob).to_csv(out + "/a1_robustness.csv", index=False)
# committed P116 tests for comparison
t = pd.read_csv(P + "tests.csv")[["test_id", "estimate", "statistic", "p_raw", "p_adjusted"]]
t.to_csv(out + "/a1_committed_p116_tests.csv", index=False)
print("done", len(rows), len(rob))
