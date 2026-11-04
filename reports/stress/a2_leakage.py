"""A2. Walk-forward leakage probe, label-shuffle placebo and day-permutation negative control on the repo's own code path
(niftygex.simulate -> features.build_dataset -> forecast.run_walkforward) with SYNTHETIC data. Seeds: market 9_700_001 (base) and
9_700_002 (alternative future); label shuffles 9_710_000+k; day permutations 9_720_000+k.
Usage: python a2_leakage.py <out.csv> [n_days] [n_shuffle]"""
import sys, time
import numpy as np, pandas as pd
from niftygex.simulate import SimConfig, simulate_market
from niftygex.features import build_dataset, build_gex_panel
from niftygex.forecast import WalkForwardConfig, run_walkforward
from niftygex.evaluation import qlike, diebold_mariano

OUT = sys.argv[1]; NDAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 300; NSHUF = int(sys.argv[3]) if len(sys.argv) > 3 else 10
MODELS = ["naive_last", "har", "mcsgarch", "linear_base", "lgbm_base", "hybrid_base"]
CFG = WalkForwardConfig(min_train=100, block=21, models=MODELS)
rows = []
import niftygex.volatility as V, niftygex.forecast as FC
_orig_dev = V.daily_egarch_variance
def causal_daily_egarch_variance(daily_returns, train_size=None, dist="t", scale=100.0):
    """Same as the repo function, but the sanity check runs on the filter of y[:n] only and the EWMA floor uses only y[:n]. PROBE-ONLY patch (it removes the selection flip; it does not make arch's full-series filter causal)."""
    from arch import arch_model
    y = np.asarray(daily_returns, float) * scale; n = len(y) if train_size is None else int(train_size)
    for d in (dist, "normal"):
        try:
            am = arch_model(y[:n], mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist=d, rescale=False); res = am.fit(disp="off", show_warning=False)
            full = arch_model(y, mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist=d, rescale=False)
            h = np.asarray(full.fix(res.params.values).conditional_volatility, float) ** 2
            if np.isnan(h[0]): h[0] = h[1]
            hc = np.asarray(arch_model(y[:n], mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist=d, rescale=False).fix(res.params.values).conditional_volatility, float) ** 2
            if np.isnan(hc[0]): hc[0] = hc[1]
            if V._egarch_ok(res.params, hc, y[:n]): return h / scale**2, {"method": f"egarch_{d}", "params": res.params}
        except Exception: pass
    ms = float(np.mean(y[:n] ** 2)) or 1e-20
    h = np.empty_like(y); h[0] = float(np.mean(y[:n] ** 2)) or ms
    for t in range(1, len(y)): h[t] = 0.94 * h[t - 1] + 0.06 * y[t - 1] ** 2
    return np.maximum(h, 0.01 * ms) / scale**2, {"method": "ewma", "params": None}
def set_patch(on):
    f = causal_daily_egarch_variance if on else _orig_dev
    V.daily_egarch_variance = f; FC.daily_egarch_variance = f
def add(test, item, value, status=""): rows.append(dict(test=test, item=item, value=value, status=status)); print(test, "|", item, "|", value, status, flush=True)

def pipeline(m, shuffle_seed=None):
    panel = build_gex_panel(m)
    ds = build_dataset(m, panel, horizon=30, step=45)
    if shuffle_seed is not None:  # permute the target among days within each anchor minute (keeps the diurnal pattern, breaks day linkage)
        rng = np.random.default_rng(shuffle_seed); ds = ds.copy()
        for mi, idx in ds.groupby("minute").groups.items():
            idx = np.asarray(idx); ds.loc[idx, "y"] = rng.permutation(ds.loc[idx, "y"].to_numpy())
    pred, info = run_walkforward(m, ds, CFG)
    return ds, pred

def gain(pred, a="hybrid_base", b="mcsgarch"):
    la, lb = qlike(pred.y, pred[a]), qlike(pred.y, pred[b])
    day = pred["date"].to_numpy()
    da, db = pd.Series(la).groupby(day).mean().to_numpy(), pd.Series(lb).groupby(day).mean().to_numpy()
    dm = diebold_mariano(db, da, alternative="greater")  # one-sided: is a better than b
    return 100 * (1 - la.mean() / lb.mean()), dm["stat"], dm["p"]

def splice(m1, m2, cut):
    T = len(m1["days"]); m = dict(m1)
    for k, v in m1.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            m[k] = np.concatenate([v[:cut], m2[k][cut:]], axis=0)
        elif isinstance(v, pd.DataFrame) and len(v) == T and k != "labels":
            m[k] = pd.concat([v.iloc[:cut], m2[k].iloc[cut:]]).reset_index(drop=True) if not isinstance(v.index, pd.DatetimeIndex) else pd.concat([v.iloc[:cut], m2[k].iloc[cut:]])
        elif isinstance(v, list) and len(v) == T:
            m[k] = v[:cut] + m2[k][cut:]
    return m

def permute_days(m, seed):
    T = len(m["days"]); p = np.random.default_rng(seed).permutation(T); out = dict(m)
    for k, v in m.items():
        if k == "labels": continue
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T: out[k] = v[p]
        elif isinstance(v, pd.DataFrame) and len(v) == T: out[k] = v.iloc[p].set_axis(v.index)
        elif isinstance(v, list) and len(v) == T: out[k] = [v[i] for i in p]
    return out

t0 = time.time()
m1 = simulate_market(SimConfig(n_days=NDAYS, kappa=0.0, seed=9_700_001))
m2 = simulate_market(SimConfig(n_days=NDAYS, kappa=0.0, seed=9_700_002))
ds1, p1 = pipeline(m1)
add("baseline", "seconds for one full pipeline run", round(time.time() - t0, 1))
g, s, p = gain(p1); add("baseline", "hybrid_base gain over mcsgarch, pct / DM stat / one-sided p (synthetic, real code path)", f"{g:.2f} / {s:.2f} / {p:.3g}")
for mname in ["har", "linear_base", "lgbm_base"]:
    g_, s_, p_ = gain(p1, mname); add("baseline", f"{mname} gain over mcsgarch, pct / stat / p", f"{g_:.2f} / {s_:.2f} / {p_:.3g}")
# ---- (a) alter all data after the cut and compare everything before it (repo code, then with the proposed causal selection)
cut = 170
m_alt = splice(m1, m2, cut)
def alter_probe(tag):
    ds_a, p_a = pipeline(m1); ds_b, p_b = pipeline(m_alt)
    a, b = ds_a[ds_a.t < cut].reset_index(drop=True), ds_b[ds_b.t < cut].reset_index(drop=True)
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    bad = [c for c in num if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-10, atol=1e-14, equal_nan=True)]
    add(tag, f"feature columns compared for rows with t < {cut} / columns that differ", f"{len(num)} / {bad if bad else 'none'}", "VERIFIED" if not bad and len(a) == len(b) else "FAILED")
    pa = p_a[p_a.t < cut].sort_values(["t", "minute"]).reset_index(drop=True); pb = p_b[p_b.t < cut].sort_values(["t", "minute"]).reset_index(drop=True)
    diffs = {mn: float(np.max(np.abs(pa[mn].to_numpy() - pb[mn].to_numpy()) / pa[mn].to_numpy())) for mn in MODELS}
    first = {mn: int(pa.t[np.abs(pa[mn] - pb[mn]) / pa[mn] > 1e-9].min()) if (np.abs(pa[mn] - pb[mn]) / pa[mn] > 1e-9).any() else None for mn in MODELS}
    add(tag, f"forecast rows with t < {cut} compared / max relative forecast difference per model", f"{len(pa)} / " + str({k: f'{v:.1e}' for k, v in diffs.items()}), "VERIFIED" if max(diffs.values()) < 1e-8 else "FAILED")
    add(tag, "first test day whose forecast differs, per model", str(first))
    pa2 = p_a[p_a.t >= cut + 30].sort_values(["t", "minute"]).reset_index(drop=True); pb2 = p_b[p_b.t >= cut + 30].sort_values(["t", "minute"]).reset_index(drop=True)
    add(tag, "positive control: forecasts after the altered data DO change (max rel diff mcsgarch / hybrid_base)", f"{np.max(np.abs(pa2.mcsgarch - pb2.mcsgarch) / pa2.mcsgarch):.2f} / {np.max(np.abs(pa2.hybrid_base - pb2.hybrid_base) / pa2.hybrid_base):.2f}")
add("alter-after-cut", "share of the future part that was replaced (max relative diff of R after the cut)", float(np.max(np.abs(m_alt["R"][cut:] - m1["R"][cut:])) / np.max(np.abs(m1["R"]))))
set_patch(False); alter_probe("alter-after-cut, repo code as committed")
set_patch(True); alter_probe("alter-after-cut, with proposed causal selection (probe-only patch)")
set_patch(False)
# ---- (b) placebos. B1: labels shuffled for training AND evaluation (design check). B2: labels shuffled for training only, scored against the REAL labels.
key = ds1.set_index(["t", "minute"]).y
gs, g2 = [], []
for k in range(NSHUF):
    dss, ps = None, None
    ds_s, ps = pipeline(m1, shuffle_seed=9_710_000 + k)
    g_, s_, p_ = gain(ps); gs.append(g_)
    real_y = key.reindex(pd.MultiIndex.from_arrays([ps.t, ps.minute])).to_numpy()
    lh, lm = qlike(real_y, ps.hybrid_base), qlike(real_y, ps.mcsgarch); g2.append(100 * (1 - lh.mean() / lm.mean()))
add("label-shuffle placebo B1 (train and test labels shuffled)", "hybrid_base gain over mcsgarch pct, per shuffle", [round(x, 1) for x in gs])
add("label-shuffle placebo B2 (train labels shuffled, scored on real labels)", "hybrid_base gain over mcsgarch pct, per shuffle", [round(x, 1) for x in g2])
add("label-shuffle placebo B2 (train labels shuffled, scored on real labels)", "mean over shuffles vs real gain", f"{np.mean(g2):.2f} vs {g:.2f}", "VERIFIED" if np.mean(g2) < 0.5 * g else "FAILED")
# ---- (c) negative control: permute the order of days
gp = []
for k in range(5):
    mp = permute_days(m1, 9_720_000 + k)
    _, pp = pipeline(mp)
    g_, s_, p_ = gain(pp); gp.append(g_)
    add("day-permutation control", f"permutation {k}: hybrid_base gain over mcsgarch pct / stat / p", f"{g_:.2f} / {s_:.2f} / {p_:.3g}")
    for mname in ["har"]:
        g2, s2, p2 = gain(pp, mname); add("day-permutation control", f"permutation {k}: har gain over mcsgarch pct", f"{g2:.2f}")
add("day-permutation control", "mean gain over permutations vs real gain", f"{np.mean(gp):.2f} vs {g:.2f}")
pd.DataFrame(rows).to_csv(OUT, index=False)
print("EXIT total seconds", round(time.time() - t0))
