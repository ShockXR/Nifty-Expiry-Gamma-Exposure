"""Null-world size test (SYNTHETIC data, not a registered study, not in scripts/reproduce_all.py).

Question: does the walk-forward machinery manufacture a "gain over mcsGARCH" when mcsGARCH is the true model?

Each world draws intraday one-minute returns from an mcsGARCH data-generating process:
    r[t, i] = sqrt(h[t] * s[i] * q[t, i]) * z[t, i],   z ~ N(0, 1)
* h[t]: daily variance from an EGARCH(1,1,1) with Gaussian errors, driven by the day's standardized
  open-to-close return. Parameters fitted (arch, zero mean) on the committed real Nifty open-to-close
  returns in results/pipeline_nse_v2/daily_market.csv (2023 onwards, real data used only for parameters).
* s[i]: a fixed U-shaped diurnal curve, summing to 1 over the 375 minutes.
* q[t, i]: unit-mean intraday GARCH(1,1) on x = r / sqrt(h s), carried across days (as McsGarch does),
  with alpha and beta from the committed full-sample Nifty fit (results/pipeline_nse_v2/mcsgarch_full_sample.csv).
No option-market effect: the option chains, VIX, participant file and volume come from
niftygex.simulate.simulate_market(kappa=0) with the same seed, and are unrelated to these returns.
An infeasible oracle (the true conditional expectation, using the true h and q) is scored too.
The repo's own code path is then run unchanged: build_gex_panel, build_dataset (horizon 30, step 45),
run_walkforward (min_train 100, block 21, one LightGBM seed).

Usage: python experiments/null_world_size_test.py [n_worlds] [n_days] [n_perm_worlds]
Writes results/null_world_size_test/{worlds.csv, permutation_control.csv, summary.csv, run_info.json}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.evaluation import diebold_mariano, qlike
from niftygex.features import build_dataset, build_gex_panel
from niftygex.forecast import WalkForwardConfig, run_walkforward
from niftygex.simulate import SimConfig, simulate_market

OUT = Path("results/null_world_size_test")
N_WORLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
N_PERM = int(sys.argv[3]) if len(sys.argv) > 3 else 3
SEED0 = 9_950_001
PERM_SEED0 = 9_960_001
M = 375
MODELS = ["naive_last", "har", "mcsgarch", "lgbm_base", "hybrid_base"]
CFG = WalkForwardConfig(min_train=100, block=21, models=MODELS, n_seeds=1)
COMPARE = ["hybrid_base", "har", "lgbm_base", "oracle"]


def fit_daily_egarch():
    from arch import arch_model
    d = pd.read_csv("results/pipeline_nse_v2/daily_market.csv")
    y = 100 * np.log(d["close_last_bar"] / d["open"]).dropna().to_numpy()
    res = arch_model(y, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal", rescale=False).fit(disp="off")
    p = res.params
    return {"omega": float(p["omega"]), "alpha": float(p["alpha[1]"]), "gamma": float(p["gamma[1]"]),
            "beta": float(p["beta[1]"]), "n_days": int(len(y)), "mean_sq_pct": float(np.mean(y ** 2))}


def intraday_params():
    f = pd.read_csv("results/pipeline_nse_v2/mcsgarch_full_sample.csv").iloc[0]
    a, b = float(f["alpha"]), float(f["beta"])
    return {"alpha": a, "beta": b, "omega": 1.0 - a - b}


def diurnal_curve():
    i = np.arange(M)
    s = 1.0 + 3.0 * np.exp(-i / 12.0) + 0.6 * np.exp(-(M - 1 - i) / 25.0)
    return s / s.sum()


def mcsgarch_returns(T, seed, eg, ip, s):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((T, M))
    R = np.empty((T, M)); h = np.empty(T); Q = np.empty((T, M))
    lh = eg["omega"] / (1 - eg["beta"])               # unconditional log variance (percent^2)
    q = 1.0; x2_prev = 1.0
    for t in range(T):
        h[t] = np.exp(lh) / 1e4                        # back from percent^2 to return^2
        for i in range(M):
            q = ip["omega"] + ip["alpha"] * x2_prev + ip["beta"] * q
            Q[t, i] = q
            x = np.sqrt(q) * z[t, i]
            R[t, i] = np.sqrt(h[t] * s[i]) * x
            x2_prev = x * x
        e = R[t].sum() / np.sqrt(h[t])
        lh = eg["omega"] + eg["alpha"] * (abs(e) - np.sqrt(2 / np.pi)) + eg["gamma"] * e + eg["beta"] * lh
    return R, h, Q


def make_world(seed, eg, ip, s, T):
    m = simulate_market(SimConfig(n_days=T, kappa=0.0, seed=seed))
    assert m["R"].shape == (T, M), m["R"].shape
    R, h, Q = mcsgarch_returns(T, seed, eg, ip, s)
    o0 = 18000.0 * np.exp(np.concatenate([[0.0], np.cumsum(R.sum(axis=1))[:-1]]))
    C = o0[:, None] * np.exp(np.cumsum(R, axis=1))
    O = np.concatenate([o0[:, None], C[:, :-1]], axis=1)
    m = dict(m)
    m.update(R=R, close=C, open=O, high=np.maximum(O, C), low=np.minimum(O, C), h_true=h, q_true=Q)
    return m


def permute_days(m, seed):
    T = len(m["days"]); p = np.random.default_rng(seed).permutation(T); out = dict(m)
    for k, v in m.items():
        if k == "labels":
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            out[k] = v[p]
        elif isinstance(v, pd.DataFrame) and len(v) == T:
            out[k] = v.iloc[p].set_axis(v.index)
        elif isinstance(v, list) and len(v) == T:
            out[k] = [v[i] for i in p]
    return out


def score(pred, a, b="mcsgarch"):
    la, lb = qlike(pred.y, pred[a]), qlike(pred.y, pred[b])
    day = pred["date"].to_numpy()
    da = pd.Series(la).groupby(day).mean().to_numpy(); db = pd.Series(lb).groupby(day).mean().to_numpy()
    dm = diebold_mariano(db, da, alternative="greater")   # one-sided: is a better than mcsgarch
    return 100 * (1 - la.mean() / lb.mean()), float(dm["stat"]), float(dm["p"])


def add_oracle(m, pred, s, ip, horizon=30):
    """Infeasible oracle: the true conditional expectation of y given the true h[t] and q[t, i] at the anchor
    (both are known one step ahead in the generating process): h[t] * sum_k s[k] * (1 + phi^(k-i) (q[t, i] - 1))."""
    phi = ip["alpha"] + ip["beta"]; h, Q = m["h_true"], m["q_true"]
    t = pred["t"].to_numpy(int); i = pred["minute"].to_numpy(int)
    f = np.zeros(len(pred))
    for j in range(horizon):
        k = i + j; ok = k < M
        f[ok] += s[k[ok]] * (1 + phi ** j * (Q[t[ok], i[ok]] - 1))
    pred = pred.copy(); pred["oracle"] = h[t] * f
    return pred


def run(m, s, ip):
    panel = build_gex_panel(m)
    ds = build_dataset(m, panel, horizon=30, step=45)
    pred, info = run_walkforward(m, ds, CFG)
    pred = add_oracle(m, pred, s, ip)
    methods = sorted({x["daily_method"] for x in info["mcs_params"]})
    return pred, methods


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eg, ip, s = fit_daily_egarch(), intraday_params(), diurnal_curve()
    print("daily EGARCH", eg, "intraday", ip, flush=True)
    rows, prow = [], []
    for k in range(N_WORLDS):
        seed = SEED0 + k; t1 = time.time()
        m = make_world(seed, eg, ip, s, N_DAYS)
        pred, methods = run(m, s, ip)
        r = {"world": k, "seed": seed, "n_days": N_DAYS, "test_anchors": len(pred), "test_days": pred["date"].nunique(),
             "daily_methods_used": "+".join(methods)}
        for a in COMPARE:
            g, st, p = score(pred, a)
            r.update({f"{a}_gain_pct": g, f"{a}_dm_stat": st, f"{a}_dm_p": p})
        g, st, p = score(pred, "hybrid_base", "oracle")
        r.update({"hybrid_over_oracle_gain_pct": g, "hybrid_over_oracle_dm_stat": st, "hybrid_over_oracle_dm_p": p})
        r["seconds"] = round(time.time() - t1, 1)
        rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "worlds.csv", index=False)
        if k < N_PERM:
            t1 = time.time()
            pp, methods = run(permute_days(m, PERM_SEED0 + k), s, ip)
            q = {"world": k, "seed": seed, "perm_seed": PERM_SEED0 + k, "daily_methods_used": "+".join(methods)}
            for a in COMPARE:
                g, st, p = score(pp, a)
                q.update({f"{a}_gain_pct": g, f"{a}_dm_stat": st, f"{a}_dm_p": p})
            q["seconds"] = round(time.time() - t1, 1)
            prow.append(q); print("perm", q, flush=True)
            pd.DataFrame(prow).to_csv(OUT / "permutation_control.csv", index=False)
    W, P = pd.DataFrame(rows), pd.DataFrame(prow)
    summ = []
    for name, df in (("null worlds", W), ("day-permutation control", P)):
        for a in COMPARE:
            if df.empty:
                continue
            summ.append({"set": name, "model": a, "n_worlds": len(df), "mean_gain_pct": df[f"{a}_gain_pct"].mean(),
                         "min_gain_pct": df[f"{a}_gain_pct"].min(), "max_gain_pct": df[f"{a}_gain_pct"].max(),
                         "n_dm_p_below_0.05": int((df[f"{a}_dm_p"] < 0.05).sum()),
                         "n_gain_positive": int((df[f"{a}_gain_pct"] > 0).sum())})
    summ.append({"set": "null worlds", "model": "hybrid_base over oracle", "n_worlds": len(W),
                 "mean_gain_pct": W["hybrid_over_oracle_gain_pct"].mean(), "min_gain_pct": W["hybrid_over_oracle_gain_pct"].min(),
                 "max_gain_pct": W["hybrid_over_oracle_gain_pct"].max(),
                 "n_dm_p_below_0.05": int((W["hybrid_over_oracle_dm_p"] < 0.05).sum()),
                 "n_gain_positive": int((W["hybrid_over_oracle_gain_pct"] > 0).sum())})
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False)
    json.dump({"data": "SYNTHETIC (mcsGARCH data-generating process; real Nifty data used only to set parameters)",
               "script": "experiments/null_world_size_test.py", "seeds": [SEED0, SEED0 + N_WORLDS - 1],
               "perm_seeds": [PERM_SEED0, PERM_SEED0 + N_PERM - 1], "n_days": N_DAYS, "minutes": M,
               "models": MODELS, "min_train": 100, "block": 21, "horizon": 30, "step": 45, "lgbm_seeds": 1,
               "daily_egarch": eg, "intraday_garch": ip, "diurnal": "1 + 3 exp(-i/12) + 0.6 exp(-(374-i)/25), normalized to sum 1",
               "registered": False, "in_reproduce_all": False, "seconds_total": round(time.time() - t0),
               "run_by": "an AI agent (Claude), 4 November 2026"}, open(OUT / "run_info.json", "w"), indent=2)
    print(pd.DataFrame(summ).to_string(), "\nEXIT total seconds", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
