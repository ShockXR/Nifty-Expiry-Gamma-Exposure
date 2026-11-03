"""F2 on raw participant positions: item 8 of the F2 referee response (POST HOC, REAL DATA; referee point M1(a)).

F2 (see experiments/referee_f2.py and docs/REFEREE_RESPONSE_F2.md) is

    r_last = a + b r_rod + d N + c (r_rod x N) + e,    N = 1[Pro-signed GEX at the previous close < 0]

and its N weights Pro's net index-call and index-put positions by the gamma of the front Nifty chain's calls
and puts. The referee (M1(a)) asks whether the gamma weighting matters, or whether F2 is carried by one of Pro's
raw net positions. This script replaces N by each raw position from the NSE participant-wise open interest
file, separately, for Pro and, for comparison as the referee asked, FII and Client (full sample only):

    call            net index calls,   (long - short) / all open index-call contracts (TOTAL long)
    put             net index puts,    (long - short) / all open index-put contracts
    fut             net index futures, (long - short) / all open index-futures contracts
    call_minus_put  call - put (a bullish option stance is long calls and short puts)
    call_plus_put   call + put (added here: the unweighted counterpart of the GEX sign, because the GEX of a
                    category is w_call C + w_put P and C and P are of similar size, so its sign is close to the
                    sign of w_call + w_put)
    gex             the category's GEX divided by gross GEX (the published F2 state, for reference)

All are shares of market open interest, so their scale does not drift with the size of the market. The file
covers all index options and futures (Nifty, Bank Nifty and the other indices, every expiry), not the front
Nifty chain alone. Each position enters in two forms:

    dummy   S = 1[position < 0] (net short), as N in F2: F2's direction is a negative interaction coefficient
    level   S = (position - mean) / sd over the regression sample: the interaction coefficient is the change
            in the slope per one standard deviation more net long, so F2's direction is a POSITIVE coefficient

and the regression is r_last = a + b r_rod + d S + c (r_rod x S). Inference for c: Newey-West (5 lags, as
published), the restricted wild cluster bootstrap (Webb weights) with clusters = episodes (runs of evenings
with the same sign of that position) and = calendar quarters, and a block permutation of S with 60-day blocks
(the t and the coefficient versions), the methods of experiments/referee_f2.py item 3.

Horse race (Pro, full sample): the GEX state and the raw states enter together, each with its own level and
r_rod interaction. If the GEX weighting adds nothing, the GEX interaction should vanish once the raw positions
are in. Inference: Newey-West and wild cluster bootstraps (episodes = runs of the joint sign state of all the
regressors, and quarters); the restricted wild bootstrap imposes only that one coefficient is zero, which is the
right null here. A block permutation is not reported for the horse race: moving one state while holding the
other fixed breaks their joint dependence, so it does not test that null.

Everything here was specified after the referee report (3 Nov 2026) and after every F2 result had been seen.
Nothing is pre-registered and no p-value is adjusted for the count printed at the end.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/referee_f2_positions.py
    PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_referee_f2_positions.py

Writes results/referee_f2/item8_*.csv, results/referee_f2/item8_run_info.json and
results/referee_f2_positions_stdout.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dealer_sign as dsg  # noqa: E402
import referee_f2 as rf  # noqa: E402
from niftygex import data as nd  # noqa: E402

SEED = rf.SEED
CATS8 = ("pro", "fii", "client")
RAW_POS = ("call", "put", "fut", "call_minus_put", "call_plus_put")
POSITIONS = RAW_POS + ("gex",)
FORMS = ("dummy", "level")
POS_LABEL = {"call": "net index calls", "put": "net index puts", "fut": "net index futures",
             "call_minus_put": "calls minus puts", "call_plus_put": "calls plus puts",
             "gex": "GEX / gross (published F2 state)"}
PERM_BLOCK = 60


# =====================================================================================
# Data
# =====================================================================================


def raw_positions(directory) -> pd.DataFrame:
    """Net index-option and index-futures positions of every category as a share of market open interest, one
    row per participant file date (the positions at that day's close)."""
    rows = {}
    for f in sorted(Path(directory).glob("*.csv")):
        try:
            day = pd.to_datetime(f.stem[-8:], format="%d%m%Y")
        except ValueError:
            continue
        t = nd.load_participant_oi(f)
        tc = float(t.loc["TOTAL", "option_index_call_long"])
        tp = float(t.loc["TOTAL", "option_index_put_long"])
        tf = float(t.loc["TOTAL", "future_index_long"])
        r = {}
        for g in ("Client", "DII", "FII", "Pro"):
            k = g.lower()
            c = (t.loc[g, "option_index_call_long"] - t.loc[g, "option_index_call_short"]) / tc
            p = (t.loc[g, "option_index_put_long"] - t.loc[g, "option_index_put_short"]) / tp
            fu = (t.loc[g, "future_index_long"] - t.loc[g, "future_index_short"]) / tf
            r[f"{k}_call"], r[f"{k}_put"], r[f"{k}_fut"] = c, p, fu
            r[f"{k}_call_minus_put"] = c - p
            r[f"{k}_call_plus_put"] = c + p
        rows[day] = r
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def evening_values(raw: pd.DataFrame, gex: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """The positions at the previous evening's close for each trading day in ``days``: forward-filled to the
    panel's trading days and lagged one trading day, exactly as experiments/referee_f2.py treats the GEX sign."""
    nt = raw.reindex(gex.index).ffill()
    out = pd.DataFrame(index=gex.index)
    for cat in CATS8:
        for pos in RAW_POS:
            out[f"{cat}_{pos}"] = nt[f"{cat}_{pos}"]
        out[f"{cat}_gex"] = gex[cat] / gex["gross"]
    return out.shift(1).reindex(days)


def make_state(v: np.ndarray, form: str) -> tuple[np.ndarray, np.ndarray]:
    """(S, sign dummy) for one position on the regression sample. The level is standardized on that sample."""
    v = np.asarray(v, float)
    dummy = (v < 0).astype(float)
    if form == "dummy":
        return dummy, dummy
    sd = v.std()
    return ((v - v.mean()) / sd if sd > 0 else np.zeros_like(v)), dummy


def joint_episodes(dummies: list[np.ndarray]) -> np.ndarray:
    """Episode labels for several sign states at once: a new episode starts whenever any of them changes."""
    code = np.zeros(len(dummies[0]))
    for k, x in enumerate(dummies):
        code = code + np.asarray(x, float) * 2.0 ** k
    return rf.episode_ids(code)


# =====================================================================================
# Inference
# =====================================================================================


def perm_state_draws(y, Z, S, M, j: int, block: int, B: int, rng) -> np.ndarray:
    """(coef, HAC t) of column j over B block relocations of the state S (dummy or level). Relocation keeps the
    values of S (so a dummy keeps its count of 1s and a level its distribution) and the runs inside each block."""
    out = np.full((B, 2), np.nan)
    for b in range(B):
        Sp = rf.block_relocate(S, block, rng)
        X = np.column_stack([Z, Sp[:, None] * M])
        if np.linalg.cond(X.T @ X) > 1e12:
            continue
        f = rf.ols_focus(y, X, j)
        out[b] = f["coef"], f["t"]
    return out


def infer_state(y, r, S, dummy, quarters, cfg: dict, seed_key: list, binary: bool, perm: bool = True) -> list[dict]:
    """F2 interaction c for one state S: X = [1, r_rod, S, r_rod x S]."""
    y, r, S, dummy = (np.asarray(a, float) for a in (y, r, S, dummy))
    n = len(y)
    ep = rf.episode_ids(dummy)
    n_neg = int(dummy.sum())
    head = {"n": n, "n_neg": n_neg, "n_neg_episodes": int(sum(1 for _, _, v in rf.runs_of(dummy) if v == 1)),
            "n_episodes": int(len(np.unique(ep))), "n_quarters": int(len(np.unique(quarters)))}
    if (binary and (n_neg < 5 or n_neg > n - 5)) or (not binary and np.std(S) == 0):
        return [{**head, "method": "not estimable", "detail": "fewer than 5 days in one sign state" if binary
                 else "no variation"}]
    Z = np.column_stack([np.ones(n), r])
    M = np.column_stack([np.ones(n), r])
    X = np.column_stack([Z, S[:, None] * M])
    j = 3
    f5 = rf.ols_focus(y, X, j)
    r0 = {**head, "coef": f5["coef"]}
    rows = [{**r0, "method": "newey_west_5", "se": f5["se"], "stat": f5["t"], "p": f5["p"], "detail": "Bartlett, 5 lags"}]
    for ci, (name, groups) in enumerate((("episode", ep), ("quarter", np.asarray(quarters)))):
        w = rf.wild_cluster_p(y, X, j, groups, cfg["n_wild"], np.random.default_rng(seed_key + [1 + ci]))
        rows.append({**r0, "method": f"wild_cluster_{name}", "stat": w["t"], "p": w["p"], "n_clusters": w["G"],
                     "detail": f"restricted, Webb weights, {cfg['n_wild']} draws"})
    if perm:
        draws = perm_state_draws(y, Z, S, M, j, PERM_BLOCK, cfg["n_perm"], np.random.default_rng(seed_key + [9]))
        nv = int(np.isfinite(draws[:, 0]).sum())
        band = (f"placebo coef 2.5 to 97.5%: {np.nanpercentile(draws[:, 0], 2.5):.4f} to "
                f"{np.nanpercentile(draws[:, 0], 97.5):.4f}") if nv else "no valid draws"
        for stat_name, real, col in (("t", f5["t"], 1), ("coef", f5["coef"], 0)):
            rows.append({**r0, "method": f"block_permutation_{PERM_BLOCK}_{stat_name}", "stat": real,
                         "p": rf.perm_p(real, draws[:, col]), "n_clusters": int(np.ceil(n / PERM_BLOCK)),
                         "detail": f"{nv} valid draws; statistic |{'HAC t' if stat_name == 't' else 'coef'}|; {band}"})
    return rows


def nw_matrix(S: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West (Bartlett) long-run covariance sum of the score matrix S (n x q), no division by n."""
    S = np.asarray(S, float)
    V = S.T @ S
    for lag in range(1, min(lags, len(S) - 1) + 1):
        G = S[lag:].T @ S[:-lag]
        V += (1.0 - lag / (lags + 1.0)) * (G + G.T)
    return V


def hac_wald(y, X, J: list[int], lags: int = rf.LAGS) -> dict:
    """Newey-West Wald test that the coefficients in columns J are all zero (chi-square with len(J) df)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    H = X @ XtX_inv[:, J]
    V = nw_matrix(H * u[:, None], lags)
    W = float(b[J] @ np.linalg.solve(V, b[J]))
    return {"W": W, "p": float(rf.stats.chi2.sf(W, len(J))), "q": len(J)}


def wild_cluster_wald(y, X, J: list[int], groups, B: int, rng, chunk: int = 500) -> dict:
    """Restricted wild cluster bootstrap (Webb weights) of the CR1 Wald statistic for b_J = 0: the null is
    imposed by dropping the columns J, restricted residuals are flipped by cluster, and the Wald statistic is
    recomputed on every draw. p = (1 + #{W* >= W}) / (1 + B)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    codes = pd.factorize(np.asarray(groups))[0]
    G = int(codes.max()) + 1
    if G < 2:
        return {"W": np.nan, "p": np.nan, "G": G}
    XtX_inv = np.linalg.inv(X.T @ X)
    Pm = XtX_inv @ X.T
    H = X @ XtX_inv[:, J]  # n x q
    c = G / (G - 1) * (n - 1) / (n - k)
    ind = np.zeros((G, n))
    ind[codes, np.arange(n)] = 1.0

    q = len(J)

    def wald(Bs, R):  # Bs: k x m coefficients, R: n x m residuals
        m = R.shape[1]
        Sg = (ind @ (H[:, :, None] * R[:, None, :]).reshape(n, q * m)).reshape(G, q, m)
        V = c * np.einsum("gqm,grm->mqr", Sg, Sg)
        bJ = Bs[J].T  # m x q
        return np.einsum("mq,mq->m", bJ, np.linalg.solve(V, bJ[:, :, None])[:, :, 0])

    b = Pm @ y
    W0 = float(wald(b[:, None], (y - X @ b)[:, None])[0])
    Xr = np.delete(X, J, axis=1)
    yr = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    ur = y - yr
    count, done = 0, 0
    while done < B:
        m = min(chunk, B - done)
        Ys = yr[:, None] + ur[:, None] * rf.webb_weights((G, m), rng)[codes]
        Bs = Pm @ Ys
        count += int(np.sum(wald(Bs, Ys - X @ Bs) >= W0))
        done += m
    return {"W": W0, "p": (1 + count) / (1 + B), "G": G}


def infer_multi(y, r, states: dict, dummies: dict, quarters, cfg: dict, seed_key: list,
                joint: list[str] | None = None) -> list[dict]:
    """Horse race: X = [1, r_rod] plus, for every state k, [S_k, r_rod x S_k]. One row per (state, method), and,
    when ``joint`` names two or more states, a joint test that all their r_rod interactions are zero."""
    y, r = np.asarray(y, float), np.asarray(r, float)
    n = len(y)
    names = list(states)
    cols = [np.ones(n), r]
    for k in names:
        S = np.asarray(states[k], float)
        cols += [S, S * r]
    X = np.column_stack(cols)
    ep = joint_episodes([dummies[k] for k in names])
    head = {"n": n, "n_episodes": int(len(np.unique(ep))), "n_quarters": int(len(np.unique(quarters)))}
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return [{**head, "state": k, "method": "not estimable", "detail": "collinear design"} for k in names]
    rows = []
    clusters = (("joint_episode", ep), ("quarter", np.asarray(quarters)))
    for i, k in enumerate(names):
        j = 3 + 2 * i
        f5 = rf.ols_focus(y, X, j)
        r0 = {**head, "state": k, "coef": f5["coef"]}
        rows.append({**r0, "method": "newey_west_5", "se": f5["se"], "stat": f5["t"], "p": f5["p"]})
        for ci, (name, groups) in enumerate(clusters):
            w = rf.wild_cluster_p(y, X, j, groups, cfg["n_wild"], np.random.default_rng(seed_key + [i, 1 + ci]))
            rows.append({**r0, "method": f"wild_cluster_{name}", "stat": w["t"], "p": w["p"], "n_clusters": w["G"]})
    if joint and len(joint) >= 2:
        J = [3 + 2 * names.index(k) for k in joint]
        r0 = {**head, "state": "joint", "detail": f"Wald test that the r_rod interactions of {', '.join(joint)} are all 0"}
        hw = hac_wald(y, X, J)
        rows.append({**r0, "method": "newey_west_5_wald", "stat": hw["W"], "p": hw["p"],
                     "detail": r0["detail"] + f"; chi-square, {hw['q']} df"})
        for ci, (name, groups) in enumerate(clusters):
            w = wild_cluster_wald(y, X, J, groups, cfg["n_wild"], np.random.default_rng(seed_key + [99, 1 + ci]))
            rows.append({**r0, "method": f"wild_cluster_{name}_wald", "stat": w["W"], "p": w["p"], "n_clusters": w["G"],
                         "detail": r0["detail"] + "; restricted wild cluster bootstrap of the CR1 Wald statistic"})
    return rows


# =====================================================================================
# Item 8
# =====================================================================================


def describe(ev: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Descriptive table on the F2 days: share net short by year, runs, persistence, agreement with the GEX sign."""
    rows = []
    for cat in CATS8:
        g = ev[f"{cat}_gex"]
        for pos in POSITIONS:
            v = ev[f"{cat}_{pos}"]
            ok = v.notna() & g.notna()
            x, gx = v[ok], g[ok]
            dm_ = (x < 0).astype(float).to_numpy()
            row = {"category": cat, "position": pos, "label": POS_LABEL[pos], "days": int(ok.sum()),
                   "mean": x.mean(), "sd": x.std(ddof=0), "share_net_short": dm_.mean()}
            for y in sorted(x.index.year.unique()):
                row[f"share_net_short_{y}"] = float(dm_[x.index.year == y].mean())
            rr = rf.runs_of(dm_)
            row["net_short_runs"] = int(sum(1 for _, _, v_ in rr if v_ == 1))
            row["sign_switches"] = int(np.sum(np.diff(dm_) != 0))
            row["acf1_sign"] = rf.acf(dm_, 1)
            row["acf1_level"] = rf.acf(x.to_numpy(), 1)
            row["n_eff_ar1_sign"] = rf.n_eff_ar1(dm_)
            row["agree_with_gex_sign"] = float(np.mean(dm_ == (gx < 0).to_numpy()))
            row["corr_with_gex_level"] = float(np.corrcoef(x, gx)[0, 1])
            rows.append(row)
    return pd.DataFrame(rows)


def item8_single(d: pd.DataFrame, ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """F2 with each position as the state: Pro on all three samples, FII and Client on the full sample."""
    rows = []
    plan = [("pro", s) for s in rf.SAMPLES] + [("fii", "full"), ("client", "full")]
    for cat, sname in plan:
        ci, si = CATS8.index(cat), list(rf.SAMPLES).index(sname)
        m = rf.sample_mask(d.index, sname)
        for pi, pos in enumerate(POSITIONS):
            v = ev[f"{cat}_{pos}"]
            ok = m & v.notna().to_numpy()
            sub = d[ok]
            for fi, form in enumerate(FORMS):
                S, dummy = make_state(v[ok].to_numpy(), form)
                new = infer_state(sub["r_last"].to_numpy(), sub["r_rod"].to_numpy(), S, dummy, sub["yq"].to_numpy(),
                                  cfg, [SEED, 8, 1, ci, si, pi, fi], binary=(form == "dummy"))
                for rw in new:
                    rw.update({"category": cat, "position": pos, "form": form, "sample": sname})
                    if "coef" in rw:
                        rw["same_direction_as_f2"] = bool(rw["coef"] < 0) if form == "dummy" else bool(rw["coef"] > 0)
                rows += new
    cols = ["category", "position", "form", "sample", "n", "n_neg", "n_neg_episodes", "n_episodes", "n_quarters",
            "coef", "same_direction_as_f2", "method", "se", "stat", "p", "n_clusters", "detail"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out]]


HORSE_RACES = [
    # (label, [(position, form), ...]); the first entry is the GEX state
    ("GEX sign + net calls sign", [("gex", "dummy"), ("call", "dummy")]),
    ("GEX sign + net puts sign", [("gex", "dummy"), ("put", "dummy")]),
    ("GEX sign + net futures sign", [("gex", "dummy"), ("fut", "dummy")]),
    ("GEX sign + calls minus puts sign", [("gex", "dummy"), ("call_minus_put", "dummy")]),
    ("GEX sign + calls plus puts sign", [("gex", "dummy"), ("call_plus_put", "dummy")]),
    ("GEX sign + call, put and futures signs", [("gex", "dummy"), ("call", "dummy"), ("put", "dummy"), ("fut", "dummy")]),
    ("GEX sign + call, put and futures levels", [("gex", "dummy"), ("call", "level"), ("put", "level"), ("fut", "level")]),
    ("GEX level + call, put and futures levels", [("gex", "level"), ("call", "level"), ("put", "level"), ("fut", "level")]),
    ("GEX level + calls plus puts level", [("gex", "level"), ("call_plus_put", "level")]),
]


def item8_horse(d: pd.DataFrame, ev: pd.DataFrame, cfg: dict, cat: str = "pro", sname: str = "full") -> pd.DataFrame:
    m = rf.sample_mask(d.index, sname)
    rows = []
    for hi, (label, spec) in enumerate(HORSE_RACES):
        cols = [f"{cat}_{p}" for p, _ in spec]
        ok = m & ev[cols].notna().all(axis=1).to_numpy()
        sub = d[ok]
        states, dummies = {}, {}
        for p, form in spec:
            key = f"{p}_{form}"
            states[key], dummies[key] = make_state(ev.loc[ok, f"{cat}_{p}"].to_numpy(), form)
        raw_keys = list(states)[1:]
        new = infer_multi(sub["r_last"].to_numpy(), sub["r_rod"].to_numpy(), states, dummies, sub["yq"].to_numpy(), cfg,
                          [SEED, 8, 2, hi], joint=raw_keys)
        for rw in new:
            if rw["state"] == "joint":
                rw.update({"category": cat, "sample": sname, "horse_race": label,
                           "position": "raw states jointly", "form": "joint"})
                continue
            p, form = rw["state"].rsplit("_", 1)
            rw.update({"category": cat, "sample": sname, "horse_race": label, "position": p, "form": form})
            if "coef" in rw:
                rw["same_direction_as_f2"] = bool(rw["coef"] < 0) if form == "dummy" else bool(rw["coef"] > 0)
        rows += new
    cols = ["horse_race", "category", "sample", "position", "form", "n", "n_episodes", "n_quarters", "coef",
            "same_direction_as_f2", "method", "se", "stat", "p", "n_clusters", "detail"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out]]


def compact(t: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per regression: coefficient, Newey-West p and the cautious p-values side by side."""
    t = t[t["method"] != "not estimable"]
    pv = t.pivot_table(index=keys, columns="method", values="p", sort=False).reset_index()
    co = t.groupby(keys, sort=False)["coef"].first().reset_index()
    out = co.merge(pv, on=keys)
    robust = [c for c in out if c.startswith("wild_") or c.startswith("block_permutation")]
    out["largest_robust_p"] = out[robust].max(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse_v2")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--external", default="data/external")
    ap.add_argument("--out", default="results/referee_f2")
    ap.add_argument("--n-perm", type=int, default=4999)
    ap.add_argument("--n-wild", type=int, default=4999)
    a = ap.parse_args()
    cfg = {"n_perm": a.n_perm, "n_wild": a.n_wild}
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        lines.append(s)
        print(s, flush=True)

    def show(title, df):
        with pd.option_context("display.width", 260, "display.max_columns", 40, "display.max_rows", 400,
                               "display.max_colwidth", 60, "display.precision", 4):
            say(f"== {title}")
            say(df.to_string(index=False))
            say()

    run, raw, ext = Path(a.run), Path(a.raw), Path(a.external)
    dm = pd.read_csv(run / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(run / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    nets = dsg.participant_nets(raw / "participant_oi")
    gex = rf.rule_gex(panel, nets)
    closes = rf.minute_closes(raw / "nifty_1min.csv", dm.index)
    sensex = rf.sensex_expiry_days(ext / "sensex_expiry_calendar.csv", dm.index)
    sebi = rf.sebi_dates(ext / "sebi_phases.csv")
    d_all = rf.build_frame(dm, gex, closes, sensex, sebi)
    d = d_all.dropna(subset=["r_rod", "r_last"]).copy()
    pos = raw_positions(raw / "participant_oi")
    ev = evening_values(pos, gex, d_all.index).reindex(d.index)

    say("REAL DATA. POST HOC (referee point M1(a)): F2 with Pro's, FII's and Client's raw net positions in place of the GEX sign.")
    say("Nothing here is pre-registered; no p-value is adjusted for the count printed at the end.")
    say(f"Inputs: {run}/daily_market.csv, {run}/daily_gex_panel.csv, {raw}/nifty_1min.csv, {raw}/participant_oi "
        f"({len(pos)} files read).")
    chk = pd.DataFrame([
        {"check": "net call and put shares here vs experiments/dealer_sign.py participant_nets (max abs diff)",
         "value": float(max((pos[f"{c}_{s}"] - nets[f"{c}_{s}"]).abs().max() for c in ("pro", "fii", "dii", "client")
                            for s in ("call", "put")))},
        {"check": "F2 days whose Pro GEX sign here differs from neg_pro of experiments/referee_f2.py",
         "value": int(((ev["pro_gex"] < 0).astype(float) != d["neg_pro"]).sum())},
        {"check": "F2 days", "value": len(d)},
        {"check": "F2 days with every raw position", "value": int(ev[[f"{c}_{p}" for c in CATS8 for p in RAW_POS]]
                                                               .notna().all(axis=1).sum())},
    ])
    chk.to_csv(out / "item8_input_checks.csv", index=False)
    show("ITEM 8. Input checks", chk)

    desc = describe(ev, d)
    desc.to_csv(out / "item8_positions_describe.csv", index=False)
    show("ITEM 8. Raw positions on the F2 days (previous evening): share net short, persistence, agreement with the "
         "GEX sign", desc)
    say(f"[{time.time() - t0:.0f}s] descriptives done")

    single = item8_single(d, ev, cfg)
    single.to_csv(out / "item8_positions_f2.csv", index=False)
    show("ITEM 8. F2 with each raw position as the state (dummy: 1 = net short, F2 direction negative; level: per "
         "1 sd more net long, F2 direction positive)", compact(single, ["category", "sample", "position", "form"]))
    say(f"[{time.time() - t0:.0f}s] single-state regressions done")

    horse = item8_horse(d, ev, cfg)
    horse.to_csv(out / "item8_horse_race.csv", index=False)
    show("ITEM 8. Horse race, Pro, full sample: GEX state and raw states together",
         compact(horse, ["horse_race", "position", "form"]))
    say(f"[{time.time() - t0:.0f}s] horse races done")

    n_p = int(single["p"].notna().sum() + horse["p"].notna().sum())
    say(f"p-values reported by this script (all post hoc, none adjusted for this count): {n_p}")
    info = {"seed": SEED, **cfg, "perm_block": PERM_BLOCK, "seconds": round(time.time() - t0, 1),
            "n_p_values_reported": n_p,
            "inputs_sha256_16": {str(p): rf._sha(p) for p in (run / "daily_market.csv", run / "daily_gex_panel.csv",
                                                             raw / "nifty_1min.csv")},
            "participant_files": len(pos),
            "status": "post hoc, in response to referee point M1(a); not pre-registered"}
    (out / "item8_run_info.json").write_text(json.dumps(info, indent=2, default=str))
    say(f"Done in {time.time() - t0:.0f}s.")
    (out.parent / "referee_f2_positions_stdout.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
