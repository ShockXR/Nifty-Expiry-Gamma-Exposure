"""Walk-forward forecast ladder for intraday realized variance.

Models (simplest first; each must beat the ones above it to earn its place):

    naive_last      realized variance over the previous ``horizon`` minutes
    naive_prevday   the same window on the previous day
    vix             India VIX squared, spread over the day with the diurnal
                    curve, with one scale factor fitted on the training window
    implied         the ATM straddle-implied variance per trading day of the
                    nearest expiry (chain published at the previous close),
                    spread over the day with the diurnal curve; one scale
                    factor for expiry days and one for other days, fitted on
                    the training window
    har             log-HAR regression with a diurnal regressor, fitted by OLS
    mcsgarch        the paper's model, refitted on each training window
    mcsgarch_iv     the same model with its daily component h_t replaced by
                    the straddle-implied variance per day (no extra parameter:
                    the diurnal curve absorbs the scale)
    linear_<set>    ridge regression of log variance on a feature set (a
                    well-specified log-HAR with the same information as the
                    tree models); ridge penalty chosen inside the training window
    lgbm_<set>      LightGBM on log variance (fixed default hyperparameters)
    lgbmt_<set>     the same, tuned inside each training window (grid of leaves and leaf
                    sizes, re-tuned every ``tune_every`` blocks)
    hybrid_<set>    mcsGARCH x exp(LightGBM on the log residual)
    hybridiv_<set>  mcsgarch_iv x exp(LightGBM on the log residual)

Feature sets (the part after the first underscore):

    base            version 2 base: realized measures, VIX, option-implied
                    block and calendar block, plus the mcsGARCH and implied
                    forecasts as features
    v1              the version 1 base (no implied or calendar block)
    gex             base + all version 1 GEX features
    v1gex           v1 + all version 1 GEX features (the first real-data run)
    gex_<name>      base + ``cfg.gex_sets[name]``

The GEX ablation compares <kind>_base with <kind>_gex_<name>: identical
models, data and training; the only difference is the positioning features.

Walk-forward protocol: the first ``min_train`` days are training only. Then,
repeatedly, every model is fitted on all days before ``start`` and forecasts
days [start, start + block). No future data enters any fit. Hyperparameters
are chosen inside the training window (last 20% of it as validation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .evaluation import qlike
from .features import BASE_FEATURES, BASE_FEATURES_V1, GEX_FEATURES
from .volatility import McsGarch, daily_egarch_variance

ALL_MODELS = ["naive_last", "naive_prevday", "vix", "implied", "har", "mcsgarch", "mcsgarch_iv",
              "linear_base", "lgbm_base", "lgbm_gex", "hybrid_base", "hybrid_gex"]

HAR_COLS = ["log_rv_back_h", "log_rv_back_60", "log_rv_today_pm", "log_rv_d1", "log_rv_w", "log_rv_m", "log_s_window"]
LEARNED_KINDS = ("lgbm", "lgbmt", "hybrid", "hybridiv", "linear")
CATEGORICAL_LEVELS = {"regime": [1, 2], "weekday": [1, 2, 3, 4]}


@dataclass
class WalkForwardConfig:
    min_train: int = 150
    block: int = 21
    mcs_refit_every: int = 63
    expiry_diurnal: bool = True
    diurnal_smooth: int = 5
    gex_features: list = field(default_factory=lambda: list(GEX_FEATURES))
    # extra named GEX feature sets: model "<kind>_gex_<name>" uses gex_sets[name]
    gex_sets: dict = field(default_factory=dict)
    base_features: list = field(default_factory=lambda: list(BASE_FEATURES))
    base_features_v1: list = field(default_factory=lambda: list(BASE_FEATURES_V1))
    tune: bool = False
    tune_every: int = 3          # re-tune every k blocks (always inside the training window)
    n_seeds: int = 1             # LightGBM: average log forecasts over this many seeds
    lgbm_threads: int = 1
    seed: int = 0
    ridge_grid: tuple = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    # Train on every anchor in ``data`` but forecast only these anchor minutes (None = all).
    # Used for the last-hour target: one test window per day, but 22 training windows per day.
    eval_minutes: list | None = None
    models: list = field(default_factory=lambda: list(ALL_MODELS))


def parse_model(name: str):
    """Split a learned model name into (kind, feature-set key)."""
    for kind in sorted(LEARNED_KINDS, key=len, reverse=True):
        if name.startswith(kind + "_"):
            return kind, name[len(kind) + 1:]
    return None, None


def model_columns(rest: str, cfg: WalkForwardConfig, extra_v2: list, extra_v1: list) -> list:
    base_v2 = list(cfg.base_features) + extra_v2
    base_v1 = list(cfg.base_features_v1) + extra_v1
    if rest == "base":
        return base_v2
    if rest == "v1":
        return base_v1
    if rest == "gex":
        return base_v2 + list(cfg.gex_features)
    if rest == "v1gex":
        return base_v1 + list(cfg.gex_features)
    if rest.startswith("v1gex_"):
        return base_v1 + list(cfg.gex_sets[rest[6:]])
    if rest.startswith("gex_"):
        return base_v2 + list(cfg.gex_sets[rest[4:]])
    raise ValueError(f"unknown feature set {rest!r}")


def _smear_fit(resid):
    return float(np.mean(np.exp(resid)))


def _ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return b, y - Xc @ b


# ---------------------------------------------------------------------------
# Ridge regression on log variance
# ---------------------------------------------------------------------------


def _expand(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = df[cols].astype(float).copy()
    for c, levels in CATEGORICAL_LEVELS.items():
        if c in X:
            for lv in levels:
                X[f"{c}_{lv}"] = (X[c] == lv).astype(float)
            X = X.drop(columns=c)
    return X


def _ridge_fit_predict(tr: pd.DataFrame, ytr: np.ndarray, te: pd.DataFrame, cols: list, grid):
    """Ridge on standardised features; penalty chosen on the last 20% of the training window by QLIKE."""
    Xtr, Xte = _expand(tr, cols), _expand(te, cols)
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    sd = Xtr.std()
    keep = sd.index[(sd > 1e-12) & sd.notna() & Xtr.notna().all()]
    mu, sd = Xtr[keep].mean(), sd[keep]
    Ztr = ((Xtr[keep] - mu) / sd).to_numpy()
    Zte = ((Xte[keep] - mu) / sd).to_numpy()
    Zte = np.nan_to_num(Zte)
    # A linear model extrapolates without bound: keep every test feature inside the range it
    # had in the training window (found on a synthetic run where two test days fell far outside).
    Zte = np.clip(Zte, Ztr.min(axis=0), Ztr.max(axis=0))

    def fit(Z, y, lam):
        ym = y.mean()
        A = Z.T @ Z + lam * len(y) * np.eye(Z.shape[1])
        b = np.linalg.solve(A, Z.T @ (y - ym))
        return ym, b

    n = len(ytr)
    split = int(n * 0.8)
    best = None
    for lam in grid:
        ym, b = fit(Ztr[:split], ytr[:split], lam)
        res_in = ytr[:split] - (ym + Ztr[:split] @ b)
        pv = ym + Ztr[split:] @ b
        score = np.mean(qlike(np.exp(ytr[split:]), np.exp(pv) * _smear_fit(res_in)))
        if best is None or score < best[0]:
            best = (score, lam)
    lam = best[1]
    ym, b = fit(Ztr, ytr, lam)
    sm = _smear_fit(ytr - (ym + Ztr @ b))
    coef = pd.Series(b / sd.to_numpy(), index=keep)  # per unit of the raw feature
    return ym + Zte @ b, sm, {"lambda": lam, "n_features": len(keep)}, coef


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


def _lgbm_params(cfg: WalkForwardConfig, seed: int, **g):
    return dict(objective="regression", learning_rate=0.03, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                num_threads=cfg.lgbm_threads, seed=seed, **g)


def _lgbm_tune(Xtr, ytr, cfg: WalkForwardConfig, min_leaf: int):
    """Choose leaves, leaf size and the number of rounds on the last 20% of the training window."""
    import lightgbm as lgb

    n = len(Xtr)
    split = int(n * 0.8)
    grid = [dict(num_leaves=15, min_data_in_leaf=min_leaf)]
    if cfg.tune:
        grid = [dict(num_leaves=nl, min_data_in_leaf=ml)
                for nl in (7, 15, 31) for ml in (max(10, min_leaf // 4), min_leaf, min_leaf * 3)]
    best = None
    for g in grid:
        params = _lgbm_params(cfg, cfg.seed, **g)
        dtr = lgb.Dataset(Xtr[:split], ytr[:split])
        dva = lgb.Dataset(Xtr[split:], ytr[split:])
        bst = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        pv = bst.predict(Xtr[split:], num_iteration=bst.best_iteration)
        sm = _smear_fit(ytr[:split] - bst.predict(Xtr[:split], num_iteration=bst.best_iteration))
        score = np.mean(qlike(np.exp(ytr[split:]), np.exp(pv) * sm))
        if best is None or score < best[0]:
            best = (score, g, max(bst.best_iteration, 10))
    return {"grid": best[1], "iters": best[2]}


def _lgbm_fit_predict(Xtr, ytr, Xte, cfg: WalkForwardConfig, hp: dict):
    """Fit with chosen hyperparameters; average log forecasts over cfg.n_seeds seeds."""
    import lightgbm as lgb

    ptr = np.zeros(len(Xtr))
    pte = np.zeros(len(Xte))
    gain = None
    for s in range(max(1, cfg.n_seeds)):
        params = _lgbm_params(cfg, cfg.seed + 1000 * s, **hp["grid"])
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=hp["iters"])
        ptr += bst.predict(Xtr) / max(1, cfg.n_seeds)
        pte += bst.predict(Xte) / max(1, cfg.n_seeds)
        g = bst.feature_importance("gain")
        gain = g if gain is None else gain + g
    sm = _smear_fit(ytr - ptr)
    return pte, sm, gain


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def _daily_implied_variance(data: pd.DataFrame, T: int) -> np.ndarray | None:
    """Per-day implied variance known before day t opens (from the chain of t-1), or None."""
    if "log_impvar_day" not in data:
        return None
    s = data.groupby("t")["log_impvar_day"].first()
    if s.notna().sum() == 0:
        return None
    s = s.reindex(range(T)).ffill().bfill()
    return np.exp(s.to_numpy(float))


def run_walkforward(market: dict, data: pd.DataFrame, cfg: WalkForwardConfig | None = None,
                    return_models: bool = False, verbose: bool = False):
    """Run the ladder. Returns (predictions, info).

    ``predictions`` has one row per test anchor with y, the forecast of every
    model, and identifiers. ``info`` has per-fold mcsGARCH parameters, chosen
    hyperparameters, feature importances and linear coefficients.
    """
    cfg = cfg or WalkForwardConfig()
    R = market["R"]
    T, M = R.shape
    is_exp = market["labels"]["is_expiry"].to_numpy()
    daily = R.sum(axis=1)
    min_leaf = 100 if data["minute"].nunique() > 3 else 15
    h_iv = _daily_implied_variance(data, T)
    want_iv = h_iv is not None and any(m == "mcsgarch_iv" or m == "implied" or m.startswith("hybridiv_")
                                       or parse_model(m)[1] not in (None, "v1", "v1gex") for m in cfg.models)

    data = data.sort_values(["t", "minute"]).reset_index(drop=True)
    out = []
    info = {"mcs_params": [], "mcs_iv_params": [], "hyper": [], "importance": [], "linear_coef": [], "timing": []}
    mcs = mcs_iv = None
    h = None
    last_refit = -10**9
    tuned: dict = {}
    for b_idx, start in enumerate(range(cfg.min_train, T, cfg.block)):
        t_block = time.time()
        end = min(start + cfg.block, T)
        if start - last_refit >= cfg.mcs_refit_every or mcs is None:
            h, hinfo = daily_egarch_variance(daily, train_size=start)
            mcs = McsGarch(expiry_diurnal=cfg.expiry_diurnal, smooth=cfg.diurnal_smooth)
            mcs.fit(R[:start], h[:start], is_exp[:start])
            info["mcs_params"].append({"start": start, "daily_method": hinfo["method"],
                                       **{k: v for k, v in mcs.params.items() if k != "tstats"}})
            if want_iv:
                mcs_iv = McsGarch(expiry_diurnal=cfg.expiry_diurnal, smooth=cfg.diurnal_smooth)
                mcs_iv.fit(R[:start], h_iv[:start], is_exp[:start])
                info["mcs_iv_params"].append({"start": start, **{k: v for k, v in mcs_iv.params.items() if k != "tstats"}})
            last_refit = start
        filt = mcs.filter(R, h, is_exp)
        d = data[data["t"] < end].copy()
        tt, ii, hh = d["t"].to_numpy(), d["minute"].to_numpy(), d["horizon"].to_numpy()
        d["mcs"] = mcs.forecast_variance(filt, h, tt, ii, hh)
        S = filt["S"]
        cS = np.concatenate([np.zeros((T, 1)), np.cumsum(S, axis=1)], axis=1)
        d["log_s_window"] = np.log(cS[tt, ii + hh] - cS[tt, ii])
        d["s_share"] = (cS[tt, ii + hh] - cS[tt, ii]) / cS[tt, M]
        d["log_mcs"] = np.log(d["mcs"])
        extra_v2 = ["log_mcs", "log_s_window"]
        if want_iv:
            filt_iv = mcs_iv.filter(R, h_iv, is_exp)
            d["mcs_iv"] = mcs_iv.forecast_variance(filt_iv, h_iv, tt, ii, hh)
            d["log_mcs_iv"] = np.log(d["mcs_iv"])
            d["log_implied_window"] = d["log_impvar_day"] + np.log(d["s_share"])
            extra_v2 = extra_v2 + ["log_mcs_iv", "log_implied_window"]
        tr = d[d["t"] < start]
        te = d[d["t"] >= start].copy()
        if cfg.eval_minutes is not None:  # pooled training, evaluation at chosen anchors only
            te = te[te["minute"].isin(cfg.eval_minutes)].copy()
        if te.empty:
            continue
        ly_tr = np.log(np.maximum(tr["y"].to_numpy(), 1e-16))

        te["naive_last"] = np.exp(te["log_rv_back_h"])
        te["naive_prevday"] = np.exp(te["log_rv_prevday_same"])
        raw_vix = lambda z: np.exp(2 * z["log_vix"]) / 252 * z["s_share"]  # noqa: E731
        te["vix"] = raw_vix(te) * (tr["y"].mean() / raw_vix(tr).mean())
        if want_iv:
            raw_imp_tr = np.exp(tr["log_implied_window"])
            raw_imp_te = np.exp(te["log_implied_window"])
            imp = np.empty(len(te))
            for g in (True, False):
                mtr = tr["is_expiry"].to_numpy(bool) == g
                mte = te["is_expiry"].to_numpy(bool) == g
                ref = mtr if mtr.any() else np.ones(len(tr), bool)
                imp[mte] = raw_imp_te[mte] * (tr["y"].to_numpy()[ref].mean() / raw_imp_tr.to_numpy()[ref].mean())
            te["implied"] = imp
            te["mcsgarch_iv"] = te["mcs_iv"]
        b, res = _ols(tr[HAR_COLS].to_numpy(), ly_tr)
        te["har"] = np.exp(np.column_stack([np.ones(len(te)), te[HAR_COLS].to_numpy()]) @ b) * _smear_fit(res)
        te["mcsgarch"] = te["mcs"]

        for name in cfg.models:
            kind, rest = parse_model(name)
            if kind is None:
                continue
            if kind == "hybridiv" and not want_iv:
                continue
            cols = model_columns(rest, cfg, extra_v2 if want_iv else ["log_mcs", "log_s_window"], ["log_mcs", "log_s_window"])
            cols = [c for c in cols if c in d.columns]
            offset = {"hybrid": "log_mcs", "hybridiv": "log_mcs_iv"}.get(kind)
            ytr = ly_tr - (tr[offset].to_numpy() if offset else 0)
            base = te[offset].to_numpy() if offset else 0
            if kind == "linear":
                pred, sm, hp, coef = _ridge_fit_predict(tr, ytr, te, cols, cfg.ridge_grid)
                info["linear_coef"].append(coef.rename(f"{name}@{start}"))
            else:
                Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
                tune_this = cfg.tune or kind == "lgbmt"
                retune_this = (b_idx % max(1, cfg.tune_every) == 0) if tune_this else True
                if name not in tuned or retune_this:
                    tcfg = WalkForwardConfig(**{**cfg.__dict__, "tune": True}) if tune_this else cfg
                    tuned[name] = _lgbm_tune(Xtr, ytr, tcfg, min_leaf)
                hp_t = tuned[name]
                pred, sm, gain = _lgbm_fit_predict(Xtr, ytr, Xte, cfg, hp_t)
                hp = {**hp_t["grid"], "iters": hp_t["iters"], "retuned": bool(retune_this)}
                info["importance"].append(pd.Series(gain / max(gain.sum(), 1e-12), index=cols, name=f"{name}@{start}"))
            te[name] = np.exp(pred + base) * sm
            info["hyper"].append({"start": start, "model": name, **hp})
        info["timing"].append({"start": start, "seconds": round(time.time() - t_block, 1)})
        if verbose:
            print(f"  block start {start} ({end - start} days) {time.time() - t_block:.0f}s", flush=True)
        out.append(te)
    pred = pd.concat(out, ignore_index=True)
    keep = ["t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y"] + [m for m in cfg.models if m in pred]
    return pred[keep], info
