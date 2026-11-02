"""Walk-forward forecast ladder for intraday realized variance.

Models (simplest first; each must beat the ones above it to earn its place):

    naive_last      realized variance over the previous ``horizon`` minutes
    naive_prevday   the same window on the previous day
    vix             India VIX squared, spread over the day with the diurnal
                    curve, with one scale factor fitted on the training window
    har             log-HAR regression with a diurnal regressor, fitted by OLS
    mcsgarch        the paper's model, refitted on each training window
    lgbm_base       LightGBM on log variance, without positioning features
    lgbm_gex        LightGBM on log variance, with the GEX block added
    hybrid_base     mcsGARCH x exp(LightGBM on the log residual), without GEX
    hybrid_gex      the same, with the GEX block

The GEX ablation compares *_base with *_gex: identical models and training,
the only difference is the positioning features.

Walk-forward protocol: the first ``min_train`` days are training only. Then,
repeatedly, every model is fitted on all days before ``start`` and forecasts
days [start, start + block). No future data enters any fit. Hyperparameters
are chosen inside the training window (last 20% of it as validation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .evaluation import qlike
from .features import BASE_FEATURES, GEX_FEATURES
from .volatility import McsGarch, daily_egarch_variance

ALL_MODELS = ["naive_last", "naive_prevday", "vix", "har", "mcsgarch",
              "lgbm_base", "lgbm_gex", "hybrid_base", "hybrid_gex"]

HAR_COLS = ["log_rv_back_h", "log_rv_back_60", "log_rv_today_pm", "log_rv_d1", "log_rv_w", "log_rv_m", "log_s_window"]


@dataclass
class WalkForwardConfig:
    min_train: int = 150
    block: int = 21
    mcs_refit_every: int = 63
    expiry_diurnal: bool = True
    diurnal_smooth: int = 5
    gex_features: list = field(default_factory=lambda: list(GEX_FEATURES))
    # extra named GEX feature sets: model "hybrid_gex_<name>" uses gex_sets[name]
    gex_sets: dict = field(default_factory=dict)
    tune: bool = False
    lgbm_threads: int = 1
    seed: int = 0
    models: list = field(default_factory=lambda: list(ALL_MODELS))


def _smear_fit(resid):
    return float(np.mean(np.exp(resid)))


def _ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return b, y - Xc @ b


def _lgbm_fit_predict(Xtr, ytr, Xte, cfg: WalkForwardConfig, min_leaf: int):
    import lightgbm as lgb

    n = len(Xtr)
    split = int(n * 0.8)
    grid = [dict(num_leaves=15, min_data_in_leaf=min_leaf)]
    if cfg.tune:
        grid = [dict(num_leaves=nl, min_data_in_leaf=ml)
                for nl in (7, 15, 31) for ml in (max(10, min_leaf // 4), min_leaf, min_leaf * 3)]
    best = None
    for g in grid:
        params = dict(objective="regression", learning_rate=0.03, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                      num_threads=cfg.lgbm_threads, seed=cfg.seed, **g)
        dtr = lgb.Dataset(Xtr[:split], ytr[:split])
        dva = lgb.Dataset(Xtr[split:], ytr[split:])
        bst = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        # Score on the validation slice in variance space (QLIKE)
        pv = bst.predict(Xtr[split:], num_iteration=bst.best_iteration)
        sm = _smear_fit(ytr[:split] - bst.predict(Xtr[:split], num_iteration=bst.best_iteration))
        score = np.mean(qlike(np.exp(ytr[split:]), np.exp(pv) * sm))
        if best is None or score < best[0]:
            best = (score, params, max(bst.best_iteration, 10))
    _, params, iters = best
    bst = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=iters)
    sm = _smear_fit(ytr - bst.predict(Xtr))
    return bst.predict(Xte) , sm, {"num_leaves": params["num_leaves"], "min_data_in_leaf": params["min_data_in_leaf"], "iters": iters}, bst


def run_walkforward(market: dict, data: pd.DataFrame, cfg: WalkForwardConfig | None = None,
                    return_models: bool = False):
    """Run the ladder. Returns (predictions, info).

    ``predictions`` has one row per test anchor with y, the forecast of every
    model, and identifiers. ``info`` has per-fold mcsGARCH parameters, chosen
    hyperparameters and feature importances.
    """
    cfg = cfg or WalkForwardConfig()
    R = market["R"]
    T, M = R.shape
    is_exp = market["labels"]["is_expiry"].to_numpy()
    daily = R.sum(axis=1)
    min_leaf = 100 if data["minute"].nunique() > 3 else 15

    data = data.sort_values(["t", "minute"]).reset_index(drop=True)
    out = []
    info = {"mcs_params": [], "hyper": [], "importance": []}
    mcs = None
    h = None
    last_refit = -10**9
    for start in range(cfg.min_train, T, cfg.block):
        end = min(start + cfg.block, T)
        if start - last_refit >= cfg.mcs_refit_every or mcs is None:
            h, _ = daily_egarch_variance(daily, train_size=start)
            mcs = McsGarch(expiry_diurnal=cfg.expiry_diurnal, smooth=cfg.diurnal_smooth)
            mcs.fit(R[:start], h[:start], is_exp[:start])
            last_refit = start
            info["mcs_params"].append({"start": start, **{k: v for k, v in mcs.params.items() if k != "tstats"}})
        filt = mcs.filter(R, h, is_exp)
        d = data[data["t"] < end].copy()
        d["mcs"] = mcs.forecast_variance(filt, h, d["t"].to_numpy(), d["minute"].to_numpy(), d["horizon"].to_numpy())
        S = filt["S"]
        cS = np.concatenate([np.zeros((T, 1)), np.cumsum(S, axis=1)], axis=1)
        tt, ii, hh = d["t"].to_numpy(), d["minute"].to_numpy(), d["horizon"].to_numpy()
        d["log_s_window"] = np.log(cS[tt, ii + hh] - cS[tt, ii])
        d["s_share"] = (cS[tt, ii + hh] - cS[tt, ii]) / cS[tt, M]
        d["log_mcs"] = np.log(d["mcs"])
        tr = d[d["t"] < start]
        te = d[d["t"] >= start].copy()
        if te.empty:
            continue
        ly_tr = np.log(np.maximum(tr["y"].to_numpy(), 1e-16))

        te["naive_last"] = np.exp(te["log_rv_back_h"])
        te["naive_prevday"] = np.exp(te["log_rv_prevday_same"])
        raw_vix = lambda z: np.exp(2 * z["log_vix"]) / 252 * z["s_share"]  # noqa: E731
        te["vix"] = raw_vix(te) * (tr["y"].mean() / raw_vix(tr).mean())
        b, res = _ols(tr[HAR_COLS].to_numpy(), ly_tr)
        te["har"] = np.exp(np.column_stack([np.ones(len(te)), te[HAR_COLS].to_numpy()]) @ b) * _smear_fit(res)
        te["mcsgarch"] = te["mcs"]

        base_cols = BASE_FEATURES + ["log_mcs", "log_s_window"]
        for name in cfg.models:
            if not name.startswith(("lgbm_", "hybrid_")):
                continue
            kind, rest = name.split("_", 1)
            if rest == "base":
                cols = base_cols
            elif rest == "gex":
                cols = base_cols + list(cfg.gex_features)
            elif rest.startswith("gex_"):
                cols = base_cols + list(cfg.gex_sets[rest[4:]])
            else:
                raise ValueError(name)
            target_offset = "log_mcs" if kind == "hybrid" else None
            ytr = ly_tr - (tr[target_offset].to_numpy() if target_offset else 0)
            pred, sm, hp, bst = _lgbm_fit_predict(tr[cols].to_numpy(float), ytr, te[cols].to_numpy(float), cfg, min_leaf)
            base = te[target_offset].to_numpy() if target_offset else 0
            te[name] = np.exp(pred + base) * sm
            info["hyper"].append({"start": start, "model": name, **hp})
            if rest != "base":
                imp = bst.feature_importance("gain")
                info["importance"].append(pd.Series(imp / max(imp.sum(), 1e-12), index=cols, name=f"{name}@{start}"))
        out.append(te)
    pred = pd.concat(out, ignore_index=True)
    keep = ["t", "date", "minute", "horizon", "is_expiry", "tdte", "regime", "y"] + [m for m in cfg.models if m in pred]
    return pred[keep], info
