"""P71: how large must an effect be for our tests to see it? Power calibrated on the REAL sample.

SIMULATIONS CALIBRATED ON REAL DATA. The regressors, the fitted values and the residuals come from the real
Nifty data (results/pipeline_nse/daily_market.csv, daily_gex_panel.csv and the validated 1-minute bars); only
the outcome is simulated, with an effect of known size added. The synthetic power study (experiments/power_study.py,
E4) is not used here.

For each key test of the real-data study we

    1. fit the null model (the test's own regression without the term being tested) on the real data,
    2. build simulated outcomes  y* = fitted null + delta x (tested regressor) + resampled residuals,
       where the residuals of step 1 are resampled in blocks of consecutive days (a stationary bootstrap,
       Politis and Romano 1994) so that autocorrelation and volatility clustering are kept,
    3. rerun exactly the same test (same regressors, Newey-West standard errors with the same number of lags,
       same 5% level), and
    4. count how often the test rejects: that share is the power at delta. At delta = 0 it is the size.

Tests (the code that defines each one is named):

    R2_tue_day        expiry-day premium in the Tuesday regime, whole day: mean of week-relative log realized
                      variance, Wilcoxon signed-rank test (real_day_level.r2_regimes); a Newey-West t-test of the
                      mean (4 weekly lags) is a second statistic. R2_tue_last_hour, R2_thu_day and R2_multi_day
                      are the same for the last hour and for the two Thursday regimes.
    R3a, R3b          ordinary-day GEX term (gex_l1) and expiry-day interaction (gex_l1_x_exp), scale-free share
                      measure (net GEX / gross GEX of the same chain), Pro rule, regime x expiry controls, Newey-West
                      5 lags (real_day_level.r3_robustness, variant "share").
    R5                end-of-day reversal interaction r_rod x 1[GEX < 0], Pro rule, all days, Newey-West 5 lags
                      (momentum_gex.py, specification "binary").
    FC                one-day-ahead forecasting gain from GEX: models fitted on 2023 to 2024 and frozen, forecasts
                      from 2025 on, squared error of log realized variance, one-sided Diebold-Mariano test with
                      Newey-West 5 lags (falsification.py, T4). The injected effect is lambda times the in-sample
                      R3 share estimates of both GEX terms, so lambda = 1 is "the effect we estimated in-sample".

Effect sizes are scanned as multiples lambda of the real estimate (delta = lambda x estimate). Larger samples
(1.5x, 2x, 3x) keep the real days and append extra days drawn in blocks from the real ones (from all days, or
from the Tuesday regime only), which assumes the future looks like the past. A second resampling scheme (a
dependent wild bootstrap: each residual stays on its own day and is multiplied by a random sign that is constant
over blocks of days) keeps the link between residual size and the regressors. Sign errors: the GEX regressor
used in the test has the wrong sign on a fraction of days (random days, spells of about 20 days, or the days
with the smallest |GEX|), while the outcome is generated with the correct sign.

    OMP_NUM_THREADS=1 python experiments/power_real.py                  # full run (about 9 minutes on 4 cores)
    OMP_NUM_THREADS=1 python experiments/power_real.py --reps 200       # quick check
    python experiments/power_real.py --plots-only                       # redraw the figures from the CSVs

Writes results/power_real/*.csv, results/power_real/*.png, results/power_real/run_info.json and
results/power_real_stdout.txt. Seeds: every cell uses numpy default_rng([SEED, cell hash, chunk]); the
entropy of each cell is in the seed_entropy column of power_curves.csv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import momentum_gex as mg  # noqa: E402
import real_day_level as rdl  # noqa: E402

SEED = 71_2026
ALPHA = 0.05
TARGET_POWER = 0.80
LAGS = 5            # Newey-West lags of R3, R5 and the forecast test, as in the original scripts
LAGS_WEEKS = 4      # Newey-West lags of the secondary R2 t-test (weekly data, as in the event study)
BLOCK_DAYS = 10     # mean block length of the stationary bootstrap for daily residuals (two trading weeks)
BLOCK_WEEKS = 4     # same for the weekly R2 residuals
SPELL_DAYS = 20     # mean length of a spell of wrong signs in the "spells" pattern
RULE = "participant_pro"
SPLIT_END = pd.Timestamp("2024-12-31")  # forecast test: fit to this date, forecast after it (falsification T4)
LAMBDAS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0,
                    4.0, 5.0, 6.0, 8.0])
MULTS = (1.0, 1.5, 2.0, 3.0)
FLIP_GRID = (0.10, 0.25, 0.40)
Z2 = stats.norm.ppf(1 - ALPHA / 2)   # two-sided 5%
Z1 = stats.norm.ppf(1 - ALPHA)       # one-sided 5% (Diebold-Mariano)

LABELS = {
    "R2_tue_day": "R2 expiry premium, Tuesday regime, whole day (log points)",
    "R2_tue_last_hour": "R2 expiry premium, Tuesday regime, last hour (log points)",
    "R2_thu_day": "R2 expiry premium, one weekly Thursday regime, whole day (log points)",
    "R2_multi_day": "R2 expiry premium, several weeklies regime, whole day (log points)",
    "R3a": "R3 ordinary-day GEX term, share measure, Pro rule (coefficient)",
    "R3b": "R3 expiry-day GEX interaction, share measure, Pro rule (coefficient)",
    "R5": "R5 reversal interaction r_rod x 1[GEX<0], Pro rule (coefficient)",
    "FC": "Forecast gain from GEX, share, Pro rule (multiples of the R3 estimate)",
}
KEY_TESTS = ["R2_tue_day", "R3a", "R3b", "R5", "FC"]
DAILY_TESTS = ["R3a", "R3b", "R5", "FC"]


# ---------------------------------------------------------------------------
# Numerical helpers (tested in tests/test_power_real.py)
# ---------------------------------------------------------------------------


def nw_weights(lags: int) -> np.ndarray:
    return 1.0 - np.arange(1, lags + 1) / (lags + 1.0)


def ols_hac_batch(X: np.ndarray, Y: np.ndarray, j: int, lags: int = LAGS) -> tuple[np.ndarray, np.ndarray]:
    """OLS of every column of Y on X; coefficient j and its Newey-West (Bartlett) standard error.

    Same as statsmodels OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": lags}) (no small-sample correction),
    which real_day_level.py and momentum_gex.py use. The variance of coefficient j is sum_t sum_s w(|t-s|)
    (a_t u_t)(a_s u_s), with a_t the t-th weight of coefficient j in (X'X)^-1 X'.
    """
    P = np.linalg.solve(X.T @ X, X.T)
    B = P @ Y
    U = Y - X @ B
    H = P[j][:, None] * U
    V = np.einsum("ij,ij->j", H, H)
    for lag, w in enumerate(nw_weights(lags), start=1):
        if lag < len(H):
            V += 2.0 * w * np.einsum("ij,ij->j", H[lag:], H[:-lag])
    return B[j], np.sqrt(np.maximum(V, 0.0))


def mean_hac_batch(D: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Column means of D and their Newey-West standard errors (regression on a constant)."""
    n = len(D)
    mu = D.mean(axis=0)
    H = (D - mu) / n
    V = np.einsum("ij,ij->j", H, H)
    for lag, w in enumerate(nw_weights(lags), start=1):
        if lag < n:
            V += 2.0 * w * np.einsum("ij,ij->j", H[lag:], H[:-lag])
    return mu, np.sqrt(np.maximum(V, 0.0))


def stationary_indices(n_src: int, n_out: int, mean_block: float, rng) -> np.ndarray:
    """Stationary bootstrap (Politis and Romano 1994): runs of consecutive positions (wrapping around) whose
    lengths are geometric with mean ``mean_block``, each starting at a uniform random position."""
    if n_out <= 0:
        return np.empty(0, int)
    starts = rng.random(n_out) < 1.0 / mean_block
    starts[0] = True
    pos = np.flatnonzero(starts)
    run = np.cumsum(starts) - 1
    first = rng.integers(0, n_src, size=len(pos))
    return (first[run] + np.arange(n_out) - pos[run]) % n_src


def block_signs(n: int, block: int, rng) -> np.ndarray:
    """Random signs (+1 or -1 with equal chance), constant within blocks of ``block`` days, random block grid."""
    off = int(rng.integers(block))
    s = rng.choice(np.array([-1.0, 1.0]), size=(n + off) // block + 1)
    return s[(np.arange(n) + off) // block]


def flip_mask(n: int, f: float, pattern: str, rng, mag: np.ndarray | None = None,
              spell: float = SPELL_DAYS) -> np.ndarray:
    """Days on which the measured GEX has the wrong sign.

    iid: each day independently with probability f. spells: a two-state Markov chain whose wrong-sign spells
    last ``spell`` days on average and cover a share f of days. smallest: the share f of days with the smallest
    |GEX| (``mag``), the most favourable case, since a wrong sign then changes the regressor least.
    """
    if f <= 0:
        return np.zeros(n, bool)
    if f >= 1:
        return np.ones(n, bool)
    if pattern == "iid":
        return rng.random(n) < f
    if pattern == "smallest":
        k = int(round(f * n))
        out = np.zeros(n, bool)
        out[np.argsort(mag, kind="stable")[:k]] = True
        return out
    if pattern == "spells":
        mean_on, mean_off = spell, spell * (1 - f) / f
        state = rng.random() < f
        out, filled = np.zeros(n, bool), 0
        while filled < n:
            length = int(rng.geometric(1.0 / (mean_on if state else mean_off)))
            out[filled:filled + length] = state
            filled += length
            state = not state
        return out
    raise ValueError(pattern)


def measured(v: np.ndarray, flip: np.ndarray, binary: bool) -> np.ndarray:
    if not flip.any():
        return v
    return np.where(flip, 1.0 - v, v) if binary else np.where(flip, -v, v)


def gex_cols(v: np.ndarray, mult: np.ndarray, tested: str) -> tuple[np.ndarray, np.ndarray]:
    """(other GEX column, tested GEX column). tested = 'level' (v) or 'interaction' (v x mult)."""
    inter = v * mult
    return (inter, v) if tested == "level" else (v, inter)


def interp_power(lams: np.ndarray, power: np.ndarray, lam: float) -> float:
    return float(np.interp(lam, lams, power))


def mde_from_curve(lams: np.ndarray, power: np.ndarray, target: float = TARGET_POWER) -> float:
    """Smallest lambda at which power reaches ``target`` (first crossing, linear interpolation); NaN if never."""
    lams, power = np.asarray(lams, float), np.asarray(power, float)
    hit = np.flatnonzero(power >= target)
    hit = hit[hit > 0]
    if len(hit) == 0:
        return float("nan")
    i = hit[0]
    p0, p1 = power[i - 1], power[i]
    if p1 == p0:
        return float(lams[i])
    return float(lams[i - 1] + (target - p0) * (lams[i] - lams[i - 1]) / (p1 - p0))


def mult_needed(mults, powers, mdes, lam_target: float, target: float = TARGET_POWER,
                extrapolate: bool = True) -> tuple[float, str]:
    """Sample multiplier at which power at ``lam_target`` reaches ``target``.

    powers[k] is the power at lam_target with mults[k]; mdes[k] the minimum detectable lambda there. If the real
    sample (mults[0] = 1) already reaches the target, the multiplier is scaled down as (MDE(1) / lam_target)^2,
    since standard errors shrink like one over the square root of the sample size. Inside the grid it is a linear
    interpolation in the multiplier; beyond the largest multiplier it is scaled up the same way from that MDE.
    """
    mults, powers, mdes = (np.asarray(a, float) for a in (mults, powers, mdes))
    if lam_target <= 0:
        return float("nan"), "target is zero"
    if powers[0] >= target:
        if np.isfinite(mdes[0]):
            return float(mults[0] * (mdes[0] / lam_target) ** 2), "current sample is enough (scaled down)"
        return float(mults[0]), "current sample is enough"
    for k in range(1, len(mults)):
        if powers[k] >= target:
            m = mults[k - 1] + (target - powers[k - 1]) * (mults[k] - mults[k - 1]) / (powers[k] - powers[k - 1])
            return float(m), "simulated (interpolated)"
    if extrapolate and np.isfinite(mdes[-1]):
        return float(mults[-1] * (mdes[-1] / lam_target) ** 2), "extrapolated beyond 3x (1/sqrt(n) scaling)"
    return float("nan"), "not reached within the simulated range"


# ---------------------------------------------------------------------------
# Designs from the real data (the same construction as the original scripts)
# ---------------------------------------------------------------------------


def _sm_fit(y: np.ndarray, X: np.ndarray, names: list, lags: int = LAGS):
    import statsmodels.api as sm
    return sm.OLS(y, pd.DataFrame(X, columns=names)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def make_reg_spec(name: str, y, C, v, mult, tested: str, binary: bool, mag, sources: dict, dates=None,
                  is_expiry=None, control_names=None, lags: int = LAGS) -> dict:
    """Spec of a regression test: fit the full and the null model on the given (real or synthetic) data."""
    y, C, v, mult = (np.asarray(a, float) for a in (y, C, v, mult))
    other, z = gex_cols(v, mult, tested)
    names = (control_names or [f"c{i}" for i in range(C.shape[1])]) + ["other", "tested"]
    full = _sm_fit(y, np.column_stack([C, other, z]), names, lags)
    A0 = np.column_stack([C, other])
    b0 = np.linalg.lstsq(A0, y, rcond=None)[0]
    fnull = A0 @ b0  # the null model: the same regression without the tested term, fitted on the data
    n = len(y)
    # Residuals of the full model: noise without the estimated effect. (The null model's own residuals would
    # contain the real effect and count it as extra noise.)
    return {"kind": "reg", "name": name, "n": n, "y": y, "C": C, "v": v, "mult": mult, "tested": tested,
            "binary": binary, "mag": np.abs(np.asarray(mag, float)), "fnull": fnull, "u": np.asarray(full.resid),
            "est": float(full.params["tested"]), "se": float(full.bse["tested"]), "p": float(full.pvalues["tested"]),
            "params": full.params, "sources": sources, "lags": lags,
            "dates": dates if dates is not None else pd.RangeIndex(n),
            "is_expiry": np.asarray(is_expiry if is_expiry is not None else np.zeros(n), bool)}


def make_fc_spec(y, X0, v, exp, h, n_train: int, sources: dict, dates=None, lags: int = LAGS) -> dict:
    """Spec of the forecast test. h is the shape of the injected effect (lambda = 1 adds h)."""
    y, X0, v, exp, h = (np.asarray(a, float) for a in (y, X0, v, exp, h))
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    fnull = X0 @ b0  # null model: no GEX terms
    X1 = np.column_stack([X0, v, v * exp])
    u = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]  # residuals of the model with GEX (noise without the effect)
    n = len(y)
    spec = {"kind": "fc", "name": "FC", "n": n, "y": y, "X0": X0, "v": v, "mult": exp, "h": h, "n_train": n_train,
            "binary": False, "mag": np.abs(v), "fnull": fnull, "u": u, "est": 1.0, "sources": sources,
            "lags": lags, "dates": dates if dates is not None else pd.RangeIndex(n), "is_expiry": exp > 0}
    real = fc_real(spec, y, v)
    spec.update(real)
    return spec


def fc_real(spec: dict, y: np.ndarray, v: np.ndarray) -> dict:
    """The forecast test on given outcomes (the real data when y is the real outcome)."""
    X0, exp, ntr = spec["X0"], spec["mult"], spec["n_train"]
    X1 = np.column_stack([X0, v, v * exp])
    L = {}
    for k, X in (("base", X0), ("gex", X1)):
        b = np.linalg.lstsq(X[:ntr], y[:ntr], rcond=None)[0]
        L[k] = (y[ntr:] - X[ntr:] @ b) ** 2
    mu, se = mean_hac_batch((L["base"] - L["gex"])[:, None], spec["lags"])
    t = float(mu[0] / se[0])
    return {"real_mse_base": float(L["base"].mean()), "real_mse_gex": float(L["gex"].mean()),
            "real_stat": float(100 * (L["gex"].mean() / L["base"].mean() - 1)), "real_dm_t": t,
            "p": float(stats.norm.sf(t)), "n_test": int(len(y) - ntr)}


def make_r2_spec(name: str, rel: np.ndarray, dates=None) -> dict:
    rel = np.asarray(rel, float)
    n = len(rel)
    mu, se = mean_hac_batch(rel[:, None], LAGS_WEEKS)
    boot = np.random.default_rng(0).choice(rel, (4000, n)).mean(axis=1)  # as in real_day_level.r2_regimes
    return {"kind": "r2", "name": name, "n": n, "u": rel - rel.mean(), "est": float(rel.mean()),
            "se": float(np.std(rel, ddof=1) / np.sqrt(n)), "se_hac": float(se[0]),
            "p": float(stats.wilcoxon(rel).pvalue), "p_hac_t": float(2 * stats.norm.sf(abs(mu[0] / se[0]))),
            "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
            "sources": {"same regime": np.arange(n)}, "dates": dates if dates is not None else pd.RangeIndex(n),
            "is_expiry": np.ones(n, bool), "binary": False}


def r3_design(dm: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """real_day_level.r3_robustness, variant 'share', Pro rule: the same columns, rows and constant handling."""
    d = dm.join(panel, how="left")
    d["y"] = np.log(d["rv_day"])
    d["y_l1"] = d["y"].shift(1)
    d["y_w"] = d["y"].shift(1).rolling(5).mean()
    d["lvix_l1"] = np.log(d["vix"]).shift(1)
    d["exp"] = d["is_expiry"].astype(float)
    for reg in rdl.REGIMES[1:]:
        d[f"reg_{reg}"] = (d["regime"] == reg).astype(float)
        d[f"exp_x_{reg}"] = d[f"reg_{reg}"] * d["exp"]
    reg_cols = [c for c in d.columns if c.startswith("reg_") or c.startswith("exp_x_")]
    share = (d[rdl.RULES[RULE]] / d["gross_gex"]).shift(1)
    X = d[["y_l1", "y_w", "lvix_l1", "exp"] + reg_cols].copy()
    X["gex_l1"] = share
    X["gex_l1_x_exp"] = share * d["exp"]
    ok = X.notna().all(axis=1) & d["y"].notna()
    controls = [c for c in X.columns[:-2] if X.loc[ok, c].std() > 0]
    C = np.column_stack([np.ones(int(ok.sum()))] + [X.loc[ok, c].to_numpy(float) for c in controls])
    return {"d": d[ok], "y": d.loc[ok, "y"].to_numpy(float), "C": C, "names": ["const"] + controls,
            "g": share[ok].to_numpy(float), "exp": d.loc[ok, "exp"].to_numpy(float),
            "regime": d.loc[ok, "regime"].to_numpy(), "dates": d.index[ok]}


def r5_design(dm: pd.DataFrame, panel: pd.DataFrame, bars: Path) -> dict:
    """momentum_gex.py, specification 'binary', Pro rule, all days."""
    d = dm.join(mg.daily_returns(bars, dm.index))
    scale = panel["gross_gex"].rolling(20, min_periods=5).median()
    g = (panel[rdl.RULES[RULE]] / scale).shift(1).reindex(d.index)
    share = (panel[rdl.RULES[RULE]] / panel["gross_gex"]).shift(1).reindex(d.index)
    d = d.assign(g=g, share=share).dropna(subset=["r_rod", "r_last"])
    neg = (d["g"] < 0).astype(float).where(d["g"].notna())
    r, y = d["r_rod"] * 1e4, d["r_last"] * 1e4
    ok = neg.notna() & r.notna() & y.notna()
    n = int(ok.sum())
    return {"y": y[ok].to_numpy(float), "C": np.column_stack([np.ones(n), r[ok].to_numpy(float)]),
            "names": ["const", "r_rod"], "neg": neg[ok].to_numpy(float), "r": r[ok].to_numpy(float),
            "share": d.loc[ok, "share"].to_numpy(float), "regime": d.loc[ok, "regime"].to_numpy(),
            "exp": d.loc[ok, "is_expiry"].to_numpy(bool), "dates": d.index[ok], "n_all_days": len(d)}


def build_specs(run: Path, raw: Path) -> tuple[dict, dict]:
    dm = pd.read_csv(run / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(run / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    specs, info = {}, {}

    # R2
    for name, measure, regime in (("R2_tue_day", "rv_day", "single_weekly_tue"),
                                  ("R2_tue_last_hour", "rv_last_hour", "single_weekly_tue"),
                                  ("R2_thu_day", "rv_day", "single_weekly_thu"),
                                  ("R2_multi_day", "rv_day", "multi_weekly")):
        wr = rdl.week_relative(dm, measure)
        w = wr[wr["regime"] == regime].dropna(subset=["rel"])
        specs[name] = make_r2_spec(name, w["rel"].to_numpy(), dates=pd.DatetimeIndex(w["date"]))
        reg_days = dm.index[dm["regime"] == regime]
        years = (reg_days.max() - reg_days.min()).days / 365.25
        specs[name].update({"measure": measure, "regime": regime, "span_years": years,
                            "units_per_year": len(w) / years})

    # R3a and R3b
    r3 = r3_design(dm, panel)
    tue = np.flatnonzero(r3["regime"] == "single_weekly_tue")
    src = {"all days": np.arange(len(r3["y"])), "Tuesday regime only": tue}
    for name, tested in (("R3a", "level"), ("R3b", "interaction")):
        specs[name] = make_reg_spec(name, r3["y"], r3["C"], r3["g"], r3["exp"], tested, False, r3["g"], src,
                                    dates=r3["dates"], is_expiry=r3["exp"] > 0, control_names=r3["names"])
    # R5
    r5 = r5_design(dm, panel, raw / "nifty_1min.csv")
    tue5 = np.flatnonzero(r5["regime"] == "single_weekly_tue")
    specs["R5"] = make_reg_spec("R5", r5["y"], r5["C"], r5["neg"], r5["r"], "interaction", True, r5["share"],
                                {"all days": np.arange(len(r5["y"])), "Tuesday regime only": tue5},
                                dates=r5["dates"], is_expiry=r5["exp"], control_names=r5["names"])
    specs["R5"]["n_all_days"] = r5["n_all_days"]
    # FC: falsification T4 design (no regime controls), effect shape from the R3 share estimates
    X0 = r3["C"][:, :5]  # const, y_l1, y_w, lvix_l1, exp
    assert r3["names"][:5] == ["const", "y_l1", "y_w", "lvix_l1", "exp"]
    g1, g2 = specs["R3a"]["est"], specs["R3b"]["est"]
    h = g1 * r3["g"] + g2 * r3["g"] * r3["exp"]
    n_train = int((r3["dates"] <= SPLIT_END).sum())
    assert (r3["dates"][:n_train] <= SPLIT_END).all()
    specs["FC"] = make_fc_spec(r3["y"], X0, r3["g"], r3["exp"], h, n_train, src, dates=r3["dates"])
    specs["FC"]["gamma_shape"] = (g1, g2)
    for name in DAILY_TESTS:
        s = specs[name]
        s["span_years"] = (s["dates"].max() - s["dates"].min()).days / 365.25
        s["units_per_year"] = s["n"] / s["span_years"]
    info["dm"], info["panel"] = dm, panel
    return specs, info


# ---------------------------------------------------------------------------
# The simulation engine
# ---------------------------------------------------------------------------


def _design_index(n: int, N: int, src: np.ndarray, block: float, rng) -> np.ndarray:
    """The real rows in their order, followed by N - n extra rows drawn in blocks from ``src``."""
    if N <= n:
        return np.arange(n)
    return np.concatenate([np.arange(n), src[stationary_indices(len(src), N - n, block, rng)]])


def _residuals(u: np.ndarray, idx: np.ndarray, N: int, scheme: str, block: float, rng) -> np.ndarray:
    if scheme == "residual":
        return u[stationary_indices(len(u), N, block, rng)]
    if scheme == "wild":
        return u[idx] * block_signs(N, int(round(block)), rng)
    raise ValueError(scheme)


def _stat_names(spec: dict) -> list:
    return {"reg": ["hac_t"], "fc": ["dm_one_sided"], "r2": ["wilcoxon", "hac_t_mean"]}[spec["kind"]]


def simulate_cell(spec: dict, cell: dict, reps: int, rng) -> dict:
    """Run ``reps`` replications of one cell; returns summed counts per lambda for each statistic."""
    lams = np.asarray(cell["lambdas"], float)
    deltas = lams * spec["est"]
    sgn = np.sign(deltas)
    n = spec["n"]
    N = int(round(cell["m"] * n))
    block = cell["block"]
    src = spec["sources"][cell["source"]]
    nl = len(lams)
    acc = {s: {k: np.zeros(nl) for k in ("rej", "rej_right", "sum_est", "sum_abs_est_rej", "n_ge_real")}
           for s in _stat_names(spec)}
    for _ in range(reps):
        idx = _design_index(n, N, src, block, rng)
        e = _residuals(spec["u"], idx, N, cell["scheme"], block, rng)
        out = {}
        if spec["kind"] == "r2":
            Rm = deltas[None, :] + e[:, None]
            mu, se = mean_hac_batch(Rm, LAGS_WEEKS)
            pw = stats.wilcoxon(Rm, axis=0).pvalue
            out["wilcoxon"] = (mu, pw < ALPHA)
            out["hac_t_mean"] = (mu, np.abs(mu / se) > Z2)
        else:
            vt, mt = spec["v"][idx], spec["mult"][idx]
            flip = flip_mask(N, cell["flip"], cell["pattern"], rng, spec["mag"][idx])
            vm = measured(vt, flip, spec["binary"])
            if spec["kind"] == "reg":
                _, zt = gex_cols(vt, mt, spec["tested"])
                om, zm = gex_cols(vm, mt, spec["tested"])
                X = np.column_stack([spec["C"][idx], om, zm])
                Y = spec["fnull"][idx][:, None] + np.outer(zt, deltas) + e[:, None]
                b, se = ols_hac_batch(X, Y, -1, spec["lags"])
                out["hac_t"] = (b, np.abs(b / se) > Z2)
            else:  # forecast test: training rows are always the real first n_train rows
                ntr = spec["n_train"]
                X0 = spec["X0"][idx]
                X1 = np.column_stack([X0, vm, vm * mt])
                Y = spec["fnull"][idx][:, None] + np.outer(spec["h"][idx], deltas) + e[:, None]
                L = []
                for X in (X0, X1):
                    B = np.linalg.lstsq(X[:ntr], Y[:ntr], rcond=None)[0]
                    L.append((Y[ntr:] - X[ntr:] @ B) ** 2)
                mu, se = mean_hac_batch(L[0] - L[1], spec["lags"])
                pct = 100 * (L[1].mean(axis=0) / L[0].mean(axis=0) - 1)
                out["dm_one_sided"] = (pct, mu / se > Z1)
        for s, (est, rej) in out.items():
            a = acc[s]
            a["rej"] += rej
            if s == "dm_one_sided":
                a["rej_right"] += rej
                a["n_ge_real"] += est >= spec.get("real_stat", np.inf)
            else:
                a["rej_right"] += rej & (np.sign(est) == sgn)
            a["sum_est"] += est
            a["sum_abs_est_rej"] += np.where(rej, np.abs(est), 0.0)
    return acc


# ---------------------------------------------------------------------------
# Cells, tasks and multiprocessing
# ---------------------------------------------------------------------------

_SPECS: dict = {}
_CELLS: list = []


def cell_key(c: dict) -> str:
    return (f"{c['test']}|{c['purpose']}|{c['scheme']}|m{c['m']}|{c['source']}|{c['pattern']}|f{c['flip']:.6f}"
            f"|b{c['block']}")


def cell_entropy(c: dict) -> int:
    return int(hashlib.sha256(cell_key(c).encode()).hexdigest()[:8], 16)


def build_cells(specs: dict, anchors: list) -> list:
    cells = []

    def add(**kw):
        base = {"scheme": "residual", "m": 1.0, "pattern": "none", "flip": 0.0, "lambdas": LAMBDAS}
        base.update(kw)
        if "block" not in kw:
            base["block"] = BLOCK_WEEKS if specs[kw["test"]]["kind"] == "r2" else BLOCK_DAYS
        cells.append(base)

    for scheme in ("residual", "wild"):
        for t in ("R2_tue_day", "R2_tue_last_hour"):
            for m in MULTS:
                add(test=t, purpose="main", scheme=scheme, m=m, source="same regime")
        for t in ("R2_thu_day", "R2_multi_day"):
            add(test=t, purpose="main", scheme=scheme, source="same regime")
        for t in DAILY_TESTS:
            add(test=t, purpose="main", scheme=scheme, source="all days")
            for m in MULTS[1:]:
                for source in ("all days", "Tuesday regime only"):
                    add(test=t, purpose="main", scheme=scheme, m=m, source=source)
    # block-length sensitivity (primary scheme, real sample size)
    for t in DAILY_TESTS:
        for b in (5, 30):
            add(test=t, purpose="block length", source="all days", block=b)
    for b in (2, 8):
        add(test="R2_tue_day", purpose="block length", source="same regime", block=b)
    # sign errors (primary scheme)
    flips = sorted(set(FLIP_GRID) | {round(a, 6) for a in anchors})
    for t in DAILY_TESTS:
        for f in flips:
            for m in MULTS:
                add(test=t, purpose="sign error", source="all days", m=m, pattern="iid", flip=f)
            for pattern in ("spells", "smallest"):
                add(test=t, purpose="sign error", source="all days", pattern=pattern, flip=f)
    return cells


def _task(args):
    ci, chunk, reps = args
    c = _CELLS[ci]
    rng = np.random.default_rng([SEED, cell_entropy(c), chunk])
    return ci, chunk, simulate_cell(_SPECS[c["test"]], c, reps, rng)


def run_cells(specs: dict, cells: list, reps: int, workers: int, chunk: int = 500) -> dict:
    global _SPECS, _CELLS
    _SPECS, _CELLS = specs, cells
    tasks = []
    for ci in range(len(cells)):
        done, k = 0, 0
        while done < reps:
            r = min(chunk, reps - done)
            tasks.append((ci, k, r))
            done += r
            k += 1
    # larger samples first so that the pool stays busy at the end
    tasks.sort(key=lambda t: -cells[t[0]]["m"])
    parts: dict = {}
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover
        ctx = mp.get_context()
    with ctx.Pool(workers) as pool:
        for ci, chunk, acc in pool.imap_unordered(_task, tasks):
            parts[(ci, chunk)] = acc
    # Sum the chunks of each cell in chunk order, so that reruns give identical files
    results: dict = {}
    for ci, chunk in sorted(parts):
        acc = parts[(ci, chunk)]
        if ci not in results:
            results[ci] = acc
        else:
            for s in acc:
                for k in acc[s]:
                    results[ci][s][k] += acc[s][k]
    return results


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def curves_frame(specs: dict, cells: list, results: dict, reps: int) -> pd.DataFrame:
    rows = []
    for ci, c in enumerate(cells):
        sp = specs[c["test"]]
        n = sp["n"]
        N = int(round(c["m"] * n))
        n_exp_real = int(sp["is_expiry"].sum())
        for s, a in results[ci].items():
            p = a["rej"] / reps
            for i, lam in enumerate(c["lambdas"]):
                delta = lam * sp["est"]
                rows.append({
                    "test": c["test"], "stat": s, "purpose": c["purpose"], "scheme": c["scheme"], "m": c["m"],
                    "future_days_from": c["source"] if c["m"] > 1 else "none (real sample)",
                    "sign_error_pattern": c["pattern"], "wrong_sign_share": c["flip"], "block": c["block"],
                    "reps": reps, "n_units": N, "n_expiries_expected": n_exp_real * N / n,
                    "lambda": lam, "delta": delta, "power": p[i], "mc_se": np.sqrt(p[i] * (1 - p[i]) / reps),
                    "power_right_sign": a["rej_right"][i] / reps, "mean_estimate": a["sum_est"][i] / reps,
                    "exaggeration_ratio": (a["sum_abs_est_rej"][i] / a["rej"][i] / abs(delta)
                                           if a["rej"][i] > 0 and delta != 0 and s != "dm_one_sided" else np.nan),
                    "share_null_mse_change_at_least_real": (a["n_ge_real"][i] / reps if s == "dm_one_sided" else np.nan),
                    "seed_entropy": f"[{SEED}, {cell_entropy(c)}, chunk 0 to {int(np.ceil(reps / 500)) - 1}]"})
    return pd.DataFrame(rows)


GROUP = ["test", "stat", "purpose", "scheme", "m", "future_days_from", "sign_error_pattern", "wrong_sign_share",
         "block"]


def mde_frame(curves: pd.DataFrame, specs: dict) -> pd.DataFrame:
    rows = []
    for key, g in curves.groupby(GROUP, sort=False):
        g = g.sort_values("lambda")
        rec = dict(zip(GROUP, key))
        sp = specs[rec["test"]]
        lam, pw = g["lambda"].to_numpy(), g["power"].to_numpy()
        mde = mde_from_curve(lam, pw)
        # Monte Carlo error of the MDE (delta method): MC error of power at the MDE / slope of the curve there
        mc = np.nan
        if np.isfinite(mde):
            i = int(np.searchsorted(lam, mde))
            i = min(max(i, 1), len(lam) - 1)
            slope = (pw[i] - pw[i - 1]) / (lam[i] - lam[i - 1])
            if slope > 0:
                mc = np.sqrt(TARGET_POWER * (1 - TARGET_POWER) / g["reps"].iloc[0]) / slope * abs(sp["est"])
        rec.update({
            "label": LABELS[rec["test"]], "n_units": int(g["n_units"].iloc[0]),
            "n_expiries_expected": g["n_expiries_expected"].iloc[0], "estimate": sp["est"],
            "mde80_lambda": mde, "mde80": mde * abs(sp["est"]) if np.isfinite(mde) else np.nan,
            "mde80_mc_se": mc,
            "size_at_zero": pw[0], "power_at_estimate": interp_power(lam, pw, 1.0),
            "power_at_estimate_mc_se": float(np.sqrt(interp_power(lam, pw, 1.0) * (1 - interp_power(lam, pw, 1.0))
                                                     / g["reps"].iloc[0])),
            "power_right_sign_at_estimate": interp_power(lam, g["power_right_sign"].to_numpy(), 1.0),
            "power_at_half_estimate": interp_power(lam, pw, 0.5),
            "power_at_twice_estimate": interp_power(lam, pw, 2.0),
            "exaggeration_at_half_estimate": float(g.loc[np.isclose(g["lambda"], 0.5), "exaggeration_ratio"].iloc[0]),
            "exaggeration_at_estimate": float(g.loc[np.isclose(g["lambda"], 1.0), "exaggeration_ratio"].iloc[0]),
        })
        if sp["kind"] != "fc":
            se = sp["se_hac"] if (sp["kind"] == "r2" and rec["stat"] == "hac_t_mean") else sp["se"]
            rec["analytic_mde80_2p8_se"] = (Z2 + stats.norm.ppf(TARGET_POWER)) * se / np.sqrt(rec["m"])
            rec["mde80_times_sqrt_m"] = rec["mde80"] * np.sqrt(rec["m"]) if np.isfinite(mde) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def estimates_frame(specs: dict, run: Path, published: dict) -> pd.DataFrame:
    rows = []
    for name, sp in specs.items():
        rec = {"test": name, "label": LABELS[name], "n_units": sp["n"], "n_expiries": int(sp["is_expiry"].sum()),
               "first_date": str(pd.Timestamp(sp["dates"].min()).date()),
               "last_date": str(pd.Timestamp(sp["dates"].max()).date()),
               "span_years": sp["span_years"], "units_per_year": sp["units_per_year"]}
        if sp["kind"] == "fc":
            rec.update({"estimate": 1.0, "real_mse_without_gex": sp["real_mse_base"],
                        "real_mse_with_gex": sp["real_mse_gex"], "real_pct_change_mse": sp["real_stat"],
                        "real_dm_t": sp["real_dm_t"], "p": sp["p"], "n_test_days": sp["n_test"],
                        "n_train_days": sp["n_train"], "effect_shape_gex_l1": sp["gamma_shape"][0],
                        "effect_shape_gex_l1_x_exp": sp["gamma_shape"][1]})
        else:
            rec.update({"estimate": sp["est"], "se": sp["se"], "p": sp["p"]})
            if sp["kind"] == "r2":
                rec.update({"ci95_low": sp["ci_low"], "ci95_high": sp["ci_high"], "ci_method": "iid bootstrap of the mean, "
                            "4000 draws, seed 0 (as in real_day_level.py)", "se_hac_4_lags": sp["se_hac"],
                            "p_hac_t": sp["p_hac_t"]})
            else:
                rec.update({"ci95_low": sp["est"] - Z2 * sp["se"], "ci95_high": sp["est"] + Z2 * sp["se"],
                            "ci_method": "estimate +/- 1.96 Newey-West standard errors"})
            u = sp["u"]
            ex = sp["is_expiry"]
            rec.update({"resid_sd_all": float(u.std()), "resid_skew": float(stats.skew(u)),
                        "resid_sd_expiry": float(u[ex].std()) if ex.any() and not ex.all() else np.nan,
                        "resid_sd_other": float(u[~ex].std()) if ex.any() and not ex.all() else np.nan,
                        "resid_acf1": float(np.corrcoef(u[1:], u[:-1])[0, 1]),
                        "resid_sq_acf1": float(np.corrcoef(u[1:] ** 2, u[:-1] ** 2)[0, 1])})
            if sp["kind"] == "reg":
                # Does the size of the noise move with the tested regressor? (The residual scheme breaks that link,
                # the wild scheme keeps it.)
                z = gex_cols(sp["v"], sp["mult"], sp["tested"])[1]
                rec["corr_sq_resid_sq_tested_regressor"] = float(np.corrcoef(u ** 2, z ** 2)[0, 1])
                if sp["binary"]:
                    rec["corr_sq_resid_sq_multiplier"] = float(np.corrcoef(u ** 2, sp["mult"] ** 2)[0, 1])
            try:
                from arch.bootstrap import optimal_block_length
                ob = optimal_block_length(u ** 2)
                rec["politis_white_block_sq_resid"] = float(ob["stationary"].iloc[0])
                rec["politis_white_block_resid"] = float(optimal_block_length(u)["stationary"].iloc[0])
            except Exception:  # pragma: no cover - arch is optional
                pass
        pub = published.get(name)
        if pub is not None:
            rec["published_value"], rec["published_p"] = pub
            val = rec["real_pct_change_mse"] if sp["kind"] == "fc" else rec["estimate"]
            rec["matches_published"] = bool(np.isclose(val, pub[0], rtol=1e-6, atol=1e-9)
                                            and np.isclose(rec["p"], pub[1], rtol=1e-4, atol=1e-9))
        rows.append(rec)
    return pd.DataFrame(rows)


def published_values(root: Path) -> dict:
    """The same numbers as written by the original scripts (for a reproduction check)."""
    out = {}
    try:
        r2 = pd.read_csv(root / "results/real_day_level/r2_expiry_effect_by_regime.csv")
        for name, meas, reg in (("R2_tue_day", "rv_day", "single_weekly_tue"),
                                ("R2_tue_last_hour", "rv_last_hour", "single_weekly_tue"),
                                ("R2_thu_day", "rv_day", "single_weekly_thu"),
                                ("R2_multi_day", "rv_day", "multi_weekly")):
            r = r2[(r2["measure"] == meas) & (r2["regime"] == reg)].iloc[0]
            out[name] = (float(r["mean_rel"]), float(r["wilcoxon_p_rel_eq_0"]))
        r3 = pd.read_csv(root / "results/real_day_level/r3_robustness.csv")
        for name, term in (("R3a", "gex_l1"), ("R3b", "gex_l1_x_exp")):
            r = r3[(r3["rule"] == RULE) & (r3["variant"] == "share") & (r3["term"] == term)].iloc[0]
            out[name] = (float(r["coef"]), float(r["p"]))
        r5 = pd.read_csv(root / "results/momentum_gex/regressions.csv")
        r = r5[(r5["spec"] == "binary") & (r5["rule"] == RULE) & (r5["subset"] == "all")
               & (r5["term"] == "r_rod_x_neg")].iloc[0]
        out["R5"] = (float(r["coef"]), float(r["p"]))
        f4 = pd.read_csv(root / "results/falsification/t4_oos_forecasts.csv")
        r = f4[(f4["sample"] == "full") & (f4["finding"] == "F1") & (f4["measure"] == "share")
               & (f4["rule"] == RULE) & (f4["design"] == "fit 2023 to 2024, test 2025 onward")].iloc[0]
        out["FC"] = (float(r["pct_change_mse"]), float(r["dm_p_one_sided_gex_better"]))
    except (FileNotFoundError, IndexError, KeyError):  # pragma: no cover
        pass
    return out


def sign_anchors(panel: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Observed sign disagreement between the three fixed rules (panel) and between the estimated dealer sign
    and the fixed rules (results/dealer_sign/sign_agreement.csv)."""
    cols = {"US convention": "net_gex_us_convention", "Pro": "net_gex_participant",
            "Client mirror": "net_gex_client_mirror"}
    rows = []
    periods = {"full sample": np.ones(len(panel), bool), "2023 to 2024": panel.index <= SPLIT_END,
               "2025 onward": panel.index > SPLIT_END}
    names = list(cols)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = np.sign(panel[cols[names[i]]]), np.sign(panel[cols[names[j]]])
            for pname, pm in periods.items():
                ok = pm & a.notna().to_numpy() & b.notna().to_numpy()
                rows.append({"source": "daily_gex_panel.csv (fixed rules)", "pair": f"{names[i]} vs {names[j]}",
                             "period": pname, "days": int(ok.sum()),
                             "share_days_different_sign": float((a[ok] != b[ok]).mean())})
    try:
        sa = pd.read_csv(root / "results/dealer_sign/sign_agreement.csv")
        sa = sa[(sa["family"] == "participants") & (sa["quarter"] == "all")]
        for _, r in sa.iterrows():
            rows.append({"source": "dealer_sign/sign_agreement.csv (estimated participant sign)",
                         "pair": f"estimated vs {r['fixed_rule']}", "period": r["period"], "days": int(r["n_days"]),
                         "share_days_different_sign": float(1 - r["share_same_sign"])})
    except FileNotFoundError:  # pragma: no cover
        pass
    out = pd.DataFrame(rows)
    out["used_in_simulation"] = False
    use = (out["period"] == "full sample") | (out["pair"] == "estimated vs pro")
    out.loc[use, "used_in_simulation"] = True
    short = {"training 2023 to 2024 (direction from that period)": "2023 to 2024", "test, expanding refits": "2025 on"}
    out["anchor_label"] = [p.replace("estimated vs pro", "estimated sign vs Pro") + f" ({short.get(q, q)})"
                           for p, q in zip(out["pair"], out["period"])]
    out["share_rounded_for_simulation"] = out["share_days_different_sign"].round(3)
    return out


def needed_frame(mde: pd.DataFrame, curves: pd.DataFrame, specs: dict, est: pd.DataFrame) -> pd.DataFrame:
    """Sample needed for 80% power at the estimate, half the estimate and the confidence bound nearest zero."""
    rows = []
    jobs = [("R2_tue_day", "wilcoxon"), ("R2_tue_day", "hac_t_mean"), ("R2_tue_last_hour", "wilcoxon"),
            ("R2_tue_last_hour", "hac_t_mean"), ("R3a", "hac_t"), ("R3b", "hac_t"), ("R5", "hac_t"),
            ("FC", "dm_one_sided")]
    for test, stat in jobs:
        sp = specs[test]
        src = "same regime" if sp["kind"] == "r2" else "all days"
        for scheme in ("residual", "wild"):
            for source in ([src] if sp["kind"] == "r2" else [src, "Tuesday regime only"]):
                sel = curves[(curves["test"] == test) & (curves["stat"] == stat) & (curves["purpose"] == "main")
                             & (curves["scheme"] == scheme)
                             & ((curves["m"] == 1.0) | (curves["future_days_from"] == source))]
                mrows = mde[(mde["test"] == test) & (mde["stat"] == stat) & (mde["purpose"] == "main")
                            & (mde["scheme"] == scheme) & ((mde["m"] == 1.0) | (mde["future_days_from"] == source))]
                mrows = mrows.sort_values("m")
                targets = {"estimate": 1.0, "half the estimate": 0.5}
                e = est[est["test"] == test].iloc[0]
                if sp["kind"] != "fc" and np.sign(e["ci95_low"]) == np.sign(e["ci95_high"]):
                    near = e["ci95_low"] if sp["est"] > 0 else e["ci95_high"]
                    targets["95% bound nearest zero"] = float(near / sp["est"])
                for tname, lam_t in targets.items():
                    pw = []
                    for m in MULTS:
                        g = sel[sel["m"] == m].sort_values("lambda")
                        pw.append(interp_power(g["lambda"].to_numpy(), g["power"].to_numpy(), lam_t))
                    m_need, method = mult_needed(MULTS, pw, mrows["mde80_lambda"].to_numpy(), lam_t,
                                                 extrapolate=sp["kind"] != "fc")
                    n_exp = int(sp["is_expiry"].sum())
                    rec = {"test": test, "stat": stat, "scheme": scheme, "future_days_from": source,
                           "target": tname, "target_lambda": lam_t, "target_delta": lam_t * sp["est"],
                           **{f"power_at_{m:g}x": p for m, p in zip(MULTS, pw)},
                           "multiplier_needed": m_need, "method": method, "n_units_real": sp["n"],
                           "expiries_real": n_exp}
                    if np.isfinite(m_need):
                        extra = max(m_need - 1.0, 0.0)
                        rec["units_needed"] = m_need * sp["n"]
                        rec["expiries_needed"] = m_need * n_exp
                        rec["extra_expiries"] = extra * n_exp
                        rec["extra_years"] = extra * sp["n"] / sp["units_per_year"]
                    rows.append(rec)
    return pd.DataFrame(rows)


HEADLINE = [("R2_tue_day", "wilcoxon"), ("R2_tue_day", "hac_t_mean"), ("R2_tue_last_hour", "wilcoxon"),
            ("R2_thu_day", "wilcoxon"), ("R2_multi_day", "wilcoxon"), ("R3a", "hac_t"), ("R3b", "hac_t"),
            ("R5", "hac_t"), ("FC", "dm_one_sided")]


def headline_frame(est: pd.DataFrame, mde: pd.DataFrame, size: pd.DataFrame, need: pd.DataFrame) -> pd.DataFrame:
    """One row per key test at the real sample size. 'Conservative' = the less favourable of the two resampling
    schemes (larger MDE, lower power, larger sample needed), so that no scheme is picked after seeing results."""
    rows = []
    for test, stat in HEADLINE:
        e = est[est["test"] == test].iloc[0]
        rec = {"test": test, "stat": stat, "label": LABELS[test], "estimate": e["estimate"],
               "ci95_low": e.get("ci95_low"), "ci95_high": e.get("ci95_high"), "p_real": e["p"],
               "n_units": int(e["n_units"]), "n_expiries": int(e["n_expiries"])}
        by = {}
        for sc in ("residual", "wild"):
            r = mde[(mde["test"] == test) & (mde["stat"] == stat) & (mde["purpose"] == "main")
                    & (mde["scheme"] == sc) & (mde["m"] == 1.0)].iloc[0]
            z = size[(size["test"] == test) & (size["stat"] == stat) & (size["purpose"] == "main")
                     & (size["scheme"] == sc) & (size["m"] == 1.0)].iloc[0]
            by[sc] = r
            rec[f"size_{sc}"] = z["rejection_rate"]
            rec[f"mde80_{sc}"] = r["mde80"]
            rec[f"power_at_estimate_{sc}"] = r["power_at_estimate"]
            rec[f"power_at_half_estimate_{sc}"] = r["power_at_half_estimate"]
        worst = max(by, key=lambda k: (by[k]["mde80"] if np.isfinite(by[k]["mde80"]) else np.inf))
        rec["conservative_scheme"] = worst
        rec["mde80_conservative"] = by[worst]["mde80"]
        rec["mde80_over_abs_estimate_conservative"] = by[worst]["mde80_lambda"]
        rec["analytic_mde80_2p8_se"] = by["residual"].get("analytic_mde80_2p8_se", np.nan)
        rec["power_at_estimate_conservative"] = min(by[k]["power_at_estimate"] for k in by)
        rec["power_at_half_estimate_conservative"] = min(by[k]["power_at_half_estimate"] for k in by)
        rec["exaggeration_at_half_estimate_conservative"] = max(by[k]["exaggeration_at_half_estimate"] for k in by)
        if test.startswith("R2"):
            # R2 effects are in log points of realized variance; exp(x) - 1 is the percentage difference
            for col in ("estimate", "ci95_low", "ci95_high", "mde80_residual", "mde80_wild", "mde80_conservative"):
                rec[f"{col}_as_pct_more_variance"] = 100 * (np.exp(rec[col]) - 1)
        nd = need[(need["test"] == test) & (need["stat"] == stat)
                  & need["future_days_from"].isin(["same regime", "all days"])]
        for tname, short in (("half the estimate", "half"), ("95% bound nearest zero", "ci_bound"),
                             ("estimate", "estimate")):
            g = nd[nd["target"] == tname]
            if g.empty:
                continue
            k = g["multiplier_needed"].fillna(np.inf).idxmax()
            row = g.loc[k]
            rec[f"multiplier_needed_{short}"] = row["multiplier_needed"]
            rec[f"expiries_needed_{short}"] = row.get("expiries_needed", np.nan)
            rec[f"extra_expiries_{short}"] = row.get("extra_expiries", np.nan)
            rec[f"extra_years_{short}"] = row.get("extra_years", np.nan)
            rec[f"method_{short}"] = row["method"]
            rec[f"scheme_{short}"] = row["scheme"]
        rows.append(rec)
    return pd.DataFrame(rows)


def sign_error_frame(mde: pd.DataFrame) -> pd.DataFrame:
    base = mde[(mde["purpose"] == "main") & (mde["scheme"] == "residual") & (mde["m"] == 1.0)
               & mde["test"].isin(DAILY_TESTS)].assign(sign_error_pattern="none", wrong_sign_share=0.0)
    se = mde[mde["purpose"] == "sign error"]
    keep = ["test", "stat", "sign_error_pattern", "wrong_sign_share", "m", "n_units", "size_at_zero",
            "power_at_estimate", "power_right_sign_at_estimate", "power_at_twice_estimate", "mde80_lambda", "mde80"]
    out = pd.concat([base[keep], se[keep]], ignore_index=True)
    out["attenuation_1_minus_2f"] = 1 - 2 * out["wrong_sign_share"]
    return out.sort_values(["test", "sign_error_pattern", "m", "wrong_sign_share"]).reset_index(drop=True)


def size_frame(curves: pd.DataFrame) -> pd.DataFrame:
    z = curves[curves["lambda"] == 0].copy()
    n = z["reps"]
    p = z["power"]
    # Wilson 95% interval for the rejection rate
    c = Z2
    centre = (p + c ** 2 / (2 * n)) / (1 + c ** 2 / n)
    half = c * np.sqrt(p * (1 - p) / n + c ** 2 / (4 * n ** 2)) / (1 + c ** 2 / n)
    z["rejection_rate"] = p
    z["wilson95_low"], z["wilson95_high"] = centre - half, centre + half
    z["nominal"] = ALPHA
    z["nominal_inside_interval"] = (z["wilson95_low"] <= ALPHA) & (ALPHA <= z["wilson95_high"])
    keep = GROUP + ["n_units", "reps", "rejection_rate", "mc_se", "wilson95_low", "wilson95_high", "nominal",
                    "nominal_inside_interval", "share_null_mse_change_at_least_real"]
    return z[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

SEQ = {1.0: "#6da7ec", 1.5: "#2a78d6", 2.0: "#1c5cab", 3.0: "#0d366b"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def plot_curves(curves: pd.DataFrame, est: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("R2_tue_day", "wilcoxon", "same regime"), ("R2_tue_last_hour", "wilcoxon", "same regime"),
              ("R3a", "hac_t", "all days"), ("R3b", "hac_t", "all days"), ("R5", "hac_t", "all days"),
              ("FC", "dm_one_sided", "all days")]
    titles = {"R2_tue_day": "R2: Tuesday expiry premium, whole day", "R2_tue_last_hour": "R2: Tuesday expiry premium, last hour",
              "R3a": "R3: ordinary-day GEX term (share, Pro)", "R3b": "R3: expiry-day GEX interaction (share, Pro)",
              "R5": "R5: reversal x [GEX < 0] (Pro)", "FC": "Forecast gain from GEX (one-sided DM)"}
    xlab = {"R2_tue_day": "true premium (log points)", "R2_tue_last_hour": "true premium (log points)",
            "R3a": "true coefficient", "R3b": "true coefficient", "R5": "true coefficient",
            "FC": "true effect, multiples of the in-sample R3 estimate"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), facecolor="#fcfcfb")
    for ax, (test, stat, source) in zip(axes.ravel(), panels):
        ax.set_facecolor("#fcfcfb")
        e = est[est["test"] == test].iloc[0]
        sel = curves[(curves["test"] == test) & (curves["stat"] == stat) & (curves["purpose"] == "main")
                     & (curves["scheme"] == "residual")]
        xmax = 2.0 if test != "FC" else 3.0
        if test != "FC":
            lo, hi = sorted([e["ci95_low"], e["ci95_high"]])
            ax.axvspan(lo, hi, color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.axvline(e["estimate"], color=INK2, lw=1.2, ls="--", zorder=1)
        ax.axhline(TARGET_POWER, color=MUTED, lw=0.8, zorder=1)
        ax.axhline(ALPHA, color=MUTED, lw=0.8, ls=":", zorder=1)
        for m in MULTS:
            g = sel[(sel["m"] == m) & ((sel["future_days_from"] == source) | (m == 1.0))].sort_values("lambda")
            g = g[g["lambda"] <= xmax]
            ax.plot(g["delta"], g["power"], color=SEQ[m], lw=2.6 if m == 1.0 else 1.8, zorder=3,
                    label="real sample" if m == 1.0 else f"{m:g} x sample")
        w = curves[(curves["test"] == test) & (curves["stat"] == stat) & (curves["purpose"] == "main")
                   & (curves["scheme"] == "wild") & (curves["m"] == 1.0)].sort_values("lambda")
        w = w[w["lambda"] <= xmax]
        ax.plot(w["delta"], w["power"], color=CAT[1], lw=1.6, ls="--", zorder=4, label="real sample, wild bootstrap")
        ax.set_ylim(0, 1.02)
        ax.set_title(titles[test], fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel(xlab[test], fontsize=9, color=INK2)
        ax.set_ylabel("power (share of simulations rejecting)", fontsize=9, color=INK2)
        ax.tick_params(colors=INK2, labelsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.text(e["estimate"], 0.03, " estimate", color=INK2, fontsize=8, ha="left", va="bottom")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="center right")
    fig.suptitle("Power of the main tests: SIMULATIONS CALIBRATED ON REAL DATA (real regressors, block-resampled "
                 "real residuals).\nBlue: residual block bootstrap at 1x to 3x the sample; orange dashed: wild block "
                 "bootstrap, real sample. Grey line at 0.8 = 80% power; dotted at 0.05 = size;\nvertical dashed line "
                 "= real estimate; grey band = its 95% interval.", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_sign_error(sign: pd.DataFrame, anchors: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 6.0), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    names = {"R3a": "R3 ordinary-day term", "R3b": "R3 expiry-day interaction", "R5": "R5 reversal interaction",
             "FC": "forecast gain (one-sided test, any sign)"}
    for col, t in zip(CAT, DAILY_TESTS):
        g = sign[(sign["test"] == t) & (sign["m"] == 1.0)
                 & sign["sign_error_pattern"].isin(["none", "iid"])].sort_values("wrong_sign_share")
        y = g["power_at_estimate"] if t == "FC" else g["power_right_sign_at_estimate"]
        ax.plot(g["wrong_sign_share"], y, color=col, lw=2, marker="o", ms=5, label=names[t])
    used = anchors[anchors["used_in_simulation"]]
    for _, r in used.iterrows():
        x = r["share_rounded_for_simulation"]
        ax.axvline(x, color=MUTED, lw=0.7, ls=":")
        ax.text(x, 0.99, r["anchor_label"] + " ", rotation=90, fontsize=7.5, color=INK2, va="top", ha="right")
    ax.axhline(TARGET_POWER, color=MUTED, lw=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("share of days on which the GEX regressor has the wrong sign (random days)", fontsize=9, color=INK2)
    ax.set_ylabel("power at the estimated effect, correct sign", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.set_title("Sign errors in the GEX regressor (SIMULATION CALIBRATED ON REAL DATA, real sample size)",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/power_real")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--plots-only", action="store_true", help="redraw the two figures from the CSVs in --out")
    a = ap.parse_args()
    if a.plots_only:
        o = Path(a.out)
        plot_curves(pd.read_csv(o / "power_curves.csv"), pd.read_csv(o / "estimates.csv"), o / "power_curves.png")
        plot_sign_error(pd.read_csv(o / "sign_error.csv"), pd.read_csv(o / "sign_disagreement_anchors.csv"),
                        o / "power_sign_error.png")
        return
    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    run, raw, out = Path(a.run), Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    specs, info = build_specs(run, raw)
    pub = published_values(root)
    est = estimates_frame(specs, run, pub)
    est.to_csv(out / "estimates.csv", index=False)
    anchors = sign_anchors(info["panel"], root)
    anchors.to_csv(out / "sign_disagreement_anchors.csv", index=False)
    anchor_vals = anchors.loc[anchors["used_in_simulation"], "share_rounded_for_simulation"].tolist()
    cells = build_cells(specs, anchor_vals)
    say("SIMULATIONS CALIBRATED ON REAL DATA (P71). Real regressors and real residuals, simulated outcomes.")
    say(f"Replications per cell: {a.reps}; cells: {len(cells)}; workers: {a.workers}; master seed {SEED}.")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 4):
        say("\n== Real-data estimates (estimates.csv)")
        say(est[["test", "n_units", "n_expiries", "estimate", "se", "ci95_low", "ci95_high", "p",
                 "matches_published"]].to_string(index=False))

    t1 = time.time()
    results = run_cells(specs, cells, a.reps, a.workers)
    sim_seconds = time.time() - t1
    curves = curves_frame(specs, cells, results, a.reps)
    curves.to_csv(out / "power_curves.csv", index=False)
    mde = mde_frame(curves, specs)
    mde.to_csv(out / "mde.csv", index=False)
    size = size_frame(curves)
    size.to_csv(out / "size.csv", index=False)
    need = needed_frame(mde, curves, specs, est)
    need.to_csv(out / "sample_needed.csv", index=False)
    sign = sign_error_frame(mde)
    sign.to_csv(out / "sign_error.csv", index=False)
    head = headline_frame(est, mde, size, need)
    head.to_csv(out / "headline.csv", index=False)
    plot_curves(curves, est, out / "power_curves.png")
    plot_sign_error(sign, anchors, out / "power_sign_error.png")

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 4,
                           "display.max_rows", 400):
        say("\n== Headline, real sample size, conservative of the two schemes (headline.csv)")
        say(head[["test", "stat", "estimate", "p_real", "size_residual", "size_wild", "mde80_residual", "mde80_wild",
                  "power_at_estimate_conservative", "power_at_half_estimate_conservative",
                  "multiplier_needed_half", "extra_expiries_half", "extra_years_half"]].to_string(index=False))
        say("\n== Size: rejection rate when the true effect is zero (size.csv), main cells")
        sm_ = size[size["purpose"] == "main"]
        say(sm_[["test", "stat", "scheme", "m", "future_days_from", "rejection_rate", "wilson95_low",
                 "wilson95_high"]].to_string(index=False))
        say("\n== Minimum detectable effect at 80% power, real sample, primary scheme (mde.csv)")
        k = mde[(mde["purpose"] == "main") & (mde["m"] == 1.0)]
        say(k[["test", "stat", "scheme", "estimate", "mde80", "mde80_lambda", "mde80_mc_se", "analytic_mde80_2p8_se",
               "power_at_estimate", "power_at_half_estimate", "exaggeration_at_half_estimate"]].to_string(index=False))
        say("\n== MDE by sample size, primary scheme (mde.csv)")
        k = mde[(mde["purpose"] == "main") & (mde["scheme"] == "residual")]
        say(k[["test", "stat", "m", "future_days_from", "n_units", "mde80", "mde80_lambda", "power_at_estimate",
               "power_at_half_estimate"]].to_string(index=False))
        say("\n== Block length sensitivity (mde.csv, purpose 'block length')")
        k = mde[mde["purpose"] == "block length"]
        say(k[["test", "stat", "block", "size_at_zero", "mde80_lambda", "power_at_estimate"]].to_string(index=False))
        say("\n== Sample needed for 80% power (sample_needed.csv)")
        say(need[["test", "scheme", "future_days_from", "target", "target_delta", "power_at_1x", "power_at_3x",
                  "multiplier_needed", "expiries_needed", "extra_expiries", "extra_years", "method"]].to_string(index=False))
        say("\n== Sign-error anchors (sign_disagreement_anchors.csv)")
        say(anchors.to_string(index=False))
        say("\n== Sign errors (sign_error.csv), real sample")
        k = sign[sign["m"] == 1.0]
        say(k[["test", "sign_error_pattern", "wrong_sign_share", "size_at_zero", "power_at_estimate",
               "power_right_sign_at_estimate", "power_at_twice_estimate", "mde80_lambda"]].to_string(index=False))
        say("\n== Sign errors at larger samples, random days (sign_error.csv)")
        k = sign[(sign["sign_error_pattern"] == "iid") & (sign["m"] > 1)]
        say(k[["test", "wrong_sign_share", "m", "power_at_estimate", "power_right_sign_at_estimate",
               "mde80_lambda"]].to_string(index=False))
    seconds = time.time() - t0
    say(f"\nRun time: {seconds:.0f} seconds ({sim_seconds:.0f} in the simulation) on {a.workers} workers.")
    import matplotlib
    import scipy
    import statsmodels
    run_info = {
        "label": "SIMULATIONS CALIBRATED ON REAL DATA",
        "master_seed": SEED,
        "cell_seeds": "numpy default_rng([master_seed, cell hash, chunk]); cell hash = first 8 hex digits of the "
                      "SHA-256 of the cell key (seed_entropy column of power_curves.csv); chunks of 500 replications",
        "r2_ci_bootstrap_seed": 0,
        "replications_per_cell": a.reps, "cells": len(cells), "workers": a.workers,
        "lambdas": LAMBDAS.tolist(), "sample_multipliers": list(MULTS), "alpha": ALPHA, "target_power": TARGET_POWER,
        "newey_west_lags_daily": LAGS, "newey_west_lags_weekly": LAGS_WEEKS,
        "mean_block_days": BLOCK_DAYS, "mean_block_weeks": BLOCK_WEEKS, "spell_days": SPELL_DAYS,
        "flip_shares": sorted(set(FLIP_GRID) | set(anchor_vals)), "rule": RULE, "forecast_split": str(SPLIT_END.date()),
        "seconds_total": round(seconds, 1), "seconds_simulation": round(sim_seconds, 1),
        "inputs": {p.name: _sha(p) for p in (run / "daily_market.csv", run / "daily_gex_panel.csv",
                                               raw / "nifty_1min.csv") if p.exists()},
        "python": platform.python_version(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__},
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    (out.parent / "power_real_stdout.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
