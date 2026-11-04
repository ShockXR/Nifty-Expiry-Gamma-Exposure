"""Multiplicative component GARCH (mcsGARCH) for 1-minute returns.

Model (Engle and Sokalska 2012, as used by Chhimwal, Pandey and Pandey 2025):

    r_{t,i} = mu_{t,i} + eps_{t,i}
    eps_{t,i} = sqrt(h_t * s_i * q_{t,i}) * z_{t,i},     z iid, mean 0, variance 1

* h_t     daily variance, forecast before day t opens (here from an
          AR(1)-EGARCH(1,1) on daily returns, as in the paper's eq. 4 to 6).
* s_i     diurnal variance pattern for minute-of-day i, estimated as the
          average over days of eps^2 / h_t (paper eq. 7).
* q_{t,i} stochastic intraday variance, a GARCH(1,1) on the normalised
          residual x = eps / sqrt(h_t s_i) (paper eq. 8 and 9).

Notation note: the paper writes eq. 3 as a product of volatilities (q, sigma,
s) but eq. 7 defines s_i on the variance scale. This module works on the
variance scale throughout, so the normalisation divides by sqrt(h_t s_i).
That matches Engle and Sokalska's own definition.

The intraday GARCH recursion runs across day boundaries (the last minute of
day t-1 feeds the first minute of day t), which is the rugarch behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Daily component
# ---------------------------------------------------------------------------


def _ewma_variance(y, lam=0.94, n_init=None):
    """One-step-ahead EWMA variance (RiskMetrics), used as a fallback.

    Started at the mean square of the first ``n_init`` observations and
    floored at 1% of the overall mean square, so it is never zero.
    """
    y = np.asarray(y, float)
    ms = float(np.mean(y**2)) or 1e-20
    h = np.empty_like(y)
    h[0] = float(np.mean(y[: n_init or min(len(y), 20)] ** 2)) or ms
    for t in range(1, len(y)):
        h[t] = lam * h[t - 1] + (1 - lam) * y[t - 1] ** 2
    return np.maximum(h, 0.01 * ms)


def _egarch_ok(params, h, y_train) -> bool:
    """Sanity checks on a fitted AR(1)-EGARCH(1,1) and its filtered variance."""
    p = params
    if not np.all(np.isfinite(p.values)):
        return False
    if abs(p.get("y[1]", 0.0)) >= 1 or not (0 <= p.get("beta[1]", 0.5) < 1):
        return False
    v = np.var(y_train)
    lh = np.log(h[np.isfinite(h)])
    return bool(np.all(np.abs(lh - np.log(v)) < 5))


def daily_egarch_variance(
    daily_returns,
    train_size: int | None = None,
    dist: str = "t",
    scale: float = 100.0,
):
    """One-step-ahead daily variance h_t from an AR(1)-EGARCH(1,1).

    The model is fitted on the first ``train_size`` days (all days if None),
    then the fitted parameters are held fixed and the filter is run over the
    whole series, so h_t for every day uses only information up to day t-1.

    On short samples the EGARCH likelihood is flat and the optimiser can
    diverge (seen in the power study, see ITERATION_LOG.md). The fit is
    therefore checked; on failure it is retried with normal errors, and if
    that also fails an EWMA variance (lambda 0.94) is used. The method used
    is returned in ``info["method"]``.

    Returns (h, info) with h on the same scale as ``daily_returns``.
    """
    from arch import arch_model

    y = np.asarray(daily_returns, float) * scale
    n = len(y) if train_size is None else int(train_size)
    for d in (dist, "normal"):
        try:
            am = arch_model(y[:n], mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist=d, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            full = arch_model(y, mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist=d, rescale=False)
            h = np.asarray(full.fix(res.params.values).conditional_volatility, float) ** 2
            if np.isnan(h[0]):
                h[0] = h[1]
            if _egarch_ok(res.params, h, y[:n]):
                return h / scale**2, {"method": f"egarch_{d}", "params": res.params}
        except Exception:  # noqa: BLE001 - any optimiser failure falls through to the next option
            pass
    h = _ewma_variance(y, n_init=n)
    return h / scale**2, {"method": "ewma", "params": None}


# ---------------------------------------------------------------------------
# Intraday GARCH(1,1) with standardised Student-t errors
# ---------------------------------------------------------------------------


def _garch_q2(x2: np.ndarray, omega: float, alpha: float, beta: float, q0: float) -> np.ndarray:
    """q2[n] = omega + alpha * x2[n-1] + beta * q2[n-1], with q2[0] = q0."""
    u = np.empty_like(x2)
    u[0] = q0
    u[1:] = omega + alpha * x2[:-1]
    # y[n] = u[n] + beta * y[n-1], and y[0] = q0 exactly.
    return lfilter([1.0], [1.0, -beta], u)


def _t_loglik(x: np.ndarray, q2: np.ndarray, nu: float) -> np.ndarray:
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return c - 0.5 * np.log(q2) - (nu + 1) / 2 * np.log1p(x * x / ((nu - 2) * q2))


def _norm_loglik(x: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2 * np.pi) + np.log(q2) + x * x / q2)


def _unpack(theta: np.ndarray, dist: str):
    omega = np.exp(theta[0])
    persistence = 1.0 / (1.0 + np.exp(-theta[1])) * 0.99999
    share = 1.0 / (1.0 + np.exp(-theta[2]))
    alpha = persistence * share
    beta = persistence * (1 - share)
    nu = 2.05 + np.exp(theta[3]) if dist == "t" else np.inf
    return omega, alpha, beta, nu


def _pack(omega, alpha, beta, nu, dist):
    p = alpha + beta
    share = alpha / p
    th = [np.log(omega), np.log(p / 0.99999 / (1 - p / 0.99999)), np.log(share / (1 - share))]
    th.append(np.log(nu - 2.05) if dist == "t" else 0.0)
    return np.array(th)


def fit_garch11(x, weights=None, dist: str = "t", start=(0.03, 0.95, 8.0)):
    """Maximum likelihood GARCH(1,1) for a mean-zero series x.

    ``weights`` (0 or 1) lets missing minutes stay in the recursion without
    entering the likelihood. Returns dict(omega, alpha, beta, nu, loglik, nobs).
    """
    x = np.asarray(x, float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    x2 = x * x
    q0 = float(np.average(x2, weights=w))
    a0, b0, nu0 = start
    omega0 = q0 * (1 - a0 - b0)
    th0 = _pack(omega0, a0, b0, nu0, dist)

    def nll(th):
        omega, alpha, beta, nu = _unpack(th, dist)
        q2 = _garch_q2(x2, omega, alpha, beta, q0)
        ll = _t_loglik(x, q2, nu) if dist == "t" else _norm_loglik(x, q2)
        val = -np.sum(w * ll)
        return val if np.isfinite(val) else 1e300

    res = minimize(nll, th0, method="L-BFGS-B")
    if dist != "t":
        res.x[3] = 0.0
    omega, alpha, beta, nu = _unpack(res.x, dist)
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "nu": float(nu),
        "loglik": float(-res.fun),
        "nobs": int(w.sum()),
        "converged": bool(res.success),
        "q0": q0,
    }


def garch11_tstats(x, params: dict, weights=None, dist: str = "t", eps: float = 1e-4):
    """z statistics for (omega, alpha, beta, nu) from a numerical Hessian.

    Uses the outer-product-free observed information in natural parameters.
    Intended for reporting, like Table 2 of the paper.
    """
    x = np.asarray(x, float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    x2 = x * x
    q0 = params["q0"]
    names = ["omega", "alpha", "beta"] + (["nu"] if dist == "t" else [])
    p0 = np.array([params[n] for n in names])

    def ll(p):
        omega, alpha, beta = p[:3]
        nu = p[3] if dist == "t" else np.inf
        q2 = _garch_q2(x2, omega, alpha, beta, q0)
        out = _t_loglik(x, q2, nu) if dist == "t" else _norm_loglik(x, q2)
        return np.sum(w * out)

    k = len(p0)
    H = np.zeros((k, k))
    step = eps * np.maximum(np.abs(p0), 1e-3)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = step[i]
            ej[j] = step[j]
            fpp = ll(p0 + ei + ej)
            fpm = ll(p0 + ei - ej)
            fmp = ll(p0 - ei + ej)
            fmm = ll(p0 - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * step[i] * step[j])
    try:
        cov = np.linalg.inv(-H)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return dict(zip(names, p0 / se))


# ---------------------------------------------------------------------------
# ARMA(1,1) mean by conditional sum of squares
# ---------------------------------------------------------------------------


def fit_arma11(r):
    """Conditional least squares ARMA(1,1): r_n = c + phi r_{n-1} + theta e_{n-1} + e_n."""
    r = np.asarray(r, float)
    rl = np.concatenate([[r.mean()], r[:-1]])

    def resid(p):
        c, phi, theta = p
        return lfilter([1.0], [1.0, theta], r - c - phi * rl)

    def sse(p):
        if abs(p[1]) >= 0.999 or abs(p[2]) >= 0.999:
            return 1e300
        e = resid(p)
        return float(np.dot(e, e))

    res = minimize(sse, np.array([r.mean(), 0.05, 0.0]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-14, "maxiter": 4000})
    c, phi, theta = res.x
    return {"c": float(c), "phi": float(phi), "theta": float(theta)}, resid(res.x)


# ---------------------------------------------------------------------------
# mcsGARCH
# ---------------------------------------------------------------------------


@dataclass
class McsGarch:
    """mcsGARCH on a (days x minutes) grid of 1-minute log returns.

    Parameters
    ----------
    mean : "zero", "constant" or "arma11"
    dist : "t" or "normal"
    expiry_diurnal : if True, estimate a separate diurnal curve s_i for expiry
        days (an extension: the paper shows expiry days behave differently).
    smooth : odd window length for a centred moving average of s_i (1 = none).
    """

    mean: str = "zero"
    dist: str = "t"
    expiry_diurnal: bool = False
    smooth: int = 1
    params: dict = field(default_factory=dict)
    mean_params: dict = field(default_factory=dict)
    s: np.ndarray | None = None
    s_expiry: np.ndarray | None = None

    # -- helpers ----------------------------------------------------------
    def _residuals(self, R: np.ndarray) -> np.ndarray:
        if self.mean == "zero":
            return R.copy()
        if self.mean == "constant":
            return R - self.mean_params["c"]
        if self.mean == "arma11":
            c, phi, theta = (self.mean_params[k] for k in ("c", "phi", "theta"))
            flat = R.ravel()
            rl = np.concatenate([[c / (1 - phi)], flat[:-1]])
            return lfilter([1.0], [1.0, theta], flat - c - phi * rl).reshape(R.shape)
        raise ValueError(self.mean)

    def _smooth(self, s: np.ndarray) -> np.ndarray:
        if self.smooth <= 1:
            return s
        k = self.smooth
        pad = np.pad(s, (k // 2, k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    def diurnal(self, expiry_mask=None) -> np.ndarray:
        """(days x minutes) diurnal variance factors for each day.

        With no mask, returns the ordinary-day pattern as a single row (1 x minutes).
        """
        if expiry_mask is None:
            return np.atleast_2d(self.s).copy()
        mask = np.asarray(expiry_mask, bool)
        S = np.tile(self.s, (len(mask), 1))
        if self.expiry_diurnal and self.s_expiry is not None:
            S[mask] = self.s_expiry
        return S

    def _pattern(self, ratio: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Average of ratio by minute over valid entries, then smoothed.

        A minute with no valid observation at all (for example the first minute
        when every day's first return is missing) would otherwise get a factor of
        0, and 0 / 0 later turns the whole GARCH input into NaN, so the fit
        silently returns its starting values. Such minutes get the average factor.
        """
        cnt = valid.sum(axis=0)
        raw = np.where(valid, ratio, 0).sum(axis=0) / np.maximum(cnt, 1)
        bad = (cnt == 0) | ~(raw > 0)
        if bad.all():
            raw = np.ones_like(raw)
        elif bad.any():
            raw = raw.copy()
            raw[bad] = raw[~bad].mean()
        return self._smooth(raw)

    # -- estimation -------------------------------------------------------
    def fit(self, R, h, expiry_mask=None, compute_tstats: bool = False):
        """Fit on returns R (days x minutes) and daily variances h (days,)."""
        R = np.asarray(R, float)
        h = np.asarray(h, float)
        T, M = R.shape
        expiry_mask = np.zeros(T, bool) if expiry_mask is None else np.asarray(expiry_mask, bool)
        valid = np.isfinite(R)
        Rf = np.where(valid, R, 0.0)

        if self.mean == "constant":
            self.mean_params = {"c": float(Rf[valid].mean())}
        elif self.mean == "arma11":
            self.mean_params, _ = fit_arma11(Rf.ravel())
        eps = self._residuals(Rf)
        eps = np.where(valid, eps, 0.0)

        ratio = eps**2 / h[:, None]
        self.s = self._pattern(ratio, valid)
        if self.expiry_diurnal and expiry_mask.any():
            self.s_expiry = self._pattern(ratio[expiry_mask], valid[expiry_mask])
        S = self.diurnal(expiry_mask)
        x = np.where(valid, eps / np.sqrt(h[:, None] * S), 0.0).ravel()
        self.params = fit_garch11(x, weights=valid.ravel().astype(float), dist=self.dist)
        if compute_tstats:
            self.params["tstats"] = garch11_tstats(x, self.params, valid.ravel().astype(float), self.dist)
        self._x_train_last = x[-1]
        return self

    # -- filtering and forecasting ---------------------------------------
    def filter(self, R, h, expiry_mask=None, q0: float | None = None):
        """Run the fitted model over R. Returns dict with eps, S, x and q2 (days x minutes).

        q2[t, i] is the one-step-ahead variance of x for minute i, known at the
        end of minute i-1, so it can be used for forecasting.
        """
        R = np.asarray(R, float)
        h = np.asarray(h, float)
        T, M = R.shape
        expiry_mask = np.zeros(T, bool) if expiry_mask is None else np.asarray(expiry_mask, bool)
        valid = np.isfinite(R)
        eps = np.where(valid, self._residuals(np.where(valid, R, 0.0)), 0.0)
        S = self.diurnal(expiry_mask)
        x = eps / np.sqrt(h[:, None] * S)
        p = self.params
        qbar = p["omega"] / max(1e-12, 1 - p["alpha"] - p["beta"])
        q2 = _garch_q2((x**2).ravel(), p["omega"], p["alpha"], p["beta"], qbar if q0 is None else q0)
        # Missing minutes: q2 still updates with x = 0, which slightly lowers q2.
        return {"eps": eps, "S": S, "x": x, "q2": q2.reshape(T, M)}

    def forecast_variance(self, filt: dict, h, anchors_t, anchors_i, horizon):
        """Forecast the variance of the sum of returns over minutes [i, i + horizon).

        ``anchors_t`` and ``anchors_i`` are arrays of day indices and minute
        indices (i = number of minutes already observed that day, so the
        forecast uses data up to the end of minute i - 1). Windows that run
        past the close are truncated at the close.
        """
        p = self.params
        persist = p["alpha"] + p["beta"]
        qbar = p["omega"] / max(1e-12, 1 - persist)
        S = filt["S"]
        q2 = filt["q2"]
        T, M = S.shape
        h = np.asarray(h, float)
        at = np.asarray(anchors_t, int)
        ai = np.asarray(anchors_i, int)
        hz = np.broadcast_to(np.asarray(horizon, int), at.shape)
        out = np.empty(len(at))
        # Vectorised over days: anchors that share (minute, horizon) share the
        # weights persist**k, so each group is two matrix-vector products.
        keys = ai.astype(np.int64) * (M + 1) + hz
        for key in np.unique(keys):
            sel = np.flatnonzero(keys == key)
            i, hn = int(ai[sel[0]]), int(hz[sel[0]])
            j = np.arange(i, min(i + hn, M))
            k = j - i
            St = S[at[sel]][:, j]
            # q2[t, i] is known at the end of minute i - 1.
            q_next = q2[at[sel], i]
            out[sel] = h[at[sel]] * (qbar * St.sum(axis=1) + (St @ persist**k) * (q_next - qbar))
        return out


# ---------------------------------------------------------------------------
# Parametric extensions of mcsGARCH (study P109, docs/prereg/P109_MCSGARCH_EXTENSIONS.md)
#
# Nothing above this line was changed for P109. McsGarchExt with every flag off runs exactly the
# code of McsGarch (it calls it), so every existing forecast stays bit-for-bit the same.
# ---------------------------------------------------------------------------

# E[1{z < 0} z^2] / E[z^2] for a symmetric error: the expected share of the leverage term,
# used for the stationarity condition and the multi-step forecasts of the asymmetric recursion.
KAPPA_NEG = 0.5

from scipy.special import expit  # noqa: E402  (P109 code path only)


def har_daily_variance(daily_rv, train_size: int | None = None, floor: float = 1e-12):
    """One-step-ahead daily variance level from a log-HAR of daily realized variance (Corsi 2009).

        log RV_t = b0 + b_d log RV_{t-1} + b_w log RV^w_t + b_m log RV^m_t + e_t,

    with RV^w_t and RV^m_t the means of RV over days t-5..t-1 and t-22..t-1 (fewer at the start of the
    sample, as in ``features.build_dataset``). Fitted by OLS on days 1 .. train_size - 1 (all days if
    None); the fitted coefficients are then applied to every day, so h_t uses only days before t.
    Day 0 has no history and gets day 1's value (day 0 is only ever a training day).

    Inside mcsGARCH only the day-to-day variation of h matters: a constant factor in h is absorbed by
    the diurnal curve (s_i = mean of eps^2 / h), so no smearing factor is applied.
    Returns (h, info) with info["params"] = (b0, b_d, b_w, b_m).
    """
    rv = np.maximum(np.asarray(daily_rv, float), floor)
    T = len(rv)
    n = T if train_size is None else int(train_size)
    c = np.concatenate([[0.0], np.cumsum(rv)])
    t = np.arange(1, T)
    lo_w = np.maximum(0, t - 5)
    lo_m = np.maximum(0, t - 22)
    w = (c[t] - c[lo_w]) / (t - lo_w)
    m = (c[t] - c[lo_m]) / (t - lo_m)
    X = np.column_stack([np.ones(T - 1), np.log(rv[t - 1]), np.log(w), np.log(m)])
    y = np.log(rv[1:])
    b, *_ = np.linalg.lstsq(X[: n - 1], y[: n - 1], rcond=None)
    lh = np.empty(T)
    lh[1:] = X @ b
    lh[0] = lh[1]
    return np.exp(lh), {"method": "har", "params": dict(zip(["b0", "b_d", "b_w", "b_m"], map(float, b)))}


def _softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


def ext_q2(x2, negx2, z, omega, alpha, beta, gam_l, gam_x, delta, q_init, qbar):
    """(days x minutes) one-step-ahead variance q2 of the extended intraday recursion.

    Within a day (minute i >= 1):
        q2[t, i] = omega + alpha x2[t, i-1] + gam_l negx2[t, i-1] + gam_x z[t, i-1] + beta q2[t, i-1]
    Across the night, with carry = the same expression evaluated at the last minute of day t-1:
        q2[t, 0] = qbar + delta (carry - qbar)
    and q2[0, 0] = q_init. negx2 = x2 where the residual is negative (else 0); z is today's running
    mean of x2. With gam_l = gam_x = 0 and delta = 1 this is the recursion of McsGarch (mathematically;
    the order of floating-point operations differs). Vectorised: one filter along the minutes of all
    days, then one filter across days for the opening values.
    """
    T, M = x2.shape
    u = omega + alpha * x2 + gam_l * negx2 + gam_x * z      # drives q2 of the next minute
    U = np.zeros((T, M))
    U[:, 1:] = u[:, :-1]
    P = lfilter([1.0], [1.0, -beta], U, axis=1)             # the part that does not depend on q2[t, 0]
    carry = u[:, -1] + beta * P[:, -1]                      # plus beta^M q2[t, 0], handled by phi below
    cvec = np.empty(T)
    cvec[0] = q_init
    cvec[1:] = (1.0 - delta) * qbar + delta * carry[:-1]
    phi = delta * beta**M
    q0 = lfilter([1.0], [1.0, -phi], cvec)
    return q0[:, None] * (beta ** np.arange(M))[None, :] + P


def _ext_inputs(X, neg):
    x2 = X * X
    negx2 = np.where(neg, x2, 0.0)
    z = np.cumsum(x2, axis=1) / np.arange(1, X.shape[1] + 1)[None, :]
    return x2, negx2, z


def _ext_components(leverage: bool, garchx_today: bool) -> list:
    return ["alpha"] + (["gam_l"] if leverage else []) + (["gam_x"] if garchx_today else [])


def _ext_unpack(th, comps, dist, overnight_decay):
    """Unconstrained theta -> parameters. Persistence pi = alpha + beta + KAPPA_NEG gam_l + gam_x < 1
    is split among the components by a softmax with beta as the reference."""
    omega = float(np.exp(th[0]))
    pers = 0.99999 * float(expit(th[1]))
    k = len(comps)
    sh = _softmax(np.concatenate([[0.0], th[2:2 + k]])) * pers
    p = {"omega": omega, "beta": float(sh[0]), "alpha": 0.0, "gam_l": 0.0, "gam_x": 0.0}
    for name, v in zip(comps, sh[1:]):
        p[name] = float(v / KAPPA_NEG) if name == "gam_l" else float(v)
    j = 2 + k
    p["nu"] = float(2.05 + np.exp(th[j])) if dist == "t" else np.inf
    p["delta"] = float(expit(th[j + 1])) if overnight_decay else 1.0
    p["persistence"] = float(pers)
    p["qbar"] = omega / max(1e-12, 1.0 - pers)
    return p


def _ext_pack(p, comps, dist, overnight_decay):
    pers = p["alpha"] + p["beta"] + KAPPA_NEG * p.get("gam_l", 0.0) + p.get("gam_x", 0.0)
    th = [np.log(p["omega"]), np.log(pers / 0.99999 / (1 - pers / 0.99999))]
    for name in comps:
        v = KAPPA_NEG * p[name] if name == "gam_l" else p[name]
        th.append(np.log(v / p["beta"]))
    th.append(np.log(p["nu"] - 2.05) if dist == "t" else 0.0)
    if overnight_decay:
        th.append(np.log(p["delta"] / (1 - p["delta"])))
    return np.array(th, float)


# Two registered starting points for the extended likelihood (the higher maximum is kept).
EXT_STARTS = (
    {"alpha": 0.05, "beta": 0.935, "gam_l": 0.01, "gam_x": 0.005, "nu": 8.0, "delta": 0.5},
    {"alpha": 0.05, "beta": 0.945, "gam_l": 0.0002, "gam_x": 0.0001, "nu": 8.0, "delta": 0.99},
)


def fit_garch_ext(X, neg, weights=None, overnight_decay=False, leverage=False, garchx_today=False,
                  dist: str = "t", starts=EXT_STARTS):
    """Maximum likelihood for the extended intraday recursion (``ext_q2``) on a (days x minutes) grid of
    normalised residuals X = eps / sqrt(h s). ``neg`` marks negative residuals; ``weights`` (0/1) keeps
    missing minutes in the recursion but out of the likelihood, as in ``fit_garch11``. Returns a dict with
    omega, alpha, beta, gam_l, gam_x, delta, nu, persistence, qbar, loglik, nobs, converged, q0, start."""
    X = np.asarray(X, float)
    w = np.ones_like(X) if weights is None else np.asarray(weights, float)
    x2, negx2, z = _ext_inputs(X, np.asarray(neg, bool))
    q0 = float(np.average(x2, weights=w))
    comps = _ext_components(leverage, garchx_today)
    flat_w = w.ravel()
    flat_x = X.ravel()

    def nll(th):
        p = _ext_unpack(th, comps, dist, overnight_decay)
        q2 = ext_q2(x2, negx2, z, p["omega"], p["alpha"], p["beta"], p["gam_l"], p["gam_x"], p["delta"],
                    q0, p["qbar"]).ravel()
        if not np.all(q2 > 0):
            return 1e300
        ll = _t_loglik(flat_x, q2, p["nu"]) if dist == "t" else _norm_loglik(flat_x, q2)
        val = -np.sum(flat_w * ll)
        return val if np.isfinite(val) else 1e300

    best = None
    for k, s in enumerate(starts):
        p0 = dict(s)
        if not leverage:
            p0["gam_l"] = 0.0
        if not garchx_today:
            p0["gam_x"] = 0.0
        pers0 = p0["alpha"] + p0["beta"] + KAPPA_NEG * p0["gam_l"] + p0["gam_x"]
        p0["omega"] = q0 * (1 - pers0)
        if dist != "t":
            p0["nu"] = np.inf
        res = minimize(nll, _ext_pack(p0, comps, dist, overnight_decay), method="L-BFGS-B")
        if best is None or res.fun < best[0].fun:
            best = (res, k)
    res, k = best
    p = _ext_unpack(res.x, comps, dist, overnight_decay)
    p.update({"loglik": float(-res.fun), "nobs": int(w.sum()), "converged": bool(res.success), "q0": q0,
              "start": int(k)})
    return p


@dataclass
class McsGarchExt(McsGarch):
    """mcsGARCH with optional extensions of the intraday component (study P109).

    overnight_decay : the intraday state decays toward its unconditional level qbar over the night,
        q2[t, 0] = qbar + delta (carry - qbar), with delta estimated (McsGarch: delta = 1, no decay).
    leverage : an asymmetric (GJR) term, gam_l x^2 1{eps < 0}, in the intraday recursion.
    garchx_today : a GARCH-X term, gam_x z, with z today's running mean of x^2 (the day's realized
        variance so far, in units of the model's own expected variance).

    With every flag off, ``fit``, ``filter`` and ``forecast_variance`` are McsGarch's own, unchanged.
    The daily component h and the diurnal curve are as in McsGarch (h is an input).
    """

    overnight_decay: bool = False
    leverage: bool = False
    garchx_today: bool = False

    @property
    def extended(self) -> bool:
        return bool(self.overnight_decay or self.leverage or self.garchx_today)

    def fit(self, R, h, expiry_mask=None, compute_tstats: bool = False):
        if not self.extended:
            return super().fit(R, h, expiry_mask, compute_tstats)
        # the diurnal steps of McsGarch.fit, unchanged
        R = np.asarray(R, float)
        h = np.asarray(h, float)
        T, M = R.shape
        expiry_mask = np.zeros(T, bool) if expiry_mask is None else np.asarray(expiry_mask, bool)
        valid = np.isfinite(R)
        Rf = np.where(valid, R, 0.0)
        if self.mean == "constant":
            self.mean_params = {"c": float(Rf[valid].mean())}
        elif self.mean == "arma11":
            self.mean_params, _ = fit_arma11(Rf.ravel())
        eps = self._residuals(Rf)
        eps = np.where(valid, eps, 0.0)
        ratio = eps**2 / h[:, None]
        self.s = self._pattern(ratio, valid)
        if self.expiry_diurnal and expiry_mask.any():
            self.s_expiry = self._pattern(ratio[expiry_mask], valid[expiry_mask])
        S = self.diurnal(expiry_mask)
        X = np.where(valid, eps / np.sqrt(h[:, None] * S), 0.0)
        self.params = fit_garch_ext(X, eps < 0, valid.astype(float), self.overnight_decay, self.leverage,
                                    self.garchx_today, dist=self.dist)
        return self

    def filter(self, R, h, expiry_mask=None, q0: float | None = None):
        if not self.extended:
            return super().filter(R, h, expiry_mask, q0)
        R = np.asarray(R, float)
        h = np.asarray(h, float)
        T, M = R.shape
        expiry_mask = np.zeros(T, bool) if expiry_mask is None else np.asarray(expiry_mask, bool)
        valid = np.isfinite(R)
        eps = np.where(valid, self._residuals(np.where(valid, R, 0.0)), 0.0)
        S = self.diurnal(expiry_mask)
        x = eps / np.sqrt(h[:, None] * S)
        x2, negx2, z = _ext_inputs(x, eps < 0)
        p = self.params
        q2 = ext_q2(x2, negx2, z, p["omega"], p["alpha"], p["beta"], p["gam_l"], p["gam_x"], p["delta"],
                    p["qbar"] if q0 is None else q0, p["qbar"])
        cum_x2 = np.concatenate([np.zeros((T, 1)), np.cumsum(x2, axis=1)], axis=1)
        return {"eps": eps, "S": S, "x": x, "q2": q2, "cum_x2": cum_x2}

    def forecast_variance(self, filt: dict, h, anchors_t, anchors_i, horizon):
        """As McsGarch.forecast_variance, with the expected path of the extended recursion:
        E[x2] = q2 for future minutes, E[1{eps < 0} x2] = KAPPA_NEG q2, and today's running mean z
        updated with the expected x2. Windows that run past the close are truncated at the close."""
        if not self.extended:
            return super().forecast_variance(filt, h, anchors_t, anchors_i, horizon)
        p = self.params
        a_eff = p["alpha"] + p["beta"] + KAPPA_NEG * p["gam_l"]
        S, q2, cx = filt["S"], filt["q2"], filt["cum_x2"]
        T, M = S.shape
        h = np.asarray(h, float)
        at = np.asarray(anchors_t, int)
        ai = np.asarray(anchors_i, int)
        hz = np.broadcast_to(np.asarray(horizon, int), at.shape)
        out = np.empty(len(at))
        keys = ai.astype(np.int64) * (M + 1) + hz
        for key in np.unique(keys):
            sel = np.flatnonzero(keys == key)
            i, hn = int(ai[sel[0]]), int(hz[sel[0]])
            tt = at[sel]
            qhat = q2[tt, i].copy()                # q2[t, i] is known at the end of minute i - 1
            run = cx[tt, i].copy()                 # sum of x2 over minutes 0 .. i-1 of the day
            tot = np.zeros(len(sel))
            for j in range(i, min(i + hn, M)):
                tot += S[tt, j] * qhat
                run += qhat
                qhat = p["omega"] + a_eff * qhat + p["gam_x"] * run / (j + 1)
            out[sel] = h[tt] * tot
        return out


# ---------------------------------------------------------------------------
# Persistence of the extended intraday recursion (study P114, docs/prereg/P114_PERSISTENCE_FIXED_MODEL.md)
#
# New code only. Nothing above this line was changed for P114, so every existing result stays bit-for-bit
# the same.
# ---------------------------------------------------------------------------

IRF_MAX_STEPS = 100_000          # a response that has not halved after this many trading minutes: half-life = inf
PERSISTENCE_CAP = 0.99999        # the upper bound of persistence used by fit_garch11 and fit_garch_ext


def ext_impulse_response(params: dict, M: int, shock_minute: int, n_steps: int = 0, stop_at_half: bool = False,
                         max_steps: int = IRF_MAX_STEPS, kappa: float = KAPPA_NEG) -> np.ndarray:
    """Response of the expected intraday variance to one standardised shock (exact recursion, no simulation).

    The shock: at minute ``shock_minute`` (0-based) of a session of ``M`` minutes, the squared normalised
    residual x^2 is one unit above its conditional expectation q; its sign is averaged, so the leverage term
    responds by ``kappa`` gam_l. Every later x^2 is replaced by its expectation, so the response is linear in
    the shock and does not depend on omega, nu or the level of q. The model's own clock is used: one step per
    minute within a day, and the night is one step (q at the next open = qbar + delta (carry - qbar));
    today's running mean z restarts each day, as in ``ext_q2``.

    Returns r with r[k - 1] the change in E[q] k trading minutes after the shock minute, for k = 1 .. n, where
    n is ``n_steps``, or, with ``stop_at_half``, the larger of ``n_steps`` and the first k at which the
    response is at most half of r[0] (never more than ``max_steps``). For GARCH(1,1) (gam_l = gam_x = 0,
    delta = 1) r[k - 1] = alpha (alpha + beta) ** (k - 1).
    """
    if not 0 <= int(shock_minute) < int(M):
        raise ValueError(f"shock_minute {shock_minute} outside a session of {M} minutes")
    al = float(params["alpha"])
    be = float(params["beta"])
    gl = float(params.get("gam_l", 0.0))
    gx = float(params.get("gam_x", 0.0))
    de = float(params.get("delta", 1.0))
    a_x = al + kappa * gl               # weight of the expected squared residual on the next q
    i = int(shock_minute)
    dq, dx2, dS = 0.0, 1.0, 1.0         # responses of q_i, x_i^2 and the day's running sum of x^2 up to minute i
    out = []
    r1 = None
    limit = max(int(n_steps), 1)
    k = 0
    while k < max_steps:
        nxt = a_x * dx2 + gx * dS / (i + 1) + be * dq
        if i == M - 1:                  # the night: decay toward qbar, and the running mean restarts
            nxt = de * nxt
            i, dS = 0, 0.0
        else:
            i += 1
        dq = nxt
        dx2 = dq
        dS += dq
        out.append(dq)
        k += 1
        if r1 is None:
            r1 = dq
        if k >= limit and (not stop_at_half or dq <= 0.5 * r1):
            break
    return np.asarray(out, float)


def irf_half_life(r) -> float:
    """Half-life of an impulse response r (r[0] = the response one step after the shock): the number of steps
    after r[0] at which the response first falls to half of r[0], log-linearly interpolated between steps.
    Equals ln(0.5) / ln(alpha + beta) for GARCH(1,1). inf if it never falls to half within r; nan if r[0] <= 0.
    A response that falls to 0 in one step (delta = 0 at the night) halves at the start of that step."""
    r = np.asarray(r, float)
    if len(r) == 0 or not r[0] > 0:
        return float("nan")
    target = 0.5 * r[0]
    below = np.flatnonzero(r[1:] <= target)
    if len(below) == 0:
        return float("inf")
    k = int(below[0]) + 1
    lo, hi = r[k - 1], r[k]
    if hi <= 0:
        return float(k - 1)
    return float(k - 1) + float((np.log(lo) - np.log(target)) / (np.log(lo) - np.log(hi)))


def equivalent_persistence(half_life) -> np.ndarray:
    """0.5 ** (1 / half-life): the alpha + beta of a GARCH(1,1) with the same half-life (inf -> 1)."""
    hl = np.asarray(half_life, float)
    with np.errstate(divide="ignore"):
        return np.where(np.isfinite(hl), 0.5 ** (1.0 / hl), np.where(np.isnan(hl), np.nan, 1.0))


def ext_persistence_summary(params: dict, M: int, shock_minute: int, max_steps: int = IRF_MAX_STEPS,
                            kappa: float = KAPPA_NEG) -> dict:
    """Persistence measures of a fitted intraday recursion (P114).

    ab: alpha + beta; total: alpha + beta + kappa gam_l + gam_x (the stationarity sum); hl_ab: ln(0.5) /
    ln(alpha + beta); hl_irf: half-life of the impulse response to one standardised shock at ``shock_minute``
    (``ext_impulse_response``, ``irf_half_life``); rho_irf: its equivalent persistence 0.5 ** (1 / hl_irf);
    share_close and share_next_open: the response at the last minute of the shock's day and at the next
    day's first minute, as shares of the first response.
    """
    n_close = M - 1 - int(shock_minute)            # steps from the shock minute to the last minute of the day
    r = ext_impulse_response(params, M, shock_minute, n_steps=n_close + 1, stop_at_half=True,
                             max_steps=max_steps, kappa=kappa)
    hl = irf_half_life(r)
    ab = float(params["alpha"]) + float(params["beta"])
    tot = ab + kappa * float(params.get("gam_l", 0.0)) + float(params.get("gam_x", 0.0))
    return {"ab": ab, "total": tot, "hl_ab": float(np.log(0.5) / np.log(ab)) if ab < 1 else float("inf"),
            "hl_irf": hl, "rho_irf": float(equivalent_persistence(hl)), "r1": float(r[0]),
            "share_close": float(r[n_close - 1] / r[0]) if n_close >= 1 else float("nan"),
            "share_next_open": float(r[n_close] / r[0])}


def _irf_hl_capped(alpha, beta, gam_x, delta, M, shock_minute, cap_steps):
    r = ext_impulse_response({"alpha": alpha, "beta": beta, "gam_x": gam_x, "delta": delta}, M, shock_minute,
                             stop_at_half=True, max_steps=cap_steps)
    return irf_half_life(r)


def solve_beta_for_half_life(alpha, gam_x, delta, M, shock_minute, half_life, cap=PERSISTENCE_CAP):
    """The beta at which the impulse-response half-life equals ``half_life`` (it increases with beta), with
    alpha + beta + gam_x below ``cap``; None if no such beta exists."""
    from scipy.optimize import brentq

    bmax = cap - alpha - gam_x
    if bmax <= 0:
        return None
    steps = int(2 * half_life) + 20

    def g(b):
        hl = _irf_hl_capped(alpha, b, gam_x, delta, M, shock_minute, steps)
        return (np.log(hl) if np.isfinite(hl) else np.log(steps)) - np.log(half_life)

    lo, hi = g(0.0), g(bmax)
    if not (lo <= 0 <= hi):
        return None
    if lo == 0:
        return 0.0
    return float(brentq(g, 0.0, bmax, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=200))


def fit_garch_ext_fixed(X, weights=None, overnight_decay=False, garchx_today=False, measure: str = "ab",
                        value: float = 0.99, M: int | None = None, shock_minute: int | None = None,
                        start: dict | None = None, dist: str = "t"):
    """Maximum likelihood of the extended intraday recursion (no leverage term) with one persistence measure
    held fixed, for profile likelihoods and likelihood-ratio tests of equal persistence (P114).

    measure "ab":  alpha + beta = value. Free: omega, the alpha share of value, gam_x (below 0.99999 - value),
                   delta and nu.
    measure "irf": the impulse-response half-life (shock at ``shock_minute`` of an ``M``-minute session)
                   = value, in minutes. Free: omega, alpha, gam_x, delta and nu; beta is solved from the
                   constraint (``solve_beta_for_half_life``).
    With both flags off this is the GARCH(1,1) of ``fit_garch11`` with alpha + beta fixed (as
    ``replication.profile_half_life``). X is the (days x minutes) grid of normalised residuals, as in
    ``fit_garch_ext``; ``start`` a fitted parameter dict for starting values (alpha, gam_x, delta, nu and the
    unconditional level are kept where the constraint allows). Returns (loglik, params, converged).
    """
    X = np.asarray(X, float)
    T, Mx = X.shape
    M = Mx if M is None else int(M)
    w = np.ones_like(X) if weights is None else np.asarray(weights, float)
    x2, negx2, z = _ext_inputs(X, np.zeros_like(X, bool))
    q0 = float(np.average(x2, weights=w))
    flat_w, flat_x = w.ravel(), X.ravel()
    cap = PERSISTENCE_CAP
    s = dict(start or {})
    a0 = float(s.get("alpha", 0.05))
    gx0 = float(s.get("gam_x", 0.0)) if garchx_today else 0.0
    de0 = float(s.get("delta", 1.0)) if overnight_decay else 1.0
    nu0 = float(s.get("nu", 8.0)) if dist == "t" else np.inf
    if "omega" in s:
        pers_s = s["alpha"] + s["beta"] + KAPPA_NEG * s.get("gam_l", 0.0) + s.get("gam_x", 0.0)
        qbar0 = float(s["omega"] / max(1e-12, 1 - pers_s))
    else:
        qbar0 = q0

    def clip(v):
        return min(max(float(v), 1e-6), 1 - 1e-6)

    def logit(v):
        v = clip(v)
        return float(np.log(v / (1 - v)))

    def unpack(th):
        qbar = float(np.exp(th[0]))       # the unconditional level, better conditioned than omega
        j = 2
        if measure == "ab":
            alpha = value * float(expit(th[1]))
            beta = value - alpha
            gam_x = (cap - value) * float(expit(th[j])) if garchx_today else 0.0
        else:
            alpha = cap * float(expit(th[1]))
            gam_x = (cap - alpha) * float(expit(th[j])) if garchx_today else 0.0
            beta = None
        j += 1 if garchx_today else 0
        nu = float(2.05 + np.exp(th[j])) if dist == "t" else np.inf
        delta = float(expit(th[j + 1])) if overnight_decay else 1.0
        if measure == "irf":
            beta = solve_beta_for_half_life(alpha, gam_x, delta, M, shock_minute, value, cap)
        return qbar, alpha, beta, gam_x, delta, nu

    def pack(qbar, alpha, gam_x, delta, nu):
        th = [np.log(qbar)]
        if measure == "ab":
            th.append(logit(alpha / value))
            if garchx_today:
                th.append(logit(gam_x / (cap - value)))
        else:
            th.append(logit(alpha / cap))
            if garchx_today:
                th.append(logit(gam_x / (cap - alpha)))
        th.append(np.log(max(nu - 2.05, 1e-3)) if dist == "t" else 0.0)
        if overnight_decay:
            th.append(logit(delta))
        return np.array(th, float)

    def nll(th):
        qbar, alpha, beta, gam_x, delta, nu = unpack(th)
        if beta is None:
            return 1e300
        pers = alpha + beta + gam_x
        if not pers < 1:
            return 1e300
        q2 = ext_q2(x2, negx2, z, qbar * (1 - pers), alpha, beta, 0.0, gam_x, delta, q0, qbar).ravel()
        if not np.all(q2 > 0):
            return 1e300
        ll = _t_loglik(flat_x, q2, nu) if dist == "t" else _norm_loglik(flat_x, q2)
        v = -np.sum(flat_w * ll)
        return v if np.isfinite(v) else 1e300

    # starting values: keep alpha, gam_x, delta, nu and the unconditional level where the constraint allows
    if measure == "ab":
        a_start = value * (a0 / (a0 + float(s.get("beta", 0.9)))) if "beta" in s else min(a0, 0.5 * value)
        g_start = min(gx0, 0.5 * (cap - value))
        th0 = pack(qbar0, a_start, g_start, de0, nu0)
    else:
        a_start, g_start, d_start, beta0 = a0, gx0, de0, None
        for _ in range(60):
            beta0 = solve_beta_for_half_life(a_start, g_start, d_start, M, shock_minute, value, cap)
            if beta0 is not None:
                break
            # an infeasible start: let more of the shock survive the night, then shrink alpha and gam_x
            if overnight_decay and d_start < 0.999:
                d_start = 1 - (1 - d_start) / 2
            else:
                a_start, g_start = a_start / 2, g_start / 2
        if beta0 is None:
            return float("nan"), {}, False
        th0 = pack(qbar0, a_start, g_start, d_start, nu0)
    # L-BFGS-B, then a Nelder-Mead polish from its result (the constrained surface can have kinks where the
    # half-life crosses the night); the better of the two is kept
    res = minimize(nll, th0, method="L-BFGS-B")
    res2 = minimize(nll, res.x, method="Nelder-Mead",
                    options={"maxiter": 4000, "maxfev": 6000, "xatol": 1e-7, "fatol": 1e-7})
    best = res2 if res2.fun < res.fun else res
    qbar, alpha, beta, gam_x, delta, nu = unpack(best.x)
    pers = alpha + beta + gam_x
    p = {"omega": qbar * (1 - pers), "alpha": alpha, "beta": beta, "gam_l": 0.0, "gam_x": gam_x, "delta": delta,
         "nu": nu, "qbar": qbar, "measure": measure, "value": float(value)}
    return float(-best.fun), p, bool(res.success or res2.success)


# ---------------------------------------------------------------------------
# The expected squared conditional mean of an ARMA(1,1) intraday mean (study P116,
# docs/prereg/P116_BENCHMARK_AS_PRINTED.md)
#
# New code only. Nothing above this line was changed for P116, so every existing result stays bit-for-bit
# the same.
# ---------------------------------------------------------------------------


def arma11_conditional_mean(model: McsGarch, R) -> np.ndarray:
    """(days x minutes) conditional mean mu[t, i] = c + phi r[t, i-1] + theta e[t, i-1] of a fitted McsGarch.

    The ARMA(1,1) runs on the flattened series exactly as ``McsGarch._residuals`` runs it: across day
    boundaries (the last minute of day t-1 feeds the first minute of day t), with first lag c / (1 - phi) and
    first residual 0. Missing returns count as 0, as in ``McsGarch.filter``. mu[t, i] uses only returns up to
    minute i - 1. A "constant" mean gives c everywhere and a "zero" mean gives 0.
    """
    R = np.asarray(R, float)
    Rf = np.where(np.isfinite(R), R, 0.0)
    if model.mean == "zero":
        return np.zeros_like(Rf)
    if model.mean == "constant":
        return np.full_like(Rf, model.mean_params["c"])
    if model.mean != "arma11":
        raise ValueError(model.mean)
    c, phi, theta = (model.mean_params[k] for k in ("c", "phi", "theta"))
    e = model._residuals(Rf).ravel()
    r = Rf.ravel()
    r_prev = np.concatenate([[c / (1 - phi)], r[:-1]])
    e_prev = np.concatenate([[0.0], e[:-1]])
    return (c + phi * r_prev + theta * e_prev).reshape(Rf.shape)


def arma11_mean_square_forecast(model: McsGarch, R, filt: dict, h, anchors_t, anchors_i, horizon) -> np.ndarray:
    """Sum over the window [i, i + horizon) of E[mu_j^2], given the data up to the end of minute i - 1.

    With r_j = mu_j + eps_j, E[r_j^2] = E[mu_j^2] + E[eps_j^2], so the model's forecast of the sum of squared
    one-minute returns over the window is ``McsGarch.forecast_variance`` (the eps part) plus this term. For an
    ARMA(1,1) mean, mu_j = c + phi r_{j-1} + theta e_{j-1}; the first mu_i is known, and for later minutes
    mu_j = c + phi mu_{j-1} + (phi + theta) e_{j-1}, so

        m_j = E[mu_j] = c + phi m_{j-1},     V_j = Var(mu_j) = phi^2 V_{j-1} + (phi + theta)^2 sigma2_{j-1},

    with m_i = mu_i, V_i = 0 and sigma2_j = h_t s_j E[q_j] the expected variance of eps_j on the same expected
    path of the intraday GARCH(1,1) that ``forecast_variance`` uses. The term is sum_j (m_j^2 + V_j). Windows
    that run past the close are cut at the close, as in ``forecast_variance``. Zero for a "zero" mean.
    Arguments as ``McsGarch.forecast_variance``, plus R, the (days x minutes) returns the filter ran on.
    """
    at = np.asarray(anchors_t, int)
    out = np.zeros(len(at))
    if model.mean == "zero":
        return out
    if getattr(model, "extended", False):
        raise NotImplementedError("the expected q path here is the plain GARCH(1,1) one; not for McsGarchExt extensions")
    mp_ = model.mean_params
    c = mp_["c"]
    phi = mp_.get("phi", 0.0)
    psi = phi + mp_.get("theta", 0.0)
    p = model.params
    persist = p["alpha"] + p["beta"]
    qbar = p["omega"] / max(1e-12, 1 - persist)
    mu = arma11_conditional_mean(model, R)
    S, q2 = filt["S"], filt["q2"]
    T, M = S.shape
    h = np.asarray(h, float)
    ai = np.asarray(anchors_i, int)
    hz = np.broadcast_to(np.asarray(horizon, int), at.shape)
    keys = ai.astype(np.int64) * (M + 1) + hz
    for key in np.unique(keys):
        sel = np.flatnonzero(keys == key)
        i, hn = int(ai[sel[0]]), int(hz[sel[0]])
        n = min(i + hn, M) - i
        tt = at[sel]
        m = mu[tt, i].copy()
        V = np.zeros(len(sel))
        q_next = q2[tt, i]
        tot = m * m
        for k in range(1, n):
            sig2_prev = h[tt] * S[tt, i + k - 1] * (qbar + persist ** (k - 1) * (q_next - qbar))
            V = phi * phi * V + psi * psi * sig2_prev
            m = c + phi * m
            tot = tot + m * m + V
        out[sel] = tot
    return out
