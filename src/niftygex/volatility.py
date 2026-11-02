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


def _ewma_variance(y, lam=0.94):
    """One-step-ahead EWMA variance (RiskMetrics), used as a fallback."""
    y = np.asarray(y, float)
    h = np.empty_like(y)
    h[0] = np.var(y[: min(len(y), 20)])
    for t in range(1, len(y)):
        h[t] = lam * h[t - 1] + (1 - lam) * y[t - 1] ** 2
    return h


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
    h = _ewma_variance(y)
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
        """(days x minutes) diurnal variance factors for each day."""
        def rows(mask):
            S = np.tile(self.s, (len(mask), 1))
            if self.expiry_diurnal and self.s_expiry is not None:
                S[np.asarray(mask, bool)] = self.s_expiry
            return S
        return rows(expiry_mask)

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
        cnt = valid.sum(axis=0)
        self.s = self._smooth(np.where(valid, ratio, 0).sum(axis=0) / np.maximum(cnt, 1))
        if self.expiry_diurnal and expiry_mask.any():
            ve = valid[expiry_mask]
            self.s_expiry = self._smooth(
                np.where(ve, ratio[expiry_mask], 0).sum(axis=0) / np.maximum(ve.sum(axis=0), 1)
            )
        S = self.diurnal(expiry_mask)
        x = (eps / np.sqrt(h[:, None] * S)).ravel()
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
        hz = np.broadcast_to(np.asarray(horizon, int), np.shape(anchors_t))
        out = np.empty(len(anchors_t))
        for n, (t, i, hn) in enumerate(zip(np.asarray(anchors_t), np.asarray(anchors_i), hz)):
            j = np.arange(i, min(i + hn, M))
            k = j - i
            # q2[t, i] is known at the end of minute i - 1.
            q_next = q2[t, i]
            eq = qbar + persist**k * (q_next - qbar)
            out[n] = h[t] * np.sum(S[t, j] * eq)
        return out
