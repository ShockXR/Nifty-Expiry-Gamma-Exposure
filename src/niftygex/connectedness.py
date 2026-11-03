"""Diebold-Yilmaz (2012) connectedness from a generalized variance decomposition.

Steps
-----
1. Fit a VAR(p) by OLS: y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,
   Cov(u_t) = Sigma.
2. Moving-average matrices: Psi_0 = I, Psi_h = sum_{l=1..min(h,p)} A_l Psi_{h-l}.
3. Generalized FEVD (Pesaran and Shin 1998), horizon H:

       theta_ij = Sigma_jj^{-1} sum_{h<H} (e_i' Psi_h Sigma e_j)^2
                  / sum_{h<H} e_i' Psi_h Sigma Psi_h' e_i

4. Normalise each row to sum to 1: theta~_ij. (``fevd_table`` and
   ``cholesky_fevd`` also give the unnormalised, column-normalised and
   Cholesky versions, used by study P113; the defaults here are unchanged.)
5. Total connectedness (DY 2012 standard scale, 0 to 100):

       C = 100 * sum_{i != j} theta~_ij / N

   "To" others for j: sum_{i != j} theta~_ij; "From" others for i:
   sum_{j != i} theta~_ij; net = to - from.

Scale note: Tables 3 and 4 of Chhimwal, Pandey and Pandey (2025) report the
bottom-right "total" as the sum of the "From" column, without dividing by
N = 2 (for example 42.71 + 41.29 = 84.00). On the standard DY scale the same
matrix gives 42.00. ``paper_scale_total`` reproduces their convention so the
two can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VarFit:
    A: np.ndarray  # (p, N, N)
    c: np.ndarray  # (N,)
    Sigma: np.ndarray  # (N, N)
    resid: np.ndarray  # (T - p, N)
    p: int


def fit_var(Y, p: int) -> VarFit:
    Y = np.asarray(Y, float)
    T, N = Y.shape
    X = np.hstack([np.ones((T - p, 1))] + [Y[p - l : T - l] for l in range(1, p + 1)])
    Z = Y[p:]
    B, *_ = np.linalg.lstsq(X, Z, rcond=None)
    U = Z - X @ B
    dof = max(T - p - X.shape[1], 1)
    Sigma = U.T @ U / dof
    A = np.stack([B[1 + (l - 1) * N : 1 + l * N].T for l in range(1, p + 1)])
    return VarFit(A=A, c=B[0], Sigma=Sigma, resid=U, p=p)


def ma_coefficients(A: np.ndarray, H: int) -> np.ndarray:
    p, N, _ = A.shape
    Psi = np.zeros((H, N, N))
    Psi[0] = np.eye(N)
    for h in range(1, H):
        for l in range(1, min(h, p) + 1):
            Psi[h] += A[l - 1] @ Psi[h - l]
    return Psi


def gfevd(A: np.ndarray, Sigma: np.ndarray, H: int, normalise: bool = True) -> np.ndarray:
    Psi = ma_coefficients(A, H)
    num = np.einsum("hij,jk->hik", Psi, Sigma) ** 2  # (e_i' Psi_h Sigma e_k)^2
    num = num.sum(axis=0) / np.diag(Sigma)[None, :]
    den = np.einsum("hij,jk,hlk->hil", Psi, Sigma, Psi)
    den = np.diagonal(den, axis1=1, axis2=2).sum(axis=0)
    theta = num / den[:, None]
    if normalise:
        theta = theta / theta.sum(axis=1, keepdims=True)
    return theta


def cholesky_fevd(A: np.ndarray, Sigma: np.ndarray, H: int, order=None) -> np.ndarray:
    """Orthogonalised (Cholesky) forecast error variance decomposition, rows sum to 1.

    The structural shocks come from the Cholesky factor of Sigma with the
    variables taken in ``order`` (a list of variable indices, first = not
    affected on impact by the others; default 0, 1, ..., N - 1). Entry (i, j)
    is the share of variable i's H-step forecast error variance due to the
    shock of variable j (sum over h = 0, ..., H - 1), as in Diebold and
    Yilmaz (2009). The result depends on the order.
    """
    Sigma = np.asarray(Sigma, float)
    N = Sigma.shape[0]
    order = list(range(N)) if order is None else list(order)
    L = np.linalg.cholesky(Sigma[np.ix_(order, order)])
    B = np.zeros((N, N))
    B[np.ix_(order, range(N))] = L  # impact of shock k (variable order[k]) on each original variable
    Psi = ma_coefficients(np.asarray(A, float), H)
    contrib = (np.einsum("hij,jk->hik", Psi, B) ** 2).sum(axis=0)  # (N, N): column k = shock order[k]
    theta = np.zeros((N, N))
    theta[:, order] = contrib
    return theta / theta.sum(axis=1, keepdims=True)


def fevd_table(A: np.ndarray, Sigma: np.ndarray, H: int, method: str = "generalized_row") -> np.ndarray:
    """Variance decomposition table (fractions) under one of several conventions.

    ``generalized_row``: Diebold and Yilmaz (2012), each row divided by its sum (same as ``gfevd``).
    ``generalized_raw``: Pesaran and Shin (1998) shares, not normalised (a row can sum to more than 1).
    ``generalized_column``: each column divided by its sum.
    ``cholesky``: orthogonalised shares with variables in their given order.
    ``cholesky_reversed``: orthogonalised shares with the order reversed.
    """
    if method == "generalized_row":
        return gfevd(A, Sigma, H, normalise=True)
    if method == "generalized_raw":
        return gfevd(A, Sigma, H, normalise=False)
    if method == "generalized_column":
        th = gfevd(A, Sigma, H, normalise=False)
        return th / th.sum(axis=0, keepdims=True)
    N = np.asarray(Sigma).shape[0]
    if method == "cholesky":
        return cholesky_fevd(A, Sigma, H, order=range(N))
    if method == "cholesky_reversed":
        return cholesky_fevd(A, Sigma, H, order=range(N - 1, -1, -1))
    raise ValueError(f"unknown method {method!r}")


@dataclass
class Connectedness:
    table: np.ndarray  # row-normalised GFEVD in percent
    total: float  # DY standard scale
    paper_scale_total: float  # sum of off-diagonal entries (Chhimwal et al. scale)
    to_others: np.ndarray
    from_others: np.ndarray
    net: np.ndarray


def connectedness(Y, p: int = 2, H: int = 10) -> Connectedness:
    fit = fit_var(Y, p)
    return connectedness_from_fit(fit, H)


def connectedness_from_fit(fit: VarFit, H: int) -> Connectedness:
    th = 100 * gfevd(fit.A, fit.Sigma, H)
    N = th.shape[0]
    off = th - np.diag(np.diag(th))
    to = off.sum(axis=0)
    frm = off.sum(axis=1)
    return Connectedness(
        table=th,
        total=float(off.sum() / N),
        paper_scale_total=float(off.sum()),
        to_others=to,
        from_others=frm,
        net=to - frm,
    )


def table_from_paper(rows) -> Connectedness:
    """Rebuild summary numbers from a published 2 x 2 (or N x N) row-percent table."""
    th = np.asarray(rows, float)
    N = th.shape[0]
    off = th - np.diag(np.diag(th))
    return Connectedness(th, float(off.sum() / N), float(off.sum()), off.sum(0), off.sum(1), off.sum(0) - off.sum(1))


def block_bootstrap_total(
    Y,
    p: int = 2,
    H: int = 10,
    n_boot: int = 500,
    block: int = 30,
    seed: int | None = 0,
    segments=None,
) -> np.ndarray:
    """Moving-block bootstrap distribution of total connectedness (DY scale).

    Resamples blocks of consecutive rows, which keeps volatility clustering
    and short-run lead-lag effects inside each block. ``segments`` (array of
    segment ids, for example one per hour or day) restricts blocks so they
    never cross from one segment into another.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, float)
    T = len(Y)
    if segments is None:
        starts = np.arange(0, T - block + 1)
    else:
        seg = np.asarray(segments)
        ok = np.array([seg[s] == seg[s + block - 1] for s in range(T - block + 1)])
        starts = np.arange(T - block + 1)[ok]
    n_blocks = int(np.ceil(T / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n_blocks)])[:T]
        out[b] = connectedness(Y[idx], p, H).total
    return out


def rolling_total(Y, window: int, p: int = 2, H: int = 10, step: int = 1) -> np.ndarray:
    Y = np.asarray(Y, float)
    idx = range(window, len(Y) + 1, step)
    return np.array([connectedness(Y[e - window : e], p, H).total for e in idx])
