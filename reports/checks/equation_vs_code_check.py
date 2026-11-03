"""OUR CHECK: independent numeric checks of load-bearing equations against niftygex code.

Run:  OMP_NUM_THREADS=1 python reports/checks/equation_vs_code_check.py
Each check recomputes the quantity from the textbook formula (or a brute-force simulation) without calling the
function under test for the reference value, and compares. Prints one line per check, PASS or FAIL.
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from niftygex import greeks, gex, evaluation, realized, connectedness as cn, volatility as vol
from niftygex.expiry_calendar import weekly_expiries

res = []
def check(name, ok, detail=""):
    res.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""))

# 1. Black-76 gamma vs second finite difference of the price (F, K, T, r, sigma typical weekly Nifty)
F, K, T, r, s = 24000.0, 24100.0, 3 / 365, 0.065, 0.14
h = 1.0
for is_call in (True, False):
    p = lambda x: float(greeks.price(x, K, T, r, s, is_call))
    fd = (p(F + h) - 2 * p(F) + p(F - h)) / h**2
    g = float(greeks.gamma(F, K, T, r, s))
    check(f"B76 gamma vs finite difference ({'call' if is_call else 'put'})", abs(fd - g) / g < 1e-5, f"fd={fd:.6e} code={g:.6e}")
# put-call parity
c = float(greeks.price(F, K, T, r, s, True)); pu = float(greeks.price(F, K, T, r, s, False))
check("B76 put-call parity C-P = e^-rT (F-K)", abs((c - pu) - np.exp(-r * T) * (F - K)) < 1e-8)
# delta via FD
fdD = (float(greeks.price(F + h, K, T, r, s, True)) - float(greeks.price(F - h, K, T, r, s, True))) / (2 * h)
check("B76 call delta vs finite difference", abs(fdD - float(greeks.delta(F, K, T, r, s, True))) < 1e-6)
# spot gamma with carry: price as function of S with F = S e^{cT}
carry = 0.06
S0 = 24000.0
pS = lambda S: float(greeks.price(S * np.exp(carry * T), K, T, r, s, True))
fdS = (pS(S0 + h) - 2 * pS(S0) + pS(S0 - h)) / h**2
sg = float(greeks.spot_gamma(S0 * np.exp(carry * T), K, T, r, s, carry))
check("spot_gamma = gamma_F * (dF/dS)^2 vs finite difference in S", abs(fdS - sg) / sg < 1e-5, f"fd={fdS:.6e} code={sg:.6e}")

# 2. GEX definition: gamma*OI*L*S^2*0.01 and equals the rupee change of the holder's delta value for a 1% move (first order)
L = 75.0; oi = 123456.0; S = 24000.0
chain = pd.DataFrame({"strike": [23900.0, 24000.0, 24100.0, 23900.0, 24000.0, 24100.0],
                      "is_call": [True, True, True, False, False, False],
                      "oi": [oi] * 6, "iv": [0.15, 0.14, 0.13, 0.16, 0.15, 0.14]})
tab = gex.strike_gex(chain, S, L, T, r, 0.0, "us_convention")
ref_gross = []
for _, row in chain.iterrows():
    d1 = (np.log(S / row.strike) + 0.5 * row.iv**2 * T) / (row.iv * np.sqrt(T))
    gam = np.exp(-r * T) * norm.pdf(d1) / (S * row.iv * np.sqrt(T))
    ref_gross.append(gam * row.oi * L * S**2 * 0.01)
check("gross GEX = Gamma*OI*L*S^2*0.01 (hand formula)", np.allclose(tab.gross_gex, ref_gross, rtol=1e-10))
# delta value change under +1% move (second-order effect dominates only through gamma): compare for a single series
row = chain.iloc[1]
dv = lambda x: float(greeks.delta(x, row.strike, T, r, row.iv, True)) * oi * L * x
fd_gex = (dv(S * 1.005) - dv(S * 0.995))  # symmetric 1% change in delta VALUE includes delta*dS term
# remove the delta*dS term: value change of delta*units at fixed S-weighting -> use delta units change * S
# first-order definition: use a small bump (0.05% each side) and scale to 1%; a full 1% bump is not first order for a 3-day option
dd = (float(greeks.delta(S * 1.0005, row.strike, T, r, row.iv, True)) - float(greeks.delta(S * 0.9995, row.strike, T, r, row.iv, True))) * 10 * oi * L * S
check("GEX = (delta change per 1% move, first order) x units x S", abs(dd - tab.gross_gex.iloc[1]) / tab.gross_gex.iloc[1] < 1e-3, f"fd={dd:.4e} code={tab.gross_gex.iloc[1]:.4e}")
dd1 = (float(greeks.delta(S * 1.005, row.strike, T, r, row.iv, True)) - float(greeks.delta(S * 0.995, row.strike, T, r, row.iv, True))) * oi * L * S
print(f"INFO  a full +/-0.5% (1%) bump gives {dd1 / tab.gross_gex.iloc[1] - 1:+.1%} versus the first-order GEX for this 3-day ATM option (gamma changes over a 1% move)")

# 3. Sign rules
w = gex.sign_weights(np.array([True, False]), "us_convention")
check("sign rule us_convention: +1 calls, -1 puts", list(w) == [1.0, -1.0])
check("sign rules long_all / short_all", list(gex.sign_weights(np.array([True, False]), "long_all")) == [1, 1] and list(gex.sign_weights(np.array([True, False]), "short_all")) == [-1, -1])
pw = gex.ParticipantWeights(call=0.2, put=-0.1)
check("sign rules participant/client_mirror use per-type weights", list(gex.sign_weights(np.array([True, False]), "participant", pw)) == [0.2, -0.1])
tbl = pd.DataFrame({"option_index_call_long": [30., 70., 100.], "option_index_call_short": [60., 40., 100.],
                    "option_index_put_long": [50., 50., 100.], "option_index_put_short": [10., 90., 100.]}, index=["Pro", "Client", "TOTAL"])
pwx = gex.ParticipantWeights.from_participant_oi(tbl, "Pro")
check("participant weight = (long-short)/total OI", np.isclose(pwx.call, (30 - 60) / 100) and np.isclose(pwx.put, (50 - 10) / 100))
pwm = gex.ParticipantWeights.from_participant_oi(tbl, "Pro", mirror=True)
check("mirror flips sign", np.isclose(pwm.call, -pwx.call) and np.isclose(pwm.put, -pwx.put))
check("net GEX = sum(weight*gross)", np.isclose(tab.gex.sum(), (tab.weight * tab.gross_gex).sum()))
check("SIGN_RULES module constant lists five rules", len(gex.SIGN_RULES) == 5)

# 4. QLIKE (Patton 2011): y/f - ln(y/f) - 1; zero at f=y, minimised in f at f=y, asymmetric (under-prediction costs more)
y = 2.0
fs = np.linspace(0.5, 6, 2000)
q = evaluation.qlike(np.full_like(fs, y), fs)
check("QLIKE minimised at f = y with value 0", abs(fs[np.argmin(q)] - y) < 0.01 and evaluation.qlike(y, y) == 0)
check("QLIKE penalises under-prediction more than equal-ratio over-prediction", evaluation.qlike(1.0, 0.5) > evaluation.qlike(1.0, 2.0))
check("QLIKE scale-free: qlike(c*y, c*f) = qlike(y, f)", np.isclose(evaluation.qlike(3e-6, 2e-6), evaluation.qlike(3.0, 2.0)))

# 5. Diebold-Mariano: independent reimplementation (Bartlett LRV, HLN h=1, Student t n-1) and size check with AR(1) differentials
rng = np.random.default_rng(1)
def dm_ref(d):
    n = len(d); lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dc = d - d.mean()
    gam = [dc[k:] @ dc[:n - k] / n for k in range(lag + 1)]
    lrv = gam[0] + 2 * sum((1 - k / (lag + 1)) * gam[k] for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(lrv / n) * np.sqrt((n - 1) / n)
    return stat, 2 * student_t.sf(abs(stat), n - 1)
d = rng.standard_normal(792) * 0.3 + 0.02
out = evaluation.diebold_mariano(d, np.zeros_like(d))
st, pp = dm_ref(d)
check("DM statistic and p equal independent reimplementation", np.isclose(out["stat"], st) and np.isclose(out["p"], pp), f"stat code={out['stat']:.4f} ref={st:.4f}")
from scipy.signal import lfilter
def size(phi, N=3000, n=792):
    rj = 0
    for _ in range(N):
        e = rng.standard_normal(n + 100)
        x = lfilter([1.0], [1.0, -phi], e)[100:]
        rj += evaluation.diebold_mariano(x, np.zeros_like(x))["p"] < 0.05
    return rj / N
s0, s3 = size(0.0), size(0.3)
print(f"INFO  DM empirical size at nominal 5%, n=792 days, 3000 reps: iid differentials {s0:.3f}, AR(1)=0.3 differentials {s3:.3f}")
check("DM size iid differentials within 4% to 6%", 0.04 <= s0 <= 0.06, f"{s0:.3f}")
check("DM size AR(1)=0.3 differentials at most 8% (mild over-rejection from the Bartlett lag rule)", s3 <= 0.08, f"{s3:.3f}")
# one-sided direction: d = l1 - l2 > 0 means model 2 better
o = evaluation.diebold_mariano(np.full(100, 1.0) + rng.standard_normal(100) * 0.1, np.zeros(100), alternative="greater")
check("DM 'greater' small p when model 2 has lower loss", o["p"] < 1e-6)

# 6. Realized measures
rng = np.random.default_rng(2)
sig = 0.0002  # per-minute sd
rets = rng.standard_normal((4000, 375)) * sig
rv = realized.realized_variance(rets)
check("RV mean = n*sigma^2 (Brownian)", abs(rv.mean() / (375 * sig**2) - 1) < 0.01)
bv = realized.bipower_variation(rets)
check("bipower variation unbiased for sigma^2*n (no jumps)", abs(bv.mean() / (375 * sig**2) - 1) < 0.01)
# Rogers-Satchell: unbiased for continuously monitored Brownian motion; discrete monitoring biases it down, converging as steps grow
def rs_ratio(nsub, nbar=6000):
    step = np.sqrt(1.0 / nsub) * 0.01
    inc = rng.standard_normal((nbar, nsub)) * step
    lp = np.hstack([np.zeros((nbar, 1)), np.cumsum(inc, axis=1)])
    rs = realized.rogers_satchell(np.exp(lp[:, 0]), np.exp(lp.max(axis=1)), np.exp(lp.min(axis=1)), np.exp(lp[:, -1]))
    return rs.mean() / 0.01**2
r60, r600, r6000 = rs_ratio(60), rs_ratio(600), rs_ratio(6000, 1500)
print(f"INFO  Rogers-Satchell mean / true bar variance with 60, 600, 6000 sub-steps per bar: {r60:.3f}, {r600:.3f}, {r6000:.3f}")
check("Rogers-Satchell ratio rises toward 1 as the sampling gets finer, within 5% at 6000 sub-steps", r60 < r600 < r6000 + 0.02 and abs(r6000 - 1) < 0.05)

# 7. Connectedness (Diebold-Yilmaz 2012, GFEVD row-normalised): independent computation on a VAR(1)
A1 = np.array([[0.5, 0.2], [0.1, 0.4]]); Sig = np.array([[1.0, 0.5], [0.5, 1.0]]); H = 10
Psi = [np.eye(2)]
for hh in range(1, H): Psi.append(A1 @ Psi[-1])
th = np.zeros((2, 2))
for i in range(2):
    den = sum((Psi[hh] @ Sig @ Psi[hh].T)[i, i] for hh in range(H))
    for j in range(2):
        th[i, j] = sum((Psi[hh] @ Sig)[i, j] ** 2 for hh in range(H)) / Sig[j, j] / den
th = th / th.sum(axis=1, keepdims=True)
C_ref = 100 * (th.sum() - np.trace(th)) / 2
cobj = cn.connectedness_from_fit(cn.VarFit(A=A1[None], c=np.zeros(2), Sigma=Sig, resid=np.zeros((1, 2)), p=1), H)
check("DY total connectedness = 100/N * sum off-diagonal of row-normalised GFEVD", np.isclose(cobj.total, C_ref), f"code={cobj.total:.4f} ref={C_ref:.4f}")
check("paper-scale total = 2x DY total for N=2", np.isclose(cobj.paper_scale_total, 2 * cobj.total))
# no dynamics, rho=0.85 -> 41.9%
cb0 = cn.connectedness_from_fit(cn.VarFit(A=np.zeros((1, 2, 2)), c=np.zeros(2), Sigma=np.array([[1, .85], [.85, 1]]), resid=np.zeros((1, 2)), p=1), 10)
check("no dynamics, rho=0.85: cross share = rho^2/(1+rho^2) = 41.9% (doc claim)", abs(cb0.table[0, 1] - 41.94) < 0.05, f"code={cb0.table[0,1]:.2f}")

# 8. mcsGARCH multi-step forecast vs brute-force simulation of the GARCH(1,1) recursion
rng = np.random.default_rng(3)
om, al, be = 0.02, 0.05, 0.92
qbar = om / (1 - al - be)
mc = vol.McsGarch()
mc.params = {"omega": om, "alpha": al, "beta": be, "nu": 8.0}
M = 30; i0 = 5; Hh = 12
S = np.linspace(1.5, 0.6, M)[None, :]
q2 = np.full((1, M), 1.7)  # q_{i0} known at end of minute i0-1
filt = {"S": S, "q2": q2}
fc = mc.forecast_variance(filt, np.array([2.0]), [0], [i0], Hh)[0]
# brute force: simulate x_j = sqrt(q_j) z_j, accumulate E[h*S_j*x_j^2]; E[x_j^2]=q_j so simulate q path means
nsim = 200000
q = np.full(nsim, 1.7); tot = np.zeros(nsim)
for k in range(Hh):
    tot += 2.0 * S[0, i0 + k] * q
    x2 = q * rng.standard_normal(nsim) ** 2
    q = om + al * x2 + be * q
check("mcsGARCH analytic multi-step variance forecast vs Monte Carlo", abs(tot.mean() / fc - 1) < 0.01, f"analytic={fc:.4f} mc={tot.mean():.4f}")
# standardised-t log-likelihood integrates to 1 and has unit variance
xx = np.linspace(-80, 80, 400001); nu = 7.0
dens = np.exp(vol._t_loglik(xx, np.ones_like(xx), nu))
check("standardised-t likelihood integrates to 1 with unit variance", abs(np.trapezoid(dens, xx) - 1) < 1e-4 and abs(np.trapezoid(dens * xx**2, xx) - 1) < 2e-3)

# 9. Expiry calendar: Thursday weekly before 1 Sep 2025, Tuesday from; holiday roll-back
days = pd.bdate_range("2025-08-18", "2025-09-19")
ex = weekly_expiries(days)
check("expiry: Thursday 28 Aug 2025 then Tuesday 2 Sep 2025", pd.Timestamp("2025-08-28") in ex and pd.Timestamp("2025-09-02") in ex and pd.Timestamp("2025-09-04") not in ex)
days2 = days.drop(pd.Timestamp("2025-09-09"))
ex2 = weekly_expiries(days2)
check("expiry: holiday on Tuesday rolls back to Monday", pd.Timestamp("2025-09-08") in weekly_expiries(days.drop(pd.Timestamp("2025-09-09"))) or pd.Timestamp("2025-09-08") in ex2)

# 10. HAR windows (t-5..t-1 and t-22..t-1)
rv_ = np.exp(rng.standard_normal(80) * 0.3) * 1e-4
hh, info = vol.har_daily_variance(rv_)
t = 40
X = [1, np.log(rv_[t - 1]), np.log(rv_[t - 5:t].mean()), np.log(rv_[t - 22:t].mean())]
check("HAR regressors use means of days t-5..t-1 and t-22..t-1", np.isclose(np.log(hh[t]), np.dot(X, list(info['params'].values()))))

bad = [n for n, ok in res if not ok]
print(f"\n{len(res) - len(bad)}/{len(res)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
