# Black-76 greeks and gamma exposure

Code: `src/niftygex/greeks.py`, `src/niftygex/gex.py`. Tests: `tests/test_greeks.py`, `tests/test_gex.py`.

## 1. Why Black-76

Nifty options are European, cash settled, and hedged with Nifty futures. Pricing off the futures price $F$ removes the need to estimate the index dividend yield. With time to expiry $T$ (years), rate $r$ and implied volatility $\sigma$:

$$d_1 = \frac{\ln(F/K) + \tfrac12\sigma^2 T}{\sigma\sqrt T}, \qquad d_2 = d_1 - \sigma\sqrt T$$

$$C = e^{-rT}\,[F\,N(d_1) - K\,N(d_2)], \qquad P = e^{-rT}\,[K\,N(-d_2) - F\,N(-d_1)]$$

Put-call parity $C - P = e^{-rT}(F - K)$ holds exactly (tested).

## 2. Greeks used

| Greek | Formula (with respect to $F$) | Use |
| --- | --- | --- |
| Delta | $e^{-rT}N(d_1)$ for calls, $e^{-rT}(N(d_1)-1)$ for puts | hedge ratio |
| Gamma | $e^{-rT}\varphi(d_1)/(F\sigma\sqrt T)$ | GEX |
| Vega | $e^{-rT}F\varphi(d_1)\sqrt T$ | IV solver |
| Vanna | $-e^{-rT}\varphi(d_1)\,d_2/\sigma$ | how hedges change when IV moves |
| Charm | $\frac{d\Delta}{dt} = r e^{-rT}N(d_1) + e^{-rT}\varphi(d_1)\frac{d_2}{2T}$ (call; put replaces $N(d_1)$ with $N(d_1)-1$) | how hedges change as time passes, largest on expiry day |

Derivation of charm: $\partial d_1/\partial T = -d_2/(2T)$, so $\partial \Delta_{call}/\partial T = -r e^{-rT}N(d_1) - e^{-rT}\varphi(d_1)\,d_2/(2T)$, and $d/dt = -\partial/\partial T$. Every greek is checked against central finite differences in the tests.

Spot versus futures gamma: with $F = S e^{bT}$ ($b$ = carry), $\Gamma_S = \Gamma_F e^{2bT}$. For a weekly option $bT < 0.002$, so the factor is below 1.004. `spot_gamma` applies it explicitly.

## 3. Implied volatility solver

Vectorised Newton on $\sigma$ with a bracket $[\sigma_{lo}, \sigma_{hi}]$ that shrinks after every price evaluation (price is increasing in $\sigma$). If the Newton step leaves the bracket, or vega is below $10^{-12}$, a bisection step is used instead. This matters on expiry day: far from the money, vega is close to zero and plain Newton diverges.

Prices outside the no-arbitrage band $(\text{intrinsic}, \text{upper bound})$ return NaN. In the round-trip test (5,000 random options including 10-minute-to-expiry ones) the maximum IV error is below $10^{-6}$ wherever the price carries time value. Options worth only intrinsic value have no IV information; the panel builder fills them from the nearest strike on the same side, which only affects their (negligible) gamma.

## 4. Gamma exposure

For one option series $k$ with spot gamma $\Gamma_k$, open interest $OI_k$ (contracts) and lot size $L$, the rupee change in the holder's delta position for a 1% index move is

$$\text{GEX}_k = \Gamma_k \cdot OI_k \cdot L \cdot S^2 \cdot 0.01$$

Derivation. The holder's delta in index units is $\Delta_k OI_k L$; its rupee value is $\Delta_k OI_k L S$. A move $dS = 0.01S$ changes $\Delta_k$ by $\Gamma_k\,dS$, so the delta value changes by $\Gamma_k OI_k L S \cdot 0.01 S$ (to first order, ignoring the $\Delta\,dS$ term that does not depend on gamma). The test `test_gex_is_rupee_delta_change_for_one_percent_move` checks this against a finite difference.

Net dealer GEX:

$$\text{GEX} = \sum_k w_k\,\text{GEX}_k, \qquad w_k \in [-1, 1]$$

where $w_k$ is the dealers' net share of the open interest in series $k$. Positive GEX means dealers are net long gamma: to stay hedged they sell into rallies and buy into dips, which damps moves. Negative GEX means they chase moves, which amplifies them (Ni, Pearson, Poteshman and White 2021).

### The sign problem

NSE does not publish who holds each strike. The available rules:

| Rule | $w$ for calls | $w$ for puts | Assumption |
| --- | --- | --- | --- |
| `us_convention` | +1 | -1 | customers sell calls and buy puts (US folklore) |
| `participant` | Pro net share | Pro net share | NSE "Pro" category ≈ dealers; same weight at every strike |
| `client_mirror` | minus Client net share | minus Client net share | dealers take the other side of retail |
| `long_all`, `short_all` | +1 or -1 | +1 or -1 | bounds |

Participant net share for a group $g$: $w = (\text{long}_g - \text{short}_g)/\text{total OI}$, with totals from the TOTAL row. Two limitations to state plainly: the participant file covers all index options (Nifty, Bank Nifty and others), and it is not by strike or by expiry. So the participant rule is a noisy, aggregate proxy. The simulation study (E4) measures what that noise costs.

Note on India: SEBI's 2024 study shows individual traders are large net option buyers. If dealers are therefore net short both calls and puts, the US rule (which assumes long calls) can get the sign of call gamma wrong. That is a hypothesis to test on real data, not an assumption we make.

### Derived levels

* **Gamma flip:** the index level $S^*$ nearest to spot where $\text{GEX}(S^*) = 0$, holding OI and IVs fixed. Found on a grid of $\pm4\%$ and refined with Brent's method. Under a rule that gives every series the same sign, there is no flip (NaN).
* **OI walls:** strikes with the largest call OI, put OI and total OI.
* **Max pain:** the strike minimising the total intrinsic value paid to option holders, $\arg\min_s \sum_k OI_k \max(\pm(s - K_k), 0)$.

## 5. Timing

The bhavcopy is published after the close. Daily GEX computed from day $t$'s bhavcopy is used only for forecasts on day $t+1$ and later. On the morning of expiry day, the relevant time to expiry is one trading day plus the session, so gamma is very concentrated near the money.
