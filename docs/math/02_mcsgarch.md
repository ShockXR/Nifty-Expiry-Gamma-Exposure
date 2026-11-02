# mcsGARCH: the paper's volatility model, rebuilt

Code: `src/niftygex/volatility.py`. Tests: `tests/test_volatility.py`. Reference: Engle and Sokalska (2012), as used in Chhimwal, Pandey and Pandey (2025), Section 3.1.

## 1. Model

For day $t$ and minute $i$ (of $M$ per day):

$$r_{t,i} = \mu_{t,i} + \varepsilon_{t,i}, \qquad \varepsilon_{t,i} = \sqrt{h_t\, s_i\, q_{t,i}}\; z_{t,i}$$

| Part | Meaning | How it is estimated |
| --- | --- | --- |
| $h_t$ | daily variance, known before the open | AR(1)-EGARCH(1,1) on daily returns (paper eq. 4 to 6), via the `arch` package, fitted on the training window only and then filtered forward with fixed parameters |
| $s_i$ | diurnal variance pattern | $s_i = \frac1T\sum_t \varepsilon_{t,i}^2 / h_t$ (paper eq. 7) |
| $q_{t,i}$ | intraday stochastic variance | GARCH(1,1) on $x_{t,i} = \varepsilon_{t,i}/\sqrt{h_t s_i}$ (paper eq. 8 and 9) |
| $z_{t,i}$ | innovation | standardised Student-$t$ with $\nu$ degrees of freedom |

$$q_{n} = \omega + \alpha x_{n-1}^2 + \beta q_{n-1}$$

with $n$ running across day boundaries (as in rugarch).

## 2. A notation point in the paper

The paper's eq. 3 writes the conditional volatility as a product $q_{t,i}\sigma_t s_i$, which reads as volatility-scale components. Its eq. 7 defines $s_i$ as an average of *squared* residuals over *squared* daily volatility, which is variance scale. Its eq. 8 then divides by "$\sigma_i s_i$". Read literally these do not combine: if $s_i$ is on the variance scale, the normalised residual must divide by $\sqrt{h_t s_i}$. Engle and Sokalska define the components so that this is consistent, and so does this implementation (everything on the variance scale, square root taken once). This is a typographical issue, not a modelling error, and it does not affect the paper's results. Eq. 9 also prints the ARCH coefficient as $\sigma$ rather than $\alpha$.

## 3. Estimation steps

1. **Mean.** Options: zero, constant, or ARMA(1,1) by conditional least squares (the paper uses ARMA(1,1)). On minute data the mean is negligible relative to volatility, which matters for point 5 below.
2. **Diurnal curve.** Average $\varepsilon^2/h_t$ by minute of day. Optional centred moving-average smoothing (the raw average is noisy: with 250 days the largest error on any single minute was about 55% in the simulation check, even though the overall shape correlates at 0.91 with the truth).
3. **Extension: expiry-day diurnal curve.** If `expiry_diurnal=True`, a second curve $s_i^{E}$ is estimated from expiry days only. The paper shows expiry days behave differently, so it makes sense to let their intraday shape differ. Expiry days are about one in five, so this curve is noisier and should be smoothed.
4. **Intraday GARCH.** Maximum likelihood with standardised $t$ errors. Parameters are mapped to an unconstrained space ($\omega = e^{a}$, persistence $\alpha+\beta = 0.99999\cdot\text{logit}^{-1}(b)$, share $\alpha/(\alpha+\beta) = \text{logit}^{-1}(c)$, $\nu = 2.05 + e^{d}$) so that the optimiser cannot leave the stationary region. The recursion is evaluated with a linear filter, so a 180,000-minute likelihood takes milliseconds. Missing minutes stay in the recursion but get zero weight in the likelihood.
5. **z statistics** from a numerical Hessian, to compare with the paper's Table 2.

## 4. Forecasting

At anchor $(t, i)$, using data up to the end of minute $i-1$, the forecast of the variance of the next $H$ minutes is

$$\hat V_{t,i}(H) = h_t \sum_{k=0}^{H-1} s_{i+k}\,\mathbb E[q_{t,i+k}], \qquad \mathbb E[q_{t,i+k}] = \bar q + (\alpha+\beta)^{k}(q_{t,i} - \bar q)$$

with $\bar q = \omega/(1-\alpha-\beta)$. The test `test_forecast_is_causal_and_sums_components` checks that changing data after the anchor cannot change the forecast, and that the one-step forecast equals $h_t s_i q_{t,i}$.

## 5. Validation

| Check | Result | Where |
| --- | --- | --- |
| Pure GARCH(1,1)-t, 60,000 obs, true $(\alpha,\beta,\nu) = (0.05, 0.90, 6)$ | recovered within 0.01, 0.02, 1.0 | `test_garch11_recovers_parameters` |
| Full mcsGARCH on 120 simulated days, true $(\alpha,\beta,\nu)=(0.03,0.95,7)$ | recovered within tolerance; diurnal shape correlation above 0.85 | `test_mcsgarch_recovers_simulated_parameters` |
| 250 simulated days (exploratory run) | $\hat\alpha=0.031$, $\hat\beta=0.949$, $\hat\nu=7.2$ | ITERATION_LOG.md, iteration 4 |
| Expiry diurnal curve finds the late-session bump | ratio within 0.5 of truth | `test_expiry_diurnal_captures_late_session_bump` |

All of these are on synthetic data with known parameters. They show the code estimates what it claims to estimate. They say nothing about real Nifty data.

## 6. Reading the paper's Table 2 with this model

Two things follow directly from the published estimates (experiment E6, `results/paper_checks/e6_table_arithmetic.csv`):

* **Unit-mean normalisation.** $\omega/(1-\alpha-\beta)$ is between 1.08 and 1.14 in all four columns, close to 1. That is what one expects if the intraday component is normalised as above, so the published estimates are internally consistent with the model.
* **Persistence fell after the change.** The half-life of an intraday volatility shock, $\ln 0.5/\ln(\alpha+\beta)$, is about 339 minutes for Nifty before and 72 minutes after; for Bank Nifty about 185 and 40 minutes. The paper does not comment on this. If it survives proper standard errors, it says intraday volatility shocks die out much faster in the new expiry regime, which is a finding in its own right and something to ask the authors about.
* **ARMA(1,1) near-cancellation.** In all four columns the AR and MA estimates are close to equal and opposite ($\phi+\theta$ between $-0.04$ and $+0.02$). An ARMA(1,1) with $\phi \approx -\theta$ is close to white noise: the implied first autocorrelation is only between $-0.04$ and $+0.02$. When the true mean dynamics are near zero, $\phi$ and $\theta$ are weakly identified and can be large and individually "significant" while cancelling out. Experiment E5 checks this by simulation. The practical consequence is small (the volatility results do not depend on it), but the significance stars on the AR and MA terms should not be read as evidence of minute-level return predictability.
