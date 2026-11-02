# Connectedness, dependence-aware inference and forecast evaluation

Code: `src/niftygex/connectedness.py`, `src/niftygex/inference.py`, `src/niftygex/evaluation.py`.

## 1. Diebold-Yilmaz connectedness

Fit a VAR($p$) by OLS, $y_t = c + \sum_{l=1}^p A_l y_{t-l} + u_t$, $\text{Cov}(u_t) = \Sigma$. Its moving-average matrices are $\Psi_0 = I$ and $\Psi_h = \sum_{l=1}^{\min(h,p)} A_l \Psi_{h-l}$.

The generalized forecast-error variance decomposition at horizon $H$ (Pesaran and Shin 1998):

$$\theta_{ij}(H) = \frac{\sigma_{jj}^{-1}\sum_{h=0}^{H-1}(e_i'\Psi_h\Sigma e_j)^2}{\sum_{h=0}^{H-1} e_i'\Psi_h\Sigma\Psi_h' e_i}$$

Rows are normalised, $\tilde\theta_{ij} = \theta_{ij}/\sum_j\theta_{ij}$, and reported in percent.

| Measure | Formula |
| --- | --- |
| Total connectedness (DY 2012) | $C = \frac{100}{N}\sum_{i\neq j}\tilde\theta_{ij}$ |
| From others to $i$ | $\sum_{j\neq i}\tilde\theta_{ij}$ |
| To others from $j$ | $\sum_{i\neq j}\tilde\theta_{ij}$ |
| Net | to minus from |

**Scale used in the paper's Tables 3 and 4.** The bottom-right "total" there is the sum of the "From" column without dividing by $N=2$. Example: Table 3, weekly and monthly expiry before the change, has off-diagonal entries 42.71 and 41.29; the paper prints 84.00, the DY index is 42.00. Every comparison in the paper is on the same scale, so its conclusions are unaffected, but anyone comparing with other DY studies must halve the published totals. `Connectedness.paper_scale_total` reproduces the paper's number.

**Can an off-diagonal share exceed 50%?** Table 3's last-hour panel has 52.79 and 52.43. With no dynamics, the generalized decomposition gives a cross share of $\rho^2/(1+\rho^2) < 50\%$ for any correlation $\rho$ (tested: $\rho=0.85$ gives 41.9%). A share above 50% needs strong lagged spillovers. Experiment E3 shows that a lead coefficient of 0.9 with $\rho=0.85$ gives 50.9%. So the published value is possible, but only with strong dynamics, which is plausible for VAR on persistent conditional-volatility series. The paper does not say whether the VAR is on returns or on mcsGARCH volatilities, nor which $p$ and $H$ were used.

**Validation:** closed form at $H=1$, zero spillover for independent series, VAR coefficient recovery and directional spillover sign (`tests/test_connectedness.py`).

**Uncertainty.** `block_bootstrap_total` resamples blocks of consecutive observations (Künsch 1989), optionally never crossing a segment boundary such as the end of an hour or day. Experiment E3 uses it to put confidence intervals on totals estimated from about 240 observations, the size of a "one trading hour on four monthly expiry days" subsample.

## 2. Before/after tests

The paper compares per-minute values (volume, Rogers-Satchell variance) before and after the expiry change with a Mann-Whitney-Wilcoxon test on pooled minutes.

The test assumes independent observations. Minutes within a day are not independent: all of them share that day's volatility level, and volatility clusters from minute to minute. With 360 minutes per day, a sample of 13 days is not 4,680 independent draws; its effective size is much closer to 13. A test that treats it as 4,680 will reject a true "no change" null far more often than 5%.

Alternatives implemented:

| Test | What it does | Assumption |
| --- | --- | --- |
| `minute_level_mwu` | paper's approach | minutes independent |
| `day_level_mwu` | median per day, then Mann-Whitney on days | days independent |
| `hac_mean_difference` | regress log value on an "after" dummy, Newey-West SE with one day of lags | dependence dies out within about a day |
| `day_cluster_bootstrap` | resample whole days, bootstrap the difference in medians | days independent |

Experiment E2 measures the actual rejection rate of each test under a true null and under 10% and 25% increases in variance.

## 3. Forecast evaluation

**Target.** Realized variance over a window, $RV_{t,i}(H) = \sum_{k=0}^{H-1} r_{t,i+k}^2$.

**Losses** (Patton 2011, robust to noise in the realized-variance proxy):

$$\text{QLIKE}(y, f) = \frac{y}{f} - \ln\frac{y}{f} - 1, \qquad \text{MSE}(y,f) = (y-f)^2$$

QLIKE is the headline: it is scale-free, so calm and volatile days count equally, and it punishes under-prediction of variance more than over-prediction.

**Log-target models and bias.** Models fitted on $\ln y$ under-predict $y$ on average, because $E[e^X] > e^{E[X]}$. Every log-scale model is multiplied by Duan's smearing factor $\frac1n\sum e^{\hat u}$ estimated on its own training residuals.

**Diebold-Mariano.** On the loss differential $d_\tau = L^{(1)}_\tau - L^{(2)}_\tau$, $DM = \bar d/\sqrt{\widehat{LRV}/n}$ with a Bartlett-kernel long-run variance and the Harvey, Leybourne and Newbold small-sample factor. Losses are first averaged within each day, so $n$ is the number of test days. This avoids treating anchors on the same day as independent, the same mistake discussed in Section 2.

**GEX ablation.** The test of whether positioning carries information: the same model (same algorithm, data, training windows and tuning) is fitted with and without the GEX feature block, and a one-sided DM test asks whether the GEX version has lower out-of-sample loss.

**Model Confidence Set** (Hansen, Lunde and Nason 2011) via `arch.bootstrap.MCS` on daily losses with a block bootstrap, to report which models cannot be statistically separated from the best.

## 4. Walk-forward protocol

* The first `min_train` days (150 by default) are training only.
* Every `block` days (21 by default, about a month), all models are refitted on everything before the block and forecast the block.
* The daily EGARCH and the mcsGARCH are fitted on training days only, then filtered forward with fixed parameters, so forecasts on test days use only past data.
* Hyperparameters (when tuning is on) are chosen on the last 20% of each training window, by QLIKE.
* Bhavcopy, participant OI and India VIX published at the close of day $t-1$ enter only day $t$'s forecasts.
