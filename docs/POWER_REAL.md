# How large must a dealer-gamma effect be for us to see it? Power on the real sample (P71)

**SIMULATIONS CALIBRATED ON REAL DATA.** Nothing in this document is a new estimate from the market. Every
number comes from simulations in which the explanatory variables, the fitted values and the noise are taken
from the real Nifty data (2 January 2023 to 2 November 2026) and only the outcome is simulated, with an effect
of known size added. The simulations say which effects our tests could and could not have seen. The real-data
estimates themselves are in [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md) (R2, R3, R5) and
[FALSIFICATION.md](FALSIFICATION.md) (the forecast test, T4). Before simulating, the script recomputes each of
them and checks it against the published file (`estimates.csv`, column `matches_published`: true for all
eight). Every number below is read from a file written by `experiments/power_real.py`; the file is named under
each table. The earlier power study (E4, `experiments/power_study.py`) was fully synthetic; this one answers
the same kind of question for our actual sample.

    OMP_NUM_THREADS=1 python experiments/power_real.py        # 408.5 seconds on 4 cores for the run reported here
    OMP_NUM_THREADS=1 python -m pytest -q tests/test_power_real.py

---

## The answer in brief

The question: given how noisy the real data are, how large must an effect be before our tests reliably see it,
and how much more data would we need to see effects of the size we estimated? Put differently, it turns "not
significant" into "we could only have seen effects larger than X".

1. **Our significant results sit just above what the sample can detect.** The smallest effect each test finds
   80% of the time (the minimum detectable effect, MDE) is close to the estimate itself: about 3.0 against an
   estimate of -3.16 for the R3 ordinary-day GEX term, 6.9 to 7.1 against 8.21 for the R3 expiry-day
   interaction, 0.042 to 0.052 against -0.068 for the R5 reversal interaction, and 0.15 to 0.18 log points
   against 0.19 for the Tuesday whole-day expiry premium. If the true effects were half as large as estimated,
   the tests would find them only 20% to 41% of the time for the Tuesday whole-day premium (depending on the
   test), 31% and 37% for the two R3 terms and 45% for R5.
2. **Two "not significant" results, translated.** In the one-weekly-Thursday period (R2, November 2024 to
   August 2025) the test could only have seen an expiry premium of at least 0.34 log points (41% more variance
   than the rest of the week) or an equally large discount; the estimate was -0.119 (p = 0.26), so a moderate
   effect in that period cannot be ruled out. For out-of-sample forecasting, even if the in-sample R3 effect
   were exactly true, the forecast test would detect it only about half the time (50% to 52%); it needs an
   effect 1.46 to 1.49 times as large. The real result (forecasts 5.1% worse with GEX) is no surprise when
   there is no effect at all: 10.7% to 14.1% of simulations without any effect were at least that bad.
3. **Significant estimates from a sample this size tend to be too large.** If the true effect were half our
   estimate, the estimates that happen to pass the 5% test would on average be 1.26 to 1.78 times the true
   value (R5 1.26 to 1.45, R3 1.58 to 1.78, Tuesday whole-day premium 1.44 to 1.60). This "winner's curse" is
   a reason to plan for effects smaller than the published estimates.
4. **Expiries needed** to detect effects half as large as our estimates with 80% power, if the future looks like
   the past (the less favourable of two resampling schemes): R5 about 2.4 times the current sample (283 more
   expiry days, about 5.5 more years); the R3 expiry-day interaction about 3.1 times (417 more, about 8.0
   years); the R3 ordinary-day term about 3.9 times (571 more, about 10.9 years); the Tuesday whole-day premium
   107 more Tuesday expiries (2.1 years) with a t-test of the mean or 214 (4.1 years) with the signed-rank test
   used in R2; the Tuesday last-hour premium 40 more (0.8 years). For the forecast test, three times the data
   still gives only 68% to 73% power at the in-sample effect size, because its model is fitted once on 2023 to
   2024 and more forecast days do not make its coefficients any more precise.
5. **Sign errors can wipe out power.** If the GEX regressor has the wrong sign on 25% of randomly chosen days,
   power at the estimated effect drops from 83% to 23% (R3 ordinary days), from 91% to 27% (R3 expiry days)
   and from 99% to 59% (R5). The three fixed sign rules disagree on 14.9% (US convention and Client mirror),
   47.0% (Pro and Client mirror) and 55.1% (US convention and Pro) of days. At 47.0% the power to find the
   effect with the right sign is 4% to 6%. When the sign is wrong on most days (84.1%, how often the dealer
   sign estimated in [DEALER_SIGN.md](DEALER_SIGN.md) disagrees with Pro in 2025 to 2026), R5 still rejects
   84% of the time, but always with the opposite sign: a significant result is no evidence that the sign rule
   is right.
6. **Size check: R3, R5 and the Tuesday R2 tests hold their 5% level.** With no effect, R3 and R5 reject 5.0%
   to 6.4% of the time, the Tuesday R2 tests 2.6% to 7.6%, and the forecast test 1.8% to 2.7% (it is
   conservative, as expected when a model is compared with its own extension). Two other R2 periods
   over-reject: one weekly on Thursday (12.8% to 17.8%; its weekly premiums are autocorrelated, lag-one
   correlation 0.26) and several weeklies (8.2% to 9.1%). Their published p-values (0.26 and 0.045) are
   therefore somewhat too small.

---

## Words used here

- **Effect**: how much an outcome changes with something else, here a regression coefficient (R3, R5), a mean
  difference in log realized variance (R2) or a forecasting gain (FC).
- **Test, p-value, significant**: a test asks whether an estimate could be noise. The p-value is the chance of
  a result at least this extreme if the true effect were zero. "Significant" here means p below 0.05.
- **Size**: how often a test says "significant" when the true effect is exactly zero. A 5% test should do so
  5% of the time. A test that does so more often over-rejects and its p-values are too small.
- **Power**: how often a test says "significant" when the effect is real and of a given size. The usual target
  is 80%.
- **Minimum detectable effect (MDE80)**: the smallest true effect the test detects 80% of the time. "We could
  only have seen effects larger than X" means X is the MDE.
- **Standard error and 95% interval**: the standard error measures how much an estimate would move from sample
  to sample; the 95% interval is roughly the estimate plus or minus two standard errors. A textbook rule says
  the MDE80 of a 5% two-sided test is about 2.8 standard errors; we report it as a check.
- **Newey-West (HAC) standard errors**: standard errors that allow the noise on nearby days to be related (here
  up to 5 days apart, as in the original scripts).
- **Residual (noise)**: the part of the outcome a regression does not explain.
- **Null model**: the test's own regression without the term being tested.
- **Block bootstrap**: building new noise series by copying stretches of consecutive days (blocks) from the
  real residuals, so that calm and turbulent stretches stay together. The **stationary bootstrap** (Politis
  and Romano 1994) uses blocks of random length with a fixed average.
- **Wild block bootstrap**: each residual stays on its own day and is multiplied by +1 or -1; the sign is the
  same for all days of a block. This keeps the link between the size of the noise and the explanatory
  variables on that day.
- **Semi-parametric simulation**: a simulation that uses the real explanatory variables and the real noise,
  and only an assumed effect, instead of inventing the whole data set.
- **Monte Carlo error**: the noise in a simulated power that comes from running a finite number of
  simulations. With 5,000 simulations it is at most 0.71 percentage points (column `mc_se` of
  `power_curves.csv`).
- **Log points**: a difference of logarithms. 0.19 log points of variance means e^0.19 = 1.21 times the
  variance, that is 21% more.
- **Signed-rank (Wilcoxon) test**: the test R2 uses; it asks whether the week-relative values are centred on
  zero, using their ranks rather than their sizes.
- **Diebold-Mariano (DM) test**: asks whether one forecast has lower average loss than another, allowing the
  daily loss differences to be related over time.
- **Exaggeration ratio (winner's curse)**: among the simulations that come out significant, the average size
  of the estimate divided by the true effect. Above 1 means significant estimates overstate the truth.
- **lambda**: the injected effect as a multiple of the real estimate (lambda = 1 is "an effect exactly as
  large as we estimated").
- **GEX, share measure, Pro rule**: GEX is option dealers' gamma exposure; the share measure is net GEX divided
  by gross GEX of the same chain (between -1 and 1). The Pro rule takes the dealer sign from the NSE
  participant file's Pro category. See [FALSIFICATION.md](FALSIFICATION.md) for full definitions.

---

## Why power matters here

A test that finds nothing can mean two different things: there is no effect, or there is one but the data are
too noisy to show it. Power tells the two apart. If our tests had 80% power for effects of a plausible size, a
null result would be informative; if they had 20%, a null result says little. For results that are
significant, power matters too: when power is modest, the estimates that clear the 5% bar are, on average,
larger than the truth, so a replication on new data will usually find a smaller effect. Both issues apply to
this project, where the samples are short (61 Tuesday expiry weeks, 937 trading days) and Nifty's volatility
is noisy from day to day.

---

## Method

### The tests

Each test is rerun with exactly the code path of the original script (same columns, rows, Newey-West lags and
5% level). The estimates are reproduced from the real data first.

| ID | What it asks (original script) | Sample | Real estimate (p) |
| --- | --- | --- | --- |
| R2 Tuesday | Is the expiry day more volatile than the rest of its week? Mean of (log realized variance on the expiry day minus the mean over the other days of the week); signed-rank test as published, plus a Newey-West t-test of the mean with 4 weekly lags (`real_day_level.r2_regimes`) | 61 Tuesday expiry weeks (whole day and last hour); also 41 one-weekly-Thursday and 98 several-weeklies weeks | whole day 0.192 (0.005), last hour 0.356 (5.0e-5) |
| R3a, R3b | Log realized variance on yesterday's log variance, its 5-day mean, yesterday's log VIX, an expiry dummy, regime x expiry controls, yesterday's GEX share under the Pro rule (R3a) and that share x expiry dummy (R3b); Newey-West 5 lags (`real_day_level.r3_robustness`, variant "share") | 937 days, 199 expiry days | R3a -3.16 (0.004), R3b 8.21 (0.001) |
| R5 | Last-30-minute return on the rest-of-day return, 1[Pro GEX < 0] and their product (the tested term); Newey-West 5 lags (`momentum_gex.py`, specification "binary", all days) | 929 days, 197 expiry days | -0.0682 (2.2e-4) |
| FC | Does adding the two GEX terms improve one-day-ahead forecasts of log realized variance? Both models fitted on 2023 to 2024 and frozen; squared error on 2025 onward; one-sided Diebold-Mariano with Newey-West 5 lags (`falsification.py`, T4) | 485 fitting days, 452 forecast days | forecasts 5.1% worse with GEX (one-sided p 0.98) |

Source: `results/power_real/estimates.csv`.

For the forecast test the injected effect has the shape of the two in-sample R3 estimates (-3.16 x share and
8.21 x share x expiry dummy), multiplied by lambda. lambda = 1 therefore means "the R3 effect is exactly
true".

### One simulation, step by step

1. Fit the null model on the real data (for R3a: everything except the ordinary-day GEX term; for R2: a mean
   of zero).
2. Take the residuals of the full model (with the tested term). We use the full model's residuals because the
   null model's residuals would still contain the effect we estimated, and that effect would then be counted
   as extra noise.
3. Build a new outcome: fitted null model + lambda x estimate x tested regressor + resampled residuals.
4. Rerun the exact test and record whether it rejects at 5%, and the estimate.
5. Repeat 5,000 times for each of 21 values of lambda (0, 0.1, 0.2, ... 1.0, 1.2, 1.4, 1.7, 2, 2.5, 3, 4, 5,
   6 and 8). The same random draws are reused across lambda, so the curves are smooth. Power is the share of
   rejections; at lambda = 0 it is the size. The MDE is where the power curve first reaches 80%, by linear
   interpolation between grid points.

The explanatory variables (including yesterday's volatility) are kept at their real values; only today's
outcome is simulated.

### Two ways to resample the noise

- **Residual block bootstrap** (the design asked for): stretches of consecutive residuals with an average
  length of 10 trading days (4 weeks for R2), each starting at a random day. It keeps autocorrelation and
  volatility clustering, but it moves each piece of noise to a different day, so it loses any link between
  the size of the noise and that day's explanatory variables.
- **Wild block bootstrap**: each residual stays on its own day and gets a random sign, constant over blocks of
  10 days (4 weeks for R2). It keeps that link, but makes the noise symmetric.

The two disagree where the data say they should. In R5 the squared residual is correlated with the squared
tested regressor (0.20; in R3 the same correlation is 0.00 and -0.04), because big days have big
last-30-minute moves, and the wild scheme gives the larger R5 MDE. In R2 the Tuesday whole-day residuals are
strongly skewed (skewness 1.86), which the residual scheme keeps and the wild scheme removes, and the residual
scheme gives the larger MDE for the signed-rank test (source: `estimates.csv`, columns
`corr_sq_resid_sq_tested_regressor` and `resid_skew`). Rather than choose a scheme after seeing the results,
the headline numbers use the **less favourable** of the two for each test (larger MDE, lower power, more data
needed); both are always shown.

The automatic block-length rule of Politis and White (2004) suggests about 0.1 to 2.9 days for the R3 and R5
residuals and their squares and 0.2 to 9.7 weeks for the R2 series (`estimates.csv`, columns
`politis_white_block_*`); we use 10 days and 4 weeks and check 5 and 30 days and 2 and 8 weeks (Results,
section 7).

### Larger samples

To ask "what if we had 1.5, 2 or 3 times the data?", the real days are kept and extra days are appended,
drawn in blocks (same average length) from the real days, together with their explanatory variables. This
assumes that the future looks like the past. Two versions of "the past": all days, or only the Tuesday-expiry
period (from 1 September 2025), which is what the coming years will most resemble. For R2 the extra weeks come
from the same expiry period. For the forecast test the extra days are added to the forecast period (the model
is still fitted on 2023 to 2024). Beyond 3 times the sample, the needed sample is extrapolated with the usual
rule that the MDE falls with the square root of the sample size; the simulations support that rule for R3 and
R5 (MDE x square root of the multiplier stays between 3.03 and 3.11 for R3a, 6.90 and 6.98 for R3b and 0.0417
and 0.0433 for R5 with the residual scheme and all days; `mde.csv`, column `mde80_times_sqrt_m`). It is not
used for the forecast test, whose MDE does not shrink that way.

"Expiries needed" = multiplier x expiry days in the real sample. Extra years = (multiplier - 1) x the length
of the sample (3.81 years for R3 and R5; 61 Tuesday weeks in 1.17 years, 52.2 per year, for R2;
`estimates.csv`, columns `span_years` and `units_per_year`).

### Sign errors

The outcome is generated with the real Pro measure, but the test sees a version whose sign is wrong on a
share of days: the GEX share times -1 (R3, FC), or the indicator 1[GEX < 0] switched (R5). Three patterns:
random days; spells averaging 20 days (sign rules tend to disagree for weeks at a time); and the days with the
smallest |GEX| first, the most favourable case. The shares tried are 10%, 25% and 40%, plus five observed
disagreement rates (anchors) between sign rules, listed in section 6. Sign errors are simulated with the
residual scheme; for R5 that scheme is the more optimistic one.

### Settings and seeds

5,000 simulations per cell, 278 cells (tests x schemes x sample sizes x sign-error settings). Master seed
712026; each cell uses `numpy.random.default_rng([712026, cell hash, chunk])`, where the cell hash is the
first 8 hex digits of the SHA-256 of the cell's description and chunks hold 500 simulations; the seed of every
row is in the `seed_entropy` column of `power_curves.csv`. The R2 confidence intervals reuse the bootstrap of
`real_day_level.py` (seed 0). Chunks are added up in a fixed order, so a rerun gives identical files. Seeds,
settings, run time, package versions and SHA-256 hashes of the inputs are in `results/power_real/run_info.json`.

---

## Results (SIMULATIONS CALIBRATED ON REAL DATA)

### 1. Size: how often each test rejects when there is no effect

Real sample size. In brackets, the 95% Monte Carlo interval. The target is 5% (for the one-sided forecast test,
at most 5%).

| Test | Residual scheme | Wild scheme |
| --- | --- | --- |
| R2: Tuesday expiry premium, whole day (signed-rank test, as published) | 7.6% (6.9% to 8.4%) | 2.6% (2.2% to 3.1%) |
| R2: same, t-test of the mean | 6.9% (6.2% to 7.6%) | 6.0% (5.3% to 6.7%) |
| R2: Tuesday expiry premium, last hour | 6.5% (5.9% to 7.2%) | 5.0% (4.5% to 5.7%) |
| R2: Tuesday last hour, t-test of the mean | 7.5% (6.8% to 8.3%) | 7.5% (6.8% to 8.2%) |
| R2: Thursday single-weekly premium, whole day | 17.8% (16.8% to 18.9%) | 12.9% (12.0% to 13.8%) |
| R2: Thursday single-weekly, t-test of the mean | 16.3% (15.3% to 17.3%) | 12.8% (11.9% to 13.7%) |
| R2: several-weeklies premium, whole day | 9.1% (8.3% to 9.9%) | 8.2% (7.5% to 9.0%) |
| R2: several weeklies, t-test of the mean | 8.5% (7.7% to 9.3%) | 8.2% (7.4% to 9.0%) |
| R3: ordinary-day GEX term | 5.7% (5.1% to 6.3%) | 5.0% (4.4% to 5.7%) |
| R3: expiry-day GEX interaction | 6.4% (5.7% to 7.1%) | 5.7% (5.1% to 6.4%) |
| R5: reversal on negative-GEX days | 5.3% (4.8% to 6.0%) | 6.1% (5.5% to 6.8%) |
| Forecast gain from GEX | 2.7% (2.3% to 3.2%) | 1.8% (1.4% to 2.2%) |

Source: `results/power_real/size.csv` (rows with purpose "main" and m = 1; the file also has the size of every
larger-sample and sign-error cell).

- R3 and R5 are close to 5% (a slight excess, as in R3b's 6.4%, is typical of Newey-West tests). Across all
  larger samples they stay between 3.6% and 6.4%, and across all sign-error settings between 0.7% and 7.0%.
- The signed-rank test of the Tuesday whole-day premium is the exception that grows with the sample: with
  the residual scheme its size reaches 16.5% at 3 times the sample. The reason is the strong skew of the
  weekly premiums (skewness 1.86): a few very volatile expiry days pull the mean above the median, and the
  signed-rank test checks whether values are centred on zero, not whether their mean is zero. The t-test of
  the mean stays between 4.3% and 6.9% at all sample sizes. For the expiry premium, which is defined as a mean,
  the t-test is the more reliable of the two.
- The two Thursday periods over-reject with both schemes, because their weekly premiums are autocorrelated
  (lag-one correlation 0.26 for one weekly on Thursday, 0.09 for several weeklies; `estimates.csv`, column
  `resid_acf1`) and 41 or 98 weeks are too few for the tests to adjust.
- The forecast test is conservative (1.8% to 2.7%): with no true effect, the model with GEX carries two
  useless coefficients and forecasts slightly worse. In the real data the GEX model forecast 5.1% worse;
  10.7% (residual scheme) and 14.1% (wild scheme) of the simulations with no effect were at least that bad
  (`size.csv`, column `share_null_mse_change_at_least_real`).

### 2. Minimum detectable effect against the estimate (real sample size)

| Test | Estimate (95% interval) | p, real data | MDE80, residual scheme | MDE80, wild scheme | Textbook 2.8 x SE | Power at the estimate | Power at half the estimate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R2: Tuesday expiry premium, whole day (signed-rank test, as published) | 0.192 (0.079 to 0.319) | 0.005 | 0.184 (0.96x) | 0.151 (0.79x) | 0.174 | 84% | 20% |
| R2: same, t-test of the mean | 0.192 (0.079 to 0.319) | 0.005 | 0.146 (0.76x) | 0.153 (0.80x) | 0.148 | 95% | 41% |
| R2: Tuesday expiry premium, last hour | 0.356 (0.196 to 0.511) | 5.0e-5 | 0.227 (0.64x) | 0.231 (0.65x) | 0.225 | 99% | 57% |
| R2: Thursday single-weekly premium, whole day | -0.119 (-0.303 to 0.073) | 0.26 | 0.344 (2.90x) | 0.338 (2.85x) | 0.280 | 25% | 17% |
| R2: several-weeklies premium, whole day | 0.108 (0.010 to 0.201) | 0.045 | 0.157 (1.45x) | 0.152 (1.40x) | 0.137 | 52% | 18% |
| R3: ordinary-day GEX term | -3.16 (-5.29 to -1.03) | 0.004 | 3.03 (0.96x) | 3.01 (0.95x) | 3.04 | 83% | 31% |
| R3: expiry-day GEX interaction | 8.21 (3.22 to 13.20) | 0.001 | 6.90 (0.84x) | 7.12 (0.87x) | 7.13 | 90% | 37% |
| R5: reversal on negative-GEX days | -0.0682 (-0.1043 to -0.0321) | 2.2e-4 | 0.0419 (0.61x) | 0.0517 (0.76x) | 0.0516 | 97% | 45% |
| Forecast gain from GEX | 1 (by definition) | 0.98 | 1.46 (1.46x) | 1.49 (1.49x) | n/a | 50% | 14% |

Source: `results/power_real/headline.csv` and `mde.csv`. MDEs are sizes (the effect can have either sign);
the number in brackets is the MDE as a multiple of the estimate. "Power at the estimate" and "at half the
estimate" use the less favourable scheme. The R2 intervals are the bootstrap intervals of R2; the others are
the estimate plus or minus 1.96 Newey-West standard errors. The textbook column uses the real standard error
(the plain one for the signed-rank rows, the Newey-West one for the t-test rows). The Monte Carlo error of the
MDEs is at most 0.053 for R3b and much smaller for the other tests (`mde.csv`, column `mde80_mc_se`).

In percentages of variance (`headline.csv`, columns ending in `_as_pct_more_variance`): the Tuesday whole-day
premium is 21.1% (8.2% to 37.6%) and its MDE 15.7% to 20.2% depending on test and scheme; the last-hour
premium is 42.8% (21.6% to 66.7%) with an MDE of 25.5% to 26.0%; the one-weekly-Thursday estimate is -11.2%
(-26.1% to 7.6%) with an MDE of 40.2% to 41.0%; the several-weeklies estimate is 11.5% (1.0% to 22.2%) with an
MDE of 16.4% to 17.0%.

What the table says:

- For R3, R5 and the Tuesday premiums, the simulated MDE agrees with the textbook 2.8 standard errors (R5 with
  the wild scheme, see the Method), which is a check that the simulation behaves.
- For the four significant estimates of the project (Tuesday whole-day premium, the two R3 terms, R5) the MDE
  is 0.61 to 0.96 times the estimate (column `mde80_lambda`; R5 has the most room). An effect half as large
  would usually have gone unseen.
- The several-weeklies premium (p = 0.045) is smaller than its MDE: the test had about a coin-flip chance of
  finding it even if it is real, and its over-rejection (section 1) makes the p-value look better than it is.
- The forecast test is the weakest: at the in-sample effect size it is a coin flip.

### 3. How the MDE shrinks with more data

MDE80 with the residual scheme / wild scheme. Extra days drawn from all past days.

| Test | Real sample | 1.5x | 2x | 3x |
| --- | --- | --- | --- | --- |
| R2: Tuesday whole day, t-test of the mean | 0.146 / 0.153 | 0.120 / 0.127 | 0.105 / 0.112 | 0.087 / 0.092 |
| R2: Tuesday whole day, signed-rank test | 0.184 / 0.151 | 0.155 / 0.120 | 0.137 / 0.102 | 0.117 / 0.082 |
| R3: ordinary-day GEX term | 3.03 / 3.01 | 2.49 / 2.48 | 2.19 / 2.16 | 1.79 / 1.79 |
| R3: expiry-day GEX interaction | 6.90 / 7.12 | 5.70 / 5.89 | 4.94 / 5.06 | 3.99 / 4.17 |
| R5: reversal on negative-GEX days | 0.0419 / 0.0517 | 0.0340 / 0.0426 | 0.0303 / 0.0375 | 0.0250 / 0.0306 |
| Forecast gain from GEX (multiples of the R3 estimate) | 1.46 / 1.49 | 1.28 / 1.36 | 1.18 / 1.28 | 1.10 / 1.18 |

Source: `results/power_real/mde.csv` (purpose "main"; real sample: 61 weeks, 937, 929 and 937 days).

The R3 and R5 MDEs shrink roughly with the square root of the sample. The forecast test's MDE barely moves:
its two GEX coefficients are estimated once from 485 days, and adding forecast days only measures that
imprecise model more accurately.

### 4. Expiries needed for 80% power

Less favourable scheme, future like all past days. "The estimate" means an effect exactly as large as
estimated, "half the estimate" a true effect half as large, and "95% bound nearest zero" the smallest effect
inside the 95% interval.

| Test | Target effect | Value | Sample needed (x real) | Expiries needed in total | Extra expiries | Extra years | How |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R2: Tuesday whole day, signed-rank test | the estimate | 0.192 | 0.92 | 56 | 0 | 0.0 | current sample is enough (scaled down) (residual) |
| R2: Tuesday whole day, signed-rank test | half the estimate | 0.096 | 4.50 | 275 | 214 | 4.1 | extrapolated beyond 3x (1/sqrt(n) scaling) (residual) |
| R2: Tuesday whole day, signed-rank test | 95% bound nearest zero | 0.079 | 6.70 | 409 | 348 | 6.7 | extrapolated beyond 3x (1/sqrt(n) scaling) (residual) |
| R2: Tuesday whole day, t-test of the mean | the estimate | 0.192 | 0.64 | 39 | 0 | 0.0 | current sample is enough (scaled down) (wild) |
| R2: Tuesday whole day, t-test of the mean | half the estimate | 0.096 | 2.75 | 168 | 107 | 2.1 | simulated (interpolated) (wild) |
| R2: Tuesday whole day, t-test of the mean | 95% bound nearest zero | 0.079 | 4.11 | 251 | 190 | 3.6 | extrapolated beyond 3x (1/sqrt(n) scaling) (wild) |
| R2: Tuesday expiry premium, last hour | the estimate | 0.356 | 0.42 | 26 | 0 | 0.0 | current sample is enough (scaled down) (wild) |
| R2: Tuesday expiry premium, last hour | half the estimate | 0.178 | 1.65 | 101 | 40 | 0.8 | simulated (interpolated) (wild) |
| R2: Tuesday expiry premium, last hour | 95% bound nearest zero | 0.196 | 1.39 | 85 | 24 | 0.5 | simulated (interpolated) (wild) |
| R3: ordinary-day GEX term | the estimate | -3.16 | 0.92 | 184 | 0 | 0.0 | current sample is enough (scaled down) (residual) |
| R3: ordinary-day GEX term | half the estimate | -1.58 | 3.87 | 770 | 571 | 10.9 | extrapolated beyond 3x (1/sqrt(n) scaling) (residual) |
| R3: ordinary-day GEX term | 95% bound nearest zero | -1.03 | 9.11 | 1814 | 1615 | 30.9 | extrapolated beyond 3x (1/sqrt(n) scaling) (residual) |
| R3: expiry-day GEX interaction | the estimate | 8.21 | 0.75 | 150 | 0 | 0.0 | current sample is enough (scaled down) (wild) |
| R3: expiry-day GEX interaction | half the estimate | 4.10 | 3.09 | 616 | 417 | 8.0 | extrapolated beyond 3x (1/sqrt(n) scaling) (wild) |
| R3: expiry-day GEX interaction | 95% bound nearest zero | 3.22 | 5.03 | 1001 | 802 | 15.4 | extrapolated beyond 3x (1/sqrt(n) scaling) (wild) |
| R5: reversal on negative-GEX days | the estimate | -0.0682 | 0.57 | 113 | 0 | 0.0 | current sample is enough (scaled down) (wild) |
| R5: reversal on negative-GEX days | half the estimate | -0.0341 | 2.44 | 480 | 283 | 5.5 | simulated (interpolated) (wild) |
| R5: reversal on negative-GEX days | 95% bound nearest zero | -0.0321 | 2.77 | 546 | 349 | 6.8 | simulated (interpolated) (wild) |
| Forecast gain from GEX | the estimate | 1.00 | n/a | n/a | n/a | n/a | not reached within the simulated range (residual) |
| Forecast gain from GEX | half the estimate | 0.50 | n/a | n/a | n/a | n/a | not reached within the simulated range (residual) |

Source: `results/power_real/headline.csv` (columns ending in `_estimate`, `_half`, `_ci_bound`) and
`sample_needed.csv` (both schemes, both versions of the future, and the power at 1x, 1.5x, 2x and 3x). For R2
the expiries are Tuesday expiry weeks; for R3 and R5 they are the expiry days inside a sample of all trading
days, so "extra years" is the more natural unit. A multiplier below 1 means the current sample was already
more than enough for that effect size.

**If the future looks like the Tuesday-expiry period only** (`sample_needed.csv`, future "Tuesday regime
only", half the estimate), R5 needs 1.44 to 2.73 times the sample (1.7 to 6.6 extra years) and the R3
ordinary-day term 3.55 to 4.32 times (9.7 to 12.7 years), close to the numbers above, but the R3 expiry-day
interaction needs 3.97 to 5.44 times (11.3 to 16.9 years): recent Tuesday-period days carry less information
about that term than the average past day.

### 5. Significant estimates overstate small effects

If the true effect is half our estimate, the estimates that pass the 5% test are on average this many times
the true value: Tuesday whole-day premium 1.60 (signed-rank, residual scheme), 1.44 to 1.52 (other
combinations); Tuesday last hour 1.21 to 1.27; R3 ordinary-day term 1.76 to 1.78; R3 expiry-day interaction
1.58 to 1.63; R5 1.26 (residual) and 1.45 (wild). If the true effect equals our estimate, the ratios for
these Tuesday, R3 and R5 tests are 0.99 to 1.11. In the one-weekly-Thursday period they are 4.19 to 4.42 at half the estimate. Source:
`results/power_real/mde.csv`, columns `exaggeration_at_half_estimate` and `exaggeration_at_estimate`.

So the published R3 and R5 estimates are probably larger than the true effects, if those effects are real;
a replication on new data should expect smaller numbers, and sample sizes should be planned for them.

### 6. Sign errors in the GEX regressor

Observed disagreement between sign rules (the anchors):

| Pair of sign rules | Period | Days | Share of days with a different sign |
| --- | --- | --- | --- |
| US convention vs Pro | full sample | 942 | 55.1% |
| US convention vs Pro | 2023 to 2024 | 490 | 75.3% |
| US convention vs Pro | 2025 onward | 452 | 33.2% |
| US convention vs Client mirror | full sample | 942 | 14.9% |
| US convention vs Client mirror | 2023 to 2024 | 490 | 12.9% |
| US convention vs Client mirror | 2025 onward | 452 | 17.0% |
| Pro vs Client mirror | full sample | 942 | 47.0% |
| Pro vs Client mirror | 2023 to 2024 | 490 | 71.0% |
| Pro vs Client mirror | 2025 onward | 452 | 21.0% |
| estimated dealer sign vs US convention | 2023 to 2024 | 461 | 86.8% |
| estimated dealer sign vs US convention | 2025 onward | 452 | 80.3% |
| estimated dealer sign vs Pro | 2023 to 2024 | 461 | 30.4% |
| estimated dealer sign vs Pro | 2025 onward | 452 | 84.1% |
| estimated dealer sign vs Client mirror | 2023 to 2024 | 461 | 100.0% |
| estimated dealer sign vs Client mirror | 2025 onward | 452 | 94.7% |

Source: `results/power_real/sign_disagreement_anchors.csv` (the fixed-rule rows are computed from
`results/pipeline_nse/daily_gex_panel.csv`; the estimated-sign rows from
`results/dealer_sign/sign_agreement.csv`, participants family). The full-sample fixed-rule rates and the two
estimated-vs-Pro rates are used in the simulation.

Power at the estimated effect, counting only rejections with the correct sign (for the one-sided forecast
test, any rejection), by share of days with the wrong sign:

| Test | Pattern | 0% | 10% | 14.9% | 25% | 30.4% | 40% | 47.0% | 55.1% | 84.1% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R3 ordinary-day term | random days | 83% | 56% | 43% | 23% | 16% | 7% | 4% | 2% | 0% |
| R3 ordinary-day term | spells of about 20 days | 83% | 57% | 45% | 26% | 20% | 10% | 6% | 2% | 0% |
| R3 ordinary-day term | smallest abs(GEX) days first | 83% | 83% | 82% | 79% | 78% | 74% | 67% | 55% | 7% |
| R3 ordinary-day term | random days, 3x sample | 100% | 95% | 86% | 54% | 35% | 12% | 4% | 1% | 0% |
| R3 expiry-day interaction | random days | 91% | 64% | 50% | 27% | 17% | 8% | 4% | 2% | 0% |
| R3 expiry-day interaction | spells of about 20 days | 91% | 65% | 52% | 30% | 21% | 10% | 5% | 2% | 0% |
| R3 expiry-day interaction | smallest abs(GEX) days first | 91% | 91% | 90% | 89% | 87% | 82% | 74% | 62% | 4% |
| R3 expiry-day interaction | random days, 3x sample | 100% | 98% | 91% | 62% | 41% | 14% | 4% | 1% | 0% |
| R5 reversal interaction | random days | 99% | 93% | 86% | 59% | 40% | 14% | 6% | 1% | 0% |
| R5 reversal interaction | spells of about 20 days | 99% | 89% | 80% | 55% | 40% | 18% | 9% | 2% | 0% |
| R5 reversal interaction | smallest abs(GEX) days first | 99% | 95% | 91% | 66% | 50% | 13% | 2% | 0% | 0% |
| R5 reversal interaction | random days, 3x sample | 100% | 100% | 100% | 94% | 80% | 31% | 6% | 0% | 0% |
| Forecast gain (one-sided, any sign) | random days | 52% | 56% | 52% | 36% | 24% | 7% | 2% | 3% | 50% |
| Forecast gain (one-sided, any sign) | spells of about 20 days | 52% | 50% | 45% | 31% | 22% | 10% | 5% | 5% | 49% |
| Forecast gain (one-sided, any sign) | smallest abs(GEX) days first | 52% | 52% | 53% | 53% | 55% | 57% | 58% | 56% | 4% |
| Forecast gain (one-sided, any sign) | random days, 3x sample | 73% | 72% | 67% | 47% | 32% | 8% | 1% | 2% | 66% |

Source: `results/power_real/sign_error.csv` (columns `power_right_sign_at_estimate` and, for the forecast test,
`power_at_estimate`; real sample, or m = 3 for the "3x sample" rows). The 0% column of the "3x sample" rows,
which has no sign errors, comes from `mde.csv` (purpose "main", residual scheme, all days, m = 3, columns
`power_right_sign_at_estimate` and `power_at_estimate`).

What this means:

- With signs wrong on random days, the effect seen by the test shrinks roughly by the factor 1 - 2 x (share
  wrong) (column `attenuation_1_minus_2f` of `sign_error.csv`): 0.8 at 10%, 0.5 at 25%, 0.06 at 47%. Power
  falls accordingly, and more data does not rescue it: at 3 times the sample, power with 40% wrong signs is
  still 12% to 31% for the three regression tests.
- If the measured sign is wrong mostly on days when net GEX is close to zero, the share measure of R3 barely
  changes and R3 keeps most of its power. R5 does not, because it uses only the sign (an indicator), so a
  near-zero day with the wrong sign counts fully.
- Above 50% wrong signs the measure is closer to the reverse of the truth. The two-sided R3 and R5 tests then
  still reject often (R5 84% of the time at 84.1% wrong signs, `power_at_estimate`), but with the wrong sign.
  The forecast test does not care about the sign, because its coefficients are estimated freely.
- The fixed rules disagree on 47.0% to 55.1% of days (Pro against the other two). If the true dealer sign
  were one of the other rules and we measured it with Pro, power at the estimated effect would be 1% to 6%
  for the correct sign. A significant Pro result therefore says little about which rule is right, and a null
  result under one rule says little about the others.
- The forecast test barely changes for 10% to 15% wrong signs (52% to 56%); it was weak to begin with.

### 7. Robustness of the MDE

| Test | Block length | Size | MDE80 | Power at the estimate |
| --- | --- | --- | --- | --- |
| R3: ordinary-day GEX term | 5 days / 10 (main) / 30 days | 5.8% / 5.7% / 6.8% | 3.04 / 3.03 / 3.05 | 83% / 83% / 83% |
| R3: expiry-day GEX interaction | 5 days / 10 (main) / 30 days | 6.2% / 6.4% / 6.3% | 6.79 / 6.90 / 6.87 | 92% / 91% / 91% |
| R5: reversal on negative-GEX days | 5 days / 10 (main) / 30 days | 5.9% / 5.3% / 5.5% | 0.0423 / 0.0419 / 0.0418 | 99% / 99% / 99% |
| Forecast gain from GEX | 5 days / 10 (main) / 30 days | 2.3% / 2.7% / 2.9% | 1.45 / 1.46 / 1.47 | 52% / 52% / 52% |
| R2: Tuesday whole day, signed-rank | 2 weeks / 4 (main) / 8 weeks | 8.8% / 7.6% / 4.6% | 0.186 / 0.184 / 0.182 | 83% / 84% / 85% |
| R2: Tuesday whole day, t-test of the mean | 2 weeks / 4 (main) / 8 weeks | 7.3% / 6.9% / 5.5% | 0.151 / 0.146 / 0.142 | 95% / 97% / 98% |

Source: `results/power_real/mde.csv` and `size.csv` (purpose "block length" and, for the main block length,
purpose "main" with m = 1, residual scheme). The block length hardly matters for the MDE.

### Figures

- `results/power_real/power_curves.png`: power against the true effect for six tests, at the real sample size
  and at 1.5, 2 and 3 times the sample (residual scheme), with the wild scheme at the real sample size as a
  dashed line, the real estimate and its 95% interval.
- `results/power_real/power_sign_error.png`: power at the estimated effect against the share of days with the
  wrong sign, with the observed disagreement rates marked.

---

## What this means for the project

- **R5 (reversal stronger after negative Pro GEX)** is the best-powered GEX result: 97% power at its estimate
  and an MDE well below it. But if the true effect is half the estimate, the current sample finds it only 45%
  of the time, and its power depends on the Pro sign being right on most days.
- **R3, expiry-day flip** is right at the edge: its MDE is 0.87 times the estimate with the wild scheme
  (`mde.csv`, `mde80_lambda`). Its failure to replicate on Bank Nifty ([FALSIFICATION.md](FALSIFICATION.md))
  and its worse out-of-sample forecasts are what one expects if the true effect is real but smaller than
  estimated, and also what one expects if it is not real. Telling these apart at half the estimated size
  takes about 8 more years of data.
- **R3, ordinary-day term** has the same power, but the falsification tests already showed that it behaves like
  a slow-moving level of total gamma. Power cannot fix that; it only says the test was able to see an effect
  of that size.
- **The forecast test** cannot be used as evidence against GEX: even an effect exactly as large as the
  in-sample estimate would be found only half the time, and the observed 5.1% loss is typical of no effect.
  A design that refits as data arrive would be more powerful; it was not simulated here.
- **Expiry premium (R2)**: the Tuesday last-hour premium is large enough to be seen with plenty of room; the
  whole-day premium is just at the edge, and the t-test of the mean is the more reliable test. The
  one-weekly-Thursday "no effect" only rules out premiums of about 41% or more.

---

## Limits

1. **The future is assumed to look like the past.** Larger samples are built from resampled real days. A new
   regulatory change, a different volatility level or a shift in who holds options would change the answer.
   The Tuesday-only version shows how much: for the R3 expiry-day interaction the needed sample grows from 3.1
   to between 4.0 and 5.4 times.
2. **Regressors are held fixed.** Yesterday's volatility enters as its real value; the injected effect does not
   feed into later days.
3. **The effect is linear and constant over time.** Real effects may differ by regime or fade.
4. **Neither resampling scheme is perfect.** The residual scheme loses the link between noise and regressors;
   the wild scheme makes the noise symmetric. We report both and use the less favourable.
5. **Short series.** R2 has 41 to 98 weeks per period; the block bootstrap then copies the sample's own chance
   patterns, so the R2 size and MDE are themselves uncertain beyond the Monte Carlo error.
6. **Sign errors are stylized.** Only the sign is wrong; the size of GEX under different rules also differs, and
   real disagreements are neither random nor in 20-day spells exactly.
7. **One test at a time.** Each MDE is for a single 5% test. With a correction for the 18 headline tests or
   the 78 original tests (as in [FALSIFICATION.md](FALSIFICATION.md)), every MDE would be larger and every
   sample needed bigger.
8. **Extrapolation.** Needs beyond 3 times the sample use the square-root rule, which the simulations support
   up to 3 times only.
9. **Monte Carlo error.** At most 0.71 percentage points on any power; the MDE errors are in `mde.csv`.

---

## Files and how to rerun

    OMP_NUM_THREADS=1 python experiments/power_real.py            # full run, 5,000 simulations per cell
    OMP_NUM_THREADS=1 python experiments/power_real.py --reps 200 # quick check
    python experiments/power_real.py --plots-only                 # redraw the figures from the CSVs
    OMP_NUM_THREADS=1 python -m pytest -q tests/test_power_real.py

Inputs: `results/pipeline_nse/daily_market.csv`, `results/pipeline_nse/daily_gex_panel.csv`,
`data/raw/nifty_1min.csv` (for R5), and, for the reproduction check and anchors,
`results/real_day_level/r2_expiry_effect_by_regime.csv`, `r3_robustness.csv`,
`results/momentum_gex/regressions.csv`, `results/falsification/t4_oos_forecasts.csv` and
`results/dealer_sign/sign_agreement.csv`.

Outputs in `results/power_real/`:

| File | Contents |
| --- | --- |
| `estimates.csv` | real-data estimates, intervals, p-values, check against the published files, residual diagnostics |
| `headline.csv` | one row per key test: size, MDE under both schemes, power, exaggeration, sample needed (less favourable scheme) |
| `power_curves.csv` | every simulated point: test, scheme, sample size, sign-error setting, lambda, power, Monte Carlo error, seed |
| `mde.csv` | MDE80 and power summaries for every cell |
| `size.csv` | rejection rate at zero effect with 95% Monte Carlo intervals |
| `sample_needed.csv` | multiplier, expiries and years needed for 80% power, both schemes and both versions of the future |
| `sign_error.csv` | power under sign errors |
| `sign_disagreement_anchors.csv` | observed sign disagreement between rules |
| `power_curves.png`, `power_sign_error.png` | the figures |
| `run_info.json` | seeds, settings, run time, input hashes, package versions |

The printed log is `results/power_real_stdout.txt`. The unit tests (`tests/test_power_real.py`, synthetic data
only) check the Newey-West code against statsmodels, the block and sign-error generators, that the engine's
size is near 5% with no effect, that power rises with the effect and with the sample, and that sign errors
lower power.
