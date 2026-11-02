# Experiment report

**Every number in this report comes either from simulated (synthetic) markets or from arithmetic on values printed in the 2025 paper. None comes from real market data.** The simulator is described in `src/niftygex/simulate.py`; its assumptions (in particular the size and form of the dealer-gamma effect) are ours, so the synthetic results show what the methods can and cannot do, not what the Nifty market does.

Raw outputs are in `results/`. Each experiment can be rerun with the command shown.

| ID | Question | Data | Command |
| --- | --- | --- | --- |
| E1 | Do the estimators recover known parameters at realistic sample sizes? | synthetic | `python experiments/estimator_validation.py --reps 30` |
| E2 | Are minute-level Wilcoxon tests reliable at the paper's sample sizes? What works instead? | synthetic | `python experiments/inference_checks.py --reps 300` |
| E3 | How uncertain is Diebold-Yilmaz connectedness at the paper's sub-sample sizes? | synthetic | same script |
| E4 | Can the GEX forecasting test detect a real effect without false alarms? How much does the sign rule matter? | synthetic | `python experiments/power_study.py --reps 8 --days 500 --kappas 0 0.5 1.0 --noises 0.1 0.5` then `python experiments/summarize_power.py` |
| E5 | Are the paper's ARMA(1,1) estimates a sign of near-cancellation? | synthetic | `python experiments/paper_checks.py` |
| E6 | Are the paper's published numbers internally consistent? | published values | same script |
| E7 | Does the whole pipeline run end to end and what does its output look like? | synthetic | `python scripts/run_pipeline.py --source synthetic --days 500 --kappa 0.5 --seed 7` |

---

## E1. Estimator validation

**E1a, mcsGARCH.** Truth set to the paper's Nifty "before" estimates ($\alpha = 0.026$, $\beta = 0.971$, $\nu = 8.9$). 30 simulated markets per sample size. "Known" uses the true daily variance; "EGARCH" estimates it, as the paper does. File: `results/estimator_validation/e1a_summary.csv`.

| Days | Daily variance | Bias in $\alpha$ | RMSE $\alpha$ | Bias in $\beta$ | Mean persistence (true 0.997) | Diurnal shape correlation |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | known | +0.0003 | 0.0021 | +0.0001 | 0.9974 | 0.74 |
| 64 | EGARCH | +0.0049 | 0.0061 | -0.0029 | 0.9990 | 0.72 |
| 250 | known | +0.0002 | 0.0010 | -0.0002 | 0.9970 | 0.90 |
| 250 | EGARCH | +0.0008 | 0.0016 | -0.0001 | 0.9977 | 0.90 |
| 500 | known | +0.0002 | 0.0006 | -0.0002 | 0.9971 | 0.95 |
| 500 | EGARCH | +0.0005 | 0.0008 | -0.0002 | 0.9973 | 0.95 |

What it shows:
* The implementation recovers the intraday GARCH parameters closely once the daily variance is right.
* At the paper's sample size (64 days), estimating the daily variance with EGARCH biases persistence upward by about 0.002. Close to 1 that matters: the implied half-life of an intraday shock goes from about 230 to about 690 minutes. So the large drop in half-life implied by the paper's Table 2 (E6) needs standard errors before it is read as a finding.
* The minute-by-minute diurnal curve is noisy with 64 days (average absolute error 20 to 22% per minute). Smoothing it is worthwhile.

**E1b, implied-vol solver.** File: `results/estimator_validation/e1b_iv_solver.csv`. 50,000 options with 1 to 6 days to expiry solve in about 0.9 s. For every option whose time value is at least one NSE tick (Rs 0.05), the largest IV error is below $3 \times 10^{-12}$. Options with less time value than a tick carry no volatility information; for those the solver can return a value up to 0.024 away from the truth, which is why the GEX panel fills them from neighbouring strikes.

## E2. Before/after tests at the paper's sample sizes

Two scenarios, 300 simulated pairs of windows each (bootstrap: 150). File: `results/inference_checks/e2_size_power.csv`. A well-behaved test rejects about 5% of the time when the variance ratio is 1.00.

**Scenario A: days independent, 64 versus 59 days, all minutes** (isolates within-day dependence).

| Test | Volatility, ratio 1.00 (should be 5%) | ratio 1.10 | ratio 1.25 | Volume, ratio 1.00 |
| --- | --- | --- | --- | --- |
| Minute-level Mann-Whitney (paper's approach) | **69%** | 85% | 100% | **87%** |
| Day-level Mann-Whitney | 6% | 27% | 88% | 6% |
| Day-cluster bootstrap | 3% | 29% | 90% | not run |
| HAC regression on logs | 5% | 15% | 49% | 10% |

**Scenario B: persistent daily volatility (AR(1) 0.9), expiry days only, 13 versus 12 weeks** (like the paper's Thursday comparison). The variance ratio here is the change in the expiry-day effect.

| Test | ratio 1.00 (should be 5%) | ratio 1.10 | ratio 1.25 |
| --- | --- | --- | --- |
| Minute-level Mann-Whitney | **82%** | 83% | 88% |
| Day-level Mann-Whitney on expiry days | **32%** | 29% | 39% |
| Difference-in-differences, weeks, Mann-Whitney | 2% | 10% | 49% |
| Difference-in-differences, weeks, Welch t | 2% | 9% | 56% |

What it shows:
* Treating minutes as independent makes "significant" differences appear most of the time even when nothing changed. Monte Carlo standard errors are about 2 to 3 percentage points, so this is not noise.
* When daily volatility is persistent, comparing raw expiry-day levels across two windows is also unreliable, because the two windows simply have different average volatility. Comparing each expiry day with the rest of its own week fixes the false-alarm rate, at the cost of power: with 13 versus 12 weeks it detects a 25% rise in expiry-day variance only about half the time.
* This does not show that the paper's conclusions are wrong. Its p-values are mostly far below 0.001, and some of its effects (a 61% rise in Thursday volume) are large. It shows the reported p-values overstate the evidence and the tests should be redone at day or week level.

## E3. Connectedness uncertainty

**Sampling spread when the true total is near the paper's level.** A bivariate VAR(1) calibrated to a true DY total of 38.9 (77.7 on the paper's scale). 400 samples per size. File: `results/inference_checks/e3b_sampling_distribution.csv`.

| Observations | Comparable to | 95% range of the estimated total (DY scale) | Same on the paper's scale |
| --- | --- | --- | --- |
| 240 | one trading hour on four monthly expiry days | 33.1 to 42.3 | 66 to 85 |
| 1,440 | one hour on 24 days | 36.8 to 40.5 | 74 to 81 |
| 4,680 | full sessions on 13 expiry days | 37.8 to 39.8 | 76 to 80 |
| 23,000 | the paper's full window | 38.4 to 39.3 | 77 to 79 |

**Sensitivity to the unreported VAR lag and horizon.** On simulated log Rogers-Satchell series (lags 1, 2, 5, 10; horizons 5 to 100), the total moved by at most 2.6 points (DY scale) on the 240-observation sample and by less than 0.2 on the larger ones. File: `results/inference_checks/e3_dy_sensitivity.csv`. On that same 240-observation sample, a block bootstrap gave an interval of 9.7 to 21.5 around an estimate of 18.6.

**Can a cross share exceed 50%?** With no lagged dynamics the cross share is $\rho^2/(1+\rho^2)$, below 50% for any correlation (41.9% at $\rho = 0.85$). With $\rho = 0.85$ and a lead coefficient of 0.9 it reaches 50.9%. File: `results/inference_checks/e3_cross_share_example.csv`.

What it shows: under idealised Gaussian conditions, the paper's large differences (for example the last hour of monthly expiry, 105.22 before versus 66.44 and 74.43 after on the paper's scale) are bigger than sampling noise at 240 observations, while differences under about 10 points on the paper's scale in the hourly panels are not distinguishable. Heavier tails in real data would widen these ranges. Reporting bootstrap intervals would settle it.

## E4. Power study for the GEX forecasting test

**Setup.** 48 simulated 500-day markets: gamma effect strength $\kappa \in \{0, 0.5, 1.0\}$ crossed with participant-file noise $\in \{0.1, 0.5\}$, 8 markets per cell. In each, the walk-forward ladder ran (150 training days, then two-month test blocks, about 350 test days and 71 expiry test days), and each GEX feature set was tested against the same model without GEX (one-sided Diebold-Mariano on daily QLIKE, 5% level). In the simulator, dealers are on average net short both calls and puts, the stylised Indian case. $\kappa = 1$ means dealer gamma moves minute variance by roughly 20% around its average, more on expiry days. File: `results/power_study_summary.csv`.

Rejection rate of "GEX adds nothing", next-30-minute target (all test days / expiry days only):

| Effect $\kappa$ | Participant noise | US convention | Participant (Pro) sign | Oracle (true positions) |
| --- | --- | --- | --- | --- |
| 0 (no effect) | 0.1 | 0% / 0% | 12% / 12% | 0% / 12% |
| 0 (no effect) | 0.5 | 0% / 0% | 0% / 0% | 12% / 0% |
| 0.5 | 0.1 | 0% / 0% | 75% / 75% | 75% / 100% |
| 0.5 | 0.5 | 0% / 0% | 0% / 0% | 50% / 50% |
| 1.0 | 0.1 | 0% / 0% | 88% / 62% | 88% / 88% |
| 1.0 | 0.5 | 0% / 0% | 25% / 0% | 100% / 75% |

Median out-of-sample QLIKE improvement from adding GEX, next-30-minute target, expiry days: participant sign 3.3% ($\kappa$ 0.5) and 2.9% ($\kappa$ 1.0) with good participant data; oracle 2.7% and 4.3%. For the last-hour target (one observation per day), every rejection rate was 25% or lower, including the oracle.

Other numbers from the same runs: the hybrid model (LightGBM correcting mcsGARCH) beat mcsGARCH in median by 2 to 3% QLIKE in every cell on the 30-minute target; HAR beat mcsGARCH by 1 to 12% on the last-hour target but was within 3% of it, mostly worse, on the 30-minute target.

What it shows (within this simulator):
1. **No false alarms of note.** With no effect, no sign rule rejected in more than 1 of the 8 markets of any cell. With 8 markets per cell the Monte Carlo error on a rate is about 10 to 15 points, so this establishes rough, not precise, size control. An earlier version of the simulator did produce false alarms, which led to a fix (ITERATION_LOG.md, iteration 8).
2. **The sign rule is decisive.** The US convention never detected the effect, even when it was strong, because in this world its call sign is wrong. A wrong sign rule produces a false null, not just a weaker test.
3. **Measurement quality of the sign is the binding constraint.** With accurate participant data the test detects the effect in 62 to 88% of markets; with noisy data, in 0 to 25%. The oracle shows the effect was there to find.
4. **Gains are small even when the effect is real** (1 to 4% QLIKE). Positioning is persistent, so much of its effect is already in recent realized volatility. Real-data expectations should be set accordingly.
5. **The last-hour target is underpowered** with about 70 expiry days of end-of-day GEX. Intraday GEX or a longer sample is needed for the last-hour claim.

## E5. ARMA(1,1) near-cancellation

150 simulated series of 23,039 minutes (the paper's "before" size) with GARCH-t errors, fitted with ARMA(1,1). File: `results/paper_checks/e5_summary.csv`.

| True first autocorrelation | Share with $|\hat\phi| > 0.5$ | Median $|\hat\phi|$ | Correlation of $\hat\phi$ and $\hat\theta$ | Median $|\hat\phi + \hat\theta|$ |
| --- | --- | --- | --- | --- |
| 0 | 51% | 0.50 | -0.9999 | 0.006 |
| -0.02 | 16% | 0.26 | -0.9998 | 0.020 |

What it shows: on data with essentially no autocorrelation, ARMA(1,1) estimates scatter widely but always nearly cancel. The paper's four pairs ($\phi + \theta$ between -0.04 and +0.02, $|\phi|$ up to 0.75) fit this pattern. We did not check whether the conventional z statistics would also look large in this situation, so we only say the AR and MA terms should not be read as evidence of minute-level predictability.

## E6. Arithmetic on the published tables

File: `results/paper_checks/e6_table_arithmetic.csv`. Inputs are numbers printed in the paper.

| Check | Result |
| --- | --- |
| Connectedness scale | Totals equal the sum of off-diagonal shares: 84.00 (paper) is 42.00 on the standard DY scale; 105.22 is 52.61 |
| Table 2 unit-mean restriction | $\omega/(1-\alpha-\beta)$ between 1.08 and 1.14 in all four columns: consistent with a normalised intraday component |
| Table 2 half-lives | Nifty 339 to 72 minutes, Bank Nifty 185 to 40 minutes (before to after); see the E1 caution |
| Table 2 ARMA | $\phi+\theta$ between -0.04 and +0.02; implied autocorrelation below 0.04 in absolute value |
| Table 2 stars | Nifty 50 before: AR and MA marked ** with z of -0.26 and 0.37 |
| Tables 5 and 6, Mann-Whitney statistic | 9 of 12 within the possible range; 3 in Table 6 exceed the maximum possible (for example 207,455,847.5 against a maximum near 31.2 million), consistent with misplaced digits |
| Minutes per day | 23,039 and 21,239 observations are 64.0 and 59.0 days of 360 minutes; weekday samples 4,680 and 4,320 are exactly 13 and 12 days of 360 minutes |

## E7. Full pipeline on one synthetic market

See the section appended below by the pipeline run (results in `results/pipeline_synthetic/`).
